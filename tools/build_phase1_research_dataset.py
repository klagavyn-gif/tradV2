import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


WATCHLIST = [
    "BTC-USD",
    "DOGE-USD",
    "ETH-USD",
    "ADA-USD",
    "XRP-USD",
    "BNB-USD",
    "SOL-USD",
    "TRX-USD",
    "NEAR-USD",
    "LINK-USD",
    "PAXG-USD",
]

PERIOD_MAP = {
    "5d": pd.Timedelta(days=5),
    "30d": pd.Timedelta(days=30),
    "60d": pd.Timedelta(days=60),
    "90d": pd.Timedelta(days=90),
    "1mo": pd.Timedelta(days=30),
    "3mo": pd.Timedelta(days=90),
    "6mo": pd.Timedelta(days=180),
    "1y": pd.Timedelta(days=365),
    "2y": pd.Timedelta(days=730),
    "5y": pd.Timedelta(days=365 * 5),
}

_WORKER_ROOT = None
_WORKER_CACHE = None
_WORKER_WATCHLIST = None
_WORKER_GROUPS = None
_WORKER_MAX_HOLD_BARS = 64
_WORKER_ENTRY_FILL_TOLERANCE_PCT = 0.15
_WORKER_RESEARCH_SUPPLEMENTS = None
_WORKER_RESEARCH_PA_MIN_CONFIDENCE = 56.0
_WORKER_RESEARCH_PA_MIN_SCORE = 58.0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Phase 1 research replay: export candidate-level dataset from historical cached market data"
    )
    parser.add_argument("--days", type=int, default=365, help="Replay window in days")
    parser.add_argument("--step", default="4h", help="Checkpoint spacing, e.g. 1h, 4h, 1d")
    parser.add_argument("--watchlist", default=",".join(WATCHLIST), help="Comma-separated symbols")
    parser.add_argument("--end-at", default="", help="Optional replay end timestamp")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for JSONL/CSV summary outputs (default: .data/research/phase1)",
    )
    parser.add_argument(
        "--groups",
        default="primary,trend_radar,daily",
        help="Candidate groups to export: primary,trend_radar,trend_state,daily",
    )
    parser.add_argument(
        "--max-hold-bars",
        type=int,
        default=64,
        help="Bars used to label future outcome after each checkpoint",
    )
    parser.add_argument(
        "--entry-fill-tolerance-pct",
        type=float,
        default=0.15,
        help="Treat current price as filled when it is within this percent of entry",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refresh local market cache before replay using the project's live fetcher",
    )
    parser.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help="Allow replay to continue when cache history is shorter than requested days",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Checkpoint workers to use. 0 = auto (up to 4 workers), 1 = sequential",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print and persist progress every N completed checkpoints",
    )
    parser.add_argument(
        "--research-strategy-supplements",
        default="PA15",
        help="Optional comma-separated research-only strategy supplements to add to Phase 1 dataset without changing live alert behavior",
    )
    parser.add_argument(
        "--research-pa-min-confidence",
        type=float,
        default=56.0,
        help="Minimum confidence for research-only PA15 candidates extracted before live dispatch gates",
    )
    parser.add_argument(
        "--research-pa-min-score",
        type=float,
        default=58.0,
        help="Minimum score for research-only PA15 candidates extracted before live dispatch gates",
    )
    return parser


def parse_watchlist(text):
    values = [part.strip() for part in str(text or "").split(",")]
    return [part for part in values if part] or list(WATCHLIST)


def parse_groups(text):
    raw = [part.strip().lower() for part in str(text or "").split(",")]
    groups = []
    for value in raw:
        if value in {"primary", "trend_radar", "trend_state", "daily"} and value not in groups:
            groups.append(value)
    return groups or ["primary", "trend_radar", "daily"]


def parse_csv_upper(text):
    values = [part.strip().upper() for part in str(text or "").split(",")]
    return [part for part in values if part]


def resolve_output_dir(root, output_dir):
    raw = str(output_dir or "").strip()
    if not raw:
        return root / ".data" / "research" / "phase1"
    path = Path(raw)
    return path if path.is_absolute() else (root / path)


def _load_trad_module(root):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import trad

    return trad


def load_cache(root, watchlist):
    trad = _load_trad_module(root)
    cache = {}
    for symbol in watchlist:
        for interval in ("15m", "1h"):
            cache_path = Path(trad.get_market_history_store_file_path(symbol, interval=interval, auto_adjust=True))
            if not cache_path.exists():
                raise FileNotFoundError(f"Missing cache for {symbol} {interval}: {cache_path}")
            df = pd.read_csv(cache_path)
            df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
            df = df.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
            cache[(symbol, interval)] = df
    return cache


def _period_for_days(days):
    days = max(1, int(days or 1))
    if days <= 5:
        return "5d"
    if days <= 30:
        return "1mo"
    if days <= 60:
        return "60d"
    if days <= 90:
        return "3mo"
    if days <= 180:
        return "6mo"
    if days <= 365:
        return "1y"
    if days <= 730:
        return "2y"
    if days <= 365 * 5:
        return "5y"
    return "max"


def refresh_cache(root, watchlist, days):
    trad = _load_trad_module(root)
    provider = str(trad.get_market_data_provider())
    intraday_days = int(days or 1) if provider == "binance" else min(int(days or 1), 60)

    periods = {
        "15m": _period_for_days(intraday_days),
        "1h": _period_for_days(days),
    }
    refreshed = []
    for symbol in watchlist:
        for interval, period in periods.items():
            df = trad.get_yf_history(symbol, period=period, interval=interval, auto_adjust=True, cache_ttl_seconds=0)
            refreshed.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "period": period,
                    "provider": provider,
                    "rows": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
                }
            )
    return refreshed


def slice_df(df, period, now):
    out = df[df.index <= now]
    if out.empty:
        return out
    delta = PERIOD_MAP.get(str(period).lower())
    if delta is not None:
        out = out[out.index >= (now - delta)]
    return out.copy()


def build_cache_coverage(cache):
    rows = []
    for (symbol, interval), df in sorted(cache.items()):
        if not isinstance(df, pd.DataFrame) or df.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "start": None,
                    "end": None,
                    "coverage_days": 0,
                    "rows": 0,
                }
            )
            continue
        start = pd.Timestamp(df.index.min())
        end = pd.Timestamp(df.index.max())
        rows.append(
            {
                "symbol": symbol,
                "interval": interval,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "coverage_days": float((end - start) / pd.Timedelta(days=1)),
                "rows": int(len(df)),
            }
        )
    return rows


def summarize_cache_coverage(cache, *, days, end_at=None):
    coverage_rows = build_cache_coverage(cache)
    available_frames = [df for df in cache.values() if isinstance(df, pd.DataFrame) and not df.empty]
    if not available_frames:
        raise RuntimeError("No cache history available for Phase 1 replay")
    latest_now = min(pd.Timestamp(df.index.max()) for df in available_frames)
    if end_at:
        latest_now = min(pd.Timestamp(end_at), pd.Timestamp(latest_now))
    requested_start = latest_now - pd.Timedelta(days=days)
    common_start = max(pd.Timestamp(df.index.min()) for df in available_frames)
    effective_start = max(requested_start, common_start)
    available_days = max(0.0, float((latest_now - common_start) / pd.Timedelta(days=1)))
    requested_days = max(0.0, float((latest_now - requested_start) / pd.Timedelta(days=1)))
    return {
        "requested_days": float(requested_days),
        "available_days": float(available_days),
        "requested_start": requested_start.isoformat(),
        "effective_start": effective_start.isoformat(),
        "latest_end": latest_now.isoformat(),
        "has_full_coverage": common_start <= requested_start,
        "rows": coverage_rows,
    }


def compute_points(cache, days, step, end_at=None, *, allow_partial_coverage=False):
    coverage = summarize_cache_coverage(cache, days=days, end_at=end_at)
    latest_now = pd.Timestamp(coverage["latest_end"])
    if allow_partial_coverage:
        start_now = pd.Timestamp(coverage["effective_start"])
    else:
        start_now = pd.Timestamp(coverage["requested_start"])
    return list(pd.date_range(start=start_now, end=latest_now, freq=step))


def resolve_worker_count(requested, total_points):
    total = max(1, int(total_points or 1))
    raw = int(requested or 0)
    if raw <= 0:
        raw = min(4, os.cpu_count() or 1)
    return max(1, min(raw, total))


def _format_duration(seconds):
    if seconds is None:
        return "unknown"
    remaining = max(0, int(round(float(seconds))))
    hours, remaining = divmod(remaining, 3600)
    minutes, secs = divmod(remaining, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def write_progress(output_dir, payload):
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "phase1_progress.json"
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def emit_progress(*, output_dir, completed, total, started_at, candidate_total, workers, checkpoint_at, last_candidate_count):
    elapsed = max(0.0, time.perf_counter() - started_at)
    rate = (float(completed) / elapsed) if elapsed > 0 else 0.0
    eta = ((float(total - completed) / rate) if rate > 0 else None)
    checkpoint_text = pd.Timestamp(checkpoint_at).isoformat() if checkpoint_at is not None else None
    payload = {
        "status": "running" if completed < total else "completed",
        "completed_checkpoints": int(completed),
        "total_checkpoints": int(total),
        "progress_pct": (float(completed) / float(total) * 100.0) if total else 0.0,
        "elapsed_seconds": float(elapsed),
        "eta_seconds": float(eta) if eta is not None else None,
        "workers": int(workers),
        "candidates_collected": int(candidate_total),
        "last_checkpoint_at": checkpoint_text,
        "last_checkpoint_candidates": int(last_candidate_count),
        "updated_at": pd.Timestamp.now().isoformat(),
    }
    write_progress(output_dir, payload)
    message = (
        f"[phase1] {completed}/{total} checkpoints "
        f"({payload['progress_pct']:.1f}%) | candidates={candidate_total} | "
        f"elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)} | "
        f"workers={workers} | last_checkpoint={checkpoint_text} | last_candidates={last_candidate_count}"
    )
    print(message, flush=True)


def _init_worker(
    root,
    cache,
    watchlist,
    groups,
    max_hold_bars,
    entry_fill_tolerance_pct,
    research_supplements,
    research_pa_min_confidence,
    research_pa_min_score,
):
    global _WORKER_ROOT, _WORKER_CACHE, _WORKER_WATCHLIST, _WORKER_GROUPS
    global _WORKER_MAX_HOLD_BARS, _WORKER_ENTRY_FILL_TOLERANCE_PCT
    global _WORKER_RESEARCH_SUPPLEMENTS, _WORKER_RESEARCH_PA_MIN_CONFIDENCE, _WORKER_RESEARCH_PA_MIN_SCORE
    _WORKER_ROOT = root
    _WORKER_CACHE = cache
    _WORKER_WATCHLIST = list(watchlist or [])
    _WORKER_GROUPS = list(groups or [])
    _WORKER_MAX_HOLD_BARS = int(max_hold_bars or 64)
    _WORKER_ENTRY_FILL_TOLERANCE_PCT = float(entry_fill_tolerance_pct or 0.15)
    _WORKER_RESEARCH_SUPPLEMENTS = list(research_supplements or [])
    _WORKER_RESEARCH_PA_MIN_CONFIDENCE = float(research_pa_min_confidence or 56.0)
    _WORKER_RESEARCH_PA_MIN_SCORE = float(research_pa_min_score or 58.0)


def _run_checkpoint_worker(index, checkpoint_at):
    rows, checkpoint_summary = run_checkpoint(
        root=Path(_WORKER_ROOT),
        cache=_WORKER_CACHE,
        checkpoint_at=checkpoint_at,
        watchlist=_WORKER_WATCHLIST,
        groups=_WORKER_GROUPS,
        max_hold_bars=_WORKER_MAX_HOLD_BARS,
        entry_fill_tolerance_pct=_WORKER_ENTRY_FILL_TOLERANCE_PCT,
        research_supplements=_WORKER_RESEARCH_SUPPLEMENTS,
        research_pa_min_confidence=_WORKER_RESEARCH_PA_MIN_CONFIDENCE,
        research_pa_min_score=_WORKER_RESEARCH_PA_MIN_SCORE,
    )
    return index, rows, checkpoint_summary


def format_coverage_error(coverage):
    rows = coverage.get("rows") or []
    worst = sorted(rows, key=lambda row: (row.get("coverage_days") or 0.0, str(row.get("symbol") or ""), str(row.get("interval") or "")))
    sample = worst[:6]
    sample_text = "; ".join(
        f"{row.get('symbol')} {row.get('interval')}={row.get('coverage_days'):.1f}d"
        for row in sample
    )
    return (
        "Phase 1 cache coverage is shorter than requested replay window. "
        f"requested_days={coverage.get('requested_days'):.1f}, "
        f"available_days={coverage.get('available_days'):.1f}, "
        f"requested_start={coverage.get('requested_start')}, "
        f"effective_start={coverage.get('effective_start')}. "
        f"Shortest cache samples: {sample_text}. "
        "Use --refresh-cache to refill local cache, lower --days, or pass --allow-partial-coverage if you intentionally want a shorter replay."
    )


def _safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _normalize_signal(value):
    text = str(value or "").strip().upper()
    return text if text in {"BUY", "SELL"} else "BUY"


def _future_price_series(df, column_name):
    if not isinstance(df, pd.DataFrame):
        return None
    for key in (column_name, column_name.lower(), column_name.upper(), column_name.title()):
        if key in df.columns:
            return df[key]
    return None


def _candidate_anchor_price(candidate, snapshot):
    item = candidate.get("item") if isinstance(candidate, dict) else None
    plan = candidate.get("plan") if isinstance(candidate, dict) else None
    for value in (
        snapshot.get("entry_price"),
        snapshot.get("current_price"),
        (item or {}).get("price") if isinstance(item, dict) else None,
        (plan or {}).get("current_price") if isinstance(plan, dict) else None,
        (plan or {}).get("price") if isinstance(plan, dict) else None,
    ):
        parsed = _safe_float(value, None)
        if parsed is not None:
            return parsed
    return None


def _research_entry_intent(*, trad, candidate):
    plan = candidate.get("plan") if isinstance(candidate, dict) else None
    item = candidate.get("item") if isinstance(candidate, dict) else None
    entry_price = _safe_float((plan or {}).get("entry_price"), None)
    current_price = _safe_float((plan or {}).get("current_price"), None)
    if current_price is None:
        current_price = _safe_float((plan or {}).get("price"), None)
    if current_price is None and isinstance(item, dict):
        current_price = _safe_float(item.get("price"), None)
    stop_loss = _safe_float((plan or {}).get("stop_loss"), None)
    if entry_price is None or current_price is None:
        return "watch", "research_unknown_entry_distance"
    distance_pct = abs(float(current_price) - float(entry_price)) / max(abs(float(entry_price)), 1e-9) * 100.0
    distance_r = None
    if isinstance(stop_loss, (int, float)) and float(stop_loss) != float(entry_price):
        risk_dist = abs(float(entry_price) - float(stop_loss))
        if risk_dist > 0:
            distance_r = abs(float(current_price) - float(entry_price)) / float(risk_dist)
    max_distance_pct = _safe_float(getattr(trad.config, "TELEGRAM_ALERT_ENTRY_MAX_DISTANCE_PCT", 2.5), 2.5)
    max_distance_r = _safe_float(getattr(trad.config, "TELEGRAM_ALERT_ENTRY_MAX_DISTANCE_R", 1.1), 1.1)
    if distance_pct <= float(max_distance_pct) or (distance_r is not None and distance_r <= float(max_distance_r)):
        return "entry", f"research_fresh_entry:d_pct={distance_pct:.4f},d_r={distance_r:.4f}" if distance_r is not None else f"research_fresh_entry:d_pct={distance_pct:.4f}"
    return "watch", f"research_stretched_entry:d_pct={distance_pct:.4f},d_r={distance_r:.4f}" if distance_r is not None else f"research_stretched_entry:d_pct={distance_pct:.4f}"


def build_research_supplement_candidates(*, trad, results, existing_candidates, supplement_strategies, pa_min_confidence, pa_min_score):
    supplements = []
    enabled = {str(value or "").strip().upper() for value in (supplement_strategies or []) if str(value or "").strip()}
    if "PA15" not in enabled:
        return supplements
    existing_keys = {
        (
            str(row.get("strategy") or "").strip().upper(),
            str(row.get("symbol") or "").strip().upper(),
            str(row.get("signal") or "").strip().upper(),
            str(row.get("cache_key") or "").strip(),
        )
        for row in (existing_candidates or [])
        if isinstance(row, dict)
    }
    for item in results or []:
        if not isinstance(item, dict) or item.get("error"):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        plan = item.get("price_action_15m")
        if not symbol or not isinstance(plan, dict):
            continue
        signal = str(plan.get("research_signal") or "").strip().upper()
        confidence = _safe_float(plan.get("research_confidence"), None)
        score = _safe_float(plan.get("research_score"), None)
        entry_price = _safe_float(plan.get("entry_price"), None)
        stop_loss = _safe_float(plan.get("stop_loss"), None)
        take_profit = _safe_float(plan.get("take_profit"), None)
        if signal not in {"BUY", "SELL"}:
            continue
        if confidence is None or confidence < float(pa_min_confidence):
            continue
        if score is None or score < float(pa_min_score):
            continue
        if entry_price is None or stop_loss is None or take_profit is None:
            continue
        plan_copy = dict(plan)
        plan_copy["signal"] = signal
        plan_copy["alert"] = True
        plan_copy["confidence"] = float(confidence)
        plan_copy["score"] = float(score)
        if not plan_copy.get("detected_pattern") and plan_copy.get("research_detected_pattern"):
            plan_copy["detected_pattern"] = plan_copy.get("research_detected_pattern")
        cache_key = "PA15RESEARCH|{symbol}|{signal}|{context}".format(
            symbol=symbol,
            signal=signal,
            context=str(plan.get("last_signal_time") or plan.get("research_detected_pattern") or plan.get("market_structure") or "na"),
        )
        dedupe_key = ("PA15", symbol, signal, cache_key)
        if dedupe_key in existing_keys:
            continue
        candidate = {
            "symbol": symbol,
            "strategy": "PA15",
            "signal": signal,
            "score": float(score),
            "confidence": float(confidence),
            "plan": plan_copy,
            "item": item,
            "source_count": int(plan.get("proxy_source_count") or 0),
            "message": None,
            "cache_key": cache_key,
            "research_candidate": True,
            "research_source": "phase1_relaxed_pa15",
        }
        intent, intent_reason = _research_entry_intent(trad=trad, candidate=candidate)
        candidate["alert_intent"] = intent
        candidate["alert_intent_reason"] = intent_reason
        supplements.append(candidate)
        existing_keys.add(dedupe_key)
    return supplements


def simulate_candidate_outcome(
    *,
    checkpoint_at,
    history_df,
    signal,
    entry_price,
    stop_loss,
    take_profit,
    current_price,
    max_hold_bars,
    entry_fill_tolerance_pct,
):
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return {
            "label_status": "no_history",
            "label_filled": False,
            "label_win": None,
        }
    entry = _safe_float(entry_price, None)
    stop = _safe_float(stop_loss, None)
    target = _safe_float(take_profit, None)
    if entry is None:
        return {
            "label_status": "missing_entry",
            "label_filled": False,
            "label_win": None,
        }

    bars = history_df[history_df.index > checkpoint_at].head(max(1, int(max_hold_bars)))
    if bars.empty:
        return {
            "label_status": "no_future_bars",
            "label_filled": False,
            "label_win": None,
        }

    signal_text = _normalize_signal(signal)
    high_series = _future_price_series(bars, "High")
    low_series = _future_price_series(bars, "Low")
    close_series = _future_price_series(bars, "Close")
    open_series = _future_price_series(bars, "Open")
    if high_series is None or low_series is None or close_series is None:
        return {
            "label_status": "missing_ohlc",
            "label_filled": False,
            "label_win": None,
        }

    side = 1.0 if signal_text == "BUY" else -1.0
    filled = False
    fill_bar = None
    fill_time = None
    exit_bar = None
    exit_time = None
    exit_price = None
    status = "timeout"
    current = _safe_float(current_price, None)
    tolerance_ratio = max(0.0, float(entry_fill_tolerance_pct or 0.0)) / 100.0
    if current is not None and abs(entry) > 0 and abs(current - entry) / abs(entry) <= tolerance_ratio:
        filled = True
        fill_bar = 0
        fill_time = pd.Timestamp(checkpoint_at)

    post_fill_highs = []
    post_fill_lows = []
    for idx, (bar_time, row) in enumerate(bars.iterrows(), start=1):
        high = _safe_float(high_series.loc[bar_time], None)
        low = _safe_float(low_series.loc[bar_time], None)
        close = _safe_float(close_series.loc[bar_time], None)
        open_price = _safe_float(open_series.loc[bar_time], close)
        if high is None or low is None or close is None:
            continue
        if not filled and low <= entry <= high:
            filled = True
            fill_bar = idx
            fill_time = pd.Timestamp(bar_time)

        if not filled:
            continue

        post_fill_highs.append(high)
        post_fill_lows.append(low)
        if signal_text == "BUY":
            if stop is not None and low <= stop:
                status = "stop_hit"
                exit_price = stop
            elif target is not None and high >= target:
                status = "tp_hit"
                exit_price = target
        else:
            if stop is not None and high >= stop:
                status = "stop_hit"
                exit_price = stop
            elif target is not None and low <= target:
                status = "tp_hit"
                exit_price = target

        if status in {"stop_hit", "tp_hit"}:
            exit_bar = idx
            exit_time = pd.Timestamp(bar_time)
            break

        exit_bar = idx
        exit_time = pd.Timestamp(bar_time)
        exit_price = close if close is not None else open_price

    if not filled:
        return {
            "label_status": "no_fill",
            "label_filled": False,
            "label_win": None,
            "label_fill_bar": None,
            "label_exit_bar": None,
            "label_fill_timestamp": None,
            "label_exit_timestamp": None,
            "label_return_pct": None,
            "label_mfe_pct": None,
            "label_mae_pct": None,
            "label_mfe_r": None,
            "label_mae_r": None,
        }

    if exit_price is None:
        status = "timeout"
        last_close = _safe_float(close_series.iloc[-1], entry)
        exit_price = last_close
        exit_bar = len(bars)
        exit_time = pd.Timestamp(bars.index[-1])

    return_pct = ((float(exit_price) - float(entry)) / abs(float(entry))) * 100.0 * side
    mfe_pct = None
    mae_pct = None
    mfe_r = None
    mae_r = None
    risk_distance = abs(float(entry) - float(stop)) if stop is not None else None
    if post_fill_highs and post_fill_lows:
        if signal_text == "BUY":
            favorable_move = max(post_fill_highs) - float(entry)
            adverse_move = min(post_fill_lows) - float(entry)
        else:
            favorable_move = float(entry) - min(post_fill_lows)
            adverse_move = float(entry) - max(post_fill_highs)
        mfe_pct = (favorable_move / abs(float(entry))) * 100.0
        mae_pct = (adverse_move / abs(float(entry))) * 100.0
        if isinstance(risk_distance, (int, float)) and risk_distance > 0:
            mfe_r = favorable_move / float(risk_distance)
            mae_r = adverse_move / float(risk_distance)

    return {
        "label_status": status,
        "label_filled": True,
        "label_win": True if status == "tp_hit" else False if status == "stop_hit" else (return_pct > 0.0),
        "label_fill_bar": int(fill_bar) if fill_bar is not None else None,
        "label_exit_bar": int(exit_bar) if exit_bar is not None else None,
        "label_fill_timestamp": fill_time.isoformat() if fill_time is not None else None,
        "label_exit_timestamp": exit_time.isoformat() if exit_time is not None else None,
        "label_return_pct": float(return_pct),
        "label_mfe_pct": float(mfe_pct) if isinstance(mfe_pct, (int, float)) else None,
        "label_mae_pct": float(mae_pct) if isinstance(mae_pct, (int, float)) else None,
        "label_mfe_r": float(mfe_r) if isinstance(mfe_r, (int, float)) else None,
        "label_mae_r": float(mae_r) if isinstance(mae_r, (int, float)) else None,
    }


def collect_candidate_rows(
    *,
    now,
    trad,
    cache,
    candidates,
    group_name,
    runtime_context,
    min_conf,
    dynamic_min_conf,
    checkpoint_stats=None,
    max_hold_bars=64,
    entry_fill_tolerance_pct=0.15,
):
    rows = []
    checkpoint_stats = checkpoint_stats if isinstance(checkpoint_stats, dict) else {}
    regime_summary = (checkpoint_stats.get("regime_summary") or (runtime_context or {}).get("regime_summary") or {})
    alert_budget = (checkpoint_stats.get("alert_budget") or (runtime_context or {}).get("alert_budget") or {})
    quality_drop_counts = checkpoint_stats.get("quality_drop_counts") or {}

    for rank, candidate in enumerate(candidates or [], start=1):
        if not isinstance(candidate, dict):
            continue
        snapshot = trad._candidate_ops_snapshot(candidate)
        symbol = str(snapshot.get("symbol") or "").strip()
        if not symbol:
            continue
        history_df = cache.get((symbol, "15m"))
        anchor_price = _candidate_anchor_price(candidate, snapshot)
        outcome = simulate_candidate_outcome(
            checkpoint_at=pd.Timestamp(now),
            history_df=history_df,
            signal=snapshot.get("signal"),
            entry_price=snapshot.get("entry_price"),
            stop_loss=snapshot.get("stop_loss"),
            take_profit=snapshot.get("take_profit"),
            current_price=anchor_price,
            max_hold_bars=max_hold_bars,
            entry_fill_tolerance_pct=entry_fill_tolerance_pct,
        )
        row = {
            "checkpoint_at": pd.Timestamp(now).isoformat(),
            "candidate_group": group_name,
            "candidate_rank": int(rank),
            "min_confidence": float(min_conf),
            "dynamic_min_confidence": float(dynamic_min_conf),
            "market_regime": str(regime_summary.get("market_regime") or "").strip() or None,
            "market_trend_bias": str(regime_summary.get("market_trend_bias") or "").strip() or None,
            "side_bias": str(regime_summary.get("side_bias") or "").strip() or None,
            "sell_bias_ratio": _safe_float(regime_summary.get("sell_bias_ratio"), None),
            "adjusted_run_cap": int(alert_budget.get("adjusted_run_cap") or 0),
            "symbol_cap": int(alert_budget.get("symbol_cap") or 0),
            "quality_drop_confidence": int(quality_drop_counts.get("candidate_profile_confidence_below_min") or 0),
            "quality_drop_entry_window": int(quality_drop_counts.get("primary_watch_intent_filtered") or 0),
            "price_at_checkpoint": _safe_float(anchor_price, None),
            "source_count": int(candidate.get("source_count") or 0)
            if isinstance(candidate.get("source_count"), (int, float))
            else (len(candidate.get("sources") or []) if isinstance(candidate.get("sources"), list) else 0),
        }
        row.update(snapshot)
        row.update(outcome)
        rows.append(row)
    return rows


def run_checkpoint(
    *,
    root,
    cache,
    checkpoint_at,
    watchlist,
    groups,
    max_hold_bars,
    entry_fill_tolerance_pct,
    research_supplements,
    research_pa_min_confidence,
    research_pa_min_score,
):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import trad

    now = pd.Timestamp(checkpoint_at).to_pydatetime()
    orig_get_yf_history = trad.get_yf_history
    orig_get_basic_info = trad.get_basic_info
    orig_get_thai_now = trad.get_thai_now
    orig_build_ctx = trad._build_alert_runtime_context
    try:
        trad._build_alert_runtime_context = lambda results, min_conf, **kwargs: orig_build_ctx(results, min_conf)
        trad.get_thai_now = lambda: pd.Timestamp(now).to_pydatetime()

        def fake_get_yf_history(symbol, period, interval=None, auto_adjust=True, cache_ttl_seconds=None):
            sym = trad.normalize_symbol(symbol)
            interval_text = str(interval or "15m").lower()
            if interval_text not in ("15m", "1h"):
                interval_text = "15m"
            df = cache.get((sym, interval_text))
            if df is None:
                return None
            sliced = slice_df(df, period, pd.Timestamp(now))
            return sliced if not sliced.empty else None

        trad.get_yf_history = fake_get_yf_history
        trad.get_basic_info = lambda symbol: {
            "name": trad.normalize_symbol(symbol),
            "sector": "N/A",
            "market_cap": 0,
            "pe_ratio": "N/A",
            "dividend_yield": 0,
        }
        trad.EMA_CROSS_15M_OPT_CACHE = {}
        results = [trad.analyze_single_symbol(symbol, "15m", include_chart_data=False) for symbol in watchlist]
        min_conf = float(trad._get_telegram_alert_min_confidence())
        runtime_context = trad._build_alert_runtime_context(results, min_conf)
        dynamic_min_conf = float((runtime_context or {}).get("dynamic_min_confidence") or min_conf)

        dataset_rows = []
        group_counts = {}
        if "primary" in groups and not bool((runtime_context or {}).get("kill")):
            primary_candidates, primary_stats = trad._build_telegram_candidates(
                results,
                dynamic_min_conf,
                runtime_context=runtime_context,
            )
            primary_candidates = list(primary_candidates or [])
            research_candidates = build_research_supplement_candidates(
                trad=trad,
                results=results,
                existing_candidates=primary_candidates,
                supplement_strategies=research_supplements,
                pa_min_confidence=research_pa_min_confidence,
                pa_min_score=research_pa_min_score,
            )
            if research_candidates:
                primary_candidates.extend(research_candidates)
            primary_rows = collect_candidate_rows(
                now=now,
                trad=trad,
                cache=cache,
                candidates=primary_candidates,
                group_name="primary",
                runtime_context=runtime_context,
                min_conf=min_conf,
                dynamic_min_conf=dynamic_min_conf,
                checkpoint_stats=primary_stats,
                max_hold_bars=max_hold_bars,
                entry_fill_tolerance_pct=entry_fill_tolerance_pct,
            )
            dataset_rows.extend(primary_rows)
            group_counts["primary"] = len(primary_rows)
            if research_candidates:
                group_counts["primary_research_supplement"] = len(research_candidates)

        if "trend_radar" in groups:
            radar_candidates = trad._build_trend_radar_candidates(results, runtime_context=runtime_context)
            radar_rows = collect_candidate_rows(
                now=now,
                trad=trad,
                cache=cache,
                candidates=radar_candidates,
                group_name="trend_radar",
                runtime_context=runtime_context,
                min_conf=min_conf,
                dynamic_min_conf=dynamic_min_conf,
                checkpoint_stats={},
                max_hold_bars=max_hold_bars,
                entry_fill_tolerance_pct=entry_fill_tolerance_pct,
            )
            dataset_rows.extend(radar_rows)
            group_counts["trend_radar"] = len(radar_rows)

        if "trend_state" in groups:
            trend_state_candidates = trad._build_trend_state_candidates(results, runtime_context=runtime_context)
            trend_state_rows = collect_candidate_rows(
                now=now,
                trad=trad,
                cache=cache,
                candidates=trend_state_candidates,
                group_name="trend_state",
                runtime_context=runtime_context,
                min_conf=min_conf,
                dynamic_min_conf=dynamic_min_conf,
                checkpoint_stats={},
                max_hold_bars=max_hold_bars,
                entry_fill_tolerance_pct=entry_fill_tolerance_pct,
            )
            dataset_rows.extend(trend_state_rows)
            group_counts["trend_state"] = len(trend_state_rows)

        if "daily" in groups:
            daily_candidates = trad._build_daily_best_pick_candidates(results, runtime_context=runtime_context)
            daily_rows = collect_candidate_rows(
                now=now,
                trad=trad,
                cache=cache,
                candidates=daily_candidates,
                group_name="daily",
                runtime_context=runtime_context,
                min_conf=min_conf,
                dynamic_min_conf=dynamic_min_conf,
                checkpoint_stats={},
                max_hold_bars=max_hold_bars,
                entry_fill_tolerance_pct=entry_fill_tolerance_pct,
            )
            dataset_rows.extend(daily_rows)
            group_counts["daily"] = len(daily_rows)

        checkpoint_summary = {
            "checkpoint_at": pd.Timestamp(now).isoformat(),
            "min_confidence": float(min_conf),
            "dynamic_min_confidence": float(dynamic_min_conf),
            "kill_switch_active": bool((runtime_context or {}).get("kill")),
            "kill_switch_reason": str((runtime_context or {}).get("kill_reason") or "") or None,
            "market_regime": str(((runtime_context or {}).get("regime_summary") or {}).get("market_regime") or "") or None,
            "side_bias": str(((runtime_context or {}).get("regime_summary") or {}).get("side_bias") or "") or None,
            "counts_by_group": group_counts,
            "candidate_count": len(dataset_rows),
        }
        return dataset_rows, checkpoint_summary
    finally:
        trad.get_yf_history = orig_get_yf_history
        trad.get_basic_info = orig_get_basic_info
        trad.get_thai_now = orig_get_thai_now
        trad._build_alert_runtime_context = orig_build_ctx


def build_summary(*, rows, checkpoints, args, watchlist, groups, cache_coverage=None, cache_refresh=None, workers=1):
    by_group = Counter()
    by_strategy = Counter()
    by_status = Counter()
    filled = 0
    wins = 0
    return_sum = 0.0
    return_count = 0
    for row in rows:
        by_group[str(row.get("candidate_group") or "unknown")] += 1
        by_strategy[str(row.get("strategy") or "UNKNOWN")] += 1
        by_status[str(row.get("label_status") or "unknown")] += 1
        if bool(row.get("label_filled")):
            filled += 1
        if row.get("label_win") is True:
            wins += 1
        value = row.get("label_return_pct")
        if isinstance(value, (int, float)):
            return_sum += float(value)
            return_count += 1
    total_rows = len(rows)
    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "window_days": int(args.days),
        "step": str(args.step),
        "end_at": str(args.end_at or "") or None,
        "watchlist": list(watchlist),
        "groups": list(groups),
        "max_hold_bars": int(args.max_hold_bars),
        "entry_fill_tolerance_pct": float(args.entry_fill_tolerance_pct),
        "research_strategy_supplements": parse_csv_upper(getattr(args, "research_strategy_supplements", "")),
        "research_pa_min_confidence": float(getattr(args, "research_pa_min_confidence", 56.0)),
        "research_pa_min_score": float(getattr(args, "research_pa_min_score", 58.0)),
        "workers": int(workers),
        "checkpoints": int(len(checkpoints)),
        "cache_requested_days": float((cache_coverage or {}).get("requested_days") or 0.0),
        "cache_available_days": float((cache_coverage or {}).get("available_days") or 0.0),
        "cache_requested_start": (cache_coverage or {}).get("requested_start"),
        "cache_effective_start": (cache_coverage or {}).get("effective_start"),
        "cache_latest_end": (cache_coverage or {}).get("latest_end"),
        "cache_has_full_coverage": bool((cache_coverage or {}).get("has_full_coverage")),
        "cache_refresh": cache_refresh or [],
        "cache_coverage_samples": (cache_coverage or {}).get("rows", [])[:10],
        "total_candidates": int(total_rows),
        "avg_candidates_per_checkpoint": (float(total_rows) / float(len(checkpoints))) if checkpoints else 0.0,
        "filled_candidates": int(filled),
        "fill_rate_pct": (float(filled) / float(total_rows) * 100.0) if total_rows else 0.0,
        "win_rate_pct": (float(wins) / float(filled) * 100.0) if filled else 0.0,
        "avg_return_pct": (return_sum / float(return_count)) if return_count else None,
        "by_group": dict(by_group),
        "by_strategy": dict(by_strategy),
        "by_label_status": dict(by_status),
        "checkpoint_samples": checkpoints[:10],
    }


def write_outputs(output_dir, rows, checkpoints, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_jsonl = output_dir / "phase1_candidates.jsonl"
    candidates_csv = output_dir / "phase1_candidates.csv"
    checkpoints_jsonl = output_dir / "phase1_checkpoints.jsonl"
    summary_json = output_dir / "phase1_summary.json"

    with candidates_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if rows:
        pd.DataFrame(rows).to_csv(candidates_csv, index=False)
    else:
        pd.DataFrame(columns=["checkpoint_at", "candidate_group", "strategy", "symbol"]).to_csv(candidates_csv, index=False)
    with checkpoints_jsonl.open("w", encoding="utf-8") as fh:
        for row in checkpoints:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "candidates_jsonl": str(candidates_jsonl),
        "candidates_csv": str(candidates_csv),
        "checkpoints_jsonl": str(checkpoints_jsonl),
        "summary_json": str(summary_json),
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    watchlist = parse_watchlist(args.watchlist)
    groups = parse_groups(args.groups)
    research_supplements = parse_csv_upper(args.research_strategy_supplements)
    output_dir = resolve_output_dir(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_refresh = []
    if args.refresh_cache:
        cache_refresh = refresh_cache(root, watchlist, args.days)
    cache = load_cache(root, watchlist)
    cache_coverage = summarize_cache_coverage(cache, days=args.days, end_at=args.end_at)
    if not bool(cache_coverage.get("has_full_coverage")) and not bool(args.allow_partial_coverage):
        raise RuntimeError(format_coverage_error(cache_coverage))
    points = compute_points(
        cache,
        args.days,
        args.step,
        end_at=args.end_at,
        allow_partial_coverage=bool(args.allow_partial_coverage),
    )
    total_points = len(points)
    if total_points <= 0:
        raise RuntimeError("No checkpoints generated for Phase 1 replay")
    workers = resolve_worker_count(args.workers, total_points)
    progress_every = max(1, int(args.progress_every or 1))
    started_at = time.perf_counter()
    print(
        f"[phase1] Starting replay with {total_points} checkpoints, workers={workers}, "
        f"groups={','.join(groups)}, step={args.step}, research_supplements={','.join(research_supplements) or 'none'}, output={output_dir}",
        flush=True,
    )
    write_progress(
        output_dir,
        {
            "status": "starting",
            "completed_checkpoints": 0,
            "total_checkpoints": int(total_points),
            "progress_pct": 0.0,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "workers": int(workers),
            "candidates_collected": 0,
            "last_checkpoint_at": None,
            "last_checkpoint_candidates": 0,
            "updated_at": pd.Timestamp.now().isoformat(),
        },
    )

    dataset_rows = []
    checkpoint_rows = []
    candidate_total = 0
    if workers <= 1:
        for index, now in enumerate(points, start=1):
            rows, checkpoint_summary = run_checkpoint(
                root=root,
                cache=cache,
                checkpoint_at=now,
                watchlist=watchlist,
                groups=groups,
                max_hold_bars=args.max_hold_bars,
                entry_fill_tolerance_pct=args.entry_fill_tolerance_pct,
                research_supplements=research_supplements,
                research_pa_min_confidence=args.research_pa_min_confidence,
                research_pa_min_score=args.research_pa_min_score,
            )
            dataset_rows.extend(rows)
            checkpoint_rows.append(checkpoint_summary)
            candidate_total += len(rows)
            if index == 1 or index == total_points or index % progress_every == 0:
                emit_progress(
                    output_dir=output_dir,
                    completed=index,
                    total=total_points,
                    started_at=started_at,
                    candidate_total=candidate_total,
                    workers=workers,
                    checkpoint_at=now,
                    last_candidate_count=len(rows),
                )
    else:
        ordered_rows = [None] * total_points
        ordered_checkpoints = [None] * total_points
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                str(root),
                cache,
                watchlist,
                groups,
                args.max_hold_bars,
                args.entry_fill_tolerance_pct,
                research_supplements,
                args.research_pa_min_confidence,
                args.research_pa_min_score,
            ),
        ) as executor:
            future_map = {
                executor.submit(_run_checkpoint_worker, index, now): (index, now)
                for index, now in enumerate(points)
            }
            completed = 0
            for future in as_completed(future_map):
                index, checkpoint_at = future_map[future]
                _, rows, checkpoint_summary = future.result()
                ordered_rows[index] = rows
                ordered_checkpoints[index] = checkpoint_summary
                completed += 1
                candidate_total += len(rows)
                if completed == 1 or completed == total_points or completed % progress_every == 0:
                    emit_progress(
                        output_dir=output_dir,
                        completed=completed,
                        total=total_points,
                        started_at=started_at,
                        candidate_total=candidate_total,
                        workers=workers,
                        checkpoint_at=checkpoint_at,
                        last_candidate_count=len(rows),
                    )
        for rows in ordered_rows:
            dataset_rows.extend(rows or [])
        checkpoint_rows = [row for row in ordered_checkpoints if row is not None]

    summary = build_summary(
        rows=dataset_rows,
        checkpoints=checkpoint_rows,
        args=args,
        watchlist=watchlist,
        groups=groups,
        cache_coverage=cache_coverage,
        cache_refresh=cache_refresh,
        workers=workers,
    )
    written = write_outputs(output_dir, dataset_rows, checkpoint_rows, summary)
    payload = {
        "output_dir": str(output_dir),
        "files": written,
        "summary": summary,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
