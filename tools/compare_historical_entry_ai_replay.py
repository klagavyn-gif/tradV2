import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import trad  # noqa: E402
from tools import replay_historical_alerts as replay  # noqa: E402


DEFAULT_WATCHLIST = [
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


def _safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _parse_watchlist(value):
    items = [str(part or "").strip() for part in str(value or "").split(",")]
    items = [item for item in items if item]
    return items or list(DEFAULT_WATCHLIST)


def _snapshot_candidate(now, candidate, daily_pick):
    entry_bucket = str(candidate.get("entry_ai_bucket") or "").strip().lower() or None
    entry_tier = str(candidate.get("entry_ai_policy_tier") or "").strip().lower() or None
    alert_intent = str(candidate.get("alert_intent") or "").strip().lower() or None
    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "cache_key": str(candidate.get("cache_key") or "").strip() or None,
        "strategy": candidate.get("strategy"),
        "symbol": candidate.get("symbol"),
        "signal": candidate.get("signal"),
        "confidence": _safe_float(candidate.get("confidence"), None),
        "score": _safe_float(candidate.get("score"), None),
        "daily_pick": bool(daily_pick),
        "alert_intent": alert_intent,
        "entry_ai_bucket": entry_bucket,
        "entry_ai_policy_tier": entry_tier,
        "entry_ai_reason": candidate.get("entry_ai_reason"),
        "entry_ai_prob_entry": _safe_float(candidate.get("entry_ai_prob_entry"), None),
        "entry_ai_prob_watch": _safe_float(candidate.get("entry_ai_prob_watch"), None),
        "entry_ai_prob_avoid": _safe_float(candidate.get("entry_ai_prob_avoid"), None),
        "entry_ai_runtime_status": candidate.get("entry_ai_runtime_status"),
        "entry_ai_runtime_reason": candidate.get("entry_ai_runtime_reason"),
    }


def _alert_key(row):
    return "|".join(
        [
            str(row.get("timestamp") or ""),
            str(row.get("strategy") or ""),
            str(row.get("symbol") or ""),
            str(row.get("signal") or ""),
        ]
    )


def _is_entry_like(row):
    bucket = str(row.get("entry_ai_bucket") or "").strip().lower()
    tier = str(row.get("entry_ai_policy_tier") or "").strip().lower()
    return bucket == "entry" or tier in {"premium", "standard"}


def _run_model_replay(model_path, *, days, step, end_at, watchlist):
    trad._ENTRY_AI_MODEL_BUNDLE = None
    config.TELEGRAM_ALERT_ENTRY_AI_MODEL_PATH = str(model_path)
    config.TELEGRAM_ALERT_ENTRY_AI_LIVE_ENABLE = True
    config.TELEGRAM_ALERT_ENTRY_AI_FILTER_ENABLE = True
    config.TELEGRAM_ALERT_ENTRY_AI_MESSAGE_ENABLE = True
    config.TELEGRAM_ALERT_ENTRY_AI_RANKING_ENABLE = True
    config.TELEGRAM_ALERT_ENTRY_AI_LIVE_STRATEGIES = {"CDCVIX15", "PA15"}
    config.TELEGRAM_ALERT_ENTRY_AI_LIVE_GROUPS = {"PRIMARY"}

    cache = replay.load_cache(PROJECT_ROOT, watchlist)
    checkpoints = replay.compute_points(cache, days, step, end_at=end_at)
    state_rows = []
    history = []
    run_reports = []

    orig_build_ctx = trad._build_alert_runtime_context
    orig_get_market_history = trad.get_market_history
    orig_get_yf_history = trad.get_yf_history
    orig_get_basic_info = trad.get_basic_info
    orig_send_telegram_alert = trad.send_telegram_alert
    orig_track_alert_performance = trad._track_alert_performance
    orig_record_alert_history = trad._record_telegram_alert_history
    orig_record_run_report = trad._record_telegram_run_report
    orig_alert_cache = getattr(trad, "_TELEGRAM_ALERT_CACHE", None)
    orig_yf_cache = getattr(trad, "_YF_CACHE", None)
    orig_yf_info_cache = getattr(trad, "_YF_INFO_CACHE", None)
    orig_ema_cache = getattr(trad, "EMA_CROSS_15M_OPT_CACHE", None)

    try:
        trad._build_alert_runtime_context = lambda results, min_conf, **kwargs: orig_build_ctx(results, min_conf)
        total_checkpoints = len(checkpoints)
        for idx, now in enumerate(checkpoints, start=1):
            print(
                "[replay] model={model} checkpoint={idx}/{total} at={ts}".format(
                    model=Path(model_path).name,
                    idx=idx,
                    total=total_checkpoints,
                    ts=pd.Timestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                ),
                flush=True,
            )
            state = {"now": pd.Timestamp(now).to_pydatetime()}
            checkpoint_history = []
            checkpoint_reports = []

            def fake_get_market_history(symbol, period, interval=None, auto_adjust=True, cache_ttl_seconds=None):
                sym = trad.normalize_symbol(symbol)
                interval_text = str(interval or "15m").lower()
                if interval_text not in ("15m", "1h"):
                    interval_text = "15m"
                df = cache.get((sym, interval_text))
                if df is None:
                    return None
                sliced = replay.slice_df(df, period, state["now"])
                return sliced if not sliced.empty else None

            trad.get_market_history = fake_get_market_history
            trad.get_yf_history = fake_get_market_history
            trad.get_basic_info = lambda symbol: {
                "name": trad.normalize_symbol(symbol),
                "sector": "N/A",
                "market_cap": 0,
                "pe_ratio": "N/A",
                "dividend_yield": 0,
            }
            trad._TELEGRAM_ALERT_CACHE = replay.ReplayTTLCache(state, seed_rows=state_rows)
            trad._YF_CACHE = replay.ReplayTTLCache(state)
            trad._YF_INFO_CACHE = replay.ReplayTTLCache(state)
            trad.send_telegram_alert = lambda message: True
            trad._track_alert_performance = lambda *args, **kwargs: None
            trad._record_telegram_alert_history = (
                lambda candidate, min_conf=None, dynamic_min_conf=None, daily_pick=False: checkpoint_history.append(
                    _snapshot_candidate(state["now"], candidate, daily_pick)
                )
            )
            trad._record_telegram_run_report = lambda **kwargs: checkpoint_reports.append(dict(kwargs))
            trad.EMA_CROSS_15M_OPT_CACHE = {}

            results = [trad.analyze_single_symbol(symbol, "15m", include_chart_data=False) for symbol in watchlist]
            trad._notify_telegram_from_results(results)

            history.extend(checkpoint_history)
            run_reports.extend(checkpoint_reports)
            state_rows = trad._TELEGRAM_ALERT_CACHE.export_rows()
    finally:
        trad._build_alert_runtime_context = orig_build_ctx
        trad.get_market_history = orig_get_market_history
        trad.get_yf_history = orig_get_yf_history
        trad.get_basic_info = orig_get_basic_info
        trad.send_telegram_alert = orig_send_telegram_alert
        trad._track_alert_performance = orig_track_alert_performance
        trad._record_telegram_alert_history = orig_record_alert_history
        trad._record_telegram_run_report = orig_record_run_report
        trad._TELEGRAM_ALERT_CACHE = orig_alert_cache
        trad._YF_CACHE = orig_yf_cache
        trad._YF_INFO_CACHE = orig_yf_info_cache
        trad.EMA_CROSS_15M_OPT_CACHE = orig_ema_cache
        trad._ENTRY_AI_MODEL_BUNDLE = None

    return {
        "model_path": str(model_path),
        "checkpoints": len(checkpoints),
        "history": history,
        "run_reports": run_reports,
    }


def _model_summary(payload):
    history = list(payload.get("history") or [])
    return {
        "alerts": len(history),
        "entry_like_alerts": sum(1 for row in history if _is_entry_like(row)),
        "watch_alerts": sum(1 for row in history if str(row.get("entry_ai_policy_tier") or "").strip().lower() == "watch"),
        "avoid_alerts": sum(1 for row in history if str(row.get("entry_ai_policy_tier") or "").strip().lower() == "avoid"),
        "by_symbol": dict(Counter(str(row.get("symbol") or "UNKNOWN") for row in history)),
        "entry_like_by_symbol": dict(
            Counter(str(row.get("symbol") or "UNKNOWN") for row in history if _is_entry_like(row))
        ),
    }


def _compare_histories(v4_payload, v53_payload):
    v4_history = list(v4_payload.get("history") or [])
    v53_history = list(v53_payload.get("history") or [])
    v4_map = {_alert_key(row): row for row in v4_history}
    v53_map = {_alert_key(row): row for row in v53_history}

    v4_keys = set(v4_map.keys())
    v53_keys = set(v53_map.keys())
    added_keys = sorted(v53_keys - v4_keys)
    removed_keys = sorted(v4_keys - v53_keys)

    upgraded_to_entry = []
    changed_common_alerts = []
    for key in sorted(v4_keys & v53_keys):
        before = v4_map[key]
        after = v53_map[key]
        if (not _is_entry_like(before)) and _is_entry_like(after):
            upgraded_to_entry.append(
                {
                    "key": key,
                    "timestamp": after.get("timestamp"),
                    "strategy": after.get("strategy"),
                    "symbol": after.get("symbol"),
                    "signal": after.get("signal"),
                    "v4_bucket": before.get("entry_ai_bucket"),
                    "v4_tier": before.get("entry_ai_policy_tier"),
                    "v53_bucket": after.get("entry_ai_bucket"),
                    "v53_tier": after.get("entry_ai_policy_tier"),
                    "v53_prob_entry": after.get("entry_ai_prob_entry"),
                    "v53_prob_avoid": after.get("entry_ai_prob_avoid"),
                }
            )
        if (
            str(before.get("entry_ai_bucket") or "").strip().lower() != str(after.get("entry_ai_bucket") or "").strip().lower()
            or str(before.get("entry_ai_policy_tier") or "").strip().lower() != str(after.get("entry_ai_policy_tier") or "").strip().lower()
        ):
            changed_common_alerts.append(
                {
                    "key": key,
                    "timestamp": after.get("timestamp"),
                    "strategy": after.get("strategy"),
                    "symbol": after.get("symbol"),
                    "signal": after.get("signal"),
                    "v4_bucket": before.get("entry_ai_bucket"),
                    "v4_tier": before.get("entry_ai_policy_tier"),
                    "v53_bucket": after.get("entry_ai_bucket"),
                    "v53_tier": after.get("entry_ai_policy_tier"),
                    "v53_prob_entry": after.get("entry_ai_prob_entry"),
                    "v53_prob_avoid": after.get("entry_ai_prob_avoid"),
                }
            )

    changed_symbols = Counter()
    for key in added_keys:
        changed_symbols[str(v53_map[key].get("symbol") or "UNKNOWN")] += 1
    for key in removed_keys:
        changed_symbols[str(v4_map[key].get("symbol") or "UNKNOWN")] += 1
    for row in upgraded_to_entry:
        changed_symbols[str(row.get("symbol") or "UNKNOWN")] += 1

    return {
        "v4_summary": _model_summary(v4_payload),
        "v53_summary": _model_summary(v53_payload),
        "delta": {
            "alerts_added": len(added_keys),
            "alerts_removed": len(removed_keys),
            "entry_like_added": len(upgraded_to_entry),
        },
        "added_alerts": [v53_map[key] for key in added_keys[:200]],
        "removed_alerts": [v4_map[key] for key in removed_keys[:200]],
        "upgraded_to_entry": upgraded_to_entry[:200],
        "changed_common_alerts": changed_common_alerts[:200],
        "changed_symbols": dict(changed_symbols.most_common()),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Compare V4 vs V5.3 entry AI with historical replay")
    parser.add_argument("--v4-model-path", required=True)
    parser.add_argument("--v53-model-path", required=True)
    parser.add_argument("--days", type=float, default=2.0)
    parser.add_argument("--step", default="1h")
    parser.add_argument("--end-at", required=True)
    parser.add_argument("--watchlist", default=",".join(DEFAULT_WATCHLIST))
    parser.add_argument("--output-path", default="")
    return parser


def main():
    args = build_parser().parse_args()
    watchlist = _parse_watchlist(args.watchlist)
    v4_model_path = Path(args.v4_model_path).expanduser().resolve()
    v53_model_path = Path(args.v53_model_path).expanduser().resolve()

    print(f"[compare] starting V4 replay: {v4_model_path}", flush=True)
    v4_payload = _run_model_replay(
        v4_model_path,
        days=float(args.days),
        step=str(args.step),
        end_at=str(args.end_at),
        watchlist=watchlist,
    )
    print(f"[compare] starting V5.3 replay: {v53_model_path}", flush=True)
    v53_payload = _run_model_replay(
        v53_model_path,
        days=float(args.days),
        step=str(args.step),
        end_at=str(args.end_at),
        watchlist=watchlist,
    )
    comparison = _compare_histories(v4_payload, v53_payload)

    output_path = str(args.output_path or "").strip()
    if not output_path:
        output_path = str(PROJECT_ROOT / ".data" / "research" / "historical_entry_ai_compare_v4_vs_v53.json")
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "request": {
            "days": float(args.days),
            "step": str(args.step),
            "end_at": str(args.end_at),
            "watchlist": watchlist,
            "v4_model_path": str(v4_model_path),
            "v53_model_path": str(v53_model_path),
        },
        "comparison": comparison,
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare] wrote {output_file}", flush=True)
    print(str(output_file), flush=True)


if __name__ == "__main__":
    main()
