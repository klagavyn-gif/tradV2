from pathlib import Path
import argparse
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import trad  # noqa: E402
from tools import replay_historical_alerts as replay  # noqa: E402


def _candidate_snapshot(now_dt, candidate):
    if not isinstance(candidate, dict):
        return {}
    plan = candidate.get("plan") if isinstance(candidate.get("plan"), dict) else {}
    regime = candidate.get("regime") if isinstance(candidate.get("regime"), dict) else {}
    item = candidate.get("item") if isinstance(candidate.get("item"), dict) else {}
    trend_snapshot = trad.infer_1h_trend_snapshot(item) if isinstance(item, dict) else {}
    signal = str(candidate.get("signal") or "").strip().upper()
    optimizer_key = "sell_optimizer" if signal == "SELL" else "optimizer"
    wf = trad._extract_walkforward_metrics(plan, optimizer_key=optimizer_key)
    realized = trad._short_play_watch_realized_metrics(candidate)
    symbol_signal = realized.get("symbol_signal") if isinstance(realized, dict) else {}
    return {
        "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": candidate.get("symbol"),
        "signal": candidate.get("signal"),
        "strategy": candidate.get("strategy"),
        "confidence": trad._safe_float(candidate.get("confidence"), None),
        "score": trad._safe_float(candidate.get("score"), None),
        "short_play_gate_tier": str(candidate.get("short_play_gate_tier") or "").strip().lower() or None,
        "short_play_gate_reason": str(candidate.get("short_play_gate_reason") or "").strip() or None,
        "short_play_gate_regime_alignment": str(candidate.get("short_play_gate_regime_alignment") or "").strip().lower() or None,
        "profile_runtime_min_win_rate_pct": trad._safe_float(candidate.get("profile_runtime_min_win_rate_pct"), None),
        "profile_runtime_min_robustness_score": trad._safe_float(candidate.get("profile_runtime_min_robustness_score"), None),
        "walkforward_valid_trades": trad._safe_float(wf.get("valid_trades"), None),
        "walkforward_valid_win_rate_pct": trad._safe_float(wf.get("valid_win_rate_pct"), None),
        "walkforward_robustness_score": trad._safe_float(wf.get("robustness_score"), None),
        "realized_symbol_signal_settled": trad._safe_float((symbol_signal or {}).get("settled_alerts"), None),
        "realized_symbol_signal_win_rate_pct": trad._safe_float((symbol_signal or {}).get("win_rate_pct"), None),
        "realized_symbol_signal_avg_rr": trad._safe_float((symbol_signal or {}).get("avg_rr_realized"), None),
        "alert_intent": str(candidate.get("alert_intent") or "").strip().lower() or None,
        "alert_intent_reason": str(candidate.get("alert_intent_reason") or "").strip() or None,
        "sell_trigger": str(plan.get("sell_trigger") or plan.get("exit_trigger") or "").strip().upper() or None,
        "sell_continuation_override_mode": str(plan.get("sell_continuation_override_mode") or "").strip().lower() or None,
        "sell_continuation_override_reason": str(plan.get("sell_continuation_override_reason") or "").strip() or None,
        "forecast_direction": str(plan.get("forecast_direction") or "").strip().upper() or None,
        "forecast_score": trad._safe_float(plan.get("forecast_score"), None),
        "trend_bias": str(plan.get("trend_bias") or "").strip().upper() or None,
        "trend_color": str(plan.get("trend_color") or "").strip().upper() or None,
        "trend_1h": str((trend_snapshot or {}).get("trend") or "").strip().upper() or None,
        "trend_1h_strength": str((trend_snapshot or {}).get("strength") or "").strip().upper() or None,
        "market_regime": str(regime.get("market_regime") or "").strip().upper() or None,
        "side_bias": str(regime.get("side_bias") or "").strip().upper() or None,
        "short_play_watch_candidate": bool(candidate.get("short_play_watch_candidate")),
        "short_play_watch_floor_reason": str(candidate.get("short_play_watch_floor_reason") or "").strip() or None,
    }


def _collect_candidates(*, cache, watchlist, checkpoints):
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
    orig_watch_enable = bool(getattr(config, "TELEGRAM_ALERT_SHORT_PLAY_WATCH_ENABLE", True))

    trad._build_alert_runtime_context = lambda results, min_conf, **kwargs: orig_build_ctx(results, min_conf)
    config.TELEGRAM_ALERT_SHORT_PLAY_WATCH_ENABLE = True

    rows = []
    state_rows = []
    try:
        for now in checkpoints:
            state = {"now": pd.Timestamp(now).to_pydatetime()}
            checkpoint_payloads = []

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
            trad._record_telegram_alert_history = lambda *args, **kwargs: None
            trad._record_telegram_run_report = lambda **kwargs: checkpoint_payloads.append(kwargs)
            trad.EMA_CROSS_15M_OPT_CACHE = {}

            results = [trad.analyze_single_symbol(symbol, "15m", include_chart_data=False) for symbol in watchlist]
            trad._notify_telegram_from_results(results)

            for payload in checkpoint_payloads:
                merged_candidates = []
                for key in ("candidates", "raw_candidates"):
                    for candidate in payload.get(key) or []:
                        if isinstance(candidate, dict):
                            row = dict(candidate)
                            row.setdefault("_scan_source", key)
                            merged_candidates.append(row)
                for candidate in merged_candidates:
                    rows.append(_candidate_snapshot(state["now"], candidate))
            state_rows = trad._TELEGRAM_ALERT_CACHE.export_rows()
    finally:
        config.TELEGRAM_ALERT_SHORT_PLAY_WATCH_ENABLE = orig_watch_enable
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
    return rows


def main():
    parser = argparse.ArgumentParser(description="Scan replay candidates for short-play walk-forward support")
    parser.add_argument("--days", type=int, default=3, help="Replay window in days")
    parser.add_argument("--step", default="2h", help="Checkpoint step")
    parser.add_argument("--watchlist", default=",".join(replay.WATCHLIST), help="Comma-separated symbols")
    parser.add_argument("--symbols", default="BTC-USD,PAXG-USD,TRX-USD", help="Focus symbols")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    args = parser.parse_args()

    watchlist = replay.parse_watchlist(args.watchlist)
    focus_symbols = {trad.normalize_symbol(symbol) for symbol in replay.parse_watchlist(args.symbols)}
    cache = replay.load_cache(PROJECT_ROOT, watchlist)
    checkpoints = replay.compute_points(cache, args.days, args.step)
    rows = _collect_candidates(cache=cache, watchlist=watchlist, checkpoints=checkpoints)

    focus_rows = [
        row for row in rows
        if str(row.get("symbol") or "") in focus_symbols and str(row.get("signal") or "").upper() == "SELL"
    ]
    wf_rows = [
        row for row in focus_rows
        if isinstance(row.get("walkforward_valid_trades"), (int, float))
        or isinstance(row.get("walkforward_valid_win_rate_pct"), (int, float))
        or isinstance(row.get("walkforward_robustness_score"), (int, float))
    ]
    payload = {
        "step": str(args.step),
        "days": int(args.days),
        "focus_symbols": sorted(focus_symbols),
        "total_candidates": len(rows),
        "focus_sell_candidates": len(focus_rows),
        "focus_sell_candidates_with_walkforward": len(wf_rows),
        "focus_sell_samples": focus_rows[:20],
        "focus_sell_walkforward_samples": wf_rows[:20],
    }

    output_path = str(args.output or "").strip()
    if output_path:
        Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
