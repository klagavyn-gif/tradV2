from pathlib import Path
from collections import Counter
import argparse
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import trad  # noqa: E402
from alerts import reporting as alerts_reporting  # noqa: E402
from tools import replay_historical_alerts as replay  # noqa: E402


def _pick_numeric(plan, keys):
    if not isinstance(plan, dict):
        return None
    for key in keys:
        value = plan.get(key)
        try:
            if value is not None and value != "":
                return float(value)
        except Exception:
            continue
    return None


def _snapshot_candidate(now_dt, candidate, *, daily_pick=False):
    plan = candidate.get("plan") if isinstance(candidate.get("plan"), dict) else {}
    edge = candidate.get("edge_metrics") if isinstance(candidate.get("edge_metrics"), dict) else {}
    regime = candidate.get("regime") if isinstance(candidate.get("regime"), dict) else {}
    item = candidate.get("item") if isinstance(candidate.get("item"), dict) else {}
    trend_snapshot = trad.infer_1h_trend_snapshot(item) if isinstance(item, dict) else {}
    source_trend_snapshot = (
        trad.infer_1h_trend_snapshot(
            item,
            include_labels=("ActionZone 15m", "Price Action 15m", "Trend Breakout 15m"),
        )
        if isinstance(item, dict)
        else {}
    )
    signal = str(candidate.get("signal") or "").strip().upper()
    optimizer_key = "sell_optimizer" if signal == "SELL" else "optimizer"
    walkforward = trad._extract_walkforward_metrics(plan, optimizer_key=optimizer_key)
    return {
        "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": candidate.get("strategy"),
        "symbol": candidate.get("symbol"),
        "signal": candidate.get("signal"),
        "daily_pick": bool(daily_pick),
        "timeframe": str(candidate.get("timeframe") or candidate.get("interval") or plan.get("interval") or "15m"),
        "evaluation_window_bars": int(alerts_reporting._candidate_evaluation_window_bars(candidate, config=config)),
        "entry_price": _pick_numeric(plan, ["entry_price", "current_price", "price"]),
        "stop_loss": _pick_numeric(plan, ["stop_loss", "entry_stop_loss", "trailing_stop"]),
        "take_profit": _pick_numeric(plan, ["take_profit", "take_profit_2", "exit_price"]),
        "confidence": alerts_reporting._safe_float(candidate.get("confidence"), None),
        "score": alerts_reporting._safe_float(candidate.get("score"), None),
        "source_count": alerts_reporting._safe_int(candidate.get("source_count"), None),
        "backtest_win_rate_pct": alerts_reporting._safe_float(edge.get("win_rate_pct"), None),
        "backtest_expectancy_rr": alerts_reporting._safe_float(edge.get("expectancy_rr"), None),
        "backtest_trades": alerts_reporting._safe_float(edge.get("trades"), None),
        "walkforward_valid_win_rate_pct": alerts_reporting._safe_float(walkforward.get("valid_win_rate_pct"), None),
        "walkforward_robustness_score": alerts_reporting._safe_float(walkforward.get("robustness_score"), None),
        "cache_key": str(candidate.get("cache_key") or "").strip() or None,
        "alert_intent": str(candidate.get("alert_intent") or "").strip().lower() or None,
        "alert_intent_reason": str(candidate.get("alert_intent_reason") or "").strip() or None,
        "sell_trigger": str(plan.get("sell_trigger") or plan.get("exit_trigger") or "").strip().upper() or None,
        "sell_signal_role": str(plan.get("sell_signal_role") or "").strip().lower() or None,
        "alert_intent_hint": str(plan.get("alert_intent_hint") or "").strip().lower() or None,
        "alert_intent_hint_reason": str(plan.get("alert_intent_hint_reason") or "").strip() or None,
        "sell_continuation_override_mode": str(plan.get("sell_continuation_override_mode") or "").strip().lower() or None,
        "sell_continuation_override_reason": str(plan.get("sell_continuation_override_reason") or "").strip() or None,
        "forecast_direction": str(plan.get("forecast_direction") or "").strip().upper() or None,
        "forecast_score": alerts_reporting._safe_float(plan.get("forecast_score"), None),
        "trend_bias": str(plan.get("trend_bias") or "").strip().upper() or None,
        "trend_color": str(plan.get("trend_color") or "").strip().upper() or None,
        "trend_1h": str((trend_snapshot or {}).get("trend") or "").strip().upper() or None,
        "trend_1h_strength": str((trend_snapshot or {}).get("strength") or "").strip().upper() or None,
        "trend_1h_source": str((source_trend_snapshot or {}).get("trend") or "").strip().upper() or None,
        "trend_1h_source_strength": str((source_trend_snapshot or {}).get("strength") or "").strip().upper() or None,
        "market_regime": str(regime.get("market_regime") or "").strip().upper() or None,
        "side_bias": str(regime.get("side_bias") or "").strip().upper() or None,
        "short_play_watch_candidate": bool(candidate.get("short_play_watch_candidate")),
        "short_play_watch_tier": str(candidate.get("short_play_watch_tier") or "").strip().lower() or None,
        "short_play_watch_reason": str(candidate.get("short_play_watch_reason") or "").strip() or None,
        "short_play_watch_gap": alerts_reporting._safe_float(candidate.get("short_play_watch_gap"), None),
        "short_play_watch_floor_reason": str(candidate.get("short_play_watch_floor_reason") or "").strip() or None,
        "short_play_watch_realized_source": str(candidate.get("short_play_watch_realized_source") or "").strip().lower() or None,
        "short_play_watch_realized_settled_alerts": alerts_reporting._safe_float(candidate.get("short_play_watch_realized_settled_alerts"), None),
        "short_play_watch_realized_win_rate_pct": alerts_reporting._safe_float(candidate.get("short_play_watch_realized_win_rate_pct"), None),
        "short_play_watch_realized_avg_rr": alerts_reporting._safe_float(candidate.get("short_play_watch_realized_avg_rr"), None),
        "short_play_watch_walkforward_valid_trades": alerts_reporting._safe_float(candidate.get("short_play_watch_walkforward_valid_trades"), None),
        "short_play_watch_walkforward_valid_win_rate_pct": alerts_reporting._safe_float(candidate.get("short_play_watch_walkforward_valid_win_rate_pct"), None),
        "short_play_watch_walkforward_robustness_score": alerts_reporting._safe_float(candidate.get("short_play_watch_walkforward_robustness_score"), None),
        "short_play_watch_required_robustness_score": alerts_reporting._safe_float(candidate.get("short_play_watch_required_robustness_score"), None),
        "short_play_watch_strategy_support_selected_alerts": alerts_reporting._safe_float(candidate.get("short_play_watch_strategy_support_selected_alerts"), None),
        "short_play_watch_strategy_support_weight_total": alerts_reporting._safe_float(candidate.get("short_play_watch_strategy_support_weight_total"), None),
        "short_play_watch_strategy_support_win_rate_pct": alerts_reporting._safe_float(candidate.get("short_play_watch_strategy_support_win_rate_pct"), None),
        "short_play_watch_strategy_support_expectancy_rr": alerts_reporting._safe_float(candidate.get("short_play_watch_strategy_support_expectancy_rr"), None),
        "short_play_watch_support_source": str(candidate.get("short_play_watch_support_source") or "").strip().lower() or None,
        "short_play_gate_tier": str(candidate.get("short_play_gate_tier") or "").strip().lower() or None,
        "short_play_gate_regime_alignment": str(candidate.get("short_play_gate_regime_alignment") or "").strip().lower() or None,
        "profile_runtime_min_win_rate_pct": alerts_reporting._safe_float(candidate.get("profile_runtime_min_win_rate_pct"), None),
        "profile_runtime_min_source_count": alerts_reporting._safe_int(candidate.get("profile_runtime_min_source_count"), None),
        "profile_runtime_min_robustness_score": alerts_reporting._safe_float(candidate.get("profile_runtime_min_robustness_score"), None),
        "short_trade_bucket": str(candidate.get("short_trade_bucket") or "").strip().lower() or None,
        "message_plain": str(candidate.get("message_plain") or candidate.get("message") or "")[:240],
    }


def _future_history_factory(cache):
    def _future_history(symbol, period, interval=None, auto_adjust=True, cache_ttl_seconds=None):
        sym = trad.normalize_symbol(symbol)
        interval_text = str(interval or "15m").lower()
        if interval_text not in ("15m", "1h"):
            interval_text = "15m"
        return cache.get((sym, interval_text))

    return _future_history


def _summarize_proxy_outcomes(history_rows, *, cache, latest_now):
    orig_get_yf_history = trad.get_yf_history
    trad.get_yf_history = _future_history_factory(cache)
    try:
        helpers = trad._reporting_module_helpers()
        directional, outcomes = alerts_reporting._resolve_directional_alert_outcomes(
            history_rows,
            helpers=helpers,
            get_now=lambda: latest_now,
        )
    finally:
        trad.get_yf_history = orig_get_yf_history

    directional_by_key = {}
    for row in directional:
        key = "|".join(
            [
                str(row.get("timestamp") or ""),
                str(row.get("strategy") or ""),
                str(row.get("symbol") or ""),
                str(row.get("signal") or ""),
            ]
        )
        directional_by_key[key] = dict(row)

    merged = []
    for outcome in outcomes:
        key = "|".join(
            [
                str(outcome.get("timestamp") or ""),
                str(outcome.get("strategy") or ""),
                str(outcome.get("symbol") or ""),
                str(outcome.get("signal") or ""),
            ]
        )
        row = dict(directional_by_key.get(key) or {})
        row.update(outcome)
        merged.append(row)

    settled = [row for row in merged if str(row.get("outcome_status") or "") == "settled"]
    wins = [row for row in settled if str(row.get("outcome_result") or "") == "win"]
    losses = [row for row in settled if str(row.get("outcome_result") or "") == "loss"]
    by_intent = {}
    for intent in ("entry", "watch"):
        subset = [row for row in settled if str(row.get("alert_intent") or "") == intent]
        win_count = sum(1 for row in subset if str(row.get("outcome_result") or "") == "win")
        by_intent[intent] = {
            "settled": len(subset),
            "wins": win_count,
            "losses": sum(1 for row in subset if str(row.get("outcome_result") or "") == "loss"),
            "win_rate_pct": round(100.0 * win_count / len(subset), 2) if subset else None,
        }
    return {
        "directional_alerts": len(directional),
        "settled": len(settled),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(100.0 * len(wins) / len(settled), 2) if settled else None,
        "by_intent": by_intent,
    }


def _run_mode(*, watch_enabled, cache, checkpoints, watchlist):
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
    config.TELEGRAM_ALERT_SHORT_PLAY_WATCH_ENABLE = bool(watch_enabled)

    history = []
    run_reports = []
    state_rows = []
    latest_now = max(df.index.max() for df in cache.values()).to_pydatetime()
    try:
        for now in checkpoints:
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
                    _snapshot_candidate(state["now"], candidate, daily_pick=daily_pick)
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

    by_intent = Counter(str(row.get("alert_intent") or "unknown") for row in history)
    by_day = Counter(str(row.get("timestamp") or "")[:10] for row in history)
    by_symbol = Counter(str(row.get("symbol") or "UNKNOWN") for row in history)
    watch_rows = [row for row in history if str(row.get("alert_intent") or "") == "watch"]
    confidence_values = [float(row.get("confidence")) for row in history if isinstance(row.get("confidence"), (int, float))]
    win_rate_values = [
        float(row.get("backtest_win_rate_pct"))
        for row in history
        if isinstance(row.get("backtest_win_rate_pct"), (int, float))
    ]
    proxy = _summarize_proxy_outcomes(history, cache=cache, latest_now=latest_now)
    return {
        "watch_enabled": bool(watch_enabled),
        "checkpoints": len(checkpoints),
        "total_alerts": len(history),
        "days_with_alerts": len(by_day),
        "alerts_per_day_avg": round(len(history) / max(1.0, len(by_day)), 2) if by_day else 0.0,
        "by_intent": dict(by_intent),
        "watch_candidates": len(watch_rows),
        "avg_confidence": round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else None,
        "avg_backtest_win_rate_pct": round(sum(win_rate_values) / len(win_rate_values), 2) if win_rate_values else None,
        "by_symbol": dict(by_symbol),
        "sample_watch": watch_rows[:5],
        "proxy_realized": proxy,
        "avg_candidates_per_run": (
            round(
                sum(len(report.get("candidates") or []) for report in run_reports) / len(run_reports),
                3,
            )
            if run_reports
            else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare replay results with short-play watch disabled versus enabled")
    parser.add_argument("--days", type=int, default=7, help="Replay window in days")
    parser.add_argument("--step", default="12h", help="Checkpoint step")
    parser.add_argument("--watchlist", default=",".join(replay.WATCHLIST), help="Comma-separated symbols")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    args = parser.parse_args()

    watchlist = replay.parse_watchlist(args.watchlist)
    cache = replay.load_cache(PROJECT_ROOT, watchlist)
    checkpoints = replay.compute_points(cache, args.days, args.step)

    baseline = _run_mode(watch_enabled=False, cache=cache, checkpoints=checkpoints, watchlist=watchlist)
    enabled = _run_mode(watch_enabled=True, cache=cache, checkpoints=checkpoints, watchlist=watchlist)
    payload = {
        "window_days": int(args.days),
        "step": str(args.step),
        "watchlist": watchlist,
        "baseline_watch_off": baseline,
        "watch_on": enabled,
    }

    output_path = str(args.output or "").strip()
    if output_path:
        Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
