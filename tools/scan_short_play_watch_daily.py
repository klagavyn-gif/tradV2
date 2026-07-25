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
from tools import compare_short_play_watch_replay as compare  # noqa: E402


def _scan_day(*, cache, watchlist, day_end, step):
    checkpoints = replay.compute_points(cache, 1, step, end_at=day_end)
    if not checkpoints:
        return {
            "end_at": str(day_end),
            "checkpoints": 0,
            "total_alerts": 0,
            "watch_rows": 0,
            "standard_tier_short_play_watch_rows": 0,
            "standard_tier_short_play_watch_samples": [],
            "proxy_realized_for_day_window": {},
        }

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

    history = []
    state_rows = []
    latest_checkpoint_now = max(checkpoints).to_pydatetime()
    try:
        for now in checkpoints:
            state = {"now": pd.Timestamp(now).to_pydatetime()}
            checkpoint_history = []

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
                    compare._snapshot_candidate(state["now"], candidate, daily_pick=daily_pick)
                )
            )
            trad._record_telegram_run_report = lambda **kwargs: None
            trad.EMA_CROSS_15M_OPT_CACHE = {}

            results = [trad.analyze_single_symbol(symbol, "15m", include_chart_data=False) for symbol in watchlist]
            trad._notify_telegram_from_results(results)

            history.extend(checkpoint_history)
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

    standard_rows = [
        row
        for row in history
        if bool(row.get("short_play_watch_candidate")) and str(row.get("short_play_watch_tier") or "") == "standard"
    ]
    proxy = compare._summarize_proxy_outcomes(history, cache=cache, latest_now=latest_checkpoint_now)
    return {
        "end_at": str(day_end),
        "checkpoints": len(checkpoints),
        "total_alerts": len(history),
        "watch_rows": len([row for row in history if str(row.get("alert_intent") or "") == "watch"]),
        "standard_tier_short_play_watch_rows": len(standard_rows),
        "standard_tier_short_play_watch_samples": standard_rows[:10],
        "proxy_realized_for_day_window": proxy,
    }


def main():
    parser = argparse.ArgumentParser(description="Scan recent daily replay windows for standard-tier short-play watch candidates")
    parser.add_argument("--days", type=int, default=5, help="How many recent calendar days to scan")
    parser.add_argument("--step", default="1h", help="Checkpoint step inside each day window")
    parser.add_argument("--watchlist", default=",".join(replay.WATCHLIST), help="Comma-separated symbols")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    args = parser.parse_args()

    watchlist = replay.parse_watchlist(args.watchlist)
    cache = replay.load_cache(PROJECT_ROOT, watchlist)
    latest_now = min(df.index.max() for df in cache.values())
    day_points = pd.date_range(end=latest_now, periods=max(1, int(args.days)), freq="1D")

    payload = {
        "cache_earliest": str(max(df.index.min() for df in cache.values())),
        "cache_latest": str(latest_now),
        "step": str(args.step),
        "watchlist": watchlist,
        "days": [
            _scan_day(
                cache=cache,
                watchlist=watchlist,
                day_end=f"{str(pd.Timestamp(day).date())} 23:59:59",
                step=args.step,
            )
            for day in day_points
        ],
    }

    output_path = str(args.output or "").strip()
    if output_path:
        Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
