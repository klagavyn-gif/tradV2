import argparse
import json
import os
from pathlib import Path

import config
from alerts.auto_tuning import (
    build_auto_tuned_thresholds,
    read_alert_history_entries,
    write_auto_tuned_thresholds,
)


def resolve_history_path():
    return Path(__file__).resolve().parent / ".data" / "telegram_alerts" / "alert_history.jsonl"


def resolve_output_path():
    raw_path = str(getattr(config, "TELEGRAM_ALERT_AUTO_TUNE_OUTPUT_PATH", "") or "").strip()
    if raw_path:
        raw = Path(raw_path)
        return raw if raw.is_absolute() else (Path(__file__).resolve().parent / raw)
    return Path(__file__).resolve().parent / ".data" / "telegram_alerts" / "auto_tuned_thresholds.json"


def main():
    parser = argparse.ArgumentParser(description="Auto tune Telegram alert thresholds from real alert history")
    parser.add_argument("--history-path", default=str(resolve_history_path()), help="Path to alert_history.jsonl")
    parser.add_argument("--output-path", default=str(resolve_output_path()), help="Path to write tuned thresholds JSON")
    parser.add_argument(
        "--history-days",
        type=int,
        default=int(getattr(config, "TELEGRAM_ALERT_AUTO_TUNE_HISTORY_DAYS", 45)),
        help="Lookback window in days",
    )
    parser.add_argument(
        "--min-alerts-per-symbol",
        type=int,
        default=int(getattr(config, "TELEGRAM_ALERT_AUTO_TUNE_MIN_ALERTS_PER_SYMBOL", 12)),
        help="Minimum alerts required before tuning a symbol profile",
    )
    parser.add_argument(
        "--min-alerts-per-strategy",
        type=int,
        default=int(getattr(config, "TELEGRAM_ALERT_AUTO_TUNE_MIN_ALERTS_PER_STRATEGY", 20)),
        help="Minimum alerts required before tuning a strategy profile",
    )
    parser.add_argument(
        "--target-alerts-per-day",
        type=float,
        default=float(getattr(config, "TELEGRAM_ALERT_AUTO_TUNE_TARGET_ALERTS_PER_DAY", 2.0)),
        help="Target total Telegram alerts per day",
    )
    parser.add_argument(
        "--target-daily-picks-per-day",
        type=float,
        default=float(getattr(config, "TELEGRAM_ALERT_AUTO_TUNE_TARGET_DAILY_PICKS_PER_DAY", 1.0)),
        help="Target daily best picks per day",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the payload without writing the output file",
    )
    args = parser.parse_args()

    entries = read_alert_history_entries(args.history_path, days=args.history_days)
    payload = build_auto_tuned_thresholds(
        entries=entries,
        base_strategy_profiles=getattr(config, "TELEGRAM_ALERT_STRATEGY_QUALITY_PROFILES", {}) or {},
        base_symbol_profiles=getattr(config, "TELEGRAM_ALERT_SYMBOL_QUALITY_PROFILES", {}) or {},
        base_cdc_profiles=getattr(config, "CDC_VIXFIX_15M_SYMBOL_PROFILES", {}) or {},
        history_days=args.history_days,
        min_alerts_per_symbol=args.min_alerts_per_symbol,
        min_alerts_per_strategy=args.min_alerts_per_strategy,
        target_alerts_per_day=args.target_alerts_per_day,
        target_daily_pick_alerts_per_day=args.target_daily_picks_per_day,
    )

    if not args.print_only:
        write_auto_tuned_thresholds(args.output_path, payload)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
