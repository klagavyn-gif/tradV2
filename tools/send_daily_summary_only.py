import argparse
import json
from pathlib import Path

from application.services.service_support import (
    analyze_symbols_batch,
    get_telegram_alert_min_confidence,
    max_symbols_per_request,
    parse_symbols_input,
)
from domain.alerts.dispatch.cache_policy import cache_contains
from domain.alerts.dispatch.delivery import dispatch_daily_summary
from domain.alerts.dispatch.throttling import resolve_dispatch_settings


def build_parser():
    parser = argparse.ArgumentParser(description="Send Telegram daily summary without primary alerts")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--period", default="15m")
    parser.add_argument("--verify-output", default="")
    parser.add_argument("--force", action="store_true")
    return parser


def _write_json(output_path, payload):
    path = Path(str(output_path or "").strip())
    if not path:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def main(argv=None):
    import trad

    args = build_parser().parse_args(argv)
    max_symbols = max_symbols_per_request(trad.config)
    symbols = parse_symbols_input(
        args.symbols,
        normalize_symbol=trad.normalize_symbol,
        default_max_symbols=max_symbols,
        max_symbols=max_symbols,
    )
    period = str(args.period or "15m")
    in_window = bool(trad._is_daily_best_pick_window())
    forced = bool(args.force)

    if not symbols:
        payload = {
            "status": "invalid_symbols",
            "symbols": [],
            "period": period,
            "in_window": in_window,
            "forced": forced,
        }
        if args.verify_output:
            _write_json(args.verify_output, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 2

    if period not in trad.VALID_PERIODS:
        payload = {
            "status": "invalid_period",
            "symbols": symbols,
            "period": period,
            "in_window": in_window,
            "forced": forced,
        }
        if args.verify_output:
            _write_json(args.verify_output, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 2

    if not in_window and not forced:
        payload = {
            "status": "skipped_outside_window",
            "symbols": symbols,
            "period": period,
            "in_window": in_window,
            "forced": forced,
        }
        if args.verify_output:
            _write_json(args.verify_output, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    results = analyze_symbols_batch(
        symbols,
        period,
        include_chart_data=False,
        analyze_single_symbol=trad.analyze_single_symbol,
        executor=trad._ANALYZE_EXECUTOR,
        repeat_values=trad.repeat,
    )
    base_min_conf = get_telegram_alert_min_confidence(trad.config)
    runtime_context = trad._build_alert_runtime_context(results, base_min_conf)
    limits = resolve_dispatch_settings(trad.config, runtime_context)
    dynamic_min_conf = float(limits["dynamic_min_conf"])
    candidates, build_stats = trad._build_telegram_candidates(
        results,
        dynamic_min_conf,
        runtime_context=runtime_context,
    )
    daily_summary = trad._build_daily_summary_message(
        results,
        existing_candidates=candidates,
        min_conf=dynamic_min_conf,
    )

    cache_key = ""
    cache_hit = False
    sent = False
    status = "no_summary_payload"
    if isinstance(daily_summary, dict):
        cache_key = str(daily_summary.get("cache_key") or "").strip()
        cache_hit = bool(cache_contains(trad._TELEGRAM_ALERT_CACHE, cache_key))
        if cache_hit:
            status = "cached"
        else:
            sent = bool(
                dispatch_daily_summary(
                    daily_summary,
                    send_telegram_alert=trad.send_telegram_alert,
                    telegram_alert_cache=trad._TELEGRAM_ALERT_CACHE,
                    record_telegram_alert_history=trad._record_telegram_alert_history,
                    limits=limits,
                )
            )
            status = "sent" if sent else "send_failed"

    payload = {
        "status": status,
        "symbols": symbols,
        "period": period,
        "in_window": in_window,
        "forced": forced,
        "sent": sent,
        "cache_hit": cache_hit,
        "cache_key": cache_key or None,
        "generated_at": trad.get_thai_now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": len(candidates or []),
        "quality_drop_counts": (build_stats or {}).get("quality_drop_counts") or {},
    }
    if args.verify_output:
        payload["verify_output_path"] = _write_json(args.verify_output, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if status in {"sent", "cached"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
