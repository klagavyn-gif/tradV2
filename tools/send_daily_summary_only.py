import argparse
import json
from pathlib import Path

from application.services.service_support import (
    max_symbols_per_request,
    parse_symbols_input,
)
from domain.alerts.dispatch.cache_policy import cache_contains
from domain.alerts.dispatch.delivery import dispatch_daily_summary


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


def _read_json(path_text):
    path = Path(str(path_text or "").strip())
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_summary_candidates(trad):
    latest_run = _read_json(trad._alert_run_report_file_path())
    if not isinstance(latest_run, dict):
        return [], {}
    for key in ("top_candidates", "raw_top_candidates"):
        rows = latest_run.get(key)
        if isinstance(rows, list):
            candidates = [row for row in rows if isinstance(row, dict)]
            if candidates:
                return candidates, latest_run
    return [], latest_run


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

    existing_candidates, latest_run = _load_summary_candidates(trad)
    latest_run_budget = latest_run.get("alert_budget") if isinstance(latest_run.get("alert_budget"), dict) else {}
    min_conf = latest_run.get("min_confidence")
    if not isinstance(min_conf, (int, float)):
        min_conf = getattr(trad.config, "TELEGRAM_DAILY_BEST_PICK_MIN_CONFIDENCE", 58.0)
    min_conf = float(min_conf)
    dynamic_min_conf = latest_run.get("dynamic_min_confidence")
    if not isinstance(dynamic_min_conf, (int, float)):
        dynamic_min_conf = min_conf
    if not isinstance(dynamic_min_conf, (int, float)):
        dynamic_min_conf = getattr(trad.config, "TELEGRAM_DAILY_BEST_PICK_MIN_CONFIDENCE", 58.0)
    dynamic_min_conf = float(dynamic_min_conf)
    limits = {
        "min_conf": min_conf,
        "dynamic_min_conf": dynamic_min_conf,
        "run_cap": latest_run_budget.get("adjusted_run_cap"),
        "per_symbol_cap": latest_run_budget.get("adjusted_per_symbol_cap"),
        "daily_pick_cap": latest_run_budget.get("adjusted_daily_pick_cap"),
    }
    daily_summary = trad._build_daily_summary_message(
        [],
        existing_candidates=existing_candidates,
        min_conf=dynamic_min_conf,
    )

    outlook_payload = None
    outlook_path = None
    outlook_sent = False
    outlook_skipped = False
    outlook_history_path = Path(trad._alert_history_dir()) / "outlook_history.jsonl"
    if bool(getattr(trad.config, "DAILY_AI_OUTLOOK_ENABLE", True)):
        try:
            from alerts.daily_outlook import build_daily_ai_outlook, mark_outlook_sent

            alert_history = trad._read_telegram_alert_history(days=2)
            outcomes_payload = _read_json(trad._alert_outcomes_file_path())
            outcomes = outcomes_payload.get("outcomes") if isinstance(outcomes_payload, dict) else []
            verify_payload = _read_json(Path(trad._alert_history_dir()) / "workflow_verify.json")
            outlook = build_daily_ai_outlook(
                config=trad.config,
                candidates=existing_candidates,
                alert_history=alert_history,
                outcomes=outcomes,
                verify_payload=verify_payload,
                history_path=outlook_history_path,
                now=trad.get_thai_now(),
            )
        except Exception as exc:
            trad.logger.warning("Daily AI outlook failed: %s", exc)
            outlook = None
        if isinstance(outlook, dict):
            outlook_payload = outlook.get("payload") or {}
            outlook_path = Path(trad._alert_history_dir()) / "daily_ai_outlook.json"
            _write_json(str(outlook_path), outlook_payload)
            if outlook_payload.get("already_sent"):
                outlook_skipped = True
            elif isinstance(outlook.get("message"), str) and outlook["message"].strip():
                outlook_sent = bool(trad.send_telegram_alert(outlook["message"]))
                if outlook_sent:
                    mark_outlook_sent(
                        outlook_history_path,
                        outlook_payload.get("record_date"),
                        trad.get_thai_now().strftime("%Y-%m-%d %H:%M:%S"),
                    )

    recent_cache_keys = trad._load_recent_alert_cache_keys(
        trad.get_thai_now,
        max_age_seconds=26 * 60 * 60,
    )
    cache_key = ""
    cache_hit = False
    sent = False
    status = "no_summary_payload"
    if isinstance(daily_summary, dict):
        cache_key = str(daily_summary.get("cache_key") or "").strip()
        cache_hit = bool(cache_contains(trad._TELEGRAM_ALERT_CACHE, cache_key) or cache_key in recent_cache_keys)
        if cache_hit:
            status = "cached"
        else:
            sent = bool(
                dispatch_daily_summary(
                    daily_summary,
                    get_now=trad.get_thai_now,
                    send_telegram_alert=trad.send_telegram_alert,
                    telegram_alert_cache=trad._TELEGRAM_ALERT_CACHE,
                    record_telegram_alert_history=trad._record_telegram_alert_history,
                    limits=limits,
                    recent_cache_keys=recent_cache_keys,
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
        "candidate_count": len(existing_candidates or []),
        "quality_drop_counts": latest_run.get("quality_drop_counts") if isinstance(latest_run, dict) else {},
        "ai_outlook": {
            "enabled": bool(getattr(trad.config, "DAILY_AI_OUTLOOK_ENABLE", True)),
            "sent": outlook_sent,
            "skipped_already_sent": outlook_skipped,
            "bias": (outlook_payload or {}).get("bias"),
            "llm_used": bool((outlook_payload or {}).get("llm_used")),
            "llm_finish_reason": (outlook_payload or {}).get("llm_finish_reason"),
            "news_count": len((outlook_payload or {}).get("news") or []),
            "calibration_buckets": len((outlook_payload or {}).get("calibration") or []),
            "scorecard": (outlook_payload or {}).get("scorecard"),
        },
        "ai_outlook_path": str(outlook_path) if outlook_path else None,
    }
    if args.verify_output:
        payload["verify_output_path"] = _write_json(args.verify_output, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if status in {"sent", "cached"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
