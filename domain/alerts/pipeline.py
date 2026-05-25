import json

from domain.alerts.dispatch.delivery import (
    dispatch_daily_candidates,
    dispatch_daily_summary,
    dispatch_primary_candidates,
)
from domain.alerts.dispatch.throttling import coerce_float, coerce_int, resolve_dispatch_settings

def notify_telegram_from_results(results, *, config, helpers, get_now, logger, runtime_context=None):
    build_alert_runtime_context = helpers["build_alert_runtime_context"]
    build_telegram_candidates = helpers["build_telegram_candidates"]
    is_daily_best_pick_window = helpers["is_daily_best_pick_window"]
    build_daily_best_pick_candidates = helpers["build_daily_best_pick_candidates"]
    build_daily_summary_message = helpers["build_daily_summary_message"]
    send_telegram_alert = helpers["send_telegram_alert"]
    telegram_alert_cache = helpers["telegram_alert_cache"]
    record_telegram_alert_history = helpers["record_telegram_alert_history"]
    track_alert_performance = helpers["track_alert_performance"]
    record_telegram_run_report = helpers["record_telegram_run_report"]

    min_conf = coerce_float(getattr(config, "TELEGRAM_ALERT_MIN_CONFIDENCE", 69.0), 69.0)
    if not isinstance(runtime_context, dict):
        runtime_context = build_alert_runtime_context(results or [], min_conf, config=config, helpers=helpers, get_now=get_now)
    else:
        try:
            min_conf = float((runtime_context or {}).get("min_confidence"))
        except Exception:
            pass
    limits = resolve_dispatch_settings(config, runtime_context)
    min_conf = limits["min_conf"]
    kill = bool((runtime_context or {}).get("kill"))
    reason = (runtime_context or {}).get("kill_reason")
    if kill:
        logger.warning("Telegram kill switch active; skip alerts (%s)", reason)

    alert_budget = limits["alert_budget"]
    dynamic_min_conf = limits["dynamic_min_conf"]
    candidates = []
    build_stats = {}
    if not kill:
        candidates, build_stats = build_telegram_candidates(results, dynamic_min_conf, runtime_context=runtime_context)

    quality_drop_counts = {}
    if isinstance(build_stats, dict):
        quality_drop_counts = build_stats.get("quality_drop_counts") or {}
        if not isinstance(alert_budget, dict) or not alert_budget:
            alert_budget = build_stats.get("alert_budget") or {}

    if not candidates:
        logger.info(
            "Telegram alerts: no primary candidates (min_conf=%.1f, dynamic_min_conf=%.1f, budget=%s quality_drops=%s)",
            min_conf,
            dynamic_min_conf,
            json.dumps(alert_budget or {}, ensure_ascii=False),
            json.dumps(quality_drop_counts, ensure_ascii=False),
        )
        if kill and not is_daily_best_pick_window():
            record_telegram_run_report(
                results=results,
                kill=kill,
                kill_reason=reason,
                min_conf=min_conf,
                dynamic_min_conf=dynamic_min_conf,
                candidates=candidates,
                sent_candidates=[],
                daily_pick_sent=0,
                daily_summary_sent=0,
                dropped_by_cache=0,
                dropped_by_symbol_cap=0,
                dropped_by_run_cap=0,
                quality_drop_counts=quality_drop_counts,
                alert_budget=alert_budget,
            )
            return 0

    candidates.sort(key=lambda c: (float(c.get("score", 0.0)), float(c.get("confidence", 0.0))), reverse=True)
    primary_dispatch = dispatch_primary_candidates(
        candidates,
        send_telegram_alert=send_telegram_alert,
        telegram_alert_cache=telegram_alert_cache,
        record_telegram_alert_history=record_telegram_alert_history,
        limits=limits,
    )
    sent = int(primary_dispatch["sent"])
    dropped_by_cache = int(primary_dispatch["dropped_by_cache"])
    dropped_by_symbol_cap = int(primary_dispatch["dropped_by_symbol_cap"])
    dropped_by_run_cap = int(primary_dispatch["dropped_by_run_cap"])
    per_symbol_sent = dict(primary_dispatch["per_symbol_sent"])
    sent_candidates = list(primary_dispatch["sent_candidates"])

    daily_pick_sent = 0
    daily_summary_sent = 0
    daily_pick_cap = coerce_int(getattr(config, "TELEGRAM_DAILY_BEST_PICK_MAX_PER_DAY", 1), 1)
    if isinstance(alert_budget, dict):
        try:
            daily_pick_cap = max(1, int(alert_budget.get("adjusted_daily_pick_cap") or daily_pick_cap))
        except Exception:
            pass
    if is_daily_best_pick_window():
        daily_candidates = build_daily_best_pick_candidates(results, runtime_context=runtime_context)
        daily_dispatch = dispatch_daily_candidates(
            daily_candidates,
            get_now=get_now,
            send_telegram_alert=send_telegram_alert,
            telegram_alert_cache=telegram_alert_cache,
            record_telegram_alert_history=record_telegram_alert_history,
            limits=limits,
            daily_pick_cap=daily_pick_cap,
            per_symbol_sent=per_symbol_sent,
        )
        daily_pick_sent = int(daily_dispatch["sent"])
        per_symbol_sent = dict(daily_dispatch["per_symbol_sent"])
        sent_candidates.extend(daily_dispatch["sent_candidates"])
        sent += daily_pick_sent
        if not daily_pick_sent:
            daily_summary = build_daily_summary_message(results, existing_candidates=candidates, min_conf=dynamic_min_conf)
            if dispatch_daily_summary(
                daily_summary,
                send_telegram_alert=send_telegram_alert,
                telegram_alert_cache=telegram_alert_cache,
                record_telegram_alert_history=record_telegram_alert_history,
                limits=limits,
            ):
                sent += 1
                daily_summary_sent = 1
            elif not daily_summary_sent:
                logger.info("Daily Best Pick window active but no directional candidate or summary was sent")

    logger.info(
        "Telegram alerts: sent=%s candidates=%s daily_pick=%s daily_summary=%s dropped(cache=%s symbol_cap=%s run_cap=%s quality=%s) min_conf=%.1f dynamic_min_conf=%.1f budget=%s",
        sent,
        len(candidates),
        daily_pick_sent,
        daily_summary_sent,
        dropped_by_cache,
        dropped_by_symbol_cap,
        dropped_by_run_cap,
        json.dumps(quality_drop_counts, ensure_ascii=False),
        min_conf,
        dynamic_min_conf,
        json.dumps(alert_budget or {}, ensure_ascii=False),
    )

    if sent_candidates:
        track_alert_performance(sent_candidates, len(sent_candidates))

    record_telegram_run_report(
        results=results,
        kill=kill,
        kill_reason=reason,
        min_conf=min_conf,
        dynamic_min_conf=dynamic_min_conf,
        candidates=candidates,
        sent_candidates=sent_candidates,
        daily_pick_sent=daily_pick_sent,
        daily_summary_sent=daily_summary_sent,
        dropped_by_cache=dropped_by_cache,
        dropped_by_symbol_cap=dropped_by_symbol_cap,
        dropped_by_run_cap=dropped_by_run_cap,
        quality_drop_counts=quality_drop_counts,
        alert_budget=alert_budget,
    )

    return sent
