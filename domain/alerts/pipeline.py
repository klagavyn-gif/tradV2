import json

from domain.alerts.dispatch.delivery import (
    dispatch_daily_candidates,
    dispatch_daily_summary,
    dispatch_primary_candidates,
    dispatch_trend_radar_candidates,
    dispatch_trend_state_candidates,
)
from domain.alerts.dispatch.throttling import coerce_float, coerce_int, resolve_dispatch_settings


def _primary_candidate_sort_key(candidate):
    if not isinstance(candidate, dict):
        return (0.0, 0.0)
    try:
        score = float(candidate.get("score", 0.0))
    except Exception:
        score = 0.0
    try:
        confidence = float(candidate.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    return (score, confidence)


def _tier_meets_minimum(tier, minimum):
    order = {"A": 4, "B": 3, "C": 2, "D": 1}
    return order.get(str(tier or "").upper(), 0) >= order.get(str(minimum or "").upper(), 0)


def _is_preferred_primary_candidate(candidate, *, config, limits):
    if not isinstance(candidate, dict):
        return False
    preferred = getattr(config, "TELEGRAM_ALERT_PRIMARY_DIVERSITY_STRATEGIES", {"PRIMARY", "SS15"})
    strategy = str(candidate.get("strategy") or "").strip().upper()
    if strategy not in preferred:
        return False
    profile = candidate.get("alert_profile")
    if not isinstance(profile, dict):
        return False
    min_tier = getattr(config, "TELEGRAM_ALERT_PRIMARY_DIVERSITY_MIN_TIER", "B")
    tier = str(profile.get("tier") or "").strip().upper()
    if not _tier_meets_minimum(tier, min_tier):
        return False
    try:
        min_profile_score = float(getattr(config, "TELEGRAM_ALERT_PRIMARY_DIVERSITY_MIN_COMPOSITE_SCORE", 80.0))
    except Exception:
        min_profile_score = 80.0
    try:
        profile_score = float(profile.get("composite_score") or 0.0)
    except Exception:
        profile_score = 0.0
    if profile_score < min_profile_score:
        return False
    try:
        confidence = float(candidate.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    try:
        min_confidence = float(getattr(config, "TELEGRAM_ALERT_PRIMARY_DIVERSITY_MIN_CONFIDENCE", 78.0))
    except Exception:
        min_confidence = 78.0
    baseline_confidence = max(float(limits.get("min_conf") or 0.0), float(min_confidence))
    if confidence < baseline_confidence:
        return False
    return True


def _rebalance_primary_candidates(candidates, *, config, limits):
    if not isinstance(candidates, list) or len(candidates) <= 1:
        return candidates
    if not bool(getattr(config, "TELEGRAM_ALERT_PRIMARY_DIVERSITY_ENABLE", True)):
        return candidates
    max_per_run = max(1, int(limits.get("max_per_run") or 1))
    if max_per_run < 2:
        return candidates
    preferred = getattr(config, "TELEGRAM_ALERT_PRIMARY_DIVERSITY_STRATEGIES", {"PRIMARY", "SS15"})
    if any(str(row.get("strategy") or "").strip().upper() in preferred for row in candidates[:max_per_run] if isinstance(row, dict)):
        return candidates
    insert_at = min(max_per_run - 1, len(candidates) - 1)
    reserved_symbols = {
        str(row.get("symbol") or "").strip().upper()
        for row in candidates[:insert_at]
        if isinstance(row, dict) and str(row.get("symbol") or "").strip()
    }
    selected_index = None
    for idx, row in enumerate(candidates):
        if idx < max_per_run and not _is_preferred_primary_candidate(row, config=config, limits=limits):
            continue
        if not _is_preferred_primary_candidate(row, config=config, limits=limits):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol and symbol in reserved_symbols:
            continue
        selected_index = idx
        break
    if selected_index is None:
        return candidates
    reordered = list(candidates)
    selected = reordered.pop(selected_index)
    reordered.insert(insert_at, selected)
    return reordered


def notify_telegram_from_results(results, *, config, helpers, get_now, logger, runtime_context=None):
    build_alert_runtime_context = helpers["build_alert_runtime_context"]
    build_telegram_candidates = helpers["build_telegram_candidates"]
    build_trend_radar_candidates = helpers["build_trend_radar_candidates"]
    build_trend_state_candidates = helpers["build_trend_state_candidates"]
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

    candidates.sort(key=_primary_candidate_sort_key, reverse=True)
    candidates = _rebalance_primary_candidates(candidates, config=config, limits=limits)
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
    trend_radar_sent = 0
    trend_state_sent = 0
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

    trend_radar_candidates = []
    if not kill:
        trend_radar_candidates = build_trend_radar_candidates(results, runtime_context=runtime_context)
    if trend_radar_candidates:
        trend_radar_max_per_run = coerce_int(getattr(config, "TREND_RADAR_MAX_PER_RUN", 2), 2)
        trend_radar_cooldown_minutes = coerce_int(getattr(config, "TREND_RADAR_COOLDOWN_MINUTES", 240), 240)
        trend_radar_max_total_per_symbol = coerce_int(getattr(config, "TREND_RADAR_MAX_TOTAL_PER_SYMBOL", 1), 1)
        trend_radar_dispatch = dispatch_trend_radar_candidates(
            trend_radar_candidates,
            send_telegram_alert=send_telegram_alert,
            telegram_alert_cache=telegram_alert_cache,
            record_telegram_alert_history=record_telegram_alert_history,
            min_conf=min_conf,
            dynamic_min_conf=dynamic_min_conf,
            cooldown_ttl=max(60, int(trend_radar_cooldown_minutes * 60)),
            max_per_run=trend_radar_max_per_run,
            per_symbol_sent=per_symbol_sent,
            suppress_if_symbol_sent=bool(getattr(config, "TREND_RADAR_SUPPRESS_IF_PRIMARY_SENT", True)),
            max_total_per_symbol=trend_radar_max_total_per_symbol,
        )
        trend_radar_sent = int(trend_radar_dispatch["sent"])
        per_symbol_sent = dict(trend_radar_dispatch["per_symbol_sent"])
        sent_candidates.extend(trend_radar_dispatch["sent_candidates"])
        sent += trend_radar_sent

    trend_state_candidates = []
    if not kill:
        trend_state_candidates = build_trend_state_candidates(results, runtime_context=runtime_context)
    if trend_state_candidates:
        trend_state_max_per_run = coerce_int(getattr(config, "TREND_STATE_ALERT_MAX_PER_RUN", 2), 2)
        trend_state_cooldown_minutes = coerce_int(getattr(config, "TREND_STATE_ALERT_COOLDOWN_MINUTES", 360), 360)
        trend_state_dispatch = dispatch_trend_state_candidates(
            trend_state_candidates,
            send_telegram_alert=send_telegram_alert,
            telegram_alert_cache=telegram_alert_cache,
            record_telegram_alert_history=record_telegram_alert_history,
            min_conf=min_conf,
            dynamic_min_conf=dynamic_min_conf,
            cooldown_ttl=max(60, int(trend_state_cooldown_minutes * 60)),
            max_per_run=trend_state_max_per_run,
            per_symbol_sent=per_symbol_sent,
            suppress_if_symbol_sent=bool(getattr(config, "TREND_STATE_ALERT_SUPPRESS_IF_PRIMARY_SENT", True)),
        )
        trend_state_sent = int(trend_state_dispatch["sent"])
        per_symbol_sent = dict(trend_state_dispatch["per_symbol_sent"])
        sent_candidates.extend(trend_state_dispatch["sent_candidates"])
        sent += trend_state_sent

    logger.info(
        "Telegram alerts: sent=%s candidates=%s daily_pick=%s daily_summary=%s trend_radar=%s trend_state=%s dropped(cache=%s symbol_cap=%s run_cap=%s quality=%s) min_conf=%.1f dynamic_min_conf=%.1f budget=%s",
        sent,
        len(candidates),
        daily_pick_sent,
        daily_summary_sent,
        trend_radar_sent,
        trend_state_sent,
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
