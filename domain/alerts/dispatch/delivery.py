from domain.alerts.dispatch.cache_policy import (
    build_daily_pick_cache_key,
    cache_contains,
    cache_mark_sent,
    mark_global_trade_alert_sent,
)

from datetime import datetime


def _parse_candidate_datetime(value):
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def _mark_candidate_sent(candidate, *, get_now):
    if not isinstance(candidate, dict) or not callable(get_now):
        return candidate
    sent_dt = get_now()
    if not isinstance(sent_dt, datetime):
        return candidate
    candidate["telegram_sent_at"] = sent_dt.strftime("%Y-%m-%d %H:%M:%S")
    analysis_dt = _parse_candidate_datetime(candidate.get("analysis_generated_at"))
    signal_dt = _parse_candidate_datetime(candidate.get("signal_timestamp"))
    if isinstance(analysis_dt, datetime):
        candidate["analysis_to_send_seconds"] = max(0.0, (sent_dt - analysis_dt).total_seconds())
    if isinstance(signal_dt, datetime):
        candidate["signal_latency_seconds"] = max(0.0, (sent_dt - signal_dt).total_seconds())
        candidate["signal_age_minutes_at_send"] = float(candidate["signal_latency_seconds"]) / 60.0
    return candidate


def _resolve_trade_budget(limits):
    if not isinstance(limits, dict):
        return None, None
    try:
        remaining = int(limits.get("max_trade_alerts_remaining"))
    except Exception:
        remaining = None
    ttl = limits.get("global_trade_alert_ttl")
    return remaining, ttl


def dispatch_primary_candidates(
    candidates,
    *,
    get_now,
    send_telegram_alert,
    telegram_alert_cache,
    record_telegram_alert_history,
    limits,
):
    sent = 0
    dropped_by_cache = 0
    dropped_by_symbol_cap = 0
    dropped_by_run_cap = 0
    dropped_by_daily_cap = 0
    per_symbol_sent = {}
    sent_candidates = []
    max_trade_remaining, global_trade_ttl = _resolve_trade_budget(limits)

    for candidate in candidates:
        if isinstance(max_trade_remaining, int) and max_trade_remaining >= 0 and sent >= max_trade_remaining:
            dropped_by_daily_cap += 1
            continue
        if sent >= int(limits["max_per_run"]):
            dropped_by_run_cap += 1
            continue
        symbol = str(candidate.get("symbol") or "")
        if not symbol:
            continue
        if int(per_symbol_sent.get(symbol, 0)) >= int(limits["max_per_symbol"]):
            dropped_by_symbol_cap += 1
            continue
        cache_key = str(candidate.get("cache_key") or "").strip()
        if not cache_key:
            continue
        if cache_contains(telegram_alert_cache, cache_key):
            dropped_by_cache += 1
            continue
        message = candidate.get("message")
        if not isinstance(message, str) or not message.strip():
            continue
        if send_telegram_alert(message):
            cache_mark_sent(telegram_alert_cache, cache_key, ttl_seconds=limits["cooldown_ttl"])
            if isinstance(max_trade_remaining, int) and max_trade_remaining >= 0:
                mark_global_trade_alert_sent(
                    telegram_alert_cache,
                    get_now,
                    ttl_seconds=global_trade_ttl,
                )
            per_symbol_sent[symbol] = int(per_symbol_sent.get(symbol, 0)) + 1
            sent += 1
            _mark_candidate_sent(candidate, get_now=get_now)
            sent_candidates.append(candidate)
            record_telegram_alert_history(
                candidate,
                min_conf=limits["min_conf"],
                dynamic_min_conf=limits["dynamic_min_conf"],
                daily_pick=False,
            )

    return {
        "sent": sent,
        "sent_candidates": sent_candidates,
        "per_symbol_sent": per_symbol_sent,
        "dropped_by_cache": dropped_by_cache,
        "dropped_by_symbol_cap": dropped_by_symbol_cap,
        "dropped_by_run_cap": dropped_by_run_cap,
        "dropped_by_daily_cap": dropped_by_daily_cap,
    }




def dispatch_daily_candidates(
    daily_candidates,
    *,
    get_now,
    send_telegram_alert,
    telegram_alert_cache,
    record_telegram_alert_history,
    limits,
    daily_pick_cap,
    per_symbol_sent,
):
    sent = 0
    sent_candidates = []
    for daily_candidate in daily_candidates:
        if not isinstance(daily_candidate, dict):
            continue
        if sent >= int(daily_pick_cap):
            break
        daily_key = build_daily_pick_cache_key(get_now, daily_candidate)
        if cache_contains(telegram_alert_cache, daily_key):
            continue
        daily_symbol = str(daily_candidate.get("symbol") or "")
        if daily_symbol and int(per_symbol_sent.get(daily_symbol, 0)) >= int(limits["max_per_symbol"]):
            continue
        daily_message = daily_candidate.get("message")
        if isinstance(daily_message, str) and daily_message.strip() and send_telegram_alert(daily_message):
            cache_mark_sent(telegram_alert_cache, daily_key, ttl_seconds=26 * 60 * 60)
            sent += 1
            _mark_candidate_sent(daily_candidate, get_now=get_now)
            sent_candidates.append(daily_candidate)
            if daily_symbol:
                per_symbol_sent[daily_symbol] = int(per_symbol_sent.get(daily_symbol, 0)) + 1
            record_telegram_alert_history(
                daily_candidate,
                min_conf=limits["min_conf"],
                dynamic_min_conf=limits["dynamic_min_conf"],
                daily_pick=True,
            )
    return {
        "sent": sent,
        "sent_candidates": sent_candidates,
        "per_symbol_sent": per_symbol_sent,
    }


def dispatch_daily_summary(
    daily_summary,
    *,
    get_now,
    send_telegram_alert,
    telegram_alert_cache,
    record_telegram_alert_history,
    limits,
):
    daily_message = daily_summary.get("message") if isinstance(daily_summary, dict) else None
    daily_key = daily_summary.get("cache_key") if isinstance(daily_summary, dict) else None
    if not isinstance(daily_summary, dict):
        return False
    if not isinstance(daily_key, str) or not daily_key.strip():
        return False
    if cache_contains(telegram_alert_cache, daily_key):
        return False
    if not isinstance(daily_message, str) or not daily_message.strip():
        return False
    if not send_telegram_alert(daily_message):
        return False
    cache_mark_sent(telegram_alert_cache, daily_key, ttl_seconds=26 * 60 * 60)
    _mark_candidate_sent(daily_summary, get_now=get_now)
    record_telegram_alert_history(
        daily_summary,
        min_conf=limits["min_conf"],
        dynamic_min_conf=limits["dynamic_min_conf"],
        daily_pick=False,
    )
    return True


def dispatch_trend_state_candidates(
    trend_state_candidates,
    *,
    get_now,
    send_telegram_alert,
    telegram_alert_cache,
    record_telegram_alert_history,
    min_conf,
    dynamic_min_conf,
    cooldown_ttl,
    max_per_run,
    per_symbol_sent,
    suppress_if_symbol_sent,
    limits=None,
):
    sent = 0
    dropped_by_cache = 0
    dropped_by_symbol_cap = 0
    dropped_by_run_cap = 0
    dropped_by_daily_cap = 0
    sent_candidates = []
    max_trade_remaining, global_trade_ttl = _resolve_trade_budget(limits)

    for candidate in trend_state_candidates:
        if not isinstance(candidate, dict):
            continue
        if isinstance(max_trade_remaining, int) and max_trade_remaining >= 0 and sent >= max_trade_remaining:
            dropped_by_daily_cap += 1
            continue
        if sent >= int(max_per_run):
            dropped_by_run_cap += 1
            continue
        symbol = str(candidate.get("symbol") or "")
        if symbol and suppress_if_symbol_sent and int(per_symbol_sent.get(symbol, 0)) > 0:
            dropped_by_symbol_cap += 1
            continue
        cache_key = str(candidate.get("cache_key") or "").strip()
        if not cache_key:
            continue
        if cache_contains(telegram_alert_cache, cache_key):
            dropped_by_cache += 1
            continue
        message = candidate.get("message")
        if not isinstance(message, str) or not message.strip():
            continue
        if not send_telegram_alert(message):
            continue
        cache_mark_sent(telegram_alert_cache, cache_key, ttl_seconds=int(cooldown_ttl))
        if isinstance(max_trade_remaining, int) and max_trade_remaining >= 0:
            mark_global_trade_alert_sent(
                telegram_alert_cache,
                get_now,
                ttl_seconds=global_trade_ttl,
            )
        if symbol:
            per_symbol_sent[symbol] = int(per_symbol_sent.get(symbol, 0)) + 1
        sent += 1
        _mark_candidate_sent(candidate, get_now=get_now)
        sent_candidates.append(candidate)
        record_telegram_alert_history(
            candidate,
            min_conf=min_conf,
            dynamic_min_conf=dynamic_min_conf,
            daily_pick=False,
        )

    return {
        "sent": sent,
        "sent_candidates": sent_candidates,
        "per_symbol_sent": per_symbol_sent,
        "dropped_by_cache": dropped_by_cache,
        "dropped_by_symbol_cap": dropped_by_symbol_cap,
        "dropped_by_run_cap": dropped_by_run_cap,
        "dropped_by_daily_cap": dropped_by_daily_cap,
    }


def dispatch_trend_radar_candidates(
    trend_radar_candidates,
    *,
    get_now,
    send_telegram_alert,
    telegram_alert_cache,
    record_telegram_alert_history,
    min_conf,
    dynamic_min_conf,
    cooldown_ttl,
    max_per_run,
    per_symbol_sent,
    suppress_if_symbol_sent,
    max_total_per_symbol,
    limits=None,
):
    sent = 0
    dropped_by_cache = 0
    dropped_by_symbol_cap = 0
    dropped_by_run_cap = 0
    dropped_by_daily_cap = 0
    sent_candidates = []
    max_trade_remaining, global_trade_ttl = _resolve_trade_budget(limits)

    for candidate in trend_radar_candidates:
        if not isinstance(candidate, dict):
            continue
        if isinstance(max_trade_remaining, int) and max_trade_remaining >= 0 and sent >= max_trade_remaining:
            dropped_by_daily_cap += 1
            continue
        if sent >= int(max_per_run):
            dropped_by_run_cap += 1
            continue
        symbol = str(candidate.get("symbol") or "")
        existing_symbol_alerts = int(per_symbol_sent.get(symbol, 0)) if symbol else 0
        effective_symbol_cap = 1 if suppress_if_symbol_sent else max(1, int(max_total_per_symbol))
        if symbol and existing_symbol_alerts >= effective_symbol_cap:
            dropped_by_symbol_cap += 1
            continue
        cache_key = str(candidate.get("cache_key") or "").strip()
        if not cache_key:
            continue
        if cache_contains(telegram_alert_cache, cache_key):
            dropped_by_cache += 1
            continue
        message = candidate.get("message")
        if not isinstance(message, str) or not message.strip():
            continue
        if not send_telegram_alert(message):
            continue
        cache_mark_sent(telegram_alert_cache, cache_key, ttl_seconds=int(cooldown_ttl))
        if isinstance(max_trade_remaining, int) and max_trade_remaining >= 0:
            mark_global_trade_alert_sent(
                telegram_alert_cache,
                get_now,
                ttl_seconds=global_trade_ttl,
            )
        if symbol:
            per_symbol_sent[symbol] = existing_symbol_alerts + 1
        sent += 1
        _mark_candidate_sent(candidate, get_now=get_now)
        sent_candidates.append(candidate)
        record_telegram_alert_history(
            candidate,
            min_conf=min_conf,
            dynamic_min_conf=dynamic_min_conf,
            daily_pick=False,
        )

    return {
        "sent": sent,
        "sent_candidates": sent_candidates,
        "per_symbol_sent": per_symbol_sent,
        "dropped_by_cache": dropped_by_cache,
        "dropped_by_symbol_cap": dropped_by_symbol_cap,
        "dropped_by_run_cap": dropped_by_run_cap,
        "dropped_by_daily_cap": dropped_by_daily_cap,
    }
