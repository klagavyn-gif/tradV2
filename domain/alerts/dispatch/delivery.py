from domain.alerts.dispatch.cache_policy import build_daily_pick_cache_key, cache_contains, cache_mark_sent


def dispatch_primary_candidates(
    candidates,
    *,
    send_telegram_alert,
    telegram_alert_cache,
    record_telegram_alert_history,
    limits,
):
    sent = 0
    dropped_by_cache = 0
    dropped_by_symbol_cap = 0
    dropped_by_run_cap = 0
    per_symbol_sent = {}
    sent_candidates = []

    for candidate in candidates:
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
            per_symbol_sent[symbol] = int(per_symbol_sent.get(symbol, 0)) + 1
            sent += 1
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
    record_telegram_alert_history(
        daily_summary,
        min_conf=limits["min_conf"],
        dynamic_min_conf=limits["dynamic_min_conf"],
        daily_pick=False,
    )
    return True
