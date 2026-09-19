def cache_contains(cache, cache_key):
    key = str(cache_key or "").strip()
    if not key:
        return False
    return bool(cache.get(key))


def cache_mark_sent(cache, cache_key, *, ttl_seconds):
    key = str(cache_key or "").strip()
    if not key:
        return
    cache.set(key, True, ttl_seconds=int(ttl_seconds))


def build_daily_pick_cache_key(get_now, candidate):
    return f"DAILYBEST|{get_now().strftime('%Y%m%d')}|{candidate.get('symbol')}|{candidate.get('signal')}"


def build_symbol_intent_key(candidate):
    """Return a stable cooldown key for intent class + symbol + side.

    The regular cache key embeds the signal timestamp, so a rolling signal
    produces a new key every bar and bypasses the cooldown. This key ignores
    the timestamp so the same symbol and side cannot be re-alerted inside the
    symbol cooldown window. Intent classes are separated so a watch alert does
    not block a later confirmed entry for the same symbol.
    """
    if not isinstance(candidate, dict):
        return ""
    symbol = str(candidate.get("symbol") or "").strip().upper()
    if not symbol:
        return ""
    signal = str(candidate.get("signal") or "").strip().upper()
    intent = str(candidate.get("alert_intent") or "").strip().lower()
    if intent == "watch":
        intent_class = "watch"
    elif intent == "entry":
        intent_class = "entry"
    elif intent == "exit":
        intent_class = "exit"
    else:
        intent_class = "other"
    return f"{intent_class}|{symbol}|{signal}"


def build_global_trade_alert_cache_key(get_now):
    return f"GLOBALTRADE|{get_now().strftime('%Y%m%d')}"


def global_trade_alert_ttl_seconds():
    return 26 * 60 * 60


def get_global_trade_alerts_sent(cache, get_now):
    key = build_global_trade_alert_cache_key(get_now)
    value = cache.get(key)
    try:
        count = int(value)
    except Exception:
        count = 0
    return max(0, count)


def mark_global_trade_alert_sent(cache, get_now, *, ttl_seconds=None):
    key = build_global_trade_alert_cache_key(get_now)
    count = get_global_trade_alerts_sent(cache, get_now) + 1
    ttl = int(global_trade_alert_ttl_seconds() if ttl_seconds is None else ttl_seconds)
    cache.set(key, count, ttl_seconds=ttl)
    return count
