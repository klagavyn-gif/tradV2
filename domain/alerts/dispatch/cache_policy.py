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
