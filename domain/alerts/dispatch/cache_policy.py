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
