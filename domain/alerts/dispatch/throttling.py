def coerce_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def coerce_int(value, default, *, minimum=1):
    try:
        number = int(value)
    except Exception:
        number = int(default)
    return max(int(minimum), number)


def resolve_dispatch_settings(config, runtime_context):
    min_conf = coerce_float(getattr(config, "TELEGRAM_ALERT_MIN_CONFIDENCE", 69.0), 69.0)
    try:
        min_conf = coerce_float((runtime_context or {}).get("min_confidence"), min_conf)
    except Exception:
        pass
    max_per_run = coerce_int(getattr(config, "TELEGRAM_ALERT_MAX_PER_RUN", 5), 5)
    max_per_symbol = coerce_int(getattr(config, "TELEGRAM_ALERT_MAX_PER_SYMBOL", 1), 1)
    cooldown_minutes = coerce_int(getattr(config, "TELEGRAM_ALERT_COOLDOWN_MINUTES", 30), 30)
    alert_budget = (runtime_context or {}).get("alert_budget") or {}
    if isinstance(alert_budget, dict):
        try:
            max_per_run = max(1, int(alert_budget.get("adjusted_run_cap") or max_per_run))
        except Exception:
            pass
        try:
            max_per_symbol = max(1, int(alert_budget.get("adjusted_per_symbol_cap") or max_per_symbol))
        except Exception:
            pass
    dynamic_min_conf = float((runtime_context or {}).get("dynamic_min_confidence") or float(min_conf))
    return {
        "min_conf": min_conf,
        "dynamic_min_conf": dynamic_min_conf,
        "max_per_run": max_per_run,
        "max_per_symbol": max_per_symbol,
        "cooldown_minutes": cooldown_minutes,
        "cooldown_ttl": max(60, int(cooldown_minutes * 60)),
        "alert_budget": alert_budget if isinstance(alert_budget, dict) else {},
    }
