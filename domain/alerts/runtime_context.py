def build_alert_runtime_context(results, min_conf, *, config, helpers, get_now):
    telegram_kill_switch_state = helpers["telegram_kill_switch_state"]
    telegram_dynamic_conf_threshold = helpers["telegram_dynamic_conf_threshold"]
    build_regime_context = helpers["build_regime_context"]

    kill, reason = telegram_kill_switch_state(results)
    regime_context = build_regime_context(results or [])
    alert_budget = (regime_context or {}).get("alert_budget") if isinstance(regime_context, dict) else {}
    budget_conf_uplift = 0.0
    if isinstance(alert_budget, dict):
        try:
            budget_conf_uplift = float(alert_budget.get("confidence_uplift") or 0.0)
        except Exception:
            budget_conf_uplift = 0.0
    dynamic_min_conf = float(telegram_dynamic_conf_threshold(min_conf, results)) + float(budget_conf_uplift)
    return {
        "generated_at": get_now().strftime("%Y-%m-%d %H:%M:%S"),
        "kill": bool(kill),
        "kill_reason": str(reason or "") if kill else None,
        "min_confidence": float(min_conf),
        "dynamic_min_confidence": float(dynamic_min_conf),
        "budget_confidence_uplift": float(budget_conf_uplift),
        "regime_context": regime_context if isinstance(regime_context, dict) else {},
        "regime_summary": {
            "enabled": (regime_context or {}).get("enabled"),
            "generated_at": (regime_context or {}).get("generated_at"),
            "market": (regime_context or {}).get("market"),
            "symbols": (regime_context or {}).get("symbols"),
            "by_symbol_regime": (regime_context or {}).get("by_symbol_regime"),
            "by_side_bias": (regime_context or {}).get("by_side_bias"),
            "alert_budget": (regime_context or {}).get("alert_budget"),
        },
        "alert_budget": alert_budget if isinstance(alert_budget, dict) else {},
    }
