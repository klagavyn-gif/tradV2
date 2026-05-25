import math
from collections import Counter


_STRATEGY_MULTIPLIERS = {
    "TREND_UP": {"PRIMARY": 1.08, "SS15": 1.08, "AW15": 1.04, "AZ15": 1.14, "TCB15": 1.12, "PA15": 0.96, "CDCVIX15": 0.92},
    "TREND_DOWN": {"PRIMARY": 1.08, "SS15": 1.08, "AW15": 1.04, "AZ15": 1.14, "TCB15": 1.12, "PA15": 0.96, "CDCVIX15": 0.94},
    "BREAKOUT_EXPANSION": {"PRIMARY": 1.10, "SS15": 1.07, "AW15": 1.05, "AZ15": 1.10, "TCB15": 1.16, "PA15": 0.94, "CDCVIX15": 0.90},
    "PANIC_REVERSAL": {"PRIMARY": 1.02, "SS15": 0.98, "AW15": 1.00, "AZ15": 0.92, "TCB15": 0.86, "PA15": 1.00, "CDCVIX15": 1.18},
    "RANGE_BALANCED": {"PRIMARY": 1.00, "SS15": 0.98, "AW15": 1.02, "AZ15": 0.95, "TCB15": 0.92, "PA15": 1.06, "CDCVIX15": 1.01},
    "RANGE_HIGH_VOL": {"PRIMARY": 0.98, "SS15": 0.96, "AW15": 1.02, "AZ15": 0.94, "TCB15": 0.88, "PA15": 1.02, "CDCVIX15": 1.08},
    "LOW_LIQUIDITY_CHOP": {"PRIMARY": 0.90, "SS15": 0.90, "AW15": 0.96, "AZ15": 0.84, "TCB15": 0.78, "PA15": 1.03, "CDCVIX15": 0.96},
    "RISK_OFF_EVENT": {"PRIMARY": 0.92, "SS15": 0.88, "AW15": 0.94, "AZ15": 0.82, "TCB15": 0.76, "PA15": 0.92, "CDCVIX15": 1.10},
}

_BLOCKED_BY_SYMBOL_REGIME = {
    "LOW_LIQUIDITY_CHOP": {"TCB15"},
    "RISK_OFF_EVENT": {"TCB15"},
}

_MARKET_ALERT_BUDGETS = {
    "TREND_UP": {"run_cap": 3, "per_symbol_cap": 1, "daily_pick_cap": 1, "confidence_uplift": 0.0},
    "TREND_DOWN": {"run_cap": 3, "per_symbol_cap": 1, "daily_pick_cap": 1, "confidence_uplift": 0.0},
    "BREAKOUT_EXPANSION": {"run_cap": 3, "per_symbol_cap": 1, "daily_pick_cap": 1, "confidence_uplift": 0.0},
    "PANIC_REVERSAL": {"run_cap": 2, "per_symbol_cap": 1, "daily_pick_cap": 1, "confidence_uplift": 1.0},
    "RANGE_BALANCED": {"run_cap": 3, "per_symbol_cap": 1, "daily_pick_cap": 1, "confidence_uplift": 1.0},
    "RANGE_HIGH_VOL": {"run_cap": 1, "per_symbol_cap": 1, "daily_pick_cap": 1, "confidence_uplift": 2.0},
    "LOW_LIQUIDITY_CHOP": {"run_cap": 1, "per_symbol_cap": 1, "daily_pick_cap": 1, "confidence_uplift": 3.0},
    "RISK_OFF_EVENT": {"run_cap": 1, "per_symbol_cap": 1, "daily_pick_cap": 1, "confidence_uplift": 4.0},
}


def _safe_float(value, default=None):
    try:
        value = float(value)
    except Exception:
        return default
    if not math.isfinite(value):
        return default
    return value


def _clip(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _median(values, default=0.0):
    values = [float(v) for v in (values or []) if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not values:
        return float(default)
    values.sort()
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _normalize_signal(value):
    signal = str(value or "").strip().upper()
    return signal if signal in ("BUY", "SELL") else None


def _extract_signal_strength(plan, *, default_weight):
    if not isinstance(plan, dict):
        return None, None
    signal = _normalize_signal(plan.get("signal"))
    if not signal:
        return None, None
    confidence = _safe_float(plan.get("confidence"), None)
    score = _safe_float(plan.get("score"), confidence)
    if score is None and confidence is None:
        strength = float(default_weight)
    else:
        base = max(v for v in (confidence, score, 50.0) if isinstance(v, (int, float)))
        strength = float(base) * float(default_weight)
    return signal, strength


def _feature_snapshot(item):
    symbol = str((item or {}).get("symbol") or "").strip().upper()
    az_plan = (item or {}).get("actionzone_15m") if isinstance(item, dict) else None
    ema_plan = (item or {}).get("ema_cross_15m") if isinstance(item, dict) else None
    cdc_plan = (item or {}).get("cdc_vixfix_15m") if isinstance(item, dict) else None
    pa_plan = (item or {}).get("price_action_15m") if isinstance(item, dict) else None
    tcb_plan = (item or {}).get("trend_breakout_15m") if isinstance(item, dict) else None

    vol_samples = []
    for plan, key in (
        (az_plan, "avg_range_pct"),
        (az_plan, "atr_pct"),
        (cdc_plan, "atr_pct"),
        (tcb_plan, "atr_pct"),
    ):
        if isinstance(plan, dict):
            value = _safe_float(plan.get(key), None)
            if isinstance(value, float) and value > 0:
                vol_samples.append(value)
    change_pct = _safe_float((item or {}).get("change"), None)
    if isinstance(change_pct, float):
        vol_samples.append(abs(change_pct))
    volatility_pct = _median(vol_samples, default=0.0)

    trend_score = 0.0
    adx = _safe_float((az_plan or {}).get("adx"), None)
    if bool((az_plan or {}).get("trend_alignment", False)):
        trend_score += 24.0
    if isinstance(adx, float):
        trend_score += _clip((adx - 14.0) * 2.4, 0.0, 28.0)
    strength = str((az_plan or {}).get("trend_strength") or "").upper()
    if strength == "STRONG":
        trend_score += 15.0
    elif strength == "MODERATE":
        trend_score += 8.0
    ema_signal = _normalize_signal((ema_plan or {}).get("signal"))
    if ema_signal:
        trend_score += 10.0
    bars_since_cross = _safe_float((ema_plan or {}).get("bars_since_cross"), None)
    if isinstance(bars_since_cross, float) and bars_since_cross <= 2.0:
        trend_score += 6.0

    expansion_score = 0.0
    tcb_score = _safe_float((tcb_plan or {}).get("score"), None)
    if isinstance(tcb_score, float):
        expansion_score += _clip((tcb_score - 50.0) * 0.75, 0.0, 28.0)
    if isinstance(volatility_pct, float):
        expansion_score += _clip((volatility_pct - 1.0) * 12.0, 0.0, 28.0)
    trend_1h = str((tcb_plan or {}).get("trend_1h") or "").upper()
    if trend_1h in ("UP", "DOWN", "BULLISH", "BEARISH"):
        expansion_score += 10.0
    rvol = _safe_float((tcb_plan or {}).get("rvol"), None)
    if isinstance(rvol, float):
        expansion_score += _clip((rvol - 1.0) * 12.0, 0.0, 18.0)

    reversal_score = 0.0
    red_green = _safe_float((cdc_plan or {}).get("red_to_green_quality_score"), None)
    if isinstance(red_green, float):
        reversal_score += _clip((red_green - 55.0) * 0.80, 0.0, 30.0)
    forecast_score = _safe_float((cdc_plan or {}).get("forecast_score"), None)
    if isinstance(forecast_score, float):
        reversal_score += _clip((forecast_score - 50.0) * 0.35, 0.0, 18.0)
    if bool((cdc_plan or {}).get("green_flip_reclaim")):
        reversal_score += 12.0
    if _normalize_signal((cdc_plan or {}).get("signal")):
        reversal_score += 8.0
    if isinstance(volatility_pct, float):
        reversal_score += _clip((volatility_pct - 1.4) * 8.0, 0.0, 12.0)

    chop_score = 0.0
    if isinstance(adx, float):
        chop_score += _clip((22.0 - adx) * 2.4, 0.0, 28.0)
    if not bool((az_plan or {}).get("trend_alignment", False)):
        chop_score += 16.0
    market_structure = str((pa_plan or {}).get("market_structure") or "").upper()
    if market_structure in ("RANGE", "SIDEWAYS", "CONSOLIDATION"):
        chop_score += 12.0
    if isinstance(volatility_pct, float) and volatility_pct < 1.0:
        chop_score += _clip((1.0 - volatility_pct) * 18.0, 0.0, 12.0)

    buy_strength = 0.0
    sell_strength = 0.0
    for plan, weight in (
        (az_plan, 1.00),
        (ema_plan, 0.60),
        (cdc_plan, 0.85),
        (pa_plan, 0.90),
        (tcb_plan, 1.05),
    ):
        signal, strength_value = _extract_signal_strength(plan, default_weight=weight)
        if signal == "BUY":
            buy_strength += float(strength_value or 0.0)
        elif signal == "SELL":
            sell_strength += float(strength_value or 0.0)
    side_gap = float(buy_strength - sell_strength)
    side_bias = "NEUTRAL"
    if side_gap >= 18.0:
        side_bias = "BUY"
    elif side_gap <= -18.0:
        side_bias = "SELL"

    return {
        "symbol": symbol,
        "volatility_pct": float(volatility_pct),
        "adx": adx,
        "trend_score": _clip(trend_score, 0.0, 100.0),
        "expansion_score": _clip(expansion_score, 0.0, 100.0),
        "reversal_score": _clip(reversal_score, 0.0, 100.0),
        "chop_score": _clip(chop_score, 0.0, 100.0),
        "buy_strength": float(buy_strength),
        "sell_strength": float(sell_strength),
        "side_gap": float(side_gap),
        "side_bias": side_bias,
    }


def _market_regime_from_snapshot(snapshot, *, config):
    total_symbols = int(snapshot.get("symbols_analyzed") or 0)
    if total_symbols <= 0:
        return "RANGE_BALANCED", "NEUTRAL", 0.0
    median_volatility = _safe_float(snapshot.get("median_volatility_pct"), 0.0)
    median_trend = _safe_float(snapshot.get("median_trend_score"), 0.0)
    median_expansion = _safe_float(snapshot.get("median_expansion_score"), 0.0)
    median_reversal = _safe_float(snapshot.get("median_reversal_score"), 0.0)
    median_chop = _safe_float(snapshot.get("median_chop_score"), 0.0)
    buy_ratio = _safe_float(snapshot.get("buy_bias_ratio"), 0.0)
    sell_ratio = _safe_float(snapshot.get("sell_bias_ratio"), 0.0)
    abs_change = _safe_float(snapshot.get("median_abs_change_pct"), 0.0)
    high_vol_threshold = _safe_float(getattr(config, "ALL_WEATHER_15M_HIGH_VOL_THRESHOLD", 2.2), 2.2)

    if median_volatility >= max(3.0, high_vol_threshold + 0.8) and sell_ratio >= 0.55 and abs_change >= 2.0:
        return "RISK_OFF_EVENT", "SELL", _clip(58.0 + median_volatility * 6.0 + sell_ratio * 18.0, 0.0, 100.0)
    if median_reversal >= 66.0 and median_volatility >= max(1.8, high_vol_threshold - 0.2):
        return "PANIC_REVERSAL", "NEUTRAL", _clip(52.0 + (median_reversal - 60.0) * 1.5, 0.0, 100.0)
    if median_expansion >= 62.0 and max(buy_ratio, sell_ratio) >= 0.42:
        side_bias = "BUY" if buy_ratio >= sell_ratio else "SELL"
        return "BREAKOUT_EXPANSION", side_bias, _clip(50.0 + median_expansion * 0.55, 0.0, 100.0)
    if median_chop >= 58.0 and median_trend < 58.0:
        return "LOW_LIQUIDITY_CHOP", "NEUTRAL", _clip(48.0 + median_chop * 0.55, 0.0, 100.0)
    if median_trend >= 62.0 and buy_ratio >= 0.44:
        return "TREND_UP", "BUY", _clip(50.0 + median_trend * 0.55 + buy_ratio * 10.0, 0.0, 100.0)
    if median_trend >= 62.0 and sell_ratio >= 0.44:
        return "TREND_DOWN", "SELL", _clip(50.0 + median_trend * 0.55 + sell_ratio * 10.0, 0.0, 100.0)
    if median_volatility >= high_vol_threshold:
        return "RANGE_HIGH_VOL", "NEUTRAL", _clip(45.0 + median_volatility * 8.0, 0.0, 100.0)
    return "RANGE_BALANCED", "NEUTRAL", _clip(44.0 + (50.0 - abs(median_trend - 50.0)) * 0.35, 0.0, 100.0)


def build_market_regime_snapshot(results, *, config, helpers):
    enabled = bool(getattr(config, "TELEGRAM_ALERT_REGIME_ENABLED", True))
    normalize_symbol = helpers["normalize_symbol"]
    symbol_features = []
    abs_changes = []
    for item in results or []:
        if not isinstance(item, dict) or item.get("error"):
            continue
        feature = _feature_snapshot(item)
        feature["symbol"] = normalize_symbol(feature.get("symbol") or "")
        if not feature["symbol"]:
            continue
        symbol_features.append(feature)
        change_pct = _safe_float(item.get("change"), None)
        if isinstance(change_pct, float):
            abs_changes.append(abs(change_pct))

    total = len(symbol_features)
    buy_count = sum(1 for row in symbol_features if row.get("side_bias") == "BUY")
    sell_count = sum(1 for row in symbol_features if row.get("side_bias") == "SELL")
    snapshot = {
        "enabled": enabled,
        "generated_symbols": total,
        "symbols_analyzed": total,
        "buy_bias_ratio": (float(buy_count) / float(total)) if total else 0.0,
        "sell_bias_ratio": (float(sell_count) / float(total)) if total else 0.0,
        "median_volatility_pct": _median([row.get("volatility_pct") for row in symbol_features], default=0.0),
        "median_trend_score": _median([row.get("trend_score") for row in symbol_features], default=0.0),
        "median_expansion_score": _median([row.get("expansion_score") for row in symbol_features], default=0.0),
        "median_reversal_score": _median([row.get("reversal_score") for row in symbol_features], default=0.0),
        "median_chop_score": _median([row.get("chop_score") for row in symbol_features], default=0.0),
        "median_abs_change_pct": _median(abs_changes, default=0.0),
        "symbol_features": symbol_features,
    }
    market_regime, side_bias, confidence = _market_regime_from_snapshot(snapshot, config=config)
    snapshot["market_regime"] = market_regime
    snapshot["side_bias"] = side_bias
    snapshot["regime_confidence"] = float(confidence)
    return snapshot


def _symbol_regime_from_features(feature, market_snapshot):
    trend_score = _safe_float(feature.get("trend_score"), 0.0)
    expansion_score = _safe_float(feature.get("expansion_score"), 0.0)
    reversal_score = _safe_float(feature.get("reversal_score"), 0.0)
    chop_score = _safe_float(feature.get("chop_score"), 0.0)
    volatility_pct = _safe_float(feature.get("volatility_pct"), 0.0)
    side_bias = str(feature.get("side_bias") or "NEUTRAL").upper()
    market_regime = str((market_snapshot or {}).get("market_regime") or "RANGE_BALANCED").upper()

    if market_regime == "RISK_OFF_EVENT":
        return "RISK_OFF_EVENT"
    if reversal_score >= 68.0 and volatility_pct >= 1.8:
        return "PANIC_REVERSAL"
    if expansion_score >= 64.0 and trend_score >= 52.0:
        return "BREAKOUT_EXPANSION"
    if chop_score >= 60.0:
        return "LOW_LIQUIDITY_CHOP"
    if trend_score >= 62.0 and side_bias == "BUY":
        return "TREND_UP"
    if trend_score >= 62.0 and side_bias == "SELL":
        return "TREND_DOWN"
    if volatility_pct >= max(2.0, _safe_float((market_snapshot or {}).get("median_volatility_pct"), 0.0) + 0.2):
        return "RANGE_HIGH_VOL"
    return "RANGE_BALANCED"


def build_symbol_regime(item, *, market_snapshot, config, helpers):
    enabled = bool(getattr(config, "TELEGRAM_ALERT_REGIME_ENABLED", True))
    normalize_symbol = helpers["normalize_symbol"]
    feature = _feature_snapshot(item or {})
    symbol = normalize_symbol(feature.get("symbol") or "")
    feature["symbol"] = symbol
    symbol_regime = _symbol_regime_from_features(feature, market_snapshot or {})
    market_regime = str((market_snapshot or {}).get("market_regime") or "RANGE_BALANCED").upper()
    side_bias = str(feature.get("side_bias") or "NEUTRAL").upper()
    if side_bias == "NEUTRAL" and market_regime in ("TREND_UP", "TREND_DOWN", "BREAKOUT_EXPANSION"):
        side_bias = str((market_snapshot or {}).get("side_bias") or "NEUTRAL").upper()

    strategy_multipliers = dict(_STRATEGY_MULTIPLIERS.get(symbol_regime, {}))
    allowed = [name for name, value in strategy_multipliers.items() if value >= 1.04]
    deprioritized = [name for name, value in strategy_multipliers.items() if value < 0.98]
    blocked = sorted(_BLOCKED_BY_SYMBOL_REGIME.get(symbol_regime, set()))
    if market_regime == "RISK_OFF_EVENT":
        blocked = sorted(set(blocked) | {"TCB15"})
        if "AZ15" not in deprioritized:
            deprioritized.append("AZ15")
    return {
        "enabled": enabled,
        "symbol": symbol,
        "market_regime": market_regime,
        "market_side_bias": str((market_snapshot or {}).get("side_bias") or "NEUTRAL").upper(),
        "symbol_regime": symbol_regime,
        "side_bias": side_bias,
        "regime_confidence": _clip(
            max(
                _safe_float(feature.get("trend_score"), 0.0),
                _safe_float(feature.get("expansion_score"), 0.0),
                _safe_float(feature.get("reversal_score"), 0.0),
                _safe_float(feature.get("chop_score"), 0.0),
            ),
            0.0,
            100.0,
        ),
        "volatility_pct": _safe_float(feature.get("volatility_pct"), 0.0),
        "trend_score": _safe_float(feature.get("trend_score"), 0.0),
        "expansion_score": _safe_float(feature.get("expansion_score"), 0.0),
        "reversal_score": _safe_float(feature.get("reversal_score"), 0.0),
        "chop_score": _safe_float(feature.get("chop_score"), 0.0),
        "allowed_strategies": allowed,
        "deprioritized_strategies": sorted(set(deprioritized)),
        "blocked_strategies": blocked,
        "strategy_multipliers": strategy_multipliers,
        "reason_tags": [symbol_regime.lower(), side_bias.lower() if side_bias else "neutral"],
    }


def apply_regime_to_candidate(candidate, *, regime_payload, config):
    updated = dict(candidate or {})
    payload = dict(regime_payload or {})
    if not bool(getattr(config, "TELEGRAM_ALERT_REGIME_ENABLED", True)) or not payload:
        updated["regime"] = payload
        updated["regime_adjustment"] = {"applied": False, "blocked": False, "multiplier": 1.0}
        return updated, updated["regime_adjustment"]

    strategy = str(updated.get("strategy") or "").strip().upper()
    signal = _normalize_signal(updated.get("signal"))
    confidence = _safe_float(updated.get("confidence"), None)
    base_score = _safe_float(updated.get("score"), 0.0)
    symbol_regime = str(payload.get("symbol_regime") or "RANGE_BALANCED").upper()
    market_regime = str(payload.get("market_regime") or "RANGE_BALANCED").upper()
    side_bias = str(payload.get("side_bias") or "NEUTRAL").upper()
    regime_confidence = _safe_float(payload.get("regime_confidence"), 0.0)
    multiplier = 1.0
    if bool(getattr(config, "TELEGRAM_ALERT_REGIME_SCORE_MULTIPLIER_ENABLED", True)):
        multiplier = _safe_float((payload.get("strategy_multipliers") or {}).get(strategy), 1.0)
        if multiplier is None:
            multiplier = 1.0

    reasons = []
    if signal in ("BUY", "SELL") and side_bias in ("BUY", "SELL") and signal != side_bias and regime_confidence >= 68.0:
        multiplier *= 0.90
        reasons.append("opposing_side_penalty")
    if market_regime == "RISK_OFF_EVENT" and signal == "BUY":
        multiplier *= 0.88
        reasons.append("risk_off_buy_penalty")
    multiplier = _clip(multiplier, 0.60, 1.35)

    min_conf_uplift = 0.0
    if symbol_regime in ("LOW_LIQUIDITY_CHOP", "RANGE_HIGH_VOL"):
        min_conf_uplift += _safe_float(getattr(config, "TELEGRAM_ALERT_REGIME_MIN_CONFIDENCE_UPLIFT", 4.0), 4.0)
    if market_regime == "RISK_OFF_EVENT":
        min_conf_uplift += _safe_float(getattr(config, "TELEGRAM_ALERT_REGIME_RISK_OFF_CONFIDENCE_UPLIFT", 6.0), 6.0)

    block_enabled = bool(getattr(config, "TELEGRAM_ALERT_REGIME_BLOCK_ENABLED", True))
    blocked_strategies = set(payload.get("blocked_strategies") or [])
    blocked = False
    block_reason = None
    if block_enabled and strategy in blocked_strategies:
        blocked = True
        block_reason = f"blocked_{symbol_regime.lower()}"
    opposing_side_min_conf = _safe_float(getattr(config, "TELEGRAM_ALERT_REGIME_OPPOSING_SIDE_MIN_CONFIDENCE", 84.0), 84.0)
    if not blocked and signal in ("BUY", "SELL") and side_bias in ("BUY", "SELL") and signal != side_bias and regime_confidence >= 72.0:
        if confidence is None or confidence < float(opposing_side_min_conf):
            blocked = True
            block_reason = "blocked_opposing_side"

    updated["raw_score"] = base_score
    updated["score"] = float(base_score) * float(multiplier)
    updated["regime"] = payload
    updated["regime_adjustment"] = {
        "applied": True,
        "blocked": blocked,
        "block_reason": block_reason,
        "multiplier": float(multiplier),
        "min_confidence_uplift": float(min_conf_uplift),
        "reasons": reasons,
    }
    return updated, updated["regime_adjustment"]


def build_regime_summary(results, *, config, helpers):
    context = build_regime_context(results, config=config, helpers=helpers)
    return {
        "enabled": context.get("enabled"),
        "generated_at": context.get("generated_at"),
        "market": context.get("market"),
        "symbols": context.get("symbols"),
        "by_symbol_regime": context.get("by_symbol_regime"),
        "by_side_bias": context.get("by_side_bias"),
        "alert_budget": context.get("alert_budget"),
    }


def build_regime_context(results, *, config, helpers):
    market_snapshot = build_market_regime_snapshot(results, config=config, helpers=helpers)
    symbols = []
    symbol_map = {}
    by_symbol_regime = Counter()
    by_side_bias = Counter()
    for item in results or []:
        if not isinstance(item, dict) or item.get("error"):
            continue
        symbol_payload = build_symbol_regime(item, market_snapshot=market_snapshot, config=config, helpers=helpers)
        symbol = str(symbol_payload.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        symbols.append(symbol_payload)
        symbol_map[symbol] = symbol_payload
        by_symbol_regime[str(symbol_payload.get("symbol_regime") or "UNKNOWN")] += 1
        by_side_bias[str(symbol_payload.get("side_bias") or "NEUTRAL")] += 1
    context = {
        "enabled": bool(getattr(config, "TELEGRAM_ALERT_REGIME_ENABLED", True)),
        "generated_at": helpers["get_now"]().strftime("%Y-%m-%d %H:%M:%S"),
        "market": market_snapshot,
        "symbols": symbols,
        "symbol_map": symbol_map,
        "by_symbol_regime": dict(by_symbol_regime),
        "by_side_bias": dict(by_side_bias),
    }
    context["alert_budget"] = build_regime_alert_budget(regime_summary=context, config=config)
    return context


def build_regime_alert_budget(*, regime_summary, config):
    enabled = bool(getattr(config, "TELEGRAM_ALERT_REGIME_BUDGET_ENABLED", True))
    base_run_cap = max(1, int(getattr(config, "TELEGRAM_ALERT_MAX_PER_RUN", 2) or 2))
    base_per_symbol_cap = max(1, int(getattr(config, "TELEGRAM_ALERT_MAX_PER_SYMBOL", 1) or 1))
    base_daily_pick_cap = max(1, int(getattr(config, "TELEGRAM_DAILY_BEST_PICK_MAX_PER_DAY", 1) or 1))
    market = (regime_summary or {}).get("market") if isinstance(regime_summary, dict) else {}
    market_regime = str((market or {}).get("market_regime") or "RANGE_BALANCED").upper()
    budget_template = dict(_MARKET_ALERT_BUDGETS.get(market_regime, _MARKET_ALERT_BUDGETS["RANGE_BALANCED"]))
    run_cap = min(base_run_cap, max(1, int(budget_template.get("run_cap", base_run_cap))))
    per_symbol_cap = min(base_per_symbol_cap, max(1, int(budget_template.get("per_symbol_cap", base_per_symbol_cap))))
    daily_pick_cap = min(base_daily_pick_cap, max(1, int(budget_template.get("daily_pick_cap", base_daily_pick_cap))))
    confidence_uplift = _safe_float(budget_template.get("confidence_uplift"), 0.0) or 0.0
    if not enabled:
        run_cap = base_run_cap
        per_symbol_cap = base_per_symbol_cap
        daily_pick_cap = base_daily_pick_cap
        confidence_uplift = 0.0
    return {
        "enabled": enabled,
        "market_regime": market_regime,
        "base_run_cap": base_run_cap,
        "base_per_symbol_cap": base_per_symbol_cap,
        "base_daily_pick_cap": base_daily_pick_cap,
        "adjusted_run_cap": int(run_cap),
        "adjusted_per_symbol_cap": int(per_symbol_cap),
        "adjusted_daily_pick_cap": int(daily_pick_cap),
        "confidence_uplift": float(confidence_uplift),
        "reason": f"budget_for_{market_regime.lower()}",
    }
