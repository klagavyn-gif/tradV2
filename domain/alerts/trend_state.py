from domain.alerts.trend_1h import infer_1h_trend_snapshot


def _safe_float(value, default=None):
    try:
        number = float(value)
    except Exception:
        return default
    return number if number == number else default


def _clip(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _normalize_signal(value):
    text = str(value or "").strip().upper()
    if text in ("BUY", "UP", "BULLISH", "LONG"):
        return "BUY"
    if text in ("SELL", "DOWN", "BEARISH", "SHORT"):
        return "SELL"
    return None


def _directional_plan_sources(item):
    plan_specs = [
        ("ActionZone 15m", item.get("actionzone_15m")),
        ("Trend Breakout 15m", item.get("trend_breakout_15m")),
        ("Price Action 15m", item.get("price_action_15m")),
        ("CDC VixFix 15m", item.get("cdc_vixfix_15m")),
        ("EMA Cross 15m", item.get("ema_cross_15m")),
    ]
    rows = []
    for label, plan in plan_specs:
        if not isinstance(plan, dict):
            continue
        signal = _normalize_signal(plan.get("signal"))
        if signal not in ("BUY", "SELL"):
            continue
        confidence = _safe_float(plan.get("confidence"), None)
        score = _safe_float(plan.get("score"), None)
        trend_alignment = bool(plan.get("trend_alignment")) if "trend_alignment" in plan else None
        alert = bool(plan.get("alert")) if "alert" in plan else None
        qualified = False
        if label == "Trend Breakout 15m":
            qualified = bool(alert) or (isinstance(score, (int, float)) and score >= 75.0)
        elif label == "ActionZone 15m":
            qualified = bool(trend_alignment) and isinstance(confidence, (int, float)) and confidence >= 70.0
        elif label == "Price Action 15m":
            qualified = isinstance(confidence, (int, float)) and confidence >= 72.0
        elif label == "CDC VixFix 15m":
            qualified = signal == "BUY" and isinstance(confidence, (int, float)) and confidence >= 72.0
        elif label == "EMA Cross 15m":
            qualified = isinstance(confidence, (int, float)) and confidence >= 75.0
        if not qualified:
            continue
        rows.append(
            {
                "label": label,
                "signal": signal,
                "confidence": confidence,
                "score": score,
                "alert": alert,
                "trend_alignment": trend_alignment,
            }
        )
    return rows


def _matching_regime(signal, symbol_regime):
    symbol_regime = str(symbol_regime or "").strip().upper()
    if signal == "BUY":
        return symbol_regime in ("TREND_UP", "BREAKOUT_EXPANSION", "PANIC_REVERSAL")
    if signal == "SELL":
        return symbol_regime in ("TREND_DOWN", "BREAKOUT_EXPANSION", "RISK_OFF_EVENT")
    return False


def build_trend_state_snapshot(item, *, config, helpers, market_snapshot=None):
    if not isinstance(item, dict) or item.get("error"):
        return None
    trend_snapshot = infer_1h_trend_snapshot(item)
    if not isinstance(trend_snapshot, dict):
        return None

    trend = str(trend_snapshot.get("trend") or "").strip().upper()
    signal = "BUY" if trend == "UP" else "SELL" if trend == "DOWN" else None
    if signal not in ("BUY", "SELL"):
        return None

    trend_confidence = _safe_float(trend_snapshot.get("confidence"), 0.0)
    min_trend_conf = _safe_float(
        getattr(config, "TREND_STATE_ALERT_STRONG_1H_MIN_CONFIDENCE", 68.0),
        68.0,
    )
    if trend_confidence < float(min_trend_conf):
        return None

    normalize_symbol = helpers["normalize_symbol"]
    build_symbol_regime = helpers["build_symbol_regime"]
    symbol = normalize_symbol(item.get("symbol") or "")
    if not symbol:
        return None

    regime_payload = {}
    if callable(build_symbol_regime):
        regime_payload = build_symbol_regime(item, market_snapshot=market_snapshot)
    if not isinstance(regime_payload, dict):
        regime_payload = {}

    symbol_regime = str(regime_payload.get("symbol_regime") or "RANGE_BALANCED").strip().upper()
    market_regime = str(regime_payload.get("market_regime") or (market_snapshot or {}).get("market_regime") or "RANGE_BALANCED").strip().upper()
    require_regime_confirmation = bool(getattr(config, "TREND_STATE_ALERT_REQUIRE_REGIME_CONFIRMATION", True))
    if require_regime_confirmation and not _matching_regime(signal, symbol_regime):
        return None

    directional_sources = [row for row in _directional_plan_sources(item) if row.get("signal") == signal]
    source_labels = list(trend_snapshot.get("source_labels") or [])
    directional_count = len(directional_sources)
    min_sources = max(1, int(getattr(config, "TREND_STATE_ALERT_MIN_DIRECTIONAL_SOURCES", 2) or 2))
    if directional_count < min_sources:
        return None

    breakout_plan = item.get("trend_breakout_15m") if isinstance(item.get("trend_breakout_15m"), dict) else {}
    actionzone_plan = item.get("actionzone_15m") if isinstance(item.get("actionzone_15m"), dict) else {}
    price_action_plan = item.get("price_action_15m") if isinstance(item.get("price_action_15m"), dict) else {}
    cdc_plan = item.get("cdc_vixfix_15m") if isinstance(item.get("cdc_vixfix_15m"), dict) else {}

    score = float(trend_confidence) * 0.50
    if str(trend_snapshot.get("strength") or "").strip().upper() == "STRONG":
        score += 8.0
    score += min(12.0, max(0.0, float(directional_count - 1) * 5.0))

    if _normalize_signal(breakout_plan.get("signal")) == signal and breakout_plan.get("alert"):
        score += 14.0
    if _normalize_signal(actionzone_plan.get("signal")) == signal and bool(actionzone_plan.get("trend_alignment")):
        score += 10.0
    if _normalize_signal(price_action_plan.get("signal")) == signal:
        score += 7.0
    if signal == "BUY" and _normalize_signal(cdc_plan.get("signal")) == "BUY":
        score += 5.0

    if signal == "BUY" and symbol_regime == "TREND_UP":
        score += 14.0
    elif signal == "SELL" and symbol_regime == "TREND_DOWN":
        score += 14.0
    elif symbol_regime == "BREAKOUT_EXPANSION":
        score += 18.0
    elif signal == "SELL" and symbol_regime == "RISK_OFF_EVENT":
        score += 16.0
    elif signal == "BUY" and symbol_regime == "TREND_DOWN":
        score -= 26.0
    elif signal == "SELL" and symbol_regime == "TREND_UP":
        score -= 26.0
    elif symbol_regime in ("RANGE_BALANCED", "LOW_LIQUIDITY_CHOP"):
        score -= 14.0

    abs_change = abs(_safe_float(item.get("change"), 0.0) or 0.0)
    if abs_change >= 4.0:
        score += 8.0
    elif abs_change >= 2.0:
        score += 4.0

    score = _clip(score, 0.0, 95.0)
    min_score = _safe_float(getattr(config, "TREND_STATE_ALERT_MIN_SCORE", 72.0), 72.0)
    if score < float(min_score):
        return None

    tags = []
    if symbol_regime == "BREAKOUT_EXPANSION":
        tags.append("BREAKOUT_EXPANSION")
    if signal == "SELL" and symbol_regime == "RISK_OFF_EVENT":
        tags.append("RISK_OFF")
    if signal == "BUY" and symbol_regime == "PANIC_REVERSAL":
        tags.append("PANIC_REVERSAL")
    if abs_change >= 4.0 or (breakout_plan.get("alert") and _normalize_signal(breakout_plan.get("signal")) == signal):
        tags.append("ACCELERATING")

    state_label = "TREND_UP_STRONG" if signal == "BUY" else "TREND_DOWN_STRONG"
    reasons = [
        f"1H {trend} {str(trend_snapshot.get('strength') or 'WEAK').strip().upper()}",
        f"Regime {symbol_regime}",
        f"Consensus {directional_count} sources",
    ]
    return {
        "symbol": symbol,
        "signal": signal,
        "trend": trend,
        "trend_strength": str(trend_snapshot.get("strength") or "").strip().upper(),
        "trend_confidence": float(trend_confidence),
        "agreement_ratio": _safe_float(trend_snapshot.get("agreement_ratio"), 0.0),
        "directional_source_count": int(directional_count),
        "source_labels": source_labels,
        "supporting_sources": [row.get("label") for row in directional_sources if row.get("label")],
        "opposing_labels": list(trend_snapshot.get("opposing_labels") or []),
        "symbol_regime": symbol_regime,
        "market_regime": market_regime,
        "regime_confidence": _safe_float(regime_payload.get("regime_confidence"), None),
        "trend_score": _safe_float(regime_payload.get("trend_score"), None),
        "expansion_score": _safe_float(regime_payload.get("expansion_score"), None),
        "volatility_pct": _safe_float(regime_payload.get("volatility_pct"), None),
        "tags": tags,
        "score": float(score),
        "state_label": state_label,
        "reason": " | ".join(reasons),
    }


def build_trend_state_candidates(results, *, config, helpers, get_now, runtime_context=None):
    if not bool(getattr(config, "TREND_STATE_ALERT_ENABLED", False)):
        return []

    normalize_symbol = helpers["normalize_symbol"]
    build_market_regime_snapshot = helpers["build_market_regime_snapshot"]
    build_trend_state_message = helpers["build_trend_state_message"]

    market_snapshot = ((runtime_context or {}).get("regime_context") or {}).get("market") if isinstance(runtime_context, dict) else None
    if not isinstance(market_snapshot, dict):
        market_snapshot = build_market_regime_snapshot(results or [])

    candidates = []
    for item in results or []:
        if not isinstance(item, dict) or item.get("error"):
            continue
        symbol = normalize_symbol(item.get("symbol") or "")
        if not symbol:
            continue
        snapshot = build_trend_state_snapshot(
            item,
            config=config,
            helpers=helpers,
            market_snapshot=market_snapshot,
        )
        if not isinstance(snapshot, dict):
            continue
        message = build_trend_state_message(item, snapshot)
        if not isinstance(message, str) or not message.strip():
            continue
        signal = str(snapshot.get("signal") or "").strip().upper()
        state_label = str(snapshot.get("state_label") or "").strip().upper()
        symbol_regime = str(snapshot.get("symbol_regime") or "RANGE_BALANCED").strip().upper()
        tags = [str(tag).strip().upper() for tag in (snapshot.get("tags") or []) if str(tag).strip()]
        tag_bucket = tags[0] if tags else "BASE"
        state_score = float(snapshot.get("score") or 0.0)
        candidates.append(
            {
                "symbol": symbol,
                "strategy": "TRENDSTATE",
                "strategy_label": "Trend State Alert",
                "signal": signal,
                "score": state_score,
                "confidence": state_score,
                "plan": snapshot,
                "item": item,
                "source_count": int(snapshot.get("directional_source_count") or 0),
                "message": message,
                "cache_key": f"TRENDSTATE|{symbol}|{state_label}|{symbol_regime}|{tag_bucket}",
            }
        )

    candidates.sort(key=lambda row: (float(row.get("score", 0.0)), float(row.get("confidence", 0.0))), reverse=True)
    return candidates
