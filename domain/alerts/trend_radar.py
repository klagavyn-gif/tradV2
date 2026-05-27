from domain.alerts.trend_1h import infer_1h_trend_snapshot


def _safe_float(value, default=None):
    try:
        number = float(value)
    except Exception:
        return default
    return number if number == number else default


def _resolve_plan_confidence(plan, *fallback_keys):
    if not isinstance(plan, dict):
        return None
    for key in ("confidence", "predicted_win_prob", "win_prob", *fallback_keys):
        value = _safe_float(plan.get(key), None)
        if isinstance(value, (int, float)):
            return value
    return None


def _clip(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _normalize_signal(value):
    text = str(value or "").strip().upper()
    if text in ("BUY", "UP", "BULLISH", "LONG"):
        return "BUY"
    if text in ("SELL", "DOWN", "BEARISH", "SHORT"):
        return "SELL"
    return None


def _resolve_signal_with_fallback(plan, *fallback_keys):
    signal = _normalize_signal((plan or {}).get("signal"))
    if signal in ("BUY", "SELL"):
        return signal, "signal"
    for key in fallback_keys:
        signal = _normalize_signal((plan or {}).get(key))
        if signal in ("BUY", "SELL"):
            return signal, key
    return None, None


def _signal_to_trend(signal):
    if signal == "BUY":
        return "UP"
    if signal == "SELL":
        return "DOWN"
    return None


def _market_bias_matches(signal, market_bias):
    text = str(market_bias or "").strip().upper()
    if signal == "BUY":
        return text == "BULLISH"
    if signal == "SELL":
        return text == "BEARISH"
    return False


def _synthetic_fallback_confidence(plan, signal, *, config):
    if not isinstance(plan, dict) or signal not in ("BUY", "SELL"):
        return None
    score = 54.0
    trend_1h = str(plan.get("trend_1h") or "").strip().upper()
    if trend_1h == _signal_to_trend(signal):
        score += 6.0
    forecast_direction = _normalize_signal(plan.get("forecast_direction"))
    if forecast_direction == signal:
        score += 5.0
    if _market_bias_matches(signal, plan.get("market_bias")):
        score += 4.0

    adx = _safe_float(plan.get("adx"), None)
    min_adx = _safe_float(getattr(config, "TREND_RADAR_MIN_ADX", 20.0), 20.0)
    if isinstance(adx, (int, float)) and adx >= min_adx:
        score += min(7.0, 2.0 + ((float(adx) - float(min_adx)) * 0.35))

    rvol = _safe_float(plan.get("rvol"), None)
    min_rvol = _safe_float(getattr(config, "TREND_RADAR_MIN_RVOL", 1.05), 1.05)
    if isinstance(rvol, (int, float)) and rvol >= min_rvol:
        score += min(5.0, 1.0 + ((float(rvol) - float(min_rvol)) * 5.0))

    forecast_probability = _safe_float(plan.get("forecast_probability"), None)
    predicted_win_prob = _safe_float(plan.get("predicted_win_prob"), None)
    if isinstance(forecast_probability, (int, float)):
        score += max(-3.0, min(7.0, (float(forecast_probability) - 50.0) * 0.28))
    elif isinstance(predicted_win_prob, (int, float)):
        score += max(-2.0, min(6.0, (float(predicted_win_prob) - 50.0) * 0.20))

    hist_wr = _safe_float(plan.get("historical_win_rate"), None)
    hist_exp = _safe_float(plan.get("historical_avg_rr"), None)
    hist_trades = _safe_float(plan.get("historical_trades"), None)
    if isinstance(hist_wr, (int, float)) and hist_wr >= 54.0:
        score += min(4.0, 1.0 + ((float(hist_wr) - 54.0) * 0.18))
    if isinstance(hist_exp, (int, float)) and hist_exp > 0.0:
        score += min(4.0, float(hist_exp) * 18.0)
    if isinstance(hist_trades, (int, float)) and hist_trades >= 6.0:
        score += min(3.0, (float(hist_trades) - 6.0) * 0.20)

    return _clip(score, 0.0, 88.0)


def _matching_regime(signal, symbol_regime):
    symbol_regime = str(symbol_regime or "").strip().upper()
    if signal == "BUY":
        return symbol_regime in ("TREND_UP", "BREAKOUT_EXPANSION", "PANIC_REVERSAL")
    if signal == "SELL":
        return symbol_regime in ("TREND_DOWN", "BREAKOUT_EXPANSION", "RISK_OFF_EVENT")
    return False


def _coerce_trend_strength(value):
    text = str(value or "").strip().upper()
    return text if text in ("STRONG", "WEAK") else "WEAK"


def _plan_source_labels(item):
    labels = []
    if isinstance(item.get("trend_breakout_15m"), dict):
        labels.append("Trend Breakout 15m")
    if isinstance(item.get("actionzone_15m"), dict):
        labels.append("ActionZone 15m")
    if isinstance(item.get("price_action_15m"), dict):
        labels.append("Price Action 15m")
    if isinstance(item.get("short_term_15m"), dict):
        labels.append("ShortTerm 15m")
    return labels


def _extract_tp2(plan):
    if not isinstance(plan, dict):
        return None
    for key in ("take_profit_2", "tp2", "take_profit_price_2"):
        value = _safe_float(plan.get(key), None)
        if isinstance(value, (int, float)):
            return value
    exit_levels = plan.get("exit_levels") or []
    if isinstance(exit_levels, list):
        for level in exit_levels:
            if not isinstance(level, dict):
                continue
            label = str(level.get("label") or "").strip().upper()
            if label == "TP2":
                target = _safe_float(level.get("target_price"), None)
                if isinstance(target, (int, float)):
                    return target
    return None


def _build_regime_payload(item, *, helpers, market_snapshot=None):
    build_symbol_regime = helpers["build_symbol_regime"]
    regime_payload = {}
    if callable(build_symbol_regime):
        regime_payload = build_symbol_regime(item, market_snapshot=market_snapshot)
    return regime_payload if isinstance(regime_payload, dict) else {}


def _base_snapshot(item, *, symbol, signal, subtype, source_label, plan, trend_snapshot, regime_payload):
    symbol_regime = str(regime_payload.get("symbol_regime") or "RANGE_BALANCED").strip().upper()
    market_regime = str(regime_payload.get("market_regime") or "RANGE_BALANCED").strip().upper()
    trend = "UP" if signal == "BUY" else "DOWN"
    return {
        "symbol": symbol,
        "signal": signal,
        "trend": trend,
        "subtype": subtype,
        "source_label": source_label,
        "price": _safe_float(item.get("price"), None),
        "change": _safe_float(item.get("change"), None),
        "trend_1h": str(trend_snapshot.get("trend") or trend).strip().upper(),
        "trend_strength_1h": _coerce_trend_strength(trend_snapshot.get("strength")),
        "trend_1h_confidence": _safe_float(trend_snapshot.get("confidence"), 0.0),
        "symbol_regime": symbol_regime,
        "market_regime": market_regime,
        "plan_confidence": _resolve_plan_confidence(plan),
        "entry_price": _safe_float(plan.get("entry_price") or plan.get("current_price") or plan.get("price"), None),
        "entry_zone_low": _safe_float(plan.get("entry_zone_low"), None),
        "entry_zone_high": _safe_float(plan.get("entry_zone_high"), None),
        "stop_loss": _safe_float(plan.get("stop_loss"), None),
        "take_profit_price": _safe_float(
            plan.get("take_profit")
            or plan.get("take_profit_price")
            or plan.get("exit_price"),
            None,
        ),
        "take_profit_price_2": _extract_tp2(plan),
        "entry_risk_pct": _safe_float(plan.get("entry_risk_pct"), None),
        "adx": _safe_float(plan.get("adx"), None),
        "rvol": _safe_float(plan.get("rvol"), None),
        "ema20": _safe_float(plan.get("ema20"), None),
        "ema50": _safe_float(plan.get("ema50"), None),
        "ema200": _safe_float(plan.get("ema200"), None),
        "supporting_sources": [],
        "reasons": [],
        "tags": [],
    }


def _score_snapshot(snapshot, *, config):
    score = 0.0
    min_adx = _safe_float(getattr(config, "TREND_RADAR_MIN_ADX", 20.0), 20.0)
    min_rvol = _safe_float(getattr(config, "TREND_RADAR_MIN_RVOL", 1.05), 1.05)
    max_entry_risk_pct = _safe_float(getattr(config, "TREND_RADAR_MAX_ENTRY_RISK_PCT", 4.5), 4.5)

    plan_conf = _safe_float(snapshot.get("plan_confidence"), 0.0) or 0.0
    trend_conf = _safe_float(snapshot.get("trend_1h_confidence"), 0.0) or 0.0
    adx = _safe_float(snapshot.get("adx"), None)
    rvol = _safe_float(snapshot.get("rvol"), None)
    risk_pct = _safe_float(snapshot.get("entry_risk_pct"), None)
    symbol_regime = str(snapshot.get("symbol_regime") or "").strip().upper()
    signal = str(snapshot.get("signal") or "").strip().upper()
    subtype = str(snapshot.get("subtype") or "").strip().upper()

    score += min(52.0, plan_conf * 0.52)
    score += min(16.0, trend_conf * 0.18)
    if str(snapshot.get("trend_strength_1h") or "").strip().upper() == "STRONG":
        score += 6.0

    if isinstance(adx, (int, float)) and adx >= min_adx:
        score += min(10.0, 4.0 + ((float(adx) - float(min_adx)) * 0.55))
    if isinstance(rvol, (int, float)) and rvol >= min_rvol:
        score += min(8.0, 3.0 + ((float(rvol) - float(min_rvol)) * 8.0))

    if subtype == "TREND_CONTINUE":
        score += 8.0
    elif subtype == "TREND_START":
        score += 6.0

    if _matching_regime(signal, symbol_regime):
        score += 8.0
    elif symbol_regime in ("RANGE_BALANCED", "LOW_LIQUIDITY_CHOP"):
        score -= 14.0
    else:
        score -= 10.0

    if isinstance(risk_pct, (int, float)):
        if risk_pct <= max_entry_risk_pct:
            score += 4.0
        else:
            score -= min(12.0, (risk_pct - max_entry_risk_pct) * 2.0)

    hist_wr = _safe_float(snapshot.get("historical_win_rate_pct"), None)
    expectancy_rr = _safe_float(snapshot.get("expectancy_rr"), None)
    if isinstance(hist_wr, (int, float)) and hist_wr >= 54.0:
        score += 4.0
    if isinstance(expectancy_rr, (int, float)) and expectancy_rr > 0.0:
        score += min(4.0, expectancy_rr * 20.0)

    return _clip(score, 0.0, 95.0)


def _build_short_term_snapshot(item, *, symbol, trend_snapshot, regime_payload, config):
    plan = item.get("short_term_15m")
    if not isinstance(plan, dict):
        return None
    plan_trend_1h = str(plan.get("trend_1h") or "").strip().upper()
    if plan_trend_1h != "UP":
        return None
    direction_15m = str(plan.get("direction_15m") or "").strip().upper()
    if direction_15m != "UP":
        return None
    setup = str(plan.get("setup") or "").strip().upper()
    if not setup.startswith("BUY"):
        return None
    entry_type = str(plan.get("entry_type") or "").strip().upper()
    if entry_type == "BREAKOUT":
        subtype = "TREND_START"
        if not bool(getattr(config, "TREND_RADAR_START_ENABLED", True)):
            return None
    elif entry_type == "PULLBACK":
        subtype = "TREND_CONTINUE"
        if not bool(getattr(config, "TREND_RADAR_CONTINUE_ENABLED", True)):
            return None
    else:
        return None

    snapshot = _base_snapshot(
        item,
        symbol=symbol,
        signal="BUY",
        subtype=subtype,
        source_label="ShortTerm 15m",
        plan=plan,
        trend_snapshot=trend_snapshot,
        regime_payload=regime_payload,
    )
    snapshot["supporting_sources"] = _plan_source_labels(item)
    snapshot["historical_win_rate_pct"] = _safe_float(plan.get("historical_win_rate_tp1"), None)
    snapshot["expectancy_rr"] = _safe_float(plan.get("expectancy_tp1_rr"), None)
    snapshot["reasons"] = [
        f"{subtype.replace('_', ' ')} ผ่านจาก ShortTerm 15m",
        f"1H {snapshot['trend_1h']} {snapshot['trend_strength_1h']}",
        f"Regime {snapshot['symbol_regime']}",
    ]
    if isinstance(snapshot.get("entry_zone_low"), (int, float)) and isinstance(snapshot.get("entry_zone_high"), (int, float)):
        snapshot["tags"].append("ENTRY_ZONE_READY")
    return snapshot


def _build_trend_breakout_snapshot(item, *, symbol, trend_snapshot, regime_payload, config):
    plan = item.get("trend_breakout_15m")
    if not isinstance(plan, dict):
        return None
    signal, signal_source = _resolve_signal_with_fallback(plan, "forecast_direction", "trend_1h", "market_bias")
    if signal not in ("BUY", "SELL"):
        return None
    confidence = _resolve_plan_confidence(plan)
    min_plan_conf = _safe_float(getattr(config, "TREND_RADAR_MIN_PLAN_CONFIDENCE", 62.0), 62.0)
    if signal_source != "signal" and (not isinstance(confidence, (int, float)) or confidence < min_plan_conf):
        confidence = _synthetic_fallback_confidence(plan, signal, config=config)
    if signal_source == "signal" and not bool(plan.get("alert")):
        return None
    if signal_source != "signal" and (not isinstance(confidence, (int, float)) or confidence < min_plan_conf):
        return None
    expected_trend = "UP" if signal == "BUY" else "DOWN"
    trend_1h = str(trend_snapshot.get("trend") or "").strip().upper()
    if trend_1h != expected_trend:
        return None
    if not bool(getattr(config, "TREND_RADAR_START_ENABLED", True)):
        return None

    snapshot = _base_snapshot(
        item,
        symbol=symbol,
        signal=signal,
        subtype="TREND_START" if signal_source == "signal" else "TREND_CONTINUE",
        source_label="Trend Breakout 15m",
        plan=plan,
        trend_snapshot=trend_snapshot,
        regime_payload=regime_payload,
    )
    snapshot["plan_confidence"] = confidence
    snapshot["supporting_sources"] = ["Trend Breakout 15m"]
    snapshot["historical_win_rate_pct"] = _safe_float(plan.get("historical_win_rate"), None)
    snapshot["expectancy_rr"] = _safe_float(plan.get("historical_avg_rr"), None)
    snapshot["reasons"] = [
        f"{signal} breakout trigger พร้อมใช้งาน" if signal_source == "signal" else f"{signal} breakout bias จาก {signal_source}",
        f"1H {snapshot['trend_1h']} {snapshot['trend_strength_1h']}",
        f"Regime {snapshot['symbol_regime']}",
    ]
    if signal_source != "signal":
        snapshot["tags"].append("DIRECTIONAL_FALLBACK")
    snapshot["tags"].append("BREAKOUT_READY")
    return snapshot


def _build_price_action_snapshot(item, *, symbol, trend_snapshot, regime_payload, config):
    plan = item.get("price_action_15m")
    if not isinstance(plan, dict):
        return None
    signal, signal_source = _resolve_signal_with_fallback(plan, "forecast_direction", "trend_1h")
    if signal not in ("BUY", "SELL"):
        return None
    expected_trend = "UP" if signal == "BUY" else "DOWN"
    trend_1h = str(trend_snapshot.get("trend") or "").strip().upper()
    if trend_1h != expected_trend:
        return None
    if not bool(getattr(config, "TREND_RADAR_CONTINUE_ENABLED", True)):
        return None
    confidence = _resolve_plan_confidence(plan, "forecast_probability")
    min_plan_conf = _safe_float(getattr(config, "TREND_RADAR_MIN_PLAN_CONFIDENCE", 62.0), 62.0)
    if signal_source != "signal" and (not isinstance(confidence, (int, float)) or confidence < min_plan_conf):
        confidence = _synthetic_fallback_confidence(plan, signal, config=config)
    if not isinstance(confidence, (int, float)) or confidence < min_plan_conf:
        return None

    snapshot = _base_snapshot(
        item,
        symbol=symbol,
        signal=signal,
        subtype="TREND_CONTINUE",
        source_label="Price Action 15m",
        plan=plan,
        trend_snapshot=trend_snapshot,
        regime_payload=regime_payload,
    )
    snapshot["plan_confidence"] = confidence
    snapshot["supporting_sources"] = ["Price Action 15m"]
    breakout_plan = item.get("trend_breakout_15m")
    if isinstance(breakout_plan, dict) and _normalize_signal(breakout_plan.get("signal")) == signal:
        snapshot["supporting_sources"].append("Trend Breakout 15m")
    snapshot["historical_win_rate_pct"] = _safe_float(plan.get("historical_win_rate"), None)
    snapshot["expectancy_rr"] = _safe_float(plan.get("historical_avg_rr"), None)
    snapshot["reasons"] = [
        f"{signal} continuation จาก Price Action 15m" if signal_source == "signal" else f"{signal} continuation bias จาก {signal_source}",
        f"1H {snapshot['trend_1h']} {snapshot['trend_strength_1h']}",
        f"Regime {snapshot['symbol_regime']}",
    ]
    if signal_source != "signal":
        snapshot["tags"].append("DIRECTIONAL_FALLBACK")
    if str(plan.get("market_structure") or "").strip():
        snapshot["tags"].append(str(plan.get("market_structure")).strip().upper().replace(" ", "_"))
    return snapshot


def build_trend_radar_snapshot(item, *, config, helpers, market_snapshot=None):
    if not isinstance(item, dict) or item.get("error"):
        return None
    trend_snapshot = infer_1h_trend_snapshot(item)
    if not isinstance(trend_snapshot, dict):
        return None
    trend = str(trend_snapshot.get("trend") or "").strip().upper()
    if trend not in ("UP", "DOWN"):
        return None
    trend_confidence = _safe_float(trend_snapshot.get("confidence"), 0.0)
    min_1h_conf = _safe_float(getattr(config, "TREND_RADAR_MIN_1H_CONFIDENCE", 68.0), 68.0)
    if trend_confidence < min_1h_conf:
        return None

    normalize_symbol = helpers["normalize_symbol"]
    symbol = normalize_symbol(item.get("symbol") or "")
    if not symbol:
        return None
    regime_payload = _build_regime_payload(item, helpers=helpers, market_snapshot=market_snapshot)
    symbol_regime = str(regime_payload.get("symbol_regime") or "RANGE_BALANCED").strip().upper()
    require_regime_confirmation = bool(getattr(config, "TREND_RADAR_REQUIRE_REGIME_CONFIRMATION", True))

    candidates = []
    for builder in (_build_short_term_snapshot, _build_trend_breakout_snapshot, _build_price_action_snapshot):
        snapshot = builder(
            item,
            symbol=symbol,
            trend_snapshot=trend_snapshot,
            regime_payload=regime_payload,
            config=config,
        )
        if not isinstance(snapshot, dict):
            continue
        if require_regime_confirmation and not _matching_regime(str(snapshot.get("signal") or "").strip().upper(), symbol_regime):
            continue
        snapshot["score"] = _score_snapshot(snapshot, config=config)
        candidates.append(snapshot)

    if not candidates:
        return None
    candidates.sort(key=lambda row: (float(row.get("score", 0.0)), float(row.get("plan_confidence") or 0.0)), reverse=True)
    best = dict(candidates[0])
    best["trend_snapshot"] = trend_snapshot
    return best


def build_trend_radar_candidates(results, *, config, helpers, get_now, runtime_context=None):
    if not bool(getattr(config, "TREND_RADAR_ENABLED", False)):
        return []

    normalize_symbol = helpers["normalize_symbol"]
    build_market_regime_snapshot = helpers["build_market_regime_snapshot"]
    build_trend_radar_message = helpers["build_trend_radar_message"]

    market_snapshot = ((runtime_context or {}).get("regime_context") or {}).get("market") if isinstance(runtime_context, dict) else None
    if not isinstance(market_snapshot, dict):
        market_snapshot = build_market_regime_snapshot(results or [])

    min_score = _safe_float(getattr(config, "TREND_RADAR_MIN_SCORE", 74.0), 74.0)
    candidates = []
    for item in results or []:
        if not isinstance(item, dict) or item.get("error"):
            continue
        symbol = normalize_symbol(item.get("symbol") or "")
        if not symbol:
            continue
        snapshot = build_trend_radar_snapshot(
            item,
            config=config,
            helpers=helpers,
            market_snapshot=market_snapshot,
        )
        if not isinstance(snapshot, dict):
            continue
        score = _safe_float(snapshot.get("score"), 0.0)
        if score < min_score:
            continue
        message = build_trend_radar_message(item, snapshot)
        if not isinstance(message, str) or not message.strip():
            continue
        subtype = str(snapshot.get("subtype") or "TREND_CONTINUE").strip().upper()
        signal = str(snapshot.get("signal") or "").strip().upper()
        source_key = str(snapshot.get("source_label") or "BASE").strip().upper().replace(" ", "_")
        entry_bucket = round(_safe_float(snapshot.get("entry_price"), 0.0) or 0.0, 4)
        candidates.append(
            {
                "symbol": symbol,
                "strategy": "TRADAR15",
                "strategy_label": "Trend Radar 15m",
                "signal": signal,
                "score": float(score),
                "confidence": float(score),
                "plan": snapshot,
                "item": item,
                "source_count": len(snapshot.get("supporting_sources") or []),
                "message": message,
                "cache_key": f"TRADAR15|{symbol}|{signal}|{subtype}|{source_key}|{entry_bucket}",
            }
        )

    candidates.sort(key=lambda row: (float(row.get("score", 0.0)), float(row.get("confidence", 0.0))), reverse=True)
    return candidates
