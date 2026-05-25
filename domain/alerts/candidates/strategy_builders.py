from domain.alerts.candidates.common import (
    add_quality_drop,
    append_candidate,
    score_with_edge_adjustments,
)


def _build_super_signal_candidates(item, symbol, context):
    evaluate_super_signal = context["helpers"]["evaluate_super_signal"]
    pick_primary_trade_plan = context["helpers"]["pick_primary_trade_plan"]
    build_super_signal_message = context["helpers"]["build_super_signal_message"]

    for signal in ("BUY", "SELL"):
        ss_ok, _, ss_meta = evaluate_super_signal(item, signal)
        if not ss_ok:
            continue
        ss_plan = pick_primary_trade_plan(
            item,
            signal=signal,
            require_quality=context["strict_60"],
            allow_cdc=context["allow_cdc"],
        )
        ss_message = build_super_signal_message(item, signal, ss_meta, primary_plan=ss_plan)
        if not ss_message:
            continue
        append_candidate(
            context,
            {
                "symbol": symbol,
                "strategy": "SS15",
                "signal": signal,
                "score": 1000.0 + (float(ss_meta.get("avg_wr", 0)) * 2.0),
                "confidence": float(ss_meta.get("avg_wr", 0)),
                "plan": ss_plan,
                "item": item,
                "edge_metrics": ss_meta,
                "message": ss_message,
                "cache_key": f"SS15|{symbol}|{signal}|{context['hour_key']}",
            },
        )


def _build_all_weather_candidates(item, symbol, context):
    build_all_weather_signal = context["helpers"]["build_all_weather_signal"]
    build_all_weather_message = context["helpers"]["build_all_weather_message"]

    aw_signal, aw_meta = build_all_weather_signal(item, context["min_conf"])
    if isinstance(aw_signal, dict):
        aw_message = build_all_weather_message(item, aw_signal)
        if aw_message:
            append_candidate(
                context,
                {
                    "symbol": symbol,
                    "strategy": "AW15",
                    "signal": aw_signal.get("signal"),
                    "score": float(aw_signal.get("score", 0.0)),
                    "confidence": float(aw_signal.get("confidence", 0.0)),
                    "plan": aw_signal.get("primary_plan"),
                    "item": item,
                    "edge_metrics": aw_signal.get("blend") or {},
                    "message": aw_message,
                    "cache_key": str(aw_signal.get("cache_key") or f"AW15|{symbol}|{context['hour_key']}"),
                },
            )
        else:
            return
    elif isinstance(aw_meta, dict):
        add_quality_drop(context, aw_meta.get("status") or "filtered", prefix="all_weather_")


def _build_cdc_candidates(item, symbol, context):
    normalize_confidence = context["helpers"]["normalize_confidence"]
    build_cdc_vixfix_message = context["helpers"]["build_cdc_vixfix_message"]
    extract_plan_edge_metrics = context["helpers"]["extract_plan_edge_metrics"]
    format_price_value = context["helpers"]["format_price_value"]
    alert_profile_score_adjustment = context["helpers"]["alert_profile_score_adjustment"]

    cdc_plan = item.get("cdc_vixfix_15m")
    if not isinstance(cdc_plan, dict):
        return
    cdc_signal = str(cdc_plan.get("signal") or "").upper()
    cdc_conf = normalize_confidence(cdc_plan.get("confidence"))
    cdc_min_conf = getattr(context["config"], "CDC_VIXFIX_15M_MIN_ALERT_CONFIDENCE", context["min_conf"])
    try:
        cdc_min_conf = float(cdc_min_conf)
    except Exception:
        cdc_min_conf = float(context["min_conf"])
    required_conf = min(float(context["min_conf"]), float(cdc_min_conf))
    if cdc_signal not in ("BUY", "SELL") or cdc_conf is None or cdc_conf < required_conf:
        return
    cdc_message = build_cdc_vixfix_message(item, cdc_plan)
    if not cdc_message:
        return
    edge = extract_plan_edge_metrics(cdc_plan)
    freshness = 6.0
    last_signal_time = str(cdc_plan.get("last_signal_time") or "").strip()
    if not last_signal_time:
        freshness = 2.0
    trigger = str(cdc_plan.get("sell_trigger") or "").upper().strip()
    score = float(cdc_conf) + freshness + (6.0 if cdc_signal == "SELL" else 4.0)
    score = score_with_edge_adjustments(
        score,
        edge,
        confidence=cdc_conf,
        alert_profile_score_adjustment=alert_profile_score_adjustment,
    )
    if cdc_signal == "SELL" and cdc_conf >= 85.0:
        score += 8.0
    context_key = last_signal_time or trigger or (format_price_value(cdc_plan.get("entry_price")) or "na")
    append_candidate(
        context,
        {
            "symbol": symbol,
            "strategy": "CDCVIX15",
            "signal": cdc_signal,
            "score": float(score),
            "confidence": float(cdc_conf),
            "plan": cdc_plan,
            "item": item,
            "edge_metrics": edge,
            "message": cdc_message,
            "cache_key": f"CDCVIX15|{symbol}|{cdc_signal}|{context_key}",
        },
    )


def _build_actionzone_candidates(item, symbol, context):
    normalize_confidence = context["helpers"]["normalize_confidence"]
    evaluate_entry_quality_gate = context["helpers"]["evaluate_entry_quality_gate"]
    build_actionzone_message = context["helpers"]["build_actionzone_message"]
    safe_float = context["helpers"]["safe_float"]
    format_price_value = context["helpers"]["format_price_value"]
    alert_profile_score_adjustment = context["helpers"]["alert_profile_score_adjustment"]

    az_plan = item.get("actionzone_15m")
    if not isinstance(az_plan, dict):
        return
    az_signal = str(az_plan.get("signal") or "").upper()
    if az_signal not in ("BUY", "SELL") or not az_plan.get("alert"):
        return
    az_min_conf = getattr(context["config"], "ACTIONZONE_15M_MIN_ALERT_CONFIDENCE", context["min_conf"])
    try:
        az_min_conf = float(az_min_conf)
    except Exception:
        az_min_conf = float(context["min_conf"])
    required_az_conf = max(float(context["min_conf"]), float(az_min_conf))
    if isinstance(context["precision60"], dict):
        required_az_conf = max(required_az_conf, float(context["precision60"].get("min_conf", required_az_conf)))
    az_conf = normalize_confidence(az_plan.get("confidence"))
    if az_conf is None or az_conf < required_az_conf:
        return
    gate_ok, gate_reason, edge = evaluate_entry_quality_gate(az_plan, az_signal)
    if not gate_ok:
        add_quality_drop(context, gate_reason)
        return
    az_message = build_actionzone_message(item, az_plan)
    if not az_message:
        return
    trend_alignment = bool(az_plan.get("trend_alignment", True))
    bars_since = safe_float(az_plan.get("bars_since_signal"), None)
    if isinstance(context["precision60"], dict):
        if bool(context["precision60"].get("require_trend_alignment", True)) and not trend_alignment:
            return
        if isinstance(bars_since, float) and bars_since > float(context["precision60"].get("max_bars_since_signal", 0)):
            return
    freshness = 0.0
    if isinstance(bars_since, float):
        if bars_since <= 0:
            freshness = 8.0
        elif bars_since <= 1:
            freshness = 5.0
        elif bars_since <= 2:
            freshness = 2.0
        else:
            freshness = -4.0
    score = float(az_conf) + freshness + (6.0 if trend_alignment else -8.0)
    score = score_with_edge_adjustments(
        score,
        edge,
        confidence=az_conf,
        alert_profile_score_adjustment=alert_profile_score_adjustment,
    )
    if az_signal == "SELL" and isinstance(edge.get("win_rate_pct"), (int, float)) and edge.get("win_rate_pct") >= 65.0:
        score += 12.0
    last_signal_time = str(az_plan.get("last_signal_time") or "").strip()
    zone = str(az_plan.get("zone") or "").upper().strip()
    entry_bucket = format_price_value(az_plan.get("entry_price")) or "na"
    context_key = last_signal_time or f"{zone}|{entry_bucket}"
    append_candidate(
        context,
        {
            "symbol": symbol,
            "strategy": "AZ15",
            "signal": az_signal,
            "score": float(score),
            "confidence": float(az_conf),
            "plan": az_plan,
            "item": item,
            "edge_metrics": edge,
            "message": az_message,
            "cache_key": f"AZ15|{symbol}|{az_signal}|{context_key}",
        },
    )


def _build_price_action_candidates(item, symbol, context):
    normalize_confidence = context["helpers"]["normalize_confidence"]
    safe_float = context["helpers"]["safe_float"]
    evaluate_entry_quality_gate = context["helpers"]["evaluate_entry_quality_gate"]
    build_price_action_message = context["helpers"]["build_price_action_message"]
    format_price_value = context["helpers"]["format_price_value"]
    alert_profile_score_adjustment = context["helpers"]["alert_profile_score_adjustment"]

    pa_plan = item.get("price_action_15m")
    if not isinstance(pa_plan, dict):
        return
    pa_signal = str(pa_plan.get("signal") or "").upper()
    pa_conf = normalize_confidence(pa_plan.get("confidence"))
    pa_min_conf = safe_float(getattr(context["config"], "PRICE_ACTION_15M_MIN_ALERT_CONFIDENCE", context["min_conf"]), float(context["min_conf"]))
    pa_min_score = safe_float(getattr(context["config"], "PRICE_ACTION_15M_MIN_SCORE", 68.0), 68.0)
    pa_score = safe_float(pa_plan.get("score"), 0.0)
    if pa_signal not in ("BUY", "SELL") or not pa_plan.get("alert") or pa_conf is None or pa_conf < pa_min_conf or pa_score < pa_min_score:
        return
    gate_ok, gate_reason, edge = evaluate_entry_quality_gate(pa_plan, pa_signal)
    if not gate_ok:
        add_quality_drop(context, gate_reason, prefix="price_action_")
        return
    pa_message = build_price_action_message(item, pa_plan)
    if not pa_message:
        return
    score = score_with_edge_adjustments(
        float(pa_score),
        edge,
        confidence=pa_conf,
        alert_profile_score_adjustment=alert_profile_score_adjustment,
    )
    context_key = "|".join(
        [
            str(pa_plan.get("last_signal_time") or ""),
            str(pa_plan.get("market_structure") or ""),
            str(pa_plan.get("chart_pattern") or pa_plan.get("detected_pattern") or ""),
        ]
    ).strip("|") or (format_price_value(pa_plan.get("entry_price")) or "na")
    append_candidate(
        context,
        {
            "symbol": symbol,
            "strategy": "PA15",
            "signal": pa_signal,
            "score": float(score),
            "confidence": float(pa_conf),
            "plan": pa_plan,
            "item": item,
            "edge_metrics": edge,
            "message": pa_message,
            "cache_key": f"PA15|{symbol}|{pa_signal}|{context_key}",
        },
    )


def _build_trend_breakout_candidates(item, symbol, context):
    normalize_confidence = context["helpers"]["normalize_confidence"]
    safe_float = context["helpers"]["safe_float"]
    evaluate_entry_quality_gate = context["helpers"]["evaluate_entry_quality_gate"]
    build_trend_breakout_message = context["helpers"]["build_trend_breakout_message"]
    format_price_value = context["helpers"]["format_price_value"]
    alert_profile_score_adjustment = context["helpers"]["alert_profile_score_adjustment"]

    tcb_plan = item.get("trend_breakout_15m")
    if not isinstance(tcb_plan, dict):
        return
    tcb_signal = str(tcb_plan.get("signal") or "").upper()
    tcb_conf = normalize_confidence(tcb_plan.get("confidence"))
    tcb_min_conf = safe_float(getattr(context["config"], "TREND_BREAKOUT_15M_MIN_ALERT_CONFIDENCE", context["min_conf"]), float(context["min_conf"]))
    tcb_min_score = safe_float(getattr(context["config"], "TREND_BREAKOUT_15M_MIN_SCORE", 68.0), 68.0)
    tcb_score = safe_float(tcb_plan.get("score"), 0.0)
    if tcb_signal not in ("BUY", "SELL") or not tcb_plan.get("alert") or tcb_conf is None or tcb_conf < tcb_min_conf or tcb_score < tcb_min_score:
        return
    gate_ok, gate_reason, edge = evaluate_entry_quality_gate(tcb_plan, tcb_signal)
    if not gate_ok:
        add_quality_drop(context, gate_reason, prefix="trend_breakout_")
        return
    tcb_message = build_trend_breakout_message(item, tcb_plan)
    if not tcb_message:
        return
    score = score_with_edge_adjustments(
        float(tcb_score),
        edge,
        confidence=tcb_conf,
        alert_profile_score_adjustment=alert_profile_score_adjustment,
    )
    context_key = "|".join(
        [
            str(tcb_plan.get("last_signal_time") or ""),
            str(tcb_plan.get("trend_1h") or ""),
            str(tcb_plan.get("breakout_level") or ""),
        ]
    ).strip("|") or (format_price_value(tcb_plan.get("entry_price")) or "na")
    append_candidate(
        context,
        {
            "symbol": symbol,
            "strategy": "TCB15",
            "signal": tcb_signal,
            "score": float(score),
            "confidence": float(tcb_conf),
            "plan": tcb_plan,
            "item": item,
            "edge_metrics": edge,
            "message": tcb_message,
            "cache_key": f"TCB15|{symbol}|{tcb_signal}|{context_key}",
        },
    )


def _build_primary_candidates(item, symbol, context):
    get_best_confidence = context["helpers"]["get_best_confidence"]
    collect_alert_sources = context["helpers"]["collect_alert_sources"]
    pick_primary_trade_plan = context["helpers"]["pick_primary_trade_plan"]
    evaluate_entry_quality_gate = context["helpers"]["evaluate_entry_quality_gate"]
    build_telegram_message = context["helpers"]["build_telegram_message"]
    format_price_value = context["helpers"]["format_price_value"]
    get_primary_plan_source_label = context["helpers"]["get_primary_plan_source_label"]
    pick_plan_value = context["helpers"]["pick_plan_value"]
    alert_profile_score_adjustment = context["helpers"]["alert_profile_score_adjustment"]

    signal = str(item.get("signal") or "").upper()
    if signal not in ("BUY", "SELL"):
        return
    best_conf = get_best_confidence(
        item,
        signal=signal,
        require_quality=context["strict_60"],
        allow_cdc=context["allow_cdc"],
    )
    if best_conf is None or best_conf < context["min_conf"]:
        return
    sources = collect_alert_sources(
        item,
        context["min_conf"],
        signal=signal,
        require_quality=context["strict_60"],
        allow_cdc=context["allow_cdc"],
    )
    primary_plan = pick_primary_trade_plan(
        item,
        signal=signal,
        require_quality=context["strict_60"],
        allow_cdc=context["allow_cdc"],
    )
    if not isinstance(primary_plan, dict):
        add_quality_drop(context, "no_primary_plan_after_strict_gate")
        return
    gate_ok, gate_reason, edge = evaluate_entry_quality_gate(primary_plan, signal)
    if not gate_ok:
        add_quality_drop(context, gate_reason)
        return
    min_sources = getattr(context["config"], "TELEGRAM_ALERT_PRIMARY_MIN_SOURCES", 2)
    single_source_min_conf = getattr(context["config"], "TELEGRAM_ALERT_PRIMARY_SINGLE_SOURCE_MIN_CONF", 90.0)
    try:
        min_sources = int(min_sources)
    except Exception:
        min_sources = 2
    try:
        single_source_min_conf = float(single_source_min_conf)
    except Exception:
        single_source_min_conf = 90.0
    source_count = len(sources)
    if source_count < max(1, min_sources) and float(best_conf) < float(single_source_min_conf):
        return
    message = build_telegram_message(item, signal, best_conf, sources, primary_plan=primary_plan)
    if not message:
        return
    source_bonus = min(8.0, float(len(sources)) * 1.5)
    score = score_with_edge_adjustments(
        float(best_conf) + source_bonus,
        edge,
        confidence=best_conf,
        alert_profile_score_adjustment=alert_profile_score_adjustment,
    )
    last_signal_time = str(primary_plan.get("last_signal_time") or "").strip() if isinstance(primary_plan, dict) else ""
    entry_bucket = format_price_value(pick_plan_value(primary_plan, ["entry_price", "current_price", "price"])) if isinstance(primary_plan, dict) else None
    pattern_bucket = str(primary_plan.get("detected_pattern") or "").strip().upper() if isinstance(primary_plan, dict) else ""
    source_bucket = get_primary_plan_source_label(item, primary_plan) if isinstance(primary_plan, dict) else "PRIMARY"
    context_key = last_signal_time or "|".join([source_bucket, entry_bucket or "na", pattern_bucket or "NOPATTERN"])
    append_candidate(
        context,
        {
            "symbol": symbol,
            "strategy": "PRIMARY",
            "signal": signal,
            "score": float(score),
            "confidence": float(best_conf),
            "plan": primary_plan,
            "item": item,
            "edge_metrics": edge,
            "source_count": source_count,
            "message": message,
            "cache_key": f"PRIMARY|{symbol}|{signal}|{context_key}",
        },
    )


def build_candidates_for_item(item, context):
    normalize_symbol = context["helpers"]["normalize_symbol"]
    if not isinstance(item, dict) or item.get("error"):
        return
    symbol = normalize_symbol(item.get("symbol") or "")
    if not symbol:
        return
    _build_super_signal_candidates(item, symbol, context)
    _build_all_weather_candidates(item, symbol, context)
    _build_cdc_candidates(item, symbol, context)
    _build_actionzone_candidates(item, symbol, context)
    _build_price_action_candidates(item, symbol, context)
    _build_trend_breakout_candidates(item, symbol, context)
    _build_primary_candidates(item, symbol, context)
