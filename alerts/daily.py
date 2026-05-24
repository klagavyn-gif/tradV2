def is_daily_best_pick_window(*, config, get_now):
    if not bool(getattr(config, "TELEGRAM_DAILY_BEST_PICK_ENABLED", True)):
        return False
    now = get_now()
    hour = getattr(config, "TELEGRAM_DAILY_BEST_PICK_HOUR", 9)
    minute = getattr(config, "TELEGRAM_DAILY_BEST_PICK_MINUTE", 0)
    window_minutes = getattr(config, "TELEGRAM_DAILY_BEST_PICK_WINDOW_MINUTES", 15)
    try:
        hour = int(hour)
    except Exception:
        hour = 9
    try:
        minute = int(minute)
    except Exception:
        minute = 0
    try:
        window_minutes = int(window_minutes)
    except Exception:
        window_minutes = 15
    if hour < 0 or hour > 23:
        hour = 9
    if minute not in (0, 15, 30, 45):
        minute = 0
    if window_minutes < 1:
        window_minutes = 1
    if window_minutes > 60:
        window_minutes = 60
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    delta_minutes = (now - target).total_seconds() / 60.0
    return 0.0 <= delta_minutes < float(window_minutes)


def _normalize_symbol_allowlist(raw_values, normalize_symbol):
    symbols = set()
    if isinstance(raw_values, (set, list, tuple)):
        iterable = list(raw_values)
    else:
        iterable = str(raw_values or "").split(",")
    for value in iterable:
        symbol = normalize_symbol(value or "")
        if symbol:
            symbols.add(symbol)
    return symbols


def _build_daily_best_cdc_followthrough_plan(item, *, config, helpers, signal):
    if str(signal or "").upper().strip() != "BUY":
        return None
    if not bool(getattr(config, "TELEGRAM_DAILY_BEST_PICK_CDC_ENABLE", True)):
        return None
    normalize_symbol = helpers["normalize_symbol"]
    plan_confidence_value = helpers["plan_confidence_value"]
    symbol = normalize_symbol(item.get("symbol") or "")
    cdc_plan = item.get("cdc_vixfix_15m")
    if not symbol or not isinstance(cdc_plan, dict):
        return None
    allowlist = _normalize_symbol_allowlist(
        getattr(config, "TELEGRAM_DAILY_BEST_PICK_SYMBOL_ALLOWLIST", ()),
        normalize_symbol,
    )
    if allowlist and symbol not in allowlist:
        return None
    if str(cdc_plan.get("signal") or "").upper().strip() == "BUY":
        return None
    trend_color = str(cdc_plan.get("trend_color") or "").upper().strip()
    if trend_color != "GREEN":
        return None
    symbol_profiles = getattr(config, "CDC_VIXFIX_15M_SYMBOL_PROFILES", {})
    profile = symbol_profiles.get(symbol) if isinstance(symbol_profiles, dict) and isinstance(symbol_profiles.get(symbol), dict) else {}
    min_red_to_green_score = profile.get(
        "daily_best_min_red_to_green_score",
        getattr(config, "TELEGRAM_DAILY_BEST_PICK_CDC_MIN_RED_TO_GREEN_SCORE", 68.0),
    )
    max_bars_since_flip = profile.get(
        "daily_best_max_bars_since_flip",
        getattr(config, "TELEGRAM_DAILY_BEST_PICK_CDC_MAX_BARS_SINCE_GREEN_FLIP", 3),
    )
    require_reclaim = profile.get(
        "daily_best_require_reclaim",
        getattr(config, "TELEGRAM_DAILY_BEST_PICK_CDC_REQUIRE_RECLAIM", True),
    )
    try:
        min_red_to_green_score = float(min_red_to_green_score)
    except Exception:
        min_red_to_green_score = 68.0
    try:
        max_bars_since_flip = max(0, int(max_bars_since_flip))
    except Exception:
        max_bars_since_flip = 3
    red_to_green_score = cdc_plan.get("red_to_green_quality_score")
    green_flip_bars_since = cdc_plan.get("green_flip_bars_since")
    green_flip_reclaim = bool(cdc_plan.get("green_flip_reclaim"))
    try:
        red_to_green_score = float(red_to_green_score)
    except Exception:
        return None
    try:
        green_flip_bars_since = int(green_flip_bars_since)
    except Exception:
        return None
    if red_to_green_score < min_red_to_green_score:
        return None
    if green_flip_bars_since > max_bars_since_flip:
        return None
    if require_reclaim and green_flip_bars_since > 0 and not green_flip_reclaim:
        return None
    forecast_dir = str(cdc_plan.get("forecast_direction") or "").upper().strip()
    forecast_score = cdc_plan.get("forecast_score")
    try:
        forecast_score = float(forecast_score)
    except Exception:
        forecast_score = None
    if forecast_dir == "SELL" and isinstance(forecast_score, float) and forecast_score >= 60.0:
        return None
    confidence = plan_confidence_value(cdc_plan)
    if not isinstance(confidence, (int, float)):
        confidence = red_to_green_score
    confidence = max(float(confidence), min(95.0, float(red_to_green_score) - 2.0))
    analysis_points = list(cdc_plan.get("analysis_points") or [])
    analysis_points.insert(
        0,
        f"Daily Best Pick ใช้ CDC red->green score {float(red_to_green_score):.0f}/100 กับเหรียญใน Telegram Alerts",
    )
    if green_flip_reclaim:
        analysis_points.insert(1, "ราคายืนยันเหนือ high ของแท่งกลับตัวล่าสุดแล้ว")
    synthetic_plan = dict(cdc_plan)
    synthetic_plan.update(
        {
            "signal": "BUY",
            "confidence": min(95.0, float(confidence)),
            "source_label": "CDC+VixFix 15m Red->Green",
            "setup": "POST_FLIP_RECLAIM" if green_flip_reclaim else "POST_FLIP_FOLLOW_THROUGH",
            "reason": "Daily Best Pick อนุญาตจังหวะ BUY หลัง CDC พลิกเขียวและมี follow-through/reclaim ตาม threshold รายเหรียญ",
            "entry_price": synthetic_plan.get("current_price") or synthetic_plan.get("entry_price"),
            "bars_since_signal": green_flip_bars_since,
            "bars_since_entry": green_flip_bars_since,
            "daily_best_cdc_followthrough": True,
            "daily_best_cdc_symbol": symbol,
            "analysis_points": analysis_points[:6],
        }
    )
    return synthetic_plan


def build_daily_best_pick_candidates(results, *, config, helpers, get_now):
    normalize_symbol = helpers["normalize_symbol"]
    pick_primary_trade_plan = helpers["pick_primary_trade_plan"]
    get_best_confidence = helpers["get_best_confidence"]
    plan_confidence_value = helpers["plan_confidence_value"]
    collect_alert_sources = helpers["collect_alert_sources"]
    extract_signal_edge_metrics = helpers["extract_signal_edge_metrics"]
    get_primary_plan_source_label = helpers["get_primary_plan_source_label"]
    safe_float = helpers["safe_float"]
    pick_plan_value = helpers["pick_plan_value"]
    build_daily_best_pick_message = helpers["build_daily_best_pick_message"]
    evaluate_candidate_backtest_gate = helpers["evaluate_candidate_backtest_gate"]
    evaluate_candidate_symbol_strategy_gate = helpers["evaluate_candidate_symbol_strategy_gate"]
    candidate_edge_metrics = helpers["candidate_edge_metrics"]
    symbol_allowlist = _normalize_symbol_allowlist(
        getattr(config, "TELEGRAM_DAILY_BEST_PICK_SYMBOL_ALLOWLIST", ()),
        normalize_symbol,
    )

    min_conf = getattr(config, "TELEGRAM_DAILY_BEST_PICK_MIN_CONFIDENCE", 58.0)
    min_score = getattr(config, "TELEGRAM_DAILY_BEST_PICK_MIN_SCORE", 74.0)
    max_per_day = getattr(config, "TELEGRAM_DAILY_BEST_PICK_MAX_PER_DAY", 1)
    require_quality = bool(getattr(config, "TELEGRAM_DAILY_BEST_PICK_REQUIRE_QUALITY", True))
    allow_cdc = bool(getattr(config, "TELEGRAM_DAILY_BEST_PICK_ALLOW_CDC", False))
    relaxed_enable = bool(getattr(config, "TELEGRAM_DAILY_BEST_PICK_RELAXED_ENABLE", True))
    relaxed_min_conf = getattr(config, "TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_CONFIDENCE", 57.0)
    relaxed_min_score = getattr(config, "TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_SCORE", 68.0)
    relaxed_min_wr = getattr(config, "TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_HIST_WIN_RATE", 55.0)
    relaxed_min_trades = getattr(config, "TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_HIST_TRADES", 4)
    relaxed_min_exp = getattr(config, "TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_EXPECTANCY_RR", 0.0)
    try:
        min_conf = float(min_conf)
    except Exception:
        min_conf = 55.0
    try:
        min_score = float(min_score)
    except Exception:
        min_score = 72.0
    try:
        max_per_day = int(max_per_day)
    except Exception:
        max_per_day = 3
    try:
        relaxed_min_conf = float(relaxed_min_conf)
    except Exception:
        relaxed_min_conf = 57.0
    try:
        relaxed_min_score = float(relaxed_min_score)
    except Exception:
        relaxed_min_score = 68.0
    try:
        relaxed_min_wr = float(relaxed_min_wr)
    except Exception:
        relaxed_min_wr = 55.0
    try:
        relaxed_min_trades = int(relaxed_min_trades)
    except Exception:
        relaxed_min_trades = 4
    try:
        relaxed_min_exp = float(relaxed_min_exp)
    except Exception:
        relaxed_min_exp = 0.0
    if max_per_day < 1:
        max_per_day = 1

    candidates = []
    relaxed_candidates = []
    today_key = get_now().strftime("%Y%m%d")
    for item in results or []:
        if not isinstance(item, dict) or item.get("error"):
            continue
        symbol = normalize_symbol(item.get("symbol") or "")
        if not symbol:
            continue
        if symbol_allowlist and symbol not in symbol_allowlist:
            continue
        top_signal = str(item.get("signal") or "").upper().strip()
        for signal in ("BUY", "SELL"):
            active_require_quality = require_quality
            synthetic_cdc_plan = _build_daily_best_cdc_followthrough_plan(
                item,
                config=config,
                helpers=helpers,
                signal=signal,
            )
            primary_plan = pick_primary_trade_plan(
                item,
                signal=signal,
                require_quality=active_require_quality,
                allow_cdc=allow_cdc,
            )
            if not isinstance(primary_plan, dict) and relaxed_enable:
                active_require_quality = False
                primary_plan = pick_primary_trade_plan(
                    item,
                    signal=signal,
                    require_quality=False,
                    allow_cdc=allow_cdc,
                )
            if not isinstance(primary_plan, dict) and isinstance(synthetic_cdc_plan, dict):
                active_require_quality = False
                primary_plan = synthetic_cdc_plan
            if not isinstance(primary_plan, dict):
                continue
            best_conf = get_best_confidence(
                item,
                signal=signal,
                require_quality=active_require_quality,
                allow_cdc=allow_cdc,
            )
            if isinstance(primary_plan, dict) and bool(primary_plan.get("daily_best_cdc_followthrough")):
                synthetic_conf = plan_confidence_value(primary_plan)
                if isinstance(synthetic_conf, (int, float)):
                    best_conf = max(float(best_conf or 0.0), float(synthetic_conf))
            if best_conf is None:
                best_conf = plan_confidence_value(primary_plan)
            if best_conf is None or float(best_conf) < float(relaxed_min_conf):
                continue
            source_floor = max(45.0, min(float(best_conf), float(min_conf)))
            sources = collect_alert_sources(
                item,
                source_floor,
                signal=signal,
                require_quality=active_require_quality,
                allow_cdc=allow_cdc,
            )
            if bool(primary_plan.get("daily_best_cdc_followthrough")):
                cdc_source = f"CDC+VixFix 15m Red->Green ({float(best_conf):.0f}%)"
                if not any("CDC+VixFix 15m" in str(row) for row in sources):
                    sources = [cdc_source] + list(sources)
                sources = sources[:3]
            edge = extract_signal_edge_metrics(primary_plan, signal)
            strategy_label = get_primary_plan_source_label(item, primary_plan)
            score = float(best_conf)
            if top_signal == signal:
                score += 6.0
            bars_since = safe_float(
                pick_plan_value(primary_plan, ["bars_since_signal", "bars_since_entry"]),
                None,
            )
            if isinstance(bars_since, float):
                if bars_since <= 0:
                    score += 8.0
                elif bars_since <= 1:
                    score += 5.0
                elif bars_since <= 2:
                    score += 2.0
                elif bars_since >= 6:
                    score -= 4.0
            trend_alignment = primary_plan.get("trend_alignment")
            if isinstance(trend_alignment, bool):
                score += 4.0 if trend_alignment else -6.0
            forecast_dir = str(
                primary_plan.get("forecast_direction")
                or ((item.get("price_forecast") or {}).get("direction") if isinstance(item.get("price_forecast"), dict) else "")
                or ""
            ).upper().strip()
            if forecast_dir == signal:
                score += 5.0
            wr = edge.get("win_rate_pct")
            exp = edge.get("expectancy_rr")
            trades = edge.get("trades")
            if isinstance(wr, (int, float)):
                score += max(-4.0, min(10.0, (float(wr) - 50.0) * 0.25))
            if isinstance(exp, (int, float)):
                score += max(-4.0, min(8.0, float(exp) * 8.0))
            if isinstance(trades, (int, float)):
                score += max(0.0, min(4.0, float(trades) / 8.0))
            if bool(primary_plan.get("daily_best_cdc_followthrough")):
                red_to_green_score = safe_float(primary_plan.get("red_to_green_quality_score"), None)
                if isinstance(red_to_green_score, float):
                    score += max(0.0, min(10.0, (float(red_to_green_score) - 65.0) * 0.35))
                if bool(primary_plan.get("green_flip_reclaim")):
                    score += 5.0
            score += min(6.0, float(len(sources)) * 1.5)
            message = build_daily_best_pick_message(
                item,
                signal,
                float(best_conf),
                sources,
                primary_plan=primary_plan,
                strategy_label=strategy_label,
                selection_score=score,
                mode_label="Strict Daily Pick",
            )
            if not isinstance(message, str) or not message.strip():
                continue
            candidate = {
                "symbol": symbol,
                "signal": signal,
                "strategy": "DAILY_BEST",
                "score": float(score),
                "confidence": float(best_conf),
                "plan": primary_plan,
                "item": item,
                "edge_metrics": edge,
                "source_count": len(sources),
                "message": message,
                "strategy_label": strategy_label,
            }
            gate_ok, _, normalized_edge = evaluate_candidate_backtest_gate(candidate)
            if gate_ok and float(best_conf) >= float(min_conf):
                candidate["edge_metrics"] = normalized_edge
                profile_ok, _, profile_edge = evaluate_candidate_symbol_strategy_gate(candidate)
                if profile_ok:
                    if isinstance(profile_edge, dict) and profile_edge:
                        candidate["edge_metrics"] = profile_edge
                    candidates.append(candidate)
                    continue
            if not relaxed_enable:
                continue
            relaxed_edge = normalized_edge if isinstance(normalized_edge, dict) else candidate_edge_metrics(candidate)
            relaxed_wr = relaxed_edge.get("win_rate_pct")
            relaxed_exp = relaxed_edge.get("expectancy_rr")
            relaxed_trades = relaxed_edge.get("trades")
            if float(score) < float(relaxed_min_score):
                continue
            if not isinstance(relaxed_wr, (int, float)) or float(relaxed_wr) < float(relaxed_min_wr):
                continue
            if not isinstance(relaxed_trades, (int, float)) or float(relaxed_trades) < float(relaxed_min_trades):
                continue
            if isinstance(relaxed_exp, (int, float)) and float(relaxed_exp) < float(relaxed_min_exp):
                continue
            relaxed_sources = sources[:]
            if not relaxed_sources and strategy_label:
                relaxed_sources = [str(strategy_label)]
            relaxed_message = build_daily_best_pick_message(
                item,
                signal,
                float(best_conf),
                relaxed_sources,
                primary_plan=primary_plan,
                strategy_label=strategy_label,
                selection_score=score,
                mode_label="Trend Pick",
            )
            if not isinstance(relaxed_message, str) or not relaxed_message.strip():
                continue
            relaxed_candidate = dict(candidate)
            relaxed_candidate["message"] = relaxed_message
            relaxed_candidate["edge_metrics"] = relaxed_edge
            relaxed_candidate["score"] = float(score)
            relaxed_candidate["strategy_label"] = (str(strategy_label or "").strip() + " | Trend Pick").strip(" |")
            relaxed_candidate["cache_key"] = f"DAILYBESTRELAX|{symbol}|{signal}|{today_key}"
            relaxed_candidate["daily_best_mode"] = "relaxed"
            profile_ok, _, profile_edge = evaluate_candidate_symbol_strategy_gate(relaxed_candidate)
            if not profile_ok:
                continue
            if isinstance(profile_edge, dict) and profile_edge:
                relaxed_candidate["edge_metrics"] = profile_edge
            relaxed_candidates.append(relaxed_candidate)
    if not candidates:
        candidates = relaxed_candidates
    elif relaxed_enable:
        used_symbols = {normalize_symbol(row.get("symbol") or "") for row in candidates if isinstance(row, dict)}
        for row in relaxed_candidates:
            sym = normalize_symbol(row.get("symbol") or "")
            if not sym or sym in used_symbols:
                continue
            candidates.append(row)
    if not candidates:
        return []
    candidates.sort(key=lambda c: (float(c.get("score", 0.0)), float(c.get("confidence", 0.0))), reverse=True)
    selected = []
    seen_symbols = set()
    for candidate in candidates:
        symbol = str(candidate.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen_symbols:
            continue
        candidate_mode = str(candidate.get("daily_best_mode") or "strict").strip().lower()
        if candidate_mode == "relaxed":
            candidate_min_score = float(relaxed_min_score)
        else:
            candidate_min_score = float(min_score)
        if float(candidate.get("score", 0.0)) < candidate_min_score:
            continue
        selected.append(candidate)
        seen_symbols.add(symbol)
        if len(selected) >= max_per_day:
            break
    return selected
