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
    candidate_edge_metrics = helpers["candidate_edge_metrics"]

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
    baseline_enable = bool(getattr(config, "TELEGRAM_DAILY_BEST_PICK_BASELINE_ENABLE", True))
    baseline_min_conf = getattr(config, "TELEGRAM_DAILY_BEST_PICK_BASELINE_MIN_CONFIDENCE", 54.0)
    baseline_min_score = getattr(config, "TELEGRAM_DAILY_BEST_PICK_BASELINE_MIN_SCORE", 60.0)
    baseline_min_wr = getattr(config, "TELEGRAM_DAILY_BEST_PICK_BASELINE_MIN_HIST_WIN_RATE", 52.0)
    baseline_min_trades = getattr(config, "TELEGRAM_DAILY_BEST_PICK_BASELINE_MIN_HIST_TRADES", 2)
    baseline_min_exp = getattr(config, "TELEGRAM_DAILY_BEST_PICK_BASELINE_MIN_EXPECTANCY_RR", -0.02)
    baseline_target_per_day = getattr(config, "TELEGRAM_DAILY_BEST_PICK_BASELINE_TARGET_PER_DAY", 1)
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
    try:
        baseline_min_conf = float(baseline_min_conf)
    except Exception:
        baseline_min_conf = 54.0
    try:
        baseline_min_score = float(baseline_min_score)
    except Exception:
        baseline_min_score = 60.0
    try:
        baseline_min_wr = float(baseline_min_wr)
    except Exception:
        baseline_min_wr = 52.0
    try:
        baseline_min_trades = int(baseline_min_trades)
    except Exception:
        baseline_min_trades = 2
    try:
        baseline_min_exp = float(baseline_min_exp)
    except Exception:
        baseline_min_exp = -0.02
    try:
        baseline_target_per_day = int(baseline_target_per_day)
    except Exception:
        baseline_target_per_day = 1
    if max_per_day < 1:
        max_per_day = 1
    if baseline_target_per_day < 0:
        baseline_target_per_day = 0

    candidates = []
    relaxed_candidates = []
    baseline_candidates = []
    today_key = get_now().strftime("%Y%m%d")
    for item in results or []:
        if not isinstance(item, dict) or item.get("error"):
            continue
        symbol = normalize_symbol(item.get("symbol") or "")
        if not symbol:
            continue
        top_signal = str(item.get("signal") or "").upper().strip()
        for signal in ("BUY", "SELL"):
            active_require_quality = require_quality
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
            if not isinstance(primary_plan, dict):
                continue
            best_conf = get_best_confidence(
                item,
                signal=signal,
                require_quality=active_require_quality,
                allow_cdc=allow_cdc,
            )
            if best_conf is None:
                best_conf = plan_confidence_value(primary_plan)
            if best_conf is None or float(best_conf) < float(min(relaxed_min_conf, baseline_min_conf)):
                continue
            source_floor = max(45.0, min(float(best_conf), float(min_conf)))
            sources = collect_alert_sources(
                item,
                source_floor,
                signal=signal,
                require_quality=active_require_quality,
                allow_cdc=allow_cdc,
            )
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
                "message": message,
                "strategy_label": strategy_label,
            }
            gate_ok, _, normalized_edge = evaluate_candidate_backtest_gate(candidate)
            if gate_ok and float(best_conf) >= float(min_conf):
                candidate["edge_metrics"] = normalized_edge
                candidates.append(candidate)
                continue
            if not relaxed_enable:
                continue
            relaxed_edge = normalized_edge if isinstance(normalized_edge, dict) else candidate_edge_metrics(candidate)
            relaxed_wr = relaxed_edge.get("win_rate_pct")
            relaxed_exp = relaxed_edge.get("expectancy_rr")
            relaxed_trades = relaxed_edge.get("trades")
            baseline_sources = sources[:]
            if not baseline_sources and strategy_label:
                baseline_sources = [str(strategy_label)]
            if baseline_enable:
                baseline_ok = True
                if float(score) < float(baseline_min_score):
                    baseline_ok = False
                if not isinstance(relaxed_wr, (int, float)) or float(relaxed_wr) < float(baseline_min_wr):
                    baseline_ok = False
                if float(baseline_min_trades) > 0:
                    baseline_trade_ok = isinstance(relaxed_trades, (int, float)) and float(relaxed_trades) >= float(baseline_min_trades)
                    if not baseline_trade_ok:
                        baseline_trade_ok = (
                            isinstance(relaxed_wr, (int, float))
                            and float(relaxed_wr) >= max(float(baseline_min_wr) + 4.0, 56.0)
                            and float(best_conf) >= max(float(baseline_min_conf) + 4.0, 58.0)
                            and float(score) >= float(baseline_min_score) + 3.0
                        )
                    if not baseline_trade_ok:
                        baseline_ok = False
                if isinstance(relaxed_exp, (int, float)) and float(relaxed_exp) < float(baseline_min_exp):
                    baseline_ok = False
                if float(best_conf) < float(baseline_min_conf):
                    baseline_ok = False
                if baseline_ok:
                    baseline_message = build_daily_best_pick_message(
                        item,
                        signal,
                        float(best_conf),
                        baseline_sources,
                        primary_plan=primary_plan,
                        strategy_label=strategy_label,
                        selection_score=score,
                        mode_label="Baseline Trend Pick",
                    )
                    if isinstance(baseline_message, str) and baseline_message.strip():
                        baseline_candidate = dict(candidate)
                        baseline_candidate["message"] = baseline_message
                        baseline_candidate["edge_metrics"] = relaxed_edge
                        baseline_candidate["score"] = float(score)
                        baseline_candidate["strategy_label"] = (str(strategy_label or "").strip() + " | Baseline Trend").strip(" |")
                        baseline_candidate["cache_key"] = f"DAILYBESTBASE|{symbol}|{signal}|{today_key}"
                        baseline_candidate["daily_best_mode"] = "baseline"
                        baseline_candidates.append(baseline_candidate)
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
    if baseline_enable and baseline_candidates:
        used_symbols = {normalize_symbol(row.get("symbol") or "") for row in candidates if isinstance(row, dict)}
        baseline_candidates.sort(key=lambda c: (float(c.get("score", 0.0)), float(c.get("confidence", 0.0))), reverse=True)
        baseline_added = 0
        for row in baseline_candidates:
            if baseline_target_per_day > 0 and baseline_added >= baseline_target_per_day:
                break
            sym = normalize_symbol(row.get("symbol") or "")
            if not sym or sym in used_symbols:
                continue
            candidates.append(row)
            used_symbols.add(sym)
            baseline_added += 1
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
        if candidate_mode == "baseline":
            candidate_min_score = float(baseline_min_score)
        elif candidate_mode == "relaxed":
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


def build_cdc_daily_trend_candidates(results, *, config, helpers, get_now, existing_candidates=None, min_conf=None):
    normalize_symbol = helpers["normalize_symbol"]
    plan_trade_direction = helpers["plan_trade_direction"]
    plan_confidence_value = helpers["plan_confidence_value"]
    candidate_edge_metrics = helpers["candidate_edge_metrics"]
    build_cdc_vixfix_message = helpers["build_cdc_vixfix_message"]

    enabled = bool(getattr(config, "CDC_VIXFIX_15M_DAILY_TREND_ENABLE", True))
    if not enabled:
        return []
    max_per_day = getattr(config, "CDC_VIXFIX_15M_DAILY_TREND_MAX_PER_DAY", 1)
    min_confidence = getattr(config, "CDC_VIXFIX_15M_DAILY_TREND_MIN_CONFIDENCE", 56.0)
    min_win_rate = getattr(config, "CDC_VIXFIX_15M_DAILY_TREND_MIN_WIN_RATE", 54.0)
    min_score = getattr(config, "CDC_VIXFIX_15M_DAILY_TREND_MIN_SCORE", 61.0)
    try:
        max_per_day = max(1, int(max_per_day))
    except Exception:
        max_per_day = 1
    try:
        min_confidence = float(min_confidence)
    except Exception:
        min_confidence = 56.0
    try:
        min_win_rate = float(min_win_rate)
    except Exception:
        min_win_rate = 54.0
    try:
        min_score = float(min_score)
    except Exception:
        min_score = 61.0
    if isinstance(existing_candidates, list):
        for row in existing_candidates:
            if isinstance(row, dict) and str(row.get("strategy") or "").upper() == "CDCVIX15":
                return []
    candidates = []
    today_key = get_now().strftime("%Y%m%d")
    for item in results or []:
        if not isinstance(item, dict) or item.get("error"):
            continue
        symbol = normalize_symbol(item.get("symbol") or "")
        cdc_plan = item.get("cdc_vixfix_15m")
        if not symbol or not isinstance(cdc_plan, dict):
            continue
        signal = plan_trade_direction(cdc_plan)
        if signal not in ("BUY", "SELL"):
            continue
        confidence = plan_confidence_value(cdc_plan)
        if not isinstance(confidence, (int, float)) or float(confidence) < float(min_confidence):
            continue
        candidate = {
            "symbol": symbol,
            "strategy": "CDCVIX15",
            "signal": signal,
            "plan": cdc_plan,
            "item": item,
            "confidence": float(confidence),
        }
        edge = candidate_edge_metrics(candidate)
        win_rate = edge.get("win_rate_pct")
        expectancy = edge.get("expectancy_rr")
        trades = edge.get("trades")
        if not isinstance(win_rate, (int, float)) or float(win_rate) < float(min_win_rate):
            continue
        score = float(confidence)
        forecast_dir = str(cdc_plan.get("forecast_direction") or "").upper().strip()
        if forecast_dir == signal:
            score += 5.0
        trigger = str(cdc_plan.get("sell_trigger") or cdc_plan.get("exit_trigger") or "").upper().strip()
        if trigger:
            score += 3.0
        if isinstance(win_rate, (int, float)):
            score += max(-3.0, min(8.0, (float(win_rate) - 50.0) * 0.25))
        if isinstance(expectancy, (int, float)):
            score += max(-3.0, min(6.0, float(expectancy) * 8.0))
        if isinstance(trades, (int, float)):
            score += max(0.0, min(4.0, float(trades) / 6.0))
        if float(score) < float(min_score):
            continue
        message = build_cdc_vixfix_message(item, cdc_plan, mode_label="Daily Trend")
        if not isinstance(message, str) or not message.strip():
            continue
        candidate.update(
            {
                "score": float(score),
                "edge_metrics": edge,
                "message": message,
                "cache_key": f"CDCVIX15DAILY|{symbol}|{signal}|{today_key}",
                "cdc_mode": "daily_trend",
            }
        )
        candidates.append(candidate)
    candidates.sort(key=lambda c: (float(c.get("score", 0.0)), float(c.get("confidence", 0.0))), reverse=True)
    selected = []
    seen_symbols = set()
    for candidate in candidates:
        symbol = normalize_symbol(candidate.get("symbol") or "")
        if not symbol or symbol in seen_symbols:
            continue
        selected.append(candidate)
        seen_symbols.add(symbol)
        if len(selected) >= max_per_day:
            break
    return selected

