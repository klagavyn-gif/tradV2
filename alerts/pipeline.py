import json
from collections import Counter


def build_telegram_candidates(results, min_conf, *, config, helpers, get_now):
    actionzone_precision60_profile = helpers["actionzone_precision60_profile"]
    strict_60_mode_enabled = helpers["strict_60_mode_enabled"]
    strict_60_allow_cdc = helpers["strict_60_allow_cdc"]
    normalize_symbol = helpers["normalize_symbol"]
    evaluate_super_signal = helpers["evaluate_super_signal"]
    pick_primary_trade_plan = helpers["pick_primary_trade_plan"]
    build_super_signal_message = helpers["build_super_signal_message"]
    build_all_weather_signal = helpers["build_all_weather_signal"]
    build_all_weather_message = helpers["build_all_weather_message"]
    normalize_confidence = helpers["normalize_confidence"]
    build_cdc_vixfix_message = helpers["build_cdc_vixfix_message"]
    extract_plan_edge_metrics = helpers["extract_plan_edge_metrics"]
    alert_profile_score_adjustment = helpers["alert_profile_score_adjustment"]
    format_price_value = helpers["format_price_value"]
    evaluate_entry_quality_gate = helpers["evaluate_entry_quality_gate"]
    build_actionzone_message = helpers["build_actionzone_message"]
    safe_float = helpers["safe_float"]
    build_price_action_message = helpers["build_price_action_message"]
    build_trend_breakout_message = helpers["build_trend_breakout_message"]
    get_best_confidence = helpers["get_best_confidence"]
    collect_alert_sources = helpers["collect_alert_sources"]
    get_primary_plan_source_label = helpers["get_primary_plan_source_label"]
    pick_plan_value = helpers["pick_plan_value"]
    build_telegram_message = helpers["build_telegram_message"]
    evaluate_candidate_backtest_gate = helpers["evaluate_candidate_backtest_gate"]
    evaluate_candidate_symbol_strategy_gate = helpers["evaluate_candidate_symbol_strategy_gate"]
    candidate_alert_profile = helpers["candidate_alert_profile"]

    candidates = []
    quality_drop_counts = Counter()
    precision60 = actionzone_precision60_profile()
    strict_60 = strict_60_mode_enabled()
    allow_cdc = strict_60_allow_cdc()
    hour_key = get_now().strftime("%Y%m%d%H")

    for item in results:
        if not isinstance(item, dict) or item.get("error"):
            continue
        symbol = normalize_symbol(item.get("symbol") or "")
        if not symbol:
            continue

        for sig in ("BUY", "SELL"):
            ss_ok, ss_reason, ss_meta = evaluate_super_signal(item, sig)
            if ss_ok:
                ss_plan = pick_primary_trade_plan(
                    item,
                    signal=sig,
                    require_quality=strict_60,
                    allow_cdc=allow_cdc,
                )
                ss_message = build_super_signal_message(item, sig, ss_meta, primary_plan=ss_plan)
                if ss_message:
                    score = 1000.0 + (float(ss_meta.get("avg_wr", 0)) * 2.0)
                    candidates.append(
                        {
                            "symbol": symbol,
                            "strategy": "SS15",
                            "signal": sig,
                            "score": score,
                            "confidence": float(ss_meta.get("avg_wr", 0)),
                            "plan": ss_plan,
                            "item": item,
                            "edge_metrics": ss_meta,
                            "message": ss_message,
                            "cache_key": f"SS15|{symbol}|{sig}|{hour_key}",
                        }
                    )

        aw_signal, aw_meta = build_all_weather_signal(item, min_conf)
        if isinstance(aw_signal, dict):
            aw_message = build_all_weather_message(item, aw_signal)
            if aw_message:
                candidates.append(
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
                        "cache_key": str(aw_signal.get("cache_key") or f"AW15|{symbol}|{hour_key}"),
                    }
                )
        elif isinstance(aw_meta, dict):
            quality_drop_counts[f"all_weather_{aw_meta.get('status') or 'filtered'}"] += 1

        cdc_plan = item.get("cdc_vixfix_15m")
        if isinstance(cdc_plan, dict):
            cdc_signal = str(cdc_plan.get("signal") or "").upper()
            cdc_conf = normalize_confidence(cdc_plan.get("confidence"))
            cdc_min_conf = getattr(config, "CDC_VIXFIX_15M_MIN_ALERT_CONFIDENCE", min_conf)
            try:
                cdc_min_conf = float(cdc_min_conf)
            except Exception:
                cdc_min_conf = float(min_conf)
            required_conf = min(float(min_conf), float(cdc_min_conf))
            if cdc_signal in ("BUY", "SELL") and cdc_conf is not None and cdc_conf >= required_conf:
                cdc_message = build_cdc_vixfix_message(item, cdc_plan)
                if cdc_message:
                    edge = extract_plan_edge_metrics(cdc_plan)
                    freshness = 6.0
                    last_signal_time = str(cdc_plan.get("last_signal_time") or "").strip()
                    if not last_signal_time:
                        freshness = 2.0
                    trigger = str(cdc_plan.get("sell_trigger") or "").upper().strip()
                    score = float(cdc_conf) + freshness + (6.0 if cdc_signal == "SELL" else 4.0)
                    score += alert_profile_score_adjustment(
                        win_rate=edge.get("win_rate_pct"),
                        confidence=cdc_conf,
                        expectancy=edge.get("expectancy_rr"),
                        trades=edge.get("trades"),
                    )
                    if cdc_signal == "SELL" and cdc_conf >= 85.0:
                        score += 8.0
                    context_key = last_signal_time or trigger or (format_price_value(cdc_plan.get("entry_price")) or "na")
                    candidates.append(
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
                        }
                    )

        az_plan = item.get("actionzone_15m")
        if isinstance(az_plan, dict):
            az_signal = str(az_plan.get("signal") or "").upper()
            if az_signal in ("BUY", "SELL") and az_plan.get("alert"):
                az_min_conf = getattr(config, "ACTIONZONE_15M_MIN_ALERT_CONFIDENCE", min_conf)
                try:
                    az_min_conf = float(az_min_conf)
                except Exception:
                    az_min_conf = float(min_conf)
                required_az_conf = max(float(min_conf), float(az_min_conf))
                if isinstance(precision60, dict):
                    required_az_conf = max(required_az_conf, float(precision60.get("min_conf", required_az_conf)))
                az_conf = normalize_confidence(az_plan.get("confidence"))
                if az_conf is not None and az_conf >= required_az_conf:
                    gate_ok, gate_reason, edge = evaluate_entry_quality_gate(az_plan, az_signal)
                    if not gate_ok:
                        quality_drop_counts[gate_reason] += 1
                        continue
                    az_message = build_actionzone_message(item, az_plan)
                    if az_message:
                        trend_alignment = bool(az_plan.get("trend_alignment", True))
                        bars_since = safe_float(az_plan.get("bars_since_signal"), None)
                        if isinstance(precision60, dict):
                            if bool(precision60.get("require_trend_alignment", True)) and not trend_alignment:
                                continue
                            if isinstance(bars_since, float) and bars_since > float(precision60.get("max_bars_since_signal", 0)):
                                continue
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
                        wr = edge.get("win_rate_pct")
                        exp = edge.get("expectancy_rr")
                        trades = edge.get("trades")
                        score += alert_profile_score_adjustment(
                            win_rate=wr,
                            confidence=az_conf,
                            expectancy=exp,
                            trades=trades,
                        )
                        if isinstance(wr, (int, float)):
                            score += max(-3.0, min(8.0, (float(wr) - 50.0) * 0.20))
                        if isinstance(exp, (int, float)):
                            score += max(-4.0, min(8.0, float(exp) * 8.0))
                        if isinstance(trades, (int, float)):
                            score += max(0.0, min(4.0, float(trades) / 8.0))
                        if az_signal == "SELL" and isinstance(wr, (int, float)) and wr >= 65.0:
                            score += 12.0
                        last_signal_time = str(az_plan.get("last_signal_time") or "").strip()
                        zone = str(az_plan.get("zone") or "").upper().strip()
                        entry_bucket = format_price_value(az_plan.get("entry_price")) or "na"
                        context_key = last_signal_time or f"{zone}|{entry_bucket}"
                        candidates.append(
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
                            }
                        )

        pa_plan = item.get("price_action_15m")
        if isinstance(pa_plan, dict):
            pa_signal = str(pa_plan.get("signal") or "").upper()
            pa_conf = normalize_confidence(pa_plan.get("confidence"))
            pa_min_conf = safe_float(getattr(config, "PRICE_ACTION_15M_MIN_ALERT_CONFIDENCE", min_conf), float(min_conf))
            pa_min_score = safe_float(getattr(config, "PRICE_ACTION_15M_MIN_SCORE", 68.0), 68.0)
            pa_score = safe_float(pa_plan.get("score"), 0.0)
            if pa_signal in ("BUY", "SELL") and pa_plan.get("alert") and pa_conf is not None and pa_conf >= pa_min_conf and pa_score >= pa_min_score:
                gate_ok, gate_reason, edge = evaluate_entry_quality_gate(pa_plan, pa_signal)
                if not gate_ok:
                    quality_drop_counts[f"price_action_{gate_reason}"] += 1
                else:
                    pa_message = build_price_action_message(item, pa_plan)
                    if pa_message:
                        score = float(pa_score)
                        score += alert_profile_score_adjustment(
                            win_rate=edge.get("win_rate_pct"),
                            confidence=pa_conf,
                            expectancy=edge.get("expectancy_rr"),
                            trades=edge.get("trades"),
                        )
                        if isinstance(edge.get("win_rate_pct"), (int, float)):
                            score += max(-3.0, min(8.0, (float(edge["win_rate_pct"]) - 50.0) * 0.20))
                        if isinstance(edge.get("expectancy_rr"), (int, float)):
                            score += max(-4.0, min(8.0, float(edge["expectancy_rr"]) * 8.0))
                        if isinstance(edge.get("trades"), (int, float)):
                            score += max(0.0, min(4.0, float(edge["trades"]) / 8.0))
                        context_key = "|".join(
                            [
                                str(pa_plan.get("last_signal_time") or ""),
                                str(pa_plan.get("market_structure") or ""),
                                str(pa_plan.get("chart_pattern") or pa_plan.get("detected_pattern") or ""),
                            ]
                        ).strip("|") or (format_price_value(pa_plan.get("entry_price")) or "na")
                        candidates.append(
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
                            }
                        )

        tcb_plan = item.get("trend_breakout_15m")
        if isinstance(tcb_plan, dict):
            tcb_signal = str(tcb_plan.get("signal") or "").upper()
            tcb_conf = normalize_confidence(tcb_plan.get("confidence"))
            tcb_min_conf = safe_float(getattr(config, "TREND_BREAKOUT_15M_MIN_ALERT_CONFIDENCE", min_conf), float(min_conf))
            tcb_min_score = safe_float(getattr(config, "TREND_BREAKOUT_15M_MIN_SCORE", 68.0), 68.0)
            tcb_score = safe_float(tcb_plan.get("score"), 0.0)
            if tcb_signal in ("BUY", "SELL") and tcb_plan.get("alert") and tcb_conf is not None and tcb_conf >= tcb_min_conf and tcb_score >= tcb_min_score:
                gate_ok, gate_reason, edge = evaluate_entry_quality_gate(tcb_plan, tcb_signal)
                if not gate_ok:
                    quality_drop_counts[f"trend_breakout_{gate_reason}"] += 1
                else:
                    tcb_message = build_trend_breakout_message(item, tcb_plan)
                    if tcb_message:
                        score = float(tcb_score)
                        score += alert_profile_score_adjustment(
                            win_rate=edge.get("win_rate_pct"),
                            confidence=tcb_conf,
                            expectancy=edge.get("expectancy_rr"),
                            trades=edge.get("trades"),
                        )
                        if isinstance(edge.get("win_rate_pct"), (int, float)):
                            score += max(-3.0, min(8.0, (float(edge["win_rate_pct"]) - 50.0) * 0.20))
                        if isinstance(edge.get("expectancy_rr"), (int, float)):
                            score += max(-4.0, min(8.0, float(edge["expectancy_rr"]) * 8.0))
                        if isinstance(edge.get("trades"), (int, float)):
                            score += max(0.0, min(4.0, float(edge["trades"]) / 8.0))
                        context_key = "|".join(
                            [
                                str(tcb_plan.get("last_signal_time") or ""),
                                str(tcb_plan.get("trend_1h") or ""),
                                str(tcb_plan.get("breakout_level") or ""),
                            ]
                        ).strip("|") or (format_price_value(tcb_plan.get("entry_price")) or "na")
                        candidates.append(
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
                            }
                        )

        signal = str(item.get("signal") or "").upper()
        if signal in ("BUY", "SELL"):
            best_conf = get_best_confidence(item, signal=signal, require_quality=strict_60, allow_cdc=allow_cdc)
            if best_conf is not None and best_conf >= min_conf:
                sources = collect_alert_sources(
                    item,
                    min_conf,
                    signal=signal,
                    require_quality=strict_60,
                    allow_cdc=allow_cdc,
                )
                primary_plan = pick_primary_trade_plan(
                    item,
                    signal=signal,
                    require_quality=strict_60,
                    allow_cdc=allow_cdc,
                )
                if not isinstance(primary_plan, dict):
                    quality_drop_counts["no_primary_plan_after_strict_gate"] += 1
                    continue
                gate_ok, gate_reason, edge = evaluate_entry_quality_gate(primary_plan, signal)
                if not gate_ok:
                    quality_drop_counts[gate_reason] += 1
                    continue
                min_sources = getattr(config, "TELEGRAM_ALERT_PRIMARY_MIN_SOURCES", 2)
                single_source_min_conf = getattr(config, "TELEGRAM_ALERT_PRIMARY_SINGLE_SOURCE_MIN_CONF", 90.0)
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
                    continue
                message = build_telegram_message(item, signal, best_conf, sources, primary_plan=primary_plan)
                if message:
                    source_bonus = min(8.0, float(len(sources)) * 1.5)
                    score = float(best_conf) + source_bonus
                    wr = edge.get("win_rate_pct")
                    exp = edge.get("expectancy_rr")
                    trades = edge.get("trades")
                    score += alert_profile_score_adjustment(
                        win_rate=wr,
                        confidence=best_conf,
                        expectancy=exp,
                        trades=trades,
                    )
                    if isinstance(wr, (int, float)):
                        score += max(-3.0, min(8.0, (float(wr) - 50.0) * 0.20))
                    if isinstance(exp, (int, float)):
                        score += max(-4.0, min(8.0, float(exp) * 8.0))
                    if isinstance(trades, (int, float)):
                        score += max(0.0, min(4.0, float(trades) / 8.0))
                    last_signal_time = str(primary_plan.get("last_signal_time") or "").strip() if isinstance(primary_plan, dict) else ""
                    entry_bucket = format_price_value(pick_plan_value(primary_plan, ["entry_price", "current_price", "price"])) if isinstance(primary_plan, dict) else None
                    pattern_bucket = str(primary_plan.get("detected_pattern") or "").strip().upper() if isinstance(primary_plan, dict) else ""
                    source_bucket = get_primary_plan_source_label(item, primary_plan) if isinstance(primary_plan, dict) else "PRIMARY"
                    context_key = last_signal_time or "|".join([source_bucket, entry_bucket or "na", pattern_bucket or "NOPATTERN"])
                    candidates.append(
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
                        }
                    )

    filtered_candidates = []
    for candidate in candidates:
        gate_ok, gate_reason, edge_metrics = evaluate_candidate_backtest_gate(candidate)
        if not gate_ok:
            quality_drop_counts[gate_reason] += 1
            continue
        candidate["edge_metrics"] = edge_metrics
        profile_ok, profile_reason, profile_metrics = evaluate_candidate_symbol_strategy_gate(candidate)
        if not profile_ok:
            quality_drop_counts[profile_reason] += 1
            continue
        if isinstance(profile_metrics, dict) and profile_metrics:
            candidate["edge_metrics"] = profile_metrics
        candidate["alert_profile"] = candidate_alert_profile(candidate)
        filtered_candidates.append(candidate)

    stats = {
        "quality_drop_counts": dict(quality_drop_counts),
    }
    return filtered_candidates, stats


def notify_telegram_from_results(results, *, config, helpers, get_now, logger):
    telegram_kill_switch_state = helpers["telegram_kill_switch_state"]
    telegram_dynamic_conf_threshold = helpers["telegram_dynamic_conf_threshold"]
    build_telegram_candidates = helpers["build_telegram_candidates"]
    is_daily_best_pick_window = helpers["is_daily_best_pick_window"]
    build_daily_best_pick_candidates = helpers["build_daily_best_pick_candidates"]
    build_daily_summary_message = helpers["build_daily_summary_message"]
    send_telegram_alert = helpers["send_telegram_alert"]
    telegram_alert_cache = helpers["telegram_alert_cache"]
    record_telegram_alert_history = helpers["record_telegram_alert_history"]
    track_alert_performance = helpers["track_alert_performance"]
    record_telegram_run_report = helpers["record_telegram_run_report"]

    kill, reason = telegram_kill_switch_state(results)
    if kill:
        logger.warning("Telegram kill switch active; skip alerts (%s)", reason)
    min_conf = getattr(config, "TELEGRAM_ALERT_MIN_CONFIDENCE", 75.0)
    max_per_run = getattr(config, "TELEGRAM_ALERT_MAX_PER_RUN", 5)
    max_per_symbol = getattr(config, "TELEGRAM_ALERT_MAX_PER_SYMBOL", 1)
    cooldown_minutes = getattr(config, "TELEGRAM_ALERT_COOLDOWN_MINUTES", 30)
    try:
        min_conf = float(min_conf)
    except Exception:
        min_conf = 75.0
    try:
        max_per_run = int(max_per_run)
    except Exception:
        max_per_run = 5
    try:
        max_per_symbol = int(max_per_symbol)
    except Exception:
        max_per_symbol = 1
    try:
        cooldown_minutes = int(cooldown_minutes)
    except Exception:
        cooldown_minutes = 30
    if max_per_run < 1:
        max_per_run = 1
    if max_per_symbol < 1:
        max_per_symbol = 1
    if cooldown_minutes < 1:
        cooldown_minutes = 1

    dynamic_min_conf = telegram_dynamic_conf_threshold(min_conf, results)
    candidates = []
    build_stats = {}
    if not kill:
        candidates, build_stats = build_telegram_candidates(results, dynamic_min_conf)

    quality_drop_counts = {}
    if isinstance(build_stats, dict):
        quality_drop_counts = build_stats.get("quality_drop_counts") or {}

    if not candidates:
        logger.info(
            "Telegram alerts: no primary candidates (min_conf=%.1f, dynamic_min_conf=%.1f, quality_drops=%s)",
            min_conf,
            dynamic_min_conf,
            json.dumps(quality_drop_counts, ensure_ascii=False),
        )
        if kill and not is_daily_best_pick_window():
            record_telegram_run_report(
                results=results,
                kill=kill,
                kill_reason=reason,
                min_conf=min_conf,
                dynamic_min_conf=dynamic_min_conf,
                candidates=candidates,
                sent_candidates=[],
                daily_pick_sent=0,
                daily_summary_sent=0,
                dropped_by_cache=0,
                dropped_by_symbol_cap=0,
                dropped_by_run_cap=0,
                quality_drop_counts=quality_drop_counts,
            )
            return 0

    candidates.sort(key=lambda c: (float(c.get("score", 0.0)), float(c.get("confidence", 0.0))), reverse=True)
    sent = 0
    dropped_by_cache = 0
    dropped_by_symbol_cap = 0
    dropped_by_run_cap = 0
    per_symbol_sent = {}
    cooldown_ttl = max(60, int(cooldown_minutes * 60))
    sent_candidates = []

    for candidate in candidates:
        if sent >= max_per_run:
            dropped_by_run_cap += 1
            continue
        symbol = str(candidate.get("symbol") or "")
        if not symbol:
            continue
        if int(per_symbol_sent.get(symbol, 0)) >= max_per_symbol:
            dropped_by_symbol_cap += 1
            continue
        cache_key = str(candidate.get("cache_key") or "").strip()
        if not cache_key:
            continue
        if telegram_alert_cache.get(cache_key):
            dropped_by_cache += 1
            continue
        message = candidate.get("message")
        if not isinstance(message, str) or not message.strip():
            continue
        if send_telegram_alert(message):
            telegram_alert_cache.set(cache_key, True, ttl_seconds=cooldown_ttl)
            per_symbol_sent[symbol] = int(per_symbol_sent.get(symbol, 0)) + 1
            sent += 1
            sent_candidates.append(candidate)
            record_telegram_alert_history(candidate, min_conf=min_conf, dynamic_min_conf=dynamic_min_conf, daily_pick=False)

    daily_pick_sent = 0
    daily_summary_sent = 0
    if is_daily_best_pick_window():
        daily_candidates = build_daily_best_pick_candidates(results)
        for daily_candidate in daily_candidates:
            if not isinstance(daily_candidate, dict):
                continue
            daily_key = f"DAILYBEST|{get_now().strftime('%Y%m%d')}|{daily_candidate.get('symbol')}|{daily_candidate.get('signal')}"
            if telegram_alert_cache.get(daily_key):
                continue
            daily_message = daily_candidate.get("message")
            if isinstance(daily_message, str) and daily_message.strip() and send_telegram_alert(daily_message):
                telegram_alert_cache.set(daily_key, True, ttl_seconds=26 * 60 * 60)
                sent += 1
                daily_pick_sent += 1
                sent_candidates.append(daily_candidate)
                record_telegram_alert_history(daily_candidate, min_conf=min_conf, dynamic_min_conf=dynamic_min_conf, daily_pick=True)
        else:
            if not daily_pick_sent:
                daily_summary = build_daily_summary_message(results, existing_candidates=candidates, min_conf=dynamic_min_conf)
                daily_message = daily_summary.get("message") if isinstance(daily_summary, dict) else None
                daily_key = daily_summary.get("cache_key") if isinstance(daily_summary, dict) else None
                if (
                    isinstance(daily_summary, dict)
                    and isinstance(daily_key, str)
                    and daily_key.strip()
                    and not telegram_alert_cache.get(daily_key)
                    and isinstance(daily_message, str)
                    and daily_message.strip()
                    and send_telegram_alert(daily_message)
                ):
                    telegram_alert_cache.set(daily_key, True, ttl_seconds=26 * 60 * 60)
                    sent += 1
                    daily_summary_sent = 1
                    record_telegram_alert_history(daily_summary, min_conf=min_conf, dynamic_min_conf=dynamic_min_conf, daily_pick=False)
                elif not daily_summary_sent:
                    logger.info("Daily Best Pick window active but no directional candidate or summary was sent")

    logger.info(
        "Telegram alerts: sent=%s candidates=%s daily_pick=%s daily_summary=%s dropped(cache=%s symbol_cap=%s run_cap=%s quality=%s) min_conf=%.1f dynamic_min_conf=%.1f",
        sent,
        len(candidates),
        daily_pick_sent,
        daily_summary_sent,
        dropped_by_cache,
        dropped_by_symbol_cap,
        dropped_by_run_cap,
        json.dumps(quality_drop_counts, ensure_ascii=False),
        min_conf,
        dynamic_min_conf,
    )

    if sent_candidates:
        track_alert_performance(sent_candidates, len(sent_candidates))

    record_telegram_run_report(
        results=results,
        kill=kill,
        kill_reason=reason,
        min_conf=min_conf,
        dynamic_min_conf=dynamic_min_conf,
        candidates=candidates,
        sent_candidates=sent_candidates,
        daily_pick_sent=daily_pick_sent,
        daily_summary_sent=daily_summary_sent,
        dropped_by_cache=dropped_by_cache,
        dropped_by_symbol_cap=dropped_by_symbol_cap,
        dropped_by_run_cap=dropped_by_run_cap,
        quality_drop_counts=quality_drop_counts,
    )

    return sent
