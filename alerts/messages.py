def build_telegram_message(
    item,
    signal,
    best_conf,
    sources,
    *,
    primary_plan=None,
    mode_label=None,
    helpers,
    get_now,
):
    normalize_symbol = helpers["normalize_symbol"]
    html_escape = helpers["html_escape"]
    format_price_value = helpers["format_price_value"]
    pick_primary_trade_plan = helpers["pick_primary_trade_plan"]
    strict_60_mode_enabled = helpers["strict_60_mode_enabled"]
    strict_60_allow_cdc = helpers["strict_60_allow_cdc"]
    extract_signal_edge_metrics = helpers["extract_signal_edge_metrics"]
    build_alert_profile_lines = helpers["build_alert_profile_lines"]
    append_pattern_context_lines = helpers["append_pattern_context_lines"]
    build_stop_context_lines = helpers["build_stop_context_lines"]
    get_plan_label = helpers["get_plan_label"]
    format_exit_levels_lines = helpers["format_exit_levels_lines"]
    format_price_forecast_lines = helpers["format_price_forecast_lines"]

    emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
    symbol = normalize_symbol(item.get("symbol") or "")
    name = html_escape(str(item.get("name") or "").strip())
    tv_symbol = symbol.replace("-", "")

    lines = [f"<b>{emoji} สัญญาณหลัก {signal} | {html_escape(symbol)}</b>"]
    if name:
        lines.append(f"<i>{name}</i>")
    lines.append("────────────────")

    price = item.get("price")
    change = item.get("change")
    price_text = format_price_value(price)
    if price_text:
        if isinstance(change, (int, float)):
            lines.append(f"<b>ราคา:</b> {price_text} ({change:+.2f}%)")
        else:
            lines.append(f"<b>ราคา:</b> {price_text}")

    if best_conf is not None:
        lines.append(f"<b>ความมั่นใจ:</b> {best_conf:.0f}%")

    if sources:
        lines.append(f"<b>แหล่งสัญญาณ:</b> " + ", ".join([html_escape(s) for s in sources]))

    if not isinstance(primary_plan, dict):
        primary_plan = pick_primary_trade_plan(
            item,
            signal=signal,
            require_quality=strict_60_mode_enabled(),
            allow_cdc=strict_60_allow_cdc(),
        )
    edge_metrics = extract_signal_edge_metrics(primary_plan, signal) if isinstance(primary_plan, dict) else {}
    lines.extend(
        build_alert_profile_lines(
            win_rate=edge_metrics.get("win_rate_pct"),
            confidence=best_conf,
            expectancy=edge_metrics.get("expectancy_rr"),
            trades=edge_metrics.get("trades"),
            mode_label=mode_label,
        )
    )
    pattern = primary_plan.get("detected_pattern") if isinstance(primary_plan, dict) else None
    append_pattern_context_lines(lines, pattern)
    if isinstance(primary_plan, dict):
        stop_lines = build_stop_context_lines(
            item,
            primary_plan,
            signal=signal,
            source_label=get_plan_label(primary_plan, item),
        )
        if stop_lines:
            lines.append("────────────────")
            lines.extend(stop_lines)

        exit_lines = format_exit_levels_lines(primary_plan)
        if exit_lines:
            if not stop_lines:
                lines.append("────────────────")
            lines.extend([html_escape(line) for line in exit_lines])

    forecast_lines = format_price_forecast_lines(item.get("price_forecast"))
    if forecast_lines:
        lines.append("────────────────")
        lines.extend([html_escape(line) for line in forecast_lines])

    lines.append("────────────────")
    lines.append("🕒 <b>เวลา:</b> " + get_now().strftime("%Y-%m-%d %H:%M"))
    lines.append(f"<a href=\"https://th.tradingview.com/chart/?symbol=CRYPTO:{tv_symbol}\">📈 ดูชาร์ตบน TradingView</a>")
    return "\n".join(lines)


def build_daily_best_pick_message(
    item,
    signal,
    best_conf,
    sources,
    *,
    primary_plan=None,
    strategy_label=None,
    selection_score=None,
    mode_label=None,
    helpers,
    get_now,
):
    normalize_symbol = helpers["normalize_symbol"]
    html_escape = helpers["html_escape"]
    alert_mode_usage_hint = helpers["alert_mode_usage_hint"]
    build_telegram_message_fn = helpers["build_telegram_message"]

    base_message = build_telegram_message_fn(
        item,
        signal,
        best_conf,
        sources,
        primary_plan=primary_plan,
        mode_label=mode_label,
    )
    if not isinstance(base_message, str) or not base_message.strip():
        return None
    symbol = normalize_symbol(item.get("symbol") or "")
    lines = base_message.splitlines()
    if not lines:
        return None
    lines[0] = f"<b>⭐ Daily Top Pick {signal} | {html_escape(symbol)}</b>"
    insert_at = 1
    if len(lines) > 1 and lines[1].startswith("<i>"):
        insert_at = 2
    daily_lines = [
        "<b>🗓️ Daily Scan:</b> หนึ่งในตัวเด่นของวันจาก watchlist",
    ]
    if mode_label:
        daily_lines.append("<b>🧭 โหมด:</b> " + html_escape(str(mode_label)))
        mode_hint = alert_mode_usage_hint(mode_label=mode_label)
        if mode_hint:
            daily_lines.append("<b>🧠 บทบาท:</b> " + html_escape(mode_hint))
    if strategy_label:
        daily_lines.append("<b>🎯 แผนหลัก:</b> " + html_escape(str(strategy_label)))
    if isinstance(selection_score, (int, float)):
        daily_lines.append(f"<b>⭐ คะแนนคัดเลือก:</b> {float(selection_score):.1f}")
    lines[insert_at:insert_at] = daily_lines
    return "\n".join(lines)


def build_price_action_message(item, plan, *, helpers, get_now):
    html_escape = helpers["html_escape"]
    normalize_symbol = helpers["normalize_symbol"]
    pick_plan_value = helpers["pick_plan_value"]
    format_price_value = helpers["format_price_value"]
    plan_confidence_value = helpers["plan_confidence_value"]
    build_alert_profile_lines = helpers["build_alert_profile_lines"]
    append_pattern_context_lines = helpers["append_pattern_context_lines"]
    build_stop_context_lines = helpers["build_stop_context_lines"]
    format_exit_levels_lines = helpers["format_exit_levels_lines"]
    format_price_forecast_lines = helpers["format_price_forecast_lines"]

    signal = str(plan.get("signal") or "").upper()
    if signal not in ("BUY", "SELL"):
        return None
    emoji = "🟢" if signal == "BUY" else "🔴"
    symbol = normalize_symbol(item.get("symbol") or "")
    name = html_escape(str(item.get("name") or "").strip())
    tv_symbol = symbol.replace("-", "")
    lines = [f"<b>{emoji} Price Action 15m {signal} | {html_escape(symbol)}</b>"]
    if name:
        lines.append(f"<i>{name}</i>")
    lines.append("────────────────")

    entry_price = pick_plan_value(plan, ["entry_price", "current_price", "price"])
    curr_price = item.get("price")
    change = item.get("change")
    entry_text = format_price_value(entry_price)
    curr_text = format_price_value(curr_price)
    if entry_text:
        lines.append(f"<b>📍 จุดเข้า:</b> {entry_text}")
    if curr_text:
        change_str = f" ({change:+.2f}%)" if isinstance(change, (int, float)) else ""
        lines.append(f"<b>📍 ราคาปัจจุบัน:</b> {curr_text}{change_str}")

    structure = str(plan.get("market_structure") or "").strip()
    trend_1h = str(plan.get("trend_1h") or "").strip()
    wyckoff = str(plan.get("wyckoff_phase") or "").strip()
    if structure or trend_1h or wyckoff:
        parts = []
        if structure:
            parts.append(f"Structure: {structure}")
        if trend_1h:
            parts.append(f"Trend 1H: {trend_1h}")
        if wyckoff:
            parts.append(f"Wyckoff: {wyckoff}")
        lines.append("<b>🧭 บริบทกราฟ:</b> " + " • ".join([html_escape(part) for part in parts]))

    chart_pattern = str(plan.get("chart_pattern") or "").strip()
    role_reversal = str(plan.get("role_reversal") or "").strip()
    setup_label = str(plan.get("setup_label") or "").strip()
    if setup_label:
        lines.append("<b>🧱 Setup:</b> " + html_escape(setup_label))
    if chart_pattern:
        lines.append("<b>📐 Chart Pattern:</b> " + html_escape(chart_pattern))
    if role_reversal:
        lines.append("<b>🔄 Role Reversal:</b> " + html_escape(role_reversal))

    demand_zone = format_price_value(plan.get("demand_zone"))
    supply_zone = format_price_value(plan.get("supply_zone"))
    nearest_support = format_price_value(plan.get("nearest_support"))
    nearest_resistance = format_price_value(plan.get("nearest_resistance"))
    zone_parts = []
    if demand_zone:
        zone_parts.append(f"Demand {demand_zone}")
    if supply_zone:
        zone_parts.append(f"Supply {supply_zone}")
    if nearest_support:
        zone_parts.append(f"Support {nearest_support}")
    if nearest_resistance:
        zone_parts.append(f"Resistance {nearest_resistance}")
    if zone_parts:
        lines.append("<b>🗺️ โซนสำคัญ:</b> " + " | ".join([html_escape(part) for part in zone_parts]))

    conf = plan_confidence_value(plan)
    lines.extend(
        build_alert_profile_lines(
            win_rate=plan.get("historical_win_rate"),
            confidence=conf,
            expectancy=plan.get("historical_avg_rr"),
            trades=plan.get("historical_trades"),
        )
    )

    proxy_sources = plan.get("proxy_sources")
    if isinstance(proxy_sources, list) and proxy_sources:
        lines.append("<b>🤝 Backtest Proxy:</b> " + " | ".join([html_escape(str(src)) for src in proxy_sources[:4]]))

    reasons = plan.get("reasons")
    if isinstance(reasons, list) and reasons:
        lines.append("<b>🔎 เหตุผลเชิงพฤติกรรมราคา:</b>")
        for reason in reasons[:5]:
            if reason:
                lines.append("• " + html_escape(str(reason)))

    pattern = plan.get("detected_pattern")
    append_pattern_context_lines(lines, pattern)

    stop_lines = build_stop_context_lines(item, plan, signal=signal, source_label="Price Action 15m")
    if stop_lines:
        lines.append("────────────────")
        lines.extend(stop_lines)

    exit_lines = format_exit_levels_lines(plan)
    if exit_lines:
        if not stop_lines:
            lines.append("────────────────")
        lines.extend([html_escape(line) for line in exit_lines])

    forecast_lines = format_price_forecast_lines(item.get("price_forecast"))
    if forecast_lines:
        lines.append("────────────────")
        lines.extend([html_escape(line) for line in forecast_lines])

    lines.append("────────────────")
    last_signal_time = plan.get("last_signal_time")
    if isinstance(last_signal_time, str) and last_signal_time:
        lines.append(f"🕒 <b>สัญญาณล่าสุด:</b> {html_escape(last_signal_time)}")
    lines.append("⏱️ <b>เวลา:</b> " + get_now().strftime("%Y-%m-%d %H:%M"))
    lines.append(f"<a href=\"https://th.tradingview.com/chart/?symbol=CRYPTO:{tv_symbol}\">📈 ดูชาร์ตบน TradingView</a>")
    return "\n".join(lines)


def build_all_weather_message(item, aw_signal, *, helpers, get_now):
    html_escape = helpers["html_escape"]
    normalize_symbol = helpers["normalize_symbol"]
    build_telegram_message_fn = helpers["build_telegram_message"]

    if not isinstance(aw_signal, dict):
        return None
    signal = str(aw_signal.get("signal") or "").upper()
    if signal not in ("BUY", "SELL"):
        return None
    base_message = build_telegram_message_fn(
        item,
        signal,
        aw_signal.get("confidence"),
        aw_signal.get("sources") or [],
        primary_plan=aw_signal.get("primary_plan"),
    )
    if not isinstance(base_message, str) or not base_message.strip():
        return None
    symbol = normalize_symbol(item.get("symbol") or "")
    lines = base_message.splitlines()
    if not lines:
        return None
    lines[0] = f"<b>🌦️ All-Weather {signal} | {html_escape(symbol)}</b>"
    insert_at = 1
    if len(lines) > 1 and lines[1].startswith("<i>"):
        insert_at = 2
    regime = str(aw_signal.get("regime") or "RANGE").upper()
    volatility_pct = aw_signal.get("volatility_pct")
    trend_score = aw_signal.get("trend_score")
    side_gap = aw_signal.get("side_gap")
    top_buy_score = aw_signal.get("top_buy_score")
    top_sell_score = aw_signal.get("top_sell_score")
    blend = aw_signal.get("blend") or {}
    selected_rows = aw_signal.get("selected_rows") or []
    confluence_labels = [str(row.get("label") or "") for row in selected_rows if isinstance(row, dict)]
    extra_lines = [f"<b>🧠 Market Regime:</b> {regime}"]
    regime_stats = []
    if isinstance(volatility_pct, (int, float)):
        regime_stats.append(f"Vol {float(volatility_pct):.2f}%")
    if isinstance(trend_score, (int, float)):
        regime_stats.append(f"Trend Score {float(trend_score):.2f}")
    if regime_stats:
        extra_lines[-1] += " | " + " | ".join(regime_stats)
    if confluence_labels:
        extra_lines.append("<b>🤝 Confluence:</b> " + ", ".join([html_escape(s) for s in confluence_labels]))
    side_stats = []
    if isinstance(top_buy_score, (int, float)):
        side_stats.append(f"BUY {float(top_buy_score):.1f}")
    if isinstance(top_sell_score, (int, float)):
        side_stats.append(f"SELL {float(top_sell_score):.1f}")
    if isinstance(side_gap, (int, float)):
        side_stats.append(f"Gap {float(side_gap):.1f}")
    if side_stats:
        extra_lines.append("<b>⚖️ Side Selection:</b> " + " | ".join(side_stats))
    blend_stats = []
    wr_blend = blend.get("win_rate_pct")
    exp_blend = blend.get("expectancy_rr")
    trades_blend = blend.get("trades")
    if isinstance(wr_blend, (int, float)):
        blend_stats.append(f"WR {float(wr_blend):.1f}%")
    if isinstance(exp_blend, (int, float)):
        blend_stats.append(f"ExpRR {float(exp_blend):.2f}")
    if isinstance(trades_blend, (int, float)) and int(trades_blend) > 0:
        blend_stats.append(f"Trades {int(trades_blend)}")
    if blend_stats:
        extra_lines.append("<b>🧪 Blended Edge:</b> " + " | ".join(blend_stats))
    lines[insert_at:insert_at] = extra_lines
    return "\n".join(lines)


def build_super_signal_message(item, signal, super_meta, *, primary_plan=None, helpers, get_now):
    html_escape = helpers["html_escape"]
    normalize_symbol = helpers["normalize_symbol"]
    format_price_value = helpers["format_price_value"]
    build_alert_profile_lines = helpers["build_alert_profile_lines"]
    pick_primary_trade_plan = helpers["pick_primary_trade_plan"]
    strict_60_mode_enabled = helpers["strict_60_mode_enabled"]
    strict_60_allow_cdc = helpers["strict_60_allow_cdc"]
    build_stop_context_lines = helpers["build_stop_context_lines"]
    format_exit_levels_lines = helpers["format_exit_levels_lines"]

    emoji = "🔥" if signal == "BUY" else "🧊" if signal == "SELL" else "⚪"
    symbol = normalize_symbol(item.get("symbol") or "")
    name = html_escape(str(item.get("name") or "").strip())
    tv_symbol = symbol.replace("-", "")
    avg_wr = super_meta.get("avg_wr", 0)
    avg_exp = super_meta.get("avg_exp", 0)
    confluence = super_meta.get("confluence", [])

    lines = [f"<b>{emoji} SUPER SIGNAL {signal} | {html_escape(symbol)}</b>"]
    if name:
        lines.append(f"<i>{name}</i>")
    lines.append("🏆 <b>ความแม่นยำย้อนหลังสูงพิเศษ (High Accuracy)</b>")
    lines.append("────────────────")
    lines.extend(
        build_alert_profile_lines(
            win_rate=avg_wr,
            expectancy=avg_exp,
            trades=super_meta.get("avg_trades"),
        )
    )

    price = item.get("price")
    change = item.get("change")
    price_text = format_price_value(price)
    if price_text:
        change_str = f" ({change:+.2f}%)" if isinstance(change, (int, float)) else ""
        lines.append(f"<b>ราคาปัจจุบัน:</b> {price_text}{change_str}")

    lines.append(f"<b>📊 Win Rate เฉลี่ย:</b> {avg_wr:.1f}%")
    lines.append(f"<b>📈 คาดการณ์กำไร (ExpRR):</b> {avg_exp:.2f}")
    lines.append(f"<b>🤝 Confluence:</b> " + ", ".join(confluence))

    if not isinstance(primary_plan, dict):
        primary_plan = pick_primary_trade_plan(
            item,
            signal=signal,
            require_quality=strict_60_mode_enabled(),
            allow_cdc=strict_60_allow_cdc(),
        )
    pattern = primary_plan.get("detected_pattern") if isinstance(primary_plan, dict) else None
    if pattern and pattern != "None":
        lines.append(f"<b>🕯️ Confirmation Pattern:</b> {html_escape(pattern)}")

    if isinstance(primary_plan, dict):
        stop_lines = build_stop_context_lines(
            item,
            primary_plan,
            signal=signal,
            source_label="Super Signal Ensemble",
        )
        if stop_lines:
            lines.append("────────────────")
            lines.extend(stop_lines)

        exit_lines = format_exit_levels_lines(primary_plan)
        if exit_lines:
            if not stop_lines:
                lines.append("────────────────")
            lines.extend([html_escape(line) for line in exit_lines])

    lines.append("────────────────")
    lines.append("🕒 <b>เวลา:</b> " + get_now().strftime("%Y-%m-%d %H:%M"))
    lines.append(f"<a href=\"https://th.tradingview.com/chart/?symbol=CRYPTO:{tv_symbol}\">📈 วิเคราะห์กราฟบน TradingView</a>")
    return "\n".join(lines)

