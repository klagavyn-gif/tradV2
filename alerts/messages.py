from domain.alerts.trend_1h import infer_1h_trend_snapshot


def _format_sources_compact(sources, html_escape, *, limit=3):
    if not isinstance(sources, list) or not sources:
        return None
    cleaned = [str(source).strip() for source in sources if str(source).strip()]
    if not cleaned:
        return None
    visible = cleaned[:limit]
    text = ", ".join(html_escape(source) for source in visible)
    if len(cleaned) > limit:
        text += f" +{len(cleaned) - limit}"
    return text


def _append_snapshot_lines(lines, *, price_text=None, change=None, confidence=None, sources=None, html_escape):
    snapshot_parts = []
    if price_text:
        price_part = f"ราคา {price_text}"
        if isinstance(change, (int, float)):
            price_part += f" ({change:+.2f}%)"
        snapshot_parts.append(price_part)
    if isinstance(confidence, (int, float)):
        snapshot_parts.append(f"Conf {float(confidence):.0f}%")
    if snapshot_parts:
        lines.append("<b>📍 Snapshot:</b> " + " | ".join(html_escape(part) for part in snapshot_parts))
    source_text = _format_sources_compact(sources, html_escape)
    if source_text:
        lines.append("<b>🧩 Source:</b> " + source_text)


def _append_edge_lines(lines, *, win_rate=None, expectancy=None, trades=None, html_escape, prefix="🧪 Edge"):
    parts = []
    if isinstance(win_rate, (int, float)):
        parts.append(f"WR {float(win_rate):.1f}%")
    if isinstance(expectancy, (int, float)):
        parts.append(f"ExpRR {float(expectancy):.2f}")
    if isinstance(trades, (int, float)) and float(trades) > 0:
        parts.append(f"Trades {int(round(float(trades)))}")
    if parts:
        lines.append(f"<b>{prefix}:</b> " + " | ".join(html_escape(part) for part in parts))


def _append_action_lines(lines, action_guidance, *, html_escape):
    if not isinstance(action_guidance, dict):
        return
    action_text = str(action_guidance.get("primary_text") or "").strip()
    if action_text:
        lines.append("<b>🎯 Action:</b> " + html_escape(action_text))
    note_text = str(action_guidance.get("note_text") or "").strip()
    if note_text:
        lines.append("<b>⚠️ Note:</b> " + html_escape(note_text))


def _resolve_plan_value(plan, pick_plan_value, keys):
    if callable(pick_plan_value):
        return pick_plan_value(plan, keys)
    if not isinstance(plan, dict):
        return None
    for key in keys:
        value = plan.get(key)
        if value not in (None, ""):
            return value
    return None


def _append_levels_lines(lines, *, plan, format_price_value, html_escape, pick_plan_value=None):
    if not isinstance(plan, dict):
        return
    entry_value = _resolve_plan_value(plan, pick_plan_value, ["entry_price", "current_price", "price"])
    stop_value = _resolve_plan_value(plan, pick_plan_value, ["stop_loss", "entry_stop_loss", "trailing_stop"])
    take_profit_value = _resolve_plan_value(plan, pick_plan_value, ["take_profit", "take_profit_2", "exit_price"])
    parts = []
    entry_text = format_price_value(entry_value)
    stop_text = format_price_value(stop_value)
    take_profit_text = format_price_value(take_profit_value)
    if entry_text:
        parts.append(f"Entry {entry_text}")
    if stop_text:
        parts.append(f"SL {stop_text}")
    if take_profit_text:
        parts.append(f"TP {take_profit_text}")
    if parts:
        lines.append("<b>📌 Plan:</b> " + " | ".join(html_escape(part) for part in parts))
    risk_pct = plan.get("entry_risk_pct")
    if isinstance(risk_pct, (int, float)):
        lines.append(f"<b>📏 Risk:</b> {float(risk_pct):.2f}%")


def _append_reason_line(lines, *, html_escape, parts=None, reasons=None, label="🧠 Context"):
    compact_parts = []
    if isinstance(parts, list):
        compact_parts.extend([str(part).strip() for part in parts if str(part).strip()])
    if isinstance(reasons, list):
        compact_parts.extend([str(reason).strip() for reason in reasons if str(reason).strip()])
    if not compact_parts:
        return
    lines.append(f"<b>{label}:</b> " + " | ".join(html_escape(part) for part in compact_parts[:3]))


def _append_footer(lines, *, get_now, tv_symbol):
    lines.append("────────────────")
    lines.append("🕒 <b>เวลา:</b> " + get_now().strftime("%Y-%m-%d %H:%M"))
    lines.append(f"<a href=\"https://th.tradingview.com/chart/?symbol=CRYPTO:{tv_symbol}\">📈 TradingView</a>")


def _append_hourly_bias_line(lines, *, item, html_escape, label="🧭 1H Trend"):
    snapshot = infer_1h_trend_snapshot(item)
    if not isinstance(snapshot, dict):
        return
    trend = str(snapshot.get("trend") or "").upper()
    if trend not in ("UP", "DOWN"):
        return
    parts = [f"{trend}"]
    strength = str(snapshot.get("strength") or "").upper()
    if strength in ("STRONG", "WEAK"):
        parts.append(strength)
    source_labels = snapshot.get("source_labels") or []
    if source_labels:
        parts.append(", ".join(str(label_text) for label_text in source_labels[:2]))
    lines.append(f"<b>{label}:</b> " + " | ".join(html_escape(part) for part in parts))


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
    get_plan_label = helpers["get_plan_label"]
    pick_plan_value = helpers["pick_plan_value"]
    build_trade_action_guidance = helpers["build_trade_action_guidance"]

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
    _append_snapshot_lines(
        lines,
        price_text=price_text,
        change=change,
        confidence=best_conf,
        sources=sources,
        html_escape=html_escape,
    )
    _append_hourly_bias_line(lines, item=item, html_escape=html_escape)

    if not isinstance(primary_plan, dict):
        primary_plan = pick_primary_trade_plan(
            item,
            signal=signal,
            require_quality=strict_60_mode_enabled(),
            allow_cdc=strict_60_allow_cdc(),
        )
    edge_metrics = extract_signal_edge_metrics(primary_plan, signal) if isinstance(primary_plan, dict) else {}
    _append_edge_lines(
        lines,
        win_rate=edge_metrics.get("win_rate_pct"),
        expectancy=edge_metrics.get("expectancy_rr"),
        trades=edge_metrics.get("trades"),
        html_escape=html_escape,
    )
    action_guidance = build_trade_action_guidance(
        signal,
        plan=primary_plan,
        mode_label=mode_label,
        source_label=get_plan_label(primary_plan, item) if isinstance(primary_plan, dict) else None,
    )
    _append_action_lines(lines, action_guidance, html_escape=html_escape)
    context_parts = []
    if isinstance(primary_plan, dict):
        source_label = get_plan_label(primary_plan, item)
        if source_label:
            context_parts.append(str(source_label))
        pattern = str(primary_plan.get("detected_pattern") or "").strip()
        if pattern and pattern.upper() != "NONE":
            context_parts.append(f"Pattern {pattern}")
    _append_reason_line(lines, html_escape=html_escape, parts=context_parts)
    _append_levels_lines(
        lines,
        plan=primary_plan,
        format_price_value=format_price_value,
        html_escape=html_escape,
        pick_plan_value=pick_plan_value,
    )
    _append_footer(lines, get_now=get_now, tv_symbol=tv_symbol)
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
    daily_lines = ["<b>🗓️ Daily Pick:</b> ตัวเด่นของวันจาก watchlist"]
    info_parts = []
    if mode_label:
        info_parts.append(str(mode_label))
        mode_hint = alert_mode_usage_hint(mode_label=mode_label)
        if mode_hint:
            info_parts.append(mode_hint)
    if strategy_label:
        info_parts.append(f"แผน {strategy_label}")
    if isinstance(selection_score, (int, float)):
        info_parts.append(f"Score {float(selection_score):.1f}")
    if info_parts:
        daily_lines.append("<b>🧠 Context:</b> " + " | ".join(html_escape(str(part)) for part in info_parts[:3]))
    lines[insert_at:insert_at] = daily_lines
    return "\n".join(lines)


def build_price_action_message(item, plan, *, helpers, get_now):
    html_escape = helpers["html_escape"]
    normalize_symbol = helpers["normalize_symbol"]
    pick_plan_value = helpers["pick_plan_value"]
    format_price_value = helpers["format_price_value"]
    plan_confidence_value = helpers["plan_confidence_value"]
    build_trade_action_guidance = helpers["build_trade_action_guidance"]

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
    _append_snapshot_lines(
        lines,
        price_text=curr_text or entry_text,
        change=change,
        confidence=plan_confidence_value(plan),
        sources=None,
        html_escape=html_escape,
    )
    _append_hourly_bias_line(lines, item=item, html_escape=html_escape)

    conf = plan_confidence_value(plan)
    _append_edge_lines(
        lines,
        win_rate=plan.get("historical_win_rate"),
        expectancy=plan.get("historical_avg_rr"),
        trades=plan.get("historical_trades"),
        html_escape=html_escape,
    )
    action_guidance = build_trade_action_guidance(
        signal,
        plan=plan,
        source_label="Price Action 15m",
    )
    _append_action_lines(lines, action_guidance, html_escape=html_escape)

    context_parts = [
        str(plan.get("setup_label") or "").strip(),
        str(plan.get("chart_pattern") or "").strip(),
        str(plan.get("market_structure") or "").strip(),
        str(plan.get("trend_1h") or "").strip(),
    ]
    _append_reason_line(lines, html_escape=html_escape, parts=context_parts, reasons=plan.get("reasons"))
    _append_levels_lines(
        lines,
        plan=plan,
        format_price_value=format_price_value,
        html_escape=html_escape,
        pick_plan_value=pick_plan_value,
    )
    _append_footer(lines, get_now=get_now, tv_symbol=tv_symbol)
    return "\n".join(lines)


def build_trend_breakout_message(item, plan, *, helpers, get_now):
    html_escape = helpers["html_escape"]
    normalize_symbol = helpers["normalize_symbol"]
    format_price_value = helpers["format_price_value"]
    plan_confidence_value = helpers["plan_confidence_value"]
    build_trade_action_guidance = helpers["build_trade_action_guidance"]

    signal = str(plan.get("signal") or "").upper()
    if signal not in ("BUY", "SELL"):
        return None
    symbol = normalize_symbol(item.get("symbol") or "")
    name = html_escape(str(item.get("name") or "").strip())
    tv_symbol = symbol.replace("-", "")
    side_label = "Breakout" if signal == "BUY" else "Breakdown"
    action_label = "BUY" if signal == "BUY" else "SHORT"
    icon = "🟢" if signal == "BUY" else "🔴"
    lines = [f"<b>{icon} Trend {side_label} 15m {action_label} | {html_escape(symbol)}</b>"]
    if name:
        lines.append(f"<i>{name}</i>")
    lines.append("────────────────")

    entry_text = format_price_value(plan.get("entry_price"))
    curr_text = format_price_value(plan.get("current_price", item.get("price")))
    breakout_text = format_price_value(plan.get("breakout_level"))
    change = item.get("change")
    _append_snapshot_lines(
        lines,
        price_text=curr_text or entry_text,
        change=change,
        confidence=plan_confidence_value(plan),
        sources=None,
        html_escape=html_escape,
    )
    _append_hourly_bias_line(lines, item=item, html_escape=html_escape)

    trend_1h = str(plan.get("trend_1h") or "").strip()
    market_bias = str(plan.get("market_bias") or "").strip()
    adx = plan.get("adx")
    rvol = plan.get("rvol")
    context_parts = []
    if market_bias:
        context_parts.append(f"Bias {market_bias}")
    if trend_1h:
        context_parts.append(f"Trend 1H {trend_1h}")
    if isinstance(adx, (int, float)):
        context_parts.append(f"ADX {float(adx):.1f}")
    if isinstance(rvol, (int, float)):
        context_parts.append(f"RVOL {float(rvol):.2f}")
    if breakout_text:
        context_parts.insert(0, f"Level {breakout_text}")

    conf = plan_confidence_value(plan)
    _append_edge_lines(
        lines,
        win_rate=plan.get("historical_win_rate"),
        expectancy=plan.get("historical_avg_rr"),
        trades=plan.get("historical_trades"),
        html_escape=html_escape,
    )
    action_guidance = build_trade_action_guidance(signal, plan=plan, source_label="Trend Breakout 15m")
    _append_action_lines(lines, action_guidance, html_escape=html_escape)
    _append_reason_line(lines, html_escape=html_escape, parts=context_parts, reasons=plan.get("reasons"))
    _append_levels_lines(lines, plan=plan, format_price_value=format_price_value, html_escape=html_escape)
    _append_footer(lines, get_now=get_now, tv_symbol=tv_symbol)
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
        extra_lines.append("<b>🤝 Confluence:</b> " + ", ".join([html_escape(s) for s in confluence_labels[:3]]))
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
    pick_primary_trade_plan = helpers["pick_primary_trade_plan"]
    strict_60_mode_enabled = helpers["strict_60_mode_enabled"]
    strict_60_allow_cdc = helpers["strict_60_allow_cdc"]
    build_trade_action_guidance = helpers["build_trade_action_guidance"]

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
    lines.append("────────────────")

    price = item.get("price")
    change = item.get("change")
    price_text = format_price_value(price)
    _append_snapshot_lines(
        lines,
        price_text=price_text,
        change=change,
        confidence=avg_wr,
        sources=confluence,
        html_escape=html_escape,
    )
    _append_hourly_bias_line(lines, item=item, html_escape=html_escape)
    _append_edge_lines(
        lines,
        win_rate=avg_wr,
        expectancy=avg_exp,
        trades=super_meta.get("avg_trades"),
        html_escape=html_escape,
        prefix="🏆 Ensemble",
    )

    if not isinstance(primary_plan, dict):
        primary_plan = pick_primary_trade_plan(
            item,
            signal=signal,
            require_quality=strict_60_mode_enabled(),
            allow_cdc=strict_60_allow_cdc(),
        )
    action_guidance = build_trade_action_guidance(
        signal,
        plan=primary_plan,
        source_label="Super Signal Ensemble",
    )
    _append_action_lines(lines, action_guidance, html_escape=html_escape)
    pattern = primary_plan.get("detected_pattern") if isinstance(primary_plan, dict) else None
    context_parts = []
    if pattern and pattern != "None":
        context_parts.append(f"Pattern {pattern}")
    _append_reason_line(lines, html_escape=html_escape, parts=context_parts)
    _append_levels_lines(lines, plan=primary_plan, format_price_value=format_price_value, html_escape=html_escape)
    _append_footer(lines, get_now=get_now, tv_symbol=tv_symbol)
    return "\n".join(lines)
