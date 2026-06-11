import math

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


def _harmonize_action_guidance(action_guidance, decision, *, signal=None):
    guidance = dict(action_guidance) if isinstance(action_guidance, dict) else {}
    decision = decision if isinstance(decision, dict) else {}
    label = str(decision.get("label") or "").strip()
    reason = str(decision.get("reason") or "").strip()
    action_code = str(guidance.get("action_code") or "").strip().upper()
    signal_text = _normalize_signal(signal or guidance.get("signal") or "BUY")

    if label == "เข้าได้":
        if signal_text == "SELL":
            guidance["action_code"] = "ENTRY SHORT"
            guidance["primary_text"] = "ถ้ายังไม่มีสถานะ สามารถกด SELL/Short ตามแผนได้"
            guidance.setdefault("note_text", "ถ้ามี BUY เดิม ควรปิด BUY ก่อนแล้วค่อยเปิด SELL/Short")
        else:
            guidance["action_code"] = "ENTRY BUY"
            guidance["primary_text"] = "ถ้ายังไม่มีสถานะ สามารถกด BUY/Long ตามแผนได้"
            guidance.setdefault("note_text", "ถ้ามี Short เดิม ควรปิด Short ก่อนแล้วค่อยเปิด BUY/Long")
        return guidance

    if label == "รอ":
        if signal_text == "SELL":
            primary_text = "ยังไม่เข้า รอราคาเข้าใกล้จุดเข้า หรือรอแท่งยืนยันก่อนค่อยเปิด SELL/Short"
        else:
            primary_text = "ยังไม่เข้า รอราคาเข้าใกล้จุดเข้า หรือรอแท่งยืนยันก่อนค่อยเปิด BUY/Long"
        guidance["action_code"] = f"WATCH {signal_text}"
        guidance["primary_text"] = primary_text
        guidance["note_text"] = reason or "ยังไม่ใช่จังหวะเปิดไม้ใหม่ทันที"
        return guidance

    if label == "ห้ามเข้า":
        if action_code.startswith("EXIT"):
            guidance["primary_text"] = "ถ้ามีสถานะเดิม ให้ปิดกำไรหรือปิดลดความเสี่ยงตามแผนนี้"
            guidance["note_text"] = "ถ้าไม่มีสถานะอยู่แล้ว ให้ข้ามข้อความนี้ ยังไม่ใช่จุดเปิดไม้ใหม่"
        elif action_code == "SELL / RISK-OFF":
            guidance["primary_text"] = "ให้มองเป็นสัญญาณลดความเสี่ยงหรือปิดสถานะเดิมก่อนเป็นหลัก"
            guidance["note_text"] = "ถ้าไม่มีสถานะอยู่แล้ว ให้ข้ามข้อความนี้ ยังไม่ควรกลับฝั่งเปิดไม้ใหม่ทันที"
        else:
            guidance["action_code"] = f"NO ENTRY {signal_text}"
            guidance["primary_text"] = "งดเปิดไม้ใหม่จากข้อความนี้ก่อน"
            guidance["note_text"] = reason or "รอแผนที่มี Entry/SL/TP ชัดกว่านี้ก่อน"
        return guidance

    return guidance


def _append_action_lines(lines, action_guidance, *, html_escape, decision=None, signal=None):
    guidance = _harmonize_action_guidance(action_guidance, decision, signal=signal)
    if not isinstance(guidance, dict):
        return
    action_text = str(guidance.get("primary_text") or "").strip()
    if action_text:
        lines.append("<b>🎯 Action:</b> " + html_escape(action_text))
    note_text = str(guidance.get("note_text") or "").strip()
    if note_text:
        lines.append("<b>⚠️ Note:</b> " + html_escape(note_text))


def _resolve_trade_decision(plan, *, signal=None, strategy_label=None, action_guidance=None, current_price=None):
    plan = plan if isinstance(plan, dict) else {}
    strategy_text = str(strategy_label or "").strip().upper()
    signal_text = _normalize_signal(signal or plan.get("signal") or "BUY")
    action_code = str((action_guidance or {}).get("action_code") or "").strip().upper()
    trigger = str(plan.get("sell_trigger") or plan.get("exit_trigger") or "").strip().upper()
    plan_reason = str(plan.get("reason") or "").strip().lower()
    continuation_mode = str(plan.get("sell_continuation_override_mode") or "").strip().lower()
    exit_triggers = {"TAKE_PROFIT", "TIME_STOP", "PRECISION60_TAKE_PROFIT", "PRECISION60_TIME_STOP"}
    exit_reason_phrases = ("ถือครบ", "ปิดรอบ", "ลดการยืดเยื้อ", "close round", "time stop", "take profit", "ปิดกำไร")

    if "TREND RADAR" in strategy_text or "TRADAR" in strategy_text:
        return {
            "label": "รอ",
            "icon": "🟡",
            "reason": "เป็นแผนเฝ้าเข้า รอราคาเข้าโซนก่อน",
        }
    if "TREND STATE" in strategy_text:
        return {
            "label": "รอ",
            "icon": "🟡",
            "reason": "เป็นการบอกสถานะเทรนด์ ยังไม่ใช่จุดเข้า",
        }
    if continuation_mode not in {"entry", "watch"} and (
        action_code.startswith("EXIT")
        or action_code == "SELL / RISK-OFF"
        or trigger in exit_triggers
        or any(phrase in plan_reason for phrase in exit_reason_phrases)
    ):
        return {
            "label": "ห้ามเข้า",
            "icon": "⛔",
            "reason": "เป็นสัญญาณปิดรอบหรือลดความเสี่ยง ไม่ใช่จุดเปิดไม้ใหม่",
        }
    if action_code.startswith("BIAS"):
        return {
            "label": "รอ",
            "icon": "🟡",
            "reason": "ใช้เป็น bias ของวัน ต้องรอระบบหลักยืนยันก่อน",
        }

    guidance = _resolve_level_guidance(plan, signal=signal_text)
    if not isinstance(guidance, dict):
        return {
            "label": "ห้ามเข้า",
            "icon": "⛔",
            "reason": "ข้อมูลแผนยังไม่ครบสำหรับคำนวณ Entry/SL/TP",
        }
    entry = _safe_float(guidance.get("entry"), None)
    stop = _safe_float(guidance.get("stop"), None)
    rr1 = _safe_float(guidance.get("rr1"), None)
    if entry is None or stop is None:
        return {
            "label": "ห้ามเข้า",
            "icon": "⛔",
            "reason": "ไม่มี Entry หรือ SL ที่ชัดพอสำหรับเข้าไม้",
        }
    if rr1 is not None and rr1 < 1.0:
        return {
            "label": "ห้ามเข้า",
            "icon": "⛔",
            "reason": "RR ถึง TP1 ต่ำกว่า 1R ไม่คุ้มสำหรับเล่นสั้น",
        }

    current_value = _safe_float(current_price, None)
    if current_value is None:
        current_value = _safe_float(plan.get("current_price") or plan.get("price"), None)
    distance_pct = None
    if entry is not None and current_value is not None and abs(entry) > 0:
        distance_pct = abs(float(current_value) - float(entry)) / abs(float(entry)) * 100.0
    risk_dist = abs(float(entry) - float(stop)) if entry is not None and stop is not None else None
    distance_r = None
    if isinstance(risk_dist, (int, float)) and risk_dist > 0 and current_value is not None:
        distance_r = abs(float(current_value) - float(entry)) / float(risk_dist)
    if isinstance(distance_pct, (int, float)) and float(distance_pct) > 2.5:
        return {
            "label": "รอ",
            "icon": "🟡",
            "reason": f"ราคาห่างจากจุดเข้า {float(distance_pct):.2f}% แล้ว ไม่ควรไล่ราคา",
        }
    if isinstance(distance_r, (int, float)) and float(distance_r) > 1.0:
        return {
            "label": "รอ",
            "icon": "🟡",
            "reason": f"ราคาห่างจากจุดเข้า {float(distance_r):.2f}R แล้ว รอจังหวะกลับเข้าโซน",
        }
    return {
        "label": "เข้าได้",
        "icon": "🟢",
        "reason": "แผนยังใกล้จุดเข้าและมี SL/TP พร้อมใช้งาน",
    }


def _append_trade_decision_lines(lines, *, plan, html_escape, signal=None, strategy_label=None, action_guidance=None, current_price=None):
    decision = _resolve_trade_decision(
        plan,
        signal=signal,
        strategy_label=strategy_label,
        action_guidance=action_guidance,
        current_price=current_price,
    )
    if not isinstance(decision, dict):
        return
    label = str(decision.get("label") or "").strip()
    reason = str(decision.get("reason") or "").strip()
    icon = str(decision.get("icon") or "🟡").strip()
    if label:
        lines.append(f"<b>🚦 สถานะ:</b> {html_escape(icon + ' ' + label)}")
    if reason:
        lines.append("<b>📝 Decision:</b> " + html_escape(reason))
    return decision


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


def _safe_float(value, default=None):
    try:
        value = float(value)
    except Exception:
        return default
    if not math.isfinite(value):
        return default
    return value


def _normalize_signal(value, default="BUY"):
    text = str(value or "").strip().upper()
    if "SELL" in text or "SHORT" in text or text == "DOWN":
        return "SELL"
    if "BUY" in text or "LONG" in text or text == "UP":
        return "BUY"
    return str(default or "BUY").strip().upper() or "BUY"


def _compute_rr_value(entry, stop, target):
    if not all(isinstance(v, (int, float)) for v in (entry, stop, target)):
        return None
    risk = abs(float(entry) - float(stop))
    if risk <= 0:
        return None
    reward = abs(float(target) - float(entry))
    if reward <= 0:
        return None
    return reward / risk


def _generate_target_levels(entry_price, stop_loss, *, signal="BUY", take_profit=None):
    entry = _safe_float(entry_price, None)
    stop = _safe_float(stop_loss, None)
    if entry is None or stop is None or entry == stop:
        return None, []
    risk_dist = abs(entry - stop)
    if risk_dist <= 0:
        return None, []
    risk_pct = (risk_dist / abs(entry)) * 100.0 if entry else None
    provided_rr = _compute_rr_value(entry, stop, _safe_float(take_profit, None))
    if isinstance(provided_rr, (int, float)) and provided_rr >= 1.0:
        r3 = max(3.0, float(provided_rr))
        r2 = max(1.9, min(2.6, r3 * 0.72))
        r1 = max(1.2, min(1.6, r2 - 0.7))
        r_levels = [r1, r2, r3]
    else:
        if isinstance(risk_pct, (int, float)) and risk_pct < 0.8:
            r_levels = [1.4, 2.4, 3.8]
        elif isinstance(risk_pct, (int, float)) and risk_pct < 1.5:
            r_levels = [1.2, 2.1, 3.2]
        else:
            r_levels = [1.0, 1.8, 2.8]
    direction = -1.0 if _normalize_signal(signal, "BUY") == "SELL" else 1.0
    levels = []
    for idx, r_mult in enumerate(r_levels[:3], start=1):
        levels.append(
            {
                "label": f"TP{idx}",
                "target_price": float(entry + (direction * risk_dist * float(r_mult))),
                "reward_r": float(r_mult),
            }
        )
    return risk_pct, levels


def _resolve_level_guidance(
    plan,
    *,
    pick_plan_value=None,
    entry_keys=None,
    stop_keys=None,
    tp_keys=None,
    tp2_keys=None,
    signal=None,
):
    if not isinstance(plan, dict):
        return None
    entry_keys = entry_keys or ["entry_price", "current_price", "price"]
    stop_keys = stop_keys or ["stop_loss", "entry_stop_loss", "trailing_stop"]
    tp_keys = tp_keys or ["take_profit", "take_profit_2", "exit_price"]
    tp2_keys = tp2_keys or ["take_profit_2", "tp2", "take_profit_price_2"]

    entry_value = _resolve_plan_value(plan, pick_plan_value, entry_keys)
    stop_value = _resolve_plan_value(plan, pick_plan_value, stop_keys)
    tp1_value = _resolve_plan_value(plan, pick_plan_value, tp_keys)
    tp2_value = _resolve_plan_value(plan, pick_plan_value, tp2_keys)
    signal_text = _normalize_signal(signal or plan.get("signal") or plan.get("raw_signal") or plan.get("setup") or plan.get("recommendation"))

    entry = _safe_float(entry_value, None)
    stop = _safe_float(stop_value, None)
    tp1 = _safe_float(tp1_value, None)
    tp2 = _safe_float(tp2_value, None)
    risk_pct = _safe_float(plan.get("entry_risk_pct"), None)

    used_fallback = False
    used_actual = any(isinstance(v, (int, float)) for v in (stop, tp1, tp2))
    if entry is not None and stop is None and isinstance(risk_pct, (int, float)) and float(risk_pct) > 0:
        risk_dist = abs(float(entry) * (float(risk_pct) / 100.0))
        if risk_dist > 0:
            used_fallback = True
            stop = float(entry - risk_dist) if signal_text == "BUY" else float(entry + risk_dist)

    generated_risk_pct, generated_levels = _generate_target_levels(entry, stop, signal=signal_text, take_profit=tp1)
    if risk_pct is None and isinstance(generated_risk_pct, (int, float)):
        risk_pct = float(generated_risk_pct)
    if tp1 is None and generated_levels:
        used_fallback = True
        tp1 = _safe_float(generated_levels[0].get("target_price"), None)
    if tp2 is None and len(generated_levels) >= 2:
        used_fallback = True
        fallback_candidates = [_safe_float(level.get("target_price"), None) for level in generated_levels[1:]]
        for candidate in fallback_candidates:
            if candidate is None:
                continue
            if tp1 is not None and math.isclose(float(candidate), float(tp1), rel_tol=1e-9, abs_tol=1e-9):
                continue
            current_rr = _compute_rr_value(entry, stop, tp1)
            candidate_rr = _compute_rr_value(entry, stop, candidate)
            if tp1 is not None and isinstance(current_rr, (int, float)) and isinstance(candidate_rr, (int, float)):
                if float(candidate_rr) <= float(current_rr):
                    continue
            tp2 = float(candidate)
            break

    if not any(isinstance(v, (int, float)) for v in (entry, stop, tp1, tp2)):
        return None

    if used_fallback and used_actual:
        level_source = "actual+fallback"
    elif used_fallback:
        level_source = "fallback"
    else:
        level_source = "actual"

    return {
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "risk_pct": risk_pct,
        "rr1": _compute_rr_value(entry, stop, tp1),
        "rr2": _compute_rr_value(entry, stop, tp2),
        "level_source": level_source,
    }


def _append_levels_lines(
    lines,
    *,
    plan,
    format_price_value,
    html_escape,
    pick_plan_value=None,
    entry_keys=None,
    stop_keys=None,
    tp_keys=None,
    tp2_keys=None,
    signal=None,
    entry_override_text=None,
    stop_label="SL",
    plan_heading="📌 Plan",
):
    guidance = _resolve_level_guidance(
        plan,
        pick_plan_value=pick_plan_value,
        entry_keys=entry_keys,
        stop_keys=stop_keys,
        tp_keys=tp_keys,
        tp2_keys=tp2_keys,
        signal=signal,
    )
    if not isinstance(guidance, dict):
        return
    parts = []
    entry_text = entry_override_text or format_price_value(guidance.get("entry"))
    stop_text = format_price_value(guidance.get("stop"))
    tp1_text = format_price_value(guidance.get("tp1"))
    tp2_text = format_price_value(guidance.get("tp2"))
    if entry_text:
        parts.append(f"Entry {entry_text}")
    if stop_text:
        parts.append(f"{stop_label} {stop_text}")
    if tp1_text:
        parts.append(f"TP1 {tp1_text}")
    if tp2_text:
        parts.append(f"TP2 {tp2_text}")
    if parts:
        lines.append(f"<b>{html_escape(plan_heading)}:</b> " + " | ".join(html_escape(part) for part in parts))
    risk_parts = []
    risk_pct = guidance.get("risk_pct")
    rr1 = guidance.get("rr1")
    rr2 = guidance.get("rr2")
    if isinstance(risk_pct, (int, float)):
        risk_parts.append(f"Risk {float(risk_pct):.2f}%")
    rr_bits = []
    if isinstance(rr1, (int, float)):
        rr_bits.append(f"TP1 {float(rr1):.2f}R")
    if isinstance(rr2, (int, float)):
        rr_bits.append(f"TP2 {float(rr2):.2f}R")
    if rr_bits:
        risk_parts.append("RR " + " / ".join(rr_bits))
    if risk_parts:
        lines.append("<b>📏 Risk:</b> " + " | ".join(html_escape(part) for part in risk_parts))
    level_source = str(guidance.get("level_source") or "").strip().lower()
    if level_source == "actual":
        lines.append("<b>🧭 Level Source:</b> actual plan")
    elif level_source == "actual+fallback":
        lines.append("<b>🧭 Level Source:</b> actual+fallback")
    elif level_source == "fallback":
        lines.append("<b>🧭 Level Source:</b> fallback risk model")


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
    decision = _append_trade_decision_lines(
        lines,
        plan=primary_plan,
        html_escape=html_escape,
        signal=signal,
        strategy_label="PRIMARY",
        action_guidance=action_guidance,
        current_price=item.get("price"),
    )
    _append_action_lines(lines, action_guidance, html_escape=html_escape, decision=decision, signal=signal)
    context_parts = []
    if isinstance(primary_plan, dict):
        source_label = get_plan_label(primary_plan, item)
        if source_label:
            context_parts.append(str(source_label))
        pattern = str(primary_plan.get("detected_pattern") or "").strip()
        if pattern and pattern.upper() != "NONE":
            context_parts.append(f"Pattern {pattern}")
    _append_reason_line(lines, html_escape=html_escape, parts=context_parts)
    plan_heading = "📌 Plan" if str((decision or {}).get("label") or "").strip() == "เข้าได้" else "📌 Reference Plan"
    _append_levels_lines(
        lines,
        plan=primary_plan,
        format_price_value=format_price_value,
        html_escape=html_escape,
        pick_plan_value=pick_plan_value,
        plan_heading=plan_heading,
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
    daily_lines = [
        "<b>🗓️ Daily Pick:</b> ตัวเด่นของวันจาก watchlist",
        "<b>⚠️ Use:</b> เป็นตัวเด่นของรอบนี้ แต่ยังต้องยึดสถานะ เข้าได้/รอ/ห้ามเข้า ด้านล่างเป็นหลัก",
    ]
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


def build_trend_state_message(item, state_snapshot, *, helpers, get_now):
    html_escape = helpers["html_escape"]
    normalize_symbol = helpers["normalize_symbol"]
    format_price_value = helpers["format_price_value"]

    if not isinstance(state_snapshot, dict):
        return None
    trend = str(state_snapshot.get("trend") or "").strip().upper()
    signal = str(state_snapshot.get("signal") or "").strip().upper()
    if trend not in ("UP", "DOWN") or signal not in ("BUY", "SELL"):
        return None

    symbol = normalize_symbol(item.get("symbol") or "")
    if not symbol:
        return None
    name = html_escape(str(item.get("name") or "").strip())
    tv_symbol = symbol.replace("-", "")
    icon = "🟢" if signal == "BUY" else "🔴"
    side_label = "ขาขึ้นแรง" if signal == "BUY" else "ขาลงแรง"

    lines = [f"<b>{icon} Trend State | {html_escape(symbol)}</b>"]
    if name:
        lines.append(f"<i>{name}</i>")
    lines.append("────────────────")

    _append_snapshot_lines(
        lines,
        price_text=format_price_value(item.get("price")),
        change=item.get("change"),
        confidence=state_snapshot.get("score"),
        sources=state_snapshot.get("supporting_sources"),
        html_escape=html_escape,
    )
    lines.append(
        "<b>🧭 State:</b> "
        + " | ".join(
            html_escape(part)
            for part in [
                side_label,
                f"1H {trend}",
                str(state_snapshot.get("trend_strength") or "WEAK").strip().upper(),
            ]
            if str(part).strip()
        )
    )

    context_parts = []
    symbol_regime = str(state_snapshot.get("symbol_regime") or "").strip().upper()
    market_regime = str(state_snapshot.get("market_regime") or "").strip().upper()
    if symbol_regime:
        context_parts.append(f"Symbol Regime {symbol_regime}")
    if market_regime:
        context_parts.append(f"Market {market_regime}")
    directional_sources = state_snapshot.get("directional_source_count")
    if isinstance(directional_sources, (int, float)):
        context_parts.append(f"Consensus {int(directional_sources)}")
    _append_reason_line(lines, html_escape=html_escape, parts=context_parts)

    tags = [str(tag).strip().upper() for tag in (state_snapshot.get("tags") or []) if str(tag).strip()]
    if tags:
        lines.append("<b>⚡ Tags:</b> " + " | ".join(html_escape(tag) for tag in tags[:3]))

    support_labels = [str(label).strip() for label in (state_snapshot.get("supporting_sources") or []) if str(label).strip()]
    if support_labels:
        lines.append("<b>🤝 Sources:</b> " + " | ".join(html_escape(label) for label in support_labels[:3]))

    lines.append("<b>⚠️ Note:</b> เป็นการแจ้งสถานะเทรนด์ ไม่ใช่จุดเข้าเทรดทันที")
    _append_footer(lines, get_now=get_now, tv_symbol=tv_symbol)
    return "\n".join(lines)


def build_trend_radar_message(item, radar_snapshot, *, helpers, get_now):
    html_escape = helpers["html_escape"]
    normalize_symbol = helpers["normalize_symbol"]
    format_price_value = helpers["format_price_value"]

    if not isinstance(radar_snapshot, dict):
        return None
    signal = str(radar_snapshot.get("signal") or "").strip().upper()
    subtype = str(radar_snapshot.get("subtype") or "").strip().upper()
    if signal not in ("BUY", "SELL") or subtype not in ("TREND_START", "TREND_CONTINUE"):
        return None

    symbol = normalize_symbol(item.get("symbol") or "")
    if not symbol:
        return None
    name = html_escape(str(item.get("name") or "").strip())
    tv_symbol = symbol.replace("-", "")
    icon = "🟢" if signal == "BUY" else "🔴"
    direction_label = "UP STRONG" if signal == "BUY" else "DOWN STRONG"
    subtype_label = "START" if subtype == "TREND_START" else "CONTINUE"

    lines = [f"<b>{icon} Trend Radar | {html_escape(symbol)}</b>"]
    if name:
        lines.append(f"<i>{name}</i>")
    lines.append("────────────────")
    _append_snapshot_lines(
        lines,
        price_text=format_price_value(radar_snapshot.get("price") or item.get("price")),
        change=radar_snapshot.get("change", item.get("change")),
        confidence=radar_snapshot.get("score"),
        sources=[radar_snapshot.get("source_label")] + list(radar_snapshot.get("supporting_sources") or []),
        html_escape=html_escape,
    )
    lines.append(f"<b>🧭 State:</b> {html_escape(direction_label)} | {html_escape(subtype_label)}")

    context_parts = []
    trend_1h = str(radar_snapshot.get("trend_1h") or "").strip().upper()
    trend_strength_1h = str(radar_snapshot.get("trend_strength_1h") or "").strip().upper()
    if trend_1h:
        if trend_strength_1h:
            context_parts.append(f"1H {trend_1h} {trend_strength_1h}")
        else:
            context_parts.append(f"1H {trend_1h}")
    symbol_regime = str(radar_snapshot.get("symbol_regime") or "").strip().upper()
    if symbol_regime:
        context_parts.append(f"Regime {symbol_regime}")
    adx = radar_snapshot.get("adx")
    if isinstance(adx, (int, float)):
        context_parts.append(f"ADX {float(adx):.1f}")
    rvol = radar_snapshot.get("rvol")
    if isinstance(rvol, (int, float)):
        context_parts.append(f"RVOL {float(rvol):.2f}")
    _append_reason_line(lines, html_escape=html_escape, parts=context_parts)

    entry_low = format_price_value(radar_snapshot.get("entry_zone_low"))
    entry_high = format_price_value(radar_snapshot.get("entry_zone_high"))
    entry_price = format_price_value(radar_snapshot.get("entry_price"))
    entry_override_text = None
    if entry_low and entry_high:
        entry_override_text = f"{entry_low}-{entry_high}"
    elif entry_price:
        entry_override_text = entry_price
    _append_levels_lines(
        lines,
        plan=radar_snapshot,
        format_price_value=format_price_value,
        html_escape=html_escape,
        entry_keys=["entry_price", "price"],
        stop_keys=["stop_loss"],
        tp_keys=["take_profit_price", "take_profit", "exit_price"],
        tp2_keys=["take_profit_price_2", "take_profit_2"],
        signal=signal,
        entry_override_text=entry_override_text,
    )
    _append_trade_decision_lines(
        lines,
        plan=radar_snapshot,
        html_escape=html_escape,
        signal=signal,
        strategy_label="Trend Radar 15m",
        current_price=radar_snapshot.get("price") or item.get("price"),
    )
    tags = [str(tag).strip().upper() for tag in (radar_snapshot.get("tags") or []) if str(tag).strip()]
    if tags:
        lines.append("<b>⚡ Tags:</b> " + " | ".join(html_escape(tag) for tag in tags[:3]))
    reasons = [str(reason).strip() for reason in (radar_snapshot.get("reasons") or []) if str(reason).strip()]
    if reasons:
        lines.append("<b>🧠 Context:</b> " + " | ".join(html_escape(reason) for reason in reasons[:3]))
    lines.append("<b>⚠️ Note:</b> trend-following setup ใช้เป็นแผนเฝ้าเข้า ไม่ใช่สัญญาณไล่ราคา")
    _append_footer(lines, get_now=get_now, tv_symbol=tv_symbol)
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
    decision = _append_trade_decision_lines(
        lines,
        plan=plan,
        html_escape=html_escape,
        signal=signal,
        strategy_label="Price Action 15m",
        action_guidance=action_guidance,
        current_price=item.get("price"),
    )
    _append_action_lines(lines, action_guidance, html_escape=html_escape, decision=decision, signal=signal)

    context_parts = [
        str(plan.get("setup_label") or "").strip(),
        str(plan.get("chart_pattern") or "").strip(),
        str(plan.get("market_structure") or "").strip(),
        str(plan.get("trend_1h") or "").strip(),
    ]
    _append_reason_line(lines, html_escape=html_escape, parts=context_parts, reasons=plan.get("reasons"))
    plan_heading = "📌 Plan" if str((decision or {}).get("label") or "").strip() == "เข้าได้" else "📌 Reference Plan"
    _append_levels_lines(
        lines,
        plan=plan,
        format_price_value=format_price_value,
        html_escape=html_escape,
        pick_plan_value=pick_plan_value,
        plan_heading=plan_heading,
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
    decision = _append_trade_decision_lines(
        lines,
        plan=plan,
        html_escape=html_escape,
        signal=signal,
        strategy_label="Trend Breakout 15m",
        action_guidance=action_guidance,
        current_price=plan.get("current_price", item.get("price")),
    )
    _append_action_lines(lines, action_guidance, html_escape=html_escape, decision=decision, signal=signal)
    _append_reason_line(lines, html_escape=html_escape, parts=context_parts, reasons=plan.get("reasons"))
    plan_heading = "📌 Plan" if str((decision or {}).get("label") or "").strip() == "เข้าได้" else "📌 Reference Plan"
    _append_levels_lines(lines, plan=plan, format_price_value=format_price_value, html_escape=html_escape, plan_heading=plan_heading)
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
    lines.append("<b>⚠️ Use:</b> เป็นสัญญาณที่ confluence สูง แต่ยังต้องยึดสถานะและแผนด้านล่าง ไม่ใช่คำสั่งเข้าอัตโนมัติ")

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
    decision = _append_trade_decision_lines(
        lines,
        plan=primary_plan,
        html_escape=html_escape,
        signal=signal,
        strategy_label="Super Signal Ensemble",
        action_guidance=action_guidance,
        current_price=item.get("price"),
    )
    _append_action_lines(lines, action_guidance, html_escape=html_escape, decision=decision, signal=signal)
    pattern = primary_plan.get("detected_pattern") if isinstance(primary_plan, dict) else None
    context_parts = []
    if pattern and pattern != "None":
        context_parts.append(f"Pattern {pattern}")
    _append_reason_line(lines, html_escape=html_escape, parts=context_parts)
    plan_heading = "📌 Plan" if str((decision or {}).get("label") or "").strip() == "เข้าได้" else "📌 Reference Plan"
    _append_levels_lines(lines, plan=primary_plan, format_price_value=format_price_value, html_escape=html_escape, plan_heading=plan_heading)
    _append_footer(lines, get_now=get_now, tv_symbol=tv_symbol)
    return "\n".join(lines)
