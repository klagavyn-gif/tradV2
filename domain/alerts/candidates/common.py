import math
from collections import Counter


_EXIT_TRIGGERS = {"TAKE_PROFIT", "TIME_STOP", "PRECISION60_TAKE_PROFIT", "PRECISION60_TIME_STOP"}
_REVERSAL_TRIGGERS = {"CDC_RED_REVERSAL", "TREND_ROLLOVER"}
_EXIT_REASON_PHRASES = (
    "ถือครบ",
    "ปิดรอบ",
    "ลดการยืดเยื้อ",
    "close round",
    "time stop",
    "take profit",
    "ปิดกำไร",
)


def _safe_float(value, default=None):
    try:
        value = float(value)
    except Exception:
        return default
    if not math.isfinite(value):
        return default
    return value


def _normalize_ai_decision(value):
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"entry", "confirmed", "ai-confirmed"}:
        return "entry"
    if text in {"watch", "neutral", "ai-neutral"}:
        return "watch"
    if text in {"avoid", "low-conviction", "low conviction", "reject"}:
        return "avoid"
    return None


def resolve_ai_dispatch_profile(candidate, *, config):
    if not bool(getattr(config, "TELEGRAM_ALERT_AI_FILTER_ENABLE", True)):
        return None
    if not isinstance(candidate, dict):
        return None
    prob_win = _safe_float(candidate.get("ai_prob_win"), None)
    expected_return = _safe_float(candidate.get("ai_expected_return_pct"), None)
    explicit_decision = _normalize_ai_decision(candidate.get("ai_decision"))
    if explicit_decision is None and prob_win is None and expected_return is None:
        return None

    confirmed_threshold = _safe_float(getattr(config, "TELEGRAM_ALERT_AI_CONFIRMED_THRESHOLD", 0.60), 0.60)
    neutral_threshold = _safe_float(getattr(config, "TELEGRAM_ALERT_AI_NEUTRAL_THRESHOLD", 0.45), 0.45)
    confirmed_bonus = _safe_float(getattr(config, "TELEGRAM_ALERT_AI_CONFIRMED_SCORE_BONUS", 6.0), 6.0)
    low_penalty = _safe_float(getattr(config, "TELEGRAM_ALERT_AI_LOW_CONVICTION_SCORE_PENALTY", 2.5), 2.5)

    bucket = "neutral"
    label = "Neutral"
    icon = "🟡"
    reason = "AI มองเป็นตัวกลาง ใช้ประกอบกับ score เดิม"
    rank_adjustment = 0.0

    if explicit_decision == "entry" or (prob_win is not None and prob_win >= confirmed_threshold):
        bucket = "confirmed"
        label = "AI-Confirmed"
        icon = "🟢"
        reason = "AI ยืนยันสัญญาณ ใช้เพิ่มน้ำหนักการจัดอันดับได้"
        if bool(getattr(config, "TELEGRAM_ALERT_AI_RANKING_ENABLE", True)):
            rank_adjustment = float(confirmed_bonus)
    elif explicit_decision == "avoid" or (prob_win is not None and prob_win < neutral_threshold):
        bucket = "low_conviction"
        label = "Low-Conviction"
        icon = "🟠"
        reason = "AI ยังไม่มั่นใจพอ ควรลดอันดับแต่ไม่บล็อกทิ้งทันที"
        if bool(getattr(config, "TELEGRAM_ALERT_AI_RANKING_ENABLE", True)):
            rank_adjustment = -abs(float(low_penalty))

    return {
        "bucket": bucket,
        "label": label,
        "icon": icon,
        "reason": reason,
        "prob_win": prob_win,
        "expected_return_pct": expected_return,
        "rank_adjustment": float(rank_adjustment),
    }


def _append_ai_message_line(message, profile, *, config):
    if not bool(getattr(config, "TELEGRAM_ALERT_AI_MESSAGE_ENABLE", True)):
        return message
    if not isinstance(message, str) or not message.strip() or not isinstance(profile, dict):
        return message
    if "🤖 AI:" in message:
        return message
    parts = [str(profile.get("label") or "Neutral")]
    prob_win = _safe_float(profile.get("prob_win"), None)
    expected_return = _safe_float(profile.get("expected_return_pct"), None)
    if prob_win is not None:
        parts.append(f"pWin {prob_win * 100.0:.0f}%")
    if expected_return is not None:
        parts.append(f"Exp {expected_return:+.2f}%")
    line = "<b>🤖 AI:</b> " + " | ".join(parts)
    return message.rstrip() + "\n" + line


def attach_ai_dispatch_context(candidate, *, config):
    if not isinstance(candidate, dict):
        return candidate
    profile = resolve_ai_dispatch_profile(candidate, config=config)
    if not isinstance(profile, dict):
        return candidate
    candidate["ai_dispatch_bucket"] = str(profile.get("bucket") or "").strip() or None
    candidate["ai_dispatch_label"] = str(profile.get("label") or "").strip() or None
    candidate["ai_dispatch_icon"] = str(profile.get("icon") or "").strip() or None
    candidate["ai_dispatch_reason"] = str(profile.get("reason") or "").strip() or None
    candidate["ai_rank_adjustment"] = _safe_float(profile.get("rank_adjustment"), 0.0)
    candidate["ai_prob_win"] = _safe_float(profile.get("prob_win"), None)
    candidate["ai_expected_return_pct"] = _safe_float(profile.get("expected_return_pct"), None)
    candidate["message"] = _append_ai_message_line(candidate.get("message"), profile, config=config)
    return candidate


def _pick_numeric(plan, keys):
    if not isinstance(plan, dict):
        return None
    for key in keys or ():
        if key in plan:
            value = _safe_float(plan.get(key), None)
            if isinstance(value, float):
                return value
    return None


def _entry_distance_metrics(candidate):
    if not isinstance(candidate, dict):
        return None, None
    plan = candidate.get("plan")
    item = candidate.get("item")
    entry_price = _pick_numeric(plan, ["entry_price", "current_price", "price"])
    current_price = _pick_numeric(plan, ["current_price", "price"])
    if current_price is None and isinstance(item, dict):
        current_price = _safe_float(item.get("price"), None)
    stop_loss = _pick_numeric(plan, ["stop_loss", "entry_stop_loss", "trailing_stop"])
    if entry_price is None or current_price is None or entry_price == 0:
        return None, None
    distance_pct = abs(float(current_price) - float(entry_price)) / abs(float(entry_price)) * 100.0
    distance_r = None
    if isinstance(stop_loss, float) and not math.isclose(float(stop_loss), float(entry_price), rel_tol=1e-9, abs_tol=1e-9):
        risk = abs(float(entry_price) - float(stop_loss))
        if risk > 0:
            distance_r = abs(float(current_price) - float(entry_price)) / risk
    return distance_pct, distance_r


def _within_entry_window(candidate, *, config):
    distance_pct, distance_r = _entry_distance_metrics(candidate)
    max_distance_pct = _safe_float(getattr(config, "TELEGRAM_ALERT_ENTRY_MAX_DISTANCE_PCT", 2.5), 2.5)
    max_distance_r = _safe_float(getattr(config, "TELEGRAM_ALERT_ENTRY_MAX_DISTANCE_R", 1.1), 1.1)
    near_by_pct = isinstance(distance_pct, float) and distance_pct <= float(max_distance_pct)
    near_by_r = isinstance(distance_r, float) and distance_r <= float(max_distance_r)
    if near_by_pct or near_by_r:
        return True, distance_pct, distance_r
    if distance_pct is None and distance_r is None:
        return None, None, None
    return False, distance_pct, distance_r


def classify_candidate_intent(candidate, *, config):
    if not isinstance(candidate, dict):
        return "watch", "invalid_candidate"
    strategy = str(candidate.get("strategy") or "").strip().upper()
    signal = str(candidate.get("signal") or "").strip().upper()
    plan = candidate.get("plan")
    trigger = str((plan or {}).get("sell_trigger") or (plan or {}).get("exit_trigger") or "").strip().upper()
    plan_reason = str((plan or {}).get("reason") or "").strip().lower()
    if strategy in {"TRADAR15", "TRENDRADAR15", "TREND_RADAR", "TRENDSTATE15"}:
        return "watch", "strategy_watch_only"
    if trigger in _EXIT_TRIGGERS or any(phrase in plan_reason for phrase in _EXIT_REASON_PHRASES):
        return "exit", f"trigger:{trigger.lower() or 'plan_reason_exit'}"
    is_entry_window, distance_pct, distance_r = _within_entry_window(candidate, config=config)
    if strategy == "CDCVIX15":
        if is_entry_window is None:
            return "watch", "cdc_unknown_entry_distance"
        if trigger in _REVERSAL_TRIGGERS and is_entry_window:
            return "entry", f"cdc_reversal_fresh:{trigger.lower()}"
        if is_entry_window and signal in ("BUY", "SELL"):
            return "entry", "cdc_fresh_signal"
        return "watch", f"cdc_stretched:d_pct={distance_pct},d_r={distance_r}"
    if strategy in {"PRIMARY", "SS15", "AZ15", "PA15", "TCB15", "AW15"}:
        if is_entry_window is None:
            return "watch", "unknown_entry_distance"
        if is_entry_window:
            return "entry", f"fresh_entry:d_pct={distance_pct},d_r={distance_r}"
        return "watch", f"stretched_entry:d_pct={distance_pct},d_r={distance_r}"
    if is_entry_window is None:
        return "watch", "unknown_entry_distance"
    if signal in ("BUY", "SELL") and is_entry_window:
        return "entry", f"generic_entry:d_pct={distance_pct},d_r={distance_r}"
    return "watch", f"generic_watch:d_pct={distance_pct},d_r={distance_r}"


def prepare_candidate_context(results, min_conf, *, config, helpers, get_now, runtime_context=None):
    actionzone_precision60_profile = helpers["actionzone_precision60_profile"]
    strict_60_mode_enabled = helpers["strict_60_mode_enabled"]
    strict_60_allow_cdc = helpers["strict_60_allow_cdc"]
    build_market_regime_snapshot = helpers["build_market_regime_snapshot"]

    precision60 = actionzone_precision60_profile()
    strict_60 = strict_60_mode_enabled()
    allow_cdc = strict_60_allow_cdc()
    hour_key = get_now().strftime("%Y%m%d%H")
    regime_context = (runtime_context or {}).get("regime_context") if isinstance(runtime_context, dict) else {}
    market_regime = (regime_context or {}).get("market")
    if not isinstance(market_regime, dict):
        market_regime = build_market_regime_snapshot(results or [])
    symbol_regime_cache = dict((regime_context or {}).get("symbol_map") or {})
    return {
        "results": results or [],
        "min_conf": float(min_conf),
        "config": config,
        "helpers": helpers,
        "runtime_context": runtime_context if isinstance(runtime_context, dict) else {},
        "quality_drop_counts": Counter(),
        "candidates": [],
        "precision60": precision60,
        "strict_60": strict_60,
        "allow_cdc": allow_cdc,
        "hour_key": hour_key,
        "regime_context": regime_context if isinstance(regime_context, dict) else {},
        "market_regime": market_regime if isinstance(market_regime, dict) else {},
        "symbol_regime_cache": symbol_regime_cache,
    }

def get_symbol_regime(context, item):
    normalize_symbol = context["helpers"]["normalize_symbol"]
    build_symbol_regime = context["helpers"]["build_symbol_regime"]
    symbol_regime_cache = context["symbol_regime_cache"]
    market_regime = context["market_regime"]
    symbol = normalize_symbol((item or {}).get("symbol") or "")
    if not symbol:
        return {}
    cached = symbol_regime_cache.get(symbol)
    if isinstance(cached, dict):
        return cached
    payload = build_symbol_regime(item, market_snapshot=market_regime)
    symbol_regime_cache[symbol] = payload if isinstance(payload, dict) else {}
    return symbol_regime_cache[symbol]


def append_candidate(context, row):
    if not isinstance(row, dict):
        return
    apply_regime_to_candidate = context["helpers"]["apply_regime_to_candidate"]
    regime_payload = get_symbol_regime(context, row.get("item"))
    adjusted, regime_meta = apply_regime_to_candidate(row, regime_payload=regime_payload)
    if isinstance(regime_meta, dict) and regime_meta.get("blocked"):
        context["quality_drop_counts"][str(regime_meta.get("block_reason") or "regime_blocked")] += 1
        return
    uplift = 0.0
    if isinstance(regime_meta, dict):
        try:
            uplift = float(regime_meta.get("min_confidence_uplift") or 0.0)
        except Exception:
            uplift = 0.0
    confidence = adjusted.get("confidence") if isinstance(adjusted, dict) else row.get("confidence")
    try:
        confidence = float(confidence)
    except Exception:
        confidence = None
    if isinstance(confidence, float) and confidence < float(context["min_conf"]) + float(uplift):
        context["quality_drop_counts"]["regime_min_confidence_not_met"] += 1
        return
    context["candidates"].append(adjusted if isinstance(adjusted, dict) else row)


def add_quality_drop(context, reason, *, prefix=None):
    text = str(reason or "").strip()
    if not text:
        return
    key = f"{prefix}{text}" if prefix else text
    context["quality_drop_counts"][key] += 1


def score_with_edge_adjustments(base_score, edge, *, confidence, alert_profile_score_adjustment):
    score = float(base_score)
    if not isinstance(edge, dict):
        return score
    wr = edge.get("win_rate_pct")
    exp = edge.get("expectancy_rr")
    trades = edge.get("trades")
    score += alert_profile_score_adjustment(
        win_rate=wr,
        confidence=confidence,
        expectancy=exp,
        trades=trades,
    )
    if isinstance(wr, (int, float)):
        score += max(-3.0, min(8.0, (float(wr) - 50.0) * 0.20))
    if isinstance(exp, (int, float)):
        score += max(-4.0, min(8.0, float(exp) * 8.0))
    if isinstance(trades, (int, float)):
        score += max(0.0, min(4.0, float(trades) / 8.0))
    return float(score)


def finalize_candidates(context):
    evaluate_candidate_backtest_gate = context["helpers"]["evaluate_candidate_backtest_gate"]
    evaluate_candidate_symbol_strategy_gate = context["helpers"]["evaluate_candidate_symbol_strategy_gate"]
    candidate_alert_profile = context["helpers"]["candidate_alert_profile"]
    build_regime_alert_budget = context["helpers"]["build_regime_alert_budget"]
    score_candidate_with_live_ai = context["helpers"].get("score_candidate_with_live_ai")

    filtered_candidates = []
    for candidate in context["candidates"]:
        gate_ok, gate_reason, edge_metrics = evaluate_candidate_backtest_gate(candidate)
        if not gate_ok:
            context["quality_drop_counts"][gate_reason] += 1
            continue
        candidate["edge_metrics"] = edge_metrics
        profile_ok, profile_reason, profile_metrics = evaluate_candidate_symbol_strategy_gate(candidate)
        if not profile_ok:
            context["quality_drop_counts"][profile_reason] += 1
            continue
        if isinstance(profile_metrics, dict) and profile_metrics:
            candidate["edge_metrics"] = profile_metrics
        candidate["alert_profile"] = candidate_alert_profile(candidate)
        alert_intent, alert_intent_reason = classify_candidate_intent(candidate, config=context["config"])
        candidate["alert_intent"] = alert_intent
        candidate["alert_intent_reason"] = alert_intent_reason
        if callable(score_candidate_with_live_ai):
            candidate = score_candidate_with_live_ai(candidate)
        candidate = attach_ai_dispatch_context(candidate, config=context["config"])
        filtered_candidates.append(candidate)

    regime_context = context["regime_context"]
    regime_summary = {
        "enabled": (regime_context or {}).get("enabled"),
        "generated_at": (regime_context or {}).get("generated_at"),
        "market": context["market_regime"],
        "symbols": list((regime_context or {}).get("symbols") or list(context["symbol_regime_cache"].values())),
        "by_symbol_regime": (regime_context or {}).get("by_symbol_regime") or {},
        "by_side_bias": (regime_context or {}).get("by_side_bias") or {},
        "alert_budget": (regime_context or {}).get("alert_budget") or {},
    }
    alert_budget = (regime_context or {}).get("alert_budget")
    if not isinstance(alert_budget, dict) or not alert_budget:
        alert_budget = build_regime_alert_budget(regime_summary=regime_summary)
    stats = {
        "quality_drop_counts": dict(context["quality_drop_counts"]),
        "regime_summary": regime_summary,
        "alert_budget": alert_budget if isinstance(alert_budget, dict) else {},
    }
    return filtered_candidates, stats
