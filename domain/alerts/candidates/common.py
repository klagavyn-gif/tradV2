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


def _runtime_ai_dispatch_profile(candidate):
    if not isinstance(candidate, dict):
        return None
    status = str(candidate.get("ai_runtime_status") or "").strip().lower()
    reason_text = str(candidate.get("ai_runtime_reason") or "").strip()
    if not status:
        return None
    if status == "scored":
        return None

    mapping = {
        "disabled": ("disabled", "AI-Disabled", "⚪", "AI scorer ถูกปิดสำหรับรอบนี้"),
        "model_unavailable": ("unavailable", "AI-Unavailable", "⚪", "รอบนี้ยังโหลด model ไม่สำเร็จ"),
        "not_allowed": ("not_scored", "AI-Not-Scored", "⚪", "alert นี้อยู่นอก scope ของ model ปัจจุบัน"),
        "score_failed": ("error", "AI-Score-Failed", "⚪", "รอบนี้ score ไม่สำเร็จ ใช้ rule เดิมแทน"),
    }
    bucket, label, icon, default_reason = mapping.get(
        status,
        ("unknown", "AI-Unknown", "⚪", "ยังไม่มีผล AI สำหรับ alert นี้"),
    )
    return {
        "bucket": bucket,
        "label": label,
        "icon": icon,
        "reason": reason_text or default_reason,
        "prob_win": None,
        "expected_return_pct": None,
        "rank_adjustment": 0.0,
    }


def resolve_ai_dispatch_profile(candidate, *, config):
    if not bool(getattr(config, "TELEGRAM_ALERT_AI_FILTER_ENABLE", True)):
        return None
    if not isinstance(candidate, dict):
        return None
    prob_win = _safe_float(candidate.get("ai_prob_win"), None)
    expected_return = _safe_float(candidate.get("ai_expected_return_pct"), None)
    explicit_decision = _normalize_ai_decision(candidate.get("ai_decision"))
    if explicit_decision is None and prob_win is None and expected_return is None:
        return _runtime_ai_dispatch_profile(candidate)

    confirmed_threshold = _safe_float(getattr(config, "TELEGRAM_ALERT_AI_CONFIRMED_THRESHOLD", 0.60), 0.60)
    neutral_threshold = _safe_float(getattr(config, "TELEGRAM_ALERT_AI_NEUTRAL_THRESHOLD", 0.45), 0.45)
    confirmed_bonus = _safe_float(getattr(config, "TELEGRAM_ALERT_AI_CONFIRMED_SCORE_BONUS", 6.0), 6.0)
    low_penalty = _safe_float(getattr(config, "TELEGRAM_ALERT_AI_LOW_CONVICTION_SCORE_PENALTY", 2.5), 2.5)

    bucket = "neutral"
    label = "กลาง"
    icon = "🟡"
    reason = "AI มองว่ากลาง ๆ ใช้ช่วยดูเพิ่ม แต่ยังไม่ใช่จังหวะเด่น"
    rank_adjustment = 0.0

    if explicit_decision == "entry" or (prob_win is not None and prob_win >= confirmed_threshold):
        bucket = "confirmed"
        label = "ยืนยันเพิ่ม"
        icon = "🟢"
        reason = "AI เห็นด้วยกับสัญญาณนี้ ใช้เพิ่มความมั่นใจได้"
        if bool(getattr(config, "TELEGRAM_ALERT_AI_RANKING_ENABLE", True)):
            rank_adjustment = float(confirmed_bonus)
    elif explicit_decision == "avoid" or (prob_win is not None and prob_win < neutral_threshold):
        bucket = "low_conviction"
        label = "ยังไม่ชัด"
        icon = "🟠"
        reason = "AI ยังไม่มั่นใจพอ ควรรอดูเพิ่ม"
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


def _ai_message_role_text(profile):
    if not isinstance(profile, dict):
        return None
    bucket = str(profile.get("bucket") or "").strip().lower()
    if bucket == "confirmed":
        return "ช่วยยืนยันสัญญาณ"
    if bucket == "neutral":
        return "กลาง ๆ ใช้ดูประกอบ"
    if bucket == "low_conviction":
        return "ยังไม่ชัด ควรรอเพิ่ม"
    if bucket in {"disabled", "unavailable", "not_scored", "error", "unknown"}:
        return "รอบนี้ยังไม่มีคะแนน AI"
    return None


def _append_ai_message_line(message, profile, *, config):
    if not bool(getattr(config, "TELEGRAM_ALERT_AI_MESSAGE_ENABLE", True)):
        return message
    if not isinstance(message, str) or not message.strip() or not isinstance(profile, dict):
        return message
    if "🤖 AI:" in message:
        return message
    parts = [str(profile.get("label") or "กลาง")]
    prob_win = _safe_float(profile.get("prob_win"), None)
    expected_return = _safe_float(profile.get("expected_return_pct"), None)
    if prob_win is not None:
        parts.append(f"p(win) {prob_win * 100.0:.0f}%")
    if expected_return is not None:
        parts.append(f"exp {expected_return:+.2f}%")
    if prob_win is None and expected_return is None:
        reason = str(profile.get("reason") or "").strip()
        if not reason:
            reason = _ai_message_role_text(profile)
        if reason:
            parts.append(reason)
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


def _runtime_entry_ai_profile(candidate):
    if not isinstance(candidate, dict):
        return None
    status = str(candidate.get("entry_ai_runtime_status") or "").strip().lower()
    reason_text = str(candidate.get("entry_ai_runtime_reason") or "").strip()
    if not status:
        return None
    if status == "scored":
        return None
    mapping = {
        "disabled": ("disabled", "ยังไม่ประเมิน", "⚪", "Entry AI ถูกปิดในรอบนี้"),
        "model_unavailable": ("unavailable", "ยังไม่ประเมิน", "⚪", "รอบนี้โหลด Entry AI ไม่สำเร็จ"),
        "not_allowed": ("not_scored", "ยังไม่ประเมิน", "⚪", "alert นี้อยู่นอกขอบเขตของ Entry AI"),
        "score_failed": ("error", "ยังไม่ประเมิน", "⚪", "รอบนี้ Entry AI ประเมินไม่สำเร็จ"),
    }
    bucket, label, icon, default_reason = mapping.get(
        status,
        ("unknown", "ยังไม่ประเมิน", "⚪", "รอบนี้ยังไม่มีผล Entry AI"),
    )
    return {
        "bucket": bucket,
        "label": label,
        "icon": icon,
        "reason": reason_text or default_reason,
        "prob_entry": None,
        "prob_watch": None,
        "prob_avoid": None,
        "rank_adjustment": 0.0,
    }


def _entry_ai_policy_text(value):
    bucket = str(value or "").strip().lower()
    if bucket == "entry":
        return "เข้าได้"
    if bucket == "avoid":
        return "ห้ามเข้า"
    if bucket == "watch":
        return "รอ"
    return None


def resolve_entry_ai_profile(candidate, *, config):
    if not isinstance(candidate, dict):
        return None
    prob_entry = _safe_float(candidate.get("entry_ai_prob_entry"), None)
    prob_watch = _safe_float(candidate.get("entry_ai_prob_watch"), None)
    prob_avoid = _safe_float(candidate.get("entry_ai_prob_avoid"), None)
    explicit_bucket = str(candidate.get("entry_ai_bucket") or "").strip().lower()
    policy_mode = str(candidate.get("entry_ai_policy_mode") or "").strip().lower()
    policy_tier = str(candidate.get("entry_ai_policy_tier") or "").strip().lower()
    premium_label = str(candidate.get("entry_ai_premium_label") or "").strip().lower()
    standard_label = str(candidate.get("entry_ai_standard_label") or "").strip().lower()
    if not explicit_bucket and prob_entry is None and prob_watch is None and prob_avoid is None:
        return _runtime_entry_ai_profile(candidate)

    entry_threshold = _safe_float(getattr(config, "TELEGRAM_ALERT_ENTRY_AI_ENTRY_THRESHOLD", 0.45), 0.45)
    avoid_threshold = _safe_float(getattr(config, "TELEGRAM_ALERT_ENTRY_AI_AVOID_THRESHOLD", 0.55), 0.55)
    entry_bonus = _safe_float(getattr(config, "TELEGRAM_ALERT_ENTRY_AI_ENTRY_SCORE_BONUS", 4.0), 4.0)
    premium_bonus = _safe_float(getattr(config, "TELEGRAM_ALERT_ENTRY_AI_PREMIUM_SCORE_BONUS", entry_bonus), entry_bonus)
    standard_bonus = _safe_float(
        getattr(config, "TELEGRAM_ALERT_ENTRY_AI_STANDARD_SCORE_BONUS", max(float(entry_bonus) * 0.5, 0.0)),
        max(float(entry_bonus) * 0.5, 0.0),
    )
    watch_penalty = _safe_float(getattr(config, "TELEGRAM_ALERT_ENTRY_AI_WATCH_SCORE_PENALTY", 0.0), 0.0)
    avoid_penalty = _safe_float(getattr(config, "TELEGRAM_ALERT_ENTRY_AI_AVOID_SCORE_PENALTY", 1.5), 1.5)

    if policy_mode == "premium_standard" or premium_label or standard_label or policy_tier in {"premium", "standard"}:
        bucket = explicit_bucket or "watch"
        label = "รอ"
        icon = "🟡"
        reason = "ยังไม่ถึงจังหวะเข้า"
        rank_adjustment = 0.0
        tier = None
        resolved_policy_tier = policy_tier if policy_tier in {"premium", "standard", "watch", "avoid"} else "watch"

        if policy_tier == "premium" or premium_label == "entry":
            bucket = "entry"
            label = "เข้าได้"
            icon = "🟢"
            tier = "Premium"
            resolved_policy_tier = "premium"
            reason = "ผ่านเกณฑ์ Premium เข้าได้ค่อนข้างดี"
            if bool(getattr(config, "TELEGRAM_ALERT_ENTRY_AI_RANKING_ENABLE", True)):
                rank_adjustment = float(premium_bonus)
        elif policy_tier == "standard" or standard_label == "entry":
            bucket = "entry"
            label = "เข้าได้"
            icon = "🟢"
            tier = "Standard"
            resolved_policy_tier = "standard"
            reason = "ผ่านเกณฑ์ Standard เข้าได้ แต่ยังไม่คมเท่า Premium"
            if bool(getattr(config, "TELEGRAM_ALERT_ENTRY_AI_RANKING_ENABLE", True)):
                rank_adjustment = float(standard_bonus)
        elif policy_tier == "avoid" or standard_label == "avoid" or premium_label == "avoid" or explicit_bucket == "avoid":
            bucket = "avoid"
            label = "ห้ามเข้า"
            icon = "⛔"
            resolved_policy_tier = "avoid"
            reason = "จุดเข้ายังไม่คุ้ม ควรหลีกเลี่ยงรอบนี้"
            if bool(getattr(config, "TELEGRAM_ALERT_ENTRY_AI_RANKING_ENABLE", True)):
                rank_adjustment = -abs(float(avoid_penalty))
        else:
            bucket = "watch"
            label = "รอ"
            icon = "🟡"
            resolved_policy_tier = "watch"
            if premium_label == "watch" and standard_label == "watch":
                reason = "ยังไม่ผ่านทั้ง Premium และ Standard"
            if bool(getattr(config, "TELEGRAM_ALERT_ENTRY_AI_RANKING_ENABLE", True)) and float(watch_penalty) > 0:
                rank_adjustment = -abs(float(watch_penalty))

        return {
            "bucket": bucket,
            "label": label,
            "icon": icon,
            "reason": reason,
            "prob_entry": prob_entry,
            "prob_watch": prob_watch,
            "prob_avoid": prob_avoid,
            "rank_adjustment": float(rank_adjustment),
            "policy_mode": "premium_standard",
            "tier": tier,
            "policy_tier": resolved_policy_tier,
            "premium_label": premium_label or None,
            "standard_label": standard_label or None,
        }

    bucket = explicit_bucket or "watch"
    if isinstance(prob_avoid, float) and prob_avoid >= avoid_threshold:
        bucket = "avoid"
    elif isinstance(prob_entry, float) and prob_entry >= entry_threshold:
        bucket = "entry"
    elif bucket not in {"entry", "watch", "avoid"}:
        bucket = "watch"

    label = "รอ"
    icon = "🟡"
    reason = "จังหวะนี้ยังไม่ชัด ควรรอดูเพิ่ม"
    rank_adjustment = 0.0
    if bucket == "entry":
        label = "เข้าได้"
        icon = "🟢"
        reason = "จุดเข้าเริ่มพร้อม ใช้ประกอบการเข้าได้"
        if bool(getattr(config, "TELEGRAM_ALERT_ENTRY_AI_RANKING_ENABLE", True)):
            rank_adjustment = float(entry_bonus)
    elif bucket == "avoid":
        label = "ห้ามเข้า"
        icon = "⛔"
        reason = "จุดเข้ายังไม่คุ้ม ควรเลี่ยงไปก่อน"
        if bool(getattr(config, "TELEGRAM_ALERT_ENTRY_AI_RANKING_ENABLE", True)):
            rank_adjustment = -abs(float(avoid_penalty))

    return {
        "bucket": bucket,
        "label": label,
        "icon": icon,
        "reason": reason,
        "prob_entry": prob_entry,
        "prob_watch": prob_watch,
        "prob_avoid": prob_avoid,
        "rank_adjustment": float(rank_adjustment),
        "policy_mode": "single",
        "tier": None,
        "policy_tier": bucket,
        "premium_label": None,
        "standard_label": None,
    }


def _append_entry_ai_message_line(message, profile, *, config):
    if not bool(getattr(config, "TELEGRAM_ALERT_ENTRY_AI_MESSAGE_ENABLE", True)):
        return message
    if not isinstance(message, str) or not message.strip() or not isinstance(profile, dict):
        return message
    if "🧠 Entry AI:" in message:
        return message
    parts = [str(profile.get("label") or "รอ")]
    tier = str(profile.get("tier") or "").strip()
    policy_mode = str(profile.get("policy_mode") or "").strip().lower()
    premium_label = _entry_ai_policy_text(profile.get("premium_label"))
    standard_label = _entry_ai_policy_text(profile.get("standard_label"))
    if tier:
        parts.append(tier)
    elif policy_mode == "premium_standard":
        if premium_label:
            parts.append(f"P {premium_label}")
        if standard_label:
            parts.append(f"S {standard_label}")
    prob_entry = _safe_float(profile.get("prob_entry"), None)
    prob_avoid = _safe_float(profile.get("prob_avoid"), None)
    if prob_entry is not None:
        parts.append(f"p(entry) {prob_entry * 100.0:.0f}%")
    if prob_avoid is not None:
        parts.append(f"p(avoid) {prob_avoid * 100.0:.0f}%")
    reason = str(profile.get("reason") or "").strip()
    if reason and prob_entry is None and prob_avoid is None:
        parts.append(reason)
    line = "<b>🧠 Entry AI:</b> " + " | ".join(parts)
    return message.rstrip() + "\n" + line


def _append_sltp_message_line(message, profile):
    if not isinstance(message, str) or not message.strip() or not isinstance(profile, dict):
        return message
    if "<b>SL/TP:</b>" in message:
        return message
    parts = [str(profile.get("label") or "Watch")]
    entry_gap_pct = _safe_float(profile.get("entry_gap_pct"), None)
    target_reward_pct = _safe_float(profile.get("target_reward_pct"), None)
    rr_ratio = _safe_float(profile.get("rr_ratio"), None)
    if entry_gap_pct is not None:
        parts.append(f"gap {entry_gap_pct:.2f}%")
    if target_reward_pct is not None:
        parts.append(f"target {target_reward_pct:.2f}%")
    if rr_ratio is not None:
        parts.append(f"RR {rr_ratio:.2f}")
    if len(parts) == 1:
        reason = str(profile.get("reason") or "").strip()
        if reason:
            parts.append(reason)
    line = "<b>SL/TP:</b> " + " | ".join(parts)
    return message.rstrip() + "\n" + line


def attach_entry_ai_context(candidate, *, config):
    if not isinstance(candidate, dict):
        return candidate
    profile = resolve_entry_ai_profile(candidate, config=config)
    if not isinstance(profile, dict):
        return candidate
    candidate["entry_ai_bucket"] = str(profile.get("bucket") or "").strip().lower() or None
    candidate["entry_ai_label"] = str(profile.get("label") or "").strip() or None
    candidate["entry_ai_icon"] = str(profile.get("icon") or "").strip() or None
    candidate["entry_ai_reason"] = str(profile.get("reason") or "").strip() or None
    candidate["entry_ai_rank_adjustment"] = _safe_float(profile.get("rank_adjustment"), 0.0)
    candidate["entry_ai_prob_entry"] = _safe_float(profile.get("prob_entry"), None)
    candidate["entry_ai_prob_watch"] = _safe_float(profile.get("prob_watch"), None)
    candidate["entry_ai_prob_avoid"] = _safe_float(profile.get("prob_avoid"), None)
    candidate["entry_ai_policy_mode"] = str(profile.get("policy_mode") or "").strip().lower() or None
    candidate["entry_ai_policy_tier"] = str(profile.get("policy_tier") or "").strip().lower() or None
    candidate["entry_ai_premium_label"] = str(profile.get("premium_label") or "").strip().lower() or None
    candidate["entry_ai_standard_label"] = str(profile.get("standard_label") or "").strip().lower() or None
    candidate["message"] = _append_entry_ai_message_line(candidate.get("message"), profile, config=config)
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


def _plan_numeric(candidate, keys):
    if not isinstance(candidate, dict):
        return None
    return _pick_numeric(candidate.get("plan"), keys)


def _candidate_source_count(candidate):
    if not isinstance(candidate, dict):
        return None
    value = candidate.get("source_count")
    if isinstance(value, (int, float)):
        try:
            return max(0, int(float(value)))
        except Exception:
            return None
    for key in ("sources", "supporting_sources", "source_labels"):
        entries = candidate.get(key)
        if isinstance(entries, (list, tuple, set)):
            return len(entries)
    plan = candidate.get("plan")
    if isinstance(plan, dict):
        for key in ("sources", "supporting_sources", "source_labels"):
            entries = plan.get(key)
            if isinstance(entries, (list, tuple, set)):
                return len(entries)
    return None


def _bars_since_signal(candidate):
    value = _plan_numeric(candidate, ["bars_since_signal", "bars_since_entry", "bars_since_cross"])
    if isinstance(value, float):
        return value
    return None


def _sell_continuation_entry_window_override(candidate, *, config):
    if not bool(getattr(config, "TELEGRAM_ALERT_SELL_CONTINUATION_ENABLE", True)):
        return None
    if not isinstance(candidate, dict):
        return None
    if _signal_side(candidate) != "SELL":
        return None
    regime = candidate.get("regime") if isinstance(candidate.get("regime"), dict) else {}
    market_regime = str(regime.get("market_regime") or "").strip().upper()
    side_bias = str(regime.get("side_bias") or "").strip().upper()
    if market_regime != "TREND_DOWN" or side_bias != "SELL":
        return None
    confidence = _safe_float(candidate.get("confidence"), None)
    min_confidence = _safe_float(getattr(config, "TELEGRAM_ALERT_SELL_CONTINUATION_MIN_CONFIDENCE", 88.0), 88.0)
    if confidence is None or confidence < float(min_confidence):
        return None
    bars_since = _bars_since_signal(candidate)
    max_bars_since = int(_safe_float(getattr(config, "TELEGRAM_ALERT_SELL_CONTINUATION_MAX_BARS_SINCE_SIGNAL", 2), 2))
    if isinstance(bars_since, float) and bars_since > float(max_bars_since):
        return None
    ai_bucket = str(candidate.get("ai_dispatch_bucket") or "").strip().lower()
    if ai_bucket == "low_conviction":
        return None
    return {
        "max_distance_pct": _safe_float(
            getattr(config, "TELEGRAM_ALERT_SELL_CONTINUATION_MAX_DISTANCE_PCT", 3.6),
            3.6,
        ),
        "max_distance_r": _safe_float(
            getattr(config, "TELEGRAM_ALERT_SELL_CONTINUATION_MAX_DISTANCE_R", 1.5),
            1.5,
        ),
    }


def _cdc_sell_continuation_override(candidate, *, config, is_entry_window=None, distance_pct=None, distance_r=None):
    if not bool(getattr(config, "TELEGRAM_ALERT_SELL_CONTINUATION_ENABLE", True)):
        return None
    if not bool(getattr(config, "TELEGRAM_ALERT_SELL_CONTINUATION_TIME_STOP_ENABLE", True)):
        return None
    if not isinstance(candidate, dict):
        return None
    if str(candidate.get("strategy") or "").strip().upper() != "CDCVIX15":
        return None
    if _signal_side(candidate) != "SELL":
        return None
    plan = candidate.get("plan") if isinstance(candidate.get("plan"), dict) else {}
    if not isinstance(plan, dict):
        return None

    plan_mode = str(plan.get("sell_continuation_override_mode") or "").strip().lower()
    plan_reason = str(plan.get("sell_continuation_override_reason") or "").strip()
    trigger = str(plan.get("sell_trigger") or plan.get("exit_trigger") or "").strip().upper()
    if plan_mode not in {"entry", "watch"} and trigger != "TIME_STOP":
        return None

    regime = candidate.get("regime") if isinstance(candidate.get("regime"), dict) else {}
    market_regime = str(regime.get("market_regime") or "").strip().upper()
    side_bias = str(regime.get("side_bias") or "").strip().upper()
    if market_regime != "TREND_DOWN" or side_bias != "SELL":
        return None

    confidence = _safe_float(candidate.get("confidence"), None)
    min_confidence = _safe_float(getattr(config, "TELEGRAM_ALERT_SELL_CONTINUATION_MIN_CONFIDENCE", 88.0), 88.0)
    if confidence is None or confidence < float(min_confidence):
        return None

    forecast_dir = str(plan.get("forecast_direction") or "").strip().upper()
    forecast_score = _safe_float(plan.get("forecast_score"), None)
    min_forecast_score = _safe_float(getattr(config, "TELEGRAM_ALERT_SELL_CONTINUATION_MIN_FORECAST_SCORE", 80.0), 80.0)
    if forecast_dir != "SELL":
        return None
    if forecast_score is not None and forecast_score < float(min_forecast_score):
        return None

    if is_entry_window is None:
        is_entry_window, distance_pct, distance_r = _within_entry_window(candidate, config=config)
    if plan_mode == "entry" and is_entry_window is False:
        plan_mode = "watch"
    if plan_mode == "watch" and is_entry_window is True:
        plan_mode = "entry"
    if plan_mode not in {"entry", "watch"}:
        plan_mode = "entry" if is_entry_window else "watch"

    reason = plan_reason or (
        f"cdc_time_stop_sell_continuation:d_pct={distance_pct},d_r={distance_r}"
        if plan_mode == "entry"
        else f"cdc_time_stop_sell_watch:d_pct={distance_pct},d_r={distance_r}"
    )
    return plan_mode, reason


def _signal_side(candidate):
    return str((candidate or {}).get("signal") or "").strip().upper()


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
    override = _sell_continuation_entry_window_override(candidate, config=config)
    if isinstance(override, dict):
        max_distance_pct = _safe_float(override.get("max_distance_pct"), max_distance_pct)
        max_distance_r = _safe_float(override.get("max_distance_r"), max_distance_r)
    near_by_pct = isinstance(distance_pct, float) and distance_pct <= float(max_distance_pct)
    near_by_r = isinstance(distance_r, float) and distance_r <= float(max_distance_r)
    if near_by_pct or near_by_r:
        return True, distance_pct, distance_r
    if distance_pct is None and distance_r is None:
        return None, None, None
    return False, distance_pct, distance_r


def _candidate_group(candidate):
    return str((candidate or {}).get("candidate_group") or "").strip().upper()


def _plan_price_metrics(candidate):
    if not isinstance(candidate, dict):
        return {}
    plan = candidate.get("plan") if isinstance(candidate.get("plan"), dict) else {}
    item = candidate.get("item") if isinstance(candidate.get("item"), dict) else {}
    entry_price = _pick_numeric(plan, ["entry_price", "current_price", "price"])
    current_price = _safe_float(candidate.get("price_at_checkpoint"), None)
    if current_price is None:
        current_price = _pick_numeric(plan, ["current_price", "price"])
    if current_price is None and isinstance(item, dict):
        current_price = _safe_float(item.get("current_price") or item.get("price"), None)
    stop_loss = _pick_numeric(plan, ["stop_loss", "entry_stop_loss", "trailing_stop"])
    take_profit = _pick_numeric(plan, ["take_profit", "take_profit_2", "exit_price"])
    rr_value = _pick_numeric(plan, ["risk_reward", "rr", "rr_ratio"])

    entry_gap_pct = None
    stop_risk_pct = None
    target_reward_pct = None
    rr_ratio = None
    if isinstance(entry_price, float) and entry_price:
        entry_abs = abs(float(entry_price))
        if isinstance(current_price, float):
            entry_gap_pct = abs(float(current_price) - float(entry_price)) / entry_abs * 100.0
        if isinstance(stop_loss, float):
            stop_risk_pct = abs(float(entry_price) - float(stop_loss)) / entry_abs * 100.0
        if isinstance(take_profit, float):
            target_reward_pct = abs(float(take_profit) - float(entry_price)) / entry_abs * 100.0
    if target_reward_pct is None and isinstance(rr_value, float) and isinstance(stop_risk_pct, float):
        target_reward_pct = abs(float(rr_value)) * float(stop_risk_pct)
    if isinstance(target_reward_pct, float) and isinstance(stop_risk_pct, float) and stop_risk_pct > 0:
        rr_ratio = float(target_reward_pct) / float(stop_risk_pct)
    elif isinstance(rr_value, float):
        rr_ratio = abs(float(rr_value))
    return {
        "entry_price": entry_price,
        "current_price": current_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "entry_gap_pct": entry_gap_pct,
        "stop_risk_pct": stop_risk_pct,
        "target_reward_pct": target_reward_pct,
        "rr_ratio": rr_ratio,
    }


def resolve_sltp_live_profile(candidate, *, config):
    if not bool(getattr(config, "TELEGRAM_ALERT_SLTP_PROFILE_ENABLE", True)):
        return None
    if not isinstance(candidate, dict):
        return None

    strategy = str(candidate.get("strategy") or "").strip().upper()
    candidate_group = _candidate_group(candidate)
    configured_strategies = set(getattr(config, "TELEGRAM_ALERT_SLTP_LIVE_STRATEGIES", set()) or set())
    configured_groups = set(getattr(config, "TELEGRAM_ALERT_SLTP_LIVE_GROUPS", set()) or set())
    if configured_strategies and strategy not in configured_strategies:
        return None
    if configured_groups and candidate_group not in configured_groups:
        return None

    intent = str(candidate.get("alert_intent") or "").strip().lower()
    signal = _signal_side(candidate)
    if intent not in {"entry", "watch"} or signal not in {"BUY", "SELL"}:
        return None

    metrics = _plan_price_metrics(candidate)
    entry_gap_pct = _safe_float(metrics.get("entry_gap_pct"), None)
    stop_risk_pct = _safe_float(metrics.get("stop_risk_pct"), None)
    target_reward_pct = _safe_float(metrics.get("target_reward_pct"), None)
    rr_ratio = _safe_float(metrics.get("rr_ratio"), None)
    if rr_ratio is None or target_reward_pct is None:
        return None

    max_entry_gap_pct = _safe_float(getattr(config, "TELEGRAM_ALERT_SLTP_MAX_ENTRY_GAP_PCT", 0.80), 0.80)
    hard_avoid_min_rr = _safe_float(getattr(config, "TELEGRAM_ALERT_SLTP_HARD_AVOID_MIN_RR", 0.70), 0.70)
    hard_avoid_min_target = _safe_float(getattr(config, "TELEGRAM_ALERT_SLTP_HARD_AVOID_MIN_TARGET_REWARD_PCT", 0.75), 0.75)
    premium_bonus = _safe_float(getattr(config, "TELEGRAM_ALERT_SLTP_PREMIUM_SCORE_BONUS", 3.0), 3.0)
    standard_bonus = _safe_float(getattr(config, "TELEGRAM_ALERT_SLTP_STANDARD_SCORE_BONUS", 1.25), 1.25)
    watch_penalty = _safe_float(getattr(config, "TELEGRAM_ALERT_SLTP_WATCH_SCORE_PENALTY", 1.5), 1.5)
    avoid_penalty = _safe_float(getattr(config, "TELEGRAM_ALERT_SLTP_AVOID_SCORE_PENALTY", 3.0), 3.0)

    side_prefix = "BUY" if signal == "BUY" else "SELL"
    min_rr = _safe_float(getattr(config, f"TELEGRAM_ALERT_SLTP_{side_prefix}_MIN_RR", 1.0), 1.0)
    min_target = _safe_float(getattr(config, f"TELEGRAM_ALERT_SLTP_{side_prefix}_MIN_TARGET_REWARD_PCT", 1.25), 1.25)
    bonus_rr = _safe_float(getattr(config, f"TELEGRAM_ALERT_SLTP_{side_prefix}_BONUS_RR", 2.0), 2.0)
    bonus_target = _safe_float(getattr(config, f"TELEGRAM_ALERT_SLTP_{side_prefix}_BONUS_TARGET_REWARD_PCT", 3.0), 3.0)

    profile = {
        "bucket": "watch",
        "label": "Watch",
        "reason": "SL/TP ยังไม่เด่นพอสำหรับเพิ่มน้ำหนักรอบนี้",
        "score_adjustment": 0.0,
        "block": False,
        "block_reason": None,
        "entry_gap_pct": entry_gap_pct,
        "stop_risk_pct": stop_risk_pct,
        "target_reward_pct": target_reward_pct,
        "rr_ratio": rr_ratio,
    }

    if rr_ratio < float(hard_avoid_min_rr) or target_reward_pct < float(hard_avoid_min_target):
        profile.update(
            {
                "bucket": "avoid",
                "label": "Avoid",
                "reason": f"{signal} reward/risk บางเกินไป (RR={rr_ratio:.2f}, target={target_reward_pct:.2f}%)",
                "score_adjustment": -abs(float(avoid_penalty)),
                "block": True,
                "block_reason": f"sltp_{signal.lower()}_hard_avoid",
            }
        )
        return profile

    if signal == "BUY":
        stop_min = _safe_float(getattr(config, "TELEGRAM_ALERT_SLTP_BUY_PREFERRED_STOP_MIN_PCT", 0.75), 0.75)
        stop_max = _safe_float(getattr(config, "TELEGRAM_ALERT_SLTP_BUY_PREFERRED_STOP_MAX_PCT", 1.80), 1.80)
        if (
            rr_ratio >= float(bonus_rr)
            and target_reward_pct >= float(bonus_target)
            and (entry_gap_pct is None or entry_gap_pct <= min(float(max_entry_gap_pct), 0.35))
            and (stop_risk_pct is None or (float(stop_min) <= stop_risk_pct <= float(stop_max)))
        ):
            profile.update(
                {
                    "bucket": "premium",
                    "label": "Premium",
                    "reason": "BUY ได้ reward เด่นและ RR สูงพอ แม้ฝั่งนี้ต้องคัดเข้ม",
                    "score_adjustment": float(premium_bonus),
                }
            )
        elif rr_ratio >= float(min_rr) and target_reward_pct >= float(min_target) and (
            entry_gap_pct is None or entry_gap_pct <= float(max_entry_gap_pct)
        ):
            profile.update(
                {
                    "bucket": "standard",
                    "label": "Standard",
                    "reason": "BUY ยังพอถือเป็น setup เข้าได้ แต่ยังไม่ใช่โซนเด่นสุด",
                    "score_adjustment": float(standard_bonus),
                }
            )
        else:
            profile.update(
                {
                    "bucket": "watch",
                    "label": "Watch",
                    "reason": "BUY ฝั่งนี้ยังต้องการ reward/RR สูงกว่านี้ก่อนค่อยเร่งน้ำหนัก",
                    "score_adjustment": -abs(float(watch_penalty)),
                }
            )
    else:
        bonus_gap = _safe_float(getattr(config, "TELEGRAM_ALERT_SLTP_SELL_BONUS_ENTRY_GAP_PCT", 0.15), 0.15)
        if (
            rr_ratio >= float(bonus_rr)
            and target_reward_pct >= float(bonus_target)
            and (entry_gap_pct is None or entry_gap_pct <= float(bonus_gap))
        ):
            profile.update(
                {
                    "bucket": "premium",
                    "label": "Premium",
                    "reason": "SELL ใกล้ entry zone และได้ reward/RR ตาม bucket เด่นของรอบเทรน",
                    "score_adjustment": float(premium_bonus),
                }
            )
        elif rr_ratio >= float(min_rr) and target_reward_pct >= float(min_target) and (
            entry_gap_pct is None or entry_gap_pct <= float(max_entry_gap_pct)
        ):
            profile.update(
                {
                    "bucket": "standard",
                    "label": "Standard",
                    "reason": "SELL ผ่านเกณฑ์ reward/RR พื้นฐาน ใช้เพิ่มน้ำหนักได้",
                    "score_adjustment": float(standard_bonus),
                }
            )
        else:
            profile.update(
                {
                    "bucket": "watch",
                    "label": "Watch",
                    "reason": "SELL ยังไม่ใกล้ entry zone หรือ reward/RR ยังไม่เด่นพอ",
                    "score_adjustment": -abs(float(watch_penalty)),
                }
            )
    return profile


def attach_sltp_live_context(candidate, *, config):
    if not isinstance(candidate, dict):
        return candidate
    profile = resolve_sltp_live_profile(candidate, config=config)
    if not isinstance(profile, dict):
        return candidate
    candidate["sltp_live_label"] = str(profile.get("label") or "").strip() or None
    candidate["sltp_live_bucket"] = str(profile.get("bucket") or "").strip().lower() or None
    candidate["sltp_live_reason"] = str(profile.get("reason") or "").strip() or None
    candidate["sltp_live_score_adjustment"] = _safe_float(profile.get("score_adjustment"), 0.0)
    candidate["sltp_live_entry_gap_pct"] = _safe_float(profile.get("entry_gap_pct"), None)
    candidate["sltp_live_stop_risk_pct"] = _safe_float(profile.get("stop_risk_pct"), None)
    candidate["sltp_live_target_reward_pct"] = _safe_float(profile.get("target_reward_pct"), None)
    candidate["sltp_live_rr_ratio"] = _safe_float(profile.get("rr_ratio"), None)
    candidate["message"] = _append_sltp_message_line(candidate.get("message"), profile)
    if profile.get("bucket") == "watch" and str(candidate.get("alert_intent") or "").strip().lower() == "entry":
        candidate["alert_intent"] = "watch"
        candidate["alert_intent_reason"] = f"sltp_watch:{profile.get('reason')}"
    return candidate


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
    is_entry_window, distance_pct, distance_r = _within_entry_window(candidate, config=config)
    if strategy == "CDCVIX15":
        continuation_override = _cdc_sell_continuation_override(
            candidate,
            config=config,
            is_entry_window=is_entry_window,
            distance_pct=distance_pct,
            distance_r=distance_r,
        )
        if isinstance(continuation_override, tuple):
            return continuation_override
    if trigger in _EXIT_TRIGGERS or any(phrase in plan_reason for phrase in _EXIT_REASON_PHRASES):
        return "exit", f"trigger:{trigger.lower() or 'plan_reason_exit'}"
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


def resolve_short_trade_profile(candidate, *, config):
    if not bool(getattr(config, "TELEGRAM_ALERT_SHORT_TRADE_ENABLE", True)):
        return None
    if not isinstance(candidate, dict):
        return None

    intent = str(candidate.get("alert_intent") or "").strip().lower()
    signal = _signal_side(candidate)
    if signal not in ("BUY", "SELL"):
        return None

    profile = {
        "bucket": "watch",
        "label": "Watch Only",
        "reason": "ยังไม่ผ่านคุณภาพจังหวะเข้าแบบเทรดสั้น",
        "score_adjustment": 0.0,
        "block": False,
        "block_reason": None,
    }

    plan = candidate.get("plan") if isinstance(candidate.get("plan"), dict) else {}
    rr_value = _safe_float(plan.get("risk_reward"), None)
    bars_since = _bars_since_signal(candidate)
    source_count = _candidate_source_count(candidate)
    confidence = _safe_float(candidate.get("confidence"), None)
    is_entry_window, distance_pct, distance_r = _within_entry_window(candidate, config=config)
    regime = candidate.get("regime") if isinstance(candidate.get("regime"), dict) else {}
    market_regime = str(regime.get("market_regime") or "").strip().upper()
    side_bias = str(regime.get("side_bias") or "").strip().upper()
    require_regime_alignment = bool(getattr(config, "TELEGRAM_ALERT_SHORT_TRADE_REQUIRE_REGIME_ALIGNMENT", True))
    regime_aligned = True
    if signal in ("BUY", "SELL") and side_bias in ("BUY", "SELL") and signal != side_bias:
        regime_aligned = False
    if market_regime == "RISK_OFF_EVENT" and signal == "BUY":
        regime_aligned = False

    rr_floor = _safe_float(getattr(config, "TELEGRAM_ALERT_SHORT_TRADE_RR_FLOOR", 1.0), 1.0)
    premium_rr = _safe_float(getattr(config, "TELEGRAM_ALERT_SHORT_TRADE_PREMIUM_MIN_RR", 1.25), 1.25)
    standard_max_bars = int(_safe_float(getattr(config, "TELEGRAM_ALERT_SHORT_TRADE_STANDARD_MAX_BARS", 3), 3))
    hard_stale_bars = int(_safe_float(getattr(config, "TELEGRAM_ALERT_SHORT_TRADE_HARD_STALE_BARS", 6), 6))
    standard_min_sources = int(_safe_float(getattr(config, "TELEGRAM_ALERT_SHORT_TRADE_STANDARD_MIN_SOURCE_COUNT", 1), 1))
    premium_min_sources = int(_safe_float(getattr(config, "TELEGRAM_ALERT_SHORT_TRADE_PREMIUM_MIN_SOURCE_COUNT", 2), 2))
    premium_min_confidence = _safe_float(getattr(config, "TELEGRAM_ALERT_SHORT_TRADE_PREMIUM_MIN_CONFIDENCE", 78.0), 78.0)
    premium_bonus = _safe_float(getattr(config, "TELEGRAM_ALERT_SHORT_TRADE_PREMIUM_SCORE_BONUS", 4.0), 4.0)
    standard_bonus = _safe_float(getattr(config, "TELEGRAM_ALERT_SHORT_TRADE_STANDARD_SCORE_BONUS", 1.5), 1.5)
    watch_penalty = _safe_float(getattr(config, "TELEGRAM_ALERT_SHORT_TRADE_WATCH_SCORE_PENALTY", 3.0), 3.0)
    ai_bucket = str(candidate.get("ai_dispatch_bucket") or "").strip().lower()

    if intent == "entry":
        if isinstance(rr_value, float) and rr_value < float(rr_floor):
            profile.update(
                {
                    "bucket": "avoid",
                    "label": "Avoid Entry",
                    "reason": f"RR ต่ำกว่า {rr_floor:.2f}R",
                    "score_adjustment": -abs(float(watch_penalty)),
                    "block": True,
                    "block_reason": "short_trade_rr_floor",
                }
            )
        elif isinstance(bars_since, float) and bars_since > float(hard_stale_bars):
            profile.update(
                {
                    "bucket": "avoid",
                    "label": "Avoid Entry",
                    "reason": f"สัญญาณเกิน {hard_stale_bars} แท่ง ไม่สดพอ",
                    "score_adjustment": -abs(float(watch_penalty)),
                    "block": True,
                    "block_reason": "short_trade_stale_entry",
                }
            )
        elif is_entry_window is False:
            profile.update(
                {
                    "bucket": "watch",
                    "label": "Watch Only",
                    "reason": "ราคาเริ่มห่าง entry zone",
                    "score_adjustment": -abs(float(watch_penalty)),
                }
            )
        elif require_regime_alignment and not regime_aligned:
            profile.update(
                {
                    "bucket": "watch",
                    "label": "Watch Only",
                    "reason": "สวน market regime หรือ side bias หลัก",
                    "score_adjustment": -abs(float(watch_penalty)),
                }
            )
        elif ai_bucket == "low_conviction":
            profile.update(
                {
                    "bucket": "watch",
                    "label": "Watch Only",
                    "reason": "AI ยังไม่มั่นใจพอสำหรับ short entry",
                    "score_adjustment": -abs(float(watch_penalty)),
                }
            )
        elif (
            (is_entry_window is True or is_entry_window is None)
            and (rr_value is None or rr_value >= float(premium_rr))
            and (bars_since is None or bars_since <= float(standard_max_bars))
            and (source_count is None or int(source_count) >= int(premium_min_sources))
            and (confidence is not None and confidence >= float(premium_min_confidence))
        ):
            profile.update(
                {
                    "bucket": "premium_entry",
                    "label": "Premium Entry",
                    "reason": "เทรนด์ตรง, RR ดี, สัญญาณสด และคุณภาพเข้าเด่น",
                    "score_adjustment": float(premium_bonus),
                }
            )
        elif (
            (is_entry_window is True or is_entry_window is None)
            and (rr_value is None or rr_value >= float(rr_floor))
            and (bars_since is None or bars_since <= float(standard_max_bars))
            and (source_count is None or int(source_count) >= int(standard_min_sources))
        ):
            profile.update(
                {
                    "bucket": "standard_entry",
                    "label": "Standard Entry",
                    "reason": "เข้าได้ตามแผนเทรดสั้น แต่ยังไม่ถึงระดับพรีเมียม",
                    "score_adjustment": float(standard_bonus),
                }
            )
        else:
            profile.update(
                {
                    "bucket": "watch",
                    "label": "Watch Only",
                    "reason": "คุณภาพจังหวะเข้าไม่ครบสำหรับ short trade",
                    "score_adjustment": -abs(float(watch_penalty)),
                }
            )
    elif intent == "watch":
        profile.update(
            {
                "bucket": "watch",
                "label": "Watch Only",
                "reason": "เป็นแผนเฝ้ารอจังหวะ ไม่ใช่ entry ทันที",
                "score_adjustment": -abs(float(watch_penalty)),
            }
        )

    profile["risk_reward"] = rr_value
    profile["bars_since_signal"] = bars_since
    profile["source_count"] = source_count
    profile["entry_distance_pct"] = distance_pct
    profile["entry_distance_r"] = distance_r
    profile["regime_aligned"] = regime_aligned
    return profile


def attach_short_trade_context(candidate, *, config):
    if not isinstance(candidate, dict):
        return candidate
    profile = resolve_short_trade_profile(candidate, config=config)
    if not isinstance(profile, dict):
        return candidate
    candidate["short_trade_bucket"] = str(profile.get("bucket") or "").strip() or None
    candidate["short_trade_label"] = str(profile.get("label") or "").strip() or None
    candidate["short_trade_reason"] = str(profile.get("reason") or "").strip() or None
    candidate["short_trade_score_adjustment"] = _safe_float(profile.get("score_adjustment"), 0.0)
    candidate["short_trade_regime_aligned"] = bool(profile.get("regime_aligned")) if "regime_aligned" in profile else None
    candidate["short_trade_entry_distance_pct"] = _safe_float(profile.get("entry_distance_pct"), None)
    candidate["short_trade_entry_distance_r"] = _safe_float(profile.get("entry_distance_r"), None)
    if profile.get("bucket") == "watch" and str(candidate.get("alert_intent") or "").strip().lower() == "entry":
        candidate["alert_intent"] = "watch"
        candidate["alert_intent_reason"] = f"short_trade_watch:{profile.get('reason')}"
    return candidate


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
    score_candidate_with_live_entry_ai = context["helpers"].get("score_candidate_with_live_entry_ai")

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
        candidate = attach_short_trade_context(candidate, config=context["config"])
        candidate = attach_sltp_live_context(candidate, config=context["config"])
        if callable(score_candidate_with_live_entry_ai):
            candidate = score_candidate_with_live_entry_ai(candidate)
        candidate = attach_entry_ai_context(candidate, config=context["config"])
        if str(candidate.get("short_trade_bucket") or "").strip().lower() == "avoid":
            context["quality_drop_counts"][str(candidate.get("short_trade_reason") or "short_trade_avoid")] += 1
            continue
        if str(candidate.get("sltp_live_bucket") or "").strip().lower() == "avoid":
            context["quality_drop_counts"][str(candidate.get("sltp_live_reason") or "sltp_live_avoid")] += 1
            continue
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
