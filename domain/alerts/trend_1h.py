def _normalize_trend_value(value):
    text = str(value or "").strip().upper()
    if text in ("UP", "BULLISH", "BUY", "LONG"):
        return "UP"
    if text in ("DOWN", "BEARISH", "SELL", "SHORT"):
        return "DOWN"
    if text in ("SIDEWAYS", "RANGE", "NEUTRAL"):
        return "SIDEWAYS"
    return None


def _collect_trend_votes(item):
    plan_specs = [
        ("ActionZone 15m", item.get("actionzone_15m")),
        ("Price Action 15m", item.get("price_action_15m")),
        ("Trend Breakout 15m", item.get("trend_breakout_15m")),
        ("Crypto Reversal 15m", item.get("crypto_reversal_15m")),
        ("Short Term 15m", item.get("short_term_15m")),
        ("Sniper 15m", item.get("sniper_15m")),
        ("Quantum 15m", item.get("quantum_15m")),
    ]
    votes = []
    for label, plan in plan_specs:
        if not isinstance(plan, dict):
            continue
        trend = _normalize_trend_value(plan.get("trend_1h"))
        if trend not in ("UP", "DOWN", "SIDEWAYS"):
            continue
        strength = str(plan.get("trend_strength_1h") or plan.get("strength") or "").strip().upper()
        weight = 1.0
        if strength == "STRONG":
            weight = 1.25
        votes.append(
            {
                "label": label,
                "trend": trend,
                "strength": strength if strength in ("STRONG", "WEAK") else None,
                "weight": weight,
            }
        )
    return votes


def infer_1h_trend_snapshot(item):
    if not isinstance(item, dict) or item.get("error"):
        return None
    votes = _collect_trend_votes(item)
    if not votes:
        return None
    up_score = sum(float(vote["weight"]) for vote in votes if vote["trend"] == "UP")
    down_score = sum(float(vote["weight"]) for vote in votes if vote["trend"] == "DOWN")
    side_score = sum(float(vote["weight"]) for vote in votes if vote["trend"] == "SIDEWAYS")
    directional_total = up_score + down_score
    if directional_total <= 0:
        return None
    trend = "UP" if up_score > down_score else "DOWN" if down_score > up_score else None
    if trend not in ("UP", "DOWN"):
        return None
    dominant_score = up_score if trend == "UP" else down_score
    agreement_ratio = dominant_score / directional_total if directional_total > 0 else 0.0
    confidence = 55.0 + min(40.0, agreement_ratio * 40.0)
    strong_votes = sum(1 for vote in votes if vote["trend"] == trend and vote.get("strength") == "STRONG")
    strength = "STRONG" if strong_votes > 0 or dominant_score >= 2.25 else "WEAK"
    source_labels = [vote["label"] for vote in votes if vote["trend"] == trend]
    opposing_labels = [vote["label"] for vote in votes if vote["trend"] != trend]
    return {
        "trend": trend,
        "strength": strength,
        "confidence": round(float(confidence), 1),
        "agreement_ratio": round(float(agreement_ratio), 3),
        "directional_votes": len([vote for vote in votes if vote["trend"] in ("UP", "DOWN")]),
        "source_labels": source_labels,
        "opposing_labels": opposing_labels,
        "votes": votes,
    }
