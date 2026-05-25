from collections import Counter


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
