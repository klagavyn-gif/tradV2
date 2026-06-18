import argparse
import json
from pathlib import Path


def _safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _policy_snapshot(policy):
    policy = policy if isinstance(policy, dict) else {}
    return {
        "is_viable": bool(policy.get("is_viable")),
        "selected_rows": _safe_int(policy.get("selected_rows"), 0),
        "alerts_per_day": _safe_float(policy.get("alerts_per_day"), None),
        "win_rate_pct": _safe_float(policy.get("win_rate_pct"), None),
        "avg_return_pct": _safe_float(policy.get("avg_return_pct"), None),
        "entry_threshold": _safe_float(policy.get("entry_threshold"), None),
        "avoid_threshold": _safe_float(policy.get("avoid_threshold"), None),
    }


def _rule_result(snapshot, *, min_selected_rows=None, min_alerts_per_day=None, max_alerts_per_day=None, min_win_rate_pct=None, min_avg_return_pct=None):
    passed = []
    failed = []

    def _check_min(label, actual, minimum):
        if minimum is None:
            return
        actual_value = _safe_float(actual, None)
        if actual_value is None:
            failed.append(f"{label}_missing")
            return
        if float(actual_value) >= float(minimum):
            passed.append(f"{label}>={minimum}")
        else:
            failed.append(f"{label}<{minimum}")

    def _check_max(label, actual, maximum):
        if maximum is None:
            return
        actual_value = _safe_float(actual, None)
        if actual_value is None:
            failed.append(f"{label}_missing")
            return
        if float(actual_value) <= float(maximum):
            passed.append(f"{label}<={maximum}")
        else:
            failed.append(f"{label}>{maximum}")

    _check_min("selected_rows", snapshot.get("selected_rows"), min_selected_rows)
    _check_min("alerts_per_day", snapshot.get("alerts_per_day"), min_alerts_per_day)
    _check_max("alerts_per_day", snapshot.get("alerts_per_day"), max_alerts_per_day)
    _check_min("win_rate_pct", snapshot.get("win_rate_pct"), min_win_rate_pct)
    _check_min("avg_return_pct", snapshot.get("avg_return_pct"), min_avg_return_pct)
    return {
        "passed": passed,
        "failed": failed,
        "ok": not failed,
    }


GLOBAL_RULES = {
    "premium": [
        {
            "stage": "promote_live",
            "reason": "global premium holdout quality and sample size are strong enough for guarded live promotion",
            "min_selected_rows": 30,
            "min_alerts_per_day": 0.5,
            "max_alerts_per_day": 1.5,
            "min_win_rate_pct": 58.0,
            "min_avg_return_pct": 2.0,
        },
        {
            "stage": "canary",
            "reason": "global premium is viable but still better treated as a controlled canary",
            "min_selected_rows": 15,
            "min_alerts_per_day": 0.25,
            "max_alerts_per_day": 2.0,
            "min_win_rate_pct": 55.0,
            "min_avg_return_pct": 1.0,
        },
    ],
    "standard": [
        {
            "stage": "promote_live",
            "reason": "global standard has enough breadth and quality for default live usage",
            "min_selected_rows": 60,
            "min_alerts_per_day": 1.0,
            "max_alerts_per_day": 3.0,
            "min_win_rate_pct": 55.0,
            "min_avg_return_pct": 1.5,
        },
        {
            "stage": "canary",
            "reason": "global standard is viable but still needs guarded rollout",
            "min_selected_rows": 30,
            "min_alerts_per_day": 0.5,
            "max_alerts_per_day": 4.0,
            "min_win_rate_pct": 53.0,
            "min_avg_return_pct": 1.0,
        },
    ],
    "watch": [
        {
            "stage": "promote_monitor",
            "reason": "global watch can be used as the monitor tier in live mode",
            "min_selected_rows": 40,
            "min_alerts_per_day": 0.8,
            "max_alerts_per_day": 3.5,
            "min_win_rate_pct": 45.0,
            "min_avg_return_pct": 0.5,
        },
        {
            "stage": "monitor_only",
            "reason": "global watch is usable for monitoring but still not strong enough for broader rollout",
            "min_selected_rows": 20,
            "min_alerts_per_day": 0.3,
            "max_alerts_per_day": 5.0,
            "min_win_rate_pct": 40.0,
            "min_avg_return_pct": 0.0,
        },
    ],
}


STRATEGY_DEFAULT_RULES = {
    "premium": [
        {
            "stage": "promote_strategy_live",
            "reason": "strategy premium is strong enough to run with strategy-specific live thresholds",
            "min_selected_rows": 20,
            "min_alerts_per_day": 0.3,
            "max_alerts_per_day": 1.6,
            "min_win_rate_pct": 57.0,
            "min_avg_return_pct": 1.0,
        },
        {
            "stage": "strategy_canary",
            "reason": "strategy premium is viable and suitable for controlled canary rollout",
            "min_selected_rows": 10,
            "min_alerts_per_day": 0.15,
            "max_alerts_per_day": 2.0,
            "min_win_rate_pct": 54.0,
            "min_avg_return_pct": 0.5,
        },
    ],
    "standard": [
        {
            "stage": "promote_strategy_live",
            "reason": "strategy standard has enough breadth for live strategy-specific use",
            "min_selected_rows": 35,
            "min_alerts_per_day": 0.5,
            "max_alerts_per_day": 4.0,
            "min_win_rate_pct": 54.0,
            "min_avg_return_pct": 0.8,
        },
        {
            "stage": "strategy_canary",
            "reason": "strategy standard is viable and suitable for canary rollout",
            "min_selected_rows": 15,
            "min_alerts_per_day": 0.2,
            "max_alerts_per_day": 4.5,
            "min_win_rate_pct": 52.0,
            "min_avg_return_pct": 0.3,
        },
    ],
    "watch": [
        {
            "stage": "promote_strategy_monitor",
            "reason": "strategy watch can be used as a live monitor tier",
            "min_selected_rows": 40,
            "min_alerts_per_day": 0.7,
            "max_alerts_per_day": 4.0,
            "min_win_rate_pct": 40.0,
            "min_avg_return_pct": 0.0,
        },
        {
            "stage": "monitor_only",
            "reason": "strategy watch is usable only as a monitor layer for now",
            "min_selected_rows": 20,
            "min_alerts_per_day": 0.2,
            "max_alerts_per_day": 5.0,
            "min_win_rate_pct": 35.0,
            "min_avg_return_pct": -0.2,
        },
    ],
}


STRATEGY_RULE_OVERRIDES = {
    "PA15": {
        "premium": [
            {
                "stage": "strategy_canary",
                "reason": "PA15 premium is viable but sample size and average return still call for canary-only rollout",
                "min_selected_rows": 10,
                "min_alerts_per_day": 0.15,
                "max_alerts_per_day": 0.6,
                "min_win_rate_pct": 55.0,
                "min_avg_return_pct": 0.2,
            }
        ],
        "standard": [
            {
                "stage": "strategy_canary",
                "reason": "PA15 standard is viable but should stay in canary mode until live evidence grows",
                "min_selected_rows": 10,
                "min_alerts_per_day": 0.15,
                "max_alerts_per_day": 0.8,
                "min_win_rate_pct": 54.0,
                "min_avg_return_pct": 0.2,
            }
        ],
        "watch": [
            {
                "stage": "promote_strategy_monitor",
                "reason": "PA15 watch is ready for strategy-specific monitoring rollout",
                "min_selected_rows": 60,
                "min_alerts_per_day": 1.0,
                "max_alerts_per_day": 3.0,
                "min_win_rate_pct": 40.0,
                "min_avg_return_pct": 0.0,
            },
            {
                "stage": "monitor_only",
                "reason": "PA15 watch is only suitable as monitor support for now",
                "min_selected_rows": 30,
                "min_alerts_per_day": 0.3,
                "max_alerts_per_day": 4.0,
                "min_win_rate_pct": 38.0,
                "min_avg_return_pct": -0.2,
            },
        ],
    }
}


def _evaluate_policy_stage(policy, rule_chain, *, fallback_stage, fallback_reason):
    snapshot = _policy_snapshot(policy)
    if not snapshot["is_viable"]:
        return {
            "promotion_stage": "hold",
            "promotion_reason": "policy is not viable on holdout metrics",
            "checks": {"passed": [], "failed": ["is_viable=false"], "ok": False},
            "policy_snapshot": snapshot,
        }
    for rule in rule_chain:
        checks = _rule_result(
            snapshot,
            min_selected_rows=rule.get("min_selected_rows"),
            min_alerts_per_day=rule.get("min_alerts_per_day"),
            max_alerts_per_day=rule.get("max_alerts_per_day"),
            min_win_rate_pct=rule.get("min_win_rate_pct"),
            min_avg_return_pct=rule.get("min_avg_return_pct"),
        )
        if checks["ok"]:
            return {
                "promotion_stage": str(rule["stage"]),
                "promotion_reason": str(rule["reason"]),
                "checks": checks,
                "policy_snapshot": snapshot,
            }
    return {
        "promotion_stage": str(fallback_stage),
        "promotion_reason": str(fallback_reason),
        "checks": {
            "passed": ["is_viable=true"],
            "failed": ["stronger_promotion_rule_not_met"],
            "ok": False,
        },
        "policy_snapshot": snapshot,
    }


def _strategy_rules(strategy_name, policy_name):
    strategy_key = str(strategy_name or "").strip().upper()
    override = (STRATEGY_RULE_OVERRIDES.get(strategy_key) or {}).get(policy_name)
    if override:
        return override
    return STRATEGY_DEFAULT_RULES[policy_name]


def _global_recommendations(metrics_payload):
    out = {}
    for policy_name in ("premium", "standard", "watch"):
        policy = metrics_payload.get(f"recommended_{policy_name}_policy") or {}
        fallback_stage = "shadow_only" if policy_name != "watch" else "monitor_only"
        fallback_reason = (
            "global policy is viable but still lacks enough evidence for live promotion"
            if policy_name != "watch"
            else "global watch remains monitor-only until more evidence accumulates"
        )
        out[policy_name] = _evaluate_policy_stage(
            policy,
            GLOBAL_RULES[policy_name],
            fallback_stage=fallback_stage,
            fallback_reason=fallback_reason,
        )
    return out


def _strategy_recommendations(metrics_payload):
    strategy_payload = metrics_payload.get("strategy_specific_policies") or {}
    out = {}
    for strategy_name, strategy_root in sorted(strategy_payload.items()):
        if not isinstance(strategy_root, dict):
            continue
        out[strategy_name] = {}
        for policy_name in ("premium", "standard", "watch"):
            policy = strategy_root.get(f"recommended_{policy_name}_policy") or {}
            fallback_stage = "shadow_only" if policy_name != "watch" else "monitor_only"
            fallback_reason = (
                f"{strategy_name} {policy_name} is viable but still better kept in shadow/canary mode"
                if policy_name != "watch"
                else f"{strategy_name} watch should stay as monitor-only for now"
            )
            out[strategy_name][policy_name] = _evaluate_policy_stage(
                policy,
                _strategy_rules(strategy_name, policy_name),
                fallback_stage=fallback_stage,
                fallback_reason=fallback_reason,
            )
    return out


def _promotion_summary(global_rules, strategy_rules, shadow_summary):
    summary = {
        "default_model_action": "hold",
        "default_model_reason": "no promotable global policy found",
        "global_live_ready_policies": [],
        "strategy_live_ready_policies": [],
        "strategy_canary_policies": [],
        "monitor_only_policies": [],
        "shadow_evidence_status": "not_provided",
    }

    shadow_filled = None
    if isinstance(shadow_summary, dict):
        inner = shadow_summary.get("summary") if isinstance(shadow_summary.get("summary"), dict) else shadow_summary
        shadow_filled = _safe_int(inner.get("filled_row_count"), 0)
        summary["shadow_evidence_status"] = "available" if shadow_filled > 0 else "empty"
        summary["shadow_filled_row_count"] = shadow_filled

    for policy_name, result in global_rules.items():
        stage = result.get("promotion_stage")
        if stage in {"promote_live", "promote_monitor"}:
            summary["global_live_ready_policies"].append(policy_name)
        elif stage in {"monitor_only"}:
            summary["monitor_only_policies"].append(f"global:{policy_name}")

    for strategy_name, strategy_root in strategy_rules.items():
        for policy_name, result in strategy_root.items():
            stage = result.get("promotion_stage")
            key = f"{strategy_name}:{policy_name}"
            if stage in {"promote_strategy_live", "promote_strategy_monitor"}:
                summary["strategy_live_ready_policies"].append(key)
            elif stage in {"strategy_canary"}:
                summary["strategy_canary_policies"].append(key)
            elif stage in {"monitor_only"}:
                summary["monitor_only_policies"].append(key)

    if "premium" in summary["global_live_ready_policies"] and "standard" in summary["global_live_ready_policies"]:
        summary["default_model_action"] = "promote_candidate_as_default"
        if shadow_filled and shadow_filled > 0:
            summary["default_model_reason"] = "global premium and standard are live-ready with shadow evidence present"
        else:
            summary["default_model_reason"] = "global premium and standard are live-ready on holdout; use guarded rollout because shadow evidence is still empty"
    elif summary["global_live_ready_policies"]:
        summary["default_model_action"] = "promote_candidate_guarded"
        summary["default_model_reason"] = "at least one global policy is live-ready, but the full tier set is not complete"
    elif summary["strategy_live_ready_policies"] or summary["strategy_canary_policies"]:
        summary["default_model_action"] = "keep_global_old_enable_strategy_canary"
        summary["default_model_reason"] = "strategy-specific policies have merit but global policy is not yet ready"
    return summary


def build_promotion_plan(metrics_payload, shadow_summary=None):
    global_rules = _global_recommendations(metrics_payload)
    strategy_rules = _strategy_recommendations(metrics_payload)
    summary = _promotion_summary(global_rules, strategy_rules, shadow_summary)
    return {
        "artifact_type": "entry_ai_promotion_plan",
        "model_version": metrics_payload.get("model_version"),
        "generated_from_metrics": True,
        "global_policy_recommendations": global_rules,
        "strategy_policy_recommendations": strategy_rules,
        "promotion_summary": summary,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Build promotion recommendations from Entry AI metrics artifacts")
    parser.add_argument("--metrics-path", required=True, help="Path to phase4_entry_quality_v4_metrics.json")
    parser.add_argument("--shadow-summary-path", default="", help="Optional live/shadow summary json path")
    parser.add_argument("--output-path", default="", help="Optional output path for promotion plan json")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path).expanduser().resolve()
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    shadow_summary = None
    if str(args.shadow_summary_path or "").strip():
        shadow_path = Path(args.shadow_summary_path).expanduser().resolve()
        if shadow_path.exists():
            shadow_summary = json.loads(shadow_path.read_text(encoding="utf-8"))

    output_path = str(args.output_path or "").strip()
    if not output_path:
        output_path = str(metrics_path.with_name(metrics_path.stem.replace("_metrics", "_promotion_plan") + ".json"))
    output_path = Path(output_path).expanduser().resolve()

    plan = build_promotion_plan(metrics_payload, shadow_summary=shadow_summary)
    output_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = plan.get("promotion_summary") or {}
    print(
        "[promotion] action={action} global={global_ready} canary={canary} output={output}".format(
            action=summary.get("default_model_action"),
            global_ready=",".join(summary.get("global_live_ready_policies") or []) or "none",
            canary=",".join(summary.get("strategy_canary_policies") or []) or "none",
            output=str(output_path),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
