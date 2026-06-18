import argparse
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for candidate_path in (PROJECT_ROOT, TOOLS_DIR):
    text_path = str(candidate_path)
    if text_path not in sys.path:
        sys.path.insert(0, text_path)

import trad  # noqa: E402
from application.services.service_support import clean_json_value  # noqa: E402
from train_phase3_entry_quality_model import apply_entry_quality_labels  # noqa: E402


def _parse_csv_list(value):
    if value is None:
        return None
    items = [str(item).strip() for item in str(value).split(",")]
    items = [item for item in items if item]
    return items or None


def _safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _mean(values):
    usable = [float(value) for value in values if isinstance(value, (int, float))]
    if not usable:
        return None
    return float(sum(usable) / float(len(usable)))


def _write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(clean_json_value(payload), handle, ensure_ascii=False, indent=2)


def _load_live_entry_ai_bundle():
    model_path = trad._resolve_live_entry_ai_model_path()
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError("Live Entry AI model bundle not found")
    import joblib

    return joblib.load(model_path), os.path.abspath(model_path)


def _decision_payload(row, metadata, *, model_type, is_v4_native, prefer_strategy_specific, entry_threshold_override, avoid_threshold_override):
    prob_map = {
        "entry": _safe_float(row.get("entry_ai_prob_entry"), None),
        "watch": _safe_float(row.get("entry_ai_prob_watch"), None),
        "avoid": _safe_float(row.get("entry_ai_prob_avoid"), None),
    }
    if not any(isinstance(prob_map.get(label), float) for label in ("entry", "watch", "avoid")):
        return None
    return trad._entry_ai_decision_from_prob_map(
        prob_map,
        metadata,
        strategy=str(row.get("strategy") or "").strip().upper(),
        model_type=model_type,
        is_v4_native=is_v4_native,
        prefer_strategy_specific=bool(prefer_strategy_specific),
        entry_threshold_override=entry_threshold_override,
        avoid_threshold_override=avoid_threshold_override,
    )


def _policy_summary(rows, prefix, *, window_days):
    entry_rows = [row for row in rows if str(row.get(f"{prefix}_entry_ai_bucket") or "").strip().lower() == "entry"]
    premium_rows = [row for row in rows if str(row.get(f"{prefix}_entry_ai_policy_tier") or "").strip().lower() == "premium"]
    standard_rows = [row for row in rows if str(row.get(f"{prefix}_entry_ai_policy_tier") or "").strip().lower() == "standard"]
    watch_rows = [row for row in rows if str(row.get(f"{prefix}_entry_ai_policy_tier") or "").strip().lower() == "watch"]
    avoid_rows = [row for row in rows if str(row.get(f"{prefix}_entry_ai_policy_tier") or "").strip().lower() == "avoid"]
    strategy_policy_rows = [row for row in rows if bool(row.get(f"{prefix}_strategy_policy_applied"))]
    wins = [row for row in entry_rows if row.get("label_win") is True]
    losses = [row for row in entry_rows if row.get("label_win") is False]
    returns = [_safe_float(row.get("label_return_pct"), None) for row in entry_rows]
    actual_entry_hits = [1.0 for row in entry_rows if str(row.get("entry_quality_label") or "").strip().lower() == "entry"]
    actual_watch_hits = [1.0 for row in entry_rows if str(row.get("entry_quality_label") or "").strip().lower() == "watch"]
    actual_avoid_hits = [1.0 for row in entry_rows if str(row.get("entry_quality_label") or "").strip().lower() == "avoid"]
    return {
        "row_count": int(len(rows)),
        "entry_rows": int(len(entry_rows)),
        "premium_rows": int(len(premium_rows)),
        "standard_rows": int(len(standard_rows)),
        "watch_rows": int(len(watch_rows)),
        "avoid_rows": int(len(avoid_rows)),
        "strategy_policy_rows": int(len(strategy_policy_rows)),
        "strategy_policy_share_pct": (float(len(strategy_policy_rows)) / float(len(rows)) * 100.0) if rows else 0.0,
        "alerts_per_day": (float(len(entry_rows)) / float(window_days)) if window_days and window_days > 0 else None,
        "win_rate_pct": (float(len(wins)) / float(len(entry_rows)) * 100.0) if entry_rows else None,
        "loss_rate_pct": (float(len(losses)) / float(len(entry_rows)) * 100.0) if entry_rows else None,
        "avg_return_pct": _mean(returns),
        "actual_entry_rate_pct": (float(sum(actual_entry_hits)) / float(len(entry_rows)) * 100.0) if entry_rows else None,
        "actual_watch_rate_pct": (float(sum(actual_watch_hits)) / float(len(entry_rows)) * 100.0) if entry_rows else None,
        "actual_avoid_rate_pct": (float(sum(actual_avoid_hits)) / float(len(entry_rows)) * 100.0) if entry_rows else None,
    }


def _summary_delta(global_summary, strategy_summary):
    keys = (
        "entry_rows",
        "premium_rows",
        "standard_rows",
        "watch_rows",
        "avoid_rows",
        "strategy_policy_rows",
        "alerts_per_day",
        "win_rate_pct",
        "avg_return_pct",
        "actual_entry_rate_pct",
        "actual_avoid_rate_pct",
    )
    out = {}
    for key in keys:
        left = global_summary.get(key)
        right = strategy_summary.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            out[f"{key}_delta"] = float(right) - float(left)
    return out


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate live native shadow outcomes for global vs strategy-specific Entry AI policies")
    parser.add_argument("--days", type=float, default=90.0, help="History window in days")
    parser.add_argument("--strategies", default="", help="Comma-separated strategy filter")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol filter")
    parser.add_argument("--include-open", action="store_true", help="Include open rows in source export; summary still uses filled rows only")
    parser.add_argument(
        "--output-path",
        default=trad._live_feedback_shadow_eval_file_path(),
        help="CSV output path",
    )
    parser.add_argument(
        "--summary-path",
        default=trad._live_feedback_shadow_summary_file_path(),
        help="JSON summary output path",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    strategies = _parse_csv_list(args.strategies)
    symbols = _parse_csv_list(args.symbols)

    print("[phase7] building live feedback shadow dataset...", flush=True)
    payload = trad._build_live_feedback_training_dataset(
        days=args.days,
        strategies=strategies,
        symbols=symbols,
        include_open=bool(args.include_open),
    )
    source_df = pd.DataFrame(payload.get("rows") or [])
    bundle, model_path = _load_live_entry_ai_bundle()
    metadata = bundle.get("metadata") if isinstance(bundle.get("metadata"), dict) else {}
    model_type = str(bundle.get("model_type") or "phase3_entry_quality_classifier")
    model_version = str(bundle.get("model_version") or metadata.get("model_version") or "").strip() or None
    is_v4_native = trad._is_v4_entry_ai_bundle(bundle, metadata)
    entry_threshold_override = _safe_float(getattr(trad.config, "TELEGRAM_ALERT_ENTRY_AI_ENTRY_THRESHOLD", None), None)
    avoid_threshold_override = _safe_float(getattr(trad.config, "TELEGRAM_ALERT_ENTRY_AI_AVOID_THRESHOLD", None), None)

    if source_df.empty:
        fieldnames = trad._live_feedback_training_fieldnames() + [
            "entry_quality_label",
            "global_entry_ai_bucket",
            "global_entry_ai_policy_tier",
            "global_strategy_policy_applied",
            "strategy_entry_ai_bucket",
            "strategy_entry_ai_policy_tier",
            "strategy_strategy_policy_applied",
        ]
        rows = []
        summary = {
            "generated_at": payload.get("generated_at"),
            "window_days": payload.get("window_days"),
            "row_count": 0,
            "filled_row_count": 0,
            "model_path": model_path,
            "model_version": model_version,
            "strategy_specific_policy_count": len(metadata.get("strategy_specific_policies") or {}),
            "global_policy": {},
            "strategy_specific_policy": {},
            "comparison": {},
            "by_strategy": {},
        }
    else:
        filled_df = source_df[source_df["label_filled"] == True].copy()
        if not filled_df.empty:
            labeled_df = apply_entry_quality_labels(filled_df)
        else:
            labeled_df = filled_df
            labeled_df["entry_quality_label"] = pd.Series(dtype="object")

        rows = []
        for _, row in labeled_df.iterrows():
            payload_row = {key: row.get(key) for key in labeled_df.columns}
            global_decision = _decision_payload(
                row,
                metadata,
                model_type=model_type,
                is_v4_native=is_v4_native,
                prefer_strategy_specific=False,
                entry_threshold_override=entry_threshold_override,
                avoid_threshold_override=avoid_threshold_override,
            )
            strategy_decision = _decision_payload(
                row,
                metadata,
                model_type=model_type,
                is_v4_native=is_v4_native,
                prefer_strategy_specific=True,
                entry_threshold_override=entry_threshold_override,
                avoid_threshold_override=avoid_threshold_override,
            )
            if not isinstance(global_decision, dict) or not isinstance(strategy_decision, dict):
                continue
            payload_row["entry_quality_label"] = str(row.get("entry_quality_label") or "").strip().lower() or None
            for key, value in global_decision.items():
                payload_row[f"global_{key}"] = value
            for key, value in strategy_decision.items():
                payload_row[f"strategy_{key}"] = value
            payload_row["shadow_bucket_changed"] = (
                str(payload_row.get("global_entry_ai_bucket") or "") != str(payload_row.get("strategy_entry_ai_bucket") or "")
            )
            payload_row["shadow_policy_tier_changed"] = (
                str(payload_row.get("global_entry_ai_policy_tier") or "") != str(payload_row.get("strategy_entry_ai_policy_tier") or "")
            )
            rows.append(payload_row)

        fieldnames = list(rows[0].keys()) if rows else (
            list(labeled_df.columns)
            + [
                "entry_quality_label",
                "global_entry_ai_bucket",
                "global_entry_ai_policy_tier",
                "global_strategy_policy_applied",
                "strategy_entry_ai_bucket",
                "strategy_entry_ai_policy_tier",
                "strategy_strategy_policy_applied",
                "shadow_bucket_changed",
                "shadow_policy_tier_changed",
            ]
        )
        global_summary = _policy_summary(rows, "global", window_days=float(payload.get("window_days") or args.days or 0.0))
        strategy_summary = _policy_summary(rows, "strategy", window_days=float(payload.get("window_days") or args.days or 0.0))
        by_strategy = {}
        by_strategy_df = pd.DataFrame(rows)
        if not by_strategy_df.empty and "strategy" in by_strategy_df.columns:
            for strategy_name, strategy_df in by_strategy_df.groupby(by_strategy_df["strategy"].fillna("UNKNOWN").astype(str)):
                strategy_rows = strategy_df.to_dict(orient="records")
                by_strategy[str(strategy_name).upper()] = {
                    "row_count": int(len(strategy_rows)),
                    "global_policy": _policy_summary(strategy_rows, "global", window_days=float(payload.get("window_days") or args.days or 0.0)),
                    "strategy_specific_policy": _policy_summary(strategy_rows, "strategy", window_days=float(payload.get("window_days") or args.days or 0.0)),
                }
                by_strategy[str(strategy_name).upper()]["comparison"] = _summary_delta(
                    by_strategy[str(strategy_name).upper()]["global_policy"],
                    by_strategy[str(strategy_name).upper()]["strategy_specific_policy"],
                )
        summary = {
            "generated_at": payload.get("generated_at"),
            "window_days": payload.get("window_days"),
            "row_count": int(len(source_df)),
            "filled_row_count": int(len(rows)),
            "model_path": model_path,
            "model_version": model_version,
            "model_type": model_type,
            "is_v4_native": bool(is_v4_native),
            "strategy_specific_policy_count": len(metadata.get("strategy_specific_policies") or {}),
            "strategy_specific_policy_keys": sorted([str(key) for key in (metadata.get("strategy_specific_policies") or {}).keys()]),
            "global_policy": global_summary,
            "strategy_specific_policy": strategy_summary,
            "comparison": _summary_delta(global_summary, strategy_summary),
            "by_strategy": by_strategy,
        }

    _write_csv(args.output_path, fieldnames, rows)
    _write_json(
        args.summary_path,
        {
            "artifact_type": "live_feedback_shadow_summary",
            "request": {
                "days": float(args.days),
                "strategies": strategies,
                "symbols": symbols,
                "include_open": bool(args.include_open),
            },
            "summary": summary,
            "files": {
                "csv": os.path.abspath(args.output_path),
                "summary_json": os.path.abspath(args.summary_path),
            },
        },
    )
    comparison = summary.get("comparison") or {}
    print(
        "[phase7] filled={filled} global_win={global_win} strategy_win={strategy_win} delta={delta} csv={csv_path}".format(
            filled=summary.get("filled_row_count"),
            global_win=(
                f"{float((summary.get('global_policy') or {}).get('win_rate_pct')):.2f}%"
                if isinstance((summary.get("global_policy") or {}).get("win_rate_pct"), (int, float))
                else "n/a"
            ),
            strategy_win=(
                f"{float((summary.get('strategy_specific_policy') or {}).get('win_rate_pct')):.2f}%"
                if isinstance((summary.get("strategy_specific_policy") or {}).get("win_rate_pct"), (int, float))
                else "n/a"
            ),
            delta=(
                f"{float(comparison.get('win_rate_pct_delta')):+.2f}pp"
                if isinstance(comparison.get("win_rate_pct_delta"), (int, float))
                else "n/a"
            ),
            csv_path=os.path.abspath(args.output_path),
        ),
        flush=True,
    )
    print(f"[phase7] summary={os.path.abspath(args.summary_path)}", flush=True)


if __name__ == "__main__":
    main()
