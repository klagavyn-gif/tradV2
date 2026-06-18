import argparse
import csv
import json
import math
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


CALIBRATION_TARGETS = [
    ("entry_ai_prob_entry", "entry_target", "prob_entry_vs_entry_target"),
    ("entry_ai_prob_watch", "watch_target", "prob_watch_vs_watch_target"),
    ("entry_ai_prob_avoid", "avoid_target", "prob_avoid_vs_avoid_target"),
    ("ai_prob_win", "win_target", "ai_prob_win_vs_win_target"),
]


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
        value = float(value)
        if not math.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _clip_prob(value, eps=1e-6):
    value = _safe_float(value, None)
    if value is None:
        return None
    return min(max(float(value), float(eps)), 1.0 - float(eps))


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


def _calibration_bins(df, prob_col, target_col, *, bin_count):
    if df.empty or prob_col not in df.columns or target_col not in df.columns:
        return []
    out = []
    total = len(df)
    for idx in range(int(bin_count)):
        lower = float(idx) / float(bin_count)
        upper = float(idx + 1) / float(bin_count)
        if idx == int(bin_count) - 1:
            bucket = df[(df[prob_col] >= lower) & (df[prob_col] <= upper)]
        else:
            bucket = df[(df[prob_col] >= lower) & (df[prob_col] < upper)]
        count = int(len(bucket))
        if count <= 0:
            continue
        avg_pred = float(bucket[prob_col].mean())
        actual_rate = float(bucket[target_col].mean())
        gap = abs(avg_pred - actual_rate)
        out.append(
            {
                "bin_index": idx,
                "prob_lower": lower,
                "prob_upper": upper,
                "row_count": count,
                "share_pct": (float(count) / float(total)) * 100.0 if total > 0 else 0.0,
                "avg_predicted_prob": avg_pred,
                "actual_positive_rate": actual_rate,
                "gap_abs": gap,
            }
        )
    return out


def _calibration_metric(df, prob_col, target_col, *, bin_count):
    if df.empty or prob_col not in df.columns or target_col not in df.columns:
        return {"row_count": 0}
    working = df[[prob_col, target_col]].copy()
    working[prob_col] = working[prob_col].map(_clip_prob)
    working[target_col] = pd.to_numeric(working[target_col], errors="coerce")
    working = working.dropna(subset=[prob_col, target_col])
    if working.empty:
        return {"row_count": 0}

    probs = working[prob_col].astype(float)
    targets = working[target_col].astype(float)
    row_count = int(len(working))
    positives = int((targets > 0.5).sum())
    brier_score = float(((probs - targets) ** 2).mean())
    log_loss = float((-(targets * probs.map(math.log)) - ((1.0 - targets) * (1.0 - probs).map(math.log))).mean())
    bins = _calibration_bins(working, prob_col, target_col, bin_count=bin_count)
    ece = 0.0
    mce = 0.0
    for bucket in bins:
        gap = float(bucket["gap_abs"])
        weight = float(bucket["row_count"]) / float(row_count)
        ece += gap * weight
        mce = max(mce, gap)
    return {
        "row_count": row_count,
        "positives": positives,
        "positive_rate": float(targets.mean()) if row_count > 0 else None,
        "avg_predicted_prob": float(probs.mean()) if row_count > 0 else None,
        "brier_score": brier_score,
        "log_loss": log_loss,
        "ece": float(ece),
        "mce": float(mce),
        "bins": bins,
    }


def _build_calibration_rows(df):
    fieldnames = list(df.columns) + [
        "entry_quality_label",
        "entry_target",
        "watch_target",
        "avoid_target",
        "win_target",
    ]
    rows = []
    for _, row in df.iterrows():
        payload = {key: row.get(key) for key in df.columns}
        label = str(row.get("entry_quality_label") or "").strip().lower()
        payload["entry_quality_label"] = label or None
        payload["entry_target"] = 1 if label == "entry" else 0
        payload["watch_target"] = 1 if label == "watch" else 0
        payload["avoid_target"] = 1 if label == "avoid" else 0
        payload["win_target"] = 1 if bool(row.get("label_win")) else 0
        rows.append(payload)
    return fieldnames, rows


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate live probability calibration for V5 Phase 3")
    parser.add_argument("--days", type=float, default=90.0, help="History window in days")
    parser.add_argument("--strategies", default="", help="Comma-separated strategy filter")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol filter")
    parser.add_argument("--include-open", action="store_true", help="Include open rows before filtering to filled rows")
    parser.add_argument("--bin-count", type=int, default=10, help="Number of calibration bins")
    parser.add_argument(
        "--output-path",
        default=trad._live_feedback_calibration_dataset_file_path(),
        help="CSV output path",
    )
    parser.add_argument(
        "--summary-path",
        default=trad._live_feedback_calibration_summary_file_path(),
        help="JSON summary output path",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    strategies = _parse_csv_list(args.strategies)
    symbols = _parse_csv_list(args.symbols)
    bin_count = max(4, int(args.bin_count))

    print("[phase3] building calibration dataset...", flush=True)
    payload = trad._build_live_feedback_training_dataset(
        days=args.days,
        strategies=strategies,
        symbols=symbols,
        include_open=bool(args.include_open),
    )
    df = pd.DataFrame(payload.get("rows") or [])
    if df.empty:
        fieldnames = trad._live_feedback_training_fieldnames() + [
            "entry_quality_label",
            "entry_target",
            "watch_target",
            "avoid_target",
            "win_target",
        ]
        rows = []
        summary = {
            "generated_at": payload.get("generated_at"),
            "window_days": payload.get("window_days"),
            "row_count": 0,
            "filled_row_count": 0,
            "entry_quality_label_counts": {},
            "metrics": {},
            "by_strategy": {},
        }
    else:
        filled_df = df[df["label_filled"] == True].copy()
        if not filled_df.empty:
            labeled_df = apply_entry_quality_labels(filled_df)
        else:
            labeled_df = filled_df
            labeled_df["entry_quality_label"] = pd.Series(dtype="object")
        fieldnames, rows = _build_calibration_rows(labeled_df)
        calibration_df = pd.DataFrame(rows)
        label_counts = {}
        if not calibration_df.empty and "entry_quality_label" in calibration_df.columns:
            label_counts = {
                str(key): int(value)
                for key, value in calibration_df["entry_quality_label"].value_counts(dropna=False).to_dict().items()
            }
        metrics = {
            metric_name: _calibration_metric(calibration_df, prob_col, target_col, bin_count=bin_count)
            for prob_col, target_col, metric_name in CALIBRATION_TARGETS
        }
        by_strategy = {}
        if not calibration_df.empty and "strategy" in calibration_df.columns:
            for strategy, strategy_df in calibration_df.groupby(calibration_df["strategy"].fillna("UNKNOWN").astype(str)):
                if len(strategy_df) < 8:
                    continue
                by_strategy[str(strategy).upper()] = {
                    "row_count": int(len(strategy_df)),
                    "entry_quality_label_counts": {
                        str(key): int(value)
                        for key, value in strategy_df["entry_quality_label"].value_counts(dropna=False).to_dict().items()
                    },
                    "metrics": {
                        metric_name: _calibration_metric(strategy_df, prob_col, target_col, bin_count=bin_count)
                        for prob_col, target_col, metric_name in CALIBRATION_TARGETS
                    },
                }
        summary = {
            "generated_at": payload.get("generated_at"),
            "window_days": payload.get("window_days"),
            "row_count": int(len(df)),
            "filled_row_count": int(len(labeled_df)),
            "entry_quality_label_counts": label_counts,
            "metrics": metrics,
            "by_strategy": by_strategy,
        }

    _write_csv(args.output_path, fieldnames, rows)
    _write_json(
        args.summary_path,
        {
            "artifact_type": "live_feedback_calibration_summary",
            "request": {
                "days": float(args.days),
                "strategies": strategies,
                "symbols": symbols,
                "include_open": bool(args.include_open),
                "bin_count": int(bin_count),
            },
            "summary": summary,
            "files": {
                "csv": os.path.abspath(args.output_path),
                "summary_json": os.path.abspath(args.summary_path),
            },
        },
    )

    entry_metric = ((summary.get("metrics") or {}).get("prob_entry_vs_entry_target") or {})
    avoid_metric = ((summary.get("metrics") or {}).get("prob_avoid_vs_avoid_target") or {})
    print(
        "[phase3] filled={filled} entry_ece={entry_ece} avoid_ece={avoid_ece} csv={csv_path}".format(
            filled=summary.get("filled_row_count"),
            entry_ece=(f"{float(entry_metric['ece']):.4f}" if isinstance(entry_metric.get("ece"), (int, float)) else "n/a"),
            avoid_ece=(f"{float(avoid_metric['ece']):.4f}" if isinstance(avoid_metric.get("ece"), (int, float)) else "n/a"),
            csv_path=os.path.abspath(args.output_path),
        ),
        flush=True,
    )
    print(f"[phase3] summary={os.path.abspath(args.summary_path)}", flush=True)


if __name__ == "__main__":
    main()
