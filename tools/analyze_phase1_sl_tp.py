import argparse
import json
from pathlib import Path

import pandas as pd

from run_phase2_walkforward import filter_dataset, load_dataset, parse_csv_list, resolve_input_path


BUCKET_SPECS = {
    "entry_gap_pct": {
        "edges": [0.0, 0.15, 0.35, 0.80, 1.50, float("inf")],
        "labels": ["<=0.15%", "0.15-0.35%", "0.35-0.80%", "0.80-1.50%", ">1.50%"],
    },
    "stop_risk_pct": {
        "edges": [0.0, 0.35, 0.75, 1.25, 1.80, 2.50, float("inf")],
        "labels": ["<=0.35%", "0.35-0.75%", "0.75-1.25%", "1.25-1.80%", "1.80-2.50%", ">2.50%"],
    },
    "target_reward_pct": {
        "edges": [0.0, 0.75, 1.25, 2.00, 3.00, float("inf")],
        "labels": ["<=0.75%", "0.75-1.25%", "1.25-2.00%", "2.00-3.00%", ">3.00%"],
    },
    "rr_ratio": {
        "edges": [0.0, 1.0, 1.3, 1.8, 2.5, 4.0, float("inf")],
        "labels": ["<=1.0", "1.0-1.3", "1.3-1.8", "1.8-2.5", "2.5-4.0", ">4.0"],
    },
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze SL/TP quality for BUY vs SELL from Phase 1 candidates dataset"
    )
    parser.add_argument(
        "--input-path",
        default="",
        help="Path to phase1_candidates.csv/jsonl (default: .data/research/phase1/phase1_candidates.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for SL/TP analysis outputs (default: alongside input under sl_tp_analysis)",
    )
    parser.add_argument(
        "--groups",
        default="primary,trend_radar,daily",
        help="Candidate groups to include",
    )
    parser.add_argument(
        "--strategies",
        default="",
        help="Optional comma-separated strategies to include",
    )
    parser.add_argument(
        "--intents",
        default="entry,watch",
        help="Alert intents to include, e.g. entry,watch or entry,watch,exit",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=10,
        help="Minimum rows per BUY/SELL bucket before using it in recommendations",
    )
    return parser


def resolve_output_dir(root, input_path, raw_output_dir):
    raw = str(raw_output_dir or "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (root / path)
    return input_path.resolve().parents[0] / "sl_tp_analysis"


def normalize_bool_series(series):
    return series.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() in {"1", "true", "yes", "y"}
    )


def build_feature_frame(df):
    out = df.copy()
    for col in (
        "entry_price",
        "stop_loss",
        "take_profit",
        "price_at_checkpoint",
        "label_return_pct",
        "label_mfe_pct",
    ):
        if col not in out.columns:
            out[col] = None
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "label_win" not in out.columns:
        out["label_win"] = False
    out["label_win"] = normalize_bool_series(out["label_win"]).fillna(False)

    if "label_filled" not in out.columns:
        out["label_filled"] = False
    out["label_filled"] = normalize_bool_series(out["label_filled"]).fillna(False)

    if "alert_intent" not in out.columns:
        out["alert_intent"] = ""
    out["alert_intent"] = out["alert_intent"].fillna("").astype(str).str.lower()

    if "signal" not in out.columns:
        out["signal"] = ""
    out["signal"] = out["signal"].fillna("").astype(str).str.upper()

    out = out[out["label_filled"] == True].copy()
    out = out[out["signal"].isin({"BUY", "SELL"})].copy()
    out = out.dropna(subset=["entry_price", "stop_loss", "price_at_checkpoint", "label_return_pct"])

    entry_abs = out["entry_price"].abs().replace(0, pd.NA)
    out["entry_gap_pct"] = ((out["price_at_checkpoint"] - out["entry_price"]).abs() / entry_abs) * 100.0
    out["stop_risk_pct"] = ((out["entry_price"] - out["stop_loss"]).abs() / entry_abs) * 100.0
    out["target_reward_pct_planned"] = ((out["take_profit"] - out["entry_price"]).abs() / entry_abs) * 100.0
    out["target_reward_pct_realized"] = pd.to_numeric(out["label_mfe_pct"], errors="coerce").abs()
    out["target_reward_pct"] = out["target_reward_pct_planned"].where(
        out["target_reward_pct_planned"].notna(),
        out["target_reward_pct_realized"],
    )
    out["target_reward_source"] = out["target_reward_pct_planned"].map(
        lambda value: "planned_take_profit" if pd.notna(value) else "realized_mfe_fallback"
    )
    out["rr_ratio"] = out["target_reward_pct"] / out["stop_risk_pct"].replace(0, pd.NA)
    out = out.dropna(subset=["entry_gap_pct", "stop_risk_pct", "target_reward_pct", "rr_ratio"])
    return out


def summarize_by(df, group_cols):
    grouped = (
        df.groupby(group_cols, dropna=False, observed=False)
        .agg(
            rows=("signal", "size"),
            win_rate_pct=("label_win", lambda s: float(s.mean()) * 100.0 if len(s) else None),
            avg_return_pct=("label_return_pct", "mean"),
            median_return_pct=("label_return_pct", "median"),
            median_entry_gap_pct=("entry_gap_pct", "median"),
            median_stop_risk_pct=("stop_risk_pct", "median"),
            median_target_reward_pct=("target_reward_pct", "median"),
            median_rr_ratio=("rr_ratio", "median"),
        )
        .reset_index()
    )
    return grouped.sort_values(group_cols).reset_index(drop=True)


def bucket_summary(df, metric_name):
    spec = BUCKET_SPECS[metric_name]
    bucket_col = f"{metric_name}_bucket"
    out = df.copy()
    out[bucket_col] = pd.cut(
        out[metric_name],
        bins=spec["edges"],
        labels=spec["labels"],
        include_lowest=True,
        right=True,
    )
    summary = summarize_by(out.dropna(subset=[bucket_col]), ["signal", bucket_col])
    return summary.rename(columns={bucket_col: "bucket"})


def choose_recommendation(summary_df, min_rows):
    recommendations = {}
    if summary_df.empty:
        return recommendations
    for signal in ("BUY", "SELL"):
        rows = summary_df[(summary_df["signal"] == signal) & (summary_df["rows"] >= int(min_rows))].copy()
        if rows.empty:
            recommendations[signal] = None
            continue
        rows = rows.sort_values(
            by=["avg_return_pct", "win_rate_pct", "rows"],
            ascending=[False, False, False],
        )
        best = rows.iloc[0].to_dict()
        best["rows"] = int(best["rows"])
        recommendations[signal] = best
    return recommendations


def json_ready_records(df):
    rows = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if pd.isna(value):
                clean[key] = None
            elif isinstance(value, (int, float, str, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        rows.append(clean)
    return rows


def main():
    parser = build_parser()
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = resolve_input_path(root, args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Phase 1 dataset not found: {input_path}")

    output_dir = resolve_output_dir(root, input_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    groups = parse_csv_list(args.groups)
    strategies = [value.strip().upper() for value in parse_csv_list(args.strategies)]
    intents = [value.strip().lower() for value in parse_csv_list(args.intents)]
    df = filter_dataset(df, groups=groups, strategies=strategies)
    if intents and "alert_intent" in df.columns:
        df = df[df["alert_intent"].fillna("").astype(str).str.lower().isin(intents)].copy()
    feature_df = build_feature_frame(df)
    if feature_df.empty:
        raise ValueError("No filled BUY/SELL rows with valid entry/stop/take-profit values were found")

    signal_summary = summarize_by(feature_df, ["signal"])
    signal_intent_summary = summarize_by(feature_df, ["signal", "alert_intent"])

    bucket_outputs = {}
    recommendations = {}
    for metric_name in BUCKET_SPECS:
        summary = bucket_summary(feature_df, metric_name)
        csv_path = output_dir / f"{metric_name}_buckets.csv"
        summary.to_csv(csv_path, index=False)
        bucket_outputs[metric_name] = {
            "csv_path": str(csv_path),
            "top_ranges": choose_recommendation(summary, args.min_rows),
        }
        recommendations[metric_name] = bucket_outputs[metric_name]["top_ranges"]

    signal_summary_path = output_dir / "signal_summary.csv"
    signal_intent_summary_path = output_dir / "signal_intent_summary.csv"
    recommendations_path = output_dir / "sl_tp_recommendations.json"

    signal_summary.to_csv(signal_summary_path, index=False)
    signal_intent_summary.to_csv(signal_intent_summary_path, index=False)

    payload = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "row_count": int(len(feature_df)),
        "groups": groups,
        "strategies": strategies,
        "intents": intents,
        "target_reward_sources": (
            feature_df["target_reward_source"].value_counts(dropna=False).to_dict()
            if "target_reward_source" in feature_df.columns
            else {}
        ),
        "signal_summary_path": str(signal_summary_path),
        "signal_intent_summary_path": str(signal_intent_summary_path),
        "bucket_outputs": bucket_outputs,
        "signal_summary": json_ready_records(signal_summary),
        "signal_intent_summary": json_ready_records(signal_intent_summary),
        "recommendations": recommendations,
    }
    recommendations_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "row_count": int(len(feature_df)),
                "signal_summary_path": str(signal_summary_path),
                "signal_intent_summary_path": str(signal_intent_summary_path),
                "recommendations_path": str(recommendations_path),
                "recommendations": recommendations,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
