import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_phase2_walkforward import filter_dataset, load_dataset, parse_csv_list, resolve_input_path


LABELS = ("entry", "watch", "avoid")
DISPLAY_LABELS = {
    "entry": "เข้าได้",
    "watch": "รอ",
    "avoid": "ห้ามเข้า",
}

DEFAULT_CATEGORICAL_FEATURES = [
    "strategy",
    "symbol",
    "signal",
    "candidate_group",
    "alert_tier",
    "alert_intent",
    "ai_dispatch_bucket",
    "ai_runtime_status",
    "short_trade_bucket",
    "forecast_direction",
]

DEFAULT_NUMERIC_FEATURES = [
    "confidence",
    "score",
    "alert_tier_score",
    "tier_rank",
    "source_count",
    "backtest_win_rate_pct",
    "backtest_expectancy_rr",
    "backtest_trades",
    "ai_prob_win",
    "ai_expected_return_pct",
    "ai_rank_adjustment",
    "short_trade_score_adjustment",
    "price_at_checkpoint",
    "entry_gap_pct",
    "stop_risk_pct",
    "target_reward_pct",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train Model C entry quality classifier (entry/watch/avoid) from Phase 1 dataset"
    )
    parser.add_argument(
        "--input-path",
        default="",
        help="Path to phase1_candidates.csv/jsonl (default: .data/research/phase1/phase1_candidates.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for model artifact and metrics (default: alongside input under phase3_entry_quality)",
    )
    parser.add_argument(
        "--groups",
        default="primary,daily",
        help="Candidate groups to include in training",
    )
    parser.add_argument(
        "--strategies",
        default="",
        help="Optional comma-separated strategies to include",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Holdout window in days from the tail of the dataset",
    )
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=45,
        help="Minimum training history required before the holdout period",
    )
    parser.add_argument(
        "--min-class-rows",
        type=int,
        default=20,
        help="Minimum rows required per class after labeling",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.45,
        help="Probability threshold for forcing predicted label to entry",
    )
    parser.add_argument(
        "--avoid-threshold",
        type=float,
        default=0.55,
        help="Probability threshold for forcing predicted label to avoid",
    )
    return parser


def resolve_output_dir(root, input_path, raw_output_dir):
    raw = str(raw_output_dir or "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (root / path)
    return input_path.resolve().parents[0] / "phase3_entry_quality"


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


def _safe_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def build_features(df):
    out = df.copy()
    numeric_seed_columns = (
        "entry_price",
        "stop_loss",
        "take_profit",
        "price_at_checkpoint",
        "confidence",
        "score",
        "alert_tier_score",
        "tier_rank",
        "source_count",
        "backtest_win_rate_pct",
        "backtest_expectancy_rr",
        "backtest_trades",
        "ai_prob_win",
        "ai_expected_return_pct",
        "ai_rank_adjustment",
        "short_trade_score_adjustment",
        "label_return_pct",
        "label_mfe_r",
        "label_mae_r",
    )
    for col in numeric_seed_columns:
        if col not in out.columns:
            out[col] = None
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in DEFAULT_CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    if "short_trade_regime_aligned" in out.columns:
        out["short_trade_regime_aligned"] = out["short_trade_regime_aligned"].map(lambda value: 1.0 if _safe_bool(value, False) else 0.0)

    entry = out["entry_price"]
    stop = out["stop_loss"]
    target = out["take_profit"]
    current = out["price_at_checkpoint"]
    entry_abs = entry.abs().replace(0, pd.NA)

    out["entry_gap_pct"] = ((current - entry).abs() / entry_abs) * 100.0
    out["stop_risk_pct"] = ((entry - stop).abs() / entry_abs) * 100.0
    out["target_reward_pct"] = ((target - entry).abs() / entry_abs) * 100.0
    return out


def derive_entry_quality_label(row):
    intent = str(row.get("alert_intent") or "").strip().lower()
    short_bucket = str(row.get("short_trade_bucket") or "").strip().lower()
    label_win = _safe_bool(row.get("label_win"), False)
    label_return_pct = _safe_float(row.get("label_return_pct"), None)
    label_mfe_r = _safe_float(row.get("label_mfe_r"), None)
    label_mae_r = _safe_float(row.get("label_mae_r"), None)
    confidence = _safe_float(row.get("confidence"), None)
    ai_prob_win = _safe_float(row.get("ai_prob_win"), None)
    ai_expected_return_pct = _safe_float(row.get("ai_expected_return_pct"), None)

    if intent == "exit":
        if (
            label_win
            and isinstance(label_return_pct, float)
            and label_return_pct >= 1.0
            and isinstance(confidence, float)
            and confidence >= 88.0
            and (ai_prob_win is None or ai_prob_win >= 0.58)
        ):
            return "watch"
        return "avoid"

    if short_bucket in {"premium_entry", "standard_entry"}:
        if (
            label_win
            and (label_return_pct is None or label_return_pct >= 0.25)
            and (label_mae_r is None or label_mae_r > -1.25)
        ):
            return "entry"

    if (
        label_win
        and isinstance(label_return_pct, float)
        and label_return_pct >= 1.25
        and (label_mae_r is None or label_mae_r > -1.20)
        and (confidence is None or confidence >= 78.0)
    ):
        return "entry"

    if isinstance(label_return_pct, float) and label_return_pct <= -1.0:
        return "avoid"

    if (not label_win) and isinstance(label_mfe_r, float) and label_mfe_r < 0.75:
        return "avoid"

    if short_bucket == "watch" or intent == "watch":
        return "watch"

    if label_win and isinstance(label_mfe_r, float) and label_mfe_r >= 1.0:
        return "watch"

    if (
        isinstance(ai_expected_return_pct, float)
        and ai_expected_return_pct >= 1.0
        and isinstance(confidence, float)
        and confidence >= 82.0
    ):
        return "watch"

    return "watch"


def apply_entry_quality_labels(df):
    out = df.copy()
    out["entry_quality_label"] = out.apply(derive_entry_quality_label, axis=1)
    return out


def available_features(df):
    categorical = [col for col in DEFAULT_CATEGORICAL_FEATURES if col in df.columns]
    numeric = [col for col in DEFAULT_NUMERIC_FEATURES if col in df.columns and df[col].notna().any()]
    if "short_trade_regime_aligned" in df.columns and df["short_trade_regime_aligned"].notna().any():
        numeric.append("short_trade_regime_aligned")
    return categorical, numeric


def build_preprocessor(categorical_features, numeric_features):
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features),
        ]
    )


def chronological_split(df, test_days, min_train_days):
    if df.empty:
        return df.copy(), df.copy()
    max_ts = pd.Timestamp(df["checkpoint_at"].max())
    test_start = max_ts - pd.Timedelta(days=int(test_days))
    train_df = df[df["checkpoint_at"] < test_start].copy()
    holdout_df = df[df["checkpoint_at"] >= test_start].copy()
    if train_df.empty or holdout_df.empty:
        split_index = int(len(df) * 0.8)
        train_df = df.iloc[:split_index].copy()
        holdout_df = df.iloc[split_index:].copy()
    if not train_df.empty:
        span = pd.Timestamp(train_df["checkpoint_at"].max()) - pd.Timestamp(train_df["checkpoint_at"].min())
        if span < pd.Timedelta(days=int(min_train_days)):
            split_index = int(len(df) * 0.75)
            train_df = df.iloc[:split_index].copy()
            holdout_df = df.iloc[split_index:].copy()
    return train_df, holdout_df


def label_counts(series):
    counts = {}
    for label in LABELS:
        counts[label] = int((series == label).sum())
    return counts


def threshold_label(prob_map, entry_threshold, avoid_threshold):
    if prob_map.get("avoid", 0.0) >= float(avoid_threshold):
        return "avoid"
    if prob_map.get("entry", 0.0) >= float(entry_threshold):
        return "entry"
    return "watch"


def classification_metrics(y_true, y_pred):
    labels = list(LABELS)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else None,
        "confusion_matrix": {
            "labels": labels,
            "values": matrix.tolist(),
        },
        "classification_report": report,
    }


def holdout_prediction_rows(model, holdout_df, feature_cols, entry_threshold, avoid_threshold):
    if holdout_df.empty:
        return []
    sample = holdout_df.copy().sort_values("checkpoint_at").tail(200)
    prob = model.predict_proba(sample[feature_cols])
    classes = [str(label) for label in model.classes_]
    rows = []
    for idx, (_, row) in enumerate(sample.iterrows()):
        prob_map = {label: float(prob[idx][classes.index(label)]) if label in classes else 0.0 for label in LABELS}
        predicted_argmax = max(prob_map.items(), key=lambda item: item[1])[0]
        predicted_threshold = threshold_label(prob_map, entry_threshold, avoid_threshold)
        rows.append(
            {
                "checkpoint_at": pd.Timestamp(row["checkpoint_at"]).isoformat() if pd.notna(row["checkpoint_at"]) else None,
                "strategy": str(row.get("strategy") or ""),
                "symbol": str(row.get("symbol") or ""),
                "signal": str(row.get("signal") or ""),
                "actual_label": str(row.get("entry_quality_label") or ""),
                "actual_label_display": DISPLAY_LABELS.get(str(row.get("entry_quality_label") or ""), str(row.get("entry_quality_label") or "")),
                "predicted_label_argmax": predicted_argmax,
                "predicted_label_argmax_display": DISPLAY_LABELS.get(predicted_argmax, predicted_argmax),
                "predicted_label_threshold": predicted_threshold,
                "predicted_label_threshold_display": DISPLAY_LABELS.get(predicted_threshold, predicted_threshold),
                "prob_entry": prob_map.get("entry", 0.0),
                "prob_watch": prob_map.get("watch", 0.0),
                "prob_avoid": prob_map.get("avoid", 0.0),
                "label_win": bool(row.get("label_win")) if pd.notna(row.get("label_win")) else None,
                "label_return_pct": _safe_float(row.get("label_return_pct"), None),
            }
        )
    return rows


def realized_summary_for_predictions(rows, label_key):
    buckets = {}
    for label in LABELS:
        selected = [row for row in rows if str(row.get(label_key) or "") == label]
        returns = [float(row["label_return_pct"]) for row in selected if isinstance(row.get("label_return_pct"), (int, float, float))]
        wins = [row for row in selected if row.get("label_win") is True]
        buckets[label] = {
            "count": len(selected),
            "win_rate_pct": (len(wins) / float(len(selected)) * 100.0) if selected else None,
            "avg_return_pct": (sum(returns) / float(len(returns))) if returns else None,
        }
    return buckets


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
    df = filter_dataset(df, groups=groups, strategies=strategies)
    df = build_features(df)
    if "label_filled" not in df.columns or "label_win" not in df.columns:
        raise ValueError("Dataset must include label_filled and label_win columns from Phase 1")

    usable_df = df[df["label_filled"] == True].copy()
    usable_df = apply_entry_quality_labels(usable_df)

    overall_label_counts = label_counts(usable_df["entry_quality_label"])
    too_small = [label for label, count in overall_label_counts.items() if count < int(args.min_class_rows)]
    if too_small:
        raise ValueError(
            f"Not enough rows per class for training: {too_small} with counts {overall_label_counts}"
        )

    train_df, holdout_df = chronological_split(usable_df, args.test_days, args.min_train_days)
    if train_df.empty or holdout_df.empty:
        raise ValueError("Unable to create chronological train/holdout split from dataset")

    train_label_counts = label_counts(train_df["entry_quality_label"])
    holdout_label_counts = label_counts(holdout_df["entry_quality_label"])
    too_small_train = [label for label, count in train_label_counts.items() if count < int(args.min_class_rows)]
    if too_small_train:
        raise ValueError(
            f"Training split has insufficient rows per class: {too_small_train} with counts {train_label_counts}"
        )

    categorical_features, numeric_features = available_features(train_df)
    feature_cols = categorical_features + numeric_features
    preprocessor = build_preprocessor(categorical_features, numeric_features)

    classifier = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=600,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_train = train_df[feature_cols]
    y_train = train_df["entry_quality_label"].astype(str)
    X_holdout = holdout_df[feature_cols]
    y_holdout = holdout_df["entry_quality_label"].astype(str)

    classifier.fit(X_train, y_train)
    holdout_prob = classifier.predict_proba(X_holdout)
    classes = [str(label) for label in classifier.classes_]

    predicted_argmax = []
    predicted_threshold = []
    for row_prob in holdout_prob:
        prob_map = {label: float(row_prob[classes.index(label)]) if label in classes else 0.0 for label in LABELS}
        predicted_argmax.append(max(prob_map.items(), key=lambda item: item[1])[0])
        predicted_threshold.append(threshold_label(prob_map, args.entry_threshold, args.avoid_threshold))

    argmax_metrics = classification_metrics(y_holdout.tolist(), predicted_argmax)
    threshold_metrics = classification_metrics(y_holdout.tolist(), predicted_threshold)

    model_bundle = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model_type": "phase3_entry_quality_classifier",
        "classes": list(LABELS),
        "display_labels": DISPLAY_LABELS,
        "feature_columns": feature_cols,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "classifier": classifier,
        "metadata": {
            "input_path": str(input_path),
            "groups": groups,
            "strategies": strategies,
            "entry_threshold": float(args.entry_threshold),
            "avoid_threshold": float(args.avoid_threshold),
            "train_label_counts": train_label_counts,
            "holdout_label_counts": holdout_label_counts,
            "overall_label_counts": overall_label_counts,
        },
    }

    artifact_path = output_dir / "phase3_entry_quality_model.joblib"
    metrics_path = output_dir / "phase3_entry_quality_metrics.json"
    sample_predictions_path = output_dir / "phase3_entry_quality_holdout_predictions.jsonl"

    sample_rows = holdout_prediction_rows(
        classifier,
        holdout_df,
        feature_cols,
        args.entry_threshold,
        args.avoid_threshold,
    )
    metrics_payload = {
        "trained_at": model_bundle["trained_at"],
        "input_path": str(input_path),
        "artifact_path": str(artifact_path),
        "row_count_total": int(len(df)),
        "row_count_usable": int(len(usable_df)),
        "row_count_train": int(len(train_df)),
        "row_count_holdout": int(len(holdout_df)),
        "groups": groups,
        "strategies": strategies,
        "entry_threshold": float(args.entry_threshold),
        "avoid_threshold": float(args.avoid_threshold),
        "overall_label_counts": overall_label_counts,
        "train_label_counts": train_label_counts,
        "holdout_label_counts": holdout_label_counts,
        "argmax_metrics": argmax_metrics,
        "threshold_metrics": threshold_metrics,
        "holdout_realized_by_argmax_prediction": realized_summary_for_predictions(sample_rows, "predicted_label_argmax"),
        "holdout_realized_by_threshold_prediction": realized_summary_for_predictions(sample_rows, "predicted_label_threshold"),
        "feature_columns": feature_cols,
    }

    joblib.dump(model_bundle, artifact_path)
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with sample_predictions_path.open("w", encoding="utf-8") as fh:
        for row in sample_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "artifact_path": str(artifact_path),
                "metrics_path": str(metrics_path),
                "sample_predictions_path": str(sample_predictions_path),
                "overall_label_counts": overall_label_counts,
                "train_label_counts": train_label_counts,
                "holdout_label_counts": holdout_label_counts,
                "argmax_metrics": {
                    "accuracy": argmax_metrics.get("accuracy"),
                    "balanced_accuracy": argmax_metrics.get("balanced_accuracy"),
                    "macro_f1": argmax_metrics.get("macro_f1"),
                },
                "threshold_metrics": {
                    "accuracy": threshold_metrics.get("accuracy"),
                    "balanced_accuracy": threshold_metrics.get("balanced_accuracy"),
                    "macro_f1": threshold_metrics.get("macro_f1"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
