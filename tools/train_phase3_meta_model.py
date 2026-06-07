import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_phase2_walkforward import filter_dataset, load_dataset, parse_csv_list, resolve_input_path


DEFAULT_CATEGORICAL_FEATURES = [
    "strategy",
    "symbol",
    "signal",
    "candidate_group",
    "market_regime",
    "market_trend_bias",
    "side_bias",
    "alert_tier",
    "alert_intent",
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
    "risk_reward",
    "adjusted_run_cap",
    "symbol_cap",
    "quality_drop_confidence",
    "quality_drop_entry_window",
    "price_at_checkpoint",
    "entry_gap_pct",
    "stop_risk_pct",
    "target_reward_pct",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Phase 3 lightweight AI trader meta-model trainer from Phase 1 research dataset"
    )
    parser.add_argument(
        "--input-path",
        default="",
        help="Path to phase1_candidates.csv/jsonl (default: .data/research/phase1/phase1_candidates.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for model artifact and metrics (default: alongside input under phase3)",
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
        "--min-filled",
        type=int,
        default=30,
        help="Minimum filled rows required to train the classifier",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=60,
        help="Holdout window in days from the tail of the dataset",
    )
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=90,
        help="Minimum training history required before the holdout period",
    )
    parser.add_argument(
        "--target-mode",
        choices=("win_only", "win_and_return"),
        default="win_and_return",
        help="Train only classifier or classifier plus return regressor",
    )
    parser.add_argument(
        "--win-prob-threshold",
        type=float,
        default=0.55,
        help="Threshold used to build suggested AI decision classes on holdout",
    )
    return parser


def resolve_output_dir(input_path, raw_output_dir):
    raw = str(raw_output_dir or "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (input_path.resolve().parents[0] / path)
    return input_path.resolve().parents[0] / "phase3"


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
        "risk_reward",
        "adjusted_run_cap",
        "symbol_cap",
        "quality_drop_confidence",
        "quality_drop_entry_window",
    )
    for col in numeric_seed_columns:
        if col not in out.columns:
            out[col] = None
    for col in numeric_seed_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    entry = out["entry_price"]
    stop = out["stop_loss"]
    target = out["take_profit"]
    current = out["price_at_checkpoint"]
    entry_abs = entry.abs().replace(0, pd.NA)

    out["entry_gap_pct"] = ((current - entry).abs() / entry_abs) * 100.0
    out["stop_risk_pct"] = ((entry - stop).abs() / entry_abs) * 100.0
    out["target_reward_pct"] = ((target - entry).abs() / entry_abs) * 100.0

    for col in DEFAULT_CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    return out


def available_features(df):
    cat = [col for col in DEFAULT_CATEGORICAL_FEATURES if col in df.columns]
    num = [
        col
        for col in DEFAULT_NUMERIC_FEATURES
        if col in df.columns and df[col].notna().any()
    ]
    return cat, num


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
    test_df = df[df["checkpoint_at"] >= test_start].copy()
    if train_df.empty or test_df.empty:
        split_index = int(len(df) * 0.8)
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()
    if not train_df.empty:
        min_train_span = pd.Timestamp(train_df["checkpoint_at"].max()) - pd.Timestamp(train_df["checkpoint_at"].min())
        if min_train_span < pd.Timedelta(days=int(min_train_days)):
            split_index = int(len(df) * 0.75)
            train_df = df.iloc[:split_index].copy()
            test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def classifier_metrics(y_true, prob, threshold):
    predicted = [float(value) >= float(threshold) for value in prob]
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)) if len(y_true) else None,
        "predicted_positive_rate_pct": (sum(predicted) / float(len(predicted)) * 100.0) if predicted else 0.0,
    }
    try:
        unique = sorted({bool(value) for value in y_true})
        if len(unique) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
        else:
            metrics["roc_auc"] = None
    except Exception:
        metrics["roc_auc"] = None
    return metrics


def regressor_metrics(y_true, pred):
    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "r2": None}
    mse = mean_squared_error(y_true, pred)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(math.sqrt(mse)) if mse is not None else None,
        "r2": float(r2_score(y_true, pred)) if len(y_true) >= 2 else None,
    }


def decision_label(prob_win, expected_return, threshold):
    prob = _safe_float(prob_win, 0.0) or 0.0
    ret = _safe_float(expected_return, None)
    if prob >= float(threshold) and (ret is None or ret > 0.0):
        return "entry"
    if prob >= max(0.45, float(threshold) - 0.1):
        return "watch"
    return "avoid"


def sample_holdout_predictions(model_bundle, holdout_df, feature_cols, threshold):
    if holdout_df.empty:
        return []
    sample = holdout_df.copy().sort_values("checkpoint_at").tail(50)
    classifier = model_bundle["classifier"]
    regressor = model_bundle.get("regressor")
    X = sample[feature_cols]
    prob = classifier.predict_proba(X)[:, 1]
    expected_return = regressor.predict(X) if regressor is not None else [None] * len(sample)
    rows = []
    for idx, (_, row) in enumerate(sample.iterrows()):
        rows.append(
            {
                "checkpoint_at": pd.Timestamp(row["checkpoint_at"]).isoformat() if pd.notna(row["checkpoint_at"]) else None,
                "strategy": str(row.get("strategy") or ""),
                "symbol": str(row.get("symbol") or ""),
                "signal": str(row.get("signal") or ""),
                "candidate_group": str(row.get("candidate_group") or ""),
                "actual_win": bool(row.get("label_win")) if pd.notna(row.get("label_win")) else None,
                "actual_return_pct": _safe_float(row.get("label_return_pct"), None),
                "ai_prob_win": float(prob[idx]),
                "ai_expected_return_pct": _safe_float(expected_return[idx], None),
                "ai_decision": decision_label(prob[idx], expected_return[idx], threshold),
            }
        )
    return rows


def main():
    parser = build_parser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    input_path = resolve_input_path(root, args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Phase 1 dataset not found: {input_path}")
    output_dir = resolve_output_dir(input_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    groups = parse_csv_list(args.groups)
    strategies = [value.strip().upper() for value in parse_csv_list(args.strategies)]
    df = filter_dataset(df, groups=groups, strategies=strategies)
    df = build_features(df)
    if "label_filled" not in df.columns or "label_win" not in df.columns:
        raise ValueError("Dataset must include label_filled and label_win columns from Phase 1")

    filled_df = df[df["label_filled"] == True].copy()
    filled_df = filled_df[filled_df["label_win"].notna()].copy()
    if len(filled_df) < int(args.min_filled):
        raise ValueError(f"Not enough filled rows for training: {len(filled_df)} < {int(args.min_filled)}")

    train_df, holdout_df = chronological_split(filled_df, args.test_days, args.min_train_days)
    if train_df.empty or holdout_df.empty:
        raise ValueError("Unable to create chronological train/holdout split from dataset")

    categorical_features, numeric_features = available_features(train_df)
    feature_cols = categorical_features + numeric_features
    preprocessor = build_preprocessor(categorical_features, numeric_features)

    classifier = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=400, class_weight="balanced")),
        ]
    )

    X_train = train_df[feature_cols]
    y_train = train_df["label_win"].astype(bool)
    X_holdout = holdout_df[feature_cols]
    y_holdout = holdout_df["label_win"].astype(bool)

    classifier.fit(X_train, y_train)
    holdout_prob = classifier.predict_proba(X_holdout)[:, 1]
    clf_metrics = {
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "train_positive_rate_pct": float(y_train.mean() * 100.0) if len(y_train) else 0.0,
        "holdout_positive_rate_pct": float(y_holdout.mean() * 100.0) if len(y_holdout) else 0.0,
    }
    clf_metrics.update(classifier_metrics(y_holdout.tolist(), holdout_prob.tolist(), args.win_prob_threshold))

    regressor = None
    reg_metrics = None
    if args.target_mode == "win_and_return" and "label_return_pct" in train_df.columns:
        reg_train = train_df[train_df["label_return_pct"].notna()].copy()
        reg_holdout = holdout_df[holdout_df["label_return_pct"].notna()].copy()
        if not reg_train.empty and not reg_holdout.empty:
            regressor = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(categorical_features, numeric_features)),
                    ("model", Ridge(alpha=1.0)),
                ]
            )
            regressor.fit(reg_train[feature_cols], reg_train["label_return_pct"].astype(float))
            reg_pred = regressor.predict(reg_holdout[feature_cols])
            reg_metrics = regressor_metrics(reg_holdout["label_return_pct"].astype(float).tolist(), reg_pred.tolist())
            reg_metrics["train_rows"] = int(len(reg_train))
            reg_metrics["holdout_rows"] = int(len(reg_holdout))

    model_bundle = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model_type": "phase3_meta_trader",
        "feature_columns": feature_cols,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "classifier": classifier,
        "regressor": regressor,
        "metadata": {
            "input_path": str(input_path),
            "groups": groups,
            "strategies": strategies,
            "target_mode": args.target_mode,
            "win_prob_threshold": float(args.win_prob_threshold),
        },
    }

    artifact_path = output_dir / "phase3_meta_model.joblib"
    metrics_path = output_dir / "phase3_training_metrics.json"
    sample_predictions_path = output_dir / "phase3_holdout_predictions.jsonl"

    sample_rows = sample_holdout_predictions(model_bundle, holdout_df, feature_cols, args.win_prob_threshold)
    metrics_payload = {
        "trained_at": model_bundle["trained_at"],
        "input_path": str(input_path),
        "artifact_path": str(artifact_path),
        "row_count_total": int(len(df)),
        "row_count_filled": int(len(filled_df)),
        "row_count_train": int(len(train_df)),
        "row_count_holdout": int(len(holdout_df)),
        "classifier_metrics": clf_metrics,
        "regressor_metrics": reg_metrics,
        "feature_columns": feature_cols,
        "groups": groups,
        "strategies": strategies,
        "target_mode": args.target_mode,
        "win_prob_threshold": float(args.win_prob_threshold),
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
                "classifier_metrics": clf_metrics,
                "regressor_metrics": reg_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
