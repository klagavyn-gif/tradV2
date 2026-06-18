import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from run_phase2_walkforward import filter_dataset, load_dataset, parse_csv_list, resolve_input_path
from train_phase3_entry_quality_model import (
    DISPLAY_LABELS,
    LABELS,
    apply_entry_quality_labels,
    available_features,
    build_features,
    build_preprocessor,
    classification_metrics,
    chronological_split,
    label_counts,
    threshold_label,
)


LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}
MODEL_VERSION = "v4_utf_v2"
MODEL_TYPE = "phase4_entry_quality_v4_utf_calibrated"
ARTIFACT_PREFIX = "phase4_entry_quality_v4"
DEFAULT_OUTPUT_DIRNAME = "phase4_entry_quality_v4"
LOG_PREFIX = "v4"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train V4 UTF prototype entry ranker with optional GPU backend and threshold optimization"
    )
    parser.add_argument(
        "--input-path",
        default="",
        help="Path to phase1_candidates.csv/jsonl (default: .data/research/phase1/phase1_candidates.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for V4 artifacts (default: alongside input under phase4_entry_quality_v4)",
    )
    parser.add_argument(
        "--groups",
        default="primary,trend_radar,daily",
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
        default=45,
        help="Holdout window in days from the tail of the dataset",
    )
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=120,
        help="Minimum training history required before the holdout period",
    )
    parser.add_argument(
        "--min-class-rows",
        type=int,
        default=20,
        help="Minimum rows required per class after labeling",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "logreg", "xgboost"),
        default="auto",
        help="Training backend. auto prefers xgboost when installed, otherwise falls back to logistic regression.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Preferred compute device. cuda is used only when xgboost is available.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=("none", "platt", "isotonic"),
        default="platt",
        help="Optional probability calibration layer applied after base model training",
    )
    parser.add_argument(
        "--calibration-days",
        type=int,
        default=21,
        help="Tail window in days, taken from the training split, reserved for calibration",
    )
    parser.add_argument(
        "--calibration-min-rows",
        type=int,
        default=60,
        help="Minimum number of calibration rows required before enabling the calibrator",
    )
    parser.add_argument(
        "--calibration-min-train-days",
        type=int,
        default=60,
        help="Minimum chronological span to preserve for the model-fit split before calibration",
    )
    parser.add_argument(
        "--entry-threshold-min",
        type=float,
        default=0.65,
        help="Minimum entry threshold to scan during holdout policy optimization",
    )
    parser.add_argument(
        "--entry-threshold-max",
        type=float,
        default=0.95,
        help="Maximum entry threshold to scan during holdout policy optimization",
    )
    parser.add_argument(
        "--entry-threshold-step",
        type=float,
        default=0.05,
        help="Entry threshold scan step",
    )
    parser.add_argument(
        "--avoid-threshold-min",
        type=float,
        default=0.75,
        help="Minimum avoid threshold to scan during holdout policy optimization",
    )
    parser.add_argument(
        "--avoid-threshold-max",
        type=float,
        default=0.95,
        help="Maximum avoid threshold to scan during holdout policy optimization",
    )
    parser.add_argument(
        "--avoid-threshold-step",
        type=float,
        default=0.05,
        help="Avoid threshold scan step",
    )
    parser.add_argument(
        "--min-selected-rows",
        type=int,
        default=80,
        help="Minimum holdout rows predicted as entry for a threshold pair to be considered",
    )
    parser.add_argument(
        "--min-alerts-per-day",
        type=float,
        default=0.25,
        help="Minimum average entry alerts per day on holdout to consider a threshold pair",
    )
    parser.add_argument(
        "--target-alerts-per-day",
        type=float,
        default=1.0,
        help="Target alert frequency used in optimization score balancing",
    )
    parser.add_argument(
        "--max-alerts-per-day",
        type=float,
        default=2.5,
        help="Soft cap for average entry alerts per day during policy optimization",
    )
    parser.add_argument(
        "--min-win-rate-pct",
        type=float,
        default=57.5,
        help="Minimum realized entry win rate required for a policy to be considered viable",
    )
    parser.add_argument(
        "--min-avg-return-pct",
        type=float,
        default=2.0,
        help="Minimum realized average entry return required for a policy to be considered viable",
    )
    parser.add_argument(
        "--premium-min-selected-rows",
        type=int,
        default=40,
        help="Minimum selected rows for the Premium policy target",
    )
    parser.add_argument(
        "--premium-target-alerts-per-day",
        type=float,
        default=1.0,
        help="Target alerts/day for Premium policy",
    )
    parser.add_argument(
        "--premium-max-alerts-per-day",
        type=float,
        default=1.5,
        help="Soft cap alerts/day for Premium policy",
    )
    parser.add_argument(
        "--premium-min-win-rate-pct",
        type=float,
        default=60.0,
        help="Minimum win rate for Premium policy",
    )
    parser.add_argument(
        "--premium-max-win-rate-pct",
        type=float,
        default=65.0,
        help="Preferred upper bound of the Premium target win-rate band",
    )
    parser.add_argument(
        "--premium-min-avg-return-pct",
        type=float,
        default=2.0,
        help="Minimum average return for Premium policy",
    )
    parser.add_argument(
        "--premium-entry-threshold-min",
        type=float,
        default=0.60,
        help="Minimum entry threshold to scan for the Premium policy",
    )
    parser.add_argument(
        "--premium-entry-threshold-max",
        type=float,
        default=0.85,
        help="Maximum entry threshold to scan for the Premium policy",
    )
    parser.add_argument(
        "--premium-entry-threshold-step",
        type=float,
        default=0.02,
        help="Entry threshold scan step for the Premium policy",
    )
    parser.add_argument(
        "--premium-avoid-threshold-min",
        type=float,
        default=0.75,
        help="Minimum avoid threshold to scan for the Premium policy",
    )
    parser.add_argument(
        "--premium-avoid-threshold-max",
        type=float,
        default=0.95,
        help="Maximum avoid threshold to scan for the Premium policy",
    )
    parser.add_argument(
        "--premium-avoid-threshold-step",
        type=float,
        default=0.05,
        help="Avoid threshold scan step for the Premium policy",
    )
    parser.add_argument(
        "--standard-min-selected-rows",
        type=int,
        default=80,
        help="Minimum selected rows for the Standard policy target",
    )
    parser.add_argument(
        "--standard-target-alerts-per-day",
        type=float,
        default=2.5,
        help="Target alerts/day for Standard policy",
    )
    parser.add_argument(
        "--standard-max-alerts-per-day",
        type=float,
        default=4.0,
        help="Soft cap alerts/day for Standard policy",
    )
    parser.add_argument(
        "--standard-min-win-rate-pct",
        type=float,
        default=55.0,
        help="Minimum win rate for Standard policy",
    )
    parser.add_argument(
        "--standard-max-win-rate-pct",
        type=float,
        default=60.0,
        help="Preferred upper bound of the Standard target win-rate band",
    )
    parser.add_argument(
        "--standard-min-avg-return-pct",
        type=float,
        default=1.5,
        help="Minimum average return for Standard policy",
    )
    parser.add_argument(
        "--standard-entry-threshold-min",
        type=float,
        default=0.55,
        help="Minimum entry threshold to scan for the Standard policy",
    )
    parser.add_argument(
        "--standard-entry-threshold-max",
        type=float,
        default=0.80,
        help="Maximum entry threshold to scan for the Standard policy",
    )
    parser.add_argument(
        "--standard-entry-threshold-step",
        type=float,
        default=0.02,
        help="Entry threshold scan step for the Standard policy",
    )
    parser.add_argument(
        "--standard-avoid-threshold-min",
        type=float,
        default=0.70,
        help="Minimum avoid threshold to scan for the Standard policy",
    )
    parser.add_argument(
        "--standard-avoid-threshold-max",
        type=float,
        default=0.90,
        help="Maximum avoid threshold to scan for the Standard policy",
    )
    parser.add_argument(
        "--standard-avoid-threshold-step",
        type=float,
        default=0.05,
        help="Avoid threshold scan step for the Standard policy",
    )
    parser.add_argument(
        "--watch-min-selected-rows",
        type=int,
        default=0,
        help="Optional sample floor for the Watch monitor tier (0 disables the requirement)",
    )
    parser.add_argument(
        "--watch-target-alerts-per-day",
        type=float,
        default=3.0,
        help="Target alerts/day for the Watch monitor tier",
    )
    parser.add_argument(
        "--watch-max-alerts-per-day",
        type=float,
        default=6.0,
        help="Soft cap alerts/day for the Watch monitor tier",
    )
    parser.add_argument(
        "--watch-min-win-rate-pct",
        type=float,
        default=48.0,
        help="Minimum realized win rate for the Watch monitor tier",
    )
    parser.add_argument(
        "--watch-max-win-rate-pct",
        type=float,
        default=57.5,
        help="Preferred upper bound of the Watch target win-rate band",
    )
    parser.add_argument(
        "--watch-min-avg-return-pct",
        type=float,
        default=1.0,
        help="Minimum average return for the Watch monitor tier",
    )
    parser.add_argument(
        "--watch-entry-threshold-min",
        type=float,
        default=0.50,
        help="Minimum entry threshold to scan for the Watch policy",
    )
    parser.add_argument(
        "--watch-entry-threshold-max",
        type=float,
        default=0.80,
        help="Maximum entry threshold to scan for the Watch policy",
    )
    parser.add_argument(
        "--watch-entry-threshold-step",
        type=float,
        default=0.05,
        help="Entry threshold scan step for the Watch policy",
    )
    parser.add_argument(
        "--watch-avoid-threshold-min",
        type=float,
        default=0.70,
        help="Minimum avoid threshold to scan for the Watch policy",
    )
    parser.add_argument(
        "--watch-avoid-threshold-max",
        type=float,
        default=0.95,
        help="Maximum avoid threshold to scan for the Watch policy",
    )
    parser.add_argument(
        "--watch-avoid-threshold-step",
        type=float,
        default=0.05,
        help="Avoid threshold scan step for the Watch policy",
    )
    parser.add_argument(
        "--watch-min-master-score",
        type=float,
        default=0.19,
        help="Minimum v4_master_score required for a Watch candidate",
    )
    parser.add_argument(
        "--watch-min-entry-precision-score",
        type=float,
        default=0.18,
        help="Minimum v4_entry_precision_score required for a Watch candidate",
    )
    parser.add_argument(
        "--watch-min-execution-utility-score",
        type=float,
        default=0.40,
        help="Minimum v4_execution_utility_score required for a Watch candidate",
    )
    parser.add_argument(
        "--watch-min-regime-score",
        type=float,
        default=0.58,
        help="Minimum v4_regime_score required for a Watch candidate",
    )
    parser.add_argument(
        "--watch-min-direction-score",
        type=float,
        default=0.46,
        help="Minimum v4_direction_score required for a Watch candidate",
    )
    parser.add_argument(
        "--watch-min-exit-quality-score",
        type=float,
        default=0.34,
        help="Minimum v4_exit_quality_score required for a Watch candidate",
    )
    parser.add_argument(
        "--policy-calibration-target-weight",
        type=float,
        default=0.10,
        help="Objective weight for selected-label probability alignment during policy optimization",
    )
    parser.add_argument(
        "--policy-calibration-avoid-weight",
        type=float,
        default=0.06,
        help="Objective weight for avoid-probability alignment during policy optimization",
    )
    parser.add_argument(
        "--policy-calibration-overconfidence-penalty-weight",
        type=float,
        default=0.08,
        help="Penalty weight when selected-label probability is materially above realized rate",
    )
    parser.add_argument(
        "--strategy-policy-enable",
        action="store_true",
        default=True,
        help="Optimize additional policy thresholds per strategy on the holdout split",
    )
    parser.add_argument(
        "--strategy-policy-min-holdout-rows",
        type=int,
        default=90,
        help="Minimum number of holdout rows required before a strategy gets its own policy set",
    )
    return parser


def resolve_output_dir(root, input_path, raw_output_dir):
    raw = str(raw_output_dir or "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (root / path)
    return input_path.resolve().parents[0] / DEFAULT_OUTPUT_DIRNAME


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


def _mean(values):
    usable = [float(value) for value in values if isinstance(value, (int, float))]
    if not usable:
        return None
    return float(sum(usable) / float(len(usable)))


def _is_true_like(value):
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "win"}
    return bool(value)


def _band_score(value, lower, upper):
    numeric = _safe_float(value, None)
    if numeric is None:
        return 0.0
    low = float(lower)
    high = float(max(upper, lower))
    if low <= numeric <= high:
        return 1.0
    width = max(high - low, 1.0)
    if numeric < low:
        return _clamp01(1.0 - ((low - numeric) / width))
    return _clamp01(1.0 - ((numeric - high) / width))


def _format_duration(seconds):
    total_seconds = max(int(round(float(seconds or 0))), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _print_step(step_no, total_steps, title):
    print(f"[{LOG_PREFIX}] Step {step_no}/{total_steps}: {title}", flush=True)


def _clamp01(value, *, default=0.0):
    numeric = _safe_float(value, None)
    if numeric is None:
        return float(default)
    return float(min(max(numeric, 0.0), 1.0))


def _percent_like_score(value, *, default=0.0):
    numeric = _safe_float(value, None)
    if numeric is None:
        return float(default)
    if 0.0 <= numeric <= 1.0:
        return _clamp01(numeric, default=default)
    return _clamp01(numeric / 100.0, default=default)


def augment_v4_features(df):
    out = df.copy()
    for col in (
        "entry_gap_pct",
        "stop_risk_pct",
        "target_reward_pct",
        "rr_ratio",
        "confidence",
        "source_count",
        "ai_prob_win",
        "ai_expected_return_pct",
        "score",
        "alert_tier_score",
        "tier_rank",
        "backtest_win_rate_pct",
        "backtest_expectancy_rr",
        "backtest_trades",
        "ai_rank_adjustment",
        "short_trade_score_adjustment",
        "short_trade_regime_aligned",
    ):
        if col not in out.columns:
            out[col] = None
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "rr_ratio" not in out.columns or out["rr_ratio"].isna().all():
        reward = pd.to_numeric(out.get("target_reward_pct"), errors="coerce")
        risk = pd.to_numeric(out.get("stop_risk_pct"), errors="coerce").replace(0, pd.NA)
        out["rr_ratio"] = reward / risk

    confidence_score = out["confidence"].map(lambda value: _percent_like_score(value, default=0.0))
    score_strength = out["score"].map(lambda value: _percent_like_score(value, default=0.0))
    alert_tier_score = out["alert_tier_score"].map(lambda value: _percent_like_score(value, default=0.0))
    source_strength_score = out["source_count"].map(lambda value: _clamp01((_safe_float(value, 0.0) or 0.0) / 3.0))
    freshness_score = out["entry_gap_pct"].map(lambda value: _clamp01(1.0 - ((_safe_float(value, 0.80) or 0.80) / 0.80)))
    reward_score = out["target_reward_pct"].map(lambda value: _clamp01((_safe_float(value, 0.0) or 0.0) / 3.0))
    rr_score = out["rr_ratio"].map(lambda value: _clamp01(((_safe_float(value, 0.80) or 0.80) - 0.80) / 2.20))
    ai_support_score = out["ai_prob_win"].map(lambda value: _clamp01((_safe_float(value, 0.50) or 0.50)))
    expected_return_score = out["ai_expected_return_pct"].map(
        lambda value: _clamp01(((_safe_float(value, 0.0) or 0.0) + 1.0) / 4.0)
    )
    regime_alignment_score = out["short_trade_regime_aligned"].map(lambda value: 1.0 if _safe_float(value, 0.0) >= 1.0 else 0.45)
    backtest_win_rate_score = out["backtest_win_rate_pct"].map(lambda value: _percent_like_score(value, default=0.50))
    backtest_expectancy_score = out["backtest_expectancy_rr"].map(
        lambda value: _clamp01(((_safe_float(value, 0.50) or 0.50) + 0.50) / 2.50)
    )
    backtest_depth_score = out["backtest_trades"].map(lambda value: _clamp01((_safe_float(value, 0.0) or 0.0) / 150.0))
    ai_rank_score = out["ai_rank_adjustment"].map(lambda value: _clamp01(((_safe_float(value, 0.0) or 0.0) + 1.0) / 2.0))
    short_trade_context_score = out["short_trade_score_adjustment"].map(
        lambda value: _clamp01(((_safe_float(value, 0.0) or 0.0) + 1.0) / 2.0)
    )
    liquidity_quality_score = (0.55 * source_strength_score + 0.45 * backtest_depth_score).clip(lower=0.0, upper=1.0)
    confirmation_quality_score = (
        0.40 * confidence_score
        + 0.20 * score_strength
        + 0.20 * alert_tier_score
        + 0.20 * ai_support_score
    ).clip(lower=0.0, upper=1.0)
    price_location_score = (
        0.40 * freshness_score
        + 0.25 * confidence_score
        + 0.20 * alert_tier_score
        + 0.15 * ai_rank_score
    ).clip(lower=0.0, upper=1.0)
    timing_quality_score = (0.65 * freshness_score + 0.35 * confirmation_quality_score).clip(lower=0.0, upper=1.0)
    stop_efficiency_score = out["stop_risk_pct"].map(lambda value: _clamp01(1.0 - (((_safe_float(value, 1.80) or 1.80) - 1.20) / 1.80)))
    stop_safety_score = out["stop_risk_pct"].map(lambda value: _clamp01(1.0 - (((_safe_float(value, 1.80) or 1.80) - 1.00) / 2.20)))
    reward_accessibility_score = (
        0.55 * reward_score + 0.45 * rr_score
    ).clip(lower=0.0, upper=1.0)
    mfe_potential_score = (
        0.45 * reward_score
        + 0.30 * expected_return_score
        + 0.25 * backtest_expectancy_score
    ).clip(lower=0.0, upper=1.0)
    mae_control_score = (
        0.60 * stop_safety_score + 0.40 * stop_efficiency_score
    ).clip(lower=0.0, upper=1.0)
    exit_stability_score = (
        0.45 * backtest_expectancy_score
        + 0.30 * backtest_win_rate_score
        + 0.25 * backtest_depth_score
    ).clip(lower=0.0, upper=1.0)

    entry_gap_penalty = out["entry_gap_pct"].map(lambda value: _clamp01(((_safe_float(value, 0.0) or 0.0) - 0.35) / 0.85))
    stop_risk_penalty = out["stop_risk_pct"].map(lambda value: _clamp01(((_safe_float(value, 0.0) or 0.0) - 1.80) / 1.20))
    low_reward_penalty = out["target_reward_pct"].map(lambda value: _clamp01((1.25 - (_safe_float(value, 1.25) or 1.25)) / 1.25))
    overextension_penalty = out["entry_gap_pct"].map(lambda value: _clamp01(((_safe_float(value, 0.0) or 0.0) - 0.60) / 0.90))
    regime_conflict_penalty = (1.0 - regime_alignment_score).clip(lower=0.0, upper=1.0)
    low_liquidity_penalty = (1.0 - liquidity_quality_score).clip(lower=0.0, upper=1.0)
    walkforward_penalty = (1.0 - exit_stability_score).clip(lower=0.0, upper=1.0)

    out["v4_confidence_score"] = confidence_score
    out["v4_score_strength"] = score_strength
    out["v4_alert_tier_score"] = alert_tier_score
    out["v4_source_strength_score"] = source_strength_score
    out["v4_freshness_score"] = freshness_score
    out["v4_reward_score"] = reward_score
    out["v4_rr_score"] = rr_score
    out["v4_ai_support_score"] = ai_support_score
    out["v4_expected_return_score"] = expected_return_score
    out["v4_regime_alignment_score"] = regime_alignment_score
    out["v4_backtest_win_rate_score"] = backtest_win_rate_score
    out["v4_backtest_expectancy_score"] = backtest_expectancy_score
    out["v4_backtest_depth_score"] = backtest_depth_score
    out["v4_ai_rank_score"] = ai_rank_score
    out["v4_short_trade_context_score"] = short_trade_context_score
    out["v4_liquidity_quality_score"] = liquidity_quality_score
    out["v4_confirmation_quality_score"] = confirmation_quality_score
    out["v4_price_location_score"] = price_location_score
    out["v4_timing_quality_score"] = timing_quality_score
    out["v4_stop_efficiency_score"] = stop_efficiency_score
    out["v4_stop_safety_score"] = stop_safety_score
    out["v4_reward_accessibility_score"] = reward_accessibility_score
    out["v4_mfe_potential_score"] = mfe_potential_score
    out["v4_mae_control_score"] = mae_control_score
    out["v4_exit_stability_score"] = exit_stability_score
    out["v4_entry_gap_penalty"] = entry_gap_penalty
    out["v4_stop_risk_penalty"] = stop_risk_penalty
    out["v4_low_reward_penalty"] = low_reward_penalty
    out["v4_overextension_penalty"] = overextension_penalty
    out["v4_regime_conflict_penalty"] = regime_conflict_penalty
    out["v4_low_liquidity_penalty"] = low_liquidity_penalty
    out["v4_walkforward_penalty"] = walkforward_penalty

    out["v4_regime_score"] = (
        0.28 * out["v4_regime_alignment_score"]
        + 0.18 * out["v4_backtest_win_rate_score"]
        + 0.14 * out["v4_short_trade_context_score"]
        + 0.14 * out["v4_ai_rank_score"]
        + 0.12 * out["v4_source_strength_score"]
        + 0.14 * out["v4_freshness_score"]
    ).clip(lower=0.0, upper=1.0)

    out["v4_direction_score"] = (
        0.30 * out["v4_ai_support_score"]
        + 0.20 * out["v4_expected_return_score"]
        + 0.16 * out["v4_backtest_expectancy_score"]
        + 0.14 * out["v4_reward_score"]
        + 0.10 * out["v4_rr_score"]
        + 0.10 * out["v4_confidence_score"]
        - 0.10 * out["v4_overextension_penalty"]
    ).clip(lower=0.0, upper=1.0)

    out["v4_entry_precision_score"] = (
        0.24 * out["v4_freshness_score"]
        + 0.18 * out["v4_stop_efficiency_score"]
        + 0.18 * out["v4_reward_accessibility_score"]
        + 0.14 * out["v4_price_location_score"]
        + 0.10 * out["v4_confirmation_quality_score"]
        + 0.08 * out["v4_liquidity_quality_score"]
        + 0.08 * out["v4_timing_quality_score"]
        - 0.22 * out["v4_entry_gap_penalty"]
        - 0.16 * out["v4_overextension_penalty"]
        - 0.12 * out["v4_low_liquidity_penalty"]
    ).clip(lower=0.0, upper=1.0)

    out["v4_exit_quality_score"] = (
        0.26 * out["v4_rr_score"]
        + 0.20 * out["v4_reward_accessibility_score"]
        + 0.16 * out["v4_stop_safety_score"]
        + 0.14 * out["v4_mfe_potential_score"]
        + 0.12 * out["v4_mae_control_score"]
        + 0.12 * out["v4_exit_stability_score"]
    ).clip(lower=0.0, upper=1.0)

    out["v4_execution_utility_score"] = (
        0.28 * out["v4_expected_return_score"]
        + 0.18 * out["v4_backtest_expectancy_score"]
        + 0.14 * out["v4_backtest_depth_score"]
        + 0.12 * out["v4_regime_score"]
        + 0.10 * out["v4_ai_support_score"]
        + 0.10 * out["v4_liquidity_quality_score"]
        + 0.08 * out["v4_source_strength_score"]
        - 0.10 * out["v4_stop_risk_penalty"]
        - 0.08 * out["v4_low_reward_penalty"]
    ).clip(lower=0.0, upper=1.0)

    out["v4_quality_core"] = (
        0.22 * out["v4_regime_score"]
        + 0.24 * out["v4_direction_score"]
        + 0.24 * out["v4_entry_precision_score"]
        + 0.18 * out["v4_exit_quality_score"]
        + 0.12 * out["v4_execution_utility_score"]
    ).clip(lower=0.0, upper=1.0)

    out["v4_risk_core"] = (
        0.24 * out["v4_entry_gap_penalty"]
        + 0.22 * out["v4_stop_risk_penalty"]
        + 0.16 * out["v4_overextension_penalty"]
        + 0.14 * out["v4_regime_conflict_penalty"]
        + 0.12 * out["v4_low_liquidity_penalty"]
        + 0.12 * out["v4_walkforward_penalty"]
    ).clip(lower=0.0, upper=1.0)

    out["v4_master_score"] = (
        out["v4_quality_core"] - (0.70 * out["v4_risk_core"])
    ).clip(lower=0.0, upper=1.0)

    # Compatibility aliases keep the older V3 summary fields usable while the trainer shifts to V4 scores.
    out["v3_quality_score"] = out["v4_entry_precision_score"]
    out["v3_trade_utility_score"] = out["v4_execution_utility_score"]
    out["v3_master_score"] = out["v4_master_score"]
    return out


def extend_v4_numeric_features(df, categorical_features, numeric_features):
    numeric = list(numeric_features)
    for col in sorted(df.columns):
        if not (col.startswith("v3_") or col.startswith("v4_")):
            continue
        if col not in numeric and pd.to_numeric(df[col], errors="coerce").notna().any():
            numeric.append(col)
    return list(categorical_features), numeric


def build_sample_weights(y_train):
    counts = label_counts(y_train)
    usable_counts = {label: count for label, count in counts.items() if count > 0}
    if not usable_counts:
        return None
    total = float(sum(usable_counts.values()))
    class_weights = {
        label: total / float(len(usable_counts) * count)
        for label, count in usable_counts.items()
        if count > 0
    }
    return [float(class_weights.get(str(label), 1.0)) for label in y_train]


def resolve_backend(requested_backend):
    requested = str(requested_backend or "auto").strip().lower()
    if requested in {"logreg", "xgboost"}:
        return requested
    try:
        import xgboost  # noqa: F401

        return "xgboost"
    except Exception:
        return "logreg"


def resolve_device_order(backend, requested_device):
    requested = str(requested_device or "auto").strip().lower()
    if backend != "xgboost":
        return ["cpu"]
    if requested == "cuda":
        return ["cuda"]
    if requested == "cpu":
        return ["cpu"]
    return ["cuda", "cpu"]


def _clip_probability(value, eps=1e-6):
    numeric = _safe_float(value, None)
    if numeric is None:
        return None
    return float(min(max(float(numeric), float(eps)), 1.0 - float(eps)))


def _normalize_probability_row(scores, fallback_scores=None):
    cleaned = []
    for score in ([] if scores is None else scores):
        numeric = _safe_float(score, 0.0)
        cleaned.append(max(float(numeric), 0.0))
    total = float(sum(cleaned))
    if total <= 0:
        fallback = []
        for score in ([] if fallback_scores is None else fallback_scores):
            numeric = _safe_float(score, 0.0)
            fallback.append(max(float(numeric), 0.0))
        fallback_total = float(sum(fallback))
        if fallback_total > 0:
            return [float(value) / float(fallback_total) for value in fallback]
        if not cleaned:
            return []
        uniform = 1.0 / float(len(cleaned))
        return [uniform for _ in cleaned]
    return [float(value) / float(total) for value in cleaned]


def bundle_predict_proba_raw(bundle, df):
    feature_cols = bundle["feature_columns"]
    if bundle["backend"] == "xgboost":
        X_encoded = bundle["preprocessor"].transform(df[feature_cols])
        predicted = bundle["model"].predict_proba(X_encoded)
    else:
        predicted = bundle["estimator"].predict_proba(df[feature_cols])
    classes = [str(label) for label in (bundle.get("classes") or [])]
    return predicted, classes


def _apply_binary_probability_calibrator(model_info, raw_probability):
    raw_prob = _clip_probability(raw_probability, eps=1e-6)
    if raw_prob is None:
        return None
    if not isinstance(model_info, dict):
        return raw_prob
    calibrator_type = str(model_info.get("type") or "identity").strip().lower()
    if calibrator_type == "platt" and model_info.get("model") is not None:
        model = model_info["model"]
        predicted = model.predict_proba(pd.DataFrame({"raw_prob": [raw_prob]}))[0][1]
        return _clip_probability(predicted, eps=1e-6)
    if calibrator_type == "isotonic" and model_info.get("model") is not None:
        model = model_info["model"]
        predicted = model.predict([raw_prob])[0]
        return _clip_probability(predicted, eps=1e-6)
    return raw_prob


def apply_probability_calibrator(bundle, predicted, classes):
    calibrator = bundle.get("probability_calibrator")
    if not isinstance(calibrator, dict) or not bool(calibrator.get("enabled")):
        return predicted
    models = calibrator.get("models") or {}
    calibrated_rows = []
    for row in ([] if predicted is None else predicted):
        row_scores = []
        for idx, label in enumerate(classes or []):
            raw_value = row[idx] if idx < len(row) else None
            row_scores.append(_apply_binary_probability_calibrator(models.get(str(label)), raw_value))
        calibrated_rows.append(_normalize_probability_row(row_scores, fallback_scores=row))
    return calibrated_rows


def bundle_predict_proba(bundle, df):
    predicted, classes = bundle_predict_proba_raw(bundle, df)
    return apply_probability_calibrator(bundle, predicted, classes)


def probability_maps_from_matrix(probability_matrix, classes):
    rows = []
    classes = [str(label) for label in (classes or [])]
    for row_prob in ([] if probability_matrix is None else probability_matrix):
        rows.append(
            {
                label: float(row_prob[classes.index(label)]) if label in classes else 0.0
                for label in LABELS
            }
        )
    return rows


def _binary_probability_metrics(probabilities, targets, *, bin_count=10):
    usable = []
    for probability, target in zip(probabilities or [], targets or []):
        prob_value = _clip_probability(probability, eps=1e-6)
        if prob_value is None:
            continue
        usable.append((float(prob_value), 1.0 if bool(target) else 0.0))
    if not usable:
        return {"row_count": 0}

    probs = [item[0] for item in usable]
    actuals = [item[1] for item in usable]
    row_count = len(usable)
    positives = int(sum(actuals))
    brier_score = sum((prob - actual) ** 2 for prob, actual in usable) / float(row_count)
    log_loss = -sum(
        (actual * math.log(prob)) + ((1.0 - actual) * math.log(1.0 - prob))
        for prob, actual in usable
    ) / float(row_count)

    bins = []
    ece = 0.0
    mce = 0.0
    bin_count = max(int(bin_count), 4)
    for idx in range(bin_count):
        lower = float(idx) / float(bin_count)
        upper = float(idx + 1) / float(bin_count)
        is_last_bin = idx == bin_count - 1
        bucket = [
            (prob, actual)
            for prob, actual in usable
            if ((prob >= lower and prob <= upper) if is_last_bin else (prob >= lower and prob < upper))
        ]
        if not bucket:
            continue
        avg_predicted = sum(prob for prob, _ in bucket) / float(len(bucket))
        actual_rate = sum(actual for _, actual in bucket) / float(len(bucket))
        gap_abs = abs(avg_predicted - actual_rate)
        bucket_payload = {
            "bin_index": int(idx),
            "prob_lower": lower,
            "prob_upper": upper,
            "row_count": int(len(bucket)),
            "share_pct": (float(len(bucket)) / float(row_count)) * 100.0,
            "avg_predicted_prob": float(avg_predicted),
            "actual_positive_rate": float(actual_rate),
            "gap_abs": float(gap_abs),
        }
        bins.append(bucket_payload)
        ece += gap_abs * (float(len(bucket)) / float(row_count))
        mce = max(mce, gap_abs)
    return {
        "row_count": int(row_count),
        "positives": int(positives),
        "positive_rate": float(sum(actuals) / float(row_count)),
        "avg_predicted_prob": float(sum(probs) / float(row_count)),
        "brier_score": float(brier_score),
        "log_loss": float(log_loss),
        "ece": float(ece),
        "mce": float(mce),
        "bins": bins,
    }


def summarize_probability_calibration(prob_maps, actual_labels):
    labels = [str(label) for label in (actual_labels or [])]
    summary = {}
    for target_label in LABELS:
        summary[target_label] = _binary_probability_metrics(
            [prob_map.get(target_label) if isinstance(prob_map, dict) else None for prob_map in (prob_maps or [])],
            [label == target_label for label in labels],
        )
    return summary


def _policy_probability_alignment(selected_rows, *, selected_label):
    if not selected_rows:
        return {
            "selected_target_prob_mean": None,
            "selected_target_actual_rate": None,
            "selected_target_gap_abs": None,
            "selected_target_alignment_score": None,
            "avoid_prob_mean": None,
            "avoid_actual_rate": None,
            "avoid_gap_abs": None,
            "avoid_alignment_score": None,
            "overconfidence_penalty": None,
        }
    selected_label = str(selected_label or "entry").strip().lower()
    target_prob_values = []
    avoid_prob_values = []
    target_hits = []
    avoid_hits = []
    for row in selected_rows:
        if selected_label == "watch":
            target_prob_values.append(_safe_float(row.get("prob_watch"), None))
        else:
            target_prob_values.append(_safe_float(row.get("prob_entry"), None))
        avoid_prob_values.append(_safe_float(row.get("prob_avoid"), None))
        actual_label = str(row.get("actual_label") or "").strip().lower()
        target_hits.append(1.0 if actual_label == selected_label else 0.0)
        avoid_hits.append(1.0 if actual_label == "avoid" else 0.0)

    target_prob_mean = _mean(target_prob_values)
    avoid_prob_mean = _mean(avoid_prob_values)
    target_actual_rate = _mean(target_hits)
    avoid_actual_rate = _mean(avoid_hits)
    target_gap_abs = abs(float(target_prob_mean) - float(target_actual_rate)) if isinstance(target_prob_mean, float) and isinstance(target_actual_rate, float) else None
    avoid_gap_abs = abs(float(avoid_prob_mean) - float(avoid_actual_rate)) if isinstance(avoid_prob_mean, float) and isinstance(avoid_actual_rate, float) else None
    overconfidence_penalty = None
    if isinstance(target_prob_mean, float) and isinstance(target_actual_rate, float):
        overconfidence_penalty = _clamp01(float(target_prob_mean) - float(target_actual_rate))
    return {
        "selected_target_prob_mean": target_prob_mean,
        "selected_target_actual_rate": target_actual_rate,
        "selected_target_gap_abs": target_gap_abs,
        "selected_target_alignment_score": (1.0 - float(target_gap_abs)) if isinstance(target_gap_abs, float) else None,
        "avoid_prob_mean": avoid_prob_mean,
        "avoid_actual_rate": avoid_actual_rate,
        "avoid_gap_abs": avoid_gap_abs,
        "avoid_alignment_score": (1.0 - float(avoid_gap_abs)) if isinstance(avoid_gap_abs, float) else None,
        "overconfidence_penalty": overconfidence_penalty,
    }


def prepare_calibration_split(train_df, args):
    requested_method = str(getattr(args, "calibration_method", "none") or "none").strip().lower()
    details = {
        "requested_method": requested_method,
        "status": "disabled" if requested_method == "none" else "pending",
        "reason": "calibration_disabled" if requested_method == "none" else None,
        "fit_row_count": int(len(train_df)),
        "calibration_row_count": 0,
    }
    if requested_method == "none":
        return train_df, train_df.iloc[0:0].copy(), details

    calibration_days = max(int(getattr(args, "calibration_days", 21) or 21), 7)
    calibration_min_train_days = max(int(getattr(args, "calibration_min_train_days", 60) or 60), 30)
    fit_df, calibration_df = chronological_split(train_df, calibration_days, calibration_min_train_days)
    if fit_df.empty or calibration_df.empty:
        details.update({"status": "skipped", "reason": "calibration_split_empty"})
        return train_df, train_df.iloc[0:0].copy(), details
    if len(calibration_df) < int(getattr(args, "calibration_min_rows", 60) or 60):
        details.update(
            {
                "status": "skipped",
                "reason": "calibration_rows_below_minimum",
                "calibration_row_count": int(len(calibration_df)),
            }
        )
        return train_df, train_df.iloc[0:0].copy(), details
    fit_label_counts = label_counts(fit_df["entry_quality_label"])
    too_small_fit = [label for label, count in fit_label_counts.items() if count < int(args.min_class_rows)]
    if too_small_fit:
        details.update(
            {
                "status": "skipped",
                "reason": f"fit_split_insufficient_classes:{','.join(too_small_fit)}",
                "calibration_row_count": int(len(calibration_df)),
            }
        )
        return train_df, train_df.iloc[0:0].copy(), details
    details.update(
        {
            "status": "enabled",
            "reason": None,
            "fit_row_count": int(len(fit_df)),
            "calibration_row_count": int(len(calibration_df)),
            "fit_label_counts": fit_label_counts,
            "calibration_label_counts": label_counts(calibration_df["entry_quality_label"]),
        }
    )
    return fit_df, calibration_df, details


def fit_probability_calibrator(bundle, calibration_df, *, method):
    info = {
        "requested_method": str(method or "none"),
        "status": "skipped",
        "applied_method": None,
        "row_count": int(len(calibration_df)),
        "classes": list(bundle.get("classes") or []),
        "per_label_models": {},
    }
    if calibration_df is None or calibration_df.empty:
        info["reason"] = "empty_calibration_df"
        return bundle, info

    raw_predicted, classes = bundle_predict_proba_raw(bundle, calibration_df)
    if raw_predicted is None or not len(raw_predicted):
        info["reason"] = "raw_predict_proba_unavailable"
        return bundle, info

    actual_labels = calibration_df["entry_quality_label"].astype(str).tolist()
    models = {}
    for label in LABELS:
        if label not in classes:
            models[label] = {"type": "identity", "reason": "class_missing_from_bundle"}
            info["per_label_models"][label] = {"type": "identity", "reason": "class_missing_from_bundle"}
            continue
        class_index = classes.index(label)
        raw_probs = [_clip_probability(row[class_index], eps=1e-6) or 0.5 for row in raw_predicted]
        targets = [1 if actual == label else 0 for actual in actual_labels]
        positives = int(sum(targets))
        negatives = int(len(targets) - positives)
        if positives < 2 or negatives < 2:
            models[label] = {
                "type": "identity",
                "reason": "insufficient_class_support",
                "positives": positives,
                "negatives": negatives,
            }
            info["per_label_models"][label] = {
                "type": "identity",
                "reason": "insufficient_class_support",
                "positives": positives,
                "negatives": negatives,
            }
            continue
        try:
            if str(method).strip().lower() == "isotonic":
                model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                model.fit(raw_probs, targets)
                model_type = "isotonic"
            else:
                model = LogisticRegression(max_iter=400)
                model.fit(pd.DataFrame({"raw_prob": raw_probs}), targets)
                model_type = "platt"
            models[label] = {
                "type": model_type,
                "model": model,
                "positives": positives,
                "negatives": negatives,
            }
            info["per_label_models"][label] = {
                "type": model_type,
                "positives": positives,
                "negatives": negatives,
            }
        except Exception as exc:
            models[label] = {
                "type": "identity",
                "reason": f"fit_failed:{exc}",
                "positives": positives,
                "negatives": negatives,
            }
            info["per_label_models"][label] = {
                "type": "identity",
                "reason": f"fit_failed:{exc}",
                "positives": positives,
                "negatives": negatives,
            }

    bundle["probability_calibrator"] = {
        "enabled": True,
        "method": str(method).strip().lower(),
        "classes": list(classes),
        "models": models,
    }
    info["status"] = "applied"
    info["applied_method"] = str(method).strip().lower()
    info["reason"] = None
    return bundle, info


def train_logreg_bundle(train_df, feature_cols, categorical_features, numeric_features):
    preprocessor = build_preprocessor(categorical_features, numeric_features)
    classifier = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=800,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    X_train = train_df[feature_cols]
    y_train = train_df["entry_quality_label"].astype(str)
    classifier.fit(X_train, y_train)
    return {
        "backend": "logreg",
        "device": "cpu",
        "estimator": classifier,
        "feature_columns": feature_cols,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "classes": list(LABELS),
    }


def train_xgboost_bundle(train_df, feature_cols, categorical_features, numeric_features, device_order):
    import xgboost as xgb

    X_train = train_df[feature_cols]
    y_train = train_df["entry_quality_label"].astype(str)
    y_index = [LABEL_TO_INDEX[str(label)] for label in y_train]
    sample_weights = build_sample_weights(y_train)
    preprocessor = build_preprocessor(categorical_features, numeric_features)
    X_encoded = preprocessor.fit_transform(X_train)
    last_error = None

    for device_name in device_order:
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(LABELS),
            n_estimators=420,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.0,
            min_child_weight=2.0,
            tree_method="hist",
            device=device_name,
            random_state=42,
            n_jobs=0,
            eval_metric="mlogloss",
        )
        try:
            model.fit(X_encoded, y_index, sample_weight=sample_weights)
            return {
                "backend": "xgboost",
                "device": device_name,
                "preprocessor": preprocessor,
                "model": model,
                "feature_columns": feature_cols,
                "categorical_features": categorical_features,
                "numeric_features": numeric_features,
                "classes": list(LABELS),
            }
        except Exception as exc:
            last_error = exc
            if device_name == device_order[-1]:
                raise

    raise RuntimeError(f"Unable to train xgboost backend: {last_error}")


def train_bundle(train_df, feature_cols, categorical_features, numeric_features, *, backend, device):
    resolved_backend = resolve_backend(backend)
    if resolved_backend == "xgboost":
        try:
            return train_xgboost_bundle(
                train_df,
                feature_cols,
                categorical_features,
                numeric_features,
                resolve_device_order("xgboost", device),
            )
        except Exception:
            if str(backend or "").strip().lower() == "xgboost":
                raise
    return train_logreg_bundle(train_df, feature_cols, categorical_features, numeric_features)


def iter_thresholds(min_value, max_value, step):
    current = float(min_value)
    values = []
    while current <= float(max_value) + 1e-9:
        values.append(round(current, 4))
        current += float(step)
    return values


def holdout_span_days(df):
    if df.empty:
        return 1.0
    start_at = pd.Timestamp(df["checkpoint_at"].min())
    end_at = pd.Timestamp(df["checkpoint_at"].max())
    span = max((end_at - start_at).total_seconds() / 86400.0, 1.0)
    return float(span)


def evaluate_policy(
    holdout_df,
    prob_maps,
    entry_threshold,
    avoid_threshold,
    *,
    policy_name,
    min_selected_rows,
    min_alerts_per_day,
    target_alerts_per_day,
    max_alerts_per_day,
    min_win_rate_pct,
    max_win_rate_pct,
    min_avg_return_pct,
    min_master_score=0.0,
    min_regime_score=0.0,
    min_direction_score=0.0,
    min_entry_precision_score=0.0,
    min_exit_quality_score=0.0,
    min_execution_utility_score=0.0,
    calibration_target_weight=0.10,
    calibration_avoid_weight=0.06,
    calibration_overconfidence_penalty_weight=0.08,
):
    predictions = [threshold_label(prob_map, entry_threshold, avoid_threshold) for prob_map in prob_maps]
    metrics = classification_metrics(holdout_df["entry_quality_label"].astype(str).tolist(), predictions)
    selected_label = "watch" if str(policy_name or "").strip().lower() == "watch" else "entry"
    selected_rows = []
    for idx, predicted in enumerate(predictions):
        if predicted != selected_label:
            continue
        row = holdout_df.iloc[idx]
        row_regime_score = _safe_float(row.get("v4_regime_score"), None)
        row_direction_score = _safe_float(row.get("v4_direction_score"), None)
        row_entry_precision_score = _safe_float(row.get("v4_entry_precision_score"), None)
        row_exit_quality_score = _safe_float(row.get("v4_exit_quality_score"), None)
        row_execution_utility_score = _safe_float(row.get("v4_execution_utility_score"), None)
        row_master_score = _safe_float(row.get("v4_master_score"), None)
        if any(
            float(threshold) > 0.0
            for threshold in (
                min_master_score,
                min_regime_score,
                min_direction_score,
                min_entry_precision_score,
                min_exit_quality_score,
                min_execution_utility_score,
            )
        ):
            if row_master_score is None or row_master_score < float(min_master_score):
                continue
            if row_regime_score is None or row_regime_score < float(min_regime_score):
                continue
            if row_direction_score is None or row_direction_score < float(min_direction_score):
                continue
            if row_entry_precision_score is None or row_entry_precision_score < float(min_entry_precision_score):
                continue
            if row_exit_quality_score is None or row_exit_quality_score < float(min_exit_quality_score):
                continue
            if row_execution_utility_score is None or row_execution_utility_score < float(min_execution_utility_score):
                continue
        selected_rows.append(
            {
                "label_win": row.get("label_win"),
                "label_return_pct": _safe_float(row.get("label_return_pct"), None),
                "signal": str(row.get("signal") or ""),
                "v4_regime_score": row_regime_score,
                "v4_direction_score": row_direction_score,
                "v4_entry_precision_score": row_entry_precision_score,
                "v4_exit_quality_score": row_exit_quality_score,
                "v4_execution_utility_score": row_execution_utility_score,
                "v4_master_score": row_master_score,
                "prob_entry": _safe_float(prob_maps[idx].get("entry"), None) if isinstance(prob_maps[idx], dict) else None,
                "prob_watch": _safe_float(prob_maps[idx].get("watch"), None) if isinstance(prob_maps[idx], dict) else None,
                "prob_avoid": _safe_float(prob_maps[idx].get("avoid"), None) if isinstance(prob_maps[idx], dict) else None,
                "actual_label": str(row.get("entry_quality_label") or "").strip().lower() or None,
            }
        )
    holdout_days = holdout_span_days(holdout_df)
    count = len(selected_rows)
    alerts_per_day = float(count) / float(holdout_days) if holdout_days > 0 else 0.0
    win_rate_pct = None
    avg_return_pct = None
    if selected_rows:
        wins = [row for row in selected_rows if _is_true_like(row.get("label_win"))]
        returns = [row.get("label_return_pct") for row in selected_rows]
        win_rate_pct = float(len(wins) / float(len(selected_rows)) * 100.0)
        avg_return_pct = _mean(returns)
    avg_regime_score = _mean([row.get("v4_regime_score") for row in selected_rows])
    avg_direction_score = _mean([row.get("v4_direction_score") for row in selected_rows])
    avg_entry_precision_score = _mean([row.get("v4_entry_precision_score") for row in selected_rows])
    avg_exit_quality_score = _mean([row.get("v4_exit_quality_score") for row in selected_rows])
    avg_execution_utility_score = _mean([row.get("v4_execution_utility_score") for row in selected_rows])
    avg_master_score = _mean([row.get("v4_master_score") for row in selected_rows])
    coverage_pct = float(count) / float(len(holdout_df)) * 100.0 if len(holdout_df) else 0.0
    is_viable = (
        count >= int(min_selected_rows)
        and alerts_per_day >= float(min_alerts_per_day)
        and alerts_per_day <= float(max(max_alerts_per_day, min_alerts_per_day))
        and isinstance(avg_return_pct, float)
        and avg_return_pct >= float(min_avg_return_pct)
        and isinstance(win_rate_pct, float)
        and win_rate_pct >= float(min_win_rate_pct)
    )
    target_alerts_per_day = float(max(target_alerts_per_day, 0.01))
    max_alerts_per_day = float(max(max_alerts_per_day, target_alerts_per_day))
    if alerts_per_day <= target_alerts_per_day:
        frequency_score = _clamp01(alerts_per_day / target_alerts_per_day)
    else:
        over_target_span = max(max_alerts_per_day - target_alerts_per_day, 0.01)
        frequency_score = _clamp01(1.0 - ((alerts_per_day - target_alerts_per_day) / over_target_span))
    expected_return_score = _clamp01(((_safe_float(avg_return_pct, 0.0) or 0.0) + 0.5) / 4.5)
    win_rate_score = _clamp01((_safe_float(win_rate_pct, 0.0) or 0.0) / 100.0)
    win_rate_band_score = _band_score(win_rate_pct, min_win_rate_pct, max_win_rate_pct)
    overfilter_penalty = _clamp01((float(min_selected_rows) - float(count)) / float(max(min_selected_rows, 1))) if count < int(min_selected_rows) else 0.0
    low_frequency_penalty = _clamp01((float(min_alerts_per_day) - float(alerts_per_day)) / float(max(min_alerts_per_day, 0.01))) if alerts_per_day < float(min_alerts_per_day) else 0.0
    over_frequency_penalty = (
        _clamp01((alerts_per_day - max_alerts_per_day) / float(max(max_alerts_per_day, 0.01)))
        if alerts_per_day > max_alerts_per_day
        else 0.0
    )
    low_win_rate_penalty = (
        _clamp01((float(min_win_rate_pct) - float(win_rate_pct or 0.0)) / float(max(min_win_rate_pct, 1.0)))
        if isinstance(win_rate_pct, float) and win_rate_pct < float(min_win_rate_pct)
        else 0.0
    )
    low_return_penalty = (
        _clamp01((float(min_avg_return_pct) - float(avg_return_pct or 0.0)) / float(max(abs(min_avg_return_pct), 0.1)))
        if isinstance(avg_return_pct, float) and avg_return_pct < float(min_avg_return_pct)
        else 0.0
    )
    probability_alignment = _policy_probability_alignment(selected_rows, selected_label=selected_label)
    selected_target_alignment_score = _clamp01(
        probability_alignment.get("selected_target_alignment_score"),
        default=0.0,
    )
    avoid_alignment_score = _clamp01(
        probability_alignment.get("avoid_alignment_score"),
        default=0.0,
    )
    overconfidence_penalty = _clamp01(
        probability_alignment.get("overconfidence_penalty"),
        default=0.0,
    )
    sample_confidence_score = _clamp01(float(count) / float(max(int(min_selected_rows), 1)))
    quality_pocket_score = _clamp01(
        0.34 * win_rate_score
        + 0.18 * expected_return_score
        + 0.16 * float(metrics.get("balanced_accuracy") or 0.0)
        + 0.16 * selected_target_alignment_score
        + 0.10 * avoid_alignment_score
        + 0.06 * float(avg_execution_utility_score or 0.0)
    )
    objective_score = None
    objective_components = None
    if isinstance(win_rate_pct, float) and isinstance(avg_return_pct, float):
        balanced_accuracy_score = float(metrics.get("balanced_accuracy") or 0.0)
        if selected_label == "watch":
            objective_score = (
                0.28 * win_rate_score
                + 0.16 * win_rate_band_score
                + 0.18 * expected_return_score
                + 0.08 * float(avg_regime_score or 0.0)
                + 0.08 * float(avg_direction_score or 0.0)
                + 0.12 * float(avg_entry_precision_score or 0.0)
                + 0.10 * float(avg_exit_quality_score or 0.0)
                + 0.11 * float(avg_execution_utility_score or 0.0)
                + 0.06 * float(avg_master_score or 0.0)
                + 0.03 * float(frequency_score)
                + 0.04 * balanced_accuracy_score
                + float(calibration_target_weight) * selected_target_alignment_score
                + float(calibration_avoid_weight) * avoid_alignment_score
                - 0.28 * overfilter_penalty
                - 0.05 * low_frequency_penalty
                - 0.42 * over_frequency_penalty
                - 0.30 * low_win_rate_penalty
                - 0.20 * low_return_penalty
                - float(calibration_overconfidence_penalty_weight) * overconfidence_penalty
            )
        else:
            objective_score = (
                0.22 * quality_pocket_score
                + 0.22 * win_rate_score
                + 0.08 * win_rate_band_score
                + 0.12 * expected_return_score
                + 0.07 * float(avg_regime_score or 0.0)
                + 0.07 * float(avg_direction_score or 0.0)
                + 0.07 * float(avg_entry_precision_score or 0.0)
                + 0.06 * float(avg_exit_quality_score or 0.0)
                + 0.08 * float(avg_execution_utility_score or 0.0)
                + 0.04 * float(avg_master_score or 0.0)
                + 0.03 * sample_confidence_score
                + 0.03 * float(frequency_score)
                + 0.06 * balanced_accuracy_score
                + float(calibration_target_weight) * selected_target_alignment_score
                + float(calibration_avoid_weight) * avoid_alignment_score
                - 0.14 * overfilter_penalty
                - 0.08 * low_frequency_penalty
                - 0.22 * over_frequency_penalty
                - 0.36 * low_win_rate_penalty
                - 0.18 * low_return_penalty
                - float(calibration_overconfidence_penalty_weight) * overconfidence_penalty
            )
        objective_components = {
            "win_rate_score": win_rate_score,
            "win_rate_band_score": win_rate_band_score,
            "expected_return_score": expected_return_score,
            "quality_pocket_score": quality_pocket_score,
            "sample_confidence_score": sample_confidence_score,
            "avg_regime_score": avg_regime_score,
            "avg_direction_score": avg_direction_score,
            "avg_entry_precision_score": avg_entry_precision_score,
            "avg_exit_quality_score": avg_exit_quality_score,
            "avg_execution_utility_score": avg_execution_utility_score,
            "avg_master_score": avg_master_score,
            "frequency_score": frequency_score,
            "balanced_accuracy_score": balanced_accuracy_score,
            "overfilter_penalty": overfilter_penalty,
            "low_frequency_penalty": low_frequency_penalty,
            "over_frequency_penalty": over_frequency_penalty,
            "low_win_rate_penalty": low_win_rate_penalty,
            "low_return_penalty": low_return_penalty,
            "selected_target_alignment_score": selected_target_alignment_score,
            "avoid_alignment_score": avoid_alignment_score,
            "overconfidence_penalty": overconfidence_penalty,
            "selected_target_prob_mean": probability_alignment.get("selected_target_prob_mean"),
            "selected_target_actual_rate": probability_alignment.get("selected_target_actual_rate"),
            "avoid_prob_mean": probability_alignment.get("avoid_prob_mean"),
            "avoid_actual_rate": probability_alignment.get("avoid_actual_rate"),
        }
    return {
        "policy_name": str(policy_name),
        "entry_threshold": float(entry_threshold),
        "avoid_threshold": float(avoid_threshold),
        "selected_rows": int(count),
        "alerts_per_day": float(alerts_per_day),
        "coverage_pct": float(coverage_pct),
        "win_rate_pct": win_rate_pct,
        "avg_return_pct": avg_return_pct,
        "selected_label": selected_label,
        "avg_regime_score": avg_regime_score,
        "avg_direction_score": avg_direction_score,
        "avg_entry_precision_score": avg_entry_precision_score,
        "avg_exit_quality_score": avg_exit_quality_score,
        "avg_execution_utility_score": avg_execution_utility_score,
        "avg_master_score": avg_master_score,
        "objective_score": objective_score,
        "objective_components": objective_components,
        "selected_target_alignment_score": probability_alignment.get("selected_target_alignment_score"),
        "avoid_alignment_score": probability_alignment.get("avoid_alignment_score"),
        "overconfidence_penalty": probability_alignment.get("overconfidence_penalty"),
        "selected_target_prob_mean": probability_alignment.get("selected_target_prob_mean"),
        "selected_target_actual_rate": probability_alignment.get("selected_target_actual_rate"),
        "avoid_prob_mean": probability_alignment.get("avoid_prob_mean"),
        "avoid_actual_rate": probability_alignment.get("avoid_actual_rate"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "is_viable": bool(is_viable),
    }


def build_policy_profiles(args):
    return {
        "premium": {
            "name": "premium",
            "selected_label": "entry",
            "entry_threshold_min": float(args.premium_entry_threshold_min),
            "entry_threshold_max": float(args.premium_entry_threshold_max),
            "entry_threshold_step": float(args.premium_entry_threshold_step),
            "avoid_threshold_min": float(args.premium_avoid_threshold_min),
            "avoid_threshold_max": float(args.premium_avoid_threshold_max),
            "avoid_threshold_step": float(args.premium_avoid_threshold_step),
            "min_selected_rows": int(args.premium_min_selected_rows),
            "min_alerts_per_day": float(args.min_alerts_per_day),
            "target_alerts_per_day": float(args.premium_target_alerts_per_day),
            "max_alerts_per_day": float(args.premium_max_alerts_per_day),
            "min_win_rate_pct": float(args.premium_min_win_rate_pct),
            "max_win_rate_pct": float(args.premium_max_win_rate_pct),
            "min_avg_return_pct": float(args.premium_min_avg_return_pct),
            "min_master_score": 0.0,
            "min_regime_score": 0.0,
            "min_direction_score": 0.0,
            "min_entry_precision_score": 0.0,
            "min_exit_quality_score": 0.0,
            "min_execution_utility_score": 0.0,
            "calibration_target_weight": float(args.policy_calibration_target_weight),
            "calibration_avoid_weight": float(args.policy_calibration_avoid_weight),
            "calibration_overconfidence_penalty_weight": float(args.policy_calibration_overconfidence_penalty_weight),
        },
        "standard": {
            "name": "standard",
            "selected_label": "entry",
            "entry_threshold_min": float(args.standard_entry_threshold_min),
            "entry_threshold_max": float(args.standard_entry_threshold_max),
            "entry_threshold_step": float(args.standard_entry_threshold_step),
            "avoid_threshold_min": float(args.standard_avoid_threshold_min),
            "avoid_threshold_max": float(args.standard_avoid_threshold_max),
            "avoid_threshold_step": float(args.standard_avoid_threshold_step),
            "min_selected_rows": int(args.standard_min_selected_rows),
            "min_alerts_per_day": float(args.min_alerts_per_day),
            "target_alerts_per_day": float(args.standard_target_alerts_per_day),
            "max_alerts_per_day": float(args.standard_max_alerts_per_day),
            "min_win_rate_pct": float(args.standard_min_win_rate_pct),
            "max_win_rate_pct": float(args.standard_max_win_rate_pct),
            "min_avg_return_pct": float(args.standard_min_avg_return_pct),
            "min_master_score": 0.0,
            "min_regime_score": 0.0,
            "min_direction_score": 0.0,
            "min_entry_precision_score": 0.0,
            "min_exit_quality_score": 0.0,
            "min_execution_utility_score": 0.0,
            "calibration_target_weight": float(args.policy_calibration_target_weight),
            "calibration_avoid_weight": float(args.policy_calibration_avoid_weight),
            "calibration_overconfidence_penalty_weight": float(args.policy_calibration_overconfidence_penalty_weight),
        },
        "watch": {
            "name": "watch",
            "selected_label": "watch",
            "entry_threshold_min": float(args.watch_entry_threshold_min),
            "entry_threshold_max": float(args.watch_entry_threshold_max),
            "entry_threshold_step": float(args.watch_entry_threshold_step),
            "avoid_threshold_min": float(args.watch_avoid_threshold_min),
            "avoid_threshold_max": float(args.watch_avoid_threshold_max),
            "avoid_threshold_step": float(args.watch_avoid_threshold_step),
            "min_selected_rows": int(args.watch_min_selected_rows),
            "min_alerts_per_day": float(args.min_alerts_per_day),
            "target_alerts_per_day": float(args.watch_target_alerts_per_day),
            "max_alerts_per_day": float(args.watch_max_alerts_per_day),
            "min_win_rate_pct": float(args.watch_min_win_rate_pct),
            "max_win_rate_pct": float(args.watch_max_win_rate_pct),
            "min_avg_return_pct": float(args.watch_min_avg_return_pct),
            "min_master_score": float(args.watch_min_master_score),
            "min_regime_score": float(args.watch_min_regime_score),
            "min_direction_score": float(args.watch_min_direction_score),
            "min_entry_precision_score": float(args.watch_min_entry_precision_score),
            "min_exit_quality_score": float(args.watch_min_exit_quality_score),
            "min_execution_utility_score": float(args.watch_min_execution_utility_score),
            "calibration_target_weight": float(args.policy_calibration_target_weight),
            "calibration_avoid_weight": float(args.policy_calibration_avoid_weight),
            "calibration_overconfidence_penalty_weight": float(args.policy_calibration_overconfidence_penalty_weight),
        },
    }


def _profile_score_quantile(df, column_name, quantile_value, *, fallback, floor=None, ceil=None):
    if df is None or column_name not in df.columns:
        return float(fallback)
    values = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if values.empty:
        return float(fallback)
    out = float(values.quantile(float(quantile_value)))
    if floor is not None:
        out = max(out, float(floor))
    if ceil is not None:
        out = min(out, float(ceil))
    return float(out)


def _profile_probability_quantile(prob_maps, label_name, quantile_value, *, fallback, floor=None, ceil=None):
    if not prob_maps:
        return float(fallback)
    values = []
    for prob_map in prob_maps:
        if not isinstance(prob_map, dict):
            continue
        value = _safe_float(prob_map.get(label_name), None)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return float(fallback)
    out = float(pd.Series(values, dtype="float64").quantile(float(quantile_value)))
    if floor is not None:
        out = max(out, float(floor))
    if ceil is not None:
        out = min(out, float(ceil))
    return float(out)


def _filter_prob_maps(prob_maps, row_mask):
    if not prob_maps or row_mask is None:
        return []
    out = []
    for idx, keep in enumerate(list(row_mask)):
        if not keep or idx >= len(prob_maps):
            continue
        prob_map = prob_maps[idx]
        if isinstance(prob_map, dict):
            out.append(prob_map)
    return out


def build_strategy_specific_policy_profiles(args, strategy_name, holdout_df, prob_maps=None):
    profiles = build_policy_profiles(args)
    strategy_key = str(strategy_name or "").strip().upper()
    if strategy_key != "PA15" or holdout_df is None or holdout_df.empty:
        return profiles, {}

    label_series = holdout_df["entry_quality_label"].astype(str).str.strip().str.lower()
    target_mask = label_series.isin({"entry", "watch"})
    entry_mask = label_series == "entry"
    target_rows = holdout_df[target_mask].copy()
    entry_rows = holdout_df[entry_mask].copy()
    target_prob_maps = _filter_prob_maps(prob_maps, target_mask.tolist())
    entry_prob_maps = _filter_prob_maps(prob_maps, entry_mask.tolist())
    if target_rows.empty:
        return profiles, {}

    premium_profile = dict(profiles.get("premium") or {})
    original_premium_profile = dict(premium_profile)
    standard_profile = dict(profiles.get("standard") or {})
    original_standard_profile = dict(standard_profile)
    watch_profile = dict(profiles.get("watch") or {})
    original_watch_profile = dict(watch_profile)

    if not entry_rows.empty:
        premium_profile["min_selected_rows"] = min(int(premium_profile["min_selected_rows"]), 4)
        premium_profile["min_alerts_per_day"] = min(float(premium_profile["min_alerts_per_day"]), 0.08)
        premium_profile["target_alerts_per_day"] = min(float(premium_profile["target_alerts_per_day"]), 0.35)
        premium_profile["max_alerts_per_day"] = min(float(premium_profile["max_alerts_per_day"]), 0.80)
        premium_profile["min_win_rate_pct"] = min(float(premium_profile["min_win_rate_pct"]), 52.0)
        premium_profile["max_win_rate_pct"] = min(float(premium_profile["max_win_rate_pct"]), 60.0)
        premium_profile["min_avg_return_pct"] = min(float(premium_profile["min_avg_return_pct"]), 0.10)
        premium_profile["entry_threshold_min"] = _profile_probability_quantile(
            entry_prob_maps,
            "entry",
            0.10,
            fallback=premium_profile["entry_threshold_min"],
            floor=0.02,
            ceil=0.20,
        )
        premium_profile["entry_threshold_max"] = _profile_probability_quantile(
            entry_prob_maps,
            "entry",
            0.90,
            fallback=premium_profile["entry_threshold_max"],
            floor=max(float(premium_profile["entry_threshold_min"]) + 0.06, 0.08),
            ceil=0.45,
        )
        premium_profile["avoid_threshold_min"] = _profile_probability_quantile(
            entry_prob_maps,
            "avoid",
            0.10,
            fallback=premium_profile["avoid_threshold_min"],
            floor=0.55,
            ceil=0.75,
        )
        premium_profile["avoid_threshold_max"] = _profile_probability_quantile(
            entry_prob_maps,
            "avoid",
            0.90,
            fallback=premium_profile["avoid_threshold_max"],
            floor=max(float(premium_profile["avoid_threshold_min"]) + 0.05, 0.65),
            ceil=0.85,
        )
        premium_profile["min_master_score"] = _profile_score_quantile(
            entry_rows,
            "v4_master_score",
            0.10,
            fallback=premium_profile["min_master_score"],
            floor=0.18,
        )
        premium_profile["min_regime_score"] = _profile_score_quantile(
            entry_rows,
            "v4_regime_score",
            0.10,
            fallback=premium_profile["min_regime_score"],
            floor=0.40,
        )
        premium_profile["min_direction_score"] = _profile_score_quantile(
            entry_rows,
            "v4_direction_score",
            0.10,
            fallback=premium_profile["min_direction_score"],
            floor=0.40,
        )
        premium_profile["min_entry_precision_score"] = _profile_score_quantile(
            entry_rows,
            "v4_entry_precision_score",
            0.10,
            fallback=premium_profile["min_entry_precision_score"],
            floor=0.26,
        )
        premium_profile["min_exit_quality_score"] = _profile_score_quantile(
            entry_rows,
            "v4_exit_quality_score",
            0.10,
            fallback=premium_profile["min_exit_quality_score"],
            floor=0.44,
        )
        premium_profile["min_execution_utility_score"] = _profile_score_quantile(
            entry_rows,
            "v4_execution_utility_score",
            0.10,
            fallback=premium_profile["min_execution_utility_score"],
            floor=0.24,
        )

        standard_profile["min_selected_rows"] = min(int(standard_profile["min_selected_rows"]), 8)
        standard_profile["min_alerts_per_day"] = min(float(standard_profile["min_alerts_per_day"]), 0.16)
        standard_profile["target_alerts_per_day"] = min(float(standard_profile["target_alerts_per_day"]), 0.40)
        standard_profile["max_alerts_per_day"] = min(float(standard_profile["max_alerts_per_day"]), 0.90)
        standard_profile["min_win_rate_pct"] = min(float(standard_profile["min_win_rate_pct"]), 48.0)
        standard_profile["max_win_rate_pct"] = min(float(standard_profile["max_win_rate_pct"]), 58.0)
        standard_profile["min_avg_return_pct"] = min(float(standard_profile["min_avg_return_pct"]), 0.05)
        standard_profile["entry_threshold_min"] = _profile_probability_quantile(
            entry_prob_maps,
            "entry",
            0.20,
            fallback=standard_profile["entry_threshold_min"],
            floor=0.03,
            ceil=0.12,
        )
        standard_profile["entry_threshold_max"] = _profile_probability_quantile(
            entry_prob_maps,
            "entry",
            0.60,
            fallback=standard_profile["entry_threshold_max"],
            floor=max(float(standard_profile["entry_threshold_min"]) + 0.04, 0.08),
            ceil=0.20,
        )
        standard_profile["avoid_threshold_min"] = _profile_probability_quantile(
            entry_prob_maps,
            "avoid",
            0.55,
            fallback=standard_profile["avoid_threshold_min"],
            floor=0.70,
            ceil=0.80,
        )
        standard_profile["avoid_threshold_max"] = _profile_probability_quantile(
            entry_prob_maps,
            "avoid",
            0.90,
            fallback=standard_profile["avoid_threshold_max"],
            floor=max(float(standard_profile["avoid_threshold_min"]) + 0.05, 0.78),
            ceil=0.85,
        )
        standard_profile["min_master_score"] = _profile_score_quantile(
            entry_rows,
            "v4_master_score",
            0.15,
            fallback=standard_profile["min_master_score"],
            floor=0.20,
        )
        standard_profile["min_regime_score"] = _profile_score_quantile(
            entry_rows,
            "v4_regime_score",
            0.15,
            fallback=standard_profile["min_regime_score"],
            floor=0.42,
        )
        standard_profile["min_direction_score"] = _profile_score_quantile(
            entry_rows,
            "v4_direction_score",
            0.15,
            fallback=standard_profile["min_direction_score"],
            floor=0.41,
        )
        standard_profile["min_entry_precision_score"] = _profile_score_quantile(
            entry_rows,
            "v4_entry_precision_score",
            0.15,
            fallback=standard_profile["min_entry_precision_score"],
            floor=0.29,
        )
        standard_profile["min_exit_quality_score"] = _profile_score_quantile(
            entry_rows,
            "v4_exit_quality_score",
            0.15,
            fallback=standard_profile["min_exit_quality_score"],
            floor=0.46,
        )
        standard_profile["min_execution_utility_score"] = _profile_score_quantile(
            entry_rows,
            "v4_execution_utility_score",
            0.15,
            fallback=standard_profile["min_execution_utility_score"],
            floor=0.27,
        )

    watch_profile["min_master_score"] = _profile_score_quantile(
        target_rows,
        "v4_master_score",
        0.20,
        fallback=watch_profile["min_master_score"],
        floor=0.08,
        ceil=watch_profile["min_master_score"],
    )
    watch_profile["min_regime_score"] = _profile_score_quantile(
        target_rows,
        "v4_regime_score",
        0.15,
        fallback=watch_profile["min_regime_score"],
        floor=0.32,
        ceil=watch_profile["min_regime_score"],
    )
    watch_profile["min_direction_score"] = _profile_score_quantile(
        target_rows,
        "v4_direction_score",
        0.15,
        fallback=watch_profile["min_direction_score"],
        floor=0.34,
        ceil=watch_profile["min_direction_score"],
    )
    watch_profile["min_entry_precision_score"] = _profile_score_quantile(
        target_rows,
        "v4_entry_precision_score",
        0.20,
        fallback=watch_profile["min_entry_precision_score"],
        floor=0.14,
        ceil=watch_profile["min_entry_precision_score"],
    )
    watch_profile["min_exit_quality_score"] = _profile_score_quantile(
        target_rows,
        "v4_exit_quality_score",
        0.15,
        fallback=watch_profile["min_exit_quality_score"],
        floor=0.34,
        ceil=watch_profile["min_exit_quality_score"],
    )
    watch_profile["min_execution_utility_score"] = _profile_score_quantile(
        target_rows,
        "v4_execution_utility_score",
        0.20,
        fallback=watch_profile["min_execution_utility_score"],
        floor=0.15,
        ceil=watch_profile["min_execution_utility_score"],
    )
    watch_profile["min_avg_return_pct"] = min(float(watch_profile["min_avg_return_pct"]), 0.0)
    watch_profile["min_win_rate_pct"] = min(float(watch_profile["min_win_rate_pct"]), 42.0)
    watch_profile["target_alerts_per_day"] = min(float(watch_profile["target_alerts_per_day"]), 2.0)
    watch_profile["max_alerts_per_day"] = min(float(watch_profile["max_alerts_per_day"]), 4.0)
    watch_profile["entry_threshold_min"] = _profile_probability_quantile(
        prob_maps,
        "entry",
        0.99,
        fallback=watch_profile["entry_threshold_min"],
        floor=0.20,
        ceil=watch_profile["entry_threshold_min"],
    )
    watch_profile["entry_threshold_max"] = _profile_probability_quantile(
        prob_maps,
        "entry",
        0.999,
        fallback=watch_profile["entry_threshold_max"],
        floor=max(float(watch_profile["entry_threshold_min"]) + 0.10, 0.35),
        ceil=watch_profile["entry_threshold_max"],
    )
    watch_profile["avoid_threshold_min"] = _profile_probability_quantile(
        prob_maps,
        "avoid",
        0.25,
        fallback=watch_profile["avoid_threshold_min"],
        floor=0.40,
        ceil=watch_profile["avoid_threshold_min"],
    )
    watch_profile["avoid_threshold_max"] = _profile_probability_quantile(
        prob_maps,
        "avoid",
        0.75,
        fallback=watch_profile["avoid_threshold_max"],
        floor=max(float(watch_profile["avoid_threshold_min"]) + 0.10, 0.55),
        ceil=watch_profile["avoid_threshold_max"],
    )

    profiles["premium"] = premium_profile
    profiles["standard"] = standard_profile
    profiles["watch"] = watch_profile
    overrides = {
        "strategy": strategy_key,
        "premium": {
            key: premium_profile[key]
            for key in (
                "entry_threshold_min",
                "entry_threshold_max",
                "avoid_threshold_min",
                "avoid_threshold_max",
                "min_selected_rows",
                "min_alerts_per_day",
                "target_alerts_per_day",
                "max_alerts_per_day",
                "min_win_rate_pct",
                "max_win_rate_pct",
                "min_avg_return_pct",
                "min_master_score",
                "min_regime_score",
                "min_direction_score",
                "min_entry_precision_score",
                "min_exit_quality_score",
                "min_execution_utility_score",
            )
            if premium_profile.get(key) != original_premium_profile.get(key)
        },
        "standard": {
            key: standard_profile[key]
            for key in (
                "entry_threshold_min",
                "entry_threshold_max",
                "avoid_threshold_min",
                "avoid_threshold_max",
                "min_selected_rows",
                "min_alerts_per_day",
                "target_alerts_per_day",
                "max_alerts_per_day",
                "min_win_rate_pct",
                "max_win_rate_pct",
                "min_avg_return_pct",
                "min_master_score",
                "min_regime_score",
                "min_direction_score",
                "min_entry_precision_score",
                "min_exit_quality_score",
                "min_execution_utility_score",
            )
            if standard_profile.get(key) != original_standard_profile.get(key)
        },
        "watch": {
            key: watch_profile[key]
            for key in (
                "entry_threshold_min",
                "entry_threshold_max",
                "avoid_threshold_min",
                "avoid_threshold_max",
                "min_master_score",
                "min_regime_score",
                "min_direction_score",
                "min_entry_precision_score",
                "min_exit_quality_score",
                "min_execution_utility_score",
                "min_win_rate_pct",
                "min_avg_return_pct",
                "target_alerts_per_day",
                "max_alerts_per_day",
            )
            if watch_profile.get(key) != original_watch_profile.get(key)
        },
    }
    return profiles, overrides


def optimize_policy_thresholds(holdout_df, prob_maps, args, *, profiles=None):
    profiles = profiles or build_policy_profiles(args)
    candidates_by_profile = {name: [] for name in profiles}
    threshold_pairs_by_profile = {}
    total_pairs = 0
    for profile_name, profile in profiles.items():
        entry_thresholds = iter_thresholds(
            profile["entry_threshold_min"],
            profile["entry_threshold_max"],
            profile["entry_threshold_step"],
        )
        avoid_thresholds = iter_thresholds(
            profile["avoid_threshold_min"],
            profile["avoid_threshold_max"],
            profile["avoid_threshold_step"],
        )
        threshold_pairs = [
            (entry_threshold, avoid_threshold)
            for entry_threshold in entry_thresholds
            for avoid_threshold in avoid_thresholds
            if avoid_threshold > entry_threshold
        ]
        threshold_pairs_by_profile[profile_name] = threshold_pairs
        total_pairs += len(threshold_pairs)
    completed = 0
    started_at = time.time()
    best_so_far = {name: None for name in profiles}
    for profile_name, profile in profiles.items():
        for entry_threshold, avoid_threshold in threshold_pairs_by_profile.get(profile_name, []):
            candidate = evaluate_policy(
                holdout_df,
                prob_maps,
                entry_threshold,
                avoid_threshold,
                policy_name=profile_name,
                min_selected_rows=profile["min_selected_rows"],
                min_alerts_per_day=profile["min_alerts_per_day"],
                target_alerts_per_day=profile["target_alerts_per_day"],
                max_alerts_per_day=profile["max_alerts_per_day"],
                min_win_rate_pct=profile["min_win_rate_pct"],
                max_win_rate_pct=profile["max_win_rate_pct"],
                min_avg_return_pct=profile["min_avg_return_pct"],
                min_master_score=profile["min_master_score"],
                min_regime_score=profile["min_regime_score"],
                min_direction_score=profile["min_direction_score"],
                min_entry_precision_score=profile["min_entry_precision_score"],
                min_exit_quality_score=profile["min_exit_quality_score"],
                min_execution_utility_score=profile["min_execution_utility_score"],
                calibration_target_weight=profile["calibration_target_weight"],
                calibration_avoid_weight=profile["calibration_avoid_weight"],
                calibration_overconfidence_penalty_weight=profile["calibration_overconfidence_penalty_weight"],
            )
            candidates_by_profile[profile_name].append(candidate)
            completed += 1
            if best_so_far[profile_name] is None or float(candidate.get("objective_score") or -1e9) > float(best_so_far[profile_name].get("objective_score") or -1e9):
                best_so_far[profile_name] = candidate
            if completed == 1 or completed == total_pairs or completed % 10 == 0:
                elapsed = max(time.time() - started_at, 0.0)
                rate = elapsed / float(max(completed, 1))
                remaining = max(total_pairs - completed, 0)
                eta = rate * float(remaining)
                pct = (float(completed) / float(max(total_pairs, 1))) * 100.0
                best_text_parts = []
                for name in profiles:
                    best_candidate = best_so_far.get(name)
                    best_value = best_candidate.get("objective_score") if isinstance(best_candidate, dict) else None
                    label = name[:3]
                    best_text_parts.append(f"{label}={float(best_value):.4f}" if isinstance(best_value, (int, float)) else f"{label}=n/a")
                print(
                    f"[{LOG_PREFIX}] policy scan {completed}/{total_pairs} pairs ({pct:.1f}%) | "
                    f"best_score={' '.join(best_text_parts)} | elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}",
                    flush=True,
                )
    ranked_by_profile = {}
    best_by_profile = {}
    for profile_name, candidates in candidates_by_profile.items():
        viable = [row for row in candidates if row.get("is_viable")]
        ranked = sorted(
            viable or candidates,
            key=lambda row: (
                -float(row.get("objective_score") or -1e9),
                -float(row.get("win_rate_pct") or -1e9),
                -float(row.get("avg_return_pct") or -1e9),
                -int(row.get("selected_rows") or 0),
            ),
        )
        ranked_by_profile[profile_name] = ranked[:25]
        best_by_profile[profile_name] = ranked[0] if ranked else None
    return best_by_profile, ranked_by_profile


def _subset_prob_maps_for_indices(df, prob_maps, indices):
    if not indices:
        return []
    index_to_position = {index_value: idx for idx, index_value in enumerate(df.index.tolist())}
    out = []
    for index_value in indices:
        position = index_to_position.get(index_value)
        if position is None or position >= len(prob_maps):
            continue
        out.append(prob_maps[position])
    return out


def optimize_strategy_specific_policies(holdout_df, prob_maps, args):
    if not bool(getattr(args, "strategy_policy_enable", True)):
        return {}
    if holdout_df is None or holdout_df.empty or "strategy" not in holdout_df.columns:
        return {}

    min_holdout_rows = max(int(getattr(args, "strategy_policy_min_holdout_rows", 90) or 90), 30)
    strategy_results = {}
    strategy_series = holdout_df["strategy"].fillna("").astype(str).str.strip().str.upper()
    for strategy_name, strategy_df in holdout_df.groupby(strategy_series):
        strategy_key = str(strategy_name or "").strip().upper()
        if not strategy_key:
            continue
        if len(strategy_df) < min_holdout_rows:
            continue
        strategy_prob_maps = _subset_prob_maps_for_indices(holdout_df, prob_maps, strategy_df.index.tolist())
        if len(strategy_prob_maps) != len(strategy_df):
            continue
        strategy_profiles, profile_overrides = build_strategy_specific_policy_profiles(
            args,
            strategy_key,
            strategy_df,
            strategy_prob_maps,
        )
        best_policies, top_policies = optimize_policy_thresholds(
            strategy_df,
            strategy_prob_maps,
            args,
            profiles=strategy_profiles,
        )
        premium_policy = best_policies.get("premium")
        standard_policy = best_policies.get("standard")
        watch_policy = best_policies.get("watch")
        if not isinstance(premium_policy, dict) or not isinstance(standard_policy, dict) or not isinstance(watch_policy, dict):
            continue

        y_strategy = strategy_df["entry_quality_label"].astype(str).tolist()
        premium_predictions = [
            threshold_label(prob_map, premium_policy["entry_threshold"], premium_policy["avoid_threshold"])
            for prob_map in strategy_prob_maps
        ]
        standard_predictions = [
            threshold_label(prob_map, standard_policy["entry_threshold"], standard_policy["avoid_threshold"])
            for prob_map in strategy_prob_maps
        ]
        watch_predictions = [
            threshold_label(prob_map, watch_policy["entry_threshold"], watch_policy["avoid_threshold"])
            for prob_map in strategy_prob_maps
        ]
        strategy_results[strategy_key] = {
            "strategy": strategy_key,
            "holdout_row_count": int(len(strategy_df)),
            "holdout_label_counts": label_counts(strategy_df["entry_quality_label"]),
            "recommended_premium_policy": premium_policy,
            "recommended_standard_policy": standard_policy,
            "recommended_watch_policy": watch_policy,
            "premium_policy_metrics": classification_metrics(y_strategy, premium_predictions),
            "standard_policy_metrics": classification_metrics(y_strategy, standard_predictions),
            "watch_policy_metrics": classification_metrics(y_strategy, watch_predictions),
            "top_premium_policy_candidates": top_policies.get("premium", []),
            "top_standard_policy_candidates": top_policies.get("standard", []),
            "top_watch_policy_candidates": top_policies.get("watch", []),
            "policy_profile_overrides": profile_overrides,
        }
    return strategy_results


def summarize_strategy_row_counts(*, usable_df, train_df, model_train_df, calibration_df, holdout_df, args):
    def _select_strategy_rows(df, strategy_name):
        if df is None or "strategy" not in df.columns:
            return pd.DataFrame()
        if df.empty:
            return df.iloc[0:0].copy()
        mask = df["strategy"].fillna("").astype(str).str.strip().str.upper() == strategy_name
        return df[mask].copy()

    strategy_names = set()
    for df in (usable_df, train_df, model_train_df, calibration_df, holdout_df):
        if df is not None and not df.empty and "strategy" in df.columns:
            strategy_names.update(
                str(value).strip().upper()
                for value in df["strategy"].fillna("").astype(str).tolist()
                if str(value).strip()
            )
    out = {}
    min_holdout_rows = max(int(getattr(args, "strategy_policy_min_holdout_rows", 90) or 90), 30)
    for strategy_name in sorted(strategy_names):
        usable_rows = _select_strategy_rows(usable_df, strategy_name)
        train_rows = _select_strategy_rows(train_df, strategy_name)
        model_train_rows = _select_strategy_rows(model_train_df, strategy_name)
        calibration_rows = _select_strategy_rows(calibration_df, strategy_name)
        holdout_rows = _select_strategy_rows(holdout_df, strategy_name)
        out[strategy_name] = {
            "usable_rows": int(len(usable_rows)),
            "train_rows": int(len(train_rows)),
            "model_train_rows": int(len(model_train_rows)),
            "calibration_rows": int(len(calibration_rows)),
            "holdout_rows": int(len(holdout_rows)),
            "holdout_label_counts": label_counts(holdout_rows["entry_quality_label"]) if len(holdout_rows) else {},
            "strategy_policy_min_holdout_rows": int(min_holdout_rows),
            "strategy_policy_eligible": bool(len(holdout_rows) >= min_holdout_rows),
        }
    return out


def build_holdout_prediction_rows(holdout_df, prob_maps, premium_policy, standard_policy, watch_policy, *, raw_prob_maps=None):
    rows = []
    for idx, (_, row) in enumerate(holdout_df.iterrows()):
        prob_map = prob_maps[idx]
        raw_prob_map = raw_prob_maps[idx] if isinstance(raw_prob_maps, list) and idx < len(raw_prob_maps) else None
        predicted_argmax = max(prob_map.items(), key=lambda item: item[1])[0]
        predicted_premium = threshold_label(prob_map, premium_policy["entry_threshold"], premium_policy["avoid_threshold"])
        predicted_standard = threshold_label(prob_map, standard_policy["entry_threshold"], standard_policy["avoid_threshold"])
        predicted_watch = threshold_label(prob_map, watch_policy["entry_threshold"], watch_policy["avoid_threshold"])
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
                "predicted_label_premium": predicted_premium,
                "predicted_label_premium_display": DISPLAY_LABELS.get(predicted_premium, predicted_premium),
                "predicted_label_standard": predicted_standard,
                "predicted_label_standard_display": DISPLAY_LABELS.get(predicted_standard, predicted_standard),
                "predicted_label_watch": predicted_watch,
                "predicted_label_watch_display": DISPLAY_LABELS.get(predicted_watch, predicted_watch),
                "prob_entry": float(prob_map.get("entry", 0.0)),
                "prob_watch": float(prob_map.get("watch", 0.0)),
                "prob_avoid": float(prob_map.get("avoid", 0.0)),
                "raw_prob_entry": float(raw_prob_map.get("entry", 0.0)) if isinstance(raw_prob_map, dict) else None,
                "raw_prob_watch": float(raw_prob_map.get("watch", 0.0)) if isinstance(raw_prob_map, dict) else None,
                "raw_prob_avoid": float(raw_prob_map.get("avoid", 0.0)) if isinstance(raw_prob_map, dict) else None,
                "label_win": bool(row.get("label_win")) if pd.notna(row.get("label_win")) else None,
                "label_return_pct": _safe_float(row.get("label_return_pct"), None),
                "v4_regime_score": _safe_float(row.get("v4_regime_score"), None),
                "v4_direction_score": _safe_float(row.get("v4_direction_score"), None),
                "v4_entry_precision_score": _safe_float(row.get("v4_entry_precision_score"), None),
                "v4_exit_quality_score": _safe_float(row.get("v4_exit_quality_score"), None),
                "v4_execution_utility_score": _safe_float(row.get("v4_execution_utility_score"), None),
                "v4_master_score": _safe_float(row.get("v4_master_score"), None),
            }
        )
    return rows


def realized_summary(rows, label_key):
    buckets = {}
    for label in LABELS:
        selected = [row for row in rows if str(row.get(label_key) or "") == label]
        returns = [float(row["label_return_pct"]) for row in selected if isinstance(row.get("label_return_pct"), (int, float))]
        wins = [row for row in selected if _is_true_like(row.get("label_win"))]
        buckets[label] = {
            "count": len(selected),
            "win_rate_pct": (len(wins) / float(len(selected)) * 100.0) if selected else None,
            "avg_return_pct": _mean(returns),
        }
    return buckets


def main():
    parser = build_parser()
    args = parser.parse_args()
    overall_started_at = time.time()

    _print_step(1, 6, "Load dataset")
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

    _print_step(2, 6, "Build V4 features")
    df = build_features(df)
    df = augment_v4_features(df)
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

    fit_df, calibration_df, calibration_plan = prepare_calibration_split(train_df, args)
    model_train_df = fit_df if str(calibration_plan.get("status") or "") == "enabled" else train_df

    categorical_features, numeric_features = available_features(model_train_df)
    categorical_features, numeric_features = extend_v4_numeric_features(model_train_df, categorical_features, numeric_features)
    feature_cols = categorical_features + numeric_features

    _print_step(3, 6, "Train model backend")
    bundle = train_bundle(
        model_train_df,
        feature_cols,
        categorical_features,
        numeric_features,
        backend=args.backend,
        device=args.device,
    )
    print(f"[{LOG_PREFIX}] backend={bundle['backend']} | device={bundle['device']} | features={len(feature_cols)}", flush=True)

    _print_step(4, 6, "Fit calibration layer")
    calibration_info = dict(calibration_plan)
    calibration_info.update(
        {
            "method_requested": str(args.calibration_method),
            "days": int(args.calibration_days),
            "min_rows": int(args.calibration_min_rows),
            "min_train_days": int(args.calibration_min_train_days),
            "model_train_row_count": int(len(model_train_df)),
            "train_row_count_before_calibration": int(len(train_df)),
        }
    )
    if str(calibration_plan.get("status") or "") == "enabled":
        bundle, fitted_calibration_info = fit_probability_calibrator(
            bundle,
            calibration_df,
            method=args.calibration_method,
        )
        calibration_info.update(fitted_calibration_info)
        calibration_raw_matrix, calibration_classes = bundle_predict_proba_raw(bundle, calibration_df[feature_cols])
        calibration_raw_prob_maps = probability_maps_from_matrix(calibration_raw_matrix, calibration_classes)
        calibration_calibrated_matrix = apply_probability_calibrator(bundle, calibration_raw_matrix, calibration_classes)
        calibration_prob_maps = probability_maps_from_matrix(calibration_calibrated_matrix, calibration_classes)
        calibration_labels = calibration_df["entry_quality_label"].astype(str).tolist()
        calibration_info["calibration_split_probability_metrics_raw"] = summarize_probability_calibration(
            calibration_raw_prob_maps,
            calibration_labels,
        )
        calibration_info["calibration_split_probability_metrics_calibrated"] = summarize_probability_calibration(
            calibration_prob_maps,
            calibration_labels,
        )
    else:
        bundle["probability_calibrator"] = None

    X_holdout = holdout_df[feature_cols]
    y_holdout = holdout_df["entry_quality_label"].astype(str)
    raw_holdout_prob, classes = bundle_predict_proba_raw(bundle, X_holdout)
    holdout_prob = apply_probability_calibrator(bundle, raw_holdout_prob, classes)
    raw_prob_maps = probability_maps_from_matrix(raw_holdout_prob, classes)
    prob_maps = probability_maps_from_matrix(holdout_prob, classes)

    raw_argmax_predictions = [max(prob_map.items(), key=lambda item: item[1])[0] for prob_map in raw_prob_maps]
    raw_argmax_metrics = classification_metrics(y_holdout.tolist(), raw_argmax_predictions)
    argmax_predictions = [max(prob_map.items(), key=lambda item: item[1])[0] for prob_map in prob_maps]
    argmax_metrics = classification_metrics(y_holdout.tolist(), argmax_predictions)
    calibration_info["holdout_probability_metrics_raw"] = summarize_probability_calibration(raw_prob_maps, y_holdout.tolist())
    calibration_info["holdout_probability_metrics_calibrated"] = summarize_probability_calibration(prob_maps, y_holdout.tolist())

    _print_step(5, 6, "Scan threshold policy")
    best_policies, top_policies = optimize_policy_thresholds(holdout_df, prob_maps, args)
    premium_policy = best_policies.get("premium")
    standard_policy = best_policies.get("standard")
    watch_policy = best_policies.get("watch")
    if not isinstance(premium_policy, dict) or not isinstance(standard_policy, dict) or not isinstance(watch_policy, dict):
        raise ValueError("Unable to derive Premium, Standard, and Watch V4 holdout policies from the threshold grid")

    premium_predictions = [
        threshold_label(prob_map, premium_policy["entry_threshold"], premium_policy["avoid_threshold"])
        for prob_map in prob_maps
    ]
    standard_predictions = [
        threshold_label(prob_map, standard_policy["entry_threshold"], standard_policy["avoid_threshold"])
        for prob_map in prob_maps
    ]
    watch_predictions = [
        threshold_label(prob_map, watch_policy["entry_threshold"], watch_policy["avoid_threshold"])
        for prob_map in prob_maps
    ]
    premium_policy_metrics = classification_metrics(y_holdout.tolist(), premium_predictions)
    standard_policy_metrics = classification_metrics(y_holdout.tolist(), standard_predictions)
    watch_policy_metrics = classification_metrics(y_holdout.tolist(), watch_predictions)
    strategy_specific_policies = optimize_strategy_specific_policies(holdout_df, prob_maps, args)
    strategy_row_counts = summarize_strategy_row_counts(
        usable_df=usable_df,
        train_df=train_df,
        model_train_df=model_train_df,
        calibration_df=calibration_df,
        holdout_df=holdout_df,
        args=args,
    )
    for strategy_name, strategy_summary in strategy_row_counts.items():
        print(
            f"[{LOG_PREFIX}] strategy={strategy_name} | usable={strategy_summary['usable_rows']} | "
            f"holdout={strategy_summary['holdout_rows']} | "
            f"strategy_policy_eligible={str(strategy_summary['strategy_policy_eligible']).lower()}",
            flush=True,
        )

    holdout_rows = build_holdout_prediction_rows(
        holdout_df,
        prob_maps,
        premium_policy,
        standard_policy,
        watch_policy,
        raw_prob_maps=raw_prob_maps,
    )

    entry_probabilities = sorted(
        [float(row["prob_entry"]) for row in holdout_rows if str(row.get("predicted_label_premium") or "") == "entry"]
    )
    premium_threshold = None
    if entry_probabilities:
        pivot = max(int(len(entry_probabilities) * 0.75) - 1, 0)
        premium_threshold = float(entry_probabilities[pivot])
        premium_threshold = max(premium_threshold, float(premium_policy["entry_threshold"]) + 0.05)
        premium_threshold = min(premium_threshold, 0.95)

    bundle["trained_at"] = datetime.utcnow().isoformat() + "Z"
    bundle["model_type"] = MODEL_TYPE
    bundle["model_version"] = MODEL_VERSION
    bundle["metadata"] = {
        "model_version": MODEL_VERSION,
        "feature_schema_version": MODEL_VERSION,
        "label_schema_version": MODEL_VERSION,
        "policy_schema_version": MODEL_VERSION,
        "input_path": str(input_path),
        "groups": groups,
        "strategies": strategies,
        "backend_requested": str(args.backend),
        "backend_used": str(bundle["backend"]),
        "device_requested": str(args.device),
        "device_used": str(bundle["device"]),
        "recommended_entry_threshold": float(premium_policy["entry_threshold"]),
        "recommended_avoid_threshold": float(premium_policy["avoid_threshold"]),
        "recommended_standard_entry_threshold": float(standard_policy["entry_threshold"]),
        "recommended_standard_avoid_threshold": float(standard_policy["avoid_threshold"]),
        "recommended_watch_entry_threshold": float(watch_policy["entry_threshold"]),
        "recommended_watch_avoid_threshold": float(watch_policy["avoid_threshold"]),
        "recommended_premium_entry_threshold": premium_threshold,
        "recommended_premium_policy": premium_policy,
        "recommended_standard_policy": standard_policy,
        "recommended_watch_policy": watch_policy,
        "train_label_counts": train_label_counts,
        "holdout_label_counts": holdout_label_counts,
        "overall_label_counts": overall_label_counts,
        "calibration": calibration_info,
        "policy_optimization": {
            "probability_source": "calibrated" if isinstance(bundle.get("probability_calibrator"), dict) and bool(bundle["probability_calibrator"].get("enabled")) else "raw",
            "calibration_target_weight": float(args.policy_calibration_target_weight),
            "calibration_avoid_weight": float(args.policy_calibration_avoid_weight),
            "calibration_overconfidence_penalty_weight": float(args.policy_calibration_overconfidence_penalty_weight),
            "strategy_policy_enable": bool(args.strategy_policy_enable),
            "strategy_policy_min_holdout_rows": int(args.strategy_policy_min_holdout_rows),
        },
        "strategy_row_counts": strategy_row_counts,
        "strategy_specific_policies": strategy_specific_policies,
    }

    _print_step(6, 6, "Write artifacts")
    artifact_path = output_dir / f"{ARTIFACT_PREFIX}_model.joblib"
    metrics_path = output_dir / f"{ARTIFACT_PREFIX}_metrics.json"
    predictions_path = output_dir / f"{ARTIFACT_PREFIX}_holdout_predictions.jsonl"
    policy_path = output_dir / f"{ARTIFACT_PREFIX}_policy_grid.json"

    metrics_payload = {
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "feature_schema_version": MODEL_VERSION,
        "label_schema_version": MODEL_VERSION,
        "policy_schema_version": MODEL_VERSION,
        "trained_at": bundle["trained_at"],
        "input_path": str(input_path),
        "artifact_path": str(artifact_path),
        "row_count_total": int(len(df)),
        "row_count_usable": int(len(usable_df)),
        "row_count_train": int(len(train_df)),
        "row_count_holdout": int(len(holdout_df)),
        "groups": groups,
        "strategies": strategies,
        "backend_requested": str(args.backend),
        "backend_used": str(bundle["backend"]),
        "device_requested": str(args.device),
        "device_used": str(bundle["device"]),
        "overall_label_counts": overall_label_counts,
        "train_label_counts": train_label_counts,
        "holdout_label_counts": holdout_label_counts,
        "row_count_model_train": int(len(model_train_df)),
        "row_count_calibration": int(len(calibration_df)),
        "raw_argmax_metrics": raw_argmax_metrics,
        "argmax_metrics": argmax_metrics,
        "premium_policy_metrics": premium_policy_metrics,
        "standard_policy_metrics": standard_policy_metrics,
        "watch_policy_metrics": watch_policy_metrics,
        "calibration": calibration_info,
        "policy_optimization": bundle["metadata"].get("policy_optimization"),
        "strategy_row_counts": strategy_row_counts,
        "strategy_specific_policies": strategy_specific_policies,
        "recommended_policy": premium_policy,
        "recommended_premium_policy": premium_policy,
        "recommended_standard_policy": standard_policy,
        "recommended_watch_policy": watch_policy,
        "top_policy_candidates": top_policies.get("premium", []),
        "top_premium_policy_candidates": top_policies.get("premium", []),
        "top_standard_policy_candidates": top_policies.get("standard", []),
        "top_watch_policy_candidates": top_policies.get("watch", []),
        "recommended_premium_entry_threshold": premium_threshold,
        "target_alerts_per_day": float(args.target_alerts_per_day),
        "holdout_realized_by_argmax_prediction": realized_summary(holdout_rows, "predicted_label_argmax"),
        "holdout_realized_by_policy_prediction": realized_summary(holdout_rows, "predicted_label_premium"),
        "holdout_realized_by_premium_policy_prediction": realized_summary(holdout_rows, "predicted_label_premium"),
        "holdout_realized_by_standard_policy_prediction": realized_summary(holdout_rows, "predicted_label_standard"),
        "holdout_realized_by_watch_policy_prediction": realized_summary(holdout_rows, "predicted_label_watch"),
        "feature_columns": feature_cols,
    }

    joblib.dump(bundle, artifact_path)
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    policy_path.write_text(json.dumps(top_policies, ensure_ascii=False, indent=2), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8") as fh:
        for row in holdout_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[{LOG_PREFIX}] done | elapsed={_format_duration(time.time() - overall_started_at)}", flush=True)

    print(
        json.dumps(
            {
                "artifact_path": str(artifact_path),
                "metrics_path": str(metrics_path),
                "policy_path": str(policy_path),
                "predictions_path": str(predictions_path),
                "backend_used": bundle["backend"],
                "device_used": bundle["device"],
                "recommended_policy": premium_policy,
                "recommended_premium_policy": premium_policy,
                "recommended_standard_policy": standard_policy,
                "recommended_watch_policy": watch_policy,
                "recommended_premium_entry_threshold": premium_threshold,
                "argmax_metrics": {
                    "accuracy": argmax_metrics.get("accuracy"),
                    "balanced_accuracy": argmax_metrics.get("balanced_accuracy"),
                    "macro_f1": argmax_metrics.get("macro_f1"),
                },
                "premium_policy_metrics": {
                    "accuracy": premium_policy_metrics.get("accuracy"),
                    "balanced_accuracy": premium_policy_metrics.get("balanced_accuracy"),
                    "macro_f1": premium_policy_metrics.get("macro_f1"),
                },
                "standard_policy_metrics": {
                    "accuracy": standard_policy_metrics.get("accuracy"),
                    "balanced_accuracy": standard_policy_metrics.get("balanced_accuracy"),
                    "macro_f1": standard_policy_metrics.get("macro_f1"),
                },
                "watch_policy_metrics": {
                    "accuracy": watch_policy_metrics.get("accuracy"),
                    "balanced_accuracy": watch_policy_metrics.get("balanced_accuracy"),
                    "macro_f1": watch_policy_metrics.get("macro_f1"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
