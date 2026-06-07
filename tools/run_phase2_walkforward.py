import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd


TIER_RANK = {
    "S": 5,
    "A": 4,
    "B": 3,
    "C": 2,
    "D": 1,
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Phase 2 walk-forward optimizer and summary builder for Phase 1 research dataset"
    )
    parser.add_argument(
        "--input-path",
        default="",
        help="Path to phase1_candidates.csv or phase1_candidates.jsonl (default: .data/research/phase1/phase1_candidates.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to write phase 2 outputs (default: alongside input under phase2)",
    )
    parser.add_argument("--train-days", type=int, default=180, help="Train window size in days")
    parser.add_argument("--valid-days", type=int, default=60, help="Validation window size in days")
    parser.add_argument("--step-days", type=int, default=30, help="Window step size in days")
    parser.add_argument(
        "--groups",
        default="primary,trend_radar,daily",
        help="Candidate groups to include, e.g. primary,daily",
    )
    parser.add_argument(
        "--strategies",
        default="",
        help="Optional comma-separated strategy filter",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=8,
        help="Minimum filled candidates required for a config to be considered",
    )
    parser.add_argument(
        "--objective",
        choices=("return", "stability", "win_rate"),
        default="stability",
        help="Optimization objective for choosing thresholds per window",
    )
    parser.add_argument(
        "--require-positive-expectancy",
        action="store_true",
        help="Keep only configs with positive average return during train",
    )
    return parser


def parse_csv_list(text):
    values = [part.strip() for part in str(text or "").split(",")]
    return [part for part in values if part]


def default_input_path(root):
    return root / ".data" / "research" / "phase1" / "phase1_candidates.csv"


def resolve_input_path(root, raw_path):
    raw = str(raw_path or "").strip()
    if not raw:
        return default_input_path(root)
    path = Path(raw)
    return path if path.is_absolute() else (root / path)


def resolve_output_dir(root, input_path, raw_output_dir):
    raw = str(raw_output_dir or "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (root / path)
    return input_path.resolve().parents[0] / "phase2"


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


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _normalize_bool_series(series):
    return series.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() in {"1", "true", "yes", "y"}
    )


def load_dataset(input_path):
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(input_path)
    if df.empty:
        return df
    if "checkpoint_at" in df.columns:
        df["checkpoint_at"] = pd.to_datetime(df["checkpoint_at"], errors="coerce")
    for col in (
        "confidence",
        "score",
        "alert_tier_score",
        "label_return_pct",
        "label_mfe_pct",
        "label_mae_pct",
        "label_mfe_r",
        "label_mae_r",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("label_filled", "label_win"):
        if col in df.columns:
            df[col] = _normalize_bool_series(df[col])
    if "alert_tier" in df.columns:
        df["tier_rank"] = df["alert_tier"].map(lambda value: TIER_RANK.get(str(value or "").strip().upper(), 0))
    else:
        df["tier_rank"] = 0
    if "market_regime" in df.columns:
        df["market_regime"] = df["market_regime"].fillna("UNKNOWN").astype(str)
    else:
        df["market_regime"] = "UNKNOWN"
    return df


def filter_dataset(df, *, groups, strategies):
    out = df.copy()
    if groups and "candidate_group" in out.columns:
        out = out[out["candidate_group"].astype(str).isin(groups)]
    if strategies and "strategy" in out.columns:
        allowed = {str(value).strip().upper() for value in strategies}
        out = out[out["strategy"].astype(str).str.upper().isin(allowed)]
    out = out.dropna(subset=["checkpoint_at"])
    return out.sort_values("checkpoint_at").reset_index(drop=True)


def aggregate_rows(df):
    rows = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    selected = len(rows)
    filled = int(rows["label_filled"].sum()) if "label_filled" in rows.columns and not rows.empty else 0
    usable = rows[rows["label_filled"] == True] if "label_filled" in rows.columns else rows.iloc[0:0]
    wins = int((usable["label_win"] == True).sum()) if not usable.empty and "label_win" in usable.columns else 0
    losses = int((usable["label_win"] == False).sum()) if not usable.empty and "label_win" in usable.columns else 0
    avg_return = usable["label_return_pct"].mean() if not usable.empty and "label_return_pct" in usable.columns else None
    avg_mfe_r = usable["label_mfe_r"].mean() if not usable.empty and "label_mfe_r" in usable.columns else None
    avg_mae_r = usable["label_mae_r"].mean() if not usable.empty and "label_mae_r" in usable.columns else None
    return {
        "selected": int(selected),
        "filled": int(filled),
        "wins": int(wins),
        "losses": int(losses),
        "fill_rate_pct": (float(filled) / float(selected) * 100.0) if selected else 0.0,
        "win_rate_pct": (float(wins) / float(filled) * 100.0) if filled else 0.0,
        "avg_return_pct": float(avg_return) if isinstance(avg_return, (int, float)) and math.isfinite(float(avg_return)) else None,
        "avg_mfe_r": float(avg_mfe_r) if isinstance(avg_mfe_r, (int, float)) and math.isfinite(float(avg_mfe_r)) else None,
        "avg_mae_r": float(avg_mae_r) if isinstance(avg_mae_r, (int, float)) and math.isfinite(float(avg_mae_r)) else None,
    }


def objective_value(metrics, objective):
    filled = int(metrics.get("filled") or 0)
    avg_return = _safe_float(metrics.get("avg_return_pct"), None)
    win_rate = _safe_float(metrics.get("win_rate_pct"), None)
    if filled <= 0:
        return None
    if objective == "return":
        if avg_return is None:
            return None
        return avg_return * math.sqrt(float(filled))
    if objective == "win_rate":
        if win_rate is None:
            return None
        return (win_rate / 100.0) * math.sqrt(float(filled))
    if avg_return is None or win_rate is None:
        return None
    return avg_return * (0.5 + (win_rate / 100.0)) * math.sqrt(float(filled))


def threshold_grid(df):
    conf_values = sorted({int(value) for value in df["confidence"].dropna().tolist() if _safe_float(value, None) is not None})
    score_values = sorted({int(value) for value in df["score"].dropna().tolist() if _safe_float(value, None) is not None})
    tier_score_values = sorted(
        {int(value) for value in df["alert_tier_score"].dropna().tolist() if _safe_float(value, None) is not None}
    )
    conf_grid = [value for value in (60, 65, 70, 75, 80, 85) if not conf_values or value <= max(conf_values)]
    score_grid = [value for value in (60, 70, 75, 80, 85, 90) if not score_values or value <= max(score_values)]
    tier_score_grid = [value for value in (50, 60, 70, 80, 85) if not tier_score_values or value <= max(tier_score_values)]
    tier_grid = ["D", "C", "B", "A"]
    return conf_grid or [0], score_grid or [0], tier_score_grid or [0], tier_grid


def apply_config(df, config):
    out = df.copy()
    min_conf = _safe_float(config.get("min_confidence"), None)
    min_score = _safe_float(config.get("min_score"), None)
    min_tier_score = _safe_float(config.get("min_tier_score"), None)
    min_tier_rank = int(config.get("min_tier_rank") or 0)
    if min_conf is not None and "confidence" in out.columns:
        out = out[out["confidence"].fillna(-1e9) >= float(min_conf)]
    if min_score is not None and "score" in out.columns:
        out = out[out["score"].fillna(-1e9) >= float(min_score)]
    if min_tier_score is not None and "alert_tier_score" in out.columns:
        out = out[out["alert_tier_score"].fillna(-1e9) >= float(min_tier_score)]
    if min_tier_rank > 0 and "tier_rank" in out.columns:
        out = out[out["tier_rank"].fillna(0) >= int(min_tier_rank)]
    return out


def choose_best_config(train_df, *, objective, min_trades, require_positive_expectancy):
    conf_grid, score_grid, tier_score_grid, tier_grid = threshold_grid(train_df)
    best = None
    debug = {
        "configs_evaluated": 0,
        "configs_rejected_min_trades": 0,
        "configs_rejected_positive_expectancy": 0,
        "configs_rejected_objective": 0,
        "configs_passing_min_trades": 0,
        "configs_passing_positive_expectancy": 0,
        "max_filled_any_config": 0,
        "max_avg_return_pct_any_config": None,
    }
    for min_conf in conf_grid:
        for min_score in score_grid:
            for min_tier_score in tier_score_grid:
                for tier_name in tier_grid:
                    debug["configs_evaluated"] += 1
                    config = {
                        "min_confidence": int(min_conf),
                        "min_score": int(min_score),
                        "min_tier_score": int(min_tier_score),
                        "min_tier": tier_name,
                        "min_tier_rank": int(TIER_RANK.get(tier_name, 0)),
                    }
                    selected = apply_config(train_df, config)
                    metrics = aggregate_rows(selected)
                    filled = int(metrics.get("filled") or 0)
                    avg_return = _safe_float(metrics.get("avg_return_pct"), None)
                    debug["max_filled_any_config"] = max(debug["max_filled_any_config"], filled)
                    if avg_return is not None:
                        prev_max_avg_return = _safe_float(debug["max_avg_return_pct_any_config"], None)
                        if prev_max_avg_return is None or avg_return > prev_max_avg_return:
                            debug["max_avg_return_pct_any_config"] = float(avg_return)
                    if filled < int(min_trades):
                        debug["configs_rejected_min_trades"] += 1
                        continue
                    debug["configs_passing_min_trades"] += 1
                    if require_positive_expectancy and (avg_return is None or avg_return <= 0.0):
                        debug["configs_rejected_positive_expectancy"] += 1
                        continue
                    debug["configs_passing_positive_expectancy"] += 1
                    score = objective_value(metrics, objective)
                    if score is None:
                        debug["configs_rejected_objective"] += 1
                        continue
                    candidate = {
                        "config": config,
                        "train_metrics": metrics,
                        "objective_value": float(score),
                    }
                    if best is None:
                        best = candidate
                        continue
                    prev_filled = int((best.get("train_metrics") or {}).get("filled") or 0)
                    prev_avg_return = _safe_float((best.get("train_metrics") or {}).get("avg_return_pct"), -1e9)
                    if (
                        float(score) > float(best.get("objective_value") or -1e18)
                        or (
                            math.isclose(float(score), float(best.get("objective_value") or -1e18))
                            and filled > prev_filled
                        )
                        or (
                            math.isclose(float(score), float(best.get("objective_value") or -1e18))
                            and filled == prev_filled
                            and (avg_return or -1e9) > prev_avg_return
                        )
                    ):
                        best = candidate
    if best is not None:
        debug["status"] = "selected"
        debug["failure_reason"] = None
    elif debug["configs_passing_min_trades"] == 0:
        debug["status"] = "blocked"
        debug["failure_reason"] = "min_trades"
    elif require_positive_expectancy and debug["configs_passing_positive_expectancy"] == 0:
        debug["status"] = "blocked"
        debug["failure_reason"] = "positive_expectancy"
    elif debug["configs_rejected_objective"] > 0:
        debug["status"] = "blocked"
        debug["failure_reason"] = "objective"
    else:
        debug["status"] = "blocked"
        debug["failure_reason"] = "no_valid_config"
    return best, debug


def window_slices(df, train_days, valid_days, step_days):
    if df.empty:
        return []
    start = pd.Timestamp(df["checkpoint_at"].min()).normalize()
    end = pd.Timestamp(df["checkpoint_at"].max()).normalize()
    train_delta = pd.Timedelta(days=int(train_days))
    valid_delta = pd.Timedelta(days=int(valid_days))
    step_delta = pd.Timedelta(days=int(step_days))
    windows = []
    cursor = start
    while cursor + train_delta + valid_delta <= end + pd.Timedelta(days=1):
        train_start = cursor
        train_end = cursor + train_delta
        valid_end = train_end + valid_delta
        train_df = df[(df["checkpoint_at"] >= train_start) & (df["checkpoint_at"] < train_end)]
        valid_df = df[(df["checkpoint_at"] >= train_end) & (df["checkpoint_at"] < valid_end)]
        if not train_df.empty and not valid_df.empty:
            windows.append(
                {
                    "train_start": train_start,
                    "train_end": train_end,
                    "valid_start": train_end,
                    "valid_end": valid_end,
                    "train_df": train_df,
                    "valid_df": valid_df,
                }
            )
        cursor = cursor + step_delta
    return windows


def summarize_group(df, keys):
    if df.empty:
        return []
    group_cols = [key for key in keys if key in df.columns]
    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for key, sub in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        record = {group_cols[idx]: key_values[idx] for idx in range(len(group_cols))}
        record.update(aggregate_rows(sub))
        rows.append(record)
    rows.sort(
        key=lambda row: (
            _safe_float(row.get("avg_return_pct"), -1e18),
            _safe_float(row.get("win_rate_pct"), -1e18),
            _safe_int(row.get("filled"), 0),
        ),
        reverse=True,
    )
    return rows


def build_dataset_summary(df):
    return {
        "overall": aggregate_rows(df),
        "by_strategy": summarize_group(df, ["strategy"]),
        "by_symbol": summarize_group(df, ["symbol"]),
        "by_regime": summarize_group(df, ["market_regime"]),
        "by_strategy_regime": summarize_group(df, ["strategy", "market_regime"]),
        "by_group": summarize_group(df, ["candidate_group"]),
    }


def run_walkforward(df, *, train_days, valid_days, step_days, min_trades, objective, require_positive_expectancy):
    results = []
    debug_rows = []
    strategies = sorted({str(value).strip().upper() for value in df["strategy"].dropna().tolist()})
    for strategy in strategies:
        strategy_df = df[df["strategy"].astype(str).str.upper() == strategy].copy()
        windows = window_slices(strategy_df, train_days, valid_days, step_days)
        strategy_results = []
        for window in windows:
            best, debug = choose_best_config(
                window["train_df"],
                objective=objective,
                min_trades=min_trades,
                require_positive_expectancy=require_positive_expectancy,
            )
            debug_row = {
                "strategy": strategy,
                "train_start": window["train_start"].isoformat(),
                "train_end": window["train_end"].isoformat(),
                "valid_start": window["valid_start"].isoformat(),
                "valid_end": window["valid_end"].isoformat(),
                "train_rows": int(len(window["train_df"])),
                "valid_rows": int(len(window["valid_df"])),
                "train_metrics": aggregate_rows(window["train_df"]),
                "valid_metrics": aggregate_rows(window["valid_df"]),
                "status": debug.get("status"),
                "failure_reason": debug.get("failure_reason"),
                "configs_evaluated": int(debug.get("configs_evaluated") or 0),
                "configs_rejected_min_trades": int(debug.get("configs_rejected_min_trades") or 0),
                "configs_rejected_positive_expectancy": int(debug.get("configs_rejected_positive_expectancy") or 0),
                "configs_rejected_objective": int(debug.get("configs_rejected_objective") or 0),
                "configs_passing_min_trades": int(debug.get("configs_passing_min_trades") or 0),
                "configs_passing_positive_expectancy": int(debug.get("configs_passing_positive_expectancy") or 0),
                "max_filled_any_config": int(debug.get("max_filled_any_config") or 0),
                "max_avg_return_pct_any_config": _safe_float(debug.get("max_avg_return_pct_any_config"), None),
                "selected_config": (best or {}).get("config"),
            }
            debug_rows.append(debug_row)
            if not best:
                continue
            valid_selected = apply_config(window["valid_df"], best["config"])
            valid_metrics = aggregate_rows(valid_selected)
            strategy_results.append(
                {
                    "strategy": strategy,
                    "train_start": window["train_start"].isoformat(),
                    "train_end": window["train_end"].isoformat(),
                    "valid_start": window["valid_start"].isoformat(),
                    "valid_end": window["valid_end"].isoformat(),
                    "config": best["config"],
                    "train_metrics": best["train_metrics"],
                    "valid_metrics": valid_metrics,
                    "objective_value": float(best["objective_value"]),
                }
            )
        results.extend(strategy_results)
    return results, debug_rows


def build_walkforward_debug_summary(debug_rows):
    summary = {
        "overall": {
            "windows_total": int(len(debug_rows)),
            "windows_selected": 0,
            "windows_blocked": 0,
            "blocked_min_trades": 0,
            "blocked_positive_expectancy": 0,
            "blocked_objective": 0,
            "blocked_no_valid_config": 0,
        },
        "by_strategy": [],
    }
    if not debug_rows:
        return summary

    strategy_rows = {}
    for row in debug_rows:
        strategy = str(row.get("strategy") or "UNKNOWN").strip().upper()
        bucket = strategy_rows.setdefault(strategy, [])
        bucket.append(row)

        if row.get("status") == "selected":
            summary["overall"]["windows_selected"] += 1
        else:
            summary["overall"]["windows_blocked"] += 1
            reason = str(row.get("failure_reason") or "no_valid_config").strip()
            key = f"blocked_{reason}"
            if key in summary["overall"]:
                summary["overall"][key] += 1

    for strategy, rows in sorted(strategy_rows.items()):
        record = {
            "strategy": strategy,
            "windows_total": int(len(rows)),
            "windows_selected": 0,
            "windows_blocked": 0,
            "blocked_min_trades": 0,
            "blocked_positive_expectancy": 0,
            "blocked_objective": 0,
            "blocked_no_valid_config": 0,
        }
        for row in rows:
            if row.get("status") == "selected":
                record["windows_selected"] += 1
                continue
            record["windows_blocked"] += 1
            reason = str(row.get("failure_reason") or "no_valid_config").strip()
            key = f"blocked_{reason}"
            if key in record:
                record[key] += 1
        summary["by_strategy"].append(record)
    return summary


def build_recommendations(window_results):
    by_strategy = {}
    for row in window_results:
        strategy = str(row.get("strategy") or "UNKNOWN").strip().upper()
        bucket = by_strategy.setdefault(strategy, [])
        bucket.append(row)

    recommendations = []
    for strategy, rows in sorted(by_strategy.items()):
        positive_rows = [
            row
            for row in rows
            if _safe_float((row.get("valid_metrics") or {}).get("avg_return_pct"), None) is not None
            and _safe_float((row.get("valid_metrics") or {}).get("avg_return_pct"), 0.0) > 0.0
        ]
        source_rows = positive_rows or rows
        if not source_rows:
            continue
        conf_values = [int((row.get("config") or {}).get("min_confidence") or 0) for row in source_rows]
        score_values = [int((row.get("config") or {}).get("min_score") or 0) for row in source_rows]
        tier_score_values = [int((row.get("config") or {}).get("min_tier_score") or 0) for row in source_rows]
        tier_names = [str((row.get("config") or {}).get("min_tier") or "D") for row in source_rows]
        valid_returns = [
            _safe_float((row.get("valid_metrics") or {}).get("avg_return_pct"), None)
            for row in source_rows
            if _safe_float((row.get("valid_metrics") or {}).get("avg_return_pct"), None) is not None
        ]
        valid_win_rates = [
            _safe_float((row.get("valid_metrics") or {}).get("win_rate_pct"), None)
            for row in source_rows
            if _safe_float((row.get("valid_metrics") or {}).get("win_rate_pct"), None) is not None
        ]
        valid_filled = [int((row.get("valid_metrics") or {}).get("filled") or 0) for row in source_rows]
        recommendation = {
            "strategy": strategy,
            "windows_evaluated": len(rows),
            "positive_validation_windows": len(positive_rows),
            "positive_window_rate_pct": (float(len(positive_rows)) / float(len(rows)) * 100.0) if rows else 0.0,
            "suggested_min_confidence": int(round(sum(conf_values) / len(conf_values))) if conf_values else None,
            "suggested_min_score": int(round(sum(score_values) / len(score_values))) if score_values else None,
            "suggested_min_tier_score": int(round(sum(tier_score_values) / len(tier_score_values))) if tier_score_values else None,
            "suggested_min_tier": Counter(tier_names).most_common(1)[0][0] if tier_names else None,
            "avg_valid_return_pct": (sum(valid_returns) / len(valid_returns)) if valid_returns else None,
            "avg_valid_win_rate_pct": (sum(valid_win_rates) / len(valid_win_rates)) if valid_win_rates else None,
            "avg_valid_filled": (sum(valid_filled) / len(valid_filled)) if valid_filled else 0.0,
        }
        recommendations.append(recommendation)
    recommendations.sort(
        key=lambda row: (
            _safe_float(row.get("avg_valid_return_pct"), -1e18),
            _safe_float(row.get("avg_valid_win_rate_pct"), -1e18),
            _safe_float(row.get("positive_window_rate_pct"), -1e18),
        ),
        reverse=True,
    )
    return recommendations


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = build_parser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    input_path = resolve_input_path(root, args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Phase 1 dataset not found: {input_path}")
    output_dir = resolve_output_dir(root, input_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(input_path)
    groups = parse_csv_list(args.groups)
    strategies = [value.strip().upper() for value in parse_csv_list(args.strategies)]
    dataset = filter_dataset(dataset, groups=groups, strategies=strategies)
    summary = build_dataset_summary(dataset)
    walkforward_rows, debug_rows = run_walkforward(
        dataset,
        train_days=args.train_days,
        valid_days=args.valid_days,
        step_days=args.step_days,
        min_trades=args.min_trades,
        objective=args.objective,
        require_positive_expectancy=bool(args.require_positive_expectancy),
    )
    recommendations = build_recommendations(walkforward_rows)
    debug_summary = build_walkforward_debug_summary(debug_rows)

    write_json(output_dir / "phase2_dataset_summary.json", summary)
    write_jsonl(output_dir / "phase2_walkforward_windows.jsonl", walkforward_rows)
    pd.DataFrame(walkforward_rows).to_csv(output_dir / "phase2_walkforward_windows.csv", index=False)
    write_json(output_dir / "phase2_walkforward_debug_summary.json", debug_summary)
    write_jsonl(output_dir / "phase2_walkforward_debug_windows.jsonl", debug_rows)
    pd.DataFrame(debug_rows).to_csv(output_dir / "phase2_walkforward_debug_windows.csv", index=False)
    write_json(output_dir / "phase2_threshold_recommendations.json", {"recommendations": recommendations})
    if recommendations:
        pd.DataFrame(recommendations).to_csv(output_dir / "phase2_threshold_recommendations.csv", index=False)
    else:
        pd.DataFrame(columns=["strategy", "suggested_min_confidence"]).to_csv(
            output_dir / "phase2_threshold_recommendations.csv",
            index=False,
        )

    payload = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "row_count": int(len(dataset)),
        "groups": groups,
        "strategies": strategies,
        "summary_file": str(output_dir / "phase2_dataset_summary.json"),
        "walkforward_windows_file": str(output_dir / "phase2_walkforward_windows.jsonl"),
        "walkforward_debug_summary_file": str(output_dir / "phase2_walkforward_debug_summary.json"),
        "walkforward_debug_windows_file": str(output_dir / "phase2_walkforward_debug_windows.jsonl"),
        "recommendations_file": str(output_dir / "phase2_threshold_recommendations.json"),
        "debug_overall": debug_summary.get("overall", {}),
        "recommendation_count": len(recommendations),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
