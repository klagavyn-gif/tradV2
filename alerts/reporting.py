import csv
import hashlib
import html
import json
import math
import os
import re
import tempfile
from calendar import monthrange
from collections import Counter
from datetime import datetime

import pandas as pd


def alert_history_csv_fieldnames():
    return [
        "alert_id",
        "timestamp",
        "strategy",
        "symbol",
        "signal",
        "timeframe",
        "evaluation_window_bars",
        "alert_tier",
        "alert_tier_score",
        "tier_action",
        "alert_mode",
        "confidence",
        "score",
        "daily_pick",
        "source_count",
        "source_label",
        "strategy_label",
        "alert_intent",
        "alert_intent_reason",
        "ai_dispatch_label",
        "ai_dispatch_bucket",
        "ai_dispatch_reason",
        "ai_prob_win",
        "ai_expected_return_pct",
        "ai_rank_adjustment",
        "ai_runtime_status",
        "ai_runtime_reason",
        "entry_ai_label",
        "entry_ai_bucket",
        "entry_ai_reason",
        "entry_ai_policy_mode",
        "entry_ai_policy_tier",
        "entry_ai_premium_label",
        "entry_ai_standard_label",
        "entry_ai_watch_label",
        "entry_ai_strategy_policy",
        "entry_ai_prob_entry",
        "entry_ai_prob_watch",
        "entry_ai_prob_avoid",
        "entry_ai_premium_entry_threshold",
        "entry_ai_premium_avoid_threshold",
        "entry_ai_standard_entry_threshold",
        "entry_ai_standard_avoid_threshold",
        "entry_ai_watch_entry_threshold",
        "entry_ai_watch_avoid_threshold",
        "entry_ai_model_type",
        "entry_ai_model_version",
        "entry_ai_model_trained_at",
        "entry_ai_feature_schema_version",
        "entry_ai_label_schema_version",
        "entry_ai_policy_schema_version",
        "entry_ai_rank_adjustment",
        "entry_ai_runtime_status",
        "entry_ai_runtime_reason",
        "entry_ai_runtime_threshold_adjustment",
        "entry_ai_runtime_base_min_confidence",
        "entry_ai_runtime_min_confidence",
        "entry_ai_runtime_threshold_reason",
        "short_trade_label",
        "short_trade_bucket",
        "short_trade_reason",
        "short_trade_score_adjustment",
        "short_trade_regime_aligned",
        "market_regime",
        "market_trend_bias",
        "symbol_regime",
        "side_bias",
        "regime_confidence",
        "regime_volatility_pct",
        "profile_runtime_threshold_applied",
        "profile_runtime_threshold_reason",
        "profile_runtime_market_regime",
        "profile_runtime_symbol_regime",
        "profile_runtime_side_bias",
        "profile_runtime_regime_alignment",
        "profile_runtime_freshness_bucket",
        "profile_runtime_bars_since_signal",
        "profile_runtime_min_confidence",
        "profile_runtime_min_score",
        "profile_runtime_min_win_rate_pct",
        "profile_runtime_min_expectancy_rr",
        "profile_runtime_min_trades",
        "profile_runtime_min_source_count",
        "profile_runtime_min_robustness_score",
        "sltp_live_label",
        "sltp_live_bucket",
        "sltp_live_reason",
        "sltp_live_score_adjustment",
        "sltp_live_entry_gap_pct",
        "sltp_live_stop_risk_pct",
        "sltp_live_target_reward_pct",
        "sltp_live_rr_ratio",
        "entry_price",
        "stop_loss",
        "take_profit",
        "risk_reward",
        "detected_pattern",
        "forecast_direction",
        "forecast_score",
        "plan_reason",
        "bars_since_signal",
        "timeframe_minutes",
        "signal_timestamp",
        "analysis_generated_at",
        "telegram_sent_at",
        "analysis_latency_seconds",
        "analysis_to_send_seconds",
        "signal_latency_seconds",
        "signal_age_minutes_at_analysis",
        "signal_age_minutes_at_send",
        "dispatch_status_label",
        "dispatch_status_reason_group",
        "dispatch_status_reason_detail",
        "entry_window_max_distance_pct",
        "entry_window_max_distance_r",
        "max_chase_price",
        "red_to_green_quality_score",
        "green_flip_reclaim",
        "min_confidence",
        "dynamic_min_confidence",
        "backtest_win_rate_pct",
        "backtest_expectancy_rr",
        "backtest_trades",
        "cache_key",
        "message_plain",
    ]


def _safe_float(value, default=None):
    try:
        number = float(value)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


def _safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def _realized_entry_price(plan, signal, *, pick_plan_value):
    signal_text = str(signal or "").strip().upper()
    if signal_text == "SELL":
        # For short/exit signals the trade is entered at the current price, not
        # the original long entry reference stored on the plan. Using the wrong
        # entry puts the stop_loss on the wrong side and corrupts realized
        # win/loss classification.
        return pick_plan_value(plan, ["current_price", "price", "entry_price"])
    return pick_plan_value(plan, ["entry_price", "current_price", "price"])


def _alert_timestamp_value(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _alert_id_value(row):
    if not isinstance(row, dict):
        return None
    existing = str(row.get("alert_id") or "").strip()
    if existing:
        return existing
    basis = "|".join(
        [
            str(row.get("timestamp") or "").strip(),
            str(row.get("strategy") or "").strip().upper(),
            str(row.get("symbol") or "").strip().upper(),
            str(row.get("signal") or "").strip().upper(),
            str(row.get("cache_key") or "").strip(),
            str(row.get("message_plain") or "").strip(),
        ]
    )
    if not basis.strip("|"):
        return None
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def _candidate_timeframe(candidate, *, config):
    plan = (candidate or {}).get("plan")
    for value in (
        (plan or {}).get("interval") if isinstance(plan, dict) else None,
        (plan or {}).get("timeframe") if isinstance(plan, dict) else None,
        (candidate or {}).get("interval"),
        getattr(config, "TELEGRAM_ALERT_REALIZED_INTERVAL", "15m"),
    ):
        text = str(value or "").strip().lower()
        if text:
            return text
    return "15m"


def _candidate_evaluation_window_bars(candidate, *, config):
    default_bars = _safe_int(getattr(config, "TELEGRAM_ALERT_REALIZED_MAX_HOLD_BARS", 64), 64)
    if default_bars is None or default_bars < 1:
        default_bars = 64
    plan = (candidate or {}).get("plan")
    strategy = str((candidate or {}).get("strategy") or "").strip().upper()
    if isinstance(plan, dict):
        for key in ("max_forward_bars", "time_stop_bars", "holding_window_bars", "max_hold_bars"):
            candidate_value = _safe_int(plan.get(key))
            if isinstance(candidate_value, int) and candidate_value > 0:
                return candidate_value
    if strategy == "DAILY_BEST":
        return max(default_bars, 96)
    return default_bars


def _normalize_price_history_df(df):
    if df is None or getattr(df, "empty", True):
        return None
    try:
        out = df.copy()
    except Exception:
        return None
    try:
        index = pd.to_datetime(out.index, errors="coerce")
    except Exception:
        return None
    try:
        if getattr(index, "tz", None) is not None:
            index = index.tz_convert("Asia/Bangkok").tz_localize(None)
    except Exception:
        try:
            index = index.tz_localize(None)
        except Exception:
            pass
    out.index = index
    out = out[~out.index.isna()]
    required = {"High", "Low", "Close"}
    if out.empty or not required.issubset(set(out.columns)):
        return None
    return out.sort_index()


def alert_history_trim_locked(path, max_rows):
    try:
        max_rows = int(max_rows)
    except Exception:
        max_rows = 0
    if max_rows < 1 or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= max_rows:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines[-max_rows:])
    except Exception:
        return


def sync_alert_history_csv_locked(*, export_enabled, jsonl_path, csv_path):
    if not export_enabled:
        return
    rows = []
    if os.path.exists(jsonl_path):
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = str(raw_line or "").strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
        except Exception:
            return
    fieldnames = alert_history_csv_fieldnames()
    try:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
    except Exception:
        return


def write_json_atomic(path, payload):
    target = str(path or "").strip()
    if not target:
        return None
    directory = os.path.dirname(os.path.abspath(target))
    os.makedirs(directory, exist_ok=True)
    fd = None
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(prefix=".tmp_verify_", suffix=".json", dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target)
        return target
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def candidate_message_preview(candidate):
    message = str((candidate or {}).get("message_plain") or (candidate or {}).get("message") or "").strip()
    if not message:
        return None
    message = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", message)).strip()
    if len(message) > 220:
        message = message[:217].rstrip() + "..."
    return message


def candidate_backtest_snapshot(candidate, *, candidate_edge_metrics):
    if not isinstance(candidate, dict):
        return {"win_rate_pct": None, "expectancy_rr": None, "trades": None}
    return candidate_edge_metrics(candidate)


def candidate_ops_snapshot(candidate, *, helpers):
    if not isinstance(candidate, dict):
        return {}
    pick_plan_value = helpers["pick_plan_value"]
    candidate_backtest_snapshot_fn = helpers["candidate_backtest_snapshot"]
    candidate_alert_profile = helpers["candidate_alert_profile"]
    normalize_symbol = helpers["normalize_symbol"]
    candidate_mode_label = helpers["candidate_mode_label"]
    get_plan_label = helpers["get_plan_label"]
    candidate_message_preview_fn = helpers["candidate_message_preview"]

    plan = candidate.get("plan")
    entry_price = _realized_entry_price(plan, candidate.get("signal"), pick_plan_value=pick_plan_value) if isinstance(plan, dict) else None
    stop_loss = pick_plan_value(plan, ["stop_loss"]) if isinstance(plan, dict) else None
    take_profit = pick_plan_value(plan, ["take_profit", "take_profit_2", "exit_price"]) if isinstance(plan, dict) else None
    snapshot = candidate_backtest_snapshot_fn(candidate)
    profile = candidate.get("alert_profile")
    if not isinstance(profile, dict):
        profile = candidate_alert_profile(candidate)
    return {
        "strategy": str(candidate.get("strategy") or "UNKNOWN").strip().upper(),
        "symbol": normalize_symbol(candidate.get("symbol") or ""),
        "signal": str(candidate.get("signal") or "").strip().upper(),
        "alert_tier": str(profile.get("tier") or "").strip() or None,
        "alert_tier_score": profile.get("composite_score") if isinstance(profile, dict) else None,
        "tier_action": str(profile.get("action_text") or "").strip() if isinstance(profile, dict) else None,
        "alert_mode": candidate_mode_label(candidate),
        "alert_intent": str(candidate.get("alert_intent") or "").strip().lower() or None,
        "alert_intent_reason": str(candidate.get("alert_intent_reason") or "").strip() or None,
        "ai_dispatch_label": str(candidate.get("ai_dispatch_label") or "").strip() or None,
        "ai_dispatch_bucket": str(candidate.get("ai_dispatch_bucket") or "").strip().lower() or None,
        "ai_dispatch_reason": str(candidate.get("ai_dispatch_reason") or "").strip() or None,
        "ai_prob_win": float(candidate.get("ai_prob_win")) if isinstance(candidate.get("ai_prob_win"), (int, float)) else None,
        "ai_expected_return_pct": float(candidate.get("ai_expected_return_pct")) if isinstance(candidate.get("ai_expected_return_pct"), (int, float)) else None,
        "ai_rank_adjustment": float(candidate.get("ai_rank_adjustment")) if isinstance(candidate.get("ai_rank_adjustment"), (int, float)) else None,
        "ai_runtime_status": str(candidate.get("ai_runtime_status") or "").strip().lower() or None,
        "ai_runtime_reason": str(candidate.get("ai_runtime_reason") or "").strip() or None,
        "entry_ai_label": str(candidate.get("entry_ai_label") or "").strip() or None,
        "entry_ai_bucket": str(candidate.get("entry_ai_bucket") or "").strip().lower() or None,
        "entry_ai_reason": str(candidate.get("entry_ai_reason") or "").strip() or None,
        "entry_ai_policy_mode": str(candidate.get("entry_ai_policy_mode") or "").strip().lower() or None,
        "entry_ai_policy_tier": str(candidate.get("entry_ai_policy_tier") or "").strip().lower() or None,
        "entry_ai_premium_label": str(candidate.get("entry_ai_premium_label") or "").strip().lower() or None,
        "entry_ai_standard_label": str(candidate.get("entry_ai_standard_label") or "").strip().lower() or None,
        "entry_ai_watch_label": str(candidate.get("entry_ai_watch_label") or "").strip().lower() or None,
        "entry_ai_strategy_policy": str(candidate.get("entry_ai_strategy_policy") or "").strip().upper() or None,
        "entry_ai_prob_entry": float(candidate.get("entry_ai_prob_entry")) if isinstance(candidate.get("entry_ai_prob_entry"), (int, float)) else None,
        "entry_ai_prob_watch": float(candidate.get("entry_ai_prob_watch")) if isinstance(candidate.get("entry_ai_prob_watch"), (int, float)) else None,
        "entry_ai_prob_avoid": float(candidate.get("entry_ai_prob_avoid")) if isinstance(candidate.get("entry_ai_prob_avoid"), (int, float)) else None,
        "entry_ai_premium_entry_threshold": float(candidate.get("entry_ai_premium_entry_threshold")) if isinstance(candidate.get("entry_ai_premium_entry_threshold"), (int, float)) else None,
        "entry_ai_premium_avoid_threshold": float(candidate.get("entry_ai_premium_avoid_threshold")) if isinstance(candidate.get("entry_ai_premium_avoid_threshold"), (int, float)) else None,
        "entry_ai_standard_entry_threshold": float(candidate.get("entry_ai_standard_entry_threshold")) if isinstance(candidate.get("entry_ai_standard_entry_threshold"), (int, float)) else None,
        "entry_ai_standard_avoid_threshold": float(candidate.get("entry_ai_standard_avoid_threshold")) if isinstance(candidate.get("entry_ai_standard_avoid_threshold"), (int, float)) else None,
        "entry_ai_watch_entry_threshold": float(candidate.get("entry_ai_watch_entry_threshold")) if isinstance(candidate.get("entry_ai_watch_entry_threshold"), (int, float)) else None,
        "entry_ai_watch_avoid_threshold": float(candidate.get("entry_ai_watch_avoid_threshold")) if isinstance(candidate.get("entry_ai_watch_avoid_threshold"), (int, float)) else None,
        "entry_ai_model_type": str(candidate.get("entry_ai_model_type") or "").strip() or None,
        "entry_ai_model_version": str(candidate.get("entry_ai_model_version") or "").strip() or None,
        "entry_ai_model_trained_at": str(candidate.get("entry_ai_model_trained_at") or "").strip() or None,
        "entry_ai_feature_schema_version": str(candidate.get("entry_ai_feature_schema_version") or "").strip() or None,
        "entry_ai_label_schema_version": str(candidate.get("entry_ai_label_schema_version") or "").strip() or None,
        "entry_ai_policy_schema_version": str(candidate.get("entry_ai_policy_schema_version") or "").strip() or None,
        "entry_ai_rank_adjustment": float(candidate.get("entry_ai_rank_adjustment")) if isinstance(candidate.get("entry_ai_rank_adjustment"), (int, float)) else None,
        "entry_ai_runtime_status": str(candidate.get("entry_ai_runtime_status") or "").strip().lower() or None,
        "entry_ai_runtime_reason": str(candidate.get("entry_ai_runtime_reason") or "").strip() or None,
        "entry_ai_runtime_threshold_adjustment": float(candidate.get("entry_ai_runtime_threshold_adjustment")) if isinstance(candidate.get("entry_ai_runtime_threshold_adjustment"), (int, float)) else None,
        "entry_ai_runtime_base_min_confidence": float(candidate.get("entry_ai_runtime_base_min_confidence")) if isinstance(candidate.get("entry_ai_runtime_base_min_confidence"), (int, float)) else None,
        "entry_ai_runtime_min_confidence": float(candidate.get("entry_ai_runtime_min_confidence")) if isinstance(candidate.get("entry_ai_runtime_min_confidence"), (int, float)) else None,
        "entry_ai_runtime_threshold_reason": str(candidate.get("entry_ai_runtime_threshold_reason") or "").strip() or None,
        "short_trade_label": str(candidate.get("short_trade_label") or "").strip() or None,
        "short_trade_bucket": str(candidate.get("short_trade_bucket") or "").strip().lower() or None,
        "short_trade_reason": str(candidate.get("short_trade_reason") or "").strip() or None,
        "short_trade_score_adjustment": float(candidate.get("short_trade_score_adjustment")) if isinstance(candidate.get("short_trade_score_adjustment"), (int, float)) else None,
        "short_trade_regime_aligned": bool(candidate.get("short_trade_regime_aligned")) if "short_trade_regime_aligned" in candidate else None,
        "market_regime": str(candidate.get("market_regime") or ((candidate.get("regime") or {}).get("market_regime")) or "").strip().upper() or None,
        "market_trend_bias": str(candidate.get("market_trend_bias") or candidate.get("market_side_bias") or ((candidate.get("regime") or {}).get("market_side_bias")) or "").strip().upper() or None,
        "symbol_regime": str(candidate.get("symbol_regime") or ((candidate.get("regime") or {}).get("symbol_regime")) or "").strip().upper() or None,
        "side_bias": str(candidate.get("side_bias") or ((candidate.get("regime") or {}).get("side_bias")) or "").strip().upper() or None,
        "regime_confidence": float(candidate.get("regime_confidence")) if isinstance(candidate.get("regime_confidence"), (int, float)) else float((candidate.get("regime") or {}).get("regime_confidence")) if isinstance((candidate.get("regime") or {}).get("regime_confidence"), (int, float)) else None,
        "regime_volatility_pct": float(candidate.get("regime_volatility_pct")) if isinstance(candidate.get("regime_volatility_pct"), (int, float)) else float((candidate.get("regime") or {}).get("volatility_pct")) if isinstance((candidate.get("regime") or {}).get("volatility_pct"), (int, float)) else None,
        "profile_runtime_threshold_applied": bool(candidate.get("profile_runtime_threshold_applied")) if "profile_runtime_threshold_applied" in candidate else None,
        "profile_runtime_threshold_reason": str(candidate.get("profile_runtime_threshold_reason") or "").strip() or None,
        "profile_runtime_market_regime": str(candidate.get("profile_runtime_market_regime") or "").strip().upper() or None,
        "profile_runtime_symbol_regime": str(candidate.get("profile_runtime_symbol_regime") or "").strip().upper() or None,
        "profile_runtime_side_bias": str(candidate.get("profile_runtime_side_bias") or "").strip().upper() or None,
        "profile_runtime_regime_alignment": str(candidate.get("profile_runtime_regime_alignment") or "").strip().lower() or None,
        "profile_runtime_freshness_bucket": str(candidate.get("profile_runtime_freshness_bucket") or "").strip().lower() or None,
        "profile_runtime_bars_since_signal": float(candidate.get("profile_runtime_bars_since_signal")) if isinstance(candidate.get("profile_runtime_bars_since_signal"), (int, float)) else None,
        "profile_runtime_min_confidence": float(candidate.get("profile_runtime_min_confidence")) if isinstance(candidate.get("profile_runtime_min_confidence"), (int, float)) else None,
        "profile_runtime_min_score": float(candidate.get("profile_runtime_min_score")) if isinstance(candidate.get("profile_runtime_min_score"), (int, float)) else None,
        "profile_runtime_min_win_rate_pct": float(candidate.get("profile_runtime_min_win_rate_pct")) if isinstance(candidate.get("profile_runtime_min_win_rate_pct"), (int, float)) else None,
        "profile_runtime_min_expectancy_rr": float(candidate.get("profile_runtime_min_expectancy_rr")) if isinstance(candidate.get("profile_runtime_min_expectancy_rr"), (int, float)) else None,
        "profile_runtime_min_trades": int(candidate.get("profile_runtime_min_trades")) if isinstance(candidate.get("profile_runtime_min_trades"), (int, float)) else None,
        "profile_runtime_min_source_count": int(candidate.get("profile_runtime_min_source_count")) if isinstance(candidate.get("profile_runtime_min_source_count"), (int, float)) else None,
        "profile_runtime_min_robustness_score": float(candidate.get("profile_runtime_min_robustness_score")) if isinstance(candidate.get("profile_runtime_min_robustness_score"), (int, float)) else None,
        "sltp_live_label": str(candidate.get("sltp_live_label") or "").strip() or None,
        "sltp_live_bucket": str(candidate.get("sltp_live_bucket") or "").strip().lower() or None,
        "sltp_live_reason": str(candidate.get("sltp_live_reason") or "").strip() or None,
        "sltp_live_score_adjustment": float(candidate.get("sltp_live_score_adjustment")) if isinstance(candidate.get("sltp_live_score_adjustment"), (int, float)) else None,
        "sltp_live_entry_gap_pct": float(candidate.get("sltp_live_entry_gap_pct")) if isinstance(candidate.get("sltp_live_entry_gap_pct"), (int, float)) else None,
        "sltp_live_stop_risk_pct": float(candidate.get("sltp_live_stop_risk_pct")) if isinstance(candidate.get("sltp_live_stop_risk_pct"), (int, float)) else None,
        "sltp_live_target_reward_pct": float(candidate.get("sltp_live_target_reward_pct")) if isinstance(candidate.get("sltp_live_target_reward_pct"), (int, float)) else None,
        "sltp_live_rr_ratio": float(candidate.get("sltp_live_rr_ratio")) if isinstance(candidate.get("sltp_live_rr_ratio"), (int, float)) else None,
        "confidence": float(candidate.get("confidence")) if isinstance(candidate.get("confidence"), (int, float)) else None,
        "score": float(candidate.get("score")) if isinstance(candidate.get("score"), (int, float)) else None,
        "source_label": get_plan_label(plan, None) if isinstance(plan, dict) else None,
        "entry_price": float(entry_price) if isinstance(entry_price, (int, float)) else None,
        "stop_loss": float(stop_loss) if isinstance(stop_loss, (int, float)) else None,
        "take_profit": float(take_profit) if isinstance(take_profit, (int, float)) else None,
        "risk_reward": float(plan.get("risk_reward")) if isinstance(plan, dict) and isinstance(plan.get("risk_reward"), (int, float)) else None,
        "detected_pattern": str(plan.get("detected_pattern") or "").strip() if isinstance(plan, dict) else None,
        "forecast_direction": str(
            plan.get("forecast_direction")
            or (((candidate.get("item") or {}).get("price_forecast") or {}).get("direction") if isinstance(candidate.get("item"), dict) else "")
            or ""
        ).strip().upper() or None,
        "plan_reason": str(plan.get("reason") or "").strip() if isinstance(plan, dict) else None,
        "timeframe_minutes": int(candidate.get("timeframe_minutes")) if isinstance(candidate.get("timeframe_minutes"), (int, float)) else None,
        "signal_timestamp": str(candidate.get("signal_timestamp") or "").strip() or None,
        "analysis_generated_at": str(candidate.get("analysis_generated_at") or "").strip() or None,
        "telegram_sent_at": str(candidate.get("telegram_sent_at") or "").strip() or None,
        "analysis_latency_seconds": float(candidate.get("analysis_latency_seconds")) if isinstance(candidate.get("analysis_latency_seconds"), (int, float)) else None,
        "analysis_to_send_seconds": float(candidate.get("analysis_to_send_seconds")) if isinstance(candidate.get("analysis_to_send_seconds"), (int, float)) else None,
        "signal_latency_seconds": float(candidate.get("signal_latency_seconds")) if isinstance(candidate.get("signal_latency_seconds"), (int, float)) else None,
        "signal_age_minutes_at_analysis": float(candidate.get("signal_age_minutes_at_analysis")) if isinstance(candidate.get("signal_age_minutes_at_analysis"), (int, float)) else None,
        "signal_age_minutes_at_send": float(candidate.get("signal_age_minutes_at_send")) if isinstance(candidate.get("signal_age_minutes_at_send"), (int, float)) else None,
        "dispatch_status_label": str(candidate.get("dispatch_status_label") or "").strip() or None,
        "dispatch_status_reason_group": str(candidate.get("dispatch_status_reason_group") or "").strip() or None,
        "dispatch_status_reason_detail": str(candidate.get("dispatch_status_reason_detail") or "").strip() or None,
        "entry_window_max_distance_pct": float(candidate.get("entry_window_max_distance_pct")) if isinstance(candidate.get("entry_window_max_distance_pct"), (int, float)) else None,
        "entry_window_max_distance_r": float(candidate.get("entry_window_max_distance_r")) if isinstance(candidate.get("entry_window_max_distance_r"), (int, float)) else None,
        "max_chase_price": float(candidate.get("max_chase_price")) if isinstance(candidate.get("max_chase_price"), (int, float)) else None,
        "backtest_win_rate_pct": snapshot.get("win_rate_pct"),
        "backtest_expectancy_rr": snapshot.get("expectancy_rr"),
        "backtest_trades": snapshot.get("trades"),
        "message_preview": candidate_message_preview_fn(candidate),
    }


def _history_period_for_window(start_dt, now_dt):
    if not isinstance(start_dt, datetime) or not isinstance(now_dt, datetime):
        return "1mo"
    age_days = max(1, (now_dt.date() - start_dt.date()).days + 2)
    if age_days <= 5:
        return "5d"
    if age_days <= 30:
        return "1mo"
    if age_days <= 90:
        return "3mo"
    if age_days <= 180:
        return "6mo"
    return "1y"


def _load_symbol_realized_history(symbol, entry_rows, *, helpers, now_dt):
    if not symbol or not entry_rows:
        return None
    interval = str(helpers["alert_realized_interval"]() or "15m").strip().lower() or "15m"
    timestamps = [
        _alert_timestamp_value(row.get("timestamp"))
        for row in entry_rows
        if isinstance(row, dict)
    ]
    timestamps = [ts for ts in timestamps if isinstance(ts, datetime)]
    earliest_ts = min(timestamps) if timestamps else now_dt
    df = None
    history_store_read = helpers.get("history_store_read")
    if callable(history_store_read):
        try:
            df = history_store_read(symbol, interval=interval, auto_adjust=True)
        except Exception:
            df = None
    df = _normalize_price_history_df(df)
    if df is not None and not df.empty:
        try:
            if df.index.min() <= pd.Timestamp(earliest_ts) and df.index.max() >= pd.Timestamp(now_dt):
                return df
        except Exception:
            pass
    get_yf_history = helpers.get("get_yf_history")
    if not callable(get_yf_history):
        return df
    period = _history_period_for_window(earliest_ts, now_dt)
    try:
        fetched = get_yf_history(symbol, period=period, interval=interval, auto_adjust=True)
    except Exception:
        fetched = None
    fetched = _normalize_price_history_df(fetched)
    return fetched if fetched is not None and not fetched.empty else df


def _directional_excursions(bars, *, signal, entry_price):
    entry_value = _safe_float(entry_price)
    if entry_value is None or entry_value <= 0 or bars is None or getattr(bars, "empty", True):
        return None, None
    highs = pd.to_numeric(bars.get("High"), errors="coerce")
    lows = pd.to_numeric(bars.get("Low"), errors="coerce")
    if str(signal or "").upper() == "BUY":
        mfe = ((highs.max() - entry_value) / entry_value) * 100.0 if len(highs.dropna()) else None
        mae = ((entry_value - lows.min()) / entry_value) * 100.0 if len(lows.dropna()) else None
    else:
        mfe = ((entry_value - lows.min()) / entry_value) * 100.0 if len(lows.dropna()) else None
        mae = ((highs.max() - entry_value) / entry_value) * 100.0 if len(highs.dropna()) else None
    return _safe_float(mfe), _safe_float(mae)


def _close_at_or_before(price_df, alert_time):
    if price_df is None or getattr(price_df, "empty", True) or not isinstance(alert_time, datetime):
        return None
    try:
        before = price_df.loc[price_df.index <= pd.Timestamp(alert_time)]
        if before is None or getattr(before, "empty", True):
            return None
        return _safe_float(before.iloc[-1].get("Close"))
    except Exception:
        return None


_WATCH_ONLY_STRATEGIES = ("TRADAR15", "TRENDRADAR15", "TREND_RADAR", "TRENDSTATE", "TRENDSTATE15")
_EXIT_PLAN_REASON_PHRASES = (
    "ถือครบ",
    "ปิดรอบ",
    "แตะเป้าปิดทำกำไร",
    "ปิดกำไร",
    "time stop",
    "take profit",
    "close round",
    "อ่อนแรง",
    "เอนลง",
    "กลับเป็นลบ",
)
_ENTRY_PLAN_REASON_PHRASES = (
    "ยังหนุน",
    "เริ่มฟื้น",
    "vixfix spike",
    "panic low",
    "regime trend",
)


def infer_alert_intent(row):
    """Infer an alert's intent for historical rows that predate alert_intent.

    Priority (highest first):
      1. plan_reason exit phrases  -> exit   (unambiguous close/tp/time-stop)
      2. existing alert_intent     -> keep   (entry/exit already classified)
      3. watch-only strategy       -> watch
      4. tier_action               -> entry/watch
      5. plan_reason entry phrases -> entry
      6. default                   -> watch  (conservative, avoid overclaiming)
    """
    if not isinstance(row, dict):
        return "watch", "invalid_row"
    existing = str(row.get("alert_intent") or "").strip().lower()
    existing_reason = str(row.get("alert_intent_reason") or "").strip()
    plan_reason = str(row.get("plan_reason") or "").strip().lower()
    strategy = str(row.get("strategy") or "").strip().upper()
    tier_action = str(row.get("tier_action") or "").strip()

    if any(phrase in plan_reason for phrase in _EXIT_PLAN_REASON_PHRASES):
        return "exit", "plan_reason_exit"
    if existing in ("entry", "exit"):
        return existing, existing_reason or "existing_intent"
    if strategy in _WATCH_ONLY_STRATEGIES:
        return "watch", "strategy_watch_only"
    if tier_action:
        if "เข้าได้" in tier_action or "เข้าเมื่อ" in tier_action:
            return "entry", "tier_action_entry"
        if "รอ" in tier_action or "ดูทิศทาง" in tier_action:
            return "watch", "tier_action_watch"
    if any(phrase in plan_reason for phrase in _ENTRY_PLAN_REASON_PHRASES):
        return "entry", "plan_reason_entry"
    if existing == "watch":
        return "watch", existing_reason or "existing_intent"
    return "watch", "default_watch"


def _resolve_directional_alert_outcome(entry, *, price_df, now_dt, max_hold_bars):
    signal = str((entry or {}).get("signal") or "").strip().upper()
    alert_time = _alert_timestamp_value((entry or {}).get("timestamp"))
    alert_id = _alert_id_value(entry)
    entry_price = _safe_float((entry or {}).get("entry_price"))
    stop_loss = _safe_float((entry or {}).get("stop_loss"))
    take_profit = _safe_float((entry or {}).get("take_profit"))
    window_bars = _safe_int((entry or {}).get("evaluation_window_bars"), max_hold_bars)
    if not isinstance(window_bars, int) or window_bars < 1:
        window_bars = max_hold_bars
    inferred_intent, inferred_intent_reason = infer_alert_intent(entry)
    outcome = {
        "alert_id": alert_id,
        "timestamp": str((entry or {}).get("timestamp") or "").strip() or None,
        "strategy": str((entry or {}).get("strategy") or "").strip().upper() or "UNKNOWN",
        "symbol": str((entry or {}).get("symbol") or "").strip().upper(),
        "signal": signal,
        "alert_intent": inferred_intent,
        "alert_intent_reason": inferred_intent_reason,
        "alert_intent_was_inferred": not bool(str((entry or {}).get("alert_intent") or "").strip()),
        "daily_pick": bool((entry or {}).get("daily_pick")),
        "timeframe": str((entry or {}).get("timeframe") or "").strip().lower() or None,
        "evaluation_window_bars": window_bars,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "outcome_status": "unsupported",
        "outcome_result": None,
        "exit_reason": None,
        "settled_at": None,
        "exit_price": None,
        "bars_observed": 0,
        "bars_to_outcome": None,
        "maturity_progress_pct": 0.0,
        "rr_realized": None,
        "pnl_pct": None,
        "mfe_pct": None,
        "mae_pct": None,
    }
    if signal not in ("BUY", "SELL"):
        outcome["exit_reason"] = "non_directional"
        return outcome
    if not isinstance(alert_time, datetime):
        outcome["exit_reason"] = "missing_timestamp"
        return outcome
    if entry_price is None or stop_loss is None:
        outcome["exit_reason"] = "missing_entry_or_stop"
        return outcome
    if price_df is None or getattr(price_df, "empty", True):
        outcome["outcome_status"] = "open"
        outcome["exit_reason"] = "history_unavailable"
        return outcome

    # Recover wrong-side entry_price from the market price at alert time. SELL
    # signals were historically recorded with the old long reference as entry,
    # which places the stop_loss on the wrong side of the short.
    if signal == "SELL" and stop_loss <= entry_price:
        recovered = _close_at_or_before(price_df, alert_time)
        if recovered is not None and stop_loss > recovered:
            entry_price = recovered
        else:
            outcome["exit_reason"] = "invalid_stop_direction"
            return outcome
    if signal == "BUY" and stop_loss >= entry_price:
        recovered = _close_at_or_before(price_df, alert_time)
        if recovered is not None and stop_loss < recovered:
            entry_price = recovered
        else:
            outcome["exit_reason"] = "invalid_stop_direction"
            return outcome
    outcome["entry_price"] = entry_price

    future = price_df.loc[price_df.index >= pd.Timestamp(alert_time)]
    if future.empty:
        outcome["outcome_status"] = "open"
        outcome["exit_reason"] = "no_future_bars"
        return outcome

    window = future.head(window_bars)
    bars_observed = len(window)
    outcome["bars_observed"] = int(bars_observed)
    outcome["maturity_progress_pct"] = round(min(100.0, (float(bars_observed) / float(window_bars)) * 100.0), 2)
    mfe_pct, mae_pct = _directional_excursions(window, signal=signal, entry_price=entry_price)
    outcome["mfe_pct"] = mfe_pct
    outcome["mae_pct"] = mae_pct

    risk = abs(float(entry_price) - float(stop_loss))
    settled_row = None
    settled_price = None
    settled_result = None
    settled_reason = None
    settled_bars = None
    for idx, (_, row) in enumerate(window.iterrows(), start=1):
        high = _safe_float(row.get("High"))
        low = _safe_float(row.get("Low"))
        if high is None or low is None:
            continue
        if signal == "BUY":
            stop_hit = low <= float(stop_loss)
            tp_hit = isinstance(take_profit, (int, float)) and high >= float(take_profit)
            if stop_hit and tp_hit:
                settled_price = float(stop_loss)
                settled_result = "loss"
                settled_reason = "same_bar_stop_and_target"
            elif stop_hit:
                settled_price = float(stop_loss)
                settled_result = "loss"
                settled_reason = "stop_loss_hit"
            elif tp_hit:
                settled_price = float(take_profit)
                settled_result = "win"
                settled_reason = "take_profit_hit"
        else:
            stop_hit = high >= float(stop_loss)
            tp_hit = isinstance(take_profit, (int, float)) and low <= float(take_profit)
            if stop_hit and tp_hit:
                settled_price = float(stop_loss)
                settled_result = "loss"
                settled_reason = "same_bar_stop_and_target"
            elif stop_hit:
                settled_price = float(stop_loss)
                settled_result = "loss"
                settled_reason = "stop_loss_hit"
            elif tp_hit:
                settled_price = float(take_profit)
                settled_result = "win"
                settled_reason = "take_profit_hit"
        if settled_reason:
            settled_row = row
            settled_bars = idx
            break

    if settled_reason:
        outcome["outcome_status"] = "settled"
        outcome["outcome_result"] = settled_result
        outcome["exit_reason"] = settled_reason
        outcome["exit_price"] = settled_price
        outcome["bars_to_outcome"] = int(settled_bars) if settled_bars is not None else None
        settled_at = pd.Timestamp(window.index[settled_bars - 1]).to_pydatetime() if settled_bars else now_dt
        outcome["settled_at"] = settled_at.strftime("%Y-%m-%d %H:%M:%S")
    elif len(future) >= window_bars and not window.empty:
        last_close = _safe_float(window.iloc[-1].get("Close"))
        if last_close is not None:
            outcome["outcome_status"] = "settled"
            outcome["exit_reason"] = "time_exit"
            outcome["exit_price"] = float(last_close)
            outcome["bars_to_outcome"] = int(len(window))
            outcome["settled_at"] = pd.Timestamp(window.index[-1]).to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
            if signal == "BUY":
                move_pct = ((float(last_close) - float(entry_price)) / float(entry_price)) * 100.0
            else:
                move_pct = ((float(entry_price) - float(last_close)) / float(entry_price)) * 100.0
            outcome["pnl_pct"] = _safe_float(move_pct)
            if risk > 0:
                if signal == "BUY":
                    rr_realized = (float(last_close) - float(entry_price)) / float(risk)
                else:
                    rr_realized = (float(entry_price) - float(last_close)) / float(risk)
                outcome["rr_realized"] = _safe_float(rr_realized)
            rr_value = _safe_float(outcome.get("rr_realized"))
            if isinstance(rr_value, (int, float)):
                if rr_value > 0:
                    outcome["outcome_result"] = "win"
                elif rr_value < 0:
                    outcome["outcome_result"] = "loss"
                else:
                    outcome["outcome_result"] = "flat"
    else:
        outcome["outcome_status"] = "open"
        outcome["exit_reason"] = "waiting_for_horizon"

    if outcome.get("pnl_pct") is None and isinstance(outcome.get("exit_price"), (int, float)):
        exit_price = float(outcome["exit_price"])
        if signal == "BUY":
            pnl_pct = ((exit_price - float(entry_price)) / float(entry_price)) * 100.0
            rr_value = (exit_price - float(entry_price)) / float(risk) if risk > 0 else None
        else:
            pnl_pct = ((float(entry_price) - exit_price) / float(entry_price)) * 100.0
            rr_value = (float(entry_price) - exit_price) / float(risk) if risk > 0 else None
        outcome["pnl_pct"] = _safe_float(pnl_pct)
        outcome["rr_realized"] = _safe_float(rr_value)
    return outcome


def _realized_metric_average(rows, field):
    values = [_safe_float(row.get(field)) for row in (rows or [])]
    values = [value for value in values if isinstance(value, (int, float))]
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _resolve_directional_alert_outcomes(entries, *, helpers, get_now):
    directional = []
    by_symbol = {}
    for entry in (entries or []):
        if not isinstance(entry, dict):
            continue
        signal = str(entry.get("signal") or "").strip().upper()
        symbol = str(entry.get("symbol") or "").strip().upper()
        if signal in ("BUY", "SELL") and symbol:
            directional.append(entry)
            by_symbol.setdefault(symbol, []).append(entry)

    outcomes = []
    now_dt = get_now()
    max_hold_bars = _safe_int(helpers["alert_realized_max_hold_bars"](), 64)
    if max_hold_bars is None or max_hold_bars < 1:
        max_hold_bars = 64
    for symbol, rows in by_symbol.items():
        history_df = _load_symbol_realized_history(symbol, rows, helpers=helpers, now_dt=now_dt)
        for entry in rows:
            outcomes.append(
                _resolve_directional_alert_outcome(
                    entry,
                    price_df=history_df,
                    now_dt=now_dt,
                    max_hold_bars=max_hold_bars,
                )
            )
    return directional, outcomes


def _build_telegram_realized_report_from_entries(entries, *, days_value, helpers, get_now, strategy_order, history_lock):
    enabled = bool(helpers["alert_realized_enabled"]())
    generated_at = get_now().strftime("%Y-%m-%d %H:%M:%S")
    summary = {
        "enabled": enabled,
        "generated_at": generated_at,
        "window_days": days_value,
        "total_alerts": len(entries or []),
        "eligible_directional_alerts": 0,
        "settled_alerts": 0,
        "open_alerts": 0,
        "unsupported_alerts": 0,
        "wins": 0,
        "losses": 0,
        "flats": 0,
        "win_rate_pct": None,
        "avg_rr_realized": None,
        "avg_pnl_pct": None,
        "alerts_per_day_avg": None,
        "alerts_per_30d_est": None,
        "by_strategy": {},
        "by_month": {},
    }
    if not enabled:
        return summary

    directional, outcomes = _resolve_directional_alert_outcomes(entries, helpers=helpers, get_now=get_now)
    summary["eligible_directional_alerts"] = len(directional)

    settled = [row for row in outcomes if row.get("outcome_status") == "settled"]
    open_rows = [row for row in outcomes if row.get("outcome_status") == "open"]
    unsupported = [row for row in outcomes if row.get("outcome_status") == "unsupported"]
    wins = [row for row in settled if row.get("outcome_result") == "win"]
    losses = [row for row in settled if row.get("outcome_result") == "loss"]
    flats = [row for row in settled if row.get("outcome_result") == "flat"]

    summary["settled_alerts"] = len(settled)
    summary["open_alerts"] = len(open_rows)
    summary["unsupported_alerts"] = len(unsupported)
    summary["wins"] = len(wins)
    summary["losses"] = len(losses)
    summary["flats"] = len(flats)
    if settled:
        summary["win_rate_pct"] = (float(len(wins)) / float(len(settled))) * 100.0
    summary["avg_rr_realized"] = _realized_metric_average(settled, "rr_realized")
    summary["avg_pnl_pct"] = _realized_metric_average(settled, "pnl_pct")
    if isinstance(days_value, (int, float)) and days_value > 0:
        alerts_per_day_avg = float(len(entries or [])) / float(days_value)
        summary["alerts_per_day_avg"] = alerts_per_day_avg
        summary["alerts_per_30d_est"] = alerts_per_day_avg * 30.0

    by_strategy = {}
    for row in outcomes:
        strategy = str(row.get("strategy") or "UNKNOWN").strip().upper()
        bucket = by_strategy.setdefault(
            strategy,
            {
                "alerts": 0,
                "settled_alerts": 0,
                "open_alerts": 0,
                "wins": 0,
                "losses": 0,
                "flats": 0,
                "avg_rr_realized": None,
                "avg_pnl_pct": None,
                "_rows": [],
            },
        )
        bucket["alerts"] += 1
        bucket["_rows"].append(row)
        if row.get("outcome_status") == "settled":
            bucket["settled_alerts"] += 1
            if row.get("outcome_result") == "win":
                bucket["wins"] += 1
            elif row.get("outcome_result") == "loss":
                bucket["losses"] += 1
            elif row.get("outcome_result") == "flat":
                bucket["flats"] += 1
        elif row.get("outcome_status") == "open":
            bucket["open_alerts"] += 1
    ordered_strategies = list(strategy_order) + sorted([key for key in by_strategy.keys() if key not in strategy_order])
    summary["by_strategy"] = {}
    for strategy in ordered_strategies:
        bucket = by_strategy.get(strategy)
        if not bucket:
            continue
        rows = bucket.pop("_rows", [])
        bucket["avg_rr_realized"] = _realized_metric_average([row for row in rows if row.get("outcome_status") == "settled"], "rr_realized")
        bucket["avg_pnl_pct"] = _realized_metric_average([row for row in rows if row.get("outcome_status") == "settled"], "pnl_pct")
        bucket["win_rate_pct"] = (
            (float(bucket["wins"]) / float(bucket["settled_alerts"])) * 100.0
            if bucket["settled_alerts"] > 0
            else None
        )
        summary["by_strategy"][strategy] = bucket

    by_month = {}
    for row in outcomes:
        timestamp = _alert_timestamp_value(row.get("timestamp"))
        if not isinstance(timestamp, datetime):
            continue
        month_key = timestamp.strftime("%Y-%m")
        days_in_month = monthrange(timestamp.year, timestamp.month)[1]
        bucket = by_month.setdefault(
            month_key,
            {
                "alerts": 0,
                "settled_alerts": 0,
                "wins": 0,
                "losses": 0,
                "flats": 0,
                "alerts_per_day_in_month": None,
                "win_rate_pct": None,
            },
        )
        bucket["alerts"] += 1
        bucket["alerts_per_day_in_month"] = float(bucket["alerts"]) / float(days_in_month)
        if row.get("outcome_status") == "settled":
            bucket["settled_alerts"] += 1
            if row.get("outcome_result") == "win":
                bucket["wins"] += 1
            elif row.get("outcome_result") == "loss":
                bucket["losses"] += 1
            elif row.get("outcome_result") == "flat":
                bucket["flats"] += 1
            if bucket["settled_alerts"] > 0:
                bucket["win_rate_pct"] = (float(bucket["wins"]) / float(bucket["settled_alerts"])) * 100.0
    summary["by_month"] = {key: by_month[key] for key in sorted(by_month.keys())}

    entry_only_rows = [row for row in outcomes if str(row.get("alert_intent") or "").strip().lower() == "entry"]
    entry_settled = [row for row in entry_only_rows if row.get("outcome_status") == "settled"]
    entry_wins = [row for row in entry_settled if row.get("outcome_result") == "win"]
    entry_losses = [row for row in entry_settled if row.get("outcome_result") == "loss"]
    entry_flats = [row for row in entry_settled if row.get("outcome_result") == "flat"]

    def _count_entry_rows(rows, *, group_by):
        buckets = {}
        for row in rows:
            timestamp = _alert_timestamp_value(row.get("timestamp"))
            if not isinstance(timestamp, datetime):
                continue
            if group_by == "month":
                key = timestamp.strftime("%Y-%m")
            else:
                key = str(row.get("strategy") or "UNKNOWN").strip().upper() or "UNKNOWN"
            bucket = buckets.setdefault(
                key,
                {"settled_alerts": 0, "wins": 0, "losses": 0, "flats": 0, "win_rate_pct": None, "_rows": []},
            )
            bucket["settled_alerts"] += 1
            bucket["_rows"].append(row)
            result = row.get("outcome_result")
            if result == "win":
                bucket["wins"] += 1
            elif result == "loss":
                bucket["losses"] += 1
            elif result == "flat":
                bucket["flats"] += 1
        for key, bucket in buckets.items():
            if bucket["settled_alerts"] > 0:
                bucket["win_rate_pct"] = (float(bucket["wins"]) / float(bucket["settled_alerts"])) * 100.0
            bucket["avg_rr_realized"] = _realized_metric_average(bucket["_rows"], "rr_realized")
            bucket["avg_pnl_pct"] = _realized_metric_average(bucket["_rows"], "pnl_pct")
            bucket.pop("_rows", None)
        return buckets

    entry_by_month = _count_entry_rows(entry_settled, group_by="month")
    entry_by_strategy = _count_entry_rows(entry_settled, group_by="strategy")

    summary["entry_only"] = {
        "settled_alerts": len(entry_settled),
        "wins": len(entry_wins),
        "losses": len(entry_losses),
        "flats": len(entry_flats),
        "win_rate_pct": (float(len(entry_wins)) / float(len(entry_settled))) * 100.0 if entry_settled else None,
        "avg_rr_realized": _realized_metric_average(entry_settled, "rr_realized"),
        "avg_pnl_pct": _realized_metric_average(entry_settled, "pnl_pct"),
        "by_month": {key: entry_by_month[key] for key in sorted(entry_by_month.keys())},
        "by_strategy": entry_by_strategy,
    }

    with history_lock:
        write_json_atomic(helpers["alert_realized_summary_file_path"](), summary)
        if helpers["alert_realized_export_outcomes"]():
            write_json_atomic(
                helpers["alert_outcomes_file_path"](),
                {
                    "generated_at": generated_at,
                    "window_days": days_value,
                    "outcomes": outcomes,
                },
            )
    return summary


def _trade_close_outcome_icon(result):
    if result == "win":
        return "✅"
    if result == "loss":
        return "❌"
    return "⏱️"


def _trade_close_exit_reason_label(exit_reason):
    text = str(exit_reason or "").strip().lower()
    if text in ("take_profit", "tp"):
        return "กระทบเป้า (TP)"
    if text in ("stop_loss", "sl"):
        return "กระทบหยุดขาดทุน (SL)"
    if text in ("time_exit", "time_stop", "time_stop_exit"):
        return "หมดเวลา Hold Plan"
    if text in ("invalid_stop_direction",):
        return "ข้อมูล Stop ผิดด้าน"
    if text in ("missing_entry_or_stop",):
        return "ข้อมูล Entry/Stop ไม่ครบ"
    if text in ("non_directional",):
        return "ไม่ใช่สัญญาณ BUY/SELL"
    if text in ("missing_timestamp",):
        return "ไม่มีเวลาสัญญาณ"
    if text in ("history_unavailable",):
        return "ไม่มีข้อมูลราคา"
    return "—"


def _build_trade_close_message(outcome, *, get_now):
    symbol = str(outcome.get("symbol") or "").strip().upper() or "—"
    signal = str(outcome.get("signal") or "").strip().upper() or "—"
    result = str(outcome.get("outcome_result") or "").strip().lower()
    icon = _trade_close_outcome_icon(result)
    result_label = {"win": "ชนะ", "loss": "แพ้", "flat": "เสมอ"}.get(result, result or "—")
    strategy = str(outcome.get("strategy") or "—").strip().upper()
    entry_price = outcome.get("entry_price")
    exit_price = outcome.get("exit_price")
    stop_loss = outcome.get("stop_loss")
    take_profit = outcome.get("take_profit")
    rr = outcome.get("rr_realized")
    pnl = outcome.get("pnl_pct")
    bars_to_outcome = outcome.get("bars_to_outcome")
    window_bars = outcome.get("evaluation_window_bars")
    exit_reason = _trade_close_exit_reason_label(outcome.get("exit_reason"))
    timestamp = str(outcome.get("timestamp") or "").strip() or "—"
    intent = str(outcome.get("alert_intent") or "").strip().lower() or "—"
    tv_symbol = symbol.replace("-", "")

    lines = []
    lines.append(f"{icon} <b>ปิดไม้แล้ว — {result_label}</b>")
    lines.append("────────────────")
    lines.append(f"<b>เหรียญ:</b> {html.escape(symbol)} | <b>สัญญาณ:</b> {html.escape(signal)}")
    lines.append(f"<b>กลยุทธ์:</b> {html.escape(strategy)} | <b>Intent:</b> {html.escape(intent)}")
    lines.append(f"<b>เวลาเข้า:</b> {html.escape(timestamp)}")
    if isinstance(entry_price, (int, float)):
        lines.append(f"<b>Entry:</b> {entry_price:.6g}")
    if isinstance(exit_price, (int, float)):
        lines.append(f"<b>Exit:</b> {exit_price:.6g}")
    if isinstance(stop_loss, (int, float)):
        lines.append(f"<b>SL:</b> {stop_loss:.6g}")
    if isinstance(take_profit, (int, float)):
        lines.append(f"<b>TP:</b> {take_profit:.6g}")
    if isinstance(rr, (int, float)):
        lines.append(f"<b>RR:</b> {rr:+.2f}R")
    if isinstance(pnl, (int, float)):
        lines.append(f"<b>PnL:</b> {pnl:+.2f}%")
    if isinstance(bars_to_outcome, (int, float)) and isinstance(window_bars, int):
        lines.append(f"<b>แท่งที่เข้าไป:</b> {int(bars_to_outcome)}/{window_bars}")
    lines.append(f"<b>สาเหตุปิด:</b> {html.escape(exit_reason)}")
    lines.append("────────────────")
    lines.append("🕒 <b>เวลา:</b> " + get_now().strftime("%Y-%m-%d %H:%M"))
    lines.append(f"<a href=\"https://th.tradingview.com/chart/?symbol=CRYPTO:{tv_symbol}\">📈 TradingView</a>")
    return "\n".join(lines)


def _load_previous_settled_outcome_ids(path):
    payload = _read_json_file(path)
    if not isinstance(payload, dict):
        return set()
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list):
        return set()
    ids = set()
    for row in outcomes:
        if not isinstance(row, dict):
            continue
        if str(row.get("outcome_status") or "").strip().lower() != "settled":
            continue
        alert_id = str(row.get("alert_id") or "").strip()
        if alert_id:
            ids.add(alert_id)
    return ids


def _load_notified_close_ids(path):
    payload = _read_json_file(path)
    if not isinstance(payload, dict):
        return set()
    ids = payload.get("notified_alert_ids")
    if not isinstance(ids, list):
        return set()
    return {str(item).strip() for item in ids if str(item).strip()}


def _save_notified_close_ids(path, ids, *, max_keep=500):
    trimmed = sorted(set(str(item).strip() for item in ids if str(item).strip()))[-max_keep:]
    write_json_atomic(path, {"notified_alert_ids": trimmed, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


def dispatch_trade_close_notifications(
    *,
    config,
    helpers,
    get_now,
    send_telegram_alert,
    history_lock,
    strategy_order,
    realized_report_days=None,
):
    """Detect newly-settled outcomes and send a close notification for each.

    Reads the previously-persisted outcomes, regenerates fresh outcomes from
    alert history, diffs to find alert_ids that are settled now but were not
    settled before (or were already notified), and sends a Telegram message
    per newly settled outcome.
    """
    if not bool(getattr(config, "TELEGRAM_ALERT_TRADE_CLOSE_NOTIFICATIONS_ENABLE", True)):
        return {"enabled": False, "sent": 0, "skipped": 0}
    if not callable(send_telegram_alert):
        return {"enabled": False, "sent": 0, "skipped": 0}
    if not bool(helpers.get("alert_realized_enabled")()):
        return {"enabled": False, "sent": 0, "skipped": 0}

    outcomes_path = helpers["alert_outcomes_file_path"]()
    notified_path = outcomes_path.replace("realized_outcomes.json", "notified_closes.json")
    max_per_run = int(getattr(config, "TELEGRAM_ALERT_TRADE_CLOSE_MAX_PER_RUN", 5) or 5)
    only_entry = bool(getattr(config, "TELEGRAM_ALERT_TRADE_CLOSE_ONLY_ENTRY", True))
    skip_flat = bool(getattr(config, "TELEGRAM_ALERT_TRADE_CLOSE_SKIP_FLAT", False))

    # Snapshot the previously-settled alert_ids BEFORE regeneration.
    previous_settled_ids = _load_previous_settled_outcome_ids(outcomes_path)
    already_notified_ids = _load_notified_close_ids(notified_path)

    # Regenerate fresh outcomes by calling the existing report builder.
    days_value = int(realized_report_days or helpers.get("alert_realized_report_days", lambda: 90)() or 90)
    summary = _build_telegram_realized_report_from_entries(
        helpers["read_telegram_alert_history"](days=days_value),
        days_value=days_value,
        helpers=helpers,
        get_now=get_now,
        strategy_order=strategy_order,
        history_lock=history_lock,
    )

    # Read the freshly-written outcomes file.
    fresh_payload = _read_json_file(outcomes_path)
    if not isinstance(fresh_payload, dict):
        return {"enabled": True, "sent": 0, "skipped": 0, "error": "no_fresh_outcomes"}
    fresh_outcomes = fresh_payload.get("outcomes")
    if not isinstance(fresh_outcomes, list):
        return {"enabled": True, "sent": 0, "skipped": 0, "error": "no_outcomes_list"}

    sent = 0
    skipped = 0
    newly_notified = list(already_notified_ids)
    for outcome in fresh_outcomes:
        if not isinstance(outcome, dict):
            continue
        if str(outcome.get("outcome_status") or "").strip().lower() != "settled":
            continue
        alert_id = str(outcome.get("alert_id") or "").strip()
        if not alert_id:
            continue
        # Skip if already notified.
        if alert_id in already_notified_ids:
            skipped += 1
            continue
        # Skip if it was already settled in the previous snapshot (we just
        # hadn't notified yet — still notify, but this prevents re-notifying
        # across multiple runs in the same cycle).
        # Only notify for outcomes that are NEWLY settled since last run.
        if alert_id in previous_settled_ids:
            # Was already settled before — mark as notified without sending.
            newly_notified.append(alert_id)
            skipped += 1
            continue
        intent = str(outcome.get("alert_intent") or "").strip().lower()
        if only_entry and intent != "entry":
            newly_notified.append(alert_id)
            skipped += 1
            continue
        result = str(outcome.get("outcome_result") or "").strip().lower()
        if skip_flat and result == "flat":
            newly_notified.append(alert_id)
            skipped += 1
            continue
        if sent >= max_per_run:
            skipped += 1
            continue
        message = _build_trade_close_message(outcome, get_now=get_now)
        if not isinstance(message, str) or not message.strip():
            continue
        if send_telegram_alert(message):
            sent += 1
            newly_notified.append(alert_id)

    # Persist notified ids so we never send the same close twice.
    if newly_notified != list(already_notified_ids):
        try:
            _save_notified_close_ids(notified_path, newly_notified)
        except Exception:
            pass

    return {
        "enabled": True,
        "sent": sent,
        "skipped": skipped,
        "previous_settled": len(previous_settled_ids),
        "already_notified": len(already_notified_ids),
    }


def record_telegram_run_report(
    *,
    results,
    kill,
    kill_reason,
    min_conf,
    dynamic_min_conf,
    candidates,
    raw_candidates=None,
    sent_candidates,
    daily_pick_sent,
    daily_summary_sent,
    dropped_by_cache,
    dropped_by_symbol_cap,
    dropped_by_run_cap,
    quality_drop_counts,
    alert_budget,
    reject_diagnostics,
    config,
    helpers,
    get_now,
    history_lock,
):
    if not helpers["alert_run_report_enabled"]():
        return
    top_n = getattr(config, "TELEGRAM_ALERT_RUN_REPORT_TOP_CANDIDATES", 5)
    max_rows = getattr(config, "TELEGRAM_ALERT_RUN_REPORT_MAX_ROWS", 500)
    try:
        top_n = max(1, int(top_n))
    except Exception:
        top_n = 5
    try:
        max_rows = int(max_rows)
    except Exception:
        max_rows = 500
    normalize_symbol = helpers["normalize_symbol"]
    candidate_ops_snapshot_fn = helpers["candidate_ops_snapshot"]
    sync_alert_history_csv_locked_fn = helpers["sync_alert_history_csv_locked"]
    alert_history_trim_locked_fn = helpers["alert_history_trim_locked"]
    latest_path = helpers["alert_run_report_file_path"]()
    log_path = helpers["alert_run_report_log_path"]()

    valid_results = [row for row in (results or []) if isinstance(row, dict) and not row.get("error")]
    by_symbol_signal = Counter()
    for row in valid_results:
        symbol = normalize_symbol(row.get("symbol") or "")
        signal = str(row.get("signal") or "").strip().upper() or "UNKNOWN"
        if symbol:
            by_symbol_signal[f"{symbol}|{signal}"] += 1
    report = {
        "generated_at": get_now().strftime("%Y-%m-%d %H:%M:%S"),
        "result_count": len(results or []),
        "valid_symbol_count": len(valid_results),
        "kill_switch_active": bool(kill),
        "kill_switch_reason": str(kill_reason or "") if kill else None,
        "min_confidence": float(min_conf) if isinstance(min_conf, (int, float)) else None,
        "dynamic_min_confidence": float(dynamic_min_conf) if isinstance(dynamic_min_conf, (int, float)) else None,
        "candidate_count": len(candidates or []),
        "raw_candidate_count": len(raw_candidates or []),
        "sent_count": len(sent_candidates or []),
        "daily_pick_sent": int(daily_pick_sent or 0),
        "daily_summary_sent": int(daily_summary_sent or 0),
        "dropped_by_cache": int(dropped_by_cache or 0),
        "dropped_by_symbol_cap": int(dropped_by_symbol_cap or 0),
        "dropped_by_run_cap": int(dropped_by_run_cap or 0),
        "quality_drop_counts": dict(quality_drop_counts or {}),
        "reject_diagnostics": dict(reject_diagnostics or {}),
        "alert_budget": dict(alert_budget or {}),
        "symbol_signal_mix": dict(by_symbol_signal),
        "top_candidates": [candidate_ops_snapshot_fn(row) for row in (candidates or [])[:top_n]],
        "raw_top_candidates": [candidate_ops_snapshot_fn(row) for row in (raw_candidates or [])[:top_n]],
        "sent_candidates": [candidate_ops_snapshot_fn(row) for row in (sent_candidates or [])],
    }
    try:
        with history_lock:
            sync_alert_history_csv_locked_fn()
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(report, ensure_ascii=False) + "\n")
            alert_history_trim_locked_fn(log_path, max_rows=max_rows)
    except Exception:
        return


def read_latest_telegram_run_report(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _read_json_file(path):
    target = str(path or "").strip()
    if not target or not os.path.exists(target):
        return None
    try:
        with open(target, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _summary_section(payload):
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return summary
    return payload


def _basename_text(path):
    text = str(path or "").strip()
    if not text:
        return None
    return os.path.basename(text.replace("\\", os.sep).replace("/", os.sep)) or text


def _fmt_number(value, digits=1, suffix=""):
    number = _safe_float(value)
    if number is None:
        return "-"
    return f"{number:.{int(digits)}f}{suffix}"


def _fmt_count(value):
    count = _safe_int(value)
    if count is None:
        return "-"
    return str(int(count))


def _format_counter_items(counter_map, *, limit=3, key_map=None):
    if not isinstance(counter_map, dict):
        return "-"
    items = []
    for key, value in counter_map.items():
        count = _safe_int(value)
        text = str(key or "").strip()
        if count is None or count <= 0 or not text:
            continue
        if isinstance(key_map, dict):
            text = str(key_map.get(text, text))
        items.append((text, int(count)))
    if not items:
        return "-"
    items.sort(key=lambda item: (-item[1], item[0]))
    return ", ".join([f"{text} {count}" for text, count in items[: max(1, int(limit))]])


_QUALITY_DROP_FRIENDLY_LABELS = {
    "all_weather_no_actionable_subplans": "ไม่มี setup ที่เข้าได้",
    "candidate_win_rate_below_min": "Win Rate ต่ำกว่าเกณฑ์",
    "candidate_expectancy_below_min": "Expectancy ต่ำกว่าเกณฑ์",
    "candidate_profile_win_rate_below_min": "WR โปรไฟล์ต่ำกว่าเกณฑ์",
    "candidate_missing_edge_metrics": "ไม่มีสถิติ edge",
    "trend_radar_watch_intent_filtered": "Trend Radar เป็นแค่ Watch",
    "realized_expectancy_below_floor": "Expectancy จริงติดลบ",
    "realized_win_rate_below_floor": "Win Rate จริงต่ำกว่าเกณฑ์",
    "regime_min_confidence_not_met": "Regime ต้องการ confidence สูงขึ้น",
    "primary_stale_entry_suppressed": "สัญญาณเข้าเก่าเกินไป",
    "primary_stale_exit_suppressed": "สัญญาณออกเก่าเกินไป",
}


def _top_quality_drop_counts(quality_drop_counts, *, limit=3, key_map=None):
    if not isinstance(quality_drop_counts, dict):
        return "-"
    cleaned = []
    for key, value in quality_drop_counts.items():
        count = _safe_int(value)
        raw = str(key or "").strip()
        if count is None or count <= 0 or not raw:
            continue
        if isinstance(key_map, dict):
            text = str(key_map.get(raw, raw.replace("_", " ")))
        else:
            text = raw.replace("_", " ")
        cleaned.append((text, int(count)))
    if not cleaned:
        return "-"
    cleaned.sort(key=lambda item: (-item[1], item[0]))
    return ", ".join([f"{text} {count}" for text, count in cleaned[: max(1, int(limit))]])


def _latest_entry_ai_runtime(entries, *, helpers):
    runtime = {
        "model_path": None,
        "model_name": None,
        "model_version": None,
        "trained_at": None,
        "policy_schema_version": None,
        "allowlist": None,
        "live_enabled": None,
    }
    model_path_getter = helpers.get("entry_ai_model_path")
    if callable(model_path_getter):
        runtime["model_path"] = str(model_path_getter() or "").strip() or None
        runtime["model_name"] = _basename_text(runtime["model_path"])
    live_enabled_getter = helpers.get("entry_ai_live_enabled")
    if callable(live_enabled_getter):
        runtime["live_enabled"] = bool(live_enabled_getter())
    allowlist_getter = helpers.get("entry_ai_allowlist_text")
    if callable(allowlist_getter):
        allowlist = str(allowlist_getter() or "").strip()
        runtime["allowlist"] = allowlist or None
    for row in (entries or []):
        if not isinstance(row, dict):
            continue
        if runtime["model_version"] is None:
            text = str(row.get("entry_ai_model_version") or "").strip()
            runtime["model_version"] = text or None
        if runtime["trained_at"] is None:
            text = str(row.get("entry_ai_model_trained_at") or "").strip()
            runtime["trained_at"] = text or None
        if runtime["policy_schema_version"] is None:
            text = str(row.get("entry_ai_policy_schema_version") or "").strip()
            runtime["policy_schema_version"] = text or None
        if runtime["model_version"] and runtime["trained_at"] and runtime["policy_schema_version"]:
            break
    return runtime


def _rank_daily_summary_candidates(existing_candidates, *, results, min_conf, top_n, helpers):
    ranked = [row for row in (existing_candidates or []) if isinstance(row, dict)]
    build_candidates = helpers.get("build_telegram_candidates")
    if not ranked and callable(build_candidates):
        try:
            built = build_candidates(results or [], min_conf, runtime_context=None)
        except TypeError:
            built = build_candidates(results or [], min_conf)
        if isinstance(built, tuple):
            ranked = [row for row in (built[0] or []) if isinstance(row, dict)]
        elif isinstance(built, list):
            ranked = [row for row in built if isinstance(row, dict)]
    ranked.sort(
        key=lambda row: (
            _safe_float(row.get("score"), 0.0) or 0.0,
            _safe_float(row.get("confidence"), 0.0) or 0.0,
        ),
        reverse=True,
    )
    normalize_symbol = helpers["normalize_symbol"]
    unique_ranked = []
    seen_symbols = set()
    for row in ranked:
        symbol = normalize_symbol(row.get("symbol") or "")
        if not symbol or symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        unique_ranked.append(row)
        if len(unique_ranked) >= top_n:
            break
    return unique_ranked


def build_telegram_daily_summary_message(
    results,
    *,
    existing_candidates=None,
    min_conf=None,
    config,
    helpers,
    get_now,
    strategy_order,
    history_lock,
):
    if not bool(getattr(config, "TELEGRAM_DAILY_SUMMARY_ENABLED", True)):
        return None
    try:
        top_n = max(1, int(getattr(config, "TELEGRAM_DAILY_SUMMARY_TOP_N", 3)))
    except Exception:
        top_n = 3
    try:
        history_days = float(getattr(config, "TELEGRAM_DAILY_SUMMARY_HISTORY_DAYS", 1.0))
    except Exception:
        history_days = 1.0
    if history_days <= 0:
        history_days = 1.0
    try:
        realized_days = float(getattr(config, "TELEGRAM_DAILY_SUMMARY_REALIZED_DAYS", 45.0))
    except Exception:
        realized_days = 45.0
    if realized_days <= 0:
        realized_days = 45.0
    if not isinstance(min_conf, (int, float)):
        min_conf = getattr(config, "TELEGRAM_DAILY_BEST_PICK_MIN_CONFIDENCE", 58.0)
    try:
        min_conf = float(min_conf)
    except Exception:
        min_conf = 58.0

    now_dt = get_now()
    now_text = now_dt.strftime("%Y-%m-%d %H:%M")
    recent_entries = helpers["read_telegram_alert_history"](days=history_days)
    latest_run = read_latest_telegram_run_report(helpers["alert_run_report_file_path"]())
    realized_payload = _read_json_file(helpers["alert_realized_summary_file_path"]()) or {}
    realized_summary = _summary_section(realized_payload)
    feedback_summary = _summary_section(_read_json_file(helpers["alert_feedback_summary_file_path"]()) or {})
    training_summary = _summary_section(_read_json_file(helpers["live_feedback_training_summary_file_path"]()) or {})
    calibration_summary = _summary_section(_read_json_file(helpers["live_feedback_calibration_summary_file_path"]()) or {})
    shadow_summary = _summary_section(_read_json_file(helpers["live_feedback_shadow_summary_file_path"]()) or {})
    runtime = _latest_entry_ai_runtime(recent_entries, helpers=helpers)

    strategies = Counter()
    symbols = Counter()
    signals = Counter()
    intents = Counter()
    tiers = Counter()
    buckets = Counter()
    daily_pick_count = 0
    directional_count = 0
    latest_alert_at = None
    for entry in recent_entries:
        if not isinstance(entry, dict):
            continue
        strategy = str(entry.get("strategy") or "").strip().upper()
        symbol = helpers["normalize_symbol"](entry.get("symbol") or "")
        signal = str(entry.get("signal") or "").strip().upper()
        intent = str(entry.get("alert_intent") or "").strip().lower()
        tier = str(entry.get("entry_ai_policy_tier") or "").strip().lower()
        bucket = str(entry.get("entry_ai_bucket") or "").strip().lower()
        if strategy:
            strategies[strategy] += 1
        if symbol:
            symbols[symbol] += 1
        if signal:
            signals[signal] += 1
        if intent:
            intents[intent] += 1
        if tier:
            tiers[tier] += 1
        if bucket:
            buckets[bucket] += 1
        if bool(entry.get("daily_pick")):
            daily_pick_count += 1
        if signal in {"BUY", "SELL"}:
            directional_count += 1
        ts_text = str(entry.get("timestamp") or "").strip()
        if ts_text and latest_alert_at is None:
            latest_alert_at = ts_text

    ranked = _rank_daily_summary_candidates(
        existing_candidates,
        results=results,
        min_conf=min_conf,
        top_n=top_n,
        helpers=helpers,
    )
    candidate_backtest_snapshot = helpers["candidate_backtest_snapshot"]
    if ranked:
        trade_status = "🟢 มีสัญญาณพร้อมพิจารณา"
    elif daily_pick_count > 0:
        trade_status = "🟡 มี Daily Pick เฝ้ารอ"
    elif isinstance(latest_run, dict) and (_safe_int(latest_run.get("sent_count")) or 0) > 0:
        trade_status = "🟢 มีสัญญาณส่งไปแล้ว"
    else:
        trade_status = "⛔ ยังไม่ควรเข้าเทรด"
    lines = [
        "<b>Daily Summary</b>",
        f"⏱️ <b>เวลา:</b> {html.escape(now_text)}",
        f"🎯 <b>สถานะวันนี้:</b> {html.escape(trade_status)}",
    ]

    runtime_parts = [
        "ON" if runtime.get("live_enabled") else "OFF",
        runtime.get("model_name") or "-",
    ]
    if runtime.get("model_version"):
        runtime_parts.append(f"ver {runtime['model_version']}")
    if runtime.get("trained_at"):
        runtime_parts.append(f"trained {runtime['trained_at']}")

    if isinstance(latest_run, dict):
        alert_budget = latest_run.get("alert_budget") if isinstance(latest_run.get("alert_budget"), dict) else {}
        regime_text = str(alert_budget.get("market_regime") or "-").strip().upper() or "-"
        run_cap = _fmt_count(alert_budget.get("adjusted_run_cap"))
        daily_cap = _fmt_count(alert_budget.get("adjusted_daily_pick_cap"))
        quality_drop_text = _top_quality_drop_counts(
            latest_run.get("quality_drop_counts"),
            key_map=_QUALITY_DROP_FRIENDLY_LABELS,
        )
        lines.append(
            "🧭 <b>ภาพรวม:</b> "
            + f"regime {html.escape(regime_text)}"
            + f" | min conf {html.escape(_fmt_number(latest_run.get('dynamic_min_confidence'), digits=1, suffix='%'))}"
            + f" | cap/run {html.escape(run_cap)} | daily pick {html.escape(daily_cap)}"
        )
        lines.append(
            "🧪 <b>รอบล่าสุด:</b> "
            + f"{html.escape(str(latest_run.get('generated_at') or '-'))}"
            + f" | raw {html.escape(_fmt_count(latest_run.get('raw_candidate_count')))}"
            + f" -> cand {html.escape(_fmt_count(latest_run.get('candidate_count')))}"
            + f" -> sent {html.escape(_fmt_count(latest_run.get('sent_count')))}"
        )
        lines.append(
            "📤 <b>สถานะส่ง:</b> "
            + f"daily pick {html.escape(_fmt_count(latest_run.get('daily_pick_sent')))}"
            + f" | daily summary {html.escape(_fmt_count(latest_run.get('daily_summary_sent')))}"
        )
        if quality_drop_text != "-":
            lines.append(f"🚫 <b>เหตุที่ยังไม่ส่ง:</b> {html.escape(quality_drop_text)}")
    else:
        lines.append("🧪 <b>รอบล่าสุด:</b> ยังไม่มี `latest_run.json` ใน runtime path")

    if recent_entries:
        activity_line = (
            "📣 <b>กิจกรรมล่าสุด:</b> "
            + f"{len(recent_entries)} alerts/{html.escape(_fmt_number(history_days, digits=1))}d"
            + f" | directional {directional_count}"
            + f" | daily pick {daily_pick_count}"
            + f" | symbols {len(symbols)}"
        )
        if latest_alert_at:
            activity_line += f" | last {html.escape(latest_alert_at)}"
        lines.append(activity_line)
        lines.append(
            "📊 <b>Flow:</b> "
            + f"intent {html.escape(_format_counter_items(dict(intents), limit=3))}"
            + f" | tier {html.escape(_format_counter_items(dict(tiers), limit=3))}"
            + f" | strategy {html.escape(_format_counter_items(dict(strategies), limit=2))}"
        )
        lines.append(
            "🗂️ <b>สัญลักษณ์เด่น:</b> "
            + f"{html.escape(_format_counter_items(dict(symbols), limit=3))}"
            + f" | signal {html.escape(_format_counter_items(dict(signals), limit=3))}"
        )
    else:
        lines.append(f"📣 <b>กิจกรรมล่าสุด:</b> ยังไม่มี alert ในช่วง {html.escape(_fmt_number(history_days, digits=1))} วัน")

    realized_generated_at = str(realized_summary.get("generated_at") or realized_payload.get("generated_at") or "").strip()
    lines.append(
        "🎯 <b>ผลงานจริง "
        + f"{html.escape(_fmt_number(realized_days, digits=0))}d:</b> "
        + f"WR {html.escape(_fmt_number(realized_summary.get('win_rate_pct'), suffix='%'))}"
        + f" | Expectancy {html.escape(_fmt_number(realized_summary.get('avg_rr_realized'), digits=2, suffix='R'))}"
        + f" | PnL {html.escape(_fmt_number(realized_summary.get('avg_pnl_pct'), digits=2, suffix='%'))}"
        + f" | settled {html.escape(_fmt_count(realized_summary.get('settled_alerts')))}"
        + f" | alerts/day {html.escape(_fmt_number(realized_summary.get('alerts_per_day_avg'), digits=1))}"
    )
    if realized_generated_at:
        lines.append(f"📅 <b>อัปเดตผลงาน:</b> {html.escape(realized_generated_at)}")

    if ranked:
        lines.append("📌 <b>ตัวเด่นตอนนี้:</b>")
        for idx, row in enumerate(ranked, start=1):
            metrics = candidate_backtest_snapshot(row)
            symbol = helpers["normalize_symbol"](row.get("symbol") or "") or "-"
            strategy = str(row.get("strategy") or "ALERT").strip().upper() or "ALERT"
            signal = str(row.get("signal") or "WAIT").strip().upper() or "WAIT"
            tier = str(row.get("entry_ai_policy_tier") or row.get("entry_ai_bucket") or "").strip().lower() or "-"
            confidence = _safe_float(row.get("confidence"))
            score = _safe_float(row.get("score"))
            wr = _safe_float(metrics.get("win_rate_pct"))
            lines.append(
                f"{idx}. <b>{html.escape(symbol)}</b> | {html.escape(signal)} | {html.escape(strategy)}"
                + f" | tier {html.escape(tier)}"
                + f" | conf {html.escape(_fmt_number(confidence, digits=0, suffix='%'))}"
                + f" | score {html.escape(_fmt_number(score, digits=1))}"
                + f" | WR {html.escape(_fmt_number(wr, digits=1, suffix='%'))}"
            )
    else:
        no_candidate_reason = "-"
        if isinstance(latest_run, dict):
            no_candidate_reason = _top_quality_drop_counts(
                latest_run.get("quality_drop_counts"),
                key_map=_QUALITY_DROP_FRIENDLY_LABELS,
            )
        lines.append(
            "📌 <b>ตัวเด่นตอนนี้:</b> ยังไม่มี candidate ที่ผ่าน gate"
            + (f" | เหตุหลัก {html.escape(no_candidate_reason)}" if no_candidate_reason != "-" else "")
        )

    system_parts = list(runtime_parts)
    if runtime.get("allowlist"):
        system_parts.append(f"promote {runtime['allowlist']}")
    shadow_model_version = str(shadow_summary.get("model_version") or "").strip()
    if shadow_model_version:
        system_parts.append(f"shadow {shadow_model_version}")
    lines.append("🤖 <b>ระบบ:</b> " + " | ".join([html.escape(str(part)) for part in system_parts if str(part).strip()]))
    lines.append(
        "🔁 <b>Learning:</b> "
        + f"feedback {html.escape(_fmt_count(feedback_summary.get('training_ready_rows')))} ready"
        + f" | train {html.escape(_fmt_count(training_summary.get('filled_rows')))}"
        + f" | calibration {html.escape(_fmt_count(calibration_summary.get('filled_row_count')))}"
        + f" | shadow {html.escape(_fmt_count(shadow_summary.get('filled_row_count')))}"
    )

    return {
        "strategy": "DAILY_SUMMARY",
        "signal": "INFO",
        "score": float(_safe_float(realized_summary.get("win_rate_pct"), 0.0) or 0.0),
        "confidence": float(_safe_float(realized_summary.get("win_rate_pct"), 0.0) or 0.0),
        "edge_metrics": {
            "win_rate_pct": _safe_float(realized_summary.get("win_rate_pct")),
            "expectancy_rr": _safe_float(realized_summary.get("avg_rr_realized")),
            "trades": _safe_float(realized_summary.get("settled_alerts")),
        },
        "message": "\n".join(lines),
        "cache_key": f"DAILYSUMMARY_EXEC|{now_dt.strftime('%Y%m%d')}",
    }


def record_telegram_alert_history(
    candidate,
    *,
    min_conf=None,
    dynamic_min_conf=None,
    daily_pick=False,
    config,
    helpers,
    get_now,
    history_lock,
):
    if not helpers["alert_history_enabled"]() or not isinstance(candidate, dict):
        return
    message = str(candidate.get("message") or "").strip()
    if not message:
        return
    candidate_backtest_snapshot_fn = helpers["candidate_backtest_snapshot"]
    pick_plan_value = helpers["pick_plan_value"]
    candidate_alert_profile = helpers["candidate_alert_profile"]
    candidate_mode_label = helpers["candidate_mode_label"]
    get_plan_label = helpers["get_plan_label"]
    normalize_symbol = helpers["normalize_symbol"]
    alert_history_trim_locked_fn = helpers["alert_history_trim_locked"]
    sync_alert_history_csv_locked_fn = helpers["sync_alert_history_csv_locked"]

    snapshot = candidate_backtest_snapshot_fn(candidate)
    plan = candidate.get("plan")
    entry_price = _realized_entry_price(plan, candidate.get("signal"), pick_plan_value=pick_plan_value) if isinstance(plan, dict) else None
    stop_loss = pick_plan_value(plan, ["stop_loss"]) if isinstance(plan, dict) else None
    take_profit = pick_plan_value(plan, ["take_profit", "take_profit_2", "exit_price"]) if isinstance(plan, dict) else None
    profile = candidate.get("alert_profile")
    if not isinstance(profile, dict):
        profile = candidate_alert_profile(candidate)
    entry = {
        "alert_id": _alert_id_value(
            {
                "timestamp": get_now().strftime("%Y-%m-%d %H:%M:%S"),
                "strategy": str(candidate.get("strategy") or "UNKNOWN").strip().upper(),
                "symbol": normalize_symbol(candidate.get("symbol") or ""),
                "signal": str(candidate.get("signal") or "").strip().upper(),
                "cache_key": str(candidate.get("cache_key") or "").strip(),
                "message_plain": re.sub(r"<[^>]+>", "", message).strip(),
            }
        ),
        "timestamp": get_now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": str(candidate.get("strategy") or "UNKNOWN").strip().upper(),
        "symbol": normalize_symbol(candidate.get("symbol") or ""),
        "signal": str(candidate.get("signal") or "").strip().upper(),
        "timeframe": _candidate_timeframe(candidate, config=config),
        "evaluation_window_bars": _candidate_evaluation_window_bars(candidate, config=config),
        "alert_tier": str(profile.get("tier") or "").strip() if isinstance(profile, dict) else None,
        "alert_tier_score": profile.get("composite_score") if isinstance(profile, dict) else None,
        "tier_action": str(profile.get("action_text") or "").strip() if isinstance(profile, dict) else None,
        "alert_mode": candidate_mode_label(candidate),
        "alert_intent": str(candidate.get("alert_intent") or "").strip().lower() or None,
        "alert_intent_reason": str(candidate.get("alert_intent_reason") or "").strip() or None,
        "ai_dispatch_label": str(candidate.get("ai_dispatch_label") or "").strip() or None,
        "ai_dispatch_bucket": str(candidate.get("ai_dispatch_bucket") or "").strip().lower() or None,
        "ai_dispatch_reason": str(candidate.get("ai_dispatch_reason") or "").strip() or None,
        "ai_prob_win": float(candidate.get("ai_prob_win")) if isinstance(candidate.get("ai_prob_win"), (int, float)) else None,
        "ai_expected_return_pct": float(candidate.get("ai_expected_return_pct")) if isinstance(candidate.get("ai_expected_return_pct"), (int, float)) else None,
        "ai_rank_adjustment": float(candidate.get("ai_rank_adjustment")) if isinstance(candidate.get("ai_rank_adjustment"), (int, float)) else None,
        "ai_runtime_status": str(candidate.get("ai_runtime_status") or "").strip().lower() or None,
        "ai_runtime_reason": str(candidate.get("ai_runtime_reason") or "").strip() or None,
        "entry_ai_label": str(candidate.get("entry_ai_label") or "").strip() or None,
        "entry_ai_bucket": str(candidate.get("entry_ai_bucket") or "").strip().lower() or None,
        "entry_ai_reason": str(candidate.get("entry_ai_reason") or "").strip() or None,
        "entry_ai_policy_mode": str(candidate.get("entry_ai_policy_mode") or "").strip().lower() or None,
        "entry_ai_policy_tier": str(candidate.get("entry_ai_policy_tier") or "").strip().lower() or None,
        "entry_ai_premium_label": str(candidate.get("entry_ai_premium_label") or "").strip().lower() or None,
        "entry_ai_standard_label": str(candidate.get("entry_ai_standard_label") or "").strip().lower() or None,
        "entry_ai_watch_label": str(candidate.get("entry_ai_watch_label") or "").strip().lower() or None,
        "entry_ai_strategy_policy": str(candidate.get("entry_ai_strategy_policy") or "").strip().upper() or None,
        "entry_ai_prob_entry": float(candidate.get("entry_ai_prob_entry")) if isinstance(candidate.get("entry_ai_prob_entry"), (int, float)) else None,
        "entry_ai_prob_watch": float(candidate.get("entry_ai_prob_watch")) if isinstance(candidate.get("entry_ai_prob_watch"), (int, float)) else None,
        "entry_ai_prob_avoid": float(candidate.get("entry_ai_prob_avoid")) if isinstance(candidate.get("entry_ai_prob_avoid"), (int, float)) else None,
        "entry_ai_premium_entry_threshold": float(candidate.get("entry_ai_premium_entry_threshold")) if isinstance(candidate.get("entry_ai_premium_entry_threshold"), (int, float)) else None,
        "entry_ai_premium_avoid_threshold": float(candidate.get("entry_ai_premium_avoid_threshold")) if isinstance(candidate.get("entry_ai_premium_avoid_threshold"), (int, float)) else None,
        "entry_ai_standard_entry_threshold": float(candidate.get("entry_ai_standard_entry_threshold")) if isinstance(candidate.get("entry_ai_standard_entry_threshold"), (int, float)) else None,
        "entry_ai_standard_avoid_threshold": float(candidate.get("entry_ai_standard_avoid_threshold")) if isinstance(candidate.get("entry_ai_standard_avoid_threshold"), (int, float)) else None,
        "entry_ai_watch_entry_threshold": float(candidate.get("entry_ai_watch_entry_threshold")) if isinstance(candidate.get("entry_ai_watch_entry_threshold"), (int, float)) else None,
        "entry_ai_watch_avoid_threshold": float(candidate.get("entry_ai_watch_avoid_threshold")) if isinstance(candidate.get("entry_ai_watch_avoid_threshold"), (int, float)) else None,
        "entry_ai_model_type": str(candidate.get("entry_ai_model_type") or "").strip() or None,
        "entry_ai_model_version": str(candidate.get("entry_ai_model_version") or "").strip() or None,
        "entry_ai_model_trained_at": str(candidate.get("entry_ai_model_trained_at") or "").strip() or None,
        "entry_ai_feature_schema_version": str(candidate.get("entry_ai_feature_schema_version") or "").strip() or None,
        "entry_ai_label_schema_version": str(candidate.get("entry_ai_label_schema_version") or "").strip() or None,
        "entry_ai_policy_schema_version": str(candidate.get("entry_ai_policy_schema_version") or "").strip() or None,
        "entry_ai_rank_adjustment": float(candidate.get("entry_ai_rank_adjustment")) if isinstance(candidate.get("entry_ai_rank_adjustment"), (int, float)) else None,
        "entry_ai_runtime_status": str(candidate.get("entry_ai_runtime_status") or "").strip().lower() or None,
        "entry_ai_runtime_reason": str(candidate.get("entry_ai_runtime_reason") or "").strip() or None,
        "entry_ai_runtime_threshold_adjustment": float(candidate.get("entry_ai_runtime_threshold_adjustment")) if isinstance(candidate.get("entry_ai_runtime_threshold_adjustment"), (int, float)) else None,
        "entry_ai_runtime_base_min_confidence": float(candidate.get("entry_ai_runtime_base_min_confidence")) if isinstance(candidate.get("entry_ai_runtime_base_min_confidence"), (int, float)) else None,
        "entry_ai_runtime_min_confidence": float(candidate.get("entry_ai_runtime_min_confidence")) if isinstance(candidate.get("entry_ai_runtime_min_confidence"), (int, float)) else None,
        "entry_ai_runtime_threshold_reason": str(candidate.get("entry_ai_runtime_threshold_reason") or "").strip() or None,
        "short_trade_label": str(candidate.get("short_trade_label") or "").strip() or None,
        "short_trade_bucket": str(candidate.get("short_trade_bucket") or "").strip().lower() or None,
        "short_trade_reason": str(candidate.get("short_trade_reason") or "").strip() or None,
        "short_trade_score_adjustment": float(candidate.get("short_trade_score_adjustment")) if isinstance(candidate.get("short_trade_score_adjustment"), (int, float)) else None,
        "short_trade_regime_aligned": bool(candidate.get("short_trade_regime_aligned")) if "short_trade_regime_aligned" in candidate else None,
        "sltp_live_label": str(candidate.get("sltp_live_label") or "").strip() or None,
        "sltp_live_bucket": str(candidate.get("sltp_live_bucket") or "").strip().lower() or None,
        "sltp_live_reason": str(candidate.get("sltp_live_reason") or "").strip() or None,
        "sltp_live_score_adjustment": float(candidate.get("sltp_live_score_adjustment")) if isinstance(candidate.get("sltp_live_score_adjustment"), (int, float)) else None,
        "sltp_live_entry_gap_pct": float(candidate.get("sltp_live_entry_gap_pct")) if isinstance(candidate.get("sltp_live_entry_gap_pct"), (int, float)) else None,
        "sltp_live_stop_risk_pct": float(candidate.get("sltp_live_stop_risk_pct")) if isinstance(candidate.get("sltp_live_stop_risk_pct"), (int, float)) else None,
        "sltp_live_target_reward_pct": float(candidate.get("sltp_live_target_reward_pct")) if isinstance(candidate.get("sltp_live_target_reward_pct"), (int, float)) else None,
        "sltp_live_rr_ratio": float(candidate.get("sltp_live_rr_ratio")) if isinstance(candidate.get("sltp_live_rr_ratio"), (int, float)) else None,
        "confidence": float(candidate.get("confidence")) if isinstance(candidate.get("confidence"), (int, float)) else None,
        "score": float(candidate.get("score")) if isinstance(candidate.get("score"), (int, float)) else None,
        "daily_pick": bool(daily_pick),
        "message": message,
        "message_plain": re.sub(r"<[^>]+>", "", message).strip(),
        "cache_key": str(candidate.get("cache_key") or "").strip(),
        "min_confidence": float(min_conf) if isinstance(min_conf, (int, float)) else None,
        "dynamic_min_confidence": float(dynamic_min_conf) if isinstance(dynamic_min_conf, (int, float)) else None,
        "backtest_win_rate_pct": snapshot.get("win_rate_pct"),
        "backtest_expectancy_rr": snapshot.get("expectancy_rr"),
        "backtest_trades": snapshot.get("trades"),
        "strategy_label": str(candidate.get("strategy_label") or "").strip() or None,
        "source_label": get_plan_label(plan, None) if isinstance(plan, dict) else None,
        "entry_price": float(entry_price) if isinstance(entry_price, (int, float)) else None,
        "stop_loss": float(stop_loss) if isinstance(stop_loss, (int, float)) else None,
        "take_profit": float(take_profit) if isinstance(take_profit, (int, float)) else None,
        "risk_reward": float(plan.get("risk_reward")) if isinstance(plan, dict) and isinstance(plan.get("risk_reward"), (int, float)) else None,
        "detected_pattern": str(plan.get("detected_pattern") or "").strip() if isinstance(plan, dict) else None,
        "forecast_direction": str(plan.get("forecast_direction") or "").strip().upper() if isinstance(plan, dict) and str(plan.get("forecast_direction") or "").strip() else None,
        "forecast_score": float(plan.get("forecast_score")) if isinstance(plan, dict) and isinstance(plan.get("forecast_score"), (int, float)) else None,
        "plan_reason": str(plan.get("reason") or "").strip() if isinstance(plan, dict) else None,
        "source_count": int(candidate.get("source_count")) if isinstance(candidate.get("source_count"), (int, float)) else None,
        "bars_since_signal": pick_plan_value(plan, ["bars_since_signal", "bars_since_entry"]) if isinstance(plan, dict) else None,
        "timeframe_minutes": int(candidate.get("timeframe_minutes")) if isinstance(candidate.get("timeframe_minutes"), (int, float)) else None,
        "signal_timestamp": str(candidate.get("signal_timestamp") or "").strip() or None,
        "analysis_generated_at": str(candidate.get("analysis_generated_at") or "").strip() or None,
        "telegram_sent_at": str(candidate.get("telegram_sent_at") or "").strip() or None,
        "analysis_latency_seconds": float(candidate.get("analysis_latency_seconds")) if isinstance(candidate.get("analysis_latency_seconds"), (int, float)) else None,
        "analysis_to_send_seconds": float(candidate.get("analysis_to_send_seconds")) if isinstance(candidate.get("analysis_to_send_seconds"), (int, float)) else None,
        "signal_latency_seconds": float(candidate.get("signal_latency_seconds")) if isinstance(candidate.get("signal_latency_seconds"), (int, float)) else None,
        "signal_age_minutes_at_analysis": float(candidate.get("signal_age_minutes_at_analysis")) if isinstance(candidate.get("signal_age_minutes_at_analysis"), (int, float)) else None,
        "signal_age_minutes_at_send": float(candidate.get("signal_age_minutes_at_send")) if isinstance(candidate.get("signal_age_minutes_at_send"), (int, float)) else None,
        "dispatch_status_label": str(candidate.get("dispatch_status_label") or "").strip() or None,
        "dispatch_status_reason_group": str(candidate.get("dispatch_status_reason_group") or "").strip() or None,
        "dispatch_status_reason_detail": str(candidate.get("dispatch_status_reason_detail") or "").strip() or None,
        "entry_window_max_distance_pct": float(candidate.get("entry_window_max_distance_pct")) if isinstance(candidate.get("entry_window_max_distance_pct"), (int, float)) else None,
        "entry_window_max_distance_r": float(candidate.get("entry_window_max_distance_r")) if isinstance(candidate.get("entry_window_max_distance_r"), (int, float)) else None,
        "max_chase_price": float(candidate.get("max_chase_price")) if isinstance(candidate.get("max_chase_price"), (int, float)) else None,
        "red_to_green_quality_score": float(plan.get("red_to_green_quality_score")) if isinstance(plan, dict) and isinstance(plan.get("red_to_green_quality_score"), (int, float)) else None,
        "green_flip_reclaim": bool(plan.get("green_flip_reclaim")) if isinstance(plan, dict) and "green_flip_reclaim" in plan else None,
    }
    path = helpers["alert_history_file_path"]()
    max_rows = getattr(config, "TELEGRAM_ALERT_HISTORY_MAX_ROWS", 5000)
    try:
        max_rows = int(max_rows)
    except Exception:
        max_rows = 5000
    try:
        with history_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            alert_history_trim_locked_fn(path, max_rows=max_rows)
            sync_alert_history_csv_locked_fn()
    except Exception:
        return


def read_telegram_alert_history(*, days=None, strategies=None, symbols=None, helpers, get_now, history_lock):
    path = helpers["alert_history_file_path"]()
    normalize_symbol = helpers["normalize_symbol"]
    if not os.path.exists(path):
        return []
    strategy_filter = {str(v or "").strip().upper() for v in (strategies or []) if str(v or "").strip()}
    symbol_filter = {normalize_symbol(v) for v in (symbols or []) if normalize_symbol(v)}
    cutoff = None
    if isinstance(days, (int, float)) and float(days) > 0:
        cutoff = get_now() - helpers["timedelta"](days=float(days))
    entries = []
    try:
        with history_lock:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
    except Exception:
        return []
    for raw_line in lines:
        line = str(raw_line or "").strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        strategy = str(row.get("strategy") or "").strip().upper()
        symbol = normalize_symbol(row.get("symbol") or "")
        if strategy_filter and strategy not in strategy_filter:
            continue
        if symbol_filter and symbol not in symbol_filter:
            continue
        ts_text = str(row.get("timestamp") or "").strip()
        ts_value = None
        if ts_text:
            try:
                ts_value = datetime.strptime(ts_text, "%Y-%m-%d %H:%M:%S")
            except Exception:
                ts_value = None
        if cutoff is not None and isinstance(ts_value, datetime) and ts_value < cutoff:
            continue
        row["_timestamp_obj"] = ts_value
        row["strategy"] = strategy
        row["symbol"] = symbol
        entries.append(row)
    entries.sort(key=lambda row: row.get("_timestamp_obj") or datetime.min, reverse=True)
    return entries


def build_telegram_alert_report(*, days=30, strategies=None, symbols=None, limit_examples_per_strategy=1, helpers, get_now, strategy_order, history_lock):
    entries = helpers["read_telegram_alert_history"](
        days=days,
        strategies=strategies,
        symbols=symbols,
    )
    try:
        days_value = float(days) if days is not None else None
    except Exception:
        days_value = None
    try:
        limit_examples_per_strategy = max(1, int(limit_examples_per_strategy))
    except Exception:
        limit_examples_per_strategy = 1
    normalize_symbol = helpers["normalize_symbol"]
    empty_realized = _build_telegram_realized_report_from_entries(
        [],
        days_value=days_value,
        helpers=helpers,
        get_now=get_now,
        strategy_order=strategy_order,
        history_lock=history_lock,
    )
    if not entries:
        table = []
        for strategy in strategy_order:
            table.append(
                {
                    "strategy": strategy,
                    "alert_count": 0,
                    "share_pct": 0.0,
                    "unique_symbols": 0,
                    "avg_confidence": None,
                    "avg_backtest_win_rate_pct": None,
                    "avg_backtest_expectancy_rr": None,
                    "avg_backtest_trades": None,
                    "realized_settled_alerts": 0,
                    "realized_open_alerts": 0,
                    "realized_win_rate_pct": None,
                    "realized_avg_rr": None,
                    "realized_avg_pnl_pct": None,
                    "signals": {},
                    "latest_alert_at": None,
                    "examples": [],
                }
            )
        return {
            "generated_at": get_now().strftime("%Y-%m-%d %H:%M:%S"),
            "window_days": days_value,
            "total_alerts": 0,
            "alerts_per_day_avg": 0.0 if isinstance(days_value, (int, float)) and days_value > 0 else None,
            "unique_symbols": 0,
            "count_by_strategy": {},
            "table": table,
            "examples_by_strategy": {},
            "realized": empty_realized,
        }

    by_strategy = {}
    unique_symbols = set()
    for entry in entries:
        strategy = str(entry.get("strategy") or "UNKNOWN").strip().upper()
        symbol = normalize_symbol(entry.get("symbol") or "")
        unique_symbols.add(symbol)
        bucket = by_strategy.setdefault(
            strategy,
            {
                "count": 0,
                "confidence_sum": 0.0,
                "confidence_count": 0,
                "wr_sum": 0.0,
                "wr_count": 0,
                "exp_sum": 0.0,
                "exp_count": 0,
                "trades_sum": 0.0,
                "trades_count": 0,
                "signals": Counter(),
                "symbols": set(),
                "latest_alert_at": None,
                "examples": [],
            },
        )
        bucket["count"] += 1
        bucket["signals"][str(entry.get("signal") or "WAIT")] += 1
        if symbol:
            bucket["symbols"].add(symbol)
        ts_text = str(entry.get("timestamp") or "").strip() or None
        if bucket["latest_alert_at"] is None and ts_text:
            bucket["latest_alert_at"] = ts_text
        confidence = entry.get("confidence")
        if isinstance(confidence, (int, float)):
            bucket["confidence_sum"] += float(confidence)
            bucket["confidence_count"] += 1
        win_rate = entry.get("backtest_win_rate_pct")
        if isinstance(win_rate, (int, float)):
            bucket["wr_sum"] += float(win_rate)
            bucket["wr_count"] += 1
        expectancy = entry.get("backtest_expectancy_rr")
        if isinstance(expectancy, (int, float)):
            bucket["exp_sum"] += float(expectancy)
            bucket["exp_count"] += 1
        trades = entry.get("backtest_trades")
        if isinstance(trades, (int, float)):
            bucket["trades_sum"] += float(trades)
            bucket["trades_count"] += 1
        if len(bucket["examples"]) < limit_examples_per_strategy:
            bucket["examples"].append(
                {
                    "timestamp": ts_text,
                    "symbol": symbol,
                    "signal": str(entry.get("signal") or "WAIT"),
                    "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
                    "message": str(entry.get("message") or ""),
                    "message_plain": str(entry.get("message_plain") or ""),
                }
            )

    total_alerts = len(entries)
    count_by_strategy = {strategy: int(bucket["count"]) for strategy, bucket in by_strategy.items()}
    realized = _build_telegram_realized_report_from_entries(
        entries,
        days_value=days_value,
        helpers=helpers,
        get_now=get_now,
        strategy_order=strategy_order,
        history_lock=history_lock,
    )
    ordered_keys = list(strategy_order) + sorted([key for key in by_strategy.keys() if key not in strategy_order])
    table = []
    examples_by_strategy = {}
    for strategy in ordered_keys:
        bucket = by_strategy.get(strategy)
        realized_bucket = (realized.get("by_strategy") or {}).get(strategy) or {}
        if not bucket:
            table.append(
                {
                    "strategy": strategy,
                    "alert_count": 0,
                    "share_pct": 0.0,
                    "unique_symbols": 0,
                    "avg_confidence": None,
                    "avg_backtest_win_rate_pct": None,
                    "avg_backtest_expectancy_rr": None,
                    "avg_backtest_trades": None,
                    "realized_settled_alerts": int(realized_bucket.get("settled_alerts") or 0),
                    "realized_open_alerts": int(realized_bucket.get("open_alerts") or 0),
                    "realized_win_rate_pct": realized_bucket.get("win_rate_pct"),
                    "realized_avg_rr": realized_bucket.get("avg_rr_realized"),
                    "realized_avg_pnl_pct": realized_bucket.get("avg_pnl_pct"),
                    "signals": {},
                    "latest_alert_at": None,
                    "examples": [],
                }
            )
            continue
        avg_conf = (bucket["confidence_sum"] / bucket["confidence_count"]) if bucket["confidence_count"] > 0 else None
        avg_wr = (bucket["wr_sum"] / bucket["wr_count"]) if bucket["wr_count"] > 0 else None
        avg_exp = (bucket["exp_sum"] / bucket["exp_count"]) if bucket["exp_count"] > 0 else None
        avg_trades = (bucket["trades_sum"] / bucket["trades_count"]) if bucket["trades_count"] > 0 else None
        row = {
            "strategy": strategy,
            "alert_count": int(bucket["count"]),
            "share_pct": (float(bucket["count"]) / float(total_alerts) * 100.0) if total_alerts > 0 else 0.0,
            "unique_symbols": len(bucket["symbols"]),
            "avg_confidence": avg_conf,
            "avg_backtest_win_rate_pct": avg_wr,
            "avg_backtest_expectancy_rr": avg_exp,
            "avg_backtest_trades": avg_trades,
            "realized_settled_alerts": int(realized_bucket.get("settled_alerts") or 0),
            "realized_open_alerts": int(realized_bucket.get("open_alerts") or 0),
            "realized_win_rate_pct": realized_bucket.get("win_rate_pct"),
            "realized_avg_rr": realized_bucket.get("avg_rr_realized"),
            "realized_avg_pnl_pct": realized_bucket.get("avg_pnl_pct"),
            "signals": dict(bucket["signals"]),
            "latest_alert_at": bucket["latest_alert_at"],
            "examples": bucket["examples"],
        }
        table.append(row)
        examples_by_strategy[strategy] = bucket["examples"]
    alerts_per_day_avg = None
    if isinstance(days_value, (int, float)) and days_value > 0:
        alerts_per_day_avg = float(total_alerts) / float(days_value)
    return {
        "generated_at": get_now().strftime("%Y-%m-%d %H:%M:%S"),
        "window_days": days_value,
        "total_alerts": int(total_alerts),
        "alerts_per_day_avg": alerts_per_day_avg,
        "unique_symbols": len([s for s in unique_symbols if s]),
        "count_by_strategy": count_by_strategy,
        "table": table,
        "examples_by_strategy": examples_by_strategy,
        "realized": realized,
    }


def build_telegram_alert_realized_report(*, days=30, strategies=None, symbols=None, helpers, get_now, strategy_order, history_lock):
    entries = helpers["read_telegram_alert_history"](
        days=days,
        strategies=strategies,
        symbols=symbols,
    )
    try:
        days_value = float(days) if days is not None else None
    except Exception:
        days_value = None
    return _build_telegram_realized_report_from_entries(
        entries,
        days_value=days_value,
        helpers=helpers,
        get_now=get_now,
        strategy_order=strategy_order,
        history_lock=history_lock,
    )


def build_telegram_alert_live_preview(results, *, limit_examples_per_strategy=1, config, helpers, get_now, strategy_order, runtime_context=None):
    try:
        limit_examples_per_strategy = max(1, int(limit_examples_per_strategy))
    except Exception:
        limit_examples_per_strategy = 1
    min_conf = getattr(config, "TELEGRAM_ALERT_MIN_CONFIDENCE", 69.0)
    try:
        min_conf = float(min_conf)
    except Exception:
        min_conf = 69.0

    build_alert_runtime_context = helpers["build_alert_runtime_context"]
    build_telegram_candidates = helpers["build_telegram_candidates"]
    build_daily_best_pick_candidates = helpers["build_daily_best_pick_candidates"]
    normalize_symbol = helpers["normalize_symbol"]
    candidate_backtest_snapshot_fn = helpers["candidate_backtest_snapshot"]

    if not isinstance(runtime_context, dict):
        runtime_context = build_alert_runtime_context(results or [], min_conf)
    else:
        try:
            min_conf = float((runtime_context or {}).get("min_confidence"))
        except Exception:
            pass
    kill = bool((runtime_context or {}).get("kill"))
    reason = (runtime_context or {}).get("kill_reason")
    alert_budget = (runtime_context or {}).get("alert_budget") or {}
    dynamic_min_conf = float((runtime_context or {}).get("dynamic_min_confidence") or float(min_conf))
    candidates = []
    build_stats = {}
    if not kill:
        candidates, build_stats = build_telegram_candidates(results, dynamic_min_conf, runtime_context=runtime_context)
    daily_candidates = build_daily_best_pick_candidates(results, runtime_context=runtime_context)
    for row in daily_candidates:
        if isinstance(row, dict):
            row.setdefault("cache_key", f"PREVIEW|{row.get('strategy')}|{row.get('symbol')}|{row.get('signal')}")
    combined = [row for row in candidates if isinstance(row, dict)] + [row for row in daily_candidates if isinstance(row, dict)]
    combined.sort(key=lambda row: (float(row.get("score", 0.0)), float(row.get("confidence", 0.0))), reverse=True)
    by_strategy = Counter()
    examples_by_strategy = {}
    for candidate in combined:
        strategy = str(candidate.get("strategy") or "UNKNOWN").strip().upper()
        by_strategy[strategy] += 1
        bucket = examples_by_strategy.setdefault(strategy, [])
        if len(bucket) >= limit_examples_per_strategy:
            continue
        snapshot = candidate_backtest_snapshot_fn(candidate)
        bucket.append(
            {
                "symbol": normalize_symbol(candidate.get("symbol") or ""),
                "signal": str(candidate.get("signal") or "").strip().upper(),
                "confidence": float(candidate.get("confidence")) if isinstance(candidate.get("confidence"), (int, float)) else None,
                "message": str(candidate.get("message") or ""),
                "backtest_win_rate_pct": snapshot.get("win_rate_pct"),
                "backtest_expectancy_rr": snapshot.get("expectancy_rr"),
                "backtest_trades": snapshot.get("trades"),
            }
        )
    table = []
    ordered_keys = list(strategy_order) + sorted([k for k in by_strategy.keys() if k not in strategy_order])
    for strategy in ordered_keys:
        table.append(
            {
                "strategy": strategy,
                "candidate_count": int(by_strategy.get(strategy, 0)),
                "examples": examples_by_strategy.get(strategy, []),
            }
        )
    quality_drop_counts = build_stats.get("quality_drop_counts") if isinstance(build_stats, dict) else {}
    regime_summary = (runtime_context or {}).get("regime_summary") or {}
    if isinstance(build_stats, dict):
        regime_summary = build_stats.get("regime_summary") or regime_summary
        alert_budget = build_stats.get("alert_budget") or alert_budget
    return {
        "generated_at": get_now().strftime("%Y-%m-%d %H:%M:%S"),
        "kill_switch_active": bool(kill),
        "kill_switch_reason": str(reason or "") if kill else None,
        "min_confidence": float(min_conf),
        "dynamic_min_confidence": float(dynamic_min_conf),
        "candidate_count": len(combined),
        "count_by_strategy": dict(by_strategy),
        "quality_drop_counts": quality_drop_counts or {},
        "regime_summary": regime_summary or {},
        "alert_budget": alert_budget or {},
        "table": table,
        "examples_by_strategy": examples_by_strategy,
    }


def write_verify_output(
    output_path,
    *,
    results,
    request_meta,
    summary,
    telegram_alerts,
    all_weather,
    backtest_rules,
    health,
    latest_run,
    live_preview,
    regime_summary,
    realized_performance,
    runtime_context=None,
    include_results=False,
    clean_json_value,
):
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "request": clean_json_value(request_meta or {}),
        "summary": clean_json_value(summary or {}),
        "telegram_alerts": clean_json_value(telegram_alerts or {}),
        "all_weather": clean_json_value(all_weather or {}),
        "backtest_rules": clean_json_value(backtest_rules or {}),
        "health": clean_json_value(health or {}),
        "latest_run": clean_json_value(latest_run or {}),
        "live_preview": clean_json_value(live_preview or {}),
        "regime_summary": clean_json_value(regime_summary or {}),
        "alert_runtime_context": clean_json_value(runtime_context or {}),
        "realized_performance": clean_json_value(realized_performance or {}),
        "artifact_type": "verify_output",
        "includes_results": bool(include_results),
    }
    if include_results:
        payload["results"] = clean_json_value(results or [])
    return write_json_atomic(output_path, payload)


def alert_feedback_export_fieldnames():
    return alert_history_csv_fieldnames() + [
        "outcome_status",
        "outcome_result",
        "exit_reason",
        "settled_at",
        "exit_price",
        "bars_observed",
        "bars_to_outcome",
        "maturity_progress_pct",
        "rr_realized",
        "pnl_pct",
        "mfe_pct",
        "mae_pct",
        "feedback_group",
        "feedback_is_directional",
        "feedback_is_settled",
        "feedback_is_win",
        "feedback_is_loss",
        "feedback_is_flat",
        "feedback_training_ready",
    ]


def _feedback_export_row(entry, outcome):
    row = {key: entry.get(key) for key in alert_history_csv_fieldnames()}
    outcome_status = str((outcome or {}).get("outcome_status") or "").strip().lower() or None
    outcome_result = str((outcome or {}).get("outcome_result") or "").strip().lower() or None
    row.update(
        {
            "outcome_status": outcome_status,
            "outcome_result": outcome_result,
            "exit_reason": str((outcome or {}).get("exit_reason") or "").strip() or None,
            "settled_at": str((outcome or {}).get("settled_at") or "").strip() or None,
            "exit_price": _safe_float((outcome or {}).get("exit_price")),
            "bars_observed": _safe_int((outcome or {}).get("bars_observed")),
            "bars_to_outcome": _safe_int((outcome or {}).get("bars_to_outcome")),
            "maturity_progress_pct": _safe_float((outcome or {}).get("maturity_progress_pct")),
            "rr_realized": _safe_float((outcome or {}).get("rr_realized")),
            "pnl_pct": _safe_float((outcome or {}).get("pnl_pct")),
            "mfe_pct": _safe_float((outcome or {}).get("mfe_pct")),
            "mae_pct": _safe_float((outcome or {}).get("mae_pct")),
            "feedback_group": outcome_status or "missing",
            "feedback_is_directional": True,
            "feedback_is_settled": outcome_status == "settled",
            "feedback_is_win": outcome_result == "win",
            "feedback_is_loss": outcome_result == "loss",
            "feedback_is_flat": outcome_result == "flat",
            "feedback_training_ready": outcome_status == "settled" and outcome_result in {"win", "loss", "flat"},
        }
    )
    return row


def build_telegram_alert_feedback_export(*, days=90, strategies=None, symbols=None, include_open=False, helpers, get_now):
    entries = helpers["read_telegram_alert_history"](
        days=days,
        strategies=strategies,
        symbols=symbols,
    )
    try:
        days_value = float(days) if days is not None else None
    except Exception:
        days_value = None

    generated_at = get_now().strftime("%Y-%m-%d %H:%M:%S")
    directional, outcomes = _resolve_directional_alert_outcomes(entries, helpers=helpers, get_now=get_now)
    outcomes_by_id = {}
    for row in outcomes:
        alert_id = str(row.get("alert_id") or "").strip()
        if alert_id:
            outcomes_by_id[alert_id] = row

    rows = []
    for entry in directional:
        alert_id = str(entry.get("alert_id") or "").strip()
        outcome = outcomes_by_id.get(alert_id) or {}
        feedback_row = _feedback_export_row(entry, outcome)
        if not include_open and not bool(feedback_row.get("feedback_training_ready")):
            continue
        rows.append(feedback_row)

    settled_rows = [row for row in rows if row.get("feedback_is_settled")]
    training_ready_rows = [row for row in rows if row.get("feedback_training_ready")]
    win_rows = [row for row in settled_rows if row.get("feedback_is_win")]
    loss_rows = [row for row in settled_rows if row.get("feedback_is_loss")]
    flat_rows = [row for row in settled_rows if row.get("feedback_is_flat")]

    by_strategy = {}
    by_policy_tier = {}
    for row in rows:
        strategy = str(row.get("strategy") or "UNKNOWN").strip().upper()
        policy_tier = str(row.get("entry_ai_policy_tier") or "unknown").strip().lower() or "unknown"
        for key, bucket_map in ((strategy, by_strategy), (policy_tier, by_policy_tier)):
            bucket = bucket_map.setdefault(
                key,
                {
                    "rows": 0,
                    "settled_rows": 0,
                    "training_ready_rows": 0,
                    "wins": 0,
                    "losses": 0,
                    "flats": 0,
                    "win_rate_pct": None,
                    "avg_rr_realized": None,
                    "avg_pnl_pct": None,
                    "_rows": [],
                },
            )
            bucket["rows"] += 1
            bucket["_rows"].append(row)
            if row.get("feedback_is_settled"):
                bucket["settled_rows"] += 1
            if row.get("feedback_training_ready"):
                bucket["training_ready_rows"] += 1
            if row.get("feedback_is_win"):
                bucket["wins"] += 1
            elif row.get("feedback_is_loss"):
                bucket["losses"] += 1
            elif row.get("feedback_is_flat"):
                bucket["flats"] += 1

    for bucket_map in (by_strategy, by_policy_tier):
        for key, bucket in list(bucket_map.items()):
            ready_rows = [row for row in bucket.pop("_rows", []) if row.get("feedback_training_ready")]
            bucket["avg_rr_realized"] = _realized_metric_average(ready_rows, "rr_realized")
            bucket["avg_pnl_pct"] = _realized_metric_average(ready_rows, "pnl_pct")
            if bucket["training_ready_rows"] > 0:
                bucket["win_rate_pct"] = (float(bucket["wins"]) / float(bucket["training_ready_rows"])) * 100.0
            bucket_map[key] = bucket

    summary = {
        "generated_at": generated_at,
        "window_days": days_value,
        "total_history_entries": len(entries or []),
        "eligible_directional_alerts": len(directional),
        "exported_rows": len(rows),
        "settled_rows": len(settled_rows),
        "training_ready_rows": len(training_ready_rows),
        "wins": len(win_rows),
        "losses": len(loss_rows),
        "flats": len(flat_rows),
        "include_open": bool(include_open),
        "win_rate_pct": (float(len(win_rows)) / float(len(training_ready_rows)) * 100.0) if training_ready_rows else None,
        "avg_rr_realized": _realized_metric_average(training_ready_rows, "rr_realized"),
        "avg_pnl_pct": _realized_metric_average(training_ready_rows, "pnl_pct"),
        "by_strategy": {key: by_strategy[key] for key in sorted(by_strategy.keys())},
        "by_entry_ai_policy_tier": {key: by_policy_tier[key] for key in sorted(by_policy_tier.keys())},
    }
    return {
        "generated_at": generated_at,
        "window_days": days_value,
        "fieldnames": alert_feedback_export_fieldnames(),
        "summary": summary,
        "rows": rows,
    }


def live_feedback_training_fieldnames():
    return [
        "checkpoint_at",
        "candidate_group",
        "candidate_rank",
        "timestamp",
        "alert_id",
        "strategy",
        "symbol",
        "signal",
        "timeframe",
        "evaluation_window_bars",
        "daily_pick",
        "alert_tier",
        "alert_tier_score",
        "tier_rank",
        "tier_action",
        "alert_mode",
        "confidence",
        "score",
        "source_count",
        "source_label",
        "strategy_label",
        "alert_intent",
        "alert_intent_reason",
        "ai_dispatch_label",
        "ai_dispatch_bucket",
        "ai_dispatch_reason",
        "ai_prob_win",
        "ai_expected_return_pct",
        "ai_rank_adjustment",
        "ai_runtime_status",
        "ai_runtime_reason",
        "entry_ai_label",
        "entry_ai_bucket",
        "entry_ai_reason",
        "entry_ai_policy_mode",
        "entry_ai_policy_tier",
        "entry_ai_premium_label",
        "entry_ai_standard_label",
        "entry_ai_watch_label",
        "entry_ai_strategy_policy",
        "entry_ai_prob_entry",
        "entry_ai_prob_watch",
        "entry_ai_prob_avoid",
        "entry_ai_model_type",
        "entry_ai_model_version",
        "entry_ai_model_trained_at",
        "entry_ai_feature_schema_version",
        "entry_ai_label_schema_version",
        "entry_ai_policy_schema_version",
        "entry_ai_rank_adjustment",
        "entry_ai_runtime_status",
        "entry_ai_runtime_reason",
        "entry_ai_runtime_threshold_adjustment",
        "entry_ai_runtime_base_min_confidence",
        "entry_ai_runtime_min_confidence",
        "entry_ai_runtime_threshold_reason",
        "short_trade_label",
        "short_trade_bucket",
        "short_trade_reason",
        "short_trade_score_adjustment",
        "short_trade_regime_aligned",
        "market_regime",
        "market_trend_bias",
        "symbol_regime",
        "side_bias",
        "regime_confidence",
        "regime_volatility_pct",
        "profile_runtime_threshold_applied",
        "profile_runtime_threshold_reason",
        "profile_runtime_market_regime",
        "profile_runtime_symbol_regime",
        "profile_runtime_side_bias",
        "profile_runtime_regime_alignment",
        "profile_runtime_freshness_bucket",
        "profile_runtime_bars_since_signal",
        "profile_runtime_min_confidence",
        "profile_runtime_min_score",
        "profile_runtime_min_win_rate_pct",
        "profile_runtime_min_expectancy_rr",
        "profile_runtime_min_trades",
        "profile_runtime_min_source_count",
        "profile_runtime_min_robustness_score",
        "sltp_live_label",
        "sltp_live_bucket",
        "sltp_live_reason",
        "sltp_live_score_adjustment",
        "entry_price",
        "price_at_checkpoint",
        "stop_loss",
        "take_profit",
        "risk_reward",
        "entry_gap_pct",
        "stop_risk_pct",
        "target_reward_pct",
        "rr_ratio",
        "detected_pattern",
        "forecast_direction",
        "forecast_score",
        "plan_reason",
        "bars_since_signal",
        "red_to_green_quality_score",
        "green_flip_reclaim",
        "min_confidence",
        "dynamic_min_confidence",
        "backtest_win_rate_pct",
        "backtest_expectancy_rr",
        "backtest_trades",
        "cache_key",
        "label_status",
        "label_filled",
        "label_win",
        "label_fill_bar",
        "label_exit_bar",
        "label_fill_timestamp",
        "label_exit_timestamp",
        "label_return_pct",
        "label_mfe_pct",
        "label_mae_pct",
        "label_mfe_r",
        "label_mae_r",
        "feedback_outcome_status",
        "feedback_outcome_result",
        "feedback_exit_reason",
        "feedback_settled_at",
        "feedback_exit_price",
        "feedback_bars_observed",
        "feedback_bars_to_outcome",
        "feedback_maturity_progress_pct",
    ]


def _derive_candidate_group_from_feedback_row(row):
    strategy = str(row.get("strategy") or "").strip().upper()
    if bool(row.get("daily_pick")) or strategy == "DAILY_BEST":
        return "daily"
    if strategy in {"TRADAR15", "TRENDRADAR15", "TREND_RADAR"}:
        return "trend_radar"
    if strategy in {"TRENDSTATE15", "TREND_STATE"}:
        return "trend_state"
    return "primary"


def _derive_tier_rank(value):
    text = str(value or "").strip().upper()
    if text == "S":
        return 5
    if text == "A":
        return 4
    if text == "B":
        return 3
    if text == "C":
        return 2
    if text:
        return 1
    return 0


def _feedback_price_at_checkpoint(row):
    entry_price = _safe_float(row.get("entry_price"), None)
    entry_gap_pct = _safe_float(row.get("sltp_live_entry_gap_pct"), None)
    if entry_price is None:
        return None
    if entry_gap_pct is None or entry_gap_pct <= 0:
        return entry_price
    return float(entry_price) * (1.0 + (float(entry_gap_pct) / 100.0))


def _feedback_training_row(row):
    stop_risk_pct = _safe_float(row.get("sltp_live_stop_risk_pct"), None)
    target_reward_pct = _safe_float(row.get("sltp_live_target_reward_pct"), None)
    entry_gap_pct = _safe_float(row.get("sltp_live_entry_gap_pct"), None)
    rr_ratio = _safe_float(row.get("sltp_live_rr_ratio"), None)
    pnl_pct = _safe_float(row.get("pnl_pct"), None)
    mfe_pct = _safe_float(row.get("mfe_pct"), None)
    mae_pct = _safe_float(row.get("mae_pct"), None)
    outcome_status = str(row.get("outcome_status") or "").strip().lower() or None
    outcome_result = str(row.get("outcome_result") or "").strip().lower() or None
    label_filled = outcome_status == "settled"
    label_win = True if outcome_result == "win" else False if label_filled else None

    label_mfe_r = None
    label_mae_r = None
    if isinstance(stop_risk_pct, (int, float)) and stop_risk_pct > 0:
        if isinstance(mfe_pct, (int, float)):
            label_mfe_r = float(mfe_pct) / float(stop_risk_pct)
        if isinstance(mae_pct, (int, float)):
            label_mae_r = -abs(float(mae_pct)) / float(stop_risk_pct)

    return {
        "checkpoint_at": str(row.get("timestamp") or "").strip() or None,
        "candidate_group": _derive_candidate_group_from_feedback_row(row),
        "candidate_rank": None,
        "timestamp": str(row.get("timestamp") or "").strip() or None,
        "alert_id": str(row.get("alert_id") or "").strip() or None,
        "strategy": str(row.get("strategy") or "").strip().upper() or None,
        "symbol": str(row.get("symbol") or "").strip().upper() or None,
        "signal": str(row.get("signal") or "").strip().upper() or None,
        "timeframe": str(row.get("timeframe") or "").strip().lower() or None,
        "evaluation_window_bars": _safe_int(row.get("evaluation_window_bars")),
        "daily_pick": bool(row.get("daily_pick")),
        "alert_tier": str(row.get("alert_tier") or "").strip() or None,
        "alert_tier_score": _safe_float(row.get("alert_tier_score")),
        "tier_rank": _derive_tier_rank(row.get("alert_tier")),
        "tier_action": str(row.get("tier_action") or "").strip() or None,
        "alert_mode": str(row.get("alert_mode") or "").strip() or None,
        "confidence": _safe_float(row.get("confidence")),
        "score": _safe_float(row.get("score")),
        "source_count": _safe_int(row.get("source_count"), 0),
        "source_label": str(row.get("source_label") or "").strip() or None,
        "strategy_label": str(row.get("strategy_label") or "").strip() or None,
        "alert_intent": str(row.get("alert_intent") or "").strip().lower() or None,
        "alert_intent_reason": str(row.get("alert_intent_reason") or "").strip() or None,
        "ai_dispatch_label": str(row.get("ai_dispatch_label") or "").strip() or None,
        "ai_dispatch_bucket": str(row.get("ai_dispatch_bucket") or "").strip().lower() or None,
        "ai_dispatch_reason": str(row.get("ai_dispatch_reason") or "").strip() or None,
        "ai_prob_win": _safe_float(row.get("ai_prob_win")),
        "ai_expected_return_pct": _safe_float(row.get("ai_expected_return_pct")),
        "ai_rank_adjustment": _safe_float(row.get("ai_rank_adjustment")),
        "ai_runtime_status": str(row.get("ai_runtime_status") or "").strip().lower() or None,
        "ai_runtime_reason": str(row.get("ai_runtime_reason") or "").strip() or None,
        "entry_ai_label": str(row.get("entry_ai_label") or "").strip() or None,
        "entry_ai_bucket": str(row.get("entry_ai_bucket") or "").strip().lower() or None,
        "entry_ai_reason": str(row.get("entry_ai_reason") or "").strip() or None,
        "entry_ai_policy_mode": str(row.get("entry_ai_policy_mode") or "").strip().lower() or None,
        "entry_ai_policy_tier": str(row.get("entry_ai_policy_tier") or "").strip().lower() or None,
        "entry_ai_premium_label": str(row.get("entry_ai_premium_label") or "").strip().lower() or None,
        "entry_ai_standard_label": str(row.get("entry_ai_standard_label") or "").strip().lower() or None,
        "entry_ai_watch_label": str(row.get("entry_ai_watch_label") or "").strip().lower() or None,
        "entry_ai_strategy_policy": str(row.get("entry_ai_strategy_policy") or "").strip().upper() or None,
        "entry_ai_prob_entry": _safe_float(row.get("entry_ai_prob_entry")),
        "entry_ai_prob_watch": _safe_float(row.get("entry_ai_prob_watch")),
        "entry_ai_prob_avoid": _safe_float(row.get("entry_ai_prob_avoid")),
        "entry_ai_model_type": str(row.get("entry_ai_model_type") or "").strip() or None,
        "entry_ai_model_version": str(row.get("entry_ai_model_version") or "").strip() or None,
        "entry_ai_model_trained_at": str(row.get("entry_ai_model_trained_at") or "").strip() or None,
        "entry_ai_feature_schema_version": str(row.get("entry_ai_feature_schema_version") or "").strip() or None,
        "entry_ai_label_schema_version": str(row.get("entry_ai_label_schema_version") or "").strip() or None,
        "entry_ai_policy_schema_version": str(row.get("entry_ai_policy_schema_version") or "").strip() or None,
        "entry_ai_rank_adjustment": _safe_float(row.get("entry_ai_rank_adjustment")),
        "entry_ai_runtime_status": str(row.get("entry_ai_runtime_status") or "").strip().lower() or None,
        "entry_ai_runtime_reason": str(row.get("entry_ai_runtime_reason") or "").strip() or None,
        "entry_ai_runtime_threshold_adjustment": _safe_float(row.get("entry_ai_runtime_threshold_adjustment")),
        "entry_ai_runtime_base_min_confidence": _safe_float(row.get("entry_ai_runtime_base_min_confidence")),
        "entry_ai_runtime_min_confidence": _safe_float(row.get("entry_ai_runtime_min_confidence")),
        "entry_ai_runtime_threshold_reason": str(row.get("entry_ai_runtime_threshold_reason") or "").strip() or None,
        "short_trade_label": str(row.get("short_trade_label") or "").strip() or None,
        "short_trade_bucket": str(row.get("short_trade_bucket") or "").strip().lower() or None,
        "short_trade_reason": str(row.get("short_trade_reason") or "").strip() or None,
        "short_trade_score_adjustment": _safe_float(row.get("short_trade_score_adjustment")),
        "short_trade_regime_aligned": bool(row.get("short_trade_regime_aligned")) if row.get("short_trade_regime_aligned") is not None else None,
        "market_regime": str(row.get("market_regime") or "").strip().upper() or None,
        "market_trend_bias": str(row.get("market_trend_bias") or "").strip().upper() or None,
        "symbol_regime": str(row.get("symbol_regime") or "").strip().upper() or None,
        "side_bias": str(row.get("side_bias") or "").strip().upper() or None,
        "regime_confidence": _safe_float(row.get("regime_confidence")),
        "regime_volatility_pct": _safe_float(row.get("regime_volatility_pct")),
        "profile_runtime_threshold_applied": bool(row.get("profile_runtime_threshold_applied")) if row.get("profile_runtime_threshold_applied") is not None else None,
        "profile_runtime_threshold_reason": str(row.get("profile_runtime_threshold_reason") or "").strip() or None,
        "profile_runtime_market_regime": str(row.get("profile_runtime_market_regime") or "").strip().upper() or None,
        "profile_runtime_symbol_regime": str(row.get("profile_runtime_symbol_regime") or "").strip().upper() or None,
        "profile_runtime_side_bias": str(row.get("profile_runtime_side_bias") or "").strip().upper() or None,
        "profile_runtime_regime_alignment": str(row.get("profile_runtime_regime_alignment") or "").strip().lower() or None,
        "profile_runtime_freshness_bucket": str(row.get("profile_runtime_freshness_bucket") or "").strip().lower() or None,
        "profile_runtime_bars_since_signal": _safe_float(row.get("profile_runtime_bars_since_signal")),
        "profile_runtime_min_confidence": _safe_float(row.get("profile_runtime_min_confidence")),
        "profile_runtime_min_score": _safe_float(row.get("profile_runtime_min_score")),
        "profile_runtime_min_win_rate_pct": _safe_float(row.get("profile_runtime_min_win_rate_pct")),
        "profile_runtime_min_expectancy_rr": _safe_float(row.get("profile_runtime_min_expectancy_rr")),
        "profile_runtime_min_trades": _safe_int(row.get("profile_runtime_min_trades")),
        "profile_runtime_min_source_count": _safe_int(row.get("profile_runtime_min_source_count")),
        "profile_runtime_min_robustness_score": _safe_float(row.get("profile_runtime_min_robustness_score")),
        "sltp_live_label": str(row.get("sltp_live_label") or "").strip() or None,
        "sltp_live_bucket": str(row.get("sltp_live_bucket") or "").strip().lower() or None,
        "sltp_live_reason": str(row.get("sltp_live_reason") or "").strip() or None,
        "sltp_live_score_adjustment": _safe_float(row.get("sltp_live_score_adjustment")),
        "entry_price": _safe_float(row.get("entry_price")),
        "price_at_checkpoint": _feedback_price_at_checkpoint(row),
        "stop_loss": _safe_float(row.get("stop_loss")),
        "take_profit": _safe_float(row.get("take_profit")),
        "risk_reward": _safe_float(row.get("risk_reward")),
        "entry_gap_pct": entry_gap_pct,
        "stop_risk_pct": stop_risk_pct,
        "target_reward_pct": target_reward_pct,
        "rr_ratio": rr_ratio,
        "detected_pattern": str(row.get("detected_pattern") or "").strip() or None,
        "forecast_direction": str(row.get("forecast_direction") or "").strip().upper() or None,
        "forecast_score": _safe_float(row.get("forecast_score")),
        "plan_reason": str(row.get("plan_reason") or "").strip() or None,
        "bars_since_signal": _safe_float(row.get("bars_since_signal")),
        "red_to_green_quality_score": _safe_float(row.get("red_to_green_quality_score")),
        "green_flip_reclaim": bool(row.get("green_flip_reclaim")) if row.get("green_flip_reclaim") is not None else None,
        "min_confidence": _safe_float(row.get("min_confidence")),
        "dynamic_min_confidence": _safe_float(row.get("dynamic_min_confidence")),
        "backtest_win_rate_pct": _safe_float(row.get("backtest_win_rate_pct")),
        "backtest_expectancy_rr": _safe_float(row.get("backtest_expectancy_rr")),
        "backtest_trades": _safe_float(row.get("backtest_trades")),
        "cache_key": str(row.get("cache_key") or "").strip() or None,
        "label_status": outcome_status or "unsupported",
        "label_filled": label_filled,
        "label_win": label_win,
        "label_fill_bar": 0 if label_filled else None,
        "label_exit_bar": _safe_int(row.get("bars_to_outcome")),
        "label_fill_timestamp": str(row.get("timestamp") or "").strip() or None,
        "label_exit_timestamp": str(row.get("settled_at") or "").strip() or None,
        "label_return_pct": pnl_pct,
        "label_mfe_pct": mfe_pct,
        "label_mae_pct": -abs(float(mae_pct)) if isinstance(mae_pct, (int, float)) else None,
        "label_mfe_r": label_mfe_r,
        "label_mae_r": label_mae_r,
        "feedback_outcome_status": outcome_status,
        "feedback_outcome_result": outcome_result,
        "feedback_exit_reason": str(row.get("exit_reason") or "").strip() or None,
        "feedback_settled_at": str(row.get("settled_at") or "").strip() or None,
        "feedback_exit_price": _safe_float(row.get("exit_price")),
        "feedback_bars_observed": _safe_int(row.get("bars_observed")),
        "feedback_bars_to_outcome": _safe_int(row.get("bars_to_outcome")),
        "feedback_maturity_progress_pct": _safe_float(row.get("maturity_progress_pct")),
    }


def build_live_feedback_training_dataset(*, days=90, strategies=None, symbols=None, include_open=False, helpers, get_now):
    feedback_payload = build_telegram_alert_feedback_export(
        days=days,
        strategies=strategies,
        symbols=symbols,
        include_open=include_open,
        helpers=helpers,
        get_now=get_now,
    )
    rows = [_feedback_training_row(row) for row in (feedback_payload.get("rows") or [])]
    training_ready_rows = [row for row in rows if bool(row.get("label_filled"))]
    wins = [row for row in training_ready_rows if row.get("label_win") is True]
    losses = [row for row in training_ready_rows if row.get("label_win") is False]

    by_group = {}
    by_strategy = {}
    for row in rows:
        for key, bucket_map in (
            (str(row.get("candidate_group") or "unknown"), by_group),
            (str(row.get("strategy") or "UNKNOWN").strip().upper(), by_strategy),
        ):
            bucket = bucket_map.setdefault(
                key,
                {
                    "rows": 0,
                    "filled_rows": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate_pct": None,
                    "avg_return_pct": None,
                    "avg_mfe_r": None,
                    "avg_mae_r": None,
                    "_rows": [],
                },
            )
            bucket["rows"] += 1
            bucket["_rows"].append(row)
            if row.get("label_filled"):
                bucket["filled_rows"] += 1
            if row.get("label_win") is True:
                bucket["wins"] += 1
            elif row.get("label_win") is False:
                bucket["losses"] += 1

    for bucket_map in (by_group, by_strategy):
        for key, bucket in list(bucket_map.items()):
            filled_rows = [row for row in bucket.pop("_rows", []) if row.get("label_filled")]
            if bucket["filled_rows"] > 0:
                bucket["win_rate_pct"] = (float(bucket["wins"]) / float(bucket["filled_rows"])) * 100.0
            bucket["avg_return_pct"] = _realized_metric_average(filled_rows, "label_return_pct")
            bucket["avg_mfe_r"] = _realized_metric_average(filled_rows, "label_mfe_r")
            bucket["avg_mae_r"] = _realized_metric_average(filled_rows, "label_mae_r")
            bucket_map[key] = bucket

    summary = {
        "generated_at": feedback_payload.get("generated_at"),
        "window_days": feedback_payload.get("window_days"),
        "total_rows": len(rows),
        "filled_rows": len(training_ready_rows),
        "wins": len(wins),
        "losses": len(losses),
        "fill_rate_pct": (float(len(training_ready_rows)) / float(len(rows)) * 100.0) if rows else 0.0,
        "win_rate_pct": (float(len(wins)) / float(len(training_ready_rows)) * 100.0) if training_ready_rows else None,
        "avg_return_pct": _realized_metric_average(training_ready_rows, "label_return_pct"),
        "avg_mfe_r": _realized_metric_average(training_ready_rows, "label_mfe_r"),
        "avg_mae_r": _realized_metric_average(training_ready_rows, "label_mae_r"),
        "by_candidate_group": {key: by_group[key] for key in sorted(by_group.keys())},
        "by_strategy": {key: by_strategy[key] for key in sorted(by_strategy.keys())},
    }
    return {
        "generated_at": feedback_payload.get("generated_at"),
        "window_days": feedback_payload.get("window_days"),
        "fieldnames": live_feedback_training_fieldnames(),
        "summary": summary,
        "rows": rows,
    }
