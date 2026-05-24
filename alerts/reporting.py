import csv
import hashlib
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
        "entry_price",
        "stop_loss",
        "take_profit",
        "risk_reward",
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
        for key in ("max_forward_bars", "time_stop_bars", "holding_window_bars"):
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
    entry_price = pick_plan_value(plan, ["entry_price", "current_price", "price"]) if isinstance(plan, dict) else None
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
    outcome = {
        "alert_id": alert_id,
        "timestamp": str((entry or {}).get("timestamp") or "").strip() or None,
        "strategy": str((entry or {}).get("strategy") or "").strip().upper() or "UNKNOWN",
        "symbol": str((entry or {}).get("symbol") or "").strip().upper(),
        "signal": signal,
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
    summary["eligible_directional_alerts"] = len(directional)

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


def record_telegram_run_report(
    *,
    results,
    kill,
    kill_reason,
    min_conf,
    dynamic_min_conf,
    candidates,
    sent_candidates,
    daily_pick_sent,
    daily_summary_sent,
    dropped_by_cache,
    dropped_by_symbol_cap,
    dropped_by_run_cap,
    quality_drop_counts,
    alert_budget,
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
        "sent_count": len(sent_candidates or []),
        "daily_pick_sent": int(daily_pick_sent or 0),
        "daily_summary_sent": int(daily_summary_sent or 0),
        "dropped_by_cache": int(dropped_by_cache or 0),
        "dropped_by_symbol_cap": int(dropped_by_symbol_cap or 0),
        "dropped_by_run_cap": int(dropped_by_run_cap or 0),
        "quality_drop_counts": dict(quality_drop_counts or {}),
        "alert_budget": dict(alert_budget or {}),
        "symbol_signal_mix": dict(by_symbol_signal),
        "top_candidates": [candidate_ops_snapshot_fn(row) for row in (candidates or [])[:top_n]],
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
    entry_price = pick_plan_value(plan, ["entry_price", "current_price", "price"]) if isinstance(plan, dict) else None
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
