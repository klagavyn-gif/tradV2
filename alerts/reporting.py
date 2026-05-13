import csv
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime


def alert_history_csv_fieldnames():
    return [
        "timestamp",
        "strategy",
        "symbol",
        "signal",
        "alert_tier",
        "alert_tier_score",
        "tier_action",
        "alert_mode",
        "confidence",
        "score",
        "daily_pick",
        "source_label",
        "strategy_label",
        "entry_price",
        "stop_loss",
        "take_profit",
        "risk_reward",
        "detected_pattern",
        "forecast_direction",
        "plan_reason",
        "min_confidence",
        "dynamic_min_confidence",
        "backtest_win_rate_pct",
        "backtest_expectancy_rr",
        "backtest_trades",
        "cache_key",
        "message_plain",
    ]


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
        "timestamp": get_now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": str(candidate.get("strategy") or "UNKNOWN").strip().upper(),
        "symbol": normalize_symbol(candidate.get("symbol") or ""),
        "signal": str(candidate.get("signal") or "").strip().upper(),
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
        "plan_reason": str(plan.get("reason") or "").strip() if isinstance(plan, dict) else None,
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
    ordered_keys = list(strategy_order) + sorted([key for key in by_strategy.keys() if key not in strategy_order])
    table = []
    examples_by_strategy = {}
    for strategy in ordered_keys:
        bucket = by_strategy.get(strategy)
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
    }


def build_telegram_alert_live_preview(results, *, limit_examples_per_strategy=1, config, helpers, get_now, strategy_order):
    try:
        limit_examples_per_strategy = max(1, int(limit_examples_per_strategy))
    except Exception:
        limit_examples_per_strategy = 1
    min_conf = getattr(config, "TELEGRAM_ALERT_MIN_CONFIDENCE", 72.0)
    try:
        min_conf = float(min_conf)
    except Exception:
        min_conf = 72.0

    telegram_kill_switch_state = helpers["telegram_kill_switch_state"]
    telegram_dynamic_conf_threshold = helpers["telegram_dynamic_conf_threshold"]
    build_telegram_candidates = helpers["build_telegram_candidates"]
    build_cdc_daily_trend_candidates = helpers["build_cdc_daily_trend_candidates"]
    build_daily_best_pick_candidates = helpers["build_daily_best_pick_candidates"]
    normalize_symbol = helpers["normalize_symbol"]
    candidate_backtest_snapshot_fn = helpers["candidate_backtest_snapshot"]

    kill, reason = telegram_kill_switch_state(results)
    dynamic_min_conf = telegram_dynamic_conf_threshold(min_conf, results)
    candidates = []
    build_stats = {}
    if not kill:
        candidates, build_stats = build_telegram_candidates(results, dynamic_min_conf)
        cdc_daily_candidates = build_cdc_daily_trend_candidates(results, existing_candidates=candidates, min_conf=dynamic_min_conf)
        if cdc_daily_candidates:
            candidates.extend([row for row in cdc_daily_candidates if isinstance(row, dict)])
    daily_candidates = build_daily_best_pick_candidates(results)
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
    return {
        "generated_at": get_now().strftime("%Y-%m-%d %H:%M:%S"),
        "kill_switch_active": bool(kill),
        "kill_switch_reason": str(reason or "") if kill else None,
        "min_confidence": float(min_conf),
        "dynamic_min_confidence": float(dynamic_min_conf),
        "candidate_count": len(combined),
        "count_by_strategy": dict(by_strategy),
        "quality_drop_counts": quality_drop_counts or {},
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
        "artifact_type": "verify_output",
        "includes_results": bool(include_results),
    }
    if include_results:
        payload["results"] = clean_json_value(results or [])
    return write_json_atomic(output_path, payload)
