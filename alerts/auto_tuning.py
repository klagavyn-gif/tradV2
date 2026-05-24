import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta

from .reporting import write_json_atomic


def _to_float(value, default=None):
    try:
        number = float(value)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


def _to_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def _clamp(value, lower=None, upper=None):
    if lower is not None:
        value = max(lower, value)
    if upper is not None:
        value = min(upper, value)
    return value


def _quantile(values, q, default=None):
    numeric = sorted(
        [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    )
    if not numeric:
        return default
    if len(numeric) == 1:
        return float(numeric[0])
    q = _clamp(float(q), 0.0, 1.0)
    pos = (len(numeric) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(numeric[low])
    weight = pos - low
    return float(numeric[low] * (1.0 - weight) + numeric[high] * weight)


def _timestamp_value(row):
    text = str((row or {}).get("timestamp") or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def read_alert_history_entries(path, *, days=None):
    if not os.path.exists(path):
        return []
    cutoff = None
    if isinstance(days, (int, float)) and float(days) > 0:
        cutoff = datetime.now() - timedelta(days=float(days))
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = str(raw_line or "").strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                ts_value = _timestamp_value(row)
                if cutoff is not None and isinstance(ts_value, datetime) and ts_value < cutoff:
                    continue
                row["_timestamp_obj"] = ts_value
                row["strategy"] = str(row.get("strategy") or "").strip().upper()
                row["signal"] = str(row.get("signal") or "").strip().upper()
                row["symbol"] = str(row.get("symbol") or "").strip().upper()
                rows.append(row)
    except Exception:
        return []
    rows.sort(key=lambda row: row.get("_timestamp_obj") or datetime.min)
    return rows


def load_auto_tuned_profiles(path):
    target = str(path or "").strip()
    if not target or not os.path.exists(target):
        return {}
    try:
        with open(target, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _bounded_value(base_value, tuned_value, *, lower_delta, upper_delta, absolute_lower=None, absolute_upper=None):
    if tuned_value is None:
        return None
    if isinstance(base_value, (int, float)) and math.isfinite(float(base_value)):
        tuned_value = _clamp(float(tuned_value), float(base_value) + float(lower_delta), float(base_value) + float(upper_delta))
    tuned_value = float(tuned_value)
    if absolute_lower is not None:
        tuned_value = max(float(absolute_lower), tuned_value)
    if absolute_upper is not None:
        tuned_value = min(float(absolute_upper), tuned_value)
    return float(tuned_value)


def _top_subset(rows, *, target_count, score_field="score", minimum_keep=4, minimum_ratio=0.25):
    if not rows:
        return []
    ranked = sorted(
        rows,
        key=lambda row: (
            _to_float(row.get(score_field), -1e9),
            _to_float(row.get("confidence"), -1e9),
            _to_float(row.get("backtest_win_rate_pct"), -1e9),
        ),
        reverse=True,
    )
    total = len(ranked)
    if not isinstance(target_count, (int, float)) or float(target_count) <= 0:
        keep_count = total
    else:
        keep_ratio = _clamp(float(target_count) / float(total), minimum_ratio, 1.0)
        keep_count = int(math.ceil(float(total) * keep_ratio))
    keep_count = max(int(minimum_keep), keep_count)
    keep_count = min(total, keep_count)
    return ranked[:keep_count]


def _build_side_tuned_profile(rows, *, base_profile, side_prefix, target_count):
    selected = _top_subset(rows, target_count=target_count)
    if not selected:
        return {}, {}

    base_conf = _to_float(base_profile.get(f"{side_prefix}min_confidence"), _to_float(base_profile.get("min_confidence")))
    base_score = _to_float(base_profile.get(f"{side_prefix}min_score"), _to_float(base_profile.get("min_score")))
    base_wr = _to_float(base_profile.get(f"{side_prefix}min_win_rate_pct"), _to_float(base_profile.get("min_win_rate_pct")))
    base_exp = _to_float(base_profile.get(f"{side_prefix}min_expectancy_rr"), _to_float(base_profile.get("min_expectancy_rr")))
    base_trades = _to_int(base_profile.get(f"{side_prefix}min_trades"), _to_int(base_profile.get("min_trades")))
    base_sources = _to_int(base_profile.get(f"{side_prefix}min_source_count"), _to_int(base_profile.get("min_source_count")))
    base_single_source = _to_float(
        base_profile.get(f"{side_prefix}single_source_min_confidence"),
        _to_float(base_profile.get("single_source_min_confidence")),
    )
    base_robustness = _to_float(
        base_profile.get(f"{side_prefix}min_robustness_score"),
        _to_float(base_profile.get("min_robustness_score")),
    )

    tuned = {}
    tuned[f"{side_prefix}min_confidence"] = _bounded_value(
        base_conf,
        _quantile([_to_float(row.get("confidence")) for row in selected], 0.10, default=base_conf),
        lower_delta=-4.0,
        upper_delta=10.0,
        absolute_lower=55.0,
        absolute_upper=95.0,
    )
    tuned[f"{side_prefix}min_score"] = _bounded_value(
        base_score,
        _quantile([_to_float(row.get("score")) for row in selected], 0.08, default=base_score),
        lower_delta=-6.0,
        upper_delta=12.0,
        absolute_lower=58.0,
        absolute_upper=98.0,
    )
    tuned[f"{side_prefix}min_win_rate_pct"] = _bounded_value(
        base_wr,
        _quantile([_to_float(row.get("backtest_win_rate_pct")) for row in selected], 0.12, default=base_wr),
        lower_delta=-2.0,
        upper_delta=6.0,
        absolute_lower=50.0,
        absolute_upper=75.0,
    )
    tuned[f"{side_prefix}min_expectancy_rr"] = _bounded_value(
        base_exp,
        _quantile([_to_float(row.get("backtest_expectancy_rr")) for row in selected], 0.15, default=base_exp),
        lower_delta=-0.03,
        upper_delta=0.08,
        absolute_lower=-0.02,
        absolute_upper=0.30,
    )
    tuned[f"{side_prefix}min_trades"] = int(
        round(
            _clamp(
                _bounded_value(
                    base_trades,
                    _quantile([_to_float(row.get("backtest_trades")) for row in selected], 0.10, default=base_trades),
                    lower_delta=-2.0,
                    upper_delta=6.0,
                    absolute_lower=4.0,
                    absolute_upper=40.0,
                )
                or float(base_trades or 6),
                4.0,
                40.0,
            )
        )
    )
    if isinstance(base_sources, int):
        source_median = _quantile([_to_float(row.get("source_count")) for row in selected], 0.50, default=float(base_sources))
        tuned[f"{side_prefix}min_source_count"] = int(max(1, round(source_median)))
    if isinstance(base_single_source, (int, float)):
        tuned[f"{side_prefix}single_source_min_confidence"] = _bounded_value(
            base_single_source,
            _quantile([_to_float(row.get("confidence")) for row in selected], 0.75, default=base_single_source),
            lower_delta=-2.0,
            upper_delta=6.0,
            absolute_lower=70.0,
            absolute_upper=95.0,
        )
    if isinstance(base_robustness, (int, float)):
        robustness = _quantile([_to_float(row.get("robustness_score")) for row in selected], 0.15, default=base_robustness)
        tuned[f"{side_prefix}min_robustness_score"] = _bounded_value(
            base_robustness,
            robustness,
            lower_delta=-3.0,
            upper_delta=8.0,
            absolute_lower=35.0,
            absolute_upper=80.0,
        )

    stats = {
        "alerts": len(rows),
        "selected_alerts": len(selected),
        "selected_score_floor": _quantile([_to_float(row.get("score")) for row in selected], 0.0),
        "selected_confidence_floor": _quantile([_to_float(row.get("confidence")) for row in selected], 0.0),
        "selected_win_rate_floor": _quantile([_to_float(row.get("backtest_win_rate_pct")) for row in selected], 0.0),
        "selected_expectancy_floor": _quantile([_to_float(row.get("backtest_expectancy_rr")) for row in selected], 0.0),
    }
    return {k: v for k, v in tuned.items() if v is not None}, stats


def _build_symbol_tuned_profiles(entries, *, base_symbol_profiles, observed_days, min_alerts_per_symbol, target_alerts_per_day):
    directional = [
        row for row in entries
        if row.get("strategy") not in ("DAILY_BEST", "DAILY_SUMMARY")
        and row.get("signal") in ("BUY", "SELL")
        and row.get("symbol")
    ]
    grouped = defaultdict(list)
    for row in directional:
        grouped[str(row.get("symbol") or "").strip().upper()].append(row)

    tuned_profiles = {}
    stats = {}
    for symbol, rows in grouped.items():
        if len(rows) < int(min_alerts_per_symbol):
            continue
        base_profile = dict(base_symbol_profiles.get(symbol) or {})
        total_target = float(target_alerts_per_day) * float(observed_days)
        buy_rows = [row for row in rows if row.get("signal") == "BUY"]
        sell_rows = [row for row in rows if row.get("signal") == "SELL"]
        side_min_rows = max(2, int(math.ceil(float(min_alerts_per_symbol) / 3.0)))
        tuned = {}
        symbol_stats = {
            "alerts": len(rows),
            "alerts_per_day": round(float(len(rows)) / float(observed_days), 4) if observed_days > 0 else None,
        }
        if len(buy_rows) >= side_min_rows:
            buy_target = max(3.0, total_target * (float(len(buy_rows)) / float(len(rows))))
            buy_tuned, buy_stats = _build_side_tuned_profile(
                buy_rows,
                base_profile=base_profile,
                side_prefix="buy_",
                target_count=buy_target,
            )
            tuned.update(buy_tuned)
            symbol_stats["buy"] = buy_stats
        if len(sell_rows) >= side_min_rows:
            sell_target = max(3.0, total_target * (float(len(sell_rows)) / float(len(rows))))
            sell_tuned, sell_stats = _build_side_tuned_profile(
                sell_rows,
                base_profile=base_profile,
                side_prefix="sell_",
                target_count=sell_target,
            )
            tuned.update(sell_tuned)
            symbol_stats["sell"] = sell_stats
        if tuned:
            tuned_profiles[symbol] = tuned
            stats[symbol] = symbol_stats
    return tuned_profiles, stats


def _build_strategy_tuned_profiles(entries, *, base_strategy_profiles, observed_days, min_alerts_per_strategy, target_total_alerts_per_day):
    directional = [row for row in entries if row.get("signal") in ("BUY", "SELL") and row.get("strategy") not in ("DAILY_SUMMARY",)]
    grouped = defaultdict(list)
    for row in directional:
        grouped[str(row.get("strategy") or "").strip().upper()].append(row)

    tuned_profiles = {}
    stats = {}
    for strategy, rows in grouped.items():
        if len(rows) < int(min_alerts_per_strategy):
            continue
        base_profile = dict(base_strategy_profiles.get(strategy) or {})
        if not base_profile:
            continue
        total_target = float(target_total_alerts_per_day) * float(observed_days)
        tuned = {}
        strategy_stats = {
            "alerts": len(rows),
            "alerts_per_day": round(float(len(rows)) / float(observed_days), 4) if observed_days > 0 else None,
        }
        buy_rows = [row for row in rows if row.get("signal") == "BUY"]
        sell_rows = [row for row in rows if row.get("signal") == "SELL"]
        if buy_rows:
            buy_target = max(3.0, total_target * (float(len(buy_rows)) / float(len(rows))))
            buy_tuned, buy_stats = _build_side_tuned_profile(
                buy_rows,
                base_profile=base_profile,
                side_prefix="buy_",
                target_count=buy_target,
            )
            tuned.update(buy_tuned)
            strategy_stats["buy"] = buy_stats
        if sell_rows:
            sell_target = max(3.0, total_target * (float(len(sell_rows)) / float(len(rows))))
            sell_tuned, sell_stats = _build_side_tuned_profile(
                sell_rows,
                base_profile=base_profile,
                side_prefix="sell_",
                target_count=sell_target,
            )
            tuned.update(sell_tuned)
            strategy_stats["sell"] = sell_stats
        if tuned:
            tuned_profiles[strategy] = tuned
            stats[strategy] = strategy_stats
    return tuned_profiles, stats


def _build_cdc_daily_best_tuned_profiles(entries, *, base_cdc_profiles, observed_days, min_rows, target_daily_pick_alerts_per_day):
    cdc_rows = []
    for row in entries:
        if str(row.get("signal") or "").strip().upper() != "BUY":
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        red_to_green_score = _to_float(row.get("red_to_green_quality_score"))
        if red_to_green_score is None:
            source_label = str(row.get("source_label") or "")
            strategy_label = str(row.get("strategy_label") or "")
            if "CDC+VixFix 15m" not in source_label and "CDC+VixFix 15m" not in strategy_label:
                continue
        cdc_rows.append(row)

    grouped = defaultdict(list)
    for row in cdc_rows:
        grouped[str(row.get("symbol") or "").strip().upper()].append(row)

    tuned_profiles = {}
    stats = {}
    for symbol, rows in grouped.items():
        if len(rows) < int(min_rows):
            continue
        base_profile = dict(base_cdc_profiles.get(symbol) or {})
        base_score = _to_float(base_profile.get("daily_best_min_red_to_green_score"), 80.0)
        target_count = float(target_daily_pick_alerts_per_day) * float(observed_days)
        selected = _top_subset(rows, target_count=max(2.0, target_count), score_field="red_to_green_quality_score", minimum_keep=2, minimum_ratio=0.30)
        red_scores = [_to_float(row.get("red_to_green_quality_score")) for row in selected]
        bars_since = [_to_float(row.get("bars_since_signal")) for row in selected]
        reclaim_rate = sum(1 for row in selected if bool(row.get("green_flip_reclaim"))) / float(len(selected)) if selected else 0.0
        tuned = {
            "daily_best_min_red_to_green_score": _bounded_value(
                base_score,
                _quantile(red_scores, 0.10, default=base_score),
                lower_delta=-4.0,
                upper_delta=12.0,
                absolute_lower=68.0,
                absolute_upper=95.0,
            ),
            "daily_best_require_reclaim": bool(base_profile.get("daily_best_require_reclaim", True) or reclaim_rate >= 0.55),
            "daily_best_max_bars_since_flip": int(
                round(
                    _clamp(
                        _quantile(bars_since, 0.75, default=float(base_profile.get("daily_best_max_bars_since_flip", 3))),
                        0.0,
                        3.0,
                    )
                )
            ),
        }
        tuned_profiles[symbol] = tuned
        stats[symbol] = {
            "alerts": len(rows),
            "selected_alerts": len(selected),
            "alerts_per_day": round(float(len(rows)) / float(observed_days), 4) if observed_days > 0 else None,
            "reclaim_rate": round(reclaim_rate * 100.0, 2),
            "selected_red_to_green_floor": _quantile(red_scores, 0.0),
        }
    return tuned_profiles, stats


def build_auto_tuned_thresholds(
    *,
    entries,
    base_strategy_profiles,
    base_symbol_profiles,
    base_cdc_profiles,
    history_days,
    min_alerts_per_symbol,
    min_alerts_per_strategy,
    target_alerts_per_day,
    target_daily_pick_alerts_per_day,
):
    directional = [row for row in (entries or []) if isinstance(row, dict) and str(row.get("signal") or "").upper() in ("BUY", "SELL")]
    timestamps = [row.get("_timestamp_obj") for row in directional if isinstance(row.get("_timestamp_obj"), datetime)]
    if timestamps:
        observed_days = max(1, (max(timestamps).date() - min(timestamps).date()).days + 1)
    else:
        observed_days = max(1, int(history_days or 30))

    tuned_strategy_profiles, strategy_stats = _build_strategy_tuned_profiles(
        directional,
        base_strategy_profiles=base_strategy_profiles,
        observed_days=observed_days,
        min_alerts_per_strategy=min_alerts_per_strategy,
        target_total_alerts_per_day=target_alerts_per_day,
    )
    tuned_symbol_profiles, symbol_stats = _build_symbol_tuned_profiles(
        directional,
        base_symbol_profiles=base_symbol_profiles,
        observed_days=observed_days,
        min_alerts_per_symbol=min_alerts_per_symbol,
        target_alerts_per_day=target_alerts_per_day,
    )
    tuned_cdc_profiles, cdc_stats = _build_cdc_daily_best_tuned_profiles(
        directional,
        base_cdc_profiles=base_cdc_profiles,
        observed_days=observed_days,
        min_rows=max(2, int(math.ceil(float(min_alerts_per_symbol) / 4.0))),
        target_daily_pick_alerts_per_day=target_daily_pick_alerts_per_day,
    )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "history_days_requested": int(history_days or 0),
        "history_rows": len(entries or []),
        "directional_rows": len(directional),
        "observed_days": int(observed_days),
        "telegram_alert_strategy_quality_profiles": tuned_strategy_profiles,
        "telegram_alert_symbol_quality_profiles": tuned_symbol_profiles,
        "cdc_vixfix_symbol_profiles": tuned_cdc_profiles,
        "stats": {
            "strategies": strategy_stats,
            "symbols": symbol_stats,
            "cdc_daily_best": cdc_stats,
        },
    }


def write_auto_tuned_thresholds(path, payload):
    return write_json_atomic(path, payload)
