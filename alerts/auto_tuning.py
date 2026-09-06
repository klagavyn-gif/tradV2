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


def _weighted_quantile(values, weights, q, default=None):
    pairs = []
    for value, weight in zip(values or [], weights or []):
        numeric = _to_float(value, None)
        if numeric is None:
            continue
        numeric_weight = _to_float(weight, 1.0)
        if numeric_weight is None:
            numeric_weight = 1.0
        pairs.append((float(numeric), max(0.0, float(numeric_weight))))
    if not pairs:
        return default
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0.0:
        return _quantile([value for value, _ in pairs], q, default=default)
    pairs.sort(key=lambda row: row[0])
    q = _clamp(float(q), 0.0, 1.0)
    threshold = float(total_weight) * q
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(pairs[-1][0])


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


def _bars_since_signal_value(row):
    if not isinstance(row, dict):
        return None
    for key in ("bars_since_signal", "bars_since_entry", "bars_since_cross"):
        value = _to_float(row.get(key), None)
        if value is not None:
            return float(value)
    return None


def _row_recency_weight(row, *, half_life_days):
    half_life = _to_float(half_life_days, None)
    ts_value = (row or {}).get("_timestamp_obj")
    if half_life is None or half_life <= 0 or not isinstance(ts_value, datetime):
        return 1.0
    age_days = max(0.0, (datetime.now() - ts_value).total_seconds() / 86400.0)
    return max(0.1, math.pow(0.5, age_days / float(half_life)))


def _row_freshness_weight(row, *, fresh_signal_max_bars, stale_signal_start_bars, stale_signal_min_weight):
    bars_since = _bars_since_signal_value(row)
    fresh_bars = _to_float(fresh_signal_max_bars, None)
    stale_bars = _to_float(stale_signal_start_bars, None)
    min_weight = _clamp(_to_float(stale_signal_min_weight, 0.45), 0.05, 1.0)
    if bars_since is None or fresh_bars is None or stale_bars is None:
        return 1.0
    if stale_bars <= fresh_bars:
        return min_weight if bars_since > fresh_bars else 1.0
    if bars_since <= fresh_bars:
        return 1.0
    if bars_since >= stale_bars:
        return min_weight
    progress = (float(bars_since) - float(fresh_bars)) / float(stale_bars - fresh_bars)
    return float(1.0 - progress * (1.0 - float(min_weight)))


def _row_selection_weight(
    row,
    *,
    recent_half_life_days,
    fresh_signal_max_bars,
    stale_signal_start_bars,
    stale_signal_min_weight,
):
    return float(
        _row_recency_weight(row, half_life_days=recent_half_life_days)
        * _row_freshness_weight(
            row,
            fresh_signal_max_bars=fresh_signal_max_bars,
            stale_signal_start_bars=stale_signal_start_bars,
            stale_signal_min_weight=stale_signal_min_weight,
        )
    )


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


def _blend_value(base_value, tuned_value, weight):
    if tuned_value is None:
        return None
    if not isinstance(base_value, (int, float)) or not math.isfinite(float(base_value)):
        return tuned_value
    if not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
        weight = 1.0
    weight = _clamp(float(weight), 0.0, 1.0)
    return float(base_value) + (float(tuned_value) - float(base_value)) * weight


def _sample_blend_weight(alert_count, *, min_alerts, full_weight_alerts, min_weight):
    try:
        alert_count = int(alert_count)
    except Exception:
        return _clamp(float(min_weight), 0.0, 1.0)
    try:
        min_alerts = int(min_alerts)
    except Exception:
        min_alerts = 12
    try:
        full_weight_alerts = int(full_weight_alerts)
    except Exception:
        full_weight_alerts = max(min_alerts, 36)
    try:
        min_weight = float(min_weight)
    except Exception:
        min_weight = 0.15
    if full_weight_alerts <= min_alerts:
        return 1.0
    raw_weight = (float(alert_count) - float(min_alerts)) / float(full_weight_alerts - min_alerts)
    return _clamp(raw_weight, _clamp(min_weight, 0.0, 1.0), 1.0)


def _min_strategy_side_value(tuned_strategy_profiles, key):
    values = []
    for profile in (tuned_strategy_profiles or {}).values():
        if not isinstance(profile, dict):
            continue
        value = _to_float(profile.get(key))
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return min(values)


def _shrink_symbol_tuned_profiles(
    tuned_symbol_profiles,
    *,
    symbol_stats,
    base_symbol_profiles,
    tuned_strategy_profiles,
    min_alerts_per_symbol,
    full_weight_alerts,
    min_blend_weight,
    confidence_cap_over_strategy,
    sell_win_rate_cap_over_base,
):
    adjusted_profiles = {}
    adjusted_stats = {}
    metric_suffixes = (
        "min_confidence",
        "min_score",
        "min_win_rate_pct",
        "min_expectancy_rr",
        "min_trades",
        "single_source_min_confidence",
        "min_robustness_score",
    )
    for symbol, profile in (tuned_symbol_profiles or {}).items():
        if not isinstance(profile, dict):
            continue
        base_profile = dict((base_symbol_profiles or {}).get(symbol) or {})
        stats = dict((symbol_stats or {}).get(symbol) or {})
        adjusted = dict(profile)
        for side in ("buy", "sell"):
            side_stats = dict(stats.get(side) or {})
            side_alerts = _to_int(side_stats.get("alerts"), _to_int(stats.get("alerts"), min_alerts_per_symbol))
            blend_weight = _sample_blend_weight(
                side_alerts,
                min_alerts=min_alerts_per_symbol,
                full_weight_alerts=full_weight_alerts,
                min_weight=min_blend_weight,
            )
            side_stats["blend_weight"] = round(float(blend_weight), 4)
            for suffix in metric_suffixes:
                key = f"{side}_{suffix}"
                if key not in adjusted:
                    continue
                tuned_value = adjusted.get(key)
                base_value = base_profile.get(key)
                blended = _blend_value(base_value, tuned_value, blend_weight)
                if suffix in ("min_trades",):
                    adjusted[key] = int(max(1, round(_to_float(blended, _to_float(tuned_value, 1.0)))))
                else:
                    adjusted[key] = float(blended) if isinstance(blended, (int, float)) else tuned_value
            conf_key = f"{side}_min_confidence"
            if conf_key in adjusted:
                strategy_anchor = _min_strategy_side_value(tuned_strategy_profiles, conf_key)
                if isinstance(strategy_anchor, (int, float)):
                    capped_conf = min(float(adjusted.get(conf_key)), float(strategy_anchor) + float(confidence_cap_over_strategy))
                    if capped_conf != float(adjusted.get(conf_key)):
                        side_stats["confidence_cap_anchor"] = float(strategy_anchor)
                        side_stats["confidence_cap_applied"] = float(capped_conf)
                    adjusted[conf_key] = float(capped_conf)
            wr_key = f"{side}_min_win_rate_pct"
            if side == "sell" and wr_key in adjusted:
                base_wr = _to_float(base_profile.get(wr_key))
                if isinstance(base_wr, (int, float)):
                    capped_wr = min(float(adjusted.get(wr_key)), float(base_wr) + float(sell_win_rate_cap_over_base))
                    if capped_wr != float(adjusted.get(wr_key)):
                        side_stats["sell_win_rate_cap_base"] = float(base_wr)
                        side_stats["sell_win_rate_cap_applied"] = float(capped_wr)
                    adjusted[wr_key] = float(capped_wr)
            if side_stats:
                stats[side] = side_stats
        adjusted_profiles[symbol] = adjusted
        adjusted_stats[symbol] = stats
    return adjusted_profiles, adjusted_stats


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


def _realized_metrics(rows):
    settled = 0
    wins = 0
    rr_sum = 0.0
    rr_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        win = row.get("_realized_win")
        rr = _to_float(row.get("_realized_rr"))
        has_win = isinstance(win, bool)
        has_rr = rr is not None
        if not has_win and not has_rr:
            continue
        settled += 1
        if has_win and win:
            wins += 1
        if has_rr:
            rr_sum += rr
            rr_count += 1
    win_rate = (float(wins) / float(settled) * 100.0) if settled > 0 else None
    expectancy = (rr_sum / rr_count) if rr_count > 0 else None
    return settled, win_rate, expectancy


def _realized_selectivity_uplift(rows, *, realized_tuning):
    if not isinstance(realized_tuning, dict) or not bool(realized_tuning.get("enable", True)):
        return 0.0, 0.0, {}
    settled, win_rate, expectancy = _realized_metrics(rows)
    if not settled:
        return 0.0, 0.0, {"realized_settled": 0}
    min_settled = _to_int(realized_tuning.get("min_settled"), 6)
    full_settled = _to_int(realized_tuning.get("full_settled"), 20)
    weight = _sample_blend_weight(
        settled,
        min_alerts=min_settled,
        full_weight_alerts=full_settled,
        min_weight=0.0,
    )
    target_wr = _to_float(realized_tuning.get("target_win_rate"), 55.0)
    wr_uplift_per_point = _to_float(realized_tuning.get("wr_uplift_per_point"), 0.5)
    exp_uplift_per_r = _to_float(realized_tuning.get("exp_uplift_per_r"), 1.5)

    wr_uplift = 0.0
    if win_rate is not None and target_wr is not None and win_rate < target_wr:
        wr_uplift = (target_wr - win_rate) * wr_uplift_per_point * weight
    exp_uplift = 0.0
    if expectancy is not None and expectancy < 0.0:
        exp_uplift = (-expectancy) * exp_uplift_per_r * weight

    stats = {
        "realized_settled": settled,
        "realized_win_rate_pct": round(win_rate, 3) if win_rate is not None else None,
        "realized_expectancy_rr": round(expectancy, 4) if expectancy is not None else None,
        "realized_blend_weight": round(weight, 4),
        "realized_wr_uplift": round(wr_uplift, 4),
        "realized_exp_uplift": round(exp_uplift, 4),
    }
    return wr_uplift, exp_uplift, stats


def _build_side_tuned_profile(
    rows,
    *,
    base_profile,
    side_prefix,
    target_count,
    recent_half_life_days,
    fresh_signal_max_bars,
    stale_signal_start_bars,
    stale_signal_min_weight,
    realized_tuning=None,
):
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
    selected_weights = [
        _row_selection_weight(
            row,
            recent_half_life_days=recent_half_life_days,
            fresh_signal_max_bars=fresh_signal_max_bars,
            stale_signal_start_bars=stale_signal_start_bars,
            stale_signal_min_weight=stale_signal_min_weight,
        )
        for row in selected
    ]
    fresh_selected = sum(
        1
        for row in selected
        if isinstance(_bars_since_signal_value(row), (int, float))
        and float(_bars_since_signal_value(row)) <= float(_to_float(fresh_signal_max_bars, 48.0))
    )

    tuned = {}
    tuned[f"{side_prefix}min_confidence"] = _bounded_value(
        base_conf,
        _weighted_quantile(
            [_to_float(row.get("confidence")) for row in selected],
            selected_weights,
            0.10,
            default=base_conf,
        ),
        lower_delta=-4.0,
        upper_delta=10.0,
        absolute_lower=55.0,
        absolute_upper=95.0,
    )
    tuned[f"{side_prefix}min_score"] = _bounded_value(
        base_score,
        _weighted_quantile(
            [_to_float(row.get("score")) for row in selected],
            selected_weights,
            0.08,
            default=base_score,
        ),
        lower_delta=-6.0,
        upper_delta=12.0,
        absolute_lower=58.0,
        absolute_upper=98.0,
    )
    realized_wr_uplift, realized_exp_uplift, realized_stats = _realized_selectivity_uplift(
        selected,
        realized_tuning=realized_tuning,
    )
    tuned[f"{side_prefix}min_win_rate_pct"] = _bounded_value(
        base_wr,
        _weighted_quantile(
            [_to_float(row.get("backtest_win_rate_pct")) for row in selected],
            selected_weights,
            0.12,
            default=base_wr,
        )
        + realized_wr_uplift,
        lower_delta=-2.0,
        upper_delta=6.0,
        absolute_lower=50.0,
        absolute_upper=75.0,
    )
    tuned[f"{side_prefix}min_expectancy_rr"] = _bounded_value(
        base_exp,
        _weighted_quantile(
            [_to_float(row.get("backtest_expectancy_rr")) for row in selected],
            selected_weights,
            0.15,
            default=base_exp,
        )
        + realized_exp_uplift,
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
                    _weighted_quantile(
                        [_to_float(row.get("backtest_trades")) for row in selected],
                        selected_weights,
                        0.10,
                        default=base_trades,
                    ),
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
        source_median = _weighted_quantile(
            [_to_float(row.get("source_count")) for row in selected],
            selected_weights,
            0.50,
            default=float(base_sources),
        )
        tuned[f"{side_prefix}min_source_count"] = int(max(1, round(source_median)))
    if isinstance(base_single_source, (int, float)):
        tuned[f"{side_prefix}single_source_min_confidence"] = _bounded_value(
            base_single_source,
            _weighted_quantile(
                [_to_float(row.get("confidence")) for row in selected],
                selected_weights,
                0.75,
                default=base_single_source,
            ),
            lower_delta=-2.0,
            upper_delta=6.0,
            absolute_lower=70.0,
            absolute_upper=95.0,
        )
    if isinstance(base_robustness, (int, float)):
        robustness = _weighted_quantile(
            [_to_float(row.get("robustness_score")) for row in selected],
            selected_weights,
            0.15,
            default=base_robustness,
        )
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
        "selected_weight_total": round(sum(selected_weights), 4),
        "fresh_signal_ratio": round(float(fresh_selected) / float(len(selected)), 4) if selected else None,
        "recent_weighted_confidence_floor": _weighted_quantile(
            [_to_float(row.get("confidence")) for row in selected],
            selected_weights,
            0.10,
        ),
        "recent_weighted_win_rate_floor": _weighted_quantile(
            [_to_float(row.get("backtest_win_rate_pct")) for row in selected],
            selected_weights,
            0.12,
        ),
        "recent_weighted_expectancy_floor": _weighted_quantile(
            [_to_float(row.get("backtest_expectancy_rr")) for row in selected],
            selected_weights,
            0.15,
        ),
    }
    if isinstance(realized_stats, dict):
        stats.update(realized_stats)
    return {k: v for k, v in tuned.items() if v is not None}, stats


def _build_symbol_tuned_profiles(
    entries,
    *,
    base_symbol_profiles,
    observed_days,
    min_alerts_per_symbol,
    target_alerts_per_day,
    recent_half_life_days,
    fresh_signal_max_bars,
    stale_signal_start_bars,
    stale_signal_min_weight,
    realized_tuning=None,
):
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
                recent_half_life_days=recent_half_life_days,
                fresh_signal_max_bars=fresh_signal_max_bars,
                stale_signal_start_bars=stale_signal_start_bars,
                stale_signal_min_weight=stale_signal_min_weight,
                realized_tuning=realized_tuning,
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
                recent_half_life_days=recent_half_life_days,
                fresh_signal_max_bars=fresh_signal_max_bars,
                stale_signal_start_bars=stale_signal_start_bars,
                stale_signal_min_weight=stale_signal_min_weight,
                realized_tuning=realized_tuning,
            )
            tuned.update(sell_tuned)
            symbol_stats["sell"] = sell_stats
        if tuned:
            tuned_profiles[symbol] = tuned
            stats[symbol] = symbol_stats
    return tuned_profiles, stats


def _build_strategy_tuned_profiles(
    entries,
    *,
    base_strategy_profiles,
    observed_days,
    min_alerts_per_strategy,
    target_total_alerts_per_day,
    recent_half_life_days,
    fresh_signal_max_bars,
    stale_signal_start_bars,
    stale_signal_min_weight,
    realized_tuning=None,
):
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
                recent_half_life_days=recent_half_life_days,
                fresh_signal_max_bars=fresh_signal_max_bars,
                stale_signal_start_bars=stale_signal_start_bars,
                stale_signal_min_weight=stale_signal_min_weight,
                realized_tuning=realized_tuning,
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
                recent_half_life_days=recent_half_life_days,
                fresh_signal_max_bars=fresh_signal_max_bars,
                stale_signal_start_bars=stale_signal_start_bars,
                stale_signal_min_weight=stale_signal_min_weight,
                realized_tuning=realized_tuning,
            )
            tuned.update(sell_tuned)
            strategy_stats["sell"] = sell_stats
        if tuned:
            tuned_profiles[strategy] = tuned
            stats[strategy] = strategy_stats
    return tuned_profiles, stats


def _build_cdc_daily_best_tuned_profiles(
    entries,
    *,
    base_cdc_profiles,
    observed_days,
    min_rows,
    target_daily_pick_alerts_per_day,
    recent_half_life_days,
    fresh_signal_max_bars,
    stale_signal_start_bars,
    stale_signal_min_weight,
):
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
        selected_weights = [
            _row_selection_weight(
                row,
                recent_half_life_days=recent_half_life_days,
                fresh_signal_max_bars=fresh_signal_max_bars,
                stale_signal_start_bars=stale_signal_start_bars,
                stale_signal_min_weight=stale_signal_min_weight,
            )
            for row in selected
        ]
        red_scores = [_to_float(row.get("red_to_green_quality_score")) for row in selected]
        bars_since = [_to_float(row.get("bars_since_signal")) for row in selected]
        reclaim_rate = sum(1 for row in selected if bool(row.get("green_flip_reclaim"))) / float(len(selected)) if selected else 0.0
        tuned = {
            "daily_best_min_red_to_green_score": _bounded_value(
                base_score,
                _weighted_quantile(red_scores, selected_weights, 0.10, default=base_score),
                lower_delta=-4.0,
                upper_delta=12.0,
                absolute_lower=68.0,
                absolute_upper=95.0,
            ),
            "daily_best_require_reclaim": bool(base_profile.get("daily_best_require_reclaim", True) or reclaim_rate >= 0.55),
            "daily_best_max_bars_since_flip": int(
                round(
                    _clamp(
                        _weighted_quantile(
                            bars_since,
                            selected_weights,
                            0.75,
                            default=float(base_profile.get("daily_best_max_bars_since_flip", 3)),
                        ),
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
    recent_half_life_days=9.0,
    fresh_signal_max_bars=48,
    stale_signal_start_bars=160,
    stale_signal_min_weight=0.45,
    symbol_blend_full_alerts=36,
    symbol_min_blend_weight=0.15,
    symbol_confidence_cap_over_strategy=2.0,
    symbol_sell_win_rate_cap_over_base=0.5,
    realized_by_alert_id=None,
    realized_tuning=None,
):
    entries = list(entries or [])
    if isinstance(realized_by_alert_id, dict) and realized_by_alert_id:
        for row in entries:
            if not isinstance(row, dict):
                continue
            realized = realized_by_alert_id.get(str(row.get("alert_id") or "").strip())
            if isinstance(realized, dict):
                row["_realized_win"] = bool(realized.get("win")) if "win" in realized else None
                row["_realized_rr"] = _to_float(realized.get("rr"))
    directional = [row for row in entries if isinstance(row, dict) and str(row.get("signal") or "").upper() in ("BUY", "SELL")]
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
        recent_half_life_days=recent_half_life_days,
        fresh_signal_max_bars=fresh_signal_max_bars,
        stale_signal_start_bars=stale_signal_start_bars,
        stale_signal_min_weight=stale_signal_min_weight,
        realized_tuning=realized_tuning,
    )
    tuned_symbol_profiles, symbol_stats = _build_symbol_tuned_profiles(
        directional,
        base_symbol_profiles=base_symbol_profiles,
        observed_days=observed_days,
        min_alerts_per_symbol=min_alerts_per_symbol,
        target_alerts_per_day=target_alerts_per_day,
        recent_half_life_days=recent_half_life_days,
        fresh_signal_max_bars=fresh_signal_max_bars,
        stale_signal_start_bars=stale_signal_start_bars,
        stale_signal_min_weight=stale_signal_min_weight,
        realized_tuning=realized_tuning,
    )
    tuned_symbol_profiles, symbol_stats = _shrink_symbol_tuned_profiles(
        tuned_symbol_profiles,
        symbol_stats=symbol_stats,
        base_symbol_profiles=base_symbol_profiles,
        tuned_strategy_profiles=tuned_strategy_profiles,
        min_alerts_per_symbol=min_alerts_per_symbol,
        full_weight_alerts=symbol_blend_full_alerts,
        min_blend_weight=symbol_min_blend_weight,
        confidence_cap_over_strategy=symbol_confidence_cap_over_strategy,
        sell_win_rate_cap_over_base=symbol_sell_win_rate_cap_over_base,
    )
    tuned_cdc_profiles, cdc_stats = _build_cdc_daily_best_tuned_profiles(
        directional,
        base_cdc_profiles=base_cdc_profiles,
        observed_days=observed_days,
        min_rows=max(2, int(math.ceil(float(min_alerts_per_symbol) / 4.0))),
        target_daily_pick_alerts_per_day=target_daily_pick_alerts_per_day,
        recent_half_life_days=recent_half_life_days,
        fresh_signal_max_bars=fresh_signal_max_bars,
        stale_signal_start_bars=stale_signal_start_bars,
        stale_signal_min_weight=stale_signal_min_weight,
    )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "history_days_requested": int(history_days or 0),
        "history_rows": len(entries or []),
        "directional_rows": len(directional),
        "observed_days": int(observed_days),
        "recent_half_life_days": float(recent_half_life_days),
        "fresh_signal_max_bars": int(fresh_signal_max_bars),
        "stale_signal_start_bars": int(stale_signal_start_bars),
        "stale_signal_min_weight": float(stale_signal_min_weight),
        "symbol_blend_full_alerts": int(symbol_blend_full_alerts),
        "symbol_min_blend_weight": float(symbol_min_blend_weight),
        "symbol_confidence_cap_over_strategy": float(symbol_confidence_cap_over_strategy),
        "symbol_sell_win_rate_cap_over_base": float(symbol_sell_win_rate_cap_over_base),
        "realized_tuning": dict(realized_tuning) if isinstance(realized_tuning, dict) else {},
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
