import math


def max_symbols_per_request(config, default=30):
    try:
        value = int(getattr(config, "MAX_SYMBOLS_PER_REQUEST", default))
    except Exception:
        value = int(default)
    return max(1, value)


def build_config_warnings(config):
    warnings = []
    debug = bool(getattr(config, "FLASK_DEBUG", True))
    if not debug and not getattr(config, "SECRET_KEY", ""):
        warnings.append("SECRET_KEY ยังไม่ถูกตั้งค่า (ไม่ควรใช้ production)")
    if getattr(config, "HTTP_VERIFY", True) is False:
        warnings.append("HTTP_VERIFY=false (เสี่ยงด้านความปลอดภัย)")
    max_workers = getattr(config, "ANALYZE_MAX_WORKERS", 5)
    try:
        max_workers = int(max_workers)
        if max_workers < 1:
            warnings.append("ANALYZE_MAX_WORKERS ควร >= 1")
    except Exception:
        warnings.append("ANALYZE_MAX_WORKERS ไม่ถูกต้อง")
    return warnings


def parse_symbols_input(raw_symbols, *, normalize_symbol, default_max_symbols, max_symbols=None):
    max_count = int(default_max_symbols) if max_symbols is None else int(max_symbols)
    if max_count < 1:
        max_count = 1
    symbols_raw = str(raw_symbols or "").upper().replace("\n", ",").replace(";", ",").split(",")
    unique_symbols = []
    seen = set()
    for raw in symbols_raw:
        symbol = normalize_symbol(raw)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        unique_symbols.append(symbol)
        if len(unique_symbols) >= max_count:
            break
    return unique_symbols


def parse_periods_input(raw_periods, *, valid_periods, default_periods=None, max_periods=6):
    defaults = list(default_periods or ["1mo", "3mo", "6mo"])
    if isinstance(raw_periods, (list, tuple)):
        raw_tokens = [str(value or "").strip() for value in raw_periods]
    else:
        text = str(raw_periods or "").strip()
        raw_tokens = text.replace("\n", ",").replace(";", ",").split(",") if text else defaults
    periods = []
    seen = set()
    for raw in raw_tokens:
        period = str(raw or "").strip()
        if not period or period in seen or period not in valid_periods:
            continue
        seen.add(period)
        periods.append(period)
        if len(periods) >= int(max_periods):
            break
    return periods if periods else [period for period in defaults if period in valid_periods][: int(max_periods)]


def parse_strategy_input(raw_strategies):
    if isinstance(raw_strategies, (list, tuple)):
        raw_tokens = [str(value or "").strip() for value in raw_strategies]
    else:
        text = str(raw_strategies or "").strip()
        raw_tokens = text.replace("\n", ",").replace(";", ",").split(",") if text else []
    values = []
    seen = set()
    for raw in raw_tokens:
        strategy = str(raw or "").strip().upper()
        if not strategy or strategy in seen:
            continue
        seen.add(strategy)
        values.append(strategy)
    return values


def clean_json_value(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (list, tuple)):
        return [clean_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    return value


def get_telegram_alert_min_confidence(config, default=69.0):
    min_conf = getattr(config, "TELEGRAM_ALERT_MIN_CONFIDENCE", default)
    try:
        min_conf = float(min_conf)
    except Exception:
        min_conf = float(default)
    return float(min_conf)


def analyze_symbols_batch(symbols, period, *, include_chart_data, analyze_single_symbol, executor, repeat_values):
    symbols = list(symbols or [])
    if len(symbols) == 1:
        return [analyze_single_symbol(symbols[0], period, include_chart_data=include_chart_data)]
    return list(executor.map(analyze_single_symbol, symbols, repeat_values(period), repeat_values(include_chart_data)))


def build_health_snapshot(
    *,
    config,
    get_now,
    load_auto_tuned_thresholds,
    data_source_health_snapshot,
    alert_auto_tune_enabled,
    alert_auto_tune_file_path,
):
    tuned = load_auto_tuned_thresholds()
    return {
        "status": "ok",
        "server_time": get_now().strftime("%Y-%m-%d %H:%M:%S (Asia/Bangkok)"),
        "warnings": build_config_warnings(config),
        "data_sources": data_source_health_snapshot(),
        "auto_tune": {
            "enabled": alert_auto_tune_enabled(),
            "file_path": alert_auto_tune_file_path(),
            "loaded": bool(tuned),
            "generated_at": tuned.get("generated_at") if isinstance(tuned, dict) else None,
        },
    }


def build_analysis_summary(
    results,
    *,
    normalize_symbol,
    build_ui_result_summary,
    requested_symbols=None,
    period=None,
):
    total_requested = int(requested_symbols or 0)
    returned = results if isinstance(results, list) else []
    success_count = 0
    error_count = 0
    buy_count = 0
    sell_count = 0
    wait_count = 0
    actionable = []
    errors = []
    for item in returned:
        if not isinstance(item, dict):
            continue
        symbol = normalize_symbol(item.get("symbol") or "")
        if item.get("error"):
            error_count += 1
            errors.append(
                {
                    "symbol": symbol,
                    "error": str(item.get("error") or "").strip(),
                }
            )
            continue
        success_count += 1
        ui_summary = item.get("ui_summary")
        if not isinstance(ui_summary, dict):
            ui_summary = build_ui_result_summary(item)
        signal = str(ui_summary.get("signal") or "WAIT").upper()
        if signal == "BUY":
            buy_count += 1
        elif signal == "SELL":
            sell_count += 1
        else:
            wait_count += 1
        if signal in ("BUY", "SELL"):
            actionable.append(
                {
                    "symbol": symbol,
                    "name": str(item.get("name") or "").strip(),
                    "signal": signal,
                    "source": str(ui_summary.get("source") or "AI Summary"),
                    "confidence": float(ui_summary.get("confidence")) if isinstance(ui_summary.get("confidence"), (int, float)) else None,
                    "pattern": ui_summary.get("pattern"),
                    "price": float(item.get("price")) if isinstance(item.get("price"), (int, float)) else None,
                    "change": float(item.get("change")) if isinstance(item.get("change"), (int, float)) else None,
                }
            )
    actionable.sort(
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            abs(float(item.get("change") or 0.0)),
            str(item.get("symbol") or ""),
        ),
        reverse=True,
    )
    dominant_signal = "WAIT"
    if buy_count and buy_count > sell_count:
        dominant_signal = "BUY"
    elif sell_count and sell_count > buy_count:
        dominant_signal = "SELL"
    elif buy_count or sell_count:
        dominant_signal = "MIXED"
    total_tracked = success_count + error_count
    success_rate_pct = None
    if total_tracked > 0:
        success_rate_pct = round((success_count / total_tracked) * 100.0, 1)
    return {
        "requested_count": total_requested or len(returned),
        "returned_count": len(returned),
        "success_count": success_count,
        "error_count": error_count,
        "success_rate_pct": success_rate_pct,
        "actionable_count": len(actionable),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "wait_count": wait_count,
        "dominant_signal": dominant_signal,
        "period": str(period or ""),
        "top_signal": actionable[0] if actionable else None,
        "action_queue": actionable[:5],
        "errors": errors[:3],
    }
