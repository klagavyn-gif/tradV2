from flask import jsonify, request

from application.services.service_support import (
    analyze_symbols_batch,
    build_health_snapshot,
    clean_json_value,
    get_telegram_alert_min_confidence,
    max_symbols_per_request,
    parse_periods_input,
    parse_strategy_input,
    parse_symbols_input,
)


def _legacy_trad():
    import trad

    return trad


def handle_telegram_alert_report_request():
    trad = _legacy_trad()
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request body"}), 400
    else:
        data = dict(request.args)
    days = data.get("days", 30)
    try:
        days = float(days)
    except Exception:
        days = 30.0
    if days <= 0:
        days = 30.0
    limit_examples = data.get("limit_examples_per_strategy", 1)
    try:
        limit_examples = max(1, int(limit_examples))
    except Exception:
        limit_examples = 1
    max_symbols = max_symbols_per_request(trad.config)
    strategies = parse_strategy_input(data.get("strategies"))
    raw_symbols = data.get("symbols", "")
    symbols = (
        parse_symbols_input(
            raw_symbols,
            normalize_symbol=trad.normalize_symbol,
            default_max_symbols=max_symbols,
            max_symbols=10000,
        )
        if raw_symbols
        else []
    )
    include_presets = bool(data.get("include_presets", True))
    include_live_preview = bool(data.get("include_live_preview", False))
    include_latest_run = bool(data.get("include_latest_run", True))
    report = trad._build_telegram_alert_report(
        days=days,
        strategies=strategies or None,
        symbols=symbols or None,
        limit_examples_per_strategy=limit_examples,
    )
    payload = {
        "request": {
            "days": days,
            "strategies": strategies,
            "symbols": symbols,
            "limit_examples_per_strategy": limit_examples,
            "include_presets": include_presets,
            "include_live_preview": include_live_preview,
            "include_latest_run": include_latest_run,
        },
        "report": report,
        "health": build_health_snapshot(
            config=trad.config,
            get_now=trad.get_thai_now,
            load_auto_tuned_thresholds=trad._load_auto_tuned_thresholds,
            data_source_health_snapshot=trad._data_source_health_snapshot,
            alert_auto_tune_enabled=trad._alert_auto_tune_enabled,
            alert_auto_tune_file_path=trad._alert_auto_tune_file_path,
        ),
        "files": {
            "alert_history_jsonl": trad._alert_history_file_path(),
            "alert_history_csv": trad._alert_history_csv_path() if trad._alert_history_export_csv_enabled() else None,
            "latest_run_json": trad._alert_run_report_file_path() if trad._alert_run_report_enabled() else None,
            "run_reports_jsonl": trad._alert_run_report_log_path() if trad._alert_run_report_enabled() else None,
            "realized_outcomes_json": trad._alert_outcomes_file_path() if trad._alert_realized_enabled() and trad._alert_realized_export_outcomes() else None,
            "realized_summary_json": trad._alert_realized_summary_file_path() if trad._alert_realized_enabled() else None,
            "auto_tuned_thresholds_json": trad._alert_auto_tune_file_path() if trad._alert_auto_tune_enabled() else None,
        },
    }
    if include_latest_run:
        payload["latest_run"] = trad._read_latest_telegram_run_report()
    if include_presets:
        payload["presets"] = {
            "1d": trad._build_telegram_alert_report(
                days=1,
                strategies=strategies or None,
                symbols=symbols or None,
                limit_examples_per_strategy=limit_examples,
            ),
            "30d": trad._build_telegram_alert_report(
                days=30,
                strategies=strategies or None,
                symbols=symbols or None,
                limit_examples_per_strategy=limit_examples,
            ),
        }
    if include_live_preview and symbols:
        period = data.get("period", "15m")
        if period not in trad.VALID_PERIODS:
            return jsonify({"error": "Invalid period"}), 400
        preview_symbols = symbols[:max_symbols]
        preview_results = analyze_symbols_batch(
            preview_symbols,
            period,
            include_chart_data=False,
            analyze_single_symbol=trad.analyze_single_symbol,
            executor=trad._ANALYZE_EXECUTOR,
            repeat_values=trad.repeat,
        )
        preview_runtime_context = trad._build_alert_runtime_context(
            preview_results,
            get_telegram_alert_min_confidence(trad.config),
        )
        payload["live_preview"] = {
            "period": period,
            "symbols": preview_symbols,
            "report": trad._build_telegram_alert_live_preview(
                preview_results,
                limit_examples_per_strategy=limit_examples,
                runtime_context=preview_runtime_context,
            ),
        }
    return jsonify(clean_json_value(payload))


def handle_all_weather_report_request():
    trad = _legacy_trad()
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request body"}), 400
    raw_symbols = data.get("symbols", "")
    max_symbols = max_symbols_per_request(trad.config)
    all_symbols = parse_symbols_input(
        raw_symbols,
        normalize_symbol=trad.normalize_symbol,
        default_max_symbols=max_symbols,
        max_symbols=10000,
    )
    symbols = all_symbols[:max_symbols]
    periods = parse_periods_input(
        data.get("periods"),
        valid_periods=trad.VALID_PERIODS,
        default_periods=["1mo", "3mo", "6mo"],
        max_periods=6,
    )
    if not all_symbols:
        return jsonify({"error": "No symbols provided"}), 400
    if len(all_symbols) > max_symbols:
        return jsonify({"error": f"Too many symbols (max {max_symbols})"}), 400
    if not periods:
        return jsonify({"error": "No valid periods provided"}), 400

    period_reports = {}
    for period in periods:
        results = analyze_symbols_batch(
            symbols,
            period,
            include_chart_data=False,
            analyze_single_symbol=trad.analyze_single_symbol,
            executor=trad._ANALYZE_EXECUTOR,
            repeat_values=trad.repeat,
        )
        period_reports[period] = trad._build_all_weather_summary(results)
    readiness = trad._build_all_weather_readiness(period_reports)
    payload = {
        "request": {
            "symbols": symbols,
            "periods": periods,
        },
        "readiness": readiness,
        "period_reports": period_reports,
        "health": build_health_snapshot(
            config=trad.config,
            get_now=trad.get_thai_now,
            load_auto_tuned_thresholds=trad._load_auto_tuned_thresholds,
            data_source_health_snapshot=trad._data_source_health_snapshot,
            alert_auto_tune_enabled=trad._alert_auto_tune_enabled,
            alert_auto_tune_file_path=trad._alert_auto_tune_file_path,
        ),
    }
    return jsonify(clean_json_value(payload))
