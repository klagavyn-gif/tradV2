from flask import jsonify, request

from application.services.service_support import (
    analyze_symbols_batch,
    build_analysis_summary,
    build_health_snapshot,
    clean_json_value,
    get_telegram_alert_min_confidence,
    max_symbols_per_request,
    parse_symbols_input,
)


def _legacy_trad():
    import trad

    return trad


def build_analyze_response_payload(data):
    trad = _legacy_trad()
    raw_symbols = data.get("symbols", "")
    max_symbols = max_symbols_per_request(trad.config)
    all_symbols = parse_symbols_input(
        raw_symbols,
        normalize_symbol=trad.normalize_symbol,
        default_max_symbols=max_symbols,
        max_symbols=10000,
    )
    symbols = all_symbols[:max_symbols]
    period = data.get("period", "1mo")
    include_chart_data = bool(data.get("include_chart_data", True))
    if period not in trad.VALID_PERIODS:
        return {"error": "Invalid period"}, 400
    if not all_symbols:
        return {"error": "No symbols provided"}, 400
    if len(all_symbols) > max_symbols:
        return {"error": f"Too many symbols (max {max_symbols})"}, 400

    results = analyze_symbols_batch(
        symbols,
        period,
        include_chart_data=include_chart_data,
        analyze_single_symbol=trad.analyze_single_symbol,
        executor=trad._ANALYZE_EXECUTOR,
        repeat_values=trad.repeat,
    )
    alert_runtime_context = trad._build_alert_runtime_context(
        results,
        get_telegram_alert_min_confidence(trad.config),
    )
    notify = bool(data.get("notify_telegram"))
    if notify:
        trad._notify_telegram_from_results(results, runtime_context=alert_runtime_context)

    alert_backtest = trad._build_alert_backtest_summary(results)
    backtest_rules = trad._build_backtest_rulebook(results)
    all_weather = trad._build_all_weather_summary(results)
    regime_summary = dict((alert_runtime_context or {}).get("regime_summary") or {})
    cleaned = [clean_json_value(row) for row in results]
    meta = {
        "request": {
            "symbols": symbols,
            "period": period,
            "include_chart_data": include_chart_data,
            "notify_telegram": notify,
        },
        "summary": build_analysis_summary(
            results,
            normalize_symbol=trad.normalize_symbol,
            build_ui_result_summary=trad._build_ui_result_summary,
            requested_symbols=len(symbols),
            period=period,
        ),
        "telegram_alerts": alert_backtest,
        "all_weather": all_weather,
        "regime_summary": regime_summary,
        "backtest_rules": backtest_rules,
        "health": build_health_snapshot(
            config=trad.config,
            get_now=trad.get_thai_now,
            load_auto_tuned_thresholds=trad._load_auto_tuned_thresholds,
            data_source_health_snapshot=trad._data_source_health_snapshot,
            alert_auto_tune_enabled=trad._alert_auto_tune_enabled,
            alert_auto_tune_file_path=trad._alert_auto_tune_file_path,
        ),
    }
    return {"results": cleaned, "meta": clean_json_value(meta)}, 200


def handle_analyze_request():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request body"}), 400
    payload, status_code = build_analyze_response_payload(data)
    return jsonify(payload), status_code


def run_once(symbols, period, notify_telegram, verify_output=None, verify_include_results=None):
    trad = _legacy_trad()
    max_symbols = max_symbols_per_request(trad.config)
    uniq = parse_symbols_input(
        symbols,
        normalize_symbol=trad.normalize_symbol,
        default_max_symbols=max_symbols,
        max_symbols=max_symbols,
    )
    if not uniq:
        return 2
    if period not in trad.VALID_PERIODS:
        return 2
    results = analyze_symbols_batch(
        uniq,
        period,
        include_chart_data=False,
        analyze_single_symbol=trad.analyze_single_symbol,
        executor=trad._ANALYZE_EXECUTOR,
        repeat_values=trad.repeat,
    )
    base_min_conf = get_telegram_alert_min_confidence(trad.config)
    alert_runtime_context = trad._build_alert_runtime_context(results, base_min_conf)
    live_preview = trad._build_telegram_alert_live_preview(
        results,
        limit_examples_per_strategy=1,
        runtime_context=alert_runtime_context,
    )
    if notify_telegram:
        trad._notify_telegram_from_results(results, runtime_context=alert_runtime_context)
    alert_backtest = trad._build_alert_backtest_summary(results)
    backtest_rules = trad._build_backtest_rulebook(results)
    all_weather = trad._build_all_weather_summary(results)
    regime_summary = dict((alert_runtime_context or {}).get("regime_summary") or {})
    health_snapshot = build_health_snapshot(
        config=trad.config,
        get_now=trad.get_thai_now,
        load_auto_tuned_thresholds=trad._load_auto_tuned_thresholds,
        data_source_health_snapshot=trad._data_source_health_snapshot,
        alert_auto_tune_enabled=trad._alert_auto_tune_enabled,
        alert_auto_tune_file_path=trad._alert_auto_tune_file_path,
    )
    latest_run = trad._read_latest_telegram_run_report()
    realized_report_days = trad._alert_realized_report_days()
    realized_performance = trad._build_telegram_alert_realized_report(days=realized_report_days)
    verify_request = {
        "symbols": uniq,
        "period": period,
        "include_chart_data": False,
        "notify_telegram": bool(notify_telegram),
    }
    verify_summary = build_analysis_summary(
        results,
        normalize_symbol=trad.normalize_symbol,
        build_ui_result_summary=trad._build_ui_result_summary,
        requested_symbols=len(uniq),
        period=period,
    )
    if verify_include_results is None:
        verify_include_results = bool(getattr(trad.config, "VERIFY_INCLUDE_RESULTS", False))
    verify_path = str(verify_output or getattr(trad.config, "VERIFY_OUTPUT_PATH", "") or "").strip()
    if verify_path:
        written_path = trad._write_verify_output(
            verify_path,
            {
                "results": results,
                "request": verify_request,
                "summary": verify_summary,
                "telegram_alerts": alert_backtest,
                "all_weather": all_weather,
                "backtest_rules": backtest_rules,
                "health": health_snapshot,
                "latest_run": latest_run,
                "live_preview": live_preview,
                "regime_summary": regime_summary,
                "alert_runtime_context": alert_runtime_context,
                "realized_performance": realized_performance,
                "include_results": bool(verify_include_results),
            },
        )
        if not written_path:
            trad.logger.error("Failed to write verify output to %s", verify_path)
            return 1
    payload = {
        "results": [clean_json_value(row) for row in results],
        "meta": clean_json_value(
            {
                "request": verify_request,
                "summary": verify_summary,
                "telegram_alerts": alert_backtest,
                "all_weather": all_weather,
                "backtest_rules": backtest_rules,
                "health": health_snapshot,
                "latest_run": latest_run,
                "live_preview": live_preview,
                "regime_summary": regime_summary,
                "alert_runtime_context": alert_runtime_context,
                "realized_performance": realized_performance,
                "verify_output_path": verify_path or None,
            }
        ),
    }
    print(trad.json.dumps(payload, ensure_ascii=False))
    return 0
