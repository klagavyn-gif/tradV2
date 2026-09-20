import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RUN_REPORTS = PROJECT_ROOT / ".data" / "telegram_alerts" / "run_reports.jsonl"
DEFAULT_OUTCOMES = PROJECT_ROOT / ".data" / "telegram_alerts" / "realized_outcomes.json"
DEFAULT_LATEST_RUN = PROJECT_ROOT / ".data" / "telegram_alerts" / "latest_run.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / ".data" / "telegram_alerts" / "gate_attribution_report.md"
DEFAULT_JSON = PROJECT_ROOT / ".data" / "telegram_alerts" / "gate_attribution_report.json"

OVER_FILTER_MARGIN_R = 0.5
STALE_CONFOUND_BARS = 4.0
DEFAULT_ENTRY_WIN_RATE_FLOOR = 57.0
DATA_GATE_TOKENS = ("missing", "not_available", "no_actionable", "no_primary_plan", "insufficient")


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _parse_time(value):
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def _mean(values):
    numbers = [value for value in values if isinstance(value, float)]
    if not numbers:
        return None
    return float(sum(numbers)) / float(len(numbers))


def _fmt(value, digits=2, suffix=""):
    if not isinstance(value, (int, float)):
        return "n/a"
    return "{:.{}f}{}".format(float(value), digits, suffix)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path):
    rows = []
    try:
        raw = pathlib.Path(path).read_text(encoding="utf-8")
    except Exception:
        return rows
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_json(path, payload):
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_text(path, text):
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text)


def build_baseline(outcomes, *, cost_bps):
    cost_pct = float(cost_bps) / 100.0
    settled = [
        row
        for row in outcomes or []
        if isinstance(row, dict)
        and str(row.get("alert_intent") or "").strip().lower() == "entry"
        and str(row.get("outcome_status") or "").strip().lower() == "settled"
    ]
    wins = [row for row in settled if str(row.get("outcome_result") or "").strip().lower() == "win"]
    net_rr_values = []
    net_wins = 0
    net_pnl_values = []
    for row in settled:
        entry = _safe_float(row.get("entry_price"))
        stop = _safe_float(row.get("stop_loss"))
        pnl = _safe_float(row.get("pnl_pct"))
        if pnl is None:
            continue
        net_pnl = pnl - cost_pct
        net_pnl_values.append(net_pnl)
        if net_pnl > 0:
            net_wins += 1
        if entry and stop and entry != 0:
            risk = abs(entry - stop) / abs(entry) * 100.0
            if risk > 0:
                net_rr_values.append(net_pnl / risk)
    settled_count = len(settled)
    return {
        "settled_entries": settled_count,
        "win_rate_pct": (float(len(wins)) / float(settled_count) * 100.0) if settled_count else None,
        "avg_rr": _mean([_safe_float(row.get("rr_realized")) for row in settled]),
        "net_win_rate_pct": (float(net_wins) / float(settled_count) * 100.0) if settled_count else None,
        "net_avg_rr": _mean(net_rr_values),
        "net_avg_pnl_pct": _mean(net_pnl_values),
        "cost_bps": float(cost_bps),
    }


def _dedupe_key(row):
    symbol = str(row.get("symbol") or "").strip().upper()
    strategy = str(row.get("strategy") or "").strip().upper()
    reason = str(row.get("reason") or "").strip()
    signal = str(row.get("signal") or "").strip().upper()
    signal_time = str(row.get("last_signal_time") or "").strip()
    entry = _safe_float(row.get("entry_price"))
    if signal_time:
        return (symbol, strategy, reason, signal, signal_time)
    return (symbol, strategy, reason, signal, round(entry, 6) if isinstance(entry, float) else None)


def aggregate_gates(run_reports, *, window_days, now):
    cutoff = None
    if isinstance(window_days, (int, float)) and window_days > 0:
        cutoff = now - timedelta(days=float(window_days))
    volume = Counter()
    unique_rejects = defaultdict(dict)
    runs_used = 0
    first_seen = None
    last_seen = None
    for report in run_reports or []:
        if not isinstance(report, dict):
            continue
        generated = _parse_time(report.get("generated_at"))
        if cutoff is not None and (generated is None or generated < cutoff):
            continue
        runs_used += 1
        if generated is not None:
            first_seen = generated if first_seen is None or generated < first_seen else first_seen
            last_seen = generated if last_seen is None or generated > last_seen else last_seen
        for reason, count in (report.get("quality_drop_counts") or {}).items():
            try:
                volume[str(reason)] += int(count or 0)
            except Exception:
                continue
        diagnostics = report.get("reject_diagnostics") or {}
        if not isinstance(diagnostics, dict):
            continue
        for rows in diagnostics.values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                reason = str(row.get("reason") or "").strip()
                if not reason:
                    continue
                key = _dedupe_key(row)
                bucket = unique_rejects[reason]
                if key not in bucket:
                    bucket[key] = row
    return {
        "runs_used": runs_used,
        "first_seen": first_seen.strftime("%Y-%m-%d %H:%M:%S") if first_seen else None,
        "last_seen": last_seen.strftime("%Y-%m-%d %H:%M:%S") if last_seen else None,
        "volume": volume,
        "unique_rejects": unique_rejects,
    }


def aggregate_passed(run_reports, *, window_days, now):
    cutoff = None
    if isinstance(window_days, (int, float)) and window_days > 0:
        cutoff = now - timedelta(days=float(window_days))
    overall = {"expectancies": [], "confidences": [], "win_rates": [], "bars": []}
    by_key = defaultdict(lambda: {"expectancies": [], "confidences": [], "win_rates": [], "bars": []})
    seen = set()
    for report in run_reports or []:
        if not isinstance(report, dict):
            continue
        generated = _parse_time(report.get("generated_at"))
        if cutoff is not None and (generated is None or generated < cutoff):
            continue
        for key in ("top_candidates", "raw_top_candidates"):
            rows = report.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                symbol = str(row.get("symbol") or "").strip().upper()
                strategy = str(row.get("strategy") or "UNKNOWN").strip().upper() or "UNKNOWN"
                signal = str(row.get("signal") or "UNKNOWN").strip().upper() or "UNKNOWN"
                stamp = str(row.get("signal_timestamp") or "").strip()
                entry = _safe_float(row.get("entry_price"))
                dedupe_key = (symbol, strategy, signal, stamp or (round(entry, 6) if isinstance(entry, float) else None))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                expectancy = _safe_float(row.get("backtest_expectancy_rr"))
                confidence = _safe_float(row.get("confidence"))
                win_rate = _safe_float(row.get("backtest_win_rate_pct"))
                bucket = by_key["{}|{}".format(strategy, signal)]
                if isinstance(expectancy, float):
                    overall["expectancies"].append(expectancy)
                    bucket["expectancies"].append(expectancy)
                if isinstance(confidence, float):
                    overall["confidences"].append(confidence)
                    bucket["confidences"].append(confidence)
                if isinstance(win_rate, float):
                    overall["win_rates"].append(win_rate)
                    bucket["win_rates"].append(win_rate)
                bars = _safe_float(row.get("profile_runtime_bars_since_signal"))
                if not isinstance(bars, float):
                    stamp = _parse_time(row.get("signal_timestamp"))
                    generated = _parse_time(row.get("analysis_generated_at"))
                    if stamp is not None and generated is not None:
                        bars = (generated - stamp).total_seconds() / (15.0 * 60.0)
                if isinstance(bars, float) and bars >= 0:
                    overall["bars"].append(bars)
                    bucket["bars"].append(bars)
    return {
        "n": len(seen),
        "avg_backtest_expectancy_rr": _mean(overall["expectancies"]),
        "avg_confidence": _mean(overall["confidences"]),
        "avg_backtest_win_rate_pct": _mean(overall["win_rates"]),
        "avg_bars_since_signal": _mean(overall["bars"]),
        "by_strategy_signal": {
            key: {
                "n": len(bucket["expectancies"]),
                "avg_backtest_expectancy_rr": _mean(bucket["expectancies"]),
                "avg_confidence": _mean(bucket["confidences"]),
                "avg_bars_since_signal": _mean(bucket["bars"]),
            }
            for key, bucket in by_key.items()
        },
    }


def profile_gate(rows):
    strategies = Counter()
    signals = Counter()
    symbols = Counter()
    strategy_signal = Counter()
    confidences = []
    win_rates = []
    expectancies = []
    trades = []
    bars_since = []
    with_levels = 0
    for row in rows:
        strategy = str(row.get("strategy") or "UNKNOWN").strip().upper() or "UNKNOWN"
        signal = str(row.get("signal") or "UNKNOWN").strip().upper() or "UNKNOWN"
        strategies[strategy] += 1
        signals[signal] += 1
        symbols[str(row.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"] += 1
        strategy_signal["{}|{}".format(strategy, signal)] += 1
        confidence = _safe_float(row.get("confidence"))
        if isinstance(confidence, float):
            confidences.append(confidence)
        win_rate = _safe_float(row.get("win_rate_pct"))
        if isinstance(win_rate, float):
            win_rates.append(win_rate)
        expectancy = _safe_float(row.get("expectancy_rr"))
        if isinstance(expectancy, float):
            expectancies.append(expectancy)
        trade_count = _safe_float(row.get("trades"))
        if isinstance(trade_count, float):
            trades.append(trade_count)
        bars = _safe_float(row.get("bars_since_signal"))
        if isinstance(bars, float):
            bars_since.append(bars)
        if _safe_float(row.get("stop_loss")) is not None and _safe_float(row.get("take_profit")) is not None:
            with_levels += 1
    return {
        "n": len(rows),
        "avg_confidence": _mean(confidences),
        "avg_backtest_win_rate_pct": _mean(win_rates),
        "avg_backtest_expectancy_rr": _mean(expectancies),
        "avg_trades": _mean(trades),
        "avg_bars_since_signal": _mean(bars_since),
        "with_plan_levels": with_levels,
        "top_strategies": strategies.most_common(3),
        "top_signals": signals.most_common(3),
        "top_symbols": symbols.most_common(3),
        "strategy_signal": dict(strategy_signal),
    }


def _is_data_gate(reason):
    text = str(reason or "").strip().lower()
    return any(token in text for token in DATA_GATE_TOKENS)


def _matched_passed_metrics(profile, passed):
    by_key = (passed or {}).get("by_strategy_signal") or {}
    mix = profile.get("strategy_signal") or {}
    exp_weighted = 0.0
    exp_weight = 0
    bars_weighted = 0.0
    bars_weight = 0
    for key, count in mix.items():
        bucket = by_key.get(key)
        if not isinstance(bucket, dict) or count <= 0:
            continue
        expectancy = bucket.get("avg_backtest_expectancy_rr")
        if isinstance(expectancy, float):
            exp_weighted += expectancy * float(count)
            exp_weight += int(count)
        bars = bucket.get("avg_bars_since_signal")
        if isinstance(bars, float):
            bars_weighted += bars * float(count)
            bars_weight += int(count)
    matched_exp = (exp_weighted / float(exp_weight)) if exp_weight else (passed or {}).get("avg_backtest_expectancy_rr")
    matched_bars = (bars_weighted / float(bars_weight)) if bars_weight else (passed or {}).get("avg_bars_since_signal")
    return (
        matched_exp if isinstance(matched_exp, float) else None,
        matched_bars if isinstance(matched_bars, float) else None,
        exp_weight,
    )


def build_gate_rows(aggregated, baseline, passed, *, min_blocked, win_rate_floor=DEFAULT_ENTRY_WIN_RATE_FLOOR):
    unique_rejects = aggregated.get("unique_rejects") or {}
    volume = aggregated.get("volume") or Counter()
    rows = []
    total_unique = sum(len(value) for value in unique_rejects.values())
    baseline_net_rr = baseline.get("net_avg_rr")
    for reason, bucket in unique_rejects.items():
        profile = profile_gate(list(bucket.values()))
        profile["reason"] = reason
        profile["raw_drops"] = int(volume.get(reason, 0))
        profile["share_pct"] = (float(profile["n"]) / float(total_unique) * 100.0) if total_unique else None
        expectancy = profile.get("avg_backtest_expectancy_rr")
        matched, matched_bars, matched_weight = _matched_passed_metrics(profile, passed)
        blocked_bars = profile.get("avg_bars_since_signal")
        profile["matched_passed_expectancy_rr"] = matched
        profile["matched_passed_bars_since_signal"] = matched_bars
        profile["matched_passed_weight"] = matched_weight
        profile["delta_vs_passed_r"] = (
            expectancy - matched if isinstance(expectancy, float) and isinstance(matched, float) else None
        )
        profile["delta_bars_vs_passed"] = (
            blocked_bars - matched_bars if isinstance(blocked_bars, float) and isinstance(matched_bars, float) else None
        )
        profile["delta_vs_baseline_r"] = (
            expectancy - baseline_net_rr if isinstance(expectancy, float) and isinstance(baseline_net_rr, float) else None
        )
        delta = profile["delta_vs_passed_r"]
        delta_bars = profile["delta_bars_vs_passed"]
        profile["regret_proxy_r"] = float(profile["n"]) * delta if isinstance(delta, float) else None
        if profile["n"] < int(min_blocked):
            profile["flag"] = "insufficient_data"
        elif _is_data_gate(reason):
            profile["flag"] = "data_gate"
        elif not isinstance(delta, float):
            profile["flag"] = "insufficient_metrics"
        elif isinstance(profile.get("avg_backtest_win_rate_pct"), float) and profile["avg_backtest_win_rate_pct"] < float(win_rate_floor):
            profile["flag"] = "low_win_rate_blocked"
        elif delta > OVER_FILTER_MARGIN_R:
            profile["flag"] = (
                "stale_confounded"
                if isinstance(delta_bars, float) and delta_bars > STALE_CONFOUND_BARS
                else "over_filtering_suspect"
            )
        elif delta < -OVER_FILTER_MARGIN_R:
            profile["flag"] = "likely_justified"
        else:
            profile["flag"] = "neutral"
        rows.append(profile)
    rows.sort(key=lambda item: (-int(item.get("n") or 0), str(item.get("reason") or "")))
    return rows


def _render_markdown(payload):
    baseline = payload.get("baseline") or {}
    request = payload.get("request") or {}
    aggregated = payload.get("aggregated") or {}
    rows = payload.get("gates") or []
    lines = [
        "# Gate Attribution Report",
        "",
        "## Scope",
        "- generated_at: {}".format(payload.get("generated_at") or "n/a"),
        "- window_days: {}".format(request.get("window_days")),
        "- runs_used: {} ({} -> {})".format(
            aggregated.get("runs_used"), aggregated.get("first_seen") or "n/a", aggregated.get("last_seen") or "n/a"
        ),
        "- cost_bps: {}".format(request.get("cost_bps")),
        "- min_blocked: {}".format(request.get("min_blocked")),
        "- entry_win_rate_floor: {}".format(request.get("entry_win_rate_floor")),
        "",
        "## Baseline (entry only, settled, net of cost)",
        "- settled: {}".format(baseline.get("settled_entries")),
        "- win_rate: {}".format(_fmt(baseline.get("win_rate_pct"), 2, "%")),
        "- avg_rr: {}".format(_fmt(baseline.get("avg_rr"))),
        "- net_win_rate: {}".format(_fmt(baseline.get("net_win_rate_pct"), 2, "%")),
        "- net_avg_rr: {}".format(_fmt(baseline.get("net_avg_rr"))),
        "",
        "## Passed Reference",
        "- unique passed candidates: {}".format((payload.get("passed") or {}).get("n")),
        "- avg backtest expectancy: {}".format(_fmt((payload.get("passed") or {}).get("avg_backtest_expectancy_rr"), 3)),
        "- avg confidence: {}".format(_fmt((payload.get("passed") or {}).get("avg_confidence"), 1)),
        "- avg bars since signal: {}".format(_fmt((payload.get("passed") or {}).get("avg_bars_since_signal"), 2)),
        "",
        "## Gates",
        "| gate | unique blocked | raw drops | share | avg conf | bt WR | bt expRR | matched expRR | delta vs passed | blocked bars | delta bars | regret proxy (R) | flag |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {reason} | {n} | {raw} | {share} | {conf} | {wr} | {exp} | {matched} | {delta} | {bars} | {dbars} | {regret} | {flag} |".format(
                reason=str(row.get("reason") or ""),
                n=int(row.get("n") or 0),
                raw=int(row.get("raw_drops") or 0),
                share=_fmt(row.get("share_pct"), 1, "%"),
                conf=_fmt(row.get("avg_confidence"), 1),
                wr=_fmt(row.get("avg_backtest_win_rate_pct"), 1, "%"),
                exp=_fmt(row.get("avg_backtest_expectancy_rr"), 3),
                matched=_fmt(row.get("matched_passed_expectancy_rr"), 3),
                delta=_fmt(row.get("delta_vs_passed_r"), 3),
                bars=_fmt(row.get("avg_bars_since_signal"), 1),
                dbars=_fmt(row.get("delta_bars_vs_passed"), 1),
                regret=_fmt(row.get("regret_proxy_r"), 1),
                flag=str(row.get("flag") or ""),
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- `unique blocked` นับสัญญาณไม่ซ้ำ (symbol+strategy+reason+signal+signal_time)")
    lines.append("- `matched expRR` = backtest expectancy เฉลี่ยของ candidate ที่ผ่าน เฉพาะ strategy+signal เดียวกัน")
    lines.append("- `delta vs passed` = expRR ของ candidate ที่ถูกบล็อก ลบ matched passed (บวก = บล็อกของที่ดีกว่าที่ส่ง)")
    lines.append("- `delta bars` = ความเก่าของสัญญาณที่ถูกบล็อก ลบของที่ผ่าน (บวกมาก = ที่ถูกบล็อกเก่ากว่า)")
    lines.append("- `stale_confounded` = delta เป็นบวกแต่ที่ถูกบล็อกเก่ากว่ามาก จึงยังสรุปว่า gate กรองเกินไม่ได้")
    lines.append("- `low_win_rate_blocked` = ที่ถูกบล็อกมี backtest WR ต่ำกว่า floor จึงถือว่าบล็อกถูกต้อง")
    lines.append("- `regret proxy` = unique blocked x delta (บวก = อาจเสียโอกาส, ลบ = อาจช่วยประหยัด)")
    lines.append("- `data_gate` = gate ที่บล็อกเพราะข้อมูลไม่พอ ไม่ใช่ตัดสินคุณภาพ")
    lines.append("- ค่านี้เป็น **proxy** จาก backtest metrics ไม่ใช่ counterfactual outcome จริง")
    lines.append("- counterfactual จริงต้องมี SL/TP ของ candidate ที่ถูกบล็อก จึงเริ่มเก็บใน reject diagnostics")
    over = [row for row in rows if row.get("flag") == "over_filtering_suspect"]
    if over:
        lines.append("")
        lines.append("## Review Candidates")
        for row in over[:5]:
            lines.append(
                "- {}: blocked {}, bt expRR {}, matched {}, regret proxy {}R".format(
                    row.get("reason"),
                    row.get("n"),
                    _fmt(row.get("avg_backtest_expectancy_rr"), 3),
                    _fmt(row.get("matched_passed_expectancy_rr"), 3),
                    _fmt(row.get("regret_proxy_r"), 1),
                )
            )
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def _telegram_summary(payload):
    baseline = payload.get("baseline") or {}
    rows = payload.get("gates") or []
    over = [row for row in rows if row.get("flag") == "over_filtering_suspect"]
    lines = [
        "<b>tradV2 gate attribution</b>",
        "baseline net RR: {} | net WR: {}".format(
            _fmt(baseline.get("net_avg_rr")), _fmt(baseline.get("net_win_rate_pct"), 1, "%")
        ),
        "gates analyzed: {}".format(len(rows)),
    ]
    if over:
        lines.append("อาจกรองเกิน:")
        for row in over[:3]:
            lines.append(
                "  {}: n={} expRR={} vs passed {} regret~{}R".format(
                    row.get("reason"),
                    row.get("n"),
                    _fmt(row.get("avg_backtest_expectancy_rr"), 2),
                    _fmt(row.get("matched_passed_expectancy_rr"), 2),
                    _fmt(row.get("regret_proxy_r"), 1),
                )
            )
    else:
        lines.append("ไม่มี gate ที่เข้าข่ายกรองเกินจากข้อมูลตอนนี้")
    stale = [row for row in rows if row.get("flag") == "stale_confounded"]
    if stale:
        lines.append("ติด staleness (ยังสรุปไม่ได้): {}".format(len(stale)))
    lines.append("เป็น proxy จาก backtest ไม่ใช่ outcome จริง")
    return "\n".join(lines)


def _send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    thread_id = os.environ.get("TELEGRAM_THREAD_ID")
    if not token or not chat_id:
        print("[gate-attribution] telegram secrets missing; skip notify")
        return False
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    if thread_id:
        payload["message_thread_id"] = thread_id
    data = json.dumps(payload).encode("utf-8")
    url = "https://api.telegram.org/bot{}/sendMessage".format(token)
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                if 200 <= int(getattr(response, "status", 0)) < 300:
                    return True
        except Exception as exc:
            print("[gate-attribution] telegram send attempt {} failed: {}".format(attempt + 1, exc))
            time.sleep(2)
    return False


def build_parser():
    parser = argparse.ArgumentParser(description="Attribute alert volume and quality to each quality gate")
    parser.add_argument("--run-reports", default=str(DEFAULT_RUN_REPORTS))
    parser.add_argument("--outcomes", default=str(DEFAULT_OUTCOMES))
    parser.add_argument("--window-days", type=float, default=45.0)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--min-blocked", type=int, default=20)
    parser.add_argument(
        "--entry-win-rate-floor",
        type=float,
        default=DEFAULT_ENTRY_WIN_RATE_FLOOR,
        help="Entry win-rate floor; gates blocking candidates below it are justified",
    )
    parser.add_argument("--notify-telegram", action="store_true")
    parser.add_argument("--output-path", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--json-output-path", default=str(DEFAULT_JSON))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_reports_path = pathlib.Path(args.run_reports).expanduser().resolve()
    outcomes_path = pathlib.Path(args.outcomes).expanduser().resolve()
    if not run_reports_path.exists():
        raise SystemExit("run reports not found: {}".format(run_reports_path))
    if not outcomes_path.exists():
        raise SystemExit("outcomes not found: {}".format(outcomes_path))

    now = datetime.now()
    run_reports = _load_jsonl(run_reports_path)
    outcomes_payload = _load_json(outcomes_path)
    outcomes = (outcomes_payload or {}).get("outcomes") or []
    baseline = build_baseline(outcomes, cost_bps=args.cost_bps)
    aggregated = aggregate_gates(run_reports, window_days=args.window_days, now=now)
    passed = aggregate_passed(run_reports, window_days=args.window_days, now=now)
    gates = build_gate_rows(
        aggregated,
        baseline,
        passed,
        min_blocked=args.min_blocked,
        win_rate_floor=args.entry_win_rate_floor,
    )

    payload = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "artifact_type": "gate_attribution_report",
        "request": {
            "run_reports_path": str(run_reports_path),
            "outcomes_path": str(outcomes_path),
            "window_days": float(args.window_days) if args.window_days and args.window_days > 0 else None,
            "cost_bps": float(args.cost_bps),
            "min_blocked": int(args.min_blocked),
            "entry_win_rate_floor": float(args.entry_win_rate_floor),
        },
        "baseline": baseline,
        "passed": {
            "n": passed.get("n"),
            "avg_backtest_expectancy_rr": passed.get("avg_backtest_expectancy_rr"),
            "avg_confidence": passed.get("avg_confidence"),
            "avg_backtest_win_rate_pct": passed.get("avg_backtest_win_rate_pct"),
            "avg_bars_since_signal": passed.get("avg_bars_since_signal"),
        },
        "aggregated": {
            "runs_used": aggregated.get("runs_used"),
            "first_seen": aggregated.get("first_seen"),
            "last_seen": aggregated.get("last_seen"),
            "total_unique_blocked": sum(len(value) for value in (aggregated.get("unique_rejects") or {}).values()),
        },
        "gates": gates,
    }

    markdown = _render_markdown(payload)
    _write_text(pathlib.Path(args.output_path), markdown)
    _write_json(pathlib.Path(args.json_output_path), payload)

    over = [row for row in gates if row.get("flag") == "over_filtering_suspect"]
    print(
        "[gate-attribution] runs={} gates={} baseline_net_rr={} over_filtering={}".format(
            aggregated.get("runs_used"), len(gates), _fmt(baseline.get("net_avg_rr")), len(over)
        ),
        flush=True,
    )
    print("[gate-attribution] markdown={}".format(pathlib.Path(args.output_path).resolve()), flush=True)
    print("[gate-attribution] json={}".format(pathlib.Path(args.json_output_path).resolve()), flush=True)
    if args.notify_telegram:
        print("[gate-attribution] telegram_sent={}".format(_send_telegram(_telegram_summary(payload))), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
