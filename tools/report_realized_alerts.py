import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import trad  # noqa: E402
from application.services.service_support import clean_json_value  # noqa: E402


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _parse_csv_list(value):
    if value is None:
        return None
    items = [str(item).strip().upper() for item in str(value).split(",")]
    items = [item for item in items if item]
    return items or None


def _parse_timestamp(value):
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def _fmt_number(value, digits=2, suffix=""):
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.{digits}f}{suffix}"


def _fmt_count(value):
    if not isinstance(value, (int, float)):
        return "0"
    return str(int(value))


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(clean_json_value(payload), handle, ensure_ascii=False, indent=2)


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _resolve_default_output_path():
    return PROJECT_ROOT / ".data" / "telegram_alerts" / "realized_report.md"


def _resolve_default_json_output_path():
    return PROJECT_ROOT / ".data" / "telegram_alerts" / "realized_report.json"


def _build_bucket():
    return {
        "alerts": 0,
        "settled_alerts": 0,
        "open_alerts": 0,
        "unsupported_alerts": 0,
        "wins": 0,
        "losses": 0,
        "flats": 0,
        "rr_sum": 0.0,
        "rr_count": 0,
        "pnl_sum": 0.0,
        "pnl_count": 0,
    }


def _update_bucket(bucket, row):
    bucket["alerts"] += 1
    status = str(row.get("outcome_status") or "").strip().lower()
    result = str(row.get("outcome_result") or "").strip().lower()
    if status == "settled":
        bucket["settled_alerts"] += 1
        if result == "win":
            bucket["wins"] += 1
        elif result == "loss":
            bucket["losses"] += 1
        elif result == "flat":
            bucket["flats"] += 1
    elif status == "open":
        bucket["open_alerts"] += 1
    else:
        bucket["unsupported_alerts"] += 1

    rr_value = _safe_float(row.get("rr_realized"))
    if isinstance(rr_value, float):
        bucket["rr_sum"] += rr_value
        bucket["rr_count"] += 1
    pnl_value = _safe_float(row.get("pnl_pct"))
    if isinstance(pnl_value, float):
        bucket["pnl_sum"] += pnl_value
        bucket["pnl_count"] += 1


def _finalize_bucket(name, bucket):
    settled = int(bucket.get("settled_alerts") or 0)
    wins = int(bucket.get("wins") or 0)
    rr_count = int(bucket.get("rr_count") or 0)
    pnl_count = int(bucket.get("pnl_count") or 0)
    return {
        "name": name,
        "alerts": int(bucket.get("alerts") or 0),
        "settled_alerts": settled,
        "open_alerts": int(bucket.get("open_alerts") or 0),
        "unsupported_alerts": int(bucket.get("unsupported_alerts") or 0),
        "wins": wins,
        "losses": int(bucket.get("losses") or 0),
        "flats": int(bucket.get("flats") or 0),
        "win_rate_pct": (float(wins) / float(settled) * 100.0) if settled > 0 else None,
        "avg_rr_realized": (float(bucket["rr_sum"]) / float(rr_count)) if rr_count > 0 else None,
        "avg_pnl_pct": (float(bucket["pnl_sum"]) / float(pnl_count)) if pnl_count > 0 else None,
    }


def _sort_bucket_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            -int(row.get("settled_alerts") or 0),
            -(float(row.get("win_rate_pct")) if isinstance(row.get("win_rate_pct"), (int, float)) else -1.0),
            -int(row.get("alerts") or 0),
            str(row.get("name") or ""),
        ),
    )


def _filter_outcomes(outcomes, *, strategies, signals, symbols, days):
    filtered = []
    cutoff = None
    if isinstance(days, (int, float)) and days > 0:
        cutoff = datetime.now() - timedelta(days=float(days))
    for row in outcomes:
        if not isinstance(row, dict):
            continue
        strategy = str(row.get("strategy") or "").strip().upper()
        signal = str(row.get("signal") or "").strip().upper()
        symbol = str(row.get("symbol") or "").strip().upper()
        timestamp = _parse_timestamp(row.get("timestamp"))
        if strategies and strategy not in strategies:
            continue
        if signals and signal not in signals:
            continue
        if symbols and symbol not in symbols:
            continue
        if cutoff and (timestamp is None or timestamp < cutoff):
            continue
        filtered.append(row)
    return filtered


def _build_report_payload(summary_payload, outcomes_payload, *, strategies, signals, symbols, days, top):
    summary = dict(summary_payload or {})
    generated_at = str((summary_payload or {}).get("generated_at") or (outcomes_payload or {}).get("generated_at") or "").strip() or None
    outcomes = list((outcomes_payload or {}).get("outcomes") or [])
    filtered = _filter_outcomes(outcomes, strategies=strategies, signals=signals, symbols=symbols, days=days)

    overall_bucket = _build_bucket()
    by_strategy = defaultdict(_build_bucket)
    by_signal = defaultdict(_build_bucket)
    by_symbol = defaultdict(_build_bucket)
    by_strategy_signal = defaultdict(_build_bucket)

    for row in filtered:
        strategy = str(row.get("strategy") or "UNKNOWN").strip().upper() or "UNKNOWN"
        signal = str(row.get("signal") or "UNKNOWN").strip().upper() or "UNKNOWN"
        symbol = str(row.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
        _update_bucket(overall_bucket, row)
        _update_bucket(by_strategy[strategy], row)
        _update_bucket(by_signal[signal], row)
        _update_bucket(by_symbol[symbol], row)
        _update_bucket(by_strategy_signal[f"{strategy} | {signal}"], row)

    overview = _finalize_bucket("ALL", overall_bucket)
    if isinstance(days, (int, float)) and days > 0:
        overview["alerts_per_day_avg"] = float(overview["alerts"]) / float(days)
    else:
        overview["alerts_per_day_avg"] = summary.get("alerts_per_day_avg")

    return {
        "generated_at": generated_at,
        "request": {
            "days": float(days) if isinstance(days, (int, float)) and days > 0 else None,
            "strategies": strategies,
            "signals": signals,
            "symbols": symbols,
            "top": int(top),
        },
        "source_summary": summary,
        "overview": overview,
        "tables": {
            "by_strategy": _sort_bucket_rows([_finalize_bucket(name, bucket) for name, bucket in by_strategy.items()]),
            "by_signal": _sort_bucket_rows([_finalize_bucket(name, bucket) for name, bucket in by_signal.items()]),
            "by_symbol": _sort_bucket_rows([_finalize_bucket(name, bucket) for name, bucket in by_symbol.items()]),
            "by_strategy_signal": _sort_bucket_rows([_finalize_bucket(name, bucket) for name, bucket in by_strategy_signal.items()]),
        },
        "filtered_outcomes_count": len(filtered),
        "all_outcomes_count": len(outcomes),
    }


def _render_table(title, rows, *, top):
    lines = [f"## {title}"]
    if not rows:
        lines.append("- ไม่มีข้อมูล")
        return "\n".join(lines)
    lines.append("| กลุ่ม | Alerts | Settled | WR | Avg RR | Avg PnL | W-L-F | Open |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for row in rows[: max(1, int(top))]:
        lines.append(
            "| {name} | {alerts} | {settled} | {wr} | {rr} | {pnl} | {wlf} | {open_alerts} |".format(
                name=str(row.get("name") or ""),
                alerts=_fmt_count(row.get("alerts")),
                settled=_fmt_count(row.get("settled_alerts")),
                wr=_fmt_number(row.get("win_rate_pct"), suffix="%"),
                rr=_fmt_number(row.get("avg_rr_realized")),
                pnl=_fmt_number(row.get("avg_pnl_pct"), suffix="%"),
                wlf="{}/{}/{}".format(
                    _fmt_count(row.get("wins")),
                    _fmt_count(row.get("losses")),
                    _fmt_count(row.get("flats")),
                ),
                open_alerts=_fmt_count(row.get("open_alerts")),
            )
        )
    return "\n".join(lines)


def _render_markdown(payload):
    overview = payload.get("overview") or {}
    request = payload.get("request") or {}
    source_summary = payload.get("source_summary") or {}
    lines = [
        "# Realized Alert Report",
        "",
        "## Overview",
        f"- generated_at: {payload.get('generated_at') or 'n/a'}",
        f"- filters: strategies={request.get('strategies') or 'ALL'} | signals={request.get('signals') or 'ALL'} | symbols={request.get('symbols') or 'ALL'} | days={request.get('days') or 'ALL'}",
        f"- alerts: {_fmt_count(overview.get('alerts'))} | settled: {_fmt_count(overview.get('settled_alerts'))} | open: {_fmt_count(overview.get('open_alerts'))}",
        f"- win_rate: {_fmt_number(overview.get('win_rate_pct'), suffix='%')} | avg_rr: {_fmt_number(overview.get('avg_rr_realized'))} | avg_pnl: {_fmt_number(overview.get('avg_pnl_pct'), suffix='%')}",
        f"- alerts_per_day: {_fmt_number(overview.get('alerts_per_day_avg'))}",
        f"- source_window_days: {source_summary.get('window_days') if source_summary.get('window_days') is not None else 'n/a'}",
        "",
        _render_table("By Strategy", payload.get("tables", {}).get("by_strategy") or [], top=request.get("top") or 20),
        "",
        _render_table("By Signal", payload.get("tables", {}).get("by_signal") or [], top=request.get("top") or 20),
        "",
        _render_table("By Strategy Signal", payload.get("tables", {}).get("by_strategy_signal") or [], top=request.get("top") or 20),
        "",
        _render_table("By Symbol", payload.get("tables", {}).get("by_symbol") or [], top=request.get("top") or 20),
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def _print_console_summary(payload, *, markdown_path, json_path):
    overview = payload.get("overview") or {}
    print("[realized] report ready", flush=True)
    print(
        "[realized] alerts={alerts} settled={settled} win_rate={wr} avg_rr={rr} alerts_per_day={apd}".format(
            alerts=_fmt_count(overview.get("alerts")),
            settled=_fmt_count(overview.get("settled_alerts")),
            wr=_fmt_number(overview.get("win_rate_pct"), suffix="%"),
            rr=_fmt_number(overview.get("avg_rr_realized")),
            apd=_fmt_number(overview.get("alerts_per_day_avg")),
        ),
        flush=True,
    )
    for section_name in ("by_strategy", "by_signal", "by_symbol"):
        rows = list(payload.get("tables", {}).get(section_name) or [])
        if not rows:
            continue
        best = rows[0]
        print(
            "[realized] top_{section}: {name} alerts={alerts} settled={settled} wr={wr}".format(
                section=section_name,
                name=str(best.get("name") or ""),
                alerts=_fmt_count(best.get("alerts")),
                settled=_fmt_count(best.get("settled_alerts")),
                wr=_fmt_number(best.get("win_rate_pct"), suffix="%"),
            ),
            flush=True,
        )
    print(f"[realized] markdown={markdown_path}", flush=True)
    print(f"[realized] json={json_path}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Build a readable realized alert report from realized_summary/outcomes artifacts")
    parser.add_argument("--summary-path", default=trad._alert_realized_summary_file_path(), help="Path to realized_summary.json")
    parser.add_argument("--outcomes-path", default=trad._alert_outcomes_file_path(), help="Path to realized_outcomes.json")
    parser.add_argument("--strategies", default="", help="Comma-separated strategy filter, e.g. PA15,CDCVIX15")
    parser.add_argument("--signals", default="", help="Comma-separated signal filter, e.g. BUY,SELL")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol filter, e.g. BTC-USD,ETH-USD")
    parser.add_argument("--days", type=float, default=0.0, help="Optional lookback in days based on outcome timestamps")
    parser.add_argument("--top", type=int, default=20, help="Rows per table in output")
    parser.add_argument("--output-path", default=str(_resolve_default_output_path()), help="Markdown output path")
    parser.add_argument("--json-output-path", default=str(_resolve_default_json_output_path()), help="JSON output path")
    parser.add_argument("--print-only", action="store_true", help="Print summary only without writing output files")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    summary_path = os.path.abspath(args.summary_path)
    outcomes_path = os.path.abspath(args.outcomes_path)
    if not os.path.exists(summary_path):
        raise SystemExit(f"summary file not found: {summary_path}")
    if not os.path.exists(outcomes_path):
        raise SystemExit(f"outcomes file not found: {outcomes_path}")

    summary_payload = _load_json(summary_path)
    outcomes_payload = _load_json(outcomes_path)
    payload = _build_report_payload(
        summary_payload,
        outcomes_payload,
        strategies=_parse_csv_list(args.strategies),
        signals=_parse_csv_list(args.signals),
        symbols=_parse_csv_list(args.symbols),
        days=float(args.days) if args.days and args.days > 0 else None,
        top=max(1, int(args.top)),
    )
    payload["files"] = {
        "summary_json": summary_path,
        "outcomes_json": outcomes_path,
        "report_markdown": os.path.abspath(args.output_path),
        "report_json": os.path.abspath(args.json_output_path),
    }
    markdown = _render_markdown(payload)
    if not args.print_only:
        _write_text(args.output_path, markdown)
        _write_json(args.json_output_path, payload)
    _print_console_summary(payload, markdown_path=os.path.abspath(args.output_path), json_path=os.path.abspath(args.json_output_path))
    print(markdown, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
