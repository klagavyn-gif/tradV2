import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTCOMES_PATH = PROJECT_ROOT / ".data" / "telegram_alerts" / "realized_outcomes.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / ".data" / "telegram_alerts" / "entry_edge_report.md"
DEFAULT_JSON_PATH = PROJECT_ROOT / ".data" / "telegram_alerts" / "entry_edge_report.json"

Z_95 = 1.959963984540054
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260919


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _parse_timestamp(value):
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


def _median(values):
    numbers = sorted(value for value in values if isinstance(value, float))
    if not numbers:
        return None
    middle = len(numbers) // 2
    if len(numbers) % 2 == 1:
        return float(numbers[middle])
    return (float(numbers[middle - 1]) + float(numbers[middle])) / 2.0


def _stdev(values):
    numbers = [value for value in values if isinstance(value, float)]
    if len(numbers) < 2:
        return None
    average = sum(numbers) / float(len(numbers))
    variance = sum((value - average) ** 2 for value in numbers) / float(len(numbers) - 1)
    return math.sqrt(variance)


def _wilson_ci(successes, total):
    if total <= 0:
        return (None, None)
    p = float(successes) / float(total)
    denominator = 1.0 + (Z_95 * Z_95) / float(total)
    center = (p + (Z_95 * Z_95) / (2.0 * float(total))) / denominator
    margin = (
        Z_95
        * math.sqrt((p * (1.0 - p)) / float(total) + (Z_95 * Z_95) / (4.0 * float(total) * float(total)))
        / denominator
    )
    return (max(0.0, center - margin) * 100.0, min(1.0, center + margin) * 100.0)


def _bootstrap_mean_ci(values):
    numbers = [value for value in values if isinstance(value, float)]
    if not numbers:
        return (None, None)
    if len(numbers) == 1:
        return (numbers[0], numbers[0])
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(numbers)
    means = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        total = 0.0
        for _ in range(count):
            total += numbers[rng.randrange(count)]
        means.append(total / float(count))
    means.sort()
    low_index = max(0, int(0.025 * BOOTSTRAP_ITERATIONS) - 1)
    high_index = min(BOOTSTRAP_ITERATIONS - 1, int(0.975 * BOOTSTRAP_ITERATIONS))
    return (means[low_index], means[high_index])


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _trade_metrics(row, cost_pct):
    entry_price = _safe_float(row.get("entry_price"))
    stop_loss = _safe_float(row.get("stop_loss"))
    pnl_pct = _safe_float(row.get("pnl_pct"))
    rr_realized = _safe_float(row.get("rr_realized"))
    risk_pct = None
    if entry_price and stop_loss and entry_price != 0:
        risk_pct = abs(entry_price - stop_loss) / abs(entry_price) * 100.0
        if risk_pct <= 0:
            risk_pct = None
    net_pnl_pct = pnl_pct - cost_pct if isinstance(pnl_pct, float) else None
    net_rr = None
    if isinstance(net_pnl_pct, float) and isinstance(risk_pct, float) and risk_pct > 0:
        net_rr = net_pnl_pct / risk_pct
    return {
        "risk_pct": risk_pct,
        "net_pnl_pct": net_pnl_pct,
        "net_rr": net_rr,
        "gross_rr": rr_realized,
        "gross_pnl_pct": pnl_pct,
        "is_net_win": bool(net_pnl_pct is not None and net_pnl_pct > 0),
    }


def _bucket_stats(rows, cost_pct):
    settled = [row for row in rows if str(row.get("outcome_status") or "").lower() == "settled"]
    wins = [row for row in settled if str(row.get("outcome_result") or "").lower() == "win"]
    losses = [row for row in settled if str(row.get("outcome_result") or "").lower() == "loss"]
    flats = [row for row in settled if str(row.get("outcome_result") or "").lower() == "flat"]
    metrics = [_trade_metrics(row, cost_pct) for row in settled]
    rr_values = [m["gross_rr"] for m in metrics if isinstance(m["gross_rr"], float)]
    pnl_values = [m["gross_pnl_pct"] for m in metrics if isinstance(m["gross_pnl_pct"], float)]
    net_rr_values = [m["net_rr"] for m in metrics if isinstance(m["net_rr"], float)]
    net_pnl_values = [m["net_pnl_pct"] for m in metrics if isinstance(m["net_pnl_pct"], float)]
    risk_values = [m["risk_pct"] for m in metrics if isinstance(m["risk_pct"], float)]
    net_wins = sum(1 for m in metrics if m["is_net_win"])
    win_rate_ci = _wilson_ci(len(wins), len(settled))
    net_win_rate_ci = _wilson_ci(net_wins, len(settled))
    return {
        "settled": len(settled),
        "wins": len(wins),
        "losses": len(losses),
        "flats": len(flats),
        "win_rate_pct": (float(len(wins)) / float(len(settled)) * 100.0) if settled else None,
        "win_rate_ci95_pct": list(win_rate_ci),
        "avg_rr": _mean(rr_values),
        "median_rr": _median(rr_values),
        "avg_rr_ci95": list(_bootstrap_mean_ci(rr_values)),
        "avg_pnl_pct": _mean(pnl_values),
        "avg_risk_pct": _mean(risk_values),
        "net_win_rate_pct": (float(net_wins) / float(len(settled)) * 100.0) if settled else None,
        "net_win_rate_ci95_pct": list(net_win_rate_ci),
        "net_avg_rr": _mean(net_rr_values),
        "net_avg_rr_ci95": list(_bootstrap_mean_ci(net_rr_values)),
        "net_avg_pnl_pct": _mean(net_pnl_values),
    }


def _group_stats(rows, key_function, cost_pct):
    buckets = defaultdict(list)
    for row in rows:
        buckets[key_function(row)].append(row)
    result = []
    for name, bucket_rows in buckets.items():
        stats = _bucket_stats(bucket_rows, cost_pct)
        stats["name"] = name
        result.append(stats)
    result.sort(key=lambda item: (-int(item.get("settled") or 0), str(item.get("name") or "")))
    return result


def _exit_reason_counts(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[str(row.get("exit_reason") or "unknown")] += 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def _filter_rows(rows, *, since, days):
    filtered = []
    cutoff = None
    if isinstance(days, (int, float)) and days > 0:
        cutoff = datetime.now() - timedelta(days=float(days))
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("alert_intent") or "").strip().lower() != "entry":
            continue
        timestamp = _parse_timestamp(row.get("timestamp"))
        if since is not None and (timestamp is None or timestamp < since):
            continue
        if cutoff is not None and (timestamp is None or timestamp < cutoff):
            continue
        filtered.append(row)
    return filtered


def _fmt(value, digits=2, suffix=""):
    if not isinstance(value, (int, float)):
        return "n/a"
    return "{:.{}f}{}".format(float(value), digits, suffix)


def _fmt_ci(ci, digits=2, suffix=""):
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return "n/a"
    low, high = ci
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return "n/a"
    return "{:.{}f} to {:.{}f}{}".format(float(low), digits, float(high), digits, suffix)


def _render_markdown(payload):
    overall = payload.get("overall") or {}
    request = payload.get("request") or {}
    progress = payload.get("sample_progress") or {}
    lines = [
        "# Entry Edge Report",
        "",
        "## Scope",
        "- generated_at: {}".format(payload.get("generated_at") or "n/a"),
        "- since: {}".format(request.get("since") or "ALL"),
        "- days: {}".format(request.get("days") or "ALL"),
        "- cost_bps: {}".format(request.get("cost_bps")),
        "",
        "## Overall (entry only, settled)",
        "- settled: {} (target {} = {})".format(
            progress.get("settled"),
            progress.get("target"),
            _fmt(progress.get("progress_pct"), 1, "%"),
        ),
        "- win_rate: {} (95% CI {})".format(
            _fmt(overall.get("win_rate_pct"), 2, "%"),
            _fmt_ci(overall.get("win_rate_ci95_pct"), 2, "%"),
        ),
        "- avg_rr: {} (95% CI {}) | median_rr: {}".format(
            _fmt(overall.get("avg_rr")),
            _fmt_ci(overall.get("avg_rr_ci95")),
            _fmt(overall.get("median_rr")),
        ),
        "- avg_pnl: {} | avg_risk: {}".format(
            _fmt(overall.get("avg_pnl_pct"), 2, "%"),
            _fmt(overall.get("avg_risk_pct"), 2, "%"),
        ),
        "- net_win_rate ({} bps): {} (95% CI {})".format(
            request.get("cost_bps"),
            _fmt(overall.get("net_win_rate_pct"), 2, "%"),
            _fmt_ci(overall.get("net_win_rate_ci95_pct"), 2, "%"),
        ),
        "- net_avg_rr: {} (95% CI {}) | net_avg_pnl: {}".format(
            _fmt(overall.get("net_avg_rr")),
            _fmt_ci(overall.get("net_avg_rr_ci95")),
            _fmt(overall.get("net_avg_pnl_pct"), 2, "%"),
        ),
        "",
        "## Exit Reasons",
    ]
    exit_reasons = payload.get("exit_reasons") or {}
    if exit_reasons:
        for name, count in exit_reasons.items():
            lines.append("- {}: {}".format(name, count))
    else:
        lines.append("- no data")
    lines.append("")
    lines.append(_render_table("By Strategy", payload.get("by_strategy") or []))
    lines.append("")
    lines.append(_render_table("By Signal", payload.get("by_signal") or []))
    lines.append("")
    lines.append(_render_table("By Symbol", payload.get("by_symbol") or []))
    lines.append("")
    lines.append(_render_table("By Month", payload.get("by_month") or []))
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_table(title, rows):
    lines = ["## {}".format(title)]
    if not rows:
        lines.append("- no data")
        return "\n".join(lines)
    lines.append("| group | settled | WR | net WR | avg RR | net RR | avg PnL | net PnL |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {name} | {settled} | {wr} | {nwr} | {rr} | {nrr} | {pnl} | {npnl} |".format(
                name=str(row.get("name") or ""),
                settled=int(row.get("settled") or 0),
                wr=_fmt(row.get("win_rate_pct"), 1, "%"),
                nwr=_fmt(row.get("net_win_rate_pct"), 1, "%"),
                rr=_fmt(row.get("avg_rr")),
                nrr=_fmt(row.get("net_avg_rr")),
                pnl=_fmt(row.get("avg_pnl_pct"), 2, "%"),
                npnl=_fmt(row.get("net_avg_pnl_pct"), 2, "%"),
            )
        )
    return "\n".join(lines)


def _telegram_summary(payload):
    overall = payload.get("overall") or {}
    progress = payload.get("sample_progress") or {}
    request = payload.get("request") or {}
    lines = [
        "<b>tradV2 entry edge report</b>",
        "since={} days={} cost={} bps".format(
            request.get("since") or "ALL", request.get("days") or "ALL", request.get("cost_bps")
        ),
        "settled entries: {} / {} ({})".format(
            progress.get("settled"), progress.get("target"), _fmt(progress.get("progress_pct"), 1, "%")
        ),
        "win rate: {} (95% CI {})".format(
            _fmt(overall.get("win_rate_pct"), 1, "%"), _fmt_ci(overall.get("win_rate_ci95_pct"), 1, "%")
        ),
        "avg RR: {} (95% CI {}) | median RR: {}".format(
            _fmt(overall.get("avg_rr")), _fmt_ci(overall.get("avg_rr_ci95")), _fmt(overall.get("median_rr"))
        ),
        "net win rate: {} | net avg RR: {} (95% CI {})".format(
            _fmt(overall.get("net_win_rate_pct"), 1, "%"),
            _fmt(overall.get("net_avg_rr")),
            _fmt_ci(overall.get("net_avg_rr_ci95")),
        ),
    ]
    by_strategy = payload.get("by_strategy") or []
    if by_strategy:
        lines.append("by strategy:")
        for row in by_strategy[:4]:
            lines.append(
                "  {}: n={} WR={} netRR={}".format(
                    row.get("name"),
                    row.get("settled"),
                    _fmt(row.get("win_rate_pct"), 1, "%"),
                    _fmt(row.get("net_avg_rr")),
                )
            )
    return "\n".join(lines)


def _send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    thread_id = os.environ.get("TELEGRAM_THREAD_ID")
    if not token or not chat_id:
        print("[entry-edge] telegram secrets missing; skip notify")
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
            print("[entry-edge] telegram send attempt {} failed: {}".format(attempt + 1, exc))
            time.sleep(2)
    return False


def build_parser():
    parser = argparse.ArgumentParser(description="Report realized entry-only performance to test the entry edge")
    parser.add_argument("--outcomes-path", default=str(DEFAULT_OUTCOMES_PATH))
    parser.add_argument("--days", type=float, default=45.0, help="Lookback window in days (0 = all)")
    parser.add_argument("--since", default="", help="Only include alerts on/after this date (YYYY-MM-DD)")
    parser.add_argument("--cost-bps", type=float, default=30.0, help="Round-trip cost in basis points")
    parser.add_argument("--target-settled", type=int, default=100, help="Settled entry sample target")
    parser.add_argument("--notify-telegram", action="store_true", help="Send the summary to Telegram")
    parser.add_argument("--output-path", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--json-output-path", default=str(DEFAULT_JSON_PATH))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    outcomes_path = Path(args.outcomes_path).expanduser().resolve()
    if not outcomes_path.exists():
        raise SystemExit("outcomes file not found: {}".format(outcomes_path))

    payload = _load_json(outcomes_path)
    rows = list((payload or {}).get("outcomes") or [])
    since = _parse_timestamp(args.since) if str(args.since or "").strip() else None
    filtered = _filter_rows(rows, since=since, days=float(args.days) if args.days and args.days > 0 else None)
    cost_pct = float(args.cost_bps) / 100.0

    settled = [row for row in filtered if str(row.get("outcome_status") or "").lower() == "settled"]
    overall = _bucket_stats(settled, cost_pct)
    target = max(1, int(args.target_settled))
    progress_pct = min(100.0, float(overall.get("settled") or 0) / float(target) * 100.0)

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "artifact_type": "entry_edge_report",
        "request": {
            "outcomes_path": str(outcomes_path),
            "days": float(args.days) if args.days and args.days > 0 else None,
            "since": str(args.since or "").strip() or None,
            "cost_bps": float(args.cost_bps),
            "target_settled": target,
        },
        "source_window_days": (payload or {}).get("window_days"),
        "entry_rows": len(filtered),
        "sample_progress": {
            "settled": int(overall.get("settled") or 0),
            "target": target,
            "progress_pct": progress_pct,
            "reached_target": bool(overall.get("settled") or 0) >= target,
        },
        "overall": overall,
        "exit_reasons": _exit_reason_counts(settled),
        "by_strategy": _group_stats(settled, lambda row: str(row.get("strategy") or "UNKNOWN").upper(), cost_pct),
        "by_signal": _group_stats(settled, lambda row: str(row.get("signal") or "UNKNOWN").upper(), cost_pct),
        "by_symbol": _group_stats(settled, lambda row: str(row.get("symbol") or "UNKNOWN").upper(), cost_pct),
        "by_month": _group_stats(
            settled,
            lambda row: (_parse_timestamp(row.get("timestamp")) or datetime.min).strftime("%Y-%m"),
            cost_pct,
        ),
    }

    markdown = _render_markdown(report)
    _write_text(Path(args.output_path), markdown)
    _write_json(Path(args.json_output_path), report)

    print(
        "[entry-edge] settled={} target={} win_rate={} avg_rr={} net_win_rate={} net_avg_rr={}".format(
            report["sample_progress"]["settled"],
            target,
            _fmt(overall.get("win_rate_pct"), 2, "%"),
            _fmt(overall.get("avg_rr")),
            _fmt(overall.get("net_win_rate_pct"), 2, "%"),
            _fmt(overall.get("net_avg_rr")),
        ),
        flush=True,
    )
    print("[entry-edge] markdown={}".format(Path(args.output_path).resolve()), flush=True)
    print("[entry-edge] json={}".format(Path(args.json_output_path).resolve()), flush=True)

    if args.notify_telegram:
        sent = _send_telegram(_telegram_summary(report))
        print("[entry-edge] telegram_sent={}".format(sent), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
