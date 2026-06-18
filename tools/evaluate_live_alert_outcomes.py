import argparse
import csv
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import trad  # noqa: E402
from application.services.service_support import clean_json_value  # noqa: E402


def _parse_csv_list(value):
    if value is None:
        return None
    items = [str(item).strip() for item in str(value).split(",")]
    items = [item for item in items if item]
    return items or None


def _write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(clean_json_value(payload), handle, ensure_ascii=False, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(description="Export live alert feedback dataset for V5 research")
    parser.add_argument("--days", type=float, default=90.0, help="History window in days")
    parser.add_argument("--strategies", default="", help="Comma-separated strategy filter")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol filter")
    parser.add_argument(
        "--include-open",
        action="store_true",
        help="Include open and unsupported rows in the exported dataset",
    )
    parser.add_argument(
        "--output-path",
        default=trad._alert_feedback_export_file_path(),
        help="CSV output path",
    )
    parser.add_argument(
        "--summary-path",
        default=trad._alert_feedback_summary_file_path(),
        help="JSON summary output path",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    strategies = _parse_csv_list(args.strategies)
    symbols = _parse_csv_list(args.symbols)

    print("[phase1] building live feedback export...", flush=True)
    payload = trad._build_telegram_alert_feedback_export(
        days=args.days,
        strategies=strategies,
        symbols=symbols,
        include_open=bool(args.include_open),
    )
    fieldnames = trad._alert_feedback_export_fieldnames()
    rows = list(payload.get("rows") or [])
    summary = dict(payload.get("summary") or {})

    _write_csv(args.output_path, fieldnames, rows)
    _write_json(
        args.summary_path,
        {
            "artifact_type": "live_feedback_summary",
            "request": {
                "days": float(args.days),
                "strategies": strategies,
                "symbols": symbols,
                "include_open": bool(args.include_open),
            },
            "summary": summary,
            "files": {
                "csv": os.path.abspath(args.output_path),
                "summary_json": os.path.abspath(args.summary_path),
            },
        },
    )

    print(
        "[phase1] rows={rows} ready={ready} win_rate={win_rate} csv={csv_path}".format(
            rows=summary.get("exported_rows"),
            ready=summary.get("training_ready_rows"),
            win_rate=(
                f"{float(summary['win_rate_pct']):.2f}%"
                if isinstance(summary.get("win_rate_pct"), (int, float))
                else "n/a"
            ),
            csv_path=os.path.abspath(args.output_path),
        ),
        flush=True,
    )
    print(f"[phase1] summary={os.path.abspath(args.summary_path)}", flush=True)


if __name__ == "__main__":
    main()
