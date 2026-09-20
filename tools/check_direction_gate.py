#!/usr/bin/env python
"""Check how many times the direction-alignment gate has fired in production.

Downloads the latest telegram-alert-data artifact from a successful main.yml
run and scans run_reports.jsonl for primary_direction_misaligned_suppressed.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", default="main.yml")
    parser.add_argument("--artifact", default="telegram-alert-data")
    args = parser.parse_args()

    listed = _run(
        f"gh run list --workflow={args.workflow} --limit 10 "
        "--json databaseId,conclusion,createdAt"
    )
    if listed.returncode != 0:
        print("gh run list failed:", listed.stderr)
        sys.exit(1)
    runs = json.loads(listed.stdout or "[]")
    latest = next((row for row in runs if row.get("conclusion") == "success"), None)
    if not latest:
        print("no successful run found")
        sys.exit(1)
    run_id = latest["databaseId"]

    tmp = tempfile.mkdtemp(prefix="tradv2-dirgate-")
    downloaded = _run(f"gh run download {run_id} -n {args.artifact} -D {tmp}")
    if downloaded.returncode != 0:
        print("download failed:", downloaded.stderr)
        sys.exit(1)

    path = Path(tmp) / "run_reports.jsonl"
    if not path.exists():
        print("run_reports.jsonl not found in artifact")
        sys.exit(1)

    total = 0
    runs_hit = 0
    per_day = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        drops = row.get("quality_drop_counts") or {}
        count = int(drops.get("primary_direction_misaligned_suppressed", 0) or 0)
        if count > 0:
            total += count
            runs_hit += 1
            day = str(row.get("generated_at") or "")[:10]
            per_day[day] = per_day.get(day, 0) + count

    print(f"run_id={run_id} createdAt={latest.get('createdAt')}")
    print(f"gate total triggers (in report buffer) = {total} across {runs_hit} runs")
    for day in sorted(per_day):
        print(f"  {day}: {per_day[day]}")


if __name__ == "__main__":
    main()
