#!/usr/bin/env python3
"""
CLI Tool for tradV2 System Doctor & Anomaly Watchdog.
Usage:
  python -m tools.system_doctor --audit
  python -m tools.system_doctor --notify-telegram
  python -m tools.system_doctor --json
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from alerts.system_doctor import run_system_health_audit, format_doctor_alert_message


def main():
    parser = argparse.ArgumentParser(description="tradV2 System Doctor & Anomaly Watchdog")
    parser.add_argument("--audit", action="store_true", help="Run full 5-point health audit")
    parser.add_argument("--notify-telegram", action="store_true", help="Send alert to Telegram if issues detected")
    parser.add_argument("--json", action="store_true", help="Output JSON audit report")
    parser.add_argument("--data-dir", default="", help="Custom data directory path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else (ROOT_DIR / ".data" / "telegram_alerts")
    report = run_system_health_audit(config, data_dir=data_dir)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print("=" * 65)
    print(f" tradV2 SYSTEM DOCTOR REPORT: {report.get('generated_at')}")
    print("=" * 65)
    print(f" Status:        {report.get('severity')} (Score: {report.get('score'):.0f}/100)")
    print(f" Checks Passed: {report.get('checks_passed')}/{report.get('checks_total')}")
    print(f" Summary Badge: {report.get('summary_badge')}")
    print("-" * 65)

    for c_name, c_data in report.get("checks", {}).items():
        status_icon = "✅" if c_data.get("status") == "PASS" else ("⚠️" if c_data.get("status") == "WARNING" else "❌")
        print(f" {status_icon} [{c_name:17s}] {c_data.get('details')}")

    if report.get("issues"):
        print("-" * 65)
        print(" ISSUES DETECTED:")
        for idx, iss in enumerate(report.get("issues"), 1):
            sev_icon = "❌" if iss.get("severity") == "CRITICAL" else "⚠️"
            print(f"  {idx}. {sev_icon} [{iss.get('type')}] {iss.get('message')}")

    if report.get("ai_diagnosis"):
        print("-" * 65)
        print(" AI DOCTOR DIAGNOSIS & REMEDY:")
        for line in report.get("ai_diagnosis").splitlines():
            print(f"  {line}")

    print("=" * 65)

    if args.notify_telegram:
        msg = format_doctor_alert_message(report)
        if msg:
            try:
                import trad
                sent = trad.send_telegram_alert(msg)
                print(f"Telegram Notification: {'SENT ✅' if sent else 'FAILED ❌'}")
            except Exception as exc:
                print(f"Telegram Notification Error: {exc}")
        else:
            print("Telegram Notification: System is healthy (No alert needed)")

    return 0 if report.get("healthy") else 1


if __name__ == "__main__":
    sys.exit(main())
