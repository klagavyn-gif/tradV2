#!/usr/bin/env python3
"""
Archive V1 Alert History & Outcomes and Initialize Clean V2 Baseline.
- Copies all current files in .data/telegram_alerts/ to .data/archive/v1_pre_oct2026/
- Initializes clean V2 state for realized_outcomes.json, alert_history.*, risk_state.json
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import shutil
import json
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT_DIR / ".data" / "telegram_alerts"
ARCHIVE_DIR = ROOT_DIR / ".data" / "archive" / "v1_pre_oct2026"

def main():
    print("=" * 60)
    print(" ARCHIVING V1 DATA AND INITIALIZING V2 CLEAN BASELINE")
    print("=" * 60)

    # 1. Ensure archive directory exists
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Copy all files from SOURCE_DIR to ARCHIVE_DIR
    if SOURCE_DIR.exists():
        count = 0
        for item in SOURCE_DIR.iterdir():
            if item.is_file():
                dest = ARCHIVE_DIR / item.name
                shutil.copy2(item, dest)
                count += 1
            elif item.is_dir() and not item.name.startswith("."):
                dest = ARCHIVE_DIR / item.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
                count += 1
        print(f"✅ Archived {count} files/folders to: {ARCHIVE_DIR.relative_to(ROOT_DIR)}")
    else:
        print("⚠️ Source directory does not exist, creating fresh...")
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_utc = datetime.utcnow().strftime("%Y-%m-%d")

    # 3. Clean and recreate V2 baseline files
    # 3A: realized_outcomes.json
    outcomes_payload = {
        "generated_at": now_str,
        "window_days": 90.0,
        "epoch": "v2",
        "description": "Clean V2 Baseline: Dynamic R TP (1.2R/2.1R), Breakeven +1.2R, Trailing Stop 0.8R, RWA Universe",
        "outcomes": [],
    }
    with open(SOURCE_DIR / "realized_outcomes.json", "w", encoding="utf-8") as f:
        json.dump(outcomes_payload, f, indent=2, ensure_ascii=False)
    print("✅ Initialized clean realized_outcomes.json (Epoch: V2)")

    # 3B: realized_summary.json
    summary_payload = {
        "generated_at": now_str,
        "epoch": "v2",
        "summary": {},
    }
    with open(SOURCE_DIR / "realized_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print("✅ Initialized clean realized_summary.json")

    # 3C: realized_report.json & .md
    report_payload = {
        "generated_at": now_str,
        "epoch": "v2",
        "reports": [],
    }
    with open(SOURCE_DIR / "realized_report.json", "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)
    
    with open(SOURCE_DIR / "realized_report.md", "w", encoding="utf-8") as f:
        f.write("# Telegram Alert Realized Report (Epoch V2 - Clean Baseline)\n\n"
                f"*Epoch Started: {now_str}*\n\n"
                "Awaiting newly settled trades under the new execution rules.\n")
    print("✅ Initialized clean realized_report.json & .md")

    # 3D: notified_closes.json
    with open(SOURCE_DIR / "notified_closes.json", "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)
    print("✅ Initialized clean notified_closes.json")

    # 3E: alert_history.jsonl & alert_history.csv
    with open(SOURCE_DIR / "alert_history.jsonl", "w", encoding="utf-8") as f:
        f.write("")  # Empty
    
    csv_header = "timestamp,alert_id,symbol,signal,confidence,strategy,message,dispatch_status_label,alert_intent\n"
    with open(SOURCE_DIR / "alert_history.csv", "w", encoding="utf-8") as f:
        f.write(csv_header)
    print("✅ Initialized clean alert_history.jsonl & alert_history.csv")

    # 3F: risk_state.json
    risk_state = {
        "consecutive_losses": 0,
        "consecutive_wins": 0,
        "circuit_breaker_active": False,
        "circuit_breaker_until": None,
        "circuit_breaker_reason": None,
        "daily_date_utc": today_utc,
        "daily_pnl_usdt": 0.0,
        "daily_loss_pct": 0.0,
        "daily_trades_count": 0,
        "history": [],
    }
    with open(SOURCE_DIR / "risk_state.json", "w", encoding="utf-8") as f:
        json.dump(risk_state, f, indent=2)
    print("✅ Initialized clean risk_state.json")

    # 3G: binance_executed_orders.json
    with open(SOURCE_DIR / "binance_executed_orders.json", "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2)
    print("✅ Initialized clean binance_executed_orders.json")

    print("\n" + "=" * 60)
    print(" V2 BASELINE INITIALIZATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
