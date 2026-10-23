#!/usr/bin/env python3
"""
tradV2 — Global Horizon Scout CLI Runner
=========================================
Runs the monthly autonomous intelligence harvesting and knowledge synthesis cycle:
1. Audits last 30-day performance and failure taxonomy.
2. Evolves a targeted, non-redundant search prompt against core blacklists.
3. Screens cutting-edge global discoveries (arXiv, GitHub, Hugging Face).
4. Persists the curated digest to `.agents/knowledge/monthly_digests/`.
5. Optionally notifies Telegram.

Usage:
  python tools/run_global_horizon_scout.py --run-cycle
  python tools/run_global_horizon_scout.py --run-cycle --notify-telegram
  python tools/run_global_horizon_scout.py --status
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Safe stdout for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from infrastructure.council.global_horizon_scout import GlobalHorizonScout
from infrastructure.notifications.telegram_gateway import send_telegram_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("horizon_scout_runner")


def print_digest_summary(digest_path: Path) -> None:
    """Print the content of a monthly horizon digest to terminal safely."""
    if not digest_path.exists():
        print(f"No digest found at: {digest_path}")
        return
    text = digest_path.read_text(encoding="utf-8")
    print("=" * 80)
    print(text)
    print("=" * 80)


def show_latest_status(digests_dir: Path) -> None:
    """Show the most recent monthly digest."""
    if not digests_dir.exists():
        print("No monthly digests directory found.")
        return
    files = sorted(digests_dir.glob("*_HORIZON_DIGEST.md"))
    if not files:
        print("No monthly digests published yet.")
        return
    latest = files[-1]
    print(f"Latest Monthly Horizon Digest: {latest.name}")
    print_digest_summary(latest)


def main() -> None:
    parser = argparse.ArgumentParser(description="tradV2 Global Horizon Scout Runner")
    parser.add_argument("--run-cycle", action="store_true", help="Execute monthly intelligence harvesting cycle")
    parser.add_argument("--dry-run", action="store_true", help="Synthesize prompt without saving digest")
    parser.add_argument("--notify-telegram", action="store_true", help="Broadcast summary to Telegram")
    parser.add_argument("--month", type=str, default=None, help="Specific month key (e.g. 2026-11)")
    parser.add_argument("--status", action="store_true", help="Display latest monthly digest")
    args = parser.parse_args()

    scout = GlobalHorizonScout()

    if args.status:
        show_latest_status(scout.digests_dir)
        return

    if args.dry_run:
        month_key = args.month or datetime.now(timezone.utc).strftime("%Y-%m")
        audit = scout.audit_monthly_performance()
        prompt = scout.synthesize_monthly_meta_prompt(audit, month_key)
        print("=" * 80)
        print(f"DRY RUN: EVOLVED MONTHLY SEARCH PROMPT ({month_key})")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        return

    if args.run_cycle:
        month_key = args.month or datetime.now(timezone.utc).strftime("%Y-%m")
        print(f"Starting Monthly Horizon Intelligence Harvest for: {month_key}...")
        digest, saved_path = scout.run_monthly_cycle(month_key=month_key)
        print_digest_summary(saved_path)

        if args.notify_telegram and digest.telegram_summary:
            print("Sending Monthly Horizon summary to Telegram...")
            try:
                ok = send_telegram_alert(digest.telegram_summary)
                print(f"Telegram notification sent: {ok}")
            except Exception as e:
                print(f"Failed to send Telegram alert: {e}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
