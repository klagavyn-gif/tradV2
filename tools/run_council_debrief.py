#!/usr/bin/env python3
"""
CLI Tool to execute the Daily Council Debrief & Adaptive Memory Update.
Can be invoked manually, via cron, or via GitHub Actions workflow dispatch.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config
from infrastructure.council.daily_debrief import DailyCouncilDebrief


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run tradV2 Trading Council Daily Debrief & Update Council Memory"
    )
    parser.add_argument(
        "--data-dir",
        default=".data/telegram_alerts",
        help="Directory containing trade outcomes and deliberations",
    )
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=24.0,
        help="Lookback period in hours (default: 24.0)",
    )
    parser.add_argument(
        "--notify-telegram",
        action="store_true",
        help="Broadcast debrief memo to Telegram",
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Force deterministic board dialogue synthesis without Gemini LLM",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    debrief = DailyCouncilDebrief(
        data_dir=args.data_dir,
        config=config,
    )

    memory = debrief.run_debrief(
        lookback_hours=args.lookback_hours,
        notify_telegram=args.notify_telegram,
        use_gemini=not args.no_gemini,
    )

    print("\n" + "=" * 70)
    print("🏛️ TRADV2 DAILY COUNCIL DEBRIEF & ADAPTIVE MEMORY")
    print("=" * 70)
    print(f"Session ID       : {memory.get('session_id')}")
    print(f"Updated At       : {memory.get('updated_at')}")
    print(f"Market Regime    : {memory.get('market_regime')}")
    print(f"Risk Stance      : {memory.get('risk_stance')}")
    print(f"Min Score (Floor): {memory.get('min_composite_score'):.1f}%")
    print(f"Flagged Symbols  : {memory.get('active_directives', {}).get('flagged_symbols')}")
    print(f"Caution Mode     : {memory.get('active_directives', {}).get('caution_mode')}")
    print("\n💡 Key Lessons Learned:")
    for idx, lesson in enumerate(memory.get("lessons_learned", []), 1):
        print(f"  {idx}. {lesson}")
    print("\n💬 Board Dialogue Highlights:")
    for turn in memory.get("dialogue_transcript", []):
        print(f"  [{turn.get('speaker')}]: {turn.get('text')}")
    print("=" * 70)
    print(f"Memory saved to: {debrief.memory_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
