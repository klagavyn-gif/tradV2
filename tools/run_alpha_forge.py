#!/usr/bin/env python3
"""
CLI Tool to execute Council AlphaForge: Quantitative Research & Formula Mining.
Can be invoked manually or scheduled via GitHub Actions (Weekly Cron).
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
from infrastructure.council.alpha_forge import CouncilAlphaForge


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run tradV2 Council AlphaForge (Autonomous Alpha Formula Mining)"
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols to backtest (default: 11 universe symbols)",
    )
    parser.add_argument(
        "--notify-telegram",
        action="store_true",
        help="Broadcast research briefing memo to Telegram",
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Force deterministic factor synthesis without calling Gemini API",
    )
    parser.add_argument(
        "--data-dir",
        default=getattr(config, "ALPHA_FORGE_DATA_DIR", ".data/council_research"),
        help="Directory to persist active alpha factors",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbols_list = None
    if args.symbols.strip():
        symbols_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    forge = CouncilAlphaForge(
        config=config,
        data_dir=args.data_dir,
    )

    summary = forge.run_mining_cycle(
        symbols=symbols_list,
        use_gemini=not args.no_gemini,
        notify_telegram=args.notify_telegram,
    )

    print("\n" + "=" * 75)
    print("🔬 TRADV2 COUNCIL ALPHAFORGE: QUANTITATIVE RESEARCH REPORT")
    print("=" * 75)
    print(f"Cycle ID         : {summary.get('cycle_id')}")
    print(f"Completed At     : {summary.get('completed_at')}")
    print(f"Elapsed Time     : {summary.get('elapsed_seconds')}s")
    print(f"Candidates Tested: {summary.get('candidates_count')}")
    print(f"Graduated Factors: {summary.get('graduated_count')}")
    print("-" * 75)

    evals = summary.get("evaluations", [])
    for idx, e in enumerate(evals, 1):
        cand = e["candidate"]
        tst = e["test_stats"]
        status = "✅ GRADUATED" if e["passed_quality_gate"] else "❌ REJECTED"
        print(f"\n[{idx}] {cand.get('factor_name')} ({cand.get('factor_id')}) -> {status}")
        print(f"    Formula : {cand.get('formula')}")
        print(f"    Rationale: {cand.get('economic_rationale')}")
        print(f"    In-Sample  : Trades={e['train_stats']['trades']} | WR={e['train_stats']['win_rate_pct']}% | PF={e['train_stats']['profit_factor']} | Expectancy={e['train_stats']['expectancy_pct']:+.2f}%")
        print(f"    Out-Sample : Trades={tst['trades']} | WR={tst['win_rate_pct']}% | PF={tst['profit_factor']} | Expectancy={tst['expectancy_pct']:+.2f}% (After 0.30% Fee)")
        print(f"    CDC Overlap: {e.get('cdc_correlation', 0.0)*100:.1f}%")

    print("\n" + "=" * 75)
    if summary.get("graduated_count", 0) > 0:
        print(f"🎉 Promoted {summary.get('graduated_count')} factors to Active Sandbox Exploration!")
    else:
        print("🛡️ All candidates rejected by strict Quality Gates. Capital preserved.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
