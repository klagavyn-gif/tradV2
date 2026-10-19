#!/usr/bin/env python3
"""
tradV2 — Council Quantum Lab CLI Runner
========================================
Executes the weekly autonomous evolutionary quant research cycle:
1. Downloads live 15m kline data (crypto + synthetic cross-asset pairs).
2. Simulates paper trades for current generation hypotheses (with 0.30% fee).
3. Analyzes error taxonomy (False Breakouts, Profit Decay, Chop Bleed).
4. Evaluates factor graduation (multi-week persistent edge).
5. Evolves the research prompt and generates next-gen radical equations.
6. Updates `prompt_evolution_journal.json` and notifies Telegram.

Usage:
  python tools/run_quantum_lab.py --run-cycle
  python tools/run_quantum_lab.py --run-cycle --notify-telegram
  python tools/run_quantum_lab.py --status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from infrastructure.notifications.telegram_gateway import send_telegram_alert
from infrastructure.council.quantum_lab import (
    QuantumFactorEvaluation,
    QuantumLabEngine,
    enrich_quantum_features,
)
from infrastructure.council.prompt_evolution import PromptEvolutionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_quantum_lab")


def fetch_binance_klines(symbol: str, interval: str = "15m", limit: int = 1000) -> Optional[Any]:
    """Fetch public Binance klines without requiring API authentication."""
    import pandas as pd

    # Normalize symbol to Binance pair
    clean = symbol.replace("-USD", "USDT").replace("-", "").upper()
    url = f"https://api.binance.com/api/v3/klines?symbol={clean}&interval={interval}&limit={limit}"
    req = Request(url, headers={"User-Agent": "tradV2-QuantumLab/1.0"})

    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.warning("Binance klines fetch failed for %s (%s). Falling back to yfinance: %s", symbol, clean, e)
        try:
            import yfinance as yf
            df_yf = yf.download(symbol, period="30d", interval="15m", progress=False)
            if not df_yf.empty:
                if hasattr(df_yf.columns, "levels"):
                    df_yf.columns = [c[0].lower() for c in df_yf.columns]
                else:
                    df_yf.columns = [c.lower() for c in df_yf.columns]
                return df_yf[["open", "high", "low", "close", "volume"]].dropna()
        except Exception as ey:
            logger.error("yfinance fallback failed for %s: %s", symbol, ey)
    return None


def run_quantum_lab_cycle(
    symbols: List[str],
    data_dir: Path,
    use_gemini: bool = True,
    notify_telegram: bool = False,
) -> Dict[str, Any]:
    """Execute the full weekly Quantum Lab research and prompt evolution cycle."""
    logger.info("Initializing QuantumLabEngine and PromptEvolutionEngine...")
    q_engine = QuantumLabEngine(
        data_dir=data_dir,
        friction_pct=getattr(config, "QUANTUM_LAB_FRICTION_PCT", 0.30),
        min_graduation_weeks=getattr(config, "QUANTUM_LAB_MIN_GRADUATION_WEEKS", 4),
        min_graduation_pf=getattr(config, "QUANTUM_LAB_MIN_GRADUATION_PF", 1.60),
        min_graduation_exp_pct=getattr(config, "QUANTUM_LAB_MIN_GRADUATION_EXPECTANCY_PCT", 0.35),
    )

    api_key = os.environ.get("GEMINI_API_KEY", "") if use_gemini else ""
    p_engine = PromptEvolutionEngine(
        journal_path=data_dir / "prompt_evolution_journal.json",
        gemini_api_key=api_key,
    )

    # 1. Fetch kline data for all symbols and cross-assets
    logger.info("Fetching market data for symbols: %s", symbols)
    market_dfs: Dict[str, Any] = {}
    for sym in symbols:
        df = fetch_binance_klines(sym, interval="15m", limit=1000)
        if df is not None and len(df) >= 100:
            market_dfs[sym] = df
            logger.info("Loaded %d bars for %s", len(df), sym)
        else:
            logger.warning("Insufficient data for %s", sym)

    if not market_dfs:
        raise RuntimeError("Failed to fetch kline data for any symbol. Aborting Quantum Lab cycle.")

    # 2. Load Current Generation Hypotheses from Journal
    journal = p_engine.load_journal()
    cur_gen = journal.get("current_generation", 0)
    logger.info("Current Journal Generation: %d", cur_gen)

    active_hypotheses: List[Dict[str, str]] = []
    if cur_gen > 0 and journal.get("generations"):
        active_hypotheses = journal["generations"][-1].get("hypotheses", [])

    # If first run or no active hypotheses, seed from Gen 1 defaults
    if not active_hypotheses:
        logger.info("No active hypotheses found. Seeding Generation 1...")
        from infrastructure.council.prompt_evolution import SEED_HYPOTHESES_GEN1
        active_hypotheses = SEED_HYPOTHESES_GEN1

    # 3. Simulate Paper Trades Across Active Hypotheses
    evaluations: List[QuantumFactorEvaluation] = []
    logger.info("Simulating %d radical factors across %d market symbols...", len(active_hypotheses), len(market_dfs))

    for h in active_hypotheses:
        fn = h.get("factor_name", "Unknown Factor")
        hyp_text = h.get("hypothesis", "")
        entry_expr = h.get("entry_expression", "")
        exit_expr = h.get("exit_expression", "")
        direction = h.get("direction", "BUY")

        # Pool trade outcomes across all available symbols
        combined_trades: List[Any] = []
        for sym, df_raw in market_dfs.items():
            try:
                ev = q_engine.backtest_extreme_factor(
                    factor_name=fn,
                    hypothesis=hyp_text,
                    entry_expr=entry_expr,
                    exit_expr=exit_expr,
                    df=df_raw,
                    symbol=sym,
                    direction=direction,
                )
                if ev.trade_outcomes:
                    combined_trades.extend(ev.trade_outcomes)
            except Exception as e:
                logger.warning("Factor '%s' evaluation error on %s: %s", fn, sym, e)

        # Recompile pooled evaluation across all symbols
        ev_compiled = q_engine._compile_evaluation(
            factor_name=fn,
            hypothesis=hyp_text,
            entry_expr=entry_expr,
            exit_expr=exit_expr,
            direction=direction,
            trades=combined_trades,
        )

        evaluations.append(ev_compiled)
        logger.info(
            "Factor '%s': Trades=%d, PF=%.2f, Exp=%+.2f%%, WR=%.1f%%, FailureMode=%s",
            fn,
            ev_compiled.total_trades,
            ev_compiled.profit_factor,
            ev_compiled.net_expectancy_pct,
            ev_compiled.win_rate_pct,
            ev_compiled.error_taxonomy.get("primary_failure_mode"),
        )

    # 4. Multi-Week Graduation Check
    graduated = q_engine.track_and_graduate(evaluations)
    if graduated:
        logger.info("🎓 GRADUATED %d FACTORS TO TRADING COUNCIL: %s", len(graduated), [g['factor_name'] for g in graduated])

    # 5. Evolve Prompt & Generate Next-Gen Hypotheses
    logger.info("Executing Prompt Evolution & Error Post-Mortem...")
    gen_result = p_engine.evolve_cycle(
        recent_evaluations=evaluations,
        current_market_regime="NORMAL",
    )
    logger.info("Successfully advanced to Generation %d via %s", gen_result["generation"], gen_result["synthesis_mode"])

    # 6. Format Telegram Briefing & Send
    telegram_msg = p_engine.format_telegram_evolution_briefing(gen_result)
    if notify_telegram:
        logger.info("Sending Quantum Lab briefing to Telegram...")
        try:
            send_telegram_alert(telegram_msg)
            logger.info("Telegram notification sent successfully! ✅")
        except Exception as et:
            logger.error("Failed to send Telegram notification: %s", et)

    return {
        "generation": gen_result["generation"],
        "synthesis_mode": gen_result["synthesis_mode"],
        "evolved_prompt": gen_result["prompt_text"],
        "hypotheses_count": len(gen_result["hypotheses"]),
        "graduated_count": len(graduated),
        "telegram_message": telegram_msg,
    }


def main():
    parser = argparse.ArgumentParser(description="tradV2 Council Quantum Lab CLI Runner")
    parser.add_argument("--run-cycle", action="store_true", help="Execute weekly evolution & backtest cycle")
    parser.add_argument("--status", action="store_true", help="Print current generation status and journal")
    parser.add_argument("--notify-telegram", action="store_true", help="Dispatch briefing to Telegram")
    parser.add_argument("--no-gemini", action="store_true", help="Force algorithmic quant mutator")
    parser.add_argument("--symbols", default="BTC-USD,ETH-USD,SOL-USD,SUI-USD,PAXG-USD", help="Symbols to test")
    parser.add_argument("--data-dir", default=".data/quantum_lab", help="Directory for quantum lab artifacts")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.status:
        j_path = data_dir / "prompt_evolution_journal.json"
        if not j_path.exists():
            print("Quantum Lab Journal not initialized yet.")
            return 0
        j_data = json.loads(j_path.read_text(encoding="utf-8"))
        print("=" * 65)
        print(f" COUNCIL QUANTUM LAB: GENERATION {j_data.get('current_generation', 0)}")
        print("=" * 65)
        if j_data.get("generations"):
            latest = j_data["generations"][-1]
            print(f"Evolved At:     {latest.get('evolved_at')}")
            print(f"Synthesis Mode: {latest.get('synthesis_mode')}")
            print(f"Evolved Prompt:\n\"{latest.get('prompt_text')}\"")
            print("-" * 65)
            print("Active Hypotheses:")
            for h in latest.get("hypotheses", []):
                print(f" • [{h.get('direction')}] {h.get('factor_name')}")
                print(f"   Entry: {h.get('entry_expression')}")
        print("=" * 65)
        return 0

    if args.run_cycle:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        res = run_quantum_lab_cycle(
            symbols=symbols,
            data_dir=data_dir,
            use_gemini=not args.no_gemini,
            notify_telegram=args.notify_telegram,
        )
        print("=" * 65)
        print(f" QUANTUM LAB CYCLE COMPLETED: ADVANCED TO GEN {res['generation']}")
        print(f" Synthesis Mode: {res['synthesis_mode']}")
        print(f" Evolved Prompt:\n \"{res['evolved_prompt']}\"")
        print(f" Next-Gen Hypotheses: {res['hypotheses_count']}")
        print(f" Graduated to Council: {res['graduated_count']}")
        print("=" * 65)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
