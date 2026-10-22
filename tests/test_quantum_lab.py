#!/usr/bin/env python3
"""
Unit test suite for Council Quantum Lab and Prompt Evolution Engine.
"""

import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from infrastructure.council.quantum_lab import (
    QuantumLabEngine,
    QuantumFactorEvaluation,
    enrich_quantum_features,
    evaluate_quantum_expression,
)
from infrastructure.council.prompt_evolution import PromptEvolutionEngine


def create_mock_candles(n_bars: int = 250) -> pd.DataFrame:
    """Generate realistic OHLCV mock candles for testing."""
    np.random.seed(42)
    t = np.linspace(0, 10 * np.pi, n_bars)
    # Synthetic wave with trend, pullback, and breakouts
    price = 60000.0 + (np.sin(t) * 1500.0) + (np.linspace(0, 4000.0, n_bars))
    noise = np.random.normal(0, 150.0, n_bars)
    close = price + noise

    high = close + np.random.uniform(50.0, 300.0, n_bars)
    low = close - np.random.uniform(50.0, 300.0, n_bars)
    open_p = close + np.random.uniform(-100.0, 100.0, n_bars)
    volume = np.random.uniform(50.0, 500.0, n_bars)

    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=pd.date_range("2026-10-01", periods=n_bars, freq="15min"))
    return df


class TestQuantumLab(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("scratch/test_quantum_lab_data")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_feature_enrichment_and_intermarket(self):
        """Verify non-linear and intermarket features enrich correctly."""
        df = create_mock_candles(150)
        paxg_df = create_mock_candles(150)
        paxg_df["close"] = 2650.0 + np.random.normal(0, 10.0, 150)

        df_rich = enrich_quantum_features(df, cross_dfs={"PAXG-USD": paxg_df})

        # Check required columns
        expected_cols = [
            "body", "upper_wick", "lower_wick", "atr14", "atr_pct",
            "vol_skew", "wick_absorption", "bb_width", "fractal_pinch",
            "kinetic_energy", "efficiency_ratio", "paxg_btc_ratio", "paxg_divergence",
            "pv_correlation", "vwap_deviation"
        ]
        for col in expected_cols:
            self.assertIn(col, df_rich.columns, f"Missing feature: {col}")
            self.assertFalse(df_rich[col].isna().all(), f"Column {col} has all NaNs")

        print(f"  [PASS] Feature enrichment verified: {len(df_rich.columns)} columns enriched successfully.")

    def test_safe_dsl_evaluation(self):
        """Verify safe execution of mathematical expressions and rejection of dangerous code."""
        df = enrich_quantum_features(create_mock_candles(100))

        # 1. Valid radical expression with WorldQuant/Qlib primitives
        valid_expr = "(wick_absorption > 1.2) & (kinetic_energy > 0.5) & (ts_rank(close, 20) > 0.4) & (pv_correlation > 0.0)"
        res = evaluate_quantum_expression(valid_expr, df)
        self.assertIsInstance(res, pd.Series)
        self.assertEqual(len(res), len(df))

        # 1b. Test signed_power and ts_corr
        power_expr = "signed_power(ts_corr(close, volume, 14), 2.0)"
        res_power = evaluate_quantum_expression(power_expr, df)
        self.assertIsInstance(res_power, pd.Series)
        self.assertEqual(len(res_power), len(df))

        # 2. Forbidden keyword: import
        with self.assertRaises(ValueError):
            evaluate_quantum_expression("import os", df)

        # 3. Forbidden keyword: eval
        with self.assertRaises(ValueError):
            evaluate_quantum_expression("eval('1+1')", df)

        # 4. Forbidden dunder access
        with self.assertRaises(ValueError):
            evaluate_quantum_expression("close.__class__", df)

        print("  [PASS] Safe mathematical DSL evaluator verified with rigorous guards.")

    def test_quantum_lab_backtest_with_friction(self):
        """Verify paper backtesting engine with fee deductions and error taxonomy analysis."""
        df = create_mock_candles(200)
        engine = QuantumLabEngine(data_dir=self.test_dir, friction_pct=0.30)

        # Factor that triggers several trades
        ev = engine.backtest_extreme_factor(
            factor_name="Test Kinetic Burst",
            hypothesis="Testing kinetic energy burst",
            entry_expr="(kinetic_energy > 0.0) & (rel_vol > 1.0)",
            exit_expr="(upper_absorption > 1.8)",
            df=df,
            symbol="BTC-USD",
            direction="BUY",
        )

        self.assertIsInstance(ev, QuantumFactorEvaluation)
        self.assertGreater(ev.total_trades, 0)
        self.assertIn("primary_failure_mode", ev.error_taxonomy)
        self.assertIn(ev.error_taxonomy["primary_failure_mode"], [
            "PROFIT_DECAY", "FALSE_BREAKOUT", "CHOP_BLEED", "GENERAL_NOISE"
        ])

        # Verify fee penalty: all net returns must be gross_return - 0.30%
        for t in ev.trade_outcomes:
            expected_net = round(t["gross_return_pct"] - 0.30, 3)
            self.assertAlmostEqual(t["net_return_pct"], expected_net, places=2)

        print(
            f"  [PASS] Simulated paper backtest verified: Trades={ev.total_trades}, "
            f"PF={ev.profit_factor:.2f}, PrimaryFailure={ev.error_taxonomy['primary_failure_mode']}"
        )

    def test_prompt_evolution_and_journal_persistence(self):
        """Verify prompt evolution loop: error post-mortem, prompt mutation, and journal persistence."""
        df = create_mock_candles(150)
        q_engine = QuantumLabEngine(data_dir=self.test_dir)
        p_engine = PromptEvolutionEngine(
            journal_path=self.test_dir / "prompt_evolution_journal.json",
            gemini_api_key="",  # Test algorithmic mutator
        )

        # Run simulated evaluation
        ev = q_engine.backtest_extreme_factor(
            factor_name="Test Seed Factor",
            hypothesis="Testing seed factor",
            entry_expr="(rel_vol > 1.2) & (delta(close, 2) > 0)",
            exit_expr="(delta(close, 2) < 0)",
            df=df,
        )

        # Evolve Cycle 1 -> 2
        gen_result = p_engine.evolve_cycle(
            recent_evaluations=[ev],
            current_market_regime="NORMAL",
        )

        self.assertEqual(gen_result["generation"], 1)
        self.assertIn("hypotheses", gen_result)
        self.assertGreaterEqual(len(gen_result["hypotheses"]), 2)
        self.assertIn("prompt_text", gen_result)

        # Verify Journal File exists and can be reloaded
        journal = p_engine.load_journal()
        self.assertEqual(journal["current_generation"], 1)
        self.assertEqual(len(journal["generations"]), 1)

        # Advance to Cycle 2
        gen2_result = p_engine.evolve_cycle(
            recent_evaluations=[ev],
            current_market_regime="NORMAL",
        )
        self.assertEqual(gen2_result["generation"], 2)
        journal2 = p_engine.load_journal()
        self.assertEqual(journal2["current_generation"], 2)
        self.assertEqual(len(journal2["generations"]), 2)

        print(f"  [PASS] Prompt evolution and journal persistence verified across Generations 1 & 2.")

    def test_factor_graduation_gate(self):
        """Verify graduation gate tracks multi-week performance and graduates elite factors."""
        q_engine = QuantumLabEngine(
            data_dir=self.test_dir,
            min_graduation_weeks=2,
            min_graduation_pf=1.60,
            min_graduation_exp_pct=0.35,
        )

        # Mock high-performing evaluation
        winning_ev = QuantumFactorEvaluation(
            factor_name="Elite Quantum Spring",
            hypothesis="High-conviction test factor",
            entry_expression="(kinetic_energy > 2.0)",
            exit_expression="(upper_absorption > 2.0)",
            direction="BUY",
            total_trades=35,
            wins=24,
            losses=11,
            flats=0,
            win_rate_pct=68.6,
            profit_factor=2.45,
            net_expectancy_pct=0.55,
            avg_reward_risk=1.12,
            max_drawdown_pct=3.2,
            is_graduated=True,
            is_promising=True,
            error_taxonomy={"primary_failure_mode": "GENERAL_NOISE"},
        )

        graduated = q_engine.track_and_graduate([winning_ev])
        self.assertGreaterEqual(len(graduated), 1)
        self.assertEqual(graduated[0]["factor_name"], "Elite Quantum Spring")
        self.assertTrue(q_engine.graduated_factors_path.exists())

        grad_json = json.loads(q_engine.graduated_factors_path.read_text(encoding="utf-8"))
        self.assertEqual(len(grad_json), 1)
        self.assertEqual(grad_json[0]["factor_name"], "Elite Quantum Spring")

        print("  [PASS] Graduation gate verified: Elite factor graduated to Council repository successfully.")

    def test_telegram_briefing_formatting(self):
        """Verify Telegram HTML briefing formatter generates clean, informative text."""
        p_engine = PromptEvolutionEngine()
        gen_entry = {
            "generation": 3,
            "synthesis_mode": "ALGORITHMIC_EVOLUTIONARY_MUTATOR",
            "prompt_text": "มุ่งเน้นการคัดกรอง Fractal Pinch ควบคู่กับ Wick Absorption",
            "post_mortem_summary": {"dominant_failure_mode": "FALSE_BREAKOUT"},
            "hypotheses": [
                {
                    "factor_name": "G3 Low-Pinch Wick Absorption",
                    "direction": "BUY",
                    "hypothesis": "เข้าซื้อเฉพาะเมื่อ Volatility บีบแคบสุดขีด",
                    "entry_expression": "(fractal_pinch < 0.75) & (wick_absorption > 1.4)",
                }
            ],
            "prior_cycle_evaluations": [
                {"factor_name": "Kinetic Asymmetric Burst", "pf": 1.55, "exp": 0.42, "win_rate": 58.0, "trades": 18}
            ],
        }

        msg = p_engine.format_telegram_evolution_briefing(gen_entry)
        self.assertIn("Council Quantum Lab", msg)
        self.assertIn("Gen 3", msg)
        self.assertIn("FALSE_BREAKOUT", msg)
        self.assertIn("G3 Low-Pinch Wick Absorption", msg)

        print("  [PASS] Telegram briefing formatter verified.")


if __name__ == "__main__":
    unittest.main()
