"""
Unit tests for the Adversarial Alpha Stress Gauntlet.
"""

import unittest
import pandas as pd
import numpy as np

from tools.run_alpha_stress_gauntlet import (
    generate_calibrated_flash_crash_candles,
    stationary_block_bootstrap,
    AdversarialStressGauntlet,
    GauntletReport,
)


class TestAlphaStressGauntlet(unittest.TestCase):

    def setUp(self):
        self.crash_df = generate_calibrated_flash_crash_candles("FTX", n_bars=100, seed=42)

    def test_calibrated_crash_candle_generator(self):
        """Verify crash synthesizer creates well-formed OHLCV DataFrame."""
        df = self.crash_df
        self.assertEqual(len(df), 100)
        for col in ["open", "high", "low", "close", "volume", "timestamp"]:
            self.assertIn(col, df.columns)

        # High should be >= low, open, close
        self.assertTrue((df["high"] >= df["low"]).all())
        self.assertTrue((df["high"] >= df["open"]).all())
        self.assertTrue((df["high"] >= df["close"]).all())
        self.assertTrue((df["low"] <= df["open"]).all())
        self.assertTrue((df["low"] <= df["close"]).all())

    def test_stationary_block_bootstrap(self):
        """Verify that stationary block bootstrap generates resampled paths of equal length."""
        bootstraps = stationary_block_bootstrap(self.crash_df, mean_block_length=10, n_samples=5, seed=123)
        self.assertEqual(len(bootstraps), 5)
        for b in bootstraps:
            self.assertEqual(len(b), len(self.crash_df))
            self.assertFalse(b["close"].isna().any())

    def test_run_gauntlet_execution(self):
        """Verify the gauntlet executes completely across crash regimes and returns valid report."""
        gauntlet = AdversarialStressGauntlet(base_df=self.crash_df)
        entry_expr = "(squeeze_ratio < 1.0500) & (rel_vol > 1.0000)"
        exit_expr = "(squeeze_momentum < -0.1000)"

        report = gauntlet.run_gauntlet(
            factor_name="Test Squeeze Alpha",
            entry_expr=entry_expr,
            exit_expr=exit_expr,
            direction="BUY",
            n_bootstraps=5,  # fast test
        )

        self.assertIsInstance(report, GauntletReport)
        self.assertEqual(report.factor_name, "Test Squeeze Alpha")
        self.assertEqual(len(report.crash_scenarios), 3)
        self.assertIsInstance(report.passed_all_gates, bool)
        self.assertIn("STRESS GAUNTLET", report.verdict_message)


if __name__ == "__main__":
    unittest.main()
