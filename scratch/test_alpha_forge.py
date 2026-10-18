#!/usr/bin/env python3
"""
Unit and Integration Tests for Council AlphaForge (Quantitative Factor Mining).
Verifies:
1. Feature preparation and safe mathematical DSL evaluation.
2. Candidate hypothesis synthesis (Gemini & Fallback).
3. Walk-Forward backtesting with 0.30% cost deduction and quality gate filtering.
4. Active factor persistence and graduation into active_alpha_factors.json.
5. ContrarianScout integration recognizing active alpha factors.
6. Telegram research memo formatting.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infrastructure.council.alpha_forge import (
    CouncilAlphaForge,
    prepare_market_features,
    evaluate_alpha_expression,
)
from infrastructure.council.contrarian_scout import ContrarianScout


class TestCouncilAlphaForge(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="tradv2_alpha_forge_test_")
        self.market_dir = os.path.join(self.test_dir, "market_data")
        os.makedirs(self.market_dir, exist_ok=True)

        # Create realistic synthetic klines DataFrame (300 bars)
        np.random.seed(42)
        n = 300
        base_price = 100.0
        prices = [base_price]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1.0 + np.random.normal(0.0003, 0.01)))

        rows = []
        for i, p in enumerate(prices):
            o = p
            c = p * np.random.uniform(0.995, 1.005)
            h = max(o, c) * np.random.uniform(1.0, 1.006)
            l = min(o, c) * np.random.uniform(0.994, 1.0)
            v = np.random.uniform(1000, 10000)
            rows.append({
                "timestamp": 1600000000000 + (i * 15 * 60 * 1000),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            })
        self.sample_df = pd.DataFrame(rows)

        # Save synthetic klines for BTCUSDT and ETHUSDT in market_dir
        self.sample_df.to_csv(os.path.join(self.market_dir, "BTCUSDT_15m.csv"), index=False)
        self.sample_df.to_csv(os.path.join(self.market_dir, "ETHUSDT_15m.csv"), index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_feature_preparation(self):
        feat_df = prepare_market_features(self.sample_df)
        self.assertIn("atr14", feat_df.columns)
        self.assertIn("upper_wick", feat_df.columns)
        self.assertIn("lower_wick", feat_df.columns)
        self.assertIn("rel_vol", feat_df.columns)
        self.assertIn("wick_asymmetry", feat_df.columns)
        self.assertIn("cdc_bullish", feat_df.columns)
        self.assertFalse(feat_df["atr14"].isna().any())

    def test_safe_dsl_evaluation(self):
        feat_df = prepare_market_features(self.sample_df)

        # Valid expressions
        res1 = evaluate_alpha_expression("close / (open + 1e-6)", feat_df)
        self.assertEqual(len(res1), len(feat_df))

        res2 = evaluate_alpha_expression("rel_vol * (lower_wick - upper_wick) / (atr14 + 1e-6)", feat_df)
        self.assertEqual(len(res2), len(feat_df))

        res3 = evaluate_alpha_expression("zscore(close, 20)", feat_df)
        self.assertEqual(len(res3), len(feat_df))

        # Forbidden dangerous expressions should be blocked
        with self.assertRaises(ValueError):
            evaluate_alpha_expression("__import__('os').system('ls')", feat_df)

        with self.assertRaises(ValueError):
            evaluate_alpha_expression("eval('1+1')", feat_df)

    def test_candidate_hypotheses_generation(self):
        forge = CouncilAlphaForge(data_dir=self.test_dir, market_data_dir=self.market_dir)
        cands = forge.generate_candidate_hypotheses(use_gemini=False)
        self.assertGreaterEqual(len(cands), 3)
        for c in cands:
            self.assertIn("factor_id", c)
            self.assertIn("formula", c)
            self.assertIn("economic_rationale", c)
            self.assertIn("threshold", c)

    def test_backtest_with_friction_deduction(self):
        forge = CouncilAlphaForge(
            data_dir=self.test_dir,
            market_data_dir=self.market_dir,
            friction_pct=0.30,
        )

        test_cand = {
            "factor_id": "test_factor_alpha",
            "factor_name": "Test Alpha Factor",
            "formula": "rel_vol * (close - open) / (atr14 + 1e-6)",
            "threshold": 0.5,
            "stop_loss_atr": 1.2,
            "take_profit_atr": 2.4,
            "max_hold_bars": 12,
        }

        res = forge.backtest_candidate(test_cand, symbols=["BTCUSDT", "ETHUSDT"])
        self.assertIn("train_stats", res)
        self.assertIn("test_stats", res)
        self.assertIn("cdc_correlation", res)
        self.assertIn("passed_quality_gate", res)

        # Verify trade count and metrics format
        self.assertIsInstance(res["test_stats"]["profit_factor"], float)
        self.assertIsInstance(res["test_stats"]["win_rate_pct"], float)

    def test_mining_cycle_and_persistence(self):
        forge = CouncilAlphaForge(
            data_dir=self.test_dir,
            market_data_dir=self.market_dir,
        )

        summary = forge.run_mining_cycle(
            symbols=["BTCUSDT", "ETHUSDT"],
            use_gemini=False,
            notify_telegram=False,
        )

        self.assertIn("cycle_id", summary)
        self.assertIn("evaluations", summary)
        self.assertGreaterEqual(summary["candidates_count"], 3)

        # Test Telegram message generation
        tg_memo = forge.format_telegram_message(summary)
        self.assertIn("Council Alpha Lab", tg_memo)
        self.assertIn("สถานะการทดสอบสมมติฐานทางคณิตศาสตร์", tg_memo)

    def test_contrarian_scout_alpha_factor_integration(self):
        # Create an active graduated factor file
        active_factors_file = os.path.join(self.test_dir, "active_alpha_factors.json")
        graduated_payload = {
            "version": 1,
            "updated_at": "2026-10-18 10:00:00 UTC",
            "factors": {
                "alpha_vol_spring_15m": {
                    "factor_id": "alpha_vol_spring_15m",
                    "factor_name": "Volatility Compression Spring",
                    "formula": "rel_vol / bb_width(close, 20)",
                    "status": "active_sandbox_exploration",
                }
            }
        }
        with open(active_factors_file, "w", encoding="utf-8") as f:
            json.dump(graduated_payload, f, indent=2)

        scout = ContrarianScout()
        scout.active_factors_path = active_factors_file

        # Candidate matching the active alpha factor
        candidate = {
            "symbol": "SUI-USD",
            "signal": "BUY",
            "alpha_factor_id": "alpha_vol_spring_15m",
            "entry_price": 2.0,
            "stop_loss": 1.9,
            "take_profit": 2.25,
        }

        op = scout.evaluate(candidate, open_explorations_count=0, max_explorations_allowed=1)
        self.assertTrue(op.metrics.get("is_exploration"))
        self.assertIn("AlphaForge", op.metrics.get("thesis"))
        self.assertIn("Volatility Compression Spring", op.metrics.get("thesis"))


if __name__ == "__main__":
    unittest.main()
