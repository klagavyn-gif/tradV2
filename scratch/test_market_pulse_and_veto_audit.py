"""
Unit tests for Market Pulse, Session Clock, and Counterfactual Veto Auditor.
Validates:
  1. MarketSessionClock session classification and opening surge window.
  2. CrossAssetPulseEngine lead-lag evaluation and flight-to-safety detection.
  3. CouncilVetoAuditor shadow recording and outcome resolution (Correct Veto vs False Veto).
  4. Veto performance calculation and automatic specialist penalty triggering.
  5. TradingCouncil dynamic specialist weighting and veto penalty override.
  6. AlphaForge session features evaluation.
"""

import os
import sys
import unittest
import shutil
import tempfile
import json
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from infrastructure.council.market_pulse import MarketSessionClock, CrossAssetPulseEngine
from infrastructure.council.veto_auditor import CouncilVetoAuditor
from infrastructure.council.blackboard import CentralBlackboard
from infrastructure.council.arbiter import TradingCouncil, ConsensusVerdict
from infrastructure.council.specialists import CouncilOpinion
from infrastructure.council.alpha_forge import prepare_market_features, evaluate_alpha_expression


class TestMarketPulseAndVetoAudit(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_pulse_veto_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_session_clock_classification(self):
        """Test MarketSessionClock across different UTC hours."""
        # Asia session (03:30 UTC)
        dt_asia = datetime(2026, 10, 19, 3, 30, tzinfo=timezone.utc)
        res_asia = MarketSessionClock.get_current_session(dt_asia)
        self.assertEqual(res_asia["session"], "ASIA")
        self.assertFalse(res_asia["is_opening_surge"])

        # London Opening Surge (07:15 UTC)
        dt_london_open = datetime(2026, 10, 19, 7, 15, tzinfo=timezone.utc)
        res_london = MarketSessionClock.get_current_session(dt_london_open)
        self.assertEqual(res_london["session"], "LONDON")
        self.assertTrue(res_london["is_opening_surge"])

        # NY Opening Surge (13:45 UTC)
        dt_ny_open = datetime(2026, 10, 19, 13, 45, tzinfo=timezone.utc)
        res_ny = MarketSessionClock.get_current_session(dt_ny_open)
        self.assertEqual(res_ny["session"], "NY_OPEN")
        self.assertTrue(res_ny["is_opening_surge"])
        self.assertGreater(res_ny["volatility_bias"], 1.2)

        # Off hours (22:00 UTC)
        dt_off = datetime(2026, 10, 19, 22, 0, tzinfo=timezone.utc)
        res_off = MarketSessionClock.get_current_session(dt_off)
        self.assertEqual(res_off["session"], "OFF_HOURS")

    def test_cross_asset_lead_lag(self):
        """Test CrossAssetPulseEngine lead-lag evaluation."""
        pulse_engine = CrossAssetPulseEngine(cache_dir=os.path.join(self.test_dir, "pulse"))

        # Create synthetic klines with BTC having a distinct 4-bar rally:
        # Bars 0..15 = 60000, Bars 16..19 step up to 61200 (+2.0% return in last 4 bars)
        btc_p = [60000.0] * 16 + [60300.0, 60600.0, 60900.0, 61200.0]
        btc_klines = pd.DataFrame({
            "timestamp": [int(time.time() * 1000) - (20 - i) * 900000 for i in range(20)],
            "open": btc_p,
            "high": [p * 1.002 for p in btc_p],
            "low": [p * 0.998 for p in btc_p],
            "close": btc_p,
            "volume": [100.0] * 20,
        })
        # Alt is flat at 100.0 (0.0% return) -> lag spread = 2.0% - 0.0% = 2.0%
        alt_p = [100.0] * 20
        alt_klines = pd.DataFrame({
            "timestamp": [int(time.time() * 1000) - (20 - i) * 900000 for i in range(20)],
            "open": alt_p,
            "high": [100.2] * 20,
            "low": [99.8] * 20,
            "close": alt_p,
            "volume": [50.0] * 20,
        })
        neutral_paxg = pd.DataFrame({
            "timestamp": [int(time.time() * 1000) - (20 - i) * 900000 for i in range(20)],
            "open": [2500.0] * 20,
            "high": [2502.0] * 20,
            "low": [2498.0] * 20,
            "close": [2500.0] * 20,
            "volume": [10.0] * 20,
        })

        os.makedirs(os.path.join(self.test_dir, "pulse"), exist_ok=True)
        btc_klines.to_csv(os.path.join(self.test_dir, "pulse", "BTCUSDT_pulse.csv"), index=False)
        alt_klines.to_csv(os.path.join(self.test_dir, "pulse", "SOLUSDT_pulse.csv"), index=False)
        neutral_paxg.to_csv(os.path.join(self.test_dir, "pulse", "PAXGUSDT_pulse.csv"), index=False)
        neutral_paxg.to_csv(os.path.join(self.test_dir, "pulse", "ONDOUSDT_pulse.csv"), index=False)

        eval_res = pulse_engine.evaluate_symbol_lead_lag("SOL-USD", "BUY")
        self.assertIn("lag_spread", eval_res)
        self.assertGreater(eval_res["lag_spread"], 1.0)
        # Verify catch-up bonus triggered
        self.assertGreater(eval_res["score_adjustment"], 0.0)

    def test_veto_auditor_recording_and_settlement(self):
        """Test recording shadow vetoes and auditing outcomes (Correct vs False Veto)."""
        auditor = CouncilVetoAuditor(data_dir=self.test_dir, file_name="test_veto.json")

        now_ts = int(time.time()) - 7200  # 2 hours ago

        # 1. Candidate 1: Saved Loss (Correct Veto)
        c1 = {
            "alert_id": "aid_sl_saved",
            "symbol": "BTC-USD",
            "signal": "BUY",
            "entry_price": 60000.0,
            "stop_loss": 59000.0,
            "take_profit": 62000.0,
        }
        v1 = ConsensusVerdict(
            decision="VETO",
            approved=False,
            composite_score=45.0,
            veto_specialist="derivatives_specialist",
            veto_reason="crowded_long_funding",
        )
        auditor.record_veto(c1, v1)

        # 2. Candidate 2: Missed Profit (False Veto)
        c2 = {
            "alert_id": "aid_tp_missed",
            "symbol": "ETH-USD",
            "signal": "BUY",
            "entry_price": 3000.0,
            "stop_loss": 2900.0,
            "take_profit": 3200.0,
        }
        v2 = ConsensusVerdict(
            decision="VETO",
            approved=False,
            composite_score=50.0,
            veto_specialist="risk_specialist",
            veto_reason="consecutive_loss_caution",
        )
        auditor.record_veto(c2, v2)

        # Mock created timestamps to be 2 hours in the past
        auditor._fetch_binance_klines = lambda sym, limit=100: (
            # Return synthetic path for BTC: dumps to 58500 (hits SL!)
            [
                {"timestamp": (now_ts + 600) * 1000, "open": 60000, "high": 60100, "low": 58800, "close": 58900}
            ] if "BTC" in sym else
            # Return synthetic path for ETH: rallies to 3250 (hits TP!)
            [
                {"timestamp": (now_ts + 600) * 1000, "open": 3000, "high": 3220, "low": 2980, "close": 3210}
            ]
        )

        # Adjust created_ts on disk
        with open(auditor.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            d["created_ts"] = now_ts
        with open(auditor.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Audit pending
        settled_count = auditor.audit_pending_vetoes(min_age_minutes=30.0)
        self.assertEqual(settled_count, 2)

        # Verify performance metrics
        perf = auditor.calculate_veto_performance(days=7)
        self.assertEqual(perf["total_audited"], 2)
        self.assertEqual(perf["correct_vetoes"], 1)
        self.assertEqual(perf["false_vetoes"], 1)
        self.assertEqual(perf["accuracy_pct"], 50.0)
        self.assertGreater(perf["capital_saved_pct"], 1.0)
        self.assertGreater(perf["missed_gain_pct"], 5.0)

        # Check specialist breakdown
        specs = perf["specialist_breakdown"]
        self.assertIn("derivatives_specialist", specs)
        self.assertEqual(specs["derivatives_specialist"]["correct"], 1)
        self.assertIn("risk_specialist", specs)
        self.assertEqual(specs["risk_specialist"]["false"], 1)

    def test_veto_penalty_dynamic_override_in_council(self):
        """Test TradingCouncil overriding a penalized specialist's veto when non-veto confidence is high."""
        bb_file = os.path.join(self.test_dir, "test_bb.json")
        bb = CentralBlackboard(file_path=bb_file)

        # Inject directive with active veto penalty against risk_specialist
        bb.update_directives({
            "veto_penalties": {
                "risk_specialist": {
                    "active": True,
                    "reason": "Veto accuracy critically low (20%), false stops observed.",
                }
            }
        })

        council = TradingCouncil(blackboard=bb, min_composite_score=60.0)

        # Candidate
        candidate = {
            "symbol": "BTC-USD",
            "signal": "BUY",
            "confidence": 85.0,
            "strategy": "CDCVIX15",
            "plan": {"entry_price": 60000, "stop_loss": 59000, "take_profit": 62000, "risk_reward": 2.0},
        }

        # Mock risk_specialist vetoing, but others approving with 80%+
        council.risk_specialist.evaluate = lambda *args, **kwargs: CouncilOpinion(
            specialist_name="risk_specialist",
            status="REJECTED",
            confidence=20.0,
            veto=True,
            veto_reason="risk_drawdown_limit",
            note="Too risky",
        )
        council.alpha_specialist.evaluate = lambda *args, **kwargs: CouncilOpinion(
            specialist_name="alpha_specialist", status="APPROVED", confidence=85.0
        )
        council.macro_specialist.evaluate = lambda *args, **kwargs: CouncilOpinion(
            specialist_name="macro_specialist", status="APPROVED", confidence=80.0
        )
        council.derivatives_specialist.evaluate = lambda *args, **kwargs: CouncilOpinion(
            specialist_name="derivatives_specialist", status="APPROVED", confidence=82.0
        )

        verdict = council.deliberate(candidate)
        # Because risk_specialist has an active penalty and non-veto confidence is ~82%,
        # the council should OVERRIDE the veto and EXECUTE!
        self.assertTrue(verdict.approved)
        self.assertEqual(verdict.decision, "EXECUTE")
        self.assertIsNone(verdict.veto_specialist)

    def test_alpha_forge_session_features(self):
        """Test AlphaForge features includes session indicators."""
        now_ms = int(time.time() * 1000)
        df_raw = pd.DataFrame({
            "timestamp": [now_ms - i * 900000 for i in range(50)][::-1],
            "open": [100.0 + i * 0.1 for i in range(50)],
            "high": [101.0 + i * 0.1 for i in range(50)],
            "low": [99.5 + i * 0.1 for i in range(50)],
            "close": [100.8 + i * 0.1 for i in range(50)],
            "volume": [1000.0] * 50,
        })

        df_enriched = prepare_market_features(df_raw)
        self.assertIn("is_ny_session", df_enriched.columns)
        self.assertIn("is_london_session", df_enriched.columns)
        self.assertIn("is_opening_surge", df_enriched.columns)

        # Test evaluating expression with session surge
        expr = "is_opening_surge * rel_vol * (lower_wick - upper_wick) / (atr14 + 1e-6)"
        res = evaluate_alpha_expression(expr, df_enriched)
        self.assertEqual(len(res), len(df_enriched))
        self.assertIsInstance(res, pd.Series)


if __name__ == "__main__":
    unittest.main()
