#!/usr/bin/env python3
"""
Unit and Integration Tests for Daily Council Debrief & Adaptive Council Memory.
Verifies:
1. Stats aggregation from outcomes and deliberations.
2. Deterministic & LLM dialogue synthesis.
3. council_memory.json atomic persistence.
4. CentralBlackboard directive merging (Doctor + Council Memory).
5. TradingCouncil dynamic threshold adjustment based on memory directives.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infrastructure.council.blackboard import CentralBlackboard
from infrastructure.council.daily_debrief import DailyCouncilDebrief
from infrastructure.council.arbiter import TradingCouncil
from infrastructure.council.specialists import (
    AlphaSpecialist,
    MacroSpecialist,
    DerivativesSpecialist,
    RiskSpecialist,
    OpsSpecialist,
    CouncilOpinion,
)


class TestDailyCouncilDebrief(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="tradv2_debrief_test_")
        self.delib_path = os.path.join(self.test_dir, "council_deliberations.json")
        self.outcomes_path = os.path.join(self.test_dir, "realized_outcomes.json")
        self.risk_state_path = os.path.join(self.test_dir, "risk_state.json")
        self.memory_path = os.path.join(self.test_dir, "council_memory.json")

        # Seed outcomes
        outcomes_data = {
            "version": 1,
            "outcomes": [
                {
                    "alert_id": "trade1",
                    "timestamp": "2026-10-17 02:00:00 UTC",
                    "symbol": "BTC-USD",
                    "signal": "BUY",
                    "outcome_status": "settled",
                    "outcome_result": "win",
                    "exit_reason": "take_profit_hit",
                    "pnl_pct": 2.50,
                    "rr_realized": 1.85,
                },
                {
                    "alert_id": "trade2",
                    "timestamp": "2026-10-17 04:00:00 UTC",
                    "symbol": "SUI-USD",
                    "signal": "BUY",
                    "outcome_status": "settled",
                    "outcome_result": "win",
                    "exit_reason": "trailing_stop_hit",
                    "pnl_pct": 4.10,
                    "rr_realized": 3.20,
                },
                {
                    "alert_id": "trade3",
                    "timestamp": "2026-10-17 06:00:00 UTC",
                    "symbol": "NEAR-USD",
                    "signal": "BUY",
                    "outcome_status": "settled",
                    "outcome_result": "loss",
                    "exit_reason": "stop_loss_hit",
                    "pnl_pct": -1.25,
                    "rr_realized": -1.00,
                    "mae_pct": 1.45,
                },
                {
                    "alert_id": "trade4",
                    "timestamp": "2026-10-17 07:00:00 UTC",
                    "symbol": "ADA-USD",
                    "signal": "BUY",
                    "outcome_status": "settled",
                    "outcome_result": "breakeven",
                    "exit_reason": "breakeven_stop_hit",
                    "pnl_pct": 0.05,
                    "rr_realized": 0.04,
                },
            ],
        }
        with open(self.outcomes_path, "w", encoding="utf-8") as f:
            json.dump(outcomes_data, f, indent=2)

        # Seed deliberations
        delib_data = {
            "version": 1,
            "deliberations": [
                {
                    "round_id": "rnd_1",
                    "symbol": "BTC-USD",
                    "signal": "BUY",
                    "verdict": {"decision": "EXECUTE", "composite_score": 82.0},
                },
                {
                    "round_id": "rnd_2",
                    "symbol": "SOL-USD",
                    "signal": "BUY",
                    "verdict": {
                        "decision": "VETO",
                        "composite_score": 45.0,
                        "veto_specialist": "derivatives_specialist",
                        "veto_reason": "funding_rate_overcrowded",
                    },
                },
            ],
        }
        with open(self.delib_path, "w", encoding="utf-8") as f:
            json.dump(delib_data, f, indent=2)

        # Seed risk state
        risk_data = {
            "consecutive_losses": 1,
            "consecutive_wins": 0,
            "daily_loss_pct": 0.50,
            "circuit_breaker_active": False,
        }
        with open(self.risk_state_path, "w", encoding="utf-8") as f:
            json.dump(risk_data, f, indent=2)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_stats_collection(self):
        debrief = DailyCouncilDebrief(data_dir=self.test_dir)
        stats = debrief.collect_trade_and_council_stats(lookback_hours=24.0)

        self.assertEqual(stats["total_settled"], 4)
        self.assertEqual(stats["wins"], 2)
        self.assertEqual(stats["losses"], 1)
        self.assertEqual(stats["breakevens"], 1)
        self.assertEqual(stats["trailing_hits"], 1)
        self.assertAlmostEqual(stats["win_rate_pct"], 66.7, places=1)
        self.assertGreater(stats["net_r"], 0.0)
        self.assertEqual(stats["delib_executed"], 1)
        self.assertEqual(stats["delib_vetoed"], 1)
        self.assertEqual(stats["vetoes_by_spec"].get("derivatives_specialist"), 1)

    def test_debrief_run_and_memory_persistence(self):
        debrief = DailyCouncilDebrief(data_dir=self.test_dir)
        memory = debrief.run_debrief(lookback_hours=24.0, notify_telegram=False, use_gemini=False)

        # Check memory structure
        self.assertTrue(os.path.exists(self.memory_path))
        self.assertIn("session_id", memory)
        self.assertIn("market_regime", memory)
        self.assertIn("risk_stance", memory)
        self.assertIn("min_composite_score", memory)
        self.assertIn("active_directives", memory)
        self.assertIn("NEAR-USD", memory["active_directives"]["flagged_symbols"])
        self.assertGreater(len(memory["lessons_learned"]), 0)
        self.assertGreater(len(memory["dialogue_transcript"]), 0)

        # Test Telegram message formatting
        tg_msg = debrief.format_telegram_message(memory)
        self.assertIn("บันทึกการประชุมสภาประจำวัน", tg_msg)
        self.assertIn("ไฮไลท์การอภิปรายโต๊ะกลม", tg_msg)
        self.assertIn("บทเรียนสำคัญ", tg_msg)

    def test_blackboard_and_arbiter_memory_integration(self):
        # 1. Run debrief to produce council_memory.json
        debrief = DailyCouncilDebrief(data_dir=self.test_dir)
        debrief.run_debrief(lookback_hours=24.0, notify_telegram=False, use_gemini=False)

        # 2. Check that CentralBlackboard loads directives from council_memory.json
        bb = CentralBlackboard(file_path=self.delib_path, memory_file_path=self.memory_path)
        directives = bb.get_active_directives()

        self.assertIn("flagged_symbols", directives)
        self.assertIn("NEAR-USD", directives["flagged_symbols"])
        self.assertIn("min_composite_score", directives)

        # 3. Test TradingCouncil using these memory directives
        council = TradingCouncil(blackboard=bb, min_composite_score=60.0)

        # Candidate for NEAR-USD (which is flagged)
        candidate_near = {
            "symbol": "NEAR-USD",
            "strategy": "CDCVIX15",
            "signal": "BUY",
            "dispatch_status_label": "เข้าได้",
            "entry_price": 5.0,
            "stop_loss": 4.8,
            "take_profit": 5.5,
            "alert_id": "test_alert_near",
        }
        verdict_near = council.deliberate(candidate_near)

        # Alpha specialist should have noted the flag from Council Memory directive
        opinions = verdict_near.opinions
        self.assertIn("Flagged", opinions["alpha_specialist"]["note"])

    def test_defensive_stance_tightening(self):
        # Seed consecutive losses to trigger DEFENSIVE stance
        with open(self.risk_state_path, "w", encoding="utf-8") as f:
            json.dump({"consecutive_losses": 3, "daily_loss_pct": 3.5}, f)

        debrief = DailyCouncilDebrief(data_dir=self.test_dir)
        memory = debrief.run_debrief(lookback_hours=24.0, notify_telegram=False, use_gemini=False)

        self.assertEqual(memory["risk_stance"], "DEFENSIVE")
        self.assertEqual(memory["market_regime"], "SIDEWAY_CHOP")
        self.assertGreaterEqual(memory["min_composite_score"], 70.0)
        self.assertTrue(memory["active_directives"]["caution_mode"])
        self.assertEqual(memory["active_directives"]["tp_target_multiplier"], 0.85)
        self.assertEqual(memory["active_directives"]["breakeven_trigger_r"], 1.0)

    def test_zero_settled_trades_stagnation_release(self):
        # Empty outcomes: market was quiet, zero trades settled
        with open(self.outcomes_path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "outcomes": []}, f)
        # Reset risk state to 0 losses
        with open(self.risk_state_path, "w", encoding="utf-8") as f:
            json.dump({"consecutive_losses": 0, "daily_loss_pct": 0.0}, f)

        debrief = DailyCouncilDebrief(data_dir=self.test_dir)
        stats = debrief.collect_trade_and_council_stats(lookback_hours=24.0)
        self.assertEqual(stats["total_settled"], 0)

        memory = debrief.run_debrief(lookback_hours=24.0, notify_telegram=False, use_gemini=False)

        # CRITICAL TEST: Stagnation Trap Deadlock must be broken!
        self.assertEqual(memory["market_regime"], "CALM_CONSOLIDATION")
        self.assertEqual(memory["risk_stance"], "NORMAL")
        self.assertEqual(memory["min_composite_score"], 65.0)
        self.assertFalse(memory["active_directives"]["caution_mode"])

        # Check dynamic directives
        directives = memory["active_directives"]
        self.assertEqual(directives["tp_target_multiplier"], 1.0)
        self.assertEqual(directives["breakeven_trigger_r"], 1.2)
        self.assertEqual(directives["trailing_activation_r"], 2.0)
        self.assertEqual(directives["max_entry_drift_pct"], 0.80)

        # Verify Telegram message does NOT say robotic "Win Rate 0.0%"
        tg_msg = debrief.format_telegram_message(memory)
        self.assertNotIn("Win Rate สุทธิ: 0.0%", tg_msg)
        self.assertIn("0 ไม้ (ไม่มีการชน SL / เงินทุนปลอดภัย 100%)", tg_msg)
        self.assertIn("CALM_CONSOLIDATION", tg_msg)
        self.assertIn("TP Scale = 1.00x", tg_msg)
        self.assertIn("กันทุน BE = +1.2R", tg_msg)


if __name__ == "__main__":
    unittest.main()
