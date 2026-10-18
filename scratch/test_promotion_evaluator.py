"""
Unit tests for Auto-Promotion Evaluator (CouncilPromotionEvaluator)
and its integration into DailyCouncilDebrief, CentralBlackboard, and AlphaSpecialist.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infrastructure.council.promotion_evaluator import CouncilPromotionEvaluator
from infrastructure.council.daily_debrief import DailyCouncilDebrief
from infrastructure.council.blackboard import CentralBlackboard
from infrastructure.council.specialists import AlphaSpecialist, CouncilOpinion


class TestCouncilPromotionEvaluator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_prom_eval_")
        self.factors_path = os.path.join(self.test_dir, "active_alpha_factors.json")
        self.promoted_path = os.path.join(self.test_dir, "promoted_production_factors.json")
        self.executed_path = os.path.join(self.test_dir, "executed_orders.json")
        self.outcomes_path = os.path.join(self.test_dir, "realized_outcomes.json")
        self.delib_path = os.path.join(self.test_dir, "council_deliberations.json")
        self.memory_path = os.path.join(self.test_dir, "council_memory.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_factor_qualification_and_promotion(self):
        """Test that factors meeting live sandbox criteria graduate to Production."""
        # 1. Setup 3 Active Sandbox Factors
        factors_data = {
            "factors": {
                "alpha_liquidity_vacuum": {
                    "factor_id": "alpha_liquidity_vacuum",
                    "factor_name": "Liquidity Absorption & Wick Vacuum",
                    "formula": "wick_asymmetry * rel_vol",
                    "threshold": 1.5,
                    "status": "active_sandbox_exploration",
                },
                "alpha_young_factor": {
                    "factor_id": "alpha_young_factor",
                    "factor_name": "Kinetic Range Acceleration",
                    "formula": "delta(close, 3) / atr14",
                    "threshold": 2.0,
                    "status": "active_sandbox_exploration",
                },
                "alpha_struggling_factor": {
                    "factor_id": "alpha_struggling_factor",
                    "factor_name": "Poor Reversal Exhaustion",
                    "formula": "rsi14 - 50",
                    "threshold": 0.0,
                    "status": "active_sandbox_exploration",
                },
            }
        }
        with open(self.factors_path, "w", encoding="utf-8") as f:
            json.dump(factors_data, f)

        # 2. Simulate Realized Sandbox Trades
        outcomes_list = []

        # Factor 1: 12 live trades (8 wins +3.0%, 4 losses -1.0%, interleaved so max consec losses = 1)
        # Total profit = +24.0%, total loss = -4.0% -> PF = 6.0, WR = 66.7%, Exp = +1.67%
        pattern = [3.0, 3.0, -1.0, 3.0, 3.0, -1.0, 3.0, 3.0, -1.0, 3.0, 3.0, -1.0]
        for pnl in pattern:
            outcomes_list.append({
                "symbol": "SOL-USD",
                "is_sandbox_exploration": True,
                "alpha_factor_id": "alpha_liquidity_vacuum",
                "outcome_status": "settled",
                "pnl_pct": pnl,
            })

        # Factor 2 (Young): Only 4 live trades (insufficient sample, min is 10)
        for i in range(4):
            outcomes_list.append({
                "symbol": "BTC-USD",
                "is_sandbox_exploration": True,
                "alpha_factor_id": "alpha_young_factor",
                "outcome_status": "settled",
                "pnl_pct": 2.5,
            })

        # Factor 3 (Struggling): 11 live trades, but 8 losses, 3 wins
        for i in range(3):
            outcomes_list.append({
                "symbol": "ETH-USD",
                "is_sandbox_exploration": True,
                "alpha_factor_id": "alpha_struggling_factor",
                "outcome_status": "settled",
                "pnl_pct": 1.0,
            })
        for i in range(8):
            outcomes_list.append({
                "symbol": "ETH-USD",
                "is_sandbox_exploration": True,
                "alpha_factor_id": "alpha_struggling_factor",
                "outcome_status": "settled",
                "pnl_pct": -2.0,
            })

        with open(self.outcomes_path, "w", encoding="utf-8") as f:
            json.dump({"outcomes": outcomes_list}, f)

        # 3. Run Evaluation
        evaluator = CouncilPromotionEvaluator(
            factors_path=self.factors_path,
            promoted_path=self.promoted_path,
            executed_path=self.executed_path,
            outcomes_path=self.outcomes_path,
            min_live_trades=10,
            min_live_pf=1.45,
            min_live_win_rate=48.0,
            min_net_expectancy=0.15,
        )

        res = evaluator.evaluate_promotions()

        # 4. Verify Outcomes
        self.assertEqual(res["newly_promoted_count"], 1)
        self.assertEqual(res["active_promoted_count"], 1)

        promoted_factor = res["newly_promoted"][0]
        self.assertEqual(promoted_factor["factor_id"], "alpha_liquidity_vacuum")
        self.assertEqual(promoted_factor["role"], "PRE_COUNCIL_ALPHA_GATE")
        self.assertGreater(promoted_factor["live_stats"]["profit_factor"], 1.45)
        self.assertGreater(promoted_factor["live_stats"]["win_rate_pct"], 50.0)

        # Verify Young factor was not promoted due to insufficient trades
        eval_young = res["evaluations"]["alpha_young_factor"]
        self.assertFalse(eval_young["qualifies_for_promotion"])
        self.assertEqual(eval_young["status"], "active_sandbox_exploration")

        # Verify Struggling factor was not promoted due to low win rate and negative expectancy
        eval_struggle = res["evaluations"]["alpha_struggling_factor"]
        self.assertFalse(eval_struggle["qualifies_for_promotion"])

        # 5. Verify Disk Persistence
        with open(self.promoted_path, "r", encoding="utf-8") as f:
            saved_promoted = json.load(f)
        self.assertIn("alpha_liquidity_vacuum", saved_promoted["promoted_factors"])

        with open(self.factors_path, "r", encoding="utf-8") as f:
            saved_factors = json.load(f)
        self.assertEqual(saved_factors["factors"]["alpha_liquidity_vacuum"]["status"], "promoted_to_production")

    def test_daily_debrief_integration_with_promotion(self):
        """Test DailyCouncilDebrief detects qualifying sandbox factors and broadcasts graduation."""
        # Setup qualifying factor
        factors_data = {
            "factors": {
                "alpha_liquidity_vacuum": {
                    "factor_id": "alpha_liquidity_vacuum",
                    "factor_name": "Liquidity Absorption & Wick Vacuum",
                    "formula": "wick_asymmetry * rel_vol",
                    "status": "active_sandbox_exploration",
                }
            }
        }
        with open(self.factors_path, "w", encoding="utf-8") as f:
            json.dump(factors_data, f)

        # 10 winning sandbox trades
        outcomes_list = [
            {
                "symbol": "SOL-USD",
                "is_sandbox_exploration": True,
                "alpha_factor_id": "alpha_liquidity_vacuum",
                "outcome_status": "settled",
                "pnl_pct": 2.5,
                "r_multiple": 1.5,
            }
            for _ in range(10)
        ]
        with open(self.outcomes_path, "w", encoding="utf-8") as f:
            json.dump({"outcomes": outcomes_list}, f)

        # Mock custom config pointing to test files
        class MockConfig:
            ALPHA_FORGE_ACTIVE_FACTORS_PATH = self.factors_path
            COUNCIL_PROMOTED_FACTORS_PATH = self.promoted_path
            COUNCIL_PROMOTION_MIN_TRADES = 10
            COUNCIL_PROMOTION_MIN_PF = 1.45
            COUNCIL_PROMOTION_MIN_WIN_RATE = 48.0
            COUNCIL_MEMORY_PATH = self.memory_path
            TRADING_COUNCIL_DELIBERATIONS_PATH = self.delib_path

        debrief = DailyCouncilDebrief(
            data_dir=self.test_dir,
            deliberations_path=self.delib_path,
            memory_path=self.memory_path,
            config=MockConfig(),
        )

        # Run debrief
        memory = debrief.run_debrief(lookback_hours=24.0)

        # 1. Check Contrarian Scout Dialogue
        transcript = memory.get("dialogue_transcript") or []
        scout_turns = [t for t in transcript if "Contrarian Scout" in t.get("speaker", "")]
        self.assertTrue(len(scout_turns) > 0)
        self.assertIn("Liquidity Absorption & Wick Vacuum", scout_turns[0]["text"])
        self.assertIn("ผ่านเกณฑ์การบ่มเพาะ", scout_turns[0]["text"])

        # 2. Check Arbiter Directives & Blackboard
        directives = memory.get("active_directives") or {}
        self.assertIn("promoted_alpha_gates", directives)
        self.assertEqual(len(directives["promoted_alpha_gates"]), 1)
        self.assertEqual(directives["promoted_alpha_gates"][0]["factor_id"], "alpha_liquidity_vacuum")

        # 3. Check Telegram Message Format
        tg_text = debrief.format_telegram_message(memory)
        self.assertIn("สภาอนุมัติเลื่อนขั้นสูตรคณิตศาสตร์ (Auto-Promotion)", tg_text)
        self.assertIn("Liquidity Absorption & Wick Vacuum", tg_text)
        self.assertIn("เลื่อนขั้นเป็น Pre-Council Alpha Gate เรียบร้อย", tg_text)

    def test_alpha_specialist_boost_on_promoted_gate(self):
        """Test AlphaSpecialist grants confidence bonus when candidate matches promoted alpha gate."""
        spec = AlphaSpecialist()

        candidate = {
            "symbol": "SOL-USD",
            "signal": "BUY",
            "strategy": "CDCVIX15",
            "dispatch_status_label": "เข้าได้",
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 156.0,  # rr = 6/5 = 1.2 (no rr >= 1.5 boost)
            "promoted_alpha_verified": True,
        }

        # 1. Deliberation with Promoted Alpha Gate active in directives
        directives_with_gate = {
            "promoted_alpha_gates": [
                {
                    "factor_id": "alpha_liquidity_vacuum",
                    "factor_name": "Liquidity Absorption & Wick Vacuum",
                }
            ]
        }
        opinion_boosted = spec.evaluate(candidate, directives=directives_with_gate)

        # 2. Deliberation without Promoted Alpha Gate
        opinion_normal = spec.evaluate(
            {**candidate, "promoted_alpha_verified": False},
            directives={}
        )

        # The boosted opinion must be higher by exactly 15%
        self.assertEqual(opinion_boosted.confidence - opinion_normal.confidence, 15.0)
        self.assertIn("ผ่านเกณฑ์สูตรหลัก", opinion_boosted.note)

    def test_auto_demotion_on_regime_failure(self):
        """Test that a promoted production factor is demoted back to sandbox if it fails in production."""
        # Setup promoted factor
        factors_data = {
            "factors": {
                "alpha_liquidity_vacuum": {
                    "factor_id": "alpha_liquidity_vacuum",
                    "factor_name": "Liquidity Absorption & Wick Vacuum",
                    "status": "promoted_to_production",
                    "promoted_at": "2026-10-18 00:00:00 UTC",
                }
            }
        }
        with open(self.factors_path, "w", encoding="utf-8") as f:
            json.dump(factors_data, f)

        promoted_data = {
            "version": 1,
            "promoted_factors": {
                "alpha_liquidity_vacuum": {
                    "factor_id": "alpha_liquidity_vacuum",
                    "factor_name": "Liquidity Absorption & Wick Vacuum",
                    "promoted_at": "2026-10-18 00:00:00 UTC",
                }
            }
        }
        with open(self.promoted_path, "w", encoding="utf-8") as f:
            json.dump(promoted_data, f)

        # 3 consecutive losses in live market regime shift
        outcomes_list = [
            {"symbol": "SOL-USD", "alpha_factor_id": "alpha_liquidity_vacuum", "outcome_status": "settled", "pnl_pct": -1.5},
            {"symbol": "SOL-USD", "alpha_factor_id": "alpha_liquidity_vacuum", "outcome_status": "settled", "pnl_pct": -2.0},
            {"symbol": "SOL-USD", "alpha_factor_id": "alpha_liquidity_vacuum", "outcome_status": "settled", "pnl_pct": -1.8},
        ]
        with open(self.outcomes_path, "w", encoding="utf-8") as f:
            json.dump({"outcomes": outcomes_list}, f)

        evaluator = CouncilPromotionEvaluator(
            factors_path=self.factors_path,
            promoted_path=self.promoted_path,
            executed_path=self.executed_path,
            outcomes_path=self.outcomes_path,
        )

        res = evaluator.evaluate_promotions()

        # Verify Auto-Demotion
        self.assertEqual(res["newly_demoted_count"], 1)
        self.assertEqual(res["active_promoted_count"], 0)
        demoted = res["newly_demoted"][0]
        self.assertEqual(demoted["factor_id"], "alpha_liquidity_vacuum")
        self.assertEqual(demoted["max_consecutive_losses"], 3)

        # Verify Disk updates
        with open(self.promoted_path, "r", encoding="utf-8") as f:
            saved_prom = json.load(f)
        self.assertNotIn("alpha_liquidity_vacuum", saved_prom["promoted_factors"])

        with open(self.factors_path, "r", encoding="utf-8") as f:
            saved_f = json.load(f)
        self.assertEqual(saved_f["factors"]["alpha_liquidity_vacuum"]["status"], "active_sandbox_exploration")


if __name__ == "__main__":
    unittest.main()
