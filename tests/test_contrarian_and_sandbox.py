import unittest
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infrastructure.council import (
    CentralBlackboard,
    TradingCouncil,
    ContrarianScout,
    AlphaSpecialist,
    MacroSpecialist,
    DerivativesSpecialist,
    RiskSpecialist,
    OpsSpecialist,
    NewsSentinel,
    CouncilOpinion,
)
from infrastructure.council.daily_debrief import DailyCouncilDebrief
from infrastructure.binance.order_manager import BinanceFuturesOrderManager


class TestContrarianAndSandbox(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.delib_path = os.path.join(self.temp_dir.name, "council_deliberations.json")
        self.memory_path = os.path.join(self.temp_dir.name, "council_memory.json")
        self.blackboard = CentralBlackboard(file_path=self.delib_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_contrarian_scout_high_rr_exploration(self):
        scout = ContrarianScout(min_asymmetric_rr=2.8, sandbox_size_multiplier=0.35)
        # Entry 100, Stop 95 (risk 5), Take Profit 115 (reward 15) -> R:R = 3.0
        candidate = {
            "symbol": "SUI-USD",
            "signal": "BUY",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 115.0,
        }
        op = scout.evaluate(candidate, open_explorations_count=0)
        self.assertEqual(op.status, "APPROVED")
        self.assertEqual(op.specialist_name, "contrarian_scout")
        self.assertTrue(op.metrics.get("is_exploration"))
        self.assertAlmostEqual(op.metrics.get("rr_ratio"), 3.0, places=1)
        self.assertEqual(op.metrics.get("sandbox_size_multiplier"), 0.35)
        self.assertIn("R:R=3.00x", op.note)

    def test_contrarian_scout_short_squeeze_thesis(self):
        scout = ContrarianScout(min_asymmetric_rr=2.8, sandbox_size_multiplier=0.35)
        # R:R is standard 1.5, but funding rate is heavily negative (-0.025%) -> Short squeeze!
        candidate = {
            "symbol": "ADA-USD",
            "signal": "BUY",
            "entry_price": 0.50,
            "stop_loss": 0.48,
            "take_profit": 0.53,
            "derivatives_pulse": {
                "funding_rate": -0.00025,
                "oi_delta_pct": 2.5,
            },
        }
        op = scout.evaluate(candidate, open_explorations_count=0)
        self.assertEqual(op.status, "APPROVED")
        self.assertTrue(op.metrics.get("is_exploration"))
        self.assertIn("Short Squeeze Setup", op.metrics.get("thesis"))

    def test_contrarian_scout_quota_exhausted(self):
        scout = ContrarianScout(min_asymmetric_rr=2.8)
        candidate = {
            "symbol": "SUI-USD",
            "signal": "BUY",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 115.0,
        }
        # Already 1 open exploration running
        op = scout.evaluate(candidate, open_explorations_count=1, max_explorations_allowed=1)
        self.assertFalse(op.metrics.get("is_exploration"))
        self.assertEqual(op.status, "NEUTRAL")

    def test_trading_council_sandbox_override(self):
        council = TradingCouncil(
            blackboard=self.blackboard,
            min_composite_score=65.0,
        )

        # Candidate with R:R = 3.2x, but standard composite score is ~55-60%
        # It would normally be STANDBY (< 65%), but Sandbox Exploration approves it with micro-sizing!
        candidate = {
            "symbol": "SUI-USD",
            "signal": "BUY",
            "strategy": "CDCVIX15",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 118.0,
            "rr_ratio": 3.6,
            "is_exploration_candidate": True,
            "alert_id": "test_sui_exp_01",
        }

        verdict = council.deliberate(
            candidate,
            total_equity=1000.0,
            open_positions_count=0,
            open_explorations_count=0,
        )

        self.assertTrue(verdict.approved)
        self.assertEqual(verdict.decision, "EXECUTE")
        self.assertTrue(verdict.is_sandbox_exploration)
        self.assertIn("Sandbox Exploration", verdict.telegram_badge)
        self.assertEqual(verdict.sizing_info.get("multiplier"), 0.35)
        # 400 * 0.35 = 140 USDT
        self.assertAlmostEqual(verdict.sizing_info.get("final_notional"), 140.0, places=1)

    def test_rescued_candidate_from_avoid_status(self):
        # Candidate labeled "ห้ามเข้า" by conservative pre-filters, but with high R:R
        candidate = {
            "symbol": "SUI-USD",
            "signal": "BUY",
            "strategy": "CDCVIX15",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 118.0,
            "rr_ratio": 3.6,
            "dispatch_status_label": "ห้ามเข้า",
            "is_exploration_candidate": True,
            "alert_id": "test_rescued_01",
        }

        council = TradingCouncil(blackboard=self.blackboard, min_composite_score=65.0)
        verdict = council.deliberate(
            candidate,
            total_equity=1000.0,
            open_positions_count=0,
            open_explorations_count=0,
        )

        # AlphaSpecialist did not hard veto; Arbiter approved for Sandbox Exploration!
        self.assertIsNone(verdict.veto_specialist)
        self.assertTrue(verdict.approved)
        self.assertTrue(verdict.is_sandbox_exploration)
        self.assertEqual(verdict.decision, "EXECUTE")


    def test_order_manager_sizing_override(self):
        mock_client = MagicMock()
        mock_client.get_positions.return_value = []
        mock_client.get_symbol_filters.return_value = {
            "tickSize": "0.01",
            "stepSize": "0.1",
            "minQty": "0.1",
            "minNotional": "5.0",
        }
        mock_client.create_order.return_value = {
            "success": True,
            "data": {
                "orderId": 12345,
                "status": "FILLED",
                "avgPrice": "100.0",
            },
        }
        mock_client.create_algo_order.return_value = {
            "success": True,
            "data": {"algoId": 6789},
        }

        mgr = BinanceFuturesOrderManager(
            client=mock_client,
            trade_notional_usdt=400.0,
            leverage=20,
            margin_type="ISOLATED",
        )

        candidate = {
            "symbol": "SUI-USD",
            "signal": "BUY",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 116.0,
        }

        # Provide sizing override for sandbox (140 USDT)
        sizing_override = {
            "final_notional": 140.0,
            "multiplier": 0.35,
            "multiplier_reason": "sandbox_exploration_bet_0.35x",
        }

        res = mgr.execute_candidate(candidate, sizing_override=sizing_override)
        self.assertTrue(res["success"])
        # Quantity: 140 / 100 = 1.4
        self.assertAlmostEqual(res["quantity"], 1.4, places=2)
        self.assertAlmostEqual(res["notional"], 140.0, places=1)

    def test_daily_debrief_contrarian_turn_and_summary(self):
        debrief = DailyCouncilDebrief(
            memory_path=self.memory_path,
            deliberations_path=self.delib_path,
        )

        stats = {
            "total_settled": 5,
            "wins": 3,
            "losses": 1,
            "breakevens": 1,
            "win_rate_pct": 60.0,
            "profit_factor": 2.5,
            "net_pnl_pct": 4.5,
            "net_r": 2.2,
            "delib_vetoed": 2,
            "sandbox_executed_count": 1,
            "loss_details": [],
            "win_details": [],
            "consecutive_losses": 0,
        }

        dialogue, meta = debrief._generate_deterministic_dialogue(stats)
        # Check that Contrarian Scout spoke
        speakers = [turn["speaker"] for turn in dialogue]
        self.assertTrue(any("Contrarian Scout" in s for s in speakers))

        # Check synthesis and telegram message formatting
        memory = debrief.synthesize_council_memory(stats, dialogue, meta)
        tg_msg = debrief.format_telegram_message(memory)

        self.assertIn("ไม้ทดลอง 🧪: <b>1 ไม้</b>", tg_msg)
        self.assertIn("Contrarian Scout", tg_msg)


    def test_scout_researcher_panic_overreaction(self):
        from infrastructure.council.scout_researcher import ScoutResearcher
        researcher = ScoutResearcher(intel_path=os.path.join(self.temp_dir.name, "intel.json"))
        # Mock cross-market snapshot: S&P down 0.3%, BTC down 1.2%
        market_snapshot = {
            "sp500_change_24h_pct": -0.3,
            "btc_change_24h_pct": -1.2,
            "gold_paxg_change_24h_pct": 0.1,
            "macro_regime": "MILD_PULLBACK",
        }
        # Candidate dumps -4.5% with negative funding rate
        candidate = {
            "symbol": "SUI-USD",
            "signal": "BUY",
            "price_change_24h_pct": -4.5,
            "derivatives_pulse": {"funding_rate": -0.00015},
        }
        eval_res = researcher.evaluate_correlation(candidate, market_snapshot)
        self.assertTrue(eval_res["has_anomaly"])
        self.assertEqual(eval_res["anomaly_type"], "PANIC_OVERREACTION_SWEEP")
        self.assertGreater(eval_res["confidence_boost"], 0)
        self.assertIn("Overreaction", eval_res["anomaly_summary"])

    def test_scout_researcher_relative_strength_alpha(self):
        from infrastructure.council.scout_researcher import ScoutResearcher
        researcher = ScoutResearcher(intel_path=os.path.join(self.temp_dir.name, "intel.json"))
        market_snapshot = {
            "sp500_change_24h_pct": -0.5,
            "btc_change_24h_pct": -2.0,
            "macro_regime": "RISK_OFF",
        }
        # Candidate SUI is UP +1.5% while BTC is DOWN -2.0% (Alpha spread = +3.5%)
        candidate = {
            "symbol": "SUI-USD",
            "signal": "BUY",
            "price_change_24h_pct": 1.5,
        }
        eval_res = researcher.evaluate_correlation(candidate, market_snapshot)
        self.assertTrue(eval_res["has_anomaly"])
        self.assertEqual(eval_res["anomaly_type"], "RELATIVE_STRENGTH_ALPHA")
        self.assertIn("Relative Strength Alpha", eval_res["anomaly_summary"])

    def test_contrarian_scout_with_researcher_brief(self):
        from infrastructure.council.scout_researcher import ScoutResearcher
        researcher = ScoutResearcher(intel_path=os.path.join(self.temp_dir.name, "intel.json"))
        scout = ContrarianScout(
            researcher=researcher,
            min_asymmetric_rr=2.8,
            sandbox_size_multiplier=0.35,
        )

        # Candidate with R:R = 2.2x (below 2.8x standard, but >= 2.0x with researcher anomaly)
        candidate = {
            "symbol": "ONDO-USD",
            "signal": "BUY",
            "entry_price": 1.0,
            "stop_loss": 0.95,
            "take_profit": 1.11,  # R:R = 0.11 / 0.05 = 2.2x
            "price_change_24h_pct": 2.0,
        }
        # Market snapshot where BTC is dumping -3.0%
        market_intel = {
            "market_snapshot": {
                "btc_change_24h_pct": -3.0,
                "sp500_change_24h_pct": -0.8,
                "gold_paxg_change_24h_pct": 0.5,
            }
        }
        op = scout.evaluate(candidate, market_intel=market_intel, open_explorations_count=0)
        self.assertEqual(op.status, "APPROVED")
        self.assertTrue(op.metrics.get("is_exploration"))
        self.assertIn("เลขาตรวจพบความสัมพันธ์", op.note)
        self.assertIn("Relative Strength Alpha", op.metrics.get("thesis"))


if __name__ == "__main__":
    unittest.main()

