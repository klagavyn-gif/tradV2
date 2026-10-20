#!/usr/bin/env python3
"""
Unit test suite for Central Blackboard Health Audit, Rogue Trade Detection,
and Board Governance Directives.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from infrastructure.council import (
    CentralBlackboard,
    AlphaSpecialist,
    RiskSpecialist,
    OpsSpecialist,
    MacroSpecialist,
    DerivativesSpecialist,
    TradingCouncil,
)
from alerts.system_doctor import (
    check_central_blackboard_and_council,
    run_system_health_audit,
    format_doctor_alert_message,
)


class TestBlackboardHealthDoctor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        self.delib_file = self.data_dir / "council_deliberations.json"
        self.exec_file = self.data_dir / "binance_executed_orders.json"
        self.latest_run_file = self.data_dir / "latest_run.json"
        self.realized_summary_file = self.data_dir / "realized_summary.json"
        self.risk_state_file = self.data_dir / "risk_state.json"

        # Baseline healthy state files
        now_str = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S")
        with open(self.latest_run_file, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": now_str,
                "quality_drop_counts": {"missing_edge_metrics": 0},
            }, f)

        with open(self.realized_summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "settled_alerts": 10,
                "by_strategy": {"CDCVIX15": {"trades": 10, "win_rate": 65.0}},
            }, f)

        with open(self.risk_state_file, "w", encoding="utf-8") as f:
            json.dump({
                "circuit_breaker_active": False,
                "daily_pnl_pct": 1.5,
            }, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_clean_health_audit_and_blackboard_briefing(self):
        """Test clean system health audit, consensus metrics, and briefing posting."""
        bb = CentralBlackboard(file_path=str(self.delib_file))
        # Record 5 council rounds: 4 approved, 1 vetoed
        for i in range(1, 6):
            approved = (i != 5)
            bb.record_round({
                "round_id": f"round_{i}",
                "alert_id": f"alert_{i}",
                "symbol": "BTC-USD",
                "timestamp": datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S"),
                "approved": approved,
                "composite_score": 85.0 if approved else 50.0,
                "decision": "APPROVED" if approved else "VETOED",
                "specialist_opinions": {
                    "OpsSpecialist": {"veto": not approved, "reason": "test_veto" if not approved else None}
                }
            })

        # Executed orders file with alert_1 (which is round_1, approved)
        with open(self.exec_file, "w", encoding="utf-8") as f:
            json.dump({
                "alert_1": {
                    "alert_id": "alert_1",
                    "symbol": "BTC-USD",
                    "status": "open",
                    "executed_at": datetime.utcnow().isoformat(),
                    "council_round_id": "round_1",
                }
            }, f)

        # Run health audit
        report = run_system_health_audit(config, data_dir=self.data_dir, blackboard=bb)

        self.assertEqual(report["severity"], "OK")
        self.assertEqual(report["score"], 100.0)
        self.assertTrue(report["blackboard_posted"])

        # Check council check results
        bb_check = report["checks"]["central_blackboard"]
        self.assertEqual(bb_check["status"], "PASS")
        self.assertEqual(bb_check["council_stats"]["total_rounds_24h"], 5)
        self.assertEqual(bb_check["council_stats"]["approved_count"], 4)
        self.assertEqual(bb_check["council_stats"]["approval_rate_pct"], 80.0)
        self.assertEqual(len(bb_check["untracked_orders"]), 0)

        # Check directives
        dirs = report["directives"]
        self.assertFalse(dirs["caution_mode"])
        self.assertEqual(dirs["risk_multiplier_cap"], 1.25)
        self.assertIsNone(dirs["max_positions_cap"])
        self.assertEqual(dirs["flagged_symbols"], [])

        # Verify Blackboard persisted briefing and active directives
        briefing = bb.get_health_briefing()
        self.assertIsNotNone(briefing)
        self.assertEqual(briefing["health_severity"], "OK")
        self.assertEqual(briefing["health_score"], 100.0)

        active_dirs = bb.get_active_directives()
        self.assertFalse(active_dirs["caution_mode"])
        self.assertEqual(active_dirs["risk_multiplier_cap"], 1.25)

    def test_rogue_untracked_trade_detection(self):
        """Test detection of an untracked order executed on Binance without Council approval."""
        bb = CentralBlackboard(file_path=str(self.delib_file))
        bb.record_round({
            "round_id": "round_1",
            "alert_id": "legit_alert_1",
            "symbol": "ETH-USD",
            "approved": True,
            "composite_score": 85.0,
            "decision": "APPROVED",
        })

        # binance_executed_orders has a rogue trade
        with open(self.exec_file, "w", encoding="utf-8") as f:
            json.dump({
                "legit_alert_1": {
                    "alert_id": "legit_alert_1",
                    "symbol": "ETH-USD",
                    "status": "open",
                    "council_round_id": "round_1",
                },
                "rogue_alert_99": {
                    "alert_id": "rogue_alert_99",
                    "symbol": "DOGE-USD",
                    "status": "open",
                    "executed_at": datetime.utcnow().isoformat(),
                    "council_round_id": None,
                }
            }, f)

        report = run_system_health_audit(config, data_dir=self.data_dir, blackboard=bb)

        # Must flag CRITICAL due to open untracked trade
        self.assertEqual(report["severity"], "CRITICAL")
        self.assertLessEqual(report["score"], 50.0)
        bb_check = report["checks"]["central_blackboard"]
        self.assertEqual(bb_check["status"], "CRITICAL")
        self.assertEqual(len(bb_check["untracked_orders"]), 1)
        self.assertEqual(bb_check["untracked_orders"][0]["alert_id"], "rogue_alert_99")

        # Directives must enforce severe caution and risk cap
        dirs = report["directives"]
        self.assertTrue(dirs["caution_mode"])
        self.assertEqual(dirs["risk_multiplier_cap"], 0.50)
        self.assertEqual(dirs["max_positions_cap"], 1)
        self.assertIn("DOGE-USD", dirs["flagged_symbols"])

        # Blackboard must hold these directives
        active_dirs = bb.get_active_directives()
        self.assertTrue(active_dirs["caution_mode"])
        self.assertEqual(active_dirs["risk_multiplier_cap"], 0.50)
        self.assertEqual(active_dirs["max_positions_cap"], 1)
        self.assertIn("DOGE-USD", active_dirs["flagged_symbols"])

        # Telegram message format test
        msg = format_doctor_alert_message(report)
        self.assertIn("Trading Council", msg)
        self.assertIn("Caution Mode ON", msg)
        self.assertIn("Rogue Trade Detected", msg)

    def test_closed_untracked_trade_warning(self):
        """Test that a closed untracked trade triggers WARNING (not CRITICAL)."""
        bb = CentralBlackboard(file_path=str(self.delib_file))
        bb.record_round({
            "round_id": "round_1",
            "alert_id": "legit_alert_1",
            "symbol": "ETH-USD",
            "approved": True,
        })

        with open(self.exec_file, "w", encoding="utf-8") as f:
            json.dump({
                "closed_rogue_alert": {
                    "alert_id": "closed_rogue_alert",
                    "symbol": "SOL-USD",
                    "status": "closed",
                    "executed_at": datetime.utcnow().isoformat(),
                }
            }, f)

        report = run_system_health_audit(config, data_dir=self.data_dir, blackboard=bb)
        self.assertEqual(report["severity"], "WARNING")
        dirs = report["directives"]
        self.assertTrue(dirs["caution_mode"])
        self.assertEqual(dirs["risk_multiplier_cap"], 0.75)
        self.assertIsNone(dirs["max_positions_cap"])

    def test_council_lockup_deadlock_detection(self):
        """Test detection when council rejects 100% of signals over 6+ rounds."""
        bb = CentralBlackboard(file_path=str(self.delib_file))
        for i in range(1, 8):
            bb.record_round({
                "round_id": f"round_{i}",
                "alert_id": f"alert_{i}",
                "symbol": "ADA-USD",
                "timestamp": datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S"),
                "approved": False,
                "composite_score": 45.0,
                "decision": "VETOED",
                "specialist_opinions": {
                    "DerivativesSpecialist": {"veto": True, "reason": "overheated_funding"}
                }
            })

        report = run_system_health_audit(config, data_dir=self.data_dir, blackboard=bb)
        self.assertEqual(report["severity"], "WARNING")
        bb_check = report["checks"]["central_blackboard"]
        self.assertEqual(bb_check["status"], "WARNING")
        self.assertEqual(bb_check["council_stats"]["approved_count"], 0)
        self.assertEqual(bb_check["council_stats"]["specialist_vetoes"]["DerivativesSpecialist"], 7)
        lockup_issue = next((iss for iss in report["issues"] if iss.get("type") == "council_deliberation_lockup"), None)
        self.assertIsNotNone(lockup_issue)
        self.assertIn("DerivativesSpecialist", lockup_issue["message"])

    def test_specialists_and_trading_council_adapt_to_directives(self):
        """Test that specialists and Council deliberate with directives from Blackboard."""
        bb = CentralBlackboard(file_path=str(self.delib_file))
        # Post caution mode directives
        bb.post_health_briefing({
            "health_score": 45.0,
            "health_severity": "CRITICAL",
            "directives": {
                "caution_mode": True,
                "risk_multiplier_cap": 0.50,
                "max_positions_cap": 1,
                "flagged_symbols": ["DOGE-USD"],
                "health_score": 45.0,
                "health_severity": "CRITICAL",
                "audited_at": "2026-10-17 09:00:00",
            }
        })

        # 1. AlphaSpecialist confidence penalty on flagged symbol
        alpha = AlphaSpecialist()
        normal_cand = {"symbol": "BTC-USD", "strategy": "CDCVIX15", "signal": "BUY", "confidence": 70.0}
        flagged_cand = {"symbol": "DOGE-USD", "strategy": "CDCVIX15", "signal": "BUY", "confidence": 70.0}

        dirs = bb.get_active_directives()
        op_normal = alpha.evaluate(normal_cand, directives=dirs)
        op_flagged = alpha.evaluate(flagged_cand, directives=dirs)

        # Caution mode gives -10.0 to normal (75-10=65), and -20.0 extra to flagged symbol (75-10-20=45)
        self.assertEqual(op_normal.confidence, 65.0)
        self.assertEqual(op_flagged.confidence, 45.0)

        # 2. RiskSpecialist multiplier cap
        risk_spec = RiskSpecialist(default_notional=400.0, leverage=20)
        risk_op = risk_spec.evaluate(normal_cand, total_equity=1000.0, directives=dirs)
        self.assertLessEqual(risk_op.metrics["multiplier"], 0.50)

        # 3. OpsSpecialist max positions cap override
        ops_spec = OpsSpecialist(max_positions=3)
        # If open_positions_count is 1, normally max 3 would allow it. But cap is 1 -> VETO!
        ops_op = ops_spec.evaluate(normal_cand, open_positions_count=1, directives=dirs)
        self.assertTrue(ops_op.veto)
        self.assertEqual(ops_op.veto_reason, "max_positions_1_reached")
        self.assertIn("Capped by Blackboard Directive", ops_op.note)

        # 4. TradingCouncil deliberate with Blackboard
        council = TradingCouncil(
            blackboard=bb,
            alpha_specialist=alpha,
            macro_specialist=MacroSpecialist(),
            derivatives_specialist=DerivativesSpecialist(),
            risk_specialist=risk_spec,
            ops_specialist=ops_spec,
        )

        verdict = council.deliberate(
            flagged_cand,
            total_equity=1000.0,
            open_positions_count=0,
        )

        # Flagged symbol confidence dropped from 70 to 50, below 65 threshold or composite score
        # Blackboard state should record round and have updated system status
        self.assertTrue(len(bb.rounds) > 0)
        last_round = bb.rounds[-1]
        self.assertEqual(last_round["symbol"], "DOGE-USD")

        # Telegram badge should include Caution Mode tag!
        badge = verdict.telegram_badge
        self.assertIn("Caution Mode", badge)


if __name__ == "__main__":
    unittest.main()
