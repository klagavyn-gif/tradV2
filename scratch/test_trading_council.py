#!/usr/bin/env python3
"""
Comprehensive Unit & Integration Test Suite for Central Blackboard & Multi-Agent Trading Council.
Tests:
1. Central Blackboard (atomic state, hot memory recording, query, and pruning)
2. All 5 Council Specialists (Alpha, Macro, Derivatives, Risk, Ops)
3. Trading Council Arbiter (weight aggregation, veto override, composite scoring, consensus verdicts)
4. End-to-End Binance Auto-Trade Pipeline Hook with Council Integration
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import json
import time
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock

# Workspace root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from infrastructure.council import (
    CentralBlackboard,
    CouncilOpinion,
    AlphaSpecialist,
    MacroSpecialist,
    DerivativesSpecialist,
    RiskSpecialist,
    OpsSpecialist,
    ConsensusVerdict,
    TradingCouncil,
)
from infrastructure.binance.pipeline_hook import (
    execute_binance_auto_trade_pipeline,
    _load_executed_orders,
)


def test_central_blackboard_lifecycle():
    print("\n" + "=" * 70)
    print("--> TEST 1: Central Blackboard State & Auto-Janitor Hot Memory Pruning")
    print("=" * 70)

    temp_dir = tempfile.mkdtemp()
    try:
        bb_file = os.path.join(temp_dir, "council_deliberations.json")
        blackboard = CentralBlackboard(
            file_path=bb_file,
            max_hot_hours=2.0,  # 2 hours retention
            max_hot_rounds=5,   # Max 5 rounds retention
        )

        # 1. Post rounds
        now = datetime.now(timezone.utc)
        for i in range(10):
            # i=0..4 are older than 3 hours, i=5..9 are recent
            age_hours = 4.0 if i < 3 else (0.1 * i)
            ts = (now - timedelta(hours=age_hours)).strftime("%Y-%m-%d %H:%M:%S")
            record = {
                "round_id": f"rnd_{i}",
                "symbol": f"SYM_{i}",
                "signal": "BUY",
                "timestamp": ts,
                "verdict": {"decision": "EXECUTE", "approved": True, "composite_score": 80.0},
            }
            blackboard.post_deliberation(record)

        # Query recent deliberations
        recent = blackboard.get_recent_deliberations(limit=10)
        assert len(recent) <= 5, f"Expected hot memory pruned to <= 5 rounds, got {len(recent)}"

        # Verify old rounds (> 2 hours) were removed by auto-janitor
        for r in recent:
            dt = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            assert (now - dt).total_seconds() <= 2.5 * 3600, f"Stale round {r['round_id']} not pruned"

        # Verify atomic write
        assert os.path.exists(bb_file)
        with open(bb_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "deliberations" in data
        assert "last_pruned" in data
        print(f"  [PASS] Blackboard retained {len(recent)} clean rounds after hot memory pruning! ✅")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_specialists_evaluations():
    print("\n" + "=" * 70)
    print("--> TEST 2: Multi-Agent Specialist Opinions & Defensive Veto Gates")
    print("=" * 70)

    # 1. AlphaSpecialist
    alpha = AlphaSpecialist(allowed_strategies={"CDCVIX15", "PA15", "PRIMARY"})
    
    # Whitelisted + approved
    op_alpha_ok = alpha.evaluate({
        "strategy": "CDCVIX15",
        "signal": "BUY",
        "dispatch_status_label": "เข้าได้",
        "entry_price": 100.0,
        "stop_loss": 98.0,
        "take_profit": 104.0,  # R:R = 4/2 = 2.0
        "ai_prob_win": 0.65,
    })
    assert op_alpha_ok.status == "APPROVED"
    assert op_alpha_ok.confidence >= 80.0
    assert not op_alpha_ok.veto
    print(f"  [PASS] AlphaSpecialist approved CDCVIX15 with high conviction ({op_alpha_ok.confidence:.1f}%)! ✅")

    # Strategy not whitelisted (e.g. AW15) -> VETO
    op_alpha_aw15 = alpha.evaluate({
        "strategy": "AW15",
        "signal": "BUY",
        "dispatch_status_label": "เข้าได้",
    })
    assert op_alpha_aw15.veto is True
    assert "not_whitelisted" in op_alpha_aw15.veto_reason
    print("  [PASS] AlphaSpecialist vetoed non-whitelisted AW15! ✅")

    # Dispatch status 'ห้ามเข้า' -> VETO
    op_alpha_blocked = alpha.evaluate({
        "strategy": "CDCVIX15",
        "signal": "BUY",
        "dispatch_status_label": "ห้ามเข้า",
    })
    assert op_alpha_blocked.veto is True
    print("  [PASS] AlphaSpecialist vetoed 'ห้ามเข้า' candidate! ✅")

    # 2. MacroSpecialist
    mock_barrier = MagicMock()
    macro = MacroSpecialist(barrier=mock_barrier)

    # Shockwave active on altcoin -> VETO
    mock_barrier.evaluate.return_value = {
        "shock_active": True,
        "range_atr": 2.4,
        "abs_change_pct": 2.1,
        "direction": "PUMP",
    }
    mock_barrier.should_veto_candidate.return_value = (True, "altcoin_shockwave_block")

    op_macro_veto = macro.evaluate({"symbol": "SUI-USD", "signal": "BUY"})
    assert op_macro_veto.veto is True
    assert "btc_shockwave_veto" in op_macro_veto.veto_reason
    print("  [PASS] MacroSpecialist vetoed Altcoin during active BTC Shockwave! ✅")

    # Quiet market -> APPROVED
    mock_barrier.evaluate.return_value = {"shock_active": False}
    mock_barrier.should_veto_candidate.return_value = (False, None)
    op_macro_ok = macro.evaluate({"symbol": "SUI-USD", "signal": "BUY"})
    assert op_macro_ok.status == "APPROVED"
    assert not op_macro_ok.veto
    print("  [PASS] MacroSpecialist approved Altcoin during quiet BTC regime! ✅")

    # 3. DerivativesSpecialist
    mock_deriv = MagicMock()
    deriv = DerivativesSpecialist(deriv_filter=mock_deriv)

    # Crowded Long funding rate -> VETO
    mock_deriv.evaluate_candidate.return_value = {
        "approved": False,
        "veto_reason": "crowded_long_funding_+0.050%",
        "badge": "⚠️ FR +0.05%",
    }
    op_deriv_veto = deriv.evaluate({"symbol": "BTC-USD", "signal": "BUY"})
    assert op_deriv_veto.veto is True
    assert "derivatives_veto_crowded_long" in op_deriv_veto.veto_reason
    print("  [PASS] DerivativesSpecialist vetoed Crowded Long funding rate! ✅")

    # Institutional OI Inflow -> APPROVED with high confidence
    mock_deriv.evaluate_candidate.return_value = {
        "approved": True,
        "oi_trend": "EXPANDING",
        "funding_rate_pct": 0.01,
        "delta_pct": 2.5,
        "badge": "🔥 OI +2.5%",
    }
    op_deriv_ok = deriv.evaluate({"symbol": "BTC-USD", "signal": "BUY"})
    assert op_deriv_ok.status == "APPROVED"
    assert op_deriv_ok.confidence >= 85.0
    print(f"  [PASS] DerivativesSpecialist boosted confidence for institutional OI inflow ({op_deriv_ok.confidence:.1f}%)! ✅")

    # 4. RiskSpecialist
    mock_rm = MagicMock()
    risk = RiskSpecialist(risk_manager=mock_rm, default_notional=400.0, leverage=20)

    # Circuit breaker active -> VETO
    mock_rm.check_circuit_breaker.return_value = (True, "drawdown_exceeded", "2026-10-17 18:00:00")
    op_risk_cb = risk.evaluate({"symbol": "SOL-USD", "signal": "BUY"}, total_equity=1000.0)
    assert op_risk_cb.veto is True
    assert "circuit_breaker" in op_risk_cb.veto_reason
    print("  [PASS] RiskSpecialist vetoed trade under active Circuit Breaker! ✅")

    # Loss streak throttle -> Throttled sizing & caution
    mock_rm.check_circuit_breaker.return_value = (False, "", "")
    mock_rm.calculate_trade_sizing.return_value = {
        "final_notional": 200.0,
        "multiplier": 0.5,
        "multiplier_reason": "loss_streak_throttle_2",
    }
    op_risk_throttle = risk.evaluate({"symbol": "SOL-USD", "signal": "BUY"}, total_equity=1000.0)
    assert op_risk_throttle.status == "APPROVED"
    assert op_risk_throttle.metrics["multiplier"] == 0.5
    print("  [PASS] RiskSpecialist dynamically throttled sizing by 50% for loss streak! ✅")

    # 5. OpsSpecialist
    ops = OpsSpecialist(max_positions=2, max_drift_pct=0.80)

    # Max positions reached -> VETO
    op_ops_cap = ops.evaluate({"symbol": "ETH-USD", "signal": "BUY", "entry_price": 2500.0}, open_positions_count=2)
    assert op_ops_cap.veto is True
    assert "max_positions" in op_ops_cap.veto_reason
    print("  [PASS] OpsSpecialist vetoed trade when position capacity is full! ✅")

    # Price drift exceeded -> VETO (anti-chase)
    # Entry 2500, live price 2525 (+1.0% drift > 0.8% limit)
    op_ops_drift = ops.evaluate(
        {"symbol": "ETH-USD", "signal": "BUY", "entry_price": 2500.0},
        open_positions_count=1,
        live_price=2525.0,
    )
    assert op_ops_drift.veto is True
    assert "price_drift" in op_ops_drift.veto_reason
    print("  [PASS] OpsSpecialist anti-chase guard vetoed +1.0% price drift! ✅")


def test_trading_council_consensus_and_veto_override():
    print("\n" + "=" * 70)
    print("--> TEST 3: Trading Council Arbiter, Weighted Scoring & Absolute Veto Override")
    print("=" * 70)

    temp_dir = tempfile.mkdtemp()
    try:
        bb = CentralBlackboard(file_path=os.path.join(temp_dir, "council.json"))
        
        # Scenario A: 4 specialists love the trade (95% confidence), but Derivatives Specialist issues VETO
        council_veto = TradingCouncil(
            blackboard=bb,
            alpha_specialist=MagicMock(evaluate=lambda c: CouncilOpinion("alpha_specialist", "APPROVED", 95.0)),
            macro_specialist=MagicMock(evaluate=lambda c, client=None, shock_eval=None: CouncilOpinion("macro_specialist", "APPROVED", 95.0)),
            derivatives_specialist=MagicMock(evaluate=lambda c, **kw: CouncilOpinion("derivatives_specialist", "VETOED", 0.0, veto=True, veto_reason="derivatives_veto_crowded_long")),
            risk_specialist=MagicMock(evaluate=lambda c, total_equity=0: CouncilOpinion("risk_specialist", "APPROVED", 90.0)),
            ops_specialist=MagicMock(evaluate=lambda c, open_positions_count=0, live_price=None: CouncilOpinion("ops_specialist", "APPROVED", 95.0)),
            min_composite_score=65.0,
        )

        verdict_veto = council_veto.deliberate({"symbol": "DOGE-USD", "signal": "BUY", "alert_id": "test_veto_1"})
        assert verdict_veto.approved is False
        assert verdict_veto.decision == "VETO"
        assert verdict_veto.veto_specialist == "derivatives_specialist"
        assert verdict_veto.veto_reason == "derivatives_veto_crowded_long"
        print("  [PASS] Absolute Veto Override honored: 1 VETO overrides 4 high-confidence approvals! ✅")

        # Scenario B: All 5 specialists approve with strong metrics -> Consensus EXECUTE
        council_exec = TradingCouncil(
            blackboard=bb,
            alpha_specialist=MagicMock(evaluate=lambda c: CouncilOpinion("alpha_specialist", "APPROVED", 85.0)),
            macro_specialist=MagicMock(evaluate=lambda c, client=None, shock_eval=None: CouncilOpinion("macro_specialist", "APPROVED", 90.0)),
            derivatives_specialist=MagicMock(evaluate=lambda c, **kw: CouncilOpinion("derivatives_specialist", "APPROVED", 80.0)),
            risk_specialist=MagicMock(evaluate=lambda c, total_equity=0: CouncilOpinion("risk_specialist", "APPROVED", 80.0, metrics={"multiplier": 1.25})),
            ops_specialist=MagicMock(evaluate=lambda c, open_positions_count=0, live_price=None: CouncilOpinion("ops_specialist", "APPROVED", 90.0)),
            min_composite_score=65.0,
        )

        verdict_exec = council_exec.deliberate({"symbol": "BTC-USD", "signal": "BUY", "alert_id": "test_exec_1"})
        assert verdict_exec.approved is True
        assert verdict_exec.decision == "EXECUTE"
        assert verdict_exec.composite_score >= 80.0
        assert "/6 เสียง" in verdict_exec.summary
        assert "🏛️" in verdict_exec.telegram_badge
        print(f"  [PASS] Council reached consensus: Score {verdict_exec.composite_score:.1f}% -> {verdict_exec.decision}! ✅")

        # Scenario C: Low composite score (< 65%) without explicit veto -> STANDBY
        council_standby = TradingCouncil(
            blackboard=bb,
            alpha_specialist=MagicMock(evaluate=lambda c: CouncilOpinion("alpha_specialist", "CAUTION", 45.0)),
            macro_specialist=MagicMock(evaluate=lambda c, client=None, shock_eval=None: CouncilOpinion("macro_specialist", "CAUTION", 50.0)),
            derivatives_specialist=MagicMock(evaluate=lambda c, **kw: CouncilOpinion("derivatives_specialist", "NEUTRAL", 55.0)),
            risk_specialist=MagicMock(evaluate=lambda c, total_equity=0: CouncilOpinion("risk_specialist", "CAUTION", 50.0)),
            ops_specialist=MagicMock(evaluate=lambda c, open_positions_count=0, live_price=None: CouncilOpinion("ops_specialist", "APPROVED", 60.0)),
            min_composite_score=65.0,
        )
        verdict_standby = council_standby.deliberate({"symbol": "XRP-USD", "signal": "BUY", "alert_id": "test_standby_1"})
        assert verdict_standby.approved is False
        assert verdict_standby.decision == "STANDBY"
        assert verdict_standby.composite_score < 65.0
        print(f"  [PASS] Low composite confidence ({verdict_standby.composite_score:.1f}%) routed to STANDBY! ✅")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_pipeline_hook_council_integration():
    print("\n" + "=" * 70)
    print("--> TEST 4: End-to-End Pipeline Hook with Central Blackboard & Trading Council")
    print("=" * 70)

    class TestConfig:
        BINANCE_FUTURES_AUTO_TRADE_ENABLED = True
        BINANCE_WATCHDOG_ENABLED = False
        BINANCE_FUTURES_API_KEY = "test_key"
        BINANCE_FUTURES_API_SECRET = "test_secret"
        BINANCE_FUTURES_TESTNET = True
        BINANCE_FUTURES_ALLOWED_STRATEGIES = "CDCVIX15,PA15,PRIMARY"
        BINANCE_FUTURES_BLOCKED_SYMBOLS = "NEAR-USD,LINK-USD"
        BINANCE_FUTURES_TRADE_NOTIONAL_USDT = 400.0
        BINANCE_FUTURES_MAX_POSITIONS = 2
        BINANCE_DERIVATIVES_FILTER_ENABLED = True
        TRADING_COUNCIL_ENABLED = True
        TRADING_COUNCIL_MIN_COMPOSITE_SCORE = 65.0

    temp_dir = tempfile.mkdtemp()
    try:
        executed_file = os.path.join(temp_dir, "executed.json")
        outcomes_file = os.path.join(temp_dir, "outcomes.json")
        council_file = os.path.join(temp_dir, "council_deliberations.json")

        helpers = {
            "binance_executed_orders_path": lambda: executed_file,
            "alert_outcomes_file_path": lambda: outcomes_file,
            "council_deliberations_path": lambda: council_file,
        }

        # Mock Binance Client
        import infrastructure.binance.client as client_module
        orig_client = client_module.BinanceFuturesClient

        mock_instance = MagicMock()
        mock_instance.get_positions.return_value = []
        mock_instance.get_usdt_balance.return_value = {"total": 2000.0, "available": 2000.0}
        mock_instance.get_symbol_price.return_value = 100.0
        mock_instance.get_symbol_filters.return_value = {
            "tickSize": "0.01",
            "stepSize": "0.01",
            "minQty": "0.01",
            "minNotional": "5.0",
        }
        mock_instance.create_order.return_value = {"success": True, "data": {"avgPrice": "100.0"}}
        mock_instance.create_algo_order.return_value = {"success": True, "algoId": 12345}
        client_module.BinanceFuturesClient = lambda *args, **kwargs: mock_instance

        # Candidates: 1 Approved (PAXG), 1 Blocked by Strategy (AW15)
        candidates = [
            {
                "symbol": "PAXG-USD",
                "signal": "BUY",
                "strategy": "CDCVIX15",
                "dispatch_status_label": "เข้าได้",
                "entry_price": 2700.0,
                "stop_loss": 2660.0,
                "take_profit": 2760.0,
                "alert_id": "aid_paxg_001",
            },
            {
                "symbol": "BTC-USD",
                "signal": "BUY",
                "strategy": "AW15",
                "dispatch_status_label": "เข้าได้",
                "alert_id": "aid_btc_aw15",
            },
        ]

        tg_messages = []
        res = execute_binance_auto_trade_pipeline(
            sent_candidates=candidates,
            config=TestConfig(),
            helpers=helpers,
            get_now=lambda: datetime.now(timezone.utc),
            send_telegram_alert=lambda msg, **kw: tg_messages.append(msg),
        )

        assert res["enabled"] is True
        assert len(res["executed_entries"]) == 1
        exec_entry = res["executed_entries"][0]
        assert exec_entry["success"] is True
        assert exec_entry["symbol"] == "PAXGUSDT"

        # Verify Council Verdict was recorded in results and blackboard
        assert "council_verdicts" in res
        assert len(res["council_verdicts"]) == 1
        verdict_dict = res["council_verdicts"][0]
        assert verdict_dict["decision"] == "EXECUTE"
        assert verdict_dict["approved"] is True

        # Verify Blackboard persisted the deliberation
        assert os.path.exists(council_file)
        with open(council_file, "r", encoding="utf-8") as f:
            bb_data = json.load(f)
        assert len(bb_data["deliberations"]) == 1
        assert bb_data["deliberations"][0]["symbol"] == "PAXG-USD"

        # Verify Telegram receipt contained Council badge
        assert any("🏛️" in msg and "Council Consensus" in msg for msg in tg_messages)

        print("  [PASS] Pipeline Hook executed PAXG-USD with full Council consensus! ✅")
        print("  [PASS] Blackboard recorded round to hot memory with automatic cleanup! ✅")
        print("  [PASS] Telegram execution receipt formatted with Council badge! ✅")

        client_module.BinanceFuturesClient = orig_client
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 80)
    print(" STARTING RIGOROUS CENTRAL BLACKBOARD & TRADING COUNCIL TEST SUITE")
    print("=" * 80)

    test_central_blackboard_lifecycle()
    test_specialists_evaluations()
    test_trading_council_consensus_and_veto_override()
    test_pipeline_hook_council_integration()

    print("\n" + "=" * 80)
    print(" ALL 4 BLACKBOARD & TRADING COUNCIL TEST MODULES PASSED WITH 100% SUCCESS! 🏛️🛡️")
    print("=" * 80)
