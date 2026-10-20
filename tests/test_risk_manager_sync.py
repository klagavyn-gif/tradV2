import os
import sys
import json
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, r"E:\5-8-69\tradV2")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from infrastructure.binance.risk_manager import BinanceFuturesRiskManager


def test_risk_manager_filters_paper_trades():
    print("--> Test 1: Testing RiskManager paper trade exclusion & real trade accounting...")
    with tempfile.TemporaryDirectory() as temp_dir:
        state_file = os.path.join(temp_dir, "risk_state.json")
        rm = BinanceFuturesRiskManager(
            state_file_path=state_file,
            dynamic_sizing_enabled=True,
            daily_max_loss_pct=4.0,
            loss_streak_max=3,
            only_executed=True,
        )

        executed_records = {
            "real_sol_001": {
                "alert_id": "real_sol_001",
                "symbol": "SOLUSDT",
                "signal": "BUY",
                "entry_price": 113.0,
                "notional": 5000.0,
                "executed_at": "2026-10-12T05:00:00",
            }
        }

        # 2 outcomes: 1 ghost paper trade (ONDO -3.0%), 1 real executed trade (SOL -1.5%)
        outcomes = [
            {
                "alert_id": "ghost_ondo_001",
                "symbol": "ONDO-USD",
                "strategy": "AW15",
                "signal": "BUY",
                "outcome_status": "settled",
                "outcome_result": "loss",
                "pnl_pct": -3.0,
                "timestamp": "2026-10-12 04:00:00",
            },
            {
                "alert_id": "real_sol_001",
                "symbol": "SOL-USD",
                "strategy": "PRIMARY",
                "signal": "BUY",
                "outcome_status": "settled",
                "outcome_result": "loss",
                "pnl_pct": -1.5,
                "timestamp": "2026-10-12 05:00:00",
            },
        ]

        state = rm.update_from_settled_outcomes(
            outcomes,
            executed_records=executed_records,
            notional_per_trade=400.0,
        )

        # Verification:
        # Ghost ONDO (-3.0%) must be completely ignored!
        # Only real SOL (-1.5%) is counted!
        print("Updated state:", state)
        assert state["daily_trades_count"] == 1, f"Expected 1 real trade, got {state['daily_trades_count']}"
        assert abs(state["daily_loss_pct"] - 1.5) < 1e-4, f"Expected 1.5% daily loss, got {state['daily_loss_pct']}"
        assert state["consecutive_losses"] == 1, f"Expected 1 consecutive loss, got {state['consecutive_losses']}"
        assert abs(state["daily_pnl_usdt"] - (-75.0)) < 1e-4, f"Expected -75 USDT, got {state['daily_pnl_usdt']}"
        assert state["circuit_breaker_active"] is False, "Circuit breaker must NOT be active (1.5% < 4.0%)"
        print("    [PASS] Paper trade ignored, real trade accurately counted!")


def test_risk_manager_clears_false_circuit_breaker():
    print("\n--> Test 2: Testing auto-reconciliation of falsely triggered circuit breaker...")
    with tempfile.TemporaryDirectory() as temp_dir:
        state_file = os.path.join(temp_dir, "risk_state.json")
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Simulate state previously contaminated by paper trades (like 7.9% drawdown)
        polluted_state = {
            "version": 1,
            "current_date_utc": today_str,
            "consecutive_losses": 2,
            "circuit_breaker_active": True,
            "circuit_breaker_until": "2026-10-13 00:00:00 UTC",
            "circuit_breaker_reason": "daily_drawdown_limit_7.90%",
            "daily_loss_pct": 7.9,
            "daily_pnl_usdt": -395.0,
            "daily_trades_count": 6,
            "processed_outcome_ids": ["real_sol_001", "ghost_sui_001", "ghost_ondo_001"],
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(polluted_state, f)

        rm = BinanceFuturesRiskManager(
            state_file_path=state_file,
            daily_max_loss_pct=4.0,
            loss_streak_max=3,
            only_executed=True,
        )

        executed_records = {
            "real_sol_001": {
                "alert_id": "real_sol_001",
                "symbol": "SOLUSDT",
                "signal": "BUY",
                "entry_price": 113.0,
                "notional": 5000.0,
                "executed_at": f"{today_str}T05:00:00",
            }
        }

        outcomes = [
            {
                "alert_id": "real_sol_001",
                "symbol": "SOL-USD",
                "outcome_status": "settled",
                "outcome_result": "loss",
                "pnl_pct": -1.61,
                "settled_at": f"{today_str} 09:45:00",
            },
            {
                "alert_id": "ghost_sui_001",
                "symbol": "SUI-USD",
                "outcome_status": "settled",
                "outcome_result": "loss",
                "pnl_pct": -1.88,
                "settled_at": f"{today_str} 09:45:00",
            },
        ]

        state = rm.update_from_settled_outcomes(outcomes, executed_records=executed_records)
        print("Reconciled state:", state)

        # Verification:
        # Circuit breaker must be CLEARED!
        assert state["circuit_breaker_active"] is False, "Circuit breaker MUST be cleared!"
        assert state["circuit_breaker_reason"] is None
        assert state["daily_trades_count"] == 1
        assert abs(state["daily_loss_pct"] - 1.61) < 1e-4
        assert state["consecutive_losses"] == 1
        print("    [PASS] False circuit breaker cleared and state reconciled to real trades!")


def test_risk_manager_triggers_on_real_breach():
    print("\n--> Test 3: Testing real breach triggers circuit breaker...")
    with tempfile.TemporaryDirectory() as temp_dir:
        state_file = os.path.join(temp_dir, "risk_state.json")
        rm = BinanceFuturesRiskManager(
            state_file_path=state_file,
            daily_max_loss_pct=4.0,
            loss_streak_max=3,
            only_executed=True,
        )

        executed_records = {
            f"real_{i}": {
                "alert_id": f"real_{i}",
                "symbol": "BTCUSDT",
                "notional": 1000.0,
                "executed_at": "2026-10-12T01:00:00",
            } for i in range(1, 4)
        }

        # 3 real consecutive losses of -1.5% each -> total 4.5% loss >= 4.0%
        outcomes = [
            {
                "alert_id": f"real_{i}",
                "symbol": "BTC-USD",
                "outcome_status": "settled",
                "outcome_result": "loss",
                "pnl_pct": -1.5,
                "settled_at": "2026-10-12 02:00:00",
            } for i in range(1, 4)
        ]

        state = rm.update_from_settled_outcomes(outcomes, executed_records=executed_records)
        print("Breached state:", state)
        assert state["circuit_breaker_active"] is True
        assert state["consecutive_losses"] == 3
        assert state["daily_loss_pct"] == 4.5
        print("    [PASS] Circuit breaker correctly triggered on real breach!")


if __name__ == "__main__":
    test_risk_manager_filters_paper_trades()
    test_risk_manager_clears_false_circuit_breaker()
    test_risk_manager_triggers_on_real_breach()
    print("\nALL RISK MANAGER SYNC TESTS PASSED! 🚀")
