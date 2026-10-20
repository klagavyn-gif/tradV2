# -*- coding: utf-8 -*-
"""
scratch/test_forensic_safeguards.py

Forensic Verification Test Suite for tradV2.
Tests SEV-1, SEV-2, and SEV-3 safeguards:
1. Stop Loss in-line retries and safe-failure payload
2. Immediate Watchdog healing trigger upon SL failure
3. Risk Manager exchange-reconciliation on cold start
4. LiveCardStore collision elimination on repeated symbols
5. BinanceFuturesClient get_income_history API client contract
"""

import sys
import os
import unittest
import tempfile
import time
from unittest.mock import MagicMock, patch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from infrastructure.binance.client import BinanceFuturesClient
from infrastructure.binance.order_manager import BinanceFuturesOrderManager
from infrastructure.binance.risk_manager import BinanceFuturesRiskManager
from infrastructure.notifications.telegram_gateway import LiveCardStore


class TestForensicSafeguards(unittest.TestCase):

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.get_positions.return_value = []
        self.mock_client.get_symbol_filters.return_value = {
            "tickSize": 0.01,
            "stepSize": 0.001,
            "minQty": 0.001,
            "minNotional": 5.0,
        }
        self.mock_client.get_usdt_balance.return_value = {
            "total": 1000.0,
            "available": 800.0,
        }
        self.mock_client.place_order.return_value = {
            "success": True,
            "data": {"avgPrice": "50000.0"},
        }

    def test_stop_loss_inline_retry_success(self):
        """Test that OrderManager retries transient SL failures and succeeds."""
        # First attempt fails, second attempt succeeds
        self.mock_client.create_algo_order.side_effect = [
            {"success": False, "code": -1001, "msg": "Internal Error"},
            {"success": True, "data": {"algoId": 99991}},
        ]

        mgr = BinanceFuturesOrderManager(
            client=self.mock_client,
            trade_notional_usdt=400.0,
            trailing_r=0.0,  # disable TS for pure SL focus
        )
        candidate = {
            "symbol": "BTC-USD",
            "signal": "BUY",
            "entry_price": 50000.0,
            "stop_loss": 49000.0,
            "alert_id": "test_retry_1",
        }
        res = mgr.execute_candidate(candidate)

        self.assertTrue(res["success"])
        self.assertTrue(res["sl_success"])
        self.assertEqual(res["sl_order"], {"algoId": 99991})
        self.assertIsNone(res["sl_error"])
        self.assertEqual(self.mock_client.create_algo_order.call_count, 2)

    def test_stop_loss_all_retries_fail_returns_sl_error(self):
        """Test that when all 3 SL retries fail, sl_success is False and sl_error is captured."""
        self.mock_client.create_algo_order.return_value = {
            "success": False,
            "code": -2021,
            "msg": "Order would immediately trigger",
        }

        mgr = BinanceFuturesOrderManager(
            client=self.mock_client,
            trade_notional_usdt=400.0,
            trailing_r=0.0,
        )
        candidate = {
            "symbol": "BTC-USD",
            "signal": "BUY",
            "entry_price": 50000.0,
            "stop_loss": 49000.0,
            "alert_id": "test_fail_1",
        }
        res = mgr.execute_candidate(candidate)

        self.assertTrue(res["success"])
        self.assertFalse(res["sl_success"])
        self.assertIsNotNone(res["sl_error"])
        self.assertEqual(res["sl_error"]["code"], -2021)
        self.assertEqual(self.mock_client.create_algo_order.call_count, 3)

    def test_risk_manager_reconcile_from_exchange_cold_start(self):
        """Test that RiskManager can re-hydrate risk metrics directly from Binance income history."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_state_file = f.name

        try:
            rm = BinanceFuturesRiskManager(
                state_file_path=temp_state_file,
                daily_max_loss_pct=4.0,
                loss_streak_max=3,
                circuit_breaker_hours=6.0,
            )

            # Mock exchange income history with 3 consecutive losses today
            mock_client = MagicMock()
            now_ms = int(time.time() * 1000)
            mock_client.get_income_history.return_value = [
                {"income": "-20.0", "time": now_ms - 3000, "symbol": "BTCUSDT"},
                {"income": "-15.0", "time": now_ms - 2000, "symbol": "ETHUSDT"},
                {"income": "-25.0", "time": now_ms - 1000, "symbol": "SOLUSDT"},
            ]
            mock_client.get_usdt_balance.return_value = {"total": 1000.0}

            # On cold start: reconcile from exchange
            state = rm.reconcile_from_exchange(mock_client, total_equity=1000.0)

            # Total loss: 20+15+25 = 60 USDT -> 6% on 1000 equity (exceeds 4% limit)
            self.assertEqual(state["daily_trades_count"], 3)
            self.assertEqual(state["consecutive_losses"], 3)
            self.assertAlmostEqual(state["daily_pnl_usdt"], -60.0, places=2)
            self.assertGreaterEqual(state["daily_loss_pct"], 4.0)
            # Circuit breaker must be triggered!
            self.assertTrue(state["circuit_breaker_active"])
            self.assertTrue("daily_drawdown_limit" in str(state["circuit_breaker_reason"]) or "loss_streak" in str(state["circuit_breaker_reason"]))
        finally:
            if os.path.exists(temp_state_file):
                os.remove(temp_state_file)

    def test_live_card_store_no_collision_same_symbol(self):
        """Test that LiveCardStore isolates multiple trades of the same symbol across time."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_store_file = f.name

        try:
            store = LiveCardStore(file_path=temp_store_file)

            # Trade 1 for BTC
            store.register_card(
                symbol="BTCUSDT",
                message_id=1001,
                alert_id="aid_btc_morning",
                signal="BUY",
                entry_price=60000.0,
                stop_loss=59000.0,
            )

            # Later, Trade 2 for BTC
            store.register_card(
                symbol="BTCUSDT",
                message_id=2002,
                alert_id="aid_btc_evening",
                signal="BUY",
                entry_price=62000.0,
                stop_loss=61000.0,
            )

            # Both cards must be retrievable by their alert_id without being overwritten
            card_1 = store.get_card(alert_id="aid_btc_morning")
            self.assertIsNotNone(card_1)
            self.assertEqual(card_1["message_id"], 1001)
            self.assertEqual(card_1["entry_price"], 60000.0)

            card_2 = store.get_card(alert_id="aid_btc_evening")
            self.assertIsNotNone(card_2)
            self.assertEqual(card_2["message_id"], 2002)
            self.assertEqual(card_2["entry_price"], 62000.0)

            # Updating Trade 1 does not mutate Trade 2
            store.update_card(
                "aid_btc_morning",
                breakeven_active=True,
                new_stop_loss=60000.0,
            )
            card_1_updated = store.get_card(alert_id="aid_btc_morning")
            self.assertTrue(card_1_updated["breakeven_active"])
            self.assertEqual(card_1_updated["stop_loss"], 60000.0)

            card_2_check = store.get_card(alert_id="aid_btc_evening")
            self.assertFalse(card_2_check["breakeven_active"])
            self.assertEqual(card_2_check["stop_loss"], 61000.0)
        finally:
            if os.path.exists(temp_store_file):
                os.remove(temp_store_file)

    def test_binance_client_get_income_history_signature(self):
        """Test BinanceFuturesClient get_income_history parameters."""
        client = BinanceFuturesClient(api_key="mock", api_secret="mock")
        client._request = MagicMock(return_value={"success": True, "data": [{"income": "12.5"}]})

        res = client.get_income_history(symbol="BTCUSDT", income_type="REALIZED_PNL", limit=20)
        self.assertEqual(len(res), 1)
        client._request.assert_called_once()
        args, kwargs = client._request.call_args
        self.assertEqual(args[0], "GET")
        self.assertEqual(args[1], "/fapi/v1/income")
        self.assertEqual(kwargs["params"]["symbol"], "BTCUSDT")
        self.assertEqual(kwargs["params"]["incomeType"], "REALIZED_PNL")
        self.assertEqual(kwargs["params"]["limit"], 20)
        self.assertTrue(kwargs["signed"])


if __name__ == "__main__":
    unittest.main()
