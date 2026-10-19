"""
Unit tests for Critical System Hardening:
1. BinanceFuturesClient Time Offset Sync & RecvWindow 10000
2. BinanceFuturesOrderManager Breakeven Buffer (+0.10R)
3. Pipeline Hook Sector Correlation Guard (L1_HIGH_BETA, MEME)
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infrastructure.binance.client import BinanceFuturesClient
from infrastructure.binance.order_manager import BinanceFuturesOrderManager
from infrastructure.binance.pipeline_hook import SECTOR_MAP


class TestHardenedSafeguards(unittest.TestCase):
    def test_binance_client_time_sync_and_recvwindow(self):
        """Test that BinanceFuturesClient synchronizes time offset and uses 10000ms recvWindow."""
        client = BinanceFuturesClient(
            api_key="mock_key",
            api_secret="mock_secret",
            testnet=True,
        )

        # Mock server time response (500ms ahead of local)
        with patch.object(client.session, "get") as mock_get:
            local_ms = 1700000000000
            server_ms = local_ms + 500
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"serverTime": server_ms}

            with patch("time.time", return_value=local_ms / 1000.0):
                offset = client.sync_time_offset()
                self.assertEqual(offset, 500)

            # Test request signing uses offset and recvWindow=10000
            with patch.object(client.session, "post") as mock_post:
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {"orderId": 12345}

                with patch("time.time", return_value=local_ms / 1000.0):
                    res = client._request("POST", "/fapi/v1/order", params={"symbol": "BTCUSDT"}, signed=True)
                    self.assertTrue(res["success"])
                    called_data = mock_post.call_args[1]["data"]
                    self.assertEqual(called_data["recvWindow"], 10000)
                    self.assertEqual(called_data["timestamp"], local_ms + 500)

    def test_breakeven_buffer_calculation(self):
        """Test that order manager places breakeven stop with buffer (+0.10R) above entry for LONG."""
        mock_client = MagicMock()
        mock_client.get_symbol_filters.return_value = {"tickSize": "0.01", "stepSize": "0.1"}
        mock_client.create_algo_order.return_value = {"success": True, "clientAlgoId": "be_1"}

        om = BinanceFuturesOrderManager(
            client=mock_client,
            breakeven_r=1.2,
            breakeven_buffer_r=0.10,  # 0.10R buffer
        )

        # Active LONG position: Entry=100.0, SL=95.0 (Risk = 5.0)
        # Mark Price = 107.0 (Gain = (107 - 100)/5 = +1.40R >= 1.20R -> Triggers BE!)
        mock_client.get_positions.return_value = [{
            "symbol": "SOLUSDT",
            "side": "LONG",
            "positionAmt": 1.0,
            "entryPrice": 100.0,
            "markPrice": 107.0,
        }]
        mock_client.get_open_algo_orders.return_value = [{
            "symbol": "SOLUSDT",
            "orderType": "STOP_MARKET",
            "triggerPrice": "95.0",
            "algoId": 101,
            "clientAlgoId": "sl_101",
        }]

        updates = om.sync_breakeven_stops()

        self.assertEqual(len(updates), 1)
        up = updates[0]
        self.assertEqual(up["symbol"], "SOLUSDT")

        # Target BE = 100.0 + (0.10 * 5.0) = 100.50!
        # Not exact 100.00! It has the 0.50 (+0.10R) cushion protecting against spread whipsaw!
        self.assertEqual(up["new_sl"], 100.50)

    def test_sector_correlation_map(self):
        """Test SECTOR_MAP covers core altcoins accurately."""
        self.assertEqual(SECTOR_MAP["SOLUSDT"], "L1_HIGH_BETA")
        self.assertEqual(SECTOR_MAP["SUIUSDT"], "L1_HIGH_BETA")
        self.assertEqual(SECTOR_MAP["ETHUSDT"], "L1_CORE")
        self.assertEqual(SECTOR_MAP["DOGEUSDT"], "MEME")
        self.assertEqual(SECTOR_MAP["ONDOUSDT"], "RWA")
        self.assertEqual(SECTOR_MAP["PAXGUSDT"], "GOLD_SAFE_HAVEN")

    def test_dynamic_atr_drift_allowance(self):
        """Test that order manager allows wider drift for high-volatility coins instead of rigid 0.80%."""
        mock_client = MagicMock()
        mock_client.get_positions.return_value = []
        mock_client.get_symbol_filters.return_value = {
            "tickSize": "0.01", "stepSize": "0.1", "minQty": "0.1", "minNotional": "5.0"
        }
        mock_client.create_order.return_value = {"success": True, "orderId": 999}
        mock_client.create_algo_order.return_value = {"success": True}

        om = BinanceFuturesOrderManager(client=mock_client, max_entry_drift_pct=0.80)

        # High-volatility coin (Risk = (100 - 97.0) / 100 = 3.0%, 0.35 * 3.0% = 1.05% allowed drift)
        candidate = {
            "symbol": "SUIUSDT",
            "signal": "BUY",
            "entry_price": 100.0,
            "stop_loss": 97.0,
            "alert_id": "test_drift_1",
        }

        # Case A: Price drifted +0.95% (exceeds 0.80%, but within 1.05% dynamic ATR drift) -> MUST PASS!
        mock_client.get_symbol_price.return_value = 100.95
        res_pass = om.execute_candidate(candidate)
        self.assertTrue(res_pass["success"])

        # Case B: Price runaway pump +2.0% -> MUST BE REJECTED!
        mock_client.get_symbol_price.return_value = 102.00
        res_reject = om.execute_candidate(candidate)
        self.assertFalse(res_reject["success"])
        self.assertEqual(res_reject["reason"], "price_drift_exceeded")

    def test_two_tier_derivatives_veto(self):
        """Test that moderate funding rate receives soft warning (-12%) while extreme funding gets hard vetoed."""
        from infrastructure.binance.derivatives_filter import BinanceDerivativesFilter
        df = BinanceDerivativesFilter()

        # Case A: Moderate Funding (+0.040%) -> Soft warning, approved = True
        with patch.object(df, "get_funding_rate", return_value=0.00040), \
             patch.object(df, "get_open_interest_stats", return_value={"available": True, "delta_pct": 0.5, "trend": "NEUTRAL"}):
            eval_soft = df.evaluate_candidate({"symbol": "SOL-USD", "signal": "BUY"})
            self.assertTrue(eval_soft["approved"])
            self.assertEqual(eval_soft["confidence_modifier"], -12.0)
            self.assertIn("elevated_long_funding_+0.040%", eval_soft["warning_flags"])

        # Case B: Extreme Bubble Funding (+0.070%) -> Hard Veto, approved = False
        with patch.object(df, "get_funding_rate", return_value=0.00070), \
             patch.object(df, "get_open_interest_stats", return_value={"available": True, "delta_pct": 0.5, "trend": "NEUTRAL"}):
            eval_hard = df.evaluate_candidate({"symbol": "SOL-USD", "signal": "BUY"})
            self.assertFalse(eval_hard["approved"])
            self.assertIn("crowded_long_funding_bubble", eval_hard["veto_reason"])


if __name__ == "__main__":
    unittest.main()
