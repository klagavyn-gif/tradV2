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


if __name__ == "__main__":
    unittest.main()
