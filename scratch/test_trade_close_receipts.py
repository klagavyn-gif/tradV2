#!/usr/bin/env python3
"""
Unit tests for BinanceFuturesWatchdog Trade Close Receipts & Full Financial Accounting.
Tests:
1. Detailed close receipt calculation and formatting on Step 5 (Market Close)
2. Closed position reconciliation on Step 5B (Exchange SL/TP or manual close)
3. Telegram message formatting with Entry, Exit, PnL %, Dollar PnL, Margin, ROI, Fee, Balance
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.binance.watchdog import BinanceFuturesWatchdog, format_position_close_receipt


class TestTradeCloseReceipts(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.get_usdt_balance.return_value = {
            "total": 1050.25,
            "available": 1025.25,
            "crossUnPnl": 0.0,
        }
        self.mock_client.get_position_mode.return_value = {"dualSidePosition": False}
        self.mock_client.get_open_orders.return_value = []
        self.mock_client.get_open_algo_orders.return_value = []
        self.mock_client.get_symbol_price.return_value = 0.5500
        self.mock_client.close_position_market.return_value = {
            "success": True,
            "data": {
                "orderId": 888123,
                "avgPrice": "0.5520",
                "executedQty": "736.0",
                "cumQuote": "406.272",
            },
        }

        self.watchdog = BinanceFuturesWatchdog(
            client=self.mock_client,
            target_leverage=20,
            target_margin_type="ISOLATED",
            max_positions=2,
            min_liquidation_distance_pct=3.0,
            default_stop_loss_pct=1.8,
        )

    def test_step5_market_close_produces_detailed_accounting_receipt(self):
        """
        Verify that when Step 5 closes a position, it extracts avgPrice, computes
        PnL %, Dollar PnL, Margin, ROI %, Fee, and fresh balance.
        """
        self.mock_client.get_positions.return_value = [{
            "symbol": "XRPUSDT",
            "side": "LONG",
            "positionAmt": "736.0",
            "entryPrice": "0.5400",
            "markPrice": "0.5515",
            "liquidationPrice": "0.5100",
            "unRealizedProfit": "8.46",
            "leverage": "20",
            "marginType": "ISOLATED",
        }]

        settled_outcomes = [
            {
                "alert_id": "aid_xrp_123",
                "symbol": "XRP-USD",
                "signal": "BUY",
                "outcome_status": "settled",
                "exit_reason": "opposite_signal",
                "pnl_pct": 2.22,
            }
        ]

        executed_records = {
            "aid_xrp_123": {
                "alert_id": "aid_xrp_123",
                "symbol": "XRPUSDT",
                "signal": "BUY",
                "entry_price": 0.5400,
                "quantity": 736.0,
                "notional": 400.0,
                "stop_loss": 0.5310,
                "status": "open",
            }
        }

        report = self.watchdog.audit_and_heal(
            settled_outcomes=settled_outcomes,
            executed_records=executed_records,
        )

        # 1. Market close was called
        self.mock_client.close_position_market.assert_called_once_with("XRPUSDT")

        # 2. Detailed close record exists in report
        closed_details = report.get("closed_positions_details", [])
        self.assertEqual(len(closed_details), 1)
        detail = closed_details[0]

        self.assertEqual(detail["symbol"], "XRPUSDT")
        self.assertEqual(detail["alert_id"], "aid_xrp_123")
        self.assertEqual(detail["side"], "LONG")
        self.assertAlmostEqual(detail["entry_price"], 0.5400)
        self.assertAlmostEqual(detail["exit_price"], 0.5520)
        # PnL % = ((0.5520 - 0.5400) / 0.5400) * 100 = 2.222%
        self.assertAlmostEqual(detail["pnl_pct"], 2.2222, places=2)
        # Dollar PnL = (2.2222% / 100) * (736 * 0.5520) = ~9.02 USDT
        self.assertGreater(detail["dollar_pnl"], 8.0)
        # ROI % = 2.2222% * 20 = ~44.4%
        self.assertGreater(detail["roi_pct"], 40.0)
        # Fresh balance recorded
        self.assertEqual(detail["balance_after"]["total"], 1050.25)

        # 3. Executed record status updated to closed
        self.assertEqual(executed_records["aid_xrp_123"]["status"], "closed")
        self.assertAlmostEqual(executed_records["aid_xrp_123"]["exit_price"], 0.5520)

        # 4. Telegram receipt format test
        msg = format_position_close_receipt(detail, testnet=True)
        self.assertIn("ใบเสร็จสรุปผลการปิดไม้ — ชนะ (กำไร)", msg)
        self.assertIn("<b>เหรียญ:</b> XRPUSDT", msg)
        self.assertIn("<b>ต้นทุนเข้า (Entry):</b> 0.54", msg)
        self.assertIn("<b>ราคาปิดจริง (Exit):</b> 0.552", msg)
        self.assertIn("<b>ผลตอบแทน (PnL):</b> +2.22%", msg)
        self.assertIn("กำไร/ขาดทุนสุทธิ:", msg)
        self.assertIn("ยอดเงินคงเหลือในพอร์ต:", msg)
        self.assertIn("1,050.25 USDT", msg)
        self.assertIn("ค่าธรรมเนียมประมาณการ:", msg)

    def test_step5b_reconciles_externally_closed_position(self):
        """
        Verify that if an order was executed by the bot but the position has closed
        on Binance (e.g. SL hit while bot was sleeping), Step 5B detects it,
        fetches the fill via get_user_trades, and formats the receipt.
        """
        # No positions open on Binance currently
        self.mock_client.get_positions.return_value = []

        # Mock user trades from Binance showing the exit fill
        self.mock_client.get_user_trades.return_value = [
            {
                "symbol": "ETHUSDT",
                "id": 55511,
                "orderId": 99912,
                "side": "SELL",
                "price": "2480.00",
                "qty": "0.16",
                "realizedPnl": "-3.20",
                "commission": "0.198",
                "commissionAsset": "USDT",
            }
        ]

        executed_records = {
            "aid_eth_999": {
                "alert_id": "aid_eth_999",
                "symbol": "ETHUSDT",
                "signal": "BUY",
                "entry_price": 2500.00,
                "quantity": 0.16,
                "notional": 400.0,
                "stop_loss": 2480.00,
                "status": "open",
            }
        }

        report = self.watchdog.audit_and_heal(
            settled_outcomes=[],
            executed_records=executed_records,
        )

        closed_details = report.get("closed_positions_details", [])
        self.assertEqual(len(closed_details), 1)
        detail = closed_details[0]

        self.assertEqual(detail["symbol"], "ETHUSDT")
        self.assertEqual(detail["alert_id"], "aid_eth_999")
        self.assertTrue(detail.get("is_reconciled"))
        self.assertAlmostEqual(detail["exit_price"], 2480.00)
        self.assertAlmostEqual(detail["dollar_pnl"], -3.20)
        self.assertAlmostEqual(detail["fee_est"], 0.198)
        self.assertEqual(detail["exit_reason"], "stop_loss_hit")

        # Telegram receipt format for loss
        msg = format_position_close_receipt(detail, testnet=False)
        self.assertIn("ใบเสร็จสรุปผลการปิดไม้ — ขาดทุน", msg)
        self.assertIn("⚡ [พอร์ตจริง Binance]", msg)
        self.assertIn("<b>ต้นทุนเข้า (Entry):</b> 2,500", msg)
        self.assertIn("<b>ราคาปิดจริง (Exit):</b> 2,480", msg)
        self.assertIn("-3.20 USDT", msg)
        self.assertIn("กระทบหยุดขาดทุน (SL)", msg)


if __name__ == "__main__":
    unittest.main()
