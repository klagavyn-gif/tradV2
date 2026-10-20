#!/usr/bin/env python3
"""
Unit tests for BinanceFuturesWatchdog Settled Positions Cleanup Sync & Order Linking.
Guarantees that active positions on Binance are NEVER closed due to historical paper settled outcomes,
and only closed if their exact matching alert_id is settled in the outcome tracker.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.binance.watchdog import BinanceFuturesWatchdog


class TestWatchdogSettledSync(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.get_usdt_balance.return_value = {"total": 5000.0, "available": 4500.0, "crossUnPnl": 10.0}
        self.mock_client.get_position_mode.return_value = {"dualSidePosition": False}
        self.mock_client.get_open_orders.return_value = []
        self.mock_client.get_open_algo_orders.return_value = []
        self.mock_client.close_position_market.return_value = {"success": True, "data": {"orderId": 99999}}

        self.watchdog = BinanceFuturesWatchdog(
            client=self.mock_client,
            target_leverage=20,
            target_margin_type="ISOLATED",
            max_positions=2,
            min_liquidation_distance_pct=3.0,
            default_stop_loss_pct=1.8,
        )

    def test_active_open_position_with_past_settled_outcomes_not_closed(self):
        """
        THE XRP INCIDENT REGRESSION TEST:
        Position XRPUSDT is open on Binance with alert_id 'aid_xrp_active'.
        In realized_outcomes, 'aid_xrp_active' is status='open'.
        There are ALSO old settled outcomes for XRP-USD (e.g. from AW15 paper trade 'aid_xrp_old_aw15').
        Watchdog MUST NOT close XRPUSDT!
        """
        self.mock_client.get_positions.return_value = [{
            "symbol": "XRPUSDT",
            "side": "LONG",
            "positionAmt": "3630.5",
            "entryPrice": "1.3416",
            "markPrice": "1.3420",
            "liquidationPrice": "1.2500",
            "unRealizedProfit": "1.45",
            "leverage": "20",
            "marginType": "ISOLATED",
        }]
        # Mock active STOP_MARKET so naked position defense is satisfied
        self.mock_client.get_open_algo_orders.return_value = [{
            "symbol": "XRPUSDT",
            "orderType": "STOP_MARKET",
            "triggerPrice": "1.3215",
        }]

        executed_records = {
            "aid_xrp_active": {
                "alert_id": "aid_xrp_active",
                "symbol": "XRPUSDT",
                "signal": "BUY",
                "entry_price": 1.3416,
                "stop_loss": 1.3215,
                "executed_at": "2026-10-16T10:48:19",
            }
        }

        settled_outcomes = [
            # Old paper trade from 3 days ago for XRP-USD that settled
            {
                "alert_id": "aid_xrp_old_aw15",
                "symbol": "XRP-USD",
                "strategy": "AW15",
                "outcome_status": "settled",
                "exit_reason": "kinetic_stop_hit",
                "pnl_pct": 0.0,
            },
            # Current active trade that is STILL OPEN!
            {
                "alert_id": "aid_xrp_active",
                "symbol": "XRP-USD",
                "strategy": "PRIMARY",
                "outcome_status": "open",
                "exit_reason": "no_future_bars",
                "pnl_pct": None,
            },
        ]

        report = self.watchdog.audit_and_heal(
            settled_outcomes=settled_outcomes,
            executed_records=executed_records,
        )

        # close_position_market must NOT be called!
        self.mock_client.close_position_market.assert_not_called()
        self.assertEqual(report["open_positions_count"], 1)
        self.assertEqual(report["open_positions"][0]["symbol"], "XRPUSDT")
        # No auto-close action reported
        auto_closes = [h for h in report["healed_actions"] if "[Auto-Close]" in h]
        self.assertEqual(len(auto_closes), 0)

    def test_stuck_position_with_exact_settled_alert_is_closed(self):
        """
        True stuck position:
        Position ETHUSDT is open on Binance for 'aid_eth_stuck'.
        In realized_outcomes, 'aid_eth_stuck' has status='settled' (e.g. TP was hit in model).
        Watchdog MUST detect this stuck position and close it on Binance.
        """
        self.mock_client.get_positions.return_value = [{
            "symbol": "ETHUSDT",
            "side": "LONG",
            "positionAmt": "0.165",
            "entryPrice": "2450.0",
            "markPrice": "2490.0",
            "liquidationPrice": "2300.0",
            "unRealizedProfit": "6.6",
            "leverage": "20",
            "marginType": "ISOLATED",
        }]
        self.mock_client.get_open_algo_orders.return_value = [{
            "symbol": "ETHUSDT",
            "orderType": "STOP_MARKET",
            "triggerPrice": "2420.0",
        }]

        executed_records = {
            "aid_eth_stuck": {
                "alert_id": "aid_eth_stuck",
                "symbol": "ETHUSDT",
                "signal": "BUY",
                "entry_price": 2450.0,
                "executed_at": "2026-10-16T08:00:00",
            }
        }

        settled_outcomes = [
            {
                "alert_id": "aid_eth_stuck",
                "symbol": "ETH-USD",
                "strategy": "CDCVIX15",
                "outcome_status": "settled",
                "exit_reason": "take_profit_hit",
                "pnl_pct": 2.5,
            }
        ]

        report = self.watchdog.audit_and_heal(
            settled_outcomes=settled_outcomes,
            executed_records=executed_records,
        )

        self.mock_client.close_position_market.assert_called_once_with("ETHUSDT")
        self.assertEqual(report["open_positions_count"], 0)
        auto_closes = [h for h in report["healed_actions"] if "[Auto-Close]" in h]
        self.assertEqual(len(auto_closes), 1)
        self.assertIn("take_profit_hit", auto_closes[0])

    def test_manual_position_without_executed_record_never_closed(self):
        """
        User manual trade:
        Position BTCUSDT is open on Binance, but it was NOT opened by the bot (not in executed_records).
        Even if there are historical settled BTC outcomes, Watchdog must NEVER close manual positions!
        """
        self.mock_client.get_positions.return_value = [{
            "symbol": "BTCUSDT",
            "side": "LONG",
            "positionAmt": "0.01",
            "entryPrice": "65000.0",
            "markPrice": "65100.0",
            "liquidationPrice": "61000.0",
            "unRealizedProfit": "1.0",
            "leverage": "20",
            "marginType": "ISOLATED",
        }]
        self.mock_client.get_open_algo_orders.return_value = [{
            "symbol": "BTCUSDT",
            "orderType": "STOP_MARKET",
            "triggerPrice": "64000.0",
        }]

        executed_records = {
            # Empty or contains other coins
            "aid_other": {"alert_id": "aid_other", "symbol": "SOLUSDT"}
        }

        settled_outcomes = [
            {
                "alert_id": "aid_btc_old",
                "symbol": "BTC-USD",
                "outcome_status": "settled",
                "exit_reason": "time_exit",
                "pnl_pct": -0.5,
            }
        ]

        report = self.watchdog.audit_and_heal(
            settled_outcomes=settled_outcomes,
            executed_records=executed_records,
        )

        self.mock_client.close_position_market.assert_not_called()
        self.assertEqual(report["open_positions_count"], 1)

    def test_find_active_executed_record_picks_latest_and_respects_side(self):
        """Verify _find_active_executed_record selects the latest order matching symbol and side."""
        records = {
            "aid_1": {"alert_id": "aid_1", "symbol": "XRP-USD", "signal": "BUY", "executed_at": "2026-10-10 10:00:00"},
            "aid_2": {"alert_id": "aid_2", "symbol": "XRPUSDT", "signal": "SELL", "executed_at": "2026-10-15 12:00:00"},
            "aid_3": {"alert_id": "aid_3", "symbol": "XRPUSDT", "signal": "BUY", "executed_at": "2026-10-16 17:48:19"},
        }

        # For LONG position, it must pick aid_3 (most recent BUY)
        rec_long = BinanceFuturesWatchdog._find_active_executed_record(records, "XRPUSDT", "LONG")
        self.assertIsNotNone(rec_long)
        self.assertEqual(rec_long["alert_id"], "aid_3")

        # For SHORT position, it must pick aid_2 (most recent SELL)
        rec_short = BinanceFuturesWatchdog._find_active_executed_record(records, "XRPUSDT", "SHORT")
        self.assertIsNotNone(rec_short)
        self.assertEqual(rec_short["alert_id"], "aid_2")


if __name__ == "__main__":
    unittest.main()
