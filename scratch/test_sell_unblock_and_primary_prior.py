#!/usr/bin/env python3
"""
Unit test suite verifying:
1. PRIMARY strategy baseline prior integration during Cold-Start.
2. Unblocking of SELL / SHORT signals from the Sell Walkforward Whitelist gate.
3. Realistic high-confidence SELL execution flow on Binance Futures.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config
import trad


class TestSellUnblockAndPrimaryPrior(unittest.TestCase):

    def test_primary_baseline_prior_resolved(self):
        """PRIMARY should resolve cleanly from STRATEGY_BASELINE_EDGE_PRIORS during Cold-Start."""
        metrics = trad._strategy_realized_proxy_metrics("PRIMARY")
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics.get("metric_source"), "strategy_baseline_prior")
        self.assertEqual(metrics.get("strategy_code"), "PRIMARY")
        self.assertAlmostEqual(metrics.get("win_rate_pct"), 65.0)
        self.assertAlmostEqual(metrics.get("expectancy_rr"), 0.50)
        self.assertAlmostEqual(metrics.get("trades"), 30.0)

    def test_sell_whitelist_gate_disabled_in_futures(self):
        """Sell whitelist gate should pass when disabled (default in futures mode)."""
        plan = {
            "forecast_direction": "SELL",
            "trend_bias": "DOWN",
            "confidence": 91.0,
            "sell_signal_role": "short_continuation",
        }
        with patch.object(config, "TELEGRAM_ALERT_SELL_WHITELIST_ENABLE", False):
            ok, reason, metrics = trad._evaluate_sell_whitelist_gate(plan)
            self.assertTrue(ok)
            self.assertEqual(reason, "sell_whitelist_disabled")

    def test_sell_whitelist_gate_graceful_fallback_when_metrics_missing(self):
        """Even if enabled, missing live optimizer metrics should not fail unless explicitly required."""
        plan = {
            "forecast_direction": "SELL",
            "trend_bias": "DOWN",
            "confidence": 91.0,
            "sell_signal_role": "short_continuation",
        }
        with patch.object(config, "TELEGRAM_ALERT_SELL_WHITELIST_ENABLE", True), \
             patch.object(config, "TELEGRAM_ALERT_SELL_REQUIRE_WALKFORWARD", False):
            ok, reason, metrics = trad._evaluate_sell_whitelist_gate(plan)
            self.assertTrue(ok)
            self.assertEqual(reason, "sell_walkforward_not_available")

    def test_high_confidence_sell_candidate_passes_entry_quality_gate(self):
        """A 91% confidence SELL candidate should no longer be killed by sell_walkforward_not_available."""
        plan = {
            "entry_price": 83800.0,
            "stop_loss": 84900.0,
            "take_profit": 81000.0,
            "confidence": 91.0,
            "win_rate_pct": 60.0,
            "expectancy_rr": 0.45,
            "trades": 30.0,
            "forecast_direction": "SELL",
            "trend_bias": "DOWN",
            "bars_since_signal": 1.0,
            "sell_signal_role": "short_continuation",
        }
        ok, reason, metrics = trad._evaluate_entry_quality_gate(plan, "SELL")
        self.assertTrue(ok, f"Gate failed with reason: {reason}")
        self.assertEqual(reason, "pass")

    def test_pipeline_hook_executes_short_candidate(self):
        """Verify pipeline hook approves and dispatches a valid SHORT candidate."""
        from infrastructure.binance.pipeline_hook import execute_binance_auto_trade_pipeline

        mock_client = MagicMock()
        mock_client.get_positions.return_value = []
        mock_client.get_usdt_balance.return_value = {"total": 500.0, "available": 500.0}
        mock_client.get_symbol_filters.return_value = {
            "tickSize": 0.1, "stepSize": 0.001, "minQty": 0.001, "minNotional": 50.0
        }
        mock_client.get_symbol_price.return_value = 83800.0
        mock_client.create_order.return_value = {
            "success": True,
            "data": {"avgPrice": "83800.0", "executedQty": "0.004", "orderId": 9999}
        }
        mock_client.create_algo_order.return_value = {"success": True, "data": {"algoId": 8888}}

        candidate = {
            "symbol": "BTC-USD",
            "signal": "SELL",
            "strategy": "CDCVIX15",
            "dispatch_status_label": "เข้าได้",
            "entry_price": 83800.0,
            "stop_loss": 84900.0,
            "confidence": 91.0,
            "alert_id": "test_btc_sell_999",
            "message": "<b>🎯 สรุป:</b> 🟢 เข้าได้\n<b>📌 แผนเข้า:</b> Short @ 83800.0",
        }

        sent_messages = []
        def mock_telegram(msg):
            sent_messages.append(msg)
            return MagicMock(message_id=12345)

        with patch("infrastructure.binance.client.BinanceFuturesClient", return_value=mock_client), \
             patch("infrastructure.binance.pipeline_hook._load_executed_orders", return_value={}), \
             patch("infrastructure.binance.pipeline_hook._save_executed_orders"), \
             patch.object(config, "BINANCE_FUTURES_AUTO_TRADE_ENABLED", True), \
             patch.object(config, "BINANCE_FUTURES_API_KEY", "dummy_key"), \
             patch.object(config, "BINANCE_FUTURES_API_SECRET", "dummy_secret"), \
             patch.object(config, "BINANCE_WATCHDOG_ENABLED", False), \
             patch.object(config, "BINANCE_DERIVATIVES_FILTER_ENABLED", False), \
             patch.object(config, "BINANCE_FUTURES_BLOCKED_SYMBOLS", ""), \
             patch.object(config, "BINANCE_FUTURES_ALLOWED_SYMBOLS", "BTC-USD,BTCUSDT"), \
             patch.object(config, "BTC_SHOCKWAVE_BARRIER_ENABLED", False):
            res = execute_binance_auto_trade_pipeline(
                sent_candidates=[candidate],
                config=config,
                helpers={},
                get_now=lambda: "2026-10-15 12:00:00",
                send_telegram_alert=mock_telegram,
            )

        executed = res.get("executed_entries", [])
        self.assertEqual(len(executed), 1)
        self.assertTrue(executed[0].get("success"))
        self.assertEqual(executed[0].get("symbol"), "BTCUSDT")
        self.assertEqual(executed[0].get("signal"), "SELL")


if __name__ == "__main__":
    unittest.main()
