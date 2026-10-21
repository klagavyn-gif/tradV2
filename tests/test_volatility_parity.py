"""Unit tests for Portfolio Volatility Parity (Bounded Inverse Volatility Sizing)."""

import unittest
from unittest.mock import MagicMock

from infrastructure.binance.risk_manager import BinanceFuturesRiskManager


class TestVolatilityParity(unittest.TestCase):
    """Test suite for Bounded Volatility Parity in RiskManager."""

    def setUp(self):
        self.rm_disabled = BinanceFuturesRiskManager(
            state_file_path=None,
            dynamic_sizing_enabled=True,
            equity_risk_pct=5.0,
            volatility_parity_enabled=False,
        )
        self.rm_disabled.load_state = MagicMock(return_value={"consecutive_losses": 0, "consecutive_wins": 0})

        self.rm_enabled = BinanceFuturesRiskManager(
            state_file_path=None,
            dynamic_sizing_enabled=True,
            equity_risk_pct=5.0,
            volatility_parity_enabled=True,
            volatility_base_risk_pct=1.80,
            volatility_min_mult=0.60,
            volatility_max_mult=1.35,
        )
        self.rm_enabled.load_state = MagicMock(return_value={"consecutive_losses": 0, "consecutive_wins": 0})

    def test_disabled_volatility_parity_keeps_standard_sizing(self):
        """When disabled, volatility parity does not alter position size."""
        candidate = {
            "symbol": "SUIUSDT",
            "entry_price": 2.00,
            "stop_loss": 1.90,  # 5.0% SL distance
        }
        sizing = self.rm_disabled.calculate_trade_sizing(
            candidate=candidate,
            total_equity=1000.0,
            leverage=20,
            fallback_notional=400.0,
        )
        # Margin 50 USDT * 20x = 1000 USDT notional
        self.assertEqual(sizing["final_notional"], 1000.0)
        self.assertEqual(sizing["volatility_multiplier"], 1.0)
        self.assertFalse(sizing["volatility_parity_active"])

    def test_high_volatility_asset_throttled_with_clamp(self):
        """High volatility asset (e.g. SUI with 3.6% SL) is downsized to clamp (0.60x)."""
        candidate = {
            "symbol": "SUIUSDT",
            "entry_price": 2.00,
            "stop_loss": 1.928,  # 3.60% SL distance
        }
        sizing = self.rm_enabled.calculate_trade_sizing(
            candidate=candidate,
            total_equity=1000.0,
            leverage=20,
            fallback_notional=400.0,
        )
        # 1.8 / 3.6 = 0.50 -> clamped to 0.60
        self.assertTrue(sizing["volatility_parity_active"])
        self.assertEqual(sizing["volatility_multiplier"], 0.60)
        self.assertAlmostEqual(sizing["stop_dist_pct"], 3.60, places=2)
        # Base 1000.0 * 0.60 = 600.0 USDT
        self.assertEqual(sizing["final_notional"], 600.0)
        self.assertEqual(sizing["final_margin"], 30.0)

    def test_low_volatility_asset_scaled_with_clamp(self):
        """Low volatility asset (e.g. ETH with 1.2% SL) is upsized to clamp (1.35x)."""
        candidate = {
            "symbol": "ETHUSDT",
            "entry_price": 2500.0,
            "stop_loss": 2470.0,  # 1.20% SL distance
        }
        sizing = self.rm_enabled.calculate_trade_sizing(
            candidate=candidate,
            total_equity=1000.0,
            leverage=20,
            fallback_notional=400.0,
        )
        # 1.8 / 1.2 = 1.50 -> clamped to 1.35
        self.assertTrue(sizing["volatility_parity_active"])
        self.assertEqual(sizing["volatility_multiplier"], 1.35)
        self.assertAlmostEqual(sizing["stop_dist_pct"], 1.20, places=2)
        # Base 1000.0 * 1.35 = 1350.0 USDT
        self.assertEqual(sizing["final_notional"], 1350.0)
        self.assertEqual(sizing["final_margin"], 67.5)

    def test_baseline_volatility_asset_neutral_sizing(self):
        """Asset with baseline risk (1.8% SL) receives exactly 1.0x sizing."""
        candidate = {
            "symbol": "BNBUSDT",
            "entry_price": 600.0,
            "stop_loss": 589.2,  # exactly 1.80% SL distance
        }
        sizing = self.rm_enabled.calculate_trade_sizing(
            candidate=candidate,
            total_equity=1000.0,
            leverage=20,
            fallback_notional=400.0,
        )
        self.assertTrue(sizing["volatility_parity_active"])
        self.assertEqual(sizing["volatility_multiplier"], 1.00)
        self.assertEqual(sizing["final_notional"], 1000.0)

    def test_missing_or_invalid_stop_loss_graceful_fallback(self):
        """When candidate lacks valid SL distance, fallback safely to 1.0x."""
        # Empty candidate
        sizing1 = self.rm_enabled.calculate_trade_sizing(
            candidate={},
            total_equity=1000.0,
            leverage=20,
        )
        self.assertFalse(sizing1["volatility_parity_active"])
        self.assertEqual(sizing1["volatility_multiplier"], 1.0)
        self.assertEqual(sizing1["final_notional"], 1000.0)

        # Zero entry price
        sizing2 = self.rm_enabled.calculate_trade_sizing(
            candidate={"entry_price": 0.0, "stop_loss": 100.0},
            total_equity=1000.0,
            leverage=20,
        )
        self.assertFalse(sizing2["volatility_parity_active"])
        self.assertEqual(sizing2["volatility_multiplier"], 1.0)

    def test_interaction_with_loss_streak_throttling(self):
        """Volatility Parity stacks multiplicatively with Loss Streak Throttle."""
        # Consecutive losses = 2 (triggers 0.5x throttle)
        self.rm_enabled.load_state = MagicMock(return_value={"consecutive_losses": 2, "consecutive_wins": 0})

        candidate = {
            "symbol": "SUIUSDT",
            "entry_price": 2.00,
            "stop_loss": 1.928,  # 3.6% SL -> 0.60x vol mult
        }
        sizing = self.rm_enabled.calculate_trade_sizing(
            candidate=candidate,
            total_equity=1000.0,
            leverage=20,
        )
        # Base 1000 * 0.5 (streak) * 0.60 (vol) = 300.0 USDT
        self.assertEqual(sizing["multiplier"], 0.5)
        self.assertEqual(sizing["volatility_multiplier"], 0.60)
        self.assertEqual(sizing["final_notional"], 300.0)
        self.assertEqual(sizing["final_margin"], 15.0)


if __name__ == "__main__":
    unittest.main()
