#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_telegram_message_consistency.py

Verifies 100% harmonization and consistency across Telegram messages:
1. When status is 'รอ':
   - '🎯 สรุป:' matches '🟡 รอ'
   - '👉 การทำต่อ:' matches 'ยังไม่เข้า รอราคาใกล้ entry zone หรือรอแท่งยืนยันเพิ่ม'
   - '📌 แผน:' is used instead of '📌 แผนเข้า:'
   - Bottom dispatch status lines match 'รอ'
   - No 'short trade' phrase appears on BUY signals
2. When status is 'ห้ามเข้า':
   - '🎯 สรุป:' matches '⛔ ห้ามเข้า'
   - '👉 การทำต่อ:' matches avoid/exit guidance
   - '📌 แผน:' is used
   - Bottom dispatch status lines match 'ห้ามเข้า'
3. When status is 'เข้าได้':
   - '🎯 สรุป:' matches '🟢 เข้าได้'
   - '👉 การทำต่อ:' matches 'เข้าได้ตามสัญญาณหลัก...'
   - '📌 แผนเข้า:' is used
   - Bottom dispatch status lines match 'เข้าได้'
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain.alerts.candidates.common import (
    _append_dispatch_readiness_lines,
    _candidate_dispatch_status,
    resolve_short_trade_profile,
)


class TestTelegramMessageConsistency(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            TELEGRAM_ALERT_ENTRY_WINDOW_ENABLE=True,
            TELEGRAM_ALERT_ENTRY_WINDOW_MAX_DISTANCE_PCT=3.0,
            TELEGRAM_ALERT_ENTRY_WINDOW_MAX_DISTANCE_R=1.3,
            TELEGRAM_ALERT_SHORT_TRADE_ENABLE=True,
            TELEGRAM_ALERT_SHORT_TRADE_RR_FLOOR=1.0,
            TELEGRAM_ALERT_SHORT_TRADE_PREMIUM_MIN_RR=1.25,
            TELEGRAM_ALERT_SHORT_TRADE_STANDARD_MAX_BARS=3,
            TELEGRAM_ALERT_SHORT_TRADE_HARD_STALE_BARS=6,
            TELEGRAM_ALERT_SHORT_TRADE_STANDARD_MIN_SOURCE_COUNT=1,
            TELEGRAM_ALERT_SHORT_TRADE_PREMIUM_MIN_SOURCE_COUNT=2,
            TELEGRAM_ALERT_SHORT_TRADE_PREMIUM_MIN_CONFIDENCE=78.0,
            TELEGRAM_ALERT_SHORT_TRADE_PREMIUM_SCORE_BONUS=4.0,
            TELEGRAM_ALERT_SHORT_TRADE_STANDARD_SCORE_BONUS=1.5,
            TELEGRAM_ALERT_SHORT_TRADE_WATCH_SCORE_PENALTY=3.0,
            TELEGRAM_ALERT_SHORT_TRADE_REQUIRE_REGIME_ALIGNMENT=True,
        )

        # Baseline raw message from alerts/messages.py before candidate evaluation
        # (originally has '🟢 เข้าได้', '📌 แผนเข้า:', etc.)
        self.raw_buy_message = (
            "<b>🟢 สัญญาณหลัก BUY | PAXG-USD</b>\n"
            "<i>PAXG-USD</i>\n"
            "────────────────\n"
            "<b>🎯 สรุป:</b> 🟢 เข้าได้\n"
            "<b>📝 เหตุผล:</b> แผนยังใกล้จุดเข้าและมี SL/TP พร้อมใช้งาน\n"
            "<b>💵 ราคาปัจจุบัน:</b> ราคา 4,184.74 (+0.00%) | Conf 98%\n"
            "<b>👉 การทำต่อ:</b> เข้าได้ตามสัญญาณหลัก ถ้ายังไม่มีสถานะสามารถเปิด BUY/Long ตามแผนได้\n"
            "<b>⚠️ หมายเหตุ:</b> วาง SL/TP ตามแผน และไม่ไล่ราคา\n"
            "<b>📌 แผนเข้า:</b> Entry 4,184.74 | SL 4,121.97 | TP1 4,247.51 | TP2 4,297.73\n"
            "<b>📏 ความเสี่ยง:</b> Risk 1.68% | RR TP1 1.00R / TP2 1.80R\n"
            "<b>🛡️ แผนกันทุน:</b> กำไร ≥ +1.2R ขยับ SL มาที่ Entry | กำไร ≥ +2.0R ใช้ Trailing Stop 0.8R\n"
            "────────────────\n"
            "🕒 <b>เวลา:</b> 2026-10-15 15:17\n"
            '<a href="https://th.tradingview.com/chart/?symbol=CRYPTO:PAXGUSD">📈 TradingView</a>'
        )

    def test_stale_buy_signal_harmonized_to_watch(self):
        """Simulate today's PAXG-USD where bars_since = 5 caused 'รอ' but top said '🟢 เข้าได้'."""
        candidate = {
            "symbol": "PAXG-USD",
            "signal": "BUY",
            "timeframe_minutes": 15,
            "bars_since_signal": 5.0,
            "signal_age_bars_at_analysis": 5.0,
            "plan": {
                "entry_price": 4184.74,
                "current_price": 4184.74,
                "stop_loss": 4121.97,
                "take_profit": 4247.51,
                "risk_reward": 1.0,
            },
            "alert_intent": "entry",
        }

        # Step 1: short trade profile should give detailed freshness reason
        profile = resolve_short_trade_profile(candidate, config=self.config)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["bucket"], "watch")
        self.assertIn("สัญญาณผ่านไปแล้ว 5 แท่ง (~75 นาที) เกินเกณฑ์เปิดไม้สด (3 แท่ง)", profile["reason"])
        self.assertNotIn("short trade", profile["reason"])  # Must not say 'short trade' for BUY!

        candidate["short_trade_bucket"] = profile["bucket"]
        candidate["short_trade_reason"] = profile["reason"]

        # Step 2: append dispatch readiness lines & harmonize
        msg = _append_dispatch_readiness_lines(self.raw_buy_message, candidate, config=self.config)

        # Assert 100% harmonization
        self.assertIn("<b>🎯 สรุป:</b> 🟡 รอ", msg)
        self.assertNotIn("<b>🎯 สรุป:</b> 🟢 เข้าได้", msg)
        self.assertIn("<b>👉 การทำต่อ:</b> ยังไม่เข้า รอราคาใกล้ entry zone หรือรอแท่งยืนยันเพิ่ม", msg)
        self.assertNotIn("เข้าได้ตามสัญญาณหลัก ถ้ายังไม่มีสถานะสามารถเปิด BUY/Long ตามแผนได้", msg)
        self.assertIn("<b>📌 แผน:</b>", msg)
        self.assertNotIn("<b>📌 แผนเข้า:</b>", msg)
        self.assertIn("<b>🧭 หมวดเหตุผล:</b> รอ คุณภาพจังหวะยังไม่ครบ", msg)
        self.assertIn("<b>📝 เหตุผลใช้งาน:</b> สัญญาณผ่านไปแล้ว 5 แท่ง (~75 นาที) เกินเกณฑ์เปิดไม้สด (3 แท่ง)", msg)
        self.assertEqual(candidate["dispatch_status_label"], "รอ")
        self.assertEqual(candidate["alert_intent"], "watch")

    def test_avoid_signal_harmonized_to_avoid(self):
        """Test candidate rejected by Entry AI or Low RR is 100% harmonized to '⛔ ห้ามเข้า'."""
        candidate = {
            "symbol": "BTC-USD",
            "signal": "BUY",
            "timeframe_minutes": 15,
            "bars_since_signal": 1.0,
            "plan": {
                "entry_price": 85000.0,
                "current_price": 85000.0,
                "stop_loss": 84000.0,
                "take_profit": 85500.0,
                "risk_reward": 0.5,
            },
            "entry_ai_bucket": "avoid",
            "entry_ai_reason": "Entry AI ประเมินว่าความเสี่ยงสูงเกินไป",
            "alert_intent": "avoid",
        }

        msg = _append_dispatch_readiness_lines(self.raw_buy_message, candidate, config=self.config)

        self.assertIn("<b>🎯 สรุป:</b> ⛔ ห้ามเข้า", msg)
        self.assertNotIn("<b>🎯 สรุป:</b> 🟢 เข้าได้", msg)
        self.assertIn("<b>👉 การทำต่อ:</b> สัญญาณนี้ใช้ลดความเสี่ยงหรือปิดสถานะเดิม ไม่ใช่จุดเปิดไม้ใหม่", msg)
        self.assertIn("<b>⚠️ หมายเหตุ:</b> ถ้าไม่มีสถานะอยู่แล้วให้ข้ามข้อความนี้", msg)
        self.assertIn("<b>📌 แผน:</b>", msg)
        self.assertNotIn("<b>📌 แผนเข้า:</b>", msg)
        self.assertIn("<b>🧭 หมวดเหตุผล:</b> ห้ามเข้า", msg)
        self.assertEqual(candidate["dispatch_status_label"], "ห้ามเข้า")
        self.assertEqual(candidate["alert_intent"], "avoid")

    def test_approved_signal_stays_entry(self):
        """Test fresh, high conviction signal remains 100% '🟢 เข้าได้'."""
        candidate = {
            "symbol": "ETH-USD",
            "signal": "BUY",
            "timeframe_minutes": 15,
            "bars_since_signal": 1.0,
            "source_count": 2,
            "confidence": 85.0,
            "plan": {
                "entry_price": 2600.0,
                "current_price": 2605.0,
                "stop_loss": 2550.0,
                "take_profit": 2700.0,
                "risk_reward": 2.0,
            },
            "alert_intent": "entry",
        }

        profile = resolve_short_trade_profile(candidate, config=self.config)
        self.assertEqual(profile["bucket"], "premium_entry")
        candidate["short_trade_bucket"] = profile["bucket"]
        candidate["short_trade_reason"] = profile["reason"]

        msg = _append_dispatch_readiness_lines(self.raw_buy_message, candidate, config=self.config)

        self.assertIn("<b>🎯 สรุป:</b> 🟢 เข้าได้", msg)
        self.assertIn("<b>👉 การทำต่อ:</b> เข้าได้ตามสัญญาณหลัก ถ้ายังไม่มีสถานะสามารถเปิด BUY/Long ตามแผนได้", msg)
        self.assertIn("<b>📌 แผนเข้า:</b>", msg)
        self.assertIn("<b>🧭 หมวดเหตุผล:</b> เข้าได้ เข้าใกล้ entry zone", msg)
        self.assertEqual(candidate["dispatch_status_label"], "เข้าได้")
        self.assertEqual(candidate["alert_intent"], "entry")

    def test_stale_sell_signal_harmonized_to_watch(self):
        """Test SELL signal in watch status specifies SELL/Short properly."""
        candidate = {
            "symbol": "SOL-USD",
            "signal": "SELL",
            "timeframe_minutes": 15,
            "bars_since_signal": 4.0,
            "plan": {
                "entry_price": 150.0,
                "current_price": 150.0,
                "stop_loss": 155.0,
                "take_profit": 140.0,
                "risk_reward": 2.0,
            },
            "alert_intent": "entry",
        }
        profile = resolve_short_trade_profile(candidate, config=self.config)
        self.assertEqual(profile["bucket"], "watch")
        self.assertIn("4 แท่ง", profile["reason"])

        candidate["short_trade_bucket"] = profile["bucket"]
        candidate["short_trade_reason"] = profile["reason"]

        msg = _append_dispatch_readiness_lines(self.raw_buy_message, candidate, config=self.config)
        self.assertIn("<b>🎯 สรุป:</b> 🟡 รอ", msg)
        self.assertIn("<b>📌 แผน:</b>", msg)
        self.assertNotIn("<b>📌 แผนเข้า:</b>", msg)
        self.assertIn("<b>🧭 หมวดเหตุผล:</b> รอ คุณภาพจังหวะยังไม่ครบ", msg)

    def test_exit_signal_harmonized_to_exit(self):
        """Test exit signal harmonized to avoid/exit guidance."""
        candidate = {
            "symbol": "BTC-USD",
            "signal": "SELL",
            "alert_intent": "exit",
            "plan": {
                "sell_trigger": "TAKE_PROFIT",
                "entry_price": 85000.0,
                "stop_loss": 84000.0,
            },
        }
        msg = _append_dispatch_readiness_lines(self.raw_buy_message, candidate, config=self.config)
        self.assertIn("<b>🎯 สรุป:</b> ⛔ ห้ามเข้า", msg)
        self.assertIn("<b>👉 การทำต่อ:</b> สัญญาณนี้ใช้ปิดกำไรหรือลดความเสี่ยง ไม่ใช่จุดเปิดไม้ใหม่", msg)
        self.assertIn("<b>📌 แผน:</b>", msg)
        self.assertNotIn("<b>📌 แผนเข้า:</b>", msg)
        self.assertEqual(candidate["dispatch_status_label"], "ห้ามเข้า")

    def test_idempotent_multiple_calls(self):
        """Verify calling _append_dispatch_readiness_lines multiple times does not duplicate lines."""
        candidate = {
            "symbol": "PAXG-USD",
            "signal": "BUY",
            "timeframe_minutes": 15,
            "bars_since_signal": 5.0,
            "plan": {
                "entry_price": 4184.74,
                "current_price": 4184.74,
                "stop_loss": 4121.97,
                "take_profit": 4247.51,
                "risk_reward": 1.0,
            },
            "alert_intent": "entry",
            "short_trade_bucket": "watch",
            "short_trade_reason": "สัญญาณผ่านไปแล้ว 5 แท่ง (~75 นาที) เกินเกณฑ์เปิดไม้สด (3 แท่ง)",
        }
        msg1 = _append_dispatch_readiness_lines(self.raw_buy_message, candidate, config=self.config)
        msg2 = _append_dispatch_readiness_lines(msg1, candidate, config=self.config)

        self.assertEqual(msg1.count("<b>🎯 สรุป:</b>"), 1)
        self.assertEqual(msg2.count("<b>🎯 สรุป:</b>"), 1)
        self.assertEqual(msg2.count("<b>🧭 หมวดเหตุผล:</b>"), 1)
        self.assertEqual(msg2.count("<b>📝 เหตุผลใช้งาน:</b>"), 1)


    def test_cdcvix_paxg_message_harmonization(self):
        """Verify CDC+VixFix message with '🚦 สถานะ:', '📝 Decision:', '🎯 Action:' is 100% harmonized."""
        raw_cdcvix_msg = (
            "<b>🔴 CDC+VixFix 15m SELL | PAXG-USD</b>\n"
            "<i>PAXG-USD</i>\n"
            "────────────────\n"
            "<b>📍 Snapshot:</b> ราคา 4,160.75 (+0.00%)\n"
            "<b>🧭 1H Trend:</b> UP | STRONG | ActionZone 15m, Price Action 15m\n"
            "<b>🧪 Edge:</b> Conf 91% | WR 60.0% | ExpRR 0.45 | Trades 30\n"
            "<b>🚦 สถานะ:</b> 🟢 เข้าได้\n"
            "<b>📝 Decision:</b> แผนยังใกล้จุดเข้าและมี SL/TP พร้อมใช้งาน\n"
            "<b>🎯 Action:</b> เข้าได้ตามสัญญาณหลัก ถ้ายังไม่มีสถานะสามารถเปิด SELL/Short ตามแผนได้\n"
            "<b>⚠️ Note:</b> ครบอายุ CDC เดิม แต่ตลาดยังลงต่อและราคายังไม่ไกลจากจุดเข้าใหม่\n"
            "<b>⏳ Hold Plan:</b> ถือได้ประมาณ 6 ชม. (24 แท่ง 15m) ถ้าไม่ไปตามทางควรลดไม้หรือปิดรอบ\n"
            "<b>🧠 Context:</b> Forecast SELL 92/100 | Trend ขาลง | R>G 11/100 | flip 34 bars\n"
            "<b>📌 Plan:</b> Entry 4,142.70 | SL 4,223.16 | TP1 4,062.24 | TP2 3,997.87\n"
            "<b>📏 Risk:</b> Risk 2.18% | RR TP1 1.00R / TP2 1.80R\n"
            "<b>🧭 Level Source:</b> actual+fallback\n"
            "<b>🤖 AI:</b> ยืนยันเพิ่ม | p(win) 60% | exp +0.78%\n"
            "<b>🧠 Entry AI:</b> รอ | Watch | p(entry) 20% | p(avoid) 63%\n"
            "<b>📘 กติกาใช้งาน:</b> รอ Watch เมื่อ p(entry) ยังไม่ถึง 80% หรือ p(avoid) ยังสูง | รอบนี้ 20% / 63%\n"
            "<b>🎯 Entry Window:</b> Entry 4,142.700195 | ใกล้ entry ไม่เกิน 3.00% | หรือ 1.30R\n"
            "<b>🏃 Max Chase:</b> SELL ไม่ควรไล่ต่ำกว่า 4,038.100824\n"
            "<b>⏱️ อายุสัญญาณ:</b> 158 แท่ง 15m | ~2370 นาที | 2026-10-15 14:15:00\n"
            "────────────────\n"
            "🕒 <b>เวลา:</b> 2026-10-17 05:33\n"
            '<a href="https://th.tradingview.com/chart/?symbol=CRYPTO:PAXGUSD">📈 TradingView</a>'
        )

        candidate = {
            "symbol": "PAXG-USD",
            "signal": "SELL",
            "strategy": "CDCVIX15",
            "timeframe_minutes": 15,
            "bars_since_signal": 158.0,
            "entry_ai_bucket": "watch",
            "entry_ai_reason": "ผ่านเกณฑ์ Watch monitor ใช้เฝ้าดูจังหวะต่อเนื่อง",
            "plan": {
                "entry_price": 4142.70,
                "current_price": 4160.75,
                "stop_loss": 4223.16,
                "take_profit": 4062.24,
                "risk_reward": 1.0,
            },
        }

        harmonized = _append_dispatch_readiness_lines(raw_cdcvix_msg, candidate, config=self.config)

        # Must harmonize 🚦 สถานะ: to 🟡 รอ
        self.assertIn("<b>🚦 สถานะ:</b> 🟡 รอ", harmonized)
        self.assertNotIn("<b>🚦 สถานะ:</b> 🟢 เข้าได้", harmonized)

        # Must harmonize 🎯 Action:
        self.assertIn("<b>🎯 Action:</b> ยังไม่เข้า รอราคาใกล้ entry zone หรือรอแท่งยืนยันเพิ่ม", harmonized)
        self.assertNotIn("เข้าได้ตามสัญญาณหลัก ถ้ายังไม่มีสถานะสามารถเปิด SELL/Short ตามแผนได้", harmonized)

        # Must harmonize 📝 Decision:
        self.assertIn("<b>📝 Decision:</b> ผ่านเกณฑ์ Watch monitor ใช้เฝ้าดูจังหวะต่อเนื่อง", harmonized)

        # Must preserve 📌 Plan:
        self.assertIn("<b>📌 Plan:</b>", harmonized)


if __name__ == "__main__":
    unittest.main()

