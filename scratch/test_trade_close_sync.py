import os
import sys
import json
import tempfile
import threading
from datetime import datetime

# Ensure repo root is on sys.path
sys.path.insert(0, r"E:\5-8-69\tradV2")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import config
import alerts.reporting as rep
from alerts.reporting import dispatch_trade_close_notifications
from infrastructure.notifications.telegram_gateway import TelegramSendResult


def test_trade_close_sync_with_binance():
    print("--> Running test_trade_close_sync_with_binance...")
    with tempfile.TemporaryDirectory() as temp_dir:
        outcomes_file = os.path.join(temp_dir, "realized_outcomes.json")
        notified_file = os.path.join(temp_dir, "notified_closes.json")
        executed_orders_file = os.path.join(temp_dir, "binance_executed_orders.json")

        # 1. Setup mock executed orders: only 'alert_real_cdcvix' was actually executed on Binance!
        executed_orders_data = {
            "alert_real_cdcvix": {
                "alert_id": "alert_real_cdcvix",
                "timestamp": "2026-10-11 12:00:00",
                "symbol": "BTC-USD",
                "signal": "BUY",
                "entry_price": 64000.0,
                "quantity": 0.006,
                "notional": 384.0,
                "executed_at": "2026-10-11T12:00:05",
            }
        }
        with open(executed_orders_file, "w", encoding="utf-8") as f:
            json.dump(executed_orders_data, f)

        # 2. Setup mock realized outcomes with 2 settled entries:
        #    a) 'alert_ghost_aw15': Unexecuted ghost entry (e.g. radar AW15 that was blocked)
        #    b) 'alert_real_cdcvix': Real executed trade on Binance
        outcomes_data = {
            "epoch": "v2",
            "outcomes": [
                {
                    "alert_id": "alert_ghost_aw15",
                    "symbol": "NEAR-USD",
                    "strategy": "AW15",
                    "signal": "BUY",
                    "alert_intent": "entry",
                    "outcome_status": "settled",
                    "outcome_result": "win",
                    "pnl_pct": 2.5,
                    "rr_realized": 1.2,
                    "entry_price": 5.0,
                    "exit_price": 5.125,
                    "timestamp": "2026-10-11 10:00:00",
                },
                {
                    "alert_id": "alert_real_cdcvix",
                    "symbol": "BTC-USD",
                    "strategy": "CDCVIX15",
                    "signal": "BUY",
                    "alert_intent": "entry",
                    "outcome_status": "settled",
                    "outcome_result": "win",
                    "pnl_pct": 3.0,
                    "rr_realized": 1.8,
                    "entry_price": 64000.0,
                    "exit_price": 65920.0,
                    "timestamp": "2026-10-11 12:00:00",
                },
            ]
        }
        with open(outcomes_file, "w", encoding="utf-8") as f:
            json.dump(outcomes_data, f)

        # Initialize notified_closes.json as empty so test doesn't trigger bootstrap of existing outcomes
        with open(notified_file, "w", encoding="utf-8") as f:
            json.dump({"notified_alert_ids": []}, f)

        # 3. Mock config with TELEGRAM_ALERT_TRADE_CLOSE_ONLY_EXECUTED = True
        class MockConfig:
            TELEGRAM_ALERT_TRADE_CLOSE_NOTIFICATIONS_ENABLE = True
            TELEGRAM_ALERT_TRADE_CLOSE_MAX_PER_RUN = 10
            TELEGRAM_ALERT_TRADE_CLOSE_ONLY_ENTRY = True
            TELEGRAM_ALERT_TRADE_CLOSE_SKIP_FLAT = False
            TELEGRAM_ALERT_TRADE_CLOSE_MAX_AGE_DAYS = 7
            TELEGRAM_ALERT_TRADE_CLOSE_ONLY_EXECUTED = True
            BINANCE_FUTURES_AUTO_TRADE_ENABLED = True
            BINANCE_FUTURES_TESTNET = True
            BINANCE_FUTURES_TRADE_NOTIONAL_USDT = 400.0
            BINANCE_FUTURES_LEVERAGE = 20
            TELEGRAM_ALERT_EDIT_MESSAGE_ENABLE = False

        sent_messages = []
        def mock_send(msg, **kwargs):
            sent_messages.append(msg)
            return TelegramSendResult(ok=True, message_id=999)

        helpers = {
            "alert_realized_enabled": lambda: True,
            "alert_outcomes_file_path": lambda: outcomes_file,
            "binance_executed_orders_path": lambda: executed_orders_file,
            "read_telegram_alert_history": lambda days=90: [],
        }

        orig_build = rep._build_telegram_realized_report_from_entries
        rep._build_telegram_realized_report_from_entries = lambda *args, **kwargs: {}

        try:
            res = dispatch_trade_close_notifications(
                config=MockConfig(),
                helpers=helpers,
                get_now=lambda: datetime(2026, 10, 11, 14, 0, 0),
                send_telegram_alert=mock_send,
                history_lock=threading.Lock(),
                strategy_order=["CDCVIX15", "AW15"],
                realized_report_days=30,
            )
            print("Dispatch result:", res)
            print("Sent messages count:", len(sent_messages))

            # Verification:
            # 1. Exactly 1 message sent (for alert_real_cdcvix)
            assert len(sent_messages) == 1, f"Expected 1 sent message, got {len(sent_messages)}"
            sent_msg = sent_messages[0]

            # 2. Check message content
            print("SENT MESSAGE:\n", sent_msg)
            assert "BTC-USD" in sent_msg, "Message must be for BTC-USD"
            assert "NEAR-USD" not in sent_msg, "Ghost NEAR-USD must NEVER be notified!"
            assert "Testnet Binance" in sent_msg or "พอร์ตจริง Binance" in sent_msg, "Must have Binance badge!"
            assert "USDT" in sent_msg, "Must show PnL in USDT!"

            # 3. Check notified closes file
            with open(notified_file, "r", encoding="utf-8") as f:
                notified_data = json.load(f)
            notified_ids = set(notified_data.get("notified_alert_ids", []))
            print("Notified IDs saved:", notified_ids)
            assert "alert_real_cdcvix" in notified_ids, "Real trade must be marked handled"
            assert "alert_ghost_aw15" in notified_ids, "Ghost trade must be marked handled so it never retries"

            # 4. Run second time -> 0 messages sent
            sent_messages.clear()
            res2 = dispatch_trade_close_notifications(
                config=MockConfig(),
                helpers=helpers,
                get_now=lambda: datetime(2026, 10, 11, 14, 15, 0),
                send_telegram_alert=mock_send,
                history_lock=threading.Lock(),
                strategy_order=["CDCVIX15", "AW15"],
                realized_report_days=30,
            )
            assert len(sent_messages) == 0, "Second run must send 0 duplicate messages"
            print("    [PASS] Approach 1 (Sync with Binance Executed Orders) verified 100%!")

        finally:
            rep._build_telegram_realized_report_from_entries = orig_build


if __name__ == "__main__":
    test_trade_close_sync_with_binance()
    print("\nALL CLOSE SYNC TESTS PASSED!")
