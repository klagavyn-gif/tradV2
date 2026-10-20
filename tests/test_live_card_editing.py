#!/usr/bin/env python3
"""
Test Live Card & Telegram Message Editing Lifecycle.
Validates:
1. LiveCardStore registration, retrieval, updating, closing.
2. format_breakeven_edit_text transforms entry message correctly.
3. format_short_close_reply builds high-visibility 1-line pings.
4. dispatch_trade_close_notifications correctly triggers in-place edit when card exists,
   and falls back to send_telegram_alert when card does not exist.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from infrastructure.notifications.telegram_gateway import (
    TelegramSendResult,
    LiveCardStore,
    format_breakeven_edit_text,
    format_short_close_reply,
)


def test_telegram_send_result_boolean():
    print("--> Test 1: TelegramSendResult backward compatibility as boolean...")
    success_res = TelegramSendResult(ok=True, message_id=12345, chat_id="-100123")
    assert bool(success_res) is True
    assert success_res.message_id == 12345
    assert success_res.chat_id == "-100123"

    fail_res = TelegramSendResult(ok=False, error_code=400, description="Bad Request")
    assert bool(fail_res) is False
    assert fail_res.message_id is None
    print("    [PASS] TelegramSendResult acts as boolean True/False while holding message_id")


def test_live_card_store_lifecycle():
    print("--> Test 2: LiveCardStore register, lookup, update, close...")
    temp_dir = tempfile.mkdtemp()
    store_file = os.path.join(temp_dir, "test_live_cards.json")
    try:
        store = LiveCardStore(file_path=store_file)

        # 1. Register card
        original_msg = (
            "🤖 <b>[Testnet] เปิดออเดอร์ Binance Futures สำเร็จ!</b>\n"
            "────────────────\n"
            "<b>เหรียญ:</b> SUIUSDT | <b>ฝั่ง:</b> BUY\n"
            "<b>ราคาเข้า:</b> 1.8200\n"
            "<b>Stop Loss:</b> 1.7800 (ตั้งคำสั่ง STOP_MARKET แล้ว)\n"
            "<b>Trailing Stop:</b> เมื่อถึง +2.0R จะเริ่มลาก SL ตามราคา"
        )
        card = store.register_card(
            symbol="SUIUSDT",
            message_id=99988,
            chat_id="-100999",
            alert_id="sui_aid_123",
            signal="BUY",
            entry_price=1.8200,
            stop_loss=1.7800,
            take_profit=1.9100,
            original_message=original_msg,
        )
        assert card["symbol"] == "SUIUSDT"
        assert card["message_id"] == 99988
        assert card["status"] == "OPEN"

        # 2. Lookup by symbol (with/without hyphens)
        c_by_sym = store.get_card(symbol="SUI-USD")
        assert c_by_sym is not None
        assert c_by_sym["message_id"] == 99988

        # 3. Lookup by alert_id
        c_by_aid = store.get_card(alert_id="sui_aid_123")
        assert c_by_aid is not None
        assert c_by_aid["message_id"] == 99988

        # 4. Update card to Breakeven
        updated = store.update_card("SUIUSDT", breakeven_active=True, new_stop_loss=1.8200)
        assert updated is True
        c_after = store.get_card(symbol="SUIUSDT")
        assert c_after["breakeven_active"] is True
        assert c_after["stop_loss"] == 1.8200

        # 5. Close card
        closed_card = store.close_card("SUIUSDT", outcome={"outcome_result": "win", "pnl_pct": 5.2, "rr_realized": 2.1})
        assert closed_card is not None
        assert closed_card["status"] == "CLOSED"
        assert closed_card["close_result"] == "win"

        active_list = store.list_active_cards()
        assert len(active_list) == 0
        print("    [PASS] LiveCardStore full lifecycle completed cleanly")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_breakeven_text_formatting():
    print("--> Test 3: format_breakeven_edit_text transforms message...")
    original_msg = (
        "🤖 <b>[Testnet] เปิดออเดอร์ Binance Futures สำเร็จ!</b>\n"
        "────────────────\n"
        "<b>เหรียญ:</b> ADAUSDT | <b>ฝั่ง:</b> BUY\n"
        "<b>ราคาเข้า:</b> 0.3500\n"
        "<b>Stop Loss:</b> 0.3420 (ตั้งคำสั่ง STOP_MARKET แล้ว)\n"
    )
    new_text = format_breakeven_edit_text(original_msg, 0.3500)
    assert "🛡️ ปรับกันทุนแล้ว" in new_text
    assert "0.35 🔒 <i>(กันทุนแล้ว — ความเสี่ยง 0%)</i>" in new_text
    print("    [PASS] format_breakeven_edit_text updated header and SL line")


def test_short_close_reply_formatting():
    print("--> Test 4: format_short_close_reply builds high-visibility ping...")
    win_outcome = {
        "symbol": "BTC-USD",
        "outcome_result": "win",
        "pnl_pct": 3.45,
        "rr_realized": 2.15,
    }
    ping_win = format_short_close_reply(win_outcome)
    assert "🏆" in ping_win
    assert "#BTC-USD" in ping_win
    assert "+3.45%" in ping_win
    assert "+2.15R" in ping_win

    flat_outcome = {
        "symbol": "SUI-USD",
        "outcome_result": "flat",
        "pnl_pct": -0.15,
        "rr_realized": 0.0,
    }
    ping_flat = format_short_close_reply(flat_outcome)
    assert "🛡️" in ping_flat
    assert "เสมอตัว" in ping_flat
    print("    [PASS] format_short_close_reply generated accurate icons and pings")


def test_close_dispatch_in_place_edit():
    print("--> Test 5: dispatch_trade_close_notifications in-place edit & reply ping...")
    from alerts.reporting import dispatch_trade_close_notifications

    temp_dir = tempfile.mkdtemp()
    store_file = os.path.join(temp_dir, "test_live_cards.json")
    outcomes_file = os.path.join(temp_dir, "realized_outcomes.json")
    notified_file = os.path.join(temp_dir, "notified_closes.json")

    try:
        # Pre-seed LiveCardStore with open card for NEAR-USD
        store = LiveCardStore(file_path=store_file)
        store.register_card(
            symbol="NEAR-USD",
            message_id=77711,
            chat_id="-100888",
            alert_id="near_aid_999",
            original_message="Entry message",
        )

        # Write outcomes file with 2 settled trades:
        # 1. NEAR-USD (has active card -> should be edited + short reply)
        # 2. SOL-USD (no active card -> should fall back to standard send)
        outcomes_data = {
            "outcomes": [
                {
                    "alert_id": "near_aid_999",
                    "symbol": "NEAR-USD",
                    "signal": "BUY",
                    "strategy": "CDCVIX15",
                    "alert_intent": "entry",
                    "outcome_status": "settled",
                    "outcome_result": "win",
                    "pnl_pct": 4.5,
                    "rr_realized": 2.0,
                    "exit_reason": "take_profit_hit",
                    "timestamp": "2026-10-09 10:00:00",
                },
                {
                    "alert_id": "sol_aid_888",
                    "symbol": "SOL-USD",
                    "signal": "BUY",
                    "strategy": "CDCVIX15",
                    "alert_intent": "entry",
                    "outcome_status": "settled",
                    "outcome_result": "win",
                    "pnl_pct": 3.0,
                    "rr_realized": 1.5,
                    "exit_reason": "take_profit_hit",
                    "timestamp": "2026-10-09 10:00:00",
                },
            ]
        }
        with open(outcomes_file, "w", encoding="utf-8") as f:
            json.dump(outcomes_data, f)

        # Mock config
        class DummyConfig:
            TELEGRAM_ALERT_TRADE_CLOSE_NOTIFICATIONS_ENABLE = True
            TELEGRAM_ALERT_TRADE_CLOSE_MAX_PER_RUN = 10
            TELEGRAM_ALERT_TRADE_CLOSE_ONLY_ENTRY = True
            TELEGRAM_ALERT_TRADE_CLOSE_SKIP_FLAT = False
            TELEGRAM_ALERT_TRADE_CLOSE_MAX_AGE_DAYS = 7
            TELEGRAM_ALERT_TRADE_CLOSE_ONLY_EXECUTED = False
            TELEGRAM_ALERT_EDIT_MESSAGE_ENABLE = True
            TELEGRAM_ALERT_EDIT_MESSAGE_NOTIFY_ON_CLOSE = True
            TELEGRAM_ALERT_EDIT_MESSAGE_STORE_PATH = store_file

        edited_calls = []
        with open(notified_file, "w", encoding="utf-8") as f:
            json.dump({"notified_alert_ids": []}, f)
        sent_calls = []

        import infrastructure.notifications.telegram_gateway as tg_gw
        orig_edit = tg_gw.edit_telegram_alert
        def mock_edit(msg_id, text, chat_id=None):
            edited_calls.append({"msg_id": msg_id, "text": text, "chat_id": chat_id})
            return TelegramSendResult(ok=True, message_id=msg_id)
        tg_gw.edit_telegram_alert = mock_edit

        def mock_send(msg, reply_to_message_id=None, **kwargs):
            sent_calls.append({"msg": msg, "reply_to": reply_to_message_id})
            return TelegramSendResult(ok=True, message_id=555)

        import threading
        lock = threading.Lock()

        import trad
        helpers = dict(trad._reporting_module_helpers())
        helpers["alert_realized_enabled"] = lambda: True
        helpers["alert_outcomes_file_path"] = lambda: outcomes_file
        helpers["read_telegram_alert_history"] = lambda days=90: []
        helpers["live_card_store_path"] = lambda: store_file

        import alerts.reporting as rep
        orig_build = rep._build_telegram_realized_report_from_entries
        rep._build_telegram_realized_report_from_entries = lambda *args, **kwargs: {}
        res = dispatch_trade_close_notifications(
            config=DummyConfig(),
            helpers=helpers,
            get_now=lambda: datetime(2026, 10, 9, 12, 0, 0),
            send_telegram_alert=mock_send,
            history_lock=lock,
            strategy_order=["CDCVIX15"],
            realized_report_days=30,
        )

        print("DEBUG RES:", res)
        assert res["sent"] == 2
        # NEAR-USD was edited in-place
        assert len(edited_calls) == 1
        assert edited_calls[0]["msg_id"] == 77711
        assert "ปิดไม้แล้ว — ชนะ" in edited_calls[0]["text"]

        # 2 sent calls:
        # 1. 1-line reply for NEAR-USD quoting msg_id 77711
        # 2. full fallback message for SOL-USD
        assert len(sent_calls) == 2
        assert sent_calls[0]["reply_to"] == 77711
        assert "#NEAR-USD ปิดไม้แล้ว" in sent_calls[0]["msg"]
        assert sent_calls[1]["reply_to"] is None
        assert "SOL-USD" in sent_calls[1]["msg"]

        tg_gw.edit_telegram_alert = orig_edit
        print("    [PASS] in-place edit and fallback dispatch verified perfectly!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_telegram_send_result_boolean()
    test_live_card_store_lifecycle()
    test_breakeven_text_formatting()
    test_short_close_reply_formatting()
    test_close_dispatch_in_place_edit()
    print("\nALL 5 LIVE CARD & MESSAGE EDITING TESTS PASSED SUCCESSFULLY! 🚀")
