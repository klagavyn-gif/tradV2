"""
Telegram Gateway & Live Card Message Manager for tradV2.

Provides:
1. TelegramSendResult (supports boolean evaluation and message_id extraction)
2. send_telegram_alert (with retry, rate limit handling, and reply_to support)
3. edit_telegram_alert (in-place Telegram message editor)
4. LiveCardStore (persists and tracks active Telegram trade cards across runs)
5. Formatters for in-place Breakeven and Trade Close transformations
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

try:
    import requests as http_requests
except ImportError:
    http_requests = None


class TelegramSendResult:
    """
    Wrapper for Telegram API response.
    Implements __bool__ for 100% backward compatibility with `if send_telegram_alert(...):`.
    """

    def __init__(
        self,
        ok: bool,
        message_id: Optional[int] = None,
        chat_id: Optional[str] = None,
        error_code: Optional[int] = None,
        description: Optional[str] = None,
    ):
        self.ok = bool(ok)
        self.message_id = message_id
        self.chat_id = str(chat_id) if chat_id is not None else None
        self.error_code = error_code
        self.description = description

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        return f"<TelegramSendResult ok={self.ok} message_id={self.message_id} error={self.description}>"


def send_telegram_alert(
    message: str,
    *,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    reply_to_message_id: Optional[int] = None,
    disable_notification: bool = False,
    parse_mode: str = "HTML",
    max_retries: int = 3,
) -> TelegramSendResult:
    """
    Send a message to Telegram.
    Returns TelegramSendResult (acts as boolean True/False while holding message_id).
    """
    token = (bot_token or os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    target_chat_id = (chat_id or os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    target_thread_id = thread_id or os.environ.get("TELEGRAM_THREAD_ID")

    if not token or not target_chat_id or not message:
        return TelegramSendResult(ok=False, description="missing_credentials_or_message")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload: Dict[str, Any] = {
        "chat_id": target_chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    if target_thread_id:
        payload["message_thread_id"] = target_thread_id
    if reply_to_message_id:
        payload["reply_to_message_id"] = int(reply_to_message_id)
    if disable_notification:
        payload["disable_notification"] = True

    for attempt in range(max_retries):
        try:
            if http_requests is not None:
                resp = http_requests.post(url, json=payload, timeout=12)
                status_code = resp.status_code
                try:
                    resp_json = resp.json()
                except Exception:
                    resp_json = {}
            else:
                import urllib.request
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=12) as response:
                    status_code = response.getcode()
                    resp_json = json.loads(response.read().decode("utf-8"))

            if 200 <= status_code < 300 and resp_json.get("ok"):
                msg_id = resp_json.get("result", {}).get("message_id")
                return TelegramSendResult(
                    ok=True,
                    message_id=msg_id,
                    chat_id=target_chat_id,
                )

            # Handle Rate Limit (429)
            if status_code == 429:
                retry_after = int(resp_json.get("parameters", {}).get("retry_after", 3))
                time.sleep(retry_after)
                continue

            err_desc = resp_json.get("description", f"status_{status_code}")
            logger.warning("Telegram sendMessage attempt %d failed: %s", attempt + 1, err_desc)
            if attempt == max_retries - 1:
                return TelegramSendResult(
                    ok=False,
                    error_code=status_code,
                    description=err_desc,
                )
        except Exception as e:
            logger.warning("Telegram sendMessage attempt %d exception: %s", attempt + 1, e)
            if attempt == max_retries - 1:
                return TelegramSendResult(ok=False, description=str(e))

    return TelegramSendResult(ok=False, description="max_retries_exceeded")


def edit_telegram_alert(
    message_id: int,
    new_message: str,
    *,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "HTML",
    max_retries: int = 3,
) -> TelegramSendResult:
    """
    Edit an existing message in Telegram in-place.
    """
    token = (bot_token or os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    target_chat_id = (chat_id or os.environ.get("TELEGRAM_CHAT_ID") or "").strip()

    if not token or not target_chat_id or not message_id or not new_message:
        return TelegramSendResult(ok=False, description="missing_parameters")

    url = f"https://api.telegram.org/bot{token}/editMessageText"
    payload: Dict[str, Any] = {
        "chat_id": target_chat_id,
        "message_id": int(message_id),
        "text": new_message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    for attempt in range(max_retries):
        try:
            if http_requests is not None:
                resp = http_requests.post(url, json=payload, timeout=12)
                status_code = resp.status_code
                try:
                    resp_json = resp.json()
                except Exception:
                    resp_json = {}
            else:
                import urllib.request
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=12) as response:
                    status_code = response.getcode()
                    resp_json = json.loads(response.read().decode("utf-8"))

            if 200 <= status_code < 300 and resp_json.get("ok"):
                return TelegramSendResult(
                    ok=True,
                    message_id=int(message_id),
                    chat_id=target_chat_id,
                )

            desc = str(resp_json.get("description", ""))
            # If message is not modified, consider it successful no-op
            if "message is not modified" in desc.lower():
                return TelegramSendResult(
                    ok=True,
                    message_id=int(message_id),
                    chat_id=target_chat_id,
                    description="not_modified",
                )

            # Handle rate limits
            if status_code == 429:
                retry_after = int(resp_json.get("parameters", {}).get("retry_after", 3))
                time.sleep(retry_after)
                continue

            logger.warning("Telegram editMessageText attempt %d failed: %s", attempt + 1, desc)
            if attempt == max_retries - 1:
                return TelegramSendResult(
                    ok=False,
                    error_code=status_code,
                    description=desc,
                )
        except Exception as e:
            logger.warning("Telegram editMessageText attempt %d exception: %s", attempt + 1, e)
            if attempt == max_retries - 1:
                return TelegramSendResult(ok=False, description=str(e))

    return TelegramSendResult(ok=False, description="max_retries_exceeded")


class LiveCardStore:
    """
    Manages active Telegram trade card tracking in .data/telegram_alerts/live_trade_messages.json
    Allows retrieving message_id by symbol or alert_id for in-place updates.
    """

    def __init__(self, file_path: str = ".data/telegram_alerts/live_trade_messages.json"):
        self.file_path = file_path

    def _normalize_symbol(self, symbol: str) -> str:
        s = str(symbol or "").strip().upper()
        s = s.replace("-", "").replace("/", "")
        if s.endswith("USD") and not s.endswith("USDT"):
            s = s[:-3] + "USDT"
        return s

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("Could not load LiveCardStore from %s: %s", self.file_path, e)
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
            # Prune old closed cards if count exceeds 150
            if len(data) > 150:
                keys = sorted(data.keys(), key=lambda k: data[k].get("updated_at", ""), reverse=True)
                data = {k: data[k] for k in keys[:150]}
            tmp_path = f"{self.file_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.file_path)
        except Exception as e:
            logger.warning("Could not save LiveCardStore to %s: %s", self.file_path, e)

    def register_card(
        self,
        symbol: str,
        message_id: int,
        chat_id: Optional[str] = None,
        *,
        alert_id: str = "",
        signal: str = "BUY",
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        original_message: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a newly dispatched Entry card."""
        sym_key = self._normalize_symbol(symbol)
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        data = self._load()

        card_entry: Dict[str, Any] = {
            "symbol": sym_key,
            "raw_symbol": symbol,
            "alert_id": str(alert_id or "").strip(),
            "message_id": int(message_id),
            "chat_id": str(chat_id or "").strip(),
            "signal": str(signal or "BUY").upper(),
            "entry_price": float(entry_price) if isinstance(entry_price, (int, float)) else None,
            "stop_loss": float(stop_loss) if isinstance(stop_loss, (int, float)) else None,
            "take_profit": float(take_profit) if isinstance(take_profit, (int, float)) else None,
            "status": "OPEN",
            "breakeven_active": False,
            "trailing_active": False,
            "created_at": now_str,
            "updated_at": now_str,
            "original_message": original_message,
        }
        if isinstance(extra, dict):
            card_entry.update(extra)

        data[sym_key] = card_entry
        if alert_id:
            data[f"aid_{alert_id}"] = {"symbol_ref": sym_key}

        self._save(data)
        logger.info("[LiveCardStore] Registered card for %s (msg_id=%s)", sym_key, message_id)
        return card_entry

    def get_card(self, symbol: Optional[str] = None, alert_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Look up active card by symbol or alert_id."""
        data = self._load()
        if alert_id:
            ref = data.get(f"aid_{alert_id}")
            if isinstance(ref, dict) and ref.get("symbol_ref"):
                card = data.get(ref["symbol_ref"])
                if isinstance(card, dict) and card.get("message_id"):
                    return card
        if symbol:
            sym_key = self._normalize_symbol(symbol)
            card = data.get(sym_key)
            if isinstance(card, dict) and card.get("message_id"):
                return card
        return None

    def update_card(
        self,
        symbol_or_alert_id: str,
        *,
        status: Optional[str] = None,
        breakeven_active: Optional[bool] = None,
        trailing_active: Optional[bool] = None,
        new_stop_loss: Optional[float] = None,
        updated_message: Optional[str] = None,
    ) -> bool:
        """Update an active card's state."""
        data = self._load()
        target_key = None

        sym_key = self._normalize_symbol(symbol_or_alert_id)
        if sym_key in data:
            target_key = sym_key
        elif f"aid_{symbol_or_alert_id}" in data:
            ref = data.get(f"aid_{symbol_or_alert_id}", {})
            target_key = ref.get("symbol_ref")

        if not target_key or target_key not in data:
            return False

        card = data[target_key]
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        if status:
            card["status"] = str(status)
        if breakeven_active is not None:
            card["breakeven_active"] = bool(breakeven_active)
        if trailing_active is not None:
            card["trailing_active"] = bool(trailing_active)
        if new_stop_loss is not None:
            card["stop_loss"] = float(new_stop_loss)
        if updated_message:
            card["original_message"] = updated_message
        card["updated_at"] = now_str

        data[target_key] = card
        self._save(data)
        return True

    def close_card(self, symbol_or_alert_id: str, outcome: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Mark a card as CLOSED and record outcome summary."""
        data = self._load()
        target_key = None

        sym_key = self._normalize_symbol(symbol_or_alert_id)
        if sym_key in data:
            target_key = sym_key
        elif f"aid_{symbol_or_alert_id}" in data:
            ref = data.get(f"aid_{symbol_or_alert_id}", {})
            target_key = ref.get("symbol_ref")

        if not target_key or target_key not in data:
            return None

        card = data[target_key]
        card["status"] = "CLOSED"
        card["updated_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(outcome, dict):
            card["close_result"] = outcome.get("outcome_result")
            card["pnl_pct"] = outcome.get("pnl_pct")
            card["rr_realized"] = outcome.get("rr_realized")
            card["exit_reason"] = outcome.get("exit_reason")

        data[target_key] = card
        self._save(data)
        logger.info("[LiveCardStore] Closed card for %s", target_key)
        return card

    def list_active_cards(self) -> List[Dict[str, Any]]:
        """Return all cards that are currently OPEN."""
        data = self._load()
        active = []
        for k, v in data.items():
            if k.startswith("aid_"):
                continue
            if isinstance(v, dict) and str(v.get("status", "")).upper() == "OPEN":
                active.append(v)
        return active


def format_breakeven_edit_text(original_text: str, new_sl: float) -> str:
    """
    Transforms the original entry message HTML to reflect the Breakeven Stop status.
    """
    text = str(original_text or "")
    if not text:
        return text

    # Update Stop Loss line with Breakeven lock indicator
    import re
    # Match patterns like <b>Stop Loss:</b> 1.7800 or Stop Loss: 1.7800
    sl_pattern = r"(<b>Stop Loss:</b>\s*)([0-9.,]+)([^\n]*)"
    replacement = rf"\g<1>{new_sl:,} 🔒 <i>(กันทุนแล้ว — ความเสี่ยง 0%)</i>"
    new_text, count = re.subn(sl_pattern, replacement, text, count=1)
    if count == 0:
        # Fallback if SL pattern didn't match exactly
        new_text = text + f"\n🛡️ <b>[Breakeven]</b> Stop Loss เลื่อนมาที่ทุนแล้ว ({new_sl:,})"

    # Update status indicator in header if present
    if "กำลังถือครอง (OPEN)" in new_text:
        new_text = new_text.replace("กำลังถือครอง (OPEN)", "🛡️ ปรับกันทุนแล้ว (RISK-FREE)")
    elif "เปิดออเดอร์ Binance Futures สำเร็จ!" in new_text:
        new_text = new_text.replace("เปิดออเดอร์ Binance Futures สำเร็จ!", "🛡️ ปรับกันทุนแล้ว — ความเสี่ยง 0%!")

    return new_text


def format_short_close_reply(outcome: Dict[str, Any]) -> str:
    """
    Formats a concise, high-visibility 1-line reply message to ping the phone on close.
    """
    symbol = str(outcome.get("symbol") or "—").strip().upper()
    result = str(outcome.get("outcome_result") or "—").strip().lower()
    pnl = outcome.get("pnl_pct")
    rr = outcome.get("rr_realized")

    icon = "🏆" if result == "win" else ("🛡️" if result == "flat" else "🛑")
    result_thai = "ชนะ (WIN)" if result == "win" else ("เสมอตัว (FLAT)" if result == "flat" else "แพ้ (LOSS)")

    pnl_str = f"{pnl:+.2f}%" if isinstance(pnl, (int, float)) else "0.0%"
    rr_str = f"{rr:+.2f}R" if isinstance(rr, (int, float)) else "—"

    dollar_str = ""
    try:
        import config
        notional = float(getattr(config, "BINANCE_FUTURES_TRADE_NOTIONAL_USDT", 400.0))
        if isinstance(pnl, (int, float)):
            dollar_val = (pnl / 100.0) * notional
            dollar_str = f" | {dollar_val:+.2f} USDT"
    except Exception:
        pass

    return f"{icon} <b>#{symbol} ปิดไม้แล้ว — {result_thai}</b>: PnL {pnl_str} ({rr_str}{dollar_str})"
