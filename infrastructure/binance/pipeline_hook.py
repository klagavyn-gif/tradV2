"""
Pipeline Hook for Binance USDT-M Futures Auto-Trading.
Integrates into tradV2 alert lifecycle to automatically:
1. Sync and update Breakeven stops on open positions.
2. Execute actionable entry candidates on Binance Futures.
3. Send Telegram execution receipts.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def execute_binance_auto_trade_pipeline(
    sent_candidates: Optional[List[Dict[str, Any]]] = None,
    *,
    config,
    helpers: Dict[str, Any],
    get_now,
    send_telegram_alert=None,
) -> Dict[str, Any]:
    """
    Main hook called after alert dispatch.
    Controlled by config.BINANCE_FUTURES_AUTO_TRADE_ENABLED.
    """
    enabled = bool(getattr(config, "BINANCE_FUTURES_AUTO_TRADE_ENABLED", False))
    if not enabled:
        return {"enabled": False, "executed": []}

    api_key = str(getattr(config, "BINANCE_FUTURES_API_KEY", "") or "").strip()
    api_secret = str(getattr(config, "BINANCE_FUTURES_API_SECRET", "") or "").strip()
    testnet = bool(getattr(config, "BINANCE_FUTURES_TESTNET", True))

    if not api_key or not api_secret:
        logger.warning("Binance Futures Auto-Trade enabled but API Key/Secret is missing!")
        return {"enabled": True, "error": "missing_credentials"}

    from .client import BinanceFuturesClient
    from .order_manager import BinanceFuturesOrderManager

    client = BinanceFuturesClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )

    trade_notional = float(getattr(config, "BINANCE_FUTURES_TRADE_NOTIONAL_USDT", 10.0))
    max_positions = int(getattr(config, "BINANCE_FUTURES_MAX_POSITIONS", 2))
    leverage = int(getattr(config, "BINANCE_FUTURES_LEVERAGE", 1))
    margin_type = str(getattr(config, "BINANCE_FUTURES_MARGIN_TYPE", "ISOLATED"))
    be_r = float(getattr(config, "TELEGRAM_ALERT_REALIZED_BREAKEVEN_R", 1.2))
    ts_r = float(getattr(config, "TELEGRAM_ALERT_REALIZED_TRAILING_R", 2.0))
    trail_dist_r = float(getattr(config, "TELEGRAM_ALERT_REALIZED_TRAILING_DISTANCE_R", 0.8))

    order_mgr = BinanceFuturesOrderManager(
        client=client,
        trade_notional_usdt=trade_notional,
        max_positions=max_positions,
        leverage=leverage,
        margin_type=margin_type,
        breakeven_r=be_r,
        trailing_r=ts_r,
        trailing_dist_r=trail_dist_r,
    )

    results = {
        "enabled": True,
        "testnet": testnet,
        "be_updates": [],
        "executed_entries": [],
    }

    # 1. Sync Breakeven Stops on any existing open positions
    try:
        be_updates = order_mgr.sync_breakeven_stops()
        results["be_updates"] = be_updates
        if be_updates and callable(send_telegram_alert):
            for u in be_updates:
                msg = (
                    f"🛡️ <b>[Auto-Trade] ปรับกันทุน (Breakeven) สำเร็จ!</b>\n"
                    f"<b>เหรียญ:</b> {u['symbol']} | <b>ฝั่ง:</b> {u['side']}\n"
                    f"<b>Stop Loss ใหม่:</b> {u['new_sl']:,} (เลื่อนมาที่ทุนเพื่อกันความเสี่ยง)"
                )
                try:
                    send_telegram_alert(msg)
                except Exception:
                    pass
    except Exception as e:
        logger.exception("Error syncing Breakeven stops: %s", e)

    # 2. Filter actionable entry candidates from this run
    actionable = []
    for c in (sent_candidates or []):
        dispatch_label = str(c.get("dispatch_status_label") or "").strip()
        alert_intent = str(c.get("alert_intent") or "").strip().lower()
        signal = str(c.get("signal") or "").strip().upper()

        if dispatch_label == "เข้าได้" and alert_intent == "entry" and signal in ("BUY", "SELL"):
            actionable.append(c)

    # 3. Execute actionable entries
    for c in actionable:
        try:
            exec_res = order_mgr.execute_candidate(c)
            results["executed_entries"].append(exec_res)
            if exec_res.get("success"):
                logger.info("Auto-traded on Binance: %s %s @ %s", exec_res["symbol"], exec_res["signal"], exec_res["entry_price"])
                if callable(send_telegram_alert):
                    env_badge = "🧪 [Testnet]" if testnet else "⚡ [Real Money]"
                    margin_used = exec_res['notional'] / max(1, leverage)
                    try:
                        usdt_bal = client.get_usdt_balance()
                        bal_str = f"\n💰 <b>เงินคงเหลือในพอร์ต:</b> {usdt_bal['available']:,.2f} USDT (รวม {usdt_bal['total']:,.2f} USDT)"
                    except Exception:
                        bal_str = ""
                    msg = (
                        f"🤖 <b>{env_badge} เปิดออเดอร์ Binance Futures สำเร็จ!</b>\n"
                        f"────────────────\n"
                        f"<b>เหรียญ:</b> {exec_res['symbol']} | <b>ฝั่ง:</b> {exec_res['signal']}\n"
                        f"<b>จำนวน:</b> {exec_res['quantity']} (มูลค่าสัญญา ~{exec_res['notional']:.2f} USDT)\n"
                        f"<b>Margin ที่ใช้:</b> ~{margin_used:.2f} USDT ({leverage}x {margin_type})\n"
                        f"<b>ราคาเข้า:</b> {exec_res['entry_price']:,}\n"
                        f"<b>Stop Loss:</b> {exec_res['stop_loss']:,} (ตั้งคำสั่ง STOP_MARKET แล้ว)\n"
                        f"<b>Trailing Stop:</b> เมื่อถึง +{ts_r}R จะเริ่มลาก SL ตามราคา ({trail_dist_r}R)"
                        f"{bal_str}"
                    )
                    try:
                        send_telegram_alert(msg)
                    except Exception:
                        pass
            else:
                logger.warning("Auto-trade skipped/failed for %s: %s", c.get("symbol"), exec_res.get("reason"))
        except Exception as e:
            logger.exception("Exception executing auto-trade candidate: %s", e)

    return results
