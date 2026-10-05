"""
Pipeline Hook for Binance USDT-M Futures Auto-Trading.
Integrates into tradV2 alert lifecycle to automatically:
1. Sync and update Breakeven stops on open positions.
2. Execute actionable entry candidates on Binance Futures.
3. Send Telegram execution receipts.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _load_executed_orders(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Could not read executed orders from %s: %s", file_path, e)
        return {}


def _save_executed_orders(file_path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        if len(data) > 300:
            keys = sorted(data.keys(), key=lambda k: data[k].get("executed_at", ""), reverse=True)
            data = {k: data[k] for k in keys[:300]}
        tmp_path = f"{file_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, file_path)
    except Exception as e:
        logger.warning("Could not save executed orders to %s: %s", file_path, e)


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
    from .risk_manager import BinanceFuturesRiskManager

    client = BinanceFuturesClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )

    trade_notional = float(getattr(config, "BINANCE_FUTURES_TRADE_NOTIONAL_USDT", 400.0))
    max_positions = int(getattr(config, "BINANCE_FUTURES_MAX_POSITIONS", 2))
    leverage = int(getattr(config, "BINANCE_FUTURES_LEVERAGE", 20))
    margin_type = str(getattr(config, "BINANCE_FUTURES_MARGIN_TYPE", "ISOLATED"))
    be_r = float(getattr(config, "TELEGRAM_ALERT_REALIZED_BREAKEVEN_R", 1.2))
    ts_r = float(getattr(config, "TELEGRAM_ALERT_REALIZED_TRAILING_R", 2.0))
    trail_dist_r = float(getattr(config, "TELEGRAM_ALERT_REALIZED_TRAILING_DISTANCE_R", 0.8))

    risk_mgr = BinanceFuturesRiskManager(
        dynamic_sizing_enabled=getattr(config, "BINANCE_FUTURES_DYNAMIC_SIZING_ENABLED", True),
        equity_risk_pct=getattr(config, "BINANCE_FUTURES_EQUITY_RISK_PCT", 0.40),
        min_notional_usdt=getattr(config, "BINANCE_FUTURES_MIN_TRADE_NOTIONAL_USDT", 20.0),
        max_notional_usdt=getattr(config, "BINANCE_FUTURES_MAX_TRADE_NOTIONAL_USDT", 1500.0),
        loss_streak_throttle=getattr(config, "BINANCE_FUTURES_LOSS_STREAK_THROTTLE", 2),
        loss_streak_max=getattr(config, "BINANCE_FUTURES_LOSS_STREAK_MAX", 3),
        circuit_breaker_hours=getattr(config, "BINANCE_FUTURES_CIRCUIT_BREAKER_HOURS", 6.0),
        daily_max_loss_pct=getattr(config, "BINANCE_FUTURES_DAILY_MAX_LOSS_PCT", 4.0),
        win_streak_scale_enable=getattr(config, "BINANCE_FUTURES_WIN_STREAK_SCALE_ENABLE", True),
        win_streak_scale_mult=getattr(config, "BINANCE_FUTURES_WIN_STREAK_SCALE_MULT", 1.25),
    )

    # Sync settled outcomes into risk manager
    try:
        outcomes_path = helpers.get("alert_outcomes_file_path", lambda: ".data/telegram_alerts/realized_outcomes.json")()
        if os.path.exists(outcomes_path):
            with open(outcomes_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "outcomes" in payload:
                risk_mgr.update_from_settled_outcomes(
                    payload["outcomes"],
                    leverage=leverage,
                    notional_per_trade=trade_notional,
                )
    except Exception as e:
        logger.warning("Could not sync outcomes to RiskManager: %s", e)

    order_mgr = BinanceFuturesOrderManager(
        client=client,
        trade_notional_usdt=trade_notional,
        max_positions=max_positions,
        leverage=leverage,
        margin_type=margin_type,
        breakeven_r=be_r,
        trailing_r=ts_r,
        trailing_dist_r=trail_dist_r,
        risk_manager=risk_mgr,
    )

    results = {
        "enabled": True,
        "testnet": testnet,
        "be_updates": [],
        "executed_entries": [],
        "cb_alerted": False,
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

    # Load executed orders history for idempotency
    executed_orders_path = helpers.get("binance_executed_orders_path", lambda: ".data/telegram_alerts/binance_executed_orders.json")()
    executed_records = _load_executed_orders(executed_orders_path)

    # 2. Filter actionable entry candidates from this run
    from alerts.reporting import infer_alert_intent
    actionable = []
    for c in (sent_candidates or []):
        signal = str(c.get("signal") or "").strip().upper()
        if signal not in ("BUY", "SELL"):
            continue
        dispatch_label = str(c.get("dispatch_status_label") or "").strip()
        if dispatch_label in ("ห้ามเข้า", "รอ"):
            continue
        intent, _ = infer_alert_intent(c)
        if intent == "entry":
            actionable.append(c)

    # 3. Execute actionable entries
    for c in actionable:
        aid = str(c.get("alert_id") or "").strip()
        if aid and aid in executed_records:
            logger.info("Alert %s already executed on Binance, skipping duplicate.", aid)
            continue

        try:
            exec_res = order_mgr.execute_candidate(c)
            results["executed_entries"].append(exec_res)
            if exec_res.get("success"):
                if aid:
                    executed_records[aid] = {
                        "alert_id": aid,
                        "timestamp": str(c.get("timestamp") or ""),
                        "symbol": exec_res.get("symbol"),
                        "signal": exec_res.get("signal"),
                        "entry_price": exec_res.get("entry_price"),
                        "quantity": exec_res.get("quantity"),
                        "notional": exec_res.get("notional"),
                        "executed_at": datetime.utcnow().isoformat(),
                    }
                    _save_executed_orders(executed_orders_path, executed_records)
                logger.info("Auto-traded on Binance: %s %s @ %s", exec_res["symbol"], exec_res["signal"], exec_res["entry_price"])
                if callable(send_telegram_alert):
                    env_badge = "🧪 [Testnet]" if testnet else "⚡ [Real Money]"
                    margin_used = exec_res['notional'] / max(1, leverage)
                    sizing_info = exec_res.get("sizing_info") or {}
                    sizing_note = ""
                    mult_reason = str(sizing_info.get("multiplier_reason") or "")
                    if mult_reason.startswith("loss_streak_throttle"):
                        streak = sizing_info.get("loss_streak", 2)
                        sizing_note = f"\n⚠️ <b>[Risk Throttle]</b> ลดขนาดไม้ลง 50% เนื่องจากผลแพ้ {streak} ไม้ล่าสุด"
                    elif mult_reason.startswith("win_streak_scale"):
                        streak = sizing_info.get("win_streak", 2)
                        sizing_note = f"\n🔥 <b>[Trend Momentum]</b> ขยายขนาดไม้ 1.25x เนื่องจากชนะต่อเนื่อง {streak} ไม้"

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
                        f"{sizing_note}"
                        f"{bal_str}"
                    )
                    try:
                        send_telegram_alert(msg)
                    except Exception:
                        pass
            else:
                reason = exec_res.get("reason")
                if reason == "circuit_breaker_active":
                    cb_reason = exec_res.get("circuit_breaker_reason") or "drawdown_protection"
                    cb_until = exec_res.get("circuit_breaker_until") or "ชั่วคราว"
                    if callable(send_telegram_alert) and not results.get("cb_alerted"):
                        msg = (
                            f"🚨 <b>[Risk Control] บอทอยู่ในโหมด Circuit Breaker พักการเทรด!</b>\n"
                            f"────────────────\n"
                            f"<b>สาเหตุ:</b> {cb_reason}\n"
                            f"<b>สถานะ:</b> หยุดเปิดไม้ชั่วคราวเพื่อปกป้องเงินทุน (พักจนถึง {cb_until})\n"
                            f"<b>คำแนะนำ:</b> รอให้สภาวะตลาดหลุดพ้นช่วงสับขาหลอก ระบบจะกลับมาเทรดอัตโนมัติ"
                        )
                        try:
                            send_telegram_alert(msg)
                            results["cb_alerted"] = True
                        except Exception:
                            pass
                logger.warning("Auto-trade skipped/failed for %s: %s", c.get("symbol"), reason)
        except Exception as e:
            logger.exception("Exception executing auto-trade candidate: %s", e)

    return results
