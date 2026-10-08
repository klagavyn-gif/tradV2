"""
Binance Futures Health & Status Watchdog.
Performs continuous reconciliation, health checks, and self-healing across Binance USDT-M Futures:
1. Naked Position Defense: Ensures every open position has an active protective STOP_MARKET algo order. Auto-heals if missing.
2. Ghost Order Defense: Identifies and cancels lingering orders/algo orders for symbols with no open position.
3. Stuck/Settled Position Sync: Closes positions on Binance that have settled in the outcome tracker.
4. Liquidation Proximity Alert: Calculates distance to liquidation price and warns if under safety threshold.
5. Leverage & Margin Type Guard: Enforces target leverage (e.g. 20x) and ISOLATED margin type.
6. Margin Utilization & Position Limits: Monitors free equity and alerts if margin utilization > 80% or positions > limit.
7. Position Mode Sanity Check: Verifies One-way Mode vs Hedge Mode.
8. Rich Telemetry & Reporting: Provides structured audit results and formatted Telegram health status.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from .client import BinanceFuturesClient
from .order_manager import normalize_futures_symbol, round_to_tick

logger = logging.getLogger(__name__)


class BinanceFuturesWatchdog:
    """Automated health audit and self-healing engine for Binance USDT-M Futures."""

    def __init__(
        self,
        client: BinanceFuturesClient,
        *,
        target_leverage: int = 20,
        target_margin_type: str = "ISOLATED",
        max_positions: int = 2,
        min_liquidation_distance_pct: float = 3.0,
        default_stop_loss_pct: float = 1.8,
    ):
        self.client = client
        self.target_leverage = int(target_leverage)
        self.target_margin_type = str(target_margin_type).upper()
        self.max_positions = int(max_positions)
        self.min_liquidation_distance_pct = float(min_liquidation_distance_pct)
        self.default_stop_loss_pct = float(default_stop_loss_pct)

    def audit_and_heal(
        self,
        *,
        settled_outcomes: Optional[List[Dict[str, Any]]] = None,
        executed_records: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a full health audit of Binance account and positions.
        Takes proactive self-healing actions and returns a detailed report.
        """
        report: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "healthy": True,
            "position_mode": "ONE_WAY",
            "open_positions_count": 0,
            "open_positions": [],
            "issues_detected": [],
            "healed_actions": [],
            "warnings": [],
            "account_equity": 0.0,
            "available_balance": 0.0,
            "total_unrealized_pnl": 0.0,
            "margin_utilization_pct": 0.0,
        }

        # 0. Account Balances
        try:
            bal = self.client.get_usdt_balance()
            report["account_equity"] = float(bal.get("total", 0.0))
            report["available_balance"] = float(bal.get("available", 0.0))
            report["total_unrealized_pnl"] = float(bal.get("crossUnPnl", 0.0))
        except Exception as e:
            logger.warning("[Watchdog] Failed to fetch balances: %s", e)
            report["warnings"].append(f"Balance check error: {e}")

        # 0.5 Position Mode Check (One-way vs Hedge Mode)
        try:
            mode_data = self.client.get_position_mode()
            is_hedge = bool(mode_data.get("dualSidePosition", False))
            report["position_mode"] = "HEDGE" if is_hedge else "ONE_WAY"
        except Exception:
            report["position_mode"] = "ONE_WAY"

        # 1. Fetch live open positions
        try:
            positions = self.client.get_positions()
        except Exception as e:
            logger.error("[Watchdog] Failed to fetch positions: %s", e)
            report["healthy"] = False
            report["issues_detected"].append(f"Failed to fetch positions: {e}")
            return report

        report["open_positions_count"] = len(positions)
        active_symbols = set()

        # Risk Guard: Max Positions check
        if len(positions) > self.max_positions:
            warn_max = f"⚠️ [Risk Alert] จำนวนไม้ที่เปิดอยู่ ({len(positions)}) เกินเพดานที่กำหนด ({self.max_positions} ไม้)!"
            report["warnings"].append(warn_max)
            logger.warning("[Watchdog] %s", warn_max)

        # 2. Fetch all open orders and algo orders in one shot
        try:
            open_regular_orders = self.client.get_open_orders()
        except Exception as e:
            logger.warning("[Watchdog] Failed to fetch regular orders: %s", e)
            open_regular_orders = []

        try:
            open_algo_orders = self.client.get_open_algo_orders()
        except Exception as e:
            logger.warning("[Watchdog] Failed to fetch algo orders: %s", e)
            open_algo_orders = []

        total_margin_used = 0.0

        # 3. Audit each open position
        for pos in positions:
            sym = pos.get("symbol", "")
            if not sym:
                continue
            active_symbols.add(sym)

            side = str(pos.get("side", "")).upper()
            amt = abs(float(pos.get("positionAmt", pos.get("amount", 0.0))))
            entry_price = float(pos.get("entryPrice", 0.0))
            mark_price = float(pos.get("markPrice", 0.0))
            liq_price = float(pos.get("liquidationPrice", 0.0))
            unrealized_pnl = float(pos.get("unRealizedProfit", 0.0))
            leverage = int(pos.get("leverage", 1))
            margin_type = str(pos.get("marginType", "")).upper()

            # Estimate Initial Margin and ROE %
            initial_margin = (amt * entry_price) / max(1, leverage)
            total_margin_used += initial_margin
            roe_pct = (unrealized_pnl / initial_margin * 100.0) if initial_margin > 0 else 0.0

            pos_info: Dict[str, Any] = {
                "symbol": sym,
                "side": side,
                "amount": amt,
                "entry_price": entry_price,
                "mark_price": mark_price,
                "liquidation_price": liq_price,
                "unrealized_pnl": unrealized_pnl,
                "roe_pct": roe_pct,
                "leverage": leverage,
                "margin_type": margin_type,
                "sl_protected": False,
                "sl_price": None,
                "ts_active": False,
            }

            # Check 3A: Stop Loss Protection (Naked Position Defense)
            sym_algo_orders = [o for o in open_algo_orders if str(o.get("symbol", "")).upper() == sym]
            sym_regular_orders = [o for o in open_regular_orders if str(o.get("symbol", "")).upper() == sym]

            sl_algo = next(
                (o for o in sym_algo_orders if str(o.get("orderType") or o.get("type") or "").upper() == "STOP_MARKET"),
                None
            )
            sl_reg = next(
                (o for o in sym_regular_orders if str(o.get("type", "")).upper() == "STOP_MARKET"),
                None
            )

            if sl_algo:
                pos_info["sl_protected"] = True
                pos_info["sl_price"] = float(sl_algo.get("triggerPrice", 0.0))
            elif sl_reg:
                pos_info["sl_protected"] = True
                pos_info["sl_price"] = float(sl_reg.get("stopPrice", 0.0))

            # Check Trailing Stop
            ts_algo = next(
                (o for o in sym_algo_orders if str(o.get("orderType") or o.get("type") or "").upper() == "TRAILING_STOP_MARKET"),
                None
            )
            if ts_algo:
                pos_info["ts_active"] = True

            # SELF-HEALING: If position has NO Stop Loss, place emergency protective Stop Loss!
            if not pos_info["sl_protected"]:
                report["healthy"] = False
                issue_text = f"Naked position detected on {sym} ({side})! No Stop Loss order found."
                report["issues_detected"].append(issue_text)
                logger.error("[Watchdog] %s Auto-healing emergency Stop Loss...", issue_text)

                filters = self.client.get_symbol_filters(sym)
                tick_size = filters["tickSize"] if filters else 0.0001

                # Check if original Stop Loss exists in executed_records
                recorded_sl = None
                if executed_records and isinstance(executed_records, dict):
                    rec = executed_records.get(sym) or executed_records.get(f"{sym}_{side}")
                    if rec and isinstance(rec, dict) and rec.get("stop_loss"):
                        try:
                            recorded_sl = float(rec["stop_loss"])
                        except (ValueError, TypeError):
                            recorded_sl = None

                if recorded_sl and ((side == "LONG" and recorded_sl < entry_price) or (side == "SHORT" and recorded_sl > entry_price)):
                    emergency_sl = round_to_tick(recorded_sl, tick_size)
                    sl_source = "สัญญาณเดิม (Strategy ATR)"
                else:
                    sl_dist = entry_price * (self.default_stop_loss_pct / 100.0)
                    if side == "LONG":
                        emergency_sl = round_to_tick(max(entry_price - sl_dist, entry_price * 0.95), tick_size)
                    else:
                        emergency_sl = round_to_tick(min(entry_price + sl_dist, entry_price * 1.05), tick_size)
                    sl_source = f"สำรองฉุกเฉิน ({self.default_stop_loss_pct}%)"

                sl_side = "SELL" if side == "LONG" else "BUY"

                heal_res = self.client.create_algo_order(
                    symbol=sym,
                    side=sl_side,
                    order_type="STOP_MARKET",
                    trigger_price=emergency_sl,
                    close_position=True,
                    client_algo_id=f"heal_sl_{int(time.time())}",
                )

                if heal_res.get("success"):
                    pos_info["sl_protected"] = True
                    pos_info["sl_price"] = emergency_sl
                    action_msg = f"🛡️ [Auto-Healed] ตั้งคำสั่ง STOP_MARKET ฉุกเฉินให้ {sym} ({side}) เรียบร้อยที่ราคา {emergency_sl} [{sl_source}]"
                    report["healed_actions"].append(action_msg)
                    logger.info("[Watchdog] %s", action_msg)
                else:
                    report["issues_detected"].append(f"Failed to auto-heal SL for {sym}: {heal_res}")

            # Check 3B: Liquidation Safety Distance
            if liq_price > 0 and mark_price > 0:
                dist_pct = abs(mark_price - liq_price) / mark_price * 100.0
                pos_info["liq_distance_pct"] = dist_pct
                if dist_pct < self.min_liquidation_distance_pct:
                    warn_msg = f"⚠️ [Liquidation Alert] {sym} ห่างจากราคา Liquidation เพียง {dist_pct:.2f}% (Mark: {mark_price}, Liq: {liq_price})"
                    report["warnings"].append(warn_msg)
                    logger.warning("[Watchdog] %s", warn_msg)

            # Check 3C: Margin Type and Leverage Guard
            if margin_type and margin_type != self.target_margin_type:
                report["warnings"].append(f"{sym} marginType is {margin_type} (expected {self.target_margin_type})")
            if leverage != self.target_leverage:
                try:
                    self.client.set_leverage(sym, self.target_leverage)
                    report["healed_actions"].append(f"Adjusted leverage on {sym} from {leverage}x to {self.target_leverage}x")
                except Exception:
                    pass

            report["open_positions"].append(pos_info)

        # Margin Utilization Guard
        if report["account_equity"] > 0:
            margin_ratio = (total_margin_used / report["account_equity"]) * 100.0
            report["margin_utilization_pct"] = margin_ratio
            if margin_ratio > 80.0:
                warn_margin = f"⚠️ [Margin Warning] มีการใช้ Margin ไปแล้ว {margin_ratio:.1f}% ของ Equity ทั้งหมด (เหลือเงินสำรองต่ำกว่า 20%)"
                report["warnings"].append(warn_margin)
                logger.warning("[Watchdog] %s", warn_margin)

        # 4. Ghost / Lingering Order Cleanup Defense
        # Check all open regular and algo orders: if order symbol has NO active position, cancel it!
        all_order_symbols = set()
        for o in open_regular_orders:
            s = o.get("symbol")
            if s:
                all_order_symbols.add(s)
        for o in open_algo_orders:
            s = o.get("symbol")
            if s:
                all_order_symbols.add(s)

        ghost_symbols = all_order_symbols - active_symbols
        for g_sym in ghost_symbols:
            logger.info("[Watchdog] Found lingering orders on %s with no active position. Cleaning up...", g_sym)
            try:
                self.client.cancel_all_open_orders(g_sym)
            except Exception:
                pass
            try:
                self.client.cancel_all_algo_orders(g_sym)
            except Exception:
                pass
            cleanup_msg = f"🧹 [Auto-Cleanup] ยกเลิกคำสั่งตกค้างของ {g_sym} เรียบร้อย (ไม่มี Position ถือครองแล้ว)"
            report["healed_actions"].append(cleanup_msg)

        # 5. Settled Positions Cleanup Sync
        # If an alert was settled in outcomes tracker, but position is still open on Binance -> Market Close!
        if settled_outcomes and active_symbols:
            settled_clean_symbols = {}
            for row in settled_outcomes:
                if not isinstance(row, dict):
                    continue
                if str(row.get("outcome_status") or "").strip().lower() == "settled":
                    s_sym = normalize_futures_symbol(str(row.get("symbol") or ""))
                    if s_sym:
                        settled_clean_symbols[s_sym] = row

            for a_sym in list(active_symbols):
                if a_sym in settled_clean_symbols:
                    outcome_row = settled_clean_symbols[a_sym]
                    exit_reason = str(outcome_row.get("exit_reason") or "settled")
                    pnl_pct = outcome_row.get("pnl_pct")
                    pnl_str = f" ({pnl_pct:+.2f}%)" if isinstance(pnl_pct, (int, float)) else ""

                    logger.info("[Watchdog] Position %s is settled in tracker (%s). Closing on Binance...", a_sym, exit_reason)
                    close_res = self.client.close_position_market(a_sym)
                    if close_res.get("success"):
                        close_msg = f"🏁 [Auto-Close] ซิงค์ปิด Position {a_sym} บน Binance สำเร็จ ({exit_reason}){pnl_str}"
                        report["healed_actions"].append(close_msg)
                        # Remove from active positions report
                        report["open_positions"] = [p for p in report["open_positions"] if p["symbol"] != a_sym]
                        report["open_positions_count"] = len(report["open_positions"])
                    else:
                        report["warnings"].append(f"Failed to auto-close settled position {a_sym}: {close_res}")

        return report

    def format_telegram_alert(self, audit_result: Dict[str, Any]) -> Optional[str]:
        """
        Format a clear, readable Telegram notification if issues were detected or healed,
        or when periodic position status is needed.
        """
        issues = audit_result.get("issues_detected", [])
        healed = audit_result.get("healed_actions", [])
        warnings = audit_result.get("warnings", [])
        positions = audit_result.get("open_positions", [])

        # Only send if there are actions, issues, warnings, or active positions
        if not issues and not healed and not warnings and not positions:
            return None

        lines = []
        if healed:
            lines.append("🛡️ <b>[Binance Watchdog] ตรวจพบและแก้ไขระบบอัตโนมัติ (Self-Healed):</b>")
            for h in healed:
                lines.append(f"• {h}")
            lines.append("────────────────")

        if issues:
            lines.append("🚨 <b>[Binance Watchdog] ข้อผิดพลาดที่ต้องตรวจสอบ:</b>")
            for i in issues:
                lines.append(f"• {i}")
            lines.append("────────────────")

        if warnings:
            lines.append("⚠️ <b>[Binance Watchdog] แจ้งเตือนความเสี่ยง:</b>")
            for w in warnings:
                lines.append(f"• {w}")
            lines.append("────────────────")

        if positions:
            lines.append(f"📊 <b>สถานะออเดอร์ใน Binance ปัจจุบัน ({len(positions)} ไม้):</b>")
            for p in positions:
                sl_badge = f"✅ SL: {p['sl_price']}" if p["sl_protected"] else "❌ ไม่มี SL"
                ts_badge = " | 🔄 TS เปิดอยู่" if p["ts_active"] else ""
                roe_sign = "+" if p["roe_pct"] >= 0 else ""
                lines.append(
                    f"• <b>{p['symbol']}</b> ({p['side']}) | ปริมาณ: {p['amount']}\n"
                    f"  ราคาเข้า: {p['entry_price']} | ปัจจุบัน: {p['mark_price']}\n"
                    f"  PnL: {roe_sign}{p['unrealized_pnl']:+.2f} USDT ({roe_sign}{p['roe_pct']:.1f}% ROE)\n"
                    f"  ความปลอดภัย: {sl_badge}{ts_badge}"
                )
            equity = audit_result.get("account_equity", 0.0)
            avail = audit_result.get("available_balance", 0.0)
            if equity > 0:
                lines.append(f"💰 <b>เงินในพอร์ต:</b> ว่าง {avail:,.2f} USDT / รวม {equity:,.2f} USDT")

        lines.append(f"🕒 <i>ตรวจสอบเมื่อ: {audit_result.get('timestamp')}</i>")
        return "\n".join(lines)
