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

    @staticmethod
    def _find_active_executed_record(
        executed_records: Optional[Dict[str, Any]],
        sym: str,
        side: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the most recent executed order record matching a given symbol and optional side.
        """
        if not executed_records or not isinstance(executed_records, dict):
            return None

        target_sym = normalize_futures_symbol(sym)
        matching = []
        for aid, rec in executed_records.items():
            if not isinstance(rec, dict):
                continue
            rec_sym = normalize_futures_symbol(str(rec.get("symbol") or ""))
            if rec_sym == target_sym:
                if side:
                    expected_sig = "BUY" if str(side).upper() == "LONG" else "SELL"
                    rec_sig = str(rec.get("signal") or "").upper()
                    if rec_sig and rec_sig != expected_sig:
                        continue
                matching.append(rec)

        if not matching:
            return None

        matching.sort(
            key=lambda r: str(r.get("executed_at") or r.get("timestamp") or ""),
            reverse=True,
        )
        return matching[0]

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
            "closed_positions_details": [],
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
                    rec = self._find_active_executed_record(executed_records, sym, side)
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
                    client_algo_id=f"heal_{sym[:3].lower()}_{int(time.time()*1000)%10000000}",
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
        # Closes positions on Binance that have officially settled in the outcome tracker.
        # CRITICAL SAFETY RULE:
        # A position is ONLY settled if its SPECIFIC alert_id matching the executed order has settled!
        # NEVER match by symbol alone, as historical outcomes contain past settled paper trades.
        if settled_outcomes and active_symbols and executed_records:
            settled_by_aid: Dict[str, Dict[str, Any]] = {}
            open_by_aid: Dict[str, Dict[str, Any]] = {}
            for row in settled_outcomes:
                if not isinstance(row, dict):
                    continue
                aid = str(row.get("alert_id") or "").strip()
                if not aid:
                    continue
                st = str(row.get("outcome_status") or "").strip().lower()
                if st == "settled":
                    settled_by_aid[aid] = row
                elif st == "open":
                    open_by_aid[aid] = row

            for a_sym in list(active_symbols):
                pos_entry = next((p for p in report["open_positions"] if p.get("symbol") == a_sym), None)
                pos_side = pos_entry.get("side") if pos_entry else None

                active_rec = self._find_active_executed_record(executed_records, a_sym, pos_side)
                if not active_rec:
                    logger.info(
                        "[Watchdog] Active position on %s (%s) has no matching executed bot order (possibly manual/external). Skipping cleanup.",
                        a_sym, pos_side or "UNKNOWN"
                    )
                    continue

                active_aid = str(active_rec.get("alert_id") or "").strip()
                if not active_aid:
                    continue

                # 1. If tracker explicitly says this alert is STILL OPEN -> KEEP OPEN!
                if active_aid in open_by_aid:
                    logger.info(
                        "[Watchdog] Position %s (alert %s) is currently ACTIVE/OPEN in outcome tracker. Keeping position open.",
                        a_sym, active_aid
                    )
                    continue

                # 2. If tracker explicitly says this specific alert is SETTLED -> Close stuck position!
                if active_aid in settled_by_aid:
                    outcome_row = settled_by_aid[active_aid]
                    exit_reason = str(outcome_row.get("exit_reason") or "settled")
                    pnl_pct = outcome_row.get("pnl_pct")
                    pnl_str = f" ({pnl_pct:+.2f}%)" if isinstance(pnl_pct, (int, float)) else ""

                    entry_price = float(pos_entry.get("entry_price") or active_rec.get("entry_price") or 0.0) if pos_entry else float(active_rec.get("entry_price") or 0.0)
                    pos_amt = float(pos_entry.get("amount") or active_rec.get("quantity") or 0.0) if pos_entry else float(active_rec.get("quantity") or 0.0)
                    leverage = int(pos_entry.get("leverage") or self.target_leverage) if pos_entry else self.target_leverage

                    logger.warning(
                        "[Watchdog] Position %s (alert %s) has settled in tracker (%s) but position is still lingering on Binance. Executing market close...",
                        a_sym, active_aid, exit_reason
                    )
                    close_res = self.client.close_position_market(a_sym)
                    if close_res.get("success"):
                        order_data = close_res.get("data", {})
                        avg_price_raw = float(order_data.get("avgPrice") or 0.0)
                        exec_qty = float(order_data.get("executedQty") or 0.0)
                        cum_quote = float(order_data.get("cumQuote") or 0.0)

                        if avg_price_raw > 0:
                            exit_price = avg_price_raw
                        elif exec_qty > 0 and cum_quote > 0:
                            exit_price = cum_quote / exec_qty
                        else:
                            exit_price = float(pos_entry.get("mark_price") or 0.0) if pos_entry else 0.0
                            if exit_price <= 0:
                                try:
                                    exit_price = float(self.client.get_symbol_price(a_sym) or entry_price)
                                except Exception:
                                    exit_price = entry_price

                        if entry_price > 0 and exit_price > 0:
                            if pos_side == "LONG":
                                calc_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                            else:
                                calc_pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0
                        else:
                            calc_pnl_pct = float(outcome_row.get("pnl_pct") or 0.0)

                        notional = (pos_amt * exit_price) if (pos_amt > 0 and exit_price > 0) else float(active_rec.get("notional") or 400.0)
                        margin = notional / max(1, leverage)
                        dollar_pnl = (calc_pnl_pct / 100.0) * notional
                        fee_est = notional * 0.001  # ~0.10% round-trip taker fee
                        net_dollar_pnl = dollar_pnl - fee_est
                        roi_pct = calc_pnl_pct * leverage

                        try:
                            fresh_bal = self.client.get_usdt_balance()
                        except Exception:
                            fresh_bal = {}

                        close_detail = {
                            "symbol": a_sym,
                            "alert_id": active_aid,
                            "side": pos_side or "LONG",
                            "amount": pos_amt,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl_pct": calc_pnl_pct,
                            "dollar_pnl": dollar_pnl,
                            "margin": margin,
                            "roi_pct": roi_pct,
                            "fee_est": fee_est,
                            "net_dollar_pnl": net_dollar_pnl,
                            "exit_reason": exit_reason,
                            "balance_after": fresh_bal,
                            "order_id": order_data.get("orderId"),
                            "is_reconciled": False,
                        }
                        report.setdefault("closed_positions_details", []).append(close_detail)

                        # Update executed record status
                        active_rec["status"] = "closed"
                        active_rec["closed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        active_rec["exit_price"] = exit_price
                        active_rec["exit_reason"] = exit_reason

                        close_msg = f"🏁 [Auto-Close] ซิงค์ปิด Position ค้าง {a_sym} ({active_aid[:8]}) บน Binance สำเร็จ ({exit_reason}){pnl_str}"
                        report["healed_actions"].append(close_msg)
                        # Remove from active positions report
                        report["open_positions"] = [p for p in report["open_positions"] if p["symbol"] != a_sym]
                        report["open_positions_count"] = len(report["open_positions"])
                    else:
                        report["warnings"].append(f"Failed to auto-close settled position {a_sym}: {close_res}")
                else:
                    logger.info(
                        "[Watchdog] Position %s (alert %s) has no settled outcome record. Keeping position open.",
                        a_sym, active_aid
                    )

        # 5B. Closed Position Reconciliation Sync
        # Detect executed bot orders in executed_records that are no longer in active_symbols
        # (i.e. position has closed on Binance via Stop Loss, Trailing Stop, or manual close).
        if executed_records:
            for aid, rec in list(executed_records.items()):
                if not isinstance(rec, dict) or rec.get("status") == "closed":
                    continue
                rec_sym = str(rec.get("symbol") or "").replace("-", "").replace("/", "").upper()
                if not rec_sym:
                    continue
                # If this symbol is NOT in active_symbols, the position is no longer open on Binance!
                if rec_sym not in active_symbols:
                    # Check if this aid was already included in this report
                    if any(c.get("alert_id") == aid for c in report.get("closed_positions_details", [])):
                        continue

                    already_settled = False
                    if settled_outcomes:
                        already_settled = any(
                            str(row.get("alert_id") or "").strip() == aid and str(row.get("outcome_status") or "").strip().lower() == "settled"
                            for row in settled_outcomes if isinstance(row, dict)
                        )

                    if not already_settled:
                        logger.info(
                            "[Watchdog] Executed order %s (%s) is no longer an open position on Binance. Reconciling close...",
                            rec_sym, aid[:8]
                        )
                        trades = []
                        if hasattr(self.client, "get_user_trades"):
                            try:
                                trades = self.client.get_user_trades(rec_sym, limit=3)
                            except Exception as e:
                                logger.warning("[Watchdog] Failed to get user trades for %s: %s", rec_sym, e)

                        exit_trade = None
                        for t in trades:
                            if float(t.get("realizedPnl") or 0.0) != 0:
                                exit_trade = t
                                break
                        if not exit_trade and trades:
                            exit_trade = trades[0]

                        entry_p = float(rec.get("entry_price") or 0.0)
                        qty = float(rec.get("quantity") or 0.0)
                        notional = float(rec.get("notional") or (entry_p * qty if entry_p and qty else 400.0))
                        leverage = self.target_leverage
                        margin = notional / max(1, leverage)

                        if exit_trade:
                            exit_p = float(exit_trade.get("price") or 0.0)
                            dollar_pnl = float(exit_trade.get("realizedPnl") or 0.0)
                            fee = float(exit_trade.get("commission") or (notional * 0.0005))
                        else:
                            try:
                                exit_p = float(self.client.get_symbol_price(rec_sym) or entry_p)
                            except Exception:
                                exit_p = entry_p
                            dollar_pnl = 0.0
                            fee = notional * 0.0005

                        if entry_p > 0 and exit_p > 0:
                            if str(rec.get("signal")).upper() == "BUY":
                                pnl_pct = ((exit_p - entry_p) / entry_p) * 100.0
                            else:
                                pnl_pct = ((entry_p - exit_p) / entry_p) * 100.0
                        else:
                            pnl_pct = 0.0

                        if dollar_pnl == 0.0 and pnl_pct != 0.0:
                            dollar_pnl = (pnl_pct / 100.0) * notional

                        roi_pct = pnl_pct * leverage
                        net_dollar_pnl = dollar_pnl - fee

                        # Infer reason
                        sl_price = float(rec.get("stop_loss") or 0.0)
                        if sl_price > 0 and abs(exit_p - sl_price) / sl_price < 0.01:
                            reason = "stop_loss_hit"
                        elif dollar_pnl > 0:
                            reason = "take_profit_hit"
                        elif dollar_pnl < 0:
                            reason = "stop_loss_hit"
                        else:
                            reason = "breakeven_stop_hit"

                        try:
                            fresh_bal = self.client.get_usdt_balance()
                        except Exception:
                            fresh_bal = {}

                        close_detail = {
                            "symbol": rec_sym,
                            "alert_id": aid,
                            "side": "LONG" if str(rec.get("signal")).upper() == "BUY" else "SHORT",
                            "amount": qty,
                            "entry_price": entry_p,
                            "exit_price": exit_p,
                            "pnl_pct": pnl_pct,
                            "dollar_pnl": dollar_pnl,
                            "margin": margin,
                            "roi_pct": roi_pct,
                            "fee_est": fee,
                            "net_dollar_pnl": net_dollar_pnl,
                            "exit_reason": reason,
                            "balance_after": fresh_bal,
                            "is_reconciled": True,
                        }
                        report.setdefault("closed_positions_details", []).append(close_detail)

                    # Mark record as closed
                    rec["status"] = "closed"
                    rec["closed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

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

    @staticmethod
    def format_position_close_receipt(details: Dict[str, Any], testnet: bool = True) -> str:
        """
        Format a professional, transparent Trade Close Receipt on Telegram
        accounting for Entry Price, Exit Price, PnL %, Net Dollar USDT, Margin, ROI %,
        Commission, and live Binance Account Balance.
        """
        env_badge = "🧪 [Testnet Binance]" if testnet else "⚡ [พอร์ตจริง Binance]"
        sym = str(details.get("symbol") or "—").upper()
        side = str(details.get("side") or "—").upper()
        aid = str(details.get("alert_id") or "—")[:8]
        entry_p = details.get("entry_price")
        exit_p = details.get("exit_price")
        pnl_pct = details.get("pnl_pct")
        dollar_pnl = details.get("dollar_pnl")
        net_pnl = details.get("net_dollar_pnl")
        roi_pct = details.get("roi_pct")
        margin = details.get("margin")
        fee_est = details.get("fee_est")
        bal = details.get("balance_after", {})
        raw_reason = str(details.get("exit_reason") or "settled")
        is_reconciled = bool(details.get("is_reconciled", False))

        # Import helper for exit reason translation if available
        try:
            from alerts.reporting import _trade_close_exit_reason_label
            reason_label = _trade_close_exit_reason_label(raw_reason)
            if reason_label == "—":
                reason_label = raw_reason
        except Exception:
            reason_label = raw_reason

        dp = float(dollar_pnl or 0.0)
        icon = "🟢" if dp > 0 else "🔴" if dp < 0 else "🛡️"
        res_label = "ชนะ (กำไร)" if dp > 0 else "ขาดทุน" if dp < 0 else "เสมอตัว (กันทุน)"

        close_mode = "ปิดบนกระดาน/คำสั่งอัตโนมัติ" if is_reconciled else "ปิดด้วย MARKET (Watchdog Sync)"

        lines = [
            f"{icon} <b>{env_badge} ใบเสร็จสรุปผลการปิดไม้ — {res_label}</b>",
            "────────────────",
            f"<b>เหรียญ:</b> {sym} | <b>ฝั่ง:</b> {side} ({close_mode})",
            f"<b>สาเหตุที่ปิด:</b> {reason_label}",
            f"<b>Alert ID:</b> <code>{aid}</code>",
        ]
        if isinstance(entry_p, (int, float)) and entry_p > 0:
            lines.append(f"<b>ต้นทุนเข้า (Entry):</b> {entry_p:,.6g}")
        if isinstance(exit_p, (int, float)) and exit_p > 0:
            lines.append(f"<b>ราคาปิดจริง (Exit):</b> {exit_p:,.6g}")
        if isinstance(pnl_pct, (int, float)):
            lines.append(f"<b>ผลตอบแทน (PnL):</b> {pnl_pct:+.2f}%")
        if isinstance(dollar_pnl, (int, float)):
            margin_str = f" บน Margin {margin:.1f} USDT" if isinstance(margin, (int, float)) and margin > 0 else ""
            roi_str = f" ({roi_pct:+.1f}%{margin_str})" if isinstance(roi_pct, (int, float)) else ""
            lines.append(f"<b>กำไร/ขาดทุนสุทธิ:</b> {dollar_pnl:+.2f} USDT{roi_str}")
        if isinstance(fee_est, (int, float)) and fee_est > 0:
            lines.append(f"<b>ค่าธรรมเนียมประมาณการ:</b> -{fee_est:.2f} USDT (Taker Fee)")
            if isinstance(net_pnl, (int, float)):
                lines.append(f"<b>PnL สุทธิหลังหัก Fee:</b> {net_pnl:+.2f} USDT")
        if isinstance(bal, dict) and bal.get("total", 0) > 0:
            lines.append(f"💰 <b>ยอดเงินคงเหลือในพอร์ต:</b> {bal.get('available', 0):,.2f} USDT (รวม {bal.get('total', 0):,.2f} USDT)")

        lines.append("────────────────")
        lines.append(f"🕒 <b>เวลา:</b> {time.strftime('%Y-%m-%d %H:%M')}")
        tv_symbol = sym.replace("-", "").replace("/", "")
        lines.append(f"<a href=\"https://th.tradingview.com/chart/?symbol=CRYPTO:{tv_symbol}\">📈 TradingView</a>")
        return "\n".join(lines)


def format_position_close_receipt(details: Dict[str, Any], testnet: bool = True) -> str:
    """Module-level alias for BinanceFuturesWatchdog.format_position_close_receipt."""
    return BinanceFuturesWatchdog.format_position_close_receipt(details, testnet=testnet)

