"""
Binance Futures Order Manager.
Handles sizing, precision rounding, idempotency, order placement,
and protective stop loss / trailing stop management on Binance USDT-M Futures.
"""

import math
import time
import logging
from typing import Dict, Any, Optional, List
from .client import BinanceFuturesClient

logger = logging.getLogger(__name__)


def round_to_step(value: float, step: float) -> float:
    """Floor value to the nearest stepSize precision."""
    if not isinstance(value, (int, float)) or not isinstance(step, (int, float)) or step <= 0:
        return float(value)
    precision = max(0, int(round(-math.log10(step)))) if step < 1.0 else 0
    stepped = math.floor(float(value) / float(step)) * float(step)
    return round(stepped, precision)


def round_to_tick(value: float, tick: float) -> float:
    """Round price to the nearest tickSize precision."""
    if not isinstance(value, (int, float)) or not isinstance(tick, (int, float)) or tick <= 0:
        return float(value)
    precision = max(0, int(round(-math.log10(tick)))) if tick < 1.0 else 0
    ticked = round(float(value) / float(tick)) * float(tick)
    return round(ticked, precision)


def normalize_futures_symbol(symbol: str) -> str:
    """Convert any symbol format (e.g. BTC-USD, BTC/USDT, BTCUSDT, NEAR) to Binance Futures format (e.g. BTCUSDT)."""
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    if text.endswith("-USD"):
        return text[:-4] + "USDT"
    if text.endswith("/USD"):
        return text[:-4] + "USDT"
    if text.endswith("/USDT"):
        return text.replace("/", "")
    if text.endswith("-USDT"):
        return text.replace("-", "")
    if text.endswith("USDT"):
        return text
    return f"{text}USDT"


class BinanceFuturesOrderManager:
    """High-level order execution and position management for Binance USDT-M Futures."""

    def __init__(
        self,
        client: BinanceFuturesClient,
        trade_notional_usdt: float = 400.0,
        max_positions: int = 2,
        leverage: int = 20,
        margin_type: str = "ISOLATED",
        breakeven_r: float = 1.2,
        trailing_r: float = 2.0,
        trailing_dist_r: float = 0.8,
        risk_manager: Optional[Any] = None,
    ):
        self.client = client
        self.trade_notional_usdt = float(trade_notional_usdt)
        self.max_positions = int(max_positions)
        self.leverage = int(leverage)
        self.margin_type = str(margin_type or "ISOLATED").upper()
        self.breakeven_r = float(breakeven_r)
        self.trailing_r = float(trailing_r)
        self.trailing_dist_r = float(trailing_dist_r)
        self.risk_manager = risk_manager

    def execute_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an entry candidate on Binance Futures with Stop Loss and Trailing Stop.
        Performs pre-flight checks: circuit breaker, balance, existing positions, max concurrent positions.
        """
        # 0. Risk Pre-flight: Check Circuit Breaker
        if self.risk_manager:
            is_blocked, cb_reason, cb_until = self.risk_manager.check_circuit_breaker()
            if is_blocked:
                return {
                    "success": False,
                    "reason": "circuit_breaker_active",
                    "circuit_breaker_reason": cb_reason,
                    "circuit_breaker_until": cb_until,
                }

        raw_symbol = str(candidate.get("symbol") or "").strip()
        signal = str(candidate.get("signal") or "").strip().upper()
        entry_price = float(candidate.get("entry_price") or 0.0)
        stop_loss = float(candidate.get("stop_loss") or 0.0)
        alert_id = str(candidate.get("alert_id") or f"trade_{int(time.time())}")[:16]

        if signal not in ("BUY", "SELL"):
            return {"success": False, "reason": "invalid_signal", "signal": signal}
        if entry_price <= 0 or stop_loss <= 0:
            return {"success": False, "reason": "invalid_entry_or_stop", "entry": entry_price, "stop": stop_loss}

        symbol = normalize_futures_symbol(raw_symbol)
        if not symbol:
            return {"success": False, "reason": "cannot_normalize_symbol", "symbol": raw_symbol}

        # 1. Pre-flight: Check active open positions
        open_positions = self.client.get_positions()
        if len(open_positions) >= self.max_positions:
            return {
                "success": False,
                "reason": "max_positions_reached",
                "open_count": len(open_positions),
                "max": self.max_positions,
            }

        # Check if symbol already has an open position
        if any(p.get("symbol") == symbol for p in open_positions):
            return {"success": False, "reason": "position_already_exists", "symbol": symbol}

        # 2. Symbol precision and sizing filters
        filters = self.client.get_symbol_filters(symbol)
        if not filters:
            return {"success": False, "reason": "symbol_filters_unavailable", "symbol": symbol}

        tick_size = filters["tickSize"]
        step_size = filters["stepSize"]
        min_qty = filters["minQty"]
        min_notional = filters["minNotional"]

        # Calculate sizing via RiskManager (Dynamic Equity / Streak Throttle)
        sizing_info = None
        if self.risk_manager:
            usdt_bal = self.client.get_usdt_balance()
            total_equity = float(usdt_bal.get("total", 0.0))
            sizing_info = self.risk_manager.calculate_trade_sizing(
                candidate=candidate,
                total_equity=total_equity,
                leverage=self.leverage,
                fallback_notional=self.trade_notional_usdt,
            )
            target_notional = sizing_info["final_notional"]
        else:
            target_notional = self.trade_notional_usdt
            sizing_info = {
                "final_notional": target_notional,
                "final_margin": target_notional / max(1, self.leverage),
                "multiplier": 1.0,
                "multiplier_reason": "normal",
            }

        # Scale order notional to meet symbol's min_notional requirement
        order_notional = max(target_notional, min_notional)
        raw_quantity = order_notional / entry_price
        quantity = round_to_step(raw_quantity, step_size)

        if quantity < min_qty or (quantity * entry_price) < min_notional:
            # Try bumping one step if slightly below minNotional
            quantity = round_to_step(quantity + step_size, step_size)

        if (quantity * entry_price) < min_notional:
            return {
                "success": False,
                "reason": "below_min_notional",
                "calculated_notional": quantity * entry_price,
                "min_notional": min_notional,
            }

        # 3. Setup margin type and leverage
        try:
            self.client.set_margin_type(symbol, self.margin_type)
        except Exception:
            pass  # Already set or not modifiable
        try:
            self.client.set_leverage(symbol, self.leverage)
        except Exception:
            pass

        # 4. Send Entry Order (MARKET for immediate clean fill on M15)
        entry_client_id = f"trad_{alert_id}_ent"
        entry_res = self.client.create_order(
            symbol=symbol,
            side=signal,
            order_type="MARKET",
            quantity=quantity,
            client_order_id=entry_client_id,
        )

        if not entry_res.get("success"):
            return {
                "success": False,
                "reason": "entry_order_failed",
                "details": entry_res,
            }

        entry_data = entry_res.get("data", {})
        exec_price = float(entry_data.get("avgPrice") or entry_price)
        if exec_price <= 0:
            exec_price = entry_price

        # 5. Place Protective Stop Loss via Algo Order API (Mandatory for Binance Futures)
        opposite_side = "SELL" if signal == "BUY" else "BUY"
        sl_client_id = f"trad_{alert_id}_sl"
        clean_sl_price = round_to_tick(stop_loss, tick_size)

        sl_res = self.client.create_algo_order(
            symbol=symbol,
            side=opposite_side,
            order_type="STOP_MARKET",
            trigger_price=clean_sl_price,
            close_position=True,
            client_algo_id=sl_client_id,
        )
        if not sl_res.get("success"):
            logger.error("Failed to place Stop Loss algo order for %s: %s", symbol, sl_res)

        # 6. Place Trailing Stop Order via Algo Order API
        risk = abs(exec_price - stop_loss)
        ts_res = None
        if self.trailing_r > 0 and risk > 0:
            if signal == "BUY":
                ts_activation = exec_price + (self.trailing_r * risk)
            else:
                ts_activation = exec_price - (self.trailing_r * risk)

            trail_dist = self.trailing_dist_r * risk
            callback_pct = round(min(5.0, max(0.1, (trail_dist / exec_price) * 100.0)), 1)
            clean_activation = round_to_tick(ts_activation, tick_size)
            ts_client_id = f"trad_{alert_id}_ts"

            ts_res = self.client.create_algo_order(
                symbol=symbol,
                side=opposite_side,
                order_type="TRAILING_STOP_MARKET",
                quantity=quantity,
                activation_price=clean_activation,
                callback_rate=callback_pct,
                reduce_only=True,
                client_algo_id=ts_client_id,
            )
            if not ts_res.get("success"):
                logger.warning("Failed to place Trailing Stop algo order for %s: %s", symbol, ts_res)

        return {
            "success": True,
            "symbol": symbol,
            "signal": signal,
            "quantity": quantity,
            "entry_price": exec_price,
            "stop_loss": clean_sl_price,
            "notional": quantity * exec_price,
            "sizing_info": sizing_info,
            "entry_order": entry_data,
            "sl_order": sl_res.get("data") if sl_res else None,
            "sl_success": bool(sl_res and sl_res.get("success")),
            "ts_order": ts_res.get("data") if ts_res else None,
            "ts_success": bool(ts_res and ts_res.get("success")),
        }

    def sync_breakeven_stops(self) -> List[Dict[str, Any]]:
        """
        Scan open positions and adjust STOP_MARKET to Entry (Breakeven)
        if markPrice has reached breakeven_r.
        """
        updated = []
        open_positions = self.client.get_positions()
        if not open_positions:
            return updated

        for pos in open_positions:
            symbol = pos["symbol"]
            side = pos["side"]
            entry_price = pos["entryPrice"]
            mark_price = pos["markPrice"]

            # Query existing open algo orders (and fallback to regular orders)
            algo_orders = self.client.get_open_algo_orders(symbol)
            sl_order = next((o for o in algo_orders if str(o.get("orderType") or o.get("type") or "").upper() == "STOP_MARKET"), None)
            is_algo = sl_order is not None

            if not sl_order:
                regular_orders = self.client.get_open_orders(symbol)
                sl_order = next((o for o in regular_orders if o.get("type") == "STOP_MARKET"), None)
                is_algo = False

            filters = self.client.get_symbol_filters(symbol)
            tick_size = filters["tickSize"] if filters else 0.01

            if not sl_order:
                # If no SL order exists on Binance at all, place one immediately!
                if side == "LONG" and mark_price > entry_price:
                    new_sl = round_to_tick(entry_price, tick_size)
                    res = self.client.create_algo_order(
                        symbol=symbol,
                        side="SELL",
                        order_type="STOP_MARKET",
                        trigger_price=new_sl,
                        close_position=True,
                        client_algo_id=f"be_{int(time.time())}",
                    )
                    updated.append({"symbol": symbol, "side": side, "new_sl": new_sl, "result": res})
                elif side == "SHORT" and mark_price < entry_price:
                    new_sl = round_to_tick(entry_price, tick_size)
                    res = self.client.create_algo_order(
                        symbol=symbol,
                        side="BUY",
                        order_type="STOP_MARKET",
                        trigger_price=new_sl,
                        close_position=True,
                        client_algo_id=f"be_{int(time.time())}",
                    )
                    updated.append({"symbol": symbol, "side": side, "new_sl": new_sl, "result": res})
                continue

            current_sl = float(sl_order.get("triggerPrice") or sl_order.get("stopPrice") or 0.0)

            # Estimate initial risk from entry and current SL
            risk = abs(entry_price - current_sl)
            if risk <= 0:
                continue

            if side == "LONG":
                gain_r = (mark_price - entry_price) / risk
                # If gain >= breakeven_r and SL is still below entry
                if gain_r >= self.breakeven_r and current_sl < entry_price:
                    # Cancel old SL
                    if is_algo:
                        self.client.cancel_algo_order(symbol, algo_id=sl_order.get("algoId"), client_algo_id=sl_order.get("clientAlgoId"))
                    else:
                        self.client.cancel_order(symbol, order_id=sl_order.get("orderId"))
                    # Place new SL at Entry (Breakeven) via Algo Order API
                    new_sl = round_to_tick(entry_price, tick_size)
                    res = self.client.create_algo_order(
                        symbol=symbol,
                        side="SELL",
                        order_type="STOP_MARKET",
                        trigger_price=new_sl,
                        close_position=True,
                        client_algo_id=f"be_{int(time.time())}",
                    )
                    updated.append({"symbol": symbol, "side": side, "new_sl": new_sl, "result": res})
            else: # SHORT
                gain_r = (entry_price - mark_price) / risk
                if gain_r >= self.breakeven_r and current_sl > entry_price:
                    if is_algo:
                        self.client.cancel_algo_order(symbol, algo_id=sl_order.get("algoId"), client_algo_id=sl_order.get("clientAlgoId"))
                    else:
                        self.client.cancel_order(symbol, order_id=sl_order.get("orderId"))
                    new_sl = round_to_tick(entry_price, tick_size)
                    res = self.client.create_algo_order(
                        symbol=symbol,
                        side="BUY",
                        order_type="STOP_MARKET",
                        trigger_price=new_sl,
                        close_position=True,
                        client_algo_id=f"be_{int(time.time())}",
                    )
                    updated.append({"symbol": symbol, "side": side, "new_sl": new_sl, "result": res})

        return updated

    def sync_close_settled_positions(
        self,
        settled_outcomes: List[Dict[str, Any]],
        executed_records: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check active Binance positions against recently settled trade outcomes.
        If an alert outcome was settled (e.g. time_exit, take_profit, stop_loss)
        and the position is still open on Binance, close it immediately via MARKET order
        to avoid orphan/stuck positions.
        """
        closed_positions = []
        open_positions = self.client.get_positions()
        if not open_positions or not settled_outcomes:
            return closed_positions

        # Build map of settled symbols from recently settled outcomes
        settled_by_symbol = {}
        for outcome in settled_outcomes:
            if not isinstance(outcome, dict):
                continue
            if str(outcome.get("outcome_status") or "").strip().lower() != "settled":
                continue
            raw_sym = str(outcome.get("symbol") or "")
            clean_sym = normalize_futures_symbol(raw_sym)
            if clean_sym:
                # Keep latest outcome
                settled_by_symbol[clean_sym] = outcome

        for pos in open_positions:
            sym = pos.get("symbol")
            if not sym or sym not in settled_by_symbol:
                continue

            outcome = settled_by_symbol[sym]
            exit_reason = str(outcome.get("exit_reason") or "settled")
            pnl_pct = outcome.get("pnl_pct")

            logger.info(
                "[Auto-Trade] Position for %s is settled in outcome tracker (reason=%s). Closing position on Binance...",
                sym, exit_reason
            )

            close_res = self.client.close_position_market(sym)
            if close_res.get("success"):
                logger.info("[Auto-Trade] Successfully closed position for %s on Binance: %s", sym, close_res)
                closed_positions.append({
                    "symbol": sym,
                    "side": pos.get("side"),
                    "amount": pos.get("amount"),
                    "exit_reason": exit_reason,
                    "pnl_pct": pnl_pct,
                    "result": close_res,
                })
            else:
                logger.warning("[Auto-Trade] Failed to close position for %s on Binance: %s", sym, close_res)

        return closed_positions
