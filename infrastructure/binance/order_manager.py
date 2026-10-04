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
        trade_notional_usdt: float = 10.0,
        max_positions: int = 2,
        leverage: int = 1,
        margin_type: str = "ISOLATED",
        breakeven_r: float = 1.2,
        trailing_r: float = 2.0,
        trailing_dist_r: float = 0.8,
    ):
        self.client = client
        self.trade_notional_usdt = float(trade_notional_usdt)
        self.max_positions = int(max_positions)
        self.leverage = int(leverage)
        self.margin_type = str(margin_type or "ISOLATED").upper()
        self.breakeven_r = float(breakeven_r)
        self.trailing_r = float(trailing_r)
        self.trailing_dist_r = float(trailing_dist_r)

    def execute_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an entry candidate on Binance Futures with Stop Loss and Trailing Stop.
        Performs pre-flight checks: balance, existing positions, max concurrent positions.
        """
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

        # Scale order notional to meet symbol's min_notional requirement
        order_notional = max(self.trade_notional_usdt, min_notional)
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

        # 5. Place Protective Stop Loss (STOP_MARKET with closePosition=True)
        opposite_side = "SELL" if signal == "BUY" else "BUY"
        sl_client_id = f"trad_{alert_id}_sl"
        clean_sl_price = round_to_tick(stop_loss, tick_size)

        sl_res = self.client.create_order(
            symbol=symbol,
            side=opposite_side,
            order_type="STOP_MARKET",
            stop_price=clean_sl_price,
            close_position=True,
            client_order_id=sl_client_id,
        )

        # 6. Place Trailing Stop Order (TRAILING_STOP_MARKET)
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

            ts_res = self.client.create_order(
                symbol=symbol,
                side=opposite_side,
                order_type="TRAILING_STOP_MARKET",
                quantity=quantity,
                activation_price=clean_activation,
                callback_rate=callback_pct,
                reduce_only=True,
                client_order_id=ts_client_id,
            )

        return {
            "success": True,
            "symbol": symbol,
            "signal": signal,
            "quantity": quantity,
            "entry_price": exec_price,
            "stop_loss": clean_sl_price,
            "notional": quantity * exec_price,
            "entry_order": entry_data,
            "sl_order": sl_res.get("data") if sl_res else None,
            "ts_order": ts_res.get("data") if ts_res else None,
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

            # Query existing open orders for symbol
            orders = self.client.get_open_orders(symbol)
            sl_order = next((o for o in orders if o.get("type") == "STOP_MARKET"), None)
            if not sl_order:
                continue

            current_sl = float(sl_order.get("stopPrice", 0.0))
            filters = self.client.get_symbol_filters(symbol)
            tick_size = filters["tickSize"] if filters else 0.01

            # Estimate initial risk from entry and current SL
            risk = abs(entry_price - current_sl)
            if risk <= 0:
                continue

            if side == "LONG":
                gain_r = (mark_price - entry_price) / risk
                # If gain >= breakeven_r and SL is still below entry
                if gain_r >= self.breakeven_r and current_sl < entry_price:
                    # Cancel old SL
                    self.client.cancel_order(symbol, order_id=sl_order["orderId"])
                    # Place new SL at Entry (Breakeven)
                    new_sl = round_to_tick(entry_price, tick_size)
                    res = self.client.create_order(
                        symbol=symbol,
                        side="SELL",
                        order_type="STOP_MARKET",
                        stop_price=new_sl,
                        close_position=True,
                        client_order_id=f"be_{int(time.time())}",
                    )
                    updated.append({"symbol": symbol, "side": side, "new_sl": new_sl, "result": res})
            else: # SHORT
                gain_r = (entry_price - mark_price) / risk
                if gain_r >= self.breakeven_r and current_sl > entry_price:
                    self.client.cancel_order(symbol, order_id=sl_order["orderId"])
                    new_sl = round_to_tick(entry_price, tick_size)
                    res = self.client.create_order(
                        symbol=symbol,
                        side="BUY",
                        order_type="STOP_MARKET",
                        stop_price=new_sl,
                        close_position=True,
                        client_order_id=f"be_{int(time.time())}",
                    )
                    updated.append({"symbol": symbol, "side": side, "new_sl": new_sl, "result": res})

        return updated
