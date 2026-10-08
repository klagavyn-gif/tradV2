"""
Binance USDT-M Futures REST API Client.
Handles authentication, request signing (HMAC-SHA256), market information,
account balances, position queries, and order execution.
Supports both Live and Testnet environments.
"""

import time
import hmac
import hashlib
import urllib.parse
import logging
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger(__name__)

LIVE_FAPI_URL = "https://fapi.binance.com"
TESTNET_FAPI_URL = "https://testnet.binancefuture.com"


class BinanceFuturesClient:
    """REST Client for Binance USDT-Margined Futures API."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        timeout: float = 10.0,
    ):
        self.api_key = str(api_key or "").strip()
        self.api_secret = str(api_secret or "").strip()
        self.testnet = bool(testnet)
        self.base_url = TESTNET_FAPI_URL if self.testnet else LIVE_FAPI_URL
        self.timeout = float(timeout)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "tradV2-binance-client/1.0",
            "Content-Type": "application/x-www-form-urlencoded",
        })
        if self.api_key:
            self.session.headers["X-MBX-APIKEY"] = self.api_key

        self._exchange_info_cache: Optional[Dict[str, Any]] = None
        self._exchange_info_time: float = 0.0
        self._cache_ttl_seconds: float = 3600.0  # Cache exchangeInfo for 1 hour

    def _sign(self, params: Dict[str, Any]) -> str:
        """Create HMAC-SHA256 signature for parameters."""
        query_string = urllib.parse.urlencode(params)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        """Execute HTTP request with error handling and signing if required."""
        url = f"{self.base_url}{path}"
        params = dict(params or {})

        if signed:
            if not self.api_key or not self.api_secret:
                raise ValueError("API Key and Secret must be provided for signed endpoints")
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            params["signature"] = self._sign(params)

        try:
            if method.upper() == "GET":
                resp = self.session.get(url, params=params, timeout=self.timeout)
            elif method.upper() == "POST":
                resp = self.session.post(url, data=params, timeout=self.timeout)
            elif method.upper() == "DELETE":
                resp = self.session.delete(url, params=params, timeout=self.timeout)
            elif method.upper() == "PUT":
                resp = self.session.put(url, data=params, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            data = resp.json()
            if resp.status_code >= 400:
                code = data.get("code")
                msg = data.get("msg")
                logger.error("Binance API error %s: %s (status %s)", code, msg, resp.status_code)
                return {"success": False, "status_code": resp.status_code, "code": code, "msg": msg, "data": data}
            return {"success": True, "status_code": resp.status_code, "data": data}
        except Exception as e:
            logger.exception("HTTP request to %s failed: %s", path, e)
            return {"success": False, "error": str(e)}

    # --- Public Market Data ---

    def ping(self) -> bool:
        """Ping the server to verify connectivity."""
        res = self._request("GET", "/fapi/v1/ping")
        return bool(res.get("success"))

    def get_server_time(self) -> Optional[int]:
        """Fetch server time in milliseconds."""
        res = self._request("GET", "/fapi/v1/time")
        if res.get("success"):
            return res.get("data", {}).get("serverTime")
        return None

    def get_exchange_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch and cache exchange info for all symbols."""
        now = time.time()
        if not force_refresh and self._exchange_info_cache and (now - self._exchange_info_time < self._cache_ttl_seconds):
            return self._exchange_info_cache

        res = self._request("GET", "/fapi/v1/exchangeInfo")
        if res.get("success"):
            self._exchange_info_cache = res.get("data", {})
            self._exchange_info_time = now
            return self._exchange_info_cache
        return {}

    def get_symbol_filters(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Extract tickSize, stepSize, minQty, and minNotional for a symbol."""
        info = self.get_exchange_info()
        symbols = info.get("symbols", [])
        clean_sym = symbol.replace("-", "").replace("/", "").upper()
        target = next((s for s in symbols if s.get("symbol") == clean_sym), None)
        if not target:
            return None

        filters = {f.get("filterType"): f for f in target.get("filters", [])}
        price_filter = filters.get("PRICE_FILTER", {})
        lot_size_filter = filters.get("LOT_SIZE", {})
        min_notional_filter = filters.get("MIN_NOTIONAL", {})

        return {
            "symbol": clean_sym,
            "status": target.get("status"),
            "baseAsset": target.get("baseAsset"),
            "quoteAsset": target.get("quoteAsset"),
            "pricePrecision": int(target.get("pricePrecision", 2)),
            "quantityPrecision": int(target.get("quantityPrecision", 2)),
            "tickSize": float(price_filter.get("tickSize", 0.01)),
            "stepSize": float(lot_size_filter.get("stepSize", 0.001)),
            "minQty": float(lot_size_filter.get("minQty", 0.001)),
            "minNotional": float(min_notional_filter.get("notional", 5.0)),
        }

    # --- Account & Positions ---

    def get_account_balance(self) -> List[Dict[str, Any]]:
        """Fetch asset balances (e.g. USDT balance, available balance)."""
        res = self._request("GET", "/fapi/v2/balance", signed=True)
        if res.get("success"):
            return res.get("data", [])
        return []

    def get_usdt_balance(self) -> Dict[str, float]:
        """Fetch available and total USDT balance."""
        balances = self.get_account_balance()
        for b in balances:
            if b.get("asset") == "USDT":
                return {
                    "total": float(b.get("balance", 0.0)),
                    "available": float(b.get("availableBalance", 0.0)),
                    "crossUnPnl": float(b.get("crossUnPnl", 0.0)),
                }
        return {"total": 0.0, "available": 0.0, "crossUnPnl": 0.0}

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch position information for a symbol or all symbols."""
        params = {}
        if symbol:
            params["symbol"] = symbol.replace("-", "").replace("/", "").upper()
        res = self._request("GET", "/fapi/v2/positionRisk", params=params, signed=True)
        if not res.get("success"):
            return []
        positions = res.get("data", [])
        # Filter only active open positions (positionAmt != 0)
        open_positions = []
        for p in positions:
            amt = float(p.get("positionAmt", 0.0))
            if abs(amt) > 1e-8:
                open_positions.append({
                    "symbol": p.get("symbol"),
                    "positionAmt": amt,
                    "entryPrice": float(p.get("entryPrice", 0.0)),
                    "markPrice": float(p.get("markPrice", 0.0)),
                    "unRealizedProfit": float(p.get("unRealizedProfit", 0.0)),
                    "liquidationPrice": float(p.get("liquidationPrice", 0.0)),
                    "leverage": int(p.get("leverage", 1)),
                    "marginType": p.get("marginType", "isolated"),
                    "side": "LONG" if amt > 0 else "SHORT",
                })
        return open_positions

    # --- Configuration ---

    def set_leverage(self, symbol: str, leverage: int = 1) -> Dict[str, Any]:
        """Set initial leverage for a symbol (e.g. 1x, 2x)."""
        clean_sym = symbol.replace("-", "").replace("/", "").upper()
        return self._request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": clean_sym, "leverage": int(leverage)},
            signed=True,
        )

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> Dict[str, Any]:
        """Set margin type: ISOLATED or CROSSED."""
        clean_sym = symbol.replace("-", "").replace("/", "").upper()
        return self._request(
            "POST",
            "/fapi/v1/marginType",
            params={"symbol": clean_sym, "marginType": margin_type.upper()},
            signed=True,
        )

    # --- Orders ---

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        callback_rate: Optional[float] = None,
        activation_price: Optional[float] = None,
        reduce_only: bool = False,
        close_position: bool = False,
        client_order_id: Optional[str] = None,
        working_type: str = "MARK_PRICE",
    ) -> Dict[str, Any]:
        """
        Place an order on Binance Futures.
        Supported types: MARKET, LIMIT, STOP_MARKET, TRAILING_STOP_MARKET.
        """
        clean_sym = symbol.replace("-", "").replace("/", "").upper()
        params: Dict[str, Any] = {
            "symbol": clean_sym,
            "side": side.upper(),
            "type": order_type.upper(),
        }

        if quantity is not None:
            params["quantity"] = quantity
        if price is not None:
            params["price"] = price
        if stop_price is not None:
            params["stopPrice"] = stop_price
        if callback_rate is not None:
            params["callbackRate"] = callback_rate
        if activation_price is not None:
            params["activationPrice"] = activation_price
        if reduce_only:
            params["reduceOnly"] = "true"
        if close_position:
            params["closePosition"] = "true"
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        if "STOP" in order_type.upper() or "TRAILING" in order_type.upper():
            params["workingType"] = working_type

        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel a single order by orderId or origClientOrderId."""
        clean_sym = symbol.replace("-", "").replace("/", "").upper()
        params: Dict[str, Any] = {"symbol": clean_sym}
        if order_id:
            params["orderId"] = int(order_id)
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided")

        return self._request("DELETE", "/fapi/v1/order", params=params, signed=True)

    def cancel_all_open_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all open orders for a symbol."""
        clean_sym = symbol.replace("-", "").replace("/", "").upper()
        return self._request("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": clean_sym}, signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders for a symbol or all symbols."""
        params = {}
        if symbol:
            params["symbol"] = symbol.replace("-", "").replace("/", "").upper()
        res = self._request("GET", "/fapi/v1/openOrders", params=params, signed=True)
        if res.get("success"):
            return res.get("data", [])
        return []

    # --- Algo Orders (Mandatory for STOP_MARKET, TRAILING_STOP_MARKET) ---

    def create_algo_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        algo_type: str = "CONDITIONAL",
        trigger_price: Optional[float] = None,
        price: Optional[float] = None,
        quantity: Optional[float] = None,
        callback_rate: Optional[float] = None,
        activation_price: Optional[float] = None,
        reduce_only: bool = False,
        close_position: bool = False,
        client_algo_id: Optional[str] = None,
        working_type: str = "MARK_PRICE",
    ) -> Dict[str, Any]:
        """
        Place an Algorithmic / Conditional order on Binance Futures.
        Supported types: STOP_MARKET, STOP, TAKE_PROFIT_MARKET, TAKE_PROFIT, TRAILING_STOP_MARKET.
        """
        clean_sym = symbol.replace("-", "").replace("/", "").upper()
        params: Dict[str, Any] = {
            "symbol": clean_sym,
            "side": side.upper(),
            "algoType": algo_type.upper(),
            "type": order_type.upper(),
        }

        if trigger_price is not None:
            params["triggerPrice"] = trigger_price
        if price is not None:
            params["price"] = price
        if quantity is not None and not close_position:
            params["quantity"] = quantity
        if callback_rate is not None:
            params["callbackRate"] = callback_rate
        if activation_price is not None:
            params["activationPrice"] = activation_price
        if close_position:
            params["closePosition"] = "true"
        elif reduce_only:
            params["reduceOnly"] = "true"
        if client_algo_id:
            # clientAlgoId max length 32 chars
            params["clientAlgoId"] = str(client_algo_id)[:32]
        if working_type:
            params["workingType"] = working_type

        return self._request("POST", "/fapi/v1/algoOrder", params=params, signed=True)

    def get_open_algo_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open conditional/algo orders for a symbol or all symbols."""
        params = {}
        if symbol:
            params["symbol"] = symbol.replace("-", "").replace("/", "").upper()
        res = self._request("GET", "/fapi/v1/openAlgoOrders", params=params, signed=True)
        if res.get("success"):
            data = res.get("data")
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "orders" in data:
                return data["orders"]
        return []

    def cancel_algo_order(
        self,
        symbol: str,
        algo_id: Optional[int] = None,
        client_algo_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel a single algo order by algoId or clientAlgoId."""
        clean_sym = symbol.replace("-", "").replace("/", "").upper()
        params: Dict[str, Any] = {"symbol": clean_sym}
        if algo_id:
            params["algoId"] = int(algo_id)
        if client_algo_id:
            params["clientAlgoId"] = str(client_algo_id)[:32]
        return self._request("DELETE", "/fapi/v1/algoOrder", params=params, signed=True)

    def cancel_all_algo_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all open algo orders for a symbol."""
        clean_sym = symbol.replace("-", "").replace("/", "").upper()
        return self._request("DELETE", "/fapi/v1/algoOpenOrders", params={"symbol": clean_sym}, signed=True)

    def close_position_market(self, symbol: str) -> Dict[str, Any]:
        """
        Immediately close an open position on Binance Futures with a MARKET order
        and cancel any pending open orders and algo orders for this symbol.
        """
        clean_sym = symbol.replace("-", "").replace("/", "").upper()
        positions = self.get_positions()
        target_pos = next((p for p in positions if p.get("symbol") == clean_sym), None)
        if not target_pos:
            return {"success": False, "reason": "no_open_position", "symbol": clean_sym}

        side = str(target_pos.get("side", "")).upper()
        close_side = "SELL" if side == "LONG" else "BUY"
        qty = abs(float(target_pos.get("amount", 0.0)))
        if qty <= 0:
            return {"success": False, "reason": "position_amount_zero", "symbol": clean_sym}

        # 1. Cancel open orders and open algo orders
        try:
            self.cancel_all_open_orders(clean_sym)
        except Exception:
            pass
        try:
            self.cancel_all_algo_orders(clean_sym)
        except Exception:
            pass

        # 2. Place market order to close position
        return self.create_order(
            symbol=clean_sym,
            side=close_side,
            order_type="MARKET",
            quantity=qty,
            reduce_only=True,
        )
