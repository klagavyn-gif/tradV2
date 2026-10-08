"""
Binance USDT-M Futures Execution Package.
Provides client and order manager for automated execution,
risk controls, Breakeven stop and Trailing Stop management.
"""

from .client import BinanceFuturesClient
from .order_manager import BinanceFuturesOrderManager, normalize_futures_symbol, round_to_step, round_to_tick
from .pipeline_hook import execute_binance_auto_trade_pipeline
from .risk_manager import BinanceFuturesRiskManager
from .watchdog import BinanceFuturesWatchdog

__all__ = [
    "BinanceFuturesClient",
    "BinanceFuturesOrderManager",
    "BinanceFuturesRiskManager",
    "BinanceFuturesWatchdog",
    "normalize_futures_symbol",
    "round_to_step",
    "round_to_tick",
    "execute_binance_auto_trade_pipeline",
]
