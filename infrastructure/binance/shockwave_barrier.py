"""
BTC Shockwave Barrier (Cross-Asset Volatility Protection).
Protects Altcoins in the trading universe from systemic liquidation cascades
and volatility contagion triggered by violent Bitcoin price moves on M15.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class BtcShockwaveBarrier:
    """
    Evaluates BTC 15m market volatility in real time to detect systemic shocks.
    When BTC experiences an abnormal expansion bar (e.g. >= 1.8x ATR or >= 1.5% move),
    all Altcoin entry orders are temporarily vetoed to protect capital.
    """

    def __init__(self, default_atr_mult: float = 1.8, default_abs_pct: float = 1.5):
        self.default_atr_mult = float(default_atr_mult)
        self.default_abs_pct = float(default_abs_pct)

    @staticmethod
    def is_altcoin(symbol: str) -> bool:
        """Return True if symbol is an Altcoin (not BTC)."""
        clean = str(symbol or "").upper().replace("-", "").replace("/", "").replace("_", "")
        return not clean.startswith("BTC")

    def calculate_atr(self, klines: List[List[Any]], period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range from Binance klines.
        Each kline: [open_time, open, high, low, close, volume, ...]
        """
        if not klines or len(klines) < period + 1:
            return None

        tr_values = []
        for i in range(1, len(klines)):
            high = float(klines[i][2])
            low = float(klines[i][3])
            prev_close = float(klines[i - 1][4])

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        if len(tr_values) < period:
            return None

        recent_tr = tr_values[-period:]
        return sum(recent_tr) / len(recent_tr)

    def evaluate(
        self,
        client=None,
        klines: Optional[List[List[Any]]] = None,
        atr_mult: Optional[float] = None,
        abs_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate BTC 15m klines to check if a Shockwave is active.
        """
        mult_threshold = float(atr_mult if atr_mult is not None else self.default_atr_mult)
        pct_threshold = float(abs_pct if abs_pct is not None else self.default_abs_pct)

        if klines is None and client is not None:
            try:
                klines = client.get_klines("BTCUSDT", interval="15m", limit=30)
            except Exception as e:
                logger.warning("Could not fetch BTCUSDT klines for Shockwave Barrier: %s", e)
                klines = None

        if not klines or len(klines) < 16:
            return {
                "shock_active": False,
                "reason": "insufficient_btc_data",
                "badge": "⚡ [BTC Pulse] ข้อมูลไม่พอ — อนุญาตตามปกติ",
                "range_atr": 1.0,
                "abs_change_pct": 0.0,
            }

        # Calculate 14-period ATR excluding the latest ongoing bar
        atr_14 = self.calculate_atr(klines[:-1], period=14)
        if not atr_14 or atr_14 <= 0:
            return {
                "shock_active": False,
                "reason": "invalid_atr",
                "badge": "⚡ [BTC Pulse] คำนวณ ATR ไม่ได้",
                "range_atr": 1.0,
                "abs_change_pct": 0.0,
            }

        # Analyze latest bar (index -1)
        latest_bar = klines[-1]
        l_open = float(latest_bar[1])
        l_high = float(latest_bar[2])
        l_low = float(latest_bar[3])
        l_close = float(latest_bar[4])

        l_range = l_high - l_low
        l_range_mult = l_range / atr_14 if atr_14 > 0 else 1.0
        l_abs_change = (abs(l_close - l_open) / l_open) * 100.0 if l_open > 0 else 0.0
        direction = "DUMP 🔴" if l_close < l_open else "SPIKE 🟢"

        # Also inspect previous closed bar (index -2) for reverberating shockwaves
        prev_bar = klines[-2]
        p_open = float(prev_bar[1])
        p_high = float(prev_bar[2])
        p_low = float(prev_bar[3])
        p_close = float(prev_bar[4])

        p_range = p_high - p_low
        p_range_mult = p_range / atr_14 if atr_14 > 0 else 1.0
        p_abs_change = (abs(p_close - p_open) / p_open) * 100.0 if p_open > 0 else 0.0

        is_shock_latest = (l_range_mult >= mult_threshold) or (l_abs_change >= pct_threshold)
        is_shock_prev = (p_range_mult >= mult_threshold * 1.15) or (p_abs_change >= pct_threshold * 1.15)

        shock_active = is_shock_latest or is_shock_prev
        active_range_mult = l_range_mult if is_shock_latest else p_range_mult
        active_abs_change = l_abs_change if is_shock_latest else p_abs_change
        shock_bar_desc = "แท่งปัจจุบัน" if is_shock_latest else "แท่งก่อนหน้า (คลื่นกระแทกตกค้าง)"

        if shock_active:
            badge = (
                f"⚡ <b>[BTC Shockwave Barrier ACTIVE]</b>\n"
                f"   • BTC ผันผวนรุนแรง {shock_bar_desc}: Range {active_range_mult:.2f}x ATR | ขยับ {active_abs_change:.2f}% ({direction})\n"
                f"   • สถานะ: ระงับเปิดไม้ Altcoins ทั้งหมดเพื่อป้องกัน Liquidation Cascade"
            )
            reason = f"btc_shockwave_active_{direction.lower()}"
        else:
            badge = f"🟢 [BTC Pulse Normal] Range {l_range_mult:.2f}x ATR ({l_abs_change:.2f}%) — ตลาดสม่ำเสมอ"
            reason = "normal"

        return {
            "shock_active": shock_active,
            "reason": reason,
            "badge": badge,
            "range_atr": round(active_range_mult, 2),
            "abs_change_pct": round(active_abs_change, 2),
            "direction": direction,
            "atr_14": round(atr_14, 2),
        }

    def should_veto_candidate(self, candidate: Dict[str, Any], shock_eval: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if candidate trade should be vetoed due to BTC Shockwave.
        Altcoins are strictly vetoed. BTC itself is permitted if aligned.
        """
        if not shock_eval.get("shock_active"):
            return False, None

        sym = str(candidate.get("symbol") or "")
        if self.is_altcoin(sym):
            return True, f"btc_shockwave_spillover_{shock_eval.get('range_atr')}x_atr"

        return False, None
