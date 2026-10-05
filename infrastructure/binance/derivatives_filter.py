"""
Binance USDT-M Futures Derivatives Alpha Filter (Funding Rate & Open Interest).

Provides real-time positioning and sentiment intelligence by evaluating:
1. Funding Rate (FR): Identifies crowded long/short leverage bubbles.
   - High positive FR (> +0.04% / 8h): Crowded Long -> High risk of Long Liquidation flush.
   - Negative FR (< -0.03% / 8h): Crowded Short -> High risk of Short Squeeze.
2. Open Interest (OI) 1h Momentum:
   - Price Breakout + Rising OI (> +1.0%): Confirmed by fresh institutional capital.
   - Price Breakout + Falling OI (< -2.0%): False breakout / Short Covering trap.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
import requests

logger = logging.getLogger(__name__)

FAPI_PUBLIC_URL = "https://fapi.binance.com"


def normalize_futures_symbol(raw_symbol: str) -> str:
    """Map ticker into Binance USDT-M Futures format."""
    s = str(raw_symbol or "").strip().upper()
    if not s:
        return ""
    s = s.replace("-USD", "USDT").replace("/USD", "USDT")
    s = s.replace("-USDT", "USDT").replace("/USDT", "USDT")
    if not s.endswith("USDT") and not s.endswith("BUSD"):
        s = f"{s}USDT"
    return s


class BinanceDerivativesFilter:
    """
    Evaluates market positioning and sentiment via Binance Public Derivatives API.
    Operates with zero API keys required and minimal rate weight.
    """

    def __init__(self, timeout: float = 4.0):
        self.timeout = float(timeout)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "tradV2-derivatives-filter/1.0",
        })
        self._funding_cache: Optional[Dict[str, float]] = None
        self._funding_cache_time: float = 0.0
        self._funding_cache_ttl: float = 180.0  # 3 minutes

        self._oi_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._oi_cache_ttl: float = 180.0

    def get_all_funding_rates(self, force_refresh: bool = False) -> Dict[str, float]:
        """Fetch current funding rates for all USDT-M Futures symbols in 1 request."""
        now = time.time()
        if not force_refresh and self._funding_cache and (now - self._funding_cache_time < self._funding_cache_ttl):
            return self._funding_cache

        try:
            url = f"{FAPI_PUBLIC_URL}/fapi/v1/premiumIndex"
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    rates = {}
                    for item in data:
                        sym = str(item.get("symbol") or "")
                        rate_val = item.get("lastFundingRate")
                        if sym and rate_val is not None:
                            try:
                                rates[sym] = float(rate_val)
                            except Exception:
                                pass
                    self._funding_cache = rates
                    self._funding_cache_time = now
                    return rates
            logger.warning("Binance premiumIndex request failed: status=%s", resp.status_code)
        except Exception as e:
            logger.warning("Could not fetch funding rates from Binance: %s", e)

        return self._funding_cache or {}

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get funding rate for a specific symbol (e.g. BTC-USD -> BTCUSDT)."""
        clean_sym = normalize_futures_symbol(symbol)
        rates = self.get_all_funding_rates()
        return rates.get(clean_sym)

    def get_open_interest_stats(self, symbol: str, bars: int = 4) -> Dict[str, Any]:
        """
        Fetch historical Open Interest on 15m timeframe and calculate momentum.
        bars=4 gives 1-hour lookback on 15m timeframe.
        """
        clean_sym = normalize_futures_symbol(symbol)
        now = time.time()

        if clean_sym in self._oi_cache:
            ts, cached_val = self._oi_cache[clean_sym]
            if now - ts < self._oi_cache_ttl:
                return cached_val

        default_res = {
            "symbol": clean_sym,
            "available": False,
            "latest_oi": 0.0,
            "prev_oi": 0.0,
            "delta_pct": 0.0,
            "trend": "UNAVAILABLE",
        }

        try:
            url = f"{FAPI_PUBLIC_URL}/futures/data/openInterestHist"
            params = {
                "symbol": clean_sym,
                "period": "15m",
                "limit": max(2, bars),
            }
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) >= 2:
                    latest = float(data[-1].get("sumOpenInterest") or 0.0)
                    prev = float(data[0].get("sumOpenInterest") or 0.0)
                    if prev > 0:
                        delta_pct = ((latest - prev) / prev) * 100.0
                    else:
                        delta_pct = 0.0

                    if delta_pct >= 1.0:
                        trend = "EXPANDING"
                    elif delta_pct <= -1.0:
                        trend = "CONTRACTING"
                    else:
                        trend = "NEUTRAL"

                    res = {
                        "symbol": clean_sym,
                        "available": True,
                        "latest_oi": latest,
                        "prev_oi": prev,
                        "delta_pct": round(delta_pct, 2),
                        "trend": trend,
                    }
                    self._oi_cache[clean_sym] = (now, res)
                    return res
        except Exception as e:
            logger.warning("Could not fetch open interest history for %s: %s", clean_sym, e)

        return default_res

    def evaluate_candidate(
        self,
        candidate: Dict[str, Any],
        *,
        max_long_funding: float = 0.0004,   # +0.04%
        min_short_funding: float = -0.0003, # -0.03%
        oi_lookback_bars: int = 4,
        oi_confirm_min_pct: float = 1.0,
        oi_contract_max_pct: float = -2.0,
        veto_enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate entry candidate against positioning and leverage filters.
        Returns evaluation dict with approval status, warnings, and formatting badge.
        """
        raw_symbol = str(candidate.get("symbol") or "")
        signal = str(candidate.get("signal") or "").strip().upper()
        clean_sym = normalize_futures_symbol(raw_symbol)

        fr = self.get_funding_rate(raw_symbol)
        oi_stats = self.get_open_interest_stats(raw_symbol, bars=oi_lookback_bars)

        approved = True
        veto_reason = None
        warning_flags = []
        conf_modifier = 0.0

        fr_pct = (fr * 100.0) if fr is not None else 0.0
        oi_delta = float(oi_stats.get("delta_pct") or 0.0)
        oi_avail = bool(oi_stats.get("available"))

        # 1. Funding Rate Checks
        if fr is not None:
            if signal == "BUY" and fr >= max_long_funding:
                warning_flags.append(f"high_funding_rate_{fr_pct:+.3f}%")
                if veto_enabled:
                    approved = False
                    veto_reason = f"crowded_long_funding_{fr_pct:+.3f}%"
            elif signal == "SELL" and fr <= min_short_funding:
                warning_flags.append(f"negative_funding_rate_{fr_pct:+.3f}%")
                if veto_enabled:
                    approved = False
                    veto_reason = f"crowded_short_funding_{fr_pct:+.3f}%"

        # 2. Open Interest Momentum Checks
        if oi_avail:
            if signal == "BUY":
                if oi_delta >= oi_confirm_min_pct:
                    conf_modifier += 2.0  # High conviction fresh capital inflow
                elif oi_delta <= oi_contract_max_pct:
                    warning_flags.append(f"fakeout_oi_drop_{oi_delta:+.2f}%")
                    conf_modifier -= 4.0
                    # If extreme contraction (< -3.5%), veto buy fakeout
                    if veto_enabled and oi_delta <= -3.5:
                        approved = False
                        veto_reason = f"fakeout_liquidation_oi_drop_{oi_delta:+.2f}%"
            elif signal == "SELL":
                if oi_delta >= oi_confirm_min_pct:
                    conf_modifier += 2.0  # Strong short expansion
                elif oi_delta <= oi_contract_max_pct:
                    warning_flags.append(f"short_covering_oi_drop_{oi_delta:+.2f}%")
                    conf_modifier -= 4.0

        # Construct readable summary badge for Telegram and reports
        fr_str = f"{fr_pct:+.3f}%" if fr is not None else "N/A"
        oi_str = f"{oi_delta:+.1f}%" if oi_avail else "N/A"

        if oi_delta >= oi_confirm_min_pct:
            oi_label = "🔥 เงินใหม่หนุน"
        elif oi_delta <= oi_contract_max_pct:
            oi_label = "⚠️ เงินไหลออก"
        else:
            oi_label = "⚖️ ทรงตัว"

        badge = f"📊 <b>Derivatives Pulse:</b> FR <code>{fr_str}</code> | OI 1h <code>{oi_str}</code> ({oi_label})"

        return {
            "approved": approved,
            "veto_reason": veto_reason,
            "symbol": clean_sym,
            "signal": signal,
            "funding_rate": fr,
            "funding_rate_pct": fr_pct,
            "oi_delta_pct": oi_delta,
            "oi_available": oi_avail,
            "oi_trend": oi_stats.get("trend"),
            "confidence_modifier": conf_modifier,
            "warning_flags": warning_flags,
            "badge": badge,
        }
