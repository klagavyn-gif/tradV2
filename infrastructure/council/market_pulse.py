"""
Market Pulse & Session Clock Engine for tradV2.
Provides real-time cross-asset intelligence (BTC lead-lag, Gold/PAXG flight-to-safety,
ONDO institutional flow) and international trading session awareness (Asia, London, NY Open,
Opening surge liquidity window).
"""

import os
import time
import json
import logging
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketSessionClock:
    """
    Tracks global financial market trading sessions and liquidity surge windows.
    Sessions in UTC:
      - ASIA      : 00:00 - 07:00 UTC (07:00 - 14:00 Thai)
      - LONDON    : 07:00 - 13:30 UTC (14:00 - 20:30 Thai)
      - NY_OPEN   : 13:30 - 15:30 UTC (20:30 - 22:30 Thai) [High volatility & fakeout surge]
      - NY_MIDDAY : 15:30 - 21:00 UTC (22:30 - 04:00 Thai)
      - OFF_HOURS : 21:00 - 24:00 UTC (04:00 - 07:00 Thai)
    """

    @staticmethod
    def get_current_session(dt: Optional[datetime] = None) -> Dict[str, Any]:
        """Classifies the market session for a given UTC datetime."""
        now = dt or datetime.now(timezone.utc)
        hour = now.hour
        minute = now.minute
        time_decimal = hour + minute / 60.0

        session_name = "OFF_HOURS"
        is_opening_surge = False
        volatility_bias = 1.0
        session_label_th = "ช่วงนอกเวลาตลาดหลัก"

        if 0.0 <= time_decimal < 7.0:
            session_name = "ASIA"
            session_label_th = "ตลาดเอเชีย (Tokyo/HK)"
            volatility_bias = 0.85
            # First 45 mins of Asia open
            if time_decimal < 0.75:
                is_opening_surge = True
        elif 7.0 <= time_decimal < 13.5:
            session_name = "LONDON"
            session_label_th = "ตลาดยุโรป (London Open)"
            volatility_bias = 1.15
            # First 60 mins of London open (07:00 - 08:00 UTC)
            if time_decimal < 8.0:
                is_opening_surge = True
        elif 13.5 <= time_decimal < 15.5:
            session_name = "NY_OPEN"
            session_label_th = "ตลาดอเมริกาเปิด (Wall Street Surge)"
            volatility_bias = 1.35
            # First 60 mins of US Equity Cash Open (13:30 - 14:30 UTC)
            if time_decimal < 14.5:
                is_opening_surge = True
        elif 15.5 <= time_decimal < 21.0:
            session_name = "NY_MIDDAY"
            session_label_th = "ตลาดอเมริกากลางวัน (Wall Street Afternoon)"
            volatility_bias = 1.10
        else:
            session_name = "OFF_HOURS"
            session_label_th = "ช่วงปิดตลาดสหรัฐฯ"
            volatility_bias = 0.80

        is_weekend = now.weekday() >= 5
        if is_weekend:
            volatility_bias *= 0.75
            session_label_th += " [วันหยุดสุดสัปดาห์ / วอลุ่มเบาบาง]"

        return {
            "session": session_name,
            "session_label_th": session_label_th,
            "utc_time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "hour_utc": hour,
            "is_opening_surge": is_opening_surge,
            "volatility_bias": volatility_bias,
            "is_weekend": is_weekend,
        }


class CrossAssetPulseEngine:
    """
    Monitors lead-lag dynamics and macroeconomic cross-asset signals across:
    1. BTCUSDT (Crypto Benchmark Leader)
    2. PAXGUSDT (Digital Gold / Flight-to-Safety)
    3. ONDOUSDT (Real World Assets / Institutional US Treasuries)
    """

    BENCHMARKS = ["BTCUSDT", "PAXGUSDT", "ONDOUSDT"]

    def __init__(self, cache_dir: str = ".data/market_pulse"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _normalize_futures_sym(symbol: str) -> str:
        text = str(symbol or "").strip().upper()
        if text.endswith("-USD") or text.endswith("/USD"):
            return text[:-4] + "USDT"
        elif text.endswith("/USDT") or text.endswith("-USDT"):
            return text.replace("/", "").replace("-", "")
        elif not text.endswith("USDT"):
            return f"{text}USDT"
        return text

    def fetch_klines(self, symbol: str, limit: int = 40) -> pd.DataFrame:
        """Fetches 15m klines from Binance Futures API."""
        clean_sym = self._normalize_futures_sym(symbol)

        cache_file = os.path.join(self.cache_dir, f"{clean_sym}_pulse.csv")
        # Use cache if < 5 minutes old
        if os.path.exists(cache_file):
            if (time.time() - os.path.getmtime(cache_file)) < 300.0:
                try:
                    df = pd.read_csv(cache_file)
                    if len(df) >= 10:
                        return df
                except Exception:
                    pass

        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={clean_sym}&interval=15m&limit={limit}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tradV2-PulseEngine/1.0"})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read())
                rows = []
                for k in data:
                    rows.append({
                        "timestamp": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    })
                df = pd.DataFrame(rows)
                df.to_csv(cache_file, index=False)
                return df
        except Exception as e:
            logger.warning("[MarketPulse] Failed to fetch live klines for %s: %s", clean_sym, e)
            if os.path.exists(cache_file):
                try:
                    return pd.read_csv(cache_file)
                except Exception:
                    pass
            # Fallback dummy dataframe
            return pd.DataFrame({"close": [100.0] * limit, "volume": [10.0] * limit})

    def get_cross_asset_pulse(self) -> Dict[str, Any]:
        """
        Calculates cross-market returns, flight-to-safety ratio, and macro bias.
        """
        btc_df = self.fetch_klines("BTCUSDT", limit=20)
        paxg_df = self.fetch_klines("PAXGUSDT", limit=20)
        ondo_df = self.fetch_klines("ONDOUSDT", limit=20)

        def _calc_ret(df: pd.DataFrame, bars: int = 4) -> float:
            if len(df) <= bars:
                return 0.0
            p_now = df["close"].iloc[-1]
            p_prev = df["close"].iloc[-1 - bars]
            if p_prev <= 0:
                return 0.0
            return (p_now - p_prev) / p_prev * 100.0

        btc_15m_ret = _calc_ret(btc_df, bars=1)
        btc_1h_ret = _calc_ret(btc_df, bars=4)
        btc_4h_ret = _calc_ret(btc_df, bars=16)

        paxg_1h_ret = _calc_ret(paxg_df, bars=4)
        ondo_1h_ret = _calc_ret(ondo_df, bars=4)

        # Flight-to-safety: Gold moving up while BTC moving down
        is_flight_to_safety = (paxg_1h_ret >= 0.25 and btc_1h_ret <= -0.25)
        # Risk-on frenzy: BTC and ONDO both rallying strongly
        is_risk_on_frenzy = (btc_1h_ret >= 1.0 and ondo_1h_ret >= 0.8)

        macro_regime = "NEUTRAL"
        macro_regime_th = "ภาวะตลาดสมดุลทั่วไป"
        if is_flight_to_safety:
            macro_regime = "FLIGHT_TO_SAFETY"
            macro_regime_th = "🛡️ เงินไหลหนีความเสี่ยงเข้าทองคำ (PAXG ขึ้น / คริปโตย่อ)"
        elif is_risk_on_frenzy:
            macro_regime = "RISK_ON_FRENZY"
            macro_regime_th = "🚀 เงินสถาบันโหมความเสี่ยง (BTC & RWA พุ่งพร้อมกัน)"
        elif btc_1h_ret <= -1.2:
            macro_regime = "MARKET_DUMP"
            macro_regime_th = "⚠️ BTC ทุบนำตลาดอย่างรวดเร็ว"

        return {
            "btc_15m_ret": round(btc_15m_ret, 3),
            "btc_1h_ret": round(btc_1h_ret, 3),
            "btc_4h_ret": round(btc_4h_ret, 3),
            "paxg_1h_ret": round(paxg_1h_ret, 3),
            "ondo_1h_ret": round(ondo_1h_ret, 3),
            "is_flight_to_safety": is_flight_to_safety,
            "is_risk_on_frenzy": is_risk_on_frenzy,
            "macro_regime": macro_regime,
            "macro_regime_th": macro_regime_th,
        }

    def evaluate_symbol_lead_lag(self, symbol: str, signal: str) -> Dict[str, Any]:
        """
        Evaluates lead-lag alignment between a specific symbol candidate and the BTC benchmark leader.
        """
        pulse = self.get_cross_asset_pulse()
        session_info = MarketSessionClock.get_current_session()

        clean_sym = self._normalize_futures_sym(symbol)
        alt_df = self.fetch_klines(clean_sym, limit=20)
        p_now = alt_df["close"].iloc[-1] if len(alt_df) > 0 else 1.0
        p_1h = alt_df["close"].iloc[-5] if len(alt_df) >= 5 else p_now
        alt_1h_ret = ((p_now - p_1h) / p_1h * 100.0) if p_1h > 0 else 0.0

        btc_1h_ret = pulse["btc_1h_ret"]
        lag_spread = round(btc_1h_ret - alt_1h_ret, 3)

        # Alignment logic
        is_aligned = True
        warning = None
        score_adj = 0.0

        sig = signal.upper()
        if sig == "BUY":
            if pulse["is_flight_to_safety"]:
                is_aligned = False
                warning = "ทองคำ (PAXG) พุ่งสวนทาง BTC บ่งชี้ความกลัวสถาบัน (Risk-Off)"
                score_adj -= 10.0
            elif btc_1h_ret <= -1.0:
                is_aligned = False
                warning = f"BTC 1h ทุบลงแรง ({btc_1h_ret:+.2f}%) อาจดึงเหรียญ {symbol} ร่วงตาม"
                score_adj -= 12.0
            elif lag_spread >= 1.5:
                # BTC leads by 1.5%+, Alt lagging with upside catch-up edge!
                warning = f"🚀 [Catch-up Lead] BTC พุ่งนำหน้า {lag_spread:+.2f}% เหรียญมีโอกาสตาม"
                score_adj += 5.0
        elif sig == "SELL":
            if btc_1h_ret >= 1.2:
                is_aligned = False
                warning = f"BTC 1h พุ่งขึ้นแรง ({btc_1h_ret:+.2f}%) ฝืนเปิด Short เสี่ยงโดนลาก"
                score_adj -= 12.0

        if session_info["is_opening_surge"]:
            warning = (warning + " | " if warning else "") + "⚡ ระวังช่วงเปิดตลาดใหญ่ 60 นาทีแรก (Liquidity Spike)"
            score_adj -= 4.0

        return {
            "symbol": symbol,
            "signal": signal,
            "alt_1h_ret": round(alt_1h_ret, 3),
            "btc_1h_ret": btc_1h_ret,
            "lag_spread": lag_spread,
            "is_aligned": is_aligned,
            "warning": warning,
            "score_adjustment": score_adj,
            "session": session_info,
            "pulse": pulse,
        }
