"""
Scout Researcher (เอเจนต์เลขาฝ่ายวิจัยความสัมพันธ์ข้ามตลาด) for ContrarianScout.
Role: Cross-Market Intelligence Secretary & Inter-Asset Correlation Analyst.

Researches non-obvious relationships, contagion effects, and market anomalies:
1. US Equity Panic Overreaction (e.g. S&P 500 / Nasdaq down 0.4%, but Altcoin dumps 4.0% -> Panic Sweep Bounce).
2. Relative Strength Alpha (e.g. Market bleeds, but ONDO/SUI holds firm with buy volume -> Institutional Accumulation).
3. Safe-Haven Contagion (e.g. Gold / PAXG / DXY surges -> Risk-Off liquidity drain).
4. Short Crowding Pressure (e.g. Extreme negative funding rates -> Squeeze potential).

Packages intelligence briefs for ContrarianScout to form high-conviction asymmetric proposals.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_INTEL_PATH = os.path.join(".data", "macro", "cross_market_intelligence.json")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


class ScoutResearcher:
    """
    Cross-Market Intelligence Secretary for ContrarianScout.
    Monitors inter-market lead-lag dynamics and detects anomalies.
    """

    def __init__(
        self,
        intel_path: Optional[str] = None,
        cache_ttl_seconds: float = 1800.0,  # 30 minutes cache TTL
    ):
        self.intel_path = intel_path or DEFAULT_INTEL_PATH
        self.cache_ttl = float(cache_ttl_seconds)
        self._memory_cache: Optional[Dict[str, Any]] = None
        self._last_fetch_time: float = 0.0

    def load_cached_intel(self) -> Dict[str, Any]:
        """Load saved intelligence brief from disk."""
        if not os.path.exists(self.intel_path):
            return {}
        try:
            with open(self.intel_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug("[ScoutResearcher] Failed to read intel file: %s", e)
            return {}

    def save_intel(self, data: Dict[str, Any]) -> bool:
        """Atomically persist intelligence brief."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.intel_path)), exist_ok=True)
            temp_path = f"{self.intel_path}.tmp_{os.getpid()}_{int(time.time()*1000)}"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.intel_path)
            self._memory_cache = data
            self._last_fetch_time = time.time()
            return True
        except Exception as e:
            logger.error("[ScoutResearcher] Failed to save intel file: %s", e)
            return False

    def fetch_intermarket_snapshot(self, client=None, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Gathers inter-market barometer snapshot:
        - Crypto benchmark: BTCUSDT 24h & 1h price change
        - Safe-haven proxy: PAXGUSDT (Gold on Binance)
        - US Equities & Dollar (via Yahoo Finance or cached/synthetic)
        """
        now_ts = time.time()
        if not force_refresh and self._memory_cache and (now_ts - self._last_fetch_time < self.cache_ttl):
            return self._memory_cache

        cached = self.load_cached_intel()
        cached_ts = float(cached.get("fetched_at_ts") or 0.0)
        if not force_refresh and cached and (now_ts - cached_ts < self.cache_ttl):
            self._memory_cache = cached
            self._last_fetch_time = cached_ts
            return cached

        snapshot: Dict[str, Any] = {
            "timestamp": _format_dt(_utc_now()),
            "fetched_at_ts": now_ts,
            "btc_change_24h": 0.0,
            "paxg_change_24h": 0.0,
            "sp500_change_24h": 0.0,
            "nasdaq_change_24h": 0.0,
            "dxy_change_24h": 0.0,
            "market_regime": "NEUTRAL",
            "risk_sentiment": "NORMAL",
            "barometers": {},
        }

        # 1. Fetch Binance Barometers (BTC & PAXG)
        if client and hasattr(client, "get_24h_ticker"):
            try:
                btc_tick = client.get_24h_ticker("BTCUSDT")
                if btc_tick and "priceChangePercent" in btc_tick:
                    snapshot["btc_change_24h"] = float(btc_tick["priceChangePercent"])
                    snapshot["barometers"]["BTC"] = {
                        "price": float(btc_tick.get("lastPrice", 0.0)),
                        "change_24h": float(btc_tick["priceChangePercent"]),
                        "volume": float(btc_tick.get("volume", 0.0)),
                    }
            except Exception as be:
                logger.debug("[ScoutResearcher] Binance BTC ticker fetch: %s", be)

            try:
                paxg_tick = client.get_24h_ticker("PAXGUSDT")
                if paxg_tick and "priceChangePercent" in paxg_tick:
                    snapshot["paxg_change_24h"] = float(paxg_tick["priceChangePercent"])
                    snapshot["barometers"]["PAXG_GOLD"] = {
                        "price": float(paxg_tick.get("lastPrice", 0.0)),
                        "change_24h": float(paxg_tick["priceChangePercent"]),
                    }
            except Exception as pe:
                logger.debug("[ScoutResearcher] Binance PAXG ticker fetch: %s", pe)

        # 2. Fetch US Equities / Macro from Yahoo Finance (if available, with quick timeout)
        try:
            import yfinance as yf
            tickers = yf.Tickers("^GSPC ^IXIC DX-Y.NYB")
            # S&P 500
            try:
                gspc = tickers.tickers.get("^GSPC")
                if gspc and gspc.fast_info:
                    prev_c = float(gspc.fast_info.get("previous_close") or 0.0)
                    last_p = float(gspc.fast_info.get("last_price") or 0.0)
                    if prev_c > 0 and last_p > 0:
                        snapshot["sp500_change_24h"] = round(((last_p - prev_c) / prev_c) * 100.0, 2)
                        snapshot["barometers"]["SP500"] = {"price": last_p, "change_24h": snapshot["sp500_change_24h"]}
            except Exception:
                pass

            # Nasdaq
            try:
                ixic = tickers.tickers.get("^IXIC")
                if ixic and ixic.fast_info:
                    prev_c = float(ixic.fast_info.get("previous_close") or 0.0)
                    last_p = float(ixic.fast_info.get("last_price") or 0.0)
                    if prev_c > 0 and last_p > 0:
                        snapshot["nasdaq_change_24h"] = round(((last_p - prev_c) / prev_c) * 100.0, 2)
                        snapshot["barometers"]["NASDAQ"] = {"price": last_p, "change_24h": snapshot["nasdaq_change_24h"]}
            except Exception:
                pass

            # DXY Dollar Index
            try:
                dxy = tickers.tickers.get("DX-Y.NYB")
                if dxy and dxy.fast_info:
                    prev_c = float(dxy.fast_info.get("previous_close") or 0.0)
                    last_p = float(dxy.fast_info.get("last_price") or 0.0)
                    if prev_c > 0 and last_p > 0:
                        snapshot["dxy_change_24h"] = round(((last_p - prev_c) / prev_c) * 100.0, 2)
                        snapshot["barometers"]["DXY"] = {"price": last_p, "change_24h": snapshot["dxy_change_24h"]}
            except Exception:
                pass
        except Exception as ye:
            logger.debug("[ScoutResearcher] Yahoo Finance macro fetch skipped: %s", ye)

        # 3. Determine Overall Macro Regime
        btc_chg = snapshot["btc_change_24h"]
        sp_chg = snapshot["sp500_change_24h"]
        paxg_chg = snapshot["paxg_change_24h"]

        if paxg_chg >= 1.5 and btc_chg <= -1.5:
            snapshot["market_regime"] = "FLIGHT_TO_SAFETY"
            snapshot["risk_sentiment"] = "RISK_OFF"
        elif sp_chg >= 0.8 and btc_chg >= 1.0:
            snapshot["market_regime"] = "RISK_ON_EXPANSION"
            snapshot["risk_sentiment"] = "BULLISH"
        elif sp_chg <= -1.0 or btc_chg <= -2.5:
            snapshot["market_regime"] = "MACRO_CONTRACTION"
            snapshot["risk_sentiment"] = "DEFENSIVE"
        else:
            snapshot["market_regime"] = "CALM_CONSOLIDATION"
            snapshot["risk_sentiment"] = "NORMAL"

        self.save_intel(snapshot)
        return snapshot

    def evaluate_correlation(
        self,
        candidate: Dict[str, Any],
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates inter-market correlation and anomalies for a candidate signal.
        Returns anomaly detection status, thesis, and conviction adjustments.
        """
        snap = snapshot or self._memory_cache or self.load_cached_intel()
        symbol = str(candidate.get("symbol") or "").upper()
        clean_sym = symbol.replace("-USD", "").replace("USDT", "")
        signal = str(candidate.get("signal") or "").upper()
        plan = candidate.get("plan") if isinstance(candidate.get("plan"), dict) else {}

        # Extract candidate metrics
        entry_p = float(candidate.get("entry_price") or plan.get("entry_price") or 0.0)
        stop_p = float(candidate.get("stop_loss") or plan.get("stop_loss") or 0.0)
        tp_p = float(candidate.get("take_profit") or plan.get("take_profit") or 0.0)
        risk = abs(entry_p - stop_p) if (entry_p > 0 and stop_p > 0) else 0.0
        reward = abs(tp_p - entry_p) if (tp_p > 0 and entry_p > 0) else 0.0
        rr_ratio = (reward / risk) if risk > 0 else 1.0

        deriv_pulse = candidate.get("derivatives_pulse") or candidate.get("deriv_eval") or {}
        funding_rate = float(deriv_pulse.get("funding_rate") or deriv_pulse.get("last_funding_rate") or 0.0)

        btc_chg = float(
            snap.get("btc_change_24h")
            if snap.get("btc_change_24h") is not None
            else (snap.get("btc_change_24h_pct") or 0.0)
        )
        sp_chg = float(
            snap.get("sp500_change_24h")
            if snap.get("sp500_change_24h") is not None
            else (snap.get("sp500_change_24h_pct") or 0.0)
        )
        paxg_chg = float(
            snap.get("paxg_change_24h")
            if snap.get("paxg_change_24h") is not None
            else (snap.get("paxg_change_24h_pct") or snap.get("gold_paxg_change_24h_pct") or 0.0)
        )

        # Check candidate 24h change if available in item/pulse
        cand_24h = float(
            candidate.get("price_change_24h")
            if candidate.get("price_change_24h") is not None
            else (
                candidate.get("price_change_24h_pct")
                if candidate.get("price_change_24h_pct") is not None
                else (
                    candidate.get("change_24h")
                    if candidate.get("change_24h") is not None
                    else (deriv_pulse.get("price_change_percent") or 0.0)
                )
            )
        )

        anomalies = []
        confidence_boost = 0.0

        # Anomaly 1: Equity & BTC Panic Overreaction (คริปโตทุบเว่อร์เกินเหตุ)
        # S&P/Nasdaq yoyo down slightly (-0.1% to -0.6%), but Altcoin crashed (-3% or more)
        # If candidate has BUY signal with high R:R -> Prime Mean Reversion Bounce!
        if signal == "BUY" and (sp_chg >= -0.8 and sp_chg <= 0.2):
            if cand_24h <= -3.0 or funding_rate <= -0.00010:
                thesis = (
                    f"ตื่นตระหนกเกินจริง (Panic Overreaction): ตลาดหุ้นย่อเพียง {sp_chg:+.2f}% "
                    f"แต่ {clean_sym} ถูกทุบกวาดสภาพคล่องลึก {cand_24h:+.1f}% "
                    f"อัตรา Funding Rate ติดลบ ({funding_rate*100:.3f}%) เข้าข่าย Panic Sweep โอกาสดีดกลับ R:R สูง"
                )
                anomalies.append({
                    "type": "PANIC_OVERREACTION_SWEEP",
                    "thesis": thesis,
                    "target": "BUY",
                })
                confidence_boost += 15.0

        # Anomaly 2: Relative Strength Alpha (เหรียญแกร่งต้านกระแส)
        # Market / BTC is down or flat, but this altcoin is holding support or rising
        if signal == "BUY" and btc_chg <= -0.5:
            if cand_24h >= btc_chg + 2.0:  # Outperforming BTC by at least 2%
                thesis = (
                    f"เหรียญแกร่งต้านกระแส (Relative Strength Alpha Leader): BTC ย่อ {btc_chg:+.1f}% "
                    f"แต่ {clean_sym} แข็งแกร่งทรงตัวบวก ({cand_24h:+.1f}%) "
                    f"มีแรงซื้อสถาบันรองรับ ต้านแรงทุบตลาดใหญ่ พร้อมพุ่งทะยานเมื่อตลาดหยุดลง"
                )
                anomalies.append({
                    "type": "RELATIVE_STRENGTH_ALPHA",
                    "thesis": thesis,
                    "target": "BUY",
                })
                confidence_boost += 15.0

        # Anomaly 3: Safe-Haven Flight Contagion (เงินหนีเข้าทองคำ/ดอลลาร์)
        # Gold/PAXG surges (+1.5%+) while alts bleed
        if paxg_chg >= 1.2:
            if signal == "BUY":
                thesis = (
                    f"เงินทุนไหลเข้าที่หลบภัย (Safe-Haven Flight): ทองคำ/PAXG พุ่ง +{paxg_chg:.1f}% "
                    f"บ่งบอกสภาวะ Risk-Off เงินทุนใน Altcoin กำลังถูกดูดออก ควรระวังการ Long สวนทาง"
                )
                anomalies.append({
                    "type": "SAFE_HAVEN_WARNING",
                    "thesis": thesis,
                    "target": "AVOID_LONG",
                })
                confidence_boost -= 20.0
            elif signal == "SELL":
                thesis = (
                    f"แรงกดดันจาก Safe-Haven Flow: เม็ดเงินไหลเข้าทองคำ (+{paxg_chg:.1f}%) "
                    f"สนับสนุนฝั่ง Short {clean_sym} ตามกระแสเงินทุนโลก"
                )
                anomalies.append({
                    "type": "SAFE_HAVEN_SHORT_CONFIRM",
                    "thesis": thesis,
                    "target": "SELL",
                })
                confidence_boost += 15.0

        # Anomaly 4: Short Squeeze Fuel Loaded
        if signal == "BUY" and funding_rate <= -0.00020:
            thesis = (
                f"มวลชน Short แน่นผิดปกติ (Squeeze Fuel Loaded): Funding Rate ดิ่งลึกถึง {funding_rate*100:.3f}% "
                f"จุด Stop Loss ของฝั่ง Short กองหนาแน่น เจ้ามือพร้อมลากบีบพอร์ตแตก"
            )
            anomalies.append({
                "type": "SHORT_SQUEEZE_FUEL",
                "thesis": thesis,
                "target": "BUY",
            })
            confidence_boost += 20.0

        has_anomaly = len(anomalies) > 0
        primary_thesis = anomalies[0]["thesis"] if has_anomaly else "ไม่พบความผิดปกติข้ามตลาดที่มีนัยสำคัญ"
        primary_type = anomalies[0]["type"] if has_anomaly else None

        return {
            "has_anomaly": has_anomaly,
            "anomaly_type": primary_type,
            "anomaly_summary": primary_thesis,
            "anomalies": anomalies,
            "primary_thesis": primary_thesis,
            "confidence_boost": confidence_boost,
            "macro_regime": snap.get("market_regime", "NEUTRAL"),
            "btc_change_24h": btc_chg,
            "sp500_change_24h": sp_chg,
            "paxg_change_24h": paxg_chg,
        }

    def get_summary_text(self) -> str:
        """Returns concise Thai summary text of inter-market intelligence."""
        snap = self._memory_cache or self.load_cached_intel()
        if not snap:
            return "เลขาฝ่ายวิจัย: ยังไม่มีข้อมูลชีพจรข้ามตลาดล่าสุด"

        regime = snap.get("market_regime", "NEUTRAL")
        sentiment = snap.get("risk_sentiment", "NORMAL")
        btc = snap.get("btc_change_24h", 0.0)
        sp = snap.get("sp500_change_24h", 0.0)
        paxg = snap.get("paxg_change_24h", 0.0)

        return (
            f"เลขาฝ่ายวิจัยความสัมพันธ์ข้ามตลาด: ตลาดภาพรวมอยู่ในโหมด '{regime}' ({sentiment}) "
            f"[BTC {btc:+.1f}% | S&P500 {sp:+.2f}% | ทองคำ/PAXG {paxg:+.1f}%]"
        )
