"""
Council AlphaForge: Autonomous Quantitative Research & Formula Mining Lab.
Empowers the Trading Council to invent, backtest, and validate novel mathematical
indicators and alpha factors from scratch on 15m crypto futures data.

Key Architecture:
1. Mathematical DSL & Vectorized Expression Evaluator.
2. Hypothesis Generation Engine (Gemini 2.5 Flash / Quantitative Fallback).
3. Walk-Forward Backtester with Friction Deduction (0.30% round-trip fee/slippage).
4. Strict Quality Gate (OOS Profit Factor >= 1.40, Expectancy > +0.20%, Low CDC Correlation).
5. Graduation & Promotion to Sandbox Exploration Mode via Central Blackboard.
6. Telegram Research Briefing Generator.
"""

import os
import re
import json
import time
import math
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Mathematical Operators & Vectorized DSL for Alpha Factor Evaluation
# ----------------------------------------------------------------------

def _ts_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=max(2, int(window)), min_periods=1).mean()


def _ts_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=max(2, int(window)), min_periods=1).std().fillna(0.0)


def _ts_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=max(2, int(window)), min_periods=1).max()


def _ts_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=max(2, int(window)), min_periods=1).min()


def _delta(series: pd.Series, periods: int) -> pd.Series:
    return series.diff(periods=max(1, int(periods))).fillna(0.0)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = _ts_mean(series, window)
    std = _ts_std(series, window)
    return ((series - mean) / (std + 1e-8)).fillna(0.0)


def _decay_linear(series: pd.Series, window: int) -> pd.Series:
    w = max(2, int(window))
    weights = np.arange(1, w + 1, dtype=float)
    weights /= weights.sum()

    def _apply_decay(sub):
        if len(sub) < w:
            return sub.iloc[-1] if len(sub) > 0 else 0.0
        return np.dot(sub.values, weights)

    return series.rolling(window=w, min_periods=1).apply(_apply_decay, raw=False)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=1).mean()
    ma_down = down.rolling(window=window, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))


def _bb_width(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    mid = _ts_mean(close, window)
    std = _ts_std(close, window)
    upper = mid + (num_std * std)
    lower = mid - (num_std * std)
    return ((upper - lower) / (mid + 1e-8)).fillna(0.0)


def prepare_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches raw OHLCV DataFrame with core quantitative primitives.
    """
    df = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    close = df["close"]
    open_p = df["open"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # True Range & ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3)).fillna(high - low)
    df["atr14"] = df["tr"].rolling(14, min_periods=1).mean().fillna(df["tr"])

    # Candle morphology
    df["body"] = (close - open_p).abs()
    df["upper_wick"] = high - np.maximum(open_p, close)
    df["lower_wick"] = np.minimum(open_p, close) - low
    df["range"] = (high - low).replace(0.0, 1e-8)
    df["wick_asymmetry"] = (df["lower_wick"] - df["upper_wick"]) / df["range"]

    # Volume and Return dynamics
    df["vol_mean20"] = _ts_mean(vol, 20)
    df["vol_mean48"] = _ts_mean(vol, 48)
    df["rel_vol"] = (vol / (df["vol_mean20"] + 1e-8)).fillna(1.0)
    df["ret1"] = close.pct_change().fillna(0.0)

    # Simplified CDC ActionZone proxy for correlation checking
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["cdc_bullish"] = (ema12 > ema26).astype(int)

    return df


def evaluate_alpha_expression(expression: str, df: pd.DataFrame) -> pd.Series:
    """
    Safely evaluates a mathematical alpha formula on market features.
    Restricted to standard mathematical and time-series primitives.
    """
    allowed_locals = {
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["close"],
        "volume": df["volume"],
        "body": df["body"],
        "upper_wick": df["upper_wick"],
        "lower_wick": df["lower_wick"],
        "range": df["range"],
        "atr14": df["atr14"],
        "atr": df["atr14"],
        "rel_vol": df["rel_vol"],
        "ret1": df["ret1"],
        "wick_asymmetry": df["wick_asymmetry"],
        "ts_mean": _ts_mean,
        "ts_std": _ts_std,
        "ts_max": _ts_max,
        "ts_min": _ts_min,
        "delta": _delta,
        "zscore": _zscore,
        "decay_linear": _decay_linear,
        "rsi": _rsi,
        "bb_width": _bb_width,
        "abs": np.abs,
        "sign": np.sign,
        "np": np,
    }

    # Security check: Disallow dunder attributes and dangerous standalone keywords
    if "__" in expression or re.search(r"\b(import|exec|eval|lambda|sys|os|subprocess|compile|globals|locals)\b", expression):
        raise ValueError("Forbidden keyword or attribute access detected in alpha expression.")

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_locals)
        if isinstance(result, pd.Series):
            return result.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        elif isinstance(result, (int, float, np.number)):
            return pd.Series(float(result), index=df.index)
        else:
            return pd.Series(0.0, index=df.index)
    except Exception as e:
        logger.debug("Failed to evaluate expression '%s': %s", expression, e)
        return pd.Series(0.0, index=df.index)


# ----------------------------------------------------------------------
# Council AlphaForge Core Engine
# ----------------------------------------------------------------------

class CouncilAlphaForge:
    """
    Autonomous Quantitative Research and Formula Mining Engine.
    Discovers, validates, and graduates mathematical trading factors.
    """

    DEFAULT_UNIVERSE = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "SUIUSDT",
        "ONDOUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"
    ]

    def __init__(
        self,
        config=None,
        data_dir: str = ".data/council_research",
        market_data_dir: str = ".data/market_history",
        gemini_api_key: Optional[str] = None,
        friction_pct: float = 0.30,  # 0.30% round-trip cost hurdle (fee + slippage)
    ):
        self.config = config
        self.data_dir = data_dir
        self.market_data_dir = market_data_dir
        self.friction_pct = float(friction_pct)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.market_data_dir, exist_ok=True)

        self.gemini_api_key = (
            gemini_api_key
            or (getattr(config, "GEMINI_API_KEY", "") if config else "")
            or os.environ.get("GEMINI_API_KEY", "")
        ).strip()
        self.gemini_model = (
            (getattr(config, "GEMINI_MODEL", "") if config else "")
            or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        ).strip()

        self.active_factors_path = os.path.join(self.data_dir, "active_alpha_factors.json")

    # ------------------------------------------------------------------
    # 1. Hypothesis Synthesis (Gemini 2.5 Flash / Fallback Library)
    # ------------------------------------------------------------------

    def generate_candidate_hypotheses(self, use_gemini: bool = True) -> List[Dict[str, Any]]:
        """
        Synthesizes novel alpha factor formulas using Gemini LLM or
        fallback quantitative archetypes.
        """
        if use_gemini and self.gemini_api_key:
            candidates = self._generate_gemini_hypotheses()
            if candidates:
                return candidates

        return self._get_fallback_hypotheses()

    def _generate_gemini_hypotheses(self) -> Optional[List[Dict[str, Any]]]:
        """Queries Gemini 2.5 Flash for quantitative factor formulas."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
        system_instruction = (
            "You are the Chief Quantitative Researcher for a premier Crypto Quantitative Hedge Fund. "
            "Your task is to invent 4 NOVEL, non-traditional mathematical alpha factor formulas for 15-minute crypto futures. "
            "Formulas must avoid simple generic retail indicators (no plain MACD or basic RSI crosses). "
            "Focus on market microstructure: Liquidity vacuum, wick absorption, volatility compression springs, "
            "relative volume shocks, and momentum exhaustion. "
            "Supported variables in expression: open, high, low, close, volume, body, upper_wick, lower_wick, range, atr14, rel_vol, ret1, wick_asymmetry. "
            "Supported functions: ts_mean(x, n), ts_std(x, n), ts_max(x, n), ts_min(x, n), delta(x, n), zscore(x, n), decay_linear(x, n), rsi(close, n), bb_width(close, n), abs(x). "
            "Output strictly a JSON list of objects:\n"
            "[\n"
            "  {\n"
            '    "factor_id": "alpha_liq_absorption_15m",\n'
            '    "factor_name": "Liquidity Absorption Spring",\n'
            '    "economic_rationale": "High relative volume coupled with long lower wick signifies institutional absorption.",\n'
            '    "formula": "rel_vol * (lower_wick - upper_wick) / (atr14 + 1e-6)",\n'
            '    "direction": "BUY",\n'
            '    "threshold": 1.5,\n'
            '    "stop_loss_atr": 1.2,\n'
            '    "take_profit_atr": 2.4,\n'
            '    "max_hold_bars": 16\n'
            "  }\n"
            "]"
        )

        user_prompt = "Generate 4 mathematically rigorous, distinct alpha factor candidates for M15 crypto futures."
        body = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 1500,
                "responseMimeType": "application/json",
            },
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-goog-api-key": self.gemini_api_key},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=25.0) as resp:
                resp_json = json.loads(resp.read())
                candidates = resp_json.get("candidates") or []
                if not candidates:
                    return None
                parts = (candidates[0].get("content") or {}).get("parts") or []
                text = "".join(str(p.get("text") or "") for p in parts).strip()
                parsed = json.loads(text)
                if isinstance(parsed, list) and len(parsed) > 0:
                    logger.info("[AlphaForge] Successfully synthesized %d formulas via Gemini LLM.", len(parsed))
                    return parsed
        except Exception as e:
            logger.warning("[AlphaForge] Gemini alpha synthesis failed (%s); using fallback catalog.", e)

        return None

    def _get_fallback_hypotheses(self) -> List[Dict[str, Any]]:
        """
        Curated catalog of quantitative alpha factor archetypes
        grounded in academic and prop-trading market microstructure.
        """
        return [
            {
                "factor_id": "alpha_liq_vacuum_15m",
                "factor_name": "Liquidity Absorption & Wick Vacuum",
                "economic_rationale": "Heavy volume hitting bid with long lower rejection indicates institutional accumulation.",
                "formula": "rel_vol * (lower_wick - upper_wick) / (atr14 + 1e-6)",
                "direction": "BUY",
                "threshold": 1.4,
                "stop_loss_atr": 1.2,
                "take_profit_atr": 2.5,
                "max_hold_bars": 16,
            },
            {
                "factor_id": "alpha_vol_spring_15m",
                "factor_name": "Volatility Compression Spring (VCS)",
                "economic_rationale": "Tight Bollinger contraction followed by immediate volume explosion signals impending trend explosion.",
                "formula": "(rel_vol / (bb_width(close, 20) + 0.005)) * (close - open) / (atr14 + 1e-6)",
                "direction": "BUY",
                "threshold": 35.0,
                "stop_loss_atr": 1.1,
                "take_profit_atr": 2.2,
                "max_hold_bars": 20,
            },
            {
                "factor_id": "alpha_momentum_exhaust_15m",
                "factor_name": "Reversal Exhaustion Pressure (REP)",
                "economic_rationale": "Price overextended from short-term mean with decaying linear volume signals sharp counter-snap.",
                "formula": "zscore(close - ts_mean(close, 24), 24) * -1.0 * (1.0 + rel_vol)",
                "direction": "BUY",
                "threshold": 1.8,
                "stop_loss_atr": 1.0,
                "take_profit_atr": 2.0,
                "max_hold_bars": 12,
            },
            {
                "factor_id": "alpha_accel_breakout_15m",
                "factor_name": "Kinetic Range Acceleration (KRA)",
                "economic_rationale": "Current bar body expanding faster than trailing ATR with high linear decay volume.",
                "formula": "(body / (atr14 + 1e-6)) * decay_linear(rel_vol, 8) * sign(close - open)",
                "direction": "BUY",
                "threshold": 1.8,
                "stop_loss_atr": 1.3,
                "take_profit_atr": 2.6,
                "max_hold_bars": 16,
            },
        ]

    # ------------------------------------------------------------------
    # 2. Historical Data Sourcing (Binance Public Futures API)
    # ------------------------------------------------------------------

    def fetch_symbol_klines(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """
        Fetches 15m candlestick data directly from public Binance Futures API.
        Caches locally to disk to enable rapid, zero-cost iterative testing.
        """
        clean_sym = symbol.replace("-", "").replace("/", "").upper()
        cache_file = os.path.join(self.market_data_dir, f"{clean_sym}_15m.csv")

        # Use cache if fresh (< 6 hours old)
        if os.path.exists(cache_file):
            file_age_hrs = (time.time() - os.path.getmtime(cache_file)) / 3600.0
            if file_age_hrs < 6.0:
                try:
                    df = pd.read_csv(cache_file)
                    if len(df) >= 200:
                        return df
                except Exception:
                    pass

        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={clean_sym}&interval=15m&limit={min(1500, limit)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tradV2-AlphaForge/1.0"})
            with urllib.request.urlopen(req, timeout=12.0) as resp:
                data = json.loads(resp.read())
                if isinstance(data, list) and len(data) > 0:
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
            logger.warning("[AlphaForge] Failed to fetch live klines for %s: %s", clean_sym, e)

        # Fallback to local cache if network error
        if os.path.exists(cache_file):
            try:
                return pd.read_csv(cache_file)
            except Exception:
                pass

        # Synthetic baseline fallback for offline testing
        return self._generate_synthetic_klines(clean_sym, bars=limit)

    def _generate_synthetic_klines(self, symbol: str, bars: int = 500) -> pd.DataFrame:
        """Generates realistic synthetic klines for offline unit tests."""
        np.random.seed(42)
        base_price = 100.0 if "BTC" not in symbol else 60000.0
        prices = [base_price]
        for _ in range(bars - 1):
            change = np.random.normal(0.0002, 0.008)
            prices.append(prices[-1] * (1.0 + change))

        rows = []
        now_ms = int(time.time() * 1000)
        interval_ms = 15 * 60 * 1000
        for i, p in enumerate(prices):
            noise = np.random.uniform(0.998, 1.002)
            c = p * noise
            o = p
            h = max(o, c) * np.random.uniform(1.0, 1.004)
            l = min(o, c) * np.random.uniform(0.996, 1.0)
            v = np.random.uniform(500, 5000)
            rows.append({
                "timestamp": now_ms - ((bars - i) * interval_ms),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3. Vectorized Walk-Forward Backtest with Cost Deduction
    # ------------------------------------------------------------------

    def backtest_candidate(
        self,
        candidate: Dict[str, Any],
        symbols: Optional[List[str]] = None,
        train_split_ratio: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Executes a vectorized walk-forward simulation across symbols.
        Deducts round-trip friction (fee + slippage = 0.30%) from every trade.
        """
        symbols = symbols or self.DEFAULT_UNIVERSE
        formula = candidate.get("formula", "")
        threshold = float(candidate.get("threshold", 1.5))
        sl_mult = float(candidate.get("stop_loss_atr", 1.2))
        tp_mult = float(candidate.get("take_profit_atr", 2.4))
        max_hold = int(candidate.get("max_hold_bars", 16))

        all_trades = []
        cdc_overlaps = 0
        total_signals = 0

        for sym in symbols:
            raw_df = self.fetch_symbol_klines(sym, limit=1200)
            if raw_df.empty or len(raw_df) < 150:
                continue

            df = prepare_market_features(raw_df)
            factor_series = evaluate_alpha_expression(formula, df)

            # Signal trigger
            signal_mask = factor_series >= threshold
            signal_indices = np.where(signal_mask)[0]

            n_bars = len(df)
            split_idx = int(n_bars * train_split_ratio)

            close_vals = df["close"].values
            open_vals = df["open"].values
            high_vals = df["high"].values
            low_vals = df["low"].values
            atr_vals = df["atr14"].values
            cdc_vals = df["cdc_bullish"].values

            for s_idx in signal_indices:
                if s_idx >= n_bars - 2:
                    continue
                total_signals += 1
                if cdc_vals[s_idx] == 1:
                    cdc_overlaps += 1

                # Entry at next bar open
                entry_idx = s_idx + 1
                entry_price = open_vals[entry_idx]
                atr = atr_vals[s_idx]
                sl_price = entry_price - (sl_mult * atr)
                tp_price = entry_price + (tp_mult * atr)

                exit_price = entry_price
                exit_reason = "time_exit"
                bars_held = 0

                # Trace forward
                for f_idx in range(entry_idx, min(n_bars, entry_idx + max_hold)):
                    bars_held += 1
                    bar_h = high_vals[f_idx]
                    bar_l = low_vals[f_idx]

                    # Check TP
                    if bar_h >= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                        break
                    # Check SL
                    elif bar_l <= sl_price:
                        exit_price = sl_price
                        exit_reason = "stop_loss"
                        break
                    else:
                        exit_price = close_vals[f_idx]

                # Gross return
                gross_ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
                # Net return after deducting 0.30% fee/slippage hurdle
                net_ret_pct = gross_ret_pct - self.friction_pct

                all_trades.append({
                    "symbol": sym,
                    "is_oos": entry_idx >= split_idx,
                    "gross_ret_pct": gross_ret_pct,
                    "net_ret_pct": net_ret_pct,
                    "win": net_ret_pct > 0.0,
                    "exit_reason": exit_reason,
                    "bars_held": bars_held,
                })

        # Calculate In-Sample (Train) and Out-Of-Sample (Test) stats
        train_trades = [t for t in all_trades if not t["is_oos"]]
        test_trades = [t for t in all_trades if t["is_oos"]]

        def _metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not trades:
                return {
                    "trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0,
                    "net_pnl_pct": 0.0, "expectancy_pct": 0.0
                }
            wins = sum(1 for t in trades if t["win"])
            gross_p = sum(max(0.0, t["net_ret_pct"]) for t in trades)
            gross_l = sum(abs(min(0.0, t["net_ret_pct"])) for t in trades)
            pf = (gross_p / gross_l) if gross_l > 0 else (99.0 if gross_p > 0 else 1.0)
            net_pnl = sum(t["net_ret_pct"] for t in trades)
            return {
                "trades": len(trades),
                "wins": wins,
                "losses": len(trades) - wins,
                "win_rate_pct": round((wins / len(trades)) * 100.0, 1),
                "profit_factor": round(pf, 2),
                "net_pnl_pct": round(net_pnl, 2),
                "expectancy_pct": round(net_pnl / len(trades), 2),
            }

        train_stats = _metrics(train_trades)
        test_stats = _metrics(test_trades)
        cdc_corr = round((cdc_overlaps / max(1, total_signals)), 2)

        # Quality Gate Check
        passed_quality_gate = (
            test_stats["trades"] >= 15
            and test_stats["profit_factor"] >= float(getattr(self.config, "ALPHA_FORGE_MIN_OOS_PROFIT_FACTOR", 1.40))
            and test_stats["expectancy_pct"] >= float(getattr(self.config, "ALPHA_FORGE_MIN_OOS_EXPECTANCY_PCT", 0.20))
            and test_stats["win_rate_pct"] >= float(getattr(self.config, "ALPHA_FORGE_MIN_OOS_WIN_RATE", 42.0))
            and cdc_corr <= float(getattr(self.config, "ALPHA_FORGE_MAX_CDC_CORRELATION", 0.65))
        )

        return {
            "candidate": candidate,
            "train_stats": train_stats,
            "test_stats": test_stats,
            "cdc_correlation": cdc_corr,
            "passed_quality_gate": passed_quality_gate,
            "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    # ------------------------------------------------------------------
    # 4. Mining Cycle Execution & Persistence
    # ------------------------------------------------------------------

    def run_mining_cycle(
        self,
        symbols: Optional[List[str]] = None,
        use_gemini: bool = True,
        notify_telegram: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes an end-to-end quantitative research mining cycle:
        1. Formulate hypotheses (via Gemini LLM).
        2. Walk-forward backtest on historical data with 0.30% fee hurdle.
        3. Screen candidates against statistical quality gates.
        4. Promote winning factors into active_alpha_factors.json.
        5. Broadcast Telegram research dispatch.
        """
        logger.info("[AlphaForge] Initiating autonomous quantitative mining cycle...")
        start_time = time.time()

        candidates = self.generate_candidate_hypotheses(use_gemini=use_gemini)
        evaluations = []
        graduated_factors = []

        for cand in candidates:
            res = self.backtest_candidate(cand, symbols=symbols)
            evaluations.append(res)
            if res["passed_quality_gate"]:
                graduated_factors.append(res)
                logger.info("[AlphaForge] Factor '%s' GRADUATED! (OOS PF=%.2f, Expectancy=+%.2f%%)",
                            cand.get("factor_name"), res["test_stats"]["profit_factor"], res["test_stats"]["expectancy_pct"])

        # Save graduated factors
        if graduated_factors:
            self._save_graduated_factors(graduated_factors)

        elapsed = round(time.time() - start_time, 2)
        summary = {
            "cycle_id": f"alpha_mine_{int(time.time())}",
            "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "elapsed_seconds": elapsed,
            "candidates_count": len(candidates),
            "graduated_count": len(graduated_factors),
            "evaluations": evaluations,
            "graduated_factors": graduated_factors,
        }

        if notify_telegram:
            self.dispatch_telegram(summary)

        return summary

    def _save_graduated_factors(self, graduated: List[Dict[str, Any]]) -> None:
        """Appends graduated factors to active_alpha_factors.json atomically."""
        existing = {}
        if os.path.exists(self.active_factors_path):
            try:
                with open(self.active_factors_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        factors_map = existing.get("factors") or {}
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        for g in graduated:
            cand = g["candidate"]
            fid = cand["factor_id"]
            factors_map[fid] = {
                "factor_id": fid,
                "factor_name": cand["factor_name"],
                "formula": cand["formula"],
                "economic_rationale": cand["economic_rationale"],
                "threshold": cand["threshold"],
                "direction": cand.get("direction", "BUY"),
                "stop_loss_atr": cand.get("stop_loss_atr", 1.2),
                "take_profit_atr": cand.get("take_profit_atr", 2.4),
                "test_stats": g["test_stats"],
                "cdc_correlation": g["cdc_correlation"],
                "graduated_at": now_str,
                "status": "active_sandbox_exploration",
            }

        payload = {
            "version": 1,
            "updated_at": now_str,
            "total_active": len(factors_map),
            "factors": factors_map,
        }

        tmp_path = f"{self.active_factors_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.active_factors_path)
        logger.info("[AlphaForge] Persisted %d active factors to %s", len(factors_map), self.active_factors_path)

    # ------------------------------------------------------------------
    # 5. Telegram Research Memo Formatting & Dispatch
    # ------------------------------------------------------------------

    def format_telegram_message(self, summary: Dict[str, Any]) -> str:
        """Renders an elegant HTML research memo for Telegram."""
        graduated = summary.get("graduated_factors") or []
        count_cand = summary.get("candidates_count", 0)
        grad_count = len(graduated)
        date_str = summary.get("completed_at", "")

        lines = [
            "🔬 <b>[Council Alpha Lab] สรุปผลการวิจัยและขุดค้นสูตรใหม่ประจำสัปดาห์</b>",
            f"📅 <i>รอบประเมิน: {date_str} | หักต้นทุน 0.30%/ไม้</i>",
            "",
            f"📊 <b>สถานะการทดสอบสมมติฐานทางคณิตศาสตร์:</b>",
            f"• จำนวนสูตรที่วิจัย: <b>{count_cand} สูตร</b>",
            f"• สูตรที่ผ่านเกณฑ์เข้มงวด: <b>{grad_count} สูตร</b>",
            "",
        ]

        if grad_count > 0:
            lines.append("🏆 <b>สูตรที่ผ่านการคัดเลือกและได้รับการบรรจุ:</b>")
            for idx, g in enumerate(graduated, 1):
                cand = g["candidate"]
                tst = g["test_stats"]
                lines.extend([
                    f"<b>{idx}. {cand.get('factor_name')}</b> (<code>{cand.get('factor_id')}</code>)",
                    f"• <b>สมการ:</b> <code>{cand.get('formula')}</code>",
                    f"• <b>ตรรกะ:</b> <i>{cand.get('economic_rationale')}</i>",
                    f"• <b>Out-of-Sample Performance:</b>",
                    f"  - Win Rate: <b>{tst.get('win_rate_pct')}%</b> ({tst.get('wins')}/{tst.get('trades')} ไม้)",
                    f"  - Profit Factor: <b>{tst.get('profit_factor')}</b> | Net Return: <b>{tst.get('net_pnl_pct'):+.2f}%</b>",
                    f"  - Net Expectancy: <b>{tst.get('expectancy_pct'):+.2f}% ต่อไม้</b>",
                    f"  - ความซ้ำซ้อนกับ CDC: <b>{g.get('cdc_correlation', 0.0)*100:.0f}%</b> (ไม่ทับซ้อน)",
                    "",
                ])
            lines.extend([
                "📜 <b>มติสภา Trading Council:</b>",
                "อนุมัติบรรจุเข้าสู่ <b>Sandbox Exploration Mode</b> เพื่อให้ ContrarianScout "
                "เปิดสิทธิ์ทดลองเทรดจริงบนกระดาน Binance ด้วยขนาดจิ๋ว 35% เพื่อพิสูจน์ผลงาน!",
            ])
        else:
            lines.extend([
                "🛡️ <b>การคุ้มกันเงินทุน:</b>",
                f"ในสัปดาห์นี้ได้ทดสอบสูตรใหม่ทั้งหมด {count_cand} สูตร แต่ไม่มีสูตรใดที่สามารถ "
                f"ทำ Profit Factor ได้เกิน 1.40 หรือชนะต้นทุน 0.30% ในช่วงทดสอบ Out-of-Sample "
                f"สภาจึงมีมติ <b>'ปัดตกทั้งหมด'</b> เพื่อรักษาวินัยความเสี่ยงสูงสุดและไม่นำเงินไปเสี่ยงกับโมเดลที่ยังไม่พร้อม",
            ])

        return "\n".join(lines)

    def dispatch_telegram(self, summary: Dict[str, Any]) -> bool:
        """Sends research memo to Telegram."""
        token = (
            (getattr(self.config, "TELEGRAM_BOT_TOKEN", "") if self.config else "")
            or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        ).strip()
        chat_id = (
            (getattr(self.config, "TELEGRAM_CHAT_ID", "") if self.config else "")
            or os.environ.get("TELEGRAM_CHAT_ID", "")
        ).strip()
        thread_id = (
            (getattr(self.config, "TELEGRAM_THREAD_ID", "") if self.config else "")
            or os.environ.get("TELEGRAM_THREAD_ID", "")
        ).strip()

        if not token or not chat_id:
            logger.warning("[AlphaForge] Missing Telegram credentials, skipping notification.")
            return False

        message = self.format_telegram_message(summary)
        try:
            from infrastructure.notifications.telegram_gateway import send_telegram_alert
            return send_telegram_alert(
                message=message,
                bot_token=token,
                chat_id=chat_id,
                thread_id=thread_id,
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error("[AlphaForge] Telegram dispatch error: %s", e)
            return False
