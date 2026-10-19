"""
tradV2 — Council Quantum Lab (Extreme Quantitative Research & Intermarket Alpha Crucible)
========================================================================================
An isolated, parallel shadow simulation laboratory designed to invent and test
radical, non-linear, and intermarket cross-asset trading expressions.

Guiding Principles:
1. 100% Shadow Simulation: NEVER touches live exchange capital or real margin budgets.
2. Extreme Exploration Space: Cross-asset ratios (PAXG/BTC, ETH/BTC, SOL/ETH),
   financial physics (kinetic momentum, entropy, volatility pinch, wick absorption).
3. Scientific Friction: Every trade is strictly penalized with a 0.30% round-trip fee & slippage.
4. Detailed Error Taxonomy: Analyzes failed trades (MAE drag, MFE decay, regime mismatch).
5. 4-Week Graduation Gate: Extreme factors that prove persistent edge (PF >= 1.60, Net Exp >= +0.35%)
   graduate into the Trading Council for production consideration.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe Mathematical DSL Evaluator for Radical Expressions
# ---------------------------------------------------------------------------

FORBIDDEN_KEYWORDS = {
    "import", "exec", "eval", "lambda", "sys", "os", "subprocess",
    "shutil", "open", "compile", "globals", "locals", "builtins",
    "getattr", "setattr", "delattr", "class", "def", "__",
}


def _validate_safe_expression(expression: str) -> None:
    """Ensure math expression contains only safe statistical and mathematical tokens."""
    lowered = expression.lower()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            raise ValueError(f"Forbidden token '{kw}' in expression: {expression}")
    if "__" in lowered:
        raise ValueError(f"Dunder access prohibited in expression: {expression}")


def evaluate_quantum_expression(expression: str, df: pd.DataFrame) -> pd.Series:
    """Evaluate a mathematical expression vectorially over the enriched feature DataFrame."""
    _validate_safe_expression(expression)

    # Primitive time-series and non-linear operators
    def ts_mean(s: pd.Series, period: int) -> pd.Series:
        return s.rolling(window=int(period), min_periods=1).mean()

    def ts_std(s: pd.Series, period: int) -> pd.Series:
        return s.rolling(window=int(period), min_periods=1).std().fillna(0.0)

    def ts_max(s: pd.Series, period: int) -> pd.Series:
        return s.rolling(window=int(period), min_periods=1).max()

    def ts_min(s: pd.Series, period: int) -> pd.Series:
        return s.rolling(window=int(period), min_periods=1).min()

    def ts_lag(s: pd.Series, period: int) -> pd.Series:
        return s.shift(int(period)).bfill()

    def ts_sum(s: pd.Series, period: int) -> pd.Series:
        return s.rolling(window=int(period), min_periods=1).sum()

    def delta(s: pd.Series, period: int = 1) -> pd.Series:
        return s.diff(int(period)).fillna(0.0)

    def zscore(s: pd.Series, period: int = 20) -> pd.Series:
        m = ts_mean(s, period)
        sd = ts_std(s, period)
        return ((s - m) / (sd + 1e-9)).fillna(0.0)

    def decay_linear(s: pd.Series, period: int) -> pd.Series:
        p = int(period)
        if p <= 1:
            return s
        weights = np.arange(1, p + 1, dtype=float)
        weights /= weights.sum()
        return s.rolling(window=p, min_periods=1).apply(
            lambda x: np.dot(x[-len(weights):], weights[-len(x):]) / weights[-len(x):].sum(),
            raw=True,
        ).fillna(0.0)

    def clamp(s: pd.Series, min_val: float, max_val: float) -> pd.Series:
        return s.clip(lower=min_val, upper=max_val)

    def sigmoid(s: pd.Series) -> pd.Series:
        return 1.0 / (1.0 + np.exp(-s.clip(-20, 20)))

    safe_env: Dict[str, Any] = {
        "__builtins__": {},
        "np": np,
        "pd": pd,
        "abs": np.abs,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "sign": np.sign,
        "sqrt": np.sqrt,
        "log": lambda s: np.log(np.maximum(s, 1e-9)),
        "ts_mean": ts_mean,
        "ts_std": ts_std,
        "ts_max": ts_max,
        "ts_min": ts_min,
        "ts_lag": ts_lag,
        "ts_sum": ts_sum,
        "delta": delta,
        "zscore": zscore,
        "decay_linear": decay_linear,
        "clamp": clamp,
        "sigmoid": sigmoid,
    }

    # Inject DataFrame columns
    for col in df.columns:
        safe_env[col] = df[col]

    result = eval(expression, safe_env)  # pylint: disable=eval-used
    if isinstance(result, (int, float)):
        result = pd.Series(float(result), index=df.index)
    elif not isinstance(result, pd.Series):
        result = pd.Series(result, index=df.index)
    return result.fillna(0.0)


# ---------------------------------------------------------------------------
# Intermarket & Non-Linear Feature Enricher
# ---------------------------------------------------------------------------

def enrich_quantum_features(
    df: pd.DataFrame,
    cross_dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Enrich candle DataFrame with standard, non-linear, and intermarket cross-asset features."""
    out = df.copy()

    # Ensure required candle columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            for alt in [col.capitalize(), col.upper()]:
                if alt in out.columns:
                    out[col] = out[alt]
                    break
        if col not in out.columns:
            raise KeyError(f"Required price column '{col}' missing from DataFrame")

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_p = out["open"].astype(float)
    volume = out["volume"].astype(float)

    # 1. Base Morphology
    body = (close - open_p).abs()
    upper_wick = high - np.maximum(close, open_p)
    lower_wick = np.minimum(close, open_p) - low
    out["body"] = body
    out["upper_wick"] = upper_wick
    out["lower_wick"] = lower_wick

    # 2. True Range & ATR(14)
    prev_close = close.shift(1).bfill()
    tr = np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )
    atr14 = tr.rolling(window=14, min_periods=1).mean()
    out["atr14"] = atr14
    out["atr_pct"] = (atr14 / close) * 100.0

    # 3. Microstructure & Wick Asymmetry
    out["vol_skew"] = (upper_wick - lower_wick) / (atr14 + 1e-9)
    out["wick_absorption"] = lower_wick / (body + 1e-9)
    out["upper_absorption"] = upper_wick / (body + 1e-9)

    # 4. Bollinger Band & Volatility Compression
    sma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std().fillna(0.0)
    out["bb_width"] = (std20 * 4.0) / (sma20 + 1e-9)
    out["fractal_pinch"] = out["bb_width"] / ((atr14 / close) + 1e-9)

    # 5. Non-linear Kinetic Momentum & Volume Flow
    vol_mean20 = volume.rolling(20, min_periods=1).mean()
    rel_vol = volume / (vol_mean20 + 1e-9)
    out["rel_vol"] = rel_vol

    # Kinetic Momentum: rel_vol * (delta_3 / atr14)^2 with direction
    d3 = close.diff(3).fillna(0.0)
    d3_norm = d3 / (atr14 + 1e-9)
    out["kinetic_energy"] = rel_vol * (d3_norm ** 2) * np.sign(d3)

    # Efficiency Ratio (Kaufman)
    direction_10 = (close - close.shift(10).bfill()).abs()
    volatility_10 = (close - close.shift(1).bfill()).abs().rolling(10, min_periods=1).sum()
    out["efficiency_ratio"] = direction_10 / (volatility_10 + 1e-9)

    # 6. Intermarket Cross-Asset Ratios (Synthetic pairs)
    cross_dfs = cross_dfs or {}
    for cross_name, c_df in cross_dfs.items():
        if "close" in c_df.columns:
            c_close = c_df["close"].reindex(out.index).ffill().bfill()
            if cross_name == "PAXG-USD":
                out["paxg_btc_ratio"] = (c_close / close).fillna(0.0)
                out["paxg_divergence"] = (
                    ((c_close.pct_change(12) - close.pct_change(12)) * 100.0).fillna(0.0)
                )
            elif cross_name == "ETH-USD":
                out["eth_btc_ratio"] = (c_close / close).fillna(0.0)
            elif cross_name == "BTC-USD":
                out["asset_btc_ratio"] = (close / c_close).fillna(0.0)
                out["btc_mom_diff"] = (
                    ((close.pct_change(12) - c_close.pct_change(12)) * 100.0).fillna(0.0)
                )

    # Fallback placeholders if cross-assets not provided
    for ph in ["paxg_btc_ratio", "paxg_divergence", "eth_btc_ratio", "asset_btc_ratio", "btc_mom_diff"]:
        if ph not in out.columns:
            out[ph] = pd.Series(0.0, index=out.index)

    return out


# ---------------------------------------------------------------------------
# Data Models for Quantum Alpha Experiments
# ---------------------------------------------------------------------------

@dataclass
class QuantumTradeOutcome:
    """Log of a simulated paper trade under an extreme factor."""
    trade_id: str
    factor_name: str
    symbol: str
    entry_bar: int
    exit_bar: int
    direction: str  # BUY or SELL
    entry_price: float
    exit_price: float
    gross_return_pct: float
    net_return_pct: float  # After 0.30% round-trip friction
    realized_r: float
    mfe_r: float
    mae_r: float
    exit_reason: str  # TAKE_PROFIT, STOP_LOSS, BREAKEVEN, TIME_EXIT
    market_regime: str  # CHOP, TRENDING, BTC_DUMP, VOLATILE


@dataclass
class QuantumFactorEvaluation:
    """Evaluation summary for a radical quantitative factor."""
    factor_name: str
    hypothesis: str
    entry_expression: str
    exit_expression: str
    direction: str
    total_trades: int
    wins: int
    losses: int
    flats: int
    win_rate_pct: float
    profit_factor: float
    net_expectancy_pct: float
    avg_reward_risk: float
    max_drawdown_pct: float
    is_graduated: bool
    is_promising: bool
    error_taxonomy: Dict[str, Any]
    trade_outcomes: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Quantum Lab Simulation Engine
# ---------------------------------------------------------------------------

class QuantumLabEngine:
    """
    Parallel Shadow Sandbox for extreme quant research.
    Runs simulated backtests on live candle data with fee deductions,
    analyzes error taxonomies, and tracks factor graduation over cycles.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        friction_pct: float = 0.30,
        min_graduation_weeks: int = 4,
        min_graduation_pf: float = 1.60,
        min_graduation_exp_pct: float = 0.35,
    ):
        self.data_dir = data_dir or Path(".data/quantum_lab")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.friction_pct = float(friction_pct)
        self.min_graduation_weeks = int(min_graduation_weeks)
        self.min_graduation_pf = float(min_graduation_pf)
        self.min_graduation_exp_pct = float(min_graduation_exp_pct)
        self.graduated_factors_path = self.data_dir / "graduated_quantum_alphas.json"
        self.lifetime_tracker_path = self.data_dir / "quantum_lifetime_tracker.json"

    def backtest_extreme_factor(
        self,
        factor_name: str,
        hypothesis: str,
        entry_expr: str,
        exit_expr: str,
        df: pd.DataFrame,
        symbol: str = "BTC-USD",
        direction: str = "BUY",
        max_holding_bars: int = 48,
        stop_atr_mult: float = 1.5,
        target_r: float = 2.0,
    ) -> QuantumFactorEvaluation:
        """
        Simulate paper trades for a radical mathematical expression.
        Includes 0.30% roundtrip friction, Breakeven Stop @ 1.2R, Trailing Stop @ 2.0R.
        """
        df_rich = enrich_quantum_features(df)
        entry_sig = evaluate_quantum_expression(entry_expr, df_rich)
        exit_sig = evaluate_quantum_expression(exit_expr, df_rich)

        close = df_rich["close"].values
        high = df_rich["high"].values
        low = df_rich["low"].values
        atr = df_rich["atr14"].values
        n_bars = len(df_rich)

        trades: List[QuantumTradeOutcome] = []
        in_pos = False
        entry_idx = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        risk_dist = 0.0
        max_favorable = 0.0
        max_adverse = 0.0

        for i in range(20, n_bars - 1):
            if not in_pos:
                # Check Entry Signal
                if entry_sig.iloc[i] > 0.5:
                    in_pos = True
                    entry_idx = i
                    entry_price = float(close[i])
                    cur_atr = float(atr[i]) if atr[i] > 0 else entry_price * 0.015
                    risk_dist = cur_atr * stop_atr_mult

                    if direction.upper() == "BUY":
                        stop_loss = entry_price - risk_dist
                        take_profit = entry_price + (risk_dist * target_r)
                    else:
                        stop_loss = entry_price + risk_dist
                        take_profit = entry_price - (risk_dist * target_r)

                    max_favorable = 0.0
                    max_adverse = 0.0
            else:
                # Manage active simulated position
                cur_high = float(high[i])
                cur_low = float(low[i])
                cur_close = float(close[i])
                holding_bars = i - entry_idx

                # Compute MFE and MAE in R-multiples
                if direction.upper() == "BUY":
                    fav_dist = max(0.0, cur_high - entry_price)
                    adv_dist = max(0.0, entry_price - cur_low)
                else:
                    fav_dist = max(0.0, entry_price - cur_low)
                    adv_dist = max(0.0, cur_high - entry_price)

                cur_mfe_r = fav_dist / (risk_dist + 1e-9)
                cur_mae_r = adv_dist / (risk_dist + 1e-9)
                max_favorable = max(max_favorable, cur_mfe_r)
                max_adverse = max(max_adverse, cur_mae_r)

                # Dynamic Breakeven Stop Adjustment (at +1.2R)
                effective_sl = stop_loss
                if max_favorable >= 1.2:
                    # Move stop to entry + 0.10R buffer
                    effective_sl = (
                        entry_price + (0.10 * risk_dist)
                        if direction.upper() == "BUY"
                        else entry_price - (0.10 * risk_dist)
                    )

                # Trailing Stop Adjustment (at +2.0R)
                if max_favorable >= 2.0:
                    trailing_dist = 0.8 * risk_dist
                    if direction.upper() == "BUY":
                        ts_price = cur_high - trailing_dist
                        effective_sl = max(effective_sl, ts_price)
                    else:
                        ts_price = cur_low + trailing_dist
                        effective_sl = min(effective_sl, ts_price)

                # Exit checks
                closed = False
                exit_price = cur_close
                exit_reason = "TIME_EXIT"

                if direction.upper() == "BUY":
                    if cur_low <= effective_sl:
                        exit_price = effective_sl
                        exit_reason = "STOP_LOSS" if max_favorable < 1.2 else "BREAKEVEN"
                        closed = True
                    elif cur_high >= take_profit:
                        exit_price = take_profit
                        exit_reason = "TAKE_PROFIT"
                        closed = True
                else:
                    if cur_high >= effective_sl:
                        exit_price = effective_sl
                        exit_reason = "STOP_LOSS" if max_favorable < 1.2 else "BREAKEVEN"
                        closed = True
                    elif cur_low <= take_profit:
                        exit_price = take_profit
                        exit_reason = "TAKE_PROFIT"
                        closed = True

                # Custom Exit Expression Trigger
                if not closed and exit_sig.iloc[i] > 0.5:
                    exit_price = cur_close
                    exit_reason = "SIGNAL_EXIT"
                    closed = True

                # Time Exit
                if not closed and holding_bars >= max_holding_bars:
                    exit_price = cur_close
                    exit_reason = "TIME_EXIT"
                    closed = True

                if closed:
                    in_pos = False
                    if direction.upper() == "BUY":
                        gross_ret = ((exit_price - entry_price) / entry_price) * 100.0
                    else:
                        gross_ret = ((entry_price - exit_price) / entry_price) * 100.0

                    net_ret = gross_ret - self.friction_pct
                    realized_r = (net_ret / ((risk_dist / entry_price) * 100.0))

                    # Classify market regime during trade
                    regime = "NORMAL"
                    if max_adverse >= 1.2 and max_favorable < 0.5:
                        regime = "FALSE_BREAKOUT"
                    elif holding_bars >= max_holding_bars and abs(net_ret) < 0.4:
                        regime = "LOW_VOL_CHOP"

                    trades.append(
                        QuantumTradeOutcome(
                            trade_id=f"{symbol}_{entry_idx}_{i}",
                            factor_name=factor_name,
                            symbol=symbol,
                            entry_bar=entry_idx,
                            exit_bar=i,
                            direction=direction,
                            entry_price=round(entry_price, 4),
                            exit_price=round(exit_price, 4),
                            gross_return_pct=round(gross_ret, 3),
                            net_return_pct=round(net_ret, 3),
                            realized_r=round(realized_r, 3),
                            mfe_r=round(max_favorable, 2),
                            mae_r=round(max_adverse, 2),
                            exit_reason=exit_reason,
                            market_regime=regime,
                        )
                    )

        return self._compile_evaluation(factor_name, hypothesis, entry_expr, exit_expr, direction, trades)

    def _compile_evaluation(
        self,
        factor_name: str,
        hypothesis: str,
        entry_expr: str,
        exit_expr: str,
        direction: str,
        trades: List[QuantumTradeOutcome],
    ) -> QuantumFactorEvaluation:
        """Compile simulation trades into a comprehensive evaluation with detailed error taxonomy."""
        total_trades = len(trades)
        if total_trades == 0:
            return QuantumFactorEvaluation(
                factor_name=factor_name,
                hypothesis=hypothesis,
                entry_expression=entry_expr,
                exit_expression=exit_expr,
                direction=direction,
                total_trades=0,
                wins=0,
                losses=0,
                flats=0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                net_expectancy_pct=0.0,
                avg_reward_risk=0.0,
                max_drawdown_pct=0.0,
                is_graduated=False,
                is_promising=False,
                error_taxonomy={"error": "no_trades_generated"},
                trade_outcomes=[],
            )

        # Ensure all items in trades are QuantumTradeOutcome instances
        normalized_trades: List[QuantumTradeOutcome] = []
        for t in trades:
            if isinstance(t, QuantumTradeOutcome):
                normalized_trades.append(t)
            elif isinstance(t, dict):
                normalized_trades.append(QuantumTradeOutcome(**t))
            else:
                normalized_trades.append(t)
        trades = normalized_trades

        wins = [t for t in trades if t.net_return_pct > 0.05]
        losses = [t for t in trades if t.net_return_pct < -0.05]
        flats = [t for t in trades if abs(t.net_return_pct) <= 0.05]

        gross_gains = sum(t.net_return_pct for t in wins)
        gross_losses = abs(sum(t.net_return_pct for t in losses))
        pf = (gross_gains / gross_losses) if gross_losses > 0 else 999.0
        win_rate = (len(wins) / total_trades) * 100.0
        net_exp = sum(t.net_return_pct for t in trades) / total_trades
        avg_rr = sum(t.realized_r for t in trades) / total_trades

        # Drawdown curve
        equity_curve = [100.0]
        cur_eq = 100.0
        for t in trades:
            cur_eq *= (1.0 + (t.net_return_pct / 100.0))
            equity_curve.append(cur_eq)
        eq_series = pd.Series(equity_curve)
        cummax = eq_series.cummax()
        dd = (eq_series - cummax) / cummax * 100.0
        max_dd = abs(float(dd.min()))

        # -------------------------------------------------------------
        # Detailed Error Taxonomy (Root-Cause Failure Analysis)
        # -------------------------------------------------------------
        false_breakouts = len([t for t in losses if t.market_regime == "FALSE_BREAKOUT"])
        chop_bleed = len([t for t in losses if t.market_regime == "LOW_VOL_CHOP"])
        mfe_decay_trades = len([t for t in losses if t.mfe_r >= 1.0])  # Had >= 1R profit but turned into loss!
        deep_mae_trades = len([t for t in losses if t.mae_r >= 1.5])   # Immediate deep flush

        error_taxonomy = {
            "total_losses": len(losses),
            "false_breakouts": false_breakouts,
            "false_breakout_pct": round((false_breakouts / len(losses)) * 100.0, 1) if losses else 0.0,
            "chop_bleed_losses": chop_bleed,
            "profit_decay_losses": mfe_decay_trades,  # Trades that had profit but bled out
            "deep_mae_flush_losses": deep_mae_trades,
            "avg_loss_holding_bars": round(
                sum(t.exit_bar - t.entry_bar for t in losses) / len(losses), 1
            ) if losses else 0.0,
            "primary_failure_mode": (
                "PROFIT_DECAY" if mfe_decay_trades >= len(losses) * 0.4 and losses
                else ("FALSE_BREAKOUT" if false_breakouts >= len(losses) * 0.4 and losses
                else ("CHOP_BLEED" if chop_bleed >= len(losses) * 0.4 and losses else "GENERAL_NOISE"))
            ),
        }

        is_promising = bool(pf >= 1.35 and net_exp >= 0.15 and total_trades >= 5)
        is_graduated = bool(pf >= self.min_graduation_pf and net_exp >= self.min_graduation_exp_pct and total_trades >= 15)

        return QuantumFactorEvaluation(
            factor_name=factor_name,
            hypothesis=hypothesis,
            entry_expression=entry_expr,
            exit_expression=exit_expr,
            direction=direction,
            total_trades=total_trades,
            wins=len(wins),
            losses=len(losses),
            flats=len(flats),
            win_rate_pct=round(win_rate, 1),
            profit_factor=round(pf, 2),
            net_expectancy_pct=round(net_exp, 3),
            avg_reward_risk=round(avg_rr, 2),
            max_drawdown_pct=round(max_dd, 2),
            is_graduated=is_graduated,
            is_promising=is_promising,
            error_taxonomy=error_taxonomy,
            trade_outcomes=[asdict(t) for t in trades],
        )

    def track_and_graduate(self, evaluations: List[QuantumFactorEvaluation]) -> List[Dict[str, Any]]:
        """Track multi-week persistent edge across cycles and graduate elite factors."""
        tracker: Dict[str, Any] = {}
        if self.lifetime_tracker_path.exists():
            try:
                tracker = json.loads(self.lifetime_tracker_path.read_text(encoding="utf-8"))
            except Exception:
                tracker = {}

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        graduated_list: List[Dict[str, Any]] = []

        for ev in evaluations:
            fn = ev.factor_name
            f_data = tracker.get(fn, {
                "factor_name": fn,
                "hypothesis": ev.hypothesis,
                "entry_expression": ev.entry_expression,
                "exit_expression": ev.exit_expression,
                "direction": ev.direction,
                "consecutive_passing_weeks": 0,
                "total_evaluations": 0,
                "lifetime_trades": 0,
                "lifetime_profit_factor": ev.profit_factor,
                "lifetime_expectancy": ev.net_expectancy_pct,
                "history": [],
            })

            f_data["total_evaluations"] += 1
            f_data["lifetime_trades"] += ev.total_trades
            f_data["history"].append({
                "date": now_str,
                "pf": ev.profit_factor,
                "exp": ev.net_expectancy_pct,
                "win_rate": ev.win_rate_pct,
                "trades": ev.total_trades,
            })

            if ev.profit_factor >= self.min_graduation_pf and ev.net_expectancy_pct >= self.min_graduation_exp_pct:
                f_data["consecutive_passing_weeks"] += 1
            else:
                f_data["consecutive_passing_weeks"] = 0

            # Graduation criterion: >= min_graduation_weeks OR super-alpha in large sample
            if (f_data["consecutive_passing_weeks"] >= self.min_graduation_weeks) or (
                ev.total_trades >= 30 and ev.profit_factor >= 1.70 and ev.net_expectancy_pct >= 0.40
            ):
                f_data["graduated_at"] = now_str
                f_data["status"] = "GRADUATED_TO_COUNCIL"
                graduated_list.append(f_data)

            tracker[fn] = f_data

        # Persist lifetime tracker
        self.lifetime_tracker_path.write_text(json.dumps(tracker, indent=2, ensure_ascii=False), encoding="utf-8")

        # Persist graduated list
        if graduated_list:
            existing_grad: List[Dict[str, Any]] = []
            if self.graduated_factors_path.exists():
                try:
                    existing_grad = json.loads(self.graduated_factors_path.read_text(encoding="utf-8"))
                except Exception:
                    existing_grad = []
            
            # Merge
            known_names = {g["factor_name"] for g in existing_grad}
            for g in graduated_list:
                if g["factor_name"] not in known_names:
                    existing_grad.append(g)
                    known_names.add(g["factor_name"])

            self.graduated_factors_path.write_text(
                json.dumps(existing_grad, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        return graduated_list
