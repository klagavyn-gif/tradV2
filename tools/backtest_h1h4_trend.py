#!/usr/bin/env python
"""Phase 1 research: backtest simple H4 trend-following signals vs benchmark.

Signals (all on H4, no lookahead, entry/exit at next bar open, cost deducted):
  1. Donchian 20-bar breakout (long/short), exit on opposite 10-bar break.
  2. EMA 50/200 crossover (long/short).
  3. EMA 50/200 bias + Donchian 10-bar entry + ATR(14) trailing stop.

Cost model: fee 0.10%/side + slippage 0.05%/side = 0.30% round trip.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WATCHLIST = [
    "BTC-USD", "ETH-USD", "DOGE-USD", "ADA-USD", "XRP-USD", "BNB-USD",
    "SOL-USD", "TRX-USD", "NEAR-USD", "LINK-USD", "PAXG-USD",
]

COST_PER_SIDE = 0.0015  # 0.15% per side = 0.30% round trip


def _find_cache(symbol, interval):
    pattern = f".data/market_history/binance/{symbol}_{interval}_adj_binance_*.csv"
    files = glob.glob(str(ROOT / pattern))
    if not files:
        raise FileNotFoundError(f"missing {symbol} {interval} cache")
    return files[0]


def load_h4(symbol):
    df = pd.read_csv(_find_cache(symbol, "1h"))
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample("4h").agg(agg).dropna()


def _atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def signal_donchian(df, entry_n=20, exit_n=10):
    """Position series on bar close: 1 long, -1 short, 0 flat."""
    high_entry = df["High"].shift(1).rolling(entry_n).max()
    low_entry = df["Low"].shift(1).rolling(entry_n).min()
    high_exit = df["High"].shift(1).rolling(exit_n).max()
    low_exit = df["Low"].shift(1).rolling(exit_n).min()
    pos = pd.Series(0, index=df.index)
    state = 0
    for i in range(len(df)):
        close = df["Close"].iloc[i]
        if state == 0:
            if close > high_entry.iloc[i]:
                state = 1
            elif close < low_entry.iloc[i]:
                state = -1
        elif state == 1:
            if close < low_exit.iloc[i]:
                state = 0
        elif state == -1:
            if close > high_exit.iloc[i]:
                state = 0
        pos.iloc[i] = state
    return pos


def signal_ema_cross(df, fast=50, slow=200):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    pos = pd.Series(0, index=df.index)
    pos[ema_fast > ema_slow] = 1
    pos[ema_fast < ema_slow] = -1
    return pos


def signal_trend_breakout(df, fast=50, slow=200, entry_n=10, atr_period=14, trail_mult=3.0):
    """EMA bias + Donchian entry + ATR trailing stop (iterative state machine)."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    atr = _atr(df, atr_period)
    high_entry = df["High"].shift(1).rolling(entry_n).max()
    low_entry = df["Low"].shift(1).rolling(entry_n).min()
    pos = pd.Series(0, index=df.index)
    state = 0
    entry_price = None
    stop = None
    for i in range(len(df)):
        close = df["Close"].iloc[i]
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        bias = 1 if ema_fast.iloc[i] > ema_slow.iloc[i] else -1 if ema_fast.iloc[i] < ema_slow.iloc[i] else 0
        if state == 0:
            if bias == 1 and close > high_entry.iloc[i]:
                state = 1
                entry_price = close
                stop = close - trail_mult * atr.iloc[i]
            elif bias == -1 and close < low_entry.iloc[i]:
                state = -1
                entry_price = close
                stop = close + trail_mult * atr.iloc[i]
        elif state == 1:
            stop = max(stop, close - trail_mult * atr.iloc[i])
            if close < stop or bias == -1:
                state = 0
        elif state == -1:
            stop = min(stop, close + trail_mult * atr.iloc[i])
            if close > stop or bias == 1:
                state = 0
        pos.iloc[i] = state
    return pos


def _trade_returns(df, pos):
    """Convert a bar-close position series into per-trade returns (entry/exit at next open)."""
    executed = pos.shift(1).fillna(0)  # decided at close t-1, executed at open t
    trades = []
    state = 0
    entry = None
    for i in range(1, len(df)):
        target = int(executed.iloc[i])
        if target == state:
            continue
        price = float(df["Open"].iloc[i])
        if state == 0 and target in (1, -1):
            entry = price
            state = target
        else:
            if state == 1:
                r = (price - entry) / entry - 2 * COST_PER_SIDE
            elif state == -1:
                r = (entry - price) / entry - 2 * COST_PER_SIDE
            else:
                r = 0.0
            trades.append(r)
            state = target
            entry = price if target in (1, -1) else None
    return trades


def _stats(trades):
    if not trades:
        return {"n": 0}
    arr = np.array(trades, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    # compounded total return
    eq = np.cumprod(1.0 + arr)
    total_return = eq[-1] - 1.0
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max((peak - eq) / peak))
    # per-trade Sharpe approximation (annualized with ~6 trades/symbol/month guess)
    sharpe = (arr.mean() / arr.std()) * np.sqrt(len(arr)) if arr.std() > 0 else 0.0
    return {
        "n": int(len(arr)),
        "win_rate_pct": round(100.0 * len(wins) / len(arr), 2),
        "avg_trade_pct": round(100.0 * arr.mean(), 3),
        "avg_win_pct": round(100.0 * wins.mean(), 3) if len(wins) else None,
        "avg_loss_pct": round(100.0 * losses.mean(), 3) if len(losses) else None,
        "profit_factor": round(float(profit_factor), 2),
        "total_return_pct": round(100.0 * total_return, 2),
        "max_drawdown_pct": round(100.0 * max_dd, 2),
        "sharpe_per_trade": round(float(sharpe), 2),
    }


def _portfolio_stats(h4_map, pos_map, symbols):
    """Equal-weight portfolio of all symbols; cost on position changes.

    Returns total return %, max drawdown %, and annualized Sharpe from the
    portfolio equity curve (H4 bars, ~6/day -> ~2190 bars/year).
    """
    daily = pd.DataFrame(index=h4_map[symbols[0]].index)
    for sym in symbols:
        df = h4_map[sym]
        pos = pos_map[sym].reindex(df.index).fillna(0)
        ret = df["Close"].pct_change().fillna(0.0)
        pos_held = pos.shift(1).fillna(0)
        chg = pos.diff().abs().fillna(0)
        daily[sym] = pos_held * ret - COST_PER_SIDE * chg
    port = daily.mean(axis=1)
    eq = (1.0 + port).cumprod()
    total_return = eq.iloc[-1] - 1.0
    peak = eq.cummax()
    max_dd = float(((peak - eq) / peak).max())
    # annualized Sharpe: ~6 bars/day * 365 days
    bars_per_year = 6 * 365
    sharpe = (port.mean() / port.std()) * np.sqrt(bars_per_year) if port.std() > 0 else 0.0
    return {
        "total_return_pct": round(100.0 * total_return, 2),
        "max_drawdown_pct": round(100.0 * max_dd, 2),
        "sharpe_annualized": round(float(sharpe), 2),
    }


def _benchmark(h4_map):
    # BTC buy-and-hold + equal-weight basket over the common window
    btc = h4_map["BTC-USD"]
    btc_ret = (btc["Close"].iloc[-1] / btc["Close"].iloc[0]) - 1.0
    basket = []
    for sym in WATCHLIST:
        df = h4_map[sym]
        basket.append((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1.0)
    return {
        "btc_buyhold_pct": round(100.0 * btc_ret, 2),
        "equal_weight_basket_pct": round(100.0 * float(np.mean(basket)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(WATCHLIST))
    ap.add_argument("--json", action="store_true", help="emit machine-readable summary")
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    h4_map = {sym: load_h4(sym) for sym in symbols}
    signals = {
        "donchian_20": lambda df: signal_donchian(df),
        "ema_50_200": lambda df: signal_ema_cross(df),
        "trend_breakout": lambda df: signal_trend_breakout(df),
    }

    results = {}
    for name, fn in signals.items():
        per_symbol = {}
        pos_map = {}
        all_trades = []
        for sym in symbols:
            pos = fn(h4_map[sym])
            pos_map[sym] = pos
            trades = _trade_returns(h4_map[sym], pos)
            per_symbol[sym] = _stats(trades)
            all_trades.extend(trades)
        agg = _stats(all_trades)
        agg.update(_portfolio_stats(h4_map, pos_map, symbols))
        results[name] = {
            "aggregate": agg,
            "symbols": per_symbol,
        }

    bench = _benchmark(h4_map)
    if args.json:
        print(json.dumps({"results": results, "benchmark": bench}, ensure_ascii=False, indent=2))
        return

    print("=" * 78)
    print("H4 trend-following backtest (no lookahead, entry/exit @next open, cost 0.30%/round)")
    print("=" * 78)
    for name in signals:
        agg = results[name]["aggregate"]
        print(f"\n[{name}] aggregate (portfolio)")
        for k in ("n", "win_rate_pct", "avg_trade_pct", "avg_win_pct", "avg_loss_pct",
                  "profit_factor", "total_return_pct", "max_drawdown_pct", "sharpe_annualized"):
            print(f"  {k:20s} = {agg.get(k)}")
    print("\n[benchmark]")
    for k, v in bench.items():
        print(f"  {k:26s} = {v}")
    print("\n[per-symbol avg_trade_pct] (signal: symbol=avg%)")
    for name in signals:
        row = {sym: results[name]["symbols"][sym].get("avg_trade_pct") for sym in symbols}
        print(f"  {name}: " + ", ".join(f"{sym.split('-')[0]}={v}" for sym, v in row.items()))


if __name__ == "__main__":
    main()
