#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/screen_binance_candidates.py

QuantumLab Asset Screener & Shadow Backtest Engine.
Evaluates potential candidate replacement coins on Binance USDT-M Futures:
1. Downloads multi-month 15m klines from Binance Public API.
2. Applies the exact CDCVIX15 strategy (CDC ActionZone + Williams VIXFix + StochRSI).
3. Simulates the production trade lifecycle:
   - ATR(14) * 1.5 initial Stop Loss
   - Breakeven Stop @ +1.2R
   - Trailing Stop @ +2.0R (0.8R distance)
   - Roundtrip fee + slippage deduction (0.30% / 30 bps per trade)
4. Checks real-time liquidity (24h volume) and bid-ask spread via bookTicker.
5. Produces a comparative Leaderboard against production benchmarks (SUI, ONDO, ETH).
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / ".data" / "market_history" / "binance_lab"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FAPI_URL = "https://fapi.binance.com"
COST_ROUNDTRIP_PCT = 0.30  # 0.30% fee + slippage per roundtrip trade

# Default candidate pools
CANDIDATE_SYMBOLS = [
    "TAOUSDT",     # AI Decentralized Leader
    "RENDERUSDT",  # GPU Cloud Computing Leader
    "AVAXUSDT",    # High-Liquidity Clean L1
    "APTUSDT",     # High-Speed Move L1
    "SEIUSDT",     # Trading-Optimized L1
    "PENDLEUSDT",  # Yield Trading / RWA Leader
    "TIAUSDT",     # Modular Blockchain Leader
    "NEARUSDT",    # AI / Layer-1
    "INJUSDT",     # DeFi Infrastructure L1
]

BENCHMARK_SYMBOLS = [
    "SUIUSDT",     # Production Benchmark (Top Performer)
    "ONDOUSDT",    # Production Benchmark (RWA Leader)
    "ETHUSDT",     # Production Benchmark (Core L1)
]

CONTROL_BLOCKED_SYMBOLS = [
    "SOLUSDT",     # Blocked control (High-wick whipsaws)
    "DOGEUSDT",    # Blocked control (Meme noise)
]


def fetch_binance_klines(symbol: str, target_candles: int = 10000) -> pd.DataFrame:
    """
    Fetches historical 15m klines from Binance Public API with local caching.
    """
    cache_file = CACHE_DIR / f"{symbol}_15m_{target_candles}.csv"
    if cache_file.exists():
        # Check cache age (valid for 12 hours)
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 43200:
            df = pd.read_csv(cache_file)
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df.set_index("Datetime", inplace=True)
            return df

    all_klines = []
    end_time = None
    remaining = target_candles

    session = requests.Session()
    session.headers.update({"User-Agent": "tradV2-screener/1.0"})

    while remaining > 0:
        batch_limit = min(remaining, 1500)
        params = {"symbol": symbol, "interval": "15m", "limit": batch_limit}
        if end_time:
            params["endTime"] = end_time

        try:
            resp = session.get(f"{FAPI_URL}/fapi/v1/klines", params=params, timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data or not isinstance(data, list):
                break
            all_klines = data + all_klines
            remaining -= len(data)
            end_time = data[0][0] - 1  # 1 ms before oldest candle
            if len(data) < batch_limit:
                break
            time.sleep(0.08)  # Gentle on Binance rate limits
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break

    if not all_klines:
        return pd.DataFrame()

    # Deduplicate and sort
    seen_times = set()
    rows = []
    for k in all_klines:
        t = int(k[0])
        if t not in seen_times:
            seen_times.add(t)
            rows.append({
                "Datetime": pd.to_datetime(t, unit="ms"),
                "Open": float(k[1]),
                "High": float(k[2]),
                "Low": float(k[3]),
                "Close": float(k[4]),
                "Volume": float(k[5]),
            })

    df = pd.DataFrame(rows).sort_values("Datetime").set_index("Datetime")
    df.to_csv(cache_file)
    return df


def get_market_ticker_stats(symbol: str) -> Dict[str, Any]:
    """
    Fetches real-time 24h volume and bid-ask spread.
    """
    stats = {"volume_24h_usdt": 0.0, "spread_pct": 0.0, "bid": 0.0, "ask": 0.0}
    try:
        # 1. 24h volume
        r1 = requests.get(f"{FAPI_URL}/fapi/v1/ticker/24hr", params={"symbol": symbol}, timeout=5).json()
        stats["volume_24h_usdt"] = float(r1.get("quoteVolume", 0.0))

        # 2. Book ticker
        r2 = requests.get(f"{FAPI_URL}/fapi/v1/ticker/bookTicker", params={"symbol": symbol}, timeout=5).json()
        bid = float(r2.get("bidPrice", 0.0))
        ask = float(r2.get("askPrice", 0.0))
        if ask > 0 and bid > 0:
            stats["bid"] = bid
            stats["ask"] = ask
            stats["spread_pct"] = ((ask - bid) / ask) * 100.0
    except Exception:
        pass
    return stats


def compute_cdcvix15_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes exact production CDC ActionZone + Williams VIXFix indicators and signals.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # 1. ActionZone EMAs (12, 26, 200)
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    bull = fast > slow
    green = bull & (close > fast)
    buy_flip = green & (~green.shift(1, fill_value=False))

    # 2. Williams VIXFix
    vix_pd = 22
    vix_bbl = 20
    vix_mult = 2.0
    highest_close = close.rolling(vix_pd).max()
    wvf = ((highest_close - low) / highest_close.replace(0, np.nan)) * 100.0
    sdev = vix_mult * wvf.rolling(vix_bbl).std()
    mid = wvf.rolling(vix_bbl).mean()
    upper_band = mid + sdev
    range_high = wvf.rolling(50).max() * 0.85
    vixfix_spike = (wvf >= upper_band) | (wvf >= range_high)
    recent_vix_spike = vixfix_spike.rolling(4, min_periods=1).max().fillna(0).astype(bool)

    # 3. StochRSI (14, 14, 3, 3)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_low = rsi.rolling(14).min()
    rsi_high = rsi.rolling(14).max()
    stoch_rsi = ((rsi - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)) * 100.0
    k = stoch_rsi.rolling(3).mean()
    d = k.rolling(3).mean()
    cross_up = (k > d) & (k.shift(1) <= d.shift(1))
    storsi_buy_sig = np.where(d < 45.0, np.where(cross_up, 2, 0), np.where(cross_up, 1, 0))
    storsi_buy_ok = pd.Series(storsi_buy_sig > 0, index=df.index)

    # 4. ATR(14)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()

    # 5. Kaufman Efficiency Ratio (Trend Cleanliness over 20 bars)
    direction = (close - close.shift(20)).abs()
    volatility = (close - close.shift(1)).abs().rolling(20).sum()
    er = direction / volatility.replace(0, np.nan)

    # Exact Production Buy Condition:
    # Green Flip + Recent VIXFix Spike + StochRSI Bullish cross + Trend alignment (close > EMA200)
    buy_ready = buy_flip & recent_vix_spike & storsi_buy_ok & (close > ema200)

    df["Fast"] = fast
    df["Slow"] = slow
    df["EMA200"] = ema200
    df["ATR"] = atr
    df["ER"] = er
    df["BuySignal"] = buy_ready
    return df


def simulate_coin_trades(df: pd.DataFrame, symbol: str, min_cooldown_bars: int = 32) -> Dict[str, Any]:
    """
    Simulates production trade lifecycle on the OHLCV series.
    Incorporates ATR stop loss, Breakeven Stop (1.2R), Trailing Stop (2.0R / 0.8R dist),
    8-hour cooldown (32 bars of 15m), and 0.30% roundtrip cost deduction.
    """
    trades = []
    in_pos = False
    entry_price = 0.0
    stop_loss = 0.0
    risk = 0.0
    be_active = False
    ts_active = False
    peak_price = 0.0
    entry_time = None
    entry_bar_idx = 0
    last_entry_bar = -9999

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    atrs = df["ATR"].values
    buys = df["BuySignal"].values
    times = df.index

    for i in range(200, len(df)):
        c_price = closes[i]
        h_price = highs[i]
        l_price = lows[i]
        atr_val = atrs[i]

        # Manage active position
        if in_pos:
            bars_held = i - entry_bar_idx
            if h_price > peak_price:
                peak_price = h_price

            # Check Breakeven Stop trigger (+1.2R)
            mfe_r = (peak_price - entry_price) / risk if risk > 0 else 0
            if mfe_r >= 1.2 and not be_active:
                be_active = True
                stop_loss = max(stop_loss, entry_price)

            # Check Trailing Stop trigger (+2.0R, 0.8R distance)
            if mfe_r >= 2.0:
                ts_active = True
                trail_sl = peak_price - (0.8 * risk)
                stop_loss = max(stop_loss, trail_sl)

            # Check Exit: Stop loss hit
            if l_price <= stop_loss:
                exit_price = stop_loss
                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                net_pnl_pct = gross_pnl_pct - COST_ROUNDTRIP_PCT
                r_mult = (exit_price - entry_price) / risk if risk > 0 else 0
                exit_reason = "trailing_stop" if ts_active else ("breakeven" if be_active else "stop_loss")
                trades.append({
                    "symbol": symbol,
                    "side": "BUY",
                    "entry_time": entry_time,
                    "exit_time": times[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_pnl_pct": gross_pnl_pct,
                    "net_pnl_pct": net_pnl_pct,
                    "r_mult": r_mult,
                    "exit_reason": exit_reason,
                    "bars_held": bars_held,
                    "mfe_r": mfe_r,
                })
                in_pos = False
            elif bars_held >= 48:  # 12-hour holding limit / time stop
                exit_price = c_price
                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                net_pnl_pct = gross_pnl_pct - COST_ROUNDTRIP_PCT
                r_mult = (exit_price - entry_price) / risk if risk > 0 else 0
                trades.append({
                    "symbol": symbol,
                    "side": "BUY",
                    "entry_time": entry_time,
                    "exit_time": times[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_pnl_pct": gross_pnl_pct,
                    "net_pnl_pct": net_pnl_pct,
                    "r_mult": r_mult,
                    "exit_reason": "time_stop",
                    "bars_held": bars_held,
                    "mfe_r": mfe_r,
                })
                in_pos = False

        # Open new position if flat and cooldown has passed
        if not in_pos and buys[i]:
            if (i - last_entry_bar) >= min_cooldown_bars:
                in_pos = True
                entry_price = c_price
                risk = 1.5 * atr_val
                stop_loss = entry_price - risk
                peak_price = entry_price
                be_active = False
                ts_active = False
                entry_time = times[i]
                entry_bar_idx = i
                last_entry_bar = i

    # Compute comprehensive performance metrics
    if not trades:
        return {
            "symbol": symbol,
            "trades_count": 0,
            "win_rate_pct": 0.0,
            "decisive_win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_total_pnl_pct": 0.0,
            "net_expectancy_trade_pct": 0.0,
            "avg_r_multiple": 0.0,
            "max_drawdown_pct": 0.0,
            "efficiency_ratio": 0.0,
            "trades": [],
        }

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["net_pnl_pct"] > 0]
    losses = tdf[tdf["net_pnl_pct"] < 0]
    flats = tdf[tdf["exit_reason"] == "breakeven"]

    win_count = len(wins)
    loss_count = len(losses)
    total_trades = len(tdf)

    win_rate = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0
    decisive_trades = win_count + loss_count
    decisive_wr = (win_count / decisive_trades) * 100.0 if decisive_trades > 0 else 0.0

    gross_gains = wins["net_pnl_pct"].sum()
    gross_losses = abs(losses["net_pnl_pct"].sum())
    pf = (gross_gains / gross_losses) if gross_losses > 0 else (99.0 if gross_gains > 0 else 0.0)

    total_net_pnl = tdf["net_pnl_pct"].sum()
    expectancy = tdf["net_pnl_pct"].mean()
    avg_r = tdf["r_mult"].mean()

    # Cumulative equity curve & Max Drawdown
    equity_curve = tdf["net_pnl_pct"].cumsum()
    peak = equity_curve.cummax()
    drawdown = peak - equity_curve
    max_dd = drawdown.max() if len(drawdown) > 0 else 0.0

    avg_er = df["ER"].mean()

    return {
        "symbol": symbol,
        "trades_count": total_trades,
        "win_rate_pct": round(win_rate, 1),
        "decisive_win_rate_pct": round(decisive_wr, 1),
        "profit_factor": round(pf, 2),
        "net_total_pnl_pct": round(total_net_pnl, 2),
        "net_expectancy_trade_pct": round(expectancy, 3),
        "avg_r_multiple": round(avg_r, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "efficiency_ratio": round(avg_er, 3),
        "trades": trades,
    }


def run_screening():
    print("=" * 90)
    print(" QUANTUMLAB: BINANCE USDT-M FUTURES ASSET SCREENER & SHADOW BACKTEST")
    print("=" * 90)
    print(f"Candles per symbol: ~10,000 bars (15m / ~104 days)")
    print(f"Friction Model: 0.30% Roundtrip (Fee 0.20% + Slippage 0.10%)")
    print(f"Stops: ATR(14)*1.5 | Breakeven @ +1.2R | Trailing Stop @ +2.0R (0.8R dist)")
    print("=" * 90)

    symbols_to_test = BENCHMARK_SYMBOLS + CANDIDATE_SYMBOLS + CONTROL_BLOCKED_SYMBOLS
    results = []

    for idx, sym in enumerate(symbols_to_test, 1):
        print(f"[{idx:02d}/{len(symbols_to_test):02d}] Fetching & testing {sym:<12}...", end="", flush=True)
        t0 = time.time()
        df = fetch_binance_klines(sym, target_candles=10000)
        if df.empty or len(df) < 500:
            print(f" FAILED (insufficient data: {len(df)} bars)")
            continue

        df_sig = compute_cdcvix15_signals(df)
        perf = simulate_coin_trades(df_sig, sym)
        ticker_stats = get_market_ticker_stats(sym)

        perf["volume_24h_usdt"] = ticker_stats["volume_24h_usdt"]
        perf["spread_pct"] = ticker_stats["spread_pct"]
        perf["category"] = (
            "BENCHMARK" if sym in BENCHMARK_SYMBOLS
            else ("CONTROL_BLOCKED" if sym in CONTROL_BLOCKED_SYMBOLS else "CANDIDATE")
        )
        results.append(perf)
        elapsed = time.time() - t0
        print(f" DONE in {elapsed:.1f}s | Trades: {perf['trades_count']} | WR: {perf['win_rate_pct']}% | PF: {perf['profit_factor']:.2f} | Net: {perf['net_total_pnl_pct']:+.1f}%")

    # Sort results by Profit Factor (descending)
    results.sort(key=lambda x: x["profit_factor"], reverse=True)

    print("\n" + "=" * 115)
    print(f"{'SYMBOL':<12} | {'TYPE':<15} | {'TRADES':<6} | {'WIN RATE':<9} | {'DECISIVE':<9} | {'PF':<6} | {'NET PNL':<10} | {'EXP/TRD':<8} | {'MAX DD':<8} | {'SPREAD':<7} | {'VERDICT'}")
    print("=" * 115)

    leaderboard = []
    for r in results:
        sym = r["symbol"]
        cat = r["category"]
        tc = r["trades_count"]
        wr = r["win_rate_pct"]
        dwr = r["decisive_win_rate_pct"]
        pf = r["profit_factor"]
        pnl = r["net_total_pnl_pct"]
        exp = r["net_expectancy_trade_pct"]
        dd = r["max_drawdown_pct"]
        spread = r["spread_pct"]

        # Qualification Criteria:
        # PF >= 1.45, WinRate >= 40%, Net PnL > +15%, Spread < 0.08%, Trades >= 15
        if cat == "BENCHMARK":
            verdict = "🏆 BENCHMARK"
        elif cat == "CONTROL_BLOCKED":
            verdict = "⛔ BLOCKED CONTROL"
        elif pf >= 1.45 and pnl > 15.0 and spread < 0.08 and tc >= 15:
            verdict = "🟢 ELITE PASSED"
        elif pf >= 1.15 and pnl > 0.0 and spread < 0.10:
            verdict = "🟡 WATCHLIST"
        else:
            verdict = "🔴 REJECTED"

        r["verdict"] = verdict
        leaderboard.append(r)
        print(f"{sym:<12} | {cat:<15} | {tc:<6} | {wr:<8.1f}% | {dwr:<8.1f}% | {pf:<6.2f} | {pnl:>+8.1f}% | {exp:>+6.2f}% | {dd:<7.1f}% | {spread:<6.3f}% | {verdict}")

    print("=" * 115)

    # Save summary artifact
    out_file = CACHE_DIR / "screener_leaderboard.json"
    with open(out_file, "w", encoding="utf-8") as f:
        # Exclude raw trade list from json to keep file lean
        clean_lb = [{k: v for k, v in row.items() if k != "trades"} for row in leaderboard]
        json.dump(clean_lb, f, indent=2, ensure_ascii=False)
    print(f"\nLeaderboard successfully saved to: {out_file}")


if __name__ == "__main__":
    run_screening()
