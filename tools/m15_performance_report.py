#!/usr/bin/env python
"""Cost-adjusted monthly performance report for the live M15 alert system.

Reads realized_outcomes.json and reports NET performance (after a round-trip
cost model) for directional entries, monthly and cumulative, with a buy-and-hold
benchmark. This is the tracking tool for proving whether the M15 edge is real
over 6-12 months.

Cost model (defaults): fee 0.10%/side + slippage 0.05%/side = 0.30% round trip.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WATCHLIST = [
    "BTC-USD", "ETH-USD", "DOGE-USD", "ADA-USD", "XRP-USD", "BNB-USD",
    "SOL-USD", "TRX-USD", "NEAR-USD", "LINK-USD", "PAXG-USD",
]
COST_ROUND_TRIP_PCT = 0.30  # configurable via --cost-pct


def load_outcomes(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list((payload or {}).get("outcomes") or [])


def _settled_entries(outcomes):
    rows = []
    for x in outcomes:
        if x.get("outcome_status") != "settled":
            continue
        if str(x.get("alert_intent") or "").strip().lower() != "entry":
            continue
        pnl = x.get("pnl_pct")
        ts = str(x.get("timestamp") or "")[:19]
        if not isinstance(pnl, (int, float)) or not ts:
            continue
        rows.append({"timestamp": ts, "gross_pnl_pct": float(pnl)})
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def _benchmark(start, end):
    """Equal-weight basket + BTC buy-and-hold over [start, end] using H4 cache."""
    rets = {}
    for sym in WATCHLIST:
        files = glob.glob(str(ROOT / f".data/market_history/binance/{sym}_1h_adj_binance_*.csv"))
        if not files:
            continue
        df = pd.read_csv(files[0])
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        h4 = df.resample("4h").agg(agg).dropna()
        window = h4.loc[(h4.index >= start) & (h4.index <= end)]
        if window.empty:
            continue
        rets[sym] = (window["Close"].iloc[-1] / window["Close"].iloc[0]) - 1.0
    btc = rets.get("BTC-USD")
    basket = float(np.mean(list(rets.values()))) if rets else None
    return {
        "btc_buyhold_pct": round(100.0 * btc, 2) if btc is not None else None,
        "basket_buyhold_pct": round(100.0 * basket, 2) if basket is not None else None,
    }


def _monthly(rows, cost_pct):
    df = pd.DataFrame(rows)
    df["net_pnl_pct"] = df["gross_pnl_pct"] - cost_pct
    df["month"] = pd.to_datetime(df["timestamp"]).dt.to_period("M").astype(str)
    out = []
    for m, g in df.groupby("month"):
        wins = (g["net_pnl_pct"] > 0).sum()
        losses = (g["net_pnl_pct"] <= 0).sum()
        win_sum = g.loc[g["net_pnl_pct"] > 0, "net_pnl_pct"].sum()
        loss_sum = abs(g.loc[g["net_pnl_pct"] <= 0, "net_pnl_pct"].sum())
        pf = win_sum / loss_sum if loss_sum > 0 else float("inf")
        out.append({
            "month": m,
            "n": int(len(g)),
            "win_rate_pct": round(100.0 * wins / len(g), 1),
            "avg_net_pnl_pct": round(float(g["net_pnl_pct"].mean()), 3),
            "profit_factor": round(float(pf), 2),
            "sum_net_pnl_pct": round(float(g["net_pnl_pct"].sum()), 2),
        })
    return df, out


def _cumulative(df):
    net = df["net_pnl_pct"].values
    eq = np.cumprod(1.0 + net / 100.0)
    total = eq[-1] - 1.0
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max((peak - eq) / peak))
    # per-trade Sharpe annualized
    days = (pd.to_datetime(df["timestamp"].max()) - pd.to_datetime(df["timestamp"].min())).days or 1
    trades_per_year = len(net) / max(days, 1) * 365.0
    sharpe = (net.mean() / net.std()) * np.sqrt(max(trades_per_year, 1.0)) if net.std() > 0 else 0.0
    return {
        "n": int(len(net)),
        "span_days": int(days),
        "gross_avg_pnl_pct": round(float(df["gross_pnl_pct"].mean()), 3),
        "net_avg_pnl_pct": round(float(net.mean()), 3),
        "net_total_return_pct": round(100.0 * total, 2),
        "max_drawdown_pct": round(100.0 * max_dd, 2),
        "sharpe_annualized": round(float(sharpe), 2),
        "win_rate_pct": round(100.0 * float((net > 0).sum()) / len(net), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes-path", required=True, help="path to realized_outcomes.json")
    ap.add_argument("--cost-pct", type=float, default=COST_ROUND_TRIP_PCT,
                    help="round-trip cost in percent (default 0.30)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = _settled_entries(load_outcomes(args.outcomes_path))
    if not rows:
        print("no settled entry outcomes found")
        return

    df, monthly = _monthly(rows, args.cost_pct)
    cum = _cumulative(df)
    start = pd.to_datetime(rows[0]["timestamp"])
    end = pd.to_datetime(rows[-1]["timestamp"])
    bench = _benchmark(start, end)

    if args.json:
        print(json.dumps({"cumulative": cum, "monthly": monthly, "benchmark": bench}, ensure_ascii=False, indent=2))
        return

    print("=" * 72)
    print(f"M15 entry performance (NET, cost {args.cost_pct}%/round trip)")
    print(f"span {start:%Y-%m-%d} -> {end:%Y-%m-%d} ({cum['span_days']} days, {cum['n']} trades)")
    print("=" * 72)
    print("\n[cumulative]  (per-trade expectancy is the honest metric; the")
    print("  total/drawdown/sharpe below assume SEQUENTIAL compounding of each")
    print("  trade, which is an optimistic upper bound for concurrent alerts)")
    labels = {
        "gross_avg_pnl_pct": "gross expectancy/trade %",
        "net_avg_pnl_pct": "NET expectancy/trade %",
        "win_rate_pct": "win rate % (net)",
        "net_total_return_pct": "total % (sequential UB)",
        "max_drawdown_pct": "max DD % (sequential)",
        "sharpe_annualized": "sharpe (per-trade, optimistic)",
    }
    for k, label in labels.items():
        print(f"  {label:30s} = {cum[k]}")
    print("\n[benchmark over same span]")
    for k, v in bench.items():
        print(f"  {k:22s} = {v}")
    print("\n[monthly]")
    print(f"  {'month':8s} {'n':>4s} {'WR%':>6s} {'avgNet%':>8s} {'PF':>6s} {'sumNet%':>8s}")
    for r in monthly:
        print(f"  {r['month']:8s} {r['n']:>4d} {r['win_rate_pct']:>6.1f} {r['avg_net_pnl_pct']:>8.3f} {r['profit_factor']:>6.2f} {r['sum_net_pnl_pct']:>8.2f}")


if __name__ == "__main__":
    main()
