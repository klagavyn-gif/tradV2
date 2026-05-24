import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def cdc_forecast_bias_at(df, pos, momentum_lookback=3):
    close = df["Close"]
    fast = df["fast"]
    slow = df["slow"]
    k = df["k"]
    d = df["d"]
    vix_spike = df["vix_spike"]
    lookback = max(1, int(momentum_lookback))
    bull_score = 0.0
    bear_score = 0.0
    close_now = close.iloc[pos]
    fast_now = fast.iloc[pos]
    slow_now = slow.iloc[pos]
    k_now = k.iloc[pos]
    d_now = d.iloc[pos]
    fast_prev = fast.iloc[pos - lookback] if pos - lookback >= 0 else np.nan
    close_prev = close.iloc[pos - lookback] if pos - lookback >= 0 else np.nan
    if pd.notna(fast_now) and pd.notna(slow_now):
        if fast_now > slow_now:
            bull_score += 18.0
        elif fast_now < slow_now:
            bear_score += 18.0
    if pd.notna(close_now) and pd.notna(fast_now):
        if close_now > fast_now:
            bull_score += 12.0
        elif close_now < fast_now:
            bear_score += 12.0
    if pd.notna(fast_now) and pd.notna(fast_prev):
        if fast_now > fast_prev:
            bull_score += 10.0
        elif fast_now < fast_prev:
            bear_score += 10.0
    if pd.notna(close_now) and pd.notna(close_prev) and close_prev != 0:
        momentum_pct = ((close_now - close_prev) / abs(close_prev)) * 100.0
        if momentum_pct > 0:
            bull_score += min(12.0, 6.0 + momentum_pct * 4.0)
        elif momentum_pct < 0:
            bear_score += min(12.0, 6.0 + abs(momentum_pct) * 4.0)
    if pd.notna(k_now) and pd.notna(d_now):
        if k_now > d_now:
            bull_score += 8.0
        elif k_now < d_now:
            bear_score += 8.0
        if k_now < 35.0 and d_now < 35.0:
            bull_score += 6.0
        if k_now > 70.0 and d_now > 70.0:
            bear_score += 6.0
    if bool(vix_spike.iloc[pos]):
        bear_score += 14.0
    gap = abs(bull_score - bear_score)
    direction = "NEUTRAL"
    if bull_score >= bear_score + 5.0:
        direction = "BUY"
    elif bear_score >= bull_score + 5.0:
        direction = "SELL"
    score = min(92.0, 50.0 + gap)
    return direction, float(max(50.0, score))


def prepare_dataframe(path):
    df = pd.read_csv(path)
    if len(df) < 220:
        return None
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = df[col].astype(float)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    df["fast"] = close.ewm(span=12, adjust=False).mean()
    df["slow"] = close.ewm(span=26, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()
    df["bull"] = df["fast"] > df["slow"]
    df["bear"] = df["fast"] < df["slow"]
    df["green"] = df["bull"] & (close > df["fast"])
    df["red"] = df["bear"] & (close < df["fast"])
    df["green_flip"] = df["green"] & (~df["green"].shift(1, fill_value=False))
    df["red_flip"] = df["red"] & (~df["red"].shift(1, fill_value=False))

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    rs = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_low = rsi.rolling(14).min()
    rsi_high = rsi.rolling(14).max()
    stoch_rsi = ((rsi - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)) * 100.0
    df["k"] = stoch_rsi.rolling(3).mean()
    df["d"] = df["k"].rolling(3).mean()
    df["cross_up"] = (df["k"] > df["d"]) & (df["k"].shift(1) <= df["d"].shift(1))
    df["cross_down"] = (df["k"] < df["d"]) & (df["k"].shift(1) >= df["d"].shift(1))

    highest_close = close.rolling(22).max()
    wvf = ((highest_close - low) / highest_close.replace(0, np.nan)) * 100.0
    df["wvf"] = wvf
    upper_band = wvf.rolling(20).mean() + (2.0 * wvf.rolling(20).std())
    range_high = wvf.rolling(50).max() * 0.85
    df["vix_spike"] = (wvf >= upper_band) | (wvf >= range_high)
    df["recent_vix_spike"] = df["vix_spike"].rolling(3, min_periods=1).max().fillna(0).astype(bool)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = (df["atr"] / close.replace(0, np.nan)) * 100.0
    df["fast_slope"] = df["fast"].diff(3)
    df["slow_slope"] = df["slow"].diff(3)
    df["ema_gap_pct"] = ((df["fast"] - df["slow"]) / df["slow"].replace(0, np.nan)) * 100.0
    return df


def score_buy_event(df, flip_pos, current_pos):
    current_price = float(df["Close"].iloc[current_pos])
    flip_high = float(df["High"].iloc[flip_pos])
    flip_low = float(df["Low"].iloc[max(0, flip_pos - 4) : flip_pos + 1].min())
    flip_close = float(df["Close"].iloc[flip_pos])
    forecast_dir, forecast_score = cdc_forecast_bias_at(df, current_pos)
    reclaim = bool(df["green"].iloc[current_pos]) and current_price > flip_high
    stoch_d = df["d"].iloc[flip_pos]
    gap = df["ema_gap_pct"].iloc[current_pos]
    score = 35.0
    score += 12.0 if bool(df["recent_vix_spike"].iloc[flip_pos]) else -4.0
    score += 10.0 if bool(df["cross_up"].iloc[flip_pos]) else -4.0
    if pd.notna(stoch_d):
        if float(stoch_d) < 40.0:
            score += 10.0
        elif float(stoch_d) < 45.0:
            score += 6.0
        elif float(stoch_d) > 55.0:
            score -= 5.0
    if reclaim:
        score += 10.0
        setup = "POST_FLIP_RECLAIM"
    elif current_price > flip_close:
        score += 4.0
        setup = "FOLLOW_THROUGH"
    else:
        setup = "EARLY_FLIP"
    score += 8.0 if bool(df["fast_slope"].iloc[current_pos] > 0) else -6.0
    if bool(df["slow_slope"].iloc[current_pos] > 0):
        score += 4.0
    if pd.notna(df["ema200"].iloc[current_pos]):
        score += 6.0 if current_price > float(df["ema200"].iloc[current_pos]) else -4.0
    if pd.notna(gap):
        if 0.01 <= float(gap) <= 0.35:
            score += 4.0
        elif float(gap) > 0.60:
            score -= 3.0
    bars_since = current_pos - flip_pos
    if bars_since == 0:
        score += 4.0
    elif bars_since == 1:
        score += 6.0
    elif bars_since == 2:
        score += 3.0
    elif bars_since == 3:
        score += 1.0
    else:
        score -= 6.0
    if forecast_dir == "BUY":
        score += 6.0
    elif forecast_dir == "SELL":
        score -= 4.0
    score += max(0.0, min(5.0, (forecast_score - 55.0) * 0.20))
    score += 4.0 if bool(df["green"].iloc[current_pos]) else -6.0
    return {
        "direction": "BUY",
        "score": max(0.0, min(95.0, float(score))),
        "setup": setup,
        "reclaim": reclaim,
        "bars_since_flip": int(bars_since),
        "flip_time": df["Datetime"].iloc[flip_pos],
        "event_time": df["Datetime"].iloc[current_pos],
        "entry_price": current_price,
        "flip_level": flip_high,
        "protect_level": flip_low,
        "forecast_direction": forecast_dir,
        "forecast_score": forecast_score,
        "atr_pct": float(df["atr_pct"].iloc[current_pos]) if pd.notna(df["atr_pct"].iloc[current_pos]) else None,
        "stoch_d": float(stoch_d) if pd.notna(stoch_d) else None,
        "recent_vix_spike": bool(df["recent_vix_spike"].iloc[flip_pos]),
    }


def score_sell_event(df, flip_pos, current_pos):
    current_price = float(df["Close"].iloc[current_pos])
    flip_low = float(df["Low"].iloc[flip_pos])
    flip_high = float(df["High"].iloc[max(0, flip_pos - 4) : flip_pos + 1].max())
    flip_close = float(df["Close"].iloc[flip_pos])
    forecast_dir, forecast_score = cdc_forecast_bias_at(df, current_pos)
    reclaim = bool(df["red"].iloc[current_pos]) and current_price < flip_low
    stoch_k = df["k"].iloc[flip_pos]
    gap = df["ema_gap_pct"].iloc[current_pos]
    score = 35.0
    score += 8.0 if forecast_dir == "SELL" else (-4.0 if forecast_dir == "BUY" else 0.0)
    score += 10.0 if bool(df["cross_down"].iloc[flip_pos]) else -4.0
    if pd.notna(stoch_k):
        if float(stoch_k) > 60.0:
            score += 10.0
        elif float(stoch_k) > 55.0:
            score += 6.0
        elif float(stoch_k) < 45.0:
            score -= 5.0
    if reclaim:
        score += 10.0
        setup = "POST_FLIP_BREAKDOWN"
    elif current_price < flip_close:
        score += 4.0
        setup = "FOLLOW_THROUGH"
    else:
        setup = "EARLY_FLIP"
    score += 8.0 if bool(df["fast_slope"].iloc[current_pos] < 0) else -6.0
    if bool(df["slow_slope"].iloc[current_pos] < 0):
        score += 4.0
    if pd.notna(df["ema200"].iloc[current_pos]):
        score += 6.0 if current_price < float(df["ema200"].iloc[current_pos]) else -4.0
    if pd.notna(gap):
        if -0.35 <= float(gap) <= -0.01:
            score += 4.0
        elif float(gap) < -0.60:
            score -= 3.0
    bars_since = current_pos - flip_pos
    if bars_since == 0:
        score += 4.0
    elif bars_since == 1:
        score += 6.0
    elif bars_since == 2:
        score += 3.0
    elif bars_since == 3:
        score += 1.0
    else:
        score -= 6.0
    score += max(0.0, min(5.0, (forecast_score - 55.0) * 0.20))
    score += 4.0 if bool(df["red"].iloc[current_pos]) else -6.0
    return {
        "direction": "SELL",
        "score": max(0.0, min(95.0, float(score))),
        "setup": setup,
        "reclaim": reclaim,
        "bars_since_flip": int(bars_since),
        "flip_time": df["Datetime"].iloc[flip_pos],
        "event_time": df["Datetime"].iloc[current_pos],
        "entry_price": current_price,
        "flip_level": flip_low,
        "protect_level": flip_high,
        "forecast_direction": forecast_dir,
        "forecast_score": forecast_score,
        "atr_pct": float(df["atr_pct"].iloc[current_pos]) if pd.notna(df["atr_pct"].iloc[current_pos]) else None,
        "stoch_k": float(stoch_k) if pd.notna(stoch_k) else None,
        "recent_vix_spike": bool(df["vix_spike"].iloc[flip_pos]),
    }


def evaluate_future_path(df, event_pos, direction, horizon_bars, protect_level):
    event_pos = int(event_pos)
    future = df.iloc[event_pos + 1 : event_pos + 1 + horizon_bars].copy()
    if future.empty:
        return None
    entry = float(df["Close"].iloc[event_pos])
    atr_pct = float(df["atr_pct"].iloc[event_pos]) if pd.notna(df["atr_pct"].iloc[event_pos]) else None
    if direction == "BUY":
        future["fav_move_pct"] = ((future["High"] - entry) / entry) * 100.0
        future["adv_move_pct"] = ((entry - future["Low"]) / entry) * 100.0
        peak_idx = int(future["fav_move_pct"].idxmax())
        favorable_move_pct = float(future["fav_move_pct"].max())
        adverse_move_pct = float(future["adv_move_pct"].max())
        end_move_pct = float(((future["Close"].iloc[-1] - entry) / entry) * 100.0)
        stop_risk_pct = ((entry - protect_level) / entry) * 100.0 if protect_level is not None else None
    else:
        future["fav_move_pct"] = ((entry - future["Low"]) / entry) * 100.0
        future["adv_move_pct"] = ((future["High"] - entry) / entry) * 100.0
        peak_idx = int(future["fav_move_pct"].idxmax())
        favorable_move_pct = float(future["fav_move_pct"].max())
        adverse_move_pct = float(future["adv_move_pct"].max())
        end_move_pct = float(((entry - future["Close"].iloc[-1]) / entry) * 100.0)
        stop_risk_pct = ((protect_level - entry) / entry) * 100.0 if protect_level is not None else None
    bars_to_peak = int(peak_idx - event_pos)
    favorable_move_atr = None
    if isinstance(atr_pct, (int, float)) and atr_pct > 0:
        favorable_move_atr = favorable_move_pct / atr_pct
    favorable_move_risk = None
    if isinstance(stop_risk_pct, (int, float)) and stop_risk_pct > 0:
        favorable_move_risk = favorable_move_pct / stop_risk_pct
    return {
        "favorable_move_pct": favorable_move_pct,
        "adverse_move_pct": adverse_move_pct,
        "end_move_pct": end_move_pct,
        "bars_to_peak": bars_to_peak,
        "favorable_move_atr": favorable_move_atr,
        "favorable_move_risk": favorable_move_risk,
        "stop_risk_pct": stop_risk_pct,
    }


def first_qualified_events(df, symbol, direction, min_score, max_wait, require_reclaim, horizon_bars, major_move_pct, major_move_atr_mult):
    rows = []
    flip_col = "green_flip" if direction == "BUY" else "red_flip"
    flip_positions = np.where(df[flip_col].fillna(False))[0]
    for flip_pos in flip_positions:
        chosen = None
        for wait in range(0, max_wait + 1):
            current_pos = flip_pos + wait
            if current_pos >= len(df) - horizon_bars - 1:
                break
            if direction == "BUY":
                payload = score_buy_event(df, flip_pos, current_pos)
                allow_color = bool(df["green"].iloc[current_pos])
            else:
                payload = score_sell_event(df, flip_pos, current_pos)
                allow_color = bool(df["red"].iloc[current_pos])
            if not allow_color:
                continue
            if float(payload["score"]) < float(min_score):
                continue
            if require_reclaim and wait > 0 and not bool(payload["reclaim"]):
                continue
            chosen = payload
            chosen["wait"] = wait
            chosen["flip_pos"] = int(flip_pos)
            chosen["event_pos"] = int(current_pos)
            break
        if not isinstance(chosen, dict):
            continue
        future = evaluate_future_path(df, chosen["event_pos"], direction, horizon_bars, chosen.get("protect_level"))
        if not isinstance(future, dict):
            continue
        cutoff = major_move_pct
        atr_pct = chosen.get("atr_pct")
        if isinstance(atr_pct, (int, float)):
            cutoff = max(float(cutoff), float(atr_pct) * float(major_move_atr_mult))
        chosen.update(future)
        chosen["symbol"] = symbol
        chosen["major_move_cutoff_pct"] = float(cutoff)
        chosen["major_move"] = bool(float(chosen["favorable_move_pct"]) >= float(cutoff))
        chosen["flip_time"] = str(pd.Timestamp(chosen["flip_time"]))
        chosen["event_time"] = str(pd.Timestamp(chosen["event_time"]))
        rows.append(chosen)
    return rows


def summarize_events(rows):
    if not rows:
        return None
    df = pd.DataFrame(rows)
    summary = {
        "signals": int(len(df)),
        "major_moves": int(df["major_move"].sum()),
        "major_move_rate_pct": round(float(df["major_move"].mean() * 100.0), 2),
        "avg_score": round(float(df["score"].mean()), 2),
        "avg_favorable_move_pct": round(float(df["favorable_move_pct"].mean()), 3),
        "median_favorable_move_pct": round(float(df["favorable_move_pct"].median()), 3),
        "avg_adverse_move_pct": round(float(df["adverse_move_pct"].mean()), 3),
        "avg_end_move_pct": round(float(df["end_move_pct"].mean()), 3),
        "avg_bars_to_peak": round(float(df["bars_to_peak"].mean()), 2),
        "avg_favorable_move_atr": round(float(df["favorable_move_atr"].dropna().mean()), 3) if df["favorable_move_atr"].notna().any() else None,
        "avg_favorable_move_risk": round(float(df["favorable_move_risk"].dropna().mean()), 3) if df["favorable_move_risk"].notna().any() else None,
        "setup_mix": {str(k): int(v) for k, v in df["setup"].value_counts().to_dict().items()},
        "wait_mix": {str(int(k)): int(v) for k, v in df["wait"].value_counts().sort_index().to_dict().items()},
    }
    threshold_grid = []
    for threshold in (60, 65, 68, 70, 75, 80):
        sub = df[df["score"] >= threshold]
        if sub.empty:
            continue
        threshold_grid.append(
            {
                "threshold": threshold,
                "signals": int(len(sub)),
                "major_move_rate_pct": round(float(sub["major_move"].mean() * 100.0), 2),
                "avg_favorable_move_pct": round(float(sub["favorable_move_pct"].mean()), 3),
                "avg_adverse_move_pct": round(float(sub["adverse_move_pct"].mean()), 3),
            }
        )
    top_events = (
        df.sort_values(["major_move", "favorable_move_pct", "score"], ascending=[False, False, False])
        .head(8)
        .to_dict(orient="records")
    )
    summary["threshold_grid"] = threshold_grid
    summary["top_events"] = top_events
    return summary


def build_report(root_dir, min_score, max_wait, require_reclaim, horizon_bars, major_move_pct, major_move_atr_mult):
    history_dir = root_dir / ".data" / "yf_history"
    symbol_reports = []
    all_rows = []
    for path in sorted(history_dir.glob("*_15m_*.csv")):
        symbol = path.name.split("_15m_")[0]
        df = prepare_dataframe(path)
        if df is None:
            continue
        buy_rows = first_qualified_events(
            df,
            symbol,
            "BUY",
            min_score,
            max_wait,
            require_reclaim,
            horizon_bars,
            major_move_pct,
            major_move_atr_mult,
        )
        sell_rows = first_qualified_events(
            df,
            symbol,
            "SELL",
            min_score,
            max_wait,
            require_reclaim,
            horizon_bars,
            major_move_pct,
            major_move_atr_mult,
        )
        all_rows.extend(buy_rows)
        all_rows.extend(sell_rows)
        symbol_reports.append(
            {
                "symbol": symbol,
                "buy": summarize_events(buy_rows),
                "sell_mirror": summarize_events(sell_rows),
            }
        )
    symbol_reports = sorted(
        symbol_reports,
        key=lambda row: max(
            float(((row.get("buy") or {}).get("major_move_rate_pct") or 0.0)),
            float(((row.get("sell_mirror") or {}).get("major_move_rate_pct") or 0.0)),
        ),
        reverse=True,
    )
    overall_buy = summarize_events([row for row in all_rows if row.get("direction") == "BUY"])
    overall_sell = summarize_events([row for row in all_rows if row.get("direction") == "SELL"])
    best_buy = [
        row
        for row in all_rows
        if row.get("direction") == "BUY" and bool(row.get("major_move"))
    ]
    best_sell = [
        row
        for row in all_rows
        if row.get("direction") == "SELL" and bool(row.get("major_move"))
    ]
    best_buy = sorted(best_buy, key=lambda row: (float(row.get("favorable_move_pct", 0.0)), float(row.get("score", 0.0))), reverse=True)[:20]
    best_sell = sorted(best_sell, key=lambda row: (float(row.get("favorable_move_pct", 0.0)), float(row.get("score", 0.0))), reverse=True)[:20]
    return {
        "request": {
            "min_score": float(min_score),
            "max_wait_bars": int(max_wait),
            "require_reclaim": bool(require_reclaim),
            "horizon_bars": int(horizon_bars),
            "major_move_pct_floor": float(major_move_pct),
            "major_move_atr_mult": float(major_move_atr_mult),
        },
        "summary": {
            "symbols": int(len(symbol_reports)),
            "events": int(len(all_rows)),
            "buy": overall_buy,
            "sell_mirror": overall_sell,
        },
        "top_major_buy_events": best_buy,
        "top_major_sell_events": best_sell,
        "per_symbol": symbol_reports,
        "event_rows": all_rows,
    }


def write_outputs(root_dir, report):
    out_dir = root_dir / ".data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cdc_red_green_backtest_report.json"
    csv_path = out_dir / "cdc_red_green_backtest_events.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    event_rows = report.get("event_rows") or []
    if event_rows:
        pd.DataFrame(event_rows).sort_values(
            ["direction", "major_move", "favorable_move_pct", "score"],
            ascending=[True, False, False, False],
        ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-score", type=float, default=68.0)
    parser.add_argument("--max-wait", type=int, default=3)
    parser.add_argument("--require-reclaim", dest="require_reclaim", action="store_true")
    parser.add_argument("--no-reclaim", dest="require_reclaim", action="store_false")
    parser.set_defaults(require_reclaim=True)
    parser.add_argument("--horizon-bars", type=int, default=64)
    parser.add_argument("--major-move-pct", type=float, default=3.0)
    parser.add_argument("--major-move-atr-mult", type=float, default=3.0)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    report = build_report(
        root_dir,
        min_score=args.min_score,
        max_wait=max(0, int(args.max_wait)),
        require_reclaim=bool(args.require_reclaim),
        horizon_bars=max(8, int(args.horizon_bars)),
        major_move_pct=max(0.5, float(args.major_move_pct)),
        major_move_atr_mult=max(0.5, float(args.major_move_atr_mult)),
    )
    json_path, csv_path = write_outputs(root_dir, report)
    print(
        json.dumps(
            {
                "json_report": str(json_path),
                "csv_report": str(csv_path),
                "summary": report.get("summary"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
