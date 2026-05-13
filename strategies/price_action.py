import math

import numpy as np
import pandas as pd


def _price_action_pivots(df, pivot_length=3):
    highs = []
    lows = []
    if df is None or getattr(df, "empty", True):
        return highs, lows
    try:
        pivot_length = max(2, int(pivot_length))
    except Exception:
        pivot_length = 3
    if len(df) < pivot_length * 2 + 5:
        return highs, lows
    high = df["High"].astype(float).reset_index(drop=True)
    low = df["Low"].astype(float).reset_index(drop=True)
    for idx in range(pivot_length, len(df) - pivot_length):
        high_window = high.iloc[idx - pivot_length : idx + pivot_length + 1]
        low_window = low.iloc[idx - pivot_length : idx + pivot_length + 1]
        high_val = high.iloc[idx]
        low_val = low.iloc[idx]
        if pd.notna(high_val) and float(high_val) >= float(high_window.max()):
            highs.append((idx, float(high_val)))
        if pd.notna(low_val) and float(low_val) <= float(low_window.min()):
            lows.append((idx, float(low_val)))
    return highs[-6:], lows[-6:]


def _price_action_market_structure(pivot_highs, pivot_lows):
    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        last_high = float(pivot_highs[-1][1])
        prev_high = float(pivot_highs[-2][1])
        last_low = float(pivot_lows[-1][1])
        prev_low = float(pivot_lows[-2][1])
        if last_high > prev_high and last_low > prev_low:
            return "UPTREND", "HH/HL"
        if last_high < prev_high and last_low < prev_low:
            return "DOWNTREND", "LH/LL"
    return "SIDEWAY", "RANGE"


def _price_action_chart_pattern(df, pivot_highs, pivot_lows, safe_float, sr_lookback=24, tolerance_pct=0.45):
    if df is None or getattr(df, "empty", True):
        return {"label": None, "signal": None, "score": 0.0}
    try:
        sr_lookback = max(8, int(sr_lookback))
    except Exception:
        sr_lookback = 24
    close_now = safe_float(df["Close"].iloc[-1], None)
    if not isinstance(close_now, (int, float)) or close_now <= 0:
        return {"label": None, "signal": None, "score": 0.0}
    recent_window = df.tail(sr_lookback + 1)
    recent_high = safe_float(
        recent_window["High"].iloc[:-1].max() if len(recent_window) > 1 else recent_window["High"].max(),
        None,
    )
    recent_low = safe_float(
        recent_window["Low"].iloc[:-1].min() if len(recent_window) > 1 else recent_window["Low"].min(),
        None,
    )
    tol_ratio = max(0.0015, float(tolerance_pct) / 100.0)
    if len(pivot_lows) >= 2:
        l1 = float(pivot_lows[-1][1])
        l2 = float(pivot_lows[-2][1])
        if abs(l1 - l2) / close_now <= tol_ratio and isinstance(recent_high, (int, float)) and close_now > (
            (l1 + recent_high) / 2.0
        ):
            return {"label": "Double Bottom", "signal": "BUY", "score": 12.0}
    if len(pivot_highs) >= 2:
        h1 = float(pivot_highs[-1][1])
        h2 = float(pivot_highs[-2][1])
        if abs(h1 - h2) / close_now <= tol_ratio and isinstance(recent_low, (int, float)) and close_now < (
            (h1 + recent_low) / 2.0
        ):
            return {"label": "Double Top", "signal": "SELL", "score": 12.0}
    breakout_buffer = max(0.0010, tol_ratio * 0.8)
    if isinstance(recent_high, (int, float)) and close_now > float(recent_high) * (1.0 + breakout_buffer):
        return {"label": "Range Breakout", "signal": "BUY", "score": 10.0}
    if isinstance(recent_low, (int, float)) and close_now < float(recent_low) * (1.0 - breakout_buffer):
        return {"label": "Range Breakdown", "signal": "SELL", "score": 10.0}
    return {"label": None, "signal": None, "score": 0.0}


def _price_action_wyckoff_phase(structure, current_price, recent_low, recent_high):
    structure = str(structure or "").upper().strip()
    if structure == "UPTREND":
        return "Markup"
    if structure == "DOWNTREND":
        return "Markdown"
    if not all(isinstance(v, (int, float)) for v in (current_price, recent_low, recent_high)):
        return "Range"
    width = float(recent_high) - float(recent_low)
    if width <= 0:
        return "Range"
    pos = (float(current_price) - float(recent_low)) / width
    if pos <= 0.40:
        return "Accumulation"
    if pos >= 0.60:
        return "Distribution"
    return "Range Balance"


def build_price_action_plan(
    symbol,
    item,
    *,
    data_15m=None,
    data_1h=None,
    order_blocks=None,
    prediction=None,
    phase_status=None,
    config,
    helpers,
):
    add_price_patterns = helpers["add_price_patterns"]
    safe_float = helpers["safe_float"]
    recent_pattern_confirmation = helpers["recent_pattern_confirmation"]
    price_action_proxy_metrics = helpers["price_action_proxy_metrics"]
    candle_based_risk = helpers["candle_based_risk"]

    if not bool(getattr(config, "PRICE_ACTION_15M_ENABLED", True)):
        return None
    if not isinstance(data_15m, pd.DataFrame) or data_15m.empty:
        return None
    if not all(col in data_15m.columns for col in ("Open", "High", "Low", "Close", "Volume")) or len(data_15m) < 80:
        return None

    df = data_15m[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = add_price_patterns(df)
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(window=14).mean()
    df["Vol_Avg"] = df["Volume"].rolling(window=20).mean()
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_sum = df["TR"].rolling(window=14).sum()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(window=14).sum() / tr_sum.replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(window=14).sum() / tr_sum.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100.0
    df["ADX"] = dx.rolling(window=14).mean()

    current_price = safe_float(df["Close"].iloc[-1], None)
    if not isinstance(current_price, (int, float)) or current_price <= 0:
        return None
    atr_value = safe_float(df["ATR"].iloc[-1], None)
    adx_value = safe_float(df["ADX"].iloc[-1], None)
    pivot_length = max(2, int(getattr(config, "PRICE_ACTION_15M_SWING_LOOKBACK", 3)))
    sr_lookback = max(8, int(getattr(config, "PRICE_ACTION_15M_SR_LOOKBACK", 24)))
    buffer_pct = safe_float(getattr(config, "PRICE_ACTION_15M_ROLE_REVERSAL_BUFFER_PCT", 0.35), 0.35)
    zone_proximity_pct = safe_float(getattr(config, "PRICE_ACTION_15M_ZONE_PROXIMITY_PCT", 1.0), 1.0)
    min_adx = safe_float(getattr(config, "PRICE_ACTION_15M_MIN_ADX", 18.0), 18.0)
    min_score = safe_float(getattr(config, "PRICE_ACTION_15M_MIN_SCORE", 68.0), 68.0)
    min_conf = safe_float(getattr(config, "PRICE_ACTION_15M_MIN_ALERT_CONFIDENCE", 66.0), 66.0)
    tp_mult = max(1.2, safe_float(getattr(config, "PRICE_ACTION_15M_TP_MULT", 2.2), 2.2))
    stop_atr_mult = max(0.8, safe_float(getattr(config, "PRICE_ACTION_15M_STOP_ATR_MULT", 1.5), 1.5))
    candle_stop_buffer = safe_float(getattr(config, "PRICE_ACTION_15M_CANDLE_STOP_BUFFER_ATR", 0.15), 0.15)
    require_pattern = bool(getattr(config, "PRICE_ACTION_15M_REQUIRE_PATTERN", True))
    require_ema200 = bool(getattr(config, "PRICE_ACTION_15M_REQUIRE_EMA200_ALIGNMENT", False))
    min_proxy_wr = safe_float(getattr(config, "PRICE_ACTION_15M_MIN_PROXY_WIN_RATE", 55.0), 55.0)
    min_proxy_exp = safe_float(getattr(config, "PRICE_ACTION_15M_MIN_PROXY_EXPECTANCY_RR", 0.0), 0.0)
    min_proxy_trades = int(getattr(config, "PRICE_ACTION_15M_MIN_PROXY_TRADES", 4))
    min_proxy_sources = int(getattr(config, "PRICE_ACTION_15M_MIN_PROXY_SOURCE_COUNT", 1))

    pivot_highs, pivot_lows = _price_action_pivots(df, pivot_length=pivot_length)
    market_structure, structure_label = _price_action_market_structure(pivot_highs, pivot_lows)
    recent_window = df.tail(sr_lookback)
    recent_high = safe_float(recent_window["High"].max(), None)
    recent_low = safe_float(recent_window["Low"].min(), None)

    trend_1h = None
    if isinstance(data_1h, pd.DataFrame) and not data_1h.empty and "Close" in data_1h.columns:
        df_1h = data_1h.copy()
        df_1h["EMA50"] = df_1h["Close"].ewm(span=50, adjust=False).mean()
        close_1h = safe_float(df_1h["Close"].iloc[-1], None)
        ema50_1h = safe_float(df_1h["EMA50"].iloc[-1], None)
        if isinstance(close_1h, (int, float)) and isinstance(ema50_1h, (int, float)):
            trend_1h = "UP" if close_1h >= ema50_1h else "DOWN"

    pattern_mode = getattr(config, "PRICE_ACTION_15M_ENTRY_CONFIRMATION_MODE", "engulfing_pinbar")
    pattern_lookback = getattr(config, "PRICE_ACTION_15M_PATTERN_LOOKBACK_BARS", 3)
    pattern_ok_buy, pattern_name_buy, pattern_idx_buy = recent_pattern_confirmation(
        df, len(df) - 1, "BUY", lookback_bars=pattern_lookback, mode=pattern_mode
    )
    pattern_ok_sell, pattern_name_sell, pattern_idx_sell = recent_pattern_confirmation(
        df, len(df) - 1, "SELL", lookback_bars=pattern_lookback, mode=pattern_mode
    )

    chart_pattern = _price_action_chart_pattern(
        df,
        pivot_highs,
        pivot_lows,
        safe_float,
        sr_lookback=sr_lookback,
        tolerance_pct=buffer_pct,
    )
    wyckoff_phase = _price_action_wyckoff_phase(market_structure, current_price, recent_low, recent_high)
    nearest_support = order_blocks.get("nearest_support") if isinstance(order_blocks, dict) else None
    nearest_resistance = order_blocks.get("nearest_resistance") if isinstance(order_blocks, dict) else None
    demand_zone = nearest_support if isinstance(nearest_support, (int, float)) else recent_low
    supply_zone = nearest_resistance if isinstance(nearest_resistance, (int, float)) else recent_high

    bull_score = 0.0
    bear_score = 0.0
    bull_reasons = []
    bear_reasons = []

    if market_structure == "UPTREND":
        bull_score += 18.0
        bull_reasons.append("โครงสร้างราคาเป็น HH/HL ตาม Dow Theory")
    elif market_structure == "DOWNTREND":
        bear_score += 18.0
        bear_reasons.append("โครงสร้างราคาเป็น LH/LL ตาม Dow Theory")
    else:
        bull_reasons.append("ตลาดกำลังแกว่งในกรอบ รออ่านพฤติกรรมที่ขอบโซน")
        bear_reasons.append("ตลาดกำลังแกว่งในกรอบ รออ่านพฤติกรรมที่ขอบโซน")

    if trend_1h == "UP":
        bull_score += 12.0
        bull_reasons.append("เทรนด์ 1H หนุนฝั่ง BUY")
    elif trend_1h == "DOWN":
        bear_score += 12.0
        bear_reasons.append("เทรนด์ 1H หนุนฝั่ง SELL")

    if isinstance(adx_value, (int, float)) and adx_value >= min_adx:
        if market_structure == "UPTREND":
            bull_score += 6.0
            bull_reasons.append(f"ADX {adx_value:.1f} หนุนแนวโน้มขึ้น")
        elif market_structure == "DOWNTREND":
            bear_score += 6.0
            bear_reasons.append(f"ADX {adx_value:.1f} หนุนแนวโน้มลง")

    if pattern_ok_buy:
        bull_score += 10.0
        bull_reasons.append(f"มีแท่งกลับตัวฝั่ง BUY: {pattern_name_buy}")
    if pattern_ok_sell:
        bear_score += 10.0
        bear_reasons.append(f"มีแท่งกลับตัวฝั่ง SELL: {pattern_name_sell}")

    if isinstance(demand_zone, (int, float)):
        demand_gap_pct = ((current_price - float(demand_zone)) / current_price) * 100.0
        if 0.0 <= demand_gap_pct <= zone_proximity_pct:
            bull_score += 12.0
            bull_reasons.append("ราคาอยู่ใกล้ Demand/Support zone")
    if isinstance(supply_zone, (int, float)):
        supply_gap_pct = ((float(supply_zone) - current_price) / current_price) * 100.0
        if 0.0 <= supply_gap_pct <= zone_proximity_pct:
            bear_score += 12.0
            bear_reasons.append("ราคาอยู่ใกล้ Supply/Resistance zone")

    if isinstance(recent_high, (int, float)) and current_price > float(recent_high) * (1.0 + max(0.0010, buffer_pct / 100.0)):
        bull_score += 10.0
        bull_reasons.append("เกิด breakout เหนือแนวต้านเดิม")
    if isinstance(recent_low, (int, float)) and current_price < float(recent_low) * (1.0 - max(0.0010, buffer_pct / 100.0)):
        bear_score += 10.0
        bear_reasons.append("เกิด breakdown หลุดแนวรับเดิม")

    chart_pattern_label = str(chart_pattern.get("label") or "").strip()
    chart_pattern_signal = str(chart_pattern.get("signal") or "").upper().strip()
    chart_pattern_score = safe_float(chart_pattern.get("score"), 0.0)
    if chart_pattern_signal == "BUY":
        bull_score += chart_pattern_score
        bull_reasons.append(f"รูปแบบกราฟสนับสนุนฝั่ง BUY: {chart_pattern_label}")
    elif chart_pattern_signal == "SELL":
        bear_score += chart_pattern_score
        bear_reasons.append(f"รูปแบบกราฟสนับสนุนฝั่ง SELL: {chart_pattern_label}")

    pred_dir = str((prediction or {}).get("direction") or "").upper().strip()
    pred_prob = safe_float((prediction or {}).get("probability"), None)
    if pred_dir == "UP":
        bull_score += 8.0
        bull_reasons.append("Forecast ภาพรวมเอนขึ้น")
        if isinstance(pred_prob, (int, float)):
            bull_score += max(0.0, min(6.0, (pred_prob - 50.0) * 0.18))
    elif pred_dir == "DOWN":
        bear_score += 8.0
        bear_reasons.append("Forecast ภาพรวมเอนลง")
        if isinstance(pred_prob, (int, float)):
            bear_score += max(0.0, min(6.0, (pred_prob - 50.0) * 0.18))

    phase_text = str(phase_status or "").lower()
    if "constructive" in phase_text or "accumulation" in phase_text:
        bull_score += 5.0
        bull_reasons.append("บริบทกว้างยังมีลักษณะสะสมหรือเสริมแรง")
    if "destructive" in phase_text or "decay" in phase_text:
        bear_score += 5.0
        bear_reasons.append("บริบทกว้างยังมีลักษณะอ่อนแรงหรือกระจายตัว")

    ema200_now = safe_float(df["EMA200"].iloc[-1], None)
    if require_ema200 and isinstance(ema200_now, (int, float)):
        if current_price >= ema200_now:
            bull_score += 4.0
        else:
            bear_score += 4.0

    signal = "WAIT"
    score = 0.0
    reasons = []
    detected_pattern = "None"
    pattern_idx = None
    role_reversal = ""
    if bull_score >= min_score and bull_score >= bear_score + 4.0:
        signal = "BUY"
        score = float(bull_score)
        reasons = bull_reasons
        detected_pattern = pattern_name_buy if pattern_ok_buy else "None"
        pattern_idx = pattern_idx_buy
        if chart_pattern_signal == "BUY" and chart_pattern_label:
            role_reversal = "Breakout & Retest ฝั่ง BUY"
    elif bear_score >= min_score and bear_score >= bull_score + 4.0:
        signal = "SELL"
        score = float(bear_score)
        reasons = bear_reasons
        detected_pattern = pattern_name_sell if pattern_ok_sell else "None"
        pattern_idx = pattern_idx_sell
        if chart_pattern_signal == "SELL" and chart_pattern_label:
            role_reversal = "Breakdown & Retest ฝั่ง SELL"

    if require_pattern and signal == "BUY" and not pattern_ok_buy and chart_pattern_signal != "BUY":
        signal = "WAIT"
    if require_pattern and signal == "SELL" and not pattern_ok_sell and chart_pattern_signal != "SELL":
        signal = "WAIT"

    proxy_metrics = price_action_proxy_metrics(item, signal)
    proxy_wr = proxy_metrics.get("win_rate_pct")
    proxy_exp = proxy_metrics.get("expectancy_rr")
    proxy_trades = proxy_metrics.get("trades")
    proxy_sources = proxy_metrics.get("source_labels") or []
    proxy_source_count = int(proxy_metrics.get("source_count") or 0)
    proxy_ok = signal in ("BUY", "SELL") and proxy_source_count >= min_proxy_sources
    if proxy_ok and isinstance(proxy_wr, (int, float)) and float(proxy_wr) < min_proxy_wr:
        proxy_ok = False
    if proxy_ok and isinstance(proxy_exp, (int, float)) and float(proxy_exp) < min_proxy_exp:
        proxy_ok = False
    if proxy_ok and isinstance(proxy_trades, (int, float)) and float(proxy_trades) < min_proxy_trades:
        proxy_ok = False
    if signal not in ("BUY", "SELL") or not proxy_ok:
        signal = "WAIT"

    confidence = None
    if signal in ("BUY", "SELL"):
        confidence = max(50.0, min(94.0, score))
        if isinstance(proxy_wr, (int, float)):
            confidence = min(96.0, confidence + max(-4.0, min(6.0, (float(proxy_wr) - 55.0) * 0.20)))

    default_stop = None
    entry_price = current_price
    if signal == "BUY":
        stop_candidates = []
        if isinstance(demand_zone, (int, float)) and float(demand_zone) < current_price:
            stop_candidates.append(float(demand_zone))
        if pivot_lows:
            last_pivot_low = float(pivot_lows[-1][1])
            if last_pivot_low < current_price:
                stop_candidates.append(last_pivot_low)
        if isinstance(atr_value, (int, float)) and atr_value > 0:
            stop_candidates.append(current_price - (float(atr_value) * stop_atr_mult))
        default_stop = min(stop_candidates) if stop_candidates else None
    elif signal == "SELL":
        stop_candidates = []
        if isinstance(supply_zone, (int, float)) and float(supply_zone) > current_price:
            stop_candidates.append(float(supply_zone))
        if pivot_highs:
            last_pivot_high = float(pivot_highs[-1][1])
            if last_pivot_high > current_price:
                stop_candidates.append(last_pivot_high)
        if isinstance(atr_value, (int, float)) and atr_value > 0:
            stop_candidates.append(current_price + (float(atr_value) * stop_atr_mult))
        default_stop = max(stop_candidates) if stop_candidates else None

    stop_loss = None
    if signal in ("BUY", "SELL") and isinstance(default_stop, (int, float)):
        stop_loss = candle_based_risk(
            df,
            pattern_idx if isinstance(pattern_idx, int) else len(df) - 1,
            signal,
            atr_value,
            default_stop,
            buffer_atr=candle_stop_buffer,
        )
    if not isinstance(stop_loss, (int, float)):
        stop_loss = default_stop

    take_profit = None
    if signal in ("BUY", "SELL") and isinstance(stop_loss, (int, float)):
        risk_dist = abs(float(entry_price) - float(stop_loss))
        if risk_dist > 0:
            if signal == "BUY":
                take_profit = float(entry_price) + (risk_dist * tp_mult)
                if isinstance(supply_zone, (int, float)) and float(supply_zone) > float(entry_price):
                    take_profit = max(float(entry_price) + (risk_dist * 1.2), min(float(take_profit), float(supply_zone)))
            else:
                take_profit = float(entry_price) - (risk_dist * tp_mult)
                if isinstance(demand_zone, (int, float)) and float(demand_zone) < float(entry_price):
                    take_profit = min(float(entry_price) - (risk_dist * 1.2), max(float(take_profit), float(demand_zone)))

    last_signal_time = None
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            last_signal_time = df.index[-1].strftime("%Y-%m-%d %H:%M")
    except Exception:
        last_signal_time = None

    setup_bits = [structure_label]
    if chart_pattern_label:
        setup_bits.append(chart_pattern_label)
    elif detected_pattern and detected_pattern != "None":
        setup_bits.append(detected_pattern)

    return {
        "strategy": "Price Action 15m",
        "signal": signal,
        "alert": bool(signal in ("BUY", "SELL") and isinstance(confidence, (int, float)) and confidence >= min_conf),
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
        "predicted_win_prob": float(confidence) if isinstance(confidence, (int, float)) else None,
        "historical_win_rate": proxy_wr,
        "historical_avg_rr": proxy_exp,
        "historical_trades": proxy_trades,
        "proxy_sources": proxy_sources,
        "proxy_source_count": proxy_source_count,
        "score": float(score) if signal in ("BUY", "SELL") else 0.0,
        "setup": signal if signal in ("BUY", "SELL") else "WAIT",
        "setup_label": " | ".join([bit for bit in setup_bits if bit]),
        "market_structure": market_structure,
        "trend_1h": trend_1h,
        "wyckoff_phase": wyckoff_phase,
        "chart_pattern": chart_pattern_label,
        "role_reversal": role_reversal,
        "detected_pattern": detected_pattern,
        "entry_price": float(entry_price),
        "current_price": float(current_price),
        "price": float(current_price),
        "stop_loss": float(stop_loss) if isinstance(stop_loss, (int, float)) else None,
        "take_profit": float(take_profit) if isinstance(take_profit, (int, float)) else None,
        "nearest_support": float(nearest_support) if isinstance(nearest_support, (int, float)) else None,
        "nearest_resistance": float(nearest_resistance) if isinstance(nearest_resistance, (int, float)) else None,
        "demand_zone": float(demand_zone) if isinstance(demand_zone, (int, float)) else None,
        "supply_zone": float(supply_zone) if isinstance(supply_zone, (int, float)) else None,
        "adx": float(adx_value) if isinstance(adx_value, (int, float)) else None,
        "atr": float(atr_value) if isinstance(atr_value, (int, float)) else None,
        "last_signal_time": last_signal_time,
        "phase_status": phase_status,
        "forecast_direction": pred_dir or None,
        "forecast_probability": float(pred_prob) if isinstance(pred_prob, (int, float)) else None,
        "reasons": reasons[:6],
    }

