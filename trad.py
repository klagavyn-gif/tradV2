from flask import Flask, render_template, request, jsonify
import argparse
import json
import logging
import os
import pandas as pd
import yfinance as yf
import numpy as np
import certifi
import math
import time
import threading
import pytz
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from curl_cffi import requests as curl_requests
import config
try:
    import requests as http_requests
except Exception:
    http_requests = None

app = Flask(__name__)
if getattr(config, "SECRET_KEY", ""):
    app.config["SECRET_KEY"] = config.SECRET_KEY

logger = logging.getLogger(__name__)

# Helper to get current Thai time (naive)
def get_thai_now():
    return datetime.now(pytz.timezone('Asia/Bangkok')).replace(tzinfo=None)


VALID_PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', 'ytd', '1h', '15m']
EMA_OPTIMIZER_BEST = {}
EMA_CROSS_15M_OPT_CACHE = {}

_THREAD_LOCAL = threading.local()


class _TTLCache:
    def __init__(self, maxsize, ttl_seconds):
        self.maxsize = int(maxsize) if isinstance(maxsize, int) and maxsize > 0 else 256
        self.ttl_seconds = float(ttl_seconds) if isinstance(ttl_seconds, (int, float)) and ttl_seconds > 0 else 180.0
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key, value, ttl_seconds=None):
        now = time.time()
        ttl = self.ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        if ttl <= 0:
            return
        expires_at = now + ttl
        with self._lock:
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)


_YF_CACHE = _TTLCache(
    maxsize=getattr(config, "YF_CACHE_MAXSIZE", 256),
    ttl_seconds=getattr(config, "YF_CACHE_TTL_SECONDS", 180),
)
_YF_INFO_CACHE = _TTLCache(
    maxsize=max(64, int(getattr(config, "YF_CACHE_MAXSIZE", 256) // 4)),
    ttl_seconds=getattr(config, "YF_INFO_CACHE_TTL_SECONDS", 6 * 60 * 60),
)
_YF_UNIVERSE_CACHE = _TTLCache(
    maxsize=128,
    ttl_seconds=getattr(config, "YF_UNIVERSE_CACHE_TTL_SECONDS", 15 * 60),
)

_ANALYZE_EXECUTOR = ThreadPoolExecutor(max_workers=int(getattr(config, "ANALYZE_MAX_WORKERS", 5)))
_STATS_CACHE = _TTLCache(
    maxsize=int(getattr(config, "STATS_CACHE_MAXSIZE", 128)),
    ttl_seconds=int(getattr(config, "STATS_CACHE_TTL_SECONDS", 10 * 60)),
)
_TELEGRAM_ALERT_CACHE = _TTLCache(
    maxsize=256,
    ttl_seconds=int(getattr(config, "TELEGRAM_ALERT_TTL_SECONDS", 1800)),
)


def _get_config_warnings():
    warnings = []
    debug = bool(getattr(config, "FLASK_DEBUG", True))
    if not debug and not getattr(config, "SECRET_KEY", ""):
        warnings.append("SECRET_KEY ยังไม่ถูกตั้งค่า (ไม่ควรใช้ production)")
    if getattr(config, "HTTP_VERIFY", True) is False:
        warnings.append("HTTP_VERIFY=false (เสี่ยงด้านความปลอดภัย)")
    max_workers = getattr(config, "ANALYZE_MAX_WORKERS", 5)
    try:
        max_workers = int(max_workers)
        if max_workers < 1:
            warnings.append("ANALYZE_MAX_WORKERS ควร >= 1")
    except Exception:
        warnings.append("ANALYZE_MAX_WORKERS ไม่ถูกต้อง")
    return warnings


def send_telegram_alert(message):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id or not message:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        if http_requests is not None:
            resp = http_requests.post(url, json=payload, timeout=10)
            return bool(getattr(resp, "ok", False))
        session = _create_curl_session()
        resp = session.post(url, json=payload, timeout=10)
        status_code = int(getattr(resp, "status_code", 0))
        ok = getattr(resp, "ok", None)
        if ok is not None:
            return bool(ok)
        return 200 <= status_code < 300
    except Exception as e:
        logger.warning("Telegram alert failed: %s", e, exc_info=True)
        return False


def _normalize_confidence(value):
    try:
        conf = float(value)
    except Exception:
        return None
    if conf <= 1:
        conf = conf * 100.0
    if conf < 0:
        conf = 0.0
    if conf > 100:
        conf = 100.0
    return conf


def _format_price_value(value):
    try:
        val = float(value)
    except Exception:
        return None
    abs_val = abs(val)
    if abs_val >= 100:
        return f"{val:,.2f}"
    if abs_val >= 1:
        return f"{val:,.2f}"
    return f"{val:,.6f}"


def _collect_alert_sources(item, min_conf):
    sources = []

    def add(label, plan, setup_key="setup", conf_key="confidence"):
        if not isinstance(plan, dict):
            return
        conf = _normalize_confidence(plan.get(conf_key))
        if conf is None or conf < min_conf:
            return
        setup = plan.get(setup_key)
        text = label
        if isinstance(setup, str) and setup.strip():
            text = f"{label} {setup.strip()}"
        sources.append((conf, text))

    add("ShortTerm 15m", item.get("short_term_15m"))
    add("Sniper 15m", item.get("sniper_15m"))
    add("Quantum 15m", item.get("quantum_15m"))
    add("EMA Cross 15m", item.get("ema_cross_15m"), setup_key="signal")
    add("ActionZone 15m", item.get("actionzone_15m"), setup_key="signal")
    add("Crypto Reversal 15m", item.get("crypto_reversal_15m"))
    sources.sort(key=lambda x: x[0], reverse=True)
    return [f"{text} ({conf:.0f}%)" for conf, text in sources[:3]]


def _get_best_confidence(item):
    best = None
    for plan, setup_key, conf_key in (
        (item.get("short_term_15m"), "setup", "confidence"),
        (item.get("sniper_15m"), "setup", "confidence"),
        (item.get("quantum_15m"), "setup", "confidence"),
        (item.get("ema_cross_15m"), "signal", "confidence"),
        (item.get("actionzone_15m"), "signal", "confidence"),
        (item.get("crypto_reversal_15m"), "setup", "confidence"),
    ):
        if not isinstance(plan, dict):
            continue
        conf = _normalize_confidence(plan.get(conf_key))
        if conf is None:
            continue
        if best is None or conf > best:
            best = conf
    return best


def _build_telegram_message(item, signal, best_conf, sources):
    emoji = "🟢" if signal == "BUY" else "🔴"
    symbol = normalize_symbol(item.get("symbol") or "")
    name = str(item.get("name") or "").strip()
    lines = [f"{emoji} {signal}"]
    line_symbol = f"สินทรัพย์: {symbol}"
    if name:
        line_symbol += f" • {name}"
    lines.append(line_symbol)
    price = item.get("price")
    change = item.get("change")
    price_text = _format_price_value(price)
    if price_text:
        if isinstance(change, (int, float)):
            lines.append(f"ราคา: {price_text} ({change:+.2f}%)")
        else:
            lines.append(f"ราคา: {price_text}")
    if best_conf is not None:
        lines.append(f"ความมั่นใจ: {best_conf:.0f}%")
    if sources:
        lines.append("แหล่งสัญญาณ: " + ", ".join(sources))
    lines.append("เวลา: " + get_thai_now().strftime("%Y-%m-%d %H:%M"))
    return "\n".join(lines)


def _build_actionzone_message(item, az_plan):
    signal = str(az_plan.get("raw_signal") or az_plan.get("signal") or "").upper()
    emoji = "🟢" if signal == "BUY" else "🔴"
    symbol = normalize_symbol(item.get("symbol") or "")
    name = str(item.get("name") or "").strip()
    lines = [f"{emoji} ActionZone 15m {signal}"]
    line_symbol = f"สินทรัพย์: {symbol}"
    if name:
        line_symbol += f" • {name}"
    lines.append(line_symbol)
    price = item.get("price")
    change = item.get("change")
    price_text = _format_price_value(price)
    if price_text:
        if isinstance(change, (int, float)):
            lines.append(f"ราคา: {price_text} ({change:+.2f}%)")
        else:
            lines.append(f"ราคา: {price_text}")
    zone = az_plan.get("zone")
    trend = az_plan.get("trend_1h")
    conf = az_plan.get("confidence")
    if zone or trend:
        parts = []
        if zone:
            parts.append(f"โซน: {zone}")
        if trend:
            parts.append(f"เทรนด์ 1H: {trend}")
        lines.append(" • ".join(parts))
    if isinstance(conf, (int, float)):
        lines.append(f"ความมั่นใจ: {float(conf):.0f}%")
    bars = az_plan.get("bars_since_signal")
    if isinstance(bars, (int, float)):
        lines.append(f"แท่งนับจากสัญญาณ: {int(bars)}")
    lines.append("เวลา: " + get_thai_now().strftime("%Y-%m-%d %H:%M"))
    return "\n".join(lines)


def _notify_telegram_from_results(results):
    min_conf = getattr(config, "TELEGRAM_ALERT_MIN_CONFIDENCE", 75.0)
    max_per_run = getattr(config, "TELEGRAM_ALERT_MAX_PER_RUN", 5)
    try:
        min_conf = float(min_conf)
    except Exception:
        min_conf = 75.0
    try:
        max_per_run = int(max_per_run)
    except Exception:
        max_per_run = 5
    sent = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("error"):
            continue
        symbol = normalize_symbol(item.get("symbol") or "")
        if not symbol:
            continue
        if sent >= max_per_run:
            break
        az_plan = item.get("actionzone_15m")
        if isinstance(az_plan, dict):
            az_signal = str(az_plan.get("raw_signal") or az_plan.get("signal") or "").upper()
            if az_signal in ("BUY", "SELL") and az_plan.get("alert"):
                az_conf = _normalize_confidence(az_plan.get("confidence"))
                if az_conf is not None and az_conf >= min_conf:
                    az_key = f"AZ15|{symbol}|{az_signal}"
                    if not _TELEGRAM_ALERT_CACHE.get(az_key):
                        az_message = _build_actionzone_message(item, az_plan)
                        if send_telegram_alert(az_message):
                            _TELEGRAM_ALERT_CACHE.set(az_key, True)
                            sent += 1
                            if sent >= max_per_run:
                                break
        if sent >= max_per_run:
            break
        signal = str(item.get("signal") or "").upper()
        if signal not in ("BUY", "SELL"):
            continue
        best_conf = _get_best_confidence(item)
        if best_conf is None or best_conf < min_conf:
            continue
        cache_key = f"{symbol}|{signal}"
        if _TELEGRAM_ALERT_CACHE.get(cache_key):
            continue
        sources = _collect_alert_sources(item, min_conf)
        message = _build_telegram_message(item, signal, best_conf, sources)
        if send_telegram_alert(message):
            _TELEGRAM_ALERT_CACHE.set(cache_key, True)
            sent += 1
            if sent >= max_per_run:
                break
    return sent


def _get_thread_curl_session():
    sess = getattr(_THREAD_LOCAL, "curl_session", None)
    if sess is None:
        sess = _create_curl_session()
        _THREAD_LOCAL.curl_session = sess
    return sess


def _normalize_df_index(df):
    if df is None or getattr(df, "empty", True):
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is not None:
                # Convert to Thai time (UTC+7) before making naive
                df.index = df.index.tz_convert(pytz.timezone('Asia/Bangkok'))
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
    return df


def get_yf_history(symbol, period, interval=None, auto_adjust=True, cache_ttl_seconds=None):
    sym = normalize_symbol(symbol)
    if not sym:
        return None
    key = ("hist", sym, str(period or ""), str(interval or ""), bool(auto_adjust))
    cached = _YF_CACHE.get(key)
    if isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached.copy()
    try:
        session = _get_thread_curl_session()
        ticker = yf.Ticker(sym, session=session)
        if interval:
            df = ticker.history(period=period, interval=interval, auto_adjust=auto_adjust)
        else:
            df = ticker.history(period=period, auto_adjust=auto_adjust)
        if df is None or df.empty:
            return None
        df = _normalize_df_index(df)
        _YF_CACHE.set(key, df, ttl_seconds=cache_ttl_seconds)
        return df.copy()
    except Exception as e:
        logger.warning("Error fetching %s: %s", sym, e, exc_info=True)
        return None


def _get_http_verify_setting():
    verify = getattr(config, "HTTP_VERIFY", True)
    if not verify:
        return False
    ca_bundle = getattr(config, "HTTP_CA_BUNDLE", "")
    if isinstance(ca_bundle, str) and ca_bundle.strip():
        return ca_bundle.strip()
    try:
        return certifi.where()
    except Exception:
        return True


def _create_curl_session():
    impersonate = getattr(config, "CURL_IMPERSONATE", "chrome110")
    return curl_requests.Session(verify=_get_http_verify_setting(), impersonate=impersonate)


def _ema_cross_15m_get_cached(symbol):
    sym = normalize_symbol(symbol)
    cached = EMA_CROSS_15M_OPT_CACHE.get(sym)
    if not isinstance(cached, dict):
        return None
    ttl = getattr(config, "EMA_CROSS_15M_CACHE_TTL_SECONDS", 0)
    try:
        ttl = int(ttl)
    except Exception:
        ttl = 0
    if ttl <= 0:
        return None
    ts = cached.get("cached_at")
    if not isinstance(ts, datetime):
        return None
    age = (get_thai_now() - ts).total_seconds()
    if age < 0 or age > ttl:
        return None
    return cached


def _ema_cross_15m_set_cached(symbol, payload):
    sym = normalize_symbol(symbol)
    if not isinstance(payload, dict):
        return
    payload = payload.copy()
    payload["cached_at"] = get_thai_now()
    EMA_CROSS_15M_OPT_CACHE[sym] = payload


def _ema_cross_15m_prepare_df(raw_df):
    if raw_df is None or getattr(raw_df, "empty", True):
        return None
    if not all(c in raw_df.columns for c in ("Open", "High", "Low", "Close", "Volume")):
        return None
    df = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(window=14).mean()
    df["Vol_Avg"] = df["Volume"].rolling(window=20).mean()
    return df


def _backtest_ema_cross_15m(df, fast_len, slow_len, tp_mult=5.0, max_forward=64, return_rr_list=False):
    if df is None or df.empty:
        return None
    try:
        fast_len = int(fast_len)
        slow_len = int(slow_len)
        tp_mult = float(tp_mult)
        max_forward = int(max_forward)
    except Exception:
        return None
    if fast_len < 2 or slow_len < 2 or fast_len >= slow_len:
        return None
    if tp_mult <= 0 or max_forward < 1:
        return None
    close = df["Close"].astype(float)
    ema_fast = close.ewm(span=fast_len, adjust=False).mean()
    ema_slow = close.ewm(span=slow_len, adjust=False).mean()
    zone = np.where(ema_fast > ema_slow, 1, -1)
    zone_change = pd.Series(zone, index=df.index).diff().fillna(0)
    wins = 0
    losses = 0
    total_rr = 0.0
    rr_list = [] if return_rr_list else None
    for i in range(1, len(df)):
        zc = zone_change.iloc[i]
        if zc not in (2, -2):
            continue
        atr_i = df["ATR"].iloc[i]
        vol_avg_i = df["Vol_Avg"].iloc[i]
        if pd.isna(atr_i) or atr_i <= 0 or pd.isna(vol_avg_i) or vol_avg_i <= 0:
            continue
        entry_i = close.iloc[i]
        if pd.isna(entry_i) or entry_i <= 0:
            continue
        direction_i = "BUY" if zc == 2 else "SELL"
        risk_i = float(atr_i)
        sl_i = entry_i - risk_i if direction_i == "BUY" else entry_i + risk_i
        tp_i = entry_i + risk_i * tp_mult if direction_i == "BUY" else entry_i - risk_i * tp_mult
        outcome = None
        end_j = min(len(df), i + 1 + max_forward)
        for j in range(i + 1, end_j):
            high_j = df["High"].iloc[j]
            low_j = df["Low"].iloc[j]
            if direction_i == "BUY":
                if low_j <= sl_i:
                    outcome = "loss"
                    break
                if high_j >= tp_i:
                    outcome = "win"
                    break
            else:
                if high_j >= sl_i:
                    outcome = "loss"
                    break
                if low_j <= tp_i:
                    outcome = "win"
                    break
        if outcome == "win":
            wins += 1
            total_rr += tp_mult
            if rr_list is not None:
                rr_list.append(float(tp_mult))
        elif outcome == "loss":
            losses += 1
            total_rr -= 1.0
            if rr_list is not None:
                rr_list.append(-1.0)
    total_trades = wins + losses
    if total_trades <= 0:
        payload = {
            "fast_len": fast_len,
            "slow_len": slow_len,
            "trades": 0,
            "win_rate_pct": None,
            "avg_rr": None,
            "expectancy_rr": None,
        }
        if rr_list is not None:
            payload["rr_list"] = []
        return payload
    win_rate = (wins / total_trades) * 100.0
    avg_rr = total_rr / total_trades
    payload = {
        "fast_len": fast_len,
        "slow_len": slow_len,
        "trades": int(total_trades),
        "win_rate_pct": float(win_rate),
        "avg_rr": float(avg_rr),
        "expectancy_rr": float(avg_rr),
    }
    if rr_list is not None:
        payload["rr_list"] = rr_list
    return payload


def optimize_best_ema_cross_15m(symbol, df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    enable = getattr(config, "EMA_CROSS_15M_ENABLE_OPTIMIZATION", True)
    if not enable:
        return None
    fast_min = getattr(config, "EMA_CROSS_15M_FAST_MIN", 6)
    fast_max = getattr(config, "EMA_CROSS_15M_FAST_MAX", 24)
    fast_step = getattr(config, "EMA_CROSS_15M_FAST_STEP", 2)
    slow_min = getattr(config, "EMA_CROSS_15M_SLOW_MIN", 18)
    slow_max = getattr(config, "EMA_CROSS_15M_SLOW_MAX", 80)
    slow_step = getattr(config, "EMA_CROSS_15M_SLOW_STEP", 2)
    min_trades = getattr(config, "EMA_CROSS_15M_MIN_TRADES", 8)
    tp_mult = getattr(config, "EMA_CROSS_15M_TP_MULT", 5.0)
    max_forward = getattr(config, "EMA_CROSS_15M_MAX_FORWARD_BARS", 64)
    try:
        fast_min = int(fast_min)
        fast_max = int(fast_max)
        fast_step = int(fast_step)
        slow_min = int(slow_min)
        slow_max = int(slow_max)
        slow_step = int(slow_step)
        min_trades = int(min_trades)
        tp_mult = float(tp_mult)
        max_forward = int(max_forward)
    except Exception:
        return None
    if fast_step < 1 or slow_step < 1:
        return None
    if fast_min < 2:
        fast_min = 2
    if slow_min < 2:
        slow_min = 2
    if fast_max < fast_min:
        fast_max = fast_min
    if slow_max < slow_min:
        slow_max = slow_min
    if min_trades < 1:
        min_trades = 1
    if tp_mult <= 0:
        tp_mult = 5.0
    if max_forward < 1:
        max_forward = 64
    best = None
    evaluated = 0
    for fast_len in range(fast_min, fast_max + 1, fast_step):
        for slow_len in range(slow_min, slow_max + 1, slow_step):
            if fast_len >= slow_len:
                continue
            r = _backtest_ema_cross_15m(df, fast_len, slow_len, tp_mult=tp_mult, max_forward=max_forward)
            evaluated += 1
            if not r:
                continue
            trades = r.get("trades")
            exp_rr = r.get("expectancy_rr")
            win_rate = r.get("win_rate_pct")
            if not isinstance(trades, int) or trades < min_trades:
                continue
            if not isinstance(exp_rr, (int, float)):
                continue
            key = (float(exp_rr), float(win_rate) if isinstance(win_rate, (int, float)) else -1e9, float(trades))
            if best is None:
                best = r.copy()
                best["_key"] = key
            else:
                if key > best.get("_key", (-1e18, -1e18, -1e18)):
                    best = r.copy()
                    best["_key"] = key
    if best is None:
        return {
            "symbol": normalize_symbol(symbol),
            "best": None,
            "evaluated": int(evaluated),
            "computed_at": get_thai_now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    best.pop("_key", None)
    return {
        "symbol": normalize_symbol(symbol),
        "best": best,
        "evaluated": int(evaluated),
        "computed_at": get_thai_now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _wilson_ci_95(wins, n):
    try:
        wins = int(wins)
        n = int(n)
    except Exception:
        return None
    if n <= 0 or wins < 0 or wins > n:
        return None
    z = 1.96
    phat = wins / n
    denom = 1.0 + (z * z / n)
    center = (phat + (z * z) / (2.0 * n)) / denom
    adj = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z / (4.0 * n * n)))
    lo = max(0.0, center - adj)
    hi = min(1.0, center + adj)
    return {"low_pct": lo * 100.0, "high_pct": hi * 100.0}


def _summarize_rr_list(rr_list, tp_mult):
    if not isinstance(rr_list, list) or not rr_list:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": None,
            "win_rate_ci95": None,
            "avg_rr": None,
            "expectancy_rr": None,
            "expectancy_ci95": None,
            "breakeven_win_rate_pct": (100.0 / (float(tp_mult) + 1.0)) if isinstance(tp_mult, (int, float)) and tp_mult > 0 else None,
        }
    vals = np.array(rr_list, dtype=float)
    n = int(vals.size)
    wins = int(np.sum(vals > 0))
    losses = int(np.sum(vals < 0))
    mean_rr = float(np.mean(vals)) if n else None
    std_rr = float(np.std(vals, ddof=1)) if n > 1 else None
    win_rate = (wins / n) * 100.0 if n else None
    win_ci = _wilson_ci_95(wins, n)
    exp_ci = None
    if std_rr is not None and n > 1:
        se = std_rr / math.sqrt(n)
        exp_ci = {"low": mean_rr - 1.96 * se, "high": mean_rr + 1.96 * se}
    breakeven = None
    if isinstance(tp_mult, (int, float)) and tp_mult > 0:
        breakeven = 100.0 / (float(tp_mult) + 1.0)
    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": float(win_rate) if win_rate is not None else None,
        "win_rate_ci95": win_ci,
        "avg_rr": mean_rr,
        "expectancy_rr": mean_rr,
        "expectancy_ci95": exp_ci,
        "breakeven_win_rate_pct": breakeven,
    }


def _walk_forward_ema_cross_15m(symbol, raw_df, folds, tp_mult, max_forward, min_train_bars):
    df = _ema_cross_15m_prepare_df(raw_df)
    if df is None or df.empty:
        return {"symbol": normalize_symbol(symbol), "error": "No data"}
    try:
        folds = int(folds)
        tp_mult = float(tp_mult)
        max_forward = int(max_forward)
        min_train_bars = int(min_train_bars)
    except Exception:
        return {"symbol": normalize_symbol(symbol), "error": "Invalid parameters"}
    if folds < 1:
        folds = 1
    if folds > 6:
        folds = 6
    if tp_mult <= 0:
        tp_mult = float(getattr(config, "EMA_CROSS_15M_TP_MULT", 5.0))
    if max_forward < 8:
        max_forward = int(getattr(config, "EMA_CROSS_15M_MAX_FORWARD_BARS", 64))
    if min_train_bars < 120:
        min_train_bars = 120

    n_total = len(df)
    remaining = n_total - min_train_bars
    if remaining < 120:
        return {"symbol": normalize_symbol(symbol), "error": "Not enough data", "bars": int(n_total)}

    test_bars = max(60, int(remaining / (folds + 1)))
    folds_used = []
    test_rr_all = []
    for i in range(folds):
        train_end = min_train_bars + i * test_bars
        test_end = train_end + test_bars
        if test_end > n_total:
            break
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        opt = optimize_best_ema_cross_15m(symbol, train_df)
        best = opt.get("best") if isinstance(opt, dict) else None
        if not isinstance(best, dict):
            continue
        fast_len = best.get("fast_len")
        slow_len = best.get("slow_len")
        if not isinstance(fast_len, int) or not isinstance(slow_len, int):
            continue

        train_bt = _backtest_ema_cross_15m(train_df, fast_len, slow_len, tp_mult=tp_mult, max_forward=max_forward, return_rr_list=True)
        test_bt = _backtest_ema_cross_15m(test_df, fast_len, slow_len, tp_mult=tp_mult, max_forward=max_forward, return_rr_list=True)
        train_rr = train_bt.get("rr_list") if isinstance(train_bt, dict) else None
        test_rr = test_bt.get("rr_list") if isinstance(test_bt, dict) else None
        train_summary = _summarize_rr_list(train_rr, tp_mult)
        test_summary = _summarize_rr_list(test_rr, tp_mult)
        if isinstance(test_rr, list) and test_rr:
            test_rr_all.extend(test_rr)

        folds_used.append(
            {
                "fold": i + 1,
                "train_bars": int(len(train_df)),
                "test_bars": int(len(test_df)),
                "best": {"fast_len": fast_len, "slow_len": slow_len},
                "train": train_summary,
                "test": test_summary,
                "evaluated": int(opt.get("evaluated")) if isinstance(opt.get("evaluated"), int) else None,
            }
        )

    overall = _summarize_rr_list(test_rr_all, tp_mult)
    return {
        "symbol": normalize_symbol(symbol),
        "bars": int(n_total),
        "tp_mult": float(tp_mult),
        "max_forward_bars": int(max_forward),
        "folds_requested": int(folds),
        "folds_used": int(len(folds_used)),
        "test_bars_per_fold": int(test_bars),
        "folds": folds_used,
        "overall_test": overall,
        "computed_at": get_thai_now().strftime("%Y-%m-%d %H:%M:%S"),
    }

def is_crypto_symbol(symbol):
    if not symbol:
        return False
    s = symbol.upper()
    if s == 'DX-Y.NYB':
        return False
    return s.endswith('-USD')

def normalize_symbol(symbol):
    if not symbol:
        return ""
    return str(symbol).strip().upper()

# --- Particle A Logic Engine ---
class ParticleAAnalyzer:
    """
    เครื่องยนต์คำนวณตามทฤษฎีอนุภาค A
    แปลง Technical Indicators เป็น Phase และ Resonance
    """
    @staticmethod
    def calculate_resonance_score(data):
        """
        คำนวณค่าความสั่นพ้องรวม (Resonance Score)
        Range: -100 (Sell Strong) ถึง +100 (Buy Strong)
        """
        if data.empty: return 0

        latest = data.iloc[-1]
        
        # 1. Trend Phase (ทิศทางแนวโน้ม) - Weight 40%
        # เทียบราคาปิดกับเส้นค่าเฉลี่ย SMA20
        trend_phase = 1 if latest['Close'] > latest['SMA20'] else -1
        
        # 2. Momentum Phase (RSI) - Weight 30%
        # แปลง RSI 0-100 เป็น -1 ถึง 1 (50 คือ 0)
        momentum_phase = (latest['RSI'] - 50) / 50
        
        # 3. Energy Amplitude (Volume) - Weight 20%
        # ถ้า Volume วันนี้มากกว่าค่าเฉลี่ย 20 วัน = มีพลังงานสูง (ขยายผลของ Trend)
        vol_avg = data['Volume'].rolling(window=20).mean().iloc[-1]
        energy_amp = 1.5 if latest['Volume'] > vol_avg else 0.8
        
        # 4. Harmonic Alignment (MACD) - Weight 10%
        harmonic = 1 if latest['MACD'] > latest['Signal'] else -1

        # คำนวณ Resonance Score
        # สูตร: (Trend * Weight) + (Momentum * Weight) + (Harmonic * Weight) * Energy
        raw_score = ( (trend_phase * 0.4) + (momentum_phase * 0.3) + (harmonic * 0.1) ) * energy_amp
        
        # Normalize เป็น -100 ถึง 100
        final_score =  max(min(raw_score * 100 * 1.5, 100), -100)
        
        return final_score

    @staticmethod
    def interpret_phase(score):
        """แปลผลคะแนนเป็นสถานะทางฟิสิกส์"""
        if score >= 60: return "Constructive Interference (สั่นพ้องเสริมแรง/ขาขึ้น)"
        if score >= 20: return "Accumulation Phase (ระยะสะสมพลังงาน)"
        if score > -20: return "Equilibrium/Noise (สมดุล/ผันผวน)"
        if score > -60: return "Decay Phase (ระยะเสื่อมถอย)"
        return "Destructive Interference (สั่นพ้องทำลาย/ขาลง)"

# --- Data Fetching & Processing ---

def get_stock_data(symbol, period='1mo'):
    sym = normalize_symbol(symbol)
    if not sym:
        return None
    if period == '1h':
        df = get_yf_history(sym, period='1mo', interval='1h', auto_adjust=True)
    elif period == '15m':
        df = get_yf_history(sym, period='5d', interval='15m', auto_adjust=True)
    else:
        df = get_yf_history(sym, period=period, interval=None, auto_adjust=True)
    if df is None or df.empty:
        return None
    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    if len(cols) < 4 or 'Close' not in cols:
        return None
    return df[cols].copy()

def calculate_technical_indicators(data):
    try:
        df = data.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = (100 - (100 / (1 + rs))).fillna(0)
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = (exp1 - exp2).fillna(0)
        df['Signal'] = df['MACD'].ewm(span=9).mean().fillna(0)
        
        # SMA & Bollinger Bands
        df['SMA20'] = df['Close'].rolling(window=20).mean().fillna(0)
        std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = (df['SMA20'] + 2 * std).fillna(0)
        df['BB_Lower'] = (df['SMA20'] - 2 * std).fillna(0)
        
        return df
    except Exception as e:
        logger.warning("Calculation error: %s", e, exc_info=True)
        return data

def get_history_data(symbol, period="2y", interval="1d"):
    sym = normalize_symbol(symbol)
    df = get_yf_history(sym, period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        return None
    if "Close" not in df.columns:
        return None
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].copy()

def backtest_ema_hold_strategy(df, ema_length, fee_bps=0.0):
    if df is None or df.empty:
        return None
    if "Close" not in df.columns:
        return None
    close = df["Close"].astype(float)
    if close.isna().all():
        return None
    ema = close.ewm(span=int(ema_length), adjust=False).mean()
    signal = (close > ema).astype(int)
    ret = close.pct_change().fillna(0.0)
    pos = signal.shift(1).fillna(0).astype(int)
    strat = ret * pos
    if isinstance(fee_bps, (int, float)) and fee_bps > 0:
        turnover = signal.diff().abs().fillna(0.0)
        fee = (turnover * (float(fee_bps) / 10000.0))
        strat = strat - fee
    equity = (1.0 + strat).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    in_market = int(pos.sum())
    wins = int(((strat > 0) & (pos == 1)).sum())
    win_rate = (wins / in_market) * 100.0 if in_market > 0 else 0.0
    entry_count = int(((signal.diff().fillna(0) == 1)).sum())
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    return {
        "ema_length": int(ema_length),
        "ema_value": float(ema.iloc[-1]) if len(ema) else None,
        "last_close": float(close.iloc[-1]) if len(close) else None,
        "total_return_pct": total_return * 100.0,
        "win_rate_pct": float(win_rate),
        "in_market_bars": in_market,
        "entries": entry_count,
        "max_drawdown_pct": max_dd * 100.0,
    }

def optimize_best_ema(symbol, period="2y", interval="1d", min_len=10, max_len=200, step=5, fee_bps=0.0, top_n=5):
    sym = normalize_symbol(symbol)
    df = get_history_data(sym, period=period, interval=interval)
    if df is None or df.empty:
        return {"symbol": sym, "error": "No Data"}
    try:
        min_len = int(min_len)
        max_len = int(max_len)
        step = int(step)
        top_n = int(top_n)
    except Exception:
        return {"symbol": sym, "error": "Invalid parameters"}
    if min_len < 2 or max_len < min_len or step < 1:
        return {"symbol": sym, "error": "Invalid parameters"}
    if max_len > 400:
        max_len = 400
    if min_len > 400:
        min_len = 400
    if top_n < 1:
        top_n = 1
    if top_n > 20:
        top_n = 20
    results = []
    for length in range(min_len, max_len + 1, step):
        r = backtest_ema_hold_strategy(df, length, fee_bps=fee_bps)
        if r:
            results.append(r)
    if not results:
        return {"symbol": sym, "error": "No results"}
    results_sorted = sorted(
        results,
        key=lambda x: x.get("total_return_pct") if isinstance(x.get("total_return_pct"), (int, float)) else -1e18,
        reverse=True
    )
    best = results_sorted[0]
    top = results_sorted[:top_n]
    payload = {
        "symbol": sym,
        "period": period,
        "interval": interval,
        "fee_bps": float(fee_bps) if isinstance(fee_bps, (int, float)) else 0.0,
        "min_len": min_len,
        "max_len": max_len,
        "step": step,
        "evaluated": len(results_sorted),
        "best": best,
        "top": top,
        "computed_at": get_thai_now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    EMA_OPTIMIZER_BEST[sym] = {
        "best_length": best.get("ema_length"),
        "period": period,
        "interval": interval,
        "fee_bps": payload["fee_bps"],
        "computed_at": payload["computed_at"],
    }
    return payload

def get_basic_info(symbol):
    sym = normalize_symbol(symbol)
    if not sym:
        return {'name': '', 'sector': 'N/A', 'market_cap': 0, 'pe_ratio': 'N/A', 'dividend_yield': 0}
    cache_key = ("info", sym)
    cached = _YF_INFO_CACHE.get(cache_key)
    if isinstance(cached, dict) and cached:
        return dict(cached)
    try:
        session = _get_thread_curl_session()
        info = yf.Ticker(sym, session=session).info or {}
        payload = {
            'name': info.get('shortName', sym),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('forwardPE', 'N/A'),
            'dividend_yield': (info.get('dividendYield', 0) * 100) if info.get('dividendYield') else 0
        }
        _YF_INFO_CACHE.set(cache_key, payload)
        return dict(payload)
    except Exception:
        payload = {'name': sym, 'sector': 'N/A', 'market_cap': 0, 'pe_ratio': 'N/A', 'dividend_yield': 0}
        _YF_INFO_CACHE.set(cache_key, payload)
        return dict(payload)

# --- Short Term 15m Strategy (Advanced) ---

class ShortTermStrategy:
    """
    ระบบคำนวณแผนเทรดระยะสั้น 15 นาที (Advanced Version)
    เพิ่มความแม่นยำด้วย: 1H Trend Filter + Volume Analysis + Confidence Scoring
    """
    
    @staticmethod
    def calculate_fibonacci_levels(data, window=40):
        """หา Fibonacci Retracement"""
        recent_data = data.tail(window)
        swing_high = recent_data['High'].max()
        swing_low = recent_data['Low'].min()
        diff = swing_high - swing_low
        
        levels = {
            '0.0% (Low)': swing_low,
            '23.6%': swing_low + 0.236 * diff,
            '38.2%': swing_low + 0.382 * diff,
            '50.0%': swing_low + 0.5 * diff,
            '61.8% (Golden)': swing_low + 0.618 * diff,
            '78.6%': swing_low + 0.786 * diff,
            '100.0% (High)': swing_high
        }
        return levels, swing_high, swing_low

    @staticmethod
    def analyze_15m_setup(symbol):
        """วิเคราะห์จุดเข้า-ออก Timeframe 15 นาที แบบความแม่นยำสูง"""
        try:
            def _to_float(x):
                try:
                    v = float(x)
                    if math.isnan(v) or math.isinf(v):
                        return None
                    return v
                except Exception:
                    return None

            def _beta_win_prob(wins, losses, alpha=1.0, beta=1.0):
                try:
                    wins = float(wins)
                    losses = float(losses)
                    alpha = float(alpha)
                    beta = float(beta)
                except Exception:
                    return None
                denom = wins + losses + alpha + beta
                if denom <= 0:
                    return None
                p = (wins + alpha) / denom
                return max(0.0, min(1.0, p))

            def _calc_adx(df_in, period=14):
                high = df_in["High"].astype(float)
                low = df_in["Low"].astype(float)
                close = df_in["Close"].astype(float)
                plus_dm = high.diff()
                minus_dm = low.diff()
                plus_dm = plus_dm.where(plus_dm > 0, 0.0)
                minus_dm = (-minus_dm).where((-minus_dm) > 0, 0.0)
                tr1 = (high - low).abs()
                tr2 = (high - close.shift(1)).abs()
                tr3 = (low - close.shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(period).mean()
                plus_di = 100.0 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
                minus_di = 100.0 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
                dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).abs()) * 100.0
                return dx.rolling(period).mean()

            sym = normalize_symbol(symbol)

            yf_period = getattr(config, "SHORT_TERM_15M_YF_PERIOD", "30d")
            data_15m = get_yf_history(sym, period=str(yf_period), interval="15m", auto_adjust=True)
            if data_15m is None or data_15m.empty:
                data_15m = get_yf_history(sym, period="5d", interval="15m", auto_adjust=True)

            trend_period = getattr(config, "SHORT_TERM_15M_TREND_1H_PERIOD", "3mo")
            data_1h = get_yf_history(sym, period=str(trend_period), interval="1h", auto_adjust=True)
            if data_1h is None or data_1h.empty:
                data_1h = get_yf_history(sym, period="1mo", interval="1h", auto_adjust=True)

            if data_15m is None or data_15m.empty or data_1h is None or data_1h.empty:
                return None

            df_1h = data_1h.copy()
            df_1h["EMA50"] = df_1h["Close"].ewm(span=50, adjust=False).mean()
            last_close_1h = _to_float(df_1h["Close"].iloc[-1])
            last_ema50_1h = _to_float(df_1h["EMA50"].iloc[-1])
            trend_1h = "UP" if last_close_1h is not None and last_ema50_1h is not None and last_close_1h > last_ema50_1h else "DOWN"
            ema50_slope = _to_float(df_1h["EMA50"].diff(10).iloc[-1]) if len(df_1h) >= 12 else None
            trend_strength_1h = "WEAK"
            if ema50_slope is not None:
                ema50_last = _to_float(df_1h["EMA50"].iloc[-1])
                if ema50_last and ema50_last > 0:
                    slope_pct = abs(ema50_slope) / ema50_last
                    if slope_pct >= 0.004:
                        trend_strength_1h = "STRONG"

            df = data_15m.copy()

            if not all(c in df.columns for c in ("Open", "High", "Low", "Close", "Volume")) or len(df) < 80:
                return None

            df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
            df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

            df["H-L"] = df["High"] - df["Low"]
            df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
            df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
            df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
            df["ATR"] = df["TR"].rolling(window=14).mean()

            df["Vol_Avg"] = df["Volume"].rolling(window=20).mean()
            df["RVOL"] = df["Volume"] / df["Vol_Avg"]
            df["ADX"] = _calc_adx(df, period=14)

            rvol_min = getattr(config, "SHORT_TERM_15M_SIGNAL_RVOL_MIN", 1.2)
            adx_min = getattr(config, "SHORT_TERM_15M_SIGNAL_ADX_MIN", 18.0)
            swing_lookback = getattr(config, "SHORT_TERM_15M_SWING_LOOKBACK", 12)
            min_r_mult = getattr(config, "SHORT_TERM_15M_RISK_ATR_MIN_MULT", 1.0)
            max_r_mult = getattr(config, "SHORT_TERM_15M_RISK_ATR_MAX_MULT", 3.0)
            stop_atr_buffer = getattr(config, "SHORT_TERM_15M_STOP_ATR_BUFFER", 0.2)
            pullback_tol_pct = getattr(config, "SHORT_TERM_15M_PULLBACK_TOL_PCT", 0.3)
            tp1_r = getattr(config, "SHORT_TERM_15M_TP1_R", 1.5)
            tp2_r = getattr(config, "SHORT_TERM_15M_TP2_R", 3.0)
            max_forward = getattr(config, "SHORT_TERM_15M_MAX_FORWARD_BARS", 32)
            min_trades = getattr(config, "SHORT_TERM_15M_MIN_TRADES", 10)
            beta_alpha = getattr(config, "SHORT_TERM_15M_BETA_ALPHA", 1.0)
            beta_beta = getattr(config, "SHORT_TERM_15M_BETA_BETA", 1.0)

            try:
                rvol_min = float(rvol_min)
            except Exception:
                rvol_min = 1.2
            try:
                adx_min = float(adx_min)
            except Exception:
                adx_min = 18.0
            try:
                swing_lookback = int(swing_lookback)
            except Exception:
                swing_lookback = 12
            try:
                min_r_mult = float(min_r_mult)
            except Exception:
                min_r_mult = 1.0
            try:
                max_r_mult = float(max_r_mult)
            except Exception:
                max_r_mult = 3.0
            try:
                stop_atr_buffer = float(stop_atr_buffer)
            except Exception:
                stop_atr_buffer = 0.2
            try:
                pullback_tol_pct = float(pullback_tol_pct)
            except Exception:
                pullback_tol_pct = 0.3
            try:
                tp1_r = float(tp1_r)
            except Exception:
                tp1_r = 1.5
            try:
                tp2_r = float(tp2_r)
            except Exception:
                tp2_r = 3.0
            try:
                max_forward = int(max_forward)
            except Exception:
                max_forward = 32
            try:
                min_trades = int(min_trades)
            except Exception:
                min_trades = 10

            if swing_lookback < 5:
                swing_lookback = 5
            if min_r_mult <= 0:
                min_r_mult = 1.0
            if max_r_mult < min_r_mult:
                max_r_mult = min_r_mult
            if max_forward < 8:
                max_forward = 8
            if tp1_r <= 0:
                tp1_r = 1.5
            if tp2_r <= tp1_r:
                tp2_r = max(tp1_r + 0.5, 3.0)

            df["PrevHigh20"] = df["High"].rolling(window=20).max().shift(1)
            tol = pullback_tol_pct / 100.0
            trend_ok = (df["Close"] > df["EMA50"]) & (df["EMA50"] > df["EMA200"])
            breakout = trend_ok & (df["Close"] > df["PrevHigh20"]) & (df["RSI"] >= 55) & (df["RSI"] <= 80) & (df["RVOL"] >= rvol_min) & (df["ADX"] >= adx_min)
            pullback = trend_ok & (df["Low"] <= (df["EMA20"] * (1 + tol))) & (df["Close"] > df["EMA20"]) & (df["RSI"] >= 50) & (df["RSI"].diff() > 0) & (df["RVOL"] >= max(1.0, rvol_min - 0.2))

            signal_type = pd.Series(np.where(breakout, "BREAKOUT", np.where(pullback, "PULLBACK", "")), index=df.index)

            def _risk_params_at(i):
                entry = _to_float(df["Close"].iloc[i])
                atr = _to_float(df["ATR"].iloc[i])
                if entry is None or atr is None or atr <= 0:
                    return None
                start = max(0, i - swing_lookback + 1)
                swing_low_val = _to_float(df["Low"].iloc[start:i + 1].min())
                if swing_low_val is None:
                    return None
                raw_sl = swing_low_val - (stop_atr_buffer * atr)
                risk_dist = entry - raw_sl
                min_dist = min_r_mult * atr
                max_dist = max_r_mult * atr
                if risk_dist is None or risk_dist <= 0:
                    risk_dist = min_dist
                risk_dist = max(min_dist, min(risk_dist, max_dist))
                sl = entry - risk_dist
                return entry, sl, risk_dist, atr, swing_low_val

            def _simulate(df_in, sig_series):
                stats = {
                    "BREAKOUT": {"tp1_w": 0, "tp1_l": 0, "tp2_w": 0, "tp2_l": 0, "tp1_t": 0, "tp2_t": 0},
                    "PULLBACK": {"tp1_w": 0, "tp1_l": 0, "tp2_w": 0, "tp2_l": 0, "tp1_t": 0, "tp2_t": 0},
                }
                highs = df_in["High"].astype(float).values
                lows = df_in["Low"].astype(float).values
                for i in range(1, len(df_in) - 2):
                    st = sig_series.iloc[i]
                    if not st:
                        continue
                    rp = _risk_params_at(i)
                    if not rp:
                        continue
                    entry_i, sl_i, risk_i, atr_i, swing_low_i = rp
                    tp1_i = entry_i + (risk_i * tp1_r)
                    tp2_i = entry_i + (risk_i * tp2_r)
                    end_j = min(len(df_in), i + 1 + max_forward)
                    outcome1 = None
                    outcome2 = None
                    for j in range(i + 1, end_j):
                        if lows[j] <= sl_i:
                            if outcome1 is None:
                                outcome1 = "loss"
                            if outcome2 is None:
                                outcome2 = "loss"
                            break
                        if outcome1 is None and highs[j] >= tp1_i:
                            outcome1 = "win"
                        if outcome2 is None and highs[j] >= tp2_i:
                            outcome2 = "win"
                        if outcome1 is not None and outcome2 is not None:
                            break
                    bucket = stats.get(st)
                    if not bucket:
                        continue
                    if outcome1 == "win":
                        bucket["tp1_w"] += 1
                        bucket["tp1_t"] += 1
                    elif outcome1 == "loss":
                        bucket["tp1_l"] += 1
                        bucket["tp1_t"] += 1
                    if outcome2 == "win":
                        bucket["tp2_w"] += 1
                        bucket["tp2_t"] += 1
                    elif outcome2 == "loss":
                        bucket["tp2_l"] += 1
                        bucket["tp2_t"] += 1
                return stats

            stats = _simulate(df, signal_type)

            latest = df.iloc[-1]
            curr_signal_type = signal_type.iloc[-1] if isinstance(signal_type.iloc[-1], str) and signal_type.iloc[-1] else ""
            entry_price = _to_float(latest["Close"])
            ema20 = _to_float(latest["EMA20"])
            ema50 = _to_float(latest["EMA50"])
            ema200 = _to_float(latest["EMA200"])
            rsi_val = _to_float(latest["RSI"])
            atr_val = _to_float(latest["ATR"])
            adx_val = _to_float(latest["ADX"])
            rvol_val = _to_float(latest["RVOL"])

            fibo_levels, swing_high, swing_low = ShortTermStrategy.calculate_fibonacci_levels(df)

            setup_type = "WAIT"
            reasons = []
            if trend_1h == "UP":
                reasons.append("เทรนด์ 1H ขาขึ้น")
                if trend_strength_1h == "STRONG":
                    reasons.append("แนวโน้ม 1H แข็งแรง")
            else:
                reasons.append("เทรนด์ 1H ไม่เป็นขาขึ้น")

            direction_15m = "DOWN"
            if ema50 is not None and entry_price is not None and entry_price > ema50:
                direction_15m = "UP"

            if curr_signal_type:
                setup_type = "BUY / LONG"
                reasons.append("สัญญาณเข้าแบบ " + curr_signal_type)
            else:
                if direction_15m == "UP":
                    reasons.append("โครงสร้าง 15m ยังเป็นขาขึ้น แต่ยังไม่เข้าเงื่อนไขจุดเข้า")
                else:
                    reasons.append("โครงสร้าง 15m ไม่เป็นขาขึ้น ให้รอ")

            stop_loss = 0.0
            tp1 = 0.0
            tp2 = 0.0
            entry_zone_low = None
            entry_zone_high = None

            rp_latest = _risk_params_at(len(df) - 1)
            if rp_latest:
                entry_i, sl_i, risk_i, atr_i, swing_low_i = rp_latest
                stop_loss = sl_i
                tp1 = entry_i + (risk_i * tp1_r)
                tp2 = entry_i + (risk_i * tp2_r)
                if curr_signal_type == "BREAKOUT":
                    prev_high = _to_float(df["High"].iloc[-21:-1].max()) if len(df) >= 22 else _to_float(df["High"].iloc[:-1].max())
                    if prev_high is not None and atr_i is not None:
                        entry_zone_low = max(prev_high, entry_i - (0.3 * atr_i))
                        entry_zone_high = entry_i
                elif curr_signal_type == "PULLBACK":
                    if ema20 is not None and ema50 is not None:
                        entry_zone_low = min(ema20, ema50)
                        entry_zone_high = entry_i

            rr_text = "N/A"
            if entry_price is not None and stop_loss and entry_price != stop_loss and tp1 and tp2:
                r1 = abs(tp1 - entry_price) / abs(entry_price - stop_loss)
                r2 = abs(tp2 - entry_price) / abs(entry_price - stop_loss)
                rr_text = f"TP1 1:{r1:.1f} | TP2 1:{r2:.1f}"

            predicted_win_prob = None
            hist_trades_tp1 = None
            hist_win_rate_tp1 = None
            expectancy_tp1 = None
            win_prob_tp2 = None
            hist_trades_tp2 = None
            hist_win_rate_tp2 = None
            expectancy_tp2 = None

            bucket = stats.get(curr_signal_type) if curr_signal_type else None
            prob_sample_quality = None
            if isinstance(bucket, dict):
                tp1_t = bucket.get("tp1_t", 0)
                tp1_w = bucket.get("tp1_w", 0)
                tp1_l = bucket.get("tp1_l", 0)
                if isinstance(tp1_t, int) and tp1_t > 0:
                    p1 = _beta_win_prob(tp1_w, tp1_l, alpha=beta_alpha, beta=beta_beta)
                    if p1 is not None:
                        predicted_win_prob = p1 * 100.0
                        hist_trades_tp1 = tp1_t
                        hist_win_rate_tp1 = (tp1_w / tp1_t) * 100.0 if tp1_t > 0 else None
                        expectancy_tp1 = (p1 * tp1_r) - ((1 - p1) * 1.0)
                        prob_sample_quality = "OK" if tp1_t >= min_trades else "LOW"
                tp2_t = bucket.get("tp2_t", 0)
                tp2_w = bucket.get("tp2_w", 0)
                tp2_l = bucket.get("tp2_l", 0)
                if isinstance(tp2_t, int) and tp2_t > 0:
                    p2 = _beta_win_prob(tp2_w, tp2_l, alpha=beta_alpha, beta=beta_beta)
                    if p2 is not None:
                        win_prob_tp2 = p2 * 100.0
                        hist_trades_tp2 = tp2_t
                        hist_win_rate_tp2 = (tp2_w / tp2_t) * 100.0 if tp2_t > 0 else None
                        expectancy_tp2 = (p2 * tp2_r) - ((1 - p2) * 1.0)

            score = 0.0
            max_score = 7.0
            if trend_1h == "UP":
                score += 1.0
                if trend_strength_1h == "STRONG":
                    score += 0.5
            if direction_15m == "UP":
                score += 1.5
            if rsi_val is not None and 50 <= rsi_val <= 80:
                score += 1.0
            if adx_val is not None and adx_val >= adx_min:
                score += 1.0
            if rvol_val is not None and rvol_val >= rvol_min:
                score += 1.0
            if curr_signal_type:
                score += 1.0
            base_conf = (score / max_score) * 100.0 if max_score else 0.0
            if isinstance(predicted_win_prob, (int, float)):
                confidence = predicted_win_prob if prob_sample_quality == "OK" else (base_conf + predicted_win_prob) / 2.0
            else:
                confidence = base_conf

            if setup_type.startswith("BUY") and confidence < 55:
                setup_type = "WAIT (Weak Long)"
                reasons = ["สัญญาณยังไม่คม/สถิติไม่พอ หรือความได้เปรียบต่ำ"]

            required_data = [
                "ข้อมูลราคา 15m (OHLCV) อย่างน้อย 30 วัน",
                "ข้อมูลราคา 1H (OHLCV) อย่างน้อย 3 เดือน",
                "EMA20/50/200, RSI14, ATR14, ADX14",
                "RVOL (Volume เทียบค่าเฉลี่ย 20 แท่ง)",
                "Swing High/Low และจุด Breakout/Pullback",
            ]

            return {
                "setup": setup_type,
                "strategy": "EdgePulse 15m Long",
                "entry_type": curr_signal_type or None,
                "direction_15m": direction_15m,
                "confidence": float(confidence) if confidence is not None else None,
                "predicted_win_prob": float(predicted_win_prob) if isinstance(predicted_win_prob, (int, float)) else None,
                "current_price": entry_price,
                "ema20": ema20,
                "ema50": ema50,
                "ema200": ema200,
                "rsi": rsi_val,
                "adx": adx_val,
                "atr": atr_val,
                "stop_loss": float(stop_loss) if isinstance(stop_loss, (int, float)) else None,
                "take_profit": float(tp1) if isinstance(tp1, (int, float)) else None,
                "take_profit_2": float(tp2) if isinstance(tp2, (int, float)) else None,
                "entry_zone_low": float(entry_zone_low) if isinstance(entry_zone_low, (int, float)) else None,
                "entry_zone_high": float(entry_zone_high) if isinstance(entry_zone_high, (int, float)) else None,
                "risk_reward": rr_text,
                "reason": " | ".join([r for r in reasons if r]),
                "trend_1h": trend_1h,
                "trend_strength_1h": trend_strength_1h,
                "rvol": rvol_val,
                "fibo_levels": fibo_levels,
                "historical_trades_tp1": hist_trades_tp1,
                "historical_win_rate_tp1": hist_win_rate_tp1,
                "expectancy_tp1_rr": expectancy_tp1,
                "historical_trades_tp2": hist_trades_tp2,
                "historical_win_rate_tp2": hist_win_rate_tp2,
                "expectancy_tp2_rr": expectancy_tp2,
                "prob_sample_quality": prob_sample_quality,
                "required_data": required_data,
            }
            
        except Exception as e:
            logger.warning("Advanced Strategy Error: %s", e, exc_info=True)
            return None

    @staticmethod
    def analyze_sniper_setup(symbol):
        try:
            sym = normalize_symbol(symbol)
            data_15m = get_yf_history(sym, period='5d', interval='15m', auto_adjust=True)
            data_1h = get_yf_history(sym, period='1mo', interval='1h', auto_adjust=True)
            if data_15m is None or data_1h is None or data_15m.empty or data_1h.empty:
                return None

            df_1h = data_1h.copy()
            df_1h['EMA50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()

            df = data_15m.copy()
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

            if len(df) < 20:
                return None

            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            df['H-L'] = df['High'] - df['Low']
            df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
            df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()

            df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
            df['RVOL'] = df['Volume'] / df['Vol_Avg']

            trend_1h = "UP" if df_1h['Close'].iloc[-1] > df_1h['EMA50'].iloc[-1] else "DOWN"

            curr = df.iloc[-1]
            prev = df.iloc[-2]

            if pd.isna(curr['ATR']) or curr['ATR'] <= 0:
                return None

            recent_window = df.iloc[-15:-1]
            recent_high = recent_window['High'].max()
            recent_low = recent_window['Low'].min()

            df['ATR_PCT'] = (df['ATR'] / df['Close']) * 100
            atr_profile = df['ATR_PCT'].tail(40).median()
            if pd.isna(atr_profile):
                atr_profile = df['ATR_PCT'].median()

            risk_profile = "MID"
            if atr_profile <= 1.5:
                risk_profile = "LOW"
            elif atr_profile >= 3.0:
                risk_profile = "HIGH"

            min_r_mult = 1.0
            max_r_mult = 2.5
            if risk_profile == "LOW":
                min_r_mult = 0.8
                max_r_mult = 2.0
            elif risk_profile == "HIGH":
                min_r_mult = 1.5
                max_r_mult = 3.0

            setup_type = "WAIT / DO NOTHING"
            confidence = 0
            reason = "ตลาดไม่ชัดเจนหรือระยะกำไรไม่คุ้มความเสี่ยง"

            atr_val = float(curr['ATR'])
            entry = float(curr['Close'])
            stop_loss = 0.0
            take_profit = 0.0

            if trend_1h == "UP":
                cond_price = curr['Close'] > curr['EMA50'] and prev['Close'] <= prev['EMA50']
                cond_break = curr['Close'] > prev['High']
                cond_rsi = (curr['RSI'] > 55) and (curr['RSI'] < 75) and (prev['RSI'] < curr['RSI'])
                cond_vol = curr['RVOL'] > 1.5

                if cond_price and cond_break and cond_rsi and cond_vol:
                    raw_sl = recent_low - (0.2 * atr_val)
                    risk_dist = entry - raw_sl
                    if risk_dist <= 0:
                        risk_dist = atr_val
                    if atr_val > 0:
                        min_dist = min_r_mult * atr_val
                        max_dist = max_r_mult * atr_val
                        risk_dist = max(min_dist, min(risk_dist, max_dist))
                    stop_loss = entry - risk_dist

                    target_high = max(recent_high, entry)
                    potential_tp = target_high + (2.0 * atr_val)
                    reward_dist = potential_tp - entry
                    rr = reward_dist / risk_dist if risk_dist > 0 else 0

                    if rr >= 1.8:
                        take_profit = potential_tp
                        setup_type = "SNIPER BUY"
                        confidence = 88
                        reason = "จังหวะตามเทรนด์ขาขึ้นหลังพักตัว โดยใช้สวิงโลว์ย้อนหลังเป็นจุดกันความเสี่ยง และกำหนดระยะทำกำไรให้ Risk/Reward ไม่เสียเปรียบ"

            elif trend_1h == "DOWN":
                cond_price = curr['Close'] < curr['EMA50'] and prev['Close'] >= prev['EMA50']
                cond_break = curr['Close'] < prev['Low']
                cond_rsi = (curr['RSI'] < 45) and (curr['RSI'] > 25) and (prev['RSI'] > curr['RSI'])
                cond_vol = curr['RVOL'] > 1.5

                if cond_price and cond_break and cond_rsi and cond_vol:
                    raw_sl = recent_high + (0.2 * atr_val)
                    risk_dist = raw_sl - entry
                    if risk_dist <= 0:
                        risk_dist = atr_val
                    if atr_val > 0:
                        min_dist = min_r_mult * atr_val
                        max_dist = max_r_mult * atr_val
                        risk_dist = max(min_dist, min(risk_dist, max_dist))
                    stop_loss = entry + risk_dist

                    target_low = min(recent_low, entry)
                    potential_tp = target_low - (2.0 * atr_val)
                    reward_dist = entry - potential_tp
                    rr = reward_dist / risk_dist if risk_dist > 0 else 0

                    if rr >= 1.8:
                        take_profit = potential_tp
                        setup_type = "SNIPER SELL"
                        confidence = 88
                        reason = "จังหวะตามเทรนด์ขาลงหลังย่อ โดยใช้สวิงไฮย้อนหลังเป็นจุดกันความเสี่ยง และกำหนดระยะทำกำไรให้ Risk/Reward ไม่เสียเปรียบ"

            if not setup_type.startswith("SNIPER"):
                return {
                    'setup': "WAIT / DO NOTHING",
                    'reason': "ตลาดไม่ชัดเจน หรือระดับราคาไม่ให้ระยะ Risk/Reward ที่ได้เปรียบ",
                    'confidence': confidence,
                    'action': "ถือเงินสด"
                }

            risk_reward = "N/A"
            if stop_loss and take_profit and entry and entry != stop_loss:
                rr_val = abs(take_profit - entry) / abs(entry - stop_loss)
                risk_reward = f"1:{rr_val:.1f}"

            return {
                'setup': setup_type,
                'current_price': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': risk_reward,
                'reason': reason,
                'confidence': confidence
            }

        except Exception:
            return None
class QuantumHunterStrategy:
    """
    ขั้นกว่าของ Sniper Mode: ใช้ Z-Score และ ADX กรองสัญญาณรบกวน
    อิงหลักสถิติชั้นสูงเพื่อหาจุดเข้าที่มีความได้เปรียบ (Edge) สูงสุด
    """

    @staticmethod
    def calculate_adx(df, period=14):
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(df['High'] - df['Low'])
        tr2 = pd.DataFrame(abs(df['High'] - df['Close'].shift(1)))
        tr3 = pd.DataFrame(abs(df['Low'] - df['Close'].shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        return adx

    @staticmethod
    def z_score(series, window=20):
        r = series.rolling(window=window)
        m = r.mean()
        s = r.std(ddof=0)
        z = (series - m) / s
        return z

    @staticmethod
    def analyze(symbol, period="5d"):
        try:
            sym = normalize_symbol(symbol)
            df = get_yf_history(sym, period=period, interval="15m", auto_adjust=True)
            if df is None or df.empty or len(df) < 60:
                return None
            return QuantumHunterStrategy.process_data(df.copy())
        except Exception:
            return None

    @staticmethod
    def process_data(df):
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['ADX'] = QuantumHunterStrategy.calculate_adx(df)
        df['Vol_Z'] = QuantumHunterStrategy.z_score(df['Volume'])
        df['Price_Z'] = QuantumHunterStrategy.z_score(df['Close'].pct_change())

        curr = df.iloc[-1]
        
        setup_type = "WAIT"
        confidence = 0

        if curr['ADX'] < 25:
            return {'setup': 'WAIT', 'reason': 'ตลาดอ่อนแรง (Low ADX)'}

        if curr['Vol_Z'] < 2.0:
            return {'setup': 'WAIT', 'reason': 'Volume ไม่ยืนยันทางสถิติ'}

        if curr['Close'] > curr['EMA50']:
            if curr['Price_Z'] > 1.5:
                setup_type = "QUANTUM BUY"
                confidence = 95
                reason_text = "Trend แข็งแกร่ง (ADX) + Volume ผิดปกติ (Z-Score > 2)"
        elif curr['Close'] < curr['EMA50']:
            if curr['Price_Z'] < -1.5:
                setup_type = "QUANTUM SELL"
                confidence = 95
                reason_text = "Trend แข็งแกร่ง (ADX) + แรงเทขายผิดปกติ"
        else:
            reason_text = "No Signal"

        if "BUY" in setup_type:
            sl = curr['Close'] - (2.0 * curr['ATR'])
            tp = curr['Close'] + (5.0 * curr['ATR'])
        elif "SELL" in setup_type:
            sl = curr['Close'] + (2.0 * curr['ATR'])
            tp = curr['Close'] - (5.0 * curr['ATR'])
        else:
            sl, tp = 0, 0

        return {
            'setup': setup_type,
            'confidence': confidence,
            'adx': float(curr['ADX']) if not pd.isna(curr['ADX']) else None,
            'vol_z_score': float(curr['Vol_Z']) if not pd.isna(curr['Vol_Z']) else None,
            'current_price': float(curr['Close']) if not pd.isna(curr['Close']) else None,
            'stop_loss': sl,
            'take_profit': tp,
            'reason': reason_text
        }

class CryptoReversal15m:
    @staticmethod
    def analyze(symbol):
        if not is_crypto_symbol(symbol):
            return None
        try:
            sym = normalize_symbol(symbol)
            data_15m = get_yf_history(sym, period='5d', interval='15m', auto_adjust=True)
            data_1h = get_yf_history(sym, period='1mo', interval='1h', auto_adjust=True)
            if data_15m is None or data_1h is None or data_15m.empty or len(data_15m) < 60 or data_1h.empty or len(data_1h) < 40:
                return None
            df = data_15m.copy()
            df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['H-L'] = df['High'] - df['Low']
            df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
            df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()
            df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
            df['RVOL'] = df['Volume'] / df['Vol_Avg']
            df['ATR_PCT'] = (df['ATR'] / df['Close']) * 100
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['SMA20'] + 2 * std20
            df['BB_Lower'] = df['SMA20'] - 2 * std20
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            rvol = float(curr['RVOL']) if not pd.isna(curr['RVOL']) else None
            atr_pct = float(curr['ATR_PCT']) if not pd.isna(curr['ATR_PCT']) else None
            rsi = float(curr['RSI']) if not pd.isna(curr['RSI']) else None
            atr = float(curr['ATR']) if not pd.isna(curr['ATR']) else None
            price = float(curr['Close'])
            body = curr['Close'] - curr['Open']
            upper_shadow = curr['High'] - max(curr['Close'], curr['Open'])
            lower_shadow = min(curr['Close'], curr['Open']) - curr['Low']
            setup_type = "WAIT"
            pattern = ""
            score = 0
            max_score = 10
            if rsi is None or atr is None or atr <= 0:
                return None
            bb_low = float(curr['BB_Lower']) if not pd.isna(curr['BB_Lower']) else None
            bb_up = float(curr['BB_Upper']) if not pd.isna(curr['BB_Upper']) else None
            ema20 = float(curr['EMA20']) if not pd.isna(curr['EMA20']) else None
            ema50 = float(curr['EMA50']) if not pd.isna(curr['EMA50']) else None
            df_1h = data_1h.copy()
            df_1h['EMA50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()
            df_1h['EMA200'] = df_1h['Close'].ewm(span=200, adjust=False).mean()
            d_h = df_1h
            d_h['H-L'] = d_h['High'] - d_h['Low']
            d_h['H-PC'] = (d_h['High'] - d_h['Close'].shift(1)).abs()
            d_h['L-PC'] = (d_h['Low'] - d_h['Close'].shift(1)).abs()
            d_h['TR'] = d_h[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            d_h['ATR'] = d_h['TR'].rolling(window=14).mean()
            delta_h = d_h['Close'].diff()
            gain_h = delta_h.where(delta_h > 0, 0).rolling(window=14).mean()
            loss_h = (-delta_h.where(delta_h < 0, 0)).rolling(window=14).mean()
            rs_h = gain_h / loss_h
            d_h['RSI'] = 100 - (100 / (1 + rs_h))
            plus_dm = d_h['High'].diff().clip(lower=0)
            minus_dm = d_h['Low'].diff().clip(lower=0)
            atr_h = d_h['ATR']
            p_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr_h)
            m_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr_h)
            dx_h = (abs(p_di - m_di) / (p_di + m_di + 1e-9)) * 100
            d_h['ADX'] = dx_h.rolling(14).mean()
            curr_h = d_h.iloc[-1]
            trend_1h = "UP" if curr_h['Close'] > curr_h['EMA50'] else "DOWN"
            trend_strength_1h = "STRONG" if curr_h['ADX'] >= 25 else "WEAK"
            was_oversold = prev['RSI'] < 25 if not pd.isna(prev['RSI']) else False
            rsi_cross_up = was_oversold and rsi >= 25
            was_overbought = prev['RSI'] > 75 if not pd.isna(prev['RSI']) else False
            rsi_cross_down = was_overbought and rsi <= 75
            near_lower_band = bb_low is not None and price <= bb_low * 1.005
            near_upper_band = bb_up is not None and price >= bb_up * 0.995
            squeeze = False
            if not pd.isna(std20.iloc[-1]):
                recent_std = df['Close'].rolling(window=40).std().iloc[-20:]
                median_std = recent_std.median()
                if median_std > 0 and std20.iloc[-1] < median_std * 0.7:
                    squeeze = True
            hammer_like = lower_shadow >= 2 * abs(body) and upper_shadow <= abs(body) * 0.6
            shooting_star_like = upper_shadow >= 2 * abs(body) and lower_shadow <= abs(body) * 0.6
            regime = "RANGE"
            if atr_pct is not None:
                if atr_pct >= 3.5:
                    regime = "HIGH_VOL"
                elif atr_pct <= 1.5:
                    regime = "LOW_VOL"
            reasons = []
            smc_setup = "NONE"
            smc_entry = None
            smc_stop_loss = None
            smc_take_profit = None
            smc_rr = None
            if rsi_cross_up and near_lower_band:
                setup_type = "CRYPTO REVERSAL BUY"
                pattern = "RSI cross up near lower Bollinger Band"
                score += 4
                reasons.append("RSI ฟื้นจากเขต Oversold ใกล้แนวล่าง Bollinger Band")
                if squeeze:
                    score += 1
                    reasons.append("เกิด Bollinger Squeeze ก่อนหน้า มีโอกาสเกิดการระเบิดทิศทาง")
                if ema20 is not None and price > ema20:
                    score += 1
                    reasons.append("ราคากลับขึ้นเหนือ EMA20")
                if ema50 is not None and price > ema50:
                    score += 1
                    reasons.append("ราคายืนเหนือ EMA50 ระยะสั้น")
                if hammer_like:
                    score += 2
                    reasons.append("แท่งเทียนมีเงาล่างยาว ลักษณะคล้าย Hammer")
                if trend_1h == "UP":
                    score += 1
                    reasons.append("ทิศทาง 1H ยังเป็นขาขึ้น การกลับตัวนี้เป็นจังหวะย่อตัวในเทรนด์ใหญ่")
                    if trend_strength_1h == "STRONG":
                        score += 1
                        reasons.append("แนวโน้ม 1H แข็งแรง สนับสนุนการดีดกลับ")
                else:
                    reasons.append("สัญญาณนี้สวนเทรนด์ 1H ต้องใช้การบริหารความเสี่ยงเข้มงวด")
                if rvol is not None:
                    if rvol > 1.2 and rvol < 3.0:
                        score += 1
                        reasons.append("ปริมาณการซื้อขายสูงกว่าค่าเฉลี่ย แต่อยู่ในระดับไม่สุดโต่ง")
                    elif rvol <= 0.8:
                        reasons.append("Volume บางกว่าปกติ อาจทำให้สัญญาณไม่น่าเชื่อถือเท่าที่ควร")
                    else:
                        reasons.append("Volume สูงมาก ผันผวนสูง ต้องตั้งจุดตัดขาดทุนให้ชัดเจน")
            elif rsi_cross_down and near_upper_band:
                setup_type = "CRYPTO REVERSAL SELL"
                pattern = "RSI cross down near upper Bollinger Band"
                score += 4
                reasons.append("RSI อ่อนตัวจากเขต Overbought ใกล้แนวบน Bollinger Band")
                if squeeze:
                    score += 1
                    reasons.append("เกิด Bollinger Squeeze ก่อนหน้า มีโอกาสกลับตัวแรง")
                if ema20 is not None and price < ema20:
                    score += 1
                    reasons.append("ราคาหลุด EMA20 ลงมา")
                if ema50 is not None and price < ema50:
                    score += 1
                    reasons.append("ราคาหลุด EMA50 ยืนยันการอ่อนแรง")
                if shooting_star_like:
                    score += 2
                    reasons.append("แท่งเทียนมีเงาบนยาว ลักษณะคล้าย Shooting Star")
                if trend_1h == "DOWN":
                    score += 1
                    reasons.append("ทิศทาง 1H ยังเป็นขาลง การกลับตัวนี้เป็นจังหวะเด้งในเทรนด์ใหญ่")
                    if trend_strength_1h == "STRONG":
                        score += 1
                        reasons.append("แนวโน้ม 1H ขาลงแข็งแรง สนับสนุนสัญญาณขาย")
                else:
                    reasons.append("สัญญาณนี้สวนเทรนด์ 1H ต้องระวังการเด้งกลับแรง")
                if rvol is not None:
                    if rvol > 1.2 and rvol < 3.0:
                        score += 1
                        reasons.append("ปริมาณการซื้อขายหนาแน่นแต่ไม่ร้อนแรงเกินไป")
                    elif rvol <= 0.8:
                        reasons.append("Volume เบาบาง อาจเกิด False Break ได้ง่าย")
                    else:
                        reasons.append("Volume สูงผิดปกติ ต้องเฝ้าระวังความผันผวน")
            else:
                return {
                    'setup': "WAIT",
                    'confidence': 0,
                    'current_price': price,
                    'rsi': rsi,
                    'atr': atr,
                    'stop_loss': None,
                    'take_profit': None,
                    'pattern': "",
                    'reason': "ยังไม่พบสัญญาณกลับตัวชัดเจน"
                }
            conf = 0
            if score > 0 and max_score > 0:
                conf = (score / max_score) * 100
            if conf > 100:
                conf = 100
            if conf < 0:
                conf = 0
            base_sl_mult = 1.5
            base_tp_mult = 2.5
            if regime == "HIGH_VOL":
                base_sl_mult = 1.8
                base_tp_mult = 3.2
            elif regime == "LOW_VOL":
                base_sl_mult = 1.2
                base_tp_mult = 2.0
            if conf >= 80:
                base_tp_mult += 0.5
            elif conf <= 50:
                base_tp_mult -= 0.3
            if "BUY" in setup_type:
                stop_loss = price - base_sl_mult * atr
                take_profit = price + base_tp_mult * atr
            elif "SELL" in setup_type:
                stop_loss = price + base_sl_mult * atr
                take_profit = price - base_tp_mult * atr
            else:
                stop_loss = None
                take_profit = None
            rr = None
            if stop_loss is not None and take_profit is not None:
                risk_dist = abs(price - stop_loss)
                reward_dist = abs(take_profit - price)
                if risk_dist > 0:
                    rr = reward_dist / risk_dist
            win_prob = conf
            if rr is not None and rr > 0:
                if rr < 1.2:
                    win_prob = conf * 0.9
                elif rr >= 2.0:
                    win_prob = conf * 1.05
            if win_prob is not None:
                if win_prob > 98:
                    win_prob = 98
                if win_prob < 0:
                    win_prob = 0
            expected_move_pct = None
            if take_profit is not None and price > 0:
                if "BUY" in setup_type:
                    expected_move_pct = ((take_profit - price) / price) * 100
                elif "SELL" in setup_type:
                    expected_move_pct = ((price - take_profit) / price) * 100
            expected_holding_bars_15m = None
            if expected_move_pct is not None and atr_pct is not None and atr_pct > 0:
                approx_steps = abs(expected_move_pct) / atr_pct
                if approx_steps <= 0:
                    approx_steps = 1
                expected_holding_bars_15m = int(max(1, min(48, round(approx_steps))))
            df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
            df['Swing_Low'] = np.where((df['Low'].shift(1) > df['Low']) & (df['Low'].shift(-1) > df['Low']), df['Low'], np.nan)
            df['Swing_High'] = np.where((df['High'].shift(1) < df['High']) & (df['High'].shift(-1) < df['High']), df['High'], np.nan)
            df['Last_Swing_Low'] = df['Swing_Low'].ffill()
            df['Last_Swing_High'] = df['Swing_High'].ffill()
            if len(df) > 200:
                curr_idx = len(df) - 1
                prev_idx = curr_idx - 1
                if prev_idx >= 1:
                    curr_smc = df.iloc[curr_idx]
                    prev_smc = df.iloc[prev_idx]
                    last_swing_low = prev_smc['Last_Swing_Low']
                    last_swing_high = prev_smc['Last_Swing_High']
                    if not pd.isna(prev_smc['ATR']) and prev_smc['ATR'] > 0 and not pd.isna(curr_smc['EMA200']):
                        atr_buffer = curr_smc['ATR'] * 0.5
                        rr_target = 1.5
                        if not pd.isna(last_swing_low) and curr_smc['Close'] > curr_smc['EMA200']:
                            if prev_smc['Low'] < last_swing_low and prev_smc['Close'] > last_swing_low and curr_smc['Close'] > prev_smc['High']:
                                stop_loss_smc = prev_smc['Low'] - atr_buffer
                                risk_smc = curr_smc['Close'] - stop_loss_smc
                                if risk_smc > 0:
                                    take_profit_smc = curr_smc['Close'] + (risk_smc * rr_target)
                                    smc_setup = "SMC LONG"
                                    smc_entry = float(curr_smc['Close'])
                                    smc_stop_loss = float(stop_loss_smc)
                                    smc_take_profit = float(take_profit_smc)
                                    smc_rr = rr_target
                        if not pd.isna(last_swing_high) and curr_smc['Close'] < curr_smc['EMA200']:
                            if prev_smc['High'] > last_swing_high and prev_smc['Close'] < last_swing_high and curr_smc['Close'] < prev_smc['Low']:
                                stop_loss_smc = prev_smc['High'] + atr_buffer
                                risk_smc = stop_loss_smc - curr_smc['Close']
                                if risk_smc > 0:
                                    take_profit_smc = curr_smc['Close'] - (risk_smc * rr_target)
                                    smc_setup = "SMC SHORT"
                                    smc_entry = float(curr_smc['Close'])
                                    smc_stop_loss = float(stop_loss_smc)
                                    smc_take_profit = float(take_profit_smc)
                                    smc_rr = rr_target
            reason_text = "สัญญาณกลับตัว 15 นาทีสำหรับคริปโต โดยอิง RSI, Bollinger Bands และเทรนด์ 1H"
            if reasons:
                reason_text = " | ".join(reasons)
            return {
                'setup': setup_type,
                'confidence': conf,
                'current_price': price,
                'rsi': rsi,
                'atr': atr,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pattern': pattern,
                'reason': reason_text,
                'trend_1h': trend_1h,
                'trend_strength_1h': trend_strength_1h,
                'rvol': rvol,
                'atr_pct': atr_pct,
                'regime': regime,
                'risk_reward': rr,
                'win_prob': win_prob,
                'expected_move_pct': expected_move_pct,
                'expected_holding_bars_15m': expected_holding_bars_15m,
                'smc_setup': smc_setup,
                'smc_entry': smc_entry,
                'smc_stop_loss': smc_stop_loss,
                'smc_take_profit': smc_take_profit,
                'smc_rr': smc_rr
            }
        except Exception:
            return None

class EMACross15m:
    @staticmethod
    def analyze(symbol):
        if not is_crypto_symbol(symbol):
            return None
        try:
            sym = normalize_symbol(symbol)
            cached = _ema_cross_15m_get_cached(sym)
            best_fast_len = None
            best_slow_len = None
            opt_meta = None
            if isinstance(cached, dict):
                best_fast_len = cached.get("fast_len")
                best_slow_len = cached.get("slow_len")
                opt_meta = cached.get("opt_meta")
            yf_period = getattr(config, "EMA_CROSS_15M_YF_PERIOD", "30d")
            data_15m = get_yf_history(sym, period=str(yf_period), interval="15m", auto_adjust=True)
            if data_15m is None or data_15m.empty:
                data_15m = get_yf_history(sym, period="5d", interval="15m", auto_adjust=True)
            if data_15m is None or data_15m.empty or len(data_15m) < 120:
                return None
            df = _ema_cross_15m_prepare_df(data_15m)
            if df is None or df.empty:
                return None
            if best_fast_len is None or best_slow_len is None:
                opt_meta = optimize_best_ema_cross_15m(sym, df)
                best = opt_meta.get("best") if isinstance(opt_meta, dict) else None
                if isinstance(best, dict):
                    best_fast_len = best.get("fast_len")
                    best_slow_len = best.get("slow_len")
                    _ema_cross_15m_set_cached(sym, {"fast_len": best_fast_len, "slow_len": best_slow_len, "opt_meta": opt_meta})
            try:
                best_fast_len = int(best_fast_len) if best_fast_len is not None else 12
                best_slow_len = int(best_slow_len) if best_slow_len is not None else 26
            except Exception:
                best_fast_len, best_slow_len = 12, 26
            if best_fast_len < 2 or best_slow_len < 2 or best_fast_len >= best_slow_len:
                best_fast_len, best_slow_len = 12, 26
            df["EMA_FAST"] = df["Close"].ewm(span=best_fast_len, adjust=False).mean()
            df["EMA_SLOW"] = df["Close"].ewm(span=best_slow_len, adjust=False).mean()
            df["Zone"] = np.where(df["EMA_FAST"] > df["EMA_SLOW"], 1, -1)
            df["Zone_Change"] = pd.Series(df["Zone"], index=df.index).diff().fillna(0)
            cross_up_idx = df.index[df["Zone_Change"] == 2]
            cross_down_idx = df.index[df["Zone_Change"] == -2]
            last_cross_type = None
            last_cross_time = None
            if len(cross_up_idx) or len(cross_down_idx):
                last_up = cross_up_idx[-1] if len(cross_up_idx) else None
                last_down = cross_down_idx[-1] if len(cross_down_idx) else None
                if last_up is not None and (last_down is None or last_up > last_down):
                    last_cross_type = "BULL"
                    last_cross_time = last_up
                elif last_down is not None:
                    last_cross_type = "BEAR"
                    last_cross_time = last_down
            best_bt = None
            if isinstance(opt_meta, dict) and isinstance(opt_meta.get("best"), dict):
                best_bt = opt_meta.get("best")
            total_trades = best_bt.get("trades") if isinstance(best_bt, dict) else None
            hist_win_rate_pct = best_bt.get("win_rate_pct") if isinstance(best_bt, dict) else None
            hist_avg_rr = best_bt.get("avg_rr") if isinstance(best_bt, dict) else None
            latest = df.iloc[-1]
            if pd.isna(latest["ATR"]) or latest["ATR"] <= 0 or pd.isna(latest["Vol_Avg"]) or latest["Vol_Avg"] <= 0:
                return None
            entry_price = float(latest["Close"])
            ema_fast = float(latest["EMA_FAST"])
            ema_slow = float(latest["EMA_SLOW"])
            atr_val = float(latest["ATR"])
            rvol = float(latest["Volume"] / latest["Vol_Avg"])
            current_zone = "GREEN" if ema_fast > ema_slow else "RED"
            trend = "UP" if current_zone == "GREEN" else "DOWN"
            bars_since_cross = None
            if last_cross_time is not None:
                try:
                    idx_pos = df.index.get_loc(last_cross_time)
                    bars_since_cross = max(0, len(df) - idx_pos - 1)
                except Exception:
                    bars_since_cross = None
            signal = "WAIT"
            direction = None
            if current_zone == "GREEN" and last_cross_type == "BULL":
                direction = "BUY"
            elif current_zone == "RED" and last_cross_type == "BEAR":
                direction = "SELL"
            max_bars_since_cross = getattr(config, "EMA_CROSS_15M_MAX_BARS_SINCE_CROSS", 4)
            try:
                max_bars_since_cross = int(max_bars_since_cross)
            except Exception:
                max_bars_since_cross = 4
            if max_bars_since_cross < 0:
                max_bars_since_cross = 0
            if bars_since_cross is not None and bars_since_cross <= max_bars_since_cross and direction is not None:
                signal = direction
            require_slope = getattr(config, "EMA_CROSS_15M_REQUIRE_SLOW_SLOPE_CONFIRM", True)
            if require_slope and signal in ("BUY", "SELL") and len(df) >= 5:
                slow_now = float(df["EMA_SLOW"].iloc[-1])
                slow_prev = float(df["EMA_SLOW"].iloc[-4])
                slope = slow_now - slow_prev
                if signal == "BUY" and slope <= 0:
                    signal = "WAIT"
                elif signal == "SELL" and slope >= 0:
                    signal = "WAIT"
            stop_loss = None
            take_profit = None
            risk_reward = None
            if signal in ["BUY", "SELL"]:
                risk_dist = atr_val
                if risk_dist <= 0:
                    risk_dist = abs(entry_price) * 0.005
                reward_mult = getattr(config, "EMA_CROSS_15M_TP_MULT", 5.0)
                try:
                    reward_mult = float(reward_mult)
                except Exception:
                    reward_mult = 5.0
                if reward_mult <= 0:
                    reward_mult = 5.0
                if signal == "BUY":
                    stop_loss = entry_price - risk_dist
                    take_profit = entry_price + risk_dist * reward_mult
                else:
                    stop_loss = entry_price + risk_dist
                    take_profit = entry_price - risk_dist * reward_mult
            if stop_loss and take_profit and entry_price and entry_price != stop_loss:
                rr_val = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
                risk_reward = rr_val
            base_conf = 0.5
            if signal in ["BUY", "SELL"]:
                base_conf += 0.2
            if bars_since_cross is not None:
                if bars_since_cross <= 2:
                    base_conf += 0.15
                elif bars_since_cross <= 6:
                    base_conf += 0.05
            ema_gap_pct = abs(ema_fast - ema_slow) / entry_price * 100 if entry_price > 0 else 0
            if ema_gap_pct >= 0.3:
                base_conf += 0.1
            elif ema_gap_pct <= 0.15:
                base_conf -= 0.05
            if rvol >= 1.5:
                base_conf += 0.1
            elif rvol >= 1.2:
                base_conf += 0.05
            elif rvol <= 0.8:
                base_conf -= 0.1
            min_trades = getattr(config, "EMA_CROSS_15M_MIN_TRADES", 8)
            try:
                min_trades = int(min_trades)
            except Exception:
                min_trades = 8
            if min_trades < 1:
                min_trades = 1
            hist_prob = None
            if hist_win_rate_pct is not None and total_trades is not None:
                try:
                    trades_val = float(total_trades)
                    if trades_val > 0:
                        wins_est = (float(hist_win_rate_pct) / 100.0) * trades_val
                        losses_est = trades_val - wins_est
                        denom = wins_est + losses_est + 2.0
                        if denom > 0:
                            hist_prob = (wins_est + 1.0) / denom
                except Exception:
                    hist_prob = None
            dynamic_factor = base_conf
            if bars_since_cross is not None and bars_since_cross > 10:
                dynamic_factor -= 0.1
            if dynamic_factor < 0:
                dynamic_factor = 0
            if dynamic_factor > 1:
                dynamic_factor = 1
            if hist_prob is None:
                predicted_prob = dynamic_factor
            else:
                hist_weight = 0.0
                if total_trades is not None:
                    try:
                        trades_val = float(total_trades)
                        if trades_val > 0:
                            hist_weight = min(1.0, trades_val / float(min_trades))
                    except Exception:
                        hist_weight = 0.0
                predicted_prob = (hist_prob * hist_weight) + (dynamic_factor * (1.0 - hist_weight))
            if predicted_prob < 0:
                predicted_prob = 0
            if predicted_prob > 1:
                predicted_prob = 1
            confidence = predicted_prob * 100.0
            expected_move_pct = None
            expected_bars_15m = None
            if take_profit is not None and entry_price > 0:
                if signal == "BUY":
                    expected_move_pct = ((take_profit - entry_price) / entry_price) * 100
                elif signal == "SELL":
                    expected_move_pct = ((entry_price - take_profit) / entry_price) * 100
            atr_pct = (atr_val / entry_price) * 100 if entry_price > 0 else None
            if expected_move_pct is not None and atr_pct is not None and atr_pct > 0:
                approx_steps = abs(expected_move_pct) / atr_pct
                if approx_steps <= 0:
                    approx_steps = 1
                expected_bars_15m = int(max(1, min(48, round(approx_steps))))
            last_cross_time_str = last_cross_time.strftime('%Y-%m-%d %H:%M') if last_cross_time is not None else None
            return {
                'signal': signal,
                'zone': current_zone,
                'trend': trend,
                'last_cross_type': last_cross_type,
                'last_cross_time': last_cross_time_str,
                'bars_since_cross': bars_since_cross,
                'current_price': entry_price,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'ema_fast_length': best_fast_len,
                'ema_slow_length': best_slow_len,
                'atr': atr_val,
                'rvol': rvol,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': risk_reward,
                'confidence': confidence,
                'historical_trades': int(total_trades) if isinstance(total_trades, (int, float)) else None,
                'historical_win_rate': float(hist_win_rate_pct) if isinstance(hist_win_rate_pct, (int, float)) else None,
                'historical_avg_rr': float(hist_avg_rr) if isinstance(hist_avg_rr, (int, float)) else None,
                'predicted_win_prob': confidence,
                'expected_move_pct': expected_move_pct,
                'expected_holding_bars_15m': expected_bars_15m,
                'optimized': True if isinstance(opt_meta, dict) and isinstance(opt_meta.get('best'), dict) else False,
                'optimizer': opt_meta
            }
        except Exception:
            return None

def _actionzone_trend_1h(symbol):
    try:
        sym = normalize_symbol(symbol)
        period = getattr(config, "ACTIONZONE_15M_TREND_1H_PERIOD", getattr(config, "SHORT_TERM_15M_TREND_1H_PERIOD", "3mo"))
        df = get_yf_history(sym, period=str(period), interval="1h", auto_adjust=True)
        if df is None or df.empty or len(df) < 80:
            return None
        close = df["Close"].astype(float)
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        curr_close = float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else None
        ema50_now = float(ema50.iloc[-1]) if pd.notna(ema50.iloc[-1]) else None
        ema200_now = float(ema200.iloc[-1]) if pd.notna(ema200.iloc[-1]) else None
        if curr_close is None or ema50_now is None:
            return None
        trend = "SIDEWAYS"
        if curr_close > ema50_now:
            trend = "UP"
        elif curr_close < ema50_now:
            trend = "DOWN"
        strength = "WEAK"
        if len(ema50) >= 6 and pd.notna(ema50.iloc[-6]):
            slope = ema50_now - float(ema50.iloc[-6])
            if trend == "UP" and slope > 0:
                strength = "STRONG"
            elif trend == "DOWN" and slope < 0:
                strength = "STRONG"
        if ema200_now is not None and curr_close is not None:
            spread = abs(ema50_now - ema200_now) / curr_close * 100 if curr_close > 0 else 0
            if spread >= 0.6:
                strength = "STRONG"
        return {
            "trend": trend,
            "strength": strength,
            "ema50": ema50_now,
            "ema200": ema200_now,
        }
    except Exception:
        return None

def _actionzone_15m_alert(symbol):
    try:
        sym = normalize_symbol(symbol)
        yf_period = getattr(config, "EMA_CROSS_15M_YF_PERIOD", "30d")
        data_15m = get_yf_history(sym, period=str(yf_period), interval="15m", auto_adjust=True)
        if data_15m is None or data_15m.empty:
            data_15m = get_yf_history(sym, period="5d", interval="15m", auto_adjust=True)
        if data_15m is None or data_15m.empty or len(data_15m) < 120:
            return None
        df = _ema_cross_15m_prepare_df(data_15m)
        if df is None or df.empty:
            return None
        best_fast_len = None
        best_slow_len = None
        cached = _ema_cross_15m_get_cached(sym)
        if isinstance(cached, dict):
            best_fast_len = cached.get("fast_len")
            best_slow_len = cached.get("slow_len")
        use_opt = getattr(config, "ACTIONZONE_15M_USE_OPTIMIZATION", True)
        if use_opt and (best_fast_len is None or best_slow_len is None):
            opt_meta = optimize_best_ema_cross_15m(sym, df)
            best = opt_meta.get("best") if isinstance(opt_meta, dict) else None
            if isinstance(best, dict):
                best_fast_len = best.get("fast_len")
                best_slow_len = best.get("slow_len")
                _ema_cross_15m_set_cached(sym, {"fast_len": best_fast_len, "slow_len": best_slow_len, "opt_meta": opt_meta})
        try:
            best_fast_len = int(best_fast_len) if best_fast_len is not None else 12
            best_slow_len = int(best_slow_len) if best_slow_len is not None else 26
        except Exception:
            best_fast_len, best_slow_len = 12, 26
        if best_fast_len < 2 or best_slow_len < 2 or best_fast_len >= best_slow_len:
            best_fast_len, best_slow_len = 12, 26
        smooth = getattr(config, "ACTIONZONE_15M_SMOOTH", 1)
        try:
            smooth = int(smooth)
        except Exception:
            smooth = 1
        if smooth < 1:
            smooth = 1
        close = df["Close"].astype(float)
        x_confirm = close.ewm(span=smooth, adjust=False).mean() if smooth > 1 else close
        fast = x_confirm.ewm(span=best_fast_len, adjust=False).mean()
        slow = x_confirm.ewm(span=best_slow_len, adjust=False).mean()
        bull = fast > slow
        bear = fast < slow
        green = bull & (x_confirm > fast)
        blue = bear & (x_confirm > fast) & (x_confirm > slow)
        lblue = bear & (x_confirm > fast) & (x_confirm < slow)
        red = bear & (x_confirm < fast)
        orange = bull & (x_confirm < fast) & (x_confirm < slow)
        yellow = bull & (x_confirm < fast) & (x_confirm > slow)
        zone = np.where(green, "GREEN",
                np.where(blue, "BLUE",
                np.where(lblue, "LBLUE",
                np.where(red, "RED",
                np.where(orange, "ORANGE",
                np.where(yellow, "YELLOW", "NEUTRAL"))))))
        zone_now = zone[-1]
        buy_signal = green & (~green.shift(1).fillna(False))
        sell_signal = red & (~red.shift(1).fillna(False))
        raw_signal = "WAIT"
        if bool(buy_signal.iloc[-1]):
            raw_signal = "BUY"
        elif bool(sell_signal.iloc[-1]):
            raw_signal = "SELL"
        last_buy_time = buy_signal[buy_signal].index[-1] if buy_signal.any() else None
        last_sell_time = sell_signal[sell_signal].index[-1] if sell_signal.any() else None
        last_signal_type = None
        last_signal_time = None
        if last_buy_time is not None and (last_sell_time is None or last_buy_time > last_sell_time):
            last_signal_type = "BUY"
            last_signal_time = last_buy_time
        elif last_sell_time is not None:
            last_signal_type = "SELL"
            last_signal_time = last_sell_time
        bars_since_signal = None
        if last_signal_time is not None:
            try:
                idx_pos = df.index.get_loc(last_signal_time)
                bars_since_signal = max(0, len(df) - idx_pos - 1)
            except Exception:
                bars_since_signal = None
        signal = "WAIT"
        if raw_signal in ("BUY", "SELL"):
            signal = raw_signal
        elif last_signal_type is not None:
            alert_bars = getattr(config, "ACTIONZONE_15M_ALERT_BARS", 2)
            try:
                alert_bars = int(alert_bars)
            except Exception:
                alert_bars = 2
            if alert_bars < 0:
                alert_bars = 0
            if bars_since_signal is not None and bars_since_signal <= alert_bars:
                signal = last_signal_type
        trend_1h = _actionzone_trend_1h(sym)
        trend_dir = trend_1h.get("trend") if isinstance(trend_1h, dict) else None
        trend_strength = trend_1h.get("strength") if isinstance(trend_1h, dict) else None
        trend_ok = True
        if signal == "BUY" and trend_dir not in (None, "UP"):
            trend_ok = False
        if signal == "SELL" and trend_dir not in (None, "DOWN"):
            trend_ok = False
        filtered_signal = signal if trend_ok else "WAIT"
        entry_price = float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else None
        ema_fast = float(fast.iloc[-1]) if pd.notna(fast.iloc[-1]) else None
        ema_slow = float(slow.iloc[-1]) if pd.notna(slow.iloc[-1]) else None
        rvol = None
        if pd.notna(df["Vol_Avg"].iloc[-1]) and df["Vol_Avg"].iloc[-1] > 0:
            rvol = float(df["Volume"].iloc[-1] / df["Vol_Avg"].iloc[-1])
        ema_gap_pct = None
        if entry_price and ema_fast is not None and ema_slow is not None:
            ema_gap_pct = abs(ema_fast - ema_slow) / entry_price * 100
        atr_val = None
        if pd.notna(df["ATR"].iloc[-1]):
            atr_val = float(df["ATR"].iloc[-1])
        conf = 0.45
        if filtered_signal in ("BUY", "SELL"):
            conf += 0.2
        if trend_ok and trend_dir in ("UP", "DOWN"):
            conf += 0.15
        if ema_gap_pct is not None:
            if ema_gap_pct >= 0.3:
                conf += 0.1
            elif ema_gap_pct <= 0.15:
                conf -= 0.05
        if rvol is not None:
            if rvol >= 1.2:
                conf += 0.05
            elif rvol <= 0.8:
                conf -= 0.05
        if conf < 0:
            conf = 0
        if conf > 1:
            conf = 1
        stop_loss = None
        if entry_price is not None and atr_val is not None and atr_val > 0:
            if filtered_signal == "BUY":
                stop_loss = entry_price - atr_val
            elif filtered_signal == "SELL":
                stop_loss = entry_price + atr_val
        alert_active = filtered_signal in ("BUY", "SELL")
        recommendation = "WAIT"
        if filtered_signal == "BUY":
            recommendation = "BUY"
        elif filtered_signal == "SELL":
            recommendation = "SELL"
        elif signal in ("BUY", "SELL") and not trend_ok:
            recommendation = "WAIT (Trend mismatch)"
        last_signal_time_str = last_signal_time.strftime("%Y-%m-%d %H:%M") if last_signal_time is not None else None
        return {
            "signal": filtered_signal,
            "raw_signal": signal,
            "zone": zone_now,
            "last_signal_type": last_signal_type,
            "last_signal_time": last_signal_time_str,
            "bars_since_signal": bars_since_signal,
            "trend_1h": trend_dir,
            "trend_strength_1h": trend_strength,
            "trend_alignment": trend_ok,
            "current_price": entry_price,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_fast_length": best_fast_len,
            "ema_slow_length": best_slow_len,
            "smooth": smooth,
            "ema_gap_pct": ema_gap_pct,
            "rvol": rvol,
            "alert": alert_active,
            "confidence": conf * 100.0,
            "win_prob": conf * 100.0,
            "stop_loss": stop_loss,
            "atr": atr_val,
            "recommendation": recommendation
        }
    except Exception:
        return None

def _order_block_levels_15m(symbol):
    try:
        sym = normalize_symbol(symbol)
        yf_period = getattr(config, "EMA_CROSS_15M_YF_PERIOD", "30d")
        data_15m = get_yf_history(sym, period=str(yf_period), interval="15m", auto_adjust=True)
        if data_15m is None or data_15m.empty:
            data_15m = get_yf_history(sym, period="5d", interval="15m", auto_adjust=True)
        if data_15m is None or data_15m.empty:
            return None
        df = data_15m[["Open", "High", "Low", "Close", "Volume"]].copy()
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = df.index.tz_localize(None)
            except Exception:
                pass
        length = getattr(config, "ORDERBLOCK_15M_PIVOT_LENGTH", 5)
        max_blocks = getattr(config, "ORDERBLOCK_15M_MAX_BLOCKS", 6)
        mitigation = str(getattr(config, "ORDERBLOCK_15M_MITIGATION", "wick")).strip().lower()
        try:
            length = int(length)
        except Exception:
            length = 5
        try:
            max_blocks = int(max_blocks)
        except Exception:
            max_blocks = 6
        if length < 1:
            length = 1
        if max_blocks < 1:
            max_blocks = 1
        if len(df) < length * 2 + 5:
            return None
        high = df["High"].astype(float).reset_index(drop=True)
        low = df["Low"].astype(float).reset_index(drop=True)
        close = df["Close"].astype(float).reset_index(drop=True)
        volume = df["Volume"].astype(float).reset_index(drop=True)
        upper = high.rolling(length).max()
        lower = low.rolling(length).min()
        os_state = 0
        bull_blocks = []
        bear_blocks = []
        for i in range(length, len(df) - length):
            idx = i - length
            upper_i = upper.iloc[i]
            lower_i = lower.iloc[i]
            if pd.notna(upper_i) and high.iloc[idx] > upper_i:
                os_state = 0
            elif pd.notna(lower_i) and low.iloc[idx] < lower_i:
                os_state = 1
            window = volume.iloc[i - length:i + length + 1]
            if window.size > 0 and volume.iloc[i] == window.max():
                if os_state == 1:
                    top = (high.iloc[idx] + low.iloc[idx]) / 2.0
                    btm = low.iloc[idx]
                    avg = (top + btm) / 2.0
                    bull_blocks.insert(0, {"top": top, "btm": btm, "avg": avg})
                else:
                    top = high.iloc[idx]
                    btm = (high.iloc[idx] + low.iloc[idx]) / 2.0
                    avg = (top + btm) / 2.0
                    bear_blocks.insert(0, {"top": top, "btm": btm, "avg": avg})
            if mitigation == "close":
                target_bull = close.iloc[max(0, i - length):i + 1].min()
                target_bear = close.iloc[max(0, i - length):i + 1].max()
            else:
                target_bull = low.iloc[max(0, i - length):i + 1].min()
                target_bear = high.iloc[max(0, i - length):i + 1].max()
            if pd.notna(target_bull):
                bull_blocks = [b for b in bull_blocks if not (target_bull < b["btm"])]
            if pd.notna(target_bear):
                bear_blocks = [b for b in bear_blocks if not (target_bear > b["top"])]
            bull_blocks = bull_blocks[:max_blocks]
            bear_blocks = bear_blocks[:max_blocks]
        current_price = float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else None
        support_levels = [float(b["avg"]) for b in bull_blocks if isinstance(b, dict) and isinstance(b.get("avg"), (int, float))]
        resistance_levels = [float(b["avg"]) for b in bear_blocks if isinstance(b, dict) and isinstance(b.get("avg"), (int, float))]
        nearest_support = None
        nearest_resistance = None
        if current_price is not None:
            supports_below = [lvl for lvl in support_levels if lvl <= current_price]
            resist_above = [lvl for lvl in resistance_levels if lvl >= current_price]
            if supports_below:
                nearest_support = max(supports_below)
            if resist_above:
                nearest_resistance = min(resist_above)
        last_time = df.index[-1].strftime("%Y-%m-%d %H:%M") if isinstance(df.index, pd.DatetimeIndex) else None
        return {
            "current_price": current_price,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "updated_at": last_time
        }
    except Exception:
        return None

class QuantumSovereign4H:
    @staticmethod
    def get_data_4h(symbol):
        try:
            sym = normalize_symbol(symbol)
            df = get_yf_history(sym, period='60d', interval='1h', auto_adjust=True)
            if df is None or df.empty:
                return None
            agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df_4h = df.resample('4h').agg(agg_dict).dropna()
            return df_4h
        except Exception:
            return None

    @staticmethod
    def calculate_indicators(df):
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean()
        df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(14).mean()
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0).rolling(14).mean())
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = df['Low'].diff().clip(lower=0)
        atr = df['ATR']
        p_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
        m_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
        dx = (abs(p_di - m_di) / (p_di + m_di + 1e-9)) * 100
        df['ADX'] = dx.rolling(14).mean()
        rolling_high = df['High'].rolling(20).max()
        rolling_low = df['Low'].rolling(20).min()
        df['Trailing_Long'] = rolling_high - (3.0 * df['ATR'])
        df['Trailing_Short'] = rolling_low + (3.0 * df['ATR'])
        return df

    @staticmethod
    def analyze(symbol, balance=10000, baseline_ema_length=None):
        df = QuantumSovereign4H.get_data_4h(symbol)
        if df is None or len(df) < 100:
            return {'error': 'ข้อมูลไม่เพียงพอ'}
        df = QuantumSovereign4H.calculate_indicators(df)
        baseline_len = None
        baseline_ema = None
        baseline_phase = None
        try:
            if baseline_ema_length is not None:
                baseline_len = int(baseline_ema_length)
                if baseline_len >= 2 and baseline_len <= 400:
                    df['EMA_BASE'] = df['Close'].ewm(span=baseline_len, adjust=False).mean()
                    baseline_ema = float(df['EMA_BASE'].iloc[-1]) if pd.notna(df['EMA_BASE'].iloc[-1]) else None
        except Exception:
            baseline_len = None
            baseline_ema = None
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        if baseline_len is not None and baseline_ema is not None:
            baseline_phase = "BULL" if curr['Close'] > baseline_ema else "BEAR"
        trend = "SIDEWAYS"
        if curr['Close'] > curr['EMA50'] > curr['EMA200']:
            trend = "BULLISH (ขาขึ้นแข็งแกร่ง)"
        elif curr['Close'] < curr['EMA50'] < curr['EMA200']:
            trend = "BEARISH (ขาลงแข็งแกร่ง)"
        trend_strength = "WEAK"
        if curr['ADX'] > 25:
            trend_strength = "STRONG"
        if curr['ADX'] > 50:
            trend_strength = "SUPER STRONG"
        action = "WAIT / HOLD"
        reason = "ไม่มีสัญญาณใหม่"
        score = 0
        if trend.startswith("BULLISH"):
            score += 40
            if 50 < curr['RSI'] < 70:
                score += 20
            if curr['ADX'] > 25:
                score += 20
            if (prev['Close'] <= prev['EMA50'] and curr['Close'] > curr['EMA50']) or (curr['Close'] > df['High'].iloc[-20:-1].max()):
                action = "🟢 BUY NOW"
                reason = "Trend ขาขึ้น + Breakout/Rebound + Momentum สนับสนุน"
            else:
                action = "🔵 HOLD (ถือต่อ)"
                reason = "เทรนด์ยังเป็นขาขึ้น ให้ถือรันกำไร (Let Profit Run)"
        elif trend.startswith("BEARISH"):
            score += 40
            if 30 < curr['RSI'] < 50:
                score += 20
            if curr['ADX'] > 25:
                score += 20
            if (prev['Close'] >= prev['EMA50'] and curr['Close'] < curr['EMA50']) or (curr['Close'] < df['Low'].iloc[-20:-1].min()):
                action = "🔴 SELL NOW"
                reason = "Trend ขาลง + Breakdown + Momentum สนับสนุน"
            else:
                action = "🔵 HOLD SHORT"
                reason = "เทรนด์ยังเป็นขาลง ให้ถือ Short ต่อไป"
        if "BULLISH" in trend:
            exit_point = curr['Trailing_Long']
            stop_loss_fixed = curr['Close'] - (2.0 * curr['ATR'])
        else:
            exit_point = curr['Trailing_Short']
            stop_loss_fixed = curr['Close'] + (2.0 * curr['ATR'])
        critical_alert = ""
        if "BULLISH" in trend and curr['Close'] < exit_point:
            action = "⚠️ CLOSE ALL BUY (ขายทิ้งทันที)"
            critical_alert = "ราคาหลุด Trailing Stop -> เทรนด์จบแล้ว"
        elif "BEARISH" in trend and curr['Close'] > exit_point:
            action = "⚠️ CLOSE ALL SELL (ปิด Short ทันที)"
            critical_alert = "ราคาเบรค Trailing Stop -> เทรนด์จบแล้ว"
        win_prob = 0.55 + (score / 400)
        kelly_pct = 0
        if "NOW" in action:
            kelly_raw = (3.0 * win_prob - (1 - win_prob)) / 3.0
            kelly_pct = max(0, kelly_raw * 0.5) * 100
        return {
            'timeframe': '4H (Ultimate Swing)',
            'trend_status': f"{trend} | Power: {trend_strength} (ADX {curr['ADX']:.1f})",
            'recommendation': action,
            'critical_alert': critical_alert,
            'current_price': float(curr['Close']) if not pd.isna(curr['Close']) else None,
            'trailing_stop': float(exit_point) if not pd.isna(exit_point) else None,
            'entry_stop_loss': float(stop_loss_fixed) if not pd.isna(stop_loss_fixed) else None,
            'baseline_ema_length': baseline_len,
            'baseline_ema': baseline_ema,
            'baseline_phase': baseline_phase,
            'score_confidence': f"{score}/100",
            'position_size_suggest': f"{kelly_pct:.1f}% ของพอร์ต" if kelly_pct > 0 else "N/A",
            'reason': reason
        }

class UniverseAnalyzer:
    def __init__(self, target_symbol, is_crypto=False):
        self.target_symbol = target_symbol
        self.is_crypto = is_crypto
        self.assets = {
            'Target': target_symbol,
            'Gold': 'GC=F',
            'Oil': 'CL=F',
            'Bond_10Y': '^TNX',
            'Dollar': 'DX-Y.NYB',
            'S&P500': '^GSPC'
        }
        if self.is_crypto:
            self.assets['Bitcoin'] = 'BTC-USD'
        self.df = self._fetch_data()

    def _fetch_data(self):
        cache_key = ("universe", normalize_symbol(self.target_symbol), bool(self.is_crypto))
        cached = _YF_UNIVERSE_CACHE.get(cache_key)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached.copy()
        session = _get_thread_curl_session()
        tickers = list(self.assets.values())
        data = yf.download(tickers, period="1y", progress=False, session=session)['Close']
        inv_map = {v: k for k, v in self.assets.items()}
        data.rename(columns=inv_map, inplace=True)
        data.ffill(inplace=True)
        data.dropna(inplace=True)
        _YF_UNIVERSE_CACHE.set(cache_key, data)
        return data.copy()

    def analyze_correlations(self):
        returns = self.df.pct_change()
        corrs = returns.corr()['Target'].drop('Target')
        return corrs

    def analyze_macro_trends(self):
        trends = {}
        for name in self.df.columns:
            if name == 'Target':
                continue
            price = self.df[name]
            sma50 = price.rolling(50).mean().iloc[-1]
            current = price.iloc[-1]
            trends[name] = "UP" if current > sma50 else "DOWN"
        return trends

    def calculate_regime_score(self):
        corrs = self.analyze_correlations()
        trends = self.analyze_macro_trends()
        score = 0
        total_factors = 0
        for asset, trend in trends.items():
            correlation = corrs.get(asset, 0)
            if abs(correlation) < 0.2:
                continue
            total_factors += 1
            if correlation > 0:
                if trend == "UP":
                    score += 1
                else:
                    score -= 1
            else:
                if trend == "DOWN":
                    score += 1
                else:
                    score -= 1
        if total_factors == 0:
            final_prob = 50
        else:
            final_prob = ((score / total_factors) + 1) * 50
        return final_prob, trends

    def predict(self):
        price_current = float(self.df['Target'].iloc[-1])
        macro_prob, macro_trends = self.calculate_regime_score()
        delta = self.df['Target'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.iloc[-1])
        final_score = (macro_prob * 0.6) + ((100 - rsi) * 0.4)
        if rsi > 70:
            final_score = min(final_score, 40)
        recommendation = "WAIT"
        if final_score > 75:
            recommendation = "STRONG BUY"
        elif final_score > 60:
            recommendation = "ACCUMULATE (ทยอยสะสม)"
        elif final_score < 40:
            recommendation = "SELL / AVOID"
        return {
            'price_current': price_current,
            'macro_prob': float(macro_prob),
            'macro_trends': macro_trends,
            'rsi': rsi,
            'final_score': float(final_score),
            'recommendation': recommendation
        }

class SovereignFetcher:
    def __init__(self):
        self.session = _get_thread_curl_session()

    def fetch_full_data(self, ticker, period="2y"):
        try:
            sym = normalize_symbol(ticker)
            df = get_yf_history(sym, period=period, interval=None, auto_adjust=True)
            if df is None or df.empty:
                return None, None
            df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                df.set_index("date", inplace=True)
            info = yf.Ticker(sym, session=self.session).info
            fundamentals = {
                "peRatio": info.get("forwardPE", info.get("trailingPE", 999)),
                "marketCap": info.get("marketCap", 0),
                "dividendYield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "sector": info.get("sector", "Unknown"),
            }
            return df, fundamentals
        except Exception:
            return None, None

class HybridSovereignEngine:
    def __init__(self, portfolio_value=100000, risk_per_trade=0.02):
        self.fetcher = SovereignFetcher()
        self.portfolio_value = portfolio_value
        self.risk_per_trade = risk_per_trade

    def analyze_asset(self, ticker, alias):
        df, fund = self.fetcher.fetch_full_data(ticker)
        if df is None or len(df) < 200:
            return None
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi14"] = 100 - (100 / (1 + rs))
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr22"] = tr.rolling(22).mean()
        adx_df = pd.DataFrame({
            "High": df["high"],
            "Low": df["low"],
            "Close": df["close"],
        })
        adx_series = QuantumHunterStrategy.calculate_adx(adx_df)
        df["adx14"] = adx_series
        last = df.iloc[-1]
        close = float(last["close"])
        atr = float(last["atr22"]) if pd.notna(last["atr22"]) else None
        rsi = float(last["rsi14"]) if pd.notna(last["rsi14"]) else None
        ema50 = float(last["ema50"]) if pd.notna(last["ema50"]) else None
        ema200 = float(last["ema200"]) if pd.notna(last["ema200"]) else None
        adx = float(last["adx14"]) if pd.notna(last["adx14"]) else None
        if atr is None or rsi is None or ema50 is None or ema200 is None or adx is None:
            return None
        score = 0
        if close > ema200:
            score += 20
        if close > ema50:
            score += 10
        if adx > 25:
            score += 10
        dist_from_ema200 = ((close - ema200) / ema200) * 100
        if 40 <= rsi <= 60:
            score += 15
        if -10 < dist_from_ema200 < 10:
            score += 15
        pe = fund.get("peRatio", 999)
        if pe < 25:
            score += 30
        elif pe < 50:
            score += 15
        else:
            score -= 20
        stop_loss = close - (3.0 * atr)
        risk_per_share = close - stop_loss
        money_at_risk = self.portfolio_value * self.risk_per_trade
        qty = int(money_at_risk / risk_per_share) if risk_per_share > 0 else 0
        position_value = qty * close
        volatility_pct = (atr / close) * 100 if close > 0 else None
        action = "WAIT"
        if score >= 70:
            action = "🟢 STRONG BUY"
        elif score >= 50:
            action = "🟡 ACCUMULATE"
        elif score < 30:
            action = "🔴 AVOID/SELL"
        bubble_risk = False
        bubble_reasons = []
        if isinstance(pe, (int, float)) and pe != 0:
            if pe > 60:
                bubble_risk = True
                bubble_reasons.append("P/E สูงมากเกิน 60 เท่า")
        if dist_from_ema200 > 35:
            bubble_risk = True
            bubble_reasons.append("ราคาทะลุเส้น EMA200 เกิน 35%")
        if volatility_pct is not None and volatility_pct > 12:
            if is_crypto_symbol(ticker) or action.startswith("🟢") or action.startswith("🟡"):
                bubble_risk = True
                bubble_reasons.append("ความผันผวนรายวันสูงเกิน 12%")
        return {
            "asset": alias,
            "symbol": ticker,
            "sector": fund.get("sector", "Unknown"),
            "price": close,
            "pe": pe,
            "score": score,
            "action": action,
            "stop_loss": stop_loss,
            "suggested_qty": qty,
            "invest_amount": position_value,
            "risk_note": f"ATR={atr:.2f}" if atr is not None else "",
            "volatility_pct": volatility_pct,
            "dist_from_ema200": dist_from_ema200,
            "adx": adx,
            "rsi": rsi,
            "bubble_risk": bubble_risk,
            "bubble_reason": "; ".join(bubble_reasons) if bubble_risk else "",
        }

def classify_asset_type(symbol, alias, sector):
    s = (symbol or "").upper()
    a = (alias or "").upper()
    sec = (sector or "").upper()
    if is_crypto_symbol(symbol):
        return "คริปโตหลัก"
    if s == "GC=F" or "GOLD" in a:
        return "ทองคำ / Safe Haven"
    if s.endswith(".BK"):
        return "หุ้นไทย"
    if "FINANCIAL" in sec:
        return "หุ้นการเงินต่างประเทศ"
    return "หุ้นต่างประเทศขนาดใหญ่"

def get_global_sovereign_overview():
    portfolio_cash = 100000
    risk_tolerance = 0.02
    engine = HybridSovereignEngine(portfolio_cash, risk_tolerance)
    universe = {
        "NVDA": "Nvidia (AI)",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "AMZN": "Amazon",
        "META": "Meta Platforms",
        "TSLA": "Tesla",
        "AAPL": "Apple",
        "AVGO": "Broadcom",
        "SPY": "S&P 500 ETF",
        "QQQ": "Nasdaq 100 ETF",
        "XLF": "Financial Select Sector ETF",
        "XLE": "Energy Select Sector ETF",
        "EEM": "Emerging Markets ETF",
        "TLT": "US 20Y Treasury ETF",
        "CPALL.BK": "CP All (TH)",
        "BDMS.BK": "BDMS (TH)",
        "AOT.BK": "AOT (TH)",
        "PTT.BK": "PTT (TH)",
        "KBANK.BK": "KBANK (TH)",
        "ADVANC.BK": "ADVANC (TH)",
        "GC=F": "Gold Futures",
        "SI=F": "Silver Futures",
        "CL=F": "Crude Oil Futures",
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
    }

    def make_dynamic_asset_name(item):
        symbol = (item.get("symbol") or "").upper()
        base = universe.get(symbol, item.get("asset") or symbol)
        action = (item.get("action") or "").upper()
        asset_type = item.get("asset_type") or ""
        tags = []
        if "STRONG BUY" in action:
            tags.append("ตัวโฟกัสฝั่งซื้อ")
        elif "ACCUMULATE" in action:
            tags.append("ทยอยสะสม")
        elif "AVOID" in action or "SELL" in action:
            tags.append("ลดน้ำหนัก/หลีกเลี่ยง")
        elif action == "WAIT":
            tags.append("เฝ้าดูจังหวะ")
        if item.get("bubble_risk"):
            tags.append("เสี่ยงฟองสบู่")
        if is_crypto_symbol(symbol):
            tags.append("คริปโตหลัก")
        if asset_type:
            tags.append(asset_type)
        if tags:
            return f"{base} ({' • '.join(tags)})"
        return base

    items = []
    for ticker, alias in universe.items():
        res = engine.analyze_asset(ticker, alias)
        if res:
            items.append(res)
    if not items:
        summary = {
            "strong_buy": 0,
            "accumulate": 0,
            "avoid": 0,
            "wait": 0,
            "mood": "NEUTRAL",
        }
        return {
            "portfolio_base": portfolio_cash,
            "risk_per_trade": risk_tolerance,
            "items": [],
            "summary": summary,
            "type_summary": [],
            "bubble_assets": [],
        }
    for item in items:
        item["asset_type"] = classify_asset_type(item["symbol"], item["asset"], item["sector"])
    type_stats = {}
    for item in items:
        t = item.get("asset_type", "อื่น ๆ")
        if t not in type_stats:
            type_stats[t] = {
                "count": 0,
                "score_sum": 0.0,
                "strong_buy": 0,
                "accumulate": 0,
                "avoid": 0,
            }
        stat = type_stats[t]
        stat["count"] += 1
        score_val = item.get("score")
        if isinstance(score_val, (int, float)):
            stat["score_sum"] += score_val
        action = item.get("action", "")
        if isinstance(action, str):
            if action.startswith("🟢"):
                stat["strong_buy"] += 1
            elif action.startswith("🟡"):
                stat["accumulate"] += 1
            elif action.startswith("🔴"):
                stat["avoid"] += 1
    type_summary = []
    for t, stat in type_stats.items():
        count = stat["count"]
        avg_score = stat["score_sum"] / count if count else 0.0
        strong = stat["strong_buy"]
        acc = stat["accumulate"]
        avoid_type = stat["avoid"]
        stance = "เฝ้าดูสถานการณ์"
        if strong >= 1 or avg_score >= 70:
            stance = "เน้นซื้อ/ทยอยสะสม"
        elif avg_score >= 50:
            stance = "ถือ/เพิ่มเล็กน้อย"
        elif avg_score < 40 and avoid_type > 0:
            stance = "ลดน้ำหนัก/หลีกเลี่ยง"
        type_summary.append({
            "type": t,
            "stance": stance,
            "avg_score": avg_score,
            "strong_buy": strong,
                "accumulate": acc,
                "avoid": avoid_type,
                "count": count,
        })
    preferred_types = set()
    avoid_types = set()
    for ts in type_summary:
        stance = (ts.get("stance") or "").strip()
        t = ts.get("type")
        if not t:
            continue
        if "เน้นซื้อ" in stance or "ทยอยสะสม" in stance or "ถือ/เพิ่มเล็กน้อย" in stance:
            preferred_types.add(t)
        elif "ลดน้ำหนัก" in stance or "หลีกเลี่ยง" in stance:
            avoid_types.add(t)
    focus_items = []
    seen_symbols = set()
    for item in items:
        sym = item.get("symbol")
        if not sym:
            continue
        t = item.get("asset_type", "อื่น ๆ")
        action = item.get("action", "")
        bubble_flag = bool(item.get("bubble_risk"))
        if isinstance(action, str):
            if t in preferred_types and (action.startswith("🟢") or action.startswith("🟡")):
                if sym not in seen_symbols:
                    focus_items.append(item)
                    seen_symbols.add(sym)
                continue
            if t in avoid_types and action.startswith("🔴"):
                if sym not in seen_symbols:
                    focus_items.append(item)
                    seen_symbols.add(sym)
                continue
        if bubble_flag and sym not in seen_symbols:
            focus_items.append(item)
            seen_symbols.add(sym)
    if not focus_items:
        focus_items = sorted(
            items,
            key=lambda x: x.get("score") if isinstance(x.get("score"), (int, float)) else 0,
            reverse=True
        )[:10]
    else:
        focus_items.sort(key=lambda x: x.get("score") if isinstance(x.get("score"), (int, float)) else 0, reverse=True)
    for item in focus_items:
        item["asset"] = make_dynamic_asset_name(item)
    strong_buy = sum(1 for x in focus_items if isinstance(x.get("action", ""), str) and x["action"].startswith("🟢"))
    accumulate = sum(1 for x in focus_items if isinstance(x.get("action", ""), str) and x["action"].startswith("🟡"))
    avoid = sum(1 for x in focus_items if isinstance(x.get("action", ""), str) and x["action"].startswith("🔴"))
    wait = sum(1 for x in focus_items if x.get("action") == "WAIT")
    mood = "NEUTRAL"
    if strong_buy + accumulate > avoid + wait:
        mood = "BULLISH"
    elif avoid > strong_buy + accumulate:
        mood = "BEARISH"
    bubble_assets = []
    for item in items:
        if item.get("bubble_risk"):
            bubble_assets.append({
                "asset": item.get("asset"),
                "symbol": item.get("symbol"),
                "asset_type": item.get("asset_type"),
                "reason": item.get("bubble_reason", ""),
                "score": item.get("score"),
                "pe": item.get("pe"),
            })
    summary = {
        "strong_buy": strong_buy,
        "accumulate": accumulate,
        "avoid": avoid,
        "wait": wait,
        "mood": mood,
    }
    return {
        "portfolio_base": portfolio_cash,
        "risk_per_trade": risk_tolerance,
        "items": focus_items,
        "summary": summary,
        "type_summary": type_summary,
        "bubble_assets": bubble_assets,
    }

def generate_gemini_particle_a_analysis(info, data, resonance_score, phase_status, support, resistance, vol_status):
    latest = data.iloc[-1]
    price = float(latest["Close"])
    name = info.get("name", "")
    band_width = None
    price_pos = None
    try:
        band_width = float(resistance) - float(support)
        if band_width > 0:
            price_pos = (price - float(support)) / band_width
    except Exception:
        band_width = None
        price_pos = None
    vol_text = "วอลุ่มปกติ"
    if vol_status == "High":
        vol_text = "วอลุ่มหนาแน่น"
    elif vol_status == "Low":
        vol_text = "วอลุ่มบาง"
    ui_signal = "Hold"
    if resonance_score is not None:
        if resonance_score >= 60:
            if price_pos is None or price_pos <= 0.8:
                ui_signal = "Buy"
        elif resonance_score <= -60:
            if price_pos is None or price_pos >= 0.2:
                ui_signal = "Sell"
    parts = []
    parts.append(f"{name} Phase {phase_status} Resonance {resonance_score:.2f} ราคา {price:.2f}")
    if price_pos is not None:
        parts.append(f"ตำแหน่งราคาประมาณ {price_pos*100:.1f}% ระหว่างแนวรับ-แนวต้าน")
    parts.append(f"สถานะวอลุ่ม: {vol_text}")
    text = " | ".join(parts)
    return ui_signal, text


def build_prediction_summary(short_term_plan, sniper_plan, quantum_plan, ema_plan, sovereign_plan, macro_analysis, resonance_score, phase_status, crypto_plan=None):
    up_score = 0.0
    down_score = 0.0
    conf_sum = 0.0
    conf_weight = 0.0

    def add_conf(val):
        nonlocal conf_sum, conf_weight
        if isinstance(val, (int, float)):
            conf_sum += float(val)
            conf_weight += 1.0

    def process_plan(plan, setup_key="setup", conf_key="confidence"):
        nonlocal up_score, down_score
        if not isinstance(plan, dict):
            return
        setup = str(plan.get(setup_key, "")).upper()
        if "BUY" in setup or "LONG" in setup:
            up_score += 1.0
        elif "SELL" in setup or "SHORT" in setup:
            down_score += 1.0
        add_conf(plan.get(conf_key))

    process_plan(short_term_plan)
    process_plan(sniper_plan)
    process_plan(quantum_plan)
    if isinstance(ema_plan, dict):
        ema_proxy = {
            "setup": ema_plan.get("signal"),
            "confidence": ema_plan.get("predicted_win_prob", ema_plan.get("confidence")),
        }
        process_plan(ema_proxy)
        add_conf(ema_plan.get("predicted_win_prob", ema_plan.get("confidence")))
    if isinstance(crypto_plan, dict):
        process_plan(crypto_plan)
    if isinstance(sovereign_plan, dict):
        reco = str(sovereign_plan.get("recommendation", "")).upper()
        if "BUY" in reco:
            up_score += 1.5
        elif "SELL" in reco:
            down_score += 1.5
    if isinstance(macro_analysis, dict):
        macro_reco = str(macro_analysis.get("recommendation", "")).upper()
        if "STRONG BUY" in macro_reco or "ACCUMULATE" in macro_reco:
            up_score += 1.5
        elif "SELL" in macro_reco or "AVOID" in macro_reco:
            down_score += 1.5
        add_conf(macro_analysis.get("final_score"))
    if isinstance(resonance_score, (int, float)):
        if resonance_score >= 50:
            up_score += 1.0
        elif resonance_score <= -50:
            down_score += 1.0
        add_conf(abs(resonance_score))
    direction = "NEUTRAL"
    if up_score > down_score and up_score >= 1.0:
        direction = "UP"
    elif down_score > up_score and down_score >= 1.0:
        direction = "DOWN"
    net_score = up_score - down_score
    base_prob = 50.0 + max(-25.0, min(25.0, net_score * 7.5))
    if conf_weight > 0:
        avg_conf = conf_sum / conf_weight
        base_prob = (base_prob + max(0.0, min(100.0, avg_conf))) / 2.0
    probability = max(0.0, min(100.0, base_prob))
    expected_move_pct = None
    expected_holding_hours = None
    short_term_signal = None
    if isinstance(ema_plan, dict):
        if isinstance(ema_plan.get("expected_move_pct"), (int, float)):
            expected_move_pct = float(ema_plan["expected_move_pct"])
        bars = ema_plan.get("expected_holding_bars_15m")
        if isinstance(bars, (int, float)) and bars > 0:
            expected_holding_hours = float(bars) * 0.25
        if isinstance(ema_plan.get("signal"), str):
            short_term_signal = ema_plan["signal"]
    macro_reco = None
    macro_score = None
    if isinstance(macro_analysis, dict):
        macro_reco = macro_analysis.get("recommendation")
        val = macro_analysis.get("final_score")
        if isinstance(val, (int, float)):
            macro_score = float(val)
    return {
        "direction": direction,
        "probability": probability,
        "expected_move_pct": expected_move_pct,
        "expected_holding_hours": expected_holding_hours,
        "short_term_signal": short_term_signal,
        "macro_recommendation": macro_reco,
        "macro_score": macro_score,
        "phase_status": phase_status,
    }


# --- Main Analysis Logic ---

def analyze_single_symbol(symbol, period):
    try:
        data = get_stock_data(symbol, period)
        if data is None:
            return {"symbol": symbol, "error": "No Data"}
        data = calculate_technical_indicators(data)
        latest = data.iloc[-1]
        best_meta = EMA_OPTIMIZER_BEST.get(normalize_symbol(symbol))
        best_len = None
        ema_baseline = None
        if isinstance(best_meta, dict):
            best_len = best_meta.get("best_length")
        try:
            if best_len is not None:
                best_len_int = int(best_len)
                if best_len_int >= 2 and best_len_int <= 400:
                    ema_val = data["Close"].ewm(span=best_len_int, adjust=False).mean().iloc[-1]
                    if pd.notna(ema_val) and pd.notna(latest["Close"]):
                        ema_baseline = {
                            "ema_length": best_len_int,
                            "ema_value": float(ema_val),
                            "phase": "BULL" if float(latest["Close"]) > float(ema_val) else "BEAR",
                            "particle_phase": 0 if float(latest["Close"]) > float(ema_val) else 180,
                            "source_period": best_meta.get("period"),
                            "source_interval": best_meta.get("interval"),
                            "computed_at": best_meta.get("computed_at"),
                        }
        except Exception:
            ema_baseline = None
        resonance_score = ParticleAAnalyzer.calculate_resonance_score(data)
        phase_status = ParticleAAnalyzer.interpret_phase(resonance_score)
        support = data["Low"].min()
        resistance = data["High"].max()
        vol_avg = data["Volume"].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else data["Volume"].mean()
        vol_status = "High" if latest["Volume"] > vol_avg * 1.2 else "Low" if latest["Volume"] < vol_avg * 0.8 else "Normal"
        info = get_basic_info(symbol)
        signal, ai_reason = generate_gemini_particle_a_analysis(info, data, resonance_score, phase_status, support, resistance, vol_status)
        chart_data = {
            "dates": [dt.strftime("%Y-%m-%d %H:%M") if period in ["1h", "15m"] else dt.strftime("%Y-%m-%d") for dt in data.index],
            "close": data["Close"].fillna(0).tolist(),
            "sma20": data["SMA20"].fillna(0).tolist(),
            "bb_upper": data["BB_Upper"].fillna(0).tolist(),
            "bb_lower": data["BB_Lower"].fillna(0).tolist(),
        }
        short_term_plan = ShortTermStrategy.analyze_15m_setup(symbol)
        sniper_plan = ShortTermStrategy.analyze_sniper_setup(symbol)
        quantum_plan = QuantumHunterStrategy.analyze(symbol)
        sovereign_4h_plan = QuantumSovereign4H.analyze(symbol, baseline_ema_length=ema_baseline["ema_length"] if isinstance(ema_baseline, dict) else None)
        crypto_reversal_plan = CryptoReversal15m.analyze(symbol)
        ema_cross_plan = EMACross15m.analyze(symbol)
        actionzone_plan = _actionzone_15m_alert(symbol)
        order_blocks_15m = _order_block_levels_15m(symbol)
        macro_analysis = None
        try:
            ua = UniverseAnalyzer(symbol, is_crypto=is_crypto_symbol(symbol))
            macro_analysis = ua.predict()
        except Exception:
            macro_analysis = None
        prediction = build_prediction_summary(short_term_plan, sniper_plan, quantum_plan, ema_cross_plan, sovereign_4h_plan, macro_analysis, resonance_score, phase_status, crypto_reversal_plan)
        return {
            "symbol": symbol,
            "name": info["name"],
            "price": float(latest["Close"]) if pd.notna(latest["Close"]) else None,
            "change": float(((latest["Close"] - latest["Open"]) / latest["Open"]) * 100) if pd.notna(latest["Close"]) and pd.notna(latest["Open"]) else None,
            "resonance_score": float(resonance_score) if pd.notna(resonance_score) else None,
            "phase_status": phase_status,
            "signal": signal,
            "reason": ai_reason,
            "chart_data": chart_data,
            "short_term_15m": short_term_plan,
            "sniper_15m": sniper_plan,
            "quantum_15m": quantum_plan,
            "sovereign_4h": sovereign_4h_plan,
            "crypto_reversal_15m": crypto_reversal_plan,
            "ema_cross_15m": ema_cross_plan,
            "actionzone_15m": actionzone_plan,
            "order_blocks_15m": order_blocks_15m,
            "ema_baseline": ema_baseline,
            "macro_analysis": macro_analysis,
            "support": float(support) if pd.notna(support) else None,
            "resistance": float(resistance) if pd.notna(resistance) else None,
            "volume_status": vol_status,
            "rsi": float(latest["RSI"]) if pd.notna(latest["RSI"]) else None,
            "macd": float(latest["MACD"]) if pd.notna(latest["MACD"]) else None,
            "prediction": prediction,
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify(
        {
            "status": "ok",
            "server_time": get_thai_now().strftime("%Y-%m-%d %H:%M:%S (Asia/Bangkok)"),
            "warnings": _get_config_warnings(),
        }
    )

@app.route('/ema_cross_15m_stats', methods=['POST'])
def ema_cross_15m_stats():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request body"}), 400
    symbol = normalize_symbol(data.get("symbol") or data.get("ticker") or "")
    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400
    yf_period = str(data.get("yf_period") or getattr(config, "EMA_CROSS_15M_YF_PERIOD", "30d"))
    folds = data.get("folds", getattr(config, "EMA_CROSS_15M_STATS_FOLDS", 3))
    tp_mult = data.get("tp_mult", getattr(config, "EMA_CROSS_15M_TP_MULT", 5.0))
    max_forward = data.get("max_forward_bars", getattr(config, "EMA_CROSS_15M_MAX_FORWARD_BARS", 64))
    min_train = data.get("min_train_bars", getattr(config, "EMA_CROSS_15M_STATS_MIN_TRAIN_BARS", 240))

    cache_key = ("ema_cross_15m_stats", symbol, yf_period, int(folds) if str(folds).isdigit() else str(folds), str(tp_mult), str(max_forward), str(min_train))
    cached = _STATS_CACHE.get(cache_key)
    if isinstance(cached, dict) and cached:
        return jsonify(cached)

    raw_df = get_yf_history(symbol, period=yf_period, interval="15m", auto_adjust=True)
    if raw_df is None or getattr(raw_df, "empty", True):
        return jsonify({"error": "No data"}), 400

    payload = _walk_forward_ema_cross_15m(symbol, raw_df, folds=folds, tp_mult=tp_mult, max_forward=max_forward, min_train_bars=min_train)
    payload["warnings"] = _get_config_warnings()
    payload["yf_period"] = yf_period
    _STATS_CACHE.set(cache_key, payload)
    if payload.get("error"):
        return jsonify(payload), 400
    return jsonify(payload)

@app.route('/global_sovereign', methods=['GET'])
def global_sovereign():
    try:
        data = get_global_sovereign_overview()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ema_optimize', methods=['POST'])
def ema_optimize():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid request body'}), 400
    symbol = normalize_symbol(data.get('symbol') or data.get('ticker') or "")
    if not symbol:
        return jsonify({'error': 'No symbol provided'}), 400
    period = str(data.get('period') or '2y')
    interval = str(data.get('interval') or '1d')
    min_len = data.get('min_len', 10)
    max_len = data.get('max_len', 200)
    step = data.get('step', 5)
    fee_bps = data.get('fee_bps', 0.0)
    top_n = data.get('top_n', 5)
    res = optimize_best_ema(
        symbol,
        period=period,
        interval=interval,
        min_len=min_len,
        max_len=max_len,
        step=step,
        fee_bps=fee_bps,
        top_n=top_n
    )
    if res.get('error'):
        return jsonify(res), 400
    return jsonify(res)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid request body'}), 400
    raw_symbols = str(data.get('symbols', '')).upper().replace('\n', ',').replace(';', ',')
    symbols_raw = [s.strip() for s in raw_symbols.split(',') if s.strip()]
    symbols = []
    seen = set()
    for s in symbols_raw:
        if s in seen:
            continue
        seen.add(s)
        symbols.append(s)
    period = data.get('period', '1mo')
    if period not in VALID_PERIODS:
        return jsonify({'error': 'Invalid period'}), 400
    if not symbols:
        return jsonify({'error': 'No symbols provided'}), 400
    if len(symbols) > 30:
        return jsonify({'error': 'Too many symbols (max 30)'}), 400

    results = list(_ANALYZE_EXECUTOR.map(lambda s: analyze_single_symbol(s, period), symbols))
    notify = bool(data.get("notify_telegram"))
    if notify:
        _notify_telegram_from_results(results)

    def clean_value(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, (list, tuple)):
            return [clean_value(x) for x in v]
        if isinstance(v, dict):
            return {k: clean_value(x) for k, x in v.items()}
        return v

    cleaned = [clean_value(r) for r in results]
    return jsonify({'results': cleaned})

def _run_once(symbols, period, notify_telegram):
    raw_symbols = str(symbols or "").upper().replace("\n", ",").replace(";", ",")
    symbols_raw = [s.strip() for s in raw_symbols.split(",") if s.strip()]
    uniq = []
    seen = set()
    for s in symbols_raw:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    if not uniq:
        return 2
    if period not in VALID_PERIODS:
        return 2
    results = [analyze_single_symbol(s, period) for s in uniq]
    if notify_telegram:
        _notify_telegram_from_results(results)
    print(json.dumps({"results": results}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="")
    parser.add_argument("--period", default="1mo")
    parser.add_argument("--notify-telegram", action="store_true")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    if str(args.symbols or "").strip():
        raise SystemExit(_run_once(args.symbols, str(args.period or "1mo"), bool(args.notify_telegram)))
    port = int(args.port) if args.port is not None else int(getattr(config, "PORT", 5000))
    app.run(debug=getattr(config, "FLASK_DEBUG", True), port=port)
