from flask import Flask, render_template, request, jsonify
import argparse
import csv
import json
import hashlib
import logging
import os
import re
import pandas as pd
import yfinance as yf
import numpy as np
import certifi
import math
import time
import threading
import tempfile
import pytz
from collections import Counter, OrderedDict
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from curl_cffi import requests as curl_requests
import config
from alerts.daily import (
    build_cdc_daily_trend_candidates as _alerts_build_cdc_daily_trend_candidates,
    build_daily_best_pick_candidates as _alerts_build_daily_best_pick_candidates,
    is_daily_best_pick_window as _alerts_is_daily_best_pick_window,
)
from alerts.messages import (
    build_all_weather_message as _alerts_build_all_weather_message,
    build_daily_best_pick_message as _alerts_build_daily_best_pick_message,
    build_price_action_message as _alerts_build_price_action_message,
    build_super_signal_message as _alerts_build_super_signal_message,
    build_telegram_message as _alerts_build_telegram_message,
)
from alerts.pipeline import (
    build_telegram_candidates as _alerts_pipeline_build_telegram_candidates,
    notify_telegram_from_results as _alerts_pipeline_notify_telegram_from_results,
)
from alerts.reporting import (
    alert_history_csv_fieldnames as _alerts_reporting_alert_history_csv_fieldnames,
    alert_history_trim_locked as _alerts_reporting_alert_history_trim_locked,
    build_telegram_alert_live_preview as _alerts_reporting_build_telegram_alert_live_preview,
    build_telegram_alert_report as _alerts_reporting_build_telegram_alert_report,
    candidate_backtest_snapshot as _alerts_reporting_candidate_backtest_snapshot,
    candidate_message_preview as _alerts_reporting_candidate_message_preview,
    candidate_ops_snapshot as _alerts_reporting_candidate_ops_snapshot,
    read_latest_telegram_run_report as _alerts_reporting_read_latest_telegram_run_report,
    read_telegram_alert_history as _alerts_reporting_read_telegram_alert_history,
    record_telegram_alert_history as _alerts_reporting_record_telegram_alert_history,
    record_telegram_run_report as _alerts_reporting_record_telegram_run_report,
    sync_alert_history_csv_locked as _alerts_reporting_sync_alert_history_csv_locked,
    write_verify_output as _alerts_reporting_write_verify_output,
)
from data_layer.yahoo import (
    build_source_health_snapshot as _data_build_source_health_snapshot,
    chart_interval as _data_chart_interval,
    clear_yf_runtime_cache as _data_clear_yf_runtime_cache,
    configure_yf_tz_cache as _data_configure_yf_tz_cache,
    create_curl_session as _data_create_curl_session,
    fetch_yahoo_chart_history as _data_fetch_yahoo_chart_history,
    get_http_verify_setting as _data_get_http_verify_setting,
    get_thread_curl_session as _data_get_thread_curl_session,
    get_yf_history as _data_get_yf_history,
    is_yf_auth_error as _data_is_yf_auth_error,
    normalize_df_index as _data_normalize_df_index,
    normalize_price_columns as _data_normalize_price_columns,
    period_to_timedelta as _data_period_to_timedelta,
    prefer_chart_api as _data_prefer_chart_api,
    project_yf_cache_dir as _data_project_yf_cache_dir,
    record_source_health_event as _data_record_source_health_event,
    remote_history_period as _data_remote_history_period,
    set_thread_curl_session as _data_set_thread_curl_session,
    slice_history_by_period as _data_slice_history_by_period,
)
from strategies.price_action import build_price_action_plan as _strategies_build_price_action_plan
try:
    import requests as http_requests
except Exception:
    http_requests = None

app = Flask(__name__)
if getattr(config, "SECRET_KEY", ""):
    app.config["SECRET_KEY"] = config.SECRET_KEY

logger = logging.getLogger(__name__)
_YF_TZ_CACHE_LOCK = threading.Lock()
_YF_TZ_CACHE_READY = False


def _project_yf_cache_dir():
    return _data_project_yf_cache_dir(__file__)


def _configure_yf_tz_cache(force_temp=False):
    return _data_configure_yf_tz_cache(__file__, force_temp=force_temp)


def _clear_yf_runtime_cache(clear_tz_cache=False):
    return _data_clear_yf_runtime_cache(__file__, clear_tz_cache=clear_tz_cache)


def _is_yf_auth_error(message):
    return _data_is_yf_auth_error(message)

# Helper to get current Thai time (naive)
def get_thai_now():
    return datetime.now(pytz.timezone('Asia/Bangkok')).replace(tzinfo=None)


VALID_PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', 'ytd', '1h', '15m']
EMA_CROSS_15M_OPT_CACHE = {}
_YF_EMPTY_SENTINEL = object()

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
_ANALYZE_EXECUTOR = ThreadPoolExecutor(max_workers=int(getattr(config, "ANALYZE_MAX_WORKERS", 5)))
_TELEGRAM_ALERT_CACHE = _TTLCache(
    maxsize=256,
    ttl_seconds=int(getattr(config, "TELEGRAM_ALERT_TTL_SECONDS", 1800)),
)

_HISTORY_STORE_LOCK = threading.Lock()
_ALERT_HISTORY_LOCK = threading.Lock()
_TELEGRAM_REPORT_STRATEGY_ORDER = ("SS15", "AW15", "CDCVIX15", "AZ15", "PA15", "PRIMARY", "DAILY_BEST")

try:
    _MAX_SYMBOLS_PER_REQUEST = int(getattr(config, "MAX_SYMBOLS_PER_REQUEST", 30))
except Exception:
    _MAX_SYMBOLS_PER_REQUEST = 30
if _MAX_SYMBOLS_PER_REQUEST < 1:
    _MAX_SYMBOLS_PER_REQUEST = 1


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


def _parse_symbols_input(raw_symbols, max_symbols=None):
    max_count = _MAX_SYMBOLS_PER_REQUEST if max_symbols is None else int(max_symbols)
    if max_count < 1:
        max_count = 1
    symbols_raw = str(raw_symbols or "").upper().replace("\n", ",").replace(";", ",").split(",")
    unique_symbols = []
    seen = set()
    for raw in symbols_raw:
        sym = normalize_symbol(raw)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        unique_symbols.append(sym)
        if len(unique_symbols) >= max_count:
            break
    return unique_symbols


def _parse_periods_input(raw_periods, default_periods=None, max_periods=6):
    defaults = list(default_periods or ["1mo", "3mo", "6mo"])
    if isinstance(raw_periods, (list, tuple)):
        raw_tokens = [str(v or "").strip() for v in raw_periods]
    else:
        text = str(raw_periods or "").strip()
        raw_tokens = text.replace("\n", ",").replace(";", ",").split(",") if text else defaults
    periods = []
    seen = set()
    for raw in raw_tokens:
        period = str(raw or "").strip()
        if not period or period in seen or period not in VALID_PERIODS:
            continue
        seen.add(period)
        periods.append(period)
        if len(periods) >= int(max_periods):
            break
    return periods if periods else [p for p in defaults if p in VALID_PERIODS][: int(max_periods)]


def _parse_strategy_input(raw_strategies):
    if isinstance(raw_strategies, (list, tuple)):
        raw_tokens = [str(v or "").strip() for v in raw_strategies]
    else:
        text = str(raw_strategies or "").strip()
        raw_tokens = text.replace("\n", ",").replace(";", ",").split(",") if text else []
    values = []
    seen = set()
    for raw in raw_tokens:
        strategy = str(raw or "").strip().upper()
        if not strategy or strategy in seen:
            continue
        seen.add(strategy)
        values.append(strategy)
    return values


def _clean_json_value(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (list, tuple)):
        return [_clean_json_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean_json_value(x) for k, x in v.items()}
    return v


def send_telegram_alert(message):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    thread_id = os.environ.get("TELEGRAM_THREAD_ID")
    if not bot_token or not chat_id or not message:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id, 
        "text": message, 
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    if thread_id:
        payload["message_thread_id"] = thread_id
        
    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if http_requests is not None:
                resp = http_requests.post(url, json=payload, timeout=10)
                if bool(getattr(resp, "ok", False)):
                    return True
                status_code = int(getattr(resp, "status_code", 0))
            else:
                session = _create_curl_session()
                resp = session.post(url, json=payload, timeout=10)
                status_code = int(getattr(resp, "status_code", 0))
                ok = getattr(resp, "ok", None)
                if ok is not None and bool(ok):
                    return True
                if 200 <= status_code < 300:
                    return True
                    
            # Handle rate limits (429)
            if status_code == 429:
                import time
                retry_after = int(resp.json().get("parameters", {}).get("retry_after", 3))
                time.sleep(retry_after)
                continue
                
            logger.warning("Telegram alert attempt %d failed (status: %d): %s", attempt + 1, status_code, resp.text if hasattr(resp, 'text') else '')
            
        except Exception as e:
            logger.warning("Telegram alert attempt %d failed with exception: %s", attempt + 1, e)
            
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


def _html_escape(text):
    if not isinstance(text, str):
        text = str(text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _generate_exit_levels(entry_price, stop_loss, signal="BUY", take_profit=None):
    try:
        entry = float(entry_price)
        stop = float(stop_loss)
    except Exception:
        return None, []
    if not math.isfinite(entry) or not math.isfinite(stop) or entry == 0:
        return None, []
    risk_dist = abs(entry - stop)
    if risk_dist <= 0:
        return None, []
    risk_pct = (risk_dist / abs(entry)) * 100.0
    tp_r = None
    try:
        if isinstance(take_profit, (int, float)):
            tp_val = float(take_profit)
            if math.isfinite(tp_val):
                tp_move = abs(tp_val - entry)
                if tp_move > 0:
                    tp_r = tp_move / risk_dist
    except Exception:
        tp_r = None
    if isinstance(tp_r, (int, float)) and tp_r >= 1.0:
        r3 = max(3.0, float(tp_r))
        r2 = max(1.9, min(2.6, r3 * 0.72))
        r1 = max(1.2, min(1.6, r2 - 0.7))
        r_levels = [r1, r2, r3]
    else:
        if risk_pct < 0.8:
            r_levels = [1.4, 2.4, 3.8]
        elif risk_pct < 1.5:
            r_levels = [1.2, 2.1, 3.2]
        else:
            r_levels = [1.0, 1.8, 2.8]
    normalized = []
    for r in r_levels:
        try:
            val = float(r)
        except Exception:
            continue
        if val <= 0:
            continue
        normalized.append(val)
    normalized = sorted(normalized)
    if not normalized:
        return risk_pct, []
    while len(normalized) < 3:
        last_val = normalized[-1]
        next_val = last_val + (0.8 if last_val < 2.0 else 1.0)
        normalized.append(next_val)
    direction = -1.0 if str(signal or "").upper() == "SELL" else 1.0
    close_weights = [35, 35, 30]
    risk_after = [risk_pct * 0.55, risk_pct * 0.25, 0.0]
    levels = []
    prev_move_pct = 0.0
    for idx, r_mult in enumerate(normalized[:3]):
        target_price = entry + (direction * risk_dist * r_mult)
        move_pct = (abs(target_price - entry) / abs(entry)) * 100.0
        spacing_pct = max(0.0, move_pct - prev_move_pct)
        prev_move_pct = move_pct
        close_ratio = close_weights[idx] if idx < len(close_weights) else 0
        remain_risk = risk_after[idx] if idx < len(risk_after) else 0.0
        levels.append(
            {
                "label": f"TP{idx + 1}",
                "target_price": float(target_price),
                "reward_r": float(r_mult),
                "profit_pct": float(move_pct),
                "spacing_pct": float(spacing_pct),
                "close_ratio_pct": float(close_ratio),
                "risk_remaining_pct": float(max(0.0, remain_risk)),
            }
        )
    return float(risk_pct), levels


def _pick_plan_value(plan, keys):
    if not isinstance(plan, dict):
        return None
    for key in keys:
        value = plan.get(key)
        if value is None:
            continue
        return value
    return None


def _infer_plan_signal(plan, default_signal="BUY"):
    if not isinstance(plan, dict):
        return default_signal
    candidates = [
        plan.get("signal"),
        plan.get("raw_signal"),
        plan.get("setup"),
        plan.get("recommendation"),
    ]
    text = " ".join([str(c) for c in candidates if c is not None]).upper()
    if "EXIT" in text and "BUY" not in text and "SELL" not in text and "LONG" not in text and "SHORT" not in text:
        return default_signal
    if "SELL" in text or "SHORT" in text:
        return "SELL"
    if "BUY" in text or "LONG" in text:
        return "BUY"
    return default_signal


def _is_entry_signal(plan, signal_hint=None):
    sig = str(signal_hint or _infer_plan_signal(plan, "")).upper()
    if sig in ("BUY", "SELL"):
        return True
    text = " ".join(
        [
            str(plan.get("signal") or ""),
            str(plan.get("raw_signal") or ""),
            str(plan.get("setup") or ""),
            str(plan.get("recommendation") or ""),
        ]
    ).upper() if isinstance(plan, dict) else ""
    if "WAIT" in text or "EXIT" in text:
        return False
    return ("BUY" in text) or ("SELL" in text) or ("LONG" in text) or ("SHORT" in text)


def _compute_entry_risk_pct_with_context(entry, stop, signal, plan=None, context=None):
    if entry is None or stop is None or entry == 0:
        return None
    raw_risk_pct = (abs(entry - stop) / abs(entry)) * 100.0
    if not math.isfinite(raw_risk_pct) or raw_risk_pct <= 0:
        return None
    factor = 1.0
    support = None
    resistance = None
    volume_status = None
    if isinstance(context, dict):
        support = context.get("support")
        resistance = context.get("resistance")
        volume_status = str(context.get("volume_status") or "").upper()
    direction = str(signal or "BUY").upper()
    if direction == "BUY":
        if isinstance(support, (int, float)) and support > 0 and support < entry:
            support_gap = ((entry - float(support)) / entry) * 100.0
            if support_gap < raw_risk_pct * 0.7:
                factor += 0.12
            elif support_gap > raw_risk_pct * 1.8:
                factor -= 0.05
        if isinstance(resistance, (int, float)) and resistance > entry:
            upside_room = ((float(resistance) - entry) / entry) * 100.0
            if upside_room < raw_risk_pct * 1.2:
                factor += 0.25
            elif upside_room > raw_risk_pct * 2.8:
                factor -= 0.08
    elif direction == "SELL":
        if isinstance(resistance, (int, float)) and resistance > entry:
            resistance_gap = ((float(resistance) - entry) / entry) * 100.0
            if resistance_gap < raw_risk_pct * 0.7:
                factor += 0.12
            elif resistance_gap > raw_risk_pct * 1.8:
                factor -= 0.05
        if isinstance(support, (int, float)) and support < entry:
            downside_room = ((entry - float(support)) / entry) * 100.0
            if downside_room < raw_risk_pct * 1.2:
                factor += 0.25
            elif downside_room > raw_risk_pct * 2.8:
                factor -= 0.08
    if volume_status == "HIGH":
        factor += 0.12
    elif volume_status == "LOW":
        factor -= 0.05
    conf = None
    if isinstance(plan, dict):
        conf = _normalize_confidence(plan.get("confidence"))
        if conf is None:
            conf = _normalize_confidence(plan.get("predicted_win_prob"))
        if conf is None:
            conf = _normalize_confidence(plan.get("win_prob"))
    if isinstance(conf, (int, float)):
        if conf >= 80:
            factor -= 0.08
        elif conf <= 55:
            factor += 0.1
    if factor < 0.6:
        factor = 0.6
    if factor > 1.6:
        factor = 1.6
    risk_pct = raw_risk_pct * factor
    return float(max(0.2, min(10.0, risk_pct)))


def _attach_exit_levels(plan, signal=None, entry_keys=None, stop_keys=None, tp_keys=None, context=None):
    if not isinstance(plan, dict):
        return plan
    entry_keys = entry_keys or ["entry_price", "current_price", "price"]
    stop_keys = stop_keys or ["stop_loss", "entry_stop_loss"]
    tp_keys = tp_keys or ["take_profit", "take_profit_2", "trailing_stop", "exit_price"]
    sig_hint = str(signal or _infer_plan_signal(plan, "")).upper()
    sig = sig_hint
    if sig not in ("BUY", "SELL"):
        sig = "BUY"
    entry = _pick_plan_value(plan, entry_keys)
    stop = _pick_plan_value(plan, stop_keys)
    tp = _pick_plan_value(plan, tp_keys)
    risk_pct, levels = _generate_exit_levels(entry, stop, signal=sig, take_profit=tp)
    if _is_entry_signal(plan, sig_hint):
        contextual_risk = _compute_entry_risk_pct_with_context(entry, stop, sig, plan=plan, context=context)
        if isinstance(contextual_risk, (int, float)):
            plan["entry_risk_pct"] = float(contextual_risk)
        elif isinstance(risk_pct, (int, float)):
            plan["entry_risk_pct"] = float(risk_pct)
    else:
        plan.pop("entry_risk_pct", None)
    if levels:
        plan["exit_levels"] = levels
    else:
        plan["exit_levels"] = []
    return plan


def _format_exit_levels_lines(plan):
    if not isinstance(plan, dict):
        return []
    levels = plan.get("exit_levels")
    if not isinstance(levels, list) or not levels:
        return []
    lines = ["🎯 แผนออกทำกำไร 3 จุด"]
    risk_pct = plan.get("entry_risk_pct")
    if isinstance(risk_pct, (int, float)):
        lines.append(f"ความเสี่ยงตั้งต้นต่อไม้: {float(risk_pct):.2f}%")
    for level in levels[:3]:
        if not isinstance(level, dict):
            continue
        label = str(level.get("label") or "TP")
        target = _format_price_value(level.get("target_price"))
        move = level.get("profit_pct")
        spacing = level.get("spacing_pct")
        risk_remain = level.get("risk_remaining_pct")
        close_ratio = level.get("close_ratio_pct")
        details = []
        if target:
            details.append(target)
        if isinstance(move, (int, float)):
            details.append(f"กำไร {float(move):.2f}%")
        if isinstance(spacing, (int, float)):
            details.append(f"ระยะห่าง {float(spacing):.2f}%")
        if isinstance(risk_remain, (int, float)):
            details.append(f"ความเสี่ยงคงเหลือ {float(risk_remain):.2f}%")
        if isinstance(close_ratio, (int, float)) and close_ratio > 0:
            details.append(f"แนะนำขาย {float(close_ratio):.0f}%")
        if details:
            lines.append(f"• {label}: " + " | ".join(details))
    return lines


def _get_plan_label(plan, item=None):
    if not isinstance(plan, dict):
        return "Trade Plan"
    strategy = str(plan.get("strategy") or "").strip()
    if strategy:
        return strategy
    if isinstance(item, dict):
        return _get_primary_plan_source_label(item, plan)
    return "Trade Plan"


def _build_stop_context_lines(item, plan, signal=None, source_label=None):
    if not isinstance(plan, dict):
        return []
    direction = _normalize_trade_direction(
        signal
        or plan.get("signal")
        or plan.get("raw_signal")
        or plan.get("setup")
        or plan.get("recommendation")
    )
    if direction not in ("BUY", "SELL"):
        return []
    stop = _pick_plan_value(plan, ["stop_loss", "entry_stop_loss", "trailing_stop"])
    entry = _pick_plan_value(plan, ["entry_price", "current_price", "price"])
    target = _pick_plan_value(plan, ["take_profit", "take_profit_2", "exit_price", "trailing_stop"])
    stop_text = _format_price_value(stop)
    if not stop_text:
        return []
    entry_risk_pct = plan.get("entry_risk_pct")
    if not isinstance(entry_risk_pct, (int, float)):
        try:
            if isinstance(entry, (int, float)) and isinstance(stop, (int, float)) and float(entry) != 0:
                entry_risk_pct = (abs(float(entry) - float(stop)) / abs(float(entry))) * 100.0
        except Exception:
            entry_risk_pct = None
    risk_text = None
    if isinstance(entry_risk_pct, (int, float)) and math.isfinite(float(entry_risk_pct)):
        risk_text = f"{float(entry_risk_pct):.2f}%"
    target_text = _format_price_value(target)
    order_blocks = item.get("order_blocks_15m") if isinstance(item, dict) else None
    nearest_support = order_blocks.get("nearest_support") if isinstance(order_blocks, dict) else None
    nearest_resistance = order_blocks.get("nearest_resistance") if isinstance(order_blocks, dict) else None
    source = str(source_label or _get_plan_label(plan, item)).strip() or "Trade Plan"
    if "ACTIONZONE" in source.upper():
        if direction == "BUY" and isinstance(nearest_support, (int, float)):
            stop_reason = f"วางใต้แนวรับ 15m ใกล้สุดบริเวณ {nearest_support:.5f} และเผื่อ buffer จากความผันผวน"
        elif direction == "SELL" and isinstance(nearest_resistance, (int, float)):
            stop_reason = f"วางเหนือแนวต้าน 15m ใกล้สุดบริเวณ {nearest_resistance:.5f} และเผื่อ buffer จากความผันผวน"
        else:
            stop_reason = "อิงโครงสร้าง EMA/ราคา 15m และ buffer ของความผันผวนล่าสุด"
    elif "EMA CROSS" in source.upper():
        stop_reason = "อิงระยะเสี่ยงจาก ATR และจะถือว่ามุมมองผิดเมื่อโครงสร้าง EMA เสีย"
    elif "CDC" in source.upper() or "VIXFIX" in source.upper():
        stop_reason = "อิงจุดที่สมมติฐานการยืดตัวหรือกลับตัวเสีย ไม่ใช่ตั้ง stop แบบตายตัว"
    elif "CRYPTO REVERSAL" in source.upper():
        stop_reason = "อิง volatility reversal setup และจุดที่รูปแบบกลับตัวถูกลบล้าง"
    else:
        stop_reason = "ใช้เป็นจุดยกเลิกมุมมอง เมื่อราคาทะลุจุดนี้ให้ถือว่า setup เดิมผิด"
    invalidation_price = None
    forecast = item.get("price_forecast") if isinstance(item, dict) else None
    if isinstance(forecast, dict):
        invalidation_price = forecast.get("invalidation_price")
    invalidation_text = _format_price_value(invalidation_price) or stop_text
    lines = [
        f"<b>🛡️ จุดตัดขาดทุน (SL):</b> {stop_text}",
        f"<b>🧠 หลักการวาง SL:</b> {_html_escape(stop_reason)}",
    ]
    if risk_text:
        lines.append(f"<b>📏 ระยะห่างถึง SL:</b> {risk_text} จากจุดเข้า")
    if target_text:
        lines.append(f"<b>🎯 เป้าหลัก:</b> {target_text}")
    if direction == "BUY":
        lines.append(f"<b>❌ Invalidation:</b> มุมมอง BUY จะเสียเมื่อราคาหลุด {invalidation_text}")
    else:
        lines.append(f"<b>❌ Invalidation:</b> มุมมอง SELL จะเสียเมื่อราคาทะลุ {invalidation_text}")
    return lines


def _is_actionable_setup_value(value):
    if value is None:
        return False
    text = str(value).strip().upper()
    if not text:
        return False
    return text not in {"WAIT", "HOLD", "NONE", "NEUTRAL"}


def _strict_60_mode_enabled():
    return bool(getattr(config, "TELEGRAM_ALERT_STRICT_60_MODE", True))


def _strict_60_allow_cdc():
    if not _strict_60_mode_enabled():
        return True
    return not bool(getattr(config, "TELEGRAM_ALERT_STRICT_60_EXCLUDE_CDC", True))


def _plan_trade_direction(plan):
    if not isinstance(plan, dict):
        return None
    return _normalize_trade_direction(
        plan.get("signal") or plan.get("raw_signal") or plan.get("setup") or plan.get("recommendation")
    )


def _plan_confidence_value(plan):
    if not isinstance(plan, dict):
        return None
    conf = _normalize_confidence(plan.get("confidence"))
    if conf is None:
        conf = _normalize_confidence(plan.get("predicted_win_prob"))
    if conf is None:
        conf = _normalize_confidence(plan.get("win_prob"))
    return conf


def _collect_alert_sources(item, min_conf, signal=None, require_quality=False, allow_cdc=True):
    sources = []

    def add(label, plan, setup_key="setup", conf_key="confidence", allow_label=True):
        if not allow_label:
            return
        if not isinstance(plan, dict):
            return
        direction = _plan_trade_direction(plan)
        if signal in ("BUY", "SELL") and direction != signal:
            return
        conf = _normalize_confidence(plan.get(conf_key))
        if conf is None or conf < min_conf:
            return
        setup = plan.get(setup_key)
        if setup_key in ("setup", "signal") and not _is_actionable_setup_value(setup):
            return
        if require_quality and direction in ("BUY", "SELL") and not _passes_entry_quality_gate(plan, direction):
            return
        text = label
        if isinstance(setup, str) and setup.strip():
            text = f"{label} {setup.strip()}"
        sources.append((conf, text))

    add("ShortTerm 15m", item.get("short_term_15m"))
    add("Sniper 15m", item.get("sniper_15m"))
    add("Quantum 15m", item.get("quantum_15m"))
    add("EMA Cross 15m", item.get("ema_cross_15m"), setup_key="signal")
    add("ActionZone 15m", item.get("actionzone_15m"), setup_key="signal")
    add("CDC+VixFix 15m", item.get("cdc_vixfix_15m"), setup_key="signal", allow_label=allow_cdc)
    add("Crypto Reversal 15m", item.get("crypto_reversal_15m"))
    sources.sort(key=lambda x: x[0], reverse=True)
    return [f"{text} ({conf:.0f}%)" for conf, text in sources[:3]]


def _extract_plan_edge_metrics(plan):
    if not isinstance(plan, dict):
        return {}

    def pick_numeric(obj, keys):
        if not isinstance(obj, dict):
            return None
        for key in keys:
            value = obj.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
        return None

    candidate_payloads = [plan]
    optimizer = plan.get("optimizer")
    if isinstance(optimizer, dict):
        best = optimizer.get("best")
        if isinstance(best, dict):
            candidate_payloads.append(best)

    best_metrics = {}
    best_score = -1.0
    for payload in candidate_payloads:
        win_rate = pick_numeric(
            payload,
            ("valid_win_rate_pct", "historical_win_rate", "win_rate_pct", "historical_win_rate_tp1", "historical_win_rate_tp2"),
        )
        trades = pick_numeric(
            payload,
            ("valid_trades", "historical_trades", "trades", "historical_trades_tp1", "historical_trades_tp2"),
        )
        expectancy = pick_numeric(
            payload,
            ("expectancy_rr", "valid_expectancy_rr", "avg_rr", "historical_avg_rr", "expectancy_tp1_rr", "expectancy_tp2_rr"),
        )
        score = 0.0
        if win_rate is not None:
            score += 1.0
        if trades is not None:
            score += 1.0
        if expectancy is not None:
            score += 1.0
        if trades is not None:
            score += min(2.0, float(trades) / 10.0)
        if score > best_score:
            best_score = score
            best_metrics = {
                "win_rate_pct": win_rate,
                "trades": trades,
                "expectancy_rr": expectancy,
            }
    return best_metrics


def _extract_optimizer_trade_metrics(optimizer):
    if not isinstance(optimizer, dict):
        return {}

    def pick_numeric(obj, keys):
        if not isinstance(obj, dict):
            return None
        for key in keys:
            value = obj.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
        return None

    best = optimizer.get("best")
    if not isinstance(best, dict):
        return {}
    return {
        "win_rate_pct": pick_numeric(
            best,
            ("valid_win_rate_pct", "historical_win_rate", "win_rate_pct", "historical_win_rate_tp1", "historical_win_rate_tp2"),
        ),
        "trades": pick_numeric(
            best,
            ("valid_trades", "historical_trades", "trades", "historical_trades_tp1", "historical_trades_tp2"),
        ),
        "expectancy_rr": pick_numeric(
            best,
            ("expectancy_rr", "valid_expectancy_rr", "avg_rr", "historical_avg_rr", "expectancy_tp1_rr", "expectancy_tp2_rr"),
        ),
    }


def _extract_signal_edge_metrics(plan, signal):
    sig = str(signal or "").upper()
    if sig == "SELL" and isinstance(plan, dict):
        sell_metrics = _extract_optimizer_trade_metrics(plan.get("sell_optimizer"))
        if any(isinstance(v, (int, float)) for v in sell_metrics.values()):
            return sell_metrics
    return _extract_plan_edge_metrics(plan)


def _normalize_edge_metrics_payload(metrics):
    edge = metrics if isinstance(metrics, dict) else {}
    win_rate = edge.get("win_rate_pct")
    expectancy = edge.get("expectancy_rr")
    trades = edge.get("trades")
    if not isinstance(win_rate, (int, float)):
        avg_wr = edge.get("avg_wr")
        if isinstance(avg_wr, (int, float)):
            win_rate = float(avg_wr)
    if not isinstance(expectancy, (int, float)):
        avg_exp = edge.get("avg_exp")
        if isinstance(avg_exp, (int, float)):
            expectancy = float(avg_exp)
    if not isinstance(trades, (int, float)):
        avg_trades = edge.get("avg_trades")
        if isinstance(avg_trades, (int, float)):
            trades = float(avg_trades)
    return {
        "win_rate_pct": float(win_rate) if isinstance(win_rate, (int, float)) and math.isfinite(float(win_rate)) else None,
        "expectancy_rr": float(expectancy) if isinstance(expectancy, (int, float)) and math.isfinite(float(expectancy)) else None,
        "trades": float(trades) if isinstance(trades, (int, float)) and math.isfinite(float(trades)) else None,
    }


def _candidate_edge_metrics(candidate):
    if not isinstance(candidate, dict):
        return {"win_rate_pct": None, "expectancy_rr": None, "trades": None}
    edge = candidate.get("edge_metrics")
    normalized = _normalize_edge_metrics_payload(edge)
    if all(isinstance(v, (int, float)) for v in normalized.values()):
        return normalized
    plan = candidate.get("plan")
    signal = str(candidate.get("signal") or "").upper()
    if isinstance(plan, dict):
        plan_metrics = _extract_signal_edge_metrics(plan, signal)
        merged = dict(plan_metrics) if isinstance(plan_metrics, dict) else {}
        if isinstance(edge, dict):
            merged.update(edge)
        normalized = _normalize_edge_metrics_payload(merged)
    if not any(isinstance(v, (int, float)) for v in normalized.values()):
        fallback = _candidate_summary_fallback_metrics(candidate)
        merged = dict(fallback) if isinstance(fallback, dict) else {}
        if isinstance(edge, dict):
            merged.update(edge)
        normalized = _normalize_edge_metrics_payload(merged)
    return normalized


def _summary_pick_numeric(payload, keys):
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _summary_strategy_edge_metrics(plan, signal=None):
    metrics = _normalize_edge_metrics_payload(_extract_signal_edge_metrics(plan, signal))
    if not isinstance(plan, dict):
        return metrics
    win_rate = metrics.get("win_rate_pct")
    trades = metrics.get("trades")
    expectancy = metrics.get("expectancy_rr")
    if not isinstance(win_rate, (int, float)):
        win_rate = _summary_pick_numeric(plan, ("predicted_win_prob", "confidence"))
    if not isinstance(trades, (int, float)):
        trades = _summary_pick_numeric(plan, ("historical_trades", "valid_trades", "trades"))
    if not isinstance(expectancy, (int, float)):
        expectancy = _summary_pick_numeric(plan, ("historical_avg_rr", "avg_rr", "expectancy_rr"))
    return _normalize_edge_metrics_payload(
        {
            "win_rate_pct": win_rate,
            "expectancy_rr": expectancy,
            "trades": trades,
        }
    )


def _summary_edge_observation(metrics, confidence=None, symbol=None, signal=None):
    normalized = _normalize_edge_metrics_payload(metrics)
    return {
        "symbol": normalize_symbol(symbol or ""),
        "signal": str(signal or "").upper().strip() or None,
        "win_rate_pct": normalized.get("win_rate_pct"),
        "expectancy_rr": normalized.get("expectancy_rr"),
        "trades": normalized.get("trades"),
        "confidence": float(confidence) if isinstance(confidence, (int, float)) and math.isfinite(float(confidence)) else None,
    }


def _aggregate_summary_observations(observations):
    total_wr_weight = 0.0
    total_wr_value = 0.0
    total_exp_weight = 0.0
    total_exp_value = 0.0
    total_trades = 0.0
    measured = 0
    symbols = set()
    signals = Counter()
    for row in observations or []:
        if not isinstance(row, dict):
            continue
        symbol = normalize_symbol(row.get("symbol") or "")
        signal = str(row.get("signal") or "").upper().strip()
        if symbol:
            symbols.add(symbol)
        if signal:
            signals[signal] += 1
        wr = row.get("win_rate_pct")
        exp = row.get("expectancy_rr")
        trades = row.get("trades")
        conf = row.get("confidence")
        if isinstance(trades, (int, float)) and math.isfinite(float(trades)) and float(trades) > 0:
            weight = float(trades)
            total_trades += weight
        elif isinstance(conf, (int, float)) and math.isfinite(float(conf)) and float(conf) > 0:
            weight = max(1.0, float(conf) / 20.0)
        else:
            weight = 1.0
        has_metric = False
        if isinstance(wr, (int, float)) and math.isfinite(float(wr)):
            total_wr_value += float(wr) * weight
            total_wr_weight += weight
            has_metric = True
        if isinstance(exp, (int, float)) and math.isfinite(float(exp)):
            total_exp_value += float(exp) * weight
            total_exp_weight += weight
            has_metric = True
        if has_metric:
            measured += 1
    return {
        "measured": int(measured),
        "estimated_win_rate_pct": (total_wr_value / total_wr_weight) if total_wr_weight > 0 else None,
        "average_expectancy_rr": (total_exp_value / total_exp_weight) if total_exp_weight > 0 else None,
        "total_historical_trades": int(total_trades) if total_trades > 0 else 0,
        "observed_symbols": len(symbols),
        "signals": dict(signals),
    }


def _price_action_proxy_metrics(item, signal):
    signal = str(signal or "").upper().strip()
    if signal not in ("BUY", "SELL") or not isinstance(item, dict):
        return {"win_rate_pct": None, "expectancy_rr": None, "trades": None, "source_count": 0, "source_labels": []}
    rows = []
    labels = []
    plan_rows = [
        ("ActionZone 15m", item.get("actionzone_15m")),
        ("EMA Cross 15m", item.get("ema_cross_15m")),
        ("CDC+VixFix 15m", item.get("cdc_vixfix_15m")),
        ("Crypto Reversal 15m", item.get("crypto_reversal_15m")),
        ("ShortTerm 15m", item.get("short_term_15m")),
        ("Sniper 15m", item.get("sniper_15m")),
        ("Quantum 15m", item.get("quantum_15m")),
    ]
    for label, plan in plan_rows:
        if not isinstance(plan, dict):
            continue
        if _plan_trade_direction(plan) != signal:
            continue
        obs = _summary_edge_observation(
            _summary_strategy_edge_metrics(plan, signal=signal),
            confidence=_plan_confidence_value(plan),
            symbol=item.get("symbol"),
            signal=signal,
        )
        if any(isinstance(obs.get(key), (int, float)) for key in ("win_rate_pct", "expectancy_rr", "trades")):
            rows.append(obs)
            labels.append(label)
    agg = _aggregate_summary_observations(rows)
    return {
        "win_rate_pct": agg.get("estimated_win_rate_pct"),
        "expectancy_rr": agg.get("average_expectancy_rr"),
        "trades": agg.get("total_historical_trades"),
        "source_count": len(rows),
        "source_labels": labels[:4],
    }


def _signal_metric_plan_candidates(item, signal, require_quality=False, allow_cdc=True):
    rows = []
    plan_rows = [
        ("SHORTTERM", item.get("short_term_15m"), True),
        ("SNIPER", item.get("sniper_15m"), True),
        ("QUANTUM", item.get("quantum_15m"), True),
        ("EMA", item.get("ema_cross_15m"), True),
        ("AZ", item.get("actionzone_15m"), True),
        ("CDC", item.get("cdc_vixfix_15m"), allow_cdc),
        ("REVERSAL", item.get("crypto_reversal_15m"), True),
    ]
    for _, plan, enabled in plan_rows:
        if not enabled or not isinstance(plan, dict):
            continue
        direction = _plan_trade_direction(plan)
        if signal in ("BUY", "SELL") and direction != signal:
            continue
        if require_quality and direction in ("BUY", "SELL") and not _passes_entry_quality_gate(plan, direction):
            continue
        conf = _plan_confidence_value(plan)
        rows.append(
            _summary_edge_observation(
                _summary_strategy_edge_metrics(plan, signal=signal),
                confidence=conf,
                symbol=item.get("symbol"),
                signal=direction,
            )
        )
    return rows


def _best_summary_signal(item):
    top_signal = str((item or {}).get("signal") or "").upper().strip()
    if top_signal in ("BUY", "SELL"):
        return top_signal
    buy_conf = _get_best_confidence(item, signal="BUY", require_quality=False, allow_cdc=True)
    sell_conf = _get_best_confidence(item, signal="SELL", require_quality=False, allow_cdc=True)
    buy_conf = float(buy_conf) if isinstance(buy_conf, (int, float)) else -1.0
    sell_conf = float(sell_conf) if isinstance(sell_conf, (int, float)) else -1.0
    return "BUY" if buy_conf >= sell_conf else "SELL"


def _build_strategy_summary_observations(item, min_conf=None):
    if not isinstance(item, dict) or item.get("error"):
        return {}
    symbol = normalize_symbol(item.get("symbol") or "")
    if not symbol:
        return {}
    signal_hint = _best_summary_signal(item)
    allow_cdc = True
    summary = {}

    # SS15: aggregate confluence plans for the stronger side even if strict super-signal gate does not pass.
    ss_rows = _signal_metric_plan_candidates(item, signal_hint, require_quality=False, allow_cdc=allow_cdc)
    if not ss_rows:
        alternate = "SELL" if signal_hint == "BUY" else "BUY"
        ss_rows = _signal_metric_plan_candidates(item, alternate, require_quality=False, allow_cdc=allow_cdc)
    if ss_rows:
        best_obs = _aggregate_summary_observations(ss_rows)
        summary["SS15"] = [
            _summary_edge_observation(
                {
                    "win_rate_pct": best_obs.get("estimated_win_rate_pct"),
                    "expectancy_rr": best_obs.get("average_expectancy_rr"),
                    "trades": best_obs.get("total_historical_trades") or len(ss_rows),
                },
                confidence=_get_best_confidence(item, signal=signal_hint, require_quality=False, allow_cdc=allow_cdc),
                symbol=symbol,
                signal=signal_hint,
            )
        ]

    # AW15: use blended metrics; if no passed payload, blend whatever candidate rows are available.
    aw_report = _build_all_weather_report_entry(item, min_conf=min_conf)
    if isinstance(aw_report, dict):
        aw_blend = aw_report.get("blend") or {}
        if not any(isinstance(aw_blend.get(k), (int, float)) for k in ("win_rate_pct", "expectancy_rr", "trades")):
            aw_candidates = aw_report.get("candidates") or []
            if aw_candidates:
                aw_blend = _all_weather_blended_metrics(aw_candidates[: max(1, min(3, len(aw_candidates)))])
        summary["AW15"] = [
            _summary_edge_observation(
                aw_blend,
                confidence=aw_report.get("final_confidence"),
                symbol=symbol,
                signal=aw_report.get("selected_side") or aw_report.get("signal"),
            )
        ]

    cdc_plan = item.get("cdc_vixfix_15m")
    if isinstance(cdc_plan, dict):
        cdc_signal = _plan_trade_direction(cdc_plan)
        summary["CDCVIX15"] = [
            _summary_edge_observation(
                _summary_strategy_edge_metrics(cdc_plan, signal=cdc_signal),
                confidence=_plan_confidence_value(cdc_plan),
                symbol=symbol,
                signal=cdc_signal,
            )
        ]

    az_plan = item.get("actionzone_15m")
    if isinstance(az_plan, dict):
        az_signal = _plan_trade_direction(az_plan)
        summary["AZ15"] = [
            _summary_edge_observation(
                _summary_strategy_edge_metrics(az_plan, signal=az_signal),
                confidence=_plan_confidence_value(az_plan),
                symbol=symbol,
                signal=az_signal,
            )
        ]

    pa_plan = item.get("price_action_15m")
    if isinstance(pa_plan, dict):
        pa_signal = _plan_trade_direction(pa_plan)
        summary["PA15"] = [
            _summary_edge_observation(
                _summary_strategy_edge_metrics(pa_plan, signal=pa_signal),
                confidence=_plan_confidence_value(pa_plan),
                symbol=symbol,
                signal=pa_signal,
            )
        ]

    primary_plan = _pick_primary_trade_plan(item, signal=signal_hint, require_quality=False, allow_cdc=allow_cdc)
    primary_obs = None
    if isinstance(primary_plan, dict):
        primary_obs = _summary_edge_observation(
            _summary_strategy_edge_metrics(primary_plan, signal=signal_hint),
            confidence=_get_best_confidence(item, signal=signal_hint, require_quality=False, allow_cdc=allow_cdc),
            symbol=symbol,
            signal=signal_hint,
        )
        if not any(isinstance(primary_obs.get(k), (int, float)) for k in ("win_rate_pct", "expectancy_rr", "trades")):
            source_rows = _signal_metric_plan_candidates(item, signal_hint, require_quality=False, allow_cdc=allow_cdc)
            agg = _aggregate_summary_observations(source_rows)
            primary_obs = _summary_edge_observation(
                {
                    "win_rate_pct": agg.get("estimated_win_rate_pct"),
                    "expectancy_rr": agg.get("average_expectancy_rr"),
                    "trades": agg.get("total_historical_trades") or len(source_rows),
                },
                confidence=_get_best_confidence(item, signal=signal_hint, require_quality=False, allow_cdc=allow_cdc),
                symbol=symbol,
                signal=signal_hint,
            )
    if isinstance(primary_obs, dict):
        summary["PRIMARY"] = [primary_obs]
        summary["DAILY_BEST"] = [dict(primary_obs)]

    return summary


def _candidate_summary_fallback_metrics(candidate):
    if not isinstance(candidate, dict):
        return {"win_rate_pct": None, "expectancy_rr": None, "trades": None}
    item = candidate.get("item")
    strategy = str(candidate.get("strategy") or "").strip().upper()
    signal = str(candidate.get("signal") or "").strip().upper()
    symbol = normalize_symbol(candidate.get("symbol") or "")
    rows = _build_strategy_summary_observations(item)
    observations = rows.get(strategy) if isinstance(rows, dict) else None
    if not isinstance(observations, list):
        return {"win_rate_pct": None, "expectancy_rr": None, "trades": None}
    matched = []
    for row in observations:
        if not isinstance(row, dict):
            continue
        row_symbol = normalize_symbol(row.get("symbol") or "")
        row_signal = str(row.get("signal") or "").strip().upper()
        if symbol and row_symbol and row_symbol != symbol:
            continue
        if signal in ("BUY", "SELL") and row_signal in ("BUY", "SELL") and row_signal != signal:
            continue
        matched.append(row)
    agg = _aggregate_summary_observations(matched or observations)
    return {
        "win_rate_pct": agg.get("estimated_win_rate_pct"),
        "expectancy_rr": agg.get("average_expectancy_rr"),
        "trades": agg.get("total_historical_trades"),
    }


def _build_strategy_backtest_summary(results, min_conf=None):
    summary_map = {name: [] for name in _TELEGRAM_REPORT_STRATEGY_ORDER}
    for item in results or []:
        rows = _build_strategy_summary_observations(item, min_conf=min_conf)
        for strategy, observations in (rows or {}).items():
            if strategy not in summary_map:
                summary_map[strategy] = []
            for row in observations or []:
                if isinstance(row, dict):
                    summary_map[strategy].append(row)
    final = {}
    for strategy, observations in summary_map.items():
        agg = _aggregate_summary_observations(observations)
        final[strategy] = {
            "alerts": 0,
            "measured": int(agg.get("measured") or 0),
            "estimated_win_rate_pct": agg.get("estimated_win_rate_pct"),
            "average_expectancy_rr": agg.get("average_expectancy_rr"),
            "total_historical_trades": int(agg.get("total_historical_trades") or 0),
            "observed_symbols": int(agg.get("observed_symbols") or 0),
            "signals": agg.get("signals") or {},
        }
    return final


def _evaluate_candidate_backtest_gate(candidate):
    if not isinstance(candidate, dict):
        return False, "missing_candidate", {"win_rate_pct": None, "expectancy_rr": None, "trades": None}
    signal = str(candidate.get("signal") or "").upper()
    if signal not in ("BUY", "SELL"):
        return False, "non_directional_candidate", {"win_rate_pct": None, "expectancy_rr": None, "trades": None}
    require_edge = bool(getattr(config, "TELEGRAM_ALERT_ENTRY_REQUIRE_EDGE_METRICS", True))
    min_wr = getattr(config, "TELEGRAM_ALERT_ENTRY_MIN_HIST_WIN_RATE", 58.0)
    min_trades = getattr(config, "TELEGRAM_ALERT_ENTRY_MIN_HIST_TRADES", 8)
    min_exp = getattr(config, "TELEGRAM_ALERT_ENTRY_MIN_EXPECTANCY_RR", 0.05)
    try:
        min_wr = float(min_wr)
    except Exception:
        min_wr = 58.0
    try:
        min_trades = int(min_trades)
    except Exception:
        min_trades = 8
    try:
        min_exp = float(min_exp)
    except Exception:
        min_exp = 0.05
    metrics = _candidate_edge_metrics(candidate)
    wr = metrics.get("win_rate_pct")
    exp = metrics.get("expectancy_rr")
    trades = metrics.get("trades")
    has_edge = any(isinstance(v, (int, float)) for v in (wr, exp, trades))
    if require_edge and not has_edge:
        return False, "candidate_missing_edge_metrics", metrics
    if not isinstance(trades, (int, float)) or float(trades) < float(min_trades):
        return False, "candidate_trades_below_min", metrics
    if not isinstance(wr, (int, float)) or float(wr) < float(min_wr):
        return False, "candidate_win_rate_below_min", metrics
    if not isinstance(exp, (int, float)) or float(exp) < float(min_exp):
        return False, "candidate_expectancy_below_min", metrics
    return True, "pass", metrics


def _extract_walkforward_metrics(plan, optimizer_key="optimizer"):
    if not isinstance(plan, dict):
        return {}
    optimizer = plan.get(optimizer_key)
    best = optimizer.get("best") if isinstance(optimizer, dict) else None
    if not isinstance(best, dict):
        return {}
    metrics = {}
    for key in (
        "valid_trades",
        "valid_win_rate_pct",
        "valid_expectancy_rr",
        "train_trades",
        "train_win_rate_pct",
        "train_expectancy_rr",
        "robustness_score",
        "consistency_penalty",
        "opt_score",
    ):
        value = best.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[key] = float(value)
    return metrics


def _evaluate_walkforward_gate(plan, signal):
    if str(signal or "").upper() not in ("BUY", "SELL"):
        return True, "not_entry_signal", {}
    enabled = bool(getattr(config, "TELEGRAM_ALERT_ENTRY_REQUIRE_WALKFORWARD", True))
    if not enabled:
        return True, "walkforward_gate_disabled", {}
    metrics = _extract_walkforward_metrics(plan)
    if not metrics:
        return True, "walkforward_not_available", {}
    min_wr = getattr(config, "TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_WIN_RATE", 60.0)
    min_exp = getattr(config, "TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_EXPECTANCY_RR", 0.05)
    min_valid_trades = getattr(config, "TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_VALID_TRADES", 6)
    min_robustness = getattr(config, "TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_ROBUSTNESS", 45.0)
    try:
        min_wr = float(min_wr)
    except Exception:
        min_wr = 60.0
    try:
        min_exp = float(min_exp)
    except Exception:
        min_exp = 0.05
    try:
        min_valid_trades = int(min_valid_trades)
    except Exception:
        min_valid_trades = 6
    try:
        min_robustness = float(min_robustness)
    except Exception:
        min_robustness = 45.0
    valid_trades = metrics.get("valid_trades")
    valid_wr = metrics.get("valid_win_rate_pct")
    valid_exp = metrics.get("valid_expectancy_rr")
    robustness = metrics.get("robustness_score")
    if isinstance(valid_trades, (int, float)) and int(valid_trades) < int(min_valid_trades):
        return False, "walkforward_valid_trades_below_min", metrics
    if isinstance(valid_wr, (int, float)) and float(valid_wr) < float(min_wr):
        return False, "walkforward_win_rate_below_min", metrics
    if isinstance(valid_exp, (int, float)) and float(valid_exp) < float(min_exp):
        return False, "walkforward_expectancy_below_min", metrics
    if isinstance(robustness, (int, float)) and float(robustness) < float(min_robustness):
        return False, "walkforward_robustness_below_min", metrics
    return True, "pass", metrics


def _evaluate_sell_whitelist_gate(plan):
    enabled = bool(getattr(config, "TELEGRAM_ALERT_SELL_WHITELIST_ENABLE", True))
    if not enabled:
        return True, "sell_whitelist_disabled", {}
    metrics = _extract_walkforward_metrics(plan, optimizer_key="sell_optimizer")
    if not metrics:
        return False, "sell_walkforward_not_available", {}
    min_wr = getattr(config, "TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_WIN_RATE", 60.0)
    min_exp = getattr(config, "TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_EXPECTANCY_RR", 0.05)
    min_valid_trades = getattr(config, "TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_VALID_TRADES", 6)
    min_robustness = getattr(config, "TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_ROBUSTNESS", 45.0)
    try:
        min_wr = float(min_wr)
    except Exception:
        min_wr = 60.0
    try:
        min_exp = float(min_exp)
    except Exception:
        min_exp = 0.05
    try:
        min_valid_trades = int(min_valid_trades)
    except Exception:
        min_valid_trades = 6
    try:
        min_robustness = float(min_robustness)
    except Exception:
        min_robustness = 45.0
    valid_trades = metrics.get("valid_trades")
    valid_wr = metrics.get("valid_win_rate_pct")
    valid_exp = metrics.get("valid_expectancy_rr")
    robustness = metrics.get("robustness_score")
    if not isinstance(valid_trades, (int, float)) or int(valid_trades) < int(min_valid_trades):
        return False, "sell_walkforward_valid_trades_below_min", metrics
    if not isinstance(valid_wr, (int, float)) or float(valid_wr) < float(min_wr):
        return False, "sell_walkforward_win_rate_below_min", metrics
    if not isinstance(valid_exp, (int, float)) or float(valid_exp) < float(min_exp):
        return False, "sell_walkforward_expectancy_below_min", metrics
    if not isinstance(robustness, (int, float)) or float(robustness) < float(min_robustness):
        return False, "sell_walkforward_robustness_below_min", metrics
    return True, "pass", metrics


def _evaluate_entry_quality_gate(plan, signal):
    if str(signal or "").upper() not in ("BUY", "SELL"):
        return True, "not_entry_signal", {}
    enabled = bool(getattr(config, "TELEGRAM_ALERT_ENTRY_QUALITY_ENABLE", True))
    if not enabled:
        return True, "quality_gate_disabled", {}
    min_wr = getattr(config, "TELEGRAM_ALERT_ENTRY_MIN_HIST_WIN_RATE", 56.0)
    min_trades = getattr(config, "TELEGRAM_ALERT_ENTRY_MIN_HIST_TRADES", 8)
    min_exp = getattr(config, "TELEGRAM_ALERT_ENTRY_MIN_EXPECTANCY_RR", 0.05)
    require_edge = bool(getattr(config, "TELEGRAM_ALERT_ENTRY_REQUIRE_EDGE_METRICS", True))
    try:
        min_wr = float(min_wr)
    except Exception:
        min_wr = 56.0
    try:
        min_trades = int(min_trades)
    except Exception:
        min_trades = 8
    try:
        min_exp = float(min_exp)
    except Exception:
        min_exp = 0.05
    if not isinstance(plan, dict):
        return False, "missing_plan", {}
    metrics = _extract_signal_edge_metrics(plan, signal)
    wr = metrics.get("win_rate_pct")
    trades = metrics.get("trades")
    exp = metrics.get("expectancy_rr")
    has_edge_metrics = any(isinstance(v, (int, float)) for v in (wr, trades, exp))
    if require_edge and not has_edge_metrics:
        return False, "missing_edge_metrics", metrics
    if isinstance(trades, (int, float)) and int(trades) < int(min_trades):
        return False, "trades_below_min", metrics
    if isinstance(wr, (int, float)) and float(wr) < float(min_wr):
        return False, "win_rate_below_min", metrics
    if isinstance(exp, (int, float)) and float(exp) < float(min_exp):
        return False, "expectancy_below_min", metrics
    wf_ok, wf_reason, wf_metrics = _evaluate_walkforward_gate(plan, signal)
    if not wf_ok:
        merged = metrics.copy()
        merged.update(wf_metrics)
        return False, wf_reason, merged
    if str(signal or "").upper() == "SELL":
        sell_ok, sell_reason, sell_metrics = _evaluate_sell_whitelist_gate(plan)
        if not sell_ok:
            merged = metrics.copy()
            merged.update(sell_metrics)
            return False, sell_reason, merged
    return True, "pass", metrics


def _evaluate_super_signal(item, signal):
    """
    Evaluates if a signal qualifies as a 'Super Signal' (Highest Win Rate)
    Requires:
    1. At least 2 strategies with high win rate (>65%) or 3+ strategies in confluence.
    2. Strong 1H Trend (ADX > 25).
    3. Positive Expectancy (>0.15).
    4. Candlestick Pattern confirmation.
    """
    if str(signal or "").upper() not in ("BUY", "SELL"):
        return False, "not_entry", {}

    # 1. Check Confluence & Backtest Stats
    allow_cdc = _strict_60_allow_cdc()
    strategies = [
        ("ActionZone", item.get("actionzone_15m"), True),
        ("EMACross", item.get("ema_cross_15m"), True),
        ("CDCVixFix", item.get("cdc_vixfix_15m"), allow_cdc),
        ("ShortTerm", item.get("short_term_15m"), True),
        ("Sniper", item.get("sniper_15m"), True),
        ("Quantum", item.get("quantum_15m"), True),
    ]

    confirmed_strategies = []
    total_wr = 0.0
    total_exp = 0.0
    total_trades = 0.0
    count = 0
    
    for name, plan, enabled in strategies:
        if not enabled:
            continue
        if not isinstance(plan, dict):
            continue
        
        plan_sig = _plan_trade_direction(plan)
        if plan_sig != signal:
            continue
        if _strict_60_mode_enabled() and not _passes_entry_quality_gate(plan, signal):
            continue
            
        metrics = _extract_plan_edge_metrics(plan)
        wr = metrics.get("win_rate_pct")
        exp = metrics.get("expectancy_rr")
        trades = metrics.get("trades")
        
        # Must have minimum history to be trusted
        if not isinstance(trades, (int, float)) or trades < 8:
            continue
            
        if isinstance(wr, (int, float)) and wr >= 60.0:
            confirmed_strategies.append(name)
            total_wr += wr
            if isinstance(exp, (int, float)):
                total_exp += exp
            total_trades += float(trades)
            count += 1

    # Quality Gate 1: Confluence
    if count < 2:
        return False, "insufficient_confluence", {}

    avg_wr = total_wr / count if count > 0 else 0
    avg_exp = total_exp / count if count > 0 else 0

    # Quality Gate 2: Strict Stats
    if avg_wr < 65.0:
        return False, "win_rate_too_low", {"avg_wr": avg_wr}
    if avg_exp < 0.15:
        return False, "expectancy_too_low", {"avg_exp": avg_exp}

    # Quality Gate 3: Trend & Market Context
    # Check 1H ADX and Trend Alignment from ActionZone (if available) or shared context
    az_plan = item.get("actionzone_15m")
    if isinstance(az_plan, dict):
        adx = az_plan.get("adx")
        trend_strength = str(az_plan.get("trend_strength") or "").upper()
        trend_ok = bool(az_plan.get("trend_alignment", True))
        
        if not trend_ok:
            return False, "trend_misalignment", {}
        if isinstance(adx, (int, float)) and adx < 25.0:
            return False, "weak_adx", {"adx": adx}
    
    # Quality Gate 4: Candlestick Confirmation
    primary_plan = _pick_primary_trade_plan(
        item,
        signal=signal,
        require_quality=_strict_60_mode_enabled(),
        allow_cdc=allow_cdc,
    )
    pattern = str(primary_plan.get("detected_pattern") or "").upper() if isinstance(primary_plan, dict) else "NONE"
    if pattern == "NONE" or not pattern:
        return False, "no_candlestick_confirmation", {}

    return True, "pass", {
        "avg_wr": avg_wr,
        "avg_exp": avg_exp,
        "avg_trades": (total_trades / count) if count > 0 else 0.0,
        "confluence": confirmed_strategies,
        "count": count
    }


def _passes_entry_quality_gate(plan, signal):
    ok, _, _ = _evaluate_entry_quality_gate(plan, signal)
    return ok


def _get_best_confidence(item, signal=None, require_quality=False, allow_cdc=True):
    best = None
    candidates = [
        (item.get("short_term_15m"), "setup", "confidence", True),
        (item.get("sniper_15m"), "setup", "confidence", True),
        (item.get("quantum_15m"), "setup", "confidence", True),
        (item.get("ema_cross_15m"), "signal", "confidence", True),
        (item.get("actionzone_15m"), "signal", "confidence", True),
        (item.get("cdc_vixfix_15m"), "signal", "confidence", allow_cdc),
        (item.get("crypto_reversal_15m"), "setup", "confidence", True),
    ]
    for plan, setup_key, conf_key, enabled in candidates:
        if not enabled:
            continue
        if not isinstance(plan, dict):
            continue
        setup = plan.get(setup_key)
        if setup_key in ("setup", "signal") and not _is_actionable_setup_value(setup):
            continue
        direction = _plan_trade_direction(plan)
        if signal in ("BUY", "SELL") and direction != signal:
            continue
        if require_quality and direction in ("BUY", "SELL") and not _passes_entry_quality_gate(plan, direction):
            continue
        conf = _normalize_confidence(plan.get(conf_key))
        if conf is None:
            conf = _plan_confidence_value(plan)
        if conf is None:
            continue
        if best is None or conf > best:
            best = conf
    return best


def _pick_primary_trade_plan(item, signal=None, require_quality=False, allow_cdc=True):
    candidates = [
        (item.get("cdc_vixfix_15m"), allow_cdc),
        (item.get("actionzone_15m"), True),
        (item.get("ema_cross_15m"), True),
        (item.get("short_term_15m"), True),
        (item.get("sniper_15m"), True),
        (item.get("quantum_15m"), True),
        (item.get("crypto_reversal_15m"), True),
    ]
    best_plan = None
    best_conf = -1.0
    for plan, enabled in candidates:
        if not enabled:
            continue
        if not isinstance(plan, dict):
            continue
        if "signal" in plan and not _is_actionable_setup_value(plan.get("signal")):
            continue
        if "setup" in plan and not _is_actionable_setup_value(plan.get("setup")):
            continue
        direction = _plan_trade_direction(plan)
        if signal in ("BUY", "SELL") and direction != signal:
            continue
        if require_quality and direction in ("BUY", "SELL") and not _passes_entry_quality_gate(plan, direction):
            continue
        conf = _plan_confidence_value(plan)
        if conf is None:
            conf = 0.0
        if conf > best_conf:
            best_conf = conf
            best_plan = plan
    return best_plan


def _get_pattern_reference(pattern):
    if not isinstance(pattern, str):
        return None
    normalized = pattern.strip().lower().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized or normalized == "none":
        return None
    if "engulf" in normalized:
        return {
            "label": "Engulfing Pattern",
            "url": "https://chartswatcher.com/pages/blog/a-trader-s-guide-to-the-engulfing-pattern-candlestick",
        }
    if "pinbar" in normalized or "pin bar" in normalized:
        return {
            "label": "Pin Bar Pattern",
            "url": "https://www.colibritrader.com/price-action-trading-patterns/",
        }
    return None


def _append_pattern_context_lines(lines, pattern):
    lines.append("<b>🧭 Timeframe Reference:</b> Trend reference: 1H | Entry pattern: 15m")
    if pattern and pattern != "None":
        lines.append(f"<b>🕯️ Price Pattern:</b> {_html_escape(pattern)}")
        reference = _get_pattern_reference(pattern)
        if isinstance(reference, dict):
            label = _html_escape(str(reference.get("label") or "Pattern"))
            url = str(reference.get("url") or "").strip()
            if url:
                lines.append(f'<a href="{url}">📚 อธิบาย Pattern นี้คืออะไร ({label})</a>')


def _alert_profile_metric_points(value, floor, ceiling, weight):
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 0.0, 0.0
    floor = float(floor)
    ceiling = float(ceiling)
    if ceiling <= floor:
        ceiling = floor + 1.0
    normalized = (float(value) - floor) / (ceiling - floor)
    normalized = max(0.0, min(1.0, normalized))
    return normalized * float(weight), float(weight)


def _alert_mode_usage_hint(mode_label=None):
    mode_text = str(mode_label or "").strip().lower()
    if not mode_text:
        return None
    if "baseline" in mode_text:
        return "ใช้หาเทรนด์เด่นของวัน แล้วรอเข้าเมื่อราคาไม่ไล่ห่างจากจุดเข้า"
    if "daily trend" in mode_text:
        return "ใช้กำหนด bias ของวัน แล้วรอระบบหลักยืนยันก่อนเข้าไม้จริง"
    if "trend pick" in mode_text or "relaxed" in mode_text:
        return "เข้าได้เมื่อ 1H, Forecast และราคาใกล้จุดเข้าไปทางเดียวกัน"
    return None


def _resolve_alert_profile_meta(win_rate=None, confidence=None, expectancy=None, trades=None, mode_label=None):
    wr_points, wr_weight = _alert_profile_metric_points(win_rate, 48.0, 66.0, 40.0)
    conf_points, conf_weight = _alert_profile_metric_points(confidence, 55.0, 90.0, 30.0)
    exp_points, exp_weight = _alert_profile_metric_points(expectancy, -0.05, 0.12, 20.0)
    trade_points, trade_weight = _alert_profile_metric_points(trades, 0.0, 10.0, 10.0)
    total_weight = wr_weight + conf_weight + exp_weight + trade_weight
    if total_weight <= 0:
        return None

    composite_score = ((wr_points + conf_points + exp_points + trade_points) / total_weight) * 100.0
    wr = float(win_rate) if isinstance(win_rate, (int, float)) and math.isfinite(float(win_rate)) else None
    conf = float(confidence) if isinstance(confidence, (int, float)) and math.isfinite(float(confidence)) else None

    if composite_score >= 85.0 and ((wr is not None and wr >= 62.0) or (conf is not None and conf >= 82.0)):
        tier = "A"
        title = "Precision Setup"
        style_text = "คัดเข้ม เน้นไม้ที่สถิติและโครงสร้างพร้อมสุด"
        action_text = "เข้าได้ตามแผนหลัก แต่ยังต้องเคารพ SL และจังหวะเข้า"
    elif composite_score >= 70.0 and ((wr is not None and wr >= 56.0) or (conf is not None and conf >= 72.0)):
        tier = "B"
        title = "Strong Setup"
        style_text = "คุณภาพดี ใช้งานจริงได้ต่อเนื่อง และยังคงความถี่เฉลี่ย"
        action_text = "เข้าได้ แต่ควรลดขนาดไม้ลงเล็กน้อยหากราคาเริ่มไล่ห่าง"
    elif composite_score >= 55.0:
        tier = "C"
        title = "Standard Setup"
        style_text = "สมดุลระหว่างความถี่กับคุณภาพ เหมาะเป็นสัญญาณใช้งานประจำ"
        action_text = "เข้าเมื่อเทรนด์ 1H, Forecast และ pattern ไม่ขัดกัน"
    else:
        tier = "D"
        title = "Watchlist Setup"
        style_text = "เน้นใช้หา bias หรือเฝ้ารอ ไม่เหมาะไล่เข้าเต็มแผน"
        action_text = "ใช้ดูทิศทางก่อน รอแท่งยืนยันหรือระบบหลักตามซ้ำ"

    mode_hint = _alert_mode_usage_hint(mode_label=mode_label)
    if mode_hint:
        action_text = mode_hint

    basis_label = None
    basis_value = None
    if wr is not None:
        basis_label = "Win Rate ย้อนหลัง"
        basis_value = f"{wr:.1f}%"
    elif conf is not None:
        basis_label = "Confidence ล่าสุด"
        basis_value = f"{conf:.0f}%"
    elif isinstance(expectancy, (int, float)) and math.isfinite(float(expectancy)):
        basis_label = "Expectancy RR"
        basis_value = f"{float(expectancy):.2f}"

    decision_parts = []
    if wr is not None:
        decision_parts.append(f"WR {wr:.1f}%")
    if conf is not None:
        decision_parts.append(f"Conf {conf:.0f}%")
    if isinstance(expectancy, (int, float)) and math.isfinite(float(expectancy)):
        decision_parts.append(f"ExpRR {float(expectancy):.2f}")
    if isinstance(trades, (int, float)) and math.isfinite(float(trades)) and float(trades) > 0:
        decision_parts.append(f"Trades {int(float(trades))}")

    return {
        "level": tier,
        "tier": tier,
        "is_high": tier in ("A", "B"),
        "title": title,
        "style_text": style_text,
        "action_text": action_text,
        "basis_label": basis_label,
        "basis_value": basis_value,
        "decision_parts": decision_parts,
        "composite_score": round(float(composite_score), 1),
    }


def _build_alert_profile_lines(win_rate=None, confidence=None, expectancy=None, trades=None, mode_label=None):
    meta = _resolve_alert_profile_meta(
        win_rate=win_rate,
        confidence=confidence,
        expectancy=expectancy,
        trades=trades,
        mode_label=mode_label,
    )
    if not isinstance(meta, dict):
        return []
    lines = [
        f"<b>🏷️ ระดับสัญญาณ:</b> {meta['tier']} | {_html_escape(str(meta['title']))}",
        f"<b>🧠 วิธีใช้:</b> {_html_escape(str(meta['action_text']))}",
        f"<b>🧪 โปรไฟล์:</b> {_html_escape(str(meta['style_text']))}",
    ]
    if meta.get("basis_label") and meta.get("basis_value"):
        lines.append(f"<b>📚 ค่าอ้างอิง:</b> {_html_escape(str(meta['basis_label']))} {meta['basis_value']}")
    if meta.get("decision_parts"):
        lines.append("<b>✅ ใช้ตัดสินใจ:</b> " + " | ".join([_html_escape(str(part)) for part in meta["decision_parts"]]))
    return lines


def _alert_profile_score_adjustment(win_rate=None, confidence=None, expectancy=None, trades=None, mode_label=None):
    meta = _resolve_alert_profile_meta(
        win_rate=win_rate,
        confidence=confidence,
        expectancy=expectancy,
        trades=trades,
        mode_label=mode_label,
    )
    if not isinstance(meta, dict):
        return 0.0
    tier = str(meta.get("tier") or "D").upper()
    if tier == "A":
        return 7.0
    if tier == "B":
        return 3.5
    if tier == "C":
        return 1.0
    return -2.0


def _candidate_mode_label(candidate):
    if not isinstance(candidate, dict):
        return None
    strategy = str(candidate.get("strategy") or "").strip().upper()
    if strategy == "DAILY_BEST":
        mode = str(candidate.get("daily_best_mode") or "").strip().lower()
        if mode == "baseline":
            return "Baseline Trend Pick"
        if mode == "relaxed":
            return "Trend Pick"
        return "Strict Pick"
    if strategy == "CDCVIX15":
        mode = str(candidate.get("cdc_mode") or "").strip().lower()
        if mode == "daily_trend":
            return "Daily Trend"
    return None


def _candidate_alert_profile(candidate):
    if not isinstance(candidate, dict):
        return None
    edge = _candidate_edge_metrics(candidate)
    return _resolve_alert_profile_meta(
        win_rate=edge.get("win_rate_pct"),
        confidence=candidate.get("confidence"),
        expectancy=edge.get("expectancy_rr"),
        trades=edge.get("trades"),
        mode_label=_candidate_mode_label(candidate),
    )


_ALL_WEATHER_REGIME_WEIGHTS = {
    "TREND": {
        "ActionZone": 1.35,
        "EMACross": 1.28,
        "CDCVixFix": 0.82,
        "CryptoReversal": 0.90,
        "ShortTerm": 1.00,
        "Sniper": 1.02,
        "Quantum": 0.95,
    },
    "RANGE": {
        "ActionZone": 0.92,
        "EMACross": 0.84,
        "CDCVixFix": 1.28,
        "CryptoReversal": 1.20,
        "ShortTerm": 1.10,
        "Sniper": 1.08,
        "Quantum": 1.02,
    },
    "HIGH_VOL": {
        "ActionZone": 1.00,
        "EMACross": 0.80,
        "CDCVixFix": 1.30,
        "CryptoReversal": 1.25,
        "ShortTerm": 0.95,
        "Sniper": 0.92,
        "Quantum": 0.90,
    },
}


def _all_weather_clip(value, low, high):
    try:
        value = float(value)
    except Exception:
        return float(low)
    return float(max(float(low), min(float(high), value)))


def _all_weather_regime_weight(label, regime):
    weights = _ALL_WEATHER_REGIME_WEIGHTS.get(str(regime or "").upper(), {})
    try:
        return float(weights.get(str(label or "").strip(), 1.0))
    except Exception:
        return 1.0


def _all_weather_market_regime(item):
    vol_samples = []
    for plan, key in (
        (item.get("actionzone_15m"), "avg_range_pct"),
        (item.get("actionzone_15m"), "atr_pct"),
        (item.get("crypto_reversal_15m"), "atr_pct"),
    ):
        if not isinstance(plan, dict):
            continue
        value = plan.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0:
            vol_samples.append(float(value))
    change_pct = item.get("change")
    if isinstance(change_pct, (int, float)) and math.isfinite(float(change_pct)):
        vol_samples.append(abs(float(change_pct)))
    volatility_pct = 0.0
    if vol_samples:
        vol_samples.sort()
        volatility_pct = float(vol_samples[len(vol_samples) // 2])

    trend_score = 0.0
    az_plan = item.get("actionzone_15m")
    if isinstance(az_plan, dict):
        if bool(az_plan.get("trend_alignment", False)):
            trend_score += 0.60
        adx = az_plan.get("adx")
        if isinstance(adx, (int, float)) and math.isfinite(float(adx)):
            adx = float(adx)
            if adx >= 30.0:
                trend_score += 1.00
            elif adx >= 24.0:
                trend_score += 0.80
            elif adx >= 18.0:
                trend_score += 0.50
        strength = str(az_plan.get("trend_strength") or "").upper()
        if strength == "STRONG":
            trend_score += 0.50
        elif strength == "MODERATE":
            trend_score += 0.25
    ema_plan = item.get("ema_cross_15m")
    if isinstance(ema_plan, dict):
        direction = _plan_trade_direction(ema_plan)
        if direction in ("BUY", "SELL"):
            trend_score += 0.25
        bars_since_cross = ema_plan.get("bars_since_cross")
        if isinstance(bars_since_cross, (int, float)) and float(bars_since_cross) <= 2:
            trend_score += 0.20

    high_vol_threshold = getattr(config, "ALL_WEATHER_15M_HIGH_VOL_THRESHOLD", 2.2)
    trend_threshold = getattr(config, "ALL_WEATHER_15M_TREND_THRESHOLD", 1.5)
    try:
        high_vol_threshold = float(high_vol_threshold)
    except Exception:
        high_vol_threshold = 2.2
    try:
        trend_threshold = float(trend_threshold)
    except Exception:
        trend_threshold = 1.5

    regime = "RANGE"
    if volatility_pct >= high_vol_threshold:
        regime = "HIGH_VOL"
    elif trend_score >= trend_threshold:
        regime = "TREND"
    return {
        "regime": regime,
        "volatility_pct": float(volatility_pct),
        "trend_score": float(trend_score),
    }


def _all_weather_freshness_bonus(bars_since_signal):
    if not isinstance(bars_since_signal, (int, float)) or not math.isfinite(float(bars_since_signal)):
        return 0.0
    bars = float(bars_since_signal)
    if bars <= 0:
        return 4.0
    if bars <= 2:
        return 2.0
    if bars > 6:
        return -3.0
    return 0.0


def _all_weather_metrics_bonus(win_rate, expectancy, trades):
    bonus_wr = 0.0
    bonus_exp = 0.0
    bonus_trades = 0.0
    if isinstance(win_rate, (int, float)) and math.isfinite(float(win_rate)):
        bonus_wr = _all_weather_clip((float(win_rate) - 50.0) * 0.25, -6.0, 10.0)
    if isinstance(expectancy, (int, float)) and math.isfinite(float(expectancy)):
        bonus_exp = _all_weather_clip(float(expectancy) * 8.0, -6.0, 8.0)
    if isinstance(trades, (int, float)) and math.isfinite(float(trades)):
        bonus_trades = _all_weather_clip(float(trades) / 6.0, 0.0, 6.0)
    return float(bonus_wr + bonus_exp + bonus_trades)


def _all_weather_plan_candidates(item, regime):
    plans = [
        ("ActionZone", item.get("actionzone_15m")),
        ("EMACross", item.get("ema_cross_15m")),
        ("CDCVixFix", item.get("cdc_vixfix_15m")),
        ("CryptoReversal", item.get("crypto_reversal_15m")),
        ("ShortTerm", item.get("short_term_15m")),
        ("Sniper", item.get("sniper_15m")),
        ("Quantum", item.get("quantum_15m")),
    ]
    candidates = []
    for label, plan in plans:
        if not isinstance(plan, dict):
            continue
        direction = _plan_trade_direction(plan)
        if direction not in ("BUY", "SELL") or not _is_entry_signal(plan, direction):
            continue
        confidence = _plan_confidence_value(plan)
        if not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
            continue
        gate_ok, _, edge = _evaluate_entry_quality_gate(plan, direction)
        if not gate_ok:
            continue
        bars_since = _pick_plan_value(plan, ["bars_since_signal", "bars_since_entry", "bars_since_cross"])
        win_rate = edge.get("win_rate_pct")
        expectancy = edge.get("expectancy_rr")
        trades = edge.get("trades")
        metrics_bonus = _all_weather_metrics_bonus(win_rate, expectancy, trades)
        freshness_bonus = _all_weather_freshness_bonus(bars_since)
        regime_weight = _all_weather_regime_weight(label, regime)
        score = float(confidence) + metrics_bonus + freshness_bonus + ((float(regime_weight) - 1.0) * 16.0)
        candidates.append(
            {
                "label": label,
                "plan": plan,
                "signal": direction,
                "confidence": float(confidence),
                "win_rate_pct": float(win_rate) if isinstance(win_rate, (int, float)) else None,
                "expectancy_rr": float(expectancy) if isinstance(expectancy, (int, float)) else None,
                "trades": float(trades) if isinstance(trades, (int, float)) else None,
                "bars_since_signal": float(bars_since) if isinstance(bars_since, (int, float)) else None,
                "regime_weight": float(regime_weight),
                "metrics_bonus": float(metrics_bonus),
                "freshness_bonus": float(freshness_bonus),
                "score": float(score),
            }
        )
    candidates.sort(key=lambda row: (float(row.get("score", 0.0)), float(row.get("confidence", 0.0))), reverse=True)
    return candidates


def _all_weather_blended_metrics(rows):
    weighted_wr = 0.0
    weighted_exp = 0.0
    total_alpha = 0.0
    trade_sum = 0.0
    measured = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        alpha = max(0.1, float(row.get("confidence", 0.0)) / 100.0)
        wr = row.get("win_rate_pct")
        exp = row.get("expectancy_rr")
        trades = row.get("trades")
        if isinstance(wr, (int, float)) and math.isfinite(float(wr)):
            weighted_wr += alpha * float(wr)
            total_alpha += alpha
        if isinstance(exp, (int, float)) and math.isfinite(float(exp)):
            weighted_exp += alpha * float(exp)
        if isinstance(trades, (int, float)) and math.isfinite(float(trades)):
            trade_sum += float(trades)
            measured += 1
    if total_alpha <= 0:
        wr_blend = None
        exp_blend = None
    else:
        wr_blend = weighted_wr / total_alpha
        exp_blend = weighted_exp / total_alpha
    trades_blend = int(round(trade_sum / measured)) if measured > 0 else 0
    return {
        "win_rate_pct": wr_blend,
        "expectancy_rr": exp_blend,
        "trades": trades_blend,
        "measured_count": measured,
    }


def _build_all_weather_report_entry(item, min_conf=None):
    symbol = normalize_symbol(item.get("symbol") or "") if isinstance(item, dict) else ""
    report = {
        "symbol": symbol,
        "status": "unknown",
        "enabled": bool(getattr(config, "ALL_WEATHER_15M_ENABLED", True)),
        "signal": "WAIT",
        "passed": False,
        "regime": None,
        "volatility_pct": None,
        "trend_score": None,
        "input_min_confidence": None,
        "required_confidence": None,
        "final_confidence": None,
        "delta": None,
        "top_k": None,
        "min_confluence": None,
        "top_buy_score": None,
        "top_sell_score": None,
        "side_gap": None,
        "selected_side": None,
        "selected_sources": [],
        "primary_source": None,
        "blend": {},
        "candidate_pool_count": 0,
        "candidates": [],
        "cache_key": None,
    }
    if not report["enabled"]:
        report["status"] = "disabled"
        return report
    if min_conf is None:
        min_conf = getattr(config, "ALL_WEATHER_15M_MIN_ALERT_CONFIDENCE", 69.0)
    try:
        min_conf = float(min_conf)
    except Exception:
        min_conf = 69.0
    report["input_min_confidence"] = float(min_conf)

    regime_meta = _all_weather_market_regime(item)
    regime = regime_meta.get("regime") or "RANGE"
    report["regime"] = regime
    report["volatility_pct"] = float(regime_meta.get("volatility_pct") or 0.0)
    report["trend_score"] = float(regime_meta.get("trend_score") or 0.0)

    candidates = _all_weather_plan_candidates(item, regime)
    report["candidate_pool_count"] = len(candidates)
    report["candidates"] = [
        {
            "label": str(row.get("label") or ""),
            "signal": str(row.get("signal") or ""),
            "confidence": float(row.get("confidence", 0.0)),
            "score": float(row.get("score", 0.0)),
            "regime_weight": float(row.get("regime_weight", 1.0)),
            "metrics_bonus": float(row.get("metrics_bonus", 0.0)),
            "freshness_bonus": float(row.get("freshness_bonus", 0.0)),
            "win_rate_pct": float(row.get("win_rate_pct")) if isinstance(row.get("win_rate_pct"), (int, float)) else None,
            "expectancy_rr": float(row.get("expectancy_rr")) if isinstance(row.get("expectancy_rr"), (int, float)) else None,
            "trades": int(row.get("trades")) if isinstance(row.get("trades"), (int, float)) else None,
            "bars_since_signal": float(row.get("bars_since_signal")) if isinstance(row.get("bars_since_signal"), (int, float)) else None,
        }
        for row in candidates
    ]
    if not candidates:
        report["status"] = "no_actionable_subplans"
        return report

    buy_rows = [row for row in candidates if row.get("signal") == "BUY"]
    sell_rows = [row for row in candidates if row.get("signal") == "SELL"]
    top_buy = buy_rows[0] if buy_rows else None
    top_sell = sell_rows[0] if sell_rows else None

    delta = getattr(config, "ALL_WEATHER_15M_DELTA", 3.0)
    top_k = getattr(config, "ALL_WEATHER_15M_TOP_K", 2)
    min_confluence = getattr(config, "ALL_WEATHER_15M_MIN_CONFLUENCE", 2)
    try:
        delta = float(delta)
    except Exception:
        delta = 3.0
    try:
        top_k = max(1, int(top_k))
    except Exception:
        top_k = 2
    try:
        min_confluence = max(1, int(min_confluence))
    except Exception:
        min_confluence = 2
    report["delta"] = float(delta)
    report["top_k"] = int(top_k)
    report["min_confluence"] = int(min_confluence)
    report["top_buy_score"] = float(top_buy.get("score", 0.0)) if top_buy else None
    report["top_sell_score"] = float(top_sell.get("score", 0.0)) if top_sell else None

    side = None
    side_gap = 0.0
    if top_buy and (not top_sell or float(top_buy["score"]) >= float(top_sell["score"]) + float(delta)):
        side = "BUY"
        side_gap = float(top_buy["score"]) - float(top_sell["score"]) if top_sell else float(top_buy["score"])
    elif top_sell and (not top_buy or float(top_sell["score"]) >= float(top_buy["score"]) + float(delta)):
        side = "SELL"
        side_gap = float(top_sell["score"]) - float(top_buy["score"]) if top_buy else float(top_sell["score"])
    report["selected_side"] = side
    report["side_gap"] = float(side_gap)
    if side not in ("BUY", "SELL"):
        report["status"] = "side_not_clear"
        return report

    selected_rows = [row for row in candidates if row.get("signal") == side][:top_k]
    source_labels = [str(row.get("label") or "") for row in selected_rows if str(row.get("label") or "").strip()]
    report["selected_sources"] = source_labels
    report["primary_source"] = source_labels[0] if source_labels else None
    if len(selected_rows) < min_confluence:
        report["status"] = "min_confluence_not_met"
        return report

    blend = _all_weather_blended_metrics(selected_rows)
    report["blend"] = {
        "win_rate_pct": float(blend.get("win_rate_pct")) if isinstance(blend.get("win_rate_pct"), (int, float)) else None,
        "expectancy_rr": float(blend.get("expectancy_rr")) if isinstance(blend.get("expectancy_rr"), (int, float)) else None,
        "trades": int(blend.get("trades")) if isinstance(blend.get("trades"), (int, float)) else 0,
        "measured_count": int(blend.get("measured_count") or 0),
    }
    require_backtest_pass = bool(getattr(config, "ALL_WEATHER_15M_REQUIRE_BACKTEST_PASS", True))
    min_valid_trades = getattr(config, "ALL_WEATHER_15M_MIN_VALID_TRADES", 6)
    min_valid_wr = getattr(config, "ALL_WEATHER_15M_MIN_VALID_WIN_RATE", 56.0)
    min_valid_exp = getattr(config, "ALL_WEATHER_15M_MIN_VALID_EXPECTANCY", 0.03)
    aw_min_conf = getattr(config, "ALL_WEATHER_15M_MIN_ALERT_CONFIDENCE", 69.0)
    try:
        min_valid_trades = int(min_valid_trades)
    except Exception:
        min_valid_trades = 6
    try:
        min_valid_wr = float(min_valid_wr)
    except Exception:
        min_valid_wr = 56.0
    try:
        min_valid_exp = float(min_valid_exp)
    except Exception:
        min_valid_exp = 0.03
    try:
        aw_min_conf = float(aw_min_conf)
    except Exception:
        aw_min_conf = 69.0
    required_conf = max(float(min_conf), float(aw_min_conf))
    report["required_confidence"] = float(required_conf)

    if require_backtest_pass:
        if int(blend.get("measured_count") or 0) <= 0:
            report["status"] = "backtest_metrics_missing"
            return report
        if int(blend.get("trades") or 0) < int(min_valid_trades):
            report["status"] = "backtest_trades_below_min"
            return report
        wr_blend = blend.get("win_rate_pct")
        if isinstance(wr_blend, (int, float)) and float(wr_blend) < float(min_valid_wr):
            report["status"] = "backtest_win_rate_below_min"
            return report
        exp_blend = blend.get("expectancy_rr")
        if isinstance(exp_blend, (int, float)) and float(exp_blend) < float(min_valid_exp):
            report["status"] = "backtest_expectancy_below_min"
            return report

    selected_top = selected_rows[0]
    wr_blend = blend.get("win_rate_pct")
    conf_wr = 0.0
    if isinstance(wr_blend, (int, float)) and math.isfinite(float(wr_blend)):
        conf_wr = _all_weather_clip((float(wr_blend) - 50.0) * 0.20, 0.0, 8.0)
    final_conf = _all_weather_clip(
        float(selected_top.get("confidence", 0.0)) + min(12.0, 3.0 * len(selected_rows)) + conf_wr,
        0.0,
        98.0,
    )
    report["final_confidence"] = float(final_conf)
    if final_conf < required_conf:
        report["status"] = "confidence_below_min"
        return report

    primary_plan = selected_top.get("plan")
    last_signal_key = ""
    if isinstance(primary_plan, dict):
        last_signal_key = str(primary_plan.get("last_signal_time") or "").strip()
    if not last_signal_key:
        last_signal_key = "|".join(source_labels) or get_thai_now().strftime("%Y%m%d%H")
    report["status"] = "pass"
    report["signal"] = side
    report["passed"] = True
    report["score"] = float(final_conf) + min(8.0, len(selected_rows) * 2.0) + min(10.0, max(0.0, float(side_gap)))
    report["cache_key"] = f"AW15|{symbol}|{side}|{regime}|{last_signal_key}"
    report["signal_payload"] = {
        "signal": side,
        "confidence": float(final_conf),
        "score": float(report["score"]),
        "primary_plan": primary_plan,
        "sources": source_labels,
        "selected_rows": selected_rows,
        "regime": regime,
        "volatility_pct": float(regime_meta.get("volatility_pct") or 0.0),
        "trend_score": float(regime_meta.get("trend_score") or 0.0),
        "side_gap": float(side_gap),
        "top_buy_score": report["top_buy_score"],
        "top_sell_score": report["top_sell_score"],
        "blend": blend,
        "cache_key": report["cache_key"],
    }
    return report


def _build_all_weather_signal(item, min_conf):
    report = _build_all_weather_report_entry(item, min_conf=min_conf)
    payload = report.get("signal_payload") if isinstance(report, dict) else None
    if isinstance(payload, dict) and bool(report.get("passed")):
        return payload, {"status": "pass"}
    meta = {
        "status": str(report.get("status") or "filtered") if isinstance(report, dict) else "filtered",
        "side": report.get("selected_side") if isinstance(report, dict) else None,
        "confidence": report.get("final_confidence") if isinstance(report, dict) else None,
    }
    return None, meta


def _message_module_helpers():
    return {
        "normalize_symbol": normalize_symbol,
        "html_escape": _html_escape,
        "format_price_value": _format_price_value,
        "pick_primary_trade_plan": _pick_primary_trade_plan,
        "strict_60_mode_enabled": _strict_60_mode_enabled,
        "strict_60_allow_cdc": _strict_60_allow_cdc,
        "extract_signal_edge_metrics": _extract_signal_edge_metrics,
        "build_alert_profile_lines": _build_alert_profile_lines,
        "append_pattern_context_lines": _append_pattern_context_lines,
        "build_stop_context_lines": _build_stop_context_lines,
        "get_plan_label": _get_plan_label,
        "format_exit_levels_lines": _format_exit_levels_lines,
        "format_price_forecast_lines": _format_price_forecast_lines,
        "alert_mode_usage_hint": _alert_mode_usage_hint,
        "plan_confidence_value": _plan_confidence_value,
        "pick_plan_value": _pick_plan_value,
    }


def _build_all_weather_message(item, aw_signal):
    return _alerts_build_all_weather_message(
        item,
        aw_signal,
        helpers={**_message_module_helpers(), "build_telegram_message": _build_telegram_message},
        get_now=get_thai_now,
    )


def _build_super_signal_message(item, signal, super_meta, primary_plan=None):
    return _alerts_build_super_signal_message(
        item,
        signal,
        super_meta,
        primary_plan=primary_plan,
        helpers=_message_module_helpers(),
        get_now=get_thai_now,
    )


def _build_telegram_message(item, signal, best_conf, sources, primary_plan=None, mode_label=None):
    return _alerts_build_telegram_message(
        item,
        signal,
        best_conf,
        sources,
        primary_plan=primary_plan,
        mode_label=mode_label,
        helpers=_message_module_helpers(),
        get_now=get_thai_now,
    )


def _build_daily_best_pick_message(item, signal, best_conf, sources, primary_plan=None, strategy_label=None, selection_score=None, mode_label=None):
    return _alerts_build_daily_best_pick_message(
        item,
        signal,
        best_conf,
        sources,
        primary_plan=primary_plan,
        strategy_label=strategy_label,
        selection_score=selection_score,
        mode_label=mode_label,
        helpers={**_message_module_helpers(), "build_telegram_message": _build_telegram_message},
        get_now=get_thai_now,
    )


def _build_price_action_message(item, plan):
    return _alerts_build_price_action_message(
        item,
        plan,
        helpers=_message_module_helpers(),
        get_now=get_thai_now,
    )


def _build_actionzone_message(item, az_plan):
    signal = str(az_plan.get("signal") or az_plan.get("raw_signal") or "").upper()
    emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
    symbol = normalize_symbol(item.get("symbol") or "")
    name = _html_escape(str(item.get("name") or "").strip())
    
    tv_symbol = symbol.replace("-", "")
    
    lines = [f"<b>{emoji} ActionZone 15m {signal} | {_html_escape(symbol)}</b>"]
    if name:
        lines.append(f"<i>{name}</i>")
    lines.append("────────────────")
    
    entry_price = az_plan.get("entry_price")
    curr_price = az_plan.get("current_price", item.get("price"))
    change = item.get("change")
    entry_text = _format_price_value(entry_price)
    curr_text = _format_price_value(curr_price)
    
    if entry_text:
        lines.append(f"<b>📍 จุดเข้า:</b> {entry_text}")
    if curr_text:
        change_str = f" ({change:+.2f}%)" if isinstance(change, (int, float)) else ""
        lines.append(f"<b>📍 ราคาปัจจุบัน:</b> {curr_text}{change_str}")
        
    zone = az_plan.get("zone")
    trend = az_plan.get("trend_1h")
    if zone or trend:
        parts = []
        if zone:
            parts.append(f"โซน: {zone}")
        if trend:
            parts.append(f"เทรนด์ 1H: {trend}")
        lines.append("<b>🧭 สภาวะตลาด:</b> " + " • ".join([_html_escape(p) for p in parts]))
        
    conf = az_plan.get("confidence")
    if isinstance(conf, (int, float)):
        lines.append(f"<b>📊 โอกาสสำเร็จ:</b> {float(conf):.0f}%")
    best = {}
    optimizer = az_plan.get("optimizer")
    if isinstance(optimizer, dict) and isinstance(optimizer.get("best"), dict):
        best = optimizer.get("best") or {}
    lines.extend(
        _build_alert_profile_lines(
            win_rate=best.get("win_rate_pct"),
            confidence=conf,
            expectancy=best.get("expectancy_rr"),
            trades=best.get("trades"),
        )
    )
    
    fast_len = az_plan.get("fast_len")
    slow_len = az_plan.get("slow_len")
    if fast_len and slow_len:
        lines.append(f"<b>⚙️ ค่าเฉลี่ย EMA:</b> {fast_len}/{slow_len}")
    
    pattern = az_plan.get("detected_pattern")
    _append_pattern_context_lines(lines, pattern)
    
    avg_range = az_plan.get("avg_range_pct")
    if isinstance(avg_range, (int, float)):
        lines.append(f"<b>📈 ความผันผวน (20 แท่ง):</b> {avg_range:.2f}%")
        
    if isinstance(best, dict):
        trades = best.get("trades")
        exp_rr = best.get("expectancy_rr")
        win_rate = best.get("win_rate_pct")
        stats = []
        if isinstance(trades, (int, float)):
            stats.append(f"ย้อนหลัง {int(trades)} ไม้")
        if isinstance(win_rate, (int, float)):
            stats.append(f"Win {float(win_rate):.0f}%")
        if isinstance(exp_rr, (int, float)):
            stats.append(f"ExpRR {float(exp_rr):.2f}")
        if stats:
            lines.append("<b>🧪 สถิติ Backtest:</b> " + " | ".join([_html_escape(s) for s in stats]))
        wf_stats = []
        valid_trades = best.get("valid_trades")
        valid_win = best.get("valid_win_rate_pct")
        valid_exp = best.get("valid_expectancy_rr")
        if isinstance(valid_trades, (int, float)):
            wf_stats.append(f"Valid {int(valid_trades)} ไม้")
        if isinstance(valid_win, (int, float)):
            wf_stats.append(f"WF Win {float(valid_win):.0f}%")
        if isinstance(valid_exp, (int, float)):
            wf_stats.append(f"WF ExpRR {float(valid_exp):.2f}")
        if wf_stats:
            lines.append("<b>🧭 Walk-forward:</b> " + " | ".join([_html_escape(s) for s in wf_stats]))
    if signal == "SELL":
        sell_optimizer = az_plan.get("sell_optimizer")
        sell_best = sell_optimizer.get("best") if isinstance(sell_optimizer, dict) else None
        if isinstance(sell_best, dict):
            sell_stats = []
            sell_valid_trades = sell_best.get("valid_trades")
            sell_valid_win = sell_best.get("valid_win_rate_pct")
            sell_valid_exp = sell_best.get("valid_expectancy_rr")
            if isinstance(sell_valid_trades, (int, float)):
                sell_stats.append(f"Valid {int(sell_valid_trades)} ไม้")
            if isinstance(sell_valid_win, (int, float)):
                sell_stats.append(f"Sell WF Win {float(sell_valid_win):.0f}%")
            if isinstance(sell_valid_exp, (int, float)):
                sell_stats.append(f"Sell WF ExpRR {float(sell_valid_exp):.2f}")
            if sell_stats:
                lines.append("<b>📉 Sell Whitelist:</b> " + " | ".join([_html_escape(s) for s in sell_stats]))
        sell_reason = str(az_plan.get("sell_whitelist_reason") or "").strip()
        if sell_reason:
            lines.append(f"<b>⚠️ Sell Gate:</b> {_html_escape(sell_reason)}")
            
    stop_lines = _build_stop_context_lines(item, az_plan, signal=signal, source_label="ActionZone 15m")
    if stop_lines:
        lines.append("────────────────")
        lines.extend(stop_lines)
        
    exit_lines = _format_exit_levels_lines(az_plan)
    if exit_lines:
        if not stop_lines:
            lines.append("────────────────")
        lines.extend([_html_escape(line) for line in exit_lines])
        
    forecast_lines = _format_price_forecast_lines(item.get("price_forecast"))
    if forecast_lines:
        lines.append("────────────────")
        lines.extend([_html_escape(line) for line in forecast_lines])
        
    lines.append("────────────────")
    last_signal_time = az_plan.get("last_signal_time")
    if isinstance(last_signal_time, str) and last_signal_time:
        lines.append(f"🕒 <b>สัญญาณล่าสุด:</b> {_html_escape(last_signal_time)}")
    lines.append("⏱️ <b>เวลา:</b> " + get_thai_now().strftime("%Y-%m-%d %H:%M"))
    lines.append(f"<a href=\"https://th.tradingview.com/chart/?symbol=CRYPTO:{tv_symbol}\">📈 ดูชาร์ตบน TradingView</a>")
    
    return "\n".join(lines)


def _build_cdc_vixfix_message(item, plan, mode_label=None):
    signal = str(plan.get("signal") or "").upper()
    if signal not in ("BUY", "SELL"):
        return None
    emoji = "🟢" if signal == "BUY" else "🔴"
    symbol = normalize_symbol(item.get("symbol") or "")
    name = _html_escape(str(item.get("name") or "").strip())
    tv_symbol = symbol.replace("-", "")
    
    lines = [f"<b>{emoji} CDC+VixFix 15m {signal} | {_html_escape(symbol)}</b>"]
    if name:
        lines.append(f"<i>{name}</i>")
    if mode_label:
        lines.append("<b>🧭 โหมด:</b> " + _html_escape(str(mode_label)))
    lines.append("────────────────")
    
    entry_text = _format_price_value(plan.get("entry_price"))
    curr_text = _format_price_value(plan.get("current_price", item.get("price")))
    stop_text = _format_price_value(plan.get("stop_loss"))
    change = item.get("change")
    move_parts = []
    if entry_text:
        move_parts.append(f"จุดอ้างอิง: {entry_text}")
    if curr_text:
        move_parts.append(f"ราคาปัจจุบัน: {curr_text}")
    if stop_text:
        move_parts.append(f"SL: {stop_text}")
    if isinstance(change, (int, float)):
        move_parts.append(f"เปลี่ยนแปลง: {change:+.2f}%")
    if move_parts:
        lines.append("<b>📍 " + " | ".join([_html_escape(m) for m in move_parts]) + "</b>")

    conf = _normalize_confidence(plan.get("confidence"))
    if conf is not None:
        lines.append(f"<b>📊 ความมั่นใจ:</b> {conf:.0f}%")
    edge = _extract_signal_edge_metrics(plan, signal)
    lines.extend(
        _build_alert_profile_lines(
            win_rate=edge.get("win_rate_pct"),
            confidence=conf,
            expectancy=edge.get("expectancy_rr"),
            trades=edge.get("trades"),
            mode_label=mode_label,
        )
    )

    forecast_dir = str(plan.get("forecast_direction") or "").strip().upper()
    forecast_score = plan.get("forecast_score")
    if forecast_dir:
        forecast_line = f"<b>🧭 คาดทิศทาง:</b> {_html_escape(forecast_dir)}"
        if isinstance(forecast_score, (int, float)):
            forecast_line += f" ({float(forecast_score):.0f}/100)"
        lines.append(forecast_line)

    trend_bias = str(plan.get("trend_bias") or "").strip()
    if trend_bias:
        lines.append("<b>📐 โครงสร้าง:</b> " + _html_escape(trend_bias))

    reason = plan.get("reason")
    if isinstance(reason, str) and reason.strip():
        lines.append("<b>🧠 มุมมองหลัก:</b> " + _html_escape(reason.strip()))

    sell_trigger = str(plan.get("sell_trigger") or plan.get("exit_trigger") or "").strip().upper()
    if sell_trigger:
        trigger_text = sell_trigger
        if sell_trigger == "CDC_RED_REVERSAL":
            trigger_text = "CDC_RED_REVERSAL (เทรนด์กลับตัว)"
        elif sell_trigger == "TREND_ROLLOVER":
            trigger_text = "TREND_ROLLOVER (โมเมนตัมอ่อนแรงต่อเนื่อง)"
        elif sell_trigger == "TAKE_PROFIT":
            trigger_text = "TAKE_PROFIT (แตะเป้าปิดทำกำไร)"
        elif sell_trigger == "TIME_STOP":
            trigger_text = "TIME_STOP (ถือครบอายุสัญญาณ)"
        lines.append("<b>🚨 Trigger:</b> " + _html_escape(trigger_text))

    stop_lines = _build_stop_context_lines(item, plan, signal=signal, source_label="CDC+VixFix 15m")
    if stop_lines:
        lines.append("────────────────")
        lines.extend(stop_lines)

    exit_lines = _format_exit_levels_lines(plan)
    if exit_lines:
        if not stop_lines:
            lines.append("────────────────")
        lines.extend([_html_escape(line) for line in exit_lines])

    analysis_points = plan.get("analysis_points")
    if isinstance(analysis_points, list):
        analysis_lines = []
        for point in analysis_points:
            text = str(point).strip()
            if text:
                analysis_lines.append("• " + _html_escape(text))
        if analysis_lines:
            lines.append("────────────────")
            lines.append("<b>🔎 เหตุผลเชิงวิเคราะห์:</b>")
            lines.extend(analysis_lines[:6])
        
    pattern = plan.get("detected_pattern")
    _append_pattern_context_lines(lines, pattern)

    lines.append("────────────────")
    last_signal_time = plan.get("last_signal_time")
    if isinstance(last_signal_time, str) and last_signal_time:
        lines.append(f"🕒 <b>สัญญาณล่าสุด:</b> {_html_escape(last_signal_time)}")

    lines.append("⏱️ <b>เวลา:</b> " + get_thai_now().strftime("%Y-%m-%d %H:%M"))
    lines.append(f"<a href=\"https://th.tradingview.com/chart/?symbol=CRYPTO:{tv_symbol}\">📈 ดูชาร์ตบน TradingView</a>")

    return "\n".join(lines)


def _normalize_trade_direction(value):
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if "BUY" in text or "LONG" in text or text == "UP":
        return "BUY"
    if "SELL" in text or "SHORT" in text or text == "DOWN":
        return "SELL"
    return None


def _pick_price_forecast_candidate(*plans):
    best = None
    best_conf = -1.0
    for label, plan in plans:
        if not isinstance(plan, dict):
            continue
        direction = _normalize_trade_direction(plan.get("signal") or plan.get("setup") or plan.get("recommendation"))
        if direction not in ("BUY", "SELL"):
            continue
        conf = _normalize_confidence(plan.get("confidence"))
        if conf is None:
            conf = _normalize_confidence(plan.get("predicted_win_prob"))
        if conf is None:
            conf = _normalize_confidence(plan.get("win_prob"))
        if conf is None:
            conf = 50.0
        if conf > best_conf:
            best_conf = conf
            best = {
                "label": label,
                "plan": plan,
                "direction": direction,
                "confidence": float(conf),
            }
    return best


def _format_price_forecast_lines(forecast):
    if not isinstance(forecast, dict):
        return []
    direction = str(forecast.get("direction") or "").upper()
    horizon = forecast.get("horizon_hours")
    base_price = _format_price_value(forecast.get("base_price"))
    low_price = _format_price_value(forecast.get("range_low"))
    high_price = _format_price_value(forecast.get("range_high"))
    invalidation = _format_price_value(forecast.get("invalidation_price"))
    source = str(forecast.get("source") or "").strip()
    bias = str(forecast.get("decision_bias") or "").strip()
    lines = []
    parts = []
    if direction:
        parts.append(direction)
    if isinstance(horizon, (int, float)) and horizon > 0:
        parts.append(f"{float(horizon):.1f}h")
    if source:
        parts.append(source)
    header = "🔮 คาดการณ์ราคา"
    if parts:
        header += ": " + " | ".join(parts)
    lines.append(header)
    detail = []
    if base_price:
        detail.append(f"ฐาน {base_price}")
    if low_price and high_price:
        detail.append(f"กรอบ {low_price} - {high_price}")
    elif high_price:
        detail.append(f"เป้า {high_price}")
    if detail:
        lines.append("📐 " + " | ".join(detail))
    if invalidation:
        lines.append("🚫 จุดยกเลิกมุมมอง: " + invalidation)
    if bias:
        lines.append("🧠 มุมมอง: " + bias)
    return lines


def _telegram_kill_switch_state(results):
    enabled = bool(getattr(config, "TELEGRAM_KILL_SWITCH_ENABLED", True))
    if not enabled:
        return False, "disabled"
    if not isinstance(results, list) or not results:
        return False, "no-results"
    min_symbols = getattr(config, "TELEGRAM_KILL_SWITCH_MIN_SYMBOLS", 4)
    sell_ratio_limit = getattr(config, "TELEGRAM_KILL_SWITCH_SELL_RATIO", 0.50)
    trend_mismatch_ratio_limit = getattr(config, "TELEGRAM_KILL_SWITCH_TREND_MISMATCH_RATIO", 0.60)
    high_vol_pct = getattr(config, "TELEGRAM_KILL_SWITCH_HIGH_VOL_PCT", 4.0)
    high_vol_ratio_limit = getattr(config, "TELEGRAM_KILL_SWITCH_HIGH_VOL_RATIO", 0.50)
    try:
        min_symbols = int(min_symbols)
    except Exception:
        min_symbols = 4
    try:
        sell_ratio_limit = float(sell_ratio_limit)
    except Exception:
        sell_ratio_limit = 0.50
    try:
        trend_mismatch_ratio_limit = float(trend_mismatch_ratio_limit)
    except Exception:
        trend_mismatch_ratio_limit = 0.60
    try:
        high_vol_pct = float(high_vol_pct)
    except Exception:
        high_vol_pct = 4.0
    try:
        high_vol_ratio_limit = float(high_vol_ratio_limit)
    except Exception:
        high_vol_ratio_limit = 0.50
    if min_symbols < 1:
        min_symbols = 1
    sell_count = 0
    valid_count = 0
    az_count = 0
    trend_mismatch_count = 0
    vol_count = 0
    high_vol_count = 0
    for item in results:
        if not isinstance(item, dict) or item.get("error"):
            continue
        valid_count += 1
        cdc_plan = item.get("cdc_vixfix_15m")
        if isinstance(cdc_plan, dict):
            cdc_signal = str(cdc_plan.get("signal") or "").upper()
            is_top = bool(cdc_plan.get("is_market_top"))
            if cdc_signal == "SELL" or is_top:
                sell_count += 1
        az_plan = item.get("actionzone_15m")
        if isinstance(az_plan, dict):
            az_count += 1
            if not bool(az_plan.get("trend_alignment", True)):
                trend_mismatch_count += 1
            avg_range_pct = az_plan.get("avg_range_pct")
            if isinstance(avg_range_pct, (int, float)):
                vol_count += 1
                if float(avg_range_pct) >= high_vol_pct:
                    high_vol_count += 1
    if valid_count < min_symbols:
        return False, "insufficient-sample"
    sell_ratio = float(sell_count) / float(valid_count) if valid_count > 0 else 0.0
    trend_mismatch_ratio = float(trend_mismatch_count) / float(az_count) if az_count > 0 else 0.0
    high_vol_ratio = float(high_vol_count) / float(vol_count) if vol_count > 0 else 0.0
    if sell_ratio >= sell_ratio_limit:
        return True, f"sell_ratio={sell_ratio:.2f}"
    if az_count >= min_symbols and trend_mismatch_ratio >= trend_mismatch_ratio_limit:
        return True, f"trend_mismatch_ratio={trend_mismatch_ratio:.2f}"
    if vol_count >= min_symbols and high_vol_ratio >= high_vol_ratio_limit:
        return True, f"high_vol_ratio={high_vol_ratio:.2f}"
    return False, "ok"


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _telegram_dynamic_conf_threshold(base_min_conf, results):
    dynamic_enable = bool(getattr(config, "TELEGRAM_ALERT_DYNAMIC_CONF_ENABLE", True))
    if not dynamic_enable:
        return float(base_min_conf)
    vol_ref = _safe_float(getattr(config, "TELEGRAM_ALERT_DYNAMIC_CONF_VOL_REF_PCT", 1.80), 1.80)
    vol_mult = _safe_float(getattr(config, "TELEGRAM_ALERT_DYNAMIC_CONF_VOL_MULT", 2.50), 2.50)
    if vol_ref <= 0:
        vol_ref = 1.80
    vol_samples = []
    for item in results or []:
        if not isinstance(item, dict):
            continue
        az_plan = item.get("actionzone_15m")
        if not isinstance(az_plan, dict):
            continue
        avg_range_pct = _safe_float(az_plan.get("avg_range_pct"), None)
        if isinstance(avg_range_pct, float) and math.isfinite(avg_range_pct) and avg_range_pct > 0:
            vol_samples.append(avg_range_pct)
    if not vol_samples:
        return float(base_min_conf)
    vol_samples.sort()
    median_vol = vol_samples[len(vol_samples) // 2]
    adjusted = float(base_min_conf) + ((float(median_vol) - float(vol_ref)) * float(vol_mult))
    return max(55.0, min(95.0, adjusted))


def _actionzone_precision60_profile():
    enabled = bool(getattr(config, "ACTIONZONE_15M_PRECISION60_ENABLED", False))
    if not enabled:
        return None
    min_conf = getattr(config, "ACTIONZONE_15M_PRECISION60_MIN_ALERT_CONFIDENCE", 90.0)
    min_rvol = getattr(config, "ACTIONZONE_15M_PRECISION60_MIN_RVOL", 1.30)
    min_ema_gap = getattr(config, "ACTIONZONE_15M_PRECISION60_MIN_EMA_GAP_PCT", 0.12)
    min_adx = getattr(config, "ACTIONZONE_15M_PRECISION60_MIN_ADX", 30.0)
    max_bars_since = getattr(config, "ACTIONZONE_15M_PRECISION60_MAX_BARS_SINCE_SIGNAL", 0)
    require_ema200 = bool(getattr(config, "ACTIONZONE_15M_PRECISION60_REQUIRE_EMA200_ALIGNMENT", True))
    require_strong = bool(getattr(config, "ACTIONZONE_15M_PRECISION60_REQUIRE_STRONG_TREND", True))
    require_trend_alignment = bool(getattr(config, "ACTIONZONE_15M_PRECISION60_REQUIRE_TREND_ALIGNMENT", True))
    try:
        min_conf = float(min_conf)
    except Exception:
        min_conf = 90.0
    try:
        min_rvol = float(min_rvol)
    except Exception:
        min_rvol = 1.30
    try:
        min_ema_gap = float(min_ema_gap)
    except Exception:
        min_ema_gap = 0.12
    try:
        min_adx = float(min_adx)
    except Exception:
        min_adx = 30.0
    try:
        max_bars_since = int(max_bars_since)
    except Exception:
        max_bars_since = 0
    if max_bars_since < 0:
        max_bars_since = 0
    return {
        "enabled": True,
        "min_conf": float(min_conf),
        "min_rvol": float(min_rvol),
        "min_ema_gap_pct": float(min_ema_gap),
        "min_adx": float(min_adx),
        "max_bars_since_signal": int(max_bars_since),
        "require_ema200_alignment": bool(require_ema200),
        "require_strong_trend": bool(require_strong),
        "require_trend_alignment": bool(require_trend_alignment),
    }


def _build_alert_backtest_summary(results, min_conf=None):
    if not isinstance(results, list) or not results:
        return {
            "candidate_count": 0,
            "measured_count": 0,
            "estimated_win_rate_pct": None,
            "average_expectancy_rr": None,
            "average_confidence": None,
            "total_historical_trades": 0,
            "dynamic_min_confidence": None,
            "alerts": [],
            "by_strategy": {name: {"alerts": 0, "measured": 0, "estimated_win_rate_pct": None, "average_expectancy_rr": None, "total_historical_trades": 0, "observed_symbols": 0, "signals": {}} for name in _TELEGRAM_REPORT_STRATEGY_ORDER},
        }
    if min_conf is None:
        min_conf = getattr(config, "TELEGRAM_ALERT_MIN_CONFIDENCE", 75.0)
    try:
        min_conf = float(min_conf)
    except Exception:
        min_conf = 75.0
    dynamic_min_conf = _telegram_dynamic_conf_threshold(min_conf, results)
    candidates, _ = _build_telegram_candidates(results, dynamic_min_conf)
    raw_strategy_summary = _build_strategy_backtest_summary(results, min_conf=dynamic_min_conf)
    alerts = []
    weighted_wr_sum = 0.0
    weighted_exp_sum = 0.0
    total_weight = 0.0
    confidence_sum = 0.0
    confidence_count = 0
    by_strategy = {}
    measured_count = 0
    for candidate in candidates:
        strategy = str(candidate.get("strategy") or "").strip() or "UNKNOWN"
        confidence = candidate.get("confidence")
        plan = candidate.get("plan")
        edge = candidate.get("edge_metrics")
        if not isinstance(edge, dict):
            edge = _extract_plan_edge_metrics(plan)
        wr = edge.get("win_rate_pct")
        exp = edge.get("expectancy_rr")
        trades = edge.get("trades")
        alert_entry = {
            "symbol": str(candidate.get("symbol") or ""),
            "strategy": strategy,
            "signal": str(candidate.get("signal") or ""),
            "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
            "historical_trades": int(trades) if isinstance(trades, (int, float)) else None,
            "historical_win_rate_pct": float(wr) if isinstance(wr, (int, float)) else None,
            "expectancy_rr": float(exp) if isinstance(exp, (int, float)) else None,
        }
        alerts.append(alert_entry)
        bucket = by_strategy.setdefault(
            strategy,
            {
                "alerts": 0,
                "measured": 0,
                "weighted_wr_sum": 0.0,
                "weighted_exp_sum": 0.0,
                "weight": 0.0,
            },
        )
        bucket["alerts"] += 1
        if isinstance(confidence, (int, float)):
            confidence_sum += float(confidence)
            confidence_count += 1
        if isinstance(trades, (int, float)) and float(trades) > 0 and (
            isinstance(wr, (int, float)) or isinstance(exp, (int, float))
        ):
            weight = float(trades)
            measured_count += 1
            total_weight += weight
            bucket["measured"] += 1
            bucket["weight"] += weight
            if isinstance(wr, (int, float)):
                weighted_wr_sum += float(wr) * weight
                bucket["weighted_wr_sum"] += float(wr) * weight
            if isinstance(exp, (int, float)):
                weighted_exp_sum += float(exp) * weight
                bucket["weighted_exp_sum"] += float(exp) * weight
    by_strategy_summary = dict(raw_strategy_summary)
    for strategy, bucket in by_strategy.items():
        weight = float(bucket.get("weight") or 0.0)
        row = dict(by_strategy_summary.get(strategy) or {})
        row["alerts"] = int(bucket.get("alerts") or 0)
        candidate_measured = int(bucket.get("measured") or 0)
        if candidate_measured > 0:
            row["measured"] = candidate_measured
            row["estimated_win_rate_pct"] = (bucket["weighted_wr_sum"] / weight) if weight > 0 else None
            row["average_expectancy_rr"] = (bucket["weighted_exp_sum"] / weight) if weight > 0 else None
            row["total_historical_trades"] = int(weight) if weight > 0 else int(row.get("total_historical_trades") or 0)
        else:
            row.setdefault("measured", 0)
            row.setdefault("estimated_win_rate_pct", None)
            row.setdefault("average_expectancy_rr", None)
            row.setdefault("total_historical_trades", 0)
        row.setdefault("observed_symbols", 0)
        row.setdefault("signals", {})
        by_strategy_summary[strategy] = row
    return {
        "candidate_count": len(candidates),
        "measured_count": measured_count,
        "estimated_win_rate_pct": (weighted_wr_sum / total_weight) if total_weight > 0 else None,
        "average_expectancy_rr": (weighted_exp_sum / total_weight) if total_weight > 0 else None,
        "average_confidence": (confidence_sum / confidence_count) if confidence_count > 0 else None,
        "total_historical_trades": int(total_weight) if total_weight > 0 else 0,
        "dynamic_min_confidence": float(dynamic_min_conf),
        "alerts": alerts,
        "by_strategy": by_strategy_summary,
    }


def _build_all_weather_summary(results, min_conf=None):
    returned = results if isinstance(results, list) else []
    if not returned:
        return {
            "generated_at": get_thai_now().strftime("%Y-%m-%d %H:%M"),
            "required_confidence": None,
            "dynamic_telegram_min_confidence": None,
            "valid_symbols": 0,
            "error_symbols": 0,
            "ready_symbols": 0,
            "readiness_rate_pct": 0.0,
            "deploy_ready": False,
            "by_status": {},
            "by_regime": {},
            "by_signal": {},
            "by_primary_source": {},
            "average_final_confidence": None,
            "per_symbol": [],
        }
    telegram_min_conf = getattr(config, "TELEGRAM_ALERT_MIN_CONFIDENCE", 72.0)
    try:
        telegram_min_conf = float(telegram_min_conf)
    except Exception:
        telegram_min_conf = 72.0
    dynamic_telegram_min = _telegram_dynamic_conf_threshold(telegram_min_conf, returned)
    aw_floor = getattr(config, "ALL_WEATHER_15M_MIN_ALERT_CONFIDENCE", 69.0)
    try:
        aw_floor = float(aw_floor)
    except Exception:
        aw_floor = 69.0
    if min_conf is None:
        min_conf = max(float(dynamic_telegram_min), float(aw_floor))
    else:
        try:
            min_conf = float(min_conf)
        except Exception:
            min_conf = max(float(dynamic_telegram_min), float(aw_floor))
    min_symbol_ready_rate = getattr(config, "ALL_WEATHER_15M_MIN_SYMBOL_READY_RATE", 35.0)
    try:
        min_symbol_ready_rate = float(min_symbol_ready_rate)
    except Exception:
        min_symbol_ready_rate = 35.0

    by_status = Counter()
    by_regime = Counter()
    by_signal = Counter()
    by_primary_source = Counter()
    per_symbol = []
    valid_symbols = 0
    error_symbols = 0
    ready_symbols = 0
    conf_sum = 0.0
    conf_count = 0
    for item in returned:
        if not isinstance(item, dict):
            continue
        symbol = normalize_symbol(item.get("symbol") or "")
        if item.get("error"):
            error_symbols += 1
            per_symbol.append(
                {
                    "symbol": symbol,
                    "name": str(item.get("name") or "").strip(),
                    "status": "error",
                    "error": str(item.get("error") or "").strip(),
                    "signal": "WAIT",
                    "passed": False,
                }
            )
            continue
        valid_symbols += 1
        report = _build_all_weather_report_entry(item, min_conf=min_conf)
        status = str(report.get("status") or "unknown")
        regime = str(report.get("regime") or "UNKNOWN")
        signal = str(report.get("signal") or "WAIT")
        by_status[status] += 1
        by_regime[regime] += 1
        by_signal[signal] += 1
        primary_source = str(report.get("primary_source") or "").strip()
        if primary_source:
            by_primary_source[primary_source] += 1
        final_conf = report.get("final_confidence")
        if isinstance(final_conf, (int, float)) and math.isfinite(float(final_conf)):
            conf_sum += float(final_conf)
            conf_count += 1
        if bool(report.get("passed")):
            ready_symbols += 1
        blend = report.get("blend") or {}
        per_symbol.append(
            {
                "symbol": symbol,
                "name": str(item.get("name") or "").strip(),
                "price": float(item.get("price")) if isinstance(item.get("price"), (int, float)) else None,
                "change": float(item.get("change")) if isinstance(item.get("change"), (int, float)) else None,
                "status": status,
                "signal": signal,
                "passed": bool(report.get("passed")),
                "regime": report.get("regime"),
                "primary_source": report.get("primary_source"),
                "selected_sources": report.get("selected_sources") or [],
                "final_confidence": float(final_conf) if isinstance(final_conf, (int, float)) else None,
                "required_confidence": float(report.get("required_confidence")) if isinstance(report.get("required_confidence"), (int, float)) else None,
                "top_buy_score": float(report.get("top_buy_score")) if isinstance(report.get("top_buy_score"), (int, float)) else None,
                "top_sell_score": float(report.get("top_sell_score")) if isinstance(report.get("top_sell_score"), (int, float)) else None,
                "side_gap": float(report.get("side_gap")) if isinstance(report.get("side_gap"), (int, float)) else None,
                "volatility_pct": float(report.get("volatility_pct")) if isinstance(report.get("volatility_pct"), (int, float)) else None,
                "trend_score": float(report.get("trend_score")) if isinstance(report.get("trend_score"), (int, float)) else None,
                "candidate_pool_count": int(report.get("candidate_pool_count") or 0),
                "blend": {
                    "win_rate_pct": float(blend.get("win_rate_pct")) if isinstance(blend.get("win_rate_pct"), (int, float)) else None,
                    "expectancy_rr": float(blend.get("expectancy_rr")) if isinstance(blend.get("expectancy_rr"), (int, float)) else None,
                    "trades": int(blend.get("trades")) if isinstance(blend.get("trades"), (int, float)) else 0,
                    "measured_count": int(blend.get("measured_count") or 0),
                },
                "candidates": report.get("candidates") or [],
            }
        )
    per_symbol.sort(
        key=lambda row: (
            1 if bool(row.get("passed")) else 0,
            float(row.get("final_confidence") or 0.0),
            float(row.get("side_gap") or 0.0),
            str(row.get("symbol") or ""),
        ),
        reverse=True,
    )
    readiness_rate = (float(ready_symbols) / float(valid_symbols) * 100.0) if valid_symbols > 0 else 0.0
    return {
        "generated_at": get_thai_now().strftime("%Y-%m-%d %H:%M"),
        "required_confidence": float(min_conf),
        "dynamic_telegram_min_confidence": float(dynamic_telegram_min),
        "valid_symbols": int(valid_symbols),
        "error_symbols": int(error_symbols),
        "ready_symbols": int(ready_symbols),
        "readiness_rate_pct": float(readiness_rate),
        "min_symbol_ready_rate_pct": float(min_symbol_ready_rate),
        "deploy_ready": bool(readiness_rate >= float(min_symbol_ready_rate)),
        "by_status": dict(by_status),
        "by_regime": dict(by_regime),
        "by_signal": dict(by_signal),
        "by_primary_source": dict(by_primary_source),
        "average_final_confidence": (conf_sum / conf_count) if conf_count > 0 else None,
        "per_symbol": per_symbol,
    }


def _build_all_weather_readiness(period_reports):
    reports = period_reports if isinstance(period_reports, dict) else {}
    min_period_passes = getattr(config, "ALL_WEATHER_15M_MIN_PERIOD_PASSES", 2)
    min_symbol_ready_rate = getattr(config, "ALL_WEATHER_15M_MIN_SYMBOL_READY_RATE", 35.0)
    try:
        min_period_passes = max(1, int(min_period_passes))
    except Exception:
        min_period_passes = 2
    try:
        min_symbol_ready_rate = float(min_symbol_ready_rate)
    except Exception:
        min_symbol_ready_rate = 35.0
    symbol_map = {}
    period_count = 0
    for period, report in reports.items():
        if not isinstance(report, dict):
            continue
        period_count += 1
        for row in report.get("per_symbol") or []:
            if not isinstance(row, dict):
                continue
            symbol = normalize_symbol(row.get("symbol") or "")
            if not symbol:
                continue
            bucket = symbol_map.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "name": str(row.get("name") or "").strip(),
                    "pass_count": 0,
                    "periods_passed": [],
                    "signals_by_period": {},
                    "status_by_period": {},
                    "best_confidence": None,
                },
            )
            passed = bool(row.get("passed"))
            bucket["signals_by_period"][period] = str(row.get("signal") or "WAIT")
            bucket["status_by_period"][period] = str(row.get("status") or "unknown")
            conf = row.get("final_confidence")
            if isinstance(conf, (int, float)) and (
                bucket["best_confidence"] is None or float(conf) > float(bucket["best_confidence"])
            ):
                bucket["best_confidence"] = float(conf)
            if passed:
                bucket["pass_count"] += 1
                bucket["periods_passed"].append(period)
    per_symbol = []
    ready_symbols = 0
    for symbol, bucket in symbol_map.items():
        ready = int(bucket["pass_count"]) >= int(min_period_passes)
        if ready:
            ready_symbols += 1
        per_symbol.append(
            {
                "symbol": symbol,
                "name": bucket.get("name"),
                "pass_count": int(bucket.get("pass_count") or 0),
                "periods_passed": bucket.get("periods_passed") or [],
                "signals_by_period": bucket.get("signals_by_period") or {},
                "status_by_period": bucket.get("status_by_period") or {},
                "best_confidence": float(bucket.get("best_confidence")) if isinstance(bucket.get("best_confidence"), (int, float)) else None,
                "ready": bool(ready),
            }
        )
    per_symbol.sort(
        key=lambda row: (
            1 if bool(row.get("ready")) else 0,
            int(row.get("pass_count") or 0),
            float(row.get("best_confidence") or 0.0),
            str(row.get("symbol") or ""),
        ),
        reverse=True,
    )
    total_symbols = len(per_symbol)
    readiness_rate = (float(ready_symbols) / float(total_symbols) * 100.0) if total_symbols > 0 else 0.0
    return {
        "evaluated_periods": int(period_count),
        "min_period_passes": int(min_period_passes),
        "min_symbol_ready_rate_pct": float(min_symbol_ready_rate),
        "ready_symbols": int(ready_symbols),
        "total_symbols": int(total_symbols),
        "readiness_rate_pct": float(readiness_rate),
        "deploy_ready": bool(readiness_rate >= float(min_symbol_ready_rate)),
        "per_symbol": per_symbol,
    }


def _weighted_average_from_rows(rows, value_key, weight_key, digits=None):
    weighted_sum = 0.0
    total_weight = 0.0
    for row in rows or []:
        value = row.get(value_key) if isinstance(row, dict) else None
        weight = row.get(weight_key) if isinstance(row, dict) else None
        if not isinstance(value, (int, float)):
            continue
        if not isinstance(weight, (int, float)) or float(weight) <= 0:
            weight = 1.0
        weighted_sum += float(value) * float(weight)
        total_weight += float(weight)
    if total_weight <= 0:
        return None
    result = weighted_sum / total_weight
    if isinstance(digits, int):
        return round(result, digits)
    return result


def _round_rule_value(value, digits=2):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return round(float(value), int(digits))


def _build_actionzone_rule_profile(results, direction):
    direction = str(direction or "").upper().strip()
    if direction not in ("BUY", "SELL"):
        return None
    optimizer_key = "sell_optimizer" if direction == "SELL" else "optimizer"
    rows = []
    passed_rows = []
    blocked_rows = []
    for item in results or []:
        if not isinstance(item, dict) or item.get("error"):
            continue
        symbol = normalize_symbol(item.get("symbol") or "")
        plan = item.get("actionzone_15m")
        if not symbol or not isinstance(plan, dict):
            continue
        optimizer = plan.get(optimizer_key)
        best = optimizer.get("best") if isinstance(optimizer, dict) else None
        if not isinstance(best, dict):
            continue
        if direction == "SELL":
            gate_ok, gate_reason, gate_metrics = _evaluate_sell_whitelist_gate({"sell_optimizer": optimizer})
        else:
            gate_ok, gate_reason, gate_metrics = _evaluate_walkforward_gate({"optimizer": optimizer}, direction)
        trades = gate_metrics.get("valid_trades")
        if not isinstance(trades, (int, float)) or float(trades) <= 0:
            trades = best.get("trades")
        if not isinstance(trades, (int, float)) or float(trades) <= 0:
            trades = 1.0
        row = {
            "symbol": symbol,
            "passed": bool(gate_ok),
            "reason": str(gate_reason or "unknown"),
            "fast_len": best.get("fast_len"),
            "slow_len": best.get("slow_len"),
            "trades": best.get("trades"),
            "win_rate_pct": best.get("win_rate_pct"),
            "expectancy_rr": best.get("expectancy_rr"),
            "valid_trades": gate_metrics.get("valid_trades"),
            "valid_win_rate_pct": gate_metrics.get("valid_win_rate_pct"),
            "valid_expectancy_rr": gate_metrics.get("valid_expectancy_rr"),
            "robustness_score": gate_metrics.get("robustness_score"),
            "weight": float(trades),
        }
        rows.append(row)
        if gate_ok:
            passed_rows.append(row)
        else:
            blocked_rows.append(
                {
                    "symbol": symbol,
                    "reason": str(gate_reason or "unknown"),
                    "valid_trades": row.get("valid_trades"),
                    "valid_win_rate_pct": row.get("valid_win_rate_pct"),
                    "valid_expectancy_rr": row.get("valid_expectancy_rr"),
                    "robustness_score": row.get("robustness_score"),
                }
            )
    source_rows = passed_rows if passed_rows else rows
    source_scope = "passed_symbols" if passed_rows else "all_profiles_fallback"
    consensus_fast = _weighted_average_from_rows(source_rows, "fast_len", "weight", digits=0)
    consensus_slow = _weighted_average_from_rows(source_rows, "slow_len", "weight", digits=0)
    consensus_wr = _weighted_average_from_rows(source_rows, "valid_win_rate_pct", "weight", digits=2)
    consensus_exp = _weighted_average_from_rows(source_rows, "valid_expectancy_rr", "weight", digits=3)
    consensus_robustness = _weighted_average_from_rows(source_rows, "robustness_score", "weight", digits=1)
    if consensus_wr is None:
        consensus_wr = _weighted_average_from_rows(source_rows, "win_rate_pct", "weight", digits=2)
    if consensus_exp is None:
        consensus_exp = _weighted_average_from_rows(source_rows, "expectancy_rr", "weight", digits=3)
    if direction == "SELL":
        min_wr = _round_rule_value(getattr(config, "TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_WIN_RATE", 60.0), 2)
        min_exp = _round_rule_value(getattr(config, "TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_EXPECTANCY_RR", 0.05), 3)
        min_valid = int(getattr(config, "TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_VALID_TRADES", 6))
        min_robust = _round_rule_value(getattr(config, "TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_ROBUSTNESS", 45.0), 1)
    else:
        min_wr = _round_rule_value(getattr(config, "TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_WIN_RATE", 60.0), 2)
        min_exp = _round_rule_value(getattr(config, "TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_EXPECTANCY_RR", 0.05), 3)
        min_valid = int(getattr(config, "TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_VALID_TRADES", 6))
        min_robust = _round_rule_value(getattr(config, "TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_ROBUSTNESS", 45.0), 1)
    filters = {
        "ema_fast_len": int(consensus_fast) if isinstance(consensus_fast, (int, float)) else None,
        "ema_slow_len": int(consensus_slow) if isinstance(consensus_slow, (int, float)) else None,
        "min_rvol": _round_rule_value(getattr(config, "ACTIONZONE_15M_MIN_RVOL", 1.15), 2),
        "min_ema_gap_pct": _round_rule_value(getattr(config, "ACTIONZONE_15M_MIN_EMA_GAP_PCT", 0.05), 3),
        "min_atr_pct": _round_rule_value(getattr(config, "ACTIONZONE_15M_MIN_ATR_PCT", 0.15), 3),
        "max_atr_pct": _round_rule_value(getattr(config, "ACTIONZONE_15M_MAX_ATR_PCT", 6.0), 2),
        "min_adx": _round_rule_value(getattr(config, "ACTIONZONE_15M_MIN_ADX", 24.0), 1),
        "require_ema200_alignment": bool(getattr(config, "ACTIONZONE_15M_REQUIRE_EMA200_ALIGNMENT", True)),
        "require_strong_trend": bool(getattr(config, "ACTIONZONE_15M_REQUIRE_STRONG_TREND", True)),
        "max_bars_since_signal": int(getattr(config, "ACTIONZONE_15M_MAX_BARS_SINCE_SIGNAL", 1)),
        "stop_atr_mult": _round_rule_value(getattr(config, "ACTIONZONE_15M_STOP_ATR_MULT", 1.8), 2),
        "tp_mult": _round_rule_value(getattr(config, "ACTIONZONE_15M_TP_MULT", 2.0), 2),
    }
    validation = {
        "min_valid_win_rate_pct": min_wr,
        "min_valid_expectancy_rr": min_exp,
        "min_valid_trades": min_valid,
        "min_robustness_score": min_robust,
        "consensus_valid_win_rate_pct": consensus_wr,
        "consensus_valid_expectancy_rr": consensus_exp,
        "consensus_robustness_score": consensus_robustness,
    }
    checklist = [
        f"EMA {filters['ema_fast_len']}/{filters['ema_slow_len']}" if filters.get("ema_fast_len") and filters.get("ema_slow_len") else "EMA ตาม profile ของ symbol",
        f"RVOL >= {filters['min_rvol']}" if filters.get("min_rvol") is not None else None,
        f"EMA gap >= {filters['min_ema_gap_pct']}%" if filters.get("min_ema_gap_pct") is not None else None,
        f"ATR% อยู่ในช่วง {filters['min_atr_pct']}% - {filters['max_atr_pct']}%" if filters.get("min_atr_pct") is not None and filters.get("max_atr_pct") is not None else None,
        f"ADX >= {filters['min_adx']}" if filters.get("min_adx") is not None else None,
        "ราคาอยู่ฝั่งเดียวกับ EMA200" if filters.get("require_ema200_alignment") else None,
        "เทรนด์ 1H ต้องสอดคล้อง" if filters.get("require_strong_trend") else None,
        f"สัญญาณใหม่ไม่เกิน {filters['max_bars_since_signal']} แท่ง" if filters.get("max_bars_since_signal") is not None else None,
        f"Walk-forward WR >= {validation['min_valid_win_rate_pct']}%" if validation.get("min_valid_win_rate_pct") is not None else None,
        f"Walk-forward ExpRR >= {validation['min_valid_expectancy_rr']}" if validation.get("min_valid_expectancy_rr") is not None else None,
        f"Valid trades >= {validation['min_valid_trades']}" if validation.get("min_valid_trades") is not None else None,
    ]
    if direction == "SELL":
        checklist.append("ผ่าน sell whitelist")
    checklist = [line for line in checklist if line]
    return {
        "strategy": "ActionZone 15m",
        "direction": direction,
        "generated_from": "backtest+walkforward",
        "profile_count": len(rows),
        "passed_count": len(passed_rows),
        "consensus_scope": source_scope,
        "use_for_alerts": bool(passed_rows),
        "filters": filters,
        "validation": validation,
        "checklist": checklist,
        "eligible_symbols": [row["symbol"] for row in passed_rows],
        "blocked_symbols": blocked_rows[:10],
        "symbol_profiles": rows,
    }


def _build_backtest_rulebook(results):
    return {
        "generated_at": get_thai_now().strftime("%Y-%m-%d %H:%M"),
        "strategies": {
            "actionzone_15m": {
                "buy": _build_actionzone_rule_profile(results, "BUY"),
                "sell": _build_actionzone_rule_profile(results, "SELL"),
            }
        },
        "notes": [
            "Rulebook นี้สรุปจาก optimizer และ walk-forward ที่ระบบคำนวณอยู่แล้ว",
            "เหมาะสำหรับใช้เป็น profile ตอนแจ้งเตือน โดยไม่ต้อง backtest ยาวทุกครั้ง",
        ],
    }


def _build_telegram_candidates(results, min_conf):
    return _alerts_pipeline_build_telegram_candidates(
        results,
        min_conf,
        config=config,
        helpers=_pipeline_module_helpers(),
        get_now=get_thai_now,
    )


def _pipeline_module_helpers():
    return {
        "actionzone_precision60_profile": _actionzone_precision60_profile,
        "strict_60_mode_enabled": _strict_60_mode_enabled,
        "strict_60_allow_cdc": _strict_60_allow_cdc,
        "normalize_symbol": normalize_symbol,
        "evaluate_super_signal": _evaluate_super_signal,
        "pick_primary_trade_plan": _pick_primary_trade_plan,
        "build_super_signal_message": _build_super_signal_message,
        "build_all_weather_signal": _build_all_weather_signal,
        "build_all_weather_message": _build_all_weather_message,
        "normalize_confidence": _normalize_confidence,
        "build_cdc_vixfix_message": _build_cdc_vixfix_message,
        "extract_plan_edge_metrics": _extract_plan_edge_metrics,
        "alert_profile_score_adjustment": _alert_profile_score_adjustment,
        "format_price_value": _format_price_value,
        "evaluate_entry_quality_gate": _evaluate_entry_quality_gate,
        "build_actionzone_message": _build_actionzone_message,
        "safe_float": _safe_float,
        "build_price_action_message": _build_price_action_message,
        "get_best_confidence": _get_best_confidence,
        "collect_alert_sources": _collect_alert_sources,
        "get_primary_plan_source_label": _get_primary_plan_source_label,
        "pick_plan_value": _pick_plan_value,
        "build_telegram_message": _build_telegram_message,
        "evaluate_candidate_backtest_gate": _evaluate_candidate_backtest_gate,
        "candidate_alert_profile": _candidate_alert_profile,
        "telegram_kill_switch_state": _telegram_kill_switch_state,
        "telegram_dynamic_conf_threshold": _telegram_dynamic_conf_threshold,
        "build_telegram_candidates": _build_telegram_candidates,
        "build_cdc_daily_trend_candidates": _build_cdc_daily_trend_candidates,
        "is_daily_best_pick_window": _is_daily_best_pick_window,
        "build_daily_best_pick_candidates": _build_daily_best_pick_candidates,
        "build_daily_summary_message": _build_daily_summary_message,
        "send_telegram_alert": send_telegram_alert,
        "telegram_alert_cache": _TELEGRAM_ALERT_CACHE,
        "record_telegram_alert_history": _record_telegram_alert_history,
        "track_alert_performance": _track_alert_performance,
        "record_telegram_run_report": _record_telegram_run_report,
    }


def _daily_alert_module_helpers():
    return {
        "normalize_symbol": normalize_symbol,
        "pick_primary_trade_plan": _pick_primary_trade_plan,
        "get_best_confidence": _get_best_confidence,
        "plan_confidence_value": _plan_confidence_value,
        "collect_alert_sources": _collect_alert_sources,
        "extract_signal_edge_metrics": _extract_signal_edge_metrics,
        "get_primary_plan_source_label": _get_primary_plan_source_label,
        "safe_float": _safe_float,
        "pick_plan_value": _pick_plan_value,
        "build_daily_best_pick_message": _build_daily_best_pick_message,
        "evaluate_candidate_backtest_gate": _evaluate_candidate_backtest_gate,
        "candidate_edge_metrics": _candidate_edge_metrics,
        "plan_trade_direction": _plan_trade_direction,
        "build_cdc_vixfix_message": _build_cdc_vixfix_message,
    }


def _is_daily_best_pick_window():
    return _alerts_is_daily_best_pick_window(config=config, get_now=get_thai_now)


def _build_daily_best_pick_candidates(results):
    return _alerts_build_daily_best_pick_candidates(
        results,
        config=config,
        helpers=_daily_alert_module_helpers(),
        get_now=get_thai_now,
    )


def _build_cdc_daily_trend_candidates(results, existing_candidates=None, min_conf=None):
    return _alerts_build_cdc_daily_trend_candidates(
        results,
        config=config,
        helpers=_daily_alert_module_helpers(),
        get_now=get_thai_now,
        existing_candidates=existing_candidates,
        min_conf=min_conf,
    )


def _build_daily_summary_message(results, existing_candidates=None, min_conf=None):
    enabled = bool(getattr(config, "TELEGRAM_DAILY_SUMMARY_ENABLED", True))
    if not enabled:
        return None
    top_n = getattr(config, "TELEGRAM_DAILY_SUMMARY_TOP_N", 3)
    try:
        top_n = max(1, int(top_n))
    except Exception:
        top_n = 3
    if not isinstance(min_conf, (int, float)):
        min_conf = getattr(config, "TELEGRAM_DAILY_BEST_PICK_MIN_CONFIDENCE", 58.0)
    try:
        min_conf = float(min_conf)
    except Exception:
        min_conf = 58.0
    ranked = []
    if isinstance(existing_candidates, list):
        ranked.extend([row for row in existing_candidates if isinstance(row, dict)])
    if not ranked:
        ranked, _ = _build_telegram_candidates(results or [], min_conf)
    ranked = [row for row in ranked if isinstance(row, dict)]
    ranked.sort(key=lambda row: (float(row.get("score", 0.0)), float(row.get("confidence", 0.0))), reverse=True)
    unique_ranked = []
    seen_symbols = set()
    for row in ranked:
        symbol = normalize_symbol(row.get("symbol") or "")
        if not symbol or symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        unique_ranked.append(row)
        if len(unique_ranked) >= top_n:
            break
    now_text = get_thai_now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "<b>🗓️ Daily Telegram Summary</b>",
        f"⏱️ <b>เวลา:</b> {now_text}",
    ]
    if unique_ranked:
        wr_values = []
        exp_values = []
        trades_values = []
        lines.append("📌 <b>Top backtest-qualified setups วันนี้:</b>")
        for idx, row in enumerate(unique_ranked, start=1):
            metrics = _candidate_edge_metrics(row)
            wr = metrics.get("win_rate_pct")
            exp = metrics.get("expectancy_rr")
            trades = metrics.get("trades")
            if isinstance(wr, (int, float)):
                wr_values.append(float(wr))
            if isinstance(exp, (int, float)):
                exp_values.append(float(exp))
            if isinstance(trades, (int, float)):
                trades_values.append(float(trades))
            lines.append(
                f"{idx}. <b>{_html_escape(str(row.get('strategy') or 'ALERT'))}</b> | "
                f"{_html_escape(str(row.get('signal') or 'WAIT'))} | "
                f"{_html_escape(normalize_symbol(row.get('symbol') or '') or '-')}"
            )
            lines.append(
                "   "
                + f"Conf {float(row.get('confidence', 0.0)):.0f}%"
                + f" | WR {float(wr):.1f}%"
                + f" | ExpRR {float(exp):.2f}"
                + f" | Trades {int(round(float(trades)))}"
            )
        lines.append("✅ ส่งเฉพาะสัญญาณที่มี backtest metrics ครบตามเกณฑ์")
        return {
            "strategy": "DAILY_SUMMARY",
            "signal": "INFO",
            "score": float(sum(wr_values) / len(wr_values)) if wr_values else 0.0,
            "confidence": float(sum(wr_values) / len(wr_values)) if wr_values else 0.0,
            "edge_metrics": {
                "win_rate_pct": float(sum(wr_values) / len(wr_values)) if wr_values else None,
                "expectancy_rr": float(sum(exp_values) / len(exp_values)) if exp_values else None,
                "trades": float(sum(trades_values) / len(trades_values)) if trades_values else None,
            },
            "message": "\n".join(lines),
            "cache_key": f"DAILYSUMMARY|{get_thai_now().strftime('%Y%m%d')}",
        }
    lines.append("🛑 <b>วันนี้ยังไม่มี setup ที่ผ่าน backtest gate สำหรับส่งเข้าเทรด</b>")
    lines.append("✅ ระบบงดส่ง directional alert เพื่อคุมคุณภาพและรักษาอัตราชนะ")
    lines.append(
        "📏 เกณฑ์ขั้นต่ำ: "
        + f"WR >= {float(getattr(config, 'TELEGRAM_ALERT_ENTRY_MIN_HIST_WIN_RATE', 58.0)):.1f}%"
        + f" | ExpRR >= {float(getattr(config, 'TELEGRAM_ALERT_ENTRY_MIN_EXPECTANCY_RR', 0.05)):.2f}"
        + f" | Trades >= {int(getattr(config, 'TELEGRAM_ALERT_ENTRY_MIN_HIST_TRADES', 8))}"
    )
    return {
        "strategy": "DAILY_SUMMARY",
        "signal": "INFO",
        "score": 0.0,
        "confidence": 0.0,
        "edge_metrics": {},
        "message": "\n".join(lines),
        "cache_key": f"DAILYSUMMARY|{get_thai_now().strftime('%Y%m%d')}",
    }


def _notify_telegram_from_results(results):
    return _alerts_pipeline_notify_telegram_from_results(
        results,
        config=config,
        helpers=_pipeline_module_helpers(),
        get_now=get_thai_now,
        logger=logger,
    )


def _track_alert_performance(candidates, sent_count):
    """Track and log performance metrics for sent alerts to ensure >60% win rate"""
    if not candidates or sent_count <= 0:
        return
    
    total_win_rate = 0.0
    total_expectancy = 0.0
    valid_count = 0
    
    for candidate in candidates:
        edge_metrics = candidate.get("edge_metrics", {})
        win_rate = edge_metrics.get("win_rate_pct")
        expectancy = edge_metrics.get("expectancy_rr")
        
        if isinstance(win_rate, (int, float)) and isinstance(expectancy, (int, float)):
            total_win_rate += win_rate
            total_expectancy += expectancy
            valid_count += 1
    
    if valid_count > 0:
        avg_win_rate = total_win_rate / valid_count
        avg_expectancy = total_expectancy / valid_count
        
        logger.info(
            "Alert Performance Tracking: Avg Win Rate=%.1f%%, Avg Expectancy=%.2f, Alerts=%d",
            avg_win_rate, avg_expectancy, sent_count
        )
        
        # Log warning if win rate drops below target
        if avg_win_rate < 60.0:
            logger.warning(
                "ALERT: Average win rate below 60%% target (%.1f%%). Consider tightening filters.",
                avg_win_rate
            )
        elif avg_win_rate >= 65.0:
            logger.info(
                "SUCCESS: Average win rate exceeds 65%% (%.1f%%). System performing well.",
                avg_win_rate
            )


def _get_thread_curl_session():
    return _data_get_thread_curl_session(config=config, logger=logger)


def _set_thread_curl_session(session):
    return _data_set_thread_curl_session(session)


def _normalize_df_index(df):
    return _data_normalize_df_index(df)


def _normalize_price_columns(df, symbol=None):
    return _data_normalize_price_columns(df, symbol=symbol, normalize_symbol_fn=normalize_symbol)


def _history_store_enabled():
    return bool(getattr(config, "YF_HISTORY_STORE_ENABLE", True))


def _history_store_dir():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "yf_history")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _history_store_file_path(symbol, interval, auto_adjust):
    sym = normalize_symbol(symbol)
    interval_text = str(interval or "1d").lower()
    adj_text = "adj" if bool(auto_adjust) else "raw"
    safe_symbol = re.sub(r"[^A-Z0-9_-]+", "_", sym or "UNKNOWN")
    digest = hashlib.sha1(f"{safe_symbol}|{interval_text}|{adj_text}".encode("utf-8")).hexdigest()[:10]
    return os.path.join(_history_store_dir(), f"{safe_symbol}_{interval_text}_{adj_text}_{digest}.csv")


def _history_store_read(symbol, interval=None, auto_adjust=True):
    if not _history_store_enabled():
        return None
    path = _history_store_file_path(symbol, interval, auto_adjust)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or "Datetime" not in df.columns:
            return None
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
        df = _normalize_df_index(df)
        df = _normalize_price_columns(df, symbol)
        return df if df is not None and not df.empty else None
    except Exception:
        return None


def _history_store_write(symbol, interval, auto_adjust, df):
    if not _history_store_enabled() or df is None or getattr(df, "empty", True):
        return
    path = _history_store_file_path(symbol, interval, auto_adjust)
    try:
        out = df.copy()
        out = _normalize_df_index(out)
        out = _normalize_price_columns(out, symbol)
        if out is None or out.empty:
            return
        out = out.sort_index()
        out.index.name = "Datetime"
        out = out.reset_index()
        with _HISTORY_STORE_LOCK:
            out.to_csv(path, index=False)
    except Exception:
        return


def _history_store_merge(existing_df, new_df, symbol=None):
    frames = []
    if isinstance(existing_df, pd.DataFrame) and not existing_df.empty:
        frames.append(existing_df.copy())
    if isinstance(new_df, pd.DataFrame) and not new_df.empty:
        frames.append(new_df.copy())
    if not frames:
        return None
    merged = pd.concat(frames, axis=0)
    merged = _normalize_df_index(merged)
    merged = _normalize_price_columns(merged, symbol)
    if merged is None or merged.empty:
        return None
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    keep_rows = getattr(config, "YF_HISTORY_STORE_MAX_ROWS", 0)
    try:
        keep_rows = int(keep_rows)
    except Exception:
        keep_rows = 0
    if keep_rows > 0 and len(merged) > keep_rows:
        merged = merged.tail(keep_rows).copy()
    return merged


def _alert_history_enabled():
    return bool(getattr(config, "TELEGRAM_ALERT_HISTORY_ENABLED", True))


def _alert_history_dir():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "telegram_alerts")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _alert_history_file_path():
    return os.path.join(_alert_history_dir(), "alert_history.jsonl")


def _alert_history_csv_path():
    return os.path.join(_alert_history_dir(), "alert_history.csv")


def _alert_run_report_enabled():
    return bool(getattr(config, "TELEGRAM_ALERT_RUN_REPORT_ENABLED", True))


def _alert_run_report_file_path():
    return os.path.join(_alert_history_dir(), "latest_run.json")


def _alert_run_report_log_path():
    return os.path.join(_alert_history_dir(), "run_reports.jsonl")


def _alert_history_export_csv_enabled():
    return bool(getattr(config, "TELEGRAM_ALERT_HISTORY_EXPORT_CSV", True))


def _alert_history_csv_fieldnames():
    return _alerts_reporting_alert_history_csv_fieldnames()


def _sync_alert_history_csv_locked():
    return _alerts_reporting_sync_alert_history_csv_locked(
        export_enabled=_alert_history_export_csv_enabled(),
        jsonl_path=_alert_history_file_path(),
        csv_path=_alert_history_csv_path(),
    )


def _candidate_message_preview(candidate):
    return _alerts_reporting_candidate_message_preview(candidate)


def _candidate_ops_snapshot(candidate):
    return _alerts_reporting_candidate_ops_snapshot(
        candidate,
        helpers=_reporting_module_helpers(),
    )


def _record_telegram_run_report(
    results,
    kill,
    kill_reason,
    min_conf,
    dynamic_min_conf,
    candidates,
    sent_candidates,
    daily_pick_sent,
    daily_summary_sent,
    dropped_by_cache,
    dropped_by_symbol_cap,
    dropped_by_run_cap,
    quality_drop_counts,
):
    return _alerts_reporting_record_telegram_run_report(
        results=results,
        kill=kill,
        kill_reason=kill_reason,
        min_conf=min_conf,
        dynamic_min_conf=dynamic_min_conf,
        candidates=candidates,
        sent_candidates=sent_candidates,
        daily_pick_sent=daily_pick_sent,
        daily_summary_sent=daily_summary_sent,
        dropped_by_cache=dropped_by_cache,
        dropped_by_symbol_cap=dropped_by_symbol_cap,
        dropped_by_run_cap=dropped_by_run_cap,
        quality_drop_counts=quality_drop_counts,
        config=config,
        helpers=_reporting_module_helpers(),
        get_now=get_thai_now,
        history_lock=_ALERT_HISTORY_LOCK,
    )


def _read_latest_telegram_run_report():
    return _alerts_reporting_read_latest_telegram_run_report(_alert_run_report_file_path())


def _alert_history_trim_locked(path, max_rows):
    return _alerts_reporting_alert_history_trim_locked(path, max_rows)


def _candidate_backtest_snapshot(candidate):
    return _alerts_reporting_candidate_backtest_snapshot(
        candidate,
        candidate_edge_metrics=_candidate_edge_metrics,
    )


def _record_telegram_alert_history(candidate, min_conf=None, dynamic_min_conf=None, daily_pick=False):
    return _alerts_reporting_record_telegram_alert_history(
        candidate,
        min_conf=min_conf,
        dynamic_min_conf=dynamic_min_conf,
        daily_pick=daily_pick,
        config=config,
        helpers=_reporting_module_helpers(),
        get_now=get_thai_now,
        history_lock=_ALERT_HISTORY_LOCK,
    )


def _read_telegram_alert_history(days=None, strategies=None, symbols=None):
    return _alerts_reporting_read_telegram_alert_history(
        days=days,
        strategies=strategies,
        symbols=symbols,
        helpers=_reporting_module_helpers(),
        get_now=get_thai_now,
        history_lock=_ALERT_HISTORY_LOCK,
    )


def _build_telegram_alert_report(days=30, strategies=None, symbols=None, limit_examples_per_strategy=1):
    return _alerts_reporting_build_telegram_alert_report(
        days=days,
        strategies=strategies,
        symbols=symbols,
        limit_examples_per_strategy=limit_examples_per_strategy,
        helpers=_reporting_module_helpers(),
        get_now=get_thai_now,
        strategy_order=_TELEGRAM_REPORT_STRATEGY_ORDER,
        history_lock=_ALERT_HISTORY_LOCK,
    )


def _build_telegram_alert_live_preview(results, limit_examples_per_strategy=1):
    return _alerts_reporting_build_telegram_alert_live_preview(
        results,
        limit_examples_per_strategy=limit_examples_per_strategy,
        config=config,
        helpers=_reporting_module_helpers(),
        get_now=get_thai_now,
        strategy_order=_TELEGRAM_REPORT_STRATEGY_ORDER,
    )


def _reporting_module_helpers():
    return {
        "alert_run_report_enabled": _alert_run_report_enabled,
        "alert_run_report_file_path": _alert_run_report_file_path,
        "alert_run_report_log_path": _alert_run_report_log_path,
        "alert_history_enabled": _alert_history_enabled,
        "alert_history_file_path": _alert_history_file_path,
        "alert_history_csv_path": _alert_history_csv_path,
        "normalize_symbol": normalize_symbol,
        "pick_plan_value": _pick_plan_value,
        "candidate_backtest_snapshot": _candidate_backtest_snapshot,
        "candidate_ops_snapshot": _candidate_ops_snapshot,
        "candidate_alert_profile": _candidate_alert_profile,
        "candidate_mode_label": _candidate_mode_label,
        "get_plan_label": _get_plan_label,
        "candidate_message_preview": _candidate_message_preview,
        "sync_alert_history_csv_locked": _sync_alert_history_csv_locked,
        "alert_history_trim_locked": _alert_history_trim_locked,
        "timedelta": timedelta,
        "read_telegram_alert_history": _read_telegram_alert_history,
        "telegram_kill_switch_state": _telegram_kill_switch_state,
        "telegram_dynamic_conf_threshold": _telegram_dynamic_conf_threshold,
        "build_telegram_candidates": _build_telegram_candidates,
        "build_cdc_daily_trend_candidates": _build_cdc_daily_trend_candidates,
        "build_daily_best_pick_candidates": _build_daily_best_pick_candidates,
    }


def _write_verify_output(output_path, payload):
    return _alerts_reporting_write_verify_output(
        output_path,
        results=payload.get("results"),
        request_meta=payload.get("request"),
        summary=payload.get("summary"),
        telegram_alerts=payload.get("telegram_alerts"),
        all_weather=payload.get("all_weather"),
        backtest_rules=payload.get("backtest_rules"),
        health=payload.get("health"),
        latest_run=payload.get("latest_run"),
        live_preview=payload.get("live_preview"),
        include_results=bool(payload.get("include_results")),
        clean_json_value=_clean_json_value,
    )


def _data_source_health_snapshot():
    return _data_build_source_health_snapshot(config=config)


def _period_to_timedelta(period):
    return _data_period_to_timedelta(period, now_getter=get_thai_now)


def _slice_history_by_period(df, period):
    return _data_slice_history_by_period(df, period, now_getter=get_thai_now)


def _remote_history_period(period, interval):
    return _data_remote_history_period(period, interval, now_getter=get_thai_now)


def _chart_interval(interval):
    return _data_chart_interval(interval)


def _prefer_chart_api():
    return _data_prefer_chart_api(config=config, environ=os.environ)


def _fetch_yahoo_chart_history(symbol, period, interval=None, auto_adjust=True):
    return _data_fetch_yahoo_chart_history(
        symbol,
        period,
        interval=interval,
        auto_adjust=auto_adjust,
        config=config,
        logger=logger,
        helpers={
            "normalize_symbol": normalize_symbol,
            "get_thread_curl_session": _get_thread_curl_session,
            "normalize_df_index": _normalize_df_index,
        },
    )


def get_yf_history(symbol, period, interval=None, auto_adjust=True, cache_ttl_seconds=None):
    return _data_get_yf_history(
        symbol,
        period,
        interval=interval,
        auto_adjust=auto_adjust,
        cache_ttl_seconds=cache_ttl_seconds,
        config=config,
        logger=logger,
        helpers={
            "normalize_symbol": normalize_symbol,
            "cache_get": _YF_CACHE.get,
            "cache_set": _YF_CACHE.set,
            "empty_sentinel": _YF_EMPTY_SENTINEL,
            "history_store_read": _history_store_read,
            "history_store_merge": _history_store_merge,
            "history_store_write": _history_store_write,
            "configure_yf_tz_cache": _configure_yf_tz_cache,
            "normalize_price_columns": _normalize_price_columns,
            "normalize_df_index": _normalize_df_index,
            "slice_history_by_period": _slice_history_by_period,
            "remote_history_period": _remote_history_period,
            "prefer_chart_api": _prefer_chart_api,
            "fetch_yahoo_chart_history": _fetch_yahoo_chart_history,
            "get_thread_curl_session": _get_thread_curl_session,
            "is_yf_auth_error": _is_yf_auth_error,
            "clear_yf_runtime_cache": _clear_yf_runtime_cache,
            "set_thread_curl_session": _set_thread_curl_session,
            "record_source_health_event": lambda *args, **kwargs: _data_record_source_health_event(*args, config=config, **kwargs),
        },
    )


def _get_http_verify_setting():
    return _data_get_http_verify_setting(config=config, logger=logger)


def _create_curl_session():
    return _data_create_curl_session(config=config, logger=logger)


def _ema_cross_15m_get_cached(symbol, profile_key="default"):
    sym = normalize_symbol(symbol)
    cache_key = f"{sym}|{str(profile_key or 'default').lower()}"
    cached = EMA_CROSS_15M_OPT_CACHE.get(cache_key)
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


def _ema_cross_15m_set_cached(symbol, payload, profile_key="default"):
    sym = normalize_symbol(symbol)
    if not isinstance(payload, dict):
        return
    payload = payload.copy()
    payload["cached_at"] = get_thai_now()
    cache_key = f"{sym}|{str(profile_key or 'default').lower()}"
    EMA_CROSS_15M_OPT_CACHE[cache_key] = payload


def _sanitize_ema_lengths(fast_len, slow_len, default_fast=12, default_slow=26):
    try:
        fast = int(fast_len) if fast_len is not None else int(default_fast)
    except Exception:
        fast = int(default_fast)
    try:
        slow = int(slow_len) if slow_len is not None else int(default_slow)
    except Exception:
        slow = int(default_slow)
    if fast < 2 or slow < 2 or fast >= slow:
        fast, slow = int(default_fast), int(default_slow)
    return fast, slow


def _resolve_optimized_ema_lengths(symbol, df, use_opt=True, profile_key="default", direction_filter=None):
    sym = normalize_symbol(symbol)
    cached = _ema_cross_15m_get_cached(sym, profile_key=profile_key)
    best_fast_len = None
    best_slow_len = None
    opt_meta = None
    if isinstance(cached, dict):
        best_fast_len = cached.get("fast_len")
        best_slow_len = cached.get("slow_len")
        opt_meta = cached.get("opt_meta")
    if use_opt and isinstance(df, pd.DataFrame) and not df.empty and (best_fast_len is None or best_slow_len is None):
        try:
            opt_meta = optimize_best_ema_cross_15m(sym, df, direction_filter=direction_filter)
        except Exception as e:
            logger.warning("EMA optimization failed for %s: %s", sym, e, exc_info=True)
            opt_meta = None
        best = opt_meta.get("best") if isinstance(opt_meta, dict) else None
        if isinstance(best, dict):
            best_fast_len = best.get("fast_len")
            best_slow_len = best.get("slow_len")
        best_fast_len, best_slow_len = _sanitize_ema_lengths(best_fast_len, best_slow_len)
        _ema_cross_15m_set_cached(sym, {
            "fast_len": best_fast_len,
            "slow_len": best_slow_len,
            "opt_meta": opt_meta,
        }, profile_key=profile_key)
    return (*_sanitize_ema_lengths(best_fast_len, best_slow_len), opt_meta)


def _add_price_patterns(df):
    """
    Adds basic price pattern flags to the DataFrame.
    """
    op = df["Open"]
    hi = df["High"]
    lo = df["Low"]
    cl = df["Close"]
    
    body = (cl - op).abs()
    upper_shadow = hi - cl.where(cl > op, op)
    lower_shadow = cl.where(cl < op, op) - lo
    hl_range = hi - lo
    
    prev_op = op.shift(1)
    prev_cl = cl.shift(1)
    is_prev_red = prev_cl < prev_op
    is_curr_green = cl > op
    df["Bullish_Engulfing"] = is_prev_red & is_curr_green & (cl > prev_op) & (op < prev_cl)
    
    is_prev_green = prev_cl > prev_op
    is_curr_red = cl < op
    df["Bearish_Engulfing"] = is_prev_green & is_curr_red & (cl < prev_op) & (op > prev_cl)
    
    df["Bullish_Pinbar"] = (lower_shadow > 2 * body) & (upper_shadow < 0.2 * hl_range) & (hl_range > 0)
    df["Bearish_Pinbar"] = (upper_shadow > 2 * body) & (lower_shadow < 0.2 * hl_range) & (hl_range > 0)
    
    df["Bullish_Pattern"] = df["Bullish_Engulfing"] | df["Bullish_Pinbar"]
    df["Bearish_Pattern"] = df["Bearish_Engulfing"] | df["Bearish_Pinbar"]
    
    df["Pattern_Name"] = np.where(df["Bullish_Engulfing"], "Bullish Engulfing",
                         np.where(df["Bearish_Engulfing"], "Bearish Engulfing",
                         np.where(df["Bullish_Pinbar"], "Bullish Pinbar",
                         np.where(df["Bearish_Pinbar"], "Bearish Pinbar", "None"))))
    
    return df


def _recent_pattern_confirmation(df, idx, direction, lookback_bars=3, mode="engulfing_pinbar"):
    if df is None or getattr(df, "empty", True):
        return False, "None", None
    if direction not in ("BUY", "SELL"):
        return False, "None", None
    try:
        idx = int(idx)
        lookback_bars = max(1, int(lookback_bars))
    except Exception:
        return False, "None", None
    if idx < 0 or idx >= len(df):
        return False, "None", None
    mode_text = str(mode or "engulfing_pinbar").strip().lower()
    start = max(0, idx - lookback_bars + 1)
    window = df.iloc[start: idx + 1]
    if window.empty:
        return False, "None", None
    if direction == "BUY":
        allowed = {"bullish engulfing", "bullish pinbar"}
    else:
        allowed = {"bearish engulfing", "bearish pinbar"}
    names = window["Pattern_Name"].tolist() if "Pattern_Name" in window.columns else []
    for rel_i in range(len(window) - 1, -1, -1):
        raw_name = str(names[rel_i] or "").strip()
        if not raw_name or raw_name.lower() not in allowed:
            continue
        abs_i = start + rel_i
        if mode_text == "engulfing_only" and "engulfing" not in raw_name.lower():
            continue
        if mode_text == "pinbar_only" and "pinbar" not in raw_name.lower():
            continue
        return True, raw_name, abs_i
    return False, "None", None


def _candle_based_risk(df, idx, direction, atr_value, default_stop, buffer_atr=0.15):
    if df is None or getattr(df, "empty", True):
        return None
    if direction not in ("BUY", "SELL"):
        return None
    try:
        idx = int(idx)
    except Exception:
        return None
    if idx < 0 or idx >= len(df):
        return None
    try:
        default_stop = float(default_stop)
    except Exception:
        return None
    try:
        atr_buf = float(buffer_atr)
    except Exception:
        atr_buf = 0.15
    atr_component = 0.0
    if isinstance(atr_value, (int, float)) and float(atr_value) > 0:
        atr_component = float(atr_value) * max(0.0, atr_buf)
    low_i = float(df["Low"].iloc[idx]) if pd.notna(df["Low"].iloc[idx]) else None
    high_i = float(df["High"].iloc[idx]) if pd.notna(df["High"].iloc[idx]) else None
    if direction == "BUY" and low_i is not None:
        return float(min(default_stop, low_i - atr_component))
    if direction == "SELL" and high_i is not None:
        return float(max(default_stop, high_i + atr_component))
    return float(default_stop)


def _price_action_strategy_helpers():
    return {
        "add_price_patterns": _add_price_patterns,
        "safe_float": _safe_float,
        "recent_pattern_confirmation": _recent_pattern_confirmation,
        "price_action_proxy_metrics": _price_action_proxy_metrics,
        "candle_based_risk": _candle_based_risk,
    }


def _price_action_15m_plan(symbol, item, data_15m=None, data_1h=None, order_blocks=None, prediction=None, phase_status=None):
    return _strategies_build_price_action_plan(
        symbol,
        item,
        data_15m=data_15m,
        data_1h=data_1h,
        order_blocks=order_blocks,
        prediction=prediction,
        phase_status=phase_status,
        config=config,
        helpers=_price_action_strategy_helpers(),
    )


def _cdc_forecast_bias(close, fast, slow, k, d, market_top, momentum_lookback=3):
    try:
        lookback = max(1, int(momentum_lookback))
    except Exception:
        lookback = 3
    bull_score = 0.0
    bear_score = 0.0
    reasons = []
    close_now = float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else None
    fast_now = float(fast.iloc[-1]) if pd.notna(fast.iloc[-1]) else None
    slow_now = float(slow.iloc[-1]) if pd.notna(slow.iloc[-1]) else None
    k_now = float(k.iloc[-1]) if pd.notna(k.iloc[-1]) else None
    d_now = float(d.iloc[-1]) if pd.notna(d.iloc[-1]) else None
    fast_prev = float(fast.iloc[-1 - lookback]) if len(fast) > lookback and pd.notna(fast.iloc[-1 - lookback]) else None
    close_prev = float(close.iloc[-1 - lookback]) if len(close) > lookback and pd.notna(close.iloc[-1 - lookback]) else None
    if isinstance(fast_now, (int, float)) and isinstance(slow_now, (int, float)):
        if fast_now > slow_now:
            bull_score += 18.0
            reasons.append("EMA เร็วเหนือ EMA ช้า")
        elif fast_now < slow_now:
            bear_score += 18.0
            reasons.append("EMA เร็วต่ำกว่า EMA ช้า")
    if isinstance(close_now, (int, float)) and isinstance(fast_now, (int, float)):
        if close_now > fast_now:
            bull_score += 12.0
            reasons.append("ราคายืนเหนือ EMA เร็ว")
        elif close_now < fast_now:
            bear_score += 12.0
            reasons.append("ราคาต่ำกว่า EMA เร็ว")
    if isinstance(fast_now, (int, float)) and isinstance(fast_prev, (int, float)):
        if fast_now > fast_prev:
            bull_score += 10.0
            reasons.append("EMA เร็วกำลังชันขึ้น")
        elif fast_now < fast_prev:
            bear_score += 10.0
            reasons.append("EMA เร็วกำลังชันลง")
    if isinstance(close_now, (int, float)) and isinstance(close_prev, (int, float)) and close_prev != 0:
        momentum_pct = ((close_now - close_prev) / abs(close_prev)) * 100.0
        if momentum_pct > 0:
            bull_score += min(12.0, 6.0 + momentum_pct * 4.0)
            reasons.append("โมเมนตัมสั้นเป็นบวก")
        elif momentum_pct < 0:
            bear_score += min(12.0, 6.0 + abs(momentum_pct) * 4.0)
            reasons.append("โมเมนตัมสั้นเป็นลบ")
    if isinstance(k_now, (int, float)) and isinstance(d_now, (int, float)):
        if k_now > d_now:
            bull_score += 8.0
            reasons.append("StochRSI K อยู่เหนือ D")
        elif k_now < d_now:
            bear_score += 8.0
            reasons.append("StochRSI K อยู่ต่ำกว่า D")
        if k_now < 35.0 and d_now < 35.0:
            bull_score += 6.0
            reasons.append("อยู่โซนเด้งกลับ")
        if k_now > 70.0 and d_now > 70.0:
            bear_score += 6.0
            reasons.append("อยู่โซนพักตัว")
    if bool(market_top):
        bear_score += 14.0
        reasons.append("VixFix เข้าพื้นที่เสี่ยงยอด")
    gap = abs(bull_score - bear_score)
    direction = "NEUTRAL"
    if bull_score >= bear_score + 8.0:
        direction = "BUY"
    elif bear_score >= bull_score + 8.0:
        direction = "SELL"
    score = min(92.0, 50.0 + gap)
    return {
        "direction": direction,
        "score": float(max(50.0, score)),
        "bull_score": float(bull_score),
        "bear_score": float(bear_score),
        "reason": ", ".join(reasons[:4]),
    }


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
    df = _add_price_patterns(df)
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
    return df


def _trade_cost_rr(entry_price, risk_dist):
    try:
        entry = float(entry_price)
        risk = float(risk_dist)
    except Exception:
        return 0.0
    if entry <= 0 or risk <= 0:
        return 0.0
    fee_bps = getattr(config, "BACKTEST_FEE_BPS", 10.0)
    slippage_bps = getattr(config, "BACKTEST_SLIPPAGE_BPS", 5.0)
    try:
        fee_bps = float(fee_bps)
    except Exception:
        fee_bps = 10.0
    try:
        slippage_bps = float(slippage_bps)
    except Exception:
        slippage_bps = 5.0
    if fee_bps < 0:
        fee_bps = 0.0
    if slippage_bps < 0:
        slippage_bps = 0.0
    # Round-trip cost: entry+exit fee plus entry+exit slippage.
    total_cost_pct = ((fee_bps * 2.0) + (slippage_bps * 2.0)) / 10000.0
    cost_price = entry * total_cost_pct
    return max(0.0, float(cost_price / risk))


def _actionzone_simulate_trade_rr(
    df,
    entry_i,
    direction,
    sl_i,
    tp2_i,
    risk_i,
    stop_mult,
    max_forward=64,
    tp1_r=None,
    tp1_fraction=0.5,
    move_sl_to_be=True,
    trailing_atr_mult=None,
    time_stop_bars=None,
):
    if df is None or df.empty:
        return None
    if direction not in ("BUY", "SELL"):
        return None
    try:
        entry_i = int(entry_i)
        max_forward = int(max_forward)
    except Exception:
        return None
    if max_forward < 1:
        return None
    try:
        entry_price = float(df["Close"].iloc[entry_i])
        sl_i = float(sl_i)
        tp2_i = float(tp2_i)
        risk_i = float(risk_i)
        stop_mult = float(stop_mult)
    except Exception:
        return None
    if not math.isfinite(entry_price) or entry_price <= 0 or not math.isfinite(risk_i) or risk_i <= 0:
        return None
    cost_rr = _trade_cost_rr(entry_price, risk_i)
    if time_stop_bars is not None:
        try:
            time_stop_bars = int(time_stop_bars)
        except Exception:
            time_stop_bars = None
        if isinstance(time_stop_bars, int) and time_stop_bars < 1:
            time_stop_bars = 1
    tp1_r_val = None
    if isinstance(tp1_r, (int, float)) and float(tp1_r) > 0:
        tp1_r_val = float(tp1_r)
    try:
        tp1_fraction = float(tp1_fraction)
    except Exception:
        tp1_fraction = 0.5
    if tp1_fraction <= 0:
        tp1_fraction = 0.0
    if tp1_fraction >= 1:
        tp1_fraction = 1.0
    trail_mult = None
    if isinstance(trailing_atr_mult, (int, float)) and float(trailing_atr_mult) > 0:
        trail_mult = float(trailing_atr_mult)

    partial_hit = False
    rr_realized = 0.0
    rem_frac = 1.0
    active_sl = float(sl_i)
    tp1_price = None
    if tp1_r_val is not None:
        if direction == "BUY":
            tp1_price = float(entry_price + (risk_i * tp1_r_val))
        else:
            tp1_price = float(entry_price - (risk_i * tp1_r_val))

    end_j = min(len(df), entry_i + 1 + max_forward)
    if isinstance(time_stop_bars, int):
        end_j = min(end_j, entry_i + 1 + time_stop_bars)
    if end_j <= entry_i + 1:
        return None

    peak = entry_price
    trough = entry_price
    for j in range(entry_i + 1, end_j):
        high_j = float(df["High"].iloc[j]) if pd.notna(df["High"].iloc[j]) else None
        low_j = float(df["Low"].iloc[j]) if pd.notna(df["Low"].iloc[j]) else None
        close_j = float(df["Close"].iloc[j]) if pd.notna(df["Close"].iloc[j]) else None
        atr_j = float(df["ATR"].iloc[j]) if "ATR" in df.columns and pd.notna(df["ATR"].iloc[j]) else None
        if high_j is None or low_j is None:
            continue

        if direction == "BUY":
            peak = max(peak, high_j)
            if trail_mult is not None and atr_j is not None and atr_j > 0:
                trail_sl = peak - (atr_j * trail_mult * max(1.0, stop_mult / 1.5))
                if math.isfinite(trail_sl):
                    active_sl = max(active_sl, float(trail_sl))
            # SL first (worst-case sequencing)
            if low_j <= active_sl:
                rr_realized += rem_frac * (-(1.0 + cost_rr))
                return float(rr_realized)
            if not partial_hit and tp1_price is not None and high_j >= tp1_price and tp1_fraction > 0:
                rr_realized += tp1_fraction * max(0.0, float(tp1_r_val) - cost_rr)
                rem_frac = max(0.0, 1.0 - tp1_fraction)
                partial_hit = True
                if move_sl_to_be:
                    active_sl = max(active_sl, float(entry_price))
            if high_j >= tp2_i:
                rr_realized += rem_frac * max(0.0, float(abs(tp2_i - entry_price) / risk_i) - cost_rr)
                return float(rr_realized)
        else:
            trough = min(trough, low_j)
            if trail_mult is not None and atr_j is not None and atr_j > 0:
                trail_sl = trough + (atr_j * trail_mult * max(1.0, stop_mult / 1.5))
                if math.isfinite(trail_sl):
                    active_sl = min(active_sl, float(trail_sl))
            if high_j >= active_sl:
                rr_realized += rem_frac * (-(1.0 + cost_rr))
                return float(rr_realized)
            if not partial_hit and tp1_price is not None and low_j <= tp1_price and tp1_fraction > 0:
                rr_realized += tp1_fraction * max(0.0, float(tp1_r_val) - cost_rr)
                rem_frac = max(0.0, 1.0 - tp1_fraction)
                partial_hit = True
                if move_sl_to_be:
                    active_sl = min(active_sl, float(entry_price))
            if low_j <= tp2_i:
                rr_realized += rem_frac * max(0.0, float(abs(entry_price - tp2_i) / risk_i) - cost_rr)
                return float(rr_realized)

        # Time stop at last bar close (if configured, we already limited end_j)
        if j == end_j - 1 and close_j is not None and math.isfinite(close_j):
            rr_move = (float(close_j - entry_price) / float(risk_i)) if direction == "BUY" else (float(entry_price - close_j) / float(risk_i))
            rr_realized += rem_frac * (float(rr_move) - cost_rr)
            return float(rr_realized)
    return None


def _backtest_ema_cross_15m(df, fast_len, slow_len, tp_mult=5.0, max_forward=64, return_rr_list=False, direction_filter=None):
    if df is None or df.empty:
        return None
    try:
        fast_len = int(fast_len)
        slow_len = int(slow_len)
        tp_mult = float(tp_mult)
        max_forward = int(max_forward)
    except Exception:
        return None
    direction_filter = str(direction_filter or "").upper().strip()
    if direction_filter not in ("", "BUY", "SELL"):
        direction_filter = ""
    if fast_len < 2 or slow_len < 2 or fast_len >= slow_len:
        return None
    if tp_mult <= 0 or max_forward < 1:
        return None
    stop_mult = getattr(config, "EMA_CROSS_15M_STOP_ATR_MULT", getattr(config, "ACTIONZONE_15M_STOP_ATR_MULT", 1.5))
    try:
        stop_mult = float(stop_mult)
    except Exception:
        stop_mult = 1.5
    if stop_mult <= 0:
        stop_mult = 1.0
    min_rvol = getattr(config, "EMA_CROSS_15M_MIN_RVOL", 1.0)
    min_ema_gap_pct = getattr(config, "EMA_CROSS_15M_MIN_EMA_GAP_PCT", 0.03)
    min_atr_pct = getattr(config, "EMA_CROSS_15M_MIN_ATR_PCT", 0.10)
    max_atr_pct = getattr(config, "EMA_CROSS_15M_MAX_ATR_PCT", 6.00)
    min_adx = getattr(config, "EMA_CROSS_15M_MIN_ADX", 18.0)
    require_ema200_alignment = bool(getattr(config, "EMA_CROSS_15M_REQUIRE_EMA200_ALIGNMENT", True))
    require_pattern = bool(getattr(config, "EMA_CROSS_15M_REQUIRE_PATTERN", False))
    pattern_lookback = getattr(config, "EMA_CROSS_15M_PATTERN_LOOKBACK_BARS", 3)
    pattern_mode = getattr(config, "EMA_CROSS_15M_ENTRY_CONFIRMATION_MODE", "engulfing_pinbar")
    use_candle_stop = bool(getattr(config, "EMA_CROSS_15M_USE_CANDLE_STOP", True))
    candle_stop_buffer_atr = getattr(config, "EMA_CROSS_15M_CANDLE_STOP_BUFFER_ATR", 0.15)
    time_stop_bars = getattr(config, "EMA_CROSS_15M_MAX_FORWARD_BARS", max_forward)
    tp1_r = 1.0
    tp1_fraction = 0.0
    move_sl_to_be = False
    trailing_atr_mult = None
    try:
        min_rvol = float(min_rvol)
    except Exception:
        min_rvol = 1.15
    try:
        min_ema_gap_pct = float(min_ema_gap_pct)
    except Exception:
        min_ema_gap_pct = 0.05
    try:
        min_atr_pct = float(min_atr_pct)
    except Exception:
        min_atr_pct = 0.15
    try:
        max_atr_pct = float(max_atr_pct)
    except Exception:
        max_atr_pct = 6.00
    try:
        min_adx = float(min_adx)
    except Exception:
        min_adx = 24.0
    close = df["Close"].astype(float)
    ema_fast = close.ewm(span=fast_len, adjust=False).mean()
    ema_slow = close.ewm(span=slow_len, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    bull = ema_fast > ema_slow
    bear = ema_fast < ema_slow
    
    # ActionZone Green/Red logic
    green = bull & (close > ema_fast)
    red = bear & (close < ema_fast)
    
    buy_signal = green & (~green.shift(1, fill_value=False))
    sell_signal = red & (~red.shift(1, fill_value=False))
    
    wins = 0
    losses = 0
    total_rr = 0.0
    rr_list = [] if return_rr_list else None
    for i in range(1, len(df)):
        if buy_signal.iloc[i]:
            direction_i = "BUY"
        elif sell_signal.iloc[i]:
            direction_i = "SELL"
        else:
            continue
        if direction_filter and direction_i != direction_filter:
            continue
            
        atr_i = df["ATR"].iloc[i]
        vol_avg_i = df["Vol_Avg"].iloc[i]
        if pd.isna(atr_i) or atr_i <= 0 or pd.isna(vol_avg_i) or vol_avg_i <= 0:
            continue
        entry_i = close.iloc[i]
        if pd.isna(entry_i) or entry_i <= 0:
            continue
        rvol_i = float(df["Volume"].iloc[i] / vol_avg_i) if pd.notna(df["Volume"].iloc[i]) and float(vol_avg_i) > 0 else None
        ema_gap_pct_i = abs(float(ema_fast.iloc[i]) - float(ema_slow.iloc[i])) / float(entry_i) * 100.0 if entry_i > 0 else None
        atr_pct_i = float(atr_i) / float(entry_i) * 100.0 if entry_i > 0 else None
        if isinstance(rvol_i, (int, float)) and float(rvol_i) < min_rvol:
            continue
        if isinstance(ema_gap_pct_i, (int, float)) and float(ema_gap_pct_i) < min_ema_gap_pct:
            continue
        if isinstance(atr_pct_i, (int, float)):
            if float(atr_pct_i) < min_atr_pct:
                continue
            if float(atr_pct_i) > max_atr_pct:
                continue
        adx_i = df["ADX"].iloc[i] if "ADX" in df.columns else None
        if isinstance(adx_i, (int, float)) and float(adx_i) < min_adx:
            continue
        if require_ema200_alignment:
            ema200_i = ema200.iloc[i]
            if pd.notna(ema200_i):
                if direction_i == "BUY" and float(entry_i) < float(ema200_i):
                    continue
                if direction_i == "SELL" and float(entry_i) > float(ema200_i):
                    continue
        
        # Risk based on ATR and multiplier
        risk_i = float(atr_i * stop_mult)
        sl_i = entry_i - risk_i if direction_i == "BUY" else entry_i + risk_i
        tp_i = entry_i + risk_i * tp_mult if direction_i == "BUY" else entry_i - risk_i * tp_mult
        rr = _actionzone_simulate_trade_rr(
            df,
            entry_i=i,
            direction=direction_i,
            sl_i=sl_i,
            tp2_i=tp_i,
            risk_i=risk_i,
            stop_mult=stop_mult,
            max_forward=max_forward,
            tp1_r=tp1_r,
            tp1_fraction=tp1_fraction,
            move_sl_to_be=move_sl_to_be,
            trailing_atr_mult=trailing_atr_mult,
            time_stop_bars=time_stop_bars,
        )
        if not isinstance(rr, (int, float)):
            continue
        rr = float(rr)
        if rr >= 0:
            wins += 1
        else:
            losses += 1
        total_rr += rr
        if rr_list is not None:
            rr_list.append(rr)
    total_trades = wins + losses
    if total_trades <= 0:
        payload = {
            "fast_len": fast_len,
            "slow_len": slow_len,
            "direction_filter": direction_filter or "BOTH",
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
        "direction_filter": direction_filter or "BOTH",
        "trades": int(total_trades),
        "win_rate_pct": float(win_rate),
        "avg_rr": float(avg_rr),
        "expectancy_rr": float(avg_rr),
    }
    if rr_list is not None:
        payload["rr_list"] = rr_list
    return payload


def _actionzone_eval_candidate(train_res, valid_res, min_trades=8, min_valid_trades=4):
    if not isinstance(train_res, dict) or not isinstance(valid_res, dict):
        return None
    train_trades = train_res.get("trades")
    valid_trades = valid_res.get("trades")
    train_exp = train_res.get("expectancy_rr")
    valid_exp = valid_res.get("expectancy_rr")
    train_win = train_res.get("win_rate_pct")
    valid_win = valid_res.get("win_rate_pct")
    target_win_rate = getattr(config, "ACTIONZONE_15M_OPT_TARGET_WIN_RATE", 60.0)
    target_expectancy = getattr(config, "ACTIONZONE_15M_OPT_TARGET_EXPECTANCY_RR", 0.08)
    try:
        target_win_rate = float(target_win_rate)
    except Exception:
        target_win_rate = 60.0
    try:
        target_expectancy = float(target_expectancy)
    except Exception:
        target_expectancy = 0.08
    if not isinstance(train_trades, int) or train_trades < int(min_trades):
        return None
    if not isinstance(valid_trades, int) or valid_trades < int(min_valid_trades):
        return None
    if not isinstance(train_exp, (int, float)) or not isinstance(valid_exp, (int, float)):
        return None
    target_trades = max(int(min_trades) * 2, 14)
    robust_train = min(1.0, float(train_trades) / float(target_trades))
    robust_valid = min(1.0, float(valid_trades) / float(max(int(min_valid_trades) * 2, 8)))
    robustness = 0.65 * robust_train + 0.35 * robust_valid
    consistency_penalty = abs(float(train_exp) - float(valid_exp))
    valid_win_val = float(valid_win) if isinstance(valid_win, (int, float)) else None
    train_win_val = float(train_win) if isinstance(train_win, (int, float)) else None
    win_edge = ((valid_win_val - 50.0) / 100.0) if valid_win_val is not None else 0.0
    win_target_bonus = 0.0
    if valid_win_val is not None:
        if valid_win_val >= target_win_rate:
            win_target_bonus += min(0.55, (valid_win_val - target_win_rate) / 40.0)
        else:
            win_target_bonus -= min(0.85, (target_win_rate - valid_win_val) / 18.0)
    exp_target_bonus = 0.0
    if float(valid_exp) >= target_expectancy:
        exp_target_bonus += min(0.35, (float(valid_exp) - target_expectancy) / max(0.05, target_expectancy + 0.20))
    else:
        exp_target_bonus -= min(0.45, (target_expectancy - float(valid_exp)) / max(0.05, target_expectancy + 0.15))
    train_valid_win_gap = abs(float(train_win_val) - float(valid_win_val)) / 100.0 if train_win_val is not None and valid_win_val is not None else 0.0
    # Practical objective: prefer validation win rate near/above 60%, keep expectancy positive,
    # and penalize unstable train/validation behavior.
    opt_score = (
        (win_edge * 0.52)
        + (float(valid_exp) * 0.28)
        + (float(train_exp) * 0.12)
        + win_target_bonus
        + exp_target_bonus
    ) * (0.58 + 0.42 * robustness) - (consistency_penalty * 0.32) - (train_valid_win_gap * 0.25)
    return {
        "train_trades": int(train_trades),
        "valid_trades": int(valid_trades),
        "train_expectancy_rr": float(train_exp),
        "valid_expectancy_rr": float(valid_exp),
        "train_win_rate_pct": float(train_win) if isinstance(train_win, (int, float)) else None,
        "valid_win_rate_pct": float(valid_win) if isinstance(valid_win, (int, float)) else None,
        "consistency_penalty": float(consistency_penalty),
        "robustness_score": float(robustness * 100.0),
        "opt_score": float(opt_score),
    }


def _ema_cross_eval_candidate(train_res, valid_res, min_trades=8, min_valid_trades=4):
    if not isinstance(train_res, dict) or not isinstance(valid_res, dict):
        return None
    train_trades = train_res.get("trades")
    valid_trades = valid_res.get("trades")
    train_exp = train_res.get("expectancy_rr")
    valid_exp = valid_res.get("expectancy_rr")
    train_win = train_res.get("win_rate_pct")
    valid_win = valid_res.get("win_rate_pct")
    target_win_rate = getattr(config, "EMA_CROSS_15M_OPT_TARGET_WIN_RATE", 60.0)
    target_expectancy = getattr(config, "EMA_CROSS_15M_OPT_TARGET_EXPECTANCY_RR", 0.10)
    try:
        target_win_rate = float(target_win_rate)
    except Exception:
        target_win_rate = 60.0
    try:
        target_expectancy = float(target_expectancy)
    except Exception:
        target_expectancy = 0.10
    if not isinstance(train_trades, int) or train_trades < int(min_trades):
        return None
    if not isinstance(valid_trades, int) or valid_trades < int(min_valid_trades):
        return None
    if not isinstance(train_exp, (int, float)) or not isinstance(valid_exp, (int, float)):
        return None
    if not isinstance(train_win, (int, float)) or not isinstance(valid_win, (int, float)):
        return None
    target_trades = max(int(min_trades) * 2, 16)
    robust_train = min(1.0, float(train_trades) / float(target_trades))
    robust_valid = min(1.0, float(valid_trades) / float(max(int(min_valid_trades) * 2, 10)))
    robustness = 0.60 * robust_train + 0.40 * robust_valid
    consistency_penalty = abs(float(train_exp) - float(valid_exp))
    valid_win_val = float(valid_win)
    train_win_val = float(train_win)
    win_edge = (valid_win_val - 50.0) / 100.0
    if valid_win_val >= target_win_rate:
        win_target_bonus = min(0.70, (valid_win_val - target_win_rate) / 35.0)
    else:
        win_target_bonus = -min(1.05, (target_win_rate - valid_win_val) / 14.0)
    if float(valid_exp) >= target_expectancy:
        exp_target_bonus = min(0.30, (float(valid_exp) - target_expectancy) / max(0.05, target_expectancy + 0.20))
    else:
        exp_target_bonus = -min(0.55, (target_expectancy - float(valid_exp)) / max(0.05, target_expectancy + 0.10))
    train_valid_win_gap = abs(train_win_val - valid_win_val) / 100.0
    opt_score = (
        (win_edge * 0.68)
        + (float(valid_exp) * 0.18)
        + (float(train_exp) * 0.07)
        + win_target_bonus
        + exp_target_bonus
    ) * (0.56 + 0.44 * robustness) - (consistency_penalty * 0.28) - (train_valid_win_gap * 0.34)
    return {
        "train_trades": int(train_trades),
        "valid_trades": int(valid_trades),
        "train_expectancy_rr": float(train_exp),
        "valid_expectancy_rr": float(valid_exp),
        "train_win_rate_pct": float(train_win_val),
        "valid_win_rate_pct": float(valid_win_val),
        "consistency_penalty": float(consistency_penalty),
        "robustness_score": float(robustness * 100.0),
        "opt_score": float(opt_score),
    }


def backtest_actionzone_15m(symbol, yf_period=None, tp_mult=None, max_forward=None, return_rr_list=False, direction_filter=None):
    sym = normalize_symbol(symbol)
    if not sym:
        return None
    yf_period = str(yf_period or getattr(config, "ACTIONZONE_15M_YF_PERIOD", getattr(config, "EMA_CROSS_15M_YF_PERIOD", "90d")))
    raw = get_yf_history(sym, period=yf_period, interval="15m", auto_adjust=True)
    if raw is None or getattr(raw, "empty", True):
        return None
    df = _ema_cross_15m_prepare_df(raw)
    if df is None or getattr(df, "empty", True):
        return None
    direction_filter = str(direction_filter or "").upper().strip()
    if direction_filter not in ("", "BUY", "SELL"):
        direction_filter = ""
    use_opt = getattr(config, "ACTIONZONE_15M_USE_OPTIMIZATION", True)
    profile_key = "sell_only" if direction_filter == "SELL" else "buy_only" if direction_filter == "BUY" else "default"
    fast_len, slow_len, _ = _resolve_optimized_ema_lengths(sym, df, use_opt=use_opt, profile_key=profile_key, direction_filter=direction_filter or None)
    tp_mult = float(tp_mult if tp_mult is not None else getattr(config, "ACTIONZONE_15M_TP_MULT", getattr(config, "EMA_CROSS_15M_TP_MULT", 5.0)))
    max_forward = int(max_forward if max_forward is not None else getattr(config, "ACTIONZONE_15M_MAX_FORWARD_BARS", getattr(config, "EMA_CROSS_15M_MAX_FORWARD_BARS", 64)))
    stop_mult = getattr(config, "ACTIONZONE_15M_STOP_ATR_MULT", 1.5)
    try:
        stop_mult = float(stop_mult)
    except Exception:
        stop_mult = 1.5
    if stop_mult <= 0:
        stop_mult = 1.0
    min_rvol = getattr(config, "ACTIONZONE_15M_MIN_RVOL", 1.15)
    min_ema_gap_pct = getattr(config, "ACTIONZONE_15M_MIN_EMA_GAP_PCT", 0.05)
    min_atr_pct = getattr(config, "ACTIONZONE_15M_MIN_ATR_PCT", 0.15)
    max_atr_pct = getattr(config, "ACTIONZONE_15M_MAX_ATR_PCT", 6.00)
    min_adx = getattr(config, "ACTIONZONE_15M_MIN_ADX", 24.0)
    require_ema200_alignment = bool(getattr(config, "ACTIONZONE_15M_REQUIRE_EMA200_ALIGNMENT", True))
    require_pattern = bool(getattr(config, "ACTIONZONE_15M_REQUIRE_PATTERN", False))
    pattern_lookback = getattr(config, "ACTIONZONE_15M_PATTERN_LOOKBACK_BARS", 3)
    pattern_mode = getattr(config, "ACTIONZONE_15M_ENTRY_CONFIRMATION_MODE", "engulfing_pinbar")
    use_candle_stop = bool(getattr(config, "ACTIONZONE_15M_USE_CANDLE_STOP", True))
    candle_stop_buffer_atr = getattr(config, "ACTIONZONE_15M_CANDLE_STOP_BUFFER_ATR", 0.15)
    time_stop_bars = getattr(config, "ACTIONZONE_15M_TIME_STOP_BARS", None)
    tp1_r = getattr(config, "ACTIONZONE_15M_TP1_R", 1.0)
    tp1_fraction = getattr(config, "ACTIONZONE_15M_TP1_FRACTION", 0.5)
    move_sl_to_be = bool(getattr(config, "ACTIONZONE_15M_MOVE_SL_TO_BE", True))
    trailing_atr_mult = getattr(config, "ACTIONZONE_15M_TRAILING_ATR_MULT", None)
    try:
        min_rvol = float(min_rvol)
    except Exception:
        min_rvol = 1.15
    try:
        min_ema_gap_pct = float(min_ema_gap_pct)
    except Exception:
        min_ema_gap_pct = 0.05
    try:
        min_atr_pct = float(min_atr_pct)
    except Exception:
        min_atr_pct = 0.15
    try:
        max_atr_pct = float(max_atr_pct)
    except Exception:
        max_atr_pct = 6.00
    try:
        min_adx = float(min_adx)
    except Exception:
        min_adx = 24.0
    if tp_mult <= 0 or max_forward < 1:
        return None
    close = df["Close"].astype(float)
    ema_fast = close.ewm(span=fast_len, adjust=False).mean()
    ema_slow = close.ewm(span=slow_len, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    bull = ema_fast > ema_slow
    bear = ema_fast < ema_slow
    x_confirm = close
    green = bull & (x_confirm > ema_fast)
    red = bear & (x_confirm < ema_fast)
    buy_signal = green & (~green.shift(1, fill_value=False))
    sell_signal = red & (~red.shift(1, fill_value=False))
    wins = 0
    losses = 0
    total_rr = 0.0
    rr_list = [] if return_rr_list else None
    for i in range(1, len(df)):
        if buy_signal.iloc[i]:
            direction_i = "BUY"
        elif sell_signal.iloc[i]:
            direction_i = "SELL"
        else:
            continue
        if direction_filter and direction_i != direction_filter:
            continue
        atr_i = df["ATR"].iloc[i]
        vol_avg_i = df["Vol_Avg"].iloc[i]
        if pd.isna(atr_i) or atr_i <= 0 or pd.isna(vol_avg_i) or vol_avg_i <= 0:
            continue
        entry_i = close.iloc[i]
        if pd.isna(entry_i) or entry_i <= 0:
            continue
        rvol_i = float(df["Volume"].iloc[i] / vol_avg_i) if pd.notna(df["Volume"].iloc[i]) and float(vol_avg_i) > 0 else None
        ema_gap_pct_i = abs(float(ema_fast.iloc[i]) - float(ema_slow.iloc[i])) / float(entry_i) * 100.0 if entry_i > 0 else None
        atr_pct_i = float(atr_i) / float(entry_i) * 100.0 if entry_i > 0 else None
        if isinstance(rvol_i, (int, float)) and float(rvol_i) < min_rvol:
            continue
        if isinstance(ema_gap_pct_i, (int, float)) and float(ema_gap_pct_i) < min_ema_gap_pct:
            continue
        if isinstance(atr_pct_i, (int, float)):
            if float(atr_pct_i) < min_atr_pct:
                continue
            if float(atr_pct_i) > max_atr_pct:
                continue
        adx_i = df["ADX"].iloc[i] if "ADX" in df.columns else None
        if isinstance(adx_i, (int, float)) and float(adx_i) < min_adx:
            continue
        if require_ema200_alignment:
            ema200_i = ema200.iloc[i]
            if pd.notna(ema200_i):
                if direction_i == "BUY" and float(entry_i) < float(ema200_i):
                    continue
                if direction_i == "SELL" and float(entry_i) > float(ema200_i):
                    continue
        pattern_ok, _, pattern_idx = _recent_pattern_confirmation(df, i, direction_i, lookback_bars=pattern_lookback, mode=pattern_mode)
        if require_pattern and not pattern_ok:
            continue
        risk_i = float(atr_i * stop_mult)
        sl_i = entry_i - risk_i if direction_i == "BUY" else entry_i + risk_i
        if use_candle_stop:
            stop_ref_idx = pattern_idx if isinstance(pattern_idx, int) else i
            candle_sl_i = _candle_based_risk(df, stop_ref_idx, direction_i, atr_i, sl_i, buffer_atr=candle_stop_buffer_atr)
            if isinstance(candle_sl_i, (int, float)):
                sl_i = float(candle_sl_i)
                risk_i = abs(float(entry_i) - float(sl_i))
        if not isinstance(risk_i, (int, float)) or float(risk_i) <= 0:
            continue
        tp_i = entry_i + risk_i * tp_mult if direction_i == "BUY" else entry_i - risk_i * tp_mult
        rr = _actionzone_simulate_trade_rr(
            df,
            entry_i=i,
            direction=direction_i,
            sl_i=sl_i,
            tp2_i=tp_i,
            risk_i=risk_i,
            stop_mult=stop_mult,
            max_forward=max_forward,
            tp1_r=tp1_r,
            tp1_fraction=tp1_fraction,
            move_sl_to_be=move_sl_to_be,
            trailing_atr_mult=trailing_atr_mult,
            time_stop_bars=time_stop_bars,
        )
        if not isinstance(rr, (int, float)):
            continue
        rr = float(rr)
        if rr >= 0:
            wins += 1
        else:
            losses += 1
        total_rr += rr
        if rr_list is not None:
            rr_list.append(rr)
    total_trades = wins + losses
    if total_trades <= 0:
        payload = {
            "symbol": sym,
            "fast_len": fast_len,
            "slow_len": slow_len,
            "direction_filter": direction_filter or "BOTH",
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
        "symbol": sym,
        "fast_len": fast_len,
        "slow_len": slow_len,
        "direction_filter": direction_filter or "BOTH",
        "trades": int(total_trades),
        "win_rate_pct": float(win_rate),
        "avg_rr": float(avg_rr),
        "expectancy_rr": float(avg_rr),
    }
    if rr_list is not None:
        payload["rr_list"] = rr_list
    return payload


def build_actionzone_sell_whitelist(symbols, yf_period=None):
    if not isinstance(symbols, (list, tuple)):
        return {"symbols": [], "passed_symbols": [], "results": [], "summary": {}}
    results = []
    passed_symbols = []
    weighted_valid_trades = 0.0
    weighted_valid_wins = 0.0
    weighted_valid_exp = 0.0
    for raw_symbol in symbols:
        sym = normalize_symbol(raw_symbol)
        if not sym:
            continue
        raw = get_yf_history(sym, period=str(yf_period or getattr(config, "ACTIONZONE_15M_YF_PERIOD", "30d")), interval="15m", auto_adjust=True)
        if raw is None or getattr(raw, "empty", True):
            results.append({"symbol": sym, "passed": False, "reason": "missing_history"})
            continue
        df = _ema_cross_15m_prepare_df(raw)
        if df is None or getattr(df, "empty", True):
            results.append({"symbol": sym, "passed": False, "reason": "prepare_df_failed"})
            continue
        _, _, sell_opt_meta = _resolve_optimized_ema_lengths(
            sym,
            df,
            use_opt=True,
            profile_key="sell_only",
            direction_filter="SELL",
        )
        gate_ok, gate_reason, gate_metrics = _evaluate_sell_whitelist_gate({"sell_optimizer": sell_opt_meta})
        best = sell_opt_meta.get("best") if isinstance(sell_opt_meta, dict) else None
        row = {
            "symbol": sym,
            "passed": bool(gate_ok),
            "reason": str(gate_reason),
            "optimizer": sell_opt_meta,
            "best": best,
            "metrics": gate_metrics,
        }
        if gate_ok:
            passed_symbols.append(sym)
            valid_trades = gate_metrics.get("valid_trades")
            valid_wr = gate_metrics.get("valid_win_rate_pct")
            valid_exp = gate_metrics.get("valid_expectancy_rr")
            if isinstance(valid_trades, (int, float)) and float(valid_trades) > 0:
                weighted_valid_trades += float(valid_trades)
                if isinstance(valid_wr, (int, float)):
                    weighted_valid_wins += float(valid_trades) * float(valid_wr) / 100.0
                if isinstance(valid_exp, (int, float)):
                    weighted_valid_exp += float(valid_trades) * float(valid_exp)
        results.append(row)
    summary = {
        "symbols": len(results),
        "passed_symbols": list(passed_symbols),
        "passed_count": len(passed_symbols),
        "weighted_valid_win_rate_pct": (weighted_valid_wins / weighted_valid_trades * 100.0) if weighted_valid_trades > 0 else None,
        "weighted_valid_expectancy_rr": (weighted_valid_exp / weighted_valid_trades) if weighted_valid_trades > 0 else None,
        "total_valid_trades": int(weighted_valid_trades) if weighted_valid_trades > 0 else 0,
    }
    return {
        "symbols": [normalize_symbol(s) for s in symbols if normalize_symbol(s)],
        "passed_symbols": list(passed_symbols),
        "results": results,
        "summary": summary,
    }


def optimize_best_ema_cross_15m(symbol, df, direction_filter=None):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    direction_filter = str(direction_filter or "").upper().strip()
    if direction_filter not in ("", "BUY", "SELL"):
        direction_filter = ""
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
    min_valid_trades = getattr(config, "EMA_CROSS_15M_MIN_VALID_TRADES", 4)
    train_ratio = getattr(config, "EMA_CROSS_15M_OPT_TRAIN_RATIO", 0.70)
    max_evaluated = getattr(config, "EMA_CROSS_15M_OPT_MAX_EVALUATED", 180)
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
        min_valid_trades = int(min_valid_trades)
        train_ratio = float(train_ratio)
        max_evaluated = int(max_evaluated)
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
    if min_valid_trades < 1:
        min_valid_trades = 1
    if tp_mult <= 0:
        tp_mult = 5.0
    if max_forward < 1:
        max_forward = 64
    if train_ratio <= 0.50 or train_ratio >= 0.90:
        train_ratio = 0.70
    if max_evaluated < 1:
        max_evaluated = 96
    if len(df) < 180:
        train_ratio = 0.65
    split_idx = int(len(df) * train_ratio)
    split_idx = max(80, min(len(df) - 60, split_idx))
    df_train = df.iloc[:split_idx].copy()
    df_valid = df.iloc[split_idx:].copy()
    if len(df_train) < 80 or len(df_valid) < 60:
        return None
    combinations = []
    for fast_len in range(fast_min, fast_max + 1, fast_step):
        for slow_len in range(slow_min, slow_max + 1, slow_step):
            if fast_len < slow_len:
                combinations.append((fast_len, slow_len))
    if len(combinations) > max_evaluated:
        anchor_pairs = [(8, 21), (9, 21), (10, 30), (12, 26), (12, 36), (20, 50)]
        trimmed = []
        seen_pairs = set()
        valid_pairs = set(combinations)
        for pair in anchor_pairs:
            if pair in valid_pairs and pair not in seen_pairs:
                trimmed.append(pair)
                seen_pairs.add(pair)
        stride = max(1, int(math.ceil(len(combinations) / float(max_evaluated))))
        for pair in combinations[::stride]:
            if pair not in seen_pairs:
                trimmed.append(pair)
                seen_pairs.add(pair)
            if len(trimmed) >= max_evaluated:
                break
        combinations = trimmed[:max_evaluated]
    best = None
    evaluated = 0
    for fast_len, slow_len in combinations:
        try:
            r_train = _backtest_ema_cross_15m(df_train, fast_len, slow_len, tp_mult=tp_mult, max_forward=max_forward, direction_filter=direction_filter or None)
            r_valid = _backtest_ema_cross_15m(df_valid, fast_len, slow_len, tp_mult=tp_mult, max_forward=max_forward, direction_filter=direction_filter or None)
            r_full = _backtest_ema_cross_15m(df, fast_len, slow_len, tp_mult=tp_mult, max_forward=max_forward, direction_filter=direction_filter or None)
        except Exception as e:
            logger.warning(
                "EMA Cross optimizer failed for %s at fast=%s slow=%s: %s",
                normalize_symbol(symbol),
                fast_len,
                slow_len,
                e,
            )
            continue
        evaluated += 1
        if not r_train or not r_valid or not r_full:
            continue
        eval_info = _ema_cross_eval_candidate(r_train, r_valid, min_trades=min_trades, min_valid_trades=min_valid_trades)
        if not isinstance(eval_info, dict):
            continue
        trades = r_full.get("trades")
        exp_rr = r_full.get("expectancy_rr")
        win_rate = r_full.get("win_rate_pct")
        if not isinstance(exp_rr, (int, float)):
            continue
        candidate = r_full.copy()
        candidate.update(eval_info)
        key = (
            float(eval_info.get("opt_score")),
            float(eval_info.get("valid_expectancy_rr")),
            float(exp_rr),
            float(win_rate) if isinstance(win_rate, (int, float)) else -1e9,
            float(trades),
        )
        if best is None:
            best = candidate
            best["_key"] = key
        else:
            if key > best.get("_key", (-1e18, -1e18, -1e18)):
                best = candidate
                best["_key"] = key
    if best is None:
        return {
            "symbol": normalize_symbol(symbol),
            "direction_filter": direction_filter or "BOTH",
            "best": None,
            "evaluated": int(evaluated),
            "computed_at": get_thai_now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    best.pop("_key", None)
    return {
        "symbol": normalize_symbol(symbol),
        "direction_filter": direction_filter or "BOTH",
        "best": best,
        "evaluated": int(evaluated),
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
    s = str(symbol).strip().upper()
    # Users often paste symbols with a leading '$' (e.g. $BTC-USD).
    # Yahoo Finance expects raw tickers without '$'.
    while s.startswith("$"):
        s = s[1:].strip()
    return s


def _get_cdc_vixfix_15m_symbol_profile(symbol):
    sym = normalize_symbol(symbol)
    profiles = getattr(config, "CDC_VIXFIX_15M_SYMBOL_PROFILES", {}) or {}
    raw_profile = profiles.get(sym)
    profile = dict(raw_profile) if isinstance(raw_profile, dict) else {}
    disabled_symbols = getattr(config, "CDC_VIXFIX_15M_DISABLED_SYMBOLS", set()) or set()
    normalized_disabled = {normalize_symbol(item) for item in disabled_symbols if item}
    profile["enabled"] = sym not in normalized_disabled
    return profile


def _get_preferred_15m_history(symbol):
    sym = normalize_symbol(symbol)
    if not sym:
        return None
    yf_period = getattr(
        config,
        "EMA_CROSS_15M_YF_PERIOD",
        getattr(config, "SHORT_TERM_15M_YF_PERIOD", "30d"),
    )
    data_15m = get_yf_history(sym, period=str(yf_period), interval="15m", auto_adjust=True)
    if data_15m is None or data_15m.empty:
        data_15m = get_yf_history(sym, period="5d", interval="15m", auto_adjust=True)
    return data_15m


def _get_preferred_1h_history(symbol):
    sym = normalize_symbol(symbol)
    if not sym:
        return None
    trend_period = getattr(
        config,
        "SHORT_TERM_15M_TREND_1H_PERIOD",
        getattr(config, "ACTIONZONE_15M_TREND_1H_PERIOD", "3mo"),
    )
    data_1h = get_yf_history(sym, period=str(trend_period), interval="1h", auto_adjust=True)
    if data_1h is None or data_1h.empty:
        data_1h = get_yf_history(sym, period="1mo", interval="1h", auto_adjust=True)
    return data_1h

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

def get_basic_info(symbol):
    sym = normalize_symbol(symbol)
    if not sym:
        return {'name': '', 'sector': 'N/A', 'market_cap': 0, 'pe_ratio': 'N/A', 'dividend_yield': 0}
    cache_key = ("info", sym)
    cached = _YF_INFO_CACHE.get(cache_key)
    if isinstance(cached, dict) and cached:
        return dict(cached)
    if _prefer_chart_api():
        payload = {'name': sym, 'sector': 'N/A', 'market_cap': 0, 'pe_ratio': 'N/A', 'dividend_yield': 0}
        _YF_INFO_CACHE.set(cache_key, payload)
        return dict(payload)
    _configure_yf_tz_cache()
    for attempt in range(3):
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
        except Exception as e:
            msg = str(e).lower()
            if attempt == 0 and ("disk i/o error" in msg or "operationalerror" in msg):
                _configure_yf_tz_cache(force_temp=True)
                _set_thread_curl_session(None)
                continue
            if attempt < 2 and _is_yf_auth_error(msg):
                logger.warning("Resetting yfinance cookie/session cache after auth error for %s info: %s", sym, e)
                _clear_yf_runtime_cache(clear_tz_cache=True)
                continue
            if attempt == 0 and ("certificate verify locations" in msg or "curl: (77)" in msg):
                _set_thread_curl_session(
                    curl_requests.Session(
                        verify=False,
                        impersonate=getattr(config, "CURL_IMPERSONATE", "chrome110"),
                    )
                )
                continue
            payload = {'name': sym, 'sector': 'N/A', 'market_cap': 0, 'pe_ratio': 'N/A', 'dividend_yield': 0}
            _YF_INFO_CACHE.set(cache_key, payload)
            return dict(payload)
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
    def analyze_15m_setup(symbol, data_15m=None, data_1h=None):
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

            if not isinstance(data_15m, pd.DataFrame) or data_15m.empty:
                yf_period = getattr(config, "SHORT_TERM_15M_YF_PERIOD", "30d")
                data_15m = get_yf_history(sym, period=str(yf_period), interval="15m", auto_adjust=True)
                if data_15m is None or data_15m.empty:
                    data_15m = get_yf_history(sym, period="5d", interval="15m", auto_adjust=True)

            if not isinstance(data_1h, pd.DataFrame) or data_1h.empty:
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
    def analyze_sniper_setup(symbol, data_15m=None, data_1h=None):
        try:
            sym = normalize_symbol(symbol)
            if not isinstance(data_15m, pd.DataFrame) or data_15m.empty:
                data_15m = _get_preferred_15m_history(sym)
            if not isinstance(data_1h, pd.DataFrame) or data_1h.empty:
                data_1h = _get_preferred_1h_history(sym)
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
    def analyze(symbol, period="5d", data_15m=None):
        try:
            sym = normalize_symbol(symbol)
            if isinstance(data_15m, pd.DataFrame) and not data_15m.empty:
                df = data_15m
            else:
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
    def analyze(symbol, data_15m=None, data_1h=None):
        if not is_crypto_symbol(symbol):
            return None
        try:
            sym = normalize_symbol(symbol)
            if not isinstance(data_15m, pd.DataFrame) or data_15m.empty:
                data_15m = _get_preferred_15m_history(sym)
            if not isinstance(data_1h, pd.DataFrame) or data_1h.empty:
                data_1h = _get_preferred_1h_history(sym)
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
    def analyze(symbol, data_15m=None):
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
            if not isinstance(data_15m, pd.DataFrame) or data_15m.empty:
                yf_period = getattr(config, "EMA_CROSS_15M_YF_PERIOD", "30d")
                data_15m = get_yf_history(sym, period=str(yf_period), interval="15m", auto_adjust=True)
                if data_15m is None or data_15m.empty:
                    data_15m = get_yf_history(sym, period="5d", interval="15m", auto_adjust=True)
            if data_15m is None or data_15m.empty or len(data_15m) < 120:
                return None
            df = _ema_cross_15m_prepare_df(data_15m)
            if df is None or df.empty:
                return None
            use_opt = getattr(config, "EMA_CROSS_15M_ENABLE_OPTIMIZATION", True)
            best_fast_len, best_slow_len, opt_meta = _resolve_optimized_ema_lengths(sym, df, use_opt=use_opt)
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
                min_stop_pct = getattr(config, "EMA_CROSS_15M_MIN_STOP_PCT", 0.8)
                try:
                    min_stop_pct = float(min_stop_pct)
                except Exception:
                    min_stop_pct = 0.8
                if min_stop_pct < 0:
                    min_stop_pct = 0.0
                min_risk_dist = abs(entry_price) * (min_stop_pct / 100.0)
                if risk_dist < min_risk_dist:
                    risk_dist = min_risk_dist
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

def _actionzone_trend_1h(symbol, data_1h=None):
    try:
        sym = normalize_symbol(symbol)
        if isinstance(data_1h, pd.DataFrame) and not data_1h.empty:
            df = data_1h
        else:
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

def _actionzone_compute_confidence(
    signal,
    trend_dir,
    bars_since_signal,
    avg_range_pct,
    rvol,
    opt_meta,
    ema_gap_pct=None,
    atr_pct=None,
    trend_strength=None,
    adx_now=None,
):
    base = 55.0
    best = opt_meta.get("best") if isinstance(opt_meta, dict) else None
    trades = best.get("trades") if isinstance(best, dict) else None
    win_rate = best.get("win_rate_pct") if isinstance(best, dict) else None
    expectancy = best.get("expectancy_rr") if isinstance(best, dict) else None
    robustness = best.get("robustness_score") if isinstance(best, dict) else None
    valid_expectancy = best.get("valid_expectancy_rr") if isinstance(best, dict) else None
    consistency_penalty = best.get("consistency_penalty") if isinstance(best, dict) else None
    if isinstance(win_rate, (int, float)):
        base += max(-10.0, min(12.0, (float(win_rate) - 50.0) * 0.45))
    if isinstance(expectancy, (int, float)):
        base += max(-12.0, min(14.0, float(expectancy) * 18.0))
    if isinstance(valid_expectancy, (int, float)):
        base += max(-10.0, min(12.0, float(valid_expectancy) * 20.0))
    if isinstance(trades, (int, float)):
        base += max(-6.0, min(10.0, (float(trades) - 8.0) * 0.5))
    if isinstance(robustness, (int, float)):
        base += max(-8.0, min(8.0, (float(robustness) - 50.0) * 0.12))
    if isinstance(consistency_penalty, (int, float)):
        base -= max(0.0, min(8.0, float(consistency_penalty) * 8.0))
    if signal in ("BUY", "SELL"):
        base += 6.0
    trend_aligned = (
        trend_dir is None
        or (signal == "BUY" and trend_dir == "UP")
        or (signal == "SELL" and trend_dir == "DOWN")
    )
    if signal in ("BUY", "SELL") and trend_dir in ("UP", "DOWN"):
        base += 8.0 if trend_aligned else -12.0
    if isinstance(bars_since_signal, (int, float)):
        bars = int(bars_since_signal)
        if bars <= 0:
            base += 6.0
        elif bars == 1:
            base += 3.0
        elif bars > 2:
            base -= min(12.0, float((bars - 2) * 4))
    low_vol = getattr(config, "ACTIONZONE_15M_VOL_LOW_PCT", 0.35)
    high_vol = getattr(config, "ACTIONZONE_15M_VOL_HIGH_PCT", 3.50)
    try:
        low_vol = float(low_vol)
    except Exception:
        low_vol = 0.35
    try:
        high_vol = float(high_vol)
    except Exception:
        high_vol = 3.50
    if high_vol < low_vol:
        high_vol = low_vol
    if isinstance(avg_range_pct, (int, float)):
        if float(avg_range_pct) < low_vol:
            base -= 8.0
        elif float(avg_range_pct) > high_vol:
            base -= 6.0
    if isinstance(rvol, (int, float)):
        if float(rvol) >= 1.20:
            base += 4.0
        elif float(rvol) <= 0.80:
            base -= 5.0
    if isinstance(ema_gap_pct, (int, float)):
        if float(ema_gap_pct) >= 0.35:
            base += 4.0
        elif float(ema_gap_pct) <= 0.08:
            base -= 4.0
    if isinstance(atr_pct, (int, float)):
        if float(atr_pct) <= 0.15:
            base -= 5.0
        elif float(atr_pct) >= 5.50:
            base -= 4.0
    if isinstance(trend_strength, str) and trend_strength.upper() == "STRONG":
        base += 2.0
    if isinstance(adx_now, (int, float)):
        if float(adx_now) >= 25.0:
            base += 4.0
        elif float(adx_now) >= 18.0:
            base += 2.0
        elif float(adx_now) < 12.0:
            base -= 4.0
    if base < 0:
        base = 0.0
    if base > 100:
        base = 100.0
    return float(base)

def _actionzone_15m_alert(symbol, data_15m=None, data_1h=None):
    try:
        sym = normalize_symbol(symbol)
        if not isinstance(data_15m, pd.DataFrame) or data_15m.empty:
            yf_period = getattr(config, "ACTIONZONE_15M_YF_PERIOD", "30d")
            data_15m = get_yf_history(sym, period=str(yf_period), interval="15m", auto_adjust=True)
            if data_15m is None or data_15m.empty:
                data_15m = get_yf_history(sym, period="5d", interval="15m", auto_adjust=True)
        if data_15m is None or data_15m.empty or len(data_15m) < 120:
            return None
        df = _ema_cross_15m_prepare_df(data_15m)
        if df is None or df.empty:
            return None
        
        # Calculate performance metrics/average behavior
        avg_vol = float(df["Volume"].tail(20).mean())
        avg_range_pct = float(((df["High"] - df["Low"]) / df["Close"] * 100).tail(20).mean())
        
        use_opt = getattr(config, "ACTIONZONE_15M_USE_OPTIMIZATION", True)
        best_fast_len, best_slow_len, opt_meta = _resolve_optimized_ema_lengths(sym, df, use_opt=use_opt, profile_key="default")
        _, _, sell_opt_meta = _resolve_optimized_ema_lengths(sym, df, use_opt=use_opt, profile_key="sell_only", direction_filter="SELL")
            
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
        ema200 = x_confirm.ewm(span=200, adjust=False).mean()
        
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
        
        buy_signal = green & (~green.shift(1, fill_value=False))
        sell_signal = red & (~red.shift(1, fill_value=False))
        
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
                
        trend_1h = _actionzone_trend_1h(sym, data_1h=data_1h)
        trend_dir = trend_1h.get("trend") if isinstance(trend_1h, dict) else None
        trend_strength = trend_1h.get("strength") if isinstance(trend_1h, dict) else None
        entry_idx = len(close) - 1
        if last_signal_time is not None:
            try:
                entry_idx = int(df.index.get_loc(last_signal_time))
            except Exception:
                entry_idx = len(close) - 1
        entry_price = float(close.iloc[entry_idx]) if pd.notna(close.iloc[entry_idx]) else None
        current_price = float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else None
        atr_now = float(df["ATR"].iloc[-1]) if pd.notna(df["ATR"].iloc[-1]) else None
        adx_now = float(df["ADX"].iloc[-1]) if "ADX" in df.columns and pd.notna(df["ADX"].iloc[-1]) else None
        rvol = None
        vol_avg_now = df["Vol_Avg"].iloc[-1]
        if pd.notna(vol_avg_now) and float(vol_avg_now) > 0 and pd.notna(df["Volume"].iloc[-1]):
            rvol = float(df["Volume"].iloc[-1] / vol_avg_now)
        trend_ok = True
        if signal == "BUY" and trend_dir not in (None, "UP"):
            trend_ok = False
        if signal == "SELL" and trend_dir not in (None, "DOWN"):
            trend_ok = False
        ema_gap_pct = None
        if current_price is not None and current_price > 0 and pd.notna(fast.iloc[-1]) and pd.notna(slow.iloc[-1]):
            ema_gap_pct = abs(float(fast.iloc[-1]) - float(slow.iloc[-1])) / float(current_price) * 100.0
        atr_pct = None
        if atr_now is not None and current_price is not None and current_price > 0:
            atr_pct = float(atr_now) / float(current_price) * 100.0
        filter_reasons = []
        precision60 = _actionzone_precision60_profile()
        min_rvol = getattr(config, "ACTIONZONE_15M_MIN_RVOL", 1.15)
        min_ema_gap_pct = getattr(config, "ACTIONZONE_15M_MIN_EMA_GAP_PCT", 0.05)
        min_atr_pct = getattr(config, "ACTIONZONE_15M_MIN_ATR_PCT", 0.15)
        max_atr_pct = getattr(config, "ACTIONZONE_15M_MAX_ATR_PCT", 6.00)
        max_bars_since = getattr(config, "ACTIONZONE_15M_MAX_BARS_SINCE_SIGNAL", 1)
        require_strong_trend = bool(getattr(config, "ACTIONZONE_15M_REQUIRE_STRONG_TREND", True))
        min_adx = getattr(config, "ACTIONZONE_15M_MIN_ADX", 24.0)
        require_ema200_alignment = bool(getattr(config, "ACTIONZONE_15M_REQUIRE_EMA200_ALIGNMENT", True))
        try:
            min_rvol = float(min_rvol)
        except Exception:
            min_rvol = 1.15
        try:
            min_ema_gap_pct = float(min_ema_gap_pct)
        except Exception:
            min_ema_gap_pct = 0.05
        try:
            min_atr_pct = float(min_atr_pct)
        except Exception:
            min_atr_pct = 0.15
        try:
            max_atr_pct = float(max_atr_pct)
        except Exception:
            max_atr_pct = 6.00
        try:
            max_bars_since = int(max_bars_since)
        except Exception:
            max_bars_since = 1
        try:
            min_adx = float(min_adx)
        except Exception:
            min_adx = 24.0
        if isinstance(precision60, dict):
            min_rvol = max(float(min_rvol), float(precision60.get("min_rvol", min_rvol)))
            min_ema_gap_pct = max(float(min_ema_gap_pct), float(precision60.get("min_ema_gap_pct", min_ema_gap_pct)))
            min_adx = max(float(min_adx), float(precision60.get("min_adx", min_adx)))
            max_bars_since = min(int(max_bars_since), int(precision60.get("max_bars_since_signal", max_bars_since)))
            require_strong_trend = bool(require_strong_trend) or bool(precision60.get("require_strong_trend", False))
            require_ema200_alignment = bool(require_ema200_alignment) or bool(precision60.get("require_ema200_alignment", False))
        if max_bars_since < 0:
            max_bars_since = 0
        filtered_signal = signal if trend_ok else "WAIT"
        if filtered_signal in ("BUY", "SELL"):
            if isinstance(rvol, (int, float)) and float(rvol) < min_rvol:
                filter_reasons.append("low_rvol")
            if isinstance(ema_gap_pct, (int, float)) and float(ema_gap_pct) < min_ema_gap_pct:
                filter_reasons.append("tight_ema_gap")
            if isinstance(atr_pct, (int, float)):
                if float(atr_pct) < min_atr_pct:
                    filter_reasons.append("low_atr")
                elif float(atr_pct) > max_atr_pct:
                    filter_reasons.append("high_atr")
            if isinstance(bars_since_signal, (int, float)) and int(bars_since_signal) > max_bars_since:
                filter_reasons.append("stale_signal")
            if require_strong_trend and trend_strength != "STRONG":
                filter_reasons.append("weak_trend")
            if isinstance(adx_now, (int, float)) and float(adx_now) < min_adx:
                filter_reasons.append("low_adx")
            if require_ema200_alignment and pd.notna(ema200.iloc[-1]):
                ema200_now = float(ema200.iloc[-1])
                if filtered_signal == "BUY" and current_price is not None and float(current_price) < ema200_now:
                    filter_reasons.append("below_ema200")
                if filtered_signal == "SELL" and current_price is not None and float(current_price) > ema200_now:
                    filter_reasons.append("above_ema200")
            
            require_pattern = bool(getattr(config, "ACTIONZONE_15M_REQUIRE_PATTERN", True))
            pattern_lookback = getattr(config, "ACTIONZONE_15M_PATTERN_LOOKBACK_BARS", 3)
            pattern_mode = getattr(config, "ACTIONZONE_15M_ENTRY_CONFIRMATION_MODE", "engulfing_pinbar")
            pattern_ok = False
            pattern_name = "None"
            pattern_idx = None
            if filtered_signal in ("BUY", "SELL"):
                pattern_ok, pattern_name, pattern_idx = _recent_pattern_confirmation(
                    df,
                    len(df) - 1,
                    filtered_signal,
                    lookback_bars=pattern_lookback,
                    mode=pattern_mode,
                )
            if require_pattern and filtered_signal in ("BUY", "SELL") and not pattern_ok:
                if filtered_signal == "BUY":
                    filter_reasons.append("no_bullish_pattern")
                elif filtered_signal == "SELL":
                    filter_reasons.append("no_bearish_pattern")
            wf_ok, wf_reason, _ = _evaluate_walkforward_gate({"optimizer": opt_meta}, filtered_signal)
            if not wf_ok:
                filter_reasons.append(wf_reason)
            if filtered_signal == "SELL":
                sell_ok, sell_reason, _ = _evaluate_sell_whitelist_gate({"sell_optimizer": sell_opt_meta})
                if not sell_ok:
                    filter_reasons.append(sell_reason)
            
            if filter_reasons:
                filtered_signal = "WAIT"
        
        detected_pattern = pattern_name if filtered_signal in ("BUY", "SELL") and pattern_ok else "None"

        stop_mult = getattr(config, "ACTIONZONE_15M_STOP_ATR_MULT", 1.5)
        min_stop_pct = getattr(config, "ACTIONZONE_15M_MIN_STOP_PCT", 1.0)
        tp_mult = getattr(config, "ACTIONZONE_15M_TP_MULT", 5.0)
        try:
            stop_mult = float(stop_mult)
        except Exception:
            stop_mult = 1.5
        try:
            min_stop_pct = float(min_stop_pct)
        except Exception:
            min_stop_pct = 1.0
        try:
            tp_mult = float(tp_mult)
        except Exception:
            tp_mult = 5.0
        if stop_mult <= 0:
            stop_mult = 1.0
        if min_stop_pct < 0:
            min_stop_pct = 0.0
        if tp_mult <= 0:
            tp_mult = 5.0
        stop_loss = None
        take_profit = None
        risk_dist = None
        if entry_price is not None and atr_now is not None and atr_now > 0 and filtered_signal in ("BUY", "SELL"):
            risk_dist = max(float(atr_now) * stop_mult, abs(float(entry_price)) * (min_stop_pct / 100.0))
            if filtered_signal == "BUY":
                stop_loss = float(entry_price - risk_dist)
            else:
                stop_loss = float(entry_price + risk_dist)
            use_candle_stop = bool(getattr(config, "ACTIONZONE_15M_USE_CANDLE_STOP", True))
            candle_stop_buffer_atr = getattr(config, "ACTIONZONE_15M_CANDLE_STOP_BUFFER_ATR", 0.15)
            if use_candle_stop:
                stop_ref_idx = pattern_idx if isinstance(pattern_idx, int) else (len(df) - 1)
                candle_stop = _candle_based_risk(df, stop_ref_idx, filtered_signal, atr_now, stop_loss, buffer_atr=candle_stop_buffer_atr)
                if isinstance(candle_stop, (int, float)):
                    stop_loss = float(candle_stop)
                    risk_dist = abs(float(entry_price) - float(stop_loss))
            if risk_dist and risk_dist > 0:
                if filtered_signal == "BUY":
                    take_profit = float(entry_price + (risk_dist * tp_mult))
                else:
                    take_profit = float(entry_price - (risk_dist * tp_mult))
        confidence = _actionzone_compute_confidence(
            filtered_signal,
            trend_dir,
            bars_since_signal,
            avg_range_pct,
            rvol,
            opt_meta,
            ema_gap_pct=ema_gap_pct,
            atr_pct=atr_pct,
            trend_strength=trend_strength,
            adx_now=adx_now,
        )
        last_signal_time_str = last_signal_time.strftime("%Y-%m-%d %H:%M") if last_signal_time is not None else None
        sell_whitelist_ok, sell_whitelist_reason, _ = _evaluate_sell_whitelist_gate({"sell_optimizer": sell_opt_meta})
        return {
            "symbol": sym,
            "signal": filtered_signal,
            "raw_signal": raw_signal,
            "zone": zone_now,
            "fast_len": best_fast_len,
            "slow_len": best_slow_len,
            "current_price": current_price,
            "entry_price": entry_price if filtered_signal in ("BUY", "SELL") else None,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "bars_since_signal": bars_since_signal,
            "last_signal_time": last_signal_time_str,
            "trend_1h": trend_dir,
            "trend_strength": trend_strength,
            "trend_alignment": trend_ok,
            "filter_reasons": filter_reasons,
            "alert": filtered_signal in ("BUY", "SELL"),
            "confidence": confidence,
            "avg_range_pct": avg_range_pct,
            "avg_vol": avg_vol,
            "atr": atr_now,
            "atr_pct": atr_pct,
            "adx": adx_now,
            "ema_gap_pct": ema_gap_pct,
            "rvol": rvol,
            "optimizer": opt_meta,
            "sell_optimizer": sell_opt_meta,
            "sell_whitelist_ok": bool(sell_whitelist_ok),
            "sell_whitelist_reason": None if sell_whitelist_ok else sell_whitelist_reason,
            "detected_pattern": detected_pattern,
        }
    except Exception as e:
        print(f"Error in _actionzone_15m_alert for {symbol}: {e}")
        return None


def _cdc_vixfix_15m_plan(symbol, data_15m=None):
    try:
        sym = normalize_symbol(symbol)
        profile = _get_cdc_vixfix_15m_symbol_profile(sym)
        if not bool(profile.get("enabled", True)):
            return None
        if not isinstance(data_15m, pd.DataFrame) or data_15m.empty:
            yf_period = getattr(config, "CDC_VIXFIX_15M_YF_PERIOD", "30d")
            data_15m = get_yf_history(sym, period=str(yf_period), interval="15m", auto_adjust=True)
            if data_15m is None or data_15m.empty:
                data_15m = get_yf_history(sym, period="5d", interval="15m", auto_adjust=True)
        if data_15m is None or data_15m.empty or len(data_15m) < 80:
            return None
        df = data_15m[["Open", "High", "Low", "Close", "Volume"]].copy()
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = df.index.tz_localize(None)
            except Exception:
                pass
        df = _add_price_patterns(df)
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        fast_len = getattr(config, "CDC_VIXFIX_15M_FAST_EMA", 12)
        slow_len = getattr(config, "CDC_VIXFIX_15M_SLOW_EMA", 26)
        smooth_len = getattr(config, "CDC_VIXFIX_15M_SMOOTH", 1)
        rsi_len = getattr(config, "CDC_VIXFIX_15M_RSI_LENGTH", 14)
        stoch_len = getattr(config, "CDC_VIXFIX_15M_STOCH_LENGTH", 14)
        smooth_k = getattr(config, "CDC_VIXFIX_15M_STOCH_SMOOTH_K", 3)
        smooth_d = getattr(config, "CDC_VIXFIX_15M_STOCH_SMOOTH_D", 3)
        os_level = getattr(config, "CDC_VIXFIX_15M_STOCH_OVERSOLD", 30.0)
        ob_level = getattr(config, "CDC_VIXFIX_15M_STOCH_OVERBOUGHT", 70.0)
        vix_pd = getattr(config, "CDC_VIXFIX_15M_VIX_LOOKBACK", 22)
        vix_bbl = getattr(config, "CDC_VIXFIX_15M_VIX_BB_LENGTH", 20)
        vix_mult = getattr(config, "CDC_VIXFIX_15M_VIX_BB_STD", 2.0)
        vix_lb = getattr(config, "CDC_VIXFIX_15M_VIX_PERCENTILE_LOOKBACK", 50)
        vix_ph = getattr(config, "CDC_VIXFIX_15M_VIX_PERCENTILE_FACTOR", 0.85)
        vix_pl = getattr(config, "CDC_VIXFIX_15M_VIX_LOW_PERCENTILE_FACTOR", 1.01)
        vix_spike_lookback = getattr(config, "CDC_VIXFIX_15M_VIX_SPIKE_LOOKBACK_BARS", 3)
        alert_bars = getattr(config, "CDC_VIXFIX_15M_ALERT_BARS", 2)
        require_pattern = bool(getattr(config, "CDC_VIXFIX_15M_REQUIRE_PATTERN", True))
        pattern_lookback = getattr(config, "CDC_VIXFIX_15M_PATTERN_LOOKBACK_BARS", 3)
        pattern_mode = getattr(config, "CDC_VIXFIX_15M_ENTRY_CONFIRMATION_MODE", "engulfing_pinbar")
        use_candle_stop = bool(getattr(config, "CDC_VIXFIX_15M_USE_CANDLE_STOP", True))
        candle_stop_buffer_atr = getattr(config, "CDC_VIXFIX_15M_CANDLE_STOP_BUFFER_ATR", 0.15)
        relaxed_entry_enable = bool(getattr(config, "CDC_VIXFIX_15M_RELAXED_ENTRY_ENABLE", True))
        relaxed_stoch_max = getattr(config, "CDC_VIXFIX_15M_RELAXED_STOCH_MAX", 45.0)
        forecast_lookback = getattr(config, "CDC_VIXFIX_15M_FORECAST_MOMENTUM_LOOKBACK", 3)
        forecast_min_score = getattr(config, "CDC_VIXFIX_15M_FORECAST_MIN_SCORE", 60.0)
        require_ema200_alignment = bool(getattr(config, "CDC_VIXFIX_15M_REQUIRE_EMA200_ALIGNMENT", False))
        take_profit_pct = getattr(config, "CDC_VIXFIX_15M_TAKE_PROFIT_PCT", 0.0)
        max_hold_bars = getattr(config, "CDC_VIXFIX_15M_MAX_HOLD_BARS", 24)
        if profile:
            require_pattern = bool(profile.get("require_pattern", require_pattern))
            vix_spike_lookback = profile.get("vix_spike_lookback_bars", vix_spike_lookback)
            os_level = profile.get("stoch_oversold", os_level)
            forecast_min_score = profile.get("forecast_min_score", forecast_min_score)
            require_ema200_alignment = bool(profile.get("require_ema200_alignment", require_ema200_alignment))
            take_profit_pct = profile.get("take_profit_pct", take_profit_pct)
            max_hold_bars = profile.get("max_hold_bars", max_hold_bars)
            if "relaxed_entry_enable" in profile:
                relaxed_entry_enable = bool(profile.get("relaxed_entry_enable"))
        try:
            fast_len = max(2, int(fast_len))
            slow_len = max(fast_len + 1, int(slow_len))
            smooth_len = max(1, int(smooth_len))
            rsi_len = max(2, int(rsi_len))
            stoch_len = max(2, int(stoch_len))
            smooth_k = max(1, int(smooth_k))
            smooth_d = max(1, int(smooth_d))
            os_level = float(os_level)
            ob_level = float(ob_level)
            vix_pd = max(2, int(vix_pd))
            vix_bbl = max(2, int(vix_bbl))
            vix_mult = max(0.1, float(vix_mult))
            vix_lb = max(2, int(vix_lb))
            vix_ph = float(vix_ph)
            vix_pl = max(1.0, float(vix_pl))
            vix_spike_lookback = max(0, int(vix_spike_lookback))
            alert_bars = max(0, int(alert_bars))
            relaxed_stoch_max = float(relaxed_stoch_max)
            forecast_lookback = max(1, int(forecast_lookback))
            forecast_min_score = float(forecast_min_score)
            take_profit_pct = max(0.0, float(take_profit_pct))
            max_hold_bars = max(1, int(max_hold_bars))
        except Exception:
            return None
        xprice = close.ewm(span=smooth_len, adjust=False).mean() if smooth_len > 1 else close.copy()
        fast = xprice.ewm(span=fast_len, adjust=False).mean()
        slow = xprice.ewm(span=slow_len, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        bull = fast > slow
        bear = fast < slow
        green = bull & (xprice > fast)
        red = bear & (xprice < fast)
        buycond = green & (~green.shift(1, fill_value=False))
        sellcond = red & (~red.shift(1, fill_value=False))
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.rolling(rsi_len).mean()
        avg_loss = loss.rolling(rsi_len).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_low = rsi.rolling(stoch_len).min()
        rsi_high = rsi.rolling(stoch_len).max()
        stoch_rsi = ((rsi - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)) * 100.0
        k = stoch_rsi.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()
        cross_up = (k > d) & (k.shift(1) <= d.shift(1))
        cross_down = (k < d) & (k.shift(1) >= d.shift(1))
        storsi_buy_sig = np.where(d < os_level, np.where(cross_up, 2, 0), np.where(cross_up & (d > os_level), 1, 0))
        storsi_buy_sig = pd.Series(np.where(bull, storsi_buy_sig, 0), index=df.index)
        storsi_sell_sig = np.where(d > ob_level, np.where(cross_down, 2, 0), np.where(cross_down & (d < ob_level), 1, 0))
        storsi_sell_sig = pd.Series(np.where(bear, storsi_sell_sig, 0), index=df.index)
        highest_close = close.rolling(vix_pd).max()
        wvf = ((highest_close - low) / highest_close.replace(0, np.nan)) * 100.0
        sdev = vix_mult * wvf.rolling(vix_bbl).std()
        mid = wvf.rolling(vix_bbl).mean()
        upper_band = mid + sdev
        range_high = wvf.rolling(vix_lb).max() * vix_ph
        range_low = wvf.rolling(vix_lb).min() * vix_pl
        vixfix_spike = (wvf >= upper_band) | (wvf >= range_high)
        recent_vixfix_spike = vixfix_spike.rolling(vix_spike_lookback + 1, min_periods=1).max().fillna(0).astype(bool)
        forecast = _cdc_forecast_bias(
            close,
            fast,
            slow,
            k,
            d,
            bool(vixfix_spike.iloc[-1]) if len(vixfix_spike) else False,
            momentum_lookback=forecast_lookback,
        )
        forecast_dir = str(forecast.get("direction") or "NEUTRAL").upper()
        forecast_score = float(forecast.get("score") or 50.0)
        fast_slope_up = fast.diff(forecast_lookback) > 0
        fast_slope_down = fast.diff(forecast_lookback) < 0
        ema200_buy_ok = (close > ema200) if require_ema200_alignment else pd.Series(True, index=df.index)
        strict_buy_condition = buycond & recent_vixfix_spike & (storsi_buy_sig > 0) & ema200_buy_ok.fillna(False)
        relaxed_long_condition = (
            green
            & recent_vixfix_spike
            & (k > d)
            & (d < relaxed_stoch_max)
            & fast_slope_up.fillna(False)
            & ema200_buy_ok.fillna(False)
        )
        strict_sell_condition = sellcond & ((storsi_sell_sig > 0) | cross_down.fillna(False))
        relaxed_sell_condition = red & (k < d) & fast_slope_down.fillna(False)
        buy_ready = strict_buy_condition | (relaxed_long_condition if relaxed_entry_enable else False)
        entry_idx = None
        if buy_ready.any():
            entry_idx = buy_ready[buy_ready].index[-1]
        bars_since_entry = None
        if entry_idx is not None:
            try:
                idx_pos = df.index.get_loc(entry_idx)
                bars_since_entry = max(0, len(df) - idx_pos - 1)
            except Exception:
                bars_since_entry = None
        entry_recent = entry_idx is not None and bars_since_entry is not None and bars_since_entry <= alert_bars
        entry_pos = None
        if entry_idx is not None:
            try:
                entry_pos = int(df.index.get_loc(entry_idx))
            except Exception:
                entry_pos = None
        buy_pattern_ok = False
        buy_pattern_name = "None"
        buy_pattern_idx = None
        if isinstance(entry_pos, int):
            buy_pattern_ok, buy_pattern_name, buy_pattern_idx = _recent_pattern_confirmation(
                df,
                entry_pos,
                "BUY",
                lookback_bars=pattern_lookback,
                mode=pattern_mode,
            )
        sell_pattern_ok, sell_pattern_name, sell_pattern_idx = _recent_pattern_confirmation(
            df,
            len(df) - 1,
            "SELL",
            lookback_bars=pattern_lookback,
            mode=pattern_mode,
        )
        entry_price = float(close.loc[entry_idx]) if entry_idx is not None and pd.notna(close.loc[entry_idx]) else None
        stop_loss = None
        if entry_idx is not None:
            try:
                pos = int(df.index.get_loc(entry_idx))
                start = max(0, pos - 4)
                sl_val = low.iloc[start:pos + 1].min()
                if pd.notna(sl_val):
                    stop_loss = float(sl_val)
            except Exception:
                stop_loss = None
        current_price = float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else None
        atr = None
        tr1 = (high - low).abs()
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean()
        if pd.notna(atr_series.iloc[-1]):
            atr = float(atr_series.iloc[-1])
        min_sl_pct = getattr(config, "CDC_VIXFIX_15M_MIN_STOP_PCT", 1.0)
        sl_atr_mult = getattr(config, "CDC_VIXFIX_15M_SL_ATR_MULT", 2.0)
        try:
            min_sl_pct = float(min_sl_pct)
        except Exception:
            min_sl_pct = 1.0
        try:
            sl_atr_mult = float(sl_atr_mult)
        except Exception:
            sl_atr_mult = 2.0
        if min_sl_pct < 0:
            min_sl_pct = 0.0
        if sl_atr_mult <= 0:
            sl_atr_mult = 2.0
        if entry_price is not None:
            min_stop_price = entry_price - (abs(entry_price) * (min_sl_pct / 100.0))
            atr_stop_price = None
            if atr is not None and atr > 0:
                atr_stop_price = entry_price - (atr * sl_atr_mult)
            candidate_stops = [x for x in (stop_loss, min_stop_price, atr_stop_price) if isinstance(x, (int, float))]
            if candidate_stops:
                stop_loss = float(min(candidate_stops))
        last_time_str = entry_idx.strftime("%Y-%m-%d %H:%M") if entry_idx is not None else None
        signal = "WAIT"
        reason = "ยังไม่เกิดจังหวะเข้าตาม CDC Action Zone + Williams Vix Fix บนกรอบ 15 นาที"
        sell_trigger = None
        detected_pattern = "None"
        take_profit_price = None
        if entry_price is not None and take_profit_pct > 0:
            take_profit_price = entry_price * (1.0 + (take_profit_pct / 100.0))
        if entry_recent:
            allow_buy = forecast_dir == "BUY" or forecast_score >= forecast_min_score
            if (not require_pattern or buy_pattern_ok) and allow_buy:
                signal = "BUY"
                if strict_buy_condition.iloc[-1]:
                    reason = "CDC เข้าสีเขียวแท่งแรก พร้อม Williams Vix Fix spike และ StochRSI ตัดขึ้น"
                else:
                    reason = "CDC ยังหนุนขาขึ้น, เพิ่งมี VixFix spike ฝั่ง panic low และโมเมนตัมเริ่มฟื้น"
                detected_pattern = buy_pattern_name if buy_pattern_ok else "None"
        elif entry_idx is not None and bars_since_entry is not None:
            if take_profit_price is not None and current_price is not None and current_price >= take_profit_price:
                signal = "SELL"
                reason = f"ราคาแตะเป้าปิดทำกำไรตามแผน CDC ที่ {take_profit_pct:.2f}% จากจุดเข้า"
                sell_trigger = "TAKE_PROFIT"
            elif max_hold_bars and bars_since_entry >= max_hold_bars:
                signal = "SELL"
                reason = f"ถือครบ {int(max_hold_bars)} แท่งของแผน CDC แล้ว จึงปิดรอบเพื่อลดการยืดเยื้อ"
                sell_trigger = "TIME_STOP"
            elif bool(strict_sell_condition.iloc[-1]):
                if not require_pattern or sell_pattern_ok:
                    signal = "SELL"
                    reason = "CDC พลิกเป็น Red แท่งแรกและ StochRSI เริ่มตัดลง เป็นจังหวะลดความเสี่ยง"
                    sell_trigger = "CDC_RED_REVERSAL"
                    detected_pattern = sell_pattern_name if sell_pattern_ok else "None"
            elif relaxed_entry_enable and bool(relaxed_sell_condition.iloc[-1]) and forecast_dir == "SELL" and forecast_score >= forecast_min_score:
                if not require_pattern or sell_pattern_ok:
                    signal = "SELL"
                    reason = "แนวโน้มสั้นอ่อนแรง, โซน CDC กลับเป็นลบ และโมเมนตัมเอนลง"
                    sell_trigger = "TREND_ROLLOVER"
                    detected_pattern = sell_pattern_name if sell_pattern_ok else "None"
        if signal == "BUY" and entry_price is not None:
            min_stop_price = entry_price - (abs(entry_price) * (min_sl_pct / 100.0))
            atr_stop_price = None
            if atr is not None and atr > 0:
                atr_stop_price = entry_price - (atr * sl_atr_mult)
            candidate_stops = [x for x in (stop_loss, min_stop_price, atr_stop_price) if isinstance(x, (int, float))]
            if candidate_stops:
                stop_loss = float(min(candidate_stops))
            if use_candle_stop:
                stop_ref_idx = buy_pattern_idx if isinstance(buy_pattern_idx, int) else (entry_pos if isinstance(entry_pos, int) else None)
                if isinstance(stop_ref_idx, int):
                    candle_stop = _candle_based_risk(df, stop_ref_idx, "BUY", atr, stop_loss, buffer_atr=candle_stop_buffer_atr)
                    if isinstance(candle_stop, (int, float)):
                        stop_loss = float(candle_stop)
        elif signal == "SELL":
            sell_entry_price = current_price if current_price is not None else entry_price
            if sell_entry_price is not None:
                min_stop_price = sell_entry_price + (abs(sell_entry_price) * (min_sl_pct / 100.0))
                atr_stop_price = None
                if atr is not None and atr > 0:
                    atr_stop_price = sell_entry_price + (atr * sl_atr_mult)
                candidate_stops = [x for x in (min_stop_price, atr_stop_price) if isinstance(x, (int, float))]
                if candidate_stops:
                    stop_loss = float(max(candidate_stops))
                if use_candle_stop:
                    stop_ref_idx = sell_pattern_idx if isinstance(sell_pattern_idx, int) else (len(df) - 1)
                    candle_stop = _candle_based_risk(df, stop_ref_idx, "SELL", atr, stop_loss, buffer_atr=candle_stop_buffer_atr)
                    if isinstance(candle_stop, (int, float)):
                        stop_loss = float(candle_stop)
        trend_bias = "ขาขึ้น" if bool(bull.iloc[-1]) else "ขาลง" if bool(bear.iloc[-1]) else "แกว่งตัว"
        analysis_points = []
        if signal == "BUY":
            if bool(bull.iloc[-1]):
                analysis_points.append(f"CDC อยู่ในโซนเขียว: EMA {fast_len}/{slow_len} เรียงตัวขาขึ้น และราคายืนเหนือ EMA เร็ว")
            if require_ema200_alignment and pd.notna(ema200.iloc[-1]):
                analysis_points.append(f"ราคายังยืนเหนือ EMA200 ที่ {float(ema200.iloc[-1]):,.2f} เพื่อกรองเทรนด์ใหญ่")
            if bool(vixfix_spike.iloc[-1]):
                analysis_points.append("Williams Vix Fix เด้งเหนือ upper band/range high บอกภาวะ panic low หรือแรงขายสุดโต่ง")
            elif bool(recent_vixfix_spike.iloc[-1]):
                analysis_points.append(f"เพิ่งเกิด Williams Vix Fix spike ภายใน {vix_spike_lookback} แท่งล่าสุด")
            if pd.notna(k.iloc[-1]) and pd.notna(d.iloc[-1]):
                if bool(cross_up.iloc[-1]) or int(storsi_buy_sig.iloc[-1]) > 0:
                    analysis_points.append(f"StochRSI ตัดขึ้นใหม่ โดย K/D = {float(k.iloc[-1]):.1f}/{float(d.iloc[-1]):.1f}")
                else:
                    analysis_points.append(f"โมเมนตัมยังหนุนฝั่งซื้อ โดย K/D = {float(k.iloc[-1]):.1f}/{float(d.iloc[-1]):.1f}")
        elif signal == "SELL":
            if bool(sellcond.iloc[-1]):
                analysis_points.append(f"CDC พลิกเป็น RED แท่งแรก และราคาปิดต่ำกว่า EMA {fast_len}")
            elif relaxed_entry_enable and bool(relaxed_sell_condition.iloc[-1]):
                analysis_points.append(f"EMA {fast_len} ชันลงต่อเนื่องและราคาหลุด EMA เร็ว")
            elif sell_trigger == "PRECISION60_TAKE_PROFIT" and take_profit_price is not None:
                analysis_points.append(f"ราคาแตะเป้าปิดทำกำไรที่ {take_profit_price:,.4f} ตามแผน Precision60")
            elif sell_trigger == "PRECISION60_TIME_STOP":
                analysis_points.append(f"ถือครบ {int(max_hold_bars)} แท่งแล้ว จึงปิดรอบเพื่อลดโอกาสคืนกำไร")
            if pd.notna(k.iloc[-1]) and pd.notna(d.iloc[-1]):
                if bool(cross_down.iloc[-1]) or int(storsi_sell_sig.iloc[-1]) > 0:
                    analysis_points.append(f"StochRSI ตัดลง โดย K/D = {float(k.iloc[-1]):.1f}/{float(d.iloc[-1]):.1f}")
                else:
                    analysis_points.append(f"StochRSI อ่อนแรง โดย K/D = {float(k.iloc[-1]):.1f}/{float(d.iloc[-1]):.1f}")
        if forecast_dir:
            if forecast_dir == signal:
                analysis_points.append(f"Forecast สอดคล้องกับสัญญาณฝั่ง {signal} ({float(forecast_score):.0f}/100)")
            else:
                analysis_points.append(f"Forecast ล่าสุดเอนทาง {forecast_dir} ({float(forecast_score):.0f}/100)")
        if isinstance(bars_since_entry, (int, float)):
            analysis_points.append(f"สัญญาณเกิดมาแล้ว {int(bars_since_entry)} แท่งบนกรอบ 15m")
        if detected_pattern and detected_pattern != "None":
            analysis_points.append(f"มี pattern ยืนยัน: {detected_pattern}")
        conf = 45.0
        if signal == "BUY":
            conf += 25.0
        if signal == "SELL":
            conf += 20.0
        if pd.notna(k.iloc[-1]) and pd.notna(d.iloc[-1]):
            if signal == "BUY" and float(k.iloc[-1]) > float(d.iloc[-1]):
                conf += 10.0
            if signal == "SELL" and float(k.iloc[-1]) < float(d.iloc[-1]):
                conf += 8.0
        if signal == "BUY" and bool(recent_vixfix_spike.iloc[-1]):
            conf += 8.0
        if signal == "SELL" and bool(sellcond.iloc[-1]):
            conf += 6.0
        conf += max(0.0, min(12.0, (forecast_score - 50.0) * 0.30))
        if forecast_dir == signal:
            conf += 6.0
        if conf > 95:
            conf = 95.0
        return {
            "signal": signal,
            "setup": "CDC Action Zone + Williams Vix Fix 15m",
            "entry_price": entry_price,
            "current_price": current_price,
            "stop_loss": stop_loss,
            "confidence": conf,
            "last_signal_time": last_time_str,
            "bars_since_entry": bars_since_entry,
            "is_market_top": False,
            "is_vixfix_spike": bool(vixfix_spike.iloc[-1]) if len(vixfix_spike) else False,
            "sell_trigger": sell_trigger,
            "trend_color": "GREEN" if bool(green.iloc[-1]) else "RED" if bool(red.iloc[-1]) else "NEUTRAL",
            "trend_bias": trend_bias,
            "forecast_direction": forecast_dir,
            "forecast_score": forecast_score,
            "forecast_reason": forecast.get("reason"),
            "stoch_k": float(k.iloc[-1]) if pd.notna(k.iloc[-1]) else None,
            "stoch_d": float(d.iloc[-1]) if pd.notna(d.iloc[-1]) else None,
            "wvf": float(wvf.iloc[-1]) if pd.notna(wvf.iloc[-1]) else None,
            "wvf_upper_band": float(upper_band.iloc[-1]) if pd.notna(upper_band.iloc[-1]) else None,
            "wvf_range_high": float(range_high.iloc[-1]) if pd.notna(range_high.iloc[-1]) else None,
            "wvf_range_low": float(range_low.iloc[-1]) if pd.notna(range_low.iloc[-1]) else None,
            "take_profit_price": float(take_profit_price) if isinstance(take_profit_price, (int, float)) else None,
            "max_hold_bars": max_hold_bars,
            "reason": reason,
            "detected_pattern": detected_pattern,
            "analysis_points": analysis_points,
        }
    except Exception:
        return None

def _order_block_levels_15m(symbol, data_15m=None):
    try:
        sym = normalize_symbol(symbol)
        if not isinstance(data_15m, pd.DataFrame) or data_15m.empty:
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
    def get_data_4h(symbol, data_1h=None):
        try:
            sym = normalize_symbol(symbol)
            if isinstance(data_1h, pd.DataFrame) and not data_1h.empty:
                df = data_1h
            else:
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
    def analyze(symbol, balance=10000, baseline_ema_length=None, data_1h=None):
        df = QuantumSovereign4H.get_data_4h(symbol, data_1h=data_1h)
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


def build_prediction_summary(short_term_plan, sniper_plan, quantum_plan, ema_plan, actionzone_plan, sovereign_plan, resonance_score, phase_status, crypto_plan=None, cdc_plan=None):
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
    if isinstance(actionzone_plan, dict):
        az_proxy = {
            "setup": actionzone_plan.get("signal"),
            "confidence": actionzone_plan.get("confidence"),
        }
        process_plan(az_proxy)
        add_conf(actionzone_plan.get("confidence"))
    if isinstance(cdc_plan, dict):
        cdc_signal = _normalize_trade_direction(cdc_plan.get("signal"))
        if cdc_signal in ("BUY", "SELL"):
            cdc_proxy = {
                "setup": cdc_signal,
                "confidence": cdc_plan.get("confidence"),
            }
            process_plan(cdc_proxy)
            add_conf(cdc_plan.get("confidence"))
    if isinstance(crypto_plan, dict):
        process_plan(crypto_plan)
    if isinstance(sovereign_plan, dict):
        reco = str(sovereign_plan.get("recommendation", "")).upper()
        if "BUY" in reco:
            up_score += 1.5
        elif "SELL" in reco:
            down_score += 1.5
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
    return {
        "direction": direction,
        "probability": probability,
        "expected_move_pct": expected_move_pct,
        "expected_holding_hours": expected_holding_hours,
        "short_term_signal": short_term_signal,
        "phase_status": phase_status,
    }


def build_price_forecast(current_price, prediction, support=None, resistance=None, actionzone_plan=None, ema_plan=None, cdc_plan=None):
    try:
        price = float(current_price)
    except Exception:
        return None
    if price <= 0:
        return None
    candidate = _pick_price_forecast_candidate(
        ("ActionZone 15m", actionzone_plan),
        ("EMA Cross 15m", ema_plan),
        ("CDC+VixFix 15m", cdc_plan),
    )
    direction = candidate.get("direction") if isinstance(candidate, dict) else None
    confidence = candidate.get("confidence") if isinstance(candidate, dict) else None
    source = candidate.get("label") if isinstance(candidate, dict) else None
    plan = candidate.get("plan") if isinstance(candidate, dict) else None
    if direction not in ("BUY", "SELL") and isinstance(prediction, dict):
        direction = _normalize_trade_direction(prediction.get("direction"))
    if confidence is None and isinstance(prediction, dict):
        confidence = _normalize_confidence(prediction.get("probability"))
    if confidence is None:
        confidence = 50.0
    horizon_hours = None
    if isinstance(plan, dict):
        bars_15m = plan.get("expected_holding_bars_15m")
        if isinstance(bars_15m, (int, float)) and bars_15m > 0:
            horizon_hours = float(bars_15m) * 0.25
    if horizon_hours is None and isinstance(prediction, dict):
        ph = prediction.get("expected_holding_hours")
        if isinstance(ph, (int, float)) and ph > 0:
            horizon_hours = float(ph)
    if horizon_hours is None:
        horizon_hours = getattr(config, "PRICE_FORECAST_DEFAULT_HOURS", 6.0)
    try:
        horizon_hours = float(horizon_hours)
    except Exception:
        horizon_hours = 6.0
    if horizon_hours <= 0:
        horizon_hours = 6.0
    atr_val = None
    if isinstance(plan, dict):
        atr_val = plan.get("atr")
    try:
        atr_val = float(atr_val)
    except Exception:
        atr_val = None
    if atr_val is None or atr_val <= 0:
        atr_val = abs(price) * 0.01
    base_atr_mult = getattr(config, "PRICE_FORECAST_BASE_ATR_MULT", 1.2)
    stretch_atr_mult = getattr(config, "PRICE_FORECAST_STRETCH_ATR_MULT", 2.4)
    try:
        base_atr_mult = float(base_atr_mult)
    except Exception:
        base_atr_mult = 1.2
    try:
        stretch_atr_mult = float(stretch_atr_mult)
    except Exception:
        stretch_atr_mult = 2.4
    if base_atr_mult <= 0:
        base_atr_mult = 1.2
    if stretch_atr_mult < base_atr_mult:
        stretch_atr_mult = base_atr_mult * 1.5
    confidence_factor = max(0.8, min(1.25, 0.8 + (float(confidence) / 100.0) * 0.45))
    expected_move_pct = None
    if isinstance(plan, dict) and isinstance(plan.get("expected_move_pct"), (int, float)):
        expected_move_pct = abs(float(plan.get("expected_move_pct")))
    elif isinstance(prediction, dict) and isinstance(prediction.get("expected_move_pct"), (int, float)):
        expected_move_pct = abs(float(prediction.get("expected_move_pct")))
    support_val = None
    resistance_val = None
    try:
        support_val = float(support)
    except Exception:
        support_val = None
    try:
        resistance_val = float(resistance)
    except Exception:
        resistance_val = None
    stop_loss = plan.get("stop_loss") if isinstance(plan, dict) else None
    take_profit = plan.get("take_profit") if isinstance(plan, dict) else None
    try:
        stop_loss = float(stop_loss)
    except Exception:
        stop_loss = None
    try:
        take_profit = float(take_profit)
    except Exception:
        take_profit = None
    move_from_atr = atr_val * base_atr_mult * confidence_factor
    stretch_from_atr = atr_val * stretch_atr_mult * confidence_factor
    if isinstance(expected_move_pct, (int, float)) and expected_move_pct > 0:
        pct_move = price * (float(expected_move_pct) / 100.0)
        move_from_atr = max(move_from_atr, pct_move * 0.55)
        stretch_from_atr = max(stretch_from_atr, pct_move)
    direction = direction or "NEUTRAL"
    if direction == "BUY":
        base_price = price + move_from_atr
        stretch_price = price + stretch_from_atr
        if isinstance(take_profit, float) and take_profit > price:
            base_price = (base_price + take_profit) / 2.0
            stretch_price = max(stretch_price, take_profit)
        if isinstance(resistance_val, float) and resistance_val > price:
            base_price = min(base_price, resistance_val)
        range_low = stop_loss if isinstance(stop_loss, float) and stop_loss < price else price - (atr_val * 0.6)
        range_high = stretch_price
        invalidation_price = stop_loss if isinstance(stop_loss, float) else range_low
        decision_bias = "ให้น้ำหนักฝั่งขึ้นเมื่อราคาไม่หลุดจุดยกเลิกมุมมอง"
    elif direction == "SELL":
        base_price = price - move_from_atr
        stretch_price = price - stretch_from_atr
        if isinstance(take_profit, float) and take_profit < price:
            base_price = (base_price + take_profit) / 2.0
            stretch_price = min(stretch_price, take_profit)
        if isinstance(support_val, float) and support_val < price:
            base_price = max(base_price, support_val)
        range_low = stretch_price
        range_high = stop_loss if isinstance(stop_loss, float) and stop_loss > price else price + (atr_val * 0.6)
        invalidation_price = stop_loss if isinstance(stop_loss, float) else range_high
        decision_bias = "ให้น้ำหนักฝั่งลงเมื่อราคาไม่ทะลุจุดยกเลิกมุมมอง"
    else:
        base_price = price
        range_low = support_val if isinstance(support_val, float) and support_val < price else price - move_from_atr
        range_high = resistance_val if isinstance(resistance_val, float) and resistance_val > price else price + move_from_atr
        invalidation_price = None
        decision_bias = "ตลาดยังไม่ชี้ทางชัด รอ breakout หรือ breakdown"
    forecast = {
        "direction": direction,
        "source": source or "Prediction Engine",
        "confidence": float(confidence),
        "current_price": float(price),
        "base_price": float(base_price),
        "range_low": float(min(range_low, range_high)),
        "range_high": float(max(range_low, range_high)),
        "invalidation_price": float(invalidation_price) if isinstance(invalidation_price, (int, float)) else None,
        "expected_move_pct": float(expected_move_pct) if isinstance(expected_move_pct, (int, float)) else None,
        "horizon_hours": float(horizon_hours),
        "decision_bias": decision_bias,
    }
    return forecast

# --- Main Analysis Logic ---

def analyze_single_symbol(symbol, period, include_chart_data=True):
    try:
        sym = normalize_symbol(symbol)
        shared_15m = _get_preferred_15m_history(sym)
        shared_1h = _get_preferred_1h_history(sym)
        data = get_stock_data(symbol, period)
        if data is None:
            return {"symbol": symbol, "error": "No Data"}
        data = calculate_technical_indicators(data)
        latest = data.iloc[-1]
        resonance_score = ParticleAAnalyzer.calculate_resonance_score(data)
        phase_status = ParticleAAnalyzer.interpret_phase(resonance_score)
        support = data["Low"].min()
        resistance = data["High"].max()
        vol_avg = data["Volume"].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else data["Volume"].mean()
        vol_status = "High" if latest["Volume"] > vol_avg * 1.2 else "Low" if latest["Volume"] < vol_avg * 0.8 else "Normal"
        info = get_basic_info(symbol)
        signal, ai_reason = generate_gemini_particle_a_analysis(info, data, resonance_score, phase_status, support, resistance, vol_status)
        chart_data = None
        if include_chart_data:
            chart_data = {
                "dates": [dt.strftime("%Y-%m-%d %H:%M") if period in ["1h", "15m"] else dt.strftime("%Y-%m-%d") for dt in data.index],
                "close": data["Close"].fillna(0).tolist(),
                "sma20": data["SMA20"].fillna(0).tolist(),
                "bb_upper": data["BB_Upper"].fillna(0).tolist(),
                "bb_lower": data["BB_Lower"].fillna(0).tolist(),
            }
        short_term_plan = ShortTermStrategy.analyze_15m_setup(symbol, data_15m=shared_15m, data_1h=shared_1h)
        sniper_plan = ShortTermStrategy.analyze_sniper_setup(symbol, data_15m=shared_15m, data_1h=shared_1h)
        quantum_plan = QuantumHunterStrategy.analyze(symbol, data_15m=shared_15m)
        sovereign_4h_plan = QuantumSovereign4H.analyze(symbol, data_1h=shared_1h)
        crypto_reversal_plan = CryptoReversal15m.analyze(symbol, data_15m=shared_15m, data_1h=shared_1h)
        ema_cross_plan = EMACross15m.analyze(symbol, data_15m=shared_15m)
        actionzone_plan = _actionzone_15m_alert(symbol, data_15m=shared_15m, data_1h=shared_1h)
        cdc_vixfix_plan = _cdc_vixfix_15m_plan(symbol, data_15m=shared_15m)
        order_blocks_15m = _order_block_levels_15m(symbol, data_15m=shared_15m)
        exit_context = {
            "support": float(support) if pd.notna(support) else None,
            "resistance": float(resistance) if pd.notna(resistance) else None,
            "volume_status": vol_status,
        }
        short_term_plan = _attach_exit_levels(
            short_term_plan,
            signal="BUY",
            entry_keys=["current_price", "entry_price", "price"],
            stop_keys=["stop_loss"],
            tp_keys=["take_profit_2", "take_profit"],
            context=exit_context,
        )
        sniper_plan = _attach_exit_levels(
            sniper_plan,
            entry_keys=["current_price", "entry_price", "price"],
            stop_keys=["stop_loss"],
            tp_keys=["take_profit"],
            context=exit_context,
        )
        quantum_plan = _attach_exit_levels(
            quantum_plan,
            entry_keys=["current_price", "entry_price", "price"],
            stop_keys=["stop_loss"],
            tp_keys=["take_profit"],
            context=exit_context,
        )
        ema_cross_plan = _attach_exit_levels(
            ema_cross_plan,
            entry_keys=["entry_price", "current_price", "price"],
            stop_keys=["stop_loss"],
            tp_keys=["take_profit"],
            context=exit_context,
        )
        actionzone_plan = _attach_exit_levels(
            actionzone_plan,
            entry_keys=["entry_price", "current_price", "price"],
            stop_keys=["stop_loss"],
            tp_keys=["take_profit", "exit_price"],
            context=exit_context,
        )
        crypto_reversal_plan = _attach_exit_levels(
            crypto_reversal_plan,
            entry_keys=["current_price", "entry_price", "price"],
            stop_keys=["stop_loss"],
            tp_keys=["take_profit", "smc_take_profit"],
            context=exit_context,
        )
        cdc_vixfix_plan = _attach_exit_levels(
            cdc_vixfix_plan,
            entry_keys=["entry_price", "current_price", "price"],
            stop_keys=["stop_loss"],
            tp_keys=["take_profit_price", "take_profit"],
            context=exit_context,
        )
        prediction = build_prediction_summary(
            short_term_plan,
            sniper_plan,
            quantum_plan,
            ema_cross_plan,
            actionzone_plan,
            sovereign_4h_plan,
            resonance_score,
            phase_status,
            crypto_reversal_plan,
            cdc_vixfix_plan,
        )
        price_action_plan = _price_action_15m_plan(
            symbol,
            {
                "symbol": symbol,
                "short_term_15m": short_term_plan,
                "sniper_15m": sniper_plan,
                "quantum_15m": quantum_plan,
                "crypto_reversal_15m": crypto_reversal_plan,
                "ema_cross_15m": ema_cross_plan,
                "actionzone_15m": actionzone_plan,
                "cdc_vixfix_15m": cdc_vixfix_plan,
            },
            data_15m=shared_15m,
            data_1h=shared_1h,
            order_blocks=order_blocks_15m,
            prediction=prediction,
            phase_status=phase_status,
        )
        price_action_plan = _attach_exit_levels(
            price_action_plan,
            entry_keys=["entry_price", "current_price", "price"],
            stop_keys=["stop_loss"],
            tp_keys=["take_profit"],
            context=exit_context,
        )
        price_forecast = build_price_forecast(
            float(latest["Close"]) if pd.notna(latest["Close"]) else None,
            prediction,
            support=float(support) if pd.notna(support) else None,
            resistance=float(resistance) if pd.notna(resistance) else None,
            actionzone_plan=actionzone_plan,
            ema_plan=ema_cross_plan,
            cdc_plan=cdc_vixfix_plan,
        )
        all_weather_report = _build_all_weather_report_entry(
            {
                "symbol": symbol,
                "name": info["name"],
                "price": float(latest["Close"]) if pd.notna(latest["Close"]) else None,
                "change": float(((latest["Close"] - latest["Open"]) / latest["Open"]) * 100) if pd.notna(latest["Close"]) and pd.notna(latest["Open"]) else None,
                "short_term_15m": short_term_plan,
                "sniper_15m": sniper_plan,
                "quantum_15m": quantum_plan,
                "crypto_reversal_15m": crypto_reversal_plan,
                "ema_cross_15m": ema_cross_plan,
                "actionzone_15m": actionzone_plan,
                "cdc_vixfix_15m": cdc_vixfix_plan,
                "price_action_15m": price_action_plan,
            }
        )
        if isinstance(all_weather_report, dict):
            all_weather_report = dict(all_weather_report)
            all_weather_report.pop("signal_payload", None)
        result = {
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
            "cdc_vixfix_15m": cdc_vixfix_plan,
            "price_action_15m": price_action_plan,
            "order_blocks_15m": order_blocks_15m,
            "support": float(support) if pd.notna(support) else None,
            "resistance": float(resistance) if pd.notna(resistance) else None,
            "volume_status": vol_status,
            "rsi": float(latest["RSI"]) if pd.notna(latest["RSI"]) else None,
            "macd": float(latest["MACD"]) if pd.notna(latest["MACD"]) else None,
            "prediction": prediction,
            "price_forecast": price_forecast,
            "all_weather_15m": all_weather_report,
        }
        result["ui_summary"] = _build_ui_result_summary(result)
        return result
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def _build_health_snapshot():
    return {
        "status": "ok",
        "server_time": get_thai_now().strftime("%Y-%m-%d %H:%M:%S (Asia/Bangkok)"),
        "warnings": _get_config_warnings(),
        "data_sources": _data_source_health_snapshot(),
    }


def _get_primary_plan_source_label(item, plan):
    if not isinstance(item, dict) or not isinstance(plan, dict):
        return "AI Summary"
    if plan is item.get("cdc_vixfix_15m"):
        return "CDC+VixFix 15m"
    if plan is item.get("actionzone_15m"):
        return "ActionZone 15m"
    if plan is item.get("ema_cross_15m"):
        return "EMA Cross 15m"
    if plan is item.get("crypto_reversal_15m"):
        return "Crypto Reversal 15m"
    if plan is item.get("short_term_15m"):
        return "Short Term 15m"
    if plan is item.get("sniper_15m"):
        return "Sniper 15m"
    if plan is item.get("quantum_15m"):
        return "Quantum 15m"
    return "AI Summary"


def _build_ui_result_summary(item):
    if not isinstance(item, dict) or item.get("error"):
        return {
            "signal": "WAIT",
            "source": "N/A",
            "confidence": None,
            "pattern": None,
            "is_actionable": False,
            "entry_risk_pct": None,
            "stop_loss": None,
            "take_profit": None,
        }
    primary_plan = _pick_primary_trade_plan(item)
    signal = None
    source = "AI Summary"
    pattern = None
    entry_risk_pct = None
    stop_loss = None
    take_profit = None
    if isinstance(primary_plan, dict):
        signal = _normalize_trade_direction(
            primary_plan.get("signal")
            or primary_plan.get("raw_signal")
            or primary_plan.get("setup")
            or primary_plan.get("recommendation")
        )
        source = _get_primary_plan_source_label(item, primary_plan)
        raw_pattern = str(primary_plan.get("detected_pattern") or "").strip()
        if raw_pattern and raw_pattern.upper() != "NONE":
            pattern = raw_pattern
        risk_value = primary_plan.get("entry_risk_pct")
        if isinstance(risk_value, (int, float)) and math.isfinite(risk_value):
            entry_risk_pct = float(risk_value)
        stop_value = _pick_plan_value(primary_plan, ["stop_loss", "entry_stop_loss"])
        if isinstance(stop_value, (int, float)) and math.isfinite(stop_value):
            stop_loss = float(stop_value)
        target_value = _pick_plan_value(primary_plan, ["take_profit", "take_profit_2", "exit_price", "trailing_stop"])
        if isinstance(target_value, (int, float)) and math.isfinite(target_value):
            take_profit = float(target_value)
    if signal not in ("BUY", "SELL"):
        signal = _normalize_trade_direction(item.get("signal")) or "WAIT"
    confidence = _get_best_confidence(item)
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        confidence = None
    return {
        "signal": signal,
        "source": source,
        "confidence": float(confidence) if confidence is not None else None,
        "pattern": pattern,
        "is_actionable": signal in ("BUY", "SELL"),
        "entry_risk_pct": entry_risk_pct,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }


def _build_analysis_summary(results, requested_symbols=None, period=None):
    total_requested = int(requested_symbols or 0)
    returned = results if isinstance(results, list) else []
    success_count = 0
    error_count = 0
    buy_count = 0
    sell_count = 0
    wait_count = 0
    actionable = []
    errors = []
    for item in returned:
        if not isinstance(item, dict):
            continue
        symbol = normalize_symbol(item.get("symbol") or "")
        if item.get("error"):
            error_count += 1
            errors.append({
                "symbol": symbol,
                "error": str(item.get("error") or "").strip(),
            })
            continue
        success_count += 1
        ui_summary = item.get("ui_summary")
        if not isinstance(ui_summary, dict):
            ui_summary = _build_ui_result_summary(item)
        signal = str(ui_summary.get("signal") or "WAIT").upper()
        if signal == "BUY":
            buy_count += 1
        elif signal == "SELL":
            sell_count += 1
        else:
            wait_count += 1
        if signal in ("BUY", "SELL"):
            actionable.append(
                {
                    "symbol": symbol,
                    "name": str(item.get("name") or "").strip(),
                    "signal": signal,
                    "source": str(ui_summary.get("source") or "AI Summary"),
                    "confidence": float(ui_summary.get("confidence")) if isinstance(ui_summary.get("confidence"), (int, float)) else None,
                    "pattern": ui_summary.get("pattern"),
                    "price": float(item.get("price")) if isinstance(item.get("price"), (int, float)) else None,
                    "change": float(item.get("change")) if isinstance(item.get("change"), (int, float)) else None,
                }
            )
    actionable.sort(
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            abs(float(item.get("change") or 0.0)),
            str(item.get("symbol") or ""),
        ),
        reverse=True,
    )
    dominant_signal = "WAIT"
    if buy_count and buy_count > sell_count:
        dominant_signal = "BUY"
    elif sell_count and sell_count > buy_count:
        dominant_signal = "SELL"
    elif buy_count or sell_count:
        dominant_signal = "MIXED"
    total_tracked = success_count + error_count
    success_rate_pct = None
    if total_tracked > 0:
        success_rate_pct = round((success_count / total_tracked) * 100.0, 1)
    return {
        "requested_count": total_requested or len(returned),
        "returned_count": len(returned),
        "success_count": success_count,
        "error_count": error_count,
        "success_rate_pct": success_rate_pct,
        "actionable_count": len(actionable),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "wait_count": wait_count,
        "dominant_signal": dominant_signal,
        "period": str(period or ""),
        "top_signal": actionable[0] if actionable else None,
        "action_queue": actionable[:5],
        "errors": errors[:3],
    }

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify(_build_health_snapshot())


def _analyze_symbols_batch(symbols, period, include_chart_data=False):
    symbols = list(symbols or [])
    if len(symbols) == 1:
        return [analyze_single_symbol(symbols[0], period, include_chart_data=include_chart_data)]
    return list(_ANALYZE_EXECUTOR.map(analyze_single_symbol, symbols, repeat(period), repeat(include_chart_data)))


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid request body'}), 400
    raw_symbols = data.get('symbols', '')
    all_symbols = _parse_symbols_input(raw_symbols, max_symbols=10000)
    symbols = all_symbols[:_MAX_SYMBOLS_PER_REQUEST]
    period = data.get('period', '1mo')
    include_chart_data = bool(data.get("include_chart_data", True))
    if period not in VALID_PERIODS:
        return jsonify({'error': 'Invalid period'}), 400
    if not all_symbols:
        return jsonify({'error': 'No symbols provided'}), 400
    if len(all_symbols) > _MAX_SYMBOLS_PER_REQUEST:
        return jsonify({'error': f'Too many symbols (max {_MAX_SYMBOLS_PER_REQUEST})'}), 400

    results = _analyze_symbols_batch(symbols, period, include_chart_data=include_chart_data)
    notify = bool(data.get("notify_telegram"))
    if notify:
        _notify_telegram_from_results(results)

    alert_backtest = _build_alert_backtest_summary(results)
    backtest_rules = _build_backtest_rulebook(results)
    all_weather = _build_all_weather_summary(results)
    cleaned = [_clean_json_value(r) for r in results]
    meta = {
        "request": {
            "symbols": symbols,
            "period": period,
            "include_chart_data": include_chart_data,
            "notify_telegram": notify,
        },
        "summary": _build_analysis_summary(results, requested_symbols=len(symbols), period=period),
        "telegram_alerts": alert_backtest,
        "all_weather": all_weather,
        "backtest_rules": backtest_rules,
        "health": _build_health_snapshot(),
    }
    return jsonify({'results': cleaned, 'meta': _clean_json_value(meta)})


@app.route('/report/telegram-alerts', methods=['GET', 'POST'])
def report_telegram_alerts():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid request body'}), 400
    else:
        data = dict(request.args)
    days = data.get("days", 30)
    try:
        days = float(days)
    except Exception:
        days = 30.0
    if days <= 0:
        days = 30.0
    limit_examples = data.get("limit_examples_per_strategy", 1)
    try:
        limit_examples = max(1, int(limit_examples))
    except Exception:
        limit_examples = 1
    strategies = _parse_strategy_input(data.get("strategies"))
    raw_symbols = data.get("symbols", "")
    symbols = _parse_symbols_input(raw_symbols, max_symbols=10000) if raw_symbols else []
    include_presets = bool(data.get("include_presets", True))
    include_live_preview = bool(data.get("include_live_preview", False))
    include_latest_run = bool(data.get("include_latest_run", True))
    report = _build_telegram_alert_report(
        days=days,
        strategies=strategies or None,
        symbols=symbols or None,
        limit_examples_per_strategy=limit_examples,
    )
    payload = {
        "request": {
            "days": days,
            "strategies": strategies,
            "symbols": symbols,
            "limit_examples_per_strategy": limit_examples,
            "include_presets": include_presets,
            "include_live_preview": include_live_preview,
            "include_latest_run": include_latest_run,
        },
        "report": report,
        "health": _build_health_snapshot(),
        "files": {
            "alert_history_jsonl": _alert_history_file_path(),
            "alert_history_csv": _alert_history_csv_path() if _alert_history_export_csv_enabled() else None,
            "latest_run_json": _alert_run_report_file_path() if _alert_run_report_enabled() else None,
            "run_reports_jsonl": _alert_run_report_log_path() if _alert_run_report_enabled() else None,
        },
    }
    if include_latest_run:
        payload["latest_run"] = _read_latest_telegram_run_report()
    if include_presets:
        payload["presets"] = {
            "1d": _build_telegram_alert_report(
                days=1,
                strategies=strategies or None,
                symbols=symbols or None,
                limit_examples_per_strategy=limit_examples,
            ),
            "30d": _build_telegram_alert_report(
                days=30,
                strategies=strategies or None,
                symbols=symbols or None,
                limit_examples_per_strategy=limit_examples,
            ),
        }
    if include_live_preview and symbols:
        period = data.get("period", "15m")
        if period not in VALID_PERIODS:
            return jsonify({'error': 'Invalid period'}), 400
        preview_symbols = symbols[:_MAX_SYMBOLS_PER_REQUEST]
        preview_results = _analyze_symbols_batch(preview_symbols, period, include_chart_data=False)
        payload["live_preview"] = {
            "period": period,
            "symbols": preview_symbols,
            "report": _build_telegram_alert_live_preview(
                preview_results,
                limit_examples_per_strategy=limit_examples,
            ),
        }
    return jsonify(_clean_json_value(payload))


@app.route('/report/all-weather', methods=['POST'])
def report_all_weather():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid request body'}), 400
    raw_symbols = data.get('symbols', '')
    all_symbols = _parse_symbols_input(raw_symbols, max_symbols=10000)
    symbols = all_symbols[:_MAX_SYMBOLS_PER_REQUEST]
    periods = _parse_periods_input(data.get('periods'), default_periods=["1mo", "3mo", "6mo"], max_periods=6)
    if not all_symbols:
        return jsonify({'error': 'No symbols provided'}), 400
    if len(all_symbols) > _MAX_SYMBOLS_PER_REQUEST:
        return jsonify({'error': f'Too many symbols (max {_MAX_SYMBOLS_PER_REQUEST})'}), 400
    if not periods:
        return jsonify({'error': 'No valid periods provided'}), 400

    period_reports = {}
    for period in periods:
        results = _analyze_symbols_batch(symbols, period, include_chart_data=False)
        period_reports[period] = _build_all_weather_summary(results)
    readiness = _build_all_weather_readiness(period_reports)
    payload = {
        "request": {
            "symbols": symbols,
            "periods": periods,
        },
        "readiness": readiness,
        "period_reports": period_reports,
        "health": _build_health_snapshot(),
    }
    return jsonify(_clean_json_value(payload))

def _run_once(symbols, period, notify_telegram, verify_output=None, verify_include_results=None):
    uniq = _parse_symbols_input(symbols, max_symbols=_MAX_SYMBOLS_PER_REQUEST)
    if not uniq:
        return 2
    if period not in VALID_PERIODS:
        return 2
    results = _analyze_symbols_batch(uniq, period, include_chart_data=False)
    live_preview = _build_telegram_alert_live_preview(results, limit_examples_per_strategy=1)
    if notify_telegram:
        _notify_telegram_from_results(results)
    alert_backtest = _build_alert_backtest_summary(results)
    backtest_rules = _build_backtest_rulebook(results)
    all_weather = _build_all_weather_summary(results)
    health_snapshot = _build_health_snapshot()
    latest_run = _read_latest_telegram_run_report()
    verify_request = {
        "symbols": uniq,
        "period": period,
        "include_chart_data": False,
        "notify_telegram": bool(notify_telegram),
    }
    verify_summary = _build_analysis_summary(results, requested_symbols=len(uniq), period=period)
    if verify_include_results is None:
        verify_include_results = bool(getattr(config, "VERIFY_INCLUDE_RESULTS", False))
    verify_path = str(verify_output or getattr(config, "VERIFY_OUTPUT_PATH", "") or "").strip()
    if verify_path:
        written_path = _write_verify_output(
            verify_path,
            {
                "results": results,
                "request": verify_request,
                "summary": verify_summary,
                "telegram_alerts": alert_backtest,
                "all_weather": all_weather,
                "backtest_rules": backtest_rules,
                "health": health_snapshot,
                "latest_run": latest_run,
                "live_preview": live_preview,
                "include_results": bool(verify_include_results),
            },
        )
        if not written_path:
            logger.error("Failed to write verify output to %s", verify_path)
            return 1
    payload = {
        "results": [_clean_json_value(r) for r in results],
        "meta": _clean_json_value(
            {
                "request": verify_request,
                "summary": verify_summary,
                "telegram_alerts": alert_backtest,
                "all_weather": all_weather,
                "backtest_rules": backtest_rules,
                "health": health_snapshot,
                "latest_run": latest_run,
                "live_preview": live_preview,
                "verify_output_path": verify_path or None,
            }
        ),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="")
    parser.add_argument("--period", default="1mo")
    parser.add_argument("--notify-telegram", action="store_true")
    parser.add_argument("--verify-output", default="")
    parser.add_argument("--verify-include-results", action="store_true", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    if str(args.symbols or "").strip():
        raise SystemExit(
            _run_once(
                args.symbols,
                str(args.period or "1mo"),
                bool(args.notify_telegram),
                verify_output=str(args.verify_output or ""),
                verify_include_results=args.verify_include_results,
            )
        )
    port = int(args.port) if args.port is not None else int(getattr(config, "PORT", 5000))
    app.run(debug=getattr(config, "FLASK_DEBUG", True), port=port)
