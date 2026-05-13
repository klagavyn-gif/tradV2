import os
import tempfile
import threading
import time
from collections import Counter, deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from curl_cffi import requests as curl_requests


_YF_TZ_CACHE_LOCK = threading.Lock()
_YF_TZ_CACHE_READY = False
_THREAD_LOCAL = threading.local()
_SOURCE_HEALTH_LOCK = threading.Lock()
_SOURCE_HEALTH_COUNTERS = Counter()
_SOURCE_HEALTH_RECENT = deque()


def project_yf_cache_dir(base_dir):
    try:
        return os.path.join(os.path.dirname(os.path.abspath(base_dir)), ".yf_cache")
    except Exception:
        return ""


def configure_yf_tz_cache(base_dir, *, force_temp=False):
    global _YF_TZ_CACHE_READY
    if _YF_TZ_CACHE_READY and not force_temp:
        return
    with _YF_TZ_CACHE_LOCK:
        if _YF_TZ_CACHE_READY and not force_temp:
            return
        candidates = []
        if not force_temp:
            try:
                candidates.append(project_yf_cache_dir(base_dir))
            except Exception:
                pass
        candidates.append(os.path.join(tempfile.gettempdir(), "trad_yf_cache"))
        for cache_dir in candidates:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                yf.set_tz_cache_location(cache_dir)
                _YF_TZ_CACHE_READY = True
                return
            except Exception:
                continue
        _YF_TZ_CACHE_READY = True


def set_thread_curl_session(session):
    _THREAD_LOCAL.curl_session = session


def clear_yf_runtime_cache(base_dir, *, clear_tz_cache=False):
    cache_dir = project_yf_cache_dir(base_dir)
    patterns = ["cookies.db", "cookies.db-shm", "cookies.db-wal"]
    if clear_tz_cache:
        patterns.extend(["tkr-tz.db", "tkr-tz.db-shm", "tkr-tz.db-wal"])
    for filename in patterns:
        if not cache_dir:
            continue
        path = os.path.join(cache_dir, filename)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            continue
    _THREAD_LOCAL.curl_session = None
    if clear_tz_cache:
        global _YF_TZ_CACHE_READY
        _YF_TZ_CACHE_READY = False


def _source_health_recent_limit(config):
    try:
        return max(10, int(getattr(config, "YF_SOURCE_HEALTH_RECENT_EVENTS", 40)))
    except Exception:
        return 40


def record_source_health_event(
    source,
    status,
    *,
    symbol=None,
    detail=None,
    attempt=None,
    period=None,
    interval=None,
    elapsed_ms=None,
    config,
):
    src = str(source or "unknown").strip().lower() or "unknown"
    state = str(status or "unknown").strip().lower() or "unknown"
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": src,
        "status": state,
        "symbol": str(symbol or "").strip().upper() or None,
        "detail": str(detail or "").strip() or None,
        "attempt": int(attempt) if isinstance(attempt, int) else None,
        "period": str(period or "").strip() or None,
        "interval": str(interval or "").strip() or None,
        "elapsed_ms": round(float(elapsed_ms), 1) if isinstance(elapsed_ms, (int, float)) else None,
    }
    with _SOURCE_HEALTH_LOCK:
        _SOURCE_HEALTH_COUNTERS[f"{src}|{state}"] += 1
        _SOURCE_HEALTH_COUNTERS[f"source:{src}"] += 1
        _SOURCE_HEALTH_COUNTERS[f"status:{state}"] += 1
        while len(_SOURCE_HEALTH_RECENT) >= _source_health_recent_limit(config):
            _SOURCE_HEALTH_RECENT.popleft()
        _SOURCE_HEALTH_RECENT.append(entry)


def build_source_health_snapshot(*, config):
    with _SOURCE_HEALTH_LOCK:
        counters = dict(_SOURCE_HEALTH_COUNTERS)
        recent = list(_SOURCE_HEALTH_RECENT)
    by_source = {}
    for key, value in counters.items():
        if key.startswith("source:") or key.startswith("status:") or "|" not in key:
            continue
        source, status = key.split("|", 1)
        bucket = by_source.setdefault(source, {})
        bucket[status] = int(value)
    return {
        "sources": by_source,
        "totals": {
            "events": int(sum(v for k, v in counters.items() if "|" in k and not k.startswith("source:") and not k.startswith("status:"))),
            "recent_events": len(recent),
        },
        "recent": recent,
    }


def is_yf_auth_error(message):
    msg = str(message or "").lower()
    return (
        "invalid crumb" in msg
        or '"code":"unauthorized"' in msg
        or "code\":\"unauthorized\"" in msg
        or "http error 401" in msg
        or "unauthorized" in msg and "finance" in msg
    )


def get_http_verify_setting(*, config, logger):
    verify = getattr(config, "HTTP_VERIFY", True)
    if not verify:
        return False
    ca_bundle = getattr(config, "HTTP_CA_BUNDLE", "")
    if isinstance(ca_bundle, str) and ca_bundle.strip():
        bundle = ca_bundle.strip()
        if os.path.exists(bundle):
            return bundle
        logger.warning("HTTP_CA_BUNDLE not found: %s; falling back to system cert store", bundle)
    return True


def create_curl_session(*, config, logger):
    impersonate = getattr(config, "CURL_IMPERSONATE", "chrome110")
    verify = get_http_verify_setting(config=config, logger=logger)
    try:
        return curl_requests.Session(verify=verify, impersonate=impersonate)
    except Exception as e:
        msg = str(e).lower()
        if "certificate verify locations" in msg or "curl: (77)" in msg:
            logger.warning("Falling back to verify=False due to CA configuration error: %s", e)
            return curl_requests.Session(verify=False, impersonate=impersonate)
        raise


def get_thread_curl_session(*, config, logger):
    sess = getattr(_THREAD_LOCAL, "curl_session", None)
    if sess is None:
        sess = create_curl_session(config=config, logger=logger)
        _THREAD_LOCAL.curl_session = sess
    return sess


def normalize_df_index(df, *, tz_name="Asia/Bangkok"):
    if df is None or getattr(df, "empty", True):
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is not None:
                df.index = df.index.tz_convert(pytz.timezone(tz_name))
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
    return df


def normalize_price_columns(df, *, symbol=None, normalize_symbol_fn):
    if df is None or getattr(df, "empty", True):
        return df
    cols = getattr(df, "columns", None)
    if isinstance(cols, pd.MultiIndex):
        try:
            df = df.copy()
            if len(cols.levels) >= 2:
                lvl0 = [str(x) for x in cols.get_level_values(0)]
                lvl1 = [str(x) for x in cols.get_level_values(1)]
                sym = normalize_symbol_fn(symbol)
                if sym and sym in set(lvl1):
                    keep = [i for i, t in enumerate(lvl1) if t == sym]
                    if keep:
                        df = df.iloc[:, keep]
                        lvl0 = [lvl0[i] for i in keep]
                df.columns = pd.Index(lvl0)
        except Exception:
            pass
    if not isinstance(df.columns, pd.Index):
        return df
    seen = set()
    keep_cols = []
    for name in ("Open", "High", "Low", "Close", "Volume"):
        if name in df.columns and name not in seen:
            keep_cols.append(name)
            seen.add(name)
    if keep_cols:
        return df[keep_cols].copy()
    return df


def period_to_timedelta(period, *, now_getter=None):
    text = str(period or "").strip().lower()
    if not text:
        return None
    if text == "ytd":
        now = now_getter() if callable(now_getter) else datetime.now()
        return now - datetime(now.year, 1, 1)
    import re

    m = re.match(r"^(\d+)(m|mo|h|d|wk|w|y)$", text)
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    if unit in ("w", "wk"):
        return timedelta(weeks=value)
    if unit in ("m", "mo"):
        return timedelta(days=value * 30)
    if unit == "y":
        return timedelta(days=value * 365)
    return None


def slice_history_by_period(df, period, *, now_getter=None):
    if df is None or getattr(df, "empty", True):
        return df
    delta = period_to_timedelta(period, now_getter=now_getter)
    if delta is None:
        return df.copy()
    end_ts = df.index.max()
    if pd.isna(end_ts):
        return df.copy()
    cutoff = end_ts - delta
    sliced = df[df.index >= cutoff].copy()
    return sliced if not sliced.empty else df.copy()


def remote_history_period(period, interval, *, now_getter=None):
    text = str(period or "").strip().lower()
    if not text:
        return period
    delta = period_to_timedelta(text, now_getter=now_getter)
    intraday = str(interval or "").strip().lower() in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
    if intraday and isinstance(delta, timedelta) and delta > timedelta(days=60):
        return "60d"
    return period


def chart_interval(interval):
    text = str(interval or "").strip().lower()
    if text == "1h":
        return "60m"
    return text or "1d"


def prefer_chart_api(*, config, environ):
    if bool(getattr(config, "YF_PREFER_CHART_API", False)):
        return True
    return str(environ.get("GITHUB_ACTIONS", "")).strip().lower() == "true"


def fetch_yahoo_chart_history(
    symbol,
    period,
    *,
    interval=None,
    auto_adjust=True,
    config,
    logger,
    helpers,
):
    normalize_symbol_fn = helpers["normalize_symbol"]
    get_thread_curl_session_fn = helpers["get_thread_curl_session"]
    normalize_df_index_fn = helpers["normalize_df_index"]

    sym = normalize_symbol_fn(symbol)
    if not sym:
        return None
    session = get_thread_curl_session_fn()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    params = {
        "range": str(period or "1mo"),
        "interval": chart_interval(interval),
        "includePrePost": "false",
        "events": "div,splits,capitalGains",
    }
    response = session.get(url, params=params, timeout=20)
    status_code = int(getattr(response, "status_code", 0))
    if status_code >= 400:
        raise RuntimeError(f"Yahoo chart API HTTP {status_code} for {sym}")
    payload = response.json()
    chart = ((payload or {}).get("chart") or {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo chart API error for {sym}: {error}")
    results = chart.get("result") or []
    if not results:
        return None
    result = results[0] or {}
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote_list = indicators.get("quote") or []
    if not timestamps or not quote_list:
        return None
    quote = quote_list[0] or {}
    df = pd.DataFrame(
        {
            "Open": quote.get("open") or [],
            "High": quote.get("high") or [],
            "Low": quote.get("low") or [],
            "Close": quote.get("close") or [],
            "Volume": quote.get("volume") or [],
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True),
    )
    if auto_adjust:
        adjclose_list = indicators.get("adjclose") or []
        if adjclose_list:
            adjclose = adjclose_list[0].get("adjclose") or []
            if len(adjclose) == len(df.index):
                raw_close = pd.to_numeric(df["Close"], errors="coerce")
                adj_close = pd.to_numeric(pd.Series(adjclose, index=df.index), errors="coerce")
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = adj_close / raw_close.replace(0, np.nan)
                ratio = ratio.replace([np.inf, -np.inf], np.nan)
                for col in ("Open", "High", "Low", "Close"):
                    series = pd.to_numeric(df[col], errors="coerce")
                    df[col] = series * ratio.fillna(1.0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="all")
    if df.empty:
        return None
    return normalize_df_index_fn(df)


def get_yf_history(
    symbol,
    period,
    *,
    interval=None,
    auto_adjust=True,
    cache_ttl_seconds=None,
    config,
    logger,
    helpers,
):
    normalize_symbol_fn = helpers["normalize_symbol"]
    cache_get = helpers["cache_get"]
    cache_set = helpers["cache_set"]
    empty_sentinel = helpers["empty_sentinel"]
    history_store_read = helpers["history_store_read"]
    history_store_merge = helpers["history_store_merge"]
    history_store_write = helpers["history_store_write"]
    configure_yf_tz_cache_fn = helpers["configure_yf_tz_cache"]
    normalize_price_columns_fn = helpers["normalize_price_columns"]
    normalize_df_index_fn = helpers["normalize_df_index"]
    slice_history_by_period_fn = helpers["slice_history_by_period"]
    remote_history_period_fn = helpers["remote_history_period"]
    prefer_chart_api_fn = helpers["prefer_chart_api"]
    fetch_yahoo_chart_history_fn = helpers["fetch_yahoo_chart_history"]
    get_thread_curl_session_fn = helpers["get_thread_curl_session"]
    is_yf_auth_error_fn = helpers["is_yf_auth_error"]
    clear_yf_runtime_cache_fn = helpers["clear_yf_runtime_cache"]
    set_thread_curl_session_fn = helpers["set_thread_curl_session"]
    record_source_health_event_fn = helpers["record_source_health_event"]

    sym = normalize_symbol_fn(symbol)
    if not sym:
        return None
    key = ("hist", sym, str(period or ""), str(interval or ""), bool(auto_adjust))
    cached = cache_get(key)
    if cached is empty_sentinel:
        return None
    if isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached.copy()
    disk_df = None
    if interval:
        disk_df = history_store_read(sym, interval=interval, auto_adjust=auto_adjust)
    configure_yf_tz_cache_fn()
    remote_period = remote_history_period_fn(period, interval)
    max_retries = max(1, int(getattr(config, "YF_FETCH_MAX_RETRIES", 3)))
    retry_backoff = max(0.0, float(getattr(config, "YF_RETRY_BACKOFF_SECONDS", 1.25)))
    if prefer_chart_api_fn():
        chart_started = time.perf_counter()
        try:
            df = fetch_yahoo_chart_history_fn(sym, remote_period, interval=interval, auto_adjust=auto_adjust)
            if isinstance(df, pd.DataFrame) and not df.empty:
                record_source_health_event_fn(
                    "chart_api",
                    "success",
                    symbol=sym,
                    period=remote_period,
                    interval=interval,
                    elapsed_ms=(time.perf_counter() - chart_started) * 1000.0,
                )
                df = normalize_price_columns_fn(df, sym)
                if interval:
                    merged_df = history_store_merge(disk_df, df, symbol=sym)
                    if isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
                        df = merged_df
                        history_store_write(sym, interval, auto_adjust, merged_df)
                sliced_df = slice_history_by_period_fn(df, period)
                cache_set(key, sliced_df, ttl_seconds=cache_ttl_seconds)
                return sliced_df.copy()
            record_source_health_event_fn(
                "chart_api",
                "empty",
                symbol=sym,
                period=remote_period,
                interval=interval,
                elapsed_ms=(time.perf_counter() - chart_started) * 1000.0,
            )
        except Exception as e:
            record_source_health_event_fn(
                "chart_api",
                "error",
                symbol=sym,
                detail=e,
                attempt=1,
                period=remote_period,
                interval=interval,
                elapsed_ms=(time.perf_counter() - chart_started) * 1000.0,
            )
            logger.warning("Preferred Yahoo chart API fetch failed for %s: %s", sym, e)
    auth_error_seen = False
    for attempt in range(max_retries):
        try:
            fetch_started = time.perf_counter()
            session = get_thread_curl_session_fn()
            ticker = yf.Ticker(sym, session=session)
            if interval:
                df = ticker.history(period=remote_period, interval=interval, auto_adjust=auto_adjust)
            else:
                df = ticker.history(period=remote_period, auto_adjust=auto_adjust)
            if isinstance(df, pd.DataFrame) and not df.empty:
                record_source_health_event_fn(
                    "yfinance_history",
                    "success",
                    symbol=sym,
                    attempt=attempt + 1,
                    period=remote_period,
                    interval=interval,
                    elapsed_ms=(time.perf_counter() - fetch_started) * 1000.0,
                )
            if df is None or df.empty:
                record_source_health_event_fn(
                    "yfinance_history",
                    "empty",
                    symbol=sym,
                    attempt=attempt + 1,
                    period=remote_period,
                    interval=interval,
                    elapsed_ms=(time.perf_counter() - fetch_started) * 1000.0,
                )
                dl_kwargs = {
                    "period": remote_period,
                    "auto_adjust": auto_adjust,
                    "progress": False,
                    "threads": False,
                    "session": session,
                }
                if interval:
                    dl_kwargs["interval"] = interval
                download_started = time.perf_counter()
                try:
                    df = yf.download(sym, **dl_kwargs)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        record_source_health_event_fn(
                            "yfinance_download",
                            "success",
                            symbol=sym,
                            attempt=attempt + 1,
                            period=remote_period,
                            interval=interval,
                            elapsed_ms=(time.perf_counter() - download_started) * 1000.0,
                        )
                except Exception as dl_e:
                    dl_msg = str(dl_e).lower()
                    record_source_health_event_fn(
                        "yfinance_download",
                        "error",
                        symbol=sym,
                        detail=dl_e,
                        attempt=attempt + 1,
                        period=remote_period,
                        interval=interval,
                        elapsed_ms=(time.perf_counter() - download_started) * 1000.0,
                    )
                    if attempt < (max_retries - 1) and is_yf_auth_error_fn(dl_msg):
                        auth_error_seen = True
                        record_source_health_event_fn(
                            "auth_recovery",
                            "reset",
                            symbol=sym,
                            detail="download_auth_error",
                            attempt=attempt + 1,
                            period=remote_period,
                            interval=interval,
                        )
                        logger.warning("Resetting yfinance cookie/session cache after auth error for %s: %s", sym, dl_e)
                        clear_yf_runtime_cache_fn(clear_tz_cache=True)
                        if retry_backoff > 0:
                            time.sleep(retry_backoff * float(attempt + 1))
                        continue
                    if "certificate verify locations" in dl_msg or "curl: (77)" in dl_msg:
                        try:
                            insecure_session = curl_requests.Session(
                                verify=False,
                                impersonate=getattr(config, "CURL_IMPERSONATE", "chrome110"),
                            )
                            set_thread_curl_session_fn(insecure_session)
                            dl_kwargs["session"] = insecure_session
                            insecure_started = time.perf_counter()
                            df = yf.download(sym, **dl_kwargs)
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                record_source_health_event_fn(
                                    "yfinance_download_insecure",
                                    "success",
                                    symbol=sym,
                                    attempt=attempt + 1,
                                    period=remote_period,
                                    interval=interval,
                                    elapsed_ms=(time.perf_counter() - insecure_started) * 1000.0,
                                )
                        except Exception:
                            df = None
                    else:
                        df = None
                if df is None or df.empty:
                    try:
                        insecure_session = curl_requests.Session(
                            verify=False,
                            impersonate=getattr(config, "CURL_IMPERSONATE", "chrome110"),
                        )
                        set_thread_curl_session_fn(insecure_session)
                        dl_kwargs["session"] = insecure_session
                        insecure_started = time.perf_counter()
                        df = yf.download(sym, **dl_kwargs)
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            record_source_health_event_fn(
                                "yfinance_download_insecure",
                                "success",
                                symbol=sym,
                                attempt=attempt + 1,
                                period=remote_period,
                                interval=interval,
                                elapsed_ms=(time.perf_counter() - insecure_started) * 1000.0,
                            )
                    except Exception:
                        df = None
                if df is None or df.empty:
                    if auth_error_seen:
                        chart_started = time.perf_counter()
                        try:
                            df = fetch_yahoo_chart_history_fn(sym, remote_period, interval=interval, auto_adjust=auto_adjust)
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                record_source_health_event_fn(
                                    "chart_api_fallback",
                                    "success",
                                    symbol=sym,
                                    attempt=attempt + 1,
                                    period=remote_period,
                                    interval=interval,
                                    elapsed_ms=(time.perf_counter() - chart_started) * 1000.0,
                                )
                        except Exception as chart_e:
                            record_source_health_event_fn(
                                "chart_api_fallback",
                                "error",
                                symbol=sym,
                                detail=chart_e,
                                attempt=attempt + 1,
                                period=remote_period,
                                interval=interval,
                                elapsed_ms=(time.perf_counter() - chart_started) * 1000.0,
                            )
                            logger.warning("Yahoo chart API fallback failed for %s: %s", sym, chart_e)
                    if isinstance(disk_df, pd.DataFrame) and not disk_df.empty:
                        sliced_disk = slice_history_by_period_fn(disk_df, period)
                        record_source_health_event_fn(
                            "disk_history",
                            "fallback",
                            symbol=sym,
                            detail="empty_remote_history",
                            attempt=attempt + 1,
                            period=period,
                            interval=interval,
                        )
                        cache_set(key, sliced_disk, ttl_seconds=cache_ttl_seconds)
                        return sliced_disk.copy()
                    record_source_health_event_fn(
                        "history_fetch",
                        "empty",
                        symbol=sym,
                        attempt=attempt + 1,
                        period=period,
                        interval=interval,
                    )
                    cache_set(key, empty_sentinel, ttl_seconds=8)
                    return None
            df = normalize_df_index_fn(df)
            df = normalize_price_columns_fn(df, sym)
            if interval:
                merged_df = history_store_merge(disk_df, df, symbol=sym)
                if isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
                    df = merged_df
                    history_store_write(sym, interval, auto_adjust, merged_df)
            sliced_df = slice_history_by_period_fn(df, period)
            cache_set(key, sliced_df, ttl_seconds=cache_ttl_seconds)
            return sliced_df.copy()
        except Exception as e:
            msg = str(e).lower()
            if attempt == 0 and ("disk i/o error" in msg or "operationalerror" in msg):
                record_source_health_event_fn(
                    "tz_cache",
                    "reset",
                    symbol=sym,
                    detail="disk_io_error",
                    attempt=attempt + 1,
                    period=period,
                    interval=interval,
                )
                configure_yf_tz_cache_fn(force_temp=True)
                set_thread_curl_session_fn(None)
                if retry_backoff > 0:
                    time.sleep(retry_backoff * float(attempt + 1))
                continue
            if attempt < (max_retries - 1) and is_yf_auth_error_fn(msg):
                auth_error_seen = True
                record_source_health_event_fn(
                    "auth_recovery",
                    "reset",
                    symbol=sym,
                    detail="history_auth_error",
                    attempt=attempt + 1,
                    period=period,
                    interval=interval,
                )
                logger.warning("Resetting yfinance cookie/session cache after auth error for %s: %s", sym, e)
                clear_yf_runtime_cache_fn(clear_tz_cache=True)
                if retry_backoff > 0:
                    time.sleep(retry_backoff * float(attempt + 1))
                continue
            if attempt == 0 and ("certificate verify locations" in msg or "curl: (77)" in msg):
                set_thread_curl_session_fn(
                    curl_requests.Session(
                        verify=False,
                        impersonate=getattr(config, "CURL_IMPERSONATE", "chrome110"),
                    )
                )
                record_source_health_event_fn(
                    "curl_session",
                    "verify_disabled",
                    symbol=sym,
                    detail="ssl_fallback",
                    attempt=attempt + 1,
                    period=period,
                    interval=interval,
                )
                if retry_backoff > 0:
                    time.sleep(retry_backoff * float(attempt + 1))
                continue
            if is_yf_auth_error_fn(msg):
                auth_error_seen = True
            if auth_error_seen:
                chart_started = time.perf_counter()
                try:
                    remote_period = remote_history_period_fn(period, interval)
                    df = fetch_yahoo_chart_history_fn(sym, remote_period, interval=interval, auto_adjust=auto_adjust)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        record_source_health_event_fn(
                            "chart_api_fallback",
                            "success",
                            symbol=sym,
                            attempt=attempt + 1,
                            period=remote_period,
                            interval=interval,
                            elapsed_ms=(time.perf_counter() - chart_started) * 1000.0,
                        )
                        df = normalize_price_columns_fn(df, sym)
                        if interval:
                            merged_df = history_store_merge(disk_df, df, symbol=sym)
                            if isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
                                df = merged_df
                                history_store_write(sym, interval, auto_adjust, merged_df)
                        sliced_df = slice_history_by_period_fn(df, period)
                        cache_set(key, sliced_df, ttl_seconds=cache_ttl_seconds)
                        return sliced_df.copy()
                except Exception as chart_e:
                    record_source_health_event_fn(
                        "chart_api_fallback",
                        "error",
                        symbol=sym,
                        detail=chart_e,
                        attempt=attempt + 1,
                        period=period,
                        interval=interval,
                        elapsed_ms=(time.perf_counter() - chart_started) * 1000.0,
                    )
                    logger.warning("Yahoo chart API fallback failed for %s: %s", sym, chart_e)
            record_source_health_event_fn(
                "history_fetch",
                "error",
                symbol=sym,
                detail=e,
                attempt=attempt + 1,
                period=period,
                interval=interval,
            )
            logger.warning("Error fetching %s: %s", sym, e, exc_info=True)
            if isinstance(disk_df, pd.DataFrame) and not disk_df.empty:
                sliced_disk = slice_history_by_period_fn(disk_df, period)
                record_source_health_event_fn(
                    "disk_history",
                    "fallback",
                    symbol=sym,
                    detail="exception_remote_history",
                    attempt=attempt + 1,
                    period=period,
                    interval=interval,
                )
                cache_set(key, sliced_disk, ttl_seconds=cache_ttl_seconds)
                return sliced_disk.copy()
            cache_set(key, empty_sentinel, ttl_seconds=8)
            return None
    if isinstance(disk_df, pd.DataFrame) and not disk_df.empty:
        sliced_disk = slice_history_by_period_fn(disk_df, period)
        record_source_health_event_fn(
            "disk_history",
            "fallback",
            symbol=sym,
            detail="post_retry_fallback",
            attempt=max_retries,
            period=period,
            interval=interval,
        )
        cache_set(key, sliced_disk, ttl_seconds=cache_ttl_seconds)
        return sliced_disk.copy()
    record_source_health_event_fn(
        "history_fetch",
        "unavailable",
        symbol=sym,
        attempt=max_retries,
        period=period,
        interval=interval,
    )
    cache_set(key, empty_sentinel, ttl_seconds=8)
    return None
