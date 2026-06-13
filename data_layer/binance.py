import json
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def _normalize_interval(interval):
    text = str(interval or "").strip().lower()
    if text == "60m":
        return "1h"
    return text or "1d"


def _interval_to_binance(interval):
    text = _normalize_interval(interval)
    if text in {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1mo"}:
        return text
    return None


def _interval_to_milliseconds(interval):
    text = _normalize_interval(interval)
    match = re.match(r"^(\d+)(m|h|d|w|mo)$", text)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return value * 60 * 1000
    if unit == "h":
        return value * 60 * 60 * 1000
    if unit == "d":
        return value * 24 * 60 * 60 * 1000
    if unit == "w":
        return value * 7 * 24 * 60 * 60 * 1000
    if unit == "mo":
        return value * 30 * 24 * 60 * 60 * 1000
    return None


def symbol_to_binance(symbol, *, normalize_symbol_fn, quote="USDT"):
    sym = normalize_symbol_fn(symbol)
    if not sym:
        return ""
    text = str(sym).strip().upper()
    if text.endswith("USDT"):
        return text
    if "-" in text:
        base = text.split("-", 1)[0].strip().upper()
    else:
        base = text
    quote_text = str(quote or "USDT").strip().upper() or "USDT"
    return f"{base}{quote_text}"


def _period_start_ms(period, *, period_to_timedelta_fn, now_getter=None):
    try:
        delta = period_to_timedelta_fn(period, now_getter=now_getter)
    except TypeError:
        delta = period_to_timedelta_fn(period)
    now_dt = now_getter() if callable(now_getter) else datetime.now(timezone.utc).replace(tzinfo=None)
    if delta is None:
        delta = timedelta(days=365)
    start_dt = now_dt - delta
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    return int(start_dt.timestamp() * 1000.0)


def _http_error_status(exc):
    try:
        return int(getattr(exc, "code", 0) or 0)
    except Exception:
        return 0


def _http_error_detail(exc):
    parts = []
    status = _http_error_status(exc)
    if status > 0:
        parts.append(f"http_{status}")
    reason = str(getattr(exc, "reason", "") or "").strip()
    if reason:
        parts.append(reason)
    try:
        payload = exc.read()
    except Exception:
        payload = b""
    if isinstance(payload, bytes) and payload:
        try:
            text = payload.decode("utf-8", errors="replace").strip()
        except Exception:
            text = ""
        if text:
            parts.append(text[:240])
    return " | ".join(part for part in parts if part) or str(exc)


def _request_klines(symbol, *, interval, start_ms, end_ms, limit, timeout_seconds):
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": int(limit),
    }
    url = f"{_BINANCE_KLINES_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "tradV2-binance-history/1.0"})
    with urlopen(request, timeout=max(5.0, float(timeout_seconds or 20.0))) as response:
        payload = response.read()
    if not payload:
        return []
    data = json.loads(payload.decode("utf-8"))
    return data if isinstance(data, list) else []


def _klines_to_dataframe(rows, *, normalize_df_index_fn):
    if not isinstance(rows, list) or not rows:
        return None
    frame = pd.DataFrame(
        [
            {
                "Datetime": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "Open": float(row[1]),
                "High": float(row[2]),
                "Low": float(row[3]),
                "Close": float(row[4]),
                "Volume": float(row[5]),
            }
            for row in rows
            if isinstance(row, list) and len(row) >= 6
        ]
    )
    if frame.empty:
        return None
    frame = frame.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return normalize_df_index_fn(frame)


def get_binance_history(
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
    normalize_df_index_fn = helpers["normalize_df_index"]
    normalize_price_columns_fn = helpers["normalize_price_columns"]
    slice_history_by_period_fn = helpers["slice_history_by_period"]
    period_to_timedelta_fn = helpers["period_to_timedelta"]
    record_source_health_event_fn = helpers["record_source_health_event"]
    yahoo_fallback_history_fn = helpers.get("yahoo_fallback_history")
    now_getter = helpers.get("now_getter")

    sym = normalize_symbol_fn(symbol)
    if not sym:
        return None
    interval_text = _interval_to_binance(interval)
    if not interval_text:
        logger.warning("Binance provider does not support interval=%s for %s", interval, sym)
        return None

    key = ("hist", "binance", sym, str(period or ""), interval_text, bool(auto_adjust))
    cached = cache_get(key)
    if cached is empty_sentinel:
        return None
    if isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached.copy()

    disk_df = history_store_read(sym, interval=interval_text, auto_adjust=auto_adjust)
    market_symbol = symbol_to_binance(
        sym,
        normalize_symbol_fn=normalize_symbol_fn,
        quote=getattr(config, "BINANCE_DEFAULT_QUOTE", "USDT"),
    )
    limit = max(1, min(1000, int(getattr(config, "BINANCE_KLINE_LIMIT", 1000) or 1000)))
    timeout_seconds = float(getattr(config, "BINANCE_REQUEST_TIMEOUT_SECONDS", 20.0) or 20.0)
    interval_ms = _interval_to_milliseconds(interval_text)
    if not isinstance(interval_ms, int) or interval_ms <= 0:
        logger.warning("Could not resolve Binance interval size for %s", interval_text)
        return None

    now_dt = now_getter() if callable(now_getter) else datetime.utcnow()
    if getattr(now_dt, "tzinfo", None) is None:
        end_dt = now_dt.replace(tzinfo=timezone.utc)
    else:
        end_dt = now_dt.astimezone(timezone.utc)
    end_ms = int(end_dt.timestamp() * 1000.0)
    start_ms = _period_start_ms(period, period_to_timedelta_fn=period_to_timedelta_fn, now_getter=now_getter)
    start_ms = max(0, start_ms - interval_ms * 2)

    rows = []
    cursor_ms = int(start_ms)
    request_count = 0
    fetch_started = time.perf_counter()
    blocked_http_error = None
    try:
        while cursor_ms < end_ms:
            page = _request_klines(
                market_symbol,
                interval=interval_text,
                start_ms=cursor_ms,
                end_ms=end_ms,
                limit=limit,
                timeout_seconds=timeout_seconds,
            )
            request_count += 1
            if not page:
                break
            rows.extend(page)
            last_open_ms = int(page[-1][0])
            next_cursor_ms = last_open_ms + interval_ms
            if next_cursor_ms <= cursor_ms:
                break
            cursor_ms = next_cursor_ms
            if len(page) < limit:
                break
            time.sleep(0.05)
        elapsed_ms = (time.perf_counter() - fetch_started) * 1000.0
        fetched_df = _klines_to_dataframe(rows, normalize_df_index_fn=normalize_df_index_fn)
        if isinstance(fetched_df, pd.DataFrame) and not fetched_df.empty:
            record_source_health_event_fn(
                "binance_klines",
                "success",
                symbol=sym,
                period=period,
                interval=interval_text,
                detail=f"requests={request_count}",
                elapsed_ms=elapsed_ms,
            )
            fetched_df = normalize_price_columns_fn(fetched_df, sym)
            merged_df = history_store_merge(disk_df, fetched_df, symbol=sym)
            if isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
                fetched_df = merged_df
                history_store_write(sym, interval_text, auto_adjust, merged_df)
            sliced_df = slice_history_by_period_fn(fetched_df, period)
            cache_set(key, sliced_df, ttl_seconds=cache_ttl_seconds)
            return sliced_df.copy()
        record_source_health_event_fn(
            "binance_klines",
            "empty",
            symbol=sym,
            period=period,
            interval=interval_text,
            detail=f"requests={request_count}",
            elapsed_ms=elapsed_ms,
        )
    except HTTPError as exc:
        blocked_http_error = exc if _http_error_status(exc) == 451 else None
        detail = _http_error_detail(exc)
        record_source_health_event_fn(
            "binance_klines",
            "error",
            symbol=sym,
            period=period,
            interval=interval_text,
            detail=detail,
            attempt=request_count or 1,
            elapsed_ms=(time.perf_counter() - fetch_started) * 1000.0,
        )
        logger.warning("Binance history fetch failed for %s (%s): %s", sym, interval_text, detail)
    except Exception as exc:
        record_source_health_event_fn(
            "binance_klines",
            "error",
            symbol=sym,
            period=period,
            interval=interval_text,
            detail=str(exc),
            attempt=request_count or 1,
            elapsed_ms=(time.perf_counter() - fetch_started) * 1000.0,
        )
        logger.warning("Binance history fetch failed for %s (%s): %s", sym, interval_text, exc)

    if blocked_http_error is not None and callable(yahoo_fallback_history_fn):
        detail = _http_error_detail(blocked_http_error)
        record_source_health_event_fn(
            "binance_klines",
            "fallback_yahoo",
            symbol=sym,
            period=period,
            interval=interval_text,
            detail=detail,
        )
        try:
            yahoo_df = yahoo_fallback_history_fn(
                sym,
                period,
                interval=interval,
                auto_adjust=auto_adjust,
                cache_ttl_seconds=cache_ttl_seconds,
            )
            if isinstance(yahoo_df, pd.DataFrame) and not yahoo_df.empty:
                cache_set(key, yahoo_df, ttl_seconds=cache_ttl_seconds)
                return yahoo_df.copy()
        except Exception as exc:
            record_source_health_event_fn(
                "yahoo_fallback",
                "error",
                symbol=sym,
                period=period,
                interval=interval_text,
                detail=str(exc),
            )
            logger.warning("Yahoo fallback after Binance 451 failed for %s (%s): %s", sym, interval_text, exc)

    if isinstance(disk_df, pd.DataFrame) and not disk_df.empty:
        sliced_disk = slice_history_by_period_fn(disk_df, period)
        if isinstance(sliced_disk, pd.DataFrame) and not sliced_disk.empty:
            cache_set(key, sliced_disk, ttl_seconds=cache_ttl_seconds)
            return sliced_disk.copy()

    cache_set(key, empty_sentinel, ttl_seconds=cache_ttl_seconds)
    return None
