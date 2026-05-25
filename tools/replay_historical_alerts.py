import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import timedelta
from pathlib import Path

import pandas as pd


WATCHLIST = [
    "BTC-USD",
    "DOGE-USD",
    "ETH-USD",
    "ADA-USD",
    "XRP-USD",
    "BNB-USD",
    "SOL-USD",
    "TRX-USD",
    "NEAR-USD",
    "LINK-USD",
    "PAXG-USD",
]

PERIOD_MAP = {
    "5d": pd.Timedelta(days=5),
    "30d": pd.Timedelta(days=30),
    "60d": pd.Timedelta(days=60),
    "90d": pd.Timedelta(days=90),
    "1mo": pd.Timedelta(days=30),
    "3mo": pd.Timedelta(days=90),
    "6mo": pd.Timedelta(days=180),
    "1y": pd.Timedelta(days=365),
    "2y": pd.Timedelta(days=730),
    "5y": pd.Timedelta(days=365 * 5),
}


class ReplayTTLCache:
    def __init__(self, state, seed_rows=None):
        self._state = state
        self._data = {}
        for row in seed_rows or []:
            key = str(row.get("key"))
            expiry_text = row.get("expiry")
            expiry = pd.Timestamp(expiry_text).to_pydatetime() if expiry_text else None
            self._data[key] = (row.get("value"), expiry)

    def get(self, key):
        row = self._data.get(str(key))
        if not row:
            return None
        value, expiry = row
        now = self._state["now"]
        if expiry is not None and now is not None and now >= expiry:
            self._data.pop(str(key), None)
            return None
        return value

    def set(self, key, value, ttl_seconds):
        now = self._state["now"]
        expiry = None if now is None else now + timedelta(seconds=int(ttl_seconds or 0))
        self._data[str(key)] = (value, expiry)

    def export_rows(self):
        now = self._state["now"]
        rows = []
        for key, (value, expiry) in list(self._data.items()):
            if expiry is not None and now is not None and now >= expiry:
                self._data.pop(key, None)
                continue
            rows.append(
                {
                    "key": key,
                    "value": value,
                    "expiry": expiry.isoformat() if expiry is not None else None,
                }
            )
        return rows


def build_parser():
    parser = argparse.ArgumentParser(description="Replay historical Telegram alerts from cached yf history")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--step", default="1h")
    parser.add_argument("--output", default="")
    parser.add_argument("--end-at", default="")
    parser.add_argument("--mode", choices=("direct", "batch-safe", "checkpoint"), default="direct")
    parser.add_argument("--watchlist", default=",".join(WATCHLIST))
    parser.add_argument("--checkpoint-at", default="")
    parser.add_argument("--state-input", default="")
    parser.add_argument("--result-output", default="")
    return parser


def parse_watchlist(text):
    values = [part.strip() for part in str(text or "").split(",")]
    return [part for part in values if part] or list(WATCHLIST)


def load_cache(root, watchlist):
    yf_dir = root / ".data" / "yf_history"
    cache = {}
    for symbol in watchlist:
        for interval in ("15m", "1h"):
            matches = sorted(yf_dir.glob(f"{symbol}_{interval}_adj_*.csv"))
            if not matches:
                raise FileNotFoundError(f"Missing cache for {symbol} {interval}")
            df = pd.read_csv(matches[0])
            df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
            df = df.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
            cache[(symbol, interval)] = df
    return cache


def slice_df(df, period, now):
    out = df[df.index <= now]
    if out.empty:
        return out
    delta = PERIOD_MAP.get(str(period).lower())
    if delta is not None:
        out = out[out.index >= (now - delta)]
    return out.copy()


def compute_points(cache, days, step, end_at=None):
    latest_now = min(df.index.max() for df in cache.values())
    if end_at:
        latest_now = min(pd.Timestamp(end_at), pd.Timestamp(latest_now))
    start_now = latest_now - pd.Timedelta(days=days)
    return list(pd.date_range(start=start_now, end=latest_now, freq=step))


def build_history_row(now, candidate, daily_pick):
    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": candidate.get("strategy"),
        "symbol": candidate.get("symbol"),
        "signal": candidate.get("signal"),
        "confidence": candidate.get("confidence"),
        "daily_pick": bool(daily_pick),
    }


def build_run_report_row(payload):
    return {
        "candidate_count": len(payload.get("candidates") or []),
        "sent_count": len(payload.get("sent_candidates") or []),
        "dropped_by_cache": int(payload.get("dropped_by_cache") or 0),
        "dropped_by_symbol_cap": int(payload.get("dropped_by_symbol_cap") or 0),
        "dropped_by_run_cap": int(payload.get("dropped_by_run_cap") or 0),
    }


def summarize_replay(*, history, run_reports, days, step, checkpoints, watchlist, mode, failed_checkpoints=None):
    strategy_symbol_matrix = {}
    for row in history:
        strategy = str(row.get("strategy") or "UNKNOWN")
        symbol = str(row.get("symbol") or "UNKNOWN")
        strategy_symbol_matrix.setdefault(strategy, {})
        strategy_symbol_matrix[strategy][symbol] = int(strategy_symbol_matrix[strategy].get(symbol, 0)) + 1

    return {
        "mode": mode,
        "window_days": days,
        "step": step,
        "checkpoints": checkpoints,
        "watchlist": list(watchlist),
        "symbols_count": len(watchlist),
        "total_alerts": len(history),
        "alerts_per_day_avg": (len(history) / float(days)) if days else 0.0,
        "by_strategy": dict(Counter(row["strategy"] for row in history)),
        "by_symbol": dict(Counter(row["symbol"] for row in history)),
        "by_signal": dict(Counter(row["signal"] for row in history)),
        "by_day": dict(Counter(str(row["timestamp"])[:10] for row in history)),
        "strategy_symbol_matrix": strategy_symbol_matrix,
        "avg_candidates_per_run": (
            sum(row["candidate_count"] for row in run_reports) / len(run_reports)
        )
        if run_reports
        else 0.0,
        "dispatch_drops": {
            "cache": sum(row["dropped_by_cache"] for row in run_reports),
            "symbol_cap": sum(row["dropped_by_symbol_cap"] for row in run_reports),
            "run_cap": sum(row["dropped_by_run_cap"] for row in run_reports),
        },
        "failed_checkpoints": failed_checkpoints or [],
        "failed_checkpoints_count": len(failed_checkpoints or []),
        "sample_alerts": history[:10],
    }


def run_checkpoint(root, checkpoint_at, watchlist, state_rows):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import trad

    orig_build_ctx = trad._build_alert_runtime_context
    trad._build_alert_runtime_context = lambda results, min_conf, **kwargs: orig_build_ctx(results, min_conf)

    cache = load_cache(root, watchlist)
    now = pd.Timestamp(checkpoint_at).to_pydatetime()
    state = {"now": now}
    checkpoint_history = []
    checkpoint_reports = []

    def fake_get_yf_history(symbol, period, interval=None, auto_adjust=True, cache_ttl_seconds=None):
        sym = trad.normalize_symbol(symbol)
        interval_text = str(interval or "15m").lower()
        if interval_text not in ("15m", "1h"):
            interval_text = "15m"
        df = cache.get((sym, interval_text))
        if df is None:
            return None
        sliced = slice_df(df, period, state["now"])
        return sliced if not sliced.empty else None

    trad.get_yf_history = fake_get_yf_history
    trad.get_basic_info = lambda symbol: {
        "name": trad.normalize_symbol(symbol),
        "sector": "N/A",
        "market_cap": 0,
        "pe_ratio": "N/A",
        "dividend_yield": 0,
    }
    trad._TELEGRAM_ALERT_CACHE = ReplayTTLCache(state, seed_rows=state_rows)
    trad._YF_CACHE = ReplayTTLCache(state)
    trad._YF_INFO_CACHE = ReplayTTLCache(state)
    trad.send_telegram_alert = lambda message: True
    trad._track_alert_performance = lambda *args, **kwargs: None
    trad._record_telegram_alert_history = (
        lambda candidate, min_conf=None, dynamic_min_conf=None, daily_pick=False: checkpoint_history.append(
            build_history_row(state["now"], candidate, daily_pick)
        )
    )
    trad._record_telegram_run_report = lambda **kwargs: checkpoint_reports.append(build_run_report_row(kwargs))

    trad.EMA_CROSS_15M_OPT_CACHE = {}
    results = [trad.analyze_single_symbol(symbol, "15m", include_chart_data=False) for symbol in watchlist]
    trad._notify_telegram_from_results(results)
    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "alerts": checkpoint_history,
        "run_reports": checkpoint_reports,
        "alert_cache": trad._TELEGRAM_ALERT_CACHE.export_rows(),
    }


def run_replay(days, step, watchlist=None, end_at=None):
    watchlist = list(watchlist or WATCHLIST)
    root = Path(__file__).resolve().parents[1]
    cache = load_cache(root, watchlist)
    points = compute_points(cache, days, step, end_at=end_at)
    history = []
    run_reports = []
    state_rows = []
    for now in points:
        result = run_checkpoint(root, now, watchlist, state_rows)
        history.extend(result.get("alerts") or [])
        run_reports.extend(result.get("run_reports") or [])
        state_rows = result.get("alert_cache") or []
    return summarize_replay(
        history=history,
        run_reports=run_reports,
        days=days,
        step=step,
        checkpoints=len(points),
        watchlist=watchlist,
        mode="direct",
    )


def run_replay_batch_safe(days, step, watchlist=None, end_at=None):
    watchlist = list(watchlist or WATCHLIST)
    root = Path(__file__).resolve().parents[1]
    cache = load_cache(root, watchlist)
    points = compute_points(cache, days, step, end_at=end_at)
    history = []
    run_reports = []
    failed_checkpoints = []
    state_payload = {"alert_cache": []}

    with tempfile.TemporaryDirectory(prefix="trad_replay_") as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        state_input = temp_dir / "state.json"
        result_output = temp_dir / "result.json"
        for now in points:
            state_input.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if result_output.exists():
                result_output.unlink()
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--mode",
                    "checkpoint",
                    "--checkpoint-at",
                    pd.Timestamp(now).isoformat(),
                    "--watchlist",
                    ",".join(watchlist),
                    "--state-input",
                    str(state_input),
                    "--result-output",
                    str(result_output),
                ],
                cwd=str(root),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 or not result_output.exists():
                failed_checkpoints.append(
                    {
                        "timestamp": pd.Timestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                        "returncode": int(proc.returncode),
                        "stdout_tail": (proc.stdout or "")[-500:],
                        "stderr_tail": (proc.stderr or "")[-500:],
                    }
                )
                continue
            result = json.loads(result_output.read_text(encoding="utf-8"))
            history.extend(result.get("alerts") or [])
            run_reports.extend(result.get("run_reports") or [])
            state_payload = {"alert_cache": result.get("alert_cache") or []}

    return summarize_replay(
        history=history,
        run_reports=run_reports,
        days=days,
        step=step,
        checkpoints=len(points),
        watchlist=watchlist,
        mode="batch-safe",
        failed_checkpoints=failed_checkpoints,
    )


def write_json(path_text, payload):
    if not path_text:
        return
    output_path = Path(path_text)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = build_parser().parse_args()
    watchlist = parse_watchlist(args.watchlist)

    if args.mode == "checkpoint":
        if not args.checkpoint_at or not args.result_output:
            raise ValueError("checkpoint mode requires --checkpoint-at and --result-output")
        state_rows = []
        if args.state_input:
            state_payload = json.loads(Path(args.state_input).read_text(encoding="utf-8"))
            state_rows = list((state_payload or {}).get("alert_cache") or [])
        payload = run_checkpoint(Path(__file__).resolve().parents[1], args.checkpoint_at, watchlist, state_rows)
        write_json(args.result_output, payload)
        print(json.dumps({"ok": True, "timestamp": payload["timestamp"]}, ensure_ascii=False))
        return

    if args.mode == "batch-safe":
        summary = run_replay_batch_safe(args.days, args.step, watchlist=watchlist, end_at=args.end_at or None)
    else:
        summary = run_replay(args.days, args.step, watchlist=watchlist, end_at=args.end_at or None)

    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
