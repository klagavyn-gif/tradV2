import os, glob, json
from datetime import timedelta
from collections import Counter
import pandas as pd
import trad

SYMS = ['BTC-USD','ETH-USD','BNB-USD','ADA-USD','SOL-USD','XRP-USD','DOGE-USD','LINK-USD','AVAX-USD','TRX-USD','NEAR-USD']
BASE = r'e:\TRAD\DDD\tradV2\.data\yf_history'
OUT = r'e:\TRAD\DDD\tradV2\_forecast_replay_out.json'
DAYS = 7
EXPECTED = ['SS15','AW15','CDCVIX15','AZ15','PA15','TCB15','PRIMARY','DAILY_BEST']

def load_csv(sym, interval):
    pattern = os.path.join(BASE, f'{sym}_{interval}_adj_*.csv')
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    df = pd.read_csv(paths[-1], parse_dates=['Datetime'])
    df = df.set_index('Datetime').sort_index()
    return df

data15 = {}
data1h = {}
datad = {}
for sym in SYMS:
    df15 = load_csv(sym, '15m')
    df1h = load_csv(sym, '1h')
    if df15 is None or df1h is None:
        continue
    data15[sym] = df15
    data1h[sym] = df1h
    daily = df15.resample('1D').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    datad[sym] = daily
SYMS = [s for s in SYMS if s in data15 and s in data1h]
if not SYMS:
    raise SystemExit('no symbols with cached data')

ref = data15[SYMS[0]].index
end_ts = ref.max()
start_ts = end_ts - pd.Timedelta(days=DAYS)
replay_times = [ts for ts in ref if ts >= start_ts]

state = {'now': end_ts}
old_now = trad.get_thai_now
old_get_yf = trad.get_yf_history
old_basic = trad.get_basic_info


def fake_now():
    return state['now'].to_pydatetime().replace(tzinfo=None)


def fake_basic(symbol):
    sym = trad.normalize_symbol(symbol)
    return {'name': sym, 'sector': 'N/A', 'market_cap': 0, 'pe_ratio': 'N/A', 'dividend_yield': 0}


def slice_df(df, period):
    if df is None or df.empty:
        return df
    try:
        return trad._slice_history_by_period(df, period)
    except Exception:
        return df


def fake_get_yf_history(symbol, period, interval=None, auto_adjust=True, cache_ttl_seconds=None):
    sym = trad.normalize_symbol(symbol)
    if interval == '15m':
        df = data15.get(sym)
    elif interval == '1h':
        df = data1h.get(sym)
    else:
        df = datad.get(sym)
    if df is None or df.empty:
        return None
    df = df[df.index <= state['now']]
    if df.empty:
        return None
    df = slice_df(df, period)
    return df.copy()

trad.get_thai_now = fake_now
trad.get_yf_history = fake_get_yf_history
trad.get_basic_info = fake_basic

try:
    min_conf_base = float(getattr(trad.config, 'TELEGRAM_ALERT_MIN_CONFIDENCE', 75.0))
    max_per_run = max(1, int(getattr(trad.config, 'TELEGRAM_ALERT_MAX_PER_RUN', 5)))
    max_per_symbol = max(1, int(getattr(trad.config, 'TELEGRAM_ALERT_MAX_PER_SYMBOL', 1)))
    cooldown_minutes = max(1, int(getattr(trad.config, 'TELEGRAM_ALERT_COOLDOWN_MINUTES', 30)))
    cooldown_delta = timedelta(minutes=cooldown_minutes)
    daybest_delta = timedelta(hours=26)

    sent_counts = Counter()
    raw_candidate_counts = Counter()
    quality_counts = Counter()
    cache_expiry = {}
    sampled_runs = 0

    for ts in replay_times:
        state['now'] = ts
        # drop expired cache entries
        expired = [k for k, v in cache_expiry.items() if v <= fake_now()]
        for k in expired:
            cache_expiry.pop(k, None)

        results = [trad.analyze_single_symbol(sym, '1mo', include_chart_data=False) for sym in SYMS]
        kill, _ = trad._telegram_kill_switch_state(results)
        dynamic_min_conf = trad._telegram_dynamic_conf_threshold(min_conf_base, results)
        candidates = []
        build_stats = {}
        if not kill:
            candidates, build_stats = trad._build_telegram_candidates(results, dynamic_min_conf)
            daily_candidates = trad._build_cdc_daily_trend_candidates(results, existing_candidates=candidates, min_conf=dynamic_min_conf)
            if daily_candidates:
                candidates.extend([c for c in daily_candidates if isinstance(c, dict)])
            if trad._is_daily_best_pick_window():
                daily_best = trad._build_daily_best_pick_candidates(results)
                if daily_best:
                    candidates.extend([c for c in daily_best if isinstance(c, dict)])
        for k, v in (build_stats.get('quality_drop_counts') or {}).items():
            quality_counts[k] += int(v)
        for c in candidates:
            raw_candidate_counts[str(c.get('strategy') or 'UNKNOWN').upper()] += 1

        candidates.sort(key=lambda c: (float(c.get('score', 0.0)), float(c.get('confidence', 0.0))), reverse=True)
        per_symbol_sent = Counter()
        sent = 0
        for c in candidates:
            if sent >= max_per_run:
                break
            symbol = str(c.get('symbol') or '')
            if not symbol:
                continue
            if per_symbol_sent[symbol] >= max_per_symbol:
                continue
            strategy = str(c.get('strategy') or 'UNKNOWN').upper()
            if strategy == 'DAILY_BEST':
                cache_key = f"DAILYBEST|{fake_now().strftime('%Y%m%d')}|{c.get('symbol')}|{c.get('signal')}"
                ttl = daybest_delta
            else:
                cache_key = str(c.get('cache_key') or '').strip()
                ttl = cooldown_delta
            if not cache_key:
                continue
            if cache_key in cache_expiry and cache_expiry[cache_key] > fake_now():
                continue
            cache_expiry[cache_key] = fake_now() + ttl
            per_symbol_sent[symbol] += 1
            sent += 1
            sent_counts[str(c.get('strategy') or 'UNKNOWN').upper()] += 1
        sampled_runs += 1

    span_days = max((replay_times[-1] - replay_times[0]).total_seconds() / 86400.0, 1.0)
    out = {
        'symbols': SYMS,
        'replay_days': DAYS,
        'sampled_runs': sampled_runs,
        'history_start': replay_times[0].strftime('%Y-%m-%d %H:%M:%S'),
        'history_end': replay_times[-1].strftime('%Y-%m-%d %H:%M:%S'),
        'span_days': round(span_days, 4),
        'sent_count_by_strategy': {k: int(sent_counts.get(k, 0)) for k in EXPECTED},
        'raw_candidate_count_by_strategy': {k: int(raw_candidate_counts.get(k, 0)) for k in EXPECTED},
        'forecast_alerts_per_30d': {k: round(float(sent_counts.get(k, 0)) / span_days * 30.0, 2) for k in EXPECTED},
        'forecast_raw_candidates_per_30d': {k: round(float(raw_candidate_counts.get(k, 0)) / span_days * 30.0, 2) for k in EXPECTED},
        'quality_drop_counts_top': dict(quality_counts.most_common(12)),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('ok')
finally:
    trad.get_thai_now = old_now
    trad.get_yf_history = old_get_yf
    trad.get_basic_info = old_basic
