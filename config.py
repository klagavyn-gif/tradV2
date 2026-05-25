import os

def _env_str(name, default=""):
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v)
    return v if v != "" else default


def _env_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name, default=0):
    v = os.getenv(name)
    if v is None:
        return int(default)
    s = str(v).strip()
    if not s:
        return int(default)
    try:
        return int(s)
    except Exception:
        return int(default)


def _env_float(name, default=0.0):
    v = os.getenv(name)
    if v is None:
        return float(default)
    s = str(v).strip()
    if not s:
        return float(default)
    try:
        return float(s)
    except Exception:
        return float(default)


def _env_csv_set(name, default=""):
    raw = _env_str(name, default)
    values = []
    for part in str(raw or "").split(","):
        text = str(part or "").strip().upper()
        if text:
            values.append(text)
    return set(values)


DATABASE_URI = _env_str("DATABASE_URI", "sqlite:///trading_app.db")

SECRET_KEY = _env_str("SECRET_KEY", "")

FLASK_DEBUG = _env_bool("FLASK_DEBUG", True)
PORT = _env_int("PORT", 5000)

CURL_IMPERSONATE = _env_str("CURL_IMPERSONATE", "chrome110")
HTTP_VERIFY = _env_bool("HTTP_VERIFY", True)
HTTP_CA_BUNDLE = _env_str("HTTP_CA_BUNDLE", "")

YF_CACHE_MAXSIZE = _env_int("YF_CACHE_MAXSIZE", 256)
YF_CACHE_TTL_SECONDS = _env_int("YF_CACHE_TTL_SECONDS", 180)
YF_INFO_CACHE_TTL_SECONDS = _env_int("YF_INFO_CACHE_TTL_SECONDS", 6 * 60 * 60)
YF_HISTORY_STORE_ENABLE = _env_bool("YF_HISTORY_STORE_ENABLE", True)
YF_HISTORY_STORE_MAX_ROWS = _env_int("YF_HISTORY_STORE_MAX_ROWS", 50000)
YF_PREFER_CHART_API = _env_bool("YF_PREFER_CHART_API", False)
YF_FETCH_MAX_RETRIES = _env_int("YF_FETCH_MAX_RETRIES", 3)
YF_RETRY_BACKOFF_SECONDS = _env_float("YF_RETRY_BACKOFF_SECONDS", 1.25)
YF_SOURCE_HEALTH_RECENT_EVENTS = _env_int("YF_SOURCE_HEALTH_RECENT_EVENTS", 40)
YF_DISK_FALLBACK_ENABLE = _env_bool("YF_DISK_FALLBACK_ENABLE", True)
YF_DISK_FALLBACK_MAX_AGE_MINUTES = _env_float("YF_DISK_FALLBACK_MAX_AGE_MINUTES", 720.0)
YF_DISK_FALLBACK_MAX_STALE_BARS = _env_int("YF_DISK_FALLBACK_MAX_STALE_BARS", 8)
YF_DISK_FALLBACK_GRACE_MINUTES = _env_float("YF_DISK_FALLBACK_GRACE_MINUTES", 5.0)
YF_AUTH_DISK_FALLBACK_ENABLE = _env_bool("YF_AUTH_DISK_FALLBACK_ENABLE", True)
YF_AUTH_DISK_FALLBACK_GITHUB_ONLY = _env_bool("YF_AUTH_DISK_FALLBACK_GITHUB_ONLY", True)
YF_AUTH_DISK_FALLBACK_MAX_AGE_MINUTES = _env_float("YF_AUTH_DISK_FALLBACK_MAX_AGE_MINUTES", 1440.0)
YF_AUTH_DISK_FALLBACK_MAX_STALE_BARS = _env_int("YF_AUTH_DISK_FALLBACK_MAX_STALE_BARS", 12)
YF_AUTH_DISK_FALLBACK_GRACE_MINUTES = _env_float("YF_AUTH_DISK_FALLBACK_GRACE_MINUTES", 5.0)
ANALYZE_MAX_WORKERS = _env_int("ANALYZE_MAX_WORKERS", 5)
VERIFY_OUTPUT_PATH = _env_str("VERIFY_OUTPUT_PATH", ".data/telegram_alerts/verify_output.json")
VERIFY_INCLUDE_RESULTS = _env_bool("VERIFY_INCLUDE_RESULTS", False)

TELEGRAM_ALERT_TTL_SECONDS = _env_int("TELEGRAM_ALERT_TTL_SECONDS", 1800)
TELEGRAM_ALERT_MIN_CONFIDENCE = _env_float("TELEGRAM_ALERT_MIN_CONFIDENCE", 69.0)
TELEGRAM_ALERT_MAX_PER_RUN = _env_int("TELEGRAM_ALERT_MAX_PER_RUN", 3)
TELEGRAM_ALERT_MAX_PER_SYMBOL = _env_int("TELEGRAM_ALERT_MAX_PER_SYMBOL", 1)
TELEGRAM_ALERT_COOLDOWN_MINUTES = _env_int("TELEGRAM_ALERT_COOLDOWN_MINUTES", 90)
TELEGRAM_ALERT_DYNAMIC_CONF_ENABLE = _env_bool("TELEGRAM_ALERT_DYNAMIC_CONF_ENABLE", True)
TELEGRAM_ALERT_DYNAMIC_CONF_VOL_REF_PCT = _env_float("TELEGRAM_ALERT_DYNAMIC_CONF_VOL_REF_PCT", 1.8)
TELEGRAM_ALERT_DYNAMIC_CONF_VOL_MULT = _env_float("TELEGRAM_ALERT_DYNAMIC_CONF_VOL_MULT", 1.8)
TELEGRAM_ALERT_ENTRY_QUALITY_ENABLE = _env_bool("TELEGRAM_ALERT_ENTRY_QUALITY_ENABLE", True)
TELEGRAM_ALERT_ENTRY_MIN_HIST_WIN_RATE = _env_float("TELEGRAM_ALERT_ENTRY_MIN_HIST_WIN_RATE", 57.0)
TELEGRAM_ALERT_ENTRY_MIN_HIST_TRADES = _env_int("TELEGRAM_ALERT_ENTRY_MIN_HIST_TRADES", 6)
TELEGRAM_ALERT_ENTRY_MIN_EXPECTANCY_RR = _env_float("TELEGRAM_ALERT_ENTRY_MIN_EXPECTANCY_RR", 0.03)
TELEGRAM_ALERT_ENTRY_REQUIRE_EDGE_METRICS = _env_bool("TELEGRAM_ALERT_ENTRY_REQUIRE_EDGE_METRICS", True)
TELEGRAM_ALERT_ENTRY_REQUIRE_WALKFORWARD = _env_bool("TELEGRAM_ALERT_ENTRY_REQUIRE_WALKFORWARD", True)
TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_WIN_RATE = _env_float("TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_WIN_RATE", 58.0)
TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_EXPECTANCY_RR = _env_float("TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_EXPECTANCY_RR", 0.03)
TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_VALID_TRADES = _env_int("TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_VALID_TRADES", 6)
TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_ROBUSTNESS = _env_float("TELEGRAM_ALERT_ENTRY_WALKFORWARD_MIN_ROBUSTNESS", 40.0)
TELEGRAM_ALERT_STRICT_60_MODE = _env_bool("TELEGRAM_ALERT_STRICT_60_MODE", True)
TELEGRAM_ALERT_STRICT_60_EXCLUDE_CDC = _env_bool("TELEGRAM_ALERT_STRICT_60_EXCLUDE_CDC", False)
TELEGRAM_ALERT_SELL_WHITELIST_ENABLE = _env_bool("TELEGRAM_ALERT_SELL_WHITELIST_ENABLE", True)
TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_WIN_RATE = _env_float("TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_WIN_RATE", 58.0)
TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_EXPECTANCY_RR = _env_float("TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_EXPECTANCY_RR", 0.03)
TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_VALID_TRADES = _env_int("TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_VALID_TRADES", 6)
TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_ROBUSTNESS = _env_float("TELEGRAM_ALERT_SELL_WALKFORWARD_MIN_ROBUSTNESS", 40.0)
TELEGRAM_ALERT_PRIMARY_MIN_SOURCES = _env_int("TELEGRAM_ALERT_PRIMARY_MIN_SOURCES", 2)
TELEGRAM_ALERT_PRIMARY_SINGLE_SOURCE_MIN_CONF = _env_float("TELEGRAM_ALERT_PRIMARY_SINGLE_SOURCE_MIN_CONF", 84.0)
TELEGRAM_ALERT_HIGH_WIN_RATE_THRESHOLD = _env_float("TELEGRAM_ALERT_HIGH_WIN_RATE_THRESHOLD", 60.0)
TELEGRAM_ALERT_HIGH_CONFIDENCE_THRESHOLD = _env_float("TELEGRAM_ALERT_HIGH_CONFIDENCE_THRESHOLD", 82.0)
TELEGRAM_ALERT_HISTORY_ENABLED = _env_bool("TELEGRAM_ALERT_HISTORY_ENABLED", True)
TELEGRAM_ALERT_HISTORY_MAX_ROWS = _env_int("TELEGRAM_ALERT_HISTORY_MAX_ROWS", 5000)
TELEGRAM_ALERT_HISTORY_EXPORT_CSV = _env_bool("TELEGRAM_ALERT_HISTORY_EXPORT_CSV", True)
TELEGRAM_ALERT_REALIZED_ENABLED = _env_bool("TELEGRAM_ALERT_REALIZED_ENABLED", True)
TELEGRAM_ALERT_REALIZED_INTERVAL = _env_str("TELEGRAM_ALERT_REALIZED_INTERVAL", "15m")
TELEGRAM_ALERT_REALIZED_MAX_HOLD_BARS = _env_int("TELEGRAM_ALERT_REALIZED_MAX_HOLD_BARS", 64)
TELEGRAM_ALERT_REALIZED_REPORT_DAYS = _env_int("TELEGRAM_ALERT_REALIZED_REPORT_DAYS", 45)
TELEGRAM_ALERT_REALIZED_EXPORT_OUTCOMES = _env_bool("TELEGRAM_ALERT_REALIZED_EXPORT_OUTCOMES", True)
TELEGRAM_ALERT_REGIME_ENABLED = _env_bool("TELEGRAM_ALERT_REGIME_ENABLED", True)
TELEGRAM_ALERT_REGIME_BLOCK_ENABLED = _env_bool("TELEGRAM_ALERT_REGIME_BLOCK_ENABLED", True)
TELEGRAM_ALERT_REGIME_SCORE_MULTIPLIER_ENABLED = _env_bool("TELEGRAM_ALERT_REGIME_SCORE_MULTIPLIER_ENABLED", True)
TELEGRAM_ALERT_REGIME_BUDGET_ENABLED = _env_bool("TELEGRAM_ALERT_REGIME_BUDGET_ENABLED", True)
TELEGRAM_ALERT_REGIME_MIN_CONFIDENCE_UPLIFT = _env_float("TELEGRAM_ALERT_REGIME_MIN_CONFIDENCE_UPLIFT", 4.0)
TELEGRAM_ALERT_REGIME_RISK_OFF_CONFIDENCE_UPLIFT = _env_float("TELEGRAM_ALERT_REGIME_RISK_OFF_CONFIDENCE_UPLIFT", 6.0)
TELEGRAM_ALERT_REGIME_OPPOSING_SIDE_MIN_CONFIDENCE = _env_float("TELEGRAM_ALERT_REGIME_OPPOSING_SIDE_MIN_CONFIDENCE", 84.0)
TREND_STATE_ALERT_ENABLED = _env_bool("TREND_STATE_ALERT_ENABLED", False)
TREND_STATE_ALERT_MIN_SCORE = _env_float("TREND_STATE_ALERT_MIN_SCORE", 78.0)
TREND_STATE_ALERT_STRONG_1H_MIN_CONFIDENCE = _env_float("TREND_STATE_ALERT_STRONG_1H_MIN_CONFIDENCE", 68.0)
TREND_STATE_ALERT_MIN_DIRECTIONAL_SOURCES = _env_int("TREND_STATE_ALERT_MIN_DIRECTIONAL_SOURCES", 2)
TREND_STATE_ALERT_COOLDOWN_MINUTES = _env_int("TREND_STATE_ALERT_COOLDOWN_MINUTES", 360)
TREND_STATE_ALERT_MAX_PER_RUN = _env_int("TREND_STATE_ALERT_MAX_PER_RUN", 2)
TREND_STATE_ALERT_SUPPRESS_IF_PRIMARY_SENT = _env_bool("TREND_STATE_ALERT_SUPPRESS_IF_PRIMARY_SENT", True)
TREND_STATE_ALERT_REQUIRE_REGIME_CONFIRMATION = _env_bool("TREND_STATE_ALERT_REQUIRE_REGIME_CONFIRMATION", True)
TELEGRAM_ALERT_RUN_REPORT_ENABLED = _env_bool("TELEGRAM_ALERT_RUN_REPORT_ENABLED", True)
TELEGRAM_ALERT_RUN_REPORT_MAX_ROWS = _env_int("TELEGRAM_ALERT_RUN_REPORT_MAX_ROWS", 500)
TELEGRAM_ALERT_RUN_REPORT_TOP_CANDIDATES = _env_int("TELEGRAM_ALERT_RUN_REPORT_TOP_CANDIDATES", 5)
TELEGRAM_ALERT_AUTO_TUNE_ENABLE = _env_bool("TELEGRAM_ALERT_AUTO_TUNE_ENABLE", True)
TELEGRAM_ALERT_AUTO_TUNE_HISTORY_DAYS = _env_int("TELEGRAM_ALERT_AUTO_TUNE_HISTORY_DAYS", 45)
TELEGRAM_ALERT_AUTO_TUNE_MIN_ALERTS_PER_SYMBOL = _env_int("TELEGRAM_ALERT_AUTO_TUNE_MIN_ALERTS_PER_SYMBOL", 12)
TELEGRAM_ALERT_AUTO_TUNE_MIN_ALERTS_PER_STRATEGY = _env_int("TELEGRAM_ALERT_AUTO_TUNE_MIN_ALERTS_PER_STRATEGY", 20)
TELEGRAM_ALERT_AUTO_TUNE_TARGET_ALERTS_PER_DAY = _env_float("TELEGRAM_ALERT_AUTO_TUNE_TARGET_ALERTS_PER_DAY", 2.0)
TELEGRAM_ALERT_AUTO_TUNE_TARGET_DAILY_PICKS_PER_DAY = _env_float("TELEGRAM_ALERT_AUTO_TUNE_TARGET_DAILY_PICKS_PER_DAY", 1.0)
TELEGRAM_ALERT_AUTO_TUNE_OUTPUT_PATH = _env_str(
    "TELEGRAM_ALERT_AUTO_TUNE_OUTPUT_PATH",
    ".data/telegram_alerts/auto_tuned_thresholds.json",
)
TELEGRAM_DAILY_BEST_PICK_ENABLED = _env_bool("TELEGRAM_DAILY_BEST_PICK_ENABLED", True)
TELEGRAM_DAILY_BEST_PICK_HOUR = _env_int("TELEGRAM_DAILY_BEST_PICK_HOUR", 9)
TELEGRAM_DAILY_BEST_PICK_MINUTE = _env_int("TELEGRAM_DAILY_BEST_PICK_MINUTE", 0)
TELEGRAM_DAILY_BEST_PICK_WINDOW_MINUTES = _env_int("TELEGRAM_DAILY_BEST_PICK_WINDOW_MINUTES", 30)
TELEGRAM_DAILY_BEST_PICK_MIN_CONFIDENCE = _env_float("TELEGRAM_DAILY_BEST_PICK_MIN_CONFIDENCE", 60.0)
TELEGRAM_DAILY_BEST_PICK_MIN_SCORE = _env_float("TELEGRAM_DAILY_BEST_PICK_MIN_SCORE", 74.0)
TELEGRAM_DAILY_BEST_PICK_MAX_PER_DAY = _env_int("TELEGRAM_DAILY_BEST_PICK_MAX_PER_DAY", 2)
TELEGRAM_DAILY_BEST_PICK_REQUIRE_QUALITY = _env_bool("TELEGRAM_DAILY_BEST_PICK_REQUIRE_QUALITY", True)
TELEGRAM_DAILY_BEST_PICK_ALLOW_CDC = _env_bool("TELEGRAM_DAILY_BEST_PICK_ALLOW_CDC", True)
TELEGRAM_DAILY_BEST_PICK_RELAXED_ENABLE = _env_bool("TELEGRAM_DAILY_BEST_PICK_RELAXED_ENABLE", True)
TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_CONFIDENCE = _env_float("TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_CONFIDENCE", 57.0)
TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_SCORE = _env_float("TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_SCORE", 68.0)
TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_HIST_WIN_RATE = _env_float("TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_HIST_WIN_RATE", 55.0)
TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_HIST_TRADES = _env_int("TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_HIST_TRADES", 4)
TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_EXPECTANCY_RR = _env_float("TELEGRAM_DAILY_BEST_PICK_RELAXED_MIN_EXPECTANCY_RR", 0.0)
TELEGRAM_DAILY_BEST_PICK_SYMBOL_ALLOWLIST = _env_csv_set(
    "TELEGRAM_DAILY_BEST_PICK_SYMBOL_ALLOWLIST",
    "BTC-USD,DOGE-USD,ETH-USD,ADA-USD,XRP-USD,BNB-USD,SOL-USD,TRX-USD,NEAR-USD,LINK-USD,PAXG-USD",
)
TELEGRAM_DAILY_BEST_PICK_CDC_ENABLE = _env_bool("TELEGRAM_DAILY_BEST_PICK_CDC_ENABLE", True)
TELEGRAM_DAILY_BEST_PICK_CDC_MIN_RED_TO_GREEN_SCORE = _env_float("TELEGRAM_DAILY_BEST_PICK_CDC_MIN_RED_TO_GREEN_SCORE", 68.0)
TELEGRAM_DAILY_BEST_PICK_CDC_MAX_BARS_SINCE_GREEN_FLIP = _env_int("TELEGRAM_DAILY_BEST_PICK_CDC_MAX_BARS_SINCE_GREEN_FLIP", 3)
TELEGRAM_DAILY_BEST_PICK_CDC_REQUIRE_RECLAIM = _env_bool("TELEGRAM_DAILY_BEST_PICK_CDC_REQUIRE_RECLAIM", True)
TELEGRAM_DAILY_SUMMARY_ENABLED = _env_bool("TELEGRAM_DAILY_SUMMARY_ENABLED", True)
TELEGRAM_DAILY_SUMMARY_TOP_N = _env_int("TELEGRAM_DAILY_SUMMARY_TOP_N", 3)
TELEGRAM_KILL_SWITCH_ENABLED = _env_bool("TELEGRAM_KILL_SWITCH_ENABLED", False)
TELEGRAM_KILL_SWITCH_MIN_SYMBOLS = _env_int("TELEGRAM_KILL_SWITCH_MIN_SYMBOLS", 4)
TELEGRAM_KILL_SWITCH_SELL_RATIO = _env_float("TELEGRAM_KILL_SWITCH_SELL_RATIO", 0.50)
TELEGRAM_KILL_SWITCH_TREND_MISMATCH_RATIO = _env_float("TELEGRAM_KILL_SWITCH_TREND_MISMATCH_RATIO", 0.60)
TELEGRAM_KILL_SWITCH_HIGH_VOL_PCT = _env_float("TELEGRAM_KILL_SWITCH_HIGH_VOL_PCT", 4.0)
TELEGRAM_KILL_SWITCH_HIGH_VOL_RATIO = _env_float("TELEGRAM_KILL_SWITCH_HIGH_VOL_RATIO", 0.50)
TELEGRAM_ALERT_STRATEGY_QUALITY_PROFILES = {
    "SS15": {
        "min_confidence": 64.0,
        "min_win_rate_pct": 65.0,
        "min_expectancy_rr": 0.15,
        "min_trades": 8,
    },
    "AW15": {
        "min_confidence": 68.0,
        "min_score": 76.0,
        "min_win_rate_pct": 56.0,
        "min_expectancy_rr": 0.03,
        "min_trades": 6,
    },
    "CDCVIX15": {
        "buy_min_confidence": 66.0,
        "sell_min_confidence": 70.0,
        "min_score": 74.0,
        "buy_min_win_rate_pct": 58.0,
        "sell_min_win_rate_pct": 60.0,
        "buy_min_expectancy_rr": 0.04,
        "sell_min_expectancy_rr": 0.05,
        "min_trades": 8,
        "sell_min_robustness_score": 45.0,
    },
    "AZ15": {
        "min_confidence": 76.0,
        "min_score": 80.0,
        "min_win_rate_pct": 58.0,
        "min_expectancy_rr": 0.05,
        "min_trades": 8,
    },
    "PA15": {
        "min_confidence": 67.0,
        "min_score": 72.0,
        "min_win_rate_pct": 57.0,
        "min_expectancy_rr": 0.04,
        "min_trades": 8,
    },
    "TCB15": {
        "min_confidence": 68.0,
        "min_score": 73.0,
        "min_win_rate_pct": 57.0,
        "min_expectancy_rr": 0.04,
        "min_trades": 8,
    },
    "PRIMARY": {
        "min_confidence": 76.0,
        "min_score": 82.0,
        "min_win_rate_pct": 60.0,
        "min_expectancy_rr": 0.06,
        "min_trades": 8,
        "min_source_count": 2,
        "single_source_min_confidence": 90.0,
    },
    "DAILY_BEST": {
        "min_confidence": 64.0,
        "min_score": 80.0,
        "min_win_rate_pct": 58.0,
        "min_expectancy_rr": 0.04,
        "min_trades": 6,
        "min_source_count": 1,
    },
}
TELEGRAM_ALERT_SYMBOL_QUALITY_PROFILES = {
    "BTC-USD": {
        "buy_min_confidence": 74.0,
        "sell_min_confidence": 75.0,
        "min_score": 84.0,
        "buy_min_win_rate_pct": 58.0,
        "sell_min_win_rate_pct": 60.0,
        "buy_min_expectancy_rr": 0.05,
        "sell_min_expectancy_rr": 0.05,
        "min_trades": 8,
    },
    "DOGE-USD": {
        "buy_min_confidence": 70.0,
        "sell_min_confidence": 71.0,
        "min_score": 80.0,
        "buy_min_win_rate_pct": 57.0,
        "sell_min_win_rate_pct": 59.0,
        "buy_min_expectancy_rr": 0.04,
        "sell_min_expectancy_rr": 0.04,
        "min_trades": 8,
    },
    "ETH-USD": {
        "buy_min_confidence": 72.0,
        "sell_min_confidence": 73.0,
        "min_score": 82.0,
        "buy_min_win_rate_pct": 58.0,
        "sell_min_win_rate_pct": 60.0,
        "buy_min_expectancy_rr": 0.05,
        "sell_min_expectancy_rr": 0.05,
        "min_trades": 8,
    },
    "ADA-USD": {
        "buy_min_confidence": 71.0,
        "sell_min_confidence": 72.0,
        "min_score": 81.0,
        "buy_min_win_rate_pct": 58.0,
        "sell_min_win_rate_pct": 60.0,
        "buy_min_expectancy_rr": 0.05,
        "sell_min_expectancy_rr": 0.05,
        "min_trades": 8,
    },
    "XRP-USD": {
        "buy_min_confidence": 72.0,
        "sell_min_confidence": 73.0,
        "min_score": 82.0,
        "buy_min_win_rate_pct": 58.0,
        "sell_min_win_rate_pct": 60.0,
        "buy_min_expectancy_rr": 0.05,
        "sell_min_expectancy_rr": 0.05,
        "min_trades": 8,
    },
    "BNB-USD": {
        "buy_min_confidence": 75.0,
        "sell_min_confidence": 76.0,
        "min_score": 86.0,
        "buy_min_win_rate_pct": 59.0,
        "sell_min_win_rate_pct": 61.0,
        "buy_min_expectancy_rr": 0.06,
        "sell_min_expectancy_rr": 0.05,
        "min_trades": 8,
    },
    "SOL-USD": {
        "buy_min_confidence": 71.0,
        "sell_min_confidence": 72.0,
        "min_score": 81.0,
        "buy_min_win_rate_pct": 58.0,
        "sell_min_win_rate_pct": 60.0,
        "buy_min_expectancy_rr": 0.05,
        "sell_min_expectancy_rr": 0.05,
        "min_trades": 8,
    },
    "TRX-USD": {
        "buy_min_confidence": 78.0,
        "sell_min_confidence": 79.0,
        "min_score": 88.0,
        "buy_min_win_rate_pct": 61.0,
        "sell_min_win_rate_pct": 62.0,
        "buy_min_expectancy_rr": 0.08,
        "sell_min_expectancy_rr": 0.06,
        "min_trades": 8,
        "min_robustness_score": 50.0,
    },
    "NEAR-USD": {
        "buy_min_confidence": 69.0,
        "sell_min_confidence": 70.0,
        "min_score": 78.0,
        "buy_min_win_rate_pct": 57.0,
        "sell_min_win_rate_pct": 59.0,
        "buy_min_expectancy_rr": 0.04,
        "sell_min_expectancy_rr": 0.04,
        "min_trades": 8,
    },
    "LINK-USD": {
        "buy_min_confidence": 73.0,
        "sell_min_confidence": 74.0,
        "min_score": 83.0,
        "buy_min_win_rate_pct": 58.0,
        "sell_min_win_rate_pct": 60.0,
        "buy_min_expectancy_rr": 0.05,
        "sell_min_expectancy_rr": 0.05,
        "min_trades": 8,
    },
    "PAXG-USD": {
        "buy_min_confidence": 77.0,
        "sell_min_confidence": 78.0,
        "min_score": 88.0,
        "buy_min_win_rate_pct": 60.0,
        "sell_min_win_rate_pct": 61.0,
        "buy_min_expectancy_rr": 0.07,
        "sell_min_expectancy_rr": 0.06,
        "min_trades": 8,
        "min_robustness_score": 50.0,
    },
}

ALL_WEATHER_15M_ENABLED = _env_bool("ALL_WEATHER_15M_ENABLED", True)
ALL_WEATHER_15M_MIN_ALERT_CONFIDENCE = _env_float("ALL_WEATHER_15M_MIN_ALERT_CONFIDENCE", 67.0)
ALL_WEATHER_15M_MIN_VALID_TRADES = _env_int("ALL_WEATHER_15M_MIN_VALID_TRADES", 4)
ALL_WEATHER_15M_MIN_VALID_WIN_RATE = _env_float("ALL_WEATHER_15M_MIN_VALID_WIN_RATE", 54.0)
ALL_WEATHER_15M_MIN_VALID_EXPECTANCY = _env_float("ALL_WEATHER_15M_MIN_VALID_EXPECTANCY", 0.01)
ALL_WEATHER_15M_MIN_CONFLUENCE = _env_int("ALL_WEATHER_15M_MIN_CONFLUENCE", 1)
ALL_WEATHER_15M_REQUIRE_BACKTEST_PASS = _env_bool("ALL_WEATHER_15M_REQUIRE_BACKTEST_PASS", True)
ALL_WEATHER_15M_TOP_K = _env_int("ALL_WEATHER_15M_TOP_K", 2)
ALL_WEATHER_15M_DELTA = _env_float("ALL_WEATHER_15M_DELTA", 2.5)
ALL_WEATHER_15M_HIGH_VOL_THRESHOLD = _env_float("ALL_WEATHER_15M_HIGH_VOL_THRESHOLD", 1.8)
ALL_WEATHER_15M_TREND_THRESHOLD = _env_float("ALL_WEATHER_15M_TREND_THRESHOLD", 1.35)
ALL_WEATHER_15M_MIN_PERIOD_PASSES = _env_int("ALL_WEATHER_15M_MIN_PERIOD_PASSES", 2)
ALL_WEATHER_15M_MIN_SYMBOL_READY_RATE = _env_float("ALL_WEATHER_15M_MIN_SYMBOL_READY_RATE", 35.0)

EMA_CROSS_15M_ENABLE_OPTIMIZATION = _env_bool(
    "EMA_CROSS_15M_ENABLE_OPTIMIZATION",
    os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true",
)
EMA_CROSS_15M_YF_PERIOD = _env_str("EMA_CROSS_15M_YF_PERIOD", "60d")
EMA_CROSS_15M_FAST_MIN = _env_int("EMA_CROSS_15M_FAST_MIN", 6)
EMA_CROSS_15M_FAST_MAX = _env_int("EMA_CROSS_15M_FAST_MAX", 24)
EMA_CROSS_15M_FAST_STEP = _env_int("EMA_CROSS_15M_FAST_STEP", 2)
EMA_CROSS_15M_SLOW_MIN = _env_int("EMA_CROSS_15M_SLOW_MIN", 18)
EMA_CROSS_15M_SLOW_MAX = _env_int("EMA_CROSS_15M_SLOW_MAX", 80)
EMA_CROSS_15M_SLOW_STEP = _env_int("EMA_CROSS_15M_SLOW_STEP", 2)
EMA_CROSS_15M_MIN_TRADES = _env_int("EMA_CROSS_15M_MIN_TRADES", 8)
EMA_CROSS_15M_MIN_VALID_TRADES = _env_int("EMA_CROSS_15M_MIN_VALID_TRADES", 4)
EMA_CROSS_15M_OPT_TRAIN_RATIO = _env_float("EMA_CROSS_15M_OPT_TRAIN_RATIO", 0.70)
EMA_CROSS_15M_OPT_MAX_EVALUATED = _env_int("EMA_CROSS_15M_OPT_MAX_EVALUATED", 180)
EMA_CROSS_15M_OPT_TARGET_WIN_RATE = _env_float("EMA_CROSS_15M_OPT_TARGET_WIN_RATE", 60.0)
EMA_CROSS_15M_OPT_TARGET_EXPECTANCY_RR = _env_float("EMA_CROSS_15M_OPT_TARGET_EXPECTANCY_RR", 0.10)
EMA_CROSS_15M_TP_MULT = _env_float("EMA_CROSS_15M_TP_MULT", 5.0)
EMA_CROSS_15M_STOP_ATR_MULT = _env_float("EMA_CROSS_15M_STOP_ATR_MULT", 1.6)
EMA_CROSS_15M_MIN_STOP_PCT = _env_float("EMA_CROSS_15M_MIN_STOP_PCT", 0.8)
EMA_CROSS_15M_MAX_FORWARD_BARS = _env_int("EMA_CROSS_15M_MAX_FORWARD_BARS", 64)
EMA_CROSS_15M_MAX_BARS_SINCE_CROSS = _env_int("EMA_CROSS_15M_MAX_BARS_SINCE_CROSS", 4)
EMA_CROSS_15M_MIN_RVOL = _env_float("EMA_CROSS_15M_MIN_RVOL", 1.0)
EMA_CROSS_15M_MIN_EMA_GAP_PCT = _env_float("EMA_CROSS_15M_MIN_EMA_GAP_PCT", 0.03)
EMA_CROSS_15M_MIN_ATR_PCT = _env_float("EMA_CROSS_15M_MIN_ATR_PCT", 0.10)
EMA_CROSS_15M_MAX_ATR_PCT = _env_float("EMA_CROSS_15M_MAX_ATR_PCT", 6.00)
EMA_CROSS_15M_MIN_ADX = _env_float("EMA_CROSS_15M_MIN_ADX", 18.0)
EMA_CROSS_15M_REQUIRE_EMA200_ALIGNMENT = _env_bool("EMA_CROSS_15M_REQUIRE_EMA200_ALIGNMENT", True)
EMA_CROSS_15M_REQUIRE_PATTERN = _env_bool("EMA_CROSS_15M_REQUIRE_PATTERN", False)
EMA_CROSS_15M_PATTERN_LOOKBACK_BARS = _env_int("EMA_CROSS_15M_PATTERN_LOOKBACK_BARS", 3)
EMA_CROSS_15M_ENTRY_CONFIRMATION_MODE = _env_str("EMA_CROSS_15M_ENTRY_CONFIRMATION_MODE", "engulfing_pinbar")
EMA_CROSS_15M_USE_CANDLE_STOP = _env_bool("EMA_CROSS_15M_USE_CANDLE_STOP", True)
EMA_CROSS_15M_CANDLE_STOP_BUFFER_ATR = _env_float("EMA_CROSS_15M_CANDLE_STOP_BUFFER_ATR", 0.15)
EMA_CROSS_15M_REQUIRE_SLOW_SLOPE_CONFIRM = _env_bool("EMA_CROSS_15M_REQUIRE_SLOW_SLOPE_CONFIRM", True)
EMA_CROSS_15M_CACHE_TTL_SECONDS = _env_int("EMA_CROSS_15M_CACHE_TTL_SECONDS", 900)

ACTIONZONE_15M_SMOOTH = _env_int("ACTIONZONE_15M_SMOOTH", 1)
ACTIONZONE_15M_ALERT_BARS = _env_int("ACTIONZONE_15M_ALERT_BARS", 1)
ACTIONZONE_15M_USE_OPTIMIZATION = _env_bool("ACTIONZONE_15M_USE_OPTIMIZATION", True)
ACTIONZONE_15M_YF_PERIOD = _env_str("ACTIONZONE_15M_YF_PERIOD", "60d")
ACTIONZONE_15M_TREND_1H_PERIOD = _env_str("ACTIONZONE_15M_TREND_1H_PERIOD", "3mo")
ACTIONZONE_15M_STOP_ATR_MULT = _env_float("ACTIONZONE_15M_STOP_ATR_MULT", 1.8)
ACTIONZONE_15M_MIN_STOP_PCT = _env_float("ACTIONZONE_15M_MIN_STOP_PCT", 0.8)
ACTIONZONE_15M_MIN_VALID_TRADES = _env_int("ACTIONZONE_15M_MIN_VALID_TRADES", 4)
ACTIONZONE_15M_OPT_TRAIN_RATIO = _env_float("ACTIONZONE_15M_OPT_TRAIN_RATIO", 0.70)
ACTIONZONE_15M_OPT_MAX_EVALUATED = _env_int("ACTIONZONE_15M_OPT_MAX_EVALUATED", 96)
ACTIONZONE_15M_OPT_TARGET_WIN_RATE = _env_float("ACTIONZONE_15M_OPT_TARGET_WIN_RATE", 60.0)
ACTIONZONE_15M_OPT_TARGET_EXPECTANCY_RR = _env_float("ACTIONZONE_15M_OPT_TARGET_EXPECTANCY_RR", 0.08)
ACTIONZONE_15M_MIN_RVOL = _env_float("ACTIONZONE_15M_MIN_RVOL", 1.15)
ACTIONZONE_15M_MIN_EMA_GAP_PCT = _env_float("ACTIONZONE_15M_MIN_EMA_GAP_PCT", 0.05)
ACTIONZONE_15M_MIN_ATR_PCT = _env_float("ACTIONZONE_15M_MIN_ATR_PCT", 0.15)
ACTIONZONE_15M_MAX_ATR_PCT = _env_float("ACTIONZONE_15M_MAX_ATR_PCT", 6.00)
ACTIONZONE_15M_MIN_ADX = _env_float("ACTIONZONE_15M_MIN_ADX", 24.0)
ACTIONZONE_15M_REQUIRE_EMA200_ALIGNMENT = _env_bool("ACTIONZONE_15M_REQUIRE_EMA200_ALIGNMENT", True)
ACTIONZONE_15M_REQUIRE_PATTERN = _env_bool("ACTIONZONE_15M_REQUIRE_PATTERN", True)
ACTIONZONE_15M_PATTERN_LOOKBACK_BARS = _env_int("ACTIONZONE_15M_PATTERN_LOOKBACK_BARS", 3)
ACTIONZONE_15M_ENTRY_CONFIRMATION_MODE = _env_str("ACTIONZONE_15M_ENTRY_CONFIRMATION_MODE", "engulfing_pinbar")
ACTIONZONE_15M_USE_CANDLE_STOP = _env_bool("ACTIONZONE_15M_USE_CANDLE_STOP", True)
ACTIONZONE_15M_CANDLE_STOP_BUFFER_ATR = _env_float("ACTIONZONE_15M_CANDLE_STOP_BUFFER_ATR", 0.15)
ACTIONZONE_15M_MAX_BARS_SINCE_SIGNAL = _env_int("ACTIONZONE_15M_MAX_BARS_SINCE_SIGNAL", 1)
ACTIONZONE_15M_REQUIRE_STRONG_TREND = _env_bool("ACTIONZONE_15M_REQUIRE_STRONG_TREND", True)
ACTIONZONE_15M_TP_MULT = _env_float("ACTIONZONE_15M_TP_MULT", 2.0)
ACTIONZONE_15M_MIN_ALERT_CONFIDENCE = _env_float("ACTIONZONE_15M_MIN_ALERT_CONFIDENCE", 74.0)
ACTIONZONE_15M_PRECISION60_ENABLED = _env_bool("ACTIONZONE_15M_PRECISION60_ENABLED", True)
ACTIONZONE_15M_PRECISION60_MIN_ALERT_CONFIDENCE = _env_float("ACTIONZONE_15M_PRECISION60_MIN_ALERT_CONFIDENCE", 90.0)
ACTIONZONE_15M_PRECISION60_MIN_RVOL = _env_float("ACTIONZONE_15M_PRECISION60_MIN_RVOL", 1.30)
ACTIONZONE_15M_PRECISION60_MIN_EMA_GAP_PCT = _env_float("ACTIONZONE_15M_PRECISION60_MIN_EMA_GAP_PCT", 0.12)
ACTIONZONE_15M_PRECISION60_MIN_ADX = _env_float("ACTIONZONE_15M_PRECISION60_MIN_ADX", 30.0)
ACTIONZONE_15M_PRECISION60_MAX_BARS_SINCE_SIGNAL = _env_int("ACTIONZONE_15M_PRECISION60_MAX_BARS_SINCE_SIGNAL", 0)
ACTIONZONE_15M_PRECISION60_REQUIRE_EMA200_ALIGNMENT = _env_bool("ACTIONZONE_15M_PRECISION60_REQUIRE_EMA200_ALIGNMENT", True)
ACTIONZONE_15M_PRECISION60_REQUIRE_STRONG_TREND = _env_bool("ACTIONZONE_15M_PRECISION60_REQUIRE_STRONG_TREND", True)
ACTIONZONE_15M_PRECISION60_REQUIRE_TREND_ALIGNMENT = _env_bool("ACTIONZONE_15M_PRECISION60_REQUIRE_TREND_ALIGNMENT", True)
ACTIONZONE_15M_TIME_STOP_BARS = _env_int("ACTIONZONE_15M_TIME_STOP_BARS", 28)
ACTIONZONE_15M_TP1_R = _env_float("ACTIONZONE_15M_TP1_R", 0.9)
ACTIONZONE_15M_TP1_FRACTION = _env_float("ACTIONZONE_15M_TP1_FRACTION", 0.55)
ACTIONZONE_15M_MOVE_SL_TO_BE = _env_bool("ACTIONZONE_15M_MOVE_SL_TO_BE", True)
ACTIONZONE_15M_TRAILING_ATR_MULT = _env_float("ACTIONZONE_15M_TRAILING_ATR_MULT", 2.2)

PRICE_ACTION_15M_ENABLED = _env_bool("PRICE_ACTION_15M_ENABLED", True)
PRICE_ACTION_15M_MIN_ALERT_CONFIDENCE = _env_float("PRICE_ACTION_15M_MIN_ALERT_CONFIDENCE", 64.0)
PRICE_ACTION_15M_MIN_SCORE = _env_float("PRICE_ACTION_15M_MIN_SCORE", 66.0)
PRICE_ACTION_15M_SWING_LOOKBACK = _env_int("PRICE_ACTION_15M_SWING_LOOKBACK", 3)
PRICE_ACTION_15M_SR_LOOKBACK = _env_int("PRICE_ACTION_15M_SR_LOOKBACK", 24)
PRICE_ACTION_15M_REQUIRE_PATTERN = _env_bool("PRICE_ACTION_15M_REQUIRE_PATTERN", True)
PRICE_ACTION_15M_PATTERN_LOOKBACK_BARS = _env_int("PRICE_ACTION_15M_PATTERN_LOOKBACK_BARS", 3)
PRICE_ACTION_15M_ENTRY_CONFIRMATION_MODE = _env_str("PRICE_ACTION_15M_ENTRY_CONFIRMATION_MODE", "engulfing_pinbar")
PRICE_ACTION_15M_MIN_ADX = _env_float("PRICE_ACTION_15M_MIN_ADX", 18.0)
PRICE_ACTION_15M_REQUIRE_EMA200_ALIGNMENT = _env_bool("PRICE_ACTION_15M_REQUIRE_EMA200_ALIGNMENT", False)
PRICE_ACTION_15M_ROLE_REVERSAL_BUFFER_PCT = _env_float("PRICE_ACTION_15M_ROLE_REVERSAL_BUFFER_PCT", 0.35)
PRICE_ACTION_15M_ZONE_PROXIMITY_PCT = _env_float("PRICE_ACTION_15M_ZONE_PROXIMITY_PCT", 1.00)
PRICE_ACTION_15M_STOP_ATR_MULT = _env_float("PRICE_ACTION_15M_STOP_ATR_MULT", 1.5)
PRICE_ACTION_15M_TP_MULT = _env_float("PRICE_ACTION_15M_TP_MULT", 2.2)
PRICE_ACTION_15M_CANDLE_STOP_BUFFER_ATR = _env_float("PRICE_ACTION_15M_CANDLE_STOP_BUFFER_ATR", 0.15)
PRICE_ACTION_15M_MIN_PROXY_WIN_RATE = _env_float("PRICE_ACTION_15M_MIN_PROXY_WIN_RATE", 55.0)
PRICE_ACTION_15M_MIN_PROXY_TRADES = _env_int("PRICE_ACTION_15M_MIN_PROXY_TRADES", 4)
PRICE_ACTION_15M_MIN_PROXY_EXPECTANCY_RR = _env_float("PRICE_ACTION_15M_MIN_PROXY_EXPECTANCY_RR", 0.0)
PRICE_ACTION_15M_MIN_PROXY_SOURCE_COUNT = _env_int("PRICE_ACTION_15M_MIN_PROXY_SOURCE_COUNT", 1)

TREND_BREAKOUT_15M_ENABLED = _env_bool("TREND_BREAKOUT_15M_ENABLED", True)
TREND_BREAKOUT_15M_MIN_ALERT_CONFIDENCE = _env_float("TREND_BREAKOUT_15M_MIN_ALERT_CONFIDENCE", 65.0)
TREND_BREAKOUT_15M_MIN_SCORE = _env_float("TREND_BREAKOUT_15M_MIN_SCORE", 66.0)
TREND_BREAKOUT_15M_LOOKBACK_BARS = _env_int("TREND_BREAKOUT_15M_LOOKBACK_BARS", 20)
TREND_BREAKOUT_15M_ALERT_BARS = _env_int("TREND_BREAKOUT_15M_ALERT_BARS", 2)
TREND_BREAKOUT_15M_MIN_ADX = _env_float("TREND_BREAKOUT_15M_MIN_ADX", 20.0)
TREND_BREAKOUT_15M_MIN_RVOL = _env_float("TREND_BREAKOUT_15M_MIN_RVOL", 1.15)
TREND_BREAKOUT_15M_MIN_EMA_GAP_PCT = _env_float("TREND_BREAKOUT_15M_MIN_EMA_GAP_PCT", 0.05)
TREND_BREAKOUT_15M_BREAKOUT_BUFFER_PCT = _env_float("TREND_BREAKOUT_15M_BREAKOUT_BUFFER_PCT", 0.15)
TREND_BREAKOUT_15M_REQUIRE_EMA200_ALIGNMENT = _env_bool("TREND_BREAKOUT_15M_REQUIRE_EMA200_ALIGNMENT", True)
TREND_BREAKOUT_15M_REQUIRE_1H_ALIGNMENT = _env_bool("TREND_BREAKOUT_15M_REQUIRE_1H_ALIGNMENT", True)
TREND_BREAKOUT_15M_REQUIRE_PATTERN = _env_bool("TREND_BREAKOUT_15M_REQUIRE_PATTERN", False)
TREND_BREAKOUT_15M_PATTERN_LOOKBACK_BARS = _env_int("TREND_BREAKOUT_15M_PATTERN_LOOKBACK_BARS", 3)
TREND_BREAKOUT_15M_ENTRY_CONFIRMATION_MODE = _env_str("TREND_BREAKOUT_15M_ENTRY_CONFIRMATION_MODE", "engulfing_pinbar")
TREND_BREAKOUT_15M_MAX_EXTENSION_ATR = _env_float("TREND_BREAKOUT_15M_MAX_EXTENSION_ATR", 1.6)
TREND_BREAKOUT_15M_STOP_ATR_MULT = _env_float("TREND_BREAKOUT_15M_STOP_ATR_MULT", 1.4)
TREND_BREAKOUT_15M_TP_MULT = _env_float("TREND_BREAKOUT_15M_TP_MULT", 2.0)
TREND_BREAKOUT_15M_CANDLE_STOP_BUFFER_ATR = _env_float("TREND_BREAKOUT_15M_CANDLE_STOP_BUFFER_ATR", 0.15)
TREND_BREAKOUT_15M_MIN_PROXY_WIN_RATE = _env_float("TREND_BREAKOUT_15M_MIN_PROXY_WIN_RATE", 55.0)
TREND_BREAKOUT_15M_MIN_PROXY_TRADES = _env_int("TREND_BREAKOUT_15M_MIN_PROXY_TRADES", 4)
TREND_BREAKOUT_15M_MIN_PROXY_EXPECTANCY_RR = _env_float("TREND_BREAKOUT_15M_MIN_PROXY_EXPECTANCY_RR", 0.0)
TREND_BREAKOUT_15M_MIN_PROXY_SOURCE_COUNT = _env_int("TREND_BREAKOUT_15M_MIN_PROXY_SOURCE_COUNT", 1)

CDC_VIXFIX_15M_YF_PERIOD = _env_str("CDC_VIXFIX_15M_YF_PERIOD", "60d")
CDC_VIXFIX_15M_FAST_EMA = _env_int("CDC_VIXFIX_15M_FAST_EMA", 12)
CDC_VIXFIX_15M_SLOW_EMA = _env_int("CDC_VIXFIX_15M_SLOW_EMA", 26)
CDC_VIXFIX_15M_RSI_LENGTH = _env_int("CDC_VIXFIX_15M_RSI_LENGTH", 14)
CDC_VIXFIX_15M_STOCH_LENGTH = _env_int("CDC_VIXFIX_15M_STOCH_LENGTH", 14)
CDC_VIXFIX_15M_STOCH_SMOOTH_K = _env_int("CDC_VIXFIX_15M_STOCH_SMOOTH_K", 3)
CDC_VIXFIX_15M_STOCH_SMOOTH_D = _env_int("CDC_VIXFIX_15M_STOCH_SMOOTH_D", 3)
CDC_VIXFIX_15M_SMOOTH = _env_int("CDC_VIXFIX_15M_SMOOTH", 1)
CDC_VIXFIX_15M_STOCH_OVERSOLD = _env_float("CDC_VIXFIX_15M_STOCH_OVERSOLD", 30.0)
CDC_VIXFIX_15M_STOCH_OVERBOUGHT = _env_float("CDC_VIXFIX_15M_STOCH_OVERBOUGHT", 70.0)
CDC_VIXFIX_15M_VIX_LOOKBACK = _env_int("CDC_VIXFIX_15M_VIX_LOOKBACK", 22)
CDC_VIXFIX_15M_VIX_BB_LENGTH = _env_int("CDC_VIXFIX_15M_VIX_BB_LENGTH", 20)
CDC_VIXFIX_15M_VIX_BB_STD = _env_float("CDC_VIXFIX_15M_VIX_BB_STD", 2.0)
CDC_VIXFIX_15M_VIX_PERCENTILE_LOOKBACK = _env_int("CDC_VIXFIX_15M_VIX_PERCENTILE_LOOKBACK", 50)
CDC_VIXFIX_15M_VIX_PERCENTILE_FACTOR = _env_float("CDC_VIXFIX_15M_VIX_PERCENTILE_FACTOR", 0.85)
CDC_VIXFIX_15M_VIX_LOW_PERCENTILE_FACTOR = _env_float("CDC_VIXFIX_15M_VIX_LOW_PERCENTILE_FACTOR", 1.01)
CDC_VIXFIX_15M_VIX_SPIKE_LOOKBACK_BARS = _env_int("CDC_VIXFIX_15M_VIX_SPIKE_LOOKBACK_BARS", 2)
CDC_VIXFIX_15M_ALERT_BARS = _env_int("CDC_VIXFIX_15M_ALERT_BARS", 2)
CDC_VIXFIX_15M_MIN_ALERT_CONFIDENCE = _env_float("CDC_VIXFIX_15M_MIN_ALERT_CONFIDENCE", 61.0)
CDC_VIXFIX_15M_MIN_STOP_PCT = _env_float("CDC_VIXFIX_15M_MIN_STOP_PCT", 1.2)
CDC_VIXFIX_15M_SL_ATR_MULT = _env_float("CDC_VIXFIX_15M_SL_ATR_MULT", 2.5)
CDC_VIXFIX_15M_REQUIRE_PATTERN = _env_bool("CDC_VIXFIX_15M_REQUIRE_PATTERN", False)
CDC_VIXFIX_15M_PATTERN_LOOKBACK_BARS = _env_int("CDC_VIXFIX_15M_PATTERN_LOOKBACK_BARS", 3)
CDC_VIXFIX_15M_ENTRY_CONFIRMATION_MODE = _env_str("CDC_VIXFIX_15M_ENTRY_CONFIRMATION_MODE", "engulfing_pinbar")
CDC_VIXFIX_15M_USE_CANDLE_STOP = _env_bool("CDC_VIXFIX_15M_USE_CANDLE_STOP", True)

CDC_VIXFIX_15M_CANDLE_STOP_BUFFER_ATR = _env_float("CDC_VIXFIX_15M_CANDLE_STOP_BUFFER_ATR", 0.15)
CDC_VIXFIX_15M_RELAXED_ENTRY_ENABLE = _env_bool("CDC_VIXFIX_15M_RELAXED_ENTRY_ENABLE", True)
CDC_VIXFIX_15M_RELAXED_STOCH_MAX = _env_float("CDC_VIXFIX_15M_RELAXED_STOCH_MAX", 45.0)
CDC_VIXFIX_15M_FORECAST_MOMENTUM_LOOKBACK = _env_int("CDC_VIXFIX_15M_FORECAST_MOMENTUM_LOOKBACK", 3)
CDC_VIXFIX_15M_FORECAST_MIN_SCORE = _env_float("CDC_VIXFIX_15M_FORECAST_MIN_SCORE", 60.0)
CDC_VIXFIX_15M_REQUIRE_EMA200_ALIGNMENT = _env_bool("CDC_VIXFIX_15M_REQUIRE_EMA200_ALIGNMENT", False)
CDC_VIXFIX_15M_TAKE_PROFIT_PCT = _env_float("CDC_VIXFIX_15M_TAKE_PROFIT_PCT", 0.2)
CDC_VIXFIX_15M_MAX_HOLD_BARS = _env_int("CDC_VIXFIX_15M_MAX_HOLD_BARS", 24)

# Precision profiles tuned toward >=60% recent win rate per symbol on 15m replay.
# Note: these profiles optimize hit rate, not expectancy, so some symbols remain negative expectancy.
CDC_VIXFIX_15M_SYMBOL_PROFILES = {
    "BTC-USD": {
        "require_pattern": False,
        "require_ema200_alignment": True,
        "vix_spike_lookback_bars": 2,
        "stoch_oversold": 30.0,
        "forecast_min_score": 60.0,
        "daily_best_min_red_to_green_score": 88.0,
        "daily_best_require_reclaim": True,
        "take_profit_pct": 0.2,
        "max_hold_bars": 24,
    },
    "DOGE-USD": {
        "require_pattern": True,
        "require_ema200_alignment": False,
        "vix_spike_lookback_bars": 2,
        "stoch_oversold": 30.0,
        "forecast_min_score": 65.0,
        "daily_best_min_red_to_green_score": 82.0,
        "daily_best_require_reclaim": True,
        "take_profit_pct": 0.2,
        "max_hold_bars": 48,
    },
    "ETH-USD": {
        "require_pattern": False,
        "require_ema200_alignment": True,
        "vix_spike_lookback_bars": 2,
        "stoch_oversold": 30.0,
        "forecast_min_score": 60.0,
        "daily_best_min_red_to_green_score": 82.0,
        "daily_best_require_reclaim": True,
        "take_profit_pct": 0.2,
        "max_hold_bars": 24,
    },
    "ADA-USD": {
        "require_pattern": False,
        "require_ema200_alignment": True,
        "vix_spike_lookback_bars": 2,
        "stoch_oversold": 30.0,
        "forecast_min_score": 60.0,
        "daily_best_min_red_to_green_score": 82.0,
        "daily_best_require_reclaim": True,
        "take_profit_pct": 0.2,
        "max_hold_bars": 24,
    },
    "XRP-USD": {
        "require_pattern": True,
        "require_ema200_alignment": False,
        "vix_spike_lookback_bars": 2,
        "stoch_oversold": 30.0,
        "forecast_min_score": 65.0,
        "daily_best_min_red_to_green_score": 84.0,
        "daily_best_require_reclaim": True,
        "take_profit_pct": 0.2,
        "max_hold_bars": 48,
    },
    "BNB-USD": {
        "require_pattern": False,
        "require_ema200_alignment": True,
        "vix_spike_lookback_bars": 2,
        "stoch_oversold": 30.0,
        "forecast_min_score": 60.0,
        "daily_best_min_red_to_green_score": 88.0,
        "daily_best_require_reclaim": True,
        "take_profit_pct": 0.2,
        "max_hold_bars": 24,
    },
    "SOL-USD": {
        "require_pattern": False,
        "require_ema200_alignment": True,
        "vix_spike_lookback_bars": 2,
        "stoch_oversold": 30.0,
        "forecast_min_score": 60.0,
        "daily_best_min_red_to_green_score": 82.0,
        "daily_best_require_reclaim": True,
        "take_profit_pct": 0.2,
        "max_hold_bars": 24,
    },
    "TRX-USD": {
        "require_pattern": False,
        "require_ema200_alignment": True,
        "vix_spike_lookback_bars": 2,
        "stoch_oversold": 25.0,
        "forecast_min_score": 65.0,
        "daily_best_min_red_to_green_score": 92.0,
        "daily_best_require_reclaim": True,
        "take_profit_pct": 0.2,
        "max_hold_bars": 24,
    },
    "NEAR-USD": {
        "require_pattern": False,
        "require_ema200_alignment": True,
        "vix_spike_lookback_bars": 2,
        "stoch_oversold": 30.0,
        "forecast_min_score": 60.0,
        "daily_best_min_red_to_green_score": 80.0,
        "daily_best_require_reclaim": True,
        "take_profit_pct": 0.2,
        "max_hold_bars": 24,
    },
    "LINK-USD": {
        "require_pattern": False,
        "require_ema200_alignment": False,
        "vix_spike_lookback_bars": 2,
        "stoch_oversold": 25.0,
        "forecast_min_score": 60.0,
        "daily_best_min_red_to_green_score": 86.0,
        "daily_best_require_reclaim": True,
        "take_profit_pct": 0.2,
        "max_hold_bars": 24,
    },
    "PAXG-USD": {
        "require_pattern": False,
        "require_ema200_alignment": True,
        "vix_spike_lookback_bars": 2,
        "stoch_oversold": 25.0,
        "forecast_min_score": 60.0,
        "daily_best_min_red_to_green_score": 90.0,
        "daily_best_require_reclaim": True,
        "take_profit_pct": 0.2,
        "max_hold_bars": 24,
    },
}
CDC_VIXFIX_15M_DISABLED_SYMBOLS = set()

ORDERBLOCK_15M_PIVOT_LENGTH = _env_int("ORDERBLOCK_15M_PIVOT_LENGTH", 5)
ORDERBLOCK_15M_MAX_BLOCKS = _env_int("ORDERBLOCK_15M_MAX_BLOCKS", 6)
ORDERBLOCK_15M_MITIGATION = _env_str("ORDERBLOCK_15M_MITIGATION", "wick")

SHORT_TERM_15M_YF_PERIOD = _env_str("SHORT_TERM_15M_YF_PERIOD", "30d")
SHORT_TERM_15M_TREND_1H_PERIOD = _env_str("SHORT_TERM_15M_TREND_1H_PERIOD", "3mo")
SHORT_TERM_15M_MIN_TRADES = _env_int("SHORT_TERM_15M_MIN_TRADES", 10)
SHORT_TERM_15M_MAX_FORWARD_BARS = _env_int("SHORT_TERM_15M_MAX_FORWARD_BARS", 32)
SHORT_TERM_15M_SIGNAL_RVOL_MIN = _env_float("SHORT_TERM_15M_SIGNAL_RVOL_MIN", 1.2)
SHORT_TERM_15M_SIGNAL_ADX_MIN = _env_float("SHORT_TERM_15M_SIGNAL_ADX_MIN", 18.0)
SHORT_TERM_15M_SWING_LOOKBACK = _env_int("SHORT_TERM_15M_SWING_LOOKBACK", 12)
SHORT_TERM_15M_RISK_ATR_MIN_MULT = _env_float("SHORT_TERM_15M_RISK_ATR_MIN_MULT", 1.0)
SHORT_TERM_15M_RISK_ATR_MAX_MULT = _env_float("SHORT_TERM_15M_RISK_ATR_MAX_MULT", 3.0)
SHORT_TERM_15M_STOP_ATR_BUFFER = _env_float("SHORT_TERM_15M_STOP_ATR_BUFFER", 0.2)
SHORT_TERM_15M_PULLBACK_TOL_PCT = _env_float("SHORT_TERM_15M_PULLBACK_TOL_PCT", 0.3)
SHORT_TERM_15M_TP1_R = _env_float("SHORT_TERM_15M_TP1_R", 1.5)
SHORT_TERM_15M_TP2_R = _env_float("SHORT_TERM_15M_TP2_R", 3.0)
SHORT_TERM_15M_BETA_ALPHA = _env_float("SHORT_TERM_15M_BETA_ALPHA", 1.0)
SHORT_TERM_15M_BETA_BETA = _env_float("SHORT_TERM_15M_BETA_BETA", 1.0)

# Realistic backtest friction (round-trip = entry+exit).
BACKTEST_FEE_BPS = _env_float("BACKTEST_FEE_BPS", 10.0)
BACKTEST_SLIPPAGE_BPS = _env_float("BACKTEST_SLIPPAGE_BPS", 5.0)
