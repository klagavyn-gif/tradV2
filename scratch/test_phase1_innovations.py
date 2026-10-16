"""
Unit and Integration Test Suite for Phase 1 Quant Innovations:
1. BTC Shockwave Barrier (Cross-asset volatility & liquidation protection)
2. Kinetic Half-Life Stop Engine (Temporal momentum decay stop engine)
"""

import os
import sys
import time

# Safeguard terminal output encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime, timezone, timedelta
import pandas as pd
from unittest.mock import MagicMock

import config
from infrastructure.binance.shockwave_barrier import BtcShockwaveBarrier
from infrastructure.binance.order_manager import BinanceFuturesOrderManager, normalize_futures_symbol
from infrastructure.binance.pipeline_hook import execute_binance_auto_trade_pipeline
from alerts.reporting import _resolve_directional_alert_outcome


def test_btc_shockwave_barrier():
    print("\n--- [Test 1] Testing BTC Shockwave Barrier ---")
    barrier = BtcShockwaveBarrier(default_atr_mult=1.8, default_abs_pct=1.5)

    # Synthetic 15m klines for BTCUSDT: 20 bars
    # Each bar: [open_time, open, high, low, close, volume, ...]
    # Standard baseline: Price ~60,000, candle range ~200 (ATR ~200)
    base_time = int(time.time() * 1000) - (20 * 15 * 60 * 1000)
    normal_klines = []
    for i in range(20):
        t = base_time + (i * 15 * 60 * 1000)
        o = 60000.0 + (i * 10.0)
        h = o + 100.0
        l = o - 100.0
        c = o + 20.0
        normal_klines.append([t, o, h, l, c, 100.0])

    # 1. Normal evaluation
    res_normal = barrier.evaluate(klines=normal_klines)
    assert not res_normal["shock_active"], "Expected shock_active to be False on calm market"
    assert "Normal" in res_normal["badge"]

    veto_alt, reason_alt = barrier.should_veto_candidate({"symbol": "SUI-USD"}, res_normal)
    assert not veto_alt, "Altcoin should NOT be vetoed on calm market"
    print("  [Pass] Normal market conditions: Shockwave False, Altcoins permitted.")

    # 2. Extreme volatility bar on latest candle (index -1)
    # Range = 800 (ATR ~200 -> 4.0x ATR)
    shock_klines = [list(k) for k in normal_klines]
    shock_klines[-1][1] = 60000.0   # open
    shock_klines[-1][2] = 60800.0   # high (+800)
    shock_klines[-1][3] = 59800.0   # low (-200, total range 1000)
    shock_klines[-1][4] = 60750.0   # close (+750)

    res_shock = barrier.evaluate(klines=shock_klines)
    assert res_shock["shock_active"], "Expected shock_active to be True on large range bar"
    assert res_shock["range_atr"] >= 1.8

    # Altcoin candidate must be vetoed
    veto_sui, reason_sui = barrier.should_veto_candidate({"symbol": "SUI-USD"}, res_shock)
    assert veto_sui, "SUI-USD should be vetoed during BTC shock"
    assert "btc_shockwave_spillover" in reason_sui

    veto_ondo, reason_ondo = barrier.should_veto_candidate({"symbol": "ONDOUSDT"}, res_shock)
    assert veto_ondo, "ONDOUSDT should be vetoed during BTC shock"

    # BTC itself should NOT be vetoed (it is trading its own trend)
    veto_btc, reason_btc = barrier.should_veto_candidate({"symbol": "BTC-USD"}, res_shock)
    assert not veto_btc, "BTC-USD should NOT be vetoed"

    veto_btc2, _ = barrier.should_veto_candidate({"symbol": "BTCUSDT"}, res_shock)
    assert not veto_btc2, "BTCUSDT should NOT be vetoed"
    print("  [Pass] Extreme BTC expansion bar: Shockwave True, Altcoins vetoed, BTC allowed.")

    # 3. Reverberating shockwave on previous closed bar (index -2)
    reverb_klines = [list(k) for k in normal_klines]
    reverb_klines[-2][1] = 60000.0
    reverb_klines[-2][2] = 60900.0
    reverb_klines[-2][3] = 59900.0
    reverb_klines[-2][4] = 60100.0
    # latest bar calm
    reverb_klines[-1][1] = 60100.0
    reverb_klines[-1][2] = 60150.0
    reverb_klines[-1][3] = 60080.0
    reverb_klines[-1][4] = 60120.0

    res_reverb = barrier.evaluate(klines=reverb_klines)
    assert res_reverb["shock_active"], "Expected shock_active to be True due to reverberating bar"
    print("  [Pass] Reverberating previous bar shock detected successfully.")


def test_kinetic_stop_order_manager():
    print("\n--- [Test 2] Testing Kinetic Half-Life Stop in Order Manager ---")
    mock_client = MagicMock()

    mgr = BinanceFuturesOrderManager(
        client=mock_client,
        breakeven_r=1.2,
        kinetic_enabled=True,
        kinetic_min_seconds=7200.0,
        kinetic_min_r=0.60,
    )

    # 1. Position younger than kinetic_min_seconds (e.g. 1,000s) -> should NOT trigger kinetic stop
    pos_young = [{
        "symbol": "SUIUSDT",
        "side": "LONG",
        "entryPrice": 2.00,
        "markPrice": 2.03,  # gain = +0.03, initial risk = 0.10 -> 0.30R
        "amount": 100.0,
    }]
    mock_client.get_positions.return_value = pos_young
    mock_client.get_open_algo_orders.return_value = [
        {"orderType": "STOP_MARKET", "triggerPrice": 1.90, "algoId": 1001, "clientAlgoId": "trad_sl_1"}
    ]
    mock_client.get_symbol_filters.return_value = {"tickSize": 0.0001, "stepSize": 0.1}

    # Executed 1000s ago
    rec_young = {
        "alert_1": {
            "symbol": "SUIUSDT",
            "executed_at": (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat(),
        }
    }
    updates_young = mgr.sync_breakeven_stops(executed_records=rec_young)
    assert len(updates_young) == 0, "Young position should not trigger Breakeven or Kinetic Stop"
    print("  [Pass] Young position (< 2h) remains active with original Stop Loss.")

    # 2. Stalled position older than 7200s (e.g. 8,000s / 2.2h), gain_r = 0.30R (< 0.60R), mark >= entry
    # -> MUST trigger Kinetic Half-Life Stop, moving SL to entry (2.00)!
    mock_client.create_algo_order.return_value = {"success": True, "data": {"algoId": 2001}}
    mock_client.cancel_algo_order.return_value = {"success": True}

    rec_stalled = {
        "alert_1": {
            "symbol": "SUIUSDT",
            "executed_at": (datetime.now(timezone.utc) - timedelta(seconds=8000)).isoformat(),
        }
    }
    updates_stalled = mgr.sync_breakeven_stops(executed_records=rec_stalled)
    assert len(updates_stalled) == 1, "Stalled position should trigger Kinetic Stop"
    up = updates_stalled[0]
    assert up["symbol"] == "SUIUSDT"
    assert up["new_sl"] == 2.00
    assert up["reason"] == "kinetic"
    assert up["elapsed_seconds"] >= 7200.0
    print(f"  [Pass] Stalled position ({up['elapsed_seconds']:.0f}s, +{up['gain_r']}R) adjusted to Kinetic Breakeven {up['new_sl']}.")

    # 3. Position older than 7200s with strong gain >= 1.2R (e.g. mark = 2.15, gain = 1.5R)
    # -> Triggers traditional profit target breakeven (reason='profit_target')
    pos_winner = [{
        "symbol": "SUIUSDT",
        "side": "LONG",
        "entryPrice": 2.00,
        "markPrice": 2.15,  # gain = 1.5R
        "amount": 100.0,
    }]
    mock_client.get_positions.return_value = pos_winner
    updates_winner = mgr.sync_breakeven_stops(executed_records=rec_stalled)
    assert len(updates_winner) == 1
    assert updates_winner[0]["reason"] == "profit_target"
    print("  [Pass] High-gain position (1.5R) triggers Profit Target Breakeven.")


def test_kinetic_stop_reporting_simulation():
    print("\n--- [Test 3] Testing Kinetic Stop in Reporting Outcome Resolution ---")
    # Simulate a trade with 15 bars:
    # Entry at 100.0, Stop Loss at 95.0 (Risk = 5.0)
    # Take Profit at 115.0 (3.0R)
    # Bars 1..7 linger around 101.5 (+0.30R)
    # Bar 8: still at 101.5 (< 0.60R) -> Kinetic stop activates, setting active_sl = 100.0!
    # Bar 10: market drops to 99.0 -> hits active_sl (100.0) -> Exits flat with kinetic_stop_hit!
    # Without kinetic stop, it would have dropped to 94.0 (Stop loss hit, -1.0R loss)!
    
    start_dt = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
    dates = [start_dt + timedelta(minutes=15 * i) for i in range(15)]
    
    # Construct DataFrame
    bars_data = []
    # Bar 0 (Alert Bar)
    bars_data.append({"Open": 100.0, "High": 100.5, "Low": 99.8, "Close": 100.0})
    # Bars 1-7: low momentum, high = 102.0 (+0.4R), low = 100.2
    for _ in range(7):
        bars_data.append({"Open": 101.0, "High": 102.0, "Low": 100.2, "Close": 101.5})
    # Bar 8: open 101.5, high 101.8, low 100.5, close 101.0
    bars_data.append({"Open": 101.5, "High": 101.8, "Low": 100.5, "Close": 101.0})
    # Bar 9: sharp drop: open 101.0, high 101.0, low 98.0, close 98.5
    bars_data.append({"Open": 101.0, "High": 101.0, "Low": 98.0, "Close": 98.5})
    # Bars 10-14: crash continues
    for _ in range(5):
        bars_data.append({"Open": 98.0, "High": 98.5, "Low": 93.0, "Close": 94.0})

    df = pd.DataFrame(bars_data, index=pd.DatetimeIndex(dates))

    entry_cand = {
        "alert_id": "sim_kinetic_test",
        "timestamp": start_dt.isoformat(),
        "strategy": "CDCVIX15",
        "symbol": "BTC-USD",
        "signal": "BUY",
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "take_profit": 115.0,
        "dispatch_status_label": "เข้าได้",
    }

    # Run with kinetic stop enabled
    outcome_with_k = _resolve_directional_alert_outcome(
        entry_cand,
        price_df=df,
        now_dt=dates[-1],
        max_hold_bars=30,
        be_r=1.2,
        kinetic_stop_enabled=True,
        kinetic_min_bars=8,
        kinetic_min_r=0.60,
    )

    assert outcome_with_k["kinetic_stop_activated"], "Kinetic stop should be activated"
    assert outcome_with_k["exit_reason"] == "kinetic_stop_hit", f"Expected kinetic_stop_hit, got {outcome_with_k['exit_reason']}"
    assert outcome_with_k["outcome_result"] == "flat"
    print("  [Pass] With Kinetic Stop: Stalled trade converted from loss to Flat/Breakeven (kinetic_stop_hit).")

    # Run without kinetic stop disabled
    outcome_without_k = _resolve_directional_alert_outcome(
        entry_cand,
        price_df=df,
        now_dt=dates[-1],
        max_hold_bars=30,
        be_r=1.2,
        kinetic_stop_enabled=False,
    )
    assert not outcome_without_k["kinetic_stop_activated"]
    assert outcome_without_k["exit_reason"] == "stop_loss_hit"
    assert outcome_without_k["outcome_result"] == "loss"
    print("  [Pass] Without Kinetic Stop: Same stalled trade crashes into full Stop Loss (-1.0R loss).")


def test_pipeline_hook_integration():
    print("\n--- [Test 4] Testing Pipeline Hook Integration with BTC Shockwave Barrier ---")
    mock_client = MagicMock()
    mock_client.get_positions.return_value = []
    mock_client.get_open_algo_orders.return_value = []
    mock_client.get_usdt_balance.return_value = {"total": 1000.0, "available": 1000.0, "crossUnPnl": 0.0}
    mock_client.get_symbol_filters.return_value = {
        "pricePrecision": 2,
        "quantityPrecision": 2,
        "tickSize": 0.01,
        "stepSize": 0.1,
        "minQty": 0.1,
        "minNotional": 5.0,
    }

    # Simulate violent BTCUSDT klines
    shock_klines = []
    base_time = int(time.time() * 1000) - (25 * 15 * 60 * 1000)
    for i in range(25):
        t = base_time + (i * 15 * 60 * 1000)
        shock_klines.append([t, 60000.0, 60100.0, 59900.0, 60050.0, 100.0])
    # Latest bar = violent expansion (range 1200, 6.0x ATR)
    shock_klines[-1] = [base_time + (24 * 15 * 60 * 1000), 60000.0, 61000.0, 59800.0, 60950.0, 500.0]
    mock_client.get_klines.return_value = shock_klines

    # Create mock candidates: SUI-USD (Altcoin) and BTC-USD
    candidates = [
        {
            "alert_id": "cand_sui",
            "symbol": "SUI-USD",
            "signal": "BUY",
            "strategy": "CDCVIX15",
            "entry_price": 2.00,
            "stop_loss": 1.90,
            "take_profit": 2.30,
            "dispatch_status_label": "เข้าได้",
        },
        {
            "alert_id": "cand_btc",
            "symbol": "BTC-USD",
            "signal": "BUY",
            "strategy": "CDCVIX15",
            "entry_price": 60500.0,
            "stop_loss": 59500.0,
            "take_profit": 63000.0,
            "dispatch_status_label": "เข้าได้",
        }
    ]

    mock_client.create_order.return_value = {"success": True, "data": {"avgPrice": 60500.0}}
    mock_client.create_algo_order.return_value = {"success": True}

    # Patch client constructor in client module
    import infrastructure.binance.client as bclient
    orig_client_cls = bclient.BinanceFuturesClient
    bclient.BinanceFuturesClient = lambda **kwargs: mock_client

    test_config = MagicMock()
    test_config.BINANCE_FUTURES_AUTO_TRADE_ENABLED = True
    test_config.BINANCE_WATCHDOG_ENABLED = False
    test_config.BINANCE_FUTURES_API_KEY = "test_key"
    test_config.BINANCE_FUTURES_API_SECRET = "test_sec"
    test_config.BINANCE_FUTURES_TESTNET = True
    test_config.BTC_SHOCKWAVE_BARRIER_ENABLED = True
    test_config.BTC_SHOCKWAVE_ATR_MULT = 1.8
    test_config.BTC_SHOCKWAVE_ABS_PCT = 1.5
    test_config.BINANCE_DERIVATIVES_FILTER_ENABLED = False
    test_config.BINANCE_FUTURES_ALLOWED_STRATEGIES = "CDCVIX15"
    test_config.BINANCE_FUTURES_BLOCKED_SYMBOLS = ""
    test_config.BINANCE_FUTURES_KINETIC_STOP_ENABLED = True
    test_config.BINANCE_FUTURES_KINETIC_MIN_SECONDS = 7200.0
    test_config.BINANCE_FUTURES_KINETIC_MIN_R = 0.60
    test_config.BINANCE_FUTURES_TRADE_NOTIONAL_USDT = 400.0
    test_config.BINANCE_FUTURES_MAX_POSITIONS = 2
    test_config.BINANCE_FUTURES_LEVERAGE = 20
    test_config.BINANCE_FUTURES_MARGIN_TYPE = "ISOLATED"
    test_config.BINANCE_FUTURES_DYNAMIC_SIZING_ENABLED = False
    test_config.BINANCE_FUTURES_MAX_ENTRY_DRIFT_PCT = 0.80

    test_orders_path = os.path.join(PROJECT_ROOT, "scratch", "test_orders.json")
    if os.path.exists(test_orders_path):
        os.remove(test_orders_path)

    helpers = {
        "binance_executed_orders_path": lambda: test_orders_path,
        "alert_outcomes_file_path": lambda: "scratch/test_outcomes.json",
    }

    tg_messages = []
    def mock_send_tg(msg, **kwargs):
        tg_messages.append(msg)

    try:
        pipeline_res = execute_binance_auto_trade_pipeline(
            sent_candidates=candidates,
            config=test_config,
            helpers=helpers,
            get_now=lambda: datetime.now(timezone.utc),
            send_telegram_alert=mock_send_tg,
        )

        entries = pipeline_res["executed_entries"]
        assert len(entries) == 2, f"Expected 2 entries evaluated, got {len(entries)}"

        # Entry 1 (SUI) must be vetoed by shockwave
        sui_res = next(e for e in entries if e.get("symbol") == "SUI-USD")
        assert not sui_res["success"], "SUI should fail execution due to veto"
        assert "btc_shockwave_veto" in sui_res["reason"], f"Expected btc_shockwave_veto, got {sui_res['reason']}"
        print("  [Pass] Pipeline Hook vetoed SUI-USD during BTC Shockwave.")

        # Entry 2 (BTC) must succeed
        btc_res = next(e for e in entries if e.get("symbol") == "BTCUSDT")
        assert btc_res["success"], f"BTCUSDT should be executed, got {btc_res}"
        print("  [Pass] Pipeline Hook successfully executed BTCUSDT candidate.")

    finally:
        bclient.BinanceFuturesClient = orig_client_cls
        if os.path.exists(test_orders_path):
            try:
                os.remove(test_orders_path)
            except Exception:
                pass


def test_extended_holding_window():
    print("\n--- [Test 5] Testing Extended Holding Window (6-8h, 24-32 bars) ---")
    mock_client = MagicMock()

    # 1. Verify default values in Order Manager
    mgr_default = BinanceFuturesOrderManager(client=mock_client)
    assert mgr_default.kinetic_min_seconds == 21600.0, f"Expected 21600.0s (6h), got {mgr_default.kinetic_min_seconds}"

    # 2. Position held for 3.5 hours (12,600s) -> should NOT trigger kinetic stop
    pos_mid = [{
        "symbol": "ETHUSDT",
        "side": "LONG",
        "entryPrice": 2472.60,
        "markPrice": 2480.00,  # small gain (< 0.60R)
        "amount": 0.5,
    }]
    mock_client.get_positions.return_value = pos_mid
    mock_client.get_open_algo_orders.return_value = [
        {"orderType": "STOP_MARKET", "triggerPrice": 2435.0, "algoId": 1001, "clientAlgoId": "trad_sl_1"}
    ]
    mock_client.get_symbol_filters.return_value = {"tickSize": 0.01, "stepSize": 0.001}

    rec_3_5h = {
        "alert_eth": {
            "symbol": "ETHUSDT",
            "executed_at": (datetime.now(timezone.utc) - timedelta(seconds=12600)).isoformat(),
        }
    }
    updates_3_5h = mgr_default.sync_breakeven_stops(executed_records=rec_3_5h)
    assert len(updates_3_5h) == 0, "Position held for 3.5h should NOT be cut short by Kinetic stop under 6h default"
    print("  [Pass] Position held for 3.5h (12,600s) allowed to develop without premature stop.")

    # 3. Position held for 6.5 hours (23,400s) with low gain (< 0.60R) -> triggers Kinetic Stop
    mock_client.create_algo_order.return_value = {"success": True, "data": {"algoId": 2002}}
    mock_client.cancel_algo_order.return_value = {"success": True}

    rec_6_5h = {
        "alert_eth": {
            "symbol": "ETHUSDT",
            "executed_at": (datetime.now(timezone.utc) - timedelta(seconds=23400)).isoformat(),
        }
    }
    updates_6_5h = mgr_default.sync_breakeven_stops(executed_records=rec_6_5h)
    assert len(updates_6_5h) == 1, "Position held > 6h with stalled momentum should trigger Kinetic Stop"
    assert updates_6_5h[0]["reason"] == "kinetic"
    assert updates_6_5h[0]["new_sl"] == 2472.60
    print(f"  [Pass] Position held for 6.5h successfully triggers Kinetic Stop at Entry ({updates_6_5h[0]['new_sl']}).")

    # 4. Verify 8h configurable override (28,800s)
    mgr_8h = BinanceFuturesOrderManager(client=mock_client, kinetic_min_seconds=28800.0)
    updates_override = mgr_8h.sync_breakeven_stops(executed_records=rec_6_5h)
    assert len(updates_override) == 0, "6.5h position should NOT trigger when configured for 8 hours (28,800s)"
    print("  [Pass] 8-hour override (28,800s) respected correctly.")


if __name__ == "__main__":
    test_btc_shockwave_barrier()
    test_kinetic_stop_order_manager()
    test_kinetic_stop_reporting_simulation()
    test_pipeline_hook_integration()
    test_extended_holding_window()
    print("\n=======================================================")
    print("ALL PHASE 1 QUANT INNOVATION TESTS PASSED PERFECTLY!")
    print("=======================================================")

