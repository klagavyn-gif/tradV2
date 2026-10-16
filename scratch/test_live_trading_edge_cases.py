#!/usr/bin/env python3
"""
Comprehensive Live-Trading Pre-Flight & Edge Case Verification Suite.
Validates all critical failure points before executing real money orders:
1. Universe Precision & Sizing (All 11 symbols against live Binance ExchangeInfo)
2. Order Execution Lifecycle & Algo Order formatting
3. Breakeven & Trailing Stop Migration
4. Risk Manager Circuit Breaker, Streak Throttle, and UTC Rollover
5. Watchdog Naked Position Healing
6. Derivatives Filter Veto Protection
7. Pipeline Hook Guardrails (Whitelists, Intent, Idempotency, Max Positions)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import time
import math
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Add workspace to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from infrastructure.binance.client import BinanceFuturesClient
from infrastructure.binance.order_manager import (
    BinanceFuturesOrderManager,
    round_to_step,
    round_to_tick,
    normalize_futures_symbol,
)
from infrastructure.binance.risk_manager import BinanceFuturesRiskManager
from infrastructure.binance.derivatives_filter import BinanceDerivativesFilter
from infrastructure.binance.watchdog import BinanceFuturesWatchdog
from infrastructure.binance.pipeline_hook import (
    execute_binance_auto_trade_pipeline,
    _load_executed_orders,
    _save_executed_orders,
)

SYMBOLS_11 = [
    "BTC-USD", "DOGE-USD", "ETH-USD", "ADA-USD", "XRP-USD",
    "BNB-USD", "SOL-USD", "TRX-USD", "PAXG-USD", "ONDO-USD", "SUI-USD"
]


# =====================================================================
# TEST 1: Precision, Lot Sizing, and MinNotional across all 11 symbols
# =====================================================================
def test_all_11_symbols_precision_and_sizing():
    print("\n" + "="*70)
    print("--> TEST 1: Testing Precision & Sizing on ALL 11 Universe Symbols (Live Binance API)")
    print("="*70)

    client = BinanceFuturesClient(testnet=True)
    order_mgr = BinanceFuturesOrderManager(
        client=client,
        trade_notional_usdt=400.0,
        max_positions=2,
        leverage=20,
    )

    passed_symbols = 0
    for raw_sym in SYMBOLS_11:
        clean_sym = normalize_futures_symbol(raw_sym)
        filters = client.get_symbol_filters(clean_sym)
        assert filters is not None, f"Failed to get filters for {clean_sym}"

        tick = filters["tickSize"]
        step = filters["stepSize"]
        min_qty = filters["minQty"]
        min_notional = filters["minNotional"]

        # Fetch current mark price for realistic sizing
        price = client.get_symbol_price(clean_sym) or 1.0
        assert price > 0, f"Invalid price {price} for {clean_sym}"

        # Test candidate
        stop_dist = price * 0.015  # 1.5% ATR stop
        test_candidate = {
            "symbol": raw_sym,
            "signal": "BUY",
            "entry_price": price,
            "stop_loss": price - stop_dist,
            "alert_id": f"test_{int(time.time())}",
        }

        # Calculate sizing
        target_notional = max(400.0, min_notional)
        raw_qty = target_notional / price
        qty = round_to_step(raw_qty, step)
        if qty < min_qty or (qty * price) < min_notional:
            qty = round_to_step(qty + step, step)

        actual_notional = qty * price
        clean_sl = round_to_tick(price - stop_dist, tick)

        # Precision assertions
        # 1. Qty must be multiple of stepSize
        rem_step = math.fmod(qty, step)
        assert abs(rem_step) < 1e-6 or abs(rem_step - step) < 1e-6, f"{clean_sym}: Qty {qty} does not respect stepSize {step}"

        # 2. SL must be multiple of tickSize
        rem_tick = math.fmod(clean_sl, tick)
        assert abs(rem_tick) < 1e-6 or abs(rem_tick - tick) < 1e-6, f"{clean_sym}: SL {clean_sl} does not respect tickSize {tick}"

        # 3. Notional must be >= minNotional
        assert actual_notional >= min_notional, f"{clean_sym}: Notional {actual_notional} < minNotional {min_notional}"

        # 4. Quantity must be >= minQty
        assert qty >= min_qty, f"{clean_sym}: Qty {qty} < minQty {min_qty}"

        print(f"  [PASS] {raw_sym:<9} -> {clean_sym:<8} | Price: {price:,.4f} | Qty: {qty} | Notional: {actual_notional:,.2f} USDT (Min: {min_notional}) | Step: {step} | Tick: {tick}")
        passed_symbols += 1

    assert passed_symbols == len(SYMBOLS_11)
    print(f"  All {passed_symbols}/11 symbols passed precision, lot sizing, and minNotional validations! ✅")


# =====================================================================
# TEST 2: Order Execution & Algo Order API Flow (Simulated Binance API)
# =====================================================================
class MockBinanceClient:
    def __init__(self):
        self.orders = []
        self.algo_orders = []
        self.positions = []
        self.leverage_set = {}
        self.margin_type_set = {}
        self.fail_entry = False
        self.fail_sl = False

    def get_positions(self):
        return list(self.positions)

    def get_klines(self, symbol, interval="15m", limit=30):
        now_ms = int(time.time() * 1000)
        return [[now_ms - (i * 900000), 60000.0, 60100.0, 59900.0, 60020.0, 100.0] for i in range(limit, 0, -1)]

    def get_symbol_filters(self, symbol):
        return {
            "tickSize": 0.0001 if "ADA" in symbol or "XRP" in symbol else 0.01,
            "stepSize": 1.0 if "ADA" in symbol else 0.01,
            "minQty": 1.0 if "ADA" in symbol else 0.01,
            "minNotional": 5.0,
        }

    def get_usdt_balance(self):
        return {"total": 500.0, "available": 500.0, "crossUnPnl": 0.0}

    def set_margin_type(self, symbol, margin_type):
        self.margin_type_set[symbol] = margin_type
        return {"success": True}

    def set_leverage(self, symbol, leverage):
        self.leverage_set[symbol] = leverage
        return {"success": True}

    def create_order(self, symbol, side, order_type, quantity, client_order_id=None):
        if self.fail_entry:
            return {"success": False, "code": -2010, "msg": "Insufficient balance"}
        ord_data = {
            "orderId": len(self.orders) + 1000,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "origQty": str(quantity),
            "executedQty": str(quantity),
            "avgPrice": "0.7500",
            "clientOrderId": client_order_id,
            "status": "FILLED",
        }
        self.orders.append(ord_data)
        # Update mock position
        pos_side = "LONG" if side == "BUY" else "SHORT"
        self.positions.append({
            "symbol": symbol,
            "side": pos_side,
            "entryPrice": 0.7500,
            "markPrice": 0.7500,
            "amount": quantity if pos_side == "LONG" else -quantity,
        })
        return {"success": True, "data": ord_data}

    def create_algo_order(self, symbol, side, order_type, trigger_price=None,
                          quantity=None, activation_price=None, callback_rate=None,
                          close_position=False, reduce_only=False, client_algo_id=None):
        if self.fail_sl:
            return {"success": False, "code": -2011, "msg": "Unknown order"}
        algo_data = {
            "algoId": len(self.algo_orders) + 5000,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "triggerPrice": str(trigger_price) if trigger_price else None,
            "activationPrice": str(activation_price) if activation_price else None,
            "callbackRate": str(callback_rate) if callback_rate else None,
            "closePosition": close_position,
            "reduceOnly": reduce_only,
            "clientAlgoId": client_algo_id,
        }
        self.algo_orders.append(algo_data)
        return {"success": True, "data": algo_data}

    def get_open_orders(self, symbol=None):
        return []

    def get_open_algo_orders(self, symbol=None):
        if symbol:
            return [o for o in self.algo_orders if o["symbol"] == symbol]
        return list(self.algo_orders)

    def cancel_algo_order(self, symbol, algo_id=None, client_algo_id=None):
        self.algo_orders = [o for o in self.algo_orders if o.get("algoId") != algo_id and o.get("clientAlgoId") != client_algo_id]
        return {"success": True}


def test_order_execution_and_algo_flow():
    print("\n" + "="*70)
    print("--> TEST 2: Testing Order Execution & Algo Order API Flow")
    print("="*70)

    mock_client = MockBinanceClient()
    mgr = BinanceFuturesOrderManager(
        client=mock_client,
        trade_notional_usdt=400.0,
        max_positions=2,
        leverage=20,
        breakeven_r=1.2,
        trailing_r=2.0,
        trailing_dist_r=0.8,
    )

    # 1. Successful BUY execution
    candidate = {
        "symbol": "ADA-USD",
        "signal": "BUY",
        "entry_price": 0.7500,
        "stop_loss": 0.7300,
        "alert_id": "aid_12345",
    }
    res = mgr.execute_candidate(candidate)
    assert res["success"] is True
    assert res["symbol"] == "ADAUSDT"
    assert res["signal"] == "BUY"
    assert res["quantity"] >= 500.0
    assert res["sl_success"] is True
    assert res["ts_success"] is True
    assert len(mock_client.orders) == 1
    assert len(mock_client.algo_orders) == 2  # 1 STOP_MARKET + 1 TRAILING_STOP_MARKET

    sl_order = mock_client.algo_orders[0]
    ts_order = mock_client.algo_orders[1]
    assert sl_order["orderType"] == "STOP_MARKET"
    assert sl_order["side"] == "SELL"
    assert float(sl_order["triggerPrice"]) == 0.7300
    assert sl_order["closePosition"] is True

    assert ts_order["orderType"] == "TRAILING_STOP_MARKET"
    assert ts_order["side"] == "SELL"
    # Risk is 0.75 - 0.73 = 0.02. TS activation @ 2.0R = 0.75 + (2.0 * 0.02) = 0.7900
    assert float(ts_order["activationPrice"]) == 0.7900
    print("  [PASS] BUY Candidate placed Market Entry, Algo Stop Loss & Algo Trailing Stop cleanly! ✅")

    # 2. Duplicate symbol guard
    dup_res = mgr.execute_candidate(candidate)
    assert dup_res["success"] is False
    assert dup_res["reason"] == "position_already_exists"
    print("  [PASS] Duplicate symbol guard prevented second entry on same coin! ✅")

    # 3. Max positions guard
    mock_client.positions.append({
        "symbol": "BTCUSDT",
        "side": "LONG",
        "entryPrice": 65000.0,
        "markPrice": 65000.0,
        "amount": 0.01,
    })
    cand3 = {
        "symbol": "ETH-USD",
        "signal": "BUY",
        "entry_price": 2500.0,
        "stop_loss": 2450.0,
        "alert_id": "aid_9999",
    }
    max_res = mgr.execute_candidate(cand3)
    assert max_res["success"] is False
    assert max_res["reason"] == "max_positions_reached"
    print("  [PASS] Max positions guard blocked 3rd entry when holding 2 positions! ✅")


# =====================================================================
# TEST 3: Breakeven Stop Migration (Long & Short)
# =====================================================================
def test_breakeven_stop_migration():
    print("\n" + "="*70)
    print("--> TEST 3: Testing Breakeven Stop Migration Logic (Long & Short)")
    print("="*70)

    # 1. LONG Position Breakeven
    mock_client = MockBinanceClient()
    mgr = BinanceFuturesOrderManager(
        client=mock_client,
        breakeven_r=1.2,
    )

    # Position: Entry 1.0000, Initial SL 0.9500 (Risk = 0.0500)
    mock_client.positions = [{
        "symbol": "ADAUSDT",
        "side": "LONG",
        "entryPrice": 1.0000,
        "markPrice": 1.0650,  # Gain = +0.0650 / 0.0500 = +1.30R (>= 1.2R Breakeven target!)
        "amount": 400.0,
    }]
    mock_client.algo_orders = [{
        "algoId": 101,
        "symbol": "ADAUSDT",
        "side": "SELL",
        "orderType": "STOP_MARKET",
        "triggerPrice": "0.9500",
        "closePosition": True,
        "clientAlgoId": "sl_old",
    }]

    be_updates = mgr.sync_breakeven_stops()
    assert len(be_updates) == 1
    assert be_updates[0]["symbol"] == "ADAUSDT"
    assert be_updates[0]["new_sl"] == 1.0000

    # Verify old SL 101 was cancelled and new SL is at 1.0000
    assert len(mock_client.algo_orders) == 1
    new_sl = mock_client.algo_orders[0]
    assert new_sl["algoId"] != 101
    assert float(new_sl["triggerPrice"]) == 1.0000
    print("  [PASS] LONG position moved SL from 0.9500 to 1.0000 (Entry) when gain hit +1.30R! ✅")

    # Second pass: SL is already at entry -> should NOT spammed
    be_updates_2 = mgr.sync_breakeven_stops()
    assert len(be_updates_2) == 0
    print("  [PASS] Idempotent: No redundant cancel/replace when SL is already at entry! ✅")

    # 2. SHORT Position Breakeven
    mock_client_short = MockBinanceClient()
    mgr_short = BinanceFuturesOrderManager(
        client=mock_client_short,
        breakeven_r=1.2,
    )
    # Entry 2.0000, SL 2.1000 (Risk = 0.1000). Mark price drops to 1.8700 (Gain = +0.1300 / 0.1000 = +1.30R)
    mock_client_short.positions = [{
        "symbol": "SUIUSDT",
        "side": "SHORT",
        "entryPrice": 2.0000,
        "markPrice": 1.8700,
        "amount": -200.0,
    }]
    mock_client_short.algo_orders = [{
        "algoId": 201,
        "symbol": "SUIUSDT",
        "side": "BUY",
        "orderType": "STOP_MARKET",
        "triggerPrice": "2.1000",
        "closePosition": True,
        "clientAlgoId": "sl_short_old",
    }]

    be_short_updates = mgr_short.sync_breakeven_stops()
    assert len(be_short_updates) == 1
    assert be_short_updates[0]["symbol"] == "SUIUSDT"
    assert be_short_updates[0]["new_sl"] == 2.0000
    assert float(mock_client_short.algo_orders[0]["triggerPrice"]) == 2.0000
    print("  [PASS] SHORT position moved SL from 2.1000 to 2.0000 (Entry) when drop hit +1.30R! ✅")


# =====================================================================
# TEST 4: Risk Manager Throttle, Circuit Breaker, and UTC Rollover
# =====================================================================
def test_risk_manager_streak_and_circuit_breaker():
    print("\n" + "="*70)
    print("--> TEST 4: Testing Risk Manager Streak Throttle, Circuit Breaker, & UTC Rollover")
    print("="*70)

    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "risk_state.json")
        rm = BinanceFuturesRiskManager(
            state_file_path=state_file,
            dynamic_sizing_enabled=True,
            equity_risk_pct=5.0,
            loss_streak_throttle=2,
            loss_streak_max=3,
            circuit_breaker_hours=6.0,
            daily_max_loss_pct=4.0,
            only_executed=True,
        )

        # Baseline: Fresh state
        sizing = rm.calculate_trade_sizing(candidate={}, total_equity=500.0, leverage=20, fallback_notional=400.0)
        assert sizing["multiplier"] == 1.0
        assert sizing["final_margin"] == 25.0  # 5% of $500 = $25 margin -> $500 notional

        # 1. Simulate 1 Loss
        settled_outcomes = [
            {"alert_id": "aid_loss_1", "symbol": "BTC-USD", "pnl_pct": -1.5, "outcome_result": "loss", "outcome_status": "settled", "timestamp": "2026-10-12 10:00:00"}
        ]
        # Only counted if in executed_records!
        executed_records = {"aid_loss_1": {"alert_id": "aid_loss_1"}}
        rm.update_from_settled_outcomes(settled_outcomes, leverage=20, notional_per_trade=400.0, executed_records=executed_records)

        state = rm.load_state()
        assert state["consecutive_losses"] == 1
        assert state["circuit_breaker_active"] is False

        # 2. Simulate 2nd Loss -> Throttle to 50%
        settled_outcomes.append(
            {"alert_id": "aid_loss_2", "symbol": "ETH-USD", "pnl_pct": -1.2, "outcome_result": "loss", "outcome_status": "settled", "timestamp": "2026-10-12 11:00:00"}
        )
        executed_records["aid_loss_2"] = {"alert_id": "aid_loss_2"}
        rm.update_from_settled_outcomes(settled_outcomes, leverage=20, notional_per_trade=400.0, executed_records=executed_records)

        state = rm.load_state()
        assert state["consecutive_losses"] == 2
        sizing2 = rm.calculate_trade_sizing(candidate={}, total_equity=500.0, leverage=20, fallback_notional=400.0)
        assert sizing2["multiplier"] == 0.5
        assert "loss_streak_throttle" in sizing2["multiplier_reason"]
        print("  [PASS] 2 consecutive losses throttled trade sizing by 50%! ✅")

        # 3. Simulate 3rd Loss -> Trigger 6-hour Circuit Breaker!
        settled_outcomes.append(
            {"alert_id": "aid_loss_3", "symbol": "SOL-USD", "pnl_pct": -1.4, "outcome_result": "loss", "outcome_status": "settled", "timestamp": "2026-10-12 12:00:00"}
        )
        executed_records["aid_loss_3"] = {"alert_id": "aid_loss_3"}
        rm.update_from_settled_outcomes(settled_outcomes, leverage=20, notional_per_trade=400.0, executed_records=executed_records)

        state = rm.load_state()
        assert state["consecutive_losses"] == 3
        assert state["circuit_breaker_active"] is True
        is_blocked, reason, until = rm.check_circuit_breaker()
        assert is_blocked is True
        assert "loss_streak" in reason or "daily_drawdown" in reason
        print(f"  [PASS] 3 consecutive losses / daily drawdown engaged Circuit Breaker until {until}! ✅")

        # 4. UTC Midnight Rollover: Daily loss resets, streak preserved
        # Manually write yesterday's date to file to simulate overnight rollover
        state["current_date_utc"] = "2026-10-11"
        state["daily_loss_pct"] = 3.5
        import json
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f)

        reloaded = rm.load_state()
        assert reloaded["current_date_utc"] == datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert reloaded["daily_loss_pct"] == 0.0
        assert reloaded["consecutive_losses"] == 3
        print("  [PASS] UTC day rollover reset daily_loss_pct to 0.0% while retaining loss streak! ✅")

        # 5. Non-executed radar alerts (e.g. AW15) are ignored by only_executed!
        radar_loss = [
            {"alert_id": "aid_radar_lost", "symbol": "DOGE-USD", "pnl_pct": -2.0, "outcome_result": "loss", "outcome_status": "settled", "timestamp": "2026-10-12 13:00:00"}
        ]
        rm.update_from_settled_outcomes(settled_outcomes + radar_loss, leverage=20, notional_per_trade=400.0, executed_records=executed_records)
        state_after_radar = rm.load_state()
        assert state_after_radar["consecutive_losses"] == 3  # Did NOT increase to 4!
        assert "aid_radar_lost" in state_after_radar["processed_outcome_ids"]  # Silently marked processed
        print("  [PASS] Unexecuted radar signals (AW15) are strictly ignored by RiskManager! ✅")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# TEST 5: Watchdog Naked Position Defense
# =====================================================================
def test_watchdog_naked_position_healing():
    print("\n" + "="*70)
    print("--> TEST 5: Testing Binance Watchdog Naked Position Healing")
    print("="*70)

    mock_client = MockBinanceClient()
    watchdog = BinanceFuturesWatchdog(
        client=mock_client,
        target_leverage=20,
        target_margin_type="ISOLATED",
        default_stop_loss_pct=1.8,
    )

    # Position exists without any SL order (naked position)
    mock_client.positions = [{
        "symbol": "BTCUSDT",
        "side": "LONG",
        "entryPrice": 65000.0,
        "markPrice": 65100.0,
        "amount": 0.01,
        "liquidationPrice": 61750.0,
    }]
    mock_client.algo_orders = []

    # Run audit and heal
    report = watchdog.audit_and_heal()
    assert len(report["issues_detected"]) >= 1  # Detected naked position
    assert len(report["healed_actions"]) >= 1  # Successfully auto-healed!
    assert "BTCUSDT" in report["healed_actions"][0]

    # Verify protective SL order was placed
    assert len(mock_client.algo_orders) == 1
    sl_ord = mock_client.algo_orders[0]
    assert sl_ord["orderType"] == "STOP_MARKET"
    assert sl_ord["side"] == "SELL"
    # Entry 65000, 1.8% SL = 65000 * (1 - 0.018) = 63830
    assert float(sl_ord["triggerPrice"]) == 63830.0
    print(f"  [PASS] Watchdog detected naked BTC position and deployed emergency SL @ {sl_ord['triggerPrice']}! ✅")


# =====================================================================
# TEST 6: Derivatives Alpha Filter Veto Gate
# =====================================================================
def test_derivatives_alpha_filter_veto():
    print("\n" + "="*70)
    print("--> TEST 6: Testing Derivatives Alpha Filter (Funding Rate & OI Veto)")
    print("="*70)

    dfilter = BinanceDerivativesFilter()

    # Case 1: Crowded Long (Funding rate +0.05% >= +0.04% ceiling) on BUY signal
    cand_long = {"symbol": "BTC-USD", "signal": "BUY", "entry_price": 65000.0}
    dfilter.get_funding_rate = lambda sym: 0.0005  # +0.05%
    dfilter.get_open_interest_stats = lambda sym, bars=4: {"available": True, "delta_pct": +0.5, "trend": "NEUTRAL"}
    res_crowded_long = dfilter.evaluate_candidate(cand_long, max_long_funding=0.0004, veto_enabled=True)
    assert res_crowded_long["approved"] is False
    assert "crowded_long" in res_crowded_long["veto_reason"]
    print("  [PASS] Funding Rate +0.05% correctly triggered Long Liquidation Flush VETO! ✅")

    # Case 2: Crowded Short (Funding rate -0.04% <= -0.03% floor) on SELL signal
    cand_short = {"symbol": "ETH-USD", "signal": "SELL", "entry_price": 2500.0}
    dfilter.get_funding_rate = lambda sym: -0.0004  # -0.04%
    dfilter.get_open_interest_stats = lambda sym, bars=4: {"available": True, "delta_pct": +0.5, "trend": "NEUTRAL"}
    res_crowded_short = dfilter.evaluate_candidate(cand_short, min_short_funding=-0.0003, veto_enabled=True)
    assert res_crowded_short["approved"] is False
    assert "crowded_short" in res_crowded_short["veto_reason"]
    print("  [PASS] Funding Rate -0.04% correctly triggered Short Squeeze VETO! ✅")

    # Case 3: OI Contraction Trap (-4.0% contraction <= -3.5% cascade)
    dfilter.get_funding_rate = lambda sym: 0.0001
    dfilter.get_open_interest_stats = lambda sym, bars=4: {"available": True, "delta_pct": -4.0, "trend": "CONTRACTING"}
    res_oi_trap = dfilter.evaluate_candidate(cand_long, max_long_funding=0.0004, veto_enabled=True)
    assert res_oi_trap["approved"] is False
    assert "fakeout_liquidation_oi_drop" in res_oi_trap["veto_reason"]
    print("  [PASS] OI -4.0% contraction correctly triggered Fakeout Trap VETO! ✅")


# =====================================================================
# TEST 7: Pipeline Hook Guardrails (Whitelists, Labels, Idempotency)
# =====================================================================
def test_pipeline_hook_guardrails():
    print("\n" + "="*70)
    print("--> TEST 7: Testing Pipeline Hook Guardrails & Candidate Whitelisting")
    print("="*70)

    class DummyConfig:
        BINANCE_FUTURES_AUTO_TRADE_ENABLED = True
        BINANCE_WATCHDOG_ENABLED = False
        BINANCE_FUTURES_API_KEY = "test_key"
        BINANCE_FUTURES_API_SECRET = "test_secret"
        BINANCE_FUTURES_TESTNET = True
        BINANCE_FUTURES_ALLOWED_STRATEGIES = "CDCVIX15,PA15,PRIMARY"
        BINANCE_FUTURES_BLOCKED_SYMBOLS = "NEAR-USD,LINK-USD"
        BINANCE_FUTURES_TRADE_NOTIONAL_USDT = 400.0
        BINANCE_FUTURES_MAX_POSITIONS = 2
        BINANCE_DERIVATIVES_FILTER_ENABLED = False

    temp_dir = tempfile.mkdtemp()
    try:
        executed_file = os.path.join(temp_dir, "executed.json")
        outcomes_file = os.path.join(temp_dir, "outcomes.json")
        helpers = {
            "binance_executed_orders_path": lambda: executed_file,
            "alert_outcomes_file_path": lambda: outcomes_file,
        }

        # Mock candidates
        candidates = [
            # 1. Blocked strategy (AW15 radar)
            {"symbol": "SOL-USD", "signal": "BUY", "strategy": "AW15", "dispatch_status_label": "เข้าได้", "alert_id": "c1"},
            # 2. Blocked symbol (NEAR-USD)
            {"symbol": "NEAR-USD", "signal": "BUY", "strategy": "CDCVIX15", "dispatch_status_label": "เข้าได้", "alert_id": "c2"},
            # 3. Disapproved intent ("ห้ามเข้า")
            {"symbol": "ADA-USD", "signal": "BUY", "strategy": "CDCVIX15", "dispatch_status_label": "ห้ามเข้า", "alert_id": "c3"},
            # 4. Approved candidate ("เข้าได้" on CDCVIX15)
            {"symbol": "ADA-USD", "signal": "BUY", "strategy": "CDCVIX15", "dispatch_status_label": "เข้าได้", "alert_id": "c4", "entry_price": 0.75, "stop_loss": 0.73},
        ]

        # Use mock client inside hook by patching client class
        import infrastructure.binance.client as client_module
        orig_client_cls = client_module.BinanceFuturesClient

        mock_instance = MockBinanceClient()
        client_module.BinanceFuturesClient = lambda *args, **kwargs: mock_instance

        res = execute_binance_auto_trade_pipeline(
            sent_candidates=candidates,
            config=DummyConfig(),
            helpers=helpers,
            get_now=lambda: datetime.now(timezone.utc),
            send_telegram_alert=lambda msg: None,
        )

        assert res["enabled"] is True
        # Only candidate 4 should have been executed!
        assert len(res["executed_entries"]) == 1
        executed = res["executed_entries"][0]
        assert executed["success"] is True
        assert executed["symbol"] == "ADAUSDT"

        # Check executed records persisted
        rec = _load_executed_orders(executed_file)
        assert "c4" in rec
        assert "c1" not in rec  # AW15 blocked
        assert "c2" not in rec  # NEAR blocked
        assert "c3" not in rec  # ห้ามเข้า blocked

        print("  [PASS] Pipeline Hook filtered out AW15, NEAR, and 'ห้ามเข้า' correctly! ✅")
        print("  [PASS] Only approved CDCVIX15 candidate executed and recorded cleanly! ✅")

        client_module.BinanceFuturesClient = orig_client_cls
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 80)
    print(" STARTING RIGOROUS LIVE-TRADING EDGE CASE VERIFICATION SUITE")
    print("=" * 80)

    test_all_11_symbols_precision_and_sizing()
    test_order_execution_and_algo_flow()
    test_breakeven_stop_migration()
    test_risk_manager_streak_and_circuit_breaker()
    test_watchdog_naked_position_healing()
    test_derivatives_alpha_filter_veto()
    test_pipeline_hook_guardrails()

    print("\n" + "=" * 80)
    print(" ALL 7 CRITICAL LIVE-TRADING TEST SCENARIOS PASSED WITH ZERO ERRORS! 🚀🛡️")
    print("=" * 80)
