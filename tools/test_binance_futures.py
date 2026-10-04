#!/usr/bin/env python3
"""
Test & Diagnostic Tool for Binance USDT-M Futures Connector.
Usage:
    python tools/test_binance_futures.py --ping
    python tools/test_binance_futures.py --balance
    python tools/test_binance_futures.py --positions
    python tools/test_binance_futures.py --dry-run
"""

import os
import sys
import argparse
from pathlib import Path

# Add workspace root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from infrastructure.binance import BinanceFuturesClient, BinanceFuturesOrderManager, normalize_futures_symbol

def main():
    parser = argparse.ArgumentParser(description="Binance USDT-M Futures Diagnostic Tool")
    parser.add_argument("--testnet", action="store_true", default=False, help="Use Testnet")
    parser.add_argument("--live", action="store_true", default=False, help="Use Live account")
    parser.add_argument("--api-key", default=None, help="Binance API Key (default: reads config/env)")
    parser.add_argument("--api-secret", default=None, help="Binance API Secret (default: reads config/env)")
    parser.add_argument("--ping", action="store_true", help="Ping Binance Futures server")
    parser.add_argument("--balance", action="store_true", help="Check account USDT balance")
    parser.add_argument("--positions", action="store_true", help="List active open positions")
    parser.add_argument("--symbol-info", default=None, help="Inspect precision and minNotional for symbol (e.g. ADA-USD or BTCUSDT)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate sizing and pre-flight checks for a test candidate")

    args = parser.parse_args()

    if args.live:
        testnet = False
    elif args.testnet:
        testnet = True
    else:
        testnet = getattr(config, "BINANCE_FUTURES_TESTNET", True)

    api_key = args.api_key or getattr(config, "BINANCE_FUTURES_API_KEY", "") or os.environ.get("BINANCE_FUTURES_API_KEY", "")
    api_secret = args.api_secret or getattr(config, "BINANCE_FUTURES_API_SECRET", "") or os.environ.get("BINANCE_FUTURES_API_SECRET", "")

    mode_label = "TESTNET (Simulation)" if testnet else "LIVE (Real Money)"
    print("=" * 70)
    print(f" BINANCE USDT-M FUTURES CONNECTOR: {mode_label}")
    print("=" * 70)

    client = BinanceFuturesClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )

    # 1. Ping
    server_time = client.get_server_time()
    if server_time:
        print(f" [PASS] Server Ping OK! Server time ms: {server_time}")
    else:
        print(" [FAIL] Cannot connect to Binance Futures endpoint!")
        return

    # 2. Symbol Info
    target_sym = args.symbol_info or "ADA-USD"
    b_sym = normalize_futures_symbol(target_sym)
    filters = client.get_symbol_filters(b_sym)
    if filters:
        print(f"\n Symbol Info for {target_sym} -> {b_sym}:")
        print(f"   Tick Size: {filters['tickSize']}")
        print(f"   Step Size: {filters['stepSize']}")
        print(f"   Min Qty:   {filters['minQty']}")
        print(f"   Min Notional: {filters['minNotional']} USDT")

    # 3. Balance & Auth Check
    if args.balance or not (args.ping or args.symbol_info or args.dry_run):
        if not api_key or not api_secret:
            print("\n [INFO] API Key / Secret not configured. (Skipping private account endpoints)")
            print(" To test authenticated endpoints, provide --api-key and --api-secret or set in .env:")
            print("   BINANCE_FUTURES_API_KEY=your_key")
            print("   BINANCE_FUTURES_API_SECRET=your_secret")
        else:
            print("\n Checking Account Balance...")
            res = client._request("GET", "/fapi/v2/balance", signed=True)
            if not res.get("success"):
                code = res.get("code")
                msg = res.get("msg")
                print(f"   [FAIL] Authentication failed! Code: {code}, Message: {msg}")
                if code == -2015:
                    print("   [HINT] Code -2015 means Invalid API-key, IP restriction, or wrong network (Testnet vs Live).")
                    if testnet:
                        print("   [HINT] Currently testing in TESTNET mode. If your key is for LIVE Binance.com, add --live")
                    else:
                        print("   [HINT] Currently testing in LIVE mode. If your key is for TESTNET, add --testnet")
            else:
                usdt = client.get_usdt_balance()
                print(f"   [PASS] Total USDT:     {usdt['total']:,.2f} USDT")
                print(f"   [PASS] Available USDT: {usdt['available']:,.2f} USDT")
                print(f"          Unrealized PnL: {usdt['crossUnPnl']:+,.2f} USDT")

    # 4. Positions Check
    if args.positions:
        if not api_key or not api_secret:
            print("\n [INFO] API Key / Secret required for positions query.")
        else:
            print("\n Checking Active Positions...")
            positions = client.get_positions()
            if not positions:
                print("   No active open positions.")
            for p in positions:
                print(f"   {p['symbol']}: {p['side']} {p['positionAmt']} @ {p['entryPrice']} (Mark: {p['markPrice']}, PnL: {p['unRealizedProfit']:+.2f} USDT)")

    # 5. Dry-Run Candidate Sizing
    if args.dry_run:
        print("\n Dry-Run Order Manager Simulation...")
        order_mgr = BinanceFuturesOrderManager(
            client=client,
            trade_notional_usdt=getattr(config, "BINANCE_FUTURES_TRADE_NOTIONAL_USDT", 10.0),
            max_positions=getattr(config, "BINANCE_FUTURES_MAX_POSITIONS", 2),
            leverage=getattr(config, "BINANCE_FUTURES_LEVERAGE", 1),
            breakeven_r=getattr(config, "TELEGRAM_ALERT_REALIZED_BREAKEVEN_R", 1.2),
            trailing_r=getattr(config, "TELEGRAM_ALERT_REALIZED_TRAILING_R", 2.0),
            trailing_dist_r=getattr(config, "TELEGRAM_ALERT_REALIZED_TRAILING_DISTANCE_R", 0.8),
        )
        sample_candidate = {
            "symbol": "ADA-USD",
            "signal": "BUY",
            "entry_price": 0.2450,
            "stop_loss": 0.2370,
            "take_profit": 0.2700,
            "alert_id": "dry_run_001",
        }
        sym = normalize_futures_symbol(sample_candidate["symbol"])
        f = client.get_symbol_filters(sym)
        if f:
            notional = max(order_mgr.trade_notional_usdt, f["minNotional"])
            from infrastructure.binance.order_manager import round_to_step, round_to_tick
            qty = round_to_step(notional / sample_candidate["entry_price"], f["stepSize"])
            risk = abs(sample_candidate["entry_price"] - sample_candidate["stop_loss"])
            be_trigger = sample_candidate["entry_price"] + (order_mgr.breakeven_r * risk)
            ts_trigger = sample_candidate["entry_price"] + (order_mgr.trailing_r * risk)
            callback_pct = round(min(5.0, max(0.1, (order_mgr.trailing_dist_r * risk / sample_candidate["entry_price"]) * 100.0)), 1)
            print(f"   Target Notional: {notional} USDT")
            print(f"   Order Quantity:  {qty} {f['baseAsset']}")
            print(f"   Initial SL:      {round_to_tick(sample_candidate['stop_loss'], f['tickSize'])}")
            print(f"   BE Trigger (+{order_mgr.breakeven_r}R): {round_to_tick(be_trigger, f['tickSize'])} (Move SL to Entry)")
            print(f"   TS Trigger (+{order_mgr.trailing_r}R): {round_to_tick(ts_trigger, f['tickSize'])} (Trail by {callback_pct}%)")
            print("   [PASS] Pre-flight calculation successful!")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
