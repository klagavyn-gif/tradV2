import trad
import config

if __name__ == "__main__":
    symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "DOT-USD", "LINK-USD", "AVAX-USD"]

    print("=== WITHOUT Price Pattern Filter ===")
    config.ACTIONZONE_15M_REQUIRE_PATTERN = False
    config.ACTIONZONE_15M_YF_PERIOD = "60d"
    config.ACTIONZONE_15M_USE_OPTIMIZATION = True

    results_no_pattern = []
    for sym in symbols:
        print(f"Testing {sym} (No Pattern)...")
        try:
            res = trad.backtest_actionzone_15m(sym)
            print(f"res: {res}")
            if res and res.get("trades", 0) > 0:
                results_no_pattern.append(res)
                print(f"{sym}: {res['trades']} trades, {res['win_rate_pct']:.2f}% WR, {res['expectancy_rr']:.2f} ExpRR")
            else:
                print(f"{sym}: No trades or returned None")
        except Exception as e:
            print(f"Error on {sym}: {e}")

    print("\n=== WITH Price Pattern Filter ===")
    config.ACTIONZONE_15M_REQUIRE_PATTERN = True

    results_pattern = []
    for sym in symbols:
        print(f"Testing {sym} (With Pattern)...")
        try:
            res = trad.backtest_actionzone_15m(sym)
            print(f"res: {res}")
            if res and res.get("trades", 0) > 0:
                results_pattern.append(res)
                print(f"{sym}: {res['trades']} trades, {res['win_rate_pct']:.2f}% WR, {res['expectancy_rr']:.2f} ExpRR")
            else:
                print(f"{sym}: No trades or returned None")
        except Exception as e:
            print(f"Error on {sym}: {e}")

    total_trades_np = sum(r["trades"] for r in results_no_pattern)
    total_wins_np = sum(r.get("trades", 0) * r.get("win_rate_pct", 0) / 100 for r in results_no_pattern)
    total_rr_np = sum(r.get("avg_rr", 0) * r.get("trades", 0) for r in results_no_pattern)

    total_trades_p = sum(r["trades"] for r in results_pattern)
    total_wins_p = sum(r.get("trades", 0) * r.get("win_rate_pct", 0) / 100 for r in results_pattern)
    total_rr_p = sum(r.get("avg_rr", 0) * r.get("trades", 0) for r in results_pattern)

    print("\n--- Summary ---")
    if total_trades_np > 0:
        print(f"No Pattern: {total_trades_np} trades, WR = {total_wins_np/total_trades_np*100:.2f}%, Total RR = {total_rr_np:.2f}")
    if total_trades_p > 0:
        print(f"With Pattern: {total_trades_p} trades, WR = {total_wins_p/total_trades_p*100:.2f}%, Total RR = {total_rr_p:.2f}")
