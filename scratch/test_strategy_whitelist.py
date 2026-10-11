import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, r"E:\5-8-69\tradV2")

import config
from application.services.service_support import parse_symbols_input
from infrastructure.binance.pipeline_hook import execute_binance_auto_trade_pipeline


def test_symbol_sanitization():
    print("--> Test 1: Testing parse_symbols_input sanitization...")
    # Legacy input with NEAR and LINK
    legacy_input = "BTC-USD,DOGE-USD,ETH-USD,ADA-USD,XRP-USD,BNB-USD,SOL-USD,TRX-USD,NEAR-USD,LINK-USD,PAXG-USD"
    
    def dummy_norm(s):
        return s.strip().upper()

    parsed = parse_symbols_input(
        legacy_input,
        normalize_symbol=dummy_norm,
        default_max_symbols=11,
    )
    print("Parsed symbols:", parsed)
    assert "NEAR-USD" not in parsed, "NEAR-USD must be purged!"
    assert "LINK-USD" not in parsed, "LINK-USD must be purged!"
    assert "ONDO-USD" in parsed, "ONDO-USD must be backfilled!"
    assert "SUI-USD" in parsed, "SUI-USD must be backfilled!"
    assert len(parsed) == 11, f"Expected 11 symbols, got {len(parsed)}"
    print("    [PASS] Symbol sanitization verified successfully!")


def test_strategy_whitelist_and_blocked_symbols():
    print("--> Test 2: Testing pipeline_hook Strategy Whitelist & Symbol Block...")
    
    # Mock config
    class MockConfig:
        BINANCE_FUTURES_AUTO_TRADE_ENABLED = True
        BINANCE_WATCHDOG_ENABLED = False
        BINANCE_FUTURES_API_KEY = "dummy_key"
        BINANCE_FUTURES_API_SECRET = "dummy_secret"
        BINANCE_FUTURES_TESTNET = True
        BINANCE_FUTURES_TRADE_NOTIONAL_USDT = 400.0
        BINANCE_FUTURES_MAX_POSITIONS = 2
        BINANCE_FUTURES_LEVERAGE = 20
        BINANCE_FUTURES_MARGIN_TYPE = "ISOLATED"
        BINANCE_FUTURES_DYNAMIC_SIZING_ENABLED = False
        BINANCE_DERIVATIVES_FILTER_ENABLED = False
        BINANCE_FUTURES_ALLOWED_STRATEGIES = "CDCVIX15,PA15,PRIMARY"
        BINANCE_FUTURES_BLOCKED_SYMBOLS = "NEAR-USD,LINK-USD,NEARUSDT,LINKUSDT"

    candidates = [
        # 1. AW15 with "เข้าได้" -> Should be SKIPPED (not in allowed strategies)
        {
            "alert_id": "aw15_test_1",
            "symbol": "BTC-USD",
            "strategy": "AW15",
            "signal": "BUY",
            "dispatch_status_label": "เข้าได้",
            "message": "🟢 เข้าได้",
        },
        # 2. TCB15 with "เข้าได้" -> Should be SKIPPED (not in allowed strategies)
        {
            "alert_id": "tcb15_test_1",
            "symbol": "ADA-USD",
            "strategy": "TCB15",
            "signal": "BUY",
            "dispatch_status_label": "เข้าได้",
            "message": "🟢 เข้าได้",
        },
        # 3. CDCVIX15 on NEAR-USD with "เข้าได้" -> Should be SKIPPED (symbol blocked)
        {
            "alert_id": "cdcvix_near_test",
            "symbol": "NEAR-USD",
            "strategy": "CDCVIX15",
            "signal": "BUY",
            "dispatch_status_label": "เข้าได้",
            "message": "🟢 เข้าได้",
        },
        # 4. PA15 on LINK-USD with "เข้าได้" -> Should be SKIPPED (symbol blocked)
        {
            "alert_id": "pa15_link_test",
            "symbol": "LINK-USD",
            "strategy": "PA15",
            "signal": "BUY",
            "dispatch_status_label": "เข้าได้",
            "message": "🟢 เข้าได้",
        },
        # 5. CDCVIX15 on BTC-USD with "เข้าได้" -> Should be APPROVED (actionable)
        {
            "alert_id": "cdcvix_btc_test",
            "symbol": "BTC-USD",
            "strategy": "CDCVIX15",
            "signal": "BUY",
            "dispatch_status_label": "เข้าได้",
            "message": "🟢 เข้าได้",
            "plan": {"entry_price": 64000.0, "stop_loss": 63000.0},
        },
        # 6. PA15 on SOL-USD with "เข้าได้" -> Should be APPROVED (actionable)
        {
            "alert_id": "pa15_sol_test",
            "symbol": "SOL-USD",
            "strategy": "PA15",
            "signal": "BUY",
            "dispatch_status_label": "เข้าได้",
            "message": "🟢 เข้าได้",
            "plan": {"entry_price": 150.0, "stop_loss": 145.0},
        },
        # 7. PRIMARY on ETH-USD with "เข้าได้" -> Should be APPROVED (actionable)
        {
            "alert_id": "primary_eth_test",
            "symbol": "ETH-USD",
            "strategy": "PRIMARY",
            "signal": "BUY",
            "dispatch_status_label": "เข้าได้",
            "message": "🟢 เข้าได้",
            "plan": {"entry_price": 2600.0, "stop_loss": 2550.0},
        },
    ]

    # Mock order_mgr execute_candidate
    executed = []
    class MockOrderManager:
        def __init__(self, *args, **kwargs):
            pass
        def sync_breakeven_stops(self):
            return []
        def execute_candidate(self, candidate):
            executed.append(candidate["alert_id"])
            return {
                "success": True,
                "alert_id": candidate["alert_id"],
                "symbol": candidate["symbol"],
                "signal": candidate["signal"],
                "entry_price": candidate.get("plan", {}).get("entry_price", 100.0),
                "stop_loss": candidate.get("plan", {}).get("stop_loss", 95.0),
                "quantity": 1.0,
                "notional": 400.0,
            }

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def get_positions(self):
            return []
        def get_usdt_balance(self):
            return {"total": 1000.0}
        def get_open_orders(self):
            return []
        def get_open_algo_orders(self):
            return []

    import infrastructure.binance.order_manager as om
    import infrastructure.binance.client as b_client
    orig_order_mgr = om.BinanceFuturesOrderManager
    om.BinanceFuturesOrderManager = MockOrderManager
    orig_client = b_client.BinanceFuturesClient
    b_client.BinanceFuturesClient = MockClient

    temp_orders_path = r"E:\5-8-69\tradV2\scratch\temp_test_executed_orders.json"
    if os.path.exists(temp_orders_path):
        os.remove(temp_orders_path)

    try:
        res = execute_binance_auto_trade_pipeline(
            candidates,
            config=MockConfig,
            helpers={"binance_executed_orders_path": lambda: temp_orders_path},
            get_now=lambda: None,
            send_telegram_alert=lambda msg: None,
        )
        print("Executed alert IDs:", executed)
        assert "aw15_test_1" not in executed, "AW15 must NOT be executed by auto-trade!"
        assert "tcb15_test_1" not in executed, "TCB15 must NOT be executed by auto-trade!"
        assert "cdcvix_near_test" not in executed, "NEAR-USD must NOT be executed by auto-trade!"
        assert "pa15_link_test" not in executed, "LINK-USD must NOT be executed by auto-trade!"
        assert "cdcvix_btc_test" in executed, "CDCVIX15 on BTC-USD MUST be executed!"
        assert "pa15_sol_test" in executed, "PA15 on SOL-USD MUST be executed!"
        assert "primary_eth_test" in executed, "PRIMARY on ETH-USD MUST be executed!"
        assert len(executed) == 3, f"Expected exactly 3 executed trades, got {len(executed)}"
        print("    [PASS] Winning Trio (CDCVIX15, PA15, PRIMARY) passed flawlessly!")
    finally:
        om.BinanceFuturesOrderManager = orig_order_mgr
        b_client.BinanceFuturesClient = orig_client


if __name__ == "__main__":
    test_symbol_sanitization()
    test_strategy_whitelist_and_blocked_symbols()
    print("\nALL TESTS PASSED! System focus & cleanliness verified.")
