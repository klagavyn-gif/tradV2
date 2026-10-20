"""
Unit test for NewsSentinel, TradingCouncil (6-specialist board), and Momentum Exhaustion Harvester.
"""

import unittest
from datetime import datetime, timezone, timedelta
from infrastructure.council.news_sentinel import NewsSentinel
from infrastructure.council.arbiter import TradingCouncil
from infrastructure.council.blackboard import CentralBlackboard
from infrastructure.council.specialists import CouncilOpinion, AlphaSpecialist, MacroSpecialist, DerivativesSpecialist, RiskSpecialist, OpsSpecialist
from infrastructure.binance.order_manager import BinanceFuturesOrderManager


class MockBinanceClient:
    def __init__(self):
        self.orders = []
        self.algo_orders = []

    def get_positions(self):
        return []

    def create_algo_order(self, **kwargs):
        self.algo_orders.append(kwargs)
        return {"success": True, "algoId": 12345, "clientAlgoId": kwargs.get("client_algo_id")}

    def cancel_algo_order(self, symbol, **kwargs):
        return {"success": True}

    def cancel_order(self, symbol, **kwargs):
        return {"success": True}

    def get_symbol_filters(self, symbol):
        return {"tick_size": 0.1, "step_size": 0.001, "min_notional": 5.0}


class TestNewsSentinelAndHarvester(unittest.TestCase):
    def test_news_sentinel_clear(self):
        sentinel = NewsSentinel(freeze_minutes_before=30.0, freeze_minutes_after=30.0)
        # Use a normal non-NFP Thursday date
        normal_dt = datetime(2026, 10, 15, 10, 0, 0, tzinfo=timezone.utc)
        is_frozen, event, details = sentinel.check_macro_news_freeze(now_dt=normal_dt, events=[])
        self.assertFalse(is_frozen)
        
        opinion = sentinel.evaluate({"symbol": "ETHUSDT", "signal": "BUY"}, now_dt=normal_dt)
        self.assertEqual(opinion.status, "APPROVED")
        self.assertFalse(opinion.veto)
        self.assertEqual(opinion.confidence, 85.0)

    def test_news_sentinel_event_veto(self):
        sentinel = NewsSentinel(freeze_minutes_before=30.0, freeze_minutes_after=30.0)
        cpi_time = datetime(2026, 10, 14, 12, 30, 0, tzinfo=timezone.utc)
        events = [
            {"event": "US CPI", "time_utc": cpi_time.isoformat(), "impact": "HIGH"}
        ]
        # Check 15 minutes before CPI
        now_dt = cpi_time - timedelta(minutes=15)
        is_frozen, event, details = sentinel.check_macro_news_freeze(now_dt=now_dt, events=events)
        self.assertTrue(is_frozen)
        self.assertEqual(event, "US CPI")
        
        opinion = sentinel.evaluate({"symbol": "ETHUSDT", "signal": "BUY"}, now_dt=now_dt)
        # Wait, if sentinel.evaluate loads from file, we can test check_macro_news_freeze directly or mock load_events
        sentinel.load_events = lambda: events
        opinion2 = sentinel.evaluate({"symbol": "ETHUSDT", "signal": "BUY"}, now_dt=now_dt)
        self.assertEqual(opinion2.status, "VETOED")
        self.assertTrue(opinion2.veto)
        self.assertIn("cpi", opinion2.veto_reason)

    def test_news_sentinel_nfp_rule(self):
        sentinel = NewsSentinel(freeze_minutes_before=30.0, freeze_minutes_after=30.0)
        # First Friday of Oct 2026 is Oct 2, 2026. NFP @ 12:30 UTC
        nfp_friday = datetime(2026, 10, 2, 12, 40, 0, tzinfo=timezone.utc)
        is_frozen, event, details = sentinel.check_macro_news_freeze(now_dt=nfp_friday, events=[])
        self.assertTrue(is_frozen)
        self.assertIn("NFP", event)

    def test_council_6_specialists_deliberation(self):
        bb = CentralBlackboard(file_path=".data/test_deliberations.json")
        sentinel = NewsSentinel()
        sentinel.check_macro_news_freeze = lambda now_dt=None, events=None: (False, None, None)
        
        council = TradingCouncil(
            blackboard=bb,
            alpha_specialist=AlphaSpecialist(),
            macro_specialist=MacroSpecialist(),
            derivatives_specialist=DerivativesSpecialist(),
            news_specialist=sentinel,
            risk_specialist=RiskSpecialist(),
            ops_specialist=OpsSpecialist(),
            min_composite_score=65.0,
        )

        candidate = {
            "symbol": "ETHUSDT",
            "strategy": "CDCVIX15",
            "signal": "BUY",
            "dispatch_status_label": "เข้าได้",
            "entry_price": 3000.0,
            "stop_loss": 2940.0,
            "take_profit": 3120.0,
            "ai_prob_win": 0.65,
        }

        verdict = council.deliberate(
            candidate,
            total_equity=1000.0,
            open_positions_count=0,
            live_price=3002.0,
        )

        self.assertEqual(verdict.decision, "EXECUTE")
        self.assertTrue(verdict.approved)
        self.assertEqual(len(verdict.opinions), 7)
        self.assertIn("news_specialist", verdict.opinions)
        self.assertIn("contrarian_scout", verdict.opinions)
        self.assertIn("/7", verdict.summary)
        print("Verdict Summary:", verdict.summary)

    def test_momentum_harvester_tightening(self):
        client = MockBinanceClient()
        order_mgr = BinanceFuturesOrderManager(
            client=client,
            breakeven_r=1.2,
            trailing_r=2.0,
            trailing_dist_r=0.8,
            harvester_trigger_r=3.0,
            harvester_dist_r=0.4,
        )

        # 1. Normal trailing stop at 2.2R (dist = 0.8R)
        # Entry = 100, Initial SL = 90 (Risk = 10)
        # Mark price = 122 -> Gain = +22 / 10 = 2.2R
        # Expected new SL = 122 - (0.8 * 10) = 114.0
        client.get_positions = lambda: [
            {"symbol": "ETHUSDT", "side": "LONG", "positionAmt": 1.0, "entryPrice": 100.0, "markPrice": 122.0}
        ]
        # Open algo stop order at 90.0
        open_algos = [
            {"symbol": "ETHUSDT", "algoId": "sl1", "triggerPrice": "90.0", "side": "SELL", "orderType": "STOP_MARKET"}
        ]
        client.get_open_algo_orders = lambda symbol=None: open_algos
        client.list_open_algo_orders = lambda symbol=None: open_algos
        client.get_open_orders = lambda symbol=None: []
        client.get_exchange_info = lambda: {
            "symbols": [{"symbol": "ETHUSDT", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.1"}]}]
        }

        updates = order_mgr.sync_breakeven_stops()
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["reason"], "trailing_stop")
        self.assertEqual(updates[0]["trailing_distance_r"], 0.8)
        self.assertEqual(updates[0]["new_sl"], 114.0)

        # 2. Momentum exhaustion tightening at 3.5R (dist = 0.4R)
        # Mark price = 135 -> Gain = +35 / 10 = 3.5R
        # Expected new SL = 135 - (0.4 * 10) = 131.0
        client.get_positions = lambda: [
            {"symbol": "ETHUSDT", "side": "LONG", "positionAmt": 1.0, "entryPrice": 100.0, "markPrice": 135.0}
        ]
        open_algos = [
            {"symbol": "ETHUSDT", "algoId": "sl2", "triggerPrice": "114.0", "side": "SELL", "orderType": "STOP_MARKET"}
        ]
        executed_records = {
            "rec1": {"symbol": "ETHUSDT", "entry_price": 100.0, "stop_loss": 90.0, "signal": "BUY", "executed_at": "2026-10-17T12:00:00Z"}
        }
        updates2 = order_mgr.sync_breakeven_stops(executed_records=executed_records)
        self.assertEqual(len(updates2), 1)
        self.assertEqual(updates2[0]["reason"], "momentum_exhaustion_tightened")
        self.assertEqual(updates2[0]["trailing_distance_r"], 0.4)
        self.assertEqual(updates2[0]["new_sl"], 131.0)
        print("Harvester tightened SL successfully to:", updates2[0]["new_sl"])


if __name__ == "__main__":
    unittest.main()
