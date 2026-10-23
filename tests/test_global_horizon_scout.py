"""
Unit tests for Council Global Horizon Scout (Monthly Knowledge Harvester).
"""

import json
import tempfile
import unittest
from pathlib import Path

from infrastructure.council.global_horizon_scout import (
    CORE_KNOWLEDGE_BLACKLIST,
    CuratedDiscovery,
    GlobalHorizonScout,
    MonthlyHorizonDigest,
)


class TestGlobalHorizonScout(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.digests_dir = Path(self.tmp_dir.name) / "digests"
        self.outcomes_path = Path(self.tmp_dir.name) / "outcomes.json"
        self.journal_path = Path(self.tmp_dir.name) / "journal.json"

        # Mock sample outcomes
        mock_outcomes = [
            {"alert_intent": "entry", "realized_pnl_pct": 2.5, "exit_reason": "take_profit_hit"},
            {"alert_intent": "entry", "realized_pnl_pct": -1.2, "exit_reason": "stop_loss_hit"},
            {"alert_intent": "entry", "realized_pnl_pct": -1.4, "exit_reason": "stop_loss_hit"},
            {"alert_intent": "entry", "realized_pnl_pct": 0.0, "exit_reason": "breakeven_stop_hit"},
            {"alert_intent": "watch", "realized_pnl_pct": -0.5, "exit_reason": "stop_loss_hit"},
        ]
        self.outcomes_path.write_text(json.dumps(mock_outcomes), encoding="utf-8")

        self.scout = GlobalHorizonScout(
            digests_dir=self.digests_dir,
            outcomes_path=self.outcomes_path,
            journal_path=self.journal_path,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_audit_monthly_performance(self):
        """Verify performance audit accurately identifies bottlenecks and tallies win rates."""
        audit = self.scout.audit_monthly_performance()
        self.assertEqual(audit["total_trades"], 4)  # 4 actionable entries
        self.assertEqual(audit["loss_reasons"].get("stop_loss_hit"), 2)
        self.assertEqual(audit["primary_bottleneck"], "FALSE_BREAKOUT_WHIPSAW")
        self.assertTrue(audit["win_rate_pct"] > 0)

    def test_synthesize_monthly_meta_prompt(self):
        """Verify dynamic prompt incorporates the bottleneck and references the deduplication blacklist."""
        audit = {"primary_bottleneck": "FALSE_BREAKOUT_WHIPSAW", "win_rate_pct": 33.3}
        prompt = self.scout.synthesize_monthly_meta_prompt(audit, "2026-11")

        self.assertIn("2026-11", prompt)
        self.assertIn("False Breakout", prompt)
        self.assertIn("ห้ามนำเสนอเรื่อง Moving Average", prompt)

    def test_curated_scout_discoveries(self):
        """Verify the scout distills robust, non-redundant discoveries with actionable code adaptations."""
        audit = {"primary_bottleneck": "FALSE_BREAKOUT_WHIPSAW", "win_rate_pct": 33.3, "net_pnl_pct": -0.1}
        digest = self.scout._curated_institutional_scout("Test Prompt", "2026-11", audit)

        self.assertIsInstance(digest, MonthlyHorizonDigest)
        self.assertEqual(digest.month_key, "2026-11")
        self.assertTrue(len(digest.discoveries) >= 3)
        for disc in digest.discoveries:
            self.assertIsInstance(disc, CuratedDiscovery)
            self.assertTrue(len(disc.core_concept) > 0)
            self.assertTrue(len(disc.actionable_adaptation) > 0)

    def test_persist_digest_markdown(self):
        """Verify markdown report creation and formatting."""
        disc = CuratedDiscovery(
            title="Novel Microstructure Alpha",
            source_type="arXiv",
            source_url="https://arxiv.org/abs/1234.5678",
            core_concept="Entropy-weighted volatility decay",
            why_relevant_for_tradv2="Improves chop exit detection",
            actionable_adaptation="Implement rolling entropy function in Python",
            suggested_dsl_expression="(wick_absorption > 1.5)",
        )
        digest = MonthlyHorizonDigest(
            month_key="2026-11",
            created_at="2026-11-01 00:00:00 UTC",
            audit_focus="False Breakout Filter",
            evolved_search_prompt="Search Prompt 2026-11",
            discoveries=[disc],
            system_action_plan="Action plan test",
            telegram_summary="TG summary test",
        )

        out_path = self.scout.persist_digest_markdown(digest)
        self.assertTrue(out_path.exists())
        content = out_path.read_text(encoding="utf-8")
        self.assertIn("2026-11_HORIZON_DIGEST.md", str(out_path))
        self.assertIn("Novel Microstructure Alpha", content)
        self.assertIn("wick_absorption > 1.5", content)

    def test_end_to_end_monthly_cycle(self):
        """Verify complete monthly cycle execution creates the file and returns valid digest."""
        digest, path = self.scout.run_monthly_cycle(month_key="2026-11")
        self.assertEqual(digest.month_key, "2026-11")
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
