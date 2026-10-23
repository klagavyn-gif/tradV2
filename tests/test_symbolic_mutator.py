"""
Unit tests for Council GP-Tree Symbolic Mutator & Pareto Alpha Forge.
"""

import unittest
import numpy as np
import pandas as pd

from infrastructure.council.symbolic_mutator import (
    parse_dsl_to_tree,
    TreeAlphaMutator,
    SymbolicAlphaForge,
    CandidateIndividual,
    CompareNode,
    LogicalNode,
    ConstantNode,
    FeatureNode,
    TsFuncNode,
)
from infrastructure.council.quantum_lab import (
    enrich_quantum_features,
    evaluate_quantum_expression,
)


class TestSymbolicAlphaMutator(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 100-bar DataFrame for feature enrichment
        np.random.seed(42)
        n = 120
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.random.uniform(0.1, 1.0, n)
        low = close - np.random.uniform(0.1, 1.0, n)
        open_p = close + np.random.randn(n) * 0.2
        volume = np.random.uniform(100, 1000, n)

        raw_df = pd.DataFrame({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        self.df = enrich_quantum_features(raw_df)

    def test_bidirectional_dsl_parsing(self):
        """Verify that standard DSL expressions parse accurately into GP trees and back to DSL."""
        expr1 = "(squeeze_ratio < 0.9500) & (rel_vol > 1.2000)"
        tree1 = parse_dsl_to_tree(expr1)
        self.assertIsInstance(tree1, LogicalNode)
        self.assertEqual(tree1.op, "&")
        self.assertIsInstance(tree1.left, CompareNode)
        self.assertEqual(tree1.left.op, "<")
        self.assertIsInstance(tree1.right, CompareNode)
        self.assertEqual(tree1.right.op, ">")

        # DSL round-trip check
        dsl_out = tree1.to_dsl()
        self.assertIn("squeeze_ratio", dsl_out)
        self.assertIn("rel_vol", dsl_out)

        # Ensure evaluated series over df returns boolean series without error
        res_series = evaluate_quantum_expression(dsl_out, self.df)
        self.assertEqual(len(res_series), len(self.df))
        self.assertTrue(res_series.dtype == bool or res_series.dtype == np.bool_)

    def test_ts_func_dsl_parsing(self):
        """Verify time-series rolling functions parse into TsFuncNode."""
        expr = "(ts_rank(kinetic_energy, 14) > 0.6000)"
        tree = parse_dsl_to_tree(expr)
        self.assertIsInstance(tree, CompareNode)
        self.assertIsInstance(tree.left, TsFuncNode)
        self.assertEqual(tree.left.func_name, "ts_rank")
        self.assertEqual(tree.left.period, 14)
        self.assertIsInstance(tree.left.child, FeatureNode)
        self.assertEqual(tree.left.child.name, "kinetic_energy")

        dsl = tree.to_dsl()
        res_series = evaluate_quantum_expression(dsl, self.df)
        self.assertEqual(len(res_series), len(self.df))

    def test_mutation_operators(self):
        """Test constant jitter, operator mutation, and subtree mutation."""
        mutator = TreeAlphaMutator()
        expr = "(squeeze_ratio < 0.9500) & (rel_vol > 1.2000)"
        tree = parse_dsl_to_tree(expr)

        # Constant jitter
        jittered = mutator.mutate_constants(tree, jitter_pct=0.20)
        self.assertNotEqual(tree.to_dsl(), jittered.to_dsl())
        # Operator toggle
        op_mut = mutator.mutate_operators(tree)
        self.assertIsInstance(op_mut, (LogicalNode, CompareNode))

        # Subtree mutation
        sub_mut = mutator.mutate_subtree(tree, max_depth=2)
        self.assertTrue(sub_mut.depth() >= 1)
        res_series = evaluate_quantum_expression(sub_mut.to_dsl(), self.df)
        self.assertEqual(len(res_series), len(self.df))

    def test_crossover(self):
        """Test subtree crossover between two distinct candidate trees."""
        mutator = TreeAlphaMutator()
        p1 = parse_dsl_to_tree("(wick_absorption > 1.5000) & (vol_skew < -0.5000)")
        p2 = parse_dsl_to_tree("(squeeze_momentum > 0.1000) & (efficiency_ratio > 0.4000)")

        c1, c2 = mutator.crossover(p1, p2)
        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)
        # Verify valid DSL generation
        res1 = evaluate_quantum_expression(c1.to_dsl(), self.df)
        res2 = evaluate_quantum_expression(c2.to_dsl(), self.df)
        self.assertEqual(len(res1), len(self.df))
        self.assertEqual(len(res2), len(self.df))

    def test_symbolic_alpha_forge_evolution(self):
        """Verify the in-memory evolution loop identifies Pareto-optimal alphas in seconds."""
        forge = SymbolicAlphaForge(seed=42)
        seed_hypotheses = [
            {
                "factor_name": "Squeeze Spring",
                "entry_expression": "(squeeze_ratio < 1.0500) & (squeeze_momentum > 0.0000)",
                "exit_expression": "(squeeze_momentum < -0.1000)",
                "direction": "BUY",
            }
        ]

        # Fast 2-generation evolution on small pop
        pareto_alphas = forge.evolve_alpha_pool(
            df=self.df,
            seed_hypotheses=seed_hypotheses,
            population_size=12,
            generations=2,
            top_k=3,
            symbol="BTCUSDT",
        )

        self.assertTrue(len(pareto_alphas) > 0)
        top_alpha = pareto_alphas[0]
        self.assertIsInstance(top_alpha, CandidateIndividual)
        self.assertTrue(top_alpha.complexity > 0)
        self.assertIsInstance(top_alpha.entry_dsl, str)


if __name__ == "__main__":
    unittest.main()
