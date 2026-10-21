"""Unit tests for Statistical Validation Engine (DSR, PSR, Permutation Test, Monte Carlo DD)."""

import math
import unittest

from domain.analysis.statistical_validator import (
    calculate_moments,
    deflated_sharpe_ratio,
    inv_normal_cdf,
    monte_carlo_drawdown_distribution,
    monte_carlo_permutation_test,
    normal_cdf,
    probabilistic_sharpe_ratio,
    run_statistical_validation,
)


class TestStatisticalValidator(unittest.TestCase):
    """Test suite for statistical validation algorithms."""

    def test_normal_and_inverse_cdf_precision(self):
        """Verify normal CDF and Acklam inverse CDF precision."""
        self.assertAlmostEqual(normal_cdf(0.0), 0.5, places=5)
        self.assertAlmostEqual(normal_cdf(1.959964), 0.975, places=4)
        self.assertAlmostEqual(normal_cdf(-1.959964), 0.025, places=4)

        self.assertAlmostEqual(inv_normal_cdf(0.5), 0.0, places=5)
        self.assertAlmostEqual(inv_normal_cdf(0.975), 1.959964, places=4)
        self.assertAlmostEqual(inv_normal_cdf(0.025), -1.959964, places=4)
        self.assertAlmostEqual(inv_normal_cdf(0.95), 1.644854, places=4)

    def test_calculate_moments_symmetric_and_skewed(self):
        """Verify calculation of mean, stdev, skewness, and kurtosis."""
        # Perfectly symmetric distribution around 0
        symmetric = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        mean, stdev, skew, kurt = calculate_moments(symmetric)
        self.assertAlmostEqual(mean, 0.0, places=5)
        self.assertAlmostEqual(skew, 0.0, places=4)

        # Positively skewed distribution (small losses, rare huge wins)
        pos_skew = [-1.0, -1.0, -1.0, -0.5, 0.0, 0.5, 10.0]
        mean_p, _, skew_p, _ = calculate_moments(pos_skew)
        self.assertGreater(skew_p, 1.0)
        self.assertGreater(mean_p, 0.0)

        # Edge cases: 0 or 1 element
        m0, s0, sk0, k0 = calculate_moments([])
        self.assertEqual(m0, 0.0)
        self.assertEqual(s0, 0.0)

        m1, s1, sk1, k1 = calculate_moments([5.0])
        self.assertEqual(m1, 5.0)
        self.assertEqual(s1, 0.0)

    def test_probabilistic_sharpe_ratio_penalties(self):
        """Verify PSR responds correctly to sample size, skewness, and fat-tails."""
        # Baseline normal return with positive Sharpe
        psr_normal = probabilistic_sharpe_ratio(
            observed_sr=0.3, benchmark_sr=0.0, n_samples=50, skewness=0.0, kurtosis=3.0
        )
        self.assertGreater(psr_normal, 0.95)

        # Negative skew (asymmetric downside risk) must reduce confidence
        psr_neg_skew = probabilistic_sharpe_ratio(
            observed_sr=0.3, benchmark_sr=0.0, n_samples=50, skewness=-1.5, kurtosis=3.0
        )
        self.assertLess(psr_neg_skew, psr_normal)

        # High kurtosis (fat tails) must reduce confidence
        psr_fat_tail = probabilistic_sharpe_ratio(
            observed_sr=0.3, benchmark_sr=0.0, n_samples=50, skewness=0.0, kurtosis=8.0
        )
        self.assertLess(psr_fat_tail, psr_normal)

    def test_deflated_sharpe_ratio_trial_penalty(self):
        """Verify DSR hurdle increases monotonically with number of trials."""
        observed_sr = 0.4
        n_samples = 60

        dsr_1, hurdle_1 = deflated_sharpe_ratio(observed_sr, n_trials=1, n_samples=n_samples)
        dsr_5, hurdle_5 = deflated_sharpe_ratio(observed_sr, n_trials=5, n_samples=n_samples)
        dsr_20, hurdle_20 = deflated_sharpe_ratio(observed_sr, n_trials=20, n_samples=n_samples)
        dsr_100, hurdle_100 = deflated_sharpe_ratio(observed_sr, n_trials=100, n_samples=n_samples)

        # Hurdle must strictly increase with more trials (selection bias penalization)
        self.assertLess(hurdle_1, hurdle_5)
        self.assertLess(hurdle_5, hurdle_20)
        self.assertLess(hurdle_20, hurdle_100)

        # DSR confidence must strictly decrease as more trials are penalized
        self.assertGreater(dsr_1, dsr_5)
        self.assertGreater(dsr_5, dsr_20)
        self.assertGreater(dsr_20, dsr_100)

    def test_monte_carlo_permutation_genuine_edge_vs_noise(self):
        """Verify permutation test distinguishes genuine positive edge from noise."""
        # Strong positive edge
        strong_edge = [1.5, 2.0, -1.0, 3.5, 0.5, -0.8, 2.8, -1.2, 4.0, 1.8] * 5  # 50 trades
        res_edge = monte_carlo_permutation_test(strong_edge, n_simulations=2000, seed=42)
        self.assertTrue(res_edge["is_significant_95"])
        self.assertLess(res_edge["p_value"], 0.05)

        # Pure symmetric noise around 0 (no edge)
        pure_noise = [-1.0, 1.0, -2.0, 2.0, -0.5, 0.5, -1.5, 1.5] * 6
        res_noise = monte_carlo_permutation_test(pure_noise, n_simulations=2000, seed=42)
        self.assertFalse(res_noise["is_significant_95"])
        self.assertGreater(res_noise["p_value"], 0.20)

    def test_monte_carlo_drawdown_stress_ordering(self):
        """Verify drawdown percentiles maintain strict risk ordering."""
        trades = [2.0, -1.5, 3.0, -2.0, 1.0, -1.0, 4.0, -3.0] * 5
        dd_res = monte_carlo_drawdown_distribution(trades, n_simulations=2000, seed=42)

        # Risk ordering: Median <= 95% VaR <= 99% VaR <= Worst Case
        self.assertLessEqual(dd_res["median_max_dd_pct"], dd_res["var_95_max_dd_pct"])
        self.assertLessEqual(dd_res["var_95_max_dd_pct"], dd_res["var_99_max_dd_pct"])
        self.assertLessEqual(dd_res["var_99_max_dd_pct"], dd_res["worst_simulated_dd_pct"])
        self.assertGreater(dd_res["historical_max_dd_pct"], 0.0)

    def test_run_statistical_validation_cold_start_and_complete(self):
        """Verify integrated battery handles cold start and full execution."""
        # Cold start (< 5 trades)
        cold_res = run_statistical_validation([1.0, -0.5, 2.0])
        self.assertEqual(cold_res["status"], "INSUFFICIENT_SAMPLE")
        self.assertEqual(cold_res["overall_verdict"], "COLD_START_INSUFFICIENT_DATA")

        # Full sample with dictionaries
        trade_dicts = [
            {"pnl_pct": 2.5},
            {"pnl_pct": -0.8},
            {"pnl_pct": 3.2},
            {"pnl_pct": -1.2},
            {"pnl_pct": 1.5},
            {"pnl_pct": -0.5},
            {"pnl_pct": 4.0},
            {"pnl_pct": -1.0},
        ] * 4  # 32 trades
        full_res = run_statistical_validation(
            trade_dicts, n_trials=11, cost_bps=30.0, n_simulations=1000, seed=42
        )
        self.assertEqual(full_res["status"], "COMPLETED")
        self.assertIn("moments", full_res)
        self.assertIn("sharpe_tests", full_res)
        self.assertIn("permutation_test", full_res)
        self.assertIn("drawdown_stress_test", full_res)
        self.assertIn(
            full_res["overall_verdict"],
            ["APPROVED_FOR_LIVE", "PROVISIONAL_PASS_WATCH", "OVERFITTING_REJECTED"],
        )


if __name__ == "__main__":
    unittest.main()
