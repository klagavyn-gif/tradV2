"""Statistical Validation Engine for Trading Edge & Overfitting Protection.

Implements rigorous econometrics and resampling hypothesis tests:
1. Marcos López de Prado (2014): Deflated Sharpe Ratio (DSR) & Selection Bias Penalty.
2. Bailey & López de Prado (2012): Probabilistic Sharpe Ratio (PSR) for Non-Normality.
3. Robert D. Good / Efron: Monte Carlo Permutation Test (Resampling without replacement).
4. Sequence-of-Returns Stress Testing: Monte Carlo 95% & 99% VaR Max Drawdown.

References:
- López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias,
  Backtest Overfitting, and Non-Normality. Journal of Portfolio Management.
- Bailey, D. H., & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier.
  Journal of Risk, 15(2), 13-44.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# Euler-Mascheroni constant
EULER_MASCHERONI: float = 0.5772156649015329
E_CONST: float = math.e


def normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function Phi(z)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def inv_normal_cdf(p: float) -> float:
    """Inverse standard normal cumulative distribution function (Probit) using Acklam's algorithm.

    Achieves absolute error < 1.15e-9 across (0, 1).
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    q_low = 0.02425
    q_high = 1.0 - q_low
    if p < q_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= q_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def calculate_moments(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """Calculate sample Mean, Standard Deviation, Skewness, and Kurtosis.

    Returns:
        (mean, stdev, skewness, kurtosis)
        Kurtosis is total kurtosis (Normal distribution = 3.0).
    """
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    n = len(clean)
    if n < 2:
        mean_val = clean[0] if clean else 0.0
        return mean_val, 0.0, 0.0, 3.0

    mean_val = sum(clean) / float(n)
    variance = sum((x - mean_val) ** 2 for x in clean) / float(n - 1)
    stdev_val = math.sqrt(variance)
    if stdev_val <= 1e-12:
        return mean_val, 0.0, 0.0, 3.0

    # Standardized residuals
    z = [(x - mean_val) / stdev_val for x in clean]

    # Sample Skewness (unbiased estimator)
    if n >= 3:
        skewness = (float(n) / float((n - 1) * (n - 2))) * sum(zi**3 for zi in z)
    else:
        skewness = sum(zi**3 for zi in z) / float(n)

    # Sample Kurtosis (unbiased estimator, where normal = 3.0)
    if n >= 4:
        term1 = (float(n * (n + 1)) / float((n - 1) * (n - 2) * (n - 3))) * sum(zi**4 for zi in z)
        term2 = (3.0 * float((n - 1) ** 2)) / float((n - 2) * (n - 3))
        kurtosis = term1 - term2 + 3.0
    else:
        kurtosis = sum(zi**4 for zi in z) / float(n)

    return mean_val, stdev_val, skewness, kurtosis


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_samples: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio (PSR) under non-normal returns.

    Computes the probability that the true Sharpe Ratio is greater than benchmark_sr,
    taking into account sample length, return skewness, and fat tails (kurtosis).

    Formula (Bailey & López de Prado, 2012):
        Z = (SR - SR*) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2)
        PSR = Phi(Z)
    """
    if n_samples < 2:
        return 0.5

    denom_sq = 1.0 - skewness * observed_sr + ((kurtosis - 1.0) / 4.0) * (observed_sr**2)
    denom = math.sqrt(max(1e-6, denom_sq))
    z = (observed_sr - benchmark_sr) * math.sqrt(float(n_samples - 1)) / denom
    return normal_cdf(z)


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_samples: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    var_trials_sr: Optional[float] = None,
) -> Tuple[float, float]:
    """Deflated Sharpe Ratio (DSR) under multiple testing selection bias.

    Corrects for the fact that testing multiple strategies/symbols increases
    the maximum Sharpe ratio observed by pure chance.

    Formula (López de Prado, 2014):
        SR* = sqrt(Var[SR_n]) * [ (1 - gamma)*Z^-1(1 - 1/N) + gamma*Z^-1(1 - 1/(N*e)) ]
        DSR = PSR(SR*)

    Returns:
        (dsr_probability, sr_hurdle)
    """
    if n_samples < 2:
        return 0.5, 0.0

    if n_trials <= 1:
        return probabilistic_sharpe_ratio(observed_sr, 0.0, n_samples, skewness, kurtosis), 0.0

    # Standard deviation of Sharpe ratios across trials
    if var_trials_sr is not None and var_trials_sr > 0.0:
        std_trials = math.sqrt(var_trials_sr)
    else:
        # Theoretical null standard error under IID Gaussian returns
        std_trials = math.sqrt(1.0 / float(n_samples - 1))

    # Expected maximum Sharpe Ratio under Null Hypothesis of zero edge
    p1 = max(1e-6, min(1.0 - 1e-6, 1.0 - 1.0 / float(n_trials)))
    p2 = max(1e-6, min(1.0 - 1e-6, 1.0 - 1.0 / float(n_trials * E_CONST)))

    z1 = inv_normal_cdf(p1)
    z2 = inv_normal_cdf(p2)
    sr_hurdle = std_trials * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)

    dsr = probabilistic_sharpe_ratio(observed_sr, sr_hurdle, n_samples, skewness, kurtosis)
    return dsr, sr_hurdle


def monte_carlo_permutation_test(
    trade_returns: Sequence[float],
    n_simulations: int = 5000,
    seed: int = 20261021,
) -> Dict[str, Any]:
    """Monte Carlo Sign-Permutation Hypothesis Test.

    Tests Null Hypothesis H0: The trading strategy has zero directional edge,
    meaning positive and negative trade outcomes occur symmetrically by chance.

    Returns:
        p_value: One-sided empirical p-value.
        is_significant_95: True if p_value < 0.05.
        is_significant_99: True if p_value < 0.01.
    """
    clean = [float(r) for r in trade_returns if isinstance(r, (int, float)) and not math.isnan(r)]
    n = len(clean)
    if n == 0:
        return {
            "sample_size": 0,
            "observed_mean": 0.0,
            "p_value": 1.0,
            "is_significant_95": False,
            "is_significant_99": False,
            "simulations_count": n_simulations,
        }

    observed_mean = sum(clean) / float(n)
    if observed_mean <= 0.0:
        return {
            "sample_size": n,
            "observed_mean": observed_mean,
            "p_value": 1.0,
            "is_significant_95": False,
            "is_significant_99": False,
            "simulations_count": n_simulations,
        }

    rng = random.Random(seed)
    greater_count = 0

    for _ in range(n_simulations):
        sim_sum = 0.0
        for val in clean:
            sign = 1.0 if rng.random() >= 0.5 else -1.0
            sim_sum += sign * abs(val)
        sim_mean = sim_sum / float(n)
        if sim_mean >= observed_mean:
            greater_count += 1

    p_value = float(greater_count + 1) / float(n_simulations + 1)

    return {
        "sample_size": n,
        "observed_mean": round(observed_mean, 4),
        "p_value": round(p_value, 5),
        "is_significant_95": bool(p_value < 0.05),
        "is_significant_99": bool(p_value < 0.01),
        "simulations_count": n_simulations,
    }


def monte_carlo_drawdown_distribution(
    trade_returns: Sequence[float],
    n_simulations: int = 5000,
    seed: int = 20261021,
) -> Dict[str, Any]:
    """Monte Carlo Sequence-of-Returns Drawdown Stress Test.

    Shuffles trade order without replacement to evaluate worst-case
    clustering of losses and compute 95% and 99% VaR Max Drawdowns.
    """
    clean = [float(r) for r in trade_returns if isinstance(r, (int, float)) and not math.isnan(r)]
    n = len(clean)
    if n == 0:
        return {
            "historical_max_dd_pct": 0.0,
            "median_max_dd_pct": 0.0,
            "var_95_max_dd_pct": 0.0,
            "var_99_max_dd_pct": 0.0,
            "worst_simulated_dd_pct": 0.0,
        }

    def _calc_max_dd(seq: List[float]) -> float:
        peak = 100.0
        equity = 100.0
        max_dd = 0.0
        for r in seq:
            equity *= 1.0 + (r / 100.0)
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    historical_dd = _calc_max_dd(clean)

    rng = random.Random(seed)
    simulated_dds: List[float] = []

    for _ in range(n_simulations):
        shuffled = list(clean)
        rng.shuffle(shuffled)
        simulated_dds.append(_calc_max_dd(shuffled))

    simulated_dds.sort()

    idx_50 = min(n_simulations - 1, int(0.50 * n_simulations))
    idx_95 = min(n_simulations - 1, int(0.95 * n_simulations))
    idx_99 = min(n_simulations - 1, int(0.99 * n_simulations))

    return {
        "historical_max_dd_pct": round(historical_dd, 2),
        "median_max_dd_pct": round(simulated_dds[idx_50], 2),
        "var_95_max_dd_pct": round(simulated_dds[idx_95], 2),
        "var_99_max_dd_pct": round(simulated_dds[idx_99], 2),
        "worst_simulated_dd_pct": round(simulated_dds[-1], 2),
    }


def run_statistical_validation(
    trades_or_returns: Sequence[Union[float, Dict[str, Any]]],
    n_trials: int = 11,
    cost_bps: float = 30.0,
    n_simulations: int = 5000,
    seed: int = 20261021,
) -> Dict[str, Any]:
    """Execute full quantitative validation battery on a set of trades.

    Args:
        trades_or_returns: List of float net returns, or list of outcome dicts.
        n_trials: Number of symbols or strategies tested (selection bias penalty).
        cost_bps: Friction in basis points to deduct if outcome dicts are passed.
        n_simulations: Number of Monte Carlo iterations.
        seed: Random seed for reproducibility.

    Returns:
        Structured audit dictionary with DSR, PSR, Permutation test, and DD stress tests.
    """
    cost_pct = float(cost_bps) / 100.0
    returns: List[float] = []

    for item in trades_or_returns:
        if isinstance(item, (int, float)):
            returns.append(float(item))
        elif isinstance(item, dict):
            pnl = item.get("pnl_pct")
            if pnl is not None:
                try:
                    returns.append(float(pnl) - cost_pct)
                except Exception:
                    pass

    n = len(returns)
    if n < 5:
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "sample_size": n,
            "min_required_sample": 5,
            "moments": None,
            "sharpe_tests": None,
            "permutation_test": None,
            "drawdown_stress_test": None,
            "overall_verdict": "COLD_START_INSUFFICIENT_DATA",
            "verdict_reason": f"Only {n} settled entries available. Need >= 5 for valid statistical inference.",
        }

    mean_ret, stdev_ret, skewness, kurtosis = calculate_moments(returns)
    observed_sr = (mean_ret / stdev_ret) if stdev_ret > 1e-12 else 0.0

    # Sharpe Tests (PSR and DSR)
    psr_zero = probabilistic_sharpe_ratio(observed_sr, 0.0, n, skewness, kurtosis)
    dsr_prob, sr_hurdle = deflated_sharpe_ratio(observed_sr, n_trials, n, skewness, kurtosis)

    # Permutation Test
    perm_result = monte_carlo_permutation_test(returns, n_simulations=n_simulations, seed=seed)

    # Drawdown Stress Test
    dd_result = monte_carlo_drawdown_distribution(returns, n_simulations=n_simulations, seed=seed)

    # Formal Verdict Synthesis
    dsr_passed = bool(dsr_prob >= 0.95)
    perm_passed = bool(perm_result.get("is_significant_95"))

    if dsr_passed and perm_passed:
        overall_verdict = "APPROVED_FOR_LIVE"
        verdict_reason = (
            f"Statistically validated edge (DSR={dsr_prob:.3f} >= 0.95, p={perm_result['p_value']:.4f} < 0.05). "
            f"Multi-testing selection bias across {n_trials} candidates rejected."
        )
    elif dsr_prob >= 0.80 and perm_passed:
        overall_verdict = "PROVISIONAL_PASS_WATCH"
        verdict_reason = (
            f"Directional edge is significant (p={perm_result['p_value']:.4f}), but DSR ({dsr_prob:.3f}) is under 0.95 "
            f"hurdle due to testing {n_trials} assets. Accumulate more sample."
        )
    else:
        overall_verdict = "OVERFITTING_REJECTED"
        verdict_reason = (
            f"Cannot distinguish from data snooping/luck (DSR={dsr_prob:.3f}, p={perm_result['p_value']:.4f}). "
            f"Fails institutional significance."
        )

    return {
        "status": "COMPLETED",
        "sample_size": n,
        "n_trials_penalized": n_trials,
        "moments": {
            "mean_net_return_pct": round(mean_ret, 3),
            "stdev_net_return_pct": round(stdev_ret, 3),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
            "per_trade_sharpe": round(observed_sr, 4),
        },
        "sharpe_tests": {
            "observed_sr": round(observed_sr, 4),
            "psr_vs_zero": round(psr_zero, 4),
            "sr_hurdle": round(sr_hurdle, 4),
            "deflated_sharpe_ratio": round(dsr_prob, 4),
            "dsr_passed_95": dsr_passed,
        },
        "permutation_test": perm_result,
        "drawdown_stress_test": dd_result,
        "overall_verdict": overall_verdict,
        "verdict_reason": verdict_reason,
    }
