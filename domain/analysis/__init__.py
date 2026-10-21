"""Analysis domain package providing quantitative metrics and hypothesis testing."""

from domain.analysis.statistical_validator import (
    calculate_moments,
    deflated_sharpe_ratio,
    monte_carlo_drawdown_distribution,
    monte_carlo_permutation_test,
    probabilistic_sharpe_ratio,
    run_statistical_validation,
)

__all__ = [
    "calculate_moments",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "monte_carlo_permutation_test",
    "monte_carlo_drawdown_distribution",
    "run_statistical_validation",
]
