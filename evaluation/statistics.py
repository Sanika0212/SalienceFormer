"""
Statistical Testing Utilities for SalienceFormer Evaluation

Significance tests, confidence intervals, and result aggregation.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

import numpy as np

# Lazy import for scipy
scipy_stats = None


def _ensure_scipy():
    """Lazily import scipy.stats."""
    global scipy_stats
    if scipy_stats is None:
        from scipy import stats
        scipy_stats = stats
    return scipy_stats


@dataclass
class StatisticalResult:
    """Result of a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""

    def __str__(self) -> str:
        sig_str = "significant" if self.significant else "not significant"
        s = f"{self.test_name}: statistic={self.statistic:.4f}, p={self.p_value:.4f} ({sig_str})"
        if self.effect_size is not None:
            s += f", effect_size={self.effect_size:.4f}"
        if self.confidence_interval:
            s += f", CI=({self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f})"
        return s


def compute_confidence_interval(
    values: Union[List[float], np.ndarray],
    confidence: float = 0.95,
    method: str = "t",
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a sample.

    Args:
        values: Sample values
        confidence: Confidence level (default 0.95 for 95% CI)
        method: 't' for t-distribution, 'bootstrap' for bootstrap

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)

    if n < 2:
        return mean, mean, mean

    if method == "t":
        stats = _ensure_scipy()
        se = np.std(values, ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_crit * se
        return mean, mean - margin, mean + margin

    elif method == "bootstrap":
        n_bootstrap = 10000
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        return mean, lower, upper

    else:
        raise ValueError(f"Unknown method: {method}")


def paired_significance_test(
    values_a: Union[List[float], np.ndarray],
    values_b: Union[List[float], np.ndarray],
    alpha: float = 0.05,
    test: str = "auto",
) -> StatisticalResult:
    """
    Perform paired significance test between two sets of measurements.

    Args:
        values_a: Measurements from condition A
        values_b: Measurements from condition B
        alpha: Significance level
        test: Test type - 'ttest', 'wilcoxon', or 'auto' (chooses based on normality)

    Returns:
        StatisticalResult with test statistics and interpretation
    """
    stats = _ensure_scipy()

    values_a = np.array(values_a)
    values_b = np.array(values_b)

    if len(values_a) != len(values_b):
        raise ValueError("Arrays must have same length for paired test")

    n = len(values_a)
    differences = values_a - values_b

    # Auto-select test based on normality
    if test == "auto":
        if n >= 20:
            _, normality_p = stats.shapiro(differences)
            test = "ttest" if normality_p > 0.05 else "wilcoxon"
        else:
            test = "wilcoxon"  # More conservative for small samples

    if test == "ttest":
        statistic, p_value = stats.ttest_rel(values_a, values_b)
        test_name = "Paired t-test"

        # Cohen's d effect size
        effect_size = np.mean(differences) / np.std(differences, ddof=1)

    elif test == "wilcoxon":
        # Handle case where all differences are zero
        if np.all(differences == 0):
            return StatisticalResult(
                test_name="Wilcoxon signed-rank test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                effect_size=0.0,
                interpretation="No difference between conditions (all differences = 0)",
            )

        statistic, p_value = stats.wilcoxon(values_a, values_b, alternative="two-sided")
        test_name = "Wilcoxon signed-rank test"

        # Rank-biserial correlation as effect size
        effect_size = 1 - (2 * statistic) / (n * (n + 1) / 2)

    else:
        raise ValueError(f"Unknown test: {test}")

    significant = p_value < alpha

    # Interpretation
    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    direction = "A > B" if mean_a > mean_b else "A < B" if mean_a < mean_b else "A = B"

    if significant:
        interpretation = f"Significant difference ({direction}), p={p_value:.4f}"
    else:
        interpretation = f"No significant difference, p={p_value:.4f}"

    # Confidence interval for mean difference
    mean_diff, ci_lower, ci_upper = compute_confidence_interval(differences)

    return StatisticalResult(
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        effect_size=effect_size,
        confidence_interval=(ci_lower, ci_upper),
        interpretation=interpretation,
    )


def unpaired_significance_test(
    values_a: Union[List[float], np.ndarray],
    values_b: Union[List[float], np.ndarray],
    alpha: float = 0.05,
    test: str = "auto",
) -> StatisticalResult:
    """
    Perform unpaired (independent) significance test.

    Args:
        values_a: Measurements from group A
        values_b: Measurements from group B
        alpha: Significance level
        test: Test type - 'ttest', 'mannwhitney', or 'auto'

    Returns:
        StatisticalResult with test statistics
    """
    stats = _ensure_scipy()

    values_a = np.array(values_a)
    values_b = np.array(values_b)

    # Auto-select test
    if test == "auto":
        # Check normality and variance homogeneity
        _, norm_p_a = stats.shapiro(values_a) if len(values_a) >= 3 else (0, 0.1)
        _, norm_p_b = stats.shapiro(values_b) if len(values_b) >= 3 else (0, 0.1)

        if norm_p_a > 0.05 and norm_p_b > 0.05:
            test = "ttest"
        else:
            test = "mannwhitney"

    if test == "ttest":
        statistic, p_value = stats.ttest_ind(values_a, values_b)
        test_name = "Independent t-test"

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(values_a) - 1) * np.var(values_a, ddof=1) +
             (len(values_b) - 1) * np.var(values_b, ddof=1)) /
            (len(values_a) + len(values_b) - 2)
        )
        effect_size = (np.mean(values_a) - np.mean(values_b)) / pooled_std if pooled_std > 0 else 0

    elif test == "mannwhitney":
        statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
        test_name = "Mann-Whitney U test"

        # Rank-biserial correlation
        n1, n2 = len(values_a), len(values_b)
        effect_size = 1 - (2 * statistic) / (n1 * n2)

    else:
        raise ValueError(f"Unknown test: {test}")

    significant = p_value < alpha
    mean_a, mean_b = np.mean(values_a), np.mean(values_b)
    direction = "A > B" if mean_a > mean_b else "A < B"

    if significant:
        interpretation = f"Significant difference ({direction}), p={p_value:.4f}"
    else:
        interpretation = f"No significant difference, p={p_value:.4f}"

    return StatisticalResult(
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        effect_size=effect_size,
        interpretation=interpretation,
    )


def aggregate_seeds(
    results: List[Dict[str, float]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results across multiple random seeds.

    Args:
        results: List of result dictionaries from different seeds
        metrics: Specific metrics to aggregate (default: all)

    Returns:
        Dictionary with mean, std, min, max, and CI for each metric
    """
    if not results:
        return {}

    if metrics is None:
        metrics = list(results[0].keys())

    aggregated = {}

    for metric in metrics:
        values = [r.get(metric) for r in results if r.get(metric) is not None]

        if not values:
            continue

        values = np.array(values)
        mean, ci_lower, ci_upper = compute_confidence_interval(values)

        aggregated[metric] = {
            "mean": mean,
            "std": np.std(values, ddof=1) if len(values) > 1 else 0.0,
            "min": np.min(values),
            "max": np.max(values),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_seeds": len(values),
        }

    return aggregated


def compare_models(
    model_results: Dict[str, List[Dict[str, float]]],
    metric: str,
    baseline: str,
    alpha: float = 0.05,
) -> Dict[str, StatisticalResult]:
    """
    Compare multiple models against a baseline.

    Args:
        model_results: Dictionary mapping model name to list of results per seed
        metric: Which metric to compare
        baseline: Name of baseline model
        alpha: Significance level

    Returns:
        Dictionary mapping model name to StatisticalResult vs baseline
    """
    if baseline not in model_results:
        raise ValueError(f"Baseline '{baseline}' not in results")

    baseline_values = [r.get(metric) for r in model_results[baseline] if r.get(metric) is not None]

    comparisons = {}

    for model_name, results in model_results.items():
        if model_name == baseline:
            continue

        model_values = [r.get(metric) for r in results if r.get(metric) is not None]

        if len(model_values) != len(baseline_values):
            # Unpaired comparison
            comparison = unpaired_significance_test(model_values, baseline_values, alpha)
        else:
            # Paired comparison (same seeds)
            comparison = paired_significance_test(model_values, baseline_values, alpha)

        comparisons[model_name] = comparison

    return comparisons


def multiple_comparison_correction(
    p_values: List[float],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Apply multiple comparison correction to p-values.

    Args:
        p_values: List of p-values
        method: Correction method - 'bonferroni', 'holm', 'fdr_bh'
        alpha: Significance level

    Returns:
        Tuple of (adjusted_p_values, significant_flags)
    """
    n = len(p_values)

    if method == "bonferroni":
        adjusted = [min(p * n, 1.0) for p in p_values]
        significant = [p < alpha for p in adjusted]

    elif method == "holm":
        # Holm-Bonferroni step-down
        sorted_idx = np.argsort(p_values)
        adjusted = [0.0] * n

        for rank, idx in enumerate(sorted_idx):
            adjusted[idx] = min(p_values[idx] * (n - rank), 1.0)

        # Ensure monotonicity
        for i in range(1, n):
            idx = sorted_idx[i]
            prev_idx = sorted_idx[i - 1]
            adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])

        significant = [p < alpha for p in adjusted]

    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR control
        sorted_idx = np.argsort(p_values)
        adjusted = [0.0] * n

        for rank, idx in enumerate(sorted_idx):
            adjusted[idx] = p_values[idx] * n / (rank + 1)

        # Ensure monotonicity (reverse)
        for i in range(n - 2, -1, -1):
            idx = sorted_idx[i]
            next_idx = sorted_idx[i + 1]
            adjusted[idx] = min(adjusted[idx], adjusted[next_idx])

        adjusted = [min(p, 1.0) for p in adjusted]
        significant = [p < alpha for p in adjusted]

    else:
        raise ValueError(f"Unknown method: {method}")

    return adjusted, significant


def effect_size_interpretation(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value

    Returns:
        Interpretation string
    """
    d = abs(d)

    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def create_results_table(
    model_results: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str],
    include_ci: bool = True,
    highlight_best: bool = True,
) -> str:
    """
    Create a formatted results table for paper.

    Args:
        model_results: Nested dict: model -> metric -> {mean, std, ci_lower, ci_upper}
        metrics: List of metrics to include
        include_ci: Include confidence intervals
        highlight_best: Highlight best values

    Returns:
        Formatted markdown table string
    """
    # Find best values for each metric
    best = {}
    for metric in metrics:
        values = []
        for model, results in model_results.items():
            if metric in results:
                values.append((model, results[metric].get("mean", 0)))

        if values:
            # Lower is better for loss/perplexity, higher for others
            is_lower_better = "loss" in metric.lower() or "perplexity" in metric.lower()
            best_model = min(values, key=lambda x: x[1]) if is_lower_better else max(values, key=lambda x: x[1])
            best[metric] = best_model[0]

    # Build table
    header = "| Model | " + " | ".join(metrics) + " |"
    separator = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"

    rows = [header, separator]

    for model, results in model_results.items():
        row = f"| {model} |"

        for metric in metrics:
            if metric not in results:
                row += " - |"
                continue

            mean = results[metric].get("mean", 0)
            std = results[metric].get("std", 0)

            cell = f" {mean:.3f}"
            if std > 0:
                cell += f" ± {std:.3f}"

            if include_ci and "ci_lower" in results[metric]:
                ci_l = results[metric]["ci_lower"]
                ci_u = results[metric]["ci_upper"]
                cell += f" [{ci_l:.3f}, {ci_u:.3f}]"

            if highlight_best and best.get(metric) == model:
                cell = f" **{cell.strip()}**"

            row += cell + " |"

        rows.append(row)

    return "\n".join(rows)
