"""
Statistical Analysis Module for Econometric Research.

This module provides comprehensive tools for statistical inference including:
- Hypothesis testing (t-tests, F-tests, chi-squared)
- Effect size calculations (Cohen's d, odds ratios, etc.)
- Power analysis and sample size determination
- Multiple testing corrections

Integration with CausalInput/CausalOutput for seamless workflow.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from scipy import stats

# Import from shared lib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'lib' / 'python'))
from data_loader import CausalInput, CausalOutput


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StatisticalResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    df: Optional[float] = None
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    effect_ci: Optional[Tuple[float, float]] = None
    group_stats: Optional[Dict[str, Dict[str, float]]] = None
    interpretation: str = ""

    def __repr__(self) -> str:
        stars = ""
        if self.p_value < 0.001:
            stars = "***"
        elif self.p_value < 0.01:
            stars = "**"
        elif self.p_value < 0.05:
            stars = "*"
        elif self.p_value < 0.10:
            stars = "."

        return (
            f"StatisticalResult(\n"
            f"  test={self.test_name},\n"
            f"  statistic={self.statistic:.4f},\n"
            f"  p_value={self.p_value:.4f}{stars},\n"
            f"  effect_size={self.effect_size or 'N/A'}\n"
            f")"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'df': self.df,
            'effect_size': self.effect_size,
            'effect_size_name': self.effect_size_name,
            'effect_ci': self.effect_ci,
            'group_stats': self.group_stats,
            'interpretation': self.interpretation
        }


@dataclass
class EffectSizeResult:
    """Result of effect size calculation."""
    value: float
    effect_size_name: str
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    se: Optional[float] = None
    variance_explained: Optional[float] = None
    interpretation: str = ""
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'effect_size_name': self.effect_size_name,
            'value': self.value,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'se': self.se,
            'variance_explained': self.variance_explained,
            'interpretation': self.interpretation
        }


@dataclass
class AssumptionResult:
    """Result of assumption test."""
    test_name: str
    statistic: float
    p_value: float
    passed: bool
    threshold: float = 0.05
    interpretation: str = ""


# =============================================================================
# Hypothesis Tests - T-Tests
# =============================================================================

def run_ttest(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    alternative: str = 'two-sided',
    equal_var: bool = True
) -> StatisticalResult:
    """
    Run two-sample t-test.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable name
    group : str
        Grouping variable (binary)
    alternative : str
        'two-sided', 'greater', or 'less'
    equal_var : bool
        Whether to assume equal variances

    Returns
    -------
    StatisticalResult
        Test results with statistics and effect size
    """
    groups = data[group].unique()
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}")

    g1 = data[data[group] == groups[0]][outcome].dropna()
    g2 = data[data[group] == groups[1]][outcome].dropna()

    # Run t-test
    if equal_var:
        t_stat, p_value = stats.ttest_ind(g1, g2, alternative=alternative)
        test_name = "Independent Samples t-Test"
        df = len(g1) + len(g2) - 2
    else:
        t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False, alternative=alternative)
        test_name = "Welch's t-Test"
        # Welch-Satterthwaite degrees of freedom
        v1, v2 = g1.var(), g2.var()
        n1, n2 = len(g1), len(g2)
        df = ((v1/n1 + v2/n2)**2) / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))

    # Calculate Cohen's d
    pooled_std = np.sqrt(((len(g1)-1)*g1.var() + (len(g2)-1)*g2.var()) / (len(g1)+len(g2)-2))
    cohens_d = (g1.mean() - g2.mean()) / pooled_std

    # Effect size CI
    se_d = np.sqrt((len(g1)+len(g2))/(len(g1)*len(g2)) + cohens_d**2/(2*(len(g1)+len(g2))))
    d_ci = (cohens_d - 1.96*se_d, cohens_d + 1.96*se_d)

    # Group statistics
    group_stats = {
        str(groups[0]): {'n': len(g1), 'mean': g1.mean(), 'std': g1.std()},
        str(groups[1]): {'n': len(g2), 'mean': g2.mean(), 'std': g2.std()}
    }

    # Interpretation
    sig_str = "significant" if p_value < 0.05 else "not significant"
    interp = (
        f"The difference between groups is {sig_str} (t({df:.1f}) = {t_stat:.3f}, p = {p_value:.4f}). "
        f"Cohen's d = {cohens_d:.3f}, indicating a "
        f"{'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} effect."
    )

    return StatisticalResult(
        test_name=test_name,
        statistic=t_stat,
        p_value=p_value,
        df=df,
        effect_size=cohens_d,
        effect_size_name="Cohen's d",
        effect_ci=d_ci,
        group_stats=group_stats,
        interpretation=interp
    )


def run_welch_test(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    alternative: str = 'two-sided'
) -> StatisticalResult:
    """Run Welch's t-test (does not assume equal variances)."""
    return run_ttest(data, outcome, group, alternative, equal_var=False)


def run_ttest_one_sample(
    data: Union[pd.Series, np.ndarray],
    hypothesized_mean: float,
    alternative: str = 'two-sided'
) -> StatisticalResult:
    """
    Run one-sample t-test.

    Parameters
    ----------
    data : array-like
        Sample data
    hypothesized_mean : float
        Value to test against
    alternative : str
        'two-sided', 'greater', or 'less'
    """
    if isinstance(data, pd.Series):
        data = data.dropna().values

    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    t_stat, p_value = stats.ttest_1samp(data, hypothesized_mean, alternative=alternative)
    df = n - 1

    # Effect size
    cohens_d = (mean - hypothesized_mean) / std

    interp = (
        f"The sample mean ({mean:.3f}) is "
        f"{'significantly' if p_value < 0.05 else 'not significantly'} "
        f"different from {hypothesized_mean} (t({df}) = {t_stat:.3f}, p = {p_value:.4f})."
    )

    return StatisticalResult(
        test_name="One-Sample t-Test",
        statistic=t_stat,
        p_value=p_value,
        df=df,
        effect_size=cohens_d,
        effect_size_name="Cohen's d",
        group_stats={'sample': {'n': n, 'mean': mean, 'std': std}},
        interpretation=interp
    )


def run_ttest_paired(
    data: pd.DataFrame,
    var1: str,
    var2: str,
    alternative: str = 'two-sided'
) -> StatisticalResult:
    """
    Run paired samples t-test.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with both variables
    var1, var2 : str
        Variable names to compare
    """
    x1 = data[var1].dropna()
    x2 = data[var2].dropna()

    # Get common indices
    common_idx = x1.index.intersection(x2.index)
    x1 = x1.loc[common_idx]
    x2 = x2.loc[common_idx]

    diff = x1 - x2
    n = len(diff)

    t_stat, p_value = stats.ttest_rel(x1, x2, alternative=alternative)
    df = n - 1

    # Effect size (Cohen's d for paired data)
    cohens_d = diff.mean() / diff.std()

    interp = (
        f"The mean difference ({diff.mean():.3f}) is "
        f"{'significantly' if p_value < 0.05 else 'not significantly'} "
        f"different from zero (t({df}) = {t_stat:.3f}, p = {p_value:.4f})."
    )

    return StatisticalResult(
        test_name="Paired Samples t-Test",
        statistic=t_stat,
        p_value=p_value,
        df=df,
        effect_size=cohens_d,
        effect_size_name="Cohen's d (paired)",
        group_stats={
            var1: {'n': n, 'mean': x1.mean(), 'std': x1.std()},
            var2: {'n': n, 'mean': x2.mean(), 'std': x2.std()},
            'difference': {'mean': diff.mean(), 'std': diff.std()}
        },
        interpretation=interp
    )


# =============================================================================
# Hypothesis Tests - Non-Parametric
# =============================================================================

def run_mann_whitney(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    alternative: str = 'two-sided'
) -> StatisticalResult:
    """
    Run Mann-Whitney U test (non-parametric alternative to t-test).

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable name
    group : str
        Grouping variable (binary)
    """
    groups = data[group].unique()
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}")

    g1 = data[data[group] == groups[0]][outcome].dropna()
    g2 = data[data[group] == groups[1]][outcome].dropna()

    u_stat, p_value = stats.mannwhitneyu(g1, g2, alternative=alternative)

    # Calculate rank-biserial correlation as effect size
    n1, n2 = len(g1), len(g2)
    r = 1 - (2*u_stat)/(n1*n2)

    interp = (
        f"Mann-Whitney U = {u_stat:.1f}, p = {p_value:.4f}. "
        f"Rank-biserial correlation r = {r:.3f}."
    )

    return StatisticalResult(
        test_name="Mann-Whitney U Test",
        statistic=u_stat,
        p_value=p_value,
        effect_size=r,
        effect_size_name="Rank-biserial r",
        group_stats={
            str(groups[0]): {'n': n1, 'median': g1.median()},
            str(groups[1]): {'n': n2, 'median': g2.median()}
        },
        interpretation=interp
    )


def run_wilcoxon(
    data: pd.DataFrame,
    var1: str,
    var2: str,
    alternative: str = 'two-sided'
) -> StatisticalResult:
    """Run Wilcoxon signed-rank test (non-parametric paired test)."""
    x1 = data[var1].dropna()
    x2 = data[var2].dropna()

    common_idx = x1.index.intersection(x2.index)
    x1 = x1.loc[common_idx]
    x2 = x2.loc[common_idx]

    stat, p_value = stats.wilcoxon(x1, x2, alternative=alternative)

    return StatisticalResult(
        test_name="Wilcoxon Signed-Rank Test",
        statistic=stat,
        p_value=p_value,
        interpretation=f"Wilcoxon statistic = {stat:.1f}, p = {p_value:.4f}"
    )


# =============================================================================
# Hypothesis Tests - ANOVA
# =============================================================================

def run_anova(
    data: pd.DataFrame,
    outcome: str,
    groups: str
) -> StatisticalResult:
    """
    Run one-way ANOVA.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable name
    groups : str
        Grouping variable
    """
    group_levels = data[groups].unique()
    group_data = [data[data[groups] == g][outcome].dropna() for g in group_levels]

    f_stat, p_value = stats.f_oneway(*group_data)

    # Degrees of freedom
    k = len(group_levels)
    n = sum(len(g) for g in group_data)
    df_between = k - 1
    df_within = n - k

    # Calculate eta-squared
    ss_total = sum((data[outcome] - data[outcome].mean())**2)
    grand_mean = data[outcome].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_data)
    eta_squared = ss_between / ss_total

    # Group statistics
    group_stats = {}
    for g, d in zip(group_levels, group_data):
        group_stats[str(g)] = {'n': len(d), 'mean': d.mean(), 'std': d.std()}

    interp = (
        f"F({df_between}, {df_within}) = {f_stat:.3f}, p = {p_value:.4f}. "
        f"Eta-squared = {eta_squared:.3f}, indicating "
        f"{eta_squared*100:.1f}% of variance explained by group membership."
    )

    return StatisticalResult(
        test_name="One-Way ANOVA",
        statistic=f_stat,
        p_value=p_value,
        df=df_between,
        effect_size=eta_squared,
        effect_size_name="Eta-squared",
        group_stats=group_stats,
        interpretation=interp
    )


def run_welch_anova(
    data: pd.DataFrame,
    outcome: str,
    groups: str
) -> StatisticalResult:
    """Run Welch's ANOVA (does not assume equal variances)."""
    try:
        from scipy.stats import alexandergovern
        group_levels = data[groups].unique()
        group_data = [data[data[groups] == g][outcome].dropna().values for g in group_levels]

        result = alexandergovern(*group_data)
        return StatisticalResult(
            test_name="Alexander-Govern Test (Welch ANOVA)",
            statistic=result.statistic,
            p_value=result.pvalue,
            interpretation=f"Test statistic = {result.statistic:.3f}, p = {result.pvalue:.4f}"
        )
    except ImportError:
        warnings.warn("Alexander-Govern test not available in this scipy version. Running standard ANOVA.")
        return run_anova(data, outcome, groups)


def run_kruskal(
    data: pd.DataFrame,
    outcome: str,
    groups: str
) -> StatisticalResult:
    """Run Kruskal-Wallis test (non-parametric ANOVA)."""
    group_levels = data[groups].unique()
    group_data = [data[data[groups] == g][outcome].dropna() for g in group_levels]

    h_stat, p_value = stats.kruskal(*group_data)

    # Epsilon-squared effect size
    n = sum(len(g) for g in group_data)
    k = len(group_levels)
    epsilon_squared = (h_stat - k + 1) / (n - k)

    return StatisticalResult(
        test_name="Kruskal-Wallis Test",
        statistic=h_stat,
        p_value=p_value,
        effect_size=epsilon_squared,
        effect_size_name="Epsilon-squared",
        interpretation=f"H = {h_stat:.3f}, p = {p_value:.4f}"
    )


# =============================================================================
# Hypothesis Tests - Chi-Squared
# =============================================================================

def run_chi_squared(
    data: pd.DataFrame,
    var1: str,
    var2: str
) -> StatisticalResult:
    """
    Run chi-squared test of independence.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    var1, var2 : str
        Categorical variable names
    """
    contingency = pd.crosstab(data[var1], data[var2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Calculate Cramer's V
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    interp = (
        f"Chi-squared({dof}) = {chi2:.3f}, p = {p_value:.4f}. "
        f"Cramer's V = {cramers_v:.3f}, indicating a "
        f"{'weak' if cramers_v < 0.2 else 'moderate' if cramers_v < 0.4 else 'strong'} association."
    )

    return StatisticalResult(
        test_name="Chi-Squared Test of Independence",
        statistic=chi2,
        p_value=p_value,
        df=dof,
        effect_size=cramers_v,
        effect_size_name="Cramer's V",
        interpretation=interp
    )


def run_chi_squared_gof(
    observed: Union[List[float], np.ndarray],
    expected: Union[List[float], np.ndarray] = None
) -> StatisticalResult:
    """Run chi-squared goodness of fit test."""
    observed = np.array(observed)

    if expected is None:
        expected = np.ones_like(observed) * observed.sum() / len(observed)
    else:
        expected = np.array(expected)
        if expected.sum() != observed.sum():
            expected = expected * observed.sum() / expected.sum()

    chi2, p_value = stats.chisquare(observed, expected)
    dof = len(observed) - 1

    return StatisticalResult(
        test_name="Chi-Squared Goodness of Fit",
        statistic=chi2,
        p_value=p_value,
        df=dof,
        interpretation=f"Chi-squared({dof}) = {chi2:.3f}, p = {p_value:.4f}"
    )


def run_fisher_exact(
    data: pd.DataFrame,
    var1: str,
    var2: str
) -> StatisticalResult:
    """Run Fisher's exact test for 2x2 contingency table."""
    contingency = pd.crosstab(data[var1], data[var2])

    if contingency.shape != (2, 2):
        raise ValueError("Fisher's exact test requires a 2x2 table")

    odds_ratio, p_value = stats.fisher_exact(contingency)

    return StatisticalResult(
        test_name="Fisher's Exact Test",
        statistic=odds_ratio,
        p_value=p_value,
        effect_size=odds_ratio,
        effect_size_name="Odds Ratio",
        interpretation=f"Odds Ratio = {odds_ratio:.3f}, p = {p_value:.4f}"
    )


# =============================================================================
# Effect Size Calculations
# =============================================================================

def calculate_cohens_d(
    data: pd.DataFrame,
    outcome: str,
    group: str
) -> EffectSizeResult:
    """
    Calculate Cohen's d for two-group comparison.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable name
    group : str
        Grouping variable (binary)
    """
    groups = data[group].unique()
    g1 = data[data[group] == groups[0]][outcome].dropna()
    g2 = data[data[group] == groups[1]][outcome].dropna()

    n1, n2 = len(g1), len(g2)
    m1, m2 = g1.mean(), g2.mean()
    s1, s2 = g1.std(), g2.std()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))

    d = (m1 - m2) / pooled_std

    # Standard error and CI
    se = np.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2)))
    ci_lower = d - 1.96 * se
    ci_upper = d + 1.96 * se

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible effect"
    elif abs_d < 0.5:
        interp = "small effect"
    elif abs_d < 0.8:
        interp = "medium effect"
    else:
        interp = "large effect"

    return EffectSizeResult(
        value=d,
        effect_size_name="Cohen's d",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        se=se,
        variance_explained=d**2 / (d**2 + 4),  # Approximate r-squared
        interpretation=f"Cohen's d = {d:.3f}: {interp}",
        details={
            'mean_diff': m1 - m2,
            'pooled_std': pooled_std,
            'n1': n1,
            'n2': n2
        }
    )


def calculate_hedges_g(
    data: pd.DataFrame,
    outcome: str,
    group: str
) -> EffectSizeResult:
    """Calculate Hedges' g (bias-corrected Cohen's d)."""
    d_result = calculate_cohens_d(data, outcome, group)

    n1 = d_result.details['n1']
    n2 = d_result.details['n2']

    # Correction factor
    J = 1 - 3 / (4*(n1+n2) - 9)
    g = d_result.value * J

    se = d_result.se * J
    ci_lower = g - 1.96 * se
    ci_upper = g + 1.96 * se

    return EffectSizeResult(
        value=g,
        effect_size_name="Hedges' g",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        se=se,
        interpretation=f"Hedges' g = {g:.3f} (bias-corrected Cohen's d)",
        details={**d_result.details, 'correction_factor': J}
    )


def calculate_glass_delta(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    control_value: Any = 0
) -> EffectSizeResult:
    """Calculate Glass's delta (uses control group SD only)."""
    groups = data[group].unique()

    # Identify control group
    if control_value in groups:
        control_group = control_value
        treatment_group = [g for g in groups if g != control_value][0]
    else:
        control_group = groups[0]
        treatment_group = groups[1]

    control_data = data[data[group] == control_group][outcome].dropna()
    treatment_data = data[data[group] == treatment_group][outcome].dropna()

    delta = (treatment_data.mean() - control_data.mean()) / control_data.std()

    return EffectSizeResult(
        value=delta,
        effect_size_name="Glass's delta",
        interpretation=f"Glass's delta = {delta:.3f} (uses control SD only)",
        details={
            'control_mean': control_data.mean(),
            'treatment_mean': treatment_data.mean(),
            'control_std': control_data.std()
        }
    )


def calculate_correlation(
    data: pd.DataFrame,
    var1: str,
    var2: str,
    method: str = 'pearson'
) -> EffectSizeResult:
    """Calculate correlation coefficient with CI."""
    x = data[var1].dropna()
    y = data[var2].dropna()

    common_idx = x.index.intersection(y.index)
    x = x.loc[common_idx]
    y = y.loc[common_idx]

    if method == 'pearson':
        r, p = stats.pearsonr(x, y)
    elif method == 'spearman':
        r, p = stats.spearmanr(x, y)
    elif method == 'kendall':
        r, p = stats.kendalltau(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Fisher's z transformation for CI
    n = len(x)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_ci = (z - 1.96*se, z + 1.96*se)
    r_ci = (np.tanh(z_ci[0]), np.tanh(z_ci[1]))

    return EffectSizeResult(
        value=r,
        effect_size_name=f"{method.capitalize()} r",
        ci_lower=r_ci[0],
        ci_upper=r_ci[1],
        se=se,
        variance_explained=r**2,
        interpretation=f"{method.capitalize()} r = {r:.3f}, r-squared = {r**2:.3f}",
        details={'n': n, 'p_value': p}
    )


def calculate_point_biserial(
    data: pd.DataFrame,
    continuous: str,
    binary: str
) -> EffectSizeResult:
    """Calculate point-biserial correlation."""
    return calculate_correlation(data, continuous, binary, method='pearson')


def calculate_odds_ratio(
    data: pd.DataFrame,
    outcome: str,
    exposure: str
) -> EffectSizeResult:
    """
    Calculate odds ratio for 2x2 table.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Binary outcome variable (0/1)
    exposure : str
        Binary exposure/treatment variable (0/1)
    """
    contingency = pd.crosstab(data[exposure], data[outcome])

    if contingency.shape != (2, 2):
        raise ValueError("Both variables must be binary")

    a = contingency.iloc[1, 1]  # Exposed, Outcome+
    b = contingency.iloc[1, 0]  # Exposed, Outcome-
    c = contingency.iloc[0, 1]  # Unexposed, Outcome+
    d = contingency.iloc[0, 0]  # Unexposed, Outcome-

    # Handle zeros
    if 0 in [a, b, c, d]:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    odds_ratio = (a * d) / (b * c)

    # CI using log transformation
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    log_or = np.log(odds_ratio)
    ci_lower = np.exp(log_or - 1.96 * se_log_or)
    ci_upper = np.exp(log_or + 1.96 * se_log_or)

    if odds_ratio > 1:
        interp = f"Exposure increases odds of outcome by {(odds_ratio-1)*100:.1f}%"
    else:
        interp = f"Exposure decreases odds of outcome by {(1-odds_ratio)*100:.1f}%"

    return EffectSizeResult(
        value=odds_ratio,
        effect_size_name="Odds Ratio",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        se=se_log_or,
        interpretation=f"OR = {odds_ratio:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]. {interp}",
        details={'a': a, 'b': b, 'c': c, 'd': d}
    )


def calculate_relative_risk(
    data: pd.DataFrame,
    outcome: str,
    exposure: str
) -> EffectSizeResult:
    """Calculate relative risk (risk ratio)."""
    contingency = pd.crosstab(data[exposure], data[outcome])

    if contingency.shape != (2, 2):
        raise ValueError("Both variables must be binary")

    a = contingency.iloc[1, 1]  # Exposed, Outcome+
    b = contingency.iloc[1, 0]  # Exposed, Outcome-
    c = contingency.iloc[0, 1]  # Unexposed, Outcome+
    d = contingency.iloc[0, 0]  # Unexposed, Outcome-

    risk_exposed = a / (a + b)
    risk_unexposed = c / (c + d)

    rr = risk_exposed / risk_unexposed

    # CI using log transformation
    se_log_rr = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    ci_lower = np.exp(np.log(rr) - 1.96 * se_log_rr)
    ci_upper = np.exp(np.log(rr) + 1.96 * se_log_rr)

    return EffectSizeResult(
        value=rr,
        effect_size_name="Relative Risk",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        interpretation=f"RR = {rr:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]",
        details={'risk_exposed': risk_exposed, 'risk_unexposed': risk_unexposed}
    )


def calculate_risk_difference(
    data: pd.DataFrame,
    outcome: str,
    exposure: str
) -> EffectSizeResult:
    """Calculate risk difference (attributable risk)."""
    contingency = pd.crosstab(data[exposure], data[outcome])

    a = contingency.iloc[1, 1]
    b = contingency.iloc[1, 0]
    c = contingency.iloc[0, 1]
    d = contingency.iloc[0, 0]

    risk_exposed = a / (a + b)
    risk_unexposed = c / (c + d)

    rd = risk_exposed - risk_unexposed

    # SE and CI
    se = np.sqrt(risk_exposed*(1-risk_exposed)/(a+b) + risk_unexposed*(1-risk_unexposed)/(c+d))
    ci_lower = rd - 1.96 * se
    ci_upper = rd + 1.96 * se

    return EffectSizeResult(
        value=rd,
        effect_size_name="Risk Difference",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        se=se,
        interpretation=f"RD = {rd:.4f} ({rd*100:.2f} percentage points)"
    )


def calculate_nnt(risk_difference: float) -> float:
    """Calculate Number Needed to Treat from risk difference."""
    if risk_difference == 0:
        return np.inf
    return 1 / abs(risk_difference)


def calculate_phi(
    data: pd.DataFrame,
    var1: str,
    var2: str
) -> EffectSizeResult:
    """Calculate phi coefficient for 2x2 table."""
    contingency = pd.crosstab(data[var1], data[var2])

    if contingency.shape != (2, 2):
        raise ValueError("Phi requires a 2x2 table")

    a, b = contingency.iloc[0, 0], contingency.iloc[0, 1]
    c, d = contingency.iloc[1, 0], contingency.iloc[1, 1]

    phi = (a*d - b*c) / np.sqrt((a+b)*(c+d)*(a+c)*(b+d))

    return EffectSizeResult(
        value=phi,
        effect_size_name="Phi coefficient",
        interpretation=f"Phi = {phi:.3f}"
    )


def calculate_cramers_v(
    data: pd.DataFrame,
    var1: str,
    var2: str
) -> EffectSizeResult:
    """Calculate Cramer's V for contingency table."""
    contingency = pd.crosstab(data[var1], data[var2])
    chi2, _, _, _ = stats.chi2_contingency(contingency)

    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1

    v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    return EffectSizeResult(
        value=v,
        effect_size_name="Cramer's V",
        interpretation=f"Cramer's V = {v:.3f}"
    )


def calculate_eta_squared(anova_result: StatisticalResult) -> EffectSizeResult:
    """Extract eta-squared from ANOVA result."""
    return EffectSizeResult(
        value=anova_result.effect_size,
        effect_size_name="Eta-squared",
        interpretation=f"Eta-squared = {anova_result.effect_size:.3f}"
    )


def calculate_omega_squared(
    ss_between: float,
    ss_within: float,
    ss_total: float,
    k: int,
    n: int
) -> EffectSizeResult:
    """Calculate omega-squared (less biased than eta-squared)."""
    ms_within = ss_within / (n - k)
    omega_sq = (ss_between - (k-1)*ms_within) / (ss_total + ms_within)

    return EffectSizeResult(
        value=omega_sq,
        effect_size_name="Omega-squared",
        interpretation=f"Omega-squared = {omega_sq:.3f} (less biased estimate)"
    )


def calculate_cohens_f(eta_squared: float) -> EffectSizeResult:
    """Calculate Cohen's f from eta-squared."""
    f = np.sqrt(eta_squared / (1 - eta_squared))

    return EffectSizeResult(
        value=f,
        effect_size_name="Cohen's f",
        interpretation=f"Cohen's f = {f:.3f}"
    )


def convert_effect_size(
    value: float,
    from_type: str,
    to_type: str
) -> float:
    """
    Convert between effect size types.

    Supports: d, r, or, eta_sq, f
    """
    # First convert to d
    if from_type == 'd':
        d = value
    elif from_type == 'r':
        d = 2 * value / np.sqrt(1 - value**2)
    elif from_type == 'or':
        d = np.log(value) * np.sqrt(3) / np.pi
    elif from_type == 'eta_sq':
        d = 2 * np.sqrt(value / (1 - value))
    elif from_type == 'f':
        d = 2 * value
    else:
        raise ValueError(f"Unknown from_type: {from_type}")

    # Then convert d to target
    if to_type == 'd':
        return d
    elif to_type == 'r':
        return d / np.sqrt(d**2 + 4)
    elif to_type == 'or':
        return np.exp(d * np.pi / np.sqrt(3))
    elif to_type == 'eta_sq':
        return d**2 / (d**2 + 4)
    elif to_type == 'f':
        return d / 2
    else:
        raise ValueError(f"Unknown to_type: {to_type}")


def standardize_coefficient(
    unstandardized_coef: float,
    sd_x: float,
    sd_y: float
) -> float:
    """Convert unstandardized to standardized regression coefficient."""
    return unstandardized_coef * (sd_x / sd_y)


# =============================================================================
# Multiple Testing Corrections
# =============================================================================

def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[float]:
    """Apply Bonferroni correction to p-values."""
    m = len(p_values)
    return [min(p * m, 1.0) for p in p_values]


def holm_correction(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni (step-down) correction."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted = np.zeros(m)
    for i, p in enumerate(sorted_p):
        adjusted[sorted_indices[i]] = min(p * (m - i), 1.0)

    # Enforce monotonicity
    for i in range(1, m):
        if adjusted[sorted_indices[i]] < adjusted[sorted_indices[i-1]]:
            adjusted[sorted_indices[i]] = adjusted[sorted_indices[i-1]]

    return adjusted.tolist()


def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05
) -> List[float]:
    """Apply Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted = np.zeros(m)
    for i, p in enumerate(sorted_p):
        adjusted[sorted_indices[i]] = min(p * m / (i + 1), 1.0)

    # Enforce monotonicity (from end to start)
    for i in range(m-2, -1, -1):
        if adjusted[sorted_indices[i]] > adjusted[sorted_indices[i+1]]:
            adjusted[sorted_indices[i]] = adjusted[sorted_indices[i+1]]

    return adjusted.tolist()


def adjust_pvalues(
    p_values: List[float],
    method: str = 'holm',
    alpha: float = 0.05
) -> List[float]:
    """
    Adjust p-values for multiple comparisons.

    Parameters
    ----------
    p_values : list
        List of p-values
    method : str
        'bonferroni', 'holm', 'bh' (Benjamini-Hochberg)
    alpha : float
        Significance level (used for some methods)
    """
    if method == 'bonferroni':
        return bonferroni_correction(p_values, alpha)
    elif method == 'holm':
        return holm_correction(p_values)
    elif method in ['bh', 'fdr', 'benjamini_hochberg']:
        return benjamini_hochberg(p_values, alpha)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Assumption Checking
# =============================================================================

def check_normality(
    data: Union[pd.Series, np.ndarray],
    method: str = 'shapiro'
) -> AssumptionResult:
    """
    Test normality assumption.

    Parameters
    ----------
    data : array-like
        Data to test
    method : str
        'shapiro', 'ks', 'dagostino', 'anderson'
    """
    if isinstance(data, pd.Series):
        data = data.dropna().values

    # Limit sample size for computational reasons
    if len(data) > 5000:
        data = np.random.choice(data, 5000, replace=False)

    if method == 'shapiro':
        stat, p_value = stats.shapiro(data)
        test_name = "Shapiro-Wilk"
    elif method == 'ks':
        stat, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        test_name = "Kolmogorov-Smirnov"
    elif method == 'dagostino':
        stat, p_value = stats.normaltest(data)
        test_name = "D'Agostino-Pearson"
    else:
        raise ValueError(f"Unknown method: {method}")

    passed = p_value > 0.05
    interp = "Data appears normally distributed" if passed else "Data deviates from normal distribution"

    return AssumptionResult(
        test_name=f"{test_name} Normality Test",
        statistic=stat,
        p_value=p_value,
        passed=passed,
        interpretation=interp
    )


def check_homogeneity(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    method: str = 'levene'
) -> AssumptionResult:
    """
    Test homogeneity of variance assumption.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable
    group : str
        Grouping variable
    method : str
        'levene', 'bartlett', 'brown_forsythe'
    """
    groups = data[group].unique()
    group_data = [data[data[group] == g][outcome].dropna() for g in groups]

    if method == 'levene':
        stat, p_value = stats.levene(*group_data)
        test_name = "Levene's Test"
    elif method == 'bartlett':
        stat, p_value = stats.bartlett(*group_data)
        test_name = "Bartlett's Test"
    elif method == 'brown_forsythe':
        stat, p_value = stats.levene(*group_data, center='median')
        test_name = "Brown-Forsythe Test"
    else:
        raise ValueError(f"Unknown method: {method}")

    passed = p_value > 0.05
    interp = "Variances appear homogeneous" if passed else "Evidence of heterogeneous variances"

    return AssumptionResult(
        test_name=test_name,
        statistic=stat,
        p_value=p_value,
        passed=passed,
        interpretation=interp
    )


def durbin_watson_test(residuals: np.ndarray) -> AssumptionResult:
    """Test for autocorrelation in residuals."""
    n = len(residuals)
    dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)

    # Rough interpretation (proper bounds depend on n and k)
    if dw < 1.5:
        interp = "Possible positive autocorrelation"
        passed = False
    elif dw > 2.5:
        interp = "Possible negative autocorrelation"
        passed = False
    else:
        interp = "No strong evidence of autocorrelation"
        passed = True

    return AssumptionResult(
        test_name="Durbin-Watson Test",
        statistic=dw,
        p_value=np.nan,  # Not a p-value based test
        passed=passed,
        interpretation=f"DW = {dw:.3f}. {interp}"
    )


# =============================================================================
# Power Analysis
# =============================================================================

def power_analysis(
    effect_size: float,
    n1: int,
    n2: int = None,
    alpha: float = 0.05,
    test_type: str = 'two_sample_ttest'
) -> float:
    """
    Calculate statistical power.

    Parameters
    ----------
    effect_size : float
        Effect size (Cohen's d for t-test)
    n1 : int
        Sample size group 1 (or total for one-sample)
    n2 : int, optional
        Sample size group 2
    alpha : float
        Significance level
    test_type : str
        'one_sample_ttest', 'two_sample_ttest', 'paired_ttest'
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)

    if test_type == 'one_sample_ttest':
        ncp = effect_size * np.sqrt(n1)
    elif test_type == 'two_sample_ttest':
        n2 = n2 or n1
        ncp = effect_size / np.sqrt(1/n1 + 1/n2)
    elif test_type == 'paired_ttest':
        ncp = effect_size * np.sqrt(n1)
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)

    return power


def minimum_detectable_effect(
    n1: int,
    n2: int = None,
    alpha: float = 0.05,
    power: float = 0.80,
    test_type: str = 'two_sample_ttest'
) -> float:
    """
    Calculate minimum detectable effect size.

    Parameters
    ----------
    n1, n2 : int
        Sample sizes
    alpha : float
        Significance level
    power : float
        Desired power
    test_type : str
        'one_sample_ttest', 'two_sample_ttest', 'paired_ttest'
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    if test_type == 'one_sample_ttest':
        mde = (z_alpha + z_beta) / np.sqrt(n1)
    elif test_type == 'two_sample_ttest':
        n2 = n2 or n1
        mde = (z_alpha + z_beta) * np.sqrt(1/n1 + 1/n2)
    elif test_type == 'paired_ttest':
        mde = (z_alpha + z_beta) / np.sqrt(n1)
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    return mde


def required_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    test_type: str = 'two_sample_ttest'
) -> int:
    """
    Calculate required sample size.

    Parameters
    ----------
    effect_size : float
        Expected effect size
    alpha : float
        Significance level
    power : float
        Desired power
    test_type : str
        'one_sample_ttest', 'two_sample_ttest', 'paired_ttest'
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    if test_type in ['one_sample_ttest', 'paired_ttest']:
        n = (z_alpha + z_beta)**2 / effect_size**2
    elif test_type == 'two_sample_ttest':
        n = 2 * (z_alpha + z_beta)**2 / effect_size**2
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    return max(2, int(np.ceil(n)))


# =============================================================================
# Full Analysis Workflow
# =============================================================================

def run_full_statistical_analysis(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    controls: List[str] = None,
    alpha: float = 0.05,
    power_target: float = 0.80,
    effect_size_type: str = 'cohens_d'
) -> CausalOutput:
    """
    Run complete statistical analysis workflow.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable
    group : str
        Grouping variable
    controls : list, optional
        Control variables (not used in basic tests)
    alpha : float
        Significance level
    power_target : float
        Target power for sample size calculations
    effect_size_type : str
        Effect size to calculate

    Returns
    -------
    CausalOutput
        Complete analysis results
    """
    results = {}

    # 1. Check assumptions
    results['normality'] = {}
    for g in data[group].unique():
        subset = data[data[group] == g][outcome].dropna()
        results['normality'][str(g)] = check_normality(subset)

    results['homogeneity'] = check_homogeneity(data, outcome, group)

    # 2. Choose and run appropriate test
    use_parametric = all(r.passed for r in results['normality'].values()) and results['homogeneity'].passed

    if use_parametric:
        test_result = run_welch_test(data, outcome, group)  # Welch's is generally safer
    else:
        test_result = run_mann_whitney(data, outcome, group)
        warnings.warn("Assumptions violated. Using non-parametric test.")

    results['test'] = test_result

    # 3. Calculate effect size
    if effect_size_type == 'cohens_d':
        effect_result = calculate_cohens_d(data, outcome, group)
    elif effect_size_type == 'hedges_g':
        effect_result = calculate_hedges_g(data, outcome, group)
    else:
        effect_result = calculate_cohens_d(data, outcome, group)

    results['effect_size'] = effect_result

    # 4. Power analysis
    n1 = len(data[data[group] == data[group].unique()[0]])
    n2 = len(data[data[group] == data[group].unique()[1]])
    achieved_power = power_analysis(effect_result.value, n1, n2, alpha)
    mde = minimum_detectable_effect(n1, n2, alpha, power_target)

    results['power'] = {
        'achieved': achieved_power,
        'target': power_target,
        'mde': mde
    }

    # Generate summary
    summary_lines = [
        "=" * 60,
        "STATISTICAL ANALYSIS RESULTS",
        "=" * 60,
        "",
        f"Test: {test_result.test_name}",
        f"Statistic: {test_result.statistic:.4f}",
        f"P-value: {test_result.p_value:.6f}",
        "",
        f"Effect Size ({effect_result.effect_size_name}): {effect_result.value:.4f}",
        f"95% CI: [{effect_result.ci_lower:.4f}, {effect_result.ci_upper:.4f}]",
        "",
        f"Power (achieved): {achieved_power:.3f}",
        f"Minimum Detectable Effect: {mde:.4f}",
        "",
        "-" * 60,
        "ASSUMPTIONS",
        "-" * 60,
    ]

    for g, norm_result in results['normality'].items():
        summary_lines.append(f"Normality ({g}): {'PASS' if norm_result.passed else 'FAIL'} (p={norm_result.p_value:.4f})")

    summary_lines.append(f"Homogeneity: {'PASS' if results['homogeneity'].passed else 'FAIL'} (p={results['homogeneity'].p_value:.4f})")
    summary_lines.append("=" * 60)

    summary_table = "\n".join(summary_lines)

    # Interpretation
    if test_result.p_value < alpha:
        sig_str = "statistically significant"
    else:
        sig_str = "not statistically significant"

    interpretation = (
        f"The difference between groups is {sig_str} at alpha = {alpha}. "
        f"{effect_result.interpretation}. "
        f"The study had {'adequate' if achieved_power >= 0.80 else 'inadequate'} power ({achieved_power:.2f})."
    )

    return CausalOutput(
        effect=effect_result.value,
        se=effect_result.se or test_result.statistic,
        ci_lower=effect_result.ci_lower,
        ci_upper=effect_result.ci_upper,
        p_value=test_result.p_value,
        diagnostics=results,
        summary_table=summary_table,
        interpretation=interpretation
    )


# =============================================================================
# Validation
# =============================================================================

def validate_estimator(verbose: bool = True) -> Dict[str, Any]:
    """Validate statistical analysis functions with known data."""

    np.random.seed(42)

    # Generate two groups with known difference
    true_effect = 0.5  # True Cohen's d
    n = 100

    g1 = np.random.normal(0, 1, n)
    g2 = np.random.normal(true_effect, 1, n)

    df = pd.DataFrame({
        'y': np.concatenate([g1, g2]),
        'group': np.repeat([0, 1], n)
    })

    # Run analysis
    result = run_ttest(df, 'y', 'group')
    d_result = calculate_cohens_d(df, 'y', 'group')

    # Check results
    estimated_d = d_result.value
    bias = estimated_d - true_effect
    bias_pct = abs(bias / true_effect) * 100

    validation = {
        'true_effect': true_effect,
        'estimated_effect': estimated_d,
        'bias': bias,
        'bias_pct': bias_pct,
        'p_value': result.p_value,
        'ci_covers_truth': d_result.ci_lower <= true_effect <= d_result.ci_upper,
        'passed': bias_pct < 20  # Allow 20% bias due to sampling
    }

    if verbose:
        print("=" * 50)
        print("STATISTICAL ANALYSIS VALIDATION")
        print("=" * 50)
        print(f"True Cohen's d: {true_effect:.4f}")
        print(f"Estimated Cohen's d: {estimated_d:.4f}")
        print(f"Bias: {bias:.4f} ({bias_pct:.2f}%)")
        print(f"P-value: {result.p_value:.4f}")
        print(f"95% CI: [{d_result.ci_lower:.4f}, {d_result.ci_upper:.4f}]")
        print(f"CI covers truth: {validation['ci_covers_truth']}")
        print("-" * 50)
        print(f"VALIDATION: {'PASSED' if validation['passed'] else 'FAILED'}")
        print("=" * 50)

    return validation


if __name__ == "__main__":
    validate_estimator(verbose=True)
