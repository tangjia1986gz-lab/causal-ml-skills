"""
Shared diagnostic functions for causal inference estimators.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DiagnosticResult:
    """Standard diagnostic test result."""
    test_name: str
    statistic: float
    p_value: float
    passed: bool
    threshold: float
    interpretation: str
    details: Dict[str, Any] = None


def parallel_trends_test(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    time_id: str,
    unit_id: str,
    treatment_time: int,
    n_pre_periods: int = 3
) -> DiagnosticResult:
    """
    Test parallel trends assumption for DID.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable name
    treatment : str
        Treatment indicator variable name
    time_id : str
        Time period variable name
    unit_id : str
        Unit identifier variable name
    treatment_time : int
        Time period when treatment starts
    n_pre_periods : int
        Number of pre-treatment periods to test

    Returns
    -------
    DiagnosticResult
        Test result with statistics and interpretation
    """
    pre_data = data[data[time_id] < treatment_time].copy()

    # Group by treatment status and time
    trends = pre_data.groupby([treatment, time_id])[outcome].mean().unstack(level=0)

    if trends.shape[1] < 2:
        return DiagnosticResult(
            test_name="Parallel Trends Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=False,
            threshold=0.05,
            interpretation="Insufficient data: need both treatment and control groups",
            details={"error": "insufficient_groups"}
        )

    # Calculate difference in trends
    trend_diff = trends.iloc[:, 1] - trends.iloc[:, 0]

    # Test if trend difference is constant (i.e., parallel)
    time_points = np.arange(len(trend_diff))
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, trend_diff.values)

    passed = p_value > 0.05  # Non-significant slope suggests parallel trends

    interpretation = (
        "Parallel trends assumption SUPPORTED: no significant divergence in pre-treatment trends"
        if passed else
        "Parallel trends assumption VIOLATED: significant divergence detected in pre-treatment period"
    )

    return DiagnosticResult(
        test_name="Parallel Trends Test (Linear Divergence)",
        statistic=slope,
        p_value=p_value,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            "slope": slope,
            "std_err": std_err,
            "r_squared": r_value**2,
            "trend_differences": trend_diff.to_dict()
        }
    )


def weak_iv_test(
    first_stage_f: float,
    n_instruments: int = 1
) -> DiagnosticResult:
    """
    Test for weak instruments using first-stage F-statistic.

    Parameters
    ----------
    first_stage_f : float
        F-statistic from first stage regression
    n_instruments : int
        Number of instruments

    Returns
    -------
    DiagnosticResult
        Test result with Stock-Yogo critical values
    """
    # Stock-Yogo critical values for 10% maximal IV size
    # Simplified: using rule of thumb F > 10
    threshold = 10.0
    passed = first_stage_f > threshold

    interpretation = (
        f"Instruments are STRONG: F = {first_stage_f:.2f} > {threshold}"
        if passed else
        f"WEAK INSTRUMENTS WARNING: F = {first_stage_f:.2f} < {threshold}. Consider LIML or other robust methods."
    )

    return DiagnosticResult(
        test_name="Weak Instrument Test (Stock-Yogo)",
        statistic=first_stage_f,
        p_value=np.nan,  # Not a p-value test
        passed=passed,
        threshold=threshold,
        interpretation=interpretation,
        details={
            "n_instruments": n_instruments,
            "rule": "Stock-Yogo rule of thumb: F > 10"
        }
    )


def mccrary_density_test(
    running_variable: np.ndarray,
    cutoff: float = 0,
    n_bins: int = 50
) -> DiagnosticResult:
    """
    McCrary density test for RD manipulation.

    Simplified implementation - tests for discontinuity in density at cutoff.

    Parameters
    ----------
    running_variable : np.ndarray
        The running variable
    cutoff : float
        The RD cutoff value
    n_bins : int
        Number of bins for histogram

    Returns
    -------
    DiagnosticResult
        Test result indicating potential manipulation
    """
    below = running_variable[running_variable < cutoff]
    above = running_variable[running_variable >= cutoff]

    # Simple density comparison near cutoff
    bandwidth = (running_variable.max() - running_variable.min()) / n_bins
    near_below = np.sum((cutoff - bandwidth <= running_variable) & (running_variable < cutoff))
    near_above = np.sum((running_variable >= cutoff) & (running_variable < cutoff + bandwidth))

    # Chi-squared test for equal density
    expected = (near_below + near_above) / 2
    if expected > 0:
        chi_sq = ((near_below - expected)**2 + (near_above - expected)**2) / expected
        p_value = 1 - stats.chi2.cdf(chi_sq, df=1)
    else:
        chi_sq = 0
        p_value = 1.0

    passed = p_value > 0.05

    interpretation = (
        "No evidence of manipulation at cutoff"
        if passed else
        "POTENTIAL MANIPULATION: significant density discontinuity at cutoff"
    )

    return DiagnosticResult(
        test_name="McCrary Density Test (Simplified)",
        statistic=chi_sq,
        p_value=p_value,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            "density_below": near_below,
            "density_above": near_above,
            "bandwidth": bandwidth
        }
    )


def balance_test(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str]
) -> Dict[str, DiagnosticResult]:
    """
    Test covariate balance between treatment and control groups.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    treatment : str
        Treatment indicator (0/1)
    covariates : List[str]
        List of covariate names to test

    Returns
    -------
    Dict[str, DiagnosticResult]
        Balance test results for each covariate
    """
    results = {}
    treated = data[data[treatment] == 1]
    control = data[data[treatment] == 0]

    for cov in covariates:
        t_stat, p_value = stats.ttest_ind(
            treated[cov].dropna(),
            control[cov].dropna()
        )

        # Standardized mean difference
        pooled_std = np.sqrt(
            (treated[cov].var() + control[cov].var()) / 2
        )
        smd = (treated[cov].mean() - control[cov].mean()) / pooled_std if pooled_std > 0 else 0

        passed = abs(smd) < 0.1  # Common threshold for balance

        results[cov] = DiagnosticResult(
            test_name=f"Balance Test: {cov}",
            statistic=smd,
            p_value=p_value,
            passed=passed,
            threshold=0.1,
            interpretation=f"{'Balanced' if passed else 'IMBALANCED'}: SMD = {smd:.3f}",
            details={
                "treated_mean": treated[cov].mean(),
                "control_mean": control[cov].mean(),
                "t_statistic": t_stat
            }
        )

    return results


def overidentification_test(
    residuals: np.ndarray,
    instruments: np.ndarray,
    n_endog: int = 1
) -> DiagnosticResult:
    """
    Sargan-Hansen overidentification test for IV.

    Parameters
    ----------
    residuals : np.ndarray
        2SLS residuals
    instruments : np.ndarray
        Instrument matrix
    n_endog : int
        Number of endogenous variables

    Returns
    -------
    DiagnosticResult
        Overidentification test result
    """
    n = len(residuals)
    n_instruments = instruments.shape[1] if instruments.ndim > 1 else 1

    if n_instruments <= n_endog:
        return DiagnosticResult(
            test_name="Sargan-Hansen Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=True,
            threshold=0.05,
            interpretation="Model is exactly identified - overidentification test not applicable",
            details={"reason": "exact_identification"}
        )

    # Sargan statistic
    if instruments.ndim == 1:
        instruments = instruments.reshape(-1, 1)

    proj = instruments @ np.linalg.lstsq(instruments, residuals, rcond=None)[0]
    sargan_stat = n * (proj @ residuals) / (residuals @ residuals)
    df = n_instruments - n_endog
    p_value = 1 - stats.chi2.cdf(sargan_stat, df=df)

    passed = p_value > 0.05

    interpretation = (
        "Overidentification test PASSED: instruments appear valid"
        if passed else
        "Overidentification test FAILED: some instruments may be invalid"
    )

    return DiagnosticResult(
        test_name="Sargan-Hansen Overidentification Test",
        statistic=sargan_stat,
        p_value=p_value,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            "degrees_of_freedom": df,
            "n_instruments": n_instruments,
            "n_endogenous": n_endog
        }
    )


def sensitivity_analysis(
    effect: float,
    se: float,
    r2_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Oster (2019) sensitivity analysis for omitted variable bias.

    Parameters
    ----------
    effect : float
        Estimated treatment effect
    se : float
        Standard error of estimate
    r2_threshold : float
        Hypothetical R-squared from full model

    Returns
    -------
    Dict[str, Any]
        Sensitivity analysis results
    """
    # Simplified sensitivity: how much would unobserved confounding
    # need to explain to nullify the result?
    t_stat = effect / se

    # Breakdown point: proportion of effect explained by confounding
    # that would make result insignificant
    critical_t = 1.96
    breakdown = 1 - (critical_t / abs(t_stat)) if abs(t_stat) > 0 else 0

    return {
        "effect": effect,
        "se": se,
        "t_statistic": t_stat,
        "breakdown_point": max(0, breakdown),
        "interpretation": (
            f"Result robust: confounding would need to explain {breakdown*100:.1f}% "
            f"of the effect to make it insignificant"
            if breakdown > 0.5 else
            f"Result sensitive: confounding explaining just {(1-breakdown)*100:.1f}% "
            f"of the effect would make it insignificant"
        )
    }
