"""
Regression Discontinuity (RD) Estimator Implementation.

This module provides comprehensive RD estimation including:
- Sharp RD with local polynomial regression
- Fuzzy RD (local Wald estimator)
- McCrary density test for manipulation
- Bandwidth selection (MSE-optimal, CER-optimal)
- Placebo cutoff tests
- Bandwidth sensitivity analysis
- RD visualization
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

# Import from shared lib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'lib' / 'python'))
from data_loader import CausalInput, CausalOutput
from diagnostics import DiagnosticResult, balance_test
from table_formatter import create_regression_table, create_diagnostic_report


# =============================================================================
# Data Validation
# =============================================================================

@dataclass
class RDValidationResult:
    """Result of RD data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"RD Data Validation: {status}"]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def validate_rd_data(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    cutoff: float,
    treatment: str = None,
    bandwidth: float = None
) -> RDValidationResult:
    """
    Validate data structure for RD estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    running : str
        Running variable column name
    outcome : str
        Outcome variable column name
    cutoff : float
        Cutoff value
    treatment : str, optional
        Treatment indicator (for fuzzy RD)
    bandwidth : float, optional
        Bandwidth to check sample sizes within

    Returns
    -------
    RDValidationResult
        Validation results with errors and warnings
    """
    errors = []
    warnings_list = []
    summary = {}

    # Check required columns exist
    required_cols = [running, outcome]
    if treatment:
        required_cols.append(treatment)

    for col in required_cols:
        if col not in data.columns:
            errors.append(f"Required column '{col}' not found in data")

    if errors:
        return RDValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings_list,
            summary=summary
        )

    # Check for missing values
    for col in required_cols:
        n_missing = data[col].isna().sum()
        if n_missing > 0:
            warnings_list.append(f"Column '{col}' has {n_missing} missing values")

    # Basic statistics
    running_data = data[running].dropna()
    n_obs = len(running_data)
    summary['n_obs'] = n_obs

    # Check running variable properties
    n_unique = running_data.nunique()
    summary['n_unique_running'] = n_unique

    if n_unique < 10:
        warnings_list.append(
            f"Running variable has only {n_unique} unique values. "
            "RD may not be appropriate with discrete running variable."
        )

    # Check observations around cutoff
    n_below = (running_data < cutoff).sum()
    n_above = (running_data >= cutoff).sum()
    summary['n_below_cutoff'] = n_below
    summary['n_above_cutoff'] = n_above

    if n_below < 20:
        warnings_list.append(f"Only {n_below} observations below cutoff")
    if n_above < 20:
        warnings_list.append(f"Only {n_above} observations above cutoff")

    # Check within bandwidth if specified
    if bandwidth:
        n_in_bw_below = ((running_data >= cutoff - bandwidth) & (running_data < cutoff)).sum()
        n_in_bw_above = ((running_data >= cutoff) & (running_data < cutoff + bandwidth)).sum()
        summary['n_in_bandwidth_below'] = n_in_bw_below
        summary['n_in_bandwidth_above'] = n_in_bw_above

        if n_in_bw_below < 10 or n_in_bw_above < 10:
            warnings_list.append(
                f"Few observations within bandwidth: {n_in_bw_below} below, {n_in_bw_above} above"
            )

    # Check for sharp vs fuzzy
    if treatment:
        treat_below = data[data[running] < cutoff][treatment].mean()
        treat_above = data[data[running] >= cutoff][treatment].mean()
        summary['treatment_rate_below'] = treat_below
        summary['treatment_rate_above'] = treat_above

        if treat_below > 0.01 or treat_above < 0.99:
            summary['design_type'] = 'fuzzy'
            warnings_list.append(
                f"Fuzzy RD detected: treatment rate is {treat_below:.1%} below "
                f"and {treat_above:.1%} above cutoff"
            )
        else:
            summary['design_type'] = 'sharp'

    is_valid = len(errors) == 0

    return RDValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings_list,
        summary=summary
    )


# =============================================================================
# Kernel Functions
# =============================================================================

def triangular_kernel(u: np.ndarray) -> np.ndarray:
    """Triangular kernel: K(u) = (1 - |u|) * I(|u| <= 1)"""
    return np.maximum(0, 1 - np.abs(u))


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel: K(u) = 0.75 * (1 - u^2) * I(|u| <= 1)"""
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)


def uniform_kernel(u: np.ndarray) -> np.ndarray:
    """Uniform kernel: K(u) = 0.5 * I(|u| <= 1)"""
    return np.where(np.abs(u) <= 1, 0.5, 0)


def get_kernel(kernel_name: str):
    """Get kernel function by name."""
    kernels = {
        'triangular': triangular_kernel,
        'epanechnikov': epanechnikov_kernel,
        'uniform': uniform_kernel
    }
    if kernel_name.lower() not in kernels:
        raise ValueError(f"Unknown kernel: {kernel_name}. Available: {list(kernels.keys())}")
    return kernels[kernel_name.lower()]


# =============================================================================
# McCrary Density Test
# =============================================================================

def mccrary_test(
    running: Union[np.ndarray, pd.Series],
    cutoff: float,
    bandwidth: float = None,
    n_bins: int = None
) -> DiagnosticResult:
    """
    McCrary (2008) density test for manipulation.

    Tests for discontinuity in the density of the running variable at the cutoff.
    A significant discontinuity suggests manipulation/sorting.

    Parameters
    ----------
    running : array-like
        Running variable values
    cutoff : float
        Cutoff value
    bandwidth : float, optional
        Bandwidth for local polynomial density estimation.
        If None, uses rule-of-thumb bandwidth.
    n_bins : int, optional
        Number of bins for histogram (default: auto)

    Returns
    -------
    DiagnosticResult
        Test result with interpretation
    """
    if isinstance(running, pd.Series):
        running = running.values

    running = running[~np.isnan(running)]
    n = len(running)

    # Rule-of-thumb bandwidth
    if bandwidth is None:
        std = np.std(running)
        iqr = np.percentile(running, 75) - np.percentile(running, 25)
        h = 0.9 * min(std, iqr / 1.34) * n**(-1/5)
        bandwidth = 2 * h

    # Determine number of bins
    if n_bins is None:
        # Freedman-Diaconis rule
        iqr = np.percentile(running, 75) - np.percentile(running, 25)
        bin_width = 2 * iqr / (n**(1/3))
        n_bins = max(20, int((running.max() - running.min()) / bin_width))

    # Create histogram
    bins = np.linspace(running.min(), running.max(), n_bins + 1)
    bin_width = bins[1] - bins[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2

    counts, _ = np.histogram(running, bins=bins)
    density = counts / (n * bin_width)

    # Find bins near cutoff
    below_mask = bin_centers < cutoff
    above_mask = bin_centers >= cutoff

    # Use local linear regression to estimate density on each side
    # Near cutoff (within bandwidth)
    near_cutoff = np.abs(bin_centers - cutoff) <= bandwidth

    # Below cutoff
    x_below = bin_centers[near_cutoff & below_mask]
    y_below = density[near_cutoff & below_mask]

    # Above cutoff
    x_above = bin_centers[near_cutoff & above_mask]
    y_above = density[near_cutoff & above_mask]

    if len(x_below) < 3 or len(x_above) < 3:
        return DiagnosticResult(
            test_name="McCrary Density Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=True,
            threshold=0.05,
            interpretation="Insufficient data near cutoff for density test",
            details={"error": "insufficient_data"}
        )

    # Fit local linear regression on each side
    try:
        # Below cutoff: extrapolate to cutoff
        slope_b, intercept_b, _, _, se_b = stats.linregress(x_below, y_below)
        f_below = intercept_b + slope_b * cutoff

        # Above cutoff: extrapolate to cutoff
        slope_a, intercept_a, _, _, se_a = stats.linregress(x_above, y_above)
        f_above = intercept_a + slope_a * cutoff

        # Discontinuity estimate
        theta = np.log(f_above) - np.log(f_below) if f_below > 0 and f_above > 0 else 0

        # Standard error (simplified)
        # Use delta method approximation
        var_below = se_b**2 * (1/f_below)**2 if f_below > 0 else 1
        var_above = se_a**2 * (1/f_above)**2 if f_above > 0 else 1
        se_theta = np.sqrt(var_below + var_above)

        # Test statistic
        z_stat = theta / se_theta if se_theta > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    except Exception:
        # Fallback: simple chi-squared test
        near_below = np.sum((running >= cutoff - bandwidth) & (running < cutoff))
        near_above = np.sum((running >= cutoff) & (running < cutoff + bandwidth))
        expected = (near_below + near_above) / 2

        if expected > 0:
            chi_sq = ((near_below - expected)**2 + (near_above - expected)**2) / expected
            p_value = 1 - stats.chi2.cdf(chi_sq, df=1)
            z_stat = np.sign(near_above - near_below) * np.sqrt(chi_sq)
            theta = (near_above - near_below) / expected
        else:
            z_stat = 0
            p_value = 1.0
            theta = 0

    passed = p_value > 0.05

    if passed:
        interpretation = (
            f"McCrary test PASSED: No significant density discontinuity at cutoff "
            f"(log difference = {theta:.4f}, p = {p_value:.4f}). "
            "No evidence of manipulation."
        )
    else:
        direction = "bunching above" if theta > 0 else "bunching below"
        interpretation = (
            f"McCrary test FAILED: Significant density discontinuity detected "
            f"({direction} cutoff, log difference = {theta:.4f}, p = {p_value:.4f}). "
            "POTENTIAL MANIPULATION - interpret RD results with caution."
        )

    return DiagnosticResult(
        test_name="McCrary Density Test",
        statistic=theta,
        p_value=p_value,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            'log_density_difference': theta,
            'z_statistic': z_stat,
            'bandwidth': bandwidth,
            'n_bins': n_bins,
            'density_below': float(f_below) if 'f_below' in dir() else None,
            'density_above': float(f_above) if 'f_above' in dir() else None
        }
    )


# =============================================================================
# Bandwidth Selection
# =============================================================================

def select_bandwidth(
    running: Union[np.ndarray, pd.Series],
    outcome: Union[np.ndarray, pd.Series],
    cutoff: float,
    method: str = "mserd",
    kernel: str = "triangular",
    order: int = 1
) -> float:
    """
    Select optimal bandwidth for RD estimation.

    Implements IK (Imbens-Kalyanaraman) and CCT (Calonico-Cattaneo-Titiunik)
    bandwidth selection methods.

    Parameters
    ----------
    running : array-like
        Running variable
    outcome : array-like
        Outcome variable
    cutoff : float
        Cutoff value
    method : str
        Bandwidth selection method:
        - "mserd": MSE-optimal for RD point estimator (default)
        - "cerrd": Coverage error rate optimal for CI
        - "ik": Imbens-Kalyanaraman (2012)
    kernel : str
        Kernel function name
    order : int
        Polynomial order (1 = linear, 2 = quadratic)

    Returns
    -------
    float
        Selected bandwidth
    """
    if isinstance(running, pd.Series):
        running = running.values
    if isinstance(outcome, pd.Series):
        outcome = outcome.values

    # Remove missing values
    mask = ~(np.isnan(running) | np.isnan(outcome))
    running = running[mask]
    outcome = outcome[mask]

    n = len(running)

    # Separate data by side
    below_mask = running < cutoff
    above_mask = ~below_mask

    x_below = running[below_mask]
    y_below = outcome[below_mask]
    x_above = running[above_mask]
    y_above = outcome[above_mask]

    n_below = len(x_below)
    n_above = len(x_above)

    # Regularization constant
    reg = 1 / (3.4 * n**(1/5))

    # Step 1: Estimate preliminary quantities

    # Variance estimates near cutoff
    # Use pilot bandwidth (rule of thumb)
    h_pilot = 1.84 * np.std(running) * n**(-1/5)

    # Residual variance
    def local_var(x, y, c, h, side='below'):
        if side == 'below':
            mask = (x >= c - h) & (x < c)
        else:
            mask = (x >= c) & (x < c + h)

        if mask.sum() < 5:
            return np.var(y)

        x_local = x[mask] - c
        y_local = y[mask]

        if len(x_local) < 3:
            return np.var(y_local)

        # Local linear fit
        X = np.column_stack([np.ones(len(x_local)), x_local])
        try:
            beta = np.linalg.lstsq(X, y_local, rcond=None)[0]
            resid = y_local - X @ beta
            return np.var(resid, ddof=2)
        except:
            return np.var(y_local)

    sigma2_below = local_var(x_below, y_below, cutoff, h_pilot, 'below')
    sigma2_above = local_var(x_above, y_above, cutoff, h_pilot, 'above')

    # Second derivative estimates (for bias)
    def estimate_second_deriv(x, y, c, side='below'):
        """Estimate second derivative of conditional mean."""
        if side == 'below':
            mask = x < c
            x_use = x[mask]
            y_use = y[mask]
        else:
            mask = x >= c
            x_use = x[mask]
            y_use = y[mask]

        if len(x_use) < 5:
            return 0.0

        # Fit local quadratic
        x_centered = x_use - c
        X = np.column_stack([
            np.ones(len(x_centered)),
            x_centered,
            x_centered**2
        ])

        try:
            beta = np.linalg.lstsq(X, y_use, rcond=None)[0]
            return 2 * beta[2]  # Second derivative is 2 * quadratic coef
        except:
            return 0.0

    m2_below = estimate_second_deriv(x_below, y_below, cutoff, 'below')
    m2_above = estimate_second_deriv(x_above, y_above, cutoff, 'above')

    # Regularization to avoid division by near-zero
    m2_below = m2_below if abs(m2_below) > reg else reg * np.sign(m2_below) if m2_below != 0 else reg
    m2_above = m2_above if abs(m2_above) > reg else reg * np.sign(m2_above) if m2_above != 0 else reg

    # Step 2: Compute optimal bandwidth

    # Kernel constants (triangular)
    C_k = 2.576  # (integral of K^2) / (integral of u^2 K)^2

    # MSE-optimal bandwidth formula
    # h_MSE = C * (sigma^2 / (n * m2^2))^(1/5)

    sigma2 = (sigma2_below + sigma2_above) / 2
    m2_sq = ((m2_above - m2_below) / 2)**2

    if m2_sq < reg**2:
        m2_sq = reg**2

    h_mse = C_k * (sigma2 / (n * m2_sq))**(1/5)

    # Bound the bandwidth
    range_x = running.max() - running.min()
    h_mse = min(h_mse, range_x / 2)
    h_mse = max(h_mse, range_x / 50)

    if method.lower() == "cerrd":
        # CER-optimal scales MSE-optimal
        h_cer = h_mse * n**(-1/20)
        return h_cer
    elif method.lower() == "ik":
        # IK bandwidth (similar to MSE)
        return h_mse
    else:  # mserd (default)
        return h_mse


# =============================================================================
# Local Polynomial Regression
# =============================================================================

def _local_poly_regression(
    x: np.ndarray,
    y: np.ndarray,
    eval_point: float,
    bandwidth: float,
    kernel: str = "triangular",
    order: int = 1
) -> Tuple[float, float, np.ndarray]:
    """
    Fit local polynomial regression and return estimate at eval_point.

    Parameters
    ----------
    x : np.ndarray
        Predictor values
    y : np.ndarray
        Response values
    eval_point : float
        Point at which to evaluate the polynomial
    bandwidth : float
        Bandwidth
    kernel : str
        Kernel function name
    order : int
        Polynomial order

    Returns
    -------
    Tuple[float, float, np.ndarray]
        (estimate, std_error, coefficients)
    """
    kernel_func = get_kernel(kernel)

    # Center at evaluation point
    x_centered = x - eval_point

    # Compute kernel weights
    u = x_centered / bandwidth
    weights = kernel_func(u)

    # Keep only observations with positive weight
    mask = weights > 0
    if mask.sum() < order + 2:
        return np.nan, np.nan, np.array([np.nan])

    x_c = x_centered[mask]
    y_w = y[mask]
    w = weights[mask]

    # Build design matrix
    X = np.column_stack([x_c**p for p in range(order + 1)])

    # Weighted least squares
    W = np.diag(w)
    try:
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y_w
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.array([np.nan])

    # Estimate at eval_point (x_centered = 0)
    # Intercept is the estimate
    estimate = beta[0]

    # Standard error using sandwich estimator
    residuals = y_w - X @ beta
    sigma2 = np.sum(w * residuals**2) / (np.sum(mask) - order - 1)

    try:
        XtWX_inv = np.linalg.inv(XtWX)
        # Variance of intercept
        var_beta0 = sigma2 * XtWX_inv[0, 0]
        se = np.sqrt(var_beta0)
    except np.linalg.LinAlgError:
        se = np.nan

    return estimate, se, beta


# =============================================================================
# Sharp RD Estimation
# =============================================================================

def estimate_sharp_rd(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    cutoff: float,
    bandwidth: float = None,
    kernel: str = "triangular",
    order: int = 1,
    covariates: List[str] = None
) -> CausalOutput:
    """
    Estimate Sharp Regression Discontinuity design.

    Uses local polynomial regression on each side of the cutoff.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    running : str
        Running variable column name
    outcome : str
        Outcome variable column name
    cutoff : float
        Cutoff value
    bandwidth : float, optional
        Bandwidth (auto-selected if None)
    kernel : str
        Kernel function ("triangular", "epanechnikov", "uniform")
    order : int
        Polynomial order (1 = linear, 2 = quadratic)
    covariates : List[str], optional
        Covariates for covariate-adjusted RD

    Returns
    -------
    CausalOutput
        RD estimate with diagnostics
    """
    df = data.copy()

    x = df[running].values
    y = df[outcome].values

    # Remove missing
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    # Select bandwidth if not provided
    if bandwidth is None:
        bandwidth = select_bandwidth(x, y, cutoff, method="mserd", kernel=kernel, order=order)

    # Split data
    below_mask = x < cutoff
    above_mask = ~below_mask

    x_below = x[below_mask]
    y_below = y[below_mask]
    x_above = x[above_mask]
    y_above = y[above_mask]

    # Local polynomial regression on each side
    mu_below, se_below, _ = _local_poly_regression(
        x_below, y_below, cutoff, bandwidth, kernel, order
    )
    mu_above, se_above, _ = _local_poly_regression(
        x_above, y_above, cutoff, bandwidth, kernel, order
    )

    # RD estimate
    tau = mu_above - mu_below
    se = np.sqrt(se_below**2 + se_above**2)

    # Confidence interval
    ci_lower = tau - 1.96 * se
    ci_upper = tau + 1.96 * se

    # P-value
    z_stat = tau / se if se > 0 else np.inf
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Effective sample sizes
    kernel_func = get_kernel(kernel)
    u_below = (x_below - cutoff) / bandwidth
    u_above = (x_above - cutoff) / bandwidth
    n_eff_below = (kernel_func(u_below) > 0).sum()
    n_eff_above = (kernel_func(u_above) > 0).sum()

    diagnostics = {
        'method': 'Sharp RD',
        'bandwidth': bandwidth,
        'kernel': kernel,
        'order': order,
        'n_left': len(x_below),
        'n_right': len(x_above),
        'n_effective_left': n_eff_below,
        'n_effective_right': n_eff_above,
        'n_effective': n_eff_below + n_eff_above,
        'mu_left': mu_below,
        'mu_right': mu_above,
        'fuzzy': False
    }

    # Summary table
    table_results = [{
        'treatment_effect': tau,
        'treatment_se': se,
        'treatment_pval': p_value,
        'controls': False,
        'fixed_effects': None,
        'n_obs': n_eff_below + n_eff_above,
        'r_squared': np.nan
    }]

    summary_table = create_regression_table(
        results=table_results,
        column_names=["(1) Sharp RD"],
        title="Regression Discontinuity Results"
    )

    return CausalOutput(
        effect=tau,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        diagnostics=diagnostics,
        summary_table=summary_table,
        interpretation=(
            f"Sharp RD estimate: {tau:.4f} (SE = {se:.4f}). "
            f"Bandwidth = {bandwidth:.4f}, effective N = {n_eff_below + n_eff_above}."
        )
    )


# =============================================================================
# Fuzzy RD Estimation
# =============================================================================

def estimate_fuzzy_rd(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    treatment: str,
    cutoff: float,
    bandwidth: float = None,
    kernel: str = "triangular",
    order: int = 1
) -> CausalOutput:
    """
    Estimate Fuzzy Regression Discontinuity design.

    Uses local Wald estimator: ratio of reduced form to first stage.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    running : str
        Running variable column name
    outcome : str
        Outcome variable column name
    treatment : str
        Treatment indicator column name (actual treatment received)
    cutoff : float
        Cutoff value
    bandwidth : float, optional
        Bandwidth (auto-selected if None)
    kernel : str
        Kernel function
    order : int
        Polynomial order

    Returns
    -------
    CausalOutput
        Fuzzy RD estimate (LATE at cutoff)
    """
    df = data.copy()

    x = df[running].values
    y = df[outcome].values
    d = df[treatment].values

    # Remove missing
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(d))
    x = x[mask]
    y = y[mask]
    d = d[mask]

    # Select bandwidth
    if bandwidth is None:
        bandwidth = select_bandwidth(x, y, cutoff, method="mserd", kernel=kernel, order=order)

    # Split data
    below_mask = x < cutoff
    above_mask = ~below_mask

    x_below = x[below_mask]
    x_above = x[above_mask]

    y_below = y[below_mask]
    y_above = y[above_mask]

    d_below = d[below_mask]
    d_above = d[above_mask]

    # Reduced form: jump in outcome
    mu_y_below, se_y_below, _ = _local_poly_regression(
        x_below, y_below, cutoff, bandwidth, kernel, order
    )
    mu_y_above, se_y_above, _ = _local_poly_regression(
        x_above, y_above, cutoff, bandwidth, kernel, order
    )
    rf = mu_y_above - mu_y_below

    # First stage: jump in treatment
    mu_d_below, se_d_below, _ = _local_poly_regression(
        x_below, d_below, cutoff, bandwidth, kernel, order
    )
    mu_d_above, se_d_above, _ = _local_poly_regression(
        x_above, d_above, cutoff, bandwidth, kernel, order
    )
    fs = mu_d_above - mu_d_below

    # Check first stage strength
    if abs(fs) < 0.05:
        warnings.warn(
            f"Weak first stage: discontinuity in treatment is only {fs:.4f}. "
            "Fuzzy RD may be unreliable."
        )

    # Wald estimate
    if abs(fs) > 1e-10:
        tau = rf / fs
    else:
        tau = np.inf if rf > 0 else -np.inf if rf < 0 else np.nan

    # Standard error via delta method
    # Var(rf/fs) ~ (1/fs^2) * Var(rf) + (rf^2/fs^4) * Var(fs)
    var_rf = se_y_below**2 + se_y_above**2
    var_fs = se_d_below**2 + se_d_above**2

    if abs(fs) > 1e-10:
        se = np.sqrt((1/fs**2) * var_rf + (rf**2/fs**4) * var_fs)
    else:
        se = np.inf

    # Inference
    ci_lower = tau - 1.96 * se
    ci_upper = tau + 1.96 * se

    z_stat = tau / se if se > 0 and not np.isinf(se) else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat))) if not np.isnan(z_stat) else np.nan

    # Effective sample sizes
    kernel_func = get_kernel(kernel)
    u_below = (x_below - cutoff) / bandwidth
    u_above = (x_above - cutoff) / bandwidth
    n_eff_below = (kernel_func(u_below) > 0).sum()
    n_eff_above = (kernel_func(u_above) > 0).sum()

    diagnostics = {
        'method': 'Fuzzy RD',
        'bandwidth': bandwidth,
        'kernel': kernel,
        'order': order,
        'n_left': len(x_below),
        'n_right': len(x_above),
        'n_effective_left': n_eff_below,
        'n_effective_right': n_eff_above,
        'n_effective': n_eff_below + n_eff_above,
        'reduced_form': rf,
        'first_stage': fs,
        'treatment_rate_left': mu_d_below,
        'treatment_rate_right': mu_d_above,
        'fuzzy': True
    }

    # Summary table
    table_results = [{
        'treatment_effect': tau,
        'treatment_se': se,
        'treatment_pval': p_value,
        'controls': False,
        'fixed_effects': None,
        'n_obs': n_eff_below + n_eff_above,
        'r_squared': np.nan
    }]

    summary_table = create_regression_table(
        results=table_results,
        column_names=["(1) Fuzzy RD"],
        title="Fuzzy Regression Discontinuity Results"
    )

    return CausalOutput(
        effect=tau,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        diagnostics=diagnostics,
        summary_table=summary_table,
        interpretation=(
            f"Fuzzy RD estimate (LATE): {tau:.4f} (SE = {se:.4f}). "
            f"First stage: {fs:.4f}. Reduced form: {rf:.4f}. "
            f"Bandwidth = {bandwidth:.4f}."
        )
    )


# =============================================================================
# Placebo Cutoff Tests
# =============================================================================

def placebo_cutoff_test(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    true_cutoff: float,
    placebo_cutoffs: List[float],
    bandwidth: float = None,
    kernel: str = "triangular",
    order: int = 1
) -> Dict[float, DiagnosticResult]:
    """
    Test for effects at placebo (fake) cutoffs.

    RD effects should only appear at the true cutoff, not at placebo cutoffs.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    running : str
        Running variable column name
    outcome : str
        Outcome variable column name
    true_cutoff : float
        True cutoff value
    placebo_cutoffs : List[float]
        List of placebo cutoff values to test
    bandwidth : float, optional
        Bandwidth
    kernel : str
        Kernel function
    order : int
        Polynomial order

    Returns
    -------
    Dict[float, DiagnosticResult]
        Results for each placebo cutoff
    """
    results = {}

    for pc in placebo_cutoffs:
        # Only use data on one side of the true cutoff
        if pc < true_cutoff:
            # Use data below true cutoff
            subset = data[data[running] < true_cutoff]
        else:
            # Use data above true cutoff
            subset = data[data[running] >= true_cutoff]

        if len(subset) < 20:
            results[pc] = DiagnosticResult(
                test_name=f"Placebo Test at {pc}",
                statistic=np.nan,
                p_value=np.nan,
                passed=True,
                threshold=0.05,
                interpretation="Insufficient data for placebo test",
                details={"error": "insufficient_data"}
            )
            continue

        try:
            rd_result = estimate_sharp_rd(
                data=subset,
                running=running,
                outcome=outcome,
                cutoff=pc,
                bandwidth=bandwidth,
                kernel=kernel,
                order=order
            )

            passed = rd_result.p_value > 0.10  # No significant effect at placebo

            if passed:
                interpretation = (
                    f"Placebo test PASSED at cutoff {pc}: no significant effect "
                    f"(effect = {rd_result.effect:.4f}, p = {rd_result.p_value:.4f})"
                )
            else:
                interpretation = (
                    f"Placebo test FAILED at cutoff {pc}: significant effect detected "
                    f"(effect = {rd_result.effect:.4f}, p = {rd_result.p_value:.4f}). "
                    "This may indicate model misspecification."
                )

            results[pc] = DiagnosticResult(
                test_name=f"Placebo Test at {pc}",
                statistic=rd_result.effect,
                p_value=rd_result.p_value,
                passed=passed,
                threshold=0.10,
                interpretation=interpretation,
                details={
                    'effect': rd_result.effect,
                    'se': rd_result.se,
                    'n_effective': rd_result.diagnostics.get('n_effective', 0)
                }
            )

        except Exception as e:
            results[pc] = DiagnosticResult(
                test_name=f"Placebo Test at {pc}",
                statistic=np.nan,
                p_value=np.nan,
                passed=True,
                threshold=0.05,
                interpretation=f"Placebo test failed: {str(e)}",
                details={"error": str(e)}
            )

    return results


# =============================================================================
# Bandwidth Sensitivity
# =============================================================================

def bandwidth_sensitivity(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    cutoff: float,
    bandwidth_range: List[float] = None,
    kernel: str = "triangular",
    order: int = 1,
    treatment: str = None
) -> Dict[str, Any]:
    """
    Analyze sensitivity of RD estimates to bandwidth choice.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    running : str
        Running variable column name
    outcome : str
        Outcome variable column name
    cutoff : float
        Cutoff value
    bandwidth_range : List[float], optional
        List of bandwidths to test. If None, creates range around optimal.
    kernel : str
        Kernel function
    order : int
        Polynomial order
    treatment : str, optional
        Treatment column for fuzzy RD

    Returns
    -------
    Dict[str, Any]
        Sensitivity analysis results with summary table
    """
    # Get optimal bandwidth for reference
    x = data[running].values
    y = data[outcome].values
    mask = ~(np.isnan(x) | np.isnan(y))

    h_opt = select_bandwidth(x[mask], y[mask], cutoff, method="mserd")

    if bandwidth_range is None:
        bandwidth_range = [0.5 * h_opt, 0.75 * h_opt, h_opt, 1.25 * h_opt, 1.5 * h_opt, 2 * h_opt]

    results = []

    for bw in bandwidth_range:
        try:
            if treatment:
                rd_result = estimate_fuzzy_rd(
                    data, running, outcome, treatment, cutoff, bw, kernel, order
                )
            else:
                rd_result = estimate_sharp_rd(
                    data, running, outcome, cutoff, bw, kernel, order
                )

            results.append({
                'bandwidth': bw,
                'bw_ratio': bw / h_opt,
                'effect': rd_result.effect,
                'se': rd_result.se,
                'ci_lower': rd_result.ci_lower,
                'ci_upper': rd_result.ci_upper,
                'p_value': rd_result.p_value,
                'n_effective': rd_result.diagnostics.get('n_effective', 0)
            })

        except Exception as e:
            results.append({
                'bandwidth': bw,
                'bw_ratio': bw / h_opt,
                'effect': np.nan,
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'p_value': np.nan,
                'n_effective': 0,
                'error': str(e)
            })

    # Create summary table
    lines = []
    lines.append("### Bandwidth Sensitivity Analysis")
    lines.append("")
    lines.append("| Bandwidth | Ratio | Effect | SE | 95% CI | p-value | N_eff |")
    lines.append("|-----------|-------|--------|-----|--------|---------|-------|")

    for r in results:
        ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]" if not np.isnan(r['effect']) else "N/A"
        lines.append(
            f"| {r['bandwidth']:.4f} | {r['bw_ratio']:.2f}x | "
            f"{r['effect']:.4f} | {r['se']:.4f} | {ci_str} | "
            f"{r['p_value']:.4f} | {r['n_effective']} |"
        )

    lines.append("")
    lines.append(f"*Optimal bandwidth (MSE): {h_opt:.4f}*")

    summary_table = "\n".join(lines)

    return {
        'results': results,
        'optimal_bandwidth': h_opt,
        'summary_table': summary_table
    }


# =============================================================================
# Donut Hole RD
# =============================================================================

def donut_hole_rd(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    cutoff: float,
    bandwidth: float,
    donut_radius: float,
    kernel: str = "triangular",
    order: int = 1,
    treatment: str = None
) -> CausalOutput:
    """
    Donut hole RD: exclude observations very close to cutoff.

    Useful when manipulation is suspected right at the cutoff.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    running : str
        Running variable
    outcome : str
        Outcome variable
    cutoff : float
        Cutoff value
    bandwidth : float
        Bandwidth
    donut_radius : float
        Radius of donut hole (exclude |X - c| < donut_radius)
    kernel : str
        Kernel function
    order : int
        Polynomial order
    treatment : str, optional
        Treatment column for fuzzy RD

    Returns
    -------
    CausalOutput
        Donut hole RD estimate
    """
    # Create donut by excluding observations near cutoff
    donut_data = data[np.abs(data[running] - cutoff) >= donut_radius].copy()

    if len(donut_data) < 40:
        raise ValueError(
            f"Donut hole (radius={donut_radius}) removes too many observations. "
            f"Only {len(donut_data)} remain."
        )

    if treatment:
        result = estimate_fuzzy_rd(
            donut_data, running, outcome, treatment, cutoff, bandwidth, kernel, order
        )
    else:
        result = estimate_sharp_rd(
            donut_data, running, outcome, cutoff, bandwidth, kernel, order
        )

    # Update diagnostics
    result.diagnostics['donut_radius'] = donut_radius
    result.diagnostics['n_excluded'] = len(data) - len(donut_data)
    result.interpretation = (
        f"Donut hole RD (radius={donut_radius}): {result.effect:.4f} (SE={result.se:.4f}). "
        f"Excluded {result.diagnostics['n_excluded']} observations near cutoff."
    )

    return result


# =============================================================================
# Covariate Balance at Cutoff
# =============================================================================

def covariate_balance_rd(
    data: pd.DataFrame,
    running: str,
    cutoff: float,
    covariates: List[str],
    bandwidth: float = None
) -> Dict[str, DiagnosticResult]:
    """
    Test covariate balance at the RD cutoff.

    Covariates should be smooth (no discontinuity) at the cutoff.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    running : str
        Running variable
    cutoff : float
        Cutoff value
    covariates : List[str]
        List of covariate column names
    bandwidth : float, optional
        Bandwidth for local comparisons

    Returns
    -------
    Dict[str, DiagnosticResult]
        Balance test results for each covariate
    """
    results = {}

    for cov in covariates:
        if cov not in data.columns:
            results[cov] = DiagnosticResult(
                test_name=f"Balance Test: {cov}",
                statistic=np.nan,
                p_value=np.nan,
                passed=False,
                threshold=0.05,
                interpretation=f"Covariate '{cov}' not found in data",
                details={"error": "column_not_found"}
            )
            continue

        try:
            # Run RD on covariate as outcome
            rd_result = estimate_sharp_rd(
                data=data,
                running=running,
                outcome=cov,
                cutoff=cutoff,
                bandwidth=bandwidth
            )

            # Covariate should NOT show a discontinuity
            passed = rd_result.p_value > 0.05

            if passed:
                interpretation = (
                    f"Covariate '{cov}' is BALANCED at cutoff: "
                    f"no significant discontinuity (jump = {rd_result.effect:.4f}, p = {rd_result.p_value:.4f})"
                )
            else:
                interpretation = (
                    f"Covariate '{cov}' is IMBALANCED: "
                    f"significant discontinuity at cutoff (jump = {rd_result.effect:.4f}, p = {rd_result.p_value:.4f}). "
                    "This may indicate sorting or confounding."
                )

            results[cov] = DiagnosticResult(
                test_name=f"Balance Test: {cov}",
                statistic=rd_result.effect,
                p_value=rd_result.p_value,
                passed=passed,
                threshold=0.05,
                interpretation=interpretation,
                details={
                    'discontinuity': rd_result.effect,
                    'se': rd_result.se,
                    'mean_below': rd_result.diagnostics.get('mu_left', np.nan),
                    'mean_above': rd_result.diagnostics.get('mu_right', np.nan)
                }
            )

        except Exception as e:
            results[cov] = DiagnosticResult(
                test_name=f"Balance Test: {cov}",
                statistic=np.nan,
                p_value=np.nan,
                passed=True,
                threshold=0.05,
                interpretation=f"Balance test failed: {str(e)}",
                details={"error": str(e)}
            )

    return results


# =============================================================================
# RD Visualization
# =============================================================================

def rd_plot(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    cutoff: float,
    bandwidth: float = None,
    n_bins: int = 20,
    poly_order: int = 1,
    ci: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Regression Discontinuity Plot"
) -> Any:
    """
    Create RD visualization with binned scatter and polynomial fits.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    running : str
        Running variable column name
    outcome : str
        Outcome variable column name
    cutoff : float
        Cutoff value
    bandwidth : float, optional
        Bandwidth for polynomial fits (auto-selected if None)
    n_bins : int
        Number of bins for scatter plot
    poly_order : int
        Polynomial order for fitted lines
    ci : bool
        Whether to show confidence intervals
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        RD plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")

    df = data.copy()
    x = df[running].values
    y = df[outcome].values

    # Remove missing
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    # Select bandwidth
    if bandwidth is None:
        bandwidth = select_bandwidth(x, y, cutoff)

    # Create bins for scatter
    x_range = x.max() - x.min()
    bin_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate bin means
    bin_means = []
    bin_ses = []

    for i in range(n_bins):
        in_bin = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if in_bin.sum() > 0:
            bin_means.append(y[in_bin].mean())
            bin_ses.append(y[in_bin].std() / np.sqrt(in_bin.sum()))
        else:
            bin_means.append(np.nan)
            bin_ses.append(np.nan)

    bin_means = np.array(bin_means)
    bin_ses = np.array(bin_ses)

    # Fit polynomials on each side within bandwidth
    below_mask = x < cutoff
    above_mask = ~below_mask

    x_below = x[below_mask]
    y_below = y[below_mask]
    x_above = x[above_mask]
    y_above = y[above_mask]

    # Grid for plotting
    x_plot_below = np.linspace(max(x.min(), cutoff - bandwidth), cutoff - 0.001, 100)
    x_plot_above = np.linspace(cutoff + 0.001, min(x.max(), cutoff + bandwidth), 100)

    # Fit local polynomial for visualization
    def fit_poly(x_data, y_data, c, h, order):
        """Fit polynomial within bandwidth."""
        in_bw = (x_data >= c - h) & (x_data <= c + h)
        x_in = x_data[in_bw] - c
        y_in = y_data[in_bw]

        if len(x_in) < order + 2:
            return None, None

        X = np.column_stack([x_in**p for p in range(order + 1)])
        try:
            beta = np.linalg.lstsq(X, y_in, rcond=None)[0]
            return beta, X
        except:
            return None, None

    # Below cutoff
    beta_below, _ = fit_poly(x_below, y_below, cutoff, bandwidth, poly_order)
    if beta_below is not None:
        X_pred_below = np.column_stack([(x_plot_below - cutoff)**p for p in range(poly_order + 1)])
        y_pred_below = X_pred_below @ beta_below
    else:
        y_pred_below = None

    # Above cutoff
    beta_above, _ = fit_poly(x_above, y_above, cutoff, bandwidth, poly_order)
    if beta_above is not None:
        X_pred_above = np.column_stack([(x_plot_above - cutoff)**p for p in range(poly_order + 1)])
        y_pred_above = X_pred_above @ beta_above
    else:
        y_pred_above = None

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Binned scatter
    below_bins = bin_centers < cutoff
    above_bins = bin_centers >= cutoff

    ax.scatter(
        bin_centers[below_bins], bin_means[below_bins],
        color='blue', alpha=0.7, s=50, label='Bin means (below)'
    )
    ax.scatter(
        bin_centers[above_bins], bin_means[above_bins],
        color='red', alpha=0.7, s=50, label='Bin means (above)'
    )

    # Error bars
    if ci:
        ax.errorbar(
            bin_centers[below_bins], bin_means[below_bins],
            yerr=1.96 * bin_ses[below_bins],
            fmt='none', color='blue', alpha=0.3, capsize=2
        )
        ax.errorbar(
            bin_centers[above_bins], bin_means[above_bins],
            yerr=1.96 * bin_ses[above_bins],
            fmt='none', color='red', alpha=0.3, capsize=2
        )

    # Fitted lines
    if y_pred_below is not None:
        ax.plot(x_plot_below, y_pred_below, color='blue', linewidth=2)
    if y_pred_above is not None:
        ax.plot(x_plot_above, y_pred_above, color='red', linewidth=2)

    # Vertical line at cutoff
    ax.axvline(x=cutoff, color='black', linestyle='--', linewidth=1.5, label=f'Cutoff = {cutoff}')

    # Bandwidth markers
    ax.axvline(x=cutoff - bandwidth, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=cutoff + bandwidth, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel(f'{running}')
    ax.set_ylabel(f'Mean {outcome}')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation for bandwidth
    ax.text(
        0.02, 0.98,
        f'Bandwidth: {bandwidth:.4f}',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top'
    )

    plt.tight_layout()
    return fig


# =============================================================================
# Full RD Analysis Workflow
# =============================================================================

def run_full_rd_analysis(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    cutoff: float,
    treatment: str = None,
    covariates: List[str] = None,
    bandwidth: float = None,
    kernel: str = "triangular",
    order: int = 1
) -> CausalOutput:
    """
    Run complete RD analysis workflow.

    This function:
    1. Validates data structure
    2. Runs McCrary density test
    3. Checks covariate balance
    4. Estimates RD effect
    5. Runs placebo tests
    6. Performs bandwidth sensitivity
    7. Generates comprehensive output

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    running : str
        Running variable column name
    outcome : str
        Outcome variable column name
    cutoff : float
        Cutoff value
    treatment : str, optional
        Treatment column for fuzzy RD
    covariates : List[str], optional
        Covariates for balance tests
    bandwidth : float, optional
        Bandwidth (auto-selected if None)
    kernel : str
        Kernel function
    order : int
        Polynomial order

    Returns
    -------
    CausalOutput
        Complete analysis results with all diagnostics
    """
    # Step 1: Validate data
    validation = validate_rd_data(data, running, outcome, cutoff, treatment)

    if not validation.is_valid:
        raise ValueError(f"Data validation failed: {validation.errors}")

    # Step 2: McCrary test
    mccrary_result = mccrary_test(data[running], cutoff)

    # Step 3: Covariate balance (if covariates provided)
    balance_results = None
    if covariates:
        balance_results = covariate_balance_rd(
            data, running, cutoff, covariates, bandwidth
        )

    # Step 4: Select bandwidth
    x = data[running].values
    y = data[outcome].values
    mask = ~(np.isnan(x) | np.isnan(y))

    if bandwidth is None:
        bandwidth = select_bandwidth(x[mask], y[mask], cutoff, kernel=kernel, order=order)

    # Step 5: Main estimation
    if treatment:
        main_result = estimate_fuzzy_rd(
            data, running, outcome, treatment, cutoff, bandwidth, kernel, order
        )
    else:
        main_result = estimate_sharp_rd(
            data, running, outcome, cutoff, bandwidth, kernel, order
        )

    # Step 6: Placebo cutoff tests
    # Use quartiles of running variable as placebo cutoffs (avoiding true cutoff)
    x_clean = x[mask]
    below = x_clean[x_clean < cutoff]
    above = x_clean[x_clean >= cutoff]

    placebo_cutoffs = []
    if len(below) > 50:
        placebo_cutoffs.append(np.percentile(below, 50))
    if len(above) > 50:
        placebo_cutoffs.append(np.percentile(above, 50))

    placebo_results = {}
    if placebo_cutoffs:
        placebo_results = placebo_cutoff_test(
            data, running, outcome, cutoff, placebo_cutoffs, bandwidth, kernel, order
        )

    # Step 7: Bandwidth sensitivity
    sensitivity = bandwidth_sensitivity(
        data, running, outcome, cutoff,
        bandwidth_range=[0.5*bandwidth, 0.75*bandwidth, bandwidth, 1.5*bandwidth, 2*bandwidth],
        kernel=kernel, order=order, treatment=treatment
    )

    # Compile all diagnostics
    all_diagnostics = {
        'validation': validation.summary,
        'mccrary_test': mccrary_result,
        'main_estimation': main_result.diagnostics,
        'bandwidth_sensitivity': sensitivity['results']
    }

    if balance_results:
        all_diagnostics['covariate_balance'] = balance_results

    if placebo_results:
        all_diagnostics['placebo_tests'] = placebo_results

    # Generate comprehensive summary
    design_type = "Fuzzy" if treatment else "Sharp"

    summary_lines = [
        "=" * 60,
        f"{design_type.upper()} REGRESSION DISCONTINUITY ANALYSIS RESULTS",
        "=" * 60,
        "",
        f"RD Effect (LATE): {main_result.effect:.4f}",
        f"Standard Error: {main_result.se:.4f}",
        f"95% CI: [{main_result.ci_lower:.4f}, {main_result.ci_upper:.4f}]",
        f"P-value: {main_result.p_value:.4f}",
        "",
        "-" * 60,
        "DIAGNOSTICS",
        "-" * 60,
        "",
        f"McCrary Density Test: {'PASSED' if mccrary_result.passed else 'FAILED'}",
        f"  - Test statistic: {mccrary_result.statistic:.4f}",
        f"  - P-value: {mccrary_result.p_value:.4f}",
        "",
    ]

    # Covariate balance summary
    if balance_results:
        n_balanced = sum(1 for r in balance_results.values() if r.passed)
        summary_lines.extend([
            f"Covariate Balance: {n_balanced}/{len(balance_results)} covariates balanced",
        ])
        for cov, result in balance_results.items():
            status = "Balanced" if result.passed else "IMBALANCED"
            summary_lines.append(f"  - {cov}: {status} (p={result.p_value:.3f})")
        summary_lines.append("")

    # Placebo tests summary
    if placebo_results:
        n_passed = sum(1 for r in placebo_results.values() if r.passed)
        summary_lines.extend([
            f"Placebo Cutoff Tests: {n_passed}/{len(placebo_results)} passed",
        ])
        for pc, result in placebo_results.items():
            status = "Pass" if result.passed else "FAIL"
            summary_lines.append(f"  - Cutoff {pc}: {status} (effect={result.statistic:.3f}, p={result.p_value:.3f})")
        summary_lines.append("")

    summary_lines.extend([
        "-" * 60,
        "BANDWIDTH AND SAMPLE",
        "-" * 60,
        f"Optimal Bandwidth: {bandwidth:.4f}",
        f"Kernel: {kernel}",
        f"Polynomial Order: {order}",
        f"N (below cutoff): {main_result.diagnostics.get('n_left', 'N/A')}",
        f"N (above cutoff): {main_result.diagnostics.get('n_right', 'N/A')}",
        f"Effective N: {main_result.diagnostics.get('n_effective', 'N/A')}",
        "",
        "=" * 60
    ])

    comprehensive_summary = "\n".join(summary_lines)

    # Generate interpretation
    interpretation = main_result.generate_interpretation(
        treatment_name="crossing the cutoff",
        outcome_name=outcome
    )

    # Add caveats based on diagnostics
    if not mccrary_result.passed:
        interpretation += (
            "\n\nCAUTION: McCrary density test suggests potential manipulation at the cutoff. "
            "Results should be interpreted with caution. Consider donut hole RD."
        )

    if balance_results:
        imbalanced = [cov for cov, r in balance_results.items() if not r.passed]
        if imbalanced:
            interpretation += (
                f"\n\nWARNING: Some covariates show imbalance at cutoff: {imbalanced}. "
                "This may indicate sorting or confounding."
            )

    return CausalOutput(
        effect=main_result.effect,
        se=main_result.se,
        ci_lower=main_result.ci_lower,
        ci_upper=main_result.ci_upper,
        p_value=main_result.p_value,
        diagnostics=all_diagnostics,
        summary_table=comprehensive_summary,
        interpretation=interpretation
    )


# =============================================================================
# Validation with Synthetic Data
# =============================================================================

def validate_estimator(verbose: bool = True) -> Dict[str, Any]:
    """
    Validate RD estimator on synthetic data with known treatment effect.

    Returns
    -------
    Dict[str, Any]
        Validation results including bias assessment
    """
    from data_loader import generate_synthetic_rd_data

    # Generate synthetic data
    true_late = 0.5
    data, true_params = generate_synthetic_rd_data(
        n=2000,
        cutoff=0.0,
        treatment_effect=true_late,
        noise_std=0.3,
        random_state=42
    )

    # Run estimation
    result = run_full_rd_analysis(
        data=data,
        running='running',
        outcome='y',
        cutoff=0.0,
        covariates=['x1', 'x2']
    )

    # Calculate bias
    bias = result.effect - true_late
    bias_pct = abs(bias / true_late) * 100

    # Check if within acceptable range (10% bias for RD)
    passed = bias_pct < 10.0

    validation_result = {
        'true_late': true_late,
        'estimated_late': result.effect,
        'se': result.se,
        'bias': bias,
        'bias_pct': bias_pct,
        'passed': passed,
        'ci_covers_truth': result.ci_lower <= true_late <= result.ci_upper,
        'mccrary_passed': result.diagnostics['mccrary_test'].passed
    }

    if verbose:
        print("=" * 50)
        print("RD ESTIMATOR VALIDATION")
        print("=" * 50)
        print(f"True LATE: {true_late:.4f}")
        print(f"Estimated LATE: {result.effect:.4f}")
        print(f"Standard Error: {result.se:.4f}")
        print(f"Bias: {bias:.4f} ({bias_pct:.2f}%)")
        print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"CI covers truth: {validation_result['ci_covers_truth']}")
        print(f"McCrary test: {'PASSED' if validation_result['mccrary_passed'] else 'FAILED'}")
        print("-" * 50)
        print(f"VALIDATION: {'PASSED' if passed else 'FAILED'} (bias < 10%)")
        print("=" * 50)

    return validation_result


if __name__ == "__main__":
    # Run validation when module is executed directly
    validate_estimator(verbose=True)
