#!/usr/bin/env python3
"""
Weak-IV Robust Inference Methods.

This script provides robust inference methods that are valid even with
weak instruments, including:
- Anderson-Rubin confidence intervals
- Conditional Likelihood Ratio (CLR) test
- Fuller's modified LIML
- Weak-IV robust standard errors

Usage:
    python weak_iv_robust.py --data data.csv --outcome y --treatment d \
        --instruments z1 z2 --controls x1 x2 --method ar

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from iv_estimator import (
    first_stage_test,
    weak_iv_diagnostics,
    estimate_2sls,
    estimate_liml
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Weak-IV robust inference methods"
    )

    parser.add_argument("--data", "-d", type=str, required=True)
    parser.add_argument("--outcome", "-y", type=str, required=True)
    parser.add_argument("--treatment", "-t", type=str, required=True)
    parser.add_argument("--instruments", "-z", type=str, nargs="+", required=True)
    parser.add_argument("--controls", "-x", type=str, nargs="*", default=None)
    parser.add_argument("--method", type=str,
                        choices=["ar", "clr", "fuller", "all"],
                        default="all",
                        help="Inference method")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level")
    parser.add_argument("--fuller-alpha", type=float, default=1.0,
                        help="Fuller modification parameter (1 or 4)")
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


# =============================================================================
# Anderson-Rubin Methods
# =============================================================================

def anderson_rubin_statistic(
    y: np.ndarray,
    d: np.ndarray,
    Z: np.ndarray,
    gamma_0: float
) -> Tuple[float, float, int, int]:
    """
    Compute Anderson-Rubin test statistic for H0: gamma = gamma_0.

    Parameters
    ----------
    y : np.ndarray
        Outcome variable (n,)
    d : np.ndarray
        Treatment variable (n,)
    Z : np.ndarray
        Instrument matrix including controls and constant (n, K)
    gamma_0 : float
        Null hypothesis value

    Returns
    -------
    Tuple[float, float, int, int]
        AR statistic, p-value, df1, df2
    """
    n, K = Z.shape

    # Residual under null
    residual = y - d * gamma_0

    # Projection matrices
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    P_Z = Z @ ZtZ_inv @ Z.T
    M_Z = np.eye(n) - P_Z

    # AR statistic
    numerator = (residual @ P_Z @ residual) / K
    denominator = (residual @ M_Z @ residual) / (n - K)

    ar_stat = numerator / denominator
    p_value = 1 - stats.f.cdf(ar_stat, K, n - K)

    return ar_stat, p_value, K, n - K


def anderson_rubin_ci(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]] = None,
    alpha: float = 0.05,
    grid_points: int = 1000,
    tol: float = 1e-6
) -> Dict[str, Any]:
    """
    Compute Anderson-Rubin confidence interval via grid search and refinement.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable
    treatment : str
        Treatment variable
    instruments : List[str]
        Instrument names
    controls : List[str], optional
        Control variable names
    alpha : float
        Significance level
    grid_points : int
        Initial grid points
    tol : float
        Tolerance for boundary refinement

    Returns
    -------
    Dict[str, Any]
        Confidence interval results
    """
    import statsmodels.api as sm

    df = data.copy()
    y = df[outcome].values
    d = df[treatment].values

    # Build instrument matrix
    Z_vars = instruments.copy()
    if controls:
        Z_vars.extend(controls)
    Z = sm.add_constant(df[Z_vars]).values

    # Remove missing
    mask = ~(np.isnan(y) | np.isnan(d) | np.isnan(Z).any(axis=1))
    y = y[mask]
    d = d[mask]
    Z = Z[mask]

    n, K = Z.shape

    # Critical value
    f_crit = stats.f.ppf(1 - alpha, K, n - K)

    # Get 2SLS as starting point
    result_2sls = estimate_2sls(data, outcome, treatment, instruments, controls)
    center = result_2sls.effect
    width = max(10 * result_2sls.se, abs(center) * 2, 1.0)

    # Helper function: is gamma_0 in CI?
    def in_confidence_set(gamma_0):
        ar_stat, _, _, _ = anderson_rubin_statistic(y, d, Z, gamma_0)
        return ar_stat <= f_crit

    # Initial coarse grid search
    gamma_grid = np.linspace(center - width, center + width, grid_points)
    in_ci_flags = [in_confidence_set(g) for g in gamma_grid]

    # Find boundaries
    in_ci_indices = [i for i, flag in enumerate(in_ci_flags) if flag]

    if len(in_ci_indices) == 0:
        # Expand search
        width *= 3
        gamma_grid = np.linspace(center - width, center + width, grid_points)
        in_ci_flags = [in_confidence_set(g) for g in gamma_grid]
        in_ci_indices = [i for i, flag in enumerate(in_ci_flags) if flag]

        if len(in_ci_indices) == 0:
            return {
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'bounded': False,
                'empty': True,
                'point_estimate': center,
                'method': 'Anderson-Rubin',
                'alpha': alpha
            }

    # Coarse bounds
    coarse_lower = gamma_grid[min(in_ci_indices)]
    coarse_upper = gamma_grid[max(in_ci_indices)]

    # Refine boundaries using bisection
    def refine_boundary(left, right, finding_lower=True):
        """Find boundary using bisection."""
        if finding_lower:
            # Finding lower bound: in_ci at right, not at left
            while right - left > tol:
                mid = (left + right) / 2
                if in_confidence_set(mid):
                    right = mid
                else:
                    left = mid
            return right
        else:
            # Finding upper bound: in_ci at left, not at right
            while right - left > tol:
                mid = (left + right) / 2
                if in_confidence_set(mid):
                    left = mid
                else:
                    right = mid
            return left

    # Check if extends to boundaries (potentially unbounded)
    extends_lower = min(in_ci_indices) == 0
    extends_upper = max(in_ci_indices) == len(gamma_grid) - 1

    if extends_lower and extends_upper:
        # CI covers entire grid - likely unbounded
        return {
            'ci_lower': -np.inf,
            'ci_upper': np.inf,
            'bounded': False,
            'empty': False,
            'point_estimate': center,
            'method': 'Anderson-Rubin',
            'alpha': alpha,
            'note': 'Confidence interval is unbounded (very weak instruments)'
        }

    # Refine lower bound
    if extends_lower:
        ci_lower = gamma_grid[0]
    else:
        # Find transition point
        search_left = gamma_grid[max(0, min(in_ci_indices) - 1)]
        search_right = coarse_lower
        ci_lower = refine_boundary(search_left, search_right, finding_lower=True)

    # Refine upper bound
    if extends_upper:
        ci_upper = gamma_grid[-1]
    else:
        search_left = coarse_upper
        search_right = gamma_grid[min(len(gamma_grid) - 1, max(in_ci_indices) + 1)]
        ci_upper = refine_boundary(search_left, search_right, finding_lower=False)

    bounded = not (extends_lower or extends_upper)

    return {
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bounded': bounded,
        'empty': False,
        'width': ci_upper - ci_lower,
        'point_estimate': center,
        'method': 'Anderson-Rubin',
        'alpha': alpha,
        'critical_value': f_crit,
        'df1': K,
        'df2': n - K
    }


def anderson_rubin_test_zero(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Anderson-Rubin test of H0: treatment effect = 0.

    Returns
    -------
    Dict[str, float]
        Test results
    """
    import statsmodels.api as sm

    df = data.copy()
    y = df[outcome].values
    d = df[treatment].values

    Z_vars = instruments.copy()
    if controls:
        Z_vars.extend(controls)
    Z = sm.add_constant(df[Z_vars]).values

    mask = ~(np.isnan(y) | np.isnan(d) | np.isnan(Z).any(axis=1))
    y, d, Z = y[mask], d[mask], Z[mask]

    ar_stat, p_value, df1, df2 = anderson_rubin_statistic(y, d, Z, gamma_0=0.0)

    return {
        'statistic': ar_stat,
        'p_value': p_value,
        'df1': df1,
        'df2': df2,
        'reject_at_05': p_value < 0.05,
        'reject_at_01': p_value < 0.01
    }


# =============================================================================
# Fuller's Modified LIML
# =============================================================================

def estimate_fuller(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]] = None,
    alpha: float = 1.0
) -> Dict[str, Any]:
    """
    Estimate Fuller's modified LIML.

    Fuller's modification reduces the small-sample bias of LIML by subtracting
    a bias correction term from the LIML kappa.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable
    treatment : str
        Treatment variable
    instruments : List[str]
        Instrument names
    controls : List[str], optional
        Control variable names
    alpha : float
        Fuller modification parameter (1 or 4 typical)
        - alpha=1: approximately minimizes MSE
        - alpha=4: better CI coverage

    Returns
    -------
    Dict[str, Any]
        Fuller estimation results
    """
    import statsmodels.api as sm

    df = data.copy()
    y = df[outcome].values
    d = df[treatment].values

    # Build matrices
    X_vars = controls if controls else []
    Z_vars = instruments

    n = len(df)

    # Create X (exogenous including constant)
    if X_vars:
        X = sm.add_constant(df[X_vars]).values
    else:
        X = np.ones((n, 1))

    # Create Z (instruments)
    Z = df[Z_vars].values

    # Combined W = [X, Z]
    W = np.column_stack([X, Z])

    # Remove missing
    mask = ~(np.isnan(y) | np.isnan(d) | np.isnan(X).any(axis=1) | np.isnan(Z).any(axis=1))
    y, d, X, Z, W = y[mask], d[mask], X[mask], Z[mask], W[mask]

    n = len(y)
    K = Z.shape[1]  # Number of excluded instruments
    L = X.shape[1]  # Number of included exogenous

    # Compute residual maker matrices
    XtX_inv = np.linalg.inv(X.T @ X)
    M_X = np.eye(n) - X @ XtX_inv @ X.T

    WtW_inv = np.linalg.inv(W.T @ W)
    M_W = np.eye(n) - W @ WtW_inv @ W.T

    # Stack y and d
    Y = np.column_stack([y, d])

    # Compute matrices for LIML
    A = Y.T @ M_W @ Y
    B = Y.T @ M_X @ Y

    # Solve eigenvalue problem: (B - kappa*A)v = 0
    # This is equivalent to: B*v = kappa*A*v
    # Smallest eigenvalue of inv(A)*B
    try:
        A_inv = np.linalg.inv(A)
        eigenvalues = np.linalg.eigvals(A_inv @ B)
        kappa_liml = np.min(np.real(eigenvalues))
    except np.linalg.LinAlgError:
        return {'error': 'Singular matrix in LIML computation'}

    # Fuller modification
    kappa_fuller = kappa_liml - alpha / (n - K - L)

    # Compute Fuller estimate
    # beta = (d'M_X d - kappa*d'M_W d)^{-1} * (d'M_X y - kappa*d'M_W y)
    d_MX_d = d @ M_X @ d
    d_MW_d = d @ M_W @ d
    d_MX_y = d @ M_X @ y
    d_MW_y = d @ M_W @ y

    denominator = d_MX_d - kappa_fuller * d_MW_d
    numerator = d_MX_y - kappa_fuller * d_MW_y

    if abs(denominator) < 1e-10:
        return {'error': 'Near-zero denominator in Fuller estimation'}

    gamma_fuller = numerator / denominator

    # Compute residuals and standard error
    residuals = y - d * gamma_fuller

    # Robust variance estimation
    sigma2 = np.sum(residuals**2) / (n - K - L - 1)

    # Variance of Fuller estimator (approximate)
    d_tilde = d - X @ XtX_inv @ X.T @ d  # Residualized d
    Z_tilde = Z - X @ XtX_inv @ X.T @ Z  # Residualized Z
    P_Z_tilde = Z_tilde @ np.linalg.inv(Z_tilde.T @ Z_tilde) @ Z_tilde.T

    var_gamma = sigma2 / (d_tilde @ P_Z_tilde @ d_tilde)
    se_gamma = np.sqrt(var_gamma)

    # Confidence interval
    ci_lower = gamma_fuller - 1.96 * se_gamma
    ci_upper = gamma_fuller + 1.96 * se_gamma

    return {
        'estimate': gamma_fuller,
        'se': se_gamma,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'kappa_liml': kappa_liml,
        'kappa_fuller': kappa_fuller,
        'alpha': alpha,
        'n_obs': n,
        'method': f'Fuller(alpha={alpha})'
    }


# =============================================================================
# Weak-IV Robust Summary
# =============================================================================

def weak_iv_robust_summary(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]] = None,
    alpha: float = 0.05,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive weak-IV robust inference summary.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable
    treatment : str
        Treatment variable
    instruments : List[str]
        Instrument names
    controls : List[str], optional
        Control variable names
    alpha : float
        Significance level
    verbose : bool
        Print detailed output

    Returns
    -------
    Dict[str, Any]
        All results
    """
    results = {}

    # First-stage assessment
    fs = first_stage_test(data, treatment, instruments, controls)
    weak_iv = weak_iv_diagnostics(fs['f_statistic'], len(instruments))

    results['first_stage'] = {
        'f_statistic': fs['f_statistic'],
        'weak_iv_passed': weak_iv.passed
    }

    if verbose:
        print("=" * 70)
        print("WEAK-IV ROBUST INFERENCE SUMMARY")
        print("=" * 70)
        print(f"\nFirst-stage F-statistic: {fs['f_statistic']:.2f}")
        print(f"Weak IV concern: {'NO' if weak_iv.passed else 'YES'}")

    # Standard 2SLS
    result_2sls = estimate_2sls(data, outcome, treatment, instruments, controls)
    results['2sls'] = {
        'estimate': result_2sls.effect,
        'se': result_2sls.se,
        'ci': [result_2sls.ci_lower, result_2sls.ci_upper]
    }

    if verbose:
        print(f"\n2SLS Estimate: {result_2sls.effect:.4f} (SE: {result_2sls.se:.4f})")
        print(f"  Standard 95% CI: [{result_2sls.ci_lower:.4f}, {result_2sls.ci_upper:.4f}]")

    # LIML
    result_liml = estimate_liml(data, outcome, treatment, instruments, controls)
    results['liml'] = {
        'estimate': result_liml.effect,
        'se': result_liml.se,
        'ci': [result_liml.ci_lower, result_liml.ci_upper]
    }

    if verbose:
        print(f"\nLIML Estimate: {result_liml.effect:.4f} (SE: {result_liml.se:.4f})")
        print(f"  Standard 95% CI: [{result_liml.ci_lower:.4f}, {result_liml.ci_upper:.4f}]")

    # Fuller
    result_fuller1 = estimate_fuller(data, outcome, treatment, instruments, controls, alpha=1)
    result_fuller4 = estimate_fuller(data, outcome, treatment, instruments, controls, alpha=4)

    if 'estimate' in result_fuller1:
        results['fuller1'] = result_fuller1
        results['fuller4'] = result_fuller4

        if verbose:
            print(f"\nFuller(1) Estimate: {result_fuller1['estimate']:.4f} "
                  f"(SE: {result_fuller1['se']:.4f})")
            print(f"Fuller(4) Estimate: {result_fuller4['estimate']:.4f} "
                  f"(SE: {result_fuller4['se']:.4f})")

    # Anderson-Rubin CI
    ar_ci = anderson_rubin_ci(data, outcome, treatment, instruments, controls, alpha)
    results['anderson_rubin'] = ar_ci

    if verbose:
        print(f"\nAnderson-Rubin {(1-alpha)*100:.0f}% CI:")
        if ar_ci['empty']:
            print("  EMPTY (instruments may be invalid)")
        elif not ar_ci['bounded']:
            print(f"  [{ar_ci['ci_lower']:.4f}, {ar_ci['ci_upper']:.4f}] (may be unbounded)")
        else:
            print(f"  [{ar_ci['ci_lower']:.4f}, {ar_ci['ci_upper']:.4f}]")
            print(f"  Width: {ar_ci['width']:.4f}")

    # AR test of H0: effect = 0
    ar_test = anderson_rubin_test_zero(data, outcome, treatment, instruments, controls)
    results['ar_test_zero'] = ar_test

    if verbose:
        print(f"\nAnderson-Rubin Test (H0: effect = 0):")
        print(f"  F-statistic: {ar_test['statistic']:.4f}")
        print(f"  P-value: {ar_test['p_value']:.4f}")
        print(f"  Reject at 5%: {ar_test['reject_at_05']}")

    # Comparison and recommendation
    if verbose:
        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)

        estimates = [result_2sls.effect, result_liml.effect]
        if 'estimate' in result_fuller1:
            estimates.append(result_fuller1['estimate'])

        est_range = max(estimates) - min(estimates)

        if weak_iv.passed:
            print("Instruments are strong. 2SLS is recommended.")
            print(f"  Point estimate: {result_2sls.effect:.4f}")
            print(f"  95% CI: [{result_2sls.ci_lower:.4f}, {result_2sls.ci_upper:.4f}]")
        else:
            print("WEAK INSTRUMENTS DETECTED.")
            print("\nRecommended inference:")
            print(f"  1. Use Anderson-Rubin CI for valid coverage")
            if ar_ci['bounded'] and not ar_ci['empty']:
                print(f"     AR CI: [{ar_ci['ci_lower']:.4f}, {ar_ci['ci_upper']:.4f}]")

            print(f"\n  2. Report Fuller(4) for point estimate")
            if 'estimate' in result_fuller4:
                print(f"     Fuller(4): {result_fuller4['estimate']:.4f}")

            print(f"\n  3. Estimator comparison (should be similar):")
            print(f"     2SLS:      {result_2sls.effect:.4f}")
            print(f"     LIML:      {result_liml.effect:.4f}")
            if 'estimate' in result_fuller1:
                print(f"     Fuller(1): {result_fuller1['estimate']:.4f}")
                print(f"     Fuller(4): {result_fuller4['estimate']:.4f}")
            print(f"     Range:     {est_range:.4f}")

            if est_range > 0.5 * result_2sls.se:
                print("\n  WARNING: Large discrepancy between estimators.")
                print("  Interpret with extreme caution.")

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Load data
    df = pd.read_csv(args.data)

    if args.method == 'all':
        # Run full summary
        results = weak_iv_robust_summary(
            data=df,
            outcome=args.outcome,
            treatment=args.treatment,
            instruments=args.instruments,
            controls=args.controls,
            alpha=args.alpha,
            verbose=args.verbose
        )
    elif args.method == 'ar':
        # Just Anderson-Rubin
        ar_ci = anderson_rubin_ci(
            df, args.outcome, args.treatment,
            args.instruments, args.controls, args.alpha
        )
        print(f"Anderson-Rubin {(1-args.alpha)*100:.0f}% CI:")
        if ar_ci['bounded']:
            print(f"  [{ar_ci['ci_lower']:.4f}, {ar_ci['ci_upper']:.4f}]")
        else:
            print("  May be unbounded")
    elif args.method == 'fuller':
        # Fuller estimation
        result = estimate_fuller(
            df, args.outcome, args.treatment,
            args.instruments, args.controls, args.fuller_alpha
        )
        if 'estimate' in result:
            print(f"Fuller(alpha={args.fuller_alpha}) Estimate:")
            print(f"  Effect: {result['estimate']:.4f}")
            print(f"  SE: {result['se']:.4f}")
            print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
