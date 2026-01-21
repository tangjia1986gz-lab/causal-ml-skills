#!/usr/bin/env python3
"""
Comprehensive Balance Testing Script for PSM

Implements multiple balance diagnostics following Imbens & Rubin (2015):
- Standardized Mean Difference (SMD)
- Variance Ratio
- Kolmogorov-Smirnov Test
- Omnibus Balance Tests
- Love Plot Generation

Usage:
    python test_balance.py --data matched.csv --treatment treat \\
                           --covariates age education income

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class BalanceStatistics:
    """Container for balance statistics of a single covariate."""
    covariate: str
    mean_treated: float
    mean_control: float
    var_treated: float
    var_control: float
    smd: float
    variance_ratio: float
    ks_statistic: float
    ks_pvalue: float
    balanced: bool


@dataclass
class OverallBalance:
    """Container for overall balance assessment."""
    n_covariates: int
    n_balanced: int
    pct_balanced: float
    max_smd: float
    mean_smd: float
    omnibus_statistic: float
    omnibus_pvalue: float
    all_balanced: bool


def calculate_weighted_stats(
    x: np.ndarray,
    weights: np.ndarray = None
) -> Tuple[float, float]:
    """
    Calculate weighted mean and variance.

    Parameters
    ----------
    x : np.ndarray
        Data values
    weights : np.ndarray, optional
        Observation weights

    Returns
    -------
    Tuple[float, float]
        (weighted mean, weighted variance)
    """
    if weights is None:
        weights = np.ones(len(x))

    # Remove NaN values
    mask = ~np.isnan(x)
    x = x[mask]
    weights = weights[mask]

    if len(x) == 0:
        return np.nan, np.nan

    # Weighted mean
    mean = np.average(x, weights=weights)

    # Weighted variance (Bessel correction for weighted samples)
    variance = np.average((x - mean) ** 2, weights=weights)

    return mean, variance


def calculate_smd(
    x_treated: np.ndarray,
    x_control: np.ndarray,
    weights_treated: np.ndarray = None,
    weights_control: np.ndarray = None
) -> float:
    """
    Calculate Standardized Mean Difference (SMD).

    SMD = (mean_T - mean_C) / sqrt((var_T + var_C) / 2)

    Parameters
    ----------
    x_treated : np.ndarray
        Covariate values for treated
    x_control : np.ndarray
        Covariate values for control
    weights_treated : np.ndarray, optional
        Weights for treated
    weights_control : np.ndarray, optional
        Weights for control

    Returns
    -------
    float
        Standardized mean difference
    """
    mean_t, var_t = calculate_weighted_stats(x_treated, weights_treated)
    mean_c, var_c = calculate_weighted_stats(x_control, weights_control)

    pooled_sd = np.sqrt((var_t + var_c) / 2)

    if pooled_sd == 0 or np.isnan(pooled_sd):
        return 0.0

    return (mean_t - mean_c) / pooled_sd


def calculate_variance_ratio(
    x_treated: np.ndarray,
    x_control: np.ndarray,
    weights_treated: np.ndarray = None,
    weights_control: np.ndarray = None
) -> float:
    """
    Calculate variance ratio.

    VR = var_T / var_C

    Parameters
    ----------
    x_treated : np.ndarray
        Covariate values for treated
    x_control : np.ndarray
        Covariate values for control
    weights_treated : np.ndarray, optional
        Weights for treated
    weights_control : np.ndarray, optional
        Weights for control

    Returns
    -------
    float
        Variance ratio
    """
    _, var_t = calculate_weighted_stats(x_treated, weights_treated)
    _, var_c = calculate_weighted_stats(x_control, weights_control)

    if var_c == 0 or np.isnan(var_c):
        return np.inf if var_t > 0 else 1.0

    return var_t / var_c


def kolmogorov_smirnov_test(
    x_treated: np.ndarray,
    x_control: np.ndarray
) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test for distributional equality.

    Parameters
    ----------
    x_treated : np.ndarray
        Covariate values for treated
    x_control : np.ndarray
        Covariate values for control

    Returns
    -------
    Tuple[float, float]
        (KS statistic, p-value)
    """
    # Remove NaN values
    x_t = x_treated[~np.isnan(x_treated)]
    x_c = x_control[~np.isnan(x_control)]

    if len(x_t) == 0 or len(x_c) == 0:
        return np.nan, np.nan

    result = stats.ks_2samp(x_t, x_c)
    return result.statistic, result.pvalue


def assess_covariate_balance(
    data: pd.DataFrame,
    treatment: str,
    covariate: str,
    weights: str = None,
    smd_threshold: float = 0.1,
    vr_lower: float = 0.5,
    vr_upper: float = 2.0
) -> BalanceStatistics:
    """
    Assess balance for a single covariate.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    treatment : str
        Treatment column name
    covariate : str
        Covariate column name
    weights : str, optional
        Weights column name
    smd_threshold : float
        SMD threshold for balance
    vr_lower : float
        Lower bound for variance ratio
    vr_upper : float
        Upper bound for variance ratio

    Returns
    -------
    BalanceStatistics
        Balance statistics for the covariate
    """
    treated_mask = data[treatment] == 1
    control_mask = data[treatment] == 0

    x_t = data.loc[treated_mask, covariate].values
    x_c = data.loc[control_mask, covariate].values

    if weights is not None and weights in data.columns:
        w_t = data.loc[treated_mask, weights].values
        w_c = data.loc[control_mask, weights].values
    else:
        w_t = None
        w_c = None

    # Calculate statistics
    mean_t, var_t = calculate_weighted_stats(x_t, w_t)
    mean_c, var_c = calculate_weighted_stats(x_c, w_c)
    smd = calculate_smd(x_t, x_c, w_t, w_c)
    vr = calculate_variance_ratio(x_t, x_c, w_t, w_c)
    ks_stat, ks_pval = kolmogorov_smirnov_test(x_t, x_c)

    # Assess balance
    smd_ok = abs(smd) < smd_threshold
    vr_ok = vr_lower <= vr <= vr_upper
    balanced = smd_ok and vr_ok

    return BalanceStatistics(
        covariate=covariate,
        mean_treated=mean_t,
        mean_control=mean_c,
        var_treated=var_t,
        var_control=var_c,
        smd=smd,
        variance_ratio=vr,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        balanced=balanced
    )


def omnibus_balance_test(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    weights: str = None
) -> Tuple[float, float]:
    """
    Omnibus test for overall balance using Hotelling's T-squared.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    treatment : str
        Treatment column
    covariates : List[str]
        Covariate columns
    weights : str, optional
        Weights column

    Returns
    -------
    Tuple[float, float]
        (T-squared statistic, p-value)
    """
    treated_mask = data[treatment] == 1
    control_mask = data[treatment] == 0

    X_t = data.loc[treated_mask, covariates].values
    X_c = data.loc[control_mask, covariates].values

    # Remove rows with NaN
    X_t = X_t[~np.isnan(X_t).any(axis=1)]
    X_c = X_c[~np.isnan(X_c).any(axis=1)]

    n_t, p = X_t.shape
    n_c = X_c.shape[0]

    if n_t < p + 1 or n_c < p + 1:
        return np.nan, np.nan

    # Means
    mean_t = X_t.mean(axis=0)
    mean_c = X_c.mean(axis=0)
    diff = mean_t - mean_c

    # Pooled covariance
    cov_t = np.cov(X_t, rowvar=False)
    cov_c = np.cov(X_c, rowvar=False)
    pooled_cov = ((n_t - 1) * cov_t + (n_c - 1) * cov_c) / (n_t + n_c - 2)

    # Regularize if needed
    if np.linalg.cond(pooled_cov) > 1e10:
        pooled_cov += np.eye(p) * 1e-6

    try:
        pooled_cov_inv = np.linalg.inv(pooled_cov)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    # T-squared statistic
    t2 = (n_t * n_c) / (n_t + n_c) * diff @ pooled_cov_inv @ diff

    # Convert to F-statistic
    f_stat = t2 * (n_t + n_c - p - 1) / (p * (n_t + n_c - 2))
    df1 = p
    df2 = n_t + n_c - p - 1

    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    return t2, p_value


def run_balance_tests(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    weights: str = None,
    smd_threshold: float = 0.1
) -> Tuple[List[BalanceStatistics], OverallBalance]:
    """
    Run comprehensive balance tests.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    treatment : str
        Treatment column
    covariates : List[str]
        Covariate columns
    weights : str, optional
        Weights column
    smd_threshold : float
        SMD threshold for balance

    Returns
    -------
    Tuple[List[BalanceStatistics], OverallBalance]
        (per-covariate results, overall assessment)
    """
    # Per-covariate tests
    covariate_results = []
    for cov in covariates:
        result = assess_covariate_balance(
            data, treatment, cov, weights, smd_threshold
        )
        covariate_results.append(result)

    # Overall assessment
    n_covariates = len(covariates)
    n_balanced = sum(1 for r in covariate_results if r.balanced)
    pct_balanced = n_balanced / n_covariates * 100 if n_covariates > 0 else 0

    smds = [r.smd for r in covariate_results]
    max_smd = max(abs(s) for s in smds)
    mean_smd = np.mean([abs(s) for s in smds])

    # Omnibus test
    t2, omnibus_pval = omnibus_balance_test(data, treatment, covariates, weights)

    overall = OverallBalance(
        n_covariates=n_covariates,
        n_balanced=n_balanced,
        pct_balanced=pct_balanced,
        max_smd=max_smd,
        mean_smd=mean_smd,
        omnibus_statistic=t2,
        omnibus_pvalue=omnibus_pval,
        all_balanced=n_balanced == n_covariates
    )

    return covariate_results, overall


def print_balance_table(
    covariate_results: List[BalanceStatistics],
    overall: OverallBalance
) -> str:
    """
    Create formatted balance table.

    Parameters
    ----------
    covariate_results : List[BalanceStatistics]
        Per-covariate results
    overall : OverallBalance
        Overall assessment

    Returns
    -------
    str
        Formatted table
    """
    lines = []
    lines.append("=" * 85)
    lines.append("COVARIATE BALANCE ASSESSMENT".center(85))
    lines.append("=" * 85)
    lines.append("")

    header = (
        f"{'Covariate':<15} {'Mean(T)':<10} {'Mean(C)':<10} "
        f"{'SMD':<10} {'VarRatio':<10} {'KS p-val':<10} {'Status':<10}"
    )
    lines.append(header)
    lines.append("-" * 85)

    for r in covariate_results:
        status = "OK" if r.balanced else "IMBALANCED"
        line = (
            f"{r.covariate:<15} {r.mean_treated:<10.3f} {r.mean_control:<10.3f} "
            f"{r.smd:<10.3f} {r.variance_ratio:<10.3f} {r.ks_pvalue:<10.3f} {status:<10}"
        )
        lines.append(line)

    lines.append("-" * 85)
    lines.append("")
    lines.append("OVERALL ASSESSMENT")
    lines.append("-" * 40)
    lines.append(f"Covariates balanced: {overall.n_balanced}/{overall.n_covariates} "
                 f"({overall.pct_balanced:.1f}%)")
    lines.append(f"Maximum |SMD|: {overall.max_smd:.4f}")
    lines.append(f"Mean |SMD|: {overall.mean_smd:.4f}")

    if not np.isnan(overall.omnibus_pvalue):
        lines.append(f"Omnibus test: T2 = {overall.omnibus_statistic:.2f}, "
                     f"p = {overall.omnibus_pvalue:.4f}")

    lines.append("")
    if overall.all_balanced:
        lines.append("RESULT: ALL COVARIATES BALANCED")
    else:
        lines.append("WARNING: BALANCE NOT ACHIEVED FOR ALL COVARIATES")

    lines.append("=" * 85)

    return "\n".join(lines)


def create_love_plot(
    covariate_results: List[BalanceStatistics],
    threshold: float = 0.1,
    figsize: Tuple[int, int] = (10, 8),
    output_path: str = None
) -> None:
    """
    Create Love plot showing SMD for each covariate.

    Parameters
    ----------
    covariate_results : List[BalanceStatistics]
        Per-covariate results
    threshold : float
        SMD threshold line
    figsize : tuple
        Figure size
    output_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    covariates = [r.covariate for r in covariate_results]
    smds = [abs(r.smd) for r in covariate_results]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(covariates))

    # Color based on balance
    colors = ['green' if s < threshold else 'red' for s in smds]

    ax.barh(y_pos, smds, color=colors, alpha=0.7)

    # Threshold line
    ax.axvline(x=threshold, color='black', linestyle='--',
               linewidth=2, label=f'Threshold ({threshold})')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates)
    ax.set_xlabel('Absolute Standardized Mean Difference')
    ax.set_title('Love Plot: Covariate Balance')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Love plot saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Balance Testing for PSM"
    )

    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to CSV data file"
    )
    parser.add_argument(
        "--treatment", "-t",
        type=str,
        required=True,
        help="Treatment column name"
    )
    parser.add_argument(
        "--covariates", "-x",
        type=str,
        nargs="+",
        required=True,
        help="Covariate column names"
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default=None,
        help="Weights column name (optional)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="SMD threshold for balance (default: 0.1)"
    )
    parser.add_argument(
        "--love-plot",
        type=str,
        default=None,
        help="Path to save Love plot (optional)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save results CSV (optional)"
    )

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data)
    print(f"Loaded data: {len(data)} observations")

    # Run balance tests
    covariate_results, overall = run_balance_tests(
        data=data,
        treatment=args.treatment,
        covariates=args.covariates,
        weights=args.weights,
        smd_threshold=args.threshold
    )

    # Print results
    table = print_balance_table(covariate_results, overall)
    print(table)

    # Create Love plot
    if args.love_plot:
        create_love_plot(covariate_results, args.threshold, output_path=args.love_plot)

    # Save results
    if args.output:
        results_df = pd.DataFrame([
            {
                'covariate': r.covariate,
                'mean_treated': r.mean_treated,
                'mean_control': r.mean_control,
                'smd': r.smd,
                'variance_ratio': r.variance_ratio,
                'ks_pvalue': r.ks_pvalue,
                'balanced': r.balanced
            }
            for r in covariate_results
        ])
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
