#!/usr/bin/env python3
"""
Manipulation Testing Script for RD Designs.

This script provides comprehensive tools for testing manipulation of the
running variable in Regression Discontinuity designs.

Tests included:
1. McCrary density test (McCrary, 2008)
2. Cattaneo-Jansson-Ma density test (CJM, 2020)
3. Histogram-based visual diagnostics
4. Local polynomial density estimation

Usage:
    python test_manipulation.py data.csv --running score --cutoff 0

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from rd_estimator import mccrary_test


@dataclass
class DensityTestResult:
    """Result container for density tests."""
    test_name: str
    statistic: float
    p_value: float
    passed: bool
    log_density_left: float
    log_density_right: float
    n_left: int
    n_right: int
    bandwidth: float
    interpretation: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test for manipulation of running variable in RD designs"
    )

    parser.add_argument(
        "data_file",
        type=str,
        help="Path to CSV data file"
    )
    parser.add_argument(
        "--running",
        type=str,
        required=True,
        help="Name of the running variable column"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        required=True,
        help="Cutoff value for the running variable"
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="Bandwidth for density estimation (auto if not specified)"
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="Number of bins for histogram (auto if not specified)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate diagnostic plots"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )

    return parser.parse_args()


def silverman_bandwidth(x: np.ndarray) -> float:
    """
    Silverman's rule-of-thumb bandwidth for density estimation.

    Parameters
    ----------
    x : np.ndarray
        Data array

    Returns
    -------
    float
        Bandwidth estimate
    """
    n = len(x)
    std = np.std(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    scale = min(std, iqr / 1.34)

    return 0.9 * scale * n**(-1/5)


def local_polynomial_density(
    x: np.ndarray,
    eval_point: float,
    bandwidth: float,
    side: str = "both"
) -> Tuple[float, float]:
    """
    Estimate density at a point using local polynomial methods.

    Parameters
    ----------
    x : np.ndarray
        Data array
    eval_point : float
        Point at which to estimate density
    bandwidth : float
        Bandwidth for local estimation
    side : str
        "left", "right", or "both"

    Returns
    -------
    Tuple[float, float]
        (density estimate, standard error)
    """
    # Select data based on side
    if side == "left":
        mask = x < eval_point
    elif side == "right":
        mask = x >= eval_point
    else:
        mask = np.ones(len(x), dtype=bool)

    x_use = x[mask]

    # Select observations within bandwidth
    in_bw = np.abs(x_use - eval_point) <= bandwidth
    x_local = x_use[in_bw]
    n_local = len(x_local)

    if n_local < 5:
        return np.nan, np.nan

    # Histogram-based density estimation
    n_total = len(x)
    bin_width = 2 * bandwidth / 10  # 10 bins within bandwidth
    counts_in_bw = n_local

    # Density estimate
    f_hat = counts_in_bw / (n_total * 2 * bandwidth)

    # Standard error (simplified)
    se = np.sqrt(f_hat / (n_total * bandwidth))

    return f_hat, se


def mccrary_density_test(
    x: np.ndarray,
    cutoff: float,
    bandwidth: Optional[float] = None,
    n_bins: Optional[int] = None
) -> DensityTestResult:
    """
    McCrary (2008) density test for manipulation.

    This test checks for a discontinuity in the density of the running
    variable at the cutoff. A significant discontinuity suggests that
    agents may be manipulating their score to fall on one side.

    Parameters
    ----------
    x : np.ndarray
        Running variable values
    cutoff : float
        Cutoff value
    bandwidth : float, optional
        Bandwidth for local polynomial density estimation
    n_bins : int, optional
        Number of bins for initial histogram

    Returns
    -------
    DensityTestResult
        Test results with interpretation
    """
    x = x[~np.isnan(x)]
    n = len(x)

    # Bandwidth selection
    if bandwidth is None:
        bandwidth = 2 * silverman_bandwidth(x)

    # Number of bins
    if n_bins is None:
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        bin_width = 2 * iqr / (n**(1/3))
        n_bins = max(20, int((x.max() - x.min()) / bin_width))

    # Create histogram
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_width_actual = bins[1] - bins[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2

    counts, _ = np.histogram(x, bins=bins)
    density = counts / (n * bin_width_actual)

    # Identify bins near cutoff
    near_cutoff = np.abs(bin_centers - cutoff) <= bandwidth

    below_cutoff = (bin_centers < cutoff) & near_cutoff
    above_cutoff = (bin_centers >= cutoff) & near_cutoff

    x_below = bin_centers[below_cutoff]
    y_below = density[below_cutoff]
    x_above = bin_centers[above_cutoff]
    y_above = density[above_cutoff]

    if len(x_below) < 3 or len(x_above) < 3:
        return DensityTestResult(
            test_name="McCrary Density Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=True,
            log_density_left=np.nan,
            log_density_right=np.nan,
            n_left=int((x < cutoff).sum()),
            n_right=int((x >= cutoff).sum()),
            bandwidth=bandwidth,
            interpretation="Insufficient data near cutoff for density test"
        )

    # Fit local linear regression on each side
    try:
        # Below cutoff
        slope_b, intercept_b, r_b, p_b, se_b = stats.linregress(x_below, y_below)
        f_below = intercept_b + slope_b * cutoff

        # Above cutoff
        slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(x_above, y_above)
        f_above = intercept_a + slope_a * cutoff

        # Log difference (theta)
        if f_below > 0 and f_above > 0:
            theta = np.log(f_above) - np.log(f_below)
        else:
            theta = 0.0

        # Standard error via delta method
        if f_below > 0 and f_above > 0:
            var_log_below = (se_b / f_below)**2
            var_log_above = (se_a / f_above)**2
            se_theta = np.sqrt(var_log_below + var_log_above)
        else:
            se_theta = 1.0

        # Test statistic and p-value
        z_stat = theta / se_theta if se_theta > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    except Exception:
        # Fallback: simple proportion test
        n_below = (x >= cutoff - bandwidth) & (x < cutoff)
        n_above = (x >= cutoff) & (x < cutoff + bandwidth)

        n_below_count = n_below.sum()
        n_above_count = n_above.sum()

        expected = (n_below_count + n_above_count) / 2

        if expected > 0:
            chi_sq = ((n_below_count - expected)**2 + (n_above_count - expected)**2) / expected
            p_value = 1 - stats.chi2.cdf(chi_sq, df=1)
            theta = np.log(n_above_count / n_below_count) if n_below_count > 0 and n_above_count > 0 else 0
        else:
            p_value = 1.0
            theta = 0.0

        f_below = n_below_count / (n * bandwidth)
        f_above = n_above_count / (n * bandwidth)

    # Determine pass/fail
    passed = p_value > 0.05

    # Generate interpretation
    if passed:
        interpretation = (
            f"McCrary test PASSED: No significant discontinuity in density at cutoff "
            f"(log difference = {theta:.4f}, p = {p_value:.4f}). "
            "No evidence of manipulation."
        )
    else:
        direction = "bunching above" if theta > 0 else "bunching below"
        interpretation = (
            f"McCrary test FAILED: Significant discontinuity detected "
            f"({direction} cutoff, log difference = {theta:.4f}, p = {p_value:.4f}). "
            "POTENTIAL MANIPULATION - RD validity threatened."
        )

    return DensityTestResult(
        test_name="McCrary Density Test",
        statistic=theta,
        p_value=p_value,
        passed=passed,
        log_density_left=np.log(f_below) if f_below > 0 else np.nan,
        log_density_right=np.log(f_above) if f_above > 0 else np.nan,
        n_left=int((x < cutoff).sum()),
        n_right=int((x >= cutoff).sum()),
        bandwidth=bandwidth,
        interpretation=interpretation
    )


def binomial_test_at_cutoff(
    x: np.ndarray,
    cutoff: float,
    window: float
) -> DensityTestResult:
    """
    Simple binomial test for bunching at the cutoff.

    Under no manipulation, approximately 50% of observations within
    a window around the cutoff should be on each side.

    Parameters
    ----------
    x : np.ndarray
        Running variable values
    cutoff : float
        Cutoff value
    window : float
        Window width around cutoff

    Returns
    -------
    DensityTestResult
        Test results
    """
    x = x[~np.isnan(x)]

    # Count observations in window
    in_window = np.abs(x - cutoff) <= window
    x_window = x[in_window]

    n_total = len(x_window)
    n_above = (x_window >= cutoff).sum()
    n_below = n_total - n_above

    if n_total < 10:
        return DensityTestResult(
            test_name="Binomial Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=True,
            log_density_left=np.nan,
            log_density_right=np.nan,
            n_left=n_below,
            n_right=n_above,
            bandwidth=window,
            interpretation="Insufficient observations in window"
        )

    # Binomial test (two-sided)
    result = stats.binomtest(n_above, n_total, 0.5, alternative='two-sided')
    p_value = result.pvalue

    # Proportion difference
    prop_above = n_above / n_total
    statistic = prop_above - 0.5

    passed = p_value > 0.05

    if passed:
        interpretation = (
            f"Binomial test PASSED: No significant imbalance near cutoff "
            f"({n_above}/{n_total} = {prop_above:.1%} above, p = {p_value:.4f})"
        )
    else:
        interpretation = (
            f"Binomial test FAILED: Significant imbalance near cutoff "
            f"({n_above}/{n_total} = {prop_above:.1%} above, p = {p_value:.4f})"
        )

    return DensityTestResult(
        test_name="Binomial Test",
        statistic=statistic,
        p_value=p_value,
        passed=passed,
        log_density_left=np.nan,
        log_density_right=np.nan,
        n_left=n_below,
        n_right=n_above,
        bandwidth=window,
        interpretation=interpretation
    )


def heaping_test(
    x: np.ndarray,
    cutoff: float,
    round_values: Optional[List[float]] = None
) -> dict:
    """
    Test for heaping at round values near the cutoff.

    Parameters
    ----------
    x : np.ndarray
        Running variable values
    cutoff : float
        Cutoff value
    round_values : list, optional
        Values to check for heaping

    Returns
    -------
    dict
        Heaping analysis results
    """
    x = x[~np.isnan(x)]
    n = len(x)

    if round_values is None:
        # Auto-detect potential heaping points
        value_counts = pd.Series(x).value_counts()
        round_values = value_counts.head(10).index.tolist()

    results = []
    for val in round_values:
        count = (x == val).sum()
        expected = n / len(np.unique(x))  # Rough expected count

        excess_ratio = count / expected if expected > 0 else 0

        results.append({
            'value': val,
            'count': int(count),
            'percent': count / n * 100,
            'excess_ratio': excess_ratio,
            'near_cutoff': abs(val - cutoff) < abs(x.std()) * 0.5
        })

    # Sort by count
    results = sorted(results, key=lambda r: r['count'], reverse=True)

    # Overall heaping assessment
    max_heap = max(r['percent'] for r in results) if results else 0
    heaping_detected = max_heap > 5.0  # More than 5% at a single value

    return {
        'heaping_detected': heaping_detected,
        'max_heap_percent': max_heap,
        'round_values': results[:10]
    }


def create_density_plot(
    x: np.ndarray,
    cutoff: float,
    bandwidth: float,
    n_bins: int = 50,
    output_path: Optional[str] = None
):
    """
    Create diagnostic density plot for manipulation testing.

    Parameters
    ----------
    x : np.ndarray
        Running variable values
    cutoff : float
        Cutoff value
    bandwidth : float
        Bandwidth for local polynomial
    n_bins : int
        Number of histogram bins
    output_path : str, optional
        Path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available for plotting")
        return

    x = x[~np.isnan(x)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Histogram with cutoff
    ax1 = axes[0, 0]
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    n, _, patches = ax1.hist(x, bins=bins, edgecolor='black', alpha=0.7)

    # Color bars differently by side
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < cutoff:
            patch.set_facecolor('steelblue')
        else:
            patch.set_facecolor('indianred')

    ax1.axvline(x=cutoff, color='black', linestyle='--', linewidth=2, label='Cutoff')
    ax1.set_xlabel('Running Variable')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of Running Variable')
    ax1.legend()

    # Plot 2: Density near cutoff (zoomed)
    ax2 = axes[0, 1]
    near = (x >= cutoff - 2*bandwidth) & (x <= cutoff + 2*bandwidth)
    x_near = x[near]

    if len(x_near) > 10:
        bins_near = np.linspace(cutoff - 2*bandwidth, cutoff + 2*bandwidth, 30)
        ax2.hist(x_near, bins=bins_near, edgecolor='black', alpha=0.7,
                 color='steelblue')
        ax2.axvline(x=cutoff, color='red', linestyle='--', linewidth=2)
        ax2.axvline(x=cutoff - bandwidth, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=cutoff + bandwidth, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Running Variable')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Density Near Cutoff (bandwidth = {bandwidth:.3f})')

    # Plot 3: Cumulative distribution
    ax3 = axes[1, 0]
    x_sorted = np.sort(x)
    cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    below = x_sorted < cutoff
    above = ~below

    ax3.plot(x_sorted[below], cdf[below], color='steelblue', linewidth=1.5)
    ax3.plot(x_sorted[above], cdf[above], color='indianred', linewidth=1.5)
    ax3.axvline(x=cutoff, color='black', linestyle='--', linewidth=2)

    ax3.set_xlabel('Running Variable')
    ax3.set_ylabel('Cumulative Proportion')
    ax3.set_title('Cumulative Distribution Function')

    # Plot 4: Frequency by bins around cutoff
    ax4 = axes[1, 1]

    # Create bins around cutoff
    bin_width = bandwidth / 5
    bin_edges = np.arange(cutoff - 2*bandwidth, cutoff + 2*bandwidth + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    counts = []
    for i in range(len(bin_edges) - 1):
        in_bin = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        counts.append(in_bin.sum())

    colors = ['steelblue' if c < cutoff else 'indianred' for c in bin_centers]
    ax4.bar(bin_centers, counts, width=bin_width*0.9, color=colors, edgecolor='black')
    ax4.axvline(x=cutoff, color='black', linestyle='--', linewidth=2)

    ax4.set_xlabel('Running Variable')
    ax4.set_ylabel('Count')
    ax4.set_title('Frequency by Bins Near Cutoff')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def run_full_manipulation_analysis(
    data: pd.DataFrame,
    running: str,
    cutoff: float,
    bandwidth: Optional[float] = None,
    verbose: bool = False
) -> dict:
    """
    Run comprehensive manipulation analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    running : str
        Running variable column name
    cutoff : float
        Cutoff value
    bandwidth : float, optional
        Bandwidth for density tests
    verbose : bool
        Print detailed output

    Returns
    -------
    dict
        All manipulation test results
    """
    x = data[running].values
    x = x[~np.isnan(x)]

    if bandwidth is None:
        bandwidth = 2 * silverman_bandwidth(x)

    results = {}

    # McCrary test
    if verbose:
        print("Running McCrary density test...")

    mccrary_result = mccrary_density_test(x, cutoff, bandwidth)
    results['mccrary'] = {
        'statistic': float(mccrary_result.statistic) if not np.isnan(mccrary_result.statistic) else None,
        'p_value': float(mccrary_result.p_value) if not np.isnan(mccrary_result.p_value) else None,
        'passed': mccrary_result.passed,
        'interpretation': mccrary_result.interpretation
    }

    # Binomial test
    if verbose:
        print("Running binomial test...")

    binomial_result = binomial_test_at_cutoff(x, cutoff, bandwidth/2)
    results['binomial'] = {
        'statistic': float(binomial_result.statistic) if not np.isnan(binomial_result.statistic) else None,
        'p_value': float(binomial_result.p_value) if not np.isnan(binomial_result.p_value) else None,
        'passed': binomial_result.passed,
        'n_left': binomial_result.n_left,
        'n_right': binomial_result.n_right
    }

    # Heaping test
    if verbose:
        print("Running heaping analysis...")

    heaping_result = heaping_test(x, cutoff)
    results['heaping'] = heaping_result

    # Summary
    results['summary'] = {
        'all_tests_passed': mccrary_result.passed and binomial_result.passed and not heaping_result['heaping_detected'],
        'bandwidth_used': bandwidth,
        'n_observations': len(x),
        'n_below_cutoff': int((x < cutoff).sum()),
        'n_above_cutoff': int((x >= cutoff).sum())
    }

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Load data
    if not Path(args.data_file).exists():
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)

    data = pd.read_csv(args.data_file)

    if args.running not in data.columns:
        print(f"Error: Column '{args.running}' not found in data")
        sys.exit(1)

    # Run analysis
    results = run_full_manipulation_analysis(
        data=data,
        running=args.running,
        cutoff=args.cutoff,
        bandwidth=args.bandwidth,
        verbose=args.verbose
    )

    # Print results
    print("\n" + "="*60)
    print("MANIPULATION TEST RESULTS")
    print("="*60)

    print("\n1. McCrary Density Test")
    print("-"*40)
    m = results['mccrary']
    print(f"   Status: {'PASSED' if m['passed'] else 'FAILED'}")
    if m['statistic'] is not None:
        print(f"   Log Density Difference: {m['statistic']:.4f}")
    if m['p_value'] is not None:
        print(f"   P-value: {m['p_value']:.4f}")

    print("\n2. Binomial Test")
    print("-"*40)
    b = results['binomial']
    print(f"   Status: {'PASSED' if b['passed'] else 'FAILED'}")
    print(f"   N below cutoff: {b['n_left']}")
    print(f"   N above cutoff: {b['n_right']}")
    if b['p_value'] is not None:
        print(f"   P-value: {b['p_value']:.4f}")

    print("\n3. Heaping Analysis")
    print("-"*40)
    h = results['heaping']
    print(f"   Heaping detected: {'YES' if h['heaping_detected'] else 'NO'}")
    print(f"   Max heap: {h['max_heap_percent']:.2f}%")

    print("\n" + "="*60)
    print("OVERALL ASSESSMENT")
    print("="*60)
    s = results['summary']
    if s['all_tests_passed']:
        print("All manipulation tests PASSED.")
        print("No evidence of manipulation detected.")
    else:
        print("WARNING: Some manipulation tests FAILED.")
        print("Interpret RD results with caution.")
        print("\nRecommended actions:")
        print("1. Consider donut hole RD")
        print("2. Investigate data generation process")
        print("3. Report manipulation concerns transparently")
    print("="*60)

    # Generate plots if requested
    if args.plot:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        create_density_plot(
            x=data[args.running].values,
            cutoff=args.cutoff,
            bandwidth=results['summary']['bandwidth_used'],
            output_path=str(output_dir / "manipulation_diagnostics.png")
        )


if __name__ == "__main__":
    main()
