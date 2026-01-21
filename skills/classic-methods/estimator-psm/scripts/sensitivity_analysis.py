#!/usr/bin/env python3
"""
Rosenbaum Bounds Sensitivity Analysis Script

Implements sensitivity analysis for propensity score matching studies
following Rosenbaum (2002). Assesses how robust treatment effect estimates
are to potential hidden bias from unobserved confounders.

Key Concept:
- Gamma represents the degree of departure from random assignment
- Gamma = 1: No hidden bias (pure random assignment within strata)
- Gamma = 2: Unobserved confounder could double the odds of treatment
- Critical Gamma: The point where the treatment effect becomes insignificant

Usage:
    python sensitivity_analysis.py --data matched.csv --outcome y --treatment treat \\
                                   --gamma-max 3.0 --output sensitivity.png

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class SensitivityBounds:
    """Bounds on treatment effect under hidden bias."""
    gamma: float
    effect_lower: float
    effect_upper: float
    p_value_lower: float
    p_value_upper: float


@dataclass
class SensitivityAnalysis:
    """Complete sensitivity analysis results."""
    bounds: List[SensitivityBounds]
    critical_gamma: float
    interpretation: str
    effect_at_gamma_1: float
    se_at_gamma_1: float


def rosenbaum_bounds_sign_test(
    diff: np.ndarray,
    gamma: float
) -> Tuple[float, float]:
    """
    Compute Rosenbaum bounds using sign-based test.

    Under hidden bias, the probability that a treated unit has a higher
    outcome can range from 1/(1+gamma) to gamma/(1+gamma).

    Parameters
    ----------
    diff : np.ndarray
        Outcome differences (treated - matched control)
    gamma : float
        Sensitivity parameter (odds ratio)

    Returns
    -------
    Tuple[float, float]
        (lower p-value, upper p-value)
    """
    n = len(diff)
    n_positive = np.sum(diff > 0)

    # Under hidden bias, probability of positive difference
    p_low = 1 / (1 + gamma)
    p_high = gamma / (1 + gamma)

    # Expected number of positive differences under different scenarios
    expected_low = n * p_low
    expected_high = n * p_high

    # Variance under each scenario
    var_low = n * p_low * (1 - p_low)
    var_high = n * p_high * (1 - p_high)

    # Z-statistics
    if var_low > 0:
        z_upper = (n_positive - expected_low) / np.sqrt(var_low)
    else:
        z_upper = 0

    if var_high > 0:
        z_lower = (n_positive - expected_high) / np.sqrt(var_high)
    else:
        z_lower = 0

    # One-sided p-values for positive effect
    p_lower = 1 - stats.norm.cdf(z_lower)
    p_upper = 1 - stats.norm.cdf(z_upper)

    return p_lower, p_upper


def rosenbaum_bounds_wilcoxon(
    diff: np.ndarray,
    gamma: float
) -> Tuple[float, float, float, float]:
    """
    Compute Rosenbaum bounds using Wilcoxon signed-rank test.

    More powerful than sign test as it uses rank information.

    Parameters
    ----------
    diff : np.ndarray
        Outcome differences (treated - matched control)
    gamma : float
        Sensitivity parameter (odds ratio)

    Returns
    -------
    Tuple[float, float, float, float]
        (effect_lower, effect_upper, p_lower, p_upper)
    """
    n = len(diff)

    # Compute ranks of absolute differences
    abs_diff = np.abs(diff)
    ranks = stats.rankdata(abs_diff)

    # Sum of ranks for positive differences
    W_obs = np.sum(ranks[diff > 0])

    # Under no hidden bias (gamma=1)
    if gamma == 1.0:
        # Standard Wilcoxon test
        _, p_value = stats.wilcoxon(diff, alternative='greater')
        effect = np.median(diff)
        return effect, effect, p_value, p_value

    # Under hidden bias
    # Probability bounds
    p_low = 1 / (1 + gamma)
    p_high = gamma / (1 + gamma)

    # Expected value bounds for W under hidden bias
    E_low = np.sum(ranks) * p_low
    E_high = np.sum(ranks) * p_high

    # Variance (approximate)
    var_W = np.sum(ranks ** 2) * p_high * (1 - p_high)

    # Z-statistics
    z_lower = (W_obs - E_high) / np.sqrt(var_W) if var_W > 0 else 0
    z_upper = (W_obs - E_low) / np.sqrt(var_W) if var_W > 0 else 0

    # P-values
    p_lower = 1 - stats.norm.cdf(z_lower)
    p_upper = 1 - stats.norm.cdf(z_upper)

    # Effect bounds (Hodges-Lehmann approach)
    # Lower bound: pessimistic assignment
    # Upper bound: optimistic assignment
    sorted_diff = np.sort(diff)
    n = len(diff)

    idx_low = int(n * p_low)
    idx_high = int(n * p_high)
    idx_low = max(0, min(idx_low, n - 1))
    idx_high = max(0, min(idx_high, n - 1))

    effect_lower = sorted_diff[idx_low]
    effect_upper = sorted_diff[idx_high]

    return effect_lower, effect_upper, p_lower, p_upper


def run_sensitivity_analysis(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    gamma_values: List[float] = None,
    method: str = 'wilcoxon'
) -> SensitivityAnalysis:
    """
    Run complete Rosenbaum bounds sensitivity analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Matched dataset
    outcome : str
        Outcome column name
    treatment : str
        Treatment column name
    gamma_values : List[float], optional
        Gamma values to test (default: 1.0 to 3.0)
    method : str
        Test method: 'wilcoxon' or 'sign'

    Returns
    -------
    SensitivityAnalysis
        Complete sensitivity analysis results
    """
    if gamma_values is None:
        gamma_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

    # Extract outcomes
    y_treated = data.loc[data[treatment] == 1, outcome].values
    y_control = data.loc[data[treatment] == 0, outcome].values

    # Compute paired differences (assuming matched order)
    n_pairs = min(len(y_treated), len(y_control))
    diff = y_treated[:n_pairs] - y_control[:n_pairs]

    # Remove pairs with missing outcomes
    valid = ~(np.isnan(diff))
    diff = diff[valid]

    bounds_list = []

    for gamma in gamma_values:
        if method == 'wilcoxon':
            eff_low, eff_up, p_low, p_up = rosenbaum_bounds_wilcoxon(diff, gamma)
        else:  # sign test
            p_low, p_up = rosenbaum_bounds_sign_test(diff, gamma)
            sorted_diff = np.sort(diff)
            n = len(diff)
            p_l = 1 / (1 + gamma)
            p_h = gamma / (1 + gamma)
            eff_low = np.percentile(diff, p_l * 100)
            eff_up = np.percentile(diff, p_h * 100)

        bounds_list.append(SensitivityBounds(
            gamma=gamma,
            effect_lower=eff_low,
            effect_upper=eff_up,
            p_value_lower=p_low,
            p_value_upper=p_up
        ))

    # Find critical gamma (where upper p-value first exceeds 0.05)
    critical_gamma = gamma_values[-1]
    for b in bounds_list:
        if b.p_value_upper > 0.05:
            critical_gamma = b.gamma
            break

    # Effect at gamma = 1
    eff_1 = bounds_list[0].effect_lower
    se_1 = np.std(diff) / np.sqrt(len(diff))

    # Interpretation
    if critical_gamma <= 1.0:
        interpretation = (
            "Results are NOT robust to hidden bias. Even small departures "
            "from random assignment could explain the observed effect."
        )
    elif critical_gamma < 1.5:
        interpretation = (
            f"Results are WEAKLY robust to hidden bias (critical Gamma = {critical_gamma:.2f}). "
            "A moderate unobserved confounder could explain the effect."
        )
    elif critical_gamma < 2.0:
        interpretation = (
            f"Results are MODERATELY robust to hidden bias (critical Gamma = {critical_gamma:.2f}). "
            "A substantial unobserved confounder would be needed."
        )
    else:
        interpretation = (
            f"Results are STRONGLY robust to hidden bias (critical Gamma = {critical_gamma:.2f}). "
            "A very strong unobserved confounder would be needed to explain the effect."
        )

    return SensitivityAnalysis(
        bounds=bounds_list,
        critical_gamma=critical_gamma,
        interpretation=interpretation,
        effect_at_gamma_1=eff_1,
        se_at_gamma_1=se_1
    )


def print_sensitivity_table(analysis: SensitivityAnalysis) -> str:
    """
    Create formatted sensitivity analysis table.

    Parameters
    ----------
    analysis : SensitivityAnalysis
        Analysis results

    Returns
    -------
    str
        Formatted table
    """
    lines = []
    lines.append("=" * 75)
    lines.append("ROSENBAUM BOUNDS SENSITIVITY ANALYSIS")
    lines.append("=" * 75)
    lines.append("")

    lines.append(f"Effect at Gamma=1: {analysis.effect_at_gamma_1:.4f} "
                 f"(SE = {analysis.se_at_gamma_1:.4f})")
    lines.append(f"Critical Gamma: {analysis.critical_gamma:.2f}")
    lines.append("")

    header = (
        f"{'Gamma':>8} {'Effect Lower':>14} {'Effect Upper':>14} "
        f"{'P-value Lower':>14} {'P-value Upper':>14}"
    )
    lines.append(header)
    lines.append("-" * 75)

    for b in analysis.bounds:
        # Mark critical gamma row
        marker = " <--" if abs(b.gamma - analysis.critical_gamma) < 0.01 else ""
        line = (
            f"{b.gamma:>8.2f} {b.effect_lower:>14.4f} {b.effect_upper:>14.4f} "
            f"{b.p_value_lower:>14.4f} {b.p_value_upper:>14.4f}{marker}"
        )
        lines.append(line)

    lines.append("-" * 75)
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 40)
    lines.append(analysis.interpretation)
    lines.append("")

    # Add explanation
    lines.append("NOTES")
    lines.append("-" * 40)
    lines.append("Gamma: Odds ratio of treatment due to unobserved confounder")
    lines.append("  Gamma=1.0: No hidden bias (unconfoundedness holds)")
    lines.append("  Gamma=2.0: Unobserved factor could double treatment odds")
    lines.append("Effect bounds: Range of possible treatment effects under bias")
    lines.append("P-value upper: Most conservative p-value under hidden bias")
    lines.append("")
    lines.append("=" * 75)

    return "\n".join(lines)


def plot_sensitivity_analysis(
    analysis: SensitivityAnalysis,
    figsize: Tuple[int, int] = (12, 8),
    output_path: str = None
) -> None:
    """
    Create sensitivity analysis visualization.

    Parameters
    ----------
    analysis : SensitivityAnalysis
        Analysis results
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

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    gammas = [b.gamma for b in analysis.bounds]
    effect_lower = [b.effect_lower for b in analysis.bounds]
    effect_upper = [b.effect_upper for b in analysis.bounds]
    p_upper = [b.p_value_upper for b in analysis.bounds]

    # Panel 1: Effect bounds
    ax1 = axes[0]
    ax1.fill_between(gammas, effect_lower, effect_upper, alpha=0.3, color='blue')
    ax1.plot(gammas, effect_lower, 'b-', linewidth=2, label='Lower bound')
    ax1.plot(gammas, effect_upper, 'b-', linewidth=2, label='Upper bound')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Null effect')
    ax1.axvline(x=analysis.critical_gamma, color='green', linestyle=':',
                linewidth=2, label=f'Critical Gamma = {analysis.critical_gamma:.2f}')

    ax1.set_xlabel('Gamma (Sensitivity Parameter)', fontsize=12)
    ax1.set_ylabel('Treatment Effect', fontsize=12)
    ax1.set_title('Effect Bounds Under Hidden Bias', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Panel 2: P-value bounds
    ax2 = axes[1]
    ax2.plot(gammas, p_upper, 'b-o', linewidth=2, markersize=6,
             label='Upper bound p-value')
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5,
                label='Significance threshold (0.05)')
    ax2.axvline(x=analysis.critical_gamma, color='green', linestyle=':',
                linewidth=2, label=f'Critical Gamma = {analysis.critical_gamma:.2f}')

    ax2.set_xlabel('Gamma (Sensitivity Parameter)', fontsize=12)
    ax2.set_ylabel('P-value (Upper Bound)', fontsize=12)
    ax2.set_title('P-value Bounds Under Hidden Bias', fontsize=14)
    ax2.set_ylim(0, min(1, max(p_upper) * 1.2))
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Rosenbaum Bounds Sensitivity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rosenbaum Bounds Sensitivity Analysis"
    )

    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to matched CSV data file"
    )
    parser.add_argument(
        "--outcome", "-y",
        type=str,
        required=True,
        help="Outcome column name"
    )
    parser.add_argument(
        "--treatment", "-t",
        type=str,
        required=True,
        help="Treatment column name"
    )
    parser.add_argument(
        "--gamma-max",
        type=float,
        default=3.0,
        help="Maximum gamma to test (default: 3.0)"
    )
    parser.add_argument(
        "--gamma-step",
        type=float,
        default=0.25,
        help="Step size for gamma values (default: 0.25)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=['wilcoxon', 'sign'],
        default='wilcoxon',
        help="Test method (default: wilcoxon)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for figure"
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Output path for JSON results"
    )

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data)
    print(f"Loaded data: {len(data)} observations")

    # Generate gamma values
    gamma_values = list(np.arange(1.0, args.gamma_max + args.gamma_step, args.gamma_step))

    # Run analysis
    analysis = run_sensitivity_analysis(
        data=data,
        outcome=args.outcome,
        treatment=args.treatment,
        gamma_values=gamma_values,
        method=args.method
    )

    # Print table
    table = print_sensitivity_table(analysis)
    print(table)

    # Create plot
    if args.output:
        plot_sensitivity_analysis(analysis, output_path=args.output)

    # Save JSON results
    if args.json_output:
        import json
        results = {
            'critical_gamma': analysis.critical_gamma,
            'effect_at_gamma_1': analysis.effect_at_gamma_1,
            'se_at_gamma_1': analysis.se_at_gamma_1,
            'interpretation': analysis.interpretation,
            'bounds': [
                {
                    'gamma': b.gamma,
                    'effect_lower': b.effect_lower,
                    'effect_upper': b.effect_upper,
                    'p_value_lower': b.p_value_lower,
                    'p_value_upper': b.p_value_upper
                }
                for b in analysis.bounds
            ]
        }
        with open(args.json_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.json_output}")


if __name__ == "__main__":
    main()
