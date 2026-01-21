#!/usr/bin/env python3
"""
Imai Sensitivity Analysis for Causal Mediation

Performs sensitivity analysis to assess robustness of mediation findings
to unmeasured confounding between mediator and outcome.

Implements the methodology from:
- Imai, K., Keele, L., & Yamamoto, T. (2010). Identification, Inference
  and Sensitivity Analysis for Causal Mediation Effects. Statistical Science.

Features:
- Computes ACME under range of sensitivity parameter (rho) values
- Finds breakpoint where ACME = 0
- Generates sensitivity plots
- Provides R-squared interpretations

Usage:
    python sensitivity_analysis.py data.csv \
        --outcome earnings --treatment training --mediator skills \
        --controls age education --plot sensitivity.png

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from mediation_estimator import (
    estimate_baron_kenny,
    create_mediation_data,
    sensitivity_analysis_mediation
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Sensitivity analysis for causal mediation (Imai et al. 2010)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sensitivity analysis
  python sensitivity_analysis.py data.csv -y earnings -d training -m skills

  # With custom rho range and plot
  python sensitivity_analysis.py data.csv -y earnings -d training -m skills \\
      --rho-min -0.7 --rho-max 0.7 --rho-step 0.02 --plot sensitivity.png

  # Output to JSON
  python sensitivity_analysis.py data.csv -y earnings -d training -m skills \\
      --output sensitivity_results.json
        """
    )

    parser.add_argument('data_file', type=str, help='Path to CSV data file')

    # Core variables
    parser.add_argument('-y', '--outcome', type=str, required=True,
                        help='Outcome variable name')
    parser.add_argument('-d', '--treatment', type=str, required=True,
                        help='Treatment variable name')
    parser.add_argument('-m', '--mediator', type=str, required=True,
                        help='Mediator variable name')
    parser.add_argument('--controls', type=str, nargs='*', default=[],
                        help='Control variable names')

    # Sensitivity parameter range
    parser.add_argument('--rho-min', type=float, default=-0.9,
                        help='Minimum rho value (default: -0.9)')
    parser.add_argument('--rho-max', type=float, default=0.9,
                        help='Maximum rho value (default: 0.9)')
    parser.add_argument('--rho-step', type=float, default=0.05,
                        help='Step size for rho (default: 0.05)')

    # Output options
    parser.add_argument('--plot', type=str, default=None,
                        help='Output file for sensitivity plot')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file for JSON results')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    return parser.parse_args()


def comprehensive_sensitivity_analysis(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: List[str],
    rho_range: np.ndarray
) -> Dict:
    """
    Comprehensive sensitivity analysis following Imai et al. (2010).

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    outcome, treatment, mediator : str
        Variable names
    controls : List[str]
        Control variable names
    rho_range : np.ndarray
        Range of sensitivity parameter values

    Returns
    -------
    Dict with complete sensitivity analysis results
    """
    # Get point estimates
    result = estimate_baron_kenny(data, outcome, treatment, mediator, controls)

    acme = result['acme']
    acme_se = result['acme_se']

    # Get residual standard deviations
    med_data = create_mediation_data(data, outcome, treatment, mediator, controls)

    # Fit models to get residuals
    import statsmodels.api as sm

    # Mediator model residuals
    df = data[[outcome, treatment, mediator] + controls].dropna()
    X_m = sm.add_constant(df[[treatment] + controls])
    model_m = sm.OLS(df[mediator], X_m).fit()
    sigma_m = np.std(model_m.resid)

    # Outcome model residuals
    X_y = sm.add_constant(df[[treatment, mediator] + controls])
    model_y = sm.OLS(df[outcome], X_y).fit()
    sigma_y = np.std(model_y.resid)

    # Calculate ACME under each rho
    acme_values = []
    ci_lower = []
    ci_upper = []

    for rho in rho_range:
        # Bias induced by correlation rho
        bias = rho * sigma_m * sigma_y
        acme_rho = acme - bias

        acme_values.append(acme_rho)
        ci_lower.append(acme_rho - 1.96 * acme_se)
        ci_upper.append(acme_rho + 1.96 * acme_se)

    acme_values = np.array(acme_values)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)

    # Find breakpoint
    sigma_product = sigma_m * sigma_y
    if sigma_product > 0:
        breakpoint = acme / sigma_product
    else:
        breakpoint = np.nan

    # R-squared equivalent
    r2_breakpoint = breakpoint ** 2 if not np.isnan(breakpoint) else np.nan

    # Find where CI excludes zero
    positive_ci = ci_lower > 0
    negative_ci = ci_upper < 0
    significant = positive_ci | negative_ci

    # Range of rho where ACME is significant
    if np.any(significant):
        sig_rho_range = (rho_range[significant].min(),
                         rho_range[significant].max())
    else:
        sig_rho_range = (np.nan, np.nan)

    # Interpretation
    if np.isnan(breakpoint):
        robustness = 'undefined'
        interpretation = 'Cannot compute breakpoint (check model residuals)'
    elif abs(breakpoint) > 0.9:
        robustness = 'very_robust'
        interpretation = (
            f"Very robust to unmeasured confounding. "
            f"An unmeasured confounder would need to induce rho > {abs(breakpoint):.2f} "
            f"(near-perfect correlation) to eliminate the mediation effect."
        )
    elif abs(breakpoint) > 0.5:
        robustness = 'robust'
        interpretation = (
            f"Robust to moderate unmeasured confounding. "
            f"Breakpoint at rho = {breakpoint:.3f} (R-squared equivalent = {r2_breakpoint:.1%}). "
            f"Strong confounding required to nullify effect."
        )
    elif abs(breakpoint) > 0.3:
        robustness = 'moderate'
        interpretation = (
            f"Moderately robust. Breakpoint at rho = {breakpoint:.3f} "
            f"(R-squared equivalent = {r2_breakpoint:.1%}). "
            f"A confounder explaining ~{r2_breakpoint*100:.0f}% of residual variance "
            f"in both M and Y could eliminate the effect."
        )
    elif abs(breakpoint) > 0.1:
        robustness = 'sensitive'
        interpretation = (
            f"Sensitive to confounding. Breakpoint at rho = {breakpoint:.3f} "
            f"(R-squared equivalent = {r2_breakpoint:.1%}). "
            f"Even modest confounding could eliminate the mediation effect."
        )
    else:
        robustness = 'very_sensitive'
        interpretation = (
            f"Very sensitive to confounding. Breakpoint at rho = {breakpoint:.3f} "
            f"(R-squared equivalent = {r2_breakpoint:.1%}). "
            f"Weak unmeasured confounding could eliminate the effect. "
            f"Exercise caution in interpretation."
        )

    return {
        'original_acme': acme,
        'original_se': acme_se,
        'rho_range': rho_range.tolist(),
        'acme_values': acme_values.tolist(),
        'ci_lower': ci_lower.tolist(),
        'ci_upper': ci_upper.tolist(),
        'breakpoint': breakpoint,
        'r2_breakpoint': r2_breakpoint,
        'sigma_m': sigma_m,
        'sigma_y': sigma_y,
        'robustness': robustness,
        'interpretation': interpretation,
        'significant_rho_range': sig_rho_range,
        'n_obs': len(df)
    }


def create_sensitivity_plot(
    results: Dict,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create comprehensive sensitivity analysis plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    rho = np.array(results['rho_range'])
    acme = np.array(results['acme_values'])
    ci_lower = np.array(results['ci_lower'])
    ci_upper = np.array(results['ci_upper'])
    breakpoint = results['breakpoint']

    # ===== Panel A: ACME vs Rho =====
    ax1 = axes[0]

    # Main line
    ax1.plot(rho, acme, 'b-', linewidth=2.5, label='ACME(rho)')

    # Confidence band
    ax1.fill_between(rho, ci_lower, ci_upper, alpha=0.25, color='blue',
                     label='95% CI')

    # Reference lines
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5,
                label='Zero effect')
    ax1.axvline(x=0, color='gray', linestyle=':', linewidth=1,
                label='No confounding')

    # Breakpoint
    if not np.isnan(breakpoint) and rho.min() < breakpoint < rho.max():
        ax1.axvline(x=breakpoint, color='orange', linestyle='--', linewidth=2)
        ax1.scatter([breakpoint], [0], color='orange', s=150, zorder=5,
                    edgecolor='black', linewidth=1.5)
        ax1.annotate(f'Breakpoint\nrho = {breakpoint:.3f}',
                     xy=(breakpoint, 0),
                     xytext=(breakpoint + 0.15, acme[len(acme)//4]),
                     fontsize=10, color='darkorange',
                     arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))

    # Mark original estimate
    ax1.scatter([0], [results['original_acme']], color='green', s=100,
                marker='D', zorder=5, label='Original ACME')

    ax1.set_xlabel('Sensitivity Parameter (rho)', fontsize=12)
    ax1.set_ylabel('ACME', fontsize=12)
    ax1.set_title('Sensitivity Analysis:\nACME vs. Unmeasured Confounding', fontsize=13)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Add robustness indicator
    robustness_colors = {
        'very_robust': 'darkgreen',
        'robust': 'green',
        'moderate': 'orange',
        'sensitive': 'red',
        'very_sensitive': 'darkred',
        'undefined': 'gray'
    }
    ax1.text(0.02, 0.98, f"Robustness: {results['robustness'].upper()}",
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             fontweight='bold', color=robustness_colors.get(results['robustness'], 'black'),
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ===== Panel B: R-squared interpretation =====
    ax2 = axes[1]

    # Convert rho to R-squared
    r2_values = rho ** 2
    acme_by_r2 = acme.copy()

    # Plot for positive and negative rho separately
    positive_mask = rho >= 0
    negative_mask = rho <= 0

    ax2.plot(r2_values[positive_mask], acme_by_r2[positive_mask],
             'b-', linewidth=2, label='Positive confounding (rho > 0)')
    ax2.plot(r2_values[negative_mask], acme_by_r2[negative_mask],
             'r--', linewidth=2, label='Negative confounding (rho < 0)')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Mark breakpoint R-squared
    r2_break = results['r2_breakpoint']
    if not np.isnan(r2_break) and 0 < r2_break < 1:
        ax2.axvline(x=r2_break, color='orange', linestyle='--', linewidth=2)
        ax2.scatter([r2_break], [0], color='orange', s=150, zorder=5,
                    edgecolor='black', linewidth=1.5)
        ax2.annotate(f'Breakpoint\nR-sq = {r2_break:.1%}',
                     xy=(r2_break, 0),
                     xytext=(r2_break + 0.1, results['original_acme'] * 0.5),
                     fontsize=10, color='darkorange',
                     arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))

    ax2.set_xlabel('R-squared (proportion of variance explained)', fontsize=12)
    ax2.set_ylabel('ACME', fontsize=12)
    ax2.set_title('R-squared Interpretation:\nConfounding Strength Required', fontsize=13)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.9)

    # Add interpretation text
    interp_text = (
        f"An unmeasured confounder would need to\n"
        f"explain {r2_break*100:.1f}% of residual variance\n"
        f"in both M and Y to nullify the effect."
        if not np.isnan(r2_break) else
        "Breakpoint outside plotted range"
    )
    ax2.text(0.98, 0.02, interp_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Plot saved to: {save_path}")

    return fig


def format_sensitivity_report(results: Dict) -> str:
    """Format sensitivity analysis results as text report."""
    lines = [
        "=" * 70,
        "SENSITIVITY ANALYSIS FOR CAUSAL MEDIATION".center(70),
        "(Imai, Keele, & Yamamoto, 2010)".center(70),
        "=" * 70,
        "",
        f"{'Original ACME:':<30} {results['original_acme']:.4f} (SE = {results['original_se']:.4f})",
        f"{'Mediator residual SD:':<30} {results['sigma_m']:.4f}",
        f"{'Outcome residual SD:':<30} {results['sigma_y']:.4f}",
        "",
        "-" * 70,
        "KEY FINDINGS",
        "-" * 70,
        "",
        f"{'Breakpoint (rho):':<30} {results['breakpoint']:.4f}",
        f"{'R-squared equivalent:':<30} {results['r2_breakpoint']:.1%}",
        f"{'Robustness level:':<30} {results['robustness'].upper()}",
        "",
        "-" * 70,
        "INTERPRETATION",
        "-" * 70,
        "",
        results['interpretation'],
        "",
        "-" * 70,
        "GUIDELINES",
        "-" * 70,
        "",
        "| Breakpoint |rho| |  Interpretation           | Confidence |",
        "|-------------|---------------------------|------------|",
        "|    > 0.9    | Near-perfect confounding  |  Very High |",
        "|  0.5 - 0.9  | Strong confounding        |  High      |",
        "|  0.3 - 0.5  | Moderate confounding      |  Medium    |",
        "|  0.1 - 0.3  | Modest confounding        |  Low       |",
        "|    < 0.1    | Weak confounding          |  Very Low  |",
        "",
        "=" * 70,
    ]

    return "\n".join(lines)


def main():
    """Main entry point."""
    args = parse_args()

    # Load data
    if args.verbose:
        print(f"Loading data from: {args.data_file}")

    data = pd.read_csv(args.data_file)

    # Validate variables
    required = [args.outcome, args.treatment, args.mediator] + args.controls
    missing = [v for v in required if v not in data.columns]
    if missing:
        print(f"Error: Variables not found: {missing}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Running sensitivity analysis...")
        print(f"Rho range: [{args.rho_min}, {args.rho_max}] step {args.rho_step}")

    # Create rho range
    rho_range = np.arange(args.rho_min, args.rho_max + args.rho_step, args.rho_step)

    # Run sensitivity analysis
    results = comprehensive_sensitivity_analysis(
        data, args.outcome, args.treatment, args.mediator, args.controls,
        rho_range
    )

    # Print text report
    print(format_sensitivity_report(results))

    # Create plot if requested
    if args.plot:
        create_sensitivity_plot(results, args.plot)

    # Save JSON if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
