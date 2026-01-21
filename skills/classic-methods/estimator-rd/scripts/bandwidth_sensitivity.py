#!/usr/bin/env python3
"""
Automated Bandwidth Sensitivity Analysis for RD Designs.

This script provides comprehensive bandwidth sensitivity analysis following
Cattaneo et al. methodology, including:
1. Sensitivity tables across bandwidth multipliers
2. Visualization of effect stability
3. Detection of potential sensitivity concerns
4. Formal sensitivity metrics

Usage:
    python bandwidth_sensitivity.py data.csv --running score --outcome y --cutoff 0

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from rd_estimator import (
    select_bandwidth,
    estimate_sharp_rd,
    estimate_fuzzy_rd
)


@dataclass
class SensitivityResult:
    """Container for bandwidth sensitivity results."""
    bandwidth: float
    bw_ratio: float
    effect: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_effective: int
    significant: bool


@dataclass
class SensitivitySummary:
    """Summary of sensitivity analysis."""
    is_robust: bool
    effect_range: float
    effect_range_percent: float
    sign_stable: bool
    significance_stable: bool
    median_effect: float
    optimal_effect: float
    concerns: List[str]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run bandwidth sensitivity analysis for RD designs"
    )

    parser.add_argument("data_file", type=str, help="Path to CSV data file")
    parser.add_argument("--running", type=str, required=True, help="Running variable column")
    parser.add_argument("--outcome", type=str, required=True, help="Outcome variable column")
    parser.add_argument("--cutoff", type=float, required=True, help="Cutoff value")
    parser.add_argument("--treatment", type=str, default=None, help="Treatment column for fuzzy RD")

    parser.add_argument("--bw-range", type=str, default="0.5,0.75,1.0,1.25,1.5,2.0",
                       help="Bandwidth multipliers (comma-separated)")
    parser.add_argument("--n-grid", type=int, default=None,
                       help="Number of grid points (overrides --bw-range)")
    parser.add_argument("--bw-min", type=float, default=0.25, help="Min bandwidth multiplier")
    parser.add_argument("--bw-max", type=float, default=3.0, help="Max bandwidth multiplier")

    parser.add_argument("--kernel", type=str, default="triangular", help="Kernel function")
    parser.add_argument("--order", type=int, default=1, help="Polynomial order")

    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--format", type=str, default="markdown",
                       choices=["markdown", "latex", "json"],
                       help="Output format")
    parser.add_argument("--plot", action="store_true", help="Generate sensitivity plot")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def run_bandwidth_sensitivity(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    cutoff: float,
    treatment: Optional[str] = None,
    bw_multipliers: Optional[List[float]] = None,
    kernel: str = "triangular",
    order: int = 1,
    verbose: bool = False
) -> Tuple[List[SensitivityResult], float, SensitivitySummary]:
    """
    Run bandwidth sensitivity analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    running : str
        Running variable column
    outcome : str
        Outcome variable column
    cutoff : float
        Cutoff value
    treatment : str, optional
        Treatment column for fuzzy RD
    bw_multipliers : list, optional
        Bandwidth multipliers to test
    kernel : str
        Kernel function
    order : int
        Polynomial order
    verbose : bool
        Print progress

    Returns
    -------
    Tuple[List[SensitivityResult], float, SensitivitySummary]
        (results list, optimal bandwidth, summary)
    """
    # Get optimal bandwidth
    x = data[running].values
    y = data[outcome].values
    mask = ~(np.isnan(x) | np.isnan(y))

    h_opt = select_bandwidth(x[mask], y[mask], cutoff, method="mserd", kernel=kernel, order=order)

    if verbose:
        print(f"Optimal bandwidth: {h_opt:.4f}")

    # Default multipliers if not specified
    if bw_multipliers is None:
        bw_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    # Run estimation at each bandwidth
    results = []

    for mult in bw_multipliers:
        bw = mult * h_opt

        if verbose:
            print(f"  Testing bandwidth {bw:.4f} ({mult:.2f}x optimal)...")

        try:
            if treatment:
                result = estimate_fuzzy_rd(
                    data, running, outcome, treatment, cutoff, bw, kernel, order
                )
            else:
                result = estimate_sharp_rd(
                    data, running, outcome, cutoff, bw, kernel, order
                )

            sens_result = SensitivityResult(
                bandwidth=bw,
                bw_ratio=mult,
                effect=result.effect,
                se=result.se,
                ci_lower=result.ci_lower,
                ci_upper=result.ci_upper,
                p_value=result.p_value,
                n_effective=result.diagnostics.get('n_effective', 0),
                significant=result.p_value < 0.05
            )

        except Exception as e:
            if verbose:
                print(f"    Warning: Estimation failed - {e}")

            sens_result = SensitivityResult(
                bandwidth=bw,
                bw_ratio=mult,
                effect=np.nan,
                se=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                p_value=np.nan,
                n_effective=0,
                significant=False
            )

        results.append(sens_result)

    # Generate summary
    summary = compute_sensitivity_summary(results, h_opt)

    return results, h_opt, summary


def compute_sensitivity_summary(
    results: List[SensitivityResult],
    h_opt: float
) -> SensitivitySummary:
    """
    Compute summary statistics for sensitivity analysis.

    Parameters
    ----------
    results : List[SensitivityResult]
        Sensitivity results
    h_opt : float
        Optimal bandwidth

    Returns
    -------
    SensitivitySummary
        Summary assessment
    """
    # Filter valid results
    valid = [r for r in results if not np.isnan(r.effect)]

    if len(valid) == 0:
        return SensitivitySummary(
            is_robust=False,
            effect_range=np.nan,
            effect_range_percent=np.nan,
            sign_stable=False,
            significance_stable=False,
            median_effect=np.nan,
            optimal_effect=np.nan,
            concerns=["No valid estimates obtained"]
        )

    effects = [r.effect for r in valid]
    sigs = [r.significant for r in valid]

    # Find optimal result
    opt_results = [r for r in valid if abs(r.bw_ratio - 1.0) < 0.01]
    optimal_effect = opt_results[0].effect if opt_results else np.median(effects)

    # Compute metrics
    effect_range = max(effects) - min(effects)
    median_effect = np.median(effects)
    effect_range_percent = (effect_range / abs(median_effect) * 100) if median_effect != 0 else np.inf

    # Sign stability
    all_positive = all(e > 0 for e in effects)
    all_negative = all(e < 0 for e in effects)
    sign_stable = all_positive or all_negative

    # Significance stability
    all_sig = all(sigs)
    all_insig = not any(sigs)
    significance_stable = all_sig or all_insig

    # Identify concerns
    concerns = []

    if effect_range_percent > 50:
        concerns.append(f"Large effect range ({effect_range_percent:.1f}% of median)")

    if not sign_stable:
        concerns.append("Effect sign changes across bandwidths")

    if not significance_stable:
        concerns.append("Significance varies across bandwidths")

    # Check for monotonic trend
    if len(effects) >= 4:
        bw_ratios = [r.bw_ratio for r in valid]
        correlation = np.corrcoef(bw_ratios, effects)[0, 1]
        if abs(correlation) > 0.9:
            direction = "increasing" if correlation > 0 else "decreasing"
            concerns.append(f"Systematic {direction} trend with bandwidth (r={correlation:.2f})")

    # Overall robustness assessment
    is_robust = (
        effect_range_percent < 30 and
        sign_stable and
        significance_stable and
        len(concerns) == 0
    )

    return SensitivitySummary(
        is_robust=is_robust,
        effect_range=effect_range,
        effect_range_percent=effect_range_percent,
        sign_stable=sign_stable,
        significance_stable=significance_stable,
        median_effect=median_effect,
        optimal_effect=optimal_effect,
        concerns=concerns
    )


def format_sensitivity_table_markdown(
    results: List[SensitivityResult],
    h_opt: float,
    summary: SensitivitySummary
) -> str:
    """Format sensitivity results as Markdown table."""
    lines = []

    lines.append("# Bandwidth Sensitivity Analysis")
    lines.append("")
    lines.append(f"**Optimal Bandwidth (MSE)**: {h_opt:.4f}")
    lines.append("")

    # Main table
    lines.append("## Sensitivity Table")
    lines.append("")
    lines.append("| Bandwidth | Ratio | Effect | SE | 95% CI | p-value | N_eff | Sig |")
    lines.append("|-----------|-------|--------|-----|--------|---------|-------|-----|")

    for r in results:
        if np.isnan(r.effect):
            lines.append(f"| {r.bandwidth:.4f} | {r.bw_ratio:.2f}x | N/A | N/A | N/A | N/A | 0 | - |")
        else:
            ci_str = f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]"
            sig = "*" if r.significant else ""
            opt_marker = " **" if abs(r.bw_ratio - 1.0) < 0.01 else ""
            lines.append(
                f"| {r.bandwidth:.4f}{opt_marker} | {r.bw_ratio:.2f}x | "
                f"{r.effect:.4f} | {r.se:.4f} | {ci_str} | {r.p_value:.4f} | "
                f"{r.n_effective} | {sig} |"
            )

    lines.append("")
    lines.append("*Note: ** indicates optimal bandwidth, * indicates p < 0.05*")
    lines.append("")

    # Summary
    lines.append("## Summary Assessment")
    lines.append("")
    lines.append(f"**Overall Robustness**: {'ROBUST' if summary.is_robust else 'SENSITIVE'}")
    lines.append("")
    lines.append(f"- Effect Range: {summary.effect_range:.4f} ({summary.effect_range_percent:.1f}% of median)")
    lines.append(f"- Median Effect: {summary.median_effect:.4f}")
    lines.append(f"- Optimal Effect: {summary.optimal_effect:.4f}")
    lines.append(f"- Sign Stable: {'Yes' if summary.sign_stable else 'No'}")
    lines.append(f"- Significance Stable: {'Yes' if summary.significance_stable else 'No'}")
    lines.append("")

    if summary.concerns:
        lines.append("### Concerns")
        lines.append("")
        for concern in summary.concerns:
            lines.append(f"- {concern}")
        lines.append("")

    return "\n".join(lines)


def format_sensitivity_table_latex(
    results: List[SensitivityResult],
    h_opt: float,
    summary: SensitivitySummary
) -> str:
    """Format sensitivity results as LaTeX table."""
    valid_results = [r for r in results if not np.isnan(r.effect)]

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Bandwidth Sensitivity Analysis}
\label{tab:bw_sensitivity}
\begin{tabular}{ccccccc}
\toprule
Bandwidth & Ratio & Effect & SE & 95\% CI & p-value & N$_{eff}$ \\
\midrule
"""

    for r in valid_results:
        stars = ""
        if r.p_value < 0.01:
            stars = "***"
        elif r.p_value < 0.05:
            stars = "**"
        elif r.p_value < 0.1:
            stars = "*"

        ci_str = f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]"
        opt = r"$^\dagger$" if abs(r.bw_ratio - 1.0) < 0.01 else ""

        latex += (
            f"{r.bandwidth:.4f}{opt} & {r.bw_ratio:.2f}$\\times$ & "
            f"{r.effect:.4f}{stars} & {r.se:.4f} & {ci_str} & "
            f"{r.p_value:.4f} & {r.n_effective} \\\\\n"
        )

    latex += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: $^\dagger$ indicates MSE-optimal bandwidth (h = """ + f"{h_opt:.4f}" + r""").
\item Effect range: """ + f"{summary.effect_range:.4f}" + r""" (""" + f"{summary.effect_range_percent:.1f}" + r"""\% of median).
\item *** p$<$0.01, ** p$<$0.05, * p$<$0.1
\end{tablenotes}
\end{table}
"""
    return latex


def create_sensitivity_plot(
    results: List[SensitivityResult],
    h_opt: float,
    summary: SensitivitySummary,
    output_path: Optional[str] = None
):
    """Create bandwidth sensitivity visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    valid = [r for r in results if not np.isnan(r.effect)]

    if len(valid) == 0:
        print("No valid results to plot")
        return

    bws = [r.bandwidth for r in valid]
    effects = [r.effect for r in valid]
    ci_lower = [r.ci_lower for r in valid]
    ci_upper = [r.ci_upper for r in valid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Effect by bandwidth with CI
    ax1 = axes[0]

    ax1.fill_between(bws, ci_lower, ci_upper, alpha=0.3, color='steelblue')
    ax1.plot(bws, effects, 'o-', color='steelblue', markersize=8, linewidth=2)

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=h_opt, color='red', linestyle=':', linewidth=1.5,
                label=f'Optimal h = {h_opt:.3f}')

    ax1.set_xlabel('Bandwidth', fontsize=12)
    ax1.set_ylabel('RD Effect Estimate', fontsize=12)
    ax1.set_title('A. Effect Stability Across Bandwidths', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Effect normalized by optimal
    ax2 = axes[1]

    opt_effect = summary.optimal_effect
    if opt_effect != 0:
        normalized = [(e / opt_effect) for e in effects]
        ax2.plot(bws, normalized, 'o-', color='indianred', markersize=8, linewidth=2)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        ax2.axhline(y=0.7, color='orange', linestyle=':', alpha=0.5, label='70% of optimal')
        ax2.axhline(y=1.3, color='orange', linestyle=':', alpha=0.5, label='130% of optimal')
        ax2.axvline(x=h_opt, color='red', linestyle=':', linewidth=1.5)

        ax2.set_ylabel('Effect / Optimal Effect', fontsize=12)
        ax2.legend(loc='best')
    else:
        ax2.text(0.5, 0.5, 'Optimal effect is zero', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14)

    ax2.set_xlabel('Bandwidth', fontsize=12)
    ax2.set_title('B. Relative Effect Stability', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add summary annotation
    status = 'ROBUST' if summary.is_robust else 'SENSITIVE'
    status_color = 'green' if summary.is_robust else 'red'

    fig.suptitle(
        f'Bandwidth Sensitivity Analysis (Assessment: {status})',
        fontsize=14, fontweight='bold', color=status_color
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Main entry point."""
    args = parse_args()

    # Load data
    if not Path(args.data_file).exists():
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)

    data = pd.read_csv(args.data_file)

    # Parse bandwidth range
    if args.n_grid:
        bw_multipliers = list(np.linspace(args.bw_min, args.bw_max, args.n_grid))
    else:
        bw_multipliers = [float(x) for x in args.bw_range.split(',')]

    # Run analysis
    if args.verbose:
        print(f"Running bandwidth sensitivity with {len(bw_multipliers)} points...")

    results, h_opt, summary = run_bandwidth_sensitivity(
        data=data,
        running=args.running,
        outcome=args.outcome,
        cutoff=args.cutoff,
        treatment=args.treatment,
        bw_multipliers=bw_multipliers,
        kernel=args.kernel,
        order=args.order,
        verbose=args.verbose
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Format and save output
    if args.format == "json":
        output = {
            "optimal_bandwidth": h_opt,
            "results": [
                {
                    "bandwidth": r.bandwidth,
                    "bw_ratio": r.bw_ratio,
                    "effect": r.effect if not np.isnan(r.effect) else None,
                    "se": r.se if not np.isnan(r.se) else None,
                    "ci_lower": r.ci_lower if not np.isnan(r.ci_lower) else None,
                    "ci_upper": r.ci_upper if not np.isnan(r.ci_upper) else None,
                    "p_value": r.p_value if not np.isnan(r.p_value) else None,
                    "n_effective": r.n_effective,
                    "significant": r.significant
                }
                for r in results
            ],
            "summary": {
                "is_robust": summary.is_robust,
                "effect_range": summary.effect_range if not np.isnan(summary.effect_range) else None,
                "effect_range_percent": summary.effect_range_percent if not np.isnan(summary.effect_range_percent) else None,
                "sign_stable": summary.sign_stable,
                "significance_stable": summary.significance_stable,
                "median_effect": summary.median_effect if not np.isnan(summary.median_effect) else None,
                "concerns": summary.concerns
            }
        }

        output_file = output_dir / "bandwidth_sensitivity.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

    elif args.format == "latex":
        output_text = format_sensitivity_table_latex(results, h_opt, summary)
        output_file = output_dir / "bandwidth_sensitivity.tex"
        with open(output_file, "w") as f:
            f.write(output_text)

    else:  # markdown
        output_text = format_sensitivity_table_markdown(results, h_opt, summary)
        output_file = output_dir / "bandwidth_sensitivity.md"
        with open(output_file, "w") as f:
            f.write(output_text)

    print(f"Results saved to {output_file}")

    # Generate plot if requested
    if args.plot:
        plot_file = output_dir / "bandwidth_sensitivity.png"
        create_sensitivity_plot(results, h_opt, summary, str(plot_file))

    # Print summary
    print("\n" + "="*60)
    print("BANDWIDTH SENSITIVITY SUMMARY")
    print("="*60)
    print(f"Optimal Bandwidth: {h_opt:.4f}")
    print(f"Effect at Optimal: {summary.optimal_effect:.4f}")
    print(f"Median Effect: {summary.median_effect:.4f}")
    print(f"Effect Range: {summary.effect_range:.4f} ({summary.effect_range_percent:.1f}%)")
    print(f"Sign Stable: {'Yes' if summary.sign_stable else 'No'}")
    print(f"Significance Stable: {'Yes' if summary.significance_stable else 'No'}")
    print("-"*60)
    print(f"ASSESSMENT: {'ROBUST' if summary.is_robust else 'SENSITIVE'}")

    if summary.concerns:
        print("\nConcerns:")
        for concern in summary.concerns:
            print(f"  - {concern}")

    print("="*60)


if __name__ == "__main__":
    main()
