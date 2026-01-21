#!/usr/bin/env python3
"""
Propensity Score Overlap Visualization Script

Creates comprehensive visualizations for assessing common support:
- Propensity score distributions
- Mirror histograms
- Box plots
- Cumulative distribution comparison
- Common support region

Usage:
    python visualize_overlap.py --data data.csv --treatment treat --ps propensity_score \\
                                --output overlap_plot.png

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np


def calculate_overlap_statistics(
    ps_treated: np.ndarray,
    ps_control: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate comprehensive overlap statistics.

    Parameters
    ----------
    ps_treated : np.ndarray
        Propensity scores for treated units
    ps_control : np.ndarray
        Propensity scores for control units

    Returns
    -------
    Dict[str, Any]
        Overlap statistics
    """
    # Remove NaN values
    ps_t = ps_treated[~np.isnan(ps_treated)]
    ps_c = ps_control[~np.isnan(ps_control)]

    # Basic statistics
    stats = {
        'n_treated': len(ps_t),
        'n_control': len(ps_c),
        'ps_treated_min': ps_t.min(),
        'ps_treated_max': ps_t.max(),
        'ps_treated_mean': ps_t.mean(),
        'ps_treated_median': np.median(ps_t),
        'ps_control_min': ps_c.min(),
        'ps_control_max': ps_c.max(),
        'ps_control_mean': ps_c.mean(),
        'ps_control_median': np.median(ps_c),
    }

    # Common support bounds
    lower_bound = max(ps_t.min(), ps_c.min())
    upper_bound = min(ps_t.max(), ps_c.max())
    stats['common_support_lower'] = lower_bound
    stats['common_support_upper'] = upper_bound

    # Check for valid common support
    if upper_bound <= lower_bound:
        stats['common_support_valid'] = False
        stats['overlap_ratio'] = 0.0
        stats['pct_treated_in_support'] = 0.0
        stats['pct_control_in_support'] = 0.0
    else:
        stats['common_support_valid'] = True

        # Overlap ratio
        total_range = max(ps_t.max(), ps_c.max()) - min(ps_t.min(), ps_c.min())
        common_range = upper_bound - lower_bound
        stats['overlap_ratio'] = common_range / total_range if total_range > 0 else 0

        # Percentage in common support
        in_support_t = np.sum((ps_t >= lower_bound) & (ps_t <= upper_bound))
        in_support_c = np.sum((ps_c >= lower_bound) & (ps_c <= upper_bound))
        stats['pct_treated_in_support'] = in_support_t / len(ps_t) * 100
        stats['pct_control_in_support'] = in_support_c / len(ps_c) * 100
        stats['n_treated_in_support'] = in_support_t
        stats['n_control_in_support'] = in_support_c

    # Trimmed bounds (more conservative)
    trim_pct = 5
    lower_trim = max(
        np.percentile(ps_t, trim_pct),
        np.percentile(ps_c, trim_pct)
    )
    upper_trim = min(
        np.percentile(ps_t, 100 - trim_pct),
        np.percentile(ps_c, 100 - trim_pct)
    )
    stats['trimmed_support_lower'] = lower_trim
    stats['trimmed_support_upper'] = upper_trim

    return stats


def create_overlap_figure(
    ps_treated: np.ndarray,
    ps_control: np.ndarray,
    stats: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 10),
    title: str = "Propensity Score Overlap Assessment"
) -> Any:
    """
    Create comprehensive overlap visualization.

    Parameters
    ----------
    ps_treated : np.ndarray
        Propensity scores for treated
    ps_control : np.ndarray
        Propensity scores for control
    stats : Dict[str, Any]
        Pre-calculated statistics
    figsize : tuple
        Figure size
    title : str
        Main title

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Remove NaN values
    ps_t = ps_treated[~np.isnan(ps_treated)]
    ps_c = ps_control[~np.isnan(ps_control)]

    fig = plt.figure(figsize=figsize)

    # Create 2x2 subplot layout
    ax1 = plt.subplot(2, 2, 1)  # Histogram
    ax2 = plt.subplot(2, 2, 2)  # Mirror histogram
    ax3 = plt.subplot(2, 2, 3)  # Box plot
    ax4 = plt.subplot(2, 2, 4)  # CDF

    # Common settings
    bins = np.linspace(0, 1, 50)
    alpha = 0.6

    # ============================================
    # Panel 1: Overlaid Histograms
    # ============================================
    ax1.hist(ps_c, bins=bins, alpha=alpha, label='Control',
             color='royalblue', density=True, edgecolor='white')
    ax1.hist(ps_t, bins=bins, alpha=alpha, label='Treated',
             color='indianred', density=True, edgecolor='white')

    # Add common support region
    if stats['common_support_valid']:
        ymin, ymax = ax1.get_ylim()
        rect = Rectangle(
            (stats['common_support_lower'], ymin),
            stats['common_support_upper'] - stats['common_support_lower'],
            ymax - ymin,
            alpha=0.2, facecolor='green',
            label='Common Support'
        )
        ax1.add_patch(rect)

    ax1.set_xlabel('Propensity Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Propensity Score Distributions')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ============================================
    # Panel 2: Mirror Histogram
    # ============================================
    hist_t, _ = np.histogram(ps_t, bins=bins, density=True)
    hist_c, _ = np.histogram(ps_c, bins=bins, density=True)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]

    ax2.bar(bin_centers, hist_t, width=width, alpha=alpha,
            label='Treated', color='indianred', edgecolor='white')
    ax2.bar(bin_centers, -hist_c, width=width, alpha=alpha,
            label='Control', color='royalblue', edgecolor='white')

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Propensity Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Mirror Histogram (Treated/Control)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # ============================================
    # Panel 3: Box Plots
    # ============================================
    data_box = [ps_c, ps_t]
    bp = ax3.boxplot(data_box, labels=['Control', 'Treated'],
                     patch_artist=True, notch=True)

    colors = ['royalblue', 'indianred']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add mean markers
    ax3.scatter([1, 2], [np.mean(ps_c), np.mean(ps_t)],
                marker='D', color='white', edgecolor='black',
                s=50, zorder=3, label='Mean')

    # Add common support region
    if stats['common_support_valid']:
        ax3.axhline(y=stats['common_support_lower'],
                    color='green', linestyle='--', alpha=0.7)
        ax3.axhline(y=stats['common_support_upper'],
                    color='green', linestyle='--', alpha=0.7)

    ax3.set_ylabel('Propensity Score')
    ax3.set_title('Box Plots by Treatment Status')
    ax3.grid(True, alpha=0.3, axis='y')

    # ============================================
    # Panel 4: Empirical CDFs
    # ============================================
    sorted_t = np.sort(ps_t)
    sorted_c = np.sort(ps_c)
    cdf_t = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
    cdf_c = np.arange(1, len(sorted_c) + 1) / len(sorted_c)

    ax4.step(sorted_t, cdf_t, where='post', label='Treated',
             color='indianred', linewidth=2)
    ax4.step(sorted_c, cdf_c, where='post', label='Control',
             color='royalblue', linewidth=2)

    # Add common support region
    if stats['common_support_valid']:
        ax4.axvline(x=stats['common_support_lower'],
                    color='green', linestyle='--', alpha=0.7)
        ax4.axvline(x=stats['common_support_upper'],
                    color='green', linestyle='--', alpha=0.7)
        ax4.axvspan(stats['common_support_lower'],
                    stats['common_support_upper'],
                    alpha=0.1, facecolor='green',
                    label='Common Support')

    ax4.set_xlabel('Propensity Score')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Empirical CDFs')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)

    # Main title
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    return fig


def print_overlap_report(stats: Dict[str, Any]) -> str:
    """
    Create text report of overlap statistics.

    Parameters
    ----------
    stats : Dict[str, Any]
        Overlap statistics

    Returns
    -------
    str
        Formatted report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PROPENSITY SCORE OVERLAP ASSESSMENT")
    lines.append("=" * 60)
    lines.append("")

    lines.append("SAMPLE SIZES")
    lines.append("-" * 40)
    lines.append(f"  Treated:  {stats['n_treated']:,}")
    lines.append(f"  Control:  {stats['n_control']:,}")
    lines.append("")

    lines.append("PROPENSITY SCORE RANGES")
    lines.append("-" * 40)
    lines.append(f"  Treated:  [{stats['ps_treated_min']:.4f}, {stats['ps_treated_max']:.4f}]")
    lines.append(f"  Control:  [{stats['ps_control_min']:.4f}, {stats['ps_control_max']:.4f}]")
    lines.append("")

    lines.append("PROPENSITY SCORE MEANS")
    lines.append("-" * 40)
    lines.append(f"  Treated mean:    {stats['ps_treated_mean']:.4f}")
    lines.append(f"  Control mean:    {stats['ps_control_mean']:.4f}")
    lines.append(f"  Treated median:  {stats['ps_treated_median']:.4f}")
    lines.append(f"  Control median:  {stats['ps_control_median']:.4f}")
    lines.append("")

    lines.append("COMMON SUPPORT")
    lines.append("-" * 40)
    if stats['common_support_valid']:
        lines.append(f"  Region:           [{stats['common_support_lower']:.4f}, "
                     f"{stats['common_support_upper']:.4f}]")
        lines.append(f"  Overlap ratio:    {stats['overlap_ratio']:.3f}")
        lines.append(f"  Treated in CS:    {stats['n_treated_in_support']:,} "
                     f"({stats['pct_treated_in_support']:.1f}%)")
        lines.append(f"  Control in CS:    {stats['n_control_in_support']:,} "
                     f"({stats['pct_control_in_support']:.1f}%)")
    else:
        lines.append("  WARNING: No valid common support region!")
    lines.append("")

    lines.append("TRIMMED SUPPORT (5th-95th percentile)")
    lines.append("-" * 40)
    lines.append(f"  Region:           [{stats['trimmed_support_lower']:.4f}, "
                 f"{stats['trimmed_support_upper']:.4f}]")
    lines.append("")

    # Assessment
    lines.append("ASSESSMENT")
    lines.append("-" * 40)

    if not stats['common_support_valid']:
        assessment = "POOR - No common support region"
        recommendation = "Consider alternative methods (IV, RD, synthetic control)"
    elif stats['overlap_ratio'] < 0.3:
        assessment = "POOR - Very limited overlap"
        recommendation = "Trim data to common support or use caliper matching"
    elif stats['overlap_ratio'] < 0.5:
        assessment = "MODERATE - Limited overlap"
        recommendation = "Use caliper matching and report results for trimmed sample"
    elif stats['pct_treated_in_support'] < 80 or stats['pct_control_in_support'] < 80:
        assessment = "MODERATE - Many observations outside support"
        recommendation = "Report results for both full and trimmed samples"
    else:
        assessment = "GOOD - Substantial overlap"
        recommendation = "Proceed with matching"

    lines.append(f"  {assessment}")
    lines.append(f"  Recommendation: {recommendation}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Propensity Score Overlap Visualization"
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
        "--ps",
        type=str,
        required=True,
        help="Propensity score column name"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for figure (e.g., overlap.png)"
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="Output path for statistics JSON"
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[14, 10],
        help="Figure size (width height)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Propensity Score Overlap Assessment",
        help="Figure title"
    )

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data)
    print(f"Loaded data: {len(data)} observations")

    # Extract propensity scores
    ps_treated = data.loc[data[args.treatment] == 1, args.ps].values
    ps_control = data.loc[data[args.treatment] == 0, args.ps].values

    # Calculate statistics
    stats = calculate_overlap_statistics(ps_treated, ps_control)

    # Print report
    report = print_overlap_report(stats)
    print(report)

    # Create visualization
    try:
        import matplotlib.pyplot as plt

        fig = create_overlap_figure(
            ps_treated, ps_control, stats,
            figsize=tuple(args.figsize),
            title=args.title
        )

        if args.output:
            fig.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {args.output}")
        else:
            plt.show()

        plt.close(fig)

    except ImportError:
        print("\nmatplotlib not available for plotting")

    # Save statistics
    if args.stats_output:
        import json
        with open(args.stats_output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {args.stats_output}")


if __name__ == "__main__":
    main()
