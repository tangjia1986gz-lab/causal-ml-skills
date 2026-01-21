#!/usr/bin/env python3
"""
Mediation Pathway Visualization

Creates publication-quality visualizations of causal mediation pathways,
including:
- Pathway diagrams with effect sizes
- Effect decomposition bar charts
- Coefficient comparison plots
- Subgroup mediation comparisons

Usage:
    python visualize_pathways.py data.csv \
        --outcome earnings --treatment training --mediator skills \
        --output pathway_diagram.png

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from mediation_estimator import (
    estimate_baron_kenny,
    run_full_mediation_analysis,
    sensitivity_analysis_mediation,
    create_mediation_data
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create mediation pathway visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter
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

    # Visualization options
    parser.add_argument('--plot-type', type=str,
                        choices=['pathway', 'decomposition', 'combined', 'all'],
                        default='pathway',
                        help='Type of visualization (default: pathway)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output file path')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output resolution (default: 300)')
    parser.add_argument('--style', type=str,
                        choices=['default', 'academic', 'minimal'],
                        default='academic',
                        help='Plot style (default: academic)')

    # Labels
    parser.add_argument('--treatment-label', type=str, default=None,
                        help='Display label for treatment')
    parser.add_argument('--mediator-label', type=str, default=None,
                        help='Display label for mediator')
    parser.add_argument('--outcome-label', type=str, default=None,
                        help='Display label for outcome')

    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    return parser.parse_args()


def create_pathway_diagram(
    results: Dict,
    treatment_name: str = "Treatment",
    mediator_name: str = "Mediator",
    outcome_name: str = "Outcome",
    style: str = 'academic',
    figsize: Tuple[int, int] = (12, 8),
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create publication-quality mediation pathway diagram.

    Parameters
    ----------
    results : Dict
        Output from estimate_baron_kenny() or similar
    treatment_name, mediator_name, outcome_name : str
        Display labels for variables
    style : str
        'academic', 'default', or 'minimal'
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    dpi : int
        Resolution for saved figure

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Style settings
    if style == 'academic':
        box_face = 'white'
        box_edge = 'black'
        box_lw = 2
        arrow_color = 'black'
        direct_color = 'gray'
        font_size = 12
        coef_size = 11
    elif style == 'minimal':
        box_face = 'none'
        box_edge = 'black'
        box_lw = 1.5
        arrow_color = 'black'
        direct_color = 'gray'
        font_size = 11
        coef_size = 10
    else:  # default
        box_face = 'lightblue'
        box_edge = 'navy'
        box_lw = 2
        arrow_color = 'navy'
        direct_color = 'darkgreen'
        font_size = 12
        coef_size = 11

    # Box style
    box_style = dict(
        boxstyle='round,pad=0.5',
        facecolor=box_face,
        edgecolor=box_edge,
        linewidth=box_lw
    )

    # Draw variable boxes
    # Treatment (left)
    ax.text(2, 4, treatment_name, ha='center', va='center',
            fontsize=font_size, fontweight='bold', bbox=box_style)

    # Mediator (top center)
    ax.text(6, 6.5, mediator_name, ha='center', va='center',
            fontsize=font_size, fontweight='bold', bbox=box_style)

    # Outcome (right)
    ax.text(10, 4, outcome_name, ha='center', va='center',
            fontsize=font_size, fontweight='bold', bbox=box_style)

    # Arrow properties
    arrow_style = dict(
        arrowstyle='-|>',
        color=arrow_color,
        linewidth=2,
        mutation_scale=15
    )

    # Treatment -> Mediator (a path)
    ax.annotate('', xy=(5, 6.2), xytext=(3, 4.5),
                arrowprops=arrow_style)

    # Get coefficients
    alpha = results.get('alpha', np.nan)
    alpha_se = results.get('alpha_se', np.nan)
    beta_m = results.get('beta_m', np.nan)
    beta_m_se = results.get('beta_m_se', np.nan)
    ade = results.get('ade', np.nan)
    ade_se = results.get('ade_se', np.nan)
    acme = results.get('acme', np.nan)
    total = results.get('total_effect', np.nan)
    prop = results.get('prop_mediated', np.nan)

    # Significance stars
    def get_stars(coef, se):
        if np.isnan(coef) or np.isnan(se) or se == 0:
            return ''
        z = abs(coef / se)
        if z > 2.58:
            return '***'
        elif z > 1.96:
            return '**'
        elif z > 1.645:
            return '*'
        return ''

    # a path label
    a_label = f"a = {alpha:.3f}{get_stars(alpha, alpha_se)}"
    if not np.isnan(alpha_se):
        a_label += f"\n({alpha_se:.3f})"
    ax.text(3.5, 5.8, a_label, fontsize=coef_size, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Mediator -> Outcome (b path)
    ax.annotate('', xy=(9, 4.5), xytext=(7, 6.2),
                arrowprops=arrow_style)

    b_label = f"b = {beta_m:.3f}{get_stars(beta_m, beta_m_se)}"
    if not np.isnan(beta_m_se):
        b_label += f"\n({beta_m_se:.3f})"
    ax.text(8.5, 5.8, b_label, fontsize=coef_size, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Direct effect (c' path)
    direct_arrow = dict(
        arrowstyle='-|>',
        color=direct_color,
        linewidth=2,
        mutation_scale=15,
        linestyle='--' if style == 'academic' else '-'
    )
    ax.annotate('', xy=(9, 4), xytext=(3, 4),
                arrowprops=direct_arrow)

    c_label = f"c' = {ade:.3f}{get_stars(ade, ade_se)}"
    if not np.isnan(ade_se):
        c_label += f" ({ade_se:.3f})"
    ax.text(6, 3.4, c_label, fontsize=coef_size, ha='center', color=direct_color,
            fontweight='bold' if style != 'minimal' else 'normal')

    # Effect decomposition box
    decomp_text = (
        f"Effect Decomposition\n"
        f"{'='*25}\n"
        f"Total Effect (c): {total:.4f}\n"
        f"Direct (c'): {ade:.4f}\n"
        f"Indirect (a*b): {acme:.4f}\n"
        f"{'='*25}\n"
        f"% Mediated: {prop*100:.1f}%"
    )

    decomp_box = dict(
        boxstyle='round,pad=0.4',
        facecolor='lightyellow' if style == 'default' else 'white',
        edgecolor='orange' if style == 'default' else 'black',
        linewidth=1.5
    )

    ax.text(6, 1.2, decomp_text, ha='center', va='center',
            fontsize=10, family='monospace', bbox=decomp_box)

    # Title
    ax.text(6, 7.7, "Causal Mediation Pathway",
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Legend
    legend_text = (
        "Path coefficients shown with standard errors in parentheses.\n"
        "*** p<0.01, ** p<0.05, * p<0.1\n"
        f"Dashed line = direct effect; Solid lines = indirect pathway"
    )
    ax.text(6, 0.2, legend_text, ha='center', va='bottom',
            fontsize=9, style='italic', color='gray')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    return fig


def create_decomposition_chart(
    results: Dict,
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create bar chart showing effect decomposition.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get values
    total = results.get('total_effect', np.nan)
    ade = results.get('ade', np.nan)
    acme = results.get('acme', np.nan)
    ade_se = results.get('ade_se', 0)
    acme_se = results.get('acme_se', 0)

    # Create bars
    effects = ['Total Effect', 'Direct (ADE)', 'Indirect (ACME)']
    values = [total, ade, acme]
    errors = [0, ade_se * 1.96, acme_se * 1.96]
    colors = ['steelblue', 'forestgreen', 'darkorange']

    bars = ax.bar(effects, values, yerr=errors, capsize=5, color=colors,
                  edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val, err in zip(bars, values, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + err + 0.01 * max(values),
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Reference line at zero
    ax.axhline(y=0, color='black', linewidth=1)

    # Proportion mediated annotation
    prop = results.get('prop_mediated', np.nan)
    if not np.isnan(prop):
        ax.text(0.98, 0.98, f"Proportion Mediated: {prop*100:.1f}%",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_ylabel('Effect Size', fontsize=12)
    ax.set_title('Causal Effect Decomposition', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Adjust y-axis to show zero and effects clearly
    y_min = min(0, min(values) - max(errors) * 1.5)
    y_max = max(0, max(values) + max(errors) * 1.5)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def create_combined_figure(
    results: Dict,
    sensitivity_results: Dict = None,
    treatment_name: str = "Treatment",
    mediator_name: str = "Mediator",
    outcome_name: str = "Outcome",
    figsize: Tuple[int, int] = (16, 12),
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create combined figure with pathway diagram and supporting plots.
    """
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

    # ===== Panel A: Pathway Diagram (top, spanning both columns) =====
    ax_pathway = fig.add_subplot(gs[0, :])
    ax_pathway.set_xlim(0, 12)
    ax_pathway.set_ylim(0, 6)
    ax_pathway.axis('off')

    # Simplified pathway drawing
    box_style = dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='black', linewidth=2)

    ax_pathway.text(2, 3, treatment_name, ha='center', va='center',
                    fontsize=11, fontweight='bold', bbox=box_style)
    ax_pathway.text(6, 5, mediator_name, ha='center', va='center',
                    fontsize=11, fontweight='bold', bbox=box_style)
    ax_pathway.text(10, 3, outcome_name, ha='center', va='center',
                    fontsize=11, fontweight='bold', bbox=box_style)

    arrow_props = dict(arrowstyle='-|>', color='black', lw=2, mutation_scale=12)

    ax_pathway.annotate('', xy=(5.2, 4.7), xytext=(2.8, 3.4), arrowprops=arrow_props)
    ax_pathway.annotate('', xy=(9.2, 3.4), xytext=(6.8, 4.7), arrowprops=arrow_props)
    ax_pathway.annotate('', xy=(9.2, 3), xytext=(2.8, 3),
                        arrowprops=dict(**arrow_props, linestyle='--', color='gray'))

    # Coefficients
    alpha = results.get('alpha', np.nan)
    beta_m = results.get('beta_m', np.nan)
    ade = results.get('ade', np.nan)

    ax_pathway.text(3.5, 4.5, f"a = {alpha:.3f}", fontsize=10, ha='center')
    ax_pathway.text(8.5, 4.5, f"b = {beta_m:.3f}", fontsize=10, ha='center')
    ax_pathway.text(6, 2.5, f"c' = {ade:.3f}", fontsize=10, ha='center', color='gray')

    ax_pathway.set_title("A. Causal Mediation Pathway", fontsize=13,
                         fontweight='bold', loc='left')

    # ===== Panel B: Effect Decomposition =====
    ax_decomp = fig.add_subplot(gs[1, 0])

    total = results.get('total_effect', np.nan)
    acme = results.get('acme', np.nan)
    prop = results.get('prop_mediated', np.nan)

    effects = ['Total', 'Direct\n(ADE)', 'Indirect\n(ACME)']
    values = [total, ade, acme]
    colors = ['steelblue', 'forestgreen', 'darkorange']

    bars = ax_decomp.bar(effects, values, color=colors, edgecolor='black', lw=1.5)

    for bar, val in zip(bars, values):
        ax_decomp.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax_decomp.axhline(0, color='black', lw=1)
    ax_decomp.set_ylabel('Effect Size', fontsize=11)
    ax_decomp.set_title("B. Effect Decomposition", fontsize=13,
                        fontweight='bold', loc='left')
    ax_decomp.grid(axis='y', alpha=0.3)

    if not np.isnan(prop):
        ax_decomp.text(0.98, 0.98, f"% Mediated: {prop*100:.1f}%",
                       transform=ax_decomp.transAxes, ha='right', va='top',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # ===== Panel C: Sensitivity or Additional Info =====
    ax_sens = fig.add_subplot(gs[1, 1])

    if sensitivity_results:
        rho = np.array(sensitivity_results.get('rho_range', [-0.5, 0.5]))
        acme_vals = np.array(sensitivity_results.get('acme_values', [acme, acme]))
        breakpoint = sensitivity_results.get('breakpoint', np.nan)

        ax_sens.plot(rho, acme_vals, 'b-', lw=2)
        ax_sens.axhline(0, color='red', ls='--', lw=1.5)
        ax_sens.axvline(0, color='gray', ls=':', lw=1)

        if not np.isnan(breakpoint) and rho.min() < breakpoint < rho.max():
            ax_sens.axvline(breakpoint, color='orange', ls='--', lw=2)
            ax_sens.scatter([breakpoint], [0], color='orange', s=80, zorder=5)

        ax_sens.set_xlabel('Sensitivity Parameter (rho)', fontsize=11)
        ax_sens.set_ylabel('ACME', fontsize=11)
        ax_sens.set_title("C. Sensitivity Analysis", fontsize=13,
                          fontweight='bold', loc='left')
        ax_sens.grid(alpha=0.3)
    else:
        # Show effect summary instead
        ax_sens.axis('off')
        summary = (
            f"Effect Summary\n"
            f"{'='*30}\n\n"
            f"Total Effect: {total:.4f}\n"
            f"Direct Effect (ADE): {ade:.4f}\n"
            f"Indirect Effect (ACME): {acme:.4f}\n\n"
            f"Proportion Mediated: {prop*100:.1f}%\n"
            f"{'='*30}"
        )
        ax_sens.text(0.5, 0.5, summary, ha='center', va='center',
                     fontsize=11, family='monospace', transform=ax_sens.transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
        ax_sens.set_title("C. Effect Summary", fontsize=13,
                          fontweight='bold', loc='left')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    return fig


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

    # Run mediation analysis
    if args.verbose:
        print("Running mediation analysis...")

    results = estimate_baron_kenny(
        data, args.outcome, args.treatment, args.mediator, args.controls
    )

    # Get labels
    treatment_label = args.treatment_label or args.treatment
    mediator_label = args.mediator_label or args.mediator
    outcome_label = args.outcome_label or args.outcome

    # Create visualizations
    output_path = Path(args.output)

    if args.plot_type == 'pathway':
        if args.verbose:
            print("Creating pathway diagram...")

        create_pathway_diagram(
            results, treatment_label, mediator_label, outcome_label,
            style=args.style, save_path=str(output_path), dpi=args.dpi
        )

    elif args.plot_type == 'decomposition':
        if args.verbose:
            print("Creating decomposition chart...")

        create_decomposition_chart(
            results, save_path=str(output_path), dpi=args.dpi
        )

    elif args.plot_type == 'combined':
        if args.verbose:
            print("Creating combined figure...")

        # Also run sensitivity analysis
        med_data = create_mediation_data(
            data, args.outcome, args.treatment, args.mediator, args.controls
        )
        sensitivity = sensitivity_analysis_mediation(
            results['acme'], results['acme_se'],
            sigma_m=med_data.m.std(), sigma_y=med_data.y.std()
        )

        create_combined_figure(
            results, sensitivity,
            treatment_label, mediator_label, outcome_label,
            save_path=str(output_path), dpi=args.dpi
        )

    elif args.plot_type == 'all':
        if args.verbose:
            print("Creating all visualizations...")

        stem = output_path.stem
        suffix = output_path.suffix

        # Pathway
        create_pathway_diagram(
            results, treatment_label, mediator_label, outcome_label,
            style=args.style,
            save_path=str(output_path.parent / f"{stem}_pathway{suffix}"),
            dpi=args.dpi
        )

        # Decomposition
        create_decomposition_chart(
            results,
            save_path=str(output_path.parent / f"{stem}_decomposition{suffix}"),
            dpi=args.dpi
        )

        # Combined
        med_data = create_mediation_data(
            data, args.outcome, args.treatment, args.mediator, args.controls
        )
        sensitivity = sensitivity_analysis_mediation(
            results['acme'], results['acme_se'],
            sigma_m=med_data.m.std(), sigma_y=med_data.y.std()
        )
        create_combined_figure(
            results, sensitivity,
            treatment_label, mediator_label, outcome_label,
            save_path=str(output_path.parent / f"{stem}_combined{suffix}"),
            dpi=args.dpi
        )

    print(f"Visualization saved to: {args.output}")


if __name__ == '__main__':
    main()
