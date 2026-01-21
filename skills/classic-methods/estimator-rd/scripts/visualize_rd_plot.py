#!/usr/bin/env python3
"""
RD Plot Generator with Binned Scatter.

This script generates publication-quality Regression Discontinuity plots
following best practices from the methodological literature.

Features:
1. Binned scatter plots
2. Local polynomial fitted lines
3. Confidence intervals
4. Customizable aesthetics
5. Multiple output formats

Usage:
    python visualize_rd_plot.py data.csv --running score --outcome y --cutoff 0

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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from rd_estimator import select_bandwidth, get_kernel


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate RD plots with binned scatter"
    )

    parser.add_argument("data_file", type=str, help="Path to CSV data file")
    parser.add_argument("--running", type=str, required=True, help="Running variable column")
    parser.add_argument("--outcome", type=str, required=True, help="Outcome variable column")
    parser.add_argument("--cutoff", type=float, required=True, help="Cutoff value")

    parser.add_argument("--bandwidth", type=float, default=None, help="Bandwidth for polynomial fit")
    parser.add_argument("--n-bins", type=int, default=20, help="Number of bins (default: 20)")
    parser.add_argument("--poly-order", type=int, default=1, choices=[1, 2], help="Polynomial order")
    parser.add_argument("--kernel", type=str, default="triangular", help="Kernel function")

    parser.add_argument("--ci", action="store_true", default=True, help="Show confidence intervals")
    parser.add_argument("--no-ci", action="store_false", dest="ci", help="Hide confidence intervals")

    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--xlabel", type=str, default=None, help="X-axis label")
    parser.add_argument("--ylabel", type=str, default=None, help="Y-axis label")

    parser.add_argument("--figsize", type=str, default="10,6", help="Figure size as 'width,height'")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg", "eps"],
                       help="Output format")

    parser.add_argument("--output", type=str, default="rd_plot", help="Output filename (without extension)")
    parser.add_argument("--style", type=str, default="publication",
                       choices=["publication", "minimal", "colorful"],
                       help="Plot style preset")

    parser.add_argument("--show", action="store_true", help="Display plot interactively")

    return parser.parse_args()


def calculate_bin_statistics(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
    cutoff: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate binned statistics for scatter plot.

    Parameters
    ----------
    x : np.ndarray
        Running variable values
    y : np.ndarray
        Outcome variable values
    n_bins : int
        Number of bins
    cutoff : float
        Cutoff value

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (bin_centers, bin_means, bin_ses, bin_counts)
    """
    # Create bin edges
    bin_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_means = np.zeros(n_bins)
    bin_ses = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        in_bin = (x >= bin_edges[i]) & (x < bin_edges[i + 1])

        if in_bin.sum() > 0:
            y_bin = y[in_bin]
            bin_counts[i] = len(y_bin)
            bin_means[i] = np.mean(y_bin)
            bin_ses[i] = np.std(y_bin, ddof=1) / np.sqrt(len(y_bin)) if len(y_bin) > 1 else 0
        else:
            bin_means[i] = np.nan
            bin_ses[i] = np.nan

    return bin_centers, bin_means, bin_ses, bin_counts


def fit_local_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    cutoff: float,
    bandwidth: float,
    order: int = 1,
    side: str = "below"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit local polynomial for plotting.

    Parameters
    ----------
    x : np.ndarray
        Running variable
    y : np.ndarray
        Outcome variable
    cutoff : float
        Cutoff value
    bandwidth : float
        Bandwidth
    order : int
        Polynomial order
    side : str
        "below" or "above"

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (x_grid, y_fit, y_se)
    """
    # Select data for this side
    if side == "below":
        mask = x < cutoff
        x_min = max(x.min(), cutoff - bandwidth)
        x_max = cutoff - 0.001
    else:
        mask = x >= cutoff
        x_min = cutoff + 0.001
        x_max = min(x.max(), cutoff + bandwidth)

    x_side = x[mask]
    y_side = y[mask]

    if len(x_side) < order + 2:
        return np.array([]), np.array([]), np.array([])

    # Center at cutoff
    x_centered = x_side - cutoff

    # Build design matrix
    X = np.column_stack([x_centered**p for p in range(order + 1)])

    try:
        # Fit OLS
        beta = np.linalg.lstsq(X, y_side, rcond=None)[0]

        # Residuals and variance
        resid = y_side - X @ beta
        sigma2 = np.var(resid, ddof=order + 1)

        # Prediction grid
        x_grid = np.linspace(x_min, x_max, 100)
        x_grid_centered = x_grid - cutoff

        X_pred = np.column_stack([x_grid_centered**p for p in range(order + 1)])
        y_fit = X_pred @ beta

        # Standard errors (simplified)
        XtX_inv = np.linalg.inv(X.T @ X)
        var_pred = sigma2 * np.diag(X_pred @ XtX_inv @ X_pred.T)
        y_se = np.sqrt(var_pred)

        return x_grid, y_fit, y_se

    except np.linalg.LinAlgError:
        return np.array([]), np.array([]), np.array([])


def get_style_config(style: str) -> dict:
    """Get style configuration for plots."""
    styles = {
        "publication": {
            "color_below": "#2166AC",  # Blue
            "color_above": "#B2182B",  # Red
            "color_cutoff": "black",
            "marker_size": 60,
            "line_width": 2,
            "font_size": 12,
            "spine_color": "black",
            "grid_alpha": 0.3,
            "fill_alpha": 0.2
        },
        "minimal": {
            "color_below": "#666666",
            "color_above": "#666666",
            "color_cutoff": "#333333",
            "marker_size": 40,
            "line_width": 1.5,
            "font_size": 11,
            "spine_color": "#cccccc",
            "grid_alpha": 0.15,
            "fill_alpha": 0.15
        },
        "colorful": {
            "color_below": "#1f77b4",
            "color_above": "#ff7f0e",
            "color_cutoff": "#2ca02c",
            "marker_size": 80,
            "line_width": 2.5,
            "font_size": 13,
            "spine_color": "black",
            "grid_alpha": 0.4,
            "fill_alpha": 0.25
        }
    }
    return styles.get(style, styles["publication"])


def create_rd_plot(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    cutoff: float,
    bandwidth: Optional[float] = None,
    n_bins: int = 20,
    poly_order: int = 1,
    show_ci: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    style: str = "publication"
):
    """
    Create publication-quality RD plot.

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
    bandwidth : float, optional
        Bandwidth for polynomial fit
    n_bins : int
        Number of bins
    poly_order : int
        Polynomial order
    show_ci : bool
        Show confidence intervals
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    figsize : tuple
        Figure size
    style : str
        Style preset

    Returns
    -------
    matplotlib.figure.Figure
        The RD plot figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")

    # Get style configuration
    config = get_style_config(style)

    # Extract and clean data
    x = data[running].values
    y = data[outcome].values

    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    # Select bandwidth if not provided
    if bandwidth is None:
        bandwidth = select_bandwidth(x, y, cutoff)

    # Calculate binned statistics
    bin_centers, bin_means, bin_ses, bin_counts = calculate_bin_statistics(x, y, n_bins, cutoff)

    # Separate bins by side
    below = bin_centers < cutoff
    above = ~below

    # Fit polynomials
    x_fit_below, y_fit_below, se_fit_below = fit_local_polynomial(
        x, y, cutoff, bandwidth, poly_order, "below"
    )
    x_fit_above, y_fit_above, se_fit_above = fit_local_polynomial(
        x, y, cutoff, bandwidth, poly_order, "above"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot binned scatter with error bars
    valid_below = ~np.isnan(bin_means[below])
    valid_above = ~np.isnan(bin_means[above])

    if show_ci:
        ax.errorbar(
            bin_centers[below][valid_below],
            bin_means[below][valid_below],
            yerr=1.96 * bin_ses[below][valid_below],
            fmt='o',
            color=config["color_below"],
            markersize=np.sqrt(config["marker_size"]),
            capsize=3,
            capthick=1,
            elinewidth=1,
            alpha=0.8,
            label='Below cutoff'
        )
        ax.errorbar(
            bin_centers[above][valid_above],
            bin_means[above][valid_above],
            yerr=1.96 * bin_ses[above][valid_above],
            fmt='o',
            color=config["color_above"],
            markersize=np.sqrt(config["marker_size"]),
            capsize=3,
            capthick=1,
            elinewidth=1,
            alpha=0.8,
            label='Above cutoff'
        )
    else:
        ax.scatter(
            bin_centers[below][valid_below],
            bin_means[below][valid_below],
            s=config["marker_size"],
            color=config["color_below"],
            alpha=0.8,
            label='Below cutoff'
        )
        ax.scatter(
            bin_centers[above][valid_above],
            bin_means[above][valid_above],
            s=config["marker_size"],
            color=config["color_above"],
            alpha=0.8,
            label='Above cutoff'
        )

    # Plot fitted polynomials
    if len(x_fit_below) > 0:
        ax.plot(x_fit_below, y_fit_below, color=config["color_below"],
                linewidth=config["line_width"])
        if show_ci and len(se_fit_below) > 0:
            ax.fill_between(
                x_fit_below,
                y_fit_below - 1.96 * se_fit_below,
                y_fit_below + 1.96 * se_fit_below,
                color=config["color_below"],
                alpha=config["fill_alpha"]
            )

    if len(x_fit_above) > 0:
        ax.plot(x_fit_above, y_fit_above, color=config["color_above"],
                linewidth=config["line_width"])
        if show_ci and len(se_fit_above) > 0:
            ax.fill_between(
                x_fit_above,
                y_fit_above - 1.96 * se_fit_above,
                y_fit_above + 1.96 * se_fit_above,
                color=config["color_above"],
                alpha=config["fill_alpha"]
            )

    # Cutoff line
    ax.axvline(x=cutoff, color=config["color_cutoff"], linestyle='--',
               linewidth=1.5, label=f'Cutoff = {cutoff}')

    # Bandwidth markers
    ax.axvline(x=cutoff - bandwidth, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=cutoff + bandwidth, color='gray', linestyle=':', alpha=0.5)

    # Labels
    ax.set_xlabel(xlabel or running, fontsize=config["font_size"])
    ax.set_ylabel(ylabel or f'Mean {outcome}', fontsize=config["font_size"])

    if title:
        ax.set_title(title, fontsize=config["font_size"] + 2)

    # Legend
    ax.legend(loc='best', frameon=True, fontsize=config["font_size"] - 1)

    # Grid
    ax.grid(True, alpha=config["grid_alpha"])

    # Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(config["spine_color"])

    # Tick parameters
    ax.tick_params(labelsize=config["font_size"] - 1)

    # Add annotation for bandwidth
    ax.text(
        0.02, 0.98,
        f'Bandwidth: {bandwidth:.4f}\nBins: {n_bins}\nPolynomial: order {poly_order}',
        transform=ax.transAxes,
        fontsize=config["font_size"] - 2,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    return fig


def create_multi_panel_rd_plot(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    cutoff: float,
    bandwidth: Optional[float] = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Create multi-panel RD diagnostic plot.

    Includes:
    1. Main RD plot
    2. Running variable histogram
    3. Bandwidth sensitivity
    4. Polynomial order comparison
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required")

    x = data[running].values
    y = data[outcome].values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if bandwidth is None:
        bandwidth = select_bandwidth(x, y, cutoff)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Main RD plot
    ax1 = axes[0, 0]
    bin_centers, bin_means, bin_ses, _ = calculate_bin_statistics(x, y, 20, cutoff)
    below = bin_centers < cutoff

    ax1.scatter(bin_centers[below], bin_means[below], color='steelblue', s=50, alpha=0.7)
    ax1.scatter(bin_centers[~below], bin_means[~below], color='indianred', s=50, alpha=0.7)
    ax1.axvline(x=cutoff, color='black', linestyle='--', linewidth=1.5)
    ax1.set_xlabel(running)
    ax1.set_ylabel(f'Mean {outcome}')
    ax1.set_title('A. Main RD Plot')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Histogram
    ax2 = axes[0, 1]
    ax2.hist(x[x < cutoff], bins=25, color='steelblue', alpha=0.7, label='Below')
    ax2.hist(x[x >= cutoff], bins=25, color='indianred', alpha=0.7, label='Above')
    ax2.axvline(x=cutoff, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel(running)
    ax2.set_ylabel('Frequency')
    ax2.set_title('B. Distribution of Running Variable')
    ax2.legend()

    # Panel 3: Effect by bandwidth
    ax3 = axes[1, 0]
    bandwidths = [0.5*bandwidth, 0.75*bandwidth, bandwidth, 1.25*bandwidth, 1.5*bandwidth, 2*bandwidth]
    effects = []
    ses = []

    from rd_estimator import estimate_sharp_rd

    for bw in bandwidths:
        try:
            result = estimate_sharp_rd(data, running, outcome, cutoff, bw)
            effects.append(result.effect)
            ses.append(result.se)
        except:
            effects.append(np.nan)
            ses.append(np.nan)

    effects = np.array(effects)
    ses = np.array(ses)

    ax3.errorbar(bandwidths, effects, yerr=1.96*ses, fmt='o-', color='steelblue',
                 capsize=4, markersize=8)
    ax3.axvline(x=bandwidth, color='red', linestyle=':', label=f'Optimal h={bandwidth:.3f}')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Bandwidth')
    ax3.set_ylabel('RD Effect')
    ax3.set_title('C. Bandwidth Sensitivity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Linear vs Quadratic
    ax4 = axes[1, 1]

    # Fit both orders
    x_fit_b1, y_fit_b1, _ = fit_local_polynomial(x, y, cutoff, bandwidth, 1, "below")
    x_fit_a1, y_fit_a1, _ = fit_local_polynomial(x, y, cutoff, bandwidth, 1, "above")
    x_fit_b2, y_fit_b2, _ = fit_local_polynomial(x, y, cutoff, bandwidth, 2, "below")
    x_fit_a2, y_fit_a2, _ = fit_local_polynomial(x, y, cutoff, bandwidth, 2, "above")

    ax4.scatter(bin_centers[below], bin_means[below], color='gray', s=30, alpha=0.5)
    ax4.scatter(bin_centers[~below], bin_means[~below], color='gray', s=30, alpha=0.5)

    if len(x_fit_b1) > 0:
        ax4.plot(x_fit_b1, y_fit_b1, 'b-', linewidth=2, label='Linear (p=1)')
        ax4.plot(x_fit_a1, y_fit_a1, 'b-', linewidth=2)
    if len(x_fit_b2) > 0:
        ax4.plot(x_fit_b2, y_fit_b2, 'r--', linewidth=2, label='Quadratic (p=2)')
        ax4.plot(x_fit_a2, y_fit_a2, 'r--', linewidth=2)

    ax4.axvline(x=cutoff, color='black', linestyle='--', linewidth=1.5)
    ax4.set_xlabel(running)
    ax4.set_ylabel(f'Mean {outcome}')
    ax4.set_title('D. Linear vs Quadratic Specification')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main entry point."""
    args = parse_args()

    # Load data
    if not Path(args.data_file).exists():
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)

    data = pd.read_csv(args.data_file)

    # Parse figure size
    figsize = tuple(map(int, args.figsize.split(',')))

    # Create plot
    fig = create_rd_plot(
        data=data,
        running=args.running,
        outcome=args.outcome,
        cutoff=args.cutoff,
        bandwidth=args.bandwidth,
        n_bins=args.n_bins,
        poly_order=args.poly_order,
        show_ci=args.ci,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        figsize=figsize,
        style=args.style
    )

    # Save plot
    output_file = f"{args.output}.{args.format}"
    fig.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Show if requested
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
