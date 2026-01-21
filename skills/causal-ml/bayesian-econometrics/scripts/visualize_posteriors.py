#!/usr/bin/env python
"""
Visualize Posteriors

Create publication-quality visualizations of posterior distributions.
Supports various plot types and customization options.

Usage:
    python visualize_posteriors.py --trace trace.nc --vars beta,ate
    python visualize_posteriors.py --trace trace.nc --type forest --output plot.pdf
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import warnings

try:
    import arviz as az
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize posterior distributions from Bayesian models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Plot types:
  posterior  - Density plots with HDI and point estimates
  forest     - Forest plot comparing parameters
  trace      - Trace and density plots side by side
  pair       - Pair plots showing correlations
  ridgeplot  - Ridge plot for comparing distributions
  kde        - Kernel density estimates
  hist       - Histograms

Examples:
  # Basic posterior plot
  python visualize_posteriors.py --trace model.nc --vars beta,sigma

  # Forest plot for comparing effects
  python visualize_posteriors.py --trace model.nc --vars beta --type forest

  # High-quality PDF output
  python visualize_posteriors.py --trace model.nc --vars ate --output posterior.pdf --dpi 300

  # Customized appearance
  python visualize_posteriors.py --trace model.nc --vars beta --hdi-prob 0.9 --ref-val 0 --style seaborn-whitegrid
        """
    )

    parser.add_argument(
        "--trace", "-t",
        required=True,
        help="Path to trace file (NetCDF format)"
    )
    parser.add_argument(
        "--vars", "-v",
        default=None,
        help="Comma-separated list of variables to plot"
    )
    parser.add_argument(
        "--type", "-p",
        default="posterior",
        choices=["posterior", "forest", "trace", "pair", "ridgeplot", "kde", "hist"],
        help="Type of plot (default: posterior)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: display)"
    )
    parser.add_argument(
        "--format", "-f",
        default=None,
        help="Output format (inferred from output path if not specified)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for raster formats (default: 150)"
    )
    parser.add_argument(
        "--hdi-prob",
        type=float,
        default=0.95,
        help="Probability for HDI shading (default: 0.95)"
    )
    parser.add_argument(
        "--ref-val",
        type=float,
        default=None,
        help="Reference value to mark (e.g., 0 for null effect)"
    )
    parser.add_argument(
        "--rope",
        type=str,
        default=None,
        help="Region of practical equivalence as 'low,high' (e.g., '-0.1,0.1')"
    )
    parser.add_argument(
        "--point-estimate",
        default="mean",
        choices=["mean", "median", "mode"],
        help="Point estimate to show (default: mean)"
    )
    parser.add_argument(
        "--style",
        default="default",
        help="Matplotlib style (default, seaborn, ggplot, etc.)"
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title"
    )
    parser.add_argument(
        "--figsize",
        default=None,
        help="Figure size as 'width,height' in inches"
    )
    parser.add_argument(
        "--colors",
        default=None,
        help="Color palette name or comma-separated colors"
    )

    return parser.parse_args()


def setup_style(style, colors=None):
    """Set up matplotlib style."""
    try:
        plt.style.use(style)
    except:
        pass  # Use default if style not found

    if colors:
        if ',' in colors:
            color_list = [c.strip() for c in colors.split(',')]
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_list)
        else:
            try:
                import seaborn as sns
                sns.set_palette(colors)
            except:
                pass


def parse_figsize(figsize_str):
    """Parse figure size string."""
    if figsize_str:
        parts = figsize_str.split(',')
        return (float(parts[0]), float(parts[1]))
    return None


def parse_rope(rope_str):
    """Parse ROPE string."""
    if rope_str:
        parts = rope_str.split(',')
        return (float(parts[0]), float(parts[1]))
    return None


def create_posterior_plot(trace, var_names, args):
    """Create posterior density plots."""
    figsize = parse_figsize(args.figsize) or (12, 3 * len(var_names) if var_names else 12)
    rope = parse_rope(args.rope)

    fig = plt.figure(figsize=figsize)
    az.plot_posterior(
        trace,
        var_names=var_names,
        hdi_prob=args.hdi_prob,
        point_estimate=args.point_estimate,
        ref_val=args.ref_val,
        rope=rope,
        figsize=figsize
    )

    if args.title:
        plt.suptitle(args.title, y=1.02)

    plt.tight_layout()
    return fig


def create_forest_plot(trace, var_names, args):
    """Create forest plot."""
    figsize = parse_figsize(args.figsize) or (10, 6)
    rope = parse_rope(args.rope)

    fig, ax = plt.subplots(figsize=figsize)
    az.plot_forest(
        trace,
        var_names=var_names,
        combined=True,
        hdi_prob=args.hdi_prob,
        r_hat=True,
        ess=True,
        ax=ax
    )

    if args.ref_val is not None:
        ax.axvline(args.ref_val, color='red', linestyle='--', alpha=0.5)

    if rope:
        ax.axvspan(rope[0], rope[1], alpha=0.1, color='gray')

    if args.title:
        ax.set_title(args.title)

    plt.tight_layout()
    return fig


def create_trace_plot(trace, var_names, args):
    """Create trace and density plots."""
    figsize = parse_figsize(args.figsize)

    fig = plt.figure(figsize=figsize or (12, 3 * len(var_names) if var_names else 12))
    az.plot_trace(trace, var_names=var_names, combined=False)

    if args.title:
        plt.suptitle(args.title, y=1.02)

    plt.tight_layout()
    return fig


def create_pair_plot(trace, var_names, args):
    """Create pair plot for correlations."""
    figsize = parse_figsize(args.figsize) or (10, 10)

    if var_names and len(var_names) > 6:
        print("Warning: Limiting pair plot to first 6 variables")
        var_names = var_names[:6]

    fig = plt.figure(figsize=figsize)
    az.plot_pair(
        trace,
        var_names=var_names,
        kind=["scatter", "kde"],
        divergences=True,
        marginals=True,
        figsize=figsize
    )

    if args.title:
        plt.suptitle(args.title, y=1.02)

    return fig


def create_ridgeplot(trace, var_names, args):
    """Create ridgeplot for comparing distributions."""
    figsize = parse_figsize(args.figsize) or (10, 6)

    fig, ax = plt.subplots(figsize=figsize)
    az.plot_violin(
        trace,
        var_names=var_names,
        combined=True,
        ax=ax
    )

    if args.ref_val is not None:
        ax.axhline(args.ref_val, color='red', linestyle='--', alpha=0.5)

    if args.title:
        ax.set_title(args.title)

    plt.tight_layout()
    return fig


def create_kde_plot(trace, var_names, args):
    """Create KDE plots."""
    figsize = parse_figsize(args.figsize) or (10, 6)

    fig, ax = plt.subplots(figsize=figsize)

    for var in var_names:
        if var in trace.posterior:
            samples = trace.posterior[var].values.flatten()
            az.plot_kde(samples, ax=ax, label=var, fill_kwargs={"alpha": 0.3})

    if args.ref_val is not None:
        ax.axvline(args.ref_val, color='red', linestyle='--', alpha=0.5, label='Reference')

    rope = parse_rope(args.rope)
    if rope:
        ax.axvspan(rope[0], rope[1], alpha=0.1, color='gray', label='ROPE')

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

    if args.title:
        ax.set_title(args.title)

    plt.tight_layout()
    return fig


def create_hist_plot(trace, var_names, args):
    """Create histogram plots."""
    n_vars = len(var_names) if var_names else 1
    figsize = parse_figsize(args.figsize) or (10, 3 * n_vars)

    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, var in enumerate(var_names or trace.posterior.data_vars):
        ax = axes[i]
        if var in trace.posterior:
            samples = trace.posterior[var].values.flatten()
            ax.hist(samples, bins=50, density=True, alpha=0.7, color='steelblue')

            # Add HDI
            hdi = az.hdi(samples, hdi_prob=args.hdi_prob)
            ax.axvline(hdi[0], color='orange', linestyle='--', alpha=0.7)
            ax.axvline(hdi[1], color='orange', linestyle='--', alpha=0.7)

            # Add point estimate
            if args.point_estimate == "mean":
                point = samples.mean()
            elif args.point_estimate == "median":
                point = np.median(samples)
            else:
                from scipy import stats
                point = float(stats.mode(samples, keepdims=True)[0][0])
            ax.axvline(point, color='red', linestyle='-', alpha=0.8)

            if args.ref_val is not None:
                ax.axvline(args.ref_val, color='black', linestyle=':', alpha=0.5)

            ax.set_xlabel(var)
            ax.set_ylabel("Density")

    if args.title:
        plt.suptitle(args.title, y=1.02)

    plt.tight_layout()
    return fig


def create_comparison_plot(trace, var_names, args):
    """Create a comprehensive comparison visualization."""
    figsize = parse_figsize(args.figsize) or (14, 10)
    rope = parse_rope(args.rope)

    fig = plt.figure(figsize=figsize)

    # Grid layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Posterior densities
    ax1 = fig.add_subplot(gs[0, 0])
    for var in var_names:
        if var in trace.posterior:
            samples = trace.posterior[var].values.flatten()
            az.plot_kde(samples, ax=ax1, label=var, fill_kwargs={"alpha": 0.3})
    ax1.set_title("Posterior Densities")
    ax1.legend()

    # Forest plot
    ax2 = fig.add_subplot(gs[0, 1])
    az.plot_forest(trace, var_names=var_names, combined=True, ax=ax2)
    ax2.set_title("Forest Plot")

    # Trace plots
    ax3 = fig.add_subplot(gs[1, 0])
    for var in var_names[:2]:  # Limit to first 2 for space
        if var in trace.posterior:
            for chain in range(trace.posterior.dims.get('chain', 1)):
                chain_samples = trace.posterior[var].sel(chain=chain).values.flatten()
                ax3.plot(chain_samples, alpha=0.5, label=f"{var} (chain {chain})")
    ax3.set_title("Trace (first 2 variables)")
    ax3.set_xlabel("Iteration")

    # Summary table as text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    summary = az.summary(trace, var_names=var_names, hdi_prob=args.hdi_prob)
    summary_text = summary.round(3).to_string()
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontfamily='monospace', fontsize=8, verticalalignment='top')
    ax4.set_title("Summary Statistics")

    if args.title:
        plt.suptitle(args.title, fontsize=14, y=1.02)

    return fig


def main():
    """Main entry point."""
    if not HAS_DEPS:
        print("Error: Required packages not installed")
        print("Install with: pip install arviz matplotlib")
        sys.exit(1)

    args = parse_args()

    # Load trace
    print(f"Loading trace from {args.trace}...")
    try:
        trace = az.from_netcdf(args.trace)
    except Exception as e:
        print(f"Error loading trace: {e}")
        sys.exit(1)

    # Parse variable names
    var_names = None
    if args.vars:
        var_names = [v.strip() for v in args.vars.split(",")]
        print(f"Plotting variables: {var_names}")
    else:
        var_names = list(trace.posterior.data_vars.keys())
        print(f"Plotting all variables: {var_names}")

    # Setup style
    setup_style(args.style, args.colors)

    # Create plot
    print(f"Creating {args.type} plot...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if args.type == "posterior":
            fig = create_posterior_plot(trace, var_names, args)
        elif args.type == "forest":
            fig = create_forest_plot(trace, var_names, args)
        elif args.type == "trace":
            fig = create_trace_plot(trace, var_names, args)
        elif args.type == "pair":
            fig = create_pair_plot(trace, var_names, args)
        elif args.type == "ridgeplot":
            fig = create_ridgeplot(trace, var_names, args)
        elif args.type == "kde":
            fig = create_kde_plot(trace, var_names, args)
        elif args.type == "hist":
            fig = create_hist_plot(trace, var_names, args)
        else:
            print(f"Unknown plot type: {args.type}")
            sys.exit(1)

    # Save or display
    if args.output:
        fmt = args.format
        if fmt is None:
            fmt = Path(args.output).suffix[1:]  # Remove leading dot

        plt.savefig(args.output, format=fmt, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved to: {args.output}")
    else:
        plt.show()

    print("Done!")


if __name__ == "__main__":
    main()
