#!/usr/bin/env python3
"""
Statistical Distribution Visualization CLI

Create visualizations for understanding data distributions, comparing groups,
and assessing statistical test assumptions.

Usage:
    python visualize_distributions.py --data data.csv --vars income,age --output figures/
    python visualize_distributions.py --data data.csv --outcome income --group treatment --type comparison
    python visualize_distributions.py --data data.csv --vars income --type qq
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV or other formats."""
    path = Path(filepath)

    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    elif path.suffix == '.dta':
        return pd.read_stata(path)
    elif path.suffix == '.parquet':
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def plot_histogram(
    df: pd.DataFrame,
    var: str,
    output_path: Path,
    title: str = None,
    bins: int = 30
):
    """Create histogram with density curve."""
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, ax = plt.subplots(figsize=(10, 6))

    data = df[var].dropna()

    # Histogram
    n, bins_edges, patches = ax.hist(data, bins=bins, density=True, alpha=0.7,
                                      color='steelblue', edgecolor='white')

    # Fit and plot normal distribution
    mu, std = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    pdf = stats.norm.pdf(x, mu, std)
    ax.plot(x, pdf, 'r-', linewidth=2, label=f'Normal fit (mu={mu:.2f}, sigma={std:.2f})')

    # Vertical line at mean
    ax.axvline(mu, color='red', linestyle='--', alpha=0.8, label=f'Mean = {mu:.2f}')
    ax.axvline(data.median(), color='green', linestyle='--', alpha=0.8, label=f'Median = {data.median():.2f}')

    ax.set_xlabel(var)
    ax.set_ylabel('Density')
    ax.set_title(title or f'Distribution of {var}')
    ax.legend()

    # Add statistics box
    stats_text = (f'N = {len(data)}\n'
                  f'Mean = {mu:.2f}\n'
                  f'SD = {std:.2f}\n'
                  f'Skew = {stats.skew(data):.2f}\n'
                  f'Kurt = {stats.kurtosis(data):.2f}')
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path / f'{var}_histogram.png', dpi=150)
    plt.close()

    print(f"Saved: {output_path / f'{var}_histogram.png'}")


def plot_qq(
    df: pd.DataFrame,
    var: str,
    output_path: Path,
    title: str = None
):
    """Create Q-Q plot for normality assessment."""
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, ax = plt.subplots(figsize=(8, 8))

    data = df[var].dropna()

    # Q-Q plot
    stats.probplot(data, dist="norm", plot=ax)

    ax.set_title(title or f'Q-Q Plot: {var}')

    # Add normality test result
    shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Limit for computational reasons
    ax.text(0.05, 0.95, f'Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path / f'{var}_qq.png', dpi=150)
    plt.close()

    print(f"Saved: {output_path / f'{var}_qq.png'}")


def plot_boxplot_comparison(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    output_path: Path,
    title: str = None
):
    """Create boxplot comparing groups."""
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, ax = plt.subplots(figsize=(10, 6))

    groups = df[group].unique()
    data = [df[df[group] == g][outcome].dropna() for g in groups]

    bp = ax.boxplot(data, labels=groups, patch_artist=True)

    # Color the boxes
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel(group)
    ax.set_ylabel(outcome)
    ax.set_title(title or f'{outcome} by {group}')

    # Add means as points
    means = [d.mean() for d in data]
    ax.scatter(range(1, len(groups) + 1), means, color='red', marker='D', s=50, zorder=3, label='Mean')

    # Add statistics
    stats_lines = []
    for i, (g, d) in enumerate(zip(groups, data)):
        stats_lines.append(f'{g}: n={len(d)}, mean={d.mean():.2f}, SD={d.std():.2f}')

    # If two groups, add t-test result
    if len(groups) == 2:
        t_stat, p_val = stats.ttest_ind(data[0], data[1])
        stats_lines.append(f't-test: t={t_stat:.2f}, p={p_val:.4f}')

    ax.text(0.02, 0.98, '\n'.join(stats_lines), transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / f'{outcome}_by_{group}_boxplot.png', dpi=150)
    plt.close()

    print(f"Saved: {output_path / f'{outcome}_by_{group}_boxplot.png'}")


def plot_density_comparison(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    output_path: Path,
    title: str = None
):
    """Create overlapping density plots for groups."""
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, ax = plt.subplots(figsize=(10, 6))

    groups = df[group].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))

    for g, color in zip(groups, colors):
        data = df[df[group] == g][outcome].dropna()

        # Kernel density estimation
        kde = stats.gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 200)
        ax.plot(x, kde(x), linewidth=2, label=f'{g} (n={len(data)}, mean={data.mean():.2f})', color=color)
        ax.fill_between(x, kde(x), alpha=0.3, color=color)

    ax.set_xlabel(outcome)
    ax.set_ylabel('Density')
    ax.set_title(title or f'Distribution of {outcome} by {group}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / f'{outcome}_by_{group}_density.png', dpi=150)
    plt.close()

    print(f"Saved: {output_path / f'{outcome}_by_{group}_density.png'}")


def plot_scatter_with_regression(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    output_path: Path,
    title: str = None
):
    """Create scatter plot with regression line."""
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, ax = plt.subplots(figsize=(10, 8))

    x = df[var1].dropna()
    y = df[var2].dropna()

    # Get common indices
    common_idx = x.index.intersection(y.index)
    x = x.loc[common_idx]
    y = y.loc[common_idx]

    # Scatter plot
    ax.scatter(x, y, alpha=0.5, s=20)

    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.3f}x + {intercept:.3f}')

    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title(title or f'{var2} vs {var1}')

    # Statistics box
    stats_text = (f'r = {r_value:.3f}\n'
                  f'r-squared = {r_value**2:.3f}\n'
                  f'p = {p_value:.4f}\n'
                  f'n = {len(x)}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / f'{var1}_vs_{var2}_scatter.png', dpi=150)
    plt.close()

    print(f"Saved: {output_path / f'{var1}_vs_{var2}_scatter.png'}")


def plot_correlation_matrix(
    df: pd.DataFrame,
    vars: List[str],
    output_path: Path,
    title: str = None
):
    """Create correlation matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 10))

    # Calculate correlation matrix
    corr = df[vars].corr()

    # Heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0, square=True, ax=ax)

    ax.set_title(title or 'Correlation Matrix')

    plt.tight_layout()
    plt.savefig(output_path / 'correlation_matrix.png', dpi=150)
    plt.close()

    print(f"Saved: {output_path / 'correlation_matrix.png'}")


def plot_power_curve(
    output_path: Path,
    effect_sizes: List[float] = None,
    sample_sizes: List[int] = None,
    alpha: float = 0.05
):
    """Create power curve visualization."""
    import matplotlib.pyplot as plt
    from scipy import stats

    if effect_sizes is None:
        effect_sizes = [0.2, 0.3, 0.5, 0.8]
    if sample_sizes is None:
        sample_sizes = list(range(10, 201, 10))

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(effect_sizes)))

    for d, color in zip(effect_sizes, colors):
        powers = []
        for n in sample_sizes:
            # Two-sample t-test power
            se = np.sqrt(2/n)
            ncp = d / se
            z_alpha = stats.norm.ppf(1 - alpha/2)
            power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
            powers.append(power)

        ax.plot(sample_sizes, powers, linewidth=2, label=f'd = {d}', color=color)

    ax.axhline(0.80, color='red', linestyle='--', alpha=0.5, label='80% power')
    ax.axhline(0.90, color='orange', linestyle='--', alpha=0.5, label='90% power')

    ax.set_xlabel('Sample Size (per group)')
    ax.set_ylabel('Statistical Power')
    ax.set_title(f'Power Curves for Two-Sample t-Test (alpha = {alpha})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path / 'power_curve.png', dpi=150)
    plt.close()

    print(f"Saved: {output_path / 'power_curve.png'}")


def plot_effect_size_interpretation(
    effect_size: float,
    output_path: Path,
    effect_type: str = 'cohens_d'
):
    """Visualize effect size interpretation."""
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, ax = plt.subplots(figsize=(10, 6))

    # Control distribution
    x = np.linspace(-4, 4 + effect_size, 1000)
    y_control = stats.norm.pdf(x, 0, 1)
    y_treatment = stats.norm.pdf(x, effect_size, 1)

    ax.fill_between(x, y_control, alpha=0.5, color='blue', label='Control')
    ax.fill_between(x, y_treatment, alpha=0.5, color='red', label='Treatment')
    ax.plot(x, y_control, 'b-', linewidth=2)
    ax.plot(x, y_treatment, 'r-', linewidth=2)

    # Overlap region
    overlap = np.minimum(y_control, y_treatment)
    ax.fill_between(x, overlap, alpha=0.3, color='purple', label='Overlap')

    ax.axvline(0, color='blue', linestyle='--', alpha=0.7)
    ax.axvline(effect_size, color='red', linestyle='--', alpha=0.7)

    ax.set_xlabel('Standardized Score')
    ax.set_ylabel('Density')
    ax.set_title(f"Effect Size Visualization (Cohen's d = {effect_size:.2f})")
    ax.legend()

    # Add interpretation
    if abs(effect_size) < 0.2:
        interp = "Negligible"
    elif abs(effect_size) < 0.5:
        interp = "Small"
    elif abs(effect_size) < 0.8:
        interp = "Medium"
    else:
        interp = "Large"

    # Calculate overlap percentage
    overlap_coef = 2 * stats.norm.cdf(-abs(effect_size)/2)
    non_overlap = 1 - overlap_coef

    ax.text(0.02, 0.98, f'Interpretation: {interp}\nNon-overlap: {non_overlap*100:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path / f'effect_size_d{effect_size:.1f}.png', dpi=150)
    plt.close()

    print(f"Saved: {output_path / f'effect_size_d{effect_size:.1f}.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Create statistical distribution visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Histograms for multiple variables
  python visualize_distributions.py --data data.csv --vars income,age,education --output figures/

  # Q-Q plots for normality
  python visualize_distributions.py --data data.csv --vars income --type qq --output figures/

  # Group comparison (boxplot and density)
  python visualize_distributions.py --data data.csv --outcome income --group treatment --type comparison --output figures/

  # Correlation matrix
  python visualize_distributions.py --data data.csv --vars x1,x2,x3,y --type correlation --output figures/

  # Scatter with regression
  python visualize_distributions.py --data data.csv --var1 education --var2 income --type scatter --output figures/

  # Power curves
  python visualize_distributions.py --type power --output figures/

  # Effect size visualization
  python visualize_distributions.py --type effect_size --effect-size 0.5 --output figures/

Visualization Types:
  histogram: Distribution with density curve
  qq: Q-Q plot for normality assessment
  comparison: Boxplot and density comparison by group
  scatter: Scatter plot with regression line
  correlation: Correlation matrix heatmap
  power: Power curve visualization
  effect_size: Effect size interpretation visualization
        """
    )

    # Data input
    parser.add_argument('--data', help='Path to data file')

    # Variable specification
    parser.add_argument('--vars', help='Variables to visualize (comma-separated)')
    parser.add_argument('--outcome', help='Outcome variable')
    parser.add_argument('--group', help='Grouping variable')
    parser.add_argument('--var1', help='First variable (scatter)')
    parser.add_argument('--var2', help='Second variable (scatter)')

    # Visualization type
    parser.add_argument('--type', default='histogram',
                       choices=['histogram', 'qq', 'comparison', 'scatter',
                               'correlation', 'power', 'effect_size'],
                       help='Visualization type')

    # Options
    parser.add_argument('--bins', type=int, default=30, help='Number of bins for histogram')
    parser.add_argument('--effect-size', type=float, help='Effect size to visualize')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level for power curves')

    # Output
    parser.add_argument('--output', required=True, help='Output directory')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check for matplotlib
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
    except ImportError:
        print("Error: matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    # Load data if needed
    if args.type not in ['power', 'effect_size']:
        if not args.data:
            print("Error: --data required for this visualization type", file=sys.stderr)
            sys.exit(1)
        df = load_data(args.data)
        print(f"Loaded {len(df)} observations")

    # Generate visualizations
    if args.type == 'histogram':
        if not args.vars:
            print("Error: --vars required for histogram", file=sys.stderr)
            sys.exit(1)
        variables = [v.strip() for v in args.vars.split(',')]
        for var in variables:
            plot_histogram(df, var, output_path, bins=args.bins)

    elif args.type == 'qq':
        if not args.vars:
            print("Error: --vars required for Q-Q plots", file=sys.stderr)
            sys.exit(1)
        variables = [v.strip() for v in args.vars.split(',')]
        for var in variables:
            plot_qq(df, var, output_path)

    elif args.type == 'comparison':
        if not args.outcome or not args.group:
            print("Error: --outcome and --group required for comparison", file=sys.stderr)
            sys.exit(1)
        plot_boxplot_comparison(df, args.outcome, args.group, output_path)
        plot_density_comparison(df, args.outcome, args.group, output_path)

    elif args.type == 'scatter':
        if not args.var1 or not args.var2:
            print("Error: --var1 and --var2 required for scatter", file=sys.stderr)
            sys.exit(1)
        plot_scatter_with_regression(df, args.var1, args.var2, output_path)

    elif args.type == 'correlation':
        if not args.vars:
            print("Error: --vars required for correlation matrix", file=sys.stderr)
            sys.exit(1)
        variables = [v.strip() for v in args.vars.split(',')]
        try:
            import seaborn
        except ImportError:
            print("Error: seaborn required. Install with: pip install seaborn", file=sys.stderr)
            sys.exit(1)
        plot_correlation_matrix(df, variables, output_path)

    elif args.type == 'power':
        plot_power_curve(output_path, alpha=args.alpha)

    elif args.type == 'effect_size':
        if args.effect_size is None:
            args.effect_size = 0.5
        plot_effect_size_interpretation(args.effect_size, output_path)

    print(f"\nAll visualizations saved to: {output_path}")


if __name__ == '__main__':
    main()
