#!/usr/bin/env python3
"""
Distribution Visualization Script for Causal Inference Preprocessing

This script creates visualizations to support preprocessing decisions
in causal inference workflows.

Usage:
    python visualize_distributions.py input.csv --output-dir plots/ \
        --treatment D --outcome Y --controls X1 X2 X3

Features:
    - Distribution plots (histograms, KDE)
    - Treatment/control comparison plots
    - Missing value patterns
    - Before/after transformation comparisons
    - Propensity score distributions
    - Covariate balance plots
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

# Add parent directory to path for preprocessing module
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        return True
    except ImportError:
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize distributions for causal inference preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic distribution plots
    python visualize_distributions.py data.csv --output-dir plots/ \\
        --treatment D --outcome Y

    # Full visualization suite
    python visualize_distributions.py data.csv --output-dir plots/ \\
        --treatment D --outcome Y --controls X1 X2 X3 \\
        --plot-all --format png

    # Specific plots only
    python visualize_distributions.py data.csv --output-dir plots/ \\
        --treatment D --outcome Y \\
        --plot-distributions --plot-balance
        """
    )

    parser.add_argument(
        'input',
        type=str,
        help='Path to input CSV file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save plots'
    )

    parser.add_argument(
        '--treatment',
        type=str,
        required=True,
        help='Name of treatment variable'
    )

    parser.add_argument(
        '--outcome',
        type=str,
        required=True,
        help='Name of outcome variable'
    )

    parser.add_argument(
        '--controls',
        type=str,
        nargs='+',
        default=None,
        help='Names of control variables'
    )

    parser.add_argument(
        '--plot-all',
        action='store_true',
        help='Generate all available plots'
    )

    parser.add_argument(
        '--plot-distributions',
        action='store_true',
        help='Plot variable distributions'
    )

    parser.add_argument(
        '--plot-missing',
        action='store_true',
        help='Plot missing value patterns'
    )

    parser.add_argument(
        '--plot-balance',
        action='store_true',
        help='Plot covariate balance'
    )

    parser.add_argument(
        '--plot-propensity',
        action='store_true',
        help='Plot propensity score distributions'
    )

    parser.add_argument(
        '--plot-transformations',
        action='store_true',
        help='Plot before/after transformations'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format for plots (default: png)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for raster formats (default: 150)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress information'
    )

    return parser.parse_args()


def plot_distributions(
    df: pd.DataFrame,
    columns: List[str],
    treatment: str,
    output_dir: Path,
    fmt: str,
    dpi: int
):
    """Plot distributions for each variable, split by treatment."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    for col in columns:
        if df[col].dtype not in [np.float64, np.int64, float, int]:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Overall distribution
        ax = axes[0]
        df[col].dropna().hist(bins=30, ax=ax, edgecolor='white', alpha=0.7)
        ax.axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
        ax.axvline(df[col].median(), color='blue', linestyle=':', label=f'Median: {df[col].median():.2f}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {col}')
        ax.legend()

        # By treatment group
        ax = axes[1]
        for group, label in [(1, 'Treated'), (0, 'Control')]:
            data = df[df[treatment] == group][col].dropna()
            if len(data) > 0:
                data.hist(bins=30, ax=ax, alpha=0.5, label=label, edgecolor='white')

        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{col} by Treatment Group')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / f'dist_{col}.{fmt}', dpi=dpi, bbox_inches='tight')
        plt.close()


def plot_missing_patterns(
    df: pd.DataFrame,
    columns: List[str],
    output_dir: Path,
    fmt: str,
    dpi: int
):
    """Plot missing value patterns."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Missing value heatmap
    missing_cols = [c for c in columns if df[c].isnull().any()]

    if not missing_cols:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Missing rate by column
    ax = axes[0]
    missing_rates = df[missing_cols].isnull().mean().sort_values(ascending=True)
    missing_rates.plot(kind='barh', ax=ax)
    ax.set_xlabel('Missing Rate')
    ax.set_title('Missing Value Rate by Variable')
    ax.axvline(0.05, color='green', linestyle='--', alpha=0.7, label='5%')
    ax.axvline(0.20, color='orange', linestyle='--', alpha=0.7, label='20%')
    ax.legend()

    # Missing pattern matrix (sample if too large)
    ax = axes[1]
    sample_size = min(500, len(df))
    sample_idx = np.random.choice(df.index, sample_size, replace=False)
    missing_matrix = df.loc[sample_idx, missing_cols].isnull().astype(int)

    sns.heatmap(missing_matrix.T, cmap='YlOrRd', cbar_kws={'label': 'Missing'},
                ax=ax, xticklabels=False)
    ax.set_xlabel('Observations (sample)')
    ax.set_title('Missing Value Pattern')

    plt.tight_layout()
    plt.savefig(output_dir / f'missing_patterns.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_balance(
    df: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    output_dir: Path,
    fmt: str,
    dpi: int
):
    """Plot covariate balance (Love plot)."""
    import matplotlib.pyplot as plt

    # Calculate standardized mean differences
    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]

    smd_data = []
    numeric_covs = []

    for cov in covariates:
        if df[cov].dtype in [np.float64, np.int64, float, int]:
            mean_t = treated[cov].mean()
            mean_c = control[cov].mean()
            var_t = treated[cov].var()
            var_c = control[cov].var()
            pooled_std = np.sqrt((var_t + var_c) / 2)

            if pooled_std > 0:
                smd = (mean_t - mean_c) / pooled_std
                smd_data.append(smd)
                numeric_covs.append(cov)

    if not smd_data:
        return

    # Sort by absolute SMD
    sorted_idx = np.argsort(np.abs(smd_data))
    smd_sorted = [smd_data[i] for i in sorted_idx]
    names_sorted = [numeric_covs[i] for i in sorted_idx]

    # Create Love plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(numeric_covs) * 0.3)))

    colors = ['green' if abs(s) < 0.1 else 'orange' if abs(s) < 0.25 else 'red'
              for s in smd_sorted]

    y_pos = range(len(names_sorted))
    ax.barh(y_pos, smd_sorted, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.axvline(-0.1, color='green', linestyle='--', alpha=0.5)
    ax.axvline(0.1, color='green', linestyle='--', alpha=0.5)
    ax.axvline(-0.25, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(0.25, color='orange', linestyle='--', alpha=0.5)

    ax.set_xlabel('Standardized Mean Difference')
    ax.set_title('Covariate Balance (Love Plot)\nGreen: |SMD|<0.1, Orange: 0.1-0.25, Red: >0.25')

    plt.tight_layout()
    plt.savefig(output_dir / f'balance_love_plot.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_propensity(
    df: pd.DataFrame,
    treatment: str,
    controls: List[str],
    output_dir: Path,
    fmt: str,
    dpi: int
):
    """Plot propensity score distributions."""
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    # Get numeric controls
    numeric_controls = [c for c in controls if df[c].dtype in [np.float64, np.int64, float, int]]

    if not numeric_controls:
        return

    # Handle missing values
    df_clean = df[[treatment] + numeric_controls].dropna()

    if len(df_clean) < 10:
        return

    # Estimate propensity scores
    try:
        ps_model = LogisticRegression(max_iter=1000, solver='lbfgs')
        ps_model.fit(df_clean[numeric_controls], df_clean[treatment])
        ps = ps_model.predict_proba(df_clean[numeric_controls])[:, 1]
    except Exception:
        return

    df_clean = df_clean.copy()
    df_clean['propensity_score'] = ps

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram by group
    ax = axes[0]
    for group, label, color in [(1, 'Treated', 'blue'), (0, 'Control', 'orange')]:
        data = df_clean[df_clean[treatment] == group]['propensity_score']
        ax.hist(data, bins=30, alpha=0.5, label=label, color=color, edgecolor='white')
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Propensity Score Distribution')
    ax.legend()

    # KDE plot
    ax = axes[1]
    for group, label, color in [(1, 'Treated', 'blue'), (0, 'Control', 'orange')]:
        data = df_clean[df_clean[treatment] == group]['propensity_score']
        try:
            from scipy import stats
            kde = stats.gaussian_kde(data)
            x = np.linspace(0, 1, 100)
            ax.plot(x, kde(x), label=label, color=color)
            ax.fill_between(x, kde(x), alpha=0.3, color=color)
        except:
            pass
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.set_title('Propensity Score Density')
    ax.legend()

    # Overlap visualization
    ax = axes[2]
    ps_treated = df_clean[df_clean[treatment] == 1]['propensity_score']
    ps_control = df_clean[df_clean[treatment] == 0]['propensity_score']

    overlap_lower = max(ps_treated.min(), ps_control.min())
    overlap_upper = min(ps_treated.max(), ps_control.max())

    ax.axvline(ps_treated.min(), color='blue', linestyle='--', alpha=0.7, label='Treated range')
    ax.axvline(ps_treated.max(), color='blue', linestyle='--', alpha=0.7)
    ax.axvline(ps_control.min(), color='orange', linestyle='--', alpha=0.7, label='Control range')
    ax.axvline(ps_control.max(), color='orange', linestyle='--', alpha=0.7)
    ax.axvspan(overlap_lower, overlap_upper, alpha=0.3, color='green', label='Overlap region')

    ax.set_xlim(0, 1)
    ax.set_xlabel('Propensity Score')
    ax.set_title(f'Overlap Region: [{overlap_lower:.3f}, {overlap_upper:.3f}]')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'propensity_scores.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_transformations(
    df: pd.DataFrame,
    columns: List[str],
    output_dir: Path,
    fmt: str,
    dpi: int
):
    """Plot before/after transformation comparisons."""
    import matplotlib.pyplot as plt
    from scipy import stats

    for col in columns:
        if df[col].dtype not in [np.float64, np.int64, float, int]:
            continue

        data = df[col].dropna()
        if len(data) < 10 or data.min() <= 0:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Original
        ax = axes[0, 0]
        data.hist(bins=30, ax=ax, edgecolor='white', alpha=0.7)
        ax.set_title(f'{col} - Original')
        ax.set_xlabel(col)

        # Log transform
        ax = axes[0, 1]
        log_data = np.log(data)
        log_data.hist(bins=30, ax=ax, edgecolor='white', alpha=0.7, color='green')
        ax.set_title(f'{col} - Log Transform')
        ax.set_xlabel(f'log({col})')

        # Square root transform
        ax = axes[0, 2]
        sqrt_data = np.sqrt(data)
        sqrt_data.hist(bins=30, ax=ax, edgecolor='white', alpha=0.7, color='orange')
        ax.set_title(f'{col} - Square Root')
        ax.set_xlabel(f'sqrt({col})')

        # Q-Q plots
        for i, (transformed, name) in enumerate([
            (data, 'Original'),
            (log_data, 'Log'),
            (sqrt_data, 'Sqrt')
        ]):
            ax = axes[1, i]
            stats.probplot(transformed, dist="norm", plot=ax)
            ax.set_title(f'{name} - Q-Q Plot')

        plt.suptitle(f'Transformation Comparison: {col}', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'transform_{col}.{fmt}', dpi=dpi, bbox_inches='tight')
        plt.close()


def create_summary_dashboard(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    controls: List[str],
    output_dir: Path,
    fmt: str,
    dpi: int
):
    """Create a summary dashboard with key preprocessing insights."""
    import matplotlib.pyplot as plt
    from preprocessing import diagnose_missing

    fig = plt.figure(figsize=(16, 12))

    # 1. Sample size info
    ax1 = fig.add_subplot(2, 3, 1)
    n_total = len(df)
    n_treated = (df[treatment] == 1).sum()
    n_control = (df[treatment] == 0).sum()

    bars = ax1.bar(['Total', 'Treated', 'Control'], [n_total, n_treated, n_control],
                   color=['gray', 'blue', 'orange'], alpha=0.7)
    ax1.set_title('Sample Sizes')
    ax1.set_ylabel('N')
    for bar, val in zip(bars, [n_total, n_treated, n_control]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val}', ha='center', va='bottom')

    # 2. Missing data summary
    ax2 = fig.add_subplot(2, 3, 2)
    all_vars = [outcome, treatment] + controls
    missing_rates = df[all_vars].isnull().mean().sort_values(ascending=True)
    colors = ['green' if r < 0.05 else 'orange' if r < 0.2 else 'red' for r in missing_rates]
    missing_rates.plot(kind='barh', ax=ax2, color=colors, alpha=0.7)
    ax2.axvline(0.05, color='green', linestyle='--', alpha=0.5)
    ax2.axvline(0.20, color='orange', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Missing Rate')
    ax2.set_title('Missing Values by Variable')

    # 3. Outcome distribution by treatment
    ax3 = fig.add_subplot(2, 3, 3)
    for group, label, color in [(1, 'Treated', 'blue'), (0, 'Control', 'orange')]:
        data = df[df[treatment] == group][outcome].dropna()
        if len(data) > 0:
            data.hist(bins=20, ax=ax3, alpha=0.5, label=label, color=color, edgecolor='white')
    ax3.set_xlabel(outcome)
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Outcome ({outcome}) by Treatment')
    ax3.legend()

    # 4. Top imbalanced covariates
    ax4 = fig.add_subplot(2, 3, 4)
    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]

    smds = []
    for cov in controls:
        if df[cov].dtype in [np.float64, np.int64, float, int]:
            mean_t = treated[cov].mean()
            mean_c = control[cov].mean()
            pooled_std = np.sqrt((treated[cov].var() + control[cov].var()) / 2)
            if pooled_std > 0:
                smds.append((cov, (mean_t - mean_c) / pooled_std))

    if smds:
        smds_sorted = sorted(smds, key=lambda x: abs(x[1]), reverse=True)[:10]
        names = [s[0] for s in smds_sorted]
        values = [s[1] for s in smds_sorted]
        colors = ['green' if abs(v) < 0.1 else 'orange' if abs(v) < 0.25 else 'red' for v in values]

        y_pos = range(len(names))
        ax4.barh(y_pos, values, color=colors, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(names)
        ax4.axvline(0, color='black', linewidth=0.5)
        ax4.axvline(-0.1, color='green', linestyle='--', alpha=0.5)
        ax4.axvline(0.1, color='green', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Standardized Mean Difference')
        ax4.set_title('Top 10 Most Imbalanced Covariates')

    # 5. Data completeness
    ax5 = fig.add_subplot(2, 3, 5)
    complete_cases = df[all_vars].dropna().shape[0]
    incomplete = n_total - complete_cases
    ax5.pie([complete_cases, incomplete],
            labels=['Complete', 'Incomplete'],
            colors=['green', 'red'],
            autopct='%1.1f%%',
            startangle=90)
    ax5.set_title(f'Data Completeness\n(Complete: {complete_cases}, Incomplete: {incomplete})')

    # 6. Summary statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
PREPROCESSING SUMMARY
=====================
Total observations: {n_total}
Treated: {n_treated} ({n_treated/n_total*100:.1f}%)
Control: {n_control} ({n_control/n_total*100:.1f}%)

Complete cases: {complete_cases} ({complete_cases/n_total*100:.1f}%)
Columns with missing: {(df[all_vars].isnull().any()).sum()}

Covariates checked: {len(controls)}
Imbalanced (|SMD|>0.1): {sum(1 for _, v in smds if abs(v) > 0.1)}
"""
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Data Quality Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'dashboard.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    """Main visualization pipeline."""
    args = parse_args()

    # Check matplotlib availability
    if not check_matplotlib():
        print("Error: matplotlib is required for visualization.")
        print("Install with: pip install matplotlib seaborn")
        sys.exit(1)

    # Load data
    if args.verbose:
        print(f"Loading data from {args.input}...")

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Validate columns
    if args.treatment not in df.columns:
        print(f"Error: Treatment '{args.treatment}' not found")
        sys.exit(1)

    if args.outcome not in df.columns:
        print(f"Error: Outcome '{args.outcome}' not found")
        sys.exit(1)

    # Determine controls
    if args.controls is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        controls = [c for c in numeric_cols if c not in [args.outcome, args.treatment]]
    else:
        controls = args.controls

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which plots to create
    if args.plot_all or not any([
        args.plot_distributions,
        args.plot_missing,
        args.plot_balance,
        args.plot_propensity,
        args.plot_transformations
    ]):
        # Default to all plots
        args.plot_distributions = True
        args.plot_missing = True
        args.plot_balance = True
        args.plot_propensity = True
        args.plot_transformations = True

    # Create plots
    if args.plot_distributions:
        if args.verbose:
            print("Creating distribution plots...")
        plot_distributions(df, controls + [args.outcome], args.treatment,
                          output_dir, args.format, args.dpi)

    if args.plot_missing:
        if args.verbose:
            print("Creating missing value plots...")
        plot_missing_patterns(df, controls + [args.outcome], output_dir, args.format, args.dpi)

    if args.plot_balance:
        if args.verbose:
            print("Creating balance plots...")
        plot_balance(df, args.treatment, controls, output_dir, args.format, args.dpi)

    if args.plot_propensity:
        if args.verbose:
            print("Creating propensity score plots...")
        plot_propensity(df, args.treatment, controls, output_dir, args.format, args.dpi)

    if args.plot_transformations:
        if args.verbose:
            print("Creating transformation comparison plots...")
        plot_transformations(df, controls, output_dir, args.format, args.dpi)

    # Always create dashboard
    if args.verbose:
        print("Creating summary dashboard...")
    create_summary_dashboard(df, args.treatment, args.outcome, controls,
                            output_dir, args.format, args.dpi)

    print(f"\nPlots saved to {output_dir}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
