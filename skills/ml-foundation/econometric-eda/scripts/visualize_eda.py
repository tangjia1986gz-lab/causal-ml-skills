#!/usr/bin/env python
"""
EDA Visualization Suite CLI

Generate comprehensive EDA visualizations for econometric research.

Usage:
    python visualize_eda.py --data data.csv --output figures/
    python visualize_eda.py --data data.csv --output figures/ --type all
    python visualize_eda.py --data data.csv --vars y,x1,x2 --output figures/ --type distributions
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate EDA visualizations for econometric research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # All visualizations
    python visualize_eda.py --data data.csv --output figures/

    # Specific visualization types
    python visualize_eda.py --data data.csv --output figures/ --type distributions
    python visualize_eda.py --data data.csv --output figures/ --type correlations
    python visualize_eda.py --data data.csv --output figures/ --type pairplot

    # With treatment indicator for group comparisons
    python visualize_eda.py --data data.csv --output figures/ --treatment D

    # Panel data visualizations
    python visualize_eda.py --data panel.csv --output figures/ --panel-id entity --time-var year

    # High-resolution output
    python visualize_eda.py --data data.csv --output figures/ --dpi 300
        """
    )

    parser.add_argument('--data', '-d', required=True,
                        help='Path to CSV data file')
    parser.add_argument('--vars', '-v',
                        help='Comma-separated list of variables (default: all numeric)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for figures')
    parser.add_argument('--type', '-t', default='all',
                        choices=['all', 'distributions', 'correlations', 'pairplot',
                                'missing', 'outliers', 'balance', 'panel'],
                        help='Type of visualization')
    parser.add_argument('--treatment',
                        help='Treatment variable for group comparisons')
    parser.add_argument('--panel-id',
                        help='Panel entity identifier')
    parser.add_argument('--time-var',
                        help='Panel time variable')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Figure DPI (default: 150)')
    parser.add_argument('--format', '-f', default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Output format')

    return parser.parse_args()


def check_plotting():
    """Check if plotting libraries are available."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        return True
    except ImportError:
        print("Error: matplotlib and seaborn are required for visualizations")
        print("Install with: pip install matplotlib seaborn")
        return False


def plot_distributions(data: pd.DataFrame, variables: list, output_dir: Path,
                       treatment: str = None, dpi: int = 150, fmt: str = 'png'):
    """Plot variable distributions."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_vars = len(variables)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    # Histograms
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, var in enumerate(variables):
        ax = axes[i]
        if treatment and treatment in data.columns:
            for t_val in data[treatment].unique():
                subset = data[data[treatment] == t_val][var].dropna()
                ax.hist(subset, bins=30, alpha=0.5, label=f'{treatment}={t_val}', density=True)
            ax.legend()
        else:
            ax.hist(data[var].dropna(), bins=30, edgecolor='black', alpha=0.7)

        ax.set_xlabel(var)
        ax.set_ylabel('Density' if treatment else 'Frequency')
        ax.set_title(f'Distribution of {var}')

        # Add statistics
        mean_val = data[var].mean()
        median_val = data[var].median()
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle=':', alpha=0.7, label=f'Median: {median_val:.2f}')
        ax.legend(fontsize=8)

    for i in range(n_vars, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_dir / f'distributions.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()

    # Box plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, var in enumerate(variables):
        ax = axes[i]
        if treatment and treatment in data.columns:
            data.boxplot(column=var, by=treatment, ax=ax)
            ax.set_title(f'{var} by {treatment}')
        else:
            ax.boxplot(data[var].dropna())
            ax.set_title(f'{var}')
        ax.set_ylabel(var)

    for i in range(n_vars, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_dir / f'boxplots.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  - Distributions: distributions.{fmt}, boxplots.{fmt}")


def plot_correlations(data: pd.DataFrame, variables: list, output_dir: Path,
                      dpi: int = 150, fmt: str = 'png'):
    """Plot correlation matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Correlation matrix
    corr = data[variables].corr()

    fig, ax = plt.subplots(figsize=(max(10, len(variables)*0.8), max(8, len(variables)*0.6)))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 8})

    ax.set_title('Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'correlation_matrix.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  - Correlations: correlation_matrix.{fmt}")


def plot_pairplot(data: pd.DataFrame, variables: list, output_dir: Path,
                  treatment: str = None, dpi: int = 150, fmt: str = 'png'):
    """Create scatterplot matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Limit to 6 variables for readability
    if len(variables) > 6:
        print(f"  Warning: Limiting pairplot to first 6 variables")
        variables = variables[:6]

    subset = data[variables + ([treatment] if treatment else [])].dropna()

    g = sns.pairplot(subset, hue=treatment, diag_kind='kde',
                     plot_kws={'alpha': 0.5, 's': 20}, corner=True)

    g.fig.suptitle('Scatterplot Matrix', y=1.02)

    g.savefig(output_dir / f'pairplot.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  - Pairplot: pairplot.{fmt}")


def plot_missing(data: pd.DataFrame, variables: list, output_dir: Path,
                 dpi: int = 150, fmt: str = 'png'):
    """Plot missing data patterns."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Missing percentages
    missing_pct = (data[variables].isna().sum() / len(data)) * 100
    missing_pct = missing_pct.sort_values(ascending=True)

    missing_pct.plot(kind='barh', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_xlabel('Percentage Missing')
    axes[0, 0].set_title('Missing Values by Variable')
    axes[0, 0].axvline(x=5, color='orange', linestyle='--', label='5% threshold')
    axes[0, 0].legend()

    # Missing matrix
    missing_matrix = data[variables].isna().astype(int)
    if len(missing_matrix) > 200:
        sample_idx = np.random.choice(len(missing_matrix), 200, replace=False)
        sample_idx = np.sort(sample_idx)
        missing_matrix = missing_matrix.iloc[sample_idx]

    sns.heatmap(missing_matrix.T, cmap='RdYlBu_r', ax=axes[0, 1],
                cbar_kws={'label': 'Missing'}, yticklabels=True)
    axes[0, 1].set_xlabel('Observation')
    axes[0, 1].set_title('Missing Value Matrix')

    # Missing correlations
    missing_corr = data[variables].isna().corr()
    mask = np.triu(np.ones_like(missing_corr, dtype=bool), k=1)
    sns.heatmap(missing_corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, ax=axes[1, 0], vmin=-1, vmax=1)
    axes[1, 0].set_title('Missing Value Correlations')

    # Completeness distribution
    completeness = 1 - data[variables].isna().mean(axis=1)
    axes[1, 1].hist(completeness, bins=20, edgecolor='black', color='steelblue')
    axes[1, 1].axvline(x=completeness.mean(), color='red', linestyle='--',
                       label=f'Mean: {completeness.mean():.2f}')
    axes[1, 1].set_xlabel('Completeness Rate')
    axes[1, 1].set_ylabel('Number of Observations')
    axes[1, 1].set_title('Observation Completeness')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'missing_patterns.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  - Missing: missing_patterns.{fmt}")


def plot_outliers(data: pd.DataFrame, variables: list, output_dir: Path,
                  dpi: int = 150, fmt: str = 'png'):
    """Plot outlier diagnostics."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_vars = len(variables)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, var in enumerate(variables):
        ax = axes[i]
        col = data[var].dropna()

        # IQR bounds
        q1, q3 = col.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Box plot
        bp = ax.boxplot(col, vert=True)

        # Count outliers
        outliers = col[(col < lower) | (col > upper)]
        n_outliers = len(outliers)

        ax.set_title(f'{var}\n({n_outliers} outliers, {n_outliers/len(col)*100:.1f}%)')
        ax.set_ylabel(var)

    for i in range(n_vars, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_dir / f'outlier_boxplots.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  - Outliers: outlier_boxplots.{fmt}")


def plot_balance(data: pd.DataFrame, variables: list, treatment: str,
                 output_dir: Path, dpi: int = 150, fmt: str = 'png'):
    """Plot covariate balance."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    # Calculate standardized differences
    std_diffs = []
    for var in variables:
        treated = data.loc[data[treatment] == 1, var]
        control = data.loc[data[treatment] == 0, var]

        mean_t = treated.mean()
        mean_c = control.mean()
        pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
        std_diff = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0

        std_diffs.append({
            'variable': var,
            'std_diff': std_diff
        })

    balance_df = pd.DataFrame(std_diffs).sort_values('std_diff')

    # Love plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(variables)*0.4)))

    colors = ['red' if abs(d) > 0.1 else 'green' for d in balance_df['std_diff']]
    ax.barh(balance_df['variable'], balance_df['std_diff'], color=colors, alpha=0.7)

    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='|d| = 0.1')
    ax.axvline(x=-0.1, color='red', linestyle='--', alpha=0.5)

    ax.set_xlabel('Standardized Difference')
    ax.set_title('Covariate Balance (Love Plot)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'balance_plot.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()

    # Distribution comparisons
    n_vars = min(6, len(variables))
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, var in enumerate(variables[:n_vars]):
        ax = axes[i]

        for t_val, label, color in [(1, 'Treated', 'blue'), (0, 'Control', 'orange')]:
            subset = data[data[treatment] == t_val][var].dropna()
            ax.hist(subset, bins=30, alpha=0.5, density=True, label=label, color=color)

        ax.set_xlabel(var)
        ax.set_ylabel('Density')
        ax.set_title(f'{var}')
        ax.legend()

    for i in range(n_vars, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_dir / f'balance_distributions.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  - Balance: balance_plot.{fmt}, balance_distributions.{fmt}")


def plot_panel(data: pd.DataFrame, variables: list, panel_id: str, time_var: str,
               output_dir: Path, dpi: int = 150, fmt: str = 'png'):
    """Plot panel data visualizations."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Panel balance
    obs_per_entity = data.groupby(panel_id).size()
    obs_per_period = data.groupby(time_var).size()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Observations per entity
    axes[0, 0].hist(obs_per_entity, bins=30, edgecolor='black')
    axes[0, 0].axvline(obs_per_entity.mean(), color='red', linestyle='--',
                       label=f'Mean: {obs_per_entity.mean():.1f}')
    axes[0, 0].set_xlabel('Number of Observations')
    axes[0, 0].set_ylabel('Number of Entities')
    axes[0, 0].set_title('Observations per Entity')
    axes[0, 0].legend()

    # Observations per period
    axes[0, 1].bar(obs_per_period.index.astype(str), obs_per_period.values)
    axes[0, 1].set_xlabel('Time Period')
    axes[0, 1].set_ylabel('Number of Entities')
    axes[0, 1].set_title('Entities per Time Period')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Sample trajectories for first variable
    if len(variables) > 0:
        var = variables[0]
        entities = data[panel_id].unique()

        # Select random sample of entities
        np.random.seed(42)
        sample_entities = np.random.choice(entities, min(10, len(entities)), replace=False)

        for entity in sample_entities:
            entity_data = data[data[panel_id] == entity].sort_values(time_var)
            axes[1, 0].plot(entity_data[time_var], entity_data[var], 'o-', alpha=0.6, markersize=4)

        # Overall mean
        time_means = data.groupby(time_var)[var].mean()
        axes[1, 0].plot(time_means.index, time_means.values, 'k-', linewidth=3, label='Overall mean')

        axes[1, 0].set_xlabel('Time Period')
        axes[1, 0].set_ylabel(var)
        axes[1, 0].set_title(f'Individual Trajectories: {var}')
        axes[1, 0].legend()

        # Within/between variation
        entity_means = data.groupby(panel_id)[var].mean()
        axes[1, 1].hist(entity_means, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(entity_means.mean(), color='red', linestyle='--',
                          label=f'Grand mean: {entity_means.mean():.2f}')
        axes[1, 1].set_xlabel(f'Entity Mean of {var}')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Between-Entity Variation')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'panel_structure.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  - Panel: panel_structure.{fmt}")


def main():
    args = parse_args()

    # Check plotting availability
    if not check_plotting():
        sys.exit(1)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'

    # Load data
    print(f"Loading data from {args.data}...")
    try:
        data = pd.read_csv(args.data)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print(f"Loaded {len(data)} observations, {len(data.columns)} variables")

    # Parse variables
    if args.vars:
        variables = [v.strip() for v in args.vars.split(',')]
    else:
        variables = [c for c in data.columns if data[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations...")
    print(f"Output directory: {output_dir}")
    print(f"Format: {args.format}, DPI: {args.dpi}")

    # Generate requested visualizations
    viz_types = ['distributions', 'correlations', 'pairplot', 'missing', 'outliers']
    if args.treatment:
        viz_types.append('balance')
    if args.panel_id and args.time_var:
        viz_types.append('panel')

    if args.type != 'all':
        viz_types = [args.type]

    for viz_type in viz_types:
        try:
            if viz_type == 'distributions':
                plot_distributions(data, variables, output_dir, args.treatment, args.dpi, args.format)
            elif viz_type == 'correlations':
                plot_correlations(data, variables, output_dir, args.dpi, args.format)
            elif viz_type == 'pairplot':
                plot_pairplot(data, variables, output_dir, args.treatment, args.dpi, args.format)
            elif viz_type == 'missing':
                plot_missing(data, variables, output_dir, args.dpi, args.format)
            elif viz_type == 'outliers':
                plot_outliers(data, variables, output_dir, args.dpi, args.format)
            elif viz_type == 'balance' and args.treatment:
                plot_balance(data, variables, args.treatment, output_dir, args.dpi, args.format)
            elif viz_type == 'panel' and args.panel_id and args.time_var:
                plot_panel(data, variables, args.panel_id, args.time_var, output_dir, args.dpi, args.format)
        except Exception as e:
            print(f"  Warning: Could not generate {viz_type}: {e}")

    print(f"\nVisualization complete!")


if __name__ == '__main__':
    main()
