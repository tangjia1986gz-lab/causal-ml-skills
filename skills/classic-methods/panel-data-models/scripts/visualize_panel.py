#!/usr/bin/env python
"""
Panel Data Visualization

Create visualizations for panel data analysis:
- Entity trends over time
- Treatment timing plots
- Event study plots
- Within vs between variation
- Residual diagnostics

Usage:
    python visualize_panel.py data.csv --entity firm_id --time year --y revenue --plot trends
    python visualize_panel.py data.csv --entity firm_id --time year --y revenue --treatment treated --plot event-study

Author: Causal ML Skills
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Panel data visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Entity trends
  python visualize_panel.py data.csv --entity firm_id --time year --y revenue --plot trends

  # Treatment timing
  python visualize_panel.py data.csv --entity firm_id --time year --treatment treated --plot timing

  # Event study
  python visualize_panel.py data.csv --entity firm_id --time year --y revenue --treatment treated --plot event-study

  # All plots
  python visualize_panel.py data.csv --entity firm_id --time year --y revenue --treatment treated --plot all
        """
    )

    parser.add_argument('data_file', type=str, help='Path to CSV data file')
    parser.add_argument('--entity', required=True, help='Entity identifier column')
    parser.add_argument('--time', required=True, help='Time period column')
    parser.add_argument('--y', required=True, help='Outcome variable column')
    parser.add_argument('--treatment', type=str, help='Treatment indicator column')
    parser.add_argument('--x', nargs='+', help='Additional covariates to visualize')

    parser.add_argument(
        '--plot',
        choices=['trends', 'timing', 'event-study', 'variation', 'diagnostics', 'all'],
        default='trends',
        help='Type of plot to create'
    )

    parser.add_argument('--n-entities', type=int, default=20, help='Number of entities to show in trends')
    parser.add_argument('--output', type=str, help='Output file (PNG or PDF)')
    parser.add_argument('--dpi', type=int, default=150, help='Figure DPI')
    parser.add_argument('--style', default='seaborn-v0_8-whitegrid', help='Matplotlib style')

    return parser.parse_args()


def plot_entity_trends(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_col: str,
    treatment_col: Optional[str] = None,
    n_entities: int = 20,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot outcome trends for sample of entities."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Sample entities
    entities = data[entity_col].unique()
    if len(entities) > n_entities:
        np.random.seed(42)
        sampled = np.random.choice(entities, n_entities, replace=False)
    else:
        sampled = entities

    # Color by treatment status if provided
    if treatment_col:
        # Determine if entity is ever treated
        ever_treated = data.groupby(entity_col)[treatment_col].max() > 0
        colors = {e: 'C1' if ever_treated.get(e, False) else 'C0' for e in sampled}
    else:
        colors = {e: 'C0' for e in sampled}

    # Plot each entity
    for entity in sampled:
        entity_data = data[data[entity_col] == entity].sort_values(time_col)
        ax.plot(
            entity_data[time_col],
            entity_data[y_col],
            color=colors[entity],
            alpha=0.5,
            linewidth=0.8
        )

    # Add overall mean
    mean_by_time = data.groupby(time_col)[y_col].mean()
    ax.plot(mean_by_time.index, mean_by_time.values, 'k-', linewidth=2, label='Overall Mean')

    ax.set_xlabel(time_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'Entity Trends Over Time (n={len(sampled)} entities shown)')

    if treatment_col:
        treated_patch = mpatches.Patch(color='C1', alpha=0.5, label='Ever Treated')
        control_patch = mpatches.Patch(color='C0', alpha=0.5, label='Never Treated')
        ax.legend(handles=[treated_patch, control_patch, ax.lines[-1]], loc='best')
    else:
        ax.legend(loc='best')

    return ax


def plot_treatment_timing(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    treatment_col: str,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Visualize treatment timing across entities."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Pivot to entity x time
    treatment_matrix = data.pivot_table(
        index=entity_col,
        columns=time_col,
        values=treatment_col,
        aggfunc='max'
    ).fillna(0)

    # Sort by first treatment period
    first_treated = treatment_matrix.idxmax(axis=1).replace({treatment_matrix.columns[0]: np.inf})
    first_treated[treatment_matrix.sum(axis=1) == 0] = np.inf
    treatment_matrix = treatment_matrix.loc[first_treated.sort_values().index]

    # Create heatmap
    cmap = LinearSegmentedColormap.from_list('treatment', ['white', 'steelblue'])
    im = ax.imshow(treatment_matrix.values, aspect='auto', cmap=cmap)

    # Labels
    ax.set_xlabel(time_col)
    ax.set_ylabel(f'{entity_col} (sorted by treatment timing)')
    ax.set_title('Treatment Timing Across Entities')

    # X-axis labels
    ax.set_xticks(range(len(treatment_matrix.columns)))
    ax.set_xticklabels(treatment_matrix.columns, rotation=45)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Treatment Status')

    # Add timing group annotations
    timing_groups = first_treated[first_treated != np.inf].value_counts().sort_index()
    if len(timing_groups) > 0:
        text = "Treatment Cohorts:\n"
        for t, count in timing_groups.items():
            text += f"  t={t}: n={count}\n"
        never_treated = (first_treated == np.inf).sum()
        if never_treated > 0:
            text += f"  Never: n={never_treated}"
        ax.text(1.15, 0.5, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='center')

    return ax


def plot_event_study(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_col: str,
    treatment_col: str,
    window: int = 5,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot event study (treatment effects by relative time).

    Simple implementation: Compare treated vs control mean difference.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Identify treatment timing for each entity
    first_treated = data.groupby(entity_col).apply(
        lambda g: g.loc[g[treatment_col] == 1, time_col].min()
        if (g[treatment_col] == 1).any() else np.inf
    )

    # Create relative time variable
    data = data.copy()
    data['first_treated'] = data[entity_col].map(first_treated)
    data['rel_time'] = data[time_col] - data['first_treated']

    # Get treated entities only
    treated_data = data[data['first_treated'] != np.inf]

    # Compute mean by relative time
    means = treated_data.groupby('rel_time')[y_col].agg(['mean', 'std', 'count'])
    means['se'] = means['std'] / np.sqrt(means['count'])

    # Filter to window
    means = means[(means.index >= -window) & (means.index <= window)]

    # Plot
    ax.errorbar(
        means.index,
        means['mean'],
        yerr=1.96 * means['se'],
        fmt='o-',
        capsize=3,
        color='C0',
        label='Mean (95% CI)'
    )

    # Reference lines
    ax.axhline(y=means.loc[means.index < 0, 'mean'].mean(), color='gray',
               linestyle='--', alpha=0.5, label='Pre-treatment mean')
    ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.7, label='Treatment')

    ax.set_xlabel('Periods Relative to Treatment')
    ax.set_ylabel(y_col)
    ax.set_title('Event Study: Treatment Effects Over Time')
    ax.legend(loc='best')

    # Add note
    ax.text(0.02, 0.02, f'N treated entities: {len(first_treated[first_treated != np.inf])}',
            transform=ax.transAxes, fontsize=9)

    return ax


def plot_variation_decomposition(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_col: str,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Visualize within vs between variation."""
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig = ax.figure
        axes = [ax]

    # Compute components
    overall_mean = data[y_col].mean()
    entity_means = data.groupby(entity_col)[y_col].transform('mean')
    time_means = data.groupby(time_col)[y_col].transform('mean')

    # Within variation (deviations from entity mean)
    within = data[y_col] - entity_means

    # Between variation (entity means - overall mean)
    between = entity_means - overall_mean

    # Plot 1: Overall distribution
    axes[0].hist(data[y_col], bins=50, alpha=0.7, color='C0')
    axes[0].axvline(overall_mean, color='red', linestyle='--', label=f'Mean: {overall_mean:.2f}')
    axes[0].set_xlabel(y_col)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Overall Distribution\nVar={data[y_col].var():.2f}')
    axes[0].legend()

    # Plot 2: Between variation (entity means)
    entity_mean_values = data.groupby(entity_col)[y_col].mean()
    axes[1].hist(entity_mean_values, bins=30, alpha=0.7, color='C1')
    axes[1].axvline(overall_mean, color='red', linestyle='--')
    axes[1].set_xlabel(f'Entity Mean of {y_col}')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Between Variation\nVar={between.var():.2f}')

    # Plot 3: Within variation (deviations from entity mean)
    axes[2].hist(within, bins=50, alpha=0.7, color='C2')
    axes[2].axvline(0, color='red', linestyle='--')
    axes[2].set_xlabel(f'Deviation from Entity Mean')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Within Variation\nVar={within.var():.2f}')

    # Add variance decomposition summary
    total_var = data[y_col].var()
    between_var = between.var()
    within_var = within.var()

    fig.suptitle(
        f'Variance Decomposition: Total={total_var:.2f}, '
        f'Between={between_var:.2f} ({100*between_var/total_var:.1f}%), '
        f'Within={within_var:.2f} ({100*within_var/total_var:.1f}%)',
        y=1.02
    )

    plt.tight_layout()
    return fig


def plot_residual_diagnostics(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_col: str,
    x_cols: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Diagnostic plots for panel model residuals."""
    from panel_estimator import PanelEstimator

    if x_cols is None:
        # Use entity-demeaned y as "residuals"
        entity_mean = data.groupby(entity_col)[y_col].transform('mean')
        residuals = (data[y_col] - entity_mean).values
        fitted = entity_mean.values
    else:
        estimator = PanelEstimator(data, entity_col, time_col, y_col, x_cols)
        result = estimator.fit_fixed_effects()
        residuals = result.residuals
        fitted = result.fitted_values

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Residuals vs Fitted
    axes[0, 0].scatter(fitted, residuals, alpha=0.3, s=10)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')

    # Add LOWESS smooth
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, fitted, frac=0.3)
        axes[0, 0].plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2)
    except:
        pass

    # Plot 2: QQ plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')

    # Plot 3: Residuals over time
    time_resid = pd.DataFrame({
        'time': data[time_col],
        'residual': residuals
    }).groupby('time')['residual'].agg(['mean', 'std'])

    axes[1, 0].errorbar(
        time_resid.index,
        time_resid['mean'],
        yerr=time_resid['std'],
        fmt='o-',
        capsize=3
    )
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel(time_col)
    axes[1, 0].set_ylabel('Mean Residual')
    axes[1, 0].set_title('Residuals by Time Period')

    # Plot 4: Residual autocorrelation
    # Compute within-entity lag-1 correlation
    df_temp = data.copy()
    df_temp['residual'] = residuals
    df_temp['residual_lag'] = df_temp.groupby(entity_col)['residual'].shift(1)
    valid = df_temp['residual_lag'].notna()

    axes[1, 1].scatter(
        df_temp.loc[valid, 'residual_lag'],
        df_temp.loc[valid, 'residual'],
        alpha=0.3,
        s=10
    )
    axes[1, 1].set_xlabel('Residual (t-1)')
    axes[1, 1].set_ylabel('Residual (t)')

    # Add correlation
    corr = df_temp.loc[valid, ['residual', 'residual_lag']].corr().iloc[0, 1]
    axes[1, 1].set_title(f'Serial Correlation (rho={corr:.3f})')

    # Add diagonal line
    lims = [
        min(axes[1, 1].get_xlim()[0], axes[1, 1].get_ylim()[0]),
        max(axes[1, 1].get_xlim()[1], axes[1, 1].get_ylim()[1])
    ]
    axes[1, 1].plot(lims, lims, 'r--', alpha=0.5)

    plt.tight_layout()
    return fig


def main():
    args = parse_args()

    # Set style
    try:
        plt.style.use(args.style)
    except:
        pass

    # Load data
    print(f"Loading data from {args.data_file}...")
    data = pd.read_csv(args.data_file)

    # Validate columns
    required_cols = [args.entity, args.time, args.y]
    if args.treatment:
        required_cols.append(args.treatment)
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        sys.exit(1)

    print(f"Panel: {data[args.entity].nunique()} entities, {data[args.time].nunique()} time periods")

    # Create plots
    plots_to_make = ['trends', 'timing', 'event-study', 'variation', 'diagnostics'] if args.plot == 'all' else [args.plot]

    figures = []

    for plot_type in plots_to_make:
        print(f"Creating {plot_type} plot...")

        if plot_type == 'trends':
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_entity_trends(
                data, args.entity, args.time, args.y,
                treatment_col=args.treatment,
                n_entities=args.n_entities,
                ax=ax
            )
            figures.append(('trends', fig))

        elif plot_type == 'timing':
            if args.treatment is None:
                print("  Skipping: requires --treatment")
                continue
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_treatment_timing(
                data, args.entity, args.time, args.treatment, ax=ax
            )
            figures.append(('timing', fig))

        elif plot_type == 'event-study':
            if args.treatment is None:
                print("  Skipping: requires --treatment")
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_event_study(
                data, args.entity, args.time, args.y, args.treatment, ax=ax
            )
            figures.append(('event_study', fig))

        elif plot_type == 'variation':
            fig = plot_variation_decomposition(
                data, args.entity, args.time, args.y
            )
            figures.append(('variation', fig))

        elif plot_type == 'diagnostics':
            fig = plot_residual_diagnostics(
                data, args.entity, args.time, args.y, x_cols=args.x
            )
            figures.append(('diagnostics', fig))

    # Save or show
    if args.output:
        if len(figures) == 1:
            figures[0][1].savefig(args.output, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved to {args.output}")
        else:
            # Save multiple files
            base = Path(args.output)
            for name, fig in figures:
                output_path = base.parent / f"{base.stem}_{name}{base.suffix}"
                fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
                print(f"Saved to {output_path}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
