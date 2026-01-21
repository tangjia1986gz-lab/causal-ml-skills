#!/usr/bin/env python3
"""
Heterogeneity Visualization CLI

Generate comprehensive visualizations for treatment effect heterogeneity
from causal forest analysis.

Usage:
    python visualize_heterogeneity.py --cate-file cate_results.csv --data data.csv \\
        --effect-modifiers X1 X2 X3 --output plots/

Author: Causal ML Skills
"""

import argparse
import sys
import os
import pickle
import warnings
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

# Check for plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate heterogeneity visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From CATE estimates file
    python visualize_heterogeneity.py --cate-file cate_results.csv \\
        --output plots/

    # With data for subgroup analysis
    python visualize_heterogeneity.py --cate-file cate_results.csv \\
        --data data.csv --effect-modifiers age income tenure \\
        --group-by segment --output plots/

    # From fitted model
    python visualize_heterogeneity.py --model model.pkl --data data.csv \\
        --effect-modifiers X1 X2 X3 --output plots/
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--cate-file', help='Path to CATE estimates CSV')
    input_group.add_argument('--model', help='Path to fitted model pickle')

    parser.add_argument('--data', help='Path to data file (for subgroup analysis)')
    parser.add_argument('--effect-modifiers', nargs='+',
                        help='Effect modifier variable names')

    # Visualization options
    parser.add_argument('--plots', nargs='+',
                        default=['distribution', 'sorted', 'importance', 'partial'],
                        choices=['distribution', 'sorted', 'importance', 'partial',
                                'subgroup', 'gates', 'heatmap', 'all'],
                        help='Types of plots to generate')
    parser.add_argument('--group-by', help='Variable for subgroup comparison')
    parser.add_argument('--top-n-vars', type=int, default=5,
                        help='Number of top variables for partial dependence')

    # Output options
    parser.add_argument('--output', '-o', default='./heterogeneity_plots',
                        help='Output directory')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                        help='Output format (default: png)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Figure resolution (default: 300)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[10, 6],
                        help='Figure size (width height)')
    parser.add_argument('--style', choices=['default', 'seaborn', 'ggplot', 'minimal'],
                        default='seaborn',
                        help='Plot style (default: seaborn)')

    parser.add_argument('--verbose', '-v', action='store_true')

    return parser.parse_args()


def setup_style(style: str):
    """Setup matplotlib style."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    if style == 'seaborn' and SEABORN_AVAILABLE:
        sns.set_style('whitegrid')
        sns.set_palette('husl')
    elif style == 'ggplot':
        plt.style.use('ggplot')
    elif style == 'minimal':
        plt.style.use('seaborn-whitegrid')


def load_cate_data(cate_file: str) -> pd.DataFrame:
    """Load CATE estimates from file."""
    if not os.path.exists(cate_file):
        raise FileNotFoundError(f"CATE file not found: {cate_file}")

    df = pd.read_csv(cate_file)

    # Validate required columns
    required = ['cate']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def load_data(data_path: str) -> pd.DataFrame:
    """Load data file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    else:
        return pd.read_csv(data_path)


def plot_cate_distribution(cate_df: pd.DataFrame, figsize: tuple,
                           output_dir: str, fmt: str, dpi: int, verbose: bool):
    """Create CATE distribution plot."""
    tau = cate_df['cate'].values
    ci_lower = cate_df.get('ci_lower', pd.Series([None])).values
    ci_upper = cate_df.get('ci_upper', pd.Series([None])).values

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))

    # Histogram with KDE
    ax1 = axes[0]
    ax1.hist(tau, bins=50, density=True, alpha=0.6, color='steelblue',
             edgecolor='white', label='Distribution')

    # Add KDE
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(tau)
        x_kde = np.linspace(tau.min(), tau.max(), 200)
        ax1.plot(x_kde, kde(x_kde), 'r-', linewidth=2, label='KDE')
    except:
        pass

    ax1.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero')
    ax1.axvline(np.mean(tau), color='green', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(tau):.3f}')

    ax1.set_xlabel('CATE', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('CATE Distribution', fontsize=14)
    ax1.legend()

    # Violin plot
    ax2 = axes[1]
    parts = ax2.violinplot([tau], positions=[1], showmeans=True,
                            showmedians=True, widths=0.8)
    parts['bodies'][0].set_facecolor('steelblue')
    parts['bodies'][0].set_alpha(0.6)

    # Add jittered points (subsample if large)
    n_points = min(500, len(tau))
    idx = np.random.choice(len(tau), n_points, replace=False)
    jitter = np.random.uniform(-0.15, 0.15, n_points)
    ax2.scatter(1 + jitter, tau[idx], alpha=0.2, s=10, color='gray')

    ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xticks([1])
    ax2.set_xticklabels(['CATE'])
    ax2.set_ylabel('Treatment Effect', fontsize=12)
    ax2.set_title('CATE Violin Plot', fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'cate_distribution.{fmt}')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: {save_path}")


def plot_sorted_cate(cate_df: pd.DataFrame, figsize: tuple,
                     output_dir: str, fmt: str, dpi: int, verbose: bool):
    """Create sorted CATE plot with confidence intervals."""
    tau = cate_df['cate'].values
    ci_lower = cate_df.get('ci_lower')
    ci_upper = cate_df.get('ci_upper')

    fig, ax = plt.subplots(figsize=figsize)

    sorted_idx = np.argsort(tau)
    n = len(tau)
    x = np.arange(n)

    # Subsample for clarity
    if n > 1000:
        step = n // 500
        x_plot = x[::step]
        tau_plot = tau[sorted_idx][::step]
        if ci_lower is not None and ci_upper is not None:
            ci_l_plot = ci_lower.values[sorted_idx][::step]
            ci_u_plot = ci_upper.values[sorted_idx][::step]
        else:
            ci_l_plot = ci_u_plot = None
    else:
        x_plot = x
        tau_plot = tau[sorted_idx]
        if ci_lower is not None and ci_upper is not None:
            ci_l_plot = ci_lower.values[sorted_idx]
            ci_u_plot = ci_upper.values[sorted_idx]
        else:
            ci_l_plot = ci_u_plot = None

    # Plot
    if ci_l_plot is not None and ci_u_plot is not None:
        ax.fill_between(x_plot, ci_l_plot, ci_u_plot, alpha=0.3, color='steelblue')

    ax.plot(x_plot, tau_plot, color='steelblue', linewidth=1)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.axhline(np.mean(tau), color='green', linestyle='-', linewidth=1, alpha=0.7)

    ax.set_xlabel('Observation (sorted by CATE)', fontsize=12)
    ax.set_ylabel('Treatment Effect', fontsize=12)
    ax.set_title('Sorted CATEs with 95% Confidence Intervals', fontsize=14)

    # Add annotations
    n_sig_pos = np.sum(ci_lower.values > 0) if ci_lower is not None else 0
    n_sig_neg = np.sum(ci_upper.values < 0) if ci_upper is not None else 0
    ax.text(0.02, 0.98, f'Sig. positive: {n_sig_pos} ({n_sig_pos/n:.1%})',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.02, 0.93, f'Sig. negative: {n_sig_neg} ({n_sig_neg/n:.1%})',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'cate_sorted.{fmt}')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: {save_path}")


def plot_subgroup_comparison(cate_df: pd.DataFrame, data: pd.DataFrame,
                             group_var: str, figsize: tuple,
                             output_dir: str, fmt: str, dpi: int, verbose: bool):
    """Create subgroup comparison plot."""
    tau = cate_df['cate'].values
    groups = data[group_var].values
    unique_groups = np.unique(groups)

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))

    # Box plot
    ax1 = axes[0]
    data_by_group = [tau[groups == g] for g in unique_groups]
    bp = ax1.boxplot(data_by_group, labels=unique_groups, patch_artist=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_groups)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.axhline(np.mean(tau), color='green', linestyle='-', linewidth=1,
                label=f'Overall mean: {np.mean(tau):.3f}')
    ax1.set_ylabel('CATE', fontsize=12)
    ax1.set_xlabel(group_var, fontsize=12)
    ax1.set_title(f'CATE by {group_var}', fontsize=14)
    ax1.legend()

    # Mean comparison
    ax2 = axes[1]
    group_means = [np.mean(tau[groups == g]) for g in unique_groups]
    group_se = [np.std(tau[groups == g]) / np.sqrt(np.sum(groups == g))
                for g in unique_groups]

    x_pos = np.arange(len(unique_groups))
    ax2.bar(x_pos, group_means, yerr=[1.96*se for se in group_se],
            capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(unique_groups)
    ax2.set_ylabel('Mean CATE', fontsize=12)
    ax2.set_xlabel(group_var, fontsize=12)
    ax2.set_title(f'Mean CATE by {group_var} (95% CI)', fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'cate_by_{group_var}.{fmt}')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: {save_path}")


def plot_partial_dependence(model, data: pd.DataFrame, effect_modifiers: List[str],
                            top_n: int, figsize: tuple, output_dir: str,
                            fmt: str, dpi: int, verbose: bool):
    """Create partial dependence plots."""
    X = data[effect_modifiers].values

    # Get variable importance to select top variables
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[::-1][:top_n]
    top_vars = [effect_modifiers[i] for i in top_idx]

    n_vars = len(top_vars)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols/2, figsize[1]*n_rows/1.5))
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (var_idx, var_name) in enumerate(zip(top_idx, top_vars)):
        ax = axes[i]

        # Compute partial dependence
        x_vals = X[:, var_idx]
        grid = np.linspace(np.percentile(x_vals, 5), np.percentile(x_vals, 95), 50)

        pd_vals = []
        pd_se = []
        for val in grid:
            X_temp = X.copy()
            X_temp[:, var_idx] = val
            tau, se = model.predict(X_temp, return_std=True)
            pd_vals.append(np.mean(tau))
            if se is not None:
                pd_se.append(np.mean(se) / np.sqrt(len(tau)))
            else:
                pd_se.append(0)

        pd_vals = np.array(pd_vals)
        pd_se = np.array(pd_se)

        # Plot
        ax.plot(grid, pd_vals, 'b-', linewidth=2)
        ax.fill_between(grid, pd_vals - 1.96*pd_se, pd_vals + 1.96*pd_se,
                       alpha=0.2, color='blue')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel(var_name, fontsize=10)
        ax.set_ylabel('CATE', fontsize=10)
        ax.set_title(f'PD: {var_name}', fontsize=12)

    # Hide unused subplots
    for i in range(len(top_vars), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'partial_dependence.{fmt}')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: {save_path}")


def plot_gates(cate_df: pd.DataFrame, data: pd.DataFrame,
               outcome: str, treatment: str, figsize: tuple,
               output_dir: str, fmt: str, dpi: int, verbose: bool):
    """Create GATES plot."""
    tau = cate_df['cate'].values
    y = data[outcome].values
    W = data[treatment].values

    # Create quintile groups
    n_groups = 5
    quantiles = np.percentile(tau, np.linspace(0, 100, n_groups + 1)[1:-1])
    groups = np.digitize(tau, quantiles)

    gates = []
    for g in range(n_groups):
        mask = groups == g
        y_g = y[mask]
        t_g = W[mask]

        # Difference in means
        ate_g = np.mean(y_g[t_g == 1]) - np.mean(y_g[t_g == 0])
        n1 = np.sum(t_g == 1)
        n0 = np.sum(t_g == 0)
        var1 = np.var(y_g[t_g == 1]) / n1 if n1 > 1 else 0
        var0 = np.var(y_g[t_g == 0]) / n0 if n0 > 1 else 0
        se_g = np.sqrt(var1 + var0)

        gates.append({
            'group': g + 1,
            'predicted_mean': tau[mask].mean(),
            'actual_ate': ate_g,
            'ate_se': se_g
        })

    gates_df = pd.DataFrame(gates)

    fig, ax = plt.subplots(figsize=figsize)

    x = gates_df['group']
    ax.errorbar(x, gates_df['actual_ate'], yerr=1.96*gates_df['ate_se'],
                fmt='o-', capsize=5, capthick=2, markersize=10,
                linewidth=2, color='steelblue', label='Actual ATE')

    ax.plot(x, gates_df['predicted_mean'], 's--', markersize=8,
            color='coral', label='Predicted CATE mean')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('CATE Quintile', fontsize=12)
    ax.set_ylabel('Treatment Effect', fontsize=12)
    ax.set_title('Group Average Treatment Effects (GATES)', fontsize=14)
    ax.legend()
    ax.set_xticks(x)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'gates.{fmt}')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: {save_path}")


def create_dashboard(cate_df: pd.DataFrame, data: Optional[pd.DataFrame],
                     effect_modifiers: Optional[List[str]], figsize: tuple,
                     output_dir: str, fmt: str, dpi: int, verbose: bool):
    """Create comprehensive dashboard."""
    tau = cate_df['cate'].values

    fig = plt.figure(figsize=(figsize[0]*2, figsize[1]*2))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(tau, bins=50, density=True, alpha=0.7, color='steelblue')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(tau), color='green', linewidth=2)
    ax1.set_xlabel('CATE')
    ax1.set_ylabel('Density')
    ax1.set_title('CATE Distribution')

    # 2. Sorted CATEs
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_tau = np.sort(tau)
    ax2.plot(sorted_tau, color='steelblue', linewidth=0.5)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel('Observation')
    ax2.set_ylabel('CATE')
    ax2.set_title('Sorted CATEs')

    # 3. Summary statistics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    summary_text = f"""
    CATE Summary Statistics
    -----------------------
    N: {len(tau):,}
    Mean: {np.mean(tau):.4f}
    Std: {np.std(tau):.4f}
    Min: {np.min(tau):.4f}
    Max: {np.max(tau):.4f}
    Median: {np.median(tau):.4f}

    % Positive: {np.mean(tau > 0):.1%}
    % Negative: {np.mean(tau < 0):.1%}
    """
    if 'ci_lower' in cate_df.columns:
        n_sig_pos = (cate_df['ci_lower'] > 0).sum()
        n_sig_neg = (cate_df['ci_upper'] < 0).sum()
        summary_text += f"\n    Sig. positive: {n_sig_pos} ({n_sig_pos/len(tau):.1%})"
        summary_text += f"\n    Sig. negative: {n_sig_neg} ({n_sig_neg/len(tau):.1%})"

    ax3.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')

    # 4. Quantile breakdown
    ax4 = fig.add_subplot(gs[1, 0])
    quantiles = [10, 25, 50, 75, 90]
    q_values = [np.percentile(tau, q) for q in quantiles]
    ax4.barh(range(len(quantiles)), q_values, color='steelblue', alpha=0.7)
    ax4.set_yticks(range(len(quantiles)))
    ax4.set_yticklabels([f'{q}th' for q in quantiles])
    ax4.axvline(0, color='red', linestyle='--')
    ax4.set_xlabel('CATE')
    ax4.set_title('CATE Percentiles')

    # 5. Violin plot
    ax5 = fig.add_subplot(gs[1, 1])
    parts = ax5.violinplot([tau], positions=[1], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('steelblue')
    parts['bodies'][0].set_alpha(0.6)
    ax5.axhline(0, color='red', linestyle='--')
    ax5.set_xticks([1])
    ax5.set_xticklabels(['CATE'])
    ax5.set_ylabel('Treatment Effect')
    ax5.set_title('CATE Violin Plot')

    # 6. Cumulative distribution
    ax6 = fig.add_subplot(gs[1, 2])
    sorted_tau = np.sort(tau)
    cdf = np.arange(1, len(tau)+1) / len(tau)
    ax6.plot(sorted_tau, cdf, color='steelblue', linewidth=2)
    ax6.axvline(0, color='red', linestyle='--')
    ax6.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax6.set_xlabel('CATE')
    ax6.set_ylabel('Cumulative Probability')
    ax6.set_title('CATE CDF')

    plt.suptitle('Treatment Effect Heterogeneity Dashboard', fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'heterogeneity_dashboard.{fmt}')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: {save_path}")


def main():
    """Main entry point."""
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib required for plotting", file=sys.stderr)
        return 1

    args = parse_args()

    try:
        # Setup
        setup_style(args.style)
        os.makedirs(args.output, exist_ok=True)
        figsize = tuple(args.figsize)

        if args.verbose:
            print(f"Output directory: {args.output}")
            print(f"Format: {args.format}, DPI: {args.dpi}")

        # Load CATE data
        if args.cate_file:
            cate_df = load_cate_data(args.cate_file)
        else:
            # Load model and compute CATEs
            with open(args.model, 'rb') as f:
                model = pickle.load(f)
            data = load_data(args.data)
            X = data[args.effect_modifiers]
            tau, se = model.predict(X.values, return_std=True)
            from scipy import stats
            z = stats.norm.ppf(0.975)
            cate_df = pd.DataFrame({
                'cate': tau,
                'std_error': se if se is not None else np.zeros_like(tau),
                'ci_lower': tau - z * (se if se is not None else 0),
                'ci_upper': tau + z * (se if se is not None else 0)
            })

        # Load additional data if provided
        data = load_data(args.data) if args.data else None

        # Determine which plots to generate
        plots = args.plots if 'all' not in args.plots else [
            'distribution', 'sorted', 'subgroup', 'partial', 'gates', 'heatmap'
        ]

        if args.verbose:
            print(f"\nGenerating plots: {', '.join(plots)}")

        # Generate plots
        if 'distribution' in plots:
            plot_cate_distribution(cate_df, figsize, args.output, args.format,
                                  args.dpi, args.verbose)

        if 'sorted' in plots:
            plot_sorted_cate(cate_df, figsize, args.output, args.format,
                           args.dpi, args.verbose)

        if 'subgroup' in plots and args.group_by and data is not None:
            plot_subgroup_comparison(cate_df, data, args.group_by, figsize,
                                    args.output, args.format, args.dpi, args.verbose)

        # Dashboard
        create_dashboard(cate_df, data, args.effect_modifiers, figsize,
                        args.output, args.format, args.dpi, args.verbose)

        print(f"\nPlots saved to {args.output}/")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
