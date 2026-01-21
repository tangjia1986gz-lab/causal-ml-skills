#!/usr/bin/env python
"""
Outlier Detection CLI

Detect outliers and influential observations in econometric data.

Usage:
    python detect_outliers.py --data data.csv --vars y,x1,x2,x3
    python detect_outliers.py --data data.csv --vars x1,x2 --method mahalanobis
    python detect_outliers.py --data data.csv --outcome y --regressors x1,x2 --influence
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from econometric_eda import EconometricEDA


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detect outliers and influential observations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # IQR-based outlier detection (default)
    python detect_outliers.py --data data.csv --vars y,x1,x2

    # Z-score method with custom threshold
    python detect_outliers.py --data data.csv --vars y,x1 --method zscore --threshold 2.5

    # MAD method (robust to outliers)
    python detect_outliers.py --data data.csv --vars y,x1 --method mad

    # Multivariate outliers (Mahalanobis distance)
    python detect_outliers.py --data data.csv --vars x1,x2,x3 --method mahalanobis

    # Isolation Forest (ML-based)
    python detect_outliers.py --data data.csv --vars x1,x2,x3 --method isolation_forest --contamination 0.05

    # Regression influence diagnostics
    python detect_outliers.py --data data.csv --outcome y --regressors x1,x2,x3 --influence

    # Compare multiple methods
    python detect_outliers.py --data data.csv --vars y,x1,x2 --compare-methods
        """
    )

    parser.add_argument('--data', '-d', required=True,
                        help='Path to CSV data file')
    parser.add_argument('--vars', '-v',
                        help='Comma-separated list of variables')
    parser.add_argument('--method', '-m', default='iqr',
                        choices=['iqr', 'zscore', 'mad', 'mahalanobis', 'isolation_forest'],
                        help='Outlier detection method')
    parser.add_argument('--threshold', '-t', type=float,
                        help='Detection threshold (method-specific)')
    parser.add_argument('--contamination', type=float, default=0.05,
                        help='Expected outlier proportion (for isolation_forest)')
    parser.add_argument('--outcome',
                        help='Outcome variable for regression influence')
    parser.add_argument('--regressors',
                        help='Comma-separated regressors for influence analysis')
    parser.add_argument('--influence', action='store_true',
                        help='Run regression influence diagnostics')
    parser.add_argument('--compare-methods', action='store_true',
                        help='Compare multiple outlier detection methods')
    parser.add_argument('--output', '-o',
                        help='Output directory for results')
    parser.add_argument('--flag', action='store_true',
                        help='Add outlier flags to data and save')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')

    return parser.parse_args()


def print_univariate_results(results: dict, method: str):
    """Print univariate outlier detection results."""
    print(f"\n## {method.upper()} Method Results\n")

    total_outliers = 0
    summary_data = []

    for var, res in results.items():
        n_outliers = res.get('n_outliers', 0)
        pct_outliers = res.get('pct_outliers', 0)
        total_outliers += n_outliers

        summary_data.append({
            'Variable': var,
            'N_Outliers': n_outliers,
            'Pct_Outliers': f"{pct_outliers:.1f}%"
        })

        if 'lower_bound' in res and 'upper_bound' in res:
            summary_data[-1]['Lower'] = f"{res['lower_bound']:.3f}"
            summary_data[-1]['Upper'] = f"{res['upper_bound']:.3f}"

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print(f"\nTotal unique observations flagged: {total_outliers}")


def print_multivariate_results(results: dict, method: str):
    """Print multivariate outlier detection results."""
    print(f"\n## {method.upper()} Method Results\n")

    print(f"  - Outliers detected: {results.get('n_outliers', 'N/A')}")
    print(f"  - Percentage: {results.get('pct_outliers', 0):.1f}%")

    if 'chi2_threshold' in results:
        print(f"  - Chi-square threshold: {results['chi2_threshold']:.3f}")

    if 'contamination' in results:
        print(f"  - Expected contamination: {results['contamination']*100:.1f}%")

    # Show top outliers
    if 'results' in results:
        res_df = results['results']
        outlier_df = res_df[res_df['is_outlier']].copy()

        if len(outlier_df) > 0:
            print(f"\n  Top 10 outliers:")
            if 'mahalanobis_distance' in outlier_df.columns:
                outlier_df = outlier_df.sort_values('mahalanobis_distance', ascending=False)
                print(outlier_df.head(10).to_string())
            elif 'anomaly_score' in outlier_df.columns:
                outlier_df = outlier_df.sort_values('anomaly_score', ascending=True)
                print(outlier_df.head(10).to_string())


def run_influence_diagnostics(data: pd.DataFrame, outcome: str, regressors: list, output_dir: str = None):
    """Run regression influence diagnostics."""
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import OLSInfluence
    except ImportError:
        print("Error: statsmodels required for influence diagnostics")
        return

    print("\n## Regression Influence Diagnostics\n")

    # Prepare data
    subset = data[[outcome] + regressors].dropna()
    y = subset[outcome]
    X = sm.add_constant(subset[regressors])

    # Fit model
    model = sm.OLS(y, X).fit()
    influence = OLSInfluence(model)

    n, p = X.shape

    # Collect diagnostics
    diagnostics = pd.DataFrame({
        'residual': model.resid,
        'std_residual': influence.resid_studentized_internal,
        'leverage': influence.hat_matrix_diag,
        'cooks_d': influence.cooks_distance[0],
        'dffits': influence.dffits[0]
    }, index=subset.index)

    # Thresholds
    cooks_threshold = 4 / n
    leverage_threshold = 2 * p / n
    dffits_threshold = 2 * np.sqrt(p / n)

    # Flag influential observations
    diagnostics['flag_cooks'] = diagnostics['cooks_d'] > cooks_threshold
    diagnostics['flag_leverage'] = diagnostics['leverage'] > leverage_threshold
    diagnostics['flag_dffits'] = np.abs(diagnostics['dffits']) > dffits_threshold
    diagnostics['flag_residual'] = np.abs(diagnostics['std_residual']) > 2

    diagnostics['any_flag'] = (
        diagnostics['flag_cooks'] |
        diagnostics['flag_leverage'] |
        diagnostics['flag_dffits']
    )

    # Summary
    print(f"Model: {outcome} ~ {' + '.join(regressors)}")
    print(f"N: {n}, k: {p-1}\n")

    print("Thresholds:")
    print(f"  - Cook's D: {cooks_threshold:.4f}")
    print(f"  - Leverage: {leverage_threshold:.4f}")
    print(f"  - DFFITS: {dffits_threshold:.4f}")
    print()

    print("Flagged observations:")
    print(f"  - Cook's D > threshold: {diagnostics['flag_cooks'].sum()}")
    print(f"  - High leverage: {diagnostics['flag_leverage'].sum()}")
    print(f"  - |DFFITS| > threshold: {diagnostics['flag_dffits'].sum()}")
    print(f"  - |Std residual| > 2: {diagnostics['flag_residual'].sum()}")
    print(f"  - Any influence flag: {diagnostics['any_flag'].sum()}")

    # Show most influential
    influential = diagnostics[diagnostics['any_flag']].copy()
    if len(influential) > 0:
        influential['total_flags'] = (
            influential['flag_cooks'].astype(int) +
            influential['flag_leverage'].astype(int) +
            influential['flag_dffits'].astype(int)
        )
        influential = influential.sort_values('total_flags', ascending=False)

        print(f"\nTop 10 most influential observations:")
        cols = ['cooks_d', 'leverage', 'dffits', 'std_residual', 'total_flags']
        print(influential[cols].head(10).to_string())

    # Save if output specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        diag_path = output_path / 'influence_diagnostics.csv'
        diagnostics.to_csv(diag_path)
        print(f"\nDiagnostics saved to: {diag_path}")

    return diagnostics


def compare_methods(data: pd.DataFrame, variables: list):
    """Compare multiple outlier detection methods."""
    print("\n## Method Comparison\n")

    eda = EconometricEDA(data)

    methods = ['iqr', 'zscore', 'mad']
    comparison = {}

    for var in variables:
        comparison[var] = {}
        for method in methods:
            results = eda.detect_outliers([var], method=method)
            comparison[var][method] = results[var]['n_outliers']

    # Try multivariate methods
    if len(variables) > 1:
        try:
            maha_results = eda.detect_outliers(variables, method='mahalanobis')
            comparison['_multivariate'] = {'mahalanobis': maha_results['n_outliers']}
        except Exception:
            pass

        try:
            iso_results = eda.detect_outliers(variables, method='isolation_forest')
            comparison['_multivariate']['isolation_forest'] = iso_results['n_outliers']
        except Exception:
            pass

    # Print comparison table
    comparison_data = []
    for var in variables:
        row = {'Variable': var}
        row.update(comparison[var])
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    if '_multivariate' in comparison:
        print("\nMultivariate methods (all variables combined):")
        for method, count in comparison['_multivariate'].items():
            print(f"  - {method}: {count} outliers")


def create_outlier_visualizations(data: pd.DataFrame, variables: list, results: dict,
                                   method: str, output_dir: str):
    """Create outlier visualizations."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not available. Skipping visualizations.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_vars = len(variables)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, var in enumerate(variables):
        ax = axes[i]

        # Box plot
        ax.boxplot(data[var].dropna(), vert=True)

        # Highlight outliers if available
        if var in results and 'outlier_mask' in results[var]:
            outlier_mask = results[var]['outlier_mask']
            outlier_values = data.loc[outlier_mask, var].dropna()
            ax.scatter([1]*len(outlier_values), outlier_values, color='red',
                      marker='o', s=50, label=f'Outliers ({len(outlier_values)})')

        ax.set_title(f'{var}\n({results[var]["n_outliers"]} outliers)')
        ax.set_ylabel(var)

    # Remove empty subplots
    for i in range(n_vars, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    fig_path = output_path / f'outliers_{method}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {fig_path}")


def main():
    args = parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    try:
        data = pd.read_csv(args.data)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print(f"Loaded {len(data)} observations, {len(data.columns)} variables")

    # Initialize EDA
    eda = EconometricEDA(data)

    print("\n" + "="*60)
    print("OUTLIER DETECTION")
    print("="*60)

    # Method comparison
    if args.compare_methods:
        if args.vars:
            variables = [v.strip() for v in args.vars.split(',')]
        else:
            variables = [c for c in data.columns if data[c].dtype in ['float64', 'int64']][:5]
        compare_methods(data, variables)
        return

    # Influence diagnostics
    if args.influence:
        if not args.outcome or not args.regressors:
            print("Error: --influence requires --outcome and --regressors")
            sys.exit(1)
        regressors = [r.strip() for r in args.regressors.split(',')]
        run_influence_diagnostics(data, args.outcome, regressors, args.output)
        return

    # Standard outlier detection
    if args.vars:
        variables = [v.strip() for v in args.vars.split(',')]
    else:
        variables = [c for c in data.columns if data[c].dtype in ['float64', 'int64']]

    # Build kwargs
    kwargs = {}
    if args.threshold is not None:
        if args.method == 'iqr':
            kwargs['k'] = args.threshold
        elif args.method in ['zscore', 'mad']:
            kwargs['threshold'] = args.threshold
        elif args.method == 'mahalanobis':
            kwargs['threshold_pvalue'] = args.threshold
    if args.method == 'isolation_forest':
        kwargs['contamination'] = args.contamination

    # Run detection
    results = eda.detect_outliers(variables, method=args.method, **kwargs)

    # Print results
    if args.method in ['iqr', 'zscore', 'mad']:
        print_univariate_results(results, args.method)
    else:
        print_multivariate_results(results, args.method)

    # Visualizations
    if args.plot and args.output:
        create_outlier_visualizations(data, variables, results, args.method, args.output)

    # Flag and save
    if args.flag and args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Add flags to data
        flagged_data = data.copy()
        for var, res in results.items():
            if 'outlier_mask' in res:
                flagged_data[f'{var}_outlier'] = res['outlier_mask']

        flagged_path = output_path / 'data_with_outlier_flags.csv'
        flagged_data.to_csv(flagged_path, index=False)
        print(f"\nFlagged data saved to: {flagged_path}")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
