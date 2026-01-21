#!/usr/bin/env python
"""
Clustered Standard Error Computation

Provides various methods for computing cluster-robust standard errors:
- Standard clustered SE (Liang-Zeger)
- Two-way clustering
- Wild cluster bootstrap
- Pairs cluster bootstrap

Usage:
    python cluster_robust_se.py data.csv --entity firm_id --time year --y revenue --x treatment
    python cluster_robust_se.py data.csv --entity firm_id --time year --y revenue --x treatment --method wild --n-bootstrap 9999

Author: Causal ML Skills
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute cluster-robust standard errors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard clustered SE at entity level
  python cluster_robust_se.py data.csv --entity firm_id --time year --y revenue --x treatment

  # Two-way clustering (entity and time)
  python cluster_robust_se.py data.csv --entity firm_id --time year --y revenue --x treatment --twoway

  # Wild cluster bootstrap (for few clusters)
  python cluster_robust_se.py data.csv --entity firm_id --time year --y revenue --x treatment --method wild

  # Compare all methods
  python cluster_robust_se.py data.csv --entity firm_id --time year --y revenue --x treatment --compare-all
        """
    )

    parser.add_argument('data_file', type=str, help='Path to CSV data file')
    parser.add_argument('--entity', required=True, help='Entity identifier column')
    parser.add_argument('--time', required=True, help='Time period column')
    parser.add_argument('--y', required=True, help='Dependent variable column')
    parser.add_argument('--x', nargs='+', required=True, help='Independent variable columns')

    parser.add_argument('--cluster', type=str, help='Cluster column (default: entity)')
    parser.add_argument(
        '--method',
        choices=['standard', 'robust', 'wild', 'pairs', 'twoway'],
        default='standard',
        help='Clustering method'
    )

    parser.add_argument('--twoway', action='store_true', help='Two-way clustering (entity and time)')
    parser.add_argument('--n-bootstrap', type=int, default=999, help='Bootstrap iterations')
    parser.add_argument('--compare-all', action='store_true', help='Compare all SE methods')

    parser.add_argument('--fe', action='store_true', help='Use fixed effects (default: pooled)')
    parser.add_argument('--output', type=str, help='Output file for results (CSV)')

    return parser.parse_args()


def ols_fit(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Basic OLS estimation."""
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    residuals = y - X @ beta
    return beta, residuals


def within_transform(data: pd.DataFrame, y_col: str, x_cols: List[str], entity_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Apply within transformation for fixed effects."""
    y = data[y_col].values
    X = data[x_cols].values

    # Entity means
    y_mean = data.groupby(entity_col)[y_col].transform('mean').values
    X_mean = data.groupby(entity_col)[x_cols].transform('mean').values

    y_within = y - y_mean
    X_within = X - X_mean

    return y_within, X_within


def standard_se(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Conventional (non-robust) standard errors."""
    n, k = X.shape
    sigma2 = (residuals ** 2).sum() / (n - k)
    XtX_inv = inv(X.T @ X)
    vcov = sigma2 * XtX_inv
    return np.sqrt(np.diag(vcov))


def heteroskedasticity_robust_se(X: np.ndarray, residuals: np.ndarray, hc_type: str = 'HC1') -> np.ndarray:
    """
    Heteroskedasticity-robust (White) standard errors.

    HC0: No adjustment
    HC1: (n/(n-k)) adjustment
    HC2: 1/(1-h_ii) adjustment
    HC3: 1/(1-h_ii)^2 adjustment
    """
    n, k = X.shape
    XtX_inv = inv(X.T @ X)

    if hc_type == 'HC0':
        meat = X.T @ np.diag(residuals ** 2) @ X
    elif hc_type == 'HC1':
        meat = (n / (n - k)) * X.T @ np.diag(residuals ** 2) @ X
    elif hc_type in ['HC2', 'HC3']:
        # Hat matrix diagonal
        H = X @ XtX_inv @ X.T
        h = np.diag(H)

        if hc_type == 'HC2':
            weights = 1 / (1 - h)
        else:  # HC3
            weights = 1 / (1 - h) ** 2

        meat = X.T @ np.diag(weights * residuals ** 2) @ X
    else:
        raise ValueError(f"Unknown HC type: {hc_type}")

    vcov = XtX_inv @ meat @ XtX_inv
    return np.sqrt(np.diag(vcov))


def clustered_se(
    X: np.ndarray,
    residuals: np.ndarray,
    clusters: np.ndarray,
    finite_sample: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster-robust standard errors (Liang-Zeger).

    Returns both SE and full variance-covariance matrix.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)

    XtX_inv = inv(X.T @ X)

    # Sum of X'uu'X within clusters
    meat = np.zeros((k, k))

    for g in unique_clusters:
        mask = clusters == g
        X_g = X[mask]
        u_g = residuals[mask]

        score_g = X_g.T @ u_g  # Sum of X_i * u_i within cluster
        meat += np.outer(score_g, score_g)

    # Sandwich: (X'X)^{-1} meat (X'X)^{-1}
    vcov = XtX_inv @ meat @ XtX_inv

    # Finite sample correction
    if finite_sample:
        correction = (G / (G - 1)) * ((n - 1) / (n - k))
        vcov = vcov * correction

    return np.sqrt(np.diag(vcov)), vcov


def twoway_clustered_se(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-way clustered standard errors (Cameron, Gelbach, Miller 2011).

    V_twoway = V_cluster1 + V_cluster2 - V_intersection
    """
    # Cluster by dimension 1
    se1, vcov1 = clustered_se(X, residuals, cluster1, finite_sample=True)

    # Cluster by dimension 2
    se2, vcov2 = clustered_se(X, residuals, cluster2, finite_sample=True)

    # Cluster by intersection
    intersection = np.array([f"{c1}_{c2}" for c1, c2 in zip(cluster1, cluster2)])
    se_int, vcov_int = clustered_se(X, residuals, intersection, finite_sample=True)

    # Two-way variance
    vcov_twoway = vcov1 + vcov2 - vcov_int

    # Ensure positive semi-definiteness
    eigvals = np.linalg.eigvalsh(vcov_twoway)
    if eigvals.min() < 0:
        # Adjust negative eigenvalues
        print("Warning: Two-way variance matrix not positive semi-definite. Applying correction.")
        vcov_twoway = vcov_twoway + (-eigvals.min() + 1e-10) * np.eye(vcov_twoway.shape[0])

    return np.sqrt(np.diag(vcov_twoway)), vcov_twoway


def wild_cluster_bootstrap_se(
    X: np.ndarray,
    y: np.ndarray,
    residuals: np.ndarray,
    clusters: np.ndarray,
    n_bootstrap: int = 999,
    weight_type: str = 'rademacher'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wild cluster bootstrap standard errors.

    Better for small number of clusters (G < 50).
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)

    # Original estimate
    beta_orig, _ = ols_fit(y, X)

    # Bootstrap
    beta_boots = np.zeros((n_bootstrap, k))

    for b in range(n_bootstrap):
        # Generate cluster-level weights
        if weight_type == 'rademacher':
            weights = np.random.choice([-1, 1], size=G)
        elif weight_type == 'mammen':
            p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
            vals = [-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2]
            weights = np.random.choice(vals, size=G, p=[p, 1-p])
        elif weight_type == 'webb':
            vals = [-np.sqrt(3/2), -np.sqrt(1/2), -np.sqrt(1/2),
                    np.sqrt(1/2), np.sqrt(1/2), np.sqrt(3/2)]
            weights = np.random.choice(vals, size=G)
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")

        # Apply weights to residuals at cluster level
        resid_boot = np.zeros(n)
        for i, g in enumerate(unique_clusters):
            mask = clusters == g
            resid_boot[mask] = residuals[mask] * weights[i]

        # Bootstrap y
        y_boot = X @ beta_orig + resid_boot

        # Re-estimate
        beta_boots[b], _ = ols_fit(y_boot, X)

    # Variance from bootstrap distribution
    vcov = np.cov(beta_boots.T)
    return np.sqrt(np.diag(vcov)), vcov


def pairs_cluster_bootstrap_se(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    n_bootstrap: int = 999
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pairs cluster bootstrap (resample entire clusters).
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)

    # Bootstrap
    beta_boots = np.zeros((n_bootstrap, k))

    for b in range(n_bootstrap):
        # Sample clusters with replacement
        sampled_clusters = np.random.choice(unique_clusters, size=G, replace=True)

        # Build bootstrap sample
        X_boot = []
        y_boot = []

        for g in sampled_clusters:
            mask = clusters == g
            X_boot.append(X[mask])
            y_boot.append(y[mask])

        X_boot = np.vstack(X_boot)
        y_boot = np.concatenate(y_boot)

        # Estimate
        try:
            beta_boots[b], _ = ols_fit(y_boot, X_boot)
        except np.linalg.LinAlgError:
            beta_boots[b] = np.nan

    # Remove failed iterations
    valid = ~np.isnan(beta_boots).any(axis=1)
    beta_boots = beta_boots[valid]

    # Variance from bootstrap distribution
    vcov = np.cov(beta_boots.T)
    return np.sqrt(np.diag(vcov)), vcov


def compute_inference(
    beta: np.ndarray,
    se: np.ndarray,
    var_names: List[str],
    df: int
) -> pd.DataFrame:
    """Compute t-stats, p-values, and confidence intervals."""
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))

    t_crit = stats.t.ppf(0.975, df)
    ci_lower = beta - t_crit * se
    ci_upper = beta + t_crit * se

    return pd.DataFrame({
        'variable': var_names,
        'coefficient': beta,
        'std_error': se,
        't_stat': t_stats,
        'p_value': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    })


def print_results(results: pd.DataFrame, method: str, n_clusters: int):
    """Pretty print results."""
    print(f"\n{'='*70}")
    print(f"Standard Errors: {method}")
    print(f"Number of clusters: {n_clusters}")
    print(f"{'='*70}")
    print(f"{'Variable':<15} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8} {'95% CI':>20}")
    print(f"{'-'*70}")

    for _, row in results.iterrows():
        sig = ""
        if row['p_value'] < 0.01:
            sig = "***"
        elif row['p_value'] < 0.05:
            sig = "**"
        elif row['p_value'] < 0.1:
            sig = "*"

        ci = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        print(f"{row['variable']:<15} {row['coefficient']:>10.4f} {row['std_error']:>10.4f} "
              f"{row['t_stat']:>8.3f} {row['p_value']:>8.4f}{sig} {ci:>20}")

    print(f"{'-'*70}")
    print("*** p<0.01, ** p<0.05, * p<0.1")


def main():
    args = parse_args()

    # Load data
    print(f"Loading data from {args.data_file}...")
    data = pd.read_csv(args.data_file)

    # Validate columns
    required_cols = [args.entity, args.time, args.y] + args.x
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        sys.exit(1)

    # Prepare data
    cluster_col = args.cluster if args.cluster else args.entity
    clusters = data[cluster_col].values
    n_clusters = len(np.unique(clusters))

    print(f"\nData summary:")
    print(f"  Observations: {len(data):,}")
    print(f"  Clusters ({cluster_col}): {n_clusters}")
    print(f"  Avg cluster size: {len(data) / n_clusters:.1f}")

    # Warn if few clusters
    if n_clusters < 30:
        print(f"\n  WARNING: Few clusters ({n_clusters} < 30)")
        print("  Consider wild cluster bootstrap for more reliable inference")

    # Fit model
    if args.fe:
        print("\nUsing within transformation (Fixed Effects)...")
        y, X = within_transform(data, args.y, args.x, args.entity)
        var_names = args.x
        df = n_clusters - 1  # Use G-1 for clustered inference
    else:
        print("\nUsing pooled OLS...")
        y = data[args.y].values
        X = np.column_stack([np.ones(len(data)), data[args.x].values])
        var_names = ['_const'] + args.x
        df = n_clusters - 1

    beta, residuals = ols_fit(y, X)

    # Compute standard errors
    all_results = {}

    if args.compare_all:
        methods = ['conventional', 'robust', 'clustered', 'wild', 'pairs']
        if args.twoway or n_clusters < 30:
            methods.append('twoway')
    else:
        methods = [args.method]
        if args.twoway:
            methods = ['twoway']

    for method in methods:
        if method == 'conventional':
            se = standard_se(X, residuals)
            method_name = 'Conventional (Non-Robust)'
            df_use = len(y) - len(beta)
        elif method == 'robust':
            se = heteroskedasticity_robust_se(X, residuals, 'HC1')
            method_name = 'Heteroskedasticity-Robust (HC1)'
            df_use = len(y) - len(beta)
        elif method in ['standard', 'clustered']:
            se, _ = clustered_se(X, residuals, clusters)
            method_name = f'Cluster-Robust ({cluster_col})'
            df_use = df
        elif method == 'twoway':
            se, _ = twoway_clustered_se(
                X, residuals,
                data[args.entity].values,
                data[args.time].values
            )
            method_name = f'Two-Way Clustered ({args.entity} x {args.time})'
            df_use = min(
                len(np.unique(data[args.entity])),
                len(np.unique(data[args.time]))
            ) - 1
        elif method == 'wild':
            se, _ = wild_cluster_bootstrap_se(
                X, y, residuals, clusters,
                n_bootstrap=args.n_bootstrap
            )
            method_name = f'Wild Cluster Bootstrap (B={args.n_bootstrap})'
            df_use = df
        elif method == 'pairs':
            se, _ = pairs_cluster_bootstrap_se(
                X, y, clusters,
                n_bootstrap=args.n_bootstrap
            )
            method_name = f'Pairs Cluster Bootstrap (B={args.n_bootstrap})'
            df_use = df
        else:
            continue

        results = compute_inference(beta, se, var_names, df_use)
        all_results[method_name] = results
        print_results(results, method_name, n_clusters)

    # Comparison summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON OF STANDARD ERRORS")
        print(f"{'='*70}")

        comparison = pd.DataFrame({
            method: results.set_index('variable')['std_error']
            for method, results in all_results.items()
        })

        print(comparison.to_string())

        # Ratio to conventional
        if 'Conventional (Non-Robust)' in comparison.columns:
            print("\nRatio to conventional SE:")
            ratios = comparison.div(comparison['Conventional (Non-Robust)'], axis=0)
            print(ratios.to_string())

    # Save output
    if args.output:
        # Save the primary method results
        primary_method = list(all_results.keys())[-1]
        all_results[primary_method].to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
