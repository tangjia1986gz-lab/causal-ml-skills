#!/usr/bin/env python3
"""
Causal Forest Analysis CLI

Complete causal forest analysis pipeline with support for both
Python (econml) and R (grf via rpy2) backends.

Usage:
    python run_causal_forest.py --data data.csv --outcome Y --treatment W \
        --effect-modifiers X1 X2 X3 --output results/

Author: Causal ML Skills
"""

import argparse
import sys
import os
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_forest import (
    fit_causal_forest,
    estimate_cate,
    variable_importance,
    best_linear_projection,
    heterogeneity_test,
    policy_learning,
    CausalForestConfig,
    PolicyConfig,
    CATEEstimates,
    CausalOutput
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run causal forest analysis for heterogeneous treatment effects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python run_causal_forest.py --data experiment.csv --outcome revenue \\
        --treatment discount --effect-modifiers age income tenure

    # With confounders and custom config
    python run_causal_forest.py --data data.csv --outcome Y --treatment W \\
        --effect-modifiers X1 X2 X3 --confounders C1 C2 \\
        --n-trees 4000 --min-node-size 10 --output results/

    # Using R grf backend
    python run_causal_forest.py --data data.csv --outcome Y --treatment W \\
        --effect-modifiers X1 X2 --backend grf --output results/

    # With policy learning
    python run_causal_forest.py --data data.csv --outcome Y --treatment W \\
        --effect-modifiers X1 X2 --policy --treatment-cost 10 --budget 0.3
        """
    )

    # Required arguments
    parser.add_argument('--data', required=True,
                        help='Path to CSV data file')
    parser.add_argument('--outcome', required=True,
                        help='Name of outcome variable')
    parser.add_argument('--treatment', required=True,
                        help='Name of treatment variable')
    parser.add_argument('--effect-modifiers', nargs='+', required=True,
                        help='Names of effect modifier variables')

    # Optional data arguments
    parser.add_argument('--confounders', nargs='+', default=None,
                        help='Names of additional confounder variables')
    parser.add_argument('--cluster', default=None,
                        help='Name of cluster variable for clustered SEs')

    # Model configuration
    parser.add_argument('--backend', choices=['auto', 'econml', 'grf', 'custom'],
                        default='auto',
                        help='Backend to use (default: auto)')
    parser.add_argument('--n-trees', type=int, default=2000,
                        help='Number of trees (default: 2000)')
    parser.add_argument('--min-node-size', type=int, default=5,
                        help='Minimum node size (default: 5)')
    parser.add_argument('--honesty-fraction', type=float, default=0.5,
                        help='Fraction for honest estimation (default: 0.5)')
    parser.add_argument('--sample-fraction', type=float, default=0.5,
                        help='Subsampling fraction (default: 0.5)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level (default: 0.05)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (default: -1 for all)')

    # Policy learning
    parser.add_argument('--policy', action='store_true',
                        help='Learn optimal treatment policy')
    parser.add_argument('--treatment-cost', type=float, default=0.0,
                        help='Cost of treatment (default: 0)')
    parser.add_argument('--budget', type=float, default=None,
                        help='Budget fraction (max proportion to treat)')
    parser.add_argument('--policy-method', choices=['threshold', 'policy_tree', 'optimal'],
                        default='threshold',
                        help='Policy learning method (default: threshold)')

    # Output options
    parser.add_argument('--output', '-o', default='./causal_forest_results',
                        help='Output directory (default: ./causal_forest_results)')
    parser.add_argument('--format', choices=['csv', 'json', 'both'],
                        default='both',
                        help='Output format (default: both)')
    parser.add_argument('--plots', action='store_true', default=True,
                        help='Generate plots (default: True)')
    parser.add_argument('--no-plots', action='store_false', dest='plots',
                        help='Disable plot generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress output')

    return parser.parse_args()


def load_data(data_path: str, required_cols: List[str]) -> pd.DataFrame:
    """Load and validate data."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Determine file type
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        df = pd.read_excel(data_path)
    else:
        # Try CSV as default
        df = pd.read_csv(data_path)

    # Validate columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    return df


def run_analysis(args) -> Dict[str, Any]:
    """Run complete causal forest analysis."""
    results = {'metadata': {
        'timestamp': datetime.now().isoformat(),
        'data_path': args.data,
        'outcome': args.outcome,
        'treatment': args.treatment,
        'effect_modifiers': args.effect_modifiers,
        'confounders': args.confounders,
        'backend': args.backend
    }}

    # Load data
    if not args.quiet:
        print(f"Loading data from {args.data}...")

    required_cols = [args.outcome, args.treatment] + args.effect_modifiers
    if args.confounders:
        required_cols += args.confounders
    if args.cluster:
        required_cols.append(args.cluster)

    data = load_data(args.data, required_cols)
    results['metadata']['n_observations'] = len(data)

    if not args.quiet:
        print(f"  Loaded {len(data):,} observations")

    # Prepare data
    X = data[args.effect_modifiers]
    y = data[args.outcome].values
    treatment = data[args.treatment].values
    X_adjust = data[args.confounders] if args.confounders else None

    # Configure model
    config = CausalForestConfig(
        n_estimators=args.n_trees,
        honesty=True,
        honesty_fraction=args.honesty_fraction,
        min_node_size=args.min_node_size,
        sample_fraction=args.sample_fraction,
        alpha=args.alpha,
        random_state=args.seed,
        n_jobs=args.n_jobs
    )

    results['config'] = {
        'n_estimators': config.n_estimators,
        'honesty': config.honesty,
        'honesty_fraction': config.honesty_fraction,
        'min_node_size': config.min_node_size,
        'sample_fraction': config.sample_fraction,
        'alpha': config.alpha,
        'random_state': config.random_state
    }

    # Fit causal forest
    if not args.quiet:
        print(f"\nFitting causal forest ({args.n_trees} trees)...")

    cf_model = fit_causal_forest(
        X=X,
        y=y,
        treatment=treatment,
        X_adjust=X_adjust,
        config=config,
        backend=args.backend
    )

    if not args.quiet:
        print("  Model fitted successfully")

    # Estimate CATEs
    if not args.quiet:
        print("\nEstimating conditional average treatment effects...")

    cate_results = estimate_cate(cf_model, X, alpha=args.alpha)

    results['cate'] = {
        'mean': float(cate_results.mean),
        'std': float(cate_results.std),
        'min': float(cate_results.estimates.min()),
        'max': float(cate_results.estimates.max()),
        'median': float(np.median(cate_results.estimates)),
        'q25': float(np.percentile(cate_results.estimates, 25)),
        'q75': float(np.percentile(cate_results.estimates, 75)),
        'proportion_positive': float(cate_results.proportion_positive),
        'proportion_significant': float(cate_results.proportion_significant)
    }

    if not args.quiet:
        print(f"  Mean CATE (ATE): {cate_results.mean:.4f}")
        print(f"  CATE Range: [{results['cate']['min']:.4f}, {results['cate']['max']:.4f}]")
        print(f"  % Positive: {cate_results.proportion_positive:.1%}")

    # Variable importance
    if not args.quiet:
        print("\nCalculating variable importance...")

    var_imp = variable_importance(cf_model)
    results['variable_importance'] = {
        name: float(score) for name, score in var_imp.ranked
    }

    if not args.quiet:
        print("  Top heterogeneity drivers:")
        for name, score in var_imp.ranked[:5]:
            print(f"    {name}: {score:.4f}")

    # Heterogeneity test
    if not args.quiet:
        print("\nTesting for heterogeneity...")

    het_test = heterogeneity_test(cf_model, X, cate_results)
    results['heterogeneity_test'] = {
        'statistic': float(het_test.statistic),
        'p_value': float(het_test.pvalue),
        'significant': het_test.significant,
        'method': het_test.method
    }

    if not args.quiet:
        sig_str = "Significant" if het_test.significant else "Not significant"
        print(f"  {sig_str} heterogeneity (p={het_test.pvalue:.4f})")

    # Best linear projection
    if not args.quiet:
        print("\nComputing best linear projection...")

    blp = best_linear_projection(cf_model, X, cate_results)
    results['blp'] = {
        'intercept': float(blp.intercept),
        'intercept_se': float(blp.intercept_se),
        'r_squared': float(blp.r_squared),
        'coefficients': {
            name: {'estimate': float(coef), 'std_error': float(se)}
            for name, coef, se in zip(blp.feature_names, blp.coefficients, blp.std_errors)
        }
    }

    if not args.quiet:
        print(f"  R-squared: {blp.r_squared:.4f}")

    # Policy learning
    if args.policy:
        if not args.quiet:
            print("\nLearning optimal treatment policy...")

        policy_config = PolicyConfig(
            treatment_cost=args.treatment_cost,
            budget_fraction=args.budget,
            method=args.policy_method
        )

        policy = policy_learning(cf_model, X, policy_config, cate_results)

        results['policy'] = {
            'treatment_rate': float(policy.treatment_rate),
            'value': float(policy.value),
            'improvement': float(policy.improvement),
            'method': args.policy_method
        }

        if not args.quiet:
            print(f"  Treatment rate: {policy.treatment_rate:.1%}")
            print(f"  Policy value: {policy.value:.4f}")
            print(f"  Improvement over treat-all: {policy.improvement:.1%}")

        # Store recommendations
        results['policy']['recommendations'] = policy.recommendations.tolist()

    # Store detailed CATE estimates
    cate_df = pd.DataFrame({
        'cate': cate_results.estimates,
        'std_error': cate_results.std_errors,
        'ci_lower': cate_results.ci_lower,
        'ci_upper': cate_results.ci_upper
    })

    return results, cate_df, cf_model, cate_results, var_imp


def save_results(results: Dict, cate_df: pd.DataFrame, output_dir: str,
                 output_format: str, quiet: bool):
    """Save analysis results."""
    os.makedirs(output_dir, exist_ok=True)

    if output_format in ['json', 'both']:
        json_path = os.path.join(output_dir, 'results.json')
        # Make results JSON serializable
        results_json = json.loads(json.dumps(results, default=str))
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        if not quiet:
            print(f"\nResults saved to {json_path}")

    if output_format in ['csv', 'both']:
        cate_path = os.path.join(output_dir, 'cate_estimates.csv')
        cate_df.to_csv(cate_path, index=False)
        if not quiet:
            print(f"CATE estimates saved to {cate_path}")

        # Save variable importance
        imp_df = pd.DataFrame([
            {'variable': k, 'importance': v}
            for k, v in results['variable_importance'].items()
        ])
        imp_path = os.path.join(output_dir, 'variable_importance.csv')
        imp_df.to_csv(imp_path, index=False)

        # Save BLP results
        blp_df = pd.DataFrame([
            {'variable': 'intercept',
             'coefficient': results['blp']['intercept'],
             'std_error': results['blp']['intercept_se']}
        ] + [
            {'variable': k,
             'coefficient': v['estimate'],
             'std_error': v['std_error']}
            for k, v in results['blp']['coefficients'].items()
        ])
        blp_path = os.path.join(output_dir, 'blp_coefficients.csv')
        blp_df.to_csv(blp_path, index=False)


def generate_plots(cf_model, cate_results, var_imp, output_dir: str, quiet: bool):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
        from causal_forest import (
            plot_cate_distribution,
            plot_variable_importance
        )
    except ImportError:
        if not quiet:
            print("Warning: matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # CATE distribution
    plot_cate_distribution(
        cate_results,
        title="Distribution of Individual Treatment Effects",
        save_path=os.path.join(output_dir, 'cate_distribution.png')
    )
    plt.close()

    # Variable importance
    plot_variable_importance(
        var_imp,
        title="Drivers of Treatment Effect Heterogeneity",
        save_path=os.path.join(output_dir, 'variable_importance.png')
    )
    plt.close()

    if not quiet:
        print(f"Plots saved to {output_dir}")


def print_summary(results: Dict):
    """Print analysis summary."""
    print("\n" + "=" * 60)
    print("CAUSAL FOREST ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nData: {results['metadata']['n_observations']:,} observations")
    print(f"Outcome: {results['metadata']['outcome']}")
    print(f"Treatment: {results['metadata']['treatment']}")
    print(f"Effect modifiers: {', '.join(results['metadata']['effect_modifiers'])}")

    print(f"\n--- Treatment Effect Estimates ---")
    print(f"Average Treatment Effect: {results['cate']['mean']:.4f}")
    print(f"CATE Std Dev: {results['cate']['std']:.4f}")
    print(f"CATE Range: [{results['cate']['min']:.4f}, {results['cate']['max']:.4f}]")
    print(f"% Positive Effects: {results['cate']['proportion_positive']:.1%}")
    print(f"% Significant Effects: {results['cate']['proportion_significant']:.1%}")

    print(f"\n--- Heterogeneity ---")
    het = results['heterogeneity_test']
    print(f"Test statistic: {het['statistic']:.2f}")
    print(f"p-value: {het['p_value']:.4f}")
    print(f"Significant: {'Yes' if het['significant'] else 'No'}")

    print(f"\n--- Top Heterogeneity Drivers ---")
    for i, (var, imp) in enumerate(list(results['variable_importance'].items())[:5], 1):
        print(f"  {i}. {var}: {imp:.4f}")

    if 'policy' in results:
        print(f"\n--- Optimal Policy ---")
        print(f"Treatment rate: {results['policy']['treatment_rate']:.1%}")
        print(f"Policy value: {results['policy']['value']:.4f}")
        print(f"Improvement: {results['policy']['improvement']:.1%}")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Run analysis
        results, cate_df, cf_model, cate_results, var_imp = run_analysis(args)

        # Save results
        save_results(results, cate_df, args.output, args.format, args.quiet)

        # Generate plots
        if args.plots:
            generate_plots(cf_model, cate_results, var_imp, args.output, args.quiet)

        # Print summary
        if not args.quiet:
            print_summary(results)

        if not args.quiet:
            print(f"\nAnalysis complete. Results saved to {args.output}/")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
