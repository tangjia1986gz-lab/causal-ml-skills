#!/usr/bin/env python3
"""
Feature Importance Analysis Script

This script provides comprehensive feature importance analysis including
impurity-based, permutation, and SHAP-based importance measures.

Usage:
    python analyze_importance.py model.joblib data.csv --target y
    python analyze_importance.py model.joblib data.csv --target y --method all
    python analyze_importance.py model.joblib data.csv --target y --shap --save-plots
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from tree_models import get_feature_importance, compute_shap_values


def load_model_and_data(model_path: str, data_path: str, target: str,
                        features: Optional[str] = None) -> Dict[str, Any]:
    """Load model and data for analysis."""
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    if features:
        feature_cols = [f.strip() for f in features.split(',')]
    else:
        feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols].values
    y = df[target].values

    return {
        'model': model,
        'X': X,
        'y': y,
        'feature_names': feature_cols,
        'df': df
    }


def analyze_impurity_importance(data: Dict[str, Any]) -> pd.DataFrame:
    """Compute impurity-based feature importance."""
    print("\n=== Impurity-Based Importance ===")

    importance = get_feature_importance(
        data['model'],
        data['feature_names'],
        method='impurity'
    )

    print("\nTop 15 Features:")
    print(importance.head(15).to_string(index=False))

    return importance


def analyze_permutation_importance(data: Dict[str, Any], n_repeats: int = 30,
                                   random_state: int = 42) -> pd.DataFrame:
    """Compute permutation importance."""
    print("\n=== Permutation Importance ===")
    print(f"(n_repeats={n_repeats})")

    importance = get_feature_importance(
        data['model'],
        data['feature_names'],
        method='permutation',
        X=data['X'],
        y=data['y'],
        n_repeats=n_repeats,
        random_state=random_state
    )

    print("\nTop 15 Features:")
    print(importance.head(15).to_string(index=False))

    return importance


def analyze_shap_importance(data: Dict[str, Any], max_samples: int = 1000,
                            plot: bool = True) -> Dict[str, Any]:
    """Compute SHAP values and importance."""
    print("\n=== SHAP Importance ===")

    # Subsample if too large
    X = data['X']
    if len(X) > max_samples:
        print(f"Subsampling to {max_samples} samples for SHAP analysis")
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]

    shap_result = compute_shap_values(
        data['model'],
        X,
        feature_names=data['feature_names'],
        plot_summary=plot,
        plot_bar=plot
    )

    print("\nTop 15 Features (by mean |SHAP|):")
    print(shap_result['feature_importance'].head(15).to_string(index=False))

    return shap_result


def compare_importance_methods(impurity: pd.DataFrame, permutation: pd.DataFrame,
                               shap_importance: pd.DataFrame) -> pd.DataFrame:
    """Compare feature rankings across importance methods."""
    print("\n=== Importance Method Comparison ===")

    # Merge all importance DataFrames
    comparison = impurity.rename(columns={'importance': 'impurity'})
    comparison = comparison.merge(
        permutation[['feature', 'importance']].rename(columns={'importance': 'permutation'}),
        on='feature'
    )
    comparison = comparison.merge(
        shap_importance[['feature', 'importance']].rename(columns={'importance': 'shap'}),
        on='feature'
    )

    # Normalize each to [0, 1]
    for col in ['impurity', 'permutation', 'shap']:
        max_val = comparison[col].max()
        if max_val > 0:
            comparison[f'{col}_norm'] = comparison[col] / max_val

    # Compute rank for each method
    comparison['rank_impurity'] = comparison['impurity'].rank(ascending=False).astype(int)
    comparison['rank_permutation'] = comparison['permutation'].rank(ascending=False).astype(int)
    comparison['rank_shap'] = comparison['shap'].rank(ascending=False).astype(int)

    # Average rank
    comparison['avg_rank'] = comparison[['rank_impurity', 'rank_permutation', 'rank_shap']].mean(axis=1)

    # Sort by average rank
    comparison = comparison.sort_values('avg_rank')

    print("\nFeature Rankings Comparison (Top 15):")
    cols_to_show = ['feature', 'rank_impurity', 'rank_permutation', 'rank_shap', 'avg_rank']
    print(comparison[cols_to_show].head(15).to_string(index=False))

    # Rank correlation
    from scipy.stats import spearmanr

    corr_imp_perm, _ = spearmanr(comparison['rank_impurity'], comparison['rank_permutation'])
    corr_imp_shap, _ = spearmanr(comparison['rank_impurity'], comparison['rank_shap'])
    corr_perm_shap, _ = spearmanr(comparison['rank_permutation'], comparison['rank_shap'])

    print("\nRank Correlations:")
    print(f"  Impurity vs Permutation: {corr_imp_perm:.3f}")
    print(f"  Impurity vs SHAP: {corr_imp_shap:.3f}")
    print(f"  Permutation vs SHAP: {corr_perm_shap:.3f}")

    return comparison


def analyze_correlation_effects(data: Dict[str, Any], importance: pd.DataFrame,
                                threshold: float = 0.7) -> Dict[str, Any]:
    """Analyze how feature correlations affect importance."""
    print("\n=== Correlation Analysis ===")

    df = pd.DataFrame(data['X'], columns=data['feature_names'])
    corr_matrix = df.corr().abs()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(data['feature_names'])):
        for j in range(i+1, len(data['feature_names'])):
            if corr_matrix.iloc[i, j] > threshold:
                f1, f2 = data['feature_names'][i], data['feature_names'][j]
                high_corr_pairs.append({
                    'feature_1': f1,
                    'feature_2': f2,
                    'correlation': corr_matrix.iloc[i, j],
                    'importance_1': importance[importance['feature'] == f1]['importance'].values[0],
                    'importance_2': importance[importance['feature'] == f2]['importance'].values[0]
                })

    if high_corr_pairs:
        print(f"\nHighly correlated feature pairs (|r| > {threshold}):")
        corr_df = pd.DataFrame(high_corr_pairs)
        corr_df = corr_df.sort_values('correlation', ascending=False)
        print(corr_df.to_string(index=False))

        print("\nWarning: Importance may be split among correlated features.")
        print("Consider grouped permutation importance or SHAP interaction values.")
    else:
        print(f"\nNo feature pairs with |r| > {threshold} found.")

    return {
        'correlation_matrix': corr_matrix,
        'high_corr_pairs': high_corr_pairs
    }


def generate_ddml_diagnostics(data: Dict[str, Any], treatment_col: Optional[str] = None) -> None:
    """Generate DDML-specific diagnostics for feature importance."""
    if treatment_col is None:
        print("\n=== DDML Diagnostics ===")
        print("Provide --treatment to analyze potential confounders.")
        return

    print(f"\n=== DDML Diagnostics (Treatment: {treatment_col}) ===")

    # Check if model predicts outcome or treatment
    # Suggest running importance on both models

    print("\nFor DDML analysis, compare feature importance between:")
    print("  1. Outcome model: E[Y|X]")
    print("  2. Propensity model: E[D|X]")
    print("\nFeatures important in BOTH models are potential confounders.")
    print("These should be included in your control set.")


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save importance analysis results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save importance DataFrames
    for name, df in results.items():
        if isinstance(df, pd.DataFrame):
            csv_path = output_path / f'{name}.csv'
            df.to_csv(csv_path, index=False)
            print(f"Saved {name} to: {csv_path}")

    # Save summary report
    report = {
        'analysis_type': 'feature_importance',
        'methods_used': list(results.keys()),
        'n_features': len(results.get('impurity', pd.DataFrame())
                         ) if 'impurity' in results else 0
    }

    if 'comparison' in results:
        top_features = results['comparison'].head(10)['feature'].tolist()
        report['top_10_features'] = top_features

    report_path = output_path / 'importance_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to: {report_path}")


def plot_importance_comparison(comparison: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """Create visualization comparing importance methods."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available for plotting")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    top_n = 15
    top_features = comparison.head(top_n)['feature'].tolist()
    plot_df = comparison[comparison['feature'].isin(top_features)].copy()

    # Plot 1: Normalized importance comparison
    methods = ['impurity_norm', 'permutation_norm', 'shap_norm']
    colors = ['#2ecc71', '#3498db', '#9b59b6']

    x = np.arange(len(top_features))
    width = 0.25

    for i, (method, color) in enumerate(zip(methods, colors)):
        values = [plot_df[plot_df['feature'] == f][method].values[0] for f in top_features]
        axes[0].barh(x + i*width, values, width, label=method.replace('_norm', '').title(),
                    color=color, alpha=0.8)

    axes[0].set_yticks(x + width)
    axes[0].set_yticklabels(top_features)
    axes[0].set_xlabel('Normalized Importance')
    axes[0].set_title('Feature Importance Comparison')
    axes[0].legend()
    axes[0].invert_yaxis()

    # Plot 2: Rank comparison heatmap
    rank_cols = ['rank_impurity', 'rank_permutation', 'rank_shap']
    rank_data = plot_df.set_index('feature')[rank_cols]
    rank_data.columns = ['Impurity', 'Permutation', 'SHAP']

    sns.heatmap(rank_data.head(top_n), annot=True, fmt='d', cmap='RdYlGn_r',
                ax=axes[1], cbar_kws={'label': 'Rank'})
    axes[1].set_title('Feature Rankings by Method')

    # Plot 3: Rank agreement
    rank_diff_ip = abs(comparison['rank_impurity'] - comparison['rank_permutation'])
    rank_diff_is = abs(comparison['rank_impurity'] - comparison['rank_shap'])
    rank_diff_ps = abs(comparison['rank_permutation'] - comparison['rank_shap'])

    axes[2].hist([rank_diff_ip, rank_diff_is, rank_diff_ps],
                 label=['Impurity vs Perm', 'Impurity vs SHAP', 'Perm vs SHAP'],
                 bins=20, alpha=0.6)
    axes[2].set_xlabel('Rank Difference')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Rank Disagreement Distribution')
    axes[2].legend()

    plt.tight_layout()

    if output_dir:
        plot_path = Path(output_dir) / 'importance_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {plot_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive feature importance analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with impurity importance
  python analyze_importance.py model.joblib data.csv --target y

  # Full analysis with all methods
  python analyze_importance.py model.joblib data.csv --target y --method all

  # SHAP analysis only
  python analyze_importance.py model.joblib data.csv --target y --method shap

  # Save results and plots
  python analyze_importance.py model.joblib data.csv --target y --method all --output ./importance_results --save-plots
        """
    )

    parser.add_argument('model', type=str, help='Path to saved model (.joblib)')
    parser.add_argument('data', type=str, help='Path to CSV data file')
    parser.add_argument('--target', '-y', type=str, required=True,
                        help='Target variable column name')
    parser.add_argument('--features', '-X', type=str, default=None,
                        help='Comma-separated feature column names')
    parser.add_argument('--method', '-m', type=str, default='all',
                        choices=['impurity', 'permutation', 'shap', 'all'],
                        help='Importance method to use (default: all)')
    parser.add_argument('--n-repeats', type=int, default=30,
                        help='Number of permutation repeats (default: 30)')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='Max samples for SHAP analysis (default: 1000)')
    parser.add_argument('--corr-threshold', type=float, default=0.7,
                        help='Correlation threshold for analysis (default: 0.7)')
    parser.add_argument('--treatment', '-t', type=str, default=None,
                        help='Treatment variable for DDML diagnostics')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save comparison plots')
    parser.add_argument('--no-plots', action='store_true',
                        help='Suppress plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    if args.quiet:
        warnings.filterwarnings('ignore')

    np.random.seed(args.seed)

    # Load model and data
    data = load_model_and_data(
        args.model,
        args.data,
        args.target,
        args.features
    )

    results = {}

    # Run requested analyses
    if args.method in ['impurity', 'all']:
        results['impurity'] = analyze_impurity_importance(data)

    if args.method in ['permutation', 'all']:
        results['permutation'] = analyze_permutation_importance(
            data, n_repeats=args.n_repeats, random_state=args.seed
        )

    if args.method in ['shap', 'all']:
        shap_result = analyze_shap_importance(
            data, max_samples=args.max_samples, plot=not args.no_plots
        )
        results['shap'] = shap_result['feature_importance']
        results['shap_values'] = shap_result['shap_values']

    # Compare methods if all were computed
    if args.method == 'all':
        results['comparison'] = compare_importance_methods(
            results['impurity'],
            results['permutation'],
            results['shap']
        )

        # Correlation analysis
        analyze_correlation_effects(data, results['permutation'], args.corr_threshold)

        # Plot comparison
        if not args.no_plots:
            plot_importance_comparison(
                results['comparison'],
                output_dir=args.output if args.save_plots else None
            )

    # DDML diagnostics
    generate_ddml_diagnostics(data, args.treatment)

    # Save results
    if args.output:
        save_results(results, args.output)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
