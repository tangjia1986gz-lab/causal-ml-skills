#!/usr/bin/env python
"""
Missing Value Analysis CLI

Diagnose missing data patterns and mechanisms in econometric data.

Usage:
    python diagnose_missing.py --data data.csv --vars y,x1,x2,x3
    python diagnose_missing.py --data data.csv --target y --predictors x1,x2,x3 --test-mar
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
        description='Analyze missing data patterns and mechanisms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic missing analysis
    python diagnose_missing.py --data data.csv

    # Analyze specific variables
    python diagnose_missing.py --data data.csv --vars y,x1,x2,x3

    # Test for MAR mechanism
    python diagnose_missing.py --data data.csv --target y --predictors x1,x2,x3 --test-mar

    # Test for MCAR
    python diagnose_missing.py --data data.csv --vars y,x1,x2 --test-mcar

    # Save visualizations
    python diagnose_missing.py --data data.csv --output missing_analysis --plot
        """
    )

    parser.add_argument('--data', '-d', required=True,
                        help='Path to CSV data file')
    parser.add_argument('--vars', '-v',
                        help='Comma-separated list of variables to analyze')
    parser.add_argument('--target',
                        help='Target variable for MAR test')
    parser.add_argument('--predictors',
                        help='Comma-separated predictors for MAR test')
    parser.add_argument('--test-mcar', action='store_true',
                        help='Run MCAR test')
    parser.add_argument('--test-mar', action='store_true',
                        help='Run MAR test (requires --target and --predictors)')
    parser.add_argument('--output', '-o',
                        help='Output directory for results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')

    return parser.parse_args()


def print_missing_summary(results: dict, verbose: bool = False):
    """Print formatted missing data summary."""
    print("\n" + "="*60)
    print("MISSING DATA ANALYSIS")
    print("="*60)

    summary = results.get('summary')
    if summary is not None and isinstance(summary, pd.DataFrame):
        print("\n## Summary by Variable\n")

        # Filter to variables with missing values
        missing_vars = summary[summary['missing_count'] > 0]

        if len(missing_vars) == 0:
            print("No missing values detected!")
        else:
            print(f"Variables with missing data: {len(missing_vars)}")
            print()

            # Print table
            cols = ['variable', 'missing_count', 'missing_pct', 'complete_count']
            print(missing_vars[cols].to_string(index=False))

    # Complete cases
    print(f"\n## Complete Cases")
    print(f"  - Complete observations: {results.get('n_complete_cases', 'N/A')}")
    print(f"  - Completeness rate: {results.get('pct_complete_cases', 0):.1f}%")

    # Missing patterns
    if verbose and 'patterns' in results:
        patterns = results['patterns']
        if isinstance(patterns, pd.DataFrame) and len(patterns) > 0:
            print(f"\n## Missing Patterns")
            print(f"  - Unique patterns: {len(patterns)}")
            print("\n  Top 5 patterns:")
            top_patterns = patterns.head(5)
            for _, row in top_patterns.iterrows():
                pattern_str = ' | '.join([f"{col}: {'M' if row[col] else 'O'}"
                                          for col in patterns.columns if col not in ['count', 'pct']])
                print(f"    {pattern_str} - {row['count']} obs ({row['pct']:.1f}%)")


def print_mcar_results(results: dict):
    """Print MCAR test results."""
    print("\n## MCAR Test (Little's Test Approximation)")
    print(f"  - Chi-square statistic: {results.get('chi2_statistic', 'N/A'):.3f}")
    print(f"  - Degrees of freedom: {results.get('df', 'N/A')}")
    print(f"  - P-value: {results.get('p_value', 'N/A'):.4f}")
    print(f"  - Conclusion: {results.get('interpretation', 'N/A')}")

    if results.get('is_mcar'):
        print("\n  Interpretation: Data appears to be Missing Completely at Random.")
        print("  Complete case analysis may be valid but less efficient than MI.")
    else:
        print("\n  Interpretation: MCAR assumption is rejected.")
        print("  Consider MAR/MNAR mechanisms. Use multiple imputation or IPW.")


def print_mar_results(results: dict, target: str):
    """Print MAR test results."""
    print(f"\n## MAR Test for {target}")
    print("  Testing whether missingness is related to observed predictors\n")

    for pred, res in results.items():
        sig_marker = "*" if res.get('significant') else ""
        print(f"  {pred}:")
        print(f"    - Test: {res.get('test', 'N/A')}")
        print(f"    - Statistic: {res.get('statistic', 'N/A'):.3f}")
        print(f"    - P-value: {res.get('p_value', 'N/A'):.4f} {sig_marker}")

        if res.get('test') == 't-test':
            print(f"    - Mean (missing group): {res.get('mean_missing_group', 'N/A'):.3f}")
            print(f"    - Mean (observed group): {res.get('mean_observed_group', 'N/A'):.3f}")
        print()

    # Summary
    n_significant = sum(1 for r in results.values() if r.get('significant'))
    if n_significant > 0:
        print(f"  Summary: {n_significant} predictor(s) significantly related to missingness")
        print("  This suggests MAR mechanism. Consider multiple imputation.")
    else:
        print("  Summary: No predictors significantly related to missingness")
        print("  This is consistent with MCAR (but does not rule out MNAR).")


def create_missing_visualizations(data: pd.DataFrame, variables: list, output_dir: str):
    """Create missing data visualizations."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not available. Skipping visualizations.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    subset = data[variables]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Bar chart of missing percentages
    missing_pct = (subset.isna().sum() / len(subset)) * 100
    missing_pct = missing_pct.sort_values(ascending=True)
    missing_pct.plot(kind='barh', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_xlabel('Percentage Missing')
    axes[0, 0].set_title('Missing Values by Variable')
    axes[0, 0].axvline(x=5, color='orange', linestyle='--', label='5% threshold')
    axes[0, 0].legend()

    # 2. Missing matrix
    missing_matrix = subset.isna().astype(int)
    # Show sample if too many rows
    if len(missing_matrix) > 200:
        sample_idx = np.random.choice(len(missing_matrix), 200, replace=False)
        sample_idx = np.sort(sample_idx)
        missing_matrix_plot = missing_matrix.iloc[sample_idx]
    else:
        missing_matrix_plot = missing_matrix

    sns.heatmap(missing_matrix_plot.T, cmap='RdYlBu_r', ax=axes[0, 1],
                cbar_kws={'label': 'Missing'}, yticklabels=True)
    axes[0, 1].set_xlabel('Observation')
    axes[0, 1].set_title('Missing Value Matrix (sample)')

    # 3. Missing correlations
    missing_corr = subset.isna().corr()
    mask = np.triu(np.ones_like(missing_corr, dtype=bool), k=1)
    sns.heatmap(missing_corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, ax=axes[1, 0],
                vmin=-1, vmax=1)
    axes[1, 0].set_title('Missing Value Correlations')

    # 4. Completeness by observation
    completeness = 1 - subset.isna().mean(axis=1)
    axes[1, 1].hist(completeness, bins=20, edgecolor='black', color='steelblue')
    axes[1, 1].axvline(x=completeness.mean(), color='red', linestyle='--',
                       label=f'Mean: {completeness.mean():.2f}')
    axes[1, 1].set_xlabel('Completeness Rate')
    axes[1, 1].set_ylabel('Number of Observations')
    axes[1, 1].set_title('Distribution of Observation Completeness')
    axes[1, 1].legend()

    plt.tight_layout()
    fig_path = output_path / 'missing_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations saved to: {fig_path}")


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

    # Parse variables
    if args.vars:
        variables = [v.strip() for v in args.vars.split(',')]
    else:
        variables = data.columns.tolist()

    # Initialize EDA
    eda = EconometricEDA(data)

    # Basic missing analysis
    results = eda.check_missing(variables)
    print_missing_summary(results, args.verbose)

    # MCAR test
    if args.test_mcar:
        numeric_vars = [v for v in variables if data[v].dtype in ['float64', 'int64']]
        if len(numeric_vars) > 1:
            mcar_results = eda.test_mcar(numeric_vars)
            print_mcar_results(mcar_results)
        else:
            print("\nMCAR test requires at least 2 numeric variables.")

    # MAR test
    if args.test_mar:
        if not args.target or not args.predictors:
            print("\nMAR test requires --target and --predictors arguments.")
        else:
            predictors = [p.strip() for p in args.predictors.split(',')]
            mar_results = eda.test_mar(args.target, predictors)
            print_mar_results(mar_results, args.target)

    # Visualizations
    if args.plot and args.output:
        create_missing_visualizations(data, variables, args.output)
    elif args.plot:
        print("\nWarning: --plot requires --output directory. Skipping visualizations.")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary as CSV
        summary_path = output_path / 'missing_summary.csv'
        results['summary'].to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

        # Save patterns
        if 'patterns' in results:
            patterns_path = output_path / 'missing_patterns.csv'
            results['patterns'].to_csv(patterns_path, index=False)
            print(f"Patterns saved to: {patterns_path}")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
