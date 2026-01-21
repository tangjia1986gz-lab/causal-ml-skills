#!/usr/bin/env python3
"""
Comprehensive Statistical Testing CLI

Run various statistical tests from the command line for econometric research.

Usage:
    python run_statistical_tests.py --data data.csv --outcome y --group treatment --test ttest
    python run_statistical_tests.py --data data.csv --outcome y --groups education --test anova
    python run_statistical_tests.py --data data.csv --var1 treatment --var2 success --test chi_squared
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from statistical_analysis import (
    run_ttest,
    run_welch_test,
    run_ttest_paired,
    run_ttest_one_sample,
    run_mann_whitney,
    run_anova,
    run_welch_anova,
    run_kruskal,
    run_chi_squared,
    run_fisher_exact,
    check_normality,
    check_homogeneity,
    StatisticalResult
)


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


def run_two_sample_test(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    test: str,
    alternative: str = 'two-sided',
    equal_var: bool = True
) -> StatisticalResult:
    """Run two-sample comparison test."""

    if test == 'ttest':
        if equal_var:
            return run_ttest(df, outcome, group, alternative=alternative, equal_var=True)
        else:
            return run_welch_test(df, outcome, group, alternative=alternative)

    elif test == 'welch':
        return run_welch_test(df, outcome, group, alternative=alternative)

    elif test == 'mann_whitney':
        return run_mann_whitney(df, outcome, group, alternative=alternative)

    else:
        raise ValueError(f"Unknown two-sample test: {test}")


def run_anova_test(
    df: pd.DataFrame,
    outcome: str,
    groups: str,
    test: str
) -> StatisticalResult:
    """Run ANOVA or related test."""

    if test == 'anova':
        return run_anova(df, outcome, groups)

    elif test == 'welch_anova':
        return run_welch_anova(df, outcome, groups)

    elif test == 'kruskal':
        return run_kruskal(df, outcome, groups)

    else:
        raise ValueError(f"Unknown ANOVA test: {test}")


def run_categorical_test(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    test: str
) -> StatisticalResult:
    """Run test for categorical variables."""

    if test == 'chi_squared':
        return run_chi_squared(df, var1, var2)

    elif test == 'fisher':
        return run_fisher_exact(df, var1, var2)

    else:
        raise ValueError(f"Unknown categorical test: {test}")


def check_assumptions_for_test(
    df: pd.DataFrame,
    outcome: str,
    group: Optional[str] = None
) -> dict:
    """Check assumptions for parametric tests."""

    results = {}

    # Normality test
    if group:
        for g in df[group].unique():
            subset = df[df[group] == g][outcome].dropna()
            results[f'normality_{g}'] = check_normality(subset)

        # Homogeneity of variance
        results['homogeneity'] = check_homogeneity(df, outcome, group)
    else:
        results['normality'] = check_normality(df[outcome].dropna())

    return results


def format_result(result: StatisticalResult, verbose: bool = True) -> str:
    """Format result for output."""

    lines = []
    lines.append("=" * 60)
    lines.append(f"TEST: {result.test_name}")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Test Statistic: {result.statistic:.4f}")
    lines.append(f"P-value: {result.p_value:.6f}")

    if hasattr(result, 'df') and result.df is not None:
        lines.append(f"Degrees of Freedom: {result.df}")

    lines.append("")

    # Significance stars
    if result.p_value < 0.001:
        sig = "*** (p < 0.001)"
    elif result.p_value < 0.01:
        sig = "** (p < 0.01)"
    elif result.p_value < 0.05:
        sig = "* (p < 0.05)"
    elif result.p_value < 0.10:
        sig = ". (p < 0.10)"
    else:
        sig = "not significant"

    lines.append(f"Significance: {sig}")

    # Effect size
    if hasattr(result, 'effect_size') and result.effect_size is not None:
        lines.append("")
        lines.append(f"Effect Size: {result.effect_size:.4f}")
        if hasattr(result, 'effect_size_name'):
            lines.append(f"  ({result.effect_size_name})")
        if hasattr(result, 'effect_ci') and result.effect_ci is not None:
            lines.append(f"  95% CI: [{result.effect_ci[0]:.4f}, {result.effect_ci[1]:.4f}]")

    # Group statistics
    if hasattr(result, 'group_stats') and result.group_stats:
        lines.append("")
        lines.append("Group Statistics:")
        for group, stats in result.group_stats.items():
            lines.append(f"  {group}:")
            lines.append(f"    N: {stats.get('n', 'N/A')}")
            lines.append(f"    Mean: {stats.get('mean', 'N/A'):.4f}" if isinstance(stats.get('mean'), (int, float)) else f"    Mean: {stats.get('mean', 'N/A')}")
            lines.append(f"    SD: {stats.get('std', 'N/A'):.4f}" if isinstance(stats.get('std'), (int, float)) else f"    SD: {stats.get('std', 'N/A')}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("INTERPRETATION:")
    lines.append(result.interpretation)
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Run statistical tests on data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Two-sample t-test
  python run_statistical_tests.py --data data.csv --outcome income --group treatment --test ttest

  # Welch's t-test (unequal variances)
  python run_statistical_tests.py --data data.csv --outcome income --group treatment --test welch

  # Mann-Whitney U (non-parametric)
  python run_statistical_tests.py --data data.csv --outcome satisfaction --group treatment --test mann_whitney

  # One-way ANOVA
  python run_statistical_tests.py --data data.csv --outcome income --groups education_level --test anova

  # Chi-squared test
  python run_statistical_tests.py --data data.csv --var1 treatment --var2 success --test chi_squared

  # One-sided test
  python run_statistical_tests.py --data data.csv --outcome income --group treatment --test ttest --alternative greater
        """
    )

    # Data input
    parser.add_argument('--data', required=True, help='Path to data file (CSV, Excel, Stata, Parquet)')

    # Variable specification
    parser.add_argument('--outcome', help='Outcome/dependent variable (for continuous tests)')
    parser.add_argument('--group', help='Grouping variable for two-sample tests')
    parser.add_argument('--groups', help='Grouping variable for ANOVA (more than 2 groups)')
    parser.add_argument('--var1', help='First variable (for categorical tests)')
    parser.add_argument('--var2', help='Second variable (for categorical tests)')
    parser.add_argument('--hypothesized-mean', type=float, help='Hypothesized mean for one-sample test')

    # Test specification
    parser.add_argument('--test', required=True,
                       choices=['ttest', 'welch', 'mann_whitney', 'ttest_paired', 'ttest_one_sample',
                               'anova', 'welch_anova', 'kruskal', 'chi_squared', 'fisher'],
                       help='Statistical test to run')

    # Test options
    parser.add_argument('--alternative', default='two-sided',
                       choices=['two-sided', 'greater', 'less'],
                       help='Alternative hypothesis (default: two-sided)')
    parser.add_argument('--equal-var', action='store_true',
                       help='Assume equal variances (for t-test)')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')

    # Output options
    parser.add_argument('--check-assumptions', action='store_true',
                       help='Check test assumptions before running')
    parser.add_argument('--output', help='Output file path (default: stdout)')
    parser.add_argument('--format', choices=['text', 'json', 'csv'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--quiet', action='store_true',
                       help='Only output essential information')

    args = parser.parse_args()

    # Load data
    try:
        df = load_data(args.data)
        if not args.quiet:
            print(f"Loaded {len(df)} observations from {args.data}")
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    # Check assumptions if requested
    if args.check_assumptions and args.outcome:
        if not args.quiet:
            print("\nChecking assumptions...")
        assumption_results = check_assumptions_for_test(
            df, args.outcome, args.group or args.groups
        )
        for name, result in assumption_results.items():
            print(f"  {name}: {'PASSED' if result.passed else 'FAILED'} (p = {result.p_value:.4f})")
        print()

    # Run the appropriate test
    try:
        if args.test in ['ttest', 'welch', 'mann_whitney']:
            if not args.outcome or not args.group:
                print("Error: Two-sample tests require --outcome and --group", file=sys.stderr)
                sys.exit(1)
            result = run_two_sample_test(
                df, args.outcome, args.group, args.test,
                alternative=args.alternative, equal_var=args.equal_var
            )

        elif args.test == 'ttest_one_sample':
            if not args.outcome or args.hypothesized_mean is None:
                print("Error: One-sample t-test requires --outcome and --hypothesized-mean", file=sys.stderr)
                sys.exit(1)
            result = run_ttest_one_sample(
                df[args.outcome], args.hypothesized_mean,
                alternative=args.alternative
            )

        elif args.test == 'ttest_paired':
            if not args.var1 or not args.var2:
                print("Error: Paired t-test requires --var1 and --var2", file=sys.stderr)
                sys.exit(1)
            result = run_ttest_paired(
                df, args.var1, args.var2,
                alternative=args.alternative
            )

        elif args.test in ['anova', 'welch_anova', 'kruskal']:
            if not args.outcome or not args.groups:
                print("Error: ANOVA tests require --outcome and --groups", file=sys.stderr)
                sys.exit(1)
            result = run_anova_test(df, args.outcome, args.groups, args.test)

        elif args.test in ['chi_squared', 'fisher']:
            if not args.var1 or not args.var2:
                print("Error: Categorical tests require --var1 and --var2", file=sys.stderr)
                sys.exit(1)
            result = run_categorical_test(df, args.var1, args.var2, args.test)

        else:
            print(f"Error: Unknown test {args.test}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error running test: {e}", file=sys.stderr)
        sys.exit(1)

    # Format and output results
    if args.format == 'text':
        output = format_result(result, verbose=not args.quiet)
    elif args.format == 'json':
        import json
        output = json.dumps(result.to_dict(), indent=2)
    elif args.format == 'csv':
        output = f"test,statistic,p_value,effect_size\n{result.test_name},{result.statistic},{result.p_value},{result.effect_size or ''}"

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()
