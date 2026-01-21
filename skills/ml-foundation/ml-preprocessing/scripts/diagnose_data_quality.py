#!/usr/bin/env python3
"""
Data Quality Diagnostic Script for Causal Inference

This script provides comprehensive data quality assessment focused on
issues that can impact causal inference validity.

Usage:
    python diagnose_data_quality.py input.csv --treatment D --outcome Y

Features:
    - Missing value analysis (patterns, mechanisms)
    - Outlier detection and characterization
    - Duplicate detection
    - Distribution analysis
    - Treatment/control balance assessment
    - Overlap diagnostics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats

# Add parent directory to path for preprocessing module
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import (
    diagnose_missing,
    detect_outliers
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Diagnose data quality for causal inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic diagnostics
    python diagnose_data_quality.py data.csv --treatment D --outcome Y

    # Full diagnostics with report
    python diagnose_data_quality.py data.csv --treatment D --outcome Y \\
        --controls X1 X2 X3 \\
        --report-path quality_report.json \\
        --verbose

    # Check specific issues
    python diagnose_data_quality.py data.csv --treatment D --outcome Y \\
        --check-missing --check-outliers --check-balance
        """
    )

    parser.add_argument(
        'input',
        type=str,
        help='Path to input CSV file'
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
        '--check-missing',
        action='store_true',
        default=True,
        help='Check missing value patterns'
    )

    parser.add_argument(
        '--check-outliers',
        action='store_true',
        default=True,
        help='Check for outliers'
    )

    parser.add_argument(
        '--check-duplicates',
        action='store_true',
        default=True,
        help='Check for duplicate observations'
    )

    parser.add_argument(
        '--check-balance',
        action='store_true',
        default=True,
        help='Check covariate balance between treatment groups'
    )

    parser.add_argument(
        '--check-distributions',
        action='store_true',
        default=True,
        help='Check variable distributions'
    )

    parser.add_argument(
        '--report-path',
        type=str,
        default=None,
        help='Path to save diagnostic report (JSON format)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )

    return parser.parse_args()


def diagnose_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Diagnose duplicate observations."""
    n_total = len(df)

    # Exact duplicates
    exact_duplicates = df.duplicated(keep=False)
    n_exact = exact_duplicates.sum()

    # Count unique duplicates
    n_unique_duplicates = df[exact_duplicates].drop_duplicates().shape[0]

    return {
        'n_total': n_total,
        'n_exact_duplicates': int(n_exact),
        'n_unique_duplicated': int(n_unique_duplicates),
        'pct_duplicated': round(n_exact / n_total * 100, 2) if n_total > 0 else 0,
        'has_duplicates': n_exact > 0
    }


def diagnose_distributions(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """Diagnose variable distributions."""
    distributions = []

    for col in columns:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            values = df[col].dropna()

            if len(values) < 3:
                distributions.append({
                    'variable': col,
                    'type': 'numeric',
                    'error': 'Insufficient data'
                })
                continue

            # Basic stats
            dist_info = {
                'variable': col,
                'type': 'numeric',
                'n': int(len(values)),
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'median': float(values.median()),
                'q25': float(values.quantile(0.25)),
                'q75': float(values.quantile(0.75)),
            }

            # Skewness and kurtosis
            if len(values) >= 8:
                dist_info['skewness'] = float(stats.skew(values))
                dist_info['kurtosis'] = float(stats.kurtosis(values))
                dist_info['is_skewed'] = abs(dist_info['skewness']) > 1

            # Normality test (Shapiro-Wilk for small samples, D'Agostino for large)
            if len(values) >= 8:
                if len(values) <= 5000:
                    try:
                        stat, pvalue = stats.shapiro(values.sample(min(5000, len(values))))
                        dist_info['normality_pvalue'] = float(pvalue)
                        dist_info['is_normal'] = pvalue > 0.05
                    except:
                        dist_info['normality_pvalue'] = None
                        dist_info['is_normal'] = None
                else:
                    try:
                        stat, pvalue = stats.normaltest(values)
                        dist_info['normality_pvalue'] = float(pvalue)
                        dist_info['is_normal'] = pvalue > 0.05
                    except:
                        dist_info['normality_pvalue'] = None
                        dist_info['is_normal'] = None

            distributions.append(dist_info)

        else:
            # Categorical variable
            value_counts = df[col].value_counts()
            distributions.append({
                'variable': col,
                'type': 'categorical',
                'n': int(len(df[col].dropna())),
                'n_unique': int(df[col].nunique()),
                'top_values': value_counts.head(5).to_dict(),
                'has_rare_categories': (value_counts < len(df) * 0.01).any()
            })

    return {'distributions': distributions}


def diagnose_balance(
    df: pd.DataFrame,
    treatment: str,
    covariates: List[str]
) -> Dict[str, Any]:
    """Diagnose covariate balance between treatment groups."""
    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]

    balance_stats = []
    imbalanced_vars = []

    for cov in covariates:
        if df[cov].dtype in [np.float64, np.int64, float, int]:
            mean_t = treated[cov].mean()
            mean_c = control[cov].mean()

            var_t = treated[cov].var()
            var_c = control[cov].var()

            pooled_std = np.sqrt((var_t + var_c) / 2)
            smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0

            is_balanced = abs(smd) < 0.1
            if not is_balanced:
                imbalanced_vars.append(cov)

            balance_stats.append({
                'variable': cov,
                'type': 'continuous',
                'mean_treated': float(mean_t),
                'mean_control': float(mean_c),
                'std_treated': float(np.sqrt(var_t)),
                'std_control': float(np.sqrt(var_c)),
                'std_mean_diff': float(smd),
                'is_balanced': is_balanced
            })

        else:
            # Categorical: compare proportions
            prop_t = treated[cov].value_counts(normalize=True)
            prop_c = control[cov].value_counts(normalize=True)

            max_diff = 0
            for cat in set(prop_t.index) | set(prop_c.index):
                diff = abs(prop_t.get(cat, 0) - prop_c.get(cat, 0))
                max_diff = max(max_diff, diff)

            is_balanced = max_diff < 0.05
            if not is_balanced:
                imbalanced_vars.append(cov)

            balance_stats.append({
                'variable': cov,
                'type': 'categorical',
                'max_prop_diff': float(max_diff),
                'is_balanced': is_balanced
            })

    return {
        'n_treated': int(len(treated)),
        'n_control': int(len(control)),
        'treatment_rate': float(len(treated) / (len(treated) + len(control))),
        'balance_stats': balance_stats,
        'n_imbalanced': len(imbalanced_vars),
        'imbalanced_variables': imbalanced_vars,
        'all_balanced': len(imbalanced_vars) == 0
    }


def diagnose_outliers(
    df: pd.DataFrame,
    columns: List[str],
    methods: List[str] = ['iqr', 'zscore']
) -> Dict[str, Any]:
    """Diagnose outliers using multiple methods."""
    outlier_results = {}

    numeric_cols = [c for c in columns if df[c].dtype in [np.float64, np.int64, float, int]]

    for method in methods:
        try:
            outlier_mask = detect_outliers(df, numeric_cols, method=method)
            outlier_results[method] = {
                'n_outliers': int(outlier_mask.sum()),
                'pct_outliers': round(outlier_mask.mean() * 100, 2)
            }

            # Per-column outliers
            per_column = {}
            for col in numeric_cols:
                col_outliers = detect_outliers(df, [col], method=method)
                per_column[col] = {
                    'n_outliers': int(col_outliers.sum()),
                    'pct_outliers': round(col_outliers.mean() * 100, 2)
                }
            outlier_results[method]['by_column'] = per_column

        except Exception as e:
            outlier_results[method] = {'error': str(e)}

    return outlier_results


def diagnose_missing_by_treatment(
    df: pd.DataFrame,
    treatment: str,
    columns: List[str]
) -> Dict[str, Any]:
    """Check if missingness differs by treatment status."""
    results = []

    for col in columns:
        if col == treatment:
            continue

        missing_treated = df[df[treatment] == 1][col].isnull().mean()
        missing_control = df[df[treatment] == 0][col].isnull().mean()

        diff = missing_treated - missing_control

        results.append({
            'variable': col,
            'missing_rate_treated': float(missing_treated),
            'missing_rate_control': float(missing_control),
            'difference': float(diff),
            'is_differential': abs(diff) > 0.05
        })

    differential_vars = [r['variable'] for r in results if r['is_differential']]

    return {
        'by_variable': results,
        'has_differential_missingness': len(differential_vars) > 0,
        'differential_variables': differential_vars
    }


def generate_recommendations(report: Dict[str, Any]) -> List[str]:
    """Generate preprocessing recommendations based on diagnostics."""
    recommendations = []

    # Missing data recommendations
    if 'missing' in report:
        missing_pct = report['missing']['summary']['overall_missing_pct']
        if missing_pct > 20:
            recommendations.append(
                f"HIGH missing data ({missing_pct:.1f}%): Consider multiple imputation or "
                "sensitivity analysis for missing data mechanism"
            )
        elif missing_pct > 5:
            recommendations.append(
                f"MODERATE missing data ({missing_pct:.1f}%): Consider imputation; "
                "create missing indicators if missingness may be informative"
            )

    # Differential missingness
    if 'missing_by_treatment' in report:
        if report['missing_by_treatment']['has_differential_missingness']:
            vars = report['missing_by_treatment']['differential_variables']
            recommendations.append(
                f"WARNING: Differential missingness by treatment for: {vars}. "
                "This may indicate selection bias."
            )

    # Balance recommendations
    if 'balance' in report:
        if not report['balance']['all_balanced']:
            vars = report['balance']['imbalanced_variables']
            recommendations.append(
                f"IMBALANCED covariates: {vars}. "
                "Consider propensity score weighting or matching."
            )

    # Outlier recommendations
    if 'outliers' in report:
        for method, results in report['outliers'].items():
            if 'pct_outliers' in results and results['pct_outliers'] > 5:
                recommendations.append(
                    f"HIGH outlier rate ({method}): {results['pct_outliers']:.1f}%. "
                    "Consider robust methods or sensitivity analysis."
                )

    # Duplicate recommendations
    if 'duplicates' in report:
        if report['duplicates']['has_duplicates']:
            recommendations.append(
                f"DUPLICATES detected: {report['duplicates']['n_exact_duplicates']} rows. "
                "Investigate and consider deduplication."
            )

    # Distribution recommendations
    if 'distributions' in report:
        skewed_vars = [
            d['variable'] for d in report['distributions']['distributions']
            if d.get('is_skewed', False)
        ]
        if skewed_vars:
            recommendations.append(
                f"SKEWED distributions: {skewed_vars}. "
                "Consider log transformation or robust methods."
            )

    return recommendations


def print_summary(report: Dict[str, Any], verbose: bool = False):
    """Print diagnostic summary to console."""
    print("\n" + "=" * 60)
    print("DATA QUALITY DIAGNOSTIC REPORT")
    print("=" * 60)

    print(f"\nDataset: {report.get('input_file', 'Unknown')}")
    print(f"Rows: {report.get('n_rows', 'Unknown')}")
    print(f"Columns: {report.get('n_columns', 'Unknown')}")

    # Missing data
    if 'missing' in report:
        print("\n--- MISSING DATA ---")
        print(f"Overall missing: {report['missing']['summary']['overall_missing_pct']:.1f}%")
        print(f"Columns with missing: {report['missing']['summary']['columns_with_missing']}")
        print(f"Complete cases: {report['missing']['summary']['complete_cases']}")

        if verbose and report['missing'].get('by_column') is not None:
            print("\nMissing by column:")
            by_col = report['missing']['by_column']
            if isinstance(by_col, pd.DataFrame):
                for idx, row in by_col.iterrows():
                    if row['missing_pct'] > 0:
                        print(f"  {idx}: {row['missing_pct']:.1f}%")

    # Differential missingness
    if 'missing_by_treatment' in report:
        if report['missing_by_treatment']['has_differential_missingness']:
            print("\n--- DIFFERENTIAL MISSINGNESS WARNING ---")
            for var in report['missing_by_treatment']['differential_variables']:
                print(f"  - {var}")

    # Duplicates
    if 'duplicates' in report:
        print("\n--- DUPLICATES ---")
        print(f"Exact duplicates: {report['duplicates']['n_exact_duplicates']}")

    # Balance
    if 'balance' in report:
        print("\n--- COVARIATE BALANCE ---")
        print(f"Treatment rate: {report['balance']['treatment_rate']:.1%}")
        print(f"N treated: {report['balance']['n_treated']}")
        print(f"N control: {report['balance']['n_control']}")
        print(f"Imbalanced variables: {report['balance']['n_imbalanced']}")

        if verbose and report['balance']['imbalanced_variables']:
            print("\nImbalanced (|SMD| > 0.1):")
            for stat in report['balance']['balance_stats']:
                if not stat.get('is_balanced', True):
                    if stat['type'] == 'continuous':
                        print(f"  {stat['variable']}: SMD = {stat['std_mean_diff']:.3f}")
                    else:
                        print(f"  {stat['variable']}: max prop diff = {stat['max_prop_diff']:.3f}")

    # Outliers
    if 'outliers' in report:
        print("\n--- OUTLIERS ---")
        for method, results in report['outliers'].items():
            if 'pct_outliers' in results:
                print(f"{method}: {results['pct_outliers']:.1f}% ({results['n_outliers']} rows)")

    # Recommendations
    if 'recommendations' in report and report['recommendations']:
        print("\n--- RECOMMENDATIONS ---")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")

    print("\n" + "=" * 60)


def main():
    """Main diagnostic pipeline."""
    args = parse_args()

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

    # Initialize report
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_file': args.input,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'treatment': args.treatment,
        'outcome': args.outcome,
        'controls': controls
    }

    # Run diagnostics
    if args.check_missing:
        if args.verbose:
            print("Checking missing values...")
        report['missing'] = diagnose_missing(df)
        report['missing_by_treatment'] = diagnose_missing_by_treatment(
            df, args.treatment, controls + [args.outcome]
        )

    if args.check_duplicates:
        if args.verbose:
            print("Checking duplicates...")
        report['duplicates'] = diagnose_duplicates(df)

    if args.check_balance:
        if args.verbose:
            print("Checking covariate balance...")
        report['balance'] = diagnose_balance(df, args.treatment, controls)

    if args.check_outliers:
        if args.verbose:
            print("Checking outliers...")
        report['outliers'] = diagnose_outliers(df, controls)

    if args.check_distributions:
        if args.verbose:
            print("Checking distributions...")
        report['distributions'] = diagnose_distributions(df, controls + [args.outcome])

    # Generate recommendations
    report['recommendations'] = generate_recommendations(report)

    # Print summary
    print_summary(report, args.verbose)

    # Save report
    if args.report_path:
        # Convert any non-serializable objects
        def make_serializable(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj

        serializable_report = json.loads(
            json.dumps(report, default=make_serializable)
        )

        with open(args.report_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        print(f"\nReport saved to {args.report_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
