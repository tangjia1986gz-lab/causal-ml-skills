#!/usr/bin/env python3
"""
Complete Preprocessing Pipeline CLI for Causal Inference

This script provides a command-line interface for preprocessing data
specifically designed for causal inference workflows.

Usage:
    python run_preprocessing.py input.csv --output processed.csv \
        --outcome Y --treatment D --controls X1 X2 X3 \
        --missing-strategy mean --outlier-method iqr

Features:
    - Missing value handling (drop, mean, median, multiple imputation)
    - Outlier detection and removal
    - Standardization of control variables
    - Missing indicators for informative missingness
    - Comprehensive preprocessing report
    - Overlap and balance diagnostics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np

# Add parent directory to path for preprocessing module
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import (
    diagnose_missing,
    handle_missing,
    detect_outliers,
    remove_outliers,
    standardize,
    preprocess_for_causal,
    validate_preprocessing
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess data for causal inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic preprocessing with defaults
    python run_preprocessing.py data.csv -o clean.csv --outcome Y --treatment D

    # Full preprocessing with all options
    python run_preprocessing.py data.csv -o clean.csv \\
        --outcome Y --treatment D \\
        --controls X1 X2 X3 age income \\
        --missing-strategy mean \\
        --outlier-method iqr --outlier-threshold 1.5 \\
        --standardize \\
        --missing-indicators \\
        --report-path report.json

    # Multiple imputation (creates multiple output files)
    python run_preprocessing.py data.csv -o imputed.csv \\
        --outcome Y --treatment D \\
        --missing-strategy multiple --n-imputations 10
        """
    )

    # Required arguments
    parser.add_argument(
        'input',
        type=str,
        help='Path to input CSV file'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Path to output CSV file'
    )

    parser.add_argument(
        '--outcome',
        type=str,
        required=True,
        help='Name of outcome variable'
    )

    parser.add_argument(
        '--treatment',
        type=str,
        required=True,
        help='Name of treatment variable'
    )

    # Optional arguments
    parser.add_argument(
        '--controls',
        type=str,
        nargs='+',
        default=None,
        help='Names of control variables (default: all numeric except outcome/treatment)'
    )

    parser.add_argument(
        '--missing-strategy',
        type=str,
        choices=['drop', 'mean', 'median', 'mode', 'multiple'],
        default='drop',
        help='Strategy for handling missing values (default: drop)'
    )

    parser.add_argument(
        '--n-imputations',
        type=int,
        default=5,
        help='Number of imputations for multiple imputation (default: 5)'
    )

    parser.add_argument(
        '--outlier-method',
        type=str,
        choices=['none', 'iqr', 'zscore', 'isolation_forest', 'mahalanobis'],
        default='none',
        help='Method for outlier detection (default: none)'
    )

    parser.add_argument(
        '--outlier-threshold',
        type=float,
        default=1.5,
        help='Threshold for outlier detection (default: 1.5 for IQR, 3.0 for zscore)'
    )

    parser.add_argument(
        '--standardize',
        action='store_true',
        help='Standardize control variables to mean=0, std=1'
    )

    parser.add_argument(
        '--missing-indicators',
        action='store_true',
        help='Create binary indicators for missing values before imputation'
    )

    parser.add_argument(
        '--report-path',
        type=str,
        default=None,
        help='Path to save preprocessing report (JSON format)'
    )

    parser.add_argument(
        '--check-overlap',
        action='store_true',
        help='Check propensity score overlap after preprocessing'
    )

    parser.add_argument(
        '--check-balance',
        action='store_true',
        help='Check covariate balance between treatment groups'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )

    return parser.parse_args()


def check_overlap(df: pd.DataFrame, treatment: str, controls: List[str]) -> Dict[str, Any]:
    """Check propensity score overlap between treatment groups."""
    from sklearn.linear_model import LogisticRegression

    # Get numeric controls only
    numeric_controls = df[controls].select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_controls:
        return {'error': 'No numeric controls for overlap check'}

    # Handle any remaining missing values
    df_clean = df[[treatment] + numeric_controls].dropna()

    if len(df_clean) < 10:
        return {'error': 'Insufficient data for overlap check'}

    # Estimate propensity scores
    try:
        ps_model = LogisticRegression(max_iter=1000, solver='lbfgs')
        ps_model.fit(df_clean[numeric_controls], df_clean[treatment])
        ps = ps_model.predict_proba(df_clean[numeric_controls])[:, 1]
    except Exception as e:
        return {'error': f'Propensity score estimation failed: {str(e)}'}

    # Analyze overlap
    ps_treated = ps[df_clean[treatment] == 1]
    ps_control = ps[df_clean[treatment] == 0]

    overlap_lower = max(ps_treated.min(), ps_control.min())
    overlap_upper = min(ps_treated.max(), ps_control.max())

    return {
        'overlap_region': [float(overlap_lower), float(overlap_upper)],
        'treated_ps_range': [float(ps_treated.min()), float(ps_treated.max())],
        'control_ps_range': [float(ps_control.min()), float(ps_control.max())],
        'treated_ps_mean': float(ps_treated.mean()),
        'control_ps_mean': float(ps_control.mean()),
        'has_good_overlap': overlap_upper > overlap_lower
    }


def check_balance(df: pd.DataFrame, treatment: str, covariates: List[str]) -> Dict[str, Any]:
    """Check covariate balance between treatment groups."""
    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]

    balance_stats = []

    for cov in covariates:
        if df[cov].dtype in [np.float64, np.int64, float, int]:
            mean_t = treated[cov].mean()
            mean_c = control[cov].mean()

            var_t = treated[cov].var()
            var_c = control[cov].var()
            pooled_std = np.sqrt((var_t + var_c) / 2)

            smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0

            balance_stats.append({
                'variable': cov,
                'mean_treated': float(mean_t),
                'mean_control': float(mean_c),
                'std_mean_diff': float(smd),
                'is_balanced': abs(smd) < 0.1  # Common threshold
            })

    imbalanced = [s['variable'] for s in balance_stats if not s['is_balanced']]

    return {
        'balance_stats': balance_stats,
        'n_imbalanced': len(imbalanced),
        'imbalanced_variables': imbalanced,
        'all_balanced': len(imbalanced) == 0
    }


def main():
    """Main preprocessing pipeline."""
    args = parse_args()

    # Load data
    if args.verbose:
        print(f"Loading data from {args.input}...")

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    if args.verbose:
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Validate required columns
    if args.outcome not in df.columns:
        print(f"Error: Outcome variable '{args.outcome}' not found in data")
        sys.exit(1)

    if args.treatment not in df.columns:
        print(f"Error: Treatment variable '{args.treatment}' not found in data")
        sys.exit(1)

    # Determine controls
    if args.controls is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        controls = [c for c in numeric_cols if c not in [args.outcome, args.treatment]]
        if args.verbose:
            print(f"Auto-detected {len(controls)} control variables")
    else:
        controls = args.controls
        missing_controls = [c for c in controls if c not in df.columns]
        if missing_controls:
            print(f"Error: Control variables not found: {missing_controls}")
            sys.exit(1)

    if args.verbose:
        print(f"Using controls: {controls}")

    # Initialize report
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_file': args.input,
        'output_file': args.output,
        'outcome': args.outcome,
        'treatment': args.treatment,
        'controls': controls,
        'n_original': len(df),
        'steps': []
    }

    # Step 1: Missing value diagnosis
    if args.verbose:
        print("\nDiagnosing missing values...")

    missing_report = diagnose_missing(df)
    report['missing_diagnosis'] = {
        'total_missing_pct': missing_report['summary']['overall_missing_pct'],
        'columns_with_missing': missing_report['summary']['columns_with_missing'],
        'complete_cases': missing_report['summary']['complete_cases'],
        'recommendations': missing_report['recommendations']
    }

    if args.verbose:
        print(f"  Missing: {missing_report['summary']['overall_missing_pct']:.1f}%")
        print(f"  Complete cases: {missing_report['summary']['complete_cases']}")

    # Run preprocessing
    if args.verbose:
        print("\nRunning preprocessing pipeline...")

    outlier_method = args.outlier_method if args.outlier_method != 'none' else None

    result = preprocess_for_causal(
        df=df,
        outcome=args.outcome,
        treatment=args.treatment,
        controls=controls,
        missing_strategy=args.missing_strategy,
        outlier_method=outlier_method,
        outlier_threshold=args.outlier_threshold,
        standardize_controls=args.standardize,
        create_missing_indicators=args.missing_indicators
    )

    df_processed = result['data']
    report['preprocessing'] = result['report']

    if args.verbose:
        print(f"  Rows after missing handling: {result['report'].get('n_after_missing', len(df_processed))}")
        if outlier_method:
            print(f"  Rows after outlier removal: {result['report'].get('n_after_outliers', len(df_processed))}")
        print(f"  Final rows: {result['report']['n_final']}")
        print(f"  Retained: {result['report']['pct_retained']:.1f}%")

    # Check overlap if requested
    if args.check_overlap:
        if args.verbose:
            print("\nChecking propensity score overlap...")

        overlap_result = check_overlap(df_processed, args.treatment, controls)
        report['overlap'] = overlap_result

        if 'error' not in overlap_result:
            if args.verbose:
                print(f"  Overlap region: [{overlap_result['overlap_region'][0]:.3f}, {overlap_result['overlap_region'][1]:.3f}]")
                print(f"  Good overlap: {overlap_result['has_good_overlap']}")
        else:
            if args.verbose:
                print(f"  {overlap_result['error']}")

    # Check balance if requested
    if args.check_balance:
        if args.verbose:
            print("\nChecking covariate balance...")

        balance_result = check_balance(df_processed, args.treatment, controls)
        report['balance'] = balance_result

        if args.verbose:
            print(f"  Imbalanced variables: {balance_result['n_imbalanced']}")
            if balance_result['imbalanced_variables']:
                print(f"    {balance_result['imbalanced_variables']}")

    # Handle multiple imputation output
    if args.missing_strategy == 'multiple':
        # Regenerate multiple imputed datasets
        imputed_dfs = handle_missing(df, strategy='multiple', n_imputations=args.n_imputations)

        output_path = Path(args.output)
        for i, imp_df in enumerate(imputed_dfs):
            imp_output = output_path.parent / f"{output_path.stem}_imp{i+1}{output_path.suffix}"
            imp_df.to_csv(imp_output, index=False)
            if args.verbose:
                print(f"Saved imputation {i+1} to {imp_output}")

        report['n_imputations'] = args.n_imputations
        report['imputation_files'] = [
            str(output_path.parent / f"{output_path.stem}_imp{i+1}{output_path.suffix}")
            for i in range(args.n_imputations)
        ]
    else:
        # Save single processed dataset
        df_processed.to_csv(args.output, index=False)
        if args.verbose:
            print(f"\nSaved processed data to {args.output}")

    # Save report
    if args.report_path:
        with open(args.report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        if args.verbose:
            print(f"Saved report to {args.report_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"Input:  {args.input} ({report['n_original']} rows)")
    print(f"Output: {args.output} ({report['preprocessing']['n_final']} rows)")
    print(f"Retained: {report['preprocessing']['pct_retained']:.1f}%")

    if report['preprocessing'].get('warnings'):
        print("\nWarnings:")
        for warning in report['preprocessing']['warnings']:
            print(f"  - {warning}")

    if args.check_balance and report.get('balance', {}).get('imbalanced_variables'):
        print("\nImbalanced variables (|SMD| > 0.1):")
        for var in report['balance']['imbalanced_variables']:
            print(f"  - {var}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
