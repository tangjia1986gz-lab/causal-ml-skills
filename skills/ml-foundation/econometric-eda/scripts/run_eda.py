#!/usr/bin/env python
"""
Complete EDA Pipeline CLI

Run comprehensive exploratory data analysis for econometric research.

Usage:
    python run_eda.py --data data.csv --outcome y --treatment D --covariates x1,x2,x3
    python run_eda.py --data panel.csv --outcome y --treatment D --covariates x1,x2 --panel-id entity --time-var year
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from econometric_eda import EconometricEDA


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run comprehensive EDA for econometric research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic EDA
    python run_eda.py --data data.csv

    # EDA with outcome and treatment
    python run_eda.py --data data.csv --outcome y --treatment D --covariates x1,x2,x3

    # Panel data EDA
    python run_eda.py --data panel.csv --outcome y --treatment D \\
        --covariates x1,x2 --panel-id entity_id --time-var year

    # Export to HTML
    python run_eda.py --data data.csv --output report --format html
        """
    )

    parser.add_argument('--data', '-d', required=True,
                        help='Path to CSV data file')
    parser.add_argument('--outcome', '-y',
                        help='Outcome variable name')
    parser.add_argument('--treatment', '-t',
                        help='Treatment variable name')
    parser.add_argument('--covariates', '-x',
                        help='Comma-separated list of covariate names')
    parser.add_argument('--panel-id',
                        help='Panel entity identifier variable')
    parser.add_argument('--time-var',
                        help='Panel time variable')
    parser.add_argument('--output', '-o', default='eda_report',
                        help='Output file path (without extension)')
    parser.add_argument('--format', '-f', default='markdown',
                        choices=['markdown', 'html', 'json'],
                        help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print verbose output')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    if args.verbose:
        print(f"Loading data from {args.data}...")

    try:
        data = pd.read_csv(args.data)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    if args.verbose:
        print(f"Loaded {len(data)} observations, {len(data.columns)} variables")

    # Parse covariates
    covariates = None
    if args.covariates:
        covariates = [c.strip() for c in args.covariates.split(',')]
        if args.verbose:
            print(f"Covariates: {covariates}")

    # Initialize EDA
    eda = EconometricEDA(data)

    # Run full report
    if args.verbose:
        print("Running EDA analysis...")

    report = eda.full_report(
        outcome_var=args.outcome,
        treatment_var=args.treatment,
        covariates=covariates,
        panel_id=args.panel_id,
        time_var=args.time_var
    )

    # Export report
    if args.verbose:
        print(f"Exporting report to {args.output}.{args.format}...")

    output_path = eda.export_report(report, format=args.format, path=args.output)

    print(f"EDA report saved to: {output_path}")

    # Print summary to console
    print("\n" + "="*60)
    print("EDA SUMMARY")
    print("="*60)

    print(f"\nDataset: {len(data)} obs x {len(data.columns)} vars")

    # Missing data summary
    if 'missing' in report:
        missing = report['missing']
        n_vars_missing = len(missing.get('variables_with_missing', []))
        pct_complete = missing.get('pct_complete_cases', 0)
        print(f"\nMissing Data:")
        print(f"  - Variables with missing: {n_vars_missing}")
        print(f"  - Complete cases: {pct_complete:.1f}%")

    # Outliers summary
    if 'outliers' in report and 'iqr' in report['outliers']:
        iqr_results = report['outliers']['iqr']
        n_vars_with_outliers = sum(1 for v in iqr_results.values() if v.get('n_outliers', 0) > 0)
        print(f"\nOutliers (IQR method):")
        print(f"  - Variables with outliers: {n_vars_with_outliers}")

    # Multicollinearity
    if 'multicollinearity' in report and not report['multicollinearity'].get('error'):
        mc = report['multicollinearity']
        print(f"\nMulticollinearity:")
        print(f"  - Condition number: {mc.get('condition_number', 0):.2f}")
        print(f"  - Variables with VIF > 10: {mc.get('n_high_vif', 0)}")

    # Balance
    if 'balance' in report and isinstance(report['balance'], pd.DataFrame):
        balance = report['balance']
        n_imbalanced = (~balance['balanced']).sum() if 'balanced' in balance.columns else 0
        print(f"\nCovariate Balance:")
        print(f"  - Imbalanced variables (|std_diff| > 0.1): {n_imbalanced}")

    # Panel
    if 'panel' in report:
        panel = report['panel']
        if 'variation' in panel:
            var = panel['variation']
            print(f"\nPanel Structure:")
            print(f"  - Entities: {var.get('n_entities', 'N/A')}")
            print(f"  - Periods: {var.get('n_periods', 'N/A')}")
            print(f"  - Balanced: {'Yes' if var.get('is_balanced') else 'No'}")

        if 'attrition' in panel:
            att = panel['attrition']
            print(f"  - Attrition rate: {att.get('overall_attrition_rate', 0)*100:.1f}%")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
