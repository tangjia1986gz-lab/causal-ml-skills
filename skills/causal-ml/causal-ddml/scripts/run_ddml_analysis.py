#!/usr/bin/env python3
"""
Complete DDML Analysis CLI

Runs a full Double/Debiased Machine Learning analysis with:
- Data validation
- Automatic learner selection
- Multiple model specifications
- Sensitivity analysis
- Comprehensive reporting

Usage:
    python run_ddml_analysis.py --data data.csv --outcome y --treatment d --controls "x1,x2,x3"
    python run_ddml_analysis.py --data data.csv --outcome y --treatment d --controls-file controls.txt
    python run_ddml_analysis.py --data data.csv --outcome y --treatment d --all-controls --model auto

Reference:
    Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for
    Treatment and Structural Parameters. The Econometrics Journal.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ddml_estimator import (
    create_ddml_data, validate_ddml_setup, estimate_plr, estimate_irm,
    compare_learners, run_full_ddml_analysis, select_first_stage_learners
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run complete DDML analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python run_ddml_analysis.py --data wages.csv --outcome log_wage --treatment education

    # With specific controls
    python run_ddml_analysis.py --data wages.csv --outcome log_wage --treatment education \\
        --controls "age,age_sq,female,married,region"

    # Use all columns as controls (except outcome and treatment)
    python run_ddml_analysis.py --data wages.csv --outcome log_wage --treatment education \\
        --all-controls

    # Compare specific learners
    python run_ddml_analysis.py --data wages.csv --outcome log_wage --treatment education \\
        --learners "lasso,ridge,random_forest,xgboost"

    # Save results to JSON
    python run_ddml_analysis.py --data wages.csv --outcome log_wage --treatment education \\
        --output results.json
        """
    )

    # Required arguments
    parser.add_argument('--data', '-d', required=True,
                        help='Path to CSV data file')
    parser.add_argument('--outcome', '-y', required=True,
                        help='Name of outcome variable')
    parser.add_argument('--treatment', '-t', required=True,
                        help='Name of treatment variable')

    # Control specification
    controls_group = parser.add_mutually_exclusive_group()
    controls_group.add_argument('--controls', '-x',
                                help='Comma-separated list of control variables')
    controls_group.add_argument('--controls-file',
                                help='File with control variable names (one per line)')
    controls_group.add_argument('--all-controls', action='store_true',
                                help='Use all columns except outcome/treatment as controls')

    # Model specification
    parser.add_argument('--model', '-m', default='auto',
                        choices=['plr', 'irm', 'auto'],
                        help='Model type: plr, irm, or auto (default: auto)')
    parser.add_argument('--learners', '-l',
                        default='lasso,ridge,random_forest,xgboost',
                        help='Comma-separated list of learners to compare')

    # Cross-fitting parameters
    parser.add_argument('--n-folds', '-k', type=int, default=5,
                        help='Number of cross-fitting folds (default: 5)')
    parser.add_argument('--n-rep', '-r', type=int, default=1,
                        help='Number of cross-fitting repetitions (default: 1)')

    # IRM-specific
    parser.add_argument('--trimming', type=float, default=0.01,
                        help='Propensity trimming threshold (default: 0.01)')

    # Output options
    parser.add_argument('--output', '-o',
                        help='Output file path (JSON format)')
    parser.add_argument('--report', action='store_true',
                        help='Generate detailed markdown report')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(path)
    return df


def get_controls(args, data: pd.DataFrame) -> list:
    """Get list of control variables from arguments."""
    if args.controls:
        controls = [c.strip() for c in args.controls.split(',')]
    elif args.controls_file:
        with open(args.controls_file) as f:
            controls = [line.strip() for line in f if line.strip()]
    elif args.all_controls:
        controls = [c for c in data.columns
                    if c not in [args.outcome, args.treatment]]
    else:
        # Default: all columns except outcome and treatment
        controls = [c for c in data.columns
                    if c not in [args.outcome, args.treatment]]

    return controls


def run_analysis(args):
    """Run the complete DDML analysis."""
    print("=" * 70)
    print("DOUBLE/DEBIASED MACHINE LEARNING ANALYSIS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    print("Loading data...")
    data = load_data(args.data)
    print(f"  Data shape: {data.shape}")

    # Get controls
    controls = get_controls(args, data)
    print(f"  Outcome: {args.outcome}")
    print(f"  Treatment: {args.treatment}")
    print(f"  Controls: {len(controls)} variables")
    if args.verbose and len(controls) <= 20:
        print(f"    {controls}")
    print()

    # Validate setup
    print("Validating data...")
    validation = validate_ddml_setup(
        data=data,
        outcome=args.outcome,
        treatment=args.treatment,
        controls=controls,
        n_folds=args.n_folds
    )

    if not validation['is_valid']:
        print("ERROR: Data validation failed!")
        for error in validation['errors']:
            print(f"  - {error}")
        sys.exit(1)

    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    print()

    # Print summary
    print("Data Summary:")
    for key, value in validation['summary'].items():
        print(f"  {key}: {value}")
    print()

    # Parse learners
    learner_list = [l.strip() for l in args.learners.split(',')]

    # Auto-select best learner
    print("Selecting best ML learners via cross-validation...")
    ddml_data = create_ddml_data(data, args.outcome, args.treatment, controls)
    best_learners = select_first_stage_learners(
        ddml_data.X, ddml_data.y, ddml_data.d
    )
    print(f"  Best for E[Y|X]: {best_learners['ml_l']}")
    print(f"  Best for E[D|X]: {best_learners['ml_m']}")
    print()

    # Determine model type
    model = args.model
    if model == 'auto':
        if ddml_data.treatment_type == 'binary':
            model = 'irm'
            print("Auto-selected model: IRM (binary treatment)")
        else:
            model = 'plr'
            print("Auto-selected model: PLR (continuous treatment)")
    print()

    # Run main estimation
    print(f"Running {model.upper()} estimation...")
    print(f"  Folds: {args.n_folds}")
    print(f"  Repetitions: {args.n_rep}")
    print()

    if model == 'plr':
        main_result = estimate_plr(
            data=data,
            outcome=args.outcome,
            treatment=args.treatment,
            controls=controls,
            ml_l=best_learners['ml_l'],
            ml_m=best_learners['ml_m'],
            n_folds=args.n_folds,
            n_rep=args.n_rep,
            random_state=args.seed
        )
    else:  # irm
        main_result = estimate_irm(
            data=data,
            outcome=args.outcome,
            treatment=args.treatment,
            controls=controls,
            ml_g=best_learners['ml_l'],
            ml_m='logistic_lasso' if best_learners['ml_m'] in ['lasso', 'ridge']
                  else best_learners['ml_m'],
            n_folds=args.n_folds,
            n_rep=args.n_rep,
            trimming_threshold=args.trimming,
            random_state=args.seed
        )

    # Print main results
    print("-" * 70)
    print("MAIN RESULTS")
    print("-" * 70)
    print(f"Treatment Effect: {main_result.effect:.6f}")
    print(f"Standard Error:   {main_result.se:.6f}")
    print(f"95% CI:           [{main_result.ci_lower:.6f}, {main_result.ci_upper:.6f}]")
    print(f"P-value:          {main_result.p_value:.6f}")
    significance = "***" if main_result.p_value < 0.01 else "**" if main_result.p_value < 0.05 else "*" if main_result.p_value < 0.1 else ""
    print(f"Significance:     {significance}")
    print()

    # Compare learners
    print("-" * 70)
    print("LEARNER COMPARISON")
    print("-" * 70)
    comparison = compare_learners(
        data=data,
        outcome=args.outcome,
        treatment=args.treatment,
        controls=controls,
        learner_list=learner_list,
        model=model,
        n_folds=args.n_folds,
        random_state=args.seed
    )

    print(comparison.summary_table)
    print()

    # Sensitivity summary
    print("-" * 70)
    print("SENSITIVITY ANALYSIS")
    print("-" * 70)
    sens = comparison.sensitivity
    print(f"Effect Range:            [{sens['min_effect']:.6f}, {sens['max_effect']:.6f}]")
    print(f"Mean Effect:             {sens['mean_effect']:.6f}")
    print(f"Std Dev:                 {sens['std_effect']:.6f}")
    print(f"Coefficient of Variation: {sens['std_effect']/abs(sens['mean_effect'])*100:.2f}%")
    print(f"All Significant (5%):    {sens['all_significant']}")
    print(f"All Same Sign:           {sens['all_same_sign']}")
    print()

    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_file': args.data,
        'outcome': args.outcome,
        'treatment': args.treatment,
        'n_controls': len(controls),
        'model': model,
        'n_folds': args.n_folds,
        'n_rep': args.n_rep,
        'main_result': {
            'effect': main_result.effect,
            'se': main_result.se,
            'ci_lower': main_result.ci_lower,
            'ci_upper': main_result.ci_upper,
            'p_value': main_result.p_value,
            'learner': best_learners['ml_l']
        },
        'diagnostics': main_result.diagnostics,
        'learner_comparison': {
            learner: {
                'effect': r.effect,
                'se': r.se,
                'p_value': r.p_value
            }
            for learner, r in comparison.results.items()
        },
        'sensitivity': {
            'min_effect': sens['min_effect'],
            'max_effect': sens['max_effect'],
            'mean_effect': sens['mean_effect'],
            'cv': sens['std_effect']/abs(sens['mean_effect']) if sens['mean_effect'] != 0 else None,
            'all_significant': sens['all_significant'],
            'all_same_sign': sens['all_same_sign']
        }
    }

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")

    # Generate report
    if args.report:
        report_path = Path(args.output).with_suffix('.md') if args.output else Path('ddml_report.md')
        generate_report(results, report_path, controls)
        print(f"Report saved to: {report_path}")

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return results


def generate_report(results: dict, output_path: Path, controls: list):
    """Generate markdown report."""
    report = f"""# DDML Analysis Report

Generated: {results['timestamp']}

## Data

- **File**: {results['data_file']}
- **Outcome**: {results['outcome']}
- **Treatment**: {results['treatment']}
- **Controls**: {results['n_controls']} variables

## Model Specification

- **Model Type**: {results['model'].upper()}
- **Cross-fitting**: {results['n_folds']} folds, {results['n_rep']} repetitions
- **Best Learner**: {results['main_result']['learner']}

## Main Results

| Statistic | Value |
|-----------|-------|
| Treatment Effect | {results['main_result']['effect']:.6f} |
| Standard Error | {results['main_result']['se']:.6f} |
| 95% CI Lower | {results['main_result']['ci_lower']:.6f} |
| 95% CI Upper | {results['main_result']['ci_upper']:.6f} |
| P-value | {results['main_result']['p_value']:.6f} |

## Learner Comparison

| Learner | Effect | SE | P-value |
|---------|--------|-----|---------|
"""
    for learner, r in results['learner_comparison'].items():
        stars = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
        report += f"| {learner} | {r['effect']:.6f}{stars} | {r['se']:.6f} | {r['p_value']:.4f} |\n"

    report += f"""
## Sensitivity Analysis

- **Effect Range**: [{results['sensitivity']['min_effect']:.6f}, {results['sensitivity']['max_effect']:.6f}]
- **Mean Effect**: {results['sensitivity']['mean_effect']:.6f}
- **CV**: {results['sensitivity']['cv']*100:.2f}%
- **All Significant**: {results['sensitivity']['all_significant']}
- **All Same Sign**: {results['sensitivity']['all_same_sign']}

## Interpretation

{generate_interpretation(results)}

---

*Generated by DDML Analysis CLI*
*Reference: Chernozhukov et al. (2018)*
"""

    with open(output_path, 'w') as f:
        f.write(report)


def generate_interpretation(results: dict) -> str:
    """Generate interpretation text."""
    effect = results['main_result']['effect']
    se = results['main_result']['se']
    p = results['main_result']['p_value']
    cv = results['sensitivity']['cv'] * 100 if results['sensitivity']['cv'] else 0

    sig_level = "1%" if p < 0.01 else "5%" if p < 0.05 else "10%" if p < 0.1 else "not statistically significant"

    interp = f"""The DDML estimate of the treatment effect is {effect:.4f} (SE = {se:.4f}), """

    if p < 0.1:
        interp += f"statistically significant at the {sig_level} level. "
    else:
        interp += f"which is {sig_level}. "

    if cv < 5:
        interp += f"Results are highly robust across ML specifications (CV = {cv:.1f}%)."
    elif cv < 15:
        interp += f"Results are moderately robust across ML specifications (CV = {cv:.1f}%)."
    else:
        interp += f"Results show sensitivity to ML specification choice (CV = {cv:.1f}%). Interpret with caution."

    return interp


if __name__ == '__main__':
    args = parse_args()
    run_analysis(args)
