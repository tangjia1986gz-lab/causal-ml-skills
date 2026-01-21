#!/usr/bin/env python3
"""
Complete Causal Mediation Analysis CLI

Runs full mediation analysis workflow including:
- Data validation
- Effect estimation (Baron-Kenny or ML-enhanced)
- Bootstrap confidence intervals
- Sensitivity analysis
- Report generation

Usage:
    python run_mediation_analysis.py data.csv \
        --outcome earnings --treatment training --mediator skills \
        --controls age education --method auto

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from mediation_estimator import (
    run_full_mediation_analysis,
    estimate_baron_kenny,
    estimate_ml_mediation,
    validate_mediation_setup,
    bootstrap_mediation_ci,
    sensitivity_analysis_mediation,
    create_mediation_data,
    compare_mediation_methods
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run causal mediation analysis on a dataset.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_mediation_analysis.py data.csv -y earnings -d training -m skills

  # With controls and custom method
  python run_mediation_analysis.py data.csv -y earnings -d training -m skills \\
      --controls age education experience --method ml_enhanced

  # Full analysis with all options
  python run_mediation_analysis.py data.csv -y earnings -d training -m skills \\
      --controls age education --method auto --n-bootstrap 1000 \\
      --output results.json --report report.md
        """
    )

    # Required arguments
    parser.add_argument(
        'data_file',
        type=str,
        help='Path to CSV data file'
    )

    # Core variables
    parser.add_argument(
        '-y', '--outcome',
        type=str,
        required=True,
        help='Name of outcome variable (Y)'
    )
    parser.add_argument(
        '-d', '--treatment',
        type=str,
        required=True,
        help='Name of treatment variable (D)'
    )
    parser.add_argument(
        '-m', '--mediator',
        type=str,
        required=True,
        help='Name of mediator variable (M)'
    )
    parser.add_argument(
        '--controls',
        type=str,
        nargs='*',
        default=[],
        help='Names of control variables (X)'
    )

    # Method options
    parser.add_argument(
        '--method',
        type=str,
        choices=['baron_kenny', 'ml_enhanced', 'auto'],
        default='auto',
        help='Estimation method (default: auto)'
    )
    parser.add_argument(
        '--ml-mediator',
        type=str,
        choices=['lasso', 'ridge', 'random_forest', 'xgboost'],
        default='lasso',
        help='ML learner for mediator model (default: lasso)'
    )
    parser.add_argument(
        '--ml-outcome',
        type=str,
        choices=['lasso', 'ridge', 'random_forest', 'xgboost'],
        default='lasso',
        help='ML learner for outcome model (default: lasso)'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of cross-fitting folds for ML methods (default: 5)'
    )

    # Bootstrap options
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=500,
        help='Number of bootstrap replications (default: 500)'
    )
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        help='Confidence level for intervals (default: 0.95)'
    )

    # Sensitivity analysis
    parser.add_argument(
        '--no-sensitivity',
        action='store_true',
        help='Skip sensitivity analysis'
    )
    parser.add_argument(
        '--rho-range',
        type=float,
        nargs=3,
        metavar=('START', 'END', 'STEP'),
        default=[-0.5, 0.5, 0.05],
        help='Range for sensitivity parameter rho (default: -0.5 0.5 0.05)'
    )

    # Output options
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file for JSON results'
    )
    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Output file for markdown report'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'text', 'both'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    parser.add_argument(
        '--compare-methods',
        action='store_true',
        help='Compare Baron-Kenny and ML methods'
    )

    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    return parser.parse_args()


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if path.suffix.lower() == '.csv':
        return pd.read_csv(file_path)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif path.suffix.lower() == '.dta':
        return pd.read_stata(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def validate_variables(data: pd.DataFrame, outcome: str, treatment: str,
                       mediator: str, controls: list):
    """Validate that all specified variables exist in the data."""
    required = [outcome, treatment, mediator] + controls
    missing = [v for v in required if v not in data.columns]

    if missing:
        raise ValueError(f"Variables not found in data: {missing}")


def format_results_text(results, args) -> str:
    """Format results as text output."""
    lines = [
        "=" * 70,
        "CAUSAL MEDIATION ANALYSIS RESULTS",
        "=" * 70,
        "",
        f"Data File: {args.data_file}",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Causal Pathway: {args.treatment} -> {args.mediator} -> {args.outcome}",
        f"Method: {results.diagnostics.get('method', args.method)}",
        f"Sample Size: {results.diagnostics['main_results']['n']:,}",
        "",
        results.summary_table,
        "",
        "-" * 70,
        "INTERPRETATION",
        "-" * 70,
        "",
        results.interpretation,
        "",
        "=" * 70,
    ]

    return "\n".join(lines)


def results_to_dict(results) -> dict:
    """Convert CausalOutput to serializable dict."""
    return {
        'effect': float(results.effect),
        'se': float(results.se),
        'ci_lower': float(results.ci_lower),
        'ci_upper': float(results.ci_upper),
        'p_value': float(results.p_value),
        'diagnostics': _make_serializable(results.diagnostics),
        'interpretation': results.interpretation
    }


def _make_serializable(obj):
    """Recursively convert numpy types to Python types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def generate_markdown_report(results, args) -> str:
    """Generate markdown report."""
    report = f"""# Causal Mediation Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Study Design

- **Treatment**: `{args.treatment}`
- **Mediator**: `{args.mediator}`
- **Outcome**: `{args.outcome}`
- **Controls**: {', '.join(f'`{c}`' for c in args.controls) if args.controls else 'None'}
- **Method**: {args.method}
- **Sample Size**: {results.diagnostics['main_results']['n']:,}

## Causal Pathway

```
{args.treatment} ──────────────────────────────────────> {args.outcome}
      │                   Direct Effect (ADE)                ↑
      │                                                      │
      └─────────> {args.mediator} ─────────────────────────>─┘
             Indirect Effect (ACME)
```

## Results

### Effect Decomposition

| Effect | Estimate | Std. Error | 95% CI | p-value |
|--------|----------|------------|--------|---------|
| Total Effect | {results.diagnostics['main_results']['total_effect']:.4f} | {results.diagnostics['main_results'].get('total_se', 'N/A')} | - | - |
| Direct (ADE) | {results.diagnostics['main_results']['ade']:.4f} | {results.diagnostics['main_results']['ade_se']:.4f} | [{results.diagnostics['main_results']['ade_ci_lower']:.4f}, {results.diagnostics['main_results']['ade_ci_upper']:.4f}] | {results.diagnostics['main_results']['ade_pvalue']:.4f} |
| Indirect (ACME) | {results.diagnostics['main_results']['acme']:.4f} | {results.diagnostics['main_results']['acme_se']:.4f} | [{results.diagnostics['main_results']['acme_ci_lower']:.4f}, {results.diagnostics['main_results']['acme_ci_upper']:.4f}] | {results.diagnostics['main_results']['acme_pvalue']:.4f} |

**Proportion Mediated**: {results.diagnostics['main_results']['prop_mediated']*100:.1f}%

### Interpretation

{results.interpretation}

"""

    # Add sensitivity analysis if available
    if results.diagnostics.get('sensitivity'):
        sens = results.diagnostics['sensitivity']
        report += f"""
## Sensitivity Analysis

The ACME estimate is sensitive to unmeasured confounding between the mediator and outcome.

- **Breakpoint (rho)**: {sens.get('breakpoint', 'N/A'):.3f}
- **Robustness**: {sens.get('robustness', 'N/A').upper()}
- **Interpretation**: {sens.get('interpretation', 'N/A')}

"""

    report += f"""
## Methods

### Estimation

{"Baron-Kenny (OLS)" if args.method == 'baron_kenny' else "ML-Enhanced" if args.method == 'ml_enhanced' else "Auto-selected"} mediation analysis was used.

### Bootstrap Inference

{args.n_bootstrap} bootstrap replications were used for confidence intervals.

### Software

Analysis performed using `causal-mediation-ml` skill (Python).

---

*Report generated by causal-ml-skills*
"""

    return report


def main():
    """Main entry point."""
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    if args.verbose:
        print(f"Loading data from: {args.data_file}")

    # Load data
    try:
        data = load_data(args.data_file)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Loaded {len(data):,} observations with {len(data.columns)} variables")

    # Validate variables
    try:
        validate_variables(data, args.outcome, args.treatment, args.mediator, args.controls)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate data structure
    if args.verbose:
        print("Validating data structure...")

    validation = validate_mediation_setup(
        data, args.outcome, args.treatment, args.mediator, args.controls
    )

    if not validation['is_valid']:
        print(f"Data validation failed: {validation['errors']}", file=sys.stderr)
        sys.exit(1)

    if validation['warnings']:
        for warning in validation['warnings']:
            print(f"Warning: {warning}")

    # Compare methods if requested
    if args.compare_methods:
        if args.verbose:
            print("Comparing estimation methods...")

        comparison = compare_mediation_methods(
            data, args.outcome, args.treatment, args.mediator, args.controls
        )
        print("\n" + comparison['summary_table'])
        print()

    # Run main analysis
    if args.verbose:
        print(f"Running {args.method} mediation analysis...")

    try:
        results = run_full_mediation_analysis(
            data=data,
            outcome=args.outcome,
            treatment=args.treatment,
            mediator=args.mediator,
            controls=args.controls,
            method=args.method,
            n_bootstrap=args.n_bootstrap,
            run_sensitivity=not args.no_sensitivity,
            random_state=args.seed
        )
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)

    # Output results
    if args.format in ['text', 'both']:
        print(format_results_text(results, args))

    if args.format in ['json', 'both'] or args.output:
        results_dict = results_to_dict(results)
        results_dict['metadata'] = {
            'data_file': args.data_file,
            'analysis_date': datetime.now().isoformat(),
            'method': args.method,
            'n_bootstrap': args.n_bootstrap,
            'random_seed': args.seed
        }

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results_dict, f, indent=2)
            if args.verbose:
                print(f"Results saved to: {args.output}")
        elif args.format == 'json':
            print(json.dumps(results_dict, indent=2))

    # Generate markdown report if requested
    if args.report:
        report = generate_markdown_report(results, args)
        with open(args.report, 'w') as f:
            f.write(report)
        if args.verbose:
            print(f"Report saved to: {args.report}")

    if args.verbose:
        print("Analysis complete.")


if __name__ == '__main__':
    main()
