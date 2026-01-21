#!/usr/bin/env python3
"""
Complete Instrumental Variables Analysis CLI Script.

This script provides a command-line interface for running comprehensive
IV analysis following Angrist-Pischke methodology.

Usage:
    python run_iv_analysis.py --data data.csv --outcome y --treatment d \
        --instruments z1 z2 --controls x1 x2 --output results/

Features:
    - Full IV workflow (validation, first-stage, estimation, diagnostics)
    - Multiple estimators (2SLS, LIML, GMM)
    - Weak instrument detection and robust inference
    - Publication-ready tables and reports

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import warnings

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from iv_estimator import (
    validate_iv_data,
    first_stage_test,
    weak_iv_diagnostics,
    estimate_2sls,
    estimate_liml,
    estimate_gmm,
    estimate_ols,
    overidentification_test,
    endogeneity_test,
    run_full_iv_analysis,
    STOCK_YOGO_CRITICAL_VALUES
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive Instrumental Variables analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic IV analysis
    python run_iv_analysis.py --data card1995.csv --outcome lwage \\
        --treatment educ --instruments nearc4 --controls exper black south

    # Multiple instruments with output directory
    python run_iv_analysis.py --data data.csv --outcome y --treatment d \\
        --instruments z1 z2 z3 --controls x1 x2 --output results/

    # Verbose mode with JSON output
    python run_iv_analysis.py --data data.csv --outcome y --treatment d \\
        --instruments z1 --verbose --format json
        """
    )

    # Required arguments
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to CSV data file"
    )
    parser.add_argument(
        "--outcome", "-y",
        type=str,
        required=True,
        help="Name of outcome variable"
    )
    parser.add_argument(
        "--treatment", "-t",
        type=str,
        required=True,
        help="Name of endogenous treatment variable"
    )
    parser.add_argument(
        "--instruments", "-z",
        type=str,
        nargs="+",
        required=True,
        help="Names of instrument variables"
    )

    # Optional arguments
    parser.add_argument(
        "--controls", "-x",
        type=str,
        nargs="*",
        default=None,
        help="Names of control variables"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["text", "json", "both"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--weak-iv-threshold",
        type=float,
        default=10.0,
        help="F-statistic threshold for weak instruments (default: 10)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--no-ar-ci",
        action="store_true",
        help="Skip Anderson-Rubin confidence interval computation"
    )
    parser.add_argument(
        "--compare-estimators",
        action="store_true",
        help="Compare all IV estimators (2SLS, LIML, GMM)"
    )

    return parser.parse_args()


def load_data(filepath: str, verbose: bool = False) -> pd.DataFrame:
    """Load data from CSV file."""
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if verbose:
        print(f"Loading data from: {filepath}")

    df = pd.read_csv(path)

    if verbose:
        print(f"  Loaded {len(df):,} observations, {len(df.columns)} variables")

    return df


def validate_and_prepare(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]],
    verbose: bool = False
) -> pd.DataFrame:
    """Validate data and prepare for analysis."""
    if verbose:
        print("\nValidating data...")

    validation = validate_iv_data(df, outcome, treatment, instruments, controls)

    if not validation.is_valid:
        print("ERROR: Data validation failed!")
        for error in validation.errors:
            print(f"  - {error}")
        sys.exit(1)

    if validation.warnings and verbose:
        print("Warnings:")
        for warning in validation.warnings:
            print(f"  - {warning}")

    if verbose:
        print("  Validation passed")
        print(f"  N observations: {validation.summary['n_obs']:,}")
        print(f"  N instruments: {validation.summary['n_instruments']}")
        print(f"  N controls: {validation.summary['n_controls']}")

    return df


def run_analysis(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]],
    weak_iv_threshold: float,
    alpha: float,
    compute_ar_ci: bool,
    compare_estimators: bool,
    verbose: bool
) -> Dict[str, Any]:
    """Run complete IV analysis."""
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "outcome": outcome,
            "treatment": treatment,
            "instruments": instruments,
            "controls": controls or [],
            "n_obs": len(df),
            "alpha": alpha,
            "weak_iv_threshold": weak_iv_threshold
        }
    }

    # Phase 1: First-Stage Analysis
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 1: FIRST-STAGE ANALYSIS")
        print("=" * 60)

    first_stage = first_stage_test(df, treatment, instruments, controls)
    results["first_stage"] = {
        "f_statistic": first_stage['f_statistic'],
        "f_pvalue": first_stage['f_pvalue'],
        "partial_r2": first_stage['partial_r2'],
        "r_squared": first_stage['r_squared'],
        "coefficients": first_stage['coefficients'],
        "std_errors": first_stage['std_errors'],
        "p_values": first_stage['p_values']
    }

    if verbose:
        print(f"First-stage F-statistic: {first_stage['f_statistic']:.2f}")
        print(f"Partial R-squared: {first_stage['partial_r2']:.4f}")
        print("\nInstrument coefficients:")
        for z in instruments:
            coef = first_stage['coefficients'][z]
            se = first_stage['std_errors'][z]
            pval = first_stage['p_values'][z]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {z}: {coef:.4f}{stars} ({se:.4f})")

    # Weak IV diagnostics
    weak_iv = weak_iv_diagnostics(
        first_stage['f_statistic'],
        len(instruments)
    )
    results["weak_iv"] = {
        "passed": weak_iv.passed,
        "f_statistic": weak_iv.statistic,
        "critical_value": weak_iv.threshold,
        "interpretation": weak_iv.interpretation
    }

    if verbose:
        print(f"\nWeak IV Test: {'PASSED' if weak_iv.passed else 'FAILED'}")
        print(f"  Critical value: {weak_iv.threshold:.2f}")

    # Phase 2: Estimation
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 2: IV ESTIMATION")
        print("=" * 60)

    # Always run 2SLS
    result_2sls = estimate_2sls(df, outcome, treatment, instruments, controls)
    results["estimates"] = {
        "2sls": {
            "effect": result_2sls.effect,
            "se": result_2sls.se,
            "ci_lower": result_2sls.ci_lower,
            "ci_upper": result_2sls.ci_upper,
            "p_value": result_2sls.p_value
        }
    }

    if verbose:
        print(f"2SLS Estimate: {result_2sls.effect:.4f} (SE: {result_2sls.se:.4f})")
        print(f"95% CI: [{result_2sls.ci_lower:.4f}, {result_2sls.ci_upper:.4f}]")

    # Run LIML if weak instruments or comparison requested
    if not weak_iv.passed or compare_estimators:
        result_liml = estimate_liml(df, outcome, treatment, instruments, controls)
        results["estimates"]["liml"] = {
            "effect": result_liml.effect,
            "se": result_liml.se,
            "ci_lower": result_liml.ci_lower,
            "ci_upper": result_liml.ci_upper,
            "p_value": result_liml.p_value
        }

        if verbose:
            print(f"LIML Estimate: {result_liml.effect:.4f} (SE: {result_liml.se:.4f})")

    # Run GMM if comparison requested
    if compare_estimators:
        result_gmm = estimate_gmm(df, outcome, treatment, instruments, controls)
        results["estimates"]["gmm"] = {
            "effect": result_gmm.effect,
            "se": result_gmm.se,
            "ci_lower": result_gmm.ci_lower,
            "ci_upper": result_gmm.ci_upper,
            "p_value": result_gmm.p_value,
            "j_statistic": result_gmm.diagnostics.get('j_statistic'),
            "j_pvalue": result_gmm.diagnostics.get('j_pvalue')
        }

        if verbose:
            print(f"GMM Estimate: {result_gmm.effect:.4f} (SE: {result_gmm.se:.4f})")

    # OLS comparison
    result_ols = estimate_ols(df, outcome, treatment, controls)
    results["estimates"]["ols"] = {
        "effect": result_ols.effect,
        "se": result_ols.se,
        "ci_lower": result_ols.ci_lower,
        "ci_upper": result_ols.ci_upper,
        "p_value": result_ols.p_value
    }

    if verbose:
        print(f"\nOLS Estimate (for comparison): {result_ols.effect:.4f} (SE: {result_ols.se:.4f})")
        print(f"IV-OLS difference: {result_2sls.effect - result_ols.effect:.4f}")

    # Phase 3: Diagnostic Tests
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 3: DIAGNOSTIC TESTS")
        print("=" * 60)

    # Endogeneity test
    endog_test = endogeneity_test(df, outcome, treatment, instruments, controls)
    results["diagnostics"] = {
        "endogeneity_test": {
            "statistic": endog_test.statistic,
            "p_value": endog_test.p_value,
            "is_endogenous": endog_test.passed,
            "interpretation": endog_test.interpretation
        }
    }

    if verbose:
        print(f"Endogeneity Test (Wu-Hausman):")
        print(f"  Statistic: {endog_test.statistic:.4f}")
        print(f"  P-value: {endog_test.p_value:.4f}")
        print(f"  Endogeneity detected: {endog_test.passed}")

    # Overidentification test (if applicable)
    if len(instruments) > 1:
        overid_test = overidentification_test(
            result_2sls,
            data=df,
            outcome=outcome,
            treatment=treatment,
            instruments=instruments,
            controls=controls
        )
        results["diagnostics"]["overidentification_test"] = {
            "statistic": overid_test.statistic if not np.isnan(overid_test.statistic) else None,
            "p_value": overid_test.p_value if not np.isnan(overid_test.p_value) else None,
            "passed": overid_test.passed,
            "interpretation": overid_test.interpretation
        }

        if verbose:
            print(f"\nOveridentification Test (Sargan-Hansen):")
            if not np.isnan(overid_test.statistic):
                print(f"  J-statistic: {overid_test.statistic:.4f}")
                print(f"  P-value: {overid_test.p_value:.4f}")
                print(f"  Passed: {overid_test.passed}")
            else:
                print("  Could not compute")

    # Anderson-Rubin CI (if weak instruments or requested)
    if compute_ar_ci and not weak_iv.passed:
        if verbose:
            print("\nComputing Anderson-Rubin confidence interval...")

        try:
            from test_instruments import anderson_rubin_ci
            ar_result = anderson_rubin_ci(df, outcome, treatment, instruments, controls, alpha)
            results["weak_iv_robust"] = {
                "ar_ci_lower": ar_result['ci_lower'],
                "ar_ci_upper": ar_result['ci_upper'],
                "bounded": ar_result['bounded']
            }
            if verbose:
                if ar_result['bounded']:
                    print(f"  AR {(1-alpha)*100:.0f}% CI: [{ar_result['ci_lower']:.4f}, {ar_result['ci_upper']:.4f}]")
                else:
                    print(f"  AR CI may be unbounded")
        except Exception as e:
            if verbose:
                print(f"  Could not compute AR CI: {e}")

    # Determine primary estimate
    if weak_iv.passed:
        primary_method = "2sls"
        primary_effect = result_2sls.effect
        primary_se = result_2sls.se
    else:
        primary_method = "liml"
        primary_effect = results["estimates"]["liml"]["effect"]
        primary_se = results["estimates"]["liml"]["se"]

    results["primary_estimate"] = {
        "method": primary_method.upper(),
        "effect": primary_effect,
        "se": primary_se,
        "reason": "Strong instruments" if weak_iv.passed else "LIML more robust to weak instruments"
    }

    return results


def generate_text_report(results: Dict[str, Any]) -> str:
    """Generate text report from results."""
    lines = []
    meta = results["metadata"]

    lines.append("=" * 70)
    lines.append("INSTRUMENTAL VARIABLES ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {meta['timestamp']}")
    lines.append("")
    lines.append("SPECIFICATION")
    lines.append("-" * 70)
    lines.append(f"Outcome variable:  {meta['outcome']}")
    lines.append(f"Treatment:         {meta['treatment']}")
    lines.append(f"Instruments:       {', '.join(meta['instruments'])}")
    lines.append(f"Controls:          {', '.join(meta['controls']) if meta['controls'] else 'None'}")
    lines.append(f"Observations:      {meta['n_obs']:,}")
    lines.append("")

    # First stage
    fs = results["first_stage"]
    lines.append("FIRST STAGE")
    lines.append("-" * 70)
    lines.append(f"F-statistic:       {fs['f_statistic']:.2f}")
    lines.append(f"Partial R-squared: {fs['partial_r2']:.4f}")
    lines.append("")
    lines.append("Instrument coefficients:")
    for z, coef in fs["coefficients"].items():
        se = fs["std_errors"][z]
        pval = fs["p_values"][z]
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        lines.append(f"  {z:20s}: {coef:10.4f}{stars} ({se:.4f})")
    lines.append("")

    # Weak IV assessment
    weak = results["weak_iv"]
    lines.append("WEAK INSTRUMENTS ASSESSMENT")
    lines.append("-" * 70)
    lines.append(f"Stock-Yogo test:   {'PASSED' if weak['passed'] else 'FAILED'}")
    lines.append(f"Critical value:    {weak['critical_value']:.2f}")
    lines.append("")

    # Estimates
    lines.append("ESTIMATION RESULTS")
    lines.append("-" * 70)
    lines.append(f"{'Method':<12} {'Estimate':>12} {'Std. Err.':>12} {'95% CI':>24}")
    lines.append("-" * 70)

    for method, est in results["estimates"].items():
        ci = f"[{est['ci_lower']:.4f}, {est['ci_upper']:.4f}]"
        lines.append(f"{method.upper():<12} {est['effect']:>12.4f} {est['se']:>12.4f} {ci:>24}")

    lines.append("")

    # Primary estimate
    primary = results["primary_estimate"]
    lines.append(f"PRIMARY ESTIMATE: {primary['method']}")
    lines.append(f"  Effect:  {primary['effect']:.4f}")
    lines.append(f"  SE:      {primary['se']:.4f}")
    lines.append(f"  Reason:  {primary['reason']}")
    lines.append("")

    # Diagnostics
    diag = results["diagnostics"]
    lines.append("DIAGNOSTIC TESTS")
    lines.append("-" * 70)

    endog = diag["endogeneity_test"]
    lines.append(f"Endogeneity (Wu-Hausman):")
    lines.append(f"  Statistic: {endog['statistic']:.4f}")
    lines.append(f"  P-value:   {endog['p_value']:.4f}")
    lines.append(f"  Result:    {'Endogeneity detected' if endog['is_endogenous'] else 'Cannot reject exogeneity'}")

    if "overidentification_test" in diag:
        overid = diag["overidentification_test"]
        lines.append("")
        lines.append(f"Overidentification (Sargan-Hansen):")
        if overid["statistic"] is not None:
            lines.append(f"  J-statistic: {overid['statistic']:.4f}")
            lines.append(f"  P-value:     {overid['p_value']:.4f}")
            lines.append(f"  Result:      {'Passed' if overid['passed'] else 'FAILED - instruments may be invalid'}")
        else:
            lines.append("  Could not compute")

    if "weak_iv_robust" in results:
        ar = results["weak_iv_robust"]
        lines.append("")
        lines.append("Weak-IV Robust Inference:")
        if ar["bounded"]:
            lines.append(f"  Anderson-Rubin CI: [{ar['ar_ci_lower']:.4f}, {ar['ar_ci_upper']:.4f}]")
        else:
            lines.append("  Anderson-Rubin CI: May be unbounded")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    output_format: str,
    verbose: bool
) -> None:
    """Save results to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_format in ["text", "both"]:
        report = generate_text_report(results)
        text_file = output_path / f"iv_analysis_{timestamp}.txt"
        text_file.write_text(report)
        if verbose:
            print(f"\nSaved text report: {text_file}")

    if output_format in ["json", "both"]:
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        results_json = convert_types(results)
        json_file = output_path / f"iv_analysis_{timestamp}.json"
        json_file.write_text(json.dumps(results_json, indent=2))
        if verbose:
            print(f"Saved JSON results: {json_file}")


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Load data
        df = load_data(args.data, args.verbose)

        # Validate and prepare
        df = validate_and_prepare(
            df,
            args.outcome,
            args.treatment,
            args.instruments,
            args.controls,
            args.verbose
        )

        # Run analysis
        results = run_analysis(
            df,
            args.outcome,
            args.treatment,
            args.instruments,
            args.controls,
            args.weak_iv_threshold,
            args.alpha,
            compute_ar_ci=not args.no_ar_ci,
            compare_estimators=args.compare_estimators,
            verbose=args.verbose
        )

        # Output results
        if args.output:
            save_results(results, args.output, args.format, args.verbose)
        else:
            # Print to stdout
            report = generate_text_report(results)
            print(report)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
