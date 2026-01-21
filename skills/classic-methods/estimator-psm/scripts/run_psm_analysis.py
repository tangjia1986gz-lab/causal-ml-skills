#!/usr/bin/env python3
"""
PSM Analysis CLI Script

Complete command-line interface for Propensity Score Matching analysis.
Implements the full Imbens-Rubin workflow:
1. Data validation
2. Propensity score estimation
3. Overlap assessment
4. Matching
5. Balance checking
6. Effect estimation
7. Sensitivity analysis

Usage:
    python run_psm_analysis.py --data data.csv --outcome y --treatment d \\
                               --covariates x1 x2 x3 --output results/

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from psm_estimator import (
    validate_psm_data,
    estimate_propensity_score,
    check_common_support,
    match_nearest_neighbor,
    match_kernel,
    match_mahalanobis,
    check_balance,
    create_balance_table,
    estimate_att,
    estimate_ate,
    rosenbaum_sensitivity,
    run_full_psm_analysis,
    CausalOutput
)


def setup_argparser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Propensity Score Matching Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python run_psm_analysis.py --data lalonde.csv --outcome re78 \\
                               --treatment treat --covariates age education black

    # With specific matching method
    python run_psm_analysis.py --data data.csv --outcome y --treatment d \\
                               --covariates x1 x2 x3 --match-method kernel

    # Full analysis with output
    python run_psm_analysis.py --data data.csv --outcome y --treatment d \\
                               --covariates x1 x2 x3 --output results/ \\
                               --save-plots --sensitivity --verbose
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
        help="Name of treatment indicator (0/1)"
    )
    parser.add_argument(
        "--covariates", "-x",
        type=str,
        nargs="+",
        required=True,
        help="Names of covariates for propensity score model"
    )

    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results (default: print to console)"
    )
    parser.add_argument(
        "--ps-method",
        type=str,
        choices=["logit", "probit", "gbm", "random_forest", "lasso"],
        default="logit",
        help="Propensity score estimation method (default: logit)"
    )
    parser.add_argument(
        "--match-method",
        type=str,
        choices=["nearest_neighbor", "kernel", "mahalanobis"],
        default="nearest_neighbor",
        help="Matching method (default: nearest_neighbor)"
    )
    parser.add_argument(
        "--caliper",
        type=float,
        default=0.1,
        help="Caliper for nearest neighbor matching in PS SD units (default: 0.1)"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=1,
        help="Number of matches per treated unit (default: 1)"
    )
    parser.add_argument(
        "--replacement",
        action="store_true",
        default=True,
        help="Match with replacement (default: True)"
    )
    parser.add_argument(
        "--no-replacement",
        action="store_true",
        help="Match without replacement"
    )
    parser.add_argument(
        "--estimand",
        type=str,
        choices=["ATT", "ATE"],
        default="ATT",
        help="Estimand: ATT or ATE (default: ATT)"
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run Rosenbaum sensitivity analysis"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save diagnostic plots"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    return parser


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.

    Parameters
    ----------
    filepath : str
        Path to CSV file

    Returns
    -------
    pd.DataFrame
        Loaded data

    Raises
    ------
    FileNotFoundError
        If file does not exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    data = pd.read_csv(filepath)
    print(f"Loaded data: {len(data)} observations, {len(data.columns)} variables")

    return data


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    width = 60
    print("\n" + char * width)
    print(title.center(width))
    print(char * width + "\n")


def run_analysis(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: List[str],
    ps_method: str = "logit",
    match_method: str = "nearest_neighbor",
    caliper: float = 0.1,
    n_neighbors: int = 1,
    replacement: bool = True,
    estimand: str = "ATT",
    run_sensitivity: bool = True,
    verbose: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run complete PSM analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    outcome : str
        Outcome variable name
    treatment : str
        Treatment indicator name
    covariates : List[str]
        Covariate names
    ps_method : str
        PS estimation method
    match_method : str
        Matching algorithm
    caliper : float
        Matching caliper
    n_neighbors : int
        Number of matches
    replacement : bool
        Match with replacement
    estimand : str
        Target estimand (ATT/ATE)
    run_sensitivity : bool
        Run sensitivity analysis
    verbose : bool
        Print progress
    seed : int
        Random seed

    Returns
    -------
    Dict[str, Any]
        Complete analysis results
    """
    np.random.seed(seed)
    results = {"timestamp": datetime.now().isoformat()}

    # =================================================================
    # PHASE 1: DATA VALIDATION
    # =================================================================
    if verbose:
        print_section("PHASE 1: DATA VALIDATION")

    validation = validate_psm_data(data, outcome, treatment, covariates)
    results["validation"] = {
        "is_valid": validation.is_valid,
        "errors": validation.errors,
        "warnings": validation.warnings,
        "summary": validation.summary
    }

    if verbose:
        print(validation)

    if not validation.is_valid:
        print("\nERROR: Data validation failed. Cannot proceed.")
        return results

    # =================================================================
    # PHASE 2: PROPENSITY SCORE ESTIMATION
    # =================================================================
    if verbose:
        print_section("PHASE 2: PROPENSITY SCORE ESTIMATION")

    ps_result = estimate_propensity_score(
        data=data,
        treatment=treatment,
        covariates=covariates,
        method=ps_method
    )

    data = data.copy()
    data["_propensity_score"] = ps_result.propensity_scores

    results["propensity_score"] = {
        "method": ps_result.method,
        "auc": ps_result.auc,
        "ps_min": float(np.nanmin(ps_result.propensity_scores)),
        "ps_max": float(np.nanmax(ps_result.propensity_scores)),
        "ps_mean": float(np.nanmean(ps_result.propensity_scores)),
        "feature_importance": ps_result.feature_importance
    }

    if verbose:
        print(ps_result.summary())
        if ps_result.feature_importance:
            print("\nFeature Importance:")
            for feat, imp in sorted(
                ps_result.feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ):
                print(f"  {feat}: {imp:.4f}")

    # =================================================================
    # PHASE 3: OVERLAP CHECK
    # =================================================================
    if verbose:
        print_section("PHASE 3: COMMON SUPPORT CHECK")

    ps_treated = data.loc[data[treatment] == 1, "_propensity_score"].values
    ps_control = data.loc[data[treatment] == 0, "_propensity_score"].values

    overlap_result = check_common_support(ps_treated, ps_control)

    results["overlap"] = {
        "passed": overlap_result.passed,
        "interpretation": overlap_result.interpretation,
        "details": overlap_result.details
    }

    if verbose:
        print(overlap_result.interpretation)
        print(f"\nOverlap ratio: {overlap_result.details['overlap_ratio']:.3f}")
        print(f"Common support: [{overlap_result.details['common_support_lower']:.4f}, "
              f"{overlap_result.details['common_support_upper']:.4f}]")

    # =================================================================
    # PHASE 4: MATCHING
    # =================================================================
    if verbose:
        print_section("PHASE 4: MATCHING")

    if match_method == "nearest_neighbor":
        match_result = match_nearest_neighbor(
            data=data,
            propensity_score="_propensity_score",
            treatment=treatment,
            n_neighbors=n_neighbors,
            replacement=replacement,
            caliper=caliper,
            caliper_scale="ps_std"
        )
    elif match_method == "kernel":
        match_result = match_kernel(
            data=data,
            propensity_score="_propensity_score",
            treatment=treatment,
            kernel="epanechnikov",
            bandwidth="optimal"
        )
    elif match_method == "mahalanobis":
        match_result = match_mahalanobis(
            data=data,
            treatment=treatment,
            covariates=covariates,
            n_neighbors=n_neighbors,
            replacement=replacement
        )

    results["matching"] = {
        "method": match_result.method,
        "n_treated": match_result.n_treated,
        "n_control_matched": match_result.n_control_matched,
        "n_unmatched": match_result.n_unmatched,
        "parameters": match_result.parameters
    }

    if verbose:
        print(match_result.summary())

    # =================================================================
    # PHASE 5: BALANCE CHECK
    # =================================================================
    if verbose:
        print_section("PHASE 5: BALANCE ASSESSMENT")

    balance_before = check_balance(data, treatment, covariates)
    balance_after = check_balance(
        match_result.matched_data,
        treatment,
        covariates,
        weights="_match_weight"
    )

    results["balance"] = {
        "before": {
            "smd": balance_before.smd_after,
            "variance_ratio": balance_before.variance_ratio_after
        },
        "after": {
            "smd": balance_after.smd_after,
            "variance_ratio": balance_after.variance_ratio_after,
            "balanced": balance_after.balanced
        }
    }

    if verbose:
        # Print balance table
        balance_table = create_balance_table(
            data, match_result.matched_data,
            treatment, covariates,
            weights_after="_match_weight"
        )
        print(balance_table)

        if not balance_after.balanced:
            print("\nWARNING: Balance not achieved for all covariates!")
            print("Consider re-specifying propensity score model or matching method.")

    # =================================================================
    # PHASE 6: EFFECT ESTIMATION
    # =================================================================
    if verbose:
        print_section("PHASE 6: TREATMENT EFFECT ESTIMATION")

    if estimand == "ATT":
        effect_result = estimate_att(
            data=match_result.matched_data,
            outcome=outcome,
            treatment=treatment,
            weights="_match_weight",
            se_method="bootstrap",
            n_bootstrap=1000,
            random_state=seed
        )
    else:  # ATE
        effect_result = estimate_ate(
            data=data,
            outcome=outcome,
            treatment=treatment,
            propensity_score="_propensity_score",
            estimator="ipw"
        )

    results["effect"] = {
        "estimand": estimand,
        "effect": float(effect_result.effect),
        "se": float(effect_result.se),
        "ci_lower": float(effect_result.ci_lower),
        "ci_upper": float(effect_result.ci_upper),
        "p_value": float(effect_result.p_value),
        "diagnostics": effect_result.diagnostics
    }

    if verbose:
        print(f"Estimand: {estimand}")
        print(f"Point Estimate: {effect_result.effect:.4f}")
        print(f"Standard Error: {effect_result.se:.4f}")
        print(f"95% CI: [{effect_result.ci_lower:.4f}, {effect_result.ci_upper:.4f}]")
        print(f"P-value: {effect_result.p_value:.4f}")

        if effect_result.p_value < 0.001:
            sig = "*** (p < 0.001)"
        elif effect_result.p_value < 0.01:
            sig = "** (p < 0.01)"
        elif effect_result.p_value < 0.05:
            sig = "* (p < 0.05)"
        else:
            sig = "(not significant)"
        print(f"Significance: {sig}")

    # =================================================================
    # PHASE 7: SENSITIVITY ANALYSIS
    # =================================================================
    if run_sensitivity:
        if verbose:
            print_section("PHASE 7: SENSITIVITY ANALYSIS")

        try:
            sensitivity_result = rosenbaum_sensitivity(
                data=match_result.matched_data,
                outcome=outcome,
                treatment=treatment,
                gamma_range=[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
            )

            results["sensitivity"] = {
                "gamma_values": sensitivity_result.gamma_values,
                "lower_bounds": [float(x) for x in sensitivity_result.lower_bounds],
                "upper_bounds": [float(x) for x in sensitivity_result.upper_bounds],
                "p_values_upper": [float(x) for x in sensitivity_result.p_values_upper],
                "critical_gamma": float(sensitivity_result.critical_gamma)
            }

            if verbose:
                print(sensitivity_result.summary())
                print(f"\nInterpretation: Results are robust to hidden bias up to "
                      f"Gamma = {sensitivity_result.critical_gamma:.2f}")

        except Exception as e:
            results["sensitivity"] = {"error": str(e)}
            if verbose:
                print(f"Sensitivity analysis failed: {e}")

    return results


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    data: pd.DataFrame = None,
    treatment: str = None,
    save_plots: bool = False
) -> None:
    """
    Save analysis results to files.

    Parameters
    ----------
    results : Dict[str, Any]
        Analysis results
    output_dir : str
        Output directory path
    data : pd.DataFrame, optional
        Data with propensity scores (for plots)
    treatment : str, optional
        Treatment column name
    save_plots : bool
        Whether to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    json_path = output_path / "psm_results.json"

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        else:
            return obj

    results_json = convert_for_json(results)

    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Save plots
    if save_plots and data is not None and treatment is not None:
        try:
            import matplotlib.pyplot as plt
            from psm_estimator import plot_propensity_overlap

            ps_treated = data.loc[data[treatment] == 1, "_propensity_score"].values
            ps_control = data.loc[data[treatment] == 0, "_propensity_score"].values

            # Overlap plot
            fig = plot_propensity_overlap(ps_treated, ps_control)
            fig_path = output_path / "overlap_plot.png"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Overlap plot saved to: {fig_path}")

        except ImportError:
            print("matplotlib not available for plotting")


def main():
    """Main entry point."""
    parser = setup_argparser()
    args = parser.parse_args()

    print_section("PROPENSITY SCORE MATCHING ANALYSIS", "=")

    # Load data
    try:
        data = load_data(args.data)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Determine replacement setting
    replacement = not args.no_replacement

    # Run analysis
    results = run_analysis(
        data=data,
        outcome=args.outcome,
        treatment=args.treatment,
        covariates=args.covariates,
        ps_method=args.ps_method,
        match_method=args.match_method,
        caliper=args.caliper,
        n_neighbors=args.n_neighbors,
        replacement=replacement,
        estimand=args.estimand,
        run_sensitivity=args.sensitivity,
        verbose=args.verbose,
        seed=args.seed
    )

    # Save results if output directory specified
    if args.output:
        save_results(
            results=results,
            output_dir=args.output,
            data=data if "_propensity_score" in data.columns else None,
            treatment=args.treatment,
            save_plots=args.save_plots
        )

    # Print summary
    print_section("ANALYSIS COMPLETE", "=")

    if "effect" in results:
        print(f"Treatment Effect ({results['effect']['estimand']}): "
              f"{results['effect']['effect']:.4f}")
        print(f"95% CI: [{results['effect']['ci_lower']:.4f}, "
              f"{results['effect']['ci_upper']:.4f}]")
        print(f"P-value: {results['effect']['p_value']:.4f}")

    if "sensitivity" in results and "critical_gamma" in results["sensitivity"]:
        print(f"Sensitivity: Robust to Gamma = {results['sensitivity']['critical_gamma']:.2f}")


if __name__ == "__main__":
    main()
