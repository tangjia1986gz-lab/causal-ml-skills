#!/usr/bin/env python
"""
Run Bayesian Model CLI

Command-line interface for fitting Bayesian econometric models.
Supports linear regression, hierarchical models, and causal inference.

Usage:
    python run_bayesian_model.py --data data.csv --outcome y --features x1,x2,x3
    python run_bayesian_model.py --data data.csv --outcome y --treatment t --features x1,x2
    python run_bayesian_model.py --data data.csv --outcome y --features x1 --groups group_id
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from bayesian_estimator import BayesianEstimator, BayesianResult
    import arviz as az
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Install requirements: pip install pymc arviz pandas numpy")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit Bayesian econometric models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic regression
  python run_bayesian_model.py --data wages.csv --outcome log_wage --features education,experience

  # Causal effect estimation
  python run_bayesian_model.py --data experiment.csv --outcome outcome --treatment treated --features age,gender

  # Hierarchical model
  python run_bayesian_model.py --data panel.csv --outcome y --features x1,x2 --groups firm_id

  # Custom priors and sampling
  python run_bayesian_model.py --data data.csv --outcome y --features x1 --prior-scale 1.0 --draws 4000 --tune 2000
        """
    )

    # Data arguments
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to CSV data file"
    )
    parser.add_argument(
        "--outcome", "-y",
        required=True,
        help="Name of outcome variable"
    )
    parser.add_argument(
        "--features", "-x",
        required=True,
        help="Comma-separated list of feature names"
    )
    parser.add_argument(
        "--treatment", "-t",
        default=None,
        help="Name of treatment variable (for causal inference)"
    )
    parser.add_argument(
        "--groups", "-g",
        default=None,
        help="Name of group variable (for hierarchical model)"
    )

    # Prior arguments
    parser.add_argument(
        "--prior-scale",
        type=float,
        default=2.5,
        help="Prior SD for coefficients (default: 2.5)"
    )
    parser.add_argument(
        "--prior-intercept-scale",
        type=float,
        default=10.0,
        help="Prior SD for intercept (default: 10.0)"
    )

    # Sampling arguments
    parser.add_argument(
        "--draws",
        type=int,
        default=2000,
        help="Number of posterior samples (default: 2000)"
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=1000,
        help="Number of tuning samples (default: 1000)"
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4)"
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.9,
        help="Target acceptance rate (default: 0.9)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # Output arguments
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--save-trace",
        default=None,
        help="Path to save trace (NetCDF format)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize features before fitting"
    )

    return parser.parse_args()


def load_data(path: str, outcome: str, features: str,
              treatment: Optional[str] = None,
              groups: Optional[str] = None):
    """Load and prepare data from CSV."""
    df = pd.read_csv(path)

    feature_list = [f.strip() for f in features.split(",")]

    # Validate columns exist
    required_cols = [outcome] + feature_list
    if treatment:
        required_cols.append(treatment)
    if groups:
        required_cols.append(groups)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    y = df[outcome].values
    X = df[feature_list].values

    result = {"y": y, "X": X, "feature_names": feature_list}

    if treatment:
        result["treatment"] = df[treatment].values
    if groups:
        # Convert groups to integer codes
        group_values = df[groups].values
        unique_groups = np.unique(group_values)
        group_map = {g: i for i, g in enumerate(unique_groups)}
        result["groups"] = np.array([group_map[g] for g in group_values])
        result["n_groups"] = len(unique_groups)

    return result


def run_regression(estimator: BayesianEstimator, data: dict, args) -> BayesianResult:
    """Run standard Bayesian regression."""
    return estimator.fit_bayesian_regression(
        y=data["y"],
        X=data["X"],
        prior_intercept_sigma=args.prior_intercept_scale,
        prior_beta_sigma=args.prior_scale,
        feature_names=data["feature_names"],
        standardize=args.standardize
    )


def run_causal(estimator: BayesianEstimator, data: dict, args) -> BayesianResult:
    """Run Bayesian causal effect estimation."""
    return estimator.fit_bayesian_ate(
        y=data["y"],
        treatment=data["treatment"],
        X=data["X"],
        prior_ate_sigma=args.prior_scale
    )


def run_hierarchical(estimator: BayesianEstimator, data: dict, args) -> BayesianResult:
    """Run hierarchical model."""
    return estimator.fit_hierarchical_model(
        y=data["y"],
        X=data["X"],
        groups=data["groups"],
        varying_intercept=True,
        varying_slope=False
    )


def format_results(result: BayesianResult, args) -> dict:
    """Format results for output."""
    output = {
        "model_type": "regression",
        "sampling": {
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "target_accept": args.target_accept
        },
        "diagnostics": result.diagnostics,
        "summary": result.summary.to_dict()
    }

    if args.treatment:
        output["model_type"] = "causal"
        if result.causal_output:
            output["causal_effect"] = {
                "ate": result.causal_output.effect,
                "se": result.causal_output.se,
                "ci_lower": result.causal_output.ci_lower,
                "ci_upper": result.causal_output.ci_upper,
                "prob_positive": result.causal_output.details.get("prob_positive")
            }

    if args.groups:
        output["model_type"] = "hierarchical"
        output["n_groups"] = args.groups

    return output


def print_results(result: BayesianResult, args):
    """Print formatted results to console."""
    print("\n" + "=" * 60)
    print("BAYESIAN MODEL RESULTS")
    print("=" * 60)

    # Model info
    if args.treatment:
        print(f"\nModel Type: Causal Effect Estimation (ATE)")
    elif args.groups:
        print(f"\nModel Type: Hierarchical/Multilevel")
    else:
        print(f"\nModel Type: Linear Regression")

    print(f"Outcome: {args.outcome}")
    print(f"Features: {args.features}")
    if args.treatment:
        print(f"Treatment: {args.treatment}")
    if args.groups:
        print(f"Groups: {args.groups}")

    # Summary
    print("\n" + "-" * 40)
    print("PARAMETER SUMMARY")
    print("-" * 40)
    print(result.summary.to_string())

    # Causal effect
    if result.causal_output:
        print("\n" + "-" * 40)
        print("CAUSAL EFFECT")
        print("-" * 40)
        co = result.causal_output
        print(f"ATE: {co.effect:.4f}")
        print(f"Posterior SD: {co.se:.4f}")
        print(f"95% HDI: [{co.ci_lower:.4f}, {co.ci_upper:.4f}]")
        print(f"P(ATE > 0): {co.details.get('prob_positive', 'N/A'):.3f}")

    # Diagnostics
    print("\n" + "-" * 40)
    print("MCMC DIAGNOSTICS")
    print("-" * 40)
    diag = result.diagnostics
    print(f"Max Rhat: {diag['rhat_max']:.3f} {'[OK]' if diag['rhat_ok'] else '[WARNING]'}")
    print(f"Min Bulk ESS: {diag['ess_bulk_min']:.0f} {'[OK]' if diag['ess_ok'] else '[WARNING]'}")
    print(f"Min Tail ESS: {diag['ess_tail_min']:.0f}")
    print(f"Divergences: {diag['n_divergences']} {'[OK]' if diag['divergences_ok'] else '[WARNING]'}")

    if diag['all_ok']:
        print("\n[OK] All diagnostics passed")
    else:
        print("\n[WARNING] Some diagnostics failed - consider:")
        if not diag['rhat_ok']:
            print("  - Increasing draws/tune for convergence")
        if not diag['ess_ok']:
            print("  - Increasing draws for better ESS")
        if not diag['divergences_ok']:
            print("  - Increasing target_accept or reparameterizing model")

    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    try:
        data = load_data(
            args.data,
            args.outcome,
            args.features,
            args.treatment,
            args.groups
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print(f"Loaded {len(data['y'])} observations")
    print(f"Features: {data['feature_names']}")

    # Initialize estimator
    estimator = BayesianEstimator(
        random_seed=args.seed,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept
    )

    # Fit appropriate model
    print("\nFitting model...")
    try:
        if args.treatment:
            result = run_causal(estimator, data, args)
        elif args.groups:
            result = run_hierarchical(estimator, data, args)
        else:
            result = run_regression(estimator, data, args)
    except Exception as e:
        print(f"Error fitting model: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Print results
    print_results(result, args)

    # Save trace if requested
    if args.save_trace:
        print(f"\nSaving trace to {args.save_trace}...")
        result.trace.to_netcdf(args.save_trace)

    # Save results if requested
    if args.output:
        print(f"Saving results to {args.output}...")
        output = format_results(result, args)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)

    print("\nDone!")


if __name__ == "__main__":
    main()
