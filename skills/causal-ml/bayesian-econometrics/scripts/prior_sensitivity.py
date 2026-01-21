#!/usr/bin/env python
"""
Prior Sensitivity Analysis

Analyze how posterior estimates change with different prior specifications.
Essential for assessing robustness of Bayesian causal inference.

Usage:
    python prior_sensitivity.py --data data.csv --outcome y --features x1,x2
    python prior_sensitivity.py --data data.csv --outcome y --treatment t --scales 0.5,1,2.5,5,10
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from typing import List, Optional

try:
    import pymc as pm
    import arviz as az
    import matplotlib.pyplot as plt
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prior sensitivity analysis for Bayesian models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sensitivity analysis
  python prior_sensitivity.py --data wages.csv --outcome log_wage --features education

  # Causal effect sensitivity
  python prior_sensitivity.py --data experiment.csv --outcome y --treatment treated --features x1,x2

  # Custom prior scales
  python prior_sensitivity.py --data data.csv --outcome y --features x1 --scales 0.1,0.5,1,2.5,5,10,25

  # Output results
  python prior_sensitivity.py --data data.csv --outcome y --features x1 --output sensitivity.json
        """
    )

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
        help="Name of treatment variable (for ATE sensitivity)"
    )
    parser.add_argument(
        "--scales", "-s",
        default="0.5,1.0,2.5,5.0,10.0",
        help="Comma-separated prior scales to test (default: 0.5,1.0,2.5,5.0,10.0)"
    )
    parser.add_argument(
        "--target-param",
        default=None,
        help="Parameter to focus on (default: ate if treatment, else first beta)"
    )
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
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Output path for sensitivity plot"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )

    return parser.parse_args()


def load_data(path: str, outcome: str, features: str, treatment: Optional[str] = None):
    """Load data from CSV."""
    df = pd.read_csv(path)
    feature_list = [f.strip() for f in features.split(",")]

    y = df[outcome].values
    X = df[feature_list].values

    result = {"y": y, "X": X, "feature_names": feature_list}

    if treatment:
        result["treatment"] = df[treatment].values

    return result


def fit_model(y, X, treatment, prior_scale, draws, tune, seed, progressbar):
    """Fit a single model with given prior scale."""
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)

        if treatment is not None:
            ate = pm.Normal("ate", mu=0, sigma=prior_scale)
            beta = pm.Normal("beta", mu=0, sigma=prior_scale, shape=X.shape[1])
            mu = alpha + ate * treatment + pm.math.dot(X, beta)
        else:
            beta = pm.Normal("beta", mu=0, sigma=prior_scale, shape=X.shape[1])
            mu = alpha + pm.math.dot(X, beta)

        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            random_seed=seed,
            progressbar=progressbar
        )

    return trace


def analyze_sensitivity(data, scales, target_param, draws, tune, seed, progressbar):
    """Run sensitivity analysis across prior scales."""
    results = []

    treatment = data.get("treatment")

    for i, scale in enumerate(scales):
        print(f"\nFitting model {i+1}/{len(scales)} with prior scale = {scale}")

        trace = fit_model(
            data["y"], data["X"], treatment,
            scale, draws, tune, seed, progressbar
        )

        # Extract target parameter samples
        if target_param:
            param_name = target_param
        elif treatment is not None:
            param_name = "ate"
        else:
            param_name = "beta"

        if param_name in trace.posterior:
            samples = trace.posterior[param_name].values.flatten()
        else:
            # Handle indexed parameters
            for key in trace.posterior.data_vars:
                if param_name in key:
                    samples = trace.posterior[key].values.flatten()
                    break
            else:
                raise ValueError(f"Parameter {param_name} not found in trace")

        # If samples are multidimensional, take first
        if samples.ndim > 1:
            samples = samples[:, 0]

        hdi = az.hdi(samples, hdi_prob=0.95)

        result = {
            "prior_scale": scale,
            "mean": float(samples.mean()),
            "median": float(np.median(samples)),
            "std": float(samples.std()),
            "hdi_2.5%": float(hdi[0]),
            "hdi_97.5%": float(hdi[1]),
            "prob_positive": float((samples > 0).mean()),
            "prob_negative": float((samples < 0).mean())
        }
        results.append(result)

        # Summary for this scale
        print(f"  Mean: {result['mean']:.4f}, SD: {result['std']:.4f}")
        print(f"  95% HDI: [{result['hdi_2.5%']:.4f}, {result['hdi_97.5%']:.4f}]")

    return pd.DataFrame(results)


def assess_robustness(results_df):
    """Assess sensitivity of results to prior specification."""
    assessment = {}

    # Range of point estimates
    mean_range = results_df["mean"].max() - results_df["mean"].min()
    assessment["mean_range"] = mean_range

    # Coefficient of variation of means
    mean_cv = results_df["mean"].std() / abs(results_df["mean"].mean()) if results_df["mean"].mean() != 0 else float('inf')
    assessment["mean_cv"] = mean_cv

    # Sign consistency
    all_positive = (results_df["prob_positive"] > 0.95).all()
    all_negative = (results_df["prob_negative"] > 0.95).all()
    assessment["sign_consistent"] = all_positive or all_negative

    # Overlapping HDIs
    min_upper = results_df["hdi_97.5%"].min()
    max_lower = results_df["hdi_2.5%"].max()
    assessment["hdis_overlap"] = min_upper > max_lower

    # Robustness score (0-1)
    score = 0
    if assessment["mean_cv"] < 0.1:
        score += 0.4
    elif assessment["mean_cv"] < 0.25:
        score += 0.2

    if assessment["sign_consistent"]:
        score += 0.3

    if assessment["hdis_overlap"]:
        score += 0.3

    assessment["robustness_score"] = score

    if score >= 0.8:
        assessment["interpretation"] = "ROBUST: Results are stable across prior specifications"
    elif score >= 0.5:
        assessment["interpretation"] = "MODERATE: Some sensitivity to prior choice"
    else:
        assessment["interpretation"] = "SENSITIVE: Results depend substantially on prior"

    return assessment


def print_results(results_df, assessment, target_param):
    """Print formatted sensitivity analysis results."""
    print("\n" + "=" * 70)
    print("PRIOR SENSITIVITY ANALYSIS")
    print("=" * 70)

    print(f"\nTarget Parameter: {target_param}")

    print("\n" + "-" * 50)
    print("RESULTS BY PRIOR SCALE")
    print("-" * 50)

    # Format dataframe for display
    display_df = results_df.copy()
    display_df = display_df.round(4)
    print(display_df.to_string(index=False))

    print("\n" + "-" * 50)
    print("ROBUSTNESS ASSESSMENT")
    print("-" * 50)

    print(f"  Range of posterior means: {assessment['mean_range']:.4f}")
    print(f"  Coefficient of variation: {assessment['mean_cv']:.4f}")
    print(f"  Sign consistent (95% prob): {assessment['sign_consistent']}")
    print(f"  HDIs overlap: {assessment['hdis_overlap']}")
    print(f"\n  Robustness Score: {assessment['robustness_score']:.2f} / 1.00")
    print(f"\n  {assessment['interpretation']}")

    print("=" * 70)


def create_sensitivity_plot(results_df, target_param, output_path):
    """Create sensitivity analysis visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Point estimates with HDI
    ax1 = axes[0]
    x = range(len(results_df))
    ax1.errorbar(
        x, results_df["mean"],
        yerr=[
            results_df["mean"] - results_df["hdi_2.5%"],
            results_df["hdi_97.5%"] - results_df["mean"]
        ],
        fmt='o', capsize=5, capthick=2, markersize=8,
        color='steelblue', ecolor='steelblue', alpha=0.7
    )
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s:.1f}" for s in results_df["prior_scale"]])
    ax1.set_xlabel("Prior Scale (SD)")
    ax1.set_ylabel(f"{target_param}")
    ax1.set_title("Posterior Mean with 95% HDI")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Probability positive
    ax2 = axes[1]
    colors = ['green' if p > 0.95 else 'red' if p < 0.05 else 'gray'
              for p in results_df["prob_positive"]]
    bars = ax2.bar(x, results_df["prob_positive"], color=colors, alpha=0.7)
    ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95%')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='5%')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s:.1f}" for s in results_df["prior_scale"]])
    ax2.set_xlabel("Prior Scale (SD)")
    ax2.set_ylabel("P(effect > 0)")
    ax2.set_title("Probability of Positive Effect")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Prior Sensitivity Analysis: {target_param}", fontsize=12, y=1.02)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved sensitivity plot to: {output_path}")


def main():
    """Main entry point."""
    if not HAS_DEPS:
        print("Error: Required packages not installed")
        print("Install with: pip install pymc arviz matplotlib pandas")
        sys.exit(1)

    args = parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(args.data, args.outcome, args.features, args.treatment)
    print(f"Loaded {len(data['y'])} observations")

    # Parse scales
    scales = [float(s.strip()) for s in args.scales.split(",")]
    print(f"Testing prior scales: {scales}")

    # Determine target parameter
    target_param = args.target_param
    if target_param is None:
        target_param = "ate" if args.treatment else "beta"

    # Run sensitivity analysis
    results_df = analyze_sensitivity(
        data, scales, target_param,
        args.draws, args.tune, args.seed,
        progressbar=not args.no_progress
    )

    # Assess robustness
    assessment = assess_robustness(results_df)

    # Print results
    print_results(results_df, assessment, target_param)

    # Save results
    if args.output:
        output = {
            "target_parameter": target_param,
            "scales_tested": scales,
            "results": results_df.to_dict(orient="records"),
            "assessment": assessment
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved results to: {args.output}")

    # Create plot
    if args.plot:
        create_sensitivity_plot(results_df, target_param, args.plot)


if __name__ == "__main__":
    main()
