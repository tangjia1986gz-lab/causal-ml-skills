#!/usr/bin/env python3
"""
Complete RD Analysis CLI Script.

This script provides a command-line interface for running comprehensive
Regression Discontinuity analysis following Cattaneo et al. methodology.

Usage:
    python run_rd_analysis.py data.csv --running score --outcome y --cutoff 0

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from rd_estimator import (
    validate_rd_data,
    mccrary_test,
    covariate_balance_rd,
    select_bandwidth,
    estimate_sharp_rd,
    estimate_fuzzy_rd,
    placebo_cutoff_test,
    bandwidth_sensitivity,
    donut_hole_rd,
    rd_plot
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run complete Regression Discontinuity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sharp RD
  python run_rd_analysis.py data.csv --running score --outcome earnings --cutoff 0

  # Fuzzy RD with treatment indicator
  python run_rd_analysis.py data.csv --running score --outcome earnings --cutoff 0 \\
      --treatment enrolled

  # With covariates and output directory
  python run_rd_analysis.py data.csv --running score --outcome earnings --cutoff 0 \\
      --covariates age gender income --output-dir results/

  # Custom bandwidth
  python run_rd_analysis.py data.csv --running score --outcome earnings --cutoff 0 \\
      --bandwidth 0.5
        """
    )

    # Required arguments
    parser.add_argument(
        "data_file",
        type=str,
        help="Path to CSV data file"
    )
    parser.add_argument(
        "--running",
        type=str,
        required=True,
        help="Name of the running variable column"
    )
    parser.add_argument(
        "--outcome",
        type=str,
        required=True,
        help="Name of the outcome variable column"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        required=True,
        help="Cutoff value for the running variable"
    )

    # Optional arguments
    parser.add_argument(
        "--treatment",
        type=str,
        default=None,
        help="Treatment indicator column (for fuzzy RD)"
    )
    parser.add_argument(
        "--covariates",
        type=str,
        nargs="+",
        default=None,
        help="List of covariate column names for balance tests"
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="Bandwidth (auto-selected if not specified)"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="triangular",
        choices=["triangular", "epanechnikov", "uniform"],
        help="Kernel function (default: triangular)"
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        choices=[1, 2],
        help="Polynomial order (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for results (default: current directory)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="markdown",
        choices=["markdown", "json", "latex"],
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )

    return parser.parse_args()


def load_data(data_file: str) -> pd.DataFrame:
    """Load data from CSV file."""
    if not Path(data_file).exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)
    return df


def run_analysis(
    data: pd.DataFrame,
    running: str,
    outcome: str,
    cutoff: float,
    treatment: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    bandwidth: Optional[float] = None,
    kernel: str = "triangular",
    order: int = 1,
    verbose: bool = False
) -> dict:
    """
    Run complete RD analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    running : str
        Running variable column name
    outcome : str
        Outcome variable column name
    cutoff : float
        Cutoff value
    treatment : str, optional
        Treatment column for fuzzy RD
    covariates : list, optional
        Covariates for balance tests
    bandwidth : float, optional
        Bandwidth (auto-selected if None)
    kernel : str
        Kernel function
    order : int
        Polynomial order
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Complete analysis results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "running": running,
            "outcome": outcome,
            "cutoff": cutoff,
            "treatment": treatment,
            "kernel": kernel,
            "order": order
        }
    }

    # Step 1: Validate data
    if verbose:
        print("Step 1: Validating data...")

    validation = validate_rd_data(
        data=data,
        running=running,
        outcome=outcome,
        cutoff=cutoff,
        treatment=treatment,
        bandwidth=bandwidth
    )

    results["validation"] = {
        "is_valid": validation.is_valid,
        "errors": validation.errors,
        "warnings": validation.warnings,
        "summary": validation.summary
    }

    if not validation.is_valid:
        print(f"ERROR: Data validation failed: {validation.errors}")
        return results

    # Step 2: McCrary test
    if verbose:
        print("Step 2: Running McCrary density test...")

    mccrary_result = mccrary_test(data[running], cutoff)

    results["mccrary_test"] = {
        "statistic": float(mccrary_result.statistic) if not np.isnan(mccrary_result.statistic) else None,
        "p_value": float(mccrary_result.p_value) if not np.isnan(mccrary_result.p_value) else None,
        "passed": mccrary_result.passed,
        "interpretation": mccrary_result.interpretation
    }

    # Step 3: Covariate balance
    if covariates:
        if verbose:
            print(f"Step 3: Testing covariate balance ({len(covariates)} covariates)...")

        balance_results = covariate_balance_rd(
            data=data,
            running=running,
            cutoff=cutoff,
            covariates=covariates,
            bandwidth=bandwidth
        )

        results["covariate_balance"] = {}
        for cov, res in balance_results.items():
            results["covariate_balance"][cov] = {
                "discontinuity": float(res.statistic) if not np.isnan(res.statistic) else None,
                "p_value": float(res.p_value) if not np.isnan(res.p_value) else None,
                "passed": res.passed
            }

    # Step 4: Bandwidth selection
    if verbose:
        print("Step 4: Selecting bandwidth...")

    x = data[running].values
    y = data[outcome].values
    mask = ~(np.isnan(x) | np.isnan(y))

    if bandwidth is None:
        bandwidth = select_bandwidth(x[mask], y[mask], cutoff, method="mserd", kernel=kernel, order=order)

    results["bandwidth"] = {
        "value": float(bandwidth),
        "method": "mserd" if bandwidth else "user-specified"
    }

    if verbose:
        print(f"   Bandwidth: {bandwidth:.4f}")

    # Step 5: Main estimation
    if verbose:
        print("Step 5: Estimating main effect...")

    if treatment:
        main_result = estimate_fuzzy_rd(
            data=data,
            running=running,
            outcome=outcome,
            treatment=treatment,
            cutoff=cutoff,
            bandwidth=bandwidth,
            kernel=kernel,
            order=order
        )
        design_type = "fuzzy"
    else:
        main_result = estimate_sharp_rd(
            data=data,
            running=running,
            outcome=outcome,
            cutoff=cutoff,
            bandwidth=bandwidth,
            kernel=kernel,
            order=order
        )
        design_type = "sharp"

    results["main_estimate"] = {
        "design_type": design_type,
        "effect": float(main_result.effect),
        "se": float(main_result.se),
        "ci_lower": float(main_result.ci_lower),
        "ci_upper": float(main_result.ci_upper),
        "p_value": float(main_result.p_value),
        "n_left": main_result.diagnostics.get("n_left"),
        "n_right": main_result.diagnostics.get("n_right"),
        "n_effective": main_result.diagnostics.get("n_effective")
    }

    if treatment:
        results["main_estimate"]["first_stage"] = float(main_result.diagnostics.get("first_stage", 0))
        results["main_estimate"]["reduced_form"] = float(main_result.diagnostics.get("reduced_form", 0))

    # Step 6: Placebo tests
    if verbose:
        print("Step 6: Running placebo cutoff tests...")

    x_clean = x[mask]
    below = x_clean[x_clean < cutoff]
    above = x_clean[x_clean >= cutoff]

    placebo_cutoffs = []
    if len(below) > 50:
        placebo_cutoffs.append(float(np.percentile(below, 50)))
    if len(above) > 50:
        placebo_cutoffs.append(float(np.percentile(above, 50)))

    if placebo_cutoffs:
        placebo_results = placebo_cutoff_test(
            data=data,
            running=running,
            outcome=outcome,
            true_cutoff=cutoff,
            placebo_cutoffs=placebo_cutoffs,
            bandwidth=bandwidth,
            kernel=kernel,
            order=order
        )

        results["placebo_tests"] = {}
        for pc, res in placebo_results.items():
            results["placebo_tests"][str(pc)] = {
                "effect": float(res.statistic) if not np.isnan(res.statistic) else None,
                "p_value": float(res.p_value) if not np.isnan(res.p_value) else None,
                "passed": res.passed
            }

    # Step 7: Bandwidth sensitivity
    if verbose:
        print("Step 7: Running bandwidth sensitivity analysis...")

    sensitivity = bandwidth_sensitivity(
        data=data,
        running=running,
        outcome=outcome,
        cutoff=cutoff,
        bandwidth_range=[0.5*bandwidth, 0.75*bandwidth, bandwidth, 1.25*bandwidth, 1.5*bandwidth, 2*bandwidth],
        kernel=kernel,
        order=order,
        treatment=treatment
    )

    results["bandwidth_sensitivity"] = []
    for r in sensitivity["results"]:
        results["bandwidth_sensitivity"].append({
            "bandwidth": float(r["bandwidth"]),
            "bw_ratio": float(r["bw_ratio"]),
            "effect": float(r["effect"]) if not np.isnan(r["effect"]) else None,
            "se": float(r["se"]) if not np.isnan(r["se"]) else None,
            "p_value": float(r["p_value"]) if not np.isnan(r["p_value"]) else None
        })

    # Step 8: Alternative specifications
    if verbose:
        print("Step 8: Running alternative specifications...")

    results["robustness"] = {}

    # Polynomial order sensitivity
    alt_order = 2 if order == 1 else 1
    if treatment:
        alt_result = estimate_fuzzy_rd(
            data, running, outcome, treatment, cutoff, bandwidth, kernel, alt_order
        )
    else:
        alt_result = estimate_sharp_rd(
            data, running, outcome, cutoff, bandwidth, kernel, alt_order
        )

    results["robustness"]["alt_polynomial"] = {
        "order": alt_order,
        "effect": float(alt_result.effect),
        "se": float(alt_result.se),
        "p_value": float(alt_result.p_value)
    }

    # Donut hole (if McCrary is marginal)
    if mccrary_result.p_value < 0.20:
        if verbose:
            print("   Running donut hole analysis (marginal McCrary result)...")

        try:
            donut_result = donut_hole_rd(
                data=data,
                running=running,
                outcome=outcome,
                cutoff=cutoff,
                bandwidth=bandwidth,
                donut_radius=bandwidth * 0.1,
                kernel=kernel,
                order=order,
                treatment=treatment
            )

            results["robustness"]["donut_hole"] = {
                "donut_radius": bandwidth * 0.1,
                "effect": float(donut_result.effect),
                "se": float(donut_result.se),
                "p_value": float(donut_result.p_value),
                "n_excluded": donut_result.diagnostics.get("n_excluded")
            }
        except Exception as e:
            results["robustness"]["donut_hole"] = {"error": str(e)}

    if verbose:
        print("\nAnalysis complete!")

    return results


def format_markdown(results: dict) -> str:
    """Format results as Markdown report."""
    lines = []

    lines.append("# Regression Discontinuity Analysis Report")
    lines.append("")
    lines.append(f"**Generated**: {results['timestamp']}")
    lines.append("")

    # Settings
    s = results["settings"]
    lines.append("## 1. Analysis Settings")
    lines.append("")
    lines.append(f"- **Running Variable**: `{s['running']}`")
    lines.append(f"- **Outcome Variable**: `{s['outcome']}`")
    lines.append(f"- **Cutoff**: {s['cutoff']}")
    lines.append(f"- **Design Type**: {'Fuzzy' if s['treatment'] else 'Sharp'}")
    if s['treatment']:
        lines.append(f"- **Treatment Variable**: `{s['treatment']}`")
    lines.append(f"- **Kernel**: {s['kernel']}")
    lines.append(f"- **Polynomial Order**: {s['order']}")
    lines.append("")

    # Validation
    v = results["validation"]
    lines.append("## 2. Data Validation")
    lines.append("")
    lines.append(f"**Status**: {'VALID' if v['is_valid'] else 'INVALID'}")
    if v["warnings"]:
        lines.append("")
        lines.append("**Warnings**:")
        for w in v["warnings"]:
            lines.append(f"- {w}")
    lines.append("")

    # McCrary test
    m = results["mccrary_test"]
    lines.append("## 3. Manipulation Test (McCrary)")
    lines.append("")
    lines.append(f"- **Result**: {'PASSED' if m['passed'] else 'FAILED'}")
    if m['statistic'] is not None:
        lines.append(f"- **Log Density Difference**: {m['statistic']:.4f}")
    if m['p_value'] is not None:
        lines.append(f"- **P-value**: {m['p_value']:.4f}")
    lines.append("")

    # Covariate balance
    if "covariate_balance" in results:
        lines.append("## 4. Covariate Balance at Cutoff")
        lines.append("")
        lines.append("| Covariate | Discontinuity | P-value | Status |")
        lines.append("|-----------|---------------|---------|--------|")
        for cov, res in results["covariate_balance"].items():
            disc = f"{res['discontinuity']:.4f}" if res['discontinuity'] else "N/A"
            pval = f"{res['p_value']:.4f}" if res['p_value'] else "N/A"
            status = "OK" if res["passed"] else "FAIL"
            lines.append(f"| {cov} | {disc} | {pval} | {status} |")
        lines.append("")

    # Main estimate
    e = results["main_estimate"]
    lines.append("## 5. Main Results")
    lines.append("")
    lines.append(f"### {e['design_type'].title()} RD Estimate")
    lines.append("")
    lines.append(f"- **Effect**: {e['effect']:.4f}")
    lines.append(f"- **Standard Error**: {e['se']:.4f}")
    lines.append(f"- **95% CI**: [{e['ci_lower']:.4f}, {e['ci_upper']:.4f}]")
    lines.append(f"- **P-value**: {e['p_value']:.4f}")
    lines.append(f"- **Bandwidth**: {results['bandwidth']['value']:.4f}")
    lines.append(f"- **Effective N**: {e['n_effective']}")
    if 'first_stage' in e:
        lines.append(f"- **First Stage**: {e['first_stage']:.4f}")
        lines.append(f"- **Reduced Form**: {e['reduced_form']:.4f}")
    lines.append("")

    # Placebo tests
    if "placebo_tests" in results:
        lines.append("## 6. Placebo Cutoff Tests")
        lines.append("")
        lines.append("| Placebo Cutoff | Effect | P-value | Status |")
        lines.append("|----------------|--------|---------|--------|")
        for pc, res in results["placebo_tests"].items():
            eff = f"{res['effect']:.4f}" if res['effect'] else "N/A"
            pval = f"{res['p_value']:.4f}" if res['p_value'] else "N/A"
            status = "PASS" if res["passed"] else "FAIL"
            lines.append(f"| {pc} | {eff} | {pval} | {status} |")
        lines.append("")

    # Bandwidth sensitivity
    lines.append("## 7. Bandwidth Sensitivity")
    lines.append("")
    lines.append("| Bandwidth | Ratio | Effect | SE | P-value |")
    lines.append("|-----------|-------|--------|-----|---------|")
    for r in results["bandwidth_sensitivity"]:
        eff = f"{r['effect']:.4f}" if r['effect'] else "N/A"
        se = f"{r['se']:.4f}" if r['se'] else "N/A"
        pval = f"{r['p_value']:.4f}" if r['p_value'] else "N/A"
        lines.append(f"| {r['bandwidth']:.4f} | {r['bw_ratio']:.2f}x | {eff} | {se} | {pval} |")
    lines.append("")

    # Robustness
    if "robustness" in results:
        lines.append("## 8. Robustness Checks")
        lines.append("")

        if "alt_polynomial" in results["robustness"]:
            alt = results["robustness"]["alt_polynomial"]
            lines.append(f"### Alternative Polynomial (Order = {alt['order']})")
            lines.append(f"- Effect: {alt['effect']:.4f} (SE: {alt['se']:.4f}, p = {alt['p_value']:.4f})")
            lines.append("")

        if "donut_hole" in results["robustness"]:
            d = results["robustness"]["donut_hole"]
            if "error" not in d:
                lines.append(f"### Donut Hole (Radius = {d['donut_radius']:.4f})")
                lines.append(f"- Effect: {d['effect']:.4f} (SE: {d['se']:.4f}, p = {d['p_value']:.4f})")
                lines.append(f"- Observations excluded: {d['n_excluded']}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by RD Analysis Script*")

    return "\n".join(lines)


def format_latex(results: dict) -> str:
    """Format results as LaTeX table."""
    e = results["main_estimate"]
    bw = results["bandwidth"]["value"]

    stars = ""
    if e["p_value"] < 0.01:
        stars = "***"
    elif e["p_value"] < 0.05:
        stars = "**"
    elif e["p_value"] < 0.1:
        stars = "*"

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Regression Discontinuity Estimates}
\label{tab:rd_results}
\begin{tabular}{lc}
\toprule
& (1) \\
& """ + e["design_type"].title() + r""" RD \\
\midrule
RD Effect & """ + f"{e['effect']:.4f}{stars}" + r""" \\
& (""" + f"{e['se']:.4f}" + r""") \\
& \\
95\% Robust CI & [""" + f"{e['ci_lower']:.4f}, {e['ci_upper']:.4f}" + r"""] \\
& \\
\midrule
Bandwidth & """ + f"{bw:.4f}" + r""" \\
Kernel & """ + results["settings"]["kernel"].title() + r""" \\
Polynomial Order & """ + str(results["settings"]["order"]) + r""" \\
N (effective) & """ + str(e["n_effective"]) + r""" \\
\midrule
McCrary p-value & """ + f"{results['mccrary_test']['p_value']:.4f}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Robust bias-corrected standard errors in parentheses.
\item *** p$<$0.01, ** p$<$0.05, * p$<$0.1
\end{tablenotes}
\end{table}
"""
    return latex


def main():
    """Main entry point."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.verbose:
        print(f"Loading data from {args.data_file}...")

    data = load_data(args.data_file)

    if args.verbose:
        print(f"   Loaded {len(data)} observations")

    # Run analysis
    results = run_analysis(
        data=data,
        running=args.running,
        outcome=args.outcome,
        cutoff=args.cutoff,
        treatment=args.treatment,
        covariates=args.covariates,
        bandwidth=args.bandwidth,
        kernel=args.kernel,
        order=args.order,
        verbose=args.verbose
    )

    # Save results
    if args.format == "json":
        output_file = output_dir / "rd_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

    elif args.format == "latex":
        output_file = output_dir / "rd_results.tex"
        latex_output = format_latex(results)
        with open(output_file, "w") as f:
            f.write(latex_output)
        print(f"LaTeX table saved to {output_file}")

    else:  # markdown
        output_file = output_dir / "rd_results.md"
        md_output = format_markdown(results)
        with open(output_file, "w") as f:
            f.write(md_output)
        print(f"Markdown report saved to {output_file}")

    # Generate plots
    if not args.no_plots:
        if args.verbose:
            print("\nGenerating RD plot...")

        try:
            fig = rd_plot(
                data=data,
                running=args.running,
                outcome=args.outcome,
                cutoff=args.cutoff,
                bandwidth=results["bandwidth"]["value"],
                n_bins=20,
                poly_order=args.order,
                ci=True
            )

            plot_file = output_dir / "rd_plot.png"
            fig.savefig(plot_file, dpi=150, bbox_inches="tight")
            print(f"RD plot saved to {plot_file}")

        except ImportError:
            print("Note: matplotlib not available, skipping plot generation")
        except Exception as e:
            print(f"Warning: Could not generate plot: {e}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Design: {results['main_estimate']['design_type'].title()} RD")
    print(f"Effect: {results['main_estimate']['effect']:.4f} "
          f"(SE = {results['main_estimate']['se']:.4f})")
    print(f"95% CI: [{results['main_estimate']['ci_lower']:.4f}, "
          f"{results['main_estimate']['ci_upper']:.4f}]")
    print(f"P-value: {results['main_estimate']['p_value']:.4f}")
    print(f"McCrary test: {'PASSED' if results['mccrary_test']['passed'] else 'FAILED'}")
    print("="*60)


if __name__ == "__main__":
    main()
