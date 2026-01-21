#!/usr/bin/env python
"""
MCMC Diagnostics Script

Generate diagnostic plots and statistics for MCMC samples.
Checks convergence (Rhat, ESS), trace plots, and posterior predictive checks.

Usage:
    python diagnose_mcmc.py --trace trace.nc --output diagnostics/
    python diagnose_mcmc.py --trace trace.nc --vars beta,sigma --format pdf
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import warnings

try:
    import arviz as az
    import matplotlib.pyplot as plt
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate MCMC diagnostic plots and statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic diagnostics
  python diagnose_mcmc.py --trace model_trace.nc

  # Specific variables with output directory
  python diagnose_mcmc.py --trace trace.nc --vars alpha,beta --output ./diagnostics

  # Different plot formats
  python diagnose_mcmc.py --trace trace.nc --format pdf --dpi 300
        """
    )

    parser.add_argument(
        "--trace", "-t",
        required=True,
        help="Path to trace file (NetCDF format from ArviZ)"
    )
    parser.add_argument(
        "--vars", "-v",
        default=None,
        help="Comma-separated list of variables to diagnose (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="Output directory for plots (default: current directory)"
    )
    parser.add_argument(
        "--format", "-f",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for plots (default: png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for raster formats (default: 150)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only print statistics, no plots"
    )
    parser.add_argument(
        "--hdi-prob",
        type=float,
        default=0.95,
        help="Probability for HDI (default: 0.95)"
    )

    return parser.parse_args()


def load_trace(path: str):
    """Load trace from NetCDF file."""
    return az.from_netcdf(path)


def compute_diagnostics(trace, var_names=None):
    """Compute comprehensive MCMC diagnostics."""
    results = {
        "convergence": {},
        "efficiency": {},
        "issues": []
    }

    # Rhat
    rhat = az.rhat(trace, var_names=var_names)
    results["rhat"] = {}
    max_rhat = 0
    for k, v in rhat.items():
        if hasattr(v, 'values'):
            val = float(v.values.max())
        else:
            val = float(v)
        results["rhat"][k] = val
        max_rhat = max(max_rhat, val)

    results["convergence"]["max_rhat"] = max_rhat
    results["convergence"]["rhat_ok"] = max_rhat < 1.01
    if max_rhat >= 1.01:
        results["issues"].append(f"High Rhat ({max_rhat:.3f}) - chains may not have converged")

    # ESS bulk
    ess_bulk = az.ess(trace, var_names=var_names, method="bulk")
    results["ess_bulk"] = {}
    min_bulk = float('inf')
    for k, v in ess_bulk.items():
        if hasattr(v, 'values'):
            val = float(v.values.min())
        else:
            val = float(v)
        results["ess_bulk"][k] = val
        min_bulk = min(min_bulk, val)

    results["efficiency"]["min_bulk_ess"] = min_bulk
    if min_bulk < 400:
        results["issues"].append(f"Low bulk ESS ({min_bulk:.0f}) - increase draws")

    # ESS tail
    ess_tail = az.ess(trace, var_names=var_names, method="tail")
    results["ess_tail"] = {}
    min_tail = float('inf')
    for k, v in ess_tail.items():
        if hasattr(v, 'values'):
            val = float(v.values.min())
        else:
            val = float(v)
        results["ess_tail"][k] = val
        min_tail = min(min_tail, val)

    results["efficiency"]["min_tail_ess"] = min_tail
    if min_tail < 400:
        results["issues"].append(f"Low tail ESS ({min_tail:.0f}) - tail estimates unreliable")

    # Divergences
    if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
        n_div = int(trace.sample_stats.diverging.sum().values)
        results["convergence"]["divergences"] = n_div
        if n_div > 0:
            results["issues"].append(f"{n_div} divergent transitions - increase target_accept or reparameterize")
    else:
        results["convergence"]["divergences"] = 0

    # MCSE
    mcse = az.mcse(trace, var_names=var_names)
    results["mcse"] = {}
    for k, v in mcse.items():
        if hasattr(v, 'values'):
            results["mcse"][k] = float(v.values.mean())
        else:
            results["mcse"][k] = float(v)

    # Overall assessment
    results["convergence"]["all_ok"] = (
        results["convergence"]["rhat_ok"] and
        min_bulk >= 400 and
        min_tail >= 400 and
        results["convergence"]["divergences"] == 0
    )

    return results


def print_diagnostics(results):
    """Print formatted diagnostic report."""
    print("\n" + "=" * 60)
    print("MCMC DIAGNOSTIC REPORT")
    print("=" * 60)

    # Convergence
    print("\n" + "-" * 40)
    print("CONVERGENCE (Rhat)")
    print("-" * 40)
    for k, v in results["rhat"].items():
        status = "[OK]" if v < 1.01 else "[WARNING]"
        print(f"  {k}: {v:.4f} {status}")
    print(f"\n  Max Rhat: {results['convergence']['max_rhat']:.4f}")
    print(f"  Convergence: {'PASSED' if results['convergence']['rhat_ok'] else 'FAILED'}")

    # Efficiency
    print("\n" + "-" * 40)
    print("EFFICIENCY (ESS)")
    print("-" * 40)
    print("Bulk ESS (for mean estimation):")
    for k, v in results["ess_bulk"].items():
        status = "[OK]" if v >= 400 else "[LOW]"
        print(f"  {k}: {v:.0f} {status}")

    print("\nTail ESS (for quantile estimation):")
    for k, v in results["ess_tail"].items():
        status = "[OK]" if v >= 400 else "[LOW]"
        print(f"  {k}: {v:.0f} {status}")

    # MCSE
    print("\n" + "-" * 40)
    print("MONTE CARLO STANDARD ERROR")
    print("-" * 40)
    for k, v in results["mcse"].items():
        print(f"  {k}: {v:.6f}")

    # Divergences
    print("\n" + "-" * 40)
    print("SAMPLING ISSUES")
    print("-" * 40)
    div = results["convergence"]["divergences"]
    print(f"  Divergent transitions: {div} {'[OK]' if div == 0 else '[WARNING]'}")

    # Issues
    if results["issues"]:
        print("\n" + "-" * 40)
        print("WARNINGS")
        print("-" * 40)
        for issue in results["issues"]:
            print(f"  ! {issue}")

    # Overall
    print("\n" + "=" * 60)
    if results["convergence"]["all_ok"]:
        print("OVERALL: ALL DIAGNOSTICS PASSED")
    else:
        print("OVERALL: DIAGNOSTICS FAILED - SEE WARNINGS ABOVE")
        print("\nRECOMMENDATIONS:")
        if not results["convergence"]["rhat_ok"]:
            print("  - Run longer chains (increase tune and/or draws)")
            print("  - Check for multimodality")
        if results["efficiency"]["min_bulk_ess"] < 400:
            print("  - Increase number of draws")
        if results["efficiency"]["min_tail_ess"] < 400:
            print("  - Increase draws for reliable tail estimates")
        if results["convergence"]["divergences"] > 0:
            print("  - Increase target_accept (try 0.95 or 0.99)")
            print("  - Use non-centered parameterization")
            print("  - Add more informative priors")
    print("=" * 60)


def create_trace_plot(trace, var_names, output_dir, fmt, dpi):
    """Create trace plots."""
    fig = plt.figure(figsize=(12, 3 * len(var_names) if var_names else 12))
    az.plot_trace(trace, var_names=var_names, combined=False)
    plt.tight_layout()
    path = Path(output_dir) / f"trace_plot.{fmt}"
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def create_rank_plot(trace, var_names, output_dir, fmt, dpi):
    """Create rank plots (more robust than trace plots)."""
    fig, axes = plt.subplots(len(var_names) if var_names else 4, 1,
                             figsize=(10, 3 * (len(var_names) if var_names else 4)))
    az.plot_rank(trace, var_names=var_names, ax=axes)
    plt.tight_layout()
    path = Path(output_dir) / f"rank_plot.{fmt}"
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def create_autocorr_plot(trace, var_names, output_dir, fmt, dpi):
    """Create autocorrelation plots."""
    fig = plt.figure(figsize=(12, 3 * (len(var_names) if var_names else 4)))
    az.plot_autocorr(trace, var_names=var_names, combined=True)
    plt.tight_layout()
    path = Path(output_dir) / f"autocorr_plot.{fmt}"
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def create_posterior_plot(trace, var_names, output_dir, fmt, dpi, hdi_prob):
    """Create posterior distribution plots."""
    fig = plt.figure(figsize=(12, 3 * (len(var_names) if var_names else 4)))
    az.plot_posterior(trace, var_names=var_names, hdi_prob=hdi_prob)
    plt.tight_layout()
    path = Path(output_dir) / f"posterior_plot.{fmt}"
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def create_energy_plot(trace, output_dir, fmt, dpi):
    """Create energy plot (NUTS diagnostic)."""
    if hasattr(trace, 'sample_stats') and 'energy' in trace.sample_stats:
        fig, ax = plt.subplots(figsize=(8, 4))
        az.plot_energy(trace, ax=ax)
        path = Path(output_dir) / f"energy_plot.{fmt}"
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def create_pair_plot(trace, var_names, output_dir, fmt, dpi):
    """Create pair plot for correlations."""
    if var_names and len(var_names) > 1 and len(var_names) <= 6:
        fig = plt.figure(figsize=(10, 10))
        az.plot_pair(trace, var_names=var_names, kind="kde",
                    divergences=True, marginals=True)
        path = Path(output_dir) / f"pair_plot.{fmt}"
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def create_all_plots(trace, var_names, output_dir, fmt, dpi, hdi_prob):
    """Create all diagnostic plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating diagnostic plots in {output_path}/")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            create_trace_plot(trace, var_names, output_dir, fmt, dpi)
        except Exception as e:
            print(f"  Warning: Could not create trace plot: {e}")

        try:
            create_posterior_plot(trace, var_names, output_dir, fmt, dpi, hdi_prob)
        except Exception as e:
            print(f"  Warning: Could not create posterior plot: {e}")

        try:
            create_autocorr_plot(trace, var_names, output_dir, fmt, dpi)
        except Exception as e:
            print(f"  Warning: Could not create autocorr plot: {e}")

        try:
            create_energy_plot(trace, output_dir, fmt, dpi)
        except Exception as e:
            print(f"  Warning: Could not create energy plot: {e}")

        # Rank and pair plots can be memory intensive
        try:
            if var_names and len(var_names) <= 8:
                create_rank_plot(trace, var_names, output_dir, fmt, dpi)
        except Exception as e:
            print(f"  Warning: Could not create rank plot: {e}")

        try:
            create_pair_plot(trace, var_names, output_dir, fmt, dpi)
        except Exception as e:
            print(f"  Warning: Could not create pair plot: {e}")


def main():
    """Main entry point."""
    if not HAS_DEPS:
        print("Error: Required packages not installed")
        print("Install with: pip install arviz matplotlib")
        sys.exit(1)

    args = parse_args()

    # Load trace
    print(f"Loading trace from {args.trace}...")
    try:
        trace = load_trace(args.trace)
    except Exception as e:
        print(f"Error loading trace: {e}")
        sys.exit(1)

    # Parse variable names
    var_names = None
    if args.vars:
        var_names = [v.strip() for v in args.vars.split(",")]
        print(f"Analyzing variables: {var_names}")

    # Compute diagnostics
    print("Computing diagnostics...")
    results = compute_diagnostics(trace, var_names)

    # Print report
    print_diagnostics(results)

    # Create plots
    if not args.no_plots:
        create_all_plots(
            trace, var_names,
            args.output, args.format, args.dpi, args.hdi_prob
        )

    # Summary table
    print("\nParameter Summary:")
    summary = az.summary(trace, var_names=var_names, hdi_prob=args.hdi_prob)
    print(summary.to_string())


if __name__ == "__main__":
    main()
