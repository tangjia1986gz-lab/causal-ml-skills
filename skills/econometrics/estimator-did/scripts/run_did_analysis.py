#!/usr/bin/env python3
"""
Difference-in-Differences Analysis CLI Tool.

This script provides a command-line interface for running complete DID analyses,
including data validation, parallel trends testing, estimation, and reporting.

Usage:
    python run_did_analysis.py data.csv --outcome y --treatment treated --unit id --time year
    python run_did_analysis.py data.csv --config analysis_config.yaml

Examples:
    # Basic analysis
    python run_did_analysis.py data.csv -o employment -t treated -u firm_id -T year -tt 2015

    # Full analysis with controls and output
    python run_did_analysis.py data.csv -o employment -t treated -u firm_id -T year \\
        --treatment-time 2015 --controls size age --cluster state \\
        --output results/ --format all

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from did_estimator import (
    validate_did_data,
    test_parallel_trends,
    plot_parallel_trends,
    estimate_did_2x2,
    estimate_did_panel,
    estimate_did_staggered,
    event_study_plot,
    placebo_test,
    run_full_did_analysis
)


@dataclass
class AnalysisConfig:
    """Configuration for DID analysis."""
    data_path: str
    outcome: str
    treatment: str
    unit_id: str
    time_id: str
    treatment_time: Optional[int] = None
    treatment_group: Optional[str] = None
    controls: Optional[List[str]] = None
    cluster: Optional[str] = None
    staggered: bool = False
    control_group: str = "nevertreated"
    pre_periods: int = 4
    post_periods: int = 4
    run_placebo: bool = True
    placebo_lag: int = 2
    output_dir: Optional[str] = None
    output_format: str = "all"
    verbose: bool = True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Difference-in-Differences analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic analysis:
    python run_did_analysis.py data.csv -o y -t treated -u id -T year -tt 2015

  With controls and clustering:
    python run_did_analysis.py data.csv -o y -t treated -u id -T year \\
        --treatment-time 2015 --controls x1 x2 --cluster state

  Staggered DID:
    python run_did_analysis.py data.csv -o y -t first_treated -u id -T year \\
        --staggered --control-group nevertreated

  From config file:
    python run_did_analysis.py --config analysis_config.yaml
        """
    )

    # Data input
    parser.add_argument(
        "data_path",
        nargs="?",
        help="Path to data file (CSV, Parquet, or Stata)"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML/JSON config file"
    )

    # Required arguments (if not using config)
    parser.add_argument(
        "--outcome", "-o",
        help="Outcome variable name"
    )
    parser.add_argument(
        "--treatment", "-t",
        help="Treatment indicator variable"
    )
    parser.add_argument(
        "--unit", "-u",
        dest="unit_id",
        help="Unit identifier variable"
    )
    parser.add_argument(
        "--time", "-T",
        dest="time_id",
        help="Time period variable"
    )

    # Optional arguments
    parser.add_argument(
        "--treatment-time", "-tt",
        type=int,
        help="Treatment start time (single timing)"
    )
    parser.add_argument(
        "--treatment-group", "-tg",
        help="Ever-treated indicator variable"
    )
    parser.add_argument(
        "--controls",
        nargs="+",
        help="Control variables to include"
    )
    parser.add_argument(
        "--cluster",
        help="Variable to cluster standard errors on"
    )

    # Staggered DID options
    parser.add_argument(
        "--staggered",
        action="store_true",
        help="Use staggered DID estimator (Callaway-Sant'Anna)"
    )
    parser.add_argument(
        "--control-group",
        choices=["nevertreated", "notyettreated"],
        default="nevertreated",
        help="Control group type for staggered DID"
    )

    # Analysis options
    parser.add_argument(
        "--pre-periods",
        type=int,
        default=4,
        help="Number of pre-treatment periods for event study"
    )
    parser.add_argument(
        "--post-periods",
        type=int,
        default=4,
        help="Number of post-treatment periods for event study"
    )
    parser.add_argument(
        "--no-placebo",
        action="store_true",
        help="Skip placebo test"
    )
    parser.add_argument(
        "--placebo-lag",
        type=int,
        default=2,
        help="Periods before treatment for placebo test"
    )

    # Output options
    parser.add_argument(
        "--output", "-O",
        dest="output_dir",
        help="Output directory for results"
    )
    parser.add_argument(
        "--format", "-f",
        dest="output_format",
        choices=["text", "json", "latex", "markdown", "all"],
        default="text",
        help="Output format"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    return parser.parse_args()


def load_config(config_path: str) -> AnalysisConfig:
    """Load analysis configuration from YAML or JSON file."""
    path = Path(config_path)

    if path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path) as f:
                config_dict = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files")
    elif path.suffix == '.json':
        with open(path) as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    return AnalysisConfig(**config_dict)


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from various file formats."""
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from {data_path}")

    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix == '.parquet':
        return pd.read_parquet(path)
    elif path.suffix == '.dta':
        return pd.read_stata(path)
    elif path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    else:
        # Try CSV as default
        logger.warning(f"Unknown file extension {path.suffix}, trying CSV")
        return pd.read_csv(path)


def create_output_dir(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    output_format: str,
    config: AnalysisConfig
) -> None:
    """Save analysis results to specified format(s)."""
    formats = [output_format] if output_format != "all" else ["text", "json", "latex", "markdown"]

    for fmt in formats:
        if fmt == "text":
            with open(output_dir / "results.txt", "w") as f:
                f.write(results.get('summary', ''))

        elif fmt == "json":
            # Convert results to JSON-serializable format
            json_results = {
                'config': asdict(config),
                'main_estimate': {
                    'effect': results['main']['effect'],
                    'se': results['main']['se'],
                    'ci_lower': results['main']['ci_lower'],
                    'ci_upper': results['main']['ci_upper'],
                    'p_value': results['main']['p_value']
                },
                'parallel_trends': {
                    'passed': results['parallel_trends'].passed,
                    'p_value': results['parallel_trends'].p_value
                }
            }
            with open(output_dir / "results.json", "w") as f:
                json.dump(json_results, f, indent=2, default=str)

        elif fmt == "latex":
            with open(output_dir / "table.tex", "w") as f:
                f.write(results.get('latex_table', ''))

        elif fmt == "markdown":
            with open(output_dir / "results.md", "w") as f:
                f.write(generate_markdown_report(results, config))

    logger.info(f"Results saved to {output_dir}")


def generate_markdown_report(results: Dict[str, Any], config: AnalysisConfig) -> str:
    """Generate a markdown report of the analysis."""
    main = results['main']
    trends = results['parallel_trends']

    report = f"""# Difference-in-Differences Analysis Report

## Analysis Configuration

| Parameter | Value |
|-----------|-------|
| Outcome | `{config.outcome}` |
| Treatment | `{config.treatment}` |
| Unit ID | `{config.unit_id}` |
| Time ID | `{config.time_id}` |
| Treatment Time | {config.treatment_time or 'Staggered'} |
| Controls | {', '.join(config.controls) if config.controls else 'None'} |
| Clustering | {config.cluster or 'Unit level'} |

## Main Results

### Treatment Effect Estimate

| Metric | Value |
|--------|-------|
| **Effect (ATT)** | **{main['effect']:.4f}** |
| Standard Error | {main['se']:.4f} |
| 95% CI Lower | {main['ci_lower']:.4f} |
| 95% CI Upper | {main['ci_upper']:.4f} |
| P-value | {main['p_value']:.4f} |

### Statistical Significance

The estimated treatment effect is {'**statistically significant**' if main['p_value'] < 0.05 else 'not statistically significant'} at the 5% level.

## Parallel Trends Analysis

| Test | Result |
|------|--------|
| Pre-trends test | {'**PASSED**' if trends.passed else '**FAILED**'} |
| Test statistic | {trends.statistic:.4f} |
| P-value | {trends.p_value:.4f} |

{trends.interpretation}

"""

    # Add placebo results if available
    if 'placebo' in results and results['placebo'] is not None:
        placebo = results['placebo']
        report += f"""
## Placebo Test

| Metric | Value |
|--------|-------|
| Placebo Effect | {placebo.statistic:.4f} |
| P-value | {placebo.p_value:.4f} |
| Result | {'**PASSED**' if placebo.passed else '**FAILED**'} |

{placebo.interpretation}

"""

    # Add validation summary
    if 'validation' in results:
        val = results['validation']
        report += f"""
## Data Summary

| Metric | Value |
|--------|-------|
| Number of Units | {val.get('n_units', 'N/A')} |
| Number of Periods | {val.get('n_periods', 'N/A')} |
| Total Observations | {val.get('n_obs', 'N/A')} |
| Balanced Panel | {val.get('balanced', 'N/A')} |

"""

    report += """
---
*Generated by DID Analysis CLI*
"""

    return report


def run_analysis(config: AnalysisConfig) -> Dict[str, Any]:
    """Run the full DID analysis pipeline."""
    results = {}

    # Load data
    data = load_data(config.data_path)
    logger.info(f"Loaded data: {len(data)} observations")

    # Validate data
    logger.info("Validating data structure...")
    validation = validate_did_data(
        data=data,
        outcome=config.outcome,
        treatment=config.treatment,
        unit_id=config.unit_id,
        time_id=config.time_id,
        treatment_group=config.treatment_group
    )

    if not validation.is_valid:
        logger.error(f"Data validation failed: {validation.errors}")
        raise ValueError(f"Data validation failed: {validation.errors}")

    if validation.warnings:
        for warning in validation.warnings:
            logger.warning(warning)

    results['validation'] = validation.summary

    # Infer treatment group if not provided
    if config.treatment_group is None:
        logger.info("Inferring treatment group from treatment indicator...")
        data['_treatment_group'] = data.groupby(config.unit_id)[config.treatment].transform('max')
        treatment_group = '_treatment_group'
    else:
        treatment_group = config.treatment_group

    # Infer treatment time if not provided and not staggered
    treatment_time = config.treatment_time
    if treatment_time is None and not config.staggered:
        treated_periods = data[data[config.treatment] == 1][config.time_id]
        if len(treated_periods) > 0:
            treatment_time = treated_periods.min()
            logger.info(f"Inferred treatment time: {treatment_time}")

    # Test parallel trends
    logger.info("Testing parallel trends assumption...")
    if treatment_time is not None:
        trends_result = test_parallel_trends(
            data=data,
            outcome=config.outcome,
            treatment_group=treatment_group,
            time_id=config.time_id,
            unit_id=config.unit_id,
            treatment_time=treatment_time,
            n_pre_periods=config.pre_periods
        )
        results['parallel_trends'] = trends_result
        logger.info(f"Parallel trends test: {'PASSED' if trends_result.passed else 'FAILED'} "
                   f"(p={trends_result.p_value:.4f})")
    else:
        logger.warning("Cannot test parallel trends without treatment_time")
        results['parallel_trends'] = None

    # Main estimation
    logger.info("Running main estimation...")
    cluster = config.cluster or config.unit_id

    if config.staggered:
        main_result = estimate_did_staggered(
            data=data,
            outcome=config.outcome,
            treatment_time=config.treatment,
            unit_id=config.unit_id,
            time_id=config.time_id,
            control_group=config.control_group,
            covariates=config.controls
        )
    else:
        main_result = estimate_did_panel(
            data=data,
            outcome=config.outcome,
            treatment=config.treatment,
            unit_id=config.unit_id,
            time_id=config.time_id,
            controls=config.controls,
            cluster=cluster
        )

    results['main'] = {
        'effect': main_result.effect,
        'se': main_result.se,
        'ci_lower': main_result.ci_lower,
        'ci_upper': main_result.ci_upper,
        'p_value': main_result.p_value,
        'diagnostics': main_result.diagnostics
    }

    logger.info(f"Main estimate: {main_result.effect:.4f} (SE={main_result.se:.4f})")

    # Placebo test
    if config.run_placebo and treatment_time is not None:
        logger.info("Running placebo test...")
        placebo_time = treatment_time - config.placebo_lag
        periods = sorted(data[config.time_id].unique())

        if placebo_time > periods[0]:
            try:
                placebo_result = placebo_test(
                    data=data,
                    outcome=config.outcome,
                    treatment_group=treatment_group,
                    unit_id=config.unit_id,
                    time_id=config.time_id,
                    actual_treatment_time=treatment_time,
                    placebo_treatment_time=placebo_time,
                    controls=config.controls
                )
                results['placebo'] = placebo_result
                logger.info(f"Placebo test: {'PASSED' if placebo_result.passed else 'FAILED'} "
                           f"(p={placebo_result.p_value:.4f})")
            except Exception as e:
                logger.warning(f"Placebo test failed: {e}")
                results['placebo'] = None
        else:
            logger.warning("Insufficient pre-treatment periods for placebo test")
            results['placebo'] = None
    else:
        results['placebo'] = None

    # Generate summary
    results['summary'] = main_result.summary_table

    # Generate LaTeX table
    results['latex_table'] = generate_latex_table(results, config)

    return results


def generate_latex_table(results: Dict[str, Any], config: AnalysisConfig) -> str:
    """Generate LaTeX table for results."""
    main = results['main']

    # Significance stars
    pval = main['p_value']
    if pval < 0.01:
        stars = "***"
    elif pval < 0.05:
        stars = "**"
    elif pval < 0.1:
        stars = "*"
    else:
        stars = ""

    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Difference-in-Differences Results}}
\\label{{tab:did_results}}
\\begin{{tabular}}{{lc}}
\\toprule
 & (1) \\\\
\\midrule
Treatment Effect & {main['effect']:.4f}{stars} \\\\
 & ({main['se']:.4f}) \\\\
 & \\\\
Controls & {'Yes' if config.controls else 'No'} \\\\
Unit FE & Yes \\\\
Time FE & Yes \\\\
Clustering & {config.cluster or config.unit_id} \\\\
 & \\\\
Observations & {results['validation'].get('n_obs', 'N/A')} \\\\
Pre-trend p-value & {results['parallel_trends'].p_value:.3f if results['parallel_trends'] else 'N/A'} \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item Notes: Robust standard errors clustered at the {config.cluster or 'unit'} level in parentheses.
\\item *** p<0.01, ** p<0.05, * p<0.1
\\end{{tablenotes}}
\\end{{table}}
"""
    return table


def print_results(results: Dict[str, Any], verbose: bool = True) -> None:
    """Print analysis results to console."""
    print("\n" + "=" * 70)
    print("DIFFERENCE-IN-DIFFERENCES ANALYSIS RESULTS")
    print("=" * 70)

    # Main estimate
    main = results['main']
    print(f"\nTreatment Effect (ATT): {main['effect']:.4f}")
    print(f"Standard Error: {main['se']:.4f}")
    print(f"95% CI: [{main['ci_lower']:.4f}, {main['ci_upper']:.4f}]")
    print(f"P-value: {main['p_value']:.4f}")

    # Significance
    if main['p_value'] < 0.01:
        sig = "*** (p < 0.01)"
    elif main['p_value'] < 0.05:
        sig = "** (p < 0.05)"
    elif main['p_value'] < 0.1:
        sig = "* (p < 0.1)"
    else:
        sig = "Not significant"
    print(f"Significance: {sig}")

    # Parallel trends
    if results.get('parallel_trends'):
        trends = results['parallel_trends']
        print(f"\nParallel Trends Test: {'PASSED' if trends.passed else 'FAILED'}")
        print(f"  Statistic: {trends.statistic:.4f}")
        print(f"  P-value: {trends.p_value:.4f}")

    # Placebo
    if results.get('placebo'):
        placebo = results['placebo']
        print(f"\nPlacebo Test: {'PASSED' if placebo.passed else 'FAILED'}")
        print(f"  Effect: {placebo.statistic:.4f}")
        print(f"  P-value: {placebo.p_value:.4f}")

    # Sample info
    if results.get('validation'):
        val = results['validation']
        print(f"\nSample:")
        print(f"  Units: {val.get('n_units', 'N/A')}")
        print(f"  Periods: {val.get('n_periods', 'N/A')}")
        print(f"  Observations: {val.get('n_obs', 'N/A')}")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Validate required arguments
        if not all([args.data_path, args.outcome, args.treatment, args.unit_id, args.time_id]):
            print("Error: Must provide either --config or all required arguments")
            print("Required: data_path, --outcome, --treatment, --unit, --time")
            sys.exit(1)

        config = AnalysisConfig(
            data_path=args.data_path,
            outcome=args.outcome,
            treatment=args.treatment,
            unit_id=args.unit_id,
            time_id=args.time_id,
            treatment_time=args.treatment_time,
            treatment_group=args.treatment_group,
            controls=args.controls,
            cluster=args.cluster,
            staggered=args.staggered,
            control_group=args.control_group,
            pre_periods=args.pre_periods,
            post_periods=args.post_periods,
            run_placebo=not args.no_placebo,
            placebo_lag=args.placebo_lag,
            output_dir=args.output_dir,
            output_format=args.output_format,
            verbose=not args.quiet
        )

    # Set logging level
    if not config.verbose:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        # Run analysis
        results = run_analysis(config)

        # Print results
        if config.verbose:
            print_results(results)

        # Save results
        if config.output_dir:
            output_path = create_output_dir(config.output_dir)
            save_results(results, output_path, config.output_format, config)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
