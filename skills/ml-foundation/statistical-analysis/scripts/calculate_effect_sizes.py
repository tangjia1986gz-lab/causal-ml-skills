#!/usr/bin/env python3
"""
Effect Size Calculator CLI

Calculate various effect sizes for two-group comparisons, correlations,
and categorical associations.

Usage:
    python calculate_effect_sizes.py --data data.csv --outcome y --group treatment --type cohens_d
    python calculate_effect_sizes.py --data data.csv --var1 education --var2 income --type correlation
    python calculate_effect_sizes.py --data data.csv --outcome success --exposure treatment --type odds_ratio
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from statistical_analysis import (
    calculate_cohens_d,
    calculate_hedges_g,
    calculate_glass_delta,
    calculate_correlation,
    calculate_point_biserial,
    calculate_odds_ratio,
    calculate_relative_risk,
    calculate_risk_difference,
    calculate_nnt,
    calculate_phi,
    calculate_cramers_v,
    calculate_eta_squared,
    calculate_omega_squared,
    calculate_cohens_f,
    convert_effect_size,
    EffectSizeResult
)


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV or other formats."""
    path = Path(filepath)

    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    elif path.suffix == '.dta':
        return pd.read_stata(path)
    elif path.suffix == '.parquet':
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_correlation(r: float) -> str:
    """Interpret correlation coefficient magnitude."""
    abs_r = abs(r)
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "small"
    elif abs_r < 0.5:
        return "medium"
    else:
        return "large"


def interpret_odds_ratio(or_val: float) -> str:
    """Interpret odds ratio."""
    if or_val < 0.5:
        return "strong protective effect"
    elif or_val < 0.7:
        return "moderate protective effect"
    elif or_val < 0.9:
        return "small protective effect"
    elif or_val <= 1.1:
        return "no substantial effect"
    elif or_val <= 1.5:
        return "small harmful effect"
    elif or_val <= 2.0:
        return "moderate harmful effect"
    else:
        return "strong harmful effect"


def format_result(result: EffectSizeResult, verbose: bool = True) -> str:
    """Format effect size result for output."""

    lines = []
    lines.append("=" * 60)
    lines.append(f"EFFECT SIZE: {result.effect_size_name}")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Value: {result.value:.4f}")

    if result.ci_lower is not None and result.ci_upper is not None:
        lines.append(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

    if result.se is not None:
        lines.append(f"Standard Error: {result.se:.4f}")

    lines.append("")
    lines.append(f"Interpretation: {result.interpretation}")

    # Additional statistics
    if hasattr(result, 'variance_explained') and result.variance_explained is not None:
        lines.append(f"Variance Explained: {result.variance_explained*100:.2f}%")

    if verbose and hasattr(result, 'details') and result.details:
        lines.append("")
        lines.append("Details:")
        for key, val in result.details.items():
            if isinstance(val, float):
                lines.append(f"  {key}: {val:.4f}")
            else:
                lines.append(f"  {key}: {val}")

    lines.append("=" * 60)

    return "\n".join(lines)


def calculate_multiple_effect_sizes(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    types: List[str]
) -> List[EffectSizeResult]:
    """Calculate multiple effect size types for same comparison."""

    results = []

    for effect_type in types:
        if effect_type == 'cohens_d':
            results.append(calculate_cohens_d(df, outcome, group))
        elif effect_type == 'hedges_g':
            results.append(calculate_hedges_g(df, outcome, group))
        elif effect_type == 'glass_delta':
            results.append(calculate_glass_delta(df, outcome, group))
        elif effect_type == 'point_biserial':
            results.append(calculate_point_biserial(df, outcome, group))

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Calculate effect sizes for statistical comparisons',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cohen's d for two-group comparison
  python calculate_effect_sizes.py --data data.csv --outcome income --group treatment --type cohens_d

  # Hedges' g (bias-corrected)
  python calculate_effect_sizes.py --data data.csv --outcome score --group treatment --type hedges_g

  # Multiple effect sizes
  python calculate_effect_sizes.py --data data.csv --outcome income --group treatment --type cohens_d,hedges_g,point_biserial

  # Correlation
  python calculate_effect_sizes.py --data data.csv --var1 education --var2 income --type correlation

  # Odds ratio for binary outcome
  python calculate_effect_sizes.py --data data.csv --outcome success --exposure treatment --type odds_ratio

  # Relative risk
  python calculate_effect_sizes.py --data data.csv --outcome success --exposure treatment --type relative_risk

  # Cramer's V for categorical association
  python calculate_effect_sizes.py --data data.csv --var1 education_level --var2 income_bracket --type cramers_v

Effect Size Types:
  Two-group (continuous): cohens_d, hedges_g, glass_delta, point_biserial
  Correlation: correlation, correlation_partial
  Categorical: odds_ratio, relative_risk, risk_difference, nnt, phi, cramers_v
  ANOVA: eta_squared, omega_squared, cohens_f
        """
    )

    # Data input
    parser.add_argument('--data', required=True, help='Path to data file')

    # Variable specification
    parser.add_argument('--outcome', help='Outcome variable (continuous)')
    parser.add_argument('--group', help='Grouping variable (for d-family)')
    parser.add_argument('--exposure', help='Exposure/treatment variable (for OR/RR)')
    parser.add_argument('--var1', help='First variable (for correlation/association)')
    parser.add_argument('--var2', help='Second variable (for correlation/association)')
    parser.add_argument('--control-value', type=int, default=0,
                       help='Value indicating control group (default: 0)')

    # Effect size type
    parser.add_argument('--type', required=True,
                       help='Effect size type(s), comma-separated')

    # Options
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level for CI (default: 0.95)')

    # Output
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', choices=['text', 'json', 'csv'], default='text',
                       help='Output format')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')

    # Conversion
    parser.add_argument('--convert-from', choices=['d', 'r', 'or', 'eta_sq', 'f'],
                       help='Convert from this effect size type')
    parser.add_argument('--convert-to', choices=['d', 'r', 'or', 'eta_sq', 'f'],
                       help='Convert to this effect size type')
    parser.add_argument('--value', type=float,
                       help='Value to convert')

    args = parser.parse_args()

    # Handle effect size conversion
    if args.convert_from and args.convert_to and args.value is not None:
        converted = convert_effect_size(
            args.value,
            from_type=args.convert_from,
            to_type=args.convert_to
        )
        print(f"Converted {args.convert_from}={args.value:.4f} to {args.convert_to}={converted:.4f}")
        return

    # Load data
    try:
        df = load_data(args.data)
        if not args.quiet:
            print(f"Loaded {len(df)} observations from {args.data}")
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse effect size types
    effect_types = [t.strip() for t in args.type.split(',')]

    results = []

    # Calculate requested effect sizes
    for effect_type in effect_types:
        try:
            if effect_type in ['cohens_d', 'hedges_g', 'glass_delta']:
                if not args.outcome or not args.group:
                    print(f"Error: {effect_type} requires --outcome and --group", file=sys.stderr)
                    sys.exit(1)

                if effect_type == 'cohens_d':
                    result = calculate_cohens_d(df, args.outcome, args.group)
                elif effect_type == 'hedges_g':
                    result = calculate_hedges_g(df, args.outcome, args.group)
                elif effect_type == 'glass_delta':
                    result = calculate_glass_delta(df, args.outcome, args.group, args.control_value)
                results.append(result)

            elif effect_type == 'point_biserial':
                if not args.outcome or not args.group:
                    print(f"Error: point_biserial requires --outcome and --group", file=sys.stderr)
                    sys.exit(1)
                result = calculate_point_biserial(df, args.outcome, args.group)
                results.append(result)

            elif effect_type == 'correlation':
                if not args.var1 or not args.var2:
                    print(f"Error: correlation requires --var1 and --var2", file=sys.stderr)
                    sys.exit(1)
                result = calculate_correlation(df, args.var1, args.var2)
                results.append(result)

            elif effect_type == 'odds_ratio':
                if not args.outcome or not args.exposure:
                    print(f"Error: odds_ratio requires --outcome and --exposure", file=sys.stderr)
                    sys.exit(1)
                result = calculate_odds_ratio(df, args.outcome, args.exposure)
                results.append(result)

            elif effect_type == 'relative_risk':
                if not args.outcome or not args.exposure:
                    print(f"Error: relative_risk requires --outcome and --exposure", file=sys.stderr)
                    sys.exit(1)
                result = calculate_relative_risk(df, args.outcome, args.exposure)
                results.append(result)

            elif effect_type == 'risk_difference':
                if not args.outcome or not args.exposure:
                    print(f"Error: risk_difference requires --outcome and --exposure", file=sys.stderr)
                    sys.exit(1)
                result = calculate_risk_difference(df, args.outcome, args.exposure)
                results.append(result)

            elif effect_type == 'nnt':
                if not args.outcome or not args.exposure:
                    print(f"Error: nnt requires --outcome and --exposure", file=sys.stderr)
                    sys.exit(1)
                rd_result = calculate_risk_difference(df, args.outcome, args.exposure)
                nnt = calculate_nnt(rd_result.value)
                results.append(EffectSizeResult(
                    value=nnt,
                    effect_size_name='Number Needed to Treat',
                    interpretation=f"NNT = {nnt:.1f}: Need to treat {nnt:.0f} patients to prevent one adverse event"
                ))

            elif effect_type == 'phi':
                if not args.var1 or not args.var2:
                    print(f"Error: phi requires --var1 and --var2", file=sys.stderr)
                    sys.exit(1)
                result = calculate_phi(df, args.var1, args.var2)
                results.append(result)

            elif effect_type == 'cramers_v':
                if not args.var1 or not args.var2:
                    print(f"Error: cramers_v requires --var1 and --var2", file=sys.stderr)
                    sys.exit(1)
                result = calculate_cramers_v(df, args.var1, args.var2)
                results.append(result)

            else:
                print(f"Unknown effect size type: {effect_type}", file=sys.stderr)
                continue

        except Exception as e:
            print(f"Error calculating {effect_type}: {e}", file=sys.stderr)
            continue

    # Output results
    if args.format == 'text':
        for result in results:
            print(format_result(result, verbose=not args.quiet))
            print()

    elif args.format == 'json':
        import json
        output = [r.to_dict() for r in results]
        print(json.dumps(output, indent=2))

    elif args.format == 'csv':
        print("effect_size_type,value,ci_lower,ci_upper,interpretation")
        for r in results:
            ci_l = r.ci_lower if r.ci_lower is not None else ''
            ci_u = r.ci_upper if r.ci_upper is not None else ''
            print(f"{r.effect_size_name},{r.value:.4f},{ci_l},{ci_u},\"{r.interpretation}\"")

    # Write to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            if args.format == 'text':
                for result in results:
                    f.write(format_result(result, verbose=not args.quiet))
                    f.write("\n\n")
            elif args.format == 'json':
                import json
                json.dump([r.to_dict() for r in results], f, indent=2)
            elif args.format == 'csv':
                f.write("effect_size_type,value,ci_lower,ci_upper,interpretation\n")
                for r in results:
                    ci_l = r.ci_lower if r.ci_lower is not None else ''
                    ci_u = r.ci_upper if r.ci_upper is not None else ''
                    f.write(f"{r.effect_size_name},{r.value:.4f},{ci_l},{ci_u},\"{r.interpretation}\"\n")
        print(f"Results written to {args.output}")


if __name__ == '__main__':
    main()
