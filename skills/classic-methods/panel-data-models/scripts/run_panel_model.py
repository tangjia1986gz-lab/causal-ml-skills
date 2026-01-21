#!/usr/bin/env python
"""
Panel Model CLI - Fit fixed effects, random effects, and dynamic panel models.

Usage:
    python run_panel_model.py data.csv --entity firm_id --time year --y revenue --x treatment size
    python run_panel_model.py data.csv --entity firm_id --time year --y revenue --x treatment --model twfe
    python run_panel_model.py data.csv --entity firm_id --time year --y revenue --x treatment --cluster firm_id

Author: Causal ML Skills
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from panel_estimator import PanelEstimator, run_panel_analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description='Panel data model estimation CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic fixed effects
  python run_panel_model.py data.csv --entity firm_id --time year --y revenue --x treatment size

  # Two-way fixed effects with clustered SE
  python run_panel_model.py data.csv --entity firm_id --time year --y revenue --x treatment --model twfe --cluster firm_id

  # Random effects model
  python run_panel_model.py data.csv --entity firm_id --time year --y revenue --x treatment --model re

  # Dynamic panel (Arellano-Bond)
  python run_panel_model.py data.csv --entity firm_id --time year --y revenue --x treatment --model dynamic --lags 1

  # Full analysis with Hausman test
  python run_panel_model.py data.csv --entity firm_id --time year --y revenue --x treatment --full-analysis
        """
    )

    parser.add_argument('data_file', type=str, help='Path to CSV data file')
    parser.add_argument('--entity', required=True, help='Entity/panel identifier column')
    parser.add_argument('--time', required=True, help='Time period column')
    parser.add_argument('--y', required=True, help='Dependent variable column')
    parser.add_argument('--x', nargs='+', required=True, help='Independent variable columns')

    parser.add_argument(
        '--model',
        choices=['fe', 'twfe', 're', 'dynamic', 'pooled'],
        default='fe',
        help='Model type: fe (entity FE), twfe (two-way FE), re (random effects), dynamic (Arellano-Bond), pooled'
    )

    parser.add_argument('--cluster', type=str, help='Column to cluster standard errors on')
    parser.add_argument(
        '--cluster-method',
        choices=['stata', 'robust', 'bootstrap', 'wild'],
        default='stata',
        help='Clustering method'
    )
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='Bootstrap iterations')

    parser.add_argument('--lags', type=int, default=1, help='Lags for dynamic panel')
    parser.add_argument(
        '--gmm-method',
        choices=['arellano_bond', 'blundell_bond'],
        default='arellano_bond',
        help='GMM method for dynamic panels'
    )

    parser.add_argument('--treatment', type=str, help='Treatment column for TWFE diagnostics')
    parser.add_argument('--full-analysis', action='store_true', help='Run comprehensive analysis')
    parser.add_argument('--hausman', action='store_true', help='Run Hausman test')
    parser.add_argument('--bacon', action='store_true', help='Run Goodman-Bacon decomposition')

    parser.add_argument('--output', type=str, help='Output file for results (CSV)')
    parser.add_argument('--latex', type=str, help='Output file for LaTeX table')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    if not args.quiet:
        print(f"Loading data from {args.data_file}...")

    try:
        data = pd.read_csv(args.data_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.data_file}")
        sys.exit(1)

    # Validate columns
    required_cols = [args.entity, args.time, args.y] + args.x
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        print(f"Available columns: {list(data.columns)}")
        sys.exit(1)

    # Initialize estimator
    estimator = PanelEstimator(
        data=data,
        entity_col=args.entity,
        time_col=args.time,
        y_col=args.y,
        x_cols=args.x
    )

    if not args.quiet:
        print(f"\nPanel structure:")
        print(f"  Entities: {estimator.n_entities:,}")
        print(f"  Time periods: {estimator.n_times}")
        print(f"  Observations: {estimator.n_obs:,}")
        print(f"  Balanced: {estimator.is_balanced}")

    # Run full analysis if requested
    if args.full_analysis:
        if not args.quiet:
            print("\n" + "=" * 70)
            print("COMPREHENSIVE PANEL ANALYSIS")
            print("=" * 70)

        results = run_panel_analysis(
            data, args.entity, args.time, args.y, args.x,
            treatment_col=args.treatment
        )

        print("\n--- Entity Fixed Effects ---")
        print(results['fe'].summary())

        print("\n--- Two-Way Fixed Effects ---")
        print(results['fe_twoway'].summary())

        print("\n--- Random Effects ---")
        print(results['re'].summary())

        print("\n--- Hausman Test ---")
        print(results['hausman'])

        print("\n--- Mundlak Test ---")
        for k, v in results['mundlak'].items():
            print(f"  {k}: {v}")

        print(f"\n--- Recommendation ---")
        print(f"  {results['recommendation']}")

        if args.treatment and 'bacon_decomp' in results:
            print("\n--- Goodman-Bacon Decomposition ---")
            bacon = results['bacon_decomp']
            print(f"  TWFE estimate: {bacon['twfe_estimate']:.4f}")
            print(f"  Number of comparisons: {bacon['n_comparisons']}")
            print(f"  Negative weight share: {bacon['negative_weight_share']:.2%}")
            if bacon['warning']:
                print(f"  WARNING: {bacon['warning']}")

        return

    # Fit specified model
    if not args.quiet:
        print(f"\nFitting {args.model.upper()} model...")

    if args.model == 'fe':
        result = estimator.fit_fixed_effects(entity_effects=True, time_effects=False)
    elif args.model == 'twfe':
        result = estimator.fit_fixed_effects(entity_effects=True, time_effects=True)
    elif args.model == 're':
        result = estimator.fit_random_effects()
    elif args.model == 'dynamic':
        result = estimator.fit_dynamic_panel(
            lags=args.lags,
            method=args.gmm_method
        )
    elif args.model == 'pooled':
        result = estimator.fit_fixed_effects(entity_effects=False, time_effects=False)

    # Apply clustered standard errors
    if args.cluster:
        if not args.quiet:
            print(f"Computing clustered standard errors (cluster: {args.cluster}, method: {args.cluster_method})...")

        result = estimator.cluster_robust_inference(
            result,
            cluster_col=args.cluster,
            method=args.cluster_method,
            n_bootstrap=args.n_bootstrap
        )

    # Print results
    if not args.quiet:
        print(result.summary())

    # Hausman test
    if args.hausman:
        if not args.quiet:
            print("\n--- Hausman Specification Test ---")
        hausman = estimator.hausman_test()
        print(hausman)

    # Bacon decomposition
    if args.bacon and args.treatment:
        if not args.quiet:
            print("\n--- Goodman-Bacon Decomposition ---")
        bacon = estimator.goodman_bacon_decomposition(args.treatment)
        print(f"TWFE estimate: {bacon['twfe_estimate']:.4f}")
        print(f"Number of 2x2 comparisons: {bacon['n_comparisons']}")
        print(f"Negative weight share: {bacon['negative_weight_share']:.2%}")
        if bacon['warning']:
            print(f"WARNING: {bacon['warning']}")

    # Save outputs
    if args.output:
        # Save coefficients to CSV
        output_df = pd.DataFrame({
            'variable': result.coefficients.index,
            'coefficient': result.coefficients.values,
            'std_error': result.std_errors.values,
            't_stat': result.t_stats.values,
            'p_value': result.p_values.values,
            'ci_lower': result.ci_lower.values,
            'ci_upper': result.ci_upper.values,
        })
        output_df.to_csv(args.output, index=False)
        if not args.quiet:
            print(f"\nResults saved to {args.output}")

    if args.latex:
        # Generate LaTeX table
        latex = generate_latex_table(result, args.model)
        with open(args.latex, 'w') as f:
            f.write(latex)
        if not args.quiet:
            print(f"LaTeX table saved to {args.latex}")


def generate_latex_table(result, model_type):
    """Generate LaTeX regression table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Panel Data Regression Results}",
        r"\label{tab:panel_results}",
        r"\begin{tabular}{lcccc}",
        r"\hline\hline",
        r"Variable & Coefficient & Std. Error & t-stat & p-value \\",
        r"\hline",
    ]

    for var in result.coefficients.index:
        coef = result.coefficients[var]
        se = result.std_errors[var]
        t = result.t_stats[var]
        p = result.p_values[var]

        stars = ""
        if p < 0.01:
            stars = "***"
        elif p < 0.05:
            stars = "**"
        elif p < 0.1:
            stars = "*"

        lines.append(
            f"{var} & {coef:.4f}{stars} & ({se:.4f}) & {t:.3f} & {p:.4f} \\\\"
        )

    lines.extend([
        r"\hline",
        f"Observations & \\multicolumn{{4}}{{c}}{{{result.n_obs:,}}} \\\\",
        f"Entities & \\multicolumn{{4}}{{c}}{{{result.n_entities:,}}} \\\\",
        f"R-squared (within) & \\multicolumn{{4}}{{c}}{{{result.r_squared_within:.4f}}} \\\\",
        f"R-squared (between) & \\multicolumn{{4}}{{c}}{{{result.r_squared_between:.4f}}} \\\\",
        f"Model & \\multicolumn{{4}}{{c}}{{{model_type.replace('_', ' ').title()}}} \\\\",
        r"\hline\hline",
        r"\multicolumn{5}{l}{\footnotesize *** p$<$0.01, ** p$<$0.05, * p$<$0.1} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


if __name__ == '__main__':
    main()
