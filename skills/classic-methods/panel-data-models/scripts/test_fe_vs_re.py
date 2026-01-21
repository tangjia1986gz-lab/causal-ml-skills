#!/usr/bin/env python
"""
Fixed Effects vs Random Effects Specification Tests

Includes:
- Hausman test
- Mundlak (correlated random effects) test
- Within-between decomposition
- Poolability test

Usage:
    python test_fe_vs_re.py data.csv --entity firm_id --time year --y revenue --x treatment size

Author: Causal ML Skills
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from panel_estimator import PanelEstimator


def parse_args():
    parser = argparse.ArgumentParser(
        description='FE vs RE specification tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all specification tests
  python test_fe_vs_re.py data.csv --entity firm_id --time year --y revenue --x treatment size

  # Just Hausman test
  python test_fe_vs_re.py data.csv --entity firm_id --time year --y revenue --x treatment --hausman-only

  # Include poolability test
  python test_fe_vs_re.py data.csv --entity firm_id --time year --y revenue --x treatment --poolability
        """
    )

    parser.add_argument('data_file', type=str, help='Path to CSV data file')
    parser.add_argument('--entity', required=True, help='Entity identifier column')
    parser.add_argument('--time', required=True, help='Time period column')
    parser.add_argument('--y', required=True, help='Dependent variable column')
    parser.add_argument('--x', nargs='+', required=True, help='Independent variable columns')

    parser.add_argument('--hausman-only', action='store_true', help='Only run Hausman test')
    parser.add_argument('--mundlak', action='store_true', help='Run Mundlak test')
    parser.add_argument('--poolability', action='store_true', help='Run poolability test')
    parser.add_argument('--all', action='store_true', help='Run all tests')

    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')

    return parser.parse_args()


def poolability_test(data, entity_col, time_col, y_col, x_cols):
    """
    Test if coefficients are constant across entities.

    H0: beta_1 = beta_2 = ... = beta_N (poolability)
    H1: Coefficients differ across entities
    """
    # Restricted model (pooled OLS)
    y = data[y_col].values
    X = sm.add_constant(data[x_cols].values)

    model_r = sm.OLS(y, X).fit()
    RSS_r = model_r.ssr
    df_r = model_r.df_resid

    # Unrestricted model (entity-specific OLS)
    RSS_ur = 0
    df_ur = 0
    entity_results = {}

    for entity in data[entity_col].unique():
        mask = data[entity_col] == entity
        y_i = data.loc[mask, y_col].values
        X_i = sm.add_constant(data.loc[mask, x_cols].values)

        if len(y_i) > len(x_cols) + 1:  # Need enough observations
            try:
                model_i = sm.OLS(y_i, X_i).fit()
                RSS_ur += model_i.ssr
                df_ur += model_i.df_resid
                entity_results[entity] = {
                    'coefficients': dict(zip(['const'] + x_cols, model_i.params)),
                    'r_squared': model_i.rsquared,
                }
            except:
                pass

    if df_ur <= 0:
        return {
            'test': 'Poolability (Chow) Test',
            'error': 'Insufficient data for entity-specific estimation',
        }

    # F-test
    k = len(x_cols) + 1  # Including constant
    N = len(entity_results)

    numerator_df = (N - 1) * k
    F = ((RSS_r - RSS_ur) / numerator_df) / (RSS_ur / df_ur)
    p_value = 1 - stats.f.cdf(F, numerator_df, df_ur)

    return {
        'test': 'Poolability (Chow) Test',
        'H0': 'Coefficients are the same across all entities',
        'H1': 'Coefficients differ across entities',
        'F_statistic': F,
        'df1': numerator_df,
        'df2': df_ur,
        'p_value': p_value,
        'n_entities_estimated': N,
        'reject_H0': p_value < 0.05,
        'conclusion': 'Reject poolability (coefficients differ across entities)'
                     if p_value < 0.05
                     else 'Cannot reject poolability (coefficients may be constant)',
    }


def breusch_pagan_test(data, entity_col, time_col, y_col, x_cols):
    """
    Breusch-Pagan LM test for random effects.

    H0: Var(alpha_i) = 0 (no random effects, pooled OLS is appropriate)
    H1: Var(alpha_i) > 0 (random effects needed)
    """
    # Pooled OLS
    y = data[y_col].values
    X = sm.add_constant(data[x_cols].values)

    pooled = sm.OLS(y, X).fit()
    resid = pooled.resid

    # Sum of squared entity-mean residuals
    data_temp = data.copy()
    data_temp['resid'] = resid

    entity_mean_resid = data_temp.groupby(entity_col)['resid'].transform('mean')
    T_i = data_temp.groupby(entity_col).size()

    N = data[entity_col].nunique()
    T_bar = T_i.mean()
    total_T = len(data)

    # LM statistic
    numerator = (entity_mean_resid.sum()) ** 2
    denominator = (resid ** 2).sum()

    # Alternative formulation
    A = (data_temp.groupby(entity_col)['resid'].sum() ** 2).sum()
    B = (resid ** 2).sum()

    LM = (total_T ** 2) / (2 * (total_T - 1)) * ((A / B) - 1) ** 2

    # Simplified balanced panel formula
    if T_i.std() < 0.01:  # Approximately balanced
        LM = (N * T_bar) / (2 * (T_bar - 1)) * (
            ((entity_mean_resid * T_bar).sum() / resid.sum()) ** 2 - 1
        ) ** 2

    p_value = 1 - stats.chi2.cdf(abs(LM), 1)

    return {
        'test': 'Breusch-Pagan LM Test for Random Effects',
        'H0': 'No random effects (pooled OLS appropriate)',
        'H1': 'Random effects present',
        'LM_statistic': LM,
        'df': 1,
        'p_value': p_value,
        'reject_H0': p_value < 0.05,
        'conclusion': 'Random effects present (use RE or FE)'
                     if p_value < 0.05
                     else 'No evidence of random effects (pooled OLS may suffice)',
    }


def f_test_fixed_effects(data, entity_col, time_col, y_col, x_cols):
    """
    F-test for joint significance of entity fixed effects.

    H0: All entity fixed effects = 0
    H1: At least one entity effect != 0
    """
    # Restricted model (pooled OLS)
    y = data[y_col].values
    X = sm.add_constant(data[x_cols].values)

    pooled = sm.OLS(y, X).fit()
    RSS_r = pooled.ssr

    # Unrestricted model (with entity dummies)
    entity_dummies = pd.get_dummies(data[entity_col], prefix='entity', drop_first=True)
    X_fe = np.column_stack([X, entity_dummies.values])

    fe = sm.OLS(y, X_fe).fit()
    RSS_ur = fe.ssr

    # F-test
    N = data[entity_col].nunique()
    n = len(data)
    k = len(x_cols) + 1

    F = ((RSS_r - RSS_ur) / (N - 1)) / (RSS_ur / (n - N - k))
    p_value = 1 - stats.f.cdf(F, N - 1, n - N - k)

    return {
        'test': 'F-test for Fixed Effects',
        'H0': 'All entity fixed effects are zero',
        'H1': 'At least one entity effect is non-zero',
        'F_statistic': F,
        'df1': N - 1,
        'df2': n - N - k,
        'p_value': p_value,
        'reject_H0': p_value < 0.05,
        'conclusion': 'Fixed effects are jointly significant'
                     if p_value < 0.05
                     else 'Cannot reject that all fixed effects are zero',
    }


def print_test_result(result, alpha=0.05):
    """Pretty print a test result."""
    print(f"\n{'='*60}")
    print(f"{result['test']}")
    print(f"{'='*60}")

    if 'error' in result:
        print(f"Error: {result['error']}")
        return

    print(f"\nHypotheses:")
    print(f"  H0: {result['H0']}")
    print(f"  H1: {result['H1']}")

    print(f"\nTest Statistics:")
    for key in ['F_statistic', 'LM_statistic', 'chi2_statistic', 'test_statistic']:
        if key in result:
            print(f"  {key}: {result[key]:.4f}")

    if 'df' in result:
        print(f"  Degrees of freedom: {result['df']}")
    if 'df1' in result and 'df2' in result:
        print(f"  Degrees of freedom: ({result['df1']}, {result['df2']})")

    print(f"  p-value: {result['p_value']:.4f}")

    print(f"\nDecision (alpha = {alpha}):")
    if result['p_value'] < alpha:
        print(f"  REJECT H0")
    else:
        print(f"  FAIL TO REJECT H0")

    print(f"\nConclusion:")
    print(f"  {result['conclusion']}")


def main():
    args = parse_args()

    # Load data
    print(f"Loading data from {args.data_file}...")
    data = pd.read_csv(args.data_file)

    # Validate columns
    required_cols = [args.entity, args.time, args.y] + args.x
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        sys.exit(1)

    # Initialize estimator
    estimator = PanelEstimator(
        data=data,
        entity_col=args.entity,
        time_col=args.time,
        y_col=args.y,
        x_cols=args.x
    )

    print(f"\nPanel structure:")
    print(f"  Entities: {estimator.n_entities:,}")
    print(f"  Time periods: {estimator.n_times}")
    print(f"  Observations: {estimator.n_obs:,}")

    results = {}

    # Run tests
    if args.hausman_only:
        # Just Hausman test
        hausman = estimator.hausman_test()
        print(hausman)
        results['hausman'] = {
            'test_statistic': hausman.test_statistic,
            'p_value': hausman.p_value,
            'df': hausman.df,
            'conclusion': hausman.conclusion,
        }
    else:
        # Full battery of tests
        print("\n" + "#" * 60)
        print("# SPECIFICATION TESTS: FIXED EFFECTS vs RANDOM EFFECTS")
        print("#" * 60)

        # 1. F-test for fixed effects
        print("\n[1/5] Testing joint significance of fixed effects...")
        f_test = f_test_fixed_effects(data, args.entity, args.time, args.y, args.x)
        print_test_result(f_test, args.alpha)
        results['f_test_fe'] = f_test

        # 2. Breusch-Pagan LM test
        print("\n[2/5] Testing for random effects (Breusch-Pagan)...")
        bp_test = breusch_pagan_test(data, args.entity, args.time, args.y, args.x)
        print_test_result(bp_test, args.alpha)
        results['breusch_pagan'] = bp_test

        # 3. Hausman test
        print("\n[3/5] Hausman specification test (FE vs RE)...")
        hausman = estimator.hausman_test()
        print(hausman)
        results['hausman'] = {
            'test': 'Hausman Specification Test',
            'H0': 'Random effects is consistent and efficient',
            'H1': 'Fixed effects is consistent, random effects is not',
            'test_statistic': hausman.test_statistic,
            'p_value': hausman.p_value,
            'df': hausman.df,
            'reject_H0': hausman.p_value < args.alpha,
            'conclusion': hausman.conclusion,
        }

        # 4. Mundlak test
        if args.mundlak or args.all:
            print("\n[4/5] Mundlak (within-between) test...")
            mundlak = estimator.within_between_test()
            print(f"\n{'='*60}")
            print("Mundlak (Within-Between) Test")
            print(f"{'='*60}")
            print(f"\nH0: Entity means are not correlated with regressors")
            print(f"H1: Entity means are correlated (use FE)")
            print(f"\nF-statistic: {mundlak['F_statistic']:.4f}")
            print(f"p-value: {mundlak['p_value']:.4f}")
            print(f"\nMean coefficients:")
            for var, coef in mundlak['mean_coefficients'].items():
                print(f"  {var}: {coef:.4f}")
            print(f"\nConclusion: {mundlak['conclusion']}")
            results['mundlak'] = mundlak

        # 5. Poolability test
        if args.poolability or args.all:
            print("\n[5/5] Poolability (Chow) test...")
            pool_test = poolability_test(data, args.entity, args.time, args.y, args.x)
            print_test_result(pool_test, args.alpha)
            results['poolability'] = pool_test

        # Summary recommendation
        print("\n" + "#" * 60)
        print("# SUMMARY AND RECOMMENDATION")
        print("#" * 60)

        print("\nTest Results Summary:")
        print(f"  - F-test for FE: {'Significant' if f_test['reject_H0'] else 'Not significant'} (p={f_test['p_value']:.4f})")
        print(f"  - Breusch-Pagan: {'Random effects present' if bp_test['reject_H0'] else 'No random effects'} (p={bp_test['p_value']:.4f})")
        print(f"  - Hausman test: {'Use FE' if results['hausman']['reject_H0'] else 'RE is efficient'} (p={results['hausman']['p_value']:.4f})")

        print("\nRecommendation:")
        if results['hausman']['reject_H0']:
            print("  USE FIXED EFFECTS")
            print("  - Hausman test rejects random effects consistency")
            print("  - Entity effects are correlated with regressors")
        elif f_test['reject_H0']:
            print("  USE RANDOM EFFECTS (or Fixed Effects)")
            print("  - Entity effects are significant")
            print("  - Hausman cannot reject RE consistency")
            print("  - RE is more efficient if assumptions hold")
        else:
            print("  POOLED OLS may be appropriate")
            print("  - No significant entity effects detected")
            print("  - But consider theory and prior evidence")

    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                return obj

            results_json = {k: {kk: convert(vv) for kk, vv in v.items()} if isinstance(v, dict) else convert(v)
                          for k, v in results.items()}
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
