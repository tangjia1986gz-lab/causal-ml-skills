#!/usr/bin/env python3
"""
Comprehensive Instrument Validity Tests.

This script provides comprehensive testing of instrument validity including:
- First-stage strength tests (Stock-Yogo, Cragg-Donald, Kleibergen-Paap)
- Overidentification tests (Sargan, Hansen J)
- Endogeneity tests (Wu-Hausman, Durbin)
- Balance tests (instrument exogeneity)
- Falsification tests (placebo outcomes)

Usage:
    python test_instruments.py --data data.csv --outcome y --treatment d \
        --instruments z1 z2 --controls x1 x2

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from iv_estimator import (
    first_stage_test,
    weak_iv_diagnostics,
    estimate_2sls,
    overidentification_test,
    endogeneity_test,
    STOCK_YOGO_CRITICAL_VALUES
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive instrument validity tests"
    )

    parser.add_argument("--data", "-d", type=str, required=True)
    parser.add_argument("--outcome", "-y", type=str, required=True)
    parser.add_argument("--treatment", "-t", type=str, required=True)
    parser.add_argument("--instruments", "-z", type=str, nargs="+", required=True)
    parser.add_argument("--controls", "-x", type=str, nargs="*", default=None)
    parser.add_argument("--balance-vars", type=str, nargs="*", default=None,
                        help="Variables for balance tests")
    parser.add_argument("--placebo-outcomes", type=str, nargs="*", default=None,
                        help="Placebo outcomes for falsification tests")
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


# =============================================================================
# First-Stage Tests
# =============================================================================

def comprehensive_first_stage(
    data: pd.DataFrame,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run comprehensive first-stage diagnostics.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    treatment : str
        Endogenous treatment variable
    instruments : List[str]
        Instrument variable names
    controls : List[str], optional
        Control variable names
    verbose : bool
        Print detailed output

    Returns
    -------
    Dict[str, Any]
        First-stage diagnostic results
    """
    results = {}

    # Basic first-stage test
    fs = first_stage_test(data, treatment, instruments, controls)
    results['f_statistic'] = fs['f_statistic']
    results['f_pvalue'] = fs['f_pvalue']
    results['partial_r2'] = fs['partial_r2']
    results['coefficients'] = fs['coefficients']
    results['std_errors'] = fs['std_errors']

    # Stock-Yogo critical values
    n_instr = len(instruments)
    if n_instr in STOCK_YOGO_CRITICAL_VALUES:
        results['stock_yogo'] = {
            'critical_10_bias': STOCK_YOGO_CRITICAL_VALUES[n_instr].get(10),
            'critical_15_bias': STOCK_YOGO_CRITICAL_VALUES[n_instr].get(15),
            'critical_20_bias': STOCK_YOGO_CRITICAL_VALUES[n_instr].get(20),
            'critical_25_bias': STOCK_YOGO_CRITICAL_VALUES[n_instr].get(25),
        }

        # Determine which threshold is passed
        for bias_level in [10, 15, 20, 25]:
            cv = STOCK_YOGO_CRITICAL_VALUES[n_instr].get(bias_level)
            if cv and fs['f_statistic'] > cv:
                results['stock_yogo']['max_bias_exceeded'] = bias_level
                break
        else:
            results['stock_yogo']['max_bias_exceeded'] = None
    else:
        results['stock_yogo'] = {'note': f'Critical values not available for {n_instr} instruments'}

    # Weak IV assessment
    weak_iv = weak_iv_diagnostics(fs['f_statistic'], n_instr)
    results['weak_iv_passed'] = weak_iv.passed
    results['weak_iv_interpretation'] = weak_iv.interpretation

    # Individual instrument strength
    results['individual_instruments'] = {}
    for z in instruments:
        # Test each instrument individually
        single_fs = first_stage_test(data, treatment, [z], controls)
        results['individual_instruments'][z] = {
            'coefficient': fs['coefficients'][z],
            'se': fs['std_errors'][z],
            't_stat': fs['coefficients'][z] / fs['std_errors'][z],
            'individual_f': single_fs['f_statistic']
        }

    if verbose:
        print("\n" + "=" * 60)
        print("FIRST-STAGE DIAGNOSTICS")
        print("=" * 60)
        print(f"Joint F-statistic: {fs['f_statistic']:.2f}")
        print(f"Partial R-squared: {fs['partial_r2']:.4f}")
        print(f"\nWeak IV Assessment: {'PASSED' if weak_iv.passed else 'FAILED'}")
        print(f"  {weak_iv.interpretation}")

        if 'max_bias_exceeded' in results.get('stock_yogo', {}):
            bias = results['stock_yogo']['max_bias_exceeded']
            if bias:
                print(f"  Stock-Yogo: Exceeds {bias}% maximal bias critical value")
            else:
                print("  Stock-Yogo: Below all critical values")

        print("\nIndividual instruments:")
        for z, stats_z in results['individual_instruments'].items():
            print(f"  {z}: coef={stats_z['coefficient']:.4f}, "
                  f"t={stats_z['t_stat']:.2f}, "
                  f"individual F={stats_z['individual_f']:.2f}")

    return results


# =============================================================================
# Anderson-Rubin Test and Confidence Interval
# =============================================================================

def anderson_rubin_test(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]] = None,
    gamma_0: float = 0.0
) -> Dict[str, float]:
    """
    Anderson-Rubin test for a specific null hypothesis value.

    Tests H0: gamma = gamma_0

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable
    treatment : str
        Treatment variable
    instruments : List[str]
        Instrument names
    controls : List[str], optional
        Control variable names
    gamma_0 : float
        Null hypothesis value (default: 0)

    Returns
    -------
    Dict[str, float]
        AR test statistic and p-value
    """
    import statsmodels.api as sm

    df = data.copy()
    y = df[outcome].values
    d = df[treatment].values

    # Build instrument/control matrix
    Z_vars = instruments.copy()
    if controls:
        Z_vars.extend(controls)
    Z = sm.add_constant(df[Z_vars]).values

    n, K = Z.shape

    # Construct residual under null
    residual = y - d * gamma_0

    # Projection matrices
    P_Z = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    M_Z = np.eye(n) - P_Z

    # AR statistic
    numerator = (residual @ P_Z @ residual) / K
    denominator = (residual @ M_Z @ residual) / (n - K)

    ar_stat = numerator / denominator
    p_value = 1 - stats.f.cdf(ar_stat, K, n - K)

    return {
        'statistic': ar_stat,
        'p_value': p_value,
        'df1': K,
        'df2': n - K,
        'gamma_0': gamma_0
    }


def anderson_rubin_ci(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]] = None,
    alpha: float = 0.05,
    grid_points: int = 1000,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute Anderson-Rubin confidence interval.

    The AR confidence set is valid regardless of instrument strength.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable
    treatment : str
        Treatment variable
    instruments : List[str]
        Instrument names
    controls : List[str], optional
        Control variable names
    alpha : float
        Significance level (default: 0.05)
    grid_points : int
        Number of grid points (default: 1000)
    verbose : bool
        Print progress

    Returns
    -------
    Dict[str, Any]
        AR confidence interval
    """
    import statsmodels.api as sm

    df = data.copy()
    y = df[outcome].values
    d = df[treatment].values

    # Build instrument matrix
    Z_vars = instruments.copy()
    if controls:
        Z_vars.extend(controls)
    Z = sm.add_constant(df[Z_vars]).values

    n, K = Z.shape

    # Critical value
    f_crit = stats.f.ppf(1 - alpha, K, n - K)

    # Get 2SLS estimate as starting point
    result_2sls = estimate_2sls(data, outcome, treatment, instruments, controls)
    center = result_2sls.effect
    width = max(10 * result_2sls.se, abs(center))

    if verbose:
        print(f"Searching for AR CI around {center:.4f} +/- {width:.4f}")

    # Projection matrices
    P_Z = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    M_Z = np.eye(n) - P_Z

    # Grid search
    gamma_grid = np.linspace(center - width, center + width, grid_points)
    in_ci = []

    for gamma_0 in gamma_grid:
        residual = y - d * gamma_0

        numerator = (residual @ P_Z @ residual) / K
        denominator = (residual @ M_Z @ residual) / (n - K)

        if denominator > 0:
            ar_stat = numerator / denominator
            if ar_stat <= f_crit:
                in_ci.append(gamma_0)

    if len(in_ci) == 0:
        # CI may be empty or we need wider search
        return {
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'bounded': False,
            'empty': True,
            'note': 'Empty confidence set - may indicate invalid instruments'
        }

    ci_lower = min(in_ci)
    ci_upper = max(in_ci)

    # Check if CI extends to grid boundaries (potentially unbounded)
    tolerance = 0.01 * width
    bounded = (ci_lower > gamma_grid[0] + tolerance and
               ci_upper < gamma_grid[-1] - tolerance)

    return {
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bounded': bounded,
        'empty': False,
        'width': ci_upper - ci_lower,
        '2sls_estimate': center,
        'note': 'Bounded interval' if bounded else 'May be unbounded - consider wider grid'
    }


# =============================================================================
# Balance Tests
# =============================================================================

def balance_tests(
    data: pd.DataFrame,
    instruments: List[str],
    balance_vars: List[str],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Test if instruments are balanced on observable characteristics.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    instruments : List[str]
        Instrument names
    balance_vars : List[str]
        Variables to check for balance
    verbose : bool
        Print detailed output

    Returns
    -------
    Dict[str, Any]
        Balance test results
    """
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS

    results = {}
    any_imbalance = False

    for var in balance_vars:
        Z = sm.add_constant(data[instruments])
        y = data[var]

        # Handle missing values
        mask = ~(Z.isna().any(axis=1) | y.isna())

        model = OLS(y[mask], Z[mask]).fit()

        # Joint F-test for all instruments
        f_stat = model.fvalue
        f_pval = model.f_pvalue

        # Individual instrument tests
        individual_tests = {}
        for z in instruments:
            individual_tests[z] = {
                'coefficient': model.params[z],
                'se': model.bse[z],
                't_stat': model.tvalues[z],
                'p_value': model.pvalues[z]
            }

        imbalanced = f_pval < 0.05
        if imbalanced:
            any_imbalance = True

        results[var] = {
            'f_statistic': f_stat,
            'f_pvalue': f_pval,
            'imbalanced': imbalanced,
            'individual_tests': individual_tests
        }

    results['_summary'] = {
        'any_imbalance_detected': any_imbalance,
        'n_vars_tested': len(balance_vars),
        'n_imbalanced': sum(1 for v in balance_vars if results[v]['imbalanced'])
    }

    if verbose:
        print("\n" + "=" * 60)
        print("BALANCE TESTS")
        print("=" * 60)
        print("Testing if instruments predict observable characteristics...")
        print()

        for var in balance_vars:
            r = results[var]
            status = "IMBALANCED" if r['imbalanced'] else "Balanced"
            print(f"{var}: F = {r['f_statistic']:.2f}, p = {r['f_pvalue']:.4f} [{status}]")

        if any_imbalance:
            print("\nWARNING: Some variables show imbalance.")
            print("This may indicate instrument exogeneity concerns.")
        else:
            print("\nAll balance tests passed.")

    return results


# =============================================================================
# Falsification Tests
# =============================================================================

def falsification_tests(
    data: pd.DataFrame,
    placebo_outcomes: List[str],
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run falsification tests using placebo outcomes.

    Instruments should NOT predict outcomes they should not affect
    (e.g., pre-treatment outcomes).

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    placebo_outcomes : List[str]
        Variables that should not be affected by instrument
    treatment : str
        Treatment variable
    instruments : List[str]
        Instrument names
    controls : List[str], optional
        Control variable names
    verbose : bool
        Print detailed output

    Returns
    -------
    Dict[str, Any]
        Falsification test results
    """
    results = {}
    any_failure = False

    for placebo in placebo_outcomes:
        try:
            # Run reduced form: regress placebo on instruments
            import statsmodels.api as sm
            from statsmodels.regression.linear_model import OLS

            Z_vars = instruments.copy()
            if controls:
                Z_vars.extend(controls)
            Z = sm.add_constant(data[Z_vars])
            y = data[placebo]

            mask = ~(Z.isna().any(axis=1) | y.isna())
            model = OLS(y[mask], Z[mask]).fit()

            # Test joint significance of instruments
            # Using Wald test on instrument coefficients
            instrument_coefs = {z: model.params[z] for z in instruments}
            instrument_pvals = {z: model.pvalues[z] for z in instruments}

            # Joint test
            r_matrix = np.zeros((len(instruments), len(model.params)))
            for i, z in enumerate(instruments):
                idx = list(model.params.index).index(z)
                r_matrix[i, idx] = 1

            wald_test = model.wald_test(r_matrix)
            joint_p = wald_test.pvalue

            failed = joint_p < 0.05
            if failed:
                any_failure = True

            results[placebo] = {
                'joint_p_value': float(joint_p),
                'failed': failed,
                'instrument_coefficients': instrument_coefs,
                'instrument_pvalues': instrument_pvals
            }

        except Exception as e:
            results[placebo] = {
                'error': str(e),
                'failed': None
            }

    results['_summary'] = {
        'any_failure': any_failure,
        'n_placebo_tested': len(placebo_outcomes),
        'n_failures': sum(1 for p in placebo_outcomes
                         if results[p].get('failed', False))
    }

    if verbose:
        print("\n" + "=" * 60)
        print("FALSIFICATION TESTS")
        print("=" * 60)
        print("Testing if instruments predict placebo outcomes...")
        print()

        for placebo in placebo_outcomes:
            r = results[placebo]
            if 'error' in r:
                print(f"{placebo}: ERROR - {r['error']}")
            else:
                status = "FAILED" if r['failed'] else "Passed"
                print(f"{placebo}: joint p = {r['joint_p_value']:.4f} [{status}]")

        if any_failure:
            print("\nWARNING: Some falsification tests failed.")
            print("This may indicate exclusion restriction violations.")
        else:
            print("\nAll falsification tests passed.")

    return results


# =============================================================================
# Subset Instrument Comparisons
# =============================================================================

def compare_instrument_subsets(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compare IV estimates using different instrument subsets.

    If instruments are valid, different subsets should give similar estimates.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable
    treatment : str
        Treatment variable
    instruments : List[str]
        All instrument names
    controls : List[str], optional
        Control variable names
    verbose : bool
        Print detailed output

    Returns
    -------
    Dict[str, Any]
        Comparison results
    """
    from itertools import combinations

    results = {}
    estimates = []

    # Full model
    try:
        full_result = estimate_2sls(data, outcome, treatment, instruments, controls)
        results['full'] = {
            'instruments': instruments,
            'estimate': full_result.effect,
            'se': full_result.se
        }
        estimates.append(full_result.effect)
    except Exception as e:
        results['full'] = {'error': str(e)}

    # Individual instruments (if multiple)
    if len(instruments) > 1:
        results['individual'] = {}
        for z in instruments:
            try:
                single_result = estimate_2sls(data, outcome, treatment, [z], controls)
                results['individual'][z] = {
                    'estimate': single_result.effect,
                    'se': single_result.se
                }
                estimates.append(single_result.effect)
            except Exception as e:
                results['individual'][z] = {'error': str(e)}

    # Pairs (if more than 2 instruments)
    if len(instruments) > 2:
        results['pairs'] = {}
        for pair in combinations(instruments, 2):
            pair_name = f"{pair[0]}_{pair[1]}"
            try:
                pair_result = estimate_2sls(data, outcome, treatment, list(pair), controls)
                results['pairs'][pair_name] = {
                    'instruments': list(pair),
                    'estimate': pair_result.effect,
                    'se': pair_result.se
                }
                estimates.append(pair_result.effect)
            except Exception as e:
                results['pairs'][pair_name] = {'error': str(e)}

    # Summary statistics
    if len(estimates) > 1:
        results['_summary'] = {
            'n_subsets': len(estimates),
            'mean_estimate': np.mean(estimates),
            'std_estimate': np.std(estimates),
            'range': max(estimates) - min(estimates),
            'min_estimate': min(estimates),
            'max_estimate': max(estimates)
        }

        # Check for large discrepancies
        if full_result:
            typical_se = full_result.se
            if results['_summary']['range'] > 2 * typical_se:
                results['_summary']['large_discrepancy'] = True
                results['_summary']['warning'] = (
                    "Large discrepancy between instrument subsets. "
                    "This may indicate heterogeneous effects or invalid instruments."
                )
            else:
                results['_summary']['large_discrepancy'] = False

    if verbose:
        print("\n" + "=" * 60)
        print("INSTRUMENT SUBSET COMPARISON")
        print("=" * 60)

        if 'full' in results and 'estimate' in results['full']:
            print(f"Full model ({', '.join(instruments)}): "
                  f"{results['full']['estimate']:.4f} (SE: {results['full']['se']:.4f})")

        if 'individual' in results:
            print("\nIndividual instruments:")
            for z, r in results['individual'].items():
                if 'estimate' in r:
                    print(f"  {z}: {r['estimate']:.4f} (SE: {r['se']:.4f})")

        if 'pairs' in results:
            print("\nInstrument pairs:")
            for name, r in results['pairs'].items():
                if 'estimate' in r:
                    print(f"  {name}: {r['estimate']:.4f} (SE: {r['se']:.4f})")

        if '_summary' in results:
            s = results['_summary']
            print(f"\nSummary: mean = {s['mean_estimate']:.4f}, "
                  f"range = {s['range']:.4f}")
            if s.get('large_discrepancy'):
                print(f"WARNING: {s['warning']}")

    return results


# =============================================================================
# Main Function
# =============================================================================

def run_all_tests(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: Optional[List[str]] = None,
    balance_vars: Optional[List[str]] = None,
    placebo_outcomes: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run all instrument validity tests.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable
    treatment : str
        Treatment variable
    instruments : List[str]
        Instrument names
    controls : List[str], optional
        Control variable names
    balance_vars : List[str], optional
        Variables for balance tests
    placebo_outcomes : List[str], optional
        Placebo outcomes for falsification
    verbose : bool
        Print detailed output

    Returns
    -------
    Dict[str, Any]
        All test results
    """
    results = {}

    print("\n" + "=" * 70)
    print("COMPREHENSIVE INSTRUMENT VALIDITY TESTS")
    print("=" * 70)

    # 1. First-stage diagnostics
    results['first_stage'] = comprehensive_first_stage(
        data, treatment, instruments, controls, verbose
    )

    # 2. Endogeneity test
    if verbose:
        print("\n" + "=" * 60)
        print("ENDOGENEITY TEST")
        print("=" * 60)

    endog = endogeneity_test(data, outcome, treatment, instruments, controls)
    results['endogeneity'] = {
        'statistic': endog.statistic,
        'p_value': endog.p_value,
        'is_endogenous': endog.passed,
        'interpretation': endog.interpretation
    }

    if verbose:
        print(f"Wu-Hausman test: t = {endog.statistic:.4f}, p = {endog.p_value:.4f}")
        print(f"  {endog.interpretation}")

    # 3. Overidentification test
    if len(instruments) > 1:
        if verbose:
            print("\n" + "=" * 60)
            print("OVERIDENTIFICATION TEST")
            print("=" * 60)

        iv_result = estimate_2sls(data, outcome, treatment, instruments, controls)
        overid = overidentification_test(
            iv_result, data, outcome, treatment, instruments, controls
        )
        results['overidentification'] = {
            'statistic': overid.statistic if not np.isnan(overid.statistic) else None,
            'p_value': overid.p_value if not np.isnan(overid.p_value) else None,
            'passed': overid.passed,
            'interpretation': overid.interpretation
        }

        if verbose and not np.isnan(overid.statistic):
            print(f"Sargan-Hansen J: {overid.statistic:.4f}, p = {overid.p_value:.4f}")
            print(f"  {overid.interpretation}")

    # 4. Anderson-Rubin CI (if weak instruments)
    if not results['first_stage']['weak_iv_passed']:
        if verbose:
            print("\n" + "=" * 60)
            print("WEAK-IV ROBUST INFERENCE")
            print("=" * 60)

        ar_ci = anderson_rubin_ci(data, outcome, treatment, instruments, controls, verbose=verbose)
        results['anderson_rubin_ci'] = ar_ci

        if verbose:
            if ar_ci['bounded']:
                print(f"AR 95% CI: [{ar_ci['ci_lower']:.4f}, {ar_ci['ci_upper']:.4f}]")
            else:
                print(f"AR CI may be unbounded (consider wider search)")

    # 5. Balance tests
    if balance_vars:
        results['balance'] = balance_tests(data, instruments, balance_vars, verbose)

    # 6. Falsification tests
    if placebo_outcomes:
        results['falsification'] = falsification_tests(
            data, placebo_outcomes, treatment, instruments, controls, verbose
        )

    # 7. Subset comparison
    if len(instruments) > 1:
        results['subset_comparison'] = compare_instrument_subsets(
            data, outcome, treatment, instruments, controls, verbose
        )

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("VALIDITY ASSESSMENT SUMMARY")
        print("=" * 70)

        issues = []

        # Check first stage
        if not results['first_stage']['weak_iv_passed']:
            issues.append("- WEAK INSTRUMENTS: First-stage F below threshold")

        # Check endogeneity
        if not results['endogeneity']['is_endogenous']:
            issues.append("- Endogeneity not detected: OLS may be consistent")

        # Check overidentification
        if 'overidentification' in results:
            if not results['overidentification']['passed']:
                issues.append("- OVERID TEST FAILED: Some instruments may be invalid")

        # Check balance
        if 'balance' in results:
            if results['balance']['_summary']['any_imbalance_detected']:
                issues.append("- BALANCE ISSUES: Instruments predict some covariates")

        # Check falsification
        if 'falsification' in results:
            if results['falsification']['_summary']['any_failure']:
                issues.append("- FALSIFICATION FAILED: Instruments predict placebo outcomes")

        # Check subset consistency
        if 'subset_comparison' in results:
            if results['subset_comparison'].get('_summary', {}).get('large_discrepancy'):
                issues.append("- INCONSISTENT SUBSETS: Different instruments give different estimates")

        if issues:
            print("CONCERNS IDENTIFIED:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("All validity tests passed.")
            print("Instruments appear to be valid and strong.")

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Load data
    df = pd.read_csv(args.data)

    # Run all tests
    results = run_all_tests(
        data=df,
        outcome=args.outcome,
        treatment=args.treatment,
        instruments=args.instruments,
        controls=args.controls,
        balance_vars=args.balance_vars,
        placebo_outcomes=args.placebo_outcomes,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
