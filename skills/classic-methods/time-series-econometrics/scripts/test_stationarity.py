#!/usr/bin/env python3
"""
Stationarity and Unit Root Testing Script

Comprehensive testing for stationarity including:
- Augmented Dickey-Fuller (ADF) test
- Phillips-Perron (PP) test
- KPSS test
- Zivot-Andrews test (with structural break)

Usage:
    python test_stationarity.py data.csv --variable y
    python test_stationarity.py data.csv --all-vars
    python test_stationarity.py data.csv --variable y --with-break
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from time_series_estimator import TimeSeriesEstimator, StationarityResult


def comprehensive_stationarity_test(
    series: pd.Series,
    estimator: TimeSeriesEstimator,
    include_break: bool = False
) -> Dict[str, Any]:
    """
    Run comprehensive battery of stationarity tests.

    Tests performed:
    1. ADF with constant only
    2. ADF with constant and trend
    3. Phillips-Perron
    4. KPSS (level)
    5. KPSS (trend)
    6. Zivot-Andrews (optional)

    Returns
    -------
    dict with all test results and recommendations
    """
    results = {}

    # ADF - constant only
    try:
        adf_c = estimator.test_unit_root(series, test='adf', regression='c')
        results['adf_constant'] = {
            'statistic': adf_c.test_statistic,
            'p_value': adf_c.p_value,
            'lags': adf_c.lags_used,
            'conclusion': 'Stationary' if adf_c.is_stationary else 'Non-stationary'
        }
    except Exception as e:
        results['adf_constant'] = {'error': str(e)}

    # ADF - constant and trend
    try:
        adf_ct = estimator.test_unit_root(series, test='adf', regression='ct')
        results['adf_trend'] = {
            'statistic': adf_ct.test_statistic,
            'p_value': adf_ct.p_value,
            'lags': adf_ct.lags_used,
            'conclusion': 'Trend-stationary' if adf_ct.is_stationary else 'Non-stationary'
        }
    except Exception as e:
        results['adf_trend'] = {'error': str(e)}

    # Phillips-Perron
    try:
        pp = estimator.test_unit_root(series, test='pp', regression='c')
        results['phillips_perron'] = {
            'statistic': pp.test_statistic,
            'p_value': pp.p_value,
            'conclusion': 'Stationary' if pp.is_stationary else 'Non-stationary'
        }
    except Exception as e:
        results['phillips_perron'] = {'error': str(e)}

    # KPSS - level
    try:
        kpss_c = estimator.test_stationarity(series, regression='c')
        results['kpss_level'] = {
            'statistic': kpss_c.test_statistic,
            'p_value': kpss_c.p_value,
            'lags': kpss_c.lags_used,
            'conclusion': 'Stationary' if kpss_c.is_stationary else 'Non-stationary'
        }
    except Exception as e:
        results['kpss_level'] = {'error': str(e)}

    # KPSS - trend
    try:
        kpss_ct = estimator.test_stationarity(series, regression='ct')
        results['kpss_trend'] = {
            'statistic': kpss_ct.test_statistic,
            'p_value': kpss_ct.p_value,
            'lags': kpss_ct.lags_used,
            'conclusion': 'Trend-stationary' if kpss_ct.is_stationary else 'Non-stationary'
        }
    except Exception as e:
        results['kpss_trend'] = {'error': str(e)}

    # Zivot-Andrews (structural break)
    if include_break:
        try:
            za = estimator.test_unit_root(series, test='za', regression='c')
            results['zivot_andrews'] = {
                'statistic': za.test_statistic,
                'p_value': za.p_value,
                'break_point': za.additional_info.get('break_point'),
                'conclusion': 'Stationary around break' if za.is_stationary else 'Non-stationary with break'
            }
        except Exception as e:
            results['zivot_andrews'] = {'error': str(e)}

    # Integration order
    try:
        d = estimator.determine_integration_order(series)
        results['integration_order'] = d
    except Exception as e:
        results['integration_order'] = {'error': str(e)}

    # Combined assessment
    results['assessment'] = generate_assessment(results)

    return results


def generate_assessment(results: Dict[str, Any]) -> Dict[str, str]:
    """Generate overall assessment from individual test results."""
    assessment = {}

    # Count votes
    adf_stationary = (
        results.get('adf_constant', {}).get('conclusion', '').startswith('Stationary') or
        results.get('adf_constant', {}).get('conclusion', '').startswith('Trend')
    )
    kpss_stationary = results.get('kpss_level', {}).get('conclusion', '') == 'Stationary'

    # Standard interpretation grid
    if adf_stationary and kpss_stationary:
        assessment['conclusion'] = 'STATIONARY'
        assessment['explanation'] = 'Both ADF (rejects unit root) and KPSS (fails to reject stationarity) agree'
        assessment['recommendation'] = 'Proceed with stationary time series methods'
    elif not adf_stationary and not kpss_stationary:
        assessment['conclusion'] = 'NON-STATIONARY (UNIT ROOT)'
        assessment['explanation'] = 'Both ADF (fails to reject unit root) and KPSS (rejects stationarity) agree'
        assessment['recommendation'] = 'Consider differencing or cointegration analysis'
    elif adf_stationary and not kpss_stationary:
        assessment['conclusion'] = 'TREND STATIONARY'
        assessment['explanation'] = 'ADF rejects unit root but KPSS rejects stationarity - suggests deterministic trend'
        assessment['recommendation'] = 'Consider detrending or including time trend in model'
    else:
        assessment['conclusion'] = 'INCONCLUSIVE'
        assessment['explanation'] = 'Tests disagree - may need longer sample or different approach'
        assessment['recommendation'] = 'Consider structural break tests or additional analysis'

    # Add integration order recommendation
    d = results.get('integration_order')
    if isinstance(d, int):
        assessment['integration_order'] = f'I({d})'
        if d > 0:
            assessment['differencing'] = f'Difference {d} time(s) for stationarity'

    return assessment


def print_stationarity_report(variable: str, results: Dict[str, Any]):
    """Print formatted stationarity report."""
    print("\n" + "=" * 70)
    print(f"STATIONARITY ANALYSIS: {variable}")
    print("=" * 70)

    # Test results table
    print("\nTest Results:")
    print("-" * 70)
    print(f"{'Test':<25} {'Statistic':>12} {'P-value':>12} {'Conclusion':<20}")
    print("-" * 70)

    for test_name in ['adf_constant', 'adf_trend', 'phillips_perron', 'kpss_level', 'kpss_trend', 'zivot_andrews']:
        if test_name in results:
            test = results[test_name]
            if 'error' in test:
                print(f"{test_name:<25} {'ERROR':<12} {'':<12} {test['error'][:20]}")
            else:
                stat = f"{test['statistic']:.4f}"
                pval = f"{test['p_value']:.4f}"
                conc = test.get('conclusion', 'N/A')
                print(f"{test_name:<25} {stat:>12} {pval:>12} {conc:<20}")

    # Assessment
    print("\n" + "-" * 70)
    print("OVERALL ASSESSMENT")
    print("-" * 70)
    assessment = results.get('assessment', {})
    print(f"Conclusion:     {assessment.get('conclusion', 'N/A')}")
    print(f"Explanation:    {assessment.get('explanation', 'N/A')}")
    print(f"Integration:    {assessment.get('integration_order', 'N/A')}")
    print(f"Recommendation: {assessment.get('recommendation', 'N/A')}")

    # Break point if available
    za = results.get('zivot_andrews', {})
    if 'break_point' in za:
        print(f"\nStructural Break Detected: observation {za['break_point']}")


def test_series_differences(series: pd.Series, estimator: TimeSeriesEstimator, max_d: int = 3):
    """Test stationarity at different differencing orders."""
    print("\nStationarity by Differencing Order:")
    print("-" * 50)
    print(f"{'d':<5} {'ADF stat':>12} {'ADF p-val':>12} {'Stationary':<12}")
    print("-" * 50)

    current = series.copy()

    for d in range(max_d + 1):
        if d > 0:
            current = np.diff(current)

        adf = estimator.test_unit_root(current, test='adf')

        stat_str = f"{adf.test_statistic:.4f}"
        pval_str = f"{adf.p_value:.4f}"
        status = "Yes" if adf.is_stationary else "No"

        print(f"{d:<5} {stat_str:>12} {pval_str:>12} {status:<12}")

        if adf.is_stationary:
            print(f"\n** Series becomes stationary after d={d} differencing **")
            break


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive stationarity testing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('data', help='Path to CSV file')
    parser.add_argument('--variable', '-v', help='Variable to test')
    parser.add_argument('--all-vars', action='store_true', help='Test all numeric variables')
    parser.add_argument('--with-break', action='store_true', help='Include structural break test')
    parser.add_argument('--show-differencing', action='store_true',
                       help='Show stationarity at different differencing orders')
    parser.add_argument('--date-col', help='Date column name')
    parser.add_argument('--output', '-o', help='Output JSON file')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    if args.date_col and args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col])
        df.set_index(args.date_col, inplace=True)

    # Initialize estimator
    estimator = TimeSeriesEstimator()

    # Determine variables to test
    if args.variable:
        variables = [args.variable]
    elif args.all_vars:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        parser.error("Specify --variable or --all-vars")

    # Run tests
    all_results = {}

    for var in variables:
        if var not in df.columns:
            print(f"Warning: Variable '{var}' not found in data")
            continue

        series = df[var].dropna()

        if len(series) < 20:
            print(f"Warning: Variable '{var}' has too few observations ({len(series)})")
            continue

        results = comprehensive_stationarity_test(series, estimator, args.with_break)
        print_stationarity_report(var, results)

        if args.show_differencing:
            test_series_differences(series, estimator)

        all_results[var] = results

    # Save results
    if args.output:
        import json

        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=convert_numpy)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
