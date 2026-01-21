#!/usr/bin/env python3
"""
Time Series Analysis CLI

Command-line interface for comprehensive time series analysis including
stationarity testing, ARIMA/VAR modeling, and Granger causality.

Usage:
    python run_time_series.py data.csv --analysis stationarity
    python run_time_series.py data.csv --analysis arima --target y
    python run_time_series.py data.csv --analysis var
    python run_time_series.py data.csv --analysis granger --target y --cause x
    python run_time_series.py data.csv --analysis cointegration
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from time_series_estimator import TimeSeriesEstimator


def load_data(filepath: str, date_col: str = None) -> pd.DataFrame:
    """Load data from CSV file."""
    df = pd.read_csv(filepath)

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

    return df


def run_stationarity_analysis(df: pd.DataFrame, estimator: TimeSeriesEstimator) -> dict:
    """Run comprehensive stationarity analysis."""
    results = {'analysis_type': 'stationarity', 'variables': {}}

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            series = df[col].dropna()

            # ADF test
            adf = estimator.test_unit_root(series, test='adf', regression='c')

            # KPSS test
            kpss = estimator.test_stationarity(series, regression='c')

            # Determine integration order
            d = estimator.determine_integration_order(series)

            results['variables'][col] = {
                'adf_test': {
                    'statistic': float(adf.test_statistic),
                    'p_value': float(adf.p_value),
                    'is_stationary': adf.is_stationary,
                    'lags_used': adf.lags_used
                },
                'kpss_test': {
                    'statistic': float(kpss.test_statistic),
                    'p_value': float(kpss.p_value),
                    'is_stationary': kpss.is_stationary
                },
                'integration_order': d,
                'combined_conclusion': _stationarity_conclusion(adf, kpss)
            }

    return results


def _stationarity_conclusion(adf_result, kpss_result) -> str:
    """Combine ADF and KPSS results for conclusion."""
    adf_stationary = adf_result.is_stationary
    kpss_stationary = kpss_result.is_stationary

    if adf_stationary and kpss_stationary:
        return "Stationary (both tests agree)"
    elif not adf_stationary and not kpss_stationary:
        return "Non-stationary / Unit root (both tests agree)"
    elif adf_stationary and not kpss_stationary:
        return "Trend stationary (ADF rejects, KPSS rejects)"
    else:
        return "Inconclusive (conflicting results)"


def run_arima_analysis(
    df: pd.DataFrame,
    target: str,
    estimator: TimeSeriesEstimator,
    auto_select: bool = True
) -> dict:
    """Run ARIMA analysis on target variable."""
    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in data")

    series = df[target].dropna()

    # Fit ARIMA
    result = estimator.fit_arima(series, auto_select=auto_select)

    # Forecast
    forecast_df = estimator.forecast_arima(result, steps=10)

    return {
        'analysis_type': 'arima',
        'target': target,
        'model': {
            'order': list(result.order),
            'aic': float(result.aic),
            'bic': float(result.bic),
            'log_likelihood': float(result.log_likelihood)
        },
        'coefficients': {k: float(v) for k, v in result.coefficients.items()},
        'p_values': {k: float(v) for k, v in result.p_values.items()},
        'diagnostics': {
            'ljung_box_passed': result.diagnostics.get('ljung_box', {}).get('passed', None),
            'jarque_bera_passed': result.diagnostics.get('jarque_bera', {}).get('passed', None),
            'arch_test_passed': result.diagnostics.get('arch_test', {}).get('passed', None)
        },
        'forecast': {
            'periods': 10,
            'mean': forecast_df['forecast'].tolist(),
            'lower_95': forecast_df['lower'].tolist(),
            'upper_95': forecast_df['upper'].tolist()
        }
    }


def run_var_analysis(df: pd.DataFrame, estimator: TimeSeriesEstimator, maxlags: int = None) -> dict:
    """Run VAR analysis on all numeric variables."""
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        raise ValueError("VAR requires at least 2 numeric variables")

    # Fit VAR
    result = estimator.fit_var(numeric_df, maxlags=maxlags)

    # Granger causality matrix
    gc_matrix = estimator.granger_causality_matrix(numeric_df, maxlag=result.lag_order)

    # Variance decomposition
    fevd = estimator.variance_decomposition(result, periods=10)

    return {
        'analysis_type': 'var',
        'variables': result.variable_names,
        'model': {
            'lag_order': result.lag_order,
            'aic': float(result.aic),
            'bic': float(result.bic),
            'is_stable': result.is_stable
        },
        'granger_causality_pvalues': gc_matrix.to_dict(),
        'variance_decomposition_horizon_10': {
            var: fevd[var].iloc[-1].to_dict() for var in result.variable_names
        },
        'note': 'Granger causality tests PREDICTION, not true causality'
    }


def run_granger_analysis(
    df: pd.DataFrame,
    target: str,
    cause: str,
    estimator: TimeSeriesEstimator,
    maxlag: int = 10
) -> dict:
    """Run Granger causality test."""
    if target not in df.columns or cause not in df.columns:
        raise ValueError(f"Variables not found in data")

    result = estimator.granger_causality(df, target, cause, maxlag)

    return {
        'analysis_type': 'granger_causality',
        'caused': result.caused,
        'causing': result.causing,
        'lags': result.lags,
        'f_statistic': float(result.f_statistic),
        'p_value': float(result.p_value),
        'is_granger_causal': result.is_granger_causal,
        'CRITICAL_WARNING': result.warning
    }


def run_cointegration_analysis(
    df: pd.DataFrame,
    estimator: TimeSeriesEstimator,
    method: str = 'johansen'
) -> dict:
    """Run cointegration analysis."""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        raise ValueError("Cointegration requires at least 2 variables")

    result = estimator.cointegration_test(numeric_df, method=method)

    output = {
        'analysis_type': 'cointegration',
        'method': result.method,
        'test_statistic': float(result.test_statistic),
        'is_cointegrated': result.is_cointegrated,
        'cointegration_rank': result.rank,
        'critical_values': {k: float(v) for k, v in result.critical_values.items()}
    }

    if result.p_value is not np.nan:
        output['p_value'] = float(result.p_value)

    if result.is_cointegrated and result.cointegrating_vectors is not None:
        output['cointegrating_vectors'] = result.cointegrating_vectors.tolist()

    return output


def run_comprehensive_analysis(df: pd.DataFrame, estimator: TimeSeriesEstimator) -> dict:
    """Run all analyses."""
    numeric_df = df.select_dtypes(include=[np.number])

    results = {
        'analysis_type': 'comprehensive',
        'timestamp': datetime.now().isoformat(),
        'data_info': {
            'n_observations': len(df),
            'n_variables': numeric_df.shape[1],
            'variables': list(numeric_df.columns)
        },
        'stationarity': run_stationarity_analysis(numeric_df, estimator),
        'cointegration': run_cointegration_analysis(numeric_df, estimator)
    }

    if numeric_df.shape[1] >= 2:
        results['var'] = run_var_analysis(numeric_df, estimator)

    return results


def print_results(results: dict):
    """Pretty print results."""
    print("\n" + "=" * 60)
    print(f"TIME SERIES ANALYSIS RESULTS")
    print(f"Analysis Type: {results['analysis_type']}")
    print("=" * 60)

    if results['analysis_type'] == 'stationarity':
        for var, tests in results['variables'].items():
            print(f"\n{var}:")
            print(f"  ADF: stat={tests['adf_test']['statistic']:.4f}, "
                  f"p={tests['adf_test']['p_value']:.4f}")
            print(f"  KPSS: stat={tests['kpss_test']['statistic']:.4f}, "
                  f"p={tests['kpss_test']['p_value']:.4f}")
            print(f"  Integration order: I({tests['integration_order']})")
            print(f"  Conclusion: {tests['combined_conclusion']}")

    elif results['analysis_type'] == 'arima':
        print(f"\nTarget: {results['target']}")
        print(f"Order: ARIMA{tuple(results['model']['order'])}")
        print(f"AIC: {results['model']['aic']:.2f}")
        print(f"BIC: {results['model']['bic']:.2f}")
        print("\nCoefficients:")
        for k, v in results['coefficients'].items():
            pval = results['p_values'].get(k, np.nan)
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            print(f"  {k}: {v:.4f} (p={pval:.4f}) {sig}")
        print("\nDiagnostics:")
        for test, passed in results['diagnostics'].items():
            status = 'PASS' if passed else 'FAIL' if passed is False else 'N/A'
            print(f"  {test}: {status}")

    elif results['analysis_type'] == 'var':
        print(f"\nVariables: {', '.join(results['variables'])}")
        print(f"Lag order: {results['model']['lag_order']}")
        print(f"Stable: {results['model']['is_stable']}")
        print("\nGranger Causality (p-values):")
        gc_df = pd.DataFrame(results['granger_causality_pvalues'])
        print(gc_df.round(4).to_string())
        print(f"\n*** {results['note']} ***")

    elif results['analysis_type'] == 'granger_causality':
        print(f"\nTest: {results['causing']} -> {results['caused']}")
        print(f"Lags: {results['lags']}")
        print(f"F-statistic: {results['f_statistic']:.4f}")
        print(f"P-value: {results['p_value']:.4f}")
        print(f"Conclusion: {'Granger causal' if results['is_granger_causal'] else 'Not Granger causal'}")
        print(f"\n*** {results['CRITICAL_WARNING']} ***")

    elif results['analysis_type'] == 'cointegration':
        print(f"\nMethod: {results['method']}")
        print(f"Test statistic: {results['test_statistic']:.4f}")
        if 'p_value' in results:
            print(f"P-value: {results['p_value']:.4f}")
        print(f"Critical values: {results['critical_values']}")
        print(f"Cointegration rank: {results['cointegration_rank']}")
        print(f"Cointegrated: {results['is_cointegrated']}")


def main():
    parser = argparse.ArgumentParser(
        description='Time Series Analysis CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv --analysis stationarity
  %(prog)s data.csv --analysis arima --target gdp
  %(prog)s data.csv --analysis var
  %(prog)s data.csv --analysis granger --target y --cause x
  %(prog)s data.csv --analysis cointegration
  %(prog)s data.csv --analysis all
        """
    )

    parser.add_argument('data', help='Path to CSV data file')
    parser.add_argument('--analysis', '-a',
                       choices=['stationarity', 'arima', 'var', 'granger', 'cointegration', 'all'],
                       default='stationarity',
                       help='Type of analysis to run')
    parser.add_argument('--target', '-t', help='Target variable (for ARIMA or Granger)')
    parser.add_argument('--cause', '-c', help='Causing variable (for Granger)')
    parser.add_argument('--date-col', '-d', help='Date column name')
    parser.add_argument('--maxlag', type=int, default=10, help='Maximum lags')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress console output')

    args = parser.parse_args()

    # Load data
    df = load_data(args.data, args.date_col)

    # Initialize estimator
    estimator = TimeSeriesEstimator()

    # Run analysis
    if args.analysis == 'stationarity':
        results = run_stationarity_analysis(df, estimator)
    elif args.analysis == 'arima':
        if not args.target:
            parser.error("--target required for ARIMA analysis")
        results = run_arima_analysis(df, args.target, estimator)
    elif args.analysis == 'var':
        results = run_var_analysis(df, estimator, args.maxlag)
    elif args.analysis == 'granger':
        if not args.target or not args.cause:
            parser.error("--target and --cause required for Granger analysis")
        results = run_granger_analysis(df, args.target, args.cause, estimator, args.maxlag)
    elif args.analysis == 'cointegration':
        results = run_cointegration_analysis(df, estimator)
    elif args.analysis == 'all':
        results = run_comprehensive_analysis(df, estimator)
    else:
        parser.error(f"Unknown analysis type: {args.analysis}")

    # Output
    if not args.quiet:
        print_results(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
