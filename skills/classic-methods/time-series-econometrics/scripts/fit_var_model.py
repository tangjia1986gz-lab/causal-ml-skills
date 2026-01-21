#!/usr/bin/env python3
"""
VAR Model Fitting and Diagnostics Script

Fits Vector Autoregression models with:
- Automatic lag selection
- Stability checking
- Granger causality testing
- Impulse response functions
- Forecast error variance decomposition

Usage:
    python fit_var_model.py data.csv
    python fit_var_model.py data.csv --maxlag 8 --granger
    python fit_var_model.py data.csv --irf --periods 24
    python fit_var_model.py data.csv --fevd --forecast 12
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from time_series_estimator import TimeSeriesEstimator, VARResult


def fit_and_diagnose_var(
    data: pd.DataFrame,
    estimator: TimeSeriesEstimator,
    maxlags: Optional[int] = None,
    ic: str = 'aic'
) -> Dict[str, Any]:
    """
    Fit VAR model with comprehensive diagnostics.

    Returns
    -------
    dict with model results and diagnostics
    """
    # Fit VAR
    var_result = estimator.fit_var(data, maxlags=maxlags, ic=ic)

    # Basic info
    results = {
        'model_info': {
            'variables': var_result.variable_names,
            'n_variables': len(var_result.variable_names),
            'lag_order': var_result.lag_order,
            'aic': var_result.aic,
            'bic': var_result.bic,
            'log_likelihood': var_result.log_likelihood
        },
        'stability': {
            'is_stable': var_result.is_stable,
            'max_eigenvalue_modulus': float(np.max(np.abs(var_result.eigenvalues))),
            'all_eigenvalue_moduli': [float(np.abs(e)) for e in var_result.eigenvalues]
        }
    }

    # Residual diagnostics
    model = var_result.model

    # Whiteness test (no autocorrelation)
    try:
        whiteness = model.test_whiteness(nlags=var_result.lag_order + 5)
        results['residual_diagnostics'] = {
            'whiteness_test': {
                'statistic': float(whiteness.test_statistic),
                'p_value': float(whiteness.pvalue),
                'passed': whiteness.pvalue > 0.05
            }
        }
    except:
        results['residual_diagnostics'] = {'whiteness_test': 'Could not compute'}

    # Normality test
    try:
        normality = model.test_normality()
        results['residual_diagnostics']['normality_test'] = {
            'statistic': float(normality.test_statistic),
            'p_value': float(normality.pvalue),
            'passed': normality.pvalue > 0.05
        }
    except:
        results['residual_diagnostics']['normality_test'] = 'Could not compute'

    return results, var_result


def run_granger_analysis(
    data: pd.DataFrame,
    var_result: VARResult,
    estimator: TimeSeriesEstimator
) -> Dict[str, Any]:
    """
    Comprehensive Granger causality analysis.
    """
    print("\n" + "=" * 70)
    print("GRANGER CAUSALITY ANALYSIS")
    print("=" * 70)
    print("\n*** WARNING: Granger causality tests PREDICTION, not true causality ***\n")

    # Get p-value matrix
    gc_matrix = estimator.granger_causality_matrix(data, maxlag=var_result.lag_order)

    print("P-values Matrix (rows = caused, columns = causing):")
    print("-" * 70)
    print(gc_matrix.round(4).to_string())

    # Identify significant relationships
    significant = []
    for caused in gc_matrix.index:
        for causing in gc_matrix.columns:
            if caused != causing:
                pval = gc_matrix.loc[caused, causing]
                if not np.isnan(pval) and pval < 0.05:
                    significant.append({
                        'causing': causing,
                        'caused': caused,
                        'p_value': pval
                    })

    print("\n" + "-" * 70)
    print("Significant Granger-Causal Relationships (p < 0.05):")
    print("-" * 70)

    if significant:
        for rel in significant:
            print(f"  {rel['causing']} -> {rel['caused']} (p = {rel['p_value']:.4f})")
    else:
        print("  No significant relationships found")

    print("\n*** Remember: This is about PREDICTION, not causation! ***")

    return {
        'granger_matrix': gc_matrix.to_dict(),
        'significant_relationships': significant,
        'warning': 'Granger causality tests prediction, not true causality'
    }


def compute_impulse_responses(
    var_result: VARResult,
    estimator: TimeSeriesEstimator,
    periods: int = 20,
    plot: bool = True
) -> Dict[str, Any]:
    """
    Compute and optionally plot impulse response functions.
    """
    print("\n" + "=" * 70)
    print("IMPULSE RESPONSE FUNCTIONS")
    print("=" * 70)

    irf_results = estimator.impulse_response(var_result, periods=periods)

    # Summary
    print(f"\nHorizon: {periods} periods")
    print(f"Variables: {', '.join(var_result.variable_names)}")
    print("Note: Using orthogonalized (Cholesky) decomposition - ORDER MATTERS!")

    if plot:
        # Plot IRFs
        n_vars = len(var_result.variable_names)
        fig, axes = plt.subplots(n_vars, n_vars, figsize=(4 * n_vars, 4 * n_vars))

        if n_vars == 1:
            axes = np.array([[axes]])
        elif n_vars == 2:
            axes = axes.reshape(2, 2)

        for i, response in enumerate(var_result.variable_names):
            for j, impulse in enumerate(var_result.variable_names):
                key = f"{impulse} -> {response}"
                irf = irf_results[key]

                ax = axes[i, j]
                ax.plot(irf.irf_values, 'b-', linewidth=2)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax.set_title(f'{impulse} -> {response}', fontsize=10)
                ax.set_xlabel('Periods')
                ax.set_ylabel('Response')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('irf_plots.png', dpi=150, bbox_inches='tight')
        print("\nIRF plots saved to: irf_plots.png")
        plt.close()

    return {
        'periods': periods,
        'irf': {k: v.irf_values.tolist() for k, v in irf_results.items()}
    }


def compute_variance_decomposition(
    var_result: VARResult,
    estimator: TimeSeriesEstimator,
    periods: int = 20,
    plot: bool = True
) -> Dict[str, Any]:
    """
    Compute forecast error variance decomposition.
    """
    print("\n" + "=" * 70)
    print("FORECAST ERROR VARIANCE DECOMPOSITION")
    print("=" * 70)

    fevd_results = estimator.variance_decomposition(var_result, periods=periods)

    # Print tables
    for var in var_result.variable_names:
        print(f"\nVariance Decomposition of {var}:")
        print("-" * 60)
        df = fevd_results[var]

        # Show selected horizons
        horizons = [1, 5, 10, min(periods, 20)]
        horizons = [h for h in horizons if h <= periods]

        print(df.loc[horizons].round(2).to_string())

    if plot:
        # Stacked bar plot
        n_vars = len(var_result.variable_names)
        fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))

        if n_vars == 1:
            axes = [axes]

        for idx, var in enumerate(var_result.variable_names):
            df = fevd_results[var]
            df.plot(kind='bar', stacked=True, ax=axes[idx], legend=(idx == n_vars - 1))
            axes[idx].set_title(f'FEVD: {var}')
            axes[idx].set_xlabel('Horizon')
            axes[idx].set_ylabel('Percent')
            axes[idx].set_xticklabels([str(i+1) for i in range(len(df))], rotation=0)

        plt.tight_layout()
        plt.savefig('fevd_plots.png', dpi=150, bbox_inches='tight')
        print("\nFEVD plots saved to: fevd_plots.png")
        plt.close()

    return {
        'periods': periods,
        'fevd': {k: v.to_dict() for k, v in fevd_results.items()}
    }


def generate_forecasts(
    var_result: VARResult,
    estimator: TimeSeriesEstimator,
    steps: int = 10,
    plot: bool = True,
    original_data: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Generate and optionally plot forecasts.
    """
    print("\n" + "=" * 70)
    print(f"VAR FORECASTS ({steps} periods)")
    print("=" * 70)

    forecast = estimator.forecast_var(var_result, steps=steps)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame(
        forecast['mean'],
        columns=var_result.variable_names,
        index=range(1, steps + 1)
    )
    forecast_df.index.name = 'Horizon'

    print("\nPoint Forecasts:")
    print(forecast_df.round(4).to_string())

    if plot and original_data is not None:
        n_vars = len(var_result.variable_names)
        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4 * n_vars))

        if n_vars == 1:
            axes = [axes]

        for idx, var in enumerate(var_result.variable_names):
            ax = axes[idx]

            # Historical data
            hist = original_data[var].values
            ax.plot(range(len(hist)), hist, 'b-', label='Historical', linewidth=1.5)

            # Forecast
            fc_idx = range(len(hist), len(hist) + steps)
            fc_mean = forecast['mean'][:, idx]
            fc_lower = forecast['lower'][:, idx]
            fc_upper = forecast['upper'][:, idx]

            ax.plot(fc_idx, fc_mean, 'r-', label='Forecast', linewidth=2)
            ax.fill_between(fc_idx, fc_lower, fc_upper, color='red', alpha=0.2, label='95% CI')

            ax.set_title(f'{var}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('var_forecast.png', dpi=150, bbox_inches='tight')
        print("\nForecast plot saved to: var_forecast.png")
        plt.close()

    return {
        'steps': steps,
        'forecasts': forecast_df.to_dict()
    }


def print_var_summary(results: Dict[str, Any], var_result: VARResult):
    """Print comprehensive VAR summary."""
    print("\n" + "=" * 70)
    print("VAR MODEL SUMMARY")
    print("=" * 70)

    info = results['model_info']
    print(f"\nVariables: {', '.join(info['variables'])}")
    print(f"Lag Order: {info['lag_order']}")
    print(f"Sample Size: {var_result.model.nobs}")

    print("\nModel Fit:")
    print(f"  AIC: {info['aic']:.4f}")
    print(f"  BIC: {info['bic']:.4f}")
    print(f"  Log-likelihood: {info['log_likelihood']:.4f}")

    stab = results['stability']
    print("\nStability:")
    print(f"  Is Stable: {stab['is_stable']}")
    print(f"  Max Eigenvalue Modulus: {stab['max_eigenvalue_modulus']:.4f}")
    if not stab['is_stable']:
        print("  *** WARNING: Model is not stable! Results may be unreliable. ***")

    diag = results['residual_diagnostics']
    print("\nResidual Diagnostics:")
    if isinstance(diag.get('whiteness_test'), dict):
        wt = diag['whiteness_test']
        status = 'PASS' if wt['passed'] else 'FAIL'
        print(f"  Whiteness (no autocorr): {status} (p = {wt['p_value']:.4f})")
    if isinstance(diag.get('normality_test'), dict):
        nt = diag['normality_test']
        status = 'PASS' if nt['passed'] else 'FAIL'
        print(f"  Normality: {status} (p = {nt['p_value']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='VAR Model Fitting and Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('data', help='Path to CSV file')
    parser.add_argument('--maxlag', type=int, default=None, help='Maximum lag order')
    parser.add_argument('--ic', choices=['aic', 'bic', 'hqic', 'fpe'], default='aic',
                       help='Information criterion for lag selection')
    parser.add_argument('--granger', action='store_true', help='Run Granger causality tests')
    parser.add_argument('--irf', action='store_true', help='Compute impulse response functions')
    parser.add_argument('--fevd', action='store_true', help='Compute variance decomposition')
    parser.add_argument('--forecast', type=int, default=0, help='Forecast horizon')
    parser.add_argument('--periods', type=int, default=20, help='Periods for IRF/FEVD')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--date-col', help='Date column name')
    parser.add_argument('--variables', nargs='+', help='Specific variables to include')
    parser.add_argument('--output', '-o', help='Output JSON file')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    if args.date_col and args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col])
        df.set_index(args.date_col, inplace=True)

    # Select variables
    if args.variables:
        df = df[args.variables]
    else:
        df = df.select_dtypes(include=[np.number])

    if df.shape[1] < 2:
        print("Error: VAR requires at least 2 numeric variables")
        sys.exit(1)

    # Drop missing
    df = df.dropna()

    print(f"Data: {len(df)} observations, {df.shape[1]} variables")
    print(f"Variables: {', '.join(df.columns)}")

    # Initialize estimator
    estimator = TimeSeriesEstimator()

    # Fit VAR
    results, var_result = fit_and_diagnose_var(df, estimator, args.maxlag, args.ic)
    print_var_summary(results, var_result)

    # Additional analyses
    all_results = {'var': results}

    if args.granger:
        granger_results = run_granger_analysis(df, var_result, estimator)
        all_results['granger'] = granger_results

    if args.irf:
        irf_results = compute_impulse_responses(
            var_result, estimator, args.periods, not args.no_plot
        )
        all_results['irf'] = irf_results

    if args.fevd:
        fevd_results = compute_variance_decomposition(
            var_result, estimator, args.periods, not args.no_plot
        )
        all_results['fevd'] = fevd_results

    if args.forecast > 0:
        forecast_results = generate_forecasts(
            var_result, estimator, args.forecast, not args.no_plot, df
        )
        all_results['forecast'] = forecast_results

    # Save results
    if args.output:
        import json

        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            return obj

        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=convert_numpy)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
