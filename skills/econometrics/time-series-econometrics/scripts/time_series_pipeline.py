#!/usr/bin/env python3
"""
Time Series Econometrics Pipeline - Self-Contained

Complete time series analysis workflow using statsmodels:
- Unit Root Tests (ADF, KPSS)
- ARIMA Model Selection and Estimation
- VAR Analysis with Granger Causality
- Cointegration Testing
- Forecasting with Confidence Intervals

Usage:
    python time_series_pipeline.py --demo
    python time_series_pipeline.py --data data.csv --target y --model arima

Dependencies:
    pip install statsmodels pandas numpy scipy matplotlib

Reference:
    Hamilton, J. (1994). Time Series Analysis.
    Enders, W. (2014). Applied Econometric Time Series.
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Set non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Data Simulation
# =============================================================================

def simulate_arima_data(
    n: int = 200,
    ar_params: List[float] = [0.7],
    ma_params: List[float] = [0.3],
    d: int = 0,
    sigma: float = 1.0,
    seed: int = 42
) -> pd.Series:
    """
    Simulate ARIMA(p, d, q) process.

    Parameters
    ----------
    n : int
        Number of observations
    ar_params : list
        AR coefficients [phi_1, phi_2, ...]
    ma_params : list
        MA coefficients [theta_1, theta_2, ...]
    d : int
        Integration order
    sigma : float
        Error standard deviation
    seed : int
        Random seed

    Returns
    -------
    pd.Series
        Simulated time series with datetime index
    """
    np.random.seed(seed)

    p = len(ar_params)
    q = len(ma_params)

    # Generate longer series to account for burn-in
    n_total = n + 100

    errors = np.random.randn(n_total) * sigma
    y = np.zeros(n_total)

    # Generate ARMA process
    for t in range(max(p, q), n_total):
        ar_term = sum(ar_params[i] * y[t-i-1] for i in range(p))
        ma_term = sum(ma_params[i] * errors[t-i-1] for i in range(q))
        y[t] = ar_term + ma_term + errors[t]

    # Remove burn-in
    y = y[100:]

    # Integrate (cumsum) d times
    for _ in range(d):
        y = np.cumsum(y)

    # Create datetime index
    dates = pd.date_range('2000-01-01', periods=n, freq='M')

    return pd.Series(y, index=dates, name='y')


def simulate_var_data(
    n: int = 300,
    coef_matrix: np.ndarray = None,
    sigma: np.ndarray = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate VAR(1) process.

    Parameters
    ----------
    n : int
        Number of observations
    coef_matrix : ndarray
        VAR coefficient matrix (k x k)
    sigma : ndarray
        Error covariance matrix
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Simulated multivariate time series
    """
    np.random.seed(seed)

    if coef_matrix is None:
        # Default: y1 Granger-causes y2
        coef_matrix = np.array([
            [0.5, 0.0],   # y1 equation
            [0.4, 0.3]    # y2 equation: y1 causes y2
        ])

    if sigma is None:
        sigma = np.eye(2) * 0.5

    k = coef_matrix.shape[0]
    n_total = n + 100

    # Generate errors
    errors = np.random.multivariate_normal(np.zeros(k), sigma, n_total)

    # Initialize
    y = np.zeros((n_total, k))

    # Generate VAR(1)
    for t in range(1, n_total):
        y[t] = coef_matrix @ y[t-1] + errors[t]

    # Remove burn-in
    y = y[100:]

    # Create DataFrame
    dates = pd.date_range('2000-01-01', periods=n, freq='M')
    columns = [f'y{i+1}' for i in range(k)]

    return pd.DataFrame(y, index=dates, columns=columns)


def simulate_cointegrated_data(
    n: int = 300,
    beta: float = 0.5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate cointegrated I(1) series.

    Parameters
    ----------
    n : int
        Number of observations
    beta : float
        Cointegrating coefficient
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Two cointegrated series
    """
    np.random.seed(seed)

    # Common stochastic trend (random walk)
    trend = np.cumsum(np.random.randn(n))

    # Cointegrated series
    y1 = trend + np.random.randn(n) * 0.5
    y2 = beta * trend + np.random.randn(n) * 0.5

    dates = pd.date_range('2000-01-01', periods=n, freq='M')

    return pd.DataFrame({'y1': y1, 'y2': y2}, index=dates)


# =============================================================================
# Unit Root Tests
# =============================================================================

def adf_test(
    series: pd.Series,
    regression: str = 'c',
    autolag: str = 'AIC'
) -> Dict[str, Any]:
    """
    Augmented Dickey-Fuller test for unit root.

    H0: Series has a unit root (non-stationary)
    H1: Series is stationary

    Parameters
    ----------
    series : pd.Series
        Time series to test
    regression : str
        'c' (constant), 'ct' (constant+trend), 'n' (none)
    autolag : str
        Lag selection method

    Returns
    -------
    dict
        Test results
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series.dropna(), regression=regression, autolag=autolag)

    return {
        'test': 'ADF',
        'statistic': result[0],
        'pvalue': result[1],
        'lags_used': result[2],
        'nobs': result[3],
        'critical_values': result[4],
        'reject_h0': result[1] < 0.05,
        'conclusion': 'Stationary' if result[1] < 0.05 else 'Unit Root'
    }


def kpss_test(
    series: pd.Series,
    regression: str = 'c',
    nlags: str = 'auto'
) -> Dict[str, Any]:
    """
    KPSS test for stationarity.

    H0: Series is stationary
    H1: Series has a unit root

    Parameters
    ----------
    series : pd.Series
        Time series to test
    regression : str
        'c' (constant), 'ct' (constant+trend)
    nlags : str
        Number of lags or 'auto'

    Returns
    -------
    dict
        Test results
    """
    from statsmodels.tsa.stattools import kpss

    result = kpss(series.dropna(), regression=regression, nlags=nlags)

    return {
        'test': 'KPSS',
        'statistic': result[0],
        'pvalue': result[1],
        'lags_used': result[2],
        'critical_values': result[3],
        'reject_h0': result[1] < 0.05,
        'conclusion': 'Non-stationary' if result[1] < 0.05 else 'Stationary'
    }


def comprehensive_unit_root_test(series: pd.Series, name: str = 'Series') -> Dict[str, Any]:
    """
    Run both ADF and KPSS tests and provide combined interpretation.
    """
    adf = adf_test(series)
    kpss = kpss_test(series)

    # Combined interpretation
    if adf['reject_h0'] and not kpss['reject_h0']:
        combined = 'Stationary (both tests agree)'
    elif not adf['reject_h0'] and kpss['reject_h0']:
        combined = 'Unit root (both tests agree)'
    elif adf['reject_h0'] and kpss['reject_h0']:
        combined = 'Trend-stationary or structural break'
    else:
        combined = 'Inconclusive (tests disagree)'

    return {
        'name': name,
        'adf': adf,
        'kpss': kpss,
        'combined_conclusion': combined
    }


# =============================================================================
# ARIMA Models
# =============================================================================

def fit_arima(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = None
) -> Dict[str, Any]:
    """
    Fit ARIMA or SARIMA model.

    Parameters
    ----------
    series : pd.Series
        Time series data
    order : tuple
        (p, d, q) order
    seasonal_order : tuple, optional
        (P, D, Q, s) seasonal order

    Returns
    -------
    dict
        Model results
    """
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if seasonal_order:
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    else:
        model = ARIMA(series, order=order)

    result = model.fit()

    return {
        'model': f"ARIMA{order}" + (f"x{seasonal_order}" if seasonal_order else ""),
        'result': result,
        'aic': result.aic,
        'bic': result.bic,
        'params': result.params.to_dict(),
        'pvalues': result.pvalues.to_dict(),
        'resid': result.resid
    }


def select_arima_order(
    series: pd.Series,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    criterion: str = 'aic'
) -> Dict[str, Any]:
    """
    Select optimal ARIMA order using information criteria.

    Parameters
    ----------
    series : pd.Series
        Time series data
    max_p, max_d, max_q : int
        Maximum orders to consider
    criterion : str
        'aic' or 'bic'

    Returns
    -------
    dict
        Best model and comparison table
    """
    from statsmodels.tsa.arima.model import ARIMA

    results = []

    for d in range(max_d + 1):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fit = model.fit()
                    results.append({
                        'order': (p, d, q),
                        'aic': fit.aic,
                        'bic': fit.bic,
                        'loglik': fit.llf
                    })
                except Exception:
                    continue

    if not results:
        return {'error': 'No models could be fitted'}

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(criterion)

    best = results_df.iloc[0]

    return {
        'best_order': best['order'],
        'best_aic': best['aic'],
        'best_bic': best['bic'],
        'all_results': results_df.head(10)
    }


def arima_diagnostics(result) -> Dict[str, Any]:
    """
    Diagnostic tests for ARIMA residuals.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy.stats import jarque_bera

    resid = result.resid.dropna()

    # Ljung-Box test for autocorrelation
    lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)

    # Jarque-Bera test for normality
    jb_result = jarque_bera(resid)
    jb_stat = jb_result.statistic if hasattr(jb_result, 'statistic') else jb_result[0]
    jb_pvalue = jb_result.pvalue if hasattr(jb_result, 'pvalue') else jb_result[1]
    skew = float(pd.Series(resid).skew())
    kurt = float(pd.Series(resid).kurtosis())

    return {
        'ljung_box': {
            'lag_10': {'statistic': lb.loc[10, 'lb_stat'], 'pvalue': lb.loc[10, 'lb_pvalue']},
            'lag_20': {'statistic': lb.loc[20, 'lb_stat'], 'pvalue': lb.loc[20, 'lb_pvalue']}
        },
        'jarque_bera': {
            'statistic': jb_stat,
            'pvalue': jb_pvalue,
            'skewness': skew,
            'kurtosis': kurt
        },
        'residual_std': resid.std(),
        'no_autocorrelation': lb.loc[10, 'lb_pvalue'] > 0.05,
        'normality': jb_pvalue > 0.05
    }


# =============================================================================
# VAR Models
# =============================================================================

def fit_var(
    df: pd.DataFrame,
    maxlags: int = 10,
    ic: str = 'aic'
) -> Dict[str, Any]:
    """
    Fit VAR model with optimal lag selection.

    Parameters
    ----------
    df : pd.DataFrame
        Multivariate time series
    maxlags : int
        Maximum lags to consider
    ic : str
        Information criterion for lag selection

    Returns
    -------
    dict
        VAR results
    """
    from statsmodels.tsa.api import VAR

    model = VAR(df)

    # Select lag order
    lag_order = model.select_order(maxlags=maxlags)
    optimal_lag = getattr(lag_order, ic)

    # Fit model
    result = model.fit(optimal_lag)

    return {
        'model': f'VAR({optimal_lag})',
        'result': result,
        'lag_order': optimal_lag,
        'lag_selection': {
            'aic': lag_order.aic,
            'bic': lag_order.bic,
            'hqic': lag_order.hqic,
            'fpe': lag_order.fpe
        },
        'aic': result.aic,
        'bic': result.bic
    }


def granger_causality_test(
    var_result,
    cause: str,
    effect: str
) -> Dict[str, Any]:
    """
    Test Granger causality from cause to effect.

    Parameters
    ----------
    var_result : VARResults
        Fitted VAR model
    cause : str
        Causing variable name
    effect : str
        Effect variable name

    Returns
    -------
    dict
        Test results
    """
    gc = var_result.test_causality(effect, [cause], kind='f')

    return {
        'cause': cause,
        'effect': effect,
        'f_statistic': gc.test_statistic,
        'pvalue': gc.pvalue,
        'df': gc.df,
        'granger_causes': gc.pvalue < 0.05,
        'conclusion': f"{cause} Granger-causes {effect}" if gc.pvalue < 0.05 else f"No Granger causality from {cause} to {effect}"
    }


def impulse_response(
    var_result,
    periods: int = 20,
    impulse: str = None,
    response: str = None
) -> Dict[str, Any]:
    """
    Compute impulse response functions.
    """
    irf = var_result.irf(periods=periods)

    return {
        'irf': irf,
        'periods': periods,
        'variables': var_result.names
    }


# =============================================================================
# Cointegration Tests
# =============================================================================

def engle_granger_test(
    y1: pd.Series,
    y2: pd.Series
) -> Dict[str, Any]:
    """
    Engle-Granger two-step cointegration test.

    H0: No cointegration
    H1: Series are cointegrated
    """
    from statsmodels.tsa.stattools import coint

    stat, pvalue, crit = coint(y1, y2)

    return {
        'test': 'Engle-Granger',
        'statistic': stat,
        'pvalue': pvalue,
        'critical_values': {'1%': crit[0], '5%': crit[1], '10%': crit[2]},
        'cointegrated': pvalue < 0.05,
        'conclusion': 'Cointegrated' if pvalue < 0.05 else 'Not cointegrated'
    }


def johansen_test(
    df: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1
) -> Dict[str, Any]:
    """
    Johansen cointegration test.

    Parameters
    ----------
    df : pd.DataFrame
        Multivariate time series
    det_order : int
        Deterministic trend order (-1, 0, 1)
    k_ar_diff : int
        Number of lagged differences

    Returns
    -------
    dict
        Test results
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    result = coint_johansen(df, det_order=det_order, k_ar_diff=k_ar_diff)

    n_vars = df.shape[1]
    trace_results = []
    max_eig_results = []

    for i in range(n_vars):
        trace_results.append({
            'r': i,
            'statistic': result.lr1[i],
            'critical_value_5pct': result.cvt[i, 1],
            'reject': result.lr1[i] > result.cvt[i, 1]
        })
        max_eig_results.append({
            'r': i,
            'statistic': result.lr2[i],
            'critical_value_5pct': result.cvm[i, 1],
            'reject': result.lr2[i] > result.cvm[i, 1]
        })

    # Determine cointegration rank
    trace_rank = sum(1 for r in trace_results if r['reject'])
    max_eig_rank = sum(1 for r in max_eig_results if r['reject'])

    return {
        'test': 'Johansen',
        'trace_test': trace_results,
        'max_eigenvalue_test': max_eig_results,
        'trace_rank': trace_rank,
        'max_eig_rank': max_eig_rank,
        'eigenvectors': result.evec,
        'eigenvalues': result.eig
    }


# =============================================================================
# Forecasting
# =============================================================================

def forecast_arima(
    result,
    steps: int = 12,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Generate forecasts with confidence intervals.
    """
    forecast = result.get_forecast(steps=steps)
    ci = forecast.conf_int(alpha=alpha)

    return {
        'mean': forecast.predicted_mean,
        'lower': ci.iloc[:, 0],
        'upper': ci.iloc[:, 1],
        'steps': steps,
        'alpha': alpha
    }


# =============================================================================
# Output Functions
# =============================================================================

def print_unit_root_results(results: Dict[str, Any]):
    """Print unit root test results."""
    print("\n" + "="*60)
    print(f"UNIT ROOT TESTS: {results['name']}")
    print("="*60)

    adf = results['adf']
    print(f"\nADF Test (H0: Unit root):")
    print(f"  Statistic: {adf['statistic']:.4f}")
    print(f"  P-value: {adf['pvalue']:.4f}")
    print(f"  Lags: {adf['lags_used']}")
    print(f"  Conclusion: {adf['conclusion']}")

    kpss = results['kpss']
    print(f"\nKPSS Test (H0: Stationary):")
    print(f"  Statistic: {kpss['statistic']:.4f}")
    print(f"  P-value: {kpss['pvalue']:.4f}")
    print(f"  Conclusion: {kpss['conclusion']}")

    print(f"\nCombined: {results['combined_conclusion']}")


def print_arima_results(arima_result: Dict[str, Any], diagnostics: Dict[str, Any]):
    """Print ARIMA estimation results."""
    print("\n" + "="*60)
    print(f"ARIMA MODEL: {arima_result['model']}")
    print("="*60)

    print(f"\nInformation Criteria:")
    print(f"  AIC: {arima_result['aic']:.4f}")
    print(f"  BIC: {arima_result['bic']:.4f}")

    print(f"\nParameters:")
    for param, value in arima_result['params'].items():
        pval = arima_result['pvalues'].get(param, np.nan)
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"  {param}: {value:.4f}{stars} (p={pval:.4f})")

    print(f"\nDiagnostics:")
    lb = diagnostics['ljung_box']
    print(f"  Ljung-Box (lag 10): stat={lb['lag_10']['statistic']:.2f}, p={lb['lag_10']['pvalue']:.4f}")
    print(f"  No autocorrelation: {'Yes' if diagnostics['no_autocorrelation'] else 'No'}")

    jb = diagnostics['jarque_bera']
    print(f"  Jarque-Bera: stat={jb['statistic']:.2f}, p={jb['pvalue']:.4f}")
    print(f"  Normality: {'Yes' if diagnostics['normality'] else 'No'}")


def print_var_results(var_result: Dict[str, Any], gc_results: List[Dict[str, Any]]):
    """Print VAR estimation and Granger causality results."""
    print("\n" + "="*60)
    print(f"VAR MODEL: {var_result['model']}")
    print("="*60)

    print(f"\nLag Selection:")
    for ic, lag in var_result['lag_selection'].items():
        print(f"  {ic.upper()}: {lag}")

    print(f"\nModel Fit:")
    print(f"  AIC: {var_result['aic']:.4f}")
    print(f"  BIC: {var_result['bic']:.4f}")

    print(f"\nGranger Causality Tests:")
    print("-"*50)
    for gc in gc_results:
        status = "YES" if gc['granger_causes'] else "NO"
        print(f"  {gc['cause']} -> {gc['effect']}: F={gc['f_statistic']:.4f}, p={gc['pvalue']:.4f} [{status}]")


def print_cointegration_results(eg_result: Dict[str, Any], johansen_result: Dict[str, Any]):
    """Print cointegration test results."""
    print("\n" + "="*60)
    print("COINTEGRATION TESTS")
    print("="*60)

    print(f"\nEngle-Granger Test:")
    print(f"  Statistic: {eg_result['statistic']:.4f}")
    print(f"  P-value: {eg_result['pvalue']:.4f}")
    print(f"  Conclusion: {eg_result['conclusion']}")

    print(f"\nJohansen Test (Trace):")
    for r in johansen_result['trace_test']:
        status = "REJECT" if r['reject'] else "FAIL TO REJECT"
        print(f"  r <= {r['r']}: stat={r['statistic']:.4f}, CV={r['critical_value_5pct']:.4f} [{status}]")

    print(f"\nCointegration rank (trace): {johansen_result['trace_rank']}")


def generate_latex_table(results: Dict[str, Any], save_path: Optional[str] = None) -> str:
    """Generate LaTeX table for time series results."""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Time Series Analysis Results}
\label{tab:ts_results}
\begin{tabular}{lcc}
\toprule
Test/Statistic & Value & Conclusion \\
\midrule
"""

    if 'adf' in results:
        adf = results['adf']
        latex += f"ADF Statistic & {adf['statistic']:.4f} & {adf['conclusion']} \\\\\n"
        latex += f"ADF P-value & {adf['pvalue']:.4f} & \\\\\n"

    if 'kpss' in results:
        kpss = results['kpss']
        latex += f"KPSS Statistic & {kpss['statistic']:.4f} & {kpss['conclusion']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {save_path}")

    return latex


def plot_diagnostics(
    series: pd.Series,
    result,
    save_path: Optional[str] = None
):
    """Plot time series diagnostics."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original series
    axes[0, 0].plot(series)
    axes[0, 0].set_title('Original Series')
    axes[0, 0].set_xlabel('Date')

    # Residuals
    axes[0, 1].plot(result.resid)
    axes[0, 1].axhline(0, color='red', linestyle='--')
    axes[0, 1].set_title('Residuals')

    # ACF of residuals
    plot_acf(result.resid.dropna(), ax=axes[1, 0], lags=20)
    axes[1, 0].set_title('ACF of Residuals')

    # Residual histogram
    axes[1, 1].hist(result.resid.dropna(), bins=30, density=True, alpha=0.7)
    axes[1, 1].set_title('Residual Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Diagnostics plot saved to: {save_path}")

    plt.close()


# =============================================================================
# Full Analysis Pipeline
# =============================================================================

def run_full_analysis(
    series: pd.Series = None,
    df: pd.DataFrame = None,
    analysis_type: str = 'arima',
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete time series analysis.

    Parameters
    ----------
    series : pd.Series
        Univariate time series (for ARIMA)
    df : pd.DataFrame
        Multivariate time series (for VAR)
    analysis_type : str
        'arima', 'var', or 'cointegration'
    output_dir : str, optional
        Output directory

    Returns
    -------
    dict
        Complete analysis results
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print("\n" + "="*70)
    print("TIME SERIES ANALYSIS")
    print("="*70)

    if analysis_type == 'arima' and series is not None:
        # Step 1: Unit root tests
        print("\n--- Step 1: Unit Root Tests ---")
        ur_results = comprehensive_unit_root_test(series, 'Series')
        results['unit_root'] = ur_results
        print_unit_root_results(ur_results)

        # Step 2: Model selection
        print("\n--- Step 2: ARIMA Order Selection ---")
        selection = select_arima_order(series, max_p=3, max_d=1, max_q=3)
        results['selection'] = selection
        print(f"Best order: {selection['best_order']}")
        print(f"Best AIC: {selection['best_aic']:.4f}")

        # Step 3: Fit best model
        print("\n--- Step 3: Model Estimation ---")
        arima_result = fit_arima(series, order=selection['best_order'])
        results['arima'] = arima_result

        # Step 4: Diagnostics
        diagnostics = arima_diagnostics(arima_result['result'])
        results['diagnostics'] = diagnostics
        print_arima_results(arima_result, diagnostics)

        # Step 5: Forecast
        print("\n--- Step 5: Forecasting ---")
        forecast = forecast_arima(arima_result['result'], steps=12)
        results['forecast'] = forecast
        print(f"Forecast (next 12 periods):")
        print(forecast['mean'].to_string())

        # Save outputs
        if output_dir:
            plot_diagnostics(series, arima_result['result'], str(output_dir / 'arima_diagnostics.png'))

    elif analysis_type == 'var' and df is not None:
        # Step 1: Unit root tests for each variable
        print("\n--- Step 1: Unit Root Tests ---")
        for col in df.columns:
            ur = comprehensive_unit_root_test(df[col], col)
            print_unit_root_results(ur)
            results[f'unit_root_{col}'] = ur

        # Step 2: Fit VAR
        print("\n--- Step 2: VAR Estimation ---")
        var_result = fit_var(df)
        results['var'] = var_result

        # Step 3: Granger causality
        print("\n--- Step 3: Granger Causality ---")
        gc_results = []
        for cause in df.columns:
            for effect in df.columns:
                if cause != effect:
                    gc = granger_causality_test(var_result['result'], cause, effect)
                    gc_results.append(gc)
        results['granger_causality'] = gc_results

        print_var_results(var_result, gc_results)

    elif analysis_type == 'cointegration' and df is not None:
        # Step 1: Unit root tests
        print("\n--- Step 1: Unit Root Tests ---")
        for col in df.columns:
            ur = comprehensive_unit_root_test(df[col], col)
            print_unit_root_results(ur)

        # Step 2: Engle-Granger test
        print("\n--- Step 2: Cointegration Tests ---")
        eg = engle_granger_test(df.iloc[:, 0], df.iloc[:, 1])
        results['engle_granger'] = eg

        # Step 3: Johansen test
        joh = johansen_test(df)
        results['johansen'] = joh

        print_cointegration_results(eg, joh)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Time Series Econometrics Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run ARIMA demo
    python time_series_pipeline.py --demo --model arima

    # Run VAR demo
    python time_series_pipeline.py --demo --model var

    # Run cointegration demo
    python time_series_pipeline.py --demo --model cointegration

    # Analyze real data
    python time_series_pipeline.py --data data.csv --target y --model arima
"""
    )

    parser.add_argument('--demo', action='store_true', help='Run demo with simulated data')
    parser.add_argument('--model', type=str, choices=['arima', 'var', 'cointegration'],
                        default='arima', help='Type of analysis')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--target', type=str, help='Target variable for ARIMA')
    parser.add_argument('--output', type=str, help='Output directory')

    args = parser.parse_args()

    if args.demo:
        print(f"Running {args.model} demo with simulated data...")

        if args.model == 'arima':
            # Simulate ARIMA data
            series = simulate_arima_data(n=200, ar_params=[0.7], ma_params=[0.3], d=1, seed=42)
            print(f"Simulated ARIMA(1,1,1) series: {len(series)} observations")

            run_full_analysis(
                series=series,
                analysis_type='arima',
                output_dir=args.output
            )

        elif args.model == 'var':
            # Simulate VAR data
            df = simulate_var_data(n=300, seed=42)
            print(f"Simulated VAR(1) data: {len(df)} observations, {df.shape[1]} variables")

            run_full_analysis(
                df=df,
                analysis_type='var',
                output_dir=args.output
            )

        elif args.model == 'cointegration':
            # Simulate cointegrated data
            df = simulate_cointegrated_data(n=300, beta=0.5, seed=42)
            print(f"Simulated cointegrated data: {len(df)} observations")

            run_full_analysis(
                df=df,
                analysis_type='cointegration',
                output_dir=args.output
            )

    elif args.data:
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)

        if args.model == 'arima' and args.target:
            series = df[args.target]
            run_full_analysis(series=series, analysis_type='arima', output_dir=args.output)
        elif args.model in ['var', 'cointegration']:
            run_full_analysis(df=df, analysis_type=args.model, output_dir=args.output)
        else:
            parser.print_help()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
