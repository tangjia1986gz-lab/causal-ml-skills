"""
Time Series Econometrics Estimator Module

This module provides tools for time series analysis in causal inference contexts,
including stationarity testing, ARIMA/VAR modeling, cointegration analysis,
and Granger causality testing.

IMPORTANT: Granger causality tests predictive relationships, NOT true causality.
For causal inference, use difference-in-differences, synthetic control, or other
methods designed for causal identification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings


@dataclass
class StationarityResult:
    """Results from stationarity tests."""
    test_name: str
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    is_stationary: bool
    null_hypothesis: str
    lags_used: int = 0
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ARIMAResult:
    """Results from ARIMA model fitting."""
    order: Tuple[int, int, int]
    coefficients: Dict[str, float]
    standard_errors: Dict[str, float]
    p_values: Dict[str, float]
    aic: float
    bic: float
    log_likelihood: float
    residuals: np.ndarray
    fitted_values: np.ndarray
    diagnostics: Dict[str, Any]
    model: Any = None


@dataclass
class VARResult:
    """Results from VAR model fitting."""
    lag_order: int
    variable_names: List[str]
    coefficients: pd.DataFrame
    aic: float
    bic: float
    log_likelihood: float
    is_stable: bool
    eigenvalues: np.ndarray
    residuals: pd.DataFrame
    model: Any = None


@dataclass
class GrangerCausalityResult:
    """Results from Granger causality tests."""
    caused: str
    causing: str
    lags: int
    f_statistic: float
    p_value: float
    is_granger_causal: bool
    warning: str = "Granger causality tests PREDICTION, not true causality"


@dataclass
class CointegrationResult:
    """Results from cointegration tests."""
    method: str
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    is_cointegrated: bool
    cointegrating_vectors: Optional[np.ndarray] = None
    adjustment_coefficients: Optional[np.ndarray] = None
    rank: int = 0


@dataclass
class ImpulseResponseResult:
    """Results from impulse response analysis."""
    periods: int
    impulse_variable: str
    response_variable: str
    irf_values: np.ndarray
    lower_ci: Optional[np.ndarray] = None
    upper_ci: Optional[np.ndarray] = None
    orthogonalized: bool = True


class TimeSeriesEstimator:
    """
    Comprehensive time series econometrics estimator.

    Provides methods for:
    - Stationarity and unit root testing
    - ARIMA model selection and fitting
    - VAR model estimation and analysis
    - Cointegration testing and VECM
    - Granger causality analysis
    - Impulse response functions

    Example
    -------
    >>> estimator = TimeSeriesEstimator()
    >>> result = estimator.test_unit_root(series, test='adf')
    >>> print(f"Stationary: {result.is_stationary}")
    """

    def __init__(self, random_state: int = 42):
        """Initialize the estimator."""
        self.random_state = random_state
        np.random.seed(random_state)

    # ==================== STATIONARITY TESTS ====================

    def test_unit_root(
        self,
        series: Union[pd.Series, np.ndarray],
        test: str = 'adf',
        regression: str = 'c',
        maxlag: Optional[int] = None,
        autolag: str = 'AIC'
    ) -> StationarityResult:
        """
        Test for unit root (non-stationarity).

        Parameters
        ----------
        series : array-like
            Time series data
        test : str
            'adf' = Augmented Dickey-Fuller
            'pp' = Phillips-Perron
            'za' = Zivot-Andrews (with structural break)
        regression : str
            'c' = constant only
            'ct' = constant + trend
            'n' = no deterministic terms
        maxlag : int, optional
            Maximum lag for augmented terms
        autolag : str
            Criterion for automatic lag selection ('AIC', 'BIC', 't-stat')

        Returns
        -------
        StationarityResult
            Test results with statistic, p-value, and conclusion
        """
        from statsmodels.tsa.stattools import adfuller

        series = np.asarray(series).flatten()
        series = series[~np.isnan(series)]

        if test == 'adf':
            result = adfuller(
                series,
                maxlag=maxlag,
                regression=regression,
                autolag=autolag
            )

            return StationarityResult(
                test_name='Augmented Dickey-Fuller',
                test_statistic=result[0],
                p_value=result[1],
                critical_values={'1%': result[4]['1%'],
                                '5%': result[4]['5%'],
                                '10%': result[4]['10%']},
                is_stationary=result[1] < 0.05,
                null_hypothesis='Unit root (non-stationary)',
                lags_used=result[2],
                additional_info={'n_obs': result[3], 'ic_best': result[5] if len(result) > 5 else None}
            )

        elif test == 'pp':
            from statsmodels.tsa.stattools import PhillipsPerron
            result = PhillipsPerron(series, trend=regression)

            return StationarityResult(
                test_name='Phillips-Perron',
                test_statistic=result.stat,
                p_value=result.pvalue,
                critical_values=result.critical_values,
                is_stationary=result.pvalue < 0.05,
                null_hypothesis='Unit root (non-stationary)',
                lags_used=result.lags
            )

        elif test == 'za':
            from statsmodels.tsa.stattools import zivot_andrews
            result = zivot_andrews(series, regression=regression, maxlag=maxlag)

            return StationarityResult(
                test_name='Zivot-Andrews (structural break)',
                test_statistic=result[0],
                p_value=result[1],
                critical_values={'1%': result[4]['1%'],
                                '5%': result[4]['5%'],
                                '10%': result[4]['10%']},
                is_stationary=result[1] < 0.05,
                null_hypothesis='Unit root with structural break',
                lags_used=result[3],
                additional_info={'break_point': result[2]}
            )

        else:
            raise ValueError(f"Unknown test: {test}. Use 'adf', 'pp', or 'za'.")

    def test_stationarity(
        self,
        series: Union[pd.Series, np.ndarray],
        regression: str = 'c',
        nlags: Union[str, int] = 'auto'
    ) -> StationarityResult:
        """
        KPSS test for stationarity (null: stationary).

        NOTE: KPSS null is OPPOSITE of ADF - null is stationarity.

        Parameters
        ----------
        series : array-like
            Time series data
        regression : str
            'c' = level stationarity
            'ct' = trend stationarity
        nlags : int or 'auto'
            Number of lags for Newey-West

        Returns
        -------
        StationarityResult
        """
        from statsmodels.tsa.stattools import kpss

        series = np.asarray(series).flatten()
        series = series[~np.isnan(series)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p_value, lags, crit = kpss(series, regression=regression, nlags=nlags)

        return StationarityResult(
            test_name='KPSS',
            test_statistic=stat,
            p_value=p_value,
            critical_values={'10%': crit['10%'], '5%': crit['5%'],
                            '2.5%': crit['2.5%'], '1%': crit['1%']},
            is_stationary=p_value > 0.05,  # Note: reversed interpretation
            null_hypothesis='Stationarity (stationary)',
            lags_used=lags
        )

    def determine_integration_order(
        self,
        series: Union[pd.Series, np.ndarray],
        max_d: int = 2,
        significance: float = 0.05
    ) -> int:
        """
        Determine integration order by successive differencing.

        Parameters
        ----------
        series : array-like
            Time series data
        max_d : int
            Maximum differencing order to try
        significance : float
            Significance level for ADF test

        Returns
        -------
        int
            Integration order (0 = stationary, 1 = I(1), etc.)
        """
        series = np.asarray(series).flatten()
        series = series[~np.isnan(series)]
        current = series.copy()

        for d in range(max_d + 1):
            if d > 0:
                current = np.diff(current)

            result = self.test_unit_root(current, test='adf')

            if result.p_value < significance:
                return d

        warnings.warn(f"Series may be I({max_d + 1}) or higher. Consider more differencing.")
        return max_d

    # ==================== ARIMA MODELS ====================

    def fit_arima(
        self,
        series: Union[pd.Series, np.ndarray],
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        trend: str = 'c',
        auto_select: bool = False,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        criterion: str = 'aic'
    ) -> ARIMAResult:
        """
        Fit ARIMA or SARIMA model.

        Parameters
        ----------
        series : array-like
            Time series data
        order : tuple, optional
            (p, d, q) order. If None and auto_select=True, determined automatically.
        seasonal_order : tuple, optional
            (P, D, Q, m) seasonal order for SARIMA
        trend : str
            'n' = no trend, 'c' = constant, 't' = trend, 'ct' = both
        auto_select : bool
            If True, automatically select order using information criterion
        max_p, max_d, max_q : int
            Maximum orders for auto selection
        criterion : str
            'aic' or 'bic' for auto selection

        Returns
        -------
        ARIMAResult
        """
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        series = pd.Series(series).dropna()

        # Auto-select order if needed
        if order is None and auto_select:
            try:
                import pmdarima as pm
                auto_model = pm.auto_arima(
                    series,
                    start_p=0, start_q=0,
                    max_p=max_p, max_d=max_d, max_q=max_q,
                    seasonal=seasonal_order is not None,
                    m=seasonal_order[3] if seasonal_order else 1,
                    information_criterion=criterion,
                    suppress_warnings=True,
                    stepwise=True
                )
                order = auto_model.order
                if seasonal_order is not None:
                    seasonal_order = auto_model.seasonal_order
            except ImportError:
                # Fallback to grid search
                order = self._select_arima_order(series, max_p, max_d, max_q, criterion)

        if order is None:
            order = (1, 0, 0)

        # Fit model
        if seasonal_order is not None:
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                          trend=trend, enforce_stationarity=False,
                          enforce_invertibility=False)
        else:
            model = ARIMA(series, order=order, trend=trend)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit()

        # Run diagnostics
        diagnostics = self._arima_diagnostics(result)

        return ARIMAResult(
            order=order,
            coefficients=result.params.to_dict(),
            standard_errors=result.bse.to_dict(),
            p_values=result.pvalues.to_dict(),
            aic=result.aic,
            bic=result.bic,
            log_likelihood=result.llf,
            residuals=result.resid.values,
            fitted_values=result.fittedvalues.values,
            diagnostics=diagnostics,
            model=result
        )

    def _select_arima_order(
        self,
        series: pd.Series,
        max_p: int,
        max_d: int,
        max_q: int,
        criterion: str
    ) -> Tuple[int, int, int]:
        """Grid search for ARIMA order."""
        from statsmodels.tsa.arima.model import ARIMA

        # Determine d
        d = self.determine_integration_order(series, max_d)

        best_score = np.inf
        best_order = (0, d, 0)

        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(series, order=(p, d, q))
                        result = model.fit()

                        score = result.aic if criterion == 'aic' else result.bic

                        if score < best_score:
                            best_score = score
                            best_order = (p, d, q)
                except:
                    continue

        return best_order

    def _arima_diagnostics(self, result) -> Dict[str, Any]:
        """Run diagnostic tests on ARIMA residuals."""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from scipy import stats

        resid = result.resid.dropna()

        diagnostics = {}

        # Ljung-Box test
        try:
            lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
            diagnostics['ljung_box'] = {
                'lag_10': {'statistic': lb['lb_stat'].iloc[0], 'p_value': lb['lb_pvalue'].iloc[0]},
                'lag_20': {'statistic': lb['lb_stat'].iloc[-1], 'p_value': lb['lb_pvalue'].iloc[-1]},
                'passed': lb['lb_pvalue'].iloc[-1] > 0.05
            }
        except:
            diagnostics['ljung_box'] = {'error': 'Could not compute'}

        # Jarque-Bera normality test
        try:
            jb_stat, jb_pvalue, skew, kurt = stats.jarque_bera(resid)
            diagnostics['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_pvalue,
                'skewness': skew,
                'kurtosis': kurt,
                'passed': jb_pvalue > 0.05
            }
        except:
            diagnostics['jarque_bera'] = {'error': 'Could not compute'}

        # Heteroskedasticity (ARCH test)
        try:
            from statsmodels.stats.diagnostic import het_arch
            arch_stat, arch_pvalue, _, _ = het_arch(resid, nlags=5)
            diagnostics['arch_test'] = {
                'statistic': arch_stat,
                'p_value': arch_pvalue,
                'passed': arch_pvalue > 0.05
            }
        except:
            diagnostics['arch_test'] = {'error': 'Could not compute'}

        return diagnostics

    # ==================== VAR MODELS ====================

    def fit_var(
        self,
        data: pd.DataFrame,
        maxlags: Optional[int] = None,
        ic: str = 'aic',
        trend: str = 'c'
    ) -> VARResult:
        """
        Fit Vector Autoregression model.

        Parameters
        ----------
        data : DataFrame
            Multivariate time series
        maxlags : int, optional
            Maximum lags to consider (default: 12*(T/100)^0.25)
        ic : str
            Information criterion: 'aic', 'bic', 'hqic', 'fpe'
        trend : str
            'c' = constant, 'ct' = constant + trend, 'n' = none

        Returns
        -------
        VARResult
        """
        from statsmodels.tsa.api import VAR

        data = data.dropna()

        if maxlags is None:
            maxlags = int(12 * (len(data) / 100) ** 0.25)

        model = VAR(data)

        # Select lag order
        lag_order = model.select_order(maxlags=maxlags)
        optimal_lag = lag_order.selected_orders[ic]

        if optimal_lag == 0:
            optimal_lag = 1

        # Fit model
        result = model.fit(maxlags=optimal_lag, trend=trend)

        # Check stability
        eigenvalues = np.abs(result.roots)
        is_stable = all(eigenvalues < 1)

        return VARResult(
            lag_order=optimal_lag,
            variable_names=list(data.columns),
            coefficients=pd.DataFrame(result.params),
            aic=result.aic,
            bic=result.bic,
            log_likelihood=result.llf,
            is_stable=is_stable,
            eigenvalues=result.roots,
            residuals=pd.DataFrame(result.resid, columns=data.columns),
            model=result
        )

    def fit_vecm(
        self,
        data: pd.DataFrame,
        coint_rank: int,
        k_ar_diff: int = 1,
        deterministic: str = 'ci'
    ) -> Dict[str, Any]:
        """
        Fit Vector Error Correction Model.

        Parameters
        ----------
        data : DataFrame
            Multivariate I(1) time series
        coint_rank : int
            Number of cointegrating relationships
        k_ar_diff : int
            Number of lagged differences
        deterministic : str
            'n' = none, 'co' = constant outside EC, 'ci' = constant inside EC,
            'lo' = linear trend outside, 'li' = linear trend inside

        Returns
        -------
        dict with VECM results
        """
        from statsmodels.tsa.vector_ar.vecm import VECM

        data = data.dropna()

        model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank,
                     deterministic=deterministic)
        result = model.fit()

        return {
            'cointegrating_vectors': pd.DataFrame(
                result.beta,
                index=data.columns,
                columns=[f'CE{i+1}' for i in range(coint_rank)]
            ),
            'adjustment_coefficients': pd.DataFrame(
                result.alpha,
                index=data.columns,
                columns=[f'CE{i+1}' for i in range(coint_rank)]
            ),
            'short_run_dynamics': result.gamma,
            'aic': result.aic,
            'bic': result.bic,
            'log_likelihood': result.llf,
            'model': result
        }

    # ==================== GRANGER CAUSALITY ====================

    def granger_causality(
        self,
        data: pd.DataFrame,
        caused: str,
        causing: Union[str, List[str]],
        maxlag: int = 10
    ) -> GrangerCausalityResult:
        """
        Test for Granger causality.

        WARNING: This tests PREDICTION, not true causality!
        X Granger-causes Y means X helps predict Y, NOT that X causes Y.

        Parameters
        ----------
        data : DataFrame
            Time series data with named columns
        caused : str
            Name of potentially caused variable
        causing : str or list
            Name(s) of potentially causing variable(s)
        maxlag : int
            Number of lags to test

        Returns
        -------
        GrangerCausalityResult
        """
        from statsmodels.tsa.stattools import grangercausalitytests

        if isinstance(causing, str):
            causing = [causing]

        # For pairwise test
        if len(causing) == 1:
            test_data = data[[caused, causing[0]]].dropna()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

            # Get best result (usually at selected lag)
            best_lag = maxlag
            best_pvalue = results[maxlag][0]['ssr_ftest'][1]
            best_fstat = results[maxlag][0]['ssr_ftest'][0]

            return GrangerCausalityResult(
                caused=caused,
                causing=causing[0],
                lags=best_lag,
                f_statistic=best_fstat,
                p_value=best_pvalue,
                is_granger_causal=best_pvalue < 0.05
            )

        # For block Granger causality (multiple causing variables)
        else:
            # Fit VAR and use test_causality
            var_result = self.fit_var(data, maxlags=maxlag)
            test_result = var_result.model.test_causality(caused, causing=causing, kind='f')

            return GrangerCausalityResult(
                caused=caused,
                causing=str(causing),
                lags=var_result.lag_order,
                f_statistic=test_result.test_statistic,
                p_value=test_result.pvalue,
                is_granger_causal=test_result.pvalue < 0.05
            )

    def granger_causality_matrix(
        self,
        data: pd.DataFrame,
        maxlag: int = 10,
        significance: float = 0.05
    ) -> pd.DataFrame:
        """
        Create matrix of pairwise Granger causality p-values.

        Returns
        -------
        DataFrame with p-values (rows = caused, cols = causing)
        """
        variables = data.columns.tolist()
        n_vars = len(variables)
        p_values = np.zeros((n_vars, n_vars))

        for i, caused in enumerate(variables):
            for j, causing in enumerate(variables):
                if i != j:
                    try:
                        result = self.granger_causality(data, caused, causing, maxlag)
                        p_values[i, j] = result.p_value
                    except:
                        p_values[i, j] = np.nan
                else:
                    p_values[i, j] = np.nan

        df = pd.DataFrame(p_values, index=variables, columns=variables)
        df.index.name = 'Caused'
        df.columns.name = 'Causing'

        return df

    # ==================== IMPULSE RESPONSE ====================

    def impulse_response(
        self,
        var_result: VARResult,
        periods: int = 20,
        orthogonalized: bool = True
    ) -> Dict[str, ImpulseResponseResult]:
        """
        Compute impulse response functions from fitted VAR.

        Parameters
        ----------
        var_result : VARResult
            Fitted VAR model
        periods : int
            Forecast horizon
        orthogonalized : bool
            If True, use Cholesky decomposition (ordering matters!)

        Returns
        -------
        dict of ImpulseResponseResult for each impulse-response pair
        """
        irf = var_result.model.irf(periods=periods)
        irf_values = irf.orth_irfs if orthogonalized else irf.irfs

        results = {}
        variables = var_result.variable_names

        for i, impulse in enumerate(variables):
            for j, response in enumerate(variables):
                key = f"{impulse} -> {response}"
                results[key] = ImpulseResponseResult(
                    periods=periods,
                    impulse_variable=impulse,
                    response_variable=response,
                    irf_values=irf_values[:, j, i],
                    orthogonalized=orthogonalized
                )

        return results

    def variance_decomposition(
        self,
        var_result: VARResult,
        periods: int = 20
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute forecast error variance decomposition.

        Returns
        -------
        dict of DataFrames (one per variable)
        """
        fevd = var_result.model.fevd(periods=periods)
        decomp = fevd.decomp

        results = {}
        variables = var_result.variable_names

        for i, var in enumerate(variables):
            df = pd.DataFrame(
                decomp[:, :, i] * 100,
                columns=variables,
                index=range(1, periods + 1)
            )
            df.index.name = 'Horizon'
            results[var] = df

        return results

    # ==================== COINTEGRATION ====================

    def cointegration_test(
        self,
        data: pd.DataFrame,
        method: str = 'johansen',
        det_order: int = 0,
        k_ar_diff: int = 1
    ) -> CointegrationResult:
        """
        Test for cointegration.

        Parameters
        ----------
        data : DataFrame
            Multivariate time series (all should be I(1))
        method : str
            'johansen' = Johansen procedure
            'engle-granger' = Engle-Granger two-step (bivariate only)
        det_order : int (Johansen only)
            -1 = no deterministic, 0 = constant, 1 = constant + trend
        k_ar_diff : int
            Number of lagged differences

        Returns
        -------
        CointegrationResult
        """
        data = data.dropna()

        if method == 'johansen':
            from statsmodels.tsa.vector_ar.vecm import coint_johansen

            result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

            # Determine rank using trace test at 5%
            rank = 0
            k = data.shape[1]
            for i in range(k):
                if result.lr1[i] > result.cvt[i, 1]:  # 5% critical value
                    rank = i + 1
                else:
                    break

            return CointegrationResult(
                method='Johansen',
                test_statistic=result.lr1[0],  # First trace statistic
                p_value=np.nan,  # Johansen doesn't provide p-values directly
                critical_values={
                    '10%': result.cvt[0, 0],
                    '5%': result.cvt[0, 1],
                    '1%': result.cvt[0, 2]
                },
                is_cointegrated=rank > 0,
                cointegrating_vectors=result.evec[:, :rank] if rank > 0 else None,
                rank=rank
            )

        elif method == 'engle-granger':
            from statsmodels.tsa.stattools import coint

            if data.shape[1] != 2:
                warnings.warn("Engle-Granger is designed for bivariate analysis. "
                             "Using first two columns.")
                data = data.iloc[:, :2]

            y = data.iloc[:, 0]
            x = data.iloc[:, 1]

            stat, pvalue, crit = coint(y, x)

            return CointegrationResult(
                method='Engle-Granger',
                test_statistic=stat,
                p_value=pvalue,
                critical_values={'1%': crit[0], '5%': crit[1], '10%': crit[2]},
                is_cointegrated=pvalue < 0.05,
                rank=1 if pvalue < 0.05 else 0
            )

        else:
            raise ValueError(f"Unknown method: {method}")

    # ==================== FORECASTING ====================

    def forecast_arima(
        self,
        arima_result: ARIMAResult,
        steps: int,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate forecasts from ARIMA model.

        Returns
        -------
        DataFrame with forecast, lower, and upper bounds
        """
        forecast = arima_result.model.get_forecast(steps=steps)
        mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=alpha)

        return pd.DataFrame({
            'forecast': mean,
            'lower': conf_int.iloc[:, 0],
            'upper': conf_int.iloc[:, 1]
        })

    def forecast_var(
        self,
        var_result: VARResult,
        steps: int,
        alpha: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecasts from VAR model.

        Returns
        -------
        dict with 'mean', 'lower', 'upper' arrays
        """
        model = var_result.model
        last_obs = model.endog[-model.k_ar:]

        forecast = model.forecast(last_obs, steps=steps)
        stderr = model.forecast_interval(last_obs, steps=steps, alpha=alpha)

        return {
            'mean': forecast,
            'lower': stderr[1],
            'upper': stderr[2]
        }


# ==================== CONVENIENCE FUNCTIONS ====================

def test_unit_root(
    series: Union[pd.Series, np.ndarray],
    test: str = 'adf',
    regression: str = 'c'
) -> StationarityResult:
    """Convenience function for unit root testing."""
    estimator = TimeSeriesEstimator()
    return estimator.test_unit_root(series, test=test, regression=regression)


def test_stationarity(
    series: Union[pd.Series, np.ndarray],
    regression: str = 'c'
) -> StationarityResult:
    """Convenience function for KPSS stationarity test."""
    estimator = TimeSeriesEstimator()
    return estimator.test_stationarity(series, regression=regression)


def fit_arima(
    series: Union[pd.Series, np.ndarray],
    order: Optional[Tuple[int, int, int]] = None,
    auto_select: bool = True
) -> ARIMAResult:
    """Convenience function for ARIMA fitting."""
    estimator = TimeSeriesEstimator()
    return estimator.fit_arima(series, order=order, auto_select=auto_select)


def fit_var(
    data: pd.DataFrame,
    maxlags: Optional[int] = None,
    ic: str = 'aic'
) -> VARResult:
    """Convenience function for VAR fitting."""
    estimator = TimeSeriesEstimator()
    return estimator.fit_var(data, maxlags=maxlags, ic=ic)


def fit_vecm(
    data: pd.DataFrame,
    coint_rank: int,
    k_ar_diff: int = 1
) -> Dict[str, Any]:
    """Convenience function for VECM fitting."""
    estimator = TimeSeriesEstimator()
    return estimator.fit_vecm(data, coint_rank=coint_rank, k_ar_diff=k_ar_diff)


def granger_causality(
    data: pd.DataFrame,
    caused: str,
    causing: str,
    maxlag: int = 10
) -> GrangerCausalityResult:
    """Convenience function for Granger causality testing."""
    estimator = TimeSeriesEstimator()
    return estimator.granger_causality(data, caused, causing, maxlag)


def impulse_response(
    var_result: VARResult,
    periods: int = 20
) -> Dict[str, ImpulseResponseResult]:
    """Convenience function for impulse response analysis."""
    estimator = TimeSeriesEstimator()
    return estimator.impulse_response(var_result, periods)


def cointegration_test(
    data: pd.DataFrame,
    method: str = 'johansen'
) -> CointegrationResult:
    """Convenience function for cointegration testing."""
    estimator = TimeSeriesEstimator()
    return estimator.cointegration_test(data, method=method)


if __name__ == "__main__":
    # Example usage
    print("Time Series Econometrics Estimator")
    print("=" * 40)

    # Generate example data
    np.random.seed(42)
    n = 200

    # Create cointegrated series
    z = np.cumsum(np.random.randn(n))  # Common trend
    y = z + np.random.randn(n) * 0.5
    x = z + np.random.randn(n) * 0.5

    data = pd.DataFrame({'y': y, 'x': x})

    # Run tests
    estimator = TimeSeriesEstimator()

    print("\n1. Unit Root Tests")
    print("-" * 40)
    adf_y = estimator.test_unit_root(y, test='adf')
    print(f"ADF test for y: stat={adf_y.test_statistic:.4f}, p={adf_y.p_value:.4f}")
    print(f"  Stationary: {adf_y.is_stationary}")

    print("\n2. Cointegration Test")
    print("-" * 40)
    coint = estimator.cointegration_test(data, method='johansen')
    print(f"Johansen test: rank={coint.rank}")
    print(f"  Cointegrated: {coint.is_cointegrated}")

    print("\n3. VAR Model")
    print("-" * 40)
    var_result = estimator.fit_var(data)
    print(f"Optimal lag order: {var_result.lag_order}")
    print(f"Stable: {var_result.is_stable}")

    print("\n4. Granger Causality")
    print("-" * 40)
    gc = estimator.granger_causality(data, 'y', 'x')
    print(f"x -> y: F={gc.f_statistic:.4f}, p={gc.p_value:.4f}")
    print(f"  Granger causal: {gc.is_granger_causal}")
    print(f"  WARNING: {gc.warning}")
