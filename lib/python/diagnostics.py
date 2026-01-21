"""
Shared diagnostic functions for causal inference estimators.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DiagnosticResult:
    """Standard diagnostic test result."""
    test_name: str
    statistic: float
    p_value: float
    passed: bool
    threshold: float
    interpretation: str
    details: Dict[str, Any] = None


def parallel_trends_test(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    time_id: str,
    unit_id: str,
    treatment_time: int,
    n_pre_periods: int = 3
) -> DiagnosticResult:
    """
    Test parallel trends assumption for DID.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable name
    treatment : str
        Treatment indicator variable name
    time_id : str
        Time period variable name
    unit_id : str
        Unit identifier variable name
    treatment_time : int
        Time period when treatment starts
    n_pre_periods : int
        Number of pre-treatment periods to test

    Returns
    -------
    DiagnosticResult
        Test result with statistics and interpretation
    """
    pre_data = data[data[time_id] < treatment_time].copy()

    # Group by treatment status and time
    trends = pre_data.groupby([treatment, time_id])[outcome].mean().unstack(level=0)

    if trends.shape[1] < 2:
        return DiagnosticResult(
            test_name="Parallel Trends Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=False,
            threshold=0.05,
            interpretation="Insufficient data: need both treatment and control groups",
            details={"error": "insufficient_groups"}
        )

    # Calculate difference in trends
    trend_diff = trends.iloc[:, 1] - trends.iloc[:, 0]

    # Test if trend difference is constant (i.e., parallel)
    time_points = np.arange(len(trend_diff))
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, trend_diff.values)

    passed = p_value > 0.05  # Non-significant slope suggests parallel trends

    interpretation = (
        "Parallel trends assumption SUPPORTED: no significant divergence in pre-treatment trends"
        if passed else
        "Parallel trends assumption VIOLATED: significant divergence detected in pre-treatment period"
    )

    return DiagnosticResult(
        test_name="Parallel Trends Test (Linear Divergence)",
        statistic=slope,
        p_value=p_value,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            "slope": slope,
            "std_err": std_err,
            "r_squared": r_value**2,
            "trend_differences": trend_diff.to_dict()
        }
    )


def weak_iv_test(
    first_stage_f: float,
    n_instruments: int = 1
) -> DiagnosticResult:
    """
    Test for weak instruments using first-stage F-statistic.

    Parameters
    ----------
    first_stage_f : float
        F-statistic from first stage regression
    n_instruments : int
        Number of instruments

    Returns
    -------
    DiagnosticResult
        Test result with Stock-Yogo critical values
    """
    # Stock-Yogo critical values for 10% maximal IV size
    # Simplified: using rule of thumb F > 10
    threshold = 10.0
    passed = first_stage_f > threshold

    interpretation = (
        f"Instruments are STRONG: F = {first_stage_f:.2f} > {threshold}"
        if passed else
        f"WEAK INSTRUMENTS WARNING: F = {first_stage_f:.2f} < {threshold}. Consider LIML or other robust methods."
    )

    return DiagnosticResult(
        test_name="Weak Instrument Test (Stock-Yogo)",
        statistic=first_stage_f,
        p_value=np.nan,  # Not a p-value test
        passed=passed,
        threshold=threshold,
        interpretation=interpretation,
        details={
            "n_instruments": n_instruments,
            "rule": "Stock-Yogo rule of thumb: F > 10"
        }
    )


def mccrary_density_test(
    running_variable: np.ndarray,
    cutoff: float = 0,
    n_bins: int = 50
) -> DiagnosticResult:
    """
    McCrary density test for RD manipulation.

    Simplified implementation - tests for discontinuity in density at cutoff.

    Parameters
    ----------
    running_variable : np.ndarray
        The running variable
    cutoff : float
        The RD cutoff value
    n_bins : int
        Number of bins for histogram

    Returns
    -------
    DiagnosticResult
        Test result indicating potential manipulation
    """
    below = running_variable[running_variable < cutoff]
    above = running_variable[running_variable >= cutoff]

    # Simple density comparison near cutoff
    bandwidth = (running_variable.max() - running_variable.min()) / n_bins
    near_below = np.sum((cutoff - bandwidth <= running_variable) & (running_variable < cutoff))
    near_above = np.sum((running_variable >= cutoff) & (running_variable < cutoff + bandwidth))

    # Chi-squared test for equal density
    expected = (near_below + near_above) / 2
    if expected > 0:
        chi_sq = ((near_below - expected)**2 + (near_above - expected)**2) / expected
        p_value = 1 - stats.chi2.cdf(chi_sq, df=1)
    else:
        chi_sq = 0
        p_value = 1.0

    passed = p_value > 0.05

    interpretation = (
        "No evidence of manipulation at cutoff"
        if passed else
        "POTENTIAL MANIPULATION: significant density discontinuity at cutoff"
    )

    return DiagnosticResult(
        test_name="McCrary Density Test (Simplified)",
        statistic=chi_sq,
        p_value=p_value,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            "density_below": near_below,
            "density_above": near_above,
            "bandwidth": bandwidth
        }
    )


def balance_test(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str]
) -> Dict[str, DiagnosticResult]:
    """
    Test covariate balance between treatment and control groups.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    treatment : str
        Treatment indicator (0/1)
    covariates : List[str]
        List of covariate names to test

    Returns
    -------
    Dict[str, DiagnosticResult]
        Balance test results for each covariate
    """
    results = {}
    treated = data[data[treatment] == 1]
    control = data[data[treatment] == 0]

    for cov in covariates:
        t_stat, p_value = stats.ttest_ind(
            treated[cov].dropna(),
            control[cov].dropna()
        )

        # Standardized mean difference
        pooled_std = np.sqrt(
            (treated[cov].var() + control[cov].var()) / 2
        )
        smd = (treated[cov].mean() - control[cov].mean()) / pooled_std if pooled_std > 0 else 0

        passed = abs(smd) < 0.1  # Common threshold for balance

        results[cov] = DiagnosticResult(
            test_name=f"Balance Test: {cov}",
            statistic=smd,
            p_value=p_value,
            passed=passed,
            threshold=0.1,
            interpretation=f"{'Balanced' if passed else 'IMBALANCED'}: SMD = {smd:.3f}",
            details={
                "treated_mean": treated[cov].mean(),
                "control_mean": control[cov].mean(),
                "t_statistic": t_stat
            }
        )

    return results


def overidentification_test(
    residuals: np.ndarray,
    instruments: np.ndarray,
    n_endog: int = 1
) -> DiagnosticResult:
    """
    Sargan-Hansen overidentification test for IV.

    Parameters
    ----------
    residuals : np.ndarray
        2SLS residuals
    instruments : np.ndarray
        Instrument matrix
    n_endog : int
        Number of endogenous variables

    Returns
    -------
    DiagnosticResult
        Overidentification test result
    """
    n = len(residuals)
    n_instruments = instruments.shape[1] if instruments.ndim > 1 else 1

    if n_instruments <= n_endog:
        return DiagnosticResult(
            test_name="Sargan-Hansen Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=True,
            threshold=0.05,
            interpretation="Model is exactly identified - overidentification test not applicable",
            details={"reason": "exact_identification"}
        )

    # Sargan statistic
    if instruments.ndim == 1:
        instruments = instruments.reshape(-1, 1)

    proj = instruments @ np.linalg.lstsq(instruments, residuals, rcond=None)[0]
    sargan_stat = n * (proj @ residuals) / (residuals @ residuals)
    df = n_instruments - n_endog
    p_value = 1 - stats.chi2.cdf(sargan_stat, df=df)

    passed = p_value > 0.05

    interpretation = (
        "Overidentification test PASSED: instruments appear valid"
        if passed else
        "Overidentification test FAILED: some instruments may be invalid"
    )

    return DiagnosticResult(
        test_name="Sargan-Hansen Overidentification Test",
        statistic=sargan_stat,
        p_value=p_value,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            "degrees_of_freedom": df,
            "n_instruments": n_instruments,
            "n_endogenous": n_endog
        }
    )


def sensitivity_analysis(
    effect: float,
    se: float,
    r2_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Oster (2019) sensitivity analysis for omitted variable bias.

    Parameters
    ----------
    effect : float
        Estimated treatment effect
    se : float
        Standard error of estimate
    r2_threshold : float
        Hypothetical R-squared from full model

    Returns
    -------
    Dict[str, Any]
        Sensitivity analysis results
    """
    # Simplified sensitivity: how much would unobserved confounding
    # need to explain to nullify the result?
    t_stat = effect / se

    # Breakdown point: proportion of effect explained by confounding
    # that would make result insignificant
    critical_t = 1.96
    breakdown = 1 - (critical_t / abs(t_stat)) if abs(t_stat) > 0 else 0

    return {
        "effect": effect,
        "se": se,
        "t_statistic": t_stat,
        "breakdown_point": max(0, breakdown),
        "interpretation": (
            f"Result robust: confounding would need to explain {breakdown*100:.1f}% "
            f"of the effect to make it insignificant"
            if breakdown > 0.5 else
            f"Result sensitive: confounding explaining just {(1-breakdown)*100:.1f}% "
            f"of the effect would make it insignificant"
        )
    }


def hausman_test(
    fe_coefficients: np.ndarray,
    re_coefficients: np.ndarray,
    fe_cov: np.ndarray,
    re_cov: np.ndarray,
    var_names: Optional[List[str]] = None
) -> DiagnosticResult:
    """
    Hausman test for fixed effects vs random effects model selection.

    The Hausman test compares the coefficients from fixed effects (FE) and
    random effects (RE) estimators. Under the null hypothesis that both
    estimators are consistent, RE is more efficient. If the null is rejected,
    FE should be preferred as RE is inconsistent.

    Parameters
    ----------
    fe_coefficients : np.ndarray
        Coefficient estimates from fixed effects model
    re_coefficients : np.ndarray
        Coefficient estimates from random effects model
    fe_cov : np.ndarray
        Variance-covariance matrix from FE model
    re_cov : np.ndarray
        Variance-covariance matrix from RE model
    var_names : Optional[List[str]]
        Names of the variables being tested

    Returns
    -------
    DiagnosticResult
        Test result with Hausman statistic and recommendation

    Notes
    -----
    H0: RE is consistent and efficient (use RE)
    H1: RE is inconsistent (use FE)

    The test statistic follows chi-squared distribution with k degrees
    of freedom, where k is the number of coefficients being compared.
    """
    fe_coefficients = np.atleast_1d(fe_coefficients)
    re_coefficients = np.atleast_1d(re_coefficients)
    fe_cov = np.atleast_2d(fe_cov)
    re_cov = np.atleast_2d(re_cov)

    # Calculate the difference in coefficients
    b_diff = fe_coefficients - re_coefficients

    # Calculate the difference in variance-covariance matrices
    # Under H0: Var(b_FE - b_RE) = Var(b_FE) - Var(b_RE)
    v_diff = fe_cov - re_cov

    # Check if v_diff is positive semi-definite
    # If not, use absolute values on diagonal (practical adjustment)
    try:
        v_diff_inv = np.linalg.inv(v_diff)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        v_diff_inv = np.linalg.pinv(v_diff)

    # Hausman statistic: (b_FE - b_RE)' * V_diff^-1 * (b_FE - b_RE)
    hausman_stat = float(b_diff @ v_diff_inv @ b_diff)

    # Degrees of freedom = number of coefficients
    df = len(fe_coefficients)

    # P-value from chi-squared distribution
    p_value = 1 - stats.chi2.cdf(hausman_stat, df=df)

    # Determine which model to use
    passed = p_value > 0.05  # "passed" means RE is appropriate

    if passed:
        interpretation = (
            f"Hausman test NOT REJECTED (p = {p_value:.4f}): "
            f"Random Effects model is consistent and efficient. "
            f"Use RE estimator."
        )
    else:
        interpretation = (
            f"Hausman test REJECTED (p = {p_value:.4f}): "
            f"Random Effects model may be inconsistent due to correlation "
            f"between unit effects and regressors. Use Fixed Effects estimator."
        )

    details = {
        "degrees_of_freedom": df,
        "recommendation": "RE" if passed else "FE",
        "coefficient_differences": b_diff.tolist()
    }

    if var_names is not None:
        details["variable_names"] = var_names

    return DiagnosticResult(
        test_name="Hausman Test (FE vs RE)",
        statistic=hausman_stat,
        p_value=p_value,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details=details
    )


def wooldridge_test(
    data: pd.DataFrame,
    outcome: str,
    regressors: List[str],
    unit_id: str,
    time_id: str,
    demean: bool = True
) -> DiagnosticResult:
    """
    Wooldridge test for serial correlation in panel data models.

    Tests for first-order autocorrelation in the idiosyncratic error term
    of a panel data model. Uses the residuals from a first-differenced
    regression.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset
    outcome : str
        Dependent variable name
    regressors : List[str]
        List of regressor variable names
    unit_id : str
        Unit identifier variable name
    time_id : str
        Time period identifier variable name
    demean : bool
        If True, demean variables by unit before testing

    Returns
    -------
    DiagnosticResult
        Test result indicating presence of serial correlation

    Notes
    -----
    H0: No first-order autocorrelation
    H1: First-order autocorrelation exists

    The test is based on Wooldridge (2002) Econometric Analysis of Cross
    Section and Panel Data, Section 10.5.4.
    """
    data = data.copy().sort_values([unit_id, time_id])

    # First-difference the data
    fd_data = data.groupby(unit_id).apply(
        lambda x: x[[outcome] + regressors].diff().iloc[1:],
        include_groups=False
    ).reset_index(drop=True)

    # Remove any missing values
    fd_data = fd_data.dropna()

    if len(fd_data) < 10:
        return DiagnosticResult(
            test_name="Wooldridge Test for Serial Correlation",
            statistic=np.nan,
            p_value=np.nan,
            passed=False,
            threshold=0.05,
            interpretation="Insufficient data for Wooldridge test after first-differencing",
            details={"error": "insufficient_data", "n_obs": len(fd_data)}
        )

    # Run regression on first-differenced data
    y_fd = fd_data[outcome].values
    X_fd = fd_data[regressors].values
    if X_fd.ndim == 1:
        X_fd = X_fd.reshape(-1, 1)

    # Add constant
    X_fd_const = np.column_stack([np.ones(len(y_fd)), X_fd])

    # OLS to get residuals
    try:
        beta = np.linalg.lstsq(X_fd_const, y_fd, rcond=None)[0]
        residuals = y_fd - X_fd_const @ beta
    except np.linalg.LinAlgError:
        return DiagnosticResult(
            test_name="Wooldridge Test for Serial Correlation",
            statistic=np.nan,
            p_value=np.nan,
            passed=False,
            threshold=0.05,
            interpretation="Numerical error in regression computation",
            details={"error": "numerical_error"}
        )

    # Regress residuals on lagged residuals
    resid_df = pd.DataFrame({'resid': residuals})

    # Get lagged residuals within each unit
    # Since we're working with first-differenced data, we need to track units
    data_with_resid = data.groupby(unit_id).apply(
        lambda x: x.iloc[1:].assign(temp_idx=range(len(x)-1)),
        include_groups=False
    ).reset_index(drop=True)

    if len(data_with_resid) == len(residuals):
        data_with_resid['resid'] = residuals
    else:
        # Fallback: simple lag
        resid_df['resid_lag'] = resid_df['resid'].shift(1)
        resid_df = resid_df.dropna()

        if len(resid_df) < 5:
            return DiagnosticResult(
                test_name="Wooldridge Test for Serial Correlation",
                statistic=np.nan,
                p_value=np.nan,
                passed=False,
                threshold=0.05,
                interpretation="Insufficient data for autocorrelation test",
                details={"error": "insufficient_data"}
            )

        resid_current = resid_df['resid'].values
        resid_lag = resid_df['resid_lag'].values

        # Regression of residuals on lagged residuals
        X_lag = np.column_stack([np.ones(len(resid_lag)), resid_lag])
        try:
            gamma = np.linalg.lstsq(X_lag, resid_current, rcond=None)[0]
        except np.linalg.LinAlgError:
            return DiagnosticResult(
                test_name="Wooldridge Test for Serial Correlation",
                statistic=np.nan,
                p_value=np.nan,
                passed=False,
                threshold=0.05,
                interpretation="Numerical error in autocorrelation regression",
                details={"error": "numerical_error"}
            )

        rho = gamma[1]  # Autocorrelation coefficient

        # Calculate standard error and t-statistic
        resid_fit = X_lag @ gamma
        ssr = np.sum((resid_current - resid_fit) ** 2)
        n = len(resid_current)
        se_rho = np.sqrt(ssr / (n - 2) / np.sum((resid_lag - resid_lag.mean()) ** 2))

        t_stat = rho / se_rho if se_rho > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

        passed = p_value > 0.05

        if passed:
            interpretation = (
                f"No significant serial correlation detected (p = {p_value:.4f}). "
                f"Estimated autocorrelation: {rho:.4f}"
            )
        else:
            interpretation = (
                f"SERIAL CORRELATION DETECTED (p = {p_value:.4f}). "
                f"Estimated first-order autocorrelation: {rho:.4f}. "
                f"Consider using clustered standard errors or AR(1) error structure."
            )

        return DiagnosticResult(
            test_name="Wooldridge Test for Serial Correlation",
            statistic=t_stat,
            p_value=p_value,
            passed=passed,
            threshold=0.05,
            interpretation=interpretation,
            details={
                "autocorrelation_coefficient": rho,
                "standard_error": se_rho,
                "n_observations": n
            }
        )

    # Alternative: simple correlation-based test
    resid_lag = np.roll(residuals, 1)
    resid_lag[0] = np.nan

    valid_idx = ~np.isnan(resid_lag)
    if np.sum(valid_idx) < 5:
        return DiagnosticResult(
            test_name="Wooldridge Test for Serial Correlation",
            statistic=np.nan,
            p_value=np.nan,
            passed=False,
            threshold=0.05,
            interpretation="Insufficient data for autocorrelation test",
            details={"error": "insufficient_data"}
        )

    rho, p_value = stats.pearsonr(residuals[valid_idx], resid_lag[valid_idx])
    passed = p_value > 0.05

    if passed:
        interpretation = (
            f"No significant serial correlation detected (p = {p_value:.4f}). "
            f"Estimated autocorrelation: {rho:.4f}"
        )
    else:
        interpretation = (
            f"SERIAL CORRELATION DETECTED (p = {p_value:.4f}). "
            f"Estimated first-order autocorrelation: {rho:.4f}. "
            f"Consider using clustered standard errors or AR(1) error structure."
        )

    return DiagnosticResult(
        test_name="Wooldridge Test for Serial Correlation",
        statistic=rho,
        p_value=p_value,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            "autocorrelation_coefficient": rho,
            "n_observations": int(np.sum(valid_idx))
        }
    )


def breusch_pagan_test(
    residuals: np.ndarray,
    exog: np.ndarray,
    robust: bool = False
) -> DiagnosticResult:
    """
    Breusch-Pagan test for heteroskedasticity.

    Tests whether the variance of residuals depends on the values of the
    independent variables. Under the null hypothesis, errors are homoskedastic.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals from the regression
    exog : np.ndarray
        Matrix of explanatory variables (including constant if desired)
    robust : bool
        If True, use Koenker's studentized version (more robust to non-normality)

    Returns
    -------
    DiagnosticResult
        Test result indicating presence of heteroskedasticity

    Notes
    -----
    H0: Homoskedasticity (constant variance)
    H1: Heteroskedasticity (variance depends on X)

    The test regresses squared residuals on the explanatory variables.
    A significant relationship indicates heteroskedasticity.
    """
    residuals = np.atleast_1d(residuals)
    exog = np.atleast_2d(exog)

    if exog.shape[0] != len(residuals):
        if exog.shape[1] == len(residuals):
            exog = exog.T

    n = len(residuals)
    k = exog.shape[1]

    # Squared residuals
    resid_sq = residuals ** 2

    if robust:
        # Koenker's studentized version: use (e^2 - sigma^2) / sigma^2
        sigma_sq = np.sum(resid_sq) / n
        dep_var = (resid_sq - sigma_sq) / sigma_sq
    else:
        # Standard BP: use e^2 / sigma^2
        sigma_sq = np.sum(resid_sq) / n
        dep_var = resid_sq / sigma_sq

    # Add constant if not present
    if not np.allclose(exog[:, 0], 1):
        exog = np.column_stack([np.ones(n), exog])
        k += 1

    # Regress transformed squared residuals on X
    try:
        beta = np.linalg.lstsq(exog, dep_var, rcond=None)[0]
        fitted = exog @ beta
        ssr = np.sum((fitted - dep_var.mean()) ** 2)
        sst = np.sum((dep_var - dep_var.mean()) ** 2)
        r_squared = ssr / sst if sst > 0 else 0
    except np.linalg.LinAlgError:
        return DiagnosticResult(
            test_name="Breusch-Pagan Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=False,
            threshold=0.05,
            interpretation="Numerical error in Breusch-Pagan computation",
            details={"error": "numerical_error"}
        )

    # Test statistic
    if robust:
        # Koenker version: n * R^2
        bp_stat = n * r_squared
    else:
        # Standard version: ESS / 2
        bp_stat = ssr / 2

    # Degrees of freedom: number of regressors excluding constant
    df = k - 1

    # P-value from chi-squared distribution
    p_value = 1 - stats.chi2.cdf(bp_stat, df=df)

    passed = p_value > 0.05

    if passed:
        interpretation = (
            f"Breusch-Pagan test NOT REJECTED (p = {p_value:.4f}): "
            f"No evidence of heteroskedasticity. "
            f"Standard errors are likely valid."
        )
    else:
        interpretation = (
            f"Breusch-Pagan test REJECTED (p = {p_value:.4f}): "
            f"HETEROSKEDASTICITY DETECTED. "
            f"Use robust (HC) standard errors or weighted least squares."
        )

    return DiagnosticResult(
        test_name="Breusch-Pagan Test" + (" (Koenker)" if robust else ""),
        statistic=bp_stat,
        p_value=p_value,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            "degrees_of_freedom": df,
            "r_squared_auxiliary": r_squared,
            "n_observations": n,
            "robust_version": robust
        }
    )


def vif_calculation(
    data: pd.DataFrame,
    variables: List[str],
    threshold: float = 10.0
) -> Dict[str, Any]:
    """
    Calculate Variance Inflation Factors (VIF) for multicollinearity diagnostics.

    VIF measures how much the variance of an estimated regression coefficient
    is inflated due to collinearity with other predictors. High VIF values
    indicate problematic multicollinearity.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing the variables
    variables : List[str]
        List of variable names to calculate VIF for
    threshold : float
        VIF threshold for flagging problematic multicollinearity
        Common thresholds: 5 (strict), 10 (moderate), 20 (lenient)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - vif_values: Dict mapping variable names to VIF values
        - problematic_vars: List of variables exceeding threshold
        - mean_vif: Average VIF across all variables
        - max_vif: Maximum VIF value
        - condition_number: Condition number of the correlation matrix
        - interpretation: Overall assessment of multicollinearity

    Notes
    -----
    VIF = 1 / (1 - R^2_j), where R^2_j is the R-squared from regressing
    variable j on all other variables.

    Rules of thumb:
    - VIF = 1: No correlation with other variables
    - VIF < 5: Low multicollinearity
    - 5 <= VIF < 10: Moderate multicollinearity
    - VIF >= 10: High multicollinearity (problematic)
    """
    if len(variables) < 2:
        return {
            "vif_values": {variables[0]: 1.0} if variables else {},
            "problematic_vars": [],
            "mean_vif": 1.0,
            "max_vif": 1.0,
            "condition_number": 1.0,
            "interpretation": "Need at least 2 variables to assess multicollinearity"
        }

    # Extract data for specified variables
    X = data[variables].dropna()

    if len(X) < len(variables) + 1:
        return {
            "vif_values": {v: np.nan for v in variables},
            "problematic_vars": [],
            "mean_vif": np.nan,
            "max_vif": np.nan,
            "condition_number": np.nan,
            "interpretation": "Insufficient observations for VIF calculation"
        }

    vif_values = {}

    for i, var in enumerate(variables):
        # Get the variable as dependent
        y = X[var].values

        # Get all other variables as predictors
        other_vars = [v for v in variables if v != var]
        X_other = X[other_vars].values

        # Add constant
        X_other_const = np.column_stack([np.ones(len(y)), X_other])

        # Regress var on other variables
        try:
            beta = np.linalg.lstsq(X_other_const, y, rcond=None)[0]
            y_hat = X_other_const @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # VIF = 1 / (1 - R^2)
            if r_squared >= 1:
                vif = np.inf
            else:
                vif = 1 / (1 - r_squared)
        except np.linalg.LinAlgError:
            vif = np.nan

        vif_values[var] = vif

    # Identify problematic variables
    problematic_vars = [v for v, vif in vif_values.items()
                        if vif is not None and not np.isnan(vif) and vif >= threshold]

    # Calculate mean and max VIF
    valid_vifs = [v for v in vif_values.values() if v is not None and not np.isnan(v) and not np.isinf(v)]
    mean_vif = np.mean(valid_vifs) if valid_vifs else np.nan
    max_vif = max(valid_vifs) if valid_vifs else np.nan

    # Calculate condition number
    try:
        X_matrix = X[variables].values
        X_standardized = (X_matrix - X_matrix.mean(axis=0)) / X_matrix.std(axis=0)
        eigenvalues = np.linalg.eigvalsh(X_standardized.T @ X_standardized)
        condition_number = np.sqrt(eigenvalues.max() / eigenvalues.min()) if eigenvalues.min() > 0 else np.inf
    except (np.linalg.LinAlgError, ValueError):
        condition_number = np.nan

    # Generate interpretation
    if len(problematic_vars) == 0:
        interpretation = (
            f"No severe multicollinearity detected. "
            f"All VIF values are below {threshold}. "
            f"Mean VIF: {mean_vif:.2f}, Max VIF: {max_vif:.2f}"
        )
    else:
        interpretation = (
            f"MULTICOLLINEARITY WARNING: {len(problematic_vars)} variable(s) "
            f"have VIF >= {threshold}: {problematic_vars}. "
            f"Consider removing or combining collinear variables. "
            f"Max VIF: {max_vif:.2f}"
        )

    return {
        "vif_values": vif_values,
        "problematic_vars": problematic_vars,
        "mean_vif": mean_vif,
        "max_vif": max_vif,
        "condition_number": condition_number,
        "threshold": threshold,
        "interpretation": interpretation
    }


def adf_test(
    series: np.ndarray,
    max_lags: Optional[int] = None,
    regression: str = "c",
    autolag: str = "AIC"
) -> DiagnosticResult:
    """
    Augmented Dickey-Fuller test for unit roots in time series.

    Tests the null hypothesis that a unit root is present in the time series.
    A unit root indicates the series is non-stationary and may need differencing.

    Parameters
    ----------
    series : np.ndarray
        Time series data to test
    max_lags : Optional[int]
        Maximum number of lags to include. If None, uses 12*(n/100)^(1/4)
    regression : str
        Regression specification:
        - "c": Constant only (default)
        - "ct": Constant and trend
        - "ctt": Constant, linear and quadratic trend
        - "n": No constant, no trend
    autolag : str
        Method for automatic lag selection:
        - "AIC": Akaike Information Criterion
        - "BIC": Bayesian Information Criterion
        - None: Use max_lags

    Returns
    -------
    DiagnosticResult
        Test result indicating whether series has unit root

    Notes
    -----
    H0: Unit root exists (series is non-stationary)
    H1: No unit root (series is stationary)

    Critical values depend on the regression specification and sample size.
    """
    series = np.atleast_1d(series).astype(float)
    series = series[~np.isnan(series)]

    n = len(series)
    if n < 10:
        return DiagnosticResult(
            test_name="Augmented Dickey-Fuller Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=False,
            threshold=0.05,
            interpretation="Insufficient data for ADF test (need at least 10 observations)",
            details={"error": "insufficient_data", "n_obs": n}
        )

    # Default max lags
    if max_lags is None:
        max_lags = int(12 * (n / 100) ** 0.25)
    max_lags = min(max_lags, n // 3)

    # First difference of series
    delta_y = np.diff(series)
    y_lag = series[:-1]

    # Prepare lagged differences for augmentation
    if max_lags > 0:
        delta_y_lags = np.column_stack([
            np.roll(delta_y, i)[max_lags:] for i in range(1, max_lags + 1)
        ])
        delta_y_trimmed = delta_y[max_lags:]
        y_lag_trimmed = y_lag[max_lags:]
        n_eff = len(delta_y_trimmed)
    else:
        delta_y_trimmed = delta_y
        y_lag_trimmed = y_lag
        delta_y_lags = np.empty((len(delta_y), 0))
        n_eff = len(delta_y)

    # Build design matrix based on regression type
    if regression == "n":
        X = np.column_stack([y_lag_trimmed.reshape(-1, 1), delta_y_lags]) if max_lags > 0 else y_lag_trimmed.reshape(-1, 1)
    elif regression == "c":
        if max_lags > 0:
            X = np.column_stack([np.ones(n_eff), y_lag_trimmed, delta_y_lags])
        else:
            X = np.column_stack([np.ones(n_eff), y_lag_trimmed])
    elif regression == "ct":
        trend = np.arange(1, n_eff + 1)
        if max_lags > 0:
            X = np.column_stack([np.ones(n_eff), trend, y_lag_trimmed, delta_y_lags])
        else:
            X = np.column_stack([np.ones(n_eff), trend, y_lag_trimmed])
    else:  # ctt
        trend = np.arange(1, n_eff + 1)
        trend_sq = trend ** 2
        if max_lags > 0:
            X = np.column_stack([np.ones(n_eff), trend, trend_sq, y_lag_trimmed, delta_y_lags])
        else:
            X = np.column_stack([np.ones(n_eff), trend, trend_sq, y_lag_trimmed])

    # OLS estimation
    try:
        beta, residuals, rank, s = np.linalg.lstsq(X, delta_y_trimmed, rcond=None)
    except np.linalg.LinAlgError:
        return DiagnosticResult(
            test_name="Augmented Dickey-Fuller Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=False,
            threshold=0.05,
            interpretation="Numerical error in ADF test computation",
            details={"error": "numerical_error"}
        )

    # Get the coefficient on y_{t-1} (gamma in delta_y = gamma * y_{t-1} + ...)
    if regression == "n":
        gamma_idx = 0
    elif regression == "c":
        gamma_idx = 1
    elif regression == "ct":
        gamma_idx = 2
    else:  # ctt
        gamma_idx = 3

    gamma = beta[gamma_idx]

    # Calculate standard error
    resid = delta_y_trimmed - X @ beta
    sigma_sq = np.sum(resid ** 2) / (n_eff - len(beta))

    try:
        cov_beta = sigma_sq * np.linalg.inv(X.T @ X)
        se_gamma = np.sqrt(cov_beta[gamma_idx, gamma_idx])
    except np.linalg.LinAlgError:
        se_gamma = np.nan

    # ADF statistic
    if se_gamma > 0 and not np.isnan(se_gamma):
        adf_stat = gamma / se_gamma
    else:
        adf_stat = np.nan

    # Approximate p-value using MacKinnon critical values
    # These are approximations for n >= 50
    critical_values = {
        "n": {0.01: -2.58, 0.05: -1.95, 0.10: -1.62},
        "c": {0.01: -3.43, 0.05: -2.86, 0.10: -2.57},
        "ct": {0.01: -3.96, 0.05: -3.41, 0.10: -3.12},
        "ctt": {0.01: -4.37, 0.05: -3.83, 0.10: -3.55}
    }

    cv = critical_values.get(regression, critical_values["c"])

    # Approximate p-value by interpolation
    if np.isnan(adf_stat):
        p_value = np.nan
    elif adf_stat <= cv[0.01]:
        p_value = 0.001
    elif adf_stat <= cv[0.05]:
        p_value = 0.01 + (adf_stat - cv[0.01]) / (cv[0.05] - cv[0.01]) * (0.05 - 0.01)
    elif adf_stat <= cv[0.10]:
        p_value = 0.05 + (adf_stat - cv[0.05]) / (cv[0.10] - cv[0.05]) * (0.10 - 0.05)
    else:
        p_value = min(0.99, 0.10 + 0.1 * (adf_stat - cv[0.10]))

    passed = p_value < 0.05  # "passed" means we reject unit root (series is stationary)

    if passed:
        interpretation = (
            f"ADF test REJECTED unit root (p = {p_value:.4f}): "
            f"Series appears STATIONARY. "
            f"Test statistic: {adf_stat:.4f}"
        )
    else:
        interpretation = (
            f"ADF test FAILED to reject unit root (p = {p_value:.4f}): "
            f"Series appears NON-STATIONARY. "
            f"Consider differencing before analysis. "
            f"Test statistic: {adf_stat:.4f}"
        )

    return DiagnosticResult(
        test_name=f"Augmented Dickey-Fuller Test ({regression})",
        statistic=adf_stat,
        p_value=p_value,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            "gamma": gamma,
            "standard_error": se_gamma,
            "n_lags": max_lags,
            "n_observations": n,
            "effective_observations": n_eff,
            "regression_type": regression,
            "critical_values": cv
        }
    )


def cointegration_test(
    y1: np.ndarray,
    y2: np.ndarray,
    max_lags: Optional[int] = None,
    trend: str = "c"
) -> DiagnosticResult:
    """
    Engle-Granger two-step cointegration test.

    Tests whether two non-stationary time series share a long-run equilibrium
    relationship (are cointegrated). If cointegrated, their linear combination
    is stationary despite each series being non-stationary individually.

    Parameters
    ----------
    y1 : np.ndarray
        First time series (treated as dependent in cointegrating regression)
    y2 : np.ndarray
        Second time series (treated as independent)
    max_lags : Optional[int]
        Maximum lags for ADF test on residuals
    trend : str
        Trend specification for cointegrating regression:
        - "n": No trend
        - "c": Constant only (default)
        - "ct": Constant and trend

    Returns
    -------
    DiagnosticResult
        Test result indicating whether series are cointegrated

    Notes
    -----
    H0: No cointegration (residuals have unit root)
    H1: Cointegration exists (residuals are stationary)

    Procedure:
    1. Regress y1 on y2 (cointegrating regression)
    2. Test residuals for unit root using ADF
    3. If residuals are stationary, series are cointegrated

    Critical values are more stringent than standard ADF due to
    generated regressor problem.
    """
    y1 = np.atleast_1d(y1).astype(float)
    y2 = np.atleast_1d(y2).astype(float)

    # Remove NaN
    valid_idx = ~(np.isnan(y1) | np.isnan(y2))
    y1 = y1[valid_idx]
    y2 = y2[valid_idx]

    n = len(y1)
    if n < 20:
        return DiagnosticResult(
            test_name="Engle-Granger Cointegration Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=False,
            threshold=0.05,
            interpretation="Insufficient data for cointegration test (need at least 20 observations)",
            details={"error": "insufficient_data", "n_obs": n}
        )

    # Step 1: Cointegrating regression
    if trend == "n":
        X = y2.reshape(-1, 1)
    elif trend == "c":
        X = np.column_stack([np.ones(n), y2])
    else:  # ct
        t = np.arange(1, n + 1)
        X = np.column_stack([np.ones(n), t, y2])

    try:
        beta = np.linalg.lstsq(X, y1, rcond=None)[0]
        residuals = y1 - X @ beta
    except np.linalg.LinAlgError:
        return DiagnosticResult(
            test_name="Engle-Granger Cointegration Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=False,
            threshold=0.05,
            interpretation="Numerical error in cointegrating regression",
            details={"error": "numerical_error"}
        )

    # Extract cointegrating coefficient
    if trend == "n":
        coint_coef = beta[0]
    elif trend == "c":
        coint_coef = beta[1]
    else:
        coint_coef = beta[2]

    # Step 2: ADF test on residuals
    adf_result = adf_test(residuals, max_lags=max_lags, regression="n")

    adf_stat = adf_result.statistic

    # Engle-Granger critical values (more stringent than standard ADF)
    # Approximate values for 2 variables
    eg_critical_values = {
        0.01: -4.07,
        0.05: -3.37,
        0.10: -3.03
    }

    # Approximate p-value
    if np.isnan(adf_stat):
        p_value = np.nan
    elif adf_stat <= eg_critical_values[0.01]:
        p_value = 0.001
    elif adf_stat <= eg_critical_values[0.05]:
        p_value = 0.01 + (adf_stat - eg_critical_values[0.01]) / (eg_critical_values[0.05] - eg_critical_values[0.01]) * (0.05 - 0.01)
    elif adf_stat <= eg_critical_values[0.10]:
        p_value = 0.05 + (adf_stat - eg_critical_values[0.05]) / (eg_critical_values[0.10] - eg_critical_values[0.05]) * (0.10 - 0.05)
    else:
        p_value = min(0.99, 0.10 + 0.2 * (adf_stat - eg_critical_values[0.10]))

    passed = p_value < 0.05

    if passed:
        interpretation = (
            f"Engle-Granger test REJECTED no cointegration (p = {p_value:.4f}): "
            f"Series appear to be COINTEGRATED with coefficient {coint_coef:.4f}. "
            f"Consider using Error Correction Model (ECM) for analysis."
        )
    else:
        interpretation = (
            f"Engle-Granger test FAILED to reject no cointegration (p = {p_value:.4f}): "
            f"No evidence of cointegration found. "
            f"Series may have no long-run equilibrium relationship."
        )

    return DiagnosticResult(
        test_name="Engle-Granger Cointegration Test",
        statistic=adf_stat,
        p_value=p_value,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            "cointegrating_coefficient": coint_coef,
            "cointegrating_regression_coefficients": beta.tolist(),
            "adf_on_residuals": adf_stat,
            "n_observations": n,
            "trend_specification": trend,
            "critical_values": eg_critical_values
        }
    )
