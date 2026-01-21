"""
Instrumental Variables (IV) Estimator Implementation.

This module provides comprehensive IV estimation including:
- Two-Stage Least Squares (2SLS)
- Limited Information Maximum Likelihood (LIML)
- Generalized Method of Moments (GMM)
- Weak instrument diagnostics
- Overidentification tests (Sargan-Hansen)
- Endogeneity tests (Hausman/Wu-Hausman)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from scipy import stats

# Import from shared lib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'lib' / 'python'))
from data_loader import CausalInput, CausalOutput
from diagnostics import DiagnosticResult, weak_iv_test, overidentification_test as _overid_test
from table_formatter import create_regression_table, create_diagnostic_report


# =============================================================================
# Stock-Yogo Critical Values
# =============================================================================

# Stock-Yogo (2005) critical values for weak instrument test
# Format: {n_instruments: {max_bias_pct: critical_value}}
STOCK_YOGO_CRITICAL_VALUES = {
    # 10% maximal IV size (most common)
    1: {10: 16.38, 15: 8.96, 20: 6.66, 25: 5.53},
    2: {10: 19.93, 15: 11.59, 20: 8.75, 25: 7.25},
    3: {10: 22.30, 15: 12.83, 20: 9.54, 25: 7.80},
    4: {10: 24.58, 15: 13.96, 20: 10.26, 25: 8.31},
    5: {10: 26.87, 15: 15.09, 20: 10.27, 25: 8.84},
    6: {10: 29.18, 15: 16.23, 20: 11.68, 25: 9.38},
    7: {10: 31.50, 15: 17.38, 20: 12.40, 25: 9.93},
    8: {10: 33.84, 15: 18.54, 20: 13.13, 25: 10.48},
}


# =============================================================================
# Data Validation
# =============================================================================

@dataclass
class IVValidationResult:
    """Result of IV data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"IV Data Validation: {status}"]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def validate_iv_data(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: List[str] = None
) -> IVValidationResult:
    """
    Validate data structure for IV estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable name
    treatment : str
        Endogenous treatment variable name
    instruments : List[str]
        Instrument variable names
    controls : List[str], optional
        Control variable names

    Returns
    -------
    IVValidationResult
        Validation results with errors and warnings
    """
    errors = []
    warnings_list = []
    summary = {}

    # Check required columns exist
    required_cols = [outcome, treatment] + instruments
    if controls:
        required_cols.extend(controls)

    for col in required_cols:
        if col not in data.columns:
            errors.append(f"Required column '{col}' not found in data")

    if errors:
        return IVValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings_list,
            summary=summary
        )

    # Check for missing values
    all_vars = [outcome, treatment] + instruments + (controls or [])
    for col in all_vars:
        n_missing = data[col].isna().sum()
        if n_missing > 0:
            warnings_list.append(f"Column '{col}' has {n_missing} missing values")

    # Sample size
    n_obs = len(data)
    n_instruments = len(instruments)
    summary['n_obs'] = n_obs
    summary['n_instruments'] = n_instruments
    summary['n_controls'] = len(controls) if controls else 0

    # Check identification
    if n_instruments < 1:
        errors.append("At least one instrument required")

    # Check instrument variation
    for z in instruments:
        if data[z].std() < 1e-10:
            errors.append(f"Instrument '{z}' has no variation")
        if data[z].nunique() == 1:
            errors.append(f"Instrument '{z}' is constant")

    # Check treatment variation
    if data[treatment].std() < 1e-10:
        errors.append(f"Treatment '{treatment}' has no variation")

    # Warn about sample size relative to instruments
    if n_obs < 100 * n_instruments:
        warnings_list.append(
            f"Small sample ({n_obs}) relative to number of instruments ({n_instruments}). "
            f"Many instruments bias may be a concern."
        )

    # Summary statistics
    summary['treatment_mean'] = data[treatment].mean()
    summary['treatment_std'] = data[treatment].std()
    summary['outcome_mean'] = data[outcome].mean()
    summary['outcome_std'] = data[outcome].std()

    is_valid = len(errors) == 0

    return IVValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings_list,
        summary=summary
    )


# =============================================================================
# First Stage Diagnostics
# =============================================================================

def first_stage_test(
    data: pd.DataFrame,
    treatment: str,
    instruments: List[str],
    controls: List[str] = None
) -> Dict[str, Any]:
    """
    Run first-stage regression and compute diagnostics.

    Tests instrument relevance by regressing the endogenous treatment
    on instruments and controls.

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

    Returns
    -------
    Dict[str, Any]
        First-stage results including:
        - f_statistic: F-test for joint significance of instruments
        - partial_r2: Partial R-squared from instruments
        - coefficients: Instrument coefficients
        - std_errors: Standard errors
        - p_values: P-values for each instrument
    """
    try:
        import statsmodels.api as sm
        from statsmodels.regression.linear_model import OLS
    except ImportError:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")

    df = data.copy()

    # Build design matrices
    X_vars = instruments.copy()
    if controls:
        X_vars.extend(controls)

    X = df[X_vars].copy()
    X = sm.add_constant(X)
    y = df[treatment]

    # Handle missing values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Fit full model (with instruments)
    model_full = OLS(y, X).fit(cov_type='HC1')

    # Fit restricted model (without instruments) for partial F-test
    if controls:
        X_restricted = sm.add_constant(df[controls])[mask]
        model_restricted = OLS(y, X_restricted).fit()
        rss_restricted = model_restricted.ssr
    else:
        # Just intercept
        X_restricted = sm.add_constant(pd.DataFrame(index=y.index))
        model_restricted = OLS(y, X_restricted).fit()
        rss_restricted = model_restricted.ssr

    # Calculate partial F-statistic
    rss_full = model_full.ssr
    n = len(y)
    k_instruments = len(instruments)
    k_full = len(X_vars) + 1  # +1 for constant

    f_statistic = ((rss_restricted - rss_full) / k_instruments) / (rss_full / (n - k_full))

    # Calculate partial R-squared
    partial_r2 = (rss_restricted - rss_full) / rss_restricted

    # Extract instrument coefficients
    coefficients = {}
    std_errors = {}
    p_values = {}

    for z in instruments:
        coefficients[z] = model_full.params[z]
        std_errors[z] = model_full.bse[z]
        p_values[z] = model_full.pvalues[z]

    # Joint significance p-value for instruments
    f_pvalue = 1 - stats.f.cdf(f_statistic, k_instruments, n - k_full)

    return {
        'f_statistic': f_statistic,
        'f_pvalue': f_pvalue,
        'partial_r2': partial_r2,
        'r_squared': model_full.rsquared,
        'coefficients': coefficients,
        'std_errors': std_errors,
        'p_values': p_values,
        'n_obs': n,
        'n_instruments': k_instruments,
        'residuals': model_full.resid.values
    }


def weak_iv_diagnostics(
    first_stage_f: float,
    n_instruments: int = 1,
    max_bias: int = 10
) -> DiagnosticResult:
    """
    Assess weak instrument problem using Stock-Yogo critical values.

    Parameters
    ----------
    first_stage_f : float
        F-statistic from first-stage regression
    n_instruments : int
        Number of instruments
    max_bias : int
        Maximum acceptable bias as percentage of OLS bias (10, 15, 20, or 25)

    Returns
    -------
    DiagnosticResult
        Weak instrument diagnostic result
    """
    # Get critical value
    if n_instruments in STOCK_YOGO_CRITICAL_VALUES:
        cv_dict = STOCK_YOGO_CRITICAL_VALUES[n_instruments]
        if max_bias in cv_dict:
            critical_value = cv_dict[max_bias]
        else:
            critical_value = 10.0  # Rule of thumb
    else:
        critical_value = 10.0  # Rule of thumb for many instruments

    passed = first_stage_f > critical_value

    if passed:
        interpretation = (
            f"Instruments are STRONG: First-stage F = {first_stage_f:.2f} > "
            f"{critical_value:.2f} (Stock-Yogo {max_bias}% maximal bias critical value)"
        )
    elif first_stage_f > 10:
        interpretation = (
            f"Instruments are moderately strong: First-stage F = {first_stage_f:.2f}. "
            f"Above rule-of-thumb (10) but below Stock-Yogo {max_bias}% critical value ({critical_value:.2f}). "
            f"Consider using LIML."
        )
    else:
        interpretation = (
            f"WEAK INSTRUMENTS WARNING: First-stage F = {first_stage_f:.2f} < 10. "
            f"2SLS estimates may be severely biased toward OLS. "
            f"Use LIML or find stronger instruments."
        )

    return DiagnosticResult(
        test_name="Weak Instrument Test (Stock-Yogo)",
        statistic=first_stage_f,
        p_value=np.nan,  # Not a p-value based test
        passed=passed,
        threshold=critical_value,
        interpretation=interpretation,
        details={
            'n_instruments': n_instruments,
            'max_bias_pct': max_bias,
            'critical_value': critical_value,
            'rule_of_thumb_passed': first_stage_f > 10
        }
    )


# =============================================================================
# 2SLS Estimation
# =============================================================================

def estimate_2sls(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: List[str] = None
) -> CausalOutput:
    """
    Estimate IV model using Two-Stage Least Squares (2SLS).

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable name (Y)
    treatment : str
        Endogenous treatment variable name (D)
    instruments : List[str]
        Instrument variable names (Z)
    controls : List[str], optional
        Exogenous control variable names (X)

    Returns
    -------
    CausalOutput
        2SLS estimate with standard errors and diagnostics
    """
    try:
        from linearmodels.iv import IV2SLS
    except ImportError:
        raise ImportError(
            "linearmodels required for IV estimation. "
            "Install with: pip install linearmodels"
        )

    df = data.copy()

    # Prepare variables
    y = df[outcome]
    endog = df[[treatment]]

    # Exogenous variables (controls + constant)
    if controls:
        exog = df[controls].copy()
        exog['const'] = 1
    else:
        exog = pd.DataFrame({'const': np.ones(len(df))})

    # Instruments
    instr = df[instruments]

    # Handle missing values
    all_data = pd.concat([y, endog, exog, instr], axis=1)
    mask = ~all_data.isna().any(axis=1)

    y = y[mask]
    endog = endog[mask]
    exog = exog[mask]
    instr = instr[mask]

    # Fit 2SLS model
    model = IV2SLS(y, exog, endog, instr)
    results = model.fit(cov_type='robust')

    # Extract treatment coefficient
    treat_coef = results.params[treatment]
    treat_se = results.std_errors[treatment]
    treat_pval = results.pvalues[treatment]
    ci_lower = treat_coef - 1.96 * treat_se
    ci_upper = treat_coef + 1.96 * treat_se

    # First-stage diagnostics
    first_stage = first_stage_test(df[mask], treatment, instruments, controls)

    # Diagnostics
    diagnostics = {
        'method': '2SLS',
        'n_obs': len(y),
        'n_instruments': len(instruments),
        'n_controls': len(controls) if controls else 0,
        'first_stage': first_stage,
        'r_squared': results.rsquared if hasattr(results, 'rsquared') else np.nan,
        'residuals': results.resids.values
    }

    # Summary table
    table_results = [{
        'treatment_effect': treat_coef,
        'treatment_se': treat_se,
        'treatment_pval': treat_pval,
        'controls': controls is not None and len(controls) > 0,
        'fixed_effects': None,
        'n_obs': len(y),
        'r_squared': diagnostics['r_squared']
    }]

    summary_table = create_regression_table(
        results=table_results,
        column_names=["(1) 2SLS"],
        title="Two-Stage Least Squares Results",
        notes=f"First-stage F = {first_stage['f_statistic']:.2f}. Robust standard errors in parentheses."
    )

    return CausalOutput(
        effect=treat_coef,
        se=treat_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=treat_pval,
        diagnostics=diagnostics,
        summary_table=summary_table,
        interpretation=(
            f"The 2SLS estimate is {treat_coef:.4f} (SE = {treat_se:.4f}). "
            f"First-stage F = {first_stage['f_statistic']:.2f}."
        )
    )


# =============================================================================
# LIML Estimation
# =============================================================================

def estimate_liml(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: List[str] = None
) -> CausalOutput:
    """
    Estimate IV model using Limited Information Maximum Likelihood (LIML).

    LIML is more robust to weak instruments than 2SLS.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable name (Y)
    treatment : str
        Endogenous treatment variable name (D)
    instruments : List[str]
        Instrument variable names (Z)
    controls : List[str], optional
        Exogenous control variable names (X)

    Returns
    -------
    CausalOutput
        LIML estimate with standard errors and diagnostics
    """
    try:
        from linearmodels.iv import IVLIML
    except ImportError:
        raise ImportError(
            "linearmodels required for IV estimation. "
            "Install with: pip install linearmodels"
        )

    df = data.copy()

    # Prepare variables
    y = df[outcome]
    endog = df[[treatment]]

    # Exogenous variables (controls + constant)
    if controls:
        exog = df[controls].copy()
        exog['const'] = 1
    else:
        exog = pd.DataFrame({'const': np.ones(len(df))})

    # Instruments
    instr = df[instruments]

    # Handle missing values
    all_data = pd.concat([y, endog, exog, instr], axis=1)
    mask = ~all_data.isna().any(axis=1)

    y = y[mask]
    endog = endog[mask]
    exog = exog[mask]
    instr = instr[mask]

    # Fit LIML model
    model = IVLIML(y, exog, endog, instr)
    results = model.fit(cov_type='robust')

    # Extract treatment coefficient
    treat_coef = results.params[treatment]
    treat_se = results.std_errors[treatment]
    treat_pval = results.pvalues[treatment]
    ci_lower = treat_coef - 1.96 * treat_se
    ci_upper = treat_coef + 1.96 * treat_se

    # First-stage diagnostics
    first_stage = first_stage_test(df[mask], treatment, instruments, controls)

    # LIML-specific diagnostics
    kappa = results.kappa if hasattr(results, 'kappa') else np.nan

    # Diagnostics
    diagnostics = {
        'method': 'LIML',
        'n_obs': len(y),
        'n_instruments': len(instruments),
        'n_controls': len(controls) if controls else 0,
        'first_stage': first_stage,
        'kappa': kappa,
        'r_squared': results.rsquared if hasattr(results, 'rsquared') else np.nan,
        'residuals': results.resids.values
    }

    # Summary table
    table_results = [{
        'treatment_effect': treat_coef,
        'treatment_se': treat_se,
        'treatment_pval': treat_pval,
        'controls': controls is not None and len(controls) > 0,
        'fixed_effects': None,
        'n_obs': len(y),
        'r_squared': diagnostics['r_squared']
    }]

    summary_table = create_regression_table(
        results=table_results,
        column_names=["(1) LIML"],
        title="Limited Information Maximum Likelihood Results",
        notes=f"First-stage F = {first_stage['f_statistic']:.2f}. Robust standard errors in parentheses."
    )

    return CausalOutput(
        effect=treat_coef,
        se=treat_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=treat_pval,
        diagnostics=diagnostics,
        summary_table=summary_table,
        interpretation=(
            f"The LIML estimate is {treat_coef:.4f} (SE = {treat_se:.4f}). "
            f"First-stage F = {first_stage['f_statistic']:.2f}. "
            f"LIML is more robust to weak instruments than 2SLS."
        )
    )


# =============================================================================
# GMM Estimation
# =============================================================================

def estimate_gmm(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: List[str] = None
) -> CausalOutput:
    """
    Estimate IV model using Generalized Method of Moments (GMM).

    GMM is efficient under heteroskedasticity when overidentified.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable name (Y)
    treatment : str
        Endogenous treatment variable name (D)
    instruments : List[str]
        Instrument variable names (Z)
    controls : List[str], optional
        Exogenous control variable names (X)

    Returns
    -------
    CausalOutput
        GMM estimate with standard errors and diagnostics
    """
    try:
        from linearmodels.iv import IVGMM
    except ImportError:
        raise ImportError(
            "linearmodels required for IV estimation. "
            "Install with: pip install linearmodels"
        )

    df = data.copy()

    # Prepare variables
    y = df[outcome]
    endog = df[[treatment]]

    # Exogenous variables (controls + constant)
    if controls:
        exog = df[controls].copy()
        exog['const'] = 1
    else:
        exog = pd.DataFrame({'const': np.ones(len(df))})

    # Instruments
    instr = df[instruments]

    # Handle missing values
    all_data = pd.concat([y, endog, exog, instr], axis=1)
    mask = ~all_data.isna().any(axis=1)

    y = y[mask]
    endog = endog[mask]
    exog = exog[mask]
    instr = instr[mask]

    # Fit GMM model
    model = IVGMM(y, exog, endog, instr)
    results = model.fit(cov_type='robust')

    # Extract treatment coefficient
    treat_coef = results.params[treatment]
    treat_se = results.std_errors[treatment]
    treat_pval = results.pvalues[treatment]
    ci_lower = treat_coef - 1.96 * treat_se
    ci_upper = treat_coef + 1.96 * treat_se

    # First-stage diagnostics
    first_stage = first_stage_test(df[mask], treatment, instruments, controls)

    # Diagnostics
    diagnostics = {
        'method': 'GMM',
        'n_obs': len(y),
        'n_instruments': len(instruments),
        'n_controls': len(controls) if controls else 0,
        'first_stage': first_stage,
        'r_squared': results.rsquared if hasattr(results, 'rsquared') else np.nan,
        'residuals': results.resids.values,
        'j_statistic': results.j_stat.stat if hasattr(results, 'j_stat') else np.nan,
        'j_pvalue': results.j_stat.pval if hasattr(results, 'j_stat') else np.nan
    }

    # Summary table
    table_results = [{
        'treatment_effect': treat_coef,
        'treatment_se': treat_se,
        'treatment_pval': treat_pval,
        'controls': controls is not None and len(controls) > 0,
        'fixed_effects': None,
        'n_obs': len(y),
        'r_squared': diagnostics['r_squared']
    }]

    summary_table = create_regression_table(
        results=table_results,
        column_names=["(1) GMM"],
        title="Generalized Method of Moments Results",
        notes=f"First-stage F = {first_stage['f_statistic']:.2f}. Robust standard errors in parentheses."
    )

    return CausalOutput(
        effect=treat_coef,
        se=treat_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=treat_pval,
        diagnostics=diagnostics,
        summary_table=summary_table,
        interpretation=(
            f"The GMM estimate is {treat_coef:.4f} (SE = {treat_se:.4f}). "
            f"First-stage F = {first_stage['f_statistic']:.2f}. "
            f"GMM is efficient under heteroskedasticity."
        )
    )


# =============================================================================
# Overidentification Test
# =============================================================================

def compute_sargan_test(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: List[str] = None,
    iv_residuals: np.ndarray = None
) -> Tuple[float, float]:
    """
    Compute Sargan-Hansen overidentification test statistic.

    The test regresses IV residuals on all exogenous variables (instruments + controls)
    and uses n*R^2 as the test statistic, which follows chi-squared with
    (K - 1) degrees of freedom where K is the number of instruments.

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
    iv_residuals : np.ndarray, optional
        Pre-computed IV residuals

    Returns
    -------
    Tuple[float, float]
        (J-statistic, p-value)
    """
    try:
        import statsmodels.api as sm
        from statsmodels.regression.linear_model import OLS
    except ImportError:
        return np.nan, np.nan

    df = data.copy()
    n = len(df)

    # If residuals not provided, we need to compute them
    if iv_residuals is None:
        return np.nan, np.nan

    # Build matrix of all exogenous variables (instruments + controls)
    Z_vars = instruments.copy()
    if controls:
        Z_vars.extend(controls)

    Z = df[Z_vars].copy()
    Z = sm.add_constant(Z)

    # Handle length mismatch (due to missing value handling)
    if len(iv_residuals) != len(Z):
        # Try to match lengths
        min_len = min(len(iv_residuals), len(Z))
        iv_residuals = iv_residuals[:min_len]
        Z = Z.iloc[:min_len]

    # Regress IV residuals on all exogenous variables
    model = OLS(iv_residuals, Z).fit()

    # Sargan statistic: n * R^2
    sargan_stat = n * model.rsquared

    # Degrees of freedom: number of overidentifying restrictions
    df_test = len(instruments) - 1  # K - 1 for single endogenous variable

    # P-value from chi-squared distribution
    p_value = 1 - stats.chi2.cdf(sargan_stat, df=df_test)

    return sargan_stat, p_value


def overidentification_test(
    model_result: CausalOutput,
    data: pd.DataFrame = None,
    outcome: str = None,
    treatment: str = None,
    instruments: List[str] = None,
    controls: List[str] = None
) -> DiagnosticResult:
    """
    Sargan-Hansen overidentification test for IV validity.

    Tests whether the overidentifying restrictions are valid (i.e., whether
    all instruments are exogenous). Only valid when model is overidentified
    (more instruments than endogenous variables).

    Parameters
    ----------
    model_result : CausalOutput
        Result from estimate_2sls, estimate_liml, or estimate_gmm
    data : pd.DataFrame, optional
        Original data (needed for proper Sargan calculation)
    outcome : str, optional
        Outcome variable name
    treatment : str, optional
        Treatment variable name
    instruments : List[str], optional
        Instrument names
    controls : List[str], optional
        Control variable names

    Returns
    -------
    DiagnosticResult
        Overidentification test result
    """
    diagnostics = model_result.diagnostics
    n_instruments = diagnostics['n_instruments']
    n_endog = 1  # Single endogenous variable

    if n_instruments <= n_endog:
        return DiagnosticResult(
            test_name="Sargan-Hansen Overidentification Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=True,
            threshold=0.05,
            interpretation=(
                "Model is exactly identified (number of instruments = number of endogenous variables). "
                "Overidentification test not applicable."
            ),
            details={'reason': 'exact_identification'}
        )

    # Get J-statistic from GMM if available (linearmodels computes it)
    if 'j_statistic' in diagnostics and not np.isnan(diagnostics.get('j_statistic', np.nan)):
        j_stat = diagnostics['j_statistic']
        j_pval = diagnostics['j_pvalue']
    elif data is not None and instruments is not None:
        # Compute Sargan test properly
        residuals = diagnostics.get('residuals')
        if residuals is not None:
            j_stat, j_pval = compute_sargan_test(
                data, outcome, treatment, instruments, controls, residuals
            )
        else:
            j_stat, j_pval = np.nan, np.nan
    else:
        # Cannot compute without data
        j_stat, j_pval = np.nan, np.nan

    passed = j_pval > 0.05 if not np.isnan(j_pval) else True

    if np.isnan(j_pval):
        interpretation = "Unable to compute overidentification test statistic"
    elif passed:
        interpretation = (
            f"Overidentification test PASSED: J = {j_stat:.4f}, p = {j_pval:.4f}. "
            f"Cannot reject that all instruments are valid."
        )
    else:
        interpretation = (
            f"Overidentification test FAILED: J = {j_stat:.4f}, p = {j_pval:.4f}. "
            f"Evidence that at least one instrument may be invalid."
        )

    return DiagnosticResult(
        test_name="Sargan-Hansen Overidentification Test",
        statistic=j_stat,
        p_value=j_pval,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            'n_instruments': n_instruments,
            'n_endogenous': n_endog,
            'degrees_of_freedom': n_instruments - n_endog
        }
    )


# =============================================================================
# Endogeneity Test
# =============================================================================

def endogeneity_test(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: List[str] = None
) -> DiagnosticResult:
    """
    Hausman/Wu-Hausman test for endogeneity.

    Tests whether the treatment variable is endogenous by comparing
    OLS and IV estimates. If they differ significantly, endogeneity
    is present and IV is needed.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable name
    treatment : str
        Treatment variable name
    instruments : List[str]
        Instrument variable names
    controls : List[str], optional
        Control variable names

    Returns
    -------
    DiagnosticResult
        Endogeneity test result
    """
    try:
        import statsmodels.api as sm
        from statsmodels.regression.linear_model import OLS
    except ImportError:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")

    df = data.copy()

    # First get first-stage residuals
    first_stage = first_stage_test(df, treatment, instruments, controls)
    v_hat = first_stage['residuals']

    # Augmented regression: Y on D, X, and first-stage residuals
    X_vars = [treatment]
    if controls:
        X_vars.extend(controls)

    X = df[X_vars].copy()
    X['_v_hat'] = v_hat
    X = sm.add_constant(X)
    y = df[outcome]

    # Handle missing values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Fit augmented model
    model = OLS(y, X).fit(cov_type='HC1')

    # The coefficient on v_hat tests endogeneity
    # If significant, treatment is endogenous
    v_coef = model.params['_v_hat']
    v_se = model.bse['_v_hat']
    v_tstat = model.tvalues['_v_hat']
    v_pval = model.pvalues['_v_hat']

    passed = v_pval < 0.05  # Note: "passed" means endogeneity detected (IV needed)

    if passed:
        interpretation = (
            f"Endogeneity test SIGNIFICANT: t = {v_tstat:.4f}, p = {v_pval:.4f}. "
            f"Treatment is endogenous. IV estimation is appropriate."
        )
    else:
        interpretation = (
            f"Endogeneity test NOT significant: t = {v_tstat:.4f}, p = {v_pval:.4f}. "
            f"Cannot reject exogeneity of treatment. OLS may be consistent, "
            f"though IV is still valid."
        )

    return DiagnosticResult(
        test_name="Wu-Hausman Endogeneity Test",
        statistic=v_tstat,
        p_value=v_pval,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            'v_hat_coefficient': v_coef,
            'v_hat_se': v_se,
            'n_obs': len(y)
        }
    )


# =============================================================================
# Full IV Analysis Workflow
# =============================================================================

def run_full_iv_analysis(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    instruments: List[str],
    controls: List[str] = None,
    weak_iv_threshold: float = 10.0
) -> CausalOutput:
    """
    Run complete IV analysis workflow.

    This function:
    1. Validates data structure
    2. Tests first-stage strength (weak IV diagnostics)
    3. Runs 2SLS (and LIML if instruments may be weak)
    4. Tests overidentification (if overidentified)
    5. Tests endogeneity
    6. Generates comprehensive output

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable name (Y)
    treatment : str
        Endogenous treatment variable name (D)
    instruments : List[str]
        Instrument variable names (Z)
    controls : List[str], optional
        Control variable names (X)
    weak_iv_threshold : float
        F-statistic threshold for weak instruments (default: 10)

    Returns
    -------
    CausalOutput
        Complete analysis results with all diagnostics
    """
    # Step 1: Validate data
    validation = validate_iv_data(data, outcome, treatment, instruments, controls)

    if not validation.is_valid:
        raise ValueError(f"Data validation failed: {validation.errors}")

    # Step 2: First-stage test
    first_stage = first_stage_test(data, treatment, instruments, controls)
    weak_iv = weak_iv_diagnostics(first_stage['f_statistic'], len(instruments))

    # Step 3: Estimation
    # Always run 2SLS
    result_2sls = estimate_2sls(data, outcome, treatment, instruments, controls)

    # Run LIML if instruments may be weak
    if first_stage['f_statistic'] < weak_iv_threshold:
        warnings.warn(
            f"First-stage F = {first_stage['f_statistic']:.2f} < {weak_iv_threshold}. "
            f"Using LIML as primary estimate due to weak instruments."
        )
        result_liml = estimate_liml(data, outcome, treatment, instruments, controls)
        main_result = result_liml
        method_used = 'LIML'
    else:
        result_liml = None
        main_result = result_2sls
        method_used = '2SLS'

    # Step 4: Overidentification test (if overidentified)
    if len(instruments) > 1:
        overid_test = overidentification_test(
            main_result,
            data=data,
            outcome=outcome,
            treatment=treatment,
            instruments=instruments,
            controls=controls
        )
    else:
        overid_test = DiagnosticResult(
            test_name="Sargan-Hansen Overidentification Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=True,
            threshold=0.05,
            interpretation="Model is exactly identified. Test not applicable.",
            details={'reason': 'exact_identification'}
        )

    # Step 5: Endogeneity test
    endog_test = endogeneity_test(data, outcome, treatment, instruments, controls)

    # Compile all diagnostics
    all_diagnostics = {
        'method': method_used,
        'validation': validation.summary,
        'first_stage': first_stage,
        'weak_iv_test': weak_iv,
        'overidentification_test': overid_test,
        'endogeneity_test': endog_test,
        '2sls_estimate': result_2sls.effect,
        '2sls_se': result_2sls.se
    }

    if result_liml:
        all_diagnostics['liml_estimate'] = result_liml.effect
        all_diagnostics['liml_se'] = result_liml.se

    # Generate comprehensive summary
    summary_lines = [
        "=" * 60,
        "INSTRUMENTAL VARIABLES ANALYSIS RESULTS",
        "=" * 60,
        "",
        f"Primary Method: {method_used}",
        f"Treatment Effect (LATE): {main_result.effect:.4f}",
        f"Standard Error: {main_result.se:.4f}",
        f"95% CI: [{main_result.ci_lower:.4f}, {main_result.ci_upper:.4f}]",
        f"P-value: {main_result.p_value:.4f}",
        "",
        "-" * 60,
        "FIRST STAGE",
        "-" * 60,
        f"F-statistic: {first_stage['f_statistic']:.2f}",
        f"Partial R-squared: {first_stage['partial_r2']:.4f}",
        f"Weak IV Test: {'PASSED' if weak_iv.passed else 'FAILED'}",
        ""
    ]

    # Instrument coefficients
    summary_lines.append("Instrument Coefficients:")
    for z in instruments:
        coef = first_stage['coefficients'][z]
        se = first_stage['std_errors'][z]
        pval = first_stage['p_values'][z]
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        summary_lines.append(f"  {z}: {coef:.4f}{stars} ({se:.4f})")

    summary_lines.extend([
        "",
        "-" * 60,
        "DIAGNOSTICS",
        "-" * 60,
        "",
        f"Endogeneity Test (Wu-Hausman): {'SIGNIFICANT' if endog_test.passed else 'Not significant'}",
        f"  - Statistic: {endog_test.statistic:.4f}",
        f"  - P-value: {endog_test.p_value:.4f}",
        ""
    ])

    if len(instruments) > 1:
        summary_lines.extend([
            f"Overidentification Test (Sargan-Hansen): {'PASSED' if overid_test.passed else 'FAILED'}",
            f"  - J-statistic: {overid_test.statistic:.4f}" if not np.isnan(overid_test.statistic) else "  - J-statistic: N/A",
            f"  - P-value: {overid_test.p_value:.4f}" if not np.isnan(overid_test.p_value) else "  - P-value: N/A",
            ""
        ])

    # Compare estimators if LIML was run
    if result_liml:
        summary_lines.extend([
            "-" * 60,
            "ESTIMATOR COMPARISON",
            "-" * 60,
            f"2SLS:  {result_2sls.effect:.4f} (SE: {result_2sls.se:.4f})",
            f"LIML:  {result_liml.effect:.4f} (SE: {result_liml.se:.4f})",
            f"Difference: {abs(result_2sls.effect - result_liml.effect):.4f}",
            ""
        ])

    summary_lines.extend([
        "-" * 60,
        "SAMPLE",
        "-" * 60,
        f"N Observations: {validation.summary.get('n_obs', 'N/A'):,}",
        f"N Instruments: {validation.summary.get('n_instruments', 'N/A')}",
        f"N Controls: {validation.summary.get('n_controls', 'N/A')}",
        "",
        "=" * 60
    ])

    comprehensive_summary = "\n".join(summary_lines)

    # Generate interpretation
    interpretation = main_result.generate_interpretation(
        treatment_name=treatment,
        outcome_name=outcome
    )

    # Add caveats based on diagnostics
    if not weak_iv.passed:
        interpretation += (
            f"\n\nCAUTION: Weak instruments detected (F = {first_stage['f_statistic']:.2f}). "
            f"Using LIML for robustness. Confidence intervals may still be unreliable."
        )

    if len(instruments) > 1 and not overid_test.passed:
        interpretation += (
            "\n\nWARNING: Overidentification test failed. "
            "At least one instrument may be invalid."
        )

    if not endog_test.passed:
        interpretation += (
            "\n\nNOTE: Endogeneity test not significant. OLS may be consistent, "
            "though IV remains valid if the instrument is truly exogenous."
        )

    interpretation += (
        f"\n\nThis is a Local Average Treatment Effect (LATE) for compliers - "
        f"units whose {treatment} was affected by the instrument(s)."
    )

    return CausalOutput(
        effect=main_result.effect,
        se=main_result.se,
        ci_lower=main_result.ci_lower,
        ci_upper=main_result.ci_upper,
        p_value=main_result.p_value,
        diagnostics=all_diagnostics,
        summary_table=comprehensive_summary,
        interpretation=interpretation
    )


# =============================================================================
# OLS Comparison (for diagnostics)
# =============================================================================

def estimate_ols(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str] = None
) -> CausalOutput:
    """
    Estimate OLS for comparison with IV.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable name
    treatment : str
        Treatment variable name
    controls : List[str], optional
        Control variable names

    Returns
    -------
    CausalOutput
        OLS estimate (potentially biased due to endogeneity)
    """
    try:
        import statsmodels.api as sm
        from statsmodels.regression.linear_model import OLS
    except ImportError:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")

    df = data.copy()

    # Build design matrix
    X_vars = [treatment]
    if controls:
        X_vars.extend(controls)

    X = df[X_vars].copy()
    X = sm.add_constant(X)
    y = df[outcome]

    # Handle missing values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Fit OLS
    model = OLS(y, X).fit(cov_type='HC1')

    treat_coef = model.params[treatment]
    treat_se = model.bse[treatment]
    treat_pval = model.pvalues[treatment]
    ci = model.conf_int().loc[treatment].values

    diagnostics = {
        'method': 'OLS',
        'n_obs': len(y),
        'r_squared': model.rsquared,
        'residuals': model.resid.values
    }

    return CausalOutput(
        effect=treat_coef,
        se=treat_se,
        ci_lower=ci[0],
        ci_upper=ci[1],
        p_value=treat_pval,
        diagnostics=diagnostics,
        summary_table=f"OLS: {treat_coef:.4f} (SE: {treat_se:.4f})",
        interpretation=f"OLS estimate (potentially biased): {treat_coef:.4f}"
    )


# =============================================================================
# Validation with Synthetic Data
# =============================================================================

def validate_estimator(verbose: bool = True) -> Dict[str, Any]:
    """
    Validate IV estimator on synthetic data with known treatment effect.

    Returns
    -------
    Dict[str, Any]
        Validation results including bias assessment
    """
    from data_loader import generate_synthetic_iv_data

    # Generate synthetic data with known parameters
    true_effect = 1.0
    data, true_params = generate_synthetic_iv_data(
        n=2000,
        treatment_effect=true_effect,
        first_stage_strength=0.5,
        noise_std=1.0,
        random_state=42
    )

    # Run IV analysis
    result = run_full_iv_analysis(
        data=data,
        outcome='y',
        treatment='d',
        instruments=['z'],
        controls=['x1', 'x2']
    )

    # Also run OLS for comparison
    ols_result = estimate_ols(
        data=data,
        outcome='y',
        treatment='d',
        controls=['x1', 'x2']
    )

    # Calculate bias
    iv_bias = result.effect - true_effect
    iv_bias_pct = abs(iv_bias / true_effect) * 100
    ols_bias = ols_result.effect - true_effect
    ols_bias_pct = abs(ols_bias / true_effect) * 100

    # Check if within acceptable range (10% bias for IV is reasonable)
    passed = iv_bias_pct < 10.0

    validation_result = {
        'true_effect': true_effect,
        'iv_estimate': result.effect,
        'iv_se': result.se,
        'iv_bias': iv_bias,
        'iv_bias_pct': iv_bias_pct,
        'ols_estimate': ols_result.effect,
        'ols_bias': ols_bias,
        'ols_bias_pct': ols_bias_pct,
        'first_stage_f': result.diagnostics['first_stage']['f_statistic'],
        'passed': passed,
        'ci_covers_truth': result.ci_lower <= true_effect <= result.ci_upper,
        'iv_corrects_ols_bias': abs(iv_bias) < abs(ols_bias)
    }

    if verbose:
        print("=" * 60)
        print("IV ESTIMATOR VALIDATION")
        print("=" * 60)
        print(f"True Effect: {true_effect:.4f}")
        print(f"OLS Estimate: {ols_result.effect:.4f} (Bias: {ols_bias:.4f}, {ols_bias_pct:.2f}%)")
        print(f"IV Estimate: {result.effect:.4f} (Bias: {iv_bias:.4f}, {iv_bias_pct:.2f}%)")
        print(f"IV Standard Error: {result.se:.4f}")
        print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"CI covers truth: {validation_result['ci_covers_truth']}")
        print(f"First-stage F: {validation_result['first_stage_f']:.2f}")
        print(f"IV corrects OLS bias: {validation_result['iv_corrects_ols_bias']}")
        print("-" * 60)
        print(f"VALIDATION: {'PASSED' if passed else 'FAILED'} (IV bias < 10%)")
        print("=" * 60)

    return validation_result


if __name__ == "__main__":
    # Run validation when module is executed directly
    validate_estimator(verbose=True)
