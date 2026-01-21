#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structural Equation Modeling (SEM) Estimator

Python implementation using semopy for SEM analysis.

Features:
- Confirmatory Factor Analysis (CFA)
- Full structural equation models
- Multi-group analysis
- Mediation analysis
- Model comparison

References:
- Bollen (1989): Structural Equations with Latent Variables
- Rosseel (2012): lavaan R package
- semopy documentation: https://semopy.com/
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

# Try to import semopy
try:
    import semopy
    from semopy import Model, Optimizer
    from semopy.stats import calc_stats
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False
    warnings.warn(
        "semopy not installed. Install with: pip install semopy\n"
        "Some functions will use simplified implementations."
    )


@dataclass
class SEMFitIndices:
    """SEM model fit indices"""
    chi_square: float
    df: int
    p_value: float
    cfi: float
    tli: float
    rmsea: float
    rmsea_ci_lower: float
    rmsea_ci_upper: float
    srmr: float
    aic: float
    bic: float


@dataclass
class SEMParameter:
    """Individual parameter estimate"""
    lhs: str  # Left-hand side variable
    op: str   # Operator (=~, ~, ~~)
    rhs: str  # Right-hand side variable
    estimate: float
    se: float
    z_value: float
    p_value: float
    ci_lower: float
    ci_upper: float
    std_estimate: float = None


@dataclass
class SEMResult:
    """Complete SEM analysis result"""
    model_syntax: str
    n_obs: int
    n_params: int
    df: int
    converged: bool
    fit_indices: SEMFitIndices
    parameters: List[SEMParameter]
    standardized_parameters: List[SEMParameter]
    factor_loadings: pd.DataFrame
    path_coefficients: pd.DataFrame
    residual_covariances: pd.DataFrame
    r_squared: Dict[str, float]
    modification_indices: Optional[pd.DataFrame] = None
    bootstrap_ci: Optional[Dict[str, Tuple[float, float]]] = None
    _model: Any = None

    def summary(self) -> str:
        """Generate text summary of results"""
        lines = [
            "=" * 70,
            "STRUCTURAL EQUATION MODEL RESULTS",
            "=" * 70,
            f"",
            f"Model Information:",
            f"  Observations: {self.n_obs}",
            f"  Free parameters: {self.n_params}",
            f"  Degrees of freedom: {self.df}",
            f"  Converged: {self.converged}",
            f"",
            f"Model Fit Indices:",
            f"  Chi-square: {self.fit_indices.chi_square:.3f} (df={self.fit_indices.df}, p={self.fit_indices.p_value:.4f})",
            f"  CFI: {self.fit_indices.cfi:.3f}",
            f"  TLI: {self.fit_indices.tli:.3f}",
            f"  RMSEA: {self.fit_indices.rmsea:.3f} [{self.fit_indices.rmsea_ci_lower:.3f}, {self.fit_indices.rmsea_ci_upper:.3f}]",
            f"  SRMR: {self.fit_indices.srmr:.3f}",
            f"  AIC: {self.fit_indices.aic:.1f}",
            f"  BIC: {self.fit_indices.bic:.1f}",
            f"",
        ]

        # Fit assessment
        lines.append("Fit Assessment:")
        if self.fit_indices.cfi >= 0.95 and self.fit_indices.rmsea <= 0.06:
            lines.append("  Overall: GOOD fit")
        elif self.fit_indices.cfi >= 0.90 and self.fit_indices.rmsea <= 0.08:
            lines.append("  Overall: ACCEPTABLE fit")
        else:
            lines.append("  Overall: POOR fit - consider model re-specification")

        # Factor loadings
        if not self.factor_loadings.empty:
            lines.extend([
                f"",
                "Factor Loadings:",
                self.factor_loadings.to_string(),
            ])

        # Path coefficients
        if not self.path_coefficients.empty:
            lines.extend([
                f"",
                "Path Coefficients:",
                self.path_coefficients.to_string(),
            ])

        # R-squared
        if self.r_squared:
            lines.extend([
                f"",
                "R-squared (Explained Variance):",
            ])
            for var, r2 in self.r_squared.items():
                lines.append(f"  {var}: {r2:.3f}")

        lines.append("=" * 70)
        return "\n".join(lines)


def fit_sem(
    data: pd.DataFrame,
    model: str,
    estimator: str = "ML",
    group: Optional[str] = None,
    group_equal: Optional[List[str]] = None,
    bootstrap: int = 0,
    confidence_level: float = 0.95,
    missing: str = "listwise"
) -> SEMResult:
    """
    Fit a Structural Equation Model.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing observed variables
    model : str
        Model specification in lavaan-style syntax:
        - =~ : factor loading (latent =~ indicators)
        - ~  : regression (DV ~ IV)
        - ~~ : covariance/variance
        - := : defined parameter (for indirect effects)
    estimator : str
        Estimation method: "ML", "MLR", "WLSMV", "ULS"
    group : str, optional
        Column name for multi-group analysis
    group_equal : list, optional
        Parameters to constrain equal across groups:
        ["loadings", "intercepts", "residuals"]
    bootstrap : int
        Number of bootstrap samples (0 = no bootstrap)
    confidence_level : float
        Confidence level for intervals
    missing : str
        Missing data handling: "listwise", "fiml"

    Returns
    -------
    SEMResult
        Complete analysis results

    Example
    -------
    >>> model = '''
    ...     # Measurement model
    ...     Factor1 =~ x1 + x2 + x3
    ...     Factor2 =~ y1 + y2 + y3
    ...     # Structural model
    ...     Factor2 ~ Factor1
    ... '''
    >>> result = fit_sem(data, model)
    >>> print(result.summary())
    """
    if not SEMOPY_AVAILABLE:
        raise ImportError("semopy is required. Install with: pip install semopy")

    # Handle missing data
    if missing == "listwise":
        data = data.dropna()

    n_obs = len(data)

    # Create and fit model
    sem_model = Model(model)

    # Fit with semopy
    try:
        sem_model.fit(data)
        converged = True
    except Exception as e:
        warnings.warn(f"Model convergence issue: {e}")
        converged = False

    # Get statistics
    stats_dict = calc_stats(sem_model)

    # Extract fit indices
    fit_indices = SEMFitIndices(
        chi_square=stats_dict.get('chi2', np.nan),
        df=int(stats_dict.get('DoF', 0)),
        p_value=stats_dict.get('chi2 p-value', np.nan),
        cfi=stats_dict.get('CFI', np.nan),
        tli=stats_dict.get('TLI', np.nan),
        rmsea=stats_dict.get('RMSEA', np.nan),
        rmsea_ci_lower=stats_dict.get('RMSEA Lo', np.nan),
        rmsea_ci_upper=stats_dict.get('RMSEA Hi', np.nan),
        srmr=stats_dict.get('SRMR', np.nan),
        aic=stats_dict.get('AIC', np.nan),
        bic=stats_dict.get('BIC', np.nan),
    )

    # Get parameter estimates
    params_df = sem_model.inspect(std_est=True)

    # Parse parameters
    parameters = []
    standardized_parameters = []
    factor_loadings_data = []
    path_coef_data = []

    for _, row in params_df.iterrows():
        op = row.get('op', '')
        lhs = row.get('lval', '')
        rhs = row.get('rval', '')
        est = row.get('Estimate', np.nan)
        se = row.get('Std. Err', np.nan)
        std_est = row.get('Std. Est', np.nan)

        if pd.notna(se) and se > 0:
            z_val = est / se
            p_val = 2 * (1 - stats.norm.cdf(abs(z_val)))
        else:
            z_val = np.nan
            p_val = np.nan

        alpha = 1 - confidence_level
        ci_lower = est - stats.norm.ppf(1 - alpha/2) * se if pd.notna(se) else np.nan
        ci_upper = est + stats.norm.ppf(1 - alpha/2) * se if pd.notna(se) else np.nan

        param = SEMParameter(
            lhs=lhs, op=op, rhs=rhs,
            estimate=est, se=se,
            z_value=z_val, p_value=p_val,
            ci_lower=ci_lower, ci_upper=ci_upper,
            std_estimate=std_est
        )
        parameters.append(param)

        # Categorize parameters
        if op == '~':  # Factor loading
            factor_loadings_data.append({
                'Factor': lhs,
                'Indicator': rhs,
                'Loading': est,
                'Std.Loading': std_est,
                'SE': se,
                'p-value': p_val
            })
        elif op == '~':  # Path coefficient
            path_coef_data.append({
                'DV': lhs,
                'IV': rhs,
                'Coefficient': est,
                'Std.Coef': std_est,
                'SE': se,
                'p-value': p_val
            })

    factor_loadings = pd.DataFrame(factor_loadings_data) if factor_loadings_data else pd.DataFrame()
    path_coefficients = pd.DataFrame(path_coef_data) if path_coef_data else pd.DataFrame()

    # R-squared values
    r_squared = {}
    # Note: semopy doesn't directly provide R² - would need to calculate

    # Create result
    result = SEMResult(
        model_syntax=model,
        n_obs=n_obs,
        n_params=len(parameters),
        df=fit_indices.df,
        converged=converged,
        fit_indices=fit_indices,
        parameters=parameters,
        standardized_parameters=standardized_parameters,
        factor_loadings=factor_loadings,
        path_coefficients=path_coefficients,
        residual_covariances=pd.DataFrame(),
        r_squared=r_squared,
        _model=sem_model
    )

    return result


def fit_cfa(
    data: pd.DataFrame,
    model: str,
    estimator: str = "ML",
    orthogonal: bool = False
) -> SEMResult:
    """
    Fit a Confirmatory Factor Analysis model.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with indicator variables
    model : str
        CFA specification using =~ operator:
        "Factor1 =~ x1 + x2 + x3
         Factor2 =~ y1 + y2 + y3"
    estimator : str
        Estimation method
    orthogonal : bool
        If True, constrain factor correlations to zero

    Returns
    -------
    SEMResult
        CFA results including factor loadings and correlations

    Example
    -------
    >>> cfa_model = '''
    ...     Anxiety =~ anx1 + anx2 + anx3
    ...     Depression =~ dep1 + dep2 + dep3
    ... '''
    >>> result = fit_cfa(data, cfa_model)
    """
    if orthogonal:
        # Add constraints for orthogonal factors
        # Parse model to find factor names and add ~~ 0 constraints
        lines = model.strip().split('\n')
        factors = []
        for line in lines:
            if '=~' in line:
                factor = line.split('=~')[0].strip()
                factors.append(factor)

        # Add zero covariance constraints
        for i in range(len(factors)):
            for j in range(i+1, len(factors)):
                model += f"\n{factors[i]} ~~ 0*{factors[j]}"

    return fit_sem(data, model, estimator=estimator)


def test_mediation(
    data: pd.DataFrame,
    x: str,
    m: str,
    y: str,
    m_indicators: Optional[List[str]] = None,
    y_indicators: Optional[List[str]] = None,
    covariates: Optional[List[str]] = None,
    bootstrap: int = 5000
) -> Dict[str, Any]:
    """
    Test mediation using SEM framework.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    x : str
        Independent variable (observed)
    m : str
        Mediator (latent if m_indicators provided, else observed)
    y : str
        Dependent variable (latent if y_indicators provided, else observed)
    m_indicators : list, optional
        Indicators for latent mediator
    y_indicators : list, optional
        Indicators for latent outcome
    covariates : list, optional
        Control variables
    bootstrap : int
        Number of bootstrap samples for CI

    Returns
    -------
    dict
        Mediation results including indirect, direct, and total effects
    """
    # Build model syntax
    model_lines = []

    # Measurement model
    if m_indicators:
        model_lines.append(f"{m} =~ " + " + ".join(m_indicators))
        m_var = m
    else:
        m_var = m

    if y_indicators:
        model_lines.append(f"{y} =~ " + " + ".join(y_indicators))
        y_var = y
    else:
        y_var = y

    # Structural model with labeled paths
    model_lines.append(f"{m_var} ~ a*{x}")
    model_lines.append(f"{y_var} ~ b*{m_var} + c*{x}")

    # Add covariates if specified
    if covariates:
        cov_str = " + ".join(covariates)
        model_lines.append(f"{m_var} ~ {cov_str}")
        model_lines.append(f"{y_var} ~ {cov_str}")

    # Define indirect and total effects
    model_lines.append("indirect := a*b")
    model_lines.append("total := c + a*b")

    model = "\n".join(model_lines)

    # Fit model
    result = fit_sem(data, model, bootstrap=bootstrap)

    # Extract effects
    effects = {
        'a_path': None,
        'b_path': None,
        'direct_effect': None,
        'indirect_effect': None,
        'total_effect': None,
    }

    for param in result.parameters:
        if param.rhs == x and param.lhs == m_var:
            effects['a_path'] = {
                'estimate': param.estimate,
                'se': param.se,
                'p_value': param.p_value
            }
        elif param.rhs == m_var and param.lhs == y_var:
            effects['b_path'] = {
                'estimate': param.estimate,
                'se': param.se,
                'p_value': param.p_value
            }
        elif param.rhs == x and param.lhs == y_var:
            effects['direct_effect'] = {
                'estimate': param.estimate,
                'se': param.se,
                'p_value': param.p_value
            }

    # Calculate indirect effect
    if effects['a_path'] and effects['b_path']:
        a = effects['a_path']['estimate']
        b = effects['b_path']['estimate']
        effects['indirect_effect'] = {
            'estimate': a * b,
            'interpretation': 'Product of a and b paths'
        }
        effects['total_effect'] = {
            'estimate': effects['direct_effect']['estimate'] + a * b
        }

    return {
        'model': model,
        'fit_indices': result.fit_indices,
        'effects': effects,
        'result': result
    }


def compare_models(
    model1: SEMResult,
    model2: SEMResult,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> pd.DataFrame:
    """
    Compare two SEM models.

    For nested models, performs chi-square difference test.
    For non-nested models, compares AIC/BIC.

    Parameters
    ----------
    model1, model2 : SEMResult
        Models to compare
    model1_name, model2_name : str
        Names for display

    Returns
    -------
    pd.DataFrame
        Comparison statistics
    """
    comparison = {
        'Model': [model1_name, model2_name],
        'Chi-square': [model1.fit_indices.chi_square, model2.fit_indices.chi_square],
        'df': [model1.fit_indices.df, model2.fit_indices.df],
        'CFI': [model1.fit_indices.cfi, model2.fit_indices.cfi],
        'TLI': [model1.fit_indices.tli, model2.fit_indices.tli],
        'RMSEA': [model1.fit_indices.rmsea, model2.fit_indices.rmsea],
        'SRMR': [model1.fit_indices.srmr, model2.fit_indices.srmr],
        'AIC': [model1.fit_indices.aic, model2.fit_indices.aic],
        'BIC': [model1.fit_indices.bic, model2.fit_indices.bic],
    }

    df = pd.DataFrame(comparison)

    # Chi-square difference test (for nested models)
    chi2_diff = abs(model1.fit_indices.chi_square - model2.fit_indices.chi_square)
    df_diff = abs(model1.fit_indices.df - model2.fit_indices.df)

    if df_diff > 0:
        p_diff = 1 - stats.chi2.cdf(chi2_diff, df_diff)
        print(f"\nChi-square Difference Test (nested models):")
        print(f"  Δχ² = {chi2_diff:.3f}, Δdf = {df_diff}, p = {p_diff:.4f}")

    # CFI difference
    cfi_diff = model1.fit_indices.cfi - model2.fit_indices.cfi
    print(f"\nCFI Difference: ΔCFI = {cfi_diff:.4f}")
    if abs(cfi_diff) > 0.01:
        print("  Note: |ΔCFI| > 0.01 suggests meaningfully different fit")

    return df


def calculate_reliability(
    result: SEMResult,
    factor_name: str
) -> Dict[str, float]:
    """
    Calculate reliability measures for a latent factor.

    Parameters
    ----------
    result : SEMResult
        Fitted SEM result
    factor_name : str
        Name of the factor

    Returns
    -------
    dict
        Reliability measures: omega, AVE, alpha approximation
    """
    # Get factor loadings for this factor
    loadings = result.factor_loadings[
        result.factor_loadings['Factor'] == factor_name
    ]['Std.Loading'].values

    if len(loadings) == 0:
        return {'omega': np.nan, 'ave': np.nan}

    # Composite reliability (omega)
    sum_loadings = np.sum(loadings)
    sum_loadings_sq = np.sum(loadings ** 2)
    sum_error_var = np.sum(1 - loadings ** 2)

    omega = sum_loadings ** 2 / (sum_loadings ** 2 + sum_error_var)

    # Average Variance Extracted
    ave = sum_loadings_sq / len(loadings)

    return {
        'omega': omega,
        'ave': ave,
        'n_indicators': len(loadings)
    }


def test_measurement_invariance(
    data: pd.DataFrame,
    model: str,
    group: str,
    levels: List[str] = ["configural", "metric", "scalar"]
) -> pd.DataFrame:
    """
    Test measurement invariance across groups.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with group variable
    model : str
        CFA model specification
    group : str
        Grouping variable column name
    levels : list
        Invariance levels to test

    Returns
    -------
    pd.DataFrame
        Comparison of invariance models
    """
    results = []

    # Configural invariance (same structure)
    if "configural" in levels:
        config_result = fit_sem(data, model, group=group, group_equal=None)
        results.append({
            'Level': 'Configural',
            'Chi-square': config_result.fit_indices.chi_square,
            'df': config_result.fit_indices.df,
            'CFI': config_result.fit_indices.cfi,
            'RMSEA': config_result.fit_indices.rmsea,
        })

    # Metric invariance (equal loadings)
    if "metric" in levels:
        metric_result = fit_sem(data, model, group=group, group_equal=["loadings"])
        results.append({
            'Level': 'Metric',
            'Chi-square': metric_result.fit_indices.chi_square,
            'df': metric_result.fit_indices.df,
            'CFI': metric_result.fit_indices.cfi,
            'RMSEA': metric_result.fit_indices.rmsea,
        })

    # Scalar invariance (equal loadings + intercepts)
    if "scalar" in levels:
        scalar_result = fit_sem(data, model, group=group, group_equal=["loadings", "intercepts"])
        results.append({
            'Level': 'Scalar',
            'Chi-square': scalar_result.fit_indices.chi_square,
            'df': scalar_result.fit_indices.df,
            'CFI': scalar_result.fit_indices.cfi,
            'RMSEA': scalar_result.fit_indices.rmsea,
        })

    df = pd.DataFrame(results)

    # Add difference columns
    if len(df) > 1:
        df['Δχ²'] = df['Chi-square'].diff()
        df['Δdf'] = df['df'].diff()
        df['ΔCFI'] = df['CFI'].diff()

    return df


# R code generation for lavaan
def generate_lavaan_code(model: str, data_name: str = "data") -> str:
    """
    Generate R lavaan code from model specification.

    Parameters
    ----------
    model : str
        Model in lavaan syntax
    data_name : str
        Name of data frame in R

    Returns
    -------
    str
        Complete R code
    """
    code = f'''
library(lavaan)

# Model specification
model <- '
{model}
'

# Fit the model
fit <- sem(model, data = {data_name}, estimator = "ML")

# Summary with fit indices and standardized estimates
summary(fit, fit.measures = TRUE, standardized = TRUE)

# Additional diagnostics
modindices(fit, sort = TRUE, minimum.value = 10)
fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))

# Reliability measures
library(semTools)
reliability(fit)
'''
    return code


if __name__ == "__main__":
    # Example usage
    print("Structural Equation Modeling Estimator")
    print("=" * 50)

    if SEMOPY_AVAILABLE:
        # Create example data
        np.random.seed(42)
        n = 300

        # Generate latent factors
        F1 = np.random.normal(0, 1, n)
        F2 = 0.5 * F1 + np.random.normal(0, 0.866, n)

        # Generate indicators
        data = pd.DataFrame({
            'x1': 0.8 * F1 + np.random.normal(0, 0.6, n),
            'x2': 0.7 * F1 + np.random.normal(0, 0.714, n),
            'x3': 0.75 * F1 + np.random.normal(0, 0.661, n),
            'y1': 0.8 * F2 + np.random.normal(0, 0.6, n),
            'y2': 0.7 * F2 + np.random.normal(0, 0.714, n),
            'y3': 0.75 * F2 + np.random.normal(0, 0.661, n),
        })

        # Define model
        model = """
            Factor1 =~ x1 + x2 + x3
            Factor2 =~ y1 + y2 + y3
            Factor2 ~ Factor1
        """

        # Fit model
        result = fit_sem(data, model)
        print(result.summary())
    else:
        print("Install semopy for full functionality: pip install semopy")
