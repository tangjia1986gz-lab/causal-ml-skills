"""
ML-Enhanced Causal Mediation Analysis Implementation.

This module provides comprehensive mediation analysis including:
- Traditional Baron-Kenny (1986) approach
- ML-enhanced mediation using DDML principles
- Sensitivity analysis (Imai et al., 2010)
- Effect decomposition (ADE, ACME, proportion mediated)
- Bootstrap confidence intervals
- Visualization of mediation pathways

References:
- Baron & Kenny (1986). The Moderator-Mediator Variable Distinction.
- Imai, Keele, & Tingley (2010). A General Approach to Causal Mediation Analysis.
- Farbmacher et al. (2022). Causal Mediation Analysis with Double Machine Learning.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Import from shared lib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'lib' / 'python'))
from data_loader import CausalInput, CausalOutput
from diagnostics import DiagnosticResult
from table_formatter import create_regression_table, create_diagnostic_report


# =============================================================================
# Data Preparation and Validation
# =============================================================================

@dataclass
class MediationData:
    """Prepared data structure for mediation analysis."""
    y: np.ndarray            # Outcome variable
    d: np.ndarray            # Treatment variable
    m: np.ndarray            # Mediator variable
    X: np.ndarray            # Control variables
    feature_names: List[str]  # Names of control variables
    n: int                    # Number of observations
    p: int                    # Number of controls
    treatment_type: str       # 'binary' or 'continuous'
    mediator_type: str        # 'binary' or 'continuous'

    def __repr__(self):
        return (
            f"MediationData(n={self.n}, p={self.p}, "
            f"treatment={self.treatment_type}, mediator={self.mediator_type})"
        )


def create_mediation_data(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: List[str],
    drop_na: bool = True
) -> MediationData:
    """
    Prepare data for mediation analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    outcome : str
        Name of outcome variable (Y)
    treatment : str
        Name of treatment variable (D)
    mediator : str
        Name of mediator variable (M)
    controls : List[str]
        Names of control variables (X)
    drop_na : bool
        Whether to drop rows with missing values

    Returns
    -------
    MediationData
        Prepared data structure
    """
    df = data.copy()

    # Select columns
    cols = [outcome, treatment, mediator] + controls
    df = df[cols]

    # Handle missing values
    if drop_na:
        n_before = len(df)
        df = df.dropna()
        n_after = len(df)
        if n_before != n_after:
            warnings.warn(f"Dropped {n_before - n_after} rows with missing values")

    # Extract arrays
    y = df[outcome].values.astype(np.float64)
    d = df[treatment].values.astype(np.float64)
    m = df[mediator].values.astype(np.float64)
    X = df[controls].values.astype(np.float64) if controls else np.zeros((len(df), 0))

    # Determine variable types
    def get_var_type(arr):
        unique_vals = np.unique(arr[~np.isnan(arr)])
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
            return 'binary'
        return 'continuous'

    treatment_type = get_var_type(d)
    mediator_type = get_var_type(m)

    return MediationData(
        y=y,
        d=d,
        m=m,
        X=X,
        feature_names=controls if controls else [],
        n=len(y),
        p=len(controls) if controls else 0,
        treatment_type=treatment_type,
        mediator_type=mediator_type
    )


def validate_mediation_setup(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: List[str]
) -> Dict[str, Any]:
    """
    Validate data setup for mediation analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    outcome : str
        Outcome variable name
    treatment : str
        Treatment variable name
    mediator : str
        Mediator variable name
    controls : List[str]
        Control variable names

    Returns
    -------
    Dict[str, Any]
        Validation results and recommendations
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }

    # Check columns exist
    required = [outcome, treatment, mediator] + controls
    missing = [c for c in required if c not in data.columns]
    if missing:
        validation['is_valid'] = False
        validation['errors'].append(f"Missing columns: {missing}")
        return validation

    # Sample size
    n = len(data)
    validation['summary']['n'] = n

    if n < 100:
        validation['warnings'].append(
            f"Small sample size (n={n}). Standard errors may be unreliable."
        )

    if n < 500:
        validation['warnings'].append(
            f"Sample size (n={n}) may be too small for ML-enhanced mediation. "
            "Consider traditional Baron-Kenny approach."
        )

    # Missing values
    n_missing = data[required].isna().any(axis=1).sum()
    if n_missing > 0:
        validation['warnings'].append(
            f"{n_missing} rows have missing values and will be dropped."
        )

    # Treatment distribution
    d = data[treatment].dropna()
    unique_d = d.nunique()
    validation['summary']['treatment_unique_values'] = unique_d

    if unique_d == 2:
        validation['summary']['treatment_type'] = 'binary'
        prop_treated = d.mean()
        validation['summary']['prop_treated'] = prop_treated

        if prop_treated < 0.1 or prop_treated > 0.9:
            validation['warnings'].append(
                f"Imbalanced treatment ({prop_treated:.1%} treated). "
                "Mediation estimates may be unstable."
            )
    else:
        validation['summary']['treatment_type'] = 'continuous'

    # Mediator distribution
    m = data[mediator].dropna()
    validation['summary']['mediator_mean'] = m.mean()
    validation['summary']['mediator_std'] = m.std()

    # Check mediator variation by treatment
    if unique_d == 2:
        m_treated = data.loc[data[treatment] == 1, mediator].dropna()
        m_control = data.loc[data[treatment] == 0, mediator].dropna()

        if len(m_treated) > 0 and len(m_control) > 0:
            m_diff = m_treated.mean() - m_control.mean()
            validation['summary']['mediator_diff_by_treatment'] = m_diff

            if abs(m_diff) < 0.01 * m.std():
                validation['warnings'].append(
                    "Treatment has minimal effect on mediator. "
                    "Mediation effect will likely be near zero."
                )

    # Outcome variation
    y = data[outcome].dropna()
    validation['summary']['outcome_mean'] = y.mean()
    validation['summary']['outcome_std'] = y.std()

    if y.std() < 1e-10:
        validation['is_valid'] = False
        validation['errors'].append("Outcome has no variation.")

    return validation


# =============================================================================
# Learner Utilities
# =============================================================================

def _get_learner(learner_name: str, task: str = 'regression'):
    """
    Get a scikit-learn compatible learner by name.

    Parameters
    ----------
    learner_name : str
        Name of learner: 'lasso', 'ridge', 'random_forest', 'xgboost', 'ols'
    task : str
        'regression' or 'classification'

    Returns
    -------
    sklearn-compatible estimator
    """
    from sklearn.linear_model import (
        LinearRegression, LassoCV, RidgeCV, ElasticNetCV,
        LogisticRegressionCV
    )
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    learner_name = learner_name.lower().replace('-', '_').replace(' ', '_')

    if task == 'regression':
        if learner_name == 'ols' or learner_name == 'linear':
            return LinearRegression()
        elif learner_name == 'lasso':
            return LassoCV(cv=5, n_alphas=50, max_iter=10000)
        elif learner_name == 'ridge':
            return RidgeCV(cv=5, alphas=np.logspace(-4, 4, 50))
        elif learner_name == 'elastic_net':
            return ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                               n_alphas=50, max_iter=10000)
        elif learner_name == 'random_forest' or learner_name == 'rf':
            return RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                n_jobs=-1, random_state=42
            )
        elif learner_name == 'xgboost' or learner_name == 'xgb':
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    n_jobs=-1, random_state=42, verbosity=0
                )
            except ImportError:
                warnings.warn("XGBoost not installed. Falling back to RandomForest.")
                return RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1)
        else:
            raise ValueError(f"Unknown regression learner: {learner_name}")

    elif task == 'classification':
        if learner_name in ['lasso', 'logistic_lasso', 'logistic']:
            return LogisticRegressionCV(
                cv=5, penalty='l1', solver='saga', max_iter=10000,
                Cs=np.logspace(-4, 4, 20), random_state=42
            )
        elif learner_name == 'random_forest' or learner_name == 'rf':
            return RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                n_jobs=-1, random_state=42
            )
        else:
            raise ValueError(f"Unknown classification learner: {learner_name}")

    else:
        raise ValueError(f"Unknown task: {task}")


# =============================================================================
# Total Effect Estimation
# =============================================================================

def estimate_total_effect(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str],
    method: str = 'ols'
) -> Dict[str, Any]:
    """
    Estimate the total effect of treatment on outcome (ignoring mediator).

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    outcome : str
        Outcome variable name
    treatment : str
        Treatment variable name
    controls : List[str]
        Control variable names
    method : str
        Estimation method: 'ols', 'lasso', 'random_forest'

    Returns
    -------
    Dict[str, Any]
        Total effect estimate with standard error
    """
    from sklearn.linear_model import LinearRegression

    # Prepare data
    df = data[[outcome, treatment] + controls].dropna()
    y = df[outcome].values
    d = df[treatment].values
    X = df[controls].values if controls else np.zeros((len(df), 0))

    # Combine treatment and controls
    if X.shape[1] > 0:
        X_full = np.column_stack([d, X])
    else:
        X_full = d.reshape(-1, 1)

    # Fit model
    if method == 'ols':
        model = LinearRegression()
        model.fit(X_full, y)
        total_effect = model.coef_[0]

        # Calculate standard error
        y_pred = model.predict(X_full)
        residuals = y - y_pred
        n = len(y)
        p = X_full.shape[1]
        mse = np.sum(residuals ** 2) / (n - p)

        # Variance of coefficient
        XtX_inv = np.linalg.inv(X_full.T @ X_full)
        var_coef = mse * XtX_inv[0, 0]
        se = np.sqrt(var_coef)

    else:
        # Use cross-fitting for ML methods
        from sklearn.model_selection import cross_val_predict

        learner = _get_learner(method, task='regression')

        # Residualize outcome on X
        if X.shape[1] > 0:
            y_pred_x = cross_val_predict(learner, X, y, cv=5)
            y_resid = y - y_pred_x
            d_pred_x = cross_val_predict(learner, X, d, cv=5)
            d_resid = d - d_pred_x
        else:
            y_resid = y - np.mean(y)
            d_resid = d - np.mean(d)

        # Estimate effect from residuals
        total_effect = np.sum(y_resid * d_resid) / np.sum(d_resid ** 2)

        # Bootstrap SE
        n_bootstrap = 200
        effects = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(y), len(y), replace=True)
            eff = np.sum(y_resid[idx] * d_resid[idx]) / np.sum(d_resid[idx] ** 2)
            effects.append(eff)
        se = np.std(effects)

    # Inference
    z_stat = total_effect / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    ci_lower = total_effect - 1.96 * se
    ci_upper = total_effect + 1.96 * se

    return {
        'effect': total_effect,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'method': method,
        'n': len(y)
    }


# =============================================================================
# Traditional Baron-Kenny Mediation
# =============================================================================

def estimate_baron_kenny(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: List[str],
    robust_se: bool = True
) -> Dict[str, Any]:
    """
    Estimate mediation effects using traditional Baron-Kenny approach.

    This implements the classic 4-step mediation analysis with product-of-
    coefficients estimation (Sobel test).

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    outcome : str
        Outcome variable name (Y)
    treatment : str
        Treatment variable name (D)
    mediator : str
        Mediator variable name (M)
    controls : List[str]
        Control variable names (X)
    robust_se : bool
        Use heteroskedasticity-robust standard errors

    Returns
    -------
    Dict[str, Any]
        Mediation analysis results including ADE, ACME, and proportion mediated
    """
    from sklearn.linear_model import LinearRegression

    # Prepare data
    med_data = create_mediation_data(data, outcome, treatment, mediator, controls)
    y, d, m, X = med_data.y, med_data.d, med_data.m, med_data.X
    n = med_data.n

    # Step 1: Total effect (D -> Y)
    if X.shape[1] > 0:
        X_total = np.column_stack([d, X])
    else:
        X_total = d.reshape(-1, 1)

    model_total = LinearRegression()
    model_total.fit(X_total, y)
    total_effect = model_total.coef_[0]

    # Step 2: Treatment -> Mediator (D -> M)
    if X.shape[1] > 0:
        X_med = np.column_stack([d, X])
    else:
        X_med = d.reshape(-1, 1)

    model_med = LinearRegression()
    model_med.fit(X_med, m)
    alpha = model_med.coef_[0]  # Effect of D on M

    # Step 3 & 4: Full model (D, M -> Y)
    if X.shape[1] > 0:
        X_full = np.column_stack([d, m, X])
    else:
        X_full = np.column_stack([d, m])

    model_full = LinearRegression()
    model_full.fit(X_full, y)
    beta_d = model_full.coef_[0]   # Direct effect (ADE)
    beta_m = model_full.coef_[1]   # Effect of M on Y

    # Calculate effects
    ade = beta_d                   # Average Direct Effect
    acme = alpha * beta_m          # Average Causal Mediation Effect (indirect)

    # Standard errors
    # For mediator model
    m_pred = model_med.predict(X_med)
    resid_m = m - m_pred
    mse_m = np.sum(resid_m ** 2) / (n - X_med.shape[1])

    if robust_se:
        # Robust (sandwich) standard errors
        XtX_inv_m = np.linalg.inv(X_med.T @ X_med)
        meat_m = X_med.T @ np.diag(resid_m ** 2) @ X_med
        var_alpha = (XtX_inv_m @ meat_m @ XtX_inv_m)[0, 0]
    else:
        XtX_inv_m = np.linalg.inv(X_med.T @ X_med)
        var_alpha = mse_m * XtX_inv_m[0, 0]

    se_alpha = np.sqrt(var_alpha)

    # For outcome model
    y_pred = model_full.predict(X_full)
    resid_y = y - y_pred
    mse_y = np.sum(resid_y ** 2) / (n - X_full.shape[1])

    if robust_se:
        XtX_inv_y = np.linalg.inv(X_full.T @ X_full)
        meat_y = X_full.T @ np.diag(resid_y ** 2) @ X_full
        var_coef_y = XtX_inv_y @ meat_y @ XtX_inv_y
    else:
        XtX_inv_y = np.linalg.inv(X_full.T @ X_full)
        var_coef_y = mse_y * XtX_inv_y

    se_beta_d = np.sqrt(var_coef_y[0, 0])
    se_beta_m = np.sqrt(var_coef_y[1, 1])

    # Sobel test for ACME (delta method)
    # Var(alpha * beta_m) = alpha^2 * Var(beta_m) + beta_m^2 * Var(alpha)
    var_acme = alpha**2 * se_beta_m**2 + beta_m**2 * se_alpha**2
    se_acme = np.sqrt(var_acme)

    # SE for total effect
    y_pred_total = model_total.predict(X_total)
    resid_total = y - y_pred_total
    mse_total = np.sum(resid_total ** 2) / (n - X_total.shape[1])
    XtX_inv_total = np.linalg.inv(X_total.T @ X_total)
    se_total = np.sqrt(mse_total * XtX_inv_total[0, 0])

    # Inference for ACME
    z_acme = acme / se_acme
    p_acme = 2 * (1 - stats.norm.cdf(abs(z_acme)))

    # Inference for ADE
    z_ade = ade / se_beta_d
    p_ade = 2 * (1 - stats.norm.cdf(abs(z_ade)))

    # Proportion mediated
    if abs(total_effect) > 1e-10:
        prop_mediated = acme / total_effect
        # Delta method for proportion SE
        var_prop = (1/total_effect**2) * var_acme + (acme**2/total_effect**4) * se_total**2
        se_prop = np.sqrt(var_prop)
    else:
        prop_mediated = np.nan
        se_prop = np.nan

    return {
        'total_effect': total_effect,
        'total_se': se_total,
        'ade': ade,
        'ade_se': se_beta_d,
        'ade_ci_lower': ade - 1.96 * se_beta_d,
        'ade_ci_upper': ade + 1.96 * se_beta_d,
        'ade_pvalue': p_ade,
        'acme': acme,
        'acme_se': se_acme,
        'acme_ci_lower': acme - 1.96 * se_acme,
        'acme_ci_upper': acme + 1.96 * se_acme,
        'acme_pvalue': p_acme,
        'prop_mediated': prop_mediated,
        'prop_mediated_se': se_prop,
        'alpha': alpha,          # D -> M coefficient
        'alpha_se': se_alpha,
        'beta_m': beta_m,        # M -> Y coefficient (controlling for D)
        'beta_m_se': se_beta_m,
        'method': 'baron_kenny',
        'n': n,
        'robust_se': robust_se
    }


# =============================================================================
# ML-Enhanced Mediation
# =============================================================================

def estimate_ml_mediation(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: List[str],
    ml_m: str = 'lasso',
    ml_y: str = 'lasso',
    n_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Estimate mediation effects using ML-enhanced approach (DDML-style).

    Uses cross-fitting to estimate nuisance functions flexibly while
    maintaining valid inference for mediation effects.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    outcome : str
        Outcome variable name (Y)
    treatment : str
        Treatment variable name (D)
    mediator : str
        Mediator variable name (M)
    controls : List[str]
        Control variable names (X)
    ml_m : str
        ML learner for mediator model. Options: 'lasso', 'ridge', 'random_forest', 'xgboost'
    ml_y : str
        ML learner for outcome model. Options: same as ml_m
    n_folds : int
        Number of cross-fitting folds
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Any]
        Mediation analysis results with ML-based estimates
    """
    # Prepare data
    med_data = create_mediation_data(data, outcome, treatment, mediator, controls)
    y, d, m, X = med_data.y, med_data.d, med_data.m, med_data.X
    n = med_data.n

    np.random.seed(random_state)

    # Generate fold indices
    fold_indices = np.random.permutation(n) % n_folds

    # Storage for out-of-sample predictions
    m_hat_d1 = np.zeros(n)   # E[M|D=1, X]
    m_hat_d0 = np.zeros(n)   # E[M|D=0, X]
    y_hat_d1_m = np.zeros(n)  # E[Y|D=1, M, X]
    y_hat_d0_m = np.zeros(n)  # E[Y|D=0, M, X]

    for k in range(n_folds):
        train_mask = fold_indices != k
        test_mask = fold_indices == k

        X_train, X_test = X[train_mask], X[test_mask]
        d_train, d_test = d[train_mask], d[test_mask]
        m_train, m_test = m[train_mask], m[test_mask]
        y_train = y[train_mask]

        # Train mediator model: M = f(D, X)
        if X.shape[1] > 0:
            DX_train = np.column_stack([d_train, X_train])
            DX_test = np.column_stack([d_test, X_test])
            X_d1_test = np.column_stack([np.ones(X_test.shape[0]), X_test])
            X_d0_test = np.column_stack([np.zeros(X_test.shape[0]), X_test])
        else:
            DX_train = d_train.reshape(-1, 1)
            DX_test = d_test.reshape(-1, 1)
            X_d1_test = np.ones((len(d_test), 1))
            X_d0_test = np.zeros((len(d_test), 1))

        learner_m = _get_learner(ml_m, task='regression')
        learner_m.fit(DX_train, m_train)

        # Predict M under D=1 and D=0
        m_hat_d1[test_mask] = learner_m.predict(X_d1_test)
        m_hat_d0[test_mask] = learner_m.predict(X_d0_test)

        # Train outcome model: Y = g(D, M, X)
        if X.shape[1] > 0:
            DMX_train = np.column_stack([d_train, m_train, X_train])
        else:
            DMX_train = np.column_stack([d_train, m_train])

        learner_y = _get_learner(ml_y, task='regression')
        learner_y.fit(DMX_train, y_train)

        # Predict Y under different (D, M) combinations
        # For ACME: compare Y(d, M(1)) vs Y(d, M(0))
        # For ADE: compare Y(1, M(d)) vs Y(0, M(d))
        if X.shape[1] > 0:
            DMX_d1_m = np.column_stack([np.ones(X_test.shape[0]), m_test, X_test])
            DMX_d0_m = np.column_stack([np.zeros(X_test.shape[0]), m_test, X_test])
        else:
            DMX_d1_m = np.column_stack([np.ones(len(d_test)), m_test])
            DMX_d0_m = np.column_stack([np.zeros(len(d_test)), m_test])

        y_hat_d1_m[test_mask] = learner_y.predict(DMX_d1_m)
        y_hat_d0_m[test_mask] = learner_y.predict(DMX_d0_m)

    # Calculate treatment effect on mediator
    alpha = np.mean(m_hat_d1 - m_hat_d0)

    # Calculate ADE: E[Y(1, M(d)) - Y(0, M(d))]
    # Using observed M as proxy for M(d)
    ade = np.mean(y_hat_d1_m - y_hat_d0_m)

    # Calculate total effect
    total_effect = estimate_total_effect(data, outcome, treatment, controls, method=ml_y)['effect']

    # ACME = Total Effect - ADE
    acme = total_effect - ade

    # Bootstrap for standard errors
    n_bootstrap = 500
    ade_boot = []
    acme_boot = []
    total_boot = []

    for b in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)

        # Recompute with bootstrap sample
        y_b, d_b, m_b, X_b = y[idx], d[idx], m[idx], X[idx]

        # Simple resampling estimator for ADE
        ade_b = np.mean((y_hat_d1_m - y_hat_d0_m)[idx])
        ade_boot.append(ade_b)

        # Bootstrap total effect
        total_b = np.mean(y_b[d_b == 1]) - np.mean(y_b[d_b == 0]) if med_data.treatment_type == 'binary' else total_effect
        total_boot.append(total_b)

        # ACME
        acme_boot.append(total_b - ade_b)

    se_ade = np.std(ade_boot)
    se_acme = np.std(acme_boot)
    se_total = np.std(total_boot)

    # Inference
    z_ade = ade / se_ade if se_ade > 0 else 0
    p_ade = 2 * (1 - stats.norm.cdf(abs(z_ade)))

    z_acme = acme / se_acme if se_acme > 0 else 0
    p_acme = 2 * (1 - stats.norm.cdf(abs(z_acme)))

    # Proportion mediated
    if abs(total_effect) > 1e-10:
        prop_mediated = acme / total_effect
        # Bootstrap SE for proportion
        prop_boot = [a / t if abs(t) > 1e-10 else np.nan for a, t in zip(acme_boot, total_boot)]
        se_prop = np.nanstd(prop_boot)
    else:
        prop_mediated = np.nan
        se_prop = np.nan

    return {
        'total_effect': total_effect,
        'total_se': se_total,
        'ade': ade,
        'ade_se': se_ade,
        'ade_ci_lower': ade - 1.96 * se_ade,
        'ade_ci_upper': ade + 1.96 * se_ade,
        'ade_pvalue': p_ade,
        'acme': acme,
        'acme_se': se_acme,
        'acme_ci_lower': acme - 1.96 * se_acme,
        'acme_ci_upper': acme + 1.96 * se_acme,
        'acme_pvalue': p_acme,
        'prop_mediated': prop_mediated,
        'prop_mediated_se': se_prop,
        'alpha': alpha,  # D -> M effect
        'method': 'ml_enhanced',
        'ml_m': ml_m,
        'ml_y': ml_y,
        'n_folds': n_folds,
        'n': n
    }


def compute_ade(
    outcome_model,
    mediator_model,
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: List[str],
    treatment_value: int = 0
) -> Dict[str, float]:
    """
    Compute Average Direct Effect using fitted models.

    ADE(d) = E[Y(1, M(d)) - Y(0, M(d))]

    Parameters
    ----------
    outcome_model : sklearn estimator
        Fitted model for E[Y|D, M, X]
    mediator_model : sklearn estimator
        Fitted model for E[M|D, X]
    data : pd.DataFrame
        Input dataset
    outcome, treatment, mediator, controls : str, List[str]
        Variable names
    treatment_value : int
        Value of d in M(d): 0 for ADE(0), 1 for ADE(1)

    Returns
    -------
    Dict[str, float]
        ADE estimate
    """
    med_data = create_mediation_data(data, outcome, treatment, mediator, controls)
    X = med_data.X
    n = med_data.n

    # Predict M under treatment_value
    if X.shape[1] > 0:
        DX = np.column_stack([np.full(n, treatment_value), X])
    else:
        DX = np.full((n, 1), treatment_value)

    m_pred = mediator_model.predict(DX)

    # Predict Y under D=1 and D=0 with this M
    if X.shape[1] > 0:
        DMX_d1 = np.column_stack([np.ones(n), m_pred, X])
        DMX_d0 = np.column_stack([np.zeros(n), m_pred, X])
    else:
        DMX_d1 = np.column_stack([np.ones(n), m_pred])
        DMX_d0 = np.column_stack([np.zeros(n), m_pred])

    y_d1 = outcome_model.predict(DMX_d1)
    y_d0 = outcome_model.predict(DMX_d0)

    ade = np.mean(y_d1 - y_d0)

    return {'ade': ade, 'treatment_value': treatment_value}


def compute_acme(
    outcome_model,
    mediator_model,
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: List[str],
    treatment_value: int = 1
) -> Dict[str, float]:
    """
    Compute Average Causal Mediation Effect using fitted models.

    ACME(d) = E[Y(d, M(1)) - Y(d, M(0))]

    Parameters
    ----------
    outcome_model : sklearn estimator
        Fitted model for E[Y|D, M, X]
    mediator_model : sklearn estimator
        Fitted model for E[M|D, X]
    data : pd.DataFrame
        Input dataset
    outcome, treatment, mediator, controls : str, List[str]
        Variable names
    treatment_value : int
        Value of d in Y(d, M): 0 for ACME(0), 1 for ACME(1)

    Returns
    -------
    Dict[str, float]
        ACME estimate
    """
    med_data = create_mediation_data(data, outcome, treatment, mediator, controls)
    X = med_data.X
    n = med_data.n

    # Predict M under D=1 and D=0
    if X.shape[1] > 0:
        DX_d1 = np.column_stack([np.ones(n), X])
        DX_d0 = np.column_stack([np.zeros(n), X])
    else:
        DX_d1 = np.ones((n, 1))
        DX_d0 = np.zeros((n, 1))

    m_d1 = mediator_model.predict(DX_d1)
    m_d0 = mediator_model.predict(DX_d0)

    # Predict Y under fixed D with M(1) and M(0)
    if X.shape[1] > 0:
        DMX_m1 = np.column_stack([np.full(n, treatment_value), m_d1, X])
        DMX_m0 = np.column_stack([np.full(n, treatment_value), m_d0, X])
    else:
        DMX_m1 = np.column_stack([np.full(n, treatment_value), m_d1])
        DMX_m0 = np.column_stack([np.full(n, treatment_value), m_d0])

    y_m1 = outcome_model.predict(DMX_m1)
    y_m0 = outcome_model.predict(DMX_m0)

    acme = np.mean(y_m1 - y_m0)

    return {'acme': acme, 'treatment_value': treatment_value}


def proportion_mediated(acme: float, total_effect: float) -> Dict[str, float]:
    """
    Calculate proportion of total effect that is mediated.

    Parameters
    ----------
    acme : float
        Average Causal Mediation Effect (indirect effect)
    total_effect : float
        Total effect

    Returns
    -------
    Dict[str, float]
        Proportion mediated and related quantities
    """
    if abs(total_effect) < 1e-10:
        return {
            'proportion': np.nan,
            'interpretation': 'Total effect is approximately zero; proportion undefined'
        }

    prop = acme / total_effect

    if prop > 1.0:
        interpretation = (
            "Proportion > 100% indicates inconsistent mediation "
            "(direct and indirect effects have opposite signs)"
        )
    elif prop < 0:
        interpretation = (
            "Negative proportion indicates suppression effect "
            "(mediator suppresses direct effect)"
        )
    elif prop > 0.8:
        interpretation = "Strong mediation (most of effect through mediator)"
    elif prop > 0.5:
        interpretation = "Substantial mediation"
    elif prop > 0.2:
        interpretation = "Moderate mediation"
    else:
        interpretation = "Weak mediation (most of effect is direct)"

    return {
        'proportion': prop,
        'proportion_pct': prop * 100,
        'direct_proportion': 1 - prop,
        'interpretation': interpretation
    }


# =============================================================================
# Sensitivity Analysis
# =============================================================================

def sensitivity_analysis_mediation(
    acme: float,
    acme_se: float,
    rho_range: np.ndarray = None,
    sigma_m: float = 1.0,
    sigma_y: float = 1.0
) -> Dict[str, Any]:
    """
    Perform sensitivity analysis for unmeasured confounding.

    Assesses how robust the ACME estimate is to violations of
    sequential ignorability.

    Parameters
    ----------
    acme : float
        Estimated ACME
    acme_se : float
        Standard error of ACME
    rho_range : np.ndarray, optional
        Range of sensitivity parameter values (correlation between
        M and Y errors). Default: [-0.5, 0.5]
    sigma_m : float
        Standard deviation of mediator model residuals
    sigma_y : float
        Standard deviation of outcome model residuals

    Returns
    -------
    Dict[str, Any]
        Sensitivity analysis results including breakpoint
    """
    if rho_range is None:
        rho_range = np.arange(-0.5, 0.51, 0.05)

    # Under sensitivity parameter rho:
    # ACME(rho) = ACME(0) - rho * sigma_m * sigma_y
    # (simplified linear approximation)

    acme_adjusted = []
    ci_lower_adjusted = []
    ci_upper_adjusted = []

    for rho in rho_range:
        bias = rho * sigma_m * sigma_y
        acme_rho = acme - bias
        acme_adjusted.append(acme_rho)
        ci_lower_adjusted.append(acme_rho - 1.96 * acme_se)
        ci_upper_adjusted.append(acme_rho + 1.96 * acme_se)

    acme_adjusted = np.array(acme_adjusted)
    ci_lower_adjusted = np.array(ci_lower_adjusted)
    ci_upper_adjusted = np.array(ci_upper_adjusted)

    # Find breakpoint where ACME = 0
    # ACME - rho * sigma_m * sigma_y = 0
    # rho = ACME / (sigma_m * sigma_y)
    breakpoint = acme / (sigma_m * sigma_y) if sigma_m * sigma_y != 0 else np.nan

    # Determine robustness
    if np.isnan(breakpoint):
        robustness = 'undefined'
        interpretation = 'Cannot assess robustness (sigma values needed)'
    elif abs(breakpoint) > 0.5:
        robustness = 'very_robust'
        interpretation = (
            f'Results are VERY ROBUST. An unmeasured confounder would need '
            f'to induce rho > {abs(breakpoint):.2f} to eliminate mediation effect.'
        )
    elif abs(breakpoint) > 0.3:
        robustness = 'robust'
        interpretation = (
            f'Results are ROBUST. Breakpoint rho = {abs(breakpoint):.2f} is '
            f'moderately large.'
        )
    elif abs(breakpoint) > 0.1:
        robustness = 'moderate'
        interpretation = (
            f'Results are MODERATELY SENSITIVE. Breakpoint rho = {abs(breakpoint):.2f}. '
            f'A modest unmeasured confounder could eliminate the effect.'
        )
    else:
        robustness = 'sensitive'
        interpretation = (
            f'Results are SENSITIVE to confounding. Breakpoint rho = {abs(breakpoint):.2f}. '
            f'Even weak unmeasured confounding could eliminate the mediation effect.'
        )

    return {
        'rho_range': rho_range.tolist(),
        'acme_adjusted': acme_adjusted.tolist(),
        'ci_lower': ci_lower_adjusted.tolist(),
        'ci_upper': ci_upper_adjusted.tolist(),
        'breakpoint': breakpoint,
        'robustness': robustness,
        'interpretation': interpretation
    }


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_mediation_ci(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: List[str],
    n_bootstrap: int = 1000,
    method: str = 'baron_kenny',
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Compute bootstrap confidence intervals for mediation effects.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    outcome : str
        Outcome variable name
    treatment : str
        Treatment variable name
    mediator : str
        Mediator variable name
    controls : List[str]
        Control variable names
    n_bootstrap : int
        Number of bootstrap replications
    method : str
        'baron_kenny' or 'ml_enhanced'
    confidence_level : float
        Confidence level (default 0.95)
    random_state : int
        Random seed

    Returns
    -------
    Dict[str, Any]
        Bootstrap confidence intervals for all mediation effects
    """
    np.random.seed(random_state)
    n = len(data)

    # Storage for bootstrap estimates
    boot_total = []
    boot_ade = []
    boot_acme = []
    boot_prop = []

    for b in range(n_bootstrap):
        # Resample
        idx = np.random.choice(n, n, replace=True)
        boot_data = data.iloc[idx].reset_index(drop=True)

        try:
            if method == 'baron_kenny':
                result = estimate_baron_kenny(
                    boot_data, outcome, treatment, mediator, controls
                )
            else:
                result = estimate_ml_mediation(
                    boot_data, outcome, treatment, mediator, controls,
                    n_folds=3  # Fewer folds for bootstrap
                )

            boot_total.append(result['total_effect'])
            boot_ade.append(result['ade'])
            boot_acme.append(result['acme'])
            if not np.isnan(result['prop_mediated']):
                boot_prop.append(result['prop_mediated'])

        except Exception:
            # Skip failed bootstrap iterations
            continue

    # Calculate percentile confidence intervals
    alpha = 1 - confidence_level
    lower_q = alpha / 2 * 100
    upper_q = (1 - alpha / 2) * 100

    def percentile_ci(boot_vals):
        if len(boot_vals) < 10:
            return np.nan, np.nan
        return np.percentile(boot_vals, lower_q), np.percentile(boot_vals, upper_q)

    total_ci = percentile_ci(boot_total)
    ade_ci = percentile_ci(boot_ade)
    acme_ci = percentile_ci(boot_acme)
    prop_ci = percentile_ci(boot_prop)

    return {
        'n_bootstrap': n_bootstrap,
        'n_successful': len(boot_total),
        'confidence_level': confidence_level,
        'total_effect': {
            'mean': np.mean(boot_total),
            'se': np.std(boot_total),
            'ci_lower': total_ci[0],
            'ci_upper': total_ci[1]
        },
        'ade': {
            'mean': np.mean(boot_ade),
            'se': np.std(boot_ade),
            'ci_lower': ade_ci[0],
            'ci_upper': ade_ci[1]
        },
        'acme': {
            'mean': np.mean(boot_acme),
            'se': np.std(boot_acme),
            'ci_lower': acme_ci[0],
            'ci_upper': acme_ci[1]
        },
        'prop_mediated': {
            'mean': np.mean(boot_prop) if boot_prop else np.nan,
            'se': np.std(boot_prop) if boot_prop else np.nan,
            'ci_lower': prop_ci[0],
            'ci_upper': prop_ci[1]
        }
    }


# =============================================================================
# Full Analysis Workflow
# =============================================================================

def run_full_mediation_analysis(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: List[str],
    method: str = 'auto',
    n_bootstrap: int = 500,
    run_sensitivity: bool = True,
    random_state: int = 42
) -> CausalOutput:
    """
    Run complete mediation analysis workflow.

    This function:
    1. Validates data structure
    2. Runs main mediation estimation
    3. Computes bootstrap confidence intervals
    4. Performs sensitivity analysis
    5. Generates comprehensive output

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    outcome : str
        Outcome variable name
    treatment : str
        Treatment variable name
    mediator : str
        Mediator variable name
    controls : List[str]
        Control variable names
    method : str
        'baron_kenny', 'ml_enhanced', or 'auto' (selects based on sample size and p)
    n_bootstrap : int
        Number of bootstrap replications
    run_sensitivity : bool
        Whether to run sensitivity analysis
    random_state : int
        Random seed

    Returns
    -------
    CausalOutput
        Comprehensive mediation analysis results
    """
    # Step 1: Validate setup
    validation = validate_mediation_setup(data, outcome, treatment, mediator, controls)

    if not validation['is_valid']:
        raise ValueError(f"Data validation failed: {validation['errors']}")

    if validation['warnings']:
        for w in validation['warnings']:
            warnings.warn(w)

    # Step 2: Choose method
    n = len(data)
    p = len(controls) if controls else 0

    if method == 'auto':
        if n >= 500 and p >= 20:
            method = 'ml_enhanced'
        else:
            method = 'baron_kenny'

    # Step 3: Run main estimation
    if method == 'baron_kenny':
        result = estimate_baron_kenny(data, outcome, treatment, mediator, controls)
    else:
        result = estimate_ml_mediation(
            data, outcome, treatment, mediator, controls,
            random_state=random_state
        )

    # Step 4: Bootstrap confidence intervals
    boot_ci = bootstrap_mediation_ci(
        data, outcome, treatment, mediator, controls,
        n_bootstrap=n_bootstrap, method=method, random_state=random_state
    )

    # Use bootstrap CIs if available
    if boot_ci['n_successful'] >= 100:
        result['acme_ci_lower'] = boot_ci['acme']['ci_lower']
        result['acme_ci_upper'] = boot_ci['acme']['ci_upper']
        result['ade_ci_lower'] = boot_ci['ade']['ci_lower']
        result['ade_ci_upper'] = boot_ci['ade']['ci_upper']

    # Step 5: Sensitivity analysis
    if run_sensitivity:
        # Estimate residual standard deviations
        med_data = create_mediation_data(data, outcome, treatment, mediator, controls)

        sensitivity = sensitivity_analysis_mediation(
            acme=result['acme'],
            acme_se=result['acme_se'],
            sigma_m=med_data.m.std(),
            sigma_y=med_data.y.std()
        )
    else:
        sensitivity = {}

    # Step 6: Generate output
    # Create summary table
    summary_lines = [
        "=" * 70,
        "CAUSAL MEDIATION ANALYSIS RESULTS".center(70),
        "=" * 70,
        "",
        f"Causal Pathway: {treatment} -> {mediator} -> {outcome}",
        f"Method: {method}",
        f"Sample Size: {n:,}",
        "",
        "-" * 70,
        "Effect Decomposition",
        "-" * 70,
        "",
        f"{'Effect':<25} {'Estimate':>12} {'Std.Err':>12} {'95% CI':>20}",
        "-" * 70,
    ]

    # Total effect
    total_stars = '***' if result.get('total_se', 0) > 0 and abs(result['total_effect'] / result.get('total_se', 1)) > 2.58 else ('**' if abs(result['total_effect'] / result.get('total_se', 1)) > 1.96 else '')
    summary_lines.append(
        f"{'Total Effect':<25} {result['total_effect']:>12.4f}{total_stars:<4} ({result.get('total_se', np.nan):>8.4f}) "
        f"[{result['total_effect'] - 1.96*result.get('total_se', 0):.4f}, {result['total_effect'] + 1.96*result.get('total_se', 0):.4f}]"
    )

    # ADE
    ade_stars = '***' if result['ade_pvalue'] < 0.01 else ('**' if result['ade_pvalue'] < 0.05 else ('*' if result['ade_pvalue'] < 0.1 else ''))
    summary_lines.append(
        f"{'Direct Effect (ADE)':<25} {result['ade']:>12.4f}{ade_stars:<4} ({result['ade_se']:>8.4f}) "
        f"[{result['ade_ci_lower']:.4f}, {result['ade_ci_upper']:.4f}]"
    )

    # ACME
    acme_stars = '***' if result['acme_pvalue'] < 0.01 else ('**' if result['acme_pvalue'] < 0.05 else ('*' if result['acme_pvalue'] < 0.1 else ''))
    summary_lines.append(
        f"{'Indirect Effect (ACME)':<25} {result['acme']:>12.4f}{acme_stars:<4} ({result['acme_se']:>8.4f}) "
        f"[{result['acme_ci_lower']:.4f}, {result['acme_ci_upper']:.4f}]"
    )

    summary_lines.extend([
        "-" * 70,
        f"Proportion Mediated: {result['prop_mediated']*100:.1f}% "
        f"(SE = {result.get('prop_mediated_se', np.nan)*100:.1f}%)",
        "",
    ])

    # Sensitivity
    if sensitivity:
        summary_lines.extend([
            "-" * 70,
            "Sensitivity Analysis",
            "-" * 70,
            "",
            f"ACME crosses zero at rho = {sensitivity.get('breakpoint', np.nan):.3f}",
            f"Robustness: {sensitivity.get('robustness', 'N/A').upper()}",
            f"Interpretation: {sensitivity.get('interpretation', 'N/A')}",
            "",
        ])

    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append("*Notes: *** p<0.01, ** p<0.05, * p<0.1")
    summary_lines.append(f"Bootstrap CI based on {boot_ci.get('n_successful', 0)} replications.")

    summary_table = "\n".join(summary_lines)

    # Generate interpretation
    prop_info = proportion_mediated(result['acme'], result['total_effect'])
    interpretation = (
        f"The total effect of {treatment} on {outcome} is {result['total_effect']:.4f}. "
        f"This decomposes into: (1) a direct effect of {result['ade']:.4f} "
        f"({'significant' if result['ade_pvalue'] < 0.05 else 'not significant'} at 5%), and "
        f"(2) an indirect effect through {mediator} of {result['acme']:.4f} "
        f"({'significant' if result['acme_pvalue'] < 0.05 else 'not significant'} at 5%). "
        f"{prop_info['interpretation']}."
    )

    if sensitivity:
        interpretation += f" {sensitivity.get('interpretation', '')}"

    # Compile diagnostics
    diagnostics = {
        'method': method,
        'validation': validation['summary'],
        'main_results': result,
        'bootstrap': boot_ci,
        'sensitivity': sensitivity,
        'proportion_mediated_info': prop_info
    }

    return CausalOutput(
        effect=result['acme'],  # Primary effect is the mediation (indirect) effect
        se=result['acme_se'],
        ci_lower=result['acme_ci_lower'],
        ci_upper=result['acme_ci_upper'],
        p_value=result['acme_pvalue'],
        diagnostics=diagnostics,
        summary_table=summary_table,
        interpretation=interpretation
    )


# =============================================================================
# Visualization
# =============================================================================

def plot_mediation_pathway(
    result: Dict[str, Any],
    treatment_name: str = "Treatment",
    mediator_name: str = "Mediator",
    outcome_name: str = "Outcome",
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None
) -> plt.Figure:
    """
    Create visualization of mediation pathway with effect sizes.

    Parameters
    ----------
    result : Dict[str, Any]
        Output from mediation estimation function
    treatment_name, mediator_name, outcome_name : str
        Names for display
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
        Mediation pathway diagram
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Draw boxes for variables
    box_style = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='navy')

    # Treatment box (left)
    ax.text(1.5, 3, treatment_name, ha='center', va='center', fontsize=12,
            fontweight='bold', bbox=box_style)

    # Mediator box (top center)
    ax.text(5, 5, mediator_name, ha='center', va='center', fontsize=12,
            fontweight='bold', bbox=box_style)

    # Outcome box (right)
    ax.text(8.5, 3, outcome_name, ha='center', va='center', fontsize=12,
            fontweight='bold', bbox=box_style)

    # Draw arrows
    arrow_style = dict(arrowstyle='->', color='navy', lw=2)

    # Treatment -> Mediator
    ax.annotate('', xy=(4, 4.7), xytext=(2.2, 3.4),
                arrowprops=arrow_style)
    alpha = result.get('alpha', np.nan)
    ax.text(2.8, 4.3, f'a = {alpha:.3f}', fontsize=10, ha='center')

    # Mediator -> Outcome
    ax.annotate('', xy=(7.8, 3.4), xytext=(6, 4.7),
                arrowprops=arrow_style)
    beta_m = result.get('beta_m', np.nan)
    ax.text(7.2, 4.3, f'b = {beta_m:.3f}', fontsize=10, ha='center')

    # Treatment -> Outcome (direct)
    ax.annotate('', xy=(7.6, 3), xytext=(2.4, 3),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    ade = result.get('ade', np.nan)
    ade_stars = '***' if result.get('ade_pvalue', 1) < 0.01 else '**' if result.get('ade_pvalue', 1) < 0.05 else '*' if result.get('ade_pvalue', 1) < 0.1 else ''
    ax.text(5, 2.5, f"Direct (c') = {ade:.3f}{ade_stars}", fontsize=11,
            ha='center', color='darkgreen', fontweight='bold')

    # Effect decomposition box
    total = result.get('total_effect', np.nan)
    acme = result.get('acme', np.nan)
    prop = result.get('prop_mediated', np.nan)

    acme_stars = '***' if result.get('acme_pvalue', 1) < 0.01 else '**' if result.get('acme_pvalue', 1) < 0.05 else '*' if result.get('acme_pvalue', 1) < 0.1 else ''

    text_box = (
        f"Effect Decomposition\n"
        f"───────────────────\n"
        f"Total Effect: {total:.4f}\n"
        f"Direct (ADE): {ade:.4f}\n"
        f"Indirect (ACME): {acme:.4f}{acme_stars}\n"
        f"% Mediated: {prop*100:.1f}%"
    )

    ax.text(5, 0.8, text_box, ha='center', va='center', fontsize=10,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.8))

    ax.set_title("Causal Mediation Pathway", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_sensitivity_analysis(
    sensitivity: Dict[str, Any],
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None
) -> plt.Figure:
    """
    Plot sensitivity analysis results.

    Parameters
    ----------
    sensitivity : Dict[str, Any]
        Output from sensitivity_analysis_mediation()
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
        Sensitivity analysis plot
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    rho = np.array(sensitivity['rho_range'])
    acme = np.array(sensitivity['acme_adjusted'])
    ci_lower = np.array(sensitivity['ci_lower'])
    ci_upper = np.array(sensitivity['ci_upper'])

    # Plot ACME vs rho
    ax.plot(rho, acme, 'b-', linewidth=2, label='ACME(rho)')
    ax.fill_between(rho, ci_lower, ci_upper, alpha=0.3, color='blue', label='95% CI')

    # Reference lines
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Zero effect')
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)

    # Mark breakpoint
    breakpoint = sensitivity.get('breakpoint', np.nan)
    if not np.isnan(breakpoint) and -0.5 <= breakpoint <= 0.5:
        ax.axvline(x=breakpoint, color='orange', linestyle='--', linewidth=2)
        ax.scatter([breakpoint], [0], color='orange', s=100, zorder=5)
        ax.annotate(f'Breakpoint\nrho = {breakpoint:.2f}',
                   xy=(breakpoint, 0), xytext=(breakpoint + 0.1, acme[len(acme)//2] * 0.3),
                   fontsize=10, color='orange',
                   arrowprops=dict(arrowstyle='->', color='orange'))

    ax.set_xlabel('Sensitivity Parameter (rho)', fontsize=12)
    ax.set_ylabel('ACME', fontsize=12)
    ax.set_title('Sensitivity Analysis: ACME vs. Unmeasured Confounding', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    interp = sensitivity.get('interpretation', '')
    if interp:
        ax.text(0.02, 0.02, interp, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Compare Methods
# =============================================================================

def compare_mediation_methods(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    controls: List[str]
) -> Dict[str, Any]:
    """
    Compare Baron-Kenny vs ML-enhanced mediation results.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    outcome, treatment, mediator : str
        Variable names
    controls : List[str]
        Control variable names

    Returns
    -------
    Dict[str, Any]
        Comparison of methods
    """
    # Traditional Baron-Kenny
    bk_result = estimate_baron_kenny(data, outcome, treatment, mediator, controls)

    # ML-enhanced with different learners
    ml_lasso = estimate_ml_mediation(
        data, outcome, treatment, mediator, controls,
        ml_m='lasso', ml_y='lasso'
    )

    ml_rf = estimate_ml_mediation(
        data, outcome, treatment, mediator, controls,
        ml_m='random_forest', ml_y='random_forest'
    )

    # Create comparison table
    table_results = [
        {
            'treatment_effect': bk_result['acme'],
            'treatment_se': bk_result['acme_se'],
            'treatment_pval': bk_result['acme_pvalue'],
            'controls': True,
            'fixed_effects': 'OLS',
            'n_obs': bk_result['n'],
            'r_squared': bk_result['prop_mediated']
        },
        {
            'treatment_effect': ml_lasso['acme'],
            'treatment_se': ml_lasso['acme_se'],
            'treatment_pval': ml_lasso['acme_pvalue'],
            'controls': True,
            'fixed_effects': 'ML (Lasso)',
            'n_obs': ml_lasso['n'],
            'r_squared': ml_lasso['prop_mediated']
        },
        {
            'treatment_effect': ml_rf['acme'],
            'treatment_se': ml_rf['acme_se'],
            'treatment_pval': ml_rf['acme_pvalue'],
            'controls': True,
            'fixed_effects': 'ML (RF)',
            'n_obs': ml_rf['n'],
            'r_squared': ml_rf['prop_mediated']
        }
    ]

    summary_table = create_regression_table(
        results=table_results,
        column_names=["(1) Baron-Kenny", "(2) ML-Lasso", "(3) ML-RF"],
        title="Mediation Analysis: Method Comparison (ACME)",
        notes="ACME = Average Causal Mediation Effect (indirect effect). "
              "R-squared column shows proportion mediated."
    )

    return {
        'baron_kenny': bk_result,
        'ml_lasso': ml_lasso,
        'ml_rf': ml_rf,
        'summary_table': summary_table,
        'acme_range': [
            min(bk_result['acme'], ml_lasso['acme'], ml_rf['acme']),
            max(bk_result['acme'], ml_lasso['acme'], ml_rf['acme'])
        ],
        'all_significant': all([
            bk_result['acme_pvalue'] < 0.05,
            ml_lasso['acme_pvalue'] < 0.05,
            ml_rf['acme_pvalue'] < 0.05
        ])
    }


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_mediation_data(
    n: int = 1000,
    ade: float = 0.5,
    acme: float = 0.3,
    alpha: float = 0.6,
    p_controls: int = 5,
    noise_std: float = 1.0,
    treatment_type: str = 'binary',
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic mediation data with known effects.

    Parameters
    ----------
    n : int
        Number of observations
    ade : float
        True Average Direct Effect
    acme : float
        True Average Causal Mediation Effect
    alpha : float
        Effect of D on M (D -> M)
    p_controls : int
        Number of control variables
    noise_std : float
        Noise standard deviation
    treatment_type : str
        'binary' or 'continuous'
    random_state : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (data, true_params)
    """
    np.random.seed(random_state)

    # Control variables
    X = np.random.randn(n, p_controls)

    # Treatment
    if treatment_type == 'binary':
        propensity = 1 / (1 + np.exp(-X[:, 0] * 0.5))
        d = np.random.binomial(1, propensity, n).astype(float)
    else:
        d = X[:, 0] * 0.5 + np.random.randn(n) * 0.5

    # Mediator: M = alpha * D + X effects + noise
    beta_m = alpha / acme if acme != 0 else 1.0  # Calibrate so acme = alpha * beta_m
    m = alpha * d + X @ np.random.randn(p_controls) * 0.2 + np.random.randn(n) * noise_std * 0.5

    # Outcome: Y = ade * D + beta_m * M + X effects + noise
    y = ade * d + beta_m * m + X @ np.random.randn(p_controls) * 0.3 + np.random.randn(n) * noise_std

    # Create DataFrame
    columns = [f'x{i+1}' for i in range(p_controls)]
    df = pd.DataFrame(X, columns=columns)
    df['d'] = d
    df['m'] = m
    df['y'] = y

    true_params = {
        'true_ade': ade,
        'true_acme': alpha * beta_m,
        'true_total': ade + alpha * beta_m,
        'true_alpha': alpha,
        'true_beta_m': beta_m,
        'true_prop_mediated': (alpha * beta_m) / (ade + alpha * beta_m) if (ade + alpha * beta_m) != 0 else np.nan,
        'n': n,
        'p_controls': p_controls,
        'treatment_type': treatment_type
    }

    return df, true_params


# =============================================================================
# Validation
# =============================================================================

def validate_estimator(verbose: bool = True) -> Dict[str, Any]:
    """
    Validate mediation estimator on synthetic data with known effects.

    Tests both Baron-Kenny and ML-enhanced methods.

    Returns
    -------
    Dict[str, Any]
        Validation results including bias assessment
    """
    # Generate synthetic data
    true_ade = 0.5
    true_acme = 0.3
    alpha = 0.6

    df, true_params = generate_synthetic_mediation_data(
        n=2000,
        ade=true_ade,
        acme=true_acme,
        alpha=alpha,
        p_controls=10,
        noise_std=1.0,
        treatment_type='binary',
        random_state=42
    )

    controls = [c for c in df.columns if c.startswith('x')]

    # Test Baron-Kenny
    result_bk = estimate_baron_kenny(df, 'y', 'd', 'm', controls)

    # Test ML-enhanced
    result_ml = estimate_ml_mediation(df, 'y', 'd', 'm', controls)

    # Calculate bias
    bk_acme_bias = result_bk['acme'] - true_params['true_acme']
    bk_acme_bias_pct = abs(bk_acme_bias / true_params['true_acme']) * 100

    ml_acme_bias = result_ml['acme'] - true_params['true_acme']
    ml_acme_bias_pct = abs(ml_acme_bias / true_params['true_acme']) * 100

    # Validation passes if bias < 20% for both
    bk_passed = bk_acme_bias_pct < 20.0
    ml_passed = ml_acme_bias_pct < 20.0
    overall_passed = bk_passed and ml_passed

    validation_result = {
        'true_params': true_params,
        'baron_kenny': {
            'estimated_acme': result_bk['acme'],
            'estimated_ade': result_bk['ade'],
            'acme_se': result_bk['acme_se'],
            'acme_bias': bk_acme_bias,
            'acme_bias_pct': bk_acme_bias_pct,
            'passed': bk_passed,
            'ci_covers_truth': (result_bk['acme_ci_lower'] <=
                               true_params['true_acme'] <=
                               result_bk['acme_ci_upper'])
        },
        'ml_enhanced': {
            'estimated_acme': result_ml['acme'],
            'estimated_ade': result_ml['ade'],
            'acme_se': result_ml['acme_se'],
            'acme_bias': ml_acme_bias,
            'acme_bias_pct': ml_acme_bias_pct,
            'passed': ml_passed,
            'ci_covers_truth': (result_ml['acme_ci_lower'] <=
                               true_params['true_acme'] <=
                               result_ml['acme_ci_upper'])
        },
        'overall_passed': overall_passed
    }

    if verbose:
        print("=" * 60)
        print("MEDIATION ESTIMATOR VALIDATION")
        print("=" * 60)
        print(f"\nSynthetic Data:")
        print(f"  n = {true_params['n']}")
        print(f"  True ADE: {true_params['true_ade']:.4f}")
        print(f"  True ACME: {true_params['true_acme']:.4f}")
        print(f"  True Total: {true_params['true_total']:.4f}")
        print(f"  True Prop. Mediated: {true_params['true_prop_mediated']:.1%}")
        print()
        print("-" * 60)
        print("BARON-KENNY METHOD")
        print("-" * 60)
        print(f"  Estimated ACME: {result_bk['acme']:.4f}")
        print(f"  Estimated ADE: {result_bk['ade']:.4f}")
        print(f"  Bias (ACME): {bk_acme_bias:.4f} ({bk_acme_bias_pct:.2f}%)")
        print(f"  95% CI: [{result_bk['acme_ci_lower']:.4f}, {result_bk['acme_ci_upper']:.4f}]")
        print(f"  CI covers truth: {validation_result['baron_kenny']['ci_covers_truth']}")
        print(f"  VALIDATION: {'PASSED' if bk_passed else 'FAILED'}")
        print()
        print("-" * 60)
        print("ML-ENHANCED METHOD")
        print("-" * 60)
        print(f"  Estimated ACME: {result_ml['acme']:.4f}")
        print(f"  Estimated ADE: {result_ml['ade']:.4f}")
        print(f"  Bias (ACME): {ml_acme_bias:.4f} ({ml_acme_bias_pct:.2f}%)")
        print(f"  95% CI: [{result_ml['acme_ci_lower']:.4f}, {result_ml['acme_ci_upper']:.4f}]")
        print(f"  CI covers truth: {validation_result['ml_enhanced']['ci_covers_truth']}")
        print(f"  VALIDATION: {'PASSED' if ml_passed else 'FAILED'}")
        print()
        print("=" * 60)
        print(f"OVERALL VALIDATION: {'PASSED' if overall_passed else 'FAILED'}")
        print("=" * 60)

    return validation_result


if __name__ == "__main__":
    # Run validation when module is executed directly
    validate_estimator(verbose=True)
