"""
Double/Debiased Machine Learning (DDML) Estimator Implementation.

This module provides comprehensive DDML estimation including:
- Partially Linear Regression (PLR) model
- Interactive Regression Model (IRM) for binary treatment
- Multiple first-stage learner support (Lasso, RF, XGBoost, LightGBM)
- Automatic learner selection via cross-validation
- Sensitivity analysis across ML specifications
- Full DDML workflow with diagnostics

References:
- Chernozhukov et al. (2018). Double/Debiased Machine Learning for
  Treatment and Structural Parameters. The Econometrics Journal.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from scipy import stats

# Import from shared lib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'lib' / 'python'))
from data_loader import CausalInput, CausalOutput
from diagnostics import DiagnosticResult, balance_test
from table_formatter import create_regression_table, create_diagnostic_report


# =============================================================================
# Learner Registry and Utilities
# =============================================================================

def _get_learner(learner_name: str, task: str = 'regression'):
    """
    Get a scikit-learn compatible learner by name.

    Parameters
    ----------
    learner_name : str
        Name of learner: 'lasso', 'ridge', 'elastic_net', 'random_forest',
        'xgboost', 'lightgbm', 'logistic_lasso', 'logistic'
    task : str
        'regression' or 'classification'

    Returns
    -------
    sklearn-compatible estimator
    """
    from sklearn.linear_model import (
        LassoCV, RidgeCV, ElasticNetCV,
        LogisticRegressionCV, LogisticRegression
    )
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    learner_name = learner_name.lower().replace('-', '_').replace(' ', '_')

    # Regression learners
    if task == 'regression':
        if learner_name == 'lasso':
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
        elif learner_name == 'lightgbm' or learner_name == 'lgb':
            try:
                from lightgbm import LGBMRegressor
                return LGBMRegressor(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    n_jobs=-1, random_state=42, verbose=-1
                )
            except ImportError:
                warnings.warn("LightGBM not installed. Falling back to RandomForest.")
                return RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1)
        else:
            raise ValueError(f"Unknown regression learner: {learner_name}")

    # Classification learners
    elif task == 'classification':
        if learner_name in ['lasso', 'logistic_lasso', 'logistic']:
            return LogisticRegressionCV(
                cv=5, penalty='l1', solver='saga', max_iter=10000,
                Cs=np.logspace(-4, 4, 20), random_state=42
            )
        elif learner_name == 'ridge' or learner_name == 'logistic_ridge':
            return LogisticRegressionCV(
                cv=5, penalty='l2', max_iter=10000,
                Cs=np.logspace(-4, 4, 20), random_state=42
            )
        elif learner_name == 'random_forest' or learner_name == 'rf':
            return RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                n_jobs=-1, random_state=42
            )
        elif learner_name == 'xgboost' or learner_name == 'xgb':
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    n_jobs=-1, random_state=42, verbosity=0, use_label_encoder=False
                )
            except ImportError:
                warnings.warn("XGBoost not installed. Falling back to RandomForest.")
                return RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1)
        elif learner_name == 'lightgbm' or learner_name == 'lgb':
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    n_jobs=-1, random_state=42, verbose=-1
                )
            except ImportError:
                warnings.warn("LightGBM not installed. Falling back to RandomForest.")
                return RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1)
        else:
            raise ValueError(f"Unknown classification learner: {learner_name}")

    else:
        raise ValueError(f"Unknown task: {task}. Use 'regression' or 'classification'.")


# =============================================================================
# Data Preparation
# =============================================================================

@dataclass
class DDMLData:
    """Prepared data structure for DDML estimation."""
    y: np.ndarray            # Outcome variable
    d: np.ndarray            # Treatment variable
    X: np.ndarray            # Control variables (high-dimensional)
    feature_names: List[str]  # Names of control variables
    n: int                    # Number of observations
    p: int                    # Number of controls
    treatment_type: str       # 'binary' or 'continuous'

    def __repr__(self):
        return (
            f"DDMLData(n={self.n}, p={self.p}, "
            f"treatment={self.treatment_type})"
        )


def create_ddml_data(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str],
    drop_na: bool = True
) -> DDMLData:
    """
    Prepare data for DDML estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    outcome : str
        Name of outcome variable (Y)
    treatment : str
        Name of treatment variable (D)
    controls : List[str]
        Names of control variables (X)
    drop_na : bool
        Whether to drop rows with missing values

    Returns
    -------
    DDMLData
        Prepared data structure
    """
    df = data.copy()

    # Select columns
    cols = [outcome, treatment] + controls
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
    X = df[controls].values.astype(np.float64)

    # Determine treatment type
    unique_d = np.unique(d)
    if len(unique_d) == 2 and set(unique_d).issubset({0, 1}):
        treatment_type = 'binary'
    else:
        treatment_type = 'continuous'

    return DDMLData(
        y=y,
        d=d,
        X=X,
        feature_names=controls,
        n=len(y),
        p=len(controls),
        treatment_type=treatment_type
    )


def validate_ddml_setup(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str],
    n_folds: int = 5
) -> Dict[str, Any]:
    """
    Validate data setup for DDML estimation.

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
    n_folds : int
        Number of cross-fitting folds

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
    required = [outcome, treatment] + controls
    missing = [c for c in required if c not in data.columns]
    if missing:
        validation['is_valid'] = False
        validation['errors'].append(f"Missing columns: {missing}")
        return validation

    # Sample size
    n = len(data)
    p = len(controls)
    validation['summary']['n'] = n
    validation['summary']['p'] = p
    validation['summary']['n_per_fold'] = n // n_folds

    if n < 100:
        validation['warnings'].append(
            f"Small sample size (n={n}). DDML may be unstable."
        )

    if n // n_folds < 50:
        validation['warnings'].append(
            f"Only {n // n_folds} observations per fold. Consider reducing n_folds."
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
                "Overlap assumption may be violated."
            )
    else:
        validation['summary']['treatment_type'] = 'continuous'
        validation['summary']['treatment_mean'] = d.mean()
        validation['summary']['treatment_std'] = d.std()

    # Outcome variation
    y = data[outcome].dropna()
    validation['summary']['outcome_mean'] = y.mean()
    validation['summary']['outcome_std'] = y.std()

    if y.std() < 1e-10:
        validation['is_valid'] = False
        validation['errors'].append("Outcome has no variation.")

    # High-dimensionality check
    if p > n:
        validation['warnings'].append(
            f"p ({p}) > n ({n}). Ensure using regularized learners."
        )
        validation['summary']['high_dimensional'] = True
    else:
        validation['summary']['high_dimensional'] = False

    return validation


# =============================================================================
# Learner Selection
# =============================================================================

def select_first_stage_learners(
    X: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    cv_folds: int = 5,
    candidate_learners: List[str] = None,
    scoring: str = 'neg_mean_squared_error'
) -> Dict[str, Any]:
    """
    Automatically select best ML models for nuisance estimation.

    Parameters
    ----------
    X : np.ndarray
        Control variables
    y : np.ndarray
        Outcome variable
    d : np.ndarray
        Treatment variable
    cv_folds : int
        Number of cross-validation folds
    candidate_learners : List[str], optional
        List of learner names to consider
    scoring : str
        Scoring metric for cross-validation

    Returns
    -------
    Dict[str, Any]
        Best learners and their CV scores
    """
    from sklearn.model_selection import cross_val_score

    if candidate_learners is None:
        candidate_learners = ['lasso', 'ridge', 'random_forest', 'xgboost']

    results = {
        'ml_l': None,      # Best learner for E[Y|X]
        'ml_m': None,      # Best learner for E[D|X]
        'cv_scores_y': {},
        'cv_scores_d': {}
    }

    # Determine if d is binary for classification
    is_binary_d = len(np.unique(d)) == 2 and set(np.unique(d)).issubset({0, 1})

    # Evaluate learners for Y|X (regression)
    best_score_y = -np.inf
    for learner_name in candidate_learners:
        try:
            learner = _get_learner(learner_name, task='regression')
            scores = cross_val_score(learner, X, y, cv=cv_folds, scoring=scoring)
            mean_score = scores.mean()
            results['cv_scores_y'][learner_name] = {
                'mean': mean_score,
                'std': scores.std()
            }

            if mean_score > best_score_y:
                best_score_y = mean_score
                results['ml_l'] = learner_name
        except Exception as e:
            warnings.warn(f"Learner {learner_name} failed for Y|X: {e}")

    # Evaluate learners for D|X
    best_score_d = -np.inf
    task_d = 'classification' if is_binary_d else 'regression'
    scoring_d = 'accuracy' if is_binary_d else scoring

    for learner_name in candidate_learners:
        try:
            learner = _get_learner(learner_name, task=task_d)
            scores = cross_val_score(learner, X, d, cv=cv_folds, scoring=scoring_d)
            mean_score = scores.mean()
            results['cv_scores_d'][learner_name] = {
                'mean': mean_score,
                'std': scores.std()
            }

            if mean_score > best_score_d:
                best_score_d = mean_score
                results['ml_m'] = learner_name
        except Exception as e:
            warnings.warn(f"Learner {learner_name} failed for D|X: {e}")

    return results


# =============================================================================
# PLR Estimation (Partially Linear Regression)
# =============================================================================

def estimate_plr(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str],
    ml_l: str = 'lasso',
    ml_m: str = 'lasso',
    n_folds: int = 5,
    n_rep: int = 1,
    score: str = 'partialling out',
    random_state: int = 42
) -> CausalOutput:
    """
    Estimate Partially Linear Regression Model using DDML.

    Model: Y = D * theta + g(X) + epsilon
           D = m(X) + V

    Uses cross-fitting and Neyman-orthogonal score.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    outcome : str
        Outcome variable name (Y)
    treatment : str
        Treatment variable name (D)
    controls : List[str]
        Control variable names (X)
    ml_l : str
        Learner for E[Y|X]. Options: 'lasso', 'ridge', 'random_forest', 'xgboost'
    ml_m : str
        Learner for E[D|X]. Options: same as ml_l
    n_folds : int
        Number of cross-fitting folds (default 5)
    n_rep : int
        Number of cross-fitting repetitions (default 1)
    score : str
        Score type: 'partialling out' or 'IV-type'
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    CausalOutput
        Treatment effect estimate with inference
    """
    # Prepare data
    ddml_data = create_ddml_data(data, outcome, treatment, controls)
    y, d, X = ddml_data.y, ddml_data.d, ddml_data.X
    n = ddml_data.n

    np.random.seed(random_state)

    # Storage for multiple repetitions
    theta_estimates = []
    se_estimates = []

    for rep in range(n_rep):
        # Generate fold indices
        fold_indices = np.random.permutation(n) % n_folds

        # Storage for out-of-sample predictions
        l_hat = np.zeros(n)  # E[Y|X] predictions
        m_hat = np.zeros(n)  # E[D|X] predictions

        for k in range(n_folds):
            # Split data
            train_mask = fold_indices != k
            test_mask = fold_indices == k

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, d_train = y[train_mask], d[train_mask]

            # Fit learner for E[Y|X]
            learner_l = _get_learner(ml_l, task='regression')
            learner_l.fit(X_train, y_train)
            l_hat[test_mask] = learner_l.predict(X_test)

            # Fit learner for E[D|X]
            # Check if binary treatment for classification
            is_binary = ddml_data.treatment_type == 'binary'
            if is_binary:
                learner_m = _get_learner(ml_m, task='classification')
                learner_m.fit(X_train, d_train)
                # Get probability of D=1
                m_hat[test_mask] = learner_m.predict_proba(X_test)[:, 1]
            else:
                learner_m = _get_learner(ml_m, task='regression')
                learner_m.fit(X_train, d_train)
                m_hat[test_mask] = learner_m.predict(X_test)

        # Compute residuals
        V = d - m_hat      # Treatment residual
        U = y - l_hat      # Outcome residual (not partialling out D)

        # Partialling out score: theta = E[U*V] / E[V^2]
        if score == 'partialling out':
            theta = np.sum(U * V) / np.sum(V ** 2)

            # Influence function for variance
            psi = (U - theta * V) * V
            var_theta = np.mean(psi ** 2) / (np.mean(V ** 2) ** 2)
            se_theta = np.sqrt(var_theta / n)

        elif score == 'IV-type':
            # IV-type score
            theta = np.mean(V * y) / np.mean(V * d)

            psi = V * (y - theta * d)
            var_theta = np.mean(psi ** 2) / (np.mean(V * d) ** 2)
            se_theta = np.sqrt(var_theta / n)

        else:
            raise ValueError(f"Unknown score type: {score}")

        theta_estimates.append(theta)
        se_estimates.append(se_theta)

    # Aggregate across repetitions
    theta_final = np.mean(theta_estimates)
    # Combine variances accounting for repetition variability
    if n_rep > 1:
        within_var = np.mean([se ** 2 for se in se_estimates])
        between_var = np.var(theta_estimates)
        se_final = np.sqrt(within_var + between_var * (1 + 1/n_rep))
    else:
        se_final = se_estimates[0]

    # Inference
    z_stat = theta_final / se_final
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    ci_lower = theta_final - 1.96 * se_final
    ci_upper = theta_final + 1.96 * se_final

    # Diagnostics
    diagnostics = {
        'method': 'DDML PLR (Partially Linear Regression)',
        'ml_l': ml_l,
        'ml_m': ml_m,
        'n_folds': n_folds,
        'n_rep': n_rep,
        'score': score,
        'n_obs': n,
        'n_controls': ddml_data.p,
        'treatment_type': ddml_data.treatment_type,
        'residual_variance_y': np.var(U),
        'residual_variance_d': np.var(V),
        'r2_y_given_x': 1 - np.var(U) / np.var(y),
        'r2_d_given_x': 1 - np.var(V) / np.var(d)
    }

    if ddml_data.treatment_type == 'binary':
        diagnostics['propensity_summary'] = {
            'min': np.min(m_hat),
            'max': np.max(m_hat),
            'mean': np.mean(m_hat),
            'n_extreme_low': np.sum(m_hat < 0.01),
            'n_extreme_high': np.sum(m_hat > 0.99)
        }

    # Create summary table
    table_results = [{
        'treatment_effect': theta_final,
        'treatment_se': se_final,
        'treatment_pval': p_value,
        'controls': True,
        'fixed_effects': 'ML-adjusted',
        'n_obs': n,
        'r_squared': diagnostics['r2_y_given_x']
    }]

    summary_table = create_regression_table(
        results=table_results,
        column_names=[f"(1) PLR-{ml_l.upper()}"],
        title="Double/Debiased ML Results (PLR Model)",
        notes=f"Cross-fitting with {n_folds} folds. ML: {ml_l} for Y|X, {ml_m} for D|X."
    )

    return CausalOutput(
        effect=theta_final,
        se=se_final,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        diagnostics=diagnostics,
        summary_table=summary_table,
        interpretation=(
            f"The DDML PLR estimate is {theta_final:.4f} (SE = {se_final:.4f}). "
            f"Using {ml_l} for outcome and {ml_m} for treatment nuisance functions. "
            f"Cross-fitting with {n_folds} folds."
        )
    )


# =============================================================================
# IRM Estimation (Interactive Regression Model)
# =============================================================================

def estimate_irm(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str],
    ml_g: str = 'random_forest',
    ml_m: str = 'logistic_lasso',
    n_folds: int = 5,
    n_rep: int = 1,
    trimming_threshold: float = 0.01,
    random_state: int = 42
) -> CausalOutput:
    """
    Estimate Interactive Regression Model using DDML.

    Model: Y = g(D, X) + epsilon
           D ~ Bernoulli(m(X))

    Allows for heterogeneous treatment effects.
    Estimates ATE = E[g(1,X) - g(0,X)].

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    outcome : str
        Outcome variable name (Y)
    treatment : str
        Treatment variable name (D) - MUST be binary (0/1)
    controls : List[str]
        Control variable names (X)
    ml_g : str
        Learner for E[Y|D,X]. Options: 'lasso', 'random_forest', 'xgboost'
    ml_m : str
        Learner for P(D=1|X). Options: 'logistic_lasso', 'random_forest'
    n_folds : int
        Number of cross-fitting folds
    n_rep : int
        Number of cross-fitting repetitions
    trimming_threshold : float
        Trim propensity scores outside [threshold, 1-threshold]
    random_state : int
        Random seed

    Returns
    -------
    CausalOutput
        ATE estimate with inference
    """
    # Prepare data
    ddml_data = create_ddml_data(data, outcome, treatment, controls)
    y, d, X = ddml_data.y, ddml_data.d, ddml_data.X
    n = ddml_data.n

    # Verify binary treatment
    if ddml_data.treatment_type != 'binary':
        raise ValueError(
            "IRM requires binary treatment (0/1). "
            "For continuous treatment, use estimate_plr()."
        )

    np.random.seed(random_state)

    theta_estimates = []
    se_estimates = []
    n_trimmed_total = 0

    for rep in range(n_rep):
        fold_indices = np.random.permutation(n) % n_folds

        # Storage for predictions
        g1_hat = np.zeros(n)  # E[Y|D=1, X]
        g0_hat = np.zeros(n)  # E[Y|D=0, X]
        m_hat = np.zeros(n)   # P(D=1|X)

        for k in range(n_folds):
            train_mask = fold_indices != k
            test_mask = fold_indices == k

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, d_train = y[train_mask], d[train_mask]

            # Fit propensity score model
            learner_m = _get_learner(ml_m, task='classification')
            learner_m.fit(X_train, d_train)
            m_hat[test_mask] = learner_m.predict_proba(X_test)[:, 1]

            # Fit outcome model for treated (D=1)
            treat_train_mask = d_train == 1
            if np.sum(treat_train_mask) > 0:
                learner_g1 = _get_learner(ml_g, task='regression')
                learner_g1.fit(X_train[treat_train_mask], y_train[treat_train_mask])
                g1_hat[test_mask] = learner_g1.predict(X_test)

            # Fit outcome model for control (D=0)
            ctrl_train_mask = d_train == 0
            if np.sum(ctrl_train_mask) > 0:
                learner_g0 = _get_learner(ml_g, task='regression')
                learner_g0.fit(X_train[ctrl_train_mask], y_train[ctrl_train_mask])
                g0_hat[test_mask] = learner_g0.predict(X_test)

        # Trim extreme propensity scores
        trimmed = (m_hat < trimming_threshold) | (m_hat > 1 - trimming_threshold)
        n_trimmed = np.sum(trimmed)
        n_trimmed_total += n_trimmed

        if n_trimmed > 0:
            m_hat = np.clip(m_hat, trimming_threshold, 1 - trimming_threshold)

        # AIPW (Augmented IPW) score for ATE
        # ATE = E[(D/m - (1-D)/(1-m)) * Y - (D-m)/(m(1-m)) * (g1 - g0)]
        # Simplified: use efficient influence function

        # IPW component
        ipw1 = d * y / m_hat
        ipw0 = (1 - d) * y / (1 - m_hat)

        # Augmentation
        aug1 = (d - m_hat) / m_hat * g1_hat
        aug0 = (d - m_hat) / (1 - m_hat) * g0_hat

        # AIPW estimator
        psi = (ipw1 - ipw0) - aug1 + aug0 - (g1_hat - g0_hat)
        theta = np.mean(g1_hat - g0_hat) + np.mean(psi)

        # Simpler: just use the doubly robust score directly
        dr_score = (g1_hat - g0_hat +
                   d * (y - g1_hat) / m_hat -
                   (1 - d) * (y - g0_hat) / (1 - m_hat))

        theta = np.mean(dr_score)
        se = np.std(dr_score) / np.sqrt(n)

        theta_estimates.append(theta)
        se_estimates.append(se)

    # Aggregate
    theta_final = np.mean(theta_estimates)
    if n_rep > 1:
        within_var = np.mean([se ** 2 for se in se_estimates])
        between_var = np.var(theta_estimates)
        se_final = np.sqrt(within_var + between_var * (1 + 1/n_rep))
    else:
        se_final = se_estimates[0]

    # Inference
    z_stat = theta_final / se_final
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    ci_lower = theta_final - 1.96 * se_final
    ci_upper = theta_final + 1.96 * se_final

    # Diagnostics
    diagnostics = {
        'method': 'DDML IRM (Interactive Regression Model)',
        'ml_g': ml_g,
        'ml_m': ml_m,
        'n_folds': n_folds,
        'n_rep': n_rep,
        'trimming_threshold': trimming_threshold,
        'n_trimmed': n_trimmed_total // n_rep,
        'n_obs': n,
        'n_controls': ddml_data.p,
        'propensity_summary': {
            'min': np.min(m_hat),
            'max': np.max(m_hat),
            'mean': np.mean(m_hat)
        },
        'outcome_model_summary': {
            'g1_mean': np.mean(g1_hat),
            'g0_mean': np.mean(g0_hat),
            'g1_g0_diff_mean': np.mean(g1_hat - g0_hat)
        }
    }

    # Summary table
    table_results = [{
        'treatment_effect': theta_final,
        'treatment_se': se_final,
        'treatment_pval': p_value,
        'controls': True,
        'fixed_effects': 'ML-adjusted (AIPW)',
        'n_obs': n,
        'r_squared': np.nan
    }]

    summary_table = create_regression_table(
        results=table_results,
        column_names=[f"(1) IRM-{ml_g.upper()}"],
        title="Double/Debiased ML Results (IRM Model)",
        notes=f"AIPW estimator. {n_folds}-fold cross-fitting. "
              f"ML: {ml_g} for Y|D,X, {ml_m} for propensity."
    )

    return CausalOutput(
        effect=theta_final,
        se=se_final,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        diagnostics=diagnostics,
        summary_table=summary_table,
        interpretation=(
            f"The DDML IRM estimate of ATE is {theta_final:.4f} (SE = {se_final:.4f}). "
            f"Using {ml_g} for outcome model and {ml_m} for propensity. "
            f"{n_trimmed_total // n_rep if n_rep > 0 else 0} observations trimmed due to extreme propensities."
        )
    )


# =============================================================================
# Compare Multiple Learners
# =============================================================================

@dataclass
class LearnerComparisonResult:
    """Results from comparing multiple learner specifications."""
    results: Dict[str, CausalOutput]
    summary_table: str
    sensitivity: Dict[str, Any]


def compare_learners(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str],
    learner_list: List[str] = None,
    model: str = 'plr',
    n_folds: int = 5,
    random_state: int = 42
) -> LearnerComparisonResult:
    """
    Compare DDML estimates across different ML learner specifications.

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
    learner_list : List[str], optional
        List of learner names to compare
    model : str
        'plr' for Partially Linear or 'irm' for Interactive
    n_folds : int
        Number of cross-fitting folds
    random_state : int
        Random seed

    Returns
    -------
    LearnerComparisonResult
        Comparison results across learners
    """
    if learner_list is None:
        learner_list = ['lasso', 'ridge', 'random_forest', 'xgboost']

    results = {}
    table_results = []
    column_names = []

    for i, learner in enumerate(learner_list):
        try:
            if model == 'plr':
                result = estimate_plr(
                    data=data,
                    outcome=outcome,
                    treatment=treatment,
                    controls=controls,
                    ml_l=learner,
                    ml_m=learner,
                    n_folds=n_folds,
                    random_state=random_state + i
                )
            elif model == 'irm':
                # For IRM, use logistic version for propensity
                ml_m = 'logistic_lasso' if learner in ['lasso', 'ridge'] else learner
                result = estimate_irm(
                    data=data,
                    outcome=outcome,
                    treatment=treatment,
                    controls=controls,
                    ml_g=learner,
                    ml_m=ml_m,
                    n_folds=n_folds,
                    random_state=random_state + i
                )
            else:
                raise ValueError(f"Unknown model: {model}")

            results[learner] = result

            table_results.append({
                'treatment_effect': result.effect,
                'treatment_se': result.se,
                'treatment_pval': result.p_value,
                'controls': True,
                'fixed_effects': f'{model.upper()}-{learner}',
                'n_obs': result.diagnostics.get('n_obs', len(data)),
                'r_squared': result.diagnostics.get('r2_y_given_x', np.nan)
            })
            column_names.append(f"({i+1}) {learner.upper()}")

        except Exception as e:
            warnings.warn(f"Learner {learner} failed: {e}")

    # Create comparison table
    summary_table = create_regression_table(
        results=table_results,
        column_names=column_names,
        title=f"DDML {model.upper()} Results: Learner Comparison",
        notes="Each column uses a different ML learner for nuisance functions."
    )

    # Sensitivity analysis
    effects = [r.effect for r in results.values()]
    ses = [r.se for r in results.values()]
    pvals = [r.p_value for r in results.values()]

    sensitivity = {
        'min_effect': min(effects) if effects else np.nan,
        'max_effect': max(effects) if effects else np.nan,
        'effect_range': max(effects) - min(effects) if effects else np.nan,
        'mean_effect': np.mean(effects) if effects else np.nan,
        'std_effect': np.std(effects) if effects else np.nan,
        'all_significant': all(p < 0.05 for p in pvals) if pvals else False,
        'all_same_sign': (all(e > 0 for e in effects) or
                         all(e < 0 for e in effects)) if effects else False
    }

    return LearnerComparisonResult(
        results=results,
        summary_table=summary_table,
        sensitivity=sensitivity
    )


def sensitivity_analysis_ddml(
    results: LearnerComparisonResult,
    ml_specs: List[str] = None
) -> Dict[str, Any]:
    """
    Analyze sensitivity of DDML results to learner choice.

    Parameters
    ----------
    results : LearnerComparisonResult
        Results from compare_learners()
    ml_specs : List[str], optional
        Subset of learners to analyze

    Returns
    -------
    Dict[str, Any]
        Sensitivity analysis summary
    """
    if ml_specs is not None:
        filtered_results = {k: v for k, v in results.results.items() if k in ml_specs}
    else:
        filtered_results = results.results

    effects = [r.effect for r in filtered_results.values()]
    ses = [r.se for r in filtered_results.values()]

    if not effects:
        return {'error': 'No results to analyze'}

    # Meta-analysis style combination
    weights = [1 / (se ** 2) for se in ses if se > 0]
    if weights:
        weighted_mean = sum(e * w for e, w in zip(effects, weights)) / sum(weights)
        weighted_se = np.sqrt(1 / sum(weights))
    else:
        weighted_mean = np.mean(effects)
        weighted_se = np.std(effects) / np.sqrt(len(effects))

    sensitivity = {
        'learners_analyzed': list(filtered_results.keys()),
        'n_learners': len(effects),
        'min_effect': min(effects),
        'max_effect': max(effects),
        'effect_range': max(effects) - min(effects),
        'mean_effect': np.mean(effects),
        'std_effect': np.std(effects),
        'cv_effect': np.std(effects) / abs(np.mean(effects)) if np.mean(effects) != 0 else np.nan,
        'weighted_mean': weighted_mean,
        'weighted_se': weighted_se,
        'all_significant': all(r.p_value < 0.05 for r in filtered_results.values()),
        'all_same_sign': all(e > 0 for e in effects) or all(e < 0 for e in effects),
        'interpretation': ''
    }

    # Generate interpretation
    if sensitivity['cv_effect'] < 0.1:
        sensitivity['interpretation'] = (
            "Results are ROBUST: Effect estimates vary by less than 10% across learners."
        )
    elif sensitivity['cv_effect'] < 0.25:
        sensitivity['interpretation'] = (
            "Results are MODERATELY SENSITIVE: Some variation across learners. "
            "Consider reporting range or using ensemble."
        )
    else:
        sensitivity['interpretation'] = (
            "Results are SENSITIVE to learner choice: Large variation across specifications. "
            "Exercise caution in interpretation."
        )

    return sensitivity


# =============================================================================
# Full DDML Analysis Workflow
# =============================================================================

def run_full_ddml_analysis(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str],
    model: str = 'auto',
    learner_list: List[str] = None,
    n_folds: int = 5,
    n_rep: int = 1,
    auto_select_learner: bool = True,
    random_state: int = 42
) -> CausalOutput:
    """
    Run complete DDML analysis workflow.

    This function:
    1. Validates data structure
    2. Optionally auto-selects best learners
    3. Runs main DDML estimation
    4. Compares multiple learner specifications
    5. Performs sensitivity analysis
    6. Generates comprehensive output

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
    model : str
        'plr', 'irm', or 'auto' (selects based on treatment type)
    learner_list : List[str], optional
        Learners to compare (default: lasso, ridge, rf, xgboost)
    n_folds : int
        Number of cross-fitting folds
    n_rep : int
        Number of cross-fitting repetitions
    auto_select_learner : bool
        Whether to auto-select best learner via CV
    random_state : int
        Random seed

    Returns
    -------
    CausalOutput
        Comprehensive DDML analysis results
    """
    # Step 1: Validate setup
    validation = validate_ddml_setup(data, outcome, treatment, controls, n_folds)

    if not validation['is_valid']:
        raise ValueError(f"Data validation failed: {validation['errors']}")

    if validation['warnings']:
        for w in validation['warnings']:
            warnings.warn(w)

    # Prepare data
    ddml_data = create_ddml_data(data, outcome, treatment, controls)

    # Step 2: Determine model type
    if model == 'auto':
        if ddml_data.treatment_type == 'binary':
            model = 'irm'
        else:
            model = 'plr'

    # Step 3: Auto-select learner if requested
    if auto_select_learner:
        selection = select_first_stage_learners(
            ddml_data.X, ddml_data.y, ddml_data.d
        )
        best_learner = selection['ml_l']
    else:
        best_learner = 'lasso'  # Default

    # Step 4: Run main estimation
    if model == 'plr':
        main_result = estimate_plr(
            data=data,
            outcome=outcome,
            treatment=treatment,
            controls=controls,
            ml_l=best_learner,
            ml_m=best_learner,
            n_folds=n_folds,
            n_rep=n_rep,
            random_state=random_state
        )
    elif model == 'irm':
        ml_m = 'logistic_lasso' if best_learner in ['lasso', 'ridge'] else best_learner
        main_result = estimate_irm(
            data=data,
            outcome=outcome,
            treatment=treatment,
            controls=controls,
            ml_g=best_learner,
            ml_m=ml_m,
            n_folds=n_folds,
            n_rep=n_rep,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    # Step 5: Compare learners
    if learner_list is None:
        learner_list = ['lasso', 'ridge', 'random_forest', 'xgboost']

    try:
        comparison = compare_learners(
            data=data,
            outcome=outcome,
            treatment=treatment,
            controls=controls,
            learner_list=learner_list,
            model=model,
            n_folds=n_folds,
            random_state=random_state
        )
        sensitivity = comparison.sensitivity
    except Exception as e:
        warnings.warn(f"Learner comparison failed: {e}")
        comparison = None
        sensitivity = {}

    # Step 6: Compile comprehensive diagnostics
    all_diagnostics = {
        'validation': validation['summary'],
        'main_estimation': main_result.diagnostics,
        'model_type': model,
        'best_learner': best_learner
    }

    if auto_select_learner:
        all_diagnostics['learner_selection'] = selection

    if comparison:
        all_diagnostics['learner_comparison'] = {
            k: {'effect': v.effect, 'se': v.se, 'p_value': v.p_value}
            for k, v in comparison.results.items()
        }
        all_diagnostics['sensitivity'] = sensitivity

    # Generate comprehensive summary
    summary_lines = [
        "=" * 70,
        "DOUBLE/DEBIASED MACHINE LEARNING ANALYSIS RESULTS",
        "=" * 70,
        "",
        f"Model: {model.upper()} ({'Interactive' if model == 'irm' else 'Partially Linear'})",
        f"Best Learner: {best_learner}",
        f"Sample Size: {ddml_data.n:,}",
        f"Number of Controls: {ddml_data.p}",
        "",
        "-" * 70,
        "MAIN RESULTS",
        "-" * 70,
        "",
        f"Treatment Effect: {main_result.effect:.4f}",
        f"Standard Error: {main_result.se:.4f}",
        f"95% CI: [{main_result.ci_lower:.4f}, {main_result.ci_upper:.4f}]",
        f"P-value: {main_result.p_value:.4f}",
        "",
    ]

    if sensitivity:
        summary_lines.extend([
            "-" * 70,
            "SENSITIVITY TO LEARNER CHOICE",
            "-" * 70,
            "",
            f"Effect Range: [{sensitivity.get('min_effect', np.nan):.4f}, "
            f"{sensitivity.get('max_effect', np.nan):.4f}]",
            f"Coefficient of Variation: {sensitivity.get('cv_effect', np.nan):.2%}",
            f"All Specifications Significant: {sensitivity.get('all_significant', 'N/A')}",
            f"All Same Sign: {sensitivity.get('all_same_sign', 'N/A')}",
            "",
        ])

    if comparison:
        summary_lines.extend([
            "-" * 70,
            "LEARNER COMPARISON",
            "-" * 70,
            "",
            comparison.summary_table,
            ""
        ])

    summary_lines.append("=" * 70)
    comprehensive_summary = "\n".join(summary_lines)

    # Generate interpretation
    interpretation = main_result.generate_interpretation(
        treatment_name=treatment,
        outcome_name=outcome
    )

    if sensitivity and sensitivity.get('interpretation'):
        interpretation += f"\n\nSensitivity: {sensitivity['interpretation']}"

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
# Synthetic Data Generation for Validation
# =============================================================================

def generate_synthetic_ddml_data(
    n: int = 2000,
    p: int = 100,
    treatment_effect: float = 1.0,
    sparsity: int = 10,
    nonlinear: bool = True,
    treatment_type: str = 'binary',
    noise_std: float = 1.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic high-dimensional data for DDML validation.

    Parameters
    ----------
    n : int
        Number of observations
    p : int
        Number of control variables
    treatment_effect : float
        True ATE
    sparsity : int
        Number of truly relevant controls
    nonlinear : bool
        Whether to include nonlinear confounding
    treatment_type : str
        'binary' or 'continuous'
    noise_std : float
        Noise standard deviation
    random_state : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (data, true_params)
    """
    np.random.seed(random_state)

    # Generate high-dimensional controls
    X = np.random.randn(n, p)

    # True coefficients (sparse)
    beta_y = np.zeros(p)
    beta_d = np.zeros(p)

    relevant_vars = np.random.choice(p, sparsity, replace=False)
    beta_y[relevant_vars] = np.random.randn(sparsity)
    beta_d[relevant_vars] = np.random.randn(sparsity) * 0.5

    # Confounding through X
    g_X = X @ beta_y  # Effect on Y
    m_X = X @ beta_d  # Effect on D

    if nonlinear:
        # Add nonlinear terms
        g_X += 0.5 * np.sin(X[:, 0] * 2) + 0.3 * X[:, 1] ** 2
        m_X += 0.3 * np.cos(X[:, 0] * 2) + 0.2 * X[:, 1] ** 2

    # Generate treatment
    if treatment_type == 'binary':
        propensity = 1 / (1 + np.exp(-m_X))
        propensity = np.clip(propensity, 0.1, 0.9)  # Ensure overlap
        d = np.random.binomial(1, propensity, n).astype(float)
    else:
        d = m_X + np.random.randn(n) * 0.5

    # Generate outcome
    y = treatment_effect * d + g_X + np.random.randn(n) * noise_std

    # Create DataFrame
    columns = [f'x{i+1}' for i in range(p)]
    df = pd.DataFrame(X, columns=columns)
    df['d'] = d
    df['y'] = y

    true_params = {
        'true_ate': treatment_effect,
        'n': n,
        'p': p,
        'sparsity': sparsity,
        'nonlinear': nonlinear,
        'treatment_type': treatment_type,
        'noise_std': noise_std,
        'relevant_vars': relevant_vars.tolist(),
        'beta_y': beta_y[relevant_vars].tolist(),
        'beta_d': beta_d[relevant_vars].tolist()
    }

    return df, true_params


# =============================================================================
# Validation
# =============================================================================

def validate_estimator(verbose: bool = True) -> Dict[str, Any]:
    """
    Validate DDML estimator on synthetic data with known treatment effect.

    Tests both PLR and IRM models on high-dimensional synthetic data
    with nonlinear confounding.

    Returns
    -------
    Dict[str, Any]
        Validation results including bias assessment
    """
    # Generate synthetic data
    true_ate = 1.5
    df, true_params = generate_synthetic_ddml_data(
        n=2000,
        p=100,
        treatment_effect=true_ate,
        sparsity=10,
        nonlinear=True,
        treatment_type='binary',
        noise_std=1.0,
        random_state=42
    )

    controls = [c for c in df.columns if c.startswith('x')]

    # Test PLR
    result_plr = estimate_plr(
        data=df,
        outcome='y',
        treatment='d',
        controls=controls,
        ml_l='lasso',
        ml_m='lasso',
        n_folds=5,
        random_state=42
    )

    # Test IRM
    result_irm = estimate_irm(
        data=df,
        outcome='y',
        treatment='d',
        controls=controls,
        ml_g='random_forest',
        ml_m='logistic_lasso',
        n_folds=5,
        random_state=42
    )

    # Calculate bias
    plr_bias = result_plr.effect - true_ate
    plr_bias_pct = abs(plr_bias / true_ate) * 100

    irm_bias = result_irm.effect - true_ate
    irm_bias_pct = abs(irm_bias / true_ate) * 100

    # Validation passes if bias < 15% for both
    plr_passed = plr_bias_pct < 15.0
    irm_passed = irm_bias_pct < 15.0
    overall_passed = plr_passed and irm_passed

    validation_result = {
        'true_ate': true_ate,
        'plr': {
            'estimated_ate': result_plr.effect,
            'se': result_plr.se,
            'bias': plr_bias,
            'bias_pct': plr_bias_pct,
            'passed': plr_passed,
            'ci_covers_truth': result_plr.ci_lower <= true_ate <= result_plr.ci_upper
        },
        'irm': {
            'estimated_ate': result_irm.effect,
            'se': result_irm.se,
            'bias': irm_bias,
            'bias_pct': irm_bias_pct,
            'passed': irm_passed,
            'ci_covers_truth': result_irm.ci_lower <= true_ate <= result_irm.ci_upper
        },
        'overall_passed': overall_passed,
        'data_params': true_params
    }

    if verbose:
        print("=" * 60)
        print("DDML ESTIMATOR VALIDATION")
        print("=" * 60)
        print(f"\nSynthetic Data:")
        print(f"  n = {true_params['n']}, p = {true_params['p']}")
        print(f"  Sparsity = {true_params['sparsity']} relevant variables")
        print(f"  Nonlinear confounding: {true_params['nonlinear']}")
        print(f"  Treatment type: {true_params['treatment_type']}")
        print(f"  True ATE: {true_ate:.4f}")
        print()
        print("-" * 60)
        print("PLR MODEL (Lasso)")
        print("-" * 60)
        print(f"  Estimated ATE: {result_plr.effect:.4f}")
        print(f"  Standard Error: {result_plr.se:.4f}")
        print(f"  Bias: {plr_bias:.4f} ({plr_bias_pct:.2f}%)")
        print(f"  95% CI: [{result_plr.ci_lower:.4f}, {result_plr.ci_upper:.4f}]")
        print(f"  CI covers truth: {validation_result['plr']['ci_covers_truth']}")
        print(f"  VALIDATION: {'PASSED' if plr_passed else 'FAILED'}")
        print()
        print("-" * 60)
        print("IRM MODEL (Random Forest + Logistic Lasso)")
        print("-" * 60)
        print(f"  Estimated ATE: {result_irm.effect:.4f}")
        print(f"  Standard Error: {result_irm.se:.4f}")
        print(f"  Bias: {irm_bias:.4f} ({irm_bias_pct:.2f}%)")
        print(f"  95% CI: [{result_irm.ci_lower:.4f}, {result_irm.ci_upper:.4f}]")
        print(f"  CI covers truth: {validation_result['irm']['ci_covers_truth']}")
        print(f"  VALIDATION: {'PASSED' if irm_passed else 'FAILED'}")
        print()
        print("=" * 60)
        print(f"OVERALL VALIDATION: {'PASSED' if overall_passed else 'FAILED'}")
        print("=" * 60)

    return validation_result


if __name__ == "__main__":
    # Run validation when module is executed directly
    validate_estimator(verbose=True)
