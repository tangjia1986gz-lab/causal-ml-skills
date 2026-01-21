"""
Regularized Linear Models for Causal Inference and Machine Learning

This module provides implementations of Ridge, Lasso, and Elastic Net regression
with cross-validation, along with causal inference methods like double selection.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet,
    RidgeCV, LassoCV, ElasticNetCV,
    LinearRegression
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Optional[np.ndarray] = None,
    cv: int = 5,
    standardize: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Fit Ridge regression with cross-validation for alpha selection.

    Ridge regression adds L2 penalty to OLS, shrinking coefficients toward zero
    without setting them exactly to zero. Useful for multicollinearity and when
    all predictors are believed to be relevant.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target variable.
    alphas : np.ndarray, optional
        Array of alpha values to try. If None, uses default range.
    cv : int, default=5
        Number of cross-validation folds.
    standardize : bool, default=True
        Whether to standardize features before fitting.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - model: fitted RidgeCV model
        - alpha: optimal alpha from CV
        - coefficients: model coefficients
        - cv_scores: cross-validation scores
        - r2_score: R-squared on full data
        - scaler: fitted StandardScaler (if standardize=True)

    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> y = X @ np.random.randn(10) + np.random.randn(100) * 0.5
    >>> result = fit_ridge(X, y)
    >>> print(f"Optimal alpha: {result['alpha']:.4f}")
    """
    if alphas is None:
        alphas = np.logspace(-4, 4, 50)

    # Standardize if requested
    scaler = None
    X_fit = X.copy()
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    # Fit Ridge with CV
    ridge = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_squared_error')
    ridge.fit(X_fit, y)

    # Get CV scores for all alphas
    cv_mse = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X_fit, y, cv=kf, scoring='neg_mean_squared_error')
        cv_mse.append(-scores.mean())

    return {
        'model': ridge,
        'alpha': ridge.alpha_,
        'coefficients': ridge.coef_,
        'intercept': ridge.intercept_,
        'cv_scores': {'alphas': alphas, 'mse': np.array(cv_mse)},
        'r2_score': r2_score(y, ridge.predict(X_fit)),
        'scaler': scaler
    }


def fit_lasso(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Optional[np.ndarray] = None,
    cv: int = 5,
    standardize: bool = True,
    feature_names: Optional[List[str]] = None,
    max_iter: int = 10000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Fit Lasso regression with cross-validation, returning selected variables.

    Lasso uses L1 penalty which sets some coefficients exactly to zero,
    enabling automatic variable selection. Essential for high-dimensional
    settings and when sparsity is expected.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target variable.
    alphas : np.ndarray, optional
        Array of alpha values to try. If None, uses automatic range.
    cv : int, default=5
        Number of cross-validation folds.
    standardize : bool, default=True
        Whether to standardize features before fitting.
    feature_names : list of str, optional
        Names for features. If None, uses X0, X1, ...
    max_iter : int, default=10000
        Maximum iterations for convergence.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - model: fitted LassoCV model
        - alpha: optimal alpha from CV
        - coefficients: model coefficients
        - selected_indices: indices of non-zero coefficients
        - selected_features: names of selected features
        - n_selected: number of selected features
        - cv_scores: cross-validation MSE for each alpha
        - r2_score: R-squared on full data
        - scaler: fitted StandardScaler (if standardize=True)

    Examples
    --------
    >>> X = np.random.randn(100, 50)
    >>> true_coef = np.zeros(50)
    >>> true_coef[:5] = [1, -1, 2, -2, 0.5]
    >>> y = X @ true_coef + np.random.randn(100) * 0.5
    >>> result = fit_lasso(X, y)
    >>> print(f"Selected {result['n_selected']} features")
    """
    if feature_names is None:
        feature_names = [f'X{i}' for i in range(X.shape[1])]

    # Standardize if requested
    scaler = None
    X_fit = X.copy()
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    # Fit Lasso with CV
    lasso = LassoCV(
        alphas=alphas,
        cv=cv,
        max_iter=max_iter,
        random_state=random_state
    )
    lasso.fit(X_fit, y)

    # Extract selected variables
    selected_indices = np.where(lasso.coef_ != 0)[0]
    selected_features = [feature_names[i] for i in selected_indices]

    return {
        'model': lasso,
        'alpha': lasso.alpha_,
        'coefficients': lasso.coef_,
        'intercept': lasso.intercept_,
        'selected_indices': selected_indices,
        'selected_features': selected_features,
        'n_selected': len(selected_indices),
        'cv_scores': {
            'alphas': lasso.alphas_,
            'mse_mean': lasso.mse_path_.mean(axis=1),
            'mse_std': lasso.mse_path_.std(axis=1)
        },
        'r2_score': r2_score(y, lasso.predict(X_fit)),
        'scaler': scaler
    }


def fit_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Optional[np.ndarray] = None,
    l1_ratios: Optional[np.ndarray] = None,
    cv: int = 5,
    standardize: bool = True,
    feature_names: Optional[List[str]] = None,
    max_iter: int = 10000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Fit Elastic Net with cross-validation for alpha and l1_ratio selection.

    Elastic Net combines L1 and L2 penalties, providing both variable selection
    (like Lasso) and grouped selection of correlated variables (like Ridge).
    Controlled by l1_ratio: 1.0 = Lasso, 0.0 = Ridge.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target variable.
    alphas : np.ndarray, optional
        Array of alpha values to try. If None, uses automatic range.
    l1_ratios : np.ndarray, optional
        Array of l1_ratio values to try. If None, uses [0.1, 0.5, 0.7, 0.9, 0.95, 1.0].
    cv : int, default=5
        Number of cross-validation folds.
    standardize : bool, default=True
        Whether to standardize features before fitting.
    feature_names : list of str, optional
        Names for features. If None, uses X0, X1, ...
    max_iter : int, default=10000
        Maximum iterations for convergence.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - model: fitted ElasticNetCV model
        - alpha: optimal alpha from CV
        - l1_ratio: optimal l1_ratio from CV
        - coefficients: model coefficients
        - selected_indices: indices of non-zero coefficients
        - selected_features: names of selected features
        - n_selected: number of selected features
        - r2_score: R-squared on full data
        - scaler: fitted StandardScaler (if standardize=True)

    Examples
    --------
    >>> X = np.random.randn(100, 30)
    >>> # Create groups of correlated features
    >>> X[:, 5:10] = X[:, 0:1] + np.random.randn(100, 5) * 0.1
    >>> y = X[:, 0] + X[:, 10] + np.random.randn(100) * 0.5
    >>> result = fit_elastic_net(X, y)
    >>> print(f"l1_ratio: {result['l1_ratio']:.2f}")
    """
    if feature_names is None:
        feature_names = [f'X{i}' for i in range(X.shape[1])]

    if l1_ratios is None:
        l1_ratios = np.array([0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0])

    # Standardize if requested
    scaler = None
    X_fit = X.copy()
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    # Fit Elastic Net with CV
    enet = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=cv,
        max_iter=max_iter,
        random_state=random_state
    )
    enet.fit(X_fit, y)

    # Extract selected variables
    selected_indices = np.where(enet.coef_ != 0)[0]
    selected_features = [feature_names[i] for i in selected_indices]

    return {
        'model': enet,
        'alpha': enet.alpha_,
        'l1_ratio': enet.l1_ratio_,
        'coefficients': enet.coef_,
        'intercept': enet.intercept_,
        'selected_indices': selected_indices,
        'selected_features': selected_features,
        'n_selected': len(selected_indices),
        'r2_score': r2_score(y, enet.predict(X_fit)),
        'scaler': scaler
    }


def double_selection(
    X: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    alpha_y: Optional[float] = None,
    alpha_d: Optional[float] = None,
    cv: int = 5,
    feature_names: Optional[List[str]] = None,
    max_iter: int = 10000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Belloni, Chernozhukov, Hansen (2014) double selection for causal inference.

    Double selection addresses regularization bias when estimating treatment effects
    with high-dimensional controls. It selects controls predicting both outcome (Y)
    and treatment (D), then runs OLS with the union of selected controls.

    Algorithm:
    1. Lasso of Y on X -> select controls predicting outcome
    2. Lasso of D on X -> select controls predicting treatment
    3. OLS of Y on D and union of selected controls

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_controls)
        Control variables (potential confounders).
    y : np.ndarray of shape (n_samples,)
        Outcome variable.
    d : np.ndarray of shape (n_samples,)
        Treatment variable.
    alpha_y : float, optional
        Lasso penalty for outcome model. If None, uses CV.
    alpha_d : float, optional
        Lasso penalty for treatment model. If None, uses CV.
    cv : int, default=5
        Number of cross-validation folds.
    feature_names : list of str, optional
        Names for control variables.
    max_iter : int, default=10000
        Maximum iterations for Lasso convergence.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - treatment_effect: estimated causal effect of D on Y
        - std_error: heteroskedasticity-robust standard error
        - t_stat: t-statistic
        - p_value: two-sided p-value
        - ci_lower: lower bound of 95% CI
        - ci_upper: upper bound of 95% CI
        - selected_by_y: indices of controls selected in Y ~ X regression
        - selected_by_d: indices of controls selected in D ~ X regression
        - selected_union: union of selected controls
        - n_selected: number of controls in final model
        - ols_model: fitted OLS model (statsmodels)

    References
    ----------
    Belloni, A., Chernozhukov, V., & Hansen, C. (2014). Inference on treatment
    effects after selection among high-dimensional controls. Review of Economic
    Studies, 81(2), 608-650.

    Examples
    --------
    >>> n, p = 500, 100
    >>> X = np.random.randn(n, p)
    >>> d = 0.5 * X[:, 0] + np.random.randn(n)  # Treatment depends on X0
    >>> y = 2 * d + 1.5 * X[:, 0] + np.random.randn(n)  # True effect = 2
    >>> result = double_selection(X, y, d)
    >>> print(f"Treatment effect: {result['treatment_effect']:.3f}")
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels is required for double_selection. "
                          "Install it with: pip install statsmodels")

    if feature_names is None:
        feature_names = [f'X{i}' for i in range(X.shape[1])]

    # Standardize X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 1: Lasso Y ~ X
    if alpha_y is None:
        lasso_y = LassoCV(cv=cv, max_iter=max_iter, random_state=random_state)
    else:
        lasso_y = Lasso(alpha=alpha_y, max_iter=max_iter)
    lasso_y.fit(X_scaled, y)
    selected_by_y = set(np.where(lasso_y.coef_ != 0)[0])

    # Step 2: Lasso D ~ X
    if alpha_d is None:
        lasso_d = LassoCV(cv=cv, max_iter=max_iter, random_state=random_state)
    else:
        lasso_d = Lasso(alpha=alpha_d, max_iter=max_iter)
    lasso_d.fit(X_scaled, d)
    selected_by_d = set(np.where(lasso_d.coef_ != 0)[0])

    # Step 3: Union of selected variables
    selected_union = selected_by_y | selected_by_d

    # Step 4: OLS with selected controls
    if selected_union:
        selected_list = sorted(list(selected_union))
        X_selected = X[:, selected_list]
        X_ols = np.column_stack([d, X_selected])
        control_names = [feature_names[i] for i in selected_list]
    else:
        X_ols = d.reshape(-1, 1)
        control_names = []

    X_ols = sm.add_constant(X_ols)
    ols_model = sm.OLS(y, X_ols).fit(cov_type='HC1')

    # Extract treatment effect (coefficient on D)
    treatment_effect = ols_model.params[1]
    std_error = ols_model.bse[1]
    t_stat = ols_model.tvalues[1]
    p_value = ols_model.pvalues[1]
    ci = ols_model.conf_int()[1]

    return {
        'treatment_effect': treatment_effect,
        'std_error': std_error,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'selected_by_y': selected_by_y,
        'selected_by_y_names': [feature_names[i] for i in selected_by_y],
        'selected_by_d': selected_by_d,
        'selected_by_d_names': [feature_names[i] for i in selected_by_d],
        'selected_union': selected_union,
        'selected_union_names': control_names,
        'n_selected': len(selected_union),
        'ols_model': ols_model,
        'alpha_y': lasso_y.alpha_ if hasattr(lasso_y, 'alpha_') else alpha_y,
        'alpha_d': lasso_d.alpha_ if hasattr(lasso_d, 'alpha_') else alpha_d
    }


def get_feature_importance(
    model: Union[Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV],
    feature_names: Optional[List[str]] = None,
    sort: bool = True
) -> pd.DataFrame:
    """
    Extract and rank feature importance from a fitted regularized model.

    For Lasso/ElasticNet, only non-zero coefficients indicate selected features.
    Importance is measured by absolute coefficient magnitude.

    Parameters
    ----------
    model : sklearn regularized model
        Fitted Ridge, Lasso, or ElasticNet model.
    feature_names : list of str, optional
        Names for features. If None, uses X0, X1, ...
    sort : bool, default=True
        Whether to sort by importance (descending).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, coefficient, importance, selected.

    Examples
    --------
    >>> lasso = LassoCV(cv=5)
    >>> lasso.fit(X_scaled, y)
    >>> importance_df = get_feature_importance(lasso, feature_names)
    >>> print(importance_df.head(10))
    """
    coef = model.coef_
    n_features = len(coef)

    if feature_names is None:
        feature_names = [f'X{i}' for i in range(n_features)]

    df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef,
        'importance': np.abs(coef),
        'selected': coef != 0
    })

    if sort:
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)

    return df


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    standardize: bool = True,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Compare OLS, Ridge, Lasso, and Elastic Net on the same data.

    Useful for understanding the bias-variance tradeoff and comparing
    prediction performance and variable selection across methods.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target variable.
    cv : int, default=5
        Number of cross-validation folds.
    standardize : bool, default=True
        Whether to standardize features.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys 'OLS', 'Ridge', 'Lasso', 'ElasticNet', each containing:
        - model: fitted model
        - cv_rmse: cross-validated RMSE
        - cv_r2: cross-validated R2
        - n_nonzero: number of non-zero coefficients

    Examples
    --------
    >>> comparison = compare_models(X, y)
    >>> for name, result in comparison.items():
    ...     print(f"{name}: RMSE={result['cv_rmse']:.4f}, "
    ...           f"R2={result['cv_r2']:.4f}, n_coef={result['n_nonzero']}")
    """
    # Standardize if requested
    X_fit = X.copy()
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = {}

    # OLS
    ols = LinearRegression()
    ols_mse = -cross_val_score(ols, X_fit, y, cv=kf, scoring='neg_mean_squared_error')
    ols_r2 = cross_val_score(ols, X_fit, y, cv=kf, scoring='r2')
    ols.fit(X_fit, y)
    results['OLS'] = {
        'model': ols,
        'cv_rmse': np.sqrt(ols_mse.mean()),
        'cv_rmse_std': np.sqrt(ols_mse).std(),
        'cv_r2': ols_r2.mean(),
        'cv_r2_std': ols_r2.std(),
        'n_nonzero': np.sum(ols.coef_ != 0)
    }

    # Ridge
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=cv)
    ridge_mse = -cross_val_score(ridge, X_fit, y, cv=kf, scoring='neg_mean_squared_error')
    ridge_r2 = cross_val_score(ridge, X_fit, y, cv=kf, scoring='r2')
    ridge.fit(X_fit, y)
    results['Ridge'] = {
        'model': ridge,
        'alpha': ridge.alpha_,
        'cv_rmse': np.sqrt(ridge_mse.mean()),
        'cv_rmse_std': np.sqrt(ridge_mse).std(),
        'cv_r2': ridge_r2.mean(),
        'cv_r2_std': ridge_r2.std(),
        'n_nonzero': np.sum(ridge.coef_ != 0)
    }

    # Lasso
    lasso = LassoCV(cv=cv, random_state=random_state, max_iter=10000)
    lasso_mse = -cross_val_score(lasso, X_fit, y, cv=kf, scoring='neg_mean_squared_error')
    lasso_r2 = cross_val_score(lasso, X_fit, y, cv=kf, scoring='r2')
    lasso.fit(X_fit, y)
    results['Lasso'] = {
        'model': lasso,
        'alpha': lasso.alpha_,
        'cv_rmse': np.sqrt(lasso_mse.mean()),
        'cv_rmse_std': np.sqrt(lasso_mse).std(),
        'cv_r2': lasso_r2.mean(),
        'cv_r2_std': lasso_r2.std(),
        'n_nonzero': np.sum(lasso.coef_ != 0)
    }

    # Elastic Net
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        cv=cv,
        random_state=random_state,
        max_iter=10000
    )
    enet_mse = -cross_val_score(enet, X_fit, y, cv=kf, scoring='neg_mean_squared_error')
    enet_r2 = cross_val_score(enet, X_fit, y, cv=kf, scoring='r2')
    enet.fit(X_fit, y)
    results['ElasticNet'] = {
        'model': enet,
        'alpha': enet.alpha_,
        'l1_ratio': enet.l1_ratio_,
        'cv_rmse': np.sqrt(enet_mse.mean()),
        'cv_rmse_std': np.sqrt(enet_mse).std(),
        'cv_r2': enet_r2.mean(),
        'cv_r2_std': enet_r2.std(),
        'n_nonzero': np.sum(enet.coef_ != 0)
    }

    return results


def plot_regularization_path(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'lasso',
    feature_names: Optional[List[str]] = None,
    n_alphas: int = 100,
    standardize: bool = True,
    top_k: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> 'matplotlib.figure.Figure':
    """
    Visualize coefficient paths as regularization strength varies.

    Shows how coefficients shrink toward zero as alpha increases,
    helping to understand variable importance and selection stability.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target variable.
    model_type : str, default='lasso'
        Type of regularization: 'lasso', 'ridge', or 'elasticnet'.
    feature_names : list of str, optional
        Names for features. If None, uses X0, X1, ...
    n_alphas : int, default=100
        Number of alpha values to compute.
    standardize : bool, default=True
        Whether to standardize features.
    top_k : int, optional
        Only label the top k features by max coefficient magnitude.
    figsize : tuple, default=(12, 6)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.

    Examples
    --------
    >>> fig = plot_regularization_path(X, y, model_type='lasso')
    >>> plt.show()
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.linear_model import lasso_path, enet_path
    except ImportError:
        raise ImportError("matplotlib is required for plotting. "
                          "Install it with: pip install matplotlib")

    if feature_names is None:
        feature_names = [f'X{i}' for i in range(X.shape[1])]

    # Standardize if requested
    X_fit = X.copy()
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    # Compute regularization path
    if model_type.lower() == 'lasso':
        alphas, coefs, _ = lasso_path(X_fit, y, n_alphas=n_alphas)
        title = 'Lasso Regularization Path'
    elif model_type.lower() == 'ridge':
        # Ridge path computed manually
        alphas = np.logspace(-4, 4, n_alphas)
        coefs = []
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_fit, y)
            coefs.append(ridge.coef_)
        coefs = np.array(coefs).T
        title = 'Ridge Regularization Path'
    elif model_type.lower() == 'elasticnet':
        alphas, coefs, _ = enet_path(X_fit, y, l1_ratio=0.5, n_alphas=n_alphas)
        title = 'Elastic Net Regularization Path (l1_ratio=0.5)'
    else:
        raise ValueError(f"model_type must be 'lasso', 'ridge', or 'elasticnet', got {model_type}")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine which features to label
    max_coef_magnitude = np.abs(coefs).max(axis=1)
    if top_k is not None:
        top_indices = np.argsort(max_coef_magnitude)[-top_k:]
    else:
        top_indices = np.arange(len(feature_names))

    # Plot paths
    for i in range(coefs.shape[0]):
        if i in top_indices:
            ax.plot(np.log10(alphas), coefs[i], label=feature_names[i])
        else:
            ax.plot(np.log10(alphas), coefs[i], color='gray', alpha=0.3)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('log10(alpha)')
    ax.set_ylabel('Coefficient')
    ax.set_title(title)

    if len(top_indices) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig


def cross_validate_regularization(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'lasso',
    alphas: Optional[np.ndarray] = None,
    cv: int = 5,
    standardize: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform cross-validation for regularization parameter selection.

    Returns detailed CV results including mean and standard error of MSE
    across alpha values, useful for the 1-SE rule.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target variable.
    model_type : str, default='lasso'
        Type of model: 'lasso', 'ridge', or 'elasticnet'.
    alphas : np.ndarray, optional
        Alpha values to try.
    cv : int, default=5
        Number of CV folds.
    standardize : bool, default=True
        Whether to standardize features.
    random_state : int, default=42
        Random state.

    Returns
    -------
    dict
        Dictionary with alphas, cv_mean, cv_std, best_alpha, alpha_1se.
    """
    if alphas is None:
        alphas = np.logspace(-4, 2, 100)

    # Standardize
    X_fit = X.copy()
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_results = np.zeros((len(alphas), cv))

    for i, alpha in enumerate(alphas):
        if model_type.lower() == 'lasso':
            model = Lasso(alpha=alpha, max_iter=10000)
        elif model_type.lower() == 'ridge':
            model = Ridge(alpha=alpha)
        elif model_type.lower() == 'elasticnet':
            model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
        else:
            raise ValueError(f"model_type must be 'lasso', 'ridge', or 'elasticnet'")

        for j, (train_idx, val_idx) in enumerate(kf.split(X_fit)):
            model.fit(X_fit[train_idx], y[train_idx])
            y_pred = model.predict(X_fit[val_idx])
            cv_results[i, j] = mean_squared_error(y[val_idx], y_pred)

    cv_mean = cv_results.mean(axis=1)
    cv_std = cv_results.std(axis=1)
    cv_se = cv_std / np.sqrt(cv)

    # Best alpha (minimum CV error)
    best_idx = np.argmin(cv_mean)
    best_alpha = alphas[best_idx]

    # 1-SE rule: largest alpha within 1 SE of minimum
    threshold = cv_mean[best_idx] + cv_se[best_idx]
    valid_idx = np.where(cv_mean <= threshold)[0]
    alpha_1se = alphas[valid_idx[-1]] if len(valid_idx) > 0 else best_alpha

    return {
        'alphas': alphas,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_se': cv_se,
        'best_alpha': best_alpha,
        'best_mse': cv_mean[best_idx],
        'alpha_1se': alpha_1se,
        'mse_1se': cv_mean[valid_idx[-1]] if len(valid_idx) > 0 else cv_mean[best_idx]
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate synthetic data
    n, p = 200, 50
    X = np.random.randn(n, p)

    # Sparse true coefficients
    true_coef = np.zeros(p)
    true_coef[:5] = [2, -1.5, 1, -0.5, 0.8]

    # Add noise
    y = X @ true_coef + np.random.randn(n) * 0.5

    # Fit models
    print("=== Lasso Results ===")
    lasso_result = fit_lasso(X, y)
    print(f"Selected {lasso_result['n_selected']} features")
    print(f"Selected indices: {lasso_result['selected_indices']}")
    print(f"True non-zero indices: [0, 1, 2, 3, 4]")

    print("\n=== Model Comparison ===")
    comparison = compare_models(X, y)
    for name, result in comparison.items():
        print(f"{name}: RMSE={result['cv_rmse']:.4f}, "
              f"R2={result['cv_r2']:.4f}, n_coef={result['n_nonzero']}")

    # Double selection example
    print("\n=== Double Selection Example ===")
    # Create treatment that depends on some controls
    d = 0.5 * X[:, 0] + 0.3 * X[:, 5] + np.random.randn(n) * 0.5
    # Outcome depends on treatment and some controls (true effect = 2)
    y_causal = 2 * d + 1.5 * X[:, 0] + 0.8 * X[:, 5] + np.random.randn(n) * 0.5

    ds_result = double_selection(X, y_causal, d)
    print(f"Treatment effect: {ds_result['treatment_effect']:.3f} (true: 2.0)")
    print(f"95% CI: [{ds_result['ci_lower']:.3f}, {ds_result['ci_upper']:.3f}]")
    print(f"Selected controls: {ds_result['n_selected']}")
