"""
Tree-Based ML Models Module for Prediction and Causal Inference

This module provides comprehensive tools for training, tuning, and interpreting
tree-based machine learning models including decision trees, random forests,
XGBoost, and LightGBM.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, roc_auc_score
import warnings

# Optional imports with graceful fallback
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("xgboost not installed. XGBoost functionality will be unavailable.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("lightgbm not installed. LightGBM functionality will be unavailable.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("shap not installed. SHAP functionality will be unavailable.")


def fit_decision_tree(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    task: str = 'regression',
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    min_samples_split: int = 2,
    max_features: Optional[Union[int, float, str]] = None,
    ccp_alpha: float = 0.0,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Fit a CART decision tree for regression or classification.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Target variable.
    task : str
        'regression' or 'classification'.
    max_depth : int, optional
        Maximum depth of the tree. None for unlimited depth.
    min_samples_leaf : int
        Minimum samples required at a leaf node.
    min_samples_split : int
        Minimum samples required to split an internal node.
    max_features : int, float, or str, optional
        Number of features to consider for best split.
    ccp_alpha : float
        Complexity parameter for minimal cost-complexity pruning.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'model': Fitted decision tree
        - 'feature_importance': Feature importance scores
        - 'depth': Actual tree depth
        - 'n_leaves': Number of leaf nodes
        - 'task': Task type

    Examples
    --------
    >>> tree = fit_decision_tree(X, y, task='regression', max_depth=5)
    >>> predictions = tree['model'].predict(X_test)
    """
    X = np.array(X) if isinstance(X, pd.DataFrame) else X
    y = np.array(y) if isinstance(y, pd.Series) else y

    if task == 'regression':
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            ccp_alpha=ccp_alpha,
            random_state=random_state
        )
    elif task == 'classification':
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            ccp_alpha=ccp_alpha,
            random_state=random_state
        )
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got '{task}'")

    model.fit(X, y)

    return {
        'model': model,
        'feature_importance': model.feature_importances_,
        'depth': model.get_depth(),
        'n_leaves': model.get_n_leaves(),
        'task': task
    }


def fit_random_forest(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    task: str = 'regression',
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    min_samples_split: int = 2,
    max_features: Union[str, int, float] = 'sqrt',
    bootstrap: bool = True,
    oob_score: bool = False,
    class_weight: Optional[Union[str, Dict]] = None,
    n_jobs: int = -1,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit a Random Forest model for regression or classification.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Target variable.
    task : str
        'regression' or 'classification'.
    n_estimators : int
        Number of trees in the forest.
    max_depth : int, optional
        Maximum depth of trees. None for unlimited.
    min_samples_leaf : int
        Minimum samples required at a leaf node.
    min_samples_split : int
        Minimum samples required to split a node.
    max_features : str, int, or float
        Features to consider per split ('sqrt', 'log2', int, or float fraction).
    bootstrap : bool
        Whether to use bootstrap samples.
    oob_score : bool
        Whether to compute out-of-bag score.
    class_weight : str or dict, optional
        Class weights for classification ('balanced', 'balanced_subsample', or dict).
    n_jobs : int
        Number of parallel jobs (-1 for all cores).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'model': Fitted random forest
        - 'feature_importance': Feature importance scores
        - 'oob_score': Out-of-bag score (if oob_score=True)
        - 'task': Task type

    Examples
    --------
    >>> rf = fit_random_forest(X, y, task='regression', n_estimators=200)
    >>> predictions = rf['model'].predict(X_test)
    """
    X = np.array(X) if isinstance(X, pd.DataFrame) else X
    y = np.array(y) if isinstance(y, pd.Series) else y

    if task == 'regression':
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
    elif task == 'classification':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got '{task}'")

    model.fit(X, y)

    result = {
        'model': model,
        'feature_importance': model.feature_importances_,
        'task': task,
        'n_estimators': n_estimators
    }

    if oob_score and bootstrap:
        result['oob_score'] = model.oob_score_

    return result


def fit_xgboost(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    task: str = 'regression',
    params: Optional[Dict[str, Any]] = None,
    n_rounds: int = 100,
    early_stopping_rounds: Optional[int] = None,
    eval_set: Optional[List[Tuple]] = None,
    verbose: bool = False,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Fit an XGBoost model for regression or classification.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Target variable.
    task : str
        'regression' or 'classification'.
    params : dict, optional
        XGBoost parameters. Key parameters:
        - max_depth: Maximum tree depth (default 6)
        - learning_rate: Shrinkage (default 0.3)
        - subsample: Row sampling ratio (default 1)
        - colsample_bytree: Column sampling ratio (default 1)
        - reg_alpha: L1 regularization (default 0)
        - reg_lambda: L2 regularization (default 1)
        - scale_pos_weight: Balance positive/negative (classification)
    n_rounds : int
        Number of boosting rounds.
    early_stopping_rounds : int, optional
        Stop if no improvement for this many rounds.
    eval_set : list of (X, y) tuples, optional
        Validation sets for early stopping.
    verbose : bool
        Whether to print training progress.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'model': Fitted XGBoost model
        - 'feature_importance': Feature importance scores
        - 'best_iteration': Best iteration if early stopping used
        - 'evals_result': Evaluation history if eval_set provided
        - 'task': Task type

    Examples
    --------
    >>> xgb_model = fit_xgboost(
    ...     X, y,
    ...     task='regression',
    ...     params={'max_depth': 5, 'learning_rate': 0.1},
    ...     n_rounds=200,
    ...     early_stopping_rounds=20
    ... )
    """
    if not HAS_XGBOOST:
        raise ImportError(
            "xgboost is not installed. Install it with: pip install xgboost"
        )

    X_arr = np.array(X) if isinstance(X, pd.DataFrame) else X
    y_arr = np.array(y) if isinstance(y, pd.Series) else y

    # Default parameters
    default_params = {
        'max_depth': 6,
        'learning_rate': 0.3,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': random_state,
        'verbosity': 0 if not verbose else 1
    }

    if params is not None:
        default_params.update(params)

    # Set objective based on task
    if task == 'regression':
        default_params['objective'] = default_params.get('objective', 'reg:squarederror')
        model = xgb.XGBRegressor(**default_params, n_estimators=n_rounds)
    elif task == 'classification':
        # Determine binary vs multiclass
        n_classes = len(np.unique(y_arr))
        if n_classes == 2:
            default_params['objective'] = default_params.get('objective', 'binary:logistic')
        else:
            default_params['objective'] = default_params.get('objective', 'multi:softprob')
            default_params['num_class'] = n_classes
        model = xgb.XGBClassifier(**default_params, n_estimators=n_rounds)
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got '{task}'")

    # Prepare fit parameters
    fit_params = {}
    if early_stopping_rounds is not None:
        fit_params['early_stopping_rounds'] = early_stopping_rounds

    if eval_set is not None:
        fit_params['eval_set'] = eval_set
    elif early_stopping_rounds is not None:
        # Create validation set from training data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_arr, y_arr, test_size=0.2, random_state=random_state
        )
        fit_params['eval_set'] = [(X_val, y_val)]
        X_arr, y_arr = X_train, y_train

    fit_params['verbose'] = verbose

    model.fit(X_arr, y_arr, **fit_params)

    result = {
        'model': model,
        'feature_importance': model.feature_importances_,
        'task': task,
        'params': default_params
    }

    if early_stopping_rounds is not None:
        result['best_iteration'] = model.best_iteration

    return result


def fit_lightgbm(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    task: str = 'regression',
    params: Optional[Dict[str, Any]] = None,
    n_rounds: int = 100,
    early_stopping_rounds: Optional[int] = None,
    eval_set: Optional[List[Tuple]] = None,
    categorical_features: Optional[List[str]] = None,
    verbose: int = -1,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Fit a LightGBM model for regression or classification.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Target variable.
    task : str
        'regression' or 'classification'.
    params : dict, optional
        LightGBM parameters. Key parameters:
        - num_leaves: Max number of leaves per tree (default 31)
        - max_depth: Max tree depth (-1 for no limit)
        - learning_rate: Shrinkage (default 0.1)
        - feature_fraction: Column sampling ratio (default 1)
        - bagging_fraction: Row sampling ratio (default 1)
        - bagging_freq: Frequency for bagging (0 to disable)
        - lambda_l1: L1 regularization (default 0)
        - lambda_l2: L2 regularization (default 0)
    n_rounds : int
        Number of boosting rounds.
    early_stopping_rounds : int, optional
        Stop if no improvement for this many rounds.
    eval_set : list of (X, y) tuples, optional
        Validation sets for early stopping.
    categorical_features : list of str, optional
        Names of categorical feature columns.
    verbose : int
        Verbosity level (-1 for silent).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'model': Fitted LightGBM model
        - 'feature_importance': Feature importance scores
        - 'best_iteration': Best iteration if early stopping used
        - 'task': Task type

    Examples
    --------
    >>> lgb_model = fit_lightgbm(
    ...     X, y,
    ...     task='regression',
    ...     params={'num_leaves': 31, 'learning_rate': 0.1},
    ...     n_rounds=200,
    ...     early_stopping_rounds=20
    ... )
    """
    if not HAS_LIGHTGBM:
        raise ImportError(
            "lightgbm is not installed. Install it with: pip install lightgbm"
        )

    X_arr = np.array(X) if isinstance(X, pd.DataFrame) else X
    y_arr = np.array(y) if isinstance(y, pd.Series) else y

    # Default parameters
    default_params = {
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'lambda_l1': 0,
        'lambda_l2': 0,
        'random_state': random_state,
        'verbose': verbose,
        'force_col_wise': True  # Suppress warning about data size
    }

    if params is not None:
        default_params.update(params)

    # Set objective based on task
    if task == 'regression':
        default_params['objective'] = default_params.get('objective', 'regression')
        model = lgb.LGBMRegressor(**default_params, n_estimators=n_rounds)
    elif task == 'classification':
        n_classes = len(np.unique(y_arr))
        if n_classes == 2:
            default_params['objective'] = default_params.get('objective', 'binary')
        else:
            default_params['objective'] = default_params.get('objective', 'multiclass')
            default_params['num_class'] = n_classes
        model = lgb.LGBMClassifier(**default_params, n_estimators=n_rounds)
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got '{task}'")

    # Prepare fit parameters
    fit_params = {}

    # Handle categorical features
    if categorical_features is not None and isinstance(X, pd.DataFrame):
        cat_indices = [X.columns.get_loc(c) for c in categorical_features if c in X.columns]
        if cat_indices:
            fit_params['categorical_feature'] = cat_indices

    # Set up callbacks for early stopping
    callbacks = []
    if early_stopping_rounds is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
    if verbose >= 0:
        callbacks.append(lgb.log_evaluation(period=verbose if verbose > 0 else 100))

    if eval_set is not None:
        fit_params['eval_set'] = eval_set
    elif early_stopping_rounds is not None:
        # Create validation set from training data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_arr, y_arr, test_size=0.2, random_state=random_state
        )
        fit_params['eval_set'] = [(X_val, y_val)]
        X_arr, y_arr = X_train, y_train

    if callbacks:
        fit_params['callbacks'] = callbacks

    model.fit(X_arr, y_arr, **fit_params)

    result = {
        'model': model,
        'feature_importance': model.feature_importances_,
        'task': task,
        'params': default_params
    }

    if early_stopping_rounds is not None and hasattr(model, 'best_iteration_'):
        result['best_iteration'] = model.best_iteration_

    return result


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    method: str = 'impurity',
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Extract feature importance from a tree-based model.

    Parameters
    ----------
    model : fitted tree model
        Fitted sklearn, XGBoost, or LightGBM model.
    feature_names : list of str
        Names of features.
    method : str
        Importance method:
        - 'impurity': Impurity-based (mean decrease in impurity)
        - 'permutation': Permutation importance (requires X, y)
    X : array-like, optional
        Features for permutation importance.
    y : array-like, optional
        Target for permutation importance.
    n_repeats : int
        Number of permutation repeats.
    random_state : int
        Random seed for permutation.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature importance, sorted by importance.
        Columns: 'feature', 'importance', and for permutation also 'std'.

    Examples
    --------
    >>> importance = get_feature_importance(rf['model'], feature_names, method='impurity')
    >>> print(importance.head(10))

    >>> # Permutation importance (more reliable)
    >>> importance = get_feature_importance(
    ...     rf['model'], feature_names, method='permutation', X=X, y=y
    ... )
    """
    # Handle model wrapper dictionaries
    if isinstance(model, dict) and 'model' in model:
        model = model['model']

    if method == 'impurity':
        # Impurity-based importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise ValueError("Model does not have feature_importances_ attribute")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    elif method == 'permutation':
        if X is None or y is None:
            raise ValueError("X and y are required for permutation importance")

        X_arr = np.array(X) if isinstance(X, pd.DataFrame) else X
        y_arr = np.array(y) if isinstance(y, pd.Series) else y

        perm_importance = permutation_importance(
            model, X_arr, y_arr,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    else:
        raise ValueError(f"method must be 'impurity' or 'permutation', got '{method}'")

    return importance_df


def compute_shap_values(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    plot_summary: bool = True,
    plot_bar: bool = True,
    max_display: int = 20
) -> Dict[str, Any]:
    """
    Compute SHAP values for model interpretation.

    Parameters
    ----------
    model : fitted tree model
        Fitted sklearn, XGBoost, or LightGBM model.
    X : array-like of shape (n_samples, n_features)
        Data to explain.
    feature_names : list of str, optional
        Feature names for plotting.
    plot_summary : bool
        Whether to display SHAP summary plot.
    plot_bar : bool
        Whether to display SHAP bar plot.
    max_display : int
        Maximum features to display in plots.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'shap_values': SHAP values array
        - 'expected_value': Base value (expected prediction)
        - 'feature_importance': Mean absolute SHAP values per feature
        - 'explainer': SHAP explainer object

    Examples
    --------
    >>> shap_result = compute_shap_values(rf['model'], X)
    >>> print(shap_result['feature_importance'].head(10))
    """
    if not HAS_SHAP:
        raise ImportError(
            "shap is not installed. Install it with: pip install shap"
        )

    # Handle model wrapper dictionaries
    if isinstance(model, dict) and 'model' in model:
        model = model['model']

    X_arr = np.array(X) if isinstance(X, pd.DataFrame) else X

    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    elif feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_arr.shape[1])]

    # Create SHAP explainer (TreeExplainer for tree models)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        # Fall back to generic explainer
        explainer = shap.Explainer(model, X_arr)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_arr)

    # Handle multi-output (classification with multiple classes)
    if isinstance(shap_values, list):
        # For binary classification, use class 1
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            # For multiclass, keep all
            pass

    # Compute feature importance from SHAP values
    if isinstance(shap_values, list):
        # Multiclass: average across classes
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    result = {
        'shap_values': shap_values,
        'expected_value': explainer.expected_value,
        'feature_importance': feature_importance,
        'explainer': explainer
    }

    # Generate plots
    if plot_summary:
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            shap.summary_plot(
                shap_values, X_arr,
                feature_names=feature_names,
                max_display=max_display,
                show=True
            )
            result['summary_plot'] = 'displayed'
        except Exception as e:
            result['summary_plot'] = f'error: {str(e)}'

    if plot_bar:
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            shap.summary_plot(
                shap_values, X_arr,
                feature_names=feature_names,
                max_display=max_display,
                plot_type='bar',
                show=True
            )
            result['bar_plot'] = 'displayed'
        except Exception as e:
            result['bar_plot'] = f'error: {str(e)}'

    return result


def partial_dependence_plot(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    features: Union[List[str], List[int]],
    feature_names: Optional[List[str]] = None,
    grid_resolution: int = 100,
    percentiles: Tuple[float, float] = (0.05, 0.95)
) -> Dict[str, Any]:
    """
    Generate Partial Dependence Plots (PDP) for specified features.

    Parameters
    ----------
    model : fitted tree model
        Fitted sklearn, XGBoost, or LightGBM model.
    X : array-like of shape (n_samples, n_features)
        Data used for computing partial dependence.
    features : list of str or int
        Features to plot (names if X is DataFrame, indices otherwise).
    feature_names : list of str, optional
        Feature names if X is array.
    grid_resolution : int
        Number of points in the grid for PDP computation.
    percentiles : tuple
        Percentile range for the grid.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'pdp_values': Partial dependence values
        - 'grid_values': Grid points for each feature
        - 'feature_names': Names of plotted features

    Examples
    --------
    >>> pdp_result = partial_dependence_plot(model, X, features=['income', 'age'])
    """
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay

    # Handle model wrapper dictionaries
    if isinstance(model, dict) and 'model' in model:
        model = model['model']

    X_arr = np.array(X) if isinstance(X, pd.DataFrame) else X

    if isinstance(X, pd.DataFrame) and all(isinstance(f, str) for f in features):
        feature_indices = [X.columns.get_loc(f) for f in features]
        feature_names_used = features
    else:
        feature_indices = features
        if feature_names is not None:
            feature_names_used = [feature_names[i] for i in features]
        else:
            feature_names_used = [f'feature_{i}' for i in features]

    # Compute partial dependence
    pdp_results = partial_dependence(
        model, X_arr, feature_indices,
        grid_resolution=grid_resolution,
        percentiles=percentiles
    )

    result = {
        'pdp_values': pdp_results['average'],
        'grid_values': pdp_results['grid_values'],
        'feature_names': feature_names_used
    }

    # Generate plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(features), figsize=(5*len(features), 4))
        if len(features) == 1:
            axes = [axes]

        PartialDependenceDisplay.from_estimator(
            model, X_arr, feature_indices,
            feature_names=feature_names if feature_names else None,
            grid_resolution=grid_resolution,
            percentiles=percentiles,
            ax=axes
        )
        plt.tight_layout()
        plt.show()
        result['plot'] = 'displayed'
    except Exception as e:
        result['plot'] = f'error: {str(e)}'

    return result


def tune_hyperparameters(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    model_type: str,
    param_grid: Dict[str, List[Any]],
    task: str = 'regression',
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Tune hyperparameters for tree models using cross-validation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Target variable.
    model_type : str
        Model type: 'decision_tree', 'random_forest', 'xgboost', 'lightgbm'.
    param_grid : dict
        Dictionary of parameter names to lists of values to try.
    task : str
        'regression' or 'classification'.
    cv : int
        Number of cross-validation folds.
    scoring : str, optional
        Scoring metric. Defaults to 'neg_mean_squared_error' for regression,
        'accuracy' for classification.
    n_jobs : int
        Number of parallel jobs.
    random_state : int
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'best_params': Best hyperparameters found
        - 'best_score': Best cross-validation score
        - 'cv_results': Full cross-validation results
        - 'best_model': Model fitted with best parameters

    Examples
    --------
    >>> param_grid = {
    ...     'max_depth': [3, 5, 7],
    ...     'learning_rate': [0.01, 0.1, 0.3],
    ...     'subsample': [0.7, 0.8, 0.9]
    ... }
    >>> result = tune_hyperparameters(
    ...     X, y, model_type='xgboost', param_grid=param_grid
    ... )
    >>> print(f"Best params: {result['best_params']}")
    """
    X_arr = np.array(X) if isinstance(X, pd.DataFrame) else X
    y_arr = np.array(y) if isinstance(y, pd.Series) else y

    # Set default scoring
    if scoring is None:
        scoring = 'neg_mean_squared_error' if task == 'regression' else 'accuracy'

    # Create base model
    if model_type == 'decision_tree':
        if task == 'regression':
            base_model = DecisionTreeRegressor(random_state=random_state)
        else:
            base_model = DecisionTreeClassifier(random_state=random_state)

    elif model_type == 'random_forest':
        if task == 'regression':
            base_model = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
        else:
            base_model = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)

    elif model_type == 'xgboost':
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed")
        if task == 'regression':
            base_model = xgb.XGBRegressor(random_state=random_state, verbosity=0)
        else:
            base_model = xgb.XGBClassifier(random_state=random_state, verbosity=0)

    elif model_type == 'lightgbm':
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed")
        if task == 'regression':
            base_model = lgb.LGBMRegressor(random_state=random_state, verbose=-1)
        else:
            base_model = lgb.LGBMClassifier(random_state=random_state, verbose=-1)

    else:
        raise ValueError(
            f"model_type must be 'decision_tree', 'random_forest', 'xgboost', or 'lightgbm', "
            f"got '{model_type}'"
        )

    # Run grid search
    grid_search = GridSearchCV(
        base_model, param_grid,
        cv=cv, scoring=scoring,
        n_jobs=n_jobs, refit=True
    )
    grid_search.fit(X_arr, y_arr)

    # Prepare results
    cv_results = pd.DataFrame(grid_search.cv_results_)

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': cv_results,
        'best_model': grid_search.best_estimator_
    }


def compare_tree_models(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    task: str = 'regression',
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Compare performance of different tree-based models.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Target variable.
    task : str
        'regression' or 'classification'.
    cv : int
        Number of cross-validation folds.
    random_state : int
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'summary': DataFrame with model comparison
        - 'models': Dict of fitted models
        - 'cv_scores': Dict of cross-validation scores per model

    Examples
    --------
    >>> comparison = compare_tree_models(X, y, task='regression')
    >>> print(comparison['summary'])
    """
    X_arr = np.array(X) if isinstance(X, pd.DataFrame) else X
    y_arr = np.array(y) if isinstance(y, pd.Series) else y

    # Define models
    models = {}

    if task == 'regression':
        scoring = 'neg_mean_squared_error'
        models['Decision Tree'] = DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=5, random_state=random_state
        )
        models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=None, random_state=random_state, n_jobs=-1
        )
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=random_state, verbosity=0
            )
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100, num_leaves=31, learning_rate=0.1,
                random_state=random_state, verbose=-1
            )
    else:
        scoring = 'accuracy'
        models['Decision Tree'] = DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=5, random_state=random_state
        )
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=random_state, n_jobs=-1
        )
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=random_state, verbosity=0
            )
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100, num_leaves=31, learning_rate=0.1,
                random_state=random_state, verbose=-1
            )

    # Evaluate each model
    results = []
    cv_scores = {}
    fitted_models = {}

    for name, model in models.items():
        # Cross-validation
        scores = cross_val_score(model, X_arr, y_arr, cv=cv, scoring=scoring, n_jobs=-1)
        cv_scores[name] = scores

        # Fit on full data
        model.fit(X_arr, y_arr)
        fitted_models[name] = model

        # Compute metrics
        if task == 'regression':
            train_pred = model.predict(X_arr)
            train_rmse = np.sqrt(mean_squared_error(y_arr, train_pred))
            train_r2 = r2_score(y_arr, train_pred)
            cv_rmse = np.sqrt(-scores.mean())
            cv_rmse_std = np.sqrt(-scores).std()

            results.append({
                'Model': name,
                'Train RMSE': train_rmse,
                'Train R2': train_r2,
                'CV RMSE (mean)': cv_rmse,
                'CV RMSE (std)': cv_rmse_std
            })
        else:
            train_pred = model.predict(X_arr)
            train_acc = accuracy_score(y_arr, train_pred)
            cv_acc = scores.mean()
            cv_acc_std = scores.std()

            results.append({
                'Model': name,
                'Train Accuracy': train_acc,
                'CV Accuracy (mean)': cv_acc,
                'CV Accuracy (std)': cv_acc_std
            })

    summary = pd.DataFrame(results)

    # Sort by CV performance
    if task == 'regression':
        summary = summary.sort_values('CV RMSE (mean)')
    else:
        summary = summary.sort_values('CV Accuracy (mean)', ascending=False)

    # Generate comparison plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        if task == 'regression':
            metric_col = 'CV RMSE (mean)'
            std_col = 'CV RMSE (std)'
            ylabel = 'RMSE (Cross-Validation)'
        else:
            metric_col = 'CV Accuracy (mean)'
            std_col = 'CV Accuracy (std)'
            ylabel = 'Accuracy (Cross-Validation)'

        x_pos = np.arange(len(summary))
        bars = ax.bar(x_pos, summary[metric_col], yerr=summary[std_col], capsize=5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary['Model'], rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title('Tree Model Comparison')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

        result_plot = 'displayed'
    except Exception as e:
        result_plot = f'error: {str(e)}'

    return {
        'summary': summary,
        'models': fitted_models,
        'cv_scores': cv_scores,
        'plot': result_plot
    }


# Convenience functions

def quick_tree_model(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    task: str = 'regression',
    model_type: str = 'random_forest'
) -> Dict[str, Any]:
    """
    Quick model fitting with sensible defaults.

    Parameters
    ----------
    X : array-like
        Features.
    y : array-like
        Target.
    task : str
        'regression' or 'classification'.
    model_type : str
        'decision_tree', 'random_forest', 'xgboost', or 'lightgbm'.

    Returns
    -------
    Dict[str, Any]
        Fitted model result.
    """
    if model_type == 'decision_tree':
        return fit_decision_tree(X, y, task=task, max_depth=10, min_samples_leaf=5)
    elif model_type == 'random_forest':
        return fit_random_forest(X, y, task=task, n_estimators=100)
    elif model_type == 'xgboost':
        return fit_xgboost(X, y, task=task, n_rounds=100)
    elif model_type == 'lightgbm':
        return fit_lightgbm(X, y, task=task, n_rounds=100)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_propensity_scores(
    X: Union[pd.DataFrame, np.ndarray],
    treatment: Union[pd.Series, np.ndarray],
    model_type: str = 'random_forest',
    **kwargs
) -> np.ndarray:
    """
    Estimate propensity scores using tree models.

    Parameters
    ----------
    X : array-like
        Covariates.
    treatment : array-like
        Binary treatment indicator.
    model_type : str
        Model type for propensity estimation.
    **kwargs
        Additional arguments passed to the model.

    Returns
    -------
    np.ndarray
        Estimated propensity scores.

    Examples
    --------
    >>> ps = get_propensity_scores(X, D, model_type='xgboost')
    >>> print(f"PS range: [{ps.min():.3f}, {ps.max():.3f}]")
    """
    X_arr = np.array(X) if isinstance(X, pd.DataFrame) else X
    t_arr = np.array(treatment) if isinstance(treatment, pd.Series) else treatment

    if model_type == 'decision_tree':
        result = fit_decision_tree(X_arr, t_arr, task='classification', **kwargs)
    elif model_type == 'random_forest':
        result = fit_random_forest(X_arr, t_arr, task='classification', **kwargs)
    elif model_type == 'xgboost':
        result = fit_xgboost(X_arr, t_arr, task='classification', **kwargs)
    elif model_type == 'lightgbm':
        result = fit_lightgbm(X_arr, t_arr, task='classification', **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = result['model']
    propensity_scores = model.predict_proba(X_arr)[:, 1]

    return propensity_scores
