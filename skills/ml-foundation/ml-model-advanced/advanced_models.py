"""
Advanced Machine Learning Models: SVM and Neural Networks

This module provides implementations of Support Vector Machines and Multi-Layer
Perceptrons with proper standardization pipelines, hyperparameter tuning, and
comparison utilities.

IMPORTANT: Both SVM and MLP require feature standardization. All functions in
this module return pipelines with StandardScaler built-in.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, Literal, Dict, Any, List, Tuple

from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.calibration import CalibratedClassifierCV


# =============================================================================
# Support Vector Machine Functions
# =============================================================================

def fit_svc(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Literal['linear', 'rbf', 'poly', 'sigmoid'] = 'rbf',
    C: float = 1.0,
    gamma: Union[str, float] = 'scale',
    degree: int = 3,
    probability: bool = True,
    random_state: int = 42
) -> Pipeline:
    """
    Fit a Support Vector Classifier with built-in scaling.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,).
    kernel : str
        Kernel type: 'linear', 'rbf', 'poly', or 'sigmoid'. Default 'rbf'.
    C : float
        Regularization parameter. Larger C = stricter margin. Default 1.0.
    gamma : str or float
        Kernel coefficient. 'scale' = 1/(n_features * X.var()),
        'auto' = 1/n_features, or float value. Default 'scale'.
    degree : int
        Degree for polynomial kernel. Default 3.
    probability : bool
        Whether to enable probability estimates. Default True.
        Required for propensity score estimation.
    random_state : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    Pipeline
        Fitted pipeline with StandardScaler and SVC.

    Example
    -------
    >>> svc_model = fit_svc(X_train, y_train, kernel='rbf', C=10)
    >>> predictions = svc_model.predict(X_test)
    >>> probabilities = svc_model.predict_proba(X_test)[:, 1]
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            probability=probability,
            random_state=random_state
        ))
    ])

    pipeline.fit(X, y)
    return pipeline


def fit_svr(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Literal['linear', 'rbf', 'poly', 'sigmoid'] = 'rbf',
    C: float = 1.0,
    gamma: Union[str, float] = 'scale',
    epsilon: float = 0.1,
    degree: int = 3
) -> Pipeline:
    """
    Fit a Support Vector Regressor with built-in scaling.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    kernel : str
        Kernel type: 'linear', 'rbf', 'poly', or 'sigmoid'. Default 'rbf'.
    C : float
        Regularization parameter. Default 1.0.
    gamma : str or float
        Kernel coefficient. Default 'scale'.
    epsilon : float
        Epsilon in the epsilon-SVR model. Specifies the tube within
        which no penalty is associated. Default 0.1.
    degree : int
        Degree for polynomial kernel. Default 3.

    Returns
    -------
    Pipeline
        Fitted pipeline with StandardScaler and SVR.

    Example
    -------
    >>> svr_model = fit_svr(X_train, y_train, kernel='rbf', C=10, epsilon=0.05)
    >>> predictions = svr_model.predict(X_test)
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(
            kernel=kernel,
            C=C,
            gamma=gamma,
            epsilon=epsilon,
            degree=degree
        ))
    ])

    pipeline.fit(X, y)
    return pipeline


# =============================================================================
# Neural Network (MLP) Functions
# =============================================================================

def fit_mlp_classifier(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layers: Tuple[int, ...] = (100, 50),
    activation: Literal['relu', 'tanh', 'logistic'] = 'relu',
    learning_rate: float = 0.001,
    alpha: float = 0.0001,
    solver: Literal['adam', 'sgd', 'lbfgs'] = 'adam',
    max_iter: int = 500,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    random_state: int = 42
) -> Pipeline:
    """
    Fit a Multi-Layer Perceptron Classifier with built-in scaling.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,).
    hidden_layers : tuple of int
        Number of neurons in each hidden layer. Default (100, 50).
    activation : str
        Activation function: 'relu', 'tanh', or 'logistic'. Default 'relu'.
    learning_rate : float
        Initial learning rate for adam/sgd. Default 0.001.
    alpha : float
        L2 regularization term. Default 0.0001.
    solver : str
        Optimization algorithm: 'adam', 'sgd', or 'lbfgs'. Default 'adam'.
    max_iter : int
        Maximum number of iterations. Default 500.
    early_stopping : bool
        Whether to use early stopping. Default True.
    validation_fraction : float
        Fraction of training data for validation when early_stopping=True.
        Default 0.1.
    random_state : int
        Random seed. Default 42.

    Returns
    -------
    Pipeline
        Fitted pipeline with StandardScaler and MLPClassifier.

    Example
    -------
    >>> mlp = fit_mlp_classifier(X_train, y_train, hidden_layers=(128, 64, 32))
    >>> predictions = mlp.predict(X_test)
    >>> probabilities = mlp.predict_proba(X_test)[:, 1]
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_state
        ))
    ])

    pipeline.fit(X, y)
    return pipeline


def fit_mlp_regressor(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layers: Tuple[int, ...] = (100, 50),
    activation: Literal['relu', 'tanh', 'logistic', 'identity'] = 'relu',
    learning_rate: float = 0.001,
    alpha: float = 0.0001,
    solver: Literal['adam', 'sgd', 'lbfgs'] = 'adam',
    max_iter: int = 500,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    random_state: int = 42
) -> Pipeline:
    """
    Fit a Multi-Layer Perceptron Regressor with built-in scaling.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    hidden_layers : tuple of int
        Number of neurons in each hidden layer. Default (100, 50).
    activation : str
        Activation function. Default 'relu'.
    learning_rate : float
        Initial learning rate. Default 0.001.
    alpha : float
        L2 regularization term. Default 0.0001.
    solver : str
        Optimization algorithm. Default 'adam'.
    max_iter : int
        Maximum iterations. Default 500.
    early_stopping : bool
        Whether to use early stopping. Default True.
    validation_fraction : float
        Validation fraction for early stopping. Default 0.1.
    random_state : int
        Random seed. Default 42.

    Returns
    -------
    Pipeline
        Fitted pipeline with StandardScaler and MLPRegressor.

    Example
    -------
    >>> mlp = fit_mlp_regressor(X_train, y_train, hidden_layers=(64, 32))
    >>> predictions = mlp.predict(X_test)
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_state
        ))
    ])

    pipeline.fit(X, y)
    return pipeline


# =============================================================================
# Hyperparameter Tuning Functions
# =============================================================================

def tune_svm(
    X: np.ndarray,
    y: np.ndarray,
    task: Literal['classification', 'regression'] = 'classification',
    param_grid: Optional[Dict[str, List[Any]]] = None,
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Tune SVM hyperparameters using GridSearchCV.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target array.
    task : str
        'classification' or 'regression'. Default 'classification'.
    param_grid : dict, optional
        Parameter grid. If None, uses default grid.
    cv : int
        Number of cross-validation folds. Default 5.
    scoring : str, optional
        Scoring metric. Default None (uses accuracy/r2).
    n_jobs : int
        Number of parallel jobs. Default -1 (all cores).

    Returns
    -------
    dict
        Dictionary with 'best_params', 'best_score', 'best_model',
        'cv_results'.

    Example
    -------
    >>> results = tune_svm(X_train, y_train, task='classification')
    >>> best_model = results['best_model']
    >>> print(f"Best params: {results['best_params']}")
    """
    if task == 'classification':
        estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=42))
        ])
        default_scoring = 'accuracy'
        prefix = 'svm'
    else:
        estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVR())
        ])
        default_scoring = 'r2'
        prefix = 'svm'

    if param_grid is None:
        param_grid = {
            f'{prefix}__C': [0.1, 1, 10, 100],
            f'{prefix}__gamma': ['scale', 'auto', 0.01, 0.1, 1],
            f'{prefix}__kernel': ['rbf', 'linear', 'poly']
        }

    scoring = scoring or default_scoring

    grid_search = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=True
    )

    grid_search.fit(X, y)

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_,
        'cv_results': pd.DataFrame(grid_search.cv_results_)
    }


def tune_mlp(
    X: np.ndarray,
    y: np.ndarray,
    task: Literal['classification', 'regression'] = 'classification',
    param_grid: Optional[Dict[str, List[Any]]] = None,
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Tune MLP hyperparameters using GridSearchCV.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target array.
    task : str
        'classification' or 'regression'. Default 'classification'.
    param_grid : dict, optional
        Parameter grid. If None, uses default grid.
    cv : int
        Number of cross-validation folds. Default 5.
    scoring : str, optional
        Scoring metric. Default None.
    n_jobs : int
        Number of parallel jobs. Default -1.

    Returns
    -------
    dict
        Dictionary with 'best_params', 'best_score', 'best_model',
        'cv_results'.

    Example
    -------
    >>> results = tune_mlp(X_train, y_train, task='classification')
    >>> best_model = results['best_model']
    """
    if task == 'classification':
        estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(max_iter=500, early_stopping=True, random_state=42))
        ])
        default_scoring = 'accuracy'
    else:
        estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(max_iter=500, early_stopping=True, random_state=42))
        ])
        default_scoring = 'r2'

    if param_grid is None:
        param_grid = {
            'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50), (128, 64, 32)],
            'mlp__activation': ['relu', 'tanh'],
            'mlp__alpha': [0.0001, 0.001, 0.01],
            'mlp__learning_rate_init': [0.001, 0.01]
        }

    scoring = scoring or default_scoring

    grid_search = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=True
    )

    grid_search.fit(X, y)

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_,
        'cv_results': pd.DataFrame(grid_search.cv_results_)
    }


# =============================================================================
# Model Comparison Functions
# =============================================================================

def compare_advanced_models(
    X: np.ndarray,
    y: np.ndarray,
    task: Literal['classification', 'regression'] = 'classification',
    cv: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare SVM, MLP, and baseline models.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target array.
    task : str
        'classification' or 'regression'. Default 'classification'.
    cv : int
        Number of cross-validation folds. Default 5.
    random_state : int
        Random seed. Default 42.

    Returns
    -------
    pd.DataFrame
        Comparison results with model names, mean scores, and std.

    Example
    -------
    >>> comparison = compare_advanced_models(X, y, task='classification')
    >>> print(comparison)
    """
    if task == 'classification':
        models = {
            'Logistic Regression (baseline)': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(random_state=random_state))
            ]),
            'SVM (linear)': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(kernel='linear', random_state=random_state))
            ]),
            'SVM (RBF)': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(kernel='rbf', random_state=random_state))
            ]),
            'MLP (100,)': Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPClassifier(
                    hidden_layer_sizes=(100,),
                    early_stopping=True,
                    random_state=random_state
                ))
            ]),
            'MLP (100, 50)': Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    early_stopping=True,
                    random_state=random_state
                ))
            ])
        }
        scoring = 'accuracy'
    else:
        models = {
            'Ridge Regression (baseline)': Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(random_state=random_state))
            ]),
            'SVR (linear)': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='linear'))
            ]),
            'SVR (RBF)': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='rbf'))
            ]),
            'MLP (100,)': Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPRegressor(
                    hidden_layer_sizes=(100,),
                    early_stopping=True,
                    random_state=random_state
                ))
            ]),
            'MLP (100, 50)': Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    early_stopping=True,
                    random_state=random_state
                ))
            ])
        }
        scoring = 'r2'

    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        results.append({
            'Model': name,
            'Mean Score': scores.mean(),
            'Std': scores.std(),
            'Scores': scores
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Mean Score', ascending=False)
    df_results = df_results.reset_index(drop=True)

    return df_results[['Model', 'Mean Score', 'Std']]


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_decision_boundary(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[Tuple[str, str]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    resolution: int = 100
) -> plt.Axes:
    """
    Plot decision boundary for 2D classification.

    Parameters
    ----------
    model : Pipeline
        Fitted model pipeline.
    X : np.ndarray
        Feature matrix with exactly 2 features.
    y : np.ndarray
        Target labels.
    feature_names : tuple of str, optional
        Names for x and y axes.
    title : str, optional
        Plot title.
    ax : plt.Axes, optional
        Matplotlib axes to plot on.
    resolution : int
        Grid resolution for decision boundary. Default 100.

    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.

    Example
    -------
    >>> model = fit_svc(X[:, :2], y, kernel='rbf')
    >>> plot_decision_boundary(model, X[:, :2], y)
    >>> plt.show()
    """
    if X.shape[1] != 2:
        raise ValueError("plot_decision_boundary requires exactly 2 features")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Get predictions for mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5)

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                         edgecolors='black', s=50)

    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Decision Boundary')

    plt.colorbar(scatter, ax=ax, label='Class')

    return ax


def get_model_complexity_curve(
    X: np.ndarray,
    y: np.ndarray,
    model_type: Literal['svm', 'mlp'] = 'svm',
    param_name: str = 'C',
    param_range: Optional[np.ndarray] = None,
    task: Literal['classification', 'regression'] = 'classification',
    cv: int = 5
) -> Dict[str, Any]:
    """
    Analyze model complexity by varying a hyperparameter.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target array.
    model_type : str
        'svm' or 'mlp'. Default 'svm'.
    param_name : str
        Parameter to vary. For SVM: 'C', 'gamma'.
        For MLP: 'alpha', 'hidden_layer_sizes'. Default 'C'.
    param_range : np.ndarray, optional
        Range of parameter values. If None, uses default range.
    task : str
        'classification' or 'regression'. Default 'classification'.
    cv : int
        Number of CV folds. Default 5.

    Returns
    -------
    dict
        Dictionary with 'param_range', 'train_scores', 'test_scores'.

    Example
    -------
    >>> results = get_model_complexity_curve(X, y, model_type='svm', param_name='C')
    >>> plt.plot(results['param_range'], results['test_scores'].mean(axis=1))
    """
    # Set default parameter ranges
    default_ranges = {
        'svm': {
            'C': np.logspace(-3, 3, 7),
            'gamma': np.logspace(-4, 1, 6)
        },
        'mlp': {
            'alpha': np.logspace(-5, 1, 7),
            'hidden_layer_sizes': [(10,), (50,), (100,), (100, 50), (100, 100), (200, 100)]
        }
    }

    if param_range is None:
        param_range = default_ranges[model_type].get(param_name)
        if param_range is None:
            raise ValueError(f"No default range for {param_name}. Please provide param_range.")

    train_scores_list = []
    test_scores_list = []

    for param_value in param_range:
        if model_type == 'svm':
            if task == 'classification':
                base_model = SVC(random_state=42)
            else:
                base_model = SVR()
            setattr(base_model, param_name, param_value)
        else:  # mlp
            if task == 'classification':
                base_model = MLPClassifier(max_iter=500, early_stopping=True, random_state=42)
            else:
                base_model = MLPRegressor(max_iter=500, early_stopping=True, random_state=42)

            if param_name == 'hidden_layer_sizes':
                base_model.hidden_layer_sizes = param_value
            else:
                setattr(base_model, param_name, param_value)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])

        # Get train and test scores via learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline, X, y, cv=cv,
            train_sizes=[1.0],
            scoring='accuracy' if task == 'classification' else 'r2',
            n_jobs=-1
        )

        train_scores_list.append(train_scores.flatten())
        test_scores_list.append(test_scores.flatten())

    return {
        'param_range': param_range,
        'param_name': param_name,
        'train_scores': np.array(train_scores_list),
        'test_scores': np.array(test_scores_list)
    }


def plot_complexity_curve(
    results: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None
) -> plt.Axes:
    """
    Plot the model complexity curve from get_model_complexity_curve results.

    Parameters
    ----------
    results : dict
        Output from get_model_complexity_curve.
    ax : plt.Axes, optional
        Matplotlib axes.
    title : str, optional
        Plot title.

    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    param_range = results['param_range']
    train_mean = results['train_scores'].mean(axis=1)
    train_std = results['train_scores'].std(axis=1)
    test_mean = results['test_scores'].mean(axis=1)
    test_std = results['test_scores'].std(axis=1)

    # Convert to string for non-numeric param ranges (like hidden_layer_sizes)
    if isinstance(param_range[0], tuple):
        x_labels = [str(p) for p in param_range]
        x_range = range(len(param_range))
        ax.set_xticks(x_range)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    else:
        x_range = param_range
        ax.set_xscale('log')

    ax.plot(x_range, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(x_range, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color='blue')

    ax.plot(x_range, test_mean, 'o-', color='red', label='CV Test Score')
    ax.fill_between(x_range, test_mean - test_std, test_mean + test_std,
                    alpha=0.1, color='red')

    ax.set_xlabel(results['param_name'])
    ax.set_ylabel('Score')
    ax.set_title(title or f"Model Complexity Curve ({results['param_name']})")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    return ax


# =============================================================================
# Causal Inference Utilities
# =============================================================================

def fit_calibrated_propensity_model(
    X: np.ndarray,
    treatment: np.ndarray,
    model_type: Literal['svm', 'mlp'] = 'mlp',
    cv: int = 5,
    **model_kwargs
) -> CalibratedClassifierCV:
    """
    Fit a calibrated model for propensity score estimation.

    Calibration improves probability estimates, which is important
    for propensity score methods in causal inference.

    Parameters
    ----------
    X : np.ndarray
        Confounders/features.
    treatment : np.ndarray
        Binary treatment indicator.
    model_type : str
        'svm' or 'mlp'. Default 'mlp'.
    cv : int
        CV folds for calibration. Default 5.
    **model_kwargs
        Additional arguments for the base model.

    Returns
    -------
    CalibratedClassifierCV
        Calibrated model for propensity score estimation.

    Example
    -------
    >>> calibrated_model = fit_calibrated_propensity_model(X, T, model_type='mlp')
    >>> propensity_scores = calibrated_model.predict_proba(X)[:, 1]
    """
    if model_type == 'svm':
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(probability=True, random_state=42, **model_kwargs))
        ])
    else:
        default_mlp_kwargs = {
            'hidden_layer_sizes': (64, 32),
            'activation': 'relu',
            'early_stopping': True,
            'random_state': 42
        }
        default_mlp_kwargs.update(model_kwargs)
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(**default_mlp_kwargs))
        ])

    calibrated_model = CalibratedClassifierCV(base_pipeline, cv=cv)
    calibrated_model.fit(X, treatment)

    return calibrated_model


def get_propensity_scores(
    model: Union[Pipeline, CalibratedClassifierCV],
    X: np.ndarray
) -> np.ndarray:
    """
    Extract propensity scores from a fitted model.

    Parameters
    ----------
    model : Pipeline or CalibratedClassifierCV
        Fitted classification model.
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    np.ndarray
        Propensity scores (probability of treatment=1).
    """
    return model.predict_proba(X)[:, 1]


# =============================================================================
# Model Diagnostics
# =============================================================================

def check_convergence(model: Pipeline) -> Dict[str, Any]:
    """
    Check if MLP model converged during training.

    Parameters
    ----------
    model : Pipeline
        Fitted MLP pipeline.

    Returns
    -------
    dict
        Convergence information.
    """
    # Get the MLP from pipeline
    mlp = None
    for name, step in model.named_steps.items():
        if isinstance(step, (MLPClassifier, MLPRegressor)):
            mlp = step
            break

    if mlp is None:
        return {'error': 'No MLP found in pipeline'}

    converged = mlp.n_iter_ < mlp.max_iter

    return {
        'converged': converged,
        'iterations': mlp.n_iter_,
        'max_iterations': mlp.max_iter,
        'final_loss': mlp.loss_,
        'loss_curve': mlp.loss_curve_ if hasattr(mlp, 'loss_curve_') else None,
        'recommendation': None if converged else
            'Model did not converge. Consider increasing max_iter or adjusting learning_rate.'
    }


def plot_learning_curve_advanced(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot learning curve for an advanced model.

    Parameters
    ----------
    model : Pipeline
        Model pipeline (unfitted).
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target array.
    cv : int
        CV folds. Default 5.
    train_sizes : np.ndarray
        Fractions of training data to use.
    ax : plt.Axes, optional
        Matplotlib axes.

    Returns
    -------
    plt.Axes
        Axes with learning curve plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color='blue')

    ax.plot(train_sizes_abs, test_mean, 'o-', color='red', label='CV Score')
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                    alpha=0.1, color='red')

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    return ax


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("Advanced ML Models - Demo")
    print("=" * 60)

    # Generate classification data
    X_clf, y_clf = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_redundant=2, random_state=42
    )
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    # Generate regression data
    X_reg, y_reg = make_regression(
        n_samples=500, n_features=10, n_informative=5,
        noise=10, random_state=42
    )
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Demo 1: SVM Classification
    print("\n1. SVM Classification")
    print("-" * 40)
    svc_model = fit_svc(X_train_clf, y_train_clf, kernel='rbf', C=1.0)
    svc_acc = accuracy_score(y_test_clf, svc_model.predict(X_test_clf))
    print(f"   SVC Accuracy: {svc_acc:.4f}")

    # Demo 2: SVM Regression
    print("\n2. SVM Regression")
    print("-" * 40)
    svr_model = fit_svr(X_train_reg, y_train_reg, kernel='rbf', C=10)
    svr_r2 = r2_score(y_test_reg, svr_model.predict(X_test_reg))
    print(f"   SVR R-squared: {svr_r2:.4f}")

    # Demo 3: MLP Classification
    print("\n3. MLP Classification")
    print("-" * 40)
    mlp_clf = fit_mlp_classifier(
        X_train_clf, y_train_clf,
        hidden_layers=(100, 50),
        learning_rate=0.001
    )
    mlp_acc = accuracy_score(y_test_clf, mlp_clf.predict(X_test_clf))
    print(f"   MLP Accuracy: {mlp_acc:.4f}")

    # Check convergence
    conv_info = check_convergence(mlp_clf)
    print(f"   Converged: {conv_info['converged']} ({conv_info['iterations']}/{conv_info['max_iterations']} iterations)")

    # Demo 4: MLP Regression
    print("\n4. MLP Regression")
    print("-" * 40)
    mlp_reg = fit_mlp_regressor(
        X_train_reg, y_train_reg,
        hidden_layers=(128, 64)
    )
    mlp_r2 = r2_score(y_test_reg, mlp_reg.predict(X_test_reg))
    print(f"   MLP R-squared: {mlp_r2:.4f}")

    # Demo 5: Model Comparison
    print("\n5. Model Comparison (Classification)")
    print("-" * 40)
    comparison = compare_advanced_models(X_clf, y_clf, task='classification', cv=5)
    print(comparison.to_string(index=False))

    # Demo 6: Hyperparameter Tuning (small grid for demo)
    print("\n6. SVM Hyperparameter Tuning")
    print("-" * 40)
    small_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 0.1],
        'svm__kernel': ['rbf', 'linear']
    }
    tune_results = tune_svm(X_train_clf, y_train_clf, param_grid=small_grid)
    print(f"   Best params: {tune_results['best_params']}")
    print(f"   Best CV score: {tune_results['best_score']:.4f}")

    # Demo 7: Propensity Score Estimation
    print("\n7. Propensity Score Estimation")
    print("-" * 40)
    # Simulate treatment assignment
    np.random.seed(42)
    treatment = (X_clf[:, 0] + np.random.randn(len(X_clf)) > 0).astype(int)

    calibrated_ps = fit_calibrated_propensity_model(X_clf, treatment, model_type='mlp')
    propensity_scores = get_propensity_scores(calibrated_ps, X_clf)
    print(f"   Propensity score range: [{propensity_scores.min():.3f}, {propensity_scores.max():.3f}]")
    print(f"   Mean propensity score: {propensity_scores.mean():.3f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
