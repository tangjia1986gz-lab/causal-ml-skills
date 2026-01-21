#!/usr/bin/env python3
"""
Advanced Model Training CLI

Train SVM or MLP models with proper preprocessing, hyperparameter tuning,
and evaluation. Supports both classification and regression tasks.

Usage Examples:
    # Train SVM classifier with RBF kernel
    python run_advanced_model.py --model svm --task classification --kernel rbf

    # Train MLP regressor with custom architecture
    python run_advanced_model.py --model mlp --task regression --layers 128 64 32

    # Train with hyperparameter tuning
    python run_advanced_model.py --model svm --tune --cv 5

    # Load data from file
    python run_advanced_model.py --model mlp --data data.csv --target y_column
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.datasets import make_classification, make_regression

warnings.filterwarnings('ignore')


def load_data(
    data_path: Optional[str] = None,
    target_col: Optional[str] = None,
    task: str = 'classification'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from file or generate synthetic data.

    Parameters
    ----------
    data_path : str, optional
        Path to CSV file. If None, generates synthetic data.
    target_col : str, optional
        Name of target column in CSV.
    task : str
        'classification' or 'regression'.

    Returns
    -------
    X, y : arrays
        Feature matrix and target vector.
    """
    if data_path is not None:
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)

        if target_col is None:
            # Assume last column is target
            target_col = df.columns[-1]

        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        print(f"  Loaded {X.shape[0]} samples with {X.shape[1]} features")

    else:
        print("Generating synthetic data...")
        if task == 'classification':
            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=10,
                n_redundant=5,
                n_classes=2,
                random_state=42
            )
        else:
            X, y = make_regression(
                n_samples=1000,
                n_features=20,
                n_informative=10,
                noise=10,
                random_state=42
            )
        print(f"  Generated {X.shape[0]} samples with {X.shape[1]} features")

    return X, y


def build_svm_pipeline(
    task: str,
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: str = 'scale',
    epsilon: float = 0.1
) -> Pipeline:
    """Build SVM pipeline with scaling."""
    if task == 'classification':
        model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=42
        )
    else:
        model = SVR(
            kernel=kernel,
            C=C,
            gamma=gamma,
            epsilon=epsilon
        )

    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])


def build_mlp_pipeline(
    task: str,
    hidden_layers: Tuple[int, ...] = (100, 50),
    activation: str = 'relu',
    alpha: float = 0.0001,
    learning_rate: float = 0.001,
    max_iter: int = 500
) -> Pipeline:
    """Build MLP pipeline with scaling."""
    if task == 'classification':
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
    else:
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )

    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])


def get_tuning_grid(model_type: str, task: str) -> dict:
    """Get hyperparameter grid for tuning."""
    if model_type == 'svm':
        return {
            'model__C': [0.1, 1, 10, 100],
            'model__gamma': ['scale', 0.01, 0.1, 1],
            'model__kernel': ['rbf', 'linear', 'poly']
        }
    else:  # mlp
        return {
            'model__hidden_layer_sizes': [(50,), (100,), (100, 50), (128, 64)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': [0.0001, 0.001, 0.01],
            'model__learning_rate_init': [0.001, 0.01]
        }


def evaluate_classification(y_true, y_pred, y_proba=None):
    """Evaluate classification model."""
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }

    if y_proba is not None and len(np.unique(y_true)) == 2:
        results['roc_auc'] = roc_auc_score(y_true, y_proba)

    return results


def evaluate_regression(y_true, y_pred):
    """Evaluate regression model."""
    return {
        'r2_score': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }


def print_results(results: dict, title: str = "Results"):
    """Print formatted results."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print('='*50)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    print('='*50)


def main():
    parser = argparse.ArgumentParser(
        description='Train advanced ML models (SVM or MLP)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model selection
    parser.add_argument(
        '--model', '-m',
        choices=['svm', 'mlp'],
        default='svm',
        help='Model type: svm or mlp (default: svm)'
    )
    parser.add_argument(
        '--task', '-t',
        choices=['classification', 'regression'],
        default='classification',
        help='Task type (default: classification)'
    )

    # Data options
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Path to CSV data file (optional, uses synthetic data if not provided)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Target column name in CSV'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )

    # SVM options
    parser.add_argument(
        '--kernel', '-k',
        choices=['linear', 'rbf', 'poly', 'sigmoid'],
        default='rbf',
        help='SVM kernel type (default: rbf)'
    )
    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='SVM regularization parameter (default: 1.0)'
    )
    parser.add_argument(
        '--gamma',
        type=str,
        default='scale',
        help='SVM gamma parameter (default: scale)'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.1,
        help='SVR epsilon parameter (default: 0.1)'
    )

    # MLP options
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=[100, 50],
        help='MLP hidden layer sizes (default: 100 50)'
    )
    parser.add_argument(
        '--activation',
        choices=['relu', 'tanh', 'logistic'],
        default='relu',
        help='MLP activation function (default: relu)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.0001,
        help='MLP L2 regularization (default: 0.0001)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='MLP learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=500,
        help='MLP maximum iterations (default: 500)'
    )

    # Tuning options
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning with GridSearchCV'
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Cross-validation folds (default: 5)'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for results JSON'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Load data
    X, y = load_data(args.data, args.target, args.task)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Build model
    if args.model == 'svm':
        # Parse gamma
        try:
            gamma = float(args.gamma)
        except ValueError:
            gamma = args.gamma

        pipeline = build_svm_pipeline(
            task=args.task,
            kernel=args.kernel,
            C=args.C,
            gamma=gamma,
            epsilon=args.epsilon
        )
        model_name = f"SVM ({args.kernel} kernel)"
    else:
        pipeline = build_mlp_pipeline(
            task=args.task,
            hidden_layers=tuple(args.layers),
            activation=args.activation,
            alpha=args.alpha,
            learning_rate=args.lr,
            max_iter=args.max_iter
        )
        model_name = f"MLP {tuple(args.layers)}"

    print(f"\nModel: {model_name}")

    # Hyperparameter tuning
    if args.tune:
        print(f"\nPerforming GridSearchCV with {args.cv} folds...")
        param_grid = get_tuning_grid(args.model, args.task)
        scoring = 'accuracy' if args.task == 'classification' else 'r2'

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=args.cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1 if args.verbose else 0
        )
        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        pipeline = grid_search.best_estimator_
    else:
        # Fit model
        print("\nTraining model...")
        pipeline.fit(X_train, y_train)

    # Check MLP convergence
    if args.model == 'mlp':
        mlp = pipeline.named_steps['model']
        if mlp.n_iter_ == mlp.max_iter:
            print(f"\nWARNING: MLP did not converge ({mlp.n_iter_}/{mlp.max_iter} iterations)")
            print("Consider increasing --max-iter or adjusting --lr")
        else:
            print(f"MLP converged in {mlp.n_iter_} iterations")

    # Cross-validation score
    print(f"\nCross-validation ({args.cv} folds)...")
    scoring = 'accuracy' if args.task == 'classification' else 'r2'
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=args.cv, scoring=scoring)
    print(f"CV {scoring}: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Test set evaluation
    y_pred = pipeline.predict(X_test)

    if args.task == 'classification':
        y_proba = None
        if hasattr(pipeline, 'predict_proba'):
            y_proba = pipeline.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            else:
                y_proba = None  # Multi-class

        results = evaluate_classification(y_test, y_pred, y_proba)

        if args.verbose:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
    else:
        results = evaluate_regression(y_test, y_pred)

    # Add model info to results
    results['model'] = args.model
    results['task'] = args.task
    results['cv_mean'] = cv_scores.mean()
    results['cv_std'] = cv_scores.std()

    if args.tune:
        results['best_params'] = str(grid_search.best_params_)

    print_results(results, f"Test Set Results ({model_name})")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
