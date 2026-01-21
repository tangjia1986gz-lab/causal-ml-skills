#!/usr/bin/env python3
"""
Tree Model Training CLI with Hyperparameter Tuning

This script provides a command-line interface for training tree-based models
with support for hyperparameter tuning and cross-validation.

Usage:
    python run_tree_model.py data.csv --target y --model xgboost --tune
    python run_tree_model.py data.csv --target y --model random_forest --cv 5
    python run_tree_model.py data.csv --target y --model lightgbm --ddml-mode
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from tree_models import (
    fit_decision_tree,
    fit_random_forest,
    fit_xgboost,
    fit_lightgbm,
    tune_hyperparameters,
    compare_tree_models
)


# DDML-optimized hyperparameters
DDML_PARAMS = {
    'xgboost': {
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 3
    },
    'lightgbm': {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_data_in_leaf': 20
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_leaf': 10,
        'max_features': 'sqrt'
    }
}

# Default tuning grids
TUNING_GRIDS = {
    'xgboost': {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [1.0, 2.0, 5.0]
    },
    'lightgbm': {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'feature_fraction': [0.7, 0.8, 0.9],
        'lambda_l1': [0, 0.1, 1.0],
        'lambda_l2': [0, 1.0, 5.0]
    },
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2', 0.5]
    },
    'decision_tree': {
        'max_depth': [3, 5, 7, 10],
        'min_samples_leaf': [1, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
}


def load_data(filepath: str, target: str, features: Optional[str] = None,
              test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """Load and prepare data for modeling."""
    print(f"\nLoading data from: {filepath}")

    # Load data
    df = pd.read_csv(filepath)
    print(f"  Shape: {df.shape}")

    # Handle features
    if features:
        feature_cols = [f.strip() for f in features.split(',')]
    else:
        feature_cols = [c for c in df.columns if c != target]

    print(f"  Target: {target}")
    print(f"  Features ({len(feature_cols)}): {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")

    # Extract X and y
    X = df[feature_cols].values
    y = df[target].values

    # Determine task type
    unique_values = len(np.unique(y))
    if unique_values == 2:
        task = 'classification'
        print(f"  Task: Binary classification")
    elif unique_values < 10 and np.all(y == y.astype(int)):
        task = 'classification'
        print(f"  Task: Multi-class classification ({unique_values} classes)")
    else:
        task = 'regression'
        print(f"  Task: Regression")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Train/Test split: {len(X_train)}/{len(X_test)}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols,
        'task': task
    }


def train_model(data: Dict[str, Any], model_type: str, params: Optional[Dict] = None,
                n_rounds: int = 100, early_stopping: int = 20) -> Dict[str, Any]:
    """Train a tree model with given parameters."""
    print(f"\nTraining {model_type} model...")

    X_train = data['X_train']
    y_train = data['y_train']
    task = data['task']

    if model_type == 'decision_tree':
        result = fit_decision_tree(X_train, y_train, task=task, **(params or {}))
    elif model_type == 'random_forest':
        result = fit_random_forest(X_train, y_train, task=task, **(params or {}))
    elif model_type == 'xgboost':
        result = fit_xgboost(
            X_train, y_train, task=task,
            params=params,
            n_rounds=n_rounds,
            early_stopping_rounds=early_stopping
        )
    elif model_type == 'lightgbm':
        result = fit_lightgbm(
            X_train, y_train, task=task,
            params=params,
            n_rounds=n_rounds,
            early_stopping_rounds=early_stopping
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return result


def evaluate_model(data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate model on test data."""
    from sklearn.metrics import (
        mean_squared_error, r2_score, mean_absolute_error,
        accuracy_score, roc_auc_score, f1_score
    )

    model = result['model']
    X_test = data['X_test']
    y_test = data['y_test']
    task = data['task']

    y_pred = model.predict(X_test)

    metrics = {}
    if task == 'regression':
        metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
    else:
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                metrics['auc'] = roc_auc_score(y_test, y_proba[:, 1])
        metrics['f1'] = f1_score(y_test, y_pred, average='weighted')

    return metrics


def run_tuning(data: Dict[str, Any], model_type: str, param_grid: Optional[Dict] = None,
               cv: int = 5) -> Dict[str, Any]:
    """Run hyperparameter tuning."""
    print(f"\nTuning {model_type} hyperparameters...")
    print(f"  Cross-validation folds: {cv}")

    if param_grid is None:
        param_grid = TUNING_GRIDS.get(model_type, {})

    print(f"  Parameter grid: {param_grid}")

    result = tune_hyperparameters(
        data['X_train'], data['y_train'],
        model_type=model_type,
        param_grid=param_grid,
        task=data['task'],
        cv=cv
    )

    print(f"\n  Best parameters: {result['best_params']}")
    print(f"  Best CV score: {result['best_score']:.4f}")

    return result


def run_comparison(data: Dict[str, Any], cv: int = 5) -> Dict[str, Any]:
    """Compare all tree models."""
    print(f"\nComparing all tree models...")

    result = compare_tree_models(
        data['X_train'], data['y_train'],
        task=data['task'],
        cv=cv
    )

    print("\nModel Comparison Results:")
    print(result['summary'].to_string(index=False))

    return result


def save_results(result: Dict[str, Any], metrics: Dict[str, float],
                 output_path: str, data: Dict[str, Any]) -> None:
    """Save model and results."""
    import joblib
    from datetime import datetime

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_path / 'model.joblib'
    joblib.dump(result['model'], model_path)
    print(f"\nModel saved to: {model_path}")

    # Save metrics and metadata
    report = {
        'timestamp': datetime.now().isoformat(),
        'task': data['task'],
        'n_features': len(data['feature_names']),
        'feature_names': data['feature_names'],
        'n_train': len(data['X_train']),
        'n_test': len(data['X_test']),
        'metrics': metrics,
        'feature_importance': dict(zip(
            data['feature_names'],
            result['feature_importance'].tolist()
        ))
    }

    if 'params' in result:
        report['params'] = result['params']
    if 'best_iteration' in result:
        report['best_iteration'] = result['best_iteration']

    report_path = output_path / 'report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train tree-based models with optional hyperparameter tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train XGBoost with default settings
  python run_tree_model.py data.csv --target y --model xgboost

  # Train with hyperparameter tuning
  python run_tree_model.py data.csv --target y --model xgboost --tune

  # Train with DDML-optimized settings
  python run_tree_model.py data.csv --target y --model xgboost --ddml-mode

  # Compare all tree models
  python run_tree_model.py data.csv --target y --compare

  # Specify features and output directory
  python run_tree_model.py data.csv --target y --features "x1,x2,x3" --output ./results
        """
    )

    parser.add_argument('data', type=str, help='Path to CSV data file')
    parser.add_argument('--target', '-y', type=str, required=True,
                        help='Target variable column name')
    parser.add_argument('--features', '-X', type=str, default=None,
                        help='Comma-separated feature column names (default: all except target)')
    parser.add_argument('--model', '-m', type=str, default='xgboost',
                        choices=['decision_tree', 'random_forest', 'xgboost', 'lightgbm'],
                        help='Model type (default: xgboost)')
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all tree models')
    parser.add_argument('--ddml-mode', action='store_true',
                        help='Use DDML-optimized hyperparameters')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--n-rounds', type=int, default=200,
                        help='Number of boosting rounds (default: 200)')
    parser.add_argument('--early-stopping', type=int, default=30,
                        help='Early stopping rounds (default: 30)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')
    parser.add_argument('--output', '-o', type=str, default='./model_output',
                        help='Output directory for model and results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    if args.quiet:
        warnings.filterwarnings('ignore')

    # Load data
    data = load_data(
        args.data,
        target=args.target,
        features=args.features,
        test_size=args.test_size,
        random_state=args.seed
    )

    # Compare mode
    if args.compare:
        comparison = run_comparison(data, cv=args.cv)
        return

    # Tuning mode
    if args.tune:
        tune_result = run_tuning(data, args.model, cv=args.cv)
        params = tune_result['best_params']
    elif args.ddml_mode:
        print(f"\nUsing DDML-optimized parameters for {args.model}")
        params = DDML_PARAMS.get(args.model, {})
    else:
        params = None

    # Train model
    result = train_model(
        data, args.model,
        params=params,
        n_rounds=args.n_rounds,
        early_stopping=args.early_stopping
    )

    # Evaluate
    metrics = evaluate_model(data, result)

    print("\nTest Set Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Feature importance (top 10)
    importance = pd.DataFrame({
        'feature': data['feature_names'],
        'importance': result['feature_importance']
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Feature Importance:")
    print(importance.head(10).to_string(index=False))

    # Save results
    save_results(result, metrics, args.output, data)

    print("\nDone!")


if __name__ == '__main__':
    main()
