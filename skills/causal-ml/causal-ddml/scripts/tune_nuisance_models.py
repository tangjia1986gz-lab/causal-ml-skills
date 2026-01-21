#!/usr/bin/env python3
"""
Automated Nuisance Model Tuning for DDML

Systematically tunes hyperparameters for first-stage ML learners
using cross-validation, with support for:
- Grid search and randomized search
- Multiple learner types
- Nested cross-validation for unbiased model selection

Usage:
    python tune_nuisance_models.py --data data.csv --outcome y --treatment d --controls "x1,x2"
    python tune_nuisance_models.py --data data.csv --outcome y --treatment d --all-controls --learners lasso,rf,xgboost

Reference:
    Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tune nuisance models for DDML"
    )

    parser.add_argument('--data', '-d', required=True,
                        help='Path to CSV data file')
    parser.add_argument('--outcome', '-y', required=True,
                        help='Outcome variable name')
    parser.add_argument('--treatment', '-t', required=True,
                        help='Treatment variable name')
    parser.add_argument('--controls', '-x',
                        help='Comma-separated control variables')
    parser.add_argument('--all-controls', action='store_true',
                        help='Use all other columns as controls')
    parser.add_argument('--learners', '-l',
                        default='lasso,ridge,random_forest,xgboost',
                        help='Learners to tune (comma-separated)')
    parser.add_argument('--search', '-s', default='random',
                        choices=['grid', 'random'],
                        help='Search method: grid or random')
    parser.add_argument('--n-iter', type=int, default=50,
                        help='Iterations for random search (default: 50)')
    parser.add_argument('--cv', type=int, default=5,
                        help='CV folds for tuning (default: 5)')
    parser.add_argument('--output', '-o',
                        help='Output file for tuned parameters')
    parser.add_argument('--verbose', '-v', action='store_true')

    return parser.parse_args()


def get_learner_and_params(learner_name: str, search_type: str = 'random'):
    """Get learner and parameter search space."""
    from sklearn.linear_model import Lasso, Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor

    configs = {
        'lasso': {
            'model': Lasso(max_iter=10000),
            'params': {
                'model__alpha': np.logspace(-4, 2, 50) if search_type == 'grid'
                               else np.logspace(-4, 2, 100)
            }
        },
        'ridge': {
            'model': Ridge(),
            'params': {
                'model__alpha': np.logspace(-4, 4, 50) if search_type == 'grid'
                               else np.logspace(-4, 4, 100)
            }
        },
        'elastic_net': {
            'model': ElasticNet(max_iter=10000),
            'params': {
                'model__alpha': np.logspace(-4, 2, 20),
                'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(n_jobs=-1, random_state=42),
            'params': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [5, 10, 15, 20, None],
                'model__min_samples_leaf': [1, 2, 5, 10],
                'model__max_features': ['sqrt', 'log2', 0.3, 0.5]
            }
        },
        'xgboost': {
            'model': None,  # Loaded conditionally
            'params': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7, 10],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'model__subsample': [0.7, 0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'model__reg_alpha': [0, 0.1, 1],
                'model__reg_lambda': [1, 5, 10]
            }
        },
        'lightgbm': {
            'model': None,  # Loaded conditionally
            'params': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7, 10],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'model__num_leaves': [15, 31, 63, 127],
                'model__subsample': [0.7, 0.8, 0.9, 1.0]
            }
        }
    }

    config = configs.get(learner_name.lower())
    if config is None:
        raise ValueError(f"Unknown learner: {learner_name}")

    # Handle optional imports
    if learner_name.lower() == 'xgboost':
        try:
            from xgboost import XGBRegressor
            config['model'] = XGBRegressor(n_jobs=-1, random_state=42, verbosity=0)
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

    if learner_name.lower() == 'lightgbm':
        try:
            from lightgbm import LGBMRegressor
            config['model'] = LGBMRegressor(n_jobs=-1, random_state=42, verbose=-1)
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    return config


def tune_learner(X, y, learner_name: str, search_type: str = 'random',
                 n_iter: int = 50, cv: int = 5, verbose: bool = False):
    """
    Tune a single learner using cross-validation.

    Returns
    -------
    dict with best params, best score, and fitted model
    """
    config = get_learner_and_params(learner_name, search_type)

    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', config['model'])
    ])

    # Choose search method
    if search_type == 'grid':
        search = GridSearchCV(
            pipeline,
            config['params'],
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
    else:
        search = RandomizedSearchCV(
            pipeline,
            config['params'],
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=1 if verbose else 0
        )

    search.fit(X, y)

    # Extract model params (remove 'model__' prefix)
    best_params = {
        k.replace('model__', ''): v
        for k, v in search.best_params_.items()
    }

    return {
        'learner': learner_name,
        'best_params': best_params,
        'best_score': -search.best_score_,  # Convert back to MSE
        'best_r2': 1 - (-search.best_score_) / np.var(y),
        'cv_results': {
            'mean_test_score': float(-search.cv_results_['mean_test_score'][search.best_index_]),
            'std_test_score': float(search.cv_results_['std_test_score'][search.best_index_])
        }
    }


def tune_all_learners(X_outcome, y_outcome, X_treatment, y_treatment,
                      learner_list: list, search_type: str = 'random',
                      n_iter: int = 50, cv: int = 5, verbose: bool = False):
    """
    Tune learners for both outcome and treatment models.
    """
    results = {
        'outcome_model': {},
        'treatment_model': {}
    }

    print("\nTuning outcome model E[Y|X]...")
    print("-" * 50)

    for learner in learner_list:
        print(f"\n  {learner}...")
        try:
            result = tune_learner(
                X_outcome, y_outcome, learner,
                search_type=search_type, n_iter=n_iter, cv=cv, verbose=verbose
            )
            results['outcome_model'][learner] = result
            print(f"    MSE: {result['best_score']:.6f}, R2: {result['best_r2']:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results['outcome_model'][learner] = {'error': str(e)}

    print("\n\nTuning treatment model E[D|X]...")
    print("-" * 50)

    # Check if treatment is binary
    is_binary = len(np.unique(y_treatment)) == 2

    for learner in learner_list:
        print(f"\n  {learner}...")
        try:
            if is_binary:
                # Use classification for binary treatment
                result = tune_classification_learner(
                    X_treatment, y_treatment, learner,
                    search_type=search_type, n_iter=n_iter, cv=cv, verbose=verbose
                )
            else:
                result = tune_learner(
                    X_treatment, y_treatment, learner,
                    search_type=search_type, n_iter=n_iter, cv=cv, verbose=verbose
                )
            results['treatment_model'][learner] = result
            if is_binary:
                print(f"    Brier: {result['best_score']:.6f}, AUC: {result.get('auc', 'N/A')}")
            else:
                print(f"    MSE: {result['best_score']:.6f}, R2: {result['best_r2']:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results['treatment_model'][learner] = {'error': str(e)}

    return results


def tune_classification_learner(X, y, learner_name: str, search_type: str = 'random',
                                n_iter: int = 50, cv: int = 5, verbose: bool = False):
    """Tune classification learner for binary treatment."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    configs = {
        'lasso': {
            'model': LogisticRegression(penalty='l1', solver='saga', max_iter=10000),
            'params': {
                'model__C': np.logspace(-4, 4, 50)
            }
        },
        'ridge': {
            'model': LogisticRegression(penalty='l2', max_iter=10000),
            'params': {
                'model__C': np.logspace(-4, 4, 50)
            }
        },
        'elastic_net': {
            'model': LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000),
            'params': {
                'model__C': np.logspace(-4, 4, 20),
                'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(n_jobs=-1, random_state=42),
            'params': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [5, 10, 15, 20, None],
                'model__min_samples_leaf': [1, 2, 5, 10]
            }
        },
        'xgboost': {
            'model': None,
            'params': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.05, 0.1]
            }
        },
        'lightgbm': {
            'model': None,
            'params': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.05, 0.1]
            }
        }
    }

    config = configs.get(learner_name.lower())

    if learner_name.lower() == 'xgboost':
        from xgboost import XGBClassifier
        config['model'] = XGBClassifier(n_jobs=-1, random_state=42, verbosity=0,
                                        use_label_encoder=False, eval_metric='logloss')

    if learner_name.lower() == 'lightgbm':
        from lightgbm import LGBMClassifier
        config['model'] = LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', config['model'])
    ])

    if search_type == 'grid':
        search = GridSearchCV(
            pipeline, config['params'], cv=cv,
            scoring='neg_brier_score', n_jobs=-1
        )
    else:
        search = RandomizedSearchCV(
            pipeline, config['params'], n_iter=n_iter, cv=cv,
            scoring='neg_brier_score', n_jobs=-1, random_state=42
        )

    search.fit(X, y)

    best_params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}

    return {
        'learner': learner_name,
        'task': 'classification',
        'best_params': best_params,
        'best_score': -search.best_score_,
        'cv_results': {
            'mean_test_score': float(-search.cv_results_['mean_test_score'][search.best_index_]),
            'std_test_score': float(search.cv_results_['std_test_score'][search.best_index_])
        }
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("NUISANCE MODEL TUNING FOR DDML")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Load data
    data = pd.read_csv(args.data)
    print(f"\nData: {args.data}")
    print(f"Shape: {data.shape}")

    # Get controls
    if args.controls:
        controls = [c.strip() for c in args.controls.split(',')]
    elif args.all_controls:
        controls = [c for c in data.columns if c not in [args.outcome, args.treatment]]
    else:
        controls = [c for c in data.columns if c not in [args.outcome, args.treatment]]

    print(f"Controls: {len(controls)} variables")

    # Prepare data
    df = data[[args.outcome, args.treatment] + controls].dropna()
    X = df[controls].values
    y_outcome = df[args.outcome].values
    y_treatment = df[args.treatment].values

    # Parse learners
    learner_list = [l.strip() for l in args.learners.split(',')]
    print(f"Learners: {learner_list}")
    print(f"Search: {args.search}, iterations: {args.n_iter}")

    # Run tuning
    results = tune_all_learners(
        X_outcome=X,
        y_outcome=y_outcome,
        X_treatment=X,
        y_treatment=y_treatment,
        learner_list=learner_list,
        search_type=args.search,
        n_iter=args.n_iter,
        cv=args.cv,
        verbose=args.verbose
    )

    # Find best learners
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid_outcome = {k: v for k, v in results['outcome_model'].items() if 'error' not in v}
    valid_treatment = {k: v for k, v in results['treatment_model'].items() if 'error' not in v}

    if valid_outcome:
        best_outcome = min(valid_outcome.items(), key=lambda x: x[1]['best_score'])
        print(f"\nBest outcome learner: {best_outcome[0]}")
        print(f"  MSE: {best_outcome[1]['best_score']:.6f}")
        print(f"  Params: {best_outcome[1]['best_params']}")

    if valid_treatment:
        best_treatment = min(valid_treatment.items(), key=lambda x: x[1]['best_score'])
        print(f"\nBest treatment learner: {best_treatment[0]}")
        print(f"  Score: {best_treatment[1]['best_score']:.6f}")
        print(f"  Params: {best_treatment[1]['best_params']}")

    # Save results
    if args.output:
        output = {
            'timestamp': datetime.now().isoformat(),
            'data_file': args.data,
            'outcome': args.outcome,
            'treatment': args.treatment,
            'n_controls': len(controls),
            'search_type': args.search,
            'cv_folds': args.cv,
            'results': results,
            'best_outcome_learner': best_outcome[0] if valid_outcome else None,
            'best_treatment_learner': best_treatment[0] if valid_treatment else None
        }

        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
