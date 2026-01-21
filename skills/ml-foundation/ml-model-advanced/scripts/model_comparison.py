#!/usr/bin/env python3
"""
Cross-Validated Model Comparison

Compare multiple advanced models (SVM, MLP) against baselines using
cross-validation. Generates detailed comparison reports.

Usage Examples:
    # Basic comparison on synthetic data
    python model_comparison.py

    # Compare on custom data with classification
    python model_comparison.py --data data.csv --target y --task classification

    # Quick comparison with fewer folds
    python model_comparison.py --cv 3 --quick

    # Generate markdown report
    python model_comparison.py --output comparison_report.md
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_classification, make_regression

warnings.filterwarnings('ignore')


def load_data(
    data_path: Optional[str] = None,
    target_col: Optional[str] = None,
    task: str = 'classification',
    n_samples: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from file or generate synthetic data."""
    if data_path is not None:
        df = pd.read_csv(data_path)
        if target_col is None:
            target_col = df.columns[-1]
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        return X, y

    if task == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=20,
            n_informative=10,
            noise=10,
            random_state=42
        )
    return X, y


def get_classification_models(quick: bool = False) -> Dict:
    """Get dictionary of classification models to compare."""
    models = {
        # Baselines
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42))
        ]),

        # SVM variants
        'SVM (Linear)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='linear', random_state=42))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', random_state=42))
        ]),

        # MLP variants
        'MLP (64,)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(
                hidden_layer_sizes=(64,),
                early_stopping=True,
                random_state=42
            ))
        ]),
        'MLP (100, 50)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                early_stopping=True,
                random_state=42
            ))
        ]),
    }

    if not quick:
        # Add tree-based models for comparison
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100, random_state=42
        )
        models['SVM (Poly)'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='poly', degree=3, random_state=42))
        ])
        models['MLP (128, 64, 32)'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                early_stopping=True,
                random_state=42
            ))
        ])

    return models


def get_regression_models(quick: bool = False) -> Dict:
    """Get dictionary of regression models to compare."""
    models = {
        # Baselines
        'Ridge Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(random_state=42))
        ]),

        # SVM variants
        'SVR (Linear)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='linear'))
        ]),
        'SVR (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf'))
        ]),

        # MLP variants
        'MLP (64,)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(64,),
                early_stopping=True,
                random_state=42
            ))
        ]),
        'MLP (100, 50)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(100, 50),
                early_stopping=True,
                random_state=42
            ))
        ]),
    }

    if not quick:
        models['Lasso Regression'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(random_state=42))
        ])
        models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )
        models['SVR (Poly)'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='poly', degree=3))
        ])
        models['MLP (128, 64, 32)'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                early_stopping=True,
                random_state=42
            ))
        ])

    return models


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    models: Dict,
    cv: int = 5,
    scoring: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare models using cross-validation.

    Parameters
    ----------
    X : array
        Feature matrix.
    y : array
        Target vector.
    models : dict
        Dictionary of model name -> model object.
    cv : int
        Number of CV folds.
    scoring : list, optional
        List of scoring metrics.
    verbose : bool
        Print progress.

    Returns
    -------
    DataFrame
        Comparison results.
    """
    results = []

    for name, model in models.items():
        if verbose:
            print(f"  Evaluating: {name}...", end=' ', flush=True)

        try:
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )

            result = {'Model': name}

            for metric in scoring:
                test_key = f'test_{metric}'
                train_key = f'train_{metric}'

                if test_key in cv_results:
                    result[f'{metric}_mean'] = cv_results[test_key].mean()
                    result[f'{metric}_std'] = cv_results[test_key].std()
                    result[f'{metric}_train'] = cv_results[train_key].mean()

            result['fit_time'] = cv_results['fit_time'].mean()

            results.append(result)

            if verbose:
                primary_metric = f'{scoring[0]}_mean'
                print(f"{result[primary_metric]:.4f}")

        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            results.append({'Model': name, 'error': str(e)})

    return pd.DataFrame(results)


def analyze_overfitting(df: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    """Analyze overfitting by comparing train vs test scores."""
    train_col = f'{primary_metric}_train'
    test_col = f'{primary_metric}_mean'

    if train_col in df.columns and test_col in df.columns:
        df['overfit_gap'] = df[train_col] - df[test_col]
        df['overfit_ratio'] = df[train_col] / df[test_col].replace(0, np.nan)

    return df


def generate_markdown_report(
    df: pd.DataFrame,
    task: str,
    n_samples: int,
    n_features: int,
    cv: int,
    output_path: str
) -> None:
    """Generate markdown comparison report."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    primary_metric = 'accuracy' if task == 'classification' else 'r2'
    df_sorted = df.sort_values(f'{primary_metric}_mean', ascending=False)

    report = f"""# Model Comparison Report

Generated: {timestamp}

## Dataset Summary

| Property | Value |
|----------|-------|
| Task | {task.capitalize()} |
| Samples | {n_samples} |
| Features | {n_features} |
| CV Folds | {cv} |

## Model Rankings

### By {primary_metric.upper()} Score

| Rank | Model | {primary_metric.upper()} | Std | Train Score | Overfit Gap |
|------|-------|----------|-----|-------------|-------------|
"""

    for i, row in df_sorted.iterrows():
        rank = df_sorted.index.get_loc(i) + 1
        model = row['Model']
        test_score = row.get(f'{primary_metric}_mean', np.nan)
        std = row.get(f'{primary_metric}_std', np.nan)
        train_score = row.get(f'{primary_metric}_train', np.nan)
        gap = row.get('overfit_gap', np.nan)

        report += f"| {rank} | {model} | {test_score:.4f} | {std:.4f} | {train_score:.4f} | {gap:.4f} |\n"

    # Add analysis section
    best_model = df_sorted.iloc[0]['Model']
    best_score = df_sorted.iloc[0][f'{primary_metric}_mean']

    report += f"""
## Analysis

### Best Performing Model

**{best_model}** achieved the highest {primary_metric} score of **{best_score:.4f}**.

### Key Observations

"""

    # Check for overfitting
    df_overfit = df_sorted[df_sorted['overfit_gap'] > 0.1]
    if len(df_overfit) > 0:
        report += "**Overfitting Warning:** The following models show significant train-test gaps (>0.1):\n"
        for _, row in df_overfit.iterrows():
            report += f"- {row['Model']}: gap = {row['overfit_gap']:.4f}\n"
        report += "\n"

    # Training time comparison
    fastest = df_sorted.nsmallest(3, 'fit_time')
    report += "**Fastest Models (fit time):**\n"
    for _, row in fastest.iterrows():
        report += f"- {row['Model']}: {row['fit_time']:.3f}s\n"

    report += """
## Recommendations

1. **For production**: Consider the trade-off between accuracy and training time
2. **For interpretability**: Prefer simpler models (Linear, Logistic) if performance is comparable
3. **For complex patterns**: Use ensemble methods or neural networks
4. **For small data**: Use simpler models with regularization to avoid overfitting
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")


def print_comparison_table(df: pd.DataFrame, primary_metric: str) -> None:
    """Print formatted comparison table."""
    df_sorted = df.sort_values(f'{primary_metric}_mean', ascending=False)

    print("\n" + "="*80)
    print(" MODEL COMPARISON RESULTS")
    print("="*80)

    # Format for display
    display_cols = ['Model', f'{primary_metric}_mean', f'{primary_metric}_std', 'fit_time']
    available_cols = [c for c in display_cols if c in df_sorted.columns]

    print(df_sorted[available_cols].to_string(index=False))
    print("="*80)

    # Print winner
    winner = df_sorted.iloc[0]
    print(f"\nBest Model: {winner['Model']}")
    print(f"Score: {winner[f'{primary_metric}_mean']:.4f} (+/- {winner[f'{primary_metric}_std']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='Compare advanced ML models with cross-validation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--task', '-t',
        choices=['classification', 'regression'],
        default='classification',
        help='Task type (default: classification)'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Path to CSV data file'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Target column name'
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick comparison with fewer models'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for markdown report'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Output path for CSV results'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples for synthetic data (default: 1000)'
    )

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    X, y = load_data(args.data, args.target, args.task, args.n_samples)
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # Get models
    if args.task == 'classification':
        models = get_classification_models(args.quick)
        scoring = ['accuracy', 'f1', 'roc_auc']
        primary_metric = 'accuracy'
    else:
        models = get_regression_models(args.quick)
        scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        primary_metric = 'r2'

    print(f"\nComparing {len(models)} models with {args.cv}-fold CV...")
    print("-" * 50)

    # Run comparison
    results_df = compare_models(X, y, models, cv=args.cv, scoring=scoring)

    # Analyze overfitting
    results_df = analyze_overfitting(results_df, primary_metric)

    # Print results
    print_comparison_table(results_df, primary_metric)

    # Save CSV
    if args.csv:
        results_df.to_csv(args.csv, index=False)
        print(f"\nCSV results saved to: {args.csv}")

    # Generate report
    if args.output:
        generate_markdown_report(
            results_df,
            task=args.task,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            cv=args.cv,
            output_path=args.output
        )


if __name__ == '__main__':
    main()
