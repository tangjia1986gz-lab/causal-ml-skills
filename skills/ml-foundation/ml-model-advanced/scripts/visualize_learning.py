#!/usr/bin/env python3
"""
Learning and Validation Curve Visualization

Generate learning curves and validation curves for advanced models
to diagnose overfitting, underfitting, and optimal hyperparameters.

Usage Examples:
    # Generate learning curve for MLP
    python visualize_learning.py --model mlp --plot learning

    # Generate validation curve for SVM C parameter
    python visualize_learning.py --model svm --plot validation --param C

    # Save plots to file
    python visualize_learning.py --model mlp --plot learning --output learning_curve.png

    # Generate all diagnostic plots
    python visualize_learning.py --model mlp --plot all --output-dir plots/
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import make_classification, make_regression

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def load_data(
    task: str = 'classification',
    n_samples: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for visualization."""
    if task == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=10,
            n_redundant=5,
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


def build_model(
    model_type: str,
    task: str,
    **kwargs
) -> Pipeline:
    """Build model pipeline."""
    if model_type == 'svm':
        if task == 'classification':
            model = SVC(random_state=42, **kwargs)
        else:
            model = SVR(**kwargs)
    else:  # mlp
        if task == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                early_stopping=True,
                random_state=42,
                **kwargs
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                early_stopping=True,
                random_state=42,
                **kwargs
            )

    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])


def plot_learning_curve(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    scoring: str = 'accuracy',
    title: str = 'Learning Curve',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot learning curve showing training and validation scores
    as a function of training set size.

    Helps diagnose:
    - Overfitting: Large gap between train and validation
    - Underfitting: Both curves plateau at low scores
    - Need more data: Curves still converging
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    # Plot training scores
    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1, color='blue'
    )
    ax.plot(
        train_sizes_abs, train_mean,
        'o-', color='blue', label='Training Score'
    )

    # Plot validation scores
    ax.fill_between(
        train_sizes_abs,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.1, color='red'
    )
    ax.plot(
        train_sizes_abs, test_mean,
        'o-', color='red', label='Cross-Validation Score'
    )

    # Add gap annotation
    final_gap = train_mean[-1] - test_mean[-1]
    ax.annotate(
        f'Gap: {final_gap:.3f}',
        xy=(train_sizes_abs[-1], (train_mean[-1] + test_mean[-1]) / 2),
        xytext=(10, 0),
        textcoords='offset points',
        fontsize=10,
        color='gray'
    )

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add interpretation guide
    ax.text(
        0.02, 0.02,
        'High gap = Overfitting | Low scores = Underfitting | Converging = Need more data',
        transform=ax.transAxes,
        fontsize=8,
        color='gray',
        verticalalignment='bottom'
    )

    return ax


def plot_validation_curve(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    param_name: str,
    param_range: np.ndarray,
    cv: int = 5,
    scoring: str = 'accuracy',
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True
) -> plt.Axes:
    """
    Plot validation curve showing training and validation scores
    as a function of a hyperparameter.

    Helps find optimal hyperparameter values.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    train_scores, test_scores = validation_curve(
        model, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    # Plot
    ax.fill_between(
        param_range,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1, color='blue'
    )
    ax.plot(
        param_range, train_mean,
        'o-', color='blue', label='Training Score'
    )

    ax.fill_between(
        param_range,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.1, color='red'
    )
    ax.plot(
        param_range, test_mean,
        'o-', color='red', label='Validation Score'
    )

    # Mark best parameter
    best_idx = test_mean.argmax()
    best_param = param_range[best_idx]
    best_score = test_mean[best_idx]

    ax.axvline(
        best_param, color='green', linestyle='--', alpha=0.7,
        label=f'Best: {param_name}={best_param:.4g}'
    )
    ax.scatter([best_param], [best_score], color='green', s=100, zorder=5)

    if log_scale:
        ax.set_xscale('log')

    ax.set_xlabel(param_name)
    ax.set_ylabel('Score')
    ax.set_title(title or f'Validation Curve ({param_name})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    return ax


def plot_loss_curve(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    title: str = 'Loss Curve',
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Plot MLP training loss curve.

    Only works for fitted MLP models.
    """
    # Fit model first
    model.fit(X, y)

    # Get MLP from pipeline
    mlp = model.named_steps.get('model')
    if mlp is None or not hasattr(mlp, 'loss_curve_'):
        print("Warning: loss_curve_ not available (not an MLP or not fitted)")
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    loss_curve = mlp.loss_curve_
    epochs = range(1, len(loss_curve) + 1)

    ax.plot(epochs, loss_curve, 'b-', linewidth=2)

    # Mark convergence
    if hasattr(mlp, 'best_loss_'):
        ax.axhline(
            mlp.best_loss_, color='green', linestyle='--',
            label=f'Best Loss: {mlp.best_loss_:.4f}'
        )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add convergence info
    converged = mlp.n_iter_ < mlp.max_iter
    status = 'Converged' if converged else 'Did not converge'
    ax.text(
        0.98, 0.98,
        f'{status} ({mlp.n_iter_}/{mlp.max_iter} iterations)',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        color='green' if converged else 'red'
    )

    return ax


def plot_complexity_analysis(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    task: str,
    cv: int = 5,
    scoring: str = 'accuracy'
) -> plt.Figure:
    """
    Generate comprehensive complexity analysis with multiple parameters.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if model_type == 'svm':
        # C parameter
        model = build_model(model_type, task)
        param_range = np.logspace(-3, 3, 7)
        plot_validation_curve(
            model, X, y,
            param_name='model__C',
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            title='Regularization (C)',
            ax=axes[0, 0]
        )

        # Gamma parameter (for RBF)
        model = build_model(model_type, task, kernel='rbf')
        param_range = np.logspace(-4, 1, 6)
        plot_validation_curve(
            model, X, y,
            param_name='model__gamma',
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            title='RBF Kernel Width (gamma)',
            ax=axes[0, 1]
        )

        # Learning curve
        model = build_model(model_type, task)
        plot_learning_curve(
            model, X, y,
            cv=cv,
            scoring=scoring,
            title='Learning Curve (SVM)',
            ax=axes[1, 0]
        )

        # Kernel comparison (as text summary)
        axes[1, 1].axis('off')
        summary = """
        SVM Complexity Analysis Summary

        C Parameter:
        - Low C: More regularization, simpler model
        - High C: Less regularization, can overfit

        Gamma Parameter (RBF):
        - Low gamma: Wider influence, smoother boundary
        - High gamma: Localized influence, can overfit

        Kernel Selection:
        - Linear: Fast, high-dimensional data
        - RBF: General purpose, most common
        - Poly: When interactions matter
        """
        axes[1, 1].text(
            0.1, 0.9, summary,
            transform=axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment='top',
            family='monospace'
        )

    else:  # MLP
        # Alpha (L2 regularization)
        model = build_model(model_type, task)
        param_range = np.logspace(-5, 1, 7)
        plot_validation_curve(
            model, X, y,
            param_name='model__alpha',
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            title='L2 Regularization (alpha)',
            ax=axes[0, 0]
        )

        # Learning rate
        model = build_model(model_type, task)
        param_range = np.logspace(-4, -1, 4)
        plot_validation_curve(
            model, X, y,
            param_name='model__learning_rate_init',
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            title='Learning Rate',
            ax=axes[0, 1]
        )

        # Learning curve
        model = build_model(model_type, task)
        plot_learning_curve(
            model, X, y,
            cv=cv,
            scoring=scoring,
            title='Learning Curve (MLP)',
            ax=axes[1, 0]
        )

        # Loss curve
        model = build_model(model_type, task)
        plot_loss_curve(
            model, X, y,
            title='Training Loss Curve',
            ax=axes[1, 1]
        )

    fig.suptitle(f'{model_type.upper()} Complexity Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize learning and validation curves for advanced models'
    )

    parser.add_argument(
        '--model', '-m',
        choices=['svm', 'mlp'],
        default='mlp',
        help='Model type (default: mlp)'
    )
    parser.add_argument(
        '--task', '-t',
        choices=['classification', 'regression'],
        default='classification',
        help='Task type (default: classification)'
    )
    parser.add_argument(
        '--plot', '-p',
        choices=['learning', 'validation', 'loss', 'all'],
        default='learning',
        help='Plot type (default: learning)'
    )
    parser.add_argument(
        '--param',
        type=str,
        default=None,
        help='Parameter name for validation curve (e.g., model__C, model__alpha)'
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='CV folds (default: 5)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples (default: 1000)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path for plot'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for multiple plots'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show plot interactively'
    )

    args = parser.parse_args()

    # Load data
    print(f"Generating {args.n_samples} samples...")
    X, y = load_data(args.task, args.n_samples)

    # Set scoring
    scoring = 'accuracy' if args.task == 'classification' else 'r2'

    # Create output directory if needed
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot == 'all':
        # Generate comprehensive analysis
        print(f"Generating complexity analysis for {args.model.upper()}...")
        fig = plot_complexity_analysis(
            X, y,
            model_type=args.model,
            task=args.task,
            cv=args.cv,
            scoring=scoring
        )

        if args.output:
            fig.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"Saved to: {args.output}")
        elif args.output_dir:
            path = output_dir / f'{args.model}_complexity_analysis.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {path}")

    elif args.plot == 'learning':
        print("Generating learning curve...")
        model = build_model(args.model, args.task)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_learning_curve(
            model, X, y,
            cv=args.cv,
            scoring=scoring,
            title=f'Learning Curve ({args.model.upper()})',
            ax=ax
        )

        if args.output:
            fig.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"Saved to: {args.output}")

    elif args.plot == 'validation':
        if args.param is None:
            # Use default parameter
            if args.model == 'svm':
                args.param = 'model__C'
                param_range = np.logspace(-3, 3, 7)
            else:
                args.param = 'model__alpha'
                param_range = np.logspace(-5, 1, 7)
        else:
            param_range = np.logspace(-3, 3, 7)

        print(f"Generating validation curve for {args.param}...")
        model = build_model(args.model, args.task)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_validation_curve(
            model, X, y,
            param_name=args.param,
            param_range=param_range,
            cv=args.cv,
            scoring=scoring,
            ax=ax
        )

        if args.output:
            fig.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"Saved to: {args.output}")

    elif args.plot == 'loss':
        if args.model != 'mlp':
            print("Loss curve is only available for MLP models")
            sys.exit(1)

        print("Generating loss curve...")
        model = build_model(args.model, args.task)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_loss_curve(model, X, y, ax=ax)

        if args.output:
            fig.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"Saved to: {args.output}")

    if args.show:
        plt.show()
    elif not args.output and not args.output_dir:
        print("\nTip: Use --output to save the plot or --show to display it")


if __name__ == '__main__':
    main()
