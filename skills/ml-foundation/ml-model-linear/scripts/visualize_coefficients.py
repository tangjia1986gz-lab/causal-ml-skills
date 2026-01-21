#!/usr/bin/env python3
"""
Coefficient Visualization CLI

Command-line tool for visualizing linear model coefficients:
- Coefficient bar plots with confidence intervals
- Regularization paths (Lasso, Ridge, Elastic Net)
- Cross-validation error curves
- Variable importance rankings

Usage:
    python visualize_coefficients.py data.csv --outcome y --model lasso --plot-type coef
    python visualize_coefficients.py data.csv --outcome y --model lasso --plot-type path
    python visualize_coefficients.py data.csv --outcome y --model lasso --plot-type cv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd


def load_data(
    filepath: str,
    outcome: str,
    features: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Load and prepare data."""
    df = pd.read_csv(filepath)

    if features:
        feature_cols = features
    else:
        exclude_cols = {outcome}
        if exclude:
            exclude_cols.update(exclude)
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c not in exclude_cols]

    X = df[feature_cols].values
    y = df[outcome].values

    return {
        'X': X,
        'y': y,
        'feature_names': feature_cols,
        'n_samples': len(y),
        'n_features': len(feature_cols)
    }


def plot_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_type: str = 'lasso',
    top_k: int = 20,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8)
) -> None:
    """Plot coefficient bar chart with optional confidence intervals."""
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Fit model
    if model_type == 'lasso':
        model = LassoCV(cv=5, max_iter=10000)
    elif model_type == 'ridge':
        model = RidgeCV(cv=5)
    elif model_type == 'elasticnet':
        model = ElasticNetCV(cv=5, max_iter=10000)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model.fit(X_std, y)
    coef = model.coef_

    # Sort by absolute value
    importance = np.abs(coef)
    sorted_idx = np.argsort(importance)[::-1][:top_k]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(sorted_idx))
    colors = ['steelblue' if c >= 0 else 'indianred' for c in coef[sorted_idx]]

    ax.barh(y_pos, coef[sorted_idx], color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Coefficient Value (standardized)')
    ax.set_title(f'{model_type.capitalize()} Coefficients (top {top_k})')
    ax.invert_yaxis()

    # Add alpha annotation
    if hasattr(model, 'alpha_'):
        ax.text(0.02, 0.98, f'alpha = {model.alpha_:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved coefficient plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_regularization_path(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_type: str = 'lasso',
    top_k: int = 10,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8)
) -> None:
    """Plot regularization path showing coefficients vs lambda."""
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import lasso_path, enet_path, Ridge, LassoCV

    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Compute path
    if model_type == 'lasso':
        alphas, coefs, _ = lasso_path(X_std, y, n_alphas=100)
        title = 'Lasso Regularization Path'
    elif model_type == 'elasticnet':
        alphas, coefs, _ = enet_path(X_std, y, l1_ratio=0.5, n_alphas=100)
        title = 'Elastic Net Regularization Path (l1_ratio=0.5)'
    elif model_type == 'ridge':
        alphas = np.logspace(-4, 4, 100)
        coefs = []
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_std, y)
            coefs.append(ridge.coef_)
        coefs = np.array(coefs).T
        title = 'Ridge Regularization Path'
    else:
        raise ValueError(f"Unknown model: {model_type}")

    # Get CV-optimal alpha for Lasso
    cv_alpha = None
    if model_type == 'lasso':
        lasso_cv = LassoCV(cv=5, max_iter=10000)
        lasso_cv.fit(X_std, y)
        cv_alpha = lasso_cv.alpha_

    # Determine which features to label (top k by max magnitude)
    max_magnitude = np.abs(coefs).max(axis=1)
    top_idx = np.argsort(max_magnitude)[-top_k:]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot all paths (gray for non-top)
    for i in range(coefs.shape[0]):
        if i in top_idx:
            ax.plot(np.log10(alphas), coefs[i], label=feature_names[i], linewidth=2)
        else:
            ax.plot(np.log10(alphas), coefs[i], color='gray', alpha=0.3, linewidth=0.5)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    # Mark CV-optimal alpha
    if cv_alpha is not None:
        ax.axvline(x=np.log10(cv_alpha), color='red', linestyle='--',
                   label=f'CV-optimal (alpha={cv_alpha:.4f})')

    ax.set_xlabel('log10(alpha)')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved regularization path to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_cv_curve(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'lasso',
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """Plot cross-validation error curve."""
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV, ElasticNetCV
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Lasso, Ridge, ElasticNet
    from sklearn.metrics import mean_squared_error

    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    if model_type == 'lasso':
        model = LassoCV(cv=5, max_iter=10000)
        model.fit(X_std, y)
        alphas = model.alphas_
        mse_mean = model.mse_path_.mean(axis=1)
        mse_std = model.mse_path_.std(axis=1)
        best_alpha = model.alpha_
        title = 'Lasso Cross-Validation Error'

    elif model_type == 'elasticnet':
        model = ElasticNetCV(cv=5, max_iter=10000)
        model.fit(X_std, y)
        alphas = model.alphas_
        mse_mean = model.mse_path_.mean(axis=1)
        mse_std = model.mse_path_.std(axis=1)
        best_alpha = model.alpha_
        title = f'Elastic Net CV Error (l1_ratio={model.l1_ratio_:.2f})'

    elif model_type == 'ridge':
        alphas = np.logspace(-4, 4, 50)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_path = np.zeros((len(alphas), 5))

        for i, alpha in enumerate(alphas):
            for j, (train_idx, val_idx) in enumerate(kf.split(X_std)):
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_std[train_idx], y[train_idx])
                pred = ridge.predict(X_std[val_idx])
                mse_path[i, j] = mean_squared_error(y[val_idx], pred)

        mse_mean = mse_path.mean(axis=1)
        mse_std = mse_path.std(axis=1)
        best_idx = np.argmin(mse_mean)
        best_alpha = alphas[best_idx]
        title = 'Ridge Cross-Validation Error'
    else:
        raise ValueError(f"Unknown model: {model_type}")

    # Calculate 1-SE rule threshold
    mse_se = mse_std / np.sqrt(5)
    best_idx = np.argmin(mse_mean)
    threshold = mse_mean[best_idx] + mse_se[best_idx]
    valid_mask = mse_mean <= threshold
    alpha_1se = alphas[valid_mask][0] if model_type != 'ridge' else alphas[np.where(valid_mask)[0][-1]]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(np.log10(alphas), mse_mean, 'b-', linewidth=2, label='Mean CV Error')
    ax.fill_between(np.log10(alphas),
                    mse_mean - mse_se,
                    mse_mean + mse_se,
                    alpha=0.2, color='blue', label='1 SE band')

    # Mark optimal alpha
    ax.axvline(x=np.log10(best_alpha), color='red', linestyle='--',
               label=f'CV-min (alpha={best_alpha:.4f})')
    ax.axvline(x=np.log10(alpha_1se), color='green', linestyle='--',
               label=f'1-SE rule (alpha={alpha_1se:.4f})')
    ax.axhline(y=threshold, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('log10(alpha)')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved CV curve to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_variable_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_type: str = 'lasso',
    n_bootstrap: int = 100,
    top_k: int = 20,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8)
) -> None:
    """Plot variable importance with bootstrap confidence intervals."""
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    n_samples, n_features = X_std.shape
    bootstrap_coefs = np.zeros((n_bootstrap, n_features))

    print(f"Running {n_bootstrap} bootstrap iterations...")

    for b in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot, y_boot = X_std[idx], y[idx]

        # Fit model
        if model_type == 'lasso':
            model = LassoCV(cv=5, max_iter=10000)
        elif model_type == 'ridge':
            model = RidgeCV(cv=5)
        elif model_type == 'elasticnet':
            model = ElasticNetCV(cv=5, max_iter=10000)

        model.fit(X_boot, y_boot)
        bootstrap_coefs[b] = model.coef_

    # Calculate statistics
    coef_mean = bootstrap_coefs.mean(axis=0)
    coef_std = bootstrap_coefs.std(axis=0)
    coef_lower = np.percentile(bootstrap_coefs, 2.5, axis=0)
    coef_upper = np.percentile(bootstrap_coefs, 97.5, axis=0)

    # Selection probability
    selection_prob = (bootstrap_coefs != 0).mean(axis=0)

    # Sort by selection probability * mean absolute coefficient
    importance = selection_prob * np.abs(coef_mean)
    sorted_idx = np.argsort(importance)[::-1][:top_k]

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Coefficient CIs
    ax1 = axes[0]
    y_pos = np.arange(len(sorted_idx))

    ax1.barh(y_pos, coef_mean[sorted_idx], xerr=[coef_mean[sorted_idx] - coef_lower[sorted_idx],
                                                   coef_upper[sorted_idx] - coef_mean[sorted_idx]],
             color='steelblue', alpha=0.7, capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Coefficient (95% Bootstrap CI)')
    ax1.set_title('Coefficient Estimates')
    ax1.invert_yaxis()

    # Right: Selection probability
    ax2 = axes[1]
    colors = plt.cm.RdYlGn(selection_prob[sorted_idx])
    ax2.barh(y_pos, selection_prob[sorted_idx], color=colors, edgecolor='black', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax2.set_xlabel('Selection Probability')
    ax2.set_title('Bootstrap Selection Stability')
    ax2.set_xlim(0, 1)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.invert_yaxis()

    plt.suptitle(f'{model_type.capitalize()} Variable Importance ({n_bootstrap} bootstrap samples)')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved variable importance plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_comparison(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_k: int = 15,
    output_path: Optional[str] = None,
    figsize: tuple = (14, 8)
) -> None:
    """Compare coefficients across OLS, Ridge, Lasso, and Elastic Net."""
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Fit all models
    models = {
        'OLS': LinearRegression(),
        'Ridge': RidgeCV(cv=5),
        'Lasso': LassoCV(cv=5, max_iter=10000),
        'Elastic Net': ElasticNetCV(cv=5, max_iter=10000)
    }

    coefs = {}
    for name, model in models.items():
        model.fit(X_std, y)
        coefs[name] = model.coef_

    # Select top features by max absolute coefficient across models
    max_coef = np.max([np.abs(c) for c in coefs.values()], axis=0)
    sorted_idx = np.argsort(max_coef)[::-1][:top_k]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(sorted_idx))
    width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (name, coef) in enumerate(coefs.items()):
        ax.bar(x_pos + i * width, coef[sorted_idx], width,
               label=name, color=colors[i], alpha=0.8)

    ax.set_xticks(x_pos + 1.5 * width)
    ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Coefficient Value (standardized)')
    ax.set_title('Coefficient Comparison Across Regularization Methods')
    ax.legend()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize linear model coefficients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Coefficient bar plot
  python visualize_coefficients.py data.csv --outcome y --plot-type coef --model lasso

  # Regularization path
  python visualize_coefficients.py data.csv --outcome y --plot-type path --model lasso

  # CV error curve
  python visualize_coefficients.py data.csv --outcome y --plot-type cv --model lasso

  # Variable importance with bootstrap
  python visualize_coefficients.py data.csv --outcome y --plot-type importance --bootstrap 200

  # Compare all models
  python visualize_coefficients.py data.csv --outcome y --plot-type compare

  # Save to file
  python visualize_coefficients.py data.csv --outcome y --plot-type path --output path.png
        """
    )

    parser.add_argument('data', help='Path to CSV data file')
    parser.add_argument('--outcome', '-y', required=True, help='Outcome variable name')
    parser.add_argument('--features', '-X', nargs='+', help='Feature variable names')
    parser.add_argument('--all-features', action='store_true', help='Use all numeric columns')
    parser.add_argument('--exclude', nargs='+', help='Variables to exclude')
    parser.add_argument('--model', '-m', choices=['lasso', 'ridge', 'elasticnet'],
                       default='lasso', help='Model type')
    parser.add_argument('--plot-type', '-p',
                       choices=['coef', 'path', 'cv', 'importance', 'compare'],
                       default='coef', help='Type of plot')
    parser.add_argument('--top-k', type=int, default=20, help='Number of top features to show')
    parser.add_argument('--bootstrap', type=int, default=100, help='Bootstrap iterations (for importance)')
    parser.add_argument('--output', '-o', help='Output file path (shows plot if not specified)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8], help='Figure size')

    args = parser.parse_args()

    # Check matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg' if args.output else 'TkAgg')
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        sys.exit(1)

    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(
        args.data,
        args.outcome,
        args.features if not args.all_features else None,
        args.exclude
    )

    X = data['X']
    y = data['y']
    feature_names = data['feature_names']

    print(f"Data: {data['n_samples']} samples, {data['n_features']} features")

    figsize = tuple(args.figsize)

    # Create requested plot
    if args.plot_type == 'coef':
        plot_coefficients(X, y, feature_names, args.model, args.top_k,
                         args.output, figsize)
    elif args.plot_type == 'path':
        plot_regularization_path(X, y, feature_names, args.model, args.top_k,
                                 args.output, figsize)
    elif args.plot_type == 'cv':
        plot_cv_curve(X, y, args.model, args.output, figsize)
    elif args.plot_type == 'importance':
        plot_variable_importance(X, y, feature_names, args.model, args.bootstrap,
                                args.top_k, args.output, figsize)
    elif args.plot_type == 'compare':
        plot_comparison(X, y, feature_names, args.top_k, args.output, figsize)


if __name__ == '__main__':
    main()
