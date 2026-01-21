#!/usr/bin/env python3
"""
Cross-Fitting Diagnostic Plots for DDML

Generates diagnostic visualizations for assessing cross-fitting quality:
- Fold-level estimate variation
- Residual distributions
- Propensity score overlap
- Nuisance model performance

Usage:
    python cross_fit_diagnostics.py --data data.csv --outcome y --treatment d --controls "x1,x2"
    python cross_fit_diagnostics.py --data data.csv --outcome y --treatment d --output diagnostics/

Reference:
    Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# Check for matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will not be generated.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate cross-fitting diagnostics for DDML"
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
    parser.add_argument('--learner', '-l', default='lasso',
                        help='Learner for nuisance models (default: lasso)')
    parser.add_argument('--n-folds', '-k', type=int, default=5,
                        help='Number of cross-fitting folds (default: 5)')
    parser.add_argument('--n-rep', '-r', type=int, default=10,
                        help='Repetitions for stability analysis (default: 10)')
    parser.add_argument('--output', '-o', default='.',
                        help='Output directory for plots')
    parser.add_argument('--format', default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Plot format (default: png)')

    return parser.parse_args()


def get_learner(learner_name: str, task: str = 'regression'):
    """Get sklearn learner by name."""
    from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegressionCV
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    learners = {
        'regression': {
            'lasso': LassoCV(cv=5),
            'ridge': RidgeCV(cv=5),
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1)
        },
        'classification': {
            'lasso': LogisticRegressionCV(cv=5, penalty='l1', solver='saga', max_iter=5000),
            'ridge': LogisticRegressionCV(cv=5, penalty='l2', max_iter=5000),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
        }
    }

    return learners.get(task, {}).get(learner_name)


def compute_cross_fit_predictions(X, y, learner_name: str, n_folds: int = 5,
                                   task: str = 'regression'):
    """Compute cross-validated predictions."""
    learner = get_learner(learner_name, task)
    if learner is None:
        raise ValueError(f"Unknown learner: {learner_name}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', learner)
    ])

    if task == 'classification':
        y_pred = cross_val_predict(pipeline, X, y, cv=n_folds, method='predict_proba')[:, 1]
    else:
        y_pred = cross_val_predict(pipeline, X, y, cv=n_folds)

    return y_pred


def compute_fold_estimates(X, y, d, learner_name: str, n_folds: int = 5):
    """Compute treatment effect estimate for each fold."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_estimates = []
    fold_ses = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        d_train, d_test = d[train_idx], d[test_idx]

        # Fit nuisance models on training fold
        learner_y = get_learner(learner_name, 'regression')
        learner_d = get_learner(learner_name, 'regression')

        pipeline_y = Pipeline([('scaler', StandardScaler()), ('model', learner_y)])
        pipeline_d = Pipeline([('scaler', StandardScaler()), ('model', learner_d)])

        pipeline_y.fit(X_train, y_train)
        pipeline_d.fit(X_train, d_train)

        # Predict on test fold
        y_pred = pipeline_y.predict(X_test)
        d_pred = pipeline_d.predict(X_test)

        # Compute residuals
        y_resid = y_test - y_pred
        d_resid = d_test - d_pred

        # Estimate theta for this fold
        theta = np.sum(y_resid * d_resid) / np.sum(d_resid**2)

        # SE for this fold
        psi = (y_resid - theta * d_resid) * d_resid
        var = np.mean(psi**2) / (np.mean(d_resid**2)**2)
        se = np.sqrt(var / len(test_idx))

        fold_estimates.append(theta)
        fold_ses.append(se)

    return np.array(fold_estimates), np.array(fold_ses)


def compute_repetition_estimates(X, y, d, learner_name: str,
                                  n_folds: int = 5, n_rep: int = 10):
    """Compute estimates across multiple random fold assignments."""
    estimates = []

    for rep in range(n_rep):
        np.random.seed(rep * 42)
        perm = np.random.permutation(len(y))
        fold_idx = perm % n_folds

        y_pred = np.zeros_like(y)
        d_pred = np.zeros_like(d)

        for k in range(n_folds):
            train_mask = fold_idx != k
            test_mask = fold_idx == k

            learner_y = get_learner(learner_name, 'regression')
            learner_d = get_learner(learner_name, 'regression')

            pipeline_y = Pipeline([('scaler', StandardScaler()), ('model', learner_y)])
            pipeline_d = Pipeline([('scaler', StandardScaler()), ('model', learner_d)])

            pipeline_y.fit(X[train_mask], y[train_mask])
            pipeline_d.fit(X[train_mask], d[train_mask])

            y_pred[test_mask] = pipeline_y.predict(X[test_mask])
            d_pred[test_mask] = pipeline_d.predict(X[test_mask])

        y_resid = y - y_pred
        d_resid = d - d_pred
        theta = np.sum(y_resid * d_resid) / np.sum(d_resid**2)
        estimates.append(theta)

    return np.array(estimates)


def plot_fold_variation(fold_estimates, fold_ses, output_dir: Path, fmt: str = 'png'):
    """Plot estimate variation across folds."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Estimates by fold
    ax1 = axes[0]
    folds = range(1, len(fold_estimates) + 1)
    ax1.errorbar(folds, fold_estimates,
                 yerr=1.96 * fold_ses,
                 fmt='o', capsize=5, capthick=2, markersize=8)
    ax1.axhline(np.mean(fold_estimates), color='red', linestyle='--',
                label=f'Mean: {np.mean(fold_estimates):.4f}')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Treatment Effect Estimate')
    ax1.set_title('Estimates by Cross-Fitting Fold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Distribution
    ax2 = axes[1]
    ax2.hist(fold_estimates, bins=max(3, len(fold_estimates)//2),
             edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(fold_estimates), color='red', linestyle='--',
                label=f'Mean: {np.mean(fold_estimates):.4f}')
    ax2.set_xlabel('Treatment Effect Estimate')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Fold Estimates')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'fold_variation.{fmt}', dpi=150, bbox_inches='tight')
    plt.close()


def plot_repetition_stability(rep_estimates, output_dir: Path, fmt: str = 'png'):
    """Plot stability across repetitions."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Estimates over repetitions
    ax1 = axes[0]
    ax1.plot(range(1, len(rep_estimates) + 1), rep_estimates, 'o-', markersize=6)
    ax1.axhline(np.mean(rep_estimates), color='red', linestyle='--',
                label=f'Mean: {np.mean(rep_estimates):.4f}')
    ax1.fill_between(range(1, len(rep_estimates) + 1),
                     np.mean(rep_estimates) - np.std(rep_estimates),
                     np.mean(rep_estimates) + np.std(rep_estimates),
                     alpha=0.2, color='red')
    ax1.set_xlabel('Repetition')
    ax1.set_ylabel('Treatment Effect Estimate')
    ax1.set_title('Estimates Across Repetitions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Histogram
    ax2 = axes[1]
    ax2.hist(rep_estimates, bins=15, edgecolor='black', alpha=0.7, density=True)

    # Fit normal
    mu, sigma = np.mean(rep_estimates), np.std(rep_estimates)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal fit')
    ax2.axvline(mu, color='red', linestyle='--')
    ax2.set_xlabel('Treatment Effect Estimate')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Distribution (CV = {sigma/abs(mu)*100:.1f}%)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'repetition_stability.{fmt}', dpi=150, bbox_inches='tight')
    plt.close()


def plot_residuals(y, d, y_pred, d_pred, output_dir: Path, fmt: str = 'png'):
    """Plot residual diagnostics."""
    if not HAS_MATPLOTLIB:
        return

    y_resid = y - y_pred
    d_resid = d - d_pred

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Outcome residuals
    ax1 = axes[0, 0]
    ax1.hist(y_resid, bins=50, edgecolor='black', alpha=0.7, density=True)
    x = np.linspace(y_resid.min(), y_resid.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, 0, y_resid.std()), 'r-', lw=2)
    ax1.set_xlabel('Residual')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Outcome Residuals (Y - E[Y|X])\nStd = {y_resid.std():.4f}')

    # Treatment residuals
    ax2 = axes[0, 1]
    ax2.hist(d_resid, bins=50, edgecolor='black', alpha=0.7, density=True)
    x = np.linspace(d_resid.min(), d_resid.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, 0, d_resid.std()), 'r-', lw=2)
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Treatment Residuals (D - E[D|X])\nStd = {d_resid.std():.4f}')

    # Residual scatter
    ax3 = axes[1, 0]
    ax3.scatter(d_resid, y_resid, alpha=0.3, s=10)
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.axvline(0, color='gray', linestyle='--')

    # Add regression line
    slope = np.sum(y_resid * d_resid) / np.sum(d_resid**2)
    x_line = np.array([d_resid.min(), d_resid.max()])
    ax3.plot(x_line, slope * x_line, 'r-', lw=2, label=f'Slope = {slope:.4f}')
    ax3.set_xlabel('Treatment Residual')
    ax3.set_ylabel('Outcome Residual')
    ax3.set_title('Residual Scatter (slope = treatment effect)')
    ax3.legend()

    # Q-Q plot for outcome residuals
    ax4 = axes[1, 1]
    stats.probplot(y_resid, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Outcome Residuals)')

    plt.tight_layout()
    plt.savefig(output_dir / f'residuals.{fmt}', dpi=150, bbox_inches='tight')
    plt.close()


def plot_propensity_overlap(d, propensity, output_dir: Path, fmt: str = 'png'):
    """Plot propensity score distribution for binary treatment."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram by group
    ax1 = axes[0]
    ax1.hist(propensity[d == 0], bins=30, alpha=0.6, label='Control', density=True)
    ax1.hist(propensity[d == 1], bins=30, alpha=0.6, label='Treated', density=True)
    ax1.set_xlabel('Propensity Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Propensity Score Distribution')
    ax1.legend()

    # Overlap statistics
    n_low = np.sum(propensity < 0.05)
    n_high = np.sum(propensity > 0.95)
    ax1.axvline(0.05, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(0.95, color='red', linestyle='--', alpha=0.5)

    # Box plot by group
    ax2 = axes[1]
    ax2.boxplot([propensity[d == 0], propensity[d == 1]],
                labels=['Control', 'Treated'])
    ax2.set_ylabel('Propensity Score')
    ax2.set_title(f'Propensity by Group\n{n_low} below 0.05, {n_high} above 0.95')

    plt.tight_layout()
    plt.savefig(output_dir / f'propensity_overlap.{fmt}', dpi=150, bbox_inches='tight')
    plt.close()


def plot_nuisance_performance(y, y_pred, d, d_pred, output_dir: Path, fmt: str = 'png'):
    """Plot nuisance model performance."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Outcome model
    ax1 = axes[0]
    ax1.scatter(y_pred, y, alpha=0.3, s=10)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax1.plot(lims, lims, 'r--', label='Perfect fit')
    r2_y = 1 - np.var(y - y_pred) / np.var(y)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title(f'Outcome Model E[Y|X]\nR-squared = {r2_y:.4f}')
    ax1.legend()

    # Treatment model
    ax2 = axes[1]
    ax2.scatter(d_pred, d, alpha=0.3, s=10)
    lims = [min(d.min(), d_pred.min()), max(d.max(), d_pred.max())]
    ax2.plot(lims, lims, 'r--', label='Perfect fit')
    r2_d = 1 - np.var(d - d_pred) / np.var(d)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title(f'Treatment Model E[D|X]\nR-squared = {r2_d:.4f}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'nuisance_performance.{fmt}', dpi=150, bbox_inches='tight')
    plt.close()


def generate_diagnostic_summary(fold_estimates, fold_ses, rep_estimates,
                                y, y_pred, d, d_pred, output_dir: Path):
    """Generate text summary of diagnostics."""
    y_resid = y - y_pred
    d_resid = d - d_pred

    r2_y = 1 - np.var(y_resid) / np.var(y)
    r2_d = 1 - np.var(d_resid) / np.var(d)

    theta = np.sum(y_resid * d_resid) / np.sum(d_resid**2)
    psi = (y_resid - theta * d_resid) * d_resid
    se = np.sqrt(np.mean(psi**2) / (np.mean(d_resid**2)**2) / len(y))

    summary = f"""# DDML Cross-Fitting Diagnostics

Generated: {datetime.now().isoformat()}

## Main Estimate

- **Treatment Effect**: {theta:.6f}
- **Standard Error**: {se:.6f}
- **95% CI**: [{theta - 1.96*se:.6f}, {theta + 1.96*se:.6f}]

## Fold-Level Analysis

- **Number of Folds**: {len(fold_estimates)}
- **Mean Estimate**: {np.mean(fold_estimates):.6f}
- **Std Across Folds**: {np.std(fold_estimates):.6f}
- **CV (fold)**: {np.std(fold_estimates)/abs(np.mean(fold_estimates))*100:.2f}%
- **Range**: [{min(fold_estimates):.6f}, {max(fold_estimates):.6f}]

## Repetition Stability

- **Number of Repetitions**: {len(rep_estimates)}
- **Mean Estimate**: {np.mean(rep_estimates):.6f}
- **Std Across Reps**: {np.std(rep_estimates):.6f}
- **CV (rep)**: {np.std(rep_estimates)/abs(np.mean(rep_estimates))*100:.2f}%
- **Range**: [{min(rep_estimates):.6f}, {max(rep_estimates):.6f}]

## Nuisance Model Quality

### Outcome Model E[Y|X]
- **R-squared**: {r2_y:.4f}
- **Residual Std**: {np.std(y_resid):.4f}
- **Residual Mean**: {np.mean(y_resid):.6f}

### Treatment Model E[D|X]
- **R-squared**: {r2_d:.4f}
- **Residual Std**: {np.std(d_resid):.4f}
- **Residual Mean**: {np.mean(d_resid):.6f}

## Diagnostic Assessment

"""
    # Add assessments
    fold_cv = np.std(fold_estimates)/abs(np.mean(fold_estimates))
    rep_cv = np.std(rep_estimates)/abs(np.mean(rep_estimates))

    if fold_cv < 0.1 and rep_cv < 0.1:
        summary += "- **Cross-fitting stability**: EXCELLENT\n"
    elif fold_cv < 0.2 and rep_cv < 0.2:
        summary += "- **Cross-fitting stability**: GOOD\n"
    else:
        summary += "- **Cross-fitting stability**: MODERATE - consider more data or different learner\n"

    if r2_y > 0.3:
        summary += "- **Outcome model fit**: GOOD\n"
    elif r2_y > 0.1:
        summary += "- **Outcome model fit**: MODERATE\n"
    else:
        summary += "- **Outcome model fit**: LOW (may be fine if X doesn't predict Y strongly)\n"

    if r2_d > 0.3:
        summary += "- **Treatment model fit**: GOOD\n"
    elif r2_d > 0.1:
        summary += "- **Treatment model fit**: MODERATE\n"
    else:
        summary += "- **Treatment model fit**: LOW (may indicate weak selection on observables)\n"

    with open(output_dir / 'diagnostics_summary.md', 'w') as f:
        f.write(summary)


def main():
    args = parse_args()

    print("=" * 60)
    print("CROSS-FITTING DIAGNOSTICS FOR DDML")
    print("=" * 60)

    # Load data
    data = pd.read_csv(args.data)

    # Get controls
    if args.controls:
        controls = [c.strip() for c in args.controls.split(',')]
    elif args.all_controls:
        controls = [c for c in data.columns if c not in [args.outcome, args.treatment]]
    else:
        controls = [c for c in data.columns if c not in [args.outcome, args.treatment]]

    # Prepare data
    df = data[[args.outcome, args.treatment] + controls].dropna()
    X = df[controls].values
    y = df[args.outcome].values
    d = df[args.treatment].values

    print(f"\nData: {len(df)} observations, {len(controls)} controls")
    print(f"Learner: {args.learner}")
    print(f"Folds: {args.n_folds}, Repetitions: {args.n_rep}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute cross-validated predictions
    print("\nComputing cross-validated predictions...")
    y_pred = compute_cross_fit_predictions(X, y, args.learner, args.n_folds, 'regression')
    d_pred = compute_cross_fit_predictions(X, d, args.learner, args.n_folds, 'regression')

    # Compute fold estimates
    print("Computing fold-level estimates...")
    fold_estimates, fold_ses = compute_fold_estimates(X, y, d, args.learner, args.n_folds)

    # Compute repetition estimates
    print("Computing repetition stability...")
    rep_estimates = compute_repetition_estimates(X, y, d, args.learner, args.n_folds, args.n_rep)

    # Generate plots
    if HAS_MATPLOTLIB:
        print("\nGenerating diagnostic plots...")
        plot_fold_variation(fold_estimates, fold_ses, output_dir, args.format)
        plot_repetition_stability(rep_estimates, output_dir, args.format)
        plot_residuals(y, d, y_pred, d_pred, output_dir, args.format)
        plot_nuisance_performance(y, y_pred, d, d_pred, output_dir, args.format)

        # Propensity plot for binary treatment
        if len(np.unique(d)) == 2:
            propensity = compute_cross_fit_predictions(X, d, args.learner, args.n_folds, 'classification')
            plot_propensity_overlap(d, propensity, output_dir, args.format)

    # Generate summary
    print("Generating summary report...")
    generate_diagnostic_summary(fold_estimates, fold_ses, rep_estimates,
                               y, y_pred, d, d_pred, output_dir)

    print(f"\nDiagnostics saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
