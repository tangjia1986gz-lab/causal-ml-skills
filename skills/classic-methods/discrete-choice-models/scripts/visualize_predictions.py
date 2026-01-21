#!/usr/bin/env python3
"""
Visualization Tools for Discrete Choice Models

Creates plots for:
- Predicted probabilities across covariate values
- Marginal effect plots
- ROC curves and calibration plots (binary)
- Category probability plots (ordered/multinomial)
- Predicted vs observed (count)

Usage:
    python visualize_predictions.py --data data.csv --y outcome --x "var1 var2" --model logit --plot probabilities --vary var1
    python visualize_predictions.py --data data.csv --y outcome --x "var1 var2" --model logit --plot marginal --output mfx_plot.png
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import (
    Logit, Probit, Poisson, NegativeBinomial, MNLogit
)
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize discrete choice model predictions'
    )

    parser.add_argument('--data', '-d', required=True, help='Path to CSV data')
    parser.add_argument('--y', required=True, help='Outcome variable')
    parser.add_argument('--x', required=True, help='Covariates (space-separated)')
    parser.add_argument('--model', '-m', required=True,
                        choices=['logit', 'probit', 'ologit', 'oprobit', 'mlogit', 'poisson', 'negbin'],
                        help='Model type')
    parser.add_argument('--plot', '-p', required=True,
                        choices=['probabilities', 'marginal', 'roc', 'calibration',
                                 'categories', 'predicted', 'all'],
                        help='Type of plot')
    parser.add_argument('--vary', help='Variable to vary for probability plot')
    parser.add_argument('--treatment', '-t', help='Treatment variable for effect plots')
    parser.add_argument('--output', '-o', help='Output file (PNG)')
    parser.add_argument('--dpi', type=int, default=150, help='Plot DPI')
    parser.add_argument('--figsize', default='10,6', help='Figure size (width,height)')

    return parser.parse_args()


def load_and_fit(data_path: str, y_var: str, x_vars: List[str], model_type: str) -> Tuple:
    """Load data and fit model."""
    df = pd.read_csv(data_path)
    y = df[y_var].values
    X = df[x_vars].values

    # Drop missing
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y = y[mask]
    X = X[mask]
    df_clean = df.loc[mask, [y_var] + x_vars].reset_index(drop=True)

    X_const = sm.add_constant(X)

    # Fit model
    if model_type == 'logit':
        model = Logit(y, X_const).fit(disp=0)
    elif model_type == 'probit':
        model = Probit(y, X_const).fit(disp=0)
    elif model_type == 'ologit':
        model = OrderedModel(y, X, distr='logit').fit(method='bfgs', disp=0)
    elif model_type == 'oprobit':
        model = OrderedModel(y, X, distr='probit').fit(method='bfgs', disp=0)
    elif model_type == 'mlogit':
        model = MNLogit(y, X_const).fit(disp=0)
    elif model_type == 'poisson':
        model = Poisson(y, X_const).fit(disp=0)
    elif model_type == 'negbin':
        model = NegativeBinomial(y, X_const).fit(disp=0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return y, X, X_const, model, df_clean, x_vars


# =============================================================================
# Binary Model Plots
# =============================================================================

def plot_predicted_probabilities(model, X: np.ndarray, X_const: np.ndarray,
                                 vary_var: str, var_names: List[str],
                                 model_type: str, ax=None) -> plt.Figure:
    """
    Plot predicted probability as function of one variable,
    holding others at their means.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    vary_idx = var_names.index(vary_var)
    vary_col = vary_idx + 1  # +1 for constant

    # Create range of values
    x_range = np.linspace(X[:, vary_idx].min(), X[:, vary_idx].max(), 100)

    # Create prediction matrix (others at mean)
    X_pred = np.zeros((len(x_range), X_const.shape[1]))
    X_pred[:, 0] = 1  # constant
    X_pred[:, 1:] = np.mean(X, axis=0)
    X_pred[:, vary_col] = x_range

    # Predict
    if model_type in ['logit', 'probit']:
        probs = model.predict(X_pred)
        ax.plot(x_range, probs, color=COLORS[0], linewidth=2, label='Predicted P(Y=1)')

        # Add confidence band (approximate)
        se = np.sqrt(np.diag(X_pred @ model.cov_params() @ X_pred.T))
        if model_type == 'logit':
            from scipy.special import expit
            linear = X_pred @ model.params
            lower = expit(linear - 1.96 * se)
            upper = expit(linear + 1.96 * se)
        else:
            linear = X_pred @ model.params
            lower = stats.norm.cdf(linear - 1.96 * se)
            upper = stats.norm.cdf(linear + 1.96 * se)

        ax.fill_between(x_range, lower, upper, alpha=0.2, color=COLORS[0])

        ax.set_ylabel('P(Y = 1)')
        ax.set_ylim(-0.05, 1.05)

    elif model_type in ['poisson', 'negbin']:
        mu = model.predict(X_pred)
        ax.plot(x_range, mu, color=COLORS[0], linewidth=2, label='Expected Count')

        # Confidence band
        se = np.sqrt(np.diag(X_pred @ model.cov_params() @ X_pred.T))
        linear = X_pred @ model.params
        lower = np.exp(linear - 1.96 * se)
        upper = np.exp(linear + 1.96 * se)
        ax.fill_between(x_range, lower, upper, alpha=0.2, color=COLORS[0])

        ax.set_ylabel('E[Y]')

    ax.set_xlabel(vary_var)
    ax.set_title(f'Predicted Values by {vary_var}\n(other variables at mean)')
    ax.legend()

    return fig


def plot_marginal_effects(model, X: np.ndarray, var_names: List[str],
                          model_type: str, ax=None) -> plt.Figure:
    """
    Bar plot of average marginal effects with confidence intervals.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if model_type in ['logit', 'probit']:
        mfx = model.get_margeff(at='overall')
        ame = mfx.margeff
        se = mfx.margeff_se
    elif model_type in ['poisson', 'negbin']:
        mu_mean = np.mean(model.fittedvalues)
        ame = model.params[1:] * mu_mean
        se = model.bse[1:] * mu_mean
    else:
        ame = model.params[1:] if hasattr(model, 'params') else np.zeros(len(var_names))
        se = model.bse[1:] if hasattr(model, 'bse') else np.zeros(len(var_names))

    # Create bar plot
    positions = np.arange(len(var_names))
    colors = [COLORS[0] if a >= 0 else COLORS[1] for a in ame]

    bars = ax.barh(positions, ame, xerr=1.96*se, color=colors,
                   capsize=3, alpha=0.7, edgecolor='black')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels(var_names)
    ax.set_xlabel('Average Marginal Effect')
    ax.set_title('Average Marginal Effects (95% CI)')

    # Add significance markers
    for i, (a, s) in enumerate(zip(ame, se)):
        if abs(a) > 1.96 * s:
            ax.annotate('*', (a + 1.96*s + 0.01*max(abs(ame)), i),
                        fontsize=14, fontweight='bold')

    return fig


def plot_roc_curve(model, y: np.ndarray, X_const: np.ndarray, ax=None) -> plt.Figure:
    """
    ROC curve for binary classification.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    probs = model.predict(X_const)
    fpr, tpr, thresholds = roc_curve(y, probs)
    auc = roc_auc_score(y, probs)

    ax.plot(fpr, tpr, color=COLORS[2], linewidth=2,
            label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
            label='Random classifier')

    # Mark optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    ax.scatter(fpr[best_idx], tpr[best_idx], color=COLORS[1], s=100, zorder=5,
               label=f'Optimal threshold = {thresholds[best_idx]:.3f}')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    return fig


def plot_calibration(model, y: np.ndarray, X_const: np.ndarray,
                     n_bins: int = 10, ax=None) -> plt.Figure:
    """
    Calibration plot: observed vs predicted probabilities.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    probs = model.predict(X_const)

    # Bin predictions
    df = pd.DataFrame({'y': y, 'p': probs})
    df['bin'] = pd.qcut(df['p'], n_bins, labels=False, duplicates='drop')

    calibration = df.groupby('bin').agg({
        'y': 'mean',
        'p': 'mean'
    }).reset_index()

    ax.scatter(calibration['p'], calibration['y'], s=100, color=COLORS[2],
               edgecolor='black', zorder=5, label='Observed')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
            label='Perfect calibration')

    # Add error bars (binomial SE)
    bin_counts = df.groupby('bin').size()
    se = np.sqrt(calibration['y'] * (1 - calibration['y']) / bin_counts.values)
    ax.errorbar(calibration['p'], calibration['y'], yerr=1.96*se,
                fmt='none', color=COLORS[2], capsize=3)

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Probability')
    ax.set_title('Calibration Plot')
    ax.legend()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    return fig


# =============================================================================
# Ordered/Multinomial Plots
# =============================================================================

def plot_category_probabilities(model, X: np.ndarray, vary_var: str,
                                var_names: List[str], model_type: str,
                                ax=None) -> plt.Figure:
    """
    Plot probabilities for each category as function of one variable.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    vary_idx = var_names.index(vary_var)

    # Create range
    x_range = np.linspace(X[:, vary_idx].min(), X[:, vary_idx].max(), 100)

    # Prediction matrix
    X_pred = np.zeros((len(x_range), X.shape[1]))
    X_pred[:] = np.mean(X, axis=0)
    X_pred[:, vary_idx] = x_range

    # Predict probabilities
    if model_type in ['ologit', 'oprobit']:
        probs = model.predict(X_pred)
    elif model_type == 'mlogit':
        X_pred_const = sm.add_constant(X_pred)
        probs = model.predict(X_pred_const)
    else:
        raise ValueError(f"Not applicable for {model_type}")

    n_cat = probs.shape[1]

    # Stacked area plot
    for j in range(n_cat):
        label = f'Category {j}'
        ax.fill_between(x_range,
                        probs[:, :j].sum(axis=1) if j > 0 else np.zeros(len(x_range)),
                        probs[:, :j+1].sum(axis=1),
                        alpha=0.7, color=COLORS[j % len(COLORS)], label=label)

    ax.set_xlabel(vary_var)
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Category Probabilities by {vary_var}')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim(0, 1)

    return fig


def plot_category_lines(model, X: np.ndarray, vary_var: str,
                        var_names: List[str], model_type: str,
                        ax=None) -> plt.Figure:
    """
    Line plot of probabilities for each category.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    vary_idx = var_names.index(vary_var)
    x_range = np.linspace(X[:, vary_idx].min(), X[:, vary_idx].max(), 100)

    X_pred = np.zeros((len(x_range), X.shape[1]))
    X_pred[:] = np.mean(X, axis=0)
    X_pred[:, vary_idx] = x_range

    if model_type in ['ologit', 'oprobit']:
        probs = model.predict(X_pred)
    elif model_type == 'mlogit':
        X_pred_const = sm.add_constant(X_pred)
        probs = model.predict(X_pred_const)
    else:
        raise ValueError(f"Not applicable for {model_type}")

    n_cat = probs.shape[1]

    for j in range(n_cat):
        ax.plot(x_range, probs[:, j], color=COLORS[j % len(COLORS)],
                linewidth=2, label=f'P(Y={j})')

    ax.set_xlabel(vary_var)
    ax.set_ylabel('Probability')
    ax.set_title(f'Category Probabilities by {vary_var}')
    ax.legend(loc='best')
    ax.set_ylim(-0.02, 1.02)

    return fig


# =============================================================================
# Count Model Plots
# =============================================================================

def plot_predicted_vs_observed(model, y: np.ndarray, X_const: np.ndarray,
                               ax=None) -> plt.Figure:
    """
    Plot predicted vs observed counts.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    mu = model.fittedvalues

    # Scatter plot with jitter
    jitter = np.random.normal(0, 0.1, len(y))
    ax.scatter(y + jitter, mu, alpha=0.3, color=COLORS[2], s=30)

    # 45-degree line
    max_val = max(y.max(), mu.max())
    ax.plot([0, max_val], [0, max_val], color='gray', linestyle='--',
            linewidth=1, label='Perfect prediction')

    # Add smoothed line
    from scipy.ndimage import uniform_filter1d
    sorted_idx = np.argsort(y)
    y_sorted = y[sorted_idx]
    mu_sorted = mu[sorted_idx]

    # Bin and average
    bins = np.unique(y_sorted)
    means = [mu_sorted[y_sorted == b].mean() for b in bins]
    ax.plot(bins, means, color=COLORS[1], linewidth=2, label='Mean prediction')

    ax.set_xlabel('Observed Count')
    ax.set_ylabel('Predicted Count')
    ax.set_title('Predicted vs Observed Counts')
    ax.legend()

    return fig


def plot_rootogram(model, y: np.ndarray, X_const: np.ndarray, ax=None) -> plt.Figure:
    """
    Hanging rootogram for count model diagnostics.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    mu = model.fittedvalues

    # Observed frequencies
    max_count = min(int(y.max()), 20)
    counts = np.arange(max_count + 1)
    observed = np.array([np.sum(y == c) for c in counts])

    # Expected frequencies (Poisson)
    expected = np.sum([stats.poisson.pmf(counts, m) for m in mu], axis=0)

    # Square root transformation
    sqrt_obs = np.sqrt(observed)
    sqrt_exp = np.sqrt(expected)

    # Hanging rootogram
    ax.bar(counts, sqrt_exp, width=0.8, color='lightgray', edgecolor='gray',
           label='Expected (sqrt)')
    ax.bar(counts, sqrt_exp - sqrt_obs, bottom=sqrt_obs, width=0.8,
           color=COLORS[1], alpha=0.5, label='Deviation')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Count')
    ax.set_ylabel('Square Root of Frequency')
    ax.set_title('Hanging Rootogram')
    ax.legend()

    return fig


# =============================================================================
# Treatment Effect Plots
# =============================================================================

def plot_treatment_effect_by_covariate(model, X: np.ndarray, X_const: np.ndarray,
                                        treatment_var: str, vary_var: str,
                                        var_names: List[str], model_type: str,
                                        ax=None) -> plt.Figure:
    """
    Plot treatment effect as function of another covariate.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    treat_idx = var_names.index(treatment_var) + 1  # +1 for constant
    vary_idx = var_names.index(vary_var) + 1

    # Range of varying variable
    x_range = np.linspace(X[:, vary_idx-1].min(), X[:, vary_idx-1].max(), 50)

    effects = []
    for x_val in x_range:
        # Create X at this value (others at mean)
        X_temp = np.mean(X_const, axis=0).reshape(1, -1).repeat(2, axis=0)
        X_temp[0, treat_idx] = 0  # Control
        X_temp[1, treat_idx] = 1  # Treated
        X_temp[:, vary_idx] = x_val

        if model_type in ['logit', 'probit']:
            p0, p1 = model.predict(X_temp)
            effect = p1 - p0
        elif model_type in ['poisson', 'negbin']:
            mu0, mu1 = model.predict(X_temp)
            effect = mu1 - mu0
        else:
            effect = 0

        effects.append(effect)

    ax.plot(x_range, effects, color=COLORS[0], linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    ax.set_xlabel(vary_var)
    ax.set_ylabel(f'Treatment Effect ({treatment_var})')
    ax.set_title(f'Heterogeneous Treatment Effect by {vary_var}')

    # Add average effect line
    avg_effect = np.mean(effects)
    ax.axhline(y=avg_effect, color=COLORS[1], linestyle=':', linewidth=1,
               label=f'Average effect = {avg_effect:.4f}')
    ax.legend()

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    x_vars = args.x.split()
    figsize = tuple(map(float, args.figsize.split(',')))

    # Load and fit
    y, X, X_const, model, df, var_names = load_and_fit(
        args.data, args.y, x_vars, args.model
    )

    print(f"Fitted {args.model} model with N = {len(y)}")

    figures = []

    # Create plots based on model type and request
    if args.plot in ['probabilities', 'all']:
        vary = args.vary or var_names[0]
        if args.model in ['logit', 'probit', 'poisson', 'negbin']:
            fig = plot_predicted_probabilities(model, X, X_const, vary, var_names, args.model)
            figures.append(('predicted_probs', fig))
        elif args.model in ['ologit', 'oprobit', 'mlogit']:
            fig = plot_category_probabilities(model, X, vary, var_names, args.model)
            figures.append(('category_probs', fig))
            fig2 = plot_category_lines(model, X, vary, var_names, args.model)
            figures.append(('category_lines', fig2))

    if args.plot in ['marginal', 'all'] and args.model in ['logit', 'probit', 'poisson', 'negbin']:
        fig = plot_marginal_effects(model, X, var_names, args.model)
        figures.append(('marginal_effects', fig))

    if args.plot in ['roc', 'all'] and args.model in ['logit', 'probit']:
        fig = plot_roc_curve(model, y, X_const)
        figures.append(('roc_curve', fig))

    if args.plot in ['calibration', 'all'] and args.model in ['logit', 'probit']:
        fig = plot_calibration(model, y, X_const)
        figures.append(('calibration', fig))

    if args.plot in ['predicted', 'all'] and args.model in ['poisson', 'negbin']:
        fig = plot_predicted_vs_observed(model, y, X_const)
        figures.append(('pred_vs_obs', fig))
        fig2 = plot_rootogram(model, y, X_const)
        figures.append(('rootogram', fig2))

    if args.treatment and args.vary and args.model in ['logit', 'probit', 'poisson', 'negbin']:
        fig = plot_treatment_effect_by_covariate(
            model, X, X_const, args.treatment, args.vary, var_names, args.model
        )
        figures.append(('treatment_effect', fig))

    # Save or show
    if args.output:
        if len(figures) == 1:
            figures[0][1].savefig(args.output, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved: {args.output}")
        else:
            base = args.output.rsplit('.', 1)[0]
            ext = args.output.rsplit('.', 1)[1] if '.' in args.output else 'png'
            for name, fig in figures:
                path = f"{base}_{name}.{ext}"
                fig.savefig(path, dpi=args.dpi, bbox_inches='tight')
                print(f"Saved: {path}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
