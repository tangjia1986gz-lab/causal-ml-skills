#!/usr/bin/env python3
"""
Marginal Effects Calculator for Discrete Choice Models

Computes Average Marginal Effects (AME) and Marginal Effects at Mean (MEM)
with proper standard errors via delta method or bootstrap.

Usage:
    python calculate_marginal_effects.py --data data.csv --y outcome --x "var1 var2" --model logit --method ame
    python calculate_marginal_effects.py --data data.csv --y outcome --x "var1 var2" --model probit --bootstrap 1000
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import Logit, Probit, Poisson, NegativeBinomial


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate marginal effects for discrete choice models'
    )

    parser.add_argument('--data', '-d', required=True, help='Path to CSV data file')
    parser.add_argument('--y', required=True, help='Outcome variable name')
    parser.add_argument('--x', required=True, help='Covariate names (space-separated)')
    parser.add_argument('--model', '-m', required=True,
                        choices=['logit', 'probit', 'poisson', 'negbin', 'lpm'],
                        help='Model type')
    parser.add_argument('--method', default='ame',
                        choices=['ame', 'mem', 'mer'],
                        help='Marginal effect type: ame (average), mem (at mean), mer (at values)')
    parser.add_argument('--at-values', help='JSON string of values for MER (e.g., \'{"x1": 0.5}\')')
    parser.add_argument('--bootstrap', '-b', type=int, default=0,
                        help='Number of bootstrap iterations (0 for delta method)')
    parser.add_argument('--discrete', help='Space-separated discrete variables for discrete change')
    parser.add_argument('--output', '-o', help='Output CSV file')
    parser.add_argument('--confidence', type=float, default=0.95,
                        help='Confidence level (default: 0.95)')

    return parser.parse_args()


def logit_pdf(z: np.ndarray) -> np.ndarray:
    """PDF of logistic distribution."""
    exp_z = np.exp(-np.abs(z))
    return exp_z / (1 + exp_z)**2


def logit_cdf(z: np.ndarray) -> np.ndarray:
    """CDF of logistic distribution."""
    return 1 / (1 + np.exp(-z))


def compute_ame_logit(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Compute Average Marginal Effects for logit model.

    AME_j = (1/N) * sum_i [beta_j * Lambda(X_i*beta) * (1 - Lambda(X_i*beta))]
    """
    linear_pred = X @ beta
    prob = logit_cdf(linear_pred)
    d_prob = prob * (1 - prob)

    # AME for each coefficient (excluding constant)
    ame = np.mean(d_prob[:, np.newaxis] * beta[1:], axis=0)
    return ame


def compute_ame_probit(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Compute Average Marginal Effects for probit model.

    AME_j = (1/N) * sum_i [beta_j * phi(X_i*beta)]
    """
    linear_pred = X @ beta
    d_prob = stats.norm.pdf(linear_pred)

    # AME for each coefficient (excluding constant)
    ame = np.mean(d_prob[:, np.newaxis] * beta[1:], axis=0)
    return ame


def compute_ame_count(X: np.ndarray, beta: np.ndarray, model_type: str = 'poisson') -> np.ndarray:
    """
    Compute Average Marginal Effects for count models.

    For log-linear models: AME_j = beta_j * mean(mu)
    """
    linear_pred = X @ beta
    mu = np.exp(linear_pred)

    # AME = beta * mean(mu)
    ame = beta[1:] * np.mean(mu)
    return ame


def compute_mem_logit(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Marginal Effects at Mean for logit."""
    x_mean = np.mean(X, axis=0)
    linear_pred = x_mean @ beta
    prob = logit_cdf(linear_pred)
    d_prob = prob * (1 - prob)

    mem = d_prob * beta[1:]
    return mem


def compute_mem_probit(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Marginal Effects at Mean for probit."""
    x_mean = np.mean(X, axis=0)
    linear_pred = x_mean @ beta
    d_prob = stats.norm.pdf(linear_pred)

    mem = d_prob * beta[1:]
    return mem


def compute_mem_count(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Marginal Effects at Mean for count models."""
    x_mean = np.mean(X, axis=0)
    linear_pred = x_mean @ beta
    mu = np.exp(linear_pred)

    mem = beta[1:] * mu
    return mem


def compute_mer(X: np.ndarray, beta: np.ndarray, at_values: dict,
                model_type: str, var_names: List[str]) -> np.ndarray:
    """
    Marginal Effects at Representative Values.

    Parameters
    ----------
    at_values : dict
        Dictionary mapping variable names to values
    """
    x_rep = np.mean(X, axis=0)  # Start with means

    # Override with specified values
    for var, val in at_values.items():
        if var in var_names:
            idx = var_names.index(var) + 1  # +1 for constant
            x_rep[idx] = val

    linear_pred = x_rep @ beta

    if model_type == 'logit':
        prob = logit_cdf(linear_pred)
        d_prob = prob * (1 - prob)
        mer = d_prob * beta[1:]
    elif model_type == 'probit':
        d_prob = stats.norm.pdf(linear_pred)
        mer = d_prob * beta[1:]
    elif model_type in ['poisson', 'negbin']:
        mu = np.exp(linear_pred)
        mer = beta[1:] * mu
    else:
        mer = beta[1:]

    return mer


def discrete_change_effect(X: np.ndarray, beta: np.ndarray, var_idx: int,
                           model_type: str) -> float:
    """
    Compute discrete change effect for binary variable.

    Delta = P(Y=1|X, D=1) - P(Y=1|X, D=0)
    """
    X1 = X.copy()
    X0 = X.copy()
    X1[:, var_idx] = 1
    X0[:, var_idx] = 0

    if model_type == 'logit':
        p1 = logit_cdf(X1 @ beta)
        p0 = logit_cdf(X0 @ beta)
    elif model_type == 'probit':
        p1 = stats.norm.cdf(X1 @ beta)
        p0 = stats.norm.cdf(X0 @ beta)
    elif model_type in ['poisson', 'negbin']:
        p1 = np.exp(X1 @ beta)
        p0 = np.exp(X0 @ beta)
    else:
        p1 = X1 @ beta
        p0 = X0 @ beta

    return np.mean(p1 - p0)


def delta_method_se(result, X: np.ndarray, model_type: str, method: str = 'ame') -> np.ndarray:
    """
    Compute standard errors via delta method.

    For AME: Var(AME) approx = G' * Var(beta) * G
    where G is the gradient of AME with respect to beta.
    """
    beta = result.params
    V = result.cov_params()
    n, k = X.shape

    if model_type in ['logit', 'probit']:
        # Numerical gradient
        eps = 1e-6
        k_vars = len(beta) - 1  # Exclude constant

        G = np.zeros((k_vars, len(beta)))

        for j in range(len(beta)):
            beta_plus = beta.copy()
            beta_minus = beta.copy()
            beta_plus[j] += eps
            beta_minus[j] -= eps

            if method == 'ame':
                if model_type == 'logit':
                    ame_plus = compute_ame_logit(X, beta_plus)
                    ame_minus = compute_ame_logit(X, beta_minus)
                else:
                    ame_plus = compute_ame_probit(X, beta_plus)
                    ame_minus = compute_ame_probit(X, beta_minus)
            else:  # mem
                if model_type == 'logit':
                    ame_plus = compute_mem_logit(X, beta_plus)
                    ame_minus = compute_mem_logit(X, beta_minus)
                else:
                    ame_plus = compute_mem_probit(X, beta_plus)
                    ame_minus = compute_mem_probit(X, beta_minus)

            G[:, j] = (ame_plus - ame_minus) / (2 * eps)

        # Variance of AME
        var_ame = np.diag(G @ V @ G.T)
        se = np.sqrt(var_ame)

    elif model_type in ['poisson', 'negbin']:
        # Simplified: treat mu as fixed
        mu_mean = np.mean(np.exp(X @ beta))
        se = result.bse[1:] * mu_mean

    else:  # LPM
        se = result.bse[1:]

    return se


def bootstrap_marginal_effects(y: np.ndarray, X: np.ndarray, model_type: str,
                               method: str, n_bootstrap: int = 500,
                               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute marginal effects with bootstrap standard errors.

    Returns
    -------
    ame : array of marginal effects
    se : array of bootstrap standard errors
    """
    np.random.seed(random_state)
    n = len(y)

    # Choose model class
    if model_type == 'logit':
        model_class = Logit
        ame_func = compute_ame_logit if method == 'ame' else compute_mem_logit
    elif model_type == 'probit':
        model_class = Probit
        ame_func = compute_ame_probit if method == 'ame' else compute_mem_probit
    elif model_type == 'poisson':
        model_class = Poisson
        ame_func = lambda X, b: compute_ame_count(X, b, 'poisson')
    elif model_type == 'negbin':
        model_class = NegativeBinomial
        ame_func = lambda X, b: compute_ame_count(X, b, 'negbin')
    else:
        model_class = sm.OLS
        ame_func = lambda X, b: b[1:]

    ame_samples = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        y_b = y[idx]
        X_b = X[idx]

        try:
            model = model_class(y_b, X_b)
            result = model.fit(disp=0)
            ame = ame_func(X, result.params)
            ame_samples.append(ame)
        except:
            continue

    ame_samples = np.array(ame_samples)

    # Point estimate from full sample
    model = model_class(y, X)
    result = model.fit(disp=0)
    ame = ame_func(X, result.params)

    # Bootstrap SE
    se = np.std(ame_samples, axis=0)

    return ame, se


def fit_and_compute_mfx(y: np.ndarray, X: np.ndarray, model_type: str,
                        method: str, var_names: List[str],
                        bootstrap: int = 0, at_values: dict = None,
                        discrete_vars: List[str] = None) -> Dict:
    """
    Fit model and compute marginal effects.
    """
    X_const = sm.add_constant(X)

    # Choose model
    if model_type == 'logit':
        model = Logit(y, X_const)
    elif model_type == 'probit':
        model = Probit(y, X_const)
    elif model_type == 'poisson':
        model = Poisson(y, X_const)
    elif model_type == 'negbin':
        model = NegativeBinomial(y, X_const)
    else:  # lpm
        model = sm.OLS(y, X_const)

    result = model.fit(disp=0)

    # Compute marginal effects
    if bootstrap > 0:
        ame, se = bootstrap_marginal_effects(y, X_const, model_type, method, bootstrap)
    else:
        # Point estimates
        if method == 'ame':
            if model_type == 'logit':
                ame = compute_ame_logit(X_const, result.params)
            elif model_type == 'probit':
                ame = compute_ame_probit(X_const, result.params)
            elif model_type in ['poisson', 'negbin']:
                ame = compute_ame_count(X_const, result.params, model_type)
            else:
                ame = result.params[1:]

        elif method == 'mem':
            if model_type == 'logit':
                ame = compute_mem_logit(X_const, result.params)
            elif model_type == 'probit':
                ame = compute_mem_probit(X_const, result.params)
            elif model_type in ['poisson', 'negbin']:
                ame = compute_mem_count(X_const, result.params)
            else:
                ame = result.params[1:]

        elif method == 'mer' and at_values:
            ame = compute_mer(X_const, result.params, at_values, model_type, var_names)

        else:
            ame = result.params[1:]

        # Standard errors via delta method
        se = delta_method_se(result, X_const, model_type, method)

    # Discrete change effects for binary variables
    discrete_effects = {}
    if discrete_vars:
        for var in discrete_vars:
            if var in var_names:
                idx = var_names.index(var) + 1  # +1 for constant
                dce = discrete_change_effect(X_const, result.params, idx, model_type)
                discrete_effects[var] = dce

    return {
        'ame': ame,
        'se': se,
        'var_names': var_names,
        'method': method,
        'model_type': model_type,
        'model_result': result,
        'discrete_effects': discrete_effects
    }


def format_results(results: Dict, confidence: float = 0.95) -> pd.DataFrame:
    """Format results as a DataFrame."""
    z_crit = stats.norm.ppf((1 + confidence) / 2)

    records = []
    for name, ame, se in zip(results['var_names'], results['ame'], results['se']):
        z = ame / se if se > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci_low = ame - z_crit * se
        ci_high = ame + z_crit * se

        records.append({
            'Variable': name,
            f'{results["method"].upper()}': ame,
            'Std.Err.': se,
            'z': z,
            'P>|z|': p,
            f'CI_low_{int(confidence*100)}': ci_low,
            f'CI_high_{int(confidence*100)}': ci_high
        })

    return pd.DataFrame(records)


def main():
    args = parse_args()

    # Parse variables
    x_vars = args.x.split()
    discrete_vars = args.discrete.split() if args.discrete else None

    # Parse at-values for MER
    at_values = None
    if args.at_values:
        import json
        at_values = json.loads(args.at_values)

    # Load data
    df = pd.read_csv(args.data)
    y = df[args.y].values
    X = df[x_vars].values

    # Drop missing
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y = y[mask]
    X = X[mask]

    print(f"Model: {args.model.upper()}")
    print(f"Method: {args.method.upper()}")
    print(f"N = {len(y)}")

    # Compute marginal effects
    results = fit_and_compute_mfx(
        y, X, args.model, args.method, x_vars,
        bootstrap=args.bootstrap,
        at_values=at_values,
        discrete_vars=discrete_vars
    )

    # Format and display
    df_results = format_results(results, args.confidence)

    print("\n--- Marginal Effects ---")
    print(df_results.to_string(index=False))

    # Discrete change effects
    if results['discrete_effects']:
        print("\n--- Discrete Change Effects ---")
        for var, dce in results['discrete_effects'].items():
            print(f"{var}: {dce:.6f}")

    # Save if requested
    if args.output:
        df_results.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
