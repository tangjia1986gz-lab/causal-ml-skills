"""
Data Generating Process for DDML synthetic data.

Generates high-dimensional data with nonlinear confounding for validating
Double/Debiased Machine Learning estimators.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_high_dim_linear(
    n: int = 2000,
    p: int = 100,
    s: int = 10,
    treatment_effect: float = 2.0,
    noise_std: float = 1.0,
    random_state: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Generate high-dimensional data with sparse linear confounding.

    Parameters
    ----------
    n : int
        Number of observations
    p : int
        Number of covariates
    s : int
        Number of truly relevant covariates (sparsity)
    treatment_effect : float
        True ATE
    noise_std : float
        Noise standard deviation
    random_state : int
        Random seed

    Returns
    -------
    data : pd.DataFrame
        High-dimensional dataset
    params : dict
        True parameters
    """
    np.random.seed(random_state)

    # Generate covariates
    X = np.random.normal(0, 1, (n, p))

    # Sparse coefficients for outcome model
    beta_y = np.zeros(p)
    beta_y[:s] = np.random.uniform(0.5, 1.5, s) * np.random.choice([-1, 1], s)

    # Sparse coefficients for treatment model
    beta_d = np.zeros(p)
    beta_d[:s] = np.random.uniform(0.3, 0.8, s) * np.random.choice([-1, 1], s)

    # Treatment (binary)
    propensity = 1 / (1 + np.exp(-X @ beta_d))
    d = np.random.binomial(1, propensity, n)

    # Outcome
    y = X @ beta_y + treatment_effect * d + np.random.normal(0, noise_std, n)

    # Create DataFrame
    data = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
    data['d'] = d
    data['y'] = y

    params = {
        'true_ate': treatment_effect,
        'n': n,
        'p': p,
        's': s,
        'relevant_vars': [f'x{i}' for i in range(s)],
        'beta_y': beta_y,
        'beta_d': beta_d
    }

    return data, params


def generate_high_dim_nonlinear(
    n: int = 2000,
    p: int = 50,
    treatment_effect: float = 2.0,
    noise_std: float = 1.0,
    nonlinearity: str = "moderate",
    random_state: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Generate high-dimensional data with nonlinear confounding.

    This DGP requires flexible ML methods - linear methods will be biased.

    Parameters
    ----------
    n : int
        Sample size
    p : int
        Number of covariates
    treatment_effect : float
        True ATE
    noise_std : float
        Noise std
    nonlinearity : str
        Level of nonlinearity: "mild", "moderate", "severe"
    random_state : int
        Random seed

    Returns
    -------
    data : pd.DataFrame
    params : dict
    """
    np.random.seed(random_state)

    # Generate covariates
    X = np.random.normal(0, 1, (n, p))

    # Nonlinear confounding function
    if nonlinearity == "mild":
        g = X[:, 0] + 0.5 * X[:, 1]**2 + X[:, 2] * X[:, 3]
        m = 0.5 * X[:, 0] + 0.3 * np.abs(X[:, 1]) + 0.2 * X[:, 2]
    elif nonlinearity == "moderate":
        g = (X[:, 0] + 0.5 * X[:, 1]**2 + X[:, 2] * X[:, 3] +
             np.sin(X[:, 4]) + np.exp(-X[:, 5]**2))
        m = (0.5 * X[:, 0] + 0.3 * np.abs(X[:, 1]) +
             0.2 * np.maximum(X[:, 2], 0) + 0.1 * X[:, 3]**2)
    else:  # severe
        g = (X[:, 0] + X[:, 1]**2 + X[:, 2] * X[:, 3] +
             np.sin(2 * X[:, 4]) + np.exp(-X[:, 5]**2) +
             np.maximum(X[:, 6], 0) * X[:, 7])
        m = (0.5 * X[:, 0] + 0.3 * np.abs(X[:, 1]) +
             0.2 * np.maximum(X[:, 2], 0) + 0.1 * X[:, 3]**2 +
             0.2 * np.sin(X[:, 4]) + 0.1 * X[:, 5] * X[:, 6])

    # Treatment
    propensity = 1 / (1 + np.exp(-m))
    propensity = np.clip(propensity, 0.1, 0.9)  # Overlap
    d = np.random.binomial(1, propensity, n)

    # Outcome: Y = g(X) + tau*D + epsilon
    y = g + treatment_effect * d + np.random.normal(0, noise_std, n)

    # DataFrame
    data = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
    data['d'] = d
    data['y'] = y

    params = {
        'true_ate': treatment_effect,
        'n': n,
        'p': p,
        'nonlinearity': nonlinearity,
        'ols_would_be_biased': True,
        'description': f"Nonlinear confounding ({nonlinearity}): requires ML methods"
    }

    return data, params


def generate_heterogeneous_effects(
    n: int = 2000,
    p: int = 20,
    base_effect: float = 1.0,
    heterogeneity_vars: int = 3,
    noise_std: float = 1.0,
    random_state: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Generate data with heterogeneous treatment effects for CATE estimation.

    Parameters
    ----------
    n : int
        Sample size
    p : int
        Number of covariates
    base_effect : float
        Base treatment effect
    heterogeneity_vars : int
        Number of variables driving heterogeneity
    noise_std : float
        Noise std
    random_state : int
        Random seed

    Returns
    -------
    data : pd.DataFrame
    params : dict
    """
    np.random.seed(random_state)

    X = np.random.normal(0, 1, (n, p))

    # Heterogeneous treatment effect: tau(x) = base + sum of effects
    tau = base_effect + np.zeros(n)
    for j in range(heterogeneity_vars):
        tau += 0.5 * X[:, j]  # Effect increases with x0, x1, x2

    # Confounding
    m = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2]
    propensity = 1 / (1 + np.exp(-m))
    propensity = np.clip(propensity, 0.1, 0.9)
    d = np.random.binomial(1, propensity, n)

    # Outcome with heterogeneous effects
    g = X[:, 0] + 0.5 * X[:, 1]**2 + 0.3 * X[:, 2]
    y = g + tau * d + np.random.normal(0, noise_std, n)

    # DataFrame
    data = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
    data['d'] = d
    data['y'] = y
    data['true_cate'] = tau  # Ground truth for validation

    params = {
        'true_ate': base_effect + 0.5 * heterogeneity_vars * 0,  # E[tau] ≈ base_effect
        'cate_formula': f"tau(x) = {base_effect} + 0.5*x0 + 0.5*x1 + 0.5*x2",
        'heterogeneity_vars': [f'x{i}' for i in range(heterogeneity_vars)],
        'n': n,
        'p': p
    }

    return data, params


if __name__ == "__main__":
    output_dir = Path(__file__).parent

    # High-dimensional linear
    data, params = generate_high_dim_linear(n=2000, p=100, s=10)
    data.to_csv(output_dir / "ddml_highdim_linear.csv", index=False)
    print(f"Generated ddml_highdim_linear.csv (n=2000, p=100, s=10)")
    print(f"  True ATE = {params['true_ate']}")

    # High-dimensional nonlinear
    data_nl, params_nl = generate_high_dim_nonlinear(n=2000, p=50, nonlinearity="moderate")
    data_nl.to_csv(output_dir / "ddml_highdim_nonlinear.csv", index=False)
    print(f"Generated ddml_highdim_nonlinear.csv (n=2000, p=50)")
    print(f"  True ATE = {params_nl['true_ate']}")

    # Heterogeneous effects
    data_het, params_het = generate_heterogeneous_effects(n=2000, p=20)
    data_het.to_csv(output_dir / "ddml_heterogeneous.csv", index=False)
    print(f"Generated ddml_heterogeneous.csv with CATE")
    print(f"  CATE formula: {params_het['cate_formula']}")

    print("\nDatasets saved to:", output_dir)
