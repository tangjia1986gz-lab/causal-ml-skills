"""
Data Generating Process for RD synthetic data.

Generates regression discontinuity data with known local treatment effects.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_sharp_rd(
    n: int = 2000,
    cutoff: float = 0.0,
    treatment_effect: float = 0.5,
    noise_std: float = 0.5,
    running_dist: str = "uniform",
    polynomial_order: int = 2,
    random_state: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Generate Sharp RD data.

    Parameters
    ----------
    n : int
        Number of observations
    cutoff : float
        RD cutoff value
    treatment_effect : float
        True discontinuity jump at cutoff (LATE)
    noise_std : float
        Standard deviation of noise
    running_dist : str
        Distribution of running variable: "uniform" or "normal"
    polynomial_order : int
        Order of polynomial in conditional expectation function
    random_state : int
        Random seed

    Returns
    -------
    data : pd.DataFrame
        RD dataset with columns:
        - running: running/forcing variable
        - treatment: treatment indicator (running >= cutoff)
        - y: outcome variable
        - x1, x2: covariates (should be balanced at cutoff)
    params : dict
        True parameters
    """
    np.random.seed(random_state)

    # Running variable centered at cutoff
    if running_dist == "uniform":
        running = np.random.uniform(-1, 1, n)
    else:
        running = np.clip(np.random.normal(0, 0.4, n), -1, 1)

    # Treatment indicator (sharp assignment)
    treatment = (running >= cutoff).astype(int)

    # Potential outcomes with smooth conditional expectation
    # E[Y(0) | X] = polynomial in running variable
    if polynomial_order == 1:
        m0 = 0.5 * running
    elif polynomial_order == 2:
        m0 = 0.5 * running + 0.3 * running**2
    else:
        m0 = 0.5 * running + 0.3 * running**2 - 0.2 * running**3

    # Outcome = m0 + tau*D + noise (constant treatment effect)
    y = m0 + treatment_effect * treatment + np.random.normal(0, noise_std, n)

    # Covariates (balanced at cutoff by construction)
    x1 = running + np.random.normal(0, 0.2, n)
    x2 = np.random.binomial(1, 0.5 + 0.1 * running, n)

    data = pd.DataFrame({
        'running': running,
        'treatment': treatment,
        'y': y,
        'x1': x1,
        'x2': x2
    })

    params = {
        'true_late': treatment_effect,
        'cutoff': cutoff,
        'n': n,
        'noise_std': noise_std,
        'polynomial_order': polynomial_order,
        'running_dist': running_dist
    }

    return data, params


def generate_fuzzy_rd(
    n: int = 2000,
    cutoff: float = 0.0,
    treatment_effect: float = 1.0,
    compliance_below: float = 0.1,
    compliance_above: float = 0.7,
    noise_std: float = 0.5,
    random_state: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Generate Fuzzy RD data with imperfect compliance.

    Parameters
    ----------
    n : int
        Number of observations
    cutoff : float
        RD cutoff value
    treatment_effect : float
        True treatment effect for compliers
    compliance_below : float
        Probability of treatment when running < cutoff
    compliance_above : float
        Probability of treatment when running >= cutoff
    noise_std : float
        Noise standard deviation
    random_state : int
        Random seed

    Returns
    -------
    data : pd.DataFrame
        Fuzzy RD data with:
        - running: running variable
        - above_cutoff: indicator for running >= cutoff
        - treatment: actual treatment received
        - y: outcome
    params : dict
        True parameters including first stage effect
    """
    np.random.seed(random_state)

    running = np.random.uniform(-1, 1, n)
    above_cutoff = (running >= cutoff).astype(int)

    # Fuzzy treatment assignment
    prob_treat = np.where(above_cutoff, compliance_above, compliance_below)
    treatment = np.random.binomial(1, prob_treat, n)

    # Outcome
    m0 = 0.5 * running + 0.3 * running**2
    y = m0 + treatment_effect * treatment + np.random.normal(0, noise_std, n)

    # Covariates
    x1 = running + np.random.normal(0, 0.2, n)

    data = pd.DataFrame({
        'running': running,
        'above_cutoff': above_cutoff,
        'treatment': treatment,
        'y': y,
        'x1': x1
    })

    first_stage = compliance_above - compliance_below

    params = {
        'true_late': treatment_effect,
        'cutoff': cutoff,
        'compliance_below': compliance_below,
        'compliance_above': compliance_above,
        'first_stage_effect': first_stage,
        'reduced_form_effect': treatment_effect * first_stage
    }

    return data, params


if __name__ == "__main__":
    output_dir = Path(__file__).parent

    # Sharp RD
    data, params = generate_sharp_rd(n=2000)
    data.to_csv(output_dir / "rd_sharp.csv", index=False)
    print(f"Generated rd_sharp.csv with true LATE = {params['true_late']}")

    # Fuzzy RD
    data_fuzzy, params_fuzzy = generate_fuzzy_rd(n=2000)
    data_fuzzy.to_csv(output_dir / "rd_fuzzy.csv", index=False)
    print(f"Generated rd_fuzzy.csv with true LATE = {params_fuzzy['true_late']}")
    print(f"  First stage effect: {params_fuzzy['first_stage_effect']:.3f}")

    print("\nDatasets saved to:", output_dir)
