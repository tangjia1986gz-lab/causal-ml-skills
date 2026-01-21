"""
Data Generating Process for DID synthetic data.

Generates panel data with known treatment effects for validating DID estimators.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_did_panel(
    n_units: int = 200,
    n_periods: int = 10,
    treatment_period: int = 5,
    treatment_effect: float = 2.0,
    treatment_share: float = 0.5,
    noise_std: float = 1.0,
    unit_fe_std: float = 2.0,
    time_trend: float = 0.5,
    random_state: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Generate synthetic DID panel data.

    Parameters
    ----------
    n_units : int
        Number of cross-sectional units
    n_periods : int
        Number of time periods
    treatment_period : int
        Period when treatment begins (0-indexed)
    treatment_effect : float
        True average treatment effect
    treatment_share : float
        Proportion of units receiving treatment
    noise_std : float
        Standard deviation of idiosyncratic errors
    unit_fe_std : float
        Standard deviation of unit fixed effects
    time_trend : float
        Coefficient on time trend
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    data : pd.DataFrame
        Panel dataset with columns:
        - unit_id: unit identifier
        - time: time period
        - treatment_group: 1 if treated unit, 0 otherwise
        - post: 1 if post-treatment period
        - treated: interaction (treatment_group * post)
        - y: outcome variable
        - x1, x2: control variables
    params : dict
        True data generating parameters
    """
    np.random.seed(random_state)

    # Create panel structure
    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(n_periods), n_units)

    # Assign treatment groups (random assignment)
    n_treated = int(n_units * treatment_share)
    treated_units = np.random.choice(n_units, n_treated, replace=False)
    treatment_group = np.isin(units, treated_units).astype(int)

    # Post-treatment indicator
    post = (periods >= treatment_period).astype(int)

    # DID interaction
    treated = treatment_group * post

    # Unit fixed effects
    unit_fe = np.random.normal(0, unit_fe_std, n_units)[units]

    # Time fixed effects (common trend)
    time_fe = time_trend * periods

    # Control variables
    x1 = np.random.normal(0, 1, len(units))
    x2 = np.random.binomial(1, 0.5, len(units))

    # Generate outcome with parallel trends
    y = (
        unit_fe +                           # Unit fixed effect
        time_fe +                           # Common time trend
        0.5 * x1 +                          # Covariate effect
        0.3 * x2 +                          # Covariate effect
        treatment_effect * treated +         # Treatment effect
        np.random.normal(0, noise_std, len(units))  # Noise
    )

    data = pd.DataFrame({
        'unit_id': units,
        'time': periods,
        'treatment_group': treatment_group,
        'post': post,
        'treated': treated,
        'y': y,
        'x1': x1,
        'x2': x2
    })

    params = {
        'true_ate': treatment_effect,
        'n_units': n_units,
        'n_periods': n_periods,
        'treatment_period': treatment_period,
        'n_treated_units': n_treated,
        'n_control_units': n_units - n_treated,
        'noise_std': noise_std,
        'unit_fe_std': unit_fe_std,
        'time_trend': time_trend,
        'random_state': random_state
    }

    return data, params


def generate_staggered_did(
    n_units: int = 300,
    n_periods: int = 12,
    treatment_times: list = None,
    treatment_effects: dict = None,
    noise_std: float = 1.0,
    random_state: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Generate staggered DID data with multiple treatment cohorts.

    Parameters
    ----------
    n_units : int
        Number of units
    n_periods : int
        Number of time periods
    treatment_times : list
        List of treatment start times for different cohorts
        Default: [4, 6, 8] creating 3 cohorts
    treatment_effects : dict
        Cohort-specific treatment effects {time: effect}
        Default: {4: 2.0, 6: 1.5, 8: 1.0}
    noise_std : float
        Error standard deviation
    random_state : int
        Random seed

    Returns
    -------
    data : pd.DataFrame
        Staggered DID panel data
    params : dict
        True parameters
    """
    np.random.seed(random_state)

    if treatment_times is None:
        treatment_times = [4, 6, 8]
    if treatment_effects is None:
        treatment_effects = {4: 2.0, 6: 1.5, 8: 1.0}

    n_cohorts = len(treatment_times)
    n_never_treated = n_units // (n_cohorts + 1)
    n_per_cohort = (n_units - n_never_treated) // n_cohorts

    # Assign cohorts
    cohort_assignments = np.zeros(n_units, dtype=int)
    idx = n_never_treated
    for i, t in enumerate(treatment_times):
        cohort_assignments[idx:idx + n_per_cohort] = t
        idx += n_per_cohort

    # Shuffle assignments
    np.random.shuffle(cohort_assignments)

    # Create panel
    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(n_periods), n_units)
    cohort = cohort_assignments[units]

    # Treatment indicator (treated if t >= cohort time and cohort > 0)
    treated = ((periods >= cohort) & (cohort > 0)).astype(int)

    # Unit and time effects
    unit_fe = np.random.normal(0, 2, n_units)[units]
    time_fe = 0.3 * periods

    # Treatment effect (cohort-specific)
    te = np.zeros(len(units))
    for t, effect in treatment_effects.items():
        mask = (cohort == t) & (treated == 1)
        te[mask] = effect

    y = unit_fe + time_fe + te + np.random.normal(0, noise_std, len(units))

    data = pd.DataFrame({
        'unit_id': units,
        'time': periods,
        'cohort': cohort,
        'treated': treated,
        'y': y
    })

    params = {
        'treatment_times': treatment_times,
        'treatment_effects': treatment_effects,
        'n_units': n_units,
        'n_periods': n_periods,
        'n_never_treated': n_never_treated
    }

    return data, params


if __name__ == "__main__":
    # Generate and save datasets
    output_dir = Path(__file__).parent

    # Standard DID panel
    data, params = generate_did_panel(n_units=200, n_periods=10)
    data.to_csv(output_dir / "did_panel.csv", index=False)
    print(f"Generated did_panel.csv with true ATE = {params['true_ate']}")

    # Staggered DID
    data_stag, params_stag = generate_staggered_did(n_units=300, n_periods=12)
    data_stag.to_csv(output_dir / "did_staggered.csv", index=False)
    print(f"Generated did_staggered.csv with cohort effects = {params_stag['treatment_effects']}")

    print("\nDatasets saved to:", output_dir)
