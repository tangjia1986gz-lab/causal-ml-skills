"""
Data loading utilities for causal inference skills.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path


@dataclass
class CausalInput:
    """Standard input structure for causal inference skills."""

    # Core data
    data: pd.DataFrame

    # Key variables
    outcome: str          # Y: outcome variable
    treatment: str        # D: treatment variable

    # Control variables
    controls: List[str] = None

    # Panel structure (optional)
    unit_id: str = None   # Individual/entity identifier
    time_id: str = None   # Time period identifier

    # Method-specific parameters
    params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate input data structure."""
        errors = []

        # Check required columns exist
        if self.outcome not in self.data.columns:
            errors.append(f"Outcome variable '{self.outcome}' not found in data")
        if self.treatment not in self.data.columns:
            errors.append(f"Treatment variable '{self.treatment}' not found in data")

        if self.controls:
            missing = [c for c in self.controls if c not in self.data.columns]
            if missing:
                errors.append(f"Control variables not found: {missing}")

        if self.unit_id and self.unit_id not in self.data.columns:
            errors.append(f"Unit ID '{self.unit_id}' not found in data")
        if self.time_id and self.time_id not in self.data.columns:
            errors.append(f"Time ID '{self.time_id}' not found in data")

        return len(errors) == 0, errors

    def summary(self) -> str:
        """Generate data summary."""
        lines = []
        lines.append("=" * 50)
        lines.append("CAUSAL INPUT SUMMARY")
        lines.append("=" * 50)
        lines.append(f"Observations: {len(self.data):,}")
        lines.append(f"Variables: {len(self.data.columns)}")
        lines.append("")
        lines.append(f"Outcome: {self.outcome}")
        lines.append(f"  - Mean: {self.data[self.outcome].mean():.4f}")
        lines.append(f"  - Std:  {self.data[self.outcome].std():.4f}")
        lines.append("")
        lines.append(f"Treatment: {self.treatment}")
        if self.data[self.treatment].nunique() <= 2:
            lines.append(f"  - Treated:   {(self.data[self.treatment] == 1).sum():,}")
            lines.append(f"  - Control:   {(self.data[self.treatment] == 0).sum():,}")
        else:
            lines.append(f"  - Mean: {self.data[self.treatment].mean():.4f}")

        if self.controls:
            lines.append(f"\nControls: {len(self.controls)} variables")
            for c in self.controls[:5]:
                lines.append(f"  - {c}")
            if len(self.controls) > 5:
                lines.append(f"  ... and {len(self.controls) - 5} more")

        if self.unit_id and self.time_id:
            lines.append(f"\nPanel Structure:")
            lines.append(f"  - Units:   {self.data[self.unit_id].nunique():,}")
            lines.append(f"  - Periods: {self.data[self.time_id].nunique()}")

        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class CausalOutput:
    """Standard output structure for causal inference skills."""

    # Core estimates
    effect: float                    # Point estimate
    se: float                        # Standard error
    ci_lower: float                  # 95% CI lower bound
    ci_upper: float                  # 95% CI upper bound
    p_value: float                   # P-value

    # Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Visualizations
    figures: List[Any] = None

    # Formatted output
    summary_table: str = ""
    interpretation: str = ""

    def __repr__(self):
        stars = ""
        if self.p_value < 0.01:
            stars = "***"
        elif self.p_value < 0.05:
            stars = "**"
        elif self.p_value < 0.1:
            stars = "*"

        return (
            f"CausalOutput(\n"
            f"  effect={self.effect:.4f}{stars},\n"
            f"  se={self.se:.4f},\n"
            f"  95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}],\n"
            f"  p_value={self.p_value:.4f}\n"
            f")"
        )

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if effect is statistically significant."""
        return self.p_value < alpha

    def generate_interpretation(self, treatment_name: str = "treatment", outcome_name: str = "outcome") -> str:
        """Generate human-readable interpretation."""
        direction = "increases" if self.effect > 0 else "decreases"
        abs_effect = abs(self.effect)

        sig_level = ""
        if self.p_value < 0.01:
            sig_level = "highly significant (p < 0.01)"
        elif self.p_value < 0.05:
            sig_level = "significant (p < 0.05)"
        elif self.p_value < 0.1:
            sig_level = "marginally significant (p < 0.1)"
        else:
            sig_level = "not statistically significant"

        interpretation = (
            f"The estimated effect of {treatment_name} on {outcome_name} is "
            f"{self.effect:.4f} (SE = {self.se:.4f}). "
            f"This suggests that {treatment_name} {direction} {outcome_name} by "
            f"{abs_effect:.4f} units. "
            f"The effect is {sig_level}. "
            f"The 95% confidence interval is [{self.ci_lower:.4f}, {self.ci_upper:.4f}]."
        )

        return interpretation


def load_benchmark_data(dataset_name: str) -> pd.DataFrame:
    """
    Load standard benchmark datasets for causal inference.

    Parameters
    ----------
    dataset_name : str
        One of: 'lalonde', 'card', 'lee', 'smoking'

    Returns
    -------
    pd.DataFrame
        Benchmark dataset
    """
    datasets_path = Path(__file__).parent.parent.parent / "tests" / "data" / "benchmark"

    dataset_files = {
        'lalonde': 'lalonde_nsw.csv',
        'card': 'card_proximity.csv',
        'lee': 'lee_regression.csv',
        'smoking': 'smoking_did.csv'
    }

    if dataset_name not in dataset_files:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_files.keys())}")

    file_path = datasets_path / dataset_files[dataset_name]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}. "
            f"Run data generation scripts first."
        )

    return pd.read_csv(file_path)


def generate_synthetic_did_data(
    n_units: int = 100,
    n_periods: int = 10,
    treatment_period: int = 5,
    treatment_effect: float = 2.0,
    treatment_share: float = 0.5,
    noise_std: float = 1.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic DID panel data with known treatment effect.

    Parameters
    ----------
    n_units : int
        Number of units (individuals, firms, etc.)
    n_periods : int
        Number of time periods
    treatment_period : int
        Period when treatment starts
    treatment_effect : float
        True treatment effect (for validation)
    treatment_share : float
        Proportion of units that receive treatment
    noise_std : float
        Standard deviation of noise
    random_state : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (data, true_params) where true_params contains ground truth
    """
    np.random.seed(random_state)

    # Create panel structure
    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(n_periods), n_units)

    # Assign treatment groups
    n_treated = int(n_units * treatment_share)
    treated_units = np.random.choice(n_units, n_treated, replace=False)
    treatment_group = np.isin(units, treated_units).astype(int)

    # Post-treatment indicator
    post = (periods >= treatment_period).astype(int)

    # DID interaction
    did = treatment_group * post

    # Unit fixed effects
    unit_fe = np.random.normal(0, 2, n_units)[units]

    # Time fixed effects (common trend)
    time_fe = 0.5 * periods

    # Generate outcome
    y = (
        unit_fe +
        time_fe +
        treatment_effect * did +
        np.random.normal(0, noise_std, len(units))
    )

    # Add covariates
    x1 = np.random.normal(0, 1, len(units))
    x2 = np.random.binomial(1, 0.5, len(units))

    data = pd.DataFrame({
        'unit_id': units,
        'time': periods,
        'treatment_group': treatment_group,
        'post': post,
        'treated': did,
        'y': y,
        'x1': x1,
        'x2': x2
    })

    true_params = {
        'true_ate': treatment_effect,
        'n_units': n_units,
        'n_periods': n_periods,
        'treatment_period': treatment_period,
        'n_treated_units': n_treated,
        'noise_std': noise_std
    }

    return data, true_params


def generate_synthetic_rd_data(
    n: int = 1000,
    cutoff: float = 0.0,
    treatment_effect: float = 0.5,
    noise_std: float = 0.5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic RD data with known treatment effect.

    Parameters
    ----------
    n : int
        Number of observations
    cutoff : float
        RD cutoff value
    treatment_effect : float
        True discontinuity jump at cutoff
    noise_std : float
        Standard deviation of noise
    random_state : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (data, true_params)
    """
    np.random.seed(random_state)

    # Running variable (centered at cutoff)
    running = np.random.uniform(-1, 1, n)

    # Treatment indicator
    treatment = (running >= cutoff).astype(int)

    # Outcome with smooth function + discontinuity
    y = (
        0.5 * running +                      # Linear trend
        0.3 * running**2 +                   # Curvature
        treatment_effect * treatment +        # Treatment effect
        np.random.normal(0, noise_std, n)    # Noise
    )

    # Covariates
    x1 = running + np.random.normal(0, 0.2, n)
    x2 = np.random.binomial(1, 0.5, n)

    data = pd.DataFrame({
        'running': running,
        'treatment': treatment,
        'y': y,
        'x1': x1,
        'x2': x2
    })

    true_params = {
        'true_late': treatment_effect,
        'cutoff': cutoff,
        'n': n,
        'noise_std': noise_std
    }

    return data, true_params


def generate_synthetic_iv_data(
    n: int = 1000,
    treatment_effect: float = 1.0,
    first_stage_strength: float = 0.5,
    noise_std: float = 1.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic IV data with endogeneity and known treatment effect.

    Parameters
    ----------
    n : int
        Number of observations
    treatment_effect : float
        True causal effect
    first_stage_strength : float
        Correlation between instrument and treatment
    noise_std : float
        Standard deviation of noise
    random_state : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (data, true_params)
    """
    np.random.seed(random_state)

    # Confounder (unobserved)
    u = np.random.normal(0, 1, n)

    # Instrument (exogenous)
    z = np.random.normal(0, 1, n)

    # Endogenous treatment (affected by both instrument and confounder)
    d = first_stage_strength * z + 0.5 * u + np.random.normal(0, 0.5, n)

    # Outcome (affected by treatment and confounder)
    y = treatment_effect * d + 0.8 * u + np.random.normal(0, noise_std, n)

    # Observable covariates
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.binomial(1, 0.5, n)

    data = pd.DataFrame({
        'y': y,
        'd': d,
        'z': z,
        'x1': x1,
        'x2': x2
    })

    # Calculate theoretical first-stage F
    from scipy import stats
    slope, _, r_value, _, _ = stats.linregress(z, d)
    theoretical_f = (r_value**2 / (1 - r_value**2)) * (n - 2)

    true_params = {
        'true_effect': treatment_effect,
        'first_stage_strength': first_stage_strength,
        'ols_bias': 0.8 * 0.5 / 1.0,  # Approximate OLS bias
        'theoretical_first_stage_f': theoretical_f,
        'n': n,
        'noise_std': noise_std
    }

    return data, true_params
