"""
Data loading utilities for causal inference skills.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path


@dataclass
class CausalInput:
    """
    Standard input structure for causal inference skills.

    This class provides a unified interface for specifying causal inference
    data structures, including panel data, time series, and cross-sectional data.

    Attributes
    ----------
    data : pd.DataFrame
        The dataset containing all variables
    outcome : str
        Name of the outcome (Y) variable
    treatment : str
        Name of the treatment (D) variable
    controls : List[str], optional
        List of control variable names
    unit_id : str, optional
        Individual/entity identifier for panel data
    time_id : str, optional
        Time period identifier for panel/time-series data
    panel_type : str, optional
        Type of panel data: "balanced" | "unbalanced" | None
    time_series_type : str, optional
        Type of time structure: "cross_section" | "panel" | "time_series" | "repeated_cross_section"
    cluster_var : str, optional
        Variable for clustered standard errors
    weights : str, optional
        Sample weights variable name
    instrument : str, optional
        Instrument variable for IV estimation
    running_var : str, optional
        Running variable for regression discontinuity
    cutoff : float, optional
        Cutoff value for regression discontinuity
    mediator : str, optional
        Mediator variable for mediation analysis
    params : Dict[str, Any]
        Additional method-specific parameters
    """

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

    # Panel and time series type indicators
    panel_type: str = None  # "balanced" | "unbalanced" | None
    time_series_type: str = None  # "cross_section" | "panel" | "time_series" | "repeated_cross_section"

    # Standard error and weighting options
    cluster_var: str = None  # Variable for clustered standard errors
    weights: str = None  # Sample weights variable

    # IV-specific fields
    instrument: str = None  # Instrument variable for IV estimation

    # RD-specific fields
    running_var: str = None  # Running variable for regression discontinuity
    cutoff: float = None  # Cutoff for RD

    # Mediation-specific fields
    mediator: str = None  # Mediator for mediation analysis

    # Method-specific parameters
    params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate input data structure.

        Checks that all specified variables exist in the data and that
        method-specific fields are properly configured.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_error_messages)
        """
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

        # Validate cluster_var
        if self.cluster_var and self.cluster_var not in self.data.columns:
            errors.append(f"Cluster variable '{self.cluster_var}' not found in data")

        # Validate weights
        if self.weights and self.weights not in self.data.columns:
            errors.append(f"Weights variable '{self.weights}' not found in data")

        # Validate instrument (IV)
        if self.instrument and self.instrument not in self.data.columns:
            errors.append(f"Instrument variable '{self.instrument}' not found in data")

        # Validate running_var (RD)
        if self.running_var and self.running_var not in self.data.columns:
            errors.append(f"Running variable '{self.running_var}' not found in data")

        # Validate cutoff requires running_var
        if self.cutoff is not None and self.running_var is None:
            errors.append("Cutoff specified but no running variable provided")

        # Validate mediator
        if self.mediator and self.mediator not in self.data.columns:
            errors.append(f"Mediator variable '{self.mediator}' not found in data")

        # Validate panel_type values
        if self.panel_type is not None and self.panel_type not in ["balanced", "unbalanced"]:
            errors.append(f"panel_type must be 'balanced' or 'unbalanced', got '{self.panel_type}'")

        # Validate time_series_type values
        valid_ts_types = ["cross_section", "panel", "time_series", "repeated_cross_section"]
        if self.time_series_type is not None and self.time_series_type not in valid_ts_types:
            errors.append(f"time_series_type must be one of {valid_ts_types}, got '{self.time_series_type}'")

        # Panel type requires panel structure
        if self.panel_type is not None and (self.unit_id is None or self.time_id is None):
            errors.append("panel_type specified but unit_id and/or time_id not provided")

        return len(errors) == 0, errors

    def is_panel(self) -> bool:
        """
        Check if data has panel structure.

        Returns True if both unit_id and time_id are specified.

        Returns
        -------
        bool
            True if data has panel structure
        """
        return self.unit_id is not None and self.time_id is not None

    def is_balanced(self) -> bool:
        """
        Check if panel is balanced.

        A balanced panel has the same number of time periods for each unit.
        Returns False if data is not panel data.

        Returns
        -------
        bool
            True if panel is balanced, False otherwise or if not panel data
        """
        if not self.is_panel():
            return False

        # If panel_type is explicitly set, use it
        if self.panel_type is not None:
            return self.panel_type == "balanced"

        # Otherwise, check the data
        periods_per_unit = self.data.groupby(self.unit_id)[self.time_id].nunique()
        return periods_per_unit.nunique() == 1

    def get_data_type(self) -> str:
        """
        Return detected data type.

        Returns one of: "panel", "time_series", "cross_section", "repeated_cross_section"

        Returns
        -------
        str
            Detected data type
        """
        # If explicitly set, return it
        if self.time_series_type is not None:
            return self.time_series_type

        # Auto-detect based on structure
        has_unit = self.unit_id is not None
        has_time = self.time_id is not None

        if has_unit and has_time:
            return "panel"
        elif has_time and not has_unit:
            return "time_series"
        elif has_unit and not has_time:
            # Could be cross-section with group structure
            return "cross_section"
        else:
            return "cross_section"

    def detect_structure(self) -> Dict[str, Any]:
        """
        Auto-detect panel/time-series structure from data.

        Analyzes the data to determine its structure, including whether
        it's panel data, whether it's balanced, and relevant statistics.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - data_type: str - Detected data type
            - is_panel: bool - Whether data has panel structure
            - is_balanced: bool - Whether panel is balanced (if applicable)
            - n_units: int - Number of unique units (if panel)
            - n_periods: int - Number of unique time periods (if panel/time_series)
            - n_obs: int - Total number of observations
            - periods_per_unit: Dict - Statistics on periods per unit (if panel)
        """
        result = {
            "data_type": self.get_data_type(),
            "is_panel": self.is_panel(),
            "is_balanced": self.is_balanced(),
            "n_obs": len(self.data)
        }

        if self.unit_id is not None:
            result["n_units"] = self.data[self.unit_id].nunique()

        if self.time_id is not None:
            result["n_periods"] = self.data[self.time_id].nunique()

        if self.is_panel():
            periods_per_unit = self.data.groupby(self.unit_id)[self.time_id].nunique()
            result["periods_per_unit"] = {
                "min": int(periods_per_unit.min()),
                "max": int(periods_per_unit.max()),
                "mean": float(periods_per_unit.mean()),
                "median": float(periods_per_unit.median())
            }

            # Determine balance status
            if periods_per_unit.nunique() == 1:
                result["panel_balance"] = "balanced"
            else:
                result["panel_balance"] = "unbalanced"

        return result

    def summary(self) -> str:
        """
        Generate data summary.

        Returns
        -------
        str
            Formatted summary of the causal input data
        """
        lines = []
        lines.append("=" * 50)
        lines.append("CAUSAL INPUT SUMMARY")
        lines.append("=" * 50)
        lines.append(f"Observations: {len(self.data):,}")
        lines.append(f"Variables: {len(self.data.columns)}")
        lines.append(f"Data Type: {self.get_data_type()}")
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
            lines.append(f"  - Balanced: {self.is_balanced()}")
            if self.panel_type:
                lines.append(f"  - Panel Type: {self.panel_type}")

        # Method-specific fields
        method_fields = []
        if self.cluster_var:
            method_fields.append(f"Cluster Variable: {self.cluster_var}")
        if self.weights:
            method_fields.append(f"Weights Variable: {self.weights}")
        if self.instrument:
            method_fields.append(f"Instrument (IV): {self.instrument}")
        if self.running_var:
            method_fields.append(f"Running Variable (RD): {self.running_var}")
        if self.cutoff is not None:
            method_fields.append(f"Cutoff (RD): {self.cutoff}")
        if self.mediator:
            method_fields.append(f"Mediator: {self.mediator}")

        if method_fields:
            lines.append("\nMethod-Specific Fields:")
            for field_info in method_fields:
                lines.append(f"  - {field_info}")

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


def generate_synthetic_psm_data(
    n: int = 1000,
    n_confounders: int = 5,
    treatment_effect: float = 2.0,
    selection_strength: float = 1.0,
    noise_std: float = 1.0,
    overlap: str = "good",
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic data for propensity score matching with known treatment effect.

    Creates data with selection on observables, where treatment assignment depends
    on observable confounders. This is suitable for testing PSM and related methods.

    Parameters
    ----------
    n : int
        Number of observations
    n_confounders : int
        Number of confounding variables to generate
    treatment_effect : float
        True average treatment effect (ATE)
    selection_strength : float
        How strongly confounders affect treatment selection (higher = more selection)
    noise_std : float
        Standard deviation of outcome noise
    overlap : str
        Overlap quality: "good" (balanced), "moderate" (some imbalance), or "poor" (limited overlap)
    random_state : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (data, true_params) where true_params contains ground truth values
    """
    np.random.seed(random_state)

    # Generate confounders
    X = np.random.normal(0, 1, (n, n_confounders))
    confounder_names = [f'x{i+1}' for i in range(n_confounders)]

    # Generate propensity score based on confounders
    # Linear combination of confounders affects treatment probability
    propensity_index = np.sum(X[:, :min(3, n_confounders)] * selection_strength, axis=1)

    # Adjust overlap based on parameter
    if overlap == "poor":
        propensity_index *= 1.5  # More extreme propensities
    elif overlap == "moderate":
        propensity_index *= 1.0
    else:  # good
        propensity_index *= 0.5  # More moderate propensities

    # Convert to probabilities via logistic function
    true_propensity = 1 / (1 + np.exp(-propensity_index))

    # Assign treatment based on propensity
    treatment = np.random.binomial(1, true_propensity)

    # Generate potential outcomes
    # Y(0): baseline outcome depends on confounders
    y0 = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.normal(0, noise_std, n)

    # Y(1): treated outcome with additional effect
    y1 = y0 + treatment_effect

    # Observed outcome
    y = treatment * y1 + (1 - treatment) * y0

    # Create DataFrame
    data = pd.DataFrame(X, columns=confounder_names)
    data['treatment'] = treatment
    data['y'] = y
    data['true_propensity'] = true_propensity

    # Calculate true effects
    true_ate = treatment_effect
    true_att = treatment_effect  # No heterogeneity in this simple design

    true_params = {
        'true_ate': true_ate,
        'true_att': true_att,
        'n': n,
        'n_confounders': n_confounders,
        'selection_strength': selection_strength,
        'overlap': overlap,
        'noise_std': noise_std,
        'treatment_rate': treatment.mean(),
        'propensity_range': (true_propensity.min(), true_propensity.max())
    }

    return data, true_params


def generate_synthetic_ddml_data(
    n: int = 2000,
    n_controls: int = 10,
    n_instruments: int = 0,
    treatment_effect: float = 1.5,
    nonlinearity: str = "moderate",
    sparsity: float = 0.5,
    noise_std: float = 1.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic data for Double/Debiased Machine Learning (DDML).

    Creates high-dimensional data with complex (potentially nonlinear) relationships
    between confounders, treatment, and outcome. Suitable for testing DML methods
    like partially linear regression (PLR) and interactive regression models (IRM).

    Parameters
    ----------
    n : int
        Number of observations
    n_controls : int
        Number of control/confounding variables
    n_instruments : int
        Number of additional instrument-like variables (0 for standard DDML)
    treatment_effect : float
        True causal effect of treatment on outcome
    nonlinearity : str
        Degree of nonlinearity: "none" (linear), "moderate" (polynomial), "high" (complex)
    sparsity : float
        Proportion of controls that truly affect outcome (0-1)
    noise_std : float
        Standard deviation of outcome noise
    random_state : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (data, true_params) where true_params contains ground truth values
    """
    np.random.seed(random_state)

    # Generate control variables
    X = np.random.normal(0, 1, (n, n_controls))
    control_names = [f'x{i+1}' for i in range(n_controls)]

    # Determine which controls are "active" (truly affect outcome)
    n_active = max(1, int(n_controls * sparsity))
    active_indices = np.random.choice(n_controls, n_active, replace=False)
    active_mask = np.zeros(n_controls, dtype=bool)
    active_mask[active_indices] = True

    # Generate treatment propensity based on controls
    if nonlinearity == "none":
        # Linear propensity
        g_x = 0.3 * np.sum(X[:, active_mask], axis=1)
    elif nonlinearity == "moderate":
        # Polynomial terms
        g_x = (0.3 * np.sum(X[:, active_mask], axis=1) +
               0.1 * np.sum(X[:, active_mask]**2, axis=1))
    else:  # high
        # More complex nonlinearity
        g_x = (0.2 * np.sum(X[:, active_mask], axis=1) +
               0.1 * np.sum(X[:, active_mask]**2, axis=1) +
               0.05 * np.sum(np.sin(X[:, active_mask]), axis=1))

    # Convert to probabilities
    true_propensity = 1 / (1 + np.exp(-g_x))
    treatment = np.random.binomial(1, true_propensity)

    # Generate outcome based on controls (nuisance function m(X))
    if nonlinearity == "none":
        m_x = 0.5 * np.sum(X[:, active_mask], axis=1)
    elif nonlinearity == "moderate":
        m_x = (0.5 * np.sum(X[:, active_mask], axis=1) +
               0.2 * np.sum(X[:, active_mask]**2, axis=1))
    else:  # high
        m_x = (0.3 * np.sum(X[:, active_mask], axis=1) +
               0.2 * np.sum(X[:, active_mask]**2, axis=1) +
               0.1 * np.sum(np.sin(2 * X[:, active_mask]), axis=1) +
               0.1 * X[:, active_mask[0] if active_mask.any() else 0] *
               X[:, active_mask[-1] if active_mask.any() else 0])

    # Outcome: Y = theta * D + m(X) + noise
    y = treatment_effect * treatment + m_x + np.random.normal(0, noise_std, n)

    # Create DataFrame
    data = pd.DataFrame(X, columns=control_names)
    data['d'] = treatment
    data['y'] = y

    # Add instruments if requested
    if n_instruments > 0:
        Z = np.random.normal(0, 1, (n, n_instruments))
        for i in range(n_instruments):
            data[f'z{i+1}'] = Z[:, i]

    true_params = {
        'true_effect': treatment_effect,
        'n': n,
        'n_controls': n_controls,
        'n_instruments': n_instruments,
        'nonlinearity': nonlinearity,
        'sparsity': sparsity,
        'n_active_controls': n_active,
        'active_control_indices': active_indices.tolist(),
        'noise_std': noise_std,
        'treatment_rate': treatment.mean()
    }

    return data, true_params


def generate_synthetic_panel_data(
    n_units: int = 100,
    n_periods: int = 10,
    treatment_effect: float = 1.5,
    unit_fe_std: float = 2.0,
    time_trend: float = 0.1,
    ar_coef: float = 0.5,
    noise_std: float = 1.0,
    treatment_type: str = "staggered",
    treatment_share: float = 0.5,
    balanced: bool = True,
    missing_rate: float = 0.0,
    n_covariates: int = 3,
    cluster_var: bool = True,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic panel data with flexible treatment assignment patterns.

    Creates panel data suitable for various estimators including DID, two-way
    fixed effects, synthetic control, and event study designs.

    Parameters
    ----------
    n_units : int
        Number of units (individuals, firms, states, etc.)
    n_periods : int
        Number of time periods
    treatment_effect : float
        True treatment effect
    unit_fe_std : float
        Standard deviation of unit fixed effects
    time_trend : float
        Linear time trend coefficient
    ar_coef : float
        AR(1) coefficient for outcome dynamics (0 = no persistence)
    noise_std : float
        Standard deviation of idiosyncratic noise
    treatment_type : str
        Treatment assignment pattern:
        - "staggered": Units adopt treatment at different times
        - "simultaneous": All treated units treated at same time
        - "random": Random treatment assignment each period
    treatment_share : float
        Proportion of units that ever receive treatment (for staggered/simultaneous)
    balanced : bool
        If True, create balanced panel; if False, randomly drop observations
    missing_rate : float
        Proportion of observations to randomly drop (only if balanced=False)
    n_covariates : int
        Number of time-varying covariates to generate
    cluster_var : bool
        If True, add a cluster variable (e.g., for clustered SEs)
    random_state : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (data, true_params) where true_params contains ground truth values
    """
    np.random.seed(random_state)

    # Create panel structure
    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(n_periods), n_units)
    n_obs = len(units)

    # Generate unit fixed effects
    unit_fe = np.random.normal(0, unit_fe_std, n_units)

    # Generate time fixed effects (common shocks)
    time_fe = time_trend * np.arange(n_periods) + np.random.normal(0, 0.5, n_periods)

    # Assign treatment based on type
    n_treated_units = int(n_units * treatment_share)
    treated_units = np.random.choice(n_units, n_treated_units, replace=False)

    if treatment_type == "staggered":
        # Each treated unit has a random treatment start time
        treatment_starts = {}
        for u in treated_units:
            # Treatment starts somewhere in the middle periods
            treatment_starts[u] = np.random.randint(n_periods // 3, 2 * n_periods // 3)
        treatment = np.array([
            1 if units[i] in treatment_starts and periods[i] >= treatment_starts[units[i]]
            else 0
            for i in range(n_obs)
        ])
    elif treatment_type == "simultaneous":
        # All treated units treated at same time
        treatment_period = n_periods // 2
        treatment = np.array([
            1 if units[i] in treated_units and periods[i] >= treatment_period
            else 0
            for i in range(n_obs)
        ])
    else:  # random
        # Random treatment each period
        treatment = np.random.binomial(1, treatment_share, n_obs)

    # Generate time-varying covariates
    covariates = {}
    for j in range(n_covariates):
        if j == 0:
            # First covariate is continuous
            covariates[f'x{j+1}'] = np.random.normal(0, 1, n_obs)
        elif j == 1:
            # Second is binary
            covariates[f'x{j+1}'] = np.random.binomial(1, 0.5, n_obs)
        else:
            # Others are continuous
            covariates[f'x{j+1}'] = np.random.normal(0, 1, n_obs)

    # Generate outcome with AR(1) dynamics
    y = np.zeros(n_obs)
    for t in range(n_periods):
        for u in range(n_units):
            idx = u * n_periods + t
            base = unit_fe[u] + time_fe[t]

            # Add AR component if not first period
            if t > 0 and ar_coef > 0:
                prev_idx = u * n_periods + (t - 1)
                ar_component = ar_coef * (y[prev_idx] - unit_fe[u] - time_fe[t-1])
            else:
                ar_component = 0

            # Add covariate effects
            cov_effect = sum(0.3 * covariates[f'x{j+1}'][idx] for j in range(min(2, n_covariates)))

            # Add treatment effect
            treat_effect = treatment_effect * treatment[idx]

            # Add noise
            noise = np.random.normal(0, noise_std)

            y[idx] = base + ar_component + cov_effect + treat_effect + noise

    # Create DataFrame
    data = pd.DataFrame({
        'unit_id': units,
        'time': periods,
        'treatment': treatment,
        'y': y,
        **covariates
    })

    # Add cluster variable (e.g., groups of units)
    if cluster_var:
        n_clusters = max(5, n_units // 10)
        cluster_assignment = np.random.randint(0, n_clusters, n_units)
        data['cluster'] = cluster_assignment[data['unit_id']]

    # Add treatment group indicator (ever treated)
    data['ever_treated'] = data['unit_id'].isin(treated_units).astype(int)

    # Make unbalanced if requested
    if not balanced and missing_rate > 0:
        n_to_drop = int(n_obs * missing_rate)
        drop_indices = np.random.choice(n_obs, n_to_drop, replace=False)
        data = data.drop(drop_indices).reset_index(drop=True)

    # Add sample weights (for demonstration)
    data['weights'] = np.random.uniform(0.5, 1.5, len(data))

    # Calculate actual panel balance
    periods_per_unit = data.groupby('unit_id')['time'].nunique()
    is_balanced = periods_per_unit.nunique() == 1

    true_params = {
        'true_ate': treatment_effect,
        'n_units': n_units,
        'n_periods': n_periods,
        'n_treated_units': n_treated_units,
        'treatment_type': treatment_type,
        'unit_fe_std': unit_fe_std,
        'time_trend': time_trend,
        'ar_coef': ar_coef,
        'noise_std': noise_std,
        'n_covariates': n_covariates,
        'is_balanced': is_balanced,
        'n_obs_final': len(data),
        'treatment_rate': data['treatment'].mean()
    }

    return data, true_params
