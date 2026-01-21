"""
Propensity Score Matching (PSM) Estimator Implementation.

This module provides comprehensive PSM estimation including:
- Propensity score estimation (logit, ML methods)
- Multiple matching algorithms (NN, caliper, kernel, Mahalanobis)
- Balance checking and visualization
- ATT/ATE estimation
- Rosenbaum sensitivity analysis
- PSM + DID combination
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

# Import from shared lib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'lib' / 'python'))
from data_loader import CausalInput, CausalOutput
from diagnostics import DiagnosticResult, balance_test
from table_formatter import create_regression_table, create_diagnostic_report


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PSMValidationResult:
    """Result of PSM data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"PSM Data Validation: {status}"]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


@dataclass
class PropensityScoreResult:
    """Result of propensity score estimation."""
    propensity_scores: np.ndarray
    method: str
    model: Any
    auc: float = None
    feature_importance: Dict[str, float] = None

    def summary(self) -> str:
        lines = [
            "Propensity Score Estimation",
            f"  Method: {self.method}",
            f"  Min PS: {self.propensity_scores.min():.4f}",
            f"  Max PS: {self.propensity_scores.max():.4f}",
            f"  Mean PS: {self.propensity_scores.mean():.4f}"
        ]
        if self.auc is not None:
            lines.append(f"  AUC: {self.auc:.4f}")
        return "\n".join(lines)


@dataclass
class MatchingResult:
    """Result of matching procedure."""
    matched_data: pd.DataFrame
    matched_indices_treated: np.ndarray
    matched_indices_control: np.ndarray
    weights: np.ndarray
    n_treated: int
    n_control_matched: int
    n_unmatched: int
    method: str
    parameters: Dict[str, Any]

    def summary(self) -> str:
        return (
            f"Matching Result ({self.method})\n"
            f"  Treated units: {self.n_treated}\n"
            f"  Matched controls: {self.n_control_matched}\n"
            f"  Unmatched treated: {self.n_unmatched}\n"
            f"  Match ratio: {self.n_control_matched / max(1, self.n_treated):.2f}"
        )


@dataclass
class BalanceResult:
    """Result of balance checking."""
    covariates: List[str]
    smd_before: Dict[str, float]
    smd_after: Dict[str, float]
    variance_ratio_before: Dict[str, float]
    variance_ratio_after: Dict[str, float]
    balanced: bool
    threshold: float = 0.1

    def summary(self) -> str:
        lines = ["Covariate Balance Summary"]
        lines.append("-" * 50)
        lines.append(f"{'Variable':<15} {'SMD Before':>12} {'SMD After':>12} {'Status':>10}")
        lines.append("-" * 50)
        for cov in self.covariates:
            before = self.smd_before.get(cov, np.nan)
            after = self.smd_after.get(cov, np.nan)
            status = "OK" if abs(after) < self.threshold else "IMBALANCED"
            lines.append(f"{cov:<15} {before:>12.4f} {after:>12.4f} {status:>10}")
        lines.append("-" * 50)
        lines.append(f"Overall balance achieved: {'YES' if self.balanced else 'NO'}")
        return "\n".join(lines)


@dataclass
class SensitivityResult:
    """Result of Rosenbaum sensitivity analysis."""
    gamma_values: List[float]
    lower_bounds: List[float]
    upper_bounds: List[float]
    p_values_upper: List[float]
    critical_gamma: float

    def summary(self) -> str:
        lines = ["Rosenbaum Sensitivity Analysis"]
        lines.append("-" * 55)
        lines.append(f"{'Gamma':>8} {'Lower Bound':>14} {'Upper Bound':>14} {'P-value':>12}")
        lines.append("-" * 55)
        for g, lb, ub, p in zip(self.gamma_values, self.lower_bounds,
                                self.upper_bounds, self.p_values_upper):
            lines.append(f"{g:>8.2f} {lb:>14.4f} {ub:>14.4f} {p:>12.4f}")
        lines.append("-" * 55)
        lines.append(f"Critical Gamma (significance lost): {self.critical_gamma:.2f}")
        return "\n".join(lines)


# =============================================================================
# Data Validation
# =============================================================================

def validate_psm_data(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: List[str]
) -> PSMValidationResult:
    """
    Validate data structure for PSM estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable name
    treatment : str
        Treatment indicator (0/1)
    covariates : List[str]
        Pre-treatment covariates

    Returns
    -------
    PSMValidationResult
        Validation results with errors and warnings
    """
    errors = []
    warnings_list = []
    summary = {}

    # Check required columns exist
    required_cols = [outcome, treatment] + covariates
    for col in required_cols:
        if col not in data.columns:
            errors.append(f"Required column '{col}' not found in data")

    if errors:
        return PSMValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings_list,
            summary=summary
        )

    # Check treatment is binary
    treatment_vals = data[treatment].dropna().unique()
    if not set(treatment_vals).issubset({0, 1}):
        errors.append(
            f"Treatment variable must be binary (0/1). Found values: {treatment_vals}"
        )

    # Check for missing values
    for col in required_cols:
        n_missing = data[col].isna().sum()
        if n_missing > 0:
            warnings_list.append(f"Column '{col}' has {n_missing} missing values")

    # Sample sizes
    n_treated = (data[treatment] == 1).sum()
    n_control = (data[treatment] == 0).sum()
    summary['n_treated'] = n_treated
    summary['n_control'] = n_control
    summary['n_total'] = len(data)

    if n_treated == 0:
        errors.append("No treated observations found")
    if n_control == 0:
        errors.append("No control observations found")

    if n_treated < 10:
        warnings_list.append(f"Very few treated observations ({n_treated})")
    if n_control < 10:
        warnings_list.append(f"Very few control observations ({n_control})")

    # Check covariate variation
    for cov in covariates:
        if data[cov].nunique() == 1:
            warnings_list.append(f"Covariate '{cov}' has no variation")

    is_valid = len(errors) == 0

    return PSMValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings_list,
        summary=summary
    )


# =============================================================================
# Propensity Score Estimation
# =============================================================================

def estimate_propensity_score(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    method: str = "logit",
    **kwargs
) -> PropensityScoreResult:
    """
    Estimate propensity scores using various methods.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    treatment : str
        Treatment indicator (0/1)
    covariates : List[str]
        Pre-treatment covariates
    method : str
        Estimation method: 'logit', 'probit', 'gbm', 'random_forest', 'lasso'
    **kwargs
        Method-specific parameters

    Returns
    -------
    PropensityScoreResult
        Estimated propensity scores and model diagnostics
    """
    X = data[covariates].values
    y = data[treatment].values

    # Handle missing values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]

    if method == "logit":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            penalty=kwargs.get('penalty', 'l2'),
            C=kwargs.get('C', 1.0),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42)
        )
        model.fit(X_clean, y_clean)
        ps = model.predict_proba(X_clean)[:, 1]

        # Feature importance (coefficients)
        feature_importance = dict(zip(covariates, model.coef_[0]))

    elif method == "probit":
        # Use statsmodels for probit
        try:
            import statsmodels.api as sm
            X_const = sm.add_constant(X_clean)
            model = sm.Probit(y_clean, X_const)
            result = model.fit(disp=0)
            ps = result.predict(X_const)
            feature_importance = dict(zip(covariates, result.params[1:]))
        except ImportError:
            raise ImportError("statsmodels required for probit. Install with: pip install statsmodels")

    elif method == "gbm":
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 3),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
            model.fit(X_clean, y_clean)
            ps = model.predict_proba(X_clean)[:, 1]
            feature_importance = dict(zip(covariates, model.feature_importances_))
        except ImportError:
            raise ImportError("scikit-learn required for GBM")

    elif method == "random_forest":
        try:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 5),
                random_state=kwargs.get('random_state', 42)
            )
            model.fit(X_clean, y_clean)
            ps = model.predict_proba(X_clean)[:, 1]
            feature_importance = dict(zip(covariates, model.feature_importances_))
        except ImportError:
            raise ImportError("scikit-learn required for random forest")

    elif method == "lasso":
        from sklearn.linear_model import LogisticRegressionCV

        model = LogisticRegressionCV(
            penalty='l1',
            solver='saga',
            cv=kwargs.get('cv', 5),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42)
        )
        model.fit(X_clean, y_clean)
        ps = model.predict_proba(X_clean)[:, 1]
        feature_importance = dict(zip(covariates, model.coef_[0]))

    else:
        raise ValueError(f"Unknown method: {method}. Use 'logit', 'probit', 'gbm', 'random_forest', or 'lasso'")

    # Calculate AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_clean, ps)
    except Exception:
        auc = None

    # Map back to full dataset
    ps_full = np.full(len(data), np.nan)
    ps_full[mask] = ps

    return PropensityScoreResult(
        propensity_scores=ps_full,
        method=method,
        model=model,
        auc=auc,
        feature_importance=feature_importance
    )


# =============================================================================
# Common Support / Overlap Checking
# =============================================================================

def check_common_support(
    ps_treated: np.ndarray,
    ps_control: np.ndarray,
    method: str = "minmax",
    trim_pct: float = 0.05
) -> DiagnosticResult:
    """
    Check common support (overlap) assumption.

    Parameters
    ----------
    ps_treated : np.ndarray
        Propensity scores for treated units
    ps_control : np.ndarray
        Propensity scores for control units
    method : str
        Method for defining common support:
        - 'minmax': [max(min_T, min_C), min(max_T, max_C)]
        - 'trim_pct': Trim extreme percentiles
    trim_pct : float
        Percentile to trim (if method='trim_pct')

    Returns
    -------
    DiagnosticResult
        Overlap diagnostic result
    """
    ps_treated = ps_treated[~np.isnan(ps_treated)]
    ps_control = ps_control[~np.isnan(ps_control)]

    if method == "minmax":
        lower = max(ps_treated.min(), ps_control.min())
        upper = min(ps_treated.max(), ps_control.max())
    elif method == "trim_pct":
        lower = max(
            np.percentile(ps_treated, trim_pct * 100),
            np.percentile(ps_control, trim_pct * 100)
        )
        upper = min(
            np.percentile(ps_treated, (1 - trim_pct) * 100),
            np.percentile(ps_control, (1 - trim_pct) * 100)
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Count observations in common support
    n_treated_in = np.sum((ps_treated >= lower) & (ps_treated <= upper))
    n_control_in = np.sum((ps_control >= lower) & (ps_control <= upper))
    n_treated_out = len(ps_treated) - n_treated_in
    n_control_out = len(ps_control) - n_control_in

    pct_treated_in = n_treated_in / len(ps_treated) * 100
    pct_control_in = n_control_in / len(ps_control) * 100

    # Overlap measure: proportion of PS range that is common
    total_range = max(ps_treated.max(), ps_control.max()) - min(ps_treated.min(), ps_control.min())
    common_range = upper - lower
    overlap_ratio = common_range / total_range if total_range > 0 else 0

    # Determine if overlap is sufficient
    passed = (pct_treated_in >= 90) and (pct_control_in >= 90) and (overlap_ratio >= 0.5)

    if passed:
        interpretation = (
            f"Common support assumption SATISFIED: "
            f"{pct_treated_in:.1f}% of treated and {pct_control_in:.1f}% of controls "
            f"fall within common support region [{lower:.4f}, {upper:.4f}]"
        )
    else:
        interpretation = (
            f"Common support assumption VIOLATED: "
            f"Only {pct_treated_in:.1f}% of treated and {pct_control_in:.1f}% of controls "
            f"fall within common support. Consider trimming or caliper matching."
        )

    return DiagnosticResult(
        test_name="Common Support (Overlap) Check",
        statistic=overlap_ratio,
        p_value=np.nan,
        passed=passed,
        threshold=0.5,
        interpretation=interpretation,
        details={
            'common_support_lower': lower,
            'common_support_upper': upper,
            'n_treated_in_support': n_treated_in,
            'n_control_in_support': n_control_in,
            'n_treated_out': n_treated_out,
            'n_control_out': n_control_out,
            'pct_treated_in': pct_treated_in,
            'pct_control_in': pct_control_in,
            'overlap_ratio': overlap_ratio
        }
    )


def plot_propensity_overlap(
    ps_treated: np.ndarray,
    ps_control: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Propensity Score Distribution by Treatment Status"
) -> Any:
    """
    Plot propensity score distributions for overlap visualization.

    Parameters
    ----------
    ps_treated : np.ndarray
        Propensity scores for treated units
    ps_control : np.ndarray
        Propensity scores for control units
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        Overlap plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1 = axes[0]
    bins = np.linspace(0, 1, 50)
    ax1.hist(ps_control, bins=bins, alpha=0.6, label='Control', color='blue', density=True)
    ax1.hist(ps_treated, bins=bins, alpha=0.6, label='Treated', color='red', density=True)
    ax1.set_xlabel('Propensity Score')
    ax1.set_ylabel('Density')
    ax1.set_title('PS Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2 = axes[1]
    data_box = [ps_control[~np.isnan(ps_control)], ps_treated[~np.isnan(ps_treated)]]
    bp = ax2.boxplot(data_box, labels=['Control', 'Treated'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Propensity Score')
    ax2.set_title('PS Box Plot')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    return fig


# =============================================================================
# Matching Algorithms
# =============================================================================

def match_nearest_neighbor(
    data: pd.DataFrame,
    propensity_score: str,
    treatment: str,
    n_neighbors: int = 1,
    replacement: bool = True,
    caliper: float = None,
    caliper_scale: str = "ps_std"
) -> MatchingResult:
    """
    Perform nearest neighbor matching on propensity scores.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with propensity scores
    propensity_score : str
        Column name for propensity scores
    treatment : str
        Treatment indicator column
    n_neighbors : int
        Number of control matches per treated unit
    replacement : bool
        Whether to match with replacement
    caliper : float
        Maximum allowed PS distance (None for no caliper)
    caliper_scale : str
        Scale for caliper: 'ps_std' (multiples of PS std) or 'ps_raw'

    Returns
    -------
    MatchingResult
        Matching results with indices and weights
    """
    from sklearn.neighbors import NearestNeighbors

    df = data.copy()

    # Separate treated and control
    treated_mask = df[treatment] == 1
    control_mask = df[treatment] == 0

    ps_treated = df.loc[treated_mask, propensity_score].values.reshape(-1, 1)
    ps_control = df.loc[control_mask, propensity_score].values.reshape(-1, 1)

    treated_idx = df.index[treated_mask].values
    control_idx = df.index[control_mask].values

    # Handle missing PS values
    valid_treated = ~np.isnan(ps_treated.ravel())
    valid_control = ~np.isnan(ps_control.ravel())

    ps_treated_clean = ps_treated[valid_treated]
    ps_control_clean = ps_control[valid_control]
    treated_idx_clean = treated_idx[valid_treated]
    control_idx_clean = control_idx[valid_control]

    # Compute caliper if specified
    if caliper is not None:
        if caliper_scale == "ps_std":
            ps_std = np.std(df[propensity_score].dropna())
            caliper_raw = caliper * ps_std
        else:
            caliper_raw = caliper
    else:
        caliper_raw = np.inf

    # Build nearest neighbor model on controls
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(ps_control_clean)), metric='euclidean')
    nn.fit(ps_control_clean)

    # Find matches for each treated unit
    distances, indices = nn.kneighbors(ps_treated_clean)

    matched_treated = []
    matched_control = []
    weights = []
    unmatched_count = 0

    used_controls = set() if not replacement else None

    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        valid_matches = []

        for d, j in zip(dist_row, idx_row):
            # Check caliper
            if d > caliper_raw:
                continue

            # Check if already used (if no replacement)
            if not replacement and j in used_controls:
                continue

            valid_matches.append((j, d))

        if len(valid_matches) == 0:
            unmatched_count += 1
            continue

        # Use top n_neighbors valid matches
        for j, d in valid_matches[:n_neighbors]:
            matched_treated.append(treated_idx_clean[i])
            matched_control.append(control_idx_clean[j])
            weights.append(1.0 / len(valid_matches[:n_neighbors]))

            if not replacement:
                used_controls.add(j)

    # Create matched dataset
    matched_treated = np.array(matched_treated)
    matched_control = np.array(matched_control)
    weights = np.array(weights)

    # Build matched dataframe
    treated_rows = df.loc[np.unique(matched_treated)].copy()
    treated_rows['_match_weight'] = treated_rows.index.map(
        lambda x: weights[matched_treated == x].sum()
    )

    control_weights = pd.Series(weights, index=matched_control).groupby(level=0).sum()
    control_rows = df.loc[control_weights.index].copy()
    control_rows['_match_weight'] = control_weights.values

    matched_data = pd.concat([treated_rows, control_rows])

    return MatchingResult(
        matched_data=matched_data,
        matched_indices_treated=matched_treated,
        matched_indices_control=matched_control,
        weights=weights,
        n_treated=len(np.unique(matched_treated)),
        n_control_matched=len(np.unique(matched_control)),
        n_unmatched=unmatched_count,
        method="nearest_neighbor",
        parameters={
            'n_neighbors': n_neighbors,
            'replacement': replacement,
            'caliper': caliper,
            'caliper_scale': caliper_scale
        }
    )


def match_kernel(
    data: pd.DataFrame,
    propensity_score: str,
    treatment: str,
    kernel: str = "epanechnikov",
    bandwidth: Union[str, float] = "optimal"
) -> MatchingResult:
    """
    Perform kernel matching on propensity scores.

    All controls contribute to each treated unit's counterfactual,
    weighted by kernel function of PS distance.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with propensity scores
    propensity_score : str
        Column name for propensity scores
    treatment : str
        Treatment indicator column
    kernel : str
        Kernel function: 'epanechnikov', 'gaussian', 'uniform'
    bandwidth : str or float
        Bandwidth: 'optimal' (Silverman's rule) or numeric value

    Returns
    -------
    MatchingResult
        Matching results with kernel weights
    """
    df = data.copy()

    # Separate treated and control
    treated_mask = df[treatment] == 1
    control_mask = df[treatment] == 0

    ps_treated = df.loc[treated_mask, propensity_score].values
    ps_control = df.loc[control_mask, propensity_score].values

    treated_idx = df.index[treated_mask].values
    control_idx = df.index[control_mask].values

    # Handle missing values
    valid_treated = ~np.isnan(ps_treated)
    valid_control = ~np.isnan(ps_control)

    ps_treated = ps_treated[valid_treated]
    ps_control = ps_control[valid_control]
    treated_idx = treated_idx[valid_treated]
    control_idx = control_idx[valid_control]

    # Calculate bandwidth
    if bandwidth == "optimal":
        # Silverman's rule of thumb
        ps_all = np.concatenate([ps_treated, ps_control])
        h = 1.06 * np.std(ps_all) * len(ps_all) ** (-1/5)
    else:
        h = bandwidth

    # Define kernel functions
    def epanechnikov(u):
        return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

    def gaussian(u):
        return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

    def uniform(u):
        return np.where(np.abs(u) <= 1, 0.5, 0)

    kernels = {
        'epanechnikov': epanechnikov,
        'gaussian': gaussian,
        'uniform': uniform
    }

    if kernel not in kernels:
        raise ValueError(f"Unknown kernel: {kernel}. Use 'epanechnikov', 'gaussian', or 'uniform'")

    kernel_fn = kernels[kernel]

    # Compute weights for each treated-control pair
    all_weights = []
    matched_treated = []
    matched_control = []

    for i, ps_t in enumerate(ps_treated):
        # Distance to all controls
        u = (ps_t - ps_control) / h
        weights = kernel_fn(u)

        # Normalize weights
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # No controls within kernel bandwidth
            continue

        # Store non-zero weights
        for j, w in enumerate(weights):
            if w > 0:
                matched_treated.append(treated_idx[i])
                matched_control.append(control_idx[j])
                all_weights.append(w)

    matched_treated = np.array(matched_treated)
    matched_control = np.array(matched_control)
    all_weights = np.array(all_weights)

    # Build matched dataset with weights
    # For kernel matching, all controls get some weight
    control_weights = pd.Series(all_weights, index=matched_control).groupby(level=0).sum()

    matched_data = df.copy()
    matched_data['_match_weight'] = 0.0
    matched_data.loc[matched_data[treatment] == 1, '_match_weight'] = 1.0
    matched_data.loc[control_weights.index, '_match_weight'] = control_weights.values

    # Keep only observations with positive weight
    matched_data = matched_data[matched_data['_match_weight'] > 0]

    return MatchingResult(
        matched_data=matched_data,
        matched_indices_treated=np.unique(matched_treated),
        matched_indices_control=np.unique(matched_control),
        weights=all_weights,
        n_treated=len(np.unique(matched_treated)),
        n_control_matched=len(np.unique(matched_control)),
        n_unmatched=len(ps_treated) - len(np.unique(matched_treated)),
        method="kernel",
        parameters={
            'kernel': kernel,
            'bandwidth': h
        }
    )


def match_mahalanobis(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    n_neighbors: int = 1,
    replacement: bool = True,
    caliper: float = None
) -> MatchingResult:
    """
    Perform Mahalanobis distance matching on covariates.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    treatment : str
        Treatment indicator column
    covariates : List[str]
        Covariates to match on
    n_neighbors : int
        Number of matches per treated unit
    replacement : bool
        Whether to match with replacement
    caliper : float
        Maximum Mahalanobis distance (None for no caliper)

    Returns
    -------
    MatchingResult
        Matching results
    """
    from sklearn.neighbors import NearestNeighbors

    df = data.copy()

    # Separate treated and control
    treated_mask = df[treatment] == 1
    control_mask = df[treatment] == 0

    X_treated = df.loc[treated_mask, covariates].values
    X_control = df.loc[control_mask, covariates].values

    treated_idx = df.index[treated_mask].values
    control_idx = df.index[control_mask].values

    # Handle missing values
    valid_treated = ~np.isnan(X_treated).any(axis=1)
    valid_control = ~np.isnan(X_control).any(axis=1)

    X_treated = X_treated[valid_treated]
    X_control = X_control[valid_control]
    treated_idx = treated_idx[valid_treated]
    control_idx = control_idx[valid_control]

    # Compute covariance matrix inverse for Mahalanobis distance
    X_all = np.vstack([X_treated, X_control])
    cov_matrix = np.cov(X_all.T)

    # Regularize if singular
    if np.linalg.cond(cov_matrix) > 1e10:
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6

    cov_inv = np.linalg.inv(cov_matrix)

    # Compute Mahalanobis distances
    distances = cdist(X_treated, X_control, metric='mahalanobis', VI=cov_inv)

    # Find matches
    matched_treated = []
    matched_control = []
    weights = []
    unmatched_count = 0

    used_controls = set() if not replacement else None

    for i in range(len(X_treated)):
        dist_row = distances[i]

        # Sort controls by distance
        sorted_idx = np.argsort(dist_row)
        valid_matches = []

        for j in sorted_idx:
            d = dist_row[j]

            # Check caliper
            if caliper is not None and d > caliper:
                continue

            # Check if already used
            if not replacement and j in used_controls:
                continue

            valid_matches.append((j, d))

            if len(valid_matches) >= n_neighbors:
                break

        if len(valid_matches) == 0:
            unmatched_count += 1
            continue

        for j, d in valid_matches:
            matched_treated.append(treated_idx[i])
            matched_control.append(control_idx[j])
            weights.append(1.0 / len(valid_matches))

            if not replacement:
                used_controls.add(j)

    matched_treated = np.array(matched_treated)
    matched_control = np.array(matched_control)
    weights = np.array(weights)

    # Build matched dataset
    treated_rows = df.loc[np.unique(matched_treated)].copy()
    treated_rows['_match_weight'] = treated_rows.index.map(
        lambda x: weights[matched_treated == x].sum()
    )

    control_weights = pd.Series(weights, index=matched_control).groupby(level=0).sum()
    control_rows = df.loc[control_weights.index].copy()
    control_rows['_match_weight'] = control_weights.values

    matched_data = pd.concat([treated_rows, control_rows])

    return MatchingResult(
        matched_data=matched_data,
        matched_indices_treated=matched_treated,
        matched_indices_control=matched_control,
        weights=weights,
        n_treated=len(np.unique(matched_treated)),
        n_control_matched=len(np.unique(matched_control)),
        n_unmatched=unmatched_count,
        method="mahalanobis",
        parameters={
            'covariates': covariates,
            'n_neighbors': n_neighbors,
            'replacement': replacement,
            'caliper': caliper
        }
    )


# =============================================================================
# Balance Checking
# =============================================================================

def check_balance(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    weights: str = None,
    threshold: float = 0.1
) -> BalanceResult:
    """
    Check covariate balance between treatment and control groups.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset (matched or unmatched)
    treatment : str
        Treatment indicator column
    covariates : List[str]
        Covariates to check
    weights : str
        Column name for matching weights (None for unweighted)
    threshold : float
        SMD threshold for balance (default 0.1)

    Returns
    -------
    BalanceResult
        Balance statistics
    """
    df = data.copy()

    treated_mask = df[treatment] == 1
    control_mask = df[treatment] == 0

    if weights is not None and weights in df.columns:
        w_treated = df.loc[treated_mask, weights].values
        w_control = df.loc[control_mask, weights].values
    else:
        w_treated = np.ones(treated_mask.sum())
        w_control = np.ones(control_mask.sum())

    smd = {}
    variance_ratio = {}

    for cov in covariates:
        x_treated = df.loc[treated_mask, cov].values
        x_control = df.loc[control_mask, cov].values

        # Handle missing values
        valid_t = ~np.isnan(x_treated)
        valid_c = ~np.isnan(x_control)

        x_treated = x_treated[valid_t]
        x_control = x_control[valid_c]
        wt = w_treated[valid_t]
        wc = w_control[valid_c]

        # Weighted means
        mean_t = np.average(x_treated, weights=wt) if len(x_treated) > 0 else np.nan
        mean_c = np.average(x_control, weights=wc) if len(x_control) > 0 else np.nan

        # Weighted variances
        if len(x_treated) > 1:
            var_t = np.average((x_treated - mean_t)**2, weights=wt)
        else:
            var_t = 0

        if len(x_control) > 1:
            var_c = np.average((x_control - mean_c)**2, weights=wc)
        else:
            var_c = 0

        # Standardized mean difference
        pooled_std = np.sqrt((var_t + var_c) / 2)
        if pooled_std > 0:
            smd[cov] = (mean_t - mean_c) / pooled_std
        else:
            smd[cov] = 0

        # Variance ratio
        if var_c > 0:
            variance_ratio[cov] = var_t / var_c
        else:
            variance_ratio[cov] = np.inf if var_t > 0 else 1.0

    # Check if all balanced
    balanced = all(abs(s) < threshold for s in smd.values())

    return BalanceResult(
        covariates=covariates,
        smd_before={},  # Filled by caller if comparing before/after
        smd_after=smd,
        variance_ratio_before={},
        variance_ratio_after=variance_ratio,
        balanced=balanced,
        threshold=threshold
    )


def create_balance_table(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    weights_after: str = None
) -> str:
    """
    Create formatted balance table showing before/after matching.

    Parameters
    ----------
    data_before : pd.DataFrame
        Original data
    data_after : pd.DataFrame
        Matched data
    treatment : str
        Treatment indicator
    covariates : List[str]
        Covariates to compare
    weights_after : str
        Weight column in matched data

    Returns
    -------
    str
        Formatted balance table
    """
    balance_before = check_balance(data_before, treatment, covariates, weights=None)
    balance_after = check_balance(data_after, treatment, covariates, weights=weights_after)

    # Update before SMD in after result
    balance_after.smd_before = balance_before.smd_after
    balance_after.variance_ratio_before = balance_before.variance_ratio_after

    lines = []
    lines.append("=" * 75)
    lines.append("COVARIATE BALANCE TABLE".center(75))
    lines.append("=" * 75)
    lines.append(f"{'Variable':<15} {'Treated':>10} {'Control':>10} {'SMD':>8} {'':>4} "
                 f"{'Treated':>10} {'Control':>10} {'SMD':>8}")
    lines.append(f"{'':15} {'Before':^31} {'':>4} {'After':^31}")
    lines.append("-" * 75)

    for cov in covariates:
        # Get means before
        treat_before = data_before.loc[data_before[treatment] == 1, cov].mean()
        ctrl_before = data_before.loc[data_before[treatment] == 0, cov].mean()
        smd_before = balance_before.smd_after.get(cov, np.nan)

        # Get means after
        if weights_after and weights_after in data_after.columns:
            w = data_after[weights_after]
            treat_mask = data_after[treatment] == 1
            ctrl_mask = data_after[treatment] == 0
            treat_after = np.average(data_after.loc[treat_mask, cov],
                                     weights=data_after.loc[treat_mask, weights_after])
            ctrl_after = np.average(data_after.loc[ctrl_mask, cov],
                                    weights=data_after.loc[ctrl_mask, weights_after])
        else:
            treat_after = data_after.loc[data_after[treatment] == 1, cov].mean()
            ctrl_after = data_after.loc[data_after[treatment] == 0, cov].mean()
        smd_after = balance_after.smd_after.get(cov, np.nan)

        lines.append(
            f"{cov:<15} {treat_before:>10.3f} {ctrl_before:>10.3f} {smd_before:>8.3f} {'':>4} "
            f"{treat_after:>10.3f} {ctrl_after:>10.3f} {smd_after:>8.3f}"
        )

    lines.append("-" * 75)
    lines.append(
        f"Balance achieved (|SMD| < {balance_after.threshold}): "
        f"{'YES' if balance_after.balanced else 'NO'}"
    )
    lines.append("=" * 75)

    return "\n".join(lines)


def plot_balance(
    balance_before: BalanceResult,
    balance_after: BalanceResult,
    threshold: float = 0.1,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Love Plot: Covariate Balance"
) -> Any:
    """
    Create Love plot showing balance improvement.

    Parameters
    ----------
    balance_before : BalanceResult
        Balance before matching
    balance_after : BalanceResult
        Balance after matching
    threshold : float
        SMD threshold line
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        Love plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    covariates = balance_after.covariates
    smd_before = [abs(balance_before.smd_after.get(c, balance_after.smd_before.get(c, 0)))
                  for c in covariates]
    smd_after = [abs(balance_after.smd_after.get(c, 0)) for c in covariates]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(covariates))

    ax.scatter(smd_before, y_pos, marker='o', color='red', s=100, label='Before Matching', zorder=3)
    ax.scatter(smd_after, y_pos, marker='s', color='blue', s=100, label='After Matching', zorder=3)

    # Connect before/after with lines
    for i, (sb, sa) in enumerate(zip(smd_before, smd_after)):
        ax.plot([sb, sa], [i, i], color='gray', linestyle='-', alpha=0.5, zorder=2)

    # Threshold line
    ax.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold})')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates)
    ax.set_xlabel('Absolute Standardized Mean Difference')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


# =============================================================================
# Effect Estimation
# =============================================================================

def estimate_att(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    weights: str = None,
    se_method: str = "bootstrap",
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> CausalOutput:
    """
    Estimate Average Treatment Effect on the Treated (ATT).

    Parameters
    ----------
    data : pd.DataFrame
        Matched dataset
    outcome : str
        Outcome variable
    treatment : str
        Treatment indicator
    weights : str
        Matching weights column (None for equal weights)
    se_method : str
        Standard error method: 'bootstrap', 'analytical'
    n_bootstrap : int
        Number of bootstrap iterations
    random_state : int
        Random seed

    Returns
    -------
    CausalOutput
        ATT estimate with standard error
    """
    df = data.copy()

    # Get weights
    if weights is not None and weights in df.columns:
        w = df[weights].values
    else:
        w = np.ones(len(df))

    treated_mask = df[treatment] == 1
    control_mask = df[treatment] == 0

    y_treated = df.loc[treated_mask, outcome].values
    y_control = df.loc[control_mask, outcome].values
    w_treated = w[treated_mask]
    w_control = w[control_mask]

    # Handle missing values
    valid_t = ~np.isnan(y_treated)
    valid_c = ~np.isnan(y_control)

    y_treated = y_treated[valid_t]
    y_control = y_control[valid_c]
    w_treated = w_treated[valid_t]
    w_control = w_control[valid_c]

    # Point estimate
    mean_treated = np.average(y_treated, weights=w_treated)
    mean_control = np.average(y_control, weights=w_control)
    att = mean_treated - mean_control

    # Standard error
    if se_method == "bootstrap":
        np.random.seed(random_state)
        boot_effects = []

        for _ in range(n_bootstrap):
            # Resample treated
            idx_t = np.random.choice(len(y_treated), len(y_treated), replace=True)
            # Resample control
            idx_c = np.random.choice(len(y_control), len(y_control), replace=True)

            boot_mean_t = np.average(y_treated[idx_t], weights=w_treated[idx_t])
            boot_mean_c = np.average(y_control[idx_c], weights=w_control[idx_c])
            boot_effects.append(boot_mean_t - boot_mean_c)

        se = np.std(boot_effects)
        ci_lower = np.percentile(boot_effects, 2.5)
        ci_upper = np.percentile(boot_effects, 97.5)

    else:  # analytical
        var_t = np.average((y_treated - mean_treated)**2, weights=w_treated)
        var_c = np.average((y_control - mean_control)**2, weights=w_control)

        n_eff_t = (w_treated.sum())**2 / (w_treated**2).sum()
        n_eff_c = (w_control.sum())**2 / (w_control**2).sum()

        se = np.sqrt(var_t / n_eff_t + var_c / n_eff_c)
        ci_lower = att - 1.96 * se
        ci_upper = att + 1.96 * se

    # P-value (two-sided)
    z_stat = att / se if se > 0 else np.inf
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    diagnostics = {
        'estimand': 'ATT',
        'method': 'matching',
        'se_method': se_method,
        'n_treated': len(y_treated),
        'n_control': len(y_control),
        'mean_treated': mean_treated,
        'mean_control': mean_control
    }

    return CausalOutput(
        effect=att,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        diagnostics=diagnostics,
        summary_table="",
        interpretation=f"ATT = {att:.4f} (SE = {se:.4f})"
    )


def estimate_ate(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    propensity_score: str,
    estimator: str = "ipw",
    trim: float = 0.01
) -> CausalOutput:
    """
    Estimate Average Treatment Effect (ATE) using IPW or doubly robust.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with propensity scores
    outcome : str
        Outcome variable
    treatment : str
        Treatment indicator
    propensity_score : str
        Propensity score column
    estimator : str
        'ipw' (Inverse Propensity Weighting) or 'doubly_robust'
    trim : float
        Trim propensity scores to [trim, 1-trim]

    Returns
    -------
    CausalOutput
        ATE estimate
    """
    df = data.copy()

    y = df[outcome].values
    d = df[treatment].values
    ps = df[propensity_score].values

    # Trim extreme propensity scores
    ps = np.clip(ps, trim, 1 - trim)

    # Handle missing values
    valid = ~(np.isnan(y) | np.isnan(d) | np.isnan(ps))
    y = y[valid]
    d = d[valid]
    ps = ps[valid]

    n = len(y)

    if estimator == "ipw":
        # Horvitz-Thompson IPW estimator
        # ATE = E[D*Y/ps] - E[(1-D)*Y/(1-ps)]
        ate = np.mean(d * y / ps) - np.mean((1 - d) * y / (1 - ps))

        # Normalized IPW (more stable)
        w_treated = d / ps
        w_control = (1 - d) / (1 - ps)
        ate_normalized = (
            np.sum(w_treated * y) / np.sum(w_treated) -
            np.sum(w_control * y) / np.sum(w_control)
        )
        ate = ate_normalized

        # Variance estimation
        # Influence function based variance
        psi_treated = d * (y - ate) / ps
        psi_control = (1 - d) * (y - ate) / (1 - ps)
        psi = psi_treated - psi_control

        var_ate = np.var(psi) / n
        se = np.sqrt(var_ate)

    elif estimator == "doubly_robust":
        # AIPW / Doubly Robust estimator
        # Requires outcome model
        try:
            from sklearn.linear_model import LinearRegression

            # Fit outcome model
            X = df[valid].drop(columns=[outcome, treatment, propensity_score])
            X = X.select_dtypes(include=[np.number])

            if len(X.columns) == 0:
                raise ValueError("No numeric covariates for outcome model")

            model = LinearRegression()
            model.fit(X[d == 1], y[d == 1])
            mu1 = model.predict(X)

            model.fit(X[d == 0], y[d == 0])
            mu0 = model.predict(X)

            # AIPW estimator
            ate = np.mean(
                d * (y - mu1) / ps + mu1 -
                (1 - d) * (y - mu0) / (1 - ps) - mu0
            )

            # Influence function based variance
            psi = (
                d * (y - mu1) / ps + mu1 -
                (1 - d) * (y - mu0) / (1 - ps) - mu0 - ate
            )
            var_ate = np.var(psi) / n
            se = np.sqrt(var_ate)

        except Exception as e:
            warnings.warn(f"Doubly robust estimation failed: {e}. Falling back to IPW.")
            return estimate_ate(data, outcome, treatment, propensity_score, estimator="ipw")

    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    # Inference
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    z_stat = ate / se if se > 0 else np.inf
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    diagnostics = {
        'estimand': 'ATE',
        'estimator': estimator,
        'n': n,
        'trim': trim
    }

    return CausalOutput(
        effect=ate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        diagnostics=diagnostics,
        summary_table="",
        interpretation=f"ATE = {ate:.4f} (SE = {se:.4f})"
    )


# =============================================================================
# Sensitivity Analysis
# =============================================================================

def rosenbaum_sensitivity(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    matched_pairs: np.ndarray = None,
    gamma_range: List[float] = None
) -> SensitivityResult:
    """
    Rosenbaum bounds sensitivity analysis for hidden bias.

    Parameters
    ----------
    data : pd.DataFrame
        Matched dataset
    outcome : str
        Outcome variable
    treatment : str
        Treatment indicator
    matched_pairs : np.ndarray
        Array of (treated_idx, control_idx) pairs. If None, assumes 1:1 matching.
    gamma_range : List[float]
        Range of Gamma values to test

    Returns
    -------
    SensitivityResult
        Sensitivity analysis results
    """
    if gamma_range is None:
        gamma_range = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    df = data.copy()

    # Get treated and control outcomes
    y_treated = df.loc[df[treatment] == 1, outcome].values
    y_control = df.loc[df[treatment] == 0, outcome].values

    # For simplicity, use paired difference approach
    # This assumes 1:1 matching with same ordering
    if matched_pairs is None:
        n_pairs = min(len(y_treated), len(y_control))
        y_treated = y_treated[:n_pairs]
        y_control = y_control[:n_pairs]
    else:
        # Use provided pairs
        y_treated = df.loc[matched_pairs[:, 0], outcome].values
        y_control = df.loc[matched_pairs[:, 1], outcome].values
        n_pairs = len(matched_pairs)

    # Paired differences
    diff = y_treated - y_control
    n_pairs = len(diff)

    # Wilcoxon signed-rank statistic
    # Under null, sum of positive ranks
    ranks = stats.rankdata(np.abs(diff))
    positive_ranks = ranks[diff > 0]
    W = positive_ranks.sum()

    # Expected value and variance under different Gamma
    lower_bounds = []
    upper_bounds = []
    p_values_upper = []

    for gamma in gamma_range:
        # Under hidden bias, each pair can have probability p or 1-p
        # of positive difference, where p/(1-p) varies by Gamma

        # Conservative bounds on p-value
        # Upper bound: maximize p-value (minimize evidence)
        # This uses the Wilcoxon signed-rank distribution bounds

        # For Gamma = 1, standard Wilcoxon test
        if gamma == 1.0:
            _, p_upper = stats.wilcoxon(diff, alternative='greater')
            lb = ub = np.mean(diff)
        else:
            # Approximate bounds using Hodges-Lehmann estimator approach
            # This is a simplified version

            # Probability bounds
            p_low = 1 / (1 + gamma)
            p_high = gamma / (1 + gamma)

            # Effect bounds
            sorted_diff = np.sort(diff)
            # Lower bound: pessimistic assignment
            lb_idx = int(n_pairs * (1 - p_high))
            ub_idx = int(n_pairs * p_high)

            lb = np.percentile(diff, 100 * p_low)
            ub = np.percentile(diff, 100 * p_high)

            # P-value under hidden bias (conservative)
            # Uses McNemar-type adjustment
            n_pos = (diff > 0).sum()
            n_neg = (diff < 0).sum()

            # Under gamma, probability of positive can be as low as p_low
            if n_pos > 0:
                expected_pos_low = n_pairs * p_low
                z_stat = (n_pos - expected_pos_low) / np.sqrt(n_pairs * p_low * (1 - p_low))
                p_upper = 1 - stats.norm.cdf(z_stat)
            else:
                p_upper = 1.0

        lower_bounds.append(lb)
        upper_bounds.append(ub)
        p_values_upper.append(p_upper)

    # Find critical Gamma (where p-value exceeds 0.05)
    critical_gamma = gamma_range[-1]
    for g, p in zip(gamma_range, p_values_upper):
        if p > 0.05:
            critical_gamma = g
            break

    return SensitivityResult(
        gamma_values=gamma_range,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        p_values_upper=p_values_upper,
        critical_gamma=critical_gamma
    )


# =============================================================================
# Full PSM Analysis Workflow
# =============================================================================

def run_full_psm_analysis(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: List[str],
    match_method: str = "nearest_neighbor",
    ps_method: str = "logit",
    caliper: float = 0.1,
    n_neighbors: int = 1,
    replacement: bool = True,
    estimand: str = "ATT",
    run_sensitivity: bool = True
) -> CausalOutput:
    """
    Run complete PSM analysis workflow.

    This function:
    1. Validates data structure
    2. Estimates propensity scores
    3. Checks common support
    4. Performs matching
    5. Checks covariate balance
    6. Estimates treatment effect
    7. Runs sensitivity analysis

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    outcome : str
        Outcome variable
    treatment : str
        Treatment indicator
    covariates : List[str]
        Pre-treatment covariates
    match_method : str
        Matching method: 'nearest_neighbor', 'kernel', 'mahalanobis'
    ps_method : str
        PS estimation method: 'logit', 'probit', 'gbm', 'random_forest'
    caliper : float
        Caliper for NN matching
    n_neighbors : int
        Number of matches for NN
    replacement : bool
        Match with replacement
    estimand : str
        'ATT' or 'ATE'
    run_sensitivity : bool
        Whether to run Rosenbaum sensitivity analysis

    Returns
    -------
    CausalOutput
        Complete analysis results
    """
    df = data.copy()

    # Step 1: Validate data
    validation = validate_psm_data(df, outcome, treatment, covariates)
    if not validation.is_valid:
        raise ValueError(f"Data validation failed: {validation.errors}")

    # Step 2: Estimate propensity scores
    ps_result = estimate_propensity_score(
        df, treatment, covariates, method=ps_method
    )
    df['_propensity_score'] = ps_result.propensity_scores

    # Step 3: Check common support
    ps_treated = df.loc[df[treatment] == 1, '_propensity_score'].values
    ps_control = df.loc[df[treatment] == 0, '_propensity_score'].values
    overlap_result = check_common_support(ps_treated, ps_control)

    # Step 4: Perform matching
    if match_method == "nearest_neighbor":
        match_result = match_nearest_neighbor(
            df, '_propensity_score', treatment,
            n_neighbors=n_neighbors,
            replacement=replacement,
            caliper=caliper,
            caliper_scale="ps_std"
        )
    elif match_method == "kernel":
        match_result = match_kernel(
            df, '_propensity_score', treatment,
            kernel='epanechnikov',
            bandwidth='optimal'
        )
    elif match_method == "mahalanobis":
        match_result = match_mahalanobis(
            df, treatment, covariates,
            n_neighbors=n_neighbors,
            replacement=replacement
        )
    else:
        raise ValueError(f"Unknown matching method: {match_method}")

    # Step 5: Check balance
    balance_before = check_balance(df, treatment, covariates)
    balance_after = check_balance(
        match_result.matched_data, treatment, covariates,
        weights='_match_weight'
    )
    balance_after.smd_before = balance_before.smd_after
    balance_after.variance_ratio_before = balance_before.variance_ratio_after

    # Step 6: Estimate effect
    if estimand == "ATT":
        effect_result = estimate_att(
            match_result.matched_data,
            outcome, treatment,
            weights='_match_weight'
        )
    else:
        effect_result = estimate_ate(
            df, outcome, treatment, '_propensity_score'
        )

    # Step 7: Sensitivity analysis
    sensitivity_result = None
    if run_sensitivity and estimand == "ATT":
        try:
            sensitivity_result = rosenbaum_sensitivity(
                match_result.matched_data, outcome, treatment
            )
        except Exception as e:
            warnings.warn(f"Sensitivity analysis failed: {e}")

    # Compile diagnostics
    all_diagnostics = {
        'validation': validation.summary,
        'propensity_score': {
            'method': ps_result.method,
            'auc': ps_result.auc,
            'ps_min': ps_result.propensity_scores[~np.isnan(ps_result.propensity_scores)].min(),
            'ps_max': ps_result.propensity_scores[~np.isnan(ps_result.propensity_scores)].max(),
            'ps_mean': np.nanmean(ps_result.propensity_scores)
        },
        'common_support': overlap_result,
        'matching': {
            'method': match_result.method,
            'n_treated': match_result.n_treated,
            'n_control_matched': match_result.n_control_matched,
            'n_unmatched': match_result.n_unmatched
        },
        'balance': balance_after,
        'effect': effect_result.diagnostics
    }

    if sensitivity_result:
        all_diagnostics['sensitivity'] = sensitivity_result

    # Generate summary
    summary_lines = [
        "=" * 60,
        "PROPENSITY SCORE MATCHING ANALYSIS RESULTS",
        "=" * 60,
        "",
        f"Treatment Effect ({estimand}): {effect_result.effect:.4f}",
        f"Standard Error: {effect_result.se:.4f}",
        f"95% CI: [{effect_result.ci_lower:.4f}, {effect_result.ci_upper:.4f}]",
        f"P-value: {effect_result.p_value:.4f}",
        "",
        "-" * 60,
        "PROPENSITY SCORE MODEL",
        "-" * 60,
        f"Method: {ps_result.method}",
        f"AUC: {ps_result.auc:.4f}" if ps_result.auc else "AUC: N/A",
        "",
        "-" * 60,
        "MATCHING",
        "-" * 60,
        f"Method: {match_result.method}",
        f"Treated units: {match_result.n_treated}",
        f"Matched controls: {match_result.n_control_matched}",
        f"Unmatched treated: {match_result.n_unmatched}",
        "",
        "-" * 60,
        "BALANCE",
        "-" * 60,
        f"Balance achieved: {'YES' if balance_after.balanced else 'NO'}",
        f"Max |SMD| after matching: {max(abs(s) for s in balance_after.smd_after.values()):.4f}",
        "",
    ]

    if sensitivity_result:
        summary_lines.extend([
            "-" * 60,
            "SENSITIVITY ANALYSIS (Rosenbaum Bounds)",
            "-" * 60,
            f"Critical Gamma: {sensitivity_result.critical_gamma:.2f}",
            ""
        ])

    summary_lines.append("=" * 60)
    summary_table = "\n".join(summary_lines)

    # Generate interpretation
    sig_level = ""
    if effect_result.p_value < 0.01:
        sig_level = "highly significant (p < 0.01)"
    elif effect_result.p_value < 0.05:
        sig_level = "significant (p < 0.05)"
    elif effect_result.p_value < 0.1:
        sig_level = "marginally significant (p < 0.1)"
    else:
        sig_level = "not statistically significant"

    interpretation = (
        f"The estimated {estimand} is {effect_result.effect:.4f} "
        f"(SE = {effect_result.se:.4f}), which is {sig_level}. "
        f"After matching, {'all' if balance_after.balanced else 'not all'} covariates "
        f"are balanced (|SMD| < {balance_after.threshold})."
    )

    if sensitivity_result:
        interpretation += (
            f" Sensitivity analysis indicates results are robust to hidden bias "
            f"up to Gamma = {sensitivity_result.critical_gamma:.2f}."
        )

    if not balance_after.balanced:
        interpretation += (
            "\n\nWARNING: Covariate balance not achieved. Consider re-specifying "
            "the propensity score model or using a different matching method."
        )

    if not overlap_result.passed:
        interpretation += (
            "\n\nWARNING: Limited common support. Results may rely on extrapolation."
        )

    return CausalOutput(
        effect=effect_result.effect,
        se=effect_result.se,
        ci_lower=effect_result.ci_lower,
        ci_upper=effect_result.ci_upper,
        p_value=effect_result.p_value,
        diagnostics=all_diagnostics,
        summary_table=summary_table,
        interpretation=interpretation
    )


# =============================================================================
# PSM + DID Combination
# =============================================================================

def psm_did(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    time_id: str,
    unit_id: str,
    treatment_time: int,
    covariates: List[str],
    match_method: str = "nearest_neighbor",
    **match_kwargs
) -> CausalOutput:
    """
    Combine PSM with Difference-in-Differences.

    First matches on pre-treatment characteristics, then estimates
    DID on the matched sample.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable
    treatment : str
        Treatment group indicator (ever-treated)
    time_id : str
        Time period column
    unit_id : str
        Unit identifier column
    treatment_time : int
        When treatment begins
    covariates : List[str]
        Pre-treatment covariates for matching
    match_method : str
        Matching method
    **match_kwargs
        Additional matching parameters

    Returns
    -------
    CausalOutput
        PSM-DID estimate
    """
    df = data.copy()

    # Step 1: Get pre-treatment data for matching
    pre_data = df[df[time_id] < treatment_time].copy()

    # Collapse to unit level (use first/mean of covariates)
    unit_data = pre_data.groupby(unit_id).agg({
        **{cov: 'mean' for cov in covariates},
        treatment: 'first'
    }).reset_index()

    # Step 2: Estimate propensity scores on pre-treatment data
    ps_result = estimate_propensity_score(
        unit_data, treatment, covariates, method='logit'
    )
    unit_data['_ps'] = ps_result.propensity_scores

    # Step 3: Match units
    if match_method == "nearest_neighbor":
        match_result = match_nearest_neighbor(
            unit_data, '_ps', treatment,
            **match_kwargs
        )
    else:
        match_result = match_nearest_neighbor(
            unit_data, '_ps', treatment,
            n_neighbors=1, replacement=True
        )

    # Step 4: Get matched unit IDs
    matched_units = match_result.matched_data[unit_id].unique()

    # Step 5: Filter panel data to matched units
    df_matched = df[df[unit_id].isin(matched_units)].copy()

    # Step 6: Create DID variables
    df_matched['_post'] = (df_matched[time_id] >= treatment_time).astype(int)
    df_matched['_did'] = df_matched[treatment] * df_matched['_post']

    # Step 7: Run DID on matched sample
    try:
        from did_estimator import estimate_did_panel

        did_result = estimate_did_panel(
            df_matched,
            outcome=outcome,
            treatment='_did',
            unit_id=unit_id,
            time_id=time_id,
            controls=covariates,
            cluster=unit_id
        )

    except ImportError:
        # Fallback to simple DID
        import statsmodels.api as sm

        y = df_matched[outcome]
        X = df_matched[[treatment, '_post', '_did'] + covariates]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit(cov_type='HC1')

        did_coef = model.params['_did']
        did_se = model.bse['_did']
        did_pval = model.pvalues['_did']

        did_result = CausalOutput(
            effect=did_coef,
            se=did_se,
            ci_lower=did_coef - 1.96 * did_se,
            ci_upper=did_coef + 1.96 * did_se,
            p_value=did_pval,
            diagnostics={'method': 'PSM-DID'},
            summary_table="",
            interpretation=""
        )

    # Update diagnostics
    did_result.diagnostics['psm_did'] = {
        'n_matched_units': len(matched_units),
        'n_unmatched': match_result.n_unmatched,
        'ps_method': ps_result.method
    }

    did_result.interpretation = (
        f"PSM-DID estimate: {did_result.effect:.4f} (SE = {did_result.se:.4f}). "
        f"Analysis based on {len(matched_units)} matched units."
    )

    return did_result


# =============================================================================
# Synthetic Data Generation for Validation
# =============================================================================

def generate_synthetic_psm_data(
    n: int = 2000,
    treatment_effect: float = 2.0,
    confounding_strength: float = 0.5,
    noise_std: float = 1.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic data for PSM validation.

    Parameters
    ----------
    n : int
        Sample size
    treatment_effect : float
        True treatment effect
    confounding_strength : float
        Strength of confounding (correlation between X and D/Y)
    noise_std : float
        Noise standard deviation
    random_state : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (data, true_params)
    """
    np.random.seed(random_state)

    # Covariates
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.binomial(1, 0.5, n)
    x3 = np.random.normal(0, 1, n)

    # True propensity score
    ps_true = 1 / (1 + np.exp(-(confounding_strength * x1 +
                                 0.8 * x2 +
                                 0.3 * x3)))

    # Treatment assignment
    treatment = np.random.binomial(1, ps_true)

    # Outcome with confounding
    y0 = 1 + 0.5 * x1 + 0.3 * x2 + 0.2 * x3 + np.random.normal(0, noise_std, n)
    y1 = y0 + treatment_effect

    y = np.where(treatment == 1, y1, y0)

    data = pd.DataFrame({
        'y': y,
        'treatment': treatment,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'ps_true': ps_true
    })

    true_params = {
        'true_att': treatment_effect,
        'true_ate': treatment_effect,
        'confounding_strength': confounding_strength,
        'n': n
    }

    return data, true_params


# =============================================================================
# Estimator Validation
# =============================================================================

def validate_estimator(verbose: bool = True) -> Dict[str, Any]:
    """
    Validate PSM estimator on synthetic data with known treatment effect.

    Returns
    -------
    Dict[str, Any]
        Validation results including bias assessment
    """
    # Generate synthetic data
    true_effect = 2.0
    data, true_params = generate_synthetic_psm_data(
        n=2000,
        treatment_effect=true_effect,
        confounding_strength=0.5,
        random_state=42
    )

    # Run PSM analysis
    result = run_full_psm_analysis(
        data=data,
        outcome='y',
        treatment='treatment',
        covariates=['x1', 'x2', 'x3'],
        match_method='nearest_neighbor',
        caliper=0.1,
        run_sensitivity=True
    )

    # Calculate bias
    bias = result.effect - true_effect
    bias_pct = abs(bias / true_effect) * 100

    # Check if within acceptable range
    passed = bias_pct < 10.0

    validation_result = {
        'true_att': true_effect,
        'estimated_att': result.effect,
        'se': result.se,
        'bias': bias,
        'bias_pct': bias_pct,
        'passed': passed,
        'ci_covers_truth': result.ci_lower <= true_effect <= result.ci_upper,
        'balance_achieved': result.diagnostics['balance'].balanced
    }

    if verbose:
        print("=" * 50)
        print("PSM ESTIMATOR VALIDATION")
        print("=" * 50)
        print(f"True ATT: {true_effect:.4f}")
        print(f"Estimated ATT: {result.effect:.4f}")
        print(f"Standard Error: {result.se:.4f}")
        print(f"Bias: {bias:.4f} ({bias_pct:.2f}%)")
        print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"CI covers truth: {validation_result['ci_covers_truth']}")
        print(f"Balance achieved: {validation_result['balance_achieved']}")
        print("-" * 50)
        print(f"VALIDATION: {'PASSED' if passed else 'FAILED'} (bias < 10%)")
        print("=" * 50)

    return validation_result


if __name__ == "__main__":
    # Run validation when module is executed directly
    validate_estimator(verbose=True)
