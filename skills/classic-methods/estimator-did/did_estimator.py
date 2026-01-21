"""
Difference-in-Differences (DID) Estimator Implementation.

This module provides comprehensive DID estimation including:
- Classic 2x2 DID
- Panel DID with two-way fixed effects
- Staggered DID (Callaway-Sant'Anna style)
- Parallel trends testing
- Event study visualization
- Placebo tests
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from scipy import stats

# Import from shared lib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'lib' / 'python'))
from data_loader import CausalInput, CausalOutput
from diagnostics import parallel_trends_test, DiagnosticResult, balance_test
from table_formatter import create_regression_table, create_diagnostic_report


# =============================================================================
# Data Validation
# =============================================================================

@dataclass
class DIDValidationResult:
    """Result of DID data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"DID Data Validation: {status}"]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def validate_did_data(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    unit_id: str,
    time_id: str,
    treatment_group: str = None
) -> DIDValidationResult:
    """
    Validate data structure for DID estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable name
    treatment : str
        Treatment indicator (post-treatment for treated units)
    unit_id : str
        Unit identifier column
    time_id : str
        Time period column
    treatment_group : str, optional
        Ever-treated indicator column

    Returns
    -------
    DIDValidationResult
        Validation results with errors and warnings
    """
    errors = []
    warnings_list = []
    summary = {}

    # Check required columns exist
    required_cols = [outcome, treatment, unit_id, time_id]
    if treatment_group:
        required_cols.append(treatment_group)

    for col in required_cols:
        if col not in data.columns:
            errors.append(f"Required column '{col}' not found in data")

    if errors:
        return DIDValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings_list,
            summary=summary
        )

    # Check for missing values
    for col in required_cols:
        n_missing = data[col].isna().sum()
        if n_missing > 0:
            warnings_list.append(f"Column '{col}' has {n_missing} missing values")

    # Panel structure
    n_units = data[unit_id].nunique()
    n_periods = data[time_id].nunique()
    n_obs = len(data)

    summary['n_units'] = n_units
    summary['n_periods'] = n_periods
    summary['n_obs'] = n_obs

    # Check if balanced panel
    obs_per_unit = data.groupby(unit_id).size()
    if obs_per_unit.nunique() > 1:
        warnings_list.append(
            f"Unbalanced panel: units have {obs_per_unit.min()}-{obs_per_unit.max()} observations"
        )
        summary['balanced'] = False
    else:
        summary['balanced'] = True

    # Treatment structure
    n_treated_obs = (data[treatment] == 1).sum()
    n_control_obs = (data[treatment] == 0).sum()
    summary['n_treated_obs'] = n_treated_obs
    summary['n_control_obs'] = n_control_obs

    if n_treated_obs == 0:
        errors.append("No treated observations found")
    if n_control_obs == 0:
        errors.append("No control observations found")

    # Treatment group structure (if provided)
    if treatment_group and treatment_group in data.columns:
        n_treated_units = (data.groupby(unit_id)[treatment_group].first() == 1).sum()
        n_control_units = n_units - n_treated_units
        summary['n_treated_units'] = n_treated_units
        summary['n_control_units'] = n_control_units

        if n_treated_units == 0:
            errors.append("No treated units found")
        if n_control_units == 0:
            errors.append("No control units found")

    # Check treatment timing
    treated_data = data[data[treatment] == 1]
    if len(treated_data) > 0:
        treatment_periods = treated_data[time_id].unique()
        pre_periods = data[~data[time_id].isin(treatment_periods)][time_id].nunique()
        post_periods = len(treatment_periods)
        summary['n_pre_periods'] = pre_periods
        summary['n_post_periods'] = post_periods

        if pre_periods < 2:
            warnings_list.append(
                f"Only {pre_periods} pre-treatment period(s). Parallel trends test may be limited."
            )

    is_valid = len(errors) == 0

    return DIDValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings_list,
        summary=summary
    )


# =============================================================================
# Parallel Trends Testing
# =============================================================================

def test_parallel_trends(
    data: pd.DataFrame,
    outcome: str,
    treatment_group: str,
    time_id: str,
    unit_id: str,
    treatment_time: int,
    n_pre_periods: int = None
) -> DiagnosticResult:
    """
    Test parallel trends assumption for DID.

    Performs both:
    1. Visual trend comparison (returns data for plotting)
    2. Statistical test for differential pre-trends

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable name
    treatment_group : str
        Ever-treated indicator (1 for treatment group, 0 for control)
    time_id : str
        Time period variable name
    unit_id : str
        Unit identifier variable name
    treatment_time : int
        Time period when treatment starts
    n_pre_periods : int, optional
        Number of pre-treatment periods to use (default: all available)

    Returns
    -------
    DiagnosticResult
        Test result with statistics, p-value, and interpretation
    """
    # Get pre-treatment data only
    pre_data = data[data[time_id] < treatment_time].copy()

    if len(pre_data) == 0:
        return DiagnosticResult(
            test_name="Parallel Trends Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=False,
            threshold=0.05,
            interpretation="No pre-treatment data available for parallel trends test",
            details={"error": "no_pre_data"}
        )

    # Limit to n_pre_periods if specified
    if n_pre_periods is not None:
        periods = sorted(pre_data[time_id].unique(), reverse=True)[:n_pre_periods]
        pre_data = pre_data[pre_data[time_id].isin(periods)]

    # Calculate group means by time
    trends = pre_data.groupby([treatment_group, time_id])[outcome].mean().unstack(level=0)

    if trends.shape[1] < 2:
        return DiagnosticResult(
            test_name="Parallel Trends Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=False,
            threshold=0.05,
            interpretation="Insufficient groups: need both treatment and control observations",
            details={"error": "insufficient_groups"}
        )

    # Get column names for treatment and control
    cols = trends.columns.tolist()
    control_col = 0 if 0 in cols else cols[0]
    treat_col = 1 if 1 in cols else cols[1]

    # Calculate difference in trends over time
    trend_diff = trends[treat_col] - trends[control_col]
    time_points = np.arange(len(trend_diff))

    if len(trend_diff) < 2:
        return DiagnosticResult(
            test_name="Parallel Trends Test",
            statistic=np.nan,
            p_value=np.nan,
            passed=False,
            threshold=0.05,
            interpretation="Need at least 2 pre-treatment periods to test parallel trends",
            details={"error": "insufficient_periods", "n_periods": len(trend_diff)}
        )

    # Test for significant slope in the difference
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        time_points, trend_diff.values
    )

    # Also run F-test for joint significance of pre-trend diffs
    # Use t-test for simplicity with 2+ periods
    t_stat = slope / std_err if std_err > 0 else np.inf
    df = len(trend_diff) - 2

    passed = p_value > 0.05  # Non-significant slope suggests parallel trends

    if passed:
        interpretation = (
            "Parallel trends assumption SUPPORTED: "
            "no significant divergence in pre-treatment trends "
            f"(slope = {slope:.4f}, p = {p_value:.4f})"
        )
    else:
        interpretation = (
            "Parallel trends assumption VIOLATED: "
            "significant divergence detected in pre-treatment period "
            f"(slope = {slope:.4f}, p = {p_value:.4f}). "
            "Consider synthetic control methods or group-specific trends."
        )

    return DiagnosticResult(
        test_name="Parallel Trends Test (Linear Divergence)",
        statistic=slope,
        p_value=p_value,
        passed=passed,
        threshold=0.05,
        interpretation=interpretation,
        details={
            "slope": slope,
            "intercept": intercept,
            "std_err": std_err,
            "t_statistic": t_stat,
            "r_squared": r_value**2,
            "n_pre_periods": len(trend_diff),
            "trend_differences": trend_diff.to_dict(),
            "treatment_means": trends[treat_col].to_dict(),
            "control_means": trends[control_col].to_dict()
        }
    )


def plot_parallel_trends(
    data: pd.DataFrame,
    outcome: str,
    treatment_group: str,
    time_id: str,
    treatment_time: int = None,
    unit_id: str = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Parallel Trends Check"
) -> Any:
    """
    Create parallel trends visualization.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable name
    treatment_group : str
        Ever-treated indicator
    time_id : str
        Time period variable
    treatment_time : int, optional
        When treatment begins (for vertical line)
    unit_id : str, optional
        Unit identifier (for confidence intervals)
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        Parallel trends plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")

    # Calculate group means and standard errors by time
    grouped = data.groupby([time_id, treatment_group])[outcome].agg(['mean', 'std', 'count'])
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped = grouped.reset_index()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each group
    for group_val, group_label, color in [(0, 'Control', 'blue'), (1, 'Treatment', 'red')]:
        group_data = grouped[grouped[treatment_group] == group_val].sort_values(time_id)

        if len(group_data) == 0:
            continue

        ax.plot(
            group_data[time_id],
            group_data['mean'],
            marker='o',
            label=group_label,
            color=color,
            linewidth=2
        )

        # Add confidence interval
        ax.fill_between(
            group_data[time_id],
            group_data['mean'] - 1.96 * group_data['se'],
            group_data['mean'] + 1.96 * group_data['se'],
            alpha=0.2,
            color=color
        )

    # Add vertical line at treatment time
    if treatment_time is not None:
        ax.axvline(x=treatment_time - 0.5, color='black', linestyle='--', linewidth=1.5,
                   label='Treatment Start')

    ax.set_xlabel('Time Period')
    ax.set_ylabel(f'Mean {outcome}')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# 2x2 DID Estimation
# =============================================================================

def estimate_did_2x2(
    data: pd.DataFrame,
    outcome: str,
    treatment_group: str,
    post: str,
    controls: List[str] = None,
    robust_se: bool = True
) -> CausalOutput:
    """
    Estimate classic 2x2 Difference-in-Differences.

    Model: Y = a + b1*Treated + b2*Post + delta*(Treated*Post) + X'gamma + e

    Parameters
    ----------
    data : pd.DataFrame
        Data with treatment group, post period, and outcome
    outcome : str
        Outcome variable name
    treatment_group : str
        Treatment group indicator (1 = treatment, 0 = control)
    post : str
        Post-treatment period indicator (1 = post, 0 = pre)
    controls : List[str], optional
        Control variables to include
    robust_se : bool
        Whether to use heteroskedasticity-robust standard errors

    Returns
    -------
    CausalOutput
        DID estimate with standard errors and diagnostics
    """
    try:
        import statsmodels.api as sm
        from statsmodels.regression.linear_model import OLS
    except ImportError:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")

    df = data.copy()

    # Create interaction term
    df['did_interaction'] = df[treatment_group] * df[post]

    # Build design matrix
    X_vars = [treatment_group, post, 'did_interaction']
    if controls:
        X_vars.extend(controls)

    X = df[X_vars].copy()
    X = sm.add_constant(X)
    y = df[outcome]

    # Handle missing values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Fit model
    model = OLS(y, X)
    if robust_se:
        results = model.fit(cov_type='HC1')
    else:
        results = model.fit()

    # Extract DID coefficient
    did_coef = results.params['did_interaction']
    did_se = results.bse['did_interaction']
    did_pval = results.pvalues['did_interaction']
    did_ci = results.conf_int().loc['did_interaction'].values

    # Calculate summary statistics
    n_treated_post = ((df[treatment_group] == 1) & (df[post] == 1)).sum()
    n_treated_pre = ((df[treatment_group] == 1) & (df[post] == 0)).sum()
    n_control_post = ((df[treatment_group] == 0) & (df[post] == 1)).sum()
    n_control_pre = ((df[treatment_group] == 0) & (df[post] == 0)).sum()

    # Manual 2x2 calculation for verification
    y_treat_post = df.loc[(df[treatment_group] == 1) & (df[post] == 1), outcome].mean()
    y_treat_pre = df.loc[(df[treatment_group] == 1) & (df[post] == 0), outcome].mean()
    y_ctrl_post = df.loc[(df[treatment_group] == 0) & (df[post] == 1), outcome].mean()
    y_ctrl_pre = df.loc[(df[treatment_group] == 0) & (df[post] == 0), outcome].mean()
    manual_did = (y_treat_post - y_treat_pre) - (y_ctrl_post - y_ctrl_pre)

    diagnostics = {
        'method': '2x2 DID',
        'n_treated_post': n_treated_post,
        'n_treated_pre': n_treated_pre,
        'n_control_post': n_control_post,
        'n_control_pre': n_control_pre,
        'manual_did': manual_did,
        'r_squared': results.rsquared,
        'robust_se': robust_se,
        'n_controls': len(controls) if controls else 0,
        'group_means': {
            'treated_post': y_treat_post,
            'treated_pre': y_treat_pre,
            'control_post': y_ctrl_post,
            'control_pre': y_ctrl_pre
        }
    }

    # Create summary table
    table_results = [{
        'treatment_effect': did_coef,
        'treatment_se': did_se,
        'treatment_pval': did_pval,
        'controls': controls is not None and len(controls) > 0,
        'fixed_effects': None,
        'n_obs': len(y),
        'r_squared': results.rsquared
    }]

    summary_table = create_regression_table(
        results=table_results,
        column_names=["(1) 2x2 DID"],
        title="Difference-in-Differences Results"
    )

    return CausalOutput(
        effect=did_coef,
        se=did_se,
        ci_lower=did_ci[0],
        ci_upper=did_ci[1],
        p_value=did_pval,
        diagnostics=diagnostics,
        summary_table=summary_table,
        interpretation=f"The DID estimate is {did_coef:.4f} (SE = {did_se:.4f})"
    )


# =============================================================================
# Panel DID with Fixed Effects
# =============================================================================

def estimate_did_panel(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    unit_id: str,
    time_id: str,
    controls: List[str] = None,
    cluster: str = None,
    entity_effects: bool = True,
    time_effects: bool = True
) -> CausalOutput:
    """
    Estimate Panel DID with two-way fixed effects.

    Model: Y_it = alpha_i + gamma_t + delta*D_it + X_it'beta + e_it

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable name
    treatment : str
        Treatment indicator (1 if unit i is treated at time t)
    unit_id : str
        Unit identifier column
    time_id : str
        Time period column
    controls : List[str], optional
        Control variables
    cluster : str, optional
        Variable to cluster standard errors on (typically unit_id)
    entity_effects : bool
        Include unit fixed effects (default True)
    time_effects : bool
        Include time fixed effects (default True)

    Returns
    -------
    CausalOutput
        Treatment effect estimate with clustered standard errors
    """
    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        raise ImportError(
            "linearmodels required for panel DID. "
            "Install with: pip install linearmodels"
        )

    df = data.copy()

    # Set up panel structure
    df = df.set_index([unit_id, time_id])

    # Build formula
    y = df[outcome]
    X_vars = [treatment]
    if controls:
        X_vars.extend(controls)
    X = df[X_vars]

    # Handle missing values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Fit panel model
    model = PanelOLS(
        y, X,
        entity_effects=entity_effects,
        time_effects=time_effects
    )

    # Determine clustering
    if cluster:
        # When clustering at entity level (most common for DID)
        # linearmodels handles this internally with cluster_entity=True
        if cluster == unit_id:
            results = model.fit(cov_type='clustered', cluster_entity=True)
        else:
            # For clustering on a different variable, need to pass it explicitly
            # First check if it's in the data before indexing
            cluster_values = data.set_index([unit_id, time_id]).loc[mask.values].index.get_level_values(0)
            results = model.fit(cov_type='clustered', clusters=cluster_values)
    else:
        results = model.fit(cov_type='robust')

    # Extract treatment coefficient
    treat_coef = results.params[treatment]
    treat_se = results.std_errors[treatment]
    treat_pval = results.pvalues[treatment]

    # Confidence interval
    ci_lower = treat_coef - 1.96 * treat_se
    ci_upper = treat_coef + 1.96 * treat_se

    # Diagnostics
    n_units = data[unit_id].nunique()
    n_periods = data[time_id].nunique()
    n_treated = (data[treatment] == 1).sum()

    diagnostics = {
        'method': 'Panel DID (TWFE)',
        'entity_effects': entity_effects,
        'time_effects': time_effects,
        'n_units': n_units,
        'n_periods': n_periods,
        'n_treated_obs': n_treated,
        'n_obs': len(y),
        'r_squared_within': results.rsquared_within,
        'r_squared_between': results.rsquared_between,
        'r_squared_overall': results.rsquared_overall,
        'clustered_se': cluster is not None,
        'cluster_var': cluster
    }

    # Fixed effects description
    fe_desc = []
    if entity_effects:
        fe_desc.append("Unit")
    if time_effects:
        fe_desc.append("Time")
    fe_str = " + ".join(fe_desc) if fe_desc else "None"

    # Summary table
    table_results = [{
        'treatment_effect': treat_coef,
        'treatment_se': treat_se,
        'treatment_pval': treat_pval,
        'controls': controls is not None and len(controls) > 0,
        'fixed_effects': fe_str,
        'n_obs': len(y),
        'r_squared': results.rsquared_within
    }]

    summary_table = create_regression_table(
        results=table_results,
        column_names=["(1) Panel DID"],
        title="Panel Difference-in-Differences Results"
    )

    return CausalOutput(
        effect=treat_coef,
        se=treat_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=treat_pval,
        diagnostics=diagnostics,
        summary_table=summary_table,
        interpretation=(
            f"The panel DID estimate is {treat_coef:.4f} (SE = {treat_se:.4f}, p = {treat_pval:.4f}). "
            f"Fixed effects: {fe_str}. "
            f"{'Clustered' if cluster else 'Robust'} standard errors."
        )
    )


# =============================================================================
# Staggered DID (Callaway-Sant'Anna Style)
# =============================================================================

def estimate_did_staggered(
    data: pd.DataFrame,
    outcome: str,
    treatment_time: str,
    unit_id: str,
    time_id: str,
    control_group: str = "nevertreated",
    anticipation: int = 0,
    covariates: List[str] = None
) -> CausalOutput:
    """
    Estimate staggered DID using Callaway-Sant'Anna (2021) approach.

    This implementation computes group-time average treatment effects (ATT(g,t))
    and aggregates them to overall ATT.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable name
    treatment_time : str
        Column indicating when unit was first treated (0 or NaN for never-treated)
    unit_id : str
        Unit identifier column
    time_id : str
        Time period column
    control_group : str
        Type of control group: "nevertreated" or "notyettreated"
    anticipation : int
        Number of periods before treatment to allow for anticipation
    covariates : List[str], optional
        Covariates to condition on

    Returns
    -------
    CausalOutput
        Aggregated ATT with group-time effects in diagnostics
    """
    df = data.copy()

    # Identify treatment cohorts (groups)
    df['_first_treat'] = df[treatment_time].fillna(0).astype(int)
    cohorts = df[df['_first_treat'] > 0]['_first_treat'].unique()
    cohorts = sorted(cohorts)

    periods = sorted(df[time_id].unique())
    never_treated = df[df['_first_treat'] == 0][unit_id].unique()

    if len(cohorts) == 0:
        raise ValueError("No treated cohorts found. Check treatment_time column.")

    if len(never_treated) == 0 and control_group == "nevertreated":
        warnings.warn(
            "No never-treated units found. Switching to 'notyettreated' control group."
        )
        control_group = "notyettreated"

    # Compute group-time ATTs
    att_gt = {}
    att_gt_se = {}
    att_gt_n = {}

    for g in cohorts:
        for t in periods:
            # Only compute for post-treatment periods (accounting for anticipation)
            if t < g - anticipation:
                continue

            # Get treated units in cohort g
            treated_units = df[(df['_first_treat'] == g)][unit_id].unique()

            # Get control units
            if control_group == "nevertreated":
                control_units = never_treated
            else:  # notyettreated
                control_units = df[
                    (df['_first_treat'] == 0) |
                    (df['_first_treat'] > t)
                ][unit_id].unique()
                control_units = [u for u in control_units if u not in treated_units]

            if len(treated_units) == 0 or len(control_units) == 0:
                continue

            # Get outcome data
            # Baseline period: g - 1 (or g - 1 - anticipation)
            base_period = g - 1 - anticipation

            if base_period not in periods:
                continue

            # Compute 2x2 DID for this (g, t) cell
            y_treat_t = df[
                (df[unit_id].isin(treated_units)) &
                (df[time_id] == t)
            ][outcome].mean()

            y_treat_base = df[
                (df[unit_id].isin(treated_units)) &
                (df[time_id] == base_period)
            ][outcome].mean()

            y_ctrl_t = df[
                (df[unit_id].isin(control_units)) &
                (df[time_id] == t)
            ][outcome].mean()

            y_ctrl_base = df[
                (df[unit_id].isin(control_units)) &
                (df[time_id] == base_period)
            ][outcome].mean()

            # Check for valid data
            if any(pd.isna([y_treat_t, y_treat_base, y_ctrl_t, y_ctrl_base])):
                continue

            att = (y_treat_t - y_treat_base) - (y_ctrl_t - y_ctrl_base)
            att_gt[(g, t)] = att

            # Simple SE estimation (assuming independence)
            n_treat = len(treated_units)
            n_ctrl = len(control_units)

            var_treat_t = df[
                (df[unit_id].isin(treated_units)) &
                (df[time_id] == t)
            ][outcome].var() / n_treat

            var_treat_base = df[
                (df[unit_id].isin(treated_units)) &
                (df[time_id] == base_period)
            ][outcome].var() / n_treat

            var_ctrl_t = df[
                (df[unit_id].isin(control_units)) &
                (df[time_id] == t)
            ][outcome].var() / n_ctrl

            var_ctrl_base = df[
                (df[unit_id].isin(control_units)) &
                (df[time_id] == base_period)
            ][outcome].var() / n_ctrl

            se = np.sqrt(var_treat_t + var_treat_base + var_ctrl_t + var_ctrl_base)
            att_gt_se[(g, t)] = se
            att_gt_n[(g, t)] = n_treat

    if len(att_gt) == 0:
        raise ValueError(
            "Could not compute any group-time ATTs. "
            "Check data structure and treatment timing."
        )

    # Aggregate to overall ATT (weighted by group size)
    total_n = sum(att_gt_n.values())
    overall_att = sum(att_gt[k] * att_gt_n[k] for k in att_gt.keys()) / total_n

    # SE via delta method (simplified: assume independence)
    weights = {k: att_gt_n[k] / total_n for k in att_gt.keys()}
    overall_var = sum(
        (weights[k] ** 2) * (att_gt_se[k] ** 2)
        for k in att_gt.keys()
    )
    overall_se = np.sqrt(overall_var)

    # Inference
    z_stat = overall_att / overall_se if overall_se > 0 else np.inf
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    ci_lower = overall_att - 1.96 * overall_se
    ci_upper = overall_att + 1.96 * overall_se

    # Event-time aggregation
    event_time_effects = {}
    for (g, t), att in att_gt.items():
        e = t - g  # Event time (relative to treatment)
        if e not in event_time_effects:
            event_time_effects[e] = []
        event_time_effects[e].append((att, att_gt_n[(g, t)]))

    # Aggregate by event time
    event_time_att = {}
    for e, effects in event_time_effects.items():
        total_n_e = sum(n for _, n in effects)
        event_time_att[e] = sum(att * n for att, n in effects) / total_n_e

    diagnostics = {
        'method': 'Staggered DID (Callaway-Sant\'Anna)',
        'control_group': control_group,
        'anticipation': anticipation,
        'n_cohorts': len(cohorts),
        'cohorts': list(cohorts),
        'n_never_treated': len(never_treated),
        'att_gt': att_gt,
        'att_gt_se': att_gt_se,
        'att_gt_n': att_gt_n,
        'event_time_effects': event_time_att,
        'total_treated_obs': total_n
    }

    # Summary table
    table_results = [{
        'treatment_effect': overall_att,
        'treatment_se': overall_se,
        'treatment_pval': p_value,
        'controls': covariates is not None,
        'fixed_effects': 'Cohort + Time',
        'n_obs': len(df),
        'r_squared': np.nan  # Not directly applicable
    }]

    summary_table = create_regression_table(
        results=table_results,
        column_names=["(1) Staggered DID"],
        title="Staggered Difference-in-Differences Results"
    )

    return CausalOutput(
        effect=overall_att,
        se=overall_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        diagnostics=diagnostics,
        summary_table=summary_table,
        interpretation=(
            f"The aggregated ATT is {overall_att:.4f} (SE = {overall_se:.4f}). "
            f"Based on {len(cohorts)} treatment cohorts with {control_group} control group."
        )
    )


# =============================================================================
# Event Study Plot
# =============================================================================

def event_study_plot(
    data: pd.DataFrame,
    outcome: str,
    treatment_time_var: str,
    unit_id: str,
    time_id: str,
    reference_period: int = -1,
    pre_periods: int = 4,
    post_periods: int = 4,
    cluster: str = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Event Study: Dynamic Treatment Effects"
) -> Any:
    """
    Create event study plot showing dynamic treatment effects.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable name
    treatment_time_var : str
        Column with unit's treatment time (first period treated)
    unit_id : str
        Unit identifier
    time_id : str
        Time period
    reference_period : int
        Event-time period to normalize to (default: -1, period before treatment)
    pre_periods : int
        Number of pre-treatment periods to show
    post_periods : int
        Number of post-treatment periods to show
    cluster : str, optional
        Variable to cluster standard errors on
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        Event study plot with coefficients and confidence intervals
    """
    try:
        import matplotlib.pyplot as plt
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("matplotlib and statsmodels required for event study plot")

    df = data.copy()

    # Create event time variable
    df['_treat_time'] = df[treatment_time_var].fillna(np.inf)
    df['_event_time'] = df[time_id] - df['_treat_time']

    # Only keep observations with valid event time
    df = df[df['_treat_time'] != np.inf].copy()

    # Create event-time dummies
    event_times = list(range(-pre_periods, post_periods + 1))
    event_times = [e for e in event_times if e != reference_period]

    for e in event_times:
        df[f'_et_{e}'] = (df['_event_time'] == e).astype(int)

    # Also need controls for periods outside our window
    df['_et_pre'] = (df['_event_time'] < -pre_periods).astype(int)
    df['_et_post'] = (df['_event_time'] > post_periods).astype(int)

    # Run regression with unit and time fixed effects
    try:
        from linearmodels.panel import PanelOLS

        df_panel = df.set_index([unit_id, time_id])

        y = df_panel[outcome]
        X_vars = [f'_et_{e}' for e in event_times] + ['_et_pre', '_et_post']
        X = df_panel[X_vars]

        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        model = PanelOLS(y, X, entity_effects=True, time_effects=True)

        if cluster:
            results = model.fit(cov_type='clustered', cluster_entity=True)
        else:
            results = model.fit(cov_type='robust')

    except ImportError:
        # Fallback to OLS with dummies
        from statsmodels.regression.linear_model import OLS

        y = df[outcome]
        X_vars = [f'_et_{e}' for e in event_times]
        X = df[X_vars]
        X = sm.add_constant(X)

        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        model = OLS(y, X)
        results = model.fit(cov_type='HC1')

    # Extract coefficients for event times
    coefs = []
    ses = []
    event_time_labels = []

    for e in sorted(event_times):
        var_name = f'_et_{e}'
        if var_name in results.params.index:
            coefs.append(results.params[var_name])
            ses.append(results.std_errors[var_name])
        else:
            coefs.append(np.nan)
            ses.append(np.nan)
        event_time_labels.append(e)

    # Add reference period (zero by normalization)
    ref_idx = sum(1 for e in event_time_labels if e < reference_period)
    event_time_labels.insert(ref_idx, reference_period)
    coefs.insert(ref_idx, 0)
    ses.insert(ref_idx, 0)

    coefs = np.array(coefs)
    ses = np.array(ses)
    ci_lower = coefs - 1.96 * ses
    ci_upper = coefs + 1.96 * ses

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=1.5, label='Treatment')

    ax.errorbar(
        event_time_labels, coefs,
        yerr=[coefs - ci_lower, ci_upper - coefs],
        fmt='o', color='blue', capsize=3, capthick=1, linewidth=1.5,
        markersize=6, label='Point Estimate'
    )

    ax.fill_between(
        event_time_labels, ci_lower, ci_upper,
        alpha=0.2, color='blue'
    )

    ax.set_xlabel('Event Time (periods relative to treatment)')
    ax.set_ylabel(f'Effect on {outcome}')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add note about pre-trends
    pre_coefs = [c for e, c in zip(event_time_labels, coefs) if e < 0 and e >= -pre_periods]
    pre_ses = [s for e, s in zip(event_time_labels, ses) if e < 0 and e >= -pre_periods]

    if len(pre_coefs) > 0 and not any(np.isnan(pre_coefs)):
        # Joint test for pre-trends
        pre_coefs = np.array(pre_coefs)
        pre_ses = np.array(pre_ses)
        # Simple: check if any pre-treatment coef is significant
        pre_significant = any(abs(c/s) > 1.96 for c, s in zip(pre_coefs, pre_ses) if s > 0)
        if pre_significant:
            ax.text(
                0.02, 0.98,
                'Warning: Some pre-treatment coefficients are significant',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                color='red'
            )

    plt.tight_layout()
    return fig


# =============================================================================
# Placebo Test
# =============================================================================

def placebo_test(
    data: pd.DataFrame,
    outcome: str,
    treatment_group: str,
    unit_id: str,
    time_id: str,
    actual_treatment_time: int,
    placebo_treatment_time: int,
    controls: List[str] = None
) -> DiagnosticResult:
    """
    Run placebo test with fake treatment timing.

    The placebo effect should be statistically insignificant if the
    parallel trends assumption holds.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable name
    treatment_group : str
        Ever-treated indicator
    unit_id : str
        Unit identifier
    time_id : str
        Time period
    actual_treatment_time : int
        Actual treatment start time
    placebo_treatment_time : int
        Fake treatment time for placebo test (should be before actual)
    controls : List[str], optional
        Control variables

    Returns
    -------
    DiagnosticResult
        Placebo test result
    """
    if placebo_treatment_time >= actual_treatment_time:
        raise ValueError(
            f"Placebo time ({placebo_treatment_time}) must be before "
            f"actual treatment time ({actual_treatment_time})"
        )

    # Use only pre-treatment data
    pre_data = data[data[time_id] < actual_treatment_time].copy()

    # Create placebo indicators
    pre_data['_placebo_post'] = (pre_data[time_id] >= placebo_treatment_time).astype(int)
    pre_data['_placebo_treated'] = (
        pre_data[treatment_group] * pre_data['_placebo_post']
    ).astype(int)

    # Run 2x2 DID with placebo treatment
    result = estimate_did_2x2(
        data=pre_data,
        outcome=outcome,
        treatment_group=treatment_group,
        post='_placebo_post',
        controls=controls
    )

    # Determine if placebo test passes (effect should be insignificant)
    passed = result.p_value > 0.1

    if passed:
        interpretation = (
            f"Placebo test PASSED: No significant effect at fake treatment time "
            f"(placebo ATT = {result.effect:.4f}, p = {result.p_value:.4f}). "
            "This supports the parallel trends assumption."
        )
    else:
        interpretation = (
            f"Placebo test FAILED: Significant effect detected at fake treatment time "
            f"(placebo ATT = {result.effect:.4f}, p = {result.p_value:.4f}). "
            "This suggests the parallel trends assumption may be violated."
        )

    return DiagnosticResult(
        test_name="Placebo Test (Fake Treatment Timing)",
        statistic=result.effect,
        p_value=result.p_value,
        passed=passed,
        threshold=0.1,
        interpretation=interpretation,
        details={
            'placebo_treatment_time': placebo_treatment_time,
            'actual_treatment_time': actual_treatment_time,
            'placebo_att': result.effect,
            'placebo_se': result.se,
            'placebo_ci': [result.ci_lower, result.ci_upper],
            'n_obs': result.diagnostics.get('n_treated_post', 0) +
                     result.diagnostics.get('n_treated_pre', 0) +
                     result.diagnostics.get('n_control_post', 0) +
                     result.diagnostics.get('n_control_pre', 0)
        }
    )


# =============================================================================
# Full DID Analysis Workflow
# =============================================================================

def run_full_did_analysis(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    unit_id: str,
    time_id: str,
    treatment_time: int = None,
    treatment_group: str = None,
    controls: List[str] = None,
    cluster: str = None,
    run_placebo: bool = True,
    placebo_lag: int = 2
) -> CausalOutput:
    """
    Run complete DID analysis workflow.

    This function:
    1. Validates data structure
    2. Tests parallel trends
    3. Runs panel DID estimation
    4. Performs placebo test
    5. Generates comprehensive output

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable name
    treatment : str
        Treatment indicator (1 if treated in that period)
    unit_id : str
        Unit identifier column
    time_id : str
        Time period column
    treatment_time : int, optional
        When treatment began (inferred if not provided)
    treatment_group : str, optional
        Ever-treated indicator (created if not provided)
    controls : List[str], optional
        Control variables
    cluster : str, optional
        Clustering variable (defaults to unit_id)
    run_placebo : bool
        Whether to run placebo test
    placebo_lag : int
        How many periods before treatment for placebo test

    Returns
    -------
    CausalOutput
        Complete analysis results with all diagnostics
    """
    df = data.copy()

    # Infer treatment_group if not provided
    if treatment_group is None:
        treatment_group = '_treatment_group'
        # A unit is in treatment group if ever treated
        df[treatment_group] = df.groupby(unit_id)[treatment].transform('max')

    # Infer treatment_time if not provided
    if treatment_time is None:
        # Find first period where any unit is treated
        treated_periods = df[df[treatment] == 1][time_id]
        if len(treated_periods) == 0:
            raise ValueError("No treated observations found")
        treatment_time = treated_periods.min()

    # Default clustering
    if cluster is None:
        cluster = unit_id

    # Step 1: Validate data
    validation = validate_did_data(
        df, outcome, treatment, unit_id, time_id, treatment_group
    )

    if not validation.is_valid:
        raise ValueError(f"Data validation failed: {validation.errors}")

    # Step 2: Test parallel trends
    trends_result = test_parallel_trends(
        data=df,
        outcome=outcome,
        treatment_group=treatment_group,
        time_id=time_id,
        unit_id=unit_id,
        treatment_time=treatment_time
    )

    # Step 3: Run panel DID
    main_result = estimate_did_panel(
        data=df,
        outcome=outcome,
        treatment=treatment,
        unit_id=unit_id,
        time_id=time_id,
        controls=controls,
        cluster=cluster,
        entity_effects=True,
        time_effects=True
    )

    # Step 4: Placebo test (if requested and feasible)
    placebo_result = None
    if run_placebo:
        placebo_time = treatment_time - placebo_lag
        periods = sorted(df[time_id].unique())
        if placebo_time > periods[0]:
            try:
                # Create post indicator for placebo
                df['_post'] = (df[time_id] >= treatment_time).astype(int)

                placebo_result = placebo_test(
                    data=df,
                    outcome=outcome,
                    treatment_group=treatment_group,
                    unit_id=unit_id,
                    time_id=time_id,
                    actual_treatment_time=treatment_time,
                    placebo_treatment_time=placebo_time,
                    controls=controls
                )
            except Exception as e:
                warnings.warn(f"Placebo test failed: {e}")

    # Compile all diagnostics
    all_diagnostics = {
        'validation': validation.summary,
        'parallel_trends': trends_result,
        'estimation': main_result.diagnostics,
    }

    if placebo_result:
        all_diagnostics['placebo_test'] = placebo_result

    # Generate comprehensive summary
    summary_lines = [
        "=" * 60,
        "DIFFERENCE-IN-DIFFERENCES ANALYSIS RESULTS",
        "=" * 60,
        "",
        f"Treatment Effect (ATT): {main_result.effect:.4f}",
        f"Standard Error: {main_result.se:.4f}",
        f"95% CI: [{main_result.ci_lower:.4f}, {main_result.ci_upper:.4f}]",
        f"P-value: {main_result.p_value:.4f}",
        "",
        "-" * 60,
        "DIAGNOSTICS",
        "-" * 60,
        "",
        f"Parallel Trends Test: {'PASSED' if trends_result.passed else 'FAILED'}",
        f"  - Slope: {trends_result.statistic:.4f}",
        f"  - P-value: {trends_result.p_value:.4f}",
        "",
    ]

    if placebo_result:
        summary_lines.extend([
            f"Placebo Test: {'PASSED' if placebo_result.passed else 'FAILED'}",
            f"  - Placebo ATT: {placebo_result.statistic:.4f}",
            f"  - P-value: {placebo_result.p_value:.4f}",
            ""
        ])

    summary_lines.extend([
        "-" * 60,
        "SAMPLE",
        "-" * 60,
        f"N Units: {validation.summary.get('n_units', 'N/A')}",
        f"N Periods: {validation.summary.get('n_periods', 'N/A')}",
        f"N Observations: {validation.summary.get('n_obs', 'N/A')}",
        f"Balanced Panel: {validation.summary.get('balanced', 'N/A')}",
        "",
        "=" * 60
    ])

    comprehensive_summary = "\n".join(summary_lines)

    # Generate interpretation
    interpretation = main_result.generate_interpretation(
        treatment_name="treatment",
        outcome_name=outcome
    )

    # Add caveats based on diagnostics
    if not trends_result.passed:
        interpretation += (
            "\n\nCAUTION: The parallel trends assumption appears to be violated. "
            "Results should be interpreted with caution. Consider using synthetic "
            "control methods or adding group-specific trends."
        )

    if placebo_result and not placebo_result.passed:
        interpretation += (
            "\n\nWARNING: Placebo test failed, suggesting potential issues with "
            "the identification strategy."
        )

    return CausalOutput(
        effect=main_result.effect,
        se=main_result.se,
        ci_lower=main_result.ci_lower,
        ci_upper=main_result.ci_upper,
        p_value=main_result.p_value,
        diagnostics=all_diagnostics,
        summary_table=comprehensive_summary,
        interpretation=interpretation
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def balance_test_did(
    data: pd.DataFrame,
    treatment_group: str,
    covariates: List[str],
    time_id: str,
    pre_period: int
) -> Dict[str, DiagnosticResult]:
    """
    Test covariate balance between treatment and control groups in pre-period.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    treatment_group : str
        Ever-treated indicator
    covariates : List[str]
        Covariates to test
    time_id : str
        Time period column
    pre_period : int
        Pre-treatment period to test balance in

    Returns
    -------
    Dict[str, DiagnosticResult]
        Balance test results for each covariate
    """
    pre_data = data[data[time_id] == pre_period]
    return balance_test(pre_data, treatment_group, covariates)


# =============================================================================
# Validation with Synthetic Data
# =============================================================================

def validate_estimator(verbose: bool = True) -> Dict[str, Any]:
    """
    Validate DID estimator on synthetic data with known treatment effect.

    Returns
    -------
    Dict[str, Any]
        Validation results including bias assessment
    """
    from data_loader import generate_synthetic_did_data

    # Generate synthetic data with sufficient sample size for stable estimates
    true_ate = 2.0
    data, true_params = generate_synthetic_did_data(
        n_units=200,       # Larger sample for stable estimates
        n_periods=10,
        treatment_period=5,
        treatment_effect=true_ate,
        treatment_share=0.5,
        noise_std=0.8,     # Slightly lower noise for cleaner signal
        random_state=42
    )

    # Run estimation
    result = run_full_did_analysis(
        data=data,
        outcome='y',
        treatment='treated',
        unit_id='unit_id',
        time_id='time',
        treatment_time=5,
        treatment_group='treatment_group',
        controls=['x1', 'x2'],
        cluster='unit_id',
        run_placebo=True,
        placebo_lag=2
    )

    # Calculate bias
    bias = result.effect - true_ate
    bias_pct = abs(bias / true_ate) * 100

    # Check if within acceptable range (5% bias)
    passed = bias_pct < 5.0

    validation_result = {
        'true_ate': true_ate,
        'estimated_ate': result.effect,
        'se': result.se,
        'bias': bias,
        'bias_pct': bias_pct,
        'passed': passed,
        'ci_covers_truth': result.ci_lower <= true_ate <= result.ci_upper,
        'parallel_trends_passed': result.diagnostics['parallel_trends'].passed
    }

    if verbose:
        print("=" * 50)
        print("DID ESTIMATOR VALIDATION")
        print("=" * 50)
        print(f"True ATE: {true_ate:.4f}")
        print(f"Estimated ATE: {result.effect:.4f}")
        print(f"Standard Error: {result.se:.4f}")
        print(f"Bias: {bias:.4f} ({bias_pct:.2f}%)")
        print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"CI covers truth: {validation_result['ci_covers_truth']}")
        print(f"Parallel trends test: {'PASSED' if validation_result['parallel_trends_passed'] else 'FAILED'}")
        print("-" * 50)
        print(f"VALIDATION: {'PASSED' if passed else 'FAILED'} (bias < 5%)")
        print("=" * 50)

    return validation_result


if __name__ == "__main__":
    # Run validation when module is executed directly
    validate_estimator(verbose=True)
