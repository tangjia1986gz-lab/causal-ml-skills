"""
Publication-quality visualization functions for causal inference results.

This module provides standardized plotting functions for common causal
inference visualizations including event studies, regression discontinuity,
propensity score distributions, and heterogeneous treatment effects.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Lazy imports to avoid hard dependencies
_plt = None
_mpl = None


def _get_matplotlib():
    """Lazy import matplotlib."""
    global _plt, _mpl
    if _plt is None:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        _plt = plt
        _mpl = mpl
    return _plt, _mpl


# Standard style settings for publication-quality figures
PUBLICATION_STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
}

# Color palettes for different use cases
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'tertiary': '#2ca02c',     # Green
    'treatment': '#d62728',    # Red
    'control': '#1f77b4',      # Blue
    'ci': '#9467bd',           # Purple
    'neutral': '#7f7f7f',      # Gray
    'reference': '#000000',    # Black
}


def set_publication_style():
    """Apply publication-quality matplotlib style settings."""
    plt, mpl = _get_matplotlib()
    plt.rcParams.update(PUBLICATION_STYLE)


@dataclass
class EventStudyData:
    """Data structure for event study plots."""
    periods: List[int]
    coefficients: List[float]
    ci_lower: List[float]
    ci_upper: List[float]
    reference_period: int = -1


def plot_event_study(
    data: Union[EventStudyData, Dict[str, Any], pd.DataFrame],
    outcome_label: str = "Treatment Effect",
    x_label: str = "Periods Relative to Treatment",
    y_label: str = "Coefficient",
    title: Optional[str] = None,
    reference_period: int = -1,
    show_reference_line: bool = True,
    confidence_level: float = 0.95,
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    return_fig: bool = False
) -> Optional[Any]:
    """
    Plot event study coefficients with confidence intervals.

    Creates a publication-quality event study plot showing period-by-period
    treatment effects with confidence bands, commonly used for DID analysis.

    Parameters
    ----------
    data : Union[EventStudyData, Dict[str, Any], pd.DataFrame]
        Event study data. Can be:
        - EventStudyData object
        - Dict with keys: 'periods', 'coefficients', 'ci_lower', 'ci_upper'
        - DataFrame with columns: 'period', 'coefficient', 'ci_lower', 'ci_upper'
    outcome_label : str
        Label for the legend
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    title : Optional[str]
        Plot title (if None, no title is shown)
    reference_period : int
        The omitted reference period (typically -1)
    show_reference_line : bool
        Whether to show horizontal line at y=0
    confidence_level : float
        Confidence level (for annotation purposes)
    colors : Optional[Dict[str, str]]
        Custom color scheme with keys: 'point', 'ci', 'reference'
    figsize : Tuple[float, float]
        Figure size (width, height)
    save_path : Optional[str]
        Path to save the figure
    return_fig : bool
        If True, return the figure object instead of displaying

    Returns
    -------
    Optional[matplotlib.figure.Figure]
        Figure object if return_fig=True, else None
    """
    plt, mpl = _get_matplotlib()
    set_publication_style()

    # Parse data
    if isinstance(data, EventStudyData):
        periods = data.periods
        coefficients = data.coefficients
        ci_lower = data.ci_lower
        ci_upper = data.ci_upper
    elif isinstance(data, dict):
        periods = data.get('periods', [])
        coefficients = data.get('coefficients', [])
        ci_lower = data.get('ci_lower', [])
        ci_upper = data.get('ci_upper', [])
    elif isinstance(data, pd.DataFrame):
        periods = data['period'].tolist()
        coefficients = data['coefficient'].tolist()
        ci_lower = data['ci_lower'].tolist()
        ci_upper = data['ci_upper'].tolist()
    else:
        raise ValueError("data must be EventStudyData, dict, or DataFrame")

    # Set colors
    if colors is None:
        colors = {
            'point': COLORS['primary'],
            'ci': COLORS['primary'],
            'reference': COLORS['neutral']
        }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate error bars
    yerr_lower = [c - l for c, l in zip(coefficients, ci_lower)]
    yerr_upper = [u - c for c, u in zip(coefficients, ci_upper)]

    # Plot confidence intervals as shaded region
    ax.fill_between(
        periods, ci_lower, ci_upper,
        alpha=0.2, color=colors['ci'],
        label=f'{int(confidence_level*100)}% CI'
    )

    # Plot coefficients
    ax.plot(
        periods, coefficients,
        'o-', color=colors['point'],
        markersize=8, linewidth=2,
        label=outcome_label
    )

    # Reference line at y=0
    if show_reference_line:
        ax.axhline(y=0, color=colors['reference'], linestyle='--', linewidth=1, alpha=0.7)

    # Vertical line at treatment time (period 0)
    ax.axvline(x=0, color=colors['reference'], linestyle=':', linewidth=1, alpha=0.5)

    # Mark reference period
    if reference_period in periods:
        ref_idx = periods.index(reference_period)
        ax.scatter([reference_period], [coefficients[ref_idx]],
                   marker='s', s=100, facecolors='none',
                   edgecolors=colors['reference'], linewidth=2,
                   label=f'Reference (t={reference_period})', zorder=5)

    # Labels and formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    ax.legend(loc='best', framealpha=0.9)

    # Set x-axis ticks to integers
    ax.set_xticks(periods)

    # Annotations
    ax.annotate(
        'Pre-treatment',
        xy=(-0.5, ax.get_ylim()[1]),
        xytext=(-0.5, ax.get_ylim()[1]),
        fontsize=9, alpha=0.7,
        ha='right'
    )
    ax.annotate(
        'Post-treatment',
        xy=(0.5, ax.get_ylim()[1]),
        xytext=(0.5, ax.get_ylim()[1]),
        fontsize=9, alpha=0.7,
        ha='left'
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if return_fig:
        return fig

    plt.show()
    return None


def plot_rd(
    running_var: np.ndarray,
    outcome: np.ndarray,
    cutoff: float = 0,
    treatment: Optional[np.ndarray] = None,
    fitted_left: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    fitted_right: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    n_bins: int = 20,
    polynomial_order: int = 1,
    x_label: str = "Running Variable",
    y_label: str = "Outcome",
    title: Optional[str] = None,
    show_bins: bool = True,
    show_fit: bool = True,
    bandwidth: Optional[float] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    return_fig: bool = False
) -> Optional[Any]:
    """
    Plot regression discontinuity design with binned scatter and polynomial fit.

    Creates a publication-quality RD plot showing the relationship between
    the running variable and outcome, with optional polynomial fits on
    either side of the cutoff.

    Parameters
    ----------
    running_var : np.ndarray
        The running/forcing variable
    outcome : np.ndarray
        The outcome variable
    cutoff : float
        The RD cutoff value
    treatment : Optional[np.ndarray]
        Treatment indicator (computed from running_var if not provided)
    fitted_left : Optional[Tuple[np.ndarray, np.ndarray]]
        Pre-computed fit for left of cutoff (x_values, y_values)
    fitted_right : Optional[Tuple[np.ndarray, np.ndarray]]
        Pre-computed fit for right of cutoff
    n_bins : int
        Number of bins for scatter plot
    polynomial_order : int
        Order of polynomial fit (if fitted values not provided)
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    title : Optional[str]
        Plot title
    show_bins : bool
        Whether to show binned scatter points
    show_fit : bool
        Whether to show polynomial fit
    bandwidth : Optional[float]
        Bandwidth for restricting data (if None, use all data)
    colors : Optional[Dict[str, str]]
        Custom colors with keys: 'control', 'treatment', 'fit_control', 'fit_treatment'
    figsize : Tuple[float, float]
        Figure size
    save_path : Optional[str]
        Path to save figure
    return_fig : bool
        If True, return figure object

    Returns
    -------
    Optional[matplotlib.figure.Figure]
        Figure object if return_fig=True
    """
    plt, mpl = _get_matplotlib()
    set_publication_style()

    running_var = np.asarray(running_var)
    outcome = np.asarray(outcome)

    if treatment is None:
        treatment = (running_var >= cutoff).astype(int)

    # Apply bandwidth if specified
    if bandwidth is not None:
        mask = np.abs(running_var - cutoff) <= bandwidth
        running_var = running_var[mask]
        outcome = outcome[mask]
        treatment = treatment[mask]

    # Set colors
    if colors is None:
        colors = {
            'control': COLORS['control'],
            'treatment': COLORS['treatment'],
            'fit_control': COLORS['control'],
            'fit_treatment': COLORS['treatment'],
            'cutoff': COLORS['neutral']
        }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Separate data by treatment status
    left_mask = running_var < cutoff
    right_mask = running_var >= cutoff

    # Create bins
    if show_bins:
        # Bin data on each side
        for mask, color, label in [
            (left_mask, colors['control'], 'Control'),
            (right_mask, colors['treatment'], 'Treatment')
        ]:
            if np.sum(mask) > 0:
                x_data = running_var[mask]
                y_data = outcome[mask]

                # Create bins
                bin_edges = np.linspace(x_data.min(), x_data.max(), n_bins // 2 + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_means = []

                for i in range(len(bin_edges) - 1):
                    in_bin = (x_data >= bin_edges[i]) & (x_data < bin_edges[i + 1])
                    if np.sum(in_bin) > 0:
                        bin_means.append(y_data[in_bin].mean())
                    else:
                        bin_means.append(np.nan)

                # Filter out NaN bins
                valid = ~np.isnan(bin_means)
                ax.scatter(
                    bin_centers[valid], np.array(bin_means)[valid],
                    c=color, s=60, alpha=0.7, label=f'{label} (binned)',
                    edgecolors='white', linewidth=0.5
                )

    # Plot polynomial fits
    if show_fit:
        # Fit polynomial if not provided
        if fitted_left is None and np.sum(left_mask) > polynomial_order + 1:
            x_left = running_var[left_mask]
            y_left = outcome[left_mask]
            coeffs_left = np.polyfit(x_left, y_left, polynomial_order)
            x_fit_left = np.linspace(x_left.min(), cutoff, 100)
            y_fit_left = np.polyval(coeffs_left, x_fit_left)
            fitted_left = (x_fit_left, y_fit_left)

        if fitted_right is None and np.sum(right_mask) > polynomial_order + 1:
            x_right = running_var[right_mask]
            y_right = outcome[right_mask]
            coeffs_right = np.polyfit(x_right, y_right, polynomial_order)
            x_fit_right = np.linspace(cutoff, x_right.max(), 100)
            y_fit_right = np.polyval(coeffs_right, x_fit_right)
            fitted_right = (x_fit_right, y_fit_right)

        # Plot fits
        if fitted_left is not None:
            ax.plot(fitted_left[0], fitted_left[1],
                    color=colors['fit_control'], linewidth=2,
                    label='Fit (control)')

        if fitted_right is not None:
            ax.plot(fitted_right[0], fitted_right[1],
                    color=colors['fit_treatment'], linewidth=2,
                    label='Fit (treatment)')

    # Vertical line at cutoff
    ax.axvline(x=cutoff, color=colors['cutoff'], linestyle='--',
               linewidth=1.5, alpha=0.8, label=f'Cutoff = {cutoff}')

    # Labels and formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if return_fig:
        return fig

    plt.show()
    return None


def plot_propensity_overlap(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bins: int = 50,
    x_label: str = "Propensity Score",
    y_label: str = "Density",
    title: Optional[str] = None,
    show_overlap_region: bool = True,
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    return_fig: bool = False
) -> Optional[Any]:
    """
    Plot propensity score distribution overlap between treatment and control.

    Creates a publication-quality histogram/density plot showing the
    distribution of propensity scores for treated and control units,
    useful for assessing overlap/common support assumption.

    Parameters
    ----------
    propensity_scores : np.ndarray
        Estimated propensity scores
    treatment : np.ndarray
        Treatment indicator (0/1)
    weights : Optional[np.ndarray]
        Sample weights for weighted histogram
    n_bins : int
        Number of histogram bins
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    title : Optional[str]
        Plot title
    show_overlap_region : bool
        Whether to shade the overlap region
    colors : Optional[Dict[str, str]]
        Custom colors with keys: 'control', 'treatment', 'overlap'
    figsize : Tuple[float, float]
        Figure size
    save_path : Optional[str]
        Path to save figure
    return_fig : bool
        If True, return figure object

    Returns
    -------
    Optional[matplotlib.figure.Figure]
        Figure object if return_fig=True
    """
    plt, mpl = _get_matplotlib()
    set_publication_style()

    propensity_scores = np.asarray(propensity_scores)
    treatment = np.asarray(treatment)

    # Set colors
    if colors is None:
        colors = {
            'control': COLORS['control'],
            'treatment': COLORS['treatment'],
            'overlap': COLORS['tertiary']
        }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Separate by treatment status
    ps_control = propensity_scores[treatment == 0]
    ps_treated = propensity_scores[treatment == 1]

    # Common bin edges
    bins = np.linspace(0, 1, n_bins + 1)

    # Plot histograms
    ax.hist(ps_control, bins=bins, alpha=0.5, density=True,
            color=colors['control'], label=f'Control (n={len(ps_control)})',
            edgecolor='white', linewidth=0.5)

    ax.hist(ps_treated, bins=bins, alpha=0.5, density=True,
            color=colors['treatment'], label=f'Treatment (n={len(ps_treated)})',
            edgecolor='white', linewidth=0.5)

    # Show overlap region
    if show_overlap_region:
        overlap_min = max(ps_control.min(), ps_treated.min())
        overlap_max = min(ps_control.max(), ps_treated.max())

        if overlap_min < overlap_max:
            ax.axvspan(overlap_min, overlap_max, alpha=0.1,
                       color=colors['overlap'], label='Common Support')

            # Add text annotation for overlap
            overlap_pct_control = np.mean((ps_control >= overlap_min) & (ps_control <= overlap_max)) * 100
            overlap_pct_treated = np.mean((ps_treated >= overlap_min) & (ps_treated <= overlap_max)) * 100

            ax.annotate(
                f'Overlap: {overlap_min:.2f} - {overlap_max:.2f}\n'
                f'Control: {overlap_pct_control:.1f}%\n'
                f'Treatment: {overlap_pct_treated:.1f}%',
                xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top',
                fontsize=9, alpha=0.8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

    # Labels and formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    ax.set_xlim(0, 1)
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if return_fig:
        return fig

    plt.show()
    return None


def plot_cate_heterogeneity(
    cate_values: np.ndarray,
    covariate: np.ndarray,
    covariate_name: str = "Covariate",
    cate_label: str = "CATE",
    n_bins: int = 10,
    show_ci: bool = True,
    confidence_level: float = 0.95,
    plot_type: str = "binned",
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    return_fig: bool = False
) -> Optional[Any]:
    """
    Plot heterogeneous treatment effects (CATE) by covariate.

    Creates a visualization of how treatment effects vary with a covariate,
    useful for exploring heterogeneity in causal effects.

    Parameters
    ----------
    cate_values : np.ndarray
        Conditional average treatment effect estimates for each unit
    covariate : np.ndarray
        Covariate values (continuous or categorical)
    covariate_name : str
        Name of the covariate for axis label
    cate_label : str
        Label for CATE
    n_bins : int
        Number of bins for continuous covariate
    show_ci : bool
        Whether to show confidence intervals
    confidence_level : float
        Confidence level for error bars
    plot_type : str
        Type of plot: 'binned' (binned scatter), 'scatter' (full scatter),
        'smooth' (with LOWESS)
    colors : Optional[Dict[str, str]]
        Custom colors
    figsize : Tuple[float, float]
        Figure size
    save_path : Optional[str]
        Path to save figure
    return_fig : bool
        If True, return figure object

    Returns
    -------
    Optional[matplotlib.figure.Figure]
        Figure object if return_fig=True
    """
    plt, mpl = _get_matplotlib()
    from scipy import stats
    set_publication_style()

    cate_values = np.asarray(cate_values)
    covariate = np.asarray(covariate)

    # Set colors
    if colors is None:
        colors = {
            'point': COLORS['primary'],
            'ci': COLORS['primary'],
            'reference': COLORS['neutral']
        }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Reference line at zero
    ax.axhline(y=0, color=colors['reference'], linestyle='--', linewidth=1, alpha=0.7)

    if plot_type == "scatter":
        ax.scatter(covariate, cate_values, c=colors['point'], alpha=0.3, s=20)

    elif plot_type == "smooth":
        # Scatter with LOWESS smoothing
        ax.scatter(covariate, cate_values, c=colors['point'], alpha=0.2, s=10)

        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(cate_values, covariate, frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color=colors['point'],
                    linewidth=2, label='LOWESS')
        except ImportError:
            # Fallback to polynomial fit
            sort_idx = np.argsort(covariate)
            coeffs = np.polyfit(covariate[sort_idx], cate_values[sort_idx], 3)
            x_smooth = np.linspace(covariate.min(), covariate.max(), 100)
            y_smooth = np.polyval(coeffs, x_smooth)
            ax.plot(x_smooth, y_smooth, color=colors['point'], linewidth=2, label='Polynomial fit')

    else:  # binned (default)
        # Create bins
        bin_edges = np.percentile(covariate, np.linspace(0, 100, n_bins + 1))
        bin_centers = []
        bin_means = []
        bin_sems = []

        for i in range(len(bin_edges) - 1):
            in_bin = (covariate >= bin_edges[i]) & (covariate < bin_edges[i + 1])
            if i == len(bin_edges) - 2:  # Include right edge for last bin
                in_bin = (covariate >= bin_edges[i]) & (covariate <= bin_edges[i + 1])

            if np.sum(in_bin) > 0:
                bin_centers.append(covariate[in_bin].mean())
                bin_means.append(cate_values[in_bin].mean())
                bin_sems.append(stats.sem(cate_values[in_bin]) if np.sum(in_bin) > 1 else 0)

        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_sems = np.array(bin_sems)

        # Calculate CI
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        ci_lower = bin_means - z * bin_sems
        ci_upper = bin_means + z * bin_sems

        if show_ci:
            ax.fill_between(bin_centers, ci_lower, ci_upper,
                            alpha=0.2, color=colors['ci'])

        ax.plot(bin_centers, bin_means, 'o-', color=colors['point'],
                markersize=8, linewidth=2, label=cate_label)

        if show_ci:
            ax.errorbar(bin_centers, bin_means, yerr=z * bin_sems,
                        fmt='none', color=colors['point'], alpha=0.5, capsize=3)

    # Labels and formatting
    ax.set_xlabel(covariate_name)
    ax.set_ylabel(cate_label)

    if plot_type != "scatter":
        ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if return_fig:
        return fig

    plt.show()
    return None


@dataclass
class CoefficientComparison:
    """Data structure for coefficient comparison plots."""
    model_names: List[str]
    coefficients: List[float]
    ci_lower: List[float]
    ci_upper: List[float]
    colors: Optional[List[str]] = None


def plot_coef_comparison(
    data: Union[CoefficientComparison, Dict[str, Any], List[Dict[str, Any]]],
    coef_name: str = "Treatment Effect",
    x_label: str = "Coefficient",
    y_label: str = "Model",
    title: Optional[str] = None,
    show_reference_line: bool = True,
    reference_value: float = 0,
    horizontal: bool = True,
    colors: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    return_fig: bool = False
) -> Optional[Any]:
    """
    Compare coefficients across multiple models or specifications.

    Creates a forest plot style visualization comparing point estimates
    and confidence intervals across different model specifications.

    Parameters
    ----------
    data : Union[CoefficientComparison, Dict[str, Any], List[Dict[str, Any]]]
        Coefficient data. Can be:
        - CoefficientComparison object
        - Dict with keys: 'model_names', 'coefficients', 'ci_lower', 'ci_upper'
        - List of dicts with keys: 'model', 'coef', 'ci_lower', 'ci_upper'
    coef_name : str
        Name of the coefficient being compared
    x_label : str
        X-axis label (for horizontal plot)
    y_label : str
        Y-axis label
    title : Optional[str]
        Plot title
    show_reference_line : bool
        Whether to show reference line at reference_value
    reference_value : float
        Value for reference line (typically 0)
    horizontal : bool
        If True, create horizontal forest plot; if False, vertical
    colors : Optional[List[str]]
        Custom colors for each model
    figsize : Optional[Tuple[float, float]]
        Figure size (auto-calculated if None)
    save_path : Optional[str]
        Path to save figure
    return_fig : bool
        If True, return figure object

    Returns
    -------
    Optional[matplotlib.figure.Figure]
        Figure object if return_fig=True
    """
    plt, mpl = _get_matplotlib()
    set_publication_style()

    # Parse data
    if isinstance(data, CoefficientComparison):
        model_names = data.model_names
        coefficients = data.coefficients
        ci_lower = data.ci_lower
        ci_upper = data.ci_upper
        if colors is None:
            colors = data.colors
    elif isinstance(data, dict):
        model_names = data.get('model_names', [])
        coefficients = data.get('coefficients', [])
        ci_lower = data.get('ci_lower', [])
        ci_upper = data.get('ci_upper', [])
    elif isinstance(data, list):
        model_names = [d.get('model', f'Model {i+1}') for i, d in enumerate(data)]
        coefficients = [d.get('coef', 0) for d in data]
        ci_lower = [d.get('ci_lower', 0) for d in data]
        ci_upper = [d.get('ci_upper', 0) for d in data]
    else:
        raise ValueError("data must be CoefficientComparison, dict, or list of dicts")

    n_models = len(model_names)

    # Set colors
    if colors is None:
        cmap = plt.cm.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(n_models)]

    # Auto-calculate figure size
    if figsize is None:
        if horizontal:
            figsize = (10, max(4, n_models * 0.6))
        else:
            figsize = (max(6, n_models * 1.2), 6)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate error bars
    errors = [[c - l for c, l in zip(coefficients, ci_lower)],
              [u - c for u, c in zip(coefficients, ci_upper)]]

    positions = np.arange(n_models)

    if horizontal:
        # Horizontal forest plot
        for i, (pos, coef, color, name) in enumerate(zip(positions, coefficients, colors, model_names)):
            ax.errorbar(coef, pos, xerr=[[errors[0][i]], [errors[1][i]]],
                        fmt='o', color=color, markersize=10, capsize=5,
                        capthick=2, linewidth=2)

        if show_reference_line:
            ax.axvline(x=reference_value, color=COLORS['neutral'],
                       linestyle='--', linewidth=1, alpha=0.7)

        ax.set_yticks(positions)
        ax.set_yticklabels(model_names)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Invert y-axis so first model is at top
        ax.invert_yaxis()

    else:
        # Vertical plot
        for i, (pos, coef, color, name) in enumerate(zip(positions, coefficients, colors, model_names)):
            ax.errorbar(pos, coef, yerr=[[errors[0][i]], [errors[1][i]]],
                        fmt='o', color=color, markersize=10, capsize=5,
                        capthick=2, linewidth=2)

        if show_reference_line:
            ax.axhline(y=reference_value, color=COLORS['neutral'],
                       linestyle='--', linewidth=1, alpha=0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(x_label)
        ax.set_xlabel(y_label)

    if title:
        ax.set_title(title)

    # Add coefficient values as text
    for i, (coef, lower, upper) in enumerate(zip(coefficients, ci_lower, ci_upper)):
        text = f'{coef:.3f}\n[{lower:.3f}, {upper:.3f}]'
        if horizontal:
            ax.annotate(text, xy=(coef, i), xytext=(10, 0),
                        textcoords='offset points', fontsize=8, va='center')
        else:
            ax.annotate(text, xy=(i, coef), xytext=(0, 10),
                        textcoords='offset points', fontsize=8, ha='center')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if return_fig:
        return fig

    plt.show()
    return None


def create_causal_figure(
    n_panels: int = 1,
    panel_arrangement: str = "horizontal",
    figsize: Optional[Tuple[float, float]] = None,
    share_y: bool = False,
    share_x: bool = False
) -> Tuple[Any, Any]:
    """
    Create a figure with multiple panels for causal inference results.

    Utility function for creating multi-panel figures with consistent
    styling for causal inference visualizations.

    Parameters
    ----------
    n_panels : int
        Number of panels
    panel_arrangement : str
        'horizontal' (1 row) or 'vertical' (1 column) or 'grid' (automatic)
    figsize : Optional[Tuple[float, float]]
        Figure size (auto-calculated if None)
    share_y : bool
        Whether panels share y-axis
    share_x : bool
        Whether panels share x-axis

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes or array]
        Figure and axes objects
    """
    plt, mpl = _get_matplotlib()
    set_publication_style()

    # Determine layout
    if panel_arrangement == "horizontal":
        nrows, ncols = 1, n_panels
        if figsize is None:
            figsize = (5 * n_panels, 5)
    elif panel_arrangement == "vertical":
        nrows, ncols = n_panels, 1
        if figsize is None:
            figsize = (8, 4 * n_panels)
    else:  # grid
        ncols = int(np.ceil(np.sqrt(n_panels)))
        nrows = int(np.ceil(n_panels / ncols))
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharey=share_y, sharex=share_x,
                             squeeze=False)

    if n_panels == 1:
        axes = axes[0, 0]
    else:
        axes = axes.flatten()[:n_panels]

    return fig, axes


def save_all_formats(
    fig: Any,
    base_path: str,
    formats: List[str] = ['png', 'pdf', 'svg'],
    dpi: int = 300
) -> List[str]:
    """
    Save a figure in multiple formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    base_path : str
        Base path without extension
    formats : List[str]
        List of formats to save
    dpi : int
        Resolution for raster formats

    Returns
    -------
    List[str]
        List of saved file paths
    """
    saved_paths = []
    for fmt in formats:
        path = f"{base_path}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches='tight', format=fmt)
        saved_paths.append(path)
    return saved_paths
