#!/usr/bin/env python3
"""
Event Study Visualization Module.

This module provides publication-quality event study plots with various
customization options for DID analysis.

Usage:
    # As CLI tool
    python visualize_event_study.py data.csv --outcome y --treatment-time first_treated \\
        --unit id --time year --output event_study.png

    # As module
    from visualize_event_study import EventStudyPlotter
    plotter = EventStudyPlotter(data, ...)
    fig = plotter.create_plot()

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class EventStudyEstimates:
    """Container for event study estimates."""
    event_times: List[int]
    coefficients: List[float]
    std_errors: List[float]
    ci_lower: List[float]
    ci_upper: List[float]
    reference_period: int
    joint_f_stat: Optional[float] = None
    joint_f_pval: Optional[float] = None
    method: str = "TWFE"


class EventStudyPlotter:
    """
    Create publication-quality event study plots.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable name
    treatment_time_var : str
        Column with unit's first treatment time
    unit_id : str
        Unit identifier
    time_id : str
        Time period
    """

    # Default style settings
    DEFAULT_STYLE = {
        'figure_size': (10, 6),
        'font_family': 'sans-serif',
        'title_size': 14,
        'label_size': 12,
        'tick_size': 10,
        'legend_size': 10,
        'point_color': '#1f77b4',  # Blue
        'ci_color': '#1f77b4',
        'ci_alpha': 0.2,
        'marker_size': 8,
        'line_width': 1.5,
        'treatment_line_color': '#d62728',  # Red
        'zero_line_color': 'black',
        'grid_alpha': 0.3,
        'capsize': 3,
        'dpi': 300
    }

    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment_time_var: str,
        unit_id: str,
        time_id: str
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        self.data = data.copy()
        self.outcome = outcome
        self.treatment_time_var = treatment_time_var
        self.unit_id = unit_id
        self.time_id = time_id
        self.style = self.DEFAULT_STYLE.copy()

        # Create event time variable
        self._create_event_time()

    def _create_event_time(self) -> None:
        """Create event time variable relative to treatment."""
        self.data['_treat_time'] = self.data[self.treatment_time_var].fillna(np.inf)
        self.data['_event_time'] = self.data[self.time_id] - self.data['_treat_time']

        # Identify treated units
        self.data['_is_treated'] = self.data['_treat_time'] != np.inf

    def estimate_event_study(
        self,
        pre_periods: int = 4,
        post_periods: int = 4,
        reference_period: int = -1,
        controls: List[str] = None,
        cluster: str = None,
        method: str = "TWFE"
    ) -> EventStudyEstimates:
        """
        Estimate event study coefficients.

        Parameters
        ----------
        pre_periods : int
            Number of pre-treatment periods
        post_periods : int
            Number of post-treatment periods
        reference_period : int
            Event time to normalize to zero
        controls : list
            Control variables
        cluster : str
            Clustering variable
        method : str
            Estimation method: "TWFE", "Callaway-SantAnna", "Sun-Abraham"

        Returns
        -------
        EventStudyEstimates
            Event study coefficient estimates
        """
        if method == "TWFE":
            return self._estimate_twfe(
                pre_periods, post_periods, reference_period, controls, cluster
            )
        elif method == "Callaway-SantAnna":
            return self._estimate_callaway_santanna(
                pre_periods, post_periods, reference_period
            )
        elif method == "Sun-Abraham":
            return self._estimate_sun_abraham(
                pre_periods, post_periods, reference_period, cluster
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def _estimate_twfe(
        self,
        pre_periods: int,
        post_periods: int,
        reference_period: int,
        controls: List[str],
        cluster: str
    ) -> EventStudyEstimates:
        """Estimate event study using two-way fixed effects."""
        try:
            from linearmodels.panel import PanelOLS
        except ImportError:
            raise ImportError("linearmodels required for TWFE estimation")

        df = self.data[self.data['_is_treated']].copy()

        # Create event time dummies
        event_times = list(range(-pre_periods, post_periods + 1))
        event_times = [e for e in event_times if e != reference_period]

        for e in event_times:
            df[f'_et_{e}'] = (df['_event_time'] == e).astype(int)

        # Bin endpoints
        df['_et_pre'] = (df['_event_time'] < -pre_periods).astype(int)
        df['_et_post'] = (df['_event_time'] > post_periods).astype(int)

        # Set up panel
        df_panel = df.set_index([self.unit_id, self.time_id])

        y = df_panel[self.outcome]
        X_vars = [f'_et_{e}' for e in event_times] + ['_et_pre', '_et_post']
        if controls:
            X_vars.extend(controls)
        X = df_panel[X_vars]

        # Handle missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        # Fit model
        model = PanelOLS(y, X, entity_effects=True, time_effects=True)

        if cluster:
            results = model.fit(cov_type='clustered', cluster_entity=True)
        else:
            results = model.fit(cov_type='robust')

        # Extract coefficients
        event_time_list = sorted(event_times) + [reference_period]
        event_time_list = sorted(event_time_list)
        coefficients = []
        std_errors = []

        for e in event_time_list:
            if e == reference_period:
                coefficients.append(0.0)
                std_errors.append(0.0)
            else:
                var_name = f'_et_{e}'
                if var_name in results.params.index:
                    coefficients.append(results.params[var_name])
                    std_errors.append(results.std_errors[var_name])
                else:
                    coefficients.append(np.nan)
                    std_errors.append(np.nan)

        coefficients = np.array(coefficients)
        std_errors = np.array(std_errors)
        ci_lower = coefficients - 1.96 * std_errors
        ci_upper = coefficients + 1.96 * std_errors

        # Joint F-test for pre-trends
        pre_vars = [f'_et_{e}' for e in event_times if e < 0]
        if len(pre_vars) > 0:
            pre_coefs = results.params[pre_vars].values
            pre_vcov = results.cov.loc[pre_vars, pre_vars].values
            try:
                joint_f_stat = pre_coefs @ np.linalg.inv(pre_vcov) @ pre_coefs / len(pre_vars)
                joint_f_pval = 1 - stats.f.cdf(joint_f_stat, len(pre_vars), results.df_resid)
            except np.linalg.LinAlgError:
                joint_f_stat = np.nan
                joint_f_pval = np.nan
        else:
            joint_f_stat = np.nan
            joint_f_pval = np.nan

        return EventStudyEstimates(
            event_times=event_time_list,
            coefficients=coefficients.tolist(),
            std_errors=std_errors.tolist(),
            ci_lower=ci_lower.tolist(),
            ci_upper=ci_upper.tolist(),
            reference_period=reference_period,
            joint_f_stat=joint_f_stat,
            joint_f_pval=joint_f_pval,
            method="TWFE"
        )

    def _estimate_callaway_santanna(
        self,
        pre_periods: int,
        post_periods: int,
        reference_period: int
    ) -> EventStudyEstimates:
        """
        Estimate event study using Callaway-Sant'Anna (2021).

        This is a simplified implementation that aggregates group-time ATTs
        by event time.
        """
        # Add parent directory for imports
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from did_estimator import estimate_did_staggered

        # Get staggered DID results
        result = estimate_did_staggered(
            data=self.data,
            outcome=self.outcome,
            treatment_time=self.treatment_time_var,
            unit_id=self.unit_id,
            time_id=self.time_id,
            control_group="nevertreated"
        )

        # Extract event-time effects from diagnostics
        event_time_effects = result.diagnostics.get('event_time_effects', {})

        event_times = list(range(-pre_periods, post_periods + 1))
        coefficients = []
        std_errors = []

        for e in event_times:
            if e == reference_period:
                coefficients.append(0.0)
                std_errors.append(0.0)
            elif e in event_time_effects:
                coefficients.append(event_time_effects[e])
                # Simplified SE (would need bootstrap for proper SE)
                std_errors.append(result.se * 1.5)  # Rough approximation
            else:
                coefficients.append(np.nan)
                std_errors.append(np.nan)

        coefficients = np.array(coefficients)
        std_errors = np.array(std_errors)
        ci_lower = coefficients - 1.96 * std_errors
        ci_upper = coefficients + 1.96 * std_errors

        return EventStudyEstimates(
            event_times=event_times,
            coefficients=coefficients.tolist(),
            std_errors=std_errors.tolist(),
            ci_lower=ci_lower.tolist(),
            ci_upper=ci_upper.tolist(),
            reference_period=reference_period,
            method="Callaway-Sant'Anna"
        )

    def _estimate_sun_abraham(
        self,
        pre_periods: int,
        post_periods: int,
        reference_period: int,
        cluster: str
    ) -> EventStudyEstimates:
        """
        Estimate event study using Sun and Abraham (2021) IW estimator.

        Saturates the model with cohort-specific event-time indicators.
        """
        try:
            from linearmodels.panel import PanelOLS
        except ImportError:
            raise ImportError("linearmodels required")

        df = self.data.copy()

        # Identify cohorts
        df['_cohort'] = df.groupby(self.unit_id)[self.treatment_time_var].transform('first')
        df['_cohort'] = df['_cohort'].fillna(np.inf)
        cohorts = df[df['_cohort'] != np.inf]['_cohort'].unique()

        event_times = list(range(-pre_periods, post_periods + 1))
        event_times = [e for e in event_times if e != reference_period]

        # Create cohort-specific event time dummies
        for g in cohorts:
            for e in event_times:
                df[f'_et_{int(g)}_{e}'] = (
                    (df['_cohort'] == g) &
                    (df['_event_time'] == e)
                ).astype(int)

        # Only use treated units
        df_treated = df[df['_cohort'] != np.inf].copy()
        df_panel = df_treated.set_index([self.unit_id, self.time_id])

        y = df_panel[self.outcome]
        X_vars = [c for c in df_panel.columns if c.startswith('_et_') and c not in ['_event_time']]
        X = df_panel[X_vars]

        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        model = PanelOLS(y, X, entity_effects=True, time_effects=True)

        if cluster:
            results = model.fit(cov_type='clustered', cluster_entity=True)
        else:
            results = model.fit(cov_type='robust')

        # Aggregate to event-time level (IW estimator)
        # Weight by cohort size
        cohort_sizes = df_treated.groupby('_cohort')[self.unit_id].nunique()
        total_treated = cohort_sizes.sum()
        cohort_weights = cohort_sizes / total_treated

        event_time_list = sorted(event_times + [reference_period])
        coefficients = []
        std_errors = []

        for e in event_time_list:
            if e == reference_period:
                coefficients.append(0.0)
                std_errors.append(0.0)
            else:
                # Aggregate cohort-specific coefficients
                agg_coef = 0.0
                agg_var = 0.0
                for g in cohorts:
                    var_name = f'_et_{int(g)}_{e}'
                    if var_name in results.params.index:
                        w = cohort_weights.get(g, 0)
                        agg_coef += w * results.params[var_name]
                        agg_var += (w ** 2) * (results.std_errors[var_name] ** 2)

                coefficients.append(agg_coef)
                std_errors.append(np.sqrt(agg_var))

        coefficients = np.array(coefficients)
        std_errors = np.array(std_errors)
        ci_lower = coefficients - 1.96 * std_errors
        ci_upper = coefficients + 1.96 * std_errors

        return EventStudyEstimates(
            event_times=event_time_list,
            coefficients=coefficients.tolist(),
            std_errors=std_errors.tolist(),
            ci_lower=ci_lower.tolist(),
            ci_upper=ci_upper.tolist(),
            reference_period=reference_period,
            method="Sun-Abraham"
        )

    def update_style(self, **kwargs) -> None:
        """Update plot style settings."""
        for key, value in kwargs.items():
            if key in self.style:
                self.style[key] = value
            else:
                print(f"Warning: Unknown style parameter '{key}'")

    def create_plot(
        self,
        estimates: EventStudyEstimates = None,
        pre_periods: int = 4,
        post_periods: int = 4,
        reference_period: int = -1,
        title: str = "Event Study: Dynamic Treatment Effects",
        xlabel: str = "Event Time (periods relative to treatment)",
        ylabel: str = None,
        show_pretrend_test: bool = True,
        show_legend: bool = True,
        figsize: Tuple[int, int] = None,
        ax: plt.Axes = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create event study plot.

        Parameters
        ----------
        estimates : EventStudyEstimates
            Pre-computed estimates (computed if not provided)
        pre_periods : int
            Number of pre-treatment periods
        post_periods : int
            Number of post-treatment periods
        reference_period : int
            Event time normalized to zero
        title : str
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label (default: "Effect on {outcome}")
        show_pretrend_test : bool
            Show pre-trends test result
        show_legend : bool
            Show legend
        figsize : tuple
            Figure size
        ax : matplotlib.axes.Axes
            Existing axes to plot on

        Returns
        -------
        tuple
            (figure, axes) matplotlib objects
        """
        # Compute estimates if not provided
        if estimates is None:
            estimates = self.estimate_event_study(
                pre_periods=pre_periods,
                post_periods=post_periods,
                reference_period=reference_period
            )

        # Create figure
        if ax is None:
            figsize = figsize or self.style['figure_size']
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Extract data
        event_times = estimates.event_times
        coefs = np.array(estimates.coefficients)
        ci_low = np.array(estimates.ci_lower)
        ci_high = np.array(estimates.ci_upper)

        # Reference lines
        ax.axhline(y=0, color=self.style['zero_line_color'],
                  linestyle='-', linewidth=0.5)
        ax.axvline(x=-0.5, color=self.style['treatment_line_color'],
                  linestyle='--', linewidth=2, label='Treatment')

        # Plot coefficients with error bars
        ax.errorbar(
            event_times,
            coefs,
            yerr=[coefs - ci_low, ci_high - coefs],
            fmt='o',
            color=self.style['point_color'],
            capsize=self.style['capsize'],
            capthick=self.style['line_width'],
            linewidth=self.style['line_width'],
            markersize=self.style['marker_size'],
            label='Point Estimate'
        )

        # Shade confidence interval
        ax.fill_between(
            event_times,
            ci_low,
            ci_high,
            alpha=self.style['ci_alpha'],
            color=self.style['ci_color']
        )

        # Mark reference period
        ref_idx = event_times.index(reference_period)
        ax.scatter(
            [reference_period], [0],
            marker='s', s=100, color='green', zorder=5,
            label=f'Reference (t={reference_period})'
        )

        # Labels and title
        ax.set_xlabel(xlabel, fontsize=self.style['label_size'])
        ylabel = ylabel or f'Effect on {self.outcome}'
        ax.set_ylabel(ylabel, fontsize=self.style['label_size'])
        ax.set_title(title, fontsize=self.style['title_size'])

        # Format axes
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', which='major', labelsize=self.style['tick_size'])
        ax.grid(True, alpha=self.style['grid_alpha'])

        # Legend
        if show_legend:
            ax.legend(loc='best', fontsize=self.style['legend_size'])

        # Pre-trends test annotation
        if show_pretrend_test and estimates.joint_f_pval is not None:
            status = "PASSED" if estimates.joint_f_pval > 0.05 else "CONCERN"
            color = 'green' if estimates.joint_f_pval > 0.05 else 'red'
            ax.text(
                0.02, 0.98,
                f'Pre-trends test: p = {estimates.joint_f_pval:.3f} ({status})',
                transform=ax.transAxes,
                fontsize=self.style['tick_size'],
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                color=color
            )

        # Method annotation
        ax.text(
            0.98, 0.02,
            f'Method: {estimates.method}',
            transform=ax.transAxes,
            fontsize=self.style['tick_size'] - 1,
            verticalalignment='bottom',
            horizontalalignment='right',
            style='italic',
            alpha=0.7
        )

        plt.tight_layout()

        return fig, ax

    def create_comparison_plot(
        self,
        methods: List[str] = None,
        pre_periods: int = 4,
        post_periods: int = 4,
        reference_period: int = -1,
        title: str = "Event Study: Method Comparison",
        figsize: Tuple[int, int] = (12, 6)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create comparison plot with multiple estimation methods.

        Parameters
        ----------
        methods : list
            Methods to compare (default: ["TWFE", "Sun-Abraham"])
        pre_periods : int
            Number of pre-treatment periods
        post_periods : int
            Number of post-treatment periods
        reference_period : int
            Event time normalized to zero
        title : str
            Plot title
        figsize : tuple
            Figure size

        Returns
        -------
        tuple
            (figure, axes) matplotlib objects
        """
        methods = methods or ["TWFE", "Sun-Abraham"]
        colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

        fig, ax = plt.subplots(figsize=figsize)

        # Reference lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=-0.5, color='gray', linestyle='--', linewidth=1.5)

        for i, method in enumerate(methods):
            try:
                estimates = self.estimate_event_study(
                    pre_periods=pre_periods,
                    post_periods=post_periods,
                    reference_period=reference_period,
                    method=method
                )

                event_times = np.array(estimates.event_times)
                coefs = np.array(estimates.coefficients)
                ci_low = np.array(estimates.ci_lower)
                ci_high = np.array(estimates.ci_upper)

                # Offset for visibility
                offset = (i - len(methods)/2 + 0.5) * 0.1

                ax.errorbar(
                    event_times + offset,
                    coefs,
                    yerr=[coefs - ci_low, ci_high - coefs],
                    fmt='o',
                    color=colors[i % len(colors)],
                    capsize=3,
                    linewidth=1.5,
                    markersize=6,
                    label=method
                )

            except Exception as e:
                print(f"Warning: Could not estimate {method}: {e}")

        ax.set_xlabel('Event Time', fontsize=12)
        ax.set_ylabel(f'Effect on {self.outcome}', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()

        return fig, ax

    def save_plot(
        self,
        fig: plt.Figure,
        filepath: str,
        dpi: int = None,
        format: str = None
    ) -> None:
        """
        Save plot to file.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        filepath : str
            Output file path
        dpi : int
            Resolution (default from style)
        format : str
            Output format (inferred from extension if not provided)
        """
        dpi = dpi or self.style['dpi']
        fig.savefig(filepath, dpi=dpi, format=format, bbox_inches='tight')
        print(f"Saved plot to {filepath}")


def main():
    """Command line interface for event study visualization."""
    parser = argparse.ArgumentParser(
        description="Create event study plots for DID analysis"
    )

    parser.add_argument("data_path", help="Path to data file")
    parser.add_argument("--outcome", "-o", required=True, help="Outcome variable")
    parser.add_argument("--treatment-time", "-t", required=True, help="Treatment time variable")
    parser.add_argument("--unit", "-u", required=True, help="Unit ID variable")
    parser.add_argument("--time", "-T", required=True, help="Time variable")
    parser.add_argument("--pre-periods", type=int, default=4, help="Pre-treatment periods")
    parser.add_argument("--post-periods", type=int, default=4, help="Post-treatment periods")
    parser.add_argument("--reference", type=int, default=-1, help="Reference period")
    parser.add_argument("--method", default="TWFE", choices=["TWFE", "Sun-Abraham"],
                       help="Estimation method")
    parser.add_argument("--output", "-O", default="event_study.png", help="Output file")
    parser.add_argument("--title", default="Event Study", help="Plot title")
    parser.add_argument("--compare", action="store_true", help="Compare methods")

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data_path)

    # Create plotter
    plotter = EventStudyPlotter(
        data=data,
        outcome=args.outcome,
        treatment_time_var=args.treatment_time,
        unit_id=args.unit,
        time_id=args.time
    )

    # Create plot
    if args.compare:
        fig, ax = plotter.create_comparison_plot(
            pre_periods=args.pre_periods,
            post_periods=args.post_periods,
            reference_period=args.reference,
            title=args.title
        )
    else:
        estimates = plotter.estimate_event_study(
            pre_periods=args.pre_periods,
            post_periods=args.post_periods,
            reference_period=args.reference,
            method=args.method
        )
        fig, ax = plotter.create_plot(
            estimates=estimates,
            title=args.title
        )

    # Save
    plotter.save_plot(fig, args.output)


if __name__ == "__main__":
    main()
