#!/usr/bin/env python3
"""
Parallel Trends Testing Module for Difference-in-Differences.

This module provides comprehensive parallel trends testing including:
- Visual inspection tools
- Statistical tests (linear trend, joint F-test)
- Event study specification tests
- Rambachan-Roth sensitivity analysis

Usage:
    # As CLI tool
    python test_parallel_trends.py data.csv --outcome y --treatment-group treated \\
        --unit id --time year --treatment-time 2015

    # As module
    from test_parallel_trends import ParallelTrendsAnalyzer
    analyzer = ParallelTrendsAnalyzer(data, outcome='y', ...)
    results = analyzer.run_full_analysis()

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class TrendTestResult:
    """Result from a parallel trends test."""
    test_name: str
    statistic: float
    p_value: float
    passed: bool
    threshold: float
    interpretation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (f"TrendTestResult({self.test_name}: {status}, "
                f"stat={self.statistic:.4f}, p={self.p_value:.4f})")


@dataclass
class EventStudyResult:
    """Result from event study regression."""
    coefficients: Dict[int, float]
    std_errors: Dict[int, float]
    p_values: Dict[int, float]
    ci_lower: Dict[int, float]
    ci_upper: Dict[int, float]
    joint_f_stat: float
    joint_f_pval: float
    pre_coefficients_significant: bool
    reference_period: int


class ParallelTrendsAnalyzer:
    """
    Comprehensive parallel trends analysis for DID.

    This class provides multiple methods for testing and visualizing
    the parallel trends assumption.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable name
    treatment_group : str
        Ever-treated indicator (1 for treatment group)
    unit_id : str
        Unit identifier column
    time_id : str
        Time period column
    treatment_time : int
        When treatment began (for non-staggered DID)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment_group: str,
        unit_id: str,
        time_id: str,
        treatment_time: int
    ):
        self.data = data.copy()
        self.outcome = outcome
        self.treatment_group = treatment_group
        self.unit_id = unit_id
        self.time_id = time_id
        self.treatment_time = treatment_time

        # Get pre-treatment data
        self.pre_data = self.data[self.data[time_id] < treatment_time].copy()
        self.periods = sorted(self.data[time_id].unique())
        self.pre_periods = [p for p in self.periods if p < treatment_time]
        self.post_periods = [p for p in self.periods if p >= treatment_time]

        # Calculate group means
        self._calculate_group_means()

    def _calculate_group_means(self) -> None:
        """Calculate outcome means by treatment group and time."""
        self.group_means = self.data.groupby(
            [self.time_id, self.treatment_group]
        )[self.outcome].agg(['mean', 'std', 'count']).reset_index()

        self.group_means['se'] = (
            self.group_means['std'] / np.sqrt(self.group_means['count'])
        )

    def linear_trend_test(self) -> TrendTestResult:
        """
        Test for linear divergence in pre-treatment trends.

        Tests H0: The slope of (Treatment mean - Control mean) over time = 0

        Returns
        -------
        TrendTestResult
            Test result with slope estimate and p-value
        """
        if len(self.pre_periods) < 2:
            return TrendTestResult(
                test_name="Linear Trend Test",
                statistic=np.nan,
                p_value=np.nan,
                passed=False,
                threshold=0.05,
                interpretation="Insufficient pre-treatment periods for trend test",
                details={'error': 'insufficient_periods'}
            )

        # Calculate trend difference
        treat_means = self.group_means[
            (self.group_means[self.treatment_group] == 1) &
            (self.group_means[self.time_id] < self.treatment_time)
        ].set_index(self.time_id)['mean']

        control_means = self.group_means[
            (self.group_means[self.treatment_group] == 0) &
            (self.group_means[self.time_id] < self.treatment_time)
        ].set_index(self.time_id)['mean']

        # Align indices
        common_periods = sorted(set(treat_means.index) & set(control_means.index))
        if len(common_periods) < 2:
            return TrendTestResult(
                test_name="Linear Trend Test",
                statistic=np.nan,
                p_value=np.nan,
                passed=False,
                threshold=0.05,
                interpretation="Insufficient common periods between groups",
                details={'error': 'insufficient_common_periods'}
            )

        trend_diff = treat_means.loc[common_periods] - control_means.loc[common_periods]
        time_points = np.arange(len(trend_diff))

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            time_points, trend_diff.values
        )

        passed = p_value > 0.05

        if passed:
            interpretation = (
                f"Parallel trends supported: No significant linear divergence "
                f"(slope = {slope:.4f}, p = {p_value:.4f})"
            )
        else:
            interpretation = (
                f"Parallel trends VIOLATED: Significant linear divergence detected "
                f"(slope = {slope:.4f}, p = {p_value:.4f})"
            )

        return TrendTestResult(
            test_name="Linear Trend Test",
            statistic=slope,
            p_value=p_value,
            passed=passed,
            threshold=0.05,
            interpretation=interpretation,
            details={
                'slope': slope,
                'intercept': intercept,
                'std_err': std_err,
                'r_squared': r_value ** 2,
                'n_periods': len(common_periods),
                'trend_differences': trend_diff.to_dict()
            }
        )

    def event_study_test(
        self,
        n_pre: int = None,
        n_post: int = None,
        reference_period: int = -1,
        cluster: str = None
    ) -> EventStudyResult:
        """
        Run event study regression to test for pre-treatment effects.

        Model: Y_it = a_i + l_t + sum_k(b_k * D_it^k) + e_it
        where D_it^k = 1 if unit i is k periods from treatment at time t

        Parameters
        ----------
        n_pre : int
            Number of pre-treatment periods (default: all available)
        n_post : int
            Number of post-treatment periods (default: all available)
        reference_period : int
            Event time to normalize to zero (default: -1)
        cluster : str
            Variable to cluster standard errors on

        Returns
        -------
        EventStudyResult
            Coefficients, standard errors, and joint test results
        """
        try:
            from linearmodels.panel import PanelOLS
        except ImportError:
            raise ImportError("linearmodels required for event study test")

        df = self.data.copy()

        # Create event time variable
        # For units in treatment group, event_time = time - treatment_time
        # For control units, we don't create event dummies
        df['_ever_treated'] = df[self.treatment_group].astype(int)
        df['_event_time'] = np.where(
            df['_ever_treated'] == 1,
            df[self.time_id] - self.treatment_time,
            np.nan
        )

        # Determine event time range
        if n_pre is None:
            n_pre = len(self.pre_periods)
        if n_post is None:
            n_post = len(self.post_periods)

        event_times = list(range(-n_pre, n_post + 1))
        event_times = [e for e in event_times if e != reference_period]

        # Create event time dummies
        for e in event_times:
            df[f'_et_{e}'] = (
                (df['_ever_treated'] == 1) &
                (df['_event_time'] == e)
            ).astype(int)

        # Set up panel
        df_panel = df.set_index([self.unit_id, self.time_id])

        y = df_panel[self.outcome]
        X_vars = [f'_et_{e}' for e in event_times]
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
        coefficients = {}
        std_errors = {}
        p_values = {}
        ci_lower = {}
        ci_upper = {}

        for e in sorted(event_times):
            var_name = f'_et_{e}'
            if var_name in results.params.index:
                coefficients[e] = results.params[var_name]
                std_errors[e] = results.std_errors[var_name]
                p_values[e] = results.pvalues[var_name]
                ci_lower[e] = coefficients[e] - 1.96 * std_errors[e]
                ci_upper[e] = coefficients[e] + 1.96 * std_errors[e]

        # Add reference period
        coefficients[reference_period] = 0.0
        std_errors[reference_period] = 0.0
        p_values[reference_period] = 1.0
        ci_lower[reference_period] = 0.0
        ci_upper[reference_period] = 0.0

        # Joint F-test for pre-treatment coefficients
        pre_vars = [f'_et_{e}' for e in event_times if e < 0]
        if len(pre_vars) > 0:
            # Wald test
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

        # Check if any pre-treatment coefficient is significant
        pre_significant = any(
            abs(coefficients[e] / std_errors[e]) > 1.96
            for e in event_times
            if e < 0 and e in coefficients and std_errors.get(e, 0) > 0
        )

        return EventStudyResult(
            coefficients=coefficients,
            std_errors=std_errors,
            p_values=p_values,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            joint_f_stat=joint_f_stat,
            joint_f_pval=joint_f_pval,
            pre_coefficients_significant=pre_significant,
            reference_period=reference_period
        )

    def granger_causality_test(self) -> TrendTestResult:
        """
        Test for Granger causality from treatment group to outcome.

        If future treatment "Granger causes" past outcomes, parallel trends
        may be violated.

        Returns
        -------
        TrendTestResult
            Granger causality test result
        """
        # Use pre-treatment data only
        pre_data = self.pre_data.copy()

        # Create lagged treatment indicator
        pre_data = pre_data.sort_values([self.unit_id, self.time_id])
        pre_data['_treat_lead'] = pre_data.groupby(self.unit_id)[self.treatment_group].shift(-1)

        # Regress outcome on lagged treatment
        from scipy.stats import pearsonr

        # Simple correlation test
        valid_data = pre_data.dropna(subset=[self.outcome, '_treat_lead'])
        if len(valid_data) < 10:
            return TrendTestResult(
                test_name="Granger Causality Test",
                statistic=np.nan,
                p_value=np.nan,
                passed=False,
                threshold=0.05,
                interpretation="Insufficient data for Granger test",
                details={'error': 'insufficient_data'}
            )

        corr, p_value = pearsonr(valid_data[self.outcome], valid_data['_treat_lead'])

        passed = p_value > 0.05

        if passed:
            interpretation = (
                f"No evidence of reverse causality (corr = {corr:.4f}, p = {p_value:.4f})"
            )
        else:
            interpretation = (
                f"Potential reverse causality detected (corr = {corr:.4f}, p = {p_value:.4f})"
            )

        return TrendTestResult(
            test_name="Granger Causality Test",
            statistic=corr,
            p_value=p_value,
            passed=passed,
            threshold=0.05,
            interpretation=interpretation,
            details={'correlation': corr}
        )

    def rambachan_roth_sensitivity(
        self,
        main_effect: float,
        main_se: float,
        m_range: Tuple[float, float] = (-0.5, 0.5),
        n_points: int = 21
    ) -> Dict[str, Any]:
        """
        Rambachan and Roth (2023) sensitivity analysis.

        Computes bounds on the treatment effect under violations of
        parallel trends bounded by M.

        Parameters
        ----------
        main_effect : float
            Main DID estimate
        main_se : float
            Standard error of main estimate
        m_range : tuple
            Range of M values to explore (max deviation from parallel trends)
        n_points : int
            Number of M values to evaluate

        Returns
        -------
        dict
            Sensitivity analysis results including breakdown point
        """
        # Get pre-trend slope from linear test
        linear_result = self.linear_trend_test()
        pre_slope = linear_result.details.get('slope', 0)
        pre_slope_se = linear_result.details.get('std_err', 0)

        m_values = np.linspace(m_range[0], m_range[1], n_points)

        bounds_lower = []
        bounds_upper = []

        for m in m_values:
            # Adjust effect for possible trend deviation M
            # Lower bound: assume trend deviation worked against finding effect
            # Upper bound: assume trend deviation inflated effect
            n_post = len(self.post_periods)
            max_bias = m * n_post  # Cumulative bias from trend deviation

            lower = main_effect - max_bias - 1.96 * main_se
            upper = main_effect + max_bias + 1.96 * main_se

            bounds_lower.append(lower)
            bounds_upper.append(upper)

        # Find breakdown point: smallest M where CI includes 0
        breakdown_m = None
        for i, m in enumerate(m_values):
            if bounds_lower[i] <= 0 <= bounds_upper[i]:
                breakdown_m = abs(m)
                break

        return {
            'm_values': m_values.tolist(),
            'bounds_lower': bounds_lower,
            'bounds_upper': bounds_upper,
            'breakdown_point': breakdown_m,
            'pre_slope': pre_slope,
            'pre_slope_se': pre_slope_se,
            'main_effect': main_effect,
            'main_se': main_se
        }

    def plot_trends(
        self,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Parallel Trends Check",
        save_path: str = None
    ):
        """
        Plot treatment and control group trends.

        Parameters
        ----------
        figsize : tuple
            Figure size
        title : str
            Plot title
        save_path : str
            Path to save figure

        Returns
        -------
        matplotlib.figure.Figure
            Parallel trends plot
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot each group
        for group_val, label, color in [(0, 'Control', 'blue'), (1, 'Treatment', 'red')]:
            group_data = self.group_means[
                self.group_means[self.treatment_group] == group_val
            ].sort_values(self.time_id)

            ax.plot(
                group_data[self.time_id],
                group_data['mean'],
                marker='o',
                linewidth=2,
                label=label,
                color=color
            )

            ax.fill_between(
                group_data[self.time_id],
                group_data['mean'] - 1.96 * group_data['se'],
                group_data['mean'] + 1.96 * group_data['se'],
                alpha=0.2,
                color=color
            )

        # Add treatment line
        ax.axvline(
            x=self.treatment_time - 0.5,
            color='black',
            linestyle='--',
            linewidth=2,
            label='Treatment'
        )

        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel(f'Mean {self.outcome}', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_event_study(
        self,
        event_study_result: EventStudyResult = None,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Event Study",
        save_path: str = None
    ):
        """
        Plot event study coefficients.

        Parameters
        ----------
        event_study_result : EventStudyResult
            Pre-computed event study results (computed if not provided)
        figsize : tuple
            Figure size
        title : str
            Plot title
        save_path : str
            Path to save figure

        Returns
        -------
        matplotlib.figure.Figure
            Event study plot
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        if event_study_result is None:
            event_study_result = self.event_study_test()

        coefs = event_study_result.coefficients
        ci_low = event_study_result.ci_lower
        ci_high = event_study_result.ci_upper

        event_times = sorted(coefs.keys())
        estimates = [coefs[e] for e in event_times]
        lower = [ci_low[e] for e in event_times]
        upper = [ci_high[e] for e in event_times]

        fig, ax = plt.subplots(figsize=figsize)

        # Reference lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=2, label='Treatment')

        # Plot coefficients
        ax.errorbar(
            event_times,
            estimates,
            yerr=[np.array(estimates) - np.array(lower),
                  np.array(upper) - np.array(estimates)],
            fmt='o',
            color='blue',
            capsize=3,
            capthick=1.5,
            linewidth=1.5,
            markersize=8,
            label='Point Estimate'
        )

        # Shade confidence interval
        ax.fill_between(
            event_times,
            lower,
            upper,
            alpha=0.2,
            color='blue'
        )

        # Mark reference period
        ref_period = event_study_result.reference_period
        ax.scatter([ref_period], [0], marker='s', s=100, color='green',
                  zorder=5, label=f'Reference (t={ref_period})')

        # Labels
        ax.set_xlabel('Event Time (periods relative to treatment)', fontsize=12)
        ax.set_ylabel('Estimated Effect', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add pre-trends test result
        if event_study_result.joint_f_pval is not None:
            status = "PASSED" if event_study_result.joint_f_pval > 0.05 else "FAILED"
            ax.text(
                0.02, 0.98,
                f'Pre-trends joint F-test: p = {event_study_result.joint_f_pval:.3f} ({status})',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run all parallel trends tests and return comprehensive results.

        Returns
        -------
        dict
            Complete parallel trends analysis results
        """
        results = {}

        # Linear trend test
        results['linear_trend'] = self.linear_trend_test()

        # Event study
        try:
            results['event_study'] = self.event_study_test()
        except Exception as e:
            results['event_study'] = None
            results['event_study_error'] = str(e)

        # Granger test
        results['granger'] = self.granger_causality_test()

        # Overall assessment
        tests_passed = sum([
            results['linear_trend'].passed if results['linear_trend'] else False,
            not results.get('event_study', {}).pre_coefficients_significant
            if results.get('event_study') else False,
            results['granger'].passed if results['granger'] else False
        ])

        total_tests = 3

        results['summary'] = {
            'tests_passed': tests_passed,
            'total_tests': total_tests,
            'overall_assessment': 'SUPPORTED' if tests_passed >= 2 else 'CONCERN',
            'n_pre_periods': len(self.pre_periods)
        }

        return results


def main():
    """Command line interface for parallel trends testing."""
    parser = argparse.ArgumentParser(
        description="Test parallel trends assumption for DID"
    )

    parser.add_argument("data_path", help="Path to data file")
    parser.add_argument("--outcome", "-o", required=True, help="Outcome variable")
    parser.add_argument("--treatment-group", "-g", required=True, help="Treatment group indicator")
    parser.add_argument("--unit", "-u", required=True, help="Unit ID variable")
    parser.add_argument("--time", "-t", required=True, help="Time variable")
    parser.add_argument("--treatment-time", "-tt", type=int, required=True, help="Treatment start time")
    parser.add_argument("--output", "-O", help="Output directory for plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data_path)

    # Run analysis
    analyzer = ParallelTrendsAnalyzer(
        data=data,
        outcome=args.outcome,
        treatment_group=args.treatment_group,
        unit_id=args.unit,
        time_id=args.time,
        treatment_time=args.treatment_time
    )

    results = analyzer.run_full_analysis()

    # Print results
    print("\n" + "=" * 60)
    print("PARALLEL TRENDS ANALYSIS")
    print("=" * 60)

    print(f"\nLinear Trend Test: {'PASSED' if results['linear_trend'].passed else 'FAILED'}")
    print(f"  Statistic: {results['linear_trend'].statistic:.4f}")
    print(f"  P-value: {results['linear_trend'].p_value:.4f}")

    if results.get('event_study'):
        es = results['event_study']
        print(f"\nEvent Study Pre-Trends: {'PASSED' if not es.pre_coefficients_significant else 'CONCERN'}")
        print(f"  Joint F-test p-value: {es.joint_f_pval:.4f}")

    print(f"\nGranger Test: {'PASSED' if results['granger'].passed else 'FAILED'}")
    print(f"  Statistic: {results['granger'].statistic:.4f}")
    print(f"  P-value: {results['granger'].p_value:.4f}")

    print(f"\nOverall Assessment: {results['summary']['overall_assessment']}")
    print(f"  Tests Passed: {results['summary']['tests_passed']}/{results['summary']['total_tests']}")

    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        output_dir = Path(args.output) if args.output else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)

        analyzer.plot_trends(save_path=output_dir / "parallel_trends.png")
        if results.get('event_study'):
            analyzer.plot_event_study(
                results['event_study'],
                save_path=output_dir / "event_study.png"
            )
        print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
