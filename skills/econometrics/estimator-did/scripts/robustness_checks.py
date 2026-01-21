#!/usr/bin/env python3
"""
Robustness Checks Module for Difference-in-Differences.

This module provides comprehensive robustness tests including:
- Placebo tests (timing, group, outcome)
- Goodman-Bacon decomposition
- Wild cluster bootstrap
- Randomization inference
- Specification sensitivity
- Clustering sensitivity

Usage:
    # As CLI tool
    python robustness_checks.py data.csv --outcome y --treatment treated \\
        --unit id --time year --treatment-time 2015 --output results/

    # As module
    from robustness_checks import RobustnessAnalyzer
    analyzer = RobustnessAnalyzer(data, ...)
    results = analyzer.run_all_checks()

Author: Causal ML Skills
Version: 1.0.0
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from did_estimator import (
    estimate_did_2x2,
    estimate_did_panel,
    estimate_did_staggered,
    CausalOutput
)


@dataclass
class RobustnessResult:
    """Result from a robustness check."""
    check_name: str
    baseline_effect: float
    robust_effect: float
    baseline_se: float
    robust_se: float
    difference: float
    interpretation: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_robust(self) -> bool:
        """Check if result is qualitatively similar to baseline."""
        # Same sign and overlapping CIs
        same_sign = np.sign(self.baseline_effect) == np.sign(self.robust_effect)
        baseline_ci = (self.baseline_effect - 1.96 * self.baseline_se,
                      self.baseline_effect + 1.96 * self.baseline_se)
        robust_ci = (self.robust_effect - 1.96 * self.robust_se,
                    self.robust_effect + 1.96 * self.robust_se)
        overlapping = baseline_ci[0] < robust_ci[1] and robust_ci[0] < baseline_ci[1]
        return same_sign and overlapping


class RobustnessAnalyzer:
    """
    Comprehensive robustness analysis for DID estimates.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    outcome : str
        Outcome variable name
    treatment : str
        Treatment indicator (1 if treated at time t)
    unit_id : str
        Unit identifier
    time_id : str
        Time period
    treatment_time : int
        When treatment began
    treatment_group : str, optional
        Ever-treated indicator
    controls : list, optional
        Control variables
    cluster : str, optional
        Clustering variable
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit_id: str,
        time_id: str,
        treatment_time: int,
        treatment_group: str = None,
        controls: List[str] = None,
        cluster: str = None
    ):
        self.data = data.copy()
        self.outcome = outcome
        self.treatment = treatment
        self.unit_id = unit_id
        self.time_id = time_id
        self.treatment_time = treatment_time
        self.controls = controls
        self.cluster = cluster or unit_id

        # Infer treatment group if not provided
        if treatment_group is None:
            self.data['_treatment_group'] = self.data.groupby(unit_id)[treatment].transform('max')
            self.treatment_group = '_treatment_group'
        else:
            self.treatment_group = treatment_group

        # Get baseline estimate
        self.baseline = self._get_baseline()

    def _get_baseline(self) -> CausalOutput:
        """Get baseline DID estimate."""
        return estimate_did_panel(
            data=self.data,
            outcome=self.outcome,
            treatment=self.treatment,
            unit_id=self.unit_id,
            time_id=self.time_id,
            controls=self.controls,
            cluster=self.cluster
        )

    # =========================================================================
    # Placebo Tests
    # =========================================================================

    def placebo_timing_test(
        self,
        placebo_time: int = None,
        n_placebo_tests: int = 1
    ) -> Union[RobustnessResult, List[RobustnessResult]]:
        """
        Run placebo test with fake treatment timing.

        Parameters
        ----------
        placebo_time : int
            Specific placebo treatment time (before actual treatment)
        n_placebo_tests : int
            Number of placebo tests to run (evenly spaced before treatment)

        Returns
        -------
        RobustnessResult or list of RobustnessResult
        """
        periods = sorted(self.data[self.time_id].unique())
        pre_periods = [p for p in periods if p < self.treatment_time]

        if len(pre_periods) < 3:
            return RobustnessResult(
                check_name="Placebo Timing Test",
                baseline_effect=self.baseline.effect,
                robust_effect=np.nan,
                baseline_se=self.baseline.se,
                robust_se=np.nan,
                difference=np.nan,
                interpretation="Insufficient pre-treatment periods for placebo test"
            )

        if placebo_time is not None:
            placebo_times = [placebo_time]
        else:
            # Space placebo tests evenly in pre-period
            step = max(1, len(pre_periods) // (n_placebo_tests + 1))
            placebo_times = pre_periods[step:-1:step][:n_placebo_tests]

        results = []
        for pt in placebo_times:
            # Use only data before actual treatment
            pre_data = self.data[self.data[self.time_id] < self.treatment_time].copy()

            # Create placebo treatment indicator
            pre_data['_placebo_treated'] = (
                (pre_data[self.treatment_group] == 1) &
                (pre_data[self.time_id] >= pt)
            ).astype(int)

            try:
                placebo_result = estimate_did_panel(
                    data=pre_data,
                    outcome=self.outcome,
                    treatment='_placebo_treated',
                    unit_id=self.unit_id,
                    time_id=self.time_id,
                    controls=self.controls,
                    cluster=self.cluster
                )

                is_pass = placebo_result.p_value > 0.1

                result = RobustnessResult(
                    check_name=f"Placebo Timing (t={pt})",
                    baseline_effect=self.baseline.effect,
                    robust_effect=placebo_result.effect,
                    baseline_se=self.baseline.se,
                    robust_se=placebo_result.se,
                    difference=placebo_result.effect,
                    interpretation=f"{'PASSED' if is_pass else 'FAILED'}: "
                                  f"Placebo effect = {placebo_result.effect:.4f} "
                                  f"(p = {placebo_result.p_value:.4f})",
                    details={
                        'placebo_time': pt,
                        'p_value': placebo_result.p_value,
                        'passed': is_pass
                    }
                )
                results.append(result)

            except Exception as e:
                results.append(RobustnessResult(
                    check_name=f"Placebo Timing (t={pt})",
                    baseline_effect=self.baseline.effect,
                    robust_effect=np.nan,
                    baseline_se=self.baseline.se,
                    robust_se=np.nan,
                    difference=np.nan,
                    interpretation=f"Error: {str(e)}"
                ))

        return results if len(results) > 1 else results[0]

    def placebo_outcome_test(
        self,
        placebo_outcomes: List[str]
    ) -> List[RobustnessResult]:
        """
        Test treatment effect on outcomes that should NOT be affected.

        Parameters
        ----------
        placebo_outcomes : list
            Outcome variables that should not be affected by treatment

        Returns
        -------
        list of RobustnessResult
        """
        results = []

        for outcome in placebo_outcomes:
            if outcome not in self.data.columns:
                results.append(RobustnessResult(
                    check_name=f"Placebo Outcome ({outcome})",
                    baseline_effect=self.baseline.effect,
                    robust_effect=np.nan,
                    baseline_se=self.baseline.se,
                    robust_se=np.nan,
                    difference=np.nan,
                    interpretation=f"Column '{outcome}' not found"
                ))
                continue

            try:
                placebo_result = estimate_did_panel(
                    data=self.data,
                    outcome=outcome,
                    treatment=self.treatment,
                    unit_id=self.unit_id,
                    time_id=self.time_id,
                    controls=self.controls,
                    cluster=self.cluster
                )

                is_pass = placebo_result.p_value > 0.1

                results.append(RobustnessResult(
                    check_name=f"Placebo Outcome ({outcome})",
                    baseline_effect=self.baseline.effect,
                    robust_effect=placebo_result.effect,
                    baseline_se=self.baseline.se,
                    robust_se=placebo_result.se,
                    difference=placebo_result.effect,
                    interpretation=f"{'PASSED' if is_pass else 'CONCERN'}: "
                                  f"Effect on {outcome} = {placebo_result.effect:.4f} "
                                  f"(p = {placebo_result.p_value:.4f})",
                    details={
                        'placebo_outcome': outcome,
                        'p_value': placebo_result.p_value,
                        'passed': is_pass
                    }
                ))

            except Exception as e:
                results.append(RobustnessResult(
                    check_name=f"Placebo Outcome ({outcome})",
                    baseline_effect=self.baseline.effect,
                    robust_effect=np.nan,
                    baseline_se=self.baseline.se,
                    robust_se=np.nan,
                    difference=np.nan,
                    interpretation=f"Error: {str(e)}"
                ))

        return results

    # =========================================================================
    # Goodman-Bacon Decomposition
    # =========================================================================

    def bacon_decomposition(self) -> Dict[str, Any]:
        """
        Perform Goodman-Bacon (2021) decomposition of TWFE estimate.

        Returns
        -------
        dict
            Decomposition results with component weights and estimates
        """
        df = self.data.copy()

        # Identify treatment cohorts
        df['_first_treat'] = df.groupby(self.unit_id)[self.treatment].transform(
            lambda x: df.loc[x[x == 1].index, self.time_id].min() if (x == 1).any() else np.inf
        )

        cohorts = df[df['_first_treat'] != np.inf]['_first_treat'].unique()
        cohorts = sorted(cohorts)
        never_treated = df[df['_first_treat'] == np.inf][self.unit_id].unique()
        periods = sorted(df[self.time_id].unique())

        components = {
            'earlier_vs_later': {'weight': 0, 'estimate': 0, 'pairs': []},
            'later_vs_earlier': {'weight': 0, 'estimate': 0, 'pairs': []},
            'treated_vs_never': {'weight': 0, 'estimate': 0, 'pairs': []}
        }

        total_weight = 0

        # Compute 2x2 DIDs for each comparison
        for i, g1 in enumerate(cohorts):
            # Treated vs Never Treated
            if len(never_treated) > 0:
                treated_units = df[df['_first_treat'] == g1][self.unit_id].unique()

                # Pre and post for this cohort
                pre = df[df[self.time_id] < g1]
                post = df[df[self.time_id] >= g1]

                y_t_post = post[post[self.unit_id].isin(treated_units)][self.outcome].mean()
                y_t_pre = pre[pre[self.unit_id].isin(treated_units)][self.outcome].mean()
                y_c_post = post[post[self.unit_id].isin(never_treated)][self.outcome].mean()
                y_c_pre = pre[pre[self.unit_id].isin(never_treated)][self.outcome].mean()

                if all(pd.notna([y_t_post, y_t_pre, y_c_post, y_c_pre])):
                    did = (y_t_post - y_t_pre) - (y_c_post - y_c_pre)
                    # Weight based on group size and variance contribution
                    n_t = len(treated_units)
                    n_c = len(never_treated)
                    n_pre = len([p for p in periods if p < g1])
                    n_post = len([p for p in periods if p >= g1])
                    weight = n_t * n_c * n_pre * n_post / (len(periods) ** 2)

                    components['treated_vs_never']['weight'] += weight
                    components['treated_vs_never']['estimate'] += weight * did
                    components['treated_vs_never']['pairs'].append({
                        'cohort': int(g1),
                        'comparison': 'never_treated',
                        'did': did,
                        'weight': weight
                    })
                    total_weight += weight

            # Earlier vs Later and Later vs Earlier
            for g2 in cohorts:
                if g1 >= g2:
                    continue

                # g1 is earlier treated, g2 is later treated
                units_g1 = df[df['_first_treat'] == g1][self.unit_id].unique()
                units_g2 = df[df['_first_treat'] == g2][self.unit_id].unique()

                # Periods where g1 is treated but g2 is not
                middle_periods = [p for p in periods if g1 <= p < g2]
                if len(middle_periods) == 0:
                    continue

                # Earlier vs Later: g1 treated, g2 as control
                pre_g1 = df[(df[self.time_id] < g1)]
                mid = df[(df[self.time_id] >= g1) & (df[self.time_id] < g2)]

                y_t_mid = mid[mid[self.unit_id].isin(units_g1)][self.outcome].mean()
                y_t_pre = pre_g1[pre_g1[self.unit_id].isin(units_g1)][self.outcome].mean()
                y_c_mid = mid[mid[self.unit_id].isin(units_g2)][self.outcome].mean()
                y_c_pre = pre_g1[pre_g1[self.unit_id].isin(units_g2)][self.outcome].mean()

                if all(pd.notna([y_t_mid, y_t_pre, y_c_mid, y_c_pre])):
                    did_el = (y_t_mid - y_t_pre) - (y_c_mid - y_c_pre)
                    weight_el = len(units_g1) * len(units_g2) * len(middle_periods) / len(periods)

                    components['earlier_vs_later']['weight'] += weight_el
                    components['earlier_vs_later']['estimate'] += weight_el * did_el
                    components['earlier_vs_later']['pairs'].append({
                        'cohort_treated': int(g1),
                        'cohort_control': int(g2),
                        'did': did_el,
                        'weight': weight_el
                    })
                    total_weight += weight_el

                # Later vs Earlier: g2 treated, g1 (already treated) as "control"
                # This is the problematic comparison
                post_g2 = df[df[self.time_id] >= g2]
                mid_for_later = df[(df[self.time_id] >= g1) & (df[self.time_id] < g2)]

                y_t2_post = post_g2[post_g2[self.unit_id].isin(units_g2)][self.outcome].mean()
                y_t2_mid = mid_for_later[mid_for_later[self.unit_id].isin(units_g2)][self.outcome].mean()
                y_c2_post = post_g2[post_g2[self.unit_id].isin(units_g1)][self.outcome].mean()
                y_c2_mid = mid_for_later[mid_for_later[self.unit_id].isin(units_g1)][self.outcome].mean()

                if all(pd.notna([y_t2_post, y_t2_mid, y_c2_post, y_c2_mid])):
                    did_le = (y_t2_post - y_t2_mid) - (y_c2_post - y_c2_mid)
                    weight_le = len(units_g2) * len(units_g1) * len(middle_periods) / len(periods)

                    components['later_vs_earlier']['weight'] += weight_le
                    components['later_vs_earlier']['estimate'] += weight_le * did_le
                    components['later_vs_earlier']['pairs'].append({
                        'cohort_treated': int(g2),
                        'cohort_control': int(g1),
                        'did': did_le,
                        'weight': weight_le,
                        'problematic': True
                    })
                    total_weight += weight_le

        # Normalize
        if total_weight > 0:
            for comp in components.values():
                if comp['weight'] > 0:
                    comp['estimate'] = comp['estimate'] / comp['weight']
                    comp['weight'] = comp['weight'] / total_weight

        # Compute decomposed TWFE estimate
        twfe_decomposed = sum(
            comp['weight'] * comp['estimate']
            for comp in components.values()
        )

        bad_comparison_weight = components['later_vs_earlier']['weight']

        return {
            'twfe_estimate': self.baseline.effect,
            'twfe_decomposed': twfe_decomposed,
            'components': components,
            'bad_comparison_weight': bad_comparison_weight,
            'n_cohorts': len(cohorts),
            'n_never_treated': len(never_treated),
            'interpretation': (
                f"TWFE estimate potentially biased: {bad_comparison_weight*100:.1f}% weight "
                f"on problematic comparisons (later vs earlier treated)"
                if bad_comparison_weight > 0.1
                else "TWFE estimate appears reliable: minimal weight on problematic comparisons"
            )
        }

    # =========================================================================
    # Bootstrap and Inference
    # =========================================================================

    def wild_cluster_bootstrap(
        self,
        n_bootstrap: int = 999,
        weight_type: str = 'rademacher'
    ) -> RobustnessResult:
        """
        Wild cluster bootstrap for inference with few clusters.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap iterations
        weight_type : str
            Type of wild bootstrap weights ('rademacher' or 'mammen')

        Returns
        -------
        RobustnessResult
            Bootstrap inference results
        """
        cluster_var = self.cluster
        clusters = self.data[cluster_var].unique()
        n_clusters = len(clusters)

        # Get baseline residuals
        try:
            from linearmodels.panel import PanelOLS
        except ImportError:
            raise ImportError("linearmodels required for wild bootstrap")

        df = self.data.set_index([self.unit_id, self.time_id])
        y = df[self.outcome]
        X = df[[self.treatment]]
        if self.controls:
            for c in self.controls:
                if c in df.columns:
                    X = pd.concat([X, df[[c]]], axis=1)

        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        model = PanelOLS(y, X, entity_effects=True, time_effects=True)
        results = model.fit()
        residuals = results.resids

        # Map residuals back to clusters
        df_resid = pd.DataFrame({
            'resid': residuals,
            'cluster': self.data.set_index([self.unit_id, self.time_id]).loc[mask.values, cluster_var].values
        })

        # Bootstrap
        boot_effects = []

        for b in range(n_bootstrap):
            # Generate wild weights by cluster
            if weight_type == 'rademacher':
                weights = np.random.choice([-1, 1], size=n_clusters)
            else:  # mammen
                prob = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
                weights = np.where(
                    np.random.random(n_clusters) < prob,
                    -(np.sqrt(5) - 1) / 2,
                    (np.sqrt(5) + 1) / 2
                )

            weight_map = dict(zip(clusters, weights))
            df_resid['weight'] = df_resid['cluster'].map(weight_map)
            wild_resid = df_resid['resid'] * df_resid['weight']

            # Create bootstrapped outcome
            y_boot = results.fitted_values + wild_resid

            # Re-estimate
            model_boot = PanelOLS(y_boot, X, entity_effects=True, time_effects=True)
            results_boot = model_boot.fit()
            boot_effects.append(results_boot.params[self.treatment])

        boot_effects = np.array(boot_effects)
        boot_se = boot_effects.std()
        boot_ci = np.percentile(boot_effects, [2.5, 97.5])

        # Bootstrap p-value (symmetric)
        t_stat = self.baseline.effect / self.baseline.se
        boot_t_stats = (boot_effects - self.baseline.effect) / boot_se
        p_value = 2 * min(
            (boot_t_stats > abs(t_stat)).mean(),
            (boot_t_stats < -abs(t_stat)).mean()
        )

        return RobustnessResult(
            check_name="Wild Cluster Bootstrap",
            baseline_effect=self.baseline.effect,
            robust_effect=boot_effects.mean(),
            baseline_se=self.baseline.se,
            robust_se=boot_se,
            difference=self.baseline.se - boot_se,
            interpretation=(
                f"Bootstrap SE = {boot_se:.4f} vs. Clustered SE = {self.baseline.se:.4f}. "
                f"Bootstrap 95% CI: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]"
            ),
            details={
                'n_bootstrap': n_bootstrap,
                'n_clusters': n_clusters,
                'boot_ci_lower': boot_ci[0],
                'boot_ci_upper': boot_ci[1],
                'boot_p_value': p_value,
                'weight_type': weight_type
            }
        )

    def randomization_inference(
        self,
        n_permutations: int = 1000
    ) -> RobustnessResult:
        """
        Randomization inference by permuting treatment assignment.

        Parameters
        ----------
        n_permutations : int
            Number of random permutations

        Returns
        -------
        RobustnessResult
            Randomization inference results
        """
        # Get unit-level treatment assignments
        unit_treatments = self.data.groupby(self.unit_id)[self.treatment_group].first()
        units = unit_treatments.index.tolist()
        n_treated = (unit_treatments == 1).sum()

        perm_effects = []

        for _ in range(n_permutations):
            # Randomly assign treatment to same number of units
            perm_treated = np.random.choice(units, size=n_treated, replace=False)
            perm_treatment_group = unit_treatments.copy()
            perm_treatment_group[:] = 0
            perm_treatment_group.loc[perm_treated] = 1

            # Create permuted treatment indicator
            df_perm = self.data.copy()
            df_perm['_perm_group'] = df_perm[self.unit_id].map(perm_treatment_group)
            df_perm['_perm_treated'] = (
                (df_perm['_perm_group'] == 1) &
                (df_perm[self.time_id] >= self.treatment_time)
            ).astype(int)

            try:
                result = estimate_did_panel(
                    data=df_perm,
                    outcome=self.outcome,
                    treatment='_perm_treated',
                    unit_id=self.unit_id,
                    time_id=self.time_id,
                    controls=self.controls,
                    cluster=self.cluster
                )
                perm_effects.append(result.effect)
            except:
                continue

        perm_effects = np.array(perm_effects)

        # Two-sided p-value
        p_value = (np.abs(perm_effects) >= np.abs(self.baseline.effect)).mean()

        return RobustnessResult(
            check_name="Randomization Inference",
            baseline_effect=self.baseline.effect,
            robust_effect=perm_effects.mean(),
            baseline_se=self.baseline.se,
            robust_se=perm_effects.std(),
            difference=0,
            interpretation=(
                f"RI p-value = {p_value:.4f}. "
                f"Permutation distribution: mean = {perm_effects.mean():.4f}, "
                f"SD = {perm_effects.std():.4f}"
            ),
            details={
                'n_permutations': len(perm_effects),
                'ri_p_value': p_value,
                'perm_mean': perm_effects.mean(),
                'perm_sd': perm_effects.std(),
                'actual_rank': (perm_effects <= self.baseline.effect).sum()
            }
        )

    # =========================================================================
    # Specification Sensitivity
    # =========================================================================

    def cluster_sensitivity(
        self,
        cluster_vars: List[str] = None
    ) -> List[RobustnessResult]:
        """
        Test sensitivity of standard errors to different clustering levels.

        Parameters
        ----------
        cluster_vars : list
            Variables to cluster on

        Returns
        -------
        list of RobustnessResult
        """
        if cluster_vars is None:
            cluster_vars = [self.unit_id]

        results = []

        for cluster_var in cluster_vars:
            if cluster_var not in self.data.columns:
                continue

            n_clusters = self.data[cluster_var].nunique()

            try:
                result = estimate_did_panel(
                    data=self.data,
                    outcome=self.outcome,
                    treatment=self.treatment,
                    unit_id=self.unit_id,
                    time_id=self.time_id,
                    controls=self.controls,
                    cluster=cluster_var
                )

                results.append(RobustnessResult(
                    check_name=f"Cluster: {cluster_var}",
                    baseline_effect=self.baseline.effect,
                    robust_effect=result.effect,
                    baseline_se=self.baseline.se,
                    robust_se=result.se,
                    difference=result.se - self.baseline.se,
                    interpretation=(
                        f"SE with {cluster_var} clustering ({n_clusters} clusters): "
                        f"{result.se:.4f} (baseline: {self.baseline.se:.4f})"
                    ),
                    details={
                        'cluster_var': cluster_var,
                        'n_clusters': n_clusters,
                        'p_value': result.p_value
                    }
                ))

            except Exception as e:
                results.append(RobustnessResult(
                    check_name=f"Cluster: {cluster_var}",
                    baseline_effect=self.baseline.effect,
                    robust_effect=np.nan,
                    baseline_se=self.baseline.se,
                    robust_se=np.nan,
                    difference=np.nan,
                    interpretation=f"Error: {str(e)}"
                ))

        return results

    def specification_sensitivity(
        self,
        alternative_controls: List[List[str]] = None,
        include_trends: bool = True
    ) -> List[RobustnessResult]:
        """
        Test sensitivity to different model specifications.

        Parameters
        ----------
        alternative_controls : list of lists
            Different sets of control variables
        include_trends : bool
            Test specification with group-specific linear trends

        Returns
        -------
        list of RobustnessResult
        """
        results = []

        # No controls
        try:
            result = estimate_did_panel(
                data=self.data,
                outcome=self.outcome,
                treatment=self.treatment,
                unit_id=self.unit_id,
                time_id=self.time_id,
                controls=None,
                cluster=self.cluster
            )
            results.append(RobustnessResult(
                check_name="No Controls",
                baseline_effect=self.baseline.effect,
                robust_effect=result.effect,
                baseline_se=self.baseline.se,
                robust_se=result.se,
                difference=result.effect - self.baseline.effect,
                interpretation=f"Without controls: {result.effect:.4f} (SE={result.se:.4f})"
            ))
        except Exception as e:
            pass

        # Alternative control sets
        if alternative_controls:
            for i, controls in enumerate(alternative_controls):
                try:
                    result = estimate_did_panel(
                        data=self.data,
                        outcome=self.outcome,
                        treatment=self.treatment,
                        unit_id=self.unit_id,
                        time_id=self.time_id,
                        controls=controls,
                        cluster=self.cluster
                    )
                    results.append(RobustnessResult(
                        check_name=f"Controls Set {i+1}",
                        baseline_effect=self.baseline.effect,
                        robust_effect=result.effect,
                        baseline_se=self.baseline.se,
                        robust_se=result.se,
                        difference=result.effect - self.baseline.effect,
                        interpretation=f"With {controls}: {result.effect:.4f} (SE={result.se:.4f})"
                    ))
                except:
                    pass

        return results

    # =========================================================================
    # Run All Checks
    # =========================================================================

    def run_all_checks(
        self,
        placebo_outcomes: List[str] = None,
        alternative_clusters: List[str] = None,
        n_bootstrap: int = 499
    ) -> Dict[str, Any]:
        """
        Run all robustness checks.

        Returns
        -------
        dict
            Comprehensive robustness check results
        """
        results = {
            'baseline': {
                'effect': self.baseline.effect,
                'se': self.baseline.se,
                'ci_lower': self.baseline.ci_lower,
                'ci_upper': self.baseline.ci_upper,
                'p_value': self.baseline.p_value
            }
        }

        # Placebo timing
        print("Running placebo timing tests...")
        results['placebo_timing'] = self.placebo_timing_test(n_placebo_tests=2)

        # Placebo outcomes
        if placebo_outcomes:
            print("Running placebo outcome tests...")
            results['placebo_outcomes'] = self.placebo_outcome_test(placebo_outcomes)

        # Bacon decomposition
        print("Running Bacon decomposition...")
        results['bacon_decomposition'] = self.bacon_decomposition()

        # Cluster sensitivity
        if alternative_clusters:
            print("Running cluster sensitivity...")
            results['cluster_sensitivity'] = self.cluster_sensitivity(alternative_clusters)

        # Wild bootstrap (if not too many clusters)
        n_clusters = self.data[self.cluster].nunique()
        if n_clusters < 100:
            print("Running wild cluster bootstrap...")
            results['wild_bootstrap'] = self.wild_cluster_bootstrap(n_bootstrap)

        # Specification sensitivity
        print("Running specification sensitivity...")
        results['specification'] = self.specification_sensitivity()

        # Summary
        results['summary'] = self._summarize_results(results)

        return results

    def _summarize_results(self, results: Dict) -> Dict:
        """Summarize robustness check results."""
        checks_passed = 0
        total_checks = 0

        # Placebo timing
        placebo = results.get('placebo_timing', [])
        if isinstance(placebo, list):
            for r in placebo:
                total_checks += 1
                if r.details.get('passed', False):
                    checks_passed += 1
        elif isinstance(placebo, RobustnessResult):
            total_checks += 1
            if placebo.details.get('passed', False):
                checks_passed += 1

        # Bacon decomposition
        bacon = results.get('bacon_decomposition', {})
        if bacon.get('bad_comparison_weight', 1) < 0.2:
            checks_passed += 1
        total_checks += 1

        return {
            'checks_passed': checks_passed,
            'total_checks': total_checks,
            'robustness_score': checks_passed / total_checks if total_checks > 0 else np.nan,
            'overall_assessment': (
                'ROBUST' if checks_passed / total_checks >= 0.7
                else 'SOME CONCERNS' if checks_passed / total_checks >= 0.5
                else 'POTENTIAL ISSUES'
            )
        }


def main():
    """Command line interface for robustness checks."""
    parser = argparse.ArgumentParser(
        description="Run robustness checks for DID analysis"
    )

    parser.add_argument("data_path", help="Path to data file")
    parser.add_argument("--outcome", "-o", required=True, help="Outcome variable")
    parser.add_argument("--treatment", "-t", required=True, help="Treatment indicator")
    parser.add_argument("--unit", "-u", required=True, help="Unit ID variable")
    parser.add_argument("--time", "-T", required=True, help="Time variable")
    parser.add_argument("--treatment-time", "-tt", type=int, required=True,
                       help="Treatment start time")
    parser.add_argument("--controls", nargs="+", help="Control variables")
    parser.add_argument("--cluster", help="Clustering variable")
    parser.add_argument("--output", "-O", help="Output directory")
    parser.add_argument("--n-bootstrap", type=int, default=499, help="Bootstrap iterations")

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data_path)

    # Run analysis
    analyzer = RobustnessAnalyzer(
        data=data,
        outcome=args.outcome,
        treatment=args.treatment,
        unit_id=args.unit,
        time_id=args.time,
        treatment_time=args.treatment_time,
        controls=args.controls,
        cluster=args.cluster
    )

    results = analyzer.run_all_checks(n_bootstrap=args.n_bootstrap)

    # Print summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS CHECK SUMMARY")
    print("=" * 60)

    summary = results['summary']
    print(f"\nOverall Assessment: {summary['overall_assessment']}")
    print(f"Checks Passed: {summary['checks_passed']}/{summary['total_checks']}")
    print(f"Robustness Score: {summary['robustness_score']:.2f}")

    # Save if output specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON (convert non-serializable objects)
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, RobustnessResult):
                return {
                    'check_name': obj.check_name,
                    'baseline_effect': obj.baseline_effect,
                    'robust_effect': obj.robust_effect,
                    'interpretation': obj.interpretation
                }
            return str(obj)

        with open(output_dir / "robustness_results.json", "w") as f:
            json.dump(results, f, indent=2, default=convert)

        print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
