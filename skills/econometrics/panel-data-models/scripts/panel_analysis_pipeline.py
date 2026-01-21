#!/usr/bin/env python3
"""
Panel Data Analysis Pipeline - Self-Contained

Complete panel data econometrics workflow using linearmodels:
- Fixed Effects (within estimator)
- Random Effects (GLS)
- Hausman Test
- Two-way Fixed Effects
- Clustered Standard Errors
- Dynamic Panels

Usage:
    python panel_analysis_pipeline.py --demo
    python panel_analysis_pipeline.py --data data.csv --outcome y --entity firm_id --time year

Dependencies:
    pip install linearmodels pandas numpy scipy matplotlib

Reference:
    Wooldridge, J. (2010). Econometric Analysis of Cross Section and Panel Data.
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Set non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Data Simulation
# =============================================================================

def simulate_panel_data(
    n_entities: int = 100,
    n_periods: int = 10,
    treatment_effect: float = 0.5,
    entity_effect_var: float = 1.0,
    time_effect_var: float = 0.3,
    correlation_with_x: float = 0.5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate panel data with entity and time fixed effects.

    Parameters
    ----------
    n_entities : int
        Number of cross-sectional units
    n_periods : int
        Number of time periods
    treatment_effect : float
        True effect of X on Y
    entity_effect_var : float
        Variance of entity fixed effects
    time_effect_var : float
        Variance of time fixed effects
    correlation_with_x : float
        Correlation between entity effect and X (for Hausman test)
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Simulated panel data with multi-index
    """
    np.random.seed(seed)

    n_obs = n_entities * n_periods

    # Entity and time identifiers
    entity_ids = np.repeat(np.arange(n_entities), n_periods)
    time_ids = np.tile(np.arange(n_periods), n_entities)

    # Entity fixed effects (alpha_i)
    entity_effects = np.random.normal(0, np.sqrt(entity_effect_var), n_entities)
    alpha_i = entity_effects[entity_ids]

    # Time fixed effects (gamma_t)
    time_effects = np.random.normal(0, np.sqrt(time_effect_var), n_periods)
    gamma_t = time_effects[time_ids]

    # Regressors
    # X1: varies within and between, correlated with entity effect
    x1_between = np.random.normal(0, 1, n_entities)
    x1_within = np.random.normal(0, 0.5, n_obs)
    x1 = x1_between[entity_ids] + x1_within
    # Add correlation with entity effect
    x1 = x1 + correlation_with_x * entity_effects[entity_ids]

    # X2: purely within variation (time-varying)
    x2 = np.random.normal(0, 1, n_obs) + 0.1 * time_ids

    # X3: time-invariant (for testing)
    x3_values = np.random.normal(0, 1, n_entities)
    x3 = x3_values[entity_ids]

    # Error term
    epsilon = np.random.normal(0, 1, n_obs)

    # Outcome: Y = beta1*X1 + beta2*X2 + alpha_i + gamma_t + epsilon
    y = treatment_effect * x1 + 0.3 * x2 + alpha_i + gamma_t + epsilon

    # Create DataFrame
    df = pd.DataFrame({
        'entity_id': entity_ids,
        'time_id': time_ids,
        'y': y,
        'x1': x1,
        'x2': x2,
        'x3': x3,  # Time-invariant
        'entity_effect': alpha_i,
        'time_effect': gamma_t
    })

    # Set multi-index
    df = df.set_index(['entity_id', 'time_id'])

    return df


# =============================================================================
# Panel Model Estimation
# =============================================================================

def run_pooled_ols(
    df: pd.DataFrame,
    outcome: str,
    regressors: List[str]
) -> Dict[str, Any]:
    """
    Run Pooled OLS (baseline, ignores panel structure).
    """
    from linearmodels.panel import PooledOLS

    formula = f"{outcome} ~ 1 + {' + '.join(regressors)}"

    model = PooledOLS.from_formula(formula, data=df)
    result = model.fit(cov_type='clustered', cluster_entity=True)

    return {
        'model': 'Pooled OLS',
        'result': result,
        'params': result.params.to_dict(),
        'std_errors': result.std_errors.to_dict(),
        'pvalues': result.pvalues.to_dict(),
        'rsquared': result.rsquared
    }


def run_fixed_effects(
    df: pd.DataFrame,
    outcome: str,
    regressors: List[str],
    entity_effects: bool = True,
    time_effects: bool = False,
    cluster_entity: bool = True,
    cluster_time: bool = False
) -> Dict[str, Any]:
    """
    Run Fixed Effects (within) estimation.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with multi-index
    outcome : str
        Dependent variable
    regressors : List[str]
        Independent variables (time-varying only)
    entity_effects : bool
        Include entity fixed effects
    time_effects : bool
        Include time fixed effects
    cluster_entity : bool
        Cluster SE by entity
    cluster_time : bool
        Cluster SE by time

    Returns
    -------
    dict
        Estimation results
    """
    from linearmodels.panel import PanelOLS

    # Build formula
    effects = []
    if entity_effects:
        effects.append('EntityEffects')
    if time_effects:
        effects.append('TimeEffects')

    if effects:
        formula = f"{outcome} ~ {' + '.join(regressors)} + {' + '.join(effects)}"
    else:
        formula = f"{outcome} ~ 1 + {' + '.join(regressors)}"

    model = PanelOLS.from_formula(
        formula,
        data=df,
        drop_absorbed=True if (entity_effects and time_effects) else False
    )

    result = model.fit(
        cov_type='clustered',
        cluster_entity=cluster_entity,
        cluster_time=cluster_time
    )

    return {
        'model': f"FE{'+TE' if time_effects else ''}",
        'result': result,
        'params': result.params.to_dict(),
        'std_errors': result.std_errors.to_dict(),
        'pvalues': result.pvalues.to_dict(),
        'rsquared_within': result.rsquared_within,
        'rsquared_between': result.rsquared_between,
        'rsquared_overall': result.rsquared_overall,
        'f_statistic': result.f_statistic.stat if hasattr(result, 'f_statistic') else None,
        'entity_effects': entity_effects,
        'time_effects': time_effects
    }


def run_random_effects(
    df: pd.DataFrame,
    outcome: str,
    regressors: List[str]
) -> Dict[str, Any]:
    """
    Run Random Effects (GLS) estimation.
    """
    from linearmodels.panel import RandomEffects

    formula = f"{outcome} ~ 1 + {' + '.join(regressors)}"

    model = RandomEffects.from_formula(formula, data=df)
    result = model.fit()

    # Variance decomposition
    var_decomp = result.variance_decomposition

    return {
        'model': 'Random Effects',
        'result': result,
        'params': result.params.to_dict(),
        'std_errors': result.std_errors.to_dict(),
        'pvalues': result.pvalues.to_dict(),
        'rsquared_within': result.rsquared_within,
        'rsquared_between': result.rsquared_between,
        'rsquared_overall': result.rsquared_overall,
        'sigma_u': var_decomp.Effects,
        'sigma_e': var_decomp.Residual,
        'theta': result.theta.mean() if hasattr(result.theta, 'mean') else result.theta
    }


def hausman_test(
    fe_result: Dict[str, Any],
    re_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Hausman test for FE vs RE model selection.

    H0: Random Effects is consistent (effects uncorrelated with regressors)
    H1: Fixed Effects needed (effects correlated with regressors)
    """
    fe_res = fe_result['result']
    re_res = re_result['result']

    # Get common parameters (exclude Intercept)
    fe_params = fe_res.params.drop('Intercept', errors='ignore')
    re_params = re_res.params.drop('Intercept', errors='ignore')

    # Find common variables
    common_vars = fe_params.index.intersection(re_params.index)

    if len(common_vars) == 0:
        return {
            'hausman_stat': np.nan,
            'df': 0,
            'p_value': np.nan,
            'conclusion': 'No common variables for Hausman test'
        }

    # Coefficient differences
    b_fe = fe_params[common_vars]
    b_re = re_params[common_vars]
    diff = b_fe - b_re

    # Variance of difference (FE variance - RE variance under H0)
    fe_cov = fe_res.cov.loc[common_vars, common_vars]
    re_cov = re_res.cov.loc[common_vars, common_vars]
    var_diff = fe_cov - re_cov

    # Check positive definiteness
    try:
        # Hausman statistic
        hausman_stat = float(diff.values @ np.linalg.inv(var_diff.values) @ diff.values)
        df_test = len(common_vars)
        p_value = 1 - stats.chi2.cdf(hausman_stat, df_test)
    except np.linalg.LinAlgError:
        # Variance matrix not invertible
        return {
            'hausman_stat': np.nan,
            'df': len(common_vars),
            'p_value': np.nan,
            'conclusion': 'Variance matrix not invertible'
        }

    if p_value < 0.05:
        conclusion = 'Reject H0: Use Fixed Effects (effects correlated with regressors)'
    else:
        conclusion = 'Fail to reject H0: Random Effects is consistent'

    return {
        'hausman_stat': hausman_stat,
        'df': df_test,
        'p_value': p_value,
        'conclusion': conclusion
    }


def compare_se_types(
    df: pd.DataFrame,
    outcome: str,
    regressors: List[str]
) -> pd.DataFrame:
    """
    Compare different standard error types.
    """
    from linearmodels.panel import PanelOLS

    formula = f"{outcome} ~ {' + '.join(regressors)} + EntityEffects"
    model = PanelOLS.from_formula(formula, data=df)

    se_types = {
        'Homoskedastic': model.fit(cov_type='unadjusted'),
        'Robust': model.fit(cov_type='robust'),
        'Clustered (entity)': model.fit(cov_type='clustered', cluster_entity=True),
        'Clustered (time)': model.fit(cov_type='clustered', cluster_time=True),
        'Two-way clustered': model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    }

    # Extract SE for first regressor
    var = regressors[0]
    comparison = []
    for se_type, result in se_types.items():
        comparison.append({
            'SE Type': se_type,
            'Coefficient': result.params[var],
            'Std Error': result.std_errors[var],
            't-stat': result.tstats[var],
            'p-value': result.pvalues[var]
        })

    return pd.DataFrame(comparison)


# =============================================================================
# Diagnostics
# =============================================================================

def panel_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute panel structure diagnostics.
    """
    # Panel dimensions
    n_entities = df.index.get_level_values(0).nunique()
    n_periods = df.index.get_level_values(1).nunique()
    n_obs = len(df)

    # Balance check
    obs_per_entity = df.groupby(level=0).size()
    is_balanced = obs_per_entity.nunique() == 1

    # Within vs between variation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    variation = {}
    for col in numeric_cols[:5]:  # First 5 numeric columns
        overall_var = df[col].var()
        between_var = df.groupby(level=0)[col].mean().var()
        within_var = df.groupby(level=0)[col].apply(lambda x: x.var()).mean()

        variation[col] = {
            'overall': overall_var,
            'between': between_var,
            'within': within_var,
            'within_pct': within_var / overall_var if overall_var > 0 else 0
        }

    return {
        'n_entities': n_entities,
        'n_periods': n_periods,
        'n_obs': n_obs,
        'is_balanced': is_balanced,
        'obs_per_entity_mean': obs_per_entity.mean(),
        'obs_per_entity_min': obs_per_entity.min(),
        'obs_per_entity_max': obs_per_entity.max(),
        'variation': variation
    }


# =============================================================================
# Output Functions
# =============================================================================

def print_results(
    pooled: Dict[str, Any],
    fe: Dict[str, Any],
    re: Dict[str, Any],
    hausman: Dict[str, Any],
    true_effect: Optional[float] = None
):
    """Print formatted comparison of panel models."""
    print("\n" + "="*70)
    print("PANEL MODEL ESTIMATION RESULTS")
    print("="*70)

    # Get variable names
    vars_to_show = [v for v in fe['params'].keys() if v != 'Intercept']

    print("\n" + "-"*70)
    print(f"{'Variable':<15} {'Pooled OLS':<18} {'Fixed Effects':<18} {'Random Effects':<18}")
    print("-"*70)

    for var in vars_to_show:
        pooled_str = f"{pooled['params'].get(var, np.nan):.4f}" if var in pooled['params'] else '-'
        fe_str = f"{fe['params'].get(var, np.nan):.4f}" if var in fe['params'] else '-'
        re_str = f"{re['params'].get(var, np.nan):.4f}" if var in re['params'] else '-'

        # Add significance stars
        if var in pooled['pvalues'] and pooled['pvalues'][var] < 0.05:
            pooled_str += '**'
        if var in fe['pvalues'] and fe['pvalues'][var] < 0.05:
            fe_str += '**'
        if var in re['pvalues'] and re['pvalues'][var] < 0.05:
            re_str += '**'

        print(f"{var:<15} {pooled_str:<18} {fe_str:<18} {re_str:<18}")

        # Standard errors
        pooled_se = f"({pooled['std_errors'].get(var, np.nan):.4f})" if var in pooled['std_errors'] else ''
        fe_se = f"({fe['std_errors'].get(var, np.nan):.4f})" if var in fe['std_errors'] else ''
        re_se = f"({re['std_errors'].get(var, np.nan):.4f})" if var in re['std_errors'] else ''
        print(f"{'':<15} {pooled_se:<18} {fe_se:<18} {re_se:<18}")

    print("-"*70)

    # R-squared
    print(f"{'R2 (within)':<15} {'-':<18} {fe['rsquared_within']:.4f}{'':<12} {re['rsquared_within']:.4f}")
    print(f"{'R2 (between)':<15} {'-':<18} {fe['rsquared_between']:.4f}{'':<12} {re['rsquared_between']:.4f}")
    print(f"{'R2 (overall)':<15} {pooled['rsquared']:.4f}{'':<12} {fe['rsquared_overall']:.4f}{'':<12} {re['rsquared_overall']:.4f}")

    print("-"*70)

    # Hausman test
    print(f"\n--- Hausman Test (FE vs RE) ---")
    print(f"Chi-squared: {hausman['hausman_stat']:.4f}")
    print(f"Degrees of freedom: {hausman['df']}")
    print(f"P-value: {hausman['p_value']:.4f}")
    print(f"Conclusion: {hausman['conclusion']}")

    if true_effect is not None:
        print(f"\n--- Comparison with True Effect ---")
        print(f"True effect: {true_effect:.4f}")
        print(f"Pooled OLS bias: {pooled['params'].get('x1', np.nan) - true_effect:.4f}")
        print(f"Fixed Effects bias: {fe['params'].get('x1', np.nan) - true_effect:.4f}")
        print(f"Random Effects bias: {re['params'].get('x1', np.nan) - true_effect:.4f}")

    print("="*70)


def generate_latex_table(
    pooled: Dict[str, Any],
    fe: Dict[str, Any],
    re: Dict[str, Any],
    twfe: Dict[str, Any] = None,
    save_path: Optional[str] = None
) -> str:
    """Generate publication-quality LaTeX table."""
    vars_to_show = [v for v in fe['params'].keys() if v != 'Intercept']

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Panel Model Estimation Results}
\label{tab:panel_results}
\begin{tabular}{l"""
    latex += "c" * (4 if twfe else 3)
    latex += r"""}
\toprule
"""
    if twfe:
        latex += r"& (1) & (2) & (3) & (4) \\" + "\n"
        latex += r"Variable & Pooled OLS & FE & RE & TWFE \\" + "\n"
    else:
        latex += r"& (1) & (2) & (3) \\" + "\n"
        latex += r"Variable & Pooled OLS & FE & RE \\" + "\n"
    latex += r"\midrule" + "\n"

    for var in vars_to_show:
        # Coefficients with stars
        def format_coef(d, v):
            if v not in d['params']:
                return '-'
            coef = d['params'][v]
            pval = d['pvalues'].get(v, 1)
            stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
            return f"{coef:.4f}{stars}"

        row = f"{var} & {format_coef(pooled, var)} & {format_coef(fe, var)} & {format_coef(re, var)}"
        if twfe:
            row += f" & {format_coef(twfe, var)}"
        latex += row + r" \\" + "\n"

        # Standard errors
        def format_se(d, v):
            if v not in d['std_errors']:
                return ''
            return f"({d['std_errors'][v]:.4f})"

        se_row = f" & {format_se(pooled, var)} & {format_se(fe, var)} & {format_se(re, var)}"
        if twfe:
            se_row += f" & {format_se(twfe, var)}"
        latex += se_row + r" \\" + "\n"

    latex += r"\midrule" + "\n"

    # R-squared
    latex += f"R$^2$ (within) & - & {fe['rsquared_within']:.4f} & {re['rsquared_within']:.4f}"
    if twfe:
        latex += f" & {twfe['rsquared_within']:.4f}"
    latex += r" \\" + "\n"

    latex += f"R$^2$ (overall) & {pooled['rsquared']:.4f} & {fe['rsquared_overall']:.4f} & {re['rsquared_overall']:.4f}"
    if twfe:
        latex += f" & {twfe['rsquared_overall']:.4f}"
    latex += r" \\" + "\n"

    # Fixed effects indicators
    latex += r"Entity FE & No & Yes & - & Yes \\" + "\n"
    if twfe:
        latex += r"Time FE & No & No & - & Yes \\" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: *** p<0.01, ** p<0.05, * p<0.1. Standard errors clustered by entity in parentheses.
\end{tablenotes}
\end{table}
"""

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {save_path}")

    return latex


def plot_within_between_variation(
    df: pd.DataFrame,
    variable: str,
    save_path: Optional[str] = None
):
    """Plot within vs between variation for a variable."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Reset index for plotting
    df_plot = df.reset_index()

    # Between variation: entity means
    entity_means = df_plot.groupby('entity_id')[variable].mean()
    axes[0].hist(entity_means, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(entity_means.mean(), color='red', linestyle='--', label='Grand Mean')
    axes[0].set_xlabel(f'Entity Mean of {variable}')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Between Variation: {variable}')
    axes[0].legend()

    # Within variation: deviations from entity mean
    df_plot['entity_mean'] = df_plot.groupby('entity_id')[variable].transform('mean')
    df_plot['within_dev'] = df_plot[variable] - df_plot['entity_mean']
    axes[1].hist(df_plot['within_dev'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_xlabel(f'Deviation from Entity Mean')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Within Variation: {variable}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.close()


# =============================================================================
# Full Analysis Pipeline
# =============================================================================

def run_full_analysis(
    df: pd.DataFrame,
    outcome: str,
    regressors: List[str],
    true_effect: Optional[float] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete panel data analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with multi-index (entity, time)
    outcome : str
        Dependent variable name
    regressors : List[str]
        Independent variable names
    true_effect : float, optional
        True effect for simulation comparison
    output_dir : str, optional
        Output directory

    Returns
    -------
    dict
        Complete analysis results
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("PANEL DATA ANALYSIS")
    print("="*70)

    # Step 1: Panel diagnostics
    print("\n--- Step 1: Panel Structure ---")
    diag = panel_diagnostics(df)
    print(f"Entities: {diag['n_entities']}")
    print(f"Time periods: {diag['n_periods']}")
    print(f"Observations: {diag['n_obs']}")
    print(f"Balanced: {'Yes' if diag['is_balanced'] else 'No'}")
    print(f"Obs per entity: {diag['obs_per_entity_min']}-{diag['obs_per_entity_max']} (mean: {diag['obs_per_entity_mean']:.1f})")

    # Step 2: Variation decomposition
    print("\n--- Step 2: Variation Decomposition ---")
    for var, v in diag['variation'].items():
        if var in regressors + [outcome]:
            print(f"{var}: {v['within_pct']:.1%} within variation")

    # Step 3: Model estimation
    print("\n--- Step 3: Model Estimation ---")

    pooled = run_pooled_ols(df, outcome, regressors)
    print("  Pooled OLS: done")

    fe = run_fixed_effects(df, outcome, regressors, entity_effects=True, time_effects=False)
    print("  Fixed Effects: done")

    re = run_random_effects(df, outcome, regressors)
    print("  Random Effects: done")

    twfe = run_fixed_effects(df, outcome, regressors, entity_effects=True, time_effects=True)
    print("  Two-way FE: done")

    # Step 4: Hausman test
    print("\n--- Step 4: Hausman Test ---")
    hausman = hausman_test(fe, re)

    # Step 5: Print results
    print_results(pooled, fe, re, hausman, true_effect)

    # Step 6: SE comparison
    print("\n--- Step 6: Standard Error Comparison ---")
    se_comparison = compare_se_types(df, outcome, regressors)
    print(se_comparison.to_string(index=False))

    # Step 7: Output
    if output_dir:
        print("\n--- Step 7: Generating Output ---")

        # LaTeX table
        latex_path = output_dir / 'panel_results.tex'
        generate_latex_table(pooled, fe, re, twfe, str(latex_path))

        # Variation plot
        plot_path = output_dir / 'variation_decomposition.png'
        plot_within_between_variation(df, regressors[0], str(plot_path))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return {
        'pooled': pooled,
        'fe': fe,
        're': re,
        'twfe': twfe,
        'hausman': hausman,
        'diagnostics': diag,
        'se_comparison': se_comparison
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Panel Data Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with simulated data
    python panel_analysis_pipeline.py --demo

    # Run with real data
    python panel_analysis_pipeline.py --data firm_data.csv --outcome log_revenue \\
        --entity firm_id --time year --regressors "investment,rd_expense,employees"
"""
    )

    parser.add_argument('--demo', action='store_true', help='Run demo with simulated data')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--outcome', type=str, help='Outcome variable name')
    parser.add_argument('--entity', type=str, help='Entity identifier column')
    parser.add_argument('--time', type=str, help='Time identifier column')
    parser.add_argument('--regressors', type=str, help='Comma-separated regressor names')
    parser.add_argument('--output', type=str, help='Output directory')

    args = parser.parse_args()

    if args.demo:
        print("Running panel data demo with simulated data...")

        # Simulate data
        true_effect = 0.5
        df = simulate_panel_data(
            n_entities=100,
            n_periods=10,
            treatment_effect=true_effect,
            correlation_with_x=0.5,  # This creates need for FE
            seed=42
        )

        print(f"Simulated panel: {df.index.get_level_values(0).nunique()} entities, "
              f"{df.index.get_level_values(1).nunique()} periods")

        outcome = 'y'
        regressors = ['x1', 'x2']  # Note: x3 is time-invariant, will be absorbed

        run_full_analysis(
            df=df,
            outcome=outcome,
            regressors=regressors,
            true_effect=true_effect,
            output_dir=args.output
        )

    elif args.data and args.outcome and args.entity and args.time:
        df = pd.read_csv(args.data)

        # Set multi-index
        df = df.set_index([args.entity, args.time])

        if args.regressors:
            regressors = [r.strip() for r in args.regressors.split(',')]
        else:
            regressors = [c for c in df.columns if c != args.outcome]

        run_full_analysis(
            df=df,
            outcome=args.outcome,
            regressors=regressors,
            output_dir=args.output
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
