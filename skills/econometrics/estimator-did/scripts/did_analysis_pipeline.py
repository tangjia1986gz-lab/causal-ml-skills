#!/usr/bin/env python3
"""
Complete Difference-in-Differences (DID) Analysis Pipeline

This script demonstrates a full DID workflow including:
- Data preparation with simulated panel data
- Descriptive statistics and visualization
- Pre-trends testing with event study
- TWFE estimation with clustered standard errors
- Robustness checks
- Publication-quality output

Usage:
    python did_analysis_pipeline.py
    python did_analysis_pipeline.py --data your_data.csv --outcome y --treatment treat --unit firm --time year
    python did_analysis_pipeline.py --demo  # Run with simulated data

Dependencies:
    pip install linearmodels statsmodels pandas numpy matplotlib seaborn scipy

Reference:
    Angrist, J. D., & Pischke, J. S. (2009). Mostly Harmless Econometrics.
    Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences with Multiple Time Periods.
"""

import argparse
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')


# =============================================================================
# 1. DATA SIMULATION (for demonstration)
# =============================================================================

def simulate_did_data(
    n_units: int = 500,
    n_periods: int = 10,
    treatment_period: int = 5,
    treatment_effect: float = 2.0,
    treatment_share: float = 0.5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate panel data for DID analysis.

    Parameters
    ----------
    n_units : int
        Number of units (firms, states, etc.)
    n_periods : int
        Number of time periods
    treatment_period : int
        Period when treatment begins (1-indexed)
    treatment_effect : float
        True treatment effect (ATT)
    treatment_share : float
        Share of units that are treated
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Panel data with columns: unit_id, year, outcome, treated, post, treatment
    """
    np.random.seed(seed)

    # Generate unit and time identifiers
    units = np.repeat(np.arange(1, n_units + 1), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)

    # Assign treatment status (time-invariant)
    treated_units = np.random.choice(n_units, size=int(n_units * treatment_share), replace=False) + 1
    treated = np.isin(units, treated_units).astype(int)

    # Post-treatment indicator
    post = (times >= treatment_period).astype(int)

    # Treatment indicator (only 1 for treated units in post period)
    treatment = treated * post

    # Generate outcome with:
    # - Unit fixed effects (alpha_i)
    # - Time fixed effects (gamma_t)
    # - Treatment effect (beta * D_it)
    # - Random error (epsilon_it)

    unit_fe = np.random.normal(0, 2, n_units)
    time_fe = np.random.normal(0, 1, n_periods)

    alpha_i = unit_fe[units - 1]
    gamma_t = time_fe[times - 1]
    epsilon = np.random.normal(0, 1, len(units))

    # Add parallel pre-trends (both groups follow same trend)
    trend = 0.5 * times

    # Outcome = baseline + unit FE + time FE + trend + treatment effect + noise
    outcome = 10 + alpha_i + gamma_t + trend + treatment_effect * treatment + epsilon

    # Create DataFrame
    df = pd.DataFrame({
        'unit_id': units,
        'year': times,
        'outcome': outcome,
        'treated': treated,
        'post': post,
        'treatment': treatment
    })

    # Add some control variables
    df['size'] = np.exp(np.random.normal(0, 0.5, len(df)))
    df['age'] = np.random.poisson(10, len(df))

    return df


# =============================================================================
# 2. DESCRIPTIVE ANALYSIS
# =============================================================================

def descriptive_analysis(df: pd.DataFrame, outcome: str, treatment: str,
                         unit: str, time: str) -> dict:
    """
    Generate descriptive statistics and trends.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    outcome, treatment, unit, time : str
        Column names

    Returns
    -------
    dict
        Dictionary with summary statistics
    """
    results = {}

    # Basic stats
    results['n_obs'] = len(df)
    results['n_units'] = df[unit].nunique()
    results['n_periods'] = df[time].nunique()
    results['n_treated'] = df.groupby(unit)[treatment].max().sum()

    # Summary by treatment status
    summary = df.groupby(treatment)[outcome].agg(['mean', 'std', 'count'])
    results['summary_by_treatment'] = summary

    # Pre-post comparison
    if 'post' in df.columns:
        pre_post = df.groupby(['treated', 'post'])[outcome].mean().unstack()
        results['pre_post_means'] = pre_post

        # Simple DID estimate
        if len(pre_post) == 2:
            did_simple = (pre_post.loc[1, 1] - pre_post.loc[1, 0]) - \
                         (pre_post.loc[0, 1] - pre_post.loc[0, 0])
            results['did_simple'] = did_simple

    return results


def plot_trends(df: pd.DataFrame, outcome: str, time: str,
                group: str = 'treated', treatment_period: int = None,
                save_path: str = None):
    """
    Plot outcome trends by treatment group.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate means by group and time
    trends = df.groupby([time, group])[outcome].mean().unstack()

    # Plot
    for col in trends.columns:
        label = 'Treated' if col == 1 else 'Control'
        ax.plot(trends.index, trends[col], 'o-', label=label, markersize=8)

    # Add treatment line
    if treatment_period:
        ax.axvline(x=treatment_period - 0.5, color='red', linestyle='--',
                   alpha=0.7, label='Treatment')

    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_ylabel(f'Mean {outcome}', fontsize=12)
    ax.set_title('Outcome Trends by Treatment Group', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trends plot to {save_path}")

    plt.show()


# =============================================================================
# 3. TWFE ESTIMATION
# =============================================================================

def run_twfe(df: pd.DataFrame, outcome: str, treatment: str,
             unit: str, time: str, controls: list = None,
             cluster: str = None) -> dict:
    """
    Run Two-Way Fixed Effects regression.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    outcome : str
        Dependent variable column
    treatment : str
        Treatment indicator column
    unit : str
        Unit identifier column
    time : str
        Time period column
    controls : list, optional
        List of control variable columns
    cluster : str, optional
        Cluster variable for standard errors (default: unit)

    Returns
    -------
    dict
        Dictionary with regression results
    """
    from linearmodels.panel import PanelOLS

    # Prepare data
    df_reg = df.copy()
    df_reg = df_reg.set_index([unit, time])

    # Define exogenous variables
    exog_vars = [treatment]
    if controls:
        exog_vars.extend(controls)

    # Run regression
    model = PanelOLS(
        dependent=df_reg[outcome],
        exog=df_reg[exog_vars],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )

    # Fit with clustered standard errors
    result = model.fit(cov_type='clustered', cluster_entity=True)

    # Extract results
    coef = result.params[treatment]
    se = result.std_errors[treatment]
    pval = result.pvalues[treatment]
    ci = result.conf_int().loc[treatment]

    results = {
        'coefficient': coef,
        'std_error': se,
        'p_value': pval,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'n_obs': result.nobs,
        'r_squared': result.rsquared,
        'r_squared_within': result.rsquared_within,
        'full_result': result
    }

    return results


# =============================================================================
# 4. EVENT STUDY
# =============================================================================

def run_event_study(df: pd.DataFrame, outcome: str,
                    unit: str, time: str, treatment_time: str,
                    window: tuple = (-4, 4),
                    reference: int = -1) -> dict:
    """
    Run event study regression for pre-trends testing.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    outcome : str
        Dependent variable
    unit, time : str
        Panel identifiers
    treatment_time : str
        Column indicating first treatment period for each unit
    window : tuple
        (min_lag, max_lead) for event study window
    reference : int
        Reference period (typically -1)

    Returns
    -------
    dict
        Dictionary with event study results
    """
    from linearmodels.panel import PanelOLS

    df_es = df.copy()

    # Create relative time
    df_es['rel_time'] = df_es[time] - df_es[treatment_time]

    # Handle never-treated (set to large negative)
    df_es['rel_time'] = df_es['rel_time'].fillna(-999)

    # Create event time dummies
    min_lag, max_lead = window
    event_dummies = []

    for t in range(min_lag, max_lead + 1):
        if t == reference:
            continue

        col_name = f'D_{t}' if t >= 0 else f'D_m{abs(t)}'

        if t == min_lag:
            df_es[col_name] = (df_es['rel_time'] <= t).astype(int)
        elif t == max_lead:
            df_es[col_name] = (df_es['rel_time'] >= t).astype(int)
        else:
            df_es[col_name] = (df_es['rel_time'] == t).astype(int)

        event_dummies.append(col_name)

    # Set panel index
    df_es = df_es.set_index([unit, time])

    # Run event study regression
    model = PanelOLS(
        dependent=df_es[outcome],
        exog=df_es[event_dummies],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    result = model.fit(cov_type='clustered', cluster_entity=True)

    # Extract coefficients
    coefs = result.params[event_dummies].to_dict()
    ses = result.std_errors[event_dummies].to_dict()
    pvals = result.pvalues[event_dummies].to_dict()

    # Organize results by time
    event_results = []
    for t in range(min_lag, max_lead + 1):
        if t == reference:
            event_results.append({
                'rel_time': t,
                'coefficient': 0,
                'std_error': 0,
                'p_value': np.nan,
                'ci_lower': 0,
                'ci_upper': 0
            })
        else:
            col_name = f'D_{t}' if t >= 0 else f'D_m{abs(t)}'
            ci = result.conf_int().loc[col_name]
            event_results.append({
                'rel_time': t,
                'coefficient': coefs[col_name],
                'std_error': ses[col_name],
                'p_value': pvals[col_name],
                'ci_lower': ci[0],
                'ci_upper': ci[1]
            })

    # Pre-trends test (joint F-test)
    pre_dummies = [d for d in event_dummies if 'm' in d and d != f'D_m{abs(reference)}']
    if pre_dummies:
        try:
            f_test = result.wald_test(' = '.join(pre_dummies) + ' = 0')
            pre_trends_f = f_test.stat
            pre_trends_p = f_test.pval
        except:
            pre_trends_f = np.nan
            pre_trends_p = np.nan
    else:
        pre_trends_f = np.nan
        pre_trends_p = np.nan

    return {
        'event_results': pd.DataFrame(event_results),
        'pre_trends_f': pre_trends_f,
        'pre_trends_p': pre_trends_p,
        'full_result': result
    }


def plot_event_study(event_results: pd.DataFrame,
                     treatment_time: int = 0,
                     save_path: str = None):
    """
    Plot event study coefficients with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    times = event_results['rel_time']
    coefs = event_results['coefficient']
    ci_lower = event_results['ci_lower']
    ci_upper = event_results['ci_upper']

    # Plot coefficients with error bars
    ax.errorbar(times, coefs,
                yerr=[coefs - ci_lower, ci_upper - coefs],
                fmt='o', capsize=4, capthick=2, markersize=8,
                color='blue', ecolor='blue', alpha=0.8)

    # Reference lines
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=treatment_time - 0.5, color='gray', linestyle=':',
               alpha=0.7, label='Treatment')

    # Shade pre-treatment period
    ax.axvspan(times.min() - 0.5, treatment_time - 0.5,
               alpha=0.1, color='gray', label='Pre-treatment')

    ax.set_xlabel('Periods Relative to Treatment', fontsize=12)
    ax.set_ylabel('Coefficient Estimate', fontsize=12)
    ax.set_title('Event Study: Dynamic Treatment Effects', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved event study plot to {save_path}")

    plt.show()


# =============================================================================
# 5. ROBUSTNESS CHECKS
# =============================================================================

def run_robustness_checks(df: pd.DataFrame, outcome: str, treatment: str,
                          unit: str, time: str, controls: list = None) -> dict:
    """
    Run multiple robustness specifications.
    """
    results = {}

    # 1. Basic TWFE (no controls)
    results['basic'] = run_twfe(df, outcome, treatment, unit, time)

    # 2. With controls
    if controls:
        results['with_controls'] = run_twfe(df, outcome, treatment, unit, time, controls)

    # 3. Subset analysis (if we have size variable)
    if 'size' in df.columns:
        median_size = df['size'].median()
        df_small = df[df['size'] <= median_size]
        df_large = df[df['size'] > median_size]

        results['small_firms'] = run_twfe(df_small, outcome, treatment, unit, time)
        results['large_firms'] = run_twfe(df_large, outcome, treatment, unit, time)

    return results


# =============================================================================
# 6. REPORTING
# =============================================================================

def print_results_table(results: dict, title: str = "DID Estimation Results"):
    """
    Print publication-style results table.
    """
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)

    headers = ['Specification', 'Coefficient', 'SE', 'p-value', '95% CI', 'N']
    print(f"{'Specification':<20} {'Coef':>10} {'SE':>10} {'p-val':>10} {'95% CI':>20} {'N':>8}")
    print("-" * 70)

    for name, res in results.items():
        coef = res['coefficient']
        se = res['std_error']
        pval = res['p_value']
        ci = f"[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]"
        n = res['n_obs']

        # Stars for significance
        stars = ""
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"

        print(f"{name:<20} {coef:>9.4f}{stars} {se:>10.4f} {pval:>10.4f} {ci:>20} {n:>8}")

    print("-" * 70)
    print("Notes: *** p<0.01, ** p<0.05, * p<0.1")
    print("       Standard errors clustered at unit level")
    print("=" * 70)


def generate_latex_table(results: dict, save_path: str = None) -> str:
    """
    Generate LaTeX table for publication.
    """
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Difference-in-Differences Estimation Results}
\label{tab:did_results}
\begin{tabular}{lcccc}
\toprule
& (1) & (2) & (3) & (4) \\
& Basic & Controls & Small Firms & Large Firms \\
\midrule
"""

    specs = ['basic', 'with_controls', 'small_firms', 'large_firms']

    # Treatment coefficient row
    row = "Treatment Effect "
    for spec in specs:
        if spec in results:
            coef = results[spec]['coefficient']
            pval = results[spec]['p_value']
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            row += f"& {coef:.4f}{stars} "
        else:
            row += "& - "
    latex += row + r"\\" + "\n"

    # Standard error row
    row = " "
    for spec in specs:
        if spec in results:
            se = results[spec]['std_error']
            row += f"& ({se:.4f}) "
        else:
            row += "& "
    latex += row + r"\\" + "\n"

    latex += r"\midrule" + "\n"

    # N row
    row = "Observations "
    for spec in specs:
        if spec in results:
            n = results[spec]['n_obs']
            row += f"& {n:,} "
        else:
            row += "& - "
    latex += row + r"\\" + "\n"

    # R-squared row
    row = r"$R^2$ (within) "
    for spec in specs:
        if spec in results:
            r2 = results[spec].get('r_squared_within', results[spec]['r_squared'])
            row += f"& {r2:.4f} "
        else:
            row += "& - "
    latex += row + r"\\" + "\n"

    latex += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Standard errors clustered at unit level in parentheses.
\item *** p<0.01, ** p<0.05, * p<0.1
\end{tablenotes}
\end{table}
"""

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"Saved LaTeX table to {save_path}")

    return latex


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def run_full_analysis(df: pd.DataFrame = None,
                      outcome: str = 'outcome',
                      treatment: str = 'treatment',
                      unit: str = 'unit_id',
                      time: str = 'year',
                      treatment_time: str = None,
                      treatment_period: int = None,
                      controls: list = None,
                      output_dir: str = None) -> dict:
    """
    Run complete DID analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data (if None, uses simulated data)
    outcome, treatment, unit, time : str
        Column names for analysis
    treatment_time : str
        Column with first treatment period for each unit (for event study)
    treatment_period : int
        Period when treatment starts (for trends plot)
    controls : list
        Control variable columns
    output_dir : str
        Directory to save outputs

    Returns
    -------
    dict
        Dictionary with all analysis results
    """
    all_results = {}

    # Setup output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    # Use simulated data if none provided
    if df is None:
        print("Using simulated data for demonstration...")
        df = simulate_did_data()
        treatment_time = 'treatment_year'
        treatment_period = 5

        # Add treatment year for event study
        df['treatment_year'] = df['treated'] * 5 + (1 - df['treated']) * np.nan

    print("\n" + "=" * 70)
    print("DIFFERENCE-IN-DIFFERENCES ANALYSIS".center(70))
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # 1. Descriptive Analysis
    print("\n[1/5] Descriptive Analysis...")
    desc = descriptive_analysis(df, outcome, treatment, unit, time)
    all_results['descriptive'] = desc

    print(f"  - Observations: {desc['n_obs']:,}")
    print(f"  - Units: {desc['n_units']:,}")
    print(f"  - Periods: {desc['n_periods']}")
    print(f"  - Treated units: {desc['n_treated']}")
    if 'did_simple' in desc:
        print(f"  - Simple DID estimate: {desc['did_simple']:.4f}")

    # 2. Trends Plot
    print("\n[2/5] Plotting Trends...")
    save_path = str(output_dir / 'trends.png') if output_dir else None
    plot_trends(df, outcome, time, 'treated', treatment_period, save_path)

    # 3. Event Study (Pre-Trends)
    print("\n[3/5] Event Study (Pre-Trends Test)...")
    if treatment_time and treatment_time in df.columns:
        es_results = run_event_study(df, outcome, unit, time, treatment_time)
        all_results['event_study'] = es_results

        print(f"  - Pre-trends F-statistic: {es_results['pre_trends_f']:.4f}")
        print(f"  - Pre-trends p-value: {es_results['pre_trends_p']:.4f}")

        if es_results['pre_trends_p'] > 0.1:
            print("  - PASS: No evidence against parallel trends (p > 0.1)")
        else:
            print("  - WARNING: Potential parallel trends violation")

        save_path = str(output_dir / 'event_study.png') if output_dir else None
        plot_event_study(es_results['event_results'], 0, save_path)
    else:
        print("  - Skipped (treatment_time column not specified)")

    # 4. TWFE Estimation
    print("\n[4/5] TWFE Estimation...")
    twfe_results = run_twfe(df, outcome, treatment, unit, time, controls)
    all_results['twfe'] = twfe_results

    print(f"  - Treatment Effect: {twfe_results['coefficient']:.4f}")
    print(f"  - Standard Error: {twfe_results['std_error']:.4f}")
    print(f"  - 95% CI: [{twfe_results['ci_lower']:.4f}, {twfe_results['ci_upper']:.4f}]")
    print(f"  - P-value: {twfe_results['p_value']:.4f}")

    # 5. Robustness Checks
    print("\n[5/5] Robustness Checks...")
    robust_results = run_robustness_checks(df, outcome, treatment, unit, time, controls)
    all_results['robustness'] = robust_results

    # Print summary table
    print_results_table(robust_results, "Robustness Analysis")

    # Generate LaTeX table
    if output_dir:
        latex_path = str(output_dir / 'did_table.tex')
        generate_latex_table(robust_results, latex_path)

    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE".center(70))
    print("=" * 70)

    if output_dir:
        print(f"\nOutputs saved to: {output_dir}")
        print("  - trends.png")
        print("  - event_study.png")
        print("  - did_table.tex")

    return all_results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Difference-in-Differences analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python did_analysis_pipeline.py --demo
    python did_analysis_pipeline.py --data data.csv --outcome y --treatment treat --unit firm --time year
    python did_analysis_pipeline.py --data data.csv --outcome y --treatment treat --unit firm --time year --controls size,age
        """
    )

    parser.add_argument('--demo', action='store_true',
                        help='Run with simulated data')
    parser.add_argument('--data', '-d', type=str,
                        help='Path to CSV data file')
    parser.add_argument('--outcome', '-y', type=str, default='outcome',
                        help='Outcome variable column')
    parser.add_argument('--treatment', '-t', type=str, default='treatment',
                        help='Treatment indicator column')
    parser.add_argument('--unit', '-u', type=str, default='unit_id',
                        help='Unit identifier column')
    parser.add_argument('--time', type=str, default='year',
                        help='Time period column')
    parser.add_argument('--treatment-time', type=str,
                        help='First treatment period column (for event study)')
    parser.add_argument('--treatment-period', type=int,
                        help='Treatment start period (for trends plot)')
    parser.add_argument('--controls', type=str,
                        help='Comma-separated list of control variables')
    parser.add_argument('--output', '-o', type=str, default='./did_results',
                        help='Output directory')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.demo or args.data is None:
        # Run with simulated data
        results = run_full_analysis(output_dir=args.output)
    else:
        # Load data
        df = pd.read_csv(args.data)

        # Parse controls
        controls = args.controls.split(',') if args.controls else None

        # Run analysis
        results = run_full_analysis(
            df=df,
            outcome=args.outcome,
            treatment=args.treatment,
            unit=args.unit,
            time=args.time,
            treatment_time=args.treatment_time,
            treatment_period=args.treatment_period,
            controls=controls,
            output_dir=args.output
        )
