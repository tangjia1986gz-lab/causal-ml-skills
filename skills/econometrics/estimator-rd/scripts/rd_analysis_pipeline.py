#!/usr/bin/env python3
"""
Complete Regression Discontinuity (RD) Analysis Pipeline

This script provides a comprehensive workflow for RD analysis including:
- Sharp and Fuzzy RD estimation
- Manipulation testing (McCrary/density)
- Covariate balance checks
- Bandwidth selection and sensitivity
- Publication-ready visualization and tables

Dependencies:
    pip install rdrobust rddensity numpy pandas matplotlib statsmodels scipy

Usage:
    python rd_analysis_pipeline.py --demo                    # Run with simulated data
    python rd_analysis_pipeline.py --data input.csv          # Run with CSV file
    python rd_analysis_pipeline.py --data input.csv --fuzzy  # Fuzzy RD
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import argparse
import os
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA SIMULATION
# ============================================================================

def simulate_rd_data(n: int = 1000,
                     effect: float = 2.0,
                     cutoff: float = 0.0,
                     fuzzy: bool = False,
                     manipulation: bool = False,
                     seed: int = 42) -> pd.DataFrame:
    """
    Generate simulated RD data for demonstration.

    Parameters
    ----------
    n : int
        Sample size
    effect : float
        True treatment effect at cutoff
    cutoff : float
        Cutoff value for running variable
    fuzzy : bool
        If True, generate fuzzy RD (treatment probability jumps)
    manipulation : bool
        If True, add bunching at cutoff
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Simulated RD data with columns: x, treatment, y, and covariates
    """
    np.random.seed(seed)

    # Running variable
    if manipulation:
        # Add bunching just above cutoff (manipulation)
        x_below = np.random.uniform(cutoff - 1, cutoff, int(n * 0.4))
        x_above = np.random.uniform(cutoff, cutoff + 1, int(n * 0.6))
        x = np.concatenate([x_below, x_above])
        np.random.shuffle(x)
        x = x[:n]
    else:
        x = np.random.uniform(cutoff - 1, cutoff + 1, n)

    # Treatment assignment
    if fuzzy:
        # Fuzzy RD: probability jumps from 0.3 to 0.8
        prob_treat = np.where(x >= cutoff, 0.8, 0.3)
        treatment = np.random.binomial(1, prob_treat)
    else:
        # Sharp RD: deterministic at cutoff
        treatment = (x >= cutoff).astype(int)

    # Outcome with treatment effect
    # Y = alpha + tau*D + beta*X + epsilon
    y = (0.5 +
         effect * treatment +
         0.8 * (x - cutoff) +
         0.3 * (x - cutoff)**2 +
         np.random.normal(0, 0.5, n))

    # Pre-determined covariates (should be continuous at cutoff)
    age = 35 + 10 * np.random.randn(n) + 2 * (x - cutoff)
    education = 12 + 3 * np.random.randn(n) + 1 * (x - cutoff)
    income_pre = 50000 + 15000 * np.random.randn(n) + 5000 * (x - cutoff)

    df = pd.DataFrame({
        'x': x,
        'treatment': treatment,
        'y': y,
        'age': age,
        'education': education,
        'income_pre': income_pre
    })

    return df


# ============================================================================
# CORE RD ESTIMATION
# ============================================================================

def run_sharp_rd(df: pd.DataFrame,
                 outcome: str,
                 running_var: str,
                 cutoff: float = 0.0,
                 bandwidth: float = None,
                 kernel: str = 'triangular',
                 p: int = 1) -> dict:
    """
    Run Sharp RD estimation using local polynomial regression.

    Falls back to manual implementation if rdrobust not available.

    Parameters
    ----------
    df : pd.DataFrame
        Data with outcome and running variable
    outcome : str
        Name of outcome variable
    running_var : str
        Name of running variable
    cutoff : float
        Cutoff value
    bandwidth : float
        Bandwidth (if None, use optimal)
    kernel : str
        Kernel type ('triangular', 'uniform', 'epanechnikov')
    p : int
        Polynomial order (1 = linear, 2 = quadratic)

    Returns
    -------
    dict
        Estimation results
    """
    try:
        from rdrobust import rdrobust, rdbwselect

        y = df[outcome].values
        x = df[running_var].values

        # Optimal bandwidth if not provided
        if bandwidth is None:
            bw_result = rdbwselect(y, x, c=cutoff)
            bandwidth = bw_result.bws[0]

        # Run rdrobust
        result = rdrobust(y, x, c=cutoff, h=bandwidth, kernel=kernel, p=p)

        return {
            'method': 'rdrobust',
            'estimate': result.coef[0],
            'se': result.se[0],
            'ci_lower': result.ci[0, 0],
            'ci_upper': result.ci[0, 1],
            'pvalue': result.pv[0],
            'bandwidth': bandwidth,
            'n_left': int(result.N[0]),
            'n_right': int(result.N[1]),
            'n_effective': int(result.N_h[0]) + int(result.N_h[1]),
            'raw_result': result
        }

    except ImportError:
        # Fallback to manual local linear regression
        return _manual_local_linear(df, outcome, running_var, cutoff, bandwidth)


def run_fuzzy_rd(df: pd.DataFrame,
                 outcome: str,
                 running_var: str,
                 treatment: str,
                 cutoff: float = 0.0,
                 bandwidth: float = None) -> dict:
    """
    Run Fuzzy RD estimation.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    outcome : str
        Outcome variable name
    running_var : str
        Running variable name
    treatment : str
        Treatment variable name
    cutoff : float
        Cutoff value
    bandwidth : float
        Bandwidth (optional)

    Returns
    -------
    dict
        Estimation results including first stage
    """
    try:
        from rdrobust import rdrobust, rdbwselect

        y = df[outcome].values
        x = df[running_var].values
        t = df[treatment].values

        # Optimal bandwidth
        if bandwidth is None:
            bw_result = rdbwselect(y, x, c=cutoff)
            bandwidth = bw_result.bws[0]

        # Fuzzy RD estimation
        result = rdrobust(y, x, c=cutoff, h=bandwidth, fuzzy=t)

        # First stage: treatment probability jump
        first_stage = rdrobust(t, x, c=cutoff, h=bandwidth)

        return {
            'method': 'rdrobust_fuzzy',
            'estimate': result.coef[0],
            'se': result.se[0],
            'ci_lower': result.ci[0, 0],
            'ci_upper': result.ci[0, 1],
            'pvalue': result.pv[0],
            'bandwidth': bandwidth,
            'first_stage_jump': first_stage.coef[0],
            'first_stage_se': first_stage.se[0],
            'n_effective': int(result.N_h[0]) + int(result.N_h[1]),
            'raw_result': result
        }

    except ImportError:
        return _manual_fuzzy_rd(df, outcome, running_var, treatment, cutoff, bandwidth)


def _manual_local_linear(df: pd.DataFrame,
                         outcome: str,
                         running_var: str,
                         cutoff: float,
                         bandwidth: float = None) -> dict:
    """
    Manual local linear regression implementation.

    Used as fallback when rdrobust is not available.
    """
    # Center running variable
    df = df.copy()
    df['x_centered'] = df[running_var] - cutoff
    df['treated'] = (df['x_centered'] >= 0).astype(int)

    # Simple bandwidth if not provided
    if bandwidth is None:
        bandwidth = df['x_centered'].std() * 0.5

    # Subset within bandwidth
    mask = (df['x_centered'] >= -bandwidth) & (df['x_centered'] <= bandwidth)
    subset = df[mask]

    if len(subset) < 20:
        raise ValueError(f"Too few observations within bandwidth: {len(subset)}")

    # Local linear regression with interaction
    formula = f"{outcome} ~ treated + x_centered + treated:x_centered"
    model = smf.ols(formula, data=subset).fit(cov_type='HC1')

    estimate = model.params['treated']
    se = model.bse['treated']
    ci = model.conf_int().loc['treated']

    return {
        'method': 'manual_local_linear',
        'estimate': estimate,
        'se': se,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'pvalue': model.pvalues['treated'],
        'bandwidth': bandwidth,
        'n_left': int((subset['treated'] == 0).sum()),
        'n_right': int((subset['treated'] == 1).sum()),
        'n_effective': len(subset),
        'raw_result': model
    }


def _manual_fuzzy_rd(df: pd.DataFrame,
                     outcome: str,
                     running_var: str,
                     treatment: str,
                     cutoff: float,
                     bandwidth: float = None) -> dict:
    """
    Manual fuzzy RD using 2SLS.
    """
    from linearmodels.iv import IV2SLS

    df = df.copy()
    df['x_centered'] = df[running_var] - cutoff
    df['above_cutoff'] = (df['x_centered'] >= 0).astype(int)

    if bandwidth is None:
        bandwidth = df['x_centered'].std() * 0.5

    mask = (df['x_centered'] >= -bandwidth) & (df['x_centered'] <= bandwidth)
    subset = df[mask]

    # Fuzzy RD as IV: use "above_cutoff" as instrument for treatment
    subset['const'] = 1

    # 2SLS
    formula = f"{outcome} ~ 1 + x_centered + [treatment ~ above_cutoff]"
    # Use linearmodels if available, otherwise simple 2SLS

    # Simple reduced form approach
    first_stage = smf.ols(f"{treatment} ~ above_cutoff + x_centered + above_cutoff:x_centered",
                          data=subset).fit()

    subset['treatment_hat'] = first_stage.fittedvalues
    second_stage = smf.ols(f"{outcome} ~ treatment_hat + x_centered + treatment_hat:x_centered",
                           data=subset).fit(cov_type='HC1')

    first_stage_jump = first_stage.params['above_cutoff']

    return {
        'method': 'manual_fuzzy_2sls',
        'estimate': second_stage.params['treatment_hat'],
        'se': second_stage.bse['treatment_hat'],
        'ci_lower': second_stage.conf_int().loc['treatment_hat'][0],
        'ci_upper': second_stage.conf_int().loc['treatment_hat'][1],
        'pvalue': second_stage.pvalues['treatment_hat'],
        'bandwidth': bandwidth,
        'first_stage_jump': first_stage_jump,
        'first_stage_se': first_stage.bse['above_cutoff'],
        'n_effective': len(subset),
        'raw_result': second_stage
    }


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def mccrary_test(df: pd.DataFrame,
                 running_var: str,
                 cutoff: float = 0.0) -> dict:
    """
    Manipulation test for RD running variable.

    Tests if density of running variable is continuous at cutoff.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    running_var : str
        Running variable name
    cutoff : float
        Cutoff value

    Returns
    -------
    dict
        Test results
    """
    try:
        from rddensity import rddensity

        x = df[running_var].values
        result = rddensity(x, c=cutoff)

        return {
            'method': 'rddensity',
            'test_statistic': result.T,
            'pvalue': result.pval,
            'n_left': int(result.N[0]),
            'n_right': int(result.N[1]),
            'manipulation_detected': result.pval < 0.05,
            'raw_result': result
        }

    except ImportError:
        # Fallback: simple binomial test
        return _manual_density_test(df, running_var, cutoff)


def _manual_density_test(df: pd.DataFrame,
                         running_var: str,
                         cutoff: float) -> dict:
    """
    Simple density test without rddensity package.

    Uses binomial test comparing observations just below/above cutoff.
    """
    x = df[running_var].values

    # Count in small bins around cutoff
    epsilon = (x.max() - x.min()) * 0.05
    n_left = ((x >= cutoff - epsilon) & (x < cutoff)).sum()
    n_right = ((x >= cutoff) & (x < cutoff + epsilon)).sum()

    total = n_left + n_right
    if total == 0:
        return {
            'method': 'manual_binomial',
            'test_statistic': np.nan,
            'pvalue': np.nan,
            'n_left': n_left,
            'n_right': n_right,
            'manipulation_detected': None
        }

    # Under null, expect equal proportions
    expected_prop = 0.5
    observed_prop = n_right / total

    # Binomial test
    result = stats.binomtest(n_right, total, expected_prop, alternative='two-sided')

    return {
        'method': 'manual_binomial',
        'test_statistic': (observed_prop - 0.5) / np.sqrt(0.25 / total),
        'pvalue': result.pvalue,
        'n_left': n_left,
        'n_right': n_right,
        'manipulation_detected': result.pvalue < 0.05
    }


def covariate_balance(df: pd.DataFrame,
                      covariates: list,
                      running_var: str,
                      cutoff: float = 0.0) -> pd.DataFrame:
    """
    Test covariate balance at the cutoff.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    covariates : list
        List of covariate names to test
    running_var : str
        Running variable name
    cutoff : float
        Cutoff value

    Returns
    -------
    pd.DataFrame
        Balance test results for each covariate
    """
    results = []

    for cov in covariates:
        try:
            result = run_sharp_rd(df, cov, running_var, cutoff)
            results.append({
                'covariate': cov,
                'estimate': result['estimate'],
                'se': result['se'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'pvalue': result['pvalue'],
                'significant': result['pvalue'] < 0.05
            })
        except Exception as e:
            results.append({
                'covariate': cov,
                'estimate': np.nan,
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'pvalue': np.nan,
                'significant': None,
                'error': str(e)
            })

    return pd.DataFrame(results)


def bandwidth_sensitivity(df: pd.DataFrame,
                          outcome: str,
                          running_var: str,
                          cutoff: float = 0.0,
                          bandwidth_multipliers: list = None) -> pd.DataFrame:
    """
    Test sensitivity to bandwidth choice.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    outcome : str
        Outcome variable
    running_var : str
        Running variable
    cutoff : float
        Cutoff value
    bandwidth_multipliers : list
        Multipliers of optimal bandwidth to test

    Returns
    -------
    pd.DataFrame
        Estimates at different bandwidths
    """
    if bandwidth_multipliers is None:
        bandwidth_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    # Get optimal bandwidth first
    base_result = run_sharp_rd(df, outcome, running_var, cutoff)
    h_opt = base_result['bandwidth']

    results = []
    for mult in bandwidth_multipliers:
        h = h_opt * mult
        result = run_sharp_rd(df, outcome, running_var, cutoff, bandwidth=h)
        results.append({
            'bandwidth': h,
            'multiplier': mult,
            'estimate': result['estimate'],
            'se': result['se'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'pvalue': result['pvalue'],
            'n_effective': result['n_effective']
        })

    return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_rd(df: pd.DataFrame,
            outcome: str,
            running_var: str,
            cutoff: float = 0.0,
            n_bins: int = 20,
            save_path: str = None) -> None:
    """
    Create RD plot with binscatter and fitted lines.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    outcome : str
        Outcome variable
    running_var : str
        Running variable
    cutoff : float
        Cutoff value
    n_bins : int
        Number of bins for binscatter
    save_path : str
        Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = df[running_var].values
    y = df[outcome].values

    # Split by cutoff
    left_mask = x < cutoff
    right_mask = x >= cutoff

    # Create bins for scatter
    x_left, y_left = x[left_mask], y[left_mask]
    x_right, y_right = x[right_mask], y[right_mask]

    # Bin means (binscatter)
    def bin_means(x_data, y_data, n_bins):
        bins = np.linspace(x_data.min(), x_data.max(), n_bins + 1)
        bin_centers = []
        bin_means = []
        for i in range(n_bins):
            mask = (x_data >= bins[i]) & (x_data < bins[i+1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                bin_means.append(y_data[mask].mean())
        return np.array(bin_centers), np.array(bin_means)

    x_bins_left, y_bins_left = bin_means(x_left, y_left, n_bins // 2)
    x_bins_right, y_bins_right = bin_means(x_right, y_right, n_bins // 2)

    # Plot binscatter points
    ax.scatter(x_bins_left, y_bins_left, color='blue', s=60, alpha=0.7, label='Below cutoff')
    ax.scatter(x_bins_right, y_bins_right, color='red', s=60, alpha=0.7, label='Above cutoff')

    # Fit local linear on each side
    if len(x_left) > 5:
        coef_left = np.polyfit(x_left, y_left, 1)
        x_fit_left = np.linspace(x_left.min(), cutoff, 100)
        y_fit_left = np.polyval(coef_left, x_fit_left)
        ax.plot(x_fit_left, y_fit_left, 'b-', linewidth=2)

    if len(x_right) > 5:
        coef_right = np.polyfit(x_right, y_right, 1)
        x_fit_right = np.linspace(cutoff, x_right.max(), 100)
        y_fit_right = np.polyval(coef_right, x_fit_right)
        ax.plot(x_fit_right, y_fit_right, 'r-', linewidth=2)

    # Cutoff line
    ax.axvline(cutoff, color='gray', linestyle='--', linewidth=2, label=f'Cutoff = {cutoff}')

    ax.set_xlabel(running_var, fontsize=12)
    ax.set_ylabel(outcome, fontsize=12)
    ax.set_title('Regression Discontinuity Plot', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"RD plot saved to: {save_path}")

    plt.close()


def plot_density(df: pd.DataFrame,
                 running_var: str,
                 cutoff: float = 0.0,
                 n_bins: int = 40,
                 save_path: str = None) -> None:
    """
    Plot density of running variable around cutoff.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    running_var : str
        Running variable name
    cutoff : float
        Cutoff value
    n_bins : int
        Number of bins
    save_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = df[running_var].values

    # Histogram on each side
    x_left = x[x < cutoff]
    x_right = x[x >= cutoff]

    ax.hist(x_left, bins=n_bins//2, alpha=0.6, color='blue',
            label=f'Below cutoff (n={len(x_left)})')
    ax.hist(x_right, bins=n_bins//2, alpha=0.6, color='red',
            label=f'Above cutoff (n={len(x_right)})')

    ax.axvline(cutoff, color='black', linestyle='--', linewidth=2,
               label=f'Cutoff = {cutoff}')

    ax.set_xlabel(running_var, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Density of Running Variable (Manipulation Check)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Density plot saved to: {save_path}")

    plt.close()


# ============================================================================
# REPORTING
# ============================================================================

def print_results(result: dict, title: str = "RD Estimation Results") -> None:
    """Print formatted RD results."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"Method: {result['method']}")
    print(f"Estimate: {result['estimate']:.4f}")
    print(f"Std. Error: {result['se']:.4f}")
    print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"P-value: {result['pvalue']:.4f}")

    sig = "***" if result['pvalue'] < 0.01 else ("**" if result['pvalue'] < 0.05 else ("*" if result['pvalue'] < 0.1 else ""))
    print(f"Significance: {sig}")

    print(f"\nBandwidth: {result['bandwidth']:.4f}")
    print(f"Effective N: {result['n_effective']}")

    if 'first_stage_jump' in result:
        print(f"\nFirst Stage Jump: {result['first_stage_jump']:.4f} (SE: {result['first_stage_se']:.4f})")


def generate_latex_table(results: list, save_path: str = None) -> str:
    """
    Generate LaTeX table for RD results.

    Parameters
    ----------
    results : list
        List of result dictionaries
    save_path : str
        Path to save LaTeX file

    Returns
    -------
    str
        LaTeX table code
    """
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Regression Discontinuity Estimates}
\label{tab:rd_results}
\begin{tabular}{lccccc}
\toprule
Specification & Estimate & SE & 95\% CI & p-value & Bandwidth \\
\midrule
"""

    for i, result in enumerate(results):
        name = result.get('name', f'Specification {i+1}')
        est = result['estimate']
        se = result['se']
        ci_low = result['ci_lower']
        ci_high = result['ci_upper']
        pval = result['pvalue']
        bw = result['bandwidth']

        sig = "^{***}" if pval < 0.01 else ("^{**}" if pval < 0.05 else ("^{*}" if pval < 0.1 else ""))

        latex += f"{name} & {est:.3f}{sig} & ({se:.3f}) & [{ci_low:.3f}, {ci_high:.3f}] & {pval:.3f} & {bw:.3f} \\\\\n"

    latex += r"""
\bottomrule
\multicolumn{6}{l}{\footnotesize $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$. Robust bias-corrected inference.} \\
\end{tabular}
\end{table}
"""

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {save_path}")

    return latex


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_analysis(df: pd.DataFrame,
                      outcome: str,
                      running_var: str,
                      cutoff: float = 0.0,
                      treatment: str = None,
                      covariates: list = None,
                      output_dir: str = None) -> dict:
    """
    Run complete RD analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    outcome : str
        Outcome variable name
    running_var : str
        Running variable name
    cutoff : float
        Cutoff value
    treatment : str
        Treatment variable (for fuzzy RD)
    covariates : list
        Covariates to test for balance
    output_dir : str
        Directory to save outputs

    Returns
    -------
    dict
        All analysis results
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = {}

    print("\n" + "=" * 70)
    print("REGRESSION DISCONTINUITY ANALYSIS")
    print("=" * 70)
    print(f"Outcome: {outcome}")
    print(f"Running Variable: {running_var}")
    print(f"Cutoff: {cutoff}")
    print(f"Sample Size: {len(df)}")
    print(f"N below cutoff: {(df[running_var] < cutoff).sum()}")
    print(f"N at/above cutoff: {(df[running_var] >= cutoff).sum()}")

    # Step 1: Manipulation Test
    print("\n" + "-" * 50)
    print("STEP 1: Manipulation Test")
    print("-" * 50)

    manip_result = mccrary_test(df, running_var, cutoff)
    results['manipulation_test'] = manip_result

    print(f"Test Statistic: {manip_result['test_statistic']:.4f}")
    print(f"P-value: {manip_result['pvalue']:.4f}")

    if manip_result['manipulation_detected']:
        print("WARNING: Evidence of manipulation at cutoff!")
    else:
        print("No evidence of manipulation (density continuous at cutoff)")

    # Plot density
    if output_dir:
        plot_density(df, running_var, cutoff,
                     save_path=os.path.join(output_dir, 'density_plot.png'))

    # Step 2: Covariate Balance
    if covariates:
        print("\n" + "-" * 50)
        print("STEP 2: Covariate Balance Tests")
        print("-" * 50)

        balance_df = covariate_balance(df, covariates, running_var, cutoff)
        results['covariate_balance'] = balance_df

        print(balance_df.to_string(index=False))

        n_imbalanced = balance_df['significant'].sum()
        if n_imbalanced > 0:
            print(f"\nWARNING: {n_imbalanced} covariate(s) show imbalance at cutoff!")
        else:
            print("\nAll covariates balanced at cutoff")

    # Step 3: Main Estimation
    print("\n" + "-" * 50)
    print("STEP 3: Main RD Estimation")
    print("-" * 50)

    if treatment is not None:
        # Fuzzy RD
        main_result = run_fuzzy_rd(df, outcome, running_var, treatment, cutoff)
        main_result['name'] = 'Fuzzy RD (LATE)'
    else:
        # Sharp RD
        main_result = run_sharp_rd(df, outcome, running_var, cutoff)
        main_result['name'] = 'Sharp RD'

    results['main_estimate'] = main_result
    print_results(main_result)

    # Step 4: Bandwidth Sensitivity
    print("\n" + "-" * 50)
    print("STEP 4: Bandwidth Sensitivity")
    print("-" * 50)

    sensitivity_df = bandwidth_sensitivity(df, outcome, running_var, cutoff)
    results['bandwidth_sensitivity'] = sensitivity_df

    print(sensitivity_df.to_string(index=False))

    # Step 5: Visualization
    print("\n" + "-" * 50)
    print("STEP 5: Visualization")
    print("-" * 50)

    if output_dir:
        plot_rd(df, outcome, running_var, cutoff,
                save_path=os.path.join(output_dir, 'rd_plot.png'))

        # Generate LaTeX table
        latex_results = [main_result]
        for _, row in sensitivity_df.iterrows():
            if row['multiplier'] != 1.0:
                latex_results.append({
                    'name': f"BW x{row['multiplier']:.2f}",
                    'estimate': row['estimate'],
                    'se': row['se'],
                    'ci_lower': row['ci_lower'],
                    'ci_upper': row['ci_upper'],
                    'pvalue': row['pvalue'],
                    'bandwidth': row['bandwidth']
                })

        generate_latex_table(latex_results,
                             save_path=os.path.join(output_dir, 'rd_table.tex'))

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='RD Analysis Pipeline')
    parser.add_argument('--demo', action='store_true', help='Run with simulated data')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--outcome', type=str, default='y', help='Outcome variable name')
    parser.add_argument('--running', type=str, default='x', help='Running variable name')
    parser.add_argument('--cutoff', type=float, default=0.0, help='Cutoff value')
    parser.add_argument('--treatment', type=str, help='Treatment variable (for fuzzy RD)')
    parser.add_argument('--fuzzy', action='store_true', help='Estimate fuzzy RD')
    parser.add_argument('--output', type=str, default='rd_output', help='Output directory')

    args = parser.parse_args()

    if args.demo:
        print("Running demo with simulated Sharp RD data...")
        df = simulate_rd_data(n=1000, effect=2.0, cutoff=0.0, fuzzy=False)
        covariates = ['age', 'education', 'income_pre']

        results = run_full_analysis(
            df=df,
            outcome='y',
            running_var='x',
            cutoff=0.0,
            treatment=None,
            covariates=covariates,
            output_dir=args.output
        )

        print(f"\nTrue effect: 2.0")
        print(f"Estimated effect: {results['main_estimate']['estimate']:.4f}")
        print(f"Bias: {results['main_estimate']['estimate'] - 2.0:.4f}")

    elif args.data:
        df = pd.read_csv(args.data)
        treatment = args.treatment if args.fuzzy else None

        results = run_full_analysis(
            df=df,
            outcome=args.outcome,
            running_var=args.running,
            cutoff=args.cutoff,
            treatment=treatment,
            output_dir=args.output
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
