#!/usr/bin/env python3
"""
Complete Propensity Score Matching (PSM) Analysis Pipeline

This script demonstrates a full PSM workflow including:
- Data simulation with selection on observables
- Propensity score estimation
- Multiple matching methods (NN, caliper)
- Inverse Probability Weighting (IPW)
- Doubly Robust (AIPW) estimation
- Balance diagnostics and Love plots
- Sensitivity analysis (Rosenbaum bounds)
- Publication-quality output

Usage:
    python psm_analysis_pipeline.py --demo
    python psm_analysis_pipeline.py --data data.csv --treatment T --outcome Y --covariates X1,X2,X3

Dependencies:
    pip install scikit-learn pandas numpy scipy matplotlib seaborn

Reference:
    Rosenbaum & Rubin (1983). The Central Role of the Propensity Score.
    Imbens & Rubin (2015). Causal Inference for Statistics.
"""

import argparse
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings('ignore')


# =============================================================================
# 1. DATA SIMULATION
# =============================================================================

def simulate_psm_data(
    n: int = 2000,
    ate: float = 2.0,
    selection_strength: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate observational data with selection on observables.

    Model:
        X1, X2, X3, X4, X5 ~ covariates
        P(T=1|X) = logistic(selection_strength * (X1 + X2 - 1))
        Y = ate * T + X1 + 0.5*X2 + 0.3*X3 + epsilon

    Parameters
    ----------
    n : int
        Sample size
    ate : float
        True average treatment effect
    selection_strength : float
        How strongly covariates affect treatment selection
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Simulated data with treatment, outcome, and covariates
    """
    np.random.seed(seed)

    # Generate covariates
    X1 = np.random.normal(0, 1, n)  # Confounder
    X2 = np.random.normal(0, 1, n)  # Confounder
    X3 = np.random.normal(0, 1, n)  # Affects outcome only
    X4 = np.random.binomial(1, 0.5, n)  # Binary covariate
    X5 = np.random.uniform(0, 10, n)  # Continuous covariate

    # Treatment assignment (depends on X1, X2)
    propensity = 1 / (1 + np.exp(-selection_strength * (X1 + 0.5*X2 - 0.5)))
    T = np.random.binomial(1, propensity, n)

    # Outcome (depends on treatment and covariates)
    epsilon = np.random.normal(0, 1, n)
    Y = ate * T + 1.5*X1 + 0.8*X2 + 0.3*X3 + 0.2*X4 + 0.05*X5 + epsilon

    # True propensity score for reference
    true_ps = propensity

    df = pd.DataFrame({
        'Y': Y,
        'T': T,
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X5,
        'true_ps': true_ps
    })

    return df


# =============================================================================
# 2. PROPENSITY SCORE ESTIMATION
# =============================================================================

def estimate_propensity_score(
    df: pd.DataFrame,
    treatment: str,
    covariates: list,
    method: str = 'logistic'
) -> np.ndarray:
    """
    Estimate propensity scores.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    treatment : str
        Treatment variable name
    covariates : list
        List of covariate names
    method : str
        'logistic' or 'gbm'

    Returns
    -------
    np.ndarray
        Estimated propensity scores
    """
    X = df[covariates]
    T = df[treatment]

    if method == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
        # Cross-validated predictions to reduce overfitting
        ps = cross_val_predict(model, X, T, cv=5, method='predict_proba')[:, 1]
    elif method == 'gbm':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        ps = cross_val_predict(model, X, T, cv=5, method='predict_proba')[:, 1]
    else:
        raise ValueError(f"Unknown method: {method}")

    return ps


def check_overlap(df: pd.DataFrame, treatment: str, ps_col: str = 'pscore') -> dict:
    """
    Check propensity score overlap (common support).
    """
    treated_ps = df.loc[df[treatment] == 1, ps_col]
    control_ps = df.loc[df[treatment] == 0, ps_col]

    ps_min = max(treated_ps.min(), control_ps.min())
    ps_max = min(treated_ps.max(), control_ps.max())

    overlap = {
        'treated_range': (treated_ps.min(), treated_ps.max()),
        'control_range': (control_ps.min(), control_ps.max()),
        'common_support': (ps_min, ps_max),
        'n_outside_support': ((df[ps_col] < ps_min) | (df[ps_col] > ps_max)).sum()
    }

    return overlap


# =============================================================================
# 3. MATCHING METHODS
# =============================================================================

def nearest_neighbor_matching(
    df: pd.DataFrame,
    treatment: str,
    ps_col: str = 'pscore',
    n_neighbors: int = 1,
    caliper: float = None,
    with_replacement: bool = False
) -> tuple:
    """
    Perform nearest neighbor matching on propensity score.

    Parameters
    ----------
    df : pd.DataFrame
        Data with propensity scores
    treatment : str
        Treatment variable name
    ps_col : str
        Propensity score column name
    n_neighbors : int
        Number of matches per treated unit
    caliper : float
        Maximum distance for matching (in PS units)
    with_replacement : bool
        Whether to match with replacement

    Returns
    -------
    tuple
        (matched_treated_df, matched_control_df, match_info)
    """
    treated = df[df[treatment] == 1].copy()
    control = df[df[treatment] == 0].copy()

    # Fit nearest neighbors on control
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(control[[ps_col]])

    # Find matches
    distances, indices = nn.kneighbors(treated[[ps_col]])

    # Apply caliper if specified
    if caliper is not None:
        within_caliper = distances[:, 0] <= caliper
        treated = treated[within_caliper]
        indices = indices[within_caliper]
        distances = distances[within_caliper]

    # Get matched control units
    matched_control_idx = indices.flatten()

    if not with_replacement:
        # Remove duplicates (keep first match)
        _, unique_idx = np.unique(matched_control_idx, return_index=True)
        unique_idx = np.sort(unique_idx)
        matched_control_idx = matched_control_idx[unique_idx]
        treated = treated.iloc[unique_idx]

    matched_control = control.iloc[matched_control_idx]

    match_info = {
        'n_treated': len(treated),
        'n_matched': len(matched_control),
        'match_rate': len(treated) / len(df[df[treatment] == 1]),
        'mean_distance': distances.mean() if len(distances) > 0 else np.nan
    }

    return treated.reset_index(drop=True), matched_control.reset_index(drop=True), match_info


# =============================================================================
# 4. WEIGHTING METHODS
# =============================================================================

def compute_ipw_weights(
    df: pd.DataFrame,
    treatment: str,
    ps_col: str = 'pscore',
    estimand: str = 'ATT',
    stabilized: bool = False,
    trim: float = 0.01
) -> np.ndarray:
    """
    Compute Inverse Probability Weights.

    Parameters
    ----------
    estimand : str
        'ATT' or 'ATE'
    stabilized : bool
        Whether to use stabilized weights
    trim : float
        Trim propensity scores below this value (and above 1-trim)

    Returns
    -------
    np.ndarray
        IPW weights
    """
    T = df[treatment].values
    ps = np.clip(df[ps_col].values, trim, 1 - trim)

    if estimand == 'ATT':
        # Treated: weight = 1
        # Control: weight = ps / (1 - ps)
        weights = np.where(T == 1, 1, ps / (1 - ps))
    elif estimand == 'ATE':
        # Treated: weight = 1 / ps
        # Control: weight = 1 / (1 - ps)
        weights = np.where(T == 1, 1 / ps, 1 / (1 - ps))
    else:
        raise ValueError(f"Unknown estimand: {estimand}")

    if stabilized:
        p_treat = T.mean()
        if estimand == 'ATT':
            weights = np.where(T == 1, 1, ps * (1 - p_treat) / ((1 - ps) * p_treat))
        else:
            weights = np.where(T == 1, p_treat / ps, (1 - p_treat) / (1 - ps))

    return weights


# =============================================================================
# 5. TREATMENT EFFECT ESTIMATION
# =============================================================================

def estimate_att_matching(
    treated: pd.DataFrame,
    matched_control: pd.DataFrame,
    outcome: str,
    n_bootstrap: int = 1000
) -> dict:
    """
    Estimate ATT from matched sample with bootstrap SE.
    """
    att = treated[outcome].mean() - matched_control[outcome].mean()

    # Bootstrap
    boot_atts = []
    n = len(treated)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_att = treated.iloc[idx][outcome].mean() - matched_control.iloc[idx][outcome].mean()
        boot_atts.append(boot_att)

    se = np.std(boot_atts)
    ci = (att - 1.96 * se, att + 1.96 * se)

    return {
        'estimand': 'ATT',
        'method': 'PSM',
        'estimate': att,
        'se': se,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'n_treated': len(treated),
        'n_control': len(matched_control)
    }


def estimate_att_ipw(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    weights: np.ndarray,
    n_bootstrap: int = 1000
) -> dict:
    """
    Estimate ATT using IPW with bootstrap SE.
    """
    T = df[treatment].values
    Y = df[outcome].values

    treated_mean = Y[T == 1].mean()
    control_weighted_mean = np.average(Y[T == 0], weights=weights[T == 0])
    att = treated_mean - control_weighted_mean

    # Bootstrap
    boot_atts = []
    n = len(df)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        T_b = T[idx]
        Y_b = Y[idx]
        w_b = weights[idx]
        if T_b.sum() > 0 and (1 - T_b).sum() > 0:
            att_b = Y_b[T_b == 1].mean() - np.average(Y_b[T_b == 0], weights=w_b[T_b == 0])
            boot_atts.append(att_b)

    se = np.std(boot_atts)
    ci = (att - 1.96 * se, att + 1.96 * se)

    return {
        'estimand': 'ATT',
        'method': 'IPW',
        'estimate': att,
        'se': se,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'n': len(df)
    }


def estimate_aipw(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list,
    n_bootstrap: int = 500
) -> dict:
    """
    Estimate ATE using Augmented IPW (Doubly Robust).
    """
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values
    n = len(df)

    # Propensity score
    ps_model = LogisticRegression(max_iter=500, random_state=42)
    ps_model.fit(X, T)
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)

    # Outcome models
    mu1_model = LinearRegression()
    mu1_model.fit(X[T == 1], Y[T == 1])
    mu1 = mu1_model.predict(X)

    mu0_model = LinearRegression()
    mu0_model.fit(X[T == 0], Y[T == 0])
    mu0 = mu0_model.predict(X)

    # AIPW estimator
    aipw_y1 = (T * Y / ps + (1 - T / ps) * mu1).mean()
    aipw_y0 = ((1 - T) * Y / (1 - ps) + (1 - (1 - T) / (1 - ps)) * mu0).mean()
    ate = aipw_y1 - aipw_y0

    # Bootstrap
    def aipw_boot(idx):
        X_b, T_b, Y_b = X[idx], T[idx], Y[idx]
        if T_b.sum() < 5 or (1 - T_b).sum() < 5:
            return np.nan

        ps_m = LogisticRegression(max_iter=300, random_state=42)
        ps_m.fit(X_b, T_b)
        ps_b = np.clip(ps_m.predict_proba(X_b)[:, 1], 0.01, 0.99)

        mu1_m = LinearRegression()
        mu1_m.fit(X_b[T_b == 1], Y_b[T_b == 1])
        mu1_b = mu1_m.predict(X_b)

        mu0_m = LinearRegression()
        mu0_m.fit(X_b[T_b == 0], Y_b[T_b == 0])
        mu0_b = mu0_m.predict(X_b)

        y1 = (T_b * Y_b / ps_b + (1 - T_b / ps_b) * mu1_b).mean()
        y0 = ((1 - T_b) * Y_b / (1 - ps_b) + (1 - (1 - T_b) / (1 - ps_b)) * mu0_b).mean()
        return y1 - y0

    boot_ates = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        ate_b = aipw_boot(idx)
        if not np.isnan(ate_b):
            boot_ates.append(ate_b)

    se = np.std(boot_ates)
    ci = (ate - 1.96 * se, ate + 1.96 * se)

    return {
        'estimand': 'ATE',
        'method': 'AIPW',
        'estimate': ate,
        'se': se,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'n': n
    }


# =============================================================================
# 6. BALANCE DIAGNOSTICS
# =============================================================================

def calculate_smd(treated: np.ndarray, control: np.ndarray, weights: np.ndarray = None) -> float:
    """
    Calculate Standardized Mean Difference.
    """
    if weights is not None and len(weights) == len(control):
        control_mean = np.average(control, weights=weights)
    else:
        control_mean = np.mean(control)

    treated_mean = np.mean(treated)
    pooled_std = np.sqrt((np.var(treated) + np.var(control)) / 2)

    if pooled_std == 0:
        return 0.0
    return (treated_mean - control_mean) / pooled_std


def balance_table(
    df: pd.DataFrame,
    treatment: str,
    covariates: list,
    weights: np.ndarray = None,
    matched_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Generate balance table with SMD before and after adjustment.
    """
    results = []

    for var in covariates:
        treated = df.loc[df[treatment] == 1, var].values
        control = df.loc[df[treatment] == 0, var].values

        # Unadjusted SMD
        smd_unadj = calculate_smd(treated, control)

        # Adjusted SMD
        if matched_df is not None:
            # For matched data
            m_treated = matched_df.loc[matched_df[treatment] == 1, var].values
            m_control = matched_df.loc[matched_df[treatment] == 0, var].values
            smd_adj = calculate_smd(m_treated, m_control)
        elif weights is not None:
            # For weighted data
            w_control = weights[df[treatment] == 0]
            smd_adj = calculate_smd(treated, control, weights=w_control)
        else:
            smd_adj = smd_unadj

        results.append({
            'Variable': var,
            'Mean (T)': treated.mean(),
            'Mean (C)': control.mean(),
            'SMD (Unadj)': smd_unadj,
            'SMD (Adj)': smd_adj,
            'Balanced': abs(smd_adj) < 0.1
        })

    return pd.DataFrame(results)


def plot_balance(balance_df: pd.DataFrame, save_path: str = None):
    """
    Create Love plot showing balance improvement.
    """
    fig, ax = plt.subplots(figsize=(10, max(6, len(balance_df) * 0.4)))

    y_pos = range(len(balance_df))

    # Unadjusted
    ax.scatter(balance_df['SMD (Unadj)'].abs(), y_pos, marker='o',
               color='red', s=100, label='Unadjusted', zorder=3)

    # Adjusted
    ax.scatter(balance_df['SMD (Adj)'].abs(), y_pos, marker='s',
               color='blue', s=100, label='Adjusted', zorder=3)

    # Connect points
    for i, row in balance_df.iterrows():
        ax.plot([abs(row['SMD (Unadj)']), abs(row['SMD (Adj)'])], [i, i],
                'gray', linestyle='-', linewidth=0.5, zorder=1)

    # Reference lines
    ax.axvline(x=0.1, color='green', linestyle='--', linewidth=2, label='Balance threshold (0.1)')
    ax.axvline(x=0.25, color='orange', linestyle='--', linewidth=1, label='Concern threshold (0.25)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(balance_df['Variable'])
    ax.set_xlabel('|Standardized Mean Difference|')
    ax.set_title('Covariate Balance: Love Plot')
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_pscore_distribution(df: pd.DataFrame, treatment: str, ps_col: str, save_path: str = None):
    """
    Plot propensity score distributions for treated and control.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    treated_ps = df.loc[df[treatment] == 1, ps_col]
    control_ps = df.loc[df[treatment] == 0, ps_col]

    ax.hist(control_ps, bins=50, alpha=0.5, label='Control', color='blue', density=True)
    ax.hist(treated_ps, bins=50, alpha=0.5, label='Treated', color='red', density=True)

    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.set_title('Propensity Score Distribution')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


# =============================================================================
# 7. SENSITIVITY ANALYSIS
# =============================================================================

def rosenbaum_bounds(treated_outcomes: np.ndarray, control_outcomes: np.ndarray,
                     gamma_range: list = None) -> pd.DataFrame:
    """
    Compute Rosenbaum bounds for sensitivity to unmeasured confounding.
    """
    if gamma_range is None:
        gamma_range = [1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]

    diffs = treated_outcomes - control_outcomes
    n_pairs = len(diffs)

    if n_pairs == 0:
        return pd.DataFrame()

    signs = np.sign(diffs)
    abs_diffs = np.abs(diffs)
    ranks = stats.rankdata(abs_diffs)

    W = np.sum(ranks[signs > 0])
    E_W = n_pairs * (n_pairs + 1) / 4

    results = []
    for gamma in gamma_range:
        p_upper = gamma / (1 + gamma)

        # Variance approximation
        V_W = n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 24

        # Adjusted for gamma
        V_W_adj = V_W * 4 * p_upper * (1 - p_upper)

        if V_W_adj > 0:
            z = (W - E_W) / np.sqrt(V_W_adj)
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            p_val = np.nan

        results.append({
            'Gamma': gamma,
            'P-value': p_val,
            'Significant': p_val < 0.05 if not np.isnan(p_val) else False
        })

    return pd.DataFrame(results)


# =============================================================================
# 8. REPORTING
# =============================================================================

def print_results_table(results_list: list, true_effect: float = None):
    """
    Print publication-style results table.
    """
    print("\n" + "=" * 80)
    print("PROPENSITY SCORE ANALYSIS RESULTS".center(80))
    print("=" * 80)

    print(f"{'Method':<12} {'Estimand':<8} {'Estimate':>10} {'SE':>10} {'95% CI':>25} {'N':>8}")
    print("-" * 80)

    for res in results_list:
        ci = f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]"
        n = res.get('n', res.get('n_treated', '-'))
        print(f"{res['method']:<12} {res['estimand']:<8} {res['estimate']:>10.4f} {res['se']:>10.4f} {ci:>25} {n:>8}")

    print("-" * 80)

    if true_effect is not None:
        print(f"\nTrue effect: {true_effect:.4f}")
        for res in results_list:
            bias = res['estimate'] - true_effect
            print(f"  {res['method']} bias: {bias:+.4f}")

    print("=" * 80)


# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================

def run_full_analysis(
    df: pd.DataFrame = None,
    treatment: str = 'T',
    outcome: str = 'Y',
    covariates: list = None,
    true_effect: float = None,
    output_dir: str = None
) -> dict:
    """
    Run complete PSM analysis pipeline.
    """
    all_results = {}

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    # Use simulated data if none provided
    if df is None:
        print("Using simulated data for demonstration...")
        df = simulate_psm_data(n=2000, ate=2.0, selection_strength=1.0)
        covariates = ['X1', 'X2', 'X3', 'X4', 'X5']
        true_effect = 2.0

    if covariates is None:
        raise ValueError("Must specify covariates")

    print("\n" + "=" * 70)
    print("PROPENSITY SCORE MATCHING ANALYSIS".center(70))
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Sample size: {len(df)}")
    print(f"Treated: {df[treatment].sum()} ({df[treatment].mean()*100:.1f}%)")
    if true_effect is not None:
        print(f"True effect: {true_effect}")

    # 1. Estimate propensity scores
    print("\n[1/6] Estimating propensity scores...")
    df['pscore'] = estimate_propensity_score(df, treatment, covariates, method='logistic')
    all_results['propensity_scores'] = df['pscore'].describe().to_dict()

    # Check overlap
    overlap = check_overlap(df, treatment)
    print(f"  Common support: [{overlap['common_support'][0]:.4f}, {overlap['common_support'][1]:.4f}]")
    print(f"  Obs outside support: {overlap['n_outside_support']}")

    # Plot PS distribution
    if output_dir:
        plot_pscore_distribution(df, treatment, 'pscore', str(output_dir / 'pscore_distribution.png'))
        print(f"  Saved: pscore_distribution.png")

    # 2. Nearest neighbor matching
    print("\n[2/6] Performing nearest neighbor matching...")
    caliper = 0.2 * df['pscore'].std()
    matched_treated, matched_control, match_info = nearest_neighbor_matching(
        df, treatment, 'pscore', n_neighbors=1, caliper=caliper
    )
    print(f"  Match rate: {match_info['match_rate']*100:.1f}%")
    print(f"  Mean PS distance: {match_info['mean_distance']:.4f}")

    # Create matched dataframe for balance check
    matched_df = pd.concat([matched_treated, matched_control], ignore_index=True)

    # 3. Balance diagnostics
    print("\n[3/6] Checking covariate balance...")
    balance = balance_table(df, treatment, covariates, matched_df=matched_df)
    n_balanced = balance['Balanced'].sum()
    print(f"  Balanced covariates: {n_balanced}/{len(covariates)}")
    all_results['balance'] = balance.to_dict()

    if output_dir:
        plot_balance(balance, str(output_dir / 'love_plot.png'))
        print(f"  Saved: love_plot.png")

    # 4. Estimate effects
    print("\n[4/6] Estimating treatment effects...")

    # PSM ATT
    psm_result = estimate_att_matching(matched_treated, matched_control, outcome)
    print(f"  PSM ATT: {psm_result['estimate']:.4f} (SE: {psm_result['se']:.4f})")

    # IPW ATT
    ipw_weights = compute_ipw_weights(df, treatment, 'pscore', estimand='ATT')
    ipw_result = estimate_att_ipw(df, treatment, outcome, ipw_weights)
    print(f"  IPW ATT: {ipw_result['estimate']:.4f} (SE: {ipw_result['se']:.4f})")

    # AIPW ATE
    print("\n[5/6] Estimating doubly robust (AIPW)...")
    aipw_result = estimate_aipw(df, treatment, outcome, covariates)
    print(f"  AIPW ATE: {aipw_result['estimate']:.4f} (SE: {aipw_result['se']:.4f})")

    # 6. Sensitivity analysis
    print("\n[6/6] Sensitivity analysis (Rosenbaum bounds)...")
    bounds = rosenbaum_bounds(matched_treated[outcome].values, matched_control[outcome].values)
    all_results['sensitivity'] = bounds.to_dict()

    # Find critical gamma
    critical_gamma = bounds.loc[~bounds['Significant'], 'Gamma'].min() if (~bounds['Significant']).any() else '>3.0'
    print(f"  Effect robust up to Gamma = {critical_gamma}")

    # Print summary
    results_list = [psm_result, ipw_result, aipw_result]
    print_results_table(results_list, true_effect)

    # Save results
    all_results['estimates'] = {r['method']: r for r in results_list}

    if output_dir:
        # Save balance table
        balance.to_csv(str(output_dir / 'balance_table.csv'), index=False)

        # Save sensitivity analysis
        bounds.to_csv(str(output_dir / 'sensitivity_analysis.csv'), index=False)

        print(f"\nResults saved to: {output_dir}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE".center(70))
    print("=" * 70)

    return all_results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Propensity Score Matching analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python psm_analysis_pipeline.py --demo
    python psm_analysis_pipeline.py --data data.csv --treatment T --outcome Y --covariates X1,X2,X3
        """
    )

    parser.add_argument('--demo', action='store_true', help='Run with simulated data')
    parser.add_argument('--data', '-d', type=str, help='Path to CSV data file')
    parser.add_argument('--treatment', '-t', type=str, default='T', help='Treatment variable')
    parser.add_argument('--outcome', '-y', type=str, default='Y', help='Outcome variable')
    parser.add_argument('--covariates', '-x', type=str, help='Comma-separated covariate list')
    parser.add_argument('--output', '-o', type=str, default='./psm_results', help='Output directory')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.demo or args.data is None:
        results = run_full_analysis(output_dir=args.output)
    else:
        df = pd.read_csv(args.data)
        covariates = args.covariates.split(',') if args.covariates else None

        results = run_full_analysis(
            df=df,
            treatment=args.treatment,
            outcome=args.outcome,
            covariates=covariates,
            output_dir=args.output
        )
