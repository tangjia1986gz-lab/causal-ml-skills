#!/usr/bin/env python3
"""
Complete Instrumental Variables (IV/2SLS) Analysis Pipeline

This script demonstrates a full IV workflow including:
- Data simulation with endogeneity
- First-stage diagnostics (weak instrument tests)
- 2SLS and LIML estimation
- Diagnostic tests (Hausman, Sargan-Hansen)
- Weak-instrument robust inference (Anderson-Rubin)
- Publication-quality output

Usage:
    python iv_analysis_pipeline.py --demo
    python iv_analysis_pipeline.py --data data.csv --outcome y --endogenous x --instruments z1,z2

Dependencies:
    pip install linearmodels statsmodels pandas numpy scipy matplotlib

Reference:
    Angrist, J. D., & Pischke, J. S. (2009). Mostly Harmless Econometrics.
    Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments.
"""

import argparse
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')


# =============================================================================
# 1. DATA SIMULATION
# =============================================================================

def simulate_iv_data(
    n: int = 1000,
    beta_true: float = 0.5,
    gamma: float = 0.8,
    rho: float = 0.6,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate data with endogeneity for IV analysis.

    Model:
        Y = beta * X + epsilon
        X = gamma * Z + nu
        Cov(epsilon, nu) = rho (endogeneity)

    Parameters
    ----------
    n : int
        Sample size
    beta_true : float
        True causal effect of X on Y
    gamma : float
        Effect of instrument Z on endogenous X (first stage)
    rho : float
        Correlation between epsilon and nu (endogeneity strength)
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Simulated data with columns: y, x, z, z2, control
    """
    np.random.seed(seed)

    # Generate correlated errors (source of endogeneity)
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    errors = np.random.multivariate_normal(mean, cov, n)
    epsilon = errors[:, 0]  # Error in outcome equation
    nu = errors[:, 1]       # Error in first stage

    # Instruments (exogenous)
    z1 = np.random.normal(0, 1, n)
    z2 = np.random.normal(0, 1, n)

    # Control variable
    control = np.random.normal(0, 1, n)

    # Endogenous variable (affected by instrument and correlated error)
    x = gamma * z1 + 0.3 * z2 + 0.2 * control + nu

    # Outcome (affected by X and correlated error)
    y = beta_true * x + 0.3 * control + epsilon

    df = pd.DataFrame({
        'y': y,
        'x': x,
        'z1': z1,
        'z2': z2,
        'control': control
    })
    df['const'] = 1

    return df


# =============================================================================
# 2. FIRST STAGE ANALYSIS
# =============================================================================

def first_stage_analysis(df: pd.DataFrame, endogenous: str,
                         instruments: list, controls: list = None) -> dict:
    """
    Run first-stage regression and diagnostic tests.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    endogenous : str
        Name of endogenous variable
    instruments : list
        List of instrument names
    controls : list, optional
        List of control variable names

    Returns
    -------
    dict
        First-stage results including F-statistic
    """
    import statsmodels.api as sm

    if controls is None:
        controls = []

    # Build regressor matrix
    X_vars = instruments + controls + ['const']
    X = df[X_vars]
    y = df[endogenous]

    # Run first-stage regression
    model = sm.OLS(y, X)
    result = model.fit(cov_type='HC1')

    # F-test for excluded instruments
    n_instruments = len(instruments)
    r_matrix = np.zeros((n_instruments, len(X_vars)))
    for i, inst in enumerate(instruments):
        r_matrix[i, X_vars.index(inst)] = 1

    f_test = result.f_test(r_matrix)
    # Handle both scalar and array returns
    fvalue = f_test.fvalue
    if hasattr(fvalue, '__iter__'):
        f_stat = float(fvalue[0][0]) if hasattr(fvalue[0], '__iter__') else float(fvalue[0])
    else:
        f_stat = float(fvalue)
    f_pval = float(f_test.pvalue)

    # Stock-Yogo critical values (10% maximal IV size)
    stock_yogo_cv = {
        1: 16.38,
        2: 19.93,
        3: 22.30,
        4: 24.58,
        5: 26.87
    }
    cv = stock_yogo_cv.get(n_instruments, 10)

    results = {
        'model': result,
        'f_statistic': f_stat,
        'f_pvalue': f_pval,
        'stock_yogo_cv': cv,
        'is_strong': f_stat > cv,
        'n_instruments': n_instruments,
        'coefficients': result.params[instruments].to_dict(),
        'std_errors': result.bse[instruments].to_dict(),
        'r_squared': result.rsquared
    }

    return results


# =============================================================================
# 3. IV ESTIMATION
# =============================================================================

def run_iv_estimation(df: pd.DataFrame, outcome: str, endogenous: str,
                      instruments: list, controls: list = None,
                      method: str = '2sls') -> dict:
    """
    Run IV estimation (2SLS, LIML, or GMM).

    Parameters
    ----------
    df : pd.DataFrame
        Data
    outcome : str
        Outcome variable name
    endogenous : str
        Endogenous variable name
    instruments : list
        List of instrument names
    controls : list, optional
        List of control variable names
    method : str
        Estimation method: '2sls', 'liml', or 'gmm'

    Returns
    -------
    dict
        IV estimation results
    """
    from linearmodels.iv import IV2SLS, IVLIML, IVGMM

    if controls is None:
        controls = []

    # Select estimator
    estimators = {
        '2sls': IV2SLS,
        'liml': IVLIML,
        'gmm': IVGMM
    }
    Estimator = estimators.get(method.lower(), IV2SLS)

    # Build model
    model = Estimator(
        dependent=df[outcome],
        exog=df[controls + ['const']],
        endog=df[[endogenous]],
        instruments=df[instruments]
    )

    # Fit with robust SEs
    result = model.fit(cov_type='robust')

    # Extract diagnostics
    diagnostics = {}

    # First-stage F
    if hasattr(result, 'first_stage') and result.first_stage is not None:
        try:
            fs_diag = result.first_stage.diagnostics
            if 'f.stat' in fs_diag.index:
                diagnostics['first_stage_f'] = float(fs_diag.loc['f.stat', 'statistic'])
                diagnostics['first_stage_f_pval'] = float(fs_diag.loc['f.stat', 'pvalue'])
        except (KeyError, AttributeError):
            pass

    # Wu-Hausman test
    try:
        wh = result.wu_hausman() if callable(result.wu_hausman) else result.wu_hausman
        if wh is not None and hasattr(wh, 'stat'):
            diagnostics['hausman_stat'] = wh.stat
            diagnostics['hausman_pval'] = wh.pval
    except Exception:
        pass

    # Sargan-Hansen test (if overidentified)
    try:
        sargan = result.sargan() if callable(result.sargan) else result.sargan
        if len(instruments) > 1 and sargan is not None and hasattr(sargan, 'stat'):
            diagnostics['sargan_stat'] = sargan.stat
            diagnostics['sargan_pval'] = sargan.pval
    except Exception:
        pass

    results = {
        'method': method.upper(),
        'model': result,
        'coefficient': result.params[endogenous],
        'std_error': result.std_errors[endogenous],
        'p_value': result.pvalues[endogenous],
        'ci_lower': result.conf_int().loc[endogenous, 'lower'],
        'ci_upper': result.conf_int().loc[endogenous, 'upper'],
        'n_obs': result.nobs,
        'r_squared': result.rsquared,
        'diagnostics': diagnostics
    }

    return results


# =============================================================================
# 4. OLS COMPARISON
# =============================================================================

def run_ols(df: pd.DataFrame, outcome: str, endogenous: str,
            controls: list = None) -> dict:
    """
    Run OLS for comparison with IV.
    """
    import statsmodels.api as sm

    if controls is None:
        controls = []

    X = df[[endogenous] + controls + ['const']]
    y = df[outcome]

    model = sm.OLS(y, X)
    result = model.fit(cov_type='HC1')

    return {
        'method': 'OLS',
        'coefficient': result.params[endogenous],
        'std_error': result.bse[endogenous],
        'p_value': result.pvalues[endogenous],
        'ci_lower': result.conf_int().loc[endogenous, 0],
        'ci_upper': result.conf_int().loc[endogenous, 1],
        'n_obs': int(result.nobs),
        'r_squared': result.rsquared
    }


# =============================================================================
# 5. ANDERSON-RUBIN CONFIDENCE INTERVAL
# =============================================================================

def anderson_rubin_ci(df: pd.DataFrame, outcome: str, endogenous: str,
                      instruments: list, controls: list = None,
                      alpha: float = 0.05,
                      grid_points: int = 500) -> tuple:
    """
    Compute Anderson-Rubin confidence interval (weak IV robust).

    Parameters
    ----------
    df : pd.DataFrame
        Data
    outcome, endogenous : str
        Variable names
    instruments : list
        Instrument names
    controls : list, optional
        Control variable names
    alpha : float
        Significance level
    grid_points : int
        Number of points in grid search

    Returns
    -------
    tuple
        (ci_lower, ci_upper)
    """
    import statsmodels.api as sm

    if controls is None:
        controls = []

    # Estimate range for grid
    ols_result = run_ols(df, outcome, endogenous, controls)
    ols_coef = ols_result['coefficient']
    ols_se = ols_result['std_error']

    # Grid around OLS estimate
    beta_range = max(abs(ols_coef) * 3, ols_se * 10)
    beta_grid = np.linspace(ols_coef - beta_range, ols_coef + beta_range, grid_points)

    ar_stats = []
    X_vars = instruments + controls + ['const']

    for beta in beta_grid:
        # Construct Y - beta*X
        y_adjusted = df[outcome] - beta * df[endogenous]

        # Regress on instruments and controls
        X = df[X_vars]
        model = sm.OLS(y_adjusted, X).fit()

        # F-test for instruments
        n_inst = len(instruments)
        r_matrix = np.zeros((n_inst, len(X_vars)))
        for i, inst in enumerate(instruments):
            r_matrix[i, X_vars.index(inst)] = 1

        f_test = model.f_test(r_matrix)
        fvalue = f_test.fvalue
        if hasattr(fvalue, '__iter__'):
            fval = float(fvalue[0][0]) if hasattr(fvalue[0], '__iter__') else float(fvalue[0])
        else:
            fval = float(fvalue)
        ar_stats.append(fval)

    # Critical value
    df1 = len(instruments)
    df2 = len(df) - len(X_vars)
    critical_value = stats.f.ppf(1 - alpha, df1, df2)

    # Find confidence set
    ar_stats = np.array(ar_stats)
    in_ci = ar_stats < critical_value

    if in_ci.any():
        ci_lower = beta_grid[in_ci].min()
        ci_upper = beta_grid[in_ci].max()
    else:
        ci_lower = np.nan
        ci_upper = np.nan

    return ci_lower, ci_upper


# =============================================================================
# 6. REPORTING
# =============================================================================

def print_results_table(results_list: list, title: str = "IV Estimation Results"):
    """
    Print publication-style results table.
    """
    print("\n" + "=" * 75)
    print(title.center(75))
    print("=" * 75)

    print(f"{'Method':<12} {'Coef':>10} {'SE':>10} {'p-val':>10} {'95% CI':>25} {'N':>6}")
    print("-" * 75)

    for res in results_list:
        coef = res['coefficient']
        se = res['std_error']
        pval = res['p_value']
        ci = f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]"
        n = res['n_obs']
        method = res['method']

        # Significance stars
        stars = ""
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"

        print(f"{method:<12} {coef:>9.4f}{stars} {se:>10.4f} {pval:>10.4f} {ci:>25} {n:>6}")

    print("-" * 75)
    print("Notes: *** p<0.01, ** p<0.05, * p<0.1. Robust standard errors.")
    print("=" * 75)


def print_diagnostics(first_stage: dict, iv_result: dict):
    """
    Print diagnostic tests summary.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC TESTS".center(60))
    print("=" * 60)

    # First-stage
    print("\n1. FIRST STAGE (Instrument Relevance)")
    print("-" * 40)
    print(f"   F-statistic: {first_stage['f_statistic']:.2f}")
    print(f"   P-value: {first_stage['f_pvalue']:.4f}")
    print(f"   Stock-Yogo 10% CV: {first_stage['stock_yogo_cv']:.2f}")
    if first_stage['is_strong']:
        print("   Status: STRONG (F > CV)")
    else:
        print("   Status: WEAK (F < CV) - Use LIML or AR inference")

    # Hausman
    diag = iv_result.get('diagnostics', {})
    if 'hausman_stat' in diag:
        print("\n2. WU-HAUSMAN TEST (Endogeneity)")
        print("-" * 40)
        print(f"   Statistic: {diag['hausman_stat']:.4f}")
        print(f"   P-value: {diag['hausman_pval']:.4f}")
        if diag['hausman_pval'] < 0.05:
            print("   Status: REJECT H0 - Endogeneity present, IV needed")
        else:
            print("   Status: FAIL TO REJECT H0 - OLS may be consistent")

    # Sargan
    if 'sargan_stat' in diag:
        print("\n3. SARGAN-HANSEN TEST (Overidentification)")
        print("-" * 40)
        print(f"   J-statistic: {diag['sargan_stat']:.4f}")
        print(f"   P-value: {diag['sargan_pval']:.4f}")
        if diag['sargan_pval'] < 0.05:
            print("   Status: REJECT H0 - Instruments may be invalid")
        else:
            print("   Status: FAIL TO REJECT H0 - Instruments valid")

    print("=" * 60)


def generate_latex_table(results_list: list, save_path: str = None) -> str:
    """
    Generate LaTeX table for publication.
    """
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Instrumental Variables Estimation Results}
\label{tab:iv_results}
\begin{tabular}{lcccc}
\toprule
& (1) & (2) & (3) & (4) \\
& OLS & 2SLS & LIML & AR CI \\
\midrule
"""

    for res in results_list:
        method = res['method']
        coef = res['coefficient']
        se = res['std_error']
        pval = res['p_value']
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""

        if method == 'OLS':
            latex += f"Effect & {coef:.4f}{stars} & & & \\\\\n"
            latex += f" & ({se:.4f}) & & & \\\\\n"
        elif method == '2SLS':
            latex += f" & & {coef:.4f}{stars} & & \\\\\n"
            latex += f" & & ({se:.4f}) & & \\\\\n"
        elif method == 'LIML':
            latex += f" & & & {coef:.4f}{stars} & \\\\\n"
            latex += f" & & & ({se:.4f}) & \\\\\n"

    latex += r"""
\midrule
First-stage F & - & XX.XX & XX.XX & - \\
Observations & N & N & N & N \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Robust standard errors in parentheses.
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
                      outcome: str = 'y',
                      endogenous: str = 'x',
                      instruments: list = None,
                      controls: list = None,
                      true_effect: float = None,
                      output_dir: str = None) -> dict:
    """
    Run complete IV analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Data (if None, uses simulated data)
    outcome : str
        Outcome variable name
    endogenous : str
        Endogenous variable name
    instruments : list
        List of instrument names
    controls : list
        List of control variable names
    true_effect : float
        True effect (for simulation comparison)
    output_dir : str
        Directory to save outputs

    Returns
    -------
    dict
        All analysis results
    """
    all_results = {}

    # Setup
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    # Use simulated data if none provided
    if df is None:
        print("Using simulated data for demonstration...")
        df = simulate_iv_data(n=1000, beta_true=0.5, gamma=0.8, rho=0.6)
        instruments = ['z1', 'z2']
        controls = ['control']
        true_effect = 0.5

    if instruments is None:
        raise ValueError("Must specify instruments")

    print("\n" + "=" * 70)
    print("INSTRUMENTAL VARIABLES ANALYSIS".center(70))
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if true_effect is not None:
        print(f"True Effect: {true_effect}")

    # 1. First Stage
    print("\n[1/5] First Stage Analysis...")
    first_stage = first_stage_analysis(df, endogenous, instruments, controls)
    all_results['first_stage'] = first_stage

    print(f"  - F-statistic: {first_stage['f_statistic']:.2f}")
    print(f"  - Stock-Yogo CV: {first_stage['stock_yogo_cv']:.2f}")
    print(f"  - Instrument strength: {'STRONG' if first_stage['is_strong'] else 'WEAK'}")

    # 2. OLS (biased)
    print("\n[2/5] OLS Estimation (biased if endogenous)...")
    ols_result = run_ols(df, outcome, endogenous, controls)
    all_results['ols'] = ols_result
    print(f"  - OLS coefficient: {ols_result['coefficient']:.4f}")

    # 3. 2SLS
    print("\n[3/5] 2SLS Estimation...")
    iv_2sls = run_iv_estimation(df, outcome, endogenous, instruments, controls, '2sls')
    all_results['2sls'] = iv_2sls
    print(f"  - 2SLS coefficient: {iv_2sls['coefficient']:.4f}")

    # 4. LIML
    print("\n[4/5] LIML Estimation (weak IV robust)...")
    iv_liml = run_iv_estimation(df, outcome, endogenous, instruments, controls, 'liml')
    all_results['liml'] = iv_liml
    print(f"  - LIML coefficient: {iv_liml['coefficient']:.4f}")

    # 5. Anderson-Rubin CI
    print("\n[5/5] Anderson-Rubin Confidence Interval...")
    ar_ci = anderson_rubin_ci(df, outcome, endogenous, instruments, controls)
    all_results['ar_ci'] = ar_ci
    print(f"  - AR 95% CI: [{ar_ci[0]:.4f}, {ar_ci[1]:.4f}]")

    # Print summary tables
    results_list = [ols_result, iv_2sls, iv_liml]
    print_results_table(results_list)
    print_diagnostics(first_stage, iv_2sls)

    # Comparison with true effect
    if true_effect is not None:
        print("\n" + "=" * 60)
        print("COMPARISON WITH TRUE EFFECT".center(60))
        print("=" * 60)
        print(f"True effect: {true_effect:.4f}")
        print(f"OLS bias: {ols_result['coefficient'] - true_effect:.4f}")
        print(f"2SLS bias: {iv_2sls['coefficient'] - true_effect:.4f}")
        print(f"LIML bias: {iv_liml['coefficient'] - true_effect:.4f}")
        print("=" * 60)

    # Save outputs
    if output_dir:
        latex_path = str(output_dir / 'iv_table.tex')
        generate_latex_table(results_list, latex_path)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE".center(70))
    print("=" * 70)

    return all_results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Instrumental Variables analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python iv_analysis_pipeline.py --demo
    python iv_analysis_pipeline.py --data data.csv --outcome y --endogenous x --instruments z1,z2
    python iv_analysis_pipeline.py --data data.csv --outcome wage --endogenous education --instruments distance --controls experience,female
        """
    )

    parser.add_argument('--demo', action='store_true',
                        help='Run with simulated data')
    parser.add_argument('--data', '-d', type=str,
                        help='Path to CSV data file')
    parser.add_argument('--outcome', '-y', type=str, default='y',
                        help='Outcome variable column')
    parser.add_argument('--endogenous', '-x', type=str, default='x',
                        help='Endogenous variable column')
    parser.add_argument('--instruments', '-z', type=str,
                        help='Comma-separated list of instruments')
    parser.add_argument('--controls', type=str,
                        help='Comma-separated list of control variables')
    parser.add_argument('--output', '-o', type=str, default='./iv_results',
                        help='Output directory')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.demo or args.data is None:
        results = run_full_analysis(output_dir=args.output)
    else:
        df = pd.read_csv(args.data)
        df['const'] = 1

        instruments = args.instruments.split(',') if args.instruments else None
        controls = args.controls.split(',') if args.controls else None

        results = run_full_analysis(
            df=df,
            outcome=args.outcome,
            endogenous=args.endogenous,
            instruments=instruments,
            controls=controls,
            output_dir=args.output
        )
