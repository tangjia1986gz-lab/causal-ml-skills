#!/usr/bin/env python3
"""
DDML Analysis Pipeline - Self-Contained Double/Debiased Machine Learning

This script provides a complete DDML analysis workflow using the doubleml package:
- PLR (Partially Linear Regression) for continuous/binary treatment
- IRM (Interactive Regression Model) for binary treatment with heterogeneity
- Multiple learner comparison for sensitivity analysis
- Nuisance model diagnostics
- Publication-ready output

Usage:
    python ddml_analysis_pipeline.py --demo
    python ddml_analysis_pipeline.py --data data.csv --outcome y --treatment d --controls "x1,x2,x3"
    python ddml_analysis_pipeline.py --data data.csv --outcome y --treatment d --model irm

Dependencies:
    pip install doubleml scikit-learn pandas numpy matplotlib

Reference:
    Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for
    Treatment and Structural Parameters. The Econometrics Journal, 21(1), C1-C68.
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

# Set non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# Data Simulation
# =============================================================================

def simulate_ddml_data(
    n: int = 1000,
    p: int = 20,
    treatment_effect: float = 0.5,
    treatment_type: str = 'continuous',
    nonlinear: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate data for DDML analysis with known treatment effect.

    Parameters
    ----------
    n : int
        Sample size
    p : int
        Number of control variables
    treatment_effect : float
        True causal effect of treatment on outcome
    treatment_type : str
        'continuous' or 'binary'
    nonlinear : bool
        If True, include nonlinear confounding
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Simulated data with columns: outcome, treatment, X_1, ..., X_p
    """
    np.random.seed(seed)

    # Generate control variables
    X = np.random.randn(n, p)

    # Coefficients for confounding
    beta_y = np.random.randn(p) * 0.5  # Effect on outcome
    beta_d = np.random.randn(p) * 0.3  # Effect on treatment

    # Nonlinear confounding
    if nonlinear:
        confound_y = np.sum(X[:, :5]**2, axis=1) * 0.2  # Quadratic effect
        confound_d = np.sin(X[:, 0]) * 0.3
    else:
        confound_y = 0
        confound_d = 0

    # Generate treatment
    if treatment_type == 'binary':
        propensity = 1 / (1 + np.exp(-(X @ beta_d + confound_d)))
        D = (np.random.uniform(size=n) < propensity).astype(float)
    else:
        D = X @ beta_d + confound_d + np.random.randn(n) * 0.5

    # Generate outcome
    Y = treatment_effect * D + X @ beta_y + confound_y + np.random.randn(n)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'X_{i+1}' for i in range(p)])
    df['treatment'] = D
    df['outcome'] = Y

    return df


# =============================================================================
# DDML Estimation Functions
# =============================================================================

def run_plr(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str],
    learner_l: str = 'lasso',
    learner_m: str = 'lasso',
    n_folds: int = 5,
    n_rep: int = 1
) -> Dict[str, Any]:
    """
    Run Partially Linear Regression (PLR) model.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    outcome : str
        Outcome variable name
    treatment : str
        Treatment variable name
    controls : List[str]
        List of control variable names
    learner_l : str
        Learner for outcome nuisance function ('lasso', 'ridge', 'rf')
    learner_m : str
        Learner for treatment nuisance function
    n_folds : int
        Number of cross-fitting folds
    n_rep : int
        Number of repetitions

    Returns
    -------
    dict
        Results including effect, se, ci, diagnostics
    """
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.ensemble import RandomForestRegressor
    import doubleml as dml
    from doubleml import DoubleMLData, DoubleMLPLR

    # Create DoubleML data
    dml_data = DoubleMLData(
        df,
        y_col=outcome,
        d_cols=treatment,
        x_cols=controls
    )

    # Get learners
    def get_learner(name: str, task: str = 'regression'):
        if name == 'lasso':
            return LassoCV(cv=5, n_alphas=50, max_iter=10000)
        elif name == 'ridge':
            return RidgeCV(cv=5)
        elif name == 'rf':
            return RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
        else:
            return LassoCV(cv=5)

    ml_l = get_learner(learner_l)
    ml_m = get_learner(learner_m)

    # Estimate PLR
    model = DoubleMLPLR(
        dml_data,
        ml_l=ml_l,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=n_rep,
        score='partialling out'
    )
    model.fit()

    # Extract results
    ci = model.confint()

    return {
        'model': 'PLR',
        'effect': float(model.coef[0]),
        'se': float(model.se[0]),
        't_stat': float(model.t_stat[0]),
        'p_value': float(model.pval[0]),
        'ci_lower': float(ci.iloc[0, 0]),
        'ci_upper': float(ci.iloc[0, 1]),
        'learner_l': learner_l,
        'learner_m': learner_m,
        'n_folds': n_folds,
        'n_rep': n_rep,
        'n_obs': len(df),
        'dml_model': model
    }


def run_irm(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str],
    learner_g: str = 'rf',
    learner_m: str = 'logistic',
    n_folds: int = 5,
    n_rep: int = 1,
    score: str = 'ATE',
    trimming: float = 0.01
) -> Dict[str, Any]:
    """
    Run Interactive Regression Model (IRM) for binary treatment.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    outcome : str
        Outcome variable name
    treatment : str
        Treatment variable name (must be binary)
    controls : List[str]
        List of control variable names
    learner_g : str
        Learner for outcome nuisance function
    learner_m : str
        Learner for propensity score
    n_folds : int
        Number of cross-fitting folds
    n_rep : int
        Number of repetitions
    score : str
        'ATE' or 'ATTE'
    trimming : float
        Propensity score trimming threshold

    Returns
    -------
    dict
        Results including effect, se, ci, diagnostics
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    import doubleml as dml
    from doubleml import DoubleMLData, DoubleMLIRM

    # Check binary treatment
    unique_vals = df[treatment].unique()
    if len(unique_vals) != 2 or not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"IRM requires binary treatment (0/1). Found: {unique_vals}")

    # Create DoubleML data
    dml_data = DoubleMLData(
        df,
        y_col=outcome,
        d_cols=treatment,
        x_cols=controls
    )

    # Get learners
    if learner_g == 'rf':
        ml_g = RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
    else:
        from sklearn.linear_model import LassoCV
        ml_g = LassoCV(cv=5)

    if learner_m == 'logistic':
        ml_m = LogisticRegressionCV(cv=5, max_iter=1000)
    else:
        ml_m = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)

    # Estimate IRM
    model = DoubleMLIRM(
        dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=n_rep,
        score=score,
        trimming_threshold=trimming
    )
    model.fit()

    # Extract results
    ci = model.confint()

    return {
        'model': 'IRM',
        'effect': float(model.coef[0]),
        'se': float(model.se[0]),
        't_stat': float(model.t_stat[0]),
        'p_value': float(model.pval[0]),
        'ci_lower': float(ci.iloc[0, 0]),
        'ci_upper': float(ci.iloc[0, 1]),
        'learner_g': learner_g,
        'learner_m': learner_m,
        'score': score,
        'trimming': trimming,
        'n_folds': n_folds,
        'n_rep': n_rep,
        'n_obs': len(df),
        'dml_model': model
    }


def compare_learners(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str],
    model_type: str = 'PLR',
    n_folds: int = 5
) -> pd.DataFrame:
    """
    Compare multiple ML learner specifications.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    outcome : str
        Outcome variable name
    treatment : str
        Treatment variable name
    controls : List[str]
        List of control variable names
    model_type : str
        'PLR' or 'IRM'
    n_folds : int
        Number of cross-fitting folds

    Returns
    -------
    pd.DataFrame
        Comparison table with effects across learners
    """
    if model_type == 'PLR':
        learner_configs = [
            ('Lasso', 'lasso', 'lasso'),
            ('Ridge', 'ridge', 'ridge'),
            ('Random Forest', 'rf', 'rf'),
            ('Lasso-RF', 'lasso', 'rf'),
        ]
    else:
        learner_configs = [
            ('RF-Logistic', 'rf', 'logistic'),
            ('RF-RF', 'rf', 'rf_clf'),
            ('Lasso-Logistic', 'lasso', 'logistic'),
        ]

    results = []
    for name, l1, l2 in learner_configs:
        try:
            if model_type == 'PLR':
                res = run_plr(df, outcome, treatment, controls, l1, l2, n_folds)
            else:
                res = run_irm(df, outcome, treatment, controls, l1, l2, n_folds)

            results.append({
                'Specification': name,
                'Effect': res['effect'],
                'SE': res['se'],
                'CI_lower': res['ci_lower'],
                'CI_upper': res['ci_upper'],
                'p_value': res['p_value']
            })
        except Exception as e:
            print(f"Warning: {name} failed: {e}")

    return pd.DataFrame(results)


# =============================================================================
# Diagnostics
# =============================================================================

def nuisance_diagnostics(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str]
) -> Dict[str, float]:
    """
    Compute diagnostics for nuisance model quality.

    Returns R-squared for outcome and treatment models.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import r2_score

    X = df[controls].values
    Y = df[outcome].values
    D = df[treatment].values

    # Outcome model R2
    y_pred = cross_val_predict(LassoCV(cv=5, max_iter=10000), X, Y, cv=5)
    r2_y = r2_score(Y, y_pred)

    # Treatment model R2
    d_pred = cross_val_predict(LassoCV(cv=5, max_iter=10000), X, D, cv=5)
    r2_d = r2_score(D, d_pred)

    return {
        'r2_outcome': r2_y,
        'r2_treatment': r2_d
    }


def propensity_overlap(
    df: pd.DataFrame,
    treatment: str,
    controls: List[str]
) -> Dict[str, Any]:
    """
    Check propensity score overlap for binary treatment.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import cross_val_predict

    X = df[controls].values
    D = df[treatment].values

    # Cross-validated propensity scores
    clf = LogisticRegressionCV(cv=5, max_iter=1000)
    ps = cross_val_predict(clf, X, D, cv=5, method='predict_proba')[:, 1]

    return {
        'ps_min': float(ps.min()),
        'ps_max': float(ps.max()),
        'ps_mean': float(ps.mean()),
        'n_extreme_low': int((ps < 0.01).sum()),
        'n_extreme_high': int((ps > 0.99).sum()),
        'ps_values': ps
    }


# =============================================================================
# Output Functions
# =============================================================================

def print_results(result: Dict[str, Any], true_effect: Optional[float] = None):
    """Print formatted results."""
    print("\n" + "="*60)
    print(f"DDML ESTIMATION RESULTS ({result['model']})")
    print("="*60)

    print(f"\nSample size: {result['n_obs']}")
    print(f"Cross-fitting: {result['n_folds']} folds, {result['n_rep']} repetitions")

    if result['model'] == 'PLR':
        print(f"Learners: ml_l={result['learner_l']}, ml_m={result['learner_m']}")
    else:
        print(f"Learners: ml_g={result['learner_g']}, ml_m={result['learner_m']}")
        print(f"Estimand: {result['score']}")

    print("\n--- Causal Effect ---")
    print(f"Effect estimate: {result['effect']:.4f}")
    print(f"Standard error: {result['se']:.4f}")
    print(f"t-statistic: {result['t_stat']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    if true_effect is not None:
        bias = result['effect'] - true_effect
        print(f"\n--- Comparison with True Effect ---")
        print(f"True effect: {true_effect:.4f}")
        print(f"Bias: {bias:.4f}")

        # Check if CI covers true effect
        covers = result['ci_lower'] <= true_effect <= result['ci_upper']
        print(f"CI covers true effect: {'Yes' if covers else 'No'}")

    print("="*60)


def generate_latex_table(comparison_df: pd.DataFrame, save_path: Optional[str] = None) -> str:
    """Generate publication-quality LaTeX table."""

    latex = r"""
\begin{table}[htbp]
\centering
\caption{DDML Estimation Results: Sensitivity to ML Specification}
\label{tab:ddml_sensitivity}
\begin{tabular}{lcccc}
\toprule
Specification & Effect & SE & 95\% CI & p-value \\
\midrule
"""

    for _, row in comparison_df.iterrows():
        stars = ''
        if row['p_value'] < 0.01:
            stars = '***'
        elif row['p_value'] < 0.05:
            stars = '**'
        elif row['p_value'] < 0.10:
            stars = '*'

        latex += f"{row['Specification']} & {row['Effect']:.4f}{stars} & {row['SE']:.4f} & "
        latex += f"[{row['CI_lower']:.4f}, {row['CI_upper']:.4f}] & {row['p_value']:.4f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: *** p<0.01, ** p<0.05, * p<0.1. Standard errors computed via cross-fitting.
\end{tablenotes}
\end{table}
"""

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {save_path}")

    return latex


def plot_sensitivity(
    comparison_df: pd.DataFrame,
    true_effect: Optional[float] = None,
    save_path: Optional[str] = None
):
    """Plot coefficient comparison across specifications."""

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(comparison_df))

    # Plot point estimates with error bars
    ax.errorbar(
        comparison_df['Effect'],
        y_pos,
        xerr=[comparison_df['Effect'] - comparison_df['CI_lower'],
              comparison_df['CI_upper'] - comparison_df['Effect']],
        fmt='o',
        capsize=5,
        capthick=2,
        markersize=8,
        color='navy'
    )

    # Add true effect line if provided
    if true_effect is not None:
        ax.axvline(true_effect, color='red', linestyle='--', linewidth=2, label=f'True Effect = {true_effect}')

    # Add zero line
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['Specification'])
    ax.set_xlabel('Effect Estimate (95% CI)', fontsize=12)
    ax.set_title('DDML Sensitivity Analysis: ML Learner Comparison', fontsize=14)

    if true_effect is not None:
        ax.legend(loc='best')

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
    treatment: str,
    controls: List[str],
    model_type: str = 'PLR',
    true_effect: Optional[float] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete DDML analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    outcome : str
        Outcome variable name
    treatment : str
        Treatment variable name
    controls : List[str]
        List of control variable names
    model_type : str
        'PLR' or 'IRM'
    true_effect : float, optional
        True effect for simulation comparison
    output_dir : str, optional
        Directory for output files

    Returns
    -------
    dict
        Complete analysis results
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("DOUBLE/DEBIASED MACHINE LEARNING ANALYSIS")
    print("="*70)

    # Step 1: Data summary
    print("\n--- Step 1: Data Summary ---")
    print(f"Observations: {len(df)}")
    print(f"Treatment variable: {treatment}")
    print(f"Outcome variable: {outcome}")
    print(f"Control variables: {len(controls)}")

    if model_type == 'IRM':
        unique_d = df[treatment].unique()
        print(f"Treatment values: {sorted(unique_d)}")
        print(f"Treatment rate: {df[treatment].mean():.2%}")

    # Step 2: Nuisance diagnostics
    print("\n--- Step 2: Nuisance Model Diagnostics ---")
    diagnostics = nuisance_diagnostics(df, outcome, treatment, controls)
    print(f"R2 (outcome ~ X): {diagnostics['r2_outcome']:.4f}")
    print(f"R2 (treatment ~ X): {diagnostics['r2_treatment']:.4f}")

    if diagnostics['r2_outcome'] < 0.1:
        print("WARNING: Low outcome model fit - consider different learner or more controls")
    if diagnostics['r2_treatment'] < 0.1:
        print("WARNING: Low treatment model fit - check for selection on observables")

    # Step 3: Propensity overlap (IRM only)
    if model_type == 'IRM':
        print("\n--- Step 3: Propensity Score Overlap ---")
        overlap = propensity_overlap(df, treatment, controls)
        print(f"Propensity range: [{overlap['ps_min']:.4f}, {overlap['ps_max']:.4f}]")
        print(f"Extreme low (<0.01): {overlap['n_extreme_low']}")
        print(f"Extreme high (>0.99): {overlap['n_extreme_high']}")

    # Step 4: Main estimation
    print(f"\n--- Step 4: {model_type} Estimation ---")
    if model_type == 'PLR':
        main_result = run_plr(df, outcome, treatment, controls)
    else:
        main_result = run_irm(df, outcome, treatment, controls)

    print_results(main_result, true_effect)

    # Step 5: Sensitivity analysis
    print("\n--- Step 5: Sensitivity Analysis ---")
    comparison = compare_learners(df, outcome, treatment, controls, model_type)
    print("\nComparison across ML learners:")
    print(comparison.to_string(index=False))

    effect_range = comparison['Effect'].max() - comparison['Effect'].min()
    mean_effect = comparison['Effect'].mean()
    print(f"\nEffect range across specifications: {effect_range:.4f}")
    print(f"Mean effect: {mean_effect:.4f}")
    print(f"Coefficient of variation: {comparison['Effect'].std() / abs(mean_effect):.2%}")

    # Step 6: Output
    if output_dir:
        print("\n--- Step 6: Generating Output ---")

        # LaTeX table
        latex_path = output_dir / 'ddml_results.tex'
        generate_latex_table(comparison, str(latex_path))

        # Sensitivity plot
        plot_path = output_dir / 'ddml_sensitivity.png'
        plot_sensitivity(comparison, true_effect, str(plot_path))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return {
        'main_result': main_result,
        'comparison': comparison,
        'diagnostics': diagnostics,
        'true_effect': true_effect
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DDML Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with simulated data
    python ddml_analysis_pipeline.py --demo

    # Run with real data
    python ddml_analysis_pipeline.py --data wages.csv --outcome log_wage --treatment education \\
        --controls "age,experience,female"

    # Run IRM for binary treatment
    python ddml_analysis_pipeline.py --data data.csv --outcome earnings --treatment training \\
        --model irm --controls "age,education"
"""
    )

    parser.add_argument('--demo', action='store_true', help='Run demo with simulated data')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--outcome', type=str, help='Outcome variable name')
    parser.add_argument('--treatment', type=str, help='Treatment variable name')
    parser.add_argument('--controls', type=str, help='Comma-separated control variable names')
    parser.add_argument('--model', type=str, default='PLR', choices=['PLR', 'IRM'],
                        help='Model type: PLR or IRM')
    parser.add_argument('--output', type=str, help='Output directory')

    args = parser.parse_args()

    if args.demo:
        # Run demo with simulated data
        print("Running DDML demo with simulated data...")

        # Simulate data
        true_effect = 0.5
        df = simulate_ddml_data(
            n=1000,
            p=20,
            treatment_effect=true_effect,
            treatment_type='continuous',
            nonlinear=True,
            seed=42
        )

        outcome = 'outcome'
        treatment = 'treatment'
        controls = [f'X_{i+1}' for i in range(20)]

        # Run analysis
        run_full_analysis(
            df=df,
            outcome=outcome,
            treatment=treatment,
            controls=controls,
            model_type='PLR',
            true_effect=true_effect,
            output_dir=args.output
        )

    elif args.data and args.outcome and args.treatment:
        # Run with provided data
        df = pd.read_csv(args.data)

        if args.controls:
            controls = [c.strip() for c in args.controls.split(',')]
        else:
            # Use all columns except outcome and treatment as controls
            controls = [c for c in df.columns if c not in [args.outcome, args.treatment]]

        run_full_analysis(
            df=df,
            outcome=args.outcome,
            treatment=args.treatment,
            controls=controls,
            model_type=args.model,
            output_dir=args.output
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
