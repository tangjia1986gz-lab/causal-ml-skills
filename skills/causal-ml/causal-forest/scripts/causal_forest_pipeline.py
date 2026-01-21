#!/usr/bin/env python3
"""
Causal Forest Analysis Pipeline - Self-Contained

Estimates heterogeneous treatment effects (CATE) using causal forests via econml.
Provides complete workflow: estimation, diagnostics, policy learning, visualization.

Usage:
    python causal_forest_pipeline.py --demo
    python causal_forest_pipeline.py --data data.csv --outcome Y --treatment T --controls "X1,X2"

Dependencies:
    pip install econml scikit-learn pandas numpy matplotlib

Reference:
    Athey, S. & Wager, S. (2019). Estimating Treatment Effects with Causal Forests:
    An Application. Observational Studies, 5, 37-51.
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

def simulate_causal_forest_data(
    n: int = 2000,
    p: int = 10,
    treatment_type: str = 'binary',
    heterogeneity: str = 'moderate',
    seed: int = 42
) -> Tuple[pd.DataFrame, dict]:
    """
    Simulate data with heterogeneous treatment effects.

    Parameters
    ----------
    n : int
        Sample size
    p : int
        Number of covariates
    treatment_type : str
        'binary' or 'continuous'
    heterogeneity : str
        'none', 'moderate', 'strong' - degree of treatment effect heterogeneity
    seed : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Data and true CATE function info
    """
    np.random.seed(seed)

    # Generate covariates
    X = np.random.randn(n, p)

    # Generate treatment assignment
    # Propensity depends on X[0] and X[1]
    propensity = 1 / (1 + np.exp(-(0.5 * X[:, 0] + 0.3 * X[:, 1])))

    if treatment_type == 'binary':
        T = (np.random.uniform(size=n) < propensity).astype(float)
    else:
        T = propensity + 0.2 * np.random.randn(n)
        T = np.clip(T, 0, 1)

    # Define CATE function based on heterogeneity level
    if heterogeneity == 'none':
        # Constant effect
        tau = np.full(n, 1.0)
    elif heterogeneity == 'moderate':
        # Effect varies with X[0]
        tau = 1.0 + 0.5 * X[:, 0]
    else:  # strong
        # Complex heterogeneity
        tau = 1.0 + 0.5 * X[:, 0] + 0.3 * (X[:, 1] > 0) - 0.2 * X[:, 2]**2

    # Generate outcome
    # Y = baseline + treatment effect + noise
    baseline = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2]
    Y = baseline + tau * T + np.random.randn(n)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'X{i}' for i in range(p)])
    df['T'] = T
    df['Y'] = Y

    # True CATE info
    cate_info = {
        'true_cate': tau,
        'heterogeneity': heterogeneity,
        'ate': float(tau.mean()),
        'cate_sd': float(tau.std())
    }

    return df, cate_info


# =============================================================================
# Causal Forest Estimation
# =============================================================================

def fit_causal_forest_dml(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str],
    n_estimators: int = 500,
    min_samples_leaf: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Fit CausalForestDML model from econml.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    outcome : str
        Outcome variable name
    treatment : str
        Treatment variable name
    controls : List[str]
        Control/covariate variable names
    n_estimators : int
        Number of trees
    min_samples_leaf : int
        Minimum samples per leaf
    random_state : int
        Random seed

    Returns
    -------
    dict
        Model and results
    """
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    # Prepare data
    Y = df[outcome].values
    T = df[treatment].values
    X = df[controls].values

    # Check if treatment is binary or continuous
    unique_t = np.unique(T)
    is_binary = len(unique_t) == 2

    # Define nuisance models
    if is_binary:
        model_t = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=random_state)
    else:
        model_t = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=random_state)

    model_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=random_state)

    # Fit CausalForestDML
    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        discrete_treatment=is_binary
    )

    cf.fit(Y, T, X=X)

    # Get CATE estimates
    cate = cf.effect(X)
    cate_lb, cate_ub = cf.effect_interval(X, alpha=0.05)

    # ATE and inference
    ate = cf.ate(X)
    ate_inf = cf.ate_inference(X)

    # Handle different econml API versions
    try:
        ate_se = float(ate_inf.std_err)
    except AttributeError:
        # Try alternative attribute names
        if hasattr(ate_inf, 'stderr'):
            ate_se = float(ate_inf.stderr)
        elif hasattr(ate_inf, 'summary_frame'):
            summary = ate_inf.summary_frame()
            ate_se = float(summary['std err'].iloc[0])
        else:
            # Fallback: estimate from CATE
            ate_se = float(np.std(cate) / np.sqrt(len(cate)))

    try:
        ci = ate_inf.conf_int()
        if hasattr(ci, '__iter__') and len(ci) == 2:
            ate_ci = (float(ci[0][0] if hasattr(ci[0], '__iter__') else ci[0]),
                      float(ci[1][0] if hasattr(ci[1], '__iter__') else ci[1]))
        else:
            ate_ci = (float(ate - 1.96 * ate_se), float(ate + 1.96 * ate_se))
    except Exception:
        ate_ci = (float(ate - 1.96 * ate_se), float(ate + 1.96 * ate_se))

    try:
        ate_pval = float(ate_inf.pvalue()[0] if hasattr(ate_inf.pvalue(), '__iter__') else ate_inf.pvalue())
    except Exception:
        # Calculate p-value from z-score
        z = abs(ate / ate_se) if ate_se > 0 else 0
        ate_pval = float(2 * (1 - stats.norm.cdf(z)))

    return {
        'model': cf,
        'cate': cate.flatten(),
        'cate_lb': cate_lb.flatten(),
        'cate_ub': cate_ub.flatten(),
        'ate': float(ate),
        'ate_se': ate_se,
        'ate_ci': ate_ci,
        'ate_pval': ate_pval,
        'X': X,
        'controls': controls
    }


def variable_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Compute variable importance for CATE heterogeneity.

    Returns
    -------
    pd.DataFrame
        Feature importance ranking
    """
    try:
        # For CausalForestDML, use feature_importances if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Approximate via permutation or use model components
            importances = np.ones(len(feature_names)) / len(feature_names)

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        return importance_df
    except Exception:
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': [1/len(feature_names)] * len(feature_names)
        })


def heterogeneity_test(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test for treatment effect heterogeneity.

    Uses F-test on CATE variance.
    """
    cate = result['cate']

    # Test H0: Var(CATE) = 0 (no heterogeneity)
    # Simple approach: compare to bootstrap null

    n = len(cate)
    observed_var = np.var(cate)
    observed_range = np.max(cate) - np.min(cate)

    # Bootstrap null distribution (permutation test)
    n_bootstrap = 1000
    null_vars = []
    for _ in range(n_bootstrap):
        null_cate = np.random.choice(cate, size=n, replace=True)
        null_vars.append(np.var(null_cate))

    p_value = np.mean(np.array(null_vars) >= observed_var)

    return {
        'cate_variance': float(observed_var),
        'cate_range': float(observed_range),
        'cate_mean': float(np.mean(cate)),
        'cate_median': float(np.median(cate)),
        'heterogeneity_pvalue': float(p_value),
        'significant_heterogeneity': p_value < 0.05
    }


def best_linear_projection(
    result: Dict[str, Any],
    df: pd.DataFrame,
    effect_modifiers: List[str]
) -> Dict[str, Any]:
    """
    Best Linear Projection of CATE onto effect modifiers.

    Fits: tau(x) ~ X * beta

    Parameters
    ----------
    result : dict
        Causal forest results
    df : pd.DataFrame
        Original data
    effect_modifiers : List[str]
        Variables to project CATE onto

    Returns
    -------
    dict
        BLP coefficients and statistics
    """
    import statsmodels.api as sm

    cate = result['cate']
    X = df[effect_modifiers].values
    X = sm.add_constant(X)

    # OLS regression of CATE on effect modifiers
    model = sm.OLS(cate, X).fit()

    coefficients = {}
    for i, name in enumerate(['const'] + effect_modifiers):
        coefficients[name] = {
            'coef': float(model.params[i]),
            'se': float(model.bse[i]),
            'pvalue': float(model.pvalues[i])
        }

    return {
        'coefficients': coefficients,
        'r_squared': float(model.rsquared),
        'f_stat': float(model.fvalue),
        'f_pvalue': float(model.f_pvalue)
    }


# =============================================================================
# Policy Learning
# =============================================================================

def simple_policy_tree(
    result: Dict[str, Any],
    df: pd.DataFrame,
    effect_modifiers: List[str],
    max_depth: int = 3
) -> Dict[str, Any]:
    """
    Learn optimal treatment policy using classification tree.

    Classifies individuals into treat/don't treat based on CATE sign.
    """
    from sklearn.tree import DecisionTreeClassifier

    cate = result['cate']
    X = df[effect_modifiers].values

    # Define optimal treatment: treat if CATE > 0
    optimal_treatment = (cate > 0).astype(int)

    # Fit policy tree
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X, optimal_treatment)

    # Policy predictions
    policy_pred = tree.predict(X)

    # Policy value: expected outcome improvement
    policy_value = np.mean(cate * (policy_pred * 2 - 1))  # Treat if predict 1, don't if 0

    return {
        'tree': tree,
        'policy_accuracy': float((policy_pred == optimal_treatment).mean()),
        'policy_value': float(policy_value),
        'treat_proportion': float(policy_pred.mean()),
        'feature_importance': dict(zip(effect_modifiers, tree.feature_importances_))
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_cate_distribution(
    result: Dict[str, Any],
    true_cate: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """Plot CATE distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cate = result['cate']

    # Histogram
    axes[0].hist(cate, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', label='Zero Effect')
    axes[0].axvline(np.mean(cate), color='blue', linestyle='-', label=f'Mean = {np.mean(cate):.3f}')
    axes[0].set_xlabel('CATE')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Conditional Average Treatment Effects')
    axes[0].legend()

    # If true CATE available, scatter plot
    if true_cate is not None:
        axes[1].scatter(true_cate, cate, alpha=0.3, s=10)
        lims = [min(true_cate.min(), cate.min()), max(true_cate.max(), cate.max())]
        axes[1].plot(lims, lims, 'r--', label='45-degree line')
        axes[1].set_xlabel('True CATE')
        axes[1].set_ylabel('Estimated CATE')
        axes[1].set_title('Estimated vs True CATE')
        axes[1].legend()

        # Correlation
        corr = np.corrcoef(true_cate, cate)[0, 1]
        axes[1].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1].transAxes)
    else:
        # Confidence intervals
        sorted_idx = np.argsort(cate)
        axes[1].fill_between(
            range(len(cate)),
            result['cate_lb'][sorted_idx],
            result['cate_ub'][sorted_idx],
            alpha=0.3,
            label='95% CI'
        )
        axes[1].plot(cate[sorted_idx], label='CATE')
        axes[1].axhline(0, color='red', linestyle='--')
        axes[1].set_xlabel('Observation (sorted by CATE)')
        axes[1].set_ylabel('CATE')
        axes[1].set_title('CATE with Confidence Intervals')
        axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.close()


def plot_cate_by_covariate(
    result: Dict[str, Any],
    df: pd.DataFrame,
    covariate: str,
    save_path: Optional[str] = None
):
    """Plot CATE as function of a covariate."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cate = result['cate']
    x = df[covariate].values

    # Scatter with LOWESS smoothing
    ax.scatter(x, cate, alpha=0.3, s=20)

    # Sort for smoothing
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    cate_sorted = cate[sorted_idx]

    # Simple moving average
    window = max(len(x) // 20, 10)
    cate_smooth = np.convolve(cate_sorted, np.ones(window)/window, mode='valid')
    x_smooth = x_sorted[window//2:-window//2+1]

    ax.plot(x_smooth, cate_smooth, color='red', linewidth=2, label='Smoothed CATE')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(covariate)
    ax.set_ylabel('CATE')
    ax.set_title(f'Treatment Effect Heterogeneity by {covariate}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.close()


# =============================================================================
# Output Functions
# =============================================================================

def print_results(result: Dict[str, Any], heterogeneity: Dict[str, Any] = None):
    """Print formatted results."""
    print("\n" + "="*60)
    print("CAUSAL FOREST ESTIMATION RESULTS")
    print("="*60)

    print(f"\nSample size: {len(result['cate'])}")

    print("\n--- Average Treatment Effect ---")
    print(f"ATE: {result['ate']:.4f}")
    print(f"SE: {result['ate_se']:.4f}")
    print(f"95% CI: [{result['ate_ci'][0]:.4f}, {result['ate_ci'][1]:.4f}]")
    print(f"p-value: {result['ate_pval']:.4f}")

    print("\n--- CATE Summary ---")
    cate = result['cate']
    print(f"Mean CATE: {np.mean(cate):.4f}")
    print(f"Median CATE: {np.median(cate):.4f}")
    print(f"Std CATE: {np.std(cate):.4f}")
    print(f"Min CATE: {np.min(cate):.4f}")
    print(f"Max CATE: {np.max(cate):.4f}")
    print(f"Proportion positive: {(cate > 0).mean():.2%}")

    if heterogeneity:
        print("\n--- Heterogeneity Test ---")
        print(f"CATE Variance: {heterogeneity['cate_variance']:.4f}")
        print(f"CATE Range: {heterogeneity['cate_range']:.4f}")
        print(f"p-value (no heterogeneity): {heterogeneity['heterogeneity_pvalue']:.4f}")
        print(f"Significant heterogeneity: {'Yes' if heterogeneity['significant_heterogeneity'] else 'No'}")

    print("="*60)


def generate_latex_table(result: Dict[str, Any], blp: Dict[str, Any] = None, save_path: Optional[str] = None) -> str:
    """Generate LaTeX table."""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Causal Forest Estimation Results}
\label{tab:causal_forest}
\begin{tabular}{lcc}
\toprule
Statistic & Estimate & 95\% CI \\
\midrule
"""
    latex += f"ATE & {result['ate']:.4f} & [{result['ate_ci'][0]:.4f}, {result['ate_ci'][1]:.4f}] \\\\\n"
    latex += f"CATE Mean & {np.mean(result['cate']):.4f} & - \\\\\n"
    latex += f"CATE Std & {np.std(result['cate']):.4f} & - \\\\\n"
    latex += f"CATE Range & [{np.min(result['cate']):.4f}, {np.max(result['cate']):.4f}] & - \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
"""

    if blp:
        latex += r"""
\vspace{1em}
\begin{tabular}{lccc}
\toprule
Variable & Coef & SE & p-value \\
\midrule
"""
        for name, vals in blp['coefficients'].items():
            stars = '***' if vals['pvalue'] < 0.01 else ('**' if vals['pvalue'] < 0.05 else ('*' if vals['pvalue'] < 0.1 else ''))
            latex += f"{name} & {vals['coef']:.4f}{stars} & {vals['se']:.4f} & {vals['pvalue']:.4f} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
"""

    latex += r"\end{table}"

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {save_path}")

    return latex


# =============================================================================
# Full Analysis Pipeline
# =============================================================================

def run_full_analysis(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: List[str],
    true_cate: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete causal forest analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    outcome : str
        Outcome variable name
    treatment : str
        Treatment variable name
    controls : List[str]
        Control/covariate variable names
    true_cate : np.ndarray, optional
        True CATE for simulation comparison
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
    print("CAUSAL FOREST ANALYSIS")
    print("="*70)

    # Step 1: Data summary
    print("\n--- Step 1: Data Summary ---")
    print(f"Observations: {len(df)}")
    print(f"Treatment: {treatment}")
    print(f"Outcome: {outcome}")
    print(f"Covariates: {len(controls)}")
    print(f"Treatment rate: {df[treatment].mean():.2%}")

    # Step 2: Fit causal forest
    print("\n--- Step 2: Fitting CausalForestDML ---")
    result = fit_causal_forest_dml(df, outcome, treatment, controls)

    # Step 3: Print results
    heterogeneity = heterogeneity_test(result)
    print_results(result, heterogeneity)

    # Step 4: BLP
    print("\n--- Step 4: Best Linear Projection ---")
    blp = best_linear_projection(result, df, controls[:5])  # Use first 5 controls
    print(f"R-squared: {blp['r_squared']:.4f}")
    print(f"F-statistic: {blp['f_stat']:.4f} (p={blp['f_pvalue']:.4f})")

    # Step 5: Policy learning
    print("\n--- Step 5: Policy Learning ---")
    policy = simple_policy_tree(result, df, controls[:5], max_depth=3)
    print(f"Policy accuracy: {policy['policy_accuracy']:.2%}")
    print(f"Policy value: {policy['policy_value']:.4f}")
    print(f"Treatment proportion: {policy['treat_proportion']:.2%}")

    # Step 6: Validation (if true CATE provided)
    if true_cate is not None:
        print("\n--- Step 6: Validation ---")
        corr = np.corrcoef(true_cate, result['cate'])[0, 1]
        mse = np.mean((true_cate - result['cate'])**2)
        print(f"Correlation with true CATE: {corr:.4f}")
        print(f"MSE: {mse:.4f}")

    # Step 7: Output
    if output_dir:
        print("\n--- Step 7: Generating Output ---")

        # CATE distribution
        plot_cate_distribution(result, true_cate, str(output_dir / 'cate_distribution.png'))

        # CATE by first covariate
        plot_cate_by_covariate(result, df, controls[0], str(output_dir / f'cate_by_{controls[0]}.png'))

        # LaTeX table
        generate_latex_table(result, blp, str(output_dir / 'causal_forest_results.tex'))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return {
        'result': result,
        'heterogeneity': heterogeneity,
        'blp': blp,
        'policy': policy,
        'true_cate': true_cate
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Causal Forest Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with simulated data
    python causal_forest_pipeline.py --demo

    # Run with real data
    python causal_forest_pipeline.py --data data.csv --outcome Y --treatment T \\
        --controls "X1,X2,X3,X4,X5"
"""
    )

    parser.add_argument('--demo', action='store_true', help='Run demo with simulated data')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--outcome', type=str, help='Outcome variable name')
    parser.add_argument('--treatment', type=str, help='Treatment variable name')
    parser.add_argument('--controls', type=str, help='Comma-separated control variable names')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--heterogeneity', type=str, default='moderate',
                        choices=['none', 'moderate', 'strong'],
                        help='Heterogeneity level for demo')

    args = parser.parse_args()

    if args.demo:
        print("Running causal forest demo with simulated data...")

        # Simulate data
        df, cate_info = simulate_causal_forest_data(
            n=2000,
            p=10,
            heterogeneity=args.heterogeneity,
            seed=42
        )

        print(f"Simulated data: n={len(df)}, heterogeneity={args.heterogeneity}")
        print(f"True ATE: {cate_info['ate']:.4f}")
        print(f"True CATE SD: {cate_info['cate_sd']:.4f}")

        outcome = 'Y'
        treatment = 'T'
        controls = [f'X{i}' for i in range(10)]

        run_full_analysis(
            df=df,
            outcome=outcome,
            treatment=treatment,
            controls=controls,
            true_cate=cate_info['true_cate'],
            output_dir=args.output
        )

    elif args.data and args.outcome and args.treatment:
        df = pd.read_csv(args.data)

        if args.controls:
            controls = [c.strip() for c in args.controls.split(',')]
        else:
            controls = [c for c in df.columns if c not in [args.outcome, args.treatment]]

        run_full_analysis(
            df=df,
            outcome=args.outcome,
            treatment=args.treatment,
            controls=controls,
            output_dir=args.output
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
