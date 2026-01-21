#!/usr/bin/env python3
"""
OLS Assumption Diagnostics CLI

Command-line tool for comprehensive diagnostics of linear model assumptions:
- Residual analysis (normality, heteroskedasticity)
- Multicollinearity (VIF, condition number)
- Influential observations (Cook's D, leverage)

Usage:
    python diagnose_assumptions.py data.csv --outcome y --features x1 x2 x3
    python diagnose_assumptions.py data.csv --outcome y --all-features --plot
    python diagnose_assumptions.py data.csv --outcome y --vif-only
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


def load_data(
    filepath: str,
    outcome: str,
    features: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Load and prepare data for diagnostics."""
    df = pd.read_csv(filepath)

    if features:
        feature_cols = features
    else:
        exclude_cols = {outcome}
        if exclude:
            exclude_cols.update(exclude)
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c not in exclude_cols]

    X = df[feature_cols].values
    y = df[outcome].values

    return {
        'X': X,
        'y': y,
        'feature_names': feature_cols,
        'n_samples': len(y),
        'n_features': len(feature_cols)
    }


def fit_ols(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Fit OLS and return model with diagnostics."""
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels required: pip install statsmodels")

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    return {
        'model': model,
        'residuals': model.resid,
        'fitted': model.fittedvalues,
        'X_const': X_const
    }


def diagnose_normality(residuals: np.ndarray) -> Dict[str, Any]:
    """Test residual normality."""
    # Shapiro-Wilk (for n < 5000)
    n = len(residuals)
    if n <= 5000:
        stat_sw, p_sw = stats.shapiro(residuals)
    else:
        # Sample for large datasets
        sample = np.random.choice(residuals, 5000, replace=False)
        stat_sw, p_sw = stats.shapiro(sample)

    # Jarque-Bera
    stat_jb, p_jb = stats.jarque_bera(residuals)

    # D'Agostino-Pearson
    if n >= 20:
        stat_dp, p_dp = stats.normaltest(residuals)
    else:
        stat_dp, p_dp = np.nan, np.nan

    # Skewness and kurtosis
    skew = stats.skew(residuals)
    kurt = stats.kurtosis(residuals)

    return {
        'shapiro_wilk': {'statistic': stat_sw, 'p_value': p_sw},
        'jarque_bera': {'statistic': stat_jb, 'p_value': p_jb},
        'dagostino_pearson': {'statistic': stat_dp, 'p_value': p_dp},
        'skewness': skew,
        'kurtosis': kurt,
        'normal': p_sw > 0.05 and p_jb > 0.05
    }


def diagnose_heteroskedasticity(model) -> Dict[str, Any]:
    """Test for heteroskedasticity."""
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white

    resid = model.resid
    exog = model.model.exog

    # Breusch-Pagan
    bp_stat, bp_pval, bp_f, bp_f_pval = het_breuschpagan(resid, exog)

    # White test
    try:
        w_stat, w_pval, w_f, w_f_pval = het_white(resid, exog)
    except Exception:
        # White test can fail with many variables
        w_stat, w_pval, w_f, w_f_pval = np.nan, np.nan, np.nan, np.nan

    return {
        'breusch_pagan': {
            'lm_stat': bp_stat,
            'lm_pvalue': bp_pval,
            'f_stat': bp_f,
            'f_pvalue': bp_f_pval
        },
        'white': {
            'lm_stat': w_stat,
            'lm_pvalue': w_pval,
            'f_stat': w_f,
            'f_pvalue': w_f_pval
        },
        'heteroskedastic': bp_pval < 0.05 or (not np.isnan(w_pval) and w_pval < 0.05)
    }


def diagnose_autocorrelation(model) -> Dict[str, Any]:
    """Test for autocorrelation in residuals."""
    from statsmodels.stats.stattools import durbin_watson

    dw = durbin_watson(model.resid)

    # Interpretation
    if dw < 1.5:
        interpretation = 'Positive autocorrelation likely'
    elif dw > 2.5:
        interpretation = 'Negative autocorrelation likely'
    else:
        interpretation = 'No significant autocorrelation'

    return {
        'durbin_watson': dw,
        'interpretation': interpretation,
        'autocorrelated': dw < 1.5 or dw > 2.5
    }


def diagnose_multicollinearity(
    X: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Any]:
    """Diagnose multicollinearity using VIF and condition number."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm

    # Add constant
    X_const = sm.add_constant(X)
    feature_names_const = ['const'] + list(feature_names)

    # Calculate VIF for each variable
    vif_data = []
    for i in range(X_const.shape[1]):
        try:
            vif = variance_inflation_factor(X_const, i)
        except Exception:
            vif = np.nan

        concern = 'severe' if vif > 10 else ('moderate' if vif > 5 else 'low')
        vif_data.append({
            'feature': feature_names_const[i],
            'vif': vif,
            'concern': concern
        })

    vif_df = pd.DataFrame(vif_data)

    # Condition number
    X_centered = X - X.mean(axis=0)
    X_scaled = X_centered / (X_centered.std(axis=0) + 1e-10)

    try:
        _, s, _ = np.linalg.svd(X_scaled)
        cond_number = s.max() / s.min()
    except Exception:
        cond_number = np.nan

    cond_concern = 'severe' if cond_number > 100 else ('moderate' if cond_number > 30 else 'low')

    # Correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # High correlation pairs
    high_corr = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr_matrix[i, j]) >= 0.8:
                high_corr.append({
                    'feature_1': feature_names[i],
                    'feature_2': feature_names[j],
                    'correlation': corr_matrix[i, j]
                })

    return {
        'vif': vif_df.to_dict('records'),
        'max_vif': vif_df[vif_df['feature'] != 'const']['vif'].max(),
        'n_high_vif': len(vif_df[(vif_df['feature'] != 'const') & (vif_df['vif'] > 10)]),
        'condition_number': cond_number,
        'condition_concern': cond_concern,
        'high_correlation_pairs': high_corr,
        'multicollinear': cond_number > 30 or any(v['vif'] > 10 for v in vif_data[1:])
    }


def diagnose_influential(model) -> Dict[str, Any]:
    """Diagnose influential observations."""
    influence = model.get_influence()

    # Cook's distance
    cooks_d = influence.cooks_distance[0]
    n = len(cooks_d)
    p = model.df_model + 1

    cooks_threshold = 4 / n
    influential_cooks = np.where(cooks_d > cooks_threshold)[0]

    # Leverage
    leverage = influence.hat_matrix_diag
    avg_leverage = p / n
    leverage_threshold = 2 * avg_leverage
    high_leverage = np.where(leverage > leverage_threshold)[0]

    # DFFITS
    dffits = influence.dffits[0]
    dffits_threshold = 2 * np.sqrt(p / n)
    high_dffits = np.where(np.abs(dffits) > dffits_threshold)[0]

    # Combined influential (high Cook's D AND high leverage)
    influential = np.intersect1d(influential_cooks, high_leverage)

    return {
        'cooks_distance': {
            'values': cooks_d,
            'threshold': cooks_threshold,
            'n_influential': len(influential_cooks),
            'influential_indices': influential_cooks.tolist(),
            'max_value': cooks_d.max(),
            'max_index': int(np.argmax(cooks_d))
        },
        'leverage': {
            'values': leverage,
            'threshold': leverage_threshold,
            'average': avg_leverage,
            'n_high': len(high_leverage),
            'high_indices': high_leverage.tolist()
        },
        'dffits': {
            'threshold': dffits_threshold,
            'n_high': len(high_dffits),
            'high_indices': high_dffits.tolist()
        },
        'combined_influential': influential.tolist(),
        'n_combined_influential': len(influential)
    }


def create_diagnostic_plots(
    model,
    output_dir: str,
    prefix: str = 'diagnostic'
) -> List[str]:
    """Create diagnostic plots and save to files."""
    try:
        import matplotlib.pyplot as plt
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        print("matplotlib required for plots: pip install matplotlib")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    residuals = model.resid
    fitted = model.fittedvalues
    influence = model.get_influence()
    std_resid = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]

    # 1. Residuals vs Fitted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(fitted, residuals, alpha=0.5, edgecolors='none')
    ax.axhline(y=0, color='red', linestyle='--')
    try:
        smoothed = lowess(residuals, fitted, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2)
    except Exception:
        pass
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted')
    filepath = output_path / f'{prefix}_resid_vs_fitted.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(filepath))

    # 2. Q-Q Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(std_resid, dist="norm", plot=ax)
    ax.set_title('Normal Q-Q Plot')
    filepath = output_path / f'{prefix}_qq_plot.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(filepath))

    # 3. Scale-Location
    fig, ax = plt.subplots(figsize=(10, 6))
    sqrt_std_resid = np.sqrt(np.abs(std_resid))
    ax.scatter(fitted, sqrt_std_resid, alpha=0.5, edgecolors='none')
    try:
        smoothed = lowess(sqrt_std_resid, fitted, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2)
    except Exception:
        pass
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('sqrt(|Standardized Residuals|)')
    ax.set_title('Scale-Location')
    filepath = output_path / f'{prefix}_scale_location.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(filepath))

    # 4. Residuals vs Leverage
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(leverage, std_resid, alpha=0.5, edgecolors='none')
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized Residuals')
    ax.set_title("Residuals vs Leverage")
    filepath = output_path / f'{prefix}_resid_vs_leverage.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(filepath))

    # 5. Cook's Distance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stem(range(len(cooks_d)), cooks_d, markerfmt=',')
    ax.axhline(y=4/len(cooks_d), color='red', linestyle='--', label='4/n threshold')
    ax.set_xlabel('Observation Index')
    ax.set_ylabel("Cook's Distance")
    ax.set_title("Cook's Distance")
    ax.legend()
    filepath = output_path / f'{prefix}_cooks_distance.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(filepath))

    # 6. Histogram of residuals
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
    # Overlay normal distribution
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()),
            'r-', linewidth=2, label='Normal fit')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend()
    filepath = output_path / f'{prefix}_residual_hist.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(filepath))

    return saved_files


def format_report(
    normality: Dict,
    heteroskedasticity: Dict,
    autocorrelation: Dict,
    multicollinearity: Dict,
    influential: Dict,
    feature_names: List[str]
) -> str:
    """Format comprehensive diagnostic report."""
    lines = []
    lines.append("=" * 70)
    lines.append("OLS ASSUMPTION DIAGNOSTICS REPORT")
    lines.append("=" * 70)

    # Summary
    concerns = []
    if not normality['normal']:
        concerns.append("Non-normal residuals")
    if heteroskedasticity['heteroskedastic']:
        concerns.append("Heteroskedasticity detected")
    if autocorrelation['autocorrelated']:
        concerns.append("Autocorrelation detected")
    if multicollinearity['multicollinear']:
        concerns.append("Multicollinearity present")
    if influential['n_combined_influential'] > 0:
        concerns.append(f"{influential['n_combined_influential']} influential observations")

    lines.append("\nSUMMARY OF CONCERNS:")
    if concerns:
        for c in concerns:
            lines.append(f"  - {c}")
    else:
        lines.append("  No major concerns detected")

    # Normality
    lines.append("\n" + "-" * 70)
    lines.append("1. NORMALITY OF RESIDUALS")
    lines.append("-" * 70)
    lines.append(f"  Shapiro-Wilk test:       W = {normality['shapiro_wilk']['statistic']:.4f}, "
                f"p = {normality['shapiro_wilk']['p_value']:.4f}")
    lines.append(f"  Jarque-Bera test:        JB = {normality['jarque_bera']['statistic']:.4f}, "
                f"p = {normality['jarque_bera']['p_value']:.4f}")
    lines.append(f"  Skewness:                {normality['skewness']:.4f}")
    lines.append(f"  Kurtosis (excess):       {normality['kurtosis']:.4f}")
    lines.append(f"  Conclusion:              {'Normal' if normality['normal'] else 'NON-NORMAL'}")

    # Heteroskedasticity
    lines.append("\n" + "-" * 70)
    lines.append("2. HETEROSKEDASTICITY")
    lines.append("-" * 70)
    bp = heteroskedasticity['breusch_pagan']
    lines.append(f"  Breusch-Pagan test:      LM = {bp['lm_stat']:.4f}, p = {bp['lm_pvalue']:.4f}")
    w = heteroskedasticity['white']
    if not np.isnan(w['lm_stat']):
        lines.append(f"  White test:              LM = {w['lm_stat']:.4f}, p = {w['lm_pvalue']:.4f}")
    lines.append(f"  Conclusion:              {'HETEROSKEDASTIC' if heteroskedasticity['heteroskedastic'] else 'Homoskedastic'}")

    # Autocorrelation
    lines.append("\n" + "-" * 70)
    lines.append("3. AUTOCORRELATION")
    lines.append("-" * 70)
    lines.append(f"  Durbin-Watson statistic: {autocorrelation['durbin_watson']:.4f}")
    lines.append(f"  Interpretation:          {autocorrelation['interpretation']}")

    # Multicollinearity
    lines.append("\n" + "-" * 70)
    lines.append("4. MULTICOLLINEARITY")
    lines.append("-" * 70)
    lines.append(f"  Condition number:        {multicollinearity['condition_number']:.2f} "
                f"({multicollinearity['condition_concern']})")
    lines.append(f"  Features with VIF > 10:  {multicollinearity['n_high_vif']}")

    # VIF table (excluding constant, top 10 by VIF)
    vif_list = [v for v in multicollinearity['vif'] if v['feature'] != 'const']
    vif_list.sort(key=lambda x: x['vif'] if not np.isnan(x['vif']) else 0, reverse=True)

    lines.append("\n  Top VIF values:")
    lines.append(f"  {'Feature':<25} {'VIF':>10} {'Concern':>10}")
    lines.append("  " + "-" * 45)
    for v in vif_list[:10]:
        vif_str = f"{v['vif']:.2f}" if not np.isnan(v['vif']) else 'N/A'
        lines.append(f"  {v['feature']:<25} {vif_str:>10} {v['concern']:>10}")

    # High correlations
    if multicollinearity['high_correlation_pairs']:
        lines.append("\n  High correlation pairs (|r| >= 0.8):")
        for pair in multicollinearity['high_correlation_pairs'][:10]:
            lines.append(f"    {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.3f}")

    # Influential observations
    lines.append("\n" + "-" * 70)
    lines.append("5. INFLUENTIAL OBSERVATIONS")
    lines.append("-" * 70)
    cd = influential['cooks_distance']
    lines.append(f"  Cook's D threshold (4/n): {cd['threshold']:.4f}")
    lines.append(f"  Observations > threshold: {cd['n_influential']}")
    lines.append(f"  Max Cook's D:             {cd['max_value']:.4f} (index {cd['max_index']})")

    lev = influential['leverage']
    lines.append(f"\n  Leverage threshold (2p/n): {lev['threshold']:.4f}")
    lines.append(f"  High leverage observations: {lev['n_high']}")

    lines.append(f"\n  Combined influential (high Cook's D AND leverage): "
                f"{influential['n_combined_influential']}")
    if influential['combined_influential']:
        lines.append(f"  Indices: {influential['combined_influential'][:20]}")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose OLS assumptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full diagnostics
  python diagnose_assumptions.py data.csv --outcome y --all-features

  # VIF analysis only
  python diagnose_assumptions.py data.csv --outcome y --all-features --vif-only

  # With diagnostic plots
  python diagnose_assumptions.py data.csv --outcome y --all-features --plot --plot-dir ./plots

  # Specific features
  python diagnose_assumptions.py data.csv --outcome y --features x1 x2 x3 x4
        """
    )

    parser.add_argument('data', help='Path to CSV data file')
    parser.add_argument('--outcome', '-y', required=True, help='Outcome variable name')
    parser.add_argument('--features', '-X', nargs='+', help='Feature variable names')
    parser.add_argument('--all-features', action='store_true', help='Use all numeric columns')
    parser.add_argument('--exclude', nargs='+', help='Variables to exclude')
    parser.add_argument('--vif-only', action='store_true', help='Only compute VIF analysis')
    parser.add_argument('--plot', action='store_true', help='Generate diagnostic plots')
    parser.add_argument('--plot-dir', default='./diagnostic_plots', help='Directory for plots')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(
        args.data,
        args.outcome,
        args.features if not args.all_features else None,
        args.exclude
    )

    X = data['X']
    y = data['y']
    feature_names = data['feature_names']

    print(f"Data: {data['n_samples']} samples, {data['n_features']} features\n")

    if args.vif_only:
        # Only VIF analysis
        print("Computing VIF analysis...")
        multicol = diagnose_multicollinearity(X, feature_names)

        print("\nVIF Analysis:")
        print(f"{'Feature':<25} {'VIF':>10} {'Concern':>10}")
        print("-" * 45)
        for v in multicol['vif']:
            if v['feature'] != 'const':
                vif_str = f"{v['vif']:.2f}" if not np.isnan(v['vif']) else 'N/A'
                print(f"{v['feature']:<25} {vif_str:>10} {v['concern']:>10}")

        print(f"\nCondition number: {multicol['condition_number']:.2f}")
        return

    # Full diagnostics
    print("Fitting OLS model...")
    ols_result = fit_ols(X, y)
    model = ols_result['model']

    print("Running diagnostics...")
    normality = diagnose_normality(ols_result['residuals'])
    heteroskedasticity = diagnose_heteroskedasticity(model)
    autocorrelation = diagnose_autocorrelation(model)
    multicollinearity = diagnose_multicollinearity(X, feature_names)
    influential = diagnose_influential(model)

    # Print report
    report = format_report(
        normality, heteroskedasticity, autocorrelation,
        multicollinearity, influential, feature_names
    )
    print(report)

    # Generate plots
    if args.plot:
        print(f"\nGenerating diagnostic plots in {args.plot_dir}...")
        saved_files = create_diagnostic_plots(model, args.plot_dir)
        print(f"Saved {len(saved_files)} plots:")
        for f in saved_files:
            print(f"  - {f}")


if __name__ == '__main__':
    main()
