#!/usr/bin/env python3
"""
Linear Model Fitting CLI with Cross-Validation

Command-line tool for fitting OLS, Ridge, Lasso, and Elastic Net models
with cross-validation and comprehensive output.

Usage:
    python run_linear_model.py data.csv --outcome y --features x1 x2 x3 --model lasso
    python run_linear_model.py data.csv --outcome y --all-features --model ridge --cv 10
    python run_linear_model.py data.csv --outcome y --treatment d --model double-selection
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    RidgeCV, LassoCV, ElasticNetCV
)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_data(
    filepath: str,
    outcome: str,
    features: Optional[List[str]] = None,
    treatment: Optional[str] = None,
    exclude: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Load and prepare data for modeling."""
    df = pd.read_csv(filepath)

    # Determine feature columns
    if features:
        feature_cols = features
    else:
        # Use all numeric columns except outcome and treatment
        exclude_cols = {outcome}
        if treatment:
            exclude_cols.add(treatment)
        if exclude:
            exclude_cols.update(exclude)
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c not in exclude_cols]

    # Extract arrays
    X = df[feature_cols].values
    y = df[outcome].values

    result = {
        'X': X,
        'y': y,
        'feature_names': feature_cols,
        'n_samples': len(y),
        'n_features': len(feature_cols)
    }

    if treatment:
        result['d'] = df[treatment].values
        result['treatment_name'] = treatment

    return result


def fit_ols(X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
    """Fit OLS regression with cross-validation metrics."""
    model = LinearRegression()

    # Cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_mse = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')

    # Fit on full data
    model.fit(X, y)
    y_pred = model.predict(X)

    return {
        'model_type': 'OLS',
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'r2_full': r2_score(y, y_pred),
        'mse_full': mean_squared_error(y, y_pred),
        'cv_rmse_mean': np.sqrt(cv_mse.mean()),
        'cv_rmse_std': np.sqrt(cv_mse).std(),
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std(),
        'n_nonzero': np.sum(model.coef_ != 0),
        'model': model
    }


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    alphas: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Fit Ridge regression with CV for alpha selection."""
    if alphas is None:
        alphas = np.logspace(-4, 4, 50)

    # RidgeCV for alpha selection
    model = RidgeCV(alphas=alphas, cv=cv)
    model.fit(X, y)

    # Cross-validation metrics at optimal alpha
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    ridge_opt = Ridge(alpha=model.alpha_)
    cv_mse = -cross_val_score(ridge_opt, X, y, cv=kf, scoring='neg_mean_squared_error')
    cv_r2 = cross_val_score(ridge_opt, X, y, cv=kf, scoring='r2')

    y_pred = model.predict(X)

    return {
        'model_type': 'Ridge',
        'alpha': model.alpha_,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'r2_full': r2_score(y, y_pred),
        'mse_full': mean_squared_error(y, y_pred),
        'cv_rmse_mean': np.sqrt(cv_mse.mean()),
        'cv_rmse_std': np.sqrt(cv_mse).std(),
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std(),
        'n_nonzero': np.sum(model.coef_ != 0),
        'model': model
    }


def fit_lasso(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    alphas: Optional[np.ndarray] = None,
    use_1se_rule: bool = False
) -> Dict[str, Any]:
    """Fit Lasso regression with CV for alpha selection."""
    model = LassoCV(alphas=alphas, cv=cv, max_iter=10000, random_state=42)
    model.fit(X, y)

    # Apply 1-SE rule if requested
    alpha_selected = model.alpha_
    if use_1se_rule:
        mse_mean = model.mse_path_.mean(axis=1)
        mse_std = model.mse_path_.std(axis=1)
        mse_se = mse_std / np.sqrt(model.mse_path_.shape[1])

        best_idx = np.argmin(mse_mean)
        threshold = mse_mean[best_idx] + mse_se[best_idx]
        valid_idx = np.where(mse_mean <= threshold)[0]
        alpha_selected = model.alphas_[valid_idx[0]]  # Largest valid alpha

        # Refit with 1-SE alpha
        model_1se = Lasso(alpha=alpha_selected, max_iter=10000)
        model_1se.fit(X, y)
        model = model_1se
        model.alpha_ = alpha_selected

    # Cross-validation metrics
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    lasso_opt = Lasso(alpha=alpha_selected, max_iter=10000)
    cv_mse = -cross_val_score(lasso_opt, X, y, cv=kf, scoring='neg_mean_squared_error')
    cv_r2 = cross_val_score(lasso_opt, X, y, cv=kf, scoring='r2')

    y_pred = model.predict(X)
    selected_indices = np.where(model.coef_ != 0)[0]

    return {
        'model_type': 'Lasso',
        'alpha': alpha_selected,
        'alpha_rule': '1-SE' if use_1se_rule else 'CV-min',
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'r2_full': r2_score(y, y_pred),
        'mse_full': mean_squared_error(y, y_pred),
        'cv_rmse_mean': np.sqrt(cv_mse.mean()),
        'cv_rmse_std': np.sqrt(cv_mse).std(),
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std(),
        'n_nonzero': len(selected_indices),
        'selected_indices': selected_indices.tolist(),
        'model': model
    }


def fit_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    alphas: Optional[np.ndarray] = None,
    l1_ratios: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Fit Elastic Net with CV for alpha and l1_ratio selection."""
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

    model = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=cv,
        max_iter=10000,
        random_state=42
    )
    model.fit(X, y)

    # Cross-validation metrics
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    enet_opt = ElasticNet(alpha=model.alpha_, l1_ratio=model.l1_ratio_, max_iter=10000)
    cv_mse = -cross_val_score(enet_opt, X, y, cv=kf, scoring='neg_mean_squared_error')
    cv_r2 = cross_val_score(enet_opt, X, y, cv=kf, scoring='r2')

    y_pred = model.predict(X)
    selected_indices = np.where(model.coef_ != 0)[0]

    return {
        'model_type': 'ElasticNet',
        'alpha': model.alpha_,
        'l1_ratio': model.l1_ratio_,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'r2_full': r2_score(y, y_pred),
        'mse_full': mean_squared_error(y, y_pred),
        'cv_rmse_mean': np.sqrt(cv_mse.mean()),
        'cv_rmse_std': np.sqrt(cv_mse).std(),
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std(),
        'n_nonzero': len(selected_indices),
        'selected_indices': selected_indices.tolist(),
        'model': model
    }


def fit_double_selection(
    X: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    cv: int = 5
) -> Dict[str, Any]:
    """Fit post-double-selection Lasso for causal inference."""
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels required for double selection")

    # Standardize X
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Step 1: Lasso Y ~ X
    lasso_y = LassoCV(cv=cv, max_iter=10000, random_state=42)
    lasso_y.fit(X_std, y)
    selected_y = set(np.where(lasso_y.coef_ != 0)[0])

    # Step 2: Lasso D ~ X
    lasso_d = LassoCV(cv=cv, max_iter=10000, random_state=42)
    lasso_d.fit(X_std, d)
    selected_d = set(np.where(lasso_d.coef_ != 0)[0])

    # Step 3: Union
    selected_union = sorted(selected_y | selected_d)

    # Step 4: Post-selection OLS
    if selected_union:
        X_selected = X[:, selected_union]
        X_ols = np.column_stack([d, X_selected])
    else:
        X_ols = d.reshape(-1, 1)

    X_ols = sm.add_constant(X_ols)
    ols = sm.OLS(y, X_ols).fit(cov_type='HC1')

    return {
        'model_type': 'PostDoubleSelection',
        'treatment_effect': ols.params[1],
        'std_error': ols.bse[1],
        't_stat': ols.tvalues[1],
        'p_value': ols.pvalues[1],
        'ci_lower': ols.conf_int()[1, 0],
        'ci_upper': ols.conf_int()[1, 1],
        'selected_by_y': list(selected_y),
        'selected_by_d': list(selected_d),
        'selected_union': selected_union,
        'n_selected_y': len(selected_y),
        'n_selected_d': len(selected_d),
        'n_selected_total': len(selected_union),
        'alpha_y': lasso_y.alpha_,
        'alpha_d': lasso_d.alpha_,
        'r2_ols': ols.rsquared
    }


def format_results(
    results: Dict[str, Any],
    feature_names: List[str],
    output_format: str = 'text'
) -> str:
    """Format results for display."""
    if output_format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for k, v in results.items():
            if k == 'model':
                continue
            elif isinstance(v, np.ndarray):
                json_results[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                json_results[k] = float(v)
            else:
                json_results[k] = v
        return json.dumps(json_results, indent=2)

    # Text format
    lines = []
    lines.append("=" * 60)
    lines.append(f"Model: {results['model_type']}")
    lines.append("=" * 60)

    if results['model_type'] == 'PostDoubleSelection':
        lines.append("\nCausal Effect Estimation:")
        lines.append(f"  Treatment Effect: {results['treatment_effect']:.6f}")
        lines.append(f"  Std. Error:       {results['std_error']:.6f}")
        lines.append(f"  t-statistic:      {results['t_stat']:.4f}")
        lines.append(f"  p-value:          {results['p_value']:.4f}")
        lines.append(f"  95% CI:           [{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")
        lines.append(f"\nVariable Selection:")
        lines.append(f"  Selected by Y model: {results['n_selected_y']}")
        lines.append(f"  Selected by D model: {results['n_selected_d']}")
        lines.append(f"  Union (total):       {results['n_selected_total']}")
        if results['selected_union']:
            lines.append(f"\n  Selected controls: {[feature_names[i] for i in results['selected_union']]}")
    else:
        # Regularization parameters
        if 'alpha' in results:
            lines.append(f"\nRegularization:")
            lines.append(f"  Alpha: {results['alpha']:.6f}")
            if 'l1_ratio' in results:
                lines.append(f"  L1 Ratio: {results['l1_ratio']:.4f}")
            if 'alpha_rule' in results:
                lines.append(f"  Selection Rule: {results['alpha_rule']}")

        # Performance metrics
        lines.append(f"\nModel Performance:")
        lines.append(f"  R-squared (full):    {results['r2_full']:.4f}")
        lines.append(f"  MSE (full):          {results['mse_full']:.6f}")
        lines.append(f"  CV RMSE:             {results['cv_rmse_mean']:.4f} (+/- {results['cv_rmse_std']:.4f})")
        lines.append(f"  CV R-squared:        {results['cv_r2_mean']:.4f} (+/- {results['cv_r2_std']:.4f})")

        # Variable selection
        lines.append(f"\nVariable Selection:")
        lines.append(f"  Non-zero coefficients: {results['n_nonzero']} / {len(results['coefficients'])}")

        # Top coefficients
        coef = results['coefficients']
        importance = np.abs(coef)
        top_idx = np.argsort(importance)[::-1][:10]

        lines.append(f"\nTop Coefficients (by magnitude):")
        for i in top_idx:
            if coef[i] != 0:
                lines.append(f"  {feature_names[i]:20s}: {coef[i]:+.6f}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Fit linear models with cross-validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fit Lasso on all numeric features
  python run_linear_model.py data.csv --outcome y --model lasso

  # Fit Ridge with specific features
  python run_linear_model.py data.csv --outcome y --features x1 x2 x3 --model ridge

  # Double selection for causal inference
  python run_linear_model.py data.csv --outcome y --treatment d --model double-selection

  # Lasso with 1-SE rule
  python run_linear_model.py data.csv --outcome y --model lasso --use-1se

  # Output as JSON
  python run_linear_model.py data.csv --outcome y --model lasso --output json
        """
    )

    parser.add_argument('data', help='Path to CSV data file')
    parser.add_argument('--outcome', '-y', required=True, help='Outcome variable name')
    parser.add_argument('--features', '-X', nargs='+', help='Feature variable names')
    parser.add_argument('--all-features', action='store_true', help='Use all numeric columns as features')
    parser.add_argument('--treatment', '-d', help='Treatment variable name (for double selection)')
    parser.add_argument('--exclude', nargs='+', help='Variables to exclude from features')
    parser.add_argument('--model', '-m',
                       choices=['ols', 'ridge', 'lasso', 'elasticnet', 'double-selection'],
                       default='lasso', help='Model type')
    parser.add_argument('--cv', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--standardize', action='store_true', default=True,
                       help='Standardize features before fitting')
    parser.add_argument('--use-1se', action='store_true', help='Use 1-SE rule for Lasso')
    parser.add_argument('--output', '-o', choices=['text', 'json'], default='text',
                       help='Output format')
    parser.add_argument('--save-coef', help='Save coefficients to CSV file')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(
        args.data,
        args.outcome,
        args.features if not args.all_features else None,
        args.treatment,
        args.exclude
    )

    X = data['X']
    y = data['y']
    feature_names = data['feature_names']

    print(f"Data: {data['n_samples']} samples, {data['n_features']} features")

    # Standardize if requested
    if args.standardize and args.model != 'ols':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("Features standardized")

    # Fit model
    print(f"Fitting {args.model} model with {args.cv}-fold CV...")

    if args.model == 'ols':
        results = fit_ols(X, y, cv=args.cv)
    elif args.model == 'ridge':
        results = fit_ridge(X, y, cv=args.cv)
    elif args.model == 'lasso':
        results = fit_lasso(X, y, cv=args.cv, use_1se_rule=args.use_1se)
    elif args.model == 'elasticnet':
        results = fit_elastic_net(X, y, cv=args.cv)
    elif args.model == 'double-selection':
        if args.treatment is None:
            print("Error: --treatment required for double-selection")
            sys.exit(1)
        results = fit_double_selection(X, y, data['d'], cv=args.cv)

    # Output results
    output = format_results(results, feature_names, args.output)
    print(output)

    # Save coefficients if requested
    if args.save_coef and 'coefficients' in results:
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': results['coefficients'],
            'abs_coefficient': np.abs(results['coefficients']),
            'selected': results['coefficients'] != 0
        })
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        coef_df.to_csv(args.save_coef, index=False)
        print(f"\nCoefficients saved to {args.save_coef}")


if __name__ == '__main__':
    main()
