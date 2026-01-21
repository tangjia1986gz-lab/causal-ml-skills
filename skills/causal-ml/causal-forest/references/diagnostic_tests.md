# Diagnostic Tests for Causal Forests

## Overview

Diagnostic tests are essential for validating causal forest estimates. This document covers calibration tests, overlap assessment, heterogeneity tests, and specification checks.

---

## 1. Calibration Tests

### Purpose

Calibration tests verify that the causal forest predictions are well-calibrated - i.e., they accurately capture the true treatment effect heterogeneity.

### Test Calibration (Chernozhukov et al.)

**Regression**:
$$\hat{\tau}_i^{-i} = \alpha + \beta_1 \bar{\tau} + \beta_2 (\hat{\tau}_i - \bar{\tau}) + \epsilon_i$$

Where:
- $\hat{\tau}_i^{-i}$ = out-of-bag CATE prediction for observation i
- $\bar{\tau}$ = mean CATE prediction (captures average effect)
- $(\hat{\tau}_i - \bar{\tau})$ = deviation from mean (captures heterogeneity)

**Interpretation**:
- $\beta_1 = 1$: Mean prediction is well-calibrated (ATE captured)
- $\beta_2 = 1$: Heterogeneity is well-calibrated (CATE variation captured)
- $\beta_2 = 0$: No meaningful heterogeneity detected

### R grf Calibration Test

```r
library(grf)

# Fit causal forest
cf <- causal_forest(X, Y, W, num.trees = 2000)

# Run calibration test
cal_test <- test_calibration(cf)
print(cal_test)

# Output interpretation:
# mean.forest.prediction: should be close to 1 (ATE calibration)
# differential.forest.prediction: should be close to 1 (heterogeneity calibration)
# p-values test if coefficients differ from 0
```

### Python Implementation

```python
from causal_forest import calibration_test, CausalForestModel
import numpy as np
from scipy import stats

def run_calibration_test(
    cf_model: CausalForestModel,
    X: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray
) -> dict:
    """
    Run calibration test for causal forest.

    Tests:
    1. Mean forest prediction (ATE calibration)
    2. Differential forest prediction (heterogeneity calibration)
    """
    # Get out-of-bag predictions
    tau_hat, _ = cf_model.predict(X, return_std=False)
    tau_mean = np.mean(tau_hat)
    tau_centered = tau_hat - tau_mean

    # Create pseudo-outcomes for calibration
    # Using AIPW-style pseudo-outcomes
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    # Estimate nuisance functions
    ps_model = GradientBoostingClassifier(n_estimators=100)
    e_hat = cross_val_predict(ps_model, X, treatment, method='predict_proba')[:, 1]
    e_hat = np.clip(e_hat, 0.01, 0.99)

    mu_model = GradientBoostingRegressor(n_estimators=100)
    mu_hat = cross_val_predict(mu_model, np.column_stack([X, treatment]), y)

    # AIPW pseudo-outcome
    pseudo_tau = (
        (treatment / e_hat - (1 - treatment) / (1 - e_hat)) * (y - mu_hat) +
        tau_hat
    )

    # Calibration regression
    X_cal = np.column_stack([np.ones(len(tau_hat)), tau_mean * np.ones(len(tau_hat)), tau_centered])

    # OLS estimation
    beta = np.linalg.lstsq(X_cal, pseudo_tau, rcond=None)[0]

    # Standard errors (HC robust)
    residuals = pseudo_tau - X_cal @ beta
    n = len(residuals)
    k = X_cal.shape[1]

    # Heteroskedasticity-robust variance
    bread = np.linalg.inv(X_cal.T @ X_cal)
    meat = X_cal.T @ np.diag(residuals**2) @ X_cal
    V = bread @ meat @ bread
    se = np.sqrt(np.diag(V))

    # Test statistics
    results = {
        'mean_forest_prediction': {
            'estimate': beta[1],
            'std_error': se[1],
            't_stat': beta[1] / se[1],
            'p_value': 2 * (1 - stats.t.cdf(abs(beta[1] / se[1]), df=n-k))
        },
        'differential_forest_prediction': {
            'estimate': beta[2],
            'std_error': se[2],
            't_stat': beta[2] / se[2],
            'p_value': 2 * (1 - stats.t.cdf(abs(beta[2] / se[2]), df=n-k))
        }
    }

    return results

# Usage
cal_results = run_calibration_test(cf_model, X, y, treatment)

print("Calibration Test Results:")
print(f"Mean Forest Prediction: {cal_results['mean_forest_prediction']['estimate']:.3f} "
      f"(SE: {cal_results['mean_forest_prediction']['std_error']:.3f})")
print(f"Differential Forest Prediction: {cal_results['differential_forest_prediction']['estimate']:.3f} "
      f"(SE: {cal_results['differential_forest_prediction']['std_error']:.3f})")
```

### Interpreting Calibration Results

| Coefficient | Value | Interpretation |
|-------------|-------|----------------|
| Mean Forest | = 1 | ATE well-calibrated |
| Mean Forest | < 1 | ATE overestimated |
| Mean Forest | > 1 | ATE underestimated |
| Differential | = 1 | Heterogeneity well-calibrated |
| Differential | < 1 | Heterogeneity overstated |
| Differential | = 0 | No true heterogeneity |

---

## 2. Overlap Assessment

### Propensity Score Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

def assess_overlap(X, treatment, threshold_low=0.1, threshold_high=0.9):
    """
    Assess propensity score overlap for causal inference validity.
    """
    # Estimate propensity scores
    ps_model = GradientBoostingClassifier(n_estimators=100, max_depth=4)
    propensity = cross_val_predict(ps_model, X, treatment, method='predict_proba')[:, 1]

    # Summary statistics
    results = {
        'propensity_min': propensity.min(),
        'propensity_max': propensity.max(),
        'propensity_mean': propensity.mean(),
        'propensity_std': propensity.std(),
        'n_extreme_low': (propensity < threshold_low).sum(),
        'n_extreme_high': (propensity > threshold_high).sum(),
        'pct_in_common_support': ((propensity >= threshold_low) & (propensity <= threshold_high)).mean()
    }

    return propensity, results

def plot_overlap(propensity, treatment, save_path=None):
    """Visualize propensity score overlap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram by treatment group
    ax1 = axes[0]
    ax1.hist(propensity[treatment==0], bins=50, alpha=0.6,
             label='Control', color='blue', density=True)
    ax1.hist(propensity[treatment==1], bins=50, alpha=0.6,
             label='Treated', color='red', density=True)
    ax1.axvline(0.1, color='black', linestyle='--', label='Trim threshold')
    ax1.axvline(0.9, color='black', linestyle='--')
    ax1.set_xlabel('Propensity Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Propensity Score Distribution by Treatment')
    ax1.legend()

    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot([propensity[treatment==0], propensity[treatment==1]],
                     labels=['Control', 'Treated'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.axhline(0.1, color='black', linestyle='--')
    ax2.axhline(0.9, color='black', linestyle='--')
    ax2.set_ylabel('Propensity Score')
    ax2.set_title('Propensity Score Box Plot')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig

# Usage
propensity, overlap_results = assess_overlap(X, treatment)
print(f"Propensity range: [{overlap_results['propensity_min']:.4f}, {overlap_results['propensity_max']:.4f}]")
print(f"Observations with extreme propensity: {overlap_results['n_extreme_low'] + overlap_results['n_extreme_high']}")
print(f"Common support coverage: {overlap_results['pct_in_common_support']:.1%}")

plot_overlap(propensity, treatment)
```

### Overlap Trimming

When overlap violations exist, consider trimming:

```python
def trim_for_overlap(X, y, treatment, propensity,
                     trim_low=0.1, trim_high=0.9):
    """
    Trim observations with extreme propensity scores.
    """
    mask = (propensity >= trim_low) & (propensity <= trim_high)

    print(f"Trimming: {(~mask).sum()} observations ({(~mask).mean():.1%})")
    print(f"Remaining: {mask.sum()} observations")

    if isinstance(X, pd.DataFrame):
        return X[mask], y[mask], treatment[mask]
    return X[mask], y[mask], treatment[mask]

# Apply trimming
X_trim, y_trim, treatment_trim = trim_for_overlap(X, y, treatment, propensity)
```

---

## 3. Heterogeneity Tests

### Omnibus Heterogeneity Test

Tests whether any heterogeneity exists (H0: constant treatment effect).

```python
from scipy import stats

def omnibus_heterogeneity_test(cate_estimates, std_errors):
    """
    Test H0: All CATEs are equal (no heterogeneity).

    Uses chi-squared test based on standardized CATEs.
    """
    tau = np.array(cate_estimates)
    se = np.array(std_errors)

    # Mean CATE
    tau_mean = np.mean(tau)

    # Chi-squared statistic
    chi2_stat = np.sum(((tau - tau_mean) / (se + 1e-10))**2)
    df = len(tau) - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    return {
        'chi2_statistic': chi2_stat,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### Best Linear Predictor Test

Tests heterogeneity via Best Linear Projection:

```python
def blp_heterogeneity_test(cf_model, X, A):
    """
    Test heterogeneity via Best Linear Projection.

    H0: CATE is constant (beta_A = 0 for all A variables)
    """
    from causal_forest import best_linear_projection

    blp = best_linear_projection(cf_model, A, X=X)

    # Joint F-test for all coefficients
    # H0: beta_1 = beta_2 = ... = beta_k = 0
    k = len(blp.coefficients)

    # Wald statistic
    beta = blp.coefficients
    se = blp.std_errors

    wald_stat = np.sum((beta / (se + 1e-10))**2)
    p_value = 1 - stats.chi2.cdf(wald_stat, df=k)

    return {
        'wald_statistic': wald_stat,
        'degrees_of_freedom': k,
        'p_value': p_value,
        'coefficients': dict(zip(blp.feature_names, blp.coefficients)),
        'significant': p_value < 0.05
    }
```

### Group-wise Heterogeneity Test

Tests whether CATE differs across predefined groups:

```python
def group_heterogeneity_test(cate_estimates, groups):
    """
    Test H0: CATE is equal across groups.

    Uses ANOVA-style test.
    """
    tau = np.array(cate_estimates)
    groups = np.array(groups)
    unique_groups = np.unique(groups)

    # Group means and sizes
    group_means = [np.mean(tau[groups == g]) for g in unique_groups]
    group_sizes = [np.sum(groups == g) for g in unique_groups]

    # Overall mean
    overall_mean = np.mean(tau)

    # Between-group sum of squares
    ss_between = sum(n * (m - overall_mean)**2
                     for n, m in zip(group_sizes, group_means))

    # Within-group sum of squares
    ss_within = sum(np.sum((tau[groups == g] - np.mean(tau[groups == g]))**2)
                    for g in unique_groups)

    # Degrees of freedom
    df_between = len(unique_groups) - 1
    df_within = len(tau) - len(unique_groups)

    # F statistic
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / (ms_within + 1e-10)

    p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)

    return {
        'f_statistic': f_stat,
        'df_between': df_between,
        'df_within': df_within,
        'p_value': p_value,
        'group_means': dict(zip(unique_groups, group_means)),
        'significant': p_value < 0.05
    }
```

---

## 4. Covariate Balance Diagnostics

### Standardized Mean Differences

```python
def covariate_balance(X, treatment, method='standardized_diff'):
    """
    Assess covariate balance between treatment groups.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X = X.values
    else:
        feature_names = [f'X{i}' for i in range(X.shape[1])]

    treated = treatment == 1
    control = treatment == 0

    balance_stats = []

    for j, name in enumerate(feature_names):
        x_t = X[treated, j]
        x_c = X[control, j]

        mean_t = np.mean(x_t)
        mean_c = np.mean(x_c)
        std_pooled = np.sqrt((np.var(x_t) + np.var(x_c)) / 2)

        # Standardized difference
        std_diff = (mean_t - mean_c) / (std_pooled + 1e-10)

        # Variance ratio
        var_ratio = np.var(x_t) / (np.var(x_c) + 1e-10)

        balance_stats.append({
            'variable': name,
            'mean_treated': mean_t,
            'mean_control': mean_c,
            'std_diff': std_diff,
            'var_ratio': var_ratio,
            'balanced': abs(std_diff) < 0.1
        })

    return pd.DataFrame(balance_stats)

# Usage
balance = covariate_balance(X, treatment)
print(balance)
print(f"\nUnbalanced covariates: {(~balance['balanced']).sum()}")
```

### Balance Plot

```python
def plot_balance(balance_df, save_path=None):
    """Create balance plot (Love plot)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by absolute standardized difference
    balance_sorted = balance_df.sort_values('std_diff', key=abs)

    y_pos = np.arange(len(balance_sorted))
    colors = ['green' if b else 'red' for b in balance_sorted['balanced']]

    ax.barh(y_pos, balance_sorted['std_diff'], color=colors, alpha=0.7)
    ax.axvline(-0.1, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0.1, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(balance_sorted['variable'])
    ax.set_xlabel('Standardized Mean Difference')
    ax.set_title('Covariate Balance (Love Plot)')

    # Add balance region annotation
    ax.fill_betweenx([-1, len(balance_sorted)], -0.1, 0.1,
                     alpha=0.1, color='green', label='Balanced region')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
```

---

## 5. Specification Tests

### Out-of-Bag Prediction Error

```python
def oob_prediction_error(cf_model, X, y, treatment):
    """
    Assess out-of-bag prediction quality.
    """
    # Get OOB predictions
    tau_hat, _ = cf_model.predict(X, return_std=False)

    # For evaluation, compare implied Y predictions
    # Y_hat = mu0 + tau * treatment

    # This requires nuisance models - simplified version:
    # Check if CATE predictions are reasonable

    stats = {
        'cate_mean': np.mean(tau_hat),
        'cate_std': np.std(tau_hat),
        'cate_min': np.min(tau_hat),
        'cate_max': np.max(tau_hat),
        'cate_median': np.median(tau_hat)
    }

    return stats
```

### Cross-Validation Assessment

```python
def cross_validate_cate(X, y, treatment, n_folds=5, config=None):
    """
    Cross-validate CATE estimation.
    """
    from sklearn.model_selection import KFold
    from causal_forest import fit_causal_forest, estimate_cate

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        t_train, t_test = treatment[train_idx], treatment[test_idx]

        # Fit model
        cf = fit_causal_forest(X_train, y_train, t_train, config=config)

        # Predict on test
        cate_test = estimate_cate(cf, X_test)

        cv_results.append({
            'fold': fold,
            'mean_cate': cate_test.mean,
            'std_cate': cate_test.std,
            'n_test': len(test_idx)
        })

    cv_df = pd.DataFrame(cv_results)

    print("Cross-Validation Results:")
    print(f"  Mean CATE across folds: {cv_df['mean_cate'].mean():.4f} (SD: {cv_df['mean_cate'].std():.4f})")
    print(f"  CATE heterogeneity across folds: {cv_df['std_cate'].mean():.4f}")

    return cv_df
```

---

## 6. Comprehensive Diagnostic Report

```python
def run_full_diagnostics(cf_model, X, y, treatment, save_dir=None):
    """
    Run comprehensive diagnostics and generate report.
    """
    import os
    from datetime import datetime

    results = {}

    # 1. Overlap assessment
    print("Assessing overlap...")
    propensity, overlap_results = assess_overlap(X, treatment)
    results['overlap'] = overlap_results

    # 2. Covariate balance
    print("Checking covariate balance...")
    balance = covariate_balance(X, treatment)
    results['balance'] = balance.to_dict()

    # 3. CATE estimates
    print("Computing CATE estimates...")
    cate = estimate_cate(cf_model, X)
    results['cate_summary'] = {
        'mean': cate.mean,
        'std': cate.std,
        'min': float(cate.estimates.min()),
        'max': float(cate.estimates.max()),
        'pct_positive': cate.proportion_positive,
        'pct_significant': cate.proportion_significant
    }

    # 4. Heterogeneity test
    print("Testing for heterogeneity...")
    het_test = omnibus_heterogeneity_test(cate.estimates, cate.std_errors)
    results['heterogeneity_test'] = het_test

    # 5. Calibration test
    print("Running calibration test...")
    cal_test = run_calibration_test(cf_model, X, y, treatment)
    results['calibration_test'] = cal_test

    # Generate report
    report = generate_diagnostic_report(results)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Save report
        with open(os.path.join(save_dir, 'diagnostic_report.md'), 'w') as f:
            f.write(report)

        # Save plots
        plot_overlap(propensity, treatment,
                    save_path=os.path.join(save_dir, 'overlap.png'))
        plt.close()

        plot_balance(pd.DataFrame(results['balance']),
                    save_path=os.path.join(save_dir, 'balance.png'))
        plt.close()

        print(f"Diagnostics saved to {save_dir}")

    return results, report


def generate_diagnostic_report(results):
    """Generate markdown diagnostic report."""
    report = f"""# Causal Forest Diagnostic Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Overlap Assessment

| Metric | Value |
|--------|-------|
| Propensity Min | {results['overlap']['propensity_min']:.4f} |
| Propensity Max | {results['overlap']['propensity_max']:.4f} |
| Common Support Coverage | {results['overlap']['pct_in_common_support']:.1%} |
| Extreme Low (< 0.1) | {results['overlap']['n_extreme_low']} |
| Extreme High (> 0.9) | {results['overlap']['n_extreme_high']} |

**Assessment**: {'PASS' if results['overlap']['pct_in_common_support'] > 0.9 else 'CAUTION - Limited overlap'}

## 2. CATE Summary

| Metric | Value |
|--------|-------|
| Mean CATE | {results['cate_summary']['mean']:.4f} |
| Std Dev | {results['cate_summary']['std']:.4f} |
| Range | [{results['cate_summary']['min']:.4f}, {results['cate_summary']['max']:.4f}] |
| % Positive | {results['cate_summary']['pct_positive']:.1%} |
| % Significant | {results['cate_summary']['pct_significant']:.1%} |

## 3. Heterogeneity Test

| Metric | Value |
|--------|-------|
| Chi-squared Statistic | {results['heterogeneity_test']['chi2_statistic']:.2f} |
| Degrees of Freedom | {results['heterogeneity_test']['degrees_of_freedom']} |
| p-value | {results['heterogeneity_test']['p_value']:.4f} |

**Result**: {'Significant heterogeneity detected' if results['heterogeneity_test']['significant'] else 'No significant heterogeneity'}

## 4. Calibration Test

### Mean Forest Prediction (ATE Calibration)
- Estimate: {results['calibration_test']['mean_forest_prediction']['estimate']:.4f}
- Std Error: {results['calibration_test']['mean_forest_prediction']['std_error']:.4f}
- p-value: {results['calibration_test']['mean_forest_prediction']['p_value']:.4f}

### Differential Forest Prediction (Heterogeneity Calibration)
- Estimate: {results['calibration_test']['differential_forest_prediction']['estimate']:.4f}
- Std Error: {results['calibration_test']['differential_forest_prediction']['std_error']:.4f}
- p-value: {results['calibration_test']['differential_forest_prediction']['p_value']:.4f}

## Recommendations

"""

    # Add recommendations based on results
    recommendations = []

    if results['overlap']['pct_in_common_support'] < 0.9:
        recommendations.append("- Consider trimming observations with extreme propensity scores")

    if not results['heterogeneity_test']['significant']:
        recommendations.append("- Limited heterogeneity detected; consider simpler methods for ATE")

    if results['calibration_test']['differential_forest_prediction']['estimate'] < 0.5:
        recommendations.append("- Heterogeneity may be overstated; interpret subgroup results cautiously")

    if not recommendations:
        recommendations.append("- Diagnostics look acceptable; proceed with analysis")

    report += "\n".join(recommendations)

    return report
```

---

## Summary Checklist

Before reporting causal forest results:

- [ ] Overlap: Common support > 90%, no extreme propensities
- [ ] Balance: Standardized differences < 0.1 for key covariates
- [ ] Heterogeneity: Statistically significant (if claiming heterogeneity)
- [ ] Calibration: Mean and differential predictions close to 1
- [ ] Sample size: Adequate for subgroup analysis
- [ ] Robustness: Results stable across specifications

---

## References

1. Athey, S., & Imbens, G. W. (2017). The State of Applied Econometrics: Causality and Policy Evaluation. *Journal of Economic Perspectives*, 31(2), 3-32.

2. Chernozhukov, V., Demirer, M., Duflo, E., & Fernandez-Val, I. (2018). Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments. *NBER Working Paper*.

3. grf documentation: https://grf-labs.github.io/grf/articles/diagnostics.html
