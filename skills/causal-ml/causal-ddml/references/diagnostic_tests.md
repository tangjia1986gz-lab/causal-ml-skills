# DDML Diagnostic Tests

> Comprehensive diagnostics for Double/Debiased Machine Learning estimation

## Overview

DDML estimation requires careful validation of nuisance function quality, cross-fitting stability, and sensitivity to specification choices. This document provides diagnostic procedures for each.

---

## 1. Cross-Fit Diagnostics

### 1.1 Fold-Level Variation

Check if estimates vary substantially across folds:

```python
def cross_fit_variation_diagnostic(fold_estimates, fold_ses):
    """
    Assess variation of estimates across cross-fitting folds.

    Parameters
    ----------
    fold_estimates : array-like
        Treatment effect estimates from each fold
    fold_ses : array-like
        Standard errors from each fold

    Returns
    -------
    dict
        Diagnostic results
    """
    mean_est = np.mean(fold_estimates)
    std_est = np.std(fold_estimates)
    cv = std_est / abs(mean_est) if mean_est != 0 else np.inf

    # Chi-squared test for homogeneity
    pooled_se = np.sqrt(np.mean([se**2 for se in fold_ses]))
    chi2_stat = np.sum([(e - mean_est)**2 / se**2
                        for e, se in zip(fold_estimates, fold_ses)])
    df = len(fold_estimates) - 1
    chi2_pval = 1 - stats.chi2.cdf(chi2_stat, df)

    return {
        'mean_estimate': mean_est,
        'std_across_folds': std_est,
        'cv': cv,
        'chi2_stat': chi2_stat,
        'chi2_pval': chi2_pval,
        'homogeneous': chi2_pval > 0.05,
        'interpretation': interpret_fold_variation(cv, chi2_pval)
    }

def interpret_fold_variation(cv, chi2_pval):
    if cv < 0.1 and chi2_pval > 0.05:
        return "GOOD: Estimates stable across folds"
    elif cv < 0.2:
        return "ACCEPTABLE: Moderate fold variation"
    else:
        return "WARNING: High fold variation - consider more folds or larger sample"
```

### 1.2 Repetition Stability

Run multiple cross-fitting iterations with different random seeds:

```python
def repetition_stability_diagnostic(data, outcome, treatment, controls,
                                   n_reps=10, n_folds=5):
    """
    Assess stability across multiple cross-fitting repetitions.
    """
    estimates = []

    for rep in range(n_reps):
        result = estimate_plr(
            data=data,
            outcome=outcome,
            treatment=treatment,
            controls=controls,
            n_folds=n_folds,
            n_rep=1,
            random_state=rep * 42
        )
        estimates.append(result.effect)

    return {
        'mean': np.mean(estimates),
        'std': np.std(estimates),
        'cv': np.std(estimates) / abs(np.mean(estimates)),
        'range': (min(estimates), max(estimates)),
        'all_estimates': estimates,
        'stable': np.std(estimates) / abs(np.mean(estimates)) < 0.1
    }
```

### 1.3 Fold Assignment Sensitivity

Test if results depend on which observations end up in which fold:

```python
def fold_assignment_sensitivity(data, outcome, treatment, controls,
                                n_permutations=20):
    """
    Test sensitivity to fold assignment.
    """
    estimates = []

    for perm in range(n_permutations):
        # Different random fold assignment each time
        result = estimate_plr(
            data=data.sample(frac=1, random_state=perm),  # Shuffle
            outcome=outcome,
            treatment=treatment,
            controls=controls,
            random_state=perm
        )
        estimates.append(result.effect)

    # Kolmogorov-Smirnov test against normal
    ks_stat, ks_pval = stats.kstest(
        (np.array(estimates) - np.mean(estimates)) / np.std(estimates),
        'norm'
    )

    return {
        'mean': np.mean(estimates),
        'std': np.std(estimates),
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'normally_distributed': ks_pval > 0.05
    }
```

---

## 2. Nuisance Function Quality

### 2.1 Prediction Performance

```python
def nuisance_prediction_diagnostic(X, y, d, learner_y, learner_d, n_folds=5):
    """
    Assess nuisance function prediction quality via cross-validation.
    """
    from sklearn.model_selection import cross_val_predict, cross_val_score

    # Outcome model: E[Y|X]
    y_pred = cross_val_predict(learner_y, X, y, cv=n_folds)
    y_resid = y - y_pred
    r2_y = 1 - np.var(y_resid) / np.var(y)
    mse_y = np.mean(y_resid**2)

    # Treatment model: E[D|X] or P(D=1|X)
    is_binary = len(np.unique(d)) == 2

    if is_binary:
        d_pred = cross_val_predict(learner_d, X, d, cv=n_folds, method='predict_proba')[:, 1]
        # Brier score
        mse_d = np.mean((d - d_pred)**2)
        # McFadden's pseudo R2
        ll_full = np.sum(d * np.log(d_pred + 1e-10) + (1-d) * np.log(1-d_pred + 1e-10))
        ll_null = np.sum(d * np.log(d.mean()) + (1-d) * np.log(1-d.mean()))
        r2_d = 1 - ll_full / ll_null
    else:
        d_pred = cross_val_predict(learner_d, X, d, cv=n_folds)
        d_resid = d - d_pred
        r2_d = 1 - np.var(d_resid) / np.var(d)
        mse_d = np.mean(d_resid**2)

    return {
        'outcome_model': {
            'r2': r2_y,
            'mse': mse_y,
            'quality': 'GOOD' if r2_y > 0.3 else 'MODERATE' if r2_y > 0.1 else 'POOR'
        },
        'treatment_model': {
            'r2': r2_d,
            'mse': mse_d,
            'quality': 'GOOD' if r2_d > 0.3 else 'MODERATE' if r2_d > 0.1 else 'POOR'
        },
        'overall_quality': assess_overall_quality(r2_y, r2_d)
    }

def assess_overall_quality(r2_y, r2_d):
    """
    Assess overall nuisance quality.

    Low R2 isn't necessarily bad - it means conditional expectations
    don't explain much variance, which is fine for causal inference.
    But very low R2 for treatment model may indicate weak instruments
    for variation.
    """
    if r2_y < 0.05 and r2_d < 0.05:
        return ("CAUTION: Both nuisance functions have low R2. "
                "This is fine if X doesn't predict Y or D strongly, "
                "but verify this makes sense for your setting.")
    elif r2_d < 0.05:
        return ("NOTE: Treatment model has low R2. If D is nearly random "
                "conditional on X, DDML is appropriate. If you expect "
                "selection on X, reconsider controls.")
    else:
        return "OK: Nuisance functions capture meaningful variation."
```

### 2.2 Residual Diagnostics

```python
def residual_diagnostics(y, d, y_pred, d_pred):
    """
    Diagnostic tests on residuals from nuisance estimation.
    """
    y_resid = y - y_pred
    d_resid = d - d_pred

    diagnostics = {
        'outcome_residuals': {
            'mean': np.mean(y_resid),
            'std': np.std(y_resid),
            'skewness': stats.skew(y_resid),
            'kurtosis': stats.kurtosis(y_resid),
            'mean_zero_pval': stats.ttest_1samp(y_resid, 0).pvalue
        },
        'treatment_residuals': {
            'mean': np.mean(d_resid),
            'std': np.std(d_resid),
            'skewness': stats.skew(d_resid),
            'kurtosis': stats.kurtosis(d_resid),
            'mean_zero_pval': stats.ttest_1samp(d_resid, 0).pvalue
        }
    }

    # Cross-product of residuals (identification check)
    # E[(Y-l(X))(D-m(X))] should be non-zero for identification
    cross_resid = y_resid * d_resid
    diagnostics['cross_residuals'] = {
        'mean': np.mean(cross_resid),
        'se': np.std(cross_resid) / np.sqrt(len(cross_resid)),
        't_stat': np.mean(cross_resid) / (np.std(cross_resid) / np.sqrt(len(cross_resid))),
        'nonzero': abs(np.mean(cross_resid)) > 2 * np.std(cross_resid) / np.sqrt(len(cross_resid))
    }

    return diagnostics
```

### 2.3 Learner Comparison

```python
def compare_learner_performance(X, y, d, learner_names=None, cv_folds=5):
    """
    Compare multiple learners for nuisance estimation.
    """
    if learner_names is None:
        learner_names = ['lasso', 'ridge', 'random_forest', 'xgboost']

    results = {'outcome_model': {}, 'treatment_model': {}}
    is_binary = len(np.unique(d)) == 2

    for name in learner_names:
        # Outcome model
        try:
            learner_y = _get_learner(name, task='regression')
            scores_y = cross_val_score(learner_y, X, y, cv=cv_folds,
                                       scoring='neg_mean_squared_error')
            results['outcome_model'][name] = {
                'mse': -scores_y.mean(),
                'mse_std': scores_y.std()
            }
        except Exception as e:
            results['outcome_model'][name] = {'error': str(e)}

        # Treatment model
        try:
            task_d = 'classification' if is_binary else 'regression'
            learner_d = _get_learner(name, task=task_d)
            scoring_d = 'neg_brier_score' if is_binary else 'neg_mean_squared_error'
            scores_d = cross_val_score(learner_d, X, d, cv=cv_folds,
                                       scoring=scoring_d)
            results['treatment_model'][name] = {
                'mse': -scores_d.mean(),
                'mse_std': scores_d.std()
            }
        except Exception as e:
            results['treatment_model'][name] = {'error': str(e)}

    # Find best learners
    results['best_outcome_learner'] = min(
        results['outcome_model'].items(),
        key=lambda x: x[1].get('mse', np.inf)
    )[0]
    results['best_treatment_learner'] = min(
        results['treatment_model'].items(),
        key=lambda x: x[1].get('mse', np.inf)
    )[0]

    return results
```

---

## 3. Overlap and Propensity Diagnostics

### 3.1 Propensity Score Distribution

```python
def propensity_distribution_diagnostic(propensity_scores, treatment):
    """
    Comprehensive propensity score diagnostic.
    """
    ps = np.array(propensity_scores)
    d = np.array(treatment)

    # Overall distribution
    overall = {
        'min': ps.min(),
        'max': ps.max(),
        'mean': ps.mean(),
        'std': ps.std(),
        'median': np.median(ps),
        'iqr': np.percentile(ps, 75) - np.percentile(ps, 25)
    }

    # By treatment group
    by_group = {
        'treated': {
            'mean': ps[d == 1].mean(),
            'std': ps[d == 1].std(),
            'min': ps[d == 1].min()
        },
        'control': {
            'mean': ps[d == 0].mean(),
            'std': ps[d == 0].std(),
            'max': ps[d == 0].max()
        }
    }

    # Overlap statistics
    overlap = {
        'n_extreme_low': np.sum(ps < 0.01),
        'n_extreme_high': np.sum(ps > 0.99),
        'n_moderate_low': np.sum((ps >= 0.01) & (ps < 0.05)),
        'n_moderate_high': np.sum((ps > 0.95) & (ps <= 0.99)),
        'pct_in_good_range': np.mean((ps >= 0.1) & (ps <= 0.9)) * 100
    }

    # Effective sample size (Kish)
    weights_treated = d / ps
    weights_control = (1 - d) / (1 - ps)
    ess_treated = np.sum(weights_treated)**2 / np.sum(weights_treated**2)
    ess_control = np.sum(weights_control)**2 / np.sum(weights_control**2)

    return {
        'overall': overall,
        'by_group': by_group,
        'overlap': overlap,
        'effective_sample_size': {
            'treated': ess_treated,
            'control': ess_control,
            'total': ess_treated + ess_control
        },
        'recommendation': propensity_recommendation(overlap, len(ps))
    }

def propensity_recommendation(overlap, n):
    """Generate recommendation based on overlap statistics."""
    n_extreme = overlap['n_extreme_low'] + overlap['n_extreme_high']
    pct_extreme = n_extreme / n * 100

    if pct_extreme == 0:
        return "EXCELLENT: No extreme propensity scores"
    elif pct_extreme < 1:
        return f"GOOD: Only {n_extreme} ({pct_extreme:.1f}%) extreme propensities"
    elif pct_extreme < 5:
        return f"MODERATE: {n_extreme} ({pct_extreme:.1f}%) extreme propensities. Consider trimming."
    else:
        return f"WARNING: {n_extreme} ({pct_extreme:.1f}%) extreme propensities. Overlap assumption may be violated."
```

### 3.2 Common Support Visualization Data

```python
def common_support_data(propensity_scores, treatment, n_bins=50):
    """
    Generate data for common support visualization.
    """
    ps = np.array(propensity_scores)
    d = np.array(treatment)

    bins = np.linspace(0, 1, n_bins + 1)

    treated_hist, _ = np.histogram(ps[d == 1], bins=bins, density=True)
    control_hist, _ = np.histogram(ps[d == 0], bins=bins, density=True)

    # Find overlap region
    overlap_hist = np.minimum(treated_hist, control_hist)

    return {
        'bins': bins[:-1] + np.diff(bins) / 2,
        'treated_density': treated_hist,
        'control_density': control_hist,
        'overlap_density': overlap_hist,
        'overlap_ratio': np.sum(overlap_hist) / max(np.sum(treated_hist), np.sum(control_hist))
    }
```

---

## 4. Sensitivity Analysis

### 4.1 Learner Sensitivity

```python
def learner_sensitivity_analysis(data, outcome, treatment, controls,
                                 learner_list=None, model='plr'):
    """
    Assess sensitivity of results to learner choice.
    """
    if learner_list is None:
        learner_list = ['lasso', 'ridge', 'random_forest', 'xgboost']

    results = {}

    for learner in learner_list:
        try:
            if model == 'plr':
                result = estimate_plr(
                    data=data, outcome=outcome, treatment=treatment,
                    controls=controls, ml_l=learner, ml_m=learner
                )
            else:
                result = estimate_irm(
                    data=data, outcome=outcome, treatment=treatment,
                    controls=controls, ml_g=learner
                )

            results[learner] = {
                'effect': result.effect,
                'se': result.se,
                'ci': (result.ci_lower, result.ci_upper),
                'p_value': result.p_value
            }
        except Exception as e:
            results[learner] = {'error': str(e)}

    # Analyze sensitivity
    effects = [r['effect'] for r in results.values() if 'effect' in r]

    if len(effects) > 1:
        sensitivity = {
            'range': (min(effects), max(effects)),
            'spread': max(effects) - min(effects),
            'cv': np.std(effects) / abs(np.mean(effects)) if np.mean(effects) != 0 else np.inf,
            'all_same_sign': all(e > 0 for e in effects) or all(e < 0 for e in effects),
            'all_significant': all(r.get('p_value', 1) < 0.05 for r in results.values() if 'p_value' in r)
        }
    else:
        sensitivity = {'insufficient_results': True}

    return {'by_learner': results, 'sensitivity': sensitivity}
```

### 4.2 Fold Number Sensitivity

```python
def fold_number_sensitivity(data, outcome, treatment, controls,
                           fold_range=(2, 10)):
    """
    Test sensitivity to number of cross-fitting folds.
    """
    results = {}

    for n_folds in range(fold_range[0], fold_range[1] + 1):
        result = estimate_plr(
            data=data, outcome=outcome, treatment=treatment,
            controls=controls, n_folds=n_folds
        )
        results[n_folds] = {
            'effect': result.effect,
            'se': result.se,
            'ci': (result.ci_lower, result.ci_upper)
        }

    effects = [r['effect'] for r in results.values()]

    return {
        'by_n_folds': results,
        'range': (min(effects), max(effects)),
        'cv': np.std(effects) / abs(np.mean(effects)) if np.mean(effects) != 0 else np.inf,
        'stable': np.std(effects) / abs(np.mean(effects)) < 0.05
    }
```

### 4.3 Trimming Sensitivity

```python
def trimming_sensitivity(data, outcome, treatment, controls,
                         thresholds=[0.01, 0.02, 0.05, 0.10]):
    """
    Test sensitivity to propensity score trimming threshold.
    """
    results = {}

    for thresh in thresholds:
        result = estimate_irm(
            data=data, outcome=outcome, treatment=treatment,
            controls=controls, trimming_threshold=thresh
        )
        results[thresh] = {
            'effect': result.effect,
            'se': result.se,
            'n_trimmed': result.diagnostics.get('n_trimmed', 0),
            'ci': (result.ci_lower, result.ci_upper)
        }

    effects = [r['effect'] for r in results.values()]

    return {
        'by_threshold': results,
        'range': (min(effects), max(effects)),
        'sensitivity': 'LOW' if max(effects) - min(effects) < 0.1 * abs(np.mean(effects)) else 'MODERATE'
    }
```

---

## 5. Comprehensive Diagnostic Report

```python
def generate_diagnostic_report(data, outcome, treatment, controls,
                               result, propensity_scores=None):
    """
    Generate comprehensive DDML diagnostic report.
    """
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_summary': {
            'n_obs': len(data),
            'n_controls': len(controls),
            'outcome_mean': data[outcome].mean(),
            'outcome_std': data[outcome].std(),
            'treatment_mean': data[treatment].mean()
        },
        'estimation_result': {
            'effect': result.effect,
            'se': result.se,
            'ci': (result.ci_lower, result.ci_upper),
            'p_value': result.p_value
        },
        'diagnostics': {}
    }

    # Add propensity diagnostics if available
    if propensity_scores is not None:
        report['diagnostics']['propensity'] = propensity_distribution_diagnostic(
            propensity_scores, data[treatment].values
        )

    # Add nuisance quality
    report['diagnostics']['nuisance_quality'] = result.diagnostics

    # Generate warnings
    warnings = []

    if result.diagnostics.get('r2_d_given_x', 1) < 0.01:
        warnings.append("Very low R2 for treatment model - check identification")

    if propensity_scores is not None:
        ps = np.array(propensity_scores)
        if np.sum(ps < 0.05) / len(ps) > 0.1:
            warnings.append("Many propensity scores below 0.05 - overlap concern")

    report['warnings'] = warnings
    report['overall_status'] = 'OK' if len(warnings) == 0 else 'REVIEW NEEDED'

    return report
```

---

## References

1. Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*.

2. Kennedy, E. H. (2022). Semiparametric Doubly Robust Targeted Double Machine Learning: A Review.

3. Crump, R. K., et al. (2009). Dealing with Limited Overlap in Estimation of Average Treatment Effects. *Biometrika*.
