# Heterogeneity Analysis for Causal Forests

## Overview

Treatment effect heterogeneity refers to variation in causal effects across individuals or subgroups. This document covers CATE estimation, subgroup discovery, best linear projection, and methods for characterizing who benefits most from treatment.

---

## 1. CATE Estimation

### Definition

The Conditional Average Treatment Effect (CATE) is:
$$\tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]$$

This represents the expected treatment effect for individuals with characteristics $X = x$.

### Point Estimation

```python
from causal_forest import fit_causal_forest, estimate_cate

# Fit model
cf = fit_causal_forest(
    X=data[effect_modifiers],
    y=data[outcome],
    treatment=data[treatment]
)

# Estimate CATE for each observation
cate_results = estimate_cate(cf, X_test=data[effect_modifiers])

# Access results
tau_hat = cate_results.estimates      # Point estimates
tau_se = cate_results.std_errors      # Standard errors
tau_lower = cate_results.ci_lower     # 95% CI lower
tau_upper = cate_results.ci_upper     # 95% CI upper
```

### R grf Implementation

```r
library(grf)

# Fit causal forest
cf <- causal_forest(X, Y, W)

# Predict CATE with variance
pred <- predict(cf, X.test, estimate.variance = TRUE)

tau_hat <- pred$predictions
tau_se <- sqrt(pred$variance.estimates)

# Confidence intervals
ci_lower <- tau_hat - 1.96 * tau_se
ci_upper <- tau_hat + 1.96 * tau_se

# Summary statistics
cat("ATE:", mean(tau_hat), "\n")
cat("CATE std dev:", sd(tau_hat), "\n")
cat("Range:", range(tau_hat), "\n")
```

### Interpreting CATE Estimates

**Key metrics**:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Mean CATE | $\bar{\tau}$ | Average Treatment Effect (ATE) |
| CATE Std Dev | $\text{SD}(\tau)$ | Amount of heterogeneity |
| % Positive | $P(\tau > 0)$ | Proportion benefiting |
| % Significant | $P(\tau_{CI} \not\ni 0)$ | Proportion with significant effects |

```python
def summarize_cate(cate_results):
    """Generate CATE summary statistics."""
    tau = cate_results.estimates
    ci_lower = cate_results.ci_lower
    ci_upper = cate_results.ci_upper

    summary = {
        'ATE': np.mean(tau),
        'CATE_std': np.std(tau),
        'CATE_range': (np.min(tau), np.max(tau)),
        'pct_positive': np.mean(tau > 0),
        'pct_significant_positive': np.mean(ci_lower > 0),
        'pct_significant_negative': np.mean(ci_upper < 0),
        'pct_significant_any': np.mean((ci_lower > 0) | (ci_upper < 0))
    }

    return summary
```

---

## 2. Subgroup Discovery

### Automatic Subgroup Identification

Use CATE estimates to identify high-impact subgroups:

```python
def identify_subgroups(cate_results, X, n_groups=4):
    """
    Identify subgroups based on CATE quartiles.
    """
    tau = cate_results.estimates

    # Quartile-based groups
    quantiles = np.percentile(tau, [25, 50, 75])
    groups = np.digitize(tau, quantiles)

    # Summary by group
    group_summary = []
    for g in range(n_groups):
        mask = groups == g
        group_summary.append({
            'group': g + 1,
            'n': mask.sum(),
            'mean_cate': tau[mask].mean(),
            'cate_range': (tau[mask].min(), tau[mask].max()),
            'pct_significant': ((cate_results.ci_lower[mask] > 0) |
                               (cate_results.ci_upper[mask] < 0)).mean()
        })

    return pd.DataFrame(group_summary), groups
```

### Covariate Profiling

```python
def profile_subgroups(X, groups, feature_names):
    """
    Profile subgroups by covariate means.
    """
    if isinstance(X, pd.DataFrame):
        X_df = X
    else:
        X_df = pd.DataFrame(X, columns=feature_names)

    profiles = X_df.groupby(groups).mean()
    profiles.index = [f'Group {i+1}' for i in range(len(profiles))]

    return profiles

def visualize_subgroup_profiles(profiles, title="Subgroup Profiles"):
    """Create radar/spider chart of subgroup profiles."""
    import matplotlib.pyplot as plt
    from math import pi

    categories = list(profiles.columns)
    n_cats = len(categories)

    # Normalize for comparison
    profiles_norm = (profiles - profiles.min()) / (profiles.max() - profiles.min() + 1e-10)

    angles = [n / float(n_cats) * 2 * pi for n in range(n_cats)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.viridis(np.linspace(0, 1, len(profiles)))

    for idx, (group, values) in enumerate(profiles_norm.iterrows()):
        values_list = values.tolist()
        values_list += values_list[:1]
        ax.plot(angles, values_list, 'o-', linewidth=2,
                label=group, color=colors[idx])
        ax.fill(angles, values_list, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, size=16, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    return fig
```

### Sorted Group Average Treatment Effects (GATES)

```python
def compute_gates(cate_estimates, y, treatment, n_groups=5):
    """
    Compute Group Average Treatment Effects (GATES).

    Splits sample by predicted CATE and estimates ATE within each group.
    """
    tau = cate_estimates
    quantiles = np.percentile(tau, np.linspace(0, 100, n_groups + 1)[1:-1])
    groups = np.digitize(tau, quantiles)

    gates = []
    for g in range(n_groups):
        mask = groups == g
        y_g = y[mask]
        t_g = treatment[mask]

        # Simple difference in means within group
        ate_g = np.mean(y_g[t_g == 1]) - np.mean(y_g[t_g == 0])

        # Standard error
        n1 = np.sum(t_g == 1)
        n0 = np.sum(t_g == 0)
        var1 = np.var(y_g[t_g == 1]) / n1 if n1 > 1 else 0
        var0 = np.var(y_g[t_g == 0]) / n0 if n0 > 1 else 0
        se_g = np.sqrt(var1 + var0)

        gates.append({
            'group': g + 1,
            'n': mask.sum(),
            'predicted_cate_mean': tau[mask].mean(),
            'actual_ate': ate_g,
            'ate_se': se_g,
            'ci_lower': ate_g - 1.96 * se_g,
            'ci_upper': ate_g + 1.96 * se_g
        })

    return pd.DataFrame(gates)


def plot_gates(gates_df, title="Group Average Treatment Effects"):
    """Visualize GATES."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = gates_df['group']
    y = gates_df['actual_ate']
    yerr = 1.96 * gates_df['ate_se']

    ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, capthick=2,
                markersize=10, linewidth=2, color='steelblue')

    # Add predicted CATE line
    ax.plot(x, gates_df['predicted_cate_mean'], 's--',
            markersize=8, color='coral', label='Predicted CATE mean')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('CATE Quintile Group', fontsize=12)
    ax.set_ylabel('Treatment Effect', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.legend()

    plt.tight_layout()
    return fig
```

---

## 3. Best Linear Projection (BLP)

### Purpose

BLP summarizes how CATE varies with covariates by projecting onto a linear function:
$$\tau(x) \approx \beta_0 + \beta_1 A_1 + \beta_2 A_2 + ... + \beta_k A_k$$

### R grf Implementation

```r
library(grf)

# Fit causal forest
cf <- causal_forest(X, Y, W)

# Best linear projection onto selected variables
A <- X[, c("age", "income", "tenure")]  # Variables to project onto

blp <- best_linear_projection(cf, A)
print(blp)

# Output:
#                  Estimate  Std.Error  t.value   Pr(>|t|)
# (Intercept)       0.5234    0.0821    6.375    <2e-16
# age               0.0234    0.0045    5.200    <2e-16
# income            0.0001    0.00002   5.000    <2e-16
# tenure            0.0156    0.0089    1.753    0.0798
```

### Python Implementation

```python
def best_linear_projection(cf_model, A, X=None, cate_estimates=None):
    """
    Compute Best Linear Projection of CATE onto variables A.

    Parameters
    ----------
    cf_model : fitted causal forest
    A : array-like, variables to project onto
    X : effect modifiers (needed if cate_estimates not provided)
    cate_estimates : pre-computed CATE estimates

    Returns
    -------
    BLP coefficients and standard errors
    """
    from scipy import stats

    # Get CATE estimates
    if cate_estimates is None:
        cate_estimates = estimate_cate(cf_model, X)

    tau = cate_estimates.estimates
    tau_se = cate_estimates.std_errors

    if isinstance(A, pd.DataFrame):
        feature_names = list(A.columns)
        A = A.values
    else:
        feature_names = [f'A{i}' for i in range(A.shape[1])]

    # Add intercept
    A_with_intercept = np.column_stack([np.ones(len(A)), A])

    # Weighted least squares (inverse variance weighting)
    weights = 1 / (tau_se**2 + 1e-10)
    weights = weights / weights.sum() * len(weights)

    W = np.diag(weights)

    # WLS solution
    AtWA = A_with_intercept.T @ W @ A_with_intercept
    AtWy = A_with_intercept.T @ W @ tau

    beta = np.linalg.solve(AtWA, AtWy)

    # Robust standard errors
    residuals = tau - A_with_intercept @ beta
    n, k = A_with_intercept.shape

    # Sandwich estimator
    bread = np.linalg.inv(AtWA)
    meat = A_with_intercept.T @ np.diag((weights * residuals)**2) @ A_with_intercept
    V = bread @ meat @ bread
    se = np.sqrt(np.diag(V))

    # Test statistics
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))

    # R-squared
    ss_res = np.sum(weights * residuals**2)
    ss_tot = np.sum(weights * (tau - np.average(tau, weights=weights))**2)
    r_squared = 1 - ss_res / ss_tot

    results = pd.DataFrame({
        'variable': ['intercept'] + feature_names,
        'coefficient': beta,
        'std_error': se,
        't_statistic': t_stats,
        'p_value': p_values
    })

    return results, r_squared
```

### Interpreting BLP Results

**Coefficient interpretation**:
- $\beta_j > 0$: CATE increases with variable $A_j$
- $\beta_j < 0$: CATE decreases with variable $A_j$
- Magnitude: 1-unit increase in $A_j$ associated with $\beta_j$ change in CATE

**R-squared**:
- How much heterogeneity is explained by linear projection
- Low R-squared doesn't mean no heterogeneity (may be nonlinear)

---

## 4. Variable Importance for Heterogeneity

### Forest-Based Importance

```python
def variable_importance_heterogeneity(cf_model):
    """
    Extract variable importance for driving heterogeneity.
    """
    importance = cf_model.feature_importances_

    results = pd.DataFrame({
        'variable': cf_model.feature_names_,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return results
```

### R grf Variable Importance

```r
# Variable importance
var_imp <- variable_importance(cf)

# Plot
importance_df <- data.frame(
  variable = colnames(X),
  importance = var_imp
)
importance_df <- importance_df[order(-importance_df$importance), ]

barplot(importance_df$importance,
        names.arg = importance_df$variable,
        main = "Heterogeneity Drivers",
        xlab = "Variable",
        ylab = "Importance")
```

### SHAP Values for CATE

```python
def compute_cate_shap(cf_model, X, n_samples=100):
    """
    Compute SHAP values for CATE predictions.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap package required: pip install shap")

    # Create wrapper for CATE prediction
    def cate_predict(X_input):
        tau, _ = cf_model.predict(X_input, return_std=False)
        return tau

    # Background data
    if len(X) > n_samples:
        background = X[np.random.choice(len(X), n_samples, replace=False)]
    else:
        background = X

    # SHAP explainer
    explainer = shap.KernelExplainer(cate_predict, background)
    shap_values = explainer.shap_values(X)

    return shap_values, explainer

# Usage
shap_values, explainer = compute_cate_shap(cf_model, X)
shap.summary_plot(shap_values, X, feature_names=feature_names)
```

---

## 5. Partial Dependence for CATE

### Single Variable Dependence

```python
def cate_partial_dependence(cf_model, X, variable, n_grid=50):
    """
    Compute partial dependence of CATE on a single variable.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
        if isinstance(variable, str):
            var_idx = list(X.columns).index(variable)
            var_name = variable
        else:
            var_idx = variable
            var_name = X.columns[var_idx]
    else:
        X_arr = X
        var_idx = variable
        var_name = f'X{var_idx}'

    # Grid of values
    x_values = X_arr[:, var_idx]
    grid = np.linspace(np.percentile(x_values, 5),
                       np.percentile(x_values, 95),
                       n_grid)

    # Compute PD
    pd_values = []
    pd_se = []

    for val in grid:
        X_temp = X_arr.copy()
        X_temp[:, var_idx] = val
        tau, se = cf_model.predict(X_temp, return_std=True)
        pd_values.append(np.mean(tau))
        pd_se.append(np.mean(se) / np.sqrt(len(tau)))

    return {
        'grid': grid,
        'pd_values': np.array(pd_values),
        'pd_se': np.array(pd_se),
        'variable_name': var_name
    }


def plot_cate_partial_dependence(pd_result, save_path=None):
    """Plot CATE partial dependence."""
    fig, ax = plt.subplots(figsize=(10, 6))

    grid = pd_result['grid']
    pd_vals = pd_result['pd_values']
    pd_se = pd_result['pd_se']

    ax.plot(grid, pd_vals, 'b-', linewidth=2, label='CATE')
    ax.fill_between(grid,
                    pd_vals - 1.96 * pd_se,
                    pd_vals + 1.96 * pd_se,
                    alpha=0.2, color='blue')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel(pd_result['variable_name'], fontsize=12)
    ax.set_ylabel('CATE', fontsize=12)
    ax.set_title(f'How Treatment Effect Varies with {pd_result["variable_name"]}',
                fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
```

### Two-Way Interaction

```python
def cate_interaction_plot(cf_model, X, var1, var2, n_grid=20):
    """
    Compute 2D partial dependence for CATE interaction.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
        idx1 = list(X.columns).index(var1) if isinstance(var1, str) else var1
        idx2 = list(X.columns).index(var2) if isinstance(var2, str) else var2
    else:
        X_arr = X
        idx1, idx2 = var1, var2

    # Create grids
    grid1 = np.linspace(np.percentile(X_arr[:, idx1], 5),
                        np.percentile(X_arr[:, idx1], 95), n_grid)
    grid2 = np.linspace(np.percentile(X_arr[:, idx2], 5),
                        np.percentile(X_arr[:, idx2], 95), n_grid)

    # Compute 2D PD
    pd_matrix = np.zeros((n_grid, n_grid))

    for i, v1 in enumerate(grid1):
        for j, v2 in enumerate(grid2):
            X_temp = X_arr.copy()
            X_temp[:, idx1] = v1
            X_temp[:, idx2] = v2
            tau, _ = cf_model.predict(X_temp, return_std=False)
            pd_matrix[i, j] = np.mean(tau)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.contourf(grid2, grid1, pd_matrix, levels=20, cmap='RdBu_r')
    plt.colorbar(c, label='CATE')
    ax.set_xlabel(var2 if isinstance(var2, str) else f'X{var2}')
    ax.set_ylabel(var1 if isinstance(var1, str) else f'X{var1}')
    ax.set_title('CATE Interaction Surface')

    return fig, pd_matrix
```

---

## 6. Characterizing High-Value Subgroups

### Threshold Analysis

```python
def characterize_high_cate(cate_results, X, threshold='top_quartile'):
    """
    Characterize individuals with high treatment effects.
    """
    tau = cate_results.estimates

    if threshold == 'top_quartile':
        high_mask = tau >= np.percentile(tau, 75)
    elif threshold == 'positive_significant':
        high_mask = cate_results.ci_lower > 0
    elif isinstance(threshold, (int, float)):
        high_mask = tau >= threshold
    else:
        raise ValueError(f"Unknown threshold: {threshold}")

    if isinstance(X, pd.DataFrame):
        X_high = X[high_mask]
        X_low = X[~high_mask]
    else:
        X_high = X[high_mask]
        X_low = X[~high_mask]

    # Compare distributions
    comparison = []
    for col in (X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1])):
        high_mean = X_high[col].mean() if isinstance(X, pd.DataFrame) else X_high[:, col].mean()
        low_mean = X_low[col].mean() if isinstance(X, pd.DataFrame) else X_low[:, col].mean()
        pooled_std = np.sqrt((np.var(X_high[col] if isinstance(X, pd.DataFrame) else X_high[:, col]) +
                              np.var(X_low[col] if isinstance(X, pd.DataFrame) else X_low[:, col])) / 2)

        comparison.append({
            'variable': col,
            'high_cate_mean': high_mean,
            'low_cate_mean': low_mean,
            'std_diff': (high_mean - low_mean) / (pooled_std + 1e-10)
        })

    return pd.DataFrame(comparison).sort_values('std_diff', key=abs, ascending=False)
```

### Rule Extraction via Decision Tree

```python
def extract_cate_rules(cate_results, X, max_depth=3):
    """
    Extract interpretable rules for high CATE using decision tree.
    """
    from sklearn.tree import DecisionTreeRegressor, export_text

    tau = cate_results.estimates

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = X.values
    else:
        feature_names = [f'X{i}' for i in range(X.shape[1])]
        X_arr = X

    # Fit decision tree on CATE
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=50)
    tree.fit(X_arr, tau)

    # Extract rules
    rules_text = export_text(tree, feature_names=feature_names)

    # Also get leaf-level summaries
    leaves = tree.apply(X_arr)
    leaf_summary = []

    for leaf_id in np.unique(leaves):
        mask = leaves == leaf_id
        leaf_summary.append({
            'leaf_id': leaf_id,
            'n': mask.sum(),
            'mean_cate': tau[mask].mean(),
            'std_cate': tau[mask].std()
        })

    return rules_text, pd.DataFrame(leaf_summary).sort_values('mean_cate', ascending=False)
```

---

## 7. Visualization Suite

### Comprehensive CATE Visualization

```python
def create_heterogeneity_dashboard(cf_model, cate_results, X, output_dir=None):
    """
    Create comprehensive heterogeneity visualization dashboard.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    tau = cate_results.estimates
    feature_names = cf_model.feature_names_

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. CATE distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(tau, bins=50, density=True, alpha=0.7, color='steelblue')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(tau), color='green', linewidth=2)
    ax1.set_xlabel('CATE')
    ax1.set_ylabel('Density')
    ax1.set_title('CATE Distribution')

    # 2. Sorted CATEs with CI
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_idx = np.argsort(tau)
    ax2.fill_between(range(len(tau)),
                     cate_results.ci_lower[sorted_idx],
                     cate_results.ci_upper[sorted_idx],
                     alpha=0.3, color='steelblue')
    ax2.plot(tau[sorted_idx], color='steelblue', linewidth=0.5)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel('Observation (sorted)')
    ax2.set_ylabel('CATE')
    ax2.set_title('Sorted CATEs with 95% CI')

    # 3. Variable importance
    ax3 = fig.add_subplot(gs[0, 2])
    importance = cf_model.feature_importances_
    sorted_idx_imp = np.argsort(importance)[::-1]
    ax3.barh(range(len(importance)),
             importance[sorted_idx_imp],
             color='steelblue')
    ax3.set_yticks(range(len(importance)))
    ax3.set_yticklabels([feature_names[i] for i in sorted_idx_imp])
    ax3.invert_yaxis()
    ax3.set_xlabel('Importance')
    ax3.set_title('Heterogeneity Drivers')

    # 4-6. Top 3 variable partial dependence
    top_vars = sorted_idx_imp[:3]
    for i, var_idx in enumerate(top_vars):
        ax = fig.add_subplot(gs[1, i])
        pd_result = cate_partial_dependence(cf_model, X, var_idx)
        ax.plot(pd_result['grid'], pd_result['pd_values'], 'b-', linewidth=2)
        ax.fill_between(pd_result['grid'],
                        pd_result['pd_values'] - 1.96 * pd_result['pd_se'],
                        pd_result['pd_values'] + 1.96 * pd_result['pd_se'],
                        alpha=0.2)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel(feature_names[var_idx])
        ax.set_ylabel('CATE')
        ax.set_title(f'PD: {feature_names[var_idx]}')

    # 7. GATES
    ax7 = fig.add_subplot(gs[2, 0:2])
    gates = compute_gates(tau, y, treatment)
    ax7.errorbar(gates['group'], gates['actual_ate'],
                yerr=1.96 * gates['ate_se'],
                fmt='o-', capsize=5, markersize=10)
    ax7.axhline(0, color='gray', linestyle='--')
    ax7.set_xlabel('CATE Quintile')
    ax7.set_ylabel('Group ATE')
    ax7.set_title('Group Average Treatment Effects (GATES)')

    # 8. Summary statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    summary_text = f"""
    CATE Summary Statistics

    Mean (ATE): {np.mean(tau):.4f}
    Std Dev: {np.std(tau):.4f}
    Min: {np.min(tau):.4f}
    Max: {np.max(tau):.4f}

    % Positive: {np.mean(tau > 0):.1%}
    % Significant: {cate_results.proportion_significant:.1%}

    IQR: [{np.percentile(tau, 25):.4f}, {np.percentile(tau, 75):.4f}]
    """
    ax8.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center')

    plt.suptitle('Treatment Effect Heterogeneity Dashboard', fontsize=16, y=1.02)

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'heterogeneity_dashboard.png'),
                   dpi=150, bbox_inches='tight')

    return fig
```

---

## Summary

**Key steps for heterogeneity analysis**:

1. **Estimate CATEs** with confidence intervals
2. **Test for heterogeneity** (omnibus test)
3. **Identify drivers** via variable importance and BLP
4. **Characterize subgroups** using GATES and profiling
5. **Visualize** partial dependence and distributions
6. **Extract rules** for actionable insights

---

## References

1. Chernozhukov, V., Demirer, M., Duflo, E., & Fernandez-Val, I. (2018). Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments.

2. Athey, S., & Imbens, G. W. (2019). Machine Learning Methods Economists Should Know About.

3. grf documentation: https://grf-labs.github.io/grf/articles/grf.html
