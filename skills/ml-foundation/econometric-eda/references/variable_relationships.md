# Variable Relationships for Econometric Research

## Overview

Understanding variable relationships is crucial for model specification, identifying potential confounders, and assessing the validity of causal identification strategies.

## Correlation Analysis

### Pearson Correlation

Linear correlation for continuous variables.

```python
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def pearson_correlation_matrix(data, columns=None, significance_level=0.05):
    """
    Calculate Pearson correlation matrix with significance testing.

    Returns:
    --------
    dict with correlation matrix, p-values, and significance indicators
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    subset = data[columns].dropna()
    n_vars = len(columns)

    # Calculate correlations and p-values
    corr_matrix = np.zeros((n_vars, n_vars))
    pval_matrix = np.zeros((n_vars, n_vars))

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            else:
                corr, pval = stats.pearsonr(subset[col1], subset[col2])
                corr_matrix[i, j] = corr
                pval_matrix[i, j] = pval

    corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
    pval_df = pd.DataFrame(pval_matrix, index=columns, columns=columns)

    # Significance stars
    def get_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    sig_df = pval_df.applymap(get_stars)

    return {
        'correlation': corr_df,
        'pvalues': pval_df,
        'significance': sig_df,
        'n_observations': len(subset)
    }
```

### Spearman Rank Correlation

Non-parametric correlation for monotonic relationships.

```python
def spearman_correlation_matrix(data, columns=None):
    """
    Calculate Spearman rank correlation matrix.
    Robust to outliers and non-linear monotonic relationships.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    subset = data[columns].dropna()
    n_vars = len(columns)

    corr_matrix = np.zeros((n_vars, n_vars))
    pval_matrix = np.zeros((n_vars, n_vars))

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            else:
                corr, pval = stats.spearmanr(subset[col1], subset[col2])
                corr_matrix[i, j] = corr
                pval_matrix[i, j] = pval

    return {
        'correlation': pd.DataFrame(corr_matrix, index=columns, columns=columns),
        'pvalues': pd.DataFrame(pval_matrix, index=columns, columns=columns)
    }
```

### Partial Correlation

Correlation controlling for other variables.

```python
def partial_correlation(data, var1, var2, control_vars):
    """
    Calculate partial correlation between var1 and var2
    controlling for control_vars.

    Uses regression-based approach:
    1. Regress var1 on controls, get residuals
    2. Regress var2 on controls, get residuals
    3. Correlate residuals
    """
    import statsmodels.api as sm

    subset = data[[var1, var2] + control_vars].dropna()

    # Residualize var1
    X1 = sm.add_constant(subset[control_vars])
    model1 = sm.OLS(subset[var1], X1).fit()
    resid1 = model1.resid

    # Residualize var2
    X2 = sm.add_constant(subset[control_vars])
    model2 = sm.OLS(subset[var2], X2).fit()
    resid2 = model2.resid

    # Partial correlation
    partial_corr, pval = stats.pearsonr(resid1, resid2)

    return {
        'partial_correlation': partial_corr,
        'pvalue': pval,
        'n_observations': len(subset),
        'control_vars': control_vars
    }


def partial_correlation_matrix(data, columns, control_vars):
    """
    Calculate partial correlation matrix controlling for specified variables.
    """
    n_vars = len(columns)
    partial_corr = np.zeros((n_vars, n_vars))
    pval_matrix = np.zeros((n_vars, n_vars))

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                partial_corr[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            elif i < j:
                result = partial_correlation(data, col1, col2, control_vars)
                partial_corr[i, j] = result['partial_correlation']
                partial_corr[j, i] = result['partial_correlation']
                pval_matrix[i, j] = result['pvalue']
                pval_matrix[j, i] = result['pvalue']

    return {
        'partial_correlation': pd.DataFrame(partial_corr, index=columns, columns=columns),
        'pvalues': pd.DataFrame(pval_matrix, index=columns, columns=columns)
    }
```

### Point-Biserial Correlation

For binary-continuous variable pairs.

```python
def point_biserial_correlation(data, binary_var, continuous_var):
    """
    Calculate point-biserial correlation between binary and continuous variable.
    """
    subset = data[[binary_var, continuous_var]].dropna()

    # Ensure binary variable is 0/1
    unique_vals = subset[binary_var].unique()
    if len(unique_vals) != 2:
        raise ValueError(f"{binary_var} must be binary")

    corr, pval = stats.pointbiserialr(subset[binary_var], subset[continuous_var])

    return {
        'correlation': corr,
        'pvalue': pval,
        'n_observations': len(subset)
    }
```

## Multicollinearity Diagnostics

### Variance Inflation Factor (VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def calculate_vif(data, columns):
    """
    Calculate Variance Inflation Factor for each variable.

    VIF > 5: Moderate multicollinearity
    VIF > 10: Severe multicollinearity
    """
    subset = data[columns].dropna()
    X = sm.add_constant(subset)

    vif_data = []
    for i, col in enumerate(X.columns):
        if col == 'const':
            continue
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({
            'variable': col,
            'vif': vif,
            'tolerance': 1 / vif,
            'r_squared': 1 - 1/vif,  # R^2 from regressing this var on others
            'concern': 'severe' if vif > 10 else ('moderate' if vif > 5 else 'low')
        })

    return pd.DataFrame(vif_data)
```

### Condition Number

```python
def condition_number(data, columns):
    """
    Calculate condition number of the design matrix.

    Condition number > 30: Multicollinearity concern
    Condition number > 100: Severe multicollinearity
    """
    subset = data[columns].dropna()
    X = sm.add_constant(subset)

    # Standardize for numerical stability
    X_scaled = (X - X.mean()) / X.std()
    X_scaled['const'] = 1

    # Singular values
    _, s, _ = np.linalg.svd(X_scaled)

    condition_num = s.max() / s.min()

    return {
        'condition_number': condition_num,
        'singular_values': s,
        'concern': 'severe' if condition_num > 100 else ('moderate' if condition_num > 30 else 'low')
    }
```

### Eigenvalue Analysis

```python
def eigenvalue_analysis(data, columns):
    """
    Eigenvalue decomposition for multicollinearity diagnosis.
    Small eigenvalues indicate near-linear dependencies.
    """
    subset = data[columns].dropna()
    X = sm.add_constant(subset)

    # Correlation matrix of predictors
    corr_matrix = X.corr()

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Condition indices
    condition_indices = np.sqrt(eigenvalues.max() / eigenvalues)

    # Variance proportions
    variance_proportions = []
    for i, ev in enumerate(eigenvectors.T):
        props = ev**2 / (eigenvalues[i] * (ev**2).sum())
        variance_proportions.append(props)

    return {
        'eigenvalues': eigenvalues,
        'condition_indices': condition_indices,
        'eigenvectors': pd.DataFrame(eigenvectors, index=corr_matrix.columns),
        'variance_proportions': pd.DataFrame(variance_proportions, columns=corr_matrix.columns)
    }
```

## Scatterplot Matrices

```python
def create_scatterplot_matrix(data, columns, hue=None, output_path=None):
    """
    Create scatterplot matrix (pair plot) with distributions.

    Parameters:
    -----------
    hue : str, optional
        Variable for color coding (e.g., treatment indicator)
    """
    subset = data[columns + ([hue] if hue else [])].dropna()

    g = sns.pairplot(
        subset,
        hue=hue,
        diag_kind='kde',
        plot_kws={'alpha': 0.5, 's': 20},
        corner=True
    )

    g.fig.suptitle('Scatterplot Matrix', y=1.02)

    if output_path:
        g.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return g
```

### Correlation Heatmap

```python
def correlation_heatmap(data, columns, method='pearson', annotate=True,
                        significance=True, output_path=None):
    """
    Create correlation heatmap with optional significance indicators.
    """
    subset = data[columns].dropna()

    # Calculate correlations
    if method == 'pearson':
        result = pearson_correlation_matrix(data, columns)
    else:
        result = spearman_correlation_matrix(data, columns)

    corr = result['correlation']
    pvals = result['pvalues']

    # Create annotation matrix
    if annotate and significance:
        annot = corr.round(2).astype(str)
        for i in range(len(columns)):
            for j in range(len(columns)):
                if i != j:
                    stars = ''
                    if pvals.iloc[i, j] < 0.001:
                        stars = '***'
                    elif pvals.iloc[i, j] < 0.01:
                        stars = '**'
                    elif pvals.iloc[i, j] < 0.05:
                        stars = '*'
                    annot.iloc[i, j] = f"{corr.iloc[i, j]:.2f}{stars}"
    else:
        annot = annotate

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        annot=annot if annotate else False,
        fmt='' if significance else '.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(f'{method.capitalize()} Correlation Matrix')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig
```

## Non-Linear Relationship Detection

### Polynomial Features Test

```python
def test_nonlinearity(data, x_var, y_var, max_degree=3):
    """
    Test for non-linear relationships using polynomial regression.
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    subset = data[[x_var, y_var]].dropna()
    X = subset[[x_var]].values
    y = subset[y_var].values

    results = []

    for degree in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        r2 = r2_score(y, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - degree - 1)

        results.append({
            'degree': degree,
            'r2': r2,
            'adj_r2': adj_r2
        })

    results_df = pd.DataFrame(results)

    # Test if higher-order terms are significant
    # Compare nested models using F-test
    model_comparisons = []
    for i in range(1, len(results)):
        r2_restricted = results[i-1]['r2']
        r2_unrestricted = results[i]['r2']
        df_num = 1  # One additional parameter
        df_denom = len(y) - results[i]['degree'] - 1

        f_stat = ((r2_unrestricted - r2_restricted) / df_num) / ((1 - r2_unrestricted) / df_denom)
        p_value = 1 - stats.f.cdf(f_stat, df_num, df_denom)

        model_comparisons.append({
            'comparison': f"degree {results[i]['degree']} vs {results[i-1]['degree']}",
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

    return {
        'fit_statistics': results_df,
        'model_comparisons': pd.DataFrame(model_comparisons)
    }
```

### Locally Weighted Scatterplot Smoothing (LOWESS)

```python
from statsmodels.nonparametric.smoothers_lowess import lowess

def lowess_plot(data, x_var, y_var, frac=0.3, output_path=None):
    """
    Create scatterplot with LOWESS smoother to visualize non-linear patterns.
    """
    subset = data[[x_var, y_var]].dropna()

    # Calculate LOWESS
    smoothed = lowess(subset[y_var], subset[x_var], frac=frac)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(subset[x_var], subset[y_var], alpha=0.5, s=20, label='Data')
    ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, label='LOWESS')

    # Add linear fit for comparison
    z = np.polyfit(subset[x_var], subset[y_var], 1)
    p = np.poly1d(z)
    x_line = np.linspace(subset[x_var].min(), subset[x_var].max(), 100)
    ax.plot(x_line, p(x_line), 'g--', linewidth=2, label='Linear fit')

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(f'Relationship: {y_var} vs {x_var}')
    ax.legend()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig
```

## Causal Inference Considerations

### Covariate Balance Assessment

```python
def covariate_balance_table(data, treatment_var, covariates):
    """
    Create covariate balance table comparing treatment and control groups.

    Essential for assessing selection bias before matching/weighting.
    """
    results = []

    for cov in covariates:
        treated = data.loc[data[treatment_var] == 1, cov]
        control = data.loc[data[treatment_var] == 0, cov]

        # Means
        mean_t = treated.mean()
        mean_c = control.mean()

        # Standardized difference
        pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
        std_diff = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0

        # Variance ratio
        var_ratio = treated.var() / control.var() if control.var() > 0 else np.nan

        # Statistical test
        if data[cov].dtype in ['float64', 'int64']:
            stat, pval = stats.ttest_ind(treated.dropna(), control.dropna())
            test_type = 't-test'
        else:
            contingency = pd.crosstab(data[cov], data[treatment_var])
            stat, pval, _, _ = stats.chi2_contingency(contingency)
            test_type = 'chi2'

        results.append({
            'variable': cov,
            'mean_treated': mean_t,
            'mean_control': mean_c,
            'std_diff': std_diff,
            'var_ratio': var_ratio,
            'test_type': test_type,
            'test_stat': stat,
            'p_value': pval,
            'balanced': abs(std_diff) < 0.1  # Common threshold
        })

    return pd.DataFrame(results)
```

### Overlap Assessment

```python
def overlap_diagnostic(data, treatment_var, covariate, n_bins=50, output_path=None):
    """
    Visualize covariate overlap between treatment and control groups.
    """
    treated = data.loc[data[treatment_var] == 1, covariate]
    control = data.loc[data[treatment_var] == 0, covariate]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overlapping histograms
    axes[0].hist(control, bins=n_bins, alpha=0.5, density=True, label='Control')
    axes[0].hist(treated, bins=n_bins, alpha=0.5, density=True, label='Treated')
    axes[0].set_xlabel(covariate)
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution Overlap')
    axes[0].legend()

    # Empirical CDFs
    sorted_c = np.sort(control.dropna())
    sorted_t = np.sort(treated.dropna())
    axes[1].plot(sorted_c, np.arange(1, len(sorted_c)+1)/len(sorted_c), label='Control')
    axes[1].plot(sorted_t, np.arange(1, len(sorted_t)+1)/len(sorted_t), label='Treated')
    axes[1].set_xlabel(covariate)
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Empirical CDFs')
    axes[1].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(treated.dropna(), control.dropna())

    return {
        'figure': fig,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval
    }
```

## Summary Tables

### Variable Relationship Checklist for Causal Inference

| Check | Purpose | Threshold/Concern |
|-------|---------|-------------------|
| Treatment-covariate correlation | Confounding potential | Any significant correlation |
| Outcome-covariate correlation | Predictive power | For precision, not bias |
| Covariate multicollinearity | Model stability | VIF > 10 |
| Treatment-outcome raw correlation | Unadjusted association | Just descriptive |
| Covariate balance | Selection bias | Std diff > 0.1 |
| Overlap (common support) | Positivity violation | Check distributions |

### Recommended Workflow

1. **Univariate**: Check distributions, missing patterns
2. **Bivariate**: Correlations, scatterplots, balance tables
3. **Multivariate**: VIF, condition number, overlap
4. **Non-linearity**: LOWESS, polynomial tests
5. **Report**: Document all findings for transparency
