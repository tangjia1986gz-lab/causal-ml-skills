# Data Quality Assessment for Econometric Research

## Overview

Data quality assessment is the foundation of credible econometric analysis. Poor data quality can lead to biased estimates, incorrect standard errors, and invalid causal inferences.

## Missing Values

### Types of Missing Data Mechanisms

#### Missing Completely at Random (MCAR)
- **Definition**: Missingness is independent of both observed and unobserved data
- **Test**: Little's MCAR test
- **Implications**: Complete case analysis is valid but inefficient
- **Example**: Survey responses lost due to random data entry errors

```python
from scipy import stats
import numpy as np

def littles_mcar_test(data):
    """
    Perform Little's MCAR test.
    H0: Data is MCAR
    """
    # Group data by missing patterns
    patterns = data.isna().apply(lambda x: ''.join(x.astype(int).astype(str)), axis=1)
    unique_patterns = patterns.unique()

    # Calculate test statistic (simplified version)
    chi2_stat = 0
    df = 0

    for pattern in unique_patterns:
        subset = data[patterns == pattern]
        if len(subset) > 1:
            observed_cols = [col for col in data.columns if not subset[col].isna().all()]
            for col in observed_cols:
                col_data = subset[col].dropna()
                if len(col_data) > 0:
                    overall_mean = data[col].mean()
                    overall_var = data[col].var()
                    if overall_var > 0:
                        chi2_stat += len(col_data) * ((col_data.mean() - overall_mean)**2) / overall_var
                        df += 1

    p_value = 1 - stats.chi2.cdf(chi2_stat, df) if df > 0 else 1.0
    return {'chi2': chi2_stat, 'df': df, 'p_value': p_value}
```

#### Missing at Random (MAR)
- **Definition**: Missingness depends on observed data but not on unobserved data
- **Test**: Compare distributions of observed variables across missing/non-missing groups
- **Implications**: Multiple imputation or inverse probability weighting is valid
- **Example**: Income missing more often for older respondents (age is observed)

```python
def test_mar_patterns(data, target_var, covariates):
    """
    Test for MAR by comparing covariates between missing and non-missing groups.
    """
    results = {}
    missing_mask = data[target_var].isna()

    for cov in covariates:
        if data[cov].dtype in ['float64', 'int64']:
            # T-test for continuous variables
            group_missing = data.loc[missing_mask, cov].dropna()
            group_observed = data.loc[~missing_mask, cov].dropna()
            if len(group_missing) > 1 and len(group_observed) > 1:
                stat, pval = stats.ttest_ind(group_missing, group_observed)
                results[cov] = {'test': 't-test', 'statistic': stat, 'p_value': pval}
        else:
            # Chi-square test for categorical variables
            contingency = pd.crosstab(data[cov], missing_mask)
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2, pval, dof, expected = stats.chi2_contingency(contingency)
                results[cov] = {'test': 'chi2', 'statistic': chi2, 'p_value': pval}

    return results
```

#### Missing Not at Random (MNAR)
- **Definition**: Missingness depends on the unobserved value itself
- **Test**: Cannot be directly tested; requires domain knowledge and sensitivity analysis
- **Implications**: Standard methods produce biased estimates; need selection models
- **Example**: High earners less likely to report income (income itself determines missingness)

### Missing Data Patterns

```python
import missingno as msno
import matplotlib.pyplot as plt

def analyze_missing_patterns(data, output_dir=None):
    """
    Comprehensive missing data pattern analysis.
    """
    results = {
        'summary': {},
        'patterns': {},
        'correlations': {}
    }

    # Summary statistics
    missing_counts = data.isna().sum()
    missing_pct = (missing_counts / len(data)) * 100
    results['summary'] = pd.DataFrame({
        'missing_count': missing_counts,
        'missing_pct': missing_pct,
        'complete_count': len(data) - missing_counts
    })

    # Missing patterns (combinations of missing variables)
    pattern_df = data.isna().astype(int)
    patterns = pattern_df.groupby(list(pattern_df.columns)).size().reset_index(name='count')
    patterns['pct'] = (patterns['count'] / len(data)) * 100
    results['patterns'] = patterns.sort_values('count', ascending=False)

    # Missing correlations (which variables tend to be missing together)
    missing_corr = data.isna().corr()
    results['correlations'] = missing_corr

    # Visualizations
    if output_dir:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Bar chart of missing percentages
        missing_pct.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Missing Values by Variable')
        axes[0, 0].set_ylabel('Percentage Missing')

        # Matrix plot
        msno.matrix(data, ax=axes[0, 1], sparkline=False)
        axes[0, 1].set_title('Missing Value Matrix')

        # Heatmap of missing correlations
        import seaborn as sns
        sns.heatmap(missing_corr, annot=True, cmap='RdYlBu_r', ax=axes[1, 0],
                    center=0, vmin=-1, vmax=1)
        axes[1, 0].set_title('Missing Value Correlations')

        # Dendrogram
        msno.dendrogram(data, ax=axes[1, 1])
        axes[1, 1].set_title('Missing Value Clustering')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/missing_patterns.png', dpi=150, bbox_inches='tight')
        plt.close()

    return results
```

## Duplicate Detection

### Exact Duplicates

```python
def find_exact_duplicates(data, subset=None):
    """
    Find exact duplicate rows.

    Parameters:
    -----------
    data : DataFrame
    subset : list, optional
        Columns to consider for duplicate detection
    """
    duplicates = data.duplicated(subset=subset, keep=False)
    duplicate_df = data[duplicates].sort_values(by=subset or data.columns.tolist())

    return {
        'n_duplicates': duplicates.sum(),
        'n_unique_duplicated': data.duplicated(subset=subset, keep='first').sum(),
        'duplicate_rows': duplicate_df
    }
```

### Fuzzy Duplicates

```python
from difflib import SequenceMatcher

def find_fuzzy_duplicates(data, columns, threshold=0.9):
    """
    Find near-duplicate rows based on string similarity.
    """
    potential_duplicates = []

    # Create combined string for comparison
    data['_combined'] = data[columns].astype(str).agg(' '.join, axis=1)

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            similarity = SequenceMatcher(
                None,
                data['_combined'].iloc[i],
                data['_combined'].iloc[j]
            ).ratio()

            if similarity >= threshold:
                potential_duplicates.append({
                    'index_1': data.index[i],
                    'index_2': data.index[j],
                    'similarity': similarity
                })

    data.drop('_combined', axis=1, inplace=True)
    return pd.DataFrame(potential_duplicates)
```

## Data Inconsistencies

### Type Validation

```python
def validate_data_types(data, expected_types):
    """
    Validate data types against expected schema.

    Parameters:
    -----------
    expected_types : dict
        {'column_name': 'expected_type'}
        Types: 'numeric', 'categorical', 'datetime', 'boolean'
    """
    issues = []

    for col, expected in expected_types.items():
        if col not in data.columns:
            issues.append({'column': col, 'issue': 'Column missing'})
            continue

        actual = data[col].dtype

        if expected == 'numeric':
            if not np.issubdtype(actual, np.number):
                # Try to convert
                try:
                    pd.to_numeric(data[col], errors='raise')
                except:
                    issues.append({
                        'column': col,
                        'issue': f'Expected numeric, found {actual}',
                        'non_numeric_values': data[col][pd.to_numeric(data[col], errors='coerce').isna()].unique()[:10]
                    })

        elif expected == 'categorical':
            n_unique = data[col].nunique()
            if n_unique > len(data) * 0.5:
                issues.append({
                    'column': col,
                    'issue': f'High cardinality ({n_unique} unique values) for categorical'
                })

        elif expected == 'datetime':
            if not np.issubdtype(actual, np.datetime64):
                try:
                    pd.to_datetime(data[col], errors='raise')
                except:
                    issues.append({
                        'column': col,
                        'issue': f'Cannot parse as datetime'
                    })

    return issues
```

### Range and Constraint Validation

```python
def validate_constraints(data, constraints):
    """
    Validate data against domain constraints.

    Parameters:
    -----------
    constraints : dict
        {
            'column_name': {
                'min': value,
                'max': value,
                'allowed_values': list,
                'pattern': regex_string,
                'unique': bool
            }
        }
    """
    violations = []

    for col, rules in constraints.items():
        if col not in data.columns:
            continue

        col_data = data[col].dropna()

        if 'min' in rules:
            below_min = col_data < rules['min']
            if below_min.any():
                violations.append({
                    'column': col,
                    'rule': 'min',
                    'expected': rules['min'],
                    'n_violations': below_min.sum(),
                    'examples': col_data[below_min].head().tolist()
                })

        if 'max' in rules:
            above_max = col_data > rules['max']
            if above_max.any():
                violations.append({
                    'column': col,
                    'rule': 'max',
                    'expected': rules['max'],
                    'n_violations': above_max.sum(),
                    'examples': col_data[above_max].head().tolist()
                })

        if 'allowed_values' in rules:
            invalid = ~col_data.isin(rules['allowed_values'])
            if invalid.any():
                violations.append({
                    'column': col,
                    'rule': 'allowed_values',
                    'expected': rules['allowed_values'],
                    'n_violations': invalid.sum(),
                    'invalid_values': col_data[invalid].unique().tolist()
                })

        if 'unique' in rules and rules['unique']:
            duplicated = col_data.duplicated()
            if duplicated.any():
                violations.append({
                    'column': col,
                    'rule': 'unique',
                    'n_violations': duplicated.sum(),
                    'duplicated_values': col_data[duplicated].unique()[:10].tolist()
                })

    return violations
```

### Cross-Variable Consistency

```python
def check_cross_variable_consistency(data, rules):
    """
    Check logical consistency across variables.

    Parameters:
    -----------
    rules : list of dict
        [
            {
                'name': 'rule_name',
                'condition': lambda df: boolean_series,
                'description': 'Human-readable description'
            }
        ]
    """
    results = []

    for rule in rules:
        violations = rule['condition'](data)
        if violations.any():
            results.append({
                'rule': rule['name'],
                'description': rule['description'],
                'n_violations': violations.sum(),
                'pct_violations': (violations.sum() / len(data)) * 100,
                'example_indices': data.index[violations][:10].tolist()
            })

    return results

# Example usage:
consistency_rules = [
    {
        'name': 'age_education',
        'condition': lambda df: (df['age'] < 18) & (df['education'] == 'college_degree'),
        'description': 'Age < 18 with college degree'
    },
    {
        'name': 'income_employment',
        'condition': lambda df: (df['employment_status'] == 'unemployed') & (df['wage_income'] > 0),
        'description': 'Unemployed with positive wage income'
    },
    {
        'name': 'date_ordering',
        'condition': lambda df: df['end_date'] < df['start_date'],
        'description': 'End date before start date'
    }
]
```

## Causal Inference Implications

### Missing Data and Identification

| Mechanism | Effect on Causal Estimates | Recommended Approach |
|-----------|---------------------------|---------------------|
| MCAR | Unbiased but inefficient | Complete case or MI |
| MAR | Biased without correction | MI, IPW, or full information ML |
| MNAR | Biased, hard to correct | Selection models, sensitivity analysis |

### Data Quality Checklist for Causal Analysis

1. **Treatment variable**: No missing values, correct coding
2. **Outcome variable**: Missing pattern related to treatment?
3. **Covariates**: Missing data correlated with treatment assignment?
4. **Time variables**: Correct ordering, no gaps in panel
5. **ID variables**: Unique identifiers, no duplicates
6. **Sample restrictions**: Documented and justified
