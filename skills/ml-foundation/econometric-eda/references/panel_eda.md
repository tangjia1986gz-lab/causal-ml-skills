# Panel Data EDA for Econometric Research

## Overview

Panel data (longitudinal data) requires specialized EDA techniques to understand both cross-sectional and time-series variation. This is essential for choosing appropriate estimators and identifying potential threats to causal inference.

## Panel Structure Diagnostics

### Basic Panel Summary

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def panel_summary(data, entity_var, time_var):
    """
    Summarize basic panel structure.

    Parameters:
    -----------
    data : DataFrame
    entity_var : str
        Entity identifier (e.g., individual, firm, country)
    time_var : str
        Time period identifier
    """
    n_entities = data[entity_var].nunique()
    n_periods = data[time_var].nunique()
    n_obs = len(data)

    # Check for balanced panel
    obs_per_entity = data.groupby(entity_var).size()
    is_balanced = obs_per_entity.nunique() == 1

    # Time span
    time_min = data[time_var].min()
    time_max = data[time_var].max()

    # Gaps in time series
    expected_obs = n_entities * n_periods
    missing_obs = expected_obs - n_obs

    return {
        'n_entities': n_entities,
        'n_periods': n_periods,
        'n_observations': n_obs,
        'is_balanced': is_balanced,
        'time_span': (time_min, time_max),
        'obs_per_entity': {
            'min': obs_per_entity.min(),
            'max': obs_per_entity.max(),
            'mean': obs_per_entity.mean(),
            'median': obs_per_entity.median()
        },
        'missing_obs': missing_obs,
        'completeness': n_obs / expected_obs if expected_obs > 0 else 0
    }
```

### Panel Balance Analysis

```python
def analyze_panel_balance(data, entity_var, time_var, output_dir=None):
    """
    Detailed analysis of panel balance.
    """
    # Observations per entity
    obs_per_entity = data.groupby(entity_var).size().reset_index(name='n_obs')

    # Observations per time period
    obs_per_period = data.groupby(time_var).size().reset_index(name='n_obs')

    # First and last observation per entity
    entry_exit = data.groupby(entity_var)[time_var].agg(['min', 'max']).reset_index()
    entry_exit.columns = [entity_var, 'entry_period', 'exit_period']
    entry_exit['duration'] = entry_exit['exit_period'] - entry_exit['entry_period'] + 1

    # Identify gaps within entities
    def find_gaps(group):
        times = sorted(group[time_var].unique())
        if len(times) < 2:
            return []
        gaps = []
        for i in range(len(times) - 1):
            if times[i+1] - times[i] > 1:
                gaps.append((times[i], times[i+1]))
        return gaps

    gaps_by_entity = data.groupby(entity_var).apply(find_gaps)
    entities_with_gaps = gaps_by_entity[gaps_by_entity.apply(len) > 0]

    if output_dir:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Distribution of observations per entity
        axes[0, 0].hist(obs_per_entity['n_obs'], bins=30, edgecolor='black')
        axes[0, 0].axvline(obs_per_entity['n_obs'].mean(), color='r', linestyle='--',
                          label=f'Mean: {obs_per_entity["n_obs"].mean():.1f}')
        axes[0, 0].set_xlabel('Number of Observations')
        axes[0, 0].set_ylabel('Number of Entities')
        axes[0, 0].set_title('Observations per Entity')
        axes[0, 0].legend()

        # Observations per time period
        axes[0, 1].bar(obs_per_period[time_var], obs_per_period['n_obs'])
        axes[0, 1].set_xlabel('Time Period')
        axes[0, 1].set_ylabel('Number of Entities')
        axes[0, 1].set_title('Entities per Time Period')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Entry/Exit patterns
        entry_counts = entry_exit.groupby('entry_period').size()
        exit_counts = entry_exit.groupby('exit_period').size()
        axes[1, 0].bar(entry_counts.index - 0.2, entry_counts.values, 0.4, label='Entry')
        axes[1, 0].bar(exit_counts.index + 0.2, exit_counts.values, 0.4, label='Exit')
        axes[1, 0].set_xlabel('Time Period')
        axes[1, 0].set_ylabel('Number of Entities')
        axes[1, 0].set_title('Panel Entry and Exit')
        axes[1, 0].legend()

        # Duration distribution
        axes[1, 1].hist(entry_exit['duration'], bins=30, edgecolor='black')
        axes[1, 1].set_xlabel('Duration (periods)')
        axes[1, 1].set_ylabel('Number of Entities')
        axes[1, 1].set_title('Entity Duration in Panel')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/panel_balance.png', dpi=150, bbox_inches='tight')
        plt.close()

    return {
        'obs_per_entity': obs_per_entity,
        'obs_per_period': obs_per_period,
        'entry_exit': entry_exit,
        'n_entities_with_gaps': len(entities_with_gaps),
        'entities_with_gaps': entities_with_gaps
    }
```

## Within and Between Variation

### Variance Decomposition

```python
def variance_decomposition(data, entity_var, time_var, variables):
    """
    Decompose variance into between-entity and within-entity components.

    For each variable:
    - Total variance = Between variance + Within variance
    - Between: variance of entity means
    - Within: variance around entity means (fixed effects variation)
    """
    results = []

    for var in variables:
        if data[var].dtype not in ['float64', 'int64']:
            continue

        # Overall statistics
        total_mean = data[var].mean()
        total_var = data[var].var()
        total_std = data[var].std()

        # Entity means
        entity_means = data.groupby(entity_var)[var].transform('mean')

        # Between variance (variance of entity means)
        between_var = data.groupby(entity_var)[var].mean().var()
        between_std = np.sqrt(between_var)

        # Within variance (variance of deviations from entity mean)
        within_deviations = data[var] - entity_means
        within_var = within_deviations.var()
        within_std = np.sqrt(within_var)

        # Proportions
        total_var_check = between_var + within_var  # May not be exact due to degrees of freedom

        results.append({
            'variable': var,
            'mean': total_mean,
            'total_std': total_std,
            'between_std': between_std,
            'within_std': within_std,
            'between_share': between_var / total_var if total_var > 0 else 0,
            'within_share': within_var / total_var if total_var > 0 else 0,
            'min': data[var].min(),
            'max': data[var].max()
        })

    return pd.DataFrame(results)
```

### Visualization of Within/Between Variation

```python
def plot_within_between_variation(data, entity_var, time_var, variable,
                                   n_entities=10, output_path=None):
    """
    Visualize within and between variation for a variable.
    Shows trajectories for sample of entities.
    """
    # Select random entities if too many
    entities = data[entity_var].unique()
    if len(entities) > n_entities:
        np.random.seed(42)
        selected_entities = np.random.choice(entities, n_entities, replace=False)
    else:
        selected_entities = entities

    subset = data[data[entity_var].isin(selected_entities)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Spaghetti plot (individual trajectories)
    for entity in selected_entities:
        entity_data = subset[subset[entity_var] == entity].sort_values(time_var)
        axes[0].plot(entity_data[time_var], entity_data[variable], 'o-', alpha=0.6)

    # Overall mean by time
    time_means = data.groupby(time_var)[variable].mean()
    axes[0].plot(time_means.index, time_means.values, 'k-', linewidth=3, label='Overall mean')
    axes[0].set_xlabel('Time Period')
    axes[0].set_ylabel(variable)
    axes[0].set_title('Individual Trajectories (Within Variation)')
    axes[0].legend()

    # Panel B: Entity means distribution
    entity_means = data.groupby(entity_var)[variable].mean()
    axes[1].hist(entity_means, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(entity_means.mean(), color='r', linestyle='--',
                   label=f'Grand mean: {entity_means.mean():.2f}')
    axes[1].set_xlabel(f'Entity Mean of {variable}')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Entity Means (Between Variation)')
    axes[1].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig
```

## Panel Attrition Analysis

### Attrition Patterns

```python
def analyze_attrition(data, entity_var, time_var, variables=None):
    """
    Analyze panel attrition patterns.
    """
    # Get all time periods
    all_periods = sorted(data[time_var].unique())
    first_period = all_periods[0]
    last_period = all_periods[-1]

    # Initial sample (entities present in first period)
    initial_entities = set(data[data[time_var] == first_period][entity_var])

    # Track retention
    retention = []
    for period in all_periods:
        current_entities = set(data[data[time_var] == period][entity_var])
        retained = initial_entities.intersection(current_entities)
        retention.append({
            'period': period,
            'n_initial': len(initial_entities),
            'n_retained': len(retained),
            'retention_rate': len(retained) / len(initial_entities) if len(initial_entities) > 0 else 0,
            'n_current': len(current_entities)
        })

    retention_df = pd.DataFrame(retention)

    # Attrition by period
    attrition_by_period = []
    prev_entities = initial_entities
    for i, period in enumerate(all_periods[1:], 1):
        current_entities = set(data[data[time_var] == period][entity_var])
        left = prev_entities - current_entities
        entered = current_entities - prev_entities
        attrition_by_period.append({
            'period': period,
            'n_left': len(left),
            'n_entered': len(entered),
            'net_change': len(entered) - len(left)
        })
        prev_entities = current_entities

    attrition_df = pd.DataFrame(attrition_by_period)

    return {
        'retention': retention_df,
        'attrition': attrition_df,
        'initial_sample_size': len(initial_entities),
        'final_sample_size': retention_df.iloc[-1]['n_retained'],
        'overall_attrition_rate': 1 - retention_df.iloc[-1]['retention_rate']
    }
```

### Attrition Bias Test

```python
def test_attrition_bias(data, entity_var, time_var, variables, reference_period=None):
    """
    Test whether attrition is related to baseline characteristics.
    Compare those who stay vs. those who leave.
    """
    all_periods = sorted(data[time_var].unique())

    if reference_period is None:
        reference_period = all_periods[0]

    last_period = all_periods[-1]

    # Baseline data
    baseline = data[data[time_var] == reference_period].copy()

    # Identify stayers vs. leavers
    entities_last = set(data[data[time_var] == last_period][entity_var])
    baseline['stayer'] = baseline[entity_var].isin(entities_last).astype(int)

    # Compare characteristics
    results = []
    for var in variables:
        if var not in baseline.columns:
            continue
        if baseline[var].dtype not in ['float64', 'int64']:
            continue

        stayers = baseline.loc[baseline['stayer'] == 1, var].dropna()
        leavers = baseline.loc[baseline['stayer'] == 0, var].dropna()

        if len(stayers) == 0 or len(leavers) == 0:
            continue

        # T-test
        t_stat, p_value = stats.ttest_ind(stayers, leavers)

        # Standardized difference
        pooled_std = np.sqrt((stayers.var() + leavers.var()) / 2)
        std_diff = (stayers.mean() - leavers.mean()) / pooled_std if pooled_std > 0 else 0

        results.append({
            'variable': var,
            'mean_stayers': stayers.mean(),
            'mean_leavers': leavers.mean(),
            'std_diff': std_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'n_stayers': len(stayers),
            'n_leavers': len(leavers),
            'significant_diff': p_value < 0.05
        })

    results_df = pd.DataFrame(results)

    # Overall attrition assessment
    n_significant = results_df['significant_diff'].sum()
    attrition_concern = n_significant > len(variables) * 0.1  # More than 10% significant

    return {
        'comparison': results_df,
        'n_stayers': baseline['stayer'].sum(),
        'n_leavers': len(baseline) - baseline['stayer'].sum(),
        'attrition_rate': 1 - baseline['stayer'].mean(),
        'n_significant_diffs': n_significant,
        'attrition_concern': attrition_concern
    }
```

### Visualize Attrition

```python
def plot_attrition(data, entity_var, time_var, output_path=None):
    """
    Visualize panel attrition patterns.
    """
    attrition_results = analyze_attrition(data, entity_var, time_var)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel A: Retention curve
    retention = attrition_results['retention']
    axes[0].plot(retention['period'], retention['retention_rate'], 'o-', linewidth=2)
    axes[0].fill_between(retention['period'], retention['retention_rate'], alpha=0.3)
    axes[0].set_xlabel('Time Period')
    axes[0].set_ylabel('Retention Rate')
    axes[0].set_title('Cohort Retention Rate')
    axes[0].set_ylim(0, 1.05)

    # Panel B: Sample size over time
    axes[1].plot(retention['period'], retention['n_current'], 'o-', linewidth=2, label='Current')
    axes[1].plot(retention['period'], retention['n_retained'], 's--', linewidth=2, label='Original cohort')
    axes[1].set_xlabel('Time Period')
    axes[1].set_ylabel('Number of Entities')
    axes[1].set_title('Sample Size Over Time')
    axes[1].legend()

    # Panel C: Net changes
    attrition = attrition_results['attrition']
    x = np.arange(len(attrition))
    width = 0.35
    axes[2].bar(x - width/2, attrition['n_entered'], width, label='Entered', color='green', alpha=0.7)
    axes[2].bar(x + width/2, -attrition['n_left'], width, label='Left', color='red', alpha=0.7)
    axes[2].axhline(0, color='black', linewidth=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(attrition['period'])
    axes[2].set_xlabel('Time Period')
    axes[2].set_ylabel('Number of Entities')
    axes[2].set_title('Entry and Exit by Period')
    axes[2].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig
```

## Time Series Properties

### Stationarity Tests by Entity

```python
from statsmodels.tsa.stattools import adfuller

def test_stationarity_by_entity(data, entity_var, time_var, variable, min_periods=10):
    """
    Test stationarity of a variable for each entity using ADF test.
    """
    results = []

    for entity in data[entity_var].unique():
        entity_data = data[data[entity_var] == entity].sort_values(time_var)[variable].dropna()

        if len(entity_data) < min_periods:
            continue

        try:
            adf_result = adfuller(entity_data, autolag='AIC')
            results.append({
                'entity': entity,
                'n_obs': len(entity_data),
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'lags_used': adf_result[2],
                'stationary': adf_result[1] < 0.05
            })
        except Exception as e:
            results.append({
                'entity': entity,
                'n_obs': len(entity_data),
                'error': str(e)
            })

    results_df = pd.DataFrame(results)

    return {
        'entity_results': results_df,
        'n_stationary': results_df['stationary'].sum() if 'stationary' in results_df.columns else 0,
        'n_tested': len(results_df),
        'pct_stationary': results_df['stationary'].mean() if 'stationary' in results_df.columns else 0
    }
```

### Autocorrelation Analysis

```python
from statsmodels.tsa.stattools import acf, pacf

def panel_autocorrelation(data, entity_var, time_var, variable, max_lag=10):
    """
    Analyze autocorrelation structure in panel data.
    """
    # Pooled autocorrelation (after demeaning by entity)
    entity_means = data.groupby(entity_var)[variable].transform('mean')
    demeaned = data[variable] - entity_means

    # Calculate autocorrelations by entity, then average
    acf_by_entity = []

    for entity in data[entity_var].unique():
        entity_data = data[data[entity_var] == entity].sort_values(time_var)[variable].dropna()
        if len(entity_data) > max_lag + 1:
            try:
                acf_values = acf(entity_data, nlags=max_lag, fft=False)
                acf_by_entity.append(acf_values)
            except:
                continue

    if len(acf_by_entity) > 0:
        avg_acf = np.mean(acf_by_entity, axis=0)
        std_acf = np.std(acf_by_entity, axis=0)
    else:
        avg_acf = None
        std_acf = None

    return {
        'average_acf': avg_acf,
        'std_acf': std_acf,
        'n_entities_used': len(acf_by_entity),
        'max_lag': max_lag
    }
```

## Panel-Specific Missing Data

```python
def panel_missing_patterns(data, entity_var, time_var, variables):
    """
    Analyze missing data patterns specific to panel structure.
    """
    results = {
        'by_variable': {},
        'by_period': {},
        'by_entity': {}
    }

    for var in variables:
        # Missing by time period
        missing_by_period = data.groupby(time_var)[var].apply(lambda x: x.isna().mean())

        # Missing by entity
        missing_by_entity = data.groupby(entity_var)[var].apply(lambda x: x.isna().mean())

        # Trend in missingness over time
        periods = sorted(data[time_var].unique())
        missing_trend = [data[data[time_var] == t][var].isna().mean() for t in periods]

        results['by_variable'][var] = {
            'overall_missing_rate': data[var].isna().mean(),
            'missing_by_period': missing_by_period,
            'missing_by_entity': missing_by_entity,
            'missing_trend': missing_trend,
            'n_entities_all_missing': (missing_by_entity == 1).sum(),
            'n_periods_all_missing': (missing_by_period == 1).sum()
        }

    # Overall by period
    results['by_period'] = data.groupby(time_var)[variables].apply(
        lambda x: x.isna().mean()
    )

    # Overall by entity
    results['by_entity'] = data.groupby(entity_var)[variables].apply(
        lambda x: x.isna().mean()
    )

    return results
```

## Causal Inference Considerations for Panel Data

### Fixed Effects Suitability

```python
def assess_fe_suitability(data, entity_var, time_var, treatment_var, outcome_var, covariates):
    """
    Assess whether fixed effects estimation is appropriate.
    """
    assessment = {}

    # 1. Within variation in treatment
    var_decomp = variance_decomposition(data, entity_var, time_var, [treatment_var])
    treatment_within_share = var_decomp.loc[var_decomp['variable'] == treatment_var, 'within_share'].values[0]

    assessment['treatment_within_variation'] = treatment_within_share
    assessment['sufficient_within_variation'] = treatment_within_share > 0.1

    # 2. Number of treatment switches
    def count_switches(group):
        group = group.sort_values(time_var)
        return (group[treatment_var].diff().abs() > 0).sum()

    switches_by_entity = data.groupby(entity_var).apply(count_switches)
    assessment['n_entities_with_switches'] = (switches_by_entity > 0).sum()
    assessment['pct_entities_with_switches'] = (switches_by_entity > 0).mean()

    # 3. Parallel trends plausibility (pre-treatment)
    # Would need to define pre-treatment period

    # 4. Time-invariant vs time-varying covariates
    covariate_variation = variance_decomposition(data, entity_var, time_var, covariates)
    assessment['time_invariant_covariates'] = covariate_variation[
        covariate_variation['within_share'] < 0.01
    ]['variable'].tolist()

    return assessment
```

### Summary Table

| Check | Purpose | FE Concern If... |
|-------|---------|------------------|
| Within variation in treatment | Treatment must vary within entities | < 10% within variation |
| Treatment switches | Entities must change treatment status | Few entities switch |
| Attrition | Non-random dropout biases estimates | Attrition correlated with treatment/outcome |
| Time-invariant confounders | FE absorbs these | Fine, this is the point |
| Time-varying confounders | FE doesn't address these | Must include as covariates |
| Parallel trends | Key identifying assumption | Pre-treatment trends differ |
