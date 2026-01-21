# Reporting Standards for Causal Forest Analysis

## Overview

This document establishes standards for reporting causal forest analyses, including variable importance presentation, CATE visualization, policy tree reporting, and complete analysis reports.

---

## 1. Essential Reporting Elements

### Minimum Reporting Requirements

Every causal forest analysis report should include:

1. **Study Design**
   - Data source and sample size
   - Treatment definition
   - Outcome definition
   - Effect modifier selection rationale

2. **Identification Strategy**
   - Assumptions invoked (unconfoundedness, SUTVA, overlap)
   - Supporting evidence for assumptions
   - Limitations and potential violations

3. **Model Specification**
   - Algorithm used (grf, econml, etc.)
   - Key hyperparameters
   - Honesty specification

4. **Results**
   - Average Treatment Effect (ATE) with confidence interval
   - CATE distribution summary
   - Variable importance for heterogeneity
   - Heterogeneity test results

5. **Robustness**
   - Calibration test results
   - Overlap diagnostics
   - Sensitivity analyses

---

## 2. Variable Importance Reporting

### Table Format

```markdown
| Rank | Variable | Importance | Cumulative |
|------|----------|------------|------------|
| 1 | customer_age | 0.342 | 0.342 |
| 2 | tenure_months | 0.218 | 0.560 |
| 3 | past_purchases | 0.156 | 0.716 |
| 4 | email_opens_30d | 0.124 | 0.840 |
| 5 | website_visits | 0.089 | 0.929 |
| 6 | segment | 0.071 | 1.000 |
```

### Visualization

```python
def plot_variable_importance_report(importance_df, save_path=None):
    """
    Create publication-quality variable importance plot.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    ax1 = axes[0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_df)))

    bars = ax1.barh(range(len(importance_df)),
                    importance_df['importance'],
                    color=colors)
    ax1.set_yticks(range(len(importance_df)))
    ax1.set_yticklabels(importance_df['variable'])
    ax1.invert_yaxis()
    ax1.set_xlabel('Variable Importance', fontsize=12)
    ax1.set_title('Drivers of Treatment Effect Heterogeneity', fontsize=14)

    # Add value labels
    for bar, val in zip(bars, importance_df['importance']):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)

    # Cumulative importance
    ax2 = axes[1]
    cumulative = np.cumsum(importance_df['importance'])
    ax2.plot(range(1, len(cumulative)+1), cumulative, 'o-',
             linewidth=2, markersize=8, color='steelblue')
    ax2.axhline(0.8, color='red', linestyle='--', alpha=0.7,
                label='80% threshold')
    ax2.set_xticks(range(1, len(cumulative)+1))
    ax2.set_xticklabels(importance_df['variable'], rotation=45, ha='right')
    ax2.set_ylabel('Cumulative Importance', fontsize=12)
    ax2.set_title('Cumulative Variable Importance', fontsize=14)
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
```

### Interpretation Guidelines

```markdown
**Variable Importance Interpretation Guidelines:**

1. **High importance (> 0.2)**: Strong driver of heterogeneity
   - Treatment effect varies substantially with this variable
   - Consider for subgroup analysis and policy targeting

2. **Moderate importance (0.1-0.2)**: Moderate driver
   - Some contribution to heterogeneity
   - May interact with high-importance variables

3. **Low importance (< 0.1)**: Weak driver
   - Limited contribution to heterogeneity
   - Treatment effect relatively constant across values

**Caveats:**
- Correlated variables share importance (may understate individual contribution)
- Importance does not indicate direction of effect
- Use BLP for quantitative relationships
```

---

## 3. CATE Visualization Standards

### Distribution Plot

```python
def create_cate_distribution_report(cate_results, output_path=None):
    """
    Create standardized CATE distribution visualization.
    """
    tau = cate_results.estimates
    ci_lower = cate_results.ci_lower
    ci_upper = cate_results.ci_upper

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Histogram with KDE
    ax1 = axes[0, 0]
    ax1.hist(tau, bins=50, density=True, alpha=0.6, color='steelblue',
             edgecolor='white', label='CATE distribution')

    # KDE overlay
    from scipy import stats
    kde = stats.gaussian_kde(tau)
    x_kde = np.linspace(tau.min(), tau.max(), 200)
    ax1.plot(x_kde, kde(x_kde), 'r-', linewidth=2, label='KDE')

    ax1.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero effect')
    ax1.axvline(np.mean(tau), color='green', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(tau):.3f}')
    ax1.set_xlabel('Conditional Average Treatment Effect (CATE)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('CATE Distribution', fontsize=14)
    ax1.legend()

    # 2. Box plot with individual points
    ax2 = axes[0, 1]
    parts = ax2.violinplot([tau], positions=[1], showmeans=True,
                            showmedians=True, widths=0.8)
    parts['bodies'][0].set_facecolor('steelblue')
    parts['bodies'][0].set_alpha(0.6)

    # Add jittered points
    jitter = np.random.uniform(-0.15, 0.15, len(tau))
    ax2.scatter(1 + jitter, tau, alpha=0.1, s=5, color='gray')

    ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xticks([1])
    ax2.set_xticklabels(['CATE'])
    ax2.set_ylabel('Treatment Effect', fontsize=12)
    ax2.set_title('CATE Violin Plot', fontsize=14)

    # 3. Sorted CATEs with CI
    ax3 = axes[1, 0]
    sorted_idx = np.argsort(tau)
    n = len(tau)
    x = np.arange(n)

    # Subsample for clarity if large n
    if n > 1000:
        step = n // 500
        x_plot = x[::step]
        tau_plot = tau[sorted_idx][::step]
        ci_l_plot = ci_lower[sorted_idx][::step]
        ci_u_plot = ci_upper[sorted_idx][::step]
    else:
        x_plot = x
        tau_plot = tau[sorted_idx]
        ci_l_plot = ci_lower[sorted_idx]
        ci_u_plot = ci_upper[sorted_idx]

    ax3.fill_between(x_plot, ci_l_plot, ci_u_plot, alpha=0.3, color='steelblue')
    ax3.plot(x_plot, tau_plot, color='steelblue', linewidth=1)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)

    ax3.set_xlabel('Observation (sorted by CATE)', fontsize=12)
    ax3.set_ylabel('Treatment Effect', fontsize=12)
    ax3.set_title('Sorted CATEs with 95% Confidence Intervals', fontsize=14)

    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    ┌─────────────────────────────────────────┐
    │       CATE Summary Statistics           │
    ├─────────────────────────────────────────┤
    │  Mean (ATE):          {np.mean(tau):>12.4f}   │
    │  Standard Deviation:  {np.std(tau):>12.4f}   │
    │  Median:              {np.median(tau):>12.4f}   │
    │  Min:                 {np.min(tau):>12.4f}   │
    │  Max:                 {np.max(tau):>12.4f}   │
    │  IQR:     [{np.percentile(tau, 25):>7.3f}, {np.percentile(tau, 75):>7.3f}]   │
    ├─────────────────────────────────────────┤
    │  % Positive:          {np.mean(tau > 0)*100:>10.1f}%  │
    │  % Negative:          {np.mean(tau < 0)*100:>10.1f}%  │
    │  % Significantly +ve: {np.mean(ci_lower > 0)*100:>10.1f}%  │
    │  % Significantly -ve: {np.mean(ci_upper < 0)*100:>10.1f}%  │
    └─────────────────────────────────────────┘
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig
```

### Subgroup Comparison Plot

```python
def plot_cate_by_subgroup(cate_results, group_var, group_labels=None,
                          save_path=None):
    """
    Compare CATE distributions across subgroups.
    """
    tau = cate_results.estimates
    groups = np.array(group_var)
    unique_groups = np.unique(groups)

    if group_labels is None:
        group_labels = [str(g) for g in unique_groups]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot comparison
    ax1 = axes[0]
    data_by_group = [tau[groups == g] for g in unique_groups]
    bp = ax1.boxplot(data_by_group, labels=group_labels, patch_artist=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_groups)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.axhline(np.mean(tau), color='green', linestyle='-', linewidth=1,
                label=f'Overall mean: {np.mean(tau):.3f}')

    ax1.set_ylabel('CATE', fontsize=12)
    ax1.set_title('CATE Distribution by Subgroup', fontsize=14)
    ax1.legend()

    # Mean comparison with CI
    ax2 = axes[1]
    group_means = [np.mean(tau[groups == g]) for g in unique_groups]
    group_se = [np.std(tau[groups == g]) / np.sqrt(np.sum(groups == g))
                for g in unique_groups]

    x_pos = np.arange(len(unique_groups))
    ax2.bar(x_pos, group_means, yerr=[1.96*se for se in group_se],
            capsize=5, color=colors, alpha=0.7, edgecolor='black')

    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.axhline(np.mean(tau), color='green', linestyle='-', linewidth=1)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(group_labels)
    ax2.set_ylabel('Mean CATE', fontsize=12)
    ax2.set_title('Mean CATE by Subgroup (with 95% CI)', fontsize=14)

    # Add value labels
    for i, (mean, se) in enumerate(zip(group_means, group_se)):
        ax2.text(i, mean + 1.96*se + 0.02*np.ptp(tau),
                f'{mean:.3f}', ha='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
```

---

## 4. Policy Tree Reporting

### Tree Visualization

```python
def visualize_policy_tree(policy_tree, feature_names, class_names=['Control', 'Treat'],
                          save_path=None):
    """
    Create publication-quality policy tree visualization.
    """
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(20, 12))

    plot_tree(policy_tree,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=10,
              ax=ax,
              impurity=False)

    ax.set_title('Optimal Treatment Policy Tree', fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def extract_policy_rules(policy_tree, feature_names, X):
    """
    Extract human-readable policy rules from tree.
    """
    from sklearn.tree import _tree

    tree_ = policy_tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def recurse(node, rule=""):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # Left child
            recurse(tree_.children_left[node],
                   f"{rule}{name} <= {threshold:.2f} AND ")

            # Right child
            recurse(tree_.children_right[node],
                   f"{rule}{name} > {threshold:.2f} AND ")
        else:
            # Leaf node
            value = tree_.value[node][0][0]
            n_samples = tree_.n_node_samples[node]

            # Clean up rule
            rule_clean = rule.rstrip(" AND ")

            rules.append({
                'rule': rule_clean if rule_clean else "All observations",
                'recommendation': 'TREAT' if value > 0 else 'CONTROL',
                'expected_benefit': value,
                'n_samples': n_samples
            })

    recurse(0)

    rules_df = pd.DataFrame(rules)
    rules_df = rules_df.sort_values('expected_benefit', ascending=False)

    return rules_df
```

### Policy Report Table

```markdown
## Policy Tree Rules

| Rule | Recommendation | Expected Benefit | N | % of Sample |
|------|----------------|------------------|---|-------------|
| age > 45 AND income > 80000 | TREAT | $125.50 | 1,234 | 24.7% |
| age > 45 AND income <= 80000 | TREAT | $45.20 | 987 | 19.7% |
| age <= 45 AND tenure > 24 | CONTROL | -$12.30 | 1,456 | 29.1% |
| age <= 45 AND tenure <= 24 | CONTROL | -$78.60 | 1,323 | 26.5% |

### Policy Summary

- **Optimal treatment rate**: 44.4% (vs. 50% random assignment)
- **Policy value**: $52.30 per person
- **Improvement over treat-all**: +35.2%
- **Improvement over treat-none**: +89.7%
```

---

## 5. Complete Analysis Report Template

```python
def generate_analysis_report(results, output_path):
    """
    Generate complete markdown analysis report.
    """
    report = f"""# Causal Forest Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This analysis estimates heterogeneous treatment effects using causal forests,
identifying which individuals benefit most from treatment and providing
optimal treatment allocation recommendations.

**Key Findings:**
- Average Treatment Effect (ATE): {results['ate']:.4f} (SE: {results['ate_se']:.4f})
- Significant heterogeneity detected: {'Yes' if results['heterogeneity_significant'] else 'No'}
- Top driver of heterogeneity: {results['top_driver']}
- Optimal policy improves outcomes by: {results['policy_improvement']:.1%}

---

## 1. Study Design

### Data Description
- **Sample size**: {results['n_total']:,}
- **Treatment variable**: {results['treatment_name']}
- **Outcome variable**: {results['outcome_name']}
- **Effect modifiers**: {', '.join(results['effect_modifiers'])}

### Identification Assumptions
1. **Unconfoundedness**: Treatment assignment is independent of potential outcomes
   conditional on observed covariates
2. **Positivity**: All covariate profiles have positive probability of each treatment
3. **SUTVA**: No interference between units; single version of treatment

---

## 2. Model Specification

### Causal Forest Configuration
| Parameter | Value |
|-----------|-------|
| Backend | {results['backend']} |
| Number of trees | {results['n_trees']:,} |
| Honesty | {results['honesty']} |
| Honesty fraction | {results['honesty_fraction']} |
| Minimum node size | {results['min_node_size']} |
| Sample fraction | {results['sample_fraction']} |

---

## 3. Diagnostic Results

### Overlap Assessment
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Propensity min | {results['ps_min']:.4f} | > 0.05 | {'PASS' if results['ps_min'] > 0.05 else 'FAIL'} |
| Propensity max | {results['ps_max']:.4f} | < 0.95 | {'PASS' if results['ps_max'] < 0.95 else 'FAIL'} |
| Common support | {results['common_support']:.1%} | > 90% | {'PASS' if results['common_support'] > 0.9 else 'FAIL'} |

### Calibration Test
| Component | Estimate | SE | Interpretation |
|-----------|----------|-------|----------------|
| Mean forest prediction | {results['cal_mean']:.3f} | {results['cal_mean_se']:.3f} | {'Well-calibrated' if 0.8 < results['cal_mean'] < 1.2 else 'Miscalibrated'} |
| Differential prediction | {results['cal_diff']:.3f} | {results['cal_diff_se']:.3f} | {'Heterogeneity captured' if results['cal_diff'] > 0.5 else 'Limited heterogeneity'} |

---

## 4. Treatment Effect Results

### Average Treatment Effect
$$\\hat{{\\tau}}_{{ATE}} = {results['ate']:.4f} \\quad (95\\% \\text{{ CI: }} [{results['ate_ci_lower']:.4f}, {results['ate_ci_upper']:.4f}])$$

### CATE Distribution

| Statistic | Value |
|-----------|-------|
| Mean | {results['cate_mean']:.4f} |
| Std Dev | {results['cate_std']:.4f} |
| Min | {results['cate_min']:.4f} |
| Max | {results['cate_max']:.4f} |
| Median | {results['cate_median']:.4f} |
| IQR | [{results['cate_q25']:.4f}, {results['cate_q75']:.4f}] |

### Proportion Analysis
| Category | Percentage |
|----------|------------|
| Positive effects | {results['pct_positive']:.1%} |
| Negative effects | {results['pct_negative']:.1%} |
| Significantly positive | {results['pct_sig_positive']:.1%} |
| Significantly negative | {results['pct_sig_negative']:.1%} |

---

## 5. Heterogeneity Analysis

### Omnibus Heterogeneity Test
- **Test statistic**: {results['het_stat']:.2f}
- **p-value**: {results['het_pvalue']:.4f}
- **Conclusion**: {'Significant heterogeneity detected' if results['het_pvalue'] < 0.05 else 'No significant heterogeneity'}

### Variable Importance

{results['importance_table']}

### Best Linear Projection

{results['blp_table']}

**Interpretation**: {results['blp_interpretation']}

---

## 6. Policy Recommendations

### Optimal Treatment Policy
- **Method**: {results['policy_method']}
- **Treatment rate**: {results['policy_treatment_rate']:.1%}
- **Policy value**: {results['policy_value']:.4f}
- **Improvement over treat-all**: {results['policy_improvement']:.1%}

### Policy Rules
{results['policy_rules']}

### Target Population
{results['target_population_description']}

---

## 7. Limitations

1. **Unverifiable assumptions**: Unconfoundedness cannot be directly tested
2. **External validity**: Results apply to study population; generalization requires caution
3. **Heterogeneity precision**: Individual CATE estimates have higher uncertainty than ATE
4. {results.get('additional_limitations', '')}

---

## 8. Appendix

### A. Covariate Balance
{results['balance_table']}

### B. Sensitivity Analysis
{results.get('sensitivity_results', 'Not performed')}

### C. Robustness Checks
{results.get('robustness_results', 'Not performed')}

---

## References

1. Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous
   Treatment Effects using Random Forests. *JASA*, 113(523), 1228-1242.

2. Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized Random Forests.
   *Annals of Statistics*, 47(2), 1148-1178.

---

*Report generated using Causal Forest Analysis Toolkit*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report
```

---

## 6. Checklist for Complete Reporting

### Pre-Analysis
- [ ] Clearly state research question
- [ ] Define treatment, outcome, effect modifiers
- [ ] Document data source and sample selection
- [ ] Specify identification assumptions

### Model Specification
- [ ] Report all hyperparameters
- [ ] Document software version (grf, econml)
- [ ] State whether honesty was used
- [ ] Report random seed for reproducibility

### Diagnostics
- [ ] Overlap assessment with visualization
- [ ] Covariate balance table
- [ ] Calibration test results
- [ ] Sample size adequacy discussion

### Results
- [ ] ATE with confidence interval
- [ ] CATE distribution visualization
- [ ] Variable importance (table and plot)
- [ ] BLP coefficients with interpretation
- [ ] Heterogeneity test results

### Policy (if applicable)
- [ ] Policy tree visualization
- [ ] Extracted rules in table format
- [ ] Policy value metrics
- [ ] Target population characterization

### Robustness
- [ ] Alternative specifications
- [ ] Sensitivity to hyperparameters
- [ ] Subsample analyses

### Limitations
- [ ] Unverifiable assumptions
- [ ] External validity concerns
- [ ] Uncertainty in CATE estimates
- [ ] Data limitations

---

## Summary

Proper reporting of causal forest analyses requires:

1. **Transparency**: Full disclosure of methods, assumptions, and limitations
2. **Visualization**: Clear plots of CATE distribution, importance, and policies
3. **Quantification**: Precise estimates with uncertainty measures
4. **Interpretation**: Plain-language explanation of findings
5. **Reproducibility**: Sufficient detail for replication

Follow these standards to ensure credible and actionable results.
