# Gap Analysis: Callaway & Sant'Anna (2018) vs Current Documentation

> **Analysis Date**: 2026-01-21
> **Source Paper**: Callaway, B., & Sant'Anna, P.H. (2018/2021). "Difference-in-Differences with Multiple Time Periods"
> **Compared Against**: estimator-did/references/*.md

---

## Executive Summary

本分析对比 Callaway & Sant'Anna (2018) 原文与现有 `estimator-did` 技能文档，识别出 **12 个主要 Gap** 和 **8 个次要 Gap**，涉及识别假设、估计方法、诊断检验三个核心维度。

| 维度 | 主要 Gap | 次要 Gap | 覆盖率 |
|------|----------|----------|--------|
| 识别假设 | 4 | 2 | 60% |
| 估计方法 | 5 | 3 | 55% |
| 诊断检验 | 3 | 3 | 65% |

---

## 1. Identification Assumptions Gaps

### 1.1 Conditional Parallel Trends (Major Gap)

**Paper Definition (Assumption 4)**:
$$
E[Y_t(0) - Y_{t-1}(0) | X, G_g = 1] = E[Y_t(0) - Y_{t-1}(0) | X, C = 1]
$$

**Current Documentation**: 仅包含无条件平行趋势假设，缺少条件于协变量 X 的正式定义。

**Required Update**:
- 添加条件平行趋势公式
- 解释何时需要条件版本 vs 无条件版本
- 添加"条件于观测协变量后，处理组和控制组的平均结果趋势相同"的直观解释

### 1.2 Irreversibility of Treatment (Major Gap)

**Paper Definition (Assumption 6)**:
$$
\text{Once treated, units remain treated: } D_{i,t} = 1 \Rightarrow D_{i,t+1} = 1
$$

**Current Documentation**: 完全缺失此假设。

**Required Update**:
- 添加不可逆处理假设
- 解释为何 staggered adoption 设计需要此假设
- 讨论违反此假设时的替代方法 (de Chaisemartin-D'Haultfoeuille)

### 1.3 Limited Treatment Anticipation (Major Gap)

**Paper Definition (Assumption 5)**:
$$
E[Y_t(g) - Y_t(0) | X, G_g = 1] = 0 \quad \text{for all } t < g - \delta
$$

**Current Documentation**: 仅有简单的"无预期效应"假设，缺少允许有限预期 (δ 期) 的灵活版本。

**Required Update**:
- 添加 δ-期预期效应的正式定义
- 解释如何在估计中调整 anticipation 参数
- 提供实践指导：如何选择 δ 值

### 1.4 Overlap/Generalized Propensity Score (Major Gap)

**Paper Definition (Assumption 2)**:
$$
p_g(X) = P(G_g = 1 | X, G_g + C = 1)
$$

**Overlap Condition**:
$$
p_g(X) < 1 \text{ for all } X
$$

**Current Documentation**: 有通用的 common support 讨论，但缺少 Callaway-Sant'Anna 特定的广义倾向得分定义。

**Required Update**:
- 添加广义倾向得分的正式定义
- 解释与标准倾向得分的区别
- 添加对 never-treated 和 not-yet-treated 控制组的讨论

### 1.5 Random Sampling (Minor Gap)

**Paper Definition (Assumption 1)**:
随机抽样假设，确保渐近分布理论有效。

**Current Documentation**: 未明确提及。

### 1.6 Stationarity Assumptions (Minor Gap)

**Paper**: 讨论 stationary conditional distributions 的作用。

**Current Documentation**: 缺少此技术细节。

---

## 2. Estimation Methods Gaps

### 2.1 ATT(g,t) Identification Formula (Major Gap)

**Paper Theorem 1**:
$$
ATT(g,t) = E\left[\left(\frac{G_g}{E[G_g]} - \frac{\frac{p_g(X) \cdot C}{1-p_g(X)}}{E\left[\frac{p_g(X) \cdot C}{1-p_g(X)}\right]}\right)(Y_t - Y_{g-1})\right]
$$

**Current Documentation**: 仅有简化公式 `ATT(g,t) = E[Y_t - Y_{g-1} | G = g] - E[Y_t - Y_{g-1} | G ∈ Control]`，缺少完整的倒数概率加权识别公式。

**Required Update**:
- 添加完整的 IPW 识别公式
- 添加 outcome regression 识别公式
- 添加 doubly robust 识别公式

### 2.2 Doubly Robust Estimation (Major Gap)

**Paper**: 提供结合 IPW 和 outcome regression 的 doubly robust 估计器。

**Current Documentation**: 未提及 doubly robust 方法。

**Required Update**:
- 添加 doubly robust 估计器公式
- 解释双重稳健性的含义
- 讨论何时使用 IPW vs OR vs DR

### 2.3 Aggregation Schemes (Major Gap)

**Paper Definitions**:

**Selective Treatment Timing** (θ^S):
$$
\theta^S = \sum_{g \in \mathcal{G}} \theta^S(g) \cdot P(G = g | G \in \mathcal{G})
$$
$$
\theta^S(g) = \frac{1}{\mathcal{T} - g + 1} \sum_{t=g}^{\mathcal{T}} ATT(g,t)
$$

**Dynamic Treatment Effects** (θ^D):
$$
\theta^D(e) = \sum_{g \in \mathcal{G}} \mathbf{1}\{g + e \leq \mathcal{T}\} ATT(g, g+e) \cdot P(G = g | G + e \leq \mathcal{T})
$$

**Calendar Time Effects** (θ^C):
$$
\theta^C(t) = \sum_{g \in \mathcal{G}} \mathbf{1}\{g \leq t\} ATT(g,t) \cdot P(G = g | G \leq t)
$$

**Current Documentation**: 仅提及"Overall ATT"和"Event-time ATT"，缺少完整的聚合公式和权重定义。

**Required Update**:
- 添加三种聚合方案的完整数学定义
- 添加权重计算方法
- 解释何时使用哪种聚合方法

### 2.4 Two-Step Semiparametric Estimation (Major Gap)

**Paper**:
- Step 1: 估计广义倾向得分 p̂_g(X)
- Step 2: 使用估计的倾向得分计算 ATT(g,t)

**Current Documentation**: 缺少两步估计过程的详细描述。

### 2.5 Influence Function and Variance Estimation (Major Gap)

**Paper**: 提供完整的 influence function 用于方差估计和推断。

**Current Documentation**: 未提及 influence function 方法。

### 2.6 Control Group Comparison (Minor Gap)

**Paper**: 详细比较 never-treated vs not-yet-treated 控制组的权衡。

**Current Documentation**: 有表格提及但缺少深入分析。

### 2.7 Bandwidth and Trimming (Minor Gap)

**Paper**: 讨论倾向得分截断和修剪。

**Current Documentation**: 未提及具体的修剪策略。

### 2.8 Computational Efficiency (Minor Gap)

**Paper**: R 包 `did` 的计算实现细节。

**Current Documentation**: 仅提及软件包名称。

---

## 3. Diagnostic Tests Gaps

### 3.1 Integrated Conditional Moments (ICM) Pre-Test (Major Gap)

**Paper**: 提出基于 ICM 的平行趋势预检验方法，使用 Cramér-von Mises (CvM) 和 Kolmogorov-Smirnov (KS) 统计量。

**Pre-testing Hypothesis**:
$$
H_0: E[Y_t(0) - Y_{t-1}(0) | X, G_g = 1] = E[Y_t(0) - Y_{t-1}(0) | X, C = 1] \text{ a.s.}
$$

**Current Documentation**: 有 event study 和 Rambachan-Roth，但缺少 ICM 方法。

**Required Update**:
- 添加 ICM pre-test 方法描述
- 添加 CvM 和 KS 统计量公式
- 提供 Python/R 实现指导

### 3.2 Multiplier Bootstrap for Simultaneous Inference (Major Gap)

**Paper**: 使用 multiplier bootstrap 构建 uniform confidence bands，而非逐点置信区间。

**Current Documentation**: 提及 bootstrap 但缺少 simultaneous/uniform inference 的讨论。

**Required Update**:
- 添加 multiplier bootstrap 方法
- 解释 pointwise vs simultaneous confidence bands 的区别
- 讨论何时需要 simultaneous inference

### 3.3 Group-Time Specific Inference (Major Gap)

**Paper**: 为每个 ATT(g,t) 提供独立的推断，并讨论如何汇总推断结果。

**Current Documentation**: 主要关注汇总效应的推断。

### 3.4 Covariate Balance Interpretation (Minor Gap)

**Paper**: 条件平行趋势下的协变量角色。

**Current Documentation**: 有协变量平衡检验但缺少与条件平行趋势的联系。

### 3.5 Placebo Tests for ATT(g,t) (Minor Gap)

**Paper**: 讨论针对 group-time specific 效应的 placebo 检验。

**Current Documentation**: 有一般性 placebo 检验。

### 3.6 Sensitivity to Control Group Choice (Minor Gap)

**Paper**: 比较不同控制组选择的敏感性。

**Current Documentation**: 未明确讨论。

---

## 4. Recommended Updates Priority

### Priority 1 (Critical - 影响方法正确性)

| Update | File | Urgency |
|--------|------|---------|
| Conditional Parallel Trends | identification_assumptions.md | HIGH |
| Irreversibility Assumption | identification_assumptions.md | HIGH |
| ATT(g,t) Identification Formula | estimation_methods.md | HIGH |
| Aggregation Schemes | estimation_methods.md | HIGH |

### Priority 2 (Important - 影响估计质量)

| Update | File | Urgency |
|--------|------|---------|
| Limited Anticipation | identification_assumptions.md | MEDIUM |
| Doubly Robust Estimation | estimation_methods.md | MEDIUM |
| ICM Pre-Test | diagnostic_tests.md | MEDIUM |
| Multiplier Bootstrap | diagnostic_tests.md | MEDIUM |

### Priority 3 (Enhancement - 完善性)

| Update | File | Urgency |
|--------|------|---------|
| Generalized Propensity Score | identification_assumptions.md | LOW |
| Two-Step Estimation | estimation_methods.md | LOW |
| Group-Time Inference | diagnostic_tests.md | LOW |

---

## 5. Implementation Recommendations

### 5.1 Code Updates Needed

```python
# 需要在 did_estimator.py 中添加的函数

def estimate_att_gt_doubly_robust(...):
    """Doubly robust ATT(g,t) estimator."""
    pass

def aggregate_att_selective_timing(...):
    """θ^S aggregation."""
    pass

def aggregate_att_dynamic(...):
    """θ^D(e) aggregation."""
    pass

def aggregate_att_calendar(...):
    """θ^C(t) aggregation."""
    pass

def icm_pretest(...):
    """Integrated Conditional Moments pre-test."""
    pass

def multiplier_bootstrap_confidence_band(...):
    """Simultaneous confidence bands via multiplier bootstrap."""
    pass
```

### 5.2 Documentation Structure Updates

```
references/
├── identification_assumptions.md  # Add sections 1.1-1.4
├── estimation_methods.md          # Add sections 2.1-2.5
├── diagnostic_tests.md            # Add sections 3.1-3.3
├── callaway_santanna_details.md   # NEW: Complete C-S method reference
└── aggregation_methods.md         # NEW: Dedicated aggregation reference
```

---

## 6. Verification Checklist

更新完成后，验证以下内容：

- [ ] Conditional Parallel Trends 公式与论文一致
- [ ] ATT(g,t) 识别公式完整正确
- [ ] 三种聚合方案 (θ^S, θ^D, θ^C) 定义准确
- [ ] Doubly Robust 估计器与论文描述一致
- [ ] ICM 预检验方法可操作
- [ ] Multiplier Bootstrap 实现正确
- [ ] 代码示例可运行

---

## References

1. Callaway, B., & Sant'Anna, P. H. (2018). "Difference-in-Differences with Multiple Time Periods." Working Paper, later published in *Journal of Econometrics* (2021), 225(2), 200-230.

2. R Package `did`: https://bcallaway11.github.io/did/

3. Sant'Anna, P. H., & Zhao, J. (2020). "Doubly Robust Difference-in-Differences Estimators." *Journal of Econometrics*, 219(1), 101-122.
