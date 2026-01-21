# SEM 诊断测试与模型评估

## 整体模型拟合

### 卡方检验

**统计量**:
$$
\chi^2 = (N-1) F_{ML}
$$

**假设**:
- $H_0$: $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}(\boldsymbol{\theta})$ (模型正确设定)
- $H_1$: $\boldsymbol{\Sigma} \neq \boldsymbol{\Sigma}(\boldsymbol{\theta})$

**局限**:
- 对样本量敏感 (N > 200 几乎总是显著)
- 对多元正态性假设敏感
- 不适合作为唯一拟合标准

### 替代拟合指数

| 指数 | 公式 | 可接受 | 良好 |
|------|------|--------|------|
| χ²/df | $\chi^2 / df$ | < 3 | < 2 |
| **CFI** | $1 - \frac{\max(\chi^2_t - df_t, 0)}{\max(\chi^2_t - df_t, \chi^2_0 - df_0, 0)}$ | ≥ 0.90 | ≥ 0.95 |
| **TLI** | $\frac{\chi^2_0/df_0 - \chi^2_t/df_t}{\chi^2_0/df_0 - 1}$ | ≥ 0.90 | ≥ 0.95 |
| **RMSEA** | $\sqrt{\frac{\chi^2 - df}{df(N-1)}}$ | < 0.08 | < 0.06 |
| **SRMR** | $\sqrt{\frac{2\sum\sum(s_{ij} - \hat{\sigma}_{ij})^2}{p(p+1)}}$ | < 0.10 | < 0.08 |

### Hu & Bentler (1999) 组合准则

**推荐组合**:
- CFI ≥ 0.95 **且** SRMR ≤ 0.08
- TLI ≥ 0.95 **且** RMSEA ≤ 0.06

**注意**: 这些是指导性标准，而非绝对阈值。

---

## 局部拟合诊断

### 残差分析

**标准化残差**:
$$
z_{ij} = \frac{s_{ij} - \hat{\sigma}_{ij}}{se(s_{ij} - \hat{\sigma}_{ij})}
$$

**解释**:
- |z| > 2.58: 该协方差配对拟合不佳 (α = 0.01)
- |z| > 1.96: 该协方差配对拟合有问题 (α = 0.05)

**实现**:
```python
from semopy import Model

model = Model(spec)
model.fit(data)

# 获取残差
residuals = model.inspect(what='residuals')
```

```r
# R lavaan
residuals(fit, type = "standardized")
```

### 修正指数 (Modification Indices)

**定义**: 如果释放某个固定参数，卡方统计量预期的下降量。

$$
MI_j = \frac{(\partial F / \partial \theta_j)^2}{\partial^2 F / \partial \theta_j^2}
$$

**使用准则**:
- MI > 10: 可能需要考虑添加路径
- MI > 3.84: χ²(1) 的 0.05 临界值

**警告**:
- 仅在有理论支持时添加路径
- 不要基于数据驱动过度修改模型

```r
# R lavaan
modindices(fit, sort = TRUE, minimum.value = 10)
```

### 期望参数变化 (EPC)

**定义**: 如果释放参数，该参数的预期估计值。

$$
EPC_j = -\frac{\partial F / \partial \theta_j}{\partial^2 F / \partial \theta_j^2}
$$

**标准化 EPC (SEPC)**: 更易解释的标准化版本

---

## 测量模型诊断

### 因子载荷标准

| 标准化载荷 | 解释 | 行动 |
|------------|------|------|
| < 0.40 | 差指标 | 考虑删除 |
| 0.40 - 0.70 | 可接受 | 保留但注意 |
| > 0.70 | 良好指标 | 理想情况 |

### 信度指标

**组合信度 (Composite Reliability, ω)**:
$$
\omega = \frac{(\sum_{i=1}^{k} \lambda_i)^2}{(\sum_{i=1}^{k} \lambda_i)^2 + \sum_{i=1}^{k} \theta_{ii}}
$$

标准: ω ≥ 0.70

**平均提取方差 (AVE)**:
$$
AVE = \frac{\sum_{i=1}^{k} \lambda_i^2}{k}
$$

标准: AVE ≥ 0.50 (表示潜变量解释了指标方差的 50% 以上)

### 区分效度

**Fornell-Larcker 准则**: 每个构念的 AVE 应大于其与其他构念相关系数的平方。

$$
AVE_i > r_{ij}^2 \quad \forall j \neq i
$$

**HTMT (Heterotrait-Monotrait Ratio)**:
$$
HTMT_{ij} = \frac{\bar{r}_{ij}}{\sqrt{\bar{r}_{ii} \cdot \bar{r}_{jj}}}
$$

标准: HTMT < 0.85 (严格) 或 < 0.90 (宽松)

---

## 结构模型诊断

### R² (解释方差)

$$
R^2_{\eta_j} = 1 - \frac{\psi_{jj}}{\text{Var}(\eta_j)}
$$

**解释**:
- R² < 0.25: 弱解释力
- 0.25 ≤ R² < 0.50: 中等解释力
- R² ≥ 0.50: 强解释力

### 效应分解

| 效应类型 | 定义 | 计算 |
|----------|------|------|
| **直接效应** | 直接路径系数 | $\gamma_{ij}$ 或 $\beta_{ij}$ |
| **间接效应** | 通过中介的效应 | $\prod$ 路径系数 |
| **总效应** | 直接 + 间接 | 所有路径之和 |

```r
# R lavaan
lavInspect(fit, "rsquare")
lavInspect(fit, "effects")
```

---

## 模型比较

### 嵌套模型: 卡方差异检验

$$
\Delta\chi^2 = \chi^2_{constrained} - \chi^2_{unconstrained}
$$
$$
\Delta df = df_{constrained} - df_{unconstrained}
$$

**显著性**: $p = P(\chi^2_{\Delta df} > \Delta\chi^2)$

### 稳健估计下的差异检验 (Satorra-Bentler)

$$
T_D = \frac{T_{c0} - T_{c1}}{cd}
$$

需要计算修正因子 $cd$。

```r
# R lavaan
anova(fit_constrained, fit_unconstrained)

# 对于 MLR 估计
lavTestLRT(fit_constrained, fit_unconstrained, method = "satorra.bentler.2010")
```

### 非嵌套模型: 信息准则

| 准则 | 公式 | 使用 |
|------|------|------|
| **AIC** | $\chi^2 + 2t$ | 选择较小值 |
| **BIC** | $\chi^2 + t \ln(N)$ | 选择较小值 |

**解释差异**:
- ΔAIC > 10: 强烈支持较小 AIC 模型
- ΔBIC > 10: 强烈证据

---

## 测量不变性检验

### 检验层次

| 层次 | 约束 | Δχ² 检验 | ΔCFI 准则 |
|------|------|----------|-----------|
| **形态等值** | 相同因子结构 | 基线 | — |
| **弱等值 (Metric)** | + 因子载荷相等 | 显著性检验 | |ΔCFI| < 0.01 |
| **强等值 (Scalar)** | + 截距相等 | 显著性检验 | |ΔCFI| < 0.01 |
| **严格等值** | + 残差方差相等 | 显著性检验 | |ΔCFI| < 0.01 |

**Cheung & Rensvold (2002)**: ΔCFI < 0.01 作为等值证据

```r
# R lavaan
library(semTools)

# 测量不变性检验
mi <- measurementInvariance(model, data = data, group = "group")
```

---

## Python 实现

```python
from sem_estimator import fit_sem
import pandas as pd

# 拟合模型
result = fit_sem(data, model_spec)

# 获取拟合指数
print(f"Chi-square: {result.fit_indices.chi_square:.3f}")
print(f"df: {result.fit_indices.df}")
print(f"p-value: {result.fit_indices.p_value:.4f}")
print(f"CFI: {result.fit_indices.cfi:.3f}")
print(f"TLI: {result.fit_indices.tli:.3f}")
print(f"RMSEA: {result.fit_indices.rmsea:.3f}")
print(f"SRMR: {result.fit_indices.srmr:.3f}")

# 拟合评估
def assess_fit(result):
    fit = result.fit_indices

    issues = []
    if fit.cfi < 0.90:
        issues.append("CFI < 0.90: 模型拟合不佳")
    if fit.rmsea > 0.08:
        issues.append("RMSEA > 0.08: 近似误差较大")
    if fit.srmr > 0.10:
        issues.append("SRMR > 0.10: 残差较大")

    if not issues:
        return "模型拟合可接受"
    return "\n".join(issues)

print(assess_fit(result))
```

---

## 报告模板

```markdown
### 模型拟合

整体模型拟合通过多个指标评估。卡方检验结果为 χ²(df) = XX.XX, p = .XXX。
由于卡方检验对样本量敏感，我们进一步检查了替代拟合指数：
- CFI = .XX (标准 ≥ .95)
- TLI = .XX (标准 ≥ .95)
- RMSEA = .XX, 90% CI [.XX, .XX] (标准 ≤ .06)
- SRMR = .XX (标准 ≤ .08)

综合来看，模型拟合 [良好/可接受/需要改进]。

### 测量模型

所有标准化因子载荷均大于 0.XX，表明指标与潜变量有 [强/中等/弱] 关联。
组合信度 (ω) 范围为 .XX 到 .XX (标准 ≥ .70)。
平均提取方差 (AVE) 范围为 .XX 到 .XX (标准 ≥ .50)。

### 结构模型

[报告路径系数、显著性、R²]
```

---

## 参考文献

- Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes. SEM, 6(1), 1-55.
- Cheung, G. W., & Rensvold, R. B. (2002). Evaluating goodness-of-fit indexes. SEM, 9(2), 233-255.
- Fornell, C., & Larcker, D. F. (1981). Evaluating SEM with unobservable variables. JMR, 18(1), 39-50.
