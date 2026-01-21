# SEM 报告标准

## APA 报告指南

基于 APA 第 7 版和 Schreiber et al. (2006) SEM 报告标准。

---

## 必报内容清单

### 1. Sample and Data / 样本与数据

- [ ] **Sample size** / 样本量 (N) - minimum recommended: N ≥ 200, or 10:1 ratio of cases to parameters
- [ ] Missing data handling method / 处理缺失数据的方法
- [ ] Data screening criteria / 数据筛选标准（如有）
- [ ] Multivariate normality test (if ML) / 检验多元正态性（如使用 ML）
- [ ] Descriptive statistics (M, SD, correlation matrix) / 描述统计（均值、标准差、相关矩阵）

### 2. 模型设定

- [ ] 理论模型的完整描述
- [ ] 模型图（路径图）
- [ ] 估计的自由参数数量
- [ ] 模型识别状态（自由度）
- [ ] 使用的软件和版本

### 3. 估计方法

- [ ] 估计器（ML、MLR、WLSMV 等）
- [ ] 选择估计器的理由
- [ ] 如何处理非正态数据（如适用）
- [ ] 如何处理有序数据（如适用）

### 4. 模型拟合

- [ ] 卡方统计量、自由度、p 值
- [ ] 至少两个增量拟合指数（CFI、TLI）
- [ ] 至少一个绝对拟合指数（RMSEA 含 CI、SRMR）
- [ ] 如使用 AIC/BIC 进行模型比较，需报告

### 5. Parameter Estimates / 参数估计

- [ ] All path coefficients (unstandardized and standardized) / 所有路径系数（非标准化和标准化）
- [ ] **Standard errors (SE)** / 标准误 - report for all estimated parameters
- [ ] Significance tests (z-value or p-value) / 显著性检验（z 值或 p 值）
- [ ] Factor loadings / 因子载荷
- [ ] Error variances / 误差方差
- [ ] Factor correlations/covariances / 因子间相关/协方差
- [ ] R² values for endogenous variables / R² 值（内生变量）

### 6. 模型修改（如有）

- [ ] 修改的理论依据
- [ ] 修正指数（如使用）
- [ ] 修改前后模型比较

---

## 标准报告模板

### 模型说明段落

```
为检验 [研究假设], 我们采用结构方程模型分析。分析使用 [软件名称]
(版本 X.X; [引用]) 进行, 采用 [估计方法] 估计。模型包含 [X] 个
潜变量, 由 [Y] 个观测指标测量, 共估计 [Z] 个自由参数。

样本量为 N = XXX。使用 [完整个案/FIML/多重插补] 处理缺失数据。
[如使用 ML] 多元正态性通过 Mardia 系数检验, 结果显示
[支持/不支持正态假设, 因此采用...]。
```

### 测量模型段落

```
首先通过验证性因子分析 (CFA) 评估测量模型。模型拟合指数表明
[良好/可接受/不良] 的拟合, χ²(df) = XX.XX, p = .XXX,
CFI = .XXX, TLI = .XXX, RMSEA = .XXX [90% CI: .XXX, .XXX],
SRMR = .XXX。

所有标准化因子载荷均显著 (p < .001), 范围从 .XX 到 .XX
(见表 X)。组合信度介于 .XX 至 .XX (均 > .70),
平均提取方差介于 .XX 至 .XX (均 > .50),
支持测量的收敛效度和区分效度。
```

### 结构模型段落

```
在测量模型确认后, 我们检验了假设的结构关系。结构模型显示
[良好/可接受] 的拟合, χ²(df) = XX.XX, p = .XXX,
CFI = .XXX, TLI = .XXX, RMSEA = .XXX [90% CI: .XXX, .XXX],
SRMR = .XXX。

[变量1] 对 [变量2] 有显著的 [正向/负向] 效应
(β = .XX, SE = .XX, p < .XXX), 支持假设 H1。
[其他路径结果...]

结构模型解释了 [变量Y] XX% 的方差 (R² = .XX)。
```

---

## 表格格式

### 表 1: 描述性统计与相关矩阵

```
表 1
描述性统计与相关矩阵 (N = XXX)

变量      M      SD      1      2      3      4      5
──────────────────────────────────────────────────────
1. X1    3.45   1.02    —
2. X2    3.67   0.98   .45**   —
3. X3    3.89   1.15   .52**  .48**   —
4. Y1    4.12   0.87   .38**  .41**  .44**   —
5. Y2    4.05   0.92   .35**  .39**  .42**  .67**  —

注: ** p < .01
```

### 表 2: 测量模型结果

```
表 2
验证性因子分析: 标准化因子载荷

构念/指标        λ       SE      95% CI         R²
─────────────────────────────────────────────────────
因子1 (ω = .XX, AVE = .XX)
  指标1         .78     .03    [.72, .84]     .61
  指标2         .82     .03    [.76, .88]     .67
  指标3         .75     .04    [.67, .83]     .56

因子2 (ω = .XX, AVE = .XX)
  指标4         .85     .02    [.81, .89]     .72
  指标5         .79     .03    [.73, .85]     .62
  指标6         .81     .03    [.75, .87]     .66
─────────────────────────────────────────────────────
注: 所有载荷显著, p < .001
```

### 表 3: 结构模型路径系数

```
表 3
结构方程模型: 路径系数

路径                          B       SE      β       p
───────────────────────────────────────────────────────
直接效应
  自变量 → 中介变量        0.45    0.08    .52    <.001
  中介变量 → 因变量        0.38    0.07    .44    <.001
  自变量 → 因变量          0.12    0.06    .15     .042

间接效应
  自变量 → 中介 → 因变量   0.17    0.04    .23    <.001

总效应
  自变量 → 因变量          0.29    0.08    .38    <.001
───────────────────────────────────────────────────────
模型拟合: χ²(df) = XX.XX, p = .XXX
CFI = .XXX, TLI = .XXX, RMSEA = .XXX, SRMR = .XXX
```

---

## 图表规范

### 路径图要素

1. **潜变量**: 椭圆或圆形
2. **观测变量**: 方形或矩形
3. **因果路径**: 单向箭头, 标注系数
4. **相关/协方差**: 双向弯箭头
5. **误差项**: 小圆圈或指向变量的箭头

### 示例路径图标注

```
     ┌─────────────────────────────────────────┐
     │                                          │
     │    ┌───┐     β = .52***     ┌───┐       │
     │    │ η1├────────────────────►│ η2│       │
     │    └─┬─┘                    └─┬─┘       │
     │      │                        │          │
     │  λ1  │ λ2  λ3            λ4  │ λ5  λ6  │
     │      ▼  ▼   ▼               ▼  ▼   ▼    │
     │     □  □   □               □  □   □     │
     │     x1 x2  x3              y1 y2  y3    │
     │                                          │
     └─────────────────────────────────────────┘

注: *** p < .001
    η = 潜变量; □ = 观测指标
    数字为标准化系数
```

---

## R 代码生成报告

```r
library(lavaan)
library(semTools)

# 拟合模型
fit <- sem(model, data = data)

# 自动生成报告
# 基本拟合信息
summary(fit, fit.measures = TRUE, standardized = TRUE)

# 格式化输出
parameterEstimates(fit, standardized = TRUE, ci = TRUE)

# 信度
reliability(fit)

# 模型比较
anova(fit1, fit2)

# 导出到 LaTeX (使用 stargazer 或手动)
```

---

## Python 代码生成报告

```python
from sem_estimator import fit_sem

result = fit_sem(data, model)

# 生成摘要
print(result.summary())

# 导出表格
result.factor_loadings.to_csv("factor_loadings.csv")
result.path_coefficients.to_csv("path_coefficients.csv")

# 生成 LaTeX 表格
def to_latex_table(df, caption):
    latex = df.to_latex(
        index=False,
        float_format="%.3f",
        caption=caption
    )
    return latex
```

---

## 常见报告错误

| 错误 | 正确做法 |
|------|----------|
| 仅报告显著路径 | 报告所有假设路径 |
| 忽略测量模型 | 先报告 CFA 结果 |
| 仅报告标准化系数 | 同时报告非标准化系数和 SE |
| 不报告置信区间 | 报告 95% CI |
| 遗漏模型图 | 包含清晰的路径图 |

---

## 参考文献

- Schreiber, J. B., et al. (2006). Reporting SEM and CFA results: A review. Journal of Educational Research, 99(6), 323-338.
- Kline, R. B. (2016). Principles and Practice of SEM (4th ed.). Chapter 13: Reporting Results.
- McDonald, R. P., & Ho, M. H. R. (2002). Principles and practice in reporting structural equation analyses. Psychological Methods, 7(1), 64-82.
