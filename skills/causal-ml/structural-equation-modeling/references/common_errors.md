# SEM 常见错误与避免方法

## 模型设定错误

### 1. 忽视模型识别

**错误**: 设定无法识别的模型（自由度 < 0 或经验性欠识别）。

**后果**:
- 参数估计不唯一
- 标准误无穷大
- 模型无法收敛

**避免方法**:
```python
# 检查自由度
n_vars = len(observed_variables)
n_unique = n_vars * (n_vars + 1) // 2
n_params = count_free_parameters(model)
df = n_unique - n_params

if df < 0:
    print("警告: 模型欠识别!")
    print(f"需要添加至少 {-df} 个约束")
```

**正确做法**:
- 每个因子至少 3 个指标
- 检查模型自由度
- 为每个潜变量设定尺度

---

### 2. 过度拟合模型

**错误**: 基于修正指数 (MI) 不断添加路径直到获得"好"的拟合。

**后果**:
- 模型失去理论意义
- 结果不可复制
- 过拟合样本特异性

**警示信号**:
- 添加了多个理论外的相关误差
- 模型修改超过 2-3 次
- MI 驱动而非理论驱动

**避免方法**:
```r
# 仅考虑有理论依据的修改
mi <- modindices(fit, sort = TRUE, minimum.value = 10)

# 检查每个建议是否有理论支持
# 例如: 同一方法的指标可以有相关误差
# 但不同方法的指标误差相关需要特别解释
```

**正确做法**:
- 先验设定模型
- 仅在理论支持下修改
- 使用验证样本交叉验证

---

### 3. 混淆 CFA 与 EFA

**错误**: 在数据驱动情况下使用 CFA，或在理论驱动情况下使用 EFA。

| 方法 | 适用场景 | 目的 |
|------|----------|------|
| EFA | 探索性，无先验假设 | 发现因子结构 |
| CFA | 验证性，有先验假设 | 检验因子结构 |

**避免方法**:
```python
# 如果不确定因子结构
# 步骤 1: EFA 探索
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer(n_factors=3, rotation='oblimin')
fa.fit(data)

# 步骤 2: 在新样本上用 CFA 验证
# 或使用样本分割
train, test = train_test_split(data, test_size=0.5)
```

---

## 估计错误

### 4. 样本量不足

**错误**: 使用过小的样本进行 SEM 分析。

**参考样本量**:
| 模型复杂度 | 最低 N | 推荐 N |
|------------|--------|--------|
| 简单 (< 10 指标) | 100 | 200 |
| 中等 (10-30 指标) | 200 | 400 |
| 复杂 (> 30 指标) | 400 | 800+ |

**检验公式**:
- N:q 比例 (N = 样本量, q = 自由参数): N:q ≥ 10:1
- MacCallum 等 (1999): 基于 RMSEA 的功效分析

**替代方案**:
```r
# 小样本选项
# 1. 贝叶斯 SEM
library(blavaan)
fit <- bsem(model, data = data)

# 2. PLS-SEM (对样本量要求较低)
library(seminr)
```

---

### 5. 对非正态数据使用 ML

**错误**: 数据明显非正态时仍使用标准 ML 估计。

**后果**:
- 卡方膨胀
- 标准误偏误
- 拟合指数失真

**诊断**:
```r
# 检查多元正态性
library(MVN)
mvn(data, mvnTest = "mardia")
# Mardia 系数: 偏度 < 2, 峰度 < 7 通常可接受
```

**解决方案**:
```r
# 使用稳健估计
fit <- sem(model, data = data, estimator = "MLR")

# 或 Bootstrap
fit <- sem(model, data = data, se = "bootstrap", bootstrap = 5000)
```

---

### 6. 不当处理有序数据

**错误**: 将有序分类变量（如 Likert 量表）视为连续变量。

**影响**:
- 低估因子载荷
- 标准误偏误
- 模型拟合失真

**决策指南**:
| 类别数 | 推荐处理方法 |
|--------|--------------|
| 2 类别 | WLSMV |
| 3-4 类别 | WLSMV |
| 5-6 类别 | MLR 或 WLSMV |
| 7+ 类别 | MLR (视为连续) |

**正确做法**:
```r
# 声明有序变量
fit <- sem(model, data = data,
           ordered = c("item1", "item2", "item3"),
           estimator = "WLSMV")
```

---

## 解释错误

### 7. 将相关解释为因果

**错误**: 仅基于横截面数据和模型拟合就声称因果关系。

**SEM 不能证明因果关系**, 只能检验数据与假设因果模型的一致性。

**正确表述**:
```
错误: "X 导致 Y (β = .45, p < .001)"
正确: "数据支持 X 对 Y 的假设效应 (β = .45, p < .001)"
```

**增强因果推断**:
- 使用纵向数据
- 控制混杂变量
- 利用实验设计
- 比较等价模型

---

### 8. 忽视等价模型

**错误**: 不考虑存在其他同样拟合良好的替代模型。

**示例**:
```
模型 A: X → M → Y
模型 B: X → Y → M
两个模型可能有相同的拟合!
```

**检查方法**:
```python
# 拟合替代模型
model_a = "M ~ X; Y ~ M"
model_b = "Y ~ X; M ~ Y"

result_a = fit_sem(data, model_a)
result_b = fit_sem(data, model_b)

# 比较拟合
print(f"模型 A: χ² = {result_a.fit_indices.chi_square}")
print(f"模型 B: χ² = {result_b.fit_indices.chi_square}")
```

---

### 9. 过度依赖单一拟合指数

**错误**: 仅根据 CFI > 0.95 就认为模型"完美"。

**问题**:
- 不同指数反映不同方面
- 阈值是指导性的，非绝对标准
- 高 CFI 不意味着模型正确

**正确做法**:
```python
def comprehensive_fit_assessment(result):
    fit = result.fit_indices

    report = []
    report.append(f"χ²({fit.df}) = {fit.chi_square:.2f}, p = {fit.p_value:.4f}")
    report.append(f"CFI = {fit.cfi:.3f} {'✓' if fit.cfi >= 0.95 else '✗'}")
    report.append(f"TLI = {fit.tli:.3f} {'✓' if fit.tli >= 0.95 else '✗'}")
    report.append(f"RMSEA = {fit.rmsea:.3f} {'✓' if fit.rmsea <= 0.06 else '✗'}")
    report.append(f"SRMR = {fit.srmr:.3f} {'✓' if fit.srmr <= 0.08 else '✗'}")

    # 综合判断
    good_indices = sum([
        fit.cfi >= 0.95,
        fit.tli >= 0.95,
        fit.rmsea <= 0.06,
        fit.srmr <= 0.08
    ])

    if good_indices >= 3:
        report.append("\n综合评估: 良好拟合")
    elif good_indices >= 2:
        report.append("\n综合评估: 可接受拟合")
    else:
        report.append("\n综合评估: 拟合不佳，需修正模型")

    return "\n".join(report)
```

---

### 10. Heywood 案例处理不当

**错误**: 忽视或强制约束负方差估计。

**Heywood 案例**: 估计出负的误差方差或 > 1 的标准化载荷。

**正确诊断**:
```r
# 检查不当解
inspect(fit, what = "est")$theta  # 误差方差

# 检查边界估计
lavInspect(fit, "post.check")
```

**可能原因与解决**:
| 原因 | 解决方案 |
|------|----------|
| 模型错误设定 | 重新检查因子结构 |
| 样本量过小 | 增加样本或简化模型 |
| 异常值 | 检查并处理异常值 |
| 起始值不当 | 提供更好的起始值 |

---

## 检查清单

### 分析前

- [ ] 样本量是否充足？(N:q ≥ 10:1)
- [ ] 模型是否识别？(df ≥ 0)
- [ ] 数据是否满足估计方法假设？
- [ ] 缺失数据是否适当处理？

### 分析中

- [ ] 模型是否收敛？
- [ ] 有无 Heywood 案例？
- [ ] 标准误是否合理？
- [ ] 参数估计是否在合理范围？

### 分析后

- [ ] 整体拟合是否可接受？
- [ ] 局部拟合（残差）是否良好？
- [ ] 参数解释是否合理？
- [ ] 是否考虑了等价模型？

---

## 参考文献

- Kline, R. B. (2016). Principles and Practice of SEM. Chapter 11: Troubleshooting.
- Boomsma, A. (2000). Reporting analyses of covariance structures. Structural Equation Modeling, 7(3), 461-483.
- Chen, F., et al. (2008). An empirical evaluation of the use of fixed cutoff points in RMSEA test statistic. Sociological Methods & Research, 36(4), 462-494.
