# SEM 估计方法

## 估计原理

SEM 通过最小化观测协方差矩阵 $\mathbf{S}$ 与模型隐含协方差矩阵 $\boldsymbol{\Sigma}(\boldsymbol{\theta})$ 之间的差异来估计参数。

---

## 最大似然估计 (Maximum Likelihood, ML)

### 拟合函数

$$
F_{ML} = \ln|\boldsymbol{\Sigma}(\boldsymbol{\theta})| + \text{tr}(\mathbf{S}\boldsymbol{\Sigma}^{-1}(\boldsymbol{\theta})) - \ln|\mathbf{S}| - p
$$

其中：
- $\boldsymbol{\Sigma}(\boldsymbol{\theta})$: 模型隐含协方差矩阵
- $\mathbf{S}$: 样本协方差矩阵
- $p$: 观测变量数量

### 假设

1. **多元正态性**: 观测变量服从多元正态分布
2. **大样本**: 渐近性质需要足够样本量
3. **连续变量**: 适用于连续测量

### 优点与局限

| 优点 | 局限 |
|------|------|
| 统计效率高 (正态下) | 对非正态敏感 |
| 标准误差准确 | 需要大样本 |
| 卡方检验可用 | 不适用于离散变量 |
| 软件广泛支持 | 收敛可能困难 |

---

## 稳健最大似然 (Robust ML)

### MLM (Satorra-Bentler 修正)

**卡方修正**:
$$
\chi^2_{SB} = \frac{\chi^2_{ML}}{c}
$$

其中 $c$ 是基于数据峰度的修正因子。

**适用场景**: 轻度到中度非正态数据

### MLR (Yuan-Bentler 修正)

- 稳健标准误差
- 均值和方差调整的检验统计量
- 适用于缺失数据 (FIML)

```r
# R lavaan
fit <- sem(model, data = data, estimator = "MLR")
```

---

## 加权最小二乘法 (Weighted Least Squares)

### 拟合函数

$$
F_{WLS} = (\mathbf{s} - \boldsymbol{\sigma}(\boldsymbol{\theta}))' \mathbf{W}^{-1} (\mathbf{s} - \boldsymbol{\sigma}(\boldsymbol{\theta}))
$$

其中：
- $\mathbf{s}$: 样本矩的向量化
- $\boldsymbol{\sigma}(\boldsymbol{\theta})$: 模型隐含矩的向量化
- $\mathbf{W}$: 权重矩阵

### WLS 变体

| 方法 | 权重矩阵 | 适用数据 |
|------|----------|----------|
| **WLS/ADF** | 完整 $\mathbf{W}$ | 非正态连续 |
| **DWLS** | 对角 $\mathbf{W}$ | 有序分类 |
| **ULS** | $\mathbf{I}$ (单位矩阵) | 探索性 |
| **WLSMV** | DWLS + 均值方差调整 | **有序分类推荐** |

### WLSMV 详解

**特点**:
- 使用多元多分类相关 (polychoric correlations)
- 对角权重矩阵提高效率
- 均值方差调整卡方统计量

**Python 示例**:
```python
from semopy import Model

model = Model(spec, estimator='WLSMV')
model.fit(data)
```

**R 示例**:
```r
fit <- sem(model, data = data,
           ordered = c("x1", "x2", "x3"),  # 声明有序变量
           estimator = "WLSMV")
```

---

## 贝叶斯估计

### 基本框架

$$
p(\boldsymbol{\theta} | \mathbf{S}) \propto p(\mathbf{S} | \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta})
$$

- **似然**: $p(\mathbf{S} | \boldsymbol{\theta})$
- **先验**: $p(\boldsymbol{\theta})$
- **后验**: $p(\boldsymbol{\theta} | \mathbf{S})$

### 优点

1. 小样本下更准确
2. 自然处理不确定性
3. 支持信息先验
4. 避免不正当解 (通过先验约束)

### 软件实现

```r
# R blavaan
library(blavaan)
fit <- bsem(model, data = data,
            n.chains = 4,
            burnin = 1000,
            sample = 5000)
```

---

## 两阶段最小二乘 (2SLS)

用于非递归模型或特定识别条件：

$$
\hat{\boldsymbol{\eta}} = (\mathbf{Z}'\mathbf{X})^{-1}\mathbf{Z}'\mathbf{y}
$$

其中 $\mathbf{Z}$ 是工具变量矩阵。

---

## 估计方法选择指南

```
                    数据类型？
                       │
            ┌──────────┴──────────┐
            │                     │
        连续变量               有序/分类
            │                     │
       ┌────┴────┐                │
       │         │                │
    正态分布   非正态            WLSMV
       │         │
      ML       MLR/MLM
```

### 决策表

| 数据特征 | 推荐估计方法 | 最低样本量 |
|----------|-------------|------------|
| 连续 + 正态 | ML | N ≥ 200 |
| 连续 + 非正态 | MLR | N ≥ 200 |
| 有序 (5+ 类别) | ML 或 MLR | N ≥ 200 |
| 有序 (< 5 类别) | WLSMV | N ≥ 200 |
| 二分类 | WLSMV | N ≥ 300 |
| 小样本 | Bayesian | N ≥ 50 |

---

## 缺失数据处理

### 完整信息最大似然 (FIML)

**原理**: 使用每个观测的可用信息估计似然

$$
\ell(\boldsymbol{\theta}) = \sum_{i=1}^{N} \ell_i(\boldsymbol{\theta} | \mathbf{y}_i^{obs})
$$

**假设**: 数据随机缺失 (MAR)

**实现**:
```r
# R lavaan
fit <- sem(model, data = data, missing = "fiml")
```

### 多重插补 (Multiple Imputation)

1. 生成 m 个完整数据集
2. 分别拟合模型
3. 合并结果 (Rubin's rules)

---

## 收敛问题诊断

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 不收敛 | 模型过于复杂 | 简化模型 |
| 负方差 | 模型错误设定 | 检查因子结构 |
| 大标准误 | 识别问题 | 添加约束 |
| 边界估计 | 参数接近边界 | 检查数据 |

### 改进收敛

```r
# 提供更好的起始值
start_values <- list(
    lambda = c(0.8, 0.8, 0.8),
    psi = diag(0.5, 3)
)

fit <- sem(model, data = data, start = start_values)

# 增加迭代次数
fit <- sem(model, data = data,
           control = list(iter.max = 10000))
```

---

## 参考文献

- Satorra, A., & Bentler, P. M. (2001). A scaled difference chi-square test statistic.
- Muthén, B. (1984). A general structural equation model with dichotomous, ordered categorical, and continuous latent variable indicators.
- Yuan, K. H., & Bentler, P. M. (2000). Three likelihood-based methods for mean and covariance structure analysis with nonnormal missing data.
