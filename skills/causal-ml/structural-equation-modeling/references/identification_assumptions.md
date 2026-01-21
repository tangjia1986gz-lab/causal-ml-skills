# SEM 识别假设与条件

## 模型识别的基本概念

**识别 (Identification)**: 指模型参数能否从观测数据的协方差矩阵中唯一确定。

### 识别分类

| 类型 | 定义 | 自由度 |
|------|------|--------|
| **欠识别 (Under-identified)** | 参数有无穷多解 | df < 0 |
| **恰好识别 (Just-identified)** | 参数有唯一解，但无法检验模型拟合 | df = 0 |
| **过度识别 (Over-identified)** | 参数有唯一解，可检验模型拟合 | df > 0 |

---

## 必要条件

### 1. 自由度条件 (t-Rule)

$$
df = \frac{p(p+1)}{2} - t \geq 0
$$

其中：
- $p$: 观测变量数量
- $t$: 待估计的自由参数数量
- $p(p+1)/2$: 协方差矩阵中的独立元素数

**注意**: 这是必要但非充分条件。

### 2. 尺度设定 (Scaling)

每个潜变量必须有确定的尺度。两种常用方法：

| 方法 | 实现 | 适用场景 |
|------|------|----------|
| **固定因子载荷** | 将第一个指标的载荷固定为 1 | 默认方法 |
| **固定因子方差** | 将潜变量方差固定为 1 | 需要比较载荷时 |

```
# lavaan 语法示例
# 方法1: 固定载荷（默认）
Factor =~ x1 + x2 + x3

# 方法2: 固定方差
Factor =~ NA*x1 + x2 + x3
Factor ~~ 1*Factor
```

### 3. 指标数量规则

| 因子数 | 最低指标要求 | 建议 |
|--------|-------------|------|
| 单因子 | 3 个指标 | 4+ 个指标 |
| 两因子 | 每个因子 2 个（因子相关） | 每个因子 3+ 个 |
| 多因子 | 每个因子至少 2 个 | 每个因子 3+ 个 |

---

## 充分条件

### 3 指标规则 (Three-Indicator Rule)

**CFA 模型**: 每个因子至少有 3 个指标时，测量模型通常是识别的。

**数学验证**:
- 单因子，3 个指标：
  - 观测信息: 3(3+1)/2 = 6 个
  - 参数: 3 个载荷 + 3 个误差方差 = 6 个
  - 固定 1 个载荷后: 5 个参数
  - df = 6 - 5 = 1 (过度识别)

### 2 指标规则 (Two-Indicator Rule)

当每个因子仅有 2 个指标时，需要额外约束：
- 因子间必须相关，或
- 与其他因子有结构路径

---

## 结构模型识别

### 递归模型 (Recursive Model)

**定义**: 结构方程中没有反馈回路（单向因果）。

**识别条件**: 如果测量模型识别，递归结构模型通常识别。

$$
\boldsymbol{\eta} = B\boldsymbol{\eta} + \Gamma\boldsymbol{\xi} + \boldsymbol{\zeta}
$$

其中 $B$ 是严格下三角矩阵（无对角线元素）。

### 非递归模型 (Non-Recursive Model)

**定义**: 存在反馈回路或相互因果。

**识别条件**: 需要工具变量或额外约束。

**秩条件 (Rank Condition)**:
对于方程 $i$，排除在该方程中的变量必须能够线性独立地解释其他方程中的内生变量。

---

## 常见识别问题及解决

### 1. Heywood 案例 (负误差方差)

**症状**: 估计出负的误差方差或大于 1 的标准化载荷。

**可能原因**:
- 样本量过小
- 模型错误设定
- 因子过度提取
- 多重共线性

**解决方案**:
- 检查数据质量
- 简化模型
- 增加样本量
- 固定问题参数

### 2. 经验性欠识别

**症状**: 模型理论上识别但无法收敛。

**可能原因**:
- 数据中某些关系过弱
- 起始值不当
- 数据不满足假设

**解决方案**:
- 改进起始值
- 检查数据分布
- 简化模型

### 3. 等价模型 (Equivalent Models)

**问题**: 多个结构不同的模型产生相同的模型拟合。

**处理**:
- 基于理论选择模型
- 报告可能的等价模型
- 使用纵向数据或实验设计区分

---

## 诊断代码

### Python (semopy)

```python
from semopy import Model

model = """
    F1 =~ x1 + x2 + x3
    F2 =~ y1 + y2 + y3
    F2 ~ F1
"""

sem = Model(model)

# 检查识别状态
print(f"自由参数数量: {len(sem.param_vals)}")
# 计算观测信息数量
n_vars = len(sem.vars['observed'])
n_unique = n_vars * (n_vars + 1) // 2
print(f"观测信息数量: {n_unique}")
print(f"自由度: {n_unique - len(sem.param_vals)}")
```

### R (lavaan)

```r
library(lavaan)

model <- '
    F1 =~ x1 + x2 + x3
    F2 =~ y1 + y2 + y3
    F2 ~ F1
'

# 检查识别
fit <- sem(model, data = data)
lavInspect(fit, "free")  # 查看自由参数
lavInspect(fit, "npar")  # 参数数量
fitmeasures(fit, "df")   # 自由度
```

---

## 参考文献

- Bollen, K. A. (1989). Structural Equations with Latent Variables. Chapter 7: Identification.
- Kline, R. B. (2016). Principles and Practice of SEM (4th ed.). Chapter 6: Identification.
- Kenny, D. A., & Milan, S. (2012). Identification: A Nontechnical Discussion. In Handbook of SEM.
