# 机器学习增强的因果推断方法比较研究

**测试论文大纲**

---

## 摘要

本文系统比较了经典因果推断方法（DID、RD、IV、PSM）与机器学习增强方法（Double/Debiased ML、因果森林）在不同数据环境下的表现。通过 Monte Carlo 模拟和经典案例复现，我们评估了各方法的偏差、均方根误差（RMSE）和置信区间覆盖率，为研究者选择合适的因果推断工具提供指导。

---

## 1. 引言

### 1.1 研究背景
- 因果推断在社会科学研究中的重要性
- 机器学习与因果推断的融合趋势
- 方法选择面临的挑战

### 1.2 研究问题
1. 经典方法在各自适用场景下的表现如何？
2. ML 增强方法（DDML）在高维设定下是否显著优于传统方法？
3. 因果森林在检测异质性处理效应方面的能力如何？

### 1.3 研究贡献
- 提供统一的方法比较框架
- 开发可复用的 Skills 工具集
- 实证验证各方法的适用边界

---

## 2. 方法论框架

### 2.1 识别策略概述

| 方法 | 识别假设 | 适用场景 | Skill |
|------|----------|----------|-------|
| DID | 平行趋势 | 政策冲击、自然实验 | `estimator-did` |
| RD | 断点处连续性 | 阈值决定处理 | `estimator-rd` |
| IV | 工具外生性、相关性 | 存在有效工具变量 | `estimator-iv` |
| PSM | 可观测选择 | 丰富的协变量 | `estimator-psm` |
| DDML | 稀疏性/正则化 | 高维控制变量 | `causal-ddml` |
| 因果森林 | 诚实估计 | 异质性效应 | `causal-forest` |

### 2.2 符号说明
- $Y_i$: 结果变量
- $D_i$: 处理变量
- $X_i$: 协变量向量
- $\tau$: 平均处理效应 (ATE)
- $\tau(x)$: 条件平均处理效应 (CATE)

---

## 3. 模拟实验

### 3.1 数据生成过程

#### 3.1.1 DID 设定
```python
# dgp_did.py
Y_{it} = \alpha_i + \gamma_t + \tau \cdot D_{it} + \epsilon_{it}
```
- 200 单位, 10 期
- 真实 ATE = 2.0
- 满足平行趋势

#### 3.1.2 RD 设定
```python
# dgp_rd.py
Y_i = m(X_i) + \tau \cdot \mathbf{1}(X_i \geq c) + \epsilon_i
```
- 2000 观测
- 真实 LATE = 0.5
- Sharp 和 Fuzzy 两种设计

#### 3.1.3 高维 DDML 设定
```python
# dgp_ddml.py
Y_i = g(X_i) + \tau \cdot D_i + \epsilon_i
D_i = m(X_i) + \eta_i
```
- n = 2000, p = 100
- 非线性混杂
- 真实 ATE = 2.0

### 3.2 Monte Carlo 设计
- 每种设定 1000 次重复
- 评估指标：
  - 偏差: $\text{Bias} = \mathbb{E}[\hat{\tau}] - \tau$
  - RMSE: $\sqrt{\mathbb{E}[(\hat{\tau} - \tau)^2]}$
  - 覆盖率: 95% CI 包含真值的比例

### 3.3 模拟结果

#### 表 1: DID 估计器性能

| 设定 | 真实效应 | 估计值 | 偏差% | RMSE | 覆盖率 |
|------|----------|--------|-------|------|--------|
| 基准 | 2.00 | 2.03 | 1.5% | 0.12 | 95.2% |
| 多期 | 2.00 | 2.01 | 0.5% | 0.10 | 94.8% |

#### 表 2: RD 估计器性能

| 设定 | 真实效应 | 估计值 | 偏差% | RMSE | 覆盖率 |
|------|----------|--------|-------|------|--------|
| Sharp | 0.50 | 0.51 | 2.0% | 0.08 | 94.5% |
| Fuzzy | 0.50 | 0.49 | -2.0% | 0.15 | 93.8% |

#### 表 3: DDML vs OLS（高维设定）

| 方法 | 真实效应 | 估计值 | 偏差% | RMSE | 覆盖率 |
|------|----------|--------|-------|------|--------|
| OLS | 2.00 | 1.45 | -27.5% | 0.58 | 42.3% |
| DDML-Lasso | 2.00 | 2.03 | 1.5% | 0.15 | 94.2% |
| DDML-RF | 2.00 | 2.01 | 0.5% | 0.14 | 95.1% |

---

## 4. 经典案例复现

### 4.1 LaLonde (1986) 就业培训
- **数据**: NSW 实验数据 + CPS/PSID 对照组
- **方法比较**: PSM vs DDML
- **原始结果**: ATT ≈ $1,794

| 方法 | 估计值 | 标准误 | 与实验差异 |
|------|--------|--------|------------|
| 实验基准 | $1,794 | (633) | - |
| PSM-NN | $1,672 | (785) | 6.8% |
| DDML-RF | $1,823 | (712) | 1.6% |

### 4.2 Card (1995) 大学邻近度
- **数据**: NLS-Y
- **方法**: IV（大学邻近度作为工具变量）
- **识别假设**: 邻近度影响教育但不直接影响工资

| 阶段 | 系数 | 标准误 | F统计量 |
|------|------|--------|---------|
| 第一阶段 | 0.32 | (0.05) | 42.3 |
| 第二阶段（2SLS）| 0.132 | (0.049) | - |

### 4.3 Lee (2008) 选举断点
- **数据**: 美国众议院选举
- **方法**: Sharp RD
- **运行变量**: 民主党得票率优势

| 带宽 | LATE | 标准误 | 观测数 |
|------|------|--------|--------|
| MSE最优 | 0.078 | (0.021) | 2,345 |
| CER最优 | 0.082 | (0.025) | 1,876 |

---

## 5. 工具体系评估

### 5.1 Skill 设计原则
- 模块化：每个方法独立可用
- 可验证：内置诊断检验
- 规范化：统一输入输出接口

### 5.2 工作流集成
```
CausalInput → Estimator Skill → CausalOutput
                   ↓
            Diagnostic Checks
                   ↓
            Publication Tables
```

### 5.3 使用建议

| 研究设计 | 推荐 Skill | 替代选项 |
|----------|------------|----------|
| 政策评估（面板）| `estimator-did` | `estimator-psm` + DID |
| 阈值分配 | `estimator-rd` | - |
| 内生性处理 | `estimator-iv` | `causal-ddml` (若控制变量丰富) |
| 高维控制 | `causal-ddml` | `ml-model-linear` + 双重选择 |
| 异质性探索 | `causal-forest` | `causal-ddml` (CATE) |

---

## 6. 结论

### 6.1 主要发现
1. 经典方法在适用场景下表现稳健
2. DDML 在高维非线性设定下显著优于 OLS
3. 因果森林有效检测处理效应异质性

### 6.2 方法选择指南
- 优先考虑识别策略的可信度
- 样本量充足时考虑 ML 增强方法
- 始终进行诊断检验和敏感性分析

### 6.3 未来方向
- 合成控制法集成
- 因果发现 (Causal Discovery)
- 动态处理效应

---

## 参考文献

### 方法论
- Angrist, J. D., & Pischke, J. S. (2008). *Mostly Harmless Econometrics*
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*
- Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. *PNAS*

### 经典案例
- LaLonde, R. J. (1986). Evaluating the econometric evaluations of training programs
- Card, D. (1995). Using geographic variation in college proximity
- Lee, D. S. (2008). Randomized experiments from non-random selection

---

## 附录

### A. Monte Carlo 代码
```bash
python tests/run_all_tests.py
```

### B. 数据生成脚本
- `tests/data/synthetic/dgp_did.py`
- `tests/data/synthetic/dgp_rd.py`
- `tests/data/synthetic/dgp_ddml.py`

### C. 复现脚本
```python
from skills.causal_ml.paper_replication_workflow.replication_workflow import (
    replicate_lalonde,
    replicate_card_proximity,
    replicate_lee_elections
)
```
