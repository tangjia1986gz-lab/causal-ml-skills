# 机器学习因果推断 Skills 开发路线图

## 开发阶段总览

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
  环境        经典方法     ML基础      前沿融合     综合实战
 (1 Skill)   (5 Skills)  (4 Skills)  (3 Skills)  (1 Skill)
```

---

## Phase 0: 环境与基础设施 🔧

### 0.1 setup-causal-ml-env

| 属性 | 值 |
|------|-----|
| **优先级** | P0 (前置依赖) |
| **复杂度** | 中等 |
| **依赖** | 无 |
| **类型** | Tool |

**功能清单**:
- [ ] Python 环境检测与包安装
- [ ] R 环境检测与 `rpy2` 配置
- [ ] `grf`, `mediation`, `rdrobust` R 包安装
- [ ] Stata 可用性检测 (可选)
- [ ] 生成 `env_check.py` 诊断脚本
- [ ] 生成 `requirements.txt`

**交付标准**:
```bash
# 用户运行后应看到
✅ Python 3.10+ detected
✅ econml 0.15.0 installed
✅ doubleml 0.7.0 installed
✅ R 4.3.0 detected
✅ grf 2.3.0 installed
✅ rpy2 bridge working
⚠️ Stata not detected (optional)
```

---

## Phase 1: 经典因果方法 📊

### 1.1 causal-concept-guide

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 低 |
| **依赖** | 无 |
| **类型** | Knowledge |

**功能清单**:
- [ ] 核心概念解释 (混杂、逆向因果、选择偏差)
- [ ] 反事实框架咨询模板
- [ ] 选题负面清单检查
- [ ] 方法选择决策树

---

### 1.2 estimator-did ⭐ (核心优先)

| 属性 | 值 |
|------|-----|
| **优先级** | P0 (最常用) |
| **复杂度** | 高 |
| **依赖** | setup-causal-ml-env |
| **类型** | Estimator |

**功能清单**:

*Pre-Estimation*:
- [ ] 平行趋势检验 (Parallel Trends Test)
  - 可视化趋势图
  - 统计检验 (Event Study)
- [ ] 数据平衡性检查

*Estimation*:
- [ ] 经典 2x2 DID
- [ ] 多期 DID (Staggered DID)
- [ ] DID with covariates
- [ ] Callaway-Sant'Anna 估计器 (处理异质性处理时间)

*Post-Estimation*:
- [ ] 安慰剂检验 (Placebo Test)
- [ ] 动态效应图 (Event Study Plot)
- [ ] 出版级表格输出

**Python 实现基础**:
```python
# 核心依赖
from linearmodels import PanelOLS
from statsmodels.regression.linear_model import OLS
import did  # Callaway-Sant'Anna
```

---

### 1.3 estimator-rd

| 属性 | 值 |
|------|-----|
| **优先级** | P2 |
| **复杂度** | 高 |
| **依赖** | setup-causal-ml-env |
| **类型** | Estimator |

**功能清单**:

*Pre-Estimation*:
- [ ] McCrary 密度检验 (操纵检验)
- [ ] 断点处协变量平衡检验

*Estimation*:
- [ ] Sharp RD
- [ ] Fuzzy RD
- [ ] 最优带宽选择 (MSE-optimal, CER-optimal)
- [ ] 局部多项式回归

*Post-Estimation*:
- [ ] RD 可视化 (断点图)
- [ ] 带宽敏感性分析
- [ ] 安慰剂断点检验

**核心依赖**:
```python
# Python
from rdrobust import rdrobust, rdbwselect, rdplot

# R (via rpy2)
library(rdrobust)
library(rddensity)
```

---

### 1.4 estimator-iv

| 属性 | 值 |
|------|-----|
| **优先级** | P2 |
| **复杂度** | 中等 |
| **依赖** | setup-causal-ml-env |
| **类型** | Estimator |

**功能清单**:

*Pre-Estimation*:
- [ ] 第一阶段 F 统计量 (弱工具变量检验)
- [ ] Stock-Yogo 临界值比较

*Estimation*:
- [ ] 2SLS 估计
- [ ] LIML 估计 (弱 IV 稳健)
- [ ] GMM 估计

*Post-Estimation*:
- [ ] 过度识别检验 (Sargan-Hansen)
- [ ] 工具变量外生性讨论模板

**核心依赖**:
```python
from linearmodels.iv import IV2SLS, IVLIML, IVGMM
```

---

### 1.5 estimator-psm

| 属性 | 值 |
|------|-----|
| **优先级** | P2 |
| **复杂度** | 中等 |
| **依赖** | setup-causal-ml-env |
| **类型** | Estimator |

**功能清单**:

*Pre-Estimation*:
- [ ] 倾向得分估算 (Logit/Probit)
- [ ] Common Support 检验

*Matching*:
- [ ] 最近邻匹配 (1:1, 1:k)
- [ ] 卡尺匹配 (Caliper)
- [ ] 核匹配 (Kernel)
- [ ] Mahalanobis 距离匹配

*Post-Estimation*:
- [ ] 平衡性检验 (标准化均值差)
- [ ] 匹配后 ATT/ATE 估计
- [ ] PSM-DID 组合

**核心依赖**:
```python
from causalml.match import NearestNeighborMatch
from sklearn.linear_model import LogisticRegression
```

---

## Phase 2: 机器学习基础 🤖

### 2.1 ml-preprocessing

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 中等 |
| **依赖** | setup-causal-ml-env |
| **类型** | Tool |

**功能清单**:
- [ ] 缺失值诊断与处理策略
- [ ] 异常值检测 (IQR, Z-score, Isolation Forest)
- [ ] 特征工程辅助
- [ ] 降维技术 (PCA, t-SNE for visualization)
- [ ] 聚类分析 (K-Means, DBSCAN)

---

### 2.2 ml-model-linear

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 低 |
| **依赖** | ml-preprocessing |
| **类型** | Tool |

**功能清单**:
- [ ] Ridge Regression
- [ ] Lasso Regression (变量选择)
- [ ] Elastic Net
- [ ] 交叉验证调参
- [ ] 特征重要性输出

**因果应用场景**:
- 高维控制变量筛选
- Double Selection (Belloni et al.)

---

### 2.3 ml-model-tree

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 中等 |
| **依赖** | ml-preprocessing |
| **类型** | Tool |

**功能清单**:
- [ ] 决策树 (CART)
- [ ] 随机森林
- [ ] Gradient Boosting (XGBoost, LightGBM)
- [ ] 特征重要性可视化
- [ ] 部分依赖图 (PDP)
- [ ] SHAP 值解释

---

### 2.4 ml-model-advanced

| 属性 | 值 |
|------|-----|
| **优先级** | P3 |
| **复杂度** | 高 |
| **依赖** | ml-preprocessing |
| **类型** | Tool |

**功能清单**:
- [ ] SVM (分类/回归)
- [ ] 神经网络基础 (MLP)
- [ ] 模型选择与比较框架

---

## Phase 3: 前沿因果 ML 融合 🚀

### 3.1 causal-ddml ⭐ (核心优先)

| 属性 | 值 |
|------|-----|
| **优先级** | P0 (最前沿) |
| **复杂度** | 高 |
| **依赖** | ml-model-linear, ml-model-tree |
| **类型** | Estimator |

**功能清单**:

*核心流程*:
- [ ] Stage 1: ML 预测 Y 和 D 的残差
- [ ] Stage 2: 残差回归估计因果效应
- [ ] Cross-fitting 实现

*模型选择*:
- [ ] 支持 Lasso, Random Forest, XGBoost 作为 first-stage learner
- [ ] 自动模型选择

*扩展*:
- [ ] Partially Linear Model (PLR)
- [ ] Interactive Regression Model (IRM)
- [ ] 中介机制分析接口

*输出*:
- [ ] 出版级表格
- [ ] 稳健性检验报告
- [ ] 结果解读模板

**核心依赖**:
```python
from doubleml import DoubleMLPLR, DoubleMLIRM
from econml.dml import DML, LinearDML, CausalForestDML
```

---

### 3.2 causal-mediation-ml

| 属性 | 值 |
|------|-----|
| **优先级** | P2 |
| **复杂度** | 高 |
| **依赖** | causal-ddml |
| **类型** | Estimator |

**功能清单**:
- [ ] Average Direct Effect (ADE) 估计
- [ ] Average Causal Mediation Effect (ACME) 估计
- [ ] 敏感性分析
- [ ] ML-enhanced 中介分析

**核心依赖**:
```python
# Python 自定义或
# R via rpy2
library(mediation)
```

---

### 3.3 causal-forest

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 高 |
| **依赖** | ml-model-tree |
| **类型** | Estimator |

**功能清单**:

*核心*:
- [ ] CATE (Conditional Average Treatment Effect) 估计
- [ ] 异质性处理效应可视化

*分析*:
- [ ] 变量重要性 (哪些变量驱动异质性)
- [ ] Best Linear Projection
- [ ] 政策学习 (Policy Learning)

*输出*:
- [ ] CATE 分布图
- [ ] 分组效应表格
- [ ] 政策建议模板

**核心依赖**:
```python
# 推荐使用 R 的 grf (最权威)
# R via rpy2
library(grf)
cf <- causal_forest(X, Y, W)

# 或 Python
from econml.grf import CausalForest
```

---

## Phase 4: 综合实战 📚

### 4.1 paper-replication-workflow

| 属性 | 值 |
|------|-----|
| **优先级** | P2 |
| **复杂度** | 高 |
| **依赖** | 所有 Estimator Skills |
| **类型** | Workflow |

**功能清单**:
- [ ] 论文模型设定解析
- [ ] 自动调用对应 Estimator
- [ ] 复现结果对比
- [ ] 差异诊断
- [ ] 出版级表格生成

---

## 开发优先级矩阵

```
                    价值
                    高 │
                       │  ★ causal-ddml      ★ estimator-did
                       │
                       │  ○ causal-forest    ○ ml-model-tree
                       │
                       │  ○ estimator-rd     ○ estimator-iv
                    低 │  △ ml-model-advanced
                       └──────────────────────────────────────
                          低                              高
                                     使用频率

★ P0 优先开发   ○ P1/P2 中等优先   △ P3 最后开发
```

---

## 建议开发顺序

### Sprint 1: 核心基础
1. `setup-causal-ml-env` - 环境配置
2. `estimator-did` - 最常用的经典方法
3. `causal-concept-guide` - 概念指南

### Sprint 2: ML 能力
4. `ml-preprocessing` - 数据预处理
5. `ml-model-linear` - 正则化回归
6. `ml-model-tree` - 树模型

### Sprint 3: 前沿融合
7. `causal-ddml` - 双重机器学习
8. `causal-forest` - 因果森林

### Sprint 4: 完善扩展
9. `estimator-rd` - 断点回归
10. `estimator-iv` - 工具变量
11. `estimator-psm` - 倾向得分匹配
12. `causal-mediation-ml` - 因果中介

### Sprint 5: 集成
13. `ml-model-advanced` - 高级 ML 模型
14. `paper-replication-workflow` - 论文复现流程

---

## 里程碑

| 里程碑 | 完成 Skills | 能力 |
|--------|-------------|------|
| **M1** | 1-3 | 可运行 DID 分析 |
| **M2** | 4-6 | 支持 ML 特征工程 |
| **M3** | 7-8 | 支持 DDML 和因果森林 |
| **M4** | 9-12 | 完整因果工具箱 |
| **M5** | 13-14 | 端到端论文复现 |

---

## 验收标准

每个 Skill 必须满足:

1. **功能完整**: 所有列出的功能项已实现
2. **文档齐全**: SKILL.md 按模板编写
3. **测试通过**: 至少 1 个完整示例可运行
4. **输出规范**: 表格符合出版标准
