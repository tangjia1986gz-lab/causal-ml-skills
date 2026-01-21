# 机器学习因果推断 Skills 开发路线图

## 项目状态

**全部完成** - 21 个 Skills 已实现，采用 K-Dense 结构规范

## 开发阶段总览

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
  环境        经典方法     ML基础      前沿融合     综合实战
 (2 Skills)  (8 Skills)  (6 Skills)  (5 Skills)    完成
   [完成]      [完成]      [完成]      [完成]      [完成]
```

---

## Phase 0: 环境与基础设施 [完成]

### 0.1 setup-causal-ml-env [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P0 (前置依赖) |
| **复杂度** | 中等 |
| **依赖** | 无 |
| **类型** | Tool |
| **状态** | 完成 |

**功能清单**:
- [x] Python 环境检测与包安装
- [x] R 环境检测与 `rpy2` 配置
- [x] `grf`, `mediation`, `rdrobust` R 包安装
- [x] Stata 可用性检测 (可选)
- [x] 生成 `env_check.py` 诊断脚本
- [x] 生成 `requirements.txt`

### 0.2 scientific-writing-econ [完成] (新增)

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 中等 |
| **依赖** | 无 |
| **类型** | Tool |
| **状态** | 完成 |

**功能清单**:
- [x] LaTeX 表格生成
- [x] 学术写作规范
- [x] 引用格式管理
- [x] 出版级输出模板

---

## Phase 1: 经典因果方法 [完成]

### 1.1 causal-concept-guide [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 低 |
| **依赖** | 无 |
| **类型** | Knowledge |
| **状态** | 完成 |

**功能清单**:
- [x] 核心概念解释 (混杂、逆向因果、选择偏差)
- [x] 反事实框架咨询模板
- [x] 选题负面清单检查
- [x] 方法选择决策树

---

### 1.2 estimator-did [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P0 (最常用) |
| **复杂度** | 高 |
| **依赖** | setup-causal-ml-env |
| **类型** | Estimator |
| **状态** | 完成 |

**功能清单**:

*Pre-Estimation*:
- [x] 平行趋势检验 (Parallel Trends Test)
- [x] 数据平衡性检查

*Estimation*:
- [x] 经典 2x2 DID
- [x] 多期 DID (Staggered DID)
- [x] DID with covariates
- [x] Callaway-Sant'Anna 估计器

*Post-Estimation*:
- [x] 安慰剂检验 (Placebo Test)
- [x] 动态效应图 (Event Study Plot)
- [x] 出版级表格输出

---

### 1.3 estimator-rd [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P2 |
| **复杂度** | 高 |
| **依赖** | setup-causal-ml-env |
| **类型** | Estimator |
| **状态** | 完成 |

**功能清单**:

*Pre-Estimation*:
- [x] McCrary 密度检验 (操纵检验)
- [x] 断点处协变量平衡检验

*Estimation*:
- [x] Sharp RD
- [x] Fuzzy RD
- [x] 最优带宽选择

*Post-Estimation*:
- [x] RD 可视化 (断点图)
- [x] 带宽敏感性分析
- [x] 安慰剂断点检验

---

### 1.4 estimator-iv [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P2 |
| **复杂度** | 中等 |
| **依赖** | setup-causal-ml-env |
| **类型** | Estimator |
| **状态** | 完成 |

**功能清单**:

*Pre-Estimation*:
- [x] 第一阶段 F 统计量 (弱工具变量检验)
- [x] Stock-Yogo 临界值比较

*Estimation*:
- [x] 2SLS 估计
- [x] LIML 估计
- [x] GMM 估计

*Post-Estimation*:
- [x] 过度识别检验 (Sargan-Hansen)
- [x] 工具变量外生性讨论模板

---

### 1.5 estimator-psm [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P2 |
| **复杂度** | 中等 |
| **依赖** | setup-causal-ml-env |
| **类型** | Estimator |
| **状态** | 完成 |

**功能清单**:

*Pre-Estimation*:
- [x] 倾向得分估算 (Logit/Probit)
- [x] Common Support 检验

*Matching*:
- [x] 最近邻匹配 (1:1, 1:k)
- [x] 卡尺匹配 (Caliper)
- [x] 核匹配 (Kernel)
- [x] Mahalanobis 距离匹配

*Post-Estimation*:
- [x] 平衡性检验 (标准化均值差)
- [x] 匹配后 ATT/ATE 估计
- [x] PSM-DID 组合

---

### 1.6 panel-data-models [完成] (新增)

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 中等 |
| **依赖** | setup-causal-ml-env |
| **类型** | Tool |
| **状态** | 完成 |

**功能清单**:
- [x] 固定效应模型 (FE)
- [x] 随机效应模型 (RE)
- [x] Hausman 检验
- [x] 聚类标准误

---

### 1.7 time-series-econometrics [完成] (新增)

| 属性 | 值 |
|------|-----|
| **优先级** | P2 |
| **复杂度** | 高 |
| **依赖** | setup-causal-ml-env |
| **类型** | Tool |
| **状态** | 完成 |

**功能清单**:
- [x] 平稳性检验 (ADF, KPSS)
- [x] ARIMA 模型
- [x] VAR 模型
- [x] 协整检验与 VECM

---

### 1.8 discrete-choice-models [完成] (新增)

| 属性 | 值 |
|------|-----|
| **优先级** | P2 |
| **复杂度** | 中等 |
| **依赖** | setup-causal-ml-env |
| **类型** | Tool |
| **状态** | 完成 |

**功能清单**:
- [x] 二元选择模型 (Logit/Probit)
- [x] 有序选择模型 (Ordered Logit/Probit)
- [x] 多项选择模型 (Multinomial Logit)
- [x] 计数模型 (Poisson, Negative Binomial)

---

## Phase 2: 机器学习基础 [完成]

### 2.1 ml-preprocessing [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 中等 |
| **依赖** | setup-causal-ml-env |
| **类型** | Tool |
| **状态** | 完成 |

**功能清单**:
- [x] 缺失值诊断与处理策略
- [x] 异常值检测 (IQR, Z-score, Isolation Forest)
- [x] 特征工程辅助
- [x] 降维技术 (PCA, t-SNE)
- [x] 聚类分析 (K-Means, DBSCAN)

---

### 2.2 ml-model-linear [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 低 |
| **依赖** | ml-preprocessing |
| **类型** | Tool |
| **状态** | 完成 |

**功能清单**:
- [x] Ridge Regression
- [x] Lasso Regression (变量选择)
- [x] Elastic Net
- [x] 交叉验证调参
- [x] 特征重要性输出

---

### 2.3 ml-model-tree [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 中等 |
| **依赖** | ml-preprocessing |
| **类型** | Tool |
| **状态** | 完成 |

**功能清单**:
- [x] 决策树 (CART)
- [x] 随机森林
- [x] Gradient Boosting (XGBoost, LightGBM)
- [x] 特征重要性可视化
- [x] 部分依赖图 (PDP)
- [x] SHAP 值解释

---

### 2.4 ml-model-advanced [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P3 |
| **复杂度** | 高 |
| **依赖** | ml-preprocessing |
| **类型** | Tool |
| **状态** | 完成 |

**功能清单**:
- [x] SVM (分类/回归)
- [x] 神经网络基础 (MLP)
- [x] 模型选择与比较框架
- [x] 集成学习方法

---

### 2.5 econometric-eda [完成] (新增)

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 中等 |
| **依赖** | setup-causal-ml-env |
| **类型** | Tool |
| **状态** | 完成 |

**功能清单**:
- [x] 数据质量检查
- [x] 描述统计分析
- [x] 变量关系探索
- [x] 异常值检测
- [x] 面板数据 EDA

---

### 2.6 statistical-analysis [完成] (新增)

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 中等 |
| **依赖** | setup-causal-ml-env |
| **类型** | Tool |
| **状态** | 完成 |

**功能清单**:
- [x] 假设检验框架
- [x] 置信区间计算
- [x] 功效分析
- [x] 多重比较校正

---

## Phase 3: 前沿因果 ML 融合 [完成]

### 3.1 causal-ddml [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P0 (最前沿) |
| **复杂度** | 高 |
| **依赖** | ml-model-linear, ml-model-tree |
| **类型** | Estimator |
| **状态** | 完成 |

**功能清单**:

*核心流程*:
- [x] Stage 1: ML 预测 Y 和 D 的残差
- [x] Stage 2: 残差回归估计因果效应
- [x] Cross-fitting 实现

*模型选择*:
- [x] 支持 Lasso, Random Forest, XGBoost 作为 first-stage learner
- [x] 自动模型选择

*扩展*:
- [x] Partially Linear Model (PLR)
- [x] Interactive Regression Model (IRM)
- [x] 中介机制分析接口

*输出*:
- [x] 出版级表格
- [x] 稳健性检验报告
- [x] 结果解读模板

---

### 3.2 causal-mediation-ml [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P2 |
| **复杂度** | 高 |
| **依赖** | causal-ddml |
| **类型** | Estimator |
| **状态** | 完成 |

**功能清单**:
- [x] Average Direct Effect (ADE) 估计
- [x] Average Causal Mediation Effect (ACME) 估计
- [x] 敏感性分析
- [x] ML-enhanced 中介分析

---

### 3.3 causal-forest [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P1 |
| **复杂度** | 高 |
| **依赖** | ml-model-tree |
| **类型** | Estimator |
| **状态** | 完成 |

**功能清单**:

*核心*:
- [x] CATE (Conditional Average Treatment Effect) 估计
- [x] 异质性处理效应可视化

*分析*:
- [x] 变量重要性 (哪些变量驱动异质性)
- [x] Best Linear Projection
- [x] 政策学习 (Policy Learning)

*输出*:
- [x] CATE 分布图
- [x] 分组效应表格
- [x] 政策建议模板

---

### 3.4 bayesian-econometrics [完成] (新增)

| 属性 | 值 |
|------|-----|
| **优先级** | P2 |
| **复杂度** | 高 |
| **依赖** | setup-causal-ml-env |
| **类型** | Estimator |
| **状态** | 完成 |

**功能清单**:
- [x] 贝叶斯基础概念
- [x] 先验选择指南
- [x] MCMC 诊断
- [x] 层次模型

---

### 3.5 paper-replication-workflow [完成]

| 属性 | 值 |
|------|-----|
| **优先级** | P2 |
| **复杂度** | 高 |
| **依赖** | 所有 Estimator Skills |
| **类型** | Workflow |
| **状态** | 完成 |

**功能清单**:
- [x] 论文模型设定解析
- [x] 自动调用对应 Estimator
- [x] 复现结果对比
- [x] 差异诊断
- [x] 出版级表格生成

---

## 完成汇总

### Skills 统计

| 分类 | 计划数量 | 完成数量 | 状态 |
|------|---------|---------|------|
| 基础设施 | 2 | 2 | 完成 |
| 经典因果方法 | 8 | 8 | 完成 |
| 机器学习基础 | 6 | 6 | 完成 |
| 前沿因果 ML | 5 | 5 | 完成 |
| **总计** | **21** | **21** | **完成** |

### 完整 Skills 清单

#### 基础设施 (2 个)
1. `setup-causal-ml-env` - 环境配置
2. `scientific-writing-econ` - 学术写作

#### 经典因果方法 (8 个)
3. `causal-concept-guide` - 因果概念指南
4. `estimator-did` - 双重差分
5. `estimator-rd` - 断点回归
6. `estimator-iv` - 工具变量
7. `estimator-psm` - 倾向得分匹配
8. `panel-data-models` - 面板数据模型
9. `time-series-econometrics` - 时间序列计量
10. `discrete-choice-models` - 离散选择模型

#### 机器学习基础 (6 个)
11. `ml-preprocessing` - 数据预处理
12. `ml-model-linear` - 线性模型
13. `ml-model-tree` - 树模型
14. `ml-model-advanced` - 高级 ML 模型
15. `econometric-eda` - 计量经济学 EDA
16. `statistical-analysis` - 统计分析

#### 前沿因果 ML (5 个)
17. `causal-ddml` - 双重机器学习
18. `causal-forest` - 因果森林
19. `causal-mediation-ml` - ML 中介分析
20. `bayesian-econometrics` - 贝叶斯计量
21. `paper-replication-workflow` - 论文复现工作流

---

## 里程碑完成记录

| 里程碑 | 完成 Skills | 能力 | 状态 |
|--------|-------------|------|------|
| **M1** | 1-3 | 可运行 DID 分析 | 完成 |
| **M2** | 4-6 | 支持 ML 特征工程 | 完成 |
| **M3** | 7-8 | 支持 DDML 和因果森林 | 完成 |
| **M4** | 9-16 | 完整因果工具箱 | 完成 |
| **M5** | 17-21 | 端到端论文复现 | 完成 |

---

## 验收标准 (全部满足)

每个 Skill 已满足:

1. **功能完整**: 所有列出的功能项已实现
2. **文档齐全**: SKILL.md 按 K-Dense 模板编写
3. **测试通过**: 完整示例可运行
4. **输出规范**: 表格符合出版标准
5. **结构规范**: 符合 K-Dense 结构 (references/, scripts/, assets/)

---

## 更新日志

### v2.0.0 (2025-01)
- Skills 数量从 14 扩展到 21
- 新增 7 个 Skills:
  - `scientific-writing-econ`
  - `panel-data-models`
  - `time-series-econometrics`
  - `discrete-choice-models`
  - `econometric-eda`
  - `statistical-analysis`
  - `bayesian-econometrics`
- 采用 K-Dense 结构规范
- 所有 Skills 添加 references/ 目录
- 新增开发脚本 (generate_skill_scaffold.py, validate_skill.py)
- 新增共享资源模板 (assets/latex/, assets/markdown/)
- 扩展共享库 (lib/python/visualization.py)
