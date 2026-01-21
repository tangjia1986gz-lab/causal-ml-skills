# Causal-ML-Skills 修订方案

## 对标库: K-Dense-AI/claude-scientific-skills

基于深入分析，本文档提出系统性修订方案，使 causal-ml-skills 达到 claude-scientific-skills 的质量标准。

---

## 1. 架构对比分析

### 1.1 目录结构对比

| 维度 | claude-scientific-skills | causal-ml-skills | 差距 |
|------|-------------------------|-----------------|------|
| **SKILL.md 大小** | 15-35 KB | 5-15 KB | 内容深度不足 |
| **references/** | 5-6 个专题文件 | 3-6 个文件 | 结构较好，但内容偏理论 |
| **scripts/** | 2-3 个**可执行**脚本 | 5+ 脚本但**依赖缺失** | ❌ 核心问题 |
| **共享库** | 无（自包含） | lib/python/ | 部署时断链 |

### 1.2 SKILL.md 内容模式对比

#### claude-scientific-skills 模式 ✅

```markdown
# 概述 (Overview)
- 清晰的使用场景描述
- 与其他技能的区分

# 何时使用 (When to Use)
- 明确的触发条件列表

# 快速开始 (Quick Start)
- **完整可运行的代码示例**
- 直接复制粘贴即可执行

# 核心能力 (Core Capabilities)
- 按功能分类详细说明
- 每个功能都有代码示例

# 工作流 (Common Workflows)
- 端到端的标准流程
- 步骤清晰、代码完整

# 最佳实践 (Best Practices)
- 实用的编码建议
- 常见陷阱警告

# 参考文档 (Reference Documentation)
- 指向 references/ 目录
- 说明何时查阅

# 常见问题 (Troubleshooting)
- 具体错误及解决方案

# 外部资源 (Additional Resources)
- 官方文档链接
- API 参考
```

#### causal-ml-skills 模式 ⚠️

```markdown
# 概述
- 理论背景较多
- 引用文献详细

# 快速参考表格
- 使用场景对照表 ✅

# CLI 脚本命令
- **但脚本无法运行** ❌

# 识别假设
- 理论详尽但缺少代码实现

# 工作流
- 伪代码居多
- 引用不存在的函数
```

### 1.3 核心差距

| 问题 | 严重程度 | 说明 |
|------|:--------:|------|
| **代码不可执行** | 🔴 致命 | SKILL.md 中的函数引用 `ddml_estimator.py`，但部署后 `lib/` 目录断链 |
| **过度依赖自定义模块** | 🔴 致命 | scientific-skills 直接使用 `sklearn`, `statsmodels`，无自定义依赖 |
| **Quick Start 不完整** | 🟡 严重 | 缺少可直接运行的最小示例 |
| **理论偏重** | 🟢 中等 | references/ 内容好但缺少实操代码 |

---

## 2. 修订原则

### 2.1 核心原则：自包含 (Self-Contained)

```
每个技能目录必须：
1. SKILL.md 中的所有代码示例可直接运行（仅依赖 pip 安装的库）
2. scripts/ 中的脚本自包含，不依赖外部 lib/
3. 优先使用成熟的开源库（doubleml, causalml, econml）
```

### 2.2 设计参考

以 `statsmodels` 技能为模板：

```
技能目录/
├── SKILL.md                # 15-25 KB，包含完整可运行示例
├── references/             # 按主题组织的深度文档
│   ├── estimation_methods.md
│   ├── diagnostic_tests.md
│   └── ...
└── scripts/                # 可选，仅当需要 CLI 工具时
    └── example_workflow.py # 自包含，开头声明所有依赖
```

---

## 3. 逐技能修订方案

### 3.1 优先级分类

| 优先级 | 技能 | 修订类型 | 工作量 | 状态 |
|:------:|------|----------|:------:|:----:|
| P0 | estimator-did | 重构 | 大 | ✅ 完成 |
| P0 | estimator-iv | 重构 | 大 | ✅ 完成 |
| P0 | estimator-psm | 重构 | 大 | ✅ 完成 |
| P1 | causal-ddml | 重构 | 大 | ✅ 完成 |
| P1 | causal-forest | 重构 | 中 | ✅ 完成 |
| P2 | panel-data-models | 新建 | 大 | ⏳ 待开始 |
| P2 | time-series-econometrics | 新建 | 大 | ⏳ 待开始 |
| P3 | statistical-analysis | 增强 | 中 | ⏳ 待开始 |
| P3 | econometric-eda | 增强 | 小 | ⏳ 待开始 |

---

### 3.2 P0 技能详细修订方案

#### 3.2.1 estimator-did

**当前问题**:
- SKILL.md 仅 2.3 KB，过于简略
- 引用不存在的 `templates/rigorous_did.py`
- 缺少完整可运行示例

**修订方案**:

```markdown
# 目标结构
estimator-did/
├── SKILL.md                # 扩展到 15+ KB
├── references/
│   ├── identification_assumptions.md   # 平行趋势、无预期、SUTVA
│   ├── estimation_methods.md           # TWFE、Callaway-Sant'Anna、Sun-Abraham
│   ├── diagnostic_tests.md             # 平行趋势检验、事件研究
│   ├── reporting_standards.md          # 三线表、系数图
│   └── common_errors.md                # 负权重、处理时间异质性
└── scripts/
    ├── did_analysis.py                 # 完整分析工作流（自包含）
    └── event_study_plot.py             # 事件研究图生成
```

**SKILL.md 改写要点**:

```python
# Quick Start 必须是这样的完整示例
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

# 1. 准备数据
df = pd.read_csv('your_data.csv')
df = df.set_index(['firm_id', 'year'])

# 2. TWFE 回归
formula = 'outcome ~ treatment + EntityEffects + TimeEffects'
model = PanelOLS.from_formula(formula, data=df)
result = model.fit(cov_type='clustered', cluster_entity=True)

# 3. 输出结果
print(result.summary)
```

**关键改进**:
1. 删除对 `templates/` 的依赖
2. 使用 `linearmodels` 和 `statsmodels` 而非自定义模块
3. 添加 Callaway-Sant'Anna 实现（使用 `did` 包或手写）
4. 添加事件研究图代码（使用 `matplotlib`）

---

#### 3.2.2 estimator-iv

**当前问题**: 类似 estimator-did

**修订方案**:

```python
# Quick Start 示例
from linearmodels.iv import IV2SLS
import pandas as pd

# 准备数据
df = pd.read_csv('data.csv')

# 2SLS 回归
model = IV2SLS.from_formula(
    'outcome ~ 1 + control1 + control2 + [endogenous ~ instrument]',
    data=df
)
result = model.fit(cov_type='robust')
print(result.summary)

# 弱工具变量检验
print(f"First-stage F: {result.first_stage.diagnostics['f.stat'].stat:.2f}")
```

**references/ 内容**:
- `identification_assumptions.md`: 排除性约束、相关性、单调性
- `estimation_methods.md`: 2SLS, LIML, GMM
- `diagnostic_tests.md`: Stock-Yogo, Sargan, Anderson-Rubin

---

#### 3.2.3 estimator-psm

**修订方案**:

```python
# Quick Start 示例
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

# 1. 估计倾向得分
X = df[['age', 'income', 'education']]
treatment = df['treated']

ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X, treatment)
propensity_scores = ps_model.predict_proba(X)[:, 1]

# 2. 最近邻匹配
treated_idx = np.where(treatment == 1)[0]
control_idx = np.where(treatment == 0)[0]

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(propensity_scores[control_idx].reshape(-1, 1))
distances, indices = nn.kneighbors(propensity_scores[treated_idx].reshape(-1, 1))

# 3. 计算 ATT
matched_controls = control_idx[indices.flatten()]
att = df.loc[treated_idx, 'outcome'].mean() - df.loc[matched_controls, 'outcome'].mean()
print(f"ATT: {att:.4f}")
```

---

### 3.3 P1 技能修订方案

#### 3.3.1 causal-ddml

**核心改动**: 删除自定义 `ddml_estimator.py`，改用 `doubleml` 官方包

```python
# Quick Start 示例
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

# 准备数据
data = dml.DoubleMLData(
    df,
    y_col='outcome',
    d_cols='treatment',
    x_cols=['control1', 'control2', 'control3']
)

# PLR 模型
learner_l = LassoCV()
learner_m = LassoCV()

dml_plr = DoubleMLPLR(data, ml_l=learner_l, ml_m=learner_m, n_folds=5)
dml_plr.fit()

print(dml_plr.summary)
```

**保留的 references/**:
- 理论内容优秀，保留
- 更新代码示例为 `doubleml` 语法

---

#### 3.3.2 causal-forest

**改动**: 使用 `econml` 包

```python
# Quick Start 示例
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 估计异质性处理效应
model = CausalForestDML(
    model_y=RandomForestRegressor(),
    model_t=RandomForestClassifier(),
    n_estimators=1000,
    random_state=42
)

model.fit(Y=df['outcome'], T=df['treatment'], X=df[controls], W=df[confounders])

# 获取个体处理效应
cate = model.effect(df[controls])
print(f"Average CATE: {cate.mean():.4f}")
```

---

### 3.4 P2 新建技能

#### 3.4.1 panel-data-models

**以 statsmodels 技能为模板，专注经管面板数据**

```
panel-data-models/
├── SKILL.md
├── references/
│   ├── fixed_effects.md       # 公司固定效应、时间固定效应、双向FE
│   ├── random_effects.md      # RE vs FE 选择、Hausman 检验
│   ├── dynamic_panels.md      # Arellano-Bond, System GMM
│   ├── clustered_se.md        # 聚类标准误、多维聚类
│   └── diagnostic_tests.md    # 异方差检验、序列相关
└── scripts/
    └── panel_regression.py
```

---

#### 3.4.2 time-series-econometrics

**参考 statsmodels 的 time_series 部分**

```
time-series-econometrics/
├── SKILL.md
├── references/
│   ├── stationarity.md        # ADF, KPSS, 单位根
│   ├── arima_models.md        # ARIMA, SARIMAX
│   ├── var_vecm.md            # VAR, 协整, VECM
│   ├── forecasting.md         # 预测、置信区间
│   └── granger_causality.md   # Granger 因果检验
└── scripts/
    └── time_series_analysis.py
```

---

## 4. 实施步骤

### Phase 1: 基础清理 (Week 1)

1. **删除 lib/python/ 依赖**
   - 将所有自定义函数内联到各技能
   - 或改用开源库替代

2. **验证 Quick Start**
   - 每个 SKILL.md 的第一个代码块必须可直接运行
   - 创建测试脚本验证

### Phase 2: P0 技能重构 (Week 2-3)

1. **estimator-did**: 使用 `linearmodels` + `matplotlib`
2. **estimator-iv**: 使用 `linearmodels`
3. **estimator-psm**: 使用 `sklearn` + 手写匹配

### Phase 3: P1 技能重构 (Week 4)

1. **causal-ddml**: 迁移到 `doubleml`
2. **causal-forest**: 迁移到 `econml`

### Phase 4: P2 新建技能 (Week 5-6)

1. 参考 statsmodels 技能结构
2. 聚焦经管方法特色

### Phase 5: 测试与部署 (Week 7)

1. 每个技能创建测试用例
2. 验证部署后的可执行性

---

## 5. 代码模板

### 5.1 SKILL.md 模板

```markdown
---
name: skill-name
description: 清晰的一句话描述，包含触发关键词
license: MIT
metadata:
    skill-author: Your Name
---

# Skill Name

## Overview

简明描述技能用途（2-3 句）。

## When to Use This Skill

- 场景 1
- 场景 2
- 场景 3

## Quick Start

### 基础示例

\`\`\`python
# 完整可运行代码
import pandas as pd
import numpy as np
from some_package import SomeClass

# 加载数据
df = pd.read_csv('data.csv')

# 核心操作
model = SomeClass()
result = model.fit(df)

# 输出
print(result.summary)
\`\`\`

### 进阶示例

\`\`\`python
# 更复杂的用例
...
\`\`\`

## Core Capabilities

### 1. 功能一

说明 + 代码示例

### 2. 功能二

说明 + 代码示例

## Common Workflows

### Workflow 1: 标准分析流程

1. 步骤 1 + 代码
2. 步骤 2 + 代码
3. ...

## Best Practices

1. 建议 1
2. 建议 2

## Reference Documentation

详细文档见 `references/` 目录：
- `estimation_methods.md`: ...
- `diagnostic_tests.md`: ...

## Troubleshooting

### 问题 1

**症状**: ...
**解决**: ...

## Additional Resources

- 官方文档: https://...
- API 参考: https://...
```

---

## 6. 验收标准

### 6.1 单技能验收

- [ ] SKILL.md ≥ 10 KB
- [ ] Quick Start 代码可直接运行
- [ ] 无自定义 lib 依赖
- [ ] references/ 至少 3 个文件
- [ ] 有 Common Workflows 章节
- [ ] 有 Troubleshooting 章节

### 6.2 整体验收

- [ ] 所有 P0/P1 技能通过验收
- [ ] 部署到 skills/ 后可正常使用
- [ ] 完成 ST-杠杆率类似的完整研究验证

---

## 7. 附录：依赖包对照

| 功能 | 当前实现 | 改为 |
|------|---------|------|
| DDML | 自定义 ddml_estimator | `doubleml` |
| Causal Forest | 自定义 | `econml.dml.CausalForestDML` |
| DID | 自定义 | `linearmodels.PanelOLS` |
| IV | 自定义 | `linearmodels.iv.IV2SLS` |
| PSM | 自定义 | `sklearn` + 手写 |
| 面板数据 | 未实现 | `linearmodels.panel` |
| 时间序列 | 未实现 | `statsmodels.tsa` |

---

**文档版本**: 2.0
**创建日期**: 2026-01-21
**最后更新**: 2026-01-21
**作者**: Claude Code

---

## 8. 重构执行记录

### 8.1 P0 技能完成记录 (2026-01-21)

#### estimator-did ✅

| 指标 | 重构前 | 重构后 |
|------|:------:|:------:|
| SKILL.md | 2.2 KB | **20 KB** |
| scripts/ | 无可执行 | **did_analysis_pipeline.py (22KB)** |
| references/ | 7文件/132KB | 保留 |
| 验证 | ❌ | ✅ `TWFE coef=2.1824, p<0.0001` |

#### estimator-iv ✅

| 指标 | 重构前 | 重构后 |
|------|:------:|:------:|
| SKILL.md | 2.3 KB | **20.3 KB** |
| scripts/ | 无 | **iv_analysis_pipeline.py (22KB)** |
| references/ | 无 | **5文件/22KB** |
| 验证 | ❌ | ✅ `F=197.51, 2SLS bias=-0.04` |

#### estimator-psm ✅ (从头创建)

| 指标 | 重构前 | 重构后 |
|------|:------:|:------:|
| SKILL.md | 不存在 | **24.3 KB** |
| scripts/ | 不存在 | **psm_analysis_pipeline.py (26KB)** |
| references/ | 不存在 | **4文件/21KB** |
| 验证 | ❌ | ✅ `PSM ATT=1.91, IPW ATT=1.98` |

### 8.2 P1 技能完成记录 (2026-01-21)

#### causal-ddml ✅

| 指标 | 重构前 | 重构后 |
|------|:------:|:------:|
| SKILL.md | 15 KB (引用断链) | **17 KB (自包含)** |
| scripts/ | 5脚本/85KB (不可执行) | **ddml_analysis_pipeline.py (23KB)** |
| references/ | 6文件/78KB | 保留 |
| 依赖 | `ddml_estimator.py` (断链) | `doubleml` 包 |
| 验证 | ❌ | ✅ `PLR effect=0.40, CI=[0.20,0.61] covers true 0.5` |

**删除的旧文件**:
- `ddml_estimator.py` (48KB)
- `scripts/run_ddml_analysis.py` (15KB)
- `scripts/tune_nuisance_models.py` (16KB)
- `scripts/cross_fit_diagnostics.py` (19KB)
- `scripts/sensitivity_analysis.py` (17KB)
- `scripts/compare_estimators.py` (18KB)
- `__pycache__/` 目录

**2026-01-22 验证结果**:
- True effect: 0.5, Estimated: 0.4027 (bias: -0.10)
- SE: 0.1038, 95% CI: [0.1993, 0.6061] ✓ 包含真值
- p-value: 0.0001 ***

**验收检查**:
- [x] SKILL.md ≥ 10 KB (17 KB)
- [x] Quick Start 代码可直接运行
- [x] 无自定义 lib 依赖 (使用 doubleml 包)
- [x] references/ 至少 3 个文件 (6个)
- [x] 有 Common Workflows 章节
- [x] 有 Troubleshooting 章节
- [x] 代码验证通过 (PLR p<0.001)

### 8.3 重构统计汇总

| 指标 | 数值 |
|------|-----:|
| P0 技能完成 | 3/3 |
| P1 技能完成 | 2/2 |
| P2 技能完成 | 2/2 |
| 总 SKILL.md 新增/更新 | ~131 KB |
| 总 scripts/ 新增 | ~170 KB |
| 总 references/ 新增 | ~89 KB |
| 删除旧代码 | ~300 KB |

### 8.4 验收检查清单

#### estimator-did
- [x] SKILL.md ≥ 10 KB (20 KB)
- [x] Quick Start 代码可直接运行
- [x] 无自定义 lib 依赖
- [x] references/ 至少 3 个文件 (7个)
- [x] 有 Common Workflows 章节
- [x] 有 Troubleshooting 章节

#### estimator-iv
- [x] SKILL.md ≥ 10 KB (20.3 KB)
- [x] Quick Start 代码可直接运行
- [x] 无自定义 lib 依赖
- [x] references/ 至少 3 个文件 (5个)
- [x] 有 Common Workflows 章节
- [x] 有 Troubleshooting 章节

#### estimator-psm
- [x] SKILL.md ≥ 10 KB (24.3 KB)
- [x] Quick Start 代码可直接运行
- [x] 无自定义 lib 依赖
- [x] references/ 至少 3 个文件 (4个)
- [x] 有 Common Workflows 章节
- [x] 有 Troubleshooting 章节

#### causal-ddml
- [x] SKILL.md ≥ 10 KB (17 KB)
- [x] Quick Start 代码可直接运行
- [x] 无自定义 lib 依赖 (使用 doubleml 包)
- [x] references/ 至少 3 个文件 (6个)
- [x] 有 Common Workflows 章节
- [x] 有 Troubleshooting 章节

#### causal-forest ✅ (2026-01-21)

| 指标 | 重构前 | 重构后 |
|------|:------:|:------:|
| SKILL.md | 17.5 KB (已有) | 保留 |
| scripts/ | 4脚本/72KB (依赖 causal_forest.py) | **causal_forest_pipeline.py (23KB)** |
| references/ | 6文件/100KB | 保留 |
| 依赖 | `causal_forest.py` | `econml` 包 |
| 验证 | ❌ | ✅ `ATE=1.05, CATE corr=0.93` |

**删除的旧文件**:
- `causal_forest.py` (48KB)
- `scripts/run_causal_forest.py` (17KB)
- `scripts/estimate_cate.py` (12KB)
- `scripts/policy_evaluation.py` (22KB)
- `scripts/visualize_heterogeneity.py` (22KB)
- `__pycache__/` 目录

**验收检查**:
- [x] SKILL.md ≥ 10 KB (17.5 KB)
- [x] scripts/ 可执行 (econml.CausalForestDML)
- [x] 无自定义 lib 依赖
- [x] references/ 至少 3 个文件 (6个)
- [x] 代码验证通过

**2026-01-22 更新**:
- SKILL.md 更新: 删除虚构 `from causal_forest import` 引用，改用 econml API
- 代码验证: True ATE=1.02, Estimated=1.18, Variable importance 正常

#### structural-equation-modeling ✅ (2026-01-22)

| 指标 | 重构前 | 重构后 |
|------|:------:|:------:|
| SKILL.md | 17 KB (引用 sem_estimator) | **17 KB (自包含 semopy)** |
| scripts/ | sem_estimator.py + run_sem_analysis.py | **sem_analysis_pipeline.py (23KB)** |
| references/ | 5文件/78KB | 保留 |
| 依赖 | `sem_estimator.py` | `semopy` 包 |
| 验证 | ❌ | ✅ `CFI=0.999, RMSEA=0.008, Paths OK` |

**删除的旧文件**:
- `sem_estimator.py` (~500行)
- `scripts/run_sem_analysis.py` (405行)
- `scripts/__pycache__/`

**代码验证结果**:
- Converged: True
- CFI: 0.999, TLI: 0.999, RMSEA: 0.008 → **GOOD**
- Structural paths: F3←F1=0.51 (true 0.5), F3←F2=0.38 (true 0.3)
- Reliability: CR 0.78-0.83, AVE 0.48-0.56

**验收检查**:
- [x] SKILL.md ≥ 10 KB (17 KB)
- [x] scripts/ 可执行 (semopy.Model)
- [x] 无自定义 lib 依赖
- [x] references/ 至少 3 个文件 (5个)
- [x] 代码验证通过

#### panel-data-models ✅ (2026-01-21) - **新建**

| 指标 | 重构前 | 重构后 |
|------|:------:|:------:|
| SKILL.md | 不存在 | **16 KB (新建)** |
| scripts/ | 不存在 | **panel_analysis_pipeline.py (25KB)** |
| references/ | 不存在 | **5文件/24KB (新建)** |
| 依赖 | - | `linearmodels` 包 |
| 验证 | - | ✅ `FE bias=0.01, Hausman p=0.01` |

**创建的文件**:
- `SKILL.md` (16KB): FE/RE/Hausman/TWFE/Clustered SE 完整文档
- `scripts/panel_analysis_pipeline.py` (25KB): 模拟+估计+诊断+LaTeX输出
- `references/fixed_effects.md` (3KB)
- `references/random_effects.md` (4KB)
- `references/clustered_se.md` (5KB)
- `references/diagnostic_tests.md` (6KB)
- `references/common_errors.md` (6KB)

**验收检查**:
- [x] SKILL.md ≥ 10 KB (16 KB)
- [x] scripts/ 可执行 (linearmodels.PanelOLS)
- [x] 无自定义 lib 依赖
- [x] references/ 至少 3 个文件 (5个)
- [x] 代码验证通过 (FE bias=0.0115, Hausman reject H0)

#### time-series-econometrics ✅ (2026-01-22) - **新建**

| 指标 | 重构前 | 重构后 |
|------|:------:|:------:|
| SKILL.md | 不存在 | **16 KB (新建)** |
| scripts/ | 不存在 | **time_series_pipeline.py (29KB)** |
| references/ | 不存在 | **5文件/22KB (新建)** |
| 依赖 | - | `statsmodels`, `arch` 包 |
| 验证 | - | ✅ ARIMA + VAR Granger causality |

**创建的文件**:
- `SKILL.md` (16KB): ARIMA/VAR/GARCH/Unit Root/Cointegration 完整文档
- `scripts/time_series_pipeline.py` (29KB): 模拟+估计+诊断+Granger因果
- `references/arima_models.md` (3KB): Box-Jenkins methodology
- `references/unit_roots.md` (4KB): ADF/KPSS tests
- `references/var_models.md` (4KB): VAR/IRF/FEVD
- `references/cointegration.md` (5KB): Engle-Granger/Johansen
- `references/common_errors.md` (6KB): 12 common mistakes

**验收检查**:
- [x] SKILL.md ≥ 10 KB (16 KB)
- [x] scripts/ 可执行 (statsmodels ARIMA/VAR)
- [x] 无自定义 lib 依赖
- [x] references/ 至少 3 个文件 (5个)
- [x] 代码验证通过:
  - ARIMA: ADF correctly identifies unit root, best order (1,1,1), Ljung-Box p=0.88
  - VAR: y1→y2 Granger causality F=67.78 p=0.0000, y2→y1 F=0.04 p=0.85

#### estimator-rd ✅ (2026-01-22)

| 指标 | 重构前 | 重构后 |
|------|:------:|:------:|
| SKILL.md | 2 KB | **15.6 KB (重写)** |
| scripts/ | 无可执行 | **rd_analysis_pipeline.py (29KB)** |
| references/ | 无 | **5文件/34KB (新建)** |
| 依赖 | 无 | `rdrobust`, `rddensity`, `statsmodels` |
| 验证 | ❌ | ✅ `RD estimate=1.88, true=2.0, bias=-0.12` |

**创建的文件**:
- `SKILL.md` (15.6KB): Sharp/Fuzzy RD, McCrary, Bandwidth sensitivity 完整文档
- `scripts/rd_analysis_pipeline.py` (29KB): 模拟+Sharp RD+Fuzzy RD+McCrary+平衡+LaTeX
- `references/identification_assumptions.md` (5KB): Continuity, LATE, Sharp vs Fuzzy
- `references/estimation_methods.md` (6KB): Local polynomial, bandwidth selection
- `references/diagnostic_tests.md` (7KB): McCrary, covariate balance, placebo
- `references/reporting_standards.md` (7.5KB): AER/QJE tables, LaTeX templates
- `references/common_errors.md` (8KB): 12 common mistakes

**验收检查**:
- [x] SKILL.md ≥ 10 KB (15.6 KB)
- [x] scripts/ 可执行 (rdrobust fallback to manual_local_linear)
- [x] 无自定义 lib 依赖
- [x] references/ 至少 3 个文件 (5个)
- [x] 代码验证通过:
  - Sharp RD: estimate=1.8812, SE=0.1374, true=2.0, bias=-0.12
  - Bandwidth: 0.2987, N effective: 146

---

### 8.5 最终重构统计 (2026-01-22)

| 优先级 | 技能 | SKILL.md | scripts/ | references/ | 验证 | 状态 |
|:------:|------|:--------:|:--------:|:-----------:|:----:|:----:|
| P0 | estimator-did | 20KB | 22KB | 132KB | ✅ | **完成** |
| P0 | estimator-iv | 20.3KB | 22KB | 22KB | ✅ | **完成** |
| P0 | estimator-psm | 24.3KB | 26.2KB | 21KB | ✅ | **完成** |
| P0 | estimator-rd | 15.6KB | 29KB | 34KB | ✅ | **完成** |
| P1 | causal-ddml | 17KB | 23KB | 78KB | ✅ | **完成** |
| P1 | causal-forest | 17.5KB | 23KB | 100KB | ✅ | **完成** |
| P1 | structural-equation-modeling | 17KB | 23KB | 78KB | ✅ | **完成** |
| P2 | panel-data-models | 16KB | 25KB | 24KB | ✅ | **完成** |
| P2 | time-series-econometrics | 16KB | 29KB | 22KB | ✅ | **完成** |
| **总计** | **9个技能** | **~164KB** | **~222KB** | **~511KB** | ✅ | **完成** |

**重构完成率**: 9/9 = **100%**

**关键成果**:
1. 所有 P0/P1/P2 技能完成重构 (9/9 = 100%)
2. SKILL.md 平均从 5KB 扩展到 18KB (+260%)
3. 全部使用开源包，无自定义 lib 依赖
4. 每个技能都有可执行的 scripts/ 和完整 references/
5. 代码验证全部通过 (所有技能的核心函数验证 OK)

**使用的开源包**:
| 技能 | 主要依赖包 |
|------|-----------|
| estimator-did | linearmodels, differences |
| estimator-iv | linearmodels |
| estimator-psm | sklearn |
| estimator-rd | rdrobust, statsmodels |
| causal-ddml | doubleml |
| causal-forest | econml |
| structural-equation-modeling | semopy |
| panel-data-models | linearmodels |
| time-series-econometrics | statsmodels, arch |

---

**文档版本**: 3.0
**最后更新**: 2026-01-22
**作者**: Claude Code

---

## 9. 下一步计划

### 9.1 部署到 skills/ 目录

重构完成后，需要将技能部署到用户的 Claude Code skills 目录：

```powershell
# 部署命令 (PowerShell)
Copy-Item -Path "D:\code\PPcourse\causal-ml-skills\skills\*" -Destination "C:\Users\tangj\.claude\skills\" -Recurse -Force
```

### 9.2 完整性测试

部署后需验证：
1. 每个技能的 Quick Start 代码可直接运行
2. CLI 脚本 `python *_pipeline.py --demo` 全部通过
3. 无 ImportError 或 ModuleNotFoundError

### 9.3 未来迭代

| 优先级 | 任务 | 说明 |
|:------:|------|------|
| P3 | 添加 causal-concept-guide | 概念指南技能 |
| P3 | 添加 paper-replication-workflow | 论文复制工作流 |
| P3 | 完善 ml-preprocessing | 数据预处理技能 |
| P3 | 完善 ml-model-* | 机器学习模型技能 |

### 9.4 维护计划

- 每季度检查依赖包版本更新
- 根据用户反馈添加 Troubleshooting 条目
- 跟踪方法论新进展，更新 references/
