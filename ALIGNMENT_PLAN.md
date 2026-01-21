# Causal-ML-Skills 系统对齐改进方案

> 基于 K-Dense-AI/claude-scientific-skills 的最佳实践，针对经管研究优化

## 1. 系统对比分析

### 1.1 架构对比

| 维度 | K-Dense-AI | causal-ml-skills (现状) | 差距评估 |
|------|-----------|------------------------|---------|
| **Skill 结构** | SKILL.md + references/ + scripts/ + assets/ | SKILL.md + *.py | 缺少 references 和 assets |
| **参考文档** | 每技能 5-6 个专题 md | 无 | 严重不足 |
| **脚本工具** | 独立可执行脚本 | 内嵌函数 | 部分满足 |
| **输出模板** | LaTeX/Markdown 专业模板 | 基础表格 | 需要增强 |
| **工作流指导** | 详细决策树和检查清单 | 基础描述 | 需要完善 |
| **最佳实践** | 每技能有专节 | 分散 | 需要整合 |
| **常见错误** | 详细列举和解决方案 | 较少 | 需要补充 |

### 1.2 K-Dense 核心优点（经管研究适用）

1. **统计分析技能 (statistical-analysis)**
   - 检验选择决策树
   - 假设检验完整流程
   - APA 格式报告
   - 贝叶斯分析支持

2. **Statsmodels 技能**
   - 离散选择模型完整覆盖 (Logit/Probit/Ordered/Count)
   - 时间序列分析 (ARIMA/VAR/SARIMAX)
   - 面板数据模型
   - 诊断检验大全

3. **PyMC 贝叶斯建模**
   - 层次模型模板
   - MCMC 诊断工具
   - 模型比较 (LOO/WAIC)
   - 后验预测检验

4. **科学写作技能**
   - IMRAD 结构指导
   - 多种引用格式 (APA/AMA/Chicago)
   - 报告指南 (CONSORT/STROBE/PRISMA)
   - 图表最佳实践

---

## 2. 改进方案

### 2.1 Skill 目录结构升级

**当前结构:**
```
skills/classic-methods/estimator-did/
├── SKILL.md
└── did_estimator.py
```

**升级后结构:**
```
skills/classic-methods/estimator-did/
├── SKILL.md                          # 主技能文件
├── references/                       # 参考文档目录
│   ├── identification_assumptions.md # 识别假设详解
│   ├── estimation_methods.md         # 估计方法详解
│   ├── diagnostics_tests.md          # 诊断检验指南
│   ├── staggered_did.md              # 交错DID专题
│   └── reporting_standards.md        # 结果报告规范
├── scripts/                          # 独立脚本
│   ├── run_did_analysis.py           # 完整分析流程
│   ├── parallel_trends_test.py       # 平行趋势检验
│   └── event_study_plot.py           # 事件研究图
├── assets/                           # 模板资源
│   ├── regression_table_template.tex # LaTeX表格模板
│   └── did_report_template.md        # 分析报告模板
└── did_estimator.py                  # 核心估计器
```

### 2.2 新增技能清单（经管研究核心）

#### Priority 0 - 基础设施增强

| 技能名称 | 类型 | 说明 | 对标 K-Dense |
|---------|------|------|-------------|
| `statistical-analysis` | Tool | 描述性统计+假设检验+效应量 | statistical-analysis |
| `econometric-eda` | Tool | 经济学数据 EDA | exploratory-data-analysis |
| `scientific-writing-econ` | Workflow | 经管论文写作指导 | scientific-writing |

#### Priority 1 - 核心方法增强

| 技能名称 | 类型 | 说明 | 对标 K-Dense |
|---------|------|------|-------------|
| `discrete-choice-models` | Estimator | Logit/Probit/Ordered/Count 模型 | statsmodels/discrete_choice |
| `time-series-econometrics` | Estimator | ARIMA/VAR/协整/GARCH | statsmodels/time_series |
| `panel-data-models` | Estimator | FE/RE/GMM/动态面板 | statsmodels/linear_models |
| `bayesian-econometrics` | Estimator | 贝叶斯推断方法 | pymc |

#### Priority 2 - 研究支持

| 技能名称 | 类型 | 说明 | 对标 K-Dense |
|---------|------|------|-------------|
| `latex-table-generator` | Tool | 出版级表格生成 | scientific-writing |
| `result-visualization` | Tool | 系数图/效应图/诊断图 | matplotlib/seaborn |
| `robustness-framework` | Workflow | 稳健性检验框架 | 新增 |

---

## 3. 详细改进计划

### 3.1 Phase 1: 核心 Skill 升级 (estimator-did)

**目标:** 将 estimator-did 升级为参考级 Skill

**新增 references 文档:**

```markdown
# references/identification_assumptions.md

## 平行趋势假设 (Parallel Trends)

### 定义
在无处理的反事实情况下，处理组和控制组的结果变量应呈现相同的时间趋势。

### 数学表达
E[Y(0)_{it} - Y(0)_{i,t-1} | D_i = 1] = E[Y(0)_{it} - Y(0)_{i,t-1} | D_i = 0]

### 检验方法
1. **可视化检验**: 绘制处理组和控制组的趋势图
2. **事件研究法**: 检验处理前各期的系数是否显著为0
3. **统计检验**: 趋势差异的F检验

### 常见违反情况
1. 选择偏差导致的事前趋势差异
2. 预期效应 (Anticipation Effects)
3. 组别特定的时间趋势

### 补救措施
- 匹配法 (PSM-DID)
- 合成控制法
- 组别特定线性趋势
```

**新增 scripts:**

```python
# scripts/run_did_analysis.py
"""
完整 DID 分析流程脚本

用法:
    python run_did_analysis.py data.csv --outcome y --treatment treated \
        --unit unit_id --time year --treatment_time 2015

输出:
    - did_report.md: 完整分析报告
    - figures/: 所有诊断图表
    - tables/: LaTeX 格式表格
"""
```

### 3.2 Phase 2: 新增统计分析技能

**创建 `skills/foundation/statistical-analysis/`**

```markdown
# SKILL.md 结构

---
name: statistical-analysis
description: 经管研究统计分析。检验选择、假设检验、效应量计算、APA报告。
type: Tool
---

## When to Use
- 描述性统计分析
- 组间比较 (t检验, ANOVA)
- 相关分析
- 回归诊断

## 检验选择决策树

### 比较两组
- 连续+正态 → 独立样本t检验
- 连续+非正态 → Mann-Whitney U
- 配对+正态 → 配对t检验
- 配对+非正态 → Wilcoxon符号秩检验
- 分类变量 → 卡方检验/Fisher精确检验

### 比较多组
- 连续+正态+同方差 → 单因素ANOVA
- 连续+非正态 → Kruskal-Wallis
- 重复测量 → 重复测量ANOVA/Friedman

## 效应量报告

| 检验类型 | 效应量 | 小 | 中 | 大 |
|---------|--------|-----|-----|-----|
| t检验 | Cohen's d | 0.20 | 0.50 | 0.80 |
| ANOVA | η²_p | 0.01 | 0.06 | 0.14 |
| 相关 | r | 0.10 | 0.30 | 0.50 |
| 回归 | R² | 0.02 | 0.13 | 0.26 |

## APA 格式报告模板

### 独立样本t检验
"Treatment group (M = {mean1}, SD = {sd1}) scored significantly
higher than control group (M = {mean2}, SD = {sd2}),
t({df}) = {t_stat}, p = {p_value}, d = {cohens_d},
95% CI [{ci_lower}, {ci_upper}]."
```

### 3.3 Phase 3: 时间序列计量技能

**创建 `skills/classic-methods/time-series-econometrics/`**

```
time-series-econometrics/
├── SKILL.md
├── references/
│   ├── stationarity_tests.md      # 单位根检验 (ADF, PP, KPSS)
│   ├── arima_modeling.md          # ARIMA 建模流程
│   ├── var_analysis.md            # VAR/VECM/Granger因果
│   ├── cointegration.md           # 协整分析
│   └── arch_garch.md              # 波动率建模
├── scripts/
│   ├── unit_root_tests.py
│   ├── var_analysis.py
│   └── forecast_evaluation.py
└── ts_estimator.py
```

### 3.4 Phase 4: 经管论文写作技能

**创建 `skills/workflow/scientific-writing-econ/`**

```markdown
# SKILL.md

## 核心原则
**永远使用完整段落，不要提交要点列表**

## 经管论文结构 (IMRAD)

### Introduction
1. 研究背景与重要性
2. 文献综述与研究空白
3. 研究问题与假设
4. 研究贡献

### Literature Review (经管特色)
- 理论基础
- 实证研究回顾
- 假设发展

### Data & Methodology
- 数据来源与样本
- 变量定义与测量
- 模型设定
- 识别策略

### Results
- 描述性统计
- 基准回归
- 稳健性检验
- 异质性分析
- 机制分析

### Discussion & Conclusion
- 结果解读
- 理论贡献
- 实践启示
- 局限性
- 未来方向

## 回归表格规范

### 标准元素
1. 变量名称（科学符号）
2. 系数估计值
3. 标准误（括号内）
4. 显著性标记 (*, **, ***)
5. 控制变量指示
6. 固定效应指示
7. 样本量
8. R²

### LaTeX 模板
参见 `assets/regression_table.tex`
```

---

## 4. 实施路线图

### Sprint 1 (Week 1-2): 核心升级

| 任务 | 优先级 | 交付物 |
|------|--------|--------|
| 升级 estimator-did 目录结构 | P0 | references/ + scripts/ |
| 添加 5 个 references 文档 | P0 | 详细参考文档 |
| 创建 LaTeX 表格模板 | P1 | assets/templates |
| 添加完整分析脚本 | P1 | scripts/run_did_analysis.py |

### Sprint 2 (Week 3-4): 统计分析

| 任务 | 优先级 | 交付物 |
|------|--------|--------|
| 创建 statistical-analysis 技能 | P0 | 完整技能 |
| 检验选择决策树 | P0 | references/test_selection.md |
| 效应量计算工具 | P1 | scripts/effect_sizes.py |
| APA 报告生成器 | P1 | scripts/apa_reporter.py |

### Sprint 3 (Week 5-6): 时间序列

| 任务 | 优先级 | 交付物 |
|------|--------|--------|
| 创建 time-series-econometrics | P0 | 完整技能 |
| 单位根检验模块 | P0 | unit_root_tests.py |
| VAR 分析模块 | P1 | var_analysis.py |
| 预测评估工具 | P2 | forecast_evaluation.py |

### Sprint 4 (Week 7-8): 写作与报告

| 任务 | 优先级 | 交付物 |
|------|--------|--------|
| 创建 scientific-writing-econ | P0 | 完整技能 |
| IMRAD 结构模板 | P0 | references/imrad_econ.md |
| 回归表格生成器 | P1 | latex_table_generator.py |
| 图表最佳实践 | P1 | references/figure_guidelines.md |

---

## 5. 质量标准

### 5.1 每个 Skill 必须包含

- [ ] SKILL.md 包含标准 YAML frontmatter
- [ ] "When to Use" 清晰定义
- [ ] 至少 3 个 references 文档
- [ ] 至少 1 个可独立运行的脚本
- [ ] 完整的工作流程描述
- [ ] "Best Practices" 和 "Common Pitfalls" 章节
- [ ] 代码示例和预期输出

### 5.2 输出规范

- [ ] 回归表格符合三线表规范
- [ ] 支持 LaTeX 和 Markdown 双格式
- [ ] 自动生成 APA/Chicago 格式引用
- [ ] 图表分辨率 ≥ 300 DPI

### 5.3 文档规范

- [ ] 中英文双语支持
- [ ] 数学公式使用 LaTeX 格式
- [ ] 代码块有语法高亮
- [ ] 交叉引用使用相对链接

---

## 6. 与 K-Dense 系统的关键差异

### 6.1 保留的差异（经管特色）

| 特性 | K-Dense | causal-ml-skills |
|------|---------|------------------|
| 领域焦点 | 生物医学为主 | 经管研究专用 |
| 因果方法深度 | 一般 | 深度覆盖 |
| 计量经济学 | 基础 statsmodels | 完整计量工具箱 |
| R/Stata 支持 | 无 | 有 (grf, rdrobust) |
| 面板数据 | 一般 | 核心支持 |
| 报告指南 | CONSORT/STROBE | 经管顶刊规范 |

### 6.2 借鉴的最佳实践

1. **目录结构**: references/ + scripts/ + assets/
2. **检查清单**: 每步骤有验证项
3. **常见错误**: 详细列举和解决方案
4. **模板系统**: 可复用的输出模板
5. **工作流指导**: 决策树和流程图

---

## 7. 下一步行动

### 立即执行 (Today)

1. [ ] 为 estimator-did 创建 references 目录
2. [ ] 编写 identification_assumptions.md
3. [ ] 创建 LaTeX 表格模板

### 本周完成

1. [ ] 完成 estimator-did 的全部升级
2. [ ] 开始 statistical-analysis 技能开发
3. [ ] 更新 ROADMAP.md 反映新计划

### 本月完成

1. [ ] 完成 Phase 1 和 Phase 2
2. [ ] 所有核心 Estimator 升级目录结构
3. [ ] 发布 v0.2.0 版本
