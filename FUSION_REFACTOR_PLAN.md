# causal-ml-skills 融合重构方案

> **目标**: 按照 claude-scientific-skills 设计模式，完成 causal-ml-skills 所有技能的标准化重构

---

## 1. claude-scientific-skills 核心设计模式

### 1.1 SKILL.md 标准结构 (~15-21KB, 500-600行)

```markdown
---
name: {skill-name}
description: {功能描述}. Use when {使用场景}. {与其他技能的区分}.
license: MIT
metadata:
    skill-author: Causal-ML-Skills
---

# {技能名}: {副标题}

## Overview
{1段话概述，说明核心功能和适用领域}

## When to Use This Skill
{明确的使用场景列表，8-10条}

## Quick Start Guide
### Example 1: {场景1}
{完整可运行代码：从import到结果解释}

### Example 2: {场景2}
...

## Core Capabilities
### 1. {能力1}
{详细说明，包括可用方法、When to use、Reference指向}
...

## Best Practices
### Data Preparation
### Model Building
### Inference
### Reporting

## Common Workflows
### Workflow 1: {标准工作流}
{完整步骤列表}
...

## Reference Documentation
{指向 references/*.md 的说明}

## Common Pitfalls to Avoid
{15条常见错误清单}

## Troubleshooting
{5个常见问题+解决方案}

## Getting Help
{官方文档链接}
```

### 1.2 references/ 标准结构 (5-6个文件, 每个10-20KB)

| 文件 | 内容 | 适用技能类型 |
|------|------|-------------|
| `identification_assumptions.md` | 识别假设形式化定义 | 因果推断 |
| `estimation_methods.md` | 估计量公式、算法 | 所有 |
| `diagnostic_tests.md` | 检验统计量、临界值 | 所有 |
| `reporting_standards.md` | LaTeX模板、表格规范 | 所有 |
| `common_errors.md` | 10+错误+正确做法 | 所有 |

### 1.3 scripts/ 标准结构 (1-2个文件, 每个20-30KB)

```python
"""
Complete {analysis_type} pipeline with preprocessing, estimation,
diagnostics, and reporting.
"""

import numpy as np
import pandas as pd
# ... standard imports

def simulate_data(...):
    """Generate simulated data for demonstration."""
    pass

def run_analysis(...):
    """Main analysis function."""
    pass

def print_results(...):
    """Print formatted results."""
    pass

def generate_latex_table(...):
    """Generate publication-ready LaTeX table."""
    pass

def run_full_analysis(...):
    """Complete pipeline: data → analysis → diagnostics → report."""
    pass

if __name__ == "__main__":
    # CLI with argparse
    # Demo mode using simulated data
    pass
```

---

## 2. 当前 causal-ml-skills 状态评估

### 2.1 已重构技能 (符合标准) ✅

| 技能 | SKILL.md | scripts/ | references/ | 状态 |
|------|:--------:|:--------:|:-----------:|:----:|
| estimator-did | 20KB | 25KB | 132KB (7文件) | ✅ 完成 |
| estimator-iv | 20KB | 22KB | 22KB (5文件) | ✅ 完成 |
| estimator-psm | 24KB | 26KB | 21KB (4文件) | ✅ 完成 |
| causal-ddml | 17KB | 23KB | 78KB (6文件) | ✅ 完成 |
| panel-data-models | 16KB | 25KB | 24KB (5文件) | ✅ 完成 |
| time-series-econometrics | 16KB | 29KB | 22KB (5文件) | ✅ 完成 |

### 2.2 待重构技能

| 优先级 | 技能 | SKILL.md | 主要问题 |
|:------:|------|:--------:|---------|
| **P0** | estimator-rd | 2.1KB | 极度缺失，无scripts/references |
| **P1** | causal-forest | 17.5KB | scripts需统一为单一pipeline |
| **P1** | structural-equation-modeling | 14.9KB | scripts需统一 |
| **P2** | bayesian-econometrics | 3.3KB | 极度缺失 |
| **P2** | causal-mediation-ml | 27KB | 需精简并统一scripts |
| **P2** | paper-replication-workflow | 26.8KB | 需精简 |
| **P3** | ml-preprocessing | 1.4KB | 极度缺失 |
| **P3** | econometric-eda | 3.4KB | 需扩展 |
| **P3** | ml-model-linear | 11.7KB | 需扩展至15KB+ |
| **P3** | ml-model-tree | 19KB | 基本符合 |
| **P3** | ml-model-advanced | 15.4KB | 基本符合 |
| **P3** | statistical-analysis | 15.8KB | 基本符合 |
| **P4** | scientific-writing-econ | 5.2KB | 需大幅扩展 |
| **P4** | setup-causal-ml-env | 6.4KB | 需扩展 |

---

## 3. 重构执行计划

### 3.1 P0: 紧急重构 (estimator-rd)

**目标**: 从2.1KB扩展到20KB+，新建scripts/和references/

**任务清单**:
1. 重写 SKILL.md (参考 estimator-did 模板)
   - Overview: RD设计概述
   - When to Use: Sharp/Fuzzy RD触发条件
   - Quick Start: 5个示例 (Sharp RD, Fuzzy RD, Bandwidth Selection, McCrary Test, Covariate Balance)
   - Core Capabilities: Local Polynomial, Bandwidth Selection, Diagnostics
   - Best Practices/Pitfalls/Troubleshooting

2. 创建 scripts/rd_analysis_pipeline.py (~25KB)
   - `simulate_rd_data()`: 模拟RD数据
   - `run_sharp_rd()`: Sharp RD估计 (rdrobust)
   - `run_fuzzy_rd()`: Fuzzy RD估计
   - `bandwidth_selection()`: IK/CCT bandwidth
   - `mccrary_test()`: 密度测试
   - `covariate_balance()`: 协变量平衡
   - `plot_rd()`: RD可视化
   - CLI支持

3. 创建 references/ (5文件)
   - `identification_assumptions.md`: 连续性假设、LATE解释
   - `estimation_methods.md`: Local Poly, Bandwidth, Variance
   - `diagnostic_tests.md`: McCrary, Covariate Balance, Donut
   - `reporting_standards.md`: LaTeX模板
   - `common_errors.md`: 10个常见错误

**依赖包**: `rdrobust`, `rddensity`, `rdlocrand`

### 3.2 P1: 高优先级统一 (causal-forest, SEM)

#### causal-forest
- 合并现有5个scripts为1个 `causal_forest_pipeline.py`
- 依赖: `econml.CausalForestDML`
- 关键函数: `estimate_cate()`, `variable_importance()`, `heterogeneity_analysis()`

#### structural-equation-modeling
- 合并为1个 `sem_analysis_pipeline.py`
- 依赖: `semopy`
- 关键函数: `define_model()`, `fit_sem()`, `model_fit_indices()`, `modification_indices()`

### 3.3 P2: 中等优先级 (bayesian, mediation, replication)

#### bayesian-econometrics
- 从3.3KB扩展到15KB+
- 依赖: `pymc`, `arviz`
- 新增: Quick Start (5个示例), Workflows, Pitfalls

#### causal-mediation-ml
- 精简并统一scripts
- 依赖: `causalml`, `sklearn`

#### paper-replication-workflow
- 精简SKILL.md
- 专注于工作流而非代码实现

### 3.4 P3: ML Foundation 统一

| 技能 | 目标 | 主要变更 |
|------|------|---------|
| ml-preprocessing | 15KB+ | 完全重写SKILL.md |
| econometric-eda | 15KB+ | 扩展SKILL.md |
| ml-model-linear | 保持 | 添加Pitfalls/Troubleshooting |
| ml-model-tree | 保持 | 微调 |
| ml-model-advanced | 保持 | 微调 |
| statistical-analysis | 保持 | 微调 |

### 3.5 P4: Infrastructure

| 技能 | 目标 |
|------|------|
| scientific-writing-econ | 扩展到15KB，参考claude-scientific-writer |
| setup-causal-ml-env | 扩展到10KB |

---

## 4. 验收标准

### 4.1 必须满足

| 标准 | 要求 |
|------|------|
| SKILL.md 大小 | ≥15KB |
| Quick Start | ≥4个完整可运行示例 |
| scripts/ | 1-2个统一pipeline脚本 |
| references/ | ≥4个专题文档 |
| Best Practices | 包含完整章节 |
| Common Pitfalls | ≥10条 |
| 代码可执行性 | 所有示例可直接运行 |

### 4.2 代码验证

```bash
# 每个技能必须通过
cd skills/{category}/{skill-name}/scripts
python {pipeline}.py --demo  # 使用模拟数据验证
```

---

## 5. 融合策略

### 5.1 与 claude-scientific-skills 的关系

```
claude-scientific-skills (139技能)     causal-ml-skills (22技能)
├── statsmodels ◄─────────────────► panel-data-models, time-series
├── scikit-learn ◄────────────────► ml-model-*, ml-preprocessing
├── pytorch-lightning ◄───────────► (新增 deep-learning-causal)
├── statistical-analysis ◄────────► statistical-analysis
└── (无因果推断专门技能) ◄─────────► estimator-*, causal-*
```

### 5.2 命名规范

- **技能名**: 小写连字符 (estimator-did, causal-forest)
- **脚本名**: 下划线 ({topic}_analysis_pipeline.py)
- **references文件**: 下划线 (identification_assumptions.md)

### 5.3 YAML metadata 标准

```yaml
---
name: {skill-name}
description: {完整描述，包含触发词和使用场景}
license: MIT
metadata:
    skill-author: Causal-ML-Skills
    version: "2.0.0"
    category: {econometrics|causal-ml|ml-foundation|infrastructure}
---
```

---

## 6. 执行时间线

| 阶段 | 任务 | 预计工作量 |
|------|------|-----------|
| **Phase 1** | estimator-rd 完整重构 | 1个技能 |
| **Phase 2** | causal-forest, SEM 统一 | 2个技能 |
| **Phase 3** | bayesian, mediation, replication | 3个技能 |
| **Phase 4** | ML Foundation 6个技能 | 6个技能 |
| **Phase 5** | Infrastructure 2个技能 | 2个技能 |
| **Phase 6** | 清理验证 | 全部验证 |

---

## 7. 立即执行: estimator-rd 重构

开始 P0 任务，将 estimator-rd 从 2.1KB 扩展到完整符合标准的技能。
