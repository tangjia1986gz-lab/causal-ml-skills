# 机器学习因果推断 Skills

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/tangjia1986gz-lab/causal-ml-skills)
[![Skills](https://img.shields.io/badge/skills-21-green.svg)](#skills-清单)
[![K-Dense](https://img.shields.io/badge/structure-K--Dense-orange.svg)](#k-dense-skill-结构)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> 基于《机器学习因果推断实战全流程》课程大纲构建的 Claude Code Agent Skills 集合

## 项目概览

| 指标 | 数值 |
|------|------|
| **Skills 总数** | 21 |
| **参考文档** | 102 |
| **CLI 脚本** | 72 |
| **LaTeX 模板** | 9 |
| **Markdown 模板** | 9 |
| **共享库函数** | 40+ |

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/tangjia1986gz-lab/causal-ml-skills.git
cd causal-ml-skills
```

### 2. 环境检查

```bash
python skills/infrastructure/setup-causal-ml-env/env_check.py
```

### 3. 部署到 Claude Code

```bash
# 完整部署 (推荐)
python deploy.py --batch --backup --manifest --version 2.0.0

# 预览部署
python deploy.py --dry-run

# 部署特定分类
python deploy.py --category classic-methods --validate
```

### 4. 验证部署

```bash
# 验证所有 Skills
python scripts/validate_skill.py --all

# 验证单个 Skill
python scripts/validate_skill.py skills/classic-methods/estimator-did
```

## 项目结构

```
causal-ml-skills/
├── skills/                     # 21 个 Skills
│   ├── infrastructure/         # 基础设施 (2)
│   ├── classic-methods/        # 经典因果方法 (8)
│   ├── ml-foundation/          # 机器学习基础 (6)
│   └── causal-ml/              # 前沿因果 ML (5)
│
├── lib/python/                 # 共享库
│   ├── data_loader.py          # CausalInput/CausalOutput 数据结构
│   ├── diagnostics.py          # 诊断检验 (Hausman, VIF, ADF...)
│   ├── table_formatter.py      # 出版级表格 (LaTeX/Markdown/HTML)
│   └── visualization.py        # 因果推断可视化
│
├── assets/                     # 共享资源模板
│   ├── latex/                  # 9 个 LaTeX 模板
│   └── markdown/               # 9 个 Markdown 模板
│
├── scripts/                    # 开发工具
│   ├── generate_skill_scaffold.py  # K-Dense 脚手架生成器
│   └── validate_skill.py           # 质量验证器
│
├── templates/                  # Skill 开发模板
├── tests/                      # 测试用例
└── docs/                       # 文档
```

## Skills 清单

### 基础设施 (2)

| Skill | 类型 | 触发词 |
|-------|------|--------|
| `setup-causal-ml-env` | Tool | 环境配置, python环境, R环境 |
| `scientific-writing-econ` | Tool | 论文写作, AER style, LaTeX |

### 经典因果方法 (8)

| Skill | 类型 | 触发词 |
|-------|------|--------|
| `causal-concept-guide` | Knowledge | 因果概念, 方法选择, DAG |
| `estimator-did` | Estimator | DID, 双重差分, 平行趋势, Callaway-Sant'Anna |
| `estimator-rd` | Estimator | RD, 断点回归, McCrary, 带宽选择 |
| `estimator-iv` | Estimator | IV, 工具变量, 2SLS, Stock-Yogo |
| `estimator-psm` | Estimator | PSM, 倾向得分, 匹配, Rosenbaum bounds |
| `panel-data-models` | Tool | 面板数据, 固定效应, Hausman, 聚类标准误 |
| `time-series-econometrics` | Tool | 时间序列, ARIMA, VAR, 协整, Granger |
| `discrete-choice-models` | Tool | Logit, Probit, 多项选择, 计数模型 |

### 机器学习基础 (6)

| Skill | 类型 | 触发词 |
|-------|------|--------|
| `ml-preprocessing` | Tool | 缺失值, 异常值, 特征工程, bad controls |
| `ml-model-linear` | Tool | Lasso, Ridge, 弹性网络, post-double-selection |
| `ml-model-tree` | Tool | 随机森林, XGBoost, SHAP, cross-fitting |
| `ml-model-advanced` | Tool | SVM, 神经网络, DragonNet, CEVAE |
| `econometric-eda` | Tool | EDA, 描述统计, 数据质量, Little's MCAR |
| `statistical-analysis` | Tool | 假设检验, 效应量, 功效分析, Cohen's d |

### 前沿因果 ML (5)

| Skill | 类型 | 触发词 |
|-------|------|--------|
| `causal-ddml` | Estimator | DDML, 双重机器学习, Neyman orthogonality |
| `causal-forest` | Estimator | 因果森林, CATE, GATES, policy learning |
| `causal-mediation-ml` | Estimator | 中介分析, ADE, ACME, Imai sensitivity |
| `bayesian-econometrics` | Estimator | 贝叶斯, MCMC, PyMC, 先验敏感性 |
| `paper-replication-workflow` | Workflow | 复现, AEA Data Editor, LaLonde |

## K-Dense Skill 结构

每个 Skill 采用 K-Dense 规范结构：

```
skill-name/
├── SKILL.md                    # 主文档 (YAML frontmatter)
├── *_estimator.py              # Python 实现
├── references/                 # 参考文档 (5-6 个)
│   ├── identification_assumptions.md
│   ├── diagnostic_tests.md
│   ├── estimation_methods.md
│   ├── reporting_standards.md
│   └── common_errors.md
├── scripts/                    # CLI 脚本 (3-5 个)
│   ├── run_analysis.py
│   ├── test_assumptions.py
│   └── visualize_results.py
└── assets/                     # 模板资源
    ├── latex/
    └── markdown/
```

## 部署选项

```bash
# 完整部署 (带备份和清单)
python deploy.py --batch --backup --manifest --version 2.0.0

# 预览部署
python deploy.py --dry-run

# 部署单个 Skill
python deploy.py --skill estimator-did

# 部署特定分类
python deploy.py --category classic-methods

# 带验证部署
python deploy.py --batch --validate

# 回滚到备份
python deploy.py --skill estimator-did --rollback

# 查看部署清单
python deploy.py --show-manifest

# JSON 输出 (CI/CD)
python deploy.py --batch --json
```

## 使用示例

### DID 分析

```python
from lib.python.data_loader import CausalInput
from skills.classic_methods.estimator_did.did_estimator import run_full_did_analysis

# 准备数据
causal_input = CausalInput(
    data=panel_data,
    outcome='y',
    treatment='treated',
    unit_id='firm_id',
    time_id='year'
)

# 运行分析
result = run_full_did_analysis(causal_input)
print(result.summary_table)
print(f"ATT: {result.effect:.4f} (SE: {result.se:.4f})")
```

### DDML 高维分析

```python
from skills.causal_ml.causal_ddml.ddml_estimator import run_full_ddml_analysis

result = run_full_ddml_analysis(
    data=high_dim_data,
    outcome='y',
    treatment='d',
    controls=[f'x{i}' for i in range(100)],
    ml_method='random_forest'
)
print(f"ATE: {result.effect:.4f}, 95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

### CLI 脚本使用

```bash
# DID 分析
python skills/classic-methods/estimator-did/scripts/run_did_analysis.py \
    --data panel.csv --outcome y --treatment d --unit id --time t

# 平行趋势检验
python skills/classic-methods/estimator-did/scripts/test_parallel_trends.py \
    --data panel.csv --method event_study --rambachan-roth

# IV 弱工具检验
python skills/classic-methods/estimator-iv/scripts/weak_iv_robust.py \
    --data iv_data.csv --outcome y --treatment d --instrument z
```

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 主控语言 | Python 3.10+ |
| 因果推断 | econml, doubleml, causalml |
| 计量统计 | statsmodels, linearmodels |
| 机器学习 | scikit-learn, xgboost, lightgbm |
| 贝叶斯推断 | pymc, arviz |
| R 桥接 | rpy2 (grf, rdrobust, did) |
| 可视化 | matplotlib, seaborn |

## 共享库 API

### data_loader.py

```python
from lib.python.data_loader import CausalInput, CausalOutput

# 支持字段
CausalInput(
    data, outcome, treatment, controls,
    unit_id, time_id,           # 面板数据
    panel_type,                 # 'balanced', 'unbalanced', 'staggered'
    cluster_var, weights,       # 聚类和权重
    instrument, running_var,    # IV 和 RD
    cutoff, mediator            # RD 和中介
)
```

### diagnostics.py

```python
from lib.python.diagnostics import (
    parallel_trends_test,    # DID 平行趋势
    mccrary_density_test,    # RD 密度检验
    weak_iv_test,            # IV 弱工具
    balance_test,            # PSM 平衡性
    hausman_test,            # 面板 FE vs RE
    vif_calculation,         # 多重共线性
    adf_test,                # 单位根检验
    cointegration_test       # 协整检验
)
```

### visualization.py

```python
from lib.python.visualization import (
    plot_event_study,        # 事件研究图
    plot_rd,                 # RD 图
    plot_propensity_overlap, # 倾向得分重叠
    plot_cate_heterogeneity, # CATE 异质性
    plot_coef_comparison     # 系数比较森林图
)
```

## 开发指南

### 创建新 Skill

```bash
# 使用脚手架生成器
python scripts/generate_skill_scaffold.py \
    --name my-estimator \
    --category classic-methods \
    --type estimator

# 验证结构
python scripts/validate_skill.py skills/classic-methods/my-estimator

# 部署测试
python deploy.py --skill my-estimator --dry-run
```

### 质量验证

```bash
# 验证所有 Skills
python scripts/validate_skill.py --all

# 详细输出
python scripts/validate_skill.py --all --verbose

# JSON 输出 (CI/CD)
python scripts/validate_skill.py --all --json
```

## 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献流程

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/new-skill`)
3. 使用脚手架创建 Skill
4. 通过验证检查 (`python scripts/validate_skill.py`)
5. 提交更改 (`git commit -m 'Add new skill'`)
6. 推送分支 (`git push origin feature/new-skill`)
7. 创建 Pull Request

## 许可

MIT License

---

**版本**: 2.0.0
**更新日期**: 2026-01-21
**Skills 数量**: 21
**结构规范**: K-Dense
**GitHub**: [tangjia1986gz-lab/causal-ml-skills](https://github.com/tangjia1986gz-lab/causal-ml-skills)
