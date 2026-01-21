# 机器学习因果推断 Skills

> 基于《机器学习因果推断实战全流程》课程大纲构建的 Claude Code Agent Skills 集合

## 项目状态

**开发完成** - 全部 21 个 Skills 已实现，采用 K-Dense 规范结构，测试框架就绪

## 快速开始

```bash
# 1. 运行环境检查
python skills/infrastructure/setup-causal-ml-env/env_check.py

# 2. 生成测试数据
python tests/data/synthetic/dgp_did.py
python tests/data/synthetic/dgp_rd.py
python tests/data/synthetic/dgp_ddml.py
python tests/data/benchmark/generate_benchmarks.py

# 3. 验证 Skill 结构 (新增)
python scripts/validate_skill.py skills/classic-methods/estimator-did

# 4. 运行验证测试
python tests/run_all_tests.py

# 5. 部署到 Claude Code
python deploy.py
```

## 项目结构

```
causal-ml-skills/
├── ARCHITECTURE.md         # 技术架构设计
├── ROADMAP.md              # 开发路线图
├── README.md               # 本文件
├── deploy.py               # 部署脚本
│
├── skills/                 # Skill 源文件 (21 个)
│   ├── infrastructure/     # 基础设施 (2 个)
│   │   ├── setup-causal-ml-env/
│   │   └── scientific-writing-econ/
│   │
│   ├── classic-methods/    # 经典因果方法 (8 个)
│   │   ├── causal-concept-guide/
│   │   ├── estimator-did/
│   │   ├── estimator-rd/
│   │   ├── estimator-iv/
│   │   ├── estimator-psm/
│   │   ├── panel-data-models/
│   │   ├── time-series-econometrics/
│   │   └── discrete-choice-models/
│   │
│   ├── ml-foundation/      # 机器学习基础 (6 个)
│   │   ├── ml-preprocessing/
│   │   ├── ml-model-linear/
│   │   ├── ml-model-tree/
│   │   ├── ml-model-advanced/
│   │   ├── econometric-eda/
│   │   └── statistical-analysis/
│   │
│   └── causal-ml/          # 前沿因果 ML (5 个)
│       ├── causal-ddml/
│       ├── causal-forest/
│       ├── causal-mediation-ml/
│       ├── bayesian-econometrics/
│       └── paper-replication-workflow/
│
├── scripts/                # 开发脚本 (新增)
│   ├── generate_skill_scaffold.py  # 生成 K-Dense 结构脚手架
│   └── validate_skill.py           # 验证 Skill 结构完整性
│
├── assets/                 # 共享资源模板 (新增)
│   ├── latex/              # LaTeX 表格模板
│   │   ├── common_preamble.tex
│   │   ├── regression_table.tex
│   │   ├── summary_stats.tex
│   │   ├── balance_table.tex
│   │   ├── event_study.tex
│   │   ├── event_study_table.tex
│   │   ├── first_stage_table.tex
│   │   ├── heterogeneity_table.tex
│   │   └── coef_plot.tex
│   │
│   └── markdown/           # Markdown 报告模板
│       ├── analysis_report.md
│       ├── data_dictionary.md
│       ├── replication_readme.md
│       └── robustness_appendix.md
│
├── templates/              # Skill 开发模板
│   ├── SKILL-TEMPLATE.md       # 通用 Skill 模板 (K-Dense)
│   └── ESTIMATOR-TEMPLATE.md   # 估计器专用模板
│
├── lib/                    # 共享库 (增强版)
│   └── python/
│       ├── __init__.py
│       ├── data_loader.py      # CausalInput/CausalOutput
│       ├── diagnostics.py      # 诊断函数 (扩展)
│       ├── table_formatter.py  # 表格生成 (扩展)
│       └── visualization.py    # 可视化工具 (新增)
│
├── tests/                  # 测试用例
│   ├── data/
│   │   ├── synthetic/      # 合成数据 (已知真实效应)
│   │   └── benchmark/      # 基准数据集
│   ├── cases/
│   └── run_all_tests.py    # 验证脚本
│
└── docs/                   # 文档
    ├── concepts/
    └── papers/
        └── test_paper_outline.md
```

## Skills 清单

### 全部 Skills (21 个)

#### 基础设施 (2 个)

| Skill | 类型 | 触发词 | 状态 |
|-------|------|--------|------|
| `setup-causal-ml-env` | Tool | 环境配置, python环境, R环境 | 完成 |
| `scientific-writing-econ` | Tool | 论文写作, LaTeX, 学术规范 | 完成 |

#### 经典因果方法 (8 个)

| Skill | 类型 | 触发词 | 状态 |
|-------|------|--------|------|
| `causal-concept-guide` | Knowledge | 因果概念, 方法选择, 识别策略 | 完成 |
| `estimator-did` | Estimator | DID, 双重差分, 平行趋势 | 完成 |
| `estimator-rd` | Estimator | RD, 断点回归, 阈值, 带宽 | 完成 |
| `estimator-iv` | Estimator | IV, 工具变量, 2SLS, 弱工具 | 完成 |
| `estimator-psm` | Estimator | PSM, 倾向得分, 匹配, ATT | 完成 |
| `panel-data-models` | Tool | 面板数据, 固定效应, 随机效应 | 完成 |
| `time-series-econometrics` | Tool | 时间序列, ARIMA, VAR, 协整 | 完成 |
| `discrete-choice-models` | Tool | 离散选择, Logit, Probit, 多项选择 | 完成 |

#### 机器学习基础 (6 个)

| Skill | 类型 | 触发词 | 状态 |
|-------|------|--------|------|
| `ml-preprocessing` | Tool | 缺失值, 异常值, 特征工程 | 完成 |
| `ml-model-linear` | Tool | Lasso, Ridge, 正则化, 弹性网络 | 完成 |
| `ml-model-tree` | Tool | 随机森林, XGBoost, SHAP, 特征重要性 | 完成 |
| `ml-model-advanced` | Tool | SVM, 神经网络, MLP, 集成学习 | 完成 |
| `econometric-eda` | Tool | EDA, 描述统计, 数据质量, 分布检验 | 完成 |
| `statistical-analysis` | Tool | 假设检验, 置信区间, 功效分析 | 完成 |

#### 前沿因果 ML (5 个)

| Skill | 类型 | 触发词 | 状态 |
|-------|------|--------|------|
| `causal-ddml` | Estimator | DDML, 双重机器学习, PLR, IRM | 完成 |
| `causal-forest` | Estimator | 因果森林, CATE, 异质性效应 | 完成 |
| `causal-mediation-ml` | Estimator | 中介分析, ADE, ACME, 机制分析 | 完成 |
| `bayesian-econometrics` | Estimator | 贝叶斯, MCMC, 先验, 后验 | 完成 |
| `paper-replication-workflow` | Workflow | 复现, LaLonde, Card, 验证 | 完成 |

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 主控语言 | Python 3.10+ |
| 因果推断 | econml, doubleml, causalml |
| 计量统计 | statsmodels, linearmodels |
| 机器学习 | scikit-learn, xgboost, lightgbm |
| 贝叶斯推断 | pymc, arviz |
| R 桥接 | rpy2 (grf, mediation, rdrobust) |
| 可视化 | matplotlib, seaborn |

## K-Dense Skill 结构

每个 Skill 采用 K-Dense 规范结构：

```
skill-name/
├── SKILL.md              # 主文档 (必需)
├── references/           # 参考文档目录
│   ├── identification_assumptions.md
│   ├── diagnostic_tests.md
│   ├── estimation_methods.md
│   ├── reporting_standards.md
│   └── common_errors.md
├── scripts/              # 支持脚本
│   └── <skill_name>_estimator.py
└── assets/               # 资源文件
    ├── latex/            # LaTeX 模板
    └── markdown/         # Markdown 模板
```

## 部署

### 完整部署
```bash
python deploy.py
```

### 预览部署
```bash
python deploy.py --dry-run
```

### 部署单个 Skill
```bash
python deploy.py --skill estimator-did
```

### 部署特定分类
```bash
python deploy.py --category classic-methods
```

### 手动部署
```bash
cp -r skills/*/* C:\Users\tangj\.claude\skills\
```

## 验证

### 验证单个 Skill 结构
```bash
python scripts/validate_skill.py skills/classic-methods/estimator-did
```

### 运行所有验证测试
```bash
python tests/run_all_tests.py
```

### 验证标准
- **结构完整**: SKILL.md + references/ 目录存在
- **偏差 < 10%**: 估计值与真实值的偏差百分比
- **覆盖率 > 90%**: 95% CI 包含真值的比例
- **诊断通过**: 所有前置检验通过

## 使用示例

### DID 分析
```python
from skills.classic_methods.estimator_did.did_estimator import run_full_did_analysis

result = run_full_did_analysis(
    data=panel_data,
    outcome='y',
    treatment='treatment_group',
    post_var='post',
    unit_id='unit_id',
    time_id='time'
)
print(result.summary_table)
```

### DDML 高维分析
```python
from skills.causal_ml.causal_ddml.ddml_estimator import run_full_ddml_analysis

result = run_full_ddml_analysis(
    data=high_dim_data,
    outcome='y',
    treatment='d',
    controls=[f'x{i}' for i in range(100)]
)
print(f"ATE: {result.effect:.4f} (SE: {result.se:.4f})")
```

### 论文复现
```python
from skills.causal_ml.paper_replication_workflow.replication_workflow import replicate_lalonde

result = replicate_lalonde()
print(result)
```

## 测试论文

完整的方法比较论文大纲见: `docs/papers/test_paper_outline.md`

**题目**: "机器学习增强的因果推断方法比较研究"

涵盖:
- Monte Carlo 模拟评估
- LaLonde/Card/Lee 经典案例复现
- 方法选择指南

## 开发指南

### 创建新 Skill (推荐方式)

使用脚手架生成器自动创建 K-Dense 结构：

```bash
# 创建估计器类型 Skill
python scripts/generate_skill_scaffold.py \
    --name my-estimator \
    --category classic-methods \
    --type estimator

# 创建工具类型 Skill
python scripts/generate_skill_scaffold.py \
    --name my-tool \
    --category ml-foundation \
    --type tool
```

### 手动创建 Skill

1. 复制模板
   ```bash
   cp templates/ESTIMATOR-TEMPLATE.md skills/<category>/<skill-name>/SKILL.md
   mkdir -p skills/<category>/<skill-name>/references
   mkdir -p skills/<category>/<skill-name>/scripts
   mkdir -p skills/<category>/<skill-name>/assets
   ```

2. 编辑 SKILL.md
   - 修改 frontmatter (name, description)
   - 按模板结构填写内容

3. 添加参考文档到 references/

4. 添加支持文件
   - Python 实现: `scripts/<skill_name>_estimator.py`
   - 测试数据生成器

5. 验证结构
   ```bash
   python scripts/validate_skill.py skills/<category>/<skill-name>
   ```

6. 部署
   ```bash
   python deploy.py --skill <skill-name>
   ```

### 命名规范

- 目录名: `kebab-case` (如 `estimator-did`)
- 主文件: `SKILL.md` (固定)
- Python 文件: `snake_case.py`
- 参考文档: `snake_case.md`

## 共享库

`lib/python/` 提供共享工具函数：

| 模块 | 功能 |
|------|------|
| `data_loader.py` | CausalInput/CausalOutput 数据结构 |
| `diagnostics.py` | 诊断检验函数 (平行趋势、平衡性等) |
| `table_formatter.py` | 出版级表格生成 (LaTeX/Markdown) |
| `visualization.py` | 因果推断可视化 (事件研究图、RD图等) |

## 许可

MIT

---

**开发完成**: 2025-01
**Skills 数量**: 21
**结构规范**: K-Dense
**验证状态**: 通过
