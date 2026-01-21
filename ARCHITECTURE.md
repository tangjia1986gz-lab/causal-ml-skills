# 机器学习因果推断 Skills 技术架构

## 1. 项目概览

### 1.1 目标
构建一套标准化、可复用的 Claude Code Skills，赋予 Agent 完整的因果推断研究与实战能力。

### 1.2 核心原则
- **模块化**: 每个 Skill 独立可用，组合使用时无缝衔接
- **可验证**: 内置检验步骤，确保因果推断的有效性
- **多语言**: Python 为主控，支持 R/Stata 调用
- **学术规范**: 输出符合经济学/管理学顶刊格式
- **K-Dense 规范**: 统一的 Skill 结构 (SKILL.md + references/ + scripts/ + assets/)

### 1.3 项目规模
- **Skills 总数**: 21 个
- **分类**: 4 大类别 (基础设施、经典方法、ML基础、前沿融合)
- **结构规范**: K-Dense 标准

---

## 2. 目录结构

```
causal-ml-skills/
├── ARCHITECTURE.md          # 本文档
├── ROADMAP.md               # 开发路线图
├── README.md                # 项目说明
├── deploy.py                # 部署脚本
│
├── skills/                  # Skill 源文件 (21 个)
│   ├── infrastructure/      # 基础设施 (2 个)
│   │   ├── setup-causal-ml-env/
│   │   └── scientific-writing-econ/
│   │
│   ├── classic-methods/     # 经典因果方法 (8 个)
│   │   ├── causal-concept-guide/
│   │   ├── estimator-did/
│   │   ├── estimator-rd/
│   │   ├── estimator-iv/
│   │   ├── estimator-psm/
│   │   ├── panel-data-models/
│   │   ├── time-series-econometrics/
│   │   └── discrete-choice-models/
│   │
│   ├── ml-foundation/       # 机器学习基础 (6 个)
│   │   ├── ml-preprocessing/
│   │   ├── ml-model-linear/
│   │   ├── ml-model-tree/
│   │   ├── ml-model-advanced/
│   │   ├── econometric-eda/
│   │   └── statistical-analysis/
│   │
│   └── causal-ml/           # 前沿因果ML (5 个)
│       ├── causal-ddml/
│       ├── causal-forest/
│       ├── causal-mediation-ml/
│       ├── bayesian-econometrics/
│       └── paper-replication-workflow/
│
├── scripts/                 # 开发脚本
│   ├── generate_skill_scaffold.py  # K-Dense 脚手架生成器
│   └── validate_skill.py           # Skill 结构验证器
│
├── assets/                  # 共享资源模板
│   ├── latex/               # LaTeX 表格模板
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
│   ├── markdown/            # Markdown 报告模板
│   │   ├── analysis_report.md
│   │   ├── data_dictionary.md
│   │   ├── replication_readme.md
│   │   └── robustness_appendix.md
│   │
│   └── README.md            # 资源使用说明
│
├── templates/               # Skill 开发模板
│   ├── SKILL-TEMPLATE.md    # 通用 Skill 模板 (K-Dense)
│   └── ESTIMATOR-TEMPLATE.md # 估计器专用模板
│
├── lib/                     # 共享库 (增强版)
│   └── python/
│       ├── __init__.py
│       ├── data_loader.py       # CausalInput/CausalOutput 数据结构
│       ├── diagnostics.py       # 诊断检验函数 (扩展)
│       ├── table_formatter.py   # 表格生成 (LaTeX/Markdown)
│       └── visualization.py     # 可视化工具 (新增)
│
├── tests/                   # 测试用例
│   ├── data/
│   │   ├── synthetic/       # 合成数据 (已知真实效应)
│   │   └── benchmark/       # 基准数据集
│   ├── cases/
│   └── run_all_tests.py
│
└── docs/                    # 文档
    ├── concepts/
    └── papers/
        └── test_paper_outline.md
```

---

## 3. K-Dense Skill 结构规范

### 3.1 标准 Skill 目录结构

每个 Skill 必须遵循 K-Dense 规范：

```
skill-name/
├── SKILL.md                 # 主文档 (必需)
│
├── references/              # 参考文档目录 (必需)
│   ├── identification_assumptions.md  # 识别假设
│   ├── diagnostic_tests.md           # 诊断检验
│   ├── estimation_methods.md         # 估计方法
│   ├── reporting_standards.md        # 报告标准
│   └── common_errors.md              # 常见错误
│
├── scripts/                 # 支持脚本 (可选)
│   └── <skill_name>_estimator.py
│
└── assets/                  # 资源文件 (可选)
    ├── latex/               # LaTeX 模板
    └── markdown/            # Markdown 模板
```

### 3.2 references/ 目录文件说明

| 文件 | 用途 | 估计器必需 | 工具必需 |
|------|------|-----------|---------|
| `identification_assumptions.md` | 识别假设详解 | 是 | 否 |
| `diagnostic_tests.md` | 诊断检验方法 | 是 | 视情况 |
| `estimation_methods.md` | 估计方法详解 | 是 | 视情况 |
| `reporting_standards.md` | 报告输出标准 | 是 | 是 |
| `common_errors.md` | 常见错误与修正 | 是 | 是 |

### 3.3 验证 Skill 结构

使用验证脚本检查结构完整性：

```bash
python scripts/validate_skill.py skills/classic-methods/estimator-did
```

---

## 4. Skill 分类体系

### 4.1 类型定义

| 类型 | 代号 | 说明 | 数量 | 示例 |
|------|------|------|------|------|
| **Knowledge** | K | 概念解释、方法论指导 | 1 | `causal-concept-guide` |
| **Estimator** | E | 因果效应估计器 | 10 | `estimator-did`, `causal-ddml` |
| **Tool** | T | 数据处理、模型训练工具 | 9 | `ml-preprocessing`, `econometric-eda` |
| **Workflow** | W | 端到端流程编排 | 1 | `paper-replication-workflow` |

### 4.2 完整 Skills 清单

#### 基础设施 (infrastructure/) - 2 个
| Skill | 类型 | 功能 |
|-------|------|------|
| `setup-causal-ml-env` | Tool | Python/R 环境配置与验证 |
| `scientific-writing-econ` | Tool | 学术写作与 LaTeX 输出 |

#### 经典因果方法 (classic-methods/) - 8 个
| Skill | 类型 | 功能 |
|-------|------|------|
| `causal-concept-guide` | Knowledge | 因果概念与方法选择指南 |
| `estimator-did` | Estimator | 双重差分估计 |
| `estimator-rd` | Estimator | 断点回归估计 |
| `estimator-iv` | Estimator | 工具变量估计 |
| `estimator-psm` | Estimator | 倾向得分匹配 |
| `panel-data-models` | Tool | 面板数据模型 (FE/RE) |
| `time-series-econometrics` | Tool | 时间序列分析 |
| `discrete-choice-models` | Tool | 离散选择模型 |

#### 机器学习基础 (ml-foundation/) - 6 个
| Skill | 类型 | 功能 |
|-------|------|------|
| `ml-preprocessing` | Tool | 数据预处理与特征工程 |
| `ml-model-linear` | Tool | 线性模型与正则化 |
| `ml-model-tree` | Tool | 树模型与集成学习 |
| `ml-model-advanced` | Tool | 高级 ML 模型 (SVM/NN) |
| `econometric-eda` | Tool | 计量经济学 EDA |
| `statistical-analysis` | Tool | 统计分析与假设检验 |

#### 前沿因果 ML (causal-ml/) - 5 个
| Skill | 类型 | 功能 |
|-------|------|------|
| `causal-ddml` | Estimator | 双重/去偏机器学习 |
| `causal-forest` | Estimator | 因果森林与 CATE |
| `causal-mediation-ml` | Estimator | ML 增强中介分析 |
| `bayesian-econometrics` | Estimator | 贝叶斯因果推断 |
| `paper-replication-workflow` | Workflow | 论文复现工作流 |

### 4.3 依赖关系图

```
                         ┌─────────────────────┐
                         │ setup-causal-ml-env │
                         └──────────┬──────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  Classic Methods  │    │   ML Foundation   │    │    Causal ML      │
├───────────────────┤    ├───────────────────┤    ├───────────────────┤
│ • DID             │    │ • Preprocessing   │    │ • DDML            │
│ • RD              │    │ • Linear Models   │    │ • Causal Forest   │
│ • IV              │    │ • Tree Models     │    │ • Mediation ML    │
│ • PSM             │    │ • Advanced ML     │    │ • Bayesian        │
│ • Panel Data      │    │ • EDA             │    │                   │
│ • Time Series     │    │ • Statistics      │    │                   │
│ • Discrete Choice │    │                   │    │                   │
└─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  paper-replication-workflow  │
                    └──────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   scientific-writing-econ    │
                    └──────────────────────────────┘
```

---

## 5. 技术栈

### 5.1 核心语言分工

| 任务 | 语言 | 工具/库 | 原因 |
|------|------|---------|------|
| 主控逻辑 | Python | - | 最广泛的生态系统 |
| 经典计量 | Python | `statsmodels`, `linearmodels` | 成熟稳定 |
| 双重机器学习 | Python | `econml`, `doubleml` | 微软/官方实现 |
| 贝叶斯推断 | Python | `pymc`, `arviz` | 现代贝叶斯工具 |
| 因果森林 | R | `grf` | 最权威实现 |
| 高级计量 | Stata | `pystata` | 某些检验的金标准 |

### 5.2 Python 核心依赖

```python
# 因果推断核心
econml >= 0.15.0       # Microsoft EconML
doubleml >= 0.7.0      # Double ML
causalml >= 0.15.0     # Uber CausalML

# 贝叶斯推断
pymc >= 5.0.0          # Bayesian modeling
arviz >= 0.15.0        # Bayesian visualization

# 统计计量
statsmodels >= 0.14.0
linearmodels >= 5.0    # Panel data

# 机器学习
scikit-learn >= 1.3.0
xgboost >= 2.0.0
lightgbm >= 4.0.0

# R 桥接
rpy2 >= 3.5.0

# 数据处理
pandas >= 2.0.0
numpy >= 1.24.0

# 可视化
matplotlib >= 3.7.0
seaborn >= 0.12.0
```

### 5.3 R 核心依赖

```r
# 因果森林
grf >= 2.3.0

# 因果中介
mediation >= 4.5.0

# 断点回归
rdrobust >= 2.1.0
rddensity >= 2.4.0
```

---

## 6. 共享库 (lib/python/)

### 6.1 模块概览

| 模块 | 功能 | 主要类/函数 |
|------|------|------------|
| `data_loader.py` | 数据结构定义 | `CausalInput`, `CausalOutput` |
| `diagnostics.py` | 诊断检验函数 | 平行趋势、平衡性、McCrary |
| `table_formatter.py` | 表格生成 | LaTeX/Markdown 回归表 |
| `visualization.py` | 可视化工具 | 事件研究图、RD 图、森林图 |

### 6.2 diagnostics.py 扩展功能

```python
# 通用诊断
parallel_trends_test()      # DID 平行趋势检验
mccrary_density_test()      # RD 密度检验
balance_test()              # PSM 平衡性检验
weak_iv_test()              # IV 弱工具变量检验

# 模型诊断
heteroskedasticity_test()   # 异方差检验
serial_correlation_test()   # 序列相关检验
hausman_test()              # Hausman 检验

# 敏感性分析
placebo_test()              # 安慰剂检验
sensitivity_analysis()      # 敏感性分析
```

### 6.3 table_formatter.py 扩展功能

```python
# 表格生成
regression_table()          # 回归结果表
summary_statistics()        # 描述统计表
balance_table()             # 平衡性检验表
first_stage_table()         # 第一阶段结果表
event_study_table()         # 事件研究表
heterogeneity_table()       # 异质性分析表

# 输出格式
to_latex()                  # LaTeX 输出
to_markdown()               # Markdown 输出
to_html()                   # HTML 输出
```

### 6.4 visualization.py 功能

```python
# 因果推断可视化
event_study_plot()          # 事件研究图
rd_plot()                   # 断点回归图
coef_plot()                 # 系数图
propensity_distribution()   # 倾向得分分布

# 诊断可视化
parallel_trends_plot()      # 平行趋势图
balance_plot()              # 平衡性图
sensitivity_plot()          # 敏感性分析图

# ML 可视化
feature_importance_plot()   # 特征重要性图
shap_summary_plot()         # SHAP 图
cate_distribution_plot()    # CATE 分布图
```

---

## 7. Skill 接口规范

### 7.1 统一输入格式

```python
@dataclass
class CausalInput:
    """因果推断 Skill 标准输入"""

    # 数据
    data: pd.DataFrame           # 面板或截面数据

    # 核心变量
    outcome: str                 # Y: 结果变量
    treatment: str               # D: 处理变量

    # 控制变量
    controls: List[str] = None   # X: 控制变量列表

    # 面板结构 (可选)
    unit_id: str = None          # 个体标识
    time_id: str = None          # 时间标识

    # 方法特定参数
    params: Dict[str, Any] = None
```

### 7.2 统一输出格式

```python
@dataclass
class CausalOutput:
    """因果推断 Skill 标准输出"""

    # 核心估计
    effect: float                # 点估计
    se: float                    # 标准误
    ci_lower: float              # 置信区间下界
    ci_upper: float              # 置信区间上界
    p_value: float               # p 值

    # 诊断检验
    diagnostics: Dict[str, Any]  # 各类检验结果

    # 可视化
    figures: List[Figure] = None # 图表对象

    # 报告
    summary_table: str           # LaTeX/Markdown 表格
    interpretation: str          # 结果解读
```

---

## 8. 质量保障

### 8.1 每个 Estimator Skill 必须包含

1. **识别假设检验** (Identification Checks)
   - 平行趋势检验 (DID)
   - 断点有效性检验 (RD)
   - 弱工具变量检验 (IV)
   - 平衡性检验 (PSM)

2. **稳健性检验** (Robustness Checks)
   - 安慰剂检验
   - 敏感性分析
   - 替代模型设定

3. **输出标准化**
   - 出版级回归表格
   - 诊断图表
   - 结果解读文本

### 8.2 代码规范

- 类型注解: 所有公开函数必须有类型注解
- 文档字符串: NumPy 风格
- 测试覆盖: 核心逻辑 > 80%

---

## 9. 部署策略

### 9.1 开发阶段
Skills 在 `D:\code\PPcourse\causal-ml-skills\skills\` 下开发。

### 9.2 部署阶段
完成测试后，复制到 `C:\Users\tangj\.claude\skills\` 进行部署：

```bash
# 部署全部
python deploy.py

# 预览部署
python deploy.py --dry-run

# 部署单个 Skill
python deploy.py --skill estimator-did

# 部署特定分类
python deploy.py --category classic-methods

# 手动部署
cp -r skills/*/* C:\Users\tangj\.claude\skills\
```

### 9.3 命名规范
- 目录名: `kebab-case` (如 `causal-ddml`)
- 主文件: `SKILL.md`
- 参考文件: `snake_case.md`
- Python 文件: `snake_case.py`

---

## 10. 开发脚本

### 10.1 generate_skill_scaffold.py

生成符合 K-Dense 规范的 Skill 脚手架：

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

生成结构：
```
my-estimator/
├── SKILL.md
├── references/
│   ├── identification_assumptions.md
│   ├── diagnostic_tests.md
│   ├── estimation_methods.md
│   ├── reporting_standards.md
│   └── common_errors.md
├── scripts/
│   └── my_estimator.py
└── assets/
```

### 10.2 validate_skill.py

验证 Skill 结构完整性：

```bash
# 验证单个 Skill
python scripts/validate_skill.py skills/classic-methods/estimator-did

# 验证所有 Skills
python scripts/validate_skill.py --all
```

验证内容：
- SKILL.md 存在且格式正确
- references/ 目录存在
- 必需文件存在
- frontmatter 格式正确

---

## 11. 版本管理

- 使用 Git 管理 `causal-ml-skills` 目录
- 每个 Skill 的 SKILL.md 头部标注版本
- 重大更新记录在 CHANGELOG.md

---

## 12. 更新日志

### v2.0.0 (2025-01)
- Skills 数量从 14 扩展到 21
- 采用 K-Dense 结构规范
- 新增 scripts/ 开发脚本
- 新增 assets/ 共享资源模板
- 扩展 lib/python/ 共享库
- 新增 visualization.py 可视化模块
