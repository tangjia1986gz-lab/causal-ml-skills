# Causal ML Skills for Claude Code

[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/tangjia1986gz-lab/causal-ml-skills)
[![Skills](https://img.shields.io/badge/skills-9-green.svg)](#skills-清单)
[![Self-Contained](https://img.shields.io/badge/design-self--contained-orange.svg)](#设计理念)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> 因果推断与计量经济学方法的 Claude Code Skills 集合，基于 K-Dense 规范，全部使用开源包实现

## 项目概览

| 指标 | 数值 |
|------|------|
| **Skills 总数** | 9 |
| **SKILL.md 总计** | ~164 KB |
| **Pipeline 脚本** | ~222 KB |
| **参考文档** | ~511 KB |
| **代码验证** | 100% 通过 |

## 设计理念

### 自包含 (Self-Contained)

每个 Skill 完全独立，仅依赖 pip 安装的开源包：

| 方法 | 依赖包 |
|------|--------|
| DID / IV / Panel | `linearmodels` |
| PSM | `sklearn` |
| RD | `rdrobust`, `statsmodels` |
| DDML | `doubleml` |
| Causal Forest | `econml` |
| SEM | `semopy` |
| Time Series | `statsmodels`, `arch` |

**无自定义 lib 依赖** - 所有代码可直接复制运行。

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/tangjia1986gz-lab/causal-ml-skills.git
cd causal-ml-skills
```

### 2. 安装依赖

```bash
pip install linearmodels doubleml econml semopy rdrobust arch
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn
```

### 3. 部署到 Claude Code

```powershell
# PowerShell (Windows)
Copy-Item -Path "skills\*" -Destination "$env:USERPROFILE\.claude\skills\" -Recurse -Force

# Bash (Linux/Mac)
cp -r skills/* ~/.claude/skills/
```

### 4. 验证安装

```bash
# 运行 DID 示例
cd skills/econometrics/estimator-did/scripts
python did_analysis_pipeline.py --demo

# 运行 DDML 示例
cd skills/causal-ml/causal-ddml/scripts
python ddml_analysis_pipeline.py --demo
```

## 项目结构

```
causal-ml-skills/
├── skills/
│   ├── econometrics/              # 计量经济学方法 (6)
│   │   ├── estimator-did/         # 双重差分
│   │   ├── estimator-iv/          # 工具变量
│   │   ├── estimator-psm/         # 倾向得分匹配
│   │   ├── estimator-rd/          # 断点回归
│   │   ├── panel-data-models/     # 面板数据
│   │   └── time-series-econometrics/  # 时间序列
│   │
│   └── causal-ml/                 # 因果机器学习 (3)
│       ├── causal-ddml/           # 双重机器学习
│       ├── causal-forest/         # 因果森林
│       └── structural-equation-modeling/  # 结构方程
│
├── SKILL_REVISION_PLAN.md         # 重构方案文档
└── README.md
```

## Skills 清单

### 计量经济学方法 (6)

| Skill | 触发词 | 依赖包 | 验证结果 |
|-------|--------|--------|----------|
| `estimator-did` | DID, 双重差分, 平行趋势, TWFE | linearmodels | TWFE coef=2.18, p<0.001 |
| `estimator-iv` | IV, 工具变量, 2SLS, Stock-Yogo | linearmodels | F=197.5, 2SLS bias=-0.04 |
| `estimator-psm` | PSM, 倾向得分, 匹配, IPW, AIPW | sklearn | ATT=1.91, IPW=1.98 |
| `estimator-rd` | RD, 断点回归, McCrary, 带宽 | rdrobust | RD=1.88, true=2.0 |
| `panel-data-models` | 面板, 固定效应, Hausman, 聚类SE | linearmodels | FE bias=0.01, Hausman OK |
| `time-series-econometrics` | ARIMA, VAR, 协整, Granger | statsmodels, arch | Granger F=67.8 |

### 因果机器学习 (3)

| Skill | 触发词 | 依赖包 | 验证结果 |
|-------|--------|--------|----------|
| `causal-ddml` | DDML, 双重ML, PLR, IRM | doubleml | PLR=0.40, CI covers true |
| `causal-forest` | 因果森林, CATE, 异质性 | econml | ATE=1.18, VI正常 |
| `structural-equation-modeling` | SEM, 潜变量, CFI, RMSEA | semopy | CFI=0.999, RMSEA=0.008 |

## Skill 结构

每个 Skill 采用统一结构：

```
skill-name/
├── SKILL.md                       # 主文档 (~15-25 KB)
│   ├── Overview                   # 概述
│   ├── When to Use                # 使用场景
│   ├── Quick Start                # 5个可运行示例
│   ├── Core Capabilities          # 核心功能
│   ├── Common Workflows           # 工作流
│   ├── Best Practices             # 最佳实践
│   ├── Common Pitfalls            # 常见陷阱
│   └── Troubleshooting            # 问题排查
│
├── scripts/
│   └── *_analysis_pipeline.py     # 自包含分析脚本 (~25 KB)
│       ├── simulate_*_data()      # 模拟数据
│       ├── run_*()                # 核心估计
│       ├── diagnostics()          # 诊断检验
│       ├── generate_latex_table() # LaTeX输出
│       └── run_full_analysis()    # 完整流水线 + CLI
│
└── references/                    # 参考文档 (5-6个)
    ├── identification_assumptions.md
    ├── estimation_methods.md
    ├── diagnostic_tests.md
    ├── reporting_standards.md
    └── common_errors.md
```

## 使用示例

### DID 分析

```python
from linearmodels.panel import PanelOLS
import pandas as pd

# 加载数据
df = pd.read_csv('panel_data.csv')
df = df.set_index(['firm_id', 'year'])

# TWFE 回归
model = PanelOLS.from_formula(
    'outcome ~ treatment + EntityEffects + TimeEffects',
    data=df
)
result = model.fit(cov_type='clustered', cluster_entity=True)
print(result.summary)
```

### DDML 分析

```python
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor

# 准备数据
dml_data = DoubleMLData(
    df,
    y_col='outcome',
    d_cols='treatment',
    x_cols=['x1', 'x2', 'x3']
)

# PLR 模型
model = DoubleMLPLR(
    dml_data,
    ml_l=RandomForestRegressor(n_estimators=100),
    ml_m=RandomForestRegressor(n_estimators=100),
    n_folds=5
)
model.fit()
print(model.summary)
```

### CLI 使用

```bash
# DID 分析 (模拟数据演示)
python skills/econometrics/estimator-did/scripts/did_analysis_pipeline.py --demo

# DID 分析 (真实数据)
python skills/econometrics/estimator-did/scripts/did_analysis_pipeline.py \
    --data panel.csv --outcome y --treatment d --unit firm --time year

# IV 分析
python skills/econometrics/estimator-iv/scripts/iv_analysis_pipeline.py --demo

# DDML 分析
python skills/causal-ml/causal-ddml/scripts/ddml_analysis_pipeline.py \
    --data high_dim.csv --outcome y --treatment d --model plr
```

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 语言 | Python 3.10+ |
| 面板/IV | linearmodels |
| 双重机器学习 | doubleml |
| 因果森林 | econml |
| 结构方程 | semopy |
| 断点回归 | rdrobust |
| 时间序列 | statsmodels, arch |
| 机器学习 | scikit-learn |
| 可视化 | matplotlib, seaborn |

## 重构记录

### v3.0.0 (2026-01-22)

**重大更新**: 全面重构为自包含设计

- 删除对 `lib/python/` 的依赖
- 所有 Skills 改用开源包实现
- SKILL.md 平均扩展 260% (5KB → 18KB)
- 新增 `*_analysis_pipeline.py` 自包含脚本
- 代码验证 100% 通过

详见 [SKILL_REVISION_PLAN.md](SKILL_REVISION_PLAN.md)

### v2.0.0 (2026-01-21)

- 多智能体校准框架
- 92.1% 校准通过率

### v1.0.0 (2026-01-20)

- 初始版本
- 21 个 Skills

## 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献指南

1. Fork 本仓库
2. 参考现有 Skill 结构创建新 Skill
3. 确保 Quick Start 代码可直接运行
4. 添加 `--demo` 模式进行验证
5. 提交 PR

## 许可

MIT License

---

**版本**: 3.0.0
**更新日期**: 2026-01-22
**Skills 数量**: 9
**设计**: Self-Contained
**GitHub**: [tangjia1986gz-lab/causal-ml-skills](https://github.com/tangjia1986gz-lab/causal-ml-skills)
