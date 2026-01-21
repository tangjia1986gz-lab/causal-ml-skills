# 机器学习因果推断 Skills 技术架构

## 1. 项目概览

### 1.1 目标
构建一套标准化、可复用的 Claude Code Skills，赋予 Agent 完整的因果推断研究与实战能力。

### 1.2 核心原则
- **模块化**: 每个 Skill 独立可用，组合使用时无缝衔接
- **可验证**: 内置检验步骤，确保因果推断的有效性
- **多语言**: Python 为主控，支持 R/Stata 调用
- **学术规范**: 输出符合经济学/管理学顶刊格式

---

## 2. 目录结构

```
causal-ml-skills/
├── ARCHITECTURE.md          # 本文档
├── ROADMAP.md               # 开发路线图
├── README.md                # 项目说明
│
├── skills/                  # Skill 源文件
│   ├── infrastructure/      # 环境配置
│   │   └── setup-causal-ml-env/
│   │
│   ├── classic-methods/     # 经典因果方法
│   │   ├── estimator-did/
│   │   ├── estimator-rd/
│   │   ├── estimator-iv/
│   │   ├── estimator-psm/
│   │   └── causal-concept-guide/
│   │
│   ├── ml-foundation/       # 机器学习基础
│   │   ├── ml-preprocessing/
│   │   ├── ml-model-linear/
│   │   ├── ml-model-tree/
│   │   └── ml-model-advanced/
│   │
│   └── causal-ml/           # 前沿因果ML
│       ├── causal-ddml/
│       ├── causal-mediation-ml/
│       ├── causal-forest/
│       └── paper-replication-workflow/
│
├── templates/               # 模板文件
│   ├── SKILL-TEMPLATE.md    # Skill 标准模板
│   ├── ESTIMATOR-TEMPLATE.md # 估计器专用模板
│   └── OUTPUT-TEMPLATES/    # 输出格式模板
│       ├── regression-table.md
│       └── diagnostic-report.md
│
├── lib/                     # 共享库
│   ├── python/              # Python 工具函数
│   │   ├── data_loader.py
│   │   ├── diagnostics.py
│   │   ├── table_formatter.py
│   │   └── r_bridge.py      # R 语言桥接
│   │
│   └── r/                   # R 脚本
│       └── causal_forest.R
│
├── tests/                   # 测试用例
│   ├── data/                # 测试数据集
│   └── cases/               # 测试场景
│
└── docs/                    # 文档
    ├── concepts/            # 概念解释
    └── papers/              # 论文复现指南
```

---

## 3. Skill 分类体系

### 3.1 类型定义

| 类型 | 代号 | 说明 | 示例 |
|------|------|------|------|
| **Knowledge** | K | 概念解释、方法论指导 | `causal-concept-guide` |
| **Estimator** | E | 因果效应估计器 | `estimator-did`, `causal-ddml` |
| **Tool** | T | 数据处理、模型训练工具 | `ml-preprocessing` |
| **Workflow** | W | 端到端流程编排 | `paper-replication-workflow` |

### 3.2 依赖关系图

```
                    ┌─────────────────────┐
                    │ setup-causal-ml-env │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Classic Methods │  │  ML Foundation  │  │   Causal ML     │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ • DID           │  │ • Preprocessing │  │ • DDML          │
│ • RD            │  │ • Linear Models │  │ • Mediation     │
│ • IV            │  │ • Tree Models   │  │ • Causal Forest │
│ • PSM           │  │ • Advanced      │  │                 │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                     │
         └────────────────────┼─────────────────────┘
                              │
                              ▼
               ┌──────────────────────────┐
               │ paper-replication-workflow│
               └──────────────────────────┘
```

---

## 4. 技术栈

### 4.1 核心语言分工

| 任务 | 语言 | 工具/库 | 原因 |
|------|------|---------|------|
| 主控逻辑 | Python | - | 最广泛的生态系统 |
| 经典计量 | Python | `statsmodels`, `linearmodels` | 成熟稳定 |
| 双重机器学习 | Python | `econml`, `doubleml` | 微软/官方实现 |
| 因果森林 | R | `grf` | 最权威实现 |
| 高级计量 | Stata | `pystata` | 某些检验的金标准 |

### 4.2 Python 核心依赖

```python
# 因果推断核心
econml >= 0.15.0       # Microsoft EconML
doubleml >= 0.7.0      # Double ML
causalml >= 0.15.0     # Uber CausalML

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

### 4.3 R 核心依赖

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

## 5. Skill 接口规范

### 5.1 统一输入格式

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

### 5.2 统一输出格式

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

## 6. 质量保障

### 6.1 每个 Estimator Skill 必须包含

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

### 6.2 代码规范

- 类型注解: 所有公开函数必须有类型注解
- 文档字符串: NumPy 风格
- 测试覆盖: 核心逻辑 > 80%

---

## 7. 部署策略

### 7.1 开发阶段
Skills 在 `D:\code\PPcourse\causal-ml-skills\skills\` 下开发。

### 7.2 部署阶段
完成测试后，复制到 `C:\Users\tangj\.claude\skills\` 进行部署：

```bash
# 部署单个 Skill
cp -r skills/causal-ml/causal-ddml C:\Users\tangj\.claude\skills\

# 部署全部
cp -r skills/*/* C:\Users\tangj\.claude\skills\
```

### 7.3 命名规范
- 目录名: `kebab-case` (如 `causal-ddml`)
- 主文件: `SKILL.md`
- 支持文件: `lowercase-with-hyphens.md`

---

## 8. 版本管理

- 使用 Git 管理 `causal-ml-skills` 目录
- 每个 Skill 的 SKILL.md 头部标注版本
- 重大更新记录在 CHANGELOG.md
