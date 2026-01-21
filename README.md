# 机器学习因果推断 Skills

> 基于《机器学习因果推断实战全流程》课程大纲构建的 Claude Code Agent Skills 集合

## 项目状态

✅ **开发完成** - 全部 14 个 Skills 已实现，测试框架就绪

## 快速开始

```bash
# 1. 运行环境检查
python skills/infrastructure/setup-causal-ml-env/env_check.py

# 2. 生成测试数据
python tests/data/synthetic/dgp_did.py
python tests/data/synthetic/dgp_rd.py
python tests/data/synthetic/dgp_ddml.py
python tests/data/benchmark/generate_benchmarks.py

# 3. 运行验证测试
python tests/run_all_tests.py

# 4. 部署到 Claude Code
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
├── skills/                 # Skill 源文件
│   ├── infrastructure/     # 环境配置
│   │   └── setup-causal-ml-env/
│   ├── classic-methods/    # 经典因果方法
│   │   ├── causal-concept-guide/
│   │   ├── estimator-did/
│   │   ├── estimator-rd/
│   │   ├── estimator-iv/
│   │   └── estimator-psm/
│   ├── ml-foundation/      # 机器学习基础
│   │   ├── ml-preprocessing/
│   │   ├── ml-model-linear/
│   │   ├── ml-model-tree/
│   │   └── ml-model-advanced/
│   └── causal-ml/          # 前沿因果 ML
│       ├── causal-ddml/
│       ├── causal-forest/
│       ├── causal-mediation-ml/
│       └── paper-replication-workflow/
│
├── templates/              # 模板文件
│   ├── SKILL-TEMPLATE.md
│   ├── ESTIMATOR-TEMPLATE.md
│   └── OUTPUT-TEMPLATES/
│
├── lib/                    # 共享库
│   └── python/
│       ├── data_loader.py      # CausalInput/CausalOutput
│       ├── diagnostics.py      # 诊断函数
│       └── table_formatter.py  # 表格生成
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

### 已完成 (14 个)

| 模块 | Skill | 类型 | 触发词 | 状态 |
|------|-------|------|--------|------|
| **环境** | `setup-causal-ml-env` | Tool | 环境配置, python环境 | ✅ 完成 |
| **经典方法** | `causal-concept-guide` | Knowledge | 因果概念, 方法选择 | ✅ 完成 |
| | `estimator-did` | Estimator | DID, 双重差分, 平行趋势 | ✅ 完成 |
| | `estimator-rd` | Estimator | RD, 断点回归, 阈值 | ✅ 完成 |
| | `estimator-iv` | Estimator | IV, 工具变量, 2SLS | ✅ 完成 |
| | `estimator-psm` | Estimator | PSM, 倾向得分, 匹配 | ✅ 完成 |
| **ML 基础** | `ml-preprocessing` | Tool | 缺失值, 异常值, 特征工程 | ✅ 完成 |
| | `ml-model-linear` | Tool | Lasso, Ridge, 正则化 | ✅ 完成 |
| | `ml-model-tree` | Tool | 随机森林, XGBoost, SHAP | ✅ 完成 |
| | `ml-model-advanced` | Tool | SVM, 神经网络, MLP | ✅ 完成 |
| **前沿融合** | `causal-ddml` | Estimator | DDML, 双重机器学习, PLR | ✅ 完成 |
| | `causal-forest` | Estimator | 因果森林, CATE, 异质性 | ✅ 完成 |
| | `causal-mediation-ml` | Estimator | 中介分析, ADE, ACME | ✅ 完成 |
| **综合** | `paper-replication-workflow` | Workflow | 复现, LaLonde, Card | ✅ 完成 |

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 主控语言 | Python 3.10+ |
| 因果推断 | econml, doubleml, causalml |
| 计量统计 | statsmodels, linearmodels |
| 机器学习 | scikit-learn, xgboost, lightgbm |
| R 桥接 | rpy2 (grf, mediation) |
| 可视化 | matplotlib, seaborn |

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

### 手动部署
```bash
cp -r skills/*/* C:\Users\tangj\.claude\skills\
```

## 验证

### 运行所有验证测试
```bash
python tests/run_all_tests.py
```

### 验证标准
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

### 创建新 Skill

1. 复制模板
   ```bash
   cp templates/ESTIMATOR-TEMPLATE.md skills/<category>/<skill-name>/SKILL.md
   ```

2. 编辑 SKILL.md
   - 修改 frontmatter (name, description)
   - 按模板结构填写内容

3. 添加支持文件
   - Python 实现: `<skill_name>_estimator.py`
   - 测试数据生成器

4. 验证
   ```bash
   python tests/run_all_tests.py
   ```

5. 部署
   ```bash
   python deploy.py --skill <skill-name>
   ```

### 命名规范

- 目录名: `kebab-case` (如 `estimator-did`)
- 主文件: `SKILL.md` (固定)
- Python 文件: `snake_case.py`

## 许可

MIT

---

**开发完成**: 2025-01
**Skills 数量**: 14
**验证状态**: 通过
