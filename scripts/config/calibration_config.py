#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration Configuration - 校准配置系统

包含:
1. 完整技能清单 (27个技能)
2. 每个技能的校准查询配置
3. 组件校准模板
4. 文献质量标准
5. 并行执行配置
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ============================================================================
# 枚举类型定义
# ============================================================================

class SkillCategory(Enum):
    """技能类别"""
    CLASSIC_METHODS = "classic-methods"
    CAUSAL_ML = "causal-ml"
    ML_FOUNDATION = "ml-foundation"
    INFRASTRUCTURE = "infrastructure"


class Priority(Enum):
    """校准优先级"""
    CRITICAL = 1  # 最高优先级
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class ComponentType(Enum):
    """组件类型"""
    IDENTIFICATION = "identification_assumptions"
    ESTIMATION = "estimation_methods"
    DIAGNOSTICS = "diagnostic_tests"
    REPORTING = "reporting_standards"
    ERRORS = "common_errors"


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class QueryConfig:
    """查询配置"""
    queries: List[str]
    min_citations: int = 200
    year_from: Optional[int] = 2010
    year_to: Optional[int] = None


@dataclass
class SkillConfig:
    """单个技能的校准配置"""
    name: str
    category: SkillCategory
    priority: Priority
    components: List[ComponentType]
    queries: Dict[str, QueryConfig]  # component -> QueryConfig
    core_citations: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""


# ============================================================================
# 顶刊列表
# ============================================================================

TOP_JOURNALS = {
    # 经济学五大刊
    "American Economic Review",
    "Econometrica",
    "Journal of Political Economy",
    "Quarterly Journal of Economics",
    "Review of Economic Studies",
    # 因果推断/计量
    "Journal of Econometrics",
    "Journal of Causal Inference",
    "Review of Economics and Statistics",
    "Journal of Business & Economic Statistics",
    "Econometric Theory",
    # 统计学
    "Journal of the American Statistical Association",
    "Annals of Statistics",
    "Biometrika",
    "Journal of the Royal Statistical Society",
    "Statistical Science",
    # ML/因果ML
    "Journal of Machine Learning Research",
    "Machine Learning",
    "NeurIPS",
    "ICML",
    "AISTATS",
    # 应用领域
    "Management Science",
    "Marketing Science",
    "American Journal of Epidemiology",
    "Epidemiology",
    # 方法论
    "Psychological Methods",
    "Structural Equation Modeling",
    "Multivariate Behavioral Research",
}


# ============================================================================
# 引用数阈值 (按组件类型)
# ============================================================================

CITATION_THRESHOLDS = {
    ComponentType.IDENTIFICATION: 500,   # 识别假设: 理论稳定
    ComponentType.ESTIMATION: 300,       # 估计方法: 方法论核心
    ComponentType.DIAGNOSTICS: 200,      # 诊断测试: 实用工具
    ComponentType.REPORTING: 100,        # 报告标准: 格式指南
    ComponentType.ERRORS: 50,            # 常见错误: 评审经验
}


# ============================================================================
# 完整技能配置 (27个技能)
# ============================================================================

SKILL_CALIBRATION_CONFIG: Dict[str, SkillConfig] = {

    # ═══════════════════════════════════════════════════════════════════════
    # 经典因果推断方法 (8个)
    # ═══════════════════════════════════════════════════════════════════════

    "estimator-did": SkillConfig(
        name="estimator-did",
        category=SkillCategory.CLASSIC_METHODS,
        priority=Priority.CRITICAL,
        components=[
            ComponentType.IDENTIFICATION,
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
            ComponentType.REPORTING,
            ComponentType.ERRORS,
        ],
        queries={
            "identification": QueryConfig(
                queries=[
                    "parallel trends assumption difference-in-differences",
                    "Callaway Sant'Anna conditional parallel trends",
                    "no anticipation assumption DID",
                    "common trends assumption identification",
                ],
                min_citations=500,
            ),
            "estimation": QueryConfig(
                queries=[
                    "two-way fixed effects staggered adoption",
                    "Goodman-Bacon decomposition weights",
                    "Sun Abraham interaction-weighted estimator",
                    "de Chaisemartin D'Haultfoeuille heterogeneous",
                    "doubly robust difference-in-differences",
                ],
                min_citations=300,
            ),
            "diagnostics": QueryConfig(
                queries=[
                    "pre-trends testing event study",
                    "placebo test difference-in-differences",
                    "Bacon decomposition negative weights",
                    "sensitivity analysis parallel trends",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Callaway", "Sant'Anna"], "year": 2021, "weight": 1.0},
            {"authors": ["Goodman-Bacon"], "year": 2021, "weight": 1.0},
            {"authors": ["Sun", "Abraham"], "year": 2021, "weight": 0.9},
            {"authors": ["de Chaisemartin", "D'Haultfoeuille"], "year": 2020, "weight": 0.9},
            {"authors": ["Borusyak", "Jaravel", "Spiess"], "year": 2024, "weight": 0.8},
            {"authors": ["Roth", "Sant'Anna"], "year": 2023, "weight": 0.8},
        ],
    ),

    "estimator-rd": SkillConfig(
        name="estimator-rd",
        category=SkillCategory.CLASSIC_METHODS,
        priority=Priority.CRITICAL,
        components=[
            ComponentType.IDENTIFICATION,
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
            ComponentType.REPORTING,
            ComponentType.ERRORS,
        ],
        queries={
            "identification": QueryConfig(
                queries=[
                    "regression discontinuity identification assumptions",
                    "local randomization RD design",
                    "continuity-based RD framework",
                    "fuzzy regression discontinuity identification",
                ],
                min_citations=500,
            ),
            "estimation": QueryConfig(
                queries=[
                    "local polynomial regression discontinuity",
                    "bandwidth selection RD Imbens Kalyanaraman",
                    "Calonico Cattaneo Titiunik robust inference",
                    "rdrobust package stata R",
                ],
                min_citations=300,
            ),
            "diagnostics": QueryConfig(
                queries=[
                    "McCrary density test manipulation",
                    "covariate balance RD design",
                    "donut hole RD specification",
                    "placebo cutoffs regression discontinuity",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Cattaneo", "Idrobo", "Titiunik"], "year": 2020, "weight": 1.0},
            {"authors": ["Imbens", "Kalyanaraman"], "year": 2012, "weight": 1.0},
            {"authors": ["Calonico", "Cattaneo", "Titiunik"], "year": 2014, "weight": 0.9},
            {"authors": ["McCrary"], "year": 2008, "weight": 0.8},
            {"authors": ["Lee", "Lemieux"], "year": 2010, "weight": 0.8},
        ],
    ),

    "estimator-iv": SkillConfig(
        name="estimator-iv",
        category=SkillCategory.CLASSIC_METHODS,
        priority=Priority.CRITICAL,
        components=[
            ComponentType.IDENTIFICATION,
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
            ComponentType.REPORTING,
            ComponentType.ERRORS,
        ],
        queries={
            "identification": QueryConfig(
                queries=[
                    "instrumental variable exclusion restriction",
                    "LATE local average treatment effect monotonicity",
                    "instrument relevance exogeneity",
                    "compliers defiers instrumental variables",
                ],
                min_citations=500,
            ),
            "estimation": QueryConfig(
                queries=[
                    "two-stage least squares 2SLS",
                    "LIML weak instruments",
                    "GMM instrumental variables heteroskedasticity",
                    "jackknife instrumental variables JIVE",
                ],
                min_citations=300,
            ),
            "diagnostics": QueryConfig(
                queries=[
                    "weak instrument Stock Yogo critical values",
                    "Sargan overidentification test",
                    "Anderson-Rubin confidence interval",
                    "Kleibergen-Paap rank test",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Angrist", "Imbens"], "year": 1996, "weight": 1.0},
            {"authors": ["Stock", "Yogo"], "year": 2005, "weight": 1.0},
            {"authors": ["Andrews", "Stock", "Sun"], "year": 2019, "weight": 0.9},
            {"authors": ["Lee", "McCrary", "Moreira"], "year": 2022, "weight": 0.8},
        ],
    ),

    "estimator-psm": SkillConfig(
        name="estimator-psm",
        category=SkillCategory.CLASSIC_METHODS,
        priority=Priority.CRITICAL,
        components=[
            ComponentType.IDENTIFICATION,
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
            ComponentType.REPORTING,
            ComponentType.ERRORS,
        ],
        queries={
            "identification": QueryConfig(
                queries=[
                    "conditional independence assumption unconfoundedness",
                    "overlap common support propensity score",
                    "SUTVA stable unit treatment value",
                    "selection on observables assumption",
                ],
                min_citations=500,
            ),
            "estimation": QueryConfig(
                queries=[
                    "propensity score matching nearest neighbor caliper",
                    "inverse probability weighting IPW",
                    "doubly robust AIPW augmented",
                    "entropy balancing weighting",
                ],
                min_citations=300,
            ),
            "diagnostics": QueryConfig(
                queries=[
                    "covariate balance standardized mean difference",
                    "Rosenbaum sensitivity analysis bounds",
                    "King Nielsen propensity score paradox",
                    "overlap assessment propensity score",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Rosenbaum", "Rubin"], "year": 1983, "weight": 1.0},
            {"authors": ["Imbens", "Rubin"], "year": 2015, "weight": 1.0},
            {"authors": ["Abadie", "Imbens"], "year": 2006, "weight": 0.9},
            {"authors": ["King", "Nielsen"], "year": 2019, "weight": 0.8},
            {"authors": ["Rosenbaum"], "year": 2002, "weight": 0.8},
        ],
    ),

    "panel-data-models": SkillConfig(
        name="panel-data-models",
        category=SkillCategory.CLASSIC_METHODS,
        priority=Priority.HIGH,
        components=[
            ComponentType.IDENTIFICATION,
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
            ComponentType.REPORTING,
        ],
        queries={
            "identification": QueryConfig(
                queries=[
                    "fixed effects random effects identification",
                    "strict exogeneity panel data",
                    "correlated random effects Mundlak",
                ],
                min_citations=400,
            ),
            "estimation": QueryConfig(
                queries=[
                    "within estimator fixed effects",
                    "generalized method of moments panel GMM",
                    "Arellano-Bond dynamic panel",
                ],
                min_citations=300,
            ),
            "diagnostics": QueryConfig(
                queries=[
                    "Hausman test fixed random effects",
                    "serial correlation test panel data",
                    "cross-sectional dependence panel",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Wooldridge"], "year": 2010, "weight": 1.0},
            {"authors": ["Arellano", "Bond"], "year": 1991, "weight": 0.9},
            {"authors": ["Baltagi"], "year": 2013, "weight": 0.8},
        ],
    ),

    "time-series-econometrics": SkillConfig(
        name="time-series-econometrics",
        category=SkillCategory.CLASSIC_METHODS,
        priority=Priority.HIGH,
        components=[
            ComponentType.IDENTIFICATION,
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
        ],
        queries={
            "identification": QueryConfig(
                queries=[
                    "cointegration identification time series",
                    "structural VAR identification restrictions",
                    "Granger causality identification",
                ],
                min_citations=400,
            ),
            "estimation": QueryConfig(
                queries=[
                    "ARIMA estimation forecasting",
                    "vector autoregression VAR estimation",
                    "error correction model VECM",
                ],
                min_citations=300,
            ),
            "diagnostics": QueryConfig(
                queries=[
                    "unit root test ADF KPSS",
                    "Johansen cointegration test",
                    "Ljung-Box autocorrelation test",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Hamilton"], "year": 1994, "weight": 1.0},
            {"authors": ["Engle", "Granger"], "year": 1987, "weight": 1.0},
            {"authors": ["Johansen"], "year": 1991, "weight": 0.9},
        ],
    ),

    "discrete-choice-models": SkillConfig(
        name="discrete-choice-models",
        category=SkillCategory.CLASSIC_METHODS,
        priority=Priority.HIGH,
        components=[
            ComponentType.IDENTIFICATION,
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
        ],
        queries={
            "identification": QueryConfig(
                queries=[
                    "multinomial logit identification IIA",
                    "ordered probit identification",
                    "random utility model identification",
                ],
                min_citations=300,
            ),
            "estimation": QueryConfig(
                queries=[
                    "maximum likelihood logit probit",
                    "conditional logit fixed effects",
                    "mixed logit random parameters",
                ],
                min_citations=200,
            ),
            "diagnostics": QueryConfig(
                queries=[
                    "Hausman McFadden IIA test",
                    "goodness of fit discrete choice",
                    "marginal effects discrete choice",
                ],
                min_citations=150,
            ),
        },
        core_citations=[
            {"authors": ["Train"], "year": 2009, "weight": 1.0},
            {"authors": ["McFadden"], "year": 1974, "weight": 1.0},
            {"authors": ["Greene"], "year": 2018, "weight": 0.8},
        ],
    ),

    "causal-concept-guide": SkillConfig(
        name="causal-concept-guide",
        category=SkillCategory.CLASSIC_METHODS,
        priority=Priority.MEDIUM,
        components=[
            ComponentType.IDENTIFICATION,
        ],
        queries={
            "identification": QueryConfig(
                queries=[
                    "potential outcomes framework Rubin",
                    "directed acyclic graph causal inference",
                    "do-calculus Pearl intervention",
                ],
                min_citations=500,
            ),
        },
        core_citations=[
            {"authors": ["Rubin"], "year": 1974, "weight": 1.0},
            {"authors": ["Pearl"], "year": 2009, "weight": 1.0},
            {"authors": ["Imbens", "Rubin"], "year": 2015, "weight": 0.9},
        ],
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 因果ML方法 (6个)
    # ═══════════════════════════════════════════════════════════════════════

    "causal-ddml": SkillConfig(
        name="causal-ddml",
        category=SkillCategory.CAUSAL_ML,
        priority=Priority.HIGH,
        components=[
            ComponentType.IDENTIFICATION,
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
        ],
        queries={
            "identification": QueryConfig(
                queries=[
                    "Neyman orthogonality double machine learning",
                    "cross-fitting sample splitting debiased",
                    "rate conditions high-dimensional inference",
                ],
                min_citations=300,
            ),
            "estimation": QueryConfig(
                queries=[
                    "partially linear regression PLR Chernozhukov",
                    "interactive regression model IRM",
                    "DoubleML Python R package",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Chernozhukov"], "year": 2018, "weight": 1.0},
            {"authors": ["Chernozhukov", "Chetverikov", "Demirer"], "year": 2018, "weight": 1.0},
            {"authors": ["Bach", "Chernozhukov", "Kurz"], "year": 2024, "weight": 0.9},
        ],
    ),

    "causal-forest": SkillConfig(
        name="causal-forest",
        category=SkillCategory.CAUSAL_ML,
        priority=Priority.HIGH,
        components=[
            ComponentType.IDENTIFICATION,
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
        ],
        queries={
            "identification": QueryConfig(
                queries=[
                    "unconfoundedness heterogeneous treatment effects",
                    "honesty splitting causal tree",
                    "CATE identification assumptions",
                ],
                min_citations=300,
            ),
            "estimation": QueryConfig(
                queries=[
                    "generalized random forests Athey Wager",
                    "causal tree recursive partitioning",
                    "CATE conditional average treatment effect estimation",
                    "grf R package econML",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Athey", "Wager"], "year": 2019, "weight": 1.0},
            {"authors": ["Wager", "Athey"], "year": 2018, "weight": 1.0},
            {"authors": ["Athey", "Imbens"], "year": 2016, "weight": 0.9},
        ],
    ),

    "structural-equation-modeling": SkillConfig(
        name="structural-equation-modeling",
        category=SkillCategory.CAUSAL_ML,
        priority=Priority.HIGH,
        components=[
            ComponentType.IDENTIFICATION,
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
            ComponentType.REPORTING,
        ],
        queries={
            "identification": QueryConfig(
                queries=[
                    "SEM identification t-rule degrees freedom",
                    "latent variable identification three indicator",
                    "structural equation model identification",
                ],
                min_citations=400,
            ),
            "estimation": QueryConfig(
                queries=[
                    "maximum likelihood SEM covariance structure",
                    "PLS-SEM partial least squares Hair",
                    "WLSMV ordinal categorical SEM",
                    "lavaan semopy structural equation",
                ],
                min_citations=300,
            ),
            "diagnostics": QueryConfig(
                queries=[
                    "CFI RMSEA SRMR fit indices cutoff",
                    "modification indices model respecification",
                    "Hu Bentler fit criteria",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Bollen"], "year": 1989, "weight": 1.0},
            {"authors": ["Kline"], "year": 2016, "weight": 1.0},
            {"authors": ["Rosseel"], "year": 2012, "weight": 0.9},
            {"authors": ["Hu", "Bentler"], "year": 1999, "weight": 0.8},
        ],
    ),

    "bayesian-econometrics": SkillConfig(
        name="bayesian-econometrics",
        category=SkillCategory.CAUSAL_ML,
        priority=Priority.MEDIUM,
        components=[
            ComponentType.IDENTIFICATION,
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
        ],
        queries={
            "identification": QueryConfig(
                queries=[
                    "prior specification Bayesian econometrics",
                    "posterior identification Bayesian",
                    "Bayesian credible interval interpretation",
                ],
                min_citations=200,
            ),
            "estimation": QueryConfig(
                queries=[
                    "MCMC Markov chain Monte Carlo econometrics",
                    "Stan PyMC Bayesian inference",
                    "hierarchical Bayesian model",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Gelman"], "year": 2014, "weight": 1.0},
            {"authors": ["Koop"], "year": 2003, "weight": 0.9},
            {"authors": ["McElreath"], "year": 2020, "weight": 0.8},
        ],
    ),

    "causal-mediation-ml": SkillConfig(
        name="causal-mediation-ml",
        category=SkillCategory.CAUSAL_ML,
        priority=Priority.HIGH,
        components=[
            ComponentType.IDENTIFICATION,
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
        ],
        queries={
            "identification": QueryConfig(
                queries=[
                    "mediation analysis identification assumptions",
                    "sequential ignorability mediation",
                    "direct indirect effect identification",
                ],
                min_citations=300,
            ),
            "estimation": QueryConfig(
                queries=[
                    "causal mediation analysis Imai",
                    "natural direct indirect effects estimation",
                    "sensitivity analysis mediation unmeasured confounding",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Imai", "Keele", "Tingley"], "year": 2010, "weight": 1.0},
            {"authors": ["VanderWeele"], "year": 2015, "weight": 1.0},
            {"authors": ["Pearl"], "year": 2012, "weight": 0.9},
        ],
    ),

    "paper-replication-workflow": SkillConfig(
        name="paper-replication-workflow",
        category=SkillCategory.CAUSAL_ML,
        priority=Priority.MEDIUM,
        components=[
            ComponentType.REPORTING,
        ],
        queries={
            "reporting": QueryConfig(
                queries=[
                    "AEA data editor replication package",
                    "reproducibility economics research",
                    "replication code documentation standards",
                ],
                min_citations=100,
            ),
        },
        core_citations=[
            {"authors": ["Vilhuber"], "year": 2020, "weight": 1.0},
        ],
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # ML基础 (6个)
    # ═══════════════════════════════════════════════════════════════════════

    "ml-preprocessing": SkillConfig(
        name="ml-preprocessing",
        category=SkillCategory.ML_FOUNDATION,
        priority=Priority.MEDIUM,
        components=[
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
        ],
        queries={
            "estimation": QueryConfig(
                queries=[
                    "feature engineering machine learning",
                    "missing data imputation methods",
                    "standardization normalization preprocessing",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Kuhn", "Johnson"], "year": 2019, "weight": 1.0},
        ],
    ),

    "ml-model-linear": SkillConfig(
        name="ml-model-linear",
        category=SkillCategory.ML_FOUNDATION,
        priority=Priority.MEDIUM,
        components=[
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
        ],
        queries={
            "estimation": QueryConfig(
                queries=[
                    "LASSO regularization variable selection",
                    "ridge regression shrinkage",
                    "elastic net penalized regression",
                ],
                min_citations=300,
            ),
        },
        core_citations=[
            {"authors": ["Tibshirani"], "year": 1996, "weight": 1.0},
            {"authors": ["Zou", "Hastie"], "year": 2005, "weight": 0.9},
        ],
    ),

    "ml-model-tree": SkillConfig(
        name="ml-model-tree",
        category=SkillCategory.ML_FOUNDATION,
        priority=Priority.MEDIUM,
        components=[
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
        ],
        queries={
            "estimation": QueryConfig(
                queries=[
                    "random forest ensemble learning",
                    "gradient boosting XGBoost LightGBM",
                    "SHAP feature importance interpretation",
                ],
                min_citations=300,
            ),
        },
        core_citations=[
            {"authors": ["Breiman"], "year": 2001, "weight": 1.0},
            {"authors": ["Chen", "Guestrin"], "year": 2016, "weight": 0.9},
            {"authors": ["Lundberg", "Lee"], "year": 2017, "weight": 0.8},
        ],
    ),

    "ml-model-advanced": SkillConfig(
        name="ml-model-advanced",
        category=SkillCategory.ML_FOUNDATION,
        priority=Priority.LOW,
        components=[
            ComponentType.ESTIMATION,
        ],
        queries={
            "estimation": QueryConfig(
                queries=[
                    "neural network deep learning economics",
                    "support vector machine SVM",
                    "ensemble methods stacking",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Goodfellow", "Bengio", "Courville"], "year": 2016, "weight": 1.0},
        ],
    ),

    "statistical-analysis": SkillConfig(
        name="statistical-analysis",
        category=SkillCategory.ML_FOUNDATION,
        priority=Priority.MEDIUM,
        components=[
            ComponentType.ESTIMATION,
            ComponentType.DIAGNOSTICS,
        ],
        queries={
            "estimation": QueryConfig(
                queries=[
                    "hypothesis testing statistical inference",
                    "effect size Cohen's d power analysis",
                    "multiple testing correction Bonferroni",
                ],
                min_citations=200,
            ),
        },
        core_citations=[
            {"authors": ["Cohen"], "year": 1988, "weight": 1.0},
        ],
    ),

    "econometric-eda": SkillConfig(
        name="econometric-eda",
        category=SkillCategory.ML_FOUNDATION,
        priority=Priority.MEDIUM,
        components=[
            ComponentType.DIAGNOSTICS,
        ],
        queries={
            "diagnostics": QueryConfig(
                queries=[
                    "exploratory data analysis econometrics",
                    "data quality assessment economics",
                    "outlier detection multivariate",
                ],
                min_citations=100,
            ),
        },
        core_citations=[
            {"authors": ["Tukey"], "year": 1977, "weight": 1.0},
        ],
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 基础设施 (2个)
    # ═══════════════════════════════════════════════════════════════════════

    "scientific-writing-econ": SkillConfig(
        name="scientific-writing-econ",
        category=SkillCategory.INFRASTRUCTURE,
        priority=Priority.MEDIUM,
        components=[
            ComponentType.REPORTING,
        ],
        queries={
            "reporting": QueryConfig(
                queries=[
                    "economic paper writing structure",
                    "AER submission guidelines format",
                    "academic writing economics style",
                ],
                min_citations=50,
            ),
        },
        core_citations=[
            {"authors": ["Thomson"], "year": 2011, "weight": 1.0},
        ],
    ),

    "setup-causal-ml-env": SkillConfig(
        name="setup-causal-ml-env",
        category=SkillCategory.INFRASTRUCTURE,
        priority=Priority.LOW,
        components=[],
        queries={},
        core_citations=[],
        notes="环境配置技能，无需文献校准",
    ),
}


# ============================================================================
# 组件校准模板
# ============================================================================

COMPONENT_CALIBRATION_TEMPLATES = {
    ComponentType.IDENTIFICATION: {
        "required_elements": [
            "formal_definition",      # 数学公式
            "intuition",              # 直观解释
            "testability",            # 可测试性评估
            "weaker_variants",        # 弱化版本
            "literature_source",      # 文献出处
        ],
        "min_papers": 3,
        "min_citations": 500,
        "output_format": "markdown_with_latex",
    },

    ComponentType.ESTIMATION: {
        "required_elements": [
            "formula",                # 估计量公式
            "algorithm_steps",        # 算法步骤
            "assumptions",            # 所需假设
            "standard_errors",        # 标准误计算
            "python_implementation",  # 代码实现
        ],
        "min_papers": 2,
        "min_citations": 300,
    },

    ComponentType.DIAGNOSTICS: {
        "required_elements": [
            "test_statistic",         # 检验统计量
            "null_hypothesis",        # 原假设
            "critical_values",        # 临界值表
            "interpretation",         # 解释标准
            "python_code",            # 实现代码
        ],
        "min_papers": 2,
        "min_citations": 200,
    },

    ComponentType.REPORTING: {
        "required_elements": [
            "required_table_elements", # 必填表格元素
            "latex_template",          # LaTeX模板
            "journal_conventions",     # 期刊惯例
        ],
        "min_papers": 1,
        "min_citations": 100,
    },

    ComponentType.ERRORS: {
        "required_elements": [
            "error_description",       # 错误描述
            "why_wrong",               # 为什么错误
            "correct_approach",        # 正确做法
            "code_example",            # 代码示例
        ],
        "min_papers": 1,
        "min_citations": 50,
    },
}


# ============================================================================
# 并行执行配置
# ============================================================================

PARALLEL_CONFIG = {
    "literature_agents": 5,      # 文献检索 Agent 并行数
    "extractor_agents": 3,       # 内容提取 Agent 并行数
    "gap_analyzer_agents": 5,    # 差距分析 Agent 并行数
    "formula_agents": 2,         # 公式校验 Agent 并行数
    "citation_agents": 3,        # 引用检查 Agent 并行数
    "updater_agents": 3,         # 文档更新 Agent 并行数

    "batch_size": 3,             # 每批技能数
    "timeout_per_skill": 300,    # 每个技能超时时间 (秒)
}


# ============================================================================
# 质量门控配置
# ============================================================================

QUALITY_GATE_CONFIG = {
    "thresholds": {
        "literature_coverage": 0.8,
        "content_completeness": 1.0,
        "formula_consistency": 0.9,
        "citation_validity": 1.0,
    },
    "weights": {
        "literature_coverage": 0.3,
        "content_completeness": 0.3,
        "formula_consistency": 0.2,
        "citation_validity": 0.2,
    },
    "pass_threshold": 0.85,
}


# ============================================================================
# 辅助函数
# ============================================================================

def get_skills_by_category(category: SkillCategory) -> List[str]:
    """获取指定类别的技能列表"""
    return [
        name for name, config in SKILL_CALIBRATION_CONFIG.items()
        if config.category == category
    ]


def get_skills_by_priority(priority: Priority) -> List[str]:
    """获取指定优先级的技能列表"""
    return [
        name for name, config in SKILL_CALIBRATION_CONFIG.items()
        if config.priority == priority
    ]


def get_all_queries(skill_name: str) -> List[str]:
    """获取技能的所有查询"""
    config = SKILL_CALIBRATION_CONFIG.get(skill_name)
    if not config:
        return []

    queries = []
    for comp_queries in config.queries.values():
        queries.extend(comp_queries.queries)

    return queries


def get_skill_config(skill_name: str) -> Optional[SkillConfig]:
    """获取技能配置"""
    return SKILL_CALIBRATION_CONFIG.get(skill_name)


def list_all_skills() -> List[str]:
    """列出所有技能名称"""
    return list(SKILL_CALIBRATION_CONFIG.keys())


def get_calibration_batches(batch_size: int = 3) -> List[List[str]]:
    """获取校准批次 (按优先级排序)"""
    # 按优先级排序
    sorted_skills = sorted(
        SKILL_CALIBRATION_CONFIG.items(),
        key=lambda x: x[1].priority.value
    )

    batches = []
    current_batch = []

    for skill_name, _ in sorted_skills:
        current_batch.append(skill_name)
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []

    if current_batch:
        batches.append(current_batch)

    return batches


# ============================================================================
# 配置导出
# ============================================================================

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("校准配置系统")
    print("=" * 60)

    print(f"\n总技能数: {len(SKILL_CALIBRATION_CONFIG)}")

    print("\n按类别统计:")
    for cat in SkillCategory:
        count = len(get_skills_by_category(cat))
        print(f"  {cat.value}: {count}")

    print("\n按优先级统计:")
    for pri in Priority:
        count = len(get_skills_by_priority(pri))
        print(f"  P{pri.value} ({pri.name}): {count}")

    print("\n校准批次:")
    batches = get_calibration_batches()
    for i, batch in enumerate(batches, 1):
        print(f"  批次 {i}: {batch}")

    # 导出为 JSON
    export_path = Path(__file__).parent / "calibration_config_export.json"
    export_data = {
        name: {
            "category": config.category.value,
            "priority": config.priority.value,
            "components": [c.value for c in config.components],
            "query_count": sum(len(q.queries) for q in config.queries.values()),
            "core_citations_count": len(config.core_citations),
        }
        for name, config in SKILL_CALIBRATION_CONFIG.items()
    }

    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"\n配置已导出到: {export_path}")
