# Econometrics vs. Machine Learning for Causal Inference

## Overview

Traditional econometrics and machine learning represent different philosophies for analyzing data. This guide clarifies when each approach is appropriate for causal inference and how they can be combined.

---

## Fundamental Distinction

### Econometrics: Model-Based, Theory-Driven

**Philosophy**: Specify a model based on economic theory, estimate parameters with clear causal interpretation.

**Typical Questions**:
- What is the causal effect of X on Y?
- What is the elasticity of demand?
- How large is the treatment effect?

**Strengths**:
- Clear identification strategies
- Interpretable parameters
- Formal hypothesis testing
- Focus on causality

### Machine Learning: Algorithm-Based, Data-Driven

**Philosophy**: Let algorithms find patterns in data, optimize prediction accuracy.

**Typical Questions**:
- How can we predict Y from features?
- Which features matter most for prediction?
- How can we classify observations?

**Strengths**:
- Handles high-dimensional data
- Discovers nonlinear relationships
- Excellent prediction performance
- Scales to large datasets

---

## The Core Issue: Prediction vs. Causation

### Why Good Predictions Are Not Causal

| Scenario | Good Predictor | Causal? |
|----------|---------------|---------|
| Umbrella sales predict rain | Umbrellas sold | No (reverse causation) |
| Ice cream sales predict crime | Ice cream sales | No (common cause: temperature) |
| Hospital visits predict death | Hospital admission | No (selection bias) |

**Principle**: Correlation (even strong, predictive correlation) is not causation.

### The Prediction-Causation Trade-off

| Goal | Preferred Approach | Reason |
|------|-------------------|--------|
| **Predict Y** | ML (include all predictive features) | Maximize predictive accuracy |
| **Estimate effect of X on Y** | Econometrics (include confounders, exclude colliders) | Identify causal effect |
| **Policy: Change X to change Y** | Causal inference | Need to know what happens when X changes |

---

## When to Use Traditional Econometrics

### Scenario 1: Clear Identification Strategy Exists

**Examples**:
- RCT analysis
- Natural experiments (DID, RD, IV)
- Clean quasi-experimental designs

**Why Econometrics**: The identification strategy (not sophisticated algorithms) drives causal inference.

### Scenario 2: Low-Dimensional Setting

**Characteristics**:
- Few potential confounders
- Clear theory about relevant variables
- Interpretable model specification

**Example**:
```
Outcome: Wages
Treatment: Years of education
Confounders: Experience, ability (proxied by test scores)
```

Standard regression handles this well.

### Scenario 3: Need Structural Parameters

**Goal**: Estimate policy-relevant parameters (elasticities, marginal effects).

**Example**: Labor supply elasticity needs economic interpretation, not just prediction.

### Scenario 4: Hypothesis Testing Is Primary

**Goal**: Test whether an effect exists and is statistically significant.

**Requirements**: Valid standard errors, proper specification.

---

## When to Use Machine Learning

### Scenario 1: High-Dimensional Confounding

**Characteristics**:
- Many potential confounders
- Unknown functional form
- Selection-on-observables setting

**Example**:
```
Setting: Health outcomes from observational medical records
Features: 1000+ diagnosis codes, medications, lab values
Challenge: Which features are confounders?
```

**ML Solution**: DDML, Causal Forest - let algorithms select/weight confounders.

### Scenario 2: Heterogeneous Treatment Effects

**Goal**: Discover how effects vary across individuals.

**Why ML**: Causal forests can discover subgroups with different effects without pre-specification.

### Scenario 3: Prediction Is the Goal

**Examples**:
- Credit scoring (predict default)
- Medical diagnosis (predict disease)
- Demand forecasting (predict sales)

**Note**: If you will ACT on the prediction in ways that change outcomes, you need causal inference.

### Scenario 4: Feature Engineering / Discovery

**Goal**: Find which features matter, reduce dimensionality.

**Application**: As input to subsequent causal analysis.

---

## The Modern Synthesis: Causal ML

### Key Insight

**"First, identify. Then, estimate efficiently."**

1. Use economic theory/research design for **identification**
2. Use ML for **efficient estimation** within that design

### Double/Debiased Machine Learning (DDML)

**When to Use**:
- High-dimensional confounders
- Selection-on-observables assumption is credible
- Want ATE or ATT with flexible functional forms

**How It Works**:
1. Use ML to predict treatment from confounders (propensity score)
2. Use ML to predict outcome from confounders
3. Use residuals in causal estimation
4. Cross-fitting prevents overfitting bias

**Key Requirement**: Identification assumption (conditional independence) must hold.

### Causal Forests

**When to Use**:
- Want heterogeneous treatment effects
- Have moderate-to-large sample
- Treatment is quasi-random (conditional on observed features)

**How It Works**:
1. Random forest structure
2. Optimizes for causal effect estimation, not prediction
3. "Honesty" - separate samples for splitting and estimation
4. Provides valid confidence intervals for CATE

**Key Requirement**: Unconfoundedness must hold.

### Targeted Learning (TMLE)

**When to Use**:
- Want doubly-robust estimation
- Complex treatment regimes
- Need valid inference with ML nuisance parameters

**How It Works**:
1. Initial estimates from ML
2. Targeted update to reduce bias
3. Influence function-based inference

---

## Comparison Table: Econometrics vs. ML vs. Causal ML

| Aspect | Traditional Econometrics | Pure ML | Causal ML |
|--------|------------------------|---------|-----------|
| **Primary goal** | Causal inference | Prediction | Causal inference |
| **Model specification** | Theory-driven | Data-driven | Hybrid |
| **Dimensionality** | Low | High | High |
| **Functional form** | Parametric | Flexible | Flexible |
| **Identification** | Explicit strategy | Not considered | Explicit strategy |
| **Interpretation** | Causal parameters | Features/predictions | Causal parameters |
| **Uncertainty** | Standard errors, CI | Often absent | Valid inference |
| **Data requirements** | Moderate | Large | Large |

---

## Decision Framework

### Step 1: Define Your Goal

```
What is your research question?
│
├─► Predict outcome Y? → ML
│
├─► Understand effect of X on Y?
│   │
│   └─► For policy intervention? → Causal inference (econometrics or causal ML)
│
└─► Discover patterns? → ML for exploration, then causal methods for validation
```

### Step 2: Assess Your Data

```
What is your data setting?
│
├─► Low-dimensional (< 20 covariates)?
│   └─► Traditional econometrics likely sufficient
│
├─► High-dimensional (many covariates)?
│   └─► Consider Causal ML methods
│
├─► Large sample (N > 5000)?
│   └─► ML methods more feasible
│
└─► Small sample (N < 500)?
    └─► Traditional econometrics more stable
```

### Step 3: Check Identification

```
Is causal identification credible?
│
├─► Random assignment? → RCT analysis (ML for heterogeneity)
│
├─► Natural experiment?
│   ├─► Threshold? → RD (ML for bandwidth selection)
│   ├─► Policy change? → DID (ML for parallel trends testing)
│   └─► Instrument? → IV (ML for many instruments)
│
├─► Selection on observables?
│   ├─► Low-dimensional? → PSM, regression
│   └─► High-dimensional? → DDML, Causal Forest
│
└─► No clear identification?
    └─► Be very cautious about causal claims
```

---

## Common Mistakes

### Mistake 1: Using ML for Causal Inference Without Identification

**Problem**: Applying ML to observational data and interpreting feature importance as causal effects.

**Why It Fails**: ML optimizes prediction, not causal identification. Confounders and predictors are treated the same.

**Example**:
```
ML finding: "Ice cream sales predict crime"
Causal interpretation: "Ice cream causes crime" → WRONG
Reality: Both caused by temperature
```

### Mistake 2: Ignoring ML When It Would Help

**Problem**: Using simple linear models when high-dimensional confounders exist.

**Why It Fails**: May miss important confounders or misspecify functional form.

**Solution**: Use DDML or similar to flexibly control for confounders.

### Mistake 3: Over-Trusting Causal ML

**Problem**: Treating DDML or Causal Forest as automatically solving identification.

**Reality**: These methods still require the identification assumption (unconfoundedness) to hold.

**Rule**: Causal ML is a statistical tool, not a substitute for research design.

### Mistake 4: Wrong Tool for the Job

| Task | Wrong Approach | Right Approach |
|------|---------------|----------------|
| Predict customer churn | Causal model | ML prediction |
| Estimate ad effectiveness | Correlation from observational data | RCT or quasi-experiment |
| Personalize treatment | Average treatment effect only | Heterogeneous effects (Causal Forest) |
| Select control variables | Include everything | Theory + DAG + DDML |

---

## Hybrid Approaches: Best of Both Worlds

### Approach 1: ML for Nuisance Parameters

Use ML to estimate "nuisance" parameters (propensity scores, outcome predictions), then plug into econometric estimator.

**Example**: DDML uses ML for prediction, economics for identification.

### Approach 2: Econometrics for Identification, ML for Heterogeneity

**Workflow**:
1. Establish causal effect exists (RCT, DID, RD, IV)
2. Use Causal Forest to explore heterogeneity
3. Interpret heterogeneity cautiously (exploratory)

### Approach 3: ML for Pre-Analysis Data Preparation

Use ML for:
- Feature selection (reduce dimension before causal analysis)
- Missing data imputation
- Anomaly detection

Then apply traditional causal methods.

### Approach 4: Sensitivity Analysis

Use ML to explore sensitivity:
- Different specifications
- Robustness across subsamples
- Out-of-sample validation of predictions

---

## Summary: When to Use What

| Method | Use When |
|--------|----------|
| **OLS/IV/DID/RD** | Clear identification, low-dimensional, interpretable parameters needed |
| **DDML** | Selection on observables, high-dimensional confounders, large sample |
| **Causal Forest** | Want heterogeneous effects, moderate-large sample, quasi-random treatment |
| **Pure ML (XGBoost, NN)** | Prediction is the goal, NOT causal inference |
| **Structural Models** | Need counterfactual policy simulation, have clear theory |

---

## Related Skills

- `causal-ddml` - Double/Debiased ML implementation
- `ml-model-tree` - Tree-based ML models
- `ml-preprocessing` - Data preparation for ML
- `estimator-psm` - Traditional propensity score matching

---

## Key References

### Causal ML

1. **Athey, S., & Imbens, G. W. (2019)**. Machine learning methods that economists should know about. *Annual Review of Economics*, 11, 685-725.

2. **Chernozhukov, V., et al. (2018)**. Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1), C1-C68.

3. **Wager, S., & Athey, S. (2018)**. Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228-1242.

### Traditional Econometrics

4. **Angrist, J. D., & Pischke, J.-S. (2009)**. *Mostly Harmless Econometrics*. Princeton University Press.

5. **Wooldridge, J. M. (2010)**. *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

### Machine Learning

6. **Hastie, T., Tibshirani, R., & Friedman, J. (2009)**. *The Elements of Statistical Learning* (2nd ed.). Springer.

7. **Mullainathan, S., & Spiess, J. (2017)**. Machine learning: An applied econometric approach. *Journal of Economic Perspectives*, 31(2), 87-106.
