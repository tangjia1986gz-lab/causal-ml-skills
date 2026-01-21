# PSM Identification Assumptions

## 1. Conditional Independence (Unconfoundedness)

**Formal Definition:**
```
(Y(1), Y(0)) ⊥ T | X
```

Given observed covariates X, treatment assignment T is independent of potential outcomes Y(1) and Y(0).

**Intuition:** After controlling for X, treatment is "as good as randomly assigned."

**Alternative names:**
- Selection on observables
- No unmeasured confounding
- Ignorability

**Testability:** NOT DIRECTLY TESTABLE. Must rely on theoretical arguments and sensitivity analysis.

**Key References:**
- Rosenbaum & Rubin (1983): Original formulation
- Imbens (2004): Nonparametric estimation

## 2. Overlap (Common Support / Positivity)

**Formal Definition:**
```
0 < P(T=1|X) < 1  for all X in support
```

For every covariate value, there must be positive probability of receiving either treatment.

**Intuition:** We need both treated and control units at each covariate profile to estimate effects.

**Testability:** TESTABLE via propensity score distributions.

**Violations:**
- Deterministic selection (e.g., all high-income receive treatment)
- Practical violations (very few units in tails)

**Solutions:**
- Trim propensity scores (drop obs with ps < 0.01 or > 0.99)
- Focus on ATT (requires overlap only in treated region)
- Use overlap weights

## 3. SUTVA (Stable Unit Treatment Value Assumption)

**Two components:**

**3a. No interference:**
```
Y_i does not depend on T_j for i ≠ j
```
One unit's treatment doesn't affect another's outcome.

**3b. No hidden variations:**
```
Treatment is well-defined with no variants
```
There's only one version of "treatment."

**Violations:**
- Network effects (peer influence)
- General equilibrium effects
- Heterogeneous treatment implementation

## 4. Correct Model Specification

**Requirement:**
The propensity score model must correctly specify the relationship between X and P(T=1|X).

**Note:** This is why balance checking is crucial - it verifies the functional form.

**Solutions:**
- Flexible models (GBM, random forests)
- Include interactions and polynomials
- Check covariate balance

## When Assumptions Fail

| Assumption | Consequence | Partial Solution |
|------------|-------------|------------------|
| Unconfoundedness | Biased estimates | Sensitivity analysis |
| Overlap | Extreme weights, extrapolation | Trimming, change estimand |
| SUTVA | Bias, wrong estimand | Network models, cluster-level treatment |
| Model spec | Poor balance | Flexible models, balance checking |

## Robustness Checks

1. **Unconfoundedness:**
   - Rosenbaum sensitivity analysis
   - Oster (2019) coefficient stability
   - Placebo treatments

2. **Overlap:**
   - Propensity score histograms
   - Trimming sensitivity
   - Compare ATT vs ATE

3. **Model specification:**
   - Multiple PS models
   - Balance after adjustment
   - Doubly robust estimation

## Key References

1. Rosenbaum & Rubin (1983): "The Central Role of the Propensity Score"
2. Imbens (2004): "Nonparametric Estimation of Average Treatment Effects"
3. Rosenbaum (2002): "Observational Studies" - sensitivity analysis
4. Oster (2019): "Unobservable Selection and Coefficient Stability"
