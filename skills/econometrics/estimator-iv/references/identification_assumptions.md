# IV Identification Assumptions

## 1. Instrument Relevance (First Stage)

**Formal Definition:**
```
Cov(Z, X) != 0
```

The instrument Z must be correlated with the endogenous variable X.

**Testability:** TESTABLE via first-stage regression F-statistic.

**Key References:**
- Stock & Yogo (2005): Critical values for weak instrument tests
- Lee et al. (2022): Updated weak IV inference

## 2. Exclusion Restriction (Exogeneity)

**Formal Definition:**
```
Cov(Z, epsilon) = 0
```

The instrument Z affects the outcome Y only through its effect on X, not directly.

**Testability:** NOT DIRECTLY TESTABLE. Must rely on theoretical arguments.

**Note:** Sargan-Hansen test only tests overidentifying restrictions, not the exclusion restriction for all instruments.

**Key References:**
- Angrist & Pischke (2009): Mostly Harmless Econometrics, Ch. 4

## 3. Independence (Exogeneity of Instrument)

**Formal Definition:**
```
Z is independent of (epsilon, nu) where:
- epsilon: error in outcome equation
- nu: error in first stage
```

**Intuition:** The instrument must be "as good as randomly assigned" conditional on controls.

## 4. Monotonicity (for LATE Interpretation)

**Formal Definition:**
For binary instrument and binary treatment:
```
D_i(1) >= D_i(0) for all i  (no defiers)
```

Where D_i(z) is the potential treatment status when instrument equals z.

**Interpretation:** If the instrument moves anyone toward treatment, it doesn't move others away.

**Required for:** Interpreting IV as Local Average Treatment Effect (LATE)

**Key References:**
- Imbens & Angrist (1994): Original LATE framework
- Angrist, Imbens & Rubin (1996): "Identification of Causal Effects Using Instrumental Variables"

## 5. SUTVA (Stable Unit Treatment Value Assumption)

**Definition:** No interference between units; treatment effect is well-defined.

```
Y_i = Y_i(D_i, Z_i) - only depends on own treatment and instrument
```

## When Assumptions Fail

| Assumption | Consequence of Violation |
|------------|-------------------------|
| Relevance | Weak instruments → biased estimates, invalid inference |
| Exclusion | Biased and inconsistent estimates |
| Independence | Biased estimates |
| Monotonicity | LATE interpretation invalid |
| SUTVA | Estimand unclear |

## Robustness Checks

1. **Relevance:** First-stage F > 10 (Stock-Yogo)
2. **Exclusion:** Reduced form should show Z affects Y
3. **Overidentification:** Sargan-Hansen test (if multiple instruments)
4. **Sensitivity:** Conley et al. (2012) bounds for imperfect instruments
