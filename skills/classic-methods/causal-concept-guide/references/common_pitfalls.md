# Common Pitfalls in Causal Inference

## Overview

This document catalogs the most common mistakes in causal inference, organized by category. Understanding these pitfalls is essential for both conducting and evaluating causal research.

---

## Category 1: Bad Controls

### What is a "Bad Control"?

A **bad control** is a variable that should NOT be included in your regression or matching, even though it might seem relevant. Including bad controls can:
- Remove part of the causal effect
- Introduce bias that wasn't there before
- Create spurious associations

### 1.1 Controlling for Post-Treatment Variables

**The Problem**: Variables affected by treatment are "descendants" of treatment. Conditioning on them:
- Blocks part of the causal pathway
- Opens collider paths

**Example**:
```
Treatment: Job training program
Bad control: Employment status 6 months after training
Outcome: Earnings 12 months after training

DAG: Training → Employment (6mo) → Earnings (12mo)
```

Controlling for employment removes the indirect effect through employment.

**Rule**: Only control for **pre-treatment** variables.

### 1.2 Controlling for Mediators

**The Problem**: Mediators transmit the causal effect. Controlling blocks this transmission.

**Example**:
```
Treatment: Education
Mediator: Occupation type
Outcome: Earnings

Question: What is the TOTAL effect of education on earnings?
Mistake: Control for occupation
Result: Only estimates "direct effect," missing indirect effect through occupation
```

**When intentional**: Mediation analysis explicitly decomposes direct and indirect effects, but requires strong assumptions.

### 1.3 Controlling for Colliders

**The Problem**: A collider is caused by both treatment and outcome (or their causes). Conditioning on a collider OPENS a non-causal path.

**Classic Example: Talent and Looks in Hollywood**
```
Talent → Success ← Looks

Without conditioning: Talent ⊥ Looks
Conditioning on Success: Talent appears negatively correlated with Looks
```

Among successful actors (conditional on success), those with less talent must compensate with more looks, creating spurious negative correlation.

**Example in Economics**:
```
Ability → College → Job quality ← Family connections
              ↘         ↗
               Occupation prestige
```

If you condition on occupation prestige (a collider), you bias the college effect.

### 1.4 The "Kitchen Sink" Problem

**The Problem**: Including all available variables "just to be safe."

**Why It Fails**:
- Some variables may be post-treatment
- Some may be colliders
- Some may be descendants of colliders
- Multicollinearity with treatment reduces precision

**Better Approach**: Draw a DAG, identify confounders, include only those.

### 1.5 M-Bias (Butterfly Bias)

**The Problem**: A variable M that looks like a confounder but is actually a collider.

**DAG**:
```
U₁      U₂
 \      /
  v    v
   X ← M → Y

(M is caused by U₁ and U₂, which separately affect X and Y)
```

**Mistake**: "M is associated with both X and Y, so it must be a confounder."
**Reality**: M is a collider. Conditioning on M opens the path U₁ → M ← U₂.

**Rule**: Don't assume every associated variable is a confounder. Draw the DAG.

---

## Category 2: Collider Bias (Selection Bias)

### 2.1 Sample Selection as Collider

**The Problem**: Selecting your sample based on criteria affected by treatment or outcome.

**Example: Hospitalization Studies**
```
Research question: Does treatment A improve recovery?
Sample: Hospitalized patients only
Problem: Hospitalization is affected by both illness severity and treatment received
```

Hospitalization is a collider. Conditioning on "hospitalized" creates bias.

### 2.2 Survivorship Bias

**The Problem**: Only observing units that "survived" to the measurement period.

**Example: Mutual Fund Performance**
```
Study: Average returns of mutual funds
Problem: Failed funds are removed from sample
Result: Overestimate of average fund performance
```

**Example: Company Studies**
```
Study: Characteristics of successful companies
Problem: Failed companies not in sample
Result: Survivor characteristics may not cause success
```

### 2.3 Berkson's Paradox

**The Problem**: Studying a selected population where selection depends on two factors.

**Example: Hospital Admission**
```
Two conditions: Heart disease, Diabetes
Admission: More likely if you have either condition
Within hospital patients: Heart disease and diabetes appear negatively correlated
General population: May be uncorrelated or positively correlated
```

### 2.4 Index Variable Trap

**The Problem**: An index or score that combines treatment and confounders.

**Example**:
```
Health index = diet quality + exercise + genetic factors
Treatment: Diet intervention
Mistake: Control for health index
Problem: Health index contains the treatment!
```

---

## Category 3: Confounding Mistakes

### 3.1 Omitted Variable Bias

**The Problem**: Failing to control for a confounder.

**Classic Example: Returns to Education**
```
Observed: Education → Earnings
Omitted confounder: Ability
Bias: OLS overestimates education effect because able people get more education AND higher earnings
```

**Formula**:
$$\text{Bias} = \rho_{D,U} \cdot \gamma_U$$

Where $\rho_{D,U}$ is correlation of treatment with omitted variable, $\gamma_U$ is effect of omitted variable on outcome.

### 3.2 Residual Confounding

**The Problem**: Confounders are measured with error, so controlling doesn't fully remove confounding.

**Example**:
```
True confounder: Socioeconomic status (SES)
Measured proxy: Income
Problem: Income doesn't capture education, wealth, social capital components of SES
```

**Rule**: Measurement error in confounders leaves residual bias.

### 3.3 Reverse Causality

**The Problem**: The outcome affects the treatment.

**Example: Police and Crime**
```
Observed correlation: More police → More crime
Reverse causality: More crime → More police deployed
```

**Solutions**: Use temporal ordering, instruments, or designs that break reverse causality.

### 3.4 Simultaneity

**The Problem**: Treatment and outcome are determined jointly.

**Example: Supply and Demand**
```
Price and quantity are determined simultaneously by supply and demand
Cannot estimate demand curve by regressing quantity on price
```

**Solutions**: Structural equations with instruments, or designs with exogenous variation.

---

## Category 4: Statistical Pitfalls

### 4.1 P-Hacking / Specification Searching

**The Problem**: Running many specifications until finding "significant" results.

**Manifestations**:
- Trying different sets of controls
- Different sample restrictions
- Different functional forms
- Different outcome definitions

**Rule**: Pre-register your analysis plan. Report all specifications tried.

### 4.2 Multiple Hypothesis Testing

**The Problem**: Testing many hypotheses inflates false positive rate.

**Example**:
```
Test treatment effect on 20 outcome variables
At α = 0.05, expect 1 "significant" result by chance
```

**Solutions**:
- Bonferroni correction: α / number of tests
- False Discovery Rate control
- Pre-specified primary outcome
- Report all tests conducted

### 4.3 Significance vs. Identification

**The Problem**: Treating statistical significance as evidence of causality.

**Reality**:
- A biased estimate can be statistically significant
- Statistical significance says nothing about identification
- Need **both** valid identification AND statistical significance

**Rule**: "Significant" means "precisely estimated," not "causal."

### 4.4 Low Power / Small Sample

**The Problem**: Sample too small to detect true effects.

**Consequences**:
- Null results are uninformative (underpowered)
- Significant results may be inflated (winner's curse)
- Type S errors (wrong sign) more likely

**Rule**: Conduct power analysis before study. Report achieved power.

### 4.5 Misinterpreting Null Results

**The Problem**: Concluding "no effect" from insignificant coefficient.

**Reality**: Insignificance could mean:
- No effect exists (correct conclusion)
- Effect exists but sample too small
- Effect exists but measurement error
- Effect exists but wrong functional form

**Better Approach**: Report confidence intervals, discuss power.

---

## Category 5: Method-Specific Pitfalls

### 5.1 DID: Parallel Trends Violation

**The Problem**: Treatment and control groups had different trends before treatment.

**Detection**:
- Plot pre-treatment trends
- Test for differential pre-trends
- Placebo tests with fake treatment dates

**Example of Bad DID**:
```
Treatment: State policy implemented in recession
Control: States not in recession
Problem: Economic trends were already different
```

### 5.2 DID: Anticipation Effects

**The Problem**: Units change behavior before official treatment date.

**Example**:
```
Policy: Tax increase announced 6 months before implementation
Problem: Behavior changes at announcement, not implementation
```

**Solution**: Use announcement date or account for anticipation.

### 5.3 RD: Manipulation at Cutoff

**The Problem**: Units strategically sort above or below the cutoff.

**Example**:
```
Program: Scholarship for students with GPA > 3.0
Manipulation: Grade inflation, rounding up, extra credit
```

**Detection**:
- McCrary density test (bunching at cutoff)
- Balance tests for predetermined characteristics

### 5.4 RD: Bandwidth Selection

**The Problem**: Results sensitive to bandwidth choice.

**Narrow bandwidth**: Less bias, more variance
**Wide bandwidth**: More bias, less variance

**Solution**: Use optimal bandwidth selectors (Imbens-Kalyanaraman, Calonico-Cattaneo-Titiunik)

### 5.5 IV: Weak Instruments

**The Problem**: Instrument barely affects treatment.

**Consequences**:
- Bias toward OLS
- Inflated standard errors
- Confidence intervals with poor coverage

**Detection**: First-stage F-statistic (rule of thumb: F > 10)

**Solutions**: Find stronger instrument, use weak-IV robust inference (LIML, Anderson-Rubin)

### 5.6 IV: Exclusion Restriction Violations

**The Problem**: Instrument affects outcome directly, not just through treatment.

**Example**:
```
Instrument: Quarter of birth (for education)
Treatment: Years of schooling
Potential violation: Quarter of birth affects health directly (birth season effects)
```

**Rule**: Exclusion restriction is **untestable**. Must argue from theory.

### 5.7 PSM: Poor Overlap

**The Problem**: Propensity scores don't overlap between treated and control.

**Example**:
```
Treatment: Expensive elective surgery
Treated: High income, good insurance
Control: Low income, poor insurance
Propensity scores: Minimal overlap
```

**Consequence**: Extreme weights, extrapolation, large variance.

**Detection**: Examine propensity score distributions, trim non-overlap regions.

### 5.8 PSM: Unobserved Confounding

**The Problem**: Assuming all confounders are observed (conditional independence).

**Reality**: Almost never true in observational data.

**Sensitivity Analysis**: How much unobserved confounding would be needed to change conclusions?

---

## Category 6: Interpretation Pitfalls

### 6.1 External Validity (Generalization)

**The Problem**: Effect identified in one context may not apply elsewhere.

**Examples**:
- RD effect is local to cutoff region
- LATE applies only to compliers
- ATT applies only to the treated

**Rule**: Be explicit about the population your estimate applies to.

### 6.2 Mechanism Confusion

**The Problem**: Conflating "what is the effect" with "why is there an effect."

**Identification**: Shows that X causes Y
**Mechanism**: Explains how X causes Y

**Rule**: Most causal inference methods identify effects, not mechanisms.

### 6.3 Policy Extrapolation

**The Problem**: Assuming estimated effects apply to different policy designs.

**Example**:
```
Study: Effect of increasing minimum wage by $1
Policy proposal: Increase minimum wage by $5
Problem: Nonlinear effects, different margins affected
```

**Rule**: Caution when extrapolating beyond support of data.

### 6.4 Heterogeneity Ignorance

**The Problem**: Reporting only average effect when effects vary.

**Consequence**: Average effect may be misleading for policy.

**Example**:
```
Average treatment effect: 0
Reality: Effect is +10 for half population, -10 for other half
```

**Solution**: Examine heterogeneity, report CATE if relevant.

---

## Summary: Red Flags Checklist

Before trusting a causal claim, check for these red flags:

### Design Red Flags
- [ ] No clear identification strategy stated
- [ ] Treatment assignment process unclear
- [ ] No discussion of potential confounders
- [ ] DAG not specified (even informally)

### Statistical Red Flags
- [ ] Multiple specifications reported without justification
- [ ] Only "significant" results shown
- [ ] Very large sample but small effect size
- [ ] No standard errors or confidence intervals

### Interpretation Red Flags
- [ ] Claims causality from observational correlation
- [ ] Ignores selection into treatment
- [ ] Assumes effect generalizes to all populations
- [ ] Confuses mechanism with identification

### Method-Specific Red Flags
- [ ] DID without pre-trend analysis
- [ ] RD without manipulation check
- [ ] IV without first-stage F-stat
- [ ] PSM without overlap assessment
- [ ] Any method without sensitivity analysis

---

## Related Skills

- `potential_outcomes.md` - Framework for defining causal effects
- `dags_and_graphs.md` - Graphical approach to confounding
- `selection_guide.md` - Choosing the right method
- `econometrics_vs_ml.md` - When to use ML approaches

---

## Key References

1. **Angrist, J. D., & Pischke, J.-S. (2009)**. *Mostly Harmless Econometrics*. Princeton University Press. (Chapters 3-4 on bad controls)

2. **Elwert, F., & Winship, C. (2014)**. Endogenous selection bias: The problem of conditioning on a collider variable. *Annual Review of Sociology*, 40, 31-53.

3. **Hernán, M. A. (2018)**. The C-word: Scientific euphemisms do not improve causal inference from observational data. *American Journal of Public Health*, 108(5), 616-619.

4. **Cinelli, C., Forney, A., & Pearl, J. (2022)**. A crash course in good and bad controls. *Sociological Methods & Research*.

5. **Gelman, A., & Loken, E. (2014)**. The statistical crisis in science. *American Scientist*, 102(6), 460-465.
