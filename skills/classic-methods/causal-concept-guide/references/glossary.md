# Glossary of Causal Inference Terms

## Overview

This glossary provides definitions for key terms used in causal inference. Terms are organized alphabetically with cross-references to related concepts.

---

## A

### Assignment Mechanism
The process by which units receive treatment or control status. Understanding the assignment mechanism is crucial for choosing the appropriate estimation method.
- **Random assignment**: Treatment assigned by chance (RCT)
- **Self-selection**: Units choose their own treatment
- **Rule-based**: Assignment based on observable criteria (RD setting)
See also: *Treatment*, *Selection Bias*

### ATE (Average Treatment Effect)
$$\text{ATE} = E[Y(1) - Y(0)]$$
The expected difference in potential outcomes across the entire population. Represents the effect for a randomly selected unit.
See also: *ATT*, *ATU*, *LATE*

### ATT (Average Treatment Effect on the Treated)
$$\text{ATT} = E[Y(1) - Y(0) \mid D=1]$$
The expected treatment effect for units that actually received treatment. Often the relevant quantity for evaluating existing programs.
See also: *ATE*, *Selection Bias*

### ATU (Average Treatment Effect on the Untreated)
$$\text{ATU} = E[Y(1) - Y(0) \mid D=0]$$
The expected treatment effect for units that did not receive treatment, if they were to be treated. Relevant for program expansion decisions.
See also: *ATE*, *ATT*

---

## B

### Backdoor Path
A non-causal path from treatment to outcome that goes "backward" into the treatment (i.e., starts with an arrow pointing into the treatment). Backdoor paths create confounding.
See also: *Backdoor Criterion*, *DAG*, *Confounding*

### Backdoor Criterion
A graphical criterion for identifying confounders. A set of variables Z satisfies the backdoor criterion if it blocks all backdoor paths from treatment to outcome and does not include descendants of treatment.
See also: *Backdoor Path*, *d-Separation*

### Balance
The similarity of covariate distributions between treated and control groups. In RCTs, randomization produces balance in expectation. In observational studies, matching or weighting can improve balance.
See also: *Covariate*, *Propensity Score*

### Bounds
Upper and lower limits on a causal effect when point identification is not possible. Bounds analysis makes weaker assumptions than point estimation.

---

## C

### CATE (Conditional Average Treatment Effect)
$$\text{CATE}(x) = E[Y(1) - Y(0) \mid X=x]$$
The treatment effect for units with specific covariate values X=x. Captures treatment effect heterogeneity.
See also: *Heterogeneous Treatment Effects*, *Causal Forest*

### Causal Effect
The change in an outcome that results from a change in treatment, holding all else constant. Defined as the difference between potential outcomes.
See also: *Treatment Effect*, *Potential Outcomes*

### Causal Forest
A machine learning method for estimating heterogeneous treatment effects. Uses random forest structure optimized for causal effect estimation rather than prediction.
See also: *CATE*, *GRF*

### Collider
A variable that is caused by two (or more) other variables. In DAG notation: A → C ← B. Conditioning on a collider opens a spurious path between its causes.
See also: *Collider Bias*, *DAG*

### Collider Bias
Bias introduced by conditioning on a collider variable. Unlike confounding (which is removed by conditioning), collider bias is created by conditioning.
See also: *Collider*, *Selection Bias*

### Common Support
See *Overlap*

### Compliers
In instrumental variables settings, units whose treatment status is determined by the instrument. If Z=1 implies D=1 and Z=0 implies D=0, the unit is a complier. LATE estimates the effect for compliers.
See also: *LATE*, *Instrumental Variable*, *Always-takers*, *Never-takers*

### Conditional Independence
$$\{Y(0), Y(1)\} \perp\!\!\!\perp D \mid X$$
The assumption that potential outcomes are independent of treatment assignment conditional on observed covariates. Required for selection-on-observables methods like matching.
See also: *Ignorability*, *Unconfoundedness*

### Confounding
The existence of common causes of both treatment and outcome, creating a spurious association that is not causal.
See also: *Confounder*, *Backdoor Path*, *Omitted Variable Bias*

### Confounder
A variable that causes both the treatment and the outcome. Failing to control for confounders leads to omitted variable bias.
See also: *Confounding*, *Backdoor Path*

### Counterfactual
What would have happened to a unit under an alternative treatment status. For a treated unit, the counterfactual is Y(0); for a control unit, it is Y(1). Fundamentally unobservable.
See also: *Potential Outcomes*, *Fundamental Problem*

### Covariate
A pre-treatment variable that may be used for adjustment, matching, or as a control variable. Also called a "control variable" or "feature."

### Cross-Fitting
A sample-splitting technique used in DDML to avoid overfitting. The sample is split into folds; ML models are trained on one set of folds and predictions made on held-out folds.
See also: *DDML*

---

## D

### d-Separation
A graphical criterion for determining conditional independence. Two variables are d-separated given a conditioning set if all paths between them are blocked.
See also: *DAG*, *Blocking*

### DAG (Directed Acyclic Graph)
A graph representing causal relationships where edges are directed (arrows) and no cycles exist. Used to reason about confounding, mediation, and identification.
See also: *d-Separation*, *Backdoor Criterion*, *Collider*

### DDML (Double/Debiased Machine Learning)
A method combining machine learning for nuisance parameter estimation with econometric techniques for causal inference. Uses cross-fitting to avoid regularization bias.
See also: *Cross-Fitting*, *Nuisance Parameter*

### Defiers
In instrumental variables settings, units whose treatment status goes opposite to the instrument. If Z=1 implies D=0 and Z=0 implies D=1, the unit is a defier. Standard IV assumes no defiers (monotonicity).
See also: *Compliers*, *Monotonicity*

### DID (Difference-in-Differences)
A method comparing changes over time between treated and control groups. Identifies causal effects under the parallel trends assumption.
See also: *Parallel Trends*, *Two-Way Fixed Effects*

### do-Operator
Pearl's notation for intervention. P(Y | do(X=x)) is the distribution of Y when X is set to x by intervention, distinct from P(Y | X=x) which conditions on observing X=x.
See also: *Intervention*, *DAG*

### Doubly Robust
An estimator that is consistent if either the outcome model or the propensity score model is correctly specified (but not necessarily both). Provides some protection against misspecification.
See also: *AIPW*, *IPW*

---

## E

### Endogeneity
A situation where the treatment variable is correlated with the error term in a regression, typically due to omitted variables, measurement error, or simultaneity. Makes OLS estimates inconsistent.
See also: *Exogeneity*, *Instrumental Variable*

### Estimand
The target quantity to be estimated, defined at the population level (e.g., ATE, ATT, LATE). Distinct from the estimator (the procedure) and estimate (the number).

### Exclusion Restriction
In instrumental variables, the assumption that the instrument affects the outcome only through the treatment variable, not directly. Untestable and must be argued from theory.
See also: *Instrumental Variable*, *IV*

### Exogeneity
A variable is exogenous if it is uncorrelated with the error term. Treatment exogeneity (no omitted confounders) is required for consistent causal effect estimation.
See also: *Endogeneity*, *Unconfoundedness*

### External Validity
Whether results from one study generalize to other populations, settings, or time periods. Even internally valid estimates may not have external validity.
See also: *Internal Validity*, *LATE*

---

## F

### First Stage
In instrumental variables estimation, the regression of the treatment on the instrument(s). A strong first stage (F > 10) is necessary for valid inference.
See also: *Instrumental Variable*, *Weak Instrument*

### Frontdoor Criterion
A graphical criterion for identification when backdoor adjustment is not possible. Uses mediator variables to identify causal effects despite unmeasured confounding.
See also: *Backdoor Criterion*, *DAG*

### Fundamental Problem of Causal Inference
The impossibility of observing both potential outcomes for the same unit at the same time. We observe Y(1) if treated, Y(0) if not, but never both.
See also: *Potential Outcomes*, *Counterfactual*

### Fuzzy RD
Regression discontinuity design where the probability of treatment changes discontinuously at the cutoff, but treatment is not deterministically assigned. Analyzed using IV with the threshold as instrument.
See also: *Sharp RD*, *Regression Discontinuity*

---

## G

### GRF (Generalized Random Forests)
A framework for forest-based estimation of heterogeneous treatment effects and other quantities. Provides valid statistical inference through honest estimation.
See also: *Causal Forest*, *Honest Estimation*

---

## H

### Heterogeneous Treatment Effects
Treatment effects that vary across individuals or subgroups. Captured by CATE.
See also: *CATE*, *Causal Forest*, *Effect Modification*

### Honest Estimation
Using separate samples for determining tree splits and estimating within-leaf treatment effects. Prevents overfitting and enables valid inference in Causal Forests.
See also: *Causal Forest*, *GRF*

---

## I

### Identification
The theoretical possibility of recovering a causal parameter from observed data given assumptions. A parameter is identified if different values of the parameter would produce different observable implications.
See also: *Identification Strategy*

### Identification Strategy
The research design and assumptions used to isolate causal effects from confounding. Common strategies include RCT, DID, RD, IV, and matching.

### Ignorability
See *Conditional Independence*, *Unconfoundedness*

### Instrument
See *Instrumental Variable*

### Instrumental Variable (IV)
A variable that affects treatment but does not directly affect the outcome (exclusion restriction) and is correlated with treatment (relevance). Used to identify causal effects when treatment is endogenous.
See also: *2SLS*, *LATE*, *Exclusion Restriction*, *First Stage*

### Intent-to-Treat (ITT)
The effect of being assigned to treatment, regardless of whether treatment was actually received. In presence of non-compliance, ITT differs from the treatment effect on compliers.
See also: *LATE*, *Compliers*

### Internal Validity
Whether the estimated effect is a valid causal estimate for the study population. Requires successful addressing of confounding, selection bias, and reverse causality.
See also: *External Validity*

### Intervention
An action that sets a variable to a particular value, breaking its dependence on its causes. Distinct from mere observation or conditioning.
See also: *do-Operator*, *Treatment*

### IPW (Inverse Probability Weighting)
Weighting observations by the inverse of their propensity score to create a pseudo-population where treatment is independent of confounders.
See also: *Propensity Score*, *AIPW*

---

## L

### LATE (Local Average Treatment Effect)
$$\text{LATE} = E[Y(1) - Y(0) \mid \text{Compliers}]$$
The treatment effect for compliers in an IV setting. LATE may differ from ATE if compliers are not representative of the full population.
See also: *Compliers*, *Instrumental Variable*

### Local Randomization
In RD designs, the assumption that assignment is as-if random in a small window around the cutoff. Justifies treating RD as a local experiment.
See also: *Regression Discontinuity*

---

## M

### Manipulation
In RD designs, the ability of units to strategically sort above or below the cutoff. Manipulation invalidates the RD design.
See also: *Regression Discontinuity*, *McCrary Test*

### Matching
Methods that compare treated units to similar control units based on observed characteristics. Includes nearest-neighbor matching, caliper matching, and propensity score matching.
See also: *Propensity Score*, *Conditional Independence*

### McCrary Test
A statistical test for manipulation in RD designs that examines whether the density of the running variable is continuous at the cutoff.
See also: *Manipulation*, *Regression Discontinuity*

### Mediator
A variable on the causal pathway from treatment to outcome. Treatment affects the mediator, which in turn affects the outcome.
See also: *Mediation*, *Direct Effect*, *Indirect Effect*

### Mediation
The mechanism through which treatment affects the outcome. Mediation analysis decomposes total effects into direct and indirect (mediated) effects.
See also: *Mediator*, *Direct Effect*, *Indirect Effect*

### Monotonicity
In IV settings, the assumption that the instrument affects treatment in the same direction for all units. Rules out defiers and ensures LATE is well-defined.
See also: *Compliers*, *Defiers*, *LATE*

---

## N

### Natural Experiment
A situation where variation in treatment arises from factors outside the researcher's control in a way that mimics randomization. Exploited by DID, RD, and IV designs.

### Never-takers
In instrumental variables settings, units who never take treatment regardless of the instrument value. IV estimates do not apply to never-takers.
See also: *Compliers*, *Always-takers*, *LATE*

### Nuisance Parameter
A parameter that must be estimated but is not the primary target of inference. In causal inference, propensity scores and outcome predictions are often nuisance parameters.
See also: *DDML*

---

## O

### Observational Study
A study where treatment is not randomly assigned by the researcher. Requires stronger assumptions than RCTs to identify causal effects.

### Omitted Variable Bias
Bias in causal effect estimates due to failure to control for confounders. The direction of bias depends on the correlation of the omitted variable with treatment and outcome.
See also: *Confounding*, *Confounder*

### Overlap
The assumption that all units have positive probability of receiving each treatment level: 0 < P(D=1|X) < 1. Necessary for matching and IPW methods.
See also: *Positivity*, *Propensity Score*

---

## P

### Parallel Trends
The DID assumption that treated and control groups would have followed the same trend in the absence of treatment. Can be partially tested using pre-treatment data.
See also: *DID*, *Pre-trends*

### Placebo Test
A test using an outcome that should not be affected by treatment, or a time period before treatment occurred. Failure suggests problems with the identification strategy.

### Positivity
See *Overlap*

### Potential Outcomes
The outcomes that would be observed under different treatment assignments. For binary treatment: Y(0) is the outcome if untreated, Y(1) if treated.
See also: *Counterfactual*, *Rubin Causal Model*

### Pre-trends
The pattern of outcomes before treatment. In DID, similar pre-trends support (but do not prove) the parallel trends assumption.
See also: *Parallel Trends*, *DID*

### Propensity Score
$$e(X) = P(D=1 \mid X)$$
The probability of receiving treatment given observed covariates. Used in matching, weighting, and stratification.
See also: *PSM*, *IPW*, *Overlap*

### PSM (Propensity Score Matching)
Matching treated and control units based on similarity of propensity scores. Reduces dimensionality of matching problem.
See also: *Propensity Score*, *Matching*

---

## R

### Randomization
Random assignment of units to treatment or control. Ensures that treatment is independent of potential outcomes, eliminating confounding.
See also: *RCT*, *Balance*

### RCT (Randomized Controlled Trial)
A study where treatment is randomly assigned by the researcher. The gold standard for causal inference because randomization eliminates confounding.

### Reduced Form
In IV settings, the regression of the outcome directly on the instrument. The reduced form effect divided by the first stage gives the IV estimate.
See also: *Instrumental Variable*, *First Stage*

### Regression Discontinuity (RD)
A design exploiting a cutoff in a running variable that determines treatment. Compares outcomes just above and below the cutoff.
See also: *Sharp RD*, *Fuzzy RD*, *Running Variable*, *Manipulation*

### Relevance
In IV settings, the requirement that the instrument actually affects treatment. Tested by first-stage F-statistic.
See also: *Instrumental Variable*, *First Stage*, *Weak Instrument*

### Rubin Causal Model
The potential outcomes framework for defining causal effects, developed by Donald Rubin building on Neyman's work.
See also: *Potential Outcomes*

### Running Variable
In RD designs, the continuous variable determining treatment assignment at a threshold. Also called the "forcing variable."
See also: *Regression Discontinuity*, *Bandwidth*

---

## S

### Selection Bias
Systematic differences between treated and control groups not due to treatment. Arises from non-random treatment assignment.
See also: *Self-Selection*, *Confounding*

### Selection on Observables
The assumption that conditional on observed covariates, treatment assignment is independent of potential outcomes. Weaker than randomization but necessary for matching/PSM.
See also: *Conditional Independence*, *Unconfoundedness*

### Self-Selection
When units choose their own treatment status, often based on expected benefits. Creates selection bias if selection correlates with potential outcomes.
See also: *Selection Bias*, *ATT*

### Sharp RD
Regression discontinuity design where treatment is deterministically assigned at the cutoff: D=1 if running variable >= cutoff, D=0 otherwise.
See also: *Fuzzy RD*, *Regression Discontinuity*

### Spillover
When one unit's treatment affects another unit's outcome. Violates SUTVA. Also called "interference."
See also: *SUTVA*, *Interference*

### Staggered DID
Difference-in-differences with multiple groups adopting treatment at different times. Requires careful methodology (Callaway-Sant'Anna, Sun-Abraham).
See also: *DID*, *Two-Way Fixed Effects*

### SUTVA (Stable Unit Treatment Value Assumption)
The assumption that (1) there is no interference between units, and (2) there is only one version of treatment. Foundation of the potential outcomes framework.
See also: *Interference*, *Spillover*, *Potential Outcomes*

### Synthetic Control
A method for single-treated-unit studies that constructs a counterfactual as a weighted combination of control units. Weights chosen to match pre-treatment outcomes.

---

## T

### Treatment
The intervention, policy, or exposure whose causal effect is being studied. May be binary, multi-valued, or continuous.
See also: *Treatment Effect*, *Assignment Mechanism*

### Treatment Effect
The causal impact of treatment on an outcome. At the individual level: τᵢ = Y(1) - Y(0). Usually estimated at aggregate level (ATE, ATT, LATE).
See also: *ATE*, *ATT*, *LATE*, *CATE*

### Two-Way Fixed Effects (TWFE)
A regression specification with unit and time fixed effects, commonly used for DID estimation. Has known problems with staggered treatment adoption.
See also: *DID*, *Staggered DID*

### 2SLS (Two-Stage Least Squares)
The standard estimator for instrumental variables. First stage: regress treatment on instrument. Second stage: regress outcome on predicted treatment.
See also: *Instrumental Variable*, *First Stage*

---

## U

### Unconfoundedness
See *Conditional Independence*

---

## W

### Weak Instrument
An instrument with low correlation with treatment (low first-stage F-statistic). Leads to bias toward OLS and unreliable inference. Rule of thumb: F > 10.
See also: *Instrumental Variable*, *First Stage*, *LIML*

---

## Related Documents

- `potential_outcomes.md` - Detailed treatment of potential outcomes framework
- `dags_and_graphs.md` - Graphical causal models and d-separation
- `selection_guide.md` - Choosing the right estimator
- `common_pitfalls.md` - Mistakes to avoid
- `econometrics_vs_ml.md` - When to use ML approaches
