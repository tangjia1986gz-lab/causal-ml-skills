# 校准笔记: Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects

> **技能**: estimator-did
> **论文 ID**: f2b059a83a75bd2f8fe4b4f29f53b865eedc024f
> **年份**: 2018
> **期刊**: The American Economic Review
> **引用数**: 3285

---

## 摘要

Linear regressions with period and group fixed effects are widely used to estimate treatment 
effects. We show that they identify weighted sums of the average treatment effects (ATE) in each 
group and period, with weights that may be negative. Due to the negative weights, the linear 
regression estimand may for instance be negative while all the ATEs are positive. In two articles 
that have used those regressions, half of the weights are negative. We propose another estimator 
that solves this issue. In one of the articles we revisit, it is of a different sign than the linear 
regression estimator.
Clément de Chaisemartin
Department of Economics
University of California at Santa Barbara
Santa Barbara, CA 93106
and NBER
clementdechaisemartin@ucsb.edu
Xavier D'Haultfoeuille
CREST
5 avenue Henry Le Chatelier
91764 Palaiseau cedex
FRANCE
xavier.dhaultfoeuille@ensae.fr
A data appendix is available at http://www.nber.org/data-appendix/w25904

---

## 核心假设

- Assumption 1: (Balanced panel of groups) For all (g, t) ∈{0, ..., g} × {0, ..., t}, Ng,t > 0.

- Assumption 1: requires that no group appears or disappears over time. This assumption is often
satisﬁed. Without it, our results still hold but the notation becomes more complicated as the
denominators of some of the fractions below may then be equal to zero.
Finally, for all g, let Ng,. = Pt
t=0 Ng,t denote the total number of observations in group g. For
all t, let N.,t = Pg
g=0 Ng,t denote the total number of observations in period t. For any variable
Xg,t deﬁned in each (g, t) cell, let Xg,.

- Assumption 2: (Sharp designs with a non-stochastic treatment)
1. For all (g, t) ∈{0, ..., g} × {0, ..., t}, Di,g,t = Dg,t for all i ∈{1, ..., Ng,t}.
2. For all (g, t) ∈{0, ..., g} × {0, ..., t}, Dg,t = E(Dg,t).
Point 1 of

- Assumption 2: requires that units’ treatments do not vary within each (g, t) cell, a
situation we refer to as a sharp design. This is for instance satisﬁed when the treatment is a
group-level variable, for instance a county- or a state-law. This is also mechanically satisﬁed
when Ng,t = 1 for all (g, t). In our literature review in Section 7.1, we ﬁnd that almost 80%
of the papers using two-way ﬁxed eﬀects regressions and published in the AER between 2010
and 2012 consider sharp designs. We ﬁrst

- Assumption 2: requires that the treatment status of each (g, t) cell be non-stochastic.
This assumption fails when the treatment is randomly assigned. However, two-way ﬁxed eﬀects
regressions are more often used in observational than in experimental studies, as in the latter
case one can use simpler regressions to estimate the treatment eﬀect. In observational settings,
whether the treatment should be considered as ﬁxed or stochastic is less clear.
Point 2 of

- Assumption 2: is in line with the modelling framework adopted in some articles studying DID,
see e.g. Abadie (2005). And most importantly, we relax that assumption in Section 4 and in
Section 3.1 of the Web Appendix. We just maintain it for now to ease the exposition.
3.1
A decomposition of βfe as a weighted sum of ATEs under common trends
In this section, we study βfe under the following common trends assumption.

- Assumption 3: (Common trends) For t ≥1, E(Yg,t(0)) −E(Yg,t−1(0)) does not vary across g.
The common trends assumption requires that the expectation of the outcome without the treat-
ment follow the same evolution over time in every group. This assumption is suﬃcient for the
DID estimand to identify the ATT in designs with two groups and two periods, and where only
units in group 1 and period 1 get treated (see, e.g., Abadie, 2005).
For any (g, t) ∈{0, ..., g} × {0, ..., t}, we denote the ATE in 

- Assumption 4: (Staggered adoption designs) For all g, Dg,t ≥Dg,t−1 for all t ≥1.

- Assumption 4: is satisﬁed in applications where groups adopt a treatment at heterogeneous dates
(see e.g. Athey and Stern, 2002). In that design, Borusyak and Jaravel (2017) show that βfe is
more likely to assign a negative weight to treatment eﬀects at the last periods of the panel. This
result is a special case of

- Assumption 5: (w uncorrelated with ∆) P
(g,t):Dg,t=1(Ng,t/N1)(wg,t −1)(∆g,t −∆TR) = 0.
Corollary 2 If Assumptions 1-3 and 5 hold, then βfe = ∆TR.
8 One can show that P
(g,t):Dg,t=1(Ng,t/N1)wg,t = 1.
10


---

## 方法论/识别策略

Fixed-eﬀects OLS regression
13
First-diﬀerence OLS regression
6
Fixed-eﬀects or ﬁrst-diﬀerence OLS regression, with several treatment variables
6
Fixed-eﬀects or ﬁrst-diﬀerence 2LS regression
3
Other regression
5
Panel B. Research design
Sharp design
26
Fuzzy design
7
Panel C. Are there stable groups?
Yes
12
Presumably yes
14
Presumably no
5
No
2
Notes. This table reports the estimation method and the research design used in the 33 papers using two-way
ﬁxed eﬀects regressions published in the AER from 2010 to 2012, and whether those papers have stable groups.
7.2

---

## 估计方法

that solves this issue. In one of the articles we revisit, it is of a different sign than the linear 
regression estimator.
Clément de Chaisemartin
Department of Economics
University of California at Santa Barbara
Santa Barbara, CA 93106
and NBER
clementdechaisemartin@ucsb.edu
Xavier D'Haultfoeuille
CREST
5 avenue Henry Le Chatelier
91764 Palaiseau cedex
FRANCE
xavier.dhaultfoeuille@ensae.fr
A data appendix is available at http://www.nber.org/data-appendix/w25904


1
Introduction
A popular method to estimate the eﬀect of a treatment on an outcome is to compare over time
groups experiencing diﬀerent evolutions of their exposure to treatment. In practice, this idea is
implemented by estimating regressions that control for group and time ﬁxed eﬀects. Hereafter, we
refer to those as two-way ﬁxed eﬀects (FE) regressions. We conducted a survey, and found that
20% of all empirical articles published by the American Economic Review (AER) between 2010
and 2012 have used a two-way FE regression to estimate the eﬀect of a treatment on an outcome.
When the treatment eﬀect is constant across groups and over time, such regressions identify that
eﬀect under the standard “common trends” assumption. However, it is often implausible that
the treatment eﬀect is constant. For instance, the eﬀect of the minimum wage on employment
may vary across US counties, and may change over time. The goal of this paper is to examine
the properties of two-way FE regressions when the constant eﬀect assumption is violated.
We start by assuming that all observations in the same (g, t) cell have the same treatment, as
is for instance the case when the treatment is a county- or a state-level law. We consider the
regression of Yi,g,t, the outcome of unit i in group g at period t on group ﬁxed eﬀects, period
ﬁxed eﬀects, and Dg,t, the value of the treatment in group g at period t. Let βfe denote the
expectation of the coeﬃcient of Dg,t. Then, under the common trends assumption, we show that
if the treatment is binary, βfe identiﬁes a weighted sum of the treatment eﬀect in each group
and at each period:
βfe =
X
(g,t):Dg,t=1
Wg,t∆g,t.
(1)
∆g,t is the average treatment eﬀect (ATE) in group g and period t and the weights Wg,ts sum
to one but may be negative. Negative weights arise because βfe is a weighted average of several
diﬀerence-in-diﬀerences (DID), which compare the evolution of the outcome between consecutive
time periods across pairs of groups. However, the “control group” in some of those comparisons
may be treated at both periods. Then, its treatment eﬀect at the second period gets diﬀerenced
out by the DID, hence the negative weights.
The negative weights are an issue when the ATEs are heterogeneous across groups or periods.
Then, one could have that βfe is negative while all the ATEs are positive. For instance, 1.5 ×
1 −0.5 × 4, a weighted sum of 1 and 4, is strictly negative. We revisit Enikolopov et al. (2011)
and Gentzkow et al. (2011), two articles that have estimated two-wa

---

## 关键公式

未提取到公式

---

## 校准检查清单

- [ ] 识别假设是否完整覆盖
- [ ] 估计方法是否准确描述
- [ ] 诊断检验是否包含
- [ ] 代码实现是否一致
- [ ] 参考文献是否引用

---

## 与现有文档的差异

<!-- 由 CalibrationAgent 自动填写 -->

