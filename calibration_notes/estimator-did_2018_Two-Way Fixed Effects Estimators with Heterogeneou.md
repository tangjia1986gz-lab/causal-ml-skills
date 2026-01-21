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

Assumption 2 (Sharp designs with a non-stochastic treatment)
1. For all (g, t) ∈{0, ..., g} × {0, ..., t}, Di,g,t = Dg,t for all i ∈{1, ..., Ng,t}.
2. For all (g, t) ∈{0, ..., g} × {0, ..., t}, Dg,t = E(Dg,t).
Point 1 of Assumption 2 requires that units’ treatments do not vary within each (g, t) cell, a
situation we refer to as a sharp design. This is for instance satisﬁed when the treatment is a
group-level variable, for instance a county- or a state-law. This is also mechanically satisﬁed
when Ng,t = 1 for all (g, t). In our literature review in Section 7.1, we ﬁnd that almost 80%
of the papers using two-way ﬁxed eﬀects regressions and published in the AER between 2010
and 2012 consider sharp designs. We ﬁrst focus on this special case given its prevalence, before
turning to fuzzy designs in the next section.
Point 2 of Assumption 2 requires that the treatment status of each (g, t) cell be non-stochastic.
This assumption fails when the treatment is randomly assigned. However, two-way ﬁxed eﬀects
regressions are more often used in observational than in experimental studies, as in the latter
case one can use simpler regressions to estimate the treatment eﬀect. In observational settings,
whether the treatment should be considered as ﬁxed or stochastic is less clear.
Point 2 of
Assumption 2 is in line with the modelling framework adopted in some articles studying DID,
see e.g. Abadie (2005). And most importantly, we relax that assumption in Section 4 and in
Section 3.1 of the W
---
7 and 8 are equivalent, and W pl
TC closely mimicks WTC: W pl
TC is equal to WTC, after replacing
the mean outcome in group g and at period t by its lagged value. Outside of staggered adoption
designs, some of the (g, t) cells that are used in the computation of WTC are excluded from
that of W pl
TC, because their treatment changes from t −2 to t −1. Finally, W pl
TC compares the
trends of switching and stable groups one period before the switch. It is easy to deﬁne other
placebo estimands comparing those trends, say, two or three periods before the switch. The
corresponding placebo estimators are computed by the fuzzydid and did_multipleGT Stata
package.
10See also Callaway and Sant’Anna (2018), who propose another placebo test in staggered adoption designs.
14


4
Results in fuzzy designs
In this section, the research design may be fuzzy: the treatment may vary within (g, t) cells.
For instance, Enikolopov et al. (2011) study the eﬀect of having access to an independent TV
channel in Russia, and in each Russian region some people have access to that channel while
other people do not. The treatment may also be stochastic.
As the numbers 1,...,Ng,t assigned to the observations of a (g, t) cell play no role, one can
always assume that those numbers are randomly assigned. Therefore, we assume hereafter that
conditional on D, the vector stacking together all the Dg,ts, all the variables indexed by i (e.g.,
Di,g,t or Yi,g,t(0)) are identically distributed within each (g, t) cell.
---
. However, it is often implausible that
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
The negative weights are an issue when the ATEs are
---
1-3 hold. Then,
βfe =
X
(g,t):Dg,t=1
Ng,t
N1
wg,t∆g,t.
To illustrate this theorem, we consider a simple example of a staggered adoption design with
two groups and three periods, and where group 0 is only treated at period 2, while group 1 is
treated both at periods 1 and 2. We also assume that Ng,t/Ng,t−1 does not vary across g: all
groups experience the same growth of their number of observations from t−1 to t, a requirement
that is for instance satisﬁed when the data is a balanced panel. Then, one can show that
εg,t = Dg,t −Dg,. −D.,t + D.,.,
thus implying that
ε0,2 = 1 −1/3 −1 + 1/2 = 1/6,
ε1,1 = 1 −2/3 −1/2 + 1/2 = 1/3,
ε1,2 = 1 −2/3 −1 + 1/2 = −1/6.
The residual is negative in group 1 and period 2, because the regression predicts a treatment
probability larger than one in that cell, a classic extrapolation problem with linear regressions.
Then, it follows from Theorem 1 and some algebra that under the common trends assumption,
βfe = 1/2∆0,2 + ∆1,1 −1/2∆1,2.
5εg,t arises from a regression where the dependent and independent variables only vary at the (g, t) level.
Therefore, all the units in the same (g, t) cell have the same value of εg,t.
7


βfe is equal to a weighted sum of the ATEs in group 0 at period 2, group 1 at period 1, and
group 1 at period 2, the three treated (g, t) cells.
However, the weight assigned to each ATE diﬀers from the proportion that the corresponding
cell accounts for in the population of treated observations. Therefore, βfe is not equal to ∆TR,

---
, if the ∆g,ts are
heterogeneous. In the corollary below, we propose two summary measures that can be used to
assess how serious that concern is. For any variable Xg,t deﬁned in each (g, t) cell we let X
denote the vector (Xg,t)(g,t)∈{0,...,g}×{0,...,t} collecting the values of that variable in each (g, t) cell.
For instance, ∆denotes the vector (∆g,t)(g,t)∈{0,...,g}×{0,...,t} collecting the ATE in each of the
(g, t) cells. Let
σ(∆) =


X
(g,t):Dg,t=1
Ng,t
N1
 ∆g,t −∆TR2


1/2
,
σ(w) =


X
(g,t):Dg,t=1
Ng,t
N1
(wg,t −1)2


1/2
.
7Borusyak and Jaravel (2017) assume that the treatment eﬀect of cell (g, t) only depends on the number of
periods since group g has started receiving the treatment, whereas Proposition 1 does not rely on that assumption.
9


σ(∆) is the standard deviation of the ATEs, and σ(w) is the standard deviation of the w-
weights,8 across the treated (g, t) cells. Let n = #{(g, t) : Dg,t = 1} denote the number of
treated cells.
For every i ∈{1, ..., n}, let w(i) denote the ith largest of the weights of the
treated cells: w(1) ≥w(2) ≥... ≥w(n), and let N(i) and ∆(i) be the number of observations
and the ATE of the corresponding cell. Then, for any k ∈{1, ..., n}, let Pk = P
i≥k N(i)/N1,
Sk = P
i≥k(N(i)/N1)w(i) and Tk = P
i≥k(N(i)/N1)w2
(i).
Corollary 1 Suppose that Assumptions 1-3 hold.
1. If σ(w) > 0, the minimal value of σ(∆) compatible with βfe and ∆TR = 0 is
σfe = |βfe|
σ(w).
2. If βfe ̸= 0 and at least one of the wg,t weights is strictly negative, 

---

## 方法论/识别策略



---

## 估计方法



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

<!-- 手动填写或自动对比后填写 -->

