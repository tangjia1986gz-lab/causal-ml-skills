# 校准笔记: Two-way Fixed Effects Regressions with Several Treatments

> **技能**: estimator-did
> **论文 ID**: 69b4c08723a31375f18bc70d5eed114ace2f6e5e
> **年份**: 2020
> **期刊**: Social Science Research Network
> **引用数**: 40

---

## 摘要

We study two-way-ﬁxed-eﬀects regressions (TWFE) with several treatment variables.
Under a parallel trends assumption, we show that the coeﬃcient on each treatment iden-
tiﬁes a weighted sum of that treatment’s eﬀect, with possibly negative weights, plus a
weighted sum of the eﬀects of the other treatments. Thus, those estimators are not robust
to heterogeneous eﬀects and may be contaminated by other treatments’ eﬀects. We further
show that omitting a treatment from the regression can actually reduce the estimator’s
bias, unlike what would happen under constant treatment eﬀects. We propose an alterna-
tive diﬀerence-in-diﬀerences estimator, robust to heterogeneous eﬀects and immune to the
contamination problem. In the application we consider, the TWFE regression identiﬁes a
highly non-convex combination of eﬀects, with large contamination weights, and one of its
coeﬃcients signiﬁcantly diﬀers from our heterogeneity-robust estimator.
(

---

## 核心假设

accounting for such covariates, de Chaisemartin and D’Haultfœuille (2020) show that TWFE
regressions with one treatment and some controls identify a weighted sum of the treatment
4


eﬀects across all treated (g, t) cells.
Our decomposition results are related to, but diﬀerent
from, that result.
The weighted sum in Theorem S4 of de Chaisemartin and D’Haultfœuille
(2020) is identical to the ﬁrst weighted sum in Theorem 2 below. On the other hand, the sec-
ond weighted sum in Theorem 2, the contamination term, does not appear in Theorem S4 of
de Chaisemartin and D’Haultfœuille (2020). This is because the parallel trend assumptions are
not the same in the two results. When the other variables in the regression are treatments rather
than covariates (see below for the diﬀerence between a treatment and a covariate), one can show
that the parallel trends condition in Theorem S4 implicitly assumes that the eﬀect of the other
treatments is constant, which is why the contamination term disappears.
---
In this section, we start by considering the following identifying assumption. Recall that Yg,t(d)
denotes the potential outcome of g at t, if the treatment vector is equal to d.
Assumption 4 (Strong exogeneity and common trends from t −1 to t, conditional on Dg,t−1)
For all (g, t) ∈{1, ..., G} × {2, ..., T} and all d ∈{0, 1}K,
1. E(Yg,t(d)−Yg,t−1(d)|Dg,1, ..., Dg,t−2, Dg,t−1 = d, Dg,t, ..., Dg,T) = E(Yg,t(d)−Yg,t−1(d)|Dg,t−1 =
d).
2. E(Yg,t(d) −Yg,t−1(d)|Dg,t−1 = d) does not vary across g.
Like Assumption 3, Assumption 4 imposes both a strong exogeneity and a parallel trends condi-
tion. The strong exogeneity condition requires that groups’ t −1-to-t outcome evolution, in the
counterfactual scenario where their period-t treatments all remain at their t −1 value, be mean
independent of their treatments at every period other than t−1. The parallel trends assumption
requires that groups with the same period-t−1 treatments have the same counterfactual trends.
Then, consider a group whose ﬁrst treatment changes between t −1 and t, but whose other
treatments remain constant. Under Assumption 4, the t−1-to-t evolution of its outcome had its
ﬁrst treatment not changed is identiﬁed by the outcome evolution of groups whose treatments
all remain constant and with the same period-t −1 treatments.
We now compare our new assumption, Assumption 4, to the more standard Assumption 3.
The two assumptions are non-nested, and there are two main diﬀerences between them. First,
Assumption 3 requi
---
4, one can unbiasedly estimate
δ1 = E


X
(g,t)∈S1
Ng,t
NS1
∆1
g,t

,
the average eﬀect of moving the ﬁrst treatment from 0 to 1 while keeping all other treatments
at their observed value, across all switchers.9
δ1 may diﬀer from δATT, arguably a more natural target parameter. The two parameters apply
to diﬀerent and non-nested sets of (g, t) cells. Let T1 = {(g, t) : D1
g,t = 1}. δ1 is the average of
∆1
g,t across all cells in S1. δATT is the average eﬀect of ∆1
g,t across all cells in T1. The following
---
, we show that the coeﬃcient on each treatment iden-
tiﬁes a weighted sum of that treatment’s eﬀect, with possibly negative weights, plus a
weighted sum of the eﬀects of the other treatments. Thus, those estimators are not robust
to heterogeneous eﬀects and may be contaminated by other treatments’ eﬀects. We further
show that omitting a treatment from the regression can actually reduce the estimator’s
bias, unlike what would happen under constant treatment eﬀects. We propose an alterna-
tive diﬀerence-in-diﬀerences estimator, robust to heterogeneous eﬀects and immune to the
contamination problem. In the application we consider, the TWFE regression identiﬁes a
highly non-convex combination of eﬀects, with large contamination weights, and one of its
coeﬃcients signiﬁcantly diﬀers from our heterogeneity-robust estimator.
(JEL C21, C23)
1
Introduction
To estimate treatment eﬀects, researchers often use panels of groups (e.g. counties, regions), and
estimate two-way ﬁxed eﬀect (TWFE) regressions, namely regressions of the outcome variable
on group and time ﬁxed eﬀects and the treatment. de Chaisemartin and D’Haultfœuille (2020)
∗Several of this paper’s ideas arose during conversations with Enrico Cantoni, Angelica Meinhofer, Vincent
Pons, Jimena Rico-Straﬀon, Marc Sangnier, Oliver Vanden Eynde, and Liam Wren-Lewis who shared with us
their interrogations, and sometimes their referees’ interrogations, on two-way ﬁxed eﬀects regressions with several
treatments. We are grateful to the
---
, TWFE regressions with one treatment identify a weighted
sum of the treatment eﬀects of treated (g, t) cells, with weights that may be negative and sum
to one (see de Chaisemartin and D’Haultfœuille, 2020; Borusyak and Jaravel, 2017). Because of
the negative weights, the treatment coeﬃcient in such regressions is not robust to heterogeneous
treatment eﬀects across groups and time periods: it may be, say, negative, even if the treatment
eﬀect is strictly positive in every (g, t) cell.
However, in 18% of the TWFE papers published in the AER from 2010 to 2012, the TWFE
regression has several treatment variables. By including several treatments, researchers hope to
estimate the eﬀect of each treatment holding the other treatments constant. For instance, when
studying the eﬀect of marijuana laws, as in Meinhofer et al. (2021), one may want to separate the
eﬀect of medical and recreational laws. To do so, one may estimate a regression of the outcome
of interest in state g and year t on state ﬁxed eﬀects, year ﬁxed eﬀects, an indicator for whether
state g has a medical law in year t, and an indicator for whether state g has a recreational law
in year t.
In this paper, we investigate what TWFE regressions with several treatments identify.
We
show that under a parallel trends assumption, the coeﬃcient on each treatment identiﬁes the
sum of two terms. The ﬁrst term is a weighted sum of the eﬀect of that treatment in each
group and period, with weights that may be negative and sum to o

---

## 方法论/识别策略



---

## 估计方法

is robust to heterogeneous eﬀects across groups of all treatments. ii) ensures that it is robust to
heterogeneous eﬀects over time of all treatments.
Our estimator’s robustness may come at a high price in terms of external validity and statistical
precision. For instance, in our application in Section 6, we can only match a small number
of switchers to valid control groups meeting i) and ii). Then, there may be internal-external
3


validity and bias-variance trade-oﬀs between our new estimator and less robust estimators, such
as the DIDM estimator in de Chaisemartin and D’Haultfœuille (2020) or TWFE regressions with
several treatments. To account for the fact our new estimator may sometimes be estimated on
a small sample of groups, we propose, in addition to a standard conﬁdence interval that is

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

