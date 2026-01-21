# 校准笔记: Difference-in-Differences with Multiple Time Periods

> **技能**: estimator-did
> **论文 ID**: c7ca83353b0269fc7d04fa06a28c7f323217758e
> **年份**: 2018
> **期刊**: Journal of Econometrics
> **引用数**: 4897

---

## 摘要

Diﬀerence-in-Diﬀerences (DID) is one of the most important and popular designs for eval-
uating causal eﬀects of policy changes. In its standard format, there are two time periods
and two groups: in the ﬁrst period no one is treated, and in the second period a “treatment
group” becomes treated, whereas a “control group” remains untreated. However, many em-
pirical applications of the DID design have more than two periods and variation in treatment
timing. In this article, we consider identiﬁcation and estimation of treatment eﬀect param-
eters using DID with (i) multiple time periods, (ii) variation in treatment timing, and (iii)
when the “parallel trends assumption” holds potentially only after conditioning on observed
covariates. We propose a simple two-step estimation strategy, establish the asymptotic prop-
erties of the proposed estimators, and prove the validity of a computationally convenient
bootstrap procedure. Furthermore we propose a semiparametric data-driven testing proce-
dure to assess the credibility of the DID design in our context. Finally, we analyze the eﬀect
of the minimum wage on teen employment from 2001-2007. By using our proposed methods
we confront the challenges related to variation in the timing of the state-level minimum wage
policy changes. Open-source software is available for implementing the proposed methods.

---

## 核心假设

- Assumption 1: (Sampling). {Yi1, Yi2, . . . YiT , Xi, Di1, Di2, . . . , DiT }n
i=1 is independent and identi-
cally distributed (iid).

- Assumption 2: (Conditional Parallel Trends). For all t = 2, . . . , T , g = 2, . . . , T such that g ≤t,
E[Yt(0) −Yt−1(0)|X, Gg = 1] = E[Yt(0) −Yt−1(0)|X, C = 1] a.s..
2Existence of expectations is assumed throughout.
7

- Assumption 3: (Irreversibility of Treatment). For t = 2, . . . , T ,
Dt = 1 implies that Dt+1 = 1

- Assumption 4: (Overlap). For all g = 2, . . . , T , P (Gg = 1) > 0 and pg(X) < 1 a.s..

- Assumption 1: implies that we are considering the case with panel data. The extension to
the case with repeated cross sections is relatively simple and is developed in Appendix B in the
Supplementary Appendix.

- Assumption 2: , which we refer to as the (conditional) parallel trends assumption throughout the
paper, is the crucial identifying restriction for our DID model, and it generalizes the two-period
DID assumption to the case where it holds in all periods and for all groups; see e.g. Heckman et al.
(1997, 1998), Blundell et al. (2004), and Abadie (2005). It states that, conditional on covariates,
the average outcomes for the group ﬁrst treated in period g and for the control group would
have follow

- Assumption 3: states that once an individual becomes treated, that individual will also be
treated in the next period. With regards to the minimum wage application,

- Assumption 3: says
that once a state increases its minimum wage above the federal level, it does not decrease it back
to the federal level during the analyzed period. Moreover, this assumption is consistent with most
DID setups that exploit the enacting of a policy in some location while the policy is not enacted
in another location.3
Finally,

- Assumption 4: states that a positive fraction of the population started to be treated in
period g, and that, for any possible value of the covariates X, there is some positive probability
3One could potentially relax this assumption by forming groups on the basis of having the entire path of
treatment status being the same and then perform the same analysis that we do.
8

- Assumption 5: For all g = 2, . . . , T , (i) there exists a known function Λ : R →[0, 1] such that
pg(X) = P(Gg = 1|X, Gg + C = 1) = Λ(X′π0
g); (ii) π0
g ∈int(Π), where Π is a compact subset of
Rk; (iii) the support of X, X, is a subset of a compact set S, and E[XX′|Gg + C = 1] is positive
deﬁnite; (iv) let U = {x′π : x ∈X, π ∈Π} ; ∀u ∈U, ∃ε > 0 such that Λ (u) ∈[ε, 1 −ε] , Λ (u) is
strictly increasing and twice continuously diﬀerentiable with ﬁrst derivatives bounded away from
zero and inﬁnity,


---

## 方法论/识别策略

Yit = αt + ci + βDit + θXi + ϵit,
where Yit is the outcome of interest, αt is a time ﬁxed eﬀect, ci is an individual/group ﬁxed
eﬀect, Dit is a treatment indicator that takes value one if an individual i is treated at time
t and zero otherwise, Xi is a vector of observed characteristics, and ϵit is an error term, and
interpret β as the causal eﬀect of interest. Despite the popularity of this approach, Wooldridge
(2005), Chernozhukov et al. (2013), de Chaisemartin and D’Haultfoeuille (2016), Borusyak and
Jaravel (2017), Goodman-Bacon (2017) and S loczy´nski (2017) have shown that once one allows
for heterogeneous treatment eﬀects, β does not represent an easy to interpret average treatment
1We thank Andrew Goodman-Bacon for sharing with us this statistic.
2


eﬀect parameter. As a consequence, inference about the eﬀectiveness of a given policy can be
misleading when based on such a two-way ﬁxed eﬀects regression model.
In this article we aim to ﬁll this important gap and consider identiﬁcation and inference proce-
dures for average treatment eﬀects in DID models with (i) multiple time periods, (ii) variation in
treatment timing, and (iii) when the parallel trends assumption holds potentially only after con-
ditioning on observed covariates. First, we provide conditions under which the average treatment
eﬀect for group g at time t is nonparametrically identiﬁed, where a “group” is deﬁned by when
units are ﬁrst treated. We call these causal parameters group-time average treatment eﬀects.
Second, although these disaggregated group-time average treatment eﬀects can be of interest
by themselves, in some applications there are perhaps too many of them, potentially making the
analysis of the eﬀectiveness of the policy intervention harder, particularly when the sample size
is moderate. In such cases, researchers may be interested in summarizing these disaggregated
causal eﬀects into a single, easy to interpret, causal parameter. We suggest diﬀerent ideas for
combining the group-time average treatment eﬀects, depending on whether one allows for (a)
selective treatment timing, i.e., allowing, for example, the possibility that individuals with the
largest beneﬁts from participating in a treatment choose to become treated earlier than those
with a smaller beneﬁt; (b) dynamic treatment eﬀects – where the eﬀect of a treatment can depend
on the length of exposure to the treatment; or (c) calendar time eﬀects – where the eﬀect of
treatment may depend on the time period. Overall, we note that the best way to aggregate the
group-time average treatment eﬀects is likely to be application speciﬁc. Aggregating group-time
parameters is also likely to increase statistical power.
Third, we develop the asymptotic properties for a semiparametric two-step estimator for the
group-time average treatment eﬀects, and for the diﬀerent aggregated causal parameters. Estimat-
ing these treatment eﬀects involves estimating a generalized propensity score for each group g, and
using th

---

## 估计方法

is based on stabilized (normalized) weights, whereas his proposed estimator is of the Horvitz and
Thompson (1952) type. As the simulations results in Busso et al. (2014) show, stabilized weights
can lead to important ﬁnite sample improvements when compared to Horvitz and Thompson
(1952) type estimators.
Our pre-test for the plausibility of the conditional parallel trends assumption is also related
to many papers in the goodness-of-ﬁt literature, including Bierens (1982), Bierens and Ploberger
(1997), Stute (1997), Stinchcombe and White (1998), Escanciano (2006a,b, 2008), Sant’Anna
(2017), and Sant’Anna and Song (2018); for a recent overview, see Gonz´alez-Manteiga and Cru-
jeiras (2013). Despite the similarities, we seem to be the ﬁrst to realize that such a procedure could
be used to pre-test for the reliability of the conditional parallel trends identiﬁcation assumption.
The remainder of this article is organized as follows. Section 2 presents our main identiﬁcation

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

