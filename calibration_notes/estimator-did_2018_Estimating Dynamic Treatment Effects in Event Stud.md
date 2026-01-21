# 校准笔记: Estimating Dynamic Treatment Effects in Event Studies With Heterogeneous Treatment Effects

> **技能**: estimator-did
> **论文 ID**: 3321239d27f986ef27494d3a6f6f8bbfdfcc9e00
> **年份**: 2018
> **期刊**: Journal of Econometrics
> **引用数**: 3256

---

## 摘要

To estimate the dynamic effects of an absorbing treatment, researchers often use two-way ﬁxed
effects regressions that include leads and lags of the treatment. We show that in settings with variation in
treatment timing across units, the coefﬁcient on a given lead or lag can be contaminated by effects from
other periods, and apparent pretrends can arise solely from treatment effects heterogeneity. We propose
an alternative estimator that is free of contamination, and illustrate the relative shortcomings of two-way
ﬁxed effects regressions with leads and lags through an empirical application.

---

## 核心假设

underlying the causal interpretation of these building blocks to Section 2.
The second main contribution of our paper is to propose a simple regression-based alternative estimation
strategy that produces a more sensible estimand than conventional two-way ﬁxed effects models under het-
erogeneous treatment effects. Our procedure is most similar to Callaway and Sant’Anna (2020a), but has the
4


following differences. First, in the setting where there is no never-treated group, our method uses the last co-
hort to be treated as a control group, whereas Callaway and Sant’Anna (2020a) use the set of not-yet-treated
cohorts. Our method and theirs thus rely on different, but non-nested parallel trends assumptions. Second,
our estimation method can be cast as a regression speciﬁcation and thus may be more familiar to applied
researchers. However, a third difference is that the procedure of Callaway and Sant’Anna (2020a) allows for
conditioning on time-varying covariates. de Chaisemartin and D’Haultfœuille (2020) and Goodman-Bacon
(2018) respectively propose alternative estimators and diagnostic tools for estimation of causal effects in
staggered settings, but do not consider the estimation of the dynamic path of treatment effects as we do.
2
Event studies design
In this section we ﬁrst formalize the “event studies design”. As discussed in Section 2.3, based on how this
term is deployed in the empirical literature, an event study design is a staggered adoption design where units
are 
---
With the above deﬁnitions, we formalize three potential identifying assumptions for outcomes of interest in
our event study design. The ﬁrst assumption is a generalized form of a parallel trends assumption. The sec-
ond assumption requires no anticipation of the treatment. The third assumption imposes no variation across
cohorts. For each assumption, we ﬁrst discuss its meaning and then compare it with similar assumptions
made in the literature interpreting two-way ﬁxed effects regressions. Later in Section 3 we interpret the
relative period coefﬁcients µℓfrom two-way ﬁxed effects regressions under different combinations of these
assumptions.
Assumption 1. (Parallel trends in baseline outcomes.) For all s , t, the E[Y∞
i,t −Y∞
i,s|Ei = e] is the same for
all e ∈supp(Ei).
If an application includes never-treated units so that ∞∈supp(Ei), we need to especially consider
whether these never-treated units satisfy the parallel trends assumption. Never-treated units are likely to
differ from ever-treated units in many ways, and may not share the same evolution of baseline outcomes. If
the never-treated units are unlikely to satisfy the parallel trends assumption, then we should exclude them
from the estimation to avoid violation of this assumption.
While common in the applied literature, the parallel trends assumption is strong and oftentimes violated.
For example, Ashenfelter (1978) documented that participants in job training programs experience a decline
in earnings prior to the 
---
First we show that without any assumptions, we can write µg as a linear combination of differences in trends.
---
Proposition 3. If Assumption 1 (parallel trends) holds and Assumption 2 (no anticipatory behavior in all
periods before the initial treatment) holds, the population regression coefﬁcient µg is a linear combination
of post-treatment CATTe,ℓ′ for all ℓ′ ≥0, with the same weights stated in Proposition 1:
µg =
Õ
ℓ′∈g,ℓ′>0
Õ
e
ωg
e,ℓCATTe,ℓ+
Õ
g′,g,g′∈G
Õ
ℓ′∈g′,ℓ′>0
Õ
e
ωg
e,ℓ′CATTe,ℓ′ +
Õ
ℓ′∈gexcl,ℓ′>0
Õ
e
ωg
e,ℓ′CATTe,ℓ′. (15)
Once we restrict pre-treatment CATTe,ℓ≤0 to be zero under the no anticipatory behavior assumption,
the expression for µg simpliﬁes as terms involving CATTe,ℓ≤0 drop out. However, the second term in the
expression for µg remains unless we further impose treatment effect homogeneity for its summands to cancel
out each other. Thus, µg may be non-zero for pre-treatment periods even if parallel trends holds.
This result immediately implies a shortcoming of using pre-treatment coefﬁcients (i.e. µg where g con-
tains only leads to the treatment ℓ< 0) to test for pretrends. Under the no anticipatory behavior assumption,
15


cohort-speciﬁc treatment effects prior to treatment are all zero: CATTe,ℓ= 0 for all ℓ< 0. Therefore, any
linear combination of these CATTe,ℓis also zero. However, µg is a function of post-treatment CATTe,ℓ′≥0 as
well, even when g only contains elements with ℓ< 0. We revisit this implication in greater depth in Section
---
required for these regressions to yield causally interpretable estimates. For example, Athey
and Imbens (2018), Borusyak and Jaravel (2017), Callaway and Sant’Anna (2020a), de Chaisemartin and
D’Haultfœuille (2020) and Goodman-Bacon (2018) interpret the coefﬁcient on the treatment status when
there is treatment effects heterogeneity and variation in treatment timing. Researchers are often also inter-
ested in dynamic treatment effects, which they estimate by the coefﬁcients µℓassociated with indicators for
being ℓperiods relative to the treatment, in a speciﬁcation that resembles the following:
Yi,t = αi +λt +
Õ
ℓ
µℓ1{t −Ei = ℓ} +υi,t.
(1)
Here Yi,t is the outcome of interest for unit i at time t, Ei is the time when unit i initially receives the binary
absorbing treatment, and αi and λt are the unit and time ﬁxed effects. Units are categorized into different
cohorts based on their initial treatment timing. The relative times ℓ= t −Ei included in (1) cover most of the
possible relative periods, but may still exclude some periods.
The ﬁrst goal of this paper is to uncover potential pitfalls associated with using the estimates of the
relative period coefﬁcients µℓas “reasonable” measures of dynamic treatment effects. We decompose µℓto
show it can be expressed as a linear combination of cohort-speciﬁc effects from both its own relative period
ℓand other relative periods; unless strong assumptions regarding treatment effects homogeneity hold, the
terms that include treatment effe

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

