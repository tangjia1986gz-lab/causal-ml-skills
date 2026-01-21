# 校准笔记: Estimating Dynamic Treatment Effects in Event Studies With Heterogeneous Treatment Effects

> **技能**: causal-forest
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

- Assumption 1: (Parallel trends in baseline outcomes.) For all s , t, the E[Y∞
i,t −Y∞
i,s|Ei = e] is the same for
all e ∈supp(Ei).
If an application includes never-treated units so that ∞∈supp(Ei), we need to especially consider
whether these never-treated units satisfy the parallel trends assumption. Never-treated units are likely to
differ from ever-treated units in many ways, and may not share the same evolution of baseline outcomes. If
the never-treated units are unlikely to satisfy the para

- Assumption 2: (No anticipatory behavior prior to treatment.) There is no treatment effect in pre-treatment
periods i.e. E[Y e
i,e+ℓ−Y∞
i,e+ℓ| Ei = e] = 0 for all e ∈supp(Ei) and all ℓ< 0.

- Assumption 2: requires potential outcomes in any ℓperiods before treatment to be equal to the baseline
outcome on average as in Malani and Reif (2015) and Botosaru and Gutierrez (2018). This is most plausible
if the full treatment paths are not known to units. If they have private knowledge of the future treatment path
they may change their behavior in anticipation and thus the potential outcome prior to treatment may not
represent baseline outcomes. For example, Hendren (2017) shows that knowle

- Assumption 2: , which holds for pre-periods in a subset of pre-treatment periods.
Depending on the application, it may still be plausible to assume no anticipation until K periods before the
treatment.
The no anticipation assumption proposed by Athey and Imbens (2018) is a deterministic condition which
stipulates that Y e
i,e+ℓ= Y∞
i,e+ℓfor all units i and e and ℓ< 0. By taking the “fully dynamic” speciﬁcation as
their DGP, Borusyak and Jaravel (2017) allow anticipation by including pre-trends i

- Assumption 3: (Treatment effect homogeneity.) For each relative period ℓ, CATTe,ℓdoes not depend on
cohort e and is equal to ATTℓ.
8

- Assumption 3: requires that each cohort experiences the same path of treatment effects. Treatment effects
need to be the same across cohorts in every relative period for homogeneity to hold, whereas for heterogene-
ity to occur, treatment effects only need to differ across cohorts in one relative period. The assumption of
treatment effect homogeneity is therefore strong, and in Section 3.4.1, we describe how it can be violated in
applied settings.
Our notion of treatment effect homogeneity does 

- Assumption 1: (parallel trends) only, the population regression coefﬁcient on the
indicator for relative period bin g is a linear combination of CATTe,ℓ∈g as well as CATTe,ℓ′ from other
14

- Assumption 1: , the terms in

- Assumption 1: (parallel trends) holds and

- Assumption 2: (no anticipatory behavior in all
periods before the initial treatment) holds, the population regression coefﬁcient µg is a linear combination
of post-treatment CATTe,ℓ′ for all ℓ′ ≥0, with the same weights stated in


---

## 方法论/识别策略

We propose a new estimation method that is robust to treatment effects heterogeneity. The goal of our
method is to estimate a weighted average of CATTe,ℓfor ℓ∈g with reasonable weights, namely weights that
sum to one and are non-negative. In particular, we focus on the following weighted average of CATTe,ℓ,
where the weights are shares of cohorts that experience at least ℓperiods relative to treatment, normalized
by the size of g:
νg = 1
|g|
Õ
ℓ∈g
Õ
e
CATTe,ℓPr{Ei = e | Ei ∈[−ℓ,T −ℓ]}.
(25)
One can aggregate CATTe,ℓto form other parameters of interest, such as those proposed by Callaway and
Sant’Anna (2020a). We focus on the above aggregation νg since our goal is to improve the non-convex and
non-zero weighting in µg. The weights in νg are guaranteed to be convex and have an interpretation as the
representative shares corresponding to each CATTe,ℓ. Thus, our alternative estimator bνg improves upon the
two-way ﬁxed effects estimator bµg by estimating an interpretable weighted average of CATTe,ℓ∈g.
Our method proceeds by replacing each component in νg with its consistent estimator. We ﬁrst estimate
21


each CATTe,ℓusing an interacted two-way ﬁxed effects regression, then estimate the weight Pr{Ei = e |
Ei ∈[−ℓ,T −ℓ]} using their sample analogs. In the ﬁnal step, we average over the cohort-speciﬁc estimates
associated with relative period ℓ. This method has a similar ﬂavor as the method proposed by Gibbons et
al. (2019). They ﬁrst use an interacted model to estimate the treatment effect for each ﬁxed effect group;
the resulting group-speciﬁc estimates are averaged to provide the ATE. Their method improves ﬁxed effects
regressions in a cross-sectional setting, and our method builds on theirs by improving two-way ﬁxed effects
regressions in a panel setting. We therefore follow their terminology in calling our alternative estimator an
“interaction-weighted” estimator.
4.1
Interaction-weighted estimator
We describe the estimation procedure in three steps (with more detailed deﬁnitions stated in Deﬁnition 4 of
Online Appendix B).
Step 1. We estimate CATTe,ℓusing a linear two-way ﬁxed effects speciﬁcation that interacts relative
period indicators with cohort indicators, excluding indicators for cohorts from some set C:
Yi,t = αi +λt +
Õ
e<C
Õ
ℓ,−1
δe,ℓ(1{Ei = e} · Dℓ
i,t)+ϵi,t.
(26)
The exact speciﬁcation depends on the cohort shares for a given application. If there is a never-treated
cohort, i.e. ∞∈supp{Ei}, then we may set C = {∞} and estimate regression (26) on all observations. If
there are no never-treated units, i.e. ∞< supp{Ei}, then we may set C = {max{Ei}}, i.e. the latest-treated
cohort and estimate regression (26) on observations from t = 0,...,max{Ei} −1. Lastly, if there is a cohort
that is always treated, i.e. 0 ∈supp{Ei}, then we need to exclude this cohort from estimation.
The coefﬁcient estimator bδe,ℓfrom regression (26) is a DID estimator for CATTe,ℓwith particular
choices of pre-periods and control cohorts. As DID is likely a familia

---

## 估计方法

未提取到估计部分

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

