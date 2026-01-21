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

holds by simply setting X = 1.
Assumption 3 states that once an individual becomes treated, that individual will also be
treated in the next period. With regards to the minimum wage application, Assumption 3 says
that once a state increases its minimum wage above the federal level, it does not decrease it back
to the federal level during the analyzed period. Moreover, this assumption is consistent with most
DID setups that exploit the enacting of a policy in some location while the policy is not enacted
in another location.3
Finally, Assumption 4 states that a positive fraction of the population started to be treated in
period g, and that, for any possible value of the covariates X, there is some positive probability
3One could potentially relax this assumption by forming groups on the basis of having the entire path of
treatment status being the same and then perform the same analysis that we do.
8


that an individual is not treated.4 This is a standard covariate overlap condition, see e.g. Heckman
et al. (1997, 1998), Blundell et al. (2004), Abadie (2005).
Remark 1. In some applications, eventually all units are treated, implying that C is never equal
to one. In such cases one can consider the “not yet treated” (Dt = 0) as a control group instead of
the “never treated” (C = 1). We consider this case in Appendix C in the Supplementary Appendix,
which resembles the event study research design, see e.g. Borusyak and Jaravel (2017).
---
” holds potentially only after conditioning on observed
covariates. We propose a simple two-step estimation strategy, establish the asymptotic prop-
erties of the proposed estimators, and prove the validity of a computationally convenient
bootstrap procedure. Furthermore we propose a semiparametric data-driven testing proce-
dure to assess the credibility of the DID design in our context. Finally, we analyze the eﬀect
of the minimum wage on teen employment from 2001-2007. By using our proposed methods
we confront the challenges related to variation in the timing of the state-level minimum wage
policy changes. Open-source software is available for implementing the proposed methods.
JEL: C14, C21, C23, J23, J38.
Keywords: Diﬀerence-in-Diﬀerences, Multiple Periods, Variation in Treatment Timing, Pre-
Testing, Minimum Wage.
∗We thank Andrew Goodman-Bacon, Federico Gutierrez, Na’Ama Shenhav, and seminar participants at the
2017 Southern Economics Association for valuable comments. Code to implement the methods proposed in the
paper are available in the R package did which is available on CRAN.
†Department of Economics, Temple University. Email: brantly.callaway@temple.edu
‡Department of Economics, Vanderbilt University. Email: pedro.h.santanna@vanderbilt.edu
1


1
Introduction
Diﬀerence-in-Diﬀerences (DID) has become one of the most popular designs used to evaluate the
causal eﬀect of policy interventions.
In its canonical format, there are two time periods and
two groups: in the 
---
.
Assumption 1 (Sampling). {Yi1, Yi2, . . . YiT , Xi, Di1, Di2, . . . , DiT }n
i=1 is independent and identi-
cally distributed (iid).
Assumption 2 (Conditional Parallel Trends). For all t = 2, . . . , T , g = 2, . . . , T such that g ≤t,
E[Yt(0) −Yt−1(0)|X, Gg = 1] = E[Yt(0) −Yt−1(0)|X, C = 1] a.s..
2Existence of expectations is assumed throughout.
7


Assumption 3 (Irreversibility of Treatment). For t = 2, . . . , T ,
Dt = 1 implies that Dt+1 = 1
Assumption 4 (Overlap). For all g = 2, . . . , T , P (Gg = 1) > 0 and pg(X) < 1 a.s..
Assumption 1 implies that we are considering the case with panel data. The extension to
the case with repeated cross sections is relatively simple and is developed in Appendix B in the
Supplementary Appendix.
Assumption 2, which we refer to as the (conditional) parallel trends assumption throughout the
paper, is the crucial identifying restriction for our DID model, and it generalizes the two-period
DID assumption to the case where it holds in all periods and for all groups; see e.g. Heckman et al.
(1997, 1998), Blundell et al. (2004), and Abadie (2005). It states that, conditional on covariates,
the average outcomes for the group ﬁrst treated in period g and for the control group would
have followed parallel paths in the absence of treatment. We require this assumption to hold for
all groups g and all time periods t such that g ≤t; that is, it holds in all periods after group
g is ﬁrst treated. It is important to emphasize that the parallel trend
---
1 - 4 and for 2 ≤g ≤t ≤T , the group-time average treatment
eﬀect for group g in period t is nonparametrically identiﬁed, and given by
ATT (g, t) = E






Gg
E [Gg] −
pg (X) C
1 −pg (X)
E
 pg (X) C
1 −pg (X)




(Yt −Yg−1)

.
(2.1)
---
1 - 4, a simple weighted average of “long diﬀerences”
of the outcome variable recovers the group-time average treatment eﬀect. The weights depends
on the generalized propensity score pg (X), and are normalized to one.
The intuition for the
weights is simple. One takes observations from the control group and group g, omitting other
groups and then weights up observations from the control group that have characteristics similar
to those frequently found in group g and weights down observations from the control group that
have characteristics that are rarely found in group g. Such a reweighting procedures guarantees
that the covariates of group g and the control group are balanced. Interestingly, in the standard
DID setup of two periods only, E [p2 (X) C/ (1 −p2 (X))] = E [G2], and the results of Theorem 1
reduces to Lemma 3.1 in Abadie (2005).
4In our application on the minimum wage, we must take somewhat more care here as there are some periods
where there are no states that increase their minimum wage. In this case, let G denote the set of ﬁrst treatment
times with G ⊆{1, . . . , T }. Then, one can compute ATT(g, t) for groups g ∈G with g ≤t. This is a simple
complication to deal with in practice, so we consider the notationally more convenient case where there are some
individuals treated in all periods (possibly excluding period 1) in the main text of the paper.
9


To shed light on the role of the “long diﬀerence”, we give a sketch of how this argument works
in the uncondi

---

## 方法论/识别策略

We ﬁrst introduce the notation we use throughout the article. We consider the case with T periods

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
results. We discuss estimation and inference procedures for the treatment eﬀects of interest in
Section 3. Section 4 describes our pre-tests for the credibility of the conditional parallel trends
assumption.
We revisit the eﬀect of minimum wage on employment in Section 5.
Section 6
concludes. All proofs are gathered in the Appendix.
2
Identiﬁcation
2.1
Framework
We ﬁrst introduce the notation we use throughout the article. We consider the case with T periods
and denote a particular time period by t where t = 1, . . . , T . In a standard DID setup, T = 2 and
no one is treated in period 1. Let Dt be a binary variable equal to one if an individual is treated in
period t and equal to zero otherwise. Also, deﬁne Gg to be a binary variable that is equal to one
if an individual is ﬁrst treated in period g, and deﬁne C as a binary variable that is equal to one
for individuals in the control group – these are individuals who are never treated so the notation
6


is not indexed by time. For each individual, exactly one of the Gg or C is equal to one. Denote
the generalized propensity score as pg(X) = P(Gg = 1|X, Gg + C = 1). Note that pg(X) indicates
the probability that an individual is treated conditional on having covariates X and conditional
on being a member of group g or the control group. Finally, let Yt (1) and Yt (0) be the potential
outcome at time t with and without treatment, respectively. The observed outcome in each period
can be expressed as Yt = DtYt (1) + (1 −Dt) Yt (0) .
Given that Yt (1) and Yt (0) cannot be observed for the same individual at the same time,
researchers often focus on estimating some function of the potential outcomes. For instance, in
the standard DID setup, the most popular treatment eﬀect parameter is the average treatment
eﬀect on the treated, denoted by2
ATT = E[Y2(1) −Y2(0)|G2 = 1].
Unlike the two period and two group case, when there are more than two periods and variation
in treatment timing, it is not obvious which is the main causal parameter of i

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

