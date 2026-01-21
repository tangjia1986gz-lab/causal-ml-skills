# 校准笔记: Unobservable Selection and Coefficient Stability: Theory and Validation

> **技能**: estimator-psm
> **论文 ID**: d0f95fa4ba2f5b01d7e107b20a038ba44199c010
> **年份**: 2013
> **期刊**: 
> **引用数**: 305

---

## 摘要

A common heuristic for evaluating robustness of results to omitted variable bias is to look at coefficient
movements after inclusion of controls. This heuristic is informative only if selection on observables
is proportional to selection on unobservables. I formalize this link, drawing on theory in Altonji, Elder
and Taber (2005) and show how, with this assumption, coefficient movements, along with movements
in R-squared values, can be used to calculate omitted variable bias. I discuss empirical implementation
and describe a formal bounding argument to replace the coefficient movement heuristic. I show two
validation exercises suggesting that this bounding argument would perform well empirically. I discuss
application of this procedure to a large set of publications in economics, and use evidence from randomized
studies to draw guidelines as to appropriate bounding values.
Emily Oster
University of Chicago
Booth School of Business
5807 South Woodlawn Ave
Chicago, IL 60637
and NBER
eoster@uchicago.edu

---

## 核心假设

未提取到假设部分

---

## 方法论/识别策略

Y = βX + W1 + W2
(1)
X represents the treatment and the coeﬃcient of interest is β. W1 and W2 represent confounders. Speciﬁcally,
W1 is a vector which is a linear combination of observed control variables wo
j multiplied by their true
coeﬃcients: W1 = PJo
j=1 wo
jγo
j . W2 is a vector which is a linear combination of unobserved control variables
wu
j , again multiplied by their true coeﬃcients: W2 = PJu
j=1 wu
j γu
j . Note that W2 may contain some components
which are orthogonal to X, including any measurement error in Y .
I assume that Cov(W1, W2) = 0 and, without loss of generality, that V ar(X) = 1. The assumption of
orthogonality between W1 and W2 is discussed in more detail below. The covariance matrix associated with
the vector [X, W1, W2]′ is positive deﬁnite. Note that without further assumptions on the relationship between
X, W1 and W2 there is no information provided about the bias associated with W2 by seeing the bias from W1.
Deﬁne the proportional selection relationship as δ σ1X
σ11 = σ2X
σ22 , where σiX = Cov(Wi, X), σii = V ar(Wi)
and δ is the coeﬃcient of proportionality. I assume that δ > 0 and refer to this as the proportional selection
assumption. This implies that the relationship between X and the vector containing the observables is
informative about the relationship between X and the vector containing the unobservables.
Deﬁne the coeﬃcient resulting from the short regression of Y on X as ˚β and the R-squared from that
regression as ˚
R. Deﬁne the coeﬃcient from the intermediate regression of Y on X and W1 as ˜β and the
R-squared as ˜R. Note these are in-sample values.
The omitted variable bias on ˚β and ˜β is controlled by the auxiliary regressions of (1) W1 on X; (2) W2
on X; and (3) W2 on X and W1. Denote the in-sample coeﬃcient on X from regressions of W1 and W2 on X
as ˆλW1|X and ˆλW2|X, respectively and the coeﬃcient on X from a regression of W2 on X and W1 as ˆλW2|X,W1.
Denote the population analogs of these values λW1|X, λW2|X and λW2|X,W1.
All estimates are implicitly indexed by n. Probability limits are taken as n approaches inﬁnity. All
observations are independent and identically distributed according to model (1). By standard omitted variable
bias formulas, I can express the probability limits of the short and intermediate regression coeﬃcients in terms
of these values:
˚β
p→
β + λW1|X + λW2|X
˜β
p→
β + λW2|X,W1
Lemma 1 deﬁnes the probability limit of the coeﬃcient diﬀerence.
Lemma 1. (˚β −˜β)
p→σ1X
σ2
11−σ2
1X(δσ22+σ11)
σ11(σ11−σ2
1X)
6


Proof. This follows directly from the probability limits of the auxiliary regression coeﬃcients under the
proportional selection assumption. Proof details are in Appendix A.
Denote the sample variance of Y as ˆσyy and note that ˆσyy
p→σyy. Lemma 2 deﬁnes probability limits
for functions of the R-squared values.
Lemma 2. ( ˜R −˚
R)ˆσyy
p→[σ2
11−σ2
1X(σ11+δσ22)]
2
σ2
11(σ11−σ2
1X)
and (1 −˜R)ˆσyy
p→
σ22[σ2
11−σ2
1X(σ11+δ2σ22)]
σ11(σ11−σ2
1X)
.
Proof. This follows directly from 

---

## 估计方法

Given values for Rmax and ˜δ, the object β is point identiﬁed. As I noted above, both of these objects have
natural interpretations in the data and in some cases it may be reasonable to make assumption about their
value. β will also be point identiﬁed with an assumption about δ.
Bounding and Robustness Calculations
In many cases the exact values of ˜δ and Rmax will not be clear. In such cases, it may be more feasible to make
robustness statements based on bounding values for these objects. In practice, discussion of coeﬃcient
movements are typically done as part of a robustness argument; the discussion here will suggest a more formal
way to make similar contributions.
I conceptualize this in a partial identiﬁcation framework (Tamer, 2010; Manski, 2003). Consider the
estimator β∗′(Rmax, ˜δ) which is deﬁned above and is an asymptotically consistent estimator of β under known
values of Rmax and ˜δ. Without any additional assumptions, I note that Rmax is bounded between ˜R (the
controlled regression coeﬃcient) and 1. Under the proportional selection assumption, ˜δ is bounded below at 0
and some arbitrary upper bound δ.
The estimator below delivers the identiﬁed set for β.
∆S = {β ∈R : β = β∗′(Rmax, ˜δ), for some Rmax ∈[ ˜R, 1] and δ ∈[0, δ]}
This set is bounded on one side by ˜β, which is the value of β delivered when Rmax = ˜R or δ = 0 (or both).
Without more assumptions, the other bound is either positive or negative inﬁnity, since δ is unbounded. The
insight of partial identiﬁcation is that it may be possible to use additional intuition from the problem to
further bound both Rmax and ˜δ values.
Consider ﬁrst the issue of bounding ˜δ. I argue that for many problems, ˜δ = 1 may be a reasonable upper
bound. Recall that ˜δ captures the relative importance of the index of observed and unobserved variables in
explaining X. The bound of ˜δ = 1 suggests the observables are at least as important as the unobservables. One
reason to favor this is that researchers typically focus their data collection eﬀorts (or their choice of regression
controls) on the controls they believe ex ante are the most important. A second is that ˜
W2 is residualized with
respect to W1 so, conceptually, we want to think of the omitted variables having been stripped of the portion
related to the included ones. Ultimately, this is an empirical issue, and I will discuss evidence for this bound in
Section 4.
In the case of Rmax it may be possible to generate a bound smaller than 1 by, for example, considering
measurement error in Y or evaluating variation in Y which cannot be related to X because it results from
choices made after X is determined. Deﬁne an assumed upper bound on Rmax as Rmax, with Rmax ≤1.
11


With these two bounding assumptions the identiﬁed set is: ∆s = [˜β, β∗′(Rmax, 1)].
Empirically, the question of interest in considering ∆s is whether the conclusions based on the full set
are similar to what we would draw based on observing the controlled coeﬃcient ˜β. If inclu

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

