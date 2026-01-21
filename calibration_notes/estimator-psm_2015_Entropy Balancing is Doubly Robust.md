# 校准笔记: Entropy Balancing is Doubly Robust

> **技能**: estimator-psm
> **论文 ID**: 1b73a6de77d1e3c3c5bb588e761506f424d9c117
> **年份**: 2015
> **期刊**: 
> **引用数**: 353

---

## 摘要

Covariate balance is a conventional key diagnostic for methods estimating causal eﬀects from observational
studies. Recently, there is an emerging interest in directly incorporating covariate balance in the estimation.
We study a recently proposed entropy maximization method called Entropy Balancing (EB), which exactly
matches the covariate moments for the diﬀerent experimental groups in its optimization problem. We show
EB is doubly robust with respect to linear outcome regression and logistic propensity score regression, and it
reaches the asymptotic semiparametric variance bound when both regressions are correctly speciﬁed. This is
surprising to us because there is no attempt to model the outcome or the treatment assignment in the original
proposal of EB. Our theoretical results and simulations suggest that EB is a very appealing alternative to the
conventional weighting estimators that estimate the propensity score by maximum likelihood.

---

## 核心假设

- Assumption 1: (strong ignorability). (𝑌(0), 𝑌(1))⊥𝑇|𝑋.

- Assumption 2: (overlap). 0 < P(𝑇= 1|𝑋) < 1.
Intuitively, the ﬁrst assumption says that the observed covariates contain all the information that may cause
the selection bias, i. e. there is no unmeasured confounding variable, and the second assumption ensures that
the bias-correction information is available across the entire domain of 𝑋.
Since the covariates 𝑋contain all the information of confounding bias, it is important to understand the
relationship between 𝑇, 𝑌and 𝑋. Under

- Assumption 1: (strong ignorability), the joint distribution of (𝑋, 𝑌, 𝑇) is
determined by the marginal distribution of 𝑋and two conditional distributions given 𝑋. The ﬁrst conditional
distribution 𝑒(𝑋) = 𝑃(𝑇= 1|𝑋) is often called the propensity score and plays a central role in causal inference
[1]. The second conditional distribution is the density of 𝑌(0) and 𝑌(1) given 𝑋. Since we only consider the
mean causal eﬀect in this paper, it suﬀices to study the mean regression functions 𝑔0(𝑋) = 𝐸[𝑌(

- Assumption 1: (strong ignorability) and

- Assumption 2: (overlap) be given. Additionally, assume the
expectation of c(x) exists and Var(Y(0)) < ∞. Then Entropy Balancing is doubly robust (Property 1) in the sense that
(1) If logit(𝑒(𝑥)) or 𝑔0(𝑥) is linear in 𝑐u�(𝑥), 𝑗= 1, … , 𝑅, then ̂𝛾EB is statistically consistent.
(2) Moreover, if logit(𝑒(𝑥)), 𝑔0(𝑥) and 𝑔1(𝑥) are all linear in 𝑐u�(𝑥), 𝑗= 1, … , 𝑅, then ̂𝛾EB reaches the semiparamet-
ric variance bound of 𝛾derived in Hahn [34]​ with unknown propensity score.
We give two proofs of the ﬁ

- Assumption 2: (overlap) with high probability.

- Assumption 2: (overlap) is satisﬁed and the expectation of 𝑐(𝑋) exist, then P(𝑤EBexists) →1 as
𝑛→∞. Furthermore, ∑u�
u�=1 (𝑤EB
u�)2 →0 in probability as 𝑛→∞.
Proof. Since the expectation of 𝑐(𝑋) exist, the weak law of large number says
̄𝑐(1)
p→̄𝑐∗(1) = E[𝑐(𝑋)|𝑇= 1].
Therefore

- Assumption 2: (overlap) implies ̄𝑐∗(1) hence 𝐵u�( ̄𝑐∗(1)) is in the interior of the convex hull of
Ω(𝑋) for suﬀiciently small 𝜀. Let 𝑅u�, 𝑖= 1, … , 3u�, be the 3u�boxes centered at
̄𝑐∗(1) + 3
2𝜀𝑏, where 𝑏∈u�is a vector
that each entry can be −1, 0, or 1. It is easy to check that the sets 𝑅u�are disjoint and the convex hull of {𝑥u�}3u�
u�=1
contains 𝐵u�( ̄𝑐∗(1)) if 𝑥u�∈𝑅u�, 𝑖= 1, … , 3u�. Since 0 < 𝑃(𝑇= 0|𝑋) < 1, 𝜌= minu�P(𝑋∈𝑅u�|𝑇= 0) > 0. This implies
P(∃𝑋u�∈𝑅u�and𝑇u�= 0, ∀𝑖= 1, … , 3u�) ≥1 −
3u


---

## 方法论/识别策略

̂𝑒(𝑥), the corresponding weights

---

## 估计方法

[21, 22], which is widely popular in survey sampling but perhaps not suﬀiciently recognized in causal inference
[23]. The balancing constraints in this optimization problem result in unbiasedness of the PATT estimator un-
der linear outcome regression model. The dual optimization problem of EB is ﬁtting a logistic propensity score
model with a loss function diﬀerent from the negative binomial likelihood. The Fisher-consistency of this loss
function (also called proper scoring rule in statistical decision theory, see e. g. Gneiting and Raftery [24]) ensures
the other half of double robustness – consistency under correctly speciﬁed propensity score model. Since EB
essentially just uses a diﬀerent loss function, other types of propensity score models, for example the general-
ized additive models [25] can also easily be ﬁtted. A forthcoming article by Zhao [26] oﬀers more discussion
and extension to other weighted average treatment eﬀects.
Figure 1: The role of covariate balance in doubly robust estimation. Dashed arrows: conventional procedure to achieve
double robustness. Solid arrows: double robustness of Entropy Balancing via covariate balance.
2
Setting
First, we ﬁx some notations for the causal inference problem considered in this paper. We follow the potential
outcome language of Neyman [27] and Rubin [28]. In this causal model, each unit 𝑖is associated with a pair of
potential outcomes: the response 𝑌u�(1) that is realized if 𝑇u�= 1 (treated), and another response 𝑌u�(0) realized if
𝑇u�= 0 (control). We assume the observational units are independent and identically distributed samples from
a population, for which we wish to infer the treatment’s eﬀect. The main obstacle is that only one potential
outcome is observed: 𝑌u�= 𝑇u�𝑌u�(1) −(1 −𝑇u�)𝑌u�(0), which is commonly known as the “fundamental problem of
causal inference” [29].
In this paper we focus on estimating the Population Average Treatment eﬀect on the Treated (PATT):
𝛾= E[𝑌(1)|𝑇= 1] −E[𝑌(0)|𝑇= 1]
Δ= 𝜇(1|1) −𝜇(0|1).
(1)
The counterfactual mean 𝜇(0|1) = E[𝑌(0)|𝑇= 1] also naturally occurs in survey sampling with missing data
[21, 22] by viewing 𝑌(0) as the only outcome of interest (so 𝑇= 1 stands for non-response).
Along with the treatment exposure 𝑇u�and outcome 𝑌u�, each unit 𝑖is usually associated with a set of covari-
ates denoted by 𝑋u�measured prior to the treatment assignment. In a typical observational study, both treatment
assignment and outcome may be related to the covariates, which can cause serious confounding bias. The sem-
inal work by Rosenbaum and Rubin [1] suggest that it is possible to correct the confounding bias under the
following two assumptions:


DE GRUYTER
Zhao and Percival
Assumption 1 (strong ignorability). (𝑌(0), 𝑌(1))⊥𝑇|𝑋.
Assumption 2 (overlap). 0 < P(𝑇= 1|𝑋) < 1.
Intuitively, the ﬁrst assumption says that the observed covariates contain all the information that may cause
the selection bias, i. e. there is no unmeasured confounding variable, and the second a

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

