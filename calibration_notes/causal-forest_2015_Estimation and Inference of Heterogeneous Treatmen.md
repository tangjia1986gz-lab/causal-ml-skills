# 校准笔记: Estimation and Inference of Heterogeneous Treatment Effects using Random Forests

> **技能**: causal-forest
> **论文 ID**: c2fcb00fe4b773f9cb1682aaa69749aac59f711d
> **年份**: 2015
> **期刊**: Journal of the American Statistical Association
> **引用数**: 2801

---

## 摘要

Many scientiﬁc and engineering challenges—ranging from personalized medicine to
customized marketing recommendations—require an understanding of treatment eﬀect
heterogeneity. In this paper, we develop a non-parametric causal forest for estimat-
ing heterogeneous treatment eﬀects that extends Breiman’s widely used random for-
est algorithm. In the potential outcomes framework with unconfoundedness, we show
that causal forests are pointwise consistent for the true treatment eﬀect, and have an
asymptotically Gaussian and centered sampling distribution. We also discuss a prac-
tical method for constructing asymptotic conﬁdence intervals for the true treatment
eﬀect that are centered at the causal forest estimates. Our theoretical results rely on a
generic Gaussian theory for a large family of random forest algorithms. To our knowl-
edge, this is the ﬁrst set of results that allows any type of random forest, including
classiﬁcation and regression forests, to be used for provably valid statistical inference.
In experiments, we ﬁnd causal forests to be substantially more powerful than classical
methods based on nearest-neighbor matching, especially in the presence of irrelevant
covariates.

---

## 核心假设

未提取到假设部分

---

## 方法论/识别策略

with unconfoundedness [Neyman, 1923, Rubin, 1974].
Although our main focus in this paper is causal inference, we note that there are a variety
of important applications of the asymptotic normality result in a pure prediction context.
For example, Kleinberg et al. [2015] seek to improve the allocation of medicare funding
for hip or knee replacement surgery by detecting patients who had been prescribed such a
surgery, but were in fact likely to die of other causes before the surgery would have been
useful to them. Here we need predictions for the probability that a given patient will survive
for more than, say, one year that come with rigorous conﬁdence statements; our results are
the ﬁrst that enable the use of random forests for this purpose.
Finally, we compare the performance of the causal forest algorithm against classical k-
nearest neighbor matching using simulations, ﬁnding that the causal forest dominates in
terms of both bias and variance in a variety of settings, and that its advantage increases
with the number of covariates. We also examine coverage rates of our conﬁdence intervals
for heterogeneous treatment eﬀects.
1.1
Related Work
There has been a longstanding understanding in the machine learning literature that predic-
tion methods such as random forests ought to be validated empirically [Breiman, 2001b]: if
the goal is prediction, then we should hold out a test set, and the method will be considered
as good as its error rate is on this test set. However, there are fundamental challenges with
applying a test set approach in the setting of causal inference. In the widely used potential
outcomes framework we use to formalize our results [Neyman, 1923, Rubin, 1974], a treat-
ment eﬀect is understood as a diﬀerence between two potential outcomes, e.g., would the
patient have died if they received the drug vs. if they didn’t receive it. Only one of these
potential outcomes can ever be observed in practice, and so direct test-set evaluation is in
general impossible.1 Thus, when evaluating estimators of causal eﬀects, asymptotic theory
plays a much more important role than in the standard prediction context.
From a technical point of view, the main contribution of this paper is an asymptotic nor-
mality theory enabling us to do statistical inference using random forest predictions. Recent

---

## 估计方法

in a sparse high-dimensional linear setting. Beygelzimer and Langford [2009], Dud´ık et al.
[2011], and others discuss procedures for transforming outcomes that enable oﬀ-the-shelf loss
minimization methods to be used for optimal treatment policy estimation. In the econo-
metrics literature, Bhattacharya and Dupas [2012], Dehejia [2005], Hirano and Porter [2009],
Manski [2004] estimate parametric or semi-parametric models for optimal policies, relying on
regularization for covariate selection in the case of Bhattacharya and Dupas [2012]. Taddy
et al. [2016] use Bayesian nonparametric methods with Dirichlet priors to ﬂexibly estimate
the data-generating process, and then project the estimates of heterogeneous treatment ef-
fects down onto the feature space using regularization methods or regression trees to get
low-dimensional summaries of the heterogeneity; but again, there are no guarantees about

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

