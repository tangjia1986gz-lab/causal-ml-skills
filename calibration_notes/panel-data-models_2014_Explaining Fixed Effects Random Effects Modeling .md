# 校准笔记: Explaining Fixed Effects: Random Effects Modeling of Time-Series Cross-Sectional and Panel Data*

> **技能**: panel-data-models
> **论文 ID**: fe5bb5d8d6b7ac251d87bc16e75ea5889cc92425
> **年份**: 2014
> **期刊**: Political Science Research and Methods
> **引用数**: 1407

---

## 摘要

This article challenges Fixed Effects (FE) modeling as the ‘default’ for time-series-cross-sectional and panel data. Understanding different within and between effects is crucial when choosing modeling strategies. The downside of Random Effects (RE) modeling—correlated lower-level covariates and higher-level residuals—is omitted-variable bias, solvable with Mundlak's (1978a) formulation. Consequently, RE can provide everything that FE promises and more, as confirmed by Monte-Carlo simulations, which additionally show problems with Plümper and Troeger's FE Vector Decomposition method when data are unbalanced. As well as incorporating time-invariant variables, RE models are readily extendable, with random coefficients, cross-level interactions and complex variance functions. We argue not simply for technical solutions to endogeneity, but for the substantive importance of context/heterogeneity, modeled using RE. The implications extend beyond political science to all multilevel datasets. However, omitted variables could still bias estimated higher-level variable effects; as with any model, care is required in interpretation.

---

## 核心假设

未提取到假设部分

---

## 方法论/识别策略

(Equation 1), which does a similar thing but in a single overall model.11 Stage 1 (Equation 7)
is equivalent to the RE micro model, Stage 2 (Equation 8) to the macro model and Stage 3
(Equation 10) to the combined model. Just as with RE, the higher-level residual is
assumed to be Normal (from the regression in Equation 8). What it does do differently is
also control out any between effect of x1ij in the estimation of b1, meaning that these
estimates will only include the within effect, as in standard FE models.
The FE vector decomposition (FEVD) estimator has been criticized by many in
econometrics, who argue that the standard errors are likely to be incorrectly estimated
(Breusch et al. 2011a, 2011b; Greene 2011a, 2011b, 2012). Plu¨ mper and Troeger (2011)
provide a method for calculating more appropriate standard errors, and so the FEVD
model does work (at least with balanced data) when this method is utilized. However,
our concern is that it retains many of the other ﬂaws of FE models, which we have
outlined above. It remains much less generalizable than an RE model—it cannot be
11 In the early stages of the development of the multilevel model, a very similar process to the two-stage
FEVD model was used to estimate processes at multiple levels (Burstein et al. 1978; Burstein and Miller
1980) before being superseded by the modern multilevel RE model in which an overall model is estimated
(Raudenbush and Bryk 1986). As Beck (2005, 458) argues: ‘‘perhaps at one time it could have been argued
that one-step methods were conceptually more difﬁcult, but, given current training, this can no longer be
an excuse worth taking seriously.’’
140
BELL AND JONES
https://doi.org/10.1017/psrm.2014.7 Published online by Cambridge University Press


extended to three (or more) levels, nor can coefﬁcients be allowed to vary (as in a random
coefﬁcients model). It does not provide a nice measure of variance at the higher level,
which is often interesting in its own right. Finally, it is heavily parameterized, with a
dummy variable for each higher-level entity in the ﬁrst stage, and thus can be relatively
slow to run when there is a large number of higher-level units.
Plu¨ mper and Troeger also attempt to estimate the effects of ‘rarely changing’ variables,
and their desire to do so using FE modeling suggests that they do not fully appreciate the
difference between within and between effects. While they do not quantify what ‘rarely
changing’ means, their motivation is to get signiﬁcant results where FE produces
insigniﬁcant results. FE models only estimate within effects, and so an insigniﬁcant effect
of a rarely changing variable should be taken as saying that there is no evidence for a
within effect of that variable. When Plu¨ mper and Troeger use FEVD to estimate the effects
of rarely changing variables, they are in fact estimating between effects. Using FEVD to
estimate the effects of rarely changing variables is not a technical ﬁx for the high variance
of within

---

## 估计方法

The rationale behind FE estimation is simple and persuasive, which explains why it is so
regularly used in many disciplines. To avoid the problem of heterogeneity bias, all higher-
level variance, and with it any between effects, are controlled out using the higher-level
entities themselves (Allison 2009), included in the model as dummy variables Dj:
yij 5
X
j
j51
b0jDj 1 b1xij 1 eij:
ð5Þ
To avoid having to estimate a parameter for each higher-level unit, the mean for higher-
level entity is taken away from both sides of Equation 5, such that:
ðyij  yjÞ 5 b1ðxij  xjÞ 1 ðeij  ejÞ:
ð6Þ
Because FE models only estimate within effects, they cannot suffer from heterogeneity
bias. However, this comes at the cost of being unable to estimate the effects of higher-level
processes, so RE is often preferred where the bias does not exist. In order to test for the
existence of this form of bias in the standard RE model as speciﬁed in Equation 1, the
Hausman speciﬁcation test (Hausman 1978) is often used. This takes the form of
comparing the parameter estimates of the FE and RE models (Greene 2012; Wooldridge
2002) via a Wald test of the difference between the vector of coefﬁcient estimates of each.
The Hausman test is regularly used to test whether RE can be used, or whether FE
estimation should be used instead (for example Greene 2012, 421). However, it is
problematic when the test is viewed in terms of ﬁxed and random effects, and not in terms
of what is actually going on in the data. A negative result in a Hausman test tells us only
that the between effect is not signiﬁcantly biasing an estimate of the within effect in
Equation 1. It ‘‘is simply a diagnostic of one particular assumption behind the estimation
procedure usually associated with the random effects modely it does not address the
decision framework for a wider class of problems’’ (Fielding 2004, 6). As we show later,
the RE model that we propose in this article solves the problem of heterogeneity bias
described above, and so makes the Hausman test, as a test of FE against RE, redundant.
10 Although we do deny that FE models are any better able than RE models to deal with these other
forms of bias.
138
BELL AND JONES
https://doi.org/10.1017/psrm.2014.7 Published online by Cambridge University Press


It is ‘‘neither necessary nor sufﬁcient’’ (Clark and Linzer 2012, 2) to use the Hausman test
as the sole basis of a researcher’s ultimate methodological decision.
PROBLEMS WITH FE MODELS
Clearly, there are advantages to the FE model of Equations 5-6 over the RE models in
Equation 1. By clearing out any higher-level processes, the model deals only with
occasion-level processes. In the context of longitudinal data, this means considering
differences over time, controlling out higher-level differences and processes absolutely,
and supposedly ‘‘getting rid of proper nouns’’ (King 2001, 504)—that is, distinctive,
speciﬁc characteristics of higher-level units. This is why it has become the ‘‘gold stan

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

