# 校准笔记: Regression Discontinuity Designs Using Covariates

> **技能**: estimator-rd
> **论文 ID**: 46bdc52efa60d638130d02d4ad27e76c8e56f5a6
> **年份**: 2018
> **期刊**: Review of Economics and Statistics
> **引用数**: 629

---

## 摘要

We study regression discontinuity designs when covariates are included in
the estimation. We examine local polynomial estimators that include discrete
or continuous covariates in an additive separable way, but without impos-
ing any parametric restrictions on the underlying population regression func-
tions.
We recommend a covariate-adjustment approach that retains consis-
tency under intuitive conditions, and characterize the potential for estimation
and inference improvements.
We also present new covariate-adjusted mean
squared error expansions and robust bias-corrected inference procedures, with
heteroskedasticity-consistent and cluster-robust standard errors. An empirical
illustration and an extensive simulation study is presented. All methods are
implemented in R and Stata software packages.

---

## 核心假设

未提取到假设部分

---

## 方法论/识别策略

未提取到方法论部分

---

## 估计方法

and inference improvements.
We also present new covariate-adjusted mean
squared error expansions and robust bias-corrected inference procedures, with
heteroskedasticity-consistent and cluster-robust standard errors. An empirical
illustration and an extensive simulation study is presented. All methods are
implemented in R and Stata software packages.
Keywords: regression discontinuity, covariate adjustment, causal inference, local
polynomial methods, robust inference, bias correction, tuning parameter selection.
JEL codes: C14, C18, C21.
1


1
Introduction
The Regression Discontinuity (RD) design is widely used in Economics, Political Sci-
ence, and many other social, behavioral, biomedical, and statistical sciences. Within
the causal inference framework, the RD design is considered to be one of the most
credible non-experimental strategies because it relies on weak and easy-to-interpret
nonparametric identifying assumptions, which permit ﬂexible and robust estimation
and inference for local treatment eﬀects. The key feature of the design is the exis-
tence of a score, index, or running variable for each unit in the sample, which deter-
mines treatment assignment via hard-thresholding: all units whose score is above a
known cutoﬀare oﬀered treatment, while all units whose score is below this cutoﬀ
are not. Identiﬁcation, estimation, and inference proceed by comparing the responses
of units near the cutoﬀ, taking those below (comparison group) as counterfactuals
to those above (treatment group). For literature reviews and practical introductions,
see Imbens and Lemieux (2008), Lee and Lemieux (2010), Cattaneo and Escanciano
(2017), Cattaneo, Titiunik and Vazquez-Bare (2017), Cattaneo, Idrobo and Titiunik
(2018a,b), and references therein.
Nonparametric identiﬁcation of the RD treatment eﬀect typically relies on conti-
nuity assumptions, which motivate nonparametric local polynomial methods tailored
to ﬂexibly approximate, above and below the cutoﬀ, the unknown conditional mean
function of the outcome variable given the score. In practice, researchers often choose
a local linear polynomial and perform estimation using weighted linear least squares,
giving higher weights to observations close to the cutoﬀ. These estimates are then
used to assess whether there is a discontinuity in levels, derivatives, or ratios thereof,
at the cutoﬀ. If present, this discontinuity is interpreted as some average response to
the treatment (assignment) at the cutoﬀ, depending on the setting and assumptions
under consideration.
When practitioners employ weighted least squares to estimate RD eﬀects, they of-
2


ten augment their estimation models with additional predetermined covariates such
as demographic or socio-economic characteristics of the units. Despite the perva-
siveness of this practice, there has been little formal study of the consequences of
covariate adjustment for identiﬁcation, estimation, and inference of RD eﬀects under
standard smoothness conditions a

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

