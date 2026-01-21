# 校准笔记: Manipulation of the Running Variable in the Regression Discontinuity Design: A Density Test

> **技能**: estimator-rd
> **论文 ID**: 35a9aa649c8b4d55c39dd575f5745932d63fea8c
> **年份**: 2007
> **期刊**: 
> **引用数**: 2300

---

## 摘要

未提取到摘要

---

## 核心假设

未提取到假设部分

---

## 方法论/识别策略

requires estimating the integral of the squared second derivative,
R
(f(2)(r))2dr. As is standard in the
literature, she uses a bandwidth other than h to estimate
R
(f(2)(r))2dr; to ﬁnd the optimal bandwidth
for this ancillary task requires approximating
R
f(2)(r)f(4)(r)dr, and we are back where we started. Cheng
16An analogous decomposition can be used to motivate an estimator that replaces takes the log of the histogram counts before
smoothing. Due to the covariance structure of the Yj and the nonlinearity of ln(·), a rigorous demonstration of asymptotic
normality does not appear straightforward unless one ﬁxes b and redeﬁnes the parameter of interest. Nonetheless, such an
estimator is consistent whenever bθ is, and has the same asymptotic variance as bθ, provided nb →∞.
17Software (STATA version 9) is available from the author for a period of 3 years from the date of publication.
9


(1994, Section 4.5.2) notes that the method fares poorly in the boundary setting, where the integrals are
(particularly) hard to estimate with any accuracy, and suggests further modiﬁcations.
To be practical, bandwidth selection rules need to be easy to implement. My own view is that the
best method is subjective choice, guided by an automatic procedure, particularly if the researcher agrees
to report how much the chosen bandwidth deviates from the recommendations of the automatic selector.
Here is a simple automatic bandwidth selection procedure that may be used as a guide:
1. Compute the ﬁrst-step histogram using the binsize bb = 2bσn−1/2, where bσ is the sample standard
deviation of the running variable.
2. Using the ﬁrst-step histogram, estimate a global 4th order polynomial separately on either side of
the cutoﬀ. For each side, compute κ
h
˘σ2(b −a)
± P ˘f′′(Xj)2i1/5
, and set bh equal to the average of
the these two quantities, where κ .= 3.348, ˘σ2 is the mean-squared error of the regression, b −a
equals XJ −c for the right-hand regression and c −X1 for the left-hand regression, and ˘f′′(Xj) is
the estimated second derivative implied by the global polynomial model.18
The second step of this algorithm is based on the rule-of-thumb bandwidth selector of Fan and Gijbels
(1996, Section 4.2). After implementing this selector, displaying the ﬁrst-step histogram based on bb and
the curve bf(r) based on bh provides a very detailed sense of the distribution of the running variable, upon
which subjective methods can be based. The selection method outlined in the above algorithm is used in
the simulation study in Section V, below, where an automatic method is needed. In the empirical work in
Section VI, where subjective methods are feasible, this selection method is used as a guide.
IV. Theoretical Example
To motivate the potential for identiﬁcation problems caused by manipulation, consider a simple labor
supply model. Agents strive to maximize the present discounted value of utility from income over two
periods. Each agent chooses to work full- or part-time in each p

---

## 估计方法

is badly biased, as is well-known (e.g., Marron and Ruppert 1994).
6I thank the editors for their emphasis of this important point.
5


One method that corrects for boundary bias is the local linear density estimator developed by Cheng,
Fan and Marron (1993) and Cheng (1994).7,8 The grounds for focusing on the local linear density estimator
are theoretical and practical.
Theoretically, the estimator weakly dominates other proposed methods.
Cheng et al. (1997) show that for a boundary point the local linear method is 100 percent eﬃcient among
linear estimators in a minimax sense.9 Practically, the ﬁrst-step histogram is of interest in its own right,
because it provides an analogue to the local averages typically accompanying conditional expectation
estimates in regression discontinuity applications. Moreover, among nonparametric methods showing good
performance at boundaries, local linear density estimation is simplest.
A.
Estimation
Implementing the local linear density estimator involves two steps. The ﬁrst step is a very under-
smoothed histogram. The bins for the histogram are deﬁned carefully enough that no one histogram bin
includes points both to the left and right of the point of discontinuity. The second step is local linear
smoothing of the histogram. The midpoints of the histogram bins are treated as a regressor, and the
normalized counts of the number of observations falling into the bins are treated as an outcome variable.
To accomodate the potential discontinuity in the density, local linear smoothing is conducted separately
for the bins to the right and left of the point of potential discontinuity, here denoted c.
The ﬁrst-step histogram is based on the frequency table of a discretized version of the running variable,
g(Ri) =
¹Ri −c
b
º
b + b
2 + c ∈
½
. . . , c −5b
2, c −3b
2, c −b
2, c + b
2, c + 3b
2, c + 5b
2, . . .
¾
(2)
where ⌊a⌋is the greatest integer in a.10,11 Deﬁne an equi-spaced grid X1, X2, . . . , XJ of width b covering the
7Published papers describing the local linear density approach include Fan and Gijbels (1996), Cheng (1997a,b), and Cheng
et al. (1997). The general idea of “pre-binning” the data before density estimation, and the conclusion that estimators based
on pre-binned data do not suﬀer in terms of practical performance despite theoretical loss of information, are both much older
than the idea of local linear density estimation; see, for example, Jones (1989) and references therein.
8Competing estimators for estimating a density function at a boundary are also available. Estimators from the statistics
literature include modiﬁed kernel methods (see, e.g., Chu and Cheng 1996, Cline and Hart 1991) and wavelet methods (for
references, see Hall, McKay and Turlach 1996). Among the better-known methods, one with good properties is Rice (1984).
Boundary folding methods are also used (see, for example, Schuster 1985), but their properties are not favorable. Marron
and Ruppert (1994) give a three-step transformation m

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

