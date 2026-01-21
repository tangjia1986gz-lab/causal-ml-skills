# 校准笔记: Genetic Matching for Estimating Causal Effects: A General Multivariate Matching Method for Achieving Balance in Observational Studies

> **技能**: estimator-psm
> **论文 ID**: 04bd3574794891c3f7c5038abcab7589002fb85e
> **年份**: 2006
> **期刊**: Review of Economics and Statistics
> **引用数**: 1085

---

## 摘要

Genetic matching is a new method for performing multivariate matching which uses an
evolutionary search algorithm to determine the weight each covariate is given. The method
utilizes an evolutionary algorithm developed by Mebane and Sekhon (1998; Sekhon and
Mebane 1998) that maximizes the balance of observed potential confounders across matched
treated and control units. The method is nonparametric and does not depend on knowing
or estimating the propensity score, but the method is greatly improved when a known or
estimated propensity score is incorporated.
Genetic matching reliably reduces both the
bias and the mean square error of the estimated causal eﬀect even when the property of
equal percent bias reduction (EPBR) does not hold. When this property does not hold,
matching methods—such as Mahalanobis distance and propensity score matching—often
perform poorly. Even if the EPBR property does hold and the propensity score is correctly
speciﬁed, in ﬁnite samples, estimates based on genetic matching have lower mean square
error than those based on the usual matching methods.
We present a reanalysis of the
LaLonde (1986) job training dataset which demonstrates the beneﬁts of genetic matching
and which helps to resolve a longstanding debate between Dehejia and Wahba (1997; 1999;
2002; Dehejia 2005) and Smith and Todd (2001, 2005a,b) over the ability of matching to
overcome LaLonde’s critique of nonexperimental estimators. Monte Carlos are also presented
to demonstrate the properties of our method.

---

## 核心假设

未提取到假设部分

---

## 方法论/识别策略

utilizes an evolutionary algorithm developed by Mebane and Sekhon (1998; Sekhon and
Mebane 1998) that maximizes the balance of observed potential confounders across matched
treated and control units. The method is nonparametric and does not depend on knowing
or estimating the propensity score, but the method is greatly improved when a known or
estimated propensity score is incorporated.
Genetic matching reliably reduces both the
bias and the mean square error of the estimated causal eﬀect even when the property of
equal percent bias reduction (EPBR) does not hold. When this property does not hold,
matching methods—such as Mahalanobis distance and propensity score matching—often
perform poorly. Even if the EPBR property does hold and the propensity score is correctly
speciﬁed, in ﬁnite samples, estimates based on genetic matching have lower mean square
error than those based on the usual matching methods.
We present a reanalysis of the
LaLonde (1986) job training dataset which demonstrates the beneﬁts of genetic matching
and which helps to resolve a longstanding debate between Dehejia and Wahba (1997; 1999;
2002; Dehejia 2005) and Smith and Todd (2001, 2005a,b) over the ability of matching to
overcome LaLonde’s critique of nonexperimental estimators. Monte Carlos are also presented
to demonstrate the properties of our method.


1
Introduction
Matching has become an increasingly popular method of causal inference in many ﬁelds
including statistics (e.g., Rosenbaum 2002), medicine (e.g., Christakis and Iwashyna 2003;
Rubin 1997), economics (e.g., Abadie and Imbens forthcoming; Dehejia and Wahba 1999,
2002; Galiani, Gertler, and Schargrodsky 2005), political science (e.g., Imai 2005; Sekhon
2004), sociology (e.g., Diprete and Engelhardt 2004; Smith 1997; Winship and Morgan 1999)
and even law (e.g., Epstein, Ho, King, and Segal 2005; Rubin 2001). There is, however, no
consensus on how exactly matching ought to be done, how to measure the success of the
matching procedure, and whether or not matching estimators are suﬃciently robust to mis-
speciﬁcation so as to be useful in practice (Heckman, Ichimura, Smith, and Todd 1998).
These issues have been central to an ongoing debate over how matching and other nonexper-
imental estimators perform when analyzing data from a nationwide job training experiment
(LaLonde 1986). The experimental results are used to establish benchmark estimates for
causal eﬀects. Then, to create the kind of observational data typically analyzed by social
scientists, individuals from the experimental control group are replaced by individuals from
national observational surveys. The goal is to determine which methods, if any, are able
to use the observational data to recover results obtained from the randomized experiment.
We show that the debate between Dehejia and Wahba (1997; 1999; 2002; Dehejia 2005) and
Smith and Todd (2001, 2005a,b) over the ability of matching to overcome LaLonde’s cri-
tique of nonexperimental estimators is 

---

## 估计方法

未提取到估计部分

---

## 关键公式

1. `1000. The equation that determines outcomes
Y (ﬁctional earnings) is:
Y
= 1000 T + .1 exp [.7 log(re74 + .01) + .7 log(re75 + 0.01)] + ϵ
where ϵ ∼N(0, 10), re74 is real earnings in 1974, re75 is real `

2. `886 in the LaLonde sample and`

3. `900 (see Table 3).
17LaLonde’s paper reports that he experimented with matching the comparison groups even more closely
to the pre-training characteristics of the experimental sample, but found these `

4. `1734 (see Table 3). Note that in the DW sample, it is possible
to get lucky and produce a reliable result even when balance has not been attained, which
24


helps explain why it is possible for DW to`

5. `234 and the
highest was`

6. `285.41, was shared by 383 of these estimates. Standard errors
ranged from`

7. `715. Recall that the experimental diﬀerence in means for the LaLonde
sample was`

8. `485,
suggesting that the experimental diﬀerence in means may be overstating the the central tendency of the
true eﬀect. Our best-balancing matching estimate with experimental data,`

9. `1794. Simple Mahalanobis matching produces an estimate of`


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

