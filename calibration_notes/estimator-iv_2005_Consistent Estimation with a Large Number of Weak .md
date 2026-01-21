# 校准笔记: Consistent Estimation with a Large Number of Weak Instruments

> **技能**: estimator-iv
> **论文 ID**: 5fa7acbebcdcb6af827738d5b781a9161b809462
> **年份**: 2005
> **期刊**: 
> **引用数**: 359

---

## 摘要

This paper analyzes the conditions under which consistent estimation can be achieved in instrumental
variables (IV ) regression when the available instruments are weak, in the local-to-zero sense of Staiger and
Stock (1997) and using the many-instrument framework of Morimune (1983) and Bekker (1994). Our analysis
of an extended k-class of estimators that includes Jackknife IV (JIV E) establishes that consistent estimation
depends importantly on the relative magnitudes of rn, the growth rate of the concentration parameter, and
Kn, the number of instruments. In particular, LIML and JIV E are consistent when
√Kn
rn
→0, while two-
stage least squares is consistent only if Kn
rn →0, as n →∞. We argue that the use of many instruments may
be beneﬁcial for estimation, as the resulting concentration parameter growth may allow consistent estimation,
in certain cases.

---

## 核心假设

- Assumption 1: Π = Πn = Cn
bn for some sequence of positive real numbers {bn} , non-decreasing in n, and for
some sequence of nonrandom, Kn × G parameter matrices {Cn} .

- Assumption 2: Let
©
Zn,i : i = 1, ..., n; n ≥1
ª
be a triangular array of RKn+J-valued random variables, where
Zn,i = (Z′
n,i, X′
i)′ with Z′
n,i and X′
i denoting the ith row of the matrices Zn and Xn, respectively. Moreover,
suppose that: (a) Kn →∞as n →∞such that Kn
n
→α for some constant α satisfying 0 ≤α < 1; (b) there
exists a positive integer N such that ∀n ≥N, Zn is of full column rank Kn + J almost surely; and (c) {rn}
is a non-decreasing sequence of positive real numbers such that, as 

- Assumption 3: Assume that: (a) Zn and ηi are independent for all i and n; (b) {ηi} ≡i.i.d.(0, Σ), where Σ > 0,
and Σ can be partitioned conformably with (ui, v′
i)′ as Σ =
µ
σuu
σ′
V u
σV u
ΣV V
¶
, with σg
V u and and Σ(g,h)
V V
denoting
the gth element of σV u and the (g, h)th element of ΣV V ; respectively; and (c) there exists some positive constant
Dη < ∞such that max
©
E
¡
u4
i
¢
, E
¡
v4
i1
¢
, ..., E
¡
v4
iG
¢ª
≤Dη.
The estimators we consider can be written in the form:
bβω,n =
¡
Y ′
2n


- Assumption 2: are given in the extended version of this paper, Chao and Swanson (2002).
3

- Assumption 4: Suppose that for each i and n, eωi,n can be decomposed into the sum of two components as
eωi,n = ωi,n + ξi,n, such that ωi,n is either non-random or depends only on the exogenous variables Zn, so that
ωi,n = fn,i(Zn). Also, assume that ωi,n and ξi,n satisfy the following conditions: (a) lim
n→∞ln < ∞a.s., where
ln =
sup
1≤i ≤n
|ωi,n|; (b)
nP
i=1
ωi,n (1 −hi,n) = Kn
a.s. ∀n, where hi,n is the ith diagonal element of PZn; (c)
nP
i=1
E
¡
ω2
i,n
¢
= O(Kn); and (d)
sup
1≤i ≤n
|ξi,n| = o

- Assumption 2: (c), rn can be interpreted as the rate at which the concentration
parameter Σ
−1
2
V V ΠnZ′
nQXnZnΠnΣ
−1
2
V V grows as n increases. An assumption on the rate of growth of the concen-
tration parameter seems natural here since the concentration parameter is a measure of instrumental strength.
Because we are interested in the case of weak instruments,

- Assumption 2: (c) stipulates that rn must grow no
faster than n. In fact, we will be interested primarily in the case where rn grows much slower than n. In addition,

- Assumption 3: requires the instrument matrix Zn to be independent of the disturbance vector ηi for all i and
n, and also requires the disturbances to have ﬁnite absolute fourth moments. Note that these assumptions are
weaker than the corresponding assumptions in Morimune (1983) and Bekker (1994), where ﬁxed instruments and
i.i.d. Gaussian errors are assumed.
(ii) To see the relationship between our framework and that of Staiger and Stock (1997), note that the Staiger-
Stock setup takes bn = √n, 

- Assumption 4: , and hence

- Assumption 4: , it is helpful to focus discussion on the special case where J = 0 and
G = 1 (i.e., the case where there are no included exogenous regressors and only one endogenous regressor in the
structural equation). As mentioned above, an ω−class estimator can be viewed as an IV estimator where the
8Note that Assumption J does rule out exogenous regressors of the form ei = (0, ..., 0, 1, 0, ..., 0), where ei denotes the ith column
of In, but it does not rule out dummy variable regressors in 


---

## 方法论/识别策略

未提取到方法论部分

---

## 估计方法

depends importantly on the relative magnitudes of rn, the growth rate of the concentration parameter, and
Kn, the number of instruments. In particular, LIML and JIV E are consistent when
√Kn
rn
→0, while two-
stage least squares is consistent only if Kn
rn →0, as n →∞. We argue that the use of many instruments may
be beneﬁcial for estimation, as the resulting concentration parameter growth may allow consistent estimation,
in certain cases.
JEL classiﬁcation: C13, C31.
Keywords: instrumental variables, k-class estimator, local-to-zero framework, pathwise asymptotics, weak instru-
ments.
∗John C. Chao: Department of Economics, University of Maryland, College Park, MD 20742, chao@econ.umd.edu. Norman R. Swanson:
Department of Economics, Rutgers University, New Brunswick, NJ 08901, nswanson@econ.rutgers.edu. The authors thank Don Andrews,
Alastair Hall, Atsushi Inoue, Harry Kelejian, Ingmar Prucha, Graham Elliott, Whitney Newey, Peter Phillips, Tao Zha, Eric Zivot, and
participants at the 2001 summer econometric society meeting, and workshops at North Carolina State University and Yale University for
useful comments and suggestions. Swanson gratefully acknowledges ﬁnancial support in the form of a Rutgers University Research Council
grant.


1
Introduction
In the weak instruments literature, it has become standard in recent years to analyze the properties of estimators
and test statistics using the local-to-zero framework pioneered by Staiger and Stock (1997), which takes the
coeﬃcients of the instruments in the ﬁrst-stage regression to be in a n−1
2 shrinking neighborhood of zero, where n
is the sample size. An interesting feature of the Staiger-Stock framework is that unlike the conventional asymptotic
setup, the concentration parameter does not diverge but rather, roughly speaking, stays constant in expectation
as the sample size grows. Since the concentration parameter is a natural measure of the strength of identiﬁcation
in an IV regression model, the local-to-zero device allows the Staiger-Stock framework to better mimic the weak
instrument situation than the conventional setup with ﬁxed coeﬃcients, and Staiger and Stock show that the two-
stage least squares (2SLS) and the limited information maximum likelihood (LIML) estimators are no longer
consistent and instead converge to nonstandard distributions in the limit under this framework.1,2
Another important direction that IV regression research has taken involves the study of situations where the
number of available instruments is large, using an asymptotic framework that takes the number of instruments to
inﬁnity as a function of the sample size. This approach was ﬁrst taken by Morimune (1983) and later generalized
by Bekker (1994) (see also Angrist and Krueger (1995), Bekker and van der Ploeg (1999), van Hasselt (2000),
Donald and Newey (2001), Hahn, Hausman, and Kuersteiner (2001), Hahn (2002), and Hahn and Inoue (2002)).
In contrast to the papers using the local-to-zero setup, authors taki

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

