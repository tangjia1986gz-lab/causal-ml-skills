# 校准笔记: Randomization Inference in the Regression Discontinuity Design: An Application to Party Advantages in the U.S. Senate

> **技能**: estimator-rd
> **论文 ID**: 51dd57727d4654e74ad36d207928684097a87147
> **年份**: 2015
> **期刊**: 
> **引用数**: 130

---

## 摘要

未提取到摘要

---

## 核心假设

- Assumption 1: Local Randomization. There exists a neighborhood W0 ¼ r;r
½
 with r < r0 <r such that for
all i with Ri 2 W0:
(a) FRijRi2W0ðrÞ ¼ FðrÞ, and
(b) yiðrÞ ¼ yiðzW0Þ for all r.
The first part of

- Assumption 1: says that the distribution of the score is the same for all units inside W0,
implying that the scores can be considered “as good as randomly assigned” in this window. This is a strong
assumption and would be violated if, for example, the score were affected by the potential outcomes even
near the threshold – but may be relaxed, for instance, by explicitly modeling the relationship between Ri
and potential outcomes. The second part of this assumption requires that potential outcomes

- Assumption 1: are stronger than those typically required for identification and
inference in the classical RD literature. Instead of only assuming continuity of the relevant population
functions at r0 (e.g., conditional expectations, distribution functions), our assumption implies that, in the
window W0, these functions are not only continuous but also constant as a function of the score.4 But

- Assumption 1: can also be viewed as an approximation to the standard continuity conditions in much the
same way the nonparametric large-sample approach approximates potential outcomes as locally linear.
This connection is made precise in Section 6.5.

- Assumption 1: has two main implications for our approach.
First, it means that near the threshold we can ignore the score values for purposes of statistical inference
and focus on the treatment indicators ZW0. Second, since the distribution of ZW0 does not depend on
potential outcomes, comparisons of observed outcomes across the threshold have a causal interpretation.
In most settings,

- Assumption 1: is plausible only within a narrow window of the threshold, leaving only
a small number of units for analysis. Thus, the problems of estimation and inference using this assumption
in the context of RD are complicated by small-sample concerns. Following Rosenbaum [14, 15], we propose
using exact randomization inference methods to overcome this potential small-sample problem. In the
remainder of this section, we maintain

- Assumption 1: and take as given the window W0, but we discuss
explicitly empirical methods for choosing this window in Section 3.
4 This assumption could be relaxed to FRijRi2W0ðrÞ ¼ FiðrÞ, allowing each unit to have different probabilities of treatment
assignment. However, in order to conduct exact-finite sample inference based on this weaker assumption, further parametric or
semiparametric assumptions are needed. See footnote 5 for further discussion on this point.
4
M. D. Cattaneo et al.: Ran

- Assumption 1: No treatment effect means observed outcomes are fixed regardless of
the realization of ZW0. Under this null hypothesis, potential outcomes are not a function of treatment status
inside W0; that is, yiðzÞ ¼ yi for all i within the window and for all z 2 ΩW0, where yi is a fixed scalar. The
distribution of any test statistic TðZW0; yW0Þ is known, since it depends only on the known distribution of
ZW0, and yW0, the fixed vector of observed responses. The test thus consists of computin

- Assumption 1: ) even
for a small number of units. This feature is particularly important in the RD design where the number of units
within W0 is likely to be small.
5 Under the generalization discussed in footnote 4, the parameter π in the Bernoulli randomization mechanism becomes πi
(different probabilities for different units), which could be modeled, for instance, as πi ¼ πðriÞ for a parametric choice of the
function πðÞ.
M. D. Cattaneo et al.: Randomization Inference in the Regression Disco

- Assumption 2: Local stable unit treatment value assumption. For all i with Ri 2 W0: if zi ¼ ~zi then
yiðzW0Þ ¼ yið~zW0Þ.
This assumption means that unit i’s potential outcome depends only on zi, which, together with


---

## 方法论/识别策略

extends directly to the fuzzy RD designs, offering a robust inference alternative to the traditional

---

## 估计方法

approaches. We illustrate our framework with a study of two measures of party-level advantage in U.S.
Senate elections, where the number of close races is small and our framework is well suited for the
empirical analysis.
Keywords: regression discontinuity, randomization inference, as-if randomization, incumbency advantage,
U.S. Senate
DOI 10.1515/jci-2013-0010
1 Introduction
Inference on the causal effects of a treatment is one of the basic aims of empirical research. In
observational studies, where controlled experimentation is not available, applied work relies on quasi-
experimental strategies carefully tailored to eliminate the effect of potential confounders that would
otherwise compromise the validity of the analysis. Originally proposed by Thistlethwaite and Campbell
[1], the regression discontinuity (RD) design has recently become one of the most widely used quasi-
experimental strategies. In this design, units receive treatment based on whether their value of an
observed covariate or “score” is above or below a fixed cutoff. The key feature of the design is that
the probability of receiving the treatment conditional on the score jumps discontinuously at the cutoff,
inducing variation in treatment assignment that is assumed to be unrelated to potential confounders.
Imbens and Lemieux [2], Lee and Lemieux [3] and Dinardo and Lee [4] give recent reviews, including
comprehensive lists of empirical examples.
The traditional inference approach in the RD design relies on flexible extrapolation (usually
nonparametric curve estimation techniques) using observations near the known cutoff. This approach
*Corresponding author: Rocío Titiunik, Department of Political Science, University of Michigan, 5700 Haven Hall, 505 South
State St, Ann Arbor, MI, USA, E-mail: titiunik@umich.edu
Matias D. Cattaneo, Department of Economics, University of Michigan, Ann Arbor, MI, USA, E-mail: cattaneo@umich.edu
Brigham R. Frandsen, Department of Economics, Brigham Young University, Provo, UT, USA, E-mail: frandsen@byu.edu
J. Causal Infer. 2015; 3(1): 1–24


follows the work of Hahn et al. [5], who showed that, when placement relative to the cutoff completely
determines treatment assignment, the key identifying assumption is that the conditional expectation of
a potential outcome is continuous at the threshold. Intuitively, since nothing changes abruptly at the
threshold other than the probability of receiving treatment, any jump in the conditional expectation of
the outcome variable at the threshold is attributed to the effects of the treatment. Modern RD analysis
employs local nonparametric curve estimation at either side of the threshold to estimate RD treatment
effects, with local-linear regression being the preferred choice in most cases. See Porter [6], Imbens
and Kalyanaraman [7] and Calonico et al. [8] for related theoretical results and further discussion.
Although not strictly justified by the standard framework, RD designs are routinely interpreted as
loc

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

