# 校准笔记: Wanna Get Away? Regression Discontinuity Estimation of Exam School Effects Away From the Cutoff

> **技能**: estimator-rd
> **论文 ID**: 59e799b0dab947c72729014abde156053227dd2c
> **年份**: 2015
> **期刊**: 
> **引用数**: 223

---

## 摘要

In the canonical regression discontinuity (RD) design for applicants who face an award or admissions
cutoff, causal effects are nonparametrically identified for those near the cutoff. The impact of treatment on
inframarginal applicants is also of interest, but identification of such effects requires stronger assumptions
than are required for identification at the cutoff. This paper discusses RD identification away from
the cutoff. Our identification strategy exploits the availability of dependent variable predictors other
than the running variable. Conditional on these predictors, the running variable is assumed to be ignorable.
This identification strategy is illustrated with data on applicants to Boston exam schools. Functional-form-based
extrapolation generates unsatisfying results in this context, either noisy or not very robust. By contrast,
identification based on RD-specific conditional independence assumptions produces reasonably precise
and surprisingly robust estimates of the effects of exam school attendance on inframarginal applicants.
These estimates suggest that the causal effects of exam school attendance for 9th grade applicants
with running variable values well away from admissions cutoffs differ little from those for applicants
with values that put them on the margin of acceptance. An extension to fuzzy designs is shown to identify
causal effects for compliers away from the cutoff.
Joshua Angrist
Department of Economics
MIT, E52-353
50 Memorial Drive
Cambridge, MA  02142-1347
and NBER
angrist@mit.edu
Miikka Rokkanen
Department of Economics
MIT, E52-391
50 Memorial Drive
Cambridge, MA 02142-1347
rokkanen@mit.edu


Both the tie-breaking experiment and the regression-discontinuity analysis are partic-
ularly subject to the external validity limitation of selection-X interaction in that the
eﬀect has been demonstrated only for a very narrow band of talent, i.e., only for those
at the cutting score... Broader generalizations involve the extrapolation of the below-X
ﬁt across the entire range of X values, and at each greater degree of extrapolation, the
number of plausible rival hypotheses becomes greater.
– Donald T. Campbell and Julian Stanley (1963; Experimental and Quasi-Experimental
Designs for Research)

---

## 核心假设

未提取到假设部分

---

## 方法论/识别策略

for other exam schools, including New York’s well-known selective high schools. The second oldest
Boston exam school is Boston Latin Academy (BLA), formerly Girls’ Latin School. Opened in 1877,
BLA ﬁrst admitted boys in 1972 and currently enrolls about 1,700 students. The John D. O’Bryant
High School of Mathematics and Science (formerly Boston Technical High) is Boston’s third exam
school; O’Bryant opened in 1893 and now enrolls about 1,200 students.
The Boston Public School (BPS) system spans a wide range of peer achievement. Like urban
students elsewhere in the U.S., Boston exam school applicants who fail to enroll in an exam school
end up at schools with average SAT scores well below the state average, in this case, at schools close to
the 5th percentile of the distribution of school averages in the state. By contrast, O’Bryant’s average
SAT score falls at about the 40th percentile of the state distribution of averages, a big step up from
the overall BPS average, but not elite in an absolute sense.
Successful Boston BLA applicants
ﬁnd themselves at a school with average score around the 80th percentile of the distribution of
school means, while the average SAT score at BLS is the fourth highest among public schools in
3


Massachusetts.
Abdulkadiroğlu, Angrist, and Pathak (2012) investigate the causal eﬀects of exam school atten-
dance in a fuzzy RD setup, where exam school oﬀers are used as instrumental variables for mediating
channels that might explain exam school impacts. Because Boston’s exams are ordered by selectiv-
ity, unsuccessful applicants to BLS and BLA mostly ﬁnd themselves in another exam school. Still,
applicants admitted to each of the three schools are exposed to marked changes in peer composition.
An O’Bryant oﬀer increases average baseline (4th or 8th grade) peer scores by roughly three-fourths
of a standard deviation, while the peer achievement gain is almost .4σ at the BLA cutoﬀ, and about
.7σ at the BLS cutoﬀ. First-stages for racial composition also show that exam school oﬀers induce
a 12-24 percentage point reduction in the proportion of non-white classmates at each exam school
cutoﬀ. Peer achievement and racial composition are not the only channels by which an exam school
education might matter, but they are clearly important features of the exam school experiment.
Here, we’re initially interested in the eﬀects of an exam school oﬀer for students away from
admissions cutoﬀs, without the complication of adjustment for possible mediators. We therefore
begin by focusing on what amounts to the reduced form relation in the Abdulkadiroğlu, Angrist,
and Pathak (2012) analysis.
In the background, however, is the idea that mediators like exam
school enrollment and peer composition explain why we might expect an exam school eﬀect in the
ﬁrst place. Following the discussion of reduced form oﬀer eﬀects, we also consider an extension to
fuzzy RD with causal mediators, in this case, a dummy for exam school enrollment and an ordered

---

## 估计方法

of Exam School Effects Away From the Cutoff
The MIT Faculty has made this article openly available. Please share
how this access benefits you. Your story matters.
Citation: Angrist, Joshua D. and Rokkanen, Miikka. “Wanna Get Away? Regression Discontinuity 
Estimation of Exam School Effects Away From the Cutoff.” Journal of the American Statistical 
Association 110, 512 (October 2015): 1331–1344 © 2015 American Statistical Association
As Published: http://dx.doi.org/10.1080/01621459.2015.1012259
Publisher: Informa UK Limited
Persistent URL: http://hdl.handle.net/1721.1/113692
Version: Original manuscript: author's manuscript prior to formal peer review
Terms of use: Creative Commons Attribution-Noncommercial-Share Alike


NBER WORKING PAPER SERIES
WANNA GET AWAY? RD IDENTIFICATION AWAY FROM THE CUTOFF
Joshua Angrist
Miikka Rokkanen
Working Paper 18662
http://www.nber.org/papers/w18662
NATIONAL BUREAU OF ECONOMIC RESEARCH
1050 Massachusetts Avenue
Cambridge, MA 02138
December 2012
Our thanks to Parag Pathak for many helpful discussions and comments, and to seminar participants
at Berkeley, CREST, and Stanford for helpful comments. Thanks also go to Peter Hull for expert research
assistance. Angrist gratefully acknowledges funding from the Institute for Education Sciences. The
views expressed here are those of the authors alone and do not necessarily reflect the views of the
National Bureau of Economic Research or The Institute for Education Sciences.
NBER working papers are circulated for discussion and comment purposes. They have not been peer-
reviewed or been subject to the review by the NBER Board of Directors that accompanies official
NBER publications.
© 2012 by Joshua Angrist and Miikka Rokkanen. All rights reserved. Short sections of text, not to
exceed two paragraphs, may be quoted without explicit permission provided that full credit, including
© notice, is given to the source.


Wanna Get Away? RD Identification Away from the Cutoff
Joshua Angrist and Miikka Rokkanen
NBER Working Paper No. 18662
December 2012
JEL No. C26,C31,C36,I21,I24,I28,J24
ABSTRACT
In the canonical regression discontinuity (RD) design for applicants who face an award or admissions
cutoff, causal effects are nonparametrically identified for those near the cutoff. The impact of treatment on
inframarginal applicants is also of interest, but identification of such effects requires stronger assumptions
than are required for identification at the cutoff. This paper discusses RD identification away from
the cutoff. Our identification strategy exploits the availability of dependent variable predictors other
than the running variable. Conditional on these predictors, the running variable is assumed to be ignorable.
This identification strategy is illustrated with data on applicants to Boston exam schools. Functional-form-based
extrapolation generates unsatisfying results in this context, either noisy or not very robust. By contrast,
identification based on RD-specific condit

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

