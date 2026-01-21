# 校准笔记: Semismooth Newton Coordinate Descent Algorithm for Elastic-Net Penalized Huber Loss Regression and Quantile Regression

> **技能**: ml-model-linear
> **论文 ID**: 7a8022074d8da386b809dc9e639e44e1184367ac
> **年份**: 2015
> **期刊**: 
> **引用数**: 160

---

## 摘要

We propose an algorithm, semismooth Newton coordinate descent (SNCD), for the
elastic-net penalized Huber loss regression and quantile regression in high dimensional
settings. Unlike existing coordinate descent type algorithms, the SNCD updates each
regression coeﬃcient and its corresponding subgradient simultaneously in each itera-
tion. It combines the strengths of the coordinate descent and the semismooth Newton
algorithm, and eﬀectively solves the computational challenges posed by dimensional-
ity and nonsmoothness. We establish the convergence properties of the algorithm.
In addition, we present an adaptive version of the “strong rule" for screening predic-
tors to gain extra eﬃciency. Through numerical experiments, we demonstrate that
the proposed algorithm is very eﬃcient and scalable to ultra-high dimensions. We
illustrate the application via a real data example.

---

## 核心假设

未提取到假设部分

---

## 方法论/识别策略

yi = β0 + x⊤
i β + εi
where xi is a p-dimensional vector of covariates, (β0, β) are regression coeﬃcients, and εi
is the random error. We are interested in the high dimensional case where p ≫n and the
model is sparse in the sense that only a small proportion of the coeﬃcients are nonzero. In
1
arXiv:1509.02957v2  [stat.CO]  20 May 2016


such a scenario, a key task is identifying and estimating the nonzero coeﬃcients. A popular
approach is the penalized regression
min
β0,β
1
n
X
i
ℓ(yi −β0 −x⊤
i β) + λP(β),
(1.1)
where ℓis a generic loss function and p is a penalty function with a tuning parameter λ ≥0.
We consider the elastic-net penalty (Zou and Hastie, 2005)
P(β) ≡Pα(β) = α∥β∥1 + (1 −α)1
2∥β∥2
2, 0 ≤α ≤1,
which is a convex combination of the lasso (Tibshirani, 1996) (α = 1) and the ridge penalty
(α = 0).
A common choice for ℓis the squared loss ℓ(t) = t2/2, corresponding to the least squares
regression in classical regression literature. Although the squared loss is analytically simple,
it is not suitable for data in the presence of outliers or heterogeneity. Instead, we could
consider two widely used robust alternatives, the Huber loss (Huber, 1973) and the quantile
loss (Koenker and Bassett Jr, 1978).
The Huber loss is
ℓ(t) ≡hγ(t) =





t2
2γ,
if |t| ≤γ,
|t| −γ
2,
if |t| > γ,
(1.2)
where γ > 0 is a given constant. This function is quadratic for |t| ≤γ and linear for |t| > γ.
In addition, it is convex and ﬁrst-order diﬀerentiable. These features allow it to combine
analytical tractability of the squared loss for the least squares and outlier-robustness of the
absolute loss for the LAD regression.
The quantile loss is
ℓ(t) ≡ρτ(t) = t(τ −I(t < 0)), t ∈R,
(1.3)
where 0 < τ < 1. This is a generalization of the absolute loss with τ = 1/2. Rather
than the conditional mean of the response given the covariates, quantile regression models
conditional quantiles.
For heterogeneous data, the functional relationship between the
response and the covariates may vary in diﬀerent segments of its conditional distribution.
By choosing diﬀerent τ, quantile regression provides a powerful technique for exploring

---

## 估计方法

未提取到估计部分

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

