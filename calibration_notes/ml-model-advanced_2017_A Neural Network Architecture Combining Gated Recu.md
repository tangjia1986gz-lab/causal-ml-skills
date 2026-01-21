# 校准笔记: A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and Support Vector Machine (SVM) for Intrusion Detection in Network Traffic Data

> **技能**: ml-model-advanced
> **论文 ID**: 594cd092bb357f33e8f696ad0d5e0b5355f09157
> **年份**: 2017
> **期刊**: International Conference on Machine Learning and Computing
> **引用数**: 228

---

## 摘要

Gated Recurrent Unit (GRU) is a recently-developed variation of the
long short-term memory (LSTM) unit, both of which are variants
of recurrent neural network (RNN). Through empirical evidence,
both models have been proven to be effective in a wide variety of
machine learning tasks such as natural language processing[23],
speech recognition[4], and text classification[24]. Conventionally,
like most neural networks, both of the aforementioned RNN vari-
ants employ the Softmax function as its final output layer for its
prediction, and the cross-entropy function for computing its loss.
In this paper, we present an amendment to this norm by introduc-
ing linear support vector machine (SVM) as the replacement for
Softmax in the final output layer of a GRU model. Furthermore,
the cross-entropy function shall be replaced with a margin-based
function. While there have been similar studies[2, 22], this proposal
is primarily intended for binary classification on intrusion detec-
tion using the 2013 network traffic data from the honeypot systems
of Kyoto University. Results show that the GRU-SVM model per-
forms relatively higher than the conventional GRU-Softmax model.
The proposed model reached a training accuracy of ≈81.54% and
a testing accuracy of ≈84.15%, while the latter was able to reach a
training accuracy of ≈63.07% and a testing accuracy of ≈70.75%. In
addition, the juxtaposition of these two final output layers indicate
that the SVM would outperform Softmax in prediction time - a
theoretical implication which was supported by the actual training
and testing time in the study.
CCS CONCEPTS
• Computing methodologies →Supervised learning by clas-
sification; Support vector machines; Neural networks; • Se-
curity and privacy →Intrusion detection systems;

---

## 核心假设

未提取到假设部分

---

## 方法论/识别策略

2.1
Machine Intelligence Library
Google TensorFlow[1] was used to implement the neural network
models in this study – both the proposed and its comparator.
2.2
The Dataset
The 2013 Kyoto University honeypot systems’ network traffic data[20]
was used in this study. It has 24 statistical features[20]; (1) 14 fea-
tures from the KDD Cup 1999 dataset[21], and (2) 10 additional
features, which according to Song, Takakura, & Okabe (2006)[20],
arXiv:1709.03082v8  [cs.NE]  7 Feb 2019


ICMLC 2018, February 26–28, 2018, Macau, China
Abien Fred M. Agarap
might be pivotal in a more effective investigation on intrusion
detection. Only 22 dataset features were used in the study.
2.3

---

## 估计方法

未提取到估计部分

---

## 关键公式

1. `15.00
https://doi.org/10.1145/3195106.3195117
ACM Reference Format:
Abien Fred M. Agarap. 2018. A Neural Network Architecture Combin-
ing Gated Recurrent Unit (GRU) and Support Vector Machine (SVM) fo`


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

