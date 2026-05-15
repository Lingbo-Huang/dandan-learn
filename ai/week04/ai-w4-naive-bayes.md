---
layout: default
title: "朴素贝叶斯分类器"
render_with_liquid: false
---

# 朴素贝叶斯分类器

## 直觉先行：垃圾邮件过滤器

收到一封邮件，含有"免费"、"领奖"、"点击"这些词，它是垃圾邮件的概率有多大？

朴素贝叶斯的思路：**已知邮件内容，反推它来自哪个"类别"的可能性**。

核心公式只有一个：

$$P(\text{类别} \mid \text{内容}) = \frac{P(\text{内容} \mid \text{类别}) \times P(\text{类别})}{P(\text{内容})}$$

## 贝叶斯定理推导

### 条件概率基础

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

由此可得：

$$P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

整理后得到贝叶斯定理：

$$\boxed{P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}}$$

### 用于分类

给定样本特征 $\mathbf{x} = (x_1, x_2, \ldots, x_n)$，预测类别 $y$：

$$P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) \cdot P(y)}{P(\mathbf{x})}$$

分类决策（取最大后验概率）：

$$\hat{y} = \arg\max_y P(y \mid \mathbf{x}) = \arg\max_y P(\mathbf{x} \mid y) \cdot P(y)$$

分母 $P(\mathbf{x})$ 对所有类别相同，可以忽略。

## "朴素"在哪里——条件独立假设

计算 $P(\mathbf{x} \mid y) = P(x_1, x_2, \ldots, x_n \mid y)$ 需要联合概率，组合爆炸。

**朴素假设**：在已知类别 $y$ 的条件下，各特征相互独立：

$$P(\mathbf{x} \mid y) = \prod_{i=1}^{n} P(x_i \mid y)$$

于是分类器变为：

$$\hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y)$$

实际计算取对数（避免数值下溢）：

$$\hat{y} = \arg\max_y \left[\log P(y) + \sum_{i=1}^{n} \log P(x_i \mid y)\right]$$

## 三种朴素贝叶斯变体

### 1. 高斯朴素贝叶斯（连续特征）

假设每个类别的每个特征服从高斯分布：

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_{y,i}^2}} \exp\left(-\frac{(x_i - \mu_{y,i})^2}{2\sigma_{y,i}^2}\right)$$

参数估计（用训练集 MLE）：
- $\mu_{y,i} = \frac{1}{N_y} \sum_{j: y_j=y} x_{j,i}$
- $\sigma_{y,i}^2 = \frac{1}{N_y} \sum_{j: y_j=y} (x_{j,i} - \mu_{y,i})^2$

### 2. 多项式朴素贝叶斯（文本计数特征）

$$P(x_i \mid y) = \frac{N_{y,i} + \alpha}{N_y + \alpha \cdot n}$$

其中 $\alpha$ 是拉普拉斯平滑（避免零概率）。

### 3. 伯努利朴素贝叶斯（二值特征）

$$P(x_i \mid y) = P(i \mid y)^{x_i} \cdot (1 - P(i \mid y))^{1-x_i}$$

## Python 实现（从零开始）

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors_ = {}
        self.means_ = {}
        self.vars_ = {}
        
        for c in self.classes_:
            X_c = X[y == c]
            self.priors_[c] = len(X_c) / len(X)
            self.means_[c] = X_c.mean(axis=0)
            self.vars_[c] = X_c.var(axis=0) + 1e-9  # 防零
        return self
    
    def _log_likelihood(self, x, c):
        mu = self.means_[c]
        var = self.vars_[c]
        # 高斯对数概率密度
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * var) + (x - mu)**2 / var)
        return log_prob
    
    def predict(self, X):
        predictions = []
        for x in X:
            log_posteriors = {}
            for c in self.classes_:
                log_prior = np.log(self.priors_[c])
                log_likelihood = self._log_likelihood(x, c)
                log_posteriors[c] = log_prior + log_likelihood
            predictions.append(max(log_posteriors, key=log_posteriors.get))
        return np.array(predictions)

# 测试
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"手写 GaussianNB 准确率: {accuracy:.4f}")

# 与 sklearn 对比
from sklearn.naive_bayes import GaussianNB
sk_gnb = GaussianNB()
sk_gnb.fit(X_train, y_train)
print(f"sklearn GaussianNB 准确率: {sk_gnb.score(X_test, y_test):.4f}")
```

## 拉普拉斯平滑（处理零概率）

若某个词从未在训练集的某类别中出现，直接计算概率为 0，会使整个乘积归零。

**平滑方法**：给每个计数加 $\alpha$（通常 $\alpha=1$）：

$$P(x_i \mid y) = \frac{\text{count}(x_i, y) + \alpha}{\text{count}(y) + \alpha \cdot |V|}$$

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# 文本分类示例
corpus = [
    "免费领奖点击这里",
    "会议明天下午三点",
    "恭喜您中奖免费领取",
    "项目进度汇报",
    "限时免费优惠点击",
    "周报请查收",
]
labels = [1, 0, 1, 0, 1, 0]  # 1=垃圾, 0=正常

vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,1))
X = vectorizer.fit_transform(corpus)

# 拉普拉斯平滑 alpha=1
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X, labels)

test = vectorizer.transform(["免费点击领奖"])
print(f"预测: {'垃圾' if mnb.predict(test)[0] == 1 else '正常'}")
print(f"概率: {mnb.predict_proba(test)[0]}")
```

## 优缺点分析

| 维度 | 评价 |
|------|------|
| 训练速度 | ⚡⚡⚡ 极快（只需统计均值/方差） |
| 预测速度 | ⚡⚡⚡ 极快 |
| 小样本 | ✅ 表现好，参数少不易过拟合 |
| 高维文本 | ✅ 多项式NB是文本分类经典 |
| 特征相关性强 | ❌ 条件独立假设被严重违反时退化 |
| 连续特征 | ⚠️ 需要分布假设（高斯等） |

## 面试要点

**Q: 朴素贝叶斯为什么"朴素"，这个假设实际成立吗？**

A: "朴素"指条件独立假设——已知类别时各特征独立。现实中几乎不成立（如文本中"机器"和"学习"高度相关），但朴素贝叶斯在实践中依然有效，原因在于：分类只需比较不同类别的相对大小，而非精确概率；特征间的相关性对各类别的影响可能相互抵消。

**Q: 朴素贝叶斯 vs 逻辑回归？**

A: NB 是生成式模型（建模 P(x|y)），LR 是判别式模型（直接建模 P(y|x)）。NB 在小数据集上更稳，LR 在数据充足时通常更准。

**Q: 如何处理连续特征？**

A: 三种方式：① 离散化（分箱）后用多项式NB；② 假设高斯分布用高斯NB；③ 用核密度估计（更灵活）。
