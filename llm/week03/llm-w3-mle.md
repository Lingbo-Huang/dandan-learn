---
layout: default
title: "D5 · 极大似然估计与统计学习基础"
---

# D5 · 极大似然估计

> **LLM Week 3**  
> 大模型训练的统计本质：找到最可能"生成了训练数据"的参数。

---

## 一、问题：参数估计

**场景**：你有一枚不公平的硬币，抛了 10 次，正面 7 次，反面 3 次。  
这枚硬币正面朝上的概率 $p$ 是多少？

**直觉答案**：$p = 0.7$

**问题**：为什么是 0.7，而不是 0.5 或 0.6？

极大似然估计（MLE）给出了严格的数学框架。

---

## 二、似然函数

**似然函数**：给定参数 $\theta$，观测到数据 $\mathcal{D}$ 的概率。

$$L(\theta | \mathcal{D}) = P(\mathcal{D} | \theta)$$

对于 10 次硬币实验（7 正 3 反）：

$$L(p | \text{数据}) = p^7 (1-p)^3$$

---

## 三、MLE：最大化似然

**MLE 原则**：找到使 $L(\theta|\mathcal{D})$ 最大的参数 $\hat{\theta}$。

$$\hat{\theta}_{MLE} = \arg\max_\theta L(\theta | \mathcal{D})$$

实践中通常最大化**对数似然**（避免数值下溢）：

$$\hat{\theta}_{MLE} = \arg\max_\theta \log L(\theta | \mathcal{D})$$

**硬币例子求解**：

$$\log L(p) = 7\log p + 3\log(1-p)$$

$$\frac{d \log L}{dp} = \frac{7}{p} - \frac{3}{1-p} = 0$$

$$\hat{p} = \frac{7}{7+3} = 0.7 \quad \checkmark$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# 可视化似然函数
p_values = np.linspace(0.01, 0.99, 100)
log_likelihood = 7 * np.log(p_values) + 3 * np.log(1 - p_values)

plt.plot(p_values, log_likelihood)
plt.axvline(x=0.7, color='r', linestyle='--', label='MLE = 0.7')
plt.xlabel('p（正面概率）')
plt.ylabel('对数似然')
plt.title('硬币实验的对数似然函数')
plt.legend()
plt.show()
```

---

## 四、大模型训练 = MLE

语言模型的参数 $\theta$（几百亿个权重）的 MLE：

$$\hat{\theta}_{MLE} = \arg\max_\theta \sum_{t=1}^T \log P_\theta(w_t | w_1, \ldots, w_{t-1})$$

等价于最小化负对数似然，也就是**交叉熵损失**：

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^T \log P_\theta(w_t | w_{<t})$$

**这把 MLE 和交叉熵统一起来了！**

---

## 五、MLE vs MAP（贝叶斯先验）

**MLE**：只用数据，不引入先验。

**MAP（最大后验估计）**：引入参数的先验分布。

$$\hat{\theta}_{MAP} = \arg\max_\theta \left[ \log P(\mathcal{D}|\theta) + \log P(\theta) \right]$$

**MAP = MLE + 正则化！**

| 先验 | 正则化 |
|------|--------|
| 高斯先验 $\theta \sim \mathcal{N}(0, \sigma^2)$ | L2 正则化（Ridge） |
| 拉普拉斯先验 $\theta \sim \text{Laplace}(0, b)$ | L1 正则化（Lasso） |

```python
# MLE vs Ridge（MAP with Gaussian prior）的等价性
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

# 生成数据
np.random.seed(42)
X = np.random.randn(50, 5)
y = X @ np.array([1, 2, 0, -1, 0.5]) + np.random.randn(50) * 0.5

# MLE = 普通线性回归（最小化 MSE = 最大化高斯似然）
mle_model = LinearRegression()
mle_model.fit(X, y)

# MAP with Gaussian prior = Ridge 回归
map_model = Ridge(alpha=1.0)
map_model.fit(X, y)

print("MLE (OLS) 系数:", mle_model.coef_.round(3))
print("MAP (Ridge) 系数:", map_model.coef_.round(3))
print("真实系数: [1, 2, 0, -1, 0.5]")
```

---

## 六、统计学习的泛化边界

**VC 维（Vapnik-Chervonenkis dimension）**：衡量模型复杂度的理论量。

泛化误差的上界：

$$\text{泛化误差} \leq \text{训练误差} + O\left(\sqrt{\frac{d_{VC}}{m}}\right)$$

其中 $m$ 是样本数，$d_{VC}$ 是 VC 维。

**直觉**：
- 模型越复杂（VC 维越大），需要越多数据才能保证泛化
- 大模型有几千亿参数，却在万亿 token 上训练——量变产生质变

---

## 七、大语言模型为什么有效？

从 MLE 角度理解：

```
1. 训练数据：互联网上的万亿 token
2. 目标：找到参数 θ 使得 P(数据|θ) 最大
3. 隐含假设：人类写的文字背后有规律可学
4. 神奇之处：为了"预测下一个词"，模型不得不学会：
   - 语法（不然句子不通）
   - 事实（不然内容错误）
   - 推理（不然上下文不一致）
   - 代码（训练数据里有大量代码）
```

这就是为什么预训练模型是"通才"——预测下一个词这个任务，逼着模型理解世界。

---

## 今天的关键认识

1. **MLE = 找最可能生成数据的参数**
2. **大模型训练 = MLE on 语言模型 = 最小化交叉熵损失**
3. **MAP = MLE + 先验 = 带正则化的 MLE**
4. **泛化需要足够数据**，大模型的成功是算力 + 数据的胜利

---

## 明天预告

D6：**PyTorch 自动微分实战**——用真正的框架感受自动求梯度的威力。
