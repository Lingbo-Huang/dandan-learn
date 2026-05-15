---
layout: default
title: "AdaBoost：弱学习器到强学习器"
render_with_liquid: false
---

# AdaBoost：弱学习器到强学习器

## 直觉：专家委员会

如果一个专家只能回答 60% 的问题，但你有 100 个这样的专家，让他们投票……

关键在于：**每次让下一个专家重点学习前一个专家答错的那些问题**。

## AdaBoost 算法推导

### 设置

- 训练集：$\{(x_i, y_i)\}_{i=1}^N$，$y_i \in \{-1, +1\}$
- 弱学习器：$h_t(x) \in \{-1, +1\}$（通常是 max_depth=1 决策树桩）
- 初始权重：$w_i^{(1)} = \frac{1}{N}$

### 迭代过程（共 $T$ 轮）

**Step 1**：用当前权重训练弱学习器 $h_t$

**Step 2**：计算加权错误率：

$$\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i^{(t)}$$

**Step 3**：计算弱学习器权重（错误率越低，权重越高）：

$$\alpha_t = \frac{1}{2} \ln\frac{1-\epsilon_t}{\epsilon_t}$$

**Step 4**：更新样本权重：

$$w_i^{(t+1)} = \frac{w_i^{(t)} \exp(-\alpha_t y_i h_t(x_i))}{Z_t}$$

其中 $Z_t$ 是归一化因子。

**效果**：被 $h_t$ 错分的样本权重增大（乘以 $e^{\alpha_t}$），正确分类的权重减小（乘以 $e^{-\alpha_t}$）。

### 最终分类器

$$H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$$

### 训练误差上界（指数收敛）

$$\text{Error}_{\text{train}} \leq \exp\left(-2\sum_{t=1}^T \gamma_t^2\right)$$

其中 $\gamma_t = 0.5 - \epsilon_t$（比随机猜测好多少）。只要每个弱学习器比随机好一点点，整体误差指数级下降！

## Python 实现

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.estimators = []
    
    def fit(self, X, y):
        n = len(y)
        # y 必须是 {-1, +1}
        y_ = np.where(y == 0, -1, 1)
        w = np.ones(n) / n
        
        for t in range(self.n_estimators):
            # 1. 训练加权弱学习器（决策树桩）
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y_, sample_weight=w)
            
            pred = stump.predict(X)
            
            # 2. 计算加权错误率
            wrong = (pred != y_)
            eps = w[wrong].sum()
            eps = np.clip(eps, 1e-10, 1 - 1e-10)
            
            # 3. 计算弱学习器权重
            alpha = 0.5 * np.log((1 - eps) / eps)
            
            # 4. 更新样本权重
            w = w * np.exp(-alpha * y_ * pred)
            w = w / w.sum()  # 归一化
            
            self.alphas.append(alpha)
            self.estimators.append(stump)
        
        return self
    
    def predict(self, X):
        raw = sum(alpha * est.predict(X) 
                  for alpha, est in zip(self.alphas, self.estimators))
        return (raw > 0).astype(int)
    
    def staged_predict(self, X):
        """逐步添加弱学习器，返回每步预测"""
        raw = np.zeros(len(X))
        for alpha, est in zip(self.alphas, self.estimators):
            raw += alpha * est.predict(X)
            yield (raw > 0).astype(int)


# ---- 测试 ----
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

ada = AdaBoost(n_estimators=100)
ada.fit(X_tr, y_tr)
print(f"手写 AdaBoost 准确率: {(ada.predict(X_te) == y_te).mean():.4f}")

# sklearn 对比
from sklearn.ensemble import AdaBoostClassifier
sk_ada = AdaBoostClassifier(n_estimators=100, random_state=42)
sk_ada.fit(X_tr, y_tr)
print(f"sklearn AdaBoost 准确率: {sk_ada.score(X_te, y_te):.4f}")

# ---- 学习过程可视化 ----
test_accs = []
for preds in ada.staged_predict(X_te):
    test_accs.append((preds == y_te).mean())

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(test_accs)+1), test_accs, 'b-')
plt.xlabel('弱学习器数量')
plt.ylabel('测试准确率')
plt.title('AdaBoost：随弱学习器增加的测试准确率')
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

## 面试要点

**Q: AdaBoost 的损失函数是什么？**

A: AdaBoost 等价于以**指数损失**为目标的加法模型：$L(y, F(x)) = e^{-yF(x)}$，通过前向分步算法逐步最小化。

**Q: AdaBoost 为什么不容易过拟合（在一定程度上）？**

A: 理论上随着弱学习器增多，训练误差以指数速度下降，但测试误差下降后趋于平稳甚至继续下降——因为集成增大了 margin（决策边界的置信度）。但噪声数据下仍可过拟合（异常点权重越来越大）。
