---
layout: default
title: "逻辑回归"
render_with_liquid: false
---

# 逻辑回归

## 直觉：线性回归 + 压缩函数

线性回归输出 $(-\infty, +\infty)$，而分类需要概率 $[0, 1]$。

解决办法：在线性模型外面套一个 **Sigmoid 函数**，把实数域压缩到 (0, 1)。

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w}^T \mathbf{x} + b$$

## 数学推导

### Sigmoid 函数

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

关键性质：
- $\sigma(0) = 0.5$（决策边界处）
- $\sigma(-\infty) = 0,\ \sigma(+\infty) = 1$
- **导数**：$\sigma'(z) = \sigma(z)(1 - \sigma(z))$ ← 面试必背！

### 概率解释

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \hat{y}$$
$$P(y=0 \mid \mathbf{x}) = 1 - \hat{y}$$

统一写为：

$$P(y \mid \mathbf{x}) = \hat{y}^y (1-\hat{y})^{1-y}$$

### 损失函数：交叉熵

最大似然估计（MLE）→ 最小化负对数似然：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

**为什么不用 MSE？** MSE + Sigmoid 导致梯度消失，且损失函数非凸（有多个局部最小值）。交叉熵是凸函数，保证全局最优。

### 梯度下降

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

更新规则：

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{w}}$$

注意梯度形式与线性回归几乎一样！差别在于 $\hat{y}$ 经过了 sigmoid。

### 正则化

L2（Ridge）：

$$\mathcal{L}_{reg} = \mathcal{L} + \frac{\lambda}{2}\|\mathbf{w}\|^2$$

对应 sklearn 的 `C = 1/λ`（C 越小正则越强）。

## Python 实现（从零开始）

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000, lambda_=0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.lambda_ = lambda_
        self.losses = []
    
    def sigmoid(self, z):
        # 防止溢出
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))
    
    def fit(self, X, y):
        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0.0
        
        for _ in range(self.n_iter):
            z = X @ self.w + self.b
            y_hat = self.sigmoid(z)
            
            # 交叉熵损失 + L2正则
            loss = -np.mean(y * np.log(y_hat + 1e-9) + 
                           (1-y) * np.log(1 - y_hat + 1e-9))
            loss += (self.lambda_ / 2) * np.sum(self.w**2)
            self.losses.append(loss)
            
            # 梯度
            dw = (X.T @ (y_hat - y)) / n + self.lambda_ * self.w
            db = np.mean(y_hat - y)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
        
        return self
    
    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# ---- 实验 ----
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(lr=0.1, n_iter=500, lambda_=0.01)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = (y_pred == y_test).mean()
print(f"手写 LR 准确率: {acc:.4f}")

# 绘制损失曲线
plt.figure(figsize=(8, 4))
plt.plot(model.losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
# plt.savefig("lr_loss.png")

# 与 sklearn 对比
from sklearn.linear_model import LogisticRegression as SkLR
sk_lr = SkLR(max_iter=1000, C=100)
sk_lr.fit(X_train, y_train)
print(f"sklearn LR 准确率: {sk_lr.score(X_test, y_test):.4f}")
```

## 决策边界可视化（二维数据）

```python
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                            n_clusters_per_class=1, random_state=42)

from sklearn.linear_model import LogisticRegression as SkLR
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

lr = SkLR()
lr.fit(X_s, y)

# 画决策边界
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
plt.scatter(X_s[:, 0], X_s[:, 1], c=y, cmap='RdBu', edgecolors='k', s=40)

# 画决策边界线 w1*x1 + w2*x2 + b = 0
w = lr.coef_[0]
b = lr.intercept_[0]
x_line = np.linspace(-3, 3, 100)
y_line = -(w[0] * x_line + b) / w[1]
plt.plot(x_line, y_line, 'k--', linewidth=2, label='决策边界')
plt.legend()
plt.title("逻辑回归决策边界")
plt.tight_layout()
```

## 多分类：Softmax 回归

二分类 → 多分类，Sigmoid → Softmax：

$$P(y=k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T\mathbf{x}}}$$

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
lr_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
lr_multi.fit(iris.data, iris.target)
print(f"多分类准确率: {lr_multi.score(iris.data, iris.target):.4f}")
print(f"权重矩阵形状: {lr_multi.coef_.shape}")  # (3, 4) — 3类 × 4特征
```

## 面试要点

**Q: 为什么用交叉熵而不用 MSE 作为逻辑回归的损失函数？**

A: ① MSE + Sigmoid 损失函数非凸，存在多个局部最小值，梯度下降难以找到全局最优；② 从MLE角度，假设标签服从伯努利分布，自然推导出交叉熵；③ 交叉熵梯度形式简洁，不存在梯度消失问题（梯度正比于预测误差）。

**Q: sigmoid 导数推导？**

A: $\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma(z)(1-\sigma(z))$，在 $z=0$ 处最大值为 $0.25$。

**Q: 逻辑回归能处理非线性问题吗？**

A: 原始形式只能处理线性可分问题。通过特征工程（多项式特征、核方法）可以处理非线性。本质上决策边界是 $\mathbf{w}^T\mathbf{x} + b = 0$，是线性超平面。
