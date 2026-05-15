---
layout: default
title: "支持向量机 SVM"
render_with_liquid: false
---

# 支持向量机 SVM

## 直觉：最大间隔分类器

逻辑回归找到"能分开数据的一条线"，而 SVM 找到"**离两类数据都尽可能远的那条线**"。

这条线的间隔越大，泛化能力越强——这是 SVM 的核心思想。

## 硬间隔 SVM（线性可分情形）

### 决策函数

$$f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b$$

预测规则：$\hat{y} = \text{sign}(f(\mathbf{x}))$，标签 $y \in \{-1, +1\}$。

### 函数间隔与几何间隔

点 $(\mathbf{x}_i, y_i)$ 到超平面的**几何距离**：

$$d_i = \frac{y_i(\mathbf{w}^T\mathbf{x}_i + b)}{\|\mathbf{w}\|}$$

**间隔**（margin）= 最近点到超平面距离的 2 倍：

$$\text{margin} = \frac{2}{\|\mathbf{w}\|}$$

### 优化问题

最大化间隔 = 最小化 $\|\mathbf{w}\|^2$：

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2$$
$$\text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad \forall i$$

### 对偶问题与支持向量

引入拉格朗日乘子 $\alpha_i \geq 0$：

$$\mathcal{L} = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_i \alpha_i [y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1]$$

KKT 条件：
- $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$
- $\sum_i \alpha_i y_i = 0$
- $\alpha_i \geq 0$，且 $\alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i+b)-1] = 0$

**支持向量**：$\alpha_i > 0$ 的样本（恰好在间隔边界上），其他点 $\alpha_i = 0$，对模型无影响。

对偶目标（最大化）：

$$\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i\alpha_j y_iy_j \mathbf{x}_i^T\mathbf{x}_j$$

决策函数只依赖**内积**：

$$f(\mathbf{x}) = \sum_i \alpha_i y_i \mathbf{x}_i^T\mathbf{x} + b$$

## 软间隔 SVM（允许错误）

引入松弛变量 $\xi_i \geq 0$：

$$\min_{\mathbf{w},b,\boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i$$
$$\text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i+b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**C 参数**：
- C 大 → 惩罚误分类重 → 间隔小，更拟合训练集（可能过拟合）
- C 小 → 容忍误分类 → 间隔大，更简单（可能欠拟合）

等价于 Hinge Loss + L2 正则：

$$\mathcal{L} = \frac{1}{N}\sum_i \max(0, 1 - y_i f(\mathbf{x}_i)) + \frac{1}{2C}\|\mathbf{w}\|^2$$

## 核技巧：处理非线性

关键洞察：决策函数只依赖**内积** $\mathbf{x}_i^T\mathbf{x}_j$。

若用映射 $\phi(\mathbf{x})$ 将特征变换到高维，内积变为 $\phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j) = K(\mathbf{x}_i, \mathbf{x}_j)$。

只要核函数 $K$ 满足 Mercer 条件，就不需要显式计算 $\phi$！

| 核函数 | 公式 | 适用场景 |
|--------|------|---------|
| 线性核 | $K(\mathbf{x},\mathbf{z}) = \mathbf{x}^T\mathbf{z}$ | 高维稀疏（文本） |
| 多项式核 | $K(\mathbf{x},\mathbf{z}) = (\mathbf{x}^T\mathbf{z}+c)^d$ | 中等维度 |
| RBF/高斯核 | $K(\mathbf{x},\mathbf{z}) = e^{-\gamma\|\mathbf{x}-\mathbf{z}\|^2}$ | 最常用，适合低维密集 |
| Sigmoid 核 | $K(\mathbf{x},\mathbf{z}) = \tanh(\kappa\mathbf{x}^T\mathbf{z}+\theta)$ | 类神经网络 |

## Python 实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# ---- 非线性数据：核SVM ----
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2)

# 线性核 vs RBF 核
for kernel in ['linear', 'rbf']:
    svc = SVC(kernel=kernel, C=1.0, gamma='scale')
    svc.fit(X_train, y_train)
    print(f"{kernel} kernel: {svc.score(X_test, y_test):.4f}")

# ---- 可视化决策边界 ----
def plot_svm_boundary(clf, X, y, title):
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=40)
    
    # 标注支持向量
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=150, linewidth=2, facecolors='none', edgecolors='k')
    plt.title(title)
    plt.tight_layout()

rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_svm.fit(X_s, y)
plot_svm_boundary(rbf_svm, X_s, y, "RBF SVM 决策边界（圆圈=支持向量）")

# ---- 超参数调优 ----
cancer = load_breast_cancer()
X_c = StandardScaler().fit_transform(cancer.data)
X_tr, X_te, y_tr, y_te = train_test_split(X_c, cancer.target, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf']
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_tr, y_tr)
print(f"最优参数: {grid_search.best_params_}")
print(f"测试准确率: {grid_search.best_estimator_.score(X_te, y_te):.4f}")
```

## 手写简化版 SVM（梯度下降 Hinge Loss）

```python
class SimpleSVM:
    """基于 Hinge Loss + SGD 的线性 SVM"""
    def __init__(self, C=1.0, lr=0.001, n_iter=1000):
        self.C = C
        self.lr = lr
        self.n_iter = n_iter
    
    def fit(self, X, y):
        # y 需要是 {-1, +1}
        y_ = np.where(y == 0, -1, 1)
        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0.0
        
        for _ in range(self.n_iter):
            for i in range(n):
                margin = y_[i] * (X[i] @ self.w + self.b)
                if margin < 1:
                    # 在间隔内或误分类
                    self.w += self.lr * (self.C * y_[i] * X[i] - self.w)
                    self.b += self.lr * self.C * y_[i]
                else:
                    # 正确分类且在间隔外
                    self.w += self.lr * (-self.w)
        return self
    
    def predict(self, X):
        return np.sign(X @ self.w + self.b)

# 测试
from sklearn.datasets import make_blobs
X_b, y_b = make_blobs(n_samples=200, centers=2, random_state=42)
y_b = np.where(y_b == 0, -1, 1)
X_b_s = StandardScaler().fit_transform(X_b)

svm = SimpleSVM(C=1.0, lr=0.001, n_iter=2000)
svm.fit(X_b_s, y_b)
preds = svm.predict(X_b_s)
print(f"手写 SVM 准确率: {(preds == y_b).mean():.4f}")
```

## 面试要点

**Q: 什么是支持向量？如果训练集中加入非支持向量的点，模型会变吗？**

A: 支持向量是恰好在间隔边界上的点（满足 $y_i(\mathbf{w}^T\mathbf{x}_i+b)=1$）。加入非支持向量的点对模型没有任何影响，因为 $\alpha_i=0$。这是 SVM 的优雅性质。

**Q: RBF 核的 $\gamma$ 参数有什么作用？**

A: $\gamma$ 控制单个样本的影响范围。$\gamma$ 大 → 影响范围小 → 模型复杂（过拟合风险）；$\gamma$ 小 → 影响范围大 → 模型平滑（欠拟合风险）。

**Q: SVM 和逻辑回归如何选择？**

A: 小数据集、高维稀疏特征（如文本）→ 线性SVM/LR；非线性问题、中等数据量 → RBF SVM；大数据集 → LR（训练快）或梯度提升树。
