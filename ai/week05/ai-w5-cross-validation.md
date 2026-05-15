---
layout: default
title: "交叉验证与过拟合诊断"
render_with_liquid: false
---

# 交叉验证与过拟合诊断

## 偏差-方差分解

对于回归模型，期望误差可以分解为：

$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{[\mathbb{E}[\hat{f}(x)] - f(x)]^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$

| 概念 | 含义 | 问题 |
|------|------|------|
| 偏差 Bias | 模型假设与真实规律的偏差 | 过高 → 欠拟合 |
| 方差 Variance | 对不同训练集的敏感程度 | 过高 → 过拟合 |
| 噪声 Noise | 数据本身的随机性 | 无法消除 |

**关键权衡**：降低偏差（复杂模型）往往增加方差；降低方差（简单模型/正则化/集成）往往增加偏差。

## 交叉验证策略

### K-Fold CV（最常用）

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (KFold, StratifiedKFold, 
                                      cross_val_score, cross_validate,
                                      learning_curve)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
scores = cross_val_score(dt, X, y, cv=kf, scoring='accuracy')
print(f"5-Fold CV: {scores.mean():.4f} ± {scores.std():.4f}")

# Stratified K-Fold（分类任务推荐：保证每折类别比例一致）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_s = cross_val_score(dt, X, y, cv=skf, scoring='accuracy')
print(f"Stratified 5-Fold CV: {scores_s.mean():.4f} ± {scores_s.std():.4f}")
```

### 学习曲线：诊断过拟合/欠拟合

```python
def plot_learning_curve(estimator, X, y, title, cv=5):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'b-o', label='训练集')
    plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, color='b')
    plt.plot(train_sizes, val_mean, 'r-o', label='验证集')
    plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15, color='r')
    plt.xlabel('训练样本数')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

# 欠拟合：决策树 max_depth=1（太简单）
plot_learning_curve(
    Pipeline([('scaler', StandardScaler()),
              ('clf', DecisionTreeClassifier(max_depth=1, random_state=42))]),
    X, y, "欠拟合示例：决策树 max_depth=1"
)

# 正常拟合：随机森林
plot_learning_curve(
    Pipeline([('scaler', StandardScaler()),
              ('clf', RandomForestClassifier(n_estimators=100, random_state=42))]),
    X, y, "正常拟合：随机森林"
)

# 过拟合：决策树不限深度
plot_learning_curve(
    Pipeline([('scaler', StandardScaler()),
              ('clf', DecisionTreeClassifier(random_state=42))]),
    X, y, "过拟合示例：决策树不限深度"
)
```

### 验证曲线：选最优超参数

```python
from sklearn.model_selection import validation_curve

param_range = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X, y, param_name='max_depth',
    param_range=param_range,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)

plt.figure(figsize=(8, 5))
plt.plot(param_range, train_scores.mean(axis=1), 'b-o', label='训练集')
plt.fill_between(param_range,
                 train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1),
                 alpha=0.15, color='b')
plt.plot(param_range, val_scores.mean(axis=1), 'r-o', label='验证集')
plt.fill_between(param_range,
                 val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1),
                 alpha=0.15, color='r')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('验证曲线：决策树深度选择')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# 最优点：验证集曲线最高处
```

## 面试要点

**Q: 为什么需要验证集，只用训练集和测试集不行吗？**

A: 用测试集来选超参数，等于让模型间接看到了测试集（过拟合到测试集）。正确做法：训练集→训练，验证集→选超参，测试集→最终评估（只用一次）。

**Q: Leave-One-Out CV 什么时候用？**

A: 样本极少时（< 50）用 LOO-CV，充分利用数据。但计算开销是 $O(N)$ 倍，且高方差（每次只有一个验证样本）。
