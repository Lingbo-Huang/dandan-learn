---
layout: default
title: "决策树"
render_with_liquid: false
---

# 决策树

## 直觉：二十个问题游戏

"是动物吗？" → "有四条腿吗？" → "会叫吗？" → "是狗"。

决策树就是把这个问题链自动从数据中学习出来——每个节点找一个最好的问题（特征分裂）。

## 核心概念：如何衡量"好问题"

### 信息熵（ID3 / C4.5）

$$H(S) = -\sum_{k=1}^K p_k \log_2 p_k$$

- 纯净（全是一类）：$H = 0$
- 最混乱（各类均等）：$H$ 最大

**信息增益**（选让熵减少最多的特征）：

$$IG(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

### Gini 不纯度（CART / sklearn 默认）

$$\text{Gini}(S) = 1 - \sum_{k=1}^K p_k^2$$

与熵含义类似，计算更快（无 $\log$）。

**分裂准则**：

$$\Delta\text{Gini} = \text{Gini}(S) - \sum_v \frac{|S_v|}{|S|}\text{Gini}(S_v)$$

### 两者比较

| | 熵（ID3/C4.5） | Gini（CART） |
|--|------|------|
| 计算 | 慢（log运算） | 快 |
| 倾向 | 多值特征 | 平衡 |
| 常用场景 | 离散特征 | 连续特征 |

## CART 算法（sklearn 的实现）

CART = Classification And Regression Trees，每次**二叉分裂**。

对于连续特征：枚举所有分裂点，选 Gini 增益最大的。

对于分类特征：枚举所有二分子集。

### 停止条件

- 达到最大深度（`max_depth`）
- 节点样本数小于阈值（`min_samples_split`）
- Gini/熵改善小于阈值（`min_impurity_decrease`）
- 所有样本属于同一类

### 剪枝：解决过拟合

**预剪枝**（Early Stopping）：生长时限制深度/样本数。

**后剪枝**（Cost Complexity Pruning）：先长满，再从叶节点开始剪，选验证集误差最小的子树。

sklearn 的 `ccp_alpha` 参数控制后剪枝力度。

## Python 实现

```python
import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    """简化版 CART（Gini + 二叉分裂）"""
    
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def _gini(self, y):
        counts = Counter(y)
        n = len(y)
        return 1 - sum((c/n)**2 for c in counts.values())
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        n = len(y)
        parent_gini = self._gini(y)
        
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                right_mask = ~left_mask
                if left_mask.sum() < 1 or right_mask.sum() < 1:
                    continue
                
                gain = parent_gini - (
                    left_mask.sum()/n * self._gini(y[left_mask]) +
                    right_mask.sum()/n * self._gini(y[right_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thr
        
        return best_feature, best_threshold, best_gain
    
    def _build(self, X, y, depth=0):
        # 停止条件
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return {'leaf': True, 'label': Counter(y).most_common(1)[0][0]}
        
        feat, thr, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            return {'leaf': True, 'label': Counter(y).most_common(1)[0][0]}
        
        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask
        
        return {
            'leaf': False,
            'feature': feat,
            'threshold': thr,
            'left': self._build(X[left_mask], y[left_mask], depth+1),
            'right': self._build(X[right_mask], y[right_mask], depth+1),
        }
    
    def fit(self, X, y):
        self.tree_ = self._build(X, y)
        return self
    
    def _predict_one(self, x, node):
        if node['leaf']:
            return node['label']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])


# ---- 测试 ----
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target,
                                            test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_tr, y_tr)
preds = dt.predict(X_te)
print(f"手写 DT 准确率: {(preds == y_te).mean():.4f}")

# sklearn
from sklearn.tree import DecisionTreeClassifier as SkDT, export_text
sk_dt = SkDT(max_depth=4, criterion='gini', random_state=42)
sk_dt.fit(X_tr, y_tr)
print(f"sklearn DT 准确率: {sk_dt.score(X_te, y_te):.4f}")
print("\n决策树结构：")
print(export_text(sk_dt, feature_names=iris.feature_names))
```

## 可视化决策树

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(15, 8))
plot_tree(sk_dt,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Iris 决策树（max_depth=4）")
plt.tight_layout()

# 特征重要性
importances = sk_dt.feature_importances_
for name, imp in zip(iris.feature_names, importances):
    print(f"{name}: {imp:.4f}")
```

## 过拟合与剪枝实验

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

depths = range(1, 20)
train_scores, val_scores = [], []

for d in depths:
    dt = SkDT(max_depth=d, random_state=42)
    dt.fit(X, y)
    train_scores.append(dt.score(X, y))
    val_scores.append(cross_val_score(dt, X, y, cv=5).mean())

plt.figure(figsize=(8, 5))
plt.plot(depths, train_scores, 'b-o', label='训练准确率')
plt.plot(depths, val_scores, 'r-o', label='交叉验证准确率')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('决策树深度 vs 过拟合')
plt.legend()
plt.grid(True)
plt.tight_layout()
```

## 面试要点

**Q: 决策树为什么容易过拟合？如何防止？**

A: 不限制深度时，决策树可以完美记住每个训练样本（每个叶节点一个样本），泛化能力极差。解决：① 预剪枝（限 max_depth、min_samples_leaf 等）；② 后剪枝（ccp_alpha）；③ 集成方法（随机森林、Boosting）。

**Q: 信息增益偏向多值特征，C4.5 如何解决？**

A: C4.5 用**信息增益率** = 信息增益 / 分裂信息，对特征值多的特征进行惩罚，避免 ID3 偏向 ID 类特征。

**Q: 决策树如何处理缺失值？**

A: C4.5 对含缺失值的样本按各分支比例分配权重；CART（sklearn）使用替代分裂（surrogate splits）——找与主分裂最相关的备用分裂。
