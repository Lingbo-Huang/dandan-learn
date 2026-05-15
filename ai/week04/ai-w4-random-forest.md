---
layout: default
title: "随机森林"
render_with_liquid: false
---

# 随机森林

## 直觉：民主投票 > 独裁专家

一棵决策树容易过拟合。如果有 100 棵"见解稍有不同"的树，让它们投票决策，结果会更稳健。

随机森林 = Bagging（有放回采样）+ 特征随机（每次分裂只看部分特征）。

## Bagging 原理

**Bootstrap Aggregating**：

1. 从训练集（$N$ 个样本）有放回地抽取 $N$ 个样本 → **自助集** (bootstrap sample)
2. 在自助集上训练一棵决策树（通常长到最大）
3. 重复 $B$ 次，得到 $B$ 棵树
4. 预测时：分类 → 多数投票；回归 → 平均

**关键统计性质**：

平均有约 $1-(1-1/N)^N \approx 63.2\%$ 的样本被选中，约 $36.8\%$ 未被选中（**OOB 样本**，Out-Of-Bag），可用于免费验证！

### 方差减少分析

若 $B$ 棵独立树各有方差 $\sigma^2$，相关系数 $\rho$：

$$\text{Var}(\bar{f}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

当 $B \to \infty$：$\text{Var} \to \rho\sigma^2$。

→ 树越不相关，集成效果越好！这就是**特征随机**的意义。

## 特征随机

普通 Bagging：每次分裂考虑所有 $p$ 个特征。

随机森林：每次分裂随机选 $m$ 个特征（通常 $m = \sqrt{p}$ 分类，$m = p/3$ 回归）。

效果：降低树间相关性 → 集成后方差更小。

## Python 实现

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_features='sqrt',
                 max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.rng = np.random.RandomState(random_state)
    
    def fit(self, X, y):
        n, p = X.shape
        self.trees_ = []
        self.oob_predictions_ = np.zeros((n, len(np.unique(y))))
        self.oob_counts_ = np.zeros(n)
        
        for _ in range(self.n_estimators):
            # Bootstrap 采样
            idx = self.rng.choice(n, size=n, replace=True)
            oob_idx = np.setdiff1d(np.arange(n), idx)
            
            X_boot, y_boot = X[idx], y[idx]
            
            # 确定每次分裂用的特征数
            if self.max_features == 'sqrt':
                mf = int(np.sqrt(p))
            else:
                mf = self.max_features
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=mf,
                random_state=self.rng.randint(0, 10000)
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)
            
            # OOB 预测
            if len(oob_idx) > 0:
                oob_proba = tree.predict_proba(X[oob_idx])
                self.oob_predictions_[oob_idx] += oob_proba
                self.oob_counts_[oob_idx] += 1
        
        # 计算 OOB 准确率
        valid = self.oob_counts_ > 0
        oob_preds = self.oob_predictions_[valid].argmax(axis=1)
        self.oob_score_ = (oob_preds == y[valid]).mean()
        
        self.classes_ = np.unique(y)
        return self
    
    def predict_proba(self, X):
        probas = np.zeros((len(X), len(self.classes_)))
        for tree in self.trees_:
            probas += tree.predict_proba(X)
        return probas / self.n_estimators
    
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---- 测试 ----
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf.fit(X_tr, y_tr)
print(f"手写 RF 测试准确率: {(rf.predict(X_te) == y_te).mean():.4f}")
print(f"OOB 准确率估计:     {rf.oob_score_:.4f}")

# sklearn 对比
from sklearn.ensemble import RandomForestClassifier as SkRF
sk_rf = SkRF(n_estimators=100, oob_score=True, random_state=42)
sk_rf.fit(X_tr, y_tr)
print(f"sklearn RF 测试准确率: {sk_rf.score(X_te, y_te):.4f}")
print(f"sklearn OOB 准确率:    {sk_rf.oob_score_:.4f}")
```

## 特征重要性分析

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(cancer.data, cancer.target)

# 方法1：基尼重要性（MDI）
importances = pd.Series(rf.feature_importances_, index=cancer.feature_names)
importances.sort_values(ascending=True).tail(15).plot(kind='barh', figsize=(8,6))
plt.title("特征重要性（Gini MDI）")
plt.tight_layout()

# 方法2：置换重要性（更可靠）
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(cancer.data, cancer.target,
                                            test_size=0.2, random_state=42)
perm_imp = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    'feature': cancer.feature_names,
    'importance_mean': perm_imp.importances_mean,
    'importance_std': perm_imp.importances_std
}).sort_values('importance_mean', ascending=False)
print(perm_df.head(10))
```

## 超参数调优

```python
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5],
}

rf = RandomForestClassifier(random_state=42, oob_score=True)
rs = RandomizedSearchCV(rf, param_dist, n_iter=30, cv=5,
                        scoring='accuracy', random_state=42, n_jobs=-1)
rs.fit(cancer.data, cancer.target)
print(f"最优参数: {rs.best_params_}")
print(f"最优 CV 得分: {rs.best_score_:.4f}")
```

## n_estimators 的影响

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

n_trees = [1, 5, 10, 20, 50, 100, 200, 500]
scores = []
for n in n_trees:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    score = cross_val_score(rf, X, y, cv=5).mean()
    scores.append(score)

plt.figure(figsize=(8, 4))
plt.plot(n_trees, scores, 'b-o')
plt.xscale('log')
plt.xlabel('n_estimators')
plt.ylabel('CV Accuracy')
plt.title('随机森林：树的数量 vs 性能')
plt.grid(True)
plt.tight_layout()
# 结论：~100棵后收益递减，继续增加只是浪费计算
```

## 面试要点

**Q: 随机森林为什么不会像决策树那样过拟合？**

A: 两个机制共同作用：① Bootstrap 采样引入数据扰动，每棵树学到不同的模式；② 特征随机让树更"多样化"，降低相关性。集成后，偏差基本不变，但方差大幅降低——偏差-方差权衡的典型应用。

**Q: OOB 误差为什么可以近似替代交叉验证？**

A: 每棵树约 36.8% 的样本未参与训练（OOB），用这些样本评估这棵树，汇总所有树的 OOB 预测即可得到泛化误差估计，效果接近 5-fold CV，但只需训练一次。

**Q: 特征重要性 MDI 有什么缺陷？**

A: MDI 对高基数特征（取值多的特征如 ID）有偏，会高估其重要性。更可靠的方式是置换重要性（Permutation Importance）——把特征随机打乱，看准确率下降多少。
