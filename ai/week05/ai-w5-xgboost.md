---
layout: default
title: "梯度提升与 XGBoost"
render_with_liquid: false
---

# 梯度提升与 XGBoost

## GBDT：用梯度拟合残差

### 核心思想

AdaBoost 调整样本权重；GBDT 让下一棵树**拟合当前预测的残差**（梯度方向）。

### 前向分步算法

初始化：$F_0(x) = \arg\min_c \sum_i L(y_i, c)$

对 $m = 1, 2, \ldots, M$：

**Step 1**：计算负梯度（伪残差）：

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

**Step 2**：用决策树拟合 $\{(x_i, r_{im})\}$，得树 $h_m$

**Step 3**：线搜索最优步长：

$$\gamma_m = \arg\min_\gamma \sum_i L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$$

**Step 4**：更新：$F_m(x) = F_{m-1}(x) + \eta \cdot \gamma_m \cdot h_m(x)$（$\eta$ 是学习率）

### 不同损失函数对应的伪残差

| 任务 | 损失函数 | 负梯度（伪残差） |
|------|---------|----------------|
| 回归 | MSE | $y_i - F(x_i)$（真实残差） |
| 分类 | Log Loss | $y_i - \sigma(F(x_i))$ |
| 排序 | LambdaRank | 复杂梯度 |

## XGBoost：GBDT 的工程升级

### 目标函数

$$\mathcal{L}^{(t)} = \sum_i l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

**泰勒二阶展开**（更精确的梯度利用）：

$$\mathcal{L}^{(t)} \approx \sum_i \left[g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)\right] + \Omega(f_t)$$

其中：
- $g_i = \partial_{\hat{y}} l(y_i, \hat{y})$（一阶梯度）
- $h_i = \partial^2_{\hat{y}} l(y_i, \hat{y})$（二阶梯度/Hessian）

**正则化项**：

$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$

其中 $T$ 为叶节点数，$w_j$ 为叶节点权重。

### 最优叶节点权重

对叶节点 $j$，令 $\mathcal{J}_j$ 为该叶节点的样本集合：

$$w_j^* = -\frac{\sum_{i \in \mathcal{J}_j} g_i}{\sum_{i \in \mathcal{J}_j} h_i + \lambda}$$

### 分裂增益

$$\text{Gain} = \frac{1}{2}\left[\frac{(\sum_{i\in L}g_i)^2}{\sum_{i\in L}h_i+\lambda} + \frac{(\sum_{i\in R}g_i)^2}{\sum_{i\in R}h_i+\lambda} - \frac{(\sum_i g_i)^2}{\sum_i h_i+\lambda}\right] - \gamma$$

### XGBoost vs GBDT 的核心区别

| 特性 | GBDT | XGBoost |
|------|------|---------|
| 梯度阶数 | 一阶 | 二阶（更精确） |
| 正则化 | 无 | L1/L2 + 叶节点数惩罚 |
| 分裂算法 | 贪心 | 近似直方图 |
| 缺失值处理 | 无 | 自动学习最优方向 |
| 并行化 | 串行 | 特征并行（列并行） |
| 剪枝 | 预剪枝 | 后剪枝（gamma） |

## Python 实践

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb

cancer = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# ---- 基础用法 ----
xgb_clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,         # 行采样（类似 Bagging）
    colsample_bytree=0.8,  # 列采样（每棵树）
    reg_alpha=0.1,         # L1 正则
    reg_lambda=1.0,        # L2 正则
    min_child_weight=3,    # 叶节点最小 Hessian 和
    gamma=0.1,             # 分裂最小增益
    random_state=42,
    eval_metric='auc',
    use_label_encoder=False
)

xgb_clf.fit(X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_te, y_te)],
            verbose=False)

print(f"XGBoost 测试 AUC: {roc_auc_score(y_te, xgb_clf.predict_proba(X_te)[:,1]):.4f}")

# ---- 学习曲线 ----
results = xgb_clf.evals_result()
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)

plt.figure(figsize=(8, 4))
plt.plot(x_axis, results['validation_0']['auc'], label='训练集')
plt.plot(x_axis, results['validation_1']['auc'], label='测试集')
plt.xlabel('Boosting 轮数')
plt.ylabel('AUC')
plt.title('XGBoost 学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# ---- 特征重要性（多种方式）----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, imp_type in zip(axes, ['weight', 'gain', 'cover']):
    xgb.plot_importance(xgb_clf, ax=ax, importance_type=imp_type,
                        max_num_features=10, title=f'重要性类型: {imp_type}')
plt.tight_layout()

# ---- 早停（Early Stopping）----
xgb_es = xgb.XGBClassifier(
    n_estimators=1000,  # 设大，靠早停找最优
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric='auc',
    use_label_encoder=False
)
xgb_es.fit(X_tr, y_tr,
           eval_set=[(X_te, y_te)],
           early_stopping_rounds=20,
           verbose=False)
print(f"最优迭代次数: {xgb_es.best_iteration}")
print(f"最优测试 AUC: {xgb_es.best_score:.4f}")

# ---- LightGBM 对比（速度更快）----
import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=31,        # LGB 核心参数：叶子数（控制复杂度）
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)
lgb_clf.fit(X_tr, y_tr)
print(f"LightGBM 测试 AUC: {roc_auc_score(y_te, lgb_clf.predict_proba(X_te)[:,1]):.4f}")
```

## 超参数调优技巧

```python
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.5],
}

rs = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric='auc', use_label_encoder=False),
    param_dist, n_iter=30, cv=5, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
rs.fit(cancer.data, cancer.target)
print(f"最优参数: {rs.best_params_}")
print(f"最优 CV AUC: {rs.best_score_:.4f}")
```

## 面试要点

**Q: XGBoost 为什么快/准？**

A: 快：① 列并行处理；② 预排序+直方图近似分裂；③ Cache-aware 访问。准：① 二阶梯度信息更精确；② 内置正则化（防过拟合）；③ 内置缺失值处理；④ 列/行采样增加多样性。

**Q: LightGBM 和 XGBoost 的区别？**

A: LGB：① 按叶（leaf-wise）分裂 vs XGB 按层（level-wise）→ 更精确但易过拟合（用 num_leaves 控制）；② 直方图算法 → 速度更快内存更少；③ 类别特征原生支持；④ GOSS（只用大梯度样本）和 EFB（互斥特征绑定）进一步加速。

**Q: GBDT 如何防止过拟合？**

A: ① 降低学习率（同时增加树的数量）；② 减小 max_depth；③ subsample/colsample_bytree 列行采样；④ L1/L2 正则（reg_alpha/reg_lambda）；⑤ early stopping。
