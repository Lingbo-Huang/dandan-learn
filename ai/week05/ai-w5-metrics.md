---
layout: default
title: "评估指标全家桶"
render_with_liquid: false
---

# 评估指标全家桶

## 混淆矩阵

$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix}$$

| 缩写 | 含义 |
|------|------|
| TP | 实际正，预测正 ✅ |
| TN | 实际负，预测负 ✅ |
| FP | 实际负，预测正 ❌（虚警） |
| FN | 实际正，预测负 ❌（漏报） |

## 核心指标推导

$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$$

$$\text{Precision} = \frac{TP}{TP+FP} \quad \text{（预测为正中真正是正的比例）}$$

$$\text{Recall（Sensitivity）} = \frac{TP}{TP+FN} \quad \text{（真正的正样本中被找到的比例）}$$

$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP+FP+FN}$$

$$\text{Specificity} = \frac{TN}{TN+FP} \quad \text{（真正的负样本中被找到的比例）}$$

$$\text{FPR} = \frac{FP}{FP+TN} = 1 - \text{Specificity}$$

## ROC 曲线与 AUC

ROC 曲线：纵轴 TPR（Recall），横轴 FPR，随分类阈值从高到低绘制。

$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)$$

**AUC 直觉**：随机选一正一负，AUC = 模型给正样本打分高于负样本的概率。

## PR 曲线

纵轴 Precision，横轴 Recall，绘制阈值变化曲线。

**PR-AUC vs ROC-AUC**：正负类极度不平衡时（正样本极少），PR 曲线更能反映模型质量（ROC-AUC 会虚高）。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay
)

# 不平衡数据集
X, y = make_classification(n_samples=1000, n_features=20,
                            weights=[0.9, 0.1],  # 90% 负类
                            random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            stratify=y, random_state=42)

models = {
    '逻辑回归': LogisticRegression(class_weight='balanced', max_iter=1000),
    '随机森林': RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                       random_state=42)
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2196F3', '#FF5722']

for (name, model), color in zip(models.items(), colors):
    model.fit(X_tr, y_tr)
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred = model.predict(X_te)
    
    # ROC
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=color, linewidth=2,
                 label=f"{name} (AUC={roc_auc:.3f})")
    
    # PR
    precision, recall, _ = precision_recall_curve(y_te, y_proba)
    ap = average_precision_score(y_te, y_proba)
    axes[1].plot(recall, precision, color=color, linewidth=2,
                 label=f"{name} (AP={ap:.3f})")
    
    print(f"\n{name}:")
    print(classification_report(y_te, y_pred, target_names=['负类','正类']))

# ROC 图设置
axes[0].plot([0,1],[0,1],'k--', linewidth=1)
axes[0].set_xlabel('FPR (1-Specificity)')
axes[0].set_ylabel('TPR (Recall)')
axes[0].set_title('ROC 曲线')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# PR 图设置
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('PR 曲线（不平衡数据集更有参考价值）')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# 混淆矩阵可视化
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, model) in zip(axes2, models.items()):
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['负类','正类'])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'混淆矩阵 - {name}')
plt.tight_layout()
```

## 回归评估指标

$$\text{MAE} = \frac{1}{N}\sum_i |y_i - \hat{y}_i|$$

$$\text{MSE} = \frac{1}{N}\sum_i (y_i - \hat{y}_i)^2$$

$$\text{RMSE} = \sqrt{\text{MSE}}$$

$$R^2 = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$$

- MAE：对异常值不敏感，单位与目标一致
- RMSE：对大误差惩罚更重，与梯度下降一致
- $R^2$：取值[0,1]，越接近1越好；可为负（模型比均值预测还差）

## 面试要点

**Q: 精确率和召回率哪个更重要？**

A: 取决于业务场景。**漏报代价高**（癌症筛查）→ 召回率优先；**虚警代价高**（垃圾邮件过滤）→ 精确率优先。可以用 $F_\beta$ 分数灵活权衡：$F_\beta = (1+\beta^2)\frac{PR}{\beta^2 P+R}$，$\beta>1$ 偏重召回，$\beta<1$ 偏重精确。

**Q: AUC=0.5 意味着什么？**

A: 模型等同于随机猜测（对正负样本的打分无区分能力）。AUC<0.5 意味着模型反向预测，将预测取反后 AUC=1-原AUC。
