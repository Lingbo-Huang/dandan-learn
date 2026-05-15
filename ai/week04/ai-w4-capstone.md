---
layout: default
title: "Week 4 综合实战：五大分类算法对决"
render_with_liquid: false
---

# Week 4 综合实战：五大分类算法对决

## 任务目标

在**乳腺癌数据集**上对比本周五大算法，完整走完"数据探索 → 预处理 → 训练 → 评估 → 分析"全流程，培养选模型的直觉。

## 数据集简介

- **任务**：二分类（恶性/良性肿瘤）
- **样本数**：569
- **特征数**：30 个（细胞核特征：半径、纹理、周长等）
- **类别比**：212 恶性 / 357 良性（轻微不平衡）

## 完整代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, roc_curve, classification_report)
from sklearn.pipeline import Pipeline

# ===== 1. 数据加载与探索 =====
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y

print("数据集形状:", X.shape)
print("类别分布:", pd.Series(y).value_counts().to_dict())
print("\n特征统计：")
print(df.describe().round(2))

# 特征相关性热力图
corr = df.drop('target', axis=1).corr()
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(cancer.feature_names)))
ax.set_yticks(range(len(cancer.feature_names)))
ax.set_xticklabels(cancer.feature_names, rotation=90, fontsize=6)
ax.set_yticklabels(cancer.feature_names, fontsize=6)
ax.set_title("特征相关性矩阵")
plt.tight_layout()

# ===== 2. 预处理 =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== 3. 定义五大模型 Pipeline =====
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    '朴素贝叶斯': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GaussianNB())
    ]),
    '逻辑回归': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ]),
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
    ]),
    '决策树': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(max_depth=5, random_state=42))
    ]),
    '随机森林': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
}

# ===== 4. 训练与评估 =====
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    # 测试集评估
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'test_acc': accuracy_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
        'test_auc': roc_auc_score(y_test, y_proba),
        'y_pred': y_pred,
        'y_proba': y_proba,
    }
    
    print(f"\n{'='*40}")
    print(f"模型: {name}")
    print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"测试集准确率: {results[name]['test_acc']:.4f}")
    print(f"测试集 F1: {results[name]['test_f1']:.4f}")
    print(f"测试集 AUC: {results[name]['test_auc']:.4f}")

# ===== 5. 汇总对比表 =====
summary = pd.DataFrame({
    name: {
        'CV AUC': f"{v['cv_auc_mean']:.4f}±{v['cv_auc_std']:.4f}",
        '测试准确率': f"{v['test_acc']:.4f}",
        '测试 F1': f"{v['test_f1']:.4f}",
        '测试 AUC': f"{v['test_auc']:.4f}",
    }
    for name, v in results.items()
}).T
print("\n\n===== 汇总对比 =====")
print(summary.to_string())

# ===== 6. 可视化 =====
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, figure=fig)

# ROC 曲线
ax_roc = fig.add_subplot(gs[0, :2])
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
for (name, v), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, v['y_proba'])
    ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={v['test_auc']:.3f})")
ax_roc.plot([0,1],[0,1], 'k--', linewidth=1)
ax_roc.set_xlabel('FPR')
ax_roc.set_ylabel('TPR')
ax_roc.set_title('ROC 曲线对比')
ax_roc.legend(loc='lower right')
ax_roc.grid(True, alpha=0.3)

# 准确率对比柱状图
ax_bar = fig.add_subplot(gs[0, 2])
names = list(results.keys())
accs = [results[n]['test_acc'] for n in names]
bars = ax_bar.bar(range(len(names)), accs, color=colors)
ax_bar.set_xticks(range(len(names)))
ax_bar.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax_bar.set_ylabel('准确率')
ax_bar.set_title('测试集准确率')
ax_bar.set_ylim(0.85, 1.0)
for bar, acc in zip(bars, accs):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

# 混淆矩阵（以最好的模型为例）
best_name = max(results, key=lambda k: results[k]['test_auc'])
ax_cm = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
im = ax_cm.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, cm[i, j], ha='center', va='center', fontsize=14, fontweight='bold')
ax_cm.set_xlabel('预测')
ax_cm.set_ylabel('真实')
ax_cm.set_title(f'混淆矩阵 - {best_name}')
ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
ax_cm.set_xticklabels(['恶性','良性']); ax_cm.set_yticklabels(['恶性','良性'])

# CV AUC 对比（带误差棒）
ax_cv = fig.add_subplot(gs[1, 1])
means = [results[n]['cv_auc_mean'] for n in names]
stds = [results[n]['cv_auc_std'] for n in names]
ax_cv.bar(range(len(names)), means, yerr=stds, color=colors, capsize=5)
ax_cv.set_xticks(range(len(names)))
ax_cv.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax_cv.set_ylabel('CV AUC')
ax_cv.set_title('5-Fold CV AUC（含标准差）')
ax_cv.set_ylim(0.88, 1.0)

# 随机森林特征重要性
ax_fi = fig.add_subplot(gs[1, 2])
rf_model = models['随机森林']
importances = rf_model.named_steps['clf'].feature_importances_
top_idx = np.argsort(importances)[-10:]
ax_fi.barh(range(10), importances[top_idx], color='#9C27B0')
ax_fi.set_yticks(range(10))
ax_fi.set_yticklabels([cancer.feature_names[i] for i in top_idx], fontsize=8)
ax_fi.set_xlabel('特征重要性')
ax_fi.set_title('随机森林 Top10 特征')

plt.suptitle('Week 4 · 五大分类算法对决 — 乳腺癌数据集', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('week04_classification_battle.png', dpi=150, bbox_inches='tight')
print("\n图表已保存")

# ===== 7. 学习总结 =====
print("""
===== 本周总结 =====

算法选择指南：
┌──────────────┬─────────────────────────────────────────┐
│ 算法         │ 最适合的场景                              │
├──────────────┼─────────────────────────────────────────┤
│ 朴素贝叶斯   │ 文本分类、实时预测、样本极少              │
│ 逻辑回归     │ 需要概率输出、特征可解释、大数据集        │
│ SVM          │ 小样本、高维、非线性问题（RBF核）         │
│ 决策树       │ 需要人类可读规则、特征交互重要            │
│ 随机森林     │ 通用首选、自动特征选择、鲁棒性强          │
└──────────────┴─────────────────────────────────────────┘

本数据集结论：
- 随机森林/SVM 效果最佳（AUC > 0.99）
- 逻辑回归接近最优，可解释性最好
- 决策树单棵过拟合风险高，集成后大幅提升
- 朴素贝叶斯在特征相关性强时偏差较大
""")
```

## 面试模拟问答

**Q: 这五个模型你会优先选哪个？**

A: 看场景。需要可解释性 → 逻辑回归或决策树；通用高性能 → 随机森林（首选）；小样本高维 → SVM；文本/实时 → 朴素贝叶斯。实践中先跑随机森林作为 baseline，再根据需求调整。

**Q: 乳腺癌数据集轻微不平衡，你怎么处理？**

A: 此处不平衡程度较轻（约 6:4），影响不大。若严重不平衡：① 评估指标用 F1/AUC 而非准确率；② 过采样（SMOTE）或欠采样；③ class_weight='balanced' 参数；④ 调整分类阈值。

**Q: 为什么用 AUC 而不是准确率作为主要指标？**

A: 准确率在不平衡数据集上会误导（全预测多数类也能高准确率）。AUC 衡量模型区分正负类的能力，不依赖分类阈值，更全面。此外 F1 也是好指标，综合精确率和召回率。
