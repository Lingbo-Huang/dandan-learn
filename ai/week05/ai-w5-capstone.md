---
layout: default
title: "Week 5 综合实战：Kaggle 竞赛流程模拟"
render_with_liquid: false
---

# Week 5 综合实战：Kaggle 竞赛流程模拟

## 任务

模拟 Kaggle Titanic 生存预测竞赛，走完完整 ML 竞赛流程。

## 完整代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ===== 数据生成（模拟 Titanic）=====
np.random.seed(42)
n = 891

# 模拟特征
pclass = np.random.choice([1,2,3], n, p=[0.24,0.21,0.55])
sex = np.random.choice([0,1], n, p=[0.65,0.35])  # 0=male, 1=female
age = np.where(np.random.rand(n) < 0.2, np.nan,
               np.random.normal(29, 14, n).clip(1, 80))
fare = np.random.exponential(32, n).clip(0, 512)
embarked = np.random.choice(['S','C','Q'], n, p=[0.72,0.19,0.09])
has_cabin = (np.random.rand(n) < 0.23).astype(int)
sibsp = np.random.choice([0,1,2,3,4], n, p=[0.68,0.23,0.06,0.02,0.01])
parch = np.random.choice([0,1,2,3], n, p=[0.76,0.13,0.09,0.02])

# 生存概率（模拟真实规律）
survive_prob = (
    0.4
    + 0.3 * sex                          # 女性更高
    - 0.15 * (pclass == 3)               # 三等舱低
    + 0.1 * has_cabin                    # 有舱位高
    + 0.05 * (age < 16) * 0              # 儿童优先（简化）
    + np.random.normal(0, 0.1, n)
).clip(0.05, 0.95)

survived = (np.random.rand(n) < survive_prob).astype(int)

df = pd.DataFrame({
    'Pclass': pclass, 'Sex': sex, 'Age': age,
    'Fare': fare, 'Embarked': embarked,
    'HasCabin': has_cabin, 'SibSp': sibsp, 'Parch': parch,
    'Survived': survived
})

print("数据集形状:", df.shape)
print("生存率:", survived.mean():.3f)
print("\n缺失值：")
print(df.isnull().sum())

# ===== 特征工程 =====
def feature_engineering(df):
    df = df.copy()
    
    # 1. 填充缺失值
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    
    # 2. 类别编码
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    
    # 3. 新特征
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100],
                           labels=[0,1,2,3,4]).astype(int)
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3]).astype(int)
    df['Pclass_Sex'] = df['Pclass'] * 10 + df['Sex']  # 交叉特征
    
    return df

df_feat = feature_engineering(df)
feature_cols = [c for c in df_feat.columns if c != 'Survived']
X = df_feat[feature_cols].values
y = df_feat['Survived'].values

print(f"\n特征工程后特征数: {X.shape[1]}")

# ===== 模型定义 =====
models = {
    'LR': LogisticRegression(max_iter=1000, C=1.0),
    'RF': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    'GBM': GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                       learning_rate=0.05, random_state=42),
    'XGB': xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, random_state=42,
                               eval_metric='logloss', use_label_encoder=False),
}

# ===== 交叉验证 + OOF 预测（用于 Stacking）=====
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), len(models)))
cv_results = {}

for idx, (name, model) in enumerate(models.items()):
    oof = np.zeros(len(X))
    fold_scores = []
    
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        # LR 需要标准化
        if name == 'LR':
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)
        
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = prob
        fold_scores.append(roc_auc_score(y_va, prob))
    
    oof_preds[:, idx] = oof
    cv_results[name] = {
        'mean': np.mean(fold_scores),
        'std': np.std(fold_scores),
        'oof_auc': roc_auc_score(y, oof)
    }
    print(f"{name}: CV AUC = {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f} | OOF AUC = {roc_auc_score(y, oof):.4f}")

# ===== Stacking 元学习器 =====
meta_lr = LogisticRegression(C=1.0, max_iter=1000)
meta_scores = cross_val_score(meta_lr, oof_preds, y, cv=cv, scoring='roc_auc')
print(f"\nStacking 元学习器: {meta_scores.mean():.4f} ± {meta_scores.std():.4f}")

# ===== 等权集成 =====
ensemble_pred = oof_preds.mean(axis=1)
print(f"等权集成 OOF AUC: {roc_auc_score(y, ensemble_pred):.4f}")

# ===== 结果可视化 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# CV 结果对比
names = list(cv_results.keys()) + ['Stack', 'Ensemble']
means = [v['mean'] for v in cv_results.values()] + [meta_scores.mean(), roc_auc_score(y, ensemble_pred)]
stds = [v['std'] for v in cv_results.values()] + [meta_scores.std(), 0]

colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#607D8B']
axes[0].bar(range(len(names)), means, yerr=stds, color=colors[:len(names)], capsize=5)
axes[0].set_xticks(range(len(names)))
axes[0].set_xticklabels(names)
axes[0].set_ylabel('AUC')
axes[0].set_title('各模型 CV AUC 对比')
axes[0].set_ylim(0.7, 1.0)
for i, (m, s) in enumerate(zip(means, stds)):
    axes[0].text(i, m + 0.005, f'{m:.3f}', ha='center', fontsize=9)

# OOF 预测相关性（Stacking效果预测）
corr = np.corrcoef(oof_preds.T)
im = axes[1].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_xticks(range(4)); axes[1].set_yticks(range(4))
axes[1].set_xticklabels(list(models.keys()))
axes[1].set_yticklabels(list(models.keys()))
for i in range(4):
    for j in range(4):
        axes[1].text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=10)
plt.colorbar(im, ax=axes[1])
axes[1].set_title('OOF 预测相关性\n（相关性低 → 集成收益大）')

plt.suptitle('Week 5 · Kaggle 竞赛流程模拟', fontsize=14, fontweight='bold')
plt.tight_layout()

print("""
===== 竞赛关键经验 =====
1. 特征工程通常比模型调参更重要（60% 的提升来自好特征）
2. OOF 预测相关性低的模型集成收益更大
3. 早停 + 贝叶斯超参优化 > 网格搜索
4. 最后提交：集成 > 单一最优模型
5. 公榜/私榜差距大时注意过拟合 Public LB
""")
```
