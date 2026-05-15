---
layout: default
title: "模型调优全流程"
render_with_liquid: false
---

# 模型调优全流程

## 调优框架总览

```
数据质量 → 特征工程 → 模型选择 → 超参调优 → 集成 → 部署
```

## 特征工程速查

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                    LabelEncoder, OneHotEncoder,
                                    PolynomialFeatures)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (SelectKBest, f_classif, 
                                        mutual_info_classif, RFE)

# ---- 缺失值处理 ----
imputer_mean = SimpleImputer(strategy='mean')     # 均值填充
imputer_median = SimpleImputer(strategy='median') # 中位数（抗异常值）
imputer_knn = KNNImputer(n_neighbors=5)           # KNN（利用相似样本）

# ---- 类别编码 ----
# 有序类别：OrdinalEncoder / LabelEncoder
# 无序类别：OneHotEncoder
# 高基数类别：Target Encoding（注意数据泄露！要在CV内做）

# ---- 特征选择 ----
# 方法1：统计检验
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_names = np.array(cancer.feature_names)[selector.get_support()]
print("互信息选择的Top10特征:", selected_names)

# 方法2：基于模型（RFE）
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=10)
rfe.fit(X, y)
print("RFE选择的Top10特征:", cancer.feature_names[rfe.support_])

# 方法3：L1正则化（自动稀疏）
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

l1_selector = SelectFromModel(LogisticRegression(C=0.01, penalty='l1',
                                                   solver='liblinear', max_iter=1000))
l1_selector.fit(X, y)
print("L1选择的特征数:", l1_selector.get_support().sum())

# ---- 多项式特征 ----
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X[:, :5])  # 只对前5个特征做交叉
print(f"原始特征数: 5, 交叉后: {X_poly.shape[1]}")
```

## 超参数搜索策略

```python
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, 
                                      cross_val_score)
import xgboost as xgb

# ---- Grid Search（小参数空间）----
param_grid = {'max_depth': [3, 4, 5], 'n_estimators': [100, 200]}
gs = GridSearchCV(xgb.XGBClassifier(random_state=42, eval_metric='logloss',
                                     use_label_encoder=False),
                  param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X, y)

# ---- Optuna（贝叶斯优化，推荐）----
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        model = xgb.XGBClassifier(**params, random_state=42,
                                   eval_metric='logloss', use_label_encoder=False)
        score = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    print(f"Optuna 最优 AUC: {study.best_value:.4f}")
    print(f"最优参数: {study.best_params}")
except ImportError:
    print("安装 optuna: pip install optuna")
```

## Stacking 集成

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# 第一层：多样化的基模型
base_estimators = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('svm', SVC(kernel='rbf', probability=True)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('nb', GaussianNB()),
]

# 第二层：元学习器
stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(),
    cv=5,           # 用交叉验证生成元特征（防数据泄露）
    passthrough=False,  # 不把原始特征传给元学习器
    n_jobs=-1
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

stack_pipe = Pipeline([('scaler', StandardScaler()), ('stack', stack)])
cv_score = cross_val_score(stack_pipe, X, y, cv=5, scoring='roc_auc')
print(f"Stacking CV AUC: {cv_score.mean():.4f} ± {cv_score.std():.4f}")
```

## Pipeline + 完整调优流程

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# 完整 Pipeline：预处理 → 降维 → 分类
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', SVC(kernel='rbf', probability=True))
])

param_grid = {
    'pca__n_components': [10, 15, 20, None],  # 不同降维维度
    'clf__C': [0.1, 1, 10],
    'clf__gamma': ['scale', 'auto'],
}

gs = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X, y)
print(f"Pipeline Grid Search 最优 AUC: {gs.best_score_:.4f}")
print(f"最优参数: {gs.best_params_}")
```

## 面试要点

**Q: 如何避免超参数搜索中的数据泄露？**

A: 必须在 CV 内部做所有预处理，不能先 fit 预处理器再搜索。使用 Pipeline 可以自动保证这一点——GridSearchCV + Pipeline 每个 fold 只用训练子集 fit 预处理器，验证子集只用 transform。

**Q: 贝叶斯优化（Optuna）比随机搜索好在哪？**

A: 随机搜索对每次试验是独立的；贝叶斯优化用**已有试验结果**（代理模型）预测哪个区域的参数更有希望，智能地探索参数空间，通常需要更少次试验达到相同效果。适合每次试验代价高的场景。
