---
layout: default
title: "D3 · 树模型：XGBoost/LightGBM 选股"
render_with_liquid: false
---

# D3 · 树模型：XGBoost/LightGBM 选股

> 梯度提升树是量化界最常用的 ML 模型。训练快，可解释，不需要标准化。

---

## 1. 为什么树模型在量化中流行

| 优势 | 说明 |
|------|------|
| 天然处理非线性 | 可以自动发现因子间的交互效应 |
| 对异常值鲁棒 | 不需要 Winsorize（树模型按分裂，不是线性） |
| 特征重要性 | 可解释性相对 DNN 更强 |
| 训练速度快 | LightGBM 可以快速迭代 |
| 无需标准化 | 不像线性模型需要特征同量纲 |

---

## 2. LightGBM 选股模型

```python
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class LGBMStockSelector:
    """
    LightGBM 多因子选股模型
    """
    
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'regression',  # 预测未来收益率
            'metric': 'mse',
            'num_leaves': 31,
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'feature_fraction': 0.8,    # 防过拟合
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,           # L1 正则
            'lambda_l2': 0.1,           # L2 正则
            'min_child_samples': 20,    # 叶节点最少样本数
            'verbose': -1
        }
        self.model = None
        self.feature_names = None
    
    def prepare_data(self, feature_df, return_df, lookforward=1):
        """
        准备训练数据
        feature_df: MultiIndex (date, stock) 的特征 DataFrame
        return_df: 未来 lookforward 期的收益率
        """
        X = feature_df.copy()
        y = return_df.shift(-lookforward)  # 未来收益（预测目标）
        
        # 对齐索引
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # 去除 NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        self.feature_names = X_train.columns.tolist()
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = None
        
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[val_data] if val_data else None,
            callbacks=callbacks if val_data else None
        )
        return self
    
    def predict(self, X_test):
        """预测未来收益率（作为选股打分）"""
        if self.model is None:
            raise ValueError("模型未训练")
        return pd.Series(
            self.model.predict(X_test),
            index=X_test.index
        )
    
    def get_feature_importance(self, importance_type='gain'):
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型未训练")
        importance = pd.Series(
            self.model.feature_importance(importance_type=importance_type),
            index=self.feature_names
        ).sort_values(ascending=False)
        return importance
    
    def calc_ic(self, y_pred, y_true):
        """计算预测的 Rank IC"""
        from scipy.stats import spearmanr
        corr, _ = spearmanr(y_pred, y_true)
        return corr
```

---

## 3. 交叉验证框架

```python
def lgbm_walkforward_cv(feature_df, return_df, 
                          train_months=24, test_months=3):
    """
    滚动样本外测试
    """
    dates = return_df.index.get_level_values('date').unique().sort_values()
    
    oos_results = []
    
    for i in range(train_months, len(dates) - test_months, test_months):
        train_dates = dates[i - train_months:i]
        test_dates = dates[i:i + test_months]
        
        X_train = feature_df.loc[feature_df.index.get_level_values('date').isin(train_dates)]
        y_train = return_df.loc[return_df.index.get_level_values('date').isin(train_dates)]
        
        X_test = feature_df.loc[feature_df.index.get_level_values('date').isin(test_dates)]
        y_test = return_df.loc[return_df.index.get_level_values('date').isin(test_dates)]
        
        # 训练
        model = LGBMStockSelector()
        model.train(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        ic = model.calc_ic(y_pred, y_test)
        oos_results.append({
            'train_end': train_dates[-1],
            'test_start': test_dates[0],
            'test_end': test_dates[-1],
            'oos_ic': ic
        })
    
    results_df = pd.DataFrame(oos_results)
    print(f"样本外平均 IC: {results_df['oos_ic'].mean():.4f}")
    print(f"样本外 ICIR: {results_df['oos_ic'].mean()/results_df['oos_ic'].std():.4f}")
    
    return results_df
```

---

## 4. 超参数调优

```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def objective(params):
    """Hyperopt 目标函数"""
    int_params = ['num_leaves', 'max_depth', 'min_child_samples', 'n_estimators']
    for p in int_params:
        if p in params:
            params[p] = int(params[p])
    
    # 交叉验证评估
    # ...（此处简化）
    
    return {'loss': -mean_ic, 'status': STATUS_OK}

search_space = {
    'num_leaves': hp.quniform('num_leaves', 20, 60, 5),
    'max_depth': hp.quniform('max_depth', 3, 8, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'lambda_l1': hp.loguniform('lambda_l1', np.log(1e-3), np.log(10)),
    'lambda_l2': hp.loguniform('lambda_l2', np.log(1e-3), np.log(10)),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0)
}
```

---

## 5. 模型监控：特征漂移检测

```python
def detect_feature_drift(X_train, X_test, threshold=0.1):
    """
    检测训练集和测试集的特征分布漂移
    使用 KS 检验
    """
    from scipy.stats import ks_2samp
    
    drift_results = {}
    for col in X_train.columns:
        stat, pvalue = ks_2samp(X_train[col].dropna(), X_test[col].dropna())
        drift_results[col] = {'ks_stat': stat, 'pvalue': pvalue}
    
    drift_df = pd.DataFrame(drift_results).T
    high_drift = drift_df[drift_df['ks_stat'] > threshold]
    
    if len(high_drift) > 0:
        print(f"警告：{len(high_drift)} 个特征发生显著漂移：")
        print(high_drift.sort_values('ks_stat', ascending=False))
    
    return drift_df
```

---

## 小结

| 维度 | 内容 |
|------|------|
| 推荐模型 | LightGBM（速度快，效果好）|
| 目标变量 | 未来 N 期截面收益率排名 |
| 关键参数 | num_leaves, lambda_l1/l2, feature_fraction |
| 验证方式 | Walk-Forward OOS，不能随机分割 |
| 评估指标 | OOS Rank IC / ICIR |
| 核心风险 | 过拟合（小数据集 + 多参数）|
