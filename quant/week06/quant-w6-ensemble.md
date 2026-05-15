---
layout: default
title: "D5 · 模型集成与稳定性"
render_with_liquid: false
---

# D5 · 模型集成与稳定性

> 单一模型的预测不稳定。集成多个模型是提升稳定性的核心工程手段。

---

## 1. 为什么需要集成

- 金融数据噪声大，单个模型方差高
- 不同模型在不同市场环境下各有优势
- 集成平均错误率低于单模型（偏差-方差分解）

---

## 2. 主要集成方法

### 2.1 Bagging（随机森林思想）

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import lightgbm as lgb

class BaggingFactorModel:
    """
    Bagging 集成因子模型
    训练 n 个独立模型，对不同子样本/特征子集训练，平均预测
    """
    
    def __init__(self, n_models=10, feature_fraction=0.8, 
                 sample_fraction=0.8):
        self.n_models = n_models
        self.feature_fraction = feature_fraction
        self.sample_fraction = sample_fraction
        self.models = []
        self.feature_subsets = []
    
    def fit(self, X, y):
        self.models = []
        self.feature_subsets = []
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        for i in range(self.n_models):
            # 随机抽取特征子集
            n_feat = int(n_features * self.feature_fraction)
            feat_idx = np.random.choice(n_features, n_feat, replace=False)
            self.feature_subsets.append(feat_idx)
            
            # 随机抽取样本子集
            n_samp = int(n_samples * self.sample_fraction)
            samp_idx = np.random.choice(n_samples, n_samp, replace=True)
            
            X_sub = X.iloc[samp_idx, feat_idx]
            y_sub = y.iloc[samp_idx]
            
            # 训练单个模型
            model = lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                max_depth=5,
                verbose=-1
            )
            model.fit(X_sub, y_sub)
            self.models.append(model)
        
        return self
    
    def predict(self, X):
        predictions = []
        for model, feat_idx in zip(self.models, self.feature_subsets):
            pred = model.predict(X.iloc[:, feat_idx])
            predictions.append(pred)
        # 平均预测
        return np.mean(predictions, axis=0)
```

### 2.2 不同算法集成

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from scipy.stats import rankdata

class HeterogeneousEnsemble:
    """
    异质集成：不同算法的加权平均
    """
    
    def __init__(self, weights=None):
        self.models = {
            'ridge': Ridge(alpha=10.0),
            'lgbm': lgb.LGBMRegressor(n_estimators=200, num_leaves=31, verbose=-1),
            'rf': RandomForestRegressor(n_estimators=100, max_depth=5)
        }
        self.weights = weights or {'ridge': 0.3, 'lgbm': 0.5, 'rf': 0.2}
        self.is_fitted = False
    
    def fit(self, X, y):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.models.items():
            if name == 'ridge':
                model.fit(X_scaled, y)
            else:
                model.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'ridge':
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            # 转为截面排名（统一量纲）
            predictions[name] = rankdata(pred) / len(pred)
        
        # 加权平均
        result = sum(self.weights[k] * v for k, v in predictions.items())
        return result
```

### 2.3 时间集成（滚动平均）

```python
def time_ensemble_signal(factor_df, lookback_list=[1, 3, 6]):
    """
    时间集成：对多个滞后期的因子信号加权平均
    减少单期信号噪声
    """
    weighted_signal = pd.DataFrame(0, 
                                    index=factor_df.index,
                                    columns=factor_df.columns)
    total_weight = 0
    
    for lag in lookback_list:
        # 权重随滞后增大而降低（衰减加权）
        weight = 1.0 / lag
        weighted_signal += weight * factor_df.shift(lag)
        total_weight += weight
    
    return weighted_signal / total_weight
```

---

## 3. 模型稳定性评估

```python
def model_stability_test(X, y, n_seeds=10, train_ratio=0.7):
    """
    通过多个随机种子评估模型稳定性
    输出：预测的平均 IC 和 IC 标准差
    """
    from scipy.stats import spearmanr
    
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    ics = []
    
    for seed in range(n_seeds):
        model = lgb.LGBMRegressor(
            n_estimators=200,
            num_leaves=31,
            max_depth=5,
            random_state=seed,
            verbose=-1
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        ic, _ = spearmanr(pred, y_test)
        ics.append(ic)
    
    ics = pd.Series(ics)
    print(f"IC 均值: {ics.mean():.4f}")
    print(f"IC 标准差: {ics.std():.4f}")
    print(f"IC 变异系数: {ics.std()/abs(ics.mean()):.2%}")
    
    if ics.std() / abs(ics.mean()) > 0.3:
        print("⚠️ 模型稳定性较差，建议集成或换更简单的模型")
    
    return ics
```

---

## 4. 因子衰减与再训练策略

```python
def adaptive_retraining(model_class, X, y, 
                          init_train_months=24,
                          retrain_freq_months=3,
                          ic_decay_threshold=0.5):
    """
    自适应再训练：当样本外 IC 衰减到历史最高的 50% 时触发重训
    """
    from scipy.stats import spearmanr
    
    dates = X.index.get_level_values('date').unique().sort_values()
    
    model = None
    best_ic = -np.inf
    results = []
    
    for i, date in enumerate(dates[init_train_months:], start=init_train_months):
        train_dates = dates[max(0, i-init_train_months):i]
        
        # 每隔一段时间或 IC 衰减时重训
        need_retrain = (model is None or 
                        i % retrain_freq_months == 0)
        
        if need_retrain:
            X_train = X[X.index.get_level_values('date').isin(train_dates)]
            y_train = y[y.index.get_level_values('date').isin(train_dates)]
            model = model_class().fit(X_train, y_train)
        
        # 当日预测
        X_today = X[X.index.get_level_values('date') == date]
        y_today = y[y.index.get_level_values('date') == date]
        
        if len(X_today) > 0 and len(y_today) > 0:
            pred = model.predict(X_today)
            ic, _ = spearmanr(pred, y_today)
            results.append({'date': date, 'ic': ic})
    
    return pd.DataFrame(results).set_index('date')
```

---

## 小结

| 方法 | 核心思想 | 效果 |
|------|---------|------|
| Bagging | 多子样本/特征训练取平均 | 降方差 |
| 异质集成 | 不同算法加权 | 捕获不同非线性 |
| 时间集成 | 多期信号平均 | 减少噪声 |
| 自适应再训练 | IC 衰减触发重训 | 适应市场变化 |
