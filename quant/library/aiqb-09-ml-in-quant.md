---
layout: default
title: "AIQuantBook L09 · 监督学习在量化中的应用"
source: "https://github.com/waylandzhang/ai-quant-book"
---

# L09 · 监督学习在量化中的应用

> **来源**：[AI Quant Book](https://github.com/waylandzhang/ai-quant-book) · waylandzhang

---

## 核心观点

> 机器学习在量化中是工具，不是魔法。
> 
> ML 模型能发现非线性关系、处理高维特征，但无法解决数据不足、信号弱、市场非平稳的根本问题。

---

## 量化中 ML 的正确打开方式

```
❌ 错误思路：
   "我用深度学习预测股价，准确率80%，一定能赚钱"

✅ 正确思路：
   "我已经有了一些有经济逻辑支撑的因子，
    ML 帮我找到因子之间的非线性组合，
    提升 IC 和稳定性"
```

---

## 特征工程：量化 ML 的核心

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

class QuantFeatureEngineer:
    """量化特征工程"""
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        # ====== 动量类特征 ======
        for n in [5, 10, 20, 60, 120]:
            # 简单动量
            features[f'mom_{n}d'] = df['close'].pct_change(n)
            # 风险调整动量
            vol = df['close'].pct_change().rolling(n).std()
            features[f'mom_adj_{n}d'] = features[f'mom_{n}d'] / (vol + 1e-8)
        
        # ====== 均线类特征 ======
        for n in [5, 10, 20, 60]:
            ma = df['close'].rolling(n).mean()
            features[f'ma_dev_{n}'] = (df['close'] / ma - 1)
        
        # ====== 波动率类特征 ======
        ret = df['close'].pct_change()
        for n in [5, 20, 60]:
            features[f'vol_{n}d'] = ret.rolling(n).std() * np.sqrt(252)
        
        # 已实现波动率（分钟级可用）
        features['vol_ratio'] = features['vol_5d'] / (features['vol_60d'] + 1e-8)
        
        # ====== 成交量类特征 ======
        if 'volume' in df.columns:
            for n in [5, 20]:
                features[f'vol_ratio_{n}'] = df['volume'] / (df['volume'].rolling(n).mean() + 1)
            
            # 量价背离
            features['price_vol_corr_20'] = df['close'].rolling(20).corr(df['volume'])
        
        # ====== 技术指标 ======
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        features['rsi_14'] = 100 - (100 / (1 + gain/(loss+1e-8)))
        
        # 布林带位置
        ma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        features['bb_position'] = (df['close'] - ma20) / (std20 * 2 + 1e-8)
        
        return features
    
    def cross_section_normalize(self, features: pd.DataFrame) -> pd.DataFrame:
        """截面标准化（每个截面日期内标准化）"""
        # 对于因子研究，截面标准化比时序标准化更合适
        return features.apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-8), 
            axis=1  # 按行（日期）标准化
        )
    
    def remove_extreme_values(self, features: pd.DataFrame, n_sigma: float = 3) -> pd.DataFrame:
        """去极值（按截面）"""
        def winsorize(x):
            median = x.median()
            mad = (x - median).abs().median()
            return x.clip(median - n_sigma*1.4826*mad, 
                          median + n_sigma*1.4826*mad)
        return features.apply(winsorize, axis=1)
```

---

## 正确的时序 CV

```python
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

def time_series_cv_lgb(X: pd.DataFrame, y: pd.Series,
                        n_splits: int = 5,
                        gap: int = 20) -> dict:
    """
    带 gap 的时序交叉验证
    gap: 训练集和验证集之间的间隔天数（避免泄露）
    
    时间轴示意：
    |--Train1--|gap|--Val1--|
         |--Train2--|gap|--Val2--|
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=4,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        val_pred = model.predict(X_val)
        
        # 用 IC 评估（比 MSE 更符合量化的关注点）
        from scipy.stats import spearmanr
        ic, _ = spearmanr(y_val, val_pred)
        
        fold_results.append({'fold': fold, 'ic': ic, 'n_val': len(y_val)})
        print(f"Fold {fold}: IC={ic:.4f}, Val样本数={len(y_val)}")
    
    mean_ic = np.mean([r['ic'] for r in fold_results])
    std_ic = np.std([r['ic'] for r in fold_results])
    print(f"\n平均IC: {mean_ic:.4f} ± {std_ic:.4f}")
    
    return {'fold_results': fold_results, 'mean_ic': mean_ic, 'std_ic': std_ic}
```

---

## 模型 vs 传统因子：直接对比

```python
from scipy.stats import spearmanr
import pandas as pd

def compare_ml_vs_traditional(test_data: pd.DataFrame):
    """
    对比机器学习模型和传统单因子的预测能力
    """
    results = {}
    target = test_data['next_month_return']
    
    # 1. 传统因子 IC
    for factor in ['pe_inv', 'momentum_6m', 'roe', 'volume_ratio']:
        if factor in test_data.columns:
            ic, _ = spearmanr(test_data[factor], target, nan_policy='omit')
            results[f'Factor_{factor}'] = ic
    
    # 2. ML 模型预测 IC
    ml_pred = test_data['ml_prediction']  # 假设已预测
    ic_ml, _ = spearmanr(ml_pred, target, nan_policy='omit')
    results['ML_LightGBM'] = ic_ml
    
    result_df = pd.DataFrame.from_dict(results, orient='index', columns=['IC'])
    result_df = result_df.sort_values('IC', ascending=False)
    
    print("因子/模型 IC 对比：")
    print(result_df.to_string())
    print(f"\nML 相对最优单因子提升: {(ic_ml - result_df.iloc[1]['IC']):.4f}")
    
    return result_df
```

---

## 关键认识

1. **ML 不是替代因子研究的**：ML 是在好因子基础上做非线性组合
2. **特征工程 > 模型选择**：好的因子特征比花哨的模型更重要  
3. **时序 CV 是非谈判性要求**：用随机CV测出来的结果没有意义
4. **IC 比 MSE 更重要**：量化关心的是排序，不是绝对误差

---

## 延伸阅读

- [AI Quant Book](https://github.com/waylandzhang/ai-quant-book)
- Marcos Lopez de Prado - "Machine Learning for Asset Managers"
- Stefan Jansen - "Machine Learning for Algorithmic Trading"
