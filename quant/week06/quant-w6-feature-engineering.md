---
layout: default
title: "D2 · 特征工程：时序特征与截面特征"
render_with_liquid: false
---

# D2 · 特征工程：时序特征与截面特征

> 特征工程是 ML 量化的灵魂。好特征让模型事半功倍，差特征让模型过拟合。

---

## 1. 量化特征的两个维度

| 维度 | 时序特征 | 截面特征 |
|------|---------|---------|
| 比较对象 | 该股票的历史 | 当期所有股票 |
| 信息类型 | 自身趋势/模式 | 相对排名/强弱 |
| 举例 | MA(20), RSI, MACD | 行业内排名、因子 z-score |

---

## 2. 时序特征构建

```python
import pandas as pd
import numpy as np
import talib  # 需要安装 TA-Lib

def build_time_series_features(close, high, low, volume, stock_code):
    """
    为单只股票构建时序特征
    """
    c = close[stock_code].values
    h = high[stock_code].values
    l = low[stock_code].values
    v = volume[stock_code].values
    
    features = pd.DataFrame(index=close.index)
    
    # === 动量特征 ===
    for period in [5, 10, 20, 60]:
        # 收益率
        features[f'ret_{period}'] = close[stock_code].pct_change(period)
        # 价格相对高点
        features[f'dist_high_{period}'] = close[stock_code] / \
            close[stock_code].rolling(period).max() - 1
        # 价格相对低点
        features[f'dist_low_{period}'] = close[stock_code] / \
            close[stock_code].rolling(period).min() - 1
    
    # === 波动率特征 ===
    ret = close[stock_code].pct_change()
    for period in [5, 10, 20, 60]:
        features[f'vol_{period}'] = ret.rolling(period).std()
        features[f'skew_{period}'] = ret.rolling(period).skew()
        features[f'kurt_{period}'] = ret.rolling(period).kurt()
    
    # === 技术指标 ===
    # RSI
    features['rsi_14'] = talib.RSI(c, timeperiod=14)
    
    # MACD
    macd, signal, _ = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    features['macd'] = macd
    features['macd_signal'] = signal
    features['macd_hist'] = macd - signal
    
    # 布林带
    upper, middle, lower = talib.BBANDS(c, timeperiod=20)
    features['bb_upper'] = (upper - c) / (upper - lower + 1e-8)
    features['bb_lower'] = (c - lower) / (upper - lower + 1e-8)
    features['bb_width'] = (upper - lower) / middle
    
    # === 成交量特征 ===
    features['vol_ratio_5'] = pd.Series(v).rolling(5).mean() / \
        pd.Series(v).rolling(20).mean()
    features['turnover_rate'] = pd.Series(v) / pd.Series(v).rolling(60).mean()
    
    # 量价背离
    features['price_vol_corr_5'] = ret.rolling(5).corr(
        pd.Series(v).pct_change()
    )
    
    return features
```

---

## 3. 截面特征构建

```python
def build_cross_sectional_features(factor_df):
    """
    将因子值转化为截面相对特征
    factor_df: 单个因子，index=时间，columns=股票代码
    """
    features = {}
    
    # 截面排名（百分位）
    features['rank_pct'] = factor_df.rank(axis=1, pct=True)
    
    # 截面 Z-score
    features['zscore'] = factor_df.sub(factor_df.mean(axis=1), axis=0).div(
        factor_df.std(axis=1), axis=0
    )
    
    # 行业内排名
    # 需要外部 industry_map 参数
    
    return features


def build_interaction_features(feature_df):
    """
    构建交叉特征（谨慎使用，容易过拟合）
    """
    # 因子间乘积
    cols = feature_df.columns[:5]  # 只用前5个避免组合爆炸
    interactions = pd.DataFrame(index=feature_df.index)
    
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i < j:
                interactions[f'{c1}_x_{c2}'] = feature_df[c1] * feature_df[c2]
    
    return interactions
```

---

## 4. 时间序列的正确 Train/Test 分割

**金融数据的特殊性**：必须保持时间顺序，不能随机分割！

```python
class TimeSeriesSplit:
    """
    时间序列的正确训练/测试分割
    """
    
    def purged_kfold(self, dates, n_splits=5, purge_gap=5):
        """
        带 Purge Gap 的时序交叉验证
        purge_gap: 训练集末尾和测试集开头之间的空白期（防止信息泄露）
        """
        n = len(dates)
        fold_size = n // (n_splits + 1)
        
        splits = []
        for i in range(n_splits):
            test_start = (i + 1) * fold_size
            test_end = test_start + fold_size
            train_end = test_start - purge_gap
            
            train_idx = list(range(0, train_end))
            test_idx = list(range(test_start, min(test_end, n)))
            
            splits.append((
                dates[train_idx],
                dates[test_idx]
            ))
        
        return splits
    
    def walk_forward(self, dates, train_window=36, test_window=6, step=3):
        """
        滚动窗口交叉验证（Walk-Forward）
        """
        splits = []
        start = train_window
        
        while start + test_window <= len(dates):
            train = dates[start - train_window:start]
            test = dates[start:start + test_window]
            splits.append((train, test))
            start += step
        
        return splits
```

---

## 5. 特征重要性分析

```python
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

def feature_importance_analysis(X_train, y_train, feature_names):
    """
    用梯度提升树分析特征重要性
    注意：特征重要性 != 因子 IC，不能直接比较
    """
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    
    # 前20个重要特征
    top20 = importance.head(20)
    print(top20)
    
    return importance
```

---

## 6. 特征工程的陷阱

| 陷阱 | 说明 | 防范 |
|------|------|------|
| 未来特征 | 用了未来数据计算特征 | 检查时间索引对齐 |
| 标准化泄露 | 用全量数据做标准化 | 用训练期均值/方差变换测试集 |
| 过多特征 | 特征越多过拟合风险越高 | 特征选择 + 正则化 |
| 高度相关特征 | 重复信息，浪费模型容量 | 相关性过滤 |

---

## 小结

| 维度 | 内容 |
|------|------|
| 时序特征 | 动量、波动率、技术指标、量价关系 |
| 截面特征 | 排名、Z-score、行业内相对强弱 |
| 正确分割 | 必须保持时间顺序，加 Purge Gap |
| 特征重要性 | 辅助参考，不能替代因子 IC |
| 核心原则 | 宁少勿多，每个特征都要有经济学逻辑 |
