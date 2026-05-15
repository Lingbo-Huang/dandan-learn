---
layout: default
title: "D2 · 统计套利：配对交易"
render_with_liquid: false
---

# D2 · 统计套利：配对交易

> 两个高度相关的股票，价差出现偏离时买便宜卖贵，等待均值回归。听起来简单，做好很难。

---

## 1. 配对交易的核心思想

**统计套利（Statistical Arbitrage）**：利用资产间的统计关系（均值回归）进行交易，赚取价差收敛的收益。

**配对交易**是最简单的统计套利：
1. 找到两个价格长期协整的股票
2. 当价差偏离均值（超出阈值）时建仓
3. 价差回归均值时平仓

---

## 2. 协整检验

**协整（Cointegration）** ≠ 相关性：
- 相关性：两者同向变动（短期）
- 协整：两者存在长期稳定的线性关系（即使短期偏离也会回归）

```python
from statsmodels.tsa.stattools import coint, adfuller
import numpy as np
import pandas as pd

def test_cointegration(price_a, price_b, significance=0.05):
    """
    Engle-Granger 协整检验
    H0: 两者不协整（无长期均衡关系）
    p-value < 0.05 表示拒绝 H0，即存在协整关系
    """
    score, pvalue, _ = coint(price_a, price_b)
    is_cointegrated = pvalue < significance
    return {
        'score': score,
        'pvalue': pvalue,
        'is_cointegrated': is_cointegrated
    }

def find_pairs(price_df, significance=0.05):
    """批量寻找协整配对"""
    stocks = price_df.columns.tolist()
    pairs = []
    
    for i in range(len(stocks)):
        for j in range(i+1, len(stocks)):
            s1, s2 = stocks[i], stocks[j]
            result = test_cointegration(price_df[s1], price_df[s2], significance)
            if result['is_cointegrated']:
                pairs.append({
                    'stock_a': s1,
                    'stock_b': s2,
                    'pvalue': result['pvalue']
                })
    
    return pd.DataFrame(pairs).sort_values('pvalue')
```

---

## 3. 价差计算与信号生成

```python
import statsmodels.api as sm

class PairTradingStrategy:
    def __init__(self, entry_threshold=2.0, exit_threshold=0.5, lookback=60):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.lookback = lookback
        self.hedge_ratio = None
    
    def fit(self, price_a, price_b):
        """用 OLS 估计对冲比例（hedge ratio）"""
        y = price_a
        X = sm.add_constant(price_b)
        model = sm.OLS(y, X).fit()
        self.hedge_ratio = model.params.iloc[1]
        return self
    
    def calc_spread(self, price_a, price_b):
        """计算价差"""
        if self.hedge_ratio is None:
            raise ValueError("请先调用 fit()")
        spread = price_a - self.hedge_ratio * price_b
        return spread
    
    def generate_signals(self, price_a, price_b):
        """
        基于滚动 z-score 生成交易信号
        +1: 价差高 -> 做空A做多B（等待价差下降）
        -1: 价差低 -> 做多A做空B
        0: 无仓位
        """
        spread = self.calc_spread(price_a, price_b)
        
        # 滚动 z-score
        spread_mean = spread.rolling(self.lookback).mean()
        spread_std = spread.rolling(self.lookback).std()
        z_score = (spread - spread_mean) / (spread_std + 1e-8)
        
        signal = pd.Series(0, index=spread.index)
        
        # 进场：z-score 超过阈值
        signal[z_score > self.entry_threshold] = -1  # 价差高，做空价差
        signal[z_score < -self.entry_threshold] = 1   # 价差低，做多价差
        
        # 出场：z-score 回到中性区域
        # （实际实现需要状态机，这里用简化逻辑）
        signal[(z_score.abs() < self.exit_threshold)] = 0
        
        return signal, z_score
    
    def calc_returns(self, signal, price_a, price_b):
        """计算配对策略收益"""
        ret_a = price_a.pct_change()
        ret_b = price_b.pct_change()
        
        pos_a = signal.shift(1)  # 次日执行
        pos_b = -signal.shift(1) * self.hedge_ratio  # 对冲反向
        
        # 标准化仓位（投入资金相等）
        norm = abs(pos_a) + abs(pos_b) * price_b / price_a
        port_ret = (pos_a * ret_a + pos_b * ret_b) / (norm + 1e-8)
        
        return port_ret
```

---

## 4. Kalman Filter 动态对冲

```python
from pykalman import KalmanFilter

def kalman_hedge_ratio(price_a, price_b):
    """
    用卡尔曼滤波动态估计对冲比例
    比固定 OLS 对冲比更适应市场变化
    """
    obs_mat = np.vstack([price_b.values, np.ones(len(price_b))]).T
    kf = KalmanFilter(
        n_dim_obs=1, n_dim_state=2,
        initial_state_mean=[0, 0],
        initial_state_covariance=np.eye(2),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat.reshape(-1, 1, 2),
        observation_covariance=1.0,
        transition_covariance=0.001 * np.eye(2)
    )
    state_means, _ = kf.filter(price_a.values)
    hedge_ratio = pd.Series(state_means[:, 0], index=price_a.index)
    return hedge_ratio
```

---

## 5. 配对交易的风险

| 风险 | 说明 | 应对 |
|------|------|------|
| 协整关系破裂 | 公司基本面变化（合并、退市）| 设置止损，定期重新检验 |
| 价差持续扩大 | 反向趋势单边突破 | 固定最大持仓时间 |
| 流动性风险 | 同时买卖两只股票的成本 | 选择高流动性配对 |
| 统计假阳性 | 批量测试导致虚假协整 | Bonferroni 校正 |

---

## 6. A股配对交易实践

**常见配对方向**：
- 同行业龙头：贵州茅台 vs. 五粮液
- 同一产业链：锂矿股 vs. 锂电池股
- ETF 配对：沪深 300 vs. 中证 500

```python
# A股特殊注意事项
def a_share_pair_filter(price_df, min_liquidity=1e8):
    """
    A股配对前置过滤：
    1. 剔除 ST 股
    2. 确保两只股票同期流通（无停牌）
    3. 日均成交额 > 最低流动性门槛
    """
    # 剔除停牌（价格不变）
    unchanged = (price_df.diff().abs() < 1e-6).sum() / len(price_df)
    liquid = unchanged[unchanged < 0.1].index  # 停牌天数 < 10%
    return price_df[liquid]
```

---

## 小结

| 维度 | 内容 |
|------|------|
| 核心思想 | 协整关系 + 价差均值回归 |
| 关键检验 | Engle-Granger 协整检验 |
| 对冲比率 | OLS（固定）或 Kalman（动态）|
| 主要风险 | 协整破裂、价差持续扩大 |
| A股注意 | 流动性过滤、停牌处理 |
