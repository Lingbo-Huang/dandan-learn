---
layout: default
title: "D6 · 高频数据处理与分析"
render_with_liquid: false
---

# D6 · 高频数据处理与分析

> Tick 数据、Level 2 数据的清洗与分析，是量化工程师必备的数据工程能力。

---

## 1. 高频数据类型

| 数据类型 | 频率 | 内容 | 用途 |
|---------|------|------|------|
| Tick 数据 | 每笔成交 | 价格、数量、买卖方向 | 微结构分析 |
| Level 1 | 秒级 | 最优买卖价 + 成交 | 基础量化 |
| Level 2 | 秒级 | 10档盘口 + 逐笔 | 做市、高频 |
| 快照 | 3秒/分钟 | 行情快照 | 中频策略 |

---

## 2. 高频数据清洗

```python
import pandas as pd
import numpy as np

def clean_tick_data(tick_df):
    """
    Tick 数据清洗标准流程
    """
    df = tick_df.copy()
    
    # 1. 时间戳对齐（确保单调递增）
    df = df.sort_values('timestamp')
    df = df[df['timestamp'].diff() >= 0]  # 去掉时间戳回滚的数据
    
    # 2. 去除明显异常价格（超过前值 ±20%）
    price_change = df['price'].pct_change().abs()
    df = df[price_change < 0.2]
    
    # 3. 去除零成交量
    df = df[df['volume'] > 0]
    
    # 4. 处理涨跌停导致的价格锁定
    # 连续相同价格超过 100 笔可能是封板，保留
    
    # 5. 时区对齐
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Shanghai')
    
    # 6. 过滤非交易时段（A股：9:30-11:30, 13:00-15:00）
    time = df['timestamp'].dt.time
    trading_morning = (time >= pd.Timestamp('09:30').time()) & \
                      (time <= pd.Timestamp('11:30').time())
    trading_afternoon = (time >= pd.Timestamp('13:00').time()) & \
                        (time <= pd.Timestamp('15:00').time())
    df = df[trading_morning | trading_afternoon]
    
    return df.reset_index(drop=True)


def identify_trade_direction(tick_df):
    """
    推断成交方向（主动买/卖）
    使用 Lee-Ready 算法：与前一个 tick 价格比较
    """
    df = tick_df.copy()
    df['direction'] = np.nan
    
    for i in range(1, len(df)):
        if df['price'].iloc[i] > df['price'].iloc[i-1]:
            df['direction'].iloc[i] = 1   # 主动买（价格上涨）
        elif df['price'].iloc[i] < df['price'].iloc[i-1]:
            df['direction'].iloc[i] = -1  # 主动卖（价格下跌）
        else:
            # 价格不变，沿用前一个方向
            df['direction'].iloc[i] = df['direction'].iloc[i-1]
    
    return df
```

---

## 3. Tick 数据聚合（生成分钟线）

```python
def tick_to_ohlcv(tick_df, freq='1min'):
    """
    将 Tick 数据聚合为 OHLCV 数据
    """
    df = tick_df.set_index('timestamp')
    
    ohlcv = df['price'].resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    ohlcv['volume'] = df['volume'].resample(freq).sum()
    ohlcv['amount'] = (df['price'] * df['volume']).resample(freq).sum()
    ohlcv['vwap'] = ohlcv['amount'] / ohlcv['volume']
    ohlcv['n_trades'] = df['price'].resample(freq).count()
    
    # 买卖方向聚合
    if 'direction' in df.columns:
        buy_vol = df[df['direction'] == 1]['volume'].resample(freq).sum()
        sell_vol = df[df['direction'] == -1]['volume'].resample(freq).sum()
        ohlcv['buy_ratio'] = buy_vol / (buy_vol + sell_vol + 1e-8)
    
    return ohlcv.dropna(subset=['close'])
```

---

## 4. 高频特征计算

```python
class HFTFeatures:
    """高频数据特征工程"""
    
    def calc_order_flow_imbalance(self, tick_df, window_seconds=60):
        """订单流失衡（OFI）"""
        df = tick_df.set_index('timestamp')
        
        buy_vol = (df[df['direction'] == 1]['volume']
                   .resample(f'{window_seconds}s').sum())
        sell_vol = (df[df['direction'] == -1]['volume']
                    .resample(f'{window_seconds}s').sum())
        
        ofi = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8)
        return ofi
    
    def calc_price_impact(self, tick_df, window_seconds=30):
        """
        成交价格冲击（Amihud 非流动性）
        |收益率| / 成交额，越大说明流动性越差
        """
        minute_data = tick_to_ohlcv(tick_df, freq=f'{window_seconds}s')
        ret = minute_data['close'].pct_change().abs()
        amihud = ret / (minute_data['amount'] / 1e6 + 1e-8)  # 以百万元为单位
        return amihud
    
    def calc_realized_volatility(self, tick_df, freq='5min'):
        """
        已实现波动率（Realized Volatility）
        比日度历史波动率更精准
        """
        ohlcv = tick_to_ohlcv(tick_df, freq=freq)
        log_ret = np.log(ohlcv['close'] / ohlcv['close'].shift(1))
        
        # 日度已实现波动率
        rv = log_ret.resample('1D').apply(lambda x: np.sqrt((x**2).sum()))
        
        # 年化
        rv_annual = rv * (252 ** 0.5)
        return rv_annual
    
    def calc_spread_from_tick(self, tick_df):
        """
        从 Tick 数据估算有效价差
        有效价差 = 2 × |成交价 - 中间价|
        """
        # 如果有 Level 2 数据
        if 'bid1' in tick_df.columns and 'ask1' in tick_df.columns:
            mid = (tick_df['bid1'] + tick_df['ask1']) / 2
            effective_spread = 2 * (tick_df['price'] - mid).abs() / mid
        else:
            # 用 Roll 估算器
            price_change = tick_df['price'].diff()
            cov = price_change.cov(price_change.shift(1))
            if cov < 0:
                effective_spread = 2 * np.sqrt(-cov) / tick_df['price'].mean()
            else:
                effective_spread = pd.Series(0, index=tick_df.index)
        
        return effective_spread
```

---

## 5. 日内模式分析

```python
def analyze_intraday_patterns(tick_df, stock_universe):
    """
    分析A股典型日内模式
    """
    results = {}
    
    for stock in stock_universe:
        df = tick_df[tick_df['symbol'] == stock].copy()
        df['time_bucket'] = df['timestamp'].dt.floor('5min').dt.time
        
        # 各时段成交量
        vol_by_time = df.groupby('time_bucket')['volume'].mean()
        
        # 各时段波动率
        ohlcv = tick_to_ohlcv(df, '5min')
        vol_by_time_std = ohlcv['close'].pct_change().abs().groupby(
            ohlcv.index.time
        ).mean()
        
        results[stock] = {
            'volume_pattern': vol_by_time,
            'vol_pattern': vol_by_time_std
        }
    
    # A股典型规律：
    # 9:30-9:40 开盘放量高波动
    # 11:00-11:30 临近午休缩量
    # 13:00-13:10 午后开盘
    # 14:30-15:00 尾盘放量
    
    return results
```

---

## 小结

| 工作 | 工具/方法 |
|------|---------|
| 数据清洗 | 时间戳排序、异常价格过滤、非交易时段去除 |
| 方向识别 | Lee-Ready 算法 |
| Tick 聚合 | pandas resample |
| 高频特征 | OFI、已实现波动率、Amihud 非流动性 |
| 日内规律 | 分时成交量、波动率模式 |
