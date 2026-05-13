---
layout: default
title: "WhaleQuant Ch05 · 量化择时策略"
source: "https://github.com/datawhalechina/whale-quant"
---

# 05 · 量化择时策略

> **来源**：[WhaleQuant](https://github.com/datawhalechina/whale-quant) · datawhalechina

---

## 5.1 择时策略概述

**择时（Market Timing）**：判断市场的涨跌趋势，决定何时持有多少仓位。

择时策略核心问题：
- 什么时候进入市场（买入/加仓）？
- 什么时候离开市场（卖出/减仓）？
- 仓位应该怎么变化？

---

## 5.2 均线策略

### 双均线金叉/死叉

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ma_crossover_strategy(prices: pd.Series, 
                           short_window: int = 20,
                           long_window: int = 60) -> pd.DataFrame:
    """
    双均线择时策略
    金叉（短线上穿长线）→ 买入
    死叉（短线下穿长线）→ 卖出
    """
    df = pd.DataFrame({'price': prices})
    
    # 计算均线
    df['ma_short'] = df['price'].rolling(short_window).mean()
    df['ma_long'] = df['price'].rolling(long_window).mean()
    
    # 生成信号
    df['signal'] = 0
    df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1   # 持仓
    df.loc[df['ma_short'] < df['ma_long'], 'signal'] = 0   # 空仓
    
    # 持仓变化（金叉/死叉点）
    df['position_change'] = df['signal'].diff()
    df['golden_cross'] = df['position_change'] > 0   # 金叉
    df['death_cross'] = df['position_change'] < 0    # 死叉
    
    # 策略收益
    df['daily_return'] = df['price'].pct_change()
    df['strategy_return'] = df['signal'].shift(1) * df['daily_return']
    
    # 净值
    df['strategy_nav'] = (1 + df['strategy_return']).cumprod()
    df['buyhold_nav'] = (1 + df['daily_return']).cumprod()
    
    return df

# 参数优化（遍历参数组合）
def optimize_ma_params(prices: pd.Series):
    """遍历不同均线参数，找最优"""
    results = []
    
    for short in range(5, 30, 5):
        for long in range(30, 120, 10):
            if short >= long:
                continue
            df = ma_crossover_strategy(prices, short, long)
            df = df.dropna()
            
            sharpe = (df['strategy_return'].mean() / 
                      df['strategy_return'].std() * np.sqrt(252))
            
            cum_ret = df['strategy_nav'].iloc[-1]
            max_dd = ((df['strategy_nav'] - df['strategy_nav'].cummax()) / 
                       df['strategy_nav'].cummax()).min()
            
            results.append({
                'short': short, 'long': long,
                'sharpe': sharpe,
                'cum_return': cum_ret,
                'max_drawdown': max_dd
            })
    
    return pd.DataFrame(results).sort_values('sharpe', ascending=False)
```

---

## 5.3 布林带策略

```python
def bollinger_bands_strategy(prices: pd.Series,
                              window: int = 20,
                              n_std: float = 2.0) -> pd.DataFrame:
    """
    布林带策略：
    价格触碰下轨 → 超卖，买入
    价格触碰上轨 → 超买，卖出
    （均值回归逻辑）
    """
    df = pd.DataFrame({'price': prices})
    
    df['mid'] = df['price'].rolling(window).mean()
    df['std'] = df['price'].rolling(window).std()
    df['upper'] = df['mid'] + n_std * df['std']
    df['lower'] = df['mid'] - n_std * df['std']
    
    # 百分比宽度（衡量波动率扩张/收缩）
    df['bandwidth'] = (df['upper'] - df['lower']) / df['mid']
    
    # %B 指标（价格在布林带中的位置，0-1）
    df['percent_b'] = (df['price'] - df['lower']) / (df['upper'] - df['lower'])
    
    # 交易信号
    df['signal'] = 0
    # 当%B < 0.2，超卖信号，买入
    df.loc[df['percent_b'] < 0.2, 'signal'] = 1
    # 当%B > 0.8，超买信号，卖出
    df.loc[df['percent_b'] > 0.8, 'signal'] = -1
    
    return df
```

---

## 5.4 RSI 策略

```python
def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """计算 RSI（相对强弱指数）"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_strategy(prices: pd.Series, 
                  rsi_window: int = 14,
                  oversold: float = 30,
                  overbought: float = 70) -> pd.DataFrame:
    """
    RSI 择时策略：
    RSI < 30：超卖，买入
    RSI > 70：超买，卖出
    """
    df = pd.DataFrame({'price': prices})
    df['rsi'] = calculate_rsi(prices, rsi_window)
    
    # 信号
    df['signal'] = 0
    # 超卖区域进入持仓
    df.loc[df['rsi'] < oversold, 'signal'] = 1
    # 超买区域退出持仓
    df.loc[df['rsi'] > overbought, 'signal'] = 0
    # 前向填充（保持持仓状态直到下一个信号）
    df['signal'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
    
    df['daily_return'] = df['price'].pct_change()
    df['strategy_return'] = df['signal'].shift(1) * df['daily_return']
    df['strategy_nav'] = (1 + df['strategy_return']).cumprod()
    
    return df
```

---

## 5.5 CTA 趋势跟踪

```python
def atr_position_sizing(prices: pd.Series, 
                          atr_window: int = 20,
                          risk_per_trade: float = 0.01,
                          capital: float = 1_000_000) -> pd.Series:
    """
    基于 ATR 的仓位管理
    每笔交易风险 = 总资本 × risk_per_trade
    仓位 = 风险金额 / ATR
    """
    high = prices * 1.005  # 模拟高价（实际使用真实 high）
    low = prices * 0.995   # 模拟低价
    
    tr = pd.concat([
        high - low,
        (high - prices.shift(1)).abs(),
        (low - prices.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(atr_window).mean()
    
    risk_amount = capital * risk_per_trade
    position_size = risk_amount / atr  # 股数
    
    return position_size

# 突破策略（唐奇安通道）
def donchian_channel_strategy(prices: pd.Series,
                               entry_window: int = 20,
                               exit_window: int = 10) -> pd.DataFrame:
    """
    唐奇安通道（海龟交易法则基础）
    20日最高价突破 → 买入
    10日最低价突破 → 卖出
    """
    df = pd.DataFrame({'price': prices})
    
    df['entry_high'] = df['price'].rolling(entry_window).max().shift(1)
    df['entry_low'] = df['price'].rolling(entry_window).min().shift(1)
    df['exit_high'] = df['price'].rolling(exit_window).max().shift(1)
    df['exit_low'] = df['price'].rolling(exit_window).min().shift(1)
    
    position = 0
    positions = []
    
    for i, row in df.iterrows():
        if position == 0:
            if row['price'] > row['entry_high']:  # 上轨突破
                position = 1
        elif position == 1:
            if row['price'] < row['exit_low']:    # 下轨突破退出
                position = 0
        positions.append(position)
    
    df['position'] = positions
    df['daily_return'] = df['price'].pct_change()
    df['strategy_return'] = df['position'].shift(1) * df['daily_return']
    df['strategy_nav'] = (1 + df['strategy_return']).cumprod()
    
    return df
```

---

## 延伸阅读

- [WhaleQuant 完整教程](https://github.com/datawhalechina/whale-quant)
- 《海龟交易法则》（Curtis M. Faith）
- 《主动投资组合管理》（Grinold & Kahn）
