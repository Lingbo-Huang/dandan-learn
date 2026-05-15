---
layout: default
title: "D1 · 编程题：数据处理与回测"
render_with_liquid: false
---

# D1 · 编程题：数据处理与回测

> 量化面试编程题的核心：不是算法竞赛，而是金融数据处理的工程能力。

---

## 1. 面试编程题的常见类型

| 类型 | 难度 | 频率 |
|------|------|------|
| Pandas 数据处理 | ★★ | 极高 |
| 因子计算 | ★★★ | 高 |
| 回测框架搭建 | ★★★ | 中 |
| NumPy 矩阵运算 | ★★ | 中 |
| 算法题（LeetCode 中等）| ★★★ | 低（外资多）|

---

## 2. 高频 Pandas 题

### 题目 1：计算滚动 Beta

```python
import pandas as pd
import numpy as np

# 给定日度收益率数据，计算每只股票相对大盘的 60 日滚动 Beta
def rolling_beta(stock_returns, market_returns, window=60):
    """
    stock_returns: DataFrame，index=date，columns=stock_code
    market_returns: Series，index=date
    """
    result = pd.DataFrame(index=stock_returns.index, 
                          columns=stock_returns.columns)
    
    for stock in stock_returns.columns:
        s = stock_returns[stock]
        m = market_returns
        
        # 滚动协方差
        rolling_cov = s.rolling(window).cov(m)
        # 滚动方差
        rolling_var = m.rolling(window).var()
        
        result[stock] = rolling_cov / rolling_var
    
    return result.astype(float)

# 更简洁的向量化写法
def rolling_beta_vectorized(stock_returns, market_returns, window=60):
    """使用 corr + std 的关系：beta = corr * (std_stock / std_market)"""
    std_stock = stock_returns.rolling(window).std()
    std_market = market_returns.rolling(window).std()
    corr = stock_returns.rolling(window).corr(market_returns)
    return corr * (std_stock.div(std_market, axis=0))
```

### 题目 2：计算 IC 序列

```python
def calc_ic_series(factor_df, return_df, method='rank'):
    """
    给定月度因子值和下月收益，计算每月 IC
    """
    from scipy.stats import spearmanr, pearsonr
    
    ic_list = []
    
    for date in factor_df.index:
        if date not in return_df.index:
            continue
        
        f = factor_df.loc[date].dropna()
        r = return_df.loc[date].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        
        if len(common) < 10:
            ic_list.append({'date': date, 'ic': np.nan})
            continue
        
        if method == 'rank':
            ic, _ = spearmanr(f[common], r[common])
        else:
            ic, _ = pearsonr(f[common], r[common])
        
        ic_list.append({'date': date, 'ic': ic})
    
    return pd.DataFrame(ic_list).set_index('date')['ic']
```

### 题目 3：最大回撤计算

```python
def max_drawdown(returns):
    """
    计算最大回撤
    注意：面试中要求思路清晰，能处理 Series 和 array 两种输入
    """
    if isinstance(returns, pd.Series):
        cum = (1 + returns).cumprod()
    else:
        cum = np.cumprod(1 + np.array(returns))
    
    rolling_max = np.maximum.accumulate(cum)
    drawdowns = (cum - rolling_max) / rolling_max
    return drawdowns.min()

def drawdown_series(returns):
    """返回完整回撤序列"""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    return (cum - rolling_max) / rolling_max
```

---

## 3. 因子计算题

### 题目 4：因子标准化 Pipeline

```python
def factor_preprocess(factor_df, winsorize_pct=0.025):
    """
    完整因子预处理 Pipeline
    去极值 → Z-score
    
    面试常见：考察你是否知道需要截面操作（axis=1）而不是全局
    """
    from scipy.stats.mstats import winsorize
    
    result = factor_df.copy()
    
    # 截面操作！不是全局操作！
    for date in factor_df.index:
        row = factor_df.loc[date].dropna()
        
        if len(row) < 5:
            continue
        
        # 去极值
        row_wins = pd.Series(
            winsorize(row.values, limits=[winsorize_pct, winsorize_pct]),
            index=row.index
        )
        
        # Z-score（截面）
        mu, sigma = row_wins.mean(), row_wins.std()
        if sigma > 1e-8:
            result.loc[date, row.index] = (row_wins - mu) / sigma
    
    return result


def factor_rank_pct(factor_df):
    """
    因子转百分位排名（截面）
    更简洁的实现
    """
    return factor_df.rank(axis=1, pct=True)
```

### 题目 5：因子分层回测

```python
def quantile_backtest(factor_df, return_df, n_groups=5):
    """
    因子分层回测
    考察：能否正确对齐时间（factor用t期，return用t+1期）
    """
    # 关键：factor 是 t 期数据，预测 t+1 期收益
    # forward_return 是 t+1 期收益
    forward_return = return_df.shift(-1)
    
    group_returns = {f'Q{i+1}': [] for i in range(n_groups)}
    dates = []
    
    for date in factor_df.index[:-1]:  # 最后一期没有未来收益
        if date not in forward_return.index:
            continue
        
        factor = factor_df.loc[date].dropna()
        ret = forward_return.loc[date].reindex(factor.index).dropna()
        common = factor.index.intersection(ret.index)
        
        if len(common) < n_groups * 5:
            continue
        
        # 按因子值分组
        factor_sorted = factor[common]
        labels = pd.qcut(factor_sorted, q=n_groups, labels=False, 
                          duplicates='drop')
        
        for g in range(n_groups):
            group_stocks = labels[labels == g].index
            group_ret = ret[group_stocks].mean()
            group_returns[f'Q{g+1}'].append(group_ret)
        
        dates.append(date)
    
    result = pd.DataFrame(group_returns, index=dates)
    result['LS'] = result[f'Q{n_groups}'] - result['Q1']
    
    return result
```

---

## 4. 回测框架题

### 题目 6：简单等权回测

```python
def simple_backtest(weights_df, returns_df, cost_bps=15):
    """
    给定每期权重和收益率，计算策略净值
    
    面试陷阱：
    1. 权重要用 t 期决定，但 t+1 期才执行（shift(1)）
    2. 要考虑交易成本
    3. 归一化权重
    """
    # 权重归一化
    weights = weights_df.div(weights_df.abs().sum(axis=1), axis=0)
    
    # 执行时间偏移（次期执行）
    weights_lagged = weights.shift(1)
    
    # 换手率（每期权重变化量的绝对值之和 / 2）
    turnover = (weights_lagged - weights_lagged.shift(1)).abs().sum(axis=1) / 2
    
    # 交易成本
    cost = turnover * cost_bps / 10000
    
    # 组合收益
    port_ret = (weights_lagged * returns_df).sum(axis=1) - cost
    
    # 净值
    nav = (1 + port_ret).cumprod()
    
    return port_ret, nav, turnover
```

---

## 5. 面试技巧

1. **先想清楚再写**：说出思路，询问输入假设
2. **截面 vs. 时序**：因子处理是截面操作，不要用全局均值
3. **时间对齐**：factor(t) 预测 return(t+1)，务必正确 shift
4. **边界情况**：空值处理、零除问题
5. **向量化优先**：避免显式 for 循环，用 pandas/numpy 内置函数

---

## 常见错误一览

```python
# ❌ 错误：用全局均值做 z-score（应该用截面）
factor_z = (factor_df - factor_df.mean()) / factor_df.std()  # Wrong!

# ✅ 正确：截面 z-score
factor_z = factor_df.sub(factor_df.mean(axis=1), axis=0).div(
    factor_df.std(axis=1), axis=0
)

# ❌ 错误：factor 和 return 未对齐（未来函数）
port_ret = (factor_df * return_df).sum(axis=1)  # Wrong! 同期

# ✅ 正确：factor(t) → return(t+1)
port_ret = (factor_df.shift(1) * return_df).sum(axis=1)
```
