---
layout: default
title: "D3 · 截面动量策略"
render_with_liquid: false
---

# D3 · 截面动量策略

> 截面动量是连接因子研究与实际策略的桥梁。本节从选股逻辑到实盘注意全面讲解。

---

## 1. 截面动量 vs. 时序动量

| 维度 | 截面动量 | 时序动量（CTA）|
|------|---------|--------------|
| 比较对象 | 股票间相对强弱 | 单资产自身历史 |
| 持仓方向 | 多空（买强卖弱）| 随趋势双向 |
| 应用场景 | 股票选股 | 期货/外汇 |
| 持仓周期 | 月度 | 日-月 |

---

## 2. 标准截面动量策略

```python
import pandas as pd
import numpy as np

class CrossSectionalMomentum:
    def __init__(self, lookback=12, skip=1, n_long=50, n_short=50,
                 rebalance='M'):
        self.lookback = lookback  # 回看期（月）
        self.skip = skip          # 跳过最近期数（避免短期反转）
        self.n_long = n_long      # 做多股票数
        self.n_short = n_short    # 做空股票数
    
    def calc_momentum_signal(self, prices):
        """计算截面动量因子"""
        # 月度价格（取月末价）
        monthly = prices.resample('ME').last()
        
        # 计算 lookback 月累积收益，跳过最近 skip 月
        # 即用 t-lookback 到 t-skip 的累积收益
        cum_ret = monthly.shift(self.skip) / monthly.shift(self.lookback) - 1
        return cum_ret
    
    def generate_portfolio(self, signal, date):
        """给定日期，生成多空组合"""
        scores = signal.loc[date].dropna()
        
        # 按动量排名
        sorted_scores = scores.sort_values(ascending=False)
        
        long_stocks = sorted_scores.head(self.n_long).index.tolist()
        short_stocks = sorted_scores.tail(self.n_short).index.tolist()
        
        return {'long': long_stocks, 'short': short_stocks}
    
    def backtest(self, prices, rebalance_dates=None):
        """回测截面动量策略"""
        signal = self.calc_momentum_signal(prices)
        monthly_ret = prices.pct_change(21)  # 月度收益
        
        if rebalance_dates is None:
            rebalance_dates = signal.index
        
        portfolio_returns = []
        
        for date in rebalance_dates:
            if date not in signal.index:
                continue
            
            portfolio = self.generate_portfolio(signal, date)
            
            # 持仓期为下个月
            next_date_idx = signal.index.get_loc(date) + 1
            if next_date_idx >= len(signal.index):
                break
            next_date = signal.index[next_date_idx]
            
            # 持仓期收益
            period_ret = prices.loc[date:next_date].pct_change().iloc[1:]
            
            if len(period_ret) == 0:
                continue
            
            long_ret = period_ret[portfolio['long']].mean(axis=1).mean()
            short_ret = period_ret[portfolio['short']].mean(axis=1).mean()
            
            port_ret = 0.5 * long_ret - 0.5 * short_ret
            portfolio_returns.append({'date': next_date, 'return': port_ret})
        
        return pd.DataFrame(portfolio_returns).set_index('date')['return']
```

---

## 3. 动量策略优化

### 3.1 特质动量（Industry-Neutral Momentum）

```python
def calc_residual_momentum(prices, industry_map, lookback=12, skip=1):
    """
    行业中性化后的特质动量
    去掉行业共同走势，只保留个股相对行业的超额动量
    """
    monthly = prices.resample('ME').last()
    raw_mom = monthly.shift(skip) / monthly.shift(lookback) - 1
    
    residual_mom = raw_mom.copy()
    
    for date in raw_mom.index:
        scores = raw_mom.loc[date].dropna()
        # 对每个行业内部做中性化
        for ind in industry_map.unique():
            ind_stocks = industry_map[industry_map == ind].index
            common = scores.index.intersection(ind_stocks)
            if len(common) < 2:
                continue
            ind_mean = scores[common].mean()
            residual_mom.loc[date, common] = scores[common] - ind_mean
    
    return residual_mom
```

### 3.2 52 周高点动量

```python
def calc_52w_high_momentum(prices, window=252):
    """
    George & Hwang (2004): 股价接近52周高点的股票更强
    """
    high_52w = prices.rolling(window).max()
    nearness = prices / high_52w  # 越接近1说明越接近高点
    return nearness
```

---

## 4. 中国市场截面动量的特殊性

**A股动量效应弱于美股，原因**：

1. **散户主导**：追涨杀跌导致短期过度反应，随后反转
2. **信息环境**：内幕交易导致价格提前反映，压缩动量空间
3. **制度因素**：涨跌停、T+1 制度限制套利

**A股最有效的"动量"形式**：
- **行业轮动**：行业层面的动量效应比个股强
- **短期反转**：1-5日内，反转而非动量
- **盈利超预期动量（PEAD）**：业绩超预期后股价持续上涨

```python
def calc_pead_signal(actual_eps, estimated_eps, prices, window=60):
    """
    盈利超预期动量（Post-Earnings Announcement Drift）
    SUE = (实际EPS - 预期EPS) / 股价波动率
    """
    sue = (actual_eps - estimated_eps) / prices.rolling(window).std()
    return sue
```

---

## 5. 交易成本与换手

```python
def net_of_cost_return(gross_return, turnover, one_way_cost=0.003):
    """
    考虑交易成本后的净收益
    one_way_cost: 单边成本（印花税+佣金+冲击，约0.15%-0.3%）
    """
    total_cost = turnover * one_way_cost * 2  # 双边
    return gross_return - total_cost

def calc_turnover(portfolio_prev, portfolio_curr):
    """计算换手率"""
    all_stocks = set(portfolio_prev) | set(portfolio_curr)
    change = sum(1 for s in all_stocks if 
                 (s in portfolio_prev) != (s in portfolio_curr))
    return change / max(len(portfolio_prev), len(portfolio_curr))
```

---

## 小结

| 维度 | 内容 |
|------|------|
| 核心逻辑 | 强者恒强，截面多强空弱 |
| 标准实现 | 12-1 月累积收益排名 |
| 优化方向 | 行业中性、特质动量、52周高点 |
| A股特殊 | 短期反转强，PEAD 效应值得关注 |
| 最大成本 | 换手率高，需关注净收益 |
