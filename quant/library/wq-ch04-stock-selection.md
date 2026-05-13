---
layout: default
title: "WhaleQuant Ch04 · 量化选股策略"
source: "https://github.com/datawhalechina/whale-quant"
---

# 04 · 量化选股策略

> **来源**：[WhaleQuant](https://github.com/datawhalechina/whale-quant) · datawhalechina

---

## 4.1 选股策略概述

量化选股的核心：**找到能够预测未来股票收益的因子（Alpha 因子），并据此构建投资组合**。

```
因子研究 → 因子检验 → 多因子合成 → 组合构建 → 回测评估
```

---

## 4.2 经典单因子策略

### 市盈率（PE）反转策略

**逻辑**：低 PE 股票往往被市场低估，长期有超额收益（价值效应）。

```python
import pandas as pd
import numpy as np

def pe_selection_strategy(factor_data: pd.DataFrame, 
                           top_n: int = 50,
                           low_is_good: bool = True) -> pd.DataFrame:
    """
    基于 PE 的选股策略
    factor_data: MultiIndex DataFrame (date, stock) 包含 pe 和 next_month_return
    top_n: 选股数量
    low_is_good: PE 越低越好（价值策略）
    """
    portfolios = []
    
    for date in factor_data.index.get_level_values(0).unique():
        day_data = factor_data.loc[date].copy()
        day_data = day_data.dropna(subset=['pe', 'next_return'])
        
        # 排序选股
        if low_is_good:
            selected = day_data.nsmallest(top_n, 'pe')
        else:
            selected = day_data.nlargest(top_n, 'pe')
        
        # 等权重组合收益
        port_return = selected['next_return'].mean()
        portfolios.append({'date': date, 'return': port_return, 'n_stocks': len(selected)})
    
    return pd.DataFrame(portfolios).set_index('date')

# 评估多空组合（Long-Short）
def long_short_portfolio(factor_data, factor_col='pe', 
                          top_pct=0.2, bottom_pct=0.2):
    """构建多空组合，验证因子的 Long-Short 收益"""
    results = []
    
    for date in factor_data.index.get_level_values(0).unique():
        day_data = factor_data.loc[date].dropna(subset=[factor_col, 'next_return'])
        n = len(day_data)
        
        top_n = int(n * top_pct)
        bottom_n = int(n * bottom_pct)
        
        # 多头：因子值最低（PE低=价值）
        long_port = day_data.nsmallest(top_n, factor_col)['next_return'].mean()
        # 空头：因子值最高
        short_port = day_data.nlargest(bottom_n, factor_col)['next_return'].mean()
        
        results.append({
            'date': date,
            'long_return': long_port,
            'short_return': short_port,
            'ls_return': long_port - short_port  # Long-Short 收益
        })
    
    return pd.DataFrame(results).set_index('date')
```

---

## 4.3 Barra 多因子模型框架

### 常用因子库

| 因子类别 | 常见因子 | 经济逻辑 |
|---------|---------|---------|
| **价值** | PE、PB、PS、EV/EBITDA | 低估资产均值回归 |
| **成长** | 营收增速、利润增速、ROE变化 | 高增长公司溢价 |
| **质量** | ROE、ROA、毛利率、负债率 | 高质量公司长期跑赢 |
| **动量** | 过去3/6/12月收益率 | 趋势延续效应 |
| **低波动** | 过去收益率标准差 | 低风险异常高收益 |
| **规模** | 市值、流通市值 | 小盘股溢价 |
| **流动性** | 换手率、成交额 | 流动性溢价 |

### 多因子合成

```python
import pandas as pd
import numpy as np
from scipy import stats

def process_factor(factor_series: pd.Series) -> pd.Series:
    """因子预处理：去极值 + 标准化"""
    # 3倍MAD去极值
    median = factor_series.median()
    mad = (factor_series - median).abs().median()
    factor_series = factor_series.clip(median - 3*1.4826*mad, 
                                        median + 3*1.4826*mad)
    # Z-score标准化
    return (factor_series - factor_series.mean()) / (factor_series.std() + 1e-8)

def composite_factor(factors: dict, weights: dict = None) -> pd.Series:
    """
    多因子合成
    factors: {'pe': pd.Series, 'momentum': pd.Series, ...}
    weights: 各因子权重，None 则等权
    """
    processed = {}
    for name, series in factors.items():
        processed[name] = process_factor(series)
    
    df = pd.DataFrame(processed)
    
    if weights is None:
        return df.mean(axis=1)  # 等权
    else:
        w = pd.Series(weights)
        w = w / w.sum()  # 归一化
        return (df * w).sum(axis=1)

# 完整选股流程
def run_factor_strategy(returns_panel: pd.DataFrame, 
                         factor_panel: dict,
                         n_stocks: int = 50,
                         rebalance_freq: str = 'M'):  # 月度再平衡
    """
    returns_panel: (日期 × 股票) 的收益率矩阵
    factor_panel: {因子名: (日期 × 股票) DataFrame}
    """
    monthly_returns = returns_panel.resample(rebalance_freq).apply(
        lambda x: (1 + x).prod() - 1
    )
    
    portfolio_returns = []
    
    for i, date in enumerate(monthly_returns.index[:-1]):
        # 当月因子值 → 预测下月收益
        month_factors = {}
        for name, panel in factor_panel.items():
            if date in panel.index:
                month_factors[name] = panel.loc[date]
        
        if not month_factors:
            continue
        
        # 合成因子
        score = composite_factor(month_factors)
        
        # 选 Top N
        top_stocks = score.nlargest(n_stocks).index.tolist()
        
        # 下月等权持有
        next_date = monthly_returns.index[i + 1]
        next_returns = monthly_returns.loc[next_date, top_stocks]
        port_return = next_returns.mean()
        
        portfolio_returns.append({'date': next_date, 'return': port_return})
    
    return pd.DataFrame(portfolio_returns).set_index('date')
```

---

## 4.4 策略评估指标

```python
def evaluate_strategy(port_returns: pd.Series, 
                       bench_returns: pd.Series) -> dict:
    """量化策略综合评估"""
    
    # 超额收益
    excess = port_returns - bench_returns
    
    # 累积收益
    cum_port = (1 + port_returns).cumprod()
    cum_bench = (1 + bench_returns).cumprod()
    
    # 最大回撤
    def max_drawdown(cum_ret):
        rolling_max = cum_ret.cummax()
        return ((cum_ret - rolling_max) / rolling_max).min()
    
    # 信息比率（IR）
    ir = excess.mean() / excess.std() * np.sqrt(12)  # 月频年化
    
    return {
        '年化超额收益': f'{excess.mean()*12:.2%}',
        '年化超额波动': f'{excess.std()*np.sqrt(12):.2%}',
        '信息比率(IR)': f'{ir:.3f}',
        '组合最大回撤': f'{max_drawdown(cum_port):.2%}',
        '基准最大回撤': f'{max_drawdown(cum_bench):.2%}',
        '胜率': f'{(port_returns > bench_returns).mean():.2%}',
        '最终组合净值': f'{cum_port.iloc[-1]:.3f}x',
        '最终基准净值': f'{cum_bench.iloc[-1]:.3f}x',
    }
```

---

## 延伸阅读

- [WhaleQuant 完整教程](https://github.com/datawhalechina/whale-quant)
- 《主动投资组合管理》（Grinold & Kahn）
- Barra USE4 Factor Model 文档
