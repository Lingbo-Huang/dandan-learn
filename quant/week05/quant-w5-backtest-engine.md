---
layout: default
title: "D6 · 回测引擎设计"
render_with_liquid: false
---

# D6 · 回测引擎设计

> 用好现成的回测框架（Backtrader/Zipline/Qlib），还要理解它们在黑箱里做了什么。

---

## 1. 回测引擎的核心组件

一个完整的回测系统包含：

```
数据层 → 信号层 → 组合层 → 执行层 → 风控层 → 报告层
```

| 组件 | 职责 |
|------|------|
| 数据层 | 清洗、对齐、复权行情 + 财务数据 |
| 信号层 | 因子计算、模型预测 |
| 组合层 | 选股 + 权重优化 |
| 执行层 | 模拟成交（滑点、冲击成本）|
| 风控层 | 最大仓位、止损、行业限制 |
| 报告层 | 绩效计算、可视化 |

---

## 2. 简单回测引擎实现

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Order:
    date: pd.Timestamp
    stock: str
    target_weight: float  # 目标权重（-1到1）

class SimpleBacktester:
    """
    简化版回测引擎
    功能：基于权重的日度回测，包含交易成本
    """
    
    def __init__(self, 
                 initial_capital=1e7,
                 transaction_cost=0.001,  # 单边
                 slippage=0.001):          # 滑点
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
    
    def run(self, weights_df, returns_df):
        """
        weights_df: DataFrame，index=rebalance日期，columns=股票代码，value=目标权重
        returns_df: DataFrame，index=日期，columns=股票，value=日度收益率
        """
        # 对齐日期
        all_dates = returns_df.index
        
        # 初始化
        portfolio_value = self.initial_capital
        current_weights = pd.Series(dtype=float)
        
        daily_pnl = []
        
        for i, date in enumerate(all_dates):
            # 检查是否是调仓日
            if date in weights_df.index:
                new_weights = weights_df.loc[date].dropna()
                new_weights = new_weights / new_weights.abs().sum()  # 归一化
                
                # 计算换手率和交易成本
                all_stocks = current_weights.index.union(new_weights.index)
                w_prev = current_weights.reindex(all_stocks).fillna(0)
                w_curr = new_weights.reindex(all_stocks).fillna(0)
                
                turnover = (w_curr - w_prev).abs().sum() / 2
                cost = turnover * (self.transaction_cost + self.slippage) * 2
                
                current_weights = w_curr
                portfolio_value *= (1 - cost)
            
            # 今日收益
            if len(current_weights) > 0:
                day_ret = returns_df.loc[date].reindex(current_weights.index).fillna(0)
                port_ret = (current_weights * day_ret).sum()
                portfolio_value *= (1 + port_ret)
                
                # 更新权重（价格变动导致权重漂移）
                new_vals = current_weights * (1 + day_ret)
                total = new_vals.sum()
                if total > 0:
                    current_weights = new_vals / total
            
            daily_pnl.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'return': port_ret if len(current_weights) > 0 else 0
            })
        
        result = pd.DataFrame(daily_pnl).set_index('date')
        return result
```

---

## 3. 回测常见偏差

### 3.1 未来函数（Look-Ahead Bias）

最致命的错误：使用了"未来"的数据。

```python
# 错误示例：使用当天收盘价决定当天交易
wrong_signal = factor_df.loc[today]  # 收盘后才有的数据
execute_order(today, wrong_signal)    # 当天就交易了 -> 未来函数！

# 正确方式：收盘后计算信号，次日开盘执行
correct_signal = factor_df.loc[today]
execute_order(next_trading_day, correct_signal)
```

### 3.2 幸存者偏差

```python
# 错误：只用当前还在市场上的股票做回测
current_stocks = get_current_constituents()  # 只有现在还活着的
backtest_on(current_stocks)  # 过去退市的（差公司）被排除了 -> 偏差！

# 正确：使用历史全量股票池（含退市）
all_historical_stocks = get_all_stocks_with_delisted()
backtest_on(all_historical_stocks)
```

### 3.3 复权价格处理

```python
def prepare_returns(close_adj, close_raw=None):
    """
    使用前复权价格计算收益
    避免除权日产生的虚假价格跳跃
    """
    # 前复权价格已经调整了分红、送配
    returns = close_adj.pct_change()
    
    # 过滤异常值（>50% 单日涨跌可能是复权错误）
    returns = returns.clip(-0.5, 0.5)
    return returns
```

---

## 4. Qlib 框架简介

Qlib 是微软开源的量化研究框架，适合 ML 因子挖掘：

```python
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config

# 初始化（A股数据）
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

# 定义因子表达式
from qlib.data import D

# Alpha158 因子之一：RESI5
# 过去5天收益对市场收益的残差
expression = "Resi($close/Ref($close,1)-1, Ref(($close/Ref($close,1)-1), 1), 5)"

# 获取因子数据
data = D.features(
    instruments='csi300',
    fields=[expression],
    start_time='2020-01-01',
    end_time='2023-12-31'
)
```

---

## 5. 回测基础设施检查清单

在使用回测结果前，先确认：

- [ ] 数据是否有未来函数？（信号和执行时间是否正确分离）
- [ ] 财务数据是否使用了真实披露时间？（不用期末日期，用实际公告日期）
- [ ] 股票池是否包含了历史退市股票？
- [ ] 收益率计算是否使用了复权价？
- [ ] 停牌、涨跌停股票是否正确处理？（无法成交）
- [ ] 交易成本是否合理估计？（含冲击成本）

---

## 小结

| 组件 | 核心要点 |
|------|---------|
| 数据层 | 复权价 + 历史全量股票（含退市）|
| 信号层 | 收盘后计算，次日执行 |
| 执行层 | 合理估计冲击成本 + 滑点 |
| 风控层 | 最大仓位 + 止损 |
| 常见坑 | 未来函数、幸存者偏差、复权错误 |
