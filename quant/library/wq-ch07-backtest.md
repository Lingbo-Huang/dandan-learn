---
layout: default
title: "WhaleQuant Ch07 · 量化回测"
source: "https://github.com/datawhalechina/whale-quant"
---

# 07 · 量化回测

> **来源**：[WhaleQuant](https://github.com/datawhalechina/whale-quant) · datawhalechina

---

## 7.1 回测的本质与陷阱

**回测（Backtesting）**：用历史数据模拟策略的过去表现，评估其有效性。

### 常见陷阱

| 陷阱 | 描述 | 避免方法 |
|------|------|---------|
| **前视偏差** | 使用了未来才能获得的数据 | 严格确认每个数据的可获取时间 |
| **幸存者偏差** | 只用了现存股票，忽略退市股 | 使用包含退市股的完整股票池 |
| **数据挖掘偏差** | 过度拟合历史数据 | Walk-forward 验证，样本外检验 |
| **流动性假设** | 假设可以按任意价格成交 | 加入冲击成本模型，限制持仓规模 |
| **交易成本忽视** | 不考虑手续费、印花税 | 在收益中扣除实际交易成本 |

---

## 7.2 向量化回测

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class VectorBacktest:
    """
    向量化回测引擎（快速，适合策略研究）
    不考虑日内执行细节，以收盘价成交
    """
    
    def __init__(self, 
                  prices: pd.DataFrame,    # 价格矩阵（日期×股票）
                  signals: pd.DataFrame,   # 信号矩阵（日期×股票），值为目标权重
                  commission: float = 0.001,
                  stamp_tax: float = 0.001):
        self.prices = prices
        self.signals = signals
        self.commission = commission
        self.stamp_tax = stamp_tax
    
    def run(self) -> pd.DataFrame:
        # 日收益率
        returns = self.prices.pct_change()
        
        # 持仓权重（信号滞后一天，避免前视偏差）
        weights = self.signals.shift(1)
        
        # 权重归一化（确保多头权重和为1）
        weights = weights.div(weights.abs().sum(axis=1), axis=0).fillna(0)
        
        # 组合毛收益
        gross_returns = (returns * weights).sum(axis=1)
        
        # 计算换手率
        weight_diff = weights.diff().abs()
        turnover = weight_diff.sum(axis=1)
        
        # 扣除交易成本
        # 假设买入和卖出各占换手率的一半
        buy_turnover = turnover / 2
        sell_turnover = turnover / 2
        cost = buy_turnover * self.commission + sell_turnover * (self.commission + self.stamp_tax)
        
        net_returns = gross_returns - cost
        
        # 汇总结果
        result = pd.DataFrame({
            'gross_return': gross_returns,
            'net_return': net_returns,
            'turnover': turnover,
            'cost': cost,
        })
        result['gross_nav'] = (1 + gross_returns).cumprod()
        result['net_nav'] = (1 + net_returns).cumprod()
        
        return result
    
    def report(self, result: pd.DataFrame, benchmark_returns: pd.Series = None):
        """打印回测报告"""
        r = result['net_return']
        
        ann_return = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        nav = result['net_nav']
        max_dd = ((nav - nav.cummax()) / nav.cummax()).min()
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
        
        avg_turnover = result['turnover'].mean() * 252  # 年化换手率
        
        print("=" * 50)
        print("回测报告")
        print("=" * 50)
        print(f"回测区间: {r.index[0].date()} → {r.index[-1].date()}")
        print(f"年化收益: {ann_return:.2%}")
        print(f"年化波动: {ann_vol:.2%}")
        print(f"夏普比率: {sharpe:.3f}")
        print(f"最大回撤: {max_dd:.2%}")
        print(f"Calmar比率: {calmar:.3f}")
        print(f"年化换手率: {avg_turnover:.1f}x")
        print(f"最终净值: {nav.iloc[-1]:.3f}x")
        
        if benchmark_returns is not None:
            excess = r - benchmark_returns
            ir = excess.mean() / excess.std() * np.sqrt(252)
            win_rate = (r > benchmark_returns).mean()
            print(f"信息比率: {ir:.3f}")
            print(f"月度胜率: {win_rate:.2%}")
```

---

## 7.3 Walk-Forward 验证

```python
def walk_forward_test(data: pd.DataFrame,
                       strategy_func,
                       train_window: int = 252,    # 训练窗口（1年）
                       test_window: int = 63,      # 测试窗口（1季度）
                       step: int = 21) -> pd.DataFrame:
    """
    Walk-Forward 分析：防止过拟合的核心工具
    
    时间轴：
    |--训练1--|测试1|
         |--训练2--|测试2|
              |--训练3--|测试3|
    """
    all_test_returns = []
    
    dates = data.index
    n = len(dates)
    
    for start_idx in range(0, n - train_window - test_window, step):
        train_end = start_idx + train_window
        test_end = train_end + test_window
        
        train_data = data.iloc[start_idx:train_end]
        test_data = data.iloc[train_end:test_end]
        
        # 在训练集上优化参数
        best_params = strategy_func.optimize(train_data)
        
        # 在测试集上评估
        test_returns = strategy_func.backtest(test_data, best_params)
        all_test_returns.append(test_returns)
    
    return pd.concat(all_test_returns)

# 样本内 vs 样本外衰减分析
def is_oos_analysis(returns_is: pd.Series, returns_oos: pd.Series):
    """分析样本内外的性能衰减"""
    sharpe_is = returns_is.mean() / returns_is.std() * np.sqrt(252)
    sharpe_oos = returns_oos.mean() / returns_oos.std() * np.sqrt(252)
    
    decay = (sharpe_is - sharpe_oos) / sharpe_is if sharpe_is > 0 else None
    
    print(f"样本内夏普: {sharpe_is:.3f}")
    print(f"样本外夏普: {sharpe_oos:.3f}")
    if decay:
        print(f"衰减率: {decay:.1%}")
        if decay > 0.5:
            print("⚠️ 衰减超过50%，可能存在过拟合！")
        else:
            print("✅ 衰减在合理范围内")
```

---

## 7.4 Backtrader 框架

```python
import backtrader as bt

class DualMAStrategy(bt.Strategy):
    """双均线策略（Backtrader实现）"""
    
    params = (
        ('short_period', 20),
        ('long_period', 60),
    )
    
    def __init__(self):
        self.ma_short = bt.ind.SMA(self.data.close, period=self.p.short_period)
        self.ma_long = bt.ind.SMA(self.data.close, period=self.p.long_period)
        self.crossover = bt.ind.CrossOver(self.ma_short, self.ma_long)
    
    def next(self):
        if self.crossover > 0:   # 金叉
            if not self.position:
                self.buy()
        elif self.crossover < 0:  # 死叉
            if self.position:
                self.sell()
    
    def notify_trade(self, trade):
        if trade.isclosed:
            print(f'交易完成 | 毛利润: {trade.pnl:.2f} | 净利润: {trade.pnlcomm:.2f}')

# 运行回测
def run_backtrader(data_path: str, 
                    strategy_class,
                    initial_cash: float = 100_000,
                    commission: float = 0.001):
    
    cerebro = bt.Cerebro()
    
    # 加载数据
    data = bt.feeds.PandasData(dataname=pd.read_csv(data_path, index_col='date', parse_dates=True))
    cerebro.adddata(data)
    
    # 添加策略
    cerebro.addstrategy(strategy_class)
    
    # 设置资金和手续费
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    results = cerebro.run()
    
    strat = results[0]
    print(f"最终资金: {cerebro.broker.getvalue():.2f}")
    print(f"夏普比率: {strat.analyzers.sharpe.get_analysis()['sharperatio']:.3f}")
    print(f"最大回撤: {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
    
    cerebro.plot()
    return results
```

---

## 延伸阅读

- [WhaleQuant 完整教程](https://github.com/datawhalechina/whale-quant)
- [Backtrader 文档](https://www.backtrader.com/)
- Bailey et al. (2014) - "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality"
