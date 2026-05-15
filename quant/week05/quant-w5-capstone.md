---
layout: default
title: "D7 · Week 5 综合实战：多策略组合"
render_with_liquid: false
---

# D7 · Week 5 综合实战：多策略组合

> 单一策略容量有限、风险集中。多策略组合是机构量化的标准形态。

---

## 本周核心知识回顾

| 策略 | 核心逻辑 | 适用环境 |
|------|---------|---------|
| CTA 趋势 | 动量跟踪 | 趋势明确的市场 |
| 统计套利 | 协整均值回归 | 震荡市、相关性稳定 |
| 截面动量 | 强者恒强 | 中长期 |
| 事件驱动 | 信息不对称 | 任何市场 |

---

## 多策略组合构建

```python
import pandas as pd
import numpy as np
from scipy.optimize import minimize

class MultiStrategyPortfolio:
    """
    多策略组合：将不同策略的收益流组合起来
    目标：最大化风险调整后收益
    """
    
    def __init__(self, strategies, weights=None):
        """
        strategies: {name: pd.Series of daily returns}
        weights: 初始权重（None则等权）
        """
        self.strategies = strategies
        self.returns_df = pd.DataFrame(strategies)
        self.weights = weights or {k: 1/len(strategies) for k in strategies}
    
    def calc_correlation(self):
        """分析策略间相关性"""
        return self.returns_df.corr()
    
    def optimize_weights(self, target_vol=0.15, risk_free=0.02):
        """
        均值-方差优化
        """
        mu = self.returns_df.mean() * 252
        sigma = self.returns_df.cov() * 252
        n = len(mu)
        
        def neg_sharpe(w):
            w = np.array(w)
            port_ret = w @ mu
            port_vol = np.sqrt(w @ sigma @ w)
            return -(port_ret - risk_free) / port_vol
        
        constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1}]
        bounds = [(0, 1)] * n
        
        result = minimize(neg_sharpe, x0=[1/n]*n,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        
        optimal_weights = pd.Series(result.x, index=mu.index)
        return optimal_weights
    
    def risk_parity_weights(self):
        """
        风险平价权重：让每个策略贡献相等的组合风险
        """
        sigma = self.returns_df.cov().values
        n = sigma.shape[0]
        
        def risk_parity_objective(w):
            w = np.array(w)
            port_vol = np.sqrt(w @ sigma @ w)
            # 各策略风险贡献
            marginal_risk = sigma @ w / port_vol
            risk_contribution = w * marginal_risk
            # 最小化各策略风险贡献的方差
            target = port_vol / n
            return np.sum((risk_contribution - target) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1}]
        bounds = [(0.01, 1)] * n
        
        result = minimize(risk_parity_objective, x0=[1/n]*n,
                         method='SLSQP', bounds=bounds,
                         constraints=constraints)
        
        return pd.Series(result.x, index=self.returns_df.columns)
    
    def calc_portfolio_performance(self, weights=None):
        """计算组合绩效"""
        if weights is None:
            weights = pd.Series(self.weights)
        
        port_ret = (self.returns_df * weights).sum(axis=1)
        
        annual_ret = port_ret.mean() * 252
        annual_vol = port_ret.std() * (252 ** 0.5)
        sharpe = annual_ret / annual_vol
        
        cum = (1 + port_ret).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
        
        return {
            '年化收益': f'{annual_ret:.2%}',
            '年化波动': f'{annual_vol:.2%}',
            '夏普比率': f'{sharpe:.2f}',
            '最大回撤': f'{max_dd:.2%}',
            'Calmar': f'{annual_ret / abs(max_dd):.2f}'
        }


# ===== 示例：各策略相关性分析 =====

def analyze_strategy_diversification(strategy_returns):
    """
    分析多策略组合的分散化效果
    """
    corr = strategy_returns.corr()
    print("策略间相关性矩阵：")
    print(corr.round(2))
    
    # 等权组合
    eq_weights = {k: 1/len(strategy_returns.columns) 
                  for k in strategy_returns.columns}
    port_ret = strategy_returns.mean(axis=1)
    
    # 分散化比率
    individual_vols = strategy_returns.std() * (252 ** 0.5)
    port_vol = port_ret.std() * (252 ** 0.5)
    avg_ind_vol = individual_vols.mean()
    
    diversification_ratio = avg_ind_vol / port_vol
    print(f"\n分散化比率: {diversification_ratio:.2f}")
    print("（>1 说明组合波动率低于单策略平均，分散有效）")
    
    return corr, diversification_ratio
```

---

## 各策略适用性矩阵

| 市场环境 | CTA | 统计套利 | 截面选股 | 事件驱动 |
|---------|-----|---------|---------|---------|
| 牛市单边上涨 | ✅ | ⚠️ | ✅ | ✅ |
| 熊市单边下跌 | ✅ (做空) | ⚠️ | ❌ | ✅ |
| 震荡市 | ❌ | ✅ | ⚠️ | ✅ |
| 高波动危机 | ✅ | ❌ | ❌ | ⚠️ |
| 低波动慢牛 | ⚠️ | ✅ | ✅ | ✅ |

---

## 面试核心考点回顾

1. **CTA 为什么在危机时期有正收益？** 危机时趋势明确（股票暴跌、债券上涨），顺势信号强。

2. **配对交易的核心风险？** 协整关系破裂（公司基本面变化）。

3. **截面动量在A股失效的原因？** 散户主导，短期过度反应后快速反转。

4. **回测最大的偏差是什么？** 未来函数（look-ahead bias）和幸存者偏差。

5. **多策略组合的分散化效果？** 只有低相关策略才能真正分散，同向因子叠加没有效果。

---

## 进阶方向

- **风险平价（Risk Parity）**：按风险而非资金分配，让低波动策略获得更高权重
- **策略容量估算**：信号 IC、持仓股票数、流动性共同决定策略上限
- **在线学习**：用实时数据动态更新策略参数，而非固定参数
