---
layout: default
title: "WhaleQuant Ch06 · 量化调仓策略"
source: "https://github.com/datawhalechina/whale-quant"
---

# 06 · 量化调仓策略

> **来源**：[WhaleQuant](https://github.com/datawhalechina/whale-quant) · datawhalechina

---

## 6.1 调仓频率的权衡

| 频率 | 优点 | 缺点 |
|------|------|------|
| **高频（日/周）** | 快速响应因子变化 | 交易成本高，换手率大 |
| **低频（季度/年）** | 交易成本低 | 对信号响应慢 |
| **月度** | 常见的平衡点 | 需要根据策略自定义 |

---

## 6.2 权重优化方法

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 1. 等权重（最简单）
def equal_weight(n_stocks: int) -> np.ndarray:
    return np.ones(n_stocks) / n_stocks

# 2. 市值加权（被动策略基础）
def market_cap_weight(market_caps: pd.Series) -> np.ndarray:
    return (market_caps / market_caps.sum()).values

# 3. 风险平价（各资产贡献等量风险）
def risk_parity(cov_matrix: np.ndarray) -> np.ndarray:
    n = len(cov_matrix)
    
    def portfolio_vol(weights):
        return np.sqrt(weights @ cov_matrix @ weights)
    
    def risk_contribution(weights):
        vol = portfolio_vol(weights)
        marginal_risk = cov_matrix @ weights
        return weights * marginal_risk / vol
    
    def objective(weights):
        rc = risk_contribution(weights)
        target = np.ones(n) / n  # 等风险贡献
        return np.sum((rc - target * np.sum(rc))**2)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.001, 0.5)] * n  # 每只股票 0.1%-50%
    x0 = np.ones(n) / n
    
    result = minimize(objective, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    return result.x

# 4. 均值方差优化（最大夏普比率）
def max_sharpe_portfolio(expected_returns: np.ndarray,
                          cov_matrix: np.ndarray,
                          risk_free_rate: float = 0.02/252) -> np.ndarray:
    n = len(expected_returns)
    
    def neg_sharpe(weights):
        ret = weights @ expected_returns
        vol = np.sqrt(weights @ cov_matrix @ weights)
        return -(ret - risk_free_rate) / vol
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 0.3)] * n  # 单只股票上限30%
    x0 = np.ones(n) / n
    
    result = minimize(neg_sharpe, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    return result.x

# 5. Black-Litterman 模型（结合观点和市场均衡）
def black_litterman(market_weights: np.ndarray,
                     cov_matrix: np.ndarray,
                     views_matrix: np.ndarray,
                     views_returns: np.ndarray,
                     tau: float = 0.05) -> np.ndarray:
    """
    market_weights: 市值权重
    cov_matrix: 协方差矩阵
    views_matrix P: 观点矩阵（每行代表一个观点）
    views_returns Q: 观点收益
    tau: 先验不确定性参数
    """
    # 均衡收益（CAPM均衡）
    risk_aversion = 2.5
    pi = risk_aversion * cov_matrix @ market_weights
    
    # 观点不确定性矩阵
    omega = np.diag(np.diag(views_matrix @ (tau * cov_matrix) @ views_matrix.T))
    
    # Black-Litterman 期望收益
    M = np.linalg.inv(
        np.linalg.inv(tau * cov_matrix) + 
        views_matrix.T @ np.linalg.inv(omega) @ views_matrix
    )
    
    bl_returns = M @ (
        np.linalg.inv(tau * cov_matrix) @ pi + 
        views_matrix.T @ np.linalg.inv(omega) @ views_returns
    )
    
    return bl_returns
```

---

## 6.3 交易成本模型

```python
class TransactionCostModel:
    """交易成本模型"""
    
    def __init__(self, 
                 commission_rate: float = 0.0003,   # 佣金率
                 stamp_tax: float = 0.001,           # 印花税（卖出）
                 market_impact: float = 0.001):      # 市场冲击成本
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.market_impact = market_impact
    
    def buy_cost(self, amount: float) -> float:
        """买入成本"""
        return amount * (self.commission_rate + self.market_impact)
    
    def sell_cost(self, amount: float) -> float:
        """卖出成本"""
        return amount * (self.commission_rate + self.stamp_tax + self.market_impact)
    
    def round_trip_cost(self, amount: float) -> float:
        """双边交易成本"""
        return self.buy_cost(amount) + self.sell_cost(amount)
    
    def portfolio_rebalance_cost(self, 
                                  old_weights: np.ndarray,
                                  new_weights: np.ndarray,
                                  portfolio_value: float) -> float:
        """组合再平衡的交易成本"""
        weight_diff = new_weights - old_weights
        
        buy_amount = np.sum(weight_diff[weight_diff > 0]) * portfolio_value
        sell_amount = np.sum(np.abs(weight_diff[weight_diff < 0])) * portfolio_value
        
        return self.buy_cost(buy_amount) + self.sell_cost(sell_amount)
    
    def turnover_adjusted_return(self, 
                                   gross_return: float,
                                   turnover_rate: float) -> float:
        """调整交易成本后的净收益"""
        cost = turnover_rate * (self.commission_rate * 2 + self.stamp_tax)
        return gross_return - cost

# 使用示例
cost_model = TransactionCostModel()

# 判断换手是否合算
gross_alpha = 0.005  # 月度超额收益 0.5%
turnover = 0.3       # 月度换手率 30%

net_alpha = cost_model.turnover_adjusted_return(gross_alpha, turnover)
print(f"毛Alpha: {gross_alpha:.2%}")
print(f"净Alpha（扣成本）: {net_alpha:.2%}")
print(f"是否合算: {'是' if net_alpha > 0 else '否，需降低换手率'}")
```

---

## 6.4 再平衡触发条件

```python
def should_rebalance(current_weights: np.ndarray,
                      target_weights: np.ndarray,
                      threshold: float = 0.05) -> bool:
    """
    触发再平衡的条件（漂移阈值法）
    当实际权重偏离目标权重超过 threshold 时再平衡
    """
    max_drift = np.max(np.abs(current_weights - target_weights))
    return max_drift > threshold

# 结合成本的最优再平衡频率
def rebalance_benefit(factor_decay_half_life: int,
                       monthly_gross_alpha: float,
                       turnover_cost: float) -> pd.DataFrame:
    """
    分析不同再平衡频率下的净Alpha
    factor_decay_half_life: 因子信号衰减半衰期（天）
    """
    results = []
    
    for freq_days in [5, 10, 20, 60, 120]:
        # 信号衰减损失（越长衰减越多）
        signal_retention = 0.5 ** (freq_days / factor_decay_half_life)
        effective_alpha = monthly_gross_alpha * signal_retention * (freq_days / 20)
        
        # 年化换手次数
        annual_rebalances = 252 / freq_days
        # 年化交易成本
        annual_cost = turnover_cost * annual_rebalances
        
        results.append({
            '再平衡频率（天）': freq_days,
            '信号保留率': f'{signal_retention:.2%}',
            '年化毛Alpha': f'{effective_alpha*12:.2%}',
            '年化交易成本': f'{annual_cost:.2%}',
            '年化净Alpha': f'{effective_alpha*12 - annual_cost:.2%}',
        })
    
    return pd.DataFrame(results)
```

---

## 延伸阅读

- [WhaleQuant 完整教程](https://github.com/datawhalechina/whale-quant)
- Black & Litterman (1992) - "Global Portfolio Optimization"
- Qian et al. - "Quantitative Equity Portfolio Management"
