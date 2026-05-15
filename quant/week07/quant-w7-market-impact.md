---
layout: default
title: "D4 · 冲击成本与市场影响模型"
render_with_liquid: false
---

# D4 · 冲击成本与市场影响模型

> 你买得越多，价格就涨得越高。这不是巧合，而是有数学规律的。

---

## 1. 什么是市场冲击

**市场冲击（Market Impact）**：大额订单在执行过程中推动价格不利于自己方向移动的成本。

$$
\text{冲击成本} = \frac{P_{\text{执行均价}} - P_{\text{到达价}}}{P_{\text{到达价}}}
$$

- 买入时：执行均价 > 到达价（推高了价格）
- 卖出时：执行均价 < 到达价（压低了价格）

---

## 2. 主要市场影响模型

### 2.1 线性冲击模型（Kyle 1985）

$$
\Delta P = \lambda \cdot Q
$$

- Q：订单数量
- λ：市场深度的倒数（越小流动性越好）

```python
def kyle_impact(quantity, lambda_param):
    """Kyle 线性冲击模型"""
    return lambda_param * quantity

def estimate_kyle_lambda(price_changes, order_flow):
    """从历史数据估计 Kyle λ"""
    import statsmodels.api as sm
    X = sm.add_constant(order_flow.values.reshape(-1, 1))
    model = sm.OLS(price_changes.values, X).fit()
    return model.params[1]  # 斜率即为 λ
```

### 2.2 平方根冲击模型（Square Root Law）

最被广泛接受的模型，冲击成本与订单量的平方根成正比：

$$
\text{MI} = \sigma \cdot \eta \cdot \sqrt{\frac{Q}{V}}
$$

- σ：日波动率
- η：市场冲击系数（通常 0.1-1.0）
- Q：交易量
- V：日均成交量（ADV）

```python
import numpy as np

def sqrt_market_impact(quantity, adv, daily_vol, eta=0.5):
    """
    平方根市场冲击模型
    quantity: 需要交易的数量（股）
    adv: 日均成交量（Average Daily Volume）
    daily_vol: 日收益率波动率
    eta: 冲击系数（根据市场校准）
    """
    participation = quantity / adv
    impact = daily_vol * eta * np.sqrt(participation)
    return impact

# 示例
print(sqrt_market_impact(
    quantity=100000,    # 10万股
    adv=5000000,        # 日均成交500万股
    daily_vol=0.02,     # 2%日波动率
    eta=0.5
))
```

### 2.3 Almgren-Chriss 模型（最优执行）

```python
class AlmgrenChrissModel:
    """
    Almgren-Chriss (2001) 最优执行模型
    在市场冲击成本和时间风险（价格漂移不确定性）之间做权衡
    """
    
    def __init__(self, total_qty, T, sigma, gamma, eta, epsilon=0):
        """
        total_qty: 总需交易数量
        T: 执行时间窗口（天）
        sigma: 日波动率
        gamma: 永久冲击系数
        eta: 临时冲击系数
        epsilon: 固定交易成本
        """
        self.X = total_qty
        self.T = T
        self.sigma = sigma
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
    
    def optimal_trajectory(self, risk_aversion, n_steps=10):
        """
        计算最优执行路径
        risk_aversion: 风险厌恶系数（越大越急于执行）
        """
        tau = self.T / n_steps
        
        # Almgren-Chriss 最优解
        kappa_sq = (risk_aversion * self.sigma**2) / self.eta
        kappa = np.sqrt(kappa_sq) if kappa_sq > 0 else 0
        
        times = np.linspace(0, self.T, n_steps + 1)
        
        # 最优持仓轨迹
        if kappa > 0:
            holdings = self.X * np.sinh(kappa * (self.T - times)) / np.sinh(kappa * self.T)
        else:
            holdings = self.X * (1 - times / self.T)
        
        # 每期交易量
        trades = -np.diff(holdings)
        
        return pd.DataFrame({
            'time': times[:-1],
            'holdings': holdings[:-1],
            'trade_qty': trades
        })
    
    def expected_cost(self, risk_aversion, n_steps=10):
        """计算预期执行成本"""
        trajectory = self.optimal_trajectory(risk_aversion, n_steps)
        
        trades = trajectory['trade_qty'].values
        
        # 冲击成本（临时）
        temp_impact = self.eta / 2 * np.sum(trades**2) / (self.T / n_steps)
        
        # 永久冲击（价格漂移）
        perm_impact = self.gamma / 2 * self.X**2
        
        # 时间风险（不确定性）
        time_risk = risk_aversion * self.sigma**2 * np.sum(
            trajectory['holdings'].values**2
        ) * (self.T / n_steps)
        
        return {
            'temp_impact': temp_impact,
            'perm_impact': perm_impact,
            'time_risk': time_risk,
            'total': temp_impact + perm_impact + time_risk
        }
```

---

## 3. 实战：策略容量估算

```python
def estimate_strategy_capacity(factor_ic, annual_sharpe_target=1.0,
                                n_stocks=50, adv_million=100,
                                daily_vol=0.02, eta=0.5,
                                turnover_monthly=0.5):
    """
    估算策略最大容量
    
    思路：当冲击成本使净 IC 下降到无法达到目标夏普时，就是容量上限
    """
    results = []
    
    for aum_million in [1, 5, 10, 50, 100, 500, 1000]:
        # 每只股票的交易量（元）
        per_stock_trade = aum_million * 1e6 * turnover_monthly / n_stocks
        
        # 假设股价 50 元，转换为股数
        per_stock_qty = per_stock_trade / 50
        adv_qty = adv_million * 1e6 / 50
        
        # 冲击成本（单边）
        impact = sqrt_market_impact(
            quantity=per_stock_qty,
            adv=adv_qty,
            daily_vol=daily_vol,
            eta=eta
        )
        
        # 年化冲击成本（月换手一次，双边）
        annual_impact = impact * 2 * 12
        
        # 净 IC（简化：IC - 冲击成本影响）
        net_ic = factor_ic - annual_impact / (annual_sharpe_target * daily_vol * (252**0.5))
        
        results.append({
            'AUM(亿)': aum_million / 100,
            '单边冲击': f'{impact:.2%}',
            '年化冲击': f'{annual_impact:.2%}',
            '净IC': f'{max(net_ic, 0):.4f}',
            '可行': '✅' if net_ic > 0.02 else '❌'
        })
    
    return pd.DataFrame(results)
```

---

## 4. 冲击成本的实践规则

| 规模 | 占 ADV 比例 | 预期冲击 |
|------|-----------|---------|
| 小单 | < 1% | < 5bp |
| 中单 | 1-5% | 5-20bp |
| 大单 | 5-20% | 20-60bp |
| 超大单 | > 20% | > 60bp，需分天 |

**实践建议**：
- 单只股票单次不超过 ADV 的 10%
- 最好控制在 3-5%，冲击成本可控
- 超过 20% 需要多天拆分执行

---

## 小结

| 模型 | 假设 | 使用场景 |
|------|------|---------|
| Kyle 线性 | 冲击与量线性 | 理论分析 |
| 平方根模型 | 冲击与量的开方成比例 | 实战估算 |
| Almgren-Chriss | 最优化框架 | 大额订单执行优化 |

