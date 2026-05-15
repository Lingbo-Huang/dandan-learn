---
layout: default
title: "D5 · 做市策略入门"
render_with_liquid: false
---

# D5 · 做市策略入门

> 做市商是市场流动性的提供者。理解做市策略，是理解价差和市场微结构的钥匙。

---

## 1. 做市商的盈利模式

**做市（Market Making）**：同时在买方和卖方挂单，赚取 Bid-Ask Spread。

```
做市商 以 99.95 买入，以 100.05 卖出
Spread = 0.10 元
每手成交 → 赚 0.10 元（理论）
```

**做市商的三大风险**：

| 风险 | 说明 |
|------|------|
| 库存风险 | 持仓方向错误，价格不利移动 |
| 逆选择风险 | 知情交易者利用信息优势"薅"做市商 |
| 流动性风险 | 市场恐慌时无法平仓 |

---

## 2. Avellaneda-Stoikov 做市模型

经典做市模型，考虑库存风险的最优报价：

$$
p^* = s - q \cdot \gamma \cdot \sigma^2 \cdot (T - t)
$$

$$
\text{spread} = \gamma \sigma^2 (T-t) + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right)
$$

- s：中间价
- q：当前持仓（库存）
- γ：风险厌恶系数
- σ：波动率
- κ：订单到达强度

```python
import numpy as np
import pandas as pd

class AvellanedaStoikov:
    """
    Avellaneda-Stoikov 做市模型
    """
    
    def __init__(self, gamma=0.1, kappa=1.5, sigma=0.02, T=1.0):
        """
        gamma: 风险厌恶（越大，越积极调整报价对冲库存）
        kappa: 订单到达强度
        sigma: 价格波动率
        T: 结束时间（归一化为1）
        """
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self.T = T
    
    def optimal_spread(self, t):
        """最优价差"""
        remaining = self.T - t
        term1 = self.gamma * self.sigma**2 * remaining
        term2 = (2 / self.gamma) * np.log(1 + self.gamma / self.kappa)
        return term1 + term2
    
    def optimal_quotes(self, mid_price, inventory, t):
        """
        计算最优买卖报价
        inventory: 当前库存（正数=多头，负数=空头）
        """
        remaining = self.T - t
        
        # 库存调整后的"储备价格"
        reservation_price = mid_price - inventory * self.gamma * self.sigma**2 * remaining
        
        # 最优价差
        spread = self.optimal_spread(t)
        
        bid = reservation_price - spread / 2
        ask = reservation_price + spread / 2
        
        return bid, ask
    
    def simulate(self, price_series, n_steps=1000):
        """
        模拟做市商的运营
        """
        inventory = 0
        cash = 0
        pnl_history = []
        
        dt = self.T / n_steps
        
        for i in range(len(price_series) - 1):
            t = i * dt
            mid = price_series.iloc[i]
            
            bid, ask = self.optimal_quotes(mid, inventory, t)
            
            # 模拟订单到达（泊松过程）
            bid_arrival = np.random.poisson(self.kappa * dt)
            ask_arrival = np.random.poisson(self.kappa * dt)
            
            # 买单到达（做市商卖出）
            if ask_arrival > 0 and price_series.iloc[i+1] >= ask:
                cash += ask * ask_arrival
                inventory -= ask_arrival
            
            # 卖单到达（做市商买入）
            if bid_arrival > 0 and price_series.iloc[i+1] <= bid:
                cash -= bid * bid_arrival
                inventory += bid_arrival
            
            # 当前盈亏 = 现金 + 库存市值
            total_pnl = cash + inventory * mid
            pnl_history.append({
                'step': i,
                'mid': mid,
                'bid': bid,
                'ask': ask,
                'inventory': inventory,
                'pnl': total_pnl
            })
        
        return pd.DataFrame(pnl_history)
```

---

## 3. 简单做市策略实现

```python
class SimpleMM:
    """
    简化做市策略（适合程序化交易学习）
    """
    
    def __init__(self, spread_bps=10, max_inventory=1000, 
                 inventory_skew=True):
        self.spread_bps = spread_bps  # 价差（基点）
        self.max_inventory = max_inventory
        self.inventory_skew = inventory_skew
        self.inventory = 0
    
    def get_quotes(self, mid_price):
        """生成买卖报价"""
        base_spread = mid_price * self.spread_bps / 10000
        
        if self.inventory_skew:
            # 库存偏移：持多仓时卖得便宜，持空仓时买得便宜
            skew = -self.inventory / self.max_inventory * base_spread * 0.5
        else:
            skew = 0
        
        bid = mid_price - base_spread / 2 + skew
        ask = mid_price + base_spread / 2 + skew
        
        return round(bid, 2), round(ask, 2)
    
    def on_trade(self, side, price, quantity):
        """交易回报处理"""
        if side == 'buy':   # 买方成交 -> 我们卖出
            self.inventory -= quantity
        else:               # 卖方成交 -> 我们买入
            self.inventory += quantity
        
        # 库存过大时，主动平仓
        if abs(self.inventory) > self.max_inventory:
            return 'reduce_inventory'
        return 'continue'
```

---

## 4. 做市商面对的实际挑战

### 4.1 Toxic Flow（毒性订单流）

知情交易者识别：
```python
def detect_toxic_flow(trades_df, threshold=0.7):
    """
    通过价格走势检测毒性订单流
    成交后价格持续单向运动 = 可能是知情交易者
    """
    # 成交后 N 分钟的价格变化
    post_trade_move = trades_df['price'].shift(-10) / trades_df['price'] - 1
    
    # 买单后价格上涨 / 卖单后价格下跌 = 正常
    # 买单后价格下跌 = 可能是噪声交易（对做市商有利）
    # 买单后价格大幅上涨 = 可能是知情交易（对做市商不利）
    
    buy_trades = trades_df[trades_df['side'] == 'buy']
    toxicity = (post_trade_move.reindex(buy_trades.index) > 0.005).mean()
    
    return toxicity > threshold
```

### 4.2 A股期权做市

A股 ETF 期权市场有做市商制度，做市商职责：
- 维持最大买卖价差（不超过规定上限）
- 保持最低挂单深度
- 满足最低响应率

---

## 小结

| 维度 | 内容 |
|------|------|
| 盈利来源 | Bid-Ask Spread |
| 主要风险 | 库存风险、逆选择风险 |
| 经典模型 | Avellaneda-Stoikov |
| 核心技术 | 库存调整报价、毒性流检测 |
| A股现状 | 期权、科创板有做市制度 |
