---
layout: default
title: "D1 · 市场微观结构基础"
render_with_liquid: false
---

# D1 · 市场微观结构基础

> 从宏观因子到实际成交，中间隔着"市场微观结构"。理解它，才能理解真实的交易成本。

---

## 1. 市场微观结构研究什么

**市场微观结构（Market Microstructure）**：研究价格发现过程、交易机制、信息不对称如何影响资产价格形成。

核心问题：
- 价格是如何被发现（形成）的？
- 订单流包含什么信息？
- 交易成本的来源是什么？
- 大额订单如何影响市场价格？

---

## 2. 价差（Spread）的分解

**Bid-Ask Spread** = 买价 - 卖价，是做市商的"报价利润"。

经济学分解：

$$
Spread = \\underbrace{\\text{逆选择成分}}_{\\text{信息不对称}} + \\underbrace{\\text{库存成本}}_{\\text{做市商风险}} + \\underbrace{\\text{订单处理成本}}_{\\text{固定运营成本}}
$$

### 逆选择（Adverse Selection）

知情交易者（内幕）总在正确的方向交易，做市商被"逆选"：

```
知情交易者买入 → 价格确实上涨 → 做市商亏了卖便宜了
知情交易者卖出 → 价格确实下跌 → 做市商亏了买贵了
```

做市商通过扩大价差来补偿逆选择成本。

### 实证估算

```python
def calc_bid_ask_spread(ask_prices, bid_prices, method='relative'):
    """
    计算买卖价差
    method: 'relative'=相对价差, 'absolute'=绝对价差
    """
    if method == 'absolute':
        spread = ask_prices - bid_prices
    else:
        mid = (ask_prices + bid_prices) / 2
        spread = (ask_prices - bid_prices) / mid
    return spread

def roll_spread_estimator(close_prices):
    """
    Roll (1984) 价差估算器
    从收盘价序列估算隐含价差，不需要订单簿数据
    """
    price_changes = close_prices.diff()
    # Cov(delta_p_t, delta_p_{t-1})
    cov = price_changes.cov(price_changes.shift(1))
    
    if cov >= 0:
        return 0  # 无法估算时返回0
    
    # Roll 估算: spread = 2 * sqrt(-Cov)
    return 2 * np.sqrt(-cov)
```

---

## 3. 价格发现机制

### 3.1 连续竞价（大多数股票市场）

- 买卖双方持续报价，价格高的买单和价格低的卖单成交
- 价格实时反映供需变化
- A股、美股均采用此机制

### 3.2 集合竞价（A股开盘收盘）

- 所有订单汇集，以成交量最大的价格统一成交
- A股开盘 9:25 集合竞价，收盘 14:57-15:00 集合竞价
- 重要信号：**集合竞价成交量和价格**

```python
def analyze_opening_auction(auction_data):
    """
    分析集合竞价信号
    auction_data: 竞价期间的订单数据
    """
    # 9:25 最终成交价相对前收的偏离
    gap = (auction_data['open_price'] - auction_data['prev_close']) / auction_data['prev_close']
    
    # 集合竞价成交量占全天的比例（通常 3-8%）
    vol_ratio = auction_data['auction_volume'] / auction_data['total_volume']
    
    return gap, vol_ratio
```

---

## 4. 知情交易者与噪声交易者

| 类型 | 动机 | 对价格的影响 |
|------|------|-----------|
| 知情交易者 | 私有信息 | 推动价格向"真实价值"收敛 |
| 噪声交易者 | 流动性需求/情绪 | 造成价格偏离，为做市商提供利润 |
| 做市商 | 赚取价差 | 提供流动性，稳定价格 |

**Kyle (1985) 模型**：

市场深度 λ 表示每单位订单流引起的价格变动：

$$
\\Delta p = \\lambda \\cdot q
$$

λ 越小，市场流动性越好（冲击越小）。

---

## 5. A股市场微结构特殊性

| 特征 | A股 | 美股 |
|------|-----|------|
| 涨跌停限制 | ±10%（科创板 ±20%）| 无（有熔断）|
| 交割制度 | T+1（股票）T+0（ETF）| T+2 |
| 做市制度 | 部分做市（科创板/期权）| 广泛做市 |
| 散户占比 | ~70% 交易量 | ~15% |

**A股特殊现象**：
- **打板策略**：追涨停板，利用涨停限制形成的"封板"效应
- **开盘集合竞价博弈**：机构在集合竞价期间测试方向
- **尾盘效应**：收盘前 30 分钟成交量放大，价格反转增加

---

## 6. 微观结构与策略的关系

```python
def estimate_strategy_capacity(daily_volume, ic, target_sharpe=1.0,
                               participation_rate=0.1):
    """
    基于市场微结构估算策略容量
    daily_volume: 股票日均成交额（元）
    ic: 因子 IC
    participation_rate: 目标参与率（通常不超过 ADV 的 10%）
    """
    # 可用流动性
    available_liquidity = daily_volume * participation_rate
    
    # 简化的容量估算
    # 真实的容量分析需要考虑完整的冲击成本模型
    estimated_capacity = available_liquidity * ic * (252 ** 0.5)
    
    return estimated_capacity
```

---

## 小结

| 概念 | 说明 |
|------|------|
| Bid-Ask Spread | 逆选择 + 库存成本 + 处理成本 |
| 价格发现 | 连续竞价 vs. 集合竞价 |
| Kyle λ | 市场深度，越小流动性越好 |
| A股特殊 | 涨跌停、T+1、散户主导 |
| 策略意义 | 容量、冲击成本、执行设计 |
