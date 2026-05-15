---
layout: default
title: "D1 · CTA 趋势跟踪策略"
render_with_liquid: false
---

# D1 · CTA 趋势跟踪策略

> "趋势是你的朋友。" CTA 是量化中最成熟的策略之一，在危机时期往往正收益。

---

## 1. CTA 的核心逻辑

**CTA（Commodity Trading Advisor）趋势跟踪**：
- 市场价格在宏观冲击（战争、货币政策、大宗商品供需）下形成持续趋势
- 通过移动平均等技术信号跟踪趋势，顺势交易
- 多资产、多周期分散，不依赖单一市场

**为什么趋势存在？**
- 央行政策改变缓慢，形成利率趋势
- 基本面信息扩散需要时间，价格渐进调整
- 投资者行为偏差（锚定、处置效应）延缓价格到位

---

## 2. 主要趋势信号

### 2.1 时序动量（TSMOM）

```python
def tsmom_signal(returns, lookback=252):
    """
    时序动量信号
    正过去收益 -> 做多，负过去收益 -> 做空
    """
    past_ret = returns.rolling(lookback).sum()
    signal = past_ret.apply(lambda x: 1 if x > 0 else -1)
    return signal
```

### 2.2 移动平均交叉（MACD 类）

```python
def ma_crossover_signal(prices, fast=20, slow=100):
    """
    快线上穿慢线 -> 买入；下穿 -> 卖出
    """
    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()
    signal = (ma_fast > ma_slow).astype(int) * 2 - 1  # +1 or -1
    return signal
```

### 2.3 布林带信号

```python
def bollinger_signal(prices, window=20, n_std=2):
    """
    突破上轨 -> 做多趋势信号
    突破下轨 -> 做空趋势信号
    """
    ma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    
    signal = pd.Series(0, index=prices.index)
    signal[prices > upper] = 1
    signal[prices < lower] = -1
    return signal
```

---

## 3. 仓位管理：波动率目标

CTA 的核心之一是**波动率目标化仓位**，让组合波动率保持稳定：

```python
def vol_target_position(signal, returns, target_vol=0.2, lookback=60):
    """
    波动率目标化仓位
    target_vol: 目标年化波动率（如 20%）
    """
    realized_vol = returns.rolling(lookback).std() * (252 ** 0.5)
    # 仓位 = 目标波动率 / 实现波动率 × 方向
    position = signal * (target_vol / (realized_vol + 1e-8))
    # 限制最大仓位
    position = position.clip(-2, 2)
    return position
```

---

## 4. 多品种 CTA 组合

```python
class CTAStrategy:
    def __init__(self, fast=20, slow=120, target_vol=0.15):
        self.fast = fast
        self.slow = slow
        self.target_vol = target_vol
    
    def generate_signals(self, prices_dict):
        """
        prices_dict: {asset_name: price_series}
        """
        positions = {}
        for asset, prices in prices_dict.items():
            returns = prices.pct_change()
            # 趋势信号
            ma_fast = prices.rolling(self.fast).mean()
            ma_slow = prices.rolling(self.slow).mean()
            raw_signal = (ma_fast > ma_slow).astype(float) * 2 - 1
            # 波动率调整仓位
            vol = returns.rolling(60).std() * (252 ** 0.5)
            position = raw_signal * (self.target_vol / (vol + 1e-8))
            positions[asset] = position.clip(-1, 1)
        return pd.DataFrame(positions)
    
    def calc_portfolio_returns(self, positions, returns_dict):
        """等权多品种组合收益"""
        port_ret = pd.DataFrame()
        for asset in positions.columns:
            pos = positions[asset].shift(1)  # 次日执行
            ret = returns_dict[asset]
            port_ret[asset] = pos * ret
        return port_ret.mean(axis=1)
```

---

## 5. CTA 的危机 alpha

CTA 最重要的特征：**与股票负相关，在危机时期保护组合**。

| 事件 | CTA 表现 |
|------|---------|
| 2008 金融危机 | Winton +20%，AHL +25% |
| 2020 COVID 初期 | 多数 CTA 正收益 |
| 2022 加息周期 | CTA 整体 +25%（趋势明确）|
| 2009-2011 震荡市 | CTA 表现差（趋势反复破坏）|

**为什么 CTA 在危机时期有正收益？**
- 危机往往伴随明显趋势（股票暴跌、债券上涨、美元上涨）
- CTA 做空股市、做多债券，趋势信号清晰

---

## 6. CTA 的局限性

- **震荡市**：趋势信号反复打穿止损，损失较大
- **趋势结束**：持仓到顶点才开始反向，损失高点收益
- **滑点成本**：高换手在流动性差的市场中成本高

---

## 小结

| 维度 | 内容 |
|------|------|
| 核心信号 | 移动平均交叉、时序动量、布林带 |
| 仓位管理 | 波动率目标化（风险平价思想）|
| 优势 | 危机时期保护，多资产分散 |
| 缺点 | 震荡市亏损，信号滞后 |
| 容量 | 大（期货市场流动性好）|
