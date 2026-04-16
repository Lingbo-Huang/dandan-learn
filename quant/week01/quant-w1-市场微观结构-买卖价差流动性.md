# Day 5 · 加密货币市场概览

> **Week 1 主题：金融市场概览**
> 预计学习时间：3~4 小时

---

## 🎯 学习目标

1. 理解加密货币市场的基本结构（交易所、DEX、链上数据）
2. 掌握主流资产（BTC / ETH）和市场分层
3. 了解加密市场的独特机制：24×7、高波动、链上透明
4. 动手用 ccxt 获取加密货币行情数据

---

## 📚 核心知识点

### 1. 市场结构

**两大类交易场所：**

| 类型 | 代表 | 特点 |
|------|------|------|
| 中心化交易所（CEX）| Binance、OKX、Bybit、Coinbase | 高流动性，需KYC，有对手方风险 |
| 去中心化交易所（DEX）| Uniswap、Curve、dYdX | 无需KYC，链上透明，流动性较低 |

**主要 CEX 按交易量排名（参考）：**
1. Binance（币安）：全球最大，现货+衍生品
2. OKX：功能丰富，国内用户熟悉
3. Bybit：衍生品见长
4. Coinbase：美股上市，最合规
5. Kraken：欧美市场为主

### 2. 资产分类

**Layer 1 公链原生资产：**
```
BTC  比特币：数字黄金，总量2100万，最古老
ETH  以太坊：智能合约平台，最大生态
BNB  币安链原生代币
SOL  Solana：高性能链
ADA  Cardano
```

**稳定币：**
```
USDT  Tether：最大稳定币，中心化发行
USDC  Circle发行，更合规，受监管
DAI   去中心化超额抵押稳定币
```

**DeFi 代币：**
```
UNI   Uniswap（DEX协议）
AAVE  Aave（借贷协议）
MKR   MakerDAO（DAI的治理代币）
```

**衍生品相关：**
- 永续合约（Perpetual Swap）：无到期日，通过资金费率维持与现货价格锚定
- 季度合约：有固定到期日

### 3. 核心交易规则

**与传统市场的重大差异：**

| 对比项 | 传统市场 | 加密市场 |
|--------|---------|---------|
| 交易时间 | 工作日有限时段 | **7×24 不停歇** |
| 涨跌停 | 有（多数市场）| **无** |
| 最低门槛 | 1股/1手 | **最低可买 0.001 BTC** |
| 结算 | T+1 / T+2 | **实时结算** |
| 监管 | 严格监管 | 监管框架仍在建立中 |
| 透明度 | 有限（财报等）| **链上数据完全透明** |

**永续合约的资金费率（Funding Rate）：**
```
多头 vs 空头的持仓比例决定资金费率方向：
  多头 > 空头 → 多头支付资金费给空头（防止期货过度溢价）
  空头 > 多头 → 空头支付资金费给多头

资金费率通常每8小时结算一次
年化资金费率 > 50% → 市场极度乐观（历史牛市特征）
资金费率为负 → 市场极度悲观
```

### 4. 市场特征

**高波动性：**
- BTC 单日波动 5%-10% 很常见
- 山寨币（Altcoin）可以单日 -50% 到 +100%
- 年化波动率通常在 60%-100%（股票通常15%-20%）

**市场周期（Crypto Cycle）：**
```
牛市驱动因素：BTC 减半（每4年一次）/ 宏观流动性宽松 / 应用爆发
熊市驱动因素：监管收紧 / 流动性收缩 / 黑天鹅事件（如FTX暴雷）

历史减半年份：2012 / 2016 / 2020 / 2024
```

**BTC 市值占比（Dominance）：**
- BTC 占比高 → 市场风险偏好低，资金集中在最安全资产
- BTC 占比低 → 山寨季（Altseason），资金向高风险资产流动

### 5. 链上数据的独特价值

加密市场的独特优势：**所有交易都在区块链上公开可查**

```python
# 可观察的链上指标：
# - 活跃地址数（网络使用率）
# - 巨鲸地址持仓变化（大资金动向）
# - 交易所流入/流出量（抛压/买压信号）
# - 矿工收入和算力（BTC 基本面）
# - DeFi TVL（去中心化金融锁仓量）
```

---

## 💡 示例 / 实操

### 使用 ccxt 获取加密货币数据

```python
import ccxt
import pandas as pd

# 初始化交易所（无需API Key即可获取公开数据）
exchange = ccxt.binance()

# 获取BTC/USDT的OHLCV日线数据
ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=365)
df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df = df.set_index("timestamp")
print(df.tail())
```

### 获取支持的交易对列表

```python
# 查看 Binance 支持的所有交易对
markets = exchange.load_markets()
print(f"Binance 支持 {len(markets)} 个交易对")

# 过滤出 USDT 交易对
usdt_pairs = [s for s in markets.keys() if s.endswith("/USDT")]
print(f"USDT交易对数量：{len(usdt_pairs)}")
```

### 获取永续合约资金费率

```python
# 需要使用期货交易所
future_exchange = ccxt.binance({"options": {"defaultType": "future"}})

# 获取BTC永续合约资金费率历史
funding_history = future_exchange.fetch_funding_rate_history(
    "BTC/USDT:USDT",
    limit=100
)
df_funding = pd.DataFrame(funding_history)
print(df_funding[["datetime", "fundingRate"]].tail(10))
```

---

## 🏋️ 动手练习

### 练习 1：BTC 价格历史分析

```python
import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=1000)
df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
df["ts"] = pd.to_datetime(df["ts"], unit="ms")
df = df.set_index("ts")

# 任务：
# 1. 计算年化波动率，与A股/美股对比
# 2. 找出历史最大单日跌幅的日期（对应什么事件？）
# 3. 画出BTC价格与30日均线的对比图
```

### 练习 2：BTC vs ETH vs 主流山寨币相关性

```python
# 获取 BTC、ETH、SOL、BNB 近一年日线数据
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
prices = {}
for sym in symbols:
    ohlcv = exchange.fetch_ohlcv(sym, timeframe="1d", limit=365)
    df_temp = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    prices[sym.split("/")[0]] = [row[4] for row in ohlcv]

df_prices = pd.DataFrame(prices)

# 计算相关系数矩阵
# 思考：加密市场的相关性比股票市场高还是低？原因是什么？
```

### 练习 3：分析资金费率与价格的关系

```python
# 获取BTC永续合约资金费率（最近3个月）
# 获取BTC现货价格（同期）
# 绘制资金费率和价格的对比图
# 分析：高资金费率是价格顶部信号吗？
```

---

## 📝 小结

**今天学到了什么：**

1. **加密市场 7×24 运转**，没有涨跌停，流动性高度不均匀（大盘币流动性好，山寨币差）
2. **BTC 是数字黄金，ETH 是智能合约平台**，两者定位不同，驱动逻辑不同
3. **永续合约的资金费率是独特的情绪指标**，可作为量化信号之一
4. **链上数据是加密市场的独特优势**，传统市场不具备的透明度
5. **ccxt 是 Python 接入加密交易所的标准库**，支持100+交易所，接口统一

**关键术语：**
- CEX / DEX：中心化 / 去中心化交易所
- 永续合约：无到期日，通过资金费率锚定现货
- 资金费率：多空失衡时，一方向另一方支付的费用
- BTC Dominance：比特币市值占加密市场总市值的比例
- 山寨季（Altseason）：BTC 涨完，资金扩散到小币种

**明日预告：** D6 多市场数据获取实战，用代码打通所有市场的数据管道

---

*D5 完成打卡 ✅*
