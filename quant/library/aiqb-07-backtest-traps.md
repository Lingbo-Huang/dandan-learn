---
layout: default
title: "AIQuantBook L07 · 回测系统的陷阱"
source: "https://github.com/waylandzhang/ai-quant-book"
---

# L07 · 回测系统的陷阱

> **来源**：[AI Quant Book](https://github.com/waylandzhang/ai-quant-book) · waylandzhang

---

## 核心观点

> **回测是量化的必要条件，但不是充分条件。**
> 
> 一个在回测中表现完美的策略，在实盘中可能惨败。
> 这不是偶然，而是系统性的认知陷阱。

---

## 陷阱一：前视偏差（Look-ahead Bias）

**定义**：使用了在当时实际上无法获得的未来数据。

**最常见的形式**：

```python
# ❌ 错误：使用当日收盘价作为当日买入价
# 现实：当日收盘价要到收盘后才知道
df['signal'] = (df['close'] > df['ma_20']).astype(int)
df['return'] = df['signal'] * df['close'].pct_change()  # 前视！

# ✅ 正确：信号基于当日收盘价，次日开盘执行
df['signal'] = (df['close'] > df['ma_20']).astype(int)
df['return'] = df['signal'].shift(1) * df['open'].pct_change()
```

**财务数据的前视偏差**：

```python
# ❌ 错误：使用报告期对应的财务数据
# 现实：Q1 财报通常在 Q2 结束后才公布
df['pe'] = df['price'] / df['eps_q1']  # Q1结束时用，但Q1数据还没出来！

# ✅ 正确：使用数据真实可获取日期（report_date + 公告延迟）
# 确保每个时间点只使用到该时间点为止已发布的数据
```

---

## 陷阱二：幸存者偏差（Survivorship Bias）

```
现在的沪深300成分股：300家公司
但5年前的股票池：包含很多已经退市/被踢出指数的公司

如果只用现在还存活的300家公司做回测：
→ 你的股票池本来就是"赢家"
→ 策略看起来很好，但历史上那些失败者不在你的数据里
```

```python
# 错误方式：只使用当前成分股
current_hs300 = get_current_hs300_components()  # 幸存者！

# 正确方式：使用历史成分股（各时期的实际成分）
def get_historical_universe(date: str) -> list:
    """返回给定日期有效的股票池（含历史成分股）"""
    # 需要数据库记录每次指数调整的历史
    return get_index_components_at_date('000300.SH', date)
```

**幸存者偏差的影响有多大？**

学术研究表明，幸存者偏差会让策略表现虚高约 1-2% 每年（累积很可观）。

---

## 陷阱三：过度拟合（Overfitting）

```
现象：
  样本内（In-Sample）：夏普 3.0，年化 40%
  样本外（Out-of-Sample）：夏普 0.2，年化 2%

原因：
  你"发现"的不是真实规律，而是噪声的形状
```

**过度拟合的识别**：

```python
def check_overfitting(is_sharpe: float, oos_sharpe: float) -> str:
    """简单的过拟合诊断"""
    decay = (is_sharpe - oos_sharpe) / is_sharpe if is_sharpe > 0 else None
    
    if decay is None:
        return "样本内都是负的，策略本身有问题"
    elif decay < 0.3:
        return "✅ 样本内外衰减 <30%，比较健康"
    elif decay < 0.6:
        return "⚠️ 样本内外衰减 30-60%，轻度过拟合"
    else:
        return "❌ 样本内外衰减 >60%，严重过拟合"

# 防止过拟合的方法：
# 1. 减少参数数量（奥卡姆剃刀）
# 2. 使用正则化
# 3. Walk-forward 验证
# 4. 多个独立数据集验证
# 5. 经济逻辑先行（先有逻辑，再找证据）
```

---

## 陷阱四：交易成本假设

```python
# ❌ 常见错误：忽略交易成本
gross_return = 0.25  # 年化25%，看起来很好

# ✅ 现实：每次换手都有成本
def realistic_return(gross_return: float, 
                      annual_turnover: float,
                      commission: float = 0.0003,
                      stamp_tax: float = 0.001) -> float:
    """
    gross_return: 毛收益率
    annual_turnover: 年化换手率（如 3.0 = 每年换仓3次全部）
    """
    round_trip_cost = commission * 2 + stamp_tax  # 约0.16%
    total_cost = annual_turnover * round_trip_cost
    return gross_return - total_cost

# 年换手率 10 的策略：
net = realistic_return(0.25, annual_turnover=10)
print(f"年化净收益: {net:.2%}")  # 25% - 10*0.16% = 23.4%

# 年换手率 100 的高频策略：
net_hf = realistic_return(0.25, annual_turnover=100)
print(f"高频年化净收益: {net_hf:.2%}")  # 25% - 100*0.16% = 9%
```

---

## 陷阱五：市场冲击（Market Impact）

```
小账户（100万）：可以按照回测价格成交
大账户（10亿）：你的买单本身会推高价格

问题：回测假设你能以市场价格成交，
但大资金的实际成交价远比假设差
```

```python
def estimate_market_impact(order_size: float,  # 订单金额（万元）
                             adv: float,          # 日均成交额（万元）
                             urgency: float = 0.1  # 参与率
                            ) -> float:
    """
    简化的市场冲击模型
    参与率 = order_size / (adv × days)
    冲击 ≈ sqrt(参与率) × 波动率
    """
    participation_rate = urgency
    volatility = 0.02  # 假设2%日波动
    
    # Almgren-Chriss 简化版
    impact = 0.5 * participation_rate * volatility
    return impact

# 策略容量分析
def capacity_analysis(strategy_alpha: float,
                       adv: float,
                       max_participation: float = 0.10):
    """估算策略的资金容量上限"""
    # 当市场冲击成本 = 策略Alpha时，达到容量上限
    impact_per_trade = estimate_market_impact(1, adv, max_participation)
    
    # 最大可承受的订单规模
    max_order_value = adv * max_participation
    
    # 年化换手假设为 4 次
    max_aum = max_order_value / 0.25  # 25%换手对应AUM
    
    print(f"策略日均Alpha: {strategy_alpha:.4f}")
    print(f"单次交易冲击: {impact_per_trade:.4f}")
    print(f"有效Alpha比例: {strategy_alpha / (strategy_alpha + impact_per_trade):.1%}")
    print(f"估算策略容量上限: {max_aum:.0f} 万元")
```

---

## 回测自检清单

```
数据质量
□ 是否排除了幸存者偏差？
□ 财务数据是否使用了真实发布日期？
□ 价格数据是否经过复权处理？

逻辑检查
□ 所有信号是否基于过去数据？（无前视偏差）
□ 是否考虑了T+1规则？
□ 是否考虑了涨跌停无法成交的情况？

成本模型
□ 是否扣除了佣金和印花税？
□ 是否考虑了市场冲击？
□ 是否考虑了滑点？

统计有效性
□ 是否进行了样本外验证？
□ 是否进行了多重检验校正？
□ 策略背后是否有经济逻辑支撑？
```

---

## 延伸阅读

- [AI Quant Book](https://github.com/waylandzhang/ai-quant-book)
- Bailey et al. - "The Probability of Backtest Overfitting"
- Harvey & Liu - "… and the Cross-Section of Expected Returns"
