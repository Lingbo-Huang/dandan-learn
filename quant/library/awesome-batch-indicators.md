---
layout: default
title: "技术指标批量计算与可视化"
source: "https://github.com/0voice/Awesome-QuantDev-Learn"
---

> **来源**：[Awesome-QuantDev · Python量化开发](https://github.com/0voice/Awesome-QuantDev-Learn)

## ✅ 实战示例：技术指标批量计算与可视化

---

### 🧰 使用库

* `pandas`：处理结构化数据
* `numpy`：数学计算
* `matplotlib`：绘图展示
* `talib`：技术指标计算（可选）
* `tushare`：数据获取（如你还未获取数据，也可以用 CSV）

---

### 🔍 一、读取股票数据（CSV 或 TuShare）

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib

# 读取之前下载的日线数据
df = pd.read_csv("data/000001.SZ_daily.csv", parse_dates=['trade_date'])
df.sort_values('trade_date', inplace=True)
df.set_index('trade_date', inplace=True)
```

---

### 🧠 二、计算技术指标

#### 1. 简单移动平均线（MA）

```python
df['MA_10'] = df['close'].rolling(window=10).mean()
df['MA_30'] = df['close'].rolling(window=30).mean()
```

#### 2. 指数移动平均线（EMA）

```python
df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
```

#### 3. MACD 指标

```python
df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
    df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
```

#### 4. RSI 相对强弱指数

```python
df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
```

#### 5. 布林带（Bollinger Bands）

```python
upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
df['BOLL_upper'] = upper
df['BOLL_middle'] = middle
df['BOLL_lower'] = lower
```

---

### 📈 三、可视化分析

#### 1. 收盘价 + MA 均线

```python
plt.figure(figsize=(14, 5))
plt.plot(df.index, df['close'], label='收盘价')
plt.plot(df.index, df['MA_10'], label='MA 10')
plt.plot(df.index, df['MA_30'], label='MA 30')
plt.title('收盘价与MA均线')
plt.legend()
plt.grid(True)
plt.show()
```

#### 2. 布林带指标

```python
plt.figure(figsize=(14, 5))
plt.plot(df.index, df['close'], label='收盘价')
plt.plot(df.index, df['BOLL_upper'], label='BOLL 上轨', linestyle='--', color='g')
plt.plot(df.index, df['BOLL_middle'], label='BOLL 中轨', linestyle='-.')
plt.plot(df.index, df['BOLL_lower'], label='BOLL 下轨', linestyle='--', color='r')
plt.fill_between(df.index, df['BOLL_upper'], df['BOLL_lower'], color='gray', alpha=0.2)
plt.title('布林带')
plt.legend()
plt.grid(True)
plt.show()
```

#### 3. MACD 柱状图

```python
plt.figure(figsize=(14, 3))
plt.plot(df.index, df['MACD'], label='MACD')
plt.plot(df.index, df['MACD_signal'], label='Signal')
plt.bar(df.index, df['MACD_hist'], label='Hist', color='grey', alpha=0.5)
plt.title('MACD')
plt.legend()
plt.grid(True)
plt.show()
```

#### 4. RSI 指标

```python
plt.figure(figsize=(14, 3))
plt.plot(df.index, df['RSI_14'], label='RSI(14)')
plt.axhline(70, color='r', linestyle='--')
plt.axhline(30, color='g', linestyle='--')
plt.title('RSI 指标')
plt.grid(True)
plt.show()
```

---

### 📊 四、生成综合指标报告（表格）

```python
print("最新技术指标：")
print(df[['close', 'MA_10', 'MA_30', 'EMA_12', 'EMA_26', 'MACD', 'RSI_14']].tail(5))
```

---

### 📦 五（可选）：封装为函数/类

可将指标计算封装成函数，便于批量多只股票处理：

```python
def compute_technical_indicators(df):
    df['MA_10'] = df['close'].rolling(window=10).mean()
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
    df['RSI_14'] = talib.RSI(df['close'])
    upper, middle, lower = talib.BBANDS(df['close'])
    df['BOLL_upper'] = upper
    df['BOLL_middle'] = middle
    df['BOLL_lower'] = lower
    return df
```

---

## ✅ 技术指标实战小结

| 指标       | 意义           | 常用用法             |
| -------- | ------------ | ---------------- |
| MA / EMA | 趋势识别         | 短期均线上穿长期均线为买入信号  |
| MACD     | 趋势动量变化       | 金叉买入，死叉卖出        |
| RSI      | 超买超卖判断       | RSI>70为超买，<30为超卖 |
| BOLL     | 波动率变化/支撑阻力判断 | 突破上轨为强势，突破下轨为弱势  |

