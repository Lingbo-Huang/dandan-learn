---
layout: default
title: "技术指标分析与可视化"
source: "https://github.com/0voice/Awesome-QuantDev-Learn"
---

> **来源**：[Awesome-QuantDev · Python量化开发](https://github.com/0voice/Awesome-QuantDev-Learn)

## ✅ 实战示例：分析某股票的技术指标并可视化

### 🧰 使用库：

* `pandas`：处理表格数据
* `numpy`：数值计算
* `matplotlib`：可视化
* `talib`：技术指标计算（如安装有困难可使用 pandas 实现替代版本）

---

### 📦 第一步：加载并预处理股票数据（示例为CSV格式）

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib

# 读取本地或下载的日线数据
df = pd.read_csv('stock_data.csv')  # 假设有列：date, open, high, low, close, volume
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.sort_index()

# 查看前几行
print(df.head())
```

---

### 📈 第二步：使用 NumPy 和 TA-Lib 计算技术指标

```python
# 计算对数收益率（NumPy）
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# 使用 TA-Lib 计算移动平均线
df['MA_20'] = talib.SMA(df['close'], timeperiod=20)
df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)

# 计算布林带
upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
df['BOLL_upper'] = upper
df['BOLL_middle'] = middle
df['BOLL_lower'] = lower

# RSI 指标
df['RSI'] = talib.RSI(df['close'], timeperiod=14)

# 去除NA值
df.dropna(inplace=True)
```

---

### 🖼️ 第三步：绘图可视化（Matplotlib）

#### 1. 收盘价与均线

```python
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['close'], label='收盘价', linewidth=1.5)
plt.plot(df.index, df['MA_20'], label='20日均线')
plt.plot(df.index, df['EMA_20'], label='20日EMA')
plt.title('收盘价与均线')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

#### 2. 布林带

```python
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['close'], label='收盘价')
plt.plot(df.index, df['BOLL_upper'], label='上轨', linestyle='--', color='g')
plt.plot(df.index, df['BOLL_middle'], label='中轨', linestyle='-.')
plt.plot(df.index, df['BOLL_lower'], label='下轨', linestyle='--', color='r')
plt.fill_between(df.index, df['BOLL_upper'], df['BOLL_lower'], color='lightgray', alpha=0.3)
plt.title('布林带指标')
plt.legend()
plt.grid(True)
plt.show()
```

#### 3. RSI 指标

```python
plt.figure(figsize=(14, 3))
plt.plot(df.index, df['RSI'], label='RSI(14)')
plt.axhline(70, color='r', linestyle='--', label='超买线')
plt.axhline(30, color='g', linestyle='--', label='超卖线')
plt.title('RSI 指标')
plt.legend()
plt.grid(True)
plt.show()
```

#### 4. 日对数收益率

```python
plt.figure(figsize=(14, 3))
plt.plot(df.index, df['log_return'], label='对数收益率', color='purple')
plt.axhline(0, linestyle='--', color='black')
plt.title('日对数收益率')
plt.grid(True)
plt.show()
```

---

### 📊 第四步：小结输出指标值

```python
print("最近5天技术指标预览：")
print(df[['close', 'MA_20', 'EMA_20', 'BOLL_upper', 'BOLL_lower', 'RSI']].tail())
```

---

### 🧪 示例数据获取建议：

* 可使用 [TuShare](https://tushare.pro/) 获取数据并保存为CSV：

```python
import tushare as ts
ts.set_token('your_token')
pro = ts.pro_api()
df = pro.daily(ts_code='000001.SZ', start_date='20240101', end_date='20240601')
df.to_csv('stock_data.csv', index=False)
```




