# Day 7 · Week 1 回顾与横向对比

> **Week 1 主题：金融市场概览**
> 预计学习时间：3~4 小时

---

## 🎯 学习目标

1. 系统回顾 Week 1 的所有核心知识点
2. 完成五大市场的横向对比分析（量化视角）
3. 构建个人的"市场特征认知图谱"
4. 规划 Week 2 的学习重点

---

## 📚 核心知识点

### 1. Week 1 知识点全景回顾

```
Week 1 · 金融市场概览
  ├── D1 A股
  │   ├── 三大交易所（上交所 / 深交所 / 北交所）
  │   ├── T+1 / 涨跌停 / 竞价制度
  │   └── 指数体系（沪深300 / 中证500 / 创业板指）
  │
  ├── D2 港股
  │   ├── 联交所 / H股 / 红筹股
  │   ├── T+2 / 无涨跌停 / 手数因股而异
  │   └── 沪深港通（北向/南向资金）
  │
  ├── D3 美股
  │   ├── NYSE / NASDAQ / AMEX
  │   ├── T+0 / 无涨跌停 / 完整做空
  │   └── S&P500 / VIX / 熔断机制
  │
  ├── D4 期货
  │   ├── 四大期货交易所（上期所/大商所/郑商所/中金所）
  │   ├── 杠杆 / 保证金 / 双向 / T+0
  │   └── 基差 / 主力合约 / 升贴水
  │
  ├── D5 加密货币
  │   ├── CEX / DEX / BTC / ETH
  │   ├── 7×24 / 无涨跌停 / 永续合约
  │   └── 资金费率 / 链上数据 / Crypto Cycle
  │
  └── D6 数据实战
      ├── akshare / yfinance / ccxt
      ├── 标准化数据格式 OHLCV
      └── 数据质量检验 / 夏普比率
```

### 2. 五大市场横向对比表

| 对比维度 | A股 | 港股 | 美股 | 期货 | 加密 |
|---------|-----|------|------|------|------|
| 交易时间 | 4h/日 | 5.5h/日 | 6.5h+盘前盘后 | 6.5h/日（夜盘+日盘）| 24×7 |
| 交割制度 | T+1 | T+2 | T+0 | T+0（日内无限）| 实时 |
| 涨跌停 | ±10%/±20% | 无 | 无（个股）| ±4%~±15% | 无 |
| 做空难度 | 很难 | 中等 | 容易 | 容易 | 容易 |
| 杠杆 | 有限（融资）| 有限 | 有限（期权/融资）| 高杠杆（5~20x）| 极高（最高100x）|
| 年化波动率（参考）| 15~25% | 15~30% | 12~20% | 20~40% | 60~100% |
| 流动性 | 高（主板）| 中（蓝筹高）| 极高 | 高（主力合约）| 高（BTC）|
| 监管强度 | 强 | 中 | 强（但灵活）| 强 | 弱（快速建设中）|
| 数据获取 | akshare | akshare | yfinance | akshare | ccxt |
| 量化友好度 | 中（T+1限制）| 中 | 高 | 高 | 高 |

### 3. 量化策略与市场适配性

**日内策略（High Frequency / Intraday）：**
- ✅ 最适合：美股（T+0 + 高流动性）/ 期货（T+0 + 杠杆）/ 加密（24×7）
- ❌ 不适合：A股（T+1 限制日内）

**趋势跟踪策略（Trend Following）：**
- ✅ 最适合：期货（双向 + 高杠杆）/ 加密（趋势明显）
- ⭕ 适合：美股（标普500有长期趋势）

**均值回归策略（Mean Reversion）：**
- ✅ A股（涨跌停制造强制均值回归）/ 港股（A+H溢价套利）

**套利策略（Arbitrage）：**
- AH 溢价套利（A股 vs 港股）
- 加密跨交易所套利（币安 vs OKX 价差）
- 期货跨期套利（近月 vs 远月）

**事件驱动（Event-Driven）：**
- ✅ A股（财报、政策、重组事件反应剧烈）
- ✅ 加密（BTC 减半、监管政策）

---

## 💡 示例 / 实操

### 综合分析：多市场近2年走势对比

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取 D6 生成的数据
import os
DATA_DIR = "./data/week01"

datasets = {
    "A股沪深300": "ashare_hs300.csv",
    "港股恒生":   "hkstock_hsi.csv",
    "美股标普500": "usstock_sp500.csv",
    "螺纹钢期货": "futures_rb_main.csv",
    "比特币":     "crypto_btcusdt.csv",
}

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 图1：归一化走势对比
ax1 = axes[0]
for label, fname in datasets.items():
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= "2023-01-01"].sort_values("date")
    norm = df["close"] / df["close"].iloc[0]
    ax1.plot(df["date"], norm, label=label)

ax1.axhline(1, color="gray", linestyle="--", linewidth=0.8)
ax1.set_title("五大市场归一化走势对比（2023年至今）", fontsize=13)
ax1.legend()
ax1.set_ylabel("相对价格（初始=1）")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

# 图2：年化波动率对比（柱状图）
ax2 = axes[1]
vols = []
for label, fname in datasets.items():
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["returns"] = df["close"].pct_change()
    ann_vol = df["returns"].std() * np.sqrt(252) * 100
    vols.append((label, ann_vol))

labels, values = zip(*vols)
bars = ax2.bar(labels, values, color=["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6"])
ax2.set_title("各市场年化波动率对比", fontsize=13)
ax2.set_ylabel("年化波动率（%）")
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "week01_summary_chart.png"), dpi=150)
plt.show()
print("图表已保存")
```

### 综合统计报告

```python
# 输出所有市场的统计摘要表
summary_rows = []
for label, fname in datasets.items():
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["returns"] = df["close"].pct_change().dropna()
    ann_ret = df["returns"].mean() * 252
    ann_vol = df["returns"].std() * np.sqrt(252)
    sharpe = (ann_ret - 0.03) / ann_vol
    max_dd = ((df["close"] - df["close"].cummax()) / df["close"].cummax()).min()
    best_day = df["returns"].max()
    worst_day = df["returns"].min()
    summary_rows.append({
        "市场": label,
        "年化收益": f"{ann_ret:.1%}",
        "年化波动": f"{ann_vol:.1%}",
        "夏普比率": f"{sharpe:.2f}",
        "最大回撤": f"{max_dd:.1%}",
        "最佳单日": f"{best_day:.1%}",
        "最差单日": f"{worst_day:.1%}",
    })

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))
```

---

## 🏋️ 动手练习

### 练习 1：完成五大市场综合对比图

- [ ] 运行上方代码，生成并保存对比图
- [ ] 解读图表：哪个市场表现最好？哪个波动最大？
- [ ] 用文字写出你的分析（3~5句）

### 练习 2：构建个人"市场选择矩阵"

根据以下条件，填写你认为最适合的市场（可多选）：

| 场景 | 你的答案 | 理由 |
|------|---------|------|
| 我有10万人民币，想低风险稳健投资 | ? | |
| 我想做日内高频交易 | ? | |
| 我对宏观经济很敏感，想做趋势 | ? | |
| 我想做统计套利（利用价差均值回归）| ? | |
| 我能承受高波动，追求超额收益 | ? | |

### 练习 3：写下你的 Week 1 学习总结

在 `/dandan-learn/quant/week01/my_notes.md` 中写下：
1. 最让你印象深刻的3个知识点
2. 你之前对某个市场的错误认知（如果有）
3. 你对 Week 2 最想学的内容是什么？
4. 还有什么疑问尚未解答？

---

## 📝 Week 1 小结

**5天学习，你建立了什么？**

✅ **认知层面：**
- 理解了5个主要金融市场的结构、规则、特点
- 建立了横向比较的视角：T+0/T+1、涨跌停、杠杆、波动率

✅ **工具层面：**
- akshare：A股/港股/期货数据
- yfinance：美股数据
- ccxt：加密货币数据
- pandas + matplotlib：数据处理与可视化基础

✅ **量化思维层面：**
- 学会用年化收益率、年化波动率、夏普比率评估市场
- 理解了不同市场对不同策略类型的适配性
- 知道了数据标准化的重要性

**Week 2 展望：量化基础工具箱**

建议 Week 2 的方向：
- D1: Python 量化环境完整搭建（pandas/numpy/matplotlib/ta-lib）
- D2: 技术指标计算（MA/RSI/MACD/布林带）
- D3: 回测框架入门（backtrader 或 vectorbt）
- D4: 第一个完整策略：双均线择时
- D5: 策略评估指标体系
- D6-D7: 策略优化与过拟合防范

**量化学习的底层哲学：**
> 市场不是敌人，是信息的集合体。  
> 好的量化研究员，是在海量数据中找到可重复的统计规律，并把它转化为可执行的策略。  
> Week 1 的认知地图，是你所有后续工作的底层支撑。

---

*Week 1 完成！恭喜你建立了对五大金融市场的系统认知 🎉*

---

*D7 完成打卡 ✅ · Week 1 全部完成 🏆*
