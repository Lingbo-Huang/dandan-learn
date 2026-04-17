# Day 1：对数收益率（Log Returns）

> **核心问题**：为什么量化金融更偏爱对数收益率？它有哪些数学优势？

---

## 1. 理论基础

### 1.1 两种收益率定义

**简单收益率（Simple Return）：**

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$

**对数收益率（Log Return / Continuously Compounded Return）：**

$$R_t = \ln\frac{P_t}{P_{t-1}} = \ln P_t - \ln P_{t-1}$$

其中 $P_t$ 为第 $t$ 期资产价格。

### 1.2 两者关系

$$R_t = \ln(1 + r_t) \approx r_t \quad \text{（当 } r_t \text{ 较小时）}$$

Taylor 展开：

$$\ln(1 + r_t) = r_t - \frac{r_t^2}{2} + \frac{r_t^3}{3} - \cdots$$

误差量级：当日收益率约 1%，误差约 0.005%，可以忽略。

---

## 2. 对数收益率的核心优势

### 2.1 时间可加性（最重要！）

简单收益率**不可加**：

$$1 + r_{[1,T]} = \prod_{t=1}^{T}(1 + r_t) \quad \text{（连乘）}$$

对数收益率**可加**：

$$R_{[1,T]} = \sum_{t=1}^{T} R_t \quad \text{（连加）}$$

**含义**：若每天的对数收益率独立同分布，则周/月/年收益率是单日收益率的简单求和，数学上极为方便。

### 2.2 价格非负性自动保证

对数收益率没有下界限制（$-\infty$ 到 $+\infty$ 均合法），而简单收益率下界为 $-1$（价格不可为负）。若假设 $R_t \sim \mathcal{N}(\mu, \sigma^2)$，则 $P_t = P_0 \cdot e^{\sum R_t} > 0$，价格天然非负。

### 2.3 对称性

价格翻倍的对数收益率 = +69.3%；价格减半的对数收益率 = -69.3%。简单收益率不对称（+100% vs -50%）。

### 2.4 统计性质更好

在正态假设下，对数收益率对应**对数正态**的价格分布，这是 Black-Scholes 模型的基础假设。

---

## 3. 多期收益率计算

$$R_{[t, t+k]} = \sum_{j=0}^{k-1} R_{t+j} = \ln P_{t+k} - \ln P_t$$

**年化**：若单日对数收益率均值为 $\bar{R}$，标准差为 $\sigma$：

$$\mu_{\text{annual}} = 252 \cdot \bar{R}$$
$$\sigma_{\text{annual}} = \sqrt{252} \cdot \sigma$$

（252 为年均交易日数）

---

## 4. Python 实战代码

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "akshare",
#   "pandas",
#   "numpy",
#   "matplotlib",
# ]
# ///

"""
Day 1: 对数收益率计算与可视化
运行方式: uv run quant-w2-log-returns.py
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ─── 1. 获取数据：沪深300 ───────────────────────────────────────────────────
print("正在获取沪深300指数数据...")
df = ak.stock_zh_index_daily(symbol="sh000300")
df = df.rename(columns={"date": "date", "close": "close"})
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

# 取最近3年
df = df[df.index >= "2022-01-01"][["close"]].copy()
print(f"数据区间：{df.index[0].date()} ~ {df.index[-1].date()}，共 {len(df)} 个交易日")

# ─── 2. 计算两种收益率 ──────────────────────────────────────────────────────
df["simple_return"] = df["close"].pct_change()                    # 简单收益率
df["log_return"] = np.log(df["close"] / df["close"].shift(1))    # 对数收益率
df = df.dropna()

# ─── 3. 对比统计 ────────────────────────────────────────────────────────────
print("\n=== 收益率对比统计 ===")
stats = pd.DataFrame({
    "简单收益率": df["simple_return"].describe(),
    "对数收益率": df["log_return"].describe(),
})
print(stats.round(6))

# ─── 4. 时间可加性验证 ──────────────────────────────────────────────────────
# 2023年全年对数收益率求和，应等于当年价格变化的对数
df_2023 = df.loc["2023-01-01":"2023-12-31"]
sum_log_returns = df_2023["log_return"].sum()
actual_log_change = np.log(df_2023["close"].iloc[-1] / df_2023["close"].iloc[0])

print(f"\n=== 时间可加性验证（2023年）===")
print(f"对数收益率求和：{sum_log_returns:.6f}")
print(f"实际价格对数变化：{actual_log_change:.6f}")
print(f"误差：{abs(sum_log_returns - actual_log_change):.2e}（应接近0）")

# ─── 5. 年化指标 ────────────────────────────────────────────────────────────
mu_annual = df["log_return"].mean() * 252
sigma_annual = df["log_return"].std() * np.sqrt(252)
print(f"\n=== 年化指标（日对数收益率 × 252）===")
print(f"年化收益率：{mu_annual:.2%}")
print(f"年化波动率：{sigma_annual:.2%}")
print(f"夏普比率（无风险=3%）：{(mu_annual - 0.03) / sigma_annual:.2f}")

# ─── 6. 滚动波动率（20日） ──────────────────────────────────────────────────
df["rolling_vol_20d"] = df["log_return"].rolling(20).std() * np.sqrt(252)

# ─── 7. 可视化 ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle("CSI 300: Log Returns Analysis", fontsize=14)

# 价格序列
axes[0].plot(df.index, df["close"], color="steelblue", linewidth=1)
axes[0].set_title("Price Series")
axes[0].set_ylabel("Index Level")
axes[0].grid(alpha=0.3)

# 对数收益率序列
axes[1].bar(df.index, df["log_return"], color="steelblue", width=1, alpha=0.7)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Daily Log Returns")
axes[1].set_ylabel("Log Return")
axes[1].grid(alpha=0.3)

# 滚动波动率
axes[2].plot(df.index, df["rolling_vol_20d"], color="orangered", linewidth=1)
axes[2].set_title("20-Day Rolling Annualized Volatility")
axes[2].set_ylabel("Annualized Vol")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("quant-w2-log-returns.png", dpi=150)
print("\n图表已保存为 quant-w2-log-returns.png")

# ─── 8. 简单vs对数收益率差异 ───────────────────────────────────────────────
diff = (df["log_return"] - df["simple_return"]).abs()
print(f"\n=== 简单收益率 vs 对数收益率差异 ===")
print(f"绝对差均值：{diff.mean():.2e}")
print(f"绝对差最大值：{diff.max():.2e}")
print("结论：差异极小，日频数据中两者近似相等")
```

---

## 5. 延伸：价格模型基础

若每日对数收益率 $R_t \overset{i.i.d.}{\sim} \mathcal{N}(\mu, \sigma^2)$，则：

$$P_T = P_0 \cdot e^{\sum_{t=1}^{T} R_t} = P_0 \cdot e^{\mu T + \sigma \sqrt{T} Z}, \quad Z \sim \mathcal{N}(0,1)$$

这就是**几何布朗运动**的离散化形式，Black-Scholes 期权定价的基石。

---

## 6. 小结

| 对比项 | 简单收益率 | 对数收益率 |
|--------|-----------|-----------|
| 定义 | $(P_t/P_{t-1}) - 1$ | $\ln(P_t/P_{t-1})$ |
| 多期合并 | 连乘 | **连加** ✅ |
| 价格非负 | 需额外约束 | **自动保证** ✅ |
| 对称性 | 不对称 | **对称** ✅ |
| 统计建模 | 受限于 $[-1, \infty)$ | **无约束** ✅ |
| 近似误差 | 基准 | 日频约 0.005% |

**结论**：量化金融中，**对数收益率是标准选择**。简单收益率在计算组合权重收益时有用（加权平均），但统计分析和建模几乎全用对数收益率。

---

*下一步 → [Day 2: 正态假设检验](./quant-w2-normal-hypothesis.md)*
