---
layout: default
title: "D1 · 描述统计：均值、方差、偏度、峰度"
---

# D1 · 描述统计在量化中的应用

> **Quant Week 3**  
> 在下单之前，先把数据说清楚。描述统计是量化分析的第一关。

---

## 一、为什么量化需要描述统计？

```
原始数据：股票过去250天的每日收益率
[-0.02, 0.01, 0.03, -0.05, 0.02, ...]

需要回答的问题：
• 平均收益率是多少？（期望收益）
• 波动有多大？（风险）
• 是否有"肥尾"？（极端风险）
• 分布是否对称？（正负收益是否均衡）
```

---

## 二、核心统计量

### 均值（Mean）：期望收益

$$\mu = \bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$$

**算术均值** vs **几何均值**（复利场景更准确）：

$$\mu_{geo} = \left(\prod_{i=1}^n (1+r_i)\right)^{1/n} - 1$$

### 方差与标准差：波动率

$$\sigma^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2, \quad \sigma = \sqrt{\sigma^2}$$

在量化中，年化波动率 = $\sigma_{daily} \times \sqrt{252}$

### 偏度（Skewness）：分布的不对称性

$$\text{Skew} = \frac{1}{n} \sum \left(\frac{x_i - \bar{x}}{\sigma}\right)^3$$

- **正偏**：右尾长，极端正收益更可能
- **负偏**：左尾长，极端亏损更可能（大多数股票策略！）

### 峰度（Kurtosis）：尾部厚薄

$$\text{Kurt} = \frac{1}{n} \sum \left(\frac{x_i - \bar{x}}{\sigma}\right)^4 - 3$$

- 正态分布：峰度 = 0（超额峰度）
- **金融数据通常峰度 > 0**（肥尾），意味着极端事件比正态分布预测的更频繁

---

## 三、Python 实战：分析股票收益率分布

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 模拟一年的日收益率数据（真实A股特征：负偏，肥尾）
np.random.seed(42)
n = 252  # 一年交易日

# 用 t 分布模拟肥尾（自由度 5），并加入负偏
daily_returns = stats.t.rvs(df=5, loc=0.0003, scale=0.015, size=n)

# ============ 描述统计 ============
mean_r = np.mean(daily_returns)
std_r = np.std(daily_returns, ddof=1)
skew_r = stats.skew(daily_returns)
kurt_r = stats.kurtosis(daily_returns)  # 超额峰度

print("=== 收益率描述统计 ===")
print(f"日均收益率:  {mean_r:.4f} ({mean_r*252*100:.1f}% 年化)")
print(f"日波动率:    {std_r:.4f} ({std_r*np.sqrt(252)*100:.1f}% 年化)")
print(f"偏度:        {skew_r:.4f} ({'负偏，有下行风险' if skew_r < 0 else '正偏'}")
print(f"超额峰度:    {kurt_r:.4f} ({'肥尾，极端风险大' if kurt_r > 0 else '薄尾'}")
print(f"最大单日亏损: {daily_returns.min()*100:.2f}%")
print(f"最大单日涨幅: {daily_returns.max()*100:.2f}%")

# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 收益率时序
axes[0, 0].plot(daily_returns * 100)
axes[0, 0].axhline(0, color='black', linewidth=0.5)
axes[0, 0].set_title(f'日收益率时序（均值={mean_r*100:.3f}%）')
axes[0, 0].set_ylabel('收益率 (%)')

# 2. 直方图 + 正态分布对比
axes[0, 1].hist(daily_returns * 100, bins=50, density=True, alpha=0.7, label='实际分布')
x = np.linspace(daily_returns.min()*100, daily_returns.max()*100, 100)
axes[0, 1].plot(x, stats.norm.pdf(x, mean_r*100, std_r*100), 'r-', label='正态分布')
axes[0, 1].set_title(f'分布 vs 正态（峰度={kurt_r:.2f}）')
axes[0, 1].legend()

# 3. QQ 图（判断是否正态）
stats.probplot(daily_returns, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('QQ 图（偏离对角线 = 肥尾）')

# 4. 累积收益
cumulative = (1 + daily_returns).cumprod()
axes[1, 1].plot(cumulative)
axes[1, 1].axhline(1, color='gray', linestyle='--')
axes[1, 1].set_title(f'累积收益（最终: {cumulative[-1]:.3f}x）')

plt.tight_layout()
plt.show()
```

---

## 四、最大回撤（Maximum Drawdown）

量化中最重要的风险指标之一：

```python
def max_drawdown(returns):
    """计算最大回撤"""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

returns_series = pd.Series(daily_returns)
mdd = max_drawdown(returns_series)
print(f"\n最大回撤: {mdd*100:.2f}%")

# 夏普比率（风险调整后收益）
risk_free_rate = 0.02 / 252  # 年化2%的无风险利率
sharpe = (mean_r - risk_free_rate) / std_r * np.sqrt(252)
print(f"夏普比率: {sharpe:.3f}")
```

---

## 五、厚尾的现实意义

```python
# 正态分布 vs 真实市场：极端事件频率对比
sigma = std_r

# 正态分布下，超过 3σ 的概率
prob_normal_3sigma = 2 * (1 - stats.norm.cdf(3))  # 双尾
actual_3sigma = np.mean(np.abs(daily_returns) > 3 * sigma)

print(f"\n3σ 以外的事件：")
print(f"  正态分布预测: {prob_normal_3sigma*100:.3f}%")
print(f"  实际发生:     {actual_3sigma*100:.3f}%")
print(f"  实际/预测倍数: {actual_3sigma/prob_normal_3sigma:.1f}x")
# 实际中通常是正态预测的 5-10 倍！
```

---

## 今天的关键认识

1. **均值、方差**：期望收益和风险，量化最基础的刻画
2. **偏度**：大多数股票策略有负偏，极端亏损比极端盈利更常见
3. **峰度（肥尾）**：金融数据的核心特征，风险模型必须考虑
4. **最大回撤 + 夏普比率**：比单看收益率更全面的策略评价

---

## 明天预告

D2：**概率分布**——正态分布的局限，t 分布、幂律分布在金融中的应用。
