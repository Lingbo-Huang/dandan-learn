# Day 3：胖尾分布（Fat Tails）

> **核心问题**：金融市场的极端事件比正态分布预测的频率高得多——如何更准确地建模这种"厚尾"现象？

---

## 1. 理论基础

### 1.1 什么是胖尾（Fat/Heavy Tail）？

**尾部概率**：当 $x \to \infty$ 时，分布尾部概率 $P(X > x)$ 衰减速度。

| 分布类型 | 尾部衰减 | 特征 |
|---------|---------|------|
| 正态分布 | 指数衰减（$e^{-x^2/2}$）| 极端事件极其罕见 |
| **胖尾分布** | **幂律衰减（$x^{-\alpha}$）**| **极端事件显著更频繁** |

**直观理解**：正态分布 5σ 事件约 10 亿年一遇，而实际金融市场可能几年就出现一次。

### 1.2 度量厚尾的关键指标

#### 峰度（Kurtosis）

$$K = \frac{E[(X-\mu)^4]}{\sigma^4}$$

- 正态分布：$K = 3$
- 超峰度（Excess Kurtosis）：$\kappa = K - 3$
- $\kappa > 0$：尖峰厚尾（leptokurtic）
- 金融收益率典型值：$\kappa \in [3, 20]$

#### 偏度（Skewness）

$$S = \frac{E[(X-\mu)^3]}{\sigma^3}$$

- 正态分布：$S = 0$
- 金融收益率：通常 $S < 0$（左偏，大跌比大涨更极端）

### 1.3 常用胖尾分布

#### Student's t 分布

$$f(x; \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$

- $\nu$：自由度（越小尾部越厚）
- $\nu \to \infty$：趋近正态
- 金融数据拟合通常 $\nu \approx 3 \sim 6$
- 超峰度：$\kappa = \frac{6}{\nu - 4}$（$\nu > 4$ 时存在）

#### 广义误差分布（GED / Power Exponential）

$$f(x; \nu) \propto \exp\left(-\frac{|x|^\nu}{2\lambda^\nu}\right)$$

- $\nu = 2$：正态分布
- $\nu < 2$：尖峰厚尾
- $\nu = 1$：双指数/Laplace 分布

#### 极值理论（Extreme Value Theory, EVT）

**GPD（广义帕累托分布）**用于建模超过阈值 $u$ 的损失：

$$F_u(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}$$

- $\xi > 0$：Fréchet 类（胖尾，幂律衰减）
- $\xi = 0$：Gumbel 类（指数衰减）
- $\xi < 0$：有限右端点（薄尾）

**尾部指数（Tail Index）$\alpha$**：$P(X > x) \sim x^{-\alpha}$

- 典型股票收益率：$\alpha \approx 3 \sim 5$
- S&P 500 历史：$\alpha \approx 3$（存在有限四阶矩，峰度有限）

---

## 2. Hill 估计量

估计尾部指数：选取最大的 $k$ 个观测值，

$$\hat{\alpha}_{Hill} = \left(\frac{1}{k}\sum_{i=1}^{k}\ln\frac{X_{(n-i+1)}}{X_{(n-k)}}\right)^{-1}$$

---

## 3. Python 实战代码

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "akshare",
#   "pandas",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///

"""
Day 3: 胖尾分布分析
运行方式: uv run quant-w2-fat-tails.py
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ─── 1. 获取 CSI300 对数收益率 ─────────────────────────────────────────────
print("获取沪深300数据...")
df = ak.stock_zh_index_daily(symbol="sh000300")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()
df = df[df.index >= "2010-01-01"]["close"]
log_ret = np.log(df / df.shift(1)).dropna().values

print(f"样本量：{len(log_ret)} 个交易日")
print(f"偏度：{stats.skew(log_ret):.4f}")
print(f"超峰度：{stats.kurtosis(log_ret):.4f}")

# ─── 2. 拟合 t 分布 ────────────────────────────────────────────────────────
print("\n=== 拟合 Student's t 分布 ===")
df_t, loc_t, scale_t = stats.t.fit(log_ret)
print(f"自由度 ν = {df_t:.2f}")
print(f"位置参数 μ = {loc_t:.6f}")
print(f"尺度参数 σ = {scale_t:.6f}")
print(f"对应超峰度 = {6/(df_t-4):.2f}" if df_t > 4 else f"自由度≤4，峰度无限")

# 对数似然对比
ll_normal = stats.norm.logpdf(log_ret, log_ret.mean(), log_ret.std()).sum()
ll_t = stats.t.logpdf(log_ret, df_t, loc_t, scale_t).sum()
print(f"\n对数似然对比：")
print(f"  正态分布：{ll_normal:.2f}")
print(f"  t 分布：{ll_t:.2f}")
print(f"  Δ = {ll_t - ll_normal:.2f}（正值表示 t 分布更好拟合）")

# ─── 3. 拟合 GED ──────────────────────────────────────────────────────────
# GED 即 norm 的 generalized 版本，用 scipy.stats.gennorm 实现
print("\n=== 拟合广义正态分布（GED）===")
beta_ged, loc_ged, scale_ged = stats.gennorm.fit(log_ret)
print(f"形状参数 β = {beta_ged:.2f}（=2 为正态，<2 为尖峰厚尾）")
ll_ged = stats.gennorm.logpdf(log_ret, beta_ged, loc_ged, scale_ged).sum()
print(f"对数似然：{ll_ged:.2f}")

# ─── 4. 尾部事件概率对比 ───────────────────────────────────────────────────
print("\n=== 极端事件概率（左尾，日跌幅）===")
mu = log_ret.mean()
sigma = log_ret.std()

header = f"{'阈值':>10} {'正态':>12} {'t分布':>12} {'实际频率':>12}"
print(header)
print("-" * 50)

for k in [2, 3, 4, 5]:
    threshold = -(mu - k * sigma)   # 跌幅阈值（正数）
    # 正态分布概率
    p_normal = stats.norm.sf(k)
    # t 分布概率
    p_t = stats.t.sf((threshold + mu) / scale_t, df_t)   # 近似，精确需标准化
    # 实际频率
    p_actual = (log_ret < -(mu - k * sigma)).mean()   # 负收益超过 kσ
    
    print(f"  >{k}σ 跌   {p_normal:>12.6%} {p_t:>12.6%} {p_actual:>12.6%}")

# ─── 5. Hill 估计量（尾部指数估计）────────────────────────────────────────
print("\n=== Hill 估计：尾部指数 α ===")

def hill_estimator(data: np.ndarray, k_range: np.ndarray) -> np.ndarray:
    """对右尾数据估计 Hill 尾部指数"""
    sorted_data = np.sort(np.abs(data))[::-1]   # 降序排列绝对值
    hills = []
    for k in k_range:
        if k >= len(sorted_data):
            hills.append(np.nan)
            continue
        tail = sorted_data[:k]
        threshold = sorted_data[k]
        hill = k / np.sum(np.log(tail / threshold))
        hills.append(hill)
    return np.array(hills)

k_range = np.arange(10, 300, 10)
hill_values = hill_estimator(log_ret, k_range)

# 稳定区域的 α 估计（k=50~150 之间取均值）
stable_idx = (k_range >= 50) & (k_range <= 150)
alpha_estimate = np.nanmean(hill_values[stable_idx])
print(f"Hill 尾部指数估计（k=50~150）：α ≈ {alpha_estimate:.2f}")
print(f"解读：P(|R| > x) ∝ x^(-{alpha_estimate:.1f})，有限 {int(np.floor(alpha_estimate))} 阶矩")

# ─── 6. 可视化 ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Fat Tails Analysis: CSI 300 Daily Log Returns", fontsize=13)

x_range = np.linspace(log_ret.min(), log_ret.max(), 300)

# （1）密度对比
ax = axes[0, 0]
ax.hist(log_ret, bins=100, density=True, alpha=0.5, color="steelblue", label="Empirical")
ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), "r-", lw=2, label="Normal")
ax.plot(x_range, stats.t.pdf(x_range, df_t, loc_t, scale_t), "g-", lw=2, label=f"t(ν={df_t:.1f})")
ax.plot(x_range, stats.gennorm.pdf(x_range, beta_ged, loc_ged, scale_ged), 
        "orange", lw=2, linestyle="--", label=f"GED(β={beta_ged:.1f})")
ax.set_xlim(-0.05, 0.05)
ax.legend(fontsize=8)
ax.set_title("Density: Empirical vs Parametric Fits")
ax.grid(alpha=0.3)

# （2）尾部对数对数图（log-log plot）
ax = axes[0, 1]
sorted_abs = np.sort(np.abs(log_ret))[::-1]
n = len(sorted_abs)
rank = np.arange(1, n + 1)
survival_prob = rank / n

ax.loglog(sorted_abs, survival_prob, "o", markersize=2, alpha=0.5, color="steelblue", label="Empirical")
# 正态尾部参考线
x_tail = np.linspace(0.01, 0.08, 100)
ax.loglog(x_tail, 2 * stats.norm.sf(x_tail / sigma), "r-", lw=2, label="Normal tail")
# 幂律参考线
ax.loglog(x_tail, 0.1 * (x_tail / 0.01) ** (-alpha_estimate), "g--", lw=2, 
          label=f"Power law α={alpha_estimate:.1f}")
ax.set_xlabel("Log Return (absolute)")
ax.set_ylabel("Survival Probability")
ax.set_title("Tail: Log-Log Plot")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# （3）Hill 图
ax = axes[1, 0]
ax.plot(k_range, hill_values, "steelblue", lw=1.5)
ax.axhline(alpha_estimate, color="r", linestyle="--", label=f"α ≈ {alpha_estimate:.2f}")
ax.axvline(50, color="gray", linestyle=":", alpha=0.5)
ax.axvline(150, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("k (Number of tail observations)")
ax.set_ylabel("Hill Estimator (α)")
ax.set_title("Hill Plot: Tail Index Estimation")
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 15)

# （4）QQ 图：实际 vs t 分布
ax = axes[1, 1]
quantiles_empirical = np.percentile(log_ret, np.linspace(0.5, 99.5, 200))
quantiles_t = stats.t.ppf(np.linspace(0.005, 0.995, 200), df_t, loc_t, scale_t)
quantiles_normal = stats.norm.ppf(np.linspace(0.005, 0.995, 200), mu, sigma)

ax.scatter(quantiles_t, quantiles_empirical, s=5, alpha=0.5, color="green", label="vs t-dist QQ")
ax.scatter(quantiles_normal, quantiles_empirical, s=5, alpha=0.3, color="red", label="vs Normal QQ")
min_q = min(quantiles_empirical.min(), quantiles_t.min())
max_q = max(quantiles_empirical.max(), quantiles_t.max())
ax.plot([min_q, max_q], [min_q, max_q], "k-", lw=1)
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
ax.set_title("Q-Q Plot: t-dist vs Normal")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("quant-w2-fat-tails.png", dpi=150)
print("\n图表已保存为 quant-w2-fat-tails.png")
```

---

## 4. 胖尾的实践含义

### 风险管理层面

| 假设 | 危险 |
|------|------|
| 正态分布计算 VaR | 严重低估极端损失 |
| 用历史相关性做压力测试 | 危机时相关性会跳变 |
| 认为 "3σ 事件不会发生" | A 股多次出现 4-5σ 单日跌幅 |

### 建模层面选择

```
收益率建模路径：
  正态（快速，有偏）
    → t 分布（更好的尾部，仍对称）
      → 偏 t 分布（Skewed-t，捕捉不对称性）
        → GPD（专注极值建模，EVT 框架）
```

---

## 5. 小结

| 概念 | 含义 | 量化指标 |
|------|------|---------|
| 峰度 | 尾部厚度 | $K > 3$ 为厚尾 |
| 偏度 | 分布对称性 | $S < 0$ 左偏（下行风险更大） |
| t 分布自由度 | 厚尾程度 | $\nu$ 越小尾部越厚 |
| 尾部指数 $\alpha$ | 幂律衰减速度 | $\alpha \approx 3$（A 股典型值） |

**关键结论**：
1. A 股收益率超峰度通常在 3~8，远超正态
2. t 分布（自由度 3~6）是比正态更合适的基础分布
3. 尾部指数 $\alpha \approx 3$ 意味着方差有限但四阶矩可能无限
4. **胖尾是金融市场的基本特征，不是异常**

---

*下一步 → [Day 4: 风险度量 VaR 与 CVaR](./quant-w2-risk-measures.md)*
