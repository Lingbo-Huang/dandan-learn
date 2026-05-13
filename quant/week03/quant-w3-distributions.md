---
layout: default
title: "D2 · 概率分布：正态、t分布、厚尾"
---

# D2 · 金融中的概率分布

> **Quant Week 3**  
> 正态分布是量化工作者用的最多、也误用最多的分布。

---

## 一、正态分布：优雅但危险

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**为什么量化中常用正态**：
- 数学性质好（稳定性、中心极限定理）
- 只需两个参数（μ, σ）
- 很多工具（VaR、期权定价）依赖正态假设

**为什么正态分布在金融中是错的**：

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# 1987年10月19日，道琼斯指数单日下跌 22.6%
# 在正态分布假设下，这件事发生的概率是：
mu = 0.0003
sigma = 0.01
event = -0.226

z_score = (event - mu) / sigma
prob = stats.norm.sf(abs(z_score)) * 2  # 双尾

print(f"1987年股灾：-22.6%")
print(f"Z-score: {z_score:.1f}σ")
print(f"正态分布下概率: {prob:.2e}")
print(f"宇宙年龄 137亿年，预期发生次数: {prob * 252 * 1.37e10:.4f} 次")
# 结论：按正态分布，股灾根本不该发生！
```

---

## 二、学生 t 分布：更厚的尾巴

$$f(x|\nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1+\frac{x^2}{\nu}\right)^{-(\nu+1)/2}$$

- $\nu$：自由度。$\nu \to \infty$ 时趋向正态分布
- $\nu$ 越小，尾部越厚，极端事件越多

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(-6, 6, 300)

plt.figure(figsize=(10, 5))

# 对比不同分布的尾部
plt.subplot(1, 2, 1)
plt.plot(x, stats.norm.pdf(x), 'b-', linewidth=2, label='正态分布')
for df in [3, 5, 10]:
    plt.plot(x, stats.t.pdf(x, df=df), '--', linewidth=1.5, label=f't(df={df})')
plt.title('中心区域对比')
plt.legend()
plt.xlim(-4, 4)

# 对数纵轴看尾部
plt.subplot(1, 2, 2)
plt.semilogy(x, stats.norm.pdf(x), 'b-', linewidth=2, label='正态分布')
for df in [3, 5, 10]:
    plt.semilogy(x, stats.t.pdf(x, df=df), '--', linewidth=1.5, label=f't(df={df})')
plt.title('对数坐标下的尾部（重要！）')
plt.legend()
plt.xlim(0, 6)
plt.tight_layout()
plt.show()

# 拟合真实数据的最优 t 分布自由度
np.random.seed(42)
real_returns = stats.t.rvs(df=5, loc=0.0003, scale=0.015, size=500)

df_fit, loc_fit, scale_fit = stats.t.fit(real_returns)
print(f"\n拟合结果：df={df_fit:.2f}, μ={loc_fit:.6f}, σ={scale_fit:.6f}")
print(f"金融数据的典型自由度范围：3-7")
```

---

## 三、对数正态分布：股票价格的分布

**股票价格不能为负** → 对数收益率用正态分布，价格用对数正态分布

$$\text{如果} \ln P_t \sim \mathcal{N}(\mu, \sigma^2), \text{则} P_t \sim \text{LogNormal}(\mu, \sigma^2)$$

```python
# 模拟 GBM（几何布朗运动）：期权定价的基础
def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    returns = np.random.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt), 
                               size=(n_steps, n_paths))
    price_paths = S0 * np.exp(np.cumsum(returns, axis=0))
    price_paths = np.vstack([np.full(n_paths, S0), price_paths])
    return price_paths

S0 = 100      # 初始价格
mu = 0.08     # 年化期望收益 8%
sigma = 0.20  # 年化波动率 20%
T = 1.0       # 1年
n_steps = 252 # 日频

paths = simulate_gbm(S0, mu, sigma, T, n_steps, 1000)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(paths[:, :50], alpha=0.3, linewidth=0.5)  # 只画50条
plt.axhline(y=S0, color='black', linewidth=1, linestyle='--')
plt.title('GBM 价格路径（1000条）')
plt.xlabel('交易日')
plt.ylabel('价格')

plt.subplot(1, 2, 2)
final_prices = paths[-1]
plt.hist(final_prices, bins=50, density=True, edgecolor='black')
plt.axvline(x=final_prices.mean(), color='r', linestyle='--', label=f'均值={final_prices.mean():.0f}')
plt.axvline(x=np.percentile(final_prices, 5), color='orange', linestyle='--', label=f'5%分位={np.percentile(final_prices,5):.0f}')
plt.title('1年后价格分布（对数正态）')
plt.legend()
plt.tight_layout()
plt.show()

print(f"\n1年后预期价格: {final_prices.mean():.2f}")
print(f"中位数: {np.median(final_prices):.2f}")
print(f"5% VaR: 亏损超过 {(1 - np.percentile(final_prices,5)/S0)*100:.1f}% 的概率为5%")
```

---

## 四、幂律分布：真正的肥尾

金融市场的极端事件遵循**幂律**（Power Law），而不是正态分布：

$$P(X > x) \sim x^{-\alpha}$$

$\alpha$ 越小，尾部越厚，极端事件越频繁。

```python
# 实证：分析大幅涨跌的频率
np.random.seed(42)
returns = np.concatenate([
    np.random.normal(0, 0.01, 5000),    # 正常日
    np.random.normal(0, 0.05, 200),     # 波动放大日
    np.random.normal(-0.03, 0.08, 50),  # 危机日
])

# 统计不同幅度以上的亏损频率（检验幂律）
thresholds = np.linspace(0.01, 0.10, 20)
prob_exceeds = [np.mean(np.abs(returns) > t) for t in thresholds]

plt.figure(figsize=(8, 5))
plt.loglog(thresholds, prob_exceeds, 'bo-', markersize=5)
plt.xlabel('阈值（日收益率绝对值）')
plt.ylabel('超过阈值的概率')
plt.title('对数坐标下的尾部分布（斜率 ≈ -α，直线=幂律）')
plt.grid(True)
plt.show()
```

---

## 今天的关键认识

1. **正态分布低估了极端风险**，1987年股灾等事件正态假设下"不可能"发生
2. **t 分布**（自由度3-7）更符合金融数据的实际分布
3. **价格是对数正态，收益率接近正态**（但有肥尾）
4. **幂律尾部**：真正的大崩盘比正态分布预测的频繁得多

---

## 明天预告

D3：**假设检验**——怎么用统计学判断一个因子是否真的有效？
