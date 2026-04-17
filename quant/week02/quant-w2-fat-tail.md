# D3：胖尾分布与厚尾统计

> **Week 2 · Day 3** | 量化金融基础系列

---

## 1. 什么是胖尾？

**胖尾（Fat Tail / Heavy Tail）** 是指分布的尾部概率衰减比正态分布慢得多，极端事件发生的频率远超正态假设的预期。

正态分布的尾部以指数速度衰减：

$$
f(x) \propto e^{-x^2/2}, \quad |x| \to \infty
$$

胖尾分布（如幂律分布）的尾部以代数速度衰减：

$$
f(x) \propto |x|^{-(\alpha+1)}, \quad |x| \to \infty, \quad \alpha > 0
$$

当 $\alpha \leq 2$ 时，方差甚至不存在；当 $\alpha \leq 1$ 时，均值也不存在。

---

## 2. 几种常见胖尾分布

### 2.1 Student t 分布

$$
f(x;\nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}
$$

参数 $\nu$（自由度）越小，尾部越厚：
- $\nu = 1$：Cauchy 分布（无均值，无方差）
- $\nu = 3$：常用于建模金融收益率
- $\nu \to \infty$：趋向正态分布

超额峰度：$\kappa = \dfrac{6}{\nu - 4}$（$\nu > 4$）

### 2.2 广义误差分布（GED / Power Exponential）

$$
f(x;\mu,\sigma,\nu) = \frac{\nu \exp\left(-\frac{1}{2}\left|\frac{x-\mu}{\sigma}\right|^{\nu}\right)}{2^{1+1/\nu}\sigma\Gamma(1/\nu)}
$$

- $\nu = 2$：正态分布
- $\nu < 2$：胖尾（如 $\nu = 1$ 为双指数/Laplace 分布）
- $\nu > 2$：细尾（超正态）

### 2.3 Pareto 分布与幂律

$$
P(X > x) = \left(\frac{x_m}{x}\right)^{\alpha}, \quad x \geq x_m
$$

在极端损失的尾部，Pareto 分布是最常用的建模工具（极值理论基础）。

### 2.4 正态逆高斯分布（NIG）

$$
f(x;\alpha,\beta,\mu,\delta) = \frac{\alpha\delta}{\pi} e^{\delta\sqrt{\alpha^2-\beta^2}+\beta(x-\mu)} \cdot \frac{K_1(\alpha\sqrt{\delta^2+(x-\mu)^2})}{\sqrt{\delta^2+(x-\mu)^2}}
$$

其中 $K_1$ 为修正 Bessel 函数。NIG 分布非常灵活，可以同时捕捉偏度和峰度，被广泛用于期权定价。

---

## 3. 胖尾的量化刻画

### 3.1 超额峰度

$$
\kappa = E\left[\left(\frac{X-\mu}{\sigma}\right)^4\right] - 3
$$

实证中，日收益率的超额峰度通常在 3~10 之间。

### 3.2 尾部指数（Tail Index）

通过 Hill 估计量估计幂律尾部的衰减速度：

$$
\hat{\alpha}_{Hill} = \frac{1}{\frac{1}{k}\sum_{i=1}^{k}\ln X_{(n-i+1)} - \ln X_{(n-k)}}
$$

其中 $X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$ 为排序统计量，$k$ 为阈值。

**经验参考**：金融资产的 $\hat{\alpha}$ 通常在 2.5~4.5 之间。

### 3.3 尾部概率比较

设 5σ 事件的理论概率：

| 分布 | 5σ 事件概率 |
|------|-----------|
| 正态 | $\approx 2.87 \times 10^{-7}$（约57年一遇/日） |
| t(5) | $\approx 5.2 \times 10^{-4}$（约5年一遇/日） |
| t(3) | $\approx 3.2 \times 10^{-3}$（约1年一遇/日） |

这就是为什么正态假设会严重低估极端风险。

---

## 4. 极值理论（EVT）简介

极值理论提供了描述分布尾部的严格数学框架。

### Fisher-Tippett-Gnedenko 定理

设 $M_n = \max(X_1,\ldots,X_n)$，若存在归一化常数 $a_n > 0, b_n$ 使得：

$$
P\left(\frac{M_n - b_n}{a_n} \leq x\right) \to H(x)
$$

则 $H$ 必属于以下三类广义极值分布（GEV）之一：

$$
H_\xi(x) = \exp\left\{-\left[1+\xi\left(\frac{x-\mu}{\sigma}\right)\right]^{-1/\xi}\right\}
$$

- $\xi > 0$：Fréchet（厚尾，如幂律）
- $\xi = 0$：Gumbel（薄尾，如正态、指数）
- $\xi < 0$：Weibull（有界尾）

金融资产通常属于 Fréchet 域（$\xi > 0$）。

---

## 5. 代码实战

### 5.1 分布对比可视化

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import t, norm, laplace
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n = 100000

# 生成不同分布的样本
normal_data = norm.rvs(0, 1, n)
t3_data = t.rvs(3, 0, 1, n)
t5_data = t.rvs(5, 0, 1, n)
laplace_data = laplace.rvs(0, 1/np.sqrt(2), n)  # 等方差

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 密度对比（对数坐标）
x = np.linspace(-8, 8, 1000)
axes[0].semilogy(x, norm.pdf(x), 'b-', label='正态', linewidth=2)
axes[0].semilogy(x, t.pdf(x, 3), 'r-', label='t(df=3)', linewidth=2)
axes[0].semilogy(x, t.pdf(x, 5), 'g-', label='t(df=5)', linewidth=2)
axes[0].semilogy(x, laplace.pdf(x, 0, 1/np.sqrt(2)), 'm-', label='Laplace', linewidth=2)
axes[0].set_xlim(-8, 8)
axes[0].set_ylim(1e-10, 1)
axes[0].set_title('密度函数对比（对数坐标）')
axes[0].set_xlabel('x (σ单位)')
axes[0].set_ylabel('对数密度')
axes[0].legend()
axes[0].axvline(x=3, color='gray', linestyle='--', alpha=0.5, label='3σ')

# 尾部超越概率对比
thresholds = np.linspace(1, 8, 200)
axes[1].semilogy(thresholds, norm.sf(thresholds), 'b-', label='正态', linewidth=2)
axes[1].semilogy(thresholds, t.sf(thresholds, 3), 'r-', label='t(df=3)', linewidth=2)
axes[1].semilogy(thresholds, t.sf(thresholds, 5), 'g-', label='t(df=5)', linewidth=2)
axes[1].semilogy(thresholds, laplace.sf(thresholds, 0, 1/np.sqrt(2)), 'm-', label='Laplace', linewidth=2)
axes[1].set_title('右尾超越概率 P(X > x)')
axes[1].set_xlabel('x (σ单位)')
axes[1].set_ylabel('超越概率（对数）')
axes[1].legend()

plt.tight_layout()
plt.savefig('fat_tail_comparison.png', dpi=150)
plt.show()
```

### 5.2 拟合 t 分布到收益率

```python
# 模拟胖尾收益率
np.random.seed(2024)
returns = t.rvs(df=4, loc=0.0003, scale=0.012, size=2520)

# 分别用正态和 t 分布拟合
mu_norm, std_norm = norm.fit(returns)
df_t, loc_t, scale_t = t.fit(returns)

print("=== 分布拟合结果 ===")
print(f"正态拟合: μ={mu_norm:.6f}, σ={std_norm:.6f}")
print(f"t 分布拟合: df={df_t:.2f}, loc={loc_t:.6f}, scale={scale_t:.6f}")
print(f"隐含超额峰度（t分布）: {6/(df_t-4):.4f}" if df_t > 4 else "df≤4，超额峰度不存在")

# 对数似然对比（模型选择）
ll_norm = norm.logpdf(returns, mu_norm, std_norm).sum()
ll_t = t.logpdf(returns, df_t, loc_t, scale_t).sum()
print(f"\n对数似然（正态）: {ll_norm:.2f}")
print(f"对数似然（t分布）: {ll_t:.2f}")
print(f"LR统计量（t分布更优）: {2*(ll_t - ll_norm):.2f}（临界值χ²(1)=3.84）")
```

### 5.3 Hill 估计量（尾部指数）

```python
def hill_estimator(data, k_range=None):
    """
    Hill 估计量：估计幂律尾部指数 α
    data: 正数损失数据（取绝对值）
    k_range: 阈值 k 的范围
    """
    sorted_data = np.sort(np.abs(data))[::-1]  # 从大到小排序
    n = len(sorted_data)
    
    if k_range is None:
        k_range = range(10, min(500, n//4))
    
    hill_estimates = []
    for k in k_range:
        log_ratios = np.log(sorted_data[:k]) - np.log(sorted_data[k])
        alpha_hat = k / log_ratios.sum()
        hill_estimates.append(alpha_hat)
    
    return list(k_range), hill_estimates

# 应用到模拟数据（极端损失）
losses = np.abs(returns[returns < 0])
k_vals, alpha_hats = hill_estimator(losses)

# Hill 图
plt.figure(figsize=(10, 5))
plt.plot(k_vals, alpha_hats, 'steelblue', linewidth=1.5)
plt.axhline(y=np.mean(alpha_hats[50:150]), color='red', linestyle='--', 
            label=f'稳定区间均值≈{np.mean(alpha_hats[50:150]):.2f}')
plt.xlabel('阈值 k')
plt.ylabel('尾部指数估计 α̂')
plt.title('Hill 图：尾部指数估计')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('hill_plot.png', dpi=150)
plt.show()

print(f"\n尾部指数估计（稳定区间）: α ≈ {np.mean(alpha_hats[50:150]):.3f}")
print(f"参考：α∈(2,4) 为典型金融厚尾区间")
```

### 5.4 极值理论：广义帕累托分布（GPD）拟合

```python
from scipy.stats import genpareto

# 超阈值方法（POT - Peaks over Threshold）
threshold = np.percentile(np.abs(returns), 95)  # 95%分位数为阈值
exceedances = np.abs(returns[np.abs(returns) > threshold]) - threshold

# 拟合 GPD
shape, loc_gpd, scale_gpd = genpareto.fit(exceedances, floc=0)

print(f"=== 广义帕累托分布（GPD）拟合 ===")
print(f"阈值 u: {threshold:.6f}")
print(f"超阈值样本数: {len(exceedances)}")
print(f"形状参数 ξ: {shape:.4f}  （>0 表示厚尾 Fréchet 域）")
print(f"尺度参数 β: {scale_gpd:.6f}")

# 用 GPD 估计极端分位数（99.9% VaR）
# P(X > u + y | X > u) = (1 + ξy/β)^(-1/ξ)
n_total = len(returns)
n_exceed = len(exceedances)
p_exceed = n_exceed / n_total

def gpd_var(p, u, shape, scale, p_exceed):
    """用 GPD 估计超高分位数 VaR"""
    # 转换为超阈值概率
    p_cond = (1 - p) / p_exceed
    if shape != 0:
        quantile = u + scale / shape * ((p_cond)**(-shape) - 1)
    else:
        quantile = u - scale * np.log(p_cond)
    return quantile

var_999_gpd = gpd_var(0.999, threshold, shape, scale_gpd, p_exceed)
var_999_norm = norm.ppf(0.999, loc=returns.mean(), scale=returns.std())

print(f"\n99.9% VaR 对比（损失侧）：")
print(f"  正态分布: {-var_999_norm:.6f}")
print(f"  GPD (EVT): {var_999_gpd:.6f}")
print(f"  EVT比正态高: {(var_999_gpd/(-var_999_norm)-1)*100:.1f}%")
```

### 5.5 收益率分布的矩统计

```python
# 系统比较不同分布的理论矩
print("=== 不同分布的超额峰度（理论值）===")
distributions = {
    '正态': lambda: 0,
    't(df=10)': lambda: 6/(10-4),
    't(df=5)': lambda: 6/(5-4),
    't(df=4)': lambda: float('inf'),
    'Laplace': lambda: 3,
    'Uniform': lambda: -6/5
}
for name, kurtosis_fn in distributions.items():
    try:
        k = kurtosis_fn()
        print(f"  {name:12s}: κ = {k:.4f}")
    except:
        print(f"  {name:12s}: κ = ∞")
```

---

## 6. 胖尾对量化策略的影响

| 影响维度 | 正态假设错误的后果 |
|----------|-------------------|
| VaR 计算 | 系统性低估尾部损失 |
| 期权定价 | 虚值期权价格被低估 |
| 组合优化 | 极端场景被忽视 |
| 止损设置 | 止损过近，频繁触发 |
| 压力测试 | 场景不够极端 |

---

## 7. 关键公式汇总

$$
\text{t分布密度: } f(x;\nu) \propto \left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}
$$

$$
\text{超额峰度(t): } \kappa = \frac{6}{\nu-4}, \quad \nu > 4
$$

$$
\text{GPD: } F(y) = 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi}, \quad y > 0
$$

$$
\boxed{\text{Hill估计量: } \hat{\alpha} = \left[\frac{1}{k}\sum_{i=1}^{k}\ln\frac{X_{(n-i+1)}}{X_{(n-k)}}\right]^{-1}}
$$

---

## 8. 小结

- 金融收益率普遍存在胖尾特征，正态分布对尾部风险的描述严重不足
- t 分布是最常用的胖尾替代分布，自由度越低尾部越厚
- 极值理论（GPD）提供了对极端事件更精确的统计描述
- Hill 估计量可以从样本中直接估计幂律尾部指数
- 在实际应用中，胖尾假设应贯穿 VaR 计算、压力测试和期权定价全流程

> 下一篇：D4 风险度量（VaR/CVaR/最大回撤）——从理论到实战计算三大风险指标
