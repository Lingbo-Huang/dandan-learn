# D1：对数收益率定义与性质

> **Week 2 · Day 1** | 量化金融基础系列

---

## 1. 为什么要研究收益率？

价格本身是非平稳序列，难以跨资产、跨时段比较。收益率将价格变化**标准化**为百分比形式，是量化分析的基本单元。常用的收益率有两种：

| 类型 | 定义 | 英文名 |
|------|------|--------|
| 简单收益率 | $R_t = \dfrac{P_t - P_{t-1}}{P_{t-1}}$ | Simple Return |
| 对数收益率 | $r_t = \ln\dfrac{P_t}{P_{t-1}} = \ln P_t - \ln P_{t-1}$ | Log Return |

---

## 2. 对数收益率的数学定义

设资产价格序列为 $\{P_t\}$，对数收益率定义为：

$$
r_t = \ln P_t - \ln P_{t-1} = \ln\left(1 + R_t\right)
$$

其中 $R_t$ 为简单收益率。

**泰勒展开关系**（当 $R_t$ 较小时）：

$$
r_t = \ln(1 + R_t) \approx R_t - \frac{R_t^2}{2} + \frac{R_t^3}{3} - \cdots \approx R_t
$$

这意味着在日频或更高频的场景中，两者数值极为接近。

---

## 3. 对数收益率的核心性质

### 3.1 时间可加性（最重要性质）

对数收益率在时间维度上可以直接相加：

$$
r_{t_1 \to t_n} = \sum_{i=1}^{n} r_{t_i} = \ln\frac{P_{t_n}}{P_{t_0}}
$$

而简单收益率的多期合并需要连乘：

$$
1 + R_{t_1 \to t_n} = \prod_{i=1}^{n}(1 + R_{t_i})
$$

**这一性质使得对数收益率天然适合时序建模和统计分析。**

### 3.2 对称性

对数收益率关于涨跌是对称的。若价格上涨 100%，对数收益率为 $\ln 2 \approx 69.3\%$；若价格下跌 50%（即从 2 回到 1），对数收益率为 $-\ln 2 \approx -69.3\%$。简单收益率则分别是 $+100\%$ 和 $-50\%$，不对称。

### 3.3 近似正态分布

在资产价格服从几何布朗运动（GBM）的假设下，对数收益率服从正态分布：

$$
r_t \sim \mathcal{N}(\mu \Delta t,\, \sigma^2 \Delta t)
$$

其中 $\mu$ 为漂移率，$\sigma$ 为波动率，$\Delta t$ 为时间步长。

### 3.4 下界为 $-\infty$（价格非负保证）

由于 $P_t > 0$，$\ln P_t \in (-\infty, +\infty)$，而 $P_t = P_0 \cdot e^{r_{1}+r_2+\cdots+r_t}$，无论对数收益率取何值，价格始终为正，自然规避了"价格为负"的数学问题。

### 3.5 跨资产可比性

通过对数收益率，不同价格量级的资产（如 1 元股票 vs 500 元股票）可以在同一尺度上比较波动幅度。

---

## 4. 简单收益率 vs 对数收益率：何时用哪个？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 多期收益率合并 | 对数收益率 | 可加性 |
| 截面比较（同一时点多资产） | 简单收益率 | 线性可加（组合收益率） |
| 统计建模、时序分析 | 对数收益率 | 近似正态，方差平稳 |
| 实际盈亏计算 | 简单收益率 | 直觉清晰，等同 PnL |

**注意**：对数收益率的组合性质较弱——组合的对数收益率 ≠ 各资产对数收益率的加权和（而简单收益率满足此性质）。这是使用对数收益率时需要注意的局限。

---

## 5. 代码实战

### 5.1 数据准备与收益率计算

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 模拟生成价格序列（几何布朗运动）
np.random.seed(42)
n = 252  # 一年交易日
mu = 0.0003      # 日漂移
sigma = 0.015    # 日波动率
dt = 1

# 生成对数收益率
log_returns_true = np.random.normal(mu, sigma, n)

# 从收益率还原价格序列
P0 = 100
prices = P0 * np.exp(np.cumsum(log_returns_true))
prices = np.insert(prices, 0, P0)

# 计算两种收益率
price_series = pd.Series(prices, name='Price')
simple_ret = price_series.pct_change().dropna()
log_ret = np.log(price_series / price_series.shift(1)).dropna()

print("=== 收益率基本统计 ===")
print(f"简单收益率 均值: {simple_ret.mean():.6f}, 标准差: {simple_ret.std():.6f}")
print(f"对数收益率 均值: {log_ret.mean():.6f}, 标准差: {log_ret.std():.6f}")
print(f"\n两者差异（均值）: {abs(simple_ret.mean() - log_ret.mean()):.8f}")
```

### 5.2 时间可加性验证

```python
# 验证对数收益率的时间可加性
monthly_log = log_ret[:21].sum()  # 前21个交易日的对数收益率之和
monthly_price_ratio = prices[21] / prices[0]
print(f"\n=== 时间可加性验证 ===")
print(f"前21日对数收益率之和: {monthly_log:.6f}")
print(f"ln(P_21/P_0)        : {np.log(monthly_price_ratio):.6f}")
print(f"完全相等: {np.isclose(monthly_log, np.log(monthly_price_ratio))}")

# 简单收益率的多期合并需要连乘
monthly_simple = (1 + simple_ret[:21]).prod() - 1
print(f"\n前21日简单收益率（连乘法）: {monthly_simple:.6f}")
print(f"对应对数收益率: {np.log(1 + monthly_simple):.6f}")
```

### 5.3 可视化对比

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 价格走势
axes[0, 0].plot(price_series.values, color='steelblue', linewidth=1.5)
axes[0, 0].set_title('模拟资产价格走势')
axes[0, 0].set_xlabel('交易日')
axes[0, 0].set_ylabel('价格')

# 两种收益率对比
axes[0, 1].plot(simple_ret.values, label='简单收益率', alpha=0.7, color='orange')
axes[0, 1].plot(log_ret.values, label='对数收益率', alpha=0.7, color='steelblue', linestyle='--')
axes[0, 1].set_title('简单收益率 vs 对数收益率')
axes[0, 1].legend()
axes[0, 1].set_xlabel('交易日')

# 对数收益率分布直方图
axes[1, 0].hist(log_ret.values, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
axes[1, 0].set_title('对数收益率分布')
axes[1, 0].set_xlabel('收益率')
axes[1, 0].set_ylabel('频次')

# 散点图：两种收益率的关系
axes[1, 1].scatter(simple_ret.values, log_ret.values, alpha=0.4, s=10, color='steelblue')
axes[1, 1].set_title('简单收益率 vs 对数收益率（散点）')
axes[1, 1].set_xlabel('简单收益率')
axes[1, 1].set_ylabel('对数收益率')
x_line = np.linspace(simple_ret.min(), simple_ret.max(), 100)
axes[1, 1].plot(x_line, x_line, 'r--', alpha=0.5, label='y=x')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('log_return_analysis.png', dpi=150)
plt.show()
print("图表已保存")
```

### 5.4 年化收益率与波动率

```python
# 年化指标计算
trading_days = 252

annual_return_log = log_ret.mean() * trading_days
annual_vol_log = log_ret.std() * np.sqrt(trading_days)

annual_return_simple = (1 + simple_ret.mean()) ** trading_days - 1
annual_vol_simple = simple_ret.std() * np.sqrt(trading_days)

print("=== 年化指标 ===")
print(f"对数收益率 年化收益: {annual_return_log:.4f} ({annual_return_log*100:.2f}%)")
print(f"对数收益率 年化波动: {annual_vol_log:.4f} ({annual_vol_log*100:.2f}%)")
print(f"简单收益率 年化收益: {annual_return_simple:.4f} ({annual_return_simple*100:.2f}%)")
print(f"简单收益率 年化波动: {annual_vol_simple:.4f} ({annual_vol_simple*100:.2f}%)")

# Sharpe Ratio（假设无风险利率 2%）
rf = 0.02
sharpe_log = (annual_return_log - rf) / annual_vol_log
print(f"\nSharpe Ratio（对数收益率法）: {sharpe_log:.4f}")
```

---

## 6. 关键公式汇总

$$
\boxed{r_t = \ln P_t - \ln P_{t-1} = \ln\left(\frac{P_t}{P_{t-1}}\right)}
$$

$$
r_{[0,T]} = \sum_{t=1}^{T} r_t \quad \text{（时间可加性）}
$$

$$
P_T = P_0 \cdot e^{r_1 + r_2 + \cdots + r_T} \quad \text{（价格还原）}
$$

$$
\hat{\mu}_{annual} = \bar{r} \times 252, \quad \hat{\sigma}_{annual} = s_r \times \sqrt{252}
$$

---

## 7. 小结

- 对数收益率是量化金融中最核心的基本概念，其**时间可加性**和**近似正态性**使其成为统计建模的首选
- 在日频数据中，对数收益率与简单收益率数值非常接近，可互相转换
- 实际应用中，**建模用对数收益率，计算组合收益用简单收益率**
- 年化时，对数收益率均值乘以 252，标准差乘以 $\sqrt{252}$

> 下一篇：D2 收益率正态假设检验（QQ图/JB检验）——实证检验"收益率是否真的正态分布"
