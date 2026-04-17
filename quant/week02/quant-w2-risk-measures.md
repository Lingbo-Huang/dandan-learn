# D4：风险度量（VaR / CVaR / 最大回撤）

> **Week 2 · Day 4** | 量化金融基础系列

---

## 1. 为什么需要风险度量？

"赚多少"是收益，"最坏情况亏多少"是风险。量化风险有三个目的：
1. **资本配置**：监管要求机构持有与风险匹配的资本（巴塞尔协议）
2. **头寸管理**：实时监控持仓风险，触发止损
3. **策略评估**：比较不同策略的风险调整后收益（Sharpe、Calmar 等）

---

## 2. VaR（Value at Risk）

### 2.1 定义

在给定置信水平 $\alpha$ 下，持有期 $T$ 内，**损失不超过 VaR 的概率为 $\alpha$**：

$$
P(L \leq \text{VaR}_\alpha) = \alpha
$$

等价地，损失**超过** VaR 的概率为 $1 - \alpha$：

$$
\text{VaR}_\alpha = -Q_\alpha(r) = -F^{-1}(1-\alpha)
$$

其中 $Q_\alpha(r)$ 是收益率分布的 $\alpha$ 分位数（$\alpha = 1\%$ 时取左尾 1% 分位数）。

**示例**：1日 99% VaR = 2%，意味着：有 99% 的概率，明天的损失不超过 2%。

### 2.2 三种计算方法

#### 方法一：历史模拟法

$$
\text{VaR}_\alpha^{hist} = -\hat{Q}_{1-\alpha}(r_1, r_2, \ldots, r_T)
$$

直接取历史收益率序列的 $(1-\alpha)$ 分位数，无分布假设，但受样本窗口影响大。

#### 方法二：参数法（正态假设）

$$
\text{VaR}_\alpha^{norm} = -\left(\mu - z_{1-\alpha} \cdot \sigma\right)
$$

其中 $z_{1-\alpha}$ 为标准正态分布的 $(1-\alpha)$ 分位数（99% VaR 对应 $z = 2.326$）。

#### 方法三：Monte Carlo 模拟法

基于参数模型模拟大量路径，从模拟收益率分布计算 VaR：

$$
\text{VaR}^{MC} = -\hat{Q}_{1-\alpha}\left(r_1^{sim}, r_2^{sim}, \ldots, r_N^{sim}\right)
$$

适合非线性产品（期权、结构化产品）。

### 2.3 VaR 的局限

VaR 的最大批评：它**不是相干风险度量（Coherent Risk Measure）**，不满足次可加性：

$$
\text{VaR}(A + B) > \text{VaR}(A) + \text{VaR}(B) \quad \text{可能成立}
$$

这意味着分散化投资可能"反而增加"VaR，违背直觉。

---

## 3. CVaR（条件风险价值 / Expected Shortfall）

### 3.1 定义

CVaR（也称 ES 或 Expected Shortfall）是在**损失超过 VaR 时的条件期望损失**：

$$
\text{CVaR}_\alpha = E[L \mid L > \text{VaR}_\alpha] = -E[r \mid r < -\text{VaR}_\alpha]
$$

积分形式：

$$
\text{CVaR}_\alpha = -\frac{1}{1-\alpha} \int_{-\infty}^{\text{VaR}_\alpha} r \cdot f(r)\, dr
$$

### 3.2 正态分布下的解析公式

$$
\text{CVaR}_\alpha^{norm} = \mu + \sigma \cdot \frac{\phi(z_{1-\alpha})}{1-\alpha}
$$

其中 $\phi(\cdot)$ 为标准正态密度函数。

例：99% CVaR 系数为 $\phi(2.326)/0.01 \approx 2.665$。

### 3.3 CVaR 的优势

CVaR 满足**相干风险度量**的四个公理（单调性、正齐次性、平移不变性、**次可加性**），因此：
- CVaR(A+B) ≤ CVaR(A) + CVaR(B)（分散化有益）
- 已被巴塞尔委员会（FRTB）采纳，取代 VaR 作为银行市场风险的主要度量

---

## 4. 最大回撤（Maximum Drawdown）

### 4.1 定义

最大回撤（MDD）描述从**历史最高点到随后最低点**的最大跌幅：

$$
\text{MDD}(T) = \max_{0 \leq t_1 \leq t_2 \leq T} \frac{P_{t_1} - P_{t_2}}{P_{t_1}}
$$

等价于：

$$
\text{MDD} = \max_{t} \left(1 - \frac{P_t}{\max_{s \leq t} P_s}\right)
$$

### 4.2 回撤序列

当前回撤（Drawdown at time t）：

$$
DD_t = \frac{\max_{s \leq t} P_s - P_t}{\max_{s \leq t} P_s}
$$

MDD 即 $DD_t$ 的最大值。

### 4.3 相关指标

| 指标 | 公式 | 含义 |
|------|------|------|
| 最大回撤 | $\text{MDD}$ | 最坏情况下的净值跌幅 |
| 平均回撤 | $E[DD_t]$ | 日均回撤水平 |
| 回撤持续时间 | $t_2 - t_1$ | 从高点到低点的时间 |
| 恢复时间 | $t_3 - t_1$ | 从高点跌落再恢复的时间 |
| Calmar Ratio | $\dfrac{\text{年化收益率}}{\text{MDD}}$ | 每单位最大回撤的年化收益 |

---

## 5. 风险度量的多期扩展

### 时间缩放规则（Square-Root-of-Time Rule）

在i.i.d.正态假设下，$T$ 日 VaR 可从 1 日 VaR 推算：

$$
\text{VaR}_{T} = \text{VaR}_{1} \times \sqrt{T}
$$

**注意**：当收益率存在序列相关或波动率聚集时，此规则失效。

---

## 6. 代码实战

### 6.1 三种方法计算 VaR

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm, t

np.random.seed(42)

# 模拟收益率：胖尾场景（t分布，df=5）
n = 2520  # 10年日数据
returns = t.rvs(df=5, loc=0.0003, scale=0.012, size=n)

confidence_levels = [0.95, 0.99, 0.999]

print("=== VaR 三种方法对比 ===")
print(f"{'置信水平':>8} | {'历史模拟':>10} | {'正态参数':>10} | {'Monte Carlo':>12}")
print("-" * 50)

for alpha in confidence_levels:
    # 历史模拟法
    var_hist = -np.percentile(returns, (1 - alpha) * 100)
    
    # 正态参数法
    mu_hat, sigma_hat = returns.mean(), returns.std()
    var_norm = -(mu_hat + norm.ppf(1 - alpha) * sigma_hat)
    
    # Monte Carlo（t分布）
    df_fit, loc_fit, scale_fit = t.fit(returns)
    mc_sim = t.rvs(df_fit, loc_fit, scale_fit, size=100000)
    var_mc = -np.percentile(mc_sim, (1 - alpha) * 100)
    
    print(f"{alpha:>8.1%} | {var_hist:>10.4f} | {var_norm:>10.4f} | {var_mc:>12.4f}")
```

### 6.2 CVaR 计算

```python
def calculate_cvar(returns, alpha):
    """计算 CVaR（历史模拟法）"""
    var = -np.percentile(returns, (1 - alpha) * 100)
    tail_losses = -returns[returns < -var]
    cvar = tail_losses.mean() if len(tail_losses) > 0 else var
    return var, cvar

def cvar_normal(mu, sigma, alpha):
    """正态分布下 CVaR 解析公式"""
    z = norm.ppf(1 - alpha)
    var = -(mu + z * sigma)
    cvar = -(mu - sigma * norm.pdf(z) / (1 - alpha))
    return var, cvar

print("\n=== VaR vs CVaR 对比 ===")
print(f"{'置信水平':>8} | {'VaR(历史)':>10} | {'CVaR(历史)':>12} | {'CVaR/VaR':>10}")
print("-" * 48)

for alpha in [0.95, 0.99, 0.999]:
    var_h, cvar_h = calculate_cvar(returns, alpha)
    print(f"{alpha:>8.1%} | {var_h:>10.4f} | {cvar_h:>12.4f} | {cvar_h/var_h:>10.3f}")

# 对比正态 vs 历史（胖尾的影响）
print(f"\n=== 胖尾对 CVaR 的影响（99%置信） ===")
var_n, cvar_n = cvar_normal(returns.mean(), returns.std(), 0.99)
var_h, cvar_h = calculate_cvar(returns, 0.99)
print(f"正态假设 CVaR: {cvar_n:.6f}")
print(f"历史模拟 CVaR: {cvar_h:.6f}")
print(f"低估幅度: {(cvar_h/cvar_n - 1)*100:.1f}%")
```

### 6.3 最大回撤计算

```python
def max_drawdown(returns):
    """计算最大回撤及相关指标"""
    # 净值序列
    nav = (1 + pd.Series(returns)).cumprod()
    
    # 历史最高净值
    rolling_max = nav.cummax()
    
    # 当前回撤
    drawdown = (nav - rolling_max) / rolling_max
    
    # 最大回撤
    mdd = drawdown.min()
    
    # 最大回撤发生时刻
    end_idx = drawdown.idxmin()
    start_idx = nav[:end_idx].idxmax()
    
    # 恢复时刻（如果已恢复）
    recovery_idx = None
    nav_after = nav[end_idx:]
    peak_value = rolling_max[end_idx]
    recovered = nav_after[nav_after >= peak_value]
    if len(recovered) > 0:
        recovery_idx = recovered.index[0]
    
    return {
        'mdd': mdd,
        'nav': nav,
        'drawdown': drawdown,
        'rolling_max': rolling_max,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'recovery_idx': recovery_idx
    }

result = max_drawdown(returns)

print(f"=== 最大回撤分析 ===")
print(f"最大回撤: {result['mdd']:.4f} ({result['mdd']*100:.2f}%)")
print(f"回撤开始: Day {result['start_idx']}")
print(f"回撤最深: Day {result['end_idx']}")
if result['recovery_idx']:
    print(f"恢复时间: Day {result['recovery_idx']}")
    print(f"回撤持续: {result['end_idx'] - result['start_idx']} 天")
    print(f"总恢复用时: {result['recovery_idx'] - result['start_idx']} 天")
else:
    print("尚未从最大回撤中恢复")

# Calmar Ratio
annual_return = (1 + returns.mean()) ** 252 - 1
calmar = annual_return / abs(result['mdd'])
print(f"\n年化收益率: {annual_return*100:.2f}%")
print(f"Calmar Ratio: {calmar:.4f}")
```

### 6.4 可视化

```python
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 净值曲线
nav = result['nav']
axes[0].plot(nav.values, color='steelblue', linewidth=1.2, label='净值')
axes[0].plot(result['rolling_max'].values, color='gray', linewidth=0.8, 
             linestyle='--', label='历史最高')
axes[0].axvline(result['start_idx'], color='orange', linestyle=':', alpha=0.8)
axes[0].axvline(result['end_idx'], color='red', linestyle=':', alpha=0.8, label='最大回撤')
axes[0].set_title('净值曲线与历史最高点')
axes[0].legend()
axes[0].set_ylabel('净值')

# 回撤序列
axes[1].fill_between(range(len(result['drawdown'])), 
                      result['drawdown'].values, 0, 
                      color='red', alpha=0.4)
axes[1].plot(result['drawdown'].values, color='darkred', linewidth=0.8)
axes[1].axhline(result['mdd'], color='red', linestyle='--', 
                label=f'最大回撤 {result["mdd"]*100:.2f}%')
axes[1].set_title('回撤序列')
axes[1].set_ylabel('回撤幅度')
axes[1].legend()

# 收益率分布 + VaR/CVaR 标注
var_99, cvar_99 = calculate_cvar(returns, 0.99)
axes[2].hist(returns, bins=60, density=True, color='steelblue', alpha=0.7, edgecolor='white')
axes[2].axvline(-var_99, color='orange', linewidth=2, label=f'99% VaR: {var_99:.4f}')
axes[2].axvline(-cvar_99, color='red', linewidth=2, label=f'99% CVaR: {cvar_99:.4f}')
x_range = np.linspace(returns.min(), returns.max(), 300)
axes[2].plot(x_range, norm.pdf(x_range, returns.mean(), returns.std()), 
             'g--', linewidth=2, label='正态拟合')
axes[2].set_title('收益率分布与风险度量')
axes[2].set_xlabel('日收益率')
axes[2].set_ylabel('密度')
axes[2].legend()

plt.tight_layout()
plt.savefig('risk_measures.png', dpi=150)
plt.show()
```

### 6.5 滚动 VaR 监控

```python
# 滚动 VaR（250日窗口）
window = 250
rolling_var_95 = pd.Series(returns).rolling(window).apply(
    lambda x: -np.percentile(x, 5), raw=True
)
rolling_var_99 = pd.Series(returns).rolling(window).apply(
    lambda x: -np.percentile(x, 1), raw=True
)

# 统计 VaR 穿越次数（回测检验）
ret_series = pd.Series(returns)
violations_95 = (-ret_series[window:] > rolling_var_95[window:]).sum()
violations_99 = (-ret_series[window:] > rolling_var_99[window:]).sum()
total = len(returns) - window

print("=== 滚动 VaR 回测（Backtesting）===")
print(f"95% VaR: 超出 {violations_95} 次 / {total} 天 "
      f"= {violations_95/total*100:.2f}%（期望 5%）")
print(f"99% VaR: 超出 {violations_99} 次 / {total} 天 "
      f"= {violations_99/total*100:.2f}%（期望 1%）")
```

---

## 7. 关键公式汇总

$$
\boxed{\text{VaR}_\alpha = -Q_{1-\alpha}(r)}
$$

$$
\boxed{\text{CVaR}_\alpha = -E[r \mid r \leq -\text{VaR}_\alpha]}
$$

$$
\text{CVaR}_\alpha^{norm} = -\mu + \sigma \cdot \frac{\phi(z_{1-\alpha})}{1-\alpha}
$$

$$
\text{MDD} = \max_{t_1 \leq t_2} \frac{P_{t_1} - P_{t_2}}{P_{t_1}}, \quad \text{Calmar} = \frac{R_{annual}}{|\text{MDD}|}
$$

---

## 8. 小结

- **VaR** 是最广泛使用的风险度量，但不满足次可加性，且忽略尾部形态
- **CVaR（ES）** 是 VaR 的改进，捕捉超额损失的期望，已成为监管主流
- **最大回撤** 是策略评估中最直观的风险指标，Calmar Ratio 结合了收益与回撤
- 三者应配合使用，无法互相替代

> 下一篇：D5 资产间相关性分析（Pearson/Spearman/滚动相关）——理解资产之间的联动关系
