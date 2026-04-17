# D2：收益率正态假设检验（QQ图 / JB检验）

> **Week 2 · Day 2** | 量化金融基础系列

---

## 1. 为什么要检验正态性？

Black-Scholes 期权定价、均值-方差组合理论、VaR 参数法等经典模型都隐含一个关键假设：**资产对数收益率服从正态分布**。如果这个假设不成立，这些模型的输出就会出现系统性偏差——尤其是对尾部风险的严重低估。

检验收益率是否"真的正态"，是量化建模的第一道关口。

---

## 2. 正态分布的统计特征

正态分布 $\mathcal{N}(\mu, \sigma^2)$ 有两个关键矩特征：

| 统计量 | 定义 | 正态分布取值 |
|--------|------|------------|
| 偏度 (Skewness) | $\gamma_1 = \dfrac{E[(X-\mu)^3]}{\sigma^3}$ | $= 0$（左右对称） |
| 峰度 (Kurtosis) | $\gamma_2 = \dfrac{E[(X-\mu)^4]}{\sigma^4}$ | $= 3$（超额峰度 = 0） |

**超额峰度（Excess Kurtosis）**：$\kappa = \gamma_2 - 3$

- $\kappa > 0$：尖峰厚尾（Leptokurtic），金融收益率常见
- $\kappa < 0$：平顶薄尾（Platykurtic）
- $\kappa = 0$：与正态一致

实证中，金融资产收益率普遍表现为：**负偏（左偏）+ 显著正超额峰度**，这正是"胖尾"的体现。

---

## 3. 正态性检验方法

### 3.1 图形法：QQ 图（Quantile-Quantile Plot）

QQ 图将样本分位数与理论正态分布分位数相互对照：

$$
x_{(i)} \approx \mu + \sigma \cdot z_{\frac{i-0.5}{n}}
$$

其中 $x_{(i)}$ 是第 $i$ 个顺序统计量，$z_p$ 是标准正态的 $p$ 分位数。

- **点落在对角线上** → 数据近似正态
- **两端向上翘**（S形反向）→ 厚尾
- **系统性偏斜** → 非对称

### 3.2 统计检验：Jarque-Bera 检验

JB 检验基于偏度与峰度同时检验正态性：

$$
\text{JB} = \frac{n}{6}\left(\hat{\gamma}_1^2 + \frac{(\hat{\kappa})^2}{4}\right)
$$

其中：
- $n$ 为样本量
- $\hat{\gamma}_1$ 为样本偏度
- $\hat{\kappa} = \hat{\gamma}_2 - 3$ 为样本超额峰度

在原假设（数据来自正态分布）下，$\text{JB} \sim \chi^2(2)$，即自由度为 2 的卡方分布。

**决策规则**：若 $p < 0.05$，拒绝正态假设。

### 3.3 其他检验方法

| 方法 | 适用场景 | 特点 |
|------|----------|------|
| Shapiro-Wilk | 小样本（$n < 5000$） | 功效最高 |
| Kolmogorov-Smirnov | 大样本 | 对尾部不敏感 |
| Anderson-Darling | 通用 | 对尾部更敏感 |
| Lilliefors | 均值/方差未知 | KS 的改进版 |

---

## 4. 代码实战

### 4.1 环境准备与数据生成

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import jarque_bera, shapiro, normaltest
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# 模拟正态收益率（基准）
n = 1000
normal_returns = np.random.normal(0, 0.01, n)

# 模拟胖尾收益率（t分布，自由度=3）
fat_tail_returns = stats.t.rvs(df=3, scale=0.01, size=n)

# 模拟实际场景：混合分布（正常市场 + 偶发冲击）
shock_prob = 0.02
shocks = np.random.choice([0, 1], size=n, p=[1-shock_prob, shock_prob])
mixed_returns = np.random.normal(0, 0.01, n) + shocks * np.random.normal(0, 0.05, n)

datasets = {
    '正态分布': normal_returns,
    't分布(df=3)': fat_tail_returns,
    '混合分布': mixed_returns
}
```

### 4.2 描述性统计

```python
print("=== 描述性统计 ===\n")
for name, data in datasets.items():
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)  # 超额峰度（scipy默认返回excess kurtosis）
    print(f"【{name}】")
    print(f"  均值: {data.mean():.6f}")
    print(f"  标准差: {data.std():.6f}")
    print(f"  偏度: {skew:.4f}  （正态=0）")
    print(f"  超额峰度: {kurt:.4f}  （正态=0）")
    print()
```

### 4.3 JB 检验

```python
print("=== Jarque-Bera 检验 ===\n")
for name, data in datasets.items():
    jb_stat, jb_p = jarque_bera(data)
    n_data = len(data)
    
    # 手动计算验证
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    jb_manual = n_data / 6 * (skew**2 + kurt**2 / 4)
    
    conclusion = "❌ 拒绝正态假设" if jb_p < 0.05 else "✅ 不能拒绝正态假设"
    
    print(f"【{name}】")
    print(f"  JB统计量: {jb_stat:.4f}  (手动验证: {jb_manual:.4f})")
    print(f"  p值: {jb_p:.6f}")
    print(f"  结论: {conclusion}")
    print()
```

### 4.4 多重检验对比

```python
print("=== 多种正态性检验对比 ===\n")
for name, data in datasets.items():
    # Shapiro-Wilk（取前5000个）
    sw_stat, sw_p = shapiro(data[:5000])
    # D'Agostino-Pearson
    da_stat, da_p = normaltest(data)
    # JB
    jb_stat, jb_p = jarque_bera(data)
    
    print(f"【{name}】")
    print(f"  Shapiro-Wilk: p = {sw_p:.6f}")
    print(f"  D'Agostino-Pearson: p = {da_p:.6f}")
    print(f"  Jarque-Bera: p = {jb_p:.6f}")
    print()
```

### 4.5 QQ 图绘制

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for col, (name, data) in enumerate(datasets.items()):
    # 直方图 + 正态曲线
    ax_hist = axes[0, col]
    ax_hist.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    
    x = np.linspace(data.min(), data.max(), 200)
    normal_pdf = stats.norm.pdf(x, data.mean(), data.std())
    ax_hist.plot(x, normal_pdf, 'r-', linewidth=2, label='正态曲线')
    ax_hist.set_title(f'{name}\n偏度={stats.skew(data):.3f}, 峰度={stats.kurtosis(data):.3f}')
    ax_hist.legend(fontsize=8)
    
    # QQ 图
    ax_qq = axes[1, col]
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
    ax_qq.scatter(osm, osr, alpha=0.3, s=5, color='steelblue')
    ax_qq.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2)
    ax_qq.set_title(f'QQ图 - {name}\n$R^2$ = {r**2:.4f}')
    ax_qq.set_xlabel('理论分位数（正态）')
    ax_qq.set_ylabel('样本分位数')

plt.tight_layout()
plt.savefig('normality_test.png', dpi=150, bbox_inches='tight')
plt.show()
print("图表已保存")
```

### 4.6 实际数据场景（模拟沪深300日收益率）

```python
# 用真实金融分布特征模拟沪深300日收益率
np.random.seed(2024)
# 参数来自历史估计：年化收益约8%，年化波动约22%
daily_mu = 0.08 / 252
daily_sigma = 0.22 / np.sqrt(252)

# 混合正态模型（正常态 + 市场崩溃态）
n_sim = 2520  # 10年
regime = np.random.choice([0, 1], size=n_sim, p=[0.97, 0.03])
hs300_sim = np.where(
    regime == 0,
    np.random.normal(daily_mu, daily_sigma, n_sim),
    np.random.normal(-0.02, 0.04, n_sim)  # 崩溃态
)

# 全面检验
print("=== 模拟沪深300收益率正态性检验 ===")
print(f"样本量: {n_sim} 个交易日 (~10年)")
print(f"均值: {hs300_sim.mean():.6f}")
print(f"标准差: {hs300_sim.std():.6f}")
print(f"偏度: {stats.skew(hs300_sim):.4f}")
print(f"超额峰度: {stats.kurtosis(hs300_sim):.4f}")

jb_stat, jb_p = jarque_bera(hs300_sim)
print(f"\nJB统计量: {jb_stat:.2f}, p值: {jb_p:.2e}")
if jb_p < 0.01:
    print("结论：强烈拒绝正态假设（p < 0.01）")
    
# 计算真实VaR vs 正态假设VaR
confidence = 0.99
var_normal = stats.norm.ppf(1 - confidence, daily_mu, daily_sigma)
var_empirical = np.percentile(hs300_sim, (1 - confidence) * 100)

print(f"\n=== VaR 对比（99%置信度）===")
print(f"正态假设 VaR: {var_normal:.4f} ({var_normal*100:.2f}%)")
print(f"历史模拟 VaR: {var_empirical:.4f} ({var_empirical*100:.2f}%)")
print(f"正态法低估风险: {abs(var_normal/var_empirical - 1)*100:.1f}%")
```

---

## 5. 正态假设在金融中的局限

$$
\text{实证现象：} \quad \hat{\kappa} \gg 0 \text{ 且 } \hat{\gamma}_1 < 0
$$

**主要原因**：
1. **跳跃过程**：市场存在非连续的价格跳跃（政策冲击、黑天鹅事件）
2. **波动率聚集**：高波动时期后还有高波动（ARCH 效应）
3. **杠杆效应**：下跌时波动率往往更高（负相关）
4. **流动性冲击**：极端流动性不足导致价格瞬间剧烈波动

---

## 6. 关键公式汇总

$$
\text{Skewness: } \hat{\gamma}_1 = \frac{\frac{1}{n}\sum(x_i-\bar{x})^3}{s^3}
$$

$$
\text{Kurtosis: } \hat{\gamma}_2 = \frac{\frac{1}{n}\sum(x_i-\bar{x})^4}{s^4}
$$

$$
\boxed{\text{JB} = \frac{n}{6}\left(\hat{\gamma}_1^2 + \frac{(\hat{\gamma}_2-3)^2}{4}\right) \sim \chi^2(2)}
$$

---

## 7. 小结

- 正态假设是经典金融理论的基石，但实证上几乎总是被拒绝
- QQ 图是最直观的诊断工具，"两端翘起"是胖尾的视觉信号
- JB 检验综合偏度和峰度，是金融领域最常用的正态性检验
- 正态假设下 VaR 会系统性低估极端损失，这是实践中必须注意的风险

> 下一篇：D3 胖尾分布与厚尾统计——正态假设失效之后，我们用什么替代？
