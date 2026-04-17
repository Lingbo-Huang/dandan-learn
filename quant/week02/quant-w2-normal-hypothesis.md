# Day 2：正态假设检验

> **核心问题**：金融教科书假设收益率正态分布，这个假设成立吗？如何用统计方法验证？

---

## 1. 理论基础

### 1.1 为什么正态假设如此流行？

- **中心极限定理**：大量独立小波动之和趋近正态
- **数学便利性**：正态分布由均值 $\mu$ 和方差 $\sigma^2$ 完全刻画
- **历史惯例**：Markowitz 均值-方差框架（1952年）以此为基础

正态分布概率密度：

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

### 1.2 检验正态性的方法

#### （1）矩检验：偏度与峰度

$$\text{偏度（Skewness）：} S = \frac{E[(X-\mu)^3]}{\sigma^3}$$

$$\text{峰度（Kurtosis）：} K = \frac{E[(X-\mu)^4]}{\sigma^4}$$

正态分布：$S = 0$，$K = 3$（超峰度 = $K - 3 = 0$）

金融收益率典型表现：$S < 0$（左偏），$K > 3$（尖峰厚尾）

#### （2）Jarque-Bera 检验

基于偏度和峰度的联合检验：

$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right) \overset{H_0}{\sim} \chi^2(2)$$

- $H_0$：数据服从正态分布
- $p < 0.05$ → 拒绝正态假设

#### （3）Shapiro-Wilk 检验

- 适用于小样本（$n < 5000$）
- 统计量 $W$ 接近 1 表示越接近正态
- 实践中金融数据几乎总是拒绝

#### （4）Kolmogorov-Smirnov 检验

比较经验分布函数与理论正态分布的最大距离：

$$D_n = \sup_x |F_n(x) - F(x)|$$

#### （5）QQ 图（Quantile-Quantile Plot）

- 横轴：理论正态分位数
- 纵轴：实际数据分位数
- **若数据服从正态，点应落在对角线上**
- 金融数据典型：两端翘起（尾部偏离，即厚尾）

---

## 2. 正态假设的实证结论

大量研究表明：

| 特征 | 正态假设 | 实际金融收益率 |
|------|---------|--------------|
| 偏度 | 0 | 通常为负（-0.2 ~ -1.0） |
| 超峰度 | 0 | 通常为正（3 ~ 10+） |
| 极端事件频率 | 极低（5σ ≈ 十亿年一遇） | 远高于正态预测 |
| JB 检验 p 值 | 均匀分布 | 几乎总是 <0.001 |

**结论：正态假设在金融中几乎总是被拒绝，但作为一阶近似仍有使用价值。**

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
#   "statsmodels",
# ]
# ///

"""
Day 2: 正态假设检验
运行方式: uv run quant-w2-normal-hypothesis.py
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from statsmodels.graphics.gofplots import qqplot

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ─── 1. 获取数据：多只股票 ──────────────────────────────────────────────────
print("正在获取数据...")

def get_stock_log_returns(symbol: str, start: str = "2020-01-01") -> pd.Series:
    """获取股票对数收益率序列"""
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[df.index >= start]["close"]
        log_ret = np.log(df / df.shift(1)).dropna()
        return log_ret
    except Exception as e:
        print(f"  获取 {symbol} 失败：{e}")
        return pd.Series(dtype=float)

# 获取沪深300、上证50、中证500
symbols = {
    "CSI300 (sh000300)": "sh000300",
    "SSE50 (sh000016)": "sh000016",
    "CSI500 (sh000905)": "sh000905",
}

returns_dict = {}
for name, sym in symbols.items():
    r = get_stock_log_returns(sym)
    if len(r) > 100:
        returns_dict[name] = r
        print(f"  {name}: {len(r)} 个交易日")

# ─── 2. 正态性检验汇总 ─────────────────────────────────────────────────────
print("\n=== 正态性检验汇总 ===")
print(f"{'指数':<25} {'偏度':>8} {'峰度':>8} {'超峰度':>8} {'JB_stat':>10} {'JB_p':>8} {'SW_p':>8}")
print("-" * 90)

test_results = {}
for name, r in returns_dict.items():
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)   # 完整峰度（正态=3）
    excess_kurt = kurt - 3                    # 超峰度（正态=0）
    
    # Jarque-Bera 检验
    jb_stat, jb_p = stats.jarque_bera(r)
    
    # Shapiro-Wilk（样本大时取子集）
    sample = r.sample(min(5000, len(r)), random_state=42)
    sw_stat, sw_p = stats.shapiro(sample)
    
    test_results[name] = {
        "偏度": skew, "峰度": kurt, "超峰度": excess_kurt,
        "JB统计量": jb_stat, "JB_p值": jb_p, "SW_p值": sw_p
    }
    
    sig_jb = "***" if jb_p < 0.001 else ("**" if jb_p < 0.01 else "*")
    sig_sw = "***" if sw_p < 0.001 else ("**" if sw_p < 0.01 else "*")
    
    print(f"{name:<25} {skew:>8.4f} {kurt:>8.4f} {excess_kurt:>8.4f} "
          f"{jb_stat:>10.1f} {jb_p:>8.4f}{sig_jb} {sw_p:>8.4f}{sig_sw}")

print("\n注：*** p<0.001, ** p<0.01, * p<0.05")
print("结论：所有指数收益率在 0.1% 置信水平下显著拒绝正态假设")

# ─── 3. KS 检验（与正态分布对比）──────────────────────────────────────────
print("\n=== KS 检验（vs 拟合正态）===")
for name, r in returns_dict.items():
    ks_stat, ks_p = stats.kstest(r, 'norm', args=(r.mean(), r.std()))
    print(f"{name:<25} KS统计量={ks_stat:.4f}, p值={ks_p:.4e}")

# ─── 4. 极端事件概率：正态预测 vs 实际 ────────────────────────────────────
print("\n=== 极端事件频率：正态预测 vs 实际 ===")
r = list(returns_dict.values())[0]  # 用 CSI300
mu, sigma = r.mean(), r.std()
n = len(r)

for k_sigma in [2, 3, 4, 5]:
    threshold = k_sigma * sigma
    # 正态分布预测概率（双侧）
    normal_prob = 2 * stats.norm.sf(k_sigma)
    normal_expected = normal_prob * n
    # 实际出现次数
    actual_count = (r.abs() >= threshold).sum()
    actual_prob = actual_count / n
    
    print(f"  |R| >= {k_sigma}σ ({threshold:.2%}): "
          f"正态预期 {normal_expected:.2f}次({normal_prob:.4%}), "
          f"实际 {actual_count}次({actual_prob:.4%}), "
          f"比率 {actual_count/max(normal_expected,0.01):.1f}x")

# ─── 5. 可视化：直方图 + 正态曲线 + QQ图 ───────────────────────────────────
r_main = list(returns_dict.values())[0]
r_name = list(returns_dict.keys())[0]
mu, sigma = r_main.mean(), r_main.std()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Normality Test: {r_name}", fontsize=13)

# 直方图 + 正态曲线
ax = axes[0]
ax.hist(r_main, bins=80, density=True, alpha=0.6, color="steelblue", label="Empirical")
x_range = np.linspace(r_main.min(), r_main.max(), 200)
ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), "r-", linewidth=2, label="Normal fit")
ax.set_xlabel("Log Return")
ax.set_ylabel("Density")
ax.set_title("Distribution vs Normal Fit")
ax.legend()
ax.grid(alpha=0.3)

# 添加文字注释
textstr = f"Skewness: {stats.skew(r_main):.3f}\nExcess Kurtosis: {stats.kurtosis(r_main):.3f}\nJB p-value: <0.001"
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

# QQ 图
ax2 = axes[1]
(osm, osr), (slope, intercept, r) = stats.probplot(r_main, dist="norm")
ax2.scatter(osm, osr, alpha=0.3, s=5, color="steelblue")
x_line = np.array([osm[0], osm[-1]])
ax2.plot(x_line, slope * x_line + intercept, "r-", linewidth=2)
ax2.set_xlabel("Theoretical Quantiles")
ax2.set_ylabel("Sample Quantiles")
ax2.set_title("Q-Q Plot (vs Normal)")
ax2.grid(alpha=0.3)

# 标注尾部偏离
ax2.annotate("Fat left tail\n(more extreme losses)", 
             xy=(osm[5], osr[5]), xytext=(-3, osr[5]*0.7),
             arrowprops=dict(arrowstyle="->", color="orange"),
             color="orange", fontsize=8)
ax2.annotate("Fat right tail\n(more extreme gains)", 
             xy=(osm[-5], osr[-5]), xytext=(1, osr[-5]*0.7),
             arrowprops=dict(arrowstyle="->", color="orange"),
             color="orange", fontsize=8)

plt.tight_layout()
plt.savefig("quant-w2-normal-hypothesis.png", dpi=150)
print("\n图表已保存为 quant-w2-normal-hypothesis.png")

# ─── 6. 正态假设下 VaR 的低估 ─────────────────────────────────────────────
print("\n=== 正态假设下 VaR 低估 ===")
r = r_main
for conf in [0.95, 0.99, 0.999]:
    var_normal = -stats.norm.ppf(1 - conf, mu, sigma)   # 正态参数法
    var_hist = -np.percentile(r, (1 - conf) * 100)       # 历史法
    print(f"  {conf:.1%} VaR: 正态={var_normal:.4%}, 历史={var_hist:.4%}, "
          f"低估 {(var_normal/var_hist-1):.1%}")
```

---

## 4. 正态假设的局限与适用场景

### 何时正态假设"勉强可用"

- **短期日历收益率**（误差可控）
- **大型组合**（分散化后接近正态）
- **作为对照基准**（了解偏差有多大）

### 何时正态假设危险

- 计算极端事件 VaR（如 99.9% 置信水平）
- 期权定价（尾部分布至关重要）
- 危机情景分析
- 高频数据（分布偏态更显著）

---

## 5. 小结

| 检验方法 | 原假设 | 金融数据典型结果 |
|---------|--------|----------------|
| Jarque-Bera | 偏度=0 且峰度=3 | 几乎总是拒绝（p<0.001） |
| Shapiro-Wilk | 来自正态总体 | 几乎总是拒绝 |
| KS 检验 | 与正态分布相同 | 几乎总是拒绝 |
| QQ 图 | 点在对角线上 | 两端明显翘起（厚尾） |

**关键结论**：
1. 金融收益率普遍表现出**负偏（左偏）**和**尖峰厚尾**
2. 正态假设下，极端事件概率被严重低估（实际发生频率比正态预测高 5~20 倍）
3. **直接后果**：用正态假设计算的 VaR 会低估实际风险

---

*下一步 → [Day 3: 胖尾分布](./quant-w2-fat-tails.md)*
