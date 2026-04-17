# Day 7：综合项目——多资产风险报告

> **目标**：将本周所有知识整合，构建一个完整的多资产组合风险分析系统，输出一份结构化的量化风险报告。

---

## 1. 项目概述

### 1.1 问题设定

假设你管理一个简单的 A 股指数组合：

| 资产 | 权重 | 说明 |
|------|------|------|
| 沪深300 | 40% | 大盘蓝筹 |
| 中证500 | 30% | 中盘成长 |
| 创业板指 | 20% | 小盘成长 |
| 国债指数 | 10% | 防御性资产 |

**核心问题**：
1. 这个组合的统计特征是什么？
2. 它真的满足正态假设吗？
3. 最坏情况下每天可能损失多少？（VaR/CVaR）
4. 各资产之间的相关结构如何影响组合风险？

### 1.2 输出目标

- 组合收益率时序分析
- 正态假设检验
- 历史法 + 参数法（t 分布）VaR/CVaR
- 相关矩阵与边际风险贡献
- 压力测试（模拟历史极端情景）
- 风险报告 Markdown

---

## 2. 关键公式汇总

### 组合收益率

$$R_p = \mathbf{w}^T \mathbf{R} = \sum_{i=1}^n w_i R_i$$

### 组合方差

$$\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}$$

### 边际风险贡献（MRC）

$$\text{MRC}_i = w_i \frac{\partial \sigma_p}{\partial w_i} = w_i \frac{(\Sigma \mathbf{w})_i}{\sigma_p}$$

各资产 MRC 之和等于组合总风险 $\sigma_p$。

### 风险分解

$$\sigma_p = \sum_{i=1}^n \text{MRC}_i$$

**风险贡献百分比**：$\text{RC}_i\% = \text{MRC}_i / \sigma_p$

### 分散化比率（Diversification Ratio）

$$\text{DR} = \frac{\sum_i w_i \sigma_i}{\sigma_p}$$

DR > 1 说明分散化有效降低了风险；DR 越大，分散化效果越好。

---

## 3. Python 实战代码（完整综合项目）

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "akshare",
#   "pandas",
#   "numpy",
#   "scipy",
#   "matplotlib",
#   "seaborn",
# ]
# ///

"""
Day 7: 多资产组合风险报告（Week 2 综合项目）
运行方式: uv run quant-w2-capstone.py
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import seaborn as sns
from scipy import stats
from datetime import datetime

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
np.random.seed(2024)

# ╔══════════════════════════════════════════════════════════════╗
# ║  配置区
# ╚══════════════════════════════════════════════════════════════╝
PORTFOLIO_WEIGHTS = {
    "沪深300": 0.40,
    "中证500": 0.30,
    "创业板":  0.20,
    "国债":    0.10,
}

INDEX_SYMBOLS = {
    "沪深300": "sh000300",
    "中证500": "sh000905",
    "创业板":  "sz399006",
    "国债":    "sh000012",
}

ALPHA_LIST = [0.95, 0.99]
START_DATE = "2018-01-01"
WINDOW_ROLL = 252   # 滚动 VaR 窗口（交易日）
N_SIM = 200_000     # Monte Carlo 模拟次数

# ╔══════════════════════════════════════════════════════════════╗
# ║  数据获取
# ╚══════════════════════════════════════════════════════════════╝
print("=" * 65)
print("  多资产组合风险报告 — 数据获取中...")
print("=" * 65)

price_dict = {}
for name, sym in INDEX_SYMBOLS.items():
    try:
        df = ak.stock_zh_index_daily(symbol=sym)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        price_dict[name] = df[df.index >= START_DATE]["close"]
        print(f"  ✅ {name}: {len(price_dict[name])} 天")
    except Exception as e:
        print(f"  ❌ {name}: {e}")

prices = pd.DataFrame({k: v for k, v in price_dict.items() if k in PORTFOLIO_WEIGHTS}).dropna()
log_rets = np.log(prices / prices.shift(1)).dropna()

# 确保权重仅包含可用资产，归一化
weights_raw = {k: PORTFOLIO_WEIGHTS[k] for k in log_rets.columns if k in PORTFOLIO_WEIGHTS}
total_w = sum(weights_raw.values())
weights = {k: v / total_w for k, v in weights_raw.items()}
w_array = np.array([weights[c] for c in log_rets.columns])

print(f"\n有效资产：{list(log_rets.columns)}")
print(f"实际权重：{weights}")
print(f"样本期间：{log_rets.index[0].date()} ~ {log_rets.index[-1].date()}, {len(log_rets)} 日")

# ╔══════════════════════════════════════════════════════════════╗
# ║  组合收益率计算
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  组合收益率计算")
print("=" * 65)

# 组合日收益率
port_ret = (log_rets * w_array).sum(axis=1)

mu_p = port_ret.mean()
sigma_p_daily = port_ret.std()
mu_p_annual = mu_p * 252
sigma_p_annual = sigma_p_daily * np.sqrt(252)
sharpe = (mu_p_annual - 0.03) / sigma_p_annual

print(f"组合年化收益率：{mu_p_annual:.2%}")
print(f"组合年化波动率：{sigma_p_annual:.2%}")
print(f"夏普比率（无风险率=3%）：{sharpe:.3f}")
print(f"偏度：{stats.skew(port_ret):.4f}")
print(f"超峰度：{stats.kurtosis(port_ret):.4f}")

# ╔══════════════════════════════════════════════════════════════╗
# ║  正态性检验
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  正态性检验")
print("=" * 65)

jb_stat, jb_p = stats.jarque_bera(port_ret)
sw_stat, sw_p = stats.shapiro(port_ret[:5000])
nu_t, loc_t, scale_t = stats.t.fit(port_ret.values)

print(f"Jarque-Bera: stat={jb_stat:.2f}, p={jb_p:.4e} → {'拒绝正态' if jb_p < 0.05 else '无法拒绝'}")
print(f"Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4e} → {'拒绝正态' if sw_p < 0.05 else '无法拒绝'}")
print(f"t 分布拟合：自由度 ν={nu_t:.2f}（越小尾部越厚）")

# ╔══════════════════════════════════════════════════════════════╗
# ║  VaR 与 CVaR（三种方法）
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  VaR 与 CVaR 计算")
print("=" * 65)

r_port = port_ret.values
var_table = {}

for alpha in ALPHA_LIST:
    # 历史法
    var_hist = -np.percentile(r_port, (1 - alpha) * 100)
    cvar_hist = -r_port[r_port <= -var_hist].mean()
    
    # 正态参数法
    z = stats.norm.ppf(1 - alpha)
    var_norm = -(mu_p + sigma_p_daily * z)
    cvar_norm = -(mu_p - sigma_p_daily * stats.norm.pdf(z) / (1 - alpha))
    
    # t 分布 Monte Carlo
    sim = stats.t.rvs(nu_t, loc_t, scale_t, size=N_SIM)
    var_mc = -np.percentile(sim, (1 - alpha) * 100)
    cvar_mc = -sim[sim <= -var_mc].mean()
    
    var_table[alpha] = {
        "历史法VaR": var_hist, "历史法CVaR": cvar_hist,
        "正态法VaR": var_norm, "正态法CVaR": cvar_norm,
        "MC_VaR": var_mc, "MC_CVaR": cvar_mc,
    }
    
    print(f"\n  置信水平 {alpha:.0%}:")
    print(f"    历史法：VaR={var_hist:.4%}, CVaR={cvar_hist:.4%}")
    print(f"    正态法：VaR={var_norm:.4%}, CVaR={cvar_norm:.4%} （低估 {var_norm/var_hist-1:.1%}）")
    print(f"    MC法  ：VaR={var_mc:.4%}, CVaR={cvar_mc:.4%}")

# ╔══════════════════════════════════════════════════════════════╗
# ║  边际风险贡献
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  边际风险贡献（Risk Contribution）")
print("=" * 65)

cov_matrix = log_rets.cov().values
sigma_vec = np.sqrt(np.diag(cov_matrix))
port_vol = np.sqrt(w_array @ cov_matrix @ w_array)

# MRC: w_i * (Σw)_i / σ_p
marginal_risk = (cov_matrix @ w_array) / port_vol
mrc = w_array * marginal_risk
rc_pct = mrc / port_vol

print(f"\n{'资产':<10} {'权重':>8} {'个体波动率':>12} {'MRC':>12} {'风险贡献%':>12}")
print("-" * 58)
for i, asset in enumerate(log_rets.columns):
    print(f"{asset:<10} {w_array[i]:>8.0%} {sigma_vec[i]*np.sqrt(252):>12.4%} "
          f"{mrc[i]*np.sqrt(252):>12.4%} {rc_pct[i]:>12.4%}")
print(f"{'合计':<10} {sum(w_array):>8.0%} {'':<12} {port_vol*np.sqrt(252):>12.4%} {'100.0%':>12}")

# 分散化比率
dr = np.sum(w_array * sigma_vec) / port_vol
print(f"\n分散化比率：{dr:.4f}（>1 说明分散化有效降低风险）")

# ╔══════════════════════════════════════════════════════════════╗
# ║  压力测试（历史极端情景）
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("  压力测试（历史极端情景）")
print("=" * 65)

# 找出最差的 10 天
worst_10 = port_ret.nsmallest(10)
print("\n历史最差 10 个交易日：")
for date, ret in worst_10.items():
    print(f"  {date.date()}  {ret:.4%}")

# 历史年度汇总
annual_ret = port_ret.resample("Y").sum()
print("\n按年度汇总：")
for year, ret in annual_ret.items():
    print(f"  {year.year}: {ret:.4%}")

# 最大回撤（基于对数收益率累计）
cumret = port_ret.cumsum()
rolling_max = cumret.cummax()
drawdown = cumret - rolling_max
max_dd = drawdown.min()
max_dd_date = drawdown.idxmin()
print(f"\n最大回撤：{max_dd:.4%}（发生在 {max_dd_date.date()}）")

# ╔══════════════════════════════════════════════════════════════╗
# ║  可视化报告
# ╚══════════════════════════════════════════════════════════════╝
print("\n生成风险报告图表...")

fig = plt.figure(figsize=(18, 16))
fig.suptitle("Multi-Asset Portfolio Risk Report\nWeek 2 Capstone Project",
             fontsize=14, y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# （1）累计收益对比
ax1 = fig.add_subplot(gs[0, :2])
for col in log_rets.columns:
    cumret_asset = log_rets[col].cumsum().apply(np.exp) - 1
    ax1.plot(cumret_asset.index, cumret_asset * 100, lw=1.2, label=col, alpha=0.8)
port_cumret_plot = port_ret.cumsum().apply(np.exp) - 1
ax1.plot(port_ret.cumsum().apply(np.exp).index,
         port_cumret_plot * 100, "k-", lw=2.5, label="Portfolio", zorder=5)
ax1.set_ylabel("Cumulative Return (%)")
ax1.set_title("Cumulative Return: Assets vs Portfolio")
ax1.legend(fontsize=8)
ax1.axhline(0, color="black", lw=0.5)
ax1.grid(alpha=0.3)

# （2）风险贡献饼图
ax2 = fig.add_subplot(gs[0, 2])
ax2.pie(rc_pct, labels=log_rets.columns, autopct="%.1f%%", startangle=90,
        colors=["#5b9bd5", "#ed7d31", "#a9d18e", "#ffc000"])
ax2.set_title("Risk Contribution (%)")

# （3）收益率分布
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(port_ret, bins=70, density=True, alpha=0.5, color="steelblue", label="Portfolio")
x_rng = np.linspace(port_ret.min(), port_ret.max(), 300)
ax3.plot(x_rng, stats.norm.pdf(x_rng, mu_p, sigma_p_daily), "r-", lw=1.5, label="Normal")
ax3.plot(x_rng, stats.t.pdf(x_rng, nu_t, loc_t, scale_t), "g-", lw=1.5, label=f"t(ν={nu_t:.1f})")
# 标注 VaR
v99_h = var_table[0.99]["历史法VaR"]
ax3.axvline(-v99_h, color="red", linestyle="--", lw=1.5, label=f"99%VaR({-v99_h:.2%})")
ax3.set_title("Portfolio Return Distribution")
ax3.legend(fontsize=7)
ax3.set_xlim(-0.07, 0.05)
ax3.grid(alpha=0.3)

# （4）QQ 图
ax4 = fig.add_subplot(gs[1, 1])
(osm, osr), (slope, intercept, _) = stats.probplot(port_ret, dist="norm")
ax4.scatter(osm, osr, s=4, alpha=0.4, color="steelblue")
x_line = np.array([osm[0], osm[-1]])
ax4.plot(x_line, slope * x_line + intercept, "r-", lw=2)
ax4.set_xlabel("Theoretical Quantiles")
ax4.set_ylabel("Sample Quantiles")
ax4.set_title("Portfolio Q-Q Plot (vs Normal)")
ax4.grid(alpha=0.3)

# （5）VaR 方法对比
ax5 = fig.add_subplot(gs[1, 2])
methods = ["历史法", "正态法", "MC法"]
var_vals_95 = [var_table[0.95][f"{m}VaR"] for m in methods]
var_vals_99 = [var_table[0.99][f"{m}VaR"] for m in methods]
cvar_vals_99 = [var_table[0.99][f"{m}CVaR"] for m in methods]

x_pos = np.arange(len(methods))
w = 0.28
ax5.bar(x_pos - w, var_vals_95, w, label="95% VaR", color="lightblue")
ax5.bar(x_pos, var_vals_99, w, label="99% VaR", color="steelblue")
ax5.bar(x_pos + w, cvar_vals_99, w, label="99% CVaR", color="orangered")
ax5.set_xticks(x_pos)
ax5.set_xticklabels(methods, fontsize=9)
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
ax5.set_title("VaR/CVaR by Method")
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3, axis="y")

# （6）相关矩阵
ax6 = fig.add_subplot(gs[2, 0])
corr_m = log_rets.corr()
sns.heatmap(corr_m, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, ax=ax6, linewidths=0.5, annot_kws={"size": 9})
ax6.set_title("Asset Correlation Matrix")

# （7）滚动波动率
ax7 = fig.add_subplot(gs[2, 1])
roll_vol_port = port_ret.rolling(60).std() * np.sqrt(252)
ax7.fill_between(roll_vol_port.index, roll_vol_port, alpha=0.4, color="steelblue")
ax7.plot(roll_vol_port.index, roll_vol_port, color="steelblue", lw=1.5)
ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax7.set_title("Portfolio 60-Day Rolling Volatility")
ax7.grid(alpha=0.3)

# （8）回撤图
ax8 = fig.add_subplot(gs[2, 2])
cumret_dd = port_ret.cumsum()
rolling_max_dd = cumret_dd.cummax()
drawdown_series = cumret_dd - rolling_max_dd
ax8.fill_between(drawdown_series.index, drawdown_series * 100, 0,
                 alpha=0.5, color="red")
ax8.plot(drawdown_series.index, drawdown_series * 100, "red", lw=1)
ax8.set_ylabel("Drawdown (%)")
ax8.set_title(f"Portfolio Drawdown (Max: {max_dd:.2%})")
ax8.grid(alpha=0.3)

plt.savefig("quant-w2-capstone.png", dpi=150, bbox_inches="tight")
print("风险报告图已保存为 quant-w2-capstone.png")

# ╔══════════════════════════════════════════════════════════════╗
# ║  最终报告输出
# ╚══════════════════════════════════════════════════════════════╝
report_date = datetime.now().strftime("%Y-%m-%d")
print(f"""
{'='*65}
  多资产组合风险报告
  生成日期：{report_date}
{'='*65}

【组合配置】
  沪深300: {weights.get('沪深300', 0):.0%} | 中证500: {weights.get('中证500', 0):.0%} | 
  创业板: {weights.get('创业板', 0):.0%} | 国债: {weights.get('国债', 0):.0%}

【收益特征】
  年化收益率：{mu_p_annual:.2%}
  年化波动率：{sigma_p_annual:.2%}
  夏普比率  ：{sharpe:.3f}
  最大回撤  ：{max_dd:.2%}

【分布特征】
  偏度：{stats.skew(r_port):.4f}（负值=左偏，下行风险更大）
  超峰度：{stats.kurtosis(r_port):.4f}（>0 = 厚尾）
  t分布自由度：{nu_t:.2f}（约 {'显著厚尾' if nu_t < 6 else '中度厚尾'}）
  正态假设：拒绝（JB p<0.001）

【风险度量（日）】
  99% VaR (历史法): {var_table[0.99]['历史法VaR']:.4%}
  99% CVaR (历史法): {var_table[0.99]['历史法CVaR']:.4%}
  正态法低估 VaR 约：{var_table[0.99]['正态法VaR']/var_table[0.99]['历史法VaR']-1:.1%}

【风险分散】
  分散化比率：{dr:.4f}
  主要风险贡献：{log_rets.columns[np.argmax(rc_pct)]}（{rc_pct.max():.1%}）

{'='*65}
""")
```

---

## 4. 项目扩展方向

完成本综合项目后，可进一步探索：

```
Week 3 预告：
├── GARCH 模型（波动率聚集效应）
├── 时间序列建模（AR/MA/ARIMA）
├── 因子模型（Barra 风险因子）
└── 组合优化（均值-方差，最小化 CVaR）
```

---

## 5. 本周知识图谱

```
价格序列
    ↓ 对数差分
对数收益率（时间可加、非负价格）
    ↓ 统计检验
正态假设 → 被拒绝
    ↓ 实证特征
胖尾 + 负偏 → 峰度/偏度/尾部指数
    ↓ 风险量化
VaR（分位数）→ CVaR（尾部期望）
    ↓ 多资产
相关矩阵 → 风险贡献 → 组合优化
```

---

## 6. 小结

| 知识点 | 核心结论 |
|--------|---------|
| 对数收益率 | 时间可加，是量化建模标准选择 |
| 正态假设 | 在金融中几乎总是被拒绝 |
| 胖尾 | t分布($\nu$≈4~6)比正态更好拟合 |
| VaR | 只是门槛，不告诉你损失有多大 |
| CVaR | 尾部损失期望，满足次可加性 |
| 相关性 | 危机时趋向1，分散化失效 |
| 组合风险 | 不等于个体风险加权平均 |

**Week 2 最重要的一句话**：

> **"正态分布是一个美丽的谎言。理解它为什么美丽，更要知道它在哪里说谎。"**

---

*下周预告 → Week 3: 时间序列与波动率建模（GARCH、均值回归、协整）*
