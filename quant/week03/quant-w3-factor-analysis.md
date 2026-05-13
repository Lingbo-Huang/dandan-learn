---
layout: default
title: "D6 · 实战：完整因子分析流程"
---

# D6 · 实战：构建统计显著的量化因子

> **Quant Week 3**  
> 把这周学的所有统计工具串成一个完整的因子研究流程。

---

## 完整因子研究流程

```
1. 因子定义与计算
2. 因子预处理（去极值、标准化、市值中性化）
3. IC 分析（预测力评估）
4. 分组回测（因子收益稳定性）
5. 多因子显著性检验（Fama-MacBeth）
6. 结论与应用
```

---

## 实战：动量因子分析

```python
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 1. 数据生成（模拟A股月度数据）
# ============================================================
n_stocks = 300
n_months = 60  # 5年月频

# 模拟股票月度收益率（含各类效应）
base_returns = np.random.normal(0.005, 0.08, (n_months, n_stocks))

# 加入动量效应：过去12个月收益高的股票，下月收益也更高
momentum_signal = np.random.randn(n_months, n_stocks)
for t in range(12, n_months):
    past_12m = base_returns[t-12:t].sum(axis=0)  # 过去12月累积收益
    momentum_signal[t] = past_12m

# 实际收益：动量效应 + 特质风险
actual_returns = pd.DataFrame(
    base_returns + 0.3 * (momentum_signal - momentum_signal.mean(axis=1, keepdims=True)) / 
    (momentum_signal.std(axis=1, keepdims=True) + 1e-8),
    columns=[f'stock_{i}' for i in range(n_stocks)]
)

# ============================================================
# 2. 因子计算：12个月动量（跳过最近1个月）
# ============================================================
momentum_factor = pd.DataFrame(index=actual_returns.index, columns=actual_returns.columns, dtype=float)

for t in range(13, n_months):
    # 过去12月收益（跳过最近1月，避免反转效应）
    past_12m_return = actual_returns.iloc[t-12:t-1].sum(axis=0)
    momentum_factor.iloc[t] = past_12m_return

momentum_factor = momentum_factor.dropna(how='all')

# ============================================================
# 3. 因子预处理
# ============================================================
def preprocess_factor(factor_values):
    """去极值（3倍MAD）+ 标准化"""
    # 去极值
    median = factor_values.median()
    mad = (factor_values - median).abs().median()
    lower = median - 3 * 1.4826 * mad
    upper = median + 3 * 1.4826 * mad
    factor_values = factor_values.clip(lower, upper)
    
    # 标准化
    factor_values = (factor_values - factor_values.mean()) / (factor_values.std() + 1e-8)
    
    return factor_values

momentum_processed = momentum_factor.apply(preprocess_factor, axis=1)

# ============================================================
# 4. IC 分析
# ============================================================
valid_periods = momentum_processed.index[momentum_processed.index < len(actual_returns) - 1]

ic_monthly = []
for t in valid_periods:
    if t + 1 >= len(actual_returns):
        break
    factor_t = momentum_processed.iloc[t]
    return_t1 = actual_returns.iloc[t + 1]
    
    valid_mask = factor_t.notna() & return_t1.notna()
    if valid_mask.sum() < 30:
        continue
    
    ic, _ = stats.spearmanr(factor_t[valid_mask], return_t1[valid_mask])
    ic_monthly.append(ic)

ic_series = np.array(ic_monthly)

print("=== 动量因子 IC 分析 ===")
print(f"IC 均值:        {ic_series.mean():.4f}  (>0.05 有参考价值)")
print(f"IC 标准差:      {ic_series.std():.4f}")
print(f"ICIR:           {ic_series.mean()/ic_series.std():.4f}  (>0.5 有较强预测力)")
print(f"IC>0 占比:      {(ic_series>0).mean():.2%}  (>55% 方向稳定)")

# t 检验：IC 均值是否显著不为零
t_ic, p_ic = stats.ttest_1samp(ic_series, 0)
print(f"\nIC 显著性: t={t_ic:.3f}, p={p_ic:.4f}")
print(f"结论: 动量因子{'✅ 统计显著' if p_ic < 0.05 else '❌ 不显著'}")

# ============================================================
# 5. 分组回测（5组，看因子值从低到高的收益差异）
# ============================================================
n_groups = 5
group_returns = {i: [] for i in range(n_groups)}

for t in valid_periods:
    if t + 1 >= len(actual_returns):
        break
    factor_t = momentum_processed.iloc[t]
    return_t1 = actual_returns.iloc[t + 1]
    
    valid_mask = factor_t.notna() & return_t1.notna()
    if valid_mask.sum() < n_groups * 10:
        continue
    
    # 按因子值排序，分5组
    ranks = factor_t[valid_mask].rank(pct=True)
    for g in range(n_groups):
        lower_pct = g / n_groups
        upper_pct = (g + 1) / n_groups
        group_mask = (ranks >= lower_pct) & (ranks < upper_pct)
        group_return = return_t1[valid_mask][group_mask.values].mean()
        group_returns[g].append(group_return)

# 可视化分组收益
avg_group_returns = [np.mean(group_returns[g]) * 12 * 100 for g in range(n_groups)]  # 年化

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, n_groups+1), avg_group_returns, 
        color=['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4'])
plt.xlabel('因子分组（1=低动量, 5=高动量）')
plt.ylabel('年化收益率 (%)')
plt.title('动量因子分组收益')
plt.axhline(y=np.mean(avg_group_returns), color='black', linestyle='--', label='等权平均')
plt.legend()

plt.subplot(1, 2, 2)
ic_plot = pd.Series(ic_series)
ic_plot.plot(kind='bar', color=['green' if x > 0 else 'red' for x in ic_series], alpha=0.7)
plt.axhline(y=ic_series.mean(), color='black', linestyle='--', label=f'均值={ic_series.mean():.4f}')
plt.xlabel('月份')
plt.ylabel('IC')
plt.title('IC 时序')
plt.legend()

plt.tight_layout()
plt.show()

print(f"\n多空组合（第5组-第1组）年化收益: {avg_group_returns[4]-avg_group_returns[0]:.2f}%")
```

---

## 今天完成的事

✅ 因子定义 → ✅ 预处理（去极值+标准化）→ ✅ IC 分析 → ✅ 分组回测 → ✅ 显著性检验

这是量化研究员的标准工作流。

---

## 明天预告

D7：**综合实战**——Week 3 完整项目，构建多因子选股组合。
