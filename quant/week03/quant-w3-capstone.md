---
layout: default
title: "D7 · 综合实战：多因子选股"
---

# D7 · Quant Week 3 综合实战

> **构建第一个多因子选股模型**

---

## 本周核心认识

| 工具 | 量化应用 |
|------|---------|
| 描述统计 | 评估策略收益分布，识别肥尾风险 |
| 概率分布 | 不要用正态分布估算极端风险，用 t 分布更准确 |
| 假设检验 | 判断因子是否显著有效，避免数据挖掘偏差 |
| 相关分析 | IC 评估因子质量，避免使用高度相关的冗余因子 |
| 线性回归 | Fama-MacBeth 检验因子溢价，量化研究标准方法 |

---

## 实战：三因子选股模型

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)
n_stocks = 400
n_periods = 48  # 4年月频

# 生成三个因子（价值、质量、低波动）
value_factor = pd.DataFrame(np.random.randn(n_periods, n_stocks))
quality_factor = pd.DataFrame(np.random.randn(n_periods, n_stocks))
lowvol_factor = pd.DataFrame(-np.abs(np.random.randn(n_periods, n_stocks)))  # 低波动：值越高越好

# 实际收益：三个因子都有效
returns = pd.DataFrame(
    np.random.normal(0.005, 0.06, (n_periods, n_stocks))
    + 0.02 * ((value_factor - value_factor.mean(axis=1).values.reshape(-1,1)) / 
               value_factor.std(axis=1).values.reshape(-1,1))
    + 0.015 * ((quality_factor - quality_factor.mean(axis=1).values.reshape(-1,1)) / 
                quality_factor.std(axis=1).values.reshape(-1,1))
    + 0.01 * ((lowvol_factor - lowvol_factor.mean(axis=1).values.reshape(-1,1)) / 
               lowvol_factor.std(axis=1).values.reshape(-1,1))
)

# ============================================================
# 合成因子（等权）
# ============================================================
def standardize(df):
    return df.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=1)

composite_factor = (
    standardize(value_factor) + 
    standardize(quality_factor) + 
    standardize(lowvol_factor)
) / 3

# ============================================================
# 月度再平衡选股回测
# ============================================================
portfolio_returns = []
benchmark_returns = []  # 等权基准

for t in range(1, n_periods):
    factor_t = composite_factor.iloc[t-1]
    return_t = returns.iloc[t]
    
    # 选因子得分最高的 20% 股票
    threshold = factor_t.quantile(0.8)
    selected = factor_t >= threshold
    
    if selected.sum() > 0:
        port_return = return_t[selected].mean()
    else:
        port_return = return_t.mean()
    
    portfolio_returns.append(port_return)
    benchmark_returns.append(return_t.mean())

port_returns = np.array(portfolio_returns)
bench_returns = np.array(benchmark_returns)
excess_returns = port_returns - bench_returns

# ============================================================
# 绩效评估
# ============================================================
def performance_stats(returns, benchmark=None, name="策略"):
    ann_factor = 12  # 月频
    
    ann_return = returns.mean() * ann_factor * 100
    ann_vol = returns.std() * np.sqrt(ann_factor) * 100
    sharpe = returns.mean() / returns.std() * np.sqrt(ann_factor)
    
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    mdd = ((cumulative - rolling_max) / rolling_max).min() * 100
    
    t_stat, p_val = stats.ttest_1samp(returns, 0)
    
    print(f"\n=== {name} 绩效 ===")
    print(f"年化收益率: {ann_return:.2f}%")
    print(f"年化波动率: {ann_vol:.2f}%")
    print(f"夏普比率:   {sharpe:.3f}")
    print(f"最大回撤:   {mdd:.2f}%")
    print(f"t统计量:    {t_stat:.3f} (p={p_val:.4f})")
    print(f"显著性:     {'✅ 5%水平显著' if p_val < 0.05 else '❌ 不显著'}")
    
    return cumulative

port_cumret = performance_stats(port_returns, name="多因子组合")
bench_cumret = performance_stats(bench_returns, name="等权基准")
performance_stats(excess_returns, name="超额收益")

# 可视化
plt.figure(figsize=(12, 5))
plt.plot((1 + pd.Series(port_returns)).cumprod(), label='多因子组合', linewidth=2, color='#2196F3')
plt.plot((1 + pd.Series(bench_returns)).cumprod(), label='等权基准', linewidth=1.5, color='#FF9800', linestyle='--')
plt.xlabel('月份')
plt.ylabel('累积收益')
plt.title('多因子组合 vs 基准')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Week 3 完成！

🎉 **Quant Week 3 全部完成！**

- ✅ 描述统计：均值/方差/偏度/峰度，理解收益率分布
- ✅ 概率分布：正态 vs t 分布 vs 幂律，金融肥尾
- ✅ 假设检验：t 检验 + 多重检验校正，避免数据挖掘偏差
- ✅ 相关分析：IC / ICIR，因子协方差矩阵
- ✅ 线性回归：Fama-MacBeth，行业标准因子检验方法
- ✅ 综合实战：多因子选股组合

**Week 4 预告**：金融数学基础——利率、时间价值、期权定价（BSM 模型）
