---
layout: default
title: "D3 · 假设检验：判断因子是否有效"
---

# D3 · 假设检验在量化中的应用

> **Quant Week 3**  
> "这个因子有用吗？"——假设检验给你一个统计严谨的答案。

---

## 一、量化中的假设检验场景

| 场景 | 原假设 (H₀) | 备择假设 (H₁) |
|------|------------|--------------|
| 策略是否盈利 | 平均收益率 = 0 | 平均收益率 ≠ 0 |
| 动量因子是否有效 | 因子 IC = 0 | 因子 IC ≠ 0 |
| 两组股票收益是否不同 | μ₁ = μ₂ | μ₁ ≠ μ₂ |
| 收益率是否随机游走 | 无自相关 | 存在自相关（可预测） |

---

## 二、t 检验：最常用的工具

**单样本 t 检验**：均值是否显著不为零？

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# 场景：策略过去250天的日收益率，均值是否显著不为零？
daily_returns = np.random.normal(0.0008, 0.015, 250)  # 略微正收益

# 单样本 t 检验
t_stat, p_value = stats.ttest_1samp(daily_returns, popmean=0)
n = len(daily_returns)

print("=== 策略收益率显著性检验 ===")
print(f"样本量:      {n}")
print(f"样本均值:    {daily_returns.mean():.6f} ({daily_returns.mean()*252*100:.1f}% 年化)")
print(f"t 统计量:    {t_stat:.4f}")
print(f"p 值:        {p_value:.4f}")
print(f"显著性:      {'✅ 5% 显著水平下拒绝 H₀（收益率显著不为零）' if p_value < 0.05 else '❌ 未能拒绝 H₀（无法证明收益率不为零）'}")

# 95% 置信区间
ci = stats.t.interval(0.95, df=n-1, loc=daily_returns.mean(), scale=stats.sem(daily_returns))
print(f"95% 置信区间: [{ci[0]:.6f}, {ci[1]:.6f}]")
print(f"（如果包含0，则在95%水平下不显著）")
```

---

## 三、p 值的正确理解

**p 值 = 在 H₀ 为真的条件下，观测到这么极端（或更极端）结果的概率**

**常见误解**：
- ❌ p < 0.05 意味着因子有效的概率 > 95%（错！）
- ❌ p > 0.05 意味着 H₀ 为真（错！只是没有足够证据拒绝）
- ✅ p 值是在"H₀ 成立"的假设下，数据出现的概率

```python
# 可视化 t 检验
fig, ax = plt.subplots(figsize=(10, 5))

df = n - 1
x = np.linspace(-5, 5, 200)
ax.plot(x, stats.t.pdf(x, df=df), 'b-', linewidth=2, label=f't分布 (df={df})')

# 标记 t 统计量
ax.axvline(x=t_stat, color='red', linestyle='--', linewidth=2, label=f't={t_stat:.3f}')
ax.axvline(x=-t_stat, color='red', linestyle='--', linewidth=2)

# 标记拒绝域（双尾，α=0.05）
critical = stats.t.ppf(0.975, df=df)
ax.fill_between(x, 0, stats.t.pdf(x, df=df), where=x > critical, alpha=0.3, color='red', label='拒绝域 (α=0.05)')
ax.fill_between(x, 0, stats.t.pdf(x, df=df), where=x < -critical, alpha=0.3, color='red')

ax.set_title(f'假设检验可视化（p={p_value:.4f}）')
ax.legend()
plt.show()
```

---

## 四、多重检验问题：量化的重大陷阱

> **警告：这是量化回测最常见的坑！**

```python
# 如果测试 100 个因子，即使全都无效，也期望有 5 个看起来"显著"（p<0.05）
import numpy as np

np.random.seed(42)
n_tests = 100
n_significant = 0

for _ in range(n_tests):
    # 随机数据（真的无效因子）
    returns = np.random.normal(0, 0.015, 100)
    _, p = stats.ttest_1samp(returns, 0)
    if p < 0.05:
        n_significant += 1

print(f"100个随机因子中，'显著'的有: {n_significant} 个")
print(f"期望值: {100 * 0.05:.0f} 个（假阳性率5%）")
# 这就是"数据挖掘偏差"（Data Snooping Bias）
```

**解决方案：Bonferroni 校正**

```python
from statsmodels.stats.multitest import multipletests

# 假设测试了20个因子
p_values = np.random.uniform(0, 1, 20)
p_values[3] = 0.001   # 人为加入一个真正有效的
p_values[7] = 0.03    # 一个边界情况

# Bonferroni 校正（最保守）
reject_bonferroni, p_corrected_bonf, _, _ = multipletests(p_values, method='bonferroni')

# FDR 控制（Benjamini-Hochberg，更宽松但更常用）
reject_fdr, p_corrected_fdr, _, _ = multipletests(p_values, method='fdr_bh')

print("因子 | 原始p值 | Bonferroni校正 | FDR校正")
for i in range(len(p_values)):
    print(f"  {i+1:2d} | {p_values[i]:.4f}  | {'✅' if reject_bonferroni[i] else '❌'}             | {'✅' if reject_fdr[i] else '❌'}")
```

---

## 五、A/B 测试：对比两组策略

```python
# 策略 A vs 策略 B，哪个更好？
np.random.seed(42)
strategy_A = np.random.normal(0.0005, 0.015, 252)
strategy_B = np.random.normal(0.0008, 0.016, 252)

# 双样本 t 检验（独立样本）
t_stat, p_value = stats.ttest_ind(strategy_A, strategy_B)

print(f"策略A 年化收益: {strategy_A.mean()*252*100:.2f}%")
print(f"策略B 年化收益: {strategy_B.mean()*252*100:.2f}%")
print(f"差异是否显著: {'是' if p_value < 0.05 else '否'} (p={p_value:.4f})")

# 配对 t 检验（同期对比，更敏感）
t_paired, p_paired = stats.ttest_rel(strategy_A, strategy_B)
print(f"配对检验 p值: {p_paired:.4f}")
```

---

## 今天的关键认识

1. **假设检验**：用统计方法量化"因子有效"的置信程度
2. **p 值不是因子有效的概率**，是在无效假设下数据出现的概率
3. **多重检验是量化最大的坑**：测试越多因子，假阳性越多
4. **Bonferroni/FDR 校正**：测试多个因子时的必备手段

---

## 明天预告

D4：**相关分析**——因子之间的相关性，构建不相关因子组合的方法。
