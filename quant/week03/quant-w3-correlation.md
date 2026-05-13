---
layout: default
title: "D4 · 相关分析：因子相关性与协方差矩阵"
---

# D4 · 相关分析在量化中的应用

> **Quant Week 3**  
> 因子选股的核心不只是"哪个因子好"，而是"哪些因子合在一起更好"。

---

## 一、相关系数的种类

| 方法 | 适用场景 | 特点 |
|------|---------|------|
| **Pearson** | 线性关系，正态分布 | 对异常值敏感 |
| **Spearman** | 非线性，排名关系 | 鲁棒，量化常用 |
| **Kendall** | 排名一致性 | 最鲁棒 |

**量化中多用 Spearman**：因子值到收益率的关系往往是非线性的，排名更稳定。

---

## 二、因子 IC（Information Coefficient）

**IC = 因子值与下期收益率的 Spearman 相关系数**

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)
n_stocks = 500  # 股票数量
n_periods = 60  # 60个月

# 模拟因子值和收益率
factor_values = np.random.randn(n_periods, n_stocks)
# 假设因子有一定预测力：因子值高 → 下期收益更高（加噪声）
future_returns = 0.3 * factor_values + np.random.randn(n_periods, n_stocks) * 1.5

# 计算每期 IC
ic_series = []
for t in range(n_periods):
    ic, _ = stats.spearmanr(factor_values[t], future_returns[t])
    ic_series.append(ic)

ic_series = np.array(ic_series)

print("=== 因子 IC 分析 ===")
print(f"IC 均值 (ICIR分子):  {ic_series.mean():.4f}")
print(f"IC 标准差:           {ic_series.std():.4f}")
print(f"ICIR (IR ratio):    {ic_series.mean()/ic_series.std():.4f}")
print(f"IC > 0 的比率:       {(ic_series > 0).mean():.2%}")
print()
print("行业标准：")
print("  IC 均值 > 0.05: 有参考价值")
print("  ICIR > 0.5:    有较强预测力")
print("  IC > 0 占比 > 55%: 方向稳定")

# 可视化
plt.figure(figsize=(10, 4))
plt.bar(range(len(ic_series)), ic_series, 
        color=['green' if x > 0 else 'red' for x in ic_series], alpha=0.7)
plt.axhline(y=ic_series.mean(), color='black', linewidth=2, linestyle='--', 
            label=f'均值={ic_series.mean():.4f}')
plt.axhline(y=0, color='gray', linewidth=1)
plt.xlabel('期数')
plt.ylabel('IC')
plt.title('因子 IC 时序')
plt.legend()
plt.show()
```

---

## 三、协方差矩阵：多因子风险模型的核心

```python
# 构建多因子协方差矩阵
n_factors = 5
factor_names = ['市值', '价值', '动量', '质量', '波动率']

# 模拟5个因子的月度收益（时序）
factor_returns = pd.DataFrame(
    np.random.multivariate_normal(
        mean=[0.005, 0.003, 0.004, 0.002, -0.001],
        cov=np.array([
            [0.04, 0.02, -0.01, 0.01, -0.02],
            [0.02, 0.03, -0.01, 0.02, -0.01],
            [-0.01, -0.01, 0.05, -0.01, -0.03],
            [0.01, 0.02, -0.01, 0.02, 0.01],
            [-0.02, -0.01, -0.03, 0.01, 0.06],
        ]),
        size=120  # 10年月频数据
    ),
    columns=factor_names
)

# 计算相关矩阵
corr_matrix = factor_returns.corr(method='spearman')

import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 相关矩阵热图
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title('因子 Spearman 相关矩阵')

# 因子收益率时序
factor_returns.cumsum().plot(ax=axes[1])
axes[1].set_title('因子累积收益')
axes[1].set_xlabel('月份')
plt.tight_layout()
plt.show()

print("高度相关的因子对（|corr| > 0.5）：")
for i in range(len(factor_names)):
    for j in range(i+1, len(factor_names)):
        if abs(corr_matrix.iloc[i,j]) > 0.3:
            print(f"  {factor_names[i]} vs {factor_names[j]}: {corr_matrix.iloc[i,j]:.3f}")
```

---

## 四、因子正交化：去除相关性

```python
# Gram-Schmidt 正交化
from sklearn.preprocessing import StandardScaler
from numpy.linalg import qr

# 标准化因子
scaler = StandardScaler()
X = scaler.fit_transform(factor_returns)

# QR 分解实现正交化
Q, R = qr(X)

orthogonal_factors = pd.DataFrame(Q, columns=[f'{n}_正交' for n in factor_names])

# 验证正交化后的相关性
print("正交化后的相关矩阵（应接近单位阵）：")
print(orthogonal_factors.corr().round(3))
```

---

## 今天的关键认识

1. **IC = Spearman(因子值, 下期收益)**：评估因子预测力的核心指标
2. **ICIR = IC均值/IC标准差**：风险调整后的因子质量
3. **协方差矩阵**：多因子风险模型的数学基础
4. **高度相关的因子**：叠加使用会产生多重共线性，需要正交化或选择

---

## 明天预告

D5：**线性回归**——用 Barra 模型的思路，量化检验因子的统计显著性。
