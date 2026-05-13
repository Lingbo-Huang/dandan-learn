---
layout: default
title: "D5 · 线性回归：因子统计检验"
---

# D5 · 线性回归在量化因子研究中的应用

> **Quant Week 3**  
> Fama-MacBeth 回归是量化因子研究的标准方法——今天学会用它。

---

## 一、截面回归：因子溢价检验

**问题**：市盈率（PE）低的股票，下个月收益率是否更高？

$$r_{i,t+1} = \alpha + \beta \cdot \text{PE\_inv}_{i,t} + \epsilon_{i,t}$$

```python
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)
n_stocks = 300

# 模拟一期截面数据
PE_inv = np.random.lognormal(0, 0.8, n_stocks)  # 1/PE（市盈率倒数，越高越便宜）
market_cap = np.random.lognormal(10, 2, n_stocks)  # 市值

# 下期收益率：与 PE_inv 正相关（价值效应），与市值负相关（小盘效应）
future_return = (
    0.005                                       # 截距（市场整体收益）
    + 0.008 * (PE_inv - PE_inv.mean()) / PE_inv.std()   # 价值因子
    - 0.003 * (np.log(market_cap) - np.log(market_cap).mean()) / np.log(market_cap).std()  # 市值因子
    + np.random.normal(0, 0.05, n_stocks)       # 特质风险
)

# OLS 回归
X = sm.add_constant(np.column_stack([
    (PE_inv - PE_inv.mean()) / PE_inv.std(),
    (np.log(market_cap) - np.log(market_cap).mean()) / np.log(market_cap).std()
]))
model = sm.OLS(future_return, X).fit()

print(model.summary())
print("\n关键输出解读：")
print(f"PE倒数系数: {model.params[1]:.4f}, t={model.tvalues[1]:.2f}, p={model.pvalues[1]:.4f}")
print(f"  → 价值因子{'显著有效 ✅' if model.pvalues[1] < 0.05 else '不显著 ❌'}")
print(f"市值系数: {model.params[2]:.4f}, t={model.tvalues[2]:.2f}, p={model.pvalues[2]:.4f}")
print(f"  → 市值因子{'显著有效 ✅' if model.pvalues[2] < 0.05 else '不显著 ❌'}")
```

---

## 二、Fama-MacBeth 回归：标准的因子研究方法

**思路**：每期做一次截面回归，得到因子收益（lambda）的时间序列，再对这个时序做 t 检验。

```python
def fama_macbeth_regression(returns_panel, factor_panel):
    """
    Fama-MacBeth 两步法
    returns_panel: DataFrame (periods × stocks)
    factor_panel: dict of DataFrames (periods × stocks)
    """
    factor_names = list(factor_panel.keys())
    n_periods = len(returns_panel)
    
    # 每期截面回归
    lambdas = {name: [] for name in factor_names}
    
    for t in range(n_periods - 1):
        # 当期因子值，下期收益率
        y = returns_panel.iloc[t+1]
        X_dict = {name: factor_panel[name].iloc[t] for name in factor_names}
        X_df = pd.DataFrame(X_dict).dropna()
        y_aligned = y[X_df.index].dropna()
        X_aligned = X_df.loc[y_aligned.index]
        
        if len(y_aligned) < 10:  # 样本太少跳过
            continue
        
        # 标准化因子
        X_std = (X_aligned - X_aligned.mean()) / X_aligned.std()
        X_with_const = sm.add_constant(X_std)
        
        try:
            model = sm.OLS(y_aligned, X_with_const).fit()
            for name in factor_names:
                lambdas[name].append(model.params.get(name, np.nan))
        except:
            pass
    
    # 对 lambda 时序做 t 检验
    results = {}
    for name in factor_names:
        lmb = np.array(lambdas[name])
        lmb = lmb[~np.isnan(lmb)]
        if len(lmb) > 0:
            t_stat, p_val = stats.ttest_1samp(lmb, 0)
            results[name] = {
                'lambda_mean': lmb.mean(),
                'lambda_std': lmb.std(),
                'IR': lmb.mean() / lmb.std() * np.sqrt(len(lmb)),
                't_stat': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
    
    return pd.DataFrame(results).T

# 模拟 60 期数据
n_periods = 60
n_stocks = 200

returns = pd.DataFrame(np.random.normal(0.005, 0.05, (n_periods, n_stocks)))
factor1 = pd.DataFrame(np.random.randn(n_periods, n_stocks))  # 价值因子
factor2 = pd.DataFrame(np.random.randn(n_periods, n_stocks))  # 动量因子

# 人为添加因子1的有效性
returns = returns + 0.015 * factor1.shift(1)

result = fama_macbeth_regression(returns, {'价值因子': factor1, '动量因子': factor2})
print("Fama-MacBeth 回归结果:")
print(result.round(4))
```

---

## 三、多重共线性诊断

```python
# VIF（方差膨胀因子）检验多重共线性
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = pd.DataFrame({
    '价值': np.random.randn(200),
    '价值_平方': np.random.randn(200),  # 与价值高度相关
    '动量': np.random.randn(200),
    '质量': np.random.randn(200),
})
X['价值_平方'] = X['价值'] ** 2 + np.random.randn(200) * 0.1  # 人为制造相关性

vif_data = pd.DataFrame({
    '因子': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})

print("VIF 诊断结果:")
print(vif_data)
print("\nVIF 解读：")
print("  VIF < 5:  可接受")
print("  VIF 5-10: 需要关注")
print("  VIF > 10: 严重共线性，需要处理")
```

---

## 今天的关键认识

1. **截面回归**：某一时点，用因子值横截面地解释收益率差异
2. **Fama-MacBeth**：每期截面回归 → 对 lambda 时序 t 检验，这是学界标准方法
3. **t 值 > 2**（约 p < 0.05）：因子溢价显著的经验标准
4. **VIF**：检验多重共线性，VIF > 10 的因子需要处理

---

## 明天预告

D6：**实战**——用 A 股真实数据跑一遍完整的因子分析流程。
