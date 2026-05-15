---
layout: default
title: "D6 · 多因子合成：从打分法到优化法"
render_with_liquid: false
---

# D6 · 多因子合成：从打分法到优化法

> 单因子预测能力有限。多因子合成是将分散的信号聚合成稳健 alpha 的关键工程。

---

## 1. 为什么要多因子合成

- 单因子在不同市场环境下表现各异
- 低相关因子组合可以降低特定风险
- 提高信号稳定性，降低换手率

---

## 2. 因子标准化（预处理）

合成之前必须标准化，否则量纲不同无法比较。

### 步骤

1. **去极值**：winsorize 或 MAD 法
2. **标准化**：截面 Z-score
3. **中性化**：市值 + 行业中性化

```python
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def preprocess_factor(factor_series, industry_series=None, mktcap_series=None):
    """因子预处理：去极值 -> 中性化 -> 标准化"""
    # 1. 去极值（上下 2.5%）
    f = factor_series.dropna()
    f_wins = pd.Series(
        winsorize(f.values, limits=[0.025, 0.025]),
        index=f.index
    )
    
    # 2. 市值/行业中性化（截面回归取残差）
    if industry_series is not None or mktcap_series is not None:
        X_parts = []
        if industry_series is not None:
            X_parts.append(pd.get_dummies(industry_series.reindex(f.index)))
        if mktcap_series is not None:
            X_parts.append(np.log(mktcap_series.reindex(f.index)).rename('log_mktcap'))
        X = pd.concat(X_parts, axis=1).dropna()
        common = f_wins.index.intersection(X.index)
        coef, _, _, _ = np.linalg.lstsq(X.loc[common].values, f_wins[common].values, rcond=None)
        residual = f_wins[common] - X.loc[common].values @ coef
        f_wins = residual
    
    # 3. 截面 Z-score
    f_z = (f_wins - f_wins.mean()) / (f_wins.std() + 1e-8)
    return f_z
```

---

## 3. 等权合成（打分法）

最简单的合成方式，直接等权平均各因子 Z-score：

```python
def equal_weight_composite(factors_dict):
    """
    factors_dict: {factor_name: factor_series (already z-scored)}
    """
    df = pd.DataFrame(factors_dict)
    return df.mean(axis=1)
```

**优点**：简单、鲁棒、不容易过拟合
**缺点**：忽略因子预测力差异

---

## 4. IC 加权合成

用历史 IC（或 ICIR）作为权重：

```python
def ic_weighted_composite(factors_dict, ic_dict):
    """
    ic_dict: {factor_name: 历史平均 IC}
    """
    df = pd.DataFrame(factors_dict)
    weights = pd.Series(ic_dict)
    # 用 IC 绝对值作为权重（方向已在 z-score 中体现）
    weights = weights.abs()
    weights = weights / weights.sum()  # 归一化
    return (df * weights).sum(axis=1)
```

---

## 5. 优化法合成（均值-方差）

利用因子的协方差矩阵，最大化信息比率：

```python
from scipy.optimize import minimize

def optimize_factor_weights(factor_returns_df, lambda_=0.5):
    """
    factor_returns_df: 各因子多空组合的历史收益
    lambda_: 风险厌恶系数
    最大化 w'μ - lambda * w'Σw
    """
    mu = factor_returns_df.mean()
    sigma = factor_returns_df.cov()
    n = len(mu)
    
    def neg_utility(w):
        w = np.array(w)
        return -(w @ mu - lambda_ * w @ sigma @ w)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n  # 不允许做空单个因子
    
    result = minimize(neg_utility, x0=[1/n]*n, 
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    return pd.Series(result.x, index=mu.index)
```

---

## 6. 因子正交化

处理因子间的相关性，避免重复计数：

```python
def orthogonalize_factors(primary_factor, control_factors):
    """
    对主因子做截面回归，剔除控制因子的影响
    primary_factor: pd.Series
    control_factors: pd.DataFrame
    """
    X = control_factors.reindex(primary_factor.index).dropna()
    y = primary_factor.reindex(X.index).dropna()
    common = y.index.intersection(X.index)
    
    coef, _, _, _ = np.linalg.lstsq(X.loc[common].values, y[common].values, rcond=None)
    residual = y[common] - X.loc[common].values @ coef
    return residual
```

---

## 7. 合成方法比较

| 方法 | 复杂度 | 样本需求 | 过拟合风险 | 适用场景 |
|------|--------|---------|----------|---------|
| 等权 | 低 | 少 | 低 | 通用首选 |
| IC 加权 | 中 | 中 | 中 | 因子预测力差异明显 |
| 优化法 | 高 | 多 | 高 | 因子数量多、数据充足 |
| 机器学习 | 高 | 多 | 很高 | 需严格控制过拟合 |

---

## 小结

- 等权法是 baseline，在数据不足时最稳健
- IC 加权给预测力强的因子更多权重
- 优化法理论最优，但实践中容易过拟合
- 任何方法之前都必须做标准化和中性化
