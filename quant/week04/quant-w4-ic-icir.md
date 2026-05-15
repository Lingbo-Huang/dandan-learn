---
layout: default
title: "D5 · IC/ICIR 因子评价体系"
render_with_liquid: false
---

# D5 · IC/ICIR 因子评价体系

> IC 是量化研究员的日常语言。不懂 IC，面试必挂。

---

## 1. IC 的定义

**IC（Information Coefficient）**：因子值与未来收益率的截面相关系数。

IC_t = Corr(f_{i,t}, r_{i,t+1})

- IC > 0：正向预测
- IC < 0：负向预测（取反使用）

### Rank IC vs. Pearson IC

| 类型 | 方法 | 特点 |
|------|------|------|
| Pearson IC | 直接线性相关 | 对极端值敏感 |
| Rank IC | Spearman 秩相关 | 鲁棒，业界主流 |

```python
from scipy import stats
import numpy as np
import pandas as pd

def calc_rank_ic(factor_series, return_series):
    common = factor_series.dropna().index.intersection(return_series.dropna().index)
    if len(common) < 10:
        return np.nan
    corr, _ = stats.spearmanr(factor_series[common], return_series[common])
    return corr

def calc_ic_series(factor_df, return_df):
    ic_list = []
    for date in factor_df.index:
        if date not in return_df.index:
            continue
        ic = calc_rank_ic(factor_df.loc[date], return_df.loc[date])
        ic_list.append({'date': date, 'ic': ic})
    return pd.DataFrame(ic_list).set_index('date')['ic']
```

---

## 2. ICIR：稳定性度量

ICIR = mean(IC) / std(IC)

类比夏普比率：IC 均值衡量预测能力，标准差衡量稳定性。

```python
def calc_icir(ic_series, window=None):
    if window is None:
        return ic_series.mean() / ic_series.std()
    return ic_series.rolling(window).mean() / ic_series.rolling(window).std()
```

---

## 3. IC 评价标准（月频）

| 指标 | 差 | 可用 | 好 | 优秀 |
|------|----|----|----|----|
| 平均 IC | <0.02 | 0.02-0.04 | 0.04-0.06 | >0.06 |
| ICIR | <0.3 | 0.3-0.5 | 0.5-0.8 | >0.8 |
| 胜率（IC>0）| <50% | 50-55% | 55-60% | >60% |

---

## 4. 因子分层分析

IC 是线性度量，分层分析能捕捉非线性关系：

```python
def factor_quantile_analysis(factor_df, return_df, n_quantiles=5):
    results = []
    for date in factor_df.index:
        if date not in return_df.index:
            continue
        factor = factor_df.loc[date].dropna()
        ret = return_df.loc[date].reindex(factor.index).dropna()
        common = factor.index.intersection(ret.index)
        if len(common) < n_quantiles * 5: continue
        labels = pd.qcut(factor[common], q=n_quantiles, labels=False, duplicates='drop')
        group_ret = ret[common].groupby(labels).mean()
        row = {'date': date}
        for q in range(n_quantiles):
            row[f'Q{q+1}'] = group_ret.get(q, np.nan)
        results.append(row)
    df = pd.DataFrame(results).set_index('date')
    df['LS'] = df[f'Q{n_quantiles}'] - df['Q1']
    return df
```

---

## 5. Grinold 基本法则

IR = IC × sqrt(BR)

- IR：策略信息比率
- IC：因子平均 IC
- BR（Breadth）：独立下注次数

IC 越高、下注次数越多，策略 IR 越高。

---

## 6. 面试高频 Q&A

**Q：IC 为负的因子能用吗？**
A：能，取反即可。IC 绝对值衡量预测力，符号只决定方向。

**Q：为什么用 Rank IC 而不是 Pearson IC？**
A：Rank IC 对极端值鲁棒，不要求正态分布，在实际中更稳健。

**Q：IC 很高但 ICIR 很低，说明什么？**
A：因子不稳定，某些时期预测力强某些时期失效，实盘风险高。

---

## 小结

| 指标 | 含义 |
|------|------|
| IC | 截面预测能力 |
| Rank IC | 鲁棒预测能力（主流）|
| ICIR | 稳定性（类夏普比率）|
| IC 胜率 | 一致性（>55% 较好）|
| 因子分层 | 捕捉非线性预测关系 |
