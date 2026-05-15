---
layout: default
title: "D2 · 价值因子：PE/PB/PS/DCF 体系"
render_with_liquid: false
---

# D2 · 价值因子：PE/PB/PS/DCF 体系

> 价值投资的量化化：从格雷厄姆到 Fama-French，从主观判断到系统因子。

---

## 1. 价值因子的核心思想

**价值因子**：被市场"低估"的股票，长期来看会均值回归，产生超额收益。

Fama & French (1992) 发现 **账面市值比（B/M = 1/PB）** 能解释截面收益差异，这是第一个被严格统计验证的价值因子。

---

## 2. 主要价值指标

### 2.1 E/P（收益率）

$$
\text{E/P} = \frac{\text{归母净利润(TTM)}}{\text{总市值}}
$$

使用 E/P 而非 PE，原因是负 PE 无意义但可以有负 E/P（负值说明亏损）。

### 2.2 B/M（账面市值比）

$$
\text{B/M} = \frac{\text{净资产}}{\text{总市值}}
$$

Fama-French 三因子中的价值代理变量，对金融股更有意义。

### 2.3 S/P（市销倒数）

$$
\text{S/P} = \frac{\text{营业收入(TTM)}}{\text{总市值}}
$$

适合亏损的成长公司估值。

### 2.4 CF/P（现金流收益率）

$$
\text{CF/P} = \frac{\text{经营现金流(TTM)}}{\text{总市值}}
$$

比 E/P 更难被会计操纵。

---

## 3. 价值因子工程实现

```python
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

class ValueFactor:
    def compute_ep(self, net_profit_ttm, total_mktcap):
        """E/P 因子"""
        return net_profit_ttm / total_mktcap
    
    def compute_bm(self, book_value, total_mktcap):
        """B/M 因子"""
        return book_value / total_mktcap
    
    def compute_sp(self, revenue_ttm, total_mktcap):
        """S/P 因子"""
        return revenue_ttm / total_mktcap
    
    def compute_cfp(self, cfo_ttm, total_mktcap):
        """CF/P 因子"""
        return cfo_ttm / total_mktcap
    
    def composite_value(self, ep, bm, sp, cfp):
        """合成价值因子：各分项 Z-score 等权平均"""
        factors = {'ep': ep, 'bm': bm, 'sp': sp, 'cfp': cfp}
        zscores = {}
        for name, f in factors.items():
            f_clean = f.replace([np.inf, -np.inf], np.nan).dropna()
            # winsorize 去极值（上下 2.5%）
            f_wins = pd.Series(
                winsorize(f_clean.values, limits=[0.025, 0.025]),
                index=f_clean.index
            )
            zscores[name] = (f_wins - f_wins.mean()) / f_wins.std()
        return pd.DataFrame(zscores).mean(axis=1)
```

---

## 4. 价值陷阱的识别

低估值未必是好机会，需要过滤"价值陷阱"：

| 陷阱类型 | 识别方法 |
|---------|---------|
| 行业结构性衰退 | 结合行业景气指标 |
| 管理层质量差 | ROE 持续下滑、大量关联交易 |
| 财务造假 | 应计利润率高、现金流与利润背离 |
| 债务危机 | 利息覆盖倍数 < 1.5 |

```python
def filter_value_traps(value_factor, quality_score, min_quality=3):
    """
    过滤价值陷阱：保留质量分数高于阈值的低估值股票
    quality_score: Piotroski F-Score (0-9)
    """
    mask = quality_score >= min_quality
    return value_factor.where(mask)
```

---

## 5. 价值因子的时效性

A股价值因子呈现明显的**周期性**：

- 经济上行、利率上升期：价值跑赢
- 流动性宽松、成长溢价高期：价值跑输
- 市场恐慌时：价值股因低估值提供安全垫

**策略建议**：不要单押价值，与成长/动量因子组合使用。

---

## 小结

| 因子 | 优点 | 缺点 |
|------|------|------|
| E/P | 最直接的盈利估值 | 周期行业利润波动大 |
| B/M | 经典 FF 因子，研究充分 | 轻资产公司失效 |
| S/P | 适合亏损公司 | 忽略盈利能力 |
| CF/P | 财务质量更高 | 计算复杂，数据要求高 |
