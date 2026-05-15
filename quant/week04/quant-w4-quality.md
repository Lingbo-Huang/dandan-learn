---
layout: default
title: "D3 · 质量因子：盈利能力与财务稳健性"
render_with_liquid: false
---

# D3 · 质量因子：盈利能力与财务稳健性

> 好公司长期跑赢。质量因子把"好公司"量化成可以系统执行的信号。

---

## 1. 质量因子的经济学逻辑

- 市场持续低估高质量公司的盈利持续性
- 高质量公司破产风险低，但定价未完全反映
- Novy-Marx (2013) 发现毛利/总资产（GP/A）是很强的质量信号

---

## 2. 核心质量指标

### ROE 杜邦分解

ROE = (净利润/收入) × (收入/总资产) × (总资产/净资产)

三个驱动：净利率高、周转快、杠杆低 → 不同质量来源，需要区别对待。

### 盈利质量：应计项目

Accruals = (净利润 - 经营现金流) / 总资产

应计项目越低，利润中现金含量越高，盈利质量越好。

---

## 3. Piotroski F-Score

Piotroski (2000) 提出 9 个财务指标打分（0-9分），高分跑赢低分：

```python
def calc_f_score(data):
    score = 0
    # 盈利信号
    if data['roa'] > 0: score += 1
    if data['cfo'] > 0: score += 1
    if data['roa'] > data['roa_prev']: score += 1
    if data['cfo'] / data['assets'] > data['roa']: score += 1
    # 财务杠杆/流动性
    if data['leverage'] < data['leverage_prev']: score += 1
    if data['current_ratio'] > data['current_ratio_prev']: score += 1
    if data['shares'] <= data['shares_prev']: score += 1
    # 运营效率
    if data['gross_margin'] > data['gross_margin_prev']: score += 1
    if data['asset_turnover'] > data['asset_turnover_prev']: score += 1
    return score
```

---

## 4. 综合质量因子（QMJ 框架）

```python
import pandas as pd
import numpy as np

def build_quality_factor(df):
    def zscore(s):
        return (s - s.mean()) / (s.std() + 1e-8)
    
    # 盈利能力
    profitability = (zscore(df['roe']) + zscore(df['roa']) + 
                     zscore(df['gross_margin'])) / 3
    # 安全性（取负）
    safety = (zscore(-df['accruals']) + zscore(-df['debt_ratio'])) / 2
    # 成长性
    growth = (zscore(df['roe_growth']) + zscore(df['roa_growth'])) / 2
    
    return (profitability + growth + safety) / 3
```

---

## 5. A股实践要点

- 财务造假较多 → 应计项目指标更重要
- 国企盈利质量普遍低于民企
- 用 F-Score >= 6 过滤股票池，再结合价值因子
- 避开 ROE 来自高杠杆的金融股

---

## 小结

| 指标 | 方向 | 代表意义 |
|------|------|---------|
| ROE / ROA | 越高越好 | 盈利能力 |
| GP/Assets | 越高越好 | 毛利质量 |
| Accruals | 越低越好 | 现金含量 |
| Leverage | 越低越好 | 财务稳健 |
| F-Score | >=6 优质 | 综合质量 |
