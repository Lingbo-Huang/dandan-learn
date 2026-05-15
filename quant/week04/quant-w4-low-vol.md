---
layout: default
title: "D4 · 低波动因子：风险异象与实现"
render_with_liquid: false
---

# D4 · 低波动因子：风险异象与实现

> 经典理论说高风险高回报，但实证发现低波动股票长期收益更好。这是最奇特的市场异象之一。

---

## 1. 低波动异象

**低波动异象（Low Volatility Anomaly）**：历史波动率低的股票，未来收益反而更高。

这直接违背了 CAPM：高 beta 不等于高回报。

经典实证：
- Black, Jensen & Scholes (1972)：beta 与收益的非线性关系
- Baker, Bradley & Wurgler (2011)：机构约束导致异象持续
- Frazzini & Pedersen (2014)：BAB 因子年化超额约 8%

---

## 2. 主要计算方式

### 历史波动率

```python
def calc_historical_vol(returns_df, window=252):
    """年化历史波动率"""
    return returns_df.rolling(window).std() * (252 ** 0.5)
```

### Beta 因子

```python
def calc_beta(stock_returns, market_returns, window=252):
    betas = {}
    for col in stock_returns.columns:
        s = stock_returns[col].dropna()
        m = market_returns.reindex(s.index).dropna()
        common = s.index.intersection(m.index)
        rolling_cov = s[common].rolling(window).cov(m[common])
        rolling_var = m[common].rolling(window).var()
        betas[col] = rolling_cov / rolling_var
    return pd.DataFrame(betas)
```

### 特质波动率（IVOL）

去除市场因子后的残差波动率，对低波动异象解释力更强。

---

## 3. 低波动异象的经济学解释

| 机制 | 说明 |
|------|------|
| 代理人问题 | 基金经理以相对收益考核，不愿持"无聊"低波股 |
| 杠杆约束 | 散户无法加杠杆，偏好高 beta 股替代 |
| 彩票偏好 | 散户喜欢高波动的"彩票股" |
| 信息稀缺 | 低波动股分析师覆盖少，可能长期低估 |

---

## 4. A股实践注意

- 涨跌停机制使极端波动率失真
- 小市值股票高波动，需市值中性化
- 行业内低波动效果优于全市场比较

```python
def vol_within_industry(vol_df, industry_df):
    """行业内标准化低波动因子"""
    result = vol_df.copy()
    for date in vol_df.index:
        for ind in industry_df.loc[date].unique():
            mask = industry_df.loc[date] == ind
            stocks = mask[mask].index
            slice_ = vol_df.loc[date, stocks].dropna()
            if len(slice_) < 3: continue
            z = (slice_ - slice_.mean()) / slice_.std()
            result.loc[date, slice_.index] = -z  # 取负：低波动得高分
    return result
```

---

## 小结

| 维度 | 内容 |
|------|------|
| 核心发现 | 低波动股票长期收益更高（违反 CAPM）|
| 常用指标 | 历史波动率、Beta、特质波动率 IVOL |
| 经典因子 | BAB (Frazzini & Pedersen) |
| A股注意 | 需市值/行业中性化 |
