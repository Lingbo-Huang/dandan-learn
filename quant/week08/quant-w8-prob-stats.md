---
layout: default
title: "D2 · 概率与统计考题"
render_with_liquid: false
---

# D2 · 概率与统计考题

> 量化面试的概率统计题，考察的是思维方式，而不是死记公式。

---

## 1. 期望与方差

### 基础题

**Q1：抛一枚硬币，正面赢 1 元，反面输 1 元，赢 3 次或累计亏 1 次停止。期望收益？**

```
状态分析：
- 赢3次：概率 = (1/2)^3 = 1/8，收益 = 3
- 中途某次亏损停止，亏损 = 1

用递推或暴力枚举：
E[收益] = P(赢3次) × 3 + P(先亏) × (-1)
= 1/8 × 3 + 7/8 × (-1) = 3/8 - 7/8 = -4/8 = -0.5 元

结论：期望亏损 0.5 元，不参与这个游戏。
```

**Q2：N 个点均匀分布在圆上，相邻两点距离的期望？**

```
圆周长为 2π（标准化），N 个间距之和 = 2π
每个间距期望 = 2π/N

答：E[间距] = 2π/N
```

---

## 2. 概率推理

**Q3：蒙提霍尔问题**

你选了门 A，主持人打开门 B（空）。换还是不换？

```
不换：P(赢) = 1/3（选对了 A）
换：  P(赢) = 2/3（一开始选错了，换就赢）

应该换！直觉陷阱：很多人以为 50/50，实际上主持人的行为包含信息。

Python 验证：
import random
def monty_hall(switch=True, n=100000):
    wins = 0
    for _ in range(n):
        car = random.randint(0, 2)
        choice = random.randint(0, 2)
        # 主持人打开一扇没车且不是你选的门
        doors = [0, 1, 2]
        host_opens = [d for d in doors if d != choice and d != car][0]
        if switch:
            choice = [d for d in doors if d != choice and d != host_opens][0]
        if choice == car:
            wins += 1
    return wins / n

print(monty_hall(switch=True))   # ~0.667
print(monty_hall(switch=False))  # ~0.333
```

---

## 3. 贝叶斯推理

**Q4：某测试敏感度 99%，特异度 99%，疾病发病率 1%。测试阳性，真正患病概率？**

```
P(阳性|患病) = 0.99（敏感度）
P(阳性|未患病) = 0.01（假阳性率 = 1 - 特异度）
P(患病) = 0.01

贝叶斯：
P(患病|阳性) = P(阳性|患病) × P(患病) / P(阳性)
= 0.99 × 0.01 / (0.99×0.01 + 0.01×0.99)
= 0.0099 / (0.0099 + 0.0099)
= 50%！

直觉误导：看起来 99% 准确的测试，阳性者只有 50% 真正患病。
原因：基率太低（1%），假阳性数量和真阳性差不多。
```

这道题对量化的启示：**因子预测精度再高，如果信号稀少（基率低），实际准确率可能很低。**

---

## 4. 随机过程

**Q5：随机游走 — 期望到达时间**

在数轴上从 0 出发，每步 ±1 等概率，到达 N 或 -N 停止。期望步数？

```
设 E[n] = 从 0 出发的期望步数
答案：E[n] = N²

推导（对称随机游走）：
令 E[k] = 从位置 k 到达 ±N 的期望步数
E[k] = 1 + 0.5 * E[k+1] + 0.5 * E[k-1]
边界：E[N] = E[-N] = 0

解：E[k] = N² - k²
从 k=0 出发：E[0] = N²
```

**Q6：布朗运动与期权定价（Black-Scholes 核心）**

股价服从几何布朗运动：

$$
dS = \mu S dt + \sigma S dW_t
$$

- μ：漂移率（年化收益）
- σ：波动率
- dW：维纳过程（标准布朗运动的增量）

欧式看涨期权价格：
$$
C = S_0 N(d_1) - K e^{-rT} N(d_2)
$$

```python
from scipy.stats import norm
import numpy as np

def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes 欧式看涨期权定价
    S: 当前股价
    K: 行权价
    T: 到期时间（年）
    r: 无风险利率
    sigma: 波动率
    """
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return C

def bs_greeks(S, K, T, r, sigma):
    """计算期权 Greeks"""
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1)  # 期权对股价的敏感度
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))  # Delta 的变化率
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
             r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # 对波动率的敏感度（每1%）
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}
```

---

## 5. 统计检验

**Q7：如何检验一个因子是否有效？**

```python
from scipy import stats

def test_factor_significance(ic_series, alpha=0.05):
    """
    检验因子 IC 是否显著不为零
    H0: mean(IC) = 0（无预测能力）
    """
    n = len(ic_series.dropna())
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    
    # t 统计量
    t_stat = mean_ic / (std_ic / np.sqrt(n))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    # Harvey et al. (2016) 建议量化领域用 t > 3.0
    harvey_significant = abs(t_stat) > 3.0
    
    print(f"样本量: {n}")
    print(f"平均 IC: {mean_ic:.4f}")
    print(f"t 统计量: {t_stat:.2f}")
    print(f"p 值: {p_value:.4f}")
    print(f"传统显著性(p<0.05): {p_value < alpha}")
    print(f"Harvey 准则(t>3.0): {harvey_significant}")
    
    return t_stat, p_value
```

---

## 6. 面试答题技巧

1. **画图/列举**：概率题先画事件树或列所有情况
2. **验证简单情况**：N=2 或 N=1 时结果是否合理
3. **说清楚假设**：独立性、分布假设
4. **写代码验证**：大多数概率题可以用模拟验证

---

## 常考公式速记

| 公式 | 内容 |
|------|------|
| E[X+Y] = E[X] + E[Y] | 期望线性性 |
| Var[X+Y] = Var[X] + Var[Y] + 2Cov(X,Y) | 方差加法 |
| Cov(X,Y) = E[XY] - E[X]E[Y] | 协方差定义 |
| E[X] = ΣP(X=x)·x | 离散期望 |
| 贝叶斯 P(A|B) = P(B|A)P(A)/P(B) | 条件概率更新 |
| N(d1)-N(d2) | B-S 期权公式核心 |
