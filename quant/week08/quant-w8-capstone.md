---
layout: default
title: "D7 · Week 8 终极综合模拟面试"
render_with_liquid: false
---

# D7 · Week 8 终极综合模拟面试

> 把 8 周学到的所有知识整合成一套完整的面试应对体系。

---

## 量化面试全景图

```
量化研究员面试
├── 技术轮（编程题）
│   ├── Pandas/NumPy 数据处理
│   ├── 因子计算与回测
│   └── 算法题（视公司而定）
├── 技术轮（统计/数学）
│   ├── 概率推理
│   ├── 统计检验
│   └── 随机过程基础
├── 案例轮
│   ├── 因子设计
│   ├── 策略评估
│   └── 问题分析
└── HR 轮（行为面试）
    ├── 项目展示
    └── 价值观匹配
```

---

## 模拟面试题库

### 【一面：技术编程】

**题目 1**：给定日度 OHLCV 数据，计算每只股票的 20 日布林带信号。

```python
def bollinger_signal(close_df, window=20, n_std=2):
    """
    布林带信号
    返回：1=突破上轨（趋势强），-1=突破下轨，0=在带内
    """
    ma = close_df.rolling(window).mean()
    std = close_df.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    
    signal = pd.DataFrame(0, index=close_df.index, columns=close_df.columns)
    signal[close_df > upper] = 1
    signal[close_df < lower] = -1
    
    return signal, ma, upper, lower
```

**题目 2**：实现 Fama-MacBeth 横截面回归

```python
def fama_macbeth(returns_df, factors_dict, min_stocks=20):
    """
    Fama-MacBeth 两步回归
    Step 1: 每期截面回归，得到因子收益
    Step 2: 对时序因子收益做统计检验
    
    returns_df: 个股收益率（index=date, columns=stock）
    factors_dict: {factor_name: factor_df} 同结构
    """
    import statsmodels.api as sm
    from scipy import stats
    
    factor_names = list(factors_dict.keys())
    dates = returns_df.index
    
    period_coefs = []
    
    for date in dates:
        y = returns_df.loc[date].dropna()
        
        X_parts = {}
        for name, factor_df in factors_dict.items():
            if date in factor_df.index:
                X_parts[name] = factor_df.loc[date].reindex(y.index)
        
        if not X_parts:
            continue
        
        X = pd.DataFrame(X_parts).dropna()
        y = y.reindex(X.index).dropna()
        X = X.reindex(y.index)
        
        if len(y) < min_stocks:
            continue
        
        X_const = sm.add_constant(X)
        try:
            model = sm.OLS(y, X_const).fit()
            period_coefs.append(model.params.to_dict())
        except:
            continue
    
    coefs_df = pd.DataFrame(period_coefs)
    
    # Step 2: t 检验
    results = {}
    for col in coefs_df.columns:
        series = coefs_df[col].dropna()
        t_stat, p_value = stats.ttest_1samp(series, 0)
        results[col] = {
            'mean': series.mean(),
            't_stat': t_stat,
            'p_value': p_value
        }
    
    return pd.DataFrame(results).T
```

---

### 【二面：统计数学】

**Q1：什么是协方差矩阵的特征值？在量化中有什么用？**

```
特征值和特征向量分解：Σ = QΛQ^T

在量化中的应用：
1. PCA 降维：用前 K 个主成分替代 N 个因子，去除冗余
2. 风险分解：特征值越大，对应主成分承担越多风险
3. 最优组合：最小方差组合 = 用最小特征值对应的特征向量构建
4. 因子正交化：将相关因子变成正交因子

Python 实现：
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
# 按特征值降序排列
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
```

**Q2：OLS 回归的四大假设，违反时如何处理？**

```
假设 1: 线性关系 → 违反时: 加非线性项，或用非参数方法
假设 2: 无多重共线性 → 违反时: PCA 降维，或 Ridge 回归
假设 3: 同方差 → 违反时: WLS（加权最小二乘）
假设 4: 误差独立 → 违反时: Newey-West 标准误（时序自相关）

量化实践中：
- Fama-MacBeth 用截面回归 + 时序均值，自然处理截面相关
- 因子之间多重共线性常见，通常用正交化处理
```

---

### 【三面：案例设计】

**完整案例**：设计一个适合 5 亿 AUM 的量化策略

```
Step 1: 容量分析
5亿 AUM，月换手 50%，分散到 50 只股票
每只股票平均换手金额 = 5亿 × 50% / 50 = 500万/股
需要 ADV > 5000万（500万/10% 参与率上限）
→ 需选流动性好的中大盘股（沪深300成分股为主）

Step 2: 因子选择
考虑容量约束：
- 大盘股动量因子有效性较好（小盘动量更强但容量小）
- 价值+质量组合：周期性相对稳定
- 低波动：容量大，与市值因子正相关

Step 3: 风险约束
- 单股最大仓位：3%
- 行业集中度：单行业不超过 30%
- 相对沪深300 active share：目标 50-70%

Step 4: 验证标准
- 目标：样本外年化超额 5%，跟踪误差 < 8%，信息比率 > 0.8
- 最大回撤 < -15%（相对基准）
- Walk-Forward 样本外 IC > 0.03
```

---

## 8 周学习回顾

| 周次 | 核心知识 | 面试权重 |
|------|---------|---------|
| W1-W3 | 金融基础、统计 | 背景知识 |
| W4 | 因子体系：动量/价值/质量/低波 | ★★★★★ |
| W5 | 策略开发：CTA/套利/评估框架 | ★★★★ |
| W6 | ML 因子：Alpha158/LightGBM/过拟合 | ★★★★ |
| W7 | 高频执行：微结构/VWAP/冲击成本 | ★★★ |
| W8 | 面试准备：编程/概率/案例/简历 | ★★★★★ |

---

## 最后叮嘱

**技术是门票，但不是全部。**

量化面试官最终想知道的：
1. **你能独立解决问题吗？**（不是照着教程复现，而是面对新问题）
2. **你理解背后的逻辑吗？**（不只是会调参，知道为什么这么做）
3. **你有量化思维吗？**（怀疑数据，量化一切，有假设有验证）

带着这三个标准审视你的每一个项目，你已经准备好了。

---

**Good Luck！**
