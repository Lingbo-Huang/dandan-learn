---
layout: default
title: "D5 · 策略评估框架"
render_with_liquid: false
---

# D5 · 策略评估框架

> 会跑回测不代表会评估策略。这一节把"看数据"变成系统性的评估方法论。

---

## 1. 核心绩效指标

### 1.1 收益类

```python
import pandas as pd
import numpy as np

def calc_performance_metrics(returns, freq=252, risk_free=0.02):
    """
    计算完整绩效指标
    returns: pd.Series，日度收益率
    freq: 年化频率（日=252，月=12）
    """
    # 年化收益
    annual_ret = returns.mean() * freq
    
    # 年化波动
    annual_vol = returns.std() * (freq ** 0.5)
    
    # 夏普比率
    sharpe = (annual_ret - risk_free) / annual_vol
    
    # 最大回撤
    cum_ret = (1 + returns).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Calmar 比率 = 年化收益 / |最大回撤|
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else np.nan
    
    # 胜率
    win_rate = (returns > 0).mean()
    
    # 盈亏比
    avg_win = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    
    # 信息比率（超额收益相对于基准的风险调整）
    # 此处假设基准为零
    ir = annual_ret / annual_vol
    
    metrics = {
        '年化收益': f'{annual_ret:.2%}',
        '年化波动': f'{annual_vol:.2%}',
        '夏普比率': f'{sharpe:.2f}',
        '最大回撤': f'{max_dd:.2%}',
        'Calmar 比率': f'{calmar:.2f}',
        '胜率': f'{win_rate:.2%}',
        '盈亏比': f'{payoff_ratio:.2f}'
    }
    return pd.Series(metrics)
```

---

## 2. 回撤分析

```python
def drawdown_analysis(returns):
    """详细回撤分析"""
    cum_ret = (1 + returns).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    
    # 找出每次回撤的开始/结束/恢复时间
    in_drawdown = drawdown < 0
    
    drawdown_periods = []
    start = None
    
    for date, is_dd in in_drawdown.items():
        if is_dd and start is None:
            start = date
        elif not is_dd and start is not None:
            # 回撤结束
            period = drawdown[start:date]
            drawdown_periods.append({
                'start': start,
                'trough': period.idxmin(),
                'end': date,
                'max_drawdown': period.min(),
                'duration': len(period)
            })
            start = None
    
    return pd.DataFrame(drawdown_periods)
```

---

## 3. 风险归因

```python
def factor_attribution(portfolio_returns, factor_returns):
    """
    因子收益归因
    portfolio_returns: 组合日度收益
    factor_returns: DataFrame，各因子日度收益
    """
    import statsmodels.api as sm
    
    X = sm.add_constant(factor_returns.reindex(portfolio_returns.index))
    model = sm.OLS(portfolio_returns, X).fit()
    
    attribution = {
        'alpha': model.params['const'] * 252,  # 年化 alpha
        'factor_betas': model.params.drop('const'),
        'r_squared': model.rsquared,
        'residual_vol': model.resid.std() * (252 ** 0.5)
    }
    return attribution
```

---

## 4. 策略稳健性检验

```python
def robustness_check(returns, n_bootstrap=1000):
    """
    Bootstrap 检验夏普比率的统计显著性
    """
    sharpes = []
    n = len(returns)
    
    for _ in range(n_bootstrap):
        sample = returns.sample(n, replace=True)
        s = sample.mean() / sample.std() * (252 ** 0.5)
        sharpes.append(s)
    
    sharpe_dist = pd.Series(sharpes)
    
    return {
        '原始夏普': returns.mean() / returns.std() * (252 ** 0.5),
        'Bootstrap均值': sharpe_dist.mean(),
        'Bootstrap标准差': sharpe_dist.std(),
        'p值（>0）': (sharpe_dist > 0).mean(),
        '95% CI下限': sharpe_dist.quantile(0.025),
        '95% CI上限': sharpe_dist.quantile(0.975)
    }

def walk_forward_validation(factor_df, return_df, 
                             train_window=36, test_window=6):
    """
    滚动样本外测试（Walk-Forward Analysis）
    避免过拟合，验证策略是否真实有效
    """
    results = []
    dates = factor_df.index
    
    for i in range(train_window, len(dates) - test_window, test_window):
        # 训练期：用于参数优化（这里简化为不优化）
        train_start = i - train_window
        train_end = i
        
        # 测试期：样本外
        test_start = i
        test_end = min(i + test_window, len(dates))
        
        test_factor = factor_df.iloc[test_start:test_end]
        test_return = return_df.iloc[test_start:test_end]
        
        # 计算测试期 IC
        for date in test_factor.index:
            if date in test_return.index:
                from scipy import stats
                f = test_factor.loc[date].dropna()
                r = test_return.loc[date].reindex(f.index).dropna()
                common = f.index.intersection(r.index)
                if len(common) > 10:
                    ic, _ = stats.spearmanr(f[common], r[common])
                    results.append({'date': date, 'ic': ic, 'is_oos': True})
    
    return pd.DataFrame(results).set_index('date')
```

---

## 5. 常见绩效陷阱

### 5.1 夏普比率的局限

- 假设收益正态分布（实际有胖尾）
- 不区分上行波动和下行波动
- 不考虑流动性风险

**替代指标**：Sortino Ratio（只计下行波动）

```python
def sortino_ratio(returns, risk_free=0.02, freq=252):
    """Sortino 比率：用下行波动替代总波动"""
    excess = returns - risk_free / freq
    downside = excess[excess < 0].std() * (freq ** 0.5)
    annual_excess = excess.mean() * freq
    return annual_excess / downside
```

### 5.2 多重检验问题

测试 100 个策略，即使都是随机的，也有约 5 个会显著（p < 0.05）。

**应对**：
- Bonferroni 校正：将显著性水平除以测试数量
- Benjamini-Hochberg：控制错误发现率（FDR）
- Harvey et al. (2016)：量化领域的发现至少需要 t 统计量 > 3.0

---

## 6. 评估清单

评估一个新策略时，依次检查：

- [ ] 年化收益 > 无风险利率
- [ ] 夏普比率 > 0.5（期货）或 > 1.0（股票长多）
- [ ] 最大回撤 < 20%（或满足风险预算）
- [ ] 样本外表现 > 样本内的 60%（无明显过拟合）
- [ ] 因子 IC > 0.03，ICIR > 0.5
- [ ] 交易成本后净夏普 > 0.5
- [ ] 策略有清晰的经济学逻辑（不只是统计）

---

## 小结

| 指标 | 公式 | 及格线 |
|------|------|-------|
| 年化夏普 | 年化超额收益/年化波动 | >1.0 |
| 最大回撤 | 峰值到谷底跌幅 | <20% |
| Calmar | 年化收益/|最大回撤| | >1.0 |
| 胜率 | 正收益天数比例 | >50% |
| 盈亏比 | 平均盈利/平均亏损 | >1.5 |
| Sortino | 超额收益/下行波动 | >1.5 |
