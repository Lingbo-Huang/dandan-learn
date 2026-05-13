---
layout: default
title: "WhaleQuant Ch02 · 金融市场基础概念"
source: "https://github.com/datawhalechina/whale-quant"
---

# 02 · 金融市场基础概念

> **来源**：[WhaleQuant](https://github.com/datawhalechina/whale-quant) · datawhalechina

---

## 2.1 宏观经济学基础概念

### GDP 与经济周期

**GDP（国内生产总值）**：一国在特定时期内生产的所有最终商品和服务的市场价值之和。

**经济周期四阶段**：
```
扩张 → 顶峰 → 衰退 → 谷底 → （新一轮扩张）
```

量化策略与经济周期：
- **扩张期**：成长股、周期股表现好
- **衰退期**：防御股（消费、医疗、公用事业）表现好
- **通胀上升**：商品、能源、TIPS 债券
- **通胀下降**：长期债券、成长股

### 货币政策工具

| 工具 | 紧缩（加息/降准） | 宽松（降息/扩表） |
|------|----------------|----------------|
| 基准利率 | 提高 → 借贷成本上升 | 降低 → 刺激信贷 |
| 存款准备金率 | 提高 → 银行放贷减少 | 降低 → 信贷扩张 |
| 公开市场操作 | 卖出国债 → 回收流动性 | 买入国债 → 释放流动性 |

---

## 2.2 货币金融学基础

### 利率与债券

**债券定价公式**：

$$P = \sum_{t=1}^{T} \frac{C}{(1+r)^t} + \frac{F}{(1+r)^T}$$

- $C$：每期票息
- $F$：面值
- $r$：折现率（市场利率）
- 利率↑ → 债券价格↓（反向关系）

**久期（Duration）**：债券价格对利率变动的敏感度
$$\text{修正久期} = -\frac{1}{P} \cdot \frac{dP}{dr}$$

### 汇率与量化

```python
# 汇率对股票影响示例
import pandas as pd
import numpy as np

# 出口型企业：人民币升值 → 利润减少（以外币结算的营收换算回来少了）
# 进口型企业：人民币升值 → 成本降低（进口商品更便宜）

def estimate_fx_impact(revenue_usd, cost_rmb, usd_cny_rate):
    """估算汇率变动对企业利润的影响"""
    revenue_rmb = revenue_usd * usd_cny_rate
    profit = revenue_rmb - cost_rmb
    return profit
```

---

## 2.3 投资学基础概念

### 收益率计算

```python
import numpy as np
import pandas as pd

# 简单收益率
def simple_return(p1, p0):
    return (p1 - p0) / p0

# 对数收益率（连续复利）
def log_return(p1, p0):
    return np.log(p1 / p0)

# 年化收益率（日频数据）
def annualize_return(daily_returns, trading_days=252):
    return daily_returns.mean() * trading_days

# 年化波动率
def annualize_vol(daily_returns, trading_days=252):
    return daily_returns.std() * np.sqrt(trading_days)

# 夏普比率
def sharpe_ratio(daily_returns, risk_free_rate=0.02, trading_days=252):
    excess = daily_returns - risk_free_rate / trading_days
    return excess.mean() / excess.std() * np.sqrt(trading_days)
```

### CAPM 模型

$$E(R_i) = R_f + \beta_i \cdot (E(R_m) - R_f)$$

- $R_f$：无风险利率（如国债）
- $\beta_i$：股票相对市场的系统性风险
- $E(R_m) - R_f$：市场风险溢价

```python
from scipy import stats

def estimate_beta(stock_returns, market_returns):
    """OLS 估计 Beta"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        market_returns, stock_returns
    )
    return slope, intercept  # beta, alpha

# Beta > 1：比市场波动更大（进攻型）
# Beta < 1：比市场波动更小（防守型）
# Beta < 0：与市场负相关（对冲工具）
```

### 现代投资组合理论（MPT）

```python
import numpy as np

def portfolio_performance(weights, returns, cov_matrix, rf=0.02/252):
    """计算投资组合的收益和风险"""
    port_return = np.dot(weights, returns.mean()) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (port_return - rf * 252) / port_vol
    return port_return, port_vol, sharpe

def efficient_frontier(returns, n_portfolios=10000):
    """蒙特卡洛模拟有效前沿"""
    n_assets = len(returns.columns)
    results = []
    
    for _ in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))
        ret, vol, sr = portfolio_performance(weights, returns, returns.cov())
        results.append([ret, vol, sr])
    
    return np.array(results)
```

---

## 2.4 数理统计基本概念

### 在量化中最常用的统计量

```python
import pandas as pd
import numpy as np
from scipy import stats

def quant_stats(returns: pd.Series):
    """量化策略核心统计量"""
    
    # 基础统计
    mean = returns.mean() * 252
    std = returns.std() * np.sqrt(252)
    
    # 高阶矩
    skewness = stats.skew(returns)   # 偏度：负偏=下行风险大
    kurtosis = stats.kurtosis(returns)  # 超额峰度：正=肥尾
    
    # 风险指标
    var_95 = np.percentile(returns, 5)   # 历史 VaR（95%置信）
    
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    max_dd = ((cumulative - rolling_max) / rolling_max).min()
    
    # 绩效指标
    sharpe = mean / std if std > 0 else 0
    calmar = mean / abs(max_dd) if max_dd != 0 else 0
    
    return {
        '年化收益': f'{mean:.2%}',
        '年化波动': f'{std:.2%}',
        '夏普比率': f'{sharpe:.3f}',
        '最大回撤': f'{max_dd:.2%}',
        'Calmar比率': f'{calmar:.3f}',
        '偏度': f'{skewness:.3f}',
        '超额峰度': f'{kurtosis:.3f}',
        '日VaR(95%)': f'{var_95:.2%}',
    }

# 使用
import yfinance as yf
data = yf.download('000001.SS', start='2020-01-01')['Close']
daily_returns = data.pct_change().dropna()
stats_result = quant_stats(daily_returns)
for k, v in stats_result.items():
    print(f'{k}: {v}')
```

---

## 延伸阅读

- [WhaleQuant 完整教程](https://github.com/datawhalechina/whale-quant)
- 《投资学》（博迪 / Bodie）
- 《量化投资：数量技术与策略》（朱嘉明）
