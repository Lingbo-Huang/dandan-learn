---
layout: default
title: "D7 · Week 4 综合实战：完整多因子选股框架"
render_with_liquid: false
---

# D7 · Week 4 综合实战：完整多因子选股框架

> 把本周所学的所有因子拼在一起，构建一个从数据到信号的完整 Pipeline。

---

## 本周核心知识回顾

| 因子 | 核心逻辑 | 主要风险 |
|------|---------|---------|
| 动量 | 反应不足，价格趋势延续 | Momentum Crash |
| 价值 | 低估值均值回归 | 价值陷阱 |
| 质量 | 高质量公司盈利持续 | 行业偏移 |
| 低波动 | 低风险异象 | 市值偏移 |
| 多因子合成 | 分散化，提高稳定性 | 过拟合 |

---

## 完整多因子框架代码

```python
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize

class MultiFactor:
    """
    完整多因子选股框架
    输入：价格、财务数据
    输出：综合因子得分 + 持仓信号
    """
    
    def __init__(self, n_select=50, rebalance_freq='M'):
        self.n_select = n_select
        self.rebalance_freq = rebalance_freq
    
    # ===== 因子计算 =====
    
    def calc_momentum(self, close, lookback=6, skip=1):
        """截面动量因子"""
        ret = np.log(close / close.shift(1))
        mom = ret.rolling(lookback - skip).sum().shift(skip)
        return mom
    
    def calc_value(self, net_profit_ttm, mktcap):
        """E/P 价值因子"""
        return net_profit_ttm / mktcap
    
    def calc_quality(self, roe, cfo, total_assets, net_profit):
        """质量因子：ROE + 应计项目"""
        accruals = (net_profit - cfo) / total_assets
        quality = roe - accruals  # 高ROE + 低应计 = 高质量
        return quality
    
    def calc_low_vol(self, returns, window=60):
        """低波动因子（取负）"""
        return -returns.rolling(window).std()
    
    # ===== 预处理 =====
    
    def preprocess(self, factor, industry=None, mktcap=None):
        """
        截面预处理：winsorize -> 中性化 -> z-score
        """
        result = {}
        for date in factor.index:
            f = factor.loc[date].dropna()
            
            # 去极值
            f_clean = pd.Series(
                winsorize(f.values, limits=[0.025, 0.025]),
                index=f.index
            )
            
            # 市值中性化（简化版：减去 log_mktcap 回归）
            if mktcap is not None and date in mktcap.index:
                mc = np.log(mktcap.loc[date]).reindex(f_clean.index).dropna()
                common = f_clean.index.intersection(mc.index)
                if len(common) > 10:
                    coef = np.polyfit(mc[common].values, f_clean[common].values, 1)
                    f_clean = f_clean[common] - np.polyval(coef, mc[common])
            
            # Z-score
            mu, sigma = f_clean.mean(), f_clean.std()
            if sigma > 0:
                result[date] = (f_clean - mu) / sigma
        
        return pd.DataFrame(result).T
    
    # ===== 合成 =====
    
    def composite(self, factors_dict, weights=None):
        """
        多因子合成
        factors_dict: {name: processed_factor_df}
        weights: 各因子权重（默认等权）
        """
        dfs = list(factors_dict.values())
        names = list(factors_dict.keys())
        
        if weights is None:
            weights = [1.0 / len(dfs)] * len(dfs)
        
        composite = None
        for df, w in zip(dfs, weights):
            if composite is None:
                composite = df * w
            else:
                composite = composite.add(df * w, fill_value=0)
        
        return composite
    
    # ===== 选股 =====
    
    def select_stocks(self, composite_factor):
        """
        基于综合因子得分选股
        每期选 top N 个股票等权持仓
        """
        holdings = {}
        for date in composite_factor.index:
            scores = composite_factor.loc[date].dropna()
            top_stocks = scores.nlargest(self.n_select).index.tolist()
            holdings[date] = top_stocks
        return holdings
    
    # ===== 回测（简化版）=====
    
    def backtest(self, holdings, returns):
        """
        简单等权回测
        holdings: {date: [stock_list]}
        returns: 日度收益率 DataFrame
        """
        portfolio_returns = []
        dates = sorted(holdings.keys())
        
        for i, date in enumerate(dates):
            next_date = dates[i+1] if i+1 < len(dates) else returns.index[-1]
            stocks = holdings[date]
            
            # 持仓期收益（当前调仓日到下次调仓日）
            period = returns.loc[date:next_date, stocks].dropna(how='all')
            eq_ret = period.mean(axis=1)  # 等权
            portfolio_returns.append(eq_ret)
        
        return pd.concat(portfolio_returns).sort_index()
    
    # ===== 评估 =====
    
    def evaluate(self, port_returns, bench_returns=None):
        """基础绩效评估"""
        annual_ret = port_returns.mean() * 252
        annual_vol = port_returns.std() * (252 ** 0.5)
        sharpe = annual_ret / annual_vol
        
        # 最大回撤
        cumret = (1 + port_returns).cumprod()
        rolling_max = cumret.cummax()
        drawdown = (cumret - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        result = {
            '年化收益': f'{annual_ret:.2%}',
            '年化波动': f'{annual_vol:.2%}',
            '夏普比率': f'{sharpe:.2f}',
            '最大回撤': f'{max_dd:.2%}'
        }
        
        if bench_returns is not None:
            excess = port_returns - bench_returns.reindex(port_returns.index)
            result['年化超额收益'] = f'{excess.mean()*252:.2%}'
            result['信息比率'] = f'{excess.mean() / excess.std() * (252**0.5):.2f}'
        
        return pd.Series(result)


# ===== 使用示例 =====

def run_example():
    """示例：如何使用上述框架"""
    # 假设已有数据
    # close: 收盘价 DataFrame (index=date, columns=stock_code)
    # net_profit: 归母净利润
    # mktcap: 总市值
    # roe: ROE
    
    mf = MultiFactor(n_select=50)
    
    # 1. 计算各因子
    # momentum = mf.calc_momentum(close)
    # value = mf.calc_value(net_profit, mktcap)
    # quality = mf.calc_quality(roe, cfo, total_assets, net_profit)
    # low_vol = mf.calc_low_vol(close.pct_change())
    
    # 2. 预处理
    # mom_z = mf.preprocess(momentum, mktcap=mktcap)
    # val_z = mf.preprocess(value, mktcap=mktcap)
    # qual_z = mf.preprocess(quality, mktcap=mktcap)
    # lvol_z = mf.preprocess(low_vol, mktcap=mktcap)
    
    # 3. 合成（等权）
    # composite = mf.composite({
    #     'momentum': mom_z,
    #     'value': val_z,
    #     'quality': qual_z,
    #     'low_vol': lvol_z
    # })
    
    # 4. 选股
    # holdings = mf.select_stocks(composite)
    
    # 5. 回测
    # port_ret = mf.backtest(holdings, close.pct_change())
    
    # 6. 评估
    # metrics = mf.evaluate(port_ret)
    # print(metrics)
    pass
```

---

## 常见坑与注意事项

1. **未来函数**：使用财务数据时，确认是否已经发布（通常季报有披露延迟）
2. **幸存者偏差**：股票池必须包含当期所有可交易股票，包括后来退市的
3. **交易成本**：忽略双边 0.3% 以上的费用，回测结果会虚高
4. **因子正交性**：不同因子可能高度相关，等权合成可能只是重复信号
5. **市值效应**：多因子策略容易系统性偏向小市值，需要市值中性化

---

## 面试核心考点

- **如何判断一个因子是否有效？** → IC/ICIR + 分层收益 + 统计显著性检验
- **多因子权重如何确定？** → 等权（baseline）→ IC 加权 → 优化法（数据充足时）
- **如何避免过拟合？** → 样本外测试、参数敏感性、经济学逻辑验证
- **A股因子研究最大的坑？** → 未来函数、幸存者偏差、小市值偏移
