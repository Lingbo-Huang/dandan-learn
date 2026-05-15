---
layout: default
title: "D4 · 事件驱动策略"
render_with_liquid: false
---

# D4 · 事件驱动策略

> 公告、业绩、并购、政策变化……事件驱动策略通过解读这些信息比市场快一步。

---

## 1. 事件驱动策略分类

| 事件类型 | 举例 | 方向判断 |
|---------|------|---------|
| 业绩发布 | 盈利超预期/不及预期 | 超预期买入 |
| 分红/回购 | 大额回购公告 | 短期正面 |
| 并购重组 | 收购公告 | 目标公司买入 |
| 股东减持 | 大股东减持 | 短期负面 |
| 高管变更 | CEO 离职 | 视具体情况 |
| 监管政策 | 行业政策出台 | 受益方买入 |
| 指数调整 | 纳入沪深300 | 纳入前买入 |

---

## 2. 事件研究框架（Event Study）

```python
import pandas as pd
import numpy as np

class EventStudy:
    """
    标准事件研究方法
    计算事件前后的累积超额收益（CAR）
    """
    
    def __init__(self, estimation_window=(-250, -11),
                 event_window=(-5, 20)):
        self.est_start, self.est_end = estimation_window
        self.evt_start, self.evt_end = event_window
    
    def calc_abnormal_returns(self, stock_returns, market_returns):
        """
        计算超额收益（AR = 实际收益 - 预期收益）
        用估计期 OLS 拟合预期收益（市场模型）
        """
        # 用估计期数据拟合 beta
        est = stock_returns.iloc[self.est_start:self.est_end]
        mkt_est = market_returns.reindex(est.index)
        
        cov_mat = np.cov(est.values, mkt_est.values)
        beta = cov_mat[0, 1] / cov_mat[1, 1]
        alpha = est.mean() - beta * mkt_est.mean()
        
        # 计算事件期超额收益
        evt = stock_returns.iloc[self.evt_start:self.evt_end]
        mkt_evt = market_returns.reindex(evt.index)
        expected = alpha + beta * mkt_evt
        ar = evt - expected
        
        return ar
    
    def run(self, events_df, returns_df, market_returns):
        """
        批量运行事件研究
        events_df: DataFrame with columns ['date', 'stock', ...]
        returns_df: 全市场日度收益
        """
        cars = []
        
        for _, event in events_df.iterrows():
            stock = event['stock']
            event_date = event['date']
            
            if stock not in returns_df.columns:
                continue
            if event_date not in returns_df.index:
                continue
            
            # 对齐到事件日
            idx = returns_df.index.get_loc(event_date)
            start = idx + self.est_start
            end = idx + self.evt_end
            
            if start < 0 or end >= len(returns_df):
                continue
            
            stock_ret = returns_df[stock].iloc[start:end+1]
            mkt_ret = market_returns.iloc[start:end+1]
            
            # 重设 index 以便计算
            stock_ret.index = range(self.est_start, self.evt_end + 1)
            mkt_ret.index = range(self.est_start, self.evt_end + 1)
            
            ar = self.calc_abnormal_returns(stock_ret, mkt_ret)
            car = ar.cumsum()
            cars.append(car)
        
        if not cars:
            return pd.DataFrame()
        
        car_df = pd.DataFrame(cars)
        return car_df

    def plot_car(self, car_df):
        """绘制平均 CAR 曲线"""
        mean_car = car_df.mean()
        std_car = car_df.std()
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(mean_car.index, mean_car.values, label='平均 CAR')
        ax.fill_between(mean_car.index,
                        mean_car - 1.96 * std_car / (len(car_df)**0.5),
                        mean_car + 1.96 * std_car / (len(car_df)**0.5),
                        alpha=0.3, label='95% CI')
        ax.axvline(x=0, color='red', linestyle='--', label='事件日')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('事件日相对天数')
        ax.set_ylabel('累积超额收益 (CAR)')
        ax.legend()
        return fig
```

---

## 3. 盈利超预期策略（PEAD）

```python
def pead_strategy(earnings_df, price_df, hold_days=20):
    """
    盈利超预期后动量策略
    earnings_df: columns = ['date', 'stock', 'actual_eps', 'est_eps', 'price_std']
    """
    # 计算 SUE（标准化非预期盈利）
    earnings_df['sue'] = (
        (earnings_df['actual_eps'] - earnings_df['est_eps']) / 
        earnings_df['price_std']
    )
    
    # 选取 SUE 最高的前 20% 做多，最低的 20% 做空
    threshold_high = earnings_df['sue'].quantile(0.8)
    threshold_low = earnings_df['sue'].quantile(0.2)
    
    longs = earnings_df[earnings_df['sue'] >= threshold_high]
    shorts = earnings_df[earnings_df['sue'] <= threshold_low]
    
    return longs, shorts
```

---

## 4. 指数成分股调整套利

```python
def index_rebalancing_strategy(announcement_date, effective_date, 
                                 additions, deletions, returns_df):
    """
    利用指数调整套利：
    - 公告日买入新增股，卖出剔除股
    - 调整生效日之前平仓
    
    原理：被动基金必须在生效日调仓，在公告日到生效日之间形成预期需求
    """
    signal = {}
    
    for stock in additions:
        signal[stock] = 1  # 买入
    for stock in deletions:
        signal[stock] = -1  # 卖出
    
    # 持有从公告日到生效日的收益
    holding_period = returns_df.loc[announcement_date:effective_date]
    
    port_ret = pd.Series(0.0, index=holding_period.index)
    for stock, direction in signal.items():
        if stock in holding_period.columns:
            port_ret += direction * holding_period[stock] / len(signal)
    
    return port_ret
```

---

## 5. 事件驱动策略的风险

| 风险 | 说明 | 应对 |
|------|------|------|
| 信息泄露 | 事件前股价已经反映 | 检查事件日前的漂移 |
| 事件不确定性 | 公告结果与预期偏差大 | 分散多个事件 |
| 流动性风险 | 小市值事件股票不易成交 | 设置最低市值门槛 |
| 时序偏差 | 好事件和坏市场同期发生 | 做空市场 beta 对冲 |

---

## 小结

| 维度 | 内容 |
|------|------|
| 框架 | 事件研究（CAR = 累积超额收益）|
| 常见事件 | 业绩发布、指数调整、并购、政策 |
| PEAD | 盈利超预期后股价持续上涨 |
| 核心技术 | 事件窗口、市场模型、CAR 计算 |
| 主要风险 | 信息泄露、容量小、事件不确定 |
