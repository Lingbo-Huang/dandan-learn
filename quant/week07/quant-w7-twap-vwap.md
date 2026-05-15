---
layout: default
title: "D3 · TWAP 与 VWAP 算法执行"
render_with_liquid: false
---

# D3 · TWAP 与 VWAP 算法执行

> 如何把 100 万股的大单"悄悄"买入，不让价格飞起来？算法执行是量化工程的关键一环。

---

## 1. 为什么需要算法执行

**问题**：大额订单直接市价成交会：
1. 推高买入价格（市场影响）
2. 暴露交易意图（信息泄露）
3. 留下市场痕迹被其他人跟踪

**解决**：算法执行（Algorithmic Trading）——把大单拆成小单，分时段执行。

---

## 2. TWAP（时间加权平均价格）

**TWAP**：将总订单均匀分散在时间窗口内执行。

$$
\\text{TWAP} = \\frac{1}{T}\\sum_{t=1}^{T} p_t
$$

```python
import pandas as pd
import numpy as np

class TWAPExecution:
    """
    TWAP 算法执行器
    """
    
    def __init__(self, total_qty, start_time, end_time, interval_minutes=5):
        self.total_qty = total_qty
        self.start_time = start_time
        self.end_time = end_time
        self.interval_minutes = interval_minutes
        
        # 计算切片数量
        total_minutes = (end_time - start_time).seconds // 60
        self.n_slices = total_minutes // interval_minutes
        
        # 每片数量（均等）
        base_qty = total_qty // self.n_slices
        remainder = total_qty % self.n_slices
        
        self.slice_qtys = [base_qty] * self.n_slices
        # 余量分配到最后一片
        self.slice_qtys[-1] += remainder
    
    def get_slice_schedule(self):
        """生成执行时间表"""
        schedule = []
        current_time = self.start_time
        
        for qty in self.slice_qtys:
            schedule.append({
                'time': current_time,
                'quantity': qty,
                'strategy': 'market'  # 或 'limit'
            })
            current_time += pd.Timedelta(minutes=self.interval_minutes)
        
        return pd.DataFrame(schedule)
    
    def simulate_execution(self, price_series):
        """
        模拟执行并计算实际成交均价
        price_series: 分钟线收盘价
        """
        schedule = self.get_slice_schedule()
        total_cost = 0
        total_filled = 0
        
        for _, order in schedule.iterrows():
            time = order['time']
            qty = order['quantity']
            
            if time in price_series.index:
                price = price_series[time]
            else:
                # 找最近的价格
                idx = price_series.index.searchsorted(time)
                if idx < len(price_series):
                    price = price_series.iloc[idx]
                else:
                    continue
            
            total_cost += price * qty
            total_filled += qty
        
        avg_price = total_cost / total_filled if total_filled > 0 else 0
        return avg_price, total_filled
    
    def calc_slippage(self, exec_avg_price, arrival_price):
        """
        计算滑点
        到达价（Arrival Price）= 决定开始执行时的中间价
        """
        return (exec_avg_price - arrival_price) / arrival_price
```

---

## 3. VWAP（成交量加权平均价格）

**VWAP**：按历史成交量分布分配订单，在成交活跃时多执行，在成交清淡时少执行。

$$
\\text{VWAP} = \\frac{\\sum_t p_t \\cdot v_t}{\\sum_t v_t}
$$

```python
class VWAPExecution:
    """
    VWAP 算法执行器
    """
    
    def __init__(self, total_qty, hist_volume_profile=None):
        self.total_qty = total_qty
        # 历史成交量分布（按分钟，全天归一化）
        self.volume_profile = hist_volume_profile
        
        if hist_volume_profile is None:
            # 使用 A 股典型的 U 型成交量分布
            self.volume_profile = self._default_a_share_profile()
    
    def _default_a_share_profile(self):
        """
        A 股典型成交量分布（U 型）：
        - 早盘（9:30-10:00）成交量大
        - 午后（13:00-14:00）成交量小
        - 尾盘（14:30-15:00）成交量大
        """
        minutes = list(range(240))  # 240 分钟交易时间
        profile = np.ones(240)
        
        # 开盘放大
        profile[:30] *= 2.5
        # 午后缩量
        profile[120:180] *= 0.7
        # 尾盘放大
        profile[210:] *= 2.0
        
        # 归一化
        profile = profile / profile.sum()
        return profile
    
    def get_slice_schedule(self, trading_minutes=None):
        """
        按照历史成交量分布分配订单
        """
        if trading_minutes is None:
            trading_minutes = range(240)
        
        schedule = []
        for i, minute in enumerate(trading_minutes):
            qty = int(self.total_qty * self.volume_profile[i])
            if qty > 0:
                schedule.append({
                    'minute': minute,
                    'target_qty': qty,
                    'target_participation': self.volume_profile[i]
                })
        
        return pd.DataFrame(schedule)
    
    def adaptive_execution(self, actual_volume, target_time, remaining_qty):
        """
        自适应 VWAP：根据实时成交量调整剩余执行计划
        """
        # 剩余时间的预测成交量
        remaining_profile = self.volume_profile[target_time:]
        expected_remaining_vol = remaining_profile.sum()
        
        # 重新分配剩余订单
        participation_rate = remaining_qty / (expected_remaining_vol * actual_volume.mean())
        
        # 限制参与率不超过 20%（避免过大市场影响）
        participation_rate = min(participation_rate, 0.2)
        
        return participation_rate


def calc_vwap_benchmark(price_series, volume_series):
    """
    计算市场 VWAP（作为执行质量的基准）
    """
    return (price_series * volume_series).sum() / volume_series.sum()


def vwap_slippage(exec_prices, exec_quantities, market_vwap):
    """
    VWAP 滑点 = 实际成交均价 - 市场 VWAP
    负值表示比市场 VWAP 买入更便宜（好）
    正值表示比市场 VWAP 买入更贵（坏）
    """
    exec_vwap = (exec_prices * exec_quantities).sum() / exec_quantities.sum()
    return (exec_vwap - market_vwap) / market_vwap
```

---

## 4. TWAP vs. VWAP 选择指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| 市场成交量分布稳定 | VWAP | 更贴近市场节奏，减少冲击 |
| 成交量分布不稳定（重大事件）| TWAP | 不依赖历史分布假设 |
| 执行时间窗口短（< 1小时）| TWAP | 复杂度不值得，均匀分散即可 |
| 大额订单（>5% ADV）| IS（实施落差）| 需要优化冲击成本 |
| 对执行基准有硬性要求 | VWAP | 常见合规要求 |

---

## 5. 实施落差（Implementation Shortfall）

IS 算法的目标：最小化决策价格（信号发出时的价格）与最终执行价格之间的差距。

$$
IS = \\underbrace{\\text{Delay Cost}}_{\\text{决策到执行的延迟}} + \\underbrace{\\text{Market Impact}}_{\\text{执行中的价格变动}} + \\underbrace{\\text{Opportunity Cost}}_{\\text{未成交部分的机会成本}}
$$

---

## 小结

| 算法 | 目标 | 适用场景 |
|------|------|---------|
| TWAP | 均匀分散时间 | 简单稳健，不依赖预测 |
| VWAP | 跟踪市场节奏 | 成交量分布稳定 |
| IS | 最小化总成本 | 大额订单，有时间灵活性 |
| POV | 固定参与率 | 需要控制市场影响 |
