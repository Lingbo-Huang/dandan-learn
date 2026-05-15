---
layout: default
title: "D7 · Week 7 综合实战：执行优化框架"
render_with_liquid: false
---

# D7 · Week 7 综合实战：执行优化框架

> 把市场微结构知识转化为实际的执行质量提升方案。

---

## 本周核心知识回顾

| 主题 | 要点 |
|------|------|
| 市场微结构 | Spread = 逆选择 + 库存 + 处理成本 |
| 订单簿 | 订单流失衡是短期价格预测信号 |
| TWAP/VWAP | 大单拆分，跟踪时间/量分布 |
| 冲击成本 | 平方根模型，ADV 参与率决定成本 |
| 做市策略 | 库存调整报价，逆选择防范 |

---

## 完整执行优化框架

```python
import pandas as pd
import numpy as np

class ExecutionOptimizer:
    """
    执行优化框架
    结合市场微结构信号，动态调整执行节奏
    """
    
    def __init__(self, algo='smart_twap', max_participation=0.15):
        self.algo = algo
        self.max_participation = max_participation
        self.execution_log = []
    
    def estimate_cost(self, quantity, adv, daily_vol, eta=0.5):
        """估算执行成本"""
        participation = min(quantity / adv, 1.0)
        impact = daily_vol * eta * np.sqrt(participation)
        return impact
    
    def smart_slice(self, total_qty, price_series, volume_series, 
                    ofi_series=None):
        """
        智能切片：结合 OFI 信号调整执行节奏
        - OFI > 0（买方强）-> 做多方向稍微加速执行
        - OFI < 0（卖方强）-> 做多方向放缓执行
        """
        n = len(price_series)
        base_qty_per_period = total_qty / n
        
        schedule = []
        
        for i in range(n):
            qty = base_qty_per_period
            
            if ofi_series is not None and i < len(ofi_series):
                ofi = ofi_series.iloc[i]
                # 根据 OFI 调整执行量（± 30%）
                qty *= (1 + 0.3 * np.sign(ofi) * min(abs(ofi), 1))
            
            # 确保不超过参与率限制
            max_qty = volume_series.iloc[i] * self.max_participation
            qty = min(qty, max_qty)
            
            schedule.append({
                'period': i,
                'target_qty': int(qty),
                'price': price_series.iloc[i],
                'adv_participation': qty / (volume_series.iloc[i] + 1e-8)
            })
        
        return pd.DataFrame(schedule)
    
    def evaluate_execution(self, schedule, arrival_price):
        """
        评估执行质量
        """
        total_qty = schedule['target_qty'].sum()
        total_cost = (schedule['target_qty'] * schedule['price']).sum()
        avg_price = total_cost / total_qty
        
        # 各类成本分解
        vwap = (schedule['target_qty'] * schedule['price']).sum() / total_qty
        mkt_vwap = (schedule['volume'] * schedule['price']).sum() / schedule['volume'].sum() \
                   if 'volume' in schedule.columns else avg_price
        
        return {
            '到达价': arrival_price,
            '成交均价': avg_price,
            '市场VWAP': mkt_vwap,
            '实施落差(bps)': (avg_price - arrival_price) / arrival_price * 10000,
            'VWAP滑点(bps)': (avg_price - mkt_vwap) / mkt_vwap * 10000,
            '总成交量': total_qty,
            '平均参与率': schedule['adv_participation'].mean()
        }


# ===== 实战：不同策略的执行成本比较 =====

def compare_execution_strategies(price_data, volume_data, total_qty=100000):
    """
    比较 TWAP / VWAP / 最优 IS 三种执行策略的成本
    """
    arrival_price = price_data.iloc[0]
    results = {}
    
    # 1. TWAP：均匀切分
    n = len(price_data)
    twap_qty = total_qty / n
    twap_avg = (price_data * twap_qty).sum() / total_qty
    results['TWAP'] = {
        '成交均价': twap_avg,
        '成本(bps)': (twap_avg - arrival_price) / arrival_price * 10000
    }
    
    # 2. VWAP：按成交量分布
    vol_weights = volume_data / volume_data.sum()
    vwap_qtys = total_qty * vol_weights
    vwap_avg = (price_data * vwap_qtys).sum() / total_qty
    results['VWAP'] = {
        '成交均价': vwap_avg,
        '成本(bps)': (vwap_avg - arrival_price) / arrival_price * 10000
    }
    
    # 3. 前30分钟快速执行（激进策略，减少价格漂移风险）
    fast_n = min(6, n)  # 只用前6个时间段
    fast_qty = total_qty / fast_n
    fast_avg = price_data.iloc[:fast_n].mean()
    results['激进执行'] = {
        '成交均价': fast_avg,
        '成本(bps)': (fast_avg - arrival_price) / arrival_price * 10000
    }
    
    return pd.DataFrame(results).T
```

---

## 执行质量评估体系

| 指标 | 公式 | 含义 |
|------|------|------|
| 实施落差 (IS) | (均价 - 到达价) / 到达价 | 相对决策时刻的总成本 |
| VWAP 滑点 | (均价 - 市场VWAP) / VWAP | 相对市场的执行质量 |
| 参与率 | 成交量 / ADV | 对市场的影响程度 |
| 完成率 | 实际成交 / 目标数量 | 执行完整性 |

---

## 面试核心问题

**Q：TWAP 和 VWAP 各自的适用场景？**

TWAP：成交量分布不稳定时（重大公告、市场异常），均匀切分最保险。
VWAP：正常市场环境，跟随成交量节奏更自然，减少市场影响。

**Q：冲击成本为什么与数量的平方根成正比而不是线性？**

直觉：第一笔订单消化当前最优报价，随后要"挖掘"更深的订单簿，边际成本递增。数学上，Order Book 深度近似呈指数分布，积分后得到平方根关系。

**Q：做市商如何对抗知情交易者？**

- 扩大价差（提高逆选择成本补偿）
- 库存调整报价（偏离中间价，引导库存回归）
- 检测毒性订单流（VPIN 等指标），降低报价深度
- 拒绝部分大额订单

---

## 小结

执行优化的核心是在 **冲击成本** 和 **机会成本（价格漂移）** 之间做权衡：
- 执行太慢 → 价格漂移损失大
- 执行太快 → 冲击成本大
- 最优执行 = 根据市场深度、波动率、剩余时间动态调整节奏
