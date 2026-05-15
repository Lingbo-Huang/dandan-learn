---
layout: default
title: "D2 · 订单类型与订单簿"
render_with_liquid: false
---

# D2 · 订单类型与订单簿

> 真实的交易不是点一下"买入"就完成的。理解订单簿，是理解执行成本的基础。

---

## 1. 订单类型

| 订单类型 | 说明 | 特点 |
|---------|------|------|
| 市价单（Market Order）| 以当前最优价立即成交 | 确保成交，但价格不确定 |
| 限价单（Limit Order）| 指定价格，等待对手方 | 价格确定，可能不成交 |
| 止损单（Stop Order）| 达到触发价后变成市价单 | 风控常用 |
| 冰山单（Iceberg）| 分批显示数量，隐藏真实规模 | 减少信息泄露 |
| FAK（Fill and Kill）| 立即以限价成交，余量撤销 | 部分成交接受 |
| FOK（Fill or Kill）| 全部成交或全部撤销 | 大额确定性需求 |

---

## 2. 订单簿结构

```python
import pandas as pd
import numpy as np
from collections import defaultdict
import heapq

class OrderBook:
    """
    简化版订单簿模拟
    """
    
    def __init__(self, symbol):
        self.symbol = symbol
        # 买单（大根堆，按价格降序）
        self._bids = []  # (-price, order_id, qty)
        # 卖单（小根堆，按价格升序）
        self._asks = []  # (price, order_id, qty)
        
        self.orders = {}
        self.next_id = 1
        self.trades = []
    
    def add_bid(self, price, quantity):
        """添加买单"""
        order_id = self.next_id
        self.next_id += 1
        heapq.heappush(self._bids, (-price, order_id, quantity))
        self.orders[order_id] = {'side': 'bid', 'price': price, 'qty': quantity}
        self._match()
        return order_id
    
    def add_ask(self, price, quantity):
        """添加卖单"""
        order_id = self.next_id
        self.next_id += 1
        heapq.heappush(self._asks, (price, order_id, quantity))
        self.orders[order_id] = {'side': 'ask', 'price': price, 'qty': quantity}
        self._match()
        return order_id
    
    def _match(self):
        """撮合：最高买价 >= 最低卖价时成交"""
        while self._bids and self._asks:
            neg_bid_price, bid_id, bid_qty = self._bids[0]
            ask_price, ask_id, ask_qty = self._asks[0]
            
            bid_price = -neg_bid_price
            
            if bid_price >= ask_price:
                # 成交
                trade_qty = min(bid_qty, ask_qty)
                trade_price = ask_price  # 以挂单价成交（卖方先挂）
                
                self.trades.append({
                    'price': trade_price,
                    'quantity': trade_qty,
                    'buyer_order': bid_id,
                    'seller_order': ask_id
                })
                
                # 更新订单簿
                heapq.heappop(self._bids)
                heapq.heappop(self._asks)
                
                if bid_qty > ask_qty:
                    heapq.heappush(self._bids, (-bid_price, bid_id, bid_qty - ask_qty))
                elif ask_qty > bid_qty:
                    heapq.heappush(self._asks, (ask_price, ask_id, ask_qty - bid_qty))
            else:
                break
    
    def get_best_bid(self):
        if self._bids:
            return -self._bids[0][0]
        return None
    
    def get_best_ask(self):
        if self._asks:
            return self._asks[0][0]
        return None
    
    def get_spread(self):
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return ask - bid
        return None
    
    def get_mid_price(self):
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return (bid + ask) / 2
        return None
    
    def get_depth(self, levels=5):
        """获取订单簿深度（前 N 档）"""
        bids = sorted([(-p, q) for p, _, q in self._bids], reverse=True)[:levels]
        asks = sorted([(p, q) for p, _, q in self._asks])[:levels]
        return {
            'bids': [(p, q) for p, q in bids],
            'asks': asks
        }
```

---

## 3. 订单流分析

```python
def analyze_order_flow(trades_df):
    """
    订单流失衡（Order Flow Imbalance）分析
    
    trades_df columns: price, quantity, side (buy/sell)
    """
    # 计算买卖失衡
    buy_vol = trades_df[trades_df['side'] == 'buy']['quantity'].sum()
    sell_vol = trades_df[trades_df['side'] == 'sell']['quantity'].sum()
    
    ofi = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8)
    
    # 滚动订单流失衡（OFI）
    trades_df['signed_qty'] = trades_df.apply(
        lambda r: r['quantity'] if r['side'] == 'buy' else -r['quantity'], axis=1
    )
    rolling_ofi = trades_df['signed_qty'].rolling('5min').sum() / \
                  trades_df['quantity'].rolling('5min').sum()
    
    return ofi, rolling_ofi


def order_flow_toxicity(trades_df):
    """
    PIN（知情交易概率）简化估算
    高 PIN = 订单流毒性高 = 逆选择风险大
    """
    buy_vol = trades_df[trades_df['side'] == 'buy']['quantity']
    sell_vol = trades_df[trades_df['side'] == 'sell']['quantity']
    
    # 简化 VPIN（成交量同步 PIN）
    n_buckets = 50
    total_vol = trades_df['quantity'].sum()
    bucket_size = total_vol / n_buckets
    
    vpin_values = []
    current_bucket_buy = 0
    current_bucket_vol = 0
    
    for _, trade in trades_df.iterrows():
        current_bucket_vol += trade['quantity']
        if trade['side'] == 'buy':
            current_bucket_buy += trade['quantity']
        
        if current_bucket_vol >= bucket_size:
            vpin_values.append(abs(2 * current_bucket_buy - current_bucket_vol) / 
                               current_bucket_vol)
            current_bucket_buy = 0
            current_bucket_vol = 0
    
    return np.mean(vpin_values) if vpin_values else 0.5
```

---

## 4. A股 Level 2 数据解读

A股 Level 2 数据包含：
- **逐笔成交**：每一笔成交的价格、数量、买卖方向
- **逐笔委托**：每一笔报单的价格、数量、委托类型
- **快照行情**：10档买卖盘（最佳5档 + 次5档）

```python
def parse_level2_snapshot(snapshot):
    """
    解析 Level 2 快照数据（10档盘口）
    """
    bid_prices = [snapshot[f'bid{i}'] for i in range(1, 6)]
    bid_volumes = [snapshot[f'bid_vol{i}'] for i in range(1, 6)]
    ask_prices = [snapshot[f'ask{i}'] for i in range(1, 6)]
    ask_volumes = [snapshot[f'ask_vol{i}'] for i in range(1, 6)]
    
    # 订单簿不平衡（OBI）：买方力量 vs 卖方力量
    total_bid = sum(bid_volumes)
    total_ask = sum(ask_volumes)
    obi = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)
    
    # 买卖价差
    spread = ask_prices[0] - bid_prices[0]
    
    return {
        'bid1': bid_prices[0], 'ask1': ask_prices[0],
        'spread': spread,
        'obi': obi,
        'depth_bid': total_bid,
        'depth_ask': total_ask
    }
```

---

## 小结

| 概念 | 说明 |
|------|------|
| 市价单 | 立即成交，价格不确定 |
| 限价单 | 指定价格，可能不成交 |
| 订单簿 | 买卖双方报价的实时汇总 |
| 订单流 | 买卖力量的动态平衡信号 |
| Level 2 | 深度盘口 + 逐笔数据 |
