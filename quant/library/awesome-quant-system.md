---
layout: default
title: "AwesomeQuant · 量化交易系统核心技术"
source: "https://github.com/0voice/Awesome-QuantDev-Learn"
---

# 量化交易系统核心技术栈

> **来源**：[Awesome-QuantDev-Learn](https://github.com/0voice/Awesome-QuantDev-Learn) · 0voice

---

## 一、量化交易系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     量化交易系统架构                        │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│  数据层   │  研究层   │  策略层   │  执行层   │   监控层    │
├──────────┼──────────┼──────────┼──────────┼─────────────┤
│行情数据   │因子研究   │信号生成   │订单管理   │实时监控     │
│财务数据   │回测引擎   │仓位管理   │算法执行   │风险报警     │
│另类数据   │机器学习   │风险控制   │券商接口   │绩效归因     │
│数据清洗   │策略评估   │再平衡     │成交确认   │日志审计     │
└──────────┴──────────┴──────────┴──────────┴─────────────┘
```

---

## 二、高性能数据处理

```python
import pandas as pd
import numpy as np
from numba import jit, prange
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time

# 1. Numba JIT 加速计算密集型因子
@jit(nopython=True, parallel=True)
def compute_rolling_sharpe_fast(returns: np.ndarray, 
                                  window: int = 20,
                                  rf: float = 0.0) -> np.ndarray:
    """JIT 编译的滚动夏普比率（比 pandas 快 10-50x）"""
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in prange(window, n):
        window_returns = returns[i-window:i]
        mean = np.mean(window_returns) - rf / 252
        std = np.std(window_returns)
        if std > 0:
            result[i] = mean / std * np.sqrt(252)
    
    return result

# 性能对比
def benchmark_rolling_sharpe():
    returns = np.random.randn(10000) * 0.01
    
    # Pandas 版本
    t0 = time.time()
    s = pd.Series(returns)
    pandas_result = (s.rolling(20).mean() / s.rolling(20).std() * np.sqrt(252))
    t_pandas = time.time() - t0
    
    # Numba 版本（第一次含编译时间）
    _ = compute_rolling_sharpe_fast(returns[:100], 20)  # 预热
    t0 = time.time()
    numba_result = compute_rolling_sharpe_fast(returns, 20)
    t_numba = time.time() - t0
    
    print(f"Pandas: {t_pandas*1000:.2f}ms")
    print(f"Numba:  {t_numba*1000:.2f}ms")
    print(f"加速比: {t_pandas/t_numba:.1f}x")

# 2. Parquet 高效数据存储
class MarketDataStore:
    """基于 Parquet 的市场数据存储"""
    
    def __init__(self, base_dir: str = './market_data'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_daily_data(self, df: pd.DataFrame, date: str):
        """按日期分区存储"""
        year = date[:4]
        month = date[4:6]
        path = self.base_dir / f'year={year}' / f'month={month}' / f'{date}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为 Arrow 格式（更快）
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path, compression='snappy')
    
    def load_date_range(self, start: str, end: str,
                         columns: list = None) -> pd.DataFrame:
        """高效加载日期范围数据"""
        dataset = pq.ParquetDataset(
            self.base_dir,
            filters=[
                ('date', '>=', start),
                ('date', '<=', end)
            ]
        )
        table = dataset.read(columns=columns)
        return table.to_pandas()
    
    def get_cross_section(self, date: str, 
                           factors: list = None) -> pd.DataFrame:
        """获取截面数据（当日所有股票）"""
        path = self._date_to_path(date)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path, columns=factors)
    
    def _date_to_path(self, date: str) -> Path:
        year, month = date[:4], date[4:6]
        return self.base_dir / f'year={year}' / f'month={month}' / f'{date}.parquet'
```

---

## 三、实时行情处理

```python
import asyncio
import websockets
import json
from collections import deque
import threading

class RealtimeDataFeed:
    """实时行情接收与处理"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer = deque(maxlen=buffer_size)
        self.tick_handlers = []
        self.bar_handlers = []
        self._running = False
        
        # 分钟 K 线构建
        self.current_bars = {}
    
    def subscribe(self, handler_type: str):
        """装饰器：订阅行情回调"""
        def decorator(func):
            if handler_type == 'tick':
                self.tick_handlers.append(func)
            elif handler_type == 'bar':
                self.bar_handlers.append(func)
            return func
        return decorator
    
    async def connect_websocket(self, uri: str):
        """WebSocket 连接行情源"""
        async with websockets.connect(uri) as ws:
            self._running = True
            print(f"已连接行情源: {uri}")
            
            while self._running:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    tick = json.loads(msg)
                    
                    # 存入缓冲区
                    self.buffer.append(tick)
                    
                    # 触发回调
                    for handler in self.tick_handlers:
                        await handler(tick)
                    
                    # 构建分钟 K 线
                    self._update_bar(tick)
                    
                except asyncio.TimeoutError:
                    await ws.ping()  # 保活
                except Exception as e:
                    print(f"行情接收错误: {e}")
                    break
    
    def _update_bar(self, tick: dict):
        """从 Tick 构建分钟 K 线"""
        symbol = tick.get('symbol')
        price = tick.get('price', 0)
        volume = tick.get('volume', 0)
        timestamp = tick.get('timestamp', '')
        
        # 取当前分钟
        minute = timestamp[:16]  # 'YYYY-MM-DD HH:MM'
        key = f"{symbol}_{minute}"
        
        if key not in self.current_bars:
            self.current_bars[key] = {
                'symbol': symbol, 'minute': minute,
                'open': price, 'high': price, 'low': price, 'close': price,
                'volume': 0
            }
        
        bar = self.current_bars[key]
        bar['high'] = max(bar['high'], price)
        bar['low'] = min(bar['low'], price)
        bar['close'] = price
        bar['volume'] += volume
```

---

## 四、订单管理系统（OMS）

```python
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    TWAP = "TWAP"
    VWAP = "VWAP"

@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    strategy_id: str = ""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)
    
    @property
    def remaining_qty(self) -> int:
        return self.quantity - self.filled_qty

class OrderManagementSystem:
    """订单管理系统"""
    
    def __init__(self):
        self.orders: dict[str, Order] = {}
        self.pending_orders: list[str] = []
    
    def submit_order(self, order: Order) -> str:
        """提交订单"""
        self.orders[order.order_id] = order
        self.pending_orders.append(order.order_id)
        order.status = OrderStatus.SUBMITTED
        print(f"[OMS] 提交订单: {order.order_id} | {order.side.value} {order.symbol} ×{order.quantity}")
        return order.order_id
    
    def on_fill(self, order_id: str, filled_qty: int, fill_price: float):
        """处理成交回报"""
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        
        # 更新均价
        total_value = order.avg_fill_price * order.filled_qty + fill_price * filled_qty
        order.filled_qty += filled_qty
        order.avg_fill_price = total_value / order.filled_qty if order.filled_qty > 0 else 0
        
        if order.filled_qty >= order.quantity:
            order.status = OrderStatus.FILLED
            self.pending_orders.remove(order_id)
        else:
            order.status = OrderStatus.PARTIAL
        
        print(f"[OMS] 成交回报: {order_id} | 成交{filled_qty}股 @ {fill_price:.2f} | "
              f"累计{order.filled_qty}/{order.quantity}")
    
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        if order_id not in self.orders:
            return False
        order = self.orders[order_id]
        if not order.is_complete:
            order.status = OrderStatus.CANCELLED
            if order_id in self.pending_orders:
                self.pending_orders.remove(order_id)
            print(f"[OMS] 撤单: {order_id}")
            return True
        return False
```

---

## 延伸阅读

- [Awesome-QuantDev-Learn](https://github.com/0voice/Awesome-QuantDev-Learn)
- [vnpy 开源交易框架](https://github.com/vnpy/vnpy)
- [Numba 文档](https://numba.readthedocs.io/)
- Harris - "Trading and Exchanges: Market Microstructure for Practitioners"
