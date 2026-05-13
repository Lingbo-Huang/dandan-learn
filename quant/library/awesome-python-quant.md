---
layout: default
title: "AwesomeQuant · Python 量化开发实战"
source: "https://github.com/0voice/Awesome-QuantDev-Learn"
---

# Python 量化开发实战精选

> **来源**：[Awesome-QuantDev-Learn](https://github.com/0voice/Awesome-QuantDev-Learn) · 0voice

---

## 一、使用 TuShare 获取 A 股数据并分析

```python
import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 初始化
ts.set_token('your_token')
pro = ts.pro_api()

def fetch_and_analyze_stock(ts_code: str = '000001.SZ',
                             start_date: str = '20220101',
                             end_date: str = '20241231'):
    """获取股票数据并做基础分析"""
    
    # 获取日线数据
    df = pro.daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields='trade_date,open,high,low,close,vol,amount,pct_chg'
    )
    
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.sort_values('trade_date', inplace=True)
    df.set_index('trade_date', inplace=True)
    
    # 基础计算
    df['daily_return'] = df['close'].pct_change()
    df['cumulative_return'] = (1 + df['daily_return']).cumprod()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['volatility_20d'] = df['daily_return'].rolling(20).std() * np.sqrt(252)
    
    # 统计摘要
    stats = {
        '总收益率': f"{(df['cumulative_return'].iloc[-1] - 1):.2%}",
        '年化收益率': f"{df['daily_return'].mean() * 252:.2%}",
        '年化波动率': f"{df['daily_return'].std() * np.sqrt(252):.2%}",
        '夏普比率': f"{df['daily_return'].mean() / df['daily_return'].std() * np.sqrt(252):.2f}",
        '最大单日涨幅': f"{df['pct_chg'].max():.2f}%",
        '最大单日跌幅': f"{df['pct_chg'].min():.2f}%",
    }
    
    print(f"\n===== {ts_code} 行情分析 =====")
    for k, v in stats.items():
        print(f"{k}: {v}")
    
    # 可视化
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 价格与均线
    axes[0].plot(df.index, df['close'], label='收盘价', linewidth=1)
    axes[0].plot(df.index, df['ma5'], label='MA5', alpha=0.7, linewidth=0.8)
    axes[0].plot(df.index, df['ma20'], label='MA20', alpha=0.7, linewidth=0.8)
    axes[0].plot(df.index, df['ma60'], label='MA60', alpha=0.7, linewidth=0.8)
    axes[0].legend(loc='upper left')
    axes[0].set_title(f'{ts_code} 价格走势')
    axes[0].set_ylabel('价格（元）')
    
    # 成交量
    axes[1].bar(df.index, df['vol'] / 1e8, color='steelblue', alpha=0.7)
    axes[1].set_ylabel('成交量（亿手）')
    axes[1].set_title('成交量')
    
    # 日收益率
    axes[2].bar(df.index, df['daily_return'] * 100, 
                color=df['daily_return'].apply(lambda x: 'red' if x >= 0 else 'green'),
                alpha=0.7)
    axes[2].axhline(0, color='black', linewidth=0.5)
    axes[2].set_ylabel('日涨跌幅（%）')
    axes[2].set_title('日收益率')
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    plt.savefig(f'{ts_code}_analysis.png', dpi=150)
    plt.show()
    
    return df
```

---

## 二、技术指标计算与可视化

```python
import pandas as pd
import numpy as np

class TechnicalIndicators:
    """常用技术指标计算"""
    
    @staticmethod
    def macd(close: pd.Series, 
              fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD 指标"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        macd_hist = (dif - dea) * 2
        return pd.DataFrame({'DIF': dif, 'DEA': dea, 'MACD': macd_hist})
    
    @staticmethod
    def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2):
        """布林带"""
        mid = close.rolling(window).mean()
        std = close.rolling(window).std()
        return pd.DataFrame({
            'upper': mid + n_std * std,
            'mid': mid,
            'lower': mid - n_std * std,
            'bandwidth': (mid + n_std*std - (mid - n_std*std)) / mid,
            'percent_b': (close - (mid - n_std*std)) / (2*n_std*std)
        })
    
    @staticmethod
    def rsi(close: pd.Series, window: int = 14) -> pd.Series:
        """RSI"""
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=window-1, min_periods=window).mean()
        loss = (-delta.clip(upper=0)).ewm(com=window-1, min_periods=window).mean()
        return 100 - (100 / (1 + gain/(loss+1e-8)))
    
    @staticmethod
    def kdj(high: pd.Series, low: pd.Series, close: pd.Series, 
             n: int = 9, m1: int = 3, m2: int = 3):
        """KDJ 指标"""
        low_n = low.rolling(n).min()
        high_n = high.rolling(n).max()
        rsv = (close - low_n) / (high_n - low_n + 1e-8) * 100
        K = rsv.ewm(com=m1-1, adjust=False).mean()
        D = K.ewm(com=m2-1, adjust=False).mean()
        J = 3*K - 2*D
        return pd.DataFrame({'K': K, 'D': D, 'J': J})
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
             window: int = 14) -> pd.Series:
        """ATR（平均真实波幅）"""
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(com=window-1, min_periods=window).mean()
    
    @staticmethod
    def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
        """量价趋势（VPT）"""
        return (close.pct_change() * volume).cumsum()

# 使用示例
def run_technical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    ti = TechnicalIndicators()
    
    macd_df = ti.macd(df['close'])
    bb_df = ti.bollinger_bands(df['close'])
    
    df['rsi_14'] = ti.rsi(df['close'])
    df = pd.concat([df, macd_df.add_prefix('macd_'), bb_df.add_prefix('bb_')], axis=1)
    
    kdj_df = ti.kdj(df['high'], df['low'], df['close'])
    df = pd.concat([df, kdj_df.add_prefix('kdj_')], axis=1)
    
    df['atr_14'] = ti.atr(df['high'], df['low'], df['close'])
    
    return df
```

---

## 三、使用 Backtrader 实现均线交叉策略

```python
import backtrader as bt
import pandas as pd
from datetime import datetime

class MAcrossStrategy(bt.Strategy):
    """双均线交叉策略"""
    
    params = dict(
        fast=5,
        slow=20,
        stake=100,      # 每次买入股数
        printlog=False,
    )
    
    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        
        # 均线指标
        self.ma_fast = bt.ind.SMA(self.data.close, period=self.p.fast)
        self.ma_slow = bt.ind.SMA(self.data.close, period=self.p.slow)
        self.crossover = bt.ind.CrossOver(self.ma_fast, self.ma_slow)
        
        # 记录每笔交易
        self.trades = []
    
    def next(self):
        if self.order:
            return
        
        if not self.position:
            if self.crossover > 0:  # 金叉
                self.order = self.buy(size=self.p.stake)
                self.log(f'金叉买入, 价格: {self.dataclose[0]:.2f}')
        else:
            if self.crossover < 0:  # 死叉
                self.order = self.sell(size=self.p.stake)
                self.log(f'死叉卖出, 价格: {self.dataclose[0]:.2f}')
    
    def notify_order(self, order):
        if order.status == order.Completed:
            action = '买入' if order.isbuy() else '卖出'
            self.log(f'{action}执行: 价格={order.executed.price:.2f}, '
                     f'数量={order.executed.size}, 成本={order.executed.value:.2f}')
        self.order = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'交易结束: 毛利润={trade.pnl:.2f}, 净利润={trade.pnlcomm:.2f}')

def run_backtest_example(csv_file: str):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MAcrossStrategy, fast=5, slow=20, printlog=True)
    
    data = bt.feeds.GenericCSVData(
        dataname=csv_file,
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        dtformat='%Y%m%d'
    )
    cerebro.adddata(data)
    
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)
    
    # 分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.02, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
    
    print(f'初始资金: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    strat = results[0]
    print(f'最终资金: {cerebro.broker.getvalue():.2f}')
    
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    max_dd = strat.analyzers.dd.get_analysis().get('max', {}).get('drawdown', None)
    
    print(f'夏普比率: {sharpe:.3f}' if sharpe else '夏普: N/A')
    print(f'最大回撤: {max_dd:.2f}%' if max_dd else '最大回撤: N/A')
    
    cerebro.plot(style='candlestick')
```

---

## 四、风控与仓位管理模块

```python
import numpy as np
import pandas as pd

class RiskManager:
    """量化策略风控模块"""
    
    def __init__(self,
                 max_position_pct: float = 0.3,   # 单一仓位上限
                 max_drawdown_limit: float = 0.20,  # 最大回撤熔断
                 daily_loss_limit: float = 0.05,    # 日损失熔断
                 portfolio_vol_target: float = 0.15  # 目标组合波动率
                 ):
        self.max_position_pct = max_position_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_loss_limit = daily_loss_limit
        self.portfolio_vol_target = portfolio_vol_target
        self.is_halted = False
    
    def check_position_limit(self, weight: float, stock: str) -> float:
        """检查并限制单一仓位"""
        if weight > self.max_position_pct:
            print(f"⚠️ {stock} 仓位 {weight:.2%} 超出上限 {self.max_position_pct:.2%}，已截断")
            return self.max_position_pct
        return weight
    
    def check_drawdown_halt(self, current_nav: float, peak_nav: float) -> bool:
        """最大回撤熔断检查"""
        drawdown = (current_nav - peak_nav) / peak_nav
        if drawdown < -self.max_drawdown_limit:
            self.is_halted = True
            print(f"🚨 熔断！当前回撤 {drawdown:.2%} 超过上限 {-self.max_drawdown_limit:.2%}，停止交易")
        return self.is_halted
    
    def kelly_position_size(self, win_rate: float, win_loss_ratio: float,
                             kelly_fraction: float = 0.5) -> float:
        """
        Kelly公式仓位（半Kelly更保守）
        win_rate: 胜率
        win_loss_ratio: 盈亏比（平均盈利/平均亏损）
        """
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        return max(0, kelly * kelly_fraction)  # 半Kelly
    
    def vol_target_sizing(self, signal_weight: float,
                           asset_vol: float) -> float:
        """
        波动率目标仓位
        目的：控制每个资产对组合波动率的贡献
        """
        if asset_vol <= 0:
            return 0
        vol_scalar = self.portfolio_vol_target / asset_vol
        return signal_weight * vol_scalar
    
    def portfolio_var(self, weights: np.ndarray,
                       cov_matrix: np.ndarray,
                       confidence: float = 0.95) -> float:
        """
        参数法 VaR（Value at Risk）
        """
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        daily_var = -z_score * portfolio_vol
        annual_var = daily_var * np.sqrt(252)
        return {'daily_var': daily_var, 'annual_var': annual_var}
```

---

## 延伸阅读

- [Awesome-QuantDev-Learn](https://github.com/0voice/Awesome-QuantDev-Learn)
- [AkShare 文档](https://akshare.akfamily.xyz/)
- [Backtrader 文档](https://www.backtrader.com/)
- Narang - "Inside the Black Box"
