---
layout: default
title: "WhaleQuant Ch03 · 股票数据获取"
source: "https://github.com/datawhalechina/whale-quant"
---

# 03 · 股票数据获取

> **来源**：[WhaleQuant](https://github.com/datawhalechina/whale-quant) · datawhalechina

---

## 主要数据源对比

| 数据源 | 免费额度 | 数据类型 | 适合场景 |
|--------|---------|---------|---------|
| **AkShare** | 完全免费 | 行情/财务/宏观 | 入门首选 |
| **Tushare** | 积分制 | A股全面 | 主流选择 |
| **yfinance** | 完全免费 | 全球市场 | 海外数据 |
| **baostock** | 免费 | A股历史行情 | 轻量使用 |
| **Wind** | 付费 | 机构级全品类 | 专业机构 |

---

## 一、AkShare 数据获取

```python
import akshare as ak
import pandas as pd

# 安装：pip install akshare

# 1. 获取A股历史日线数据
def get_stock_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    symbol: 股票代码，如 '000001'（平安银行）
    start/end: 日期格式 'YYYYMMDD'
    """
    df = ak.stock_zh_a_hist(
        symbol=symbol, 
        period="daily",
        start_date=start, 
        end_date=end,
        adjust="qfq"   # 前复权
    )
    df.columns = ['date', 'open', 'close', 'high', 'low', 
                  'volume', 'amount', 'amplitude', 'pct_change', 
                  'change', 'turnover']
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# 获取平安银行5年数据
df = get_stock_history('000001', '20200101', '20241231')
print(df.head())
print(f"数据量: {len(df)} 条")

# 2. 实时行情
realtime = ak.stock_zh_a_spot_em()
print(realtime.columns.tolist())

# 3. 指数数据
hs300 = ak.stock_zh_index_daily(symbol="sh000300")
print(hs300.tail())

# 4. 财务数据
profit = ak.stock_financial_report_sina(stock="sh600519", symbol="利润表")
print(profit.head())
```

---

## 二、Tushare 数据获取

```python
import tushare as ts
import pandas as pd

# 需注册获取 token：https://tushare.pro
ts.set_token('your_token_here')
pro = ts.pro_api()

# 1. 日线行情
def get_daily(ts_code: str, start: str, end: str):
    """ts_code: '000001.SZ', '600519.SH'"""
    df = pro.daily(
        ts_code=ts_code, 
        start_date=start, 
        end_date=end,
        fields='trade_date,open,high,low,close,vol,amount,pct_chg'
    )
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.sort_values('trade_date', inplace=True)
    df.set_index('trade_date', inplace=True)
    return df

# 2. 股票基础信息
stock_basic = pro.stock_basic(
    exchange='', 
    list_status='L',
    fields='ts_code,name,area,industry,list_date'
)
print(f"A股上市公司数量: {len(stock_basic)}")

# 3. 财务指标（常用因子来源）
fina = pro.fina_indicator(
    ts_code='000001.SZ', 
    start_date='20200101',
    fields='ts_code,ann_date,roe,roa,gross_margin,current_ratio,debt_to_assets'
)
print(fina.head())

# 4. 批量获取因子数据
def get_factor_data(ts_codes: list, factor: str, start: str, end: str):
    """批量获取因子数据，注意 API 频率限制"""
    results = []
    for code in ts_codes:
        df = pro.daily_basic(ts_code=code, start_date=start, end_date=end,
                             fields=f'ts_code,trade_date,{factor}')
        results.append(df)
        time.sleep(0.1)  # 避免触发频率限制
    return pd.concat(results)
```

---

## 三、数据清洗

```python
import pandas as pd
import numpy as np

def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """股价数据清洗标准流程"""
    
    # 1. 删除重复行
    df = df.drop_duplicates()
    
    # 2. 处理缺失值
    # 日期缺失（非交易日）：正常现象，不需要填充
    # 价格缺失：可能是停牌，向前填充
    price_cols = ['open', 'high', 'low', 'close']
    df[price_cols] = df[price_cols].fillna(method='ffill')
    
    # 3. 去除明显异常值（价格 <= 0）
    df = df[df['close'] > 0]
    
    # 4. 成交量为0（停牌日）处理
    df['is_suspend'] = df['volume'] == 0
    
    # 5. 计算收益率
    df['return'] = df['close'].pct_change()
    
    # 6. 去除极端值（3σ法则）
    returns = df['return'].dropna()
    mean, std = returns.mean(), returns.std()
    df['return_clean'] = df['return'].clip(mean - 3*std, mean + 3*std)
    
    return df

# 数据对齐（多股票面板数据）
def build_panel(stock_data: dict) -> pd.DataFrame:
    """
    stock_data: {'000001': df1, '000002': df2, ...}
    返回宽表格式的收益率矩阵
    """
    close_prices = {}
    for code, df in stock_data.items():
        close_prices[code] = df['close']
    
    panel = pd.DataFrame(close_prices)
    
    # 对齐日期（取交集）
    panel = panel.dropna(how='all')
    panel = panel.fillna(method='ffill').fillna(method='bfill')
    
    # 计算收益率矩阵
    returns = panel.pct_change()
    
    return panel, returns
```

---

## 四、数据存储

```python
import sqlite3
import pandas as pd

# SQLite 本地存储（小数据集）
class LocalDataStore:
    def __init__(self, db_path='quant_data.db'):
        self.conn = sqlite3.connect(db_path)
    
    def save_daily(self, df: pd.DataFrame, table='daily_prices'):
        df.to_sql(table, self.conn, if_exists='append', index=True)
    
    def load_daily(self, symbol: str, start: str, end: str):
        query = f"""
        SELECT * FROM daily_prices 
        WHERE symbol='{symbol}' 
        AND date >= '{start}' AND date <= '{end}'
        ORDER BY date
        """
        return pd.read_sql(query, self.conn, index_col='date', parse_dates=['date'])
    
    def close(self):
        self.conn.close()

# HDF5 存储（大数据集，读写更快）
def save_to_hdf5(df: pd.DataFrame, path: str, key: str):
    df.to_hdf(path, key=key, mode='a', complevel=9)

def load_from_hdf5(path: str, key: str):
    return pd.read_hdf(path, key=key)
```

---

## 延伸阅读

- [AkShare 文档](https://akshare.akfamily.xyz/)
- [Tushare Pro 文档](https://tushare.pro/document/2)
- [WhaleQuant 完整教程](https://github.com/datawhalechina/whale-quant)
