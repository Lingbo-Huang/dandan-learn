# Day 6 · 多市场数据获取实战

> **Week 1 主题：金融市场概览**
> 预计学习时间：4~5 小时（偏实战，多写代码）

---

## 🎯 学习目标

1. 建立统一的多市场数据获取管道（A股/港股/美股/期货/加密）
2. 掌握数据清洗和标准化的基本方法
3. 将各市场数据保存为结构一致的 CSV 文件
4. 初步感受不同市场数据的"质感"差异

---

## 📚 核心知识点

### 1. 量化数据体系概览

```
行情数据（Price Data）
  ├── 日线 OHLCV
  ├── 分钟线 / Tick
  └── 期权链

基本面数据（Fundamental Data）
  ├── 财务报表（利润表/资产负债表/现金流）
  ├── 估值指标（PE/PB/PS）
  └── 分红、拆股事件

另类数据（Alternative Data）
  ├── 舆情/新闻情感
  ├── 卫星图像
  └── 信用卡消费数据
```

> Week 1 专注于**行情数据（日线 OHLCV）**的获取

### 2. 各市场数据工具对比

| 市场 | 推荐工具 | 优点 | 注意事项 |
|------|---------|------|---------|
| A股 | akshare | 免费、全面、更新及时 | 接口可能偶发失效 |
| 港股 | akshare | 同上 | 数据深度相对较浅 |
| 美股 | yfinance | 稳定、Yahoo源 | 数据质量整体较高 |
| 期货 | akshare | 涵盖国内主要品种 | 主力合约换月需注意 |
| 加密 | ccxt | 多交易所统一接口 | 注意频率限制 |

### 3. 标准数据格式设计

统一使用以下 DataFrame 结构：

```python
# 标准列名（所有市场统一）
columns = [
    "date",     # 日期 (YYYY-MM-DD)
    "open",     # 开盘价
    "high",     # 最高价
    "low",      # 最低价
    "close",    # 收盘价
    "volume",   # 成交量
    "market",   # 市场标识 (ashare/hkstock/usstock/futures/crypto)
    "symbol",   # 品种代码
]
```

### 4. 数据质量问题处理

**常见问题清单：**
- **缺失值**：停牌日、节假日、数据缺失
- **时间对齐**：不同市场的交易日历不同
- **复权**：A股/港股存在除权除息导致价格跳空
- **单位不一致**：价格货币不同（USD/CNY/HKD）
- **时区**：加密货币数据通常为 UTC，需转换

---

## 💡 示例 / 实操

### 完整的多市场数据采集脚本

```python
"""
multi_market_data.py
统一采集五大市场日线数据，保存为 CSV
"""
import os
import time
import pandas as pd
import akshare as ak
import yfinance as yf
import ccxt

OUTPUT_DIR = "./data/week01"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 工具函数 ----

def save_csv(df: pd.DataFrame, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  ✅ 已保存：{path}（{len(df)} 行）")

def standardize(df, symbol, market, date_col, open_col, high_col, low_col, close_col, vol_col):
    result = pd.DataFrame()
    result["date"]   = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    result["open"]   = pd.to_numeric(df[open_col], errors="coerce")
    result["high"]   = pd.to_numeric(df[high_col], errors="coerce")
    result["low"]    = pd.to_numeric(df[low_col], errors="coerce")
    result["close"]  = pd.to_numeric(df[close_col], errors="coerce")
    result["volume"] = pd.to_numeric(df[vol_col], errors="coerce")
    result["market"] = market
    result["symbol"] = symbol
    return result.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


# ==== 1. A股：沪深300指数 ====
print("\n[A股] 获取沪深300指数...")
try:
    df_a = ak.stock_zh_index_daily(symbol="sh000300")
    df_a_std = standardize(df_a, "000300", "ashare",
                           "date", "open", "high", "low", "close", "volume")
    # 过滤近2年
    df_a_std = df_a_std[df_a_std["date"] >= "2023-01-01"]
    save_csv(df_a_std, "ashare_hs300.csv")
except Exception as e:
    print(f"  ❌ 失败：{e}")


# ==== 2. 港股：恒生指数 ====
print("\n[港股] 获取恒生指数...")
try:
    df_hk = ak.stock_hk_index_daily_em(symbol="恒生指数")
    df_hk_std = standardize(df_hk, "HSI", "hkstock",
                            "日期", "开盘", "最高", "最低", "收盘", "成交量")
    df_hk_std = df_hk_std[df_hk_std["date"] >= "2023-01-01"]
    save_csv(df_hk_std, "hkstock_hsi.csv")
except Exception as e:
    print(f"  ❌ 失败：{e}")


# ==== 3. 美股：标普500 + 纳斯达克100 ====
print("\n[美股] 获取标普500 + 纳斯达克100...")
for ticker, fname in [("^GSPC", "sp500"), ("^NDX", "nasdaq100")]:
    try:
        df_us = yf.download(ticker, start="2023-01-01", auto_adjust=True)
        df_us = df_us.reset_index()
        df_us_std = standardize(df_us, ticker, "usstock",
                                "Date", "Open", "High", "Low", "Close", "Volume")
        save_csv(df_us_std, f"usstock_{fname}.csv")
    except Exception as e:
        print(f"  ❌ {ticker} 失败：{e}")
    time.sleep(1)


# ==== 4. 期货：螺纹钢主力 ====
print("\n[期货] 获取螺纹钢主力合约...")
try:
    df_fut = ak.futures_main_sina(symbol="RB0", start_date="20230101", end_date="20241231")
    df_fut_std = standardize(df_fut, "RB_main", "futures",
                             "date", "open", "high", "low", "close", "volume")
    save_csv(df_fut_std, "futures_rb_main.csv")
except Exception as e:
    print(f"  ❌ 失败：{e}")


# ==== 5. 加密货币：BTC/USDT ====
print("\n[加密] 获取 BTC/USDT 日线...")
try:
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=730)
    df_btc = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df_btc["ts"] = pd.to_datetime(df_btc["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    df_btc_std = standardize(df_btc, "BTCUSDT", "crypto",
                             "ts", "open", "high", "low", "close", "volume")
    df_btc_std = df_btc_std[df_btc_std["date"] >= "2023-01-01"]
    save_csv(df_btc_std, "crypto_btcusdt.csv")
except Exception as e:
    print(f"  ❌ 失败：{e}")

print("\n✅ 所有数据采集完成！")
```

### 简单的数据质量检验

```python
import pandas as pd
import os

DATA_DIR = "./data/week01"
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

for f in files:
    df = pd.read_csv(os.path.join(DATA_DIR, f))
    null_count = df[["open","high","low","close","volume"]].isnull().sum().sum()
    print(f"\n📊 {f}")
    print(f"  行数：{len(df)}")
    print(f"  日期范围：{df['date'].min()} ~ {df['date'].max()}")
    print(f"  空值总数：{null_count}")
    print(f"  收盘价范围：{df['close'].min():.2f} ~ {df['close'].max():.2f}")
```

---

## 🏋️ 动手练习

### 练习 1：运行完整采集脚本并检验数据

- [ ] 运行 `multi_market_data.py`，确保5个文件均成功生成
- [ ] 运行数据质量检验，记录每个市场的数据行数和日期范围
- [ ] 如果某个接口失败，尝试换用其他数据源（记录备选方案）

### 练习 2：计算并对比各市场的基础统计指标

```python
import pandas as pd
import numpy as np

markets_stats = []
for f in files:
    df = pd.read_csv(...)
    df["returns"] = df["close"].pct_change()
    stats = {
        "市场": df["symbol"].iloc[0],
        "年化收益率": df["returns"].mean() * 252,
        "年化波动率": df["returns"].std() * np.sqrt(252),
        "夏普比率（无风险=3%）": (df["returns"].mean() * 252 - 0.03) / (df["returns"].std() * np.sqrt(252)),
        "最大回撤": ((df["close"] - df["close"].cummax()) / df["close"].cummax()).min()
    }
    markets_stats.append(stats)

df_stats = pd.DataFrame(markets_stats)
print(df_stats.to_string(index=False))
```

### 练习 3：节假日与交易日历差异

```python
# 对比 A股、美股、加密三个市场的交易日数量
# A股：约245个交易日/年
# 美股：约252个交易日/年
# 加密：365个交易日/年（全年无休）

# 用 pandas 找出：
# - 哪些日期 A股交易但美股不交易？（中国节假日 vs 美国节假日）
# - 哪些日期美股交易但A股不交易？
```

---

## 📝 小结

**今天学到了什么：**

1. **建立了统一的多市场数据管道**，5个市场的数据用统一格式存储
2. **数据标准化是量化的基础工程**：列名统一、时间格式统一、空值处理
3. **不同市场的数据质量和更新频率不同**，加密货币最全（365天），A股最需要注意节假日
4. **接口随时可能失效**，要有备选数据源意识（akshare / tushare / baostock 可互为补充）
5. **夏普比率是衡量风险调整后收益的核心指标**，是后续策略评估的基础

**关键术语：**
- OHLCV：Open / High / Low / Close / Volume 五列数据
- 年化：将日度指标转换为年度，收益 ×252，波动 ×√252
- 夏普比率：（年化收益 - 无风险利率）/ 年化波动率
- 交易日历：各市场因节假日不同，交易日数量有差异

**明日预告：** D7 Week 1 回顾与横向对比，完成周总结

---

*D6 完成打卡 ✅*
