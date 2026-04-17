# D6：完整数据处理实战（Tushare获取 → 清洗 → 统计）

> **Week 2 · Day 6** | 量化金融基础系列

---

## 1. 量化数据工作流概览

量化策略的基础是**干净、可靠的数据**。一个完整的数据处理流程包含以下阶段：

```
原始数据获取  →  数据清洗  →  特征工程  →  统计分析  →  可视化输出
   (Tushare)     (缺失/异常)  (收益率/因子)  (描述统计)
```

每个环节的错误都会向下游传播，"垃圾进、垃圾出"是量化从业者最忌讳的问题。

---

## 2. Tushare Pro 数据接口

### 2.1 安装与初始化

```python
# 安装 tushare
# pip install tushare

import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 初始化（需要注册获取 token）
# ts.set_token('YOUR_TOKEN_HERE')
# pro = ts.pro_api()
```

### 2.2 主要数据接口

| 接口 | 函数 | 说明 |
|------|------|------|
| 日线行情 | `pro.daily()` | OHLCV + 涨跌幅 |
| 指数日线 | `pro.index_daily()` | 指数行情 |
| 复权因子 | `pro.adj_factor()` | 前后复权因子 |
| 财务数据 | `pro.fina_indicator()` | ROE/PE/PB 等 |
| 股票列表 | `pro.stock_basic()` | 股票基础信息 |
| 交易日历 | `pro.trade_cal()` | 交易日判断 |

---

## 3. 完整实战代码

### 3.1 数据获取模块（Tushare + 本地模拟两用）

```python
# ============================================================
# 方案A：Tushare Pro 真实数据（需要 token）
# ============================================================
def fetch_tushare_data(ts_code, start_date, end_date, token=None):
    """
    获取 Tushare 日线数据（含前复权）
    ts_code: 股票代码，如 '000300.SH'（沪深300）
    """
    if token:
        ts.set_token(token)
    pro = ts.pro_api()
    
    # 获取日线行情
    df = pro.daily(
        ts_code=ts_code,
        start_date=start_date.replace('-', ''),
        end_date=end_date.replace('-', '')
    )
    
    # 获取复权因子
    adj = pro.adj_factor(
        ts_code=ts_code,
        start_date=start_date.replace('-', ''),
        end_date=end_date.replace('-', '')
    )
    
    # 合并复权
    df = df.merge(adj[['trade_date', 'adj_factor']], on='trade_date', how='left')
    df['close_adj'] = df['close'] * df['adj_factor']
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    return df


# ============================================================
# 方案B：模拟数据（无需 token，演示用）
# ============================================================
def generate_mock_data(ts_code='000300.SH', start='2019-01-01', end='2024-01-01', seed=42):
    """生成模拟的股票日线数据，结构与 Tushare 完全一致"""
    np.random.seed(seed)
    
    # 生成交易日序列
    dates = pd.bdate_range(start=start, end=end, freq='B')
    n = len(dates)
    
    # 几何布朗运动
    mu = 0.06 / 252     # 年化 6% 日均
    sigma = 0.20 / np.sqrt(252)  # 年化 20% 波动
    
    log_ret = np.random.normal(mu, sigma, n)
    # 加入偶发跳跃（模拟黑天鹅）
    jumps = np.random.choice([0, 1], n, p=[0.985, 0.015])
    jump_size = np.random.normal(-0.04, 0.02, n)
    log_ret += jumps * jump_size
    
    close = 4000 * np.exp(np.cumsum(log_ret))
    
    # 构造 OHLCV
    intraday_vol = sigma * 0.5
    high = close * np.exp(np.abs(np.random.normal(0, intraday_vol, n)))
    low = close * np.exp(-np.abs(np.random.normal(0, intraday_vol, n)))
    open_ = close * np.exp(np.random.normal(0, intraday_vol * 0.3, n))
    volume = np.random.lognormal(10, 0.5, n) * 1e6
    
    # 构造缺失值（模拟真实数据问题）
    random_nan_idx = np.random.choice(n, size=int(n * 0.005), replace=False)
    volume[random_nan_idx] = np.nan
    
    # 构造异常值（模拟数据错误）
    outlier_idx = np.random.choice(n, size=3, replace=False)
    close[outlier_idx] = close[outlier_idx] * 10  # 价格异常放大
    
    df = pd.DataFrame({
        'trade_date': dates.strftime('%Y%m%d'),
        'ts_code': ts_code,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'vol': volume,
        'pct_chg': np.concatenate([[np.nan], np.diff(close) / close[:-1] * 100]),
        'adj_factor': np.ones(n) * 1.2,
    })
    df['close_adj'] = df['close'] * df['adj_factor']
    
    return df


# 使用模拟数据（实际项目中替换为 fetch_tushare_data）
raw_df = generate_mock_data('000300.SH', '2019-01-01', '2024-01-01')
print(f"获取到 {len(raw_df)} 条记录")
print(raw_df.head())
print(f"\n数据时间范围: {raw_df['trade_date'].min()} ~ {raw_df['trade_date'].max()}")
```

### 3.2 数据清洗模块

```python
def clean_price_data(df):
    """
    金融数据清洗流程
    返回清洗后的 DataFrame 和清洗报告
    """
    report = {}
    df = df.copy()
    
    # Step 1: 日期格式转换
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    # Step 2: 检测重复行
    dup_count = df.duplicated(subset=['trade_date']).sum()
    df = df.drop_duplicates(subset=['trade_date'])
    report['重复行数'] = dup_count
    
    # Step 3: 检测缺失值
    missing_before = df.isnull().sum()
    report['缺失值统计'] = missing_before.to_dict()
    
    # 成交量缺失→用前后均值填充
    df['vol'] = df['vol'].interpolate(method='linear')
    # 价格缺失→前向填充（最保守策略）
    price_cols = ['open', 'high', 'low', 'close', 'close_adj']
    df[price_cols] = df[price_cols].fillna(method='ffill')
    
    # Step 4: 价格异常值检测（Z-score法）
    def detect_outliers_zscore(series, threshold=5):
        """Z-score > threshold 视为异常"""
        log_ret = np.log(series / series.shift(1)).dropna()
        z_scores = np.abs((log_ret - log_ret.mean()) / log_ret.std())
        return z_scores[z_scores > threshold].index
    
    outlier_idx = detect_outliers_zscore(df['close_adj'])
    report['价格异常值索引'] = list(outlier_idx)
    print(f"检测到 {len(outlier_idx)} 个价格异常点: {[df.loc[i, 'trade_date'].date() for i in outlier_idx]}")
    
    # 异常价格→用前后日均值替换
    for idx in outlier_idx:
        prev_val = df.loc[idx-1, 'close_adj'] if idx > 0 else np.nan
        next_val = df.loc[idx+1, 'close_adj'] if idx < len(df)-1 else np.nan
        df.loc[idx, 'close_adj'] = np.nanmean([prev_val, next_val])
    
    # Step 5: OHLC 逻辑校验
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['close']) |
        (df['low'] > df['close'])
    ).sum()
    report['OHLC逻辑错误数'] = int(invalid_ohlc)
    if invalid_ohlc > 0:
        print(f"警告：{invalid_ohlc} 条 OHLC 逻辑错误")
    
    # Step 6: 设置日期索引
    df = df.set_index('trade_date')
    
    return df, report

clean_df, report = clean_price_data(raw_df)

print("\n=== 数据清洗报告 ===")
for k, v in report.items():
    if k == '缺失值统计':
        non_zero = {col: cnt for col, cnt in v.items() if cnt > 0}
        print(f"{k}: {non_zero}")
    else:
        print(f"{k}: {v}")
print(f"\n清洗后数据: {len(clean_df)} 条")
```

### 3.3 特征工程模块

```python
def feature_engineering(df):
    """
    计算量化分析所需的各类特征
    """
    df = df.copy()
    price = df['close_adj']
    
    # 1. 收益率
    df['log_ret'] = np.log(price / price.shift(1))
    df['simple_ret'] = price.pct_change()
    
    # 2. 多周期收益率
    for period in [5, 10, 20, 60]:
        df[f'ret_{period}d'] = np.log(price / price.shift(period))
    
    # 3. 波动率（滚动标准差）
    for window in [10, 20, 60]:
        df[f'vol_{window}d'] = df['log_ret'].rolling(window).std() * np.sqrt(252)
    
    # 4. 移动平均线
    for window in [5, 20, 60, 120]:
        df[f'ma_{window}'] = price.rolling(window).mean()
    
    # 5. ATR（真实波动幅度）
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_14'] = df['tr'].rolling(14).mean()
    
    # 6. 成交量相关
    df['vol_ma_20'] = df['vol'].rolling(20).mean()
    df['vol_ratio'] = df['vol'] / df['vol_ma_20']  # 量比
    
    # 7. 价格与均线偏离度
    df['ma20_deviation'] = (price - df['ma_20']) / df['ma_20']
    
    # 8. 最大回撤（滚动 252 日）
    rolling_max = price.rolling(252, min_periods=1).max()
    df['drawdown'] = (price - rolling_max) / rolling_max
    
    return df.dropna(subset=['log_ret'])

featured_df = feature_engineering(clean_df)
print(f"特征工程完成，特征数: {len(featured_df.columns)}")
print(f"有效数据行数: {len(featured_df)}")
print("\n特征列表:")
print(featured_df.columns.tolist())
```

### 3.4 统计分析模块

```python
import scipy.stats as scipy_stats

def statistical_analysis(df, ret_col='log_ret'):
    """全面统计分析"""
    returns = df[ret_col].dropna()
    
    # 基本描述统计
    stats_dict = {
        '样本量': len(returns),
        '均值': returns.mean(),
        '中位数': returns.median(),
        '标准差': returns.std(),
        '偏度': scipy_stats.skew(returns),
        '超额峰度': scipy_stats.kurtosis(returns),
        '最小值': returns.min(),
        '最大值': returns.max(),
    }
    
    # 年化指标
    trading_days = 252
    stats_dict['年化收益率'] = returns.mean() * trading_days
    stats_dict['年化波动率'] = returns.std() * np.sqrt(trading_days)
    stats_dict['Sharpe比率'] = (stats_dict['年化收益率'] - 0.02) / stats_dict['年化波动率']
    
    # VaR & CVaR
    for alpha in [0.95, 0.99]:
        var = -np.percentile(returns, (1-alpha)*100)
        cvar = -returns[returns < -var].mean()
        stats_dict[f'VaR_{int(alpha*100)}%'] = var
        stats_dict[f'CVaR_{int(alpha*100)}%'] = cvar
    
    # 最大回撤
    stats_dict['最大回撤'] = df['drawdown'].min()
    stats_dict['Calmar比率'] = stats_dict['年化收益率'] / abs(stats_dict['最大回撤'])
    
    # 正态性检验
    jb_stat, jb_p = scipy_stats.jarque_bera(returns)
    stats_dict['JB统计量'] = jb_stat
    stats_dict['JB_p值'] = jb_p
    
    return stats_dict

stats = statistical_analysis(featured_df)

print("=" * 50)
print("      量化统计分析报告")
print("=" * 50)
categories = {
    '基础统计': ['样本量', '均值', '中位数', '标准差', '偏度', '超额峰度', '最小值', '最大值'],
    '年化指标': ['年化收益率', '年化波动率', 'Sharpe比率'],
    '风险度量': ['VaR_95%', 'CVaR_95%', 'VaR_99%', 'CVaR_99%', '最大回撤', 'Calmar比率'],
    '正态检验': ['JB统计量', 'JB_p值']
}

for category, keys in categories.items():
    print(f"\n【{category}】")
    for k in keys:
        v = stats[k]
        if isinstance(v, float):
            if abs(v) < 0.0001:
                print(f"  {k:15s}: {v:.6e}")
            elif abs(v) > 1000:
                print(f"  {k:15s}: {v:.2f}")
            else:
                print(f"  {k:15s}: {v:.6f}")
        else:
            print(f"  {k:15s}: {v}")
```

### 3.5 完整可视化输出

```python
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

price = featured_df['close_adj']
returns = featured_df['log_ret']

# 1. 价格走势（含均线）
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(price, label='收盘价(复权)', color='steelblue', linewidth=1.2)
ax1.plot(featured_df['ma_20'], label='MA20', color='orange', linewidth=1, alpha=0.8)
ax1.plot(featured_df['ma_60'], label='MA60', color='green', linewidth=1, alpha=0.8)
ax1.set_title('沪深300指数价格走势（模拟数据）')
ax1.legend(loc='upper left')
ax1.set_ylabel('点位')

# 2. 日收益率序列
ax2 = fig.add_subplot(gs[1, :2])
ax2.bar(range(len(returns)), returns.values, 
        color=['red' if r < 0 else 'green' for r in returns.values],
        alpha=0.6, width=1)
ax2.set_title('日对数收益率序列')
ax2.set_ylabel('对数收益率')
ax2.axhline(0, color='black', linewidth=0.5)

# 3. 滚动波动率
ax3 = fig.add_subplot(gs[1, 2])
ax3.plot(featured_df['vol_20d'] * 100, color='purple', linewidth=1)
ax3.set_title('20日滚动年化波动率(%)')
ax3.set_ylabel('年化波动率 (%)')

# 4. 收益率分布
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(returns.values, bins=60, density=True, 
         color='steelblue', alpha=0.7, edgecolor='white')
x_fit = np.linspace(returns.min(), returns.max(), 200)
from scipy.stats import norm as norm_dist
ax4.plot(x_fit, norm_dist.pdf(x_fit, returns.mean(), returns.std()),
         'r-', linewidth=2, label='正态拟合')
ax4.set_title('收益率分布')
ax4.legend()

# 5. QQ图
ax5 = fig.add_subplot(gs[2, 1])
(osm, osr), (slope, intercept, r) = scipy_stats.probplot(returns.values, dist="norm")
ax5.scatter(osm, osr, alpha=0.3, s=3, color='steelblue')
ax5.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2)
ax5.set_title(f'QQ图 ($R^2$={r**2:.4f})')
ax5.set_xlabel('理论分位数')
ax5.set_ylabel('样本分位数')

# 6. 回撤图
ax6 = fig.add_subplot(gs[2, 2])
ax6.fill_between(range(len(featured_df['drawdown'])), 
                  featured_df['drawdown'].values, 0,
                  color='red', alpha=0.4)
ax6.set_title(f'回撤序列（最大: {featured_df["drawdown"].min()*100:.1f}%）')
ax6.set_ylabel('回撤幅度')

plt.savefig('full_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("完整分析图表已保存")
```

---

## 4. 常见数据问题与处理策略

| 问题类型 | 识别方式 | 处理策略 |
|----------|----------|----------|
| 价格缺失 | `isnull()` | 前向填充（避免未来函数） |
| 成交量缺失 | `isnull()` | 线性插值或均值填充 |
| 价格跳跃 | Z-score > 5σ | 前后均值替换 + 手动核查 |
| 复权错误 | 历史价格折叠 | 重新获取复权因子 |
| 重复数据 | `duplicated()` | 保留最新一条 |
| 停牌数据 | 连续相同价格 | 标记后在分析中剔除 |

---

## 5. 关键公式汇总

$$
r_t^{adj} = \ln\frac{P_t^{adj}}{P_{t-1}^{adj}}, \quad P_t^{adj} = P_t \times \text{adj\_factor}_t
$$

$$
\text{Z-score} = \frac{|r_t - \bar{r}|}{s_r} > 5 \Rightarrow \text{异常值}
$$

$$
\hat{\sigma}_{annual} = \hat{\sigma}_{daily} \times \sqrt{252}
$$

---

## 6. 小结

- 数据清洗是量化分析最重要的基础工作，脏数据会产生虚假信号
- Tushare Pro 提供了丰富的A股数据接口，复权处理必须正确
- 特征工程将原始价格转化为分析维度：收益率、波动率、技术指标
- 统计分析应涵盖描述统计、风险度量、正态检验全维度
- 可视化是快速发现数据问题和规律的最有效手段

> 下一篇：D7 综合实战（沪深300风险收益画像）——将 Week2 全部知识整合，输出完整的指数画像报告
