---
layout: default
title: "D1 · Alpha158 因子库解析"
render_with_liquid: false
---

# D1 · Alpha158 因子库解析

> Alpha158 是微软 Qlib 开源的 158 个量价因子库，是 ML 量化研究的标准 benchmark。

---

## 1. 什么是 Alpha158

**Alpha158**：基于 OHLCV（开高低收量）数据构建的 158 个技术型因子集合，设计用于机器学习模型的特征输入。

**设计哲学**：
- 纯量价，不需要财务数据，频率高，更新快
- 涵盖动量、波动率、成交量、价格形态等多个维度
- 经过标准化处理，可直接作为 ML 特征

---

## 2. Alpha158 的因子类别

### 2.1 收益率类（Return Features）

```python
import pandas as pd
import numpy as np

class Alpha158:
    """Alpha158 因子计算（部分示例）"""
    
    def __init__(self, close, open_, high, low, volume, vwap):
        self.close = close
        self.open = open_
        self.high = high
        self.low = low
        self.volume = volume
        self.vwap = vwap
    
    # ===== 收益率类 =====
    
    def RESI5(self):
        """5日残差收益（去市场 beta 后）"""
        ret = self.close.pct_change()
        mkt = ret.mean(axis=1)  # 简化：用全市场平均代替市场收益
        # 对每只股票做截面回归残差
        result = pd.DataFrame(index=ret.index, columns=ret.columns)
        for date in ret.index:
            r = ret.loc[date].dropna()
            m = mkt[date]
            if pd.isna(m): continue
            # 简单中心化作为残差
            result.loc[date, r.index] = r - m
        return result.rolling(5).mean()
    
    def ROC5(self):
        """5日变化率（Rate of Change）"""
        return self.close / self.close.shift(5) - 1
    
    def ROC10(self):
        return self.close / self.close.shift(10) - 1
    
    def ROC20(self):
        return self.close / self.close.shift(20) - 1
    
    # ===== 波动率类 =====
    
    def STD5(self):
        """5日收益率标准差"""
        return self.close.pct_change().rolling(5).std()
    
    def STD10(self):
        return self.close.pct_change().rolling(10).std()
    
    def STD20(self):
        return self.close.pct_change().rolling(20).std()
    
    def BETA5(self):
        """5日滚动 Beta"""
        ret = self.close.pct_change()
        mkt = ret.mean(axis=1)
        betas = pd.DataFrame(index=ret.index, columns=ret.columns)
        for col in ret.columns:
            s = ret[col]
            rolling_cov = s.rolling(5).cov(mkt)
            rolling_var = mkt.rolling(5).var()
            betas[col] = rolling_cov / rolling_var
        return betas
    
    # ===== 量价关系类 =====
    
    def CORD5(self):
        """5日量价相关系数"""
        ret = self.close.pct_change()
        vol_chg = self.volume.pct_change()
        result = pd.DataFrame(index=ret.index, columns=ret.columns)
        for col in ret.columns:
            result[col] = ret[col].rolling(5).corr(vol_chg[col])
        return result
    
    def WVMA5(self):
        """5日成交量加权移动均线比"""
        vwap_ma = (self.close * self.volume).rolling(5).sum() / self.volume.rolling(5).sum()
        return vwap_ma / self.close - 1
    
    # ===== 价格位置类 =====
    
    def HIGH0(self):
        """今日最高价在最近 5 日中的位置"""
        return (self.high - self.low.rolling(5).min()) / \
               (self.high.rolling(5).max() - self.low.rolling(5).min() + 1e-8)
    
    def LOW0(self):
        """今日最低价在最近 5 日中的位置"""
        return (self.low - self.low.rolling(5).min()) / \
               (self.high.rolling(5).max() - self.low.rolling(5).min() + 1e-8)
    
    def KUP(self):
        """上影线长度"""
        return self.high / self.close.apply(lambda x: max(x, self.open)) - 1
    
    def KDOWN(self):
        """下影线长度"""
        return self.close.apply(lambda x: min(x, self.open)) / self.low - 1
```

---

## 3. Alpha158 的标准处理

```python
def preprocess_alpha158(alpha_df, market_cap=None):
    """
    Alpha158 标准预处理流程
    1. 去极值（Winsorize）
    2. 市值中性化（可选）
    3. 截面标准化（Z-score）
    4. 缺失值填充
    """
    from scipy.stats.mstats import winsorize
    
    result = alpha_df.copy()
    
    for date in result.index:
        row = result.loc[date].dropna()
        
        if len(row) < 20:
            continue
        
        # 去极值
        row_wins = pd.Series(
            winsorize(row.values, limits=[0.025, 0.025]),
            index=row.index
        )
        
        # Z-score
        mu, sigma = row_wins.mean(), row_wins.std()
        if sigma > 1e-8:
            result.loc[date, row.index] = (row_wins - mu) / sigma
    
    # 填充剩余缺失值
    result = result.fillna(0)  # 或用截面均值填充
    return result
```

---

## 4. 使用 Qlib 获取 Alpha158

```python
import qlib
from qlib.constant import REG_CN
from qlib.data import D

# 初始化 Qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN)

# 获取部分 Alpha158 因子
fields = [
    '$close/Ref($close,1)-1',       # 日收益率
    'Ref($close/Ref($close,1)-1,1)', # 昨日收益率
    'Mean($volume,5)/$volume',       # 成交量比
    'Std($close/Ref($close,1)-1,5)', # 5日波动率
]

data = D.features(
    instruments='csi300',
    fields=fields,
    start_time='2020-01-01',
    end_time='2023-12-31',
    freq='day'
)
```

---

## 5. Alpha158 的局限性

| 局限 | 说明 |
|------|------|
| 纯量价 | 不包含基本面信息，在基本面驱动行情中效果差 |
| 同质化 | 被广泛使用后，信号的 alpha 已经部分衰减 |
| 短周期 | 大多数因子是短期技术指标，长期预测力弱 |
| A股适配 | 基于美股数据设计，部分因子需要针对 A 股调整 |

---

## 小结

| 维度 | 内容 |
|------|------|
| 因子数量 | 158 个 |
| 数据依赖 | 纯量价 OHLCV |
| 因子类别 | 收益率、波动率、量价关系、价格位置 |
| 预处理 | 去极值 → 中性化 → Z-score |
| 常用框架 | Qlib（微软开源）|
