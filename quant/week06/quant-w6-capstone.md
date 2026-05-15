---
layout: default
title: "D7 · Week 6 综合实战：ML 多因子框架"
render_with_liquid: false
---

# D7 · Week 6 综合实战：ML 多因子框架

> 把 Alpha158 + LightGBM + 集成 + 过拟合防范串联成完整的 ML 量化 Pipeline。

---

## 本周核心知识回顾

| 主题 | 要点 |
|------|------|
| Alpha158 | 158 个量价因子，是 ML 量化的标准 benchmark |
| 特征工程 | 时序 + 截面，必须按时间顺序分割 |
| LightGBM | 量化界最常用的 ML 模型，快且不需要标准化 |
| 神经网络 | 适合另类数据，截面选股树模型更稳健 |
| 集成 | Bagging + 异质集成，降低单模型方差 |
| 过拟合 | 最大风险，必须严格样本外测试 |

---

## 完整 ML 量化 Pipeline

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from scipy.stats.mstats import winsorize

class MLQuantPipeline:
    """
    完整的机器学习量化 Pipeline
    """
    
    def __init__(self, 
                 train_months=36,
                 test_months=3,
                 n_ensemble=5,
                 n_select=50):
        self.train_months = train_months
        self.test_months = test_months
        self.n_ensemble = n_ensemble
        self.n_select = n_select
        self.models = []
    
    # ===== Step 1: 因子计算 =====
    
    def compute_features(self, close, high, low, volume):
        """计算 Alpha158 子集特征"""
        features = pd.DataFrame()
        ret = close.pct_change()
        
        # 多期收益率
        for p in [1, 5, 10, 20, 60]:
            features[f'ret_{p}'] = close.pct_change(p)
            features[f'vol_{p}'] = ret.rolling(p).std()
        
        # 价格位置
        for p in [5, 20, 60]:
            features[f'high_pos_{p}'] = (close - close.rolling(p).min()) / \
                (close.rolling(p).max() - close.rolling(p).min() + 1e-8)
        
        # 量价关系
        features['vol_ratio_5_20'] = volume.rolling(5).mean() / volume.rolling(20).mean()
        features['price_vol_corr_5'] = ret.rolling(5).corr(volume.pct_change())
        
        # 波动率比（短期 vs 长期）
        features['vol_ratio_5_20d'] = ret.rolling(5).std() / ret.rolling(20).std()
        
        return features
    
    # ===== Step 2: 截面预处理 =====
    
    def preprocess(self, features_cross_section):
        """
        单期截面预处理
        features_cross_section: DataFrame，index=stock，columns=features
        """
        result = features_cross_section.copy()
        
        for col in result.columns:
            col_data = result[col].dropna()
            if len(col_data) < 20:
                continue
            # 去极值
            col_wins = pd.Series(
                winsorize(col_data.values, limits=[0.025, 0.025]),
                index=col_data.index
            )
            # Z-score
            result[col] = (col_wins - col_wins.mean()) / (col_wins.std() + 1e-8)
        
        return result.fillna(0)
    
    # ===== Step 3: 训练集成模型 =====
    
    def train_ensemble(self, X_train, y_train):
        """训练 n 个 LightGBM 模型（不同随机种子）"""
        self.models = []
        
        for seed in range(self.n_ensemble):
            model = lgb.LGBMRegressor(
                n_estimators=200,
                num_leaves=31,
                max_depth=5,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                lambda_l1=0.1,
                lambda_l2=0.1,
                min_child_samples=20,
                random_state=seed,
                verbose=-1
            )
            model.fit(X_train, y_train)
            self.models.append(model)
        
        return self
    
    # ===== Step 4: 集成预测 =====
    
    def predict_ensemble(self, X_test):
        """集成预测：多模型平均"""
        preds = np.array([m.predict(X_test) for m in self.models])
        return preds.mean(axis=0)
    
    # ===== Step 5: Walk-Forward 回测 =====
    
    def run_walkforward(self, features_panel, returns_panel):
        """
        完整的 Walk-Forward 样本外测试
        features_panel: dict of {date: DataFrame(stock x features)}
        returns_panel: dict of {date: Series(stock -> next_period_return)}
        """
        dates = sorted(features_panel.keys())
        
        oos_results = []
        
        for i in range(self.train_months, len(dates) - self.test_months, 
                        self.test_months):
            train_dates = dates[i - self.train_months:i]
            test_dates = dates[i:i + self.test_months]
            
            # 构建训练集
            X_list, y_list = [], []
            for d in train_dates:
                if d in features_panel and d in returns_panel:
                    X_d = self.preprocess(features_panel[d])
                    y_d = returns_panel[d].reindex(X_d.index).dropna()
                    X_d = X_d.reindex(y_d.index)
                    if len(X_d) > 20:
                        X_list.append(X_d)
                        y_list.append(y_d)
            
            if not X_list:
                continue
            
            X_train = pd.concat(X_list)
            y_train = pd.concat(y_list)
            
            # 训练
            self.train_ensemble(X_train, y_train)
            
            # 测试
            for d in test_dates:
                if d not in features_panel or d not in returns_panel:
                    continue
                
                X_test = self.preprocess(features_panel[d])
                y_test = returns_panel[d].reindex(X_test.index).dropna()
                X_test = X_test.reindex(y_test.index)
                
                if len(X_test) < 10:
                    continue
                
                pred = self.predict_ensemble(X_test.values)
                ic, _ = spearmanr(pred, y_test.values)
                
                # 选股（取前 N 名）
                scores = pd.Series(pred, index=X_test.index)
                top_stocks = scores.nlargest(self.n_select).index
                
                oos_results.append({
                    'date': d,
                    'ic': ic,
                    'selected_stocks': top_stocks.tolist(),
                    'train_end': train_dates[-1]
                })
        
        return pd.DataFrame(oos_results).set_index('date')
    
    # ===== Step 6: 绩效评估 =====
    
    def evaluate(self, oos_results, returns_panel):
        """评估 Walk-Forward 结果"""
        ic_series = oos_results['ic']
        
        print("=" * 40)
        print("ML 因子模型样本外表现")
        print("=" * 40)
        print(f"平均 IC: {ic_series.mean():.4f}")
        print(f"IC 标准差: {ic_series.std():.4f}")
        print(f"ICIR: {ic_series.mean()/ic_series.std():.4f}")
        print(f"IC > 0 胜率: {(ic_series > 0).mean():.2%}")
        print("=" * 40)
        
        return ic_series


# ===== 开发流程最佳实践 =====

"""
ML 量化开发的黄金流程：

1. 先建 Baseline（等权因子合成）
2. 数据分割：前 60% 训练，中 20% 验证，后 20% 测试（严格封存）
3. 在训练集开发因子 + 特征
4. 在验证集调参，最多用 3-5 次
5. 最终只用一次测试集，结果是最终答案
6. 若样本外/样本内 IC 比 < 60%，说明严重过拟合，回到步骤 3

记住：好的因子胜在逻辑，不胜在参数。
"""
```

---

## 常见面试问题

**Q：ML 模型如何与传统因子结合？**

ML 模型的输出（得分）可以作为一个新因子，与价值、质量等传统因子合成。实践中，ML 因子通常在短周期（月内）更有效，传统因子在长周期更稳健。

**Q：如何防止过拟合？**
1. Walk-Forward 验证，严格时间分割
2. 限制模型复杂度（树深度、正则化）
3. 每个特征必须有经济学逻辑
4. 多重测试校正（Harvey 准则：t > 3.0）

**Q：Alpha158 为什么在A股效果可能下降？**

Alpha158 基于美股数据设计，部分信号（如高频量价关系）在A股特有的涨跌停、T+1、流动性结构下表现不同，需要针对性调整。
