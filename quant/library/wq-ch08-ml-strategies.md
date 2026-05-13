---
layout: default
title: "WhaleQuant Ch08 · 机器学习量化策略"
source: "https://github.com/datawhalechina/whale-quant"
---

# 08 · 机器学习量化策略

> **来源**：[WhaleQuant](https://github.com/datawhalechina/whale-quant) · datawhalechina

---

## 8.1 机器学习在量化中的应用地图

```
监督学习（预测）
├── 回归：预测下期收益率
├── 分类：预测涨跌方向（二分类）
└── 排序：对股票打分排名（LTR）

无监督学习（模式发现）
├── 聚类：市场状态识别（牛/熊/震荡）
└── 降维：因子降维，去除冗余

强化学习（序列决策）
└── 仓位管理、执行优化

深度学习
├── LSTM/Transformer：时序预测
├── CNN：图形模式识别
└── 图神经网络：股票关联分析
```

---

## 8.2 分类模型：预测涨跌

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb

def prepare_features(df: pd.DataFrame, 
                      lookahead_days: int = 20) -> tuple:
    """
    构建特征矩阵和标签
    特征：技术指标 + 基本面因子
    标签：未来N天收益率是否为正
    """
    features = pd.DataFrame(index=df.index)
    
    # 技术指标特征
    # 动量
    for n in [5, 10, 20, 60]:
        features[f'mom_{n}'] = df['close'].pct_change(n)
    
    # 波动率
    for n in [10, 20]:
        features[f'vol_{n}'] = df['close'].pct_change().rolling(n).std()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 均线偏离度
    for n in [20, 60]:
        ma = df['close'].rolling(n).mean()
        features[f'ma_dev_{n}'] = (df['close'] - ma) / ma
    
    # 量价关系
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
    
    # 标签：未来20天收益率是否为正
    future_return = df['close'].shift(-lookahead_days) / df['close'] - 1
    labels = (future_return > 0).astype(int)
    
    # 对齐并去除NaN
    valid = features.notna().all(axis=1) & labels.notna()
    
    return features[valid], labels[valid]

def train_ml_model(features: pd.DataFrame, labels: pd.Series):
    """
    时间序列交叉验证训练
    注意：不能用随机分割，必须保持时间顺序
    """
    # 时序CV（5折）
    tscv = TimeSeriesSplit(n_splits=5)
    
    scaler = StandardScaler()
    
    # 多个模型对比
    models = {
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, 
            max_depth=6, random_state=42, verbose=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        'Logistic': LogisticRegression(C=0.1, random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        fold_scores = []
        
        for train_idx, val_idx in tscv.split(features):
            X_train = features.iloc[train_idx]
            X_val = features.iloc[val_idx]
            y_train = labels.iloc[train_idx]
            y_val = labels.iloc[val_idx]
            
            if name == 'Logistic':
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
            
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            fold_scores.append(score)
        
        results[name] = {
            'mean_accuracy': np.mean(fold_scores),
            'std_accuracy': np.std(fold_scores),
        }
        print(f"{name}: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    return results
```

---

## 8.3 LightGBM 选股因子

```python
import lightgbm as lgb
import shap

def train_lgb_factor(factor_data: pd.DataFrame,
                      target_col: str = 'next_month_return',
                      feature_cols: list = None):
    """
    用 LightGBM 训练非线性因子
    feature_cols: 因子特征列表
    """
    if feature_cols is None:
        feature_cols = [c for c in factor_data.columns if c != target_col]
    
    X = factor_data[feature_cols].fillna(0)
    y = factor_data[target_col]
    
    # 时序分割
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # 训练
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 重要特征：")
    print(importance.head(10))
    
    # SHAP 解释性分析
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[:100])
    shap.summary_plot(shap_values, X_test.iloc[:100])
    
    return model, importance
```

---

## 8.4 LSTM 时序预测

```python
import torch
import torch.nn as nn
import numpy as np

class StockLSTM(nn.Module):
    """LSTM 股价预测模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                  num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # 取最后一个时间步
        return self.fc(out)

def create_sequences(data: np.ndarray, 
                      seq_len: int = 20) -> tuple:
    """创建时序样本"""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])  # 预测收盘价收益率
    return np.array(X), np.array(y)

def train_lstm(X: np.ndarray, y: np.ndarray, 
               n_epochs: int = 100,
               lr: float = 0.001):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 时序分割
    split = int(len(X) * 0.8)
    X_train = torch.FloatTensor(X[:split]).to(device)
    y_train = torch.FloatTensor(y[:split]).to(device)
    X_val = torch.FloatTensor(X[split:]).to(device)
    y_val = torch.FloatTensor(y[split:]).to(device)
    
    model = StockLSTM(input_size=X.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train).squeeze()
        loss = criterion(pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze()
            val_loss = criterion(val_pred, y_val)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Train={loss.item():.6f}, Val={val_loss.item():.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    return model
```

---

## 延伸阅读

- [WhaleQuant 完整教程](https://github.com/datawhalechina/whale-quant)
- [LightGBM 文档](https://lightgbm.readthedocs.io/)
- Lopez de Prado - "Advances in Financial Machine Learning"
