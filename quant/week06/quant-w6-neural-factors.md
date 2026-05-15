---
layout: default
title: "D4 · 神经网络因子"
render_with_liquid: false
---

# D4 · 神经网络因子

> 深度学习进入量化：可能性与陷阱并存。

---

## 1. 神经网络在量化中的应用

| 应用场景 | 模型类型 | 特点 |
|---------|---------|------|
| 价格预测 | LSTM/Transformer | 时序依赖建模 |
| 截面选股 | MLP/TabNet | 因子非线性组合 |
| 另类数据 | CNN/BERT | 图像/文本特征提取 |
| 订单流预测 | WaveNet | 超高频 |

---

## 2. 简单 MLP 选股模型

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class StockMLP(nn.Module):
    """
    多层感知机因子模型
    输入：多因子特征向量
    输出：股票得分（用于截面排名）
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class ICLoss(nn.Module):
    """
    IC 损失函数：最大化预测值和真实收益的相关系数
    （比 MSE 更符合量化目标）
    """
    def forward(self, pred, target):
        # 截面标准化
        pred_z = (pred - pred.mean()) / (pred.std() + 1e-8)
        target_z = (target - target.mean()) / (target.std() + 1e-8)
        # 最小化负相关（即最大化 IC）
        ic = (pred_z * target_z).mean()
        return -ic


def train_mlp(X_train, y_train, X_val, y_val, epochs=100, lr=1e-3):
    """训练 MLP 选股模型"""
    model = StockMLP(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = ICLoss()
    
    X_train_t = torch.FloatTensor(X_train.values)
    y_train_t = torch.FloatTensor(y_train.values)
    X_val_t = torch.FloatTensor(X_val.values)
    y_val_t = torch.FloatTensor(y_val.values)
    
    best_val_ic = -np.inf
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_ic = criterion(val_pred, y_val_t).item() * -1  # 转回正 IC
        
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_ic={val_ic:.4f}")
    
    # 加载最优模型
    model.load_state_dict(best_model_state)
    return model
```

---

## 3. LSTM 时序因子

```python
class StockLSTM(nn.Module):
    """
    LSTM 因子模型
    输入：过去 T 天的特征序列
    输出：当期股票得分
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.attention = nn.Linear(hidden_dim, 1)
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 注意力机制：对时间步加权
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return self.output(context).squeeze(-1)


def prepare_sequence_data(feature_df, return_df, seq_len=20):
    """
    将特征数据转换为 LSTM 输入格式
    """
    samples = []
    targets = []
    stocks = []
    dates = []
    
    for stock in feature_df.columns.get_level_values(0).unique():
        if stock not in feature_df.columns:
            continue
        
        stock_feat = feature_df[stock].values  # (T, n_features)
        stock_ret = return_df[stock].values     # (T,)
        
        for t in range(seq_len, len(stock_feat)):
            if np.any(np.isnan(stock_feat[t-seq_len:t])) or np.isnan(stock_ret[t]):
                continue
            samples.append(stock_feat[t-seq_len:t])
            targets.append(stock_ret[t])
            stocks.append(stock)
            dates.append(feature_df.index[t])
    
    return (np.array(samples), np.array(targets), 
            np.array(stocks), np.array(dates))
```

---

## 4. 神经网络因子的主要风险

| 风险 | 严重程度 | 说明 |
|------|---------|------|
| 过拟合 | ★★★★★ | 参数多、样本小、噪声大 |
| 黑箱不可解释 | ★★★★ | 监管合规风险 |
| 训练不稳定 | ★★★ | 不同随机种子结果差异大 |
| 计算成本高 | ★★★ | 日频重新训练成本 |
| 特征工程依赖 | ★★★ | 输入质量决定输出质量 |

---

## 5. 实践建议

1. **先用树模型建立 baseline**，再考虑是否引入神经网络
2. **使用多种随机种子训练多个模型**，取平均（集成）降低不稳定性
3. **ICLoss > MSELoss**：用 IC 作为损失函数，更符合选股目标
4. **严格的样本外测试**：DNN 非常容易"记住"训练集

---

## 小结

| 维度 | 内容 |
|------|------|
| 适合场景 | 另类数据（文本/图像）、高频数据 |
| 截面选股 | 树模型通常更稳健 |
| 关键技巧 | IC 损失、Batch Norm、Dropout |
| 最大风险 | 过拟合（务必 OOS 测试）|
| 建议 | 先 LightGBM，再考虑 DNN |
