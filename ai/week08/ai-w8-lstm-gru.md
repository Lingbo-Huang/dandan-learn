---
layout: default
title: "LSTM 与 GRU 详解"
render_with_liquid: false
---

# LSTM 与 GRU 详解

## LSTM：长短期记忆网络

### 核心创新：门控机制 + 细胞状态

LSTM 引入**细胞状态** $c_t$（长期记忆）作为信息高速公路，通过三个门精细控制信息流动。

### 四个核心运算

**遗忘门**（决定遗忘多少历史）：

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**输入门**（决定添加多少新信息）：

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**候选细胞**（候选添加的新记忆）：

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

**细胞状态更新**（长期记忆更新）：

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**输出门**（控制当前输出）：

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(c_t)$$

### 门的直觉

| 门 | 作用 | 极端情况 |
|----|------|---------|
| 遗忘门 $f_t=0$ | 完全遗忘历史 | 无长期记忆 |
| 遗忘门 $f_t=1$ | 完全保留历史 | 无遗忘 |
| 输入门 $i_t=0$ | 不添加新信息 | 记忆被冻结 |
| 输出门 $o_t=0$ | 不输出任何信息 | 隐藏状态全零 |

### 梯度流分析

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

当 $f_t \approx 1$ 时，细胞状态梯度不衰减！LSTM 通过控制遗忘门为 1 来维持长距离梯度流。

## GRU：门控循环单元

GRU 是 LSTM 的简化版，合并细胞状态和隐藏状态，减少参数。

**重置门**（控制历史影响程度）：

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

**更新门**（控制更新比例，类似 LSTM 中 $i_t$ 和 $f_t$ 的合并）：

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

**候选隐藏状态**：

$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$

**最终更新**：

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

| | LSTM | GRU |
|--|------|-----|
| 参数量 | $4 \times (d_h^2 + d_xd_h + d_h)$ | $3 \times (d_h^2 + d_xd_h + d_h)$ |
| 状态 | $c_t$ 和 $h_t$ | 只有 $h_t$ |
| 效果 | 长序列更稳 | 短序列略快 |
| 适用 | 长文本、时序 | 短序列、资源受限 |

## Python 实现

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ===== 手写 LSTM Cell =====
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        # 4 个门的权重合并（效率更高）
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * 0.01
        self.b = np.zeros(4 * hidden_size)
        self.hidden_size = hidden_size
    
    def forward(self, x, h_prev, c_prev):
        n = self.hidden_size
        combined = np.concatenate([x, h_prev])
        gates = self.W @ combined + self.b
        
        # 分割四个门
        f = self._sigmoid(gates[0:n])       # 遗忘门
        i = self._sigmoid(gates[n:2*n])     # 输入门
        g = np.tanh(gates[2*n:3*n])         # 候选细胞
        o = self._sigmoid(gates[3*n:4*n])   # 输出门
        
        c = f * c_prev + i * g              # 细胞状态更新
        h = o * np.tanh(c)                  # 隐藏状态
        
        return h, c, (f, i, g, o)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# ===== PyTorch LSTM 实战 =====
# 情感分析任务（二分类）
import torch
from torch.utils.data import DataLoader, TensorDataset

# 模拟数据
np.random.seed(42)
torch.manual_seed(42)

vocab_size = 5000
seq_len = 100
n_samples = 2000
embed_dim = 64
hidden_size = 128

# 随机生成序列（实际应用中应用真实文本）
X = np.random.randint(0, vocab_size, (n_samples, seq_len))
y = (X.mean(axis=1) > vocab_size // 2).astype(int)  # 简单规则生成标签

X_t = torch.LongTensor(X)
y_t = torch.FloatTensor(y)

n_train = int(0.8 * n_samples)
train_ds = TensorDataset(X_t[:n_train], y_t[:n_train])
test_ds = TensorDataset(X_t[n_train:], y_t[n_train:])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers=n_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        emb = self.embedding(x)  # (B, T, E)
        out, _ = self.lstm(emb)  # (B, T, 2H)
        
        # 注意力池化（替代只取最后一步）
        attn_weights = torch.softmax(self.attention(out), dim=1)  # (B, T, 1)
        context = (out * attn_weights).sum(dim=1)                 # (B, 2H)
        
        return self.classifier(context).squeeze()

model = SentimentLSTM(vocab_size, embed_dim, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

history = {'train_loss': [], 'val_acc': []}

for epoch in range(15):
    model.train()
    total_loss = 0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        out = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds = (model(X_b).sigmoid() > 0.5)
            correct += (preds == y_b.bool()).sum().item()
    
    val_acc = correct / len(test_ds)
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    history['train_loss'].append(avg_loss)
    history['val_acc'].append(val_acc)
    print(f"Epoch {epoch:2d}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

# GRU 对比
class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_size, num_layers=n_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        return self.classifier(out[:, -1, :]).squeeze()  # 只用最后一步

# 可视化注意力权重
model.eval()
sample_x = X_t[:1]
with torch.no_grad():
    emb = model.embedding(sample_x)
    out, _ = model.lstm(emb)
    attn = torch.softmax(model.attention(out), dim=1).squeeze()

plt.figure(figsize=(12, 3))
plt.bar(range(seq_len), attn.numpy())
plt.xlabel("时间步")
plt.ylabel("注意力权重")
plt.title("LSTM 注意力权重分布")
plt.tight_layout()

# 训练曲线
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history['train_loss'], 'b-o')
axes[0].set_title('训练 Loss')
axes[0].set_xlabel('Epoch')
axes[1].plot(history['val_acc'], 'r-o')
axes[1].set_title('验证准确率')
axes[1].set_xlabel('Epoch')
plt.tight_layout()
```

## 面试要点

**Q: LSTM 为什么能解决 RNN 的梯度消失问题？**

A: 关键在于细胞状态的更新：$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$。梯度通过 $c_t$ 流回 $c_{t-1}$ 时，乘以的是 $f_t$（逐元素乘）而非矩阵乘法。当网络学会将 $f_t \approx 1$ 时，梯度几乎无衰减地流过。这相当于一种可学习的残差连接。

**Q: 双向 RNN 有什么优缺点？**

A: 优点：每个时步同时利用前向和后向信息，适合需要全局上下文的任务（如 NER、情感分析）；缺点：无法用于在线/实时预测（需要看完整个序列），推理时 latency 翻倍。语言模型（生成任务）不能用双向。
