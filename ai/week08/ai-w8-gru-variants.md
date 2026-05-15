---
layout: default
title: "GRU 与序列模型变体"
render_with_liquid: false
---

# GRU 与序列模型变体

## GRU 完整推导

### 重置门

$$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$$

控制"有多少历史信息参与候选状态计算"。$r_t \to 0$：完全忽略历史；$r_t \to 1$：完全使用历史。

### 更新门

$$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$

控制"新信息和历史信息的混合比例"。$z_t \to 0$：保留历史；$z_t \to 1$：使用新候选状态。

### 候选隐藏状态

$$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$$

### 最终更新

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

这是一个**凸组合**：以 $z_t$ 为权重，在历史状态 $h_{t-1}$ 和候选状态 $\tilde{h}_t$ 之间插值。

### GRU 和 LSTM 的关系

| | LSTM | GRU |
|--|------|-----|
| 状态 | $c_t$（长期）+ $h_t$（输出） | $h_t$（单一） |
| 门 | 输入门、遗忘门、输出门 | 重置门、更新门 |
| 参数量 | $4 \times (d_h(d_h+d_x) + d_h)$ | $3 \times (d_h(d_h+d_x) + d_h)$ |
| 效果 | 长序列更稳定 | 小数据集、短序列可能更好 |

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ===== 手写 GRU Cell =====
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 重置门、更新门和候选状态的权重合并
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.W_r(combined))  # 重置门
        z = torch.sigmoid(self.W_z(combined))  # 更新门
        
        combined_r = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_r))  # 候选状态
        
        h_new = (1 - z) * h + z * h_tilde  # 凸组合更新
        return h_new

# 与 PyTorch 内置比较
gru_manual = GRUCell(input_size=4, hidden_size=8)
gru_builtin = nn.GRUCell(input_size=4, hidden_size=8)

x = torch.randn(2, 4)
h = torch.randn(2, 8)
out_manual = gru_manual(x, h)
print(f"手写 GRU Cell 输出形状: {out_manual.shape}")

# ===== 双向 RNN（Bidirectional）=====
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='lstm'):
        super().__init__()
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size,
                                batch_first=True, bidirectional=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size,
                               batch_first=True, bidirectional=True)
    
    def forward(self, x):
        # out: (B, T, 2H) — 前向和后向拼接
        out, _ = self.rnn(x)
        # 前向：out[:, :, :H]，后向：out[:, :, H:]
        forward_out = out[:, :, :out.size(-1)//2]
        backward_out = out[:, :, out.size(-1)//2:]
        return out, forward_out, backward_out

birnn = BiRNN(10, 32, rnn_type='gru')
x = torch.randn(4, 15, 10)  # (batch, seq_len, input_size)
full_out, fwd, bwd = birnn(x)
print(f"双向 GRU 输出: {full_out.shape}")  # (4, 15, 64)
print(f"前向: {fwd.shape}, 后向: {bwd.shape}")

# ===== 堆叠 RNN（多层）=====
stacked_lstm = nn.LSTM(
    input_size=32, hidden_size=64,
    num_layers=3,              # 3 层堆叠
    batch_first=True,
    dropout=0.2,               # 层间 Dropout
    bidirectional=True
)

x = torch.randn(4, 20, 32)
out, (h_n, c_n) = stacked_lstm(x)
print(f"堆叠 BiLSTM 输出: {out.shape}")     # (4, 20, 128)
print(f"h_n 形状: {h_n.shape}")             # (6, 4, 64) = 2*3层 × batch × hidden

# ===== 参数量对比：LSTM vs GRU =====
input_size, hidden_size = 64, 256
lstm_params = 4 * ((hidden_size + input_size) * hidden_size + hidden_size)
gru_params = 3 * ((hidden_size + input_size) * hidden_size + hidden_size)
print(f"\nLSTM 参数量: {lstm_params:,}")
print(f"GRU 参数量:  {gru_params:,}")
print(f"GRU/LSTM 参数比: {gru_params/lstm_params:.2f}x")

# ===== 长程依赖实验：记忆数字位置 =====
# 任务：序列第一个数字决定标签，其余为噪声
# 测试各模型对不同序列长度的记忆能力

def generate_memory_task(n_samples=500, seq_len=20, n_classes=4):
    X = np.random.randint(0, n_classes, (n_samples, seq_len))
    y = X[:, 0]  # 标签只取第一个元素
    return X, y

results = {}
for seq_len in [10, 20, 50, 100]:
    X, y = generate_memory_task(seq_len=seq_len)
    X_t = torch.LongTensor(X)
    y_t = torch.LongTensor(y)
    
    model_scores = {}
    for rnn_type in ['lstm', 'gru', 'rnn']:
        class MemoryNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(4, 16)
                if rnn_type == 'lstm':
                    self.rnn = nn.LSTM(16, 32, batch_first=True)
                elif rnn_type == 'gru':
                    self.rnn = nn.GRU(16, 32, batch_first=True)
                else:
                    self.rnn = nn.RNN(16, 32, batch_first=True)
                self.fc = nn.Linear(32, 4)
            
            def forward(self, x):
                emb = self.emb(x)
                if rnn_type == 'lstm':
                    out, (h, _) = self.rnn(emb)
                else:
                    out, h = self.rnn(emb)
                return self.fc(out[:, -1, :])  # 只用最后一步
        
        net = MemoryNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        
        for epoch in range(100):
            opt.zero_grad()
            loss = crit(net(X_t[:400]), y_t[:400])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        
        net.eval()
        with torch.no_grad():
            acc = (net(X_t[400:]).argmax(1) == y_t[400:]).float().mean().item()
        model_scores[rnn_type.upper()] = acc
    
    results[seq_len] = model_scores
    print(f"seq_len={seq_len:3d}: " + 
          " | ".join(f"{k}={v:.3f}" for k, v in model_scores.items()))

# 可视化
seq_lens = sorted(results.keys())
fig, ax = plt.subplots(figsize=(8, 5))
for model_name in ['RNN', 'GRU', 'LSTM']:
    accs = [results[l][model_name] for l in seq_lens]
    ax.plot(seq_lens, accs, 'o-', linewidth=2, label=model_name)
ax.set_xlabel('序列长度')
ax.set_ylabel('准确率（记住第一个元素）')
ax.set_title('长程依赖能力：RNN vs GRU vs LSTM')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

## Sequence Modeling 最佳实践

```python
# ===== 实用技巧汇总 =====

# 1. Pack/Pad（处理变长序列）
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def encode_with_packing(rnn, embedded, lengths):
    """正确处理变长序列"""
    packed = pack_padded_sequence(embedded, lengths.cpu(),
                                   batch_first=True, enforce_sorted=False)
    packed_out, hidden = rnn(packed)
    out, _ = pad_packed_sequence(packed_out, batch_first=True)
    return out, hidden

# 2. 梯度裁剪（RNN 必备）
def train_step(model, x, y, optimizer, criterion, max_norm=1.0):
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    return loss.item()

# 3. 提取最后有效隐藏状态（不是 padding 的最后一步）
def get_last_valid_hidden(rnn_out, lengths):
    """从 padded output 中提取每个序列实际最后一步"""
    batch_size = rnn_out.size(0)
    last_idx = (lengths - 1).clamp(min=0)  # (B,)
    last_hidden = rnn_out[torch.arange(batch_size), last_idx, :]  # (B, H)
    return last_hidden
```

## 面试要点

**Q: GRU 更新门 $z_t \to 0$ 时会发生什么？**

A: $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$，当 $z_t \to 0$ 时，$h_t \approx h_{t-1}$，即完全保留历史状态，不接受新信息。这相当于 LSTM 中遗忘门全为 1、输入门全为 0 的情况。

**Q: 双向 LSTM 为什么不能用于实时（在线）场景？**

A: 双向 LSTM 需要看完整个序列才能计算反向状态，对于实时语音识别、股票预测等需要即时输出的场景无法使用。解决方案：① 单向 LSTM；② 只对固定窗口做双向（Chunk-based BiLSTM）；③ 用 Causal Transformer。
