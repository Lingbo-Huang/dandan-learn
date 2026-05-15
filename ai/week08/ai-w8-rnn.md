---
layout: default
title: "RNN 基础与时序反向传播"
render_with_liquid: false
---

# RNN 基础与时序反向传播

## 为什么需要 RNN？

MLP 和 CNN 假设输入是固定维度且相互独立。序列数据（文本、时序、语音）不满足：
- 长度可变
- 当前输出依赖历史

RNN 通过**隐藏状态**记录历史信息，实现对序列的建模。

## RNN 数学定义

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$y_t = W_{hy} h_t + b_y$$

- $h_t$：时刻 $t$ 的隐藏状态（记忆）
- $x_t$：时刻 $t$ 的输入
- 关键：**权重共享**（同一组 $W$ 在所有时刻使用）

## 前向传播

```
x_1 → h_1 → y_1
       ↓
x_2 → h_2 → y_2
       ↓
x_3 → h_3 → y_3
```

参数总量：$W_{hh} \in \mathbb{R}^{d_h \times d_h}$，$W_{xh} \in \mathbb{R}^{d_h \times d_x}$，与序列长度无关！

## BPTT（时序反向传播）

损失关于 $W_{hh}$ 的梯度：

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L_t}{\partial W_{hh}}$$

每个时刻的梯度需要通过时间链传回：

$$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^t \frac{\partial L_t}{\partial h_t} \left(\prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}}\right) \frac{\partial h_k}{\partial W_{hh}}$$

关键项：

$$\prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}} = \prod_{j=k+1}^t W_{hh}^T \cdot \text{diag}(\tanh'(z_j))$$

当 $t-k$ 很大时，$\|W_{hh}\|$ 的幂次导致：
- $\|W_{hh}\| > 1$：梯度**爆炸**
- $\|W_{hh}\| < 1$：梯度**消失**

## Python 实现

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ===== 手写 RNN =====
class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier 初始化
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(2/(input_size+hidden_size))
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2/(hidden_size+hidden_size))
        self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(2/(hidden_size+output_size))
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs, h_prev):
        """
        inputs: list of (input_size, 1) 向量
        h_prev: (hidden_size, 1) 初始隐藏状态
        """
        self.xs, self.hs, self.ys = {}, {}, {}
        self.hs[-1] = h_prev.copy()
        
        for t, x in enumerate(inputs):
            self.xs[t] = x
            z = self.Wxh @ x + self.Whh @ self.hs[t-1] + self.bh
            self.hs[t] = np.tanh(z)
            self.ys[t] = self.Why @ self.hs[t] + self.by
        
        return self.ys, self.hs
    
    def backward(self, dy_list, learning_rate=0.001):
        T = len(dy_list)
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(self.hs[0])
        
        for t in reversed(range(T)):
            dy = dy_list[t]
            dWhy += dy @ self.hs[t].T
            dby += dy
            
            dh = self.Why.T @ dy + dh_next
            dh_raw = (1 - self.hs[t]**2) * dh  # tanh 导数
            
            dbh += dh_raw
            dWxh += dh_raw @ self.xs[t].T
            dWhh += dh_raw @ self.hs[t-1].T
            dh_next = self.Whh.T @ dh_raw
            
            # 梯度裁剪
            for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
                np.clip(dparam, -5, 5, out=dparam)
        
        for param, dparam in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                   [dWxh, dWhh, dWhy, dbh, dby]):
            param -= learning_rate * dparam

# ===== 梯度消失实验 =====
def analyze_gradient_vanishing(seq_len=50):
    torch.manual_seed(42)
    rnn = nn.RNNCell(1, 32)
    
    x = torch.randn(seq_len, 1)
    h = torch.zeros(1, 32)
    
    # 跟踪每一步的梯度
    grad_norms = []
    h_seq = [h]
    
    for t in range(seq_len):
        h = rnn(x[t:t+1], h)
        h_seq.append(h)
    
    # 计算最终 h 对每一步 h 的梯度
    loss = h_seq[-1].sum()
    loss.backward(retain_graph=True)
    
    # 通过 Jacobian 分析
    rnn2 = nn.RNN(1, 32, batch_first=True)
    x2 = torch.randn(1, seq_len, 1, requires_grad=True)
    out, _ = rnn2(x2)
    
    # 每个时步的梯度范数
    out[:, -1, :].sum().backward()
    
    print(f"RNN 输入梯度范数: {x2.grad.norm():.6f}")
    
    # LSTM 对比
    lstm = nn.LSTM(1, 32, batch_first=True)
    x3 = torch.randn(1, seq_len, 1, requires_grad=True)
    out3, _ = lstm(x3)
    out3[:, -1, :].sum().backward()
    print(f"LSTM 输入梯度范数: {x3.grad.norm():.6f}")

analyze_gradient_vanishing(seq_len=50)

# ===== PyTorch RNN 实战 =====
# 字符级语言模型（简单示例）
text = "hello world, this is a simple character level language model example"
chars = sorted(set(text))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(chars)

def text_to_tensor(t):
    return torch.tensor([char2idx[c] for c in t], dtype=torch.long)

rnn = nn.RNN(vocab_size, 128, num_layers=2, batch_first=True, dropout=0.3)
linear = nn.Linear(128, vocab_size)

x_data = text_to_tensor(text[:-1])
y_data = text_to_tensor(text[1:])
x_onehot = torch.zeros(1, len(x_data), vocab_size)
x_onehot[0, range(len(x_data)), x_data] = 1.0

optimizer = torch.optim.Adam(list(rnn.parameters()) + list(linear.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()

losses = []
for epoch in range(200):
    optimizer.zero_grad()
    out, _ = rnn(x_onehot)
    logits = linear(out.squeeze(0))
    loss = criterion(logits, y_data)
    loss.backward()
    nn.utils.clip_grad_norm_(list(rnn.parameters()), max_norm=5.0)
    optimizer.step()
    if epoch % 50 == 0:
        losses.append(loss.item())
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

plt.figure(figsize=(6, 3))
plt.plot(losses, 'o-')
plt.xlabel("Epoch (×50)")
plt.ylabel("Loss")
plt.title("字符级 RNN 训练损失")
plt.grid(True)
plt.tight_layout()
```

## 面试要点

**Q: RNN 的梯度消失和 DNN 的有什么区别？**

A: DNN 中梯度消失是跨层的（不同的 $W$），理论上可以用好的初始化/BN/ResNet 缓解；RNN 中梯度消失是跨时间步的（同一个 $W_{hh}$ 反复相乘），长序列时无法避免——这就是为什么需要 LSTM 这样的门控机制。

**Q: Truncated BPTT 是什么？**

A: 完整 BPTT 需要存所有时步的激活，内存$O(T)$。Truncated BPTT 每 $k$ 步做一次反向传播，只传 $k$ 步，用于训练超长序列。代价是无法学习跨越 $k$ 步的长期依赖。
