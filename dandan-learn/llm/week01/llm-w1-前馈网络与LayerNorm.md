# Day 4：Multi-Head Attention & 位置编码

---

## 学习目标

- 理解 Multi-Head Attention 的设计动机与实现方式
- 掌握正弦/余弦位置编码的公式与直觉
- 了解相对位置编码（RoPE）的基本思路
- 实现 Multi-Head Attention 模块

---

## 一、核心知识点

### 1.1 为什么需要多头？

单头 Attention 每次只能学到一种"相关性模式"。多头并行让模型**同时关注不同类型的关系**：

- 第 1 头：关注语法依存（主语-谓语）
- 第 2 头：关注语义相似（同义词）
- 第 3 头：关注位置关系（相邻词）
- ……

类比：用多个不同角度的摄像头同时拍摄，再拼接信息。

### 1.2 Multi-Head Attention 公式

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

其中：
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$，$d_k = d_{model} / h$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$（输出投影）

**计算过程**：
```
输入 X: (batch, seq_len, d_model)
  ↓ 分别线性投影
Q_full: (batch, seq_len, d_model)  → reshape →  (batch, h, seq_len, d_k)
K_full: (batch, seq_len, d_model)  → reshape →  (batch, h, seq_len, d_k)
V_full: (batch, seq_len, d_model)  → reshape →  (batch, h, seq_len, d_v)
  ↓ 每个头独立做 Attention
heads: (batch, h, seq_len, d_v)
  ↓ reshape + 拼接
(batch, seq_len, h*d_v) = (batch, seq_len, d_model)
  ↓ 输出线性投影 W^O
output: (batch, seq_len, d_model)
```

### 1.3 正弦/余弦位置编码

Transformer 本身不感知位置（Attention 是置换不变的），需要**显式注入位置信息**。

**公式**（偶数维用 sin，奇数维用 cos）：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中 `pos` 是位置序号，`i` 是维度序号。

**设计优点**：
1. 每个位置有唯一编码
2. 不同频率：低维变化快（局部位置），高维变化慢（全局位置）
3. 可外推到训练时未见过的序列长度

### 1.4 可学习位置编码 vs 固定位置编码

| 方式 | 代表模型 | 特点 |
|------|---------|------|
| 固定正弦/余弦 | 原始 Transformer | 无参数，可外推 |
| 可学习绝对位置 | BERT, GPT-2 | 有参数，灵活，受限于训练长度 |
| 相对位置（RoPE） | LLaMA, Qwen | 通过旋转矩阵编码相对距离，外推能力强 |

---

## 二、示例：位置编码可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(max_seq_len, d_model):
    PE = np.zeros((max_seq_len, d_model))
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i]   = np.sin(pos / (10000 ** (2*i / d_model)))
            PE[pos, i+1] = np.cos(pos / (10000 ** (2*i / d_model)))
    return PE

PE = get_positional_encoding(50, 128)

plt.figure(figsize=(12, 5))
plt.imshow(PE, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
plt.colorbar()
plt.title("Positional Encoding (50 positions, 128 dims)")
plt.xlabel("Embedding Dimension")
plt.ylabel("Position")
plt.savefig("positional_encoding.png", dpi=100)
plt.show()
```

**观察**：
- 每行（每个位置）的编码都不同
- 低维（左侧）变化快；高维（右侧）变化慢
- 相邻位置的编码相似，体现了局部连续性

---

## 三、动手练习

### 练习 1：实现 Multi-Head Attention

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层（Q/K/V 和输出）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def forward(self, Q_in, K_in, V_in, mask=None):
        batch = Q_in.size(0)
        
        # 线性投影 + 分头
        Q = self.split_heads(self.W_q(Q_in))  # (batch, heads, seq, d_k)
        K = self.split_heads(self.W_k(K_in))
        V = self.split_heads(self.W_v(V_in))
        
        # Scaled Dot-Product Attention（批量多头）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 加权求和
        context = torch.matmul(attn_weights, V)  # (batch, heads, seq, d_k)
        
        # 拼接多头 + 输出投影
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights

# 测试
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # (batch=2, seq=10, d_model=512)
out, weights = mha(x, x, x)
print(f"输入 shape: {x.shape}")
print(f"输出 shape: {out.shape}")       # 应为 (2, 10, 512)
print(f"注意力权重 shape: {weights.shape}")  # 应为 (2, 8, 10, 10)
```

### 练习 2：实现正弦位置编码

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        PE = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        PE[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        PE[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        PE = PE.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('PE', PE)  # 不是参数，但随模型保存
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.PE[:, :x.size(1), :]
        return self.dropout(x)

# 测试
pe = PositionalEncoding(d_model=512)
x = torch.randn(2, 10, 512)
out = pe(x)
print(f"位置编码后 shape: {out.shape}")  # 应为 (2, 10, 512)
```

### 练习 3：思考题

- 如果 num_heads=8，d_model=512，那么每个头的 d_k 是多少？总计算量与单头相比如何？
- 为什么多头 Attention 拼接后还需要一个 W^O 输出投影层？

---

## 四、小结

| 项目 | 内容 |
|------|------|
| 今日完成 | Multi-Head Attention 原理 + 实现 + 正弦位置编码 |
| 核心认知 | 多头 = 多视角并行理解；位置编码 = 注入位置信息 |
| 明日预告 | D5：Feed-Forward Network、Pre/Post-LayerNorm、Dropout |

> 💡 **关键数字**：原论文用 h=8，d_model=512，d_k=d_v=64，总共 8×64=512，计算量与单头基本持平但表达能力更强。
