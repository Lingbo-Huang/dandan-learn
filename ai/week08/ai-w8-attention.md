---
layout: default
title: "Attention 机制入门"
render_with_liquid: false
---

# Attention 机制入门

## 动机：瓶颈问题的解决

Seq2Seq 将所有信息压缩到单一上下文向量，翻译长句时信息损失严重。

**Attention**：Decoder 每一步都能"回顾"Encoder 的所有隐藏状态，动态决定关注哪些位置。

## Bahdanau Attention（加法注意力）

解码第 $t$ 步时，计算与 Encoder 每个位置的相关性：

**注意力分数**：

$$e_{t,s} = a(h_{t-1}^{\text{dec}}, h_s^{\text{enc}}) = v^T \tanh(W_1 h_{t-1}^{\text{dec}} + W_2 h_s^{\text{enc}})$$

**注意力权重**（Softmax 归一化）：

$$\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{s'} \exp(e_{t,s'})}$$

**上下文向量**（加权求和）：

$$c_t = \sum_s \alpha_{t,s} h_s^{\text{enc}}$$

Decoder 使用 $c_t$ 代替固定的 $c$，每步都有"新鲜"上下文。

## Luong Attention（点积注意力）

更简单的注意力分数：

$$e_{t,s} = h_t^{\text{dec}} \cdot h_s^{\text{enc}} \quad \text{（点积形式）}$$

或缩放点积（Scaled Dot-Product）：

$$e_{t,s} = \frac{h_t^{\text{dec}} \cdot h_s^{\text{enc}}}{\sqrt{d_k}}$$

$\sqrt{d_k}$ 缩放：防止维度大时点积过大导致 Softmax 进入饱和区（梯度消失）。

## Self-Attention：Transformer 的核心

每个位置同时作为 Query、Key、Value 的来源，捕捉序列内部的依赖。

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $Q = XW^Q$（Query：当前位置"想要什么"）
- $K = XW^K$（Key：每个位置"有什么"）
- $V = XW^V$（Value：每个位置"提供什么"）

**Multi-Head Attention**：并行运行 $h$ 个注意力头，捕捉不同类型的依赖：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ===== 带 Attention 的 Seq2Seq =====
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
    
    def forward(self, query, keys):
        """
        query: (B, H) — Decoder 当前隐藏状态
        keys: (B, T, H) — Encoder 所有隐藏状态
        """
        query = query.unsqueeze(1)  # (B, 1, H)
        scores = self.V(torch.tanh(self.W1(query) + self.W2(keys)))  # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        context = (weights * keys).sum(dim=1)  # (B, H)
        return context, weights.squeeze(-1)  # (B, H), (B, T)


# ===== Scaled Dot-Product Attention =====
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (B, h, T_q, d_k)
    K: (B, h, T_k, d_k)
    V: (B, h, T_k, d_v)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (B, h, T_q, T_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)  # (B, h, T_q, d_v)
    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        
        # 线性变换并分头
        Q = self.W_Q(Q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        x, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        x = x.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.W_O(x)


# ===== 注意力可视化 =====
def visualize_attention(weights, src_tokens, trg_tokens, title="注意力热力图"):
    """
    weights: (T_trg, T_src)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(weights, aspect='auto', cmap='Blues')
    
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha='right')
    ax.set_yticks(range(len(trg_tokens)))
    ax.set_yticklabels(trg_tokens)
    
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Encoder 输入")
    ax.set_ylabel("Decoder 输出")
    ax.set_title(title)
    plt.tight_layout()

# 模拟翻译注意力
src = ["<SOS>", "机器", "学习", "很", "有趣", "<EOS>"]
trg = ["<SOS>", "Machine", "learning", "is", "interesting", "<EOS>"]

# 理想的注意力（对角线 + 语序调整）
ideal_attn = np.array([
    [0.9, 0.05, 0.01, 0.01, 0.01, 0.02],  # <SOS>→<SOS>
    [0.1, 0.75, 0.10, 0.02, 0.02, 0.01],  # <SOS>→Machine
    [0.05, 0.10, 0.75, 0.05, 0.03, 0.02], # Machine→learning
    [0.02, 0.05, 0.05, 0.80, 0.05, 0.03], # learning→is
    [0.01, 0.02, 0.05, 0.05, 0.80, 0.07], # is→interesting
    [0.01, 0.01, 0.02, 0.02, 0.05, 0.89], # interesting→<EOS>
])
visualize_attention(ideal_attn, src, trg, "模拟翻译注意力权重")

# ===== Self-Attention 测试 =====
d_model, n_heads, seq_len = 64, 4, 10
mha = MultiHeadAttention(d_model, n_heads)
x = torch.randn(2, seq_len, d_model)

# Self-attention: Q=K=V=x
out = mha(x, x, x)
print(f"Multi-Head Attention 输出形状: {out.shape}")  # (2, 10, 64)
print(f"注意力权重形状: {mha.attn_weights.shape}")   # (2, 4, 10, 10)

# 因果 mask（用于 Decoder，防止看到未来）
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
out_masked = mha(x, x, x, mask=causal_mask)
print(f"因果 Masked Attention 输出: {out_masked.shape}")
```

## 位置编码（Positional Encoding）

Self-Attention 本身是位置无关的（交换输入顺序结果不变）。需要显式注入位置信息。

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 可视化位置编码
pe = PositionalEncoding(64)
x = torch.zeros(1, 100, 64)
pe_out = pe(x)

plt.figure(figsize=(10, 5))
plt.imshow(pe_out[0].detach().numpy().T, aspect='auto', cmap='RdBu_r')
plt.colorbar()
plt.xlabel("序列位置")
plt.ylabel("维度")
plt.title("位置编码矩阵（正弦余弦交替）")
plt.tight_layout()
```

## 面试要点

**Q: Q、K、V 分别代表什么含义？**

A: 借鉴数据库查询：Q（Query）= "我要查什么"；K（Key）= "每个条目的索引"；V（Value）= "每个条目的内容"。注意力通过 $QK^T$ 计算 Query 与每个 Key 的匹配分数，Softmax 归一化后对 V 加权求和，输出"检索到的信息"。

**Q: 缩放因子 $1/\sqrt{d_k}$ 为什么重要？**

A: 当 $d_k$ 很大时，点积 $QK^T$ 方差正比于 $d_k$，值很大导致 Softmax 进入饱和区（梯度接近 0，类似 Sigmoid 的梯度消失）。除以 $\sqrt{d_k}$ 将方差归一化到 1，Softmax 梯度正常。

**Q: Attention 和 RNN 的本质区别？**

A: RNN 是序列性的（$h_t$ 依赖 $h_{t-1}$），无法并行，长距离依赖靠梯度传播（弱）；Attention 是并行的，任意两位置直接交互（路径长度为 1），捕捉长距离依赖能力强。这是 Transformer 取代 RNN 的根本原因。
