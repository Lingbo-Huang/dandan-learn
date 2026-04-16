# Day 5：Feed-Forward Network & 残差归一化深入

---

## 学习目标

- 理解 FFN 在 Transformer 中的作用与结构
- 区分 Pre-LayerNorm 和 Post-LayerNorm 的区别与影响
- 理解 Dropout 在 Transformer 中的应用位置
- 完成一个完整 Encoder Block 的代码实现

---

## 一、核心知识点

### 1.1 Feed-Forward Network（FFN）

每个 Transformer Block 中的 FFN 是一个**逐位置**（position-wise）的两层 MLP：

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

关键特点：
- **逐位置独立**：每个 token 的 FFN 计算相互独立（参数共享，输入独立）
- **升维再降维**：d_model → 4×d_model → d_model（原论文）
- **提供非线性变换**：Attention 本身是线性的（加权求和），FFN 引入非线性

```
d_model=512  →  W₁  →  d_ff=2048  →  ReLU  →  W₂  →  d_model=512
```

### 1.2 FFN 的直觉：记忆存储

Transformer 中，Attention 负责**信息路由**（决定从哪里取信息），FFN 负责**知识存储**（类似记忆库）。研究表明，大模型的大量"事实知识"存储在 FFN 的权重中。

现代变体：
- **GELU** 替代 ReLU（更平滑的激活）
- **SwiGLU**（LLaMA 系列使用）：$\text{SwiGLU}(x) = (xW_1) \odot \text{SiLU}(xW_3)$

### 1.3 Post-LN vs Pre-LN

**Post-LN**（原论文设计）：
```
output = LayerNorm(x + Sublayer(x))
```
- 训练初期梯度不稳定，需要仔细的学习率 warmup

**Pre-LN**（现代主流）：
```
output = x + Sublayer(LayerNorm(x))
```
- 训练更稳定，梯度直接通过残差连接流向输入
- BERT、GPT-2 后的大多数模型采用此设计

**对比图示**：
```
Post-LN:   x → [Sublayer] → + → [LayerNorm] → 输出
                              ↑
                              x（残差）

Pre-LN:    x → [LayerNorm] → [Sublayer] → + → 输出
                                           ↑
                                           x（残差）
```

### 1.4 Dropout 在 Transformer 中的位置

原论文在以下位置使用 Dropout（训练时）：
1. Attention weights 上（softmax 之后，矩阵乘法之前）
2. 每个 Add & Norm 的残差加法前（Sublayer 输出上）
3. Embedding + Positional Encoding 之后

```python
# 标准 Transformer 的 Dropout 位置
attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
output = x + F.dropout(sublayer_output, p=0.1, training=self.training)
```

---

## 二、推导：FFN 升维的意义

**为什么 FFN 要先升维到 4×d_model？**

假设模型要记忆 N 个知识条目，每个用 d_model 维向量表示。通过升维：
- W₁ 的每一列可以视为一个"知识探针"
- ReLU 激活相当于"找到匹配的知识"
- W₂ 将匹配的知识写回 d_model 维

4×d_model 意味着比 Attention 层有 4 倍的"存储容量"，这是经验性的设计选择。

---

## 三、动手练习

### 练习 1：实现 FFN 模块

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.linear2(self.dropout(self.act(self.linear1(x))))

# 测试
ffn = FeedForward(d_model=512, d_ff=2048, activation='gelu')
x = torch.randn(2, 10, 512)
out = ffn(x)
print(f"FFN 输入: {x.shape}, 输出: {out.shape}")  # 应相同
```

### 练习 2：实现完整 Encoder Block（Pre-LN 版本）

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 假设 MultiHeadAttention 和 FeedForward 已在前几天实现
        from day04 import MultiHeadAttention  # 或直接粘贴过来
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-LN: 先归一化，再子层，再残差
        # Self-Attention 子层
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # FFN 子层
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x

# 用 PyTorch 内置层（更简洁）
class EncoderBlockBuiltin(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,  # 使用 (batch, seq, feature) 格式
            norm_first=True    # Pre-LN
        )
        self.layer = encoder_layer
    
    def forward(self, x, src_key_padding_mask=None):
        return self.layer(x, src_key_padding_mask=src_key_padding_mask)

# 测试
block = EncoderBlockBuiltin()
x = torch.randn(2, 10, 512)
out = block(x)
print(f"Encoder Block 输入: {x.shape}, 输出: {out.shape}")
```

### 练习 3：对比 ReLU vs GELU 激活函数

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4, 4, 200)
relu = np.maximum(0, x)
gelu = x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

plt.figure(figsize=(8, 4))
plt.plot(x, relu, label='ReLU', linewidth=2)
plt.plot(x, gelu, label='GELU', linewidth=2, linestyle='--')
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.legend()
plt.title("ReLU vs GELU")
plt.grid(True, alpha=0.3)
plt.savefig("activation_comparison.png", dpi=100)
plt.show()
```

### 练习 4：思考题

- 为什么 FFN 对不同位置使用**相同的参数**，但**独立计算**？这和 CNN 的卷积核有什么相似之处？
- Pre-LN 相比 Post-LN 训练更稳定的直觉原因是什么？（提示：梯度路径）

---

## 四、小结

| 项目 | 内容 |
|------|------|
| 今日完成 | FFN 结构 + Pre/Post-LN 对比 + Dropout 位置 + 完整 Encoder Block |
| 核心认知 | FFN = 知识存储；Pre-LN = 训练稳定；Dropout = 防过拟合 |
| 明日预告 | D6：对比 Encoder-only / Decoder-only / Encoder-Decoder 三大架构变体 |

> 💡 **现代趋势**：LLaMA 等主流 LLM 使用 Pre-RMSNorm（比 LayerNorm 更高效）+ SwiGLU 激活，是 FFN 设计的现代演进。
