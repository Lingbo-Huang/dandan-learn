# Day 3：Attention 核心机制——Q/K/V 与 Scaled Dot-Product

---

## 学习目标

- 理解 Query、Key、Value 的直觉含义与矩阵形式
- 推导 Scaled Dot-Product Attention 公式
- 理解缩放因子 √dₖ 的作用
- 掌握 Attention Mask 的用法

---

## 一、核心知识点

### 1.1 Attention 的直觉

类比**数据库查询**：

| 概念 | Attention 中的对应 | 直觉 |
|------|--------------------|------|
| Query (Q) | 当前位置"想问什么" | 查询键 |
| Key (K) | 每个位置"能回答什么" | 索引键 |
| Value (V) | 每个位置"实际提供的信息" | 存储值 |

**过程**：Q 与所有 K 计算相似度 → Softmax 得到注意力权重 → 对所有 V 加权求和。

### 1.2 Scaled Dot-Product Attention 公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

各矩阵维度（单头）：

```
Q: (seq_len_q, d_k)
K: (seq_len_k, d_k)
V: (seq_len_k, d_v)

QK^T: (seq_len_q, seq_len_k)   ← attention score 矩阵
softmax(QK^T / √d_k): (seq_len_q, seq_len_k)   ← attention weight
output: (seq_len_q, d_v)
```

### 1.3 为什么要除以 √dₖ？

**问题**：当 dₖ 较大时，QK^T 的点积值也会变大，导致 Softmax 进入饱和区（梯度趋近于 0）。

**推导**：假设 Q、K 的每个元素独立同分布，均值为 0，方差为 1。则 QK^T 中每个元素：

```
qᵢ · kᵢ = Σⱼ (qⱼ · kⱼ)   (j 从 1 到 dₖ)

E[qⱼ · kⱼ] = 0
Var[qⱼ · kⱼ] = 1

∴ Var[q · k] = dₖ，标准差 = √dₖ
```

除以 √dₖ 后，方差归一为 1，Softmax 输入分布合理，梯度流动正常。

### 1.4 Attention Mask

两种 Mask：

**1. Padding Mask**（Encoder 和 Decoder 均用）

将 padding 位置的 attention score 设为 -∞，使 Softmax 后权重为 0：

```python
score.masked_fill(padding_mask == 0, float('-inf'))
```

**2. Causal Mask（Look-ahead Mask）**（只在 Decoder Self-Attention 用）

上三角矩阵，防止位置 t 看到位置 t+1 以后的 token：

```
位置:  1  2  3  4
  1: [1, 0, 0, 0]   ← 位置1只能看自己
  2: [1, 1, 0, 0]   ← 位置2看位置1和2
  3: [1, 1, 1, 0]
  4: [1, 1, 1, 1]
```

---

## 二、逐步推导示例

以简化版为例（d_model=4, d_k=4, 序列长度=3）：

**输入序列**（已 embedding）：
```python
X = [[1, 0, 1, 0],   # "I"
     [0, 2, 0, 2],   # "love"
     [1, 1, 1, 1]]   # "cats"
```

**线性投影**（简化：直接用 X 作为 Q/K/V）：
```
Q = K = V = X   (d_k = d_v = 4)
```

**计算 attention score**：
```
QK^T =
  [1,0,1,0]   [1,0,1,1]     [2, 2, 2]
  [0,2,0,2] × [0,2,1,0]  =  [4, 8, 6]
  [1,1,1,1]   [1,0,1,1]     [2, 4, 4]
              [0,2,1,1]
```

**缩放**（√4 = 2）：
```
score / 2 = [[1,  1,  1 ],
             [2,  4,  3 ],
             [1,  2,  2 ]]
```

**Softmax 后得到 attention weights**（按行）：
```python
import torch
import torch.nn.functional as F
score = torch.tensor([[1., 1., 1.], [2., 4., 3.], [1., 2., 2.]])
weights = F.softmax(score, dim=-1)
# 约为: [[0.33, 0.33, 0.33],
#        [0.09, 0.67, 0.24],
#        [0.21, 0.42, 0.37]]
```

**加权求和 V**：
```
output = weights @ V   # shape: (3, 4)
```

---

## 三、动手练习

### 练习 1：从零实现 Scaled Dot-Product Attention

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (batch, seq_q, d_k)
        K: (batch, seq_k, d_k)
        V: (batch, seq_k, d_v)
        mask: (batch, seq_q, seq_k) or None
    Returns:
        output: (batch, seq_q, d_v)
        attn_weights: (batch, seq_q, seq_k)
    """
    d_k = Q.size(-1)
    
    # Step 1: 计算 attention score
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Step 2: 应用 mask（可选）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 3: Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Step 4: 加权求和
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

# 测试
batch_size, seq_len, d_k, d_v = 2, 5, 64, 64
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")   # (2, 5, 64)
print(f"Weights shape: {weights.shape}") # (2, 5, 5)
print(f"Weights 行和（应为1）: {weights[0].sum(dim=-1)}")
```

### 练习 2：加入 Causal Mask

```python
# 生成因果 mask（下三角为 1）
def causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len))

mask = causal_mask(5)
print("Causal Mask:\n", mask)

output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
print(f"\nMasked weights（上三角应为0）:\n{weights_masked[0].detach().numpy().round(3)}")
```

### 练习 3：可视化 Attention 权重

```python
import matplotlib.pyplot as plt

tokens = ["I", "love", "cats", "a", "lot"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.imshow(weights[0].detach().numpy(), cmap='Blues')
ax1.set_title("No Mask Attention")
ax1.set_xticks(range(5)); ax1.set_xticklabels(tokens)
ax1.set_yticks(range(5)); ax1.set_yticklabels(tokens)

ax2.imshow(weights_masked[0].detach().numpy(), cmap='Blues')
ax2.set_title("Causal Mask Attention")
ax2.set_xticks(range(5)); ax2.set_xticklabels(tokens)
ax2.set_yticks(range(5)); ax2.set_yticklabels(tokens)

plt.tight_layout()
plt.savefig("attention_weights.png", dpi=100)
plt.show()
```

---

## 四、小结

| 项目 | 内容 |
|------|------|
| 今日完成 | Q/K/V 直觉 + Scaled Dot-Product Attention 公式推导 + Mask 实现 |
| 核心认知 | Attention = 用 Q 查 K 得权重，再对 V 加权求和；√dₖ 防梯度消失 |
| 明日预告 | D4：将单头 Attention 扩展为 Multi-Head，以及位置编码的设计 |

> 💡 **思考**：Self-Attention 中 Q=K=V（来自同一输入），Cross-Attention 中 Q 和 K/V 来自不同输入。你能说出为什么 Cross-Attention 要这样设计吗？
