# D3：多头注意力机制实现

> **Week 2 · Day 3** | 大模型学习路线

---

## 一、为什么需要多头注意力？

单头注意力有一个限制：它只有**一组** Q/K/V 投影，意味着每次只能"聚焦"在一种关系上。

考虑这个句子：

> "Mary gave John a book because **she** thought **he** would enjoy **it**."

要理解这个句子，模型需要同时追踪：
- **语法依赖**：`gave` 的主语是 `Mary`
- **代词指代**：`she → Mary`，`he → John`，`it → book`
- **语义关系**：`enjoy` 的宾语是 `it`（书）

单头注意力在处理 `she` 时，只能通过一组注意力权重"看"整个句子，很难同时捕捉所有这些关系。

**多头注意力（Multi-Head Attention, MHA）** 的思路：用 $h$ 个独立的注意力头，每个头学习不同的关注模式（语法、语义、指代等），最后拼接起来。

---

## 二、形式化定义

### 2.1 核心公式

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

其中每个头：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

投影矩阵：
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

### 2.2 维度设置

Transformer 原论文中：
- $d_{\text{model}} = 512$，$h = 8$
- $d_k = d_v = d_{\text{model}} / h = 64$

这样总参数量与单头（$d_k = d_{\text{model}}$）相当，但有 $h$ 个不同的关注视角。

**参数量计算**：
$$\underbrace{3 \cdot h \cdot d_{\text{model}} \cdot d_k}_{Q/K/V \text{ 投影}} + \underbrace{h \cdot d_v \cdot d_{\text{model}}}_{O \text{ 投影}} = 4 d_{\text{model}}^2$$

与单头相同（当 $d_k = d_v = d_{\text{model}} / h$ 时）。

---

## 三、并行计算的技巧

朴素实现是逐头计算：

```python
heads = []
for i in range(h):
    head_i = attention(x @ W_Q[i], x @ W_K[i], x @ W_V[i])
    heads.append(head_i)
output = concat(heads) @ W_O
```

但这效率低（h 个串行循环）。高效实现是**一次性投影，然后 reshape**：

```
(B, T, d_model) 
-> Linear -> (B, T, h*d_k) 
-> reshape -> (B, T, h, d_k) 
-> transpose -> (B, h, T, d_k)  ← 批量并行计算注意力
```

---

## 四、完整 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 - 完整实现
    
    公式: MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
    其中: head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        self.d_v = d_model // num_heads
        
        # 合并所有头的投影矩阵（高效）
        # 等价于 h 个独立的 W_i^Q，但用一个大矩阵实现
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)  # h * d_k = d_model
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)
        
        # 输出投影
        self.W_O = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout（应用于注意力权重）
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # 初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Xavier 初始化"""
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)
        if self.W_Q.bias is not None:
            nn.init.zeros_(self.W_Q.bias)
            nn.init.zeros_(self.W_K.bias)
            nn.init.zeros_(self.W_V.bias)
            nn.init.zeros_(self.W_O.bias)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将投影后的张量拆分为多头
        
        (B, T, d_model) -> (B, num_heads, T, d_k)
        """
        B, T, _ = x.shape
        # Reshape: (B, T, h*d_k) -> (B, T, h, d_k)
        x = x.view(B, T, self.num_heads, self.d_k)
        # Transpose: (B, T, h, d_k) -> (B, h, T, d_k)
        return x.transpose(1, 2)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多头结果合并
        
        (B, num_heads, T, d_v) -> (B, T, d_model)
        """
        B, h, T, d_v = x.shape
        # Transpose: (B, h, T, d_v) -> (B, T, h, d_v)
        x = x.transpose(1, 2)
        # Reshape: (B, T, h, d_v) -> (B, T, h*d_v)
        return x.contiguous().view(B, T, h * d_v)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (B, T_q, d_model) - 查询序列
            key:   (B, T_k, d_model) - 键序列
            value: (B, T_k, d_model) - 值序列
            key_padding_mask: (B, T_k) - True 表示 padding 位置
            attn_mask: (T_q, T_k) 或 (B, T_q, T_k) - True 表示遮盖位置
            return_attn_weights: 是否返回注意力权重
        
        Returns:
            output: (B, T_q, d_model)
            attn_weights: (B, num_heads, T_q, T_k) 或 None
        """
        B, T_q, _ = query.shape
        T_k = key.size(1)
        
        # ===== Step 1: 线性投影 =====
        # (B, T, d_model) -> (B, T, d_model)
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)
        
        # ===== Step 2: 拆分多头 =====
        # (B, T, d_model) -> (B, num_heads, T, d_k)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # ===== Step 3: 计算注意力分数 =====
        # (B, h, T_q, d_k) @ (B, h, d_k, T_k) -> (B, h, T_q, T_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # ===== Step 4: 应用 Mask =====
        # attn_mask: 因果遮盖或自定义遮盖
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_k)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, T_q, T_k)
            scores = scores.masked_fill(attn_mask.bool(), float('-inf'))
        
        # key_padding_mask: 遮盖 padding 位置
        if key_padding_mask is not None:
            # (B, T_k) -> (B, 1, 1, T_k)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # ===== Step 5: Softmax + Dropout =====
        attn_weights = F.softmax(scores, dim=-1)  # (B, h, T_q, T_k)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        # ===== Step 6: 加权聚合 Value =====
        # (B, h, T_q, T_k) @ (B, h, T_k, d_v) -> (B, h, T_q, d_v)
        context = torch.matmul(attn_weights, V)
        
        # ===== Step 7: 合并多头 + 输出投影 =====
        # (B, h, T_q, d_v) -> (B, T_q, d_model)
        context = self._merge_heads(context)
        
        # (B, T_q, d_model) -> (B, T_q, d_model)
        output = self.W_O(context)
        
        if return_attn_weights:
            return output, attn_weights
        return output, None


# ==================== 测试与验证 ====================

def test_multi_head_attention():
    torch.manual_seed(42)
    
    B, T, d_model, h = 2, 10, 256, 8
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=h, dropout=0.0)
    
    # 自注意力（Q = K = V）
    x = torch.randn(B, T, d_model)
    out, weights = mha(x, x, x, return_attn_weights=True)
    
    print("=" * 50)
    print("多头注意力测试")
    print("=" * 50)
    print(f"输入形状:   {x.shape}")
    print(f"输出形状:   {out.shape}")
    print(f"权重形状:   {weights.shape}")  # (B, h, T_q, T_k)
    
    # 验证每个头的权重行归一化
    row_sums = weights.sum(dim=-1)
    print(f"行归一化验证: max偏差={abs(row_sums - 1).max().item():.6f}")
    
    # 测试交叉注意力（Q 和 K/V 序列长度不同）
    T_q, T_k = 5, 12
    query = torch.randn(B, T_q, d_model)
    key_value = torch.randn(B, T_k, d_model)
    out_cross, weights_cross = mha(query, key_value, key_value, return_attn_weights=True)
    print(f"\n交叉注意力输出形状: {out_cross.shape}")
    print(f"交叉注意力权重形状: {weights_cross.shape}")
    
    # 测试因果遮盖
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    out_causal, weights_causal = mha(x, x, x, attn_mask=causal_mask, return_attn_weights=True)
    
    # 验证：权重的上三角应为 0
    upper_tri = torch.triu(weights_causal[0, 0], diagonal=1)
    print(f"\n因果遮盖验证 - 上三角权重最大值: {upper_tri.max().item():.6f} (应为 0)")
    
    # 测试 padding mask
    padding_mask = torch.zeros(B, T, dtype=torch.bool)
    padding_mask[0, 7:] = True  # 第一个样本后3个位置是 padding
    out_padded, weights_padded = mha(x, x, x, key_padding_mask=padding_mask, return_attn_weights=True)
    print(f"\nPadding mask 验证 - 被遮盖位置权重: {weights_padded[0, 0, 0, 7:].sum().item():.6f} (应为 0)")
    
    return mha, x, weights

mha, x, weights = test_multi_head_attention()
```

---

## 五、可视化不同头的注意力模式

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def visualize_multihead_attention(weights: torch.Tensor, tokens: list):
    """
    可视化多头注意力权重，展示不同头学到的不同模式
    
    Args:
        weights: (num_heads, T_q, T_k)
        tokens: token 列表
    """
    num_heads = weights.shape[0]
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for h in range(num_heads):
        ax = axes[h // cols][h % cols]
        w = weights[h].detach().cpu().numpy()
        
        im = ax.imshow(w, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'Head {h+1}')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        
        # 添加数值标注
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                ax.text(j, i, f'{w[i,j]:.2f}', ha='center', va='center', fontsize=6)
    
    # 隐藏多余的子图
    for h in range(num_heads, rows * cols):
        axes[h // cols][h % cols].axis('off')
    
    plt.suptitle('多头注意力权重可视化 - 不同头关注不同模式', fontsize=12)
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()
    plt.savefig('multihead_attention.png', dpi=100, bbox_inches='tight')
    plt.show()

# 示例可视化
tokens = ["I", "love", "deep", "learning", "too"]
mha_small = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
x_small = torch.randn(1, 5, 32)
_, w_small = mha_small(x_small, x_small, x_small, return_attn_weights=True)
# visualize_multihead_attention(w_small[0], tokens)  # 取消注释以可视化
```

---

## 六、分组查询注意力（GQA）

现代大语言模型（LLaMA-2, Mistral 等）使用**分组查询注意力（Grouped-Query Attention, GQA）**，是 MHA 和 MQA 的折中：

- **MHA**：$h$ 个头，各有独立的 Q/K/V
- **MQA**（Multi-Query Attention）：$h$ 个 Q 头，但 K/V 只有 1 个头（所有 Q 头共享同一 K/V）
- **GQA**：$h$ 个 Q 头，分成 $g$ 组，每组共享一套 K/V（$g$ 个 K/V 头）

```python
class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 GQA
    
    num_q_heads: Query 的头数
    num_kv_heads: Key/Value 的头数（< num_q_heads）
    每组 num_q_heads // num_kv_heads 个 Q 头共享一套 K/V
    """
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须是 num_kv_heads 的倍数"
        
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads  # 每个 KV 头服务的 Q 头数
        self.d_k = d_model // num_q_heads
        
        self.W_Q = nn.Linear(d_model, num_q_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)  # 更少参数
        self.W_V = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        
        Q = self.W_Q(x).view(B, T, self.num_q_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # 将 KV 头扩展到与 Q 头相同数量（通过 repeat）
        # (B, num_kv_heads, T, d_k) -> (B, num_q_heads, T, d_k)
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, V)  # (B, num_q_heads, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return self.W_O(out)

# 测试
gqa = GroupedQueryAttention(d_model=256, num_q_heads=8, num_kv_heads=2)
x = torch.randn(2, 10, 256)
out = gqa(x)
print(f"GQA 输出形状: {out.shape}")

# 参数量对比
mha_params = sum(p.numel() for p in MultiHeadAttention(256, 8).parameters())
gqa_params = sum(p.numel() for p in gqa.parameters())
print(f"MHA 参数量: {mha_params:,}")
print(f"GQA 参数量: {gqa_params:,}  (KV 头少，参数更少)")
```

---

## 七、不同注意力变体对比

| 变体 | Q 头 | K/V 头 | 适用场景 |
|------|------|--------|---------|
| MHA | $h$ | $h$ | 标准 Transformer，编码器优先 |
| MQA | $h$ | $1$ | 推理速度优先，KV Cache 更小 |
| GQA | $h$ | $g$（$1 < g < h$） | 平衡性能与效率（LLaMA-2, Mistral） |

---

## 八、小结

多头注意力的核心价值：

1. **多视角**：每个头学习不同的关注模式（语法、语义、指代等）
2. **参数不增加**：通过降低每头维度保持总参数量不变
3. **表达能力更强**：拼接多个头的输出，信息更丰富
4. **并行高效**：通过 reshape + batch matmul 实现所有头的并行计算

下一篇 D4 将分析多头注意力的复杂度瓶颈，以及 FlashAttention、稀疏注意力等优化方案的理论基础。

---

*参考文献：Vaswani et al. (2017)；Ainslie et al. (2023) "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"*
