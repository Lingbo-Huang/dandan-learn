---
layout: default
title: "D5 · 位置编码进阶：RoPE 与 ALiBi"
render_with_liquid: false
---

# D5 · 位置编码进阶：RoPE 与 ALiBi

> **为什么需要位置编码？** Attention 本身是置换不变的（permutation-invariant），必须注入位置信息告诉模型 token 的顺序。

---

## 一、绝对位置编码回顾

原始 Transformer 使用固定的正弦余弦位置编码：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

**问题**：训练长度 $T_{train}$ 固定，泛化到更长序列时性能下降严重。

---

## 二、RoPE（旋转位置编码）

RoPE（Su et al., 2021）用于 LLaMA、Qwen、Mistral 等几乎所有现代 LLM。

### 2.1 核心思想

RoPE 不是将位置编码**加**到 embedding 上，而是通过**旋转**矩阵作用于 Q 和 K：

$$\text{Attention}_{m,n} = \text{Re}[(\mathbf{R}_{\theta}^m \mathbf{q}_m)^* (\mathbf{R}_{\theta}^n \mathbf{k}_n)]$$

其中 $\mathbf{R}_{\theta}^m$ 是位置 m 的旋转矩阵，关键性质：

$$(\mathbf{R}_{\theta}^m \mathbf{q})^T (\mathbf{R}_{\theta}^n \mathbf{k}) = \mathbf{q}^T \mathbf{R}_{\theta}^{n-m} \mathbf{k}$$

只依赖**相对位置** (n-m)，天然支持相对位置建模！

### 2.2 实现

```python
import torch
import math

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    预计算旋转频率
    
    Args:
        dim: 头维度（d_head）
        seq_len: 序列长度
        theta: 基础频率（默认 10000，LLaMA-3 用 500000）
    
    Returns:
        freqs_cis: [seq_len, dim//2] 复数张量
    """
    # 计算每个维度对应的频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # freqs: [dim//2]
    
    t = torch.arange(seq_len)  # 位置序列
    freqs = torch.outer(t, freqs)  # [seq_len, dim//2]
    
    # 转换为复数形式 e^{i * freq}
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    应用 RoPE
    
    Args:
        xq: [batch, seq_len, n_heads, d_head]
        xk: [batch, seq_len, n_heads, d_head]
        freqs_cis: [seq_len, d_head//2]
    
    Returns:
        xq_out, xk_out: 旋转后的 Q 和 K
    """
    # 将实数张量视为复数（每两个维度组成一个复数）
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # xq_: [batch, seq_len, n_heads, d_head//2]（复数）
    
    # 广播：freqs_cis 形状 [seq_len, d_head//2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, d_head//2]
    
    # 复数乘法 = 旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPEAttention(torch.nn.Module):
    """带 RoPE 的因果自注意力"""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.wv = torch.nn.Linear(d_model, d_model, bias=False)
        self.wo = torch.nn.Linear(d_model, d_model, bias=False)
        
        # 预计算 RoPE 频率
        freqs = precompute_freqs_cis(self.d_head, max_seq_len, theta)
        self.register_buffer('freqs_cis', freqs)
        
        # 因果掩码
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.d_head)
        k = self.wk(x).view(B, T, self.n_heads, self.d_head)
        v = self.wv(x).view(B, T, self.n_heads, self.d_head)
        
        # 应用 RoPE
        q, k = apply_rotary_emb(q, k, self.freqs_cis[:T])
        
        # 计算 Attention
        q = q.transpose(1, 2)  # [B, n_heads, T, d_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)
```

### 2.3 RoPE 的长度外推

NTK-aware 缩放（Dynamic NTK）：

```python
def apply_ntk_scaling(freqs_cis: torch.Tensor, scale: float) -> torch.Tensor:
    """
    NTK-aware 缩放：将 theta 调整为 theta * scale^(d/(d-2))
    用于推理时外推到比训练更长的序列
    
    scale = 当前长度 / 训练长度
    """
    # 重新计算更大的 theta
    # 此处简化：实际需要根据 scale 重新计算频率
    pass

def precompute_freqs_cis_with_scale(
    dim: int, 
    seq_len: int, 
    theta: float = 10000.0,
    scale: float = 1.0
):
    """支持长度外推的 RoPE 频率"""
    if scale > 1.0:
        # NTK: 等效于增大 theta
        theta = theta * (scale ** (dim / (dim - 2)))
    
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)
```

---

## 三、ALiBi（Attention with Linear Biases）

ALiBi 不用位置编码，而是在 attention score 上加线性偏置：

$$a_{i,j} = \mathbf{q}_i \cdot \mathbf{k}_j - m \cdot |i - j|$$

其中 $m$ 是每个头的斜率，按幂律分布。

```python
import torch
import math

def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """计算 ALiBi 的斜率（每个注意力头一个）"""
    def get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]
    
    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        # 处理非 2 的幂的 head 数
        closest_power = 2 ** math.floor(math.log2(n_heads))
        base_slopes = get_slopes_power_of_2(closest_power)
        extra_slopes = get_slopes_power_of_2(2 * closest_power)[0::2]
        slopes = base_slopes + extra_slopes[:n_heads - closest_power]
    
    return torch.tensor(slopes, dtype=torch.float32)

def build_alibi_bias(n_heads: int, seq_len: int) -> torch.Tensor:
    """构建 ALiBi 偏置矩阵"""
    slopes = get_alibi_slopes(n_heads)  # [n_heads]
    
    # 相对位置矩阵 |i - j|（下三角）
    positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    # [seq_len, seq_len]，值为 j - i（负数为左侧 token）
    
    # ALiBi 只惩罚"回看"距离（causal attention）
    positions = torch.clamp(positions, max=0)  # 只保留非正值
    
    # [n_heads, seq_len, seq_len]
    alibi_bias = slopes.unsqueeze(1).unsqueeze(2) * positions.unsqueeze(0)
    
    return alibi_bias

# ALiBi vs RoPE 对比
print("ALiBi 偏置矩阵（前5个位置，1个head）:")
bias = build_alibi_bias(n_heads=8, seq_len=5)
print(bias[0])  # 第一个 head 的偏置
```

---

## 四、RoPE vs ALiBi vs 绝对位置编码

| 特性 | 绝对 PE | ALiBi | RoPE |
|------|---------|-------|------|
| 外推能力 | 弱（训练长度硬限制）| 强（线性衰减自然外推）| 中（需要 NTK 缩放）|
| 相对位置 | 间接 | 直接 | 直接 |
| 使用模型 | 原始 Transformer | Bloom, MPT | LLaMA, Qwen, Mistral |
| 参数量 | 有（$T \times d$）| 无 | 无 |
| 代码复杂度 | 简单 | 中等 | 较复杂 |

---

## 五、YaRN：LLaMA 的上下文长度扩展

```python
def yarn_rope_scaling(
    freqs: torch.Tensor,
    scale: float,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    original_max_position: int = 4096
) -> torch.Tensor:
    """
    YaRN (Yet another RoPE extensioN)
    
    对不同频率维度使用不同的缩放策略：
    - 低频（长距离）：线性插值
    - 高频（短距离）：不缩放
    - 中频：混合
    """
    low_freq_wavelen = original_max_position / low_freq_factor
    high_freq_wavelen = original_max_position / high_freq_factor
    
    # 根据波长判断频率类型
    # 此处简化，实际实现参考 HuggingFace transformers 源码
    new_freqs = freqs.clone()
    for i, freq in enumerate(freqs):
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            # 高频：不缩放
            pass
        elif wavelen > low_freq_wavelen:
            # 低频：线性插值（除以 scale）
            new_freqs[i] = freq / scale
        else:
            # 中频：插值
            smooth = (original_max_position / wavelen - low_freq_factor) / \
                     (high_freq_factor - low_freq_factor)
            new_freqs[i] = (1 - smooth) * freq / scale + smooth * freq
    
    return new_freqs
```

---

## 六、面试题精讲

**Q: RoPE 为什么能做到相对位置编码？**

A: 对复数的旋转乘法，$q_m \cdot k_n^* = q \cdot R_{\theta}^{m-n} k$，内积只依赖相对位置差 (m-n)，而不是绝对位置 m 和 n。这是 RoPE 的核心数学性质。

**Q: 为什么 LLaMA-3 把 theta 从 10000 改成 500000？**

A: 更大的 theta 意味着旋转速度更慢，低频维度的旋转周期更长，有助于模型感知更远的位置关系，从而支持更长的上下文（128K）。

**Q: 如何让 4K 训练的模型推理时支持 32K？**

A: 主要方法：
1. **位置插值（PI）**：将位置 index 除以 8（压缩到训练范围），需要微调
2. **NTK-aware 缩放**：等效增大 theta，无需微调，但有精度损失
3. **YaRN**：对不同频率分开处理，效果最好

---

## 小结

| | RoPE | ALiBi |
|--|------|-------|
| 核心 | 旋转矩阵作用于 Q/K | Attention score 减去距离惩罚 |
| 外推 | 需要 NTK/YaRN | 天然外推 |
| 现代 LLM | 主流（LLaMA/Qwen）| Bloom, MPT |
