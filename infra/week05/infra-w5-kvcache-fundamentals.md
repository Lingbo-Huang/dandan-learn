---
layout: default
title: "D1 · KV Cache 基础与内存管理挑战"
render_with_liquid: false
---

# D1 · KV Cache 基础与内存管理挑战

## 为什么需要 KV Cache？

LLM 推理是自回归的：生成第 t 个 token 时，需要 attention 操作关注所有前 t-1 个 token。

**没有 KV Cache**：每步生成都要重新计算所有历史 token 的 K, V：
```
生成第 t 个 token 的计算量: O(t × d)
总计算量（生成 T 个 token）: O(T² × d)
```

**有 KV Cache**：每层保存历史 K, V，新 token 只做增量计算：
```
每步只计算新 token 的 K, V：O(d)
总计算量: O(T × d)
```

**时间复杂度：从 O(T²) 降至 O(T)**，代价是 O(T) 额外显存。

## KV Cache 的内存占用计算

对于 LLaMA-2 7B（典型配置）：
- 层数：32
- 注意力头数：32（KV heads = 32，未使用 GQA）
- Head Dimension：128
- 数据类型：FP16（2 bytes）

**每个 Token 的 KV Cache 大小**：
```
每层: 2(K+V) × 32(heads) × 128(dim) × 2(bytes) = 16,384 bytes = 16 KB
全部 32 层: 32 × 16 KB = 512 KB per token
```

**不同序列长度的 KV Cache**：
```python
def kv_cache_size(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2  # FP16
) -> float:
    """返回 KV Cache 大小（GB）"""
    size_bytes = (
        2           # K + V
        * num_layers
        * num_kv_heads
        * head_dim
        * seq_len
        * batch_size
        * dtype_bytes
    )
    return size_bytes / 1024**3

# LLaMA-2 7B
print("LLaMA-2 7B KV Cache:")
for seq_len in [1024, 4096, 8192, 32768]:
    size_gb = kv_cache_size(32, 32, 128, seq_len)
    print(f"  seq_len={seq_len:6d}: {size_gb:.2f} GB")

# 输出：
# LLaMA-2 7B KV Cache:
#   seq_len=  1024: 0.50 GB
#   seq_len=  4096: 2.00 GB
#   seq_len=  8192: 4.00 GB
#   seq_len= 32768: 16.00 GB

# LLaMA-2 70B（GQA: 8 KV heads）
print("\nLLaMA-2 70B KV Cache (GQA, 8 KV heads):")
for seq_len in [1024, 4096, 8192, 32768]:
    size_gb = kv_cache_size(80, 8, 128, seq_len)
    print(f"  seq_len={seq_len:6d}: {size_gb:.2f} GB")
```

## 朴素 KV Cache 管理的三大问题

### 问题 1：内存碎片化（Fragmentation）

传统做法：为每个请求预分配 **最大上下文长度** 的连续内存。

```python
# 朴素实现：预分配固定大小
MAX_SEQ_LEN = 4096
cache = torch.zeros(batch_size, num_layers, 2, MAX_SEQ_LEN, num_heads, head_dim)
# 问题：即使序列只有 100 tokens，也占用 4096 tokens 的空间！
```

**内存利用率**：假设平均序列长度 1000，最大 4096：
- 平均利用率：1000/4096 ≈ **24.4%**
- 76% 的 GPU 内存被浪费！

### 问题 2：无法动态调整

请求到来时不知道最终长度，传统批处理必须事先知道最长序列才能确定 batch size。

```
静态批处理的困境：
  批次中最短序列: 100 tokens
  批次中最长序列: 3000 tokens
  
  → 为所有序列预留 3000 tokens 的空间
  → 短序列浪费 29× 内存
  → 无法在批次进行中加入新请求
```

### 问题 3：并发受限

因为每个请求要连续的显存，系统的最大并发数被 **最大内存可以容纳的最大序列数** 所限制。

实测：A100 80GB 上，LLaMA-2 7B 朴素实现：
- 模型权重：14 GB
- 可用于 KV Cache：66 GB
- 每个请求（假设 max 4096 tokens）：2 GB
- 最大并发：**33 个请求**（且内存利用率极低）

## KV Cache 的数据布局

理解内存布局对优化至关重要：

```python
# 方式 1：[batch, num_layers, 2, seq_len, num_heads, head_dim]
# 优点：层间连续，适合一次性读取一层的所有 KV
cache_v1 = torch.zeros(B, L, 2, S, H, D)

# 方式 2：[num_layers, 2, batch, num_heads, seq_len, head_dim]  
# 优点：同一层的 KV 连续，适合 attention 计算
cache_v2 = torch.zeros(L, 2, B, H, S, D)

# vLLM 的实际布局（PagedAttention）
# [num_blocks, num_kv_heads, head_size, block_size]
# block_size 通常 16，使用类似虚拟内存的分页机制
cache_paged = torch.zeros(num_blocks, H, D, block_size)
```

## GQA（Grouped Query Attention）如何减少 KV Cache

LLaMA-2 70B, Mistral, LLaMA-3 等模型使用 GQA：

```
标准 MHA: Q heads = K heads = V heads = 32
MQA:      Q heads = 32, K heads = V heads = 1  (极端压缩)
GQA:      Q heads = 32, K heads = V heads = 8  (平衡)

KV Cache 节省比例：
  MHA → GQA(8): 4× 节省（32/8）
  MHA → MQA:    32× 节省（但精度损失大）
```

```python
import torch
import torch.nn.functional as F

class GroupedQueryAttention(torch.nn.Module):
    def __init__(self, hidden_dim, num_q_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_q_heads  = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups   = num_q_heads // num_kv_heads  # 每个 KV head 对应的 Q heads 数
        self.head_dim     = head_dim
        
        self.q_proj = torch.nn.Linear(hidden_dim, num_q_heads * head_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim)
    
    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape
        
        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        
        # KV Cache 拼接
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
        
        # GQA：将 KV head 重复扩展以匹配 Q head 数量
        k = k.repeat_interleave(self.num_groups, dim=2)  # [B, S_kv, Q_heads, D]
        v = v.repeat_interleave(self.num_groups, dim=2)
        
        # Attention（使用 FlashAttention）
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, S, -1)
        
        return out, (k[:, :, ::self.num_groups], v[:, :, ::self.num_groups])  # 只存 KV heads
```

## 本节小结

| 问题 | 根因 | 解决方向 |
|------|------|---------|
| 内存浪费 76% | 连续内存 + 预留最大长度 | PagedAttention（下一节） |
| 并发上限低 | 连续显存限制 | 分页 + 动态分配 |
| 批处理僵硬 | 静态 batch | 连续批处理（D3） |

**明天：PagedAttention——把操作系统虚拟内存的思想搬到 GPU！**

## 面试题

**Q: LLaMA-2 13B 在 A100 80GB 上，最多能同时处理多少个 4096 Token 的请求（使用 KV Cache，FP16）？**

A: 
- 模型权重：13B × 2 bytes = 26 GB
- 可用于 KV Cache：80 - 26 = 54 GB
- LLaMA-2 13B 配置：40 层, 40 KV heads, head_dim=128（实际 GQA = 40 heads）
- 每请求 KV Cache：2 × 40 × 40 × 128 × 4096 × 2 = 3.28 GB
- 最大并发：floor(54 / 3.28) ≈ **16 个请求**

注：实际还需留 buffer 给激活值等，通常更少。
