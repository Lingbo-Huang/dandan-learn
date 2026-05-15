---
layout: default
title: "D1 · KV Cache"
render_with_liquid: false
---

# D1 · KV Cache

> **一句话**：自回归推理每次只生成一个 token，但如果每次都重新计算之前所有 token 的 K 和 V，就是极大的浪费。KV Cache 把这些中间结果缓存起来。

---

## 一、为什么需要 KV Cache？

### 无缓存的推理

生成序列 $[x_1, x_2, ..., x_T]$ 时：
- 生成 $x_2$：需要计算 $x_1$ 的 K, V
- 生成 $x_3$：需要重新计算 $x_1, x_2$ 的 K, V
- 生成 $x_T$：需要重新计算所有前 T-1 个 token 的 K, V

时间复杂度：$O(T^2 \cdot d)$，生成变慢为 $O(T)$ 倍！

### 有 KV Cache

- 生成 $x_2$：计算 $x_1$ 的 K, V，缓存
- 生成 $x_3$：只需计算 $x_2$ 的 K, V，与缓存的 K, V 拼接
- 每步只需 $O(d)$ 计算新的 K, V，然后拼接

---

## 二、KV Cache 的实现

```python
import torch
import torch.nn as nn
import math

class KVCacheAttention(nn.Module):
    """支持 KV Cache 的注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, T, C]，推理时 T=1
            past_kv: (past_k, past_v)，每个形状 [B, n_heads, T_past, d_head]
        
        Returns:
            output: [B, T, C]
            new_kv: 更新后的 (k, v) 缓存
        """
        B, T, C = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # 拼接 KV Cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # [B, n_heads, T_past+T, d_head]
            v = torch.cat([past_v, v], dim=2)
        
        # Attention（新的 q 对所有历史 k, v 做 attention）
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale  # [B, n_heads, T, T_past+T]
        
        # 注意：推理时不需要因果掩码（q 只有一个时间步，不会"看到未来"）
        # 但如果 T > 1（prefill 阶段），仍需要因果掩码
        if T > 1:
            mask = torch.tril(torch.ones(T, k.size(2), device=x.device))
            mask = mask[-T:, :]  # 只取最后 T 行
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v  # [B, n_heads, T, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.wo(out), (k, v)


class GPTWithKVCache(nn.Module):
    """带 KV Cache 的 GPT 推理"""
    
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, max_seq_len=2048):
        super().__init__()
        self.n_layers = n_layers
        self.emb = nn.Embedding(vocab_size, d_model)
        # 简化：只用 KVCacheAttention，省略完整 Block
        self.attn_layers = nn.ModuleList([
            KVCacheAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    @torch.no_grad()
    def generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> torch.Tensor:
        """带 KV Cache 的自回归生成"""
        
        # Prefill 阶段：处理整个 prompt
        kv_caches = [None] * self.n_layers
        x = self.emb(input_ids)
        
        for i, attn in enumerate(self.attn_layers):
            x, kv_caches[i] = attn(x, past_kv=None)
        
        logits = self.lm_head(x[:, -1:, :])  # 只取最后位置
        next_token = logits.argmax(dim=-1)
        generated = [next_token]
        
        # Decode 阶段：每次只处理一个新 token
        for _ in range(max_new_tokens - 1):
            x = self.emb(next_token)  # [B, 1, d_model]
            
            for i, attn in enumerate(self.attn_layers):
                x, kv_caches[i] = attn(x, past_kv=kv_caches[i])
            
            logits = self.lm_head(x[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            generated.append(next_token)
        
        return torch.cat([input_ids] + generated, dim=1)
```

---

## 三、KV Cache 的内存消耗

```python
def estimate_kv_cache_memory(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    d_head: int,
    dtype_bytes: int = 2,  # BF16 = 2 bytes
) -> dict:
    """
    估算 KV Cache 内存
    
    每层：2（K+V）× batch_size × n_heads × seq_len × d_head × dtype_bytes
    """
    per_layer = 2 * batch_size * n_heads * seq_len * d_head * dtype_bytes
    total = per_layer * n_layers
    
    return {
        'per_layer_MB': per_layer / 1e6,
        'total_MB': total / 1e6,
        'total_GB': total / 1e9,
    }

# Llama-2-7B 的 KV Cache
llama7b_kv = estimate_kv_cache_memory(
    batch_size=1,
    seq_len=4096,
    n_layers=32,
    n_heads=32,
    d_head=128,
)
print("Llama-2-7B KV Cache (batch=1, seq=4096):")
for k, v in llama7b_kv.items():
    print(f"  {k}: {v:.2f}")

# 不同配置
configs = [
    (1, 2048, "batch=1, seq=2K"),
    (1, 8192, "batch=1, seq=8K"),
    (8, 2048, "batch=8, seq=2K"),
    (32, 2048, "batch=32, seq=2K"),
]

print("\nLlama-2-7B KV Cache 各配置:")
for bs, sl, desc in configs:
    result = estimate_kv_cache_memory(bs, sl, 32, 32, 128)
    print(f"  {desc}: {result['total_GB']:.2f} GB")
```

---

## 四、GQA 和 MQA（减少 KV Cache）

```python
"""
MHA（Multi-Head Attention）: n_kv_heads = n_heads
GQA（Grouped Query Attention）: n_kv_heads = n_heads // groups（LLaMA-2-70B, LLaMA-3）
MQA（Multi-Query Attention）: n_kv_heads = 1（Gemma）

内存节省：
  MQA: n_kv_heads=1，KV Cache 减少到 1/n_heads
  GQA: n_kv_heads=g，KV Cache 减少到 g/n_heads
"""

class GroupedQueryAttention(nn.Module):
    """GQA：多个 Query head 共享 KV"""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        
        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)  # 更小
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)  # 更小
        self.wo = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        
        # 将 KV 扩展以匹配 Q 的 head 数（repeat_interleave）
        k = k.repeat_interleave(self.n_groups, dim=1)  # [B, n_heads, T, d_head]
        v = v.repeat_interleave(self.n_groups, dim=1)
        
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale
        # 因果掩码...
        attn = torch.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

# KV Cache 节省对比
print("KV Cache 节省（以 LLaMA-3-8B 为基准）:")
print(f"  MHA (32 heads):      100%")
print(f"  GQA (8 kv_heads):    {8/32*100:.0f}%  ← LLaMA-3-8B 实际配置")
print(f"  MQA (1 kv_head):     {1/32*100:.1f}%")
```

---

## 五、面试题精讲

**Q: KV Cache 为什么只缓存 K 和 V，不缓存 Q？**

A: 因果注意力中，当前 token 只需要用自己的 Q 去查询所有历史的 K/V。新 token 的 Q 每次都是全新的，无需缓存。而历史 token 的 K/V 在之后的步骤中不变，所以缓存 K/V 可以避免重复计算。

**Q: Prefill 和 Decode 阶段有何不同？**

A:
- **Prefill**：处理输入 prompt，并行计算所有 token 的 K/V（类似训练），计算密集型
- **Decode**：自回归生成，每次只处理一个新 token，KV Cache 命中，内存带宽密集型

两者瓶颈不同：Prefill 受计算限制，Decode 受内存带宽限制。

**Q: GQA 相比 MHA 的代价是什么？**

A: 理论上共享 KV 会减少不同 group 间的差异化能力，但实验（LLaMA-2/3）表明 GQA 的性能损失极小，同时 KV Cache 内存减少显著（如 LLaMA-3-8B 用 8 个 KV head，节省 75%），是非常划算的 trade-off。

---

## 小结

| 概念 | 要点 |
|------|------|
| KV Cache | 缓存历史 K/V，避免重复计算 |
| 内存占用 | 2 × B × L × H × T × d_head × 2B |
| GQA | n_kv_heads < n_heads，共享 KV |
| Prefill vs Decode | 计算密集 vs 内存带宽密集 |
