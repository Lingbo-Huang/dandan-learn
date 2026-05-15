---
layout: default
title: "D2 · 序列并行：突破单卡序列长度限制"
render_with_liquid: false
---

# D2 · 序列并行：突破单卡序列长度限制

## 为什么需要序列并行？

Tensor Parallel 解决了模型权重的切分问题，但 **激活值（Activations）** 仍然占用大量内存：

对于 GPT-3 175B（seq_len=2048, hidden=12288, batch=1）：
- 每层激活值：约 1.5 GB
- 96 层总计：约 144 GB

即使 TP=8，每 rank 也需要 144/8 = 18 GB（仅激活值）！

**序列并行（Sequence Parallel）**：将序列维度也切分到多个 rank，进一步减少激活值内存。

## TP 中的序列并行机会

在 Megatron 的 TP 设计中，有些操作是**逐元素**的（Layer Norm、Dropout），这些操作不需要 All-Reduce，可以在**序列的子集**上独立计算：

```
没有序列并行：
  All-Reduce → [B, S, H] → LayerNorm → [B, S, H] → ...（全量在每个 rank）
  
有序列并行：
  All-Gather → [B, S/tp, H] → LayerNorm → [B, S/tp, H] → ...（每个 rank 只处理 S/tp）
```

将 All-Reduce 替换为 **Reduce-Scatter + All-Gather**（这两个操作等价，但允许中间保持分布式状态）：

```python
"""
序列并行：Reduce-Scatter + All-Gather 替换 All-Reduce
"""
import torch
import torch.distributed as dist

def all_reduce(x, group):
    """传统 All-Reduce"""
    dist.all_reduce(x, group=group)
    return x  # [B, S, H] 在所有 rank 完全相同

def reduce_scatter_then_layer_norm(x, ln, group):
    """
    序列并行：Reduce-Scatter 后做 Layer Norm
    输出是序列的一部分（S/tp），而不是完整序列
    """
    tp_size = dist.get_world_size(group)
    
    # Reduce-Scatter：每个 rank 接收 1/tp 的结果
    # x: [B, S, H] → output: [B, S/tp, H]
    x_scattered = torch.empty(
        x.shape[0], x.shape[1] // tp_size, x.shape[2],
        dtype=x.dtype, device=x.device
    )
    dist.reduce_scatter_tensor(x_scattered, x, group=group)
    
    # Layer Norm 只在 S/tp 个 token 上做（激活值减少 tp 倍！）
    return ln(x_scattered)  # [B, S/tp, H]

def all_gather_before_attention(x, group):
    """
    在 Attention 之前 All-Gather 恢复完整序列
    """
    tp_size = dist.get_world_size(group)
    x_full = torch.empty(
        x.shape[0], x.shape[1] * tp_size, x.shape[2],
        dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(x_full, x, group=group)
    return x_full  # [B, S, H]
```

## 完整的 SP 层结构

```
传统 TP：
[All-Reduce] → [Layer Norm] → [Attention (TP)] → [All-Reduce] → [Layer Norm] → [MLP (TP)] → [All-Reduce]

加入 SP 后：
[Reduce-Scatter] → [Layer Norm on S/tp] → [All-Gather] → [Attention (TP)] → [Reduce-Scatter] → [Layer Norm on S/tp] → [All-Gather] → [MLP (TP)] → [Reduce-Scatter]

通信次数：All-Reduce × 2 → (Reduce-Scatter + All-Gather) × 2
通信量：相同（Reduce-Scatter + All-Gather ≈ All-Reduce）
内存节省：Layer Norm 的激活值减少 tp 倍！
```

## Ring Attention：极端序列并行

当序列长度超过单卡限制时（如 1M tokens），需要更激进的序列并行。

**Ring Attention**（Liu et al., 2023）：
- 将 Q 均匀分配到 $P$ 个设备：每个 rank $q_i = Q[i \cdot N/P : (i+1) \cdot N/P]$
- K, V 以环形方式在 rank 间传递
- 每个 rank 计算自己的 Q 与当前收到的 K, V 的 Attention

```python
"""
Ring Attention 概念实现
每个 rank 持有 Q[i*chunk:(i+1)*chunk]
K, V 在 ring 中循环传递
"""
import torch
import torch.distributed as dist

def ring_attention(q_chunk, k_full_local, v_full_local, process_group):
    """
    Ring Attention 前向传播
    q_chunk:       [B, S/P, H, D] - 当前 rank 的 Q 块
    k_full_local:  [B, S/P, H, D] - 当前 rank 持有的 K 块（初始）
    v_full_local:  [B, S/P, H, D] - 当前 rank 持有的 V 块（初始）
    """
    rank      = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    
    # 用 Online Softmax 累积输出
    out = torch.zeros_like(q_chunk)
    m   = torch.full(q_chunk.shape[:3] + (1,), float('-inf'))
    ell = torch.zeros(q_chunk.shape[:3] + (1,))
    
    k_buf = k_full_local.clone()
    v_buf = v_full_local.clone()
    
    for step in range(world_size):
        # 计算 q_chunk 对当前 k_buf 的 attention
        scale = q_chunk.shape[-1] ** -0.5
        qk = torch.einsum('bshd,bthd->bsht', q_chunk, k_buf) * scale
        
        # Causal mask（可选）
        # ...
        
        # Online softmax 更新
        m_block = qk.max(dim=-1, keepdim=True).values
        exp_qk  = torch.exp(qk - m_block)
        ell_block = exp_qk.sum(dim=-1, keepdim=True)
        
        m_new = torch.maximum(m, m_block)
        alpha = torch.exp(m - m_new)
        
        out = out * alpha + (exp_qk / ell_block) @ v_buf
        ell = alpha * ell + exp_qk.sum(dim=-1, keepdim=True) / ell_block
        m   = m_new
        
        if step < world_size - 1:
            # 发送 k_buf, v_buf 给下一个 rank，从上一个 rank 接收
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1) % world_size
            
            # 使用 isend/irecv 重叠通信与计算
            k_new = torch.empty_like(k_buf)
            v_new = torch.empty_like(v_buf)
            
            req_k_send = dist.isend(k_buf, dst=next_rank, group=process_group)
            req_v_send = dist.isend(v_buf, dst=next_rank, group=process_group)
            req_k_recv = dist.irecv(k_new, src=prev_rank, group=process_group)
            req_v_recv = dist.irecv(v_new, src=prev_rank, group=process_group)
            
            req_k_recv.wait()
            req_v_recv.wait()
            k_buf = k_new
            v_buf = v_new
    
    out = out / ell  # 归一化
    return out
```

## 性能数据

**序列并行内存节省（GPT-3 175B, TP=8）：**

| 组件 | 无 SP | 有 SP | 节省 |
|------|-------|-------|------|
| 权重（TP 切分） | 2.2 GB/rank | 2.2 GB/rank | 0% |
| 激活值（Layer Norm 等）| 3.1 GB/rank | 0.4 GB/rank | **87%** |
| 总内存/rank | 5.3 GB | 2.6 GB | **51%** |

**Ring Attention 支持的最长序列（A100 80GB, LLaMA-2 7B）：**

| 方案 | 最长序列 | 所需 GPU |
|------|---------|---------|
| 标准（无 SP） | 16K | 1 |
| 序列并行（TP=8） | 128K | 8 |
| Ring Attention（P=64）| 1M | 64 |

## 实践：使用 Megatron-LM 开启序列并行

```python
# Megatron-LM 配置
args = {
    'tensor_model_parallel_size': 8,
    'sequence_parallel': True,  # 开启序列并行
    # 内部会自动使用 Reduce-Scatter / All-Gather
}

# 或者在 torchrun 命令中：
# --sequence-parallel  # 开启 Megatron 的序列并行
```

## 面试题

**Q: 序列并行和 Tensor Parallel 的关系是什么？它解决了什么问题？**

A: 序列并行是 Tensor Parallel 的扩展，两者配合使用（不是替代关系）。Tensor Parallel 将权重矩阵切分到不同 rank，但逐元素操作（LayerNorm、Dropout）的激活值仍是全量的 [B, S, H]，占用大量内存。序列并行将这些逐元素操作的激活值也按序列维度切分（每个 rank 处理 S/tp 个 token），内存减少 tp 倍。通信方面，原本的 All-Reduce 被分解为 Reduce-Scatter（切分序列）+ All-Gather（合并序列），通信总量不变但激活值内存大幅下降。对于 GPT-3 175B with TP=8，序列并行可减少约 50% 的激活值内存。
