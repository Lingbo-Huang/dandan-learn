---
layout: default
title: "D5 · 专家并行与 MoE 训练挑战"
render_with_liquid: false
---

# D5 · 专家并行与 MoE 训练挑战

## 专家并行（Expert Parallel）

当 MoE 的专家数量超过单卡容量时，需要将不同专家分配到不同 GPU。

**核心通信模式**：All-to-All

```
设 8 个专家，4 个 GPU（每 GPU 2 个专家，EP=4）：

Token 路由阶段：
  GPU0 有 Token0~T99，路由到 Expert0,1,2,3
  GPU1 有 Token100~T199，路由到 Expert0,1,2,3
  GPU2 有 Token200~T299，路由到 Expert4,5,6,7
  GPU3 有 Token300~T399，路由到 Expert4,5,6,7

All-to-All 发送（dispatch）：
  GPU0 → GPU0: 路由到 Expert0,1 的 tokens（本地）
  GPU0 → GPU1: 路由到 Expert2,3 的 tokens
  GPU0 → GPU2: 路由到 Expert4,5 的 tokens
  GPU0 → GPU3: 路由到 Expert6,7 的 tokens
  ...

专家计算（每 GPU 并行处理属于本机专家的 tokens）

All-to-All 回收（combine）：
  将计算结果发回原始 GPU
```

## All-to-All 通信实现

```python
import torch
import torch.distributed as dist
from typing import Tuple

def expert_dispatch(
    tokens: torch.Tensor,           # [local_tokens, hidden] - 本地 token
    expert_indices: torch.Tensor,   # [local_tokens, top_k] - 路由结果
    num_experts: int,
    ep_group: dist.ProcessGroup,    # 专家并行通信组
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 token 发送到对应专家所在的 GPU
    返回：
      - dispatched_tokens: [total_received_tokens, hidden]
      - recv_counts: [ep_size] - 每个 GPU 发来的 token 数量
    """
    ep_size  = dist.get_world_size(ep_group)
    ep_rank  = dist.get_rank(ep_group)
    experts_per_rank = num_experts // ep_size
    
    local_tokens, hidden = tokens.shape
    top_k = expert_indices.shape[1]
    
    # 1. 统计每个 rank 需要发送多少 token
    # 专家 e 在 rank e // experts_per_rank 上
    send_counts = torch.zeros(ep_size, dtype=torch.int64)
    for k in range(top_k):
        expert_ids = expert_indices[:, k]
        target_ranks = expert_ids // experts_per_rank
        for r in range(ep_size):
            send_counts[r] += (target_ranks == r).sum()
    
    # 2. All-to-All 交换 send_counts
    recv_counts = torch.zeros(ep_size, dtype=torch.int64)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
    
    # 3. 按 target rank 排序 token
    # 构建发送缓冲区：按 rank 排序
    sorted_tokens = []
    for r in range(ep_size):
        for k in range(top_k):
            mask = (expert_indices[:, k] // experts_per_rank == r)
            sorted_tokens.append(tokens[mask])
    send_buffer = torch.cat(sorted_tokens, dim=0)  # [total_send, hidden]
    
    # 4. All-to-All 发送实际数据
    recv_buffer = torch.empty(
        recv_counts.sum().item(), hidden,
        dtype=tokens.dtype, device=tokens.device
    )
    dist.all_to_all_single(
        recv_buffer, send_buffer,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
        group=ep_group
    )
    
    return recv_buffer, recv_counts


def expert_combine(
    expert_outputs: torch.Tensor,  # [total_received_tokens, hidden]
    send_counts: torch.Tensor,     # 原始发送计数
    recv_counts: torch.Tensor,     # 接收计数
    expert_weights: torch.Tensor,  # [local_tokens, top_k] - 路由权重
    local_tokens: int,
    hidden: int,
    ep_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    将专家计算结果发回原始 GPU，并做加权求和
    """
    # All-to-All 回收（逆操作）
    output_buffer = torch.empty(
        send_counts.sum().item(), hidden,
        dtype=expert_outputs.dtype, device=expert_outputs.device
    )
    dist.all_to_all_single(
        output_buffer, expert_outputs,
        output_split_sizes=send_counts.tolist(),
        input_split_sizes=recv_counts.tolist(),
        group=ep_group
    )
    
    # 加权求和（每个 token 的多个专家输出加权合并）
    # output_buffer 按 [rank, k, tokens] 排列，需要重新排列并加权
    final_output = torch.zeros(local_tokens, hidden, device=expert_outputs.device)
    # ... 重新排列并加权（实际实现较复杂）
    
    return final_output
```

## MoE 训练的通信瓶颈

在大规模 EP 训练中，All-to-All 通信可能成为瓶颈：

```python
def estimate_alltoall_cost(
    num_tokens: int,    # 每个 rank 的 token 数
    hidden_dim: int,
    ep_size: int,       # 专家并行大小
    bandwidth_gbs: float = 200,  # 网络带宽（GB/s，InfiniBand）
    dtype_bytes: int = 2
) -> float:
    """
    估算 All-to-All 通信时间（ms）
    假设每个 rank 平均把 (ep_size-1)/ep_size 的数据发出去
    """
    # 每个 rank 发送的数据量（假设均匀分布）
    data_per_send = num_tokens * (ep_size - 1) / ep_size * hidden_dim * dtype_bytes
    
    # All-to-All 总数据量（每个 rank 都发送 data_per_send）
    total_data = data_per_send  # 单向
    
    time_ms = total_data / (bandwidth_gbs * 1e9) * 1000
    return time_ms

# 示例：16 EP，2048 tokens/rank，hidden=4096
cost = estimate_alltoall_cost(2048, 4096, 16)
print(f"All-to-All 估算时间: {cost:.2f} ms")
# 典型输出：~1-5 ms（取决于网络拓扑）
```

## 通信与计算重叠（All-to-All 流水线）

```python
"""
All-to-All 与专家计算的流水线重叠
"""
import torch
import torch.distributed as dist

def moe_layer_with_overlap(
    tokens, router, experts, ep_group,
    compute_stream, comm_stream
):
    """
    使用 CUDA Stream 重叠通信和计算
    """
    # 在计算流上路由
    with torch.cuda.stream(compute_stream):
        expert_weights, expert_indices = router(tokens)
    
    torch.cuda.synchronize()
    
    # 在通信流上发起 All-to-All（dispatch）
    with torch.cuda.stream(comm_stream):
        dispatched, recv_counts = expert_dispatch(
            tokens, expert_indices, len(experts), ep_group
        )
    
    # 在 All-to-All 进行的同时，处理本 rank 的专家计算
    # （实际需要等待 dispatch 完成，但可以调整顺序重叠）
    
    comm_stream.synchronize()  # 等待 dispatch 完成
    
    # 专家计算（可以与下一层的路由重叠）
    with torch.cuda.stream(compute_stream):
        expert_outs = compute_experts(dispatched, experts, ep_group)
    
    # 回收（combine All-to-All）
    with torch.cuda.stream(comm_stream):
        final_out = expert_combine(
            expert_outs, send_counts, recv_counts,
            expert_weights, tokens.shape[0], tokens.shape[1], ep_group
        )
    
    comm_stream.synchronize()
    return final_out
```

## MoE 与稠密模型的对比（实测）

| 模型 | 参数量 | 激活参数 | 训练 FLOPs | 推理速度 | 精度（MMLU） |
|------|-------|---------|-----------|---------|------------|
| LLaMA-2 7B | 7B | 7B | 1× | 1× | 46% |
| LLaMA-2 13B | 13B | 13B | 1.9× | 0.7× | 55% |
| Mixtral 8×7B | 47B(总) | 12B(激活) | 1.7× | 1.5× | 70% |
| LLaMA-2 70B | 70B | 70B | 10× | 0.25× | 69% |

**Mixtral 8×7B 的代价效率**：
- 精度接近 LLaMA-2 70B（70% vs 69%）
- 训练 FLOPs 只有 17%
- 推理速度是 70B 的 6×

## 面试题

**Q: MoE 的 All-to-All 通信在哪里发生？为什么是瓶颈？**

A: All-to-All 发生在每个 MoE 层的两个地方：①Dispatch（分发）：每个 GPU 将本地 token 发送到持有对应专家的 GPU；②Combine（回收）：专家计算完后，将结果发回原始 GPU。瓶颈原因：1) 通信频率高，每个 MoE 层都有两次 All-to-All；2) 数据量大，每个 token 需要传输整个 hidden_dim；3) InfiniBand 带宽（~200Gbps）远低于 NVLink（600GB/s），跨节点时尤其慢。缓解方式：EP 优先使用 NVLink 连接的 GPU（同节点），超出后再跨节点；使用异步 All-to-All 与专家计算重叠；减小 token 容量（capacity factor）降低通信量。
