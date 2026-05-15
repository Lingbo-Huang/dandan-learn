---
layout: default
title: "D3 · Pipeline 调度：1F1B 与 Interleaved"
render_with_liquid: false
---

# D3 · Pipeline 调度：1F1B 与 Interleaved

## Pipeline Parallel 的核心挑战

简单的 Pipeline 并行存在严重的 **气泡（Bubble）** 问题：
- 前向传播时，后面的 Stage 必须等前面的 Stage 完成
- 反向传播时，前面的 Stage 必须等后面的 Stage 完成
- 这导致大量的 GPU 空闲时间

## GPipe 调度（基线）

```
p=4 个 Stage，m=8 个 Microbatch：

Stage 0: |F1|F2|F3|F4|F5|F6|F7|F8|          |B8|B7|B6|B5|B4|B3|B2|B1|
Stage 1: |  |F1|F2|F3|F4|F5|F6|F7|F8|       |B8|B7|B6|B5|B4|B3|B2|B1|
Stage 2: |  |  |F1|F2|F3|F4|F5|F6|F7|F8|    |B8|B7|B6|B5|B4|B3|B2|B1|
Stage 3: |  |  |  |F1|F2|F3|F4|F5|F6|F7|F8|B8|B7|B6|B5|B4|B3|B2|B1|

Bubble 率 = (p-1) / (m + p - 1) = 3/11 ≈ 27%
内存峰值：同时存 m=8 个 microbatch 的激活！
```

GPipe 的问题：
1. **高 Bubble 率**：27%
2. **高内存**：m 个激活值同时在内存中

## 1F1B 调度（Pipedream）

PipeDream 提出 **1F1B（One Forward One Backward）** 调度：

```
p=4, m=8：

Stage 0: |F1|F2|F3|F4|B1|F5|B2|F6|B3|F7|B4|F8|B5|B6|B7|B8|
Stage 1: |  |F1|F2|F3|B1|F4|B2|F5|B3|F6|B4|F7|B5|F8|B6|B7|B8|
Stage 2: |  |  |F1|F2|B1|F3|B2|F4|B3|F5|B4|F6|B5|F7|B6|F8|B7|B8|
Stage 3: |  |  |  |F1|B1|F2|B2|F3|B3|F4|B4|F5|B5|F6|B6|F7|B7|F8|B8|

Bubble 率 = (p-1) / (m + p - 1)（与 GPipe 相同，约 27%）
内存峰值：仅需 p=4 个 microbatch 的激活！（大幅减少）
```

1F1B 的关键改进：**内存从 O(m) 降至 O(p)**，Bubble 率不变。

```python
"""
1F1B Pipeline 调度实现（简化版）
"""
import torch
import torch.distributed as dist
from typing import List, Callable

class PipelineStageRunner:
    def __init__(self, stage_id: int, num_stages: int, num_microbatches: int):
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.m = num_microbatches
        self.p = num_stages
        
        # 存储激活值（用于反向传播）
        self.activation_cache = {}
    
    def get_1f1b_schedule(self) -> List[tuple]:
        """
        生成 1F1B 调度序列
        返回 [(操作, microbatch_id), ...]
        """
        schedule = []
        
        # Warmup 阶段（每个 stage 先做 p-stage_id 次前向）
        warmup_steps = self.p - self.stage_id
        for i in range(warmup_steps):
            schedule.append(('forward', i))
        
        # 稳定阶段（1F1B）
        forward_mb = warmup_steps
        backward_mb = 0
        
        while forward_mb < self.m or backward_mb < self.m - warmup_steps:
            if backward_mb < self.m - warmup_steps:
                schedule.append(('backward', backward_mb))
                backward_mb += 1
            if forward_mb < self.m:
                schedule.append(('forward', forward_mb))
                forward_mb += 1
        
        # Cooldown：剩余反向传播
        while backward_mb < self.m:
            schedule.append(('backward', backward_mb))
            backward_mb += 1
        
        return schedule
    
    def run_1f1b(
        self, 
        microbatches: List[torch.Tensor],
        forward_fn: Callable,
        backward_fn: Callable,
    ):
        schedule = self.get_1f1b_schedule()
        
        for op, mb_id in schedule:
            if op == 'forward':
                # 从上一个 Stage 接收激活（P2P）
                if self.stage_id > 0:
                    x = self._recv_forward()
                else:
                    x = microbatches[mb_id]  # 第一个 Stage 使用原始数据
                
                # 执行前向传播
                output = forward_fn(x)
                self.activation_cache[mb_id] = (x, output)
                
                # 发送激活给下一个 Stage
                if self.stage_id < self.num_stages - 1:
                    self._send_forward(output)
                
            elif op == 'backward':
                # 从下一个 Stage 接收梯度（P2P）
                if self.stage_id < self.num_stages - 1:
                    grad_output = self._recv_backward()
                else:
                    grad_output = None  # 最后一个 Stage 自己算 loss
                
                # 取回缓存的激活值
                x, output = self.activation_cache.pop(mb_id)
                
                # 执行反向传播
                grad_input = backward_fn(output, grad_output, x)
                
                # 发送梯度给上一个 Stage
                if self.stage_id > 0:
                    self._send_backward(grad_input)
    
    def _send_forward(self, tensor):
        dist.send(tensor, dst=self.stage_id + 1)
    
    def _recv_forward(self):
        tensor = torch.empty(...)  # 需要知道 shape
        dist.recv(tensor, src=self.stage_id - 1)
        return tensor
    
    def _send_backward(self, tensor):
        dist.send(tensor, dst=self.stage_id - 1)
    
    def _recv_backward(self):
        tensor = torch.empty(...)
        dist.recv(tensor, src=self.stage_id + 1)
        return tensor
```

## Interleaved 1F1B 调度

Megatron-LM v2 进一步提出了 **Interleaved Schedule**：

**关键思想**：每个 GPU 不是连续的层，而是**交错**地持有多个"虚拟阶段"（Virtual Stage）。

```
传统 PP（p=4, 16 层）：
  GPU0: layers 0-3     GPU1: layers 4-7
  GPU2: layers 8-11    GPU3: layers 12-15

Interleaved PP（p=4, v=2 虚拟阶段, 16 层）：
  GPU0: layers 0-1, 8-9    GPU1: layers 2-3, 10-11
  GPU2: layers 4-5, 12-13  GPU3: layers 6-7, 14-15
```

Interleaved 调度的 Bubble 率：

$$\text{Bubble Rate} = \frac{p-1}{m \cdot v + p - 1} \approx \frac{1}{m \cdot v}$$

其中 $v$ 是每个设备的虚拟阶段数。通过增加 $v$，Bubble 率接近 0（但通信次数增加 $v$ 倍）。

```
Interleaved 时间线（p=4, v=2, m=4）：
GPU0(S0,S1): |F1s0|F2s0|F3s0|F4s0|F1s1|B1s1|F2s1|B2s1|F3s1|B3s1|F4s1|B4s1|B4s0|B3s0|B2s0|B1s0|
...

Bubble 率 ≈ (p-1)/(m*v + p-1) = 3/(4*2+3) = 3/11 vs 非 Interleaved 的 3/(4+3) = 43%
实际下降约 35%（m 相同时）
```

## 通信重叠优化

Interleaved 调度的通信可以与计算重叠：

```python
"""
异步 P2P 通信与计算重叠
"""
import torch
import torch.distributed as dist

def pipeline_step_with_overlap(
    current_microbatch_data,
    forward_fn,
    next_stage_rank
):
    """
    前向传播 + 异步发送（重叠通信与下一步计算）
    """
    # 计算
    output = forward_fn(current_microbatch_data)
    
    # 异步发送（不阻塞）
    send_req = dist.isend(output, dst=next_stage_rank)
    
    # 在等待发送完成的同时，可以开始下一个 microbatch 的计算
    # ... 处理下一个 microbatch ...
    
    # 等待发送完成
    send_req.wait()
    
    return output
```

## 各调度方案对比

| 调度方案 | Bubble 率 | 内存占用 | 通信量 | 适用场景 |
|---------|----------|---------|-------|---------|
| GPipe | (p-1)/(m+p-1) | O(m) | 1× | 小规模 |
| 1F1B | (p-1)/(m+p-1) | O(p) | 1× | 通用 |
| Interleaved | ~(p-1)/(m·v+p-1) | O(p) | v× | 大 m 时 |

**实践建议**：
- m 较小（< 8）：使用 Interleaved，显著降低 Bubble
- m 较大（>= 16）：1F1B 已足够，Interleaved 增加通信开销不值
- 内存敏感：优先 1F1B（O(p) vs O(m)）

## 面试题

**Q: 1F1B 和 GPipe 调度的 Bubble 率相同，1F1B 的优势在哪里？**

A: 1F1B 和 GPipe 的 Bubble 率公式相同（都是 (p-1)/(m+p-1)）。1F1B 的核心优势是**内存效率**：GPipe 需要同时保存所有 m 个 microbatch 的激活值（用于反向传播），内存峰值是 O(m)；而 1F1B 在稳定阶段每做一次前向就立即做一次反向，激活值只需要保存 p 个（等于 Stage 数），内存峰值是 O(p)。对于 m=8, p=4 的情况，内存节省 2×；m 越大，节省越显著。Interleaved 进一步降低 Bubble 率，但增加通信量 v 倍，适合 m 较小的场景。
