---
layout: default
title: "D1 · Megatron-LM 三维并行：TP + PP + DP"
render_with_liquid: false
---

# D1 · Megatron-LM 三维并行：TP + PP + DP

## Megatron-LM 的历史地位

NVIDIA Megatron-LM（2019 起）是大规模 Transformer 训练的里程碑框架：
- GPT-3、Megatron-Turing NLG 530B、LLaMA 等均使用 Megatron 或其衍生版本训练
- 提出了将 TP（Tensor Parallel）、PP（Pipeline Parallel）、DP（Data Parallel）结合的**三维并行**方案

## Tensor Parallel（Megatron 风格）

Week2 的朴素 Tensor Parallel 在每次操作后都需要 All-Reduce，Megatron 设计了更优雅的方案：**将通信融入 Transformer 结构**，减少通信次数。

### MLP 的 TP 切分

```python
"""
Megatron-LM 风格的 Tensor Parallel MLP
关键：两个矩阵乘法，只在最后做一次 All-Reduce
"""
import torch
import torch.nn.functional as F

class ColumnParallelLinear(torch.nn.Module):
    """
    按列切分权重矩阵
    W: [in_features, out_features] → 每个 rank 持有 [in_features, out_features/tp]
    """
    def __init__(self, in_features, out_features, tp_size, tp_rank):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.out_features_per_rank = out_features // tp_size
        
        # 每个 rank 只持有权重的一部分
        self.weight = torch.nn.Parameter(
            torch.randn(self.out_features_per_rank, in_features)
        )
    
    def forward(self, x):
        # x: [batch, seq, in_features]（全量，每个 rank 相同）
        # 输出: [batch, seq, out_features/tp]（部分）
        return F.linear(x, self.weight)  # 不需要通信！

class RowParallelLinear(torch.nn.Module):
    """
    按行切分权重矩阵
    W: [in_features, out_features] → 每个 rank 持有 [in_features/tp, out_features]
    """
    def __init__(self, in_features, out_features, tp_size, tp_rank, process_group):
        super().__init__()
        self.process_group = process_group
        self.in_features_per_rank = in_features // tp_size
        
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, self.in_features_per_rank)
        )
    
    def forward(self, x):
        # x: [batch, seq, in_features/tp]（部分输入，每个 rank 不同）
        local_out = F.linear(x, self.weight)  # [batch, seq, out_features]
        # 需要 All-Reduce 汇总各 rank 的部分结果
        torch.distributed.all_reduce(local_out, group=self.process_group)
        return local_out

class MegatronMLP(torch.nn.Module):
    """
    Megatron 风格的 MLP：
    只需要 1 次 All-Reduce（而非朴素实现的 2 次）
    """
    def __init__(self, hidden_size, ffn_size, tp_size, tp_rank, process_group):
        super().__init__()
        # W1: [hidden, ffn] → 按列切分，每 rank [hidden, ffn/tp]
        self.fc1 = ColumnParallelLinear(hidden_size, ffn_size, tp_size, tp_rank)
        # W2: [ffn, hidden] → 按行切分，每 rank [ffn/tp, hidden]
        self.fc2 = RowParallelLinear(ffn_size, hidden_size, tp_size, tp_rank, process_group)
    
    def forward(self, x):
        # Step 1: x @ W1 （每 rank 计算部分输出，无需通信）
        # x: [B, S, H]（每 rank 相同）→ h: [B, S, FFN/tp]（部分，各 rank 不同）
        h = F.gelu(self.fc1(x))
        
        # Step 2: h @ W2 + All-Reduce
        # h: [B, S, FFN/tp]（部分）→ All-Reduce → [B, S, H]（全量）
        return self.fc2(h)  # fc2 内部做 All-Reduce
```

### Self-Attention 的 TP 切分

```
Multi-Head Attention TP 切分：
按 Head 维度分割 Q, K, V, O：

Total heads = H, TP size = t → 每 rank H/t 个 head

Rank 0: QKV_proj[:, :H/t * d], O_proj[:H/t * d, :]
Rank 1: QKV_proj[:, H/t*d:H/2*d], O_proj[H/t*d:H/2*d, :]
...

通信模式：同 MLP，只在 O_proj 后做 All-Reduce
```

## Pipeline Parallel

### 模型切分

```python
"""
Pipeline Parallel 示例：4 层 Transformer，PP=2
"""
import torch

def split_model_for_pipeline(model, num_stages):
    """
    将模型层按 Pipeline 阶段切分
    """
    layers = list(model.transformer.layers)
    layers_per_stage = len(layers) // num_stages
    
    stages = []
    for i in range(num_stages):
        start = i * layers_per_stage
        end = start + layers_per_stage if i < num_stages - 1 else len(layers)
        stage_layers = layers[start:end]
        stages.append(torch.nn.Sequential(*stage_layers))
    
    return stages

# 使用示例
# Stage 0 (GPU 0-1): 层 0-15
# Stage 1 (GPU 2-3): 层 16-31（包含 lm_head）

class PipelineStage(torch.nn.Module):
    def __init__(self, layers, is_first=False, is_last=False):
        super().__init__()
        self.layers = layers
        self.is_first = is_first
        self.is_last = is_last
    
    def forward(self, x):
        # 接收上一 Stage 的输出（通过 P2P 通信）
        for layer in self.layers:
            x = layer(x)
        return x  # 发送给下一 Stage
```

### GPipe 与 1F1B 调度

**GPipe（朴素 Pipeline）**：

```
时间轴（4 microbatches, 2 stages）：

Stage 0: [F1][F2][F3][F4][    bubble    ][B4][B3][B2][B1]
Stage 1: [wait][F1][F2][F3][F4][B4][B3][B2][B1][wait]

Bubble 率 = (p-1) / (p + m - 1)
其中 p = stages数(2), m = microbatches数(4)
= 1/5 = 20%

内存：需要存 m 个 microbatch 的激活！
```

**1F1B（One Forward One Backward）调度**：

```
时间轴（4 microbatches, 2 stages）：

Stage 0: [F1][F2][F3][F4][B1][B2][B3][B4]
Stage 1:     [F1][B1][F2][B2][F3][B3][F4][B4]

Bubble 率 = (p-1) / (p + m - 1) = 20%（相同）
内存：只需存 p 个 microbatch 的激活（大幅节省！）
```

## 三维并行的通信分析

```python
"""
三维并行的通信量估算
"""

def estimate_communication(
    batch_size, seq_len, hidden_size,
    tp_size, pp_size, dp_size,
    dtype_bytes=2
):
    """估算每个训练步的通信量"""
    
    # TP 通信（每层）：2 次 All-Reduce（MLP + Attention）
    # 每次 All-Reduce 的数据量：[batch, seq, hidden]
    tp_data_per_allreduce = batch_size * seq_len * hidden_size * dtype_bytes
    # 修正：All-Reduce 的实际通信量 = 2 * (1 - 1/tp) * data
    tp_comm_per_layer = 2 * (1 - 1/tp_size) * 2 * tp_data_per_allreduce
    
    # PP 通信（每层间）：P2P 传输激活值
    pp_data_per_boundary = batch_size * seq_len * hidden_size * dtype_bytes
    pp_comm = 2 * (pp_size - 1) * pp_data_per_boundary  # 前向 + 反向
    
    # DP 通信：All-Reduce 梯度
    # 使用 ZeRO 时可以分散
    total_params = 12 * hidden_size ** 2  # 粗略估算每层参数
    dp_comm = 2 * total_params * dtype_bytes  # Reduce-Scatter + All-Gather
    
    print(f"TP 通信（每层）: {tp_comm_per_layer/1e6:.1f} MB")
    print(f"PP 通信（总）:   {pp_comm/1e6:.1f} MB")
    print(f"DP 通信（总）:   {dp_comm/1e9:.2f} GB")

estimate_communication(
    batch_size=1, seq_len=2048, hidden_size=12288,
    tp_size=8, pp_size=8, dp_size=16
)
```

## 硬件拓扑感知

Megatron 强调：**将通信量大的并行维度放在快速互联上**。

```
NVLink 带宽（A100 SXM）：600 GB/s
InfiniBand（NVLink 跨节点）：200 Gbps ≈ 25 GB/s

策略：
- TP（通信最频繁）→ 放在同一节点（NVLink）
- PP（偶尔跨阶段）→ 可以跨节点（InfiniBand OK）
- DP（每步一次梯度同步）→ 跨节点

典型配置（8 GPU/节点）：
  节点内：TP=8（使用 NVLink）
  跨节点：PP × DP
```

## 实战：使用 Megatron-LM

```bash
# Megatron-LM 训练命令示例（GPT-3 175B on 256× A100）
torchrun \
  --nproc_per_node=8 \
  --nnodes=32 \
  pretrain_gpt.py \
  --tensor-model-parallel-size 8 \   # TP=8（节点内）
  --pipeline-model-parallel-size 4 \ # PP=4（跨节点）
  # DP = 256 / (8×4) = 8
  --num-layers 96 \
  --hidden-size 12288 \
  --num-attention-heads 96 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --micro-batch-size 1 \
  --global-batch-size 1536 \
  --train-iters 500000 \
  --lr 6e-5 \
  --bf16 \
  --use-flash-attn
```

## 面试题

**Q: Megatron-LM 为什么 MLP 只需要 1 次 All-Reduce？**

A: Megatron 巧妙设计了 MLP 的切分方式：第一个矩阵（W1）按列切分，每个 rank 计算 [B,S,H]×[H,FFN/tp] 的结果，无需通信（每个 rank 的输入相同）；第二个矩阵（W2）按行切分，每个 rank 计算 [B,S,FFN/tp]×[FFN/tp,H] 的部分结果，然后通过 All-Reduce 求和得到完整输出。整个 MLP 只需要一次 All-Reduce，而朴素 TP 需要两次。Attention 也采用类似设计：QKV 按 head 维度切分（无需通信），O 投影按行切分（最后 All-Reduce），共 2 次 All-Reduce（含 Attention + MLP）。
