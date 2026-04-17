# D3：模型并行（流水线并行 + 张量并行）

> AI Infra Week2 · Day 3 | 作者：🥚🥚5号 ai infra

---

## 一、为什么需要模型并行

数据并行要求每张 GPU 存储一份完整的模型副本。当模型参数量超过单卡显存时，数据并行就失效了。

**一个简单估算**（以 GPT-3 175B 为例）：

| 项目 | 计算 | 大小 |
|------|------|------|
| 参数（FP16） | 175B × 2 bytes | ~350 GB |
| 优化器状态（Adam FP32） | 175B × 12 bytes | ~2.1 TB |
| 梯度（FP16） | 175B × 2 bytes | ~350 GB |

单张 A100 80GB 显存远不够，模型并行是唯一出路。

**模型并行的两大方向：**

```
          模型并行
         /        \
  流水线并行    张量并行
(按层切分)    (按算子切分)
  inter-op    intra-op
```

---

## 二、流水线并行（Pipeline Parallelism）

### 2.1 核心思想

将模型**按层**切分为若干阶段（Stage），每个阶段分配给不同 GPU。

```
              Transformer 12 层模型，4 张 GPU
              
  GPU 0        GPU 1        GPU 2        GPU 3
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Layer 0 │  │ Layer 3 │  │ Layer 6 │  │ Layer 9  │
│ Layer 1 │→ │ Layer 4 │→ │ Layer 7 │→ │ Layer 10 │
│ Layer 2 │  │ Layer 5 │  │ Layer 8 │  │ Layer 11 │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
     ↑                                       ↓
  输入 x                                  输出 y
```

**前向传播**：激活值从 GPU 0 → GPU 1 → GPU 2 → GPU 3 流动（Send/Recv）
**后向传播**：梯度从 GPU 3 → GPU 2 → GPU 1 → GPU 0 流动

### 2.2 朴素流水线的问题：Bubble

最简单的实现是"F-then-B"，但存在严重的**流水线气泡（Bubble）**：

```
时间轴 →

GPU 0: [F0][F0][F0][F0]      [B0][B0][B0][B0]
GPU 1:      [F1]      ...         [B1]
GPU 2:           [F2]...               [B2]
GPU 3:                [F3]                 [B3]

█ = 有效计算
空白 = GPU 空闲（气泡！）

气泡比例 ≈ (num_stages - 1) / num_microbatches
```

气泡越大，GPU 利用率越低。4 个 Stage、1 个 microbatch 时，GPU 0 有 75% 时间在等待。

### 2.3 GPipe：Microbatch 切分

GPipe（Google, 2019）通过将 mini-batch 切分为 **microbatch** 来减少气泡：

```
将 batch 切成 4 个 microbatch (m1, m2, m3, m4)

时间轴 →
GPU 0: [F,m1][F,m2][F,m3][F,m4]          [B,m4][B,m3][B,m2][B,m1]
GPU 1:       [F,m1][F,m2][F,m3][F,m4]    [B,m4][B,m3][B,m2][B,m1]
GPU 2:             [F,m1][F,m2][F,m3][F,m4]    [B,m4]...
GPU 3:                   [F,m1][F,m2][F,m3][F,m4]    [B,m4]...

气泡比例 = (p-1) / (p-1+m) ≈ (p-1)/m （m 越大，气泡越小）
```

**代价**：需要存储所有 microbatch 的激活值，显存开销大。可配合 **Gradient Checkpointing**（重计算激活值）缓解。

### 2.4 1F1B（PipeDream）：更高效的调度

PipeDream 提出的 **1F1B（One Forward One Backward）**调度策略：

```
时间轴 →
GPU 0: [F1][F2][F3][F4][B1][F5][B2][F6][B3][F7][B4]...
GPU 1:     [F1][F2][F3][F4][B1][F5][B2][F6][B3]...
GPU 2:         [F1][F2][F3][F4][B1][F5][B2]...
GPU 3:             [F1][F2][F3][F4][B1]...

F = 前向   B = 后向
```

稳态时每个 GPU 交替进行前向和后向，气泡比例下降到 `(p-1)/(m)` 且显存更友好。

**Interleaved 1F1B（Megatron-LM v2）**：每张 GPU 分配多个不连续的层块，进一步减少气泡：

```
# 原始分配（连续）：
GPU 0: Layers 0,1,2   GPU 1: Layers 3,4,5

# Interleaved 分配（各 GPU 分 2 个块）：
GPU 0: Layers 0,1 + Layers 6,7
GPU 1: Layers 2,3 + Layers 8,9

气泡比例减少约 chunks_per_device 倍
```

### 2.5 流水线并行代码（简化示例）

```python
import torch
import torch.nn as nn
import torch.distributed as dist

# ──── 模型切分 ────────────────────────────────────────────

class Stage0(nn.Module):
    """流水线第 0 阶段（GPU 0 运行）"""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32000, 1024)
        self.layers = nn.ModuleList([TransformerBlock(1024) for _ in range(3)])

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


class Stage1(nn.Module):
    """流水线第 1 阶段（GPU 1 运行）"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(1024) for _ in range(3)])
        self.lm_head = nn.Linear(1024, 32000)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# ──── 流水线通信（简化的 F-then-B）────────────────────────

def pipeline_forward_backward(stage, local_rank, microbatches):
    """
    简化的流水线前向 + 后向（2 个 Stage）
    实际生产使用 torchgpipe 或 Megatron-LM 的实现
    """
    if local_rank == 0:
        # Stage 0：计算前向，发送给 Stage 1
        outputs = []
        for mb in microbatches:
            out = stage(mb)
            dist.send(out, dst=1)
            outputs.append(out)
        # 接收来自 Stage 1 的梯度
        for out in outputs:
            grad = torch.zeros_like(out)
            dist.recv(grad, src=1)
            out.backward(grad)

    elif local_rank == 1:
        # Stage 1：接收激活值，计算前向 + 后向，发送梯度
        losses = []
        activations = []
        for mb in microbatches:
            act = torch.zeros(mb.shape[0], mb.shape[1], 1024)
            dist.recv(act, src=0)
            act.requires_grad_(True)
            activations.append(act)

            out = stage(act)
            loss = criterion(out, labels)
            losses.append(loss)

        total_loss = sum(losses)
        total_loss.backward()

        for act in activations:
            dist.send(act.grad, dst=0)
```

---

## 三、张量并行（Tensor Parallelism）

张量并行（TP）在**算子层面**切分参数，不同 GPU 各自计算矩阵乘法的一部分，再通过 AllReduce 合并结果。

### 3.1 矩阵乘法的两种切分方式

**场景**：计算 `Y = X @ W`，其中 `X`（输入）形状 `[B, T, d_in]`，`W`（权重）形状 `[d_in, d_out]`。

#### 方式一：按列切分 W（Column Parallel）

```
W = [W_1 | W_2]   （按列切成两块，各分到 GPU 0, GPU 1）

GPU 0: Y_1 = X @ W_1   [B, T, d_out/2]
GPU 1: Y_2 = X @ W_2   [B, T, d_out/2]

结果：Y = [Y_1, Y_2]（AllGather 拼接，或直接留给下一层）
```

#### 方式二：按行切分 W（Row Parallel）

```
W = [W_1]   （按行切成两块）
    [W_2]

X 也按列切分：
GPU 0: X_1 = X[:, :, :d_in/2],   Y_1 = X_1 @ W_1
GPU 1: X_2 = X[:, :, d_in/2:],   Y_2 = X_2 @ W_2

结果：Y = Y_1 + Y_2（AllReduce 求和）
```

### 3.2 Megatron-LM 的 Transformer 张量并行方案

Megatron-LM 经典方案：**列并行 + 行并行**配对，减少通信次数到 2 次/层（前向和后向各 1 次 AllReduce）。

```
             输入 X [B, T, d]
                   ↓
          ┌────────────────────────┐
          │    Attention Layer     │
          │  Q,K,V Linear（列并行）│
          │  GPU 0: W_Q1, W_K1, W_V1 │
          │  GPU 1: W_Q2, W_K2, W_V2 │
          │                          │
          │  各 GPU 独立计算 Attention│
          │                          │
          │  Output Linear（行并行）  │
          │  → AllReduce ←           │
          └────────────────────────┘
                   ↓ AllReduce
          ┌────────────────────────┐
          │      FFN Layer         │
          │  FC1（列并行）          │
          │  FC2（行并行）          │
          │  → AllReduce ←         │
          └────────────────────────┘
                   ↓
             输出 [B, T, d]
```

**每层只需 2 次 AllReduce**（前向 1 次 + 后向 1 次，共 2 × 2 = 4 次实际通信），通信量为 `O(B × T × d)`，与批次大小成正比。

### 3.3 张量并行的 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ProcessGroup


class ColumnParallelLinear(nn.Module):
    """
    列并行线性层：
    - 权重按列切分，各 GPU 计算 Y_i = X @ W_i
    - 若 gather_output=True，最终 AllGather 合并
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_group: ProcessGroup,
        gather_output: bool = True,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.gather_output = gather_output

        # 每个 GPU 只持有 out_features / tp_size 列
        local_out = out_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(local_out, in_features))
        self.bias = nn.Parameter(torch.zeros(local_out))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 本地矩阵乘：[B, T, in] @ [out/tp, in]^T = [B, T, out/tp]
        out = torch.nn.functional.linear(x, self.weight, self.bias)

        if self.gather_output:
            # 收集所有 GPU 的输出并拼接
            out_list = [torch.zeros_like(out) for _ in range(self.tp_size)]
            dist.all_gather(out_list, out, group=self.tp_group)
            out = torch.cat(out_list, dim=-1)

        return out


class RowParallelLinear(nn.Module):
    """
    行并行线性层：
    - 权重按行切分，各 GPU 计算 Y_i = X_i @ W_i
    - 最终 AllReduce 求和
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_group: ProcessGroup,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.input_is_parallel = input_is_parallel

        local_in = in_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(out_features, local_in))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            # 输入未切分，手动切分
            rank = dist.get_rank(self.tp_group)
            local_in = x.shape[-1] // self.tp_size
            x = x[..., rank * local_in:(rank + 1) * local_in]

        # 本地矩阵乘
        out = torch.nn.functional.linear(x, self.weight, bias=None)

        # AllReduce 合并各 GPU 的部分和
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)

        return out + self.bias


class TensorParallelAttention(nn.Module):
    """张量并行的多头注意力"""
    def __init__(self, d_model: int, num_heads: int, tp_group: ProcessGroup):
        super().__init__()
        self.tp_size = dist.get_world_size(tp_group)
        # 每个 GPU 处理 num_heads / tp_size 个头
        self.local_heads = num_heads // self.tp_size
        self.head_dim = d_model // num_heads

        # QKV：列并行（每 GPU 各算自己的头）
        self.qkv_proj = ColumnParallelLinear(
            d_model, 3 * d_model, tp_group, gather_output=False
        )
        # 输出投影：行并行（求和得到完整输出）
        self.out_proj = RowParallelLinear(d_model, d_model, tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        qkv = self.qkv_proj(x)   # [B, T, 3*d/tp]
        q, k, v = qkv.chunk(3, dim=-1)

        # 本地 Attention 计算（只处理自己分配的头）
        q = q.view(B, T, self.local_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.local_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.local_heads, self.head_dim).transpose(1, 2)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)

        # 输出投影 + AllReduce
        return self.out_proj(attn)
```

---

## 四、流水线并行 vs 张量并行

| 维度 | 流水线并行 | 张量并行 |
|------|-----------|---------|
| 切分维度 | 按模型层 | 按算子/矩阵 |
| 通信原语 | Send/Recv（点对点） | AllReduce（集合） |
| 通信量 | 激活值大小（较小） | 激活值 × d_model（较大） |
| 适合拓扑 | 跨节点（IB）也可 | 最好节点内（NVLink） |
| 气泡问题 | 存在，需 microbatch | 无气泡 |
| 实现复杂度 | 中等 | 较复杂 |

**实践建议：**
- 张量并行倾向于**节点内**（受益于 NVLink 高带宽）
- 流水线并行跨**节点间**（IB 带宽下每层激活值传输量可接受）
- 大模型通常**TP × PP × DP 三维并行**组合

```
示例：64 张 GPU，TP=4, PP=4, DP=4

DP 维度：4 个数据并行副本
每个副本 = TP×PP = 16 张 GPU
  - 节点内 4 张 GPU 做 TP（NVLink）
  - 4 个节点串联做 PP（IB）
```

---

## 五、Sequence Parallelism（序列并行）

对于超长序列（如 128K tokens），注意力计算的显存也会爆炸（`O(T²)` 的注意力矩阵）。

Megatron-LM 引入的**序列并行**：

```
  LayerNorm 和 Dropout 也按序列维度切分：

  TP 内各 GPU 在序列维度各持有 T/tp 个 token
  ──────────────────────────────────────────
  GPU 0: token[0:T/4]   → LayerNorm → Attention（列并行）
  GPU 1: token[T/4:T/2] → LayerNorm → Attention（列并行）
  ...

  Attention 后 → ReduceScatter → 行并行 → AllGather
```

序列并行将激活值的显存也均摊到 TP GPU 上，激活值显存降为 `1/tp`。

---

## 六、使用 PyTorch RPC 的简单流水线示例

```python
import torch.distributed.rpc as rpc

def run_pipeline_stage(stage_model, inputs):
    """在远程 GPU 上运行一个流水线阶段"""
    return stage_model(inputs)

# 主进程 (rank 0) 控制流水线
if rank == 0:
    x = get_input_batch()
    # 第一阶段：本地运行
    h = stage_0_model(x)
    # 第二阶段：远程运行（异步）
    fut = rpc.rpc_async(
        "worker1",
        run_pipeline_stage,
        args=(rpc.remote("worker1", Stage1), h)
    )
    output = fut.wait()
    loss = criterion(output, labels)
```

---

## 七、本章小结

| 知识点 | 核心内容 |
|--------|---------|
| 流水线并行 | 按层切分，激活值通过 Send/Recv 流动；Bubble 是关键问题 |
| GPipe | Microbatch 减少气泡，但显存大 |
| 1F1B | 稳态交替前后向，显存友好 |
| 张量并行 | 矩阵按列/行切分，每层 2 次 AllReduce |
| Megatron-LM | 列并行+行并行配对，Transformer 标准 TP 实现 |
| 三维并行 | TP（节点内）× PP（跨节点）× DP（数据复制） |

**下一篇（D4）**将深入 ZeRO 三个阶段的内存优化原理，揭示 DeepSpeed 如何把千亿参数模型的显存开销降低 8 倍以上。
