# D1：分布式训练全景与通信原语

> AI Infra Week2 · Day 1 | 作者：🥚🥚5号 ai infra

---

## 一、为什么需要分布式训练

现代大模型的参数量已从亿级跃升至千亿乃至万亿级。GPT-3 拥有 1750 亿参数，单个 A100 GPU（80GB 显存）无法容纳，训练一次需要数千张 GPU 同时工作。即便是"小一些"的 7B 模型，以 FP32 精度存储也需要 28GB，加上优化器状态（Adam 需要保存一阶与二阶矩）则轻松超过 80GB。

分布式训练的核心矛盾是：

| 挑战 | 具体表现 |
|------|---------|
| **显存不足** | 模型参数 + 优化器状态 + 激活值超出单卡显存 |
| **计算太慢** | 单卡吞吐量跟不上训练时效需求 |
| **数据量大** | 数据加载与预处理成为瓶颈 |

分布式训练通过多种并行策略（数据并行、模型并行、流水线并行等）将这三类压力分摊到多张 GPU 或多台机器上。

---

## 二、分布式训练的整体架构

### 2.1 硬件拓扑

```
┌─────────────────────────────────────────────┐
│                  数据中心                     │
│  ┌───────────────────────────────────────┐   │
│  │  节点 0 (Node 0)                      │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │   │
│  │  │ GPU0 │ │ GPU1 │ │ GPU2 │ │ GPU3 │ │   │
│  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ │   │
│  │     └────────┴─────────┴────────┘      │   │
│  │          NVLink / NVSwitch              │   │
│  └───────────────────┬───────────────────┘   │
│                       │ InfiniBand / RoCE     │
│  ┌───────────────────┴───────────────────┐   │
│  │  节点 1 (Node 1)                      │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │   │
│  │  │ GPU4 │ │ GPU5 │ │ GPU6 │ │ GPU7 │ │   │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ │   │
│  └───────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**关键带宽数字：**
- NVLink 3.0：双向 600 GB/s（单对 GPU）
- NVSwitch（H100 SXM）：节点内 all-to-all 带宽 900 GB/s
- InfiniBand HDR：200 Gbps（≈25 GB/s）
- PCIe 4.0 x16：约 32 GB/s（老拓扑，性能瓶颈）

节点内通信走 NVLink（快），节点间通信走 IB（慢），因此分布式算法设计要尽量减少跨节点通信量。

### 2.2 软件栈概览

```
┌──────────────────────────────────────┐
│         训练框架 (PyTorch/JAX)        │
├──────────────────────────────────────┤
│   并行库 (DDP / FSDP / DeepSpeed)    │
├──────────────────────────────────────┤
│       通信库 (NCCL / Gloo / MPI)     │
├──────────────────────────────────────┤
│    底层驱动 (CUDA / ROCm / cuDNN)    │
└──────────────────────────────────────┘
```

- **NCCL**（NVIDIA Collective Communications Library）：GPU 间集合通信的事实标准，支持 AllReduce、AllGather、ReduceScatter 等原语，自动选择 NVLink 或 IB 路径
- **Gloo**：CPU 端集合通信，DDP 的备选后端
- **MPI**：老牌高性能计算通信库，仍用于部分超算集群

---

## 三、并行策略大图

```
                    ┌─────────────────────┐
                    │      分布式并行       │
                    └──────────┬──────────┘
           ┌──────────────────┼──────────────────┐
    ┌──────┴──────┐  ┌────────┴───────┐  ┌───────┴───────┐
    │  数据并行   │  │   模型并行      │  │  流水线并行    │
    │ (DP/DDP)   │  │ (Tensor Par.)  │  │ (Pipeline Par.)│
    └──────┬──────┘  └────────────────┘  └───────────────┘
           │
    ┌──────┴──────────┐
    │  ZeRO 优化器     │
    │ (Stage 1/2/3)   │
    └─────────────────┘
```

各策略解决的核心问题：

| 策略 | 解决问题 | 代表实现 |
|------|---------|---------|
| 数据并行 | 吞吐量不足 | PyTorch DDP |
| 张量并行 | 单层参数过大 | Megatron-LM |
| 流水线并行 | 模型层数太多 | GPipe / PipeDream |
| ZeRO | 优化器/梯度显存占用 | DeepSpeed |
| 混合并行 | 综合所有瓶颈 | DeepSpeed + Megatron |

---

## 四、集合通信原语详解

集合通信（Collective Communication）是分布式训练的基础设施，理解这些原语是读懂任何分布式代码的前提。

假设有 `N` 个进程（Rank 0 ~ N-1），每个进程持有一段数据 `x_i`（大小为 `M` 字节）。

### 4.1 Broadcast

**语义**：Rank 0 将自己的数据广播给所有其他 Rank。

```
Before:
  Rank 0: [A]
  Rank 1: [ ]
  Rank 2: [ ]

After:
  Rank 0: [A]
  Rank 1: [A]
  Rank 2: [A]
```

**用途**：训练开始时将模型初始参数从 Rank 0 同步到所有 GPU。

**复杂度**：传输量 `O(M)`，时间 `O(log N × α + M/β)`（树形广播）

### 4.2 Reduce

**语义**：将所有 Rank 的数据聚合（如求和）到 Rank 0。

```
Before:
  Rank 0: [1]  Rank 1: [2]  Rank 2: [3]

After (sum):
  Rank 0: [6]
  Rank 1: (unchanged)
  Rank 2: (unchanged)
```

**用途**：梯度汇总到参数服务器（Parameter Server 架构）。

### 4.3 AllReduce（最核心！）

**语义**：所有 Rank 的数据聚合后，结果广播回所有 Rank。相当于 Reduce + Broadcast。

```
Before:
  Rank 0: [1]  Rank 1: [2]  Rank 2: [3]

After (sum):
  Rank 0: [6]
  Rank 1: [6]
  Rank 2: [6]
```

**用途**：DDP 梯度同步的核心操作。每个 GPU 计算出本地梯度后，通过 AllReduce 得到全局平均梯度。

**Ring AllReduce 算法**（NCCL 实际使用的高效实现）：

```
步骤一：ReduceScatter（N-1 轮）
  将数据分成 N 块，环形传递并累加
  每个 Rank 最终持有全局累加结果的 1/N

步骤二：AllGather（N-1 轮）
  每个 Rank 将自己持有的那块广播给所有人

总通信量：2 × (N-1)/N × M ≈ 2M（与 N 无关！）
```

Ring AllReduce 的精妙之处：无论多少个 GPU，每个 GPU 发送和接收的数据总量固定为约 `2M`，实现了带宽最优。

### 4.4 AllGather

**语义**：每个 Rank 将自己的数据发送给所有 Rank，最终每个 Rank 拥有所有人的数据。

```
Before:
  Rank 0: [A]  Rank 1: [B]  Rank 2: [C]

After:
  Rank 0: [A, B, C]
  Rank 1: [A, B, C]
  Rank 2: [A, B, C]
```

**用途**：ZeRO Stage 3 中在前向传播前收集分片的模型参数。

**通信量**：`(N-1) × M`

### 4.5 ReduceScatter

**语义**：AllReduce 的前半段。先对所有 Rank 的数据求和，然后将结果的第 `i` 块分发给 Rank `i`。

```
Before:
  Rank 0: [1, 2]  Rank 1: [3, 4]  Rank 2: [5, 6]

After (sum, scatter):
  Rank 0: [9]      (1+3+5)
  Rank 1: [12]     (2+4+6)
  Rank 2: (nothing left)
```

**用途**：ZeRO 优化器、Megatron 张量并行输出聚合。

### 4.6 Send / Recv（点对点）

流水线并行中，相邻流水线阶段之间通过 `Send/Recv` 传递激活值和梯度，而不用 Collective 操作。

---

## 五、进程组与通信域

NCCL 和 PyTorch 都允许创建**进程子组**（ProcessGroup），让不同的通信原语在不同的 GPU 子集上执行：

```python
import torch.distributed as dist

# 初始化全局进程组
dist.init_process_group(backend='nccl')

# 创建子组：0,1 号 GPU 做张量并行
tensor_parallel_group = dist.new_group(ranks=[0, 1])

# 创建子组：0,2 号 GPU 做数据并行
data_parallel_group = dist.new_group(ranks=[0, 2])

# 在子组内做 AllReduce
dist.all_reduce(tensor, group=tensor_parallel_group)
```

混合并行（如 Megatron-DeepSpeed）正是通过多级进程组实现：
- **TP Group**：同一流水线阶段内做张量并行的 GPU
- **PP Group**：不同流水线阶段的 GPU
- **DP Group**：完整 replica 间做数据并行的 GPU

---

## 六、分布式训练的启动方式

### 6.1 torchrun（推荐）

```bash
# 单机 4 卡
torchrun --nproc_per_node=4 train.py

# 多机 2 节点，每节点 4 卡
# 节点 0（主节点）：
torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr="10.0.0.1" \
  --master_port=29500 \
  --nproc_per_node=4 \
  train.py

# 节点 1：
torchrun \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr="10.0.0.1" \
  --master_port=29500 \
  --nproc_per_node=4 \
  train.py
```

`torchrun` 会自动设置以下环境变量，训练脚本通过这些变量感知自身位置：

| 环境变量 | 含义 |
|---------|------|
| `RANK` | 全局进程编号（0 到 world_size-1） |
| `LOCAL_RANK` | 本节点内的 GPU 编号 |
| `WORLD_SIZE` | 全局总进程数 |
| `MASTER_ADDR` | 主节点 IP |
| `MASTER_PORT` | 主节点端口（用于 rendezvous） |

### 6.2 deepspeed 启动器

```bash
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

### 6.3 SLURM 集群

大规模集群通常用 SLURM 作业调度器，配合 `srun` 启动：

```bash
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=8 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py
```

---

## 七、一个最简分布式训练骨架

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    """初始化进程组"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def train():
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    # 模型放到对应 GPU
    model = MyModel().to(device)
    # 用 DDP 包装
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch["input_ids"].to(device))
            loss = outputs.loss
            loss.backward()           # DDP 在这里自动 AllReduce 梯度
            optimizer.step()

    cleanup()

if __name__ == "__main__":
    train()
```

---

## 八、性能分析与调优思路

### 8.1 通信开销分析

分布式训练的效率取决于**计算与通信的重叠程度**。DDP 采用 Bucket 机制：
- 将梯度分桶（默认 25MB/桶）
- 某个桶内所有梯度就绪后，立即异步发起 AllReduce
- 后向传播继续执行同时通信进行——**计算通信重叠**

如果通信无法隐藏在计算后面，效率就会下降：
```
理想：[FWD][BWD][  comm  ]
                ↑ 与 BWD 重叠

现实（通信瓶颈）：[FWD][BWD][wait...comm]
```

### 8.2 扩展效率（Scaling Efficiency）

```
强扩展效率 = (单卡时间) / (N 卡时间 × N) × 100%

理想：100%（线性加速）
实际：随 N 增大而下降（通信开销 + 同步等待）
```

业界水平参考（A100 集群）：
- 64 卡内：>90% 可实现
- 1024 卡：通常 70-80%
- 更大规模：通信拓扑设计成为关键

---

## 九、本章小结

| 知识点 | 核心内容 |
|--------|---------|
| 为什么需要分布式 | 模型/数据规模超出单卡能力 |
| 硬件拓扑 | 节点内 NVLink 快，节点间 IB 慢，算法设计需考虑拓扑 |
| 并行策略 | DP/TP/PP/ZeRO，各解决不同瓶颈 |
| AllReduce | DDP 的核心原语，Ring 实现带宽最优 |
| AllGather/ReduceScatter | ZeRO 和张量并行的基础操作 |
| 进程组 | 多级并行通过子组实现独立通信域 |
| 启动方式 | torchrun / deepspeed / SLURM |

**下一篇（D2）**将深入数据并行的实现细节，包括 PyTorch DDP 的 Bucket 梯度同步机制与实战代码。
