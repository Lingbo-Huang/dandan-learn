# D2：数据并行原理与 PyTorch DDP 实现

> AI Infra Week2 · Day 2 | 作者：🥚🥚5号 ai infra

---

## 一、数据并行的核心思想

数据并行（Data Parallelism, DP）是分布式训练中最直观、应用最广泛的策略。其核心思想是：

> 每张 GPU 保存一份**完整的模型副本**，不同 GPU 处理不同的数据子集（mini-batch 的不同切片），各自完成前向和后向传播，然后通过通信原语**同步梯度**，保证所有副本的参数始终一致。

```
┌──────────────────────────────────────────────────────────┐
│                     数据并行流程                           │
│                                                          │
│  全局 Batch                                              │
│  [x1 x2 x3 x4 x5 x6 x7 x8]                             │
│       ↓           ↓                                      │
│  [x1 x2 x3 x4]  [x5 x6 x7 x8]                         │
│       ↓                ↓                                 │
│  GPU 0 (副本 0)   GPU 1 (副本 1)                        │
│  前向 + 后向      前向 + 后向                             │
│  grad_0           grad_1                                 │
│       ↓                ↓                                 │
│      AllReduce(grad_0 + grad_1) / 2                     │
│       ↓                ↓                                 │
│  更新参数         更新参数  (完全相同)                    │
└──────────────────────────────────────────────────────────┘
```

**数学等价性**：对全局 batch 做梯度下降，等价于对各子 batch 求梯度后取均值（mini-batch SGD 的线性可加性）：

$$\nabla L_{global} = \frac{1}{N} \sum_{i=0}^{N-1} \nabla L_i$$

---

## 二、PyTorch 的两种 DP 实现对比

### 2.1 DataParallel（DP）—— 旧方案，不推荐

`torch.nn.DataParallel` 是 PyTorch 最早的多 GPU 支持方式，**单进程多线程**，存在严重缺陷：

```python
# 旧方式（不推荐）
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

**架构图：**

```
    数据
     ↓
  GPU 0（主卡）
  ┌───────────┐
  │ 数据切分   │→ GPU1, GPU2, GPU3（scatter）
  │ 模型复制   │→ GPU1, GPU2, GPU3（broadcast）
  │ 并行前向   │← 从各 GPU 收集输出（gather）
  │ Loss 计算  │  (全在主卡上)
  │ 后向传播   │  (主卡 + 从卡各做一部分)
  │ 梯度收集   │← gather 梯度到主卡
  │ 参数更新   │  (主卡更新，再 broadcast)
  └───────────┘
```

**缺陷：**
- 主卡（GPU 0）负载远高于其他卡（负责 Loss 计算、梯度汇总），造成**负载不均衡**
- 单进程受 Python GIL 限制，无法充分并行
- 每个 step 都要 broadcast 模型权重，通信量大
- 不支持多机训练

### 2.2 DistributedDataParallel（DDP）—— 当前标准

`torch.nn.parallel.DistributedDataParallel` 是当前推荐方案，**多进程**，每个进程独立运行，通过 NCCL 进行 AllReduce 通信。

```
进程 0 (GPU 0)        进程 1 (GPU 1)
┌──────────────┐      ┌──────────────┐
│ 完整模型副本  │      │ 完整模型副本  │
│ 数据 batch_0 │      │ 数据 batch_1 │
│ 前向传播      │      │ 前向传播      │
│ 后向传播      │      │ 后向传播      │
│  grad_0       │◄────►│  grad_1       │
│     AllReduce(avg_grad)             │
│ optimizer.step│      │ optimizer.step│
└──────────────┘      └──────────────┘
```

**优势：**
- 每个进程绑定一个 GPU，无 GIL 瓶颈
- 梯度同步与后向传播**重叠**（计算通信 overlap）
- 支持多机多卡
- 无主卡负载不均问题

---

## 三、DDP 的 Bucket 梯度同步机制

DDP 的高效关键在于 **Bucket AllReduce** 机制，让通信与计算尽可能重叠。

### 3.1 Bucket 分配

DDP 在初始化时将模型参数分成若干个 Bucket（默认每个 Bucket 25MB）：

```
模型参数（从后往前注册）：
  layer_norm.bias  →  ┐
  layer_norm.weight →  ├── Bucket 0 (25MB)
  ffn.fc2.bias     →  ┘
  ffn.fc2.weight   →  ┐
  ffn.fc1.bias     →  ├── Bucket 1 (25MB)
  ffn.fc1.weight   →  │
  ...              →  ┘
```

**为什么从后往前？** 后向传播从最后一层开始计算梯度，先完成的梯度先触发 AllReduce，实现流水线化。

### 3.2 梯度钩子（Hook）

DDP 在每个参数上注册一个 `grad_accumulator_hook`：

```python
# DDP 内部伪代码（简化）
def make_hook(bucket):
    def hook(grad):
        bucket.add_grad(grad)
        if bucket.is_ready():  # 所有参数梯度就绪
            bucket.all_reduce()  # 异步发起 AllReduce
    return hook

for param in model.parameters():
    param.register_post_accumulate_grad_hook(make_hook(assigned_bucket))
```

### 3.3 执行时序

```
时间线（反向传播阶段）：

GPU 0:
  [BWD layer_n] → [BWD layer_n-1] → [BWD layer_n-2] → ...
                       ↓触发 bucket ready
                   [AllReduce bucket_0] 异步启动

GPU 1:
  [BWD layer_n] → [BWD layer_n-1] → [BWD layer_n-2] → ...
                       ↓
                   [AllReduce bucket_0] 

结果：AllReduce 与后续层的后向传播同时进行！
```

### 3.4 梯度压缩（可选）

对于带宽受限场景，可启用梯度压缩（如 PowerSGD）：

```python
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD

model = DDP(model, device_ids=[local_rank])
state = powerSGD.PowerSGDState(
    process_group=None,
    matrix_approximation_rank=1,
    start_powerSGD_iter=1000,
)
model.register_comm_hook(state, powerSGD.powerSGD_hook)
```

---

## 四、完整 DDP 训练代码

### 4.1 训练脚本

```python
# train_ddp.py
import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class SimpleTransformerBlock(nn.Module):
    """简化的 Transformer 块，用于演示"""
    def __init__(self, d_model=512, nhead=8, dim_ffn=2048):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.GELU(),
            nn.Linear(dim_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class FakeTextDataset(Dataset):
    """模拟文本数据集"""
    def __init__(self, num_samples=1000, seq_len=128, vocab_size=32000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def setup_distributed():
    """初始化分布式进程组"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def train(args):
    local_rank = setup_distributed()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        print(f"Training with {world_size} GPUs")

    # ── 数据集与采样器 ──────────────────────────────────────
    dataset = FakeTextDataset(num_samples=10000, seq_len=128)

    # DistributedSampler 确保各进程拿到不重叠的数据子集
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        seed=42,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # ── 模型 ────────────────────────────────────────────────
    model = nn.Sequential(
        nn.Embedding(32000, 512),
        *[SimpleTransformerBlock(512, 8, 2048) for _ in range(6)],
        nn.Linear(512, 32000),
    ).to(device)

    # DDP 包装：find_unused_parameters=False 性能更好（明确没有未用参数时）
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,  # 减少内存拷贝
    )

    # ── 优化器 ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # ── 训练循环 ────────────────────────────────────────────
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # 每 epoch 重新 shuffle，避免重复
        model.train()

        total_loss = 0.0
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 前向传播
            logits = model(input_ids)  # [B, T, vocab_size]
            loss = criterion(
                logits.view(-1, 32000),
                labels.view(-1),
            )

            # 后向传播（DDP 自动触发 AllReduce）
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            if global_rank == 0 and step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

        if global_rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")

    # ── 保存模型 ─────────────────────────────────────────────
    if global_rank == 0:
        # 只在 Rank 0 保存，model.module 是未被 DDP 包装的原始模型
        torch.save(model.module.state_dict(), "checkpoint.pt")
        print("Model saved.")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
```

### 4.2 启动命令

```bash
# 单机 4 卡
torchrun \
  --standalone \
  --nproc_per_node=4 \
  train_ddp.py \
  --epochs 10 \
  --batch_size 32

# 2 机 8 卡（每机 4 卡）
# 节点 0：
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --master_addr="192.168.1.100" \
  --master_port=29500 \
  train_ddp.py

# 节点 1：
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=1 \
  --master_addr="192.168.1.100" \
  --master_port=29500 \
  train_ddp.py
```

---

## 五、DistributedSampler 的重要性

不用 DistributedSampler 会发生什么？

```
错误做法：
  GPU 0 和 GPU 1 各自用完整数据集的随机 shuffle
  → 大概率出现数据重叠 → 等效于只用了 1 倍数据量
  → 训练收敛更快但泛化可能差（过度拟合）

正确做法（DistributedSampler）：
  每个 GPU 拿到数据集不重叠的 1/N 子集
  → 合并等价于完整遍历 1 epoch
  → 真正的线性 scaling
```

**每 epoch 必须调用 `sampler.set_epoch(epoch)`**，否则每个 epoch 的数据分配相同，无法随机化：

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # ← 必须！
    for batch in dataloader:
        ...
```

---

## 六、梯度同步与 no_sync 上下文

当使用梯度累积（Gradient Accumulation）时，中间步骤不需要做 AllReduce，只在最后一步同步：

```python
accumulation_steps = 4

for step, batch in enumerate(dataloader):
    # 前 3 步：不同步梯度（节省通信）
    is_last_step = (step + 1) % accumulation_steps == 0

    if not is_last_step:
        with model.no_sync():   # 禁用自动 AllReduce
            loss = model(batch)
            (loss / accumulation_steps).backward()
    else:
        loss = model(batch)
        (loss / accumulation_steps).backward()
        optimizer.step()        # 这一步的 backward 触发 AllReduce
        optimizer.zero_grad()
```

等效全局 batch_size = `batch_size × world_size × accumulation_steps`

---

## 七、性能调优 Checklist

| 优化项 | 方法 | 预期收益 |
|--------|------|---------|
| 关闭 find_unused_parameters | `find_unused_parameters=False` | 减少开销 5-15% |
| 开启 gradient_as_bucket_view | `gradient_as_bucket_view=True` | 减少内存拷贝 |
| 增大 Bucket 大小 | `bucket_cap_mb=50` | 减少通信次数 |
| 使用 pin_memory | `DataLoader(pin_memory=True)` | 加速 CPU-GPU 传输 |
| 合理设置 num_workers | `num_workers=4`（通常 = CPU核数/4） | 避免数据加载瓶颈 |
| 混合精度 | `torch.cuda.amp.autocast()` | 显存减半，速度 1.5-2x |
| 梯度裁剪 | `clip_grad_norm_(1.0)` | 稳定训练 |

### 调试常见问题

```python
# 问题：DDP 挂起（hang），所有进程卡住
# 原因：某个进程在 AllReduce 时缺少参数梯度（如 if 分支导致某些参数未参与前向）
# 解决：设置 find_unused_parameters=True（但有性能开销）

# 问题：训练结果不可复现
# 解决：设置随机种子
import random
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

---

## 八、FSDP：DDP 的下一代

`FullyShardedDataParallel (FSDP)` 是 PyTorch 1.12+ 引入的新并行方案，在 DDP 基础上加入了 ZeRO 思想的参数分片：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 等价于 ZeRO Stage 3
    device_id=local_rank,
)
```

| | DDP | FSDP |
|-|-----|------|
| 参数存储 | 每卡完整副本 | 分片存储 |
| 显存消耗 | 高 | 低（约 1/N） |
| 通信开销 | AllReduce | AllGather + ReduceScatter |
| 适用规模 | 数十亿参数内 | 数百亿参数 |

---

## 九、本章小结

数据并行是分布式训练的基石：
1. **DDP 是标准**，DP 已过时
2. **Bucket AllReduce** 实现计算通信重叠，是 DDP 高效的核心
3. **DistributedSampler** 保证数据不重叠
4. **no_sync** 配合梯度累积减少通信频次
5. **FSDP** 是大模型时代 DDP 的进化版

**下一篇（D3）**将讲解模型并行策略，包括流水线并行和张量并行的原理与实现。
