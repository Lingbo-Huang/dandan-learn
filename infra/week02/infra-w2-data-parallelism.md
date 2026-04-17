# 数据并行：从 DP 到 DDP

> **Week 2 · Day 2**  
> 目标：掌握数据并行的原理与 PyTorch DDP 实战

---

## 一、数据并行的核心思想

### 1.1 原理图解

```
数据并行训练流程：

全局 Batch (B=128)
      │
      ├─ GPU 0: mini-batch [0:32]   ─→ Forward ─→ Loss ─→ Backward ─→ grad_0
      ├─ GPU 1: mini-batch [32:64]  ─→ Forward ─→ Loss ─→ Backward ─→ grad_1
      ├─ GPU 2: mini-batch [64:96]  ─→ Forward ─→ Loss ─→ Backward ─→ grad_2
      └─ GPU 3: mini-batch [96:128] ─→ Forward ─→ Loss ─→ Backward ─→ grad_3
                                                                          │
                                               All-Reduce (梯度求平均) ───┘
                                                          │
                                              每张 GPU 用平均梯度更新参数
                                              (所有卡参数保持一致)
```

**效果**：等价于用 batch size=128 训练，但4张卡并行，速度约快4倍。

### 1.2 DP vs DDP

```
DataParallel (DP) — 老方式，不推荐：
  ┌──────────────────────────────────────────────┐
  │  GPU 0 (主卡):                               │
  │    1. 接收完整 batch                          │
  │    2. 分发数据到 GPU 1/2/3                   │
  │    3. 收集各卡梯度（GPU 0 做 reduce）         │
  │    4. 更新参数后广播到其他卡                  │
  │                                              │
  │  问题：GPU 0 是瓶颈；GIL 锁限制多线程性能     │
  └──────────────────────────────────────────────┘

DistributedDataParallel (DDP) — 新方式，推荐：
  ┌──────────────────────────────────────────────┐
  │  每张 GPU 独立进程                            │
  │    1. 各自加载数据（DistributedSampler）      │
  │    2. 各自前向/反向                           │
  │    3. Ring All-Reduce 同步梯度（无主卡瓶颈）  │
  │    4. 各自更新参数（保证一致性）              │
  │                                              │
  │  优点：无主卡瓶颈，多进程无 GIL 限制          │
  └──────────────────────────────────────────────┘
```

---

## 二、Ring All-Reduce 详解

```
4 GPU 的 Ring All-Reduce（Reduce-Scatter + All-Gather）：

初始状态（每张 GPU 有完整梯度 [a,b,c,d]）：
  GPU0: [a0, b0, c0, d0]
  GPU1: [a1, b1, c1, d1]
  GPU2: [a2, b2, c2, d2]
  GPU3: [a3, b3, c3, d3]

Step 1: Reduce-Scatter（2 步，每步发 N/4 数据）
  环形传递，最终：
  GPU0 持有 sum(a) = a0+a1+a2+a3
  GPU1 持有 sum(b) = b0+b1+b2+b3
  GPU2 持有 sum(c) = c0+c1+c2+c3
  GPU3 持有 sum(d) = d0+d1+d2+d3

Step 2: All-Gather（再 2 步，每步发 N/4 数据）
  每张 GPU 把自己的分片发给其他所有卡
  最终每张 GPU 都有：[sum(a), sum(b), sum(c), sum(d)]

总通信量 = 2 × (N-1)/N × model_size
（与 GPU 数量无关的线性扩展！）
```

---

## 三、PyTorch DDP 代码示例

### 3.1 最简 DDP 训练脚本

```python
# train_ddp.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup(rank, world_size):
    """初始化进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # nccl 是 GPU 通信的最优后端
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # 模型移到对应 GPU
    model = MyModel().to(rank)
    
    # 用 DDP 包装（这一行是关键！）
    model = DDP(model, device_ids=[rank])
    
    # DistributedSampler 保证每张卡看到不同数据
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(10):
        sampler.set_epoch(epoch)  # 每 epoch 重新 shuffle
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(rank), labels.to(rank)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # 反向时自动 All-Reduce 梯度
            optimizer.step()
        
        if rank == 0:  # 只在主进程打印/保存
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            torch.save(model.module.state_dict(), f"checkpoint_{epoch}.pt")
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
```

### 3.2 用 torchrun 启动（推荐方式）

```bash
# 单机 4 卡
torchrun --nproc_per_node=4 train_ddp.py

# 多机多卡（2机各4卡，共8卡）
# 机器1（主节点）
torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr="192.168.1.1" \
  --master_port=12355 \
  train_ddp.py

# 机器2（工作节点）
torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr="192.168.1.1" \
  --master_port=12355 \
  train_ddp.py
```

### 3.3 Gradient Accumulation（梯度累积）

```python
# 用梯度累积模拟更大的 batch size
accumulation_steps = 4  # 等价 batch = 32 × 4 = 128

for i, batch in enumerate(dataloader):
    inputs, labels = batch[0].to(rank), batch[1].to(rank)
    
    # 关键：非最后一步不做梯度同步（节省通信）
    with model.no_sync() if (i + 1) % accumulation_steps != 0 else nullcontext():
        outputs = model(inputs)
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 四、性能对比

### 4.1 DP vs DDP 实测

| 配置 | 吞吐量 (samples/s) | GPU 0 利用率 | GPU 1-3 利用率 |
|------|-------------------|-------------|----------------|
| 单卡 | 100 | 90% | — |
| DP (4卡) | 280 | 95% | 75% (不均衡) |
| DDP (4卡) | 380 | 88% | 88% (均衡) |
| DDP (4卡) + gradient_as_bucket_view | 400 | 90% | 90% |

**结论**：DDP 比 DP 快约 35%，主要原因是消除主卡瓶颈。

### 4.2 DDP 扩展效率

```
理想线性加速 vs 实际（1B 参数模型，A100 NVLink）：

GPU数量:  1     2     4     8     16
理想加速: 1x    2x    4x    8x    16x
实际加速: 1x   1.9x  3.7x  7.1x  13.2x

扩展效率: 100%  95%  92.5%  88.8%  82.5%
（通信开销随 GPU 数增加而增大）
```

### 4.3 通信优化开关

```python
# bucket 大小影响通信效率（默认 25MB）
model = DDP(
    model,
    device_ids=[rank],
    bucket_cap_mb=25,           # 梯度 bucket 大小
    gradient_as_bucket_view=True,  # 减少显存拷贝
    find_unused_parameters=False,  # 关掉动态图检测（静态图更快）
)
```

---

## 五、常见坑

| 问题 | 原因 | 解法 |
|------|------|------|
| 各卡 loss 不同步 | 忘记 All-Reduce loss | 用 `dist.all_reduce(loss)` |
| 保存了 DDP 包装后的参数 | `model.state_dict()` 包含 `module.` 前缀 | 用 `model.module.state_dict()` |
| 各卡数据重复 | 没用 `DistributedSampler` | 加上 sampler |
| 死锁 | 进程数与 GPU 数不匹配 | 检查 `world_size` |
| OOM | batch size 没随 GPU 数缩放 | 每卡 batch_size = global_bs / world_size |

---

## 小结

- **数据并行**是最简单、最常用的分布式训练方式，适合模型能放进单卡的场景
- **DDP 优于 DP**：无主卡瓶颈，多进程无 GIL，通信效率更高
- **Ring All-Reduce** 是 DDP 的通信核心，通信量与 GPU 数量无关（线性扩展）
- 实际扩展效率在8卡约 88%，16卡约 82%；通信是主要瓶颈
- 关键技巧：`no_sync()` 做梯度累积、`bucket_cap_mb` 调整通信粒度
