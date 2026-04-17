# D4：ZeRO 三个阶段内存优化原理

> AI Infra Week2 · Day 4 | 作者：🥚🥚5号 ai infra

---

## 一、训练内存组成：显存都去哪儿了

在分析 ZeRO 之前，必须先清楚训练过程中显存的四大消耗项。

以一个参数量为 `Ψ`（单位：billion）的模型为例：

### 1.1 模型状态（Model States）

训练时每个参数需要保存：

| 状态 | 精度 | 每参数字节 |
|------|------|-----------|
| 模型参数（weights）| FP16 | 2 bytes |
| 梯度（gradients）| FP16 | 2 bytes |
| Adam 一阶矩（m）| FP32 | 4 bytes |
| Adam 二阶矩（v）| FP32 | 4 bytes |
| Adam 参数副本（master weights）| FP32 | 4 bytes |

**合计**：每参数 `2 + 2 + 4 + 4 + 4 = 16 bytes`

对于 7B 模型：`7B × 16 bytes = 112 GB`

### 1.2 激活值（Activations）

前向传播需要保存激活值供后向传播使用：

```
激活值 ≈ seq_len × hidden_dim × num_layers × batch_size × 2 bytes

示例（GPT-3 175B, seq=2048, bs=1）：
≈ 2048 × 12288 × 96 × 1 × 2 ≈ 48 GB
```

可通过 Gradient Checkpointing 以重计算换显存（减少 ~10x，但增加 ~33% 计算量）。

### 1.3 临时缓冲区（Temporary Buffers）

通信、参数更新过程中产生的临时张量，通常数 GB。

### 1.4 碎片化内存（Fragmentation）

Python 内存管理碎片，实际可用显存会低于理论值。

---

## 二、ZeRO 的核心洞察

**ZeRO（Zero Redundancy Optimizer）** 是 DeepSpeed 团队（微软）提出的核心技术，论文发表于 SC'20。

核心洞察：**在数据并行中，每张 GPU 都持有完整的模型状态，这是冗余的！**

```
传统数据并行（4 张 GPU）：

GPU 0: [Params, Grads, Optimizer States]  ← 完整副本
GPU 1: [Params, Grads, Optimizer States]  ← 完整副本（冗余）
GPU 2: [Params, Grads, Optimizer States]  ← 完整副本（冗余）
GPU 3: [Params, Grads, Optimizer States]  ← 完整副本（冗余）

4 张 GPU，4 倍冗余！
```

ZeRO 的思路：**把模型状态切分（Shard）到所有 GPU 上，每张 GPU 只保存 1/N 的状态**。

---

## 三、ZeRO 三个阶段详解

ZeRO 分三个阶段，每个阶段在前一阶段基础上进一步消除冗余。

### 3.1 ZeRO Stage 1：优化器状态分片

**分片内容**：优化器状态（Adam 的 m, v 和 master weights）

```
4 张 GPU，参数量 Ψ

未优化：
  GPU 0~3 各自持有完整的 Optimizer States（12Ψ bytes）

ZeRO Stage 1（Pos）：
  GPU 0: Optimizer States 的第 0 块（0 到 Ψ/4 的参数对应的 m, v, master）
  GPU 1: Optimizer States 的第 1 块
  GPU 2: Optimizer States 的第 2 块
  GPU 3: Optimizer States 的第 3 块

每张 GPU 显存节省：Optimizer States 从 12Ψ → 3Ψ（节省 4x）
```

**工作流程：**

```
前向传播：各 GPU 用完整参数（FP16）计算，无变化
后向传播：各 GPU 计算完整梯度，然后 ReduceScatter
         → 每张 GPU 只得到自己负责的那部分梯度

参数更新：每张 GPU 只更新自己持有的那部分优化器状态
         对应的 FP16 参数用 AllGather 广播给所有 GPU
```

**显存节省**：`4x`（只针对优化器状态）
**额外通信**：与标准 DDP 相同（ReduceScatter + AllGather ≈ AllReduce）

### 3.2 ZeRO Stage 2：优化器状态 + 梯度分片

**分片内容**：优化器状态 + 梯度

```
ZeRO Stage 2（Pos+g）：

  GPU 0 持有：
    - 完整模型参数（FP16，2Ψ bytes）
    - Gradient 第 0 块（0.5Ψ bytes，仅 1/4）
    - Optimizer States 第 0 块（3Ψ bytes，仅 1/4）

每张 GPU 显存：2Ψ + 0.5Ψ + 3Ψ ≈ 5.5Ψ bytes
对比原始 16Ψ bytes，节省约 8x
```

**工作流程：**

```
后向传播过程：
  1. 每个 GPU 在本地积累梯度（全量）
  2. 对每个 Bucket，执行 ReduceScatter：
     - 归约操作（求和/均值）
     - 每张 GPU 只保留自己负责分片的梯度
     - 其余梯度立即释放！

参数更新：
  每张 GPU 用本地的梯度分片 + 优化器状态分片更新参数分片
  再 AllGather 让所有 GPU 得到更新后的完整参数
```

**关键细节**：ReduceScatter 而非 AllReduce！梯度收到对应分片后立即释放其他部分，大幅节省显存。

### 3.3 ZeRO Stage 3：完全分片（包含参数）

**分片内容**：优化器状态 + 梯度 + **模型参数**

```
ZeRO Stage 3（Pos+g+p）：

  GPU 0 持有：
    - 参数第 0 块（0.5Ψ bytes，1/4）
    - Gradient 第 0 块（0.5Ψ bytes，1/4）
    - Optimizer States 第 0 块（3Ψ bytes，1/4）
  合计：每张 GPU ≈ 4Ψ / N bytes

N=64 时，显存降低 64x！（理论上）
```

**工作流程（Stage 3 最复杂）：**

```
前向传播（每层前）：
  AllGather 当前层参数 → 得到完整参数 → 计算 → 释放其他分片

后向传播（每层后）：
  AllGather 当前层参数 → 计算梯度 → ReduceScatter 梯度 → 释放参数

参数更新：
  每 GPU 更新本地参数分片
```

**代价**：前向和后向各增加一次 AllGather，通信量是 Stage 2 的 1.5 倍。

---

## 四、ZeRO 三阶段对比

```
                     显存消耗（N 卡数据并行）

原始 DDP:
  每卡 = 2Ψ(params) + 2Ψ(grads) + 12Ψ(optimizer) = 16Ψ
  与 N 无关（完全冗余）

Stage 1（Pos）：
  每卡 = 2Ψ + 2Ψ + 12Ψ/N = (4 + 12/N)Ψ
  N=64 时 ≈ 4.2Ψ（优化器省 64x）

Stage 2（Pos+g）：
  每卡 = 2Ψ + 2Ψ/N + 12Ψ/N = (2 + 14/N)Ψ
  N=64 时 ≈ 2.2Ψ（梯度也省）

Stage 3（Pos+g+p）：
  每卡 = 2Ψ/N + 2Ψ/N + 12Ψ/N = 16Ψ/N
  N=64 时 ≈ 0.25Ψ（全省！）
```

| 阶段 | 分片内容 | 理论显存 | 额外通信 |
|------|---------|---------|---------|
| 基准 DDP | 无 | 16Ψ | - |
| Stage 1 | 优化器状态 | 4Ψ + 12Ψ/N | 与 DDP 相同 |
| Stage 2 | +梯度 | 2Ψ + 14Ψ/N | 与 DDP 相同 |
| Stage 3 | +参数 | 16Ψ/N | 1.5x |

---

## 五、ZeRO-Offload：CPU 内存作为扩展

ZeRO-Offload 是 Stage 2 的扩展：将**优化器状态和梯度 Offload 到 CPU 内存**。

```
GPU 显存：模型参数（FP16）+ 梯度（短暂）
CPU 内存：优化器状态（FP32）+ 梯度副本

梯度更新流程：
  GPU 计算梯度 → 传输到 CPU → CPU 更新 Adam 状态
  → 更新后参数传回 GPU
```

**适用场景**：单卡/少卡训练 10B 级别模型
**代价**：CPU-GPU 带宽（PCIe ~16 GB/s）成为瓶颈，训练速度下降

### ZeRO-Infinity：NVMe 扩展

更进一步，将优化器状态 Offload 到 **NVMe 固态硬盘**：

```
NVMe（3-5 GB/s）→ CPU（~50 GB/s）→ GPU（NVLink 600 GB/s）

理论可训练参数量：TB 级别（超出所有 GPU 显存之和）
```

---

## 六、DeepSpeed 中配置 ZeRO

### 6.1 ZeRO Stage 1（ds_config_stage1.json）

```json
{
  "zero_optimization": {
    "stage": 1,
    "reduce_bucket_size": 5e8
  }
}
```

### 6.2 ZeRO Stage 2（ds_config_stage2.json）

```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  }
}
```

### 6.3 ZeRO Stage 3（ds_config_stage3.json）

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

### 6.4 对应的训练代码

```python
import deepspeed
import torch
import torch.nn as nn

def train_with_zero():
    # 初始化 DeepSpeed
    model = MyLargeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # DeepSpeed 初始化：自动处理 ZeRO 分片
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config="ds_config_stage2.json",
    )

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)

        outputs = model_engine(input_ids, labels=labels)
        loss = outputs.loss

        # 后向传播（DeepSpeed 内部处理 ReduceScatter）
        model_engine.backward(loss)

        # 参数更新（DeepSpeed 处理 AllGather + 优化器步骤）
        model_engine.step()

        if model_engine.global_rank == 0 and step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
```

---

## 七、ZeRO Stage 3 的特殊注意事项

### 7.1 参数聚合（Gathering Parameters）

Stage 3 下参数被分片，访问完整参数需要 AllGather：

```python
# 保存模型时必须先 gather 参数
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

# 方法一：训练结束后从 checkpoint 还原
state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)

# 方法二：训练中获取（需要所有进程参与！）
with deepspeed.zero.GatheredParameters(model.parameters()):
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), "model.pt")
```

### 7.2 参数初始化

Stage 3 下，参数在初始化时就已经分片。若需要某个参数的完整值进行初始化（如权重共享），需要特殊处理：

```python
import deepspeed

# 延迟参数初始化（节省显存）
with deepspeed.zero.Init(config_dict_or_path="ds_config_stage3.json"):
    model = MyLargeModel()
    # 参数在创建时就已分片，不会在 CPU 上临时存储完整参数
```

---

## 八、ZeRO vs 模型并行

|  | ZeRO Stage 3 | 张量并行 |
|--|-------------|---------|
| 切分维度 | 参数分片（任意分割） | 算子分割（结构化） |
| 通信时机 | 每层前向/后向时 AllGather | 每次 AllReduce |
| 适用性 | 任意模型，无需修改代码 | 需要专门实现 |
| 通信量 | 参数量级 | 激活值量级 |
| 最优策略 | 通信受限时用 ZeRO | 计算受限时用 TP |

---

## 九、本章小结

```
ZeRO 核心思路：打破数据并行中模型状态的冗余

Stage 1：只分 Optimizer States → 显存降 4x（常用，几乎无额外开销）
Stage 2：再分 Gradients      → 显存降 8x（推荐，通信量与 DDP 相同）
Stage 3：再分 Parameters     → 显存降 N×（适合超大模型，通信量 1.5x）

ZeRO-Offload：把状态卸载到 CPU
ZeRO-Infinity：把状态卸载到 NVMe
```

**黄金法则**：
- 7B 模型，8 卡训练 → Stage 2 足够
- 70B 模型，单机 8 卡 → Stage 3 + CPU Offload
- 175B+ 模型 → ZeRO + 流水线并行 + 张量并行

**下一篇（D5）**将介绍 DeepSpeed 的完整配置和工程实战，包括性能调优与常见问题排查。
