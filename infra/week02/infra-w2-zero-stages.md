# ZeRO：显存优化的终结者

> **Week 2 · Day 4**  
> 目标：理解 ZeRO 三个 Stage 如何逐步消灭显存冗余

---

## 一、数据并行的显存冗余问题

### 1.1 显存占用分析

```
混合精度训练（FP16/FP32）的显存构成：

对于参数量为 Ψ 的模型：

组件                    数据类型    大小
─────────────────────────────────────────
参数 (weights)         FP16        2Ψ bytes
梯度 (gradients)       FP16        2Ψ bytes
FP32 主参数副本        FP32        4Ψ bytes  ← Adam 需要
一阶矩估计 m           FP32        4Ψ bytes  ← Adam 状态
二阶矩估计 v           FP32        4Ψ bytes  ← Adam 状态
激活值（batch相关）     FP16        varies
─────────────────────────────────────────
固定开销合计:          16Ψ bytes

以 GPT-2 (1.5B) 为例：16 × 1.5B ≈ 24 GB（不含激活值）
```

### 1.2 朴素数据并行的问题

```
DDP 中，每张 GPU 都存一份完整的：
  - FP16 参数 (2Ψ)
  - FP16 梯度 (2Ψ)
  - FP32 主参数 (4Ψ)
  - Adam m, v (8Ψ)

N 张卡的总显存 = N × 16Ψ

→ 这是极大的浪费！能不能把这些状态分摊到各卡？
```

---

## 二、ZeRO 三阶段图解

### 2.1 总览

```
ZeRO 核心思路：把"数据并行 = 复制"改为"数据并行 = 分片"

             每卡显存    通信量        描述
─────────────────────────────────────────────────────────
朴素 DDP    16Ψ         2Ψ           全部复制
─────────────────────────────────────────────────────────
ZeRO-1      4Ψ + 12Ψ/N  2Ψ         分片优化器状态
            ≈ 4Ψ (N大)             (主参数+m+v 分片)
─────────────────────────────────────────────────────────
ZeRO-2      2Ψ + 14Ψ/N  2Ψ         +分片梯度
            ≈ 2Ψ (N大)
─────────────────────────────────────────────────────────
ZeRO-3      16Ψ/N        3Ψ         +分片参数
            (随N线性下降)           (通信量略增)
─────────────────────────────────────────────────────────
```

### 2.2 ZeRO Stage 1：优化器状态分片

```
N=4 张 GPU 的情形：

参数（全部复制）:   GPU0[W]  GPU1[W]  GPU2[W]  GPU3[W]
梯度（全部复制）:   GPU0[G]  GPU1[G]  GPU2[G]  GPU3[G]
优化器状态（分片）: GPU0[OS0] GPU1[OS1] GPU2[OS2] GPU3[OS3]
                   (m0,v0)   (m1,v1)   (m2,v2)   (m3,v3)

训练流程：
① 正常前向，各卡计算各自 mini-batch
② 正常反向，各卡得到完整梯度
③ Reduce-Scatter：每卡只保留自己负责的那 1/N 梯度
④ 各卡用自己负责的梯度更新自己的那部分优化器状态
⑤ All-Gather：把各自更新后的参数广播，所有卡恢复完整参数

通信：和 DDP 一样是 All-Reduce，总量 = 2Ψ
显存节省：优化器状态从 12Ψ 降到 12Ψ/N
```

### 2.3 ZeRO Stage 2：优化器状态 + 梯度分片

```
N=4 张 GPU 的情形：

参数（全部复制）:   GPU0[W]   GPU1[W]   GPU2[W]   GPU3[W]
梯度（分片）:       GPU0[G0]  GPU1[G1]  GPU2[G2]  GPU3[G3]
优化器状态（分片）: GPU0[OS0] GPU1[OS1] GPU2[OS2] GPU3[OS3]

训练流程（反向传播时同步分片）：
① 前向：各卡完整前向
② 反向：逐层计算梯度，完成某参数块后立即 Reduce-Scatter
   → 仅保留该卡负责的那份梯度
③ 各卡用自己的梯度切片更新各自的优化器状态 + 主参数
④ All-Gather：恢复所有 GPU 的完整 FP16 参数

通信：仍是 2Ψ（Reduce-Scatter Ψ + All-Gather Ψ）
显存节省：梯度也从 2Ψ 降到 2Ψ/N
```

### 2.4 ZeRO Stage 3：参数 + 梯度 + 优化器状态全分片

```
N=4 张 GPU：

参数（分片！）:     GPU0[W0]  GPU1[W1]  GPU2[W2]  GPU3[W3]
梯度（分片）:       GPU0[G0]  GPU1[G1]  GPU2[G2]  GPU3[G3]
优化器状态（分片）: GPU0[OS0] GPU1[OS1] GPU2[OS2] GPU3[OS3]

前向传播时（实时拼凑参数）：
┌─────────────────────────────────────────────────────────┐
│ 计算 Layer k 时：                                        │
│   1. All-Gather Layer k 的参数分片 → 临时完整参数         │
│   2. 用完整参数做前向计算                                  │
│   3. 计算完立即释放其他卡的参数分片（只保留自己的那份）    │
└─────────────────────────────────────────────────────────┘

反向传播：类似，按需 All-Gather 参数 → 计算梯度 → Reduce-Scatter

通信：≈ 3Ψ（前向 All-Gather Ψ + 反向 All-Gather Ψ + 梯度 Reduce-Scatter Ψ）
显存：每卡只存 16Ψ/N（理论线性缩放！）
```

---

## 三、ZeRO 配置示例

### 3.1 DeepSpeed ZeRO-2 配置

```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,     // 200M 梯度 bucket
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,              // 通信与计算重叠
    "contiguous_gradients": true       // 梯度内存连续（减少碎片）
  },
  "fp16": {
    "enabled": true
  },
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8
}
```

### 3.2 DeepSpeed ZeRO-3 配置

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {             // 优化器状态卸载到 CPU/NVMe
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {                 // 参数卸载到 CPU/NVMe（ZeRO-Infinity）
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,            // 参数分组大小
    "reduce_bucket_size": 1e6,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

### 3.3 ZeRO-Offload（CPU 卸载）

```python
# ZeRO-Offload：把优化器状态卸载到 CPU
# 适合单卡或少卡场景
config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",          # 可选 "nvme"（更大但更慢）
            "pin_memory": True,       # 锁页内存，加速 CPU-GPU 传输
            "buffer_count": 4,
            "fast_init": False
        }
    }
}
```

---

## 四、ZeRO 各阶段性能对比

### 4.1 显存对比（GPT-2 1.5B，8卡 A100 40GB）

```
策略             每卡显存   总显存   能否运行
──────────────────────────────────────────────
朴素 DDP        ~24 GB     ~192 GB  ✅（勉强）
ZeRO-1          ~10 GB     ~80 GB   ✅
ZeRO-2          ~6 GB      ~48 GB   ✅
ZeRO-3          ~4 GB      ~32 GB   ✅
ZeRO-3 + Offload ~3 GB     ~24 GB   ✅（最省）
```

### 4.2 吞吐量对比（10B 模型，32卡 A100）

```
策略                  吞吐量 (TFLOPS/GPU)   通信开销
────────────────────────────────────────────────────
ZeRO-1               142                   低
ZeRO-2               138                   低
ZeRO-3               121                   中（前向多一次 All-Gather）
ZeRO-3 + CPU Offload  67                   高（PCIe 带宽限制）
```

### 4.3 显存节省 vs 通信开销

```
                显存节省
    大 ┤  ZeRO-3
       │    ZeRO-2
       │      ZeRO-1
    小 ┤        DDP
       └────────────────────
          低    通信开销   高
```

---

## 五、ZeRO vs FSDP（PyTorch 原生实现）

| 特性 | DeepSpeed ZeRO | PyTorch FSDP |
|------|----------------|--------------|
| 来源 | Microsoft | Meta/PyTorch 官方 |
| 对应 ZeRO Stage | Stage 1/2/3 | 近似 Stage 3 |
| CPU Offload | ✅ (ZeRO-Infinity) | ✅ |
| 易用性 | 需要 JSON 配置 | Python API |
| 与 HuggingFace 集成 | ✅ Transformers | ✅ Accelerate |
| 性能 | 旗鼓相当 | 旗鼓相当 |

```python
# PyTorch FSDP 示例
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = MyModel()
model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    ),
    cpu_offload=CPUOffload(offload_params=True),
)
```

---

## 六、选择哪个 ZeRO Stage？

```
决策树：

模型放得进单卡？
  └── 是 → 用 DDP 就好

8卡以上，模型 < 10B？
  └── 是 → ZeRO-2（显存够用，通信开销小）

模型 10B-70B？
  └── ZeRO-3（显存线性缩放）

模型 70B+，GPU 不够多？
  └── ZeRO-3 + CPU Offload（牺牲速度换显存）

极端场景（万亿参数）？
  └── ZeRO-Infinity（NVMe 卸载）+ TP/PP
```

---

## 小结

- **ZeRO 的本质**：把数据并行中的"状态复制"改为"状态分片"，显存随 GPU 数量线性下降
- **Stage 1**：分片优化器状态（最低通信代价，省最多的优化器显存）
- **Stage 2**：进一步分片梯度（通信量不变，再省一大块）
- **Stage 3**：连参数也分片（理论上显存线性缩放，但通信量增加 50%）
- **实践建议**：先试 ZeRO-2，不够再上 ZeRO-3，速度优先选 ZeRO-2
