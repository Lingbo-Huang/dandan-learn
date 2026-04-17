# DeepSpeed 入门：一站式分布式训练框架

> **Week 2 · Day 5**  
> 目标：快速上手 DeepSpeed，掌握配置文件与核心 API

---

## 一、DeepSpeed 是什么？

```
DeepSpeed = 微软出品的分布式深度学习优化框架

核心功能：
  ┌───────────────────────────────────────────────┐
  │  DeepSpeed                                    │
  │                                               │
  │  ① ZeRO 优化器 (Stage 1/2/3/Infinity)        │
  │  ② 混合精度训练 (FP16/BF16)                  │
  │  ③ 流水线并行 (PipelineModule)               │
  │  ④ 张量并行 (与 Megatron 集成)               │
  │  ⑤ 梯度压缩 (1-bit Adam, PowerSGD)          │
  │  ⑥ 推理优化 (DeepSpeed-Inference)            │
  │  ⑦ 自动调优 (DeepSpeed Autotuning)           │
  └───────────────────────────────────────────────┘

最大卖点：JSON 配置文件驱动，改几行配置就能切换并行策略
```

---

## 二、架构图解

```
用户代码层：
  ┌──────────────────────────────┐
  │  model, optimizer, dataloader│
  └─────────────┬────────────────┘
                │ deepspeed.initialize()
  ┌─────────────▼────────────────┐
  │      DeepSpeedEngine         │  ← 统一入口
  │  ┌──────────┐ ┌───────────┐  │
  │  │  ZeRO    │ │ FP16 Mgr │  │
  │  │ Stage 3  │ │ (loss scale│  │
  │  └──────────┘ └───────────┘  │
  │  ┌──────────┐ ┌───────────┐  │
  │  │ Pipeline │ │ Comm mgr  │  │
  │  │ Parallel │ │ (NCCL)    │  │
  │  └──────────┘ └───────────┘  │
  └──────────────────────────────┘
                │
  ┌─────────────▼────────────────┐
  │     PyTorch / CUDA / NCCL    │
  └──────────────────────────────┘
```

---

## 三、快速上手

### 3.1 安装

```bash
# 基础安装
pip install deepspeed

# 验证安装
ds_report

# 如果需要编译 CUDA 扩展（可选但推荐）
DS_BUILD_OPS=1 pip install deepspeed --global-option="build_ext"
```

### 3.2 最小化代码改动（5步迁移）

```python
# === 原始 PyTorch 训练代码 ===
model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# for batch in dataloader:
#     ...

# === DeepSpeed 版本（只需改5步）===
import deepspeed

# Step 1: 添加命令行参数解析
import argparse
parser = argparse.ArgumentParser()
parser = deepspeed.add_config_arguments(parser)  # 添加 --deepspeed_config 参数
args = parser.parse_args()

# Step 2: 初始化引擎（optimizer 由 DS 接管）
model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    args=args,                          # Step 3: 传入 args
    model=model,
    model_parameters=model.parameters(),
    # optimizer 可以不传，DS 会根据配置文件创建
)

# Step 4: 前向/反向用引擎 API
for batch in dataloader:
    inputs, labels = batch[0].to(model_engine.local_rank), batch[1].to(model_engine.local_rank)
    
    outputs = model_engine(inputs)
    loss = criterion(outputs, labels)
    
    model_engine.backward(loss)    # 替代 loss.backward()
    model_engine.step()            # 替代 optimizer.step()

# Step 5: 保存 checkpoint
model_engine.save_checkpoint('./checkpoints')
```

### 3.3 配置文件详解（ds_config.json）

```json
{
  // ─── 基础训练配置 ───
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  "train_batch_size": "auto",  // = micro_bs × accum × world_size
  "steps_per_print": 100,
  
  // ─── 混合精度 ───
  "fp16": {
    "enabled": true,
    "loss_scale": 0,            // 0 = 动态 loss scaling
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  
  // ─── ZeRO 优化 ───
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  
  // ─── 优化器（也可在代码中定义，不写这里）───
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  
  // ─── 学习率调度 ───
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 1000,
      "total_num_steps": 100000
    }
  },
  
  // ─── 梯度裁剪 ───
  "gradient_clipping": 1.0,
  
  // ─── 监控 ───
  "wall_clock_breakdown": false,
  "tensorboard": {
    "enabled": true,
    "output_path": "./tb_logs",
    "job_name": "my_training"
  }
}
```

### 3.4 启动命令

```bash
# 单机多卡（4 GPU）
deepspeed --num_gpus=4 train.py --deepspeed_config ds_config.json

# 多机多卡（2机各4卡）
deepspeed \
  --hostfile hostfile.txt \    # 格式: ip slots=4
  --num_gpus=4 \
  train.py \
  --deepspeed_config ds_config.json

# hostfile.txt 格式：
# node01 slots=4
# node02 slots=4

# 也可以用 torchrun + deepspeed（兼容 FSDP 工作流）
torchrun --nproc_per_node=4 train.py \
  --deepspeed --deepspeed_config ds_config.json
```

---

## 四、与 Hugging Face Transformers 集成

```python
# 使用 Hugging Face Trainer + DeepSpeed
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,
    deepspeed="ds_config.json",   # 关键：传入配置文件路径
    num_train_epochs=3,
    logging_steps=100,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

---

## 五、性能对比

### 5.1 不同框架在 GPT-2 (1.5B) 上的对比

```
测试环境：8 × A100 40GB，batch_size=32（全局）

框架配置                  | 显存/卡 | 吞吐量 (tokens/s) | 扩展效率
──────────────────────────────────────────────────────────────────
PyTorch DDP               | 38 GB  | 45,000            | 基准
DeepSpeed ZeRO-1          | 22 GB  | 43,500            | 97%
DeepSpeed ZeRO-2          | 14 GB  | 42,000            | 93%
DeepSpeed ZeRO-3          |  8 GB  | 38,000            | 84%
DeepSpeed ZeRO-3 + Offload|  5 GB  | 22,000            | 49%
```

### 5.2 DeepSpeed 核心特性

| 特性 | 效果 |
|------|------|
| Overlap 通信与计算 | 降低通信等待 15-25% |
| Fused Optimizer | Adam 计算提速 5-10x |
| Contiguous Memory | 减少显存碎片化 |
| Dynamic Loss Scale | 自动防止 FP16 溢出 |

---

## 六、Checkpoint 管理

```python
# 保存
model_engine.save_checkpoint(
    save_dir="./checkpoints",
    tag="step_1000",        # checkpoint 名
    client_state={"step": 1000, "epoch": 2}  # 自定义元数据
)

# 加载
_, client_state = model_engine.load_checkpoint(
    load_dir="./checkpoints",
    tag="step_1000"
)
step = client_state["step"]

# 提取纯 PyTorch 权重（用于推理）
# 仅在 rank 0 执行
if model_engine.local_rank == 0:
    # ZeRO-3 需要先合并参数
    with deepspeed.zero.GatheredParameters(model.parameters()):
        state_dict = model.state_dict()
        torch.save(state_dict, "model_weights.pt")
```

---

## 七、常见问题 & 调试

| 问题 | 排查方向 |
|------|---------|
| OOM（显存溢出） | 降低 micro_batch_size，或升级 ZeRO stage |
| 训练不收敛 | 检查 loss_scale_window，降低 lr，检查 gradient_clipping |
| 速度比 DDP 慢 | 关闭 ZeRO-3，检查 overlap_comm，减少 pipeline stages |
| checkpoint 加载失败 | 确保 world_size 与保存时一致 |
| NCCL 超时 | 增加 NCCL_TIMEOUT，检查网络带宽 |

```bash
# 调试开关
export DEEPSPEED_LOG_LEVEL=DEBUG
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

---

## 小结

- **DeepSpeed 的核心价值**：JSON 配置驱动，极少代码改动，一键切换 ZeRO/FP16/流水线
- **最常用功能**：ZeRO-2 + FP16，适合 95% 的微调场景
- **与 Transformers 集成**：一行代码 `deepspeed="ds_config.json"` 即可启用
- **性能调优顺序**：先用 ZeRO-2，OOM 了再 ZeRO-3，速度优先时关 Offload
