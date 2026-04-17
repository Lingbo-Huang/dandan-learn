# D5：DeepSpeed 入门配置与实战

> AI Infra Week2 · Day 5 | 作者：🥚🥚5号 ai infra

---

## 一、DeepSpeed 是什么

DeepSpeed 是微软开源的大规模分布式训练库，核心特点：

- **ZeRO 优化器**：大幅降低显存消耗（D4 已详细讲解）
- **混合精度训练**：内置 FP16/BF16 支持与 Loss Scaling
- **梯度累积**：透明化处理，无需手动管理
- **通信优化**：与 NCCL 深度整合，自动通信通信重叠
- **分布式调试**：丰富的 profiling 工具

DeepSpeed 的设计哲学：**最小代码改动，最大训练效率**。

```python
# 传统 PyTorch DDP 训练（改动量大）
model = DDP(model, device_ids=[local_rank])
optimizer = ...
# 手动管理混合精度、梯度累积、ZeRO...

# DeepSpeed（几行搞定）
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, optimizer=optimizer, config="ds_config.json"
)
```

---

## 二、安装与环境配置

```bash
# 基础安装
pip install deepspeed

# 验证安装，检查各组件状态
ds_report

# 带 APEX、Transformer Engine 等扩展的安装
DS_BUILD_OPS=1 pip install deepspeed  # 编译所有 CUDA 扩展（慢但全）

# 按需编译特定扩展
DS_BUILD_FUSED_ADAM=1 pip install deepspeed  # 编译 FusedAdam
```

`ds_report` 输出示例：

```
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be installed JIT on first use
...
[OK]  cpu_adam
[OK]  fused_adam
[OK]  async_io
...
```

---

## 三、DeepSpeed 配置文件详解

DeepSpeed 通过 JSON 配置文件控制所有训练行为。

### 3.1 最小配置（FP16 + ZeRO Stage 2）

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4,

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },

  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8
  },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 1000
    }
  },

  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

**关键参数说明：**

- `train_batch_size`：全局 batch = `micro_batch × gpu数 × grad_accum`
- `loss_scale: 0`：启用动态 Loss Scaling（推荐）
- `overlap_comm: true`：通信与计算重叠（提速）
- `contiguous_gradients: true`：梯度连续存储（减少碎片）

### 3.2 BF16 配置（A100+ 推荐）

```json
{
  "bf16": {
    "enabled": true
  },
  "fp16": {
    "enabled": false
  }
}
```

BF16 与 FP16 不同：
- BF16 指数位与 FP32 相同（8 位），不容易溢出
- BF16 **不需要 Loss Scaling**，训练更稳定
- 需要 A100/A800/H100 或更新的 GPU 才有硬件加速

### 3.3 ZeRO Stage 3 + CPU Offload 配置

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true,
      "buffer_count": 4,
      "fast_init": false
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true,
      "buffer_count": 5,
      "buffer_size": 1e8,
      "max_in_cpu": 1e9
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

---

## 四、完整 DeepSpeed 训练代码

### 4.1 标准训练脚本

```python
# train_deepspeed.py
import os
import json
import argparse
import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# ──── 数据集 ─────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, num_samples=50000, seq_len=512, vocab_size=50257):
        self.input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = self.input_ids.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }


# ──── 模型（简化的 GPT 结构）──────────────────────────────

class GPTBlock(nn.Module):
    def __init__(self, d_model=1024, nhead=16, dim_ffn=4096, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True,
                                           dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        # Pre-norm（GPT-2 风格）
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, is_causal=True)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class SimpleGPT(nn.Module):
    def __init__(
        self,
        vocab_size=50257,
        d_model=1024,
        nhead=16,
        num_layers=12,
        max_seq_len=1024,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, nhead, d_model * 4) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享（GPT-2 标准做法）
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)   # [B, T, vocab_size]

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return {"logits": logits, "loss": loss}


# ──── 主训练函数 ─────────────────────────────────────────

def train(args):
    # ① 读取 DeepSpeed 配置
    with open(args.ds_config) as f:
        ds_config = json.load(f)

    # ② 从配置中获取 micro batch 大小（可覆盖命令行参数）
    micro_batch_size = ds_config.get(
        "train_micro_batch_size_per_gpu",
        args.micro_batch_size
    )

    # ③ 数据集
    dataset = TextDataset(num_samples=100000, seq_len=512)

    # ④ DistributedSampler（deepspeed.initialize 会自动初始化 dist）
    deepspeed.init_distributed()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # ⑤ 模型（DeepSpeed 的 ZeRO Stage 3 支持延迟初始化）
    if ds_config.get("zero_optimization", {}).get("stage", 0) == 3:
        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            model = SimpleGPT(vocab_size=50257, d_model=1024, num_layers=12)
    else:
        model = SimpleGPT(vocab_size=50257, d_model=1024, num_layers=12)

    # ⑥ DeepSpeed 初始化（核心！）
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        print(f"Using ZeRO Stage: {ds_config.get('zero_optimization', {}).get('stage', 0)}")

    # ⑦ 训练循环
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model_engine.train()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)

            # 前向传播
            outputs = model_engine(input_ids, labels=labels)
            loss = outputs["loss"]

            # 后向传播（DeepSpeed 内部处理 Loss Scaling, ZeRO 通信等）
            model_engine.backward(loss)

            # 参数更新（DeepSpeed 内部处理 AllGather, 优化器步骤）
            model_engine.step()

            if rank == 0 and step % args.log_steps == 0:
                print(
                    f"Epoch {epoch} | Step {step} | "
                    f"Loss {loss.item():.4f} | "
                    f"LR {optimizer.param_groups[0]['lr']:.2e}"
                )

        # ⑧ 保存 Checkpoint
        if rank == 0 or True:  # DeepSpeed 要求所有 rank 参与保存
            model_engine.save_checkpoint(
                args.output_dir,
                tag=f"epoch_{epoch}",
            )

    # ⑨ Stage 3 下保存 FP32 权重
    if rank == 0:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        if ds_config.get("zero_optimization", {}).get("stage", 0) == 3:
            state_dict = get_fp32_state_dict_from_zero_checkpoint(args.output_dir)
            torch.save(state_dict, f"{args.output_dir}/model_fp32.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_config", type=str, default="ds_config.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    # DeepSpeed 的参数通过环境变量传入，不需要显式写 --local_rank
    args = parser.parse_args()
    train(args)
```

### 4.2 启动命令

```bash
# 单机 4 卡
deepspeed --num_gpus=4 train_deepspeed.py \
  --ds_config ds_config_stage2.json \
  --epochs 10

# 多机（2 节点，每节点 8 卡），需要一个 hostfile
# hostfile 内容：
# node0 slots=8
# node1 slots=8

deepspeed --hostfile hostfile \
  --master_addr node0 \
  --master_port 29500 \
  train_deepspeed.py \
  --ds_config ds_config_stage2.json
```

---

## 五、DeepSpeed 与 HuggingFace Trainer 集成

HuggingFace Trainer 原生支持 DeepSpeed，只需传递配置文件：

```python
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=True,
    dataloader_num_workers=4,
    logging_steps=100,
    save_steps=500,
    save_total_limit=3,
    # 关键：传入 DeepSpeed 配置文件
    deepspeed="ds_config_stage2.json",
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=lambda data: {
        "input_ids": torch.stack([d["input_ids"] for d in data]),
        "labels": torch.stack([d["input_ids"] for d in data]),
        "attention_mask": torch.stack([d["attention_mask"] for d in data]),
    },
)

trainer.train()
```

---

## 六、DeepSpeed 性能监控与调优

### 6.1 开启 Profiling

```json
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": "flops_profile.txt"
  },
  "wall_clock_breakdown": true
}
```

### 6.2 常用调优参数

```json
{
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,          // 通信计算重叠（必开）
    "contiguous_gradients": true,  // 梯度连续存储（必开）
    "reduce_bucket_size": 5e8,     // 梯度桶大小（根据带宽调整）
    "allgather_bucket_size": 5e8   // AllGather 桶大小
  },
  "activation_checkpointing": {
    "partition_activations": true, // 激活值分片（减显存）
    "cpu_checkpointing": false,    // 激活值卸载到 CPU（慢）
    "contiguous_memory_optimization": true
  }
}
```

### 6.3 自动参数调整

DeepSpeed 支持 `"auto"` 关键字，让框架自动选择最优值：

```json
{
  "zero_optimization": {
    "stage": 3,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto"
  }
}
```

---

## 七、常见报错与排查

### 7.1 OOM（显存不足）

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**排查步骤：**
1. 尝试提高 ZeRO Stage（1 → 2 → 3）
2. 减小 `train_micro_batch_size_per_gpu`
3. 启用 `offload_optimizer` 或 `offload_param`
4. 启用梯度检查点（重计算激活值）
5. 开启混合精度（FP16/BF16）

```python
# 在 Trainer 中启用梯度检查点
model.gradient_checkpointing_enable()
```

### 7.2 进程挂起（Hang）

```
# 所有进程卡在某处不动
```

**常见原因：**
- Stage 3 下，某个操作不是所有进程都执行（如 `if rank == 0: model.forward()`）
- AllGather 需要所有进程参与，若某进程跳过则永久等待

```python
# 错误做法
if rank == 0:
    output = model(input)  # Stage 3 下会挂起！

# 正确做法：所有进程都执行前向
output = model(input)
if rank == 0:
    process(output)  # 处理逻辑可以只在 rank 0
```

### 7.3 Loss 为 NaN（FP16 溢出）

```
Loss: nan
```

**解决：**

```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,          // 动态 scaling，从不手动写固定值
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1       // 下限保护
  }
}
```

或者直接换用 BF16（如果 GPU 支持）。

---

## 八、DeepSpeed 速查卡

```
启动命令：
  deepspeed --num_gpus=N train.py --deepspeed ds_config.json

常用 API：
  deepspeed.initialize(model, optimizer, config)  → (engine, optim, dl, scheduler)
  engine.backward(loss)          代替 loss.backward()
  engine.step()                  代替 optimizer.step() + lr_scheduler.step()
  engine.save_checkpoint(dir)    保存 checkpoint（所有 rank 都调）
  engine.load_checkpoint(dir)    加载 checkpoint
  engine.global_rank             当前进程全局 rank
  engine.device                  当前 GPU device

ZeRO Stage 选择：
  Stage 0（无 ZeRO）：小模型验证代码
  Stage 1：优化器状态分片，几乎无额外开销，推荐默认使用
  Stage 2：+ 梯度分片，通信量与 DDP 相同，大部分场景首选
  Stage 3：+ 参数分片，超大模型必选，注意 gather 参数的陷阱
```

---

## 九、本章小结

DeepSpeed 是工程化大模型训练的利器：
1. **配置文件驱动**，代码改动极少
2. **ZeRO + 混合精度** 的完美组合降低显存、提升速度
3. **与 HuggingFace Trainer** 无缝集成，生产首选
4. **常见坑**：Stage 3 下所有进程需参与前向；FP16 NaN 用动态 scaling 或换 BF16

**下一篇（D6）**将深入混合精度训练的原理，包括 FP16/BF16 的区别、Loss Scaling 为什么必要，以及如何在 PyTorch 中手动实现。
