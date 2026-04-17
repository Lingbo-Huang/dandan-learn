# D7：综合实战 — 用 DeepSpeed 训练 GPT-2

> AI Infra Week2 · Day 7 | 作者：🥚🥚5号 ai infra

---

## 一、本章目标与项目概览

这是 Week2 的综合实战章节，将 D1-D6 的所有知识点融合为一个完整的工程项目：

**目标**：从零开始，用 DeepSpeed + ZeRO Stage 2 + 混合精度，在 WikiText-103 数据集上训练 GPT-2（124M 参数）模型。

**技术栈：**
- PyTorch 2.x + DeepSpeed 0.12+
- HuggingFace Transformers（模型定义）
- HuggingFace Datasets（数据加载）
- BF16 混合精度（A100 环境）/ FP16 + GradScaler（V100 环境）
- ZeRO Stage 2（梯度分片）
- DistributedSampler + DataLoader

**项目结构：**

```
gpt2_deepspeed/
├── ds_config.json           # DeepSpeed 配置
├── train.py                 # 主训练脚本
├── data.py                  # 数据集处理
├── model.py                 # 模型定义
├── evaluate.py              # 评估脚本
├── run.sh                   # 启动脚本
└── checkpoints/             # 保存目录
```

---

## 二、环境准备

```bash
# 创建虚拟环境
conda create -n gpt2_ds python=3.10
conda activate gpt2_ds

# 安装依赖
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install deepspeed==0.12.6
pip install transformers==4.36.0
pip install datasets==2.16.0
pip install accelerate==0.25.0
pip install tensorboard

# 验证 DeepSpeed 安装
ds_report

# 检查 GPU 环境
python -c "import torch; print(torch.cuda.device_count(), 'GPUs'); print(torch.cuda.get_device_name(0))"
```

---

## 三、DeepSpeed 配置文件

### ds_config.json

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4,

  "bf16": {
    "enabled": true
  },

  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "round_robin_gradients": true
  },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 6e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  },

  "scheduler": {
    "type": "WarmupCosineLR",
    "params": {
      "warmup_min_lr": 6e-5,
      "warmup_max_lr": 6e-4,
      "warmup_num_steps": 2000,
      "total_num_steps": 100000
    }
  },

  "gradient_clipping": 1.0,

  "activation_checkpointing": {
    "partition_activations": false,
    "contiguous_memory_optimization": false,
    "cpu_checkpointing": false
  },

  "flops_profiler": {
    "enabled": false
  },

  "steps_per_print": 100,
  "wall_clock_breakdown": false,
  "dump_state": false
}
```

---

## 四、数据处理（data.py）

```python
# data.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import GPT2Tokenizer


class WikiTextDataset(Dataset):
    """
    WikiText-103 数据集，适配 GPT-2 语言建模任务。
    将文本拼接后切成固定长度的 chunks。
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 1024,
        cache_dir: str = "./data_cache",
    ):
        self.seq_len = seq_len

        # 加载分词器
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载数据集
        print(f"Loading WikiText-103 ({split} split)...")
        dataset = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split=split,
            cache_dir=cache_dir,
        )

        # 缓存 tokenized 数据（避免每次重新处理）
        cache_file = os.path.join(cache_dir, f"wikitext103_{split}_{seq_len}.pt")
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            self.chunks = torch.load(cache_file)
        else:
            self.chunks = self._tokenize_and_chunk(dataset)
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(self.chunks, cache_file)
            print(f"Saved cached data to {cache_file}")

        print(f"Dataset size: {len(self.chunks)} chunks of {seq_len} tokens")

    def _tokenize_and_chunk(self, dataset):
        """将所有文本拼接后切成等长的 chunks"""
        # 过滤空行
        texts = [t for t in dataset["text"] if t.strip()]

        # 分词（一次性处理所有文本）
        print("Tokenizing...")
        all_tokens = []
        for i, text in enumerate(texts):
            if i % 10000 == 0:
                print(f"  {i}/{len(texts)}")
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            all_tokens.extend(tokens)
            all_tokens.append(self.tokenizer.eos_token_id)  # 文档分隔符

        # 切成等长 chunks
        all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        n_chunks = len(all_tokens) // self.seq_len
        chunks = all_tokens[:n_chunks * self.seq_len].view(n_chunks, self.seq_len)

        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        tokens = self.chunks[idx]
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),  # GPT 自回归：标签等于输入本身（shifted 在模型内处理）
        }


def build_dataloader(
    split: str,
    seq_len: int,
    micro_batch_size: int,
    world_size: int,
    rank: int,
    num_workers: int = 4,
):
    dataset = WikiTextDataset(split=split, seq_len=seq_len)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == "train"),
    )

    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # 保持 batch 大小一致
    )

    return loader, sampler
```

---

## 五、模型定义（model.py）

```python
# model.py
from transformers import GPT2LMHeadModel, GPT2Config


def build_gpt2_model(model_size: str = "gpt2"):
    """
    构建 GPT-2 模型
    
    可选规格：
      gpt2       - 124M 参数
      gpt2-medium - 355M 参数
      gpt2-large  - 774M 参数
      gpt2-xl     - 1.5B 参数
    """
    config = GPT2Config.from_pretrained(model_size)
    model = GPT2LMHeadModel(config)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {model_size}")
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")
    print(f"Architecture: {config.n_layer} layers, "
          f"d_model={config.n_embd}, "
          f"n_heads={config.n_head}")

    return model
```

---

## 六、主训练脚本（train.py）

```python
# train.py
import os
import json
import math
import argparse
import time

import torch
import deepspeed
from deepspeed.utils import logger

from data import build_dataloader
from model import build_gpt2_model


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-2 训练（DeepSpeed）")
    parser.add_argument("--model_size", type=str, default="gpt2",
                        help="gpt2 / gpt2-medium / gpt2-large / gpt2-xl")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ds_config", type=str, default="ds_config.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=4)
    # DeepSpeed 自动添加 --local_rank
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def compute_perplexity(loss: float) -> float:
    """困惑度 = e^loss，语言模型的核心评估指标"""
    return math.exp(min(loss, 20))  # 防止 overflow


def evaluate(model_engine, eval_loader, eval_sampler, epoch: int):
    """在验证集上计算 Loss 和 Perplexity"""
    model_engine.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)

            outputs = model_engine(input_ids, labels=labels)
            total_loss += outputs.loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    # 各 GPU 上的 loss 汇总
    import torch.distributed as dist
    loss_tensor = torch.tensor([total_loss, total_tokens], device=model_engine.device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = loss_tensor[0].item() / loss_tensor[1].item()

    model_engine.train()
    return avg_loss


def train(args):
    # ──── 初始化 ──────────────────────────────────────────────
    deepspeed.init_distributed()

    with open(args.ds_config) as f:
        ds_config = json.load(f)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = (rank == 0)

    micro_batch_size = ds_config["train_micro_batch_size_per_gpu"]

    os.makedirs(args.output_dir, exist_ok=True)

    if is_main:
        print("=" * 60)
        print(f"Training GPT-2 with DeepSpeed")
        print(f"World size: {world_size}")
        print(f"ZeRO Stage: {ds_config.get('zero_optimization', {}).get('stage', 0)}")
        print(f"Precision: {'BF16' if ds_config.get('bf16', {}).get('enabled') else 'FP16'}")
        print("=" * 60)

    # ──── 数据集 ──────────────────────────────────────────────
    train_loader, train_sampler = build_dataloader(
        split="train",
        seq_len=args.seq_len,
        micro_batch_size=micro_batch_size,
        world_size=world_size,
        rank=rank,
        num_workers=args.num_workers,
    )

    val_loader, val_sampler = build_dataloader(
        split="validation",
        seq_len=args.seq_len,
        micro_batch_size=micro_batch_size,
        world_size=world_size,
        rank=rank,
        num_workers=args.num_workers,
    )

    if is_main:
        print(f"Train batches per epoch: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

    # ──── 模型 ────────────────────────────────────────────────
    model = build_gpt2_model(args.model_size)

    # ──── DeepSpeed 初始化 ────────────────────────────────────
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    # ──── 训练循环 ────────────────────────────────────────────
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model_engine.train()

        epoch_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            step_start = time.time()

            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)

            # ── 前向传播 ──
            outputs = model_engine(input_ids, labels=labels)
            loss = outputs.loss

            # ── 后向传播（DeepSpeed 内部处理混合精度 + ZeRO 通信）──
            model_engine.backward(loss)

            # ── 参数更新（DeepSpeed 内部处理 unscale + clip + step）──
            model_engine.step()

            epoch_loss += loss.item()
            global_step += 1
            step_time = time.time() - step_start

            # 吞吐量计算
            tokens_per_sec = (
                micro_batch_size * args.seq_len * world_size / step_time
            )

            # ── 日志输出 ──
            if is_main and global_step % args.log_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                ppl = compute_perplexity(avg_loss)
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Loss {loss.item():.4f} | PPL {ppl:.1f} | "
                    f"LR {lr:.2e} | {tokens_per_sec/1000:.1f}K tok/s"
                )

            # ── 验证 ──
            if global_step % args.eval_steps == 0:
                val_loss = evaluate(model_engine, val_loader, val_sampler, epoch)
                val_ppl = compute_perplexity(val_loss)

                if is_main:
                    print(f"  ► Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"  ► New best! Saving checkpoint...")

            # ── 保存 Checkpoint ──
            if global_step % args.save_steps == 0:
                # 所有 rank 都必须参与保存
                model_engine.save_checkpoint(
                    args.output_dir,
                    tag=f"step_{global_step}",
                )
                if is_main:
                    print(f"  ► Checkpoint saved: step_{global_step}")

        # ── Epoch 结束统计 ──
        epoch_time = time.time() - epoch_start
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model_engine, val_loader, val_sampler, epoch)

        if is_main:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} complete | Time: {epoch_time/60:.1f}min")
            print(f"  Train Loss: {avg_train_loss:.4f} | PPL: {compute_perplexity(avg_train_loss):.1f}")
            print(f"  Val Loss:   {val_loss:.4f} | PPL: {compute_perplexity(val_loss):.1f}")
            print(f"{'='*60}\n")

    # ──── 保存最终模型 ────────────────────────────────────────
    model_engine.save_checkpoint(args.output_dir, tag="final")

    if is_main:
        print("Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation PPL: {compute_perplexity(best_val_loss):.1f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
```

---

## 七、启动脚本（run.sh）

```bash
#!/bin/bash
# run.sh — 多种启动配置

set -e

MODEL_SIZE=${1:-"gpt2"}  # gpt2 / gpt2-medium / gpt2-large
NUM_GPUS=${2:-4}
CONFIG=${3:-"ds_config.json"}

echo "=== GPT-2 DeepSpeed 训练 ==="
echo "模型规格: $MODEL_SIZE"
echo "GPU 数量: $NUM_GPUS"
echo "配置文件: $CONFIG"
echo ""

# ── 单机训练 ──────────────────────────────────────────────────
deepspeed \
  --num_gpus=$NUM_GPUS \
  train.py \
  --model_size $MODEL_SIZE \
  --seq_len 1024 \
  --epochs 10 \
  --ds_config $CONFIG \
  --output_dir ./checkpoints/$MODEL_SIZE \
  --log_steps 50 \
  --eval_steps 500 \
  --save_steps 2000

# ── 多机训练（2节点）──────────────────────────────────────────
# 创建 hostfile（需要修改 IP）:
# cat hostfile:
#   10.0.0.1 slots=8
#   10.0.0.2 slots=8
#
# deepspeed \
#   --hostfile hostfile \
#   --master_addr 10.0.0.1 \
#   train.py \
#   --model_size gpt2-xl \
#   --ds_config ds_config.json
```

---

## 八、模型评估与文本生成（evaluate.py）

```python
# evaluate.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def load_trained_model(checkpoint_dir: str, tag: str = "final"):
    """从 DeepSpeed checkpoint 加载模型"""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 从 ZeRO checkpoint 提取 FP32 权重
    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        checkpoint_dir, tag=tag
    )

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cuda",
):
    """文本生成"""
    model = model.to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated


def compute_dataset_perplexity(model, tokenizer, texts, seq_len=1024, device="cuda"):
    """计算数据集困惑度"""
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts[:100]:  # 取前 100 条
            input_ids = tokenizer.encode(
                text, return_tensors="pt", max_length=seq_len, truncation=True
            ).to(device)

            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]

    avg_loss = total_loss / total_tokens
    return avg_loss, 2 ** avg_loss  # Loss 和 Perplexity


# ── 使用示例 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/gpt2"

    print("Loading model...")
    model, tokenizer = load_trained_model(checkpoint_dir)

    # 文本生成测试
    prompts = [
        "The history of artificial intelligence",
        "In the year 2030, scientists discovered",
        "The most important lesson I learned",
    ]

    print("\n=== 文本生成测试 ===")
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated[:200]}...")
        print("-" * 50)
```

---

## 九、实验结果与分析

### 9.1 预期训练曲线

```
GPT-2 (124M) 在 WikiText-103 上训练，10 epochs：

Step    Train Loss   Val Loss   Val PPL
------  ----------   ---------  -------
1000    5.2          5.4        221
5000    4.1          4.2        67
10000   3.8          3.9        49
20000   3.5          3.7        40
50000   3.2          3.4        30
100000  3.0          3.2        24

GPT-2 原始论文在 WebText 上达到 18.34 PPL（更大数据集）
WikiText-103 PPL < 30 是合理结果
```

### 9.2 性能基准（A100 × 4）

```
模型规格   参数量   ZeRO Stage  精度   吞吐量           显存/卡
-------   ------   ----------  ----   ------           ------
gpt2       124M    Stage 2    BF16   180K tok/s       ~15 GB
gpt2-medium 355M  Stage 2    BF16   120K tok/s       ~25 GB
gpt2-large  774M  Stage 2    BF16    80K tok/s       ~45 GB
gpt2-xl    1.5B   Stage 2    BF16    45K tok/s       ~70 GB（接近上限）
gpt2-xl    1.5B   Stage 3    BF16    30K tok/s       ~25 GB（分片后）
```

### 9.3 ZeRO Stage 对比（gpt2-xl，4×A100）

```
方案              显存/卡    训练速度      适用性
--------------   --------   ---------    --------
DDP (无ZeRO)     ~75 GB     OOM ✗
ZeRO Stage 1     ~55 GB     100%基准     ✓ 勉强
ZeRO Stage 2     ~30 GB     97%          ✓ 推荐
ZeRO Stage 3     ~12 GB     82%          ✓ 低显存首选
Stage 3+Offload  ~5 GB      50%          ✓ 极致省显存
```

---

## 十、常见问题 FAQ

### Q1：训练中突然出现 NaN，怎么排查？

```bash
# 1. 检查数据集是否有异常（空文本、乱码）
python -c "from data import WikiTextDataset; d = WikiTextDataset(); print(d[0])"

# 2. 临时切换为 FP32 验证模型数值稳定性
# 在 ds_config.json 中：
# "bf16": {"enabled": false}
# "fp32": {"enabled": true}

# 3. 降低学习率（试试 1e-4 甚至 1e-5）
# 4. 确认梯度裁剪已开启（"gradient_clipping": 1.0）
```

### Q2：多机训练 NCCL 超时

```bash
# 增大超时时间
export NCCL_TIMEOUT=3600

# 检查网络连通性
ping node1

# 查看 NCCL 调试日志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### Q3：如何从 checkpoint 恢复训练？

```python
# DeepSpeed 自动管理 checkpoint 恢复
_, client_sd = model_engine.load_checkpoint(
    args.output_dir,
    tag="step_10000",
)
# client_sd 可以存储自定义状态（如全局步数）
global_step = client_sd.get("global_step", 0)
```

### Q4：如何做 LoRA 微调（节省更多显存）？

```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],  # GPT-2 的注意力层
    lora_dropout=0.1,
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 0.36M || all params: 124M || trainable%: 0.29%

# 然后像普通模型一样用 DeepSpeed 训练
model_engine, ... = deepspeed.initialize(model=model, ...)
```

---

## 十一、Week2 总结与知识图谱

```
                    AI Infra Week2 知识图谱
                    
    D1 分布式训练全景          D2 数据并行 (DDP)
    ├── 硬件拓扑               ├── Bucket AllReduce
    ├── 通信原语               ├── DistributedSampler  
    │   ├── AllReduce          ├── no_sync 梯度累积
    │   ├── AllGather          └── FSDP（进阶）
    │   └── ReduceScatter      
    └── 启动方式               D3 模型并行
                               ├── 流水线并行
                               │   ├── GPipe (microbatch)
                               │   └── 1F1B (PipeDream)
                               └── 张量并行
                                   ├── 列并行
                                   └── 行并行
    
    D4 ZeRO 内存优化           D5 DeepSpeed 实战
    ├── Stage 1 (Optimizer)    ├── 配置文件详解
    ├── Stage 2 (+Gradient)    ├── HuggingFace 集成
    ├── Stage 3 (+Param)       └── 调优 + 排错
    └── ZeRO-Offload (CPU/NVMe)
    
    D6 混合精度                D7 综合实战 (本篇)
    ├── FP16 vs BF16           ├── 完整训练 pipeline
    ├── Loss Scaling           ├── 数据 + 模型 + 训练
    └── Master Weights         └── 评估 + 生成
```

**核心经验总结：**

1. **ZeRO Stage 2 是甜蜜点**：通信量与 DDP 相同，显存节省 8x，大部分场景首选
2. **BF16 优于 FP16**（A100+）：无需 Loss Scaling，训练更稳定
3. **计算通信重叠**（overlap_comm=true）是必开选项，可节省 10-20% 时间
4. **Gradient Checkpointing**（激活值重计算）以 33% 计算换 10x 激活值显存，长序列必用
5. **DistributedSampler.set_epoch(epoch)**：每 epoch 重新 shuffle，避免所有进程看到相同顺序

---

*AI Infra Week2 全部 7 篇完结。祝学习愉快！🚀*
