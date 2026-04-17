# Capstone 项目：端到端分布式 LLM 微调实战

> **Week 2 · Day 7**  
> 目标：综合运用本周所学，完成一个 LLaMA-3 8B 的分布式指令微调项目

---

## 一、项目概述

```
项目目标：用 4 张 A100 40GB，对 LLaMA-3 8B 做指令微调（SFT）

技术栈：
  - 模型: LLaMA-3 8B (Meta)
  - 数据: Alpaca 52K 指令数据集
  - 框架: DeepSpeed ZeRO-2 + BF16
  - 加速: Flash Attention 2 + Gradient Checkpointing

预期结果：
  - 显存/卡: ~24 GB（原始全精度约需 >80 GB/卡）
  - 训练速度: ~2000 tokens/s/GPU
  - 最终 Loss: ~0.8-1.0
```

---

## 二、项目结构

```
capstone-llm-sft/
├── data/
│   ├── download_alpaca.py       # 数据下载脚本
│   └── preprocess.py            # 数据预处理
├── configs/
│   ├── ds_config.json           # DeepSpeed 配置
│   └── training_config.yaml     # 训练超参数
├── src/
│   ├── dataset.py               # Dataset 类
│   ├── model.py                 # 模型加载与配置
│   └── trainer.py               # 自定义 Trainer
├── train.py                     # 入口脚本
├── evaluate.py                  # 评估脚本
└── README.md
```

---

## 三、环境准备

```bash
# 创建虚拟环境
conda create -n llm-sft python=3.10 -y
conda activate llm-sft

# 安装依赖
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 datasets accelerate deepspeed
pip install flash-attn --no-build-isolation  # Flash Attention 2
pip install peft  # LoRA（可选）

# 验证 GPU
nvidia-smi
python -c "import torch; print(torch.cuda.device_count(), 'GPUs available')"
ds_report  # 验证 DeepSpeed
```

---

## 四、数据准备

```python
# data/download_alpaca.py
from datasets import load_dataset
import json

dataset = load_dataset("tatsu-lab/alpaca", split="train")

# 格式化为指令模板
def format_alpaca(example):
    if example["input"]:
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": prompt}

dataset = dataset.map(format_alpaca)
dataset.to_json("data/alpaca_formatted.jsonl")
print(f"数据集大小: {len(dataset)} 条")
```

```python
# src/dataset.py
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json

class AlpacaDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path) as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # 语言模型：labels = input_ids（自回归预测）
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # padding 位置不计入 loss
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
```

---

## 五、核心训练代码

```python
# train.py
import os
import torch
import deepspeed
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.dataset import AlpacaDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--data_path", default="data/alpaca_formatted.jsonl")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型（用 bf16 节省显存）
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Flash Attention 2
        use_cache=False,  # Gradient Checkpointing 时需关闭
    )
    
    # 开启 Gradient Checkpointing（以时间换显存）
    model.gradient_checkpointing_enable()
    
    # 数据集
    dataset = AlpacaDataset(args.data_path, tokenizer, args.max_length)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # 每卡 batch size，由 ds_config 控制
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    # DeepSpeed 初始化（接管 optimizer 和 lr_scheduler）
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )
    
    # 训练循环
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        model_engine.train()
        total_loss = 0
        
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model_engine.local_rank)
            attention_mask = batch["attention_mask"].to(model_engine.local_rank)
            labels = batch["labels"].to(model_engine.local_rank)
            
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            model_engine.backward(loss)
            model_engine.step()
            
            total_loss += loss.item()
            
            if step % 100 == 0 and local_rank == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch} | Step {step} | Loss: {avg_loss:.4f}")
        
        # 保存 checkpoint
        model_engine.save_checkpoint(args.output_dir, tag=f"epoch_{epoch}")
    
    # 保存最终权重
    if local_rank == 0:
        # 合并 ZeRO 分片，导出完整模型
        model_engine.save_checkpoint(args.output_dir, tag="final")

if __name__ == "__main__":
    main()
```

---

## 六、DeepSpeed 配置文件

```json
// configs/ds_config.json
{
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 4,
  "train_batch_size": 64,

  "bf16": {
    "enabled": true
  },

  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  },

  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 2e-5,
      "warmup_num_steps": 100,
      "total_num_steps": 2000
    }
  },

  "gradient_clipping": 1.0,

  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "number_checkpoints": 4
  },

  "tensorboard": {
    "enabled": true,
    "output_path": "./tb_logs",
    "job_name": "llama3-sft"
  },

  "wall_clock_breakdown": false
}
```

---

## 七、启动与监控

```bash
# 启动训练（4 GPU）
deepspeed --num_gpus=4 train.py \
  --deepspeed_config configs/ds_config.json \
  --model_name meta-llama/Meta-Llama-3-8B \
  --data_path data/alpaca_formatted.jsonl \
  --num_epochs 3

# 实时监控 GPU 显存和利用率
watch -n 1 nvidia-smi

# TensorBoard 监控 Loss 曲线
tensorboard --logdir ./tb_logs --port 6006

# 查看 NCCL 通信日志（调试用）
NCCL_DEBUG=INFO deepspeed --num_gpus=4 train.py ...
```

---

## 八、性能 Baseline 与优化

### 8.1 预期性能

```
LLaMA-3 8B，4 × A100 40GB，BF16 + ZeRO-2 + Flash Attention 2

指标                    值
─────────────────────────────────────────────
显存占用/卡            ~24 GB（含激活值）
吞吐量                 ~2000-2500 tokens/s/GPU
全局 batch size        64（4卡 × 4micro × 4accum）
完成 1 epoch 时间       ~45 分钟（52K 条，max_len=2048）
最终 Loss              ~0.8-1.0
```

### 8.2 进一步优化技巧

```
优化项               预期提升       配置方式
─────────────────────────────────────────────────────
Flash Attention 2   +20-30% 速度  attn_implementation="flash_attention_2"
Gradient Checkpointing -30% 显存   model.gradient_checkpointing_enable()
Fused Optimizer      +5-10% 速度   ds_config 中使用 DeepSpeed 内置 AdamW
数据 num_workers=4   减少数据等待  DataLoader(num_workers=4)
Pin Memory           减少 CPU→GPU 延迟  DataLoader(pin_memory=True)
LoRA 微调            -70% 显存    使用 PEFT 库替代全参数训练
```

### 8.3 LoRA 配置（可选替代全参数微调）

```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # LoRA 秩
    lora_alpha=32,           # 缩放系数
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 20M || all params: 8B || trainable%: 0.25%
```

---

## 九、评估与推理

```python
# evaluate.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./output/epoch_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

def chat(instruction, max_new_tokens=256):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# 测试
print(chat("用Python写一个快速排序函数"))
print(chat("解释量子纠缠是什么"))
```

---

## 十、Capstone 检查清单

完成项目后，确认你能回答以下问题：

- [ ] 为什么要开 `gradient_checkpointing`？代价是什么？
- [ ] DeepSpeed ZeRO-2 帮我省了多少显存？从哪些地方省的？
- [ ] `overlap_comm` 是什么意思？为什么能提速？
- [ ] 训练时 loss 一直是 NaN，可能是什么原因？
- [ ] 如果把 ZeRO-2 换成 ZeRO-3，需要改什么？速度会变化吗？
- [ ] LoRA 和全参数微调的区别是什么？什么时候选 LoRA？

---

## 小结

本 Capstone 项目综合运用了 Week 2 的全部技术：

| 技术 | 在项目中的体现 |
|------|--------------|
| 数据并行 (DDP) | DeepSpeed 内置，多卡自动同步 |
| ZeRO-2 | 显存从 80GB 降至 24GB |
| 混合精度 (BF16) | A100 最优选择，无需 GradScaler |
| Flash Attention 2 | 长序列训练提速 30% |
| Gradient Checkpointing | 以计算换显存 |
| DeepSpeed 配置 | JSON 文件驱动，无需改大量代码 |

**下一步**：尝试把 ZeRO-2 换成 ZeRO-3，看看 70B 模型能不能在同样的硬件上跑起来。
