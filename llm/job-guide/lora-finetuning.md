---
layout: post
title: "LoRA/QLoRA 微调实战"
track: "🤖 大模型"
---

# LoRA/QLoRA 微调实战

> 2026年算法工程师标配技能。低成本、低显存、快速适配领域数据的高效微调方法。

---

## 为什么用 LoRA 而不是全量微调？

| 方案 | 显存需求 | 训练时间 | 效果 | 成本 |
|------|---------|---------|------|------|
| 全量微调（7B模型） | 80GB+ | 数天 | 最好 | 极高 |
| LoRA（7B模型） | 16GB | 数小时 | 接近全量 | 低 |
| QLoRA（7B模型） | 8GB | 数小时 | 略低于LoRA | 极低 |

**核心原理**：大模型的权重更新矩阵是低秩的，不需要更新所有参数，只需训练两个小矩阵 A 和 B，`ΔW = A × B`，参数量减少99%+。

---

## 1. LoRA 原理

```
原始层：y = W₀x  （W₀冻结，不训练）
LoRA：  y = W₀x + ΔWx = W₀x + BAx

W₀: d×d 原始权重（冻结）
A:  d×r 可训练（r << d，如r=8）
B:  r×d 可训练

参数量：原来 d²，现在 2×d×r
当d=4096, r=8时：减少99.6%的参数
```

---

## 2. QLoRA 完整实战代码

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset

# ── 步骤1：配置4bit量化（QLoRA核心）──
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4bit量化加载
    bnb_4bit_use_double_quant=True,       # 嵌套量化，进一步节省显存
    bnb_4bit_quant_type="nf4",           # NormalFloat4，比int4更准确
    bnb_4bit_compute_dtype=torch.bfloat16 # 计算时用bf16，精度更好
)

# ── 步骤2：加载模型 ──
model_name = "Qwen/Qwen2.5-7B-Instruct"  # 或 meta-llama/Llama-3.1-8B-Instruct

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",                    # 自动分配GPU/CPU
    trust_remote_code=True
)

# 开启梯度检查点，进一步节省显存（以时间换空间）
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# ── 步骤3：配置LoRA ──
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # 秩，越大效果越好但参数越多（8-64）
    lora_alpha=32,           # 缩放因子，通常=2r
    target_modules=[         # 应用LoRA的层（注意力层效果最好）
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,       # 防过拟合
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 20,971,520 || all params: 3,773,546,496 || trainable%: 0.556

# ── 步骤4：准备数据集 ──
def format_instruction(sample: dict) -> str:
    """格式化为指令微调格式"""
    return f"""### 指令:
{sample['instruction']}

### 输入:
{sample.get('input', '')}

### 回答:
{sample['output']}"""

# 加载数据集（示例：alpaca格式）
dataset = load_dataset("json", data_files="train_data.json")["train"]
# 数据格式: [{"instruction": "...", "input": "...", "output": "..."}]

# ── 步骤5：训练配置 ──
training_args = TrainingArguments(
    output_dir="./qwen-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,       # 等效batch_size=16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=False,
    bf16=True,                           # A100/H100用bf16
    optim="paged_adamw_32bit",           # 分页优化器，节省显存
    report_to="wandb",                   # 实验跟踪
    run_name="qwen-lora-v1"
)

# ── 步骤6：开始训练 ──
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    formatting_func=format_instruction,
    max_seq_length=2048,
    packing=False                        # True可提升GPU利用率
)

trainer.train()

# ── 步骤7：保存LoRA权重 ──
model.save_pretrained("./qwen-lora-adapter")
tokenizer.save_pretrained("./qwen-lora-adapter")

print("微调完成！权重保存在 ./qwen-lora-adapter")
```

---

## 3. 数据集格式与构建

```python
# 指令数据集格式（最常用）
train_data = [
    {
        "instruction": "你是一个客服助手，请回答用户问题",
        "input": "我的订单什么时候到？",
        "output": "您好！请提供您的订单号，我来为您查询物流信息。"
    },
    # ...更多数据
]

# 对话格式（多轮对话）
chat_data = [
    {
        "messages": [
            {"role": "system", "content": "你是专业的客服助手"},
            {"role": "user", "content": "我想退款"},
            {"role": "assistant", "content": "好的，请问您的订单号是多少？"},
            {"role": "user", "content": "订单号是12345"},
            {"role": "assistant", "content": "收到，您的订单12345已申请退款，预计3-5个工作日到账。"}
        ]
    }
]

# 数据质量原则
# 1. 质量 > 数量：1000条高质量 > 10000条低质量
# 2. 多样性：覆盖目标场景的各种情况
# 3. 均衡性：各类别数据量相近
# 4. 去重：用MinHash去除近重复数据
```

---

## 4. 合并与部署

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 加载基础模型（全精度，用于合并）
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu"
)

# 加载LoRA权重并合并
model = PeftModel.from_pretrained(base_model, "./qwen-lora-adapter")
merged_model = model.merge_and_unload()  # 合并：LoRA权重融入基础模型

# 保存合并后的完整模型
merged_model.save_pretrained("./qwen-merged")
tokenizer.save_pretrained("./qwen-merged")

# 推理测试
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./qwen-merged",
    tokenizer=tokenizer,
    device_map="auto"
)
result = pipe("你是一个客服助手。\n用户：我想退款\n助手：", max_new_tokens=200)
print(result[0]["generated_text"])
```

---

## 5. DPO 偏好对齐

SFT之后，用DPO让模型更符合人类偏好（减少有害输出，更有帮助）：

```python
from trl import DPOTrainer, DPOConfig

# DPO数据格式：每条包含prompt + chosen(好回答) + rejected(差回答)
dpo_data = [
    {
        "prompt": "如何提高工作效率？",
        "chosen": "可以尝试番茄工作法，专注25分钟休息5分钟...",  # 有帮助的回答
        "rejected": "我不知道，工作效率很难提高。"                # 没帮助的回答
    }
]

dpo_config = DPOConfig(
    output_dir="./qwen-dpo",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-5,              # DPO学习率比SFT低
    beta=0.1,                        # DPO温度参数，控制偏好强度
    bf16=True,
)

dpo_trainer = DPOTrainer(
    model=sft_model,                 # SFT后的模型
    ref_model=ref_model,             # 参考模型（SFT模型的副本，冻结）
    args=dpo_config,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()
```

---

## 6. 面试高频问题

**Q: LoRA的r（秩）怎么选？**
> r=8适合大多数场景；任务简单（分类/格式转换）用r=4；任务复杂（专业领域知识）用r=16-32；不要超过64，收益递减明显。

**Q: QLoRA和LoRA的主要差别？**
> QLoRA在LoRA基础上，将基础模型权重用4bit量化存储，显存从LoRA的16GB降到8GB，训练速度略慢（约10-20%），效果损失约1-3%。

**Q: 微调后模型效果差怎么排查？**
1. 检查数据质量（最常见原因）：格式是否正确、输出质量是否够高
2. 检查学习率：太大（训练损失震荡）→降低；太小（收敛慢）→提高
3. 检查数据量：领域数据太少（< 500条）→扩充数据
4. 检查target_modules：确认应用了正确的层
5. 和基础模型对比：测试基础模型能力，确认问题在微调

**Q: DPO 和 RLHF 哪个更好？**
> DPO更简单（无需奖励模型和PPO训练），训练稳定，是2026年主流选择；RLHF（PPO）理论上效果更好，但工程复杂度高（需要奖励模型+4个模型同时运行），大厂核心模型才用。

---

[← Harness架构设计](./harness-architecture) | [→ 推理加速与量化](./inference-optimization)
