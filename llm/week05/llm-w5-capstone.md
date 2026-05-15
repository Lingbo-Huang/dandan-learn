---
layout: default
title: "D6 · Capstone：端到端对齐 1.5B 模型"
render_with_liquid: false
---

# D6 · Capstone：端到端对齐一个 1.5B 模型

> **目标**：从 Qwen2.5-1.5B-Instruct 出发，走完 SFT → DPO 完整流程，训练一个专精 LLM 技术面试问答的助手。

---

## 一、项目规划

```
数据准备 → SFT 微调 → DPO 对齐 → 合并权重 → 评估 → 部署
```

**资源需求**：
- 单张 RTX 4090 (24GB) 或 A100 40GB
- SFT 训练时间：约 2-3 小时（500条数据）
- DPO 训练时间：约 1 小时

---

## 二、数据集构建

```python
# build_dataset.py
"""
构建 LLM 面试问答数据集

SFT 数据：500条问答（涵盖 W1-W5 所有主题）
DPO 数据：200条偏好对（好答案 vs 差答案）
"""

import json
import random
from pathlib import Path

# SFT 数据模板
SFT_TEMPLATES = {
    "概念解释": [
        "什么是{concept}？请解释其原理和应用场景。",
        "请深入解释{concept}的工作机制，并给出代码示例。",
        "从面试角度，如何回答关于{concept}的问题？",
    ],
    "对比分析": [
        "{concept_a}和{concept_b}有什么区别？各自的优缺点是什么？",
        "在什么情况下选择{concept_a}而不是{concept_b}？",
    ],
    "实战应用": [
        "如何在生产环境中使用{concept}？需要注意哪些问题？",
        "实现{concept}时有哪些常见的坑？如何避免？",
    ]
}

# 高质量答案特征：
# 1. 先给结论（BLUF）
# 2. 公式/代码支撑
# 3. 对比（vs 其他方法）
# 4. 实际应用场景
# 5. 面试注意点

def create_sft_sample(question: str, answer: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer}
        ]
    }

def create_dpo_sample(question: str, good_answer: str, bad_answer: str) -> dict:
    return {
        "conversations": [{"from": "human", "value": question}],
        "chosen": [{"from": "gpt", "value": good_answer}],
        "rejected": [{"from": "gpt", "value": bad_answer}]
    }

# 示例 DPO 数据
dpo_example = create_dpo_sample(
    question="什么是 Attention 机制？",
    
    good_answer="""Attention 机制让模型在处理序列中的每个位置时，能够关注输入序列的不同部分，并根据相关性动态加权。

**核心公式：**
$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

**直觉理解：**
- Q（Query）：当前想要查询的内容
- K（Key）：可被查询的索引
- V（Value）：实际的值
- $\\sqrt{d_k}$ 缩放：防止内积过大导致 softmax 梯度消失

**代码实现：**
```python
def attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ V
```

**面试要点：**
- 为什么除以 sqrt(d_k)？防止点积过大使 softmax 进入饱和区
- 时间复杂度 O(n²d)，n 是序列长度""",
    
    bad_answer="""Attention 机制就是让模型关注重要的部分。它通过计算 Query、Key、Value 来工作。公式是 softmax(QK^T/sqrt(d_k))V。这在 Transformer 中很重要。"""
)
```

---

## 三、训练脚本

```bash
#!/bin/bash
# run_training.sh

MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_BASE="./outputs/llm_interview_bot"

echo "=== Step 1: SFT 微调 ==="
llamafactory-cli train \
  --model_name_or_path $MODEL_NAME \
  --dataset llm_interview_sft \
  --template qwen \
  --finetuning_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_target all \
  --output_dir $OUTPUT_BASE/sft \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  --logging_steps 10

echo "=== Step 2: DPO 对齐 ==="
llamafactory-cli train \
  --model_name_or_path $MODEL_NAME \
  --adapter_name_or_path $OUTPUT_BASE/sft \
  --dataset llm_interview_dpo \
  --template qwen \
  --stage dpo \
  --finetuning_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_target all \
  --pref_beta 0.1 \
  --pref_loss sigmoid \
  --output_dir $OUTPUT_BASE/dpo \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-7 \
  --num_train_epochs 1 \
  --bf16 \
  --logging_steps 5

echo "=== Step 3: 合并权重 ==="
llamafactory-cli export \
  --model_name_or_path $MODEL_NAME \
  --adapter_name_or_path $OUTPUT_BASE/dpo \
  --export_dir $OUTPUT_BASE/merged \
  --export_size 2

echo "=== 训练完成 ==="
```

---

## 四、交互式评估

```python
# evaluate_chatbot.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def interactive_chat(model_path: str):
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    print("LLM 面试助手已就绪！输入 'quit' 退出\n")
    history = []
    
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == 'quit':
            break
        if not user_input:
            continue
        
        history.append({"role": "user", "content": user_input})
        
        text = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        
        response = tokenizer.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        history.append({"role": "assistant", "content": response})
        print(f"\n助手: {response}\n")

# if __name__ == "__main__":
#     interactive_chat("./outputs/llm_interview_bot/merged")
```

---

## 五、本周总结

### 知识图谱

```
预训练模型（Week 4）
    │
    ▼
SFT 指令微调
  ├─ 数据格式（Chat Template）
  ├─ 损失掩码（只算 assistant 部分）
  └─ 训练参数（lr=2e-5, epoch=3）
    │
    ▼
参数高效微调
  ├─ LoRA（低秩分解，r=16）
  └─ QLoRA（4bit量化+LoRA，5GB显存）
    │
    ▼
对齐微调
  ├─ RLHF（SFT→RM→PPO，复杂但效果好）
  └─ DPO（直接偏好优化，简单稳定）
    │
    ▼
部署就绪的助手模型
```

### Week 5 面试题汇总

| 问题 | 要点 |
|------|------|
| SFT 数据格式 | Chat Template + 损失掩码 |
| LoRA 原理 | ΔW = BA，B 初始化为零 |
| QLoRA 为何省显存 | 权重 4bit 存储，计算时 BF16 |
| DPO 推导 | 从 RLHF 闭合解推导 |
| β 的作用 | 控制偏离参照模型的程度 |
| 选择 DPO vs RLHF | 资源、数据、效果三角权衡 |
