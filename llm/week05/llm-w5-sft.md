---
layout: default
title: "D1 · 指令微调 (SFT)"
render_with_liquid: false
---

# D1 · 指令微调 (SFT)

> **SFT 的本质**：用"问题-答案"对告诉模型"怎么回答问题"，而不是"下一个词是什么"。

---

## 一、为什么需要 SFT？

预训练模型是"文本补全机器"，给它 "Q: 什么是光合作用？A:" 它可能会继续生成 "Q: 什么是呼吸作用？A:"（因为训练数据里这样的问答格式很常见）。

SFT 让模型学会"接收指令 → 执行指令"的对话模式。

---

## 二、数据格式：Chat Template

### Llama-3 格式

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

什么是光合作用？<|eot_id|><|start_header_id|>assistant<|end_header_id|>

光合作用是植物利用阳光...<|eot_id|>
```

### ChatML 格式（Qwen 使用）

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
什么是光合作用？<|im_end|>
<|im_start|>assistant
光合作用是植物利用阳光...<|im_end|>
```

---

## 三、损失掩码（关键细节）

SFT 的核心：**只在 assistant 回答部分计算 loss，忽略 user/system 部分**。

```python
import torch
from transformers import AutoTokenizer

def create_sft_input(
    tokenizer,
    messages: list[dict],
    max_length: int = 2048
) -> dict:
    """
    创建 SFT 训练输入，只在 assistant 回复处计算 loss
    
    messages: [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ]
    """
    # 使用 tokenizer 自带的 chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Tokenize
    encoding = tokenizer(
        input_text,
        max_length=max_length,
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'][0]
    labels = input_ids.clone()
    
    # 找到 assistant 回复的起始位置，对其之前的 token 设置 -100
    # 方法：逐条 message 确定位置
    
    # 构建只含 user/system 部分的文本（不含最后一条 assistant）
    non_assistant_msgs = [m for m in messages if m['role'] != 'assistant']
    
    # 简化方法：找 assistant 开始的 token 位置
    assistant_start_tokens = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False
    )
    
    # 在 input_ids 中找这个序列
    ids_list = input_ids.tolist()
    mask_end = 0
    
    for i in range(len(ids_list) - len(assistant_start_tokens)):
        if ids_list[i:i+len(assistant_start_tokens)] == assistant_start_tokens:
            mask_end = i + len(assistant_start_tokens)
            break
    
    # 对 user/system 部分设置 -100（不计算 loss）
    labels[:mask_end] = -100
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': encoding['attention_mask'][0]
    }


# 演示
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
# 
# messages = [
#     {"role": "system", "content": "你是一个有帮助的助手"},
#     {"role": "user", "content": "1+1等于几？"},
#     {"role": "assistant", "content": "1+1等于2。"},
# ]
# 
# sample = create_sft_input(tokenizer, messages)
# print(f"input_ids shape: {sample['input_ids'].shape}")
# print(f"labels (非-100 的位置): {(sample['labels'] != -100).sum().item()} tokens")
```

---

## 四、SFT 训练实现

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import torch

def run_sft(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    dataset_name: str = "tatsu-lab/alpaca",
    output_dir: str = "./sft_output",
    num_train_epochs: int = 3,
):
    """基于 TRL 的 SFT 训练"""
    
    # 加载模型和 tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    dataset = load_dataset(dataset_name, split="train")
    
    def format_alpaca(example):
        """Alpaca 格式转 ChatML"""
        if example.get('input'):
            user_msg = f"{example['instruction']}\n\n{example['input']}"
        else:
            user_msg = example['instruction']
        
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": example['output']},
        ]
        return {"messages": messages}
    
    dataset = dataset.map(format_alpaca)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        dataloader_num_workers=4,
    )
    
    # SFT Trainer（自动处理 chat template 和损失掩码）
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=2048,
    )
    
    trainer.train()
    trainer.save_model()
    print(f"模型保存到 {output_dir}")
```

---

## 五、高质量 SFT 数据的特征

### LIMA 论文的关键发现
1000 条高质量数据 > 50000 条低质量数据

```python
# 高质量 SFT 数据的构建策略

class SFTDataQualityChecker:
    """SFT 数据质量检查器"""
    
    def check_response_quality(self, response: str) -> dict:
        issues = []
        
        # 1. 回答是否真的解决了问题
        if len(response.split()) < 10:
            issues.append("回答太短")
        
        # 2. 是否有格式问题
        if response.count("```") % 2 != 0:
            issues.append("代码块不匹配")
        
        # 3. 是否包含"作为 AI 我无法..."类型的拒绝
        ai_refusals = ["as an AI", "作为一个AI", "我无法", "我不能访问"]
        if any(r in response for r in ai_refusals):
            issues.append("可能包含不必要的拒绝")
        
        # 4. 回答质量分
        score = 1.0 - len(issues) * 0.2
        return {'score': max(0, score), 'issues': issues}
    
    def diversity_check(self, instructions: list[str]) -> float:
        """检查指令多样性（基于 n-gram 重叠）"""
        from collections import Counter
        all_bigrams = []
        for inst in instructions:
            words = inst.lower().split()
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
            all_bigrams.extend(bigrams)
        
        counter = Counter(all_bigrams)
        unique_ratio = len(counter) / max(len(all_bigrams), 1)
        return unique_ratio  # 越高越多样
```

---

## 六、面试题精讲

**Q: SFT 时为什么要对 user 部分设置 label=-100？**

A: 交叉熵损失中，label=-100 的位置会被自动跳过（PyTorch 的 `ignore_index` 参数）。我们只想让模型学会"如何回答"，而不是学会"如何提问"。如果对 user 部分也计算 loss，会导致模型的回复风格更像提问，而不是回答。

**Q: SFT 与预训练的 loss 设置有何不同？**

A: 预训练：对所有 token 计算 loss（下一个词预测），label = input_ids shift 一位。SFT：只对 assistant 回复部分的 token 计算 loss，其余设为 -100。两者都用交叉熵，但计算范围不同。

---

## 小结

| 要点 | 说明 |
|------|------|
| Chat Template | 不同模型有不同格式，用 tokenizer.apply_chat_template |
| 损失掩码 | user/system 部分 label=-100，不参与 loss |
| 数据质量 | 1000 条精品 > 50000 条低质 (LIMA) |
| 训练参数 | lr 比预训练小 10-100 倍（1e-5 量级）|
