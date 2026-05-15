---
layout: default
title: "D5 · 对齐综合实战"
render_with_liquid: false
---

# D5 · 对齐综合实战：LLaMA-Factory 全流程

> **目标**：用 LLaMA-Factory 走完 SFT → DPO 完整对齐流程，训练一个能通过基础面试问题的 1.5B 模型。

---

## 一、LLaMA-Factory 快速上手

```bash
# 安装
pip install llamafactory

# 查看支持的模型和训练方法
llamafactory-cli help
```

### 1.1 数据格式

LLaMA-Factory 支持多种格式，最通用的是 ShareGPT 格式：

```json
[
  {
    "conversations": [
      {"from": "human", "value": "什么是注意力机制？"},
      {"from": "gpt", "value": "注意力机制允许模型在处理序列时..."},
      {"from": "human", "value": "能给个代码例子吗？"},
      {"from": "gpt", "value": "当然，以下是 PyTorch 实现..."}
    ]
  }
]
```

---

## 二、配置文件驱动训练

```yaml
# sft_config.yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct

# 数据
dataset: alpaca_zh,code_alpaca_en    # 中英混合
template: qwen                       # 使用 Qwen 的 chat template
cutoff_len: 2048

# 训练方法
stage: sft
finetuning_type: lora

# LoRA 配置
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.1
lora_target: all   # 应用到所有线性层

# 训练超参
output_dir: ./outputs/sft
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
logging_steps: 10
save_steps: 500
```

```yaml
# dpo_config.yaml
model_name_or_path: ./outputs/sft  # 从 SFT 模型继续

dataset: dpo_zh_demo
template: qwen
cutoff_len: 2048

stage: dpo
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all

# DPO 特有参数
pref_beta: 0.1           # β 系数
pref_loss: sigmoid       # 损失类型

output_dir: ./outputs/dpo
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 5.0e-7    # DPO 用更小的 lr
num_train_epochs: 1
```

```bash
# 启动训练
llamafactory-cli train sft_config.yaml
llamafactory-cli train dpo_config.yaml

# 合并 LoRA 权重
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_name_or_path ./outputs/dpo \
  --export_dir ./outputs/merged \
  --export_size 2

# 评估
llamafactory-cli eval \
  --model_name_or_path ./outputs/merged \
  --eval_dataset mmlu \
  --template qwen
```

---

## 三、自定义数据集构建

```python
import json
from pathlib import Path

# 构建面试问答数据集
interview_qa = [
    {
        "conversations": [
            {
                "from": "human",
                "value": "请解释 Transformer 中的 Multi-Head Attention 机制，并说明为什么需要多个头？"
            },
            {
                "from": "gpt",
                "value": """Multi-Head Attention（多头注意力）是 Transformer 的核心组件。

**基本原理：**
多头注意力将输入投影到多个不同的子空间，在每个子空间独立计算注意力，然后拼接所有头的输出：

```python
def multi_head_attention(Q, K, V, num_heads, d_model):
    d_k = d_model // num_heads
    
    # 分割为多个头
    Q_heads = Q.view(batch, seq, num_heads, d_k)
    K_heads = K.view(batch, seq, num_heads, d_k)
    V_heads = V.view(batch, seq, num_heads, d_k)
    
    # 每个头独立计算 Attention
    attention_outputs = scaled_dot_product_attention(Q_heads, K_heads, V_heads)
    
    # 拼接所有头
    return concat_and_project(attention_outputs)
```

**为什么需要多个头？**
1. **捕获不同类型的依赖关系**：不同头可以关注不同的语言现象（句法、语义等）
2. **增加模型容量**：多个注意力子空间提供更丰富的表示
3. **实验验证**：论文发现不同头确实关注不同的模式（如近距离/远距离依赖）

**面试要点：**
- h 个头，每个头维度 d_k = d_model/h，总计算量与单头相同
- 实际实现中通过 reshape 操作并行计算所有头"""
            }
        ]
    },
    {
        "conversations": [
            {
                "from": "human",
                "value": "LoRA 的低秩分解原理是什么？为什么这种方法有效？"
            },
            {
                "from": "gpt", 
                "value": """LoRA（Low-Rank Adaptation）基于一个重要假设：**微调时权重的变化是低秩的**。

**数学原理：**
原始权重矩阵 W ∈ R^{d×k}，LoRA 将增量 ΔW 分解为两个低秩矩阵的乘积：

```
W' = W + ΔW = W + BA
其中 B ∈ R^{d×r}, A ∈ R^{r×k}，r << min(d,k)
```

**为什么有效？**
1. **内在维度假设**：研究（Aghajanyan et al., 2020）发现，语言模型的微调在本质上是在一个低维子空间中进行的，即使有数十亿参数，有效的"学习"集中在少数方向上
2. **预训练权重已捕获大量知识**：微调只需要小幅调整，不需要完整的 rank-d 更新
3. **经验验证**：在多个 benchmark 上，LoRA r=16 的效果与全量微调相当

**参数量对比（以 d=k=4096 为例）：**
- 全量微调：16M 参数
- LoRA r=16：128K 参数（减少 125 倍）

**初始化技巧：** B=0, A~N(0,σ²)，保证初始 ΔW=0，不破坏预训练知识"""
            }
        ]
    }
]

# 保存为 LLaMA-Factory 格式
output_path = Path("./data/llm_interview_qa.json")
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(interview_qa, f, ensure_ascii=False, indent=2)

print(f"已创建 {len(interview_qa)} 条训练数据")
```

---

## 四、评估框架

```python
"""
评估对齐模型的常用指标和数据集

1. MT-Bench：多轮对话质量（GPT-4 打分）
2. AlpacaEval：指令跟随能力
3. MMLU：知识理解
4. HumanEval：代码生成
5. TruthfulQA：事实准确性
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ModelEvaluator:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    
    def chat(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        """多轮对话"""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        
        new_tokens = output[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def evaluate_single(self, question: str) -> str:
        """单轮评估"""
        return self.chat([{"role": "user", "content": question}])
    
    def run_benchmark(self, questions: list[str]) -> list[dict]:
        """批量评估"""
        results = []
        for q in questions:
            answer = self.evaluate_single(q)
            results.append({'question': q, 'answer': answer})
        return results

# 面试题评估
questions = [
    "请解释 Transformer 的 Attention 机制",
    "什么是 LoRA？为什么高效？",
    "RLHF 和 DPO 的区别是什么？",
    "如何解决大模型推理的显存问题？",
]

# evaluator = ModelEvaluator("./outputs/merged")
# results = evaluator.run_benchmark(questions)
# for r in results:
#     print(f"Q: {r['question']}")
#     print(f"A: {r['answer'][:200]}...")
#     print()
```

---

## 五、常见问题 Troubleshooting

```python
"""
SFT/DPO 训练常见问题

1. Loss 不下降
   - 检查 label 是否正确（user 部分是否 mask 了）
   - learning rate 是否太大/太小
   - 数据格式是否正确

2. 模型输出格式混乱
   - chat template 不匹配
   - 确认使用了正确的 template 名称

3. 显存不足
   - 开启 gradient_checkpointing
   - 减小 batch_size，增大 gradient_accumulation_steps
   - 使用 QLoRA（4bit 量化）

4. DPO 训练后模型"废了"
   - beta 太小导致过拟合偏好数据
   - 训练 epoch 太多
   - 建议 beta=0.1, epoch=1, lr=5e-7

5. LoRA 合并后效果下降
   - 检查合并是否正确（merged weights = W + α/r * BA）
   - 确认推理时没有额外加载 LoRA adapter
"""
```

---

## 小结

本周我们完整覆盖了大模型对齐的技术栈：

```
预训练模型
    ↓ SFT（指令微调）
基础对话模型
    ↓ DPO/RLHF（偏好对齐）
对齐后的助手模型

关键工具链：
- 数据处理：datasets, 自定义脚本
- 训练：LLaMA-Factory, TRL
- 评估：lm-evaluation-harness, MT-Bench
- 部署：vLLM, Ollama
```
