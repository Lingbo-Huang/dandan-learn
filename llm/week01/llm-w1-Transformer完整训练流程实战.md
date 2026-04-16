# Day 6：Encoder vs Decoder vs Encoder-Decoder 架构变体对比

---

## 学习目标

- 理解三种 Transformer 变体的结构差异与适用场景
- 对比 BERT（Encoder-only）、GPT（Decoder-only）、T5（Encoder-Decoder）
- 理解 Causal LM vs Masked LM vs Seq2Seq 预训练目标
- 能根据任务类型选择合适的架构

---

## 一、核心知识点

### 1.1 三种架构速览

```
原始 Transformer（Encoder-Decoder）
├── Encoder-only    → BERT、RoBERTa、DeBERTa
├── Decoder-only    → GPT 系列、LLaMA、Qwen、Mistral
└── Encoder-Decoder → T5、BART、mT5
```

### 1.2 Encoder-only 架构（BERT 类）

**结构**：仅保留 Encoder 栈，使用**双向 Self-Attention**（每个位置可看到全部其他位置）。

**预训练目标**：Masked Language Model（MLM）
- 随机遮盖 15% 的 token
- 模型预测被遮盖的 token

```
输入: "The cat [MASK] on the mat"
输出: 预测 [MASK] = "sat"（利用上下文双向信息）
```

**适用任务**：
- 文本分类（情感分析）
- 命名实体识别（NER）
- 问答（Extractive QA）
- 语义相似度

**不适合**：文本生成（无法自回归生成）

### 1.3 Decoder-only 架构（GPT 类）

**结构**：仅保留 Decoder 的 Self-Attention（去掉 Cross-Attention），使用**因果 Mask（单向）**。

**预训练目标**：Causal Language Model（CLM）
- 预测下一个 token

```
输入: "The cat sat"
目标: "cat sat on"  （每个位置预测下一个）
```

**适用任务**：
- 文本生成（聊天、写作）
- 代码生成
- 少样本学习（In-context Learning）
- 指令跟随（经 SFT/RLHF 后）

**现代大模型主流**：GPT-3/4、LLaMA、Qwen、Mistral、Claude 均为此架构。

### 1.4 Encoder-Decoder 架构（T5 类）

**结构**：完整的 Encoder + Decoder，Decoder 通过 Cross-Attention 读取 Encoder 的输出。

**预训练目标**：Span Corruption（T5）
- 遮盖连续 token 片段，预测被遮盖部分

**适用任务**：
- 机器翻译
- 摘要生成
- 问答（Abstractive QA）
- 数据到文本生成

### 1.5 三种架构对比表

| 维度 | Encoder-only | Decoder-only | Encoder-Decoder |
|------|-------------|--------------|-----------------|
| 代表模型 | BERT | GPT, LLaMA | T5, BART |
| Attention 方向 | 双向 | 单向（因果） | Encoder双向/Decoder单向 |
| 预训练目标 | MLM | CLM | Span Corruption |
| 参数效率（生成） | ❌ 不适合 | ✅ 高效 | ✅ 中等 |
| 理解任务表现 | ✅ 最强 | ✅ 强（大规模后） | ✅ 强 |
| 生成任务表现 | ❌ 不适合 | ✅ 最强 | ✅ 强 |
| 主流程度（2024） | 中 | **最高** | 中 |

---

## 二、深入对比：Attention Mask 差异

```python
import torch

seq_len = 5

# Encoder Self-Attention: 全局可见（无 mask）
encoder_mask = torch.ones(seq_len, seq_len)
print("Encoder Mask（全可见）:")
print(encoder_mask)

# Decoder Causal Mask: 只看过去
decoder_mask = torch.tril(torch.ones(seq_len, seq_len))
print("\nDecoder Causal Mask:")
print(decoder_mask)

# 可视化差异
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(encoder_mask, cmap='Blues', vmin=0, vmax=1)
axes[0].set_title("Encoder Attention\n（双向，全可见）")
axes[0].set_xlabel("Key Position")
axes[0].set_ylabel("Query Position")

axes[1].imshow(decoder_mask, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title("Decoder Attention\n（单向，因果Mask）")
axes[1].set_xlabel("Key Position")
axes[1].set_ylabel("Query Position")

plt.tight_layout()
plt.savefig("attention_mask_comparison.png", dpi=100)
plt.show()
```

---

## 三、动手练习

### 练习 1：用 HuggingFace 加载三种架构

```python
from transformers import (
    AutoModel, AutoModelForCausalLM,
    BertModel, GPT2LMHeadModel, T5ForConditionalGeneration,
    BertTokenizer, GPT2Tokenizer, T5Tokenizer
)

# 1. BERT（Encoder-only）
bert = BertModel.from_pretrained("bert-base-uncased")
print(f"BERT 参数量: {sum(p.numel() for p in bert.parameters()):,}")
print(f"BERT 层数: {bert.config.num_hidden_layers}")

# 2. GPT-2（Decoder-only）
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
print(f"\nGPT-2 参数量: {sum(p.numel() for p in gpt2.parameters()):,}")
print(f"GPT-2 层数: {gpt2.config.n_layer}")

# 3. T5-small（Encoder-Decoder）
t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
print(f"\nT5-small 参数量: {sum(p.numel() for p in t5.parameters()):,}")
print(f"T5 Encoder 层数: {t5.config.num_layers}")
print(f"T5 Decoder 层数: {t5.config.num_decoder_layers}")
```

### 练习 2：BERT vs GPT-2 推理对比

```python
import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel

text = "The capital of France is"

# BERT：获取上下文表示
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_inputs = bert_tokenizer(text, return_tensors="pt")
with torch.no_grad():
    bert_out = bert_model(**bert_inputs)
print(f"BERT 输出 shape: {bert_out.last_hidden_state.shape}")
# → (1, seq_len, 768)，每个 token 的上下文表示

# GPT-2：生成下一个 token
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_inputs = gpt2_tokenizer(text, return_tensors="pt")
with torch.no_grad():
    gpt2_out = gpt2_model.generate(
        **gpt2_inputs,
        max_new_tokens=5,
        do_sample=False
    )
print(f"\nGPT-2 生成: {gpt2_tokenizer.decode(gpt2_out[0])}")
```

### 练习 3：架构选择练习

根据以下任务，说明应选哪种架构，并简要解释原因：

1. 电商评论情感分类（正/负/中性）
2. 中英互译
3. AI 助手对话生成
4. 从简历中提取工作经历（NER）
5. 将表格数据转化为自然语言描述

---

## 四、小结

| 项目 | 内容 |
|------|------|
| 今日完成 | 三种 Transformer 架构对比 + HuggingFace 实践 |
| 核心认知 | 没有最好的架构，只有最适合任务的架构；Decoder-only 是当前 LLM 主流 |
| 明日预告 | D7：动手从零实现最小 Transformer + 本周总结与知识梳理 |

> 💡 **2024年现状**：随着 GPT-4、LLaMA 等超大规模 Decoder-only 模型的成功，加上 In-context Learning 能力的涌现，Decoder-only 已成为 LLM 绝对主流，即便是理解任务也往往用指令微调后的大模型来解决。

### 练习 3 参考答案

| 任务 | 推荐架构 | 理由 |
|------|---------|------|
| 情感分类 | Encoder-only (BERT) | 理解任务，需要双向上下文 |
| 机器翻译 | Encoder-Decoder (T5) | 源-目标语言对齐，Seq2Seq |
| 对话生成 | Decoder-only (GPT) | 自回归生成，支持长上下文 |
| 实体识别 | Encoder-only (BERT) | token 级分类，需要双向上下文 |
| 表格→文本 | Encoder-Decoder | 结构化输入映射到自然语言输出 |
