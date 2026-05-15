---
layout: default
title: "Week 4 周规划 · 预训练"
render_with_liquid: false
---

# 大模型线 Week 4 周规划总览

**主题：预训练——数据、Tokenizer、GPT 目标与 Scaling Law**  
**周期：Day 1 - Day 7**

---

## 本周目标

深入理解大模型预训练全链路，从数据处理、分词器设计，到 GPT 的训练目标与 Scaling Law，能够：

- 解释预训练数据处理流水线（去重、清洗、配比）
- 手写 BPE Tokenizer 的训练与编解码
- 推导 GPT 的语言建模目标（Next Token Prediction）
- 理解并会用 Scaling Law 指导模型规模设计
- 复现一个 mini-GPT 的预训练训练循环

---

## 每日主题速览

| Day | 主题 | 关键词 |
|-----|------|--------|
| D1 | 预训练数据处理 | 去重、质量过滤、数据配比、Common Crawl |
| D2 | Tokenizer 原理与实战 | BPE、WordPiece、SentencePiece、tiktoken |
| D3 | GPT 预训练目标 | CLM、NTP、交叉熵损失、困惑度 |
| D4 | Scaling Law | Chinchilla、计算最优、FLOPs 估算 |
| D5 | 位置编码进阶 | RoPE、ALiBi、NTK 外推 |
| D6 | Capstone | 手写 mini-GPT 预训练 |

---

## 面试高频题

1. BPE 和 WordPiece 的区别？
2. 为什么 GPT 用 CLM 而不是 MLM？
3. 如何估算一个模型训练需要的 FLOPs？
4. Scaling Law 说数据和参数如何权衡？
5. 分词粒度太细 vs 太粗的 trade-off 是什么？

---

## 参考资料

- [GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- [Chinchilla Paper](https://arxiv.org/abs/2203.15556)
- [RoPE Paper](https://arxiv.org/abs/2104.09864)
- [Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
