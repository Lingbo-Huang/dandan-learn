---
layout: default
title: "Week 8 周规划：RNN 与序列模型"
---

# Week 8 · RNN 与序列模型周规划

## 本周目标

掌握序列模型的发展脉络，从 RNN 到 LSTM/GRU，再到 Seq2Seq 和 Attention，理解 Transformer 的前身。

## 学习路线

| 天 | 主题 | 核心收获 |
|----|------|---------|
| D1 | RNN 基础 | 循环结构、时序反向传播、梯度消失 |
| D2 | LSTM 详解 | 四个门的数学推导、记忆单元 |
| D3 | GRU 与变体 | 简化版 LSTM、双向 RNN |
| D4 | Seq2Seq 模型 | Encoder-Decoder 架构 |
| D5 | Attention 机制 | 注意力权重推导、Self-Attention 入门 |
| D6 | 综合实战 | LSTM 情感分析 + 文本生成 |

## 知识图谱

```
序列模型进化
RNN（梯度消失）
 └→ LSTM（记忆门控）→ GRU（简化）
      └→ Seq2Seq + Attention
            └→ Transformer（Self-Attention）→ GPT/BERT
```

## 面试必备问题

1. RNN 梯度消失和 DNN 梯度消失有什么区别？
2. LSTM 的四个门分别有什么作用？遗忘门全为0/1会怎样？
3. GRU 和 LSTM 各有什么优缺点？
4. Attention 机制的本质是什么？Q、K、V 各代表什么？
5. 为什么 Transformer 能替代 RNN 处理序列？
