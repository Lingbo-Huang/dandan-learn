---
layout: default
title: "Week 7 周规划：量化与压缩"
---

# Week 7 周规划：量化与压缩

## 本周目标

掌握模型量化的核心原理与主流算法（INT8/INT4/AWQ/GPTQ/SmoothQuant），理解精度-速度-内存的三角权衡，能够在实际项目中选择合适的量化方案。

## 为什么量化如此重要？

量化将 FP16（16位）权重压缩为 INT8（8位）或 INT4（4位）：
- **内存**：减少 2× 或 4×
- **速度**：INT8 矩阵乘法通常比 FP16 快 2-4×（利用整型 Tensor Core）
- **代价**：一定的精度损失

以 LLaMA-2 70B 为例：
- FP16：140 GB（需要 2× A100 80GB）
- INT8：70 GB（可用 1× A100 80GB）
- INT4：35 GB（可用 4× 48GB GPU 或 2× A10）

量化使大模型在消费级 GPU 上运行成为可能！

## 本周文章安排

| 天 | 文件 | 主题 |
|---|------|------|
| D1 | infra-w7-quantization-basics.md | 量化基础：数值表示与量化误差 |
| D2 | infra-w7-smoothquant.md | SmoothQuant：激活量化的挑战与解决 |
| D3 | infra-w7-gptq.md | GPTQ：基于 Hessian 的权重量化 |
| D4 | infra-w7-awq.md | AWQ：激活感知权重量化 |
| D5 | infra-w7-practical-quantization.md | 实战：bitsandbytes / AutoGPTQ / llama.cpp |
| D6 | infra-w7-capstone.md | Capstone：量化方案选型与精度评估 |

## 量化算法谱系

```
量化时机：
  训练后量化（PTQ）：GPTQ, AWQ, SmoothQuant, bitsandbytes  ← 本周重点
  量化感知训练（QAT）：QLoRA 的 NF4, LLM-QAT
  
量化目标：
  仅权重量化（Weight-Only）：AWQ, GPTQ, bitsandbytes NF4
  权重+激活量化（W+A Quant）：SmoothQuant, LLM.int8()
  
量化粒度：
  Per-tensor：整个权重矩阵一个缩放因子（最粗）
  Per-channel：每行/列一个缩放因子
  Per-group：每 128 个元素一个缩放因子（AWQ/GPTQ 常用）
  Per-token：每个 token 一个缩放因子（激活量化）
```

## 本周面试题预告

1. INT8 量化为什么不总是 2× 加速？在什么情况下有加速，什么情况下没有？
2. SmoothQuant 的核心洞察是什么？为什么不直接对激活值量化？
3. GPTQ 算法的核心原理是什么？比朴素 round-to-nearest 好在哪里？
4. AWQ 如何在不使用 Hessian 的情况下实现接近 GPTQ 的精度？
5. INT4 vs INT8：在什么场景下选哪个？
