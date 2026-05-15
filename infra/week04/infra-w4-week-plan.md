---
layout: default
title: "Week 4 周规划：FlashAttention 全解析"
---

# Week 4 周规划：FlashAttention 全解析

## 本周目标

深入理解 FlashAttention 的核心思想——**IO感知的分块注意力计算**，从数学推导到 CUDA 实现，掌握 FlashAttention 1/2/3 的演进脉络，能独立解释其原理并在面试中手推公式。

## 为什么是 FlashAttention？

标准 Self-Attention 的时间/空间复杂度均为 O(N²)，当序列长度 N=8192 时，一个注意力矩阵就需要 8192×8192×2 bytes = **128 MB** 的 HBM 空间，并产生大量 HBM 读写。FlashAttention 通过 tiling + online softmax 将 HBM 访问降低 5-20×，是目前主流 LLM 推理/训练的标配。

## 本周文章安排

| 天 | 文件 | 主题 |
|---|------|------|
| D1 | infra-w4-attention-bottleneck.md | 标准 Attention 的 IO 瓶颈分析 |
| D2 | infra-w4-tiling-online-softmax.md | 分块计算与 Online Softmax 推导 |
| D3 | infra-w4-flashattn1-cuda.md | FlashAttention v1 CUDA 实现详解 |
| D4 | infra-w4-flashattn2-improvements.md | FlashAttention v2 优化点深度解析 |
| D5 | infra-w4-flashattn3-h100.md | FlashAttention v3 与 H100 特性 |
| D6 | infra-w4-capstone.md | Capstone：从零实现简化版 FlashAttention |

## 知识地图

```
Transformer Attention
    ↓
标准实现 O(N²) 内存 → HBM 瓶颈
    ↓
IO 感知算法设计
    ↓
Tiling 分块 + Online Softmax (数值稳定)
    ↓
FlashAttention v1 (2022) → v2 (2023) → v3 (2024)
    ↓
实际部署：xFormers / vLLM / PyTorch SDPA
```

## 前置知识检查

在开始本周内容前，请确认你已掌握：

- [ ] Transformer Self-Attention 的数学定义
- [ ] GPU 内存层次：寄存器 < SRAM(共享内存) < HBM(显存)
- [ ] CUDA 线程层次：Thread / Warp / Block / Grid
- [ ] 矩阵乘法（GEMM）的基本 tiling 思路（Week 1/3 内容）

## 本周面试题预告

1. FlashAttention 为什么能节省内存？核心技巧是什么？
2. Online Softmax 如何保证数值稳定性？推导三遍扫描到两遍扫描的过程
3. FlashAttention v2 相对 v1 做了哪些改进？实际加速比是多少？
4. 为什么 FlashAttention 对长序列更有优势？
5. 在 8K/32K/128K 序列场景下，FlashAttention 带来的收益如何估算？

## 延伸阅读

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135) - Dao et al., 2022
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691) - Dao, 2023
- [FlashAttention-3 论文](https://arxiv.org/abs/2407.08608) - Shah et al., 2024
