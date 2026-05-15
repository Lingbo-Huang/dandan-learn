---
layout: default
title: "Week 6 周规划：分布式训练进阶"
---

# Week 6 周规划：分布式训练进阶

## 本周目标

深入 Megatron-LM 框架与大规模并行训练的前沿技术：序列并行、专家并行和 MoE 架构。掌握千卡级别训练的系统设计能力。

## 为什么需要进阶并行？

Week2 学了数据并行、模型并行、ZeRO。但现代超大模型（GPT-4、Gemini）的训练远不止这些：
- **GPT-3 175B**：需要同时使用 TP + PP + DP 三维并行
- **Megatron-Turing 530B**：使用 TP=8, PP=35, DP=35 = 9800 卡
- **Mixtral 8×7B**：MoE 架构，实际激活参数只有 12B，但效果接近 70B

## 本周文章安排

| 天 | 文件 | 主题 |
|---|------|------|
| D1 | infra-w6-megatron-3d-parallel.md | Megatron-LM 三维并行：TP+PP+DP |
| D2 | infra-w6-sequence-parallel.md | 序列并行：突破单卡序列长度限制 |
| D3 | infra-w6-pipeline-schedule.md | Pipeline 调度：1F1B 与 Interleaved |
| D4 | infra-w6-moe-architecture.md | MoE 架构：专家路由与负载均衡 |
| D5 | infra-w6-expert-parallel.md | 专家并行与 MoE 训练挑战 |
| D6 | infra-w6-capstone.md | Capstone：设计千卡 LLM 训练系统 |

## 三维并行示意

```
                   Data Parallel (DP=4)
    ─────────────────────────────────────────→
    
    ┌──────────┬──────────┐   ┌──────────┬──────────┐
    │  GPU 0   │  GPU 1   │   │  GPU 4   │  GPU 5   │
 T  │ (TP rank0│(TP rank1)│   │(TP rank0)│(TP rank1)│  Pipeline
 P  │ PP rank0 │ PP rank0 │   │ PP rank0 │ PP rank0 │  Stage 0
 ↕  ├──────────┼──────────┤   ├──────────┼──────────┤
    │  GPU 2   │  GPU 3   │   │  GPU 6   │  GPU 7   │  Pipeline
    │(TP rank0)│(TP rank1)│   │(TP rank0)│(TP rank1)│  Stage 1
    └──────────┴──────────┘   └──────────┴──────────┘
    
    TP=2, PP=2, DP=2 → 共 8 张 GPU
```

## 本周面试题预告

1. Megatron-LM 的 Tensor Parallel 如何切分 Attention 和 FFN？
2. 序列并行和 Tensor 并行的关系是什么？
3. Pipeline Parallel 的 bubble 问题如何缓解？1F1B 调度的原理是什么？
4. MoE 的 Top-K 路由是怎么工作的？如何防止负载不均衡？
5. 专家并行（Expert Parallel）的 All-to-All 通信在哪里发生？

## 延伸阅读

- [Megatron-LM](https://arxiv.org/abs/1909.08053) - Shoeybi et al., 2019
- [序列并行](https://arxiv.org/abs/2205.05198) - Korthikanti et al., 2022
- [Switch Transformer](https://arxiv.org/abs/2101.03961) - Fedus et al., 2021
- [Mixtral](https://arxiv.org/abs/2401.04088) - Jiang et al., 2024
