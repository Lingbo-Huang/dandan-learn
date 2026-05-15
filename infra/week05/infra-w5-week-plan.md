---
layout: default
title: "Week 5 周规划：LLM 推理系统优化"
---

# Week 5 周规划：LLM 推理系统优化

## 本周目标

深入 LLM 推理系统的核心优化技术：从 KV Cache 管理到批处理策略，再到投机解码，掌握当前主流推理框架（vLLM、TensorRT-LLM）背后的关键技术。

## 为什么推理优化如此重要？

训练一次，推理百万次。对于生产级 LLM 服务：
- GPT-3 175B 单次推理需要约 350GB 显存（若朴素实现）
- 推理成本占 LLM 公司运营成本的 70-90%
- 用户体验直接取决于首 Token 延迟（TTFT）和生成速度（TPS）

## 本周文章安排

| 天 | 文件 | 主题 |
|---|------|------|
| D1 | infra-w5-kvcache-fundamentals.md | KV Cache 基础与内存管理挑战 |
| D2 | infra-w5-pagedattention.md | PagedAttention：虚拟内存思想的 GPU 应用 |
| D3 | infra-w5-continuous-batching.md | 连续批处理：从静态到动态的革命 |
| D4 | infra-w5-speculative-decoding.md | 投机解码：用小模型加速大模型 |
| D5 | infra-w5-medusa-lookahead.md | Medusa / Lookahead 解码进阶 |
| D6 | infra-w5-capstone.md | Capstone：设计一个 LLM 推理服务 |

## LLM 推理的两个阶段

```
Prefill（预填充）阶段：
  输入：完整 Prompt（N tokens）
  输出：第一个 Token + 所有层的 KV Cache
  特点：Compute-Bound（并行计算所有位置）
  
Decode（解码）阶段：
  输入：上一个 Token + KV Cache
  输出：下一个 Token（更新 KV Cache）
  特点：Memory-Bound（每次只生成1个 token）
  
关键指标：
  TTFT (Time To First Token): Prefill 时间
  TBT (Time Between Tokens): Decode 每步时间
  TPS (Tokens Per Second): 吞吐量
```

## 本周面试题预告

1. PagedAttention 解决了什么问题？与操作系统虚拟内存有什么相似之处？
2. 为什么 LLM 推理是 Memory-Bound 的？和 Prefill 阶段有什么不同？
3. 连续批处理（Continuous Batching）比传统批处理优势在哪里？
4. 投机解码（Speculative Decoding）的正确性如何保证？加速比怎么估算？
5. Medusa 和普通投机解码有什么区别？

## 延伸阅读

- [vLLM 论文](https://arxiv.org/abs/2309.06180) - Kwon et al., 2023
- [Speculative Decoding](https://arxiv.org/abs/2211.17192) - Chen et al., 2022
- [Medusa](https://arxiv.org/abs/2401.10774) - Cai et al., 2024
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
