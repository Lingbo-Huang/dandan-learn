---
layout: default
title: "Week 6 周规划 · 推理与部署"
render_with_liquid: false
---

# 大模型线 Week 6 周规划总览

**主题：推理与部署——KV Cache、量化、vLLM、投机采样**  
**周期：Day 1 - Day 6**

---

## 本周目标

掌握大模型高效推理的核心技术，能够在生产环境中部署和优化推理系统：

- 理解 KV Cache 的原理和内存管理策略
- 掌握 INT8/INT4 量化方法和权衡
- 理解 vLLM 的 PagedAttention 核心思想
- 理解投机采样的原理和适用场景
- 能够搭建一个生产级 LLM 推理服务

---

## 每日主题速览

| Day | 主题 | 关键词 |
|-----|------|--------|
| D1 | KV Cache | 推理加速、内存消耗、GQA/MQA |
| D2 | 量化 | INT8/INT4、GPTQ、AWQ、精度损失 |
| D3 | vLLM | PagedAttention、连续批处理、吞吐量 |
| D4 | 投机采样 | Draft Model、验证、加速比 |
| D5 | 推理优化综合 | Flash Attention、BatchSize、延迟 vs 吞吐 |
| D6 | Capstone | 搭建生产级推理服务 |

---

## 面试高频题

1. KV Cache 是什么？为什么能加速推理？
2. GQA/MQA 相比 MHA 省了什么？
3. INT4 量化的误差来自哪里？如何补偿？
4. vLLM 的 PagedAttention 解决了什么问题？
5. 投机采样的加速比取决于什么？
6. 如何选择批处理大小来平衡延迟和吞吐量？

---

## 参考资料

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [Speculative Sampling Paper](https://arxiv.org/abs/2302.01318)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [GQA Paper](https://arxiv.org/abs/2305.13245)
