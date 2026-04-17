# Week 2 学习计划：分布式训练基础 & DeepSpeed 入门

> **所属系列**：AI Infra 学习路线 · Phase 1 基础夯实  
> **本周主题**：分布式训练基础——数据并行 / 模型并行 / ZeRO / DeepSpeed  
> **难度等级**：⭐⭐⭐☆☆  
> **预计时长**：8–10 小时

---

## 🗺️ 本周全景图

```
分布式训练全景
├── 为什么需要分布式？           ← 单卡瓶颈分析
├── 数据并行 (Data Parallelism)  ← DDP 原理与实战
├── 模型并行 (Model Parallelism) ← 张量并行 / 流水线并行
├── ZeRO 优化器                  ← 显存碎片的终结者
├── DeepSpeed 入门               ← 一站式分布式框架
├── 混合精度训练                 ← FP16/BF16/FP8 实战
└── Capstone 项目                ← 端到端分布式训练实战
```

---

## 📅 每日安排

| 天 | 主题 | 文件 | 预计时长 |
|----|------|------|---------|
| Day 1 | 分布式训练总览 | `infra-w2-distributed-overview.md` | 1.5h |
| Day 2 | 数据并行深入 | `infra-w2-data-parallelism.md` | 1.5h |
| Day 3 | 模型并行技术 | `infra-w2-model-parallelism.md` | 2h |
| Day 4 | ZeRO 显存优化 | `infra-w2-zero-stages.md` | 1.5h |
| Day 5 | DeepSpeed 入门 | `infra-w2-deepspeed-intro.md` | 1.5h |
| Day 6 | 混合精度训练 | `infra-w2-mixed-precision.md` | 1h |
| Day 7 | Capstone 项目 | `infra-w2-capstone.md` | 2h |

---

## 🎯 学习目标

完成本周后，你应该能够：

- [ ] 说清楚数据并行与模型并行的核心区别与适用场景
- [ ] 手写一个基于 PyTorch DDP 的多 GPU 训练脚本
- [ ] 理解 ZeRO Stage 1/2/3 的显存优化机制
- [ ] 用 DeepSpeed 配置文件启动一个分布式训练任务
- [ ] 解释混合精度训练的数值稳定性挑战与解决方案
- [ ] 完成一个端到端的分布式 LLM 微调实战项目

---

## 🔗 前置知识

- Week 1：PyTorch 基础、自动微分、Transformer 结构
- 了解 GPU 架构基础（CUDA cores、显存带宽）

---

## 📚 推荐资源

| 资源 | 类型 | 重要度 |
|------|------|-------|
| [DeepSpeed 官方文档](https://www.deepspeed.ai/docs/) | 文档 | ⭐⭐⭐ |
| [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) | 教程 | ⭐⭐⭐ |
| Megatron-LM 论文 | 论文 | ⭐⭐ |
| ZeRO 原论文 (Rajbhandari et al. 2020) | 论文 | ⭐⭐⭐ |
| Hugging Face Accelerate 文档 | 文档 | ⭐⭐ |

---

## 💡 本周学习建议

1. **先理解原理再跑代码**：分布式训练坑多，不理解原理容易走弯路
2. **关注显存数字**：每个技术都对照"显存占用减少了多少"来理解
3. **从单机多卡开始**：先搞定 DDP，再考虑多机多卡
4. **记录报错**：分布式训练的报错信息往往很难看懂，记录下来反复学习

---

## 小结

Week 2 的核心是**理解分布式训练的三大维度**：
- **通信拓扑**：谁和谁交换数据？
- **显存切分**：模型、梯度、优化器状态如何分摊？
- **计算效率**：如何避免等待、减少气泡？

DeepSpeed 把这三个维度都集成到一个框架，学好它是迈向大规模训练的第一步。
