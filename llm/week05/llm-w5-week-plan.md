---
layout: default
title: "Week 5 周规划 · SFT 与对齐"
render_with_liquid: false
---

# 大模型线 Week 5 周规划总览

**主题：SFT 与对齐——指令微调、LoRA、RLHF、DPO**  
**周期：Day 1 - Day 6**

---

## 本周目标

掌握让预训练模型"听话"的全套技术栈，能够：

- 理解 SFT 指令微调的数据格式和训练细节
- 手写 LoRA 和 QLoRA，理解参数高效微调原理
- 推导 RLHF 完整流程（SFT → RM → PPO）
- 理解 DPO 如何绕过 RL，直接从偏好数据训练
- 在实际场景中选择合适的对齐方法

---

## 每日主题速览

| Day | 主题 | 关键词 |
|-----|------|--------|
| D1 | 指令微调 (SFT) | 数据格式、Chat Template、损失掩码 |
| D2 | LoRA / QLoRA | 低秩分解、4bit 量化、PEFT |
| D3 | RLHF | 奖励模型、PPO、KL 散度约束 |
| D4 | DPO | Bradley-Terry、直接偏好优化 |
| D5 | 对齐综合实战 | LLaMA-Factory、全流程实践 |
| D6 | Capstone | 端到端对齐一个小模型 |

---

## 面试高频题

1. LoRA 的低秩分解为什么有效？秩 r 如何选择？
2. RLHF 的 KL 散度惩罚项有什么作用？
3. DPO 和 RLHF 的本质区别？
4. 如何构建高质量的 SFT 数据集？
5. QLoRA 的 4bit 量化如何不损失精度？

---

## 参考资料

- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
