---
layout: default
title: "Week 8 周规划：MLOps 与系统设计"
---

# Week 8 周规划：MLOps 与系统设计

## 本周目标

掌握 AI Infra 的工程实践能力：从训练监控到故障恢复，从成本优化到系统设计面试题解答。这是求职 AI Infra 岗位的最后一块拼图——工程素质与系统思维。

## 为什么 MLOps 同样重要？

优秀的 AI Infra 工程师不只是能写 CUDA kernel，还要能：
- 监控千卡训练集群的健康状态，及时发现异常
- 设计容错机制，让一张卡挂掉不影响整体训练进度
- 计算 GPU 成本，提出优化方案节省预算
- 在系统设计面试中，清晰地设计 LLM 服务架构

## 本周文章安排

| 天 | 文件 | 主题 |
|---|------|------|
| D1 | infra-w8-training-monitoring.md | 训练监控：指标、告警与异常检测 |
| D2 | infra-w8-fault-tolerance.md | 故障恢复：Checkpoint 策略与自动重启 |
| D3 | infra-w8-cost-optimization.md | 成本优化：Spot 实例与资源调度 |
| D4 | infra-w8-system-design-llm.md | 系统设计：LLM 推理服务架构 |
| D5 | infra-w8-interview-questions.md | 面试精讲：高频题目与解题框架 |
| D6 | infra-w8-capstone.md | Capstone：Week4-8 全面复习与面试模拟 |

## 本周面试题预告

1. 如何监控分布式 LLM 训练？关键指标有哪些？
2. 千卡训练时一张 GPU 挂掉，如何设计恢复方案？
3. 如何评估 GPU 训练的成本效益？MFU 是什么？
4. 设计一个能处理 100 QPS 的 LLM 推理服务（系统设计题）
5. 如何在不降低吞吐的情况下降低 P99 延迟？
