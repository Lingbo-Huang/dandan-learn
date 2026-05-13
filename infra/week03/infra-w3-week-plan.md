---
layout: default
title: "Week 3 周规划 · cuBLAS / cuDNN 深度解析"
---

# AI Infra Week 3：cuBLAS / cuDNN 深度解析

> **系列**：AI Infra 线  
> **周次**：Week 3  
> **前置**：Week 1 GPU 架构，Week 2 CUDA 基础编程

---

## 本周主题

从 CUDA 基础到深度学习专用库。cuBLAS 和 cuDNN 是所有深度学习框架（PyTorch/TensorFlow）底层的核心加速库，理解它们才能真正做性能优化。

| 天 | 主题 |
|----|------|
| D1 | cuBLAS 架构与 GEMM 深度解析 |
| D2 | cuBLAS API 与性能调优 |
| D3 | cuDNN 卷积算法选择 |
| D4 | cuDNN Workspace 管理与算子融合 |
| D5 | NHWC vs NCHW：内存布局的性能影响 |
| D6 | 综合调优：Profile → 分析 → 优化 |
| D7 | 综合实战：手写 cuBLAS GEMM 并验证 |
