---
layout: default
title: "D5 · 面试精讲：高频题目与解题框架"
render_with_liquid: false
---

# D5 · 面试精讲：AI Infra 高频面试题与解题框架

## 面试题分类

AI Infra 岗位面试题通常分三类：
1. **原理深挖**：考察你是否真正理解底层（而非只会调 API）
2. **系统设计**：考察你是否能独立设计大规模系统
3. **工程经验**：考察你是否踩过坑、解决过实际问题

## 类别一：原理深挖题

### Q1: FlashAttention 为什么快？

**解题框架（3 层回答）**：

```
Layer 1（概念层）：
  标准 Attention 的问题是内存受限（Memory-Bound），
  需要将 Q/K/V 和中间的 S=QK^T 矩阵反复读写 HBM，
  IO 复杂度 O(N²) 成为瓶颈。

Layer 2（技术层）：
  FlashAttention 的解决方案：
  ① Tiling：将计算分块，每块完全在 SRAM 中完成
  ② Online Softmax：不需要先算完整的注意力分数再 softmax
  ③ 结果：IO 复杂度降至 O(N²/B)，B 是 block size
  
  IO 减少的来源：
  - 不再写 S（N×N 矩阵）到 HBM
  - 不再写 P（softmax 后的注意力权重）到 HBM
  - HBM 访问量：O(N·d) 而非 O(N²)

Layer 3（数据层）：
  A100 HBM 带宽 2 TB/s，SRAM 19 TB/s（10× 差距）
  FlashAttention 将 HBM 访问减少 5-20×
  实际加速：2-4×（序列越长，加速越明显）
  内存从 O(N²) 降至 O(N)（梯度 checkpointing 协作）
```

### Q2: PagedAttention 解决了什么问题？

```
Problem：
  KV Cache 大小在请求开始时未知，传统实现预分配最大长度
  → 内存碎片化：预分配但未使用的内存浪费（高达 60-80%）
  → 限制了并发请求数

Solution（类比操作系统虚拟内存）：
  将 KV Cache 组织为固定大小的"物理块"（如 16 tokens/块）
  逻辑块表（类似页表）映射到物理块
  
  优势：
  ① 按需分配：只在生成时分配实际需要的块
  ② 块共享：共同前缀可以共享物理块（prefix sharing）
  ③ 碎片几乎为零：只有块内碎片（最后一块可能不满）

Result：
  vLLM 实测：吞吐量比 FasterTransformer 高 2-4×
  并发请求数可提升 3-5×
```

### Q3: 为什么 TP 通信要放在节点内（NVLink）而不是跨节点？

```python
# 计算说明
tp_comm_per_layer = 2  # All-Reduce: 前向1次 + 反向1次
tp_data_size_per_layer = seq_len * batch * hidden * 2  # FP16

# 序列长度=2048，batch=1，hidden=4096，FP16
data_mb = 2048 * 1 * 4096 * 2 / 1e6  # 16 MB per All-Reduce

# NVLink 带宽：600 GB/s（节点内）
nvlink_time_ms = 16 / (600 * 1024) * 1000  # ~0.026ms

# InfiniBand HDR：25 GB/s（跨节点）
ib_time_ms = 16 / (25 * 1024) * 1000  # ~0.625ms

speedup = ib_time_ms / nvlink_time_ms  # 24×

# 结论：TP 通信密度高（每层 2 次 All-Reduce），
# 必须用 NVLink（节点内）才不会被通信拖垮
# PP 通信稀疏（只在层边界），适合跨节点 InfiniBand
```

## 类别二：系统设计题框架

### SCALE 框架（系统设计解题模板）

```
S - Scope（明确范围）
  "我理解这道题是要设计...，我先做几个假设：
   1. 规模：xxx QPS，xxx 并发
   2. 延迟要求：P99 < xxx ms
   3. 一致性/可用性权衡：...
   我来确认这些假设是否正确。"

C - Capacity（容量规划）
  "先做容量规划：
   - 存储：xxx GB/TB
   - 计算：xxx FLOPS，需要 xxx GPU
   - 带宽：xxx Gbps"

A - Architecture（架构设计）
  "整体架构分为 X 层：
   - 接入层：...
   - 计算层：...
   - 存储层：...
   画出组件图"

L - Latency/Load（延迟与负载）
  "关键路径的延迟分析：
   - 瓶颈在哪里？
   - 如何优化？"

E - Edge Cases（边界情况）
  "需要考虑的特殊情况：
   - 节点故障如何恢复？
   - 流量突增如何应对？
   - 安全性？"
```

### 系统设计题示例：设计训练数据 Pipeline

```
Q: 设计一个为 1T token LLM 训练提供数据的 Pipeline，
   要求不能成为训练的瓶颈。

关键数字：
- 训练吞吐：1M tokens/s（目标）
- 每个样本：2048 tokens × 2 bytes = 4 KB
- 需要数据速率：1M × 4 KB = 4 GB/s

架构：
1. 数据存储：NFS/S3（原始数据）→ 本地 SSD（预处理后）
   瓶颈分析：NFS ~1 GB/s，需要 4 GB/s → 需要 SSD 缓存

2. 预处理：
   - Tokenization（离线完成，保存 .bin 格式）
   - Shuffle（分桶 shuffle，避免全局排序）
   - Prefetch buffer：GPU 消耗时，CPU 预取下一批

3. DataLoader 优化：
   - num_workers=8（多进程加载）
   - pin_memory=True（直接到 GPU 页锁定内存）
   - prefetch_factor=4（提前加载 4 批）
   
4. 数据并行时的分片：
   - 每个 DP rank 读取不同分片
   - 确保不重复，不遗漏

代码：
```

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mmap

class TokenizedDataset(Dataset):
    """
    高效的 Token 数据集（内存映射，避免全量加载）
    """
    def __init__(self, data_path: str, seq_len: int, rank: int, world_size: int):
        # 使用 mmap 避免将整个文件加载到内存
        self.file = open(data_path, "rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # 每个 token 2 bytes (uint16)
        total_tokens = len(self.data) // 2
        self.seq_len = seq_len
        
        # DP 分片：每个 rank 处理不同数据段
        tokens_per_rank = total_tokens // world_size
        self.start = rank * tokens_per_rank
        self.end = (rank + 1) * tokens_per_rank
        
        self.num_samples = (self.end - self.start - 1) // seq_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = (self.start + idx * self.seq_len) * 2  # bytes
        end = start + (self.seq_len + 1) * 2
        
        chunk = np.frombuffer(self.data[start:end], dtype=np.uint16).copy()
        x = torch.from_numpy(chunk[:-1]).long()
        y = torch.from_numpy(chunk[1:]).long()
        return x, y

# 高性能 DataLoader 配置
dataset = TokenizedDataset(
    data_path="/data/train.bin",
    seq_len=2048,
    rank=dist.get_rank(),
    world_size=dist.get_world_size()
)

loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=8,           # 多进程 IO
    pin_memory=True,         # 锁页内存，加速 Host→GPU 拷贝
    prefetch_factor=4,       # 提前预取 4 批
    persistent_workers=True, # 避免重复初始化 worker
    shuffle=True,
)
```

## 类别三：工程经验题

### 常见问题与答案

**Q: 你遇到过最难的分布式训练 bug 是什么？**

参考框架：
```
STAR 法则：Situation（背景）→ Task（任务）→ Action（行动）→ Result（结果）

示例回答：
S: 我们在 64 节点 512 GPU 上训练一个 30B 模型，
   训练在第 2000 步左右出现 Loss NaN，重启后仍然复现。

T: 需要找到根本原因，而不只是跳过这个 checkpoint。

A: 
  1. 首先用 torch.autograd.detect_anomaly() 定位到是 LayerNorm 后面的层
  2. 对比单机和多机的中间激活值，发现在某个 PP Stage 边界数值不一致
  3. 最终发现：PP 通信用的是 BF16，但某些硬件上 BF16 的 Inf 行为不一致
     具体：一个 NaN 在 BF16 传输后变成了一个很大的正数，绕过了 NaN 检查
  4. 解法：在 PP 边界通信时加了 NaN 检查 hook

R: 训练稳定了，MFU 提升 1%（因为找到了顺带的性能问题）

技术细节要能深问：
- 为什么 BF16 Inf 行为不一致？（不同 GPU 固件版本）
- PP 边界 hook 怎么实现的？
- 为什么是 LayerNorm 后面？（梯度消失保护失效）
```

## 高频题目清单

| 类别 | 题目 | 考察点 |
|------|------|--------|
| Attention | FlashAttention IO 复杂度推导 | 内存层次理解 |
| Attention | GQA vs MHA 内存对比 | KV Cache 优化 |
| 分布式 | TP/PP/DP 的选择原则 | 并行策略权衡 |
| 分布式 | ZeRO-1/2/3 的区别 | 内存优化 |
| 量化 | GPTQ vs AWQ 的核心差异 | 量化算法理解 |
| 量化 | INT4 推理为什么快 | 硬件架构 |
| 推理 | vLLM vs TensorRT-LLM | 推理框架选型 |
| 推理 | 如何降低 P99 延迟 | 系统设计 |
| 系统设计 | 设计 LLM 推理服务 | 全栈设计能力 |
| 系统设计 | 训练数据 Pipeline 设计 | IO 优化 |
| MLOps | Loss NaN 排查流程 | 工程素质 |
| MLOps | 千卡训练故障恢复 | 可靠性设计 |

## 面试技巧

1. **先确认假设**：系统设计题先问清楚规模、SLA，再开始设计
2. **数字要量化**：说"快"不如说"快 2.4×（从 X 到 Y）"
3. **权衡要明确**：每个设计决策都有代价，主动说出权衡
4. **从简单到复杂**：先给出朴素方案，再逐步优化
5. **承认不知道**：比瞎说更好，但要说"我的推测是..."
