---
layout: default
title: "Week 8 Capstone · Week4-8 全面复习与面试模拟"
render_with_liquid: false
---

# Week 8 Capstone · Week4-8 全面复习与面试模拟

## 五周知识地图

```
Week 4: FlashAttention
  └─ IO 感知计算 → 分块 + Online Softmax → v2/v3 改进
  └─ 核心：SRAM vs HBM，IO 复杂度 O(N²) → O(N)

Week 5: 推理优化
  └─ KV Cache → PagedAttention（虚拟内存）
  └─ 连续批处理 → 投机解码 → Medusa
  └─ 核心：Memory-Bound vs Compute-Bound

Week 6: 分布式训练进阶
  └─ TP（节点内）→ PP（1F1B 调度）→ DP（ZeRO）
  └─ 序列并行 → MoE → 专家并行
  └─ 核心：通信/计算权衡，3D 并行配置

Week 7: 量化与压缩
  └─ 量化基础（均匀/NF4）→ SmoothQuant → GPTQ → AWQ
  └─ 工具链：bitsandbytes / AutoGPTQ / AutoAWQ / llama.cpp
  └─ 核心：精度-速度-内存三角权衡

Week 8: MLOps 与系统设计
  └─ 监控（MFU/loss/grad）→ 故障恢复 → 成本优化
  └─ LLM 推理架构 → 面试框架
  └─ 核心：工程可靠性与系统思维
```

## 综合面试模拟：60 分钟技术面试

### Part 1：概念验证（15 分钟）

**面试官**：请解释一下 LLM 推理中的 Memory-Bound 和 Compute-Bound 是什么意思，各自的优化思路是什么？

**参考回答**：

```
Memory-Bound（内存受限）：
  定义：计算所需时间 < 内存访问所需时间
  场景：解码阶段（每步只生成 1 个 token，batch=1）
        算术强度 = FLOPs/bytes = 2N/(2N+2KV) ≈ 1 FLOP/byte
        A100 算术强度阈值：312 TFLOPS / 2 TB/s ≈ 156 FLOP/byte
        解码的算术强度 << 156，严重 Memory-Bound
  
  优化思路：
  ① 减少内存访问量：量化（INT4 减少 4× 数据量）
  ② 增加数据复用：连续批处理（多请求共享权重读取）
  ③ 减少 KV Cache：GQA/MQA（减少 KV head 数）

Compute-Bound（计算受限）：
  定义：内存访问时间 < 计算所需时间
  场景：预填充阶段（处理长 prompt，大矩阵乘法）
        算术强度 = 2 × batch × seq × hidden / (读取数据量)
        当 batch×seq 足够大时 >> 156 FLOP/byte
  
  优化思路：
  ① 提升算力：INT8/FP8 Tensor Core（速度 2×）
  ② 提升并行度：TP 增加有效 GPU 数
  ③ 减少计算量：FlashAttention、稀疏 Attention
```

### Part 2：数学推导（15 分钟）

**题目**：推导 Transformer 前向传播的 FLOPs（以参数量 N 和 token 数 T 表示）。

```python
"""
GPT-2/3 风格的 Transformer
参数：
  d_model = h（隐层维度）
  heads = a（注意力头数）
  d_head = h/a
  d_ff = 4h（FFN 中间层）
  L 层

每一层的 FLOPs（前向传播）：

1. Self-Attention
   QKV 投影：[T, h] × [h, 3h] = T × 3h × 2h FLOPs = 6Th²
   注意力分数：[T, h] × [h, T]（per head）× a heads
     = T × T × h × 2 = 2T²h FLOPs
   Softmax：O(T²a)（忽略）
   AV 加权：[T, T] × [T, h/a] × a = 2T²h FLOPs
   输出投影：[T, h] × [h, h] = 2Th² FLOPs
   Attention 小计：8Th² + 4T²h ≈ 8Th²（T << h 时）

2. FFN（两层全连接）
   Up: [T, h] × [h, 4h] = 8Th² FLOPs
   Down: [T, 4h] × [4h, h] = 8Th² FLOPs
   FFN 小计：16Th²

每层 FLOPs ≈ 24Th²

L 层总 FLOPs ≈ 24LTh²

用参数量 N 表示：N ≈ 12Lh²（主要参数在 Attention 和 FFN）
→ FLOPs ≈ 24LTh² = 2T × 12Lh² = 2TN

前向总 FLOPs ≈ 2 × T × N
前向+反向 ≈ 6 × T × N（反向约为前向的 2 倍）

注意：这也是为什么说"训练 1B 模型处理 1T token 需要 6 × 10^21 FLOPs"
"""

def compute_training_flops(params_billion: float, tokens_billion: float) -> str:
    flops = 6 * params_billion * 1e9 * tokens_billion * 1e9
    return f"{flops:.2e} FLOPs"

print(compute_training_flops(7, 2000))    # LLaMA-2 7B, 2T tokens
print(compute_training_flops(70, 2000))   # LLaMA-2 70B, 2T tokens
print(compute_training_flops(175, 300))   # GPT-3, 300B tokens
```

### Part 3：编程题（15 分钟）

**题目**：实现一个简单的 Ring All-Reduce（数据并行梯度同步）。

```python
import torch
import torch.distributed as dist

def ring_allreduce(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ring All-Reduce：每个 rank 将梯度发送给下一个 rank
    
    算法步骤：
    Phase 1（Scatter-Reduce）：N-1 轮，每轮发送 1/N 的数据
      每个 rank 发送一块数据给下一个 rank，收到后与本地相加
    Phase 2（All-Gather）：N-1 轮，每轮发送 1/N 的数据
      将汇总后的结果广播给所有 rank
    
    通信量：2 × (N-1)/N × 数据大小
    （比朴素 All-Reduce 效率高）
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # 将 tensor 分成 world_size 块
    chunks = tensor.chunk(world_size)
    chunk_size = chunks[0].numel()
    
    # 工作缓冲区
    buffer = tensor.clone()
    recv_buffer = torch.zeros_like(chunks[0])
    
    # Phase 1：Scatter-Reduce（每轮 reduce 一块）
    for step in range(world_size - 1):
        send_chunk_idx = (rank - step) % world_size
        recv_chunk_idx = (rank - step - 1) % world_size
        
        send_chunk = buffer[send_chunk_idx * chunk_size: (send_chunk_idx+1) * chunk_size]
        
        # 点对点通信（环形）
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1) % world_size
        
        reqs = []
        reqs.append(dist.isend(send_chunk, dst=send_rank))
        reqs.append(dist.irecv(recv_buffer, src=recv_rank))
        
        for r in reqs:
            r.wait()
        
        # 累加
        buffer[recv_chunk_idx * chunk_size: (recv_chunk_idx+1) * chunk_size] += recv_buffer
    
    # Phase 2：All-Gather（每轮广播一块）
    for step in range(world_size - 1):
        send_chunk_idx = (rank - step + 1) % world_size
        recv_chunk_idx = (rank - step) % world_size
        
        send_chunk = buffer[send_chunk_idx * chunk_size: (send_chunk_idx+1) * chunk_size]
        
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1) % world_size
        
        reqs = []
        reqs.append(dist.isend(send_chunk, dst=send_rank))
        reqs.append(dist.irecv(recv_buffer, src=recv_rank))
        
        for r in reqs:
            r.wait()
        
        buffer[recv_chunk_idx * chunk_size: (recv_chunk_idx+1) * chunk_size] = recv_buffer.clone()
    
    # 除以 world_size（All-Reduce = sum，需要手动平均）
    buffer /= world_size
    tensor.copy_(buffer)
    
    return tensor
```

### Part 4：系统设计（15 分钟）

**题目**：为一家刚上线的 AI 公司设计 LLM 训练基础设施，预算 $1M/年。

```
关键决策：

1. 硬件选择：
   - 自建 vs 云：初期 → 云（灵活性 > 成本）
   - 推荐：AWS p4d.24xlarge（8× A100）
     按需：$32.77/h；Spot：~$13/h（60% 折扣）
   - 1M/年 Spot → ~77,000 GPU 小时 → 约 10 个节点运行 1 年

2. 训练配置（70B 模型目标）：
   - 并行：TP=8（节点内）, PP=4, DP=2 → 64 GPU
   - 成本：64 GPU × $1.6/GPU/h（Spot）= $102/h
   - 1M/年 → 9,800 小时有效训练 → 约 14 个月（需优化！）
   
   优化：提升 MFU 30%→45% → 9,800h × (45/30) = 等效 14,700h
   → 训练时间缩短 33%，成本不变

3. 基础设施：
   - 代码：GitHub + DVC（数据版本）
   - 实验：W&B（免费版）
   - 存储：S3（模型/数据）+ EFS（训练热数据）
   - 监控：Prometheus + Grafana（自建，免费）
   - Checkpoint：S3 自动版本化

4. 成本分配（$1M/年）：
   - GPU 计算：$700K (70%)
   - 存储（S3/EFS）：$100K (10%)
   - 网络传输：$50K (5%)
   - 工具/SaaS：$50K (5%)
   - 预备金：$100K (10%)
```

## 本系列总结

恭喜你完成了 AI Infra 学习线 Week4-8！

| 周次 | 主题 | 核心技能 |
|------|------|---------|
| Week 4 | FlashAttention | IO 感知算法设计、CUDA kernel 理解 |
| Week 5 | 推理优化 | LLM serving 系统架构 |
| Week 6 | 分布式训练 | 大规模并行配置与调优 |
| Week 7 | 量化压缩 | 模型压缩工程实践 |
| Week 8 | MLOps | 工程可靠性与系统设计 |

**你现在具备了**：
- 解释 FlashAttention、PagedAttention、GPTQ 等核心算法的能力
- 设计千卡训练系统的思路
- 量化 LLM 并部署到生产环境的技能
- 通过 AI Infra / ML Systems 面试的知识储备

**下一步**：
- 实际运行 Week4-7 的代码示例
- 在 A100/V100 上测试量化效果
- 参与开源项目（vLLM、FlashAttention、Megatron-LM）
- 阅读原始论文（见各周 D1 文章的参考链接）
