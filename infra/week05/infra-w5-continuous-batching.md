---
layout: default
title: "D3 · 连续批处理：从静态到动态的革命"
render_with_liquid: false
---

# D3 · 连续批处理：从静态到动态的革命

## 静态批处理的问题

传统 LLM 推理使用**静态批处理（Static Batching）**：

```
批次开始：[Req A (prompt 100 tokens), Req B (prompt 50 tokens), Req C (prompt 200 tokens)]
↓
所有请求一起做 Prefill
↓
循环解码（Decode）直到所有请求结束
↓
批次结束：等待所有请求都生成完毕才能开始下一批

时间线：
[Prefill][Decode][Decode][Decode]...[Decode(A结束)][Decode(等B)][Decode(等B)]...[Decode(C结束)]
                                         ↑ A早结束，但必须等B,C结束才能释放资源，新请求无法加入
```

**核心问题**：
1. **GPU 浪费**：早结束的序列对应的 GPU 计算是浪费的（padding）
2. **延迟增大**：新请求必须等待当前批次完全结束才能处理
3. **吞吐量低**：批次大小由最长序列决定，无法动态调整

## 连续批处理（Continuous Batching）

Orca 论文（2022）提出的解决方案：**在每一个 Decode 步骤后，检查是否有请求完成，立即替换新请求**。

```
连续批处理时间线：
Step 0: [Req A(prefill)] [Req B(prefill)] [Req C(prefill)]
Step 1: [Req A(decode)]  [Req B(decode)]  [Req C(decode)]
Step 2: [Req A(decode)]  [Req B完成→Req D加入(prefill)] [Req C(decode)]
Step 3: [Req A(decode)]  [Req D(decode)]  [Req C完成→Req E加入(prefill)]
...

关键：每步后立即填补空缺，GPU 利用率接近 100%！
```

## Prefill 和 Decode 的混合调度

连续批处理的一个挑战：**Prefill 和 Decode 的计算特征完全不同**。

```python
"""
理解 Prefill vs Decode 的性能特征
"""

# Prefill 阶段（处理用户输入）
# 特点：所有 token 并行计算 Q @ K^T，是 Compute-Bound
# 效率高，但会阻塞解码中的其他请求

# Decode 阶段（逐步生成）
# 特点：每次只有 1 个新 token，做 q(1×d) @ K_cache(N×d)^T
# Memory-Bound（读 KV Cache 的带宽是瓶颈）

def estimate_bottleneck(model_config, seq_len, batch_size=1):
    """估算当前操作是 compute-bound 还是memory-bound"""
    d_model = model_config['d_model']
    num_layers = model_config['num_layers']
    
    # Decode: 线性层计算量 vs 内存带宽
    flops_per_token = 2 * d_model * d_model * 4 * num_layers  # Q,K,V,O proj × 2(MAC)
    params_bytes = d_model * d_model * 4 * num_layers * 2  # FP16
    
    # A100: 312 TFLOPS FP16, 2 TB/s HBM
    compute_time_ms = flops_per_token * batch_size / 312e12 * 1000
    memory_time_ms = params_bytes / 2e12 * 1000  # 不随 batch_size 变化（单次访问）
    
    bottleneck = "Memory-Bound" if memory_time_ms > compute_time_ms else "Compute-Bound"
    
    print(f"Decode (batch={batch_size}): Compute={compute_time_ms:.2f}ms, "
          f"Memory={memory_time_ms:.2f}ms → {bottleneck}")
    
    # Prefill: 处理 seq_len 个 token
    flops_prefill = 2 * seq_len * d_model * d_model * 4 * num_layers + \
                    4 * seq_len * seq_len * d_model * num_layers  # attention
    compute_time_prefill = flops_prefill / 312e12 * 1000
    memory_time_prefill = params_bytes / 2e12 * 1000  # 同样的权重读取
    
    bottleneck_prefill = "Memory-Bound" if memory_time_prefill > compute_time_prefill else "Compute-Bound"
    print(f"Prefill (seq_len={seq_len}): Compute={compute_time_prefill:.2f}ms, "
          f"Memory={memory_time_prefill:.2f}ms → {bottleneck_prefill}")

llama7b = {'d_model': 4096, 'num_layers': 32}
estimate_bottleneck(llama7b, seq_len=1024, batch_size=1)
# 输出：
# Decode (batch=1): Compute=0.04ms, Memory=0.52ms → Memory-Bound
# Prefill (seq_len=1024): Compute=5.50ms, Memory=0.52ms → Compute-Bound
```

## vLLM 的调度策略

vLLM 实现了一个精细的调度器：

```python
"""
vLLM 调度器核心逻辑（简化版）
"""
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

class SequenceStatus(Enum):
    WAITING = "waiting"        # 等待 Prefill
    RUNNING = "running"        # 正在 Decode
    FINISHED = "finished"      # 已完成
    SWAPPED = "swapped"        # 被换出到 CPU（内存不足时）

@dataclass
class Sequence:
    seq_id: int
    prompt_tokens: List[int]
    output_tokens: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    
    @property
    def num_tokens(self):
        return len(self.prompt_tokens) + len(self.output_tokens)

class Scheduler:
    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        
        self.waiting: List[Sequence] = []   # 等待 Prefill 的队列
        self.running: List[Sequence] = []   # 正在 Decode 的队列
        self.swapped: List[Sequence] = []   # 换出到 CPU 的队列
    
    def schedule(self) -> dict:
        """
        每个 Decode Step 调用一次，决定这一步处理哪些序列
        返回调度结果：
          - prefills: 这一步要做 Prefill 的序列
          - decodes:  这一步要做 Decode 的序列
        """
        scheduled_prefills = []
        scheduled_decodes = []
        
        # 1. 已有 running 序列继续 decode（优先级高）
        num_decode_tokens = len(self.running)  # 每个序列贡献 1 个 token
        
        # 2. 从 waiting 队列中选择能加入的序列
        token_budget = self.max_num_batched_tokens - num_decode_tokens
        
        for seq in list(self.waiting):
            # 检查是否有足够的 token 预算
            if len(seq.prompt_tokens) > token_budget:
                break
            # 检查是否有足够的 KV Cache 空间
            if len(self.running) + len(scheduled_prefills) >= self.max_num_seqs:
                break
            
            scheduled_prefills.append(seq)
            token_budget -= len(seq.prompt_tokens)
            self.waiting.remove(seq)
        
        scheduled_decodes = list(self.running)
        self.running.extend(scheduled_prefills)
        
        return {
            "prefills": scheduled_prefills,
            "decodes": scheduled_decodes,
        }
    
    def update_after_step(self, finished_seq_ids: List[int]):
        """更新调度状态：移除完成的序列"""
        self.running = [s for s in self.running if s.seq_id not in finished_seq_ids]
        for seq_id in finished_seq_ids:
            # 释放 KV Cache
            pass  # 交给 KVCacheManager 处理
```

## Chunked Prefill（分块预填充）

问题：大型 Prefill 会"阻塞"正在解码的其他请求，导致延迟抖动。

解决：将 Prefill 分块执行，与 Decode 混合。

```python
# 传统调度：Prefill 和 Decode 分开
# Step 1: [Prefill(A, 1000 tokens)]   # 耗时 ~50ms，阻塞所有解码请求
# Step 2: [Decode(A), Decode(B), ...]

# Chunked Prefill：
# Step 1: [Prefill(A, chunk1=256 tokens), Decode(B), Decode(C)]  # ~5ms
# Step 2: [Prefill(A, chunk2=256 tokens), Decode(B), Decode(C)]  # ~5ms
# ...
# Step 4: [Prefill(A, chunk4=256 tokens), Decode(B), Decode(C)]  # ~5ms
# Step 5: [Decode(A), Decode(B), Decode(C)]

# 好处：
# 1. 解码请求不被 Prefill 阻塞（TBT 稳定）
# 2. Prefill 的 GPU 利用率更高（与 Decode 的 Memory-Bound 互补）
```

## 性能数据

**vLLM vs 传统推理服务（A100 80GB, LLaMA-2 13B）：**

| 指标 | 传统 (TGI) | vLLM | 提升 |
|------|-----------|------|------|
| 吞吐量（req/s） | 0.5 | 2.2 | **4.4×** |
| P50 TTFT | 120ms | 85ms | 1.4× |
| P99 TTFT | 850ms | 320ms | 2.7× |
| 最大并发 | 20 | 120 | **6×** |
| 内存利用率 | 24% | 92% | 3.8× |

## 实战配置

```python
# vLLM 生产配置示例
from vllm import AsyncLLMEngine, AsyncEngineArgs

engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-2-13b-chat-hf",
    
    # 内存管理
    gpu_memory_utilization=0.92,   # 留 8% 给 CUDA 临时内存
    block_size=16,                  # KV Cache 块大小（tokens）
    max_num_seqs=256,              # 最大并发序列
    
    # 批处理控制
    max_num_batched_tokens=8192,   # 每步最多处理的 token 数
    enable_chunked_prefill=True,   # 启用分块 Prefill
    max_num_prefill_seqs=1,        # 每步最多新增 Prefill 序列（避免阻塞）
    
    # 前缀缓存
    enable_prefix_caching=True,    # 启用 Prefix Sharing
    
    # 性能调优
    tensor_parallel_size=1,        # 单卡
    dtype="float16",
)

engine = AsyncLLMEngine.from_engine_args(engine_args)
```

## 面试题

**Q: 连续批处理如何提升 GPU 利用率？为什么传统静态批处理效率低？**

A: 传统静态批处理要等批次中所有请求都完成才能开始下一批，早完成的请求的计算资源被浪费（GPU 在等其他序列）。连续批处理（Orca 提出，vLLM 实现）在每个 Decode Step 之后立即替换完成的序列，使 GPU 始终保持满负荷运行。结合 PagedAttention 的灵活内存管理，vLLM 能同时调度 100+ 个并发请求，GPU 利用率从 ~24% 提升到 ~90%，吞吐量提升 3-4×。
