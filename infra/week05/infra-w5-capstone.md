---
layout: default
title: "Week 5 Capstone · 设计一个生产级 LLM 推理服务"
render_with_liquid: false
---

# Week 5 Capstone · 设计一个生产级 LLM 推理服务

## 系统设计题目

**题目**：为一家 AI 公司设计 LLM 推理服务，要求：
- 模型：LLaMA-2 70B
- QPS：100 req/s
- 延迟 SLA：P99 TTFT < 2s，P99 TBT < 100ms
- 平均输出长度：500 tokens
- 预算：尽量少的 GPU

## 第一步：规模估算

### 计算量估算

```python
def estimate_llm_inference(
    model_params_B: float,    # 模型参数量（十亿）
    batch_size: int,
    seq_len: int,
    output_len: int,
    gpu_flops_t: float,       # GPU TFLOPS
    gpu_memory_bw_tbs: float  # GPU 内存带宽（TB/s）
):
    """
    LLM 推理成本估算
    """
    # 模型权重大小（FP16）
    weights_gb = model_params_B * 2  # 2 bytes/param
    print(f"模型权重: {weights_gb} GB")
    
    # Prefill FLOPs（近似：2 × params × input_tokens）
    prefill_flops = 2 * model_params_B * 1e9 * seq_len
    
    # Decode FLOPs per step（近似：2 × params × 1）
    decode_flops_per_step = 2 * model_params_B * 1e9
    
    # 内存带宽需求（每次 Decode 需读一次权重）
    decode_mem_needed_gb = weights_gb
    
    # Prefill 时间（Compute-Bound）
    prefill_time_ms = prefill_flops / (gpu_flops_t * 1e12) * 1000 * batch_size
    
    # Decode 时间（Memory-Bound：读一次权重）
    decode_time_per_step_ms = decode_mem_needed_gb / (gpu_memory_bw_tbs * 1e3) * 1000
    
    # Decode 总时间
    decode_total_ms = decode_time_per_step_ms * output_len
    
    print(f"Prefill 时间（batch={batch_size}, seq={seq_len}）: {prefill_time_ms:.0f}ms")
    print(f"Decode 每步时间: {decode_time_per_step_ms:.1f}ms")
    print(f"Decode 总时间（{output_len} tokens）: {decode_total_ms:.0f}ms")
    print(f"总推理时间: {prefill_time_ms + decode_total_ms:.0f}ms")

# LLaMA-2 70B on A100 80GB (4× A100 tensor parallel)
# 每块 A100: 312 TFLOPS FP16, 2 TB/s
# 4× A100: ~1248 TFLOPS, ~8 TB/s（tensor parallel）
estimate_llm_inference(
    model_params_B=70,
    batch_size=1,
    seq_len=1024,
    output_len=500,
    gpu_flops_t=1248,
    gpu_memory_bw_tbs=8
)
# 输出（近似）：
# 模型权重: 140 GB
# Prefill 时间（batch=1, seq=1024）: 115ms
# Decode 每步时间: 17.5ms
# Decode 总时间（500 tokens）: 8750ms  ← 超过 SLA！
```

**问题**：单步 Decode 17.5ms，500 步共需 8.75s，远超 P99 TBT 100ms 的 SLA。

**解决方案**：增加 batch size（每步处理更多请求，分摊内存带宽）。

### Batch Size 对 Decode 的影响

```python
# Decode 在 Memory-Bound 时，batch_size 增加不增加时间！
# 因为瓶颈是权重读取（不变），不是计算量

for bs in [1, 4, 8, 16, 32]:
    # 每步 Decode 时间：只取决于权重读取时间
    time_per_step = 140 / 8 * 1000  # 17.5ms（不随 bs 变化）
    throughput = bs / (time_per_step / 1000)  # tokens/s per step
    tps = throughput  # 每秒生成 token 数 = bs / step_time
    print(f"BS={bs:3d}: 每步{time_per_step:.1f}ms, 吞吐={tps:.0f} tokens/s")

# 输出：
# BS=  1: 每步17.5ms, 吞吐=57 tokens/s
# BS=  4: 每步17.5ms, 吞吐=228 tokens/s  ← 线性提升！
# BS=  8: 每步17.5ms, 吞吐=457 tokens/s
# BS= 16: 每步17.5ms, 吞吐=914 tokens/s
# BS= 32: 每步17.5ms, 吞吐=1828 tokens/s
```

但 TBT 要求 < 100ms，所以每步 17.5ms 是满足的（单步），需要关注的是 **每步 Decode 时间本身**，不是总时间。

## 第二步：GPU 资源规划

### 存放 LLaMA-2 70B 需要多少卡

模型权重：70B × 2 bytes = 140 GB

选项：
1. **4× A100 80GB**：Tensor Parallel，每卡 35GB 权重，剩余 45GB 用于 KV Cache
2. **8× A100 40GB**：每卡 17.5GB 权重，剩余 22.5GB 用于 KV Cache
3. **2× H100 80GB**：配合 NVLink 高带宽，每卡 70GB

选择：**4× A100 80GB**（成本效率最高）

### KV Cache 容量规划

LLaMA-2 70B：80 层, 8 KV heads（GQA）, head_dim=128

```python
# 每个 token 的 KV Cache
kv_per_token_per_req = 2 * 80 * 8 * 128 * 2  # bytes
# = 2 * 80 * 8 * 128 * 2 = 327,680 bytes ≈ 320 KB

# 4× A100 可用 KV Cache 空间（4× 45GB = 180GB）
available_gb = 4 * (80 - 35)  # 每卡去掉权重
# = 180 GB

# 可支持的最大并发 token 数
max_tokens = 180 * 1024**3 / kv_per_token_per_req
print(f"最大并发 token 数: {max_tokens:,.0f}")
# ≈ 573,440 tokens

# 假设每请求最大 2000 tokens（1000 输入 + 1000 输出）
max_concurrent_reqs = max_tokens // 2000
print(f"最大并发请求: {max_concurrent_reqs}")
# ≈ 286 个并发请求
```

### 吞吐量计算

```python
# 目标：100 req/s，每请求 500 tokens 输出
# 需要的总 TPS = 100 × 500 = 50,000 tokens/s

# 4× A100 + Tensor Parallel 的 Decode 吞吐量
# 每步时间：140GB / (8 TB/s) = 17.5ms
# 每步吞吐：BS / 0.0175s = BS × 57 req/s

# 需要 BS = 50,000 / 57 ≈ 877
# 但最大并发 286，还不够！

# 解决方案：增加 GPU 节点（Scale Out）
# 3个节点 × 4× A100 = 12× A100
# 最大并发：3 × 286 = 858 ≈ 877 ✓
```

## 第三步：完整系统架构

```
                    Load Balancer
                        │
           ┌────────────┼────────────┐
           │            │            │
     Node 1 (4×A100) Node 2(4×A100) Node 3(4×A100)
     LLaMA-2 70B      LLaMA-2 70B   LLaMA-2 70B
     (TP=4)           (TP=4)        (TP=4)
           │            │            │
     vLLM Engine    vLLM Engine   vLLM Engine
           │
     ┌─────┴──────┐
     │            │
  Prefill      Decode
 Scheduler   Scheduler
     │
 KV Cache Manager
 (PagedAttention)
```

### 关键配置

```python
# vLLM 生产配置（每个节点）
from vllm import AsyncLLMEngine, AsyncEngineArgs

engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-2-70b-chat-hf",
    tensor_parallel_size=4,          # 4× A100
    gpu_memory_utilization=0.93,
    max_num_seqs=300,                # 每节点最大并发
    max_num_batched_tokens=16384,    # 每步最多处理的 token 数
    enable_chunked_prefill=True,     # 避免 Prefill 阻塞 Decode
    max_num_prefill_seqs=2,          # 每步最多 2 个新 Prefill
    enable_prefix_caching=True,      # 系统 Prompt 共享
    block_size=16,
    dtype="float16",
    # 投机解码（可选，需要部署 7B Draft 模型）
    # speculative_model="meta-llama/Llama-2-7b-chat-hf",
    # num_speculative_tokens=5,
)
```

## 第四步：监控与告警

```python
# 关键监控指标
metrics = {
    # 延迟指标
    "p50_ttft_ms": "P50 首 token 延迟",
    "p99_ttft_ms": "P99 首 token 延迟（SLA: < 2000ms）",
    "p50_tbt_ms":  "P50 token 间延迟",
    "p99_tbt_ms":  "P99 token 间延迟（SLA: < 100ms）",
    
    # 吞吐量指标
    "rps":         "请求/秒",
    "tps":         "tokens/秒（输出）",
    "queue_len":   "等待队列长度",
    
    # 资源指标
    "gpu_util":         "GPU 利用率（目标 > 85%）",
    "kv_cache_util":    "KV Cache 利用率（目标 > 80%）",
    "kv_cache_preempt": "KV Cache 抢占次数（越少越好）",
    
    # 模型指标
    "tokens_per_req":   "每请求 token 数分布",
    "acceptance_rate":  "投机解码接受率",
}

# 告警规则
alerts = [
    "p99_ttft_ms > 1800ms → 扩容警告",
    "p99_tbt_ms > 80ms → 立即告警",
    "queue_len > 500 → 扩容触发",
    "kv_cache_util > 95% → OOM 风险",
]
```

## 本周总结

| 技术 | 解决的问题 | 效果 |
|------|----------|------|
| KV Cache | 避免重复计算 | O(T²) → O(T) |
| PagedAttention | 内存碎片化 | 利用率 24% → 96% |
| 连续批处理 | 静态 batch 低效 | 吞吐 4× |
| 投机解码 | Decode Memory-Bound | 速度 2-3× |
| Medusa | 无需独立 Draft 模型 | 速度 2-2.5× |

**下周预告：分布式训练进阶——Megatron-LM / 序列并行 / MoE**
