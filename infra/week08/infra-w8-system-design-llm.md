---
layout: default
title: "D4 · 系统设计：LLM 推理服务架构"
render_with_liquid: false
---

# D4 · 系统设计：LLM 推理服务架构

## 面试题：设计一个支持 100 QPS 的 LLM 推理服务

**约束**：
- 模型：LLaMA-2 70B
- 目标：100 QPS，P99 首 token 延迟 < 2s，P99 生成延迟 < 30s
- 硬件预算：尽量最小化
- 请求：平均 500 input tokens，平均 500 output tokens

## 第一步：容量规划

```python
def capacity_planning():
    """
    LLM 推理容量规划
    """
    # 请求特征
    qps = 100
    avg_input_tokens = 500
    avg_output_tokens = 500
    
    # Prefill 吞吐（A100 80G，FP16，LLaMA-2 70B）
    # 参考：vLLM 实测约 10,000 tokens/s prefill（单 GPU）
    prefill_throughput_per_gpu = 10000  # tokens/s
    
    # Decode 吞吐（受 KV Cache 大小和 batch 影响）
    # 参考：单 A100 约 100-200 tokens/s（batch=1）
    # 连续批处理可以达到 1000+ tokens/s
    decode_throughput_per_gpu = 1000  # tokens/s（批处理后）
    
    # 需要的 Prefill 算力
    prefill_demand = qps * avg_input_tokens  # tokens/s
    print(f"Prefill 需求: {prefill_demand:,} tokens/s")
    
    # 需要的 Decode 算力
    decode_demand = qps * avg_output_tokens  # tokens/s
    print(f"Decode 需求: {decode_demand:,} tokens/s")
    
    # 内存：70B INT4 = 35 GB，KV Cache 约 50 GB（2× A100 = 160 GB）
    # 每个推理实例：2× A100 80GB（张量并行）
    model_mem_gb = 35  # AWQ INT4
    kv_per_token_kb = 2 * 80 * 8 * 128 * 2 / 1024  # ~327 KB
    
    gpus_per_instance = 2  # TP=2
    instance_mem_gb = 80 * gpus_per_instance
    kv_cache_gb = instance_mem_gb - model_mem_gb - 5  # 留 buffer
    max_concurrent_tokens = int(kv_cache_gb * 1e9 / (kv_per_token_kb * 1024))
    
    print(f"\n每实例（2× A100）:")
    print(f"  KV Cache 可用内存: {kv_cache_gb} GB")
    print(f"  最大并发 token 数: {max_concurrent_tokens:,}")
    print(f"  最大并发请求数: {max_concurrent_tokens // (avg_input_tokens + avg_output_tokens)}")
    
    # 需要的实例数
    # 保守估计：每实例能处理 50 QPS（连续批处理）
    qps_per_instance = 50
    num_instances = qps / qps_per_instance
    
    print(f"\n推荐实例数: {num_instances:.0f}（总 {num_instances*2:.0f}× A100）")

capacity_planning()
```

## 第二步：系统架构设计

```
┌─────────────────────────────────────────────────────┐
│                   客户端                             │
└──────────────────────┬──────────────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────────────┐
│               API Gateway / 负载均衡                  │
│  - Rate Limiting（限流）                             │
│  - Authentication（认证）                            │
│  - Request Routing（路由）                           │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┴───────────────┐
        │                             │
┌───────▼────────┐           ┌────────▼───────┐
│  调度层         │           │  流量管理       │
│  (Dispatcher)  │           │  Priority Queue │
│  - 负载感知路由  │           │  - 长请求降优先  │
│  - 批处理编排   │           │  - SLO 保障     │
└───────┬────────┘           └────────────────┘
        │
   ┌────┴────────────────────────┐
   │                             │
┌──▼──────────────┐   ┌──────────▼──────────┐
│  推理实例 #1     │   │  推理实例 #2          │
│  2× A100 80GB   │   │  2× A100 80GB        │
│  vLLM + AWQ     │   │  vLLM + AWQ          │
│  LLaMA-2 70B    │   │  LLaMA-2 70B         │
└─────────────────┘   └──────────────────────┘
        │                             │
┌───────▼─────────────────────────────▼───────┐
│               KV Cache 管理                   │
│  (vLLM PagedAttention / 未来可用分布式 KV)    │
└──────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────┐
│               监控与可观测性                   │
│  Prometheus + Grafana                        │
│  - 请求延迟（TTFT / TPOT / E2E）              │
│  - 吞吐量（TPS）                              │
│  - KV Cache 利用率                            │
└──────────────────────────────────────────────┘
```

## 第三步：vLLM 推理服务配置

```python
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ===== 服务端配置 =====
engine_args = AsyncEngineArgs(
    model="Llama-2-70b-AWQ-4bit",
    tensor_parallel_size=2,           # 2× A100
    quantization="awq",
    gpu_memory_utilization=0.90,
    max_num_seqs=256,                  # 最大并发请求数
    max_num_batched_tokens=8192,       # 最大批处理 token 数
    enable_prefix_caching=True,        # 前缀缓存（重复 prompt 加速）
    max_model_len=4096,
    dtype="float16",
    swap_space=4,                      # CPU swap（GB，超出 GPU KV Cache 时使用）
)

engine = AsyncLLMEngine.from_engine_args(engine_args)

# ===== FastAPI 服务 =====
app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """处理文本生成请求"""
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )
    
    # 异步流式生成
    request_id = f"req_{id(request)}"
    
    if request.stream:
        return StreamingResponse(
            stream_results(engine, request.prompt, sampling_params, request_id),
            media_type="text/event-stream"
        )
    
    # 非流式
    final_output = None
    async for output in engine.generate(request.prompt, sampling_params, request_id):
        final_output = output
    
    return {
        "choices": [{
            "text": final_output.outputs[0].text,
            "finish_reason": final_output.outputs[0].finish_reason,
            "usage": {
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
            }
        }]
    }

async def stream_results(engine, prompt, sampling_params, request_id):
    """SSE 流式输出"""
    prev_text = ""
    async for output in engine.generate(prompt, sampling_params, request_id):
        current_text = output.outputs[0].text
        new_text = current_text[len(prev_text):]
        prev_text = current_text
        
        yield f"data: {new_text}\n\n"
    
    yield "data: [DONE]\n\n"

# ===== 健康检查 =====
@app.get("/health")
async def health_check():
    return {"status": "ok", "engine": "ready"}

@app.get("/metrics")
async def get_metrics():
    """暴露 Prometheus 指标"""
    stats = engine.get_stats()  # vLLM 内置统计
    return {
        "num_running": stats.num_running,
        "num_waiting": stats.num_waiting,
        "gpu_cache_usage": stats.gpu_cache_usage,
        "cpu_cache_usage": stats.cpu_cache_usage,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
```

## 第四步：延迟优化

```python
"""
关键延迟指标：
- TTFT（Time To First Token）：首 token 延迟 = Prefill 时间
- TPOT（Time Per Output Token）：每个输出 token 的生成时间
- E2E：端到端延迟 = TTFT + TPOT × output_tokens
"""

class LatencyOptimizer:
    """延迟优化策略"""
    
    @staticmethod
    def prefill_chunking(max_prefill_tokens: int = 512):
        """
        分块 Prefill（Chunked Prefill）
        
        问题：长 prompt 的 Prefill 会阻塞 Decode
        解决：将 Prefill 切成小块，与 Decode 交错执行
        
        效果：P99 TTFT 降低 40-60%，吞吐轻微下降
        """
        # vLLM 配置
        return {"enable_chunked_prefill": True, "max_num_batched_tokens": max_prefill_tokens}
    
    @staticmethod
    def speculative_decoding_config(draft_model: str = "llama-2-7b"):
        """
        投机解码（Week 5 内容）
        用小模型猜测，大模型验证
        """
        return {
            "speculative_model": draft_model,
            "num_speculative_tokens": 5,
        }
    
    @staticmethod
    def request_scheduling():
        """
        请求调度策略：
        - FCFS（先来先服务）：简单但 P99 高
        - Shortest Job First：对短请求友好，长请求可能饿死
        - SJF + 超时抢占：均衡方案
        """
        pass


# 延迟测量工具
async def measure_latency(endpoint: str, prompt: str, max_tokens: int = 100):
    import aiohttp
    import time
    
    ttft = None
    e2e_start = time.time()
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{endpoint}/v1/completions",
            json={"prompt": prompt, "max_tokens": max_tokens, "stream": True}
        ) as resp:
            first_token = True
            async for chunk in resp.content.iter_any():
                if first_token and chunk.strip():
                    ttft = time.time() - e2e_start
                    first_token = False
    
    e2e = time.time() - e2e_start
    tpot = (e2e - ttft) / max_tokens if ttft else None
    
    return {"ttft_ms": ttft * 1000, "e2e_ms": e2e * 1000, "tpot_ms": tpot * 1000}
```

## 关键性能数据

**LLaMA-2 70B，2× A100 80GB，AWQ INT4：**

| 指标 | 单实例 | 优化后 |
|------|-------|--------|
| TTFT (P50) | 800ms | 400ms (chunked prefill) |
| TTFT (P99) | 2000ms | 800ms |
| TPOT (P50) | 40ms | 25ms |
| 吞吐量 | 30 QPS | 50 QPS |
| GPU 利用率 | 75% | 90% |

**达到 100 QPS：部署 2 个推理实例（4× A100 总计）**

## 面试题

**Q: LLM 推理中 TTFT 和 TPOT 分别由什么决定？如何优化？**

A: TTFT（首 token 延迟）由 **Prefill** 阶段决定——处理所有输入 token 的时间。Prefill 是 Compute-Bound（大矩阵乘法），受 GPU 算力限制，与 prompt 长度成正比。优化方法：①Chunked Prefill（分块 Prefill，避免阻塞 Decode）；②增加并行度（TP）；③使用投机解码减少验证开销。TPOT（每个输出 token 延迟）由 **Decode** 阶段决定——一次只生成 1 个 token，是 Memory-Bound 操作（读 KV Cache 和权重）。受 HBM 带宽限制，与 KV Cache 大小（batch 大小 × 序列长度）有关。优化方法：①INT4 量化减少内存读取；②连续批处理提升 batch size（均摊读取开销）；③GQA/MQA 减少 KV Cache 大小。
