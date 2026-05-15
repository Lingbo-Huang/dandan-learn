---
layout: default
title: "D6 · Capstone：生产级推理服务"
render_with_liquid: false
---

# D6 · Capstone：搭建生产级 LLM 推理服务

> **目标**：用 vLLM 搭建一个生产级推理服务，支持流式输出、多并发、监控，并对延迟和吞吐量进行基准测试。

---

## 一、服务架构

```
Client → Nginx → vLLM Server → GPU
                   ↓
              Prometheus → Grafana
```

---

## 二、vLLM 服务启动配置

```bash
#!/bin/bash
# start_vllm.sh

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000

vllm serve $MODEL \
  --host 0.0.0.0 \
  --port $PORT \
  --served-model-name "qwen2.5-7b" \
  \
  # 内存配置
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 256 \         # 最大并发序列数
  \
  # 推理优化
  --enable-chunked-prefill \   # 分块 prefill（降低长 prompt 延迟）
  --max-num-batched-tokens 8192 \
  \
  # 多 GPU（如果有）
  --tensor-parallel-size 2 \
  \
  # 量化（可选）
  # --quantization awq \
  \
  # 投机采样（可选）
  # --speculative-model "Qwen/Qwen2.5-1.5B-Instruct" \
  # --num-speculative-tokens 5 \
  \
  # API
  --api-key "your-secret-key" \
  --chat-template "/path/to/template.jinja"
```

---

## 三、Python 客户端实现

```python
# llm_client.py
import asyncio
import httpx
import time
from typing import AsyncIterator

class LLMClient:
    """生产级 LLM 客户端"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000",
        api_key: str = "your-secret-key",
        model: str = "qwen2.5-7b",
        timeout: float = 60.0,
    ):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout),
        )
    
    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """发送 chat 请求"""
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        
        if stream:
            return self._stream_chat(payload)
        else:
            resp = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    
    async def _stream_chat(self, payload: dict) -> AsyncIterator[str]:
        """流式输出"""
        import json
        
        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]
    
    async def close(self):
        await self.client.aclose()


# 使用示例
async def demo():
    client = LLMClient()
    
    # 普通调用
    response = await client.chat(
        messages=[{"role": "user", "content": "什么是 Attention 机制？"}],
        max_tokens=200
    )
    print(f"Response: {response}")
    
    # 流式调用
    print("Stream: ", end="", flush=True)
    async for chunk in await client.chat(
        messages=[{"role": "user", "content": "用一句话解释 LoRA"}],
        stream=True
    ):
        print(chunk, end="", flush=True)
    print()
    
    await client.close()

asyncio.run(demo())
```

---

## 四、基准测试

```python
# benchmark.py
import asyncio
import time
import statistics
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    concurrency: int
    total_requests: int
    total_time_s: float
    success_count: int
    error_count: int
    throughput_rps: float           # 请求/秒
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    tokens_per_second: float

async def benchmark_llm(
    base_url: str,
    concurrency: int = 10,
    num_requests: int = 100,
    max_tokens: int = 256,
) -> BenchmarkResult:
    """LLM 推理基准测试"""
    
    client = LLMClient(base_url)
    semaphore = asyncio.Semaphore(concurrency)
    
    latencies = []
    token_counts = []
    errors = 0
    
    async def single_request(i: int) -> None:
        async with semaphore:
            start = time.perf_counter()
            try:
                response = await client.chat(
                    messages=[{
                        "role": "user",
                        "content": f"请解释大模型技术概念 #{i % 20}"
                    }],
                    max_tokens=max_tokens
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)
                token_counts.append(len(response.split()))  # 近似
            except Exception as e:
                nonlocal errors
                errors += 1
    
    # 并发执行
    start_time = time.perf_counter()
    tasks = [single_request(i) for i in range(num_requests)]
    await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time
    
    await client.close()
    
    if not latencies:
        raise RuntimeError("所有请求失败")
    
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    
    return BenchmarkResult(
        concurrency=concurrency,
        total_requests=num_requests,
        total_time_s=total_time,
        success_count=len(latencies),
        error_count=errors,
        throughput_rps=len(latencies) / total_time,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=latencies_sorted[n // 2],
        p95_latency_ms=latencies_sorted[int(n * 0.95)],
        p99_latency_ms=latencies_sorted[int(n * 0.99)],
        tokens_per_second=sum(token_counts) / total_time,
    )

async def run_benchmark():
    """运行不同并发下的基准测试"""
    base_url = "http://localhost:8000"
    
    print(f"{'并发数':<8} {'QPS':<10} {'P50(ms)':<12} {'P95(ms)':<12} {'P99(ms)':<12} {'Tok/s'}")
    print("-" * 70)
    
    for concurrency in [1, 4, 8, 16, 32]:
        result = await benchmark_llm(
            base_url,
            concurrency=concurrency,
            num_requests=concurrency * 5,
        )
        print(f"{result.concurrency:<8} "
              f"{result.throughput_rps:<10.1f} "
              f"{result.p50_latency_ms:<12.0f} "
              f"{result.p95_latency_ms:<12.0f} "
              f"{result.p99_latency_ms:<12.0f} "
              f"{result.tokens_per_second:.0f}")

asyncio.run(run_benchmark())
```

---

## 五、监控与运维

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

```python
# 关键监控指标（vLLM 自带 Prometheus 指标）
"""
vllm:num_requests_running          - 当前运行中的请求数
vllm:num_requests_waiting          - 等待队列长度
vllm:gpu_cache_usage_perc          - KV Cache 使用率
vllm:generation_tokens_total       - 已生成 token 总数
vllm:request_prompt_tokens         - prompt token 分布
vllm:request_generation_tokens     - 生成 token 分布
vllm:time_to_first_token_seconds   - TTFT（首 token 时间）
vllm:time_per_output_token_seconds - TPOT（每 token 时间）

关键告警规则：
- KV Cache > 90%：即将 OOM，考虑限流
- 等待队列 > 100：服务过载
- P99 TTFT > 5s：用户体验差
"""
```

---

## 六、Week 6 总结

| 技术 | 核心概念 | 适用场景 |
|------|---------|---------|
| KV Cache | 缓存历史 K/V，避免重计算 | 所有推理场景（必须用）|
| GQA | 多 Q 共享 KV，省内存 | 现代 LLM 默认配置 |
| INT4 量化 | 4bit 存储权重，省 4x 内存 | 显存受限场景 |
| PagedAttention | OS 分页管理 KV Cache | 高并发服务（vLLM）|
| 连续批处理 | 动态调整批次，GPU 100% | 生产服务必备 |
| 投机采样 | 小模型猜+大模型验证 | 代码生成等场景 |

**实践建议**：
- 开发/测试：Ollama + 量化模型
- 生产服务：vLLM + AWQ INT4 量化 + 投机采样（可选）
- 大规模：vLLM + 张量并行 + Prometheus 监控
