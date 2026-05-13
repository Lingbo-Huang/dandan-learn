---
layout: post
title: "推理加速与量化"
track: "🤖 大模型"
---

# 推理加速与量化

> 模型训练好了，如何让它又快又省？vLLM、KV Cache、量化是2026年必备工程技能。

---

## 大模型推理的核心挑战

- **延迟（Latency）**：用户等待时间，TTFT（首token时间）< 1s是生产要求
- **吞吐量（Throughput）**：每秒能处理多少请求（tokens/s）
- **显存占用**：7B模型fp16需要14GB显存，量化可降到4-8GB
- **成本**：A100每小时$2-4，优化直接省钱

---

## 1. vLLM：生产级推理引擎

vLLM 是目前最主流的开源 LLM 推理框架，核心技术是 **PagedAttention**（分页注意力），将 KV Cache 的显存利用率从 60% 提升到 95%+。

### 快速部署

```bash
# 安装
pip install vllm

# 启动服务（兼容OpenAI API格式）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \   # 2块GPU张量并行
    --gpu-memory-utilization 0.9 \
    --port 8000
```

### 调用方式（完全兼容OpenAI）

```python
from openai import OpenAI

# 直接用openai客户端调用vLLM
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"  # vLLM不验证，随便填
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "解释什么是注意力机制"}],
    max_tokens=512,
    temperature=0.7
)
print(response.choices[0].message.content)
```

### 批量推理（离线场景）

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

prompts = [
    "解释LoRA微调原理",
    "什么是RAG系统",
    "描述Transformer架构"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}\n")
```

### 关键配置说明

```python
# 生产环境推荐配置
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=4,           # 4卡张量并行（大模型）
    pipeline_parallel_size=1,         # 流水线并行（超大模型）
    gpu_memory_utilization=0.90,      # 显存利用率，留10%余量
    max_model_len=32768,              # 最大上下文长度
    dtype="bfloat16",                 # bf16推理
    enable_chunked_prefill=True,      # 分块prefill，降低TTFT
    max_num_batched_tokens=8192,      # 批处理token上限
)
```

---

## 2. KV Cache 原理与优化

理解 KV Cache 是面试必考项。

### 什么是 KV Cache？

自回归生成时，每生成一个新 token，需要对**所有历史 token** 计算注意力。如果不缓存，计算量随序列长度平方增长。

**KV Cache**：缓存每一层的 Key 和 Value 矩阵，新 token 只需计算自己的 QKV，再与缓存的 KV 做注意力。

```
无KV Cache：生成第n个token需要 O(n²) 计算
有KV Cache：生成第n个token需要 O(n) 计算（缓存了前n-1个的KV）
```

### 显存占用计算

```python
def kv_cache_size(
    num_layers: int,      # 模型层数（Qwen2.5-7B: 28层）
    num_kv_heads: int,    # KV头数（GQA后的头数，如8）
    head_dim: int,        # 每头维度（通常128）
    seq_len: int,         # 序列长度
    batch_size: int,      # 批大小
    dtype_bytes: int = 2  # bf16=2字节
) -> float:
    """计算KV Cache显存占用（GB）"""
    # K和V各一份，所以乘2
    size = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
    return size / (1024**3)

# Qwen2.5-7B，序列8192，批大小8
size = kv_cache_size(28, 8, 128, 8192, 8)
print(f"KV Cache: {size:.2f} GB")  # 约3.5 GB
```

### PagedAttention（vLLM核心）

传统 KV Cache 按最大长度预分配连续显存，浪费严重。vLLM 的 PagedAttention 将 KV Cache 分成固定大小的"页"，按需分配：

```
传统KV Cache：为每个请求预分配max_len大小的连续显存
              → 短序列浪费显存，并发量低

PagedAttention：KV Cache按block（如16个token）动态分配
                → 无碎片，显存利用率>95%，并发量提升5-10x
```

---

## 3. 量化：降低显存和加速推理

### 量化对比

| 量化方案 | 精度损失 | 显存节省 | 推理速度 | 适用场景 |
|---------|---------|---------|---------|---------|
| FP16（不量化）| - | - | 基准 | 训练/高精度推理 |
| INT8 | 很小 | 50% | +10-30% | 平衡精度与速度 |
| INT4（GPTQ） | 较小 | 75% | +2-4x | 推理部署 |
| INT4（AWQ） | 最小 | 75% | +2-4x | **推荐：精度最好的4bit** |
| GGUF/Q4_K_M | 较小 | 75% | CPU可用 | 端侧/边缘部署 |

### AWQ 量化（推荐）

```python
# 量化（只需做一次）
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "Qwen/Qwen2.5-7B-Instruct"
quant_path = "./qwen2.5-7b-awq"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=True)

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,             # 4bit量化
    "version": "GEMM"
}

# 量化（需要校准数据集，几分钟）
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
print(f"量化完成，保存在 {quant_path}")
```

```bash
# vLLM加载AWQ量化模型
python -m vllm.entrypoints.openai.api_server \
    --model ./qwen2.5-7b-awq \
    --quantization awq \
    --dtype half \
    --port 8000
```

### GPTQ 量化

```python
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=gptq_config,
    device_map="auto"
)
model.save_pretrained("./qwen2.5-7b-gptq")
```

---

## 4. 投机解码（Speculative Decoding）

用小模型"猜"多个token，大模型批量验证，大幅提升推理速度：

```
普通生成：大模型 → token1 → token2 → token3 → ...（串行，慢）

投机解码：
1. 小模型一次猜4个token: [t1, t2, t3, t4]
2. 大模型批量验证4个token（并行，快！）
3. 接受前k个正确的，从第k+1个重新猜
→ 速度提升 2-3x，输出质量不变
```

```bash
# vLLM支持投机解码
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-model Qwen/Qwen2.5-0.5B-Instruct \  # 小模型做草稿
    --num-speculative-tokens 5 \                        # 每次猜5个
    --port 8000
```

---

## 5. 性能评测

```python
import asyncio
import httpx
import time
from statistics import mean

async def benchmark_vllm(
    num_requests: int = 100,
    prompt: str = "解释Transformer的注意力机制，详细说明",
    max_tokens: int = 256
):
    """评测vLLM性能"""
    client = httpx.AsyncClient(base_url="http://localhost:8000")
    
    latencies = []
    ttfts = []  # Time to First Token
    
    async def single_request():
        start = time.time()
        first_token_time = None
        token_count = 0
        
        async with client.stream(
            "POST", "/v1/completions",
            json={"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": prompt,
                  "max_tokens": max_tokens, "stream": True},
            timeout=60
        ) as r:
            async for line in r.aiter_lines():
                if line.startswith("data:") and "[DONE]" not in line:
                    if first_token_time is None:
                        first_token_time = time.time() - start
                    token_count += 1
        
        total_time = time.time() - start
        return first_token_time, total_time, token_count
    
    # 并发测试
    tasks = [single_request() for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)
    
    ttfts = [r[0] for r in results if r[0]]
    latencies = [r[1] for r in results]
    throughput = sum(r[2] for r in results) / sum(latencies)
    
    print(f"=== 性能报告 ===")
    print(f"请求数: {num_requests}")
    print(f"TTFT P50: {sorted(ttfts)[len(ttfts)//2]*1000:.0f}ms")
    print(f"TTFT P99: {sorted(ttfts)[int(len(ttfts)*0.99)]*1000:.0f}ms")
    print(f"总延迟 P50: {sorted(latencies)[len(latencies)//2]*1000:.0f}ms")
    print(f"吞吐量: {throughput:.1f} tokens/s")

asyncio.run(benchmark_vllm())
```

---

## 6. 面试高频问题

**Q: KV Cache为什么能加速推理？**
> 自回归生成时，每个新token需要和所有历史token做注意力计算，KV Cache缓存了历史token的Key/Value矩阵，避免重复计算，将推理复杂度从O(n²)降到O(n)。

**Q: INT4和FP16量化哪个更好？**
> 取决于场景：①对话/生成任务：INT4（AWQ）精度损失可接受，显存节省75%②数学/代码任务：精度敏感，优先INT8③训练/微调：必须FP16/BF16。

**Q: vLLM和Ollama的区别？**
> Ollama适合本地单用户开发测试，配置简单；vLLM适合生产环境多用户高并发，支持动态批处理、张量并行、量化，吞吐量高5-10倍。

**Q: 什么是张量并行（Tensor Parallel）？**
> 将模型的权重矩阵按列/行切分到多块GPU，每块GPU存储和计算模型的一部分，合并后得到完整结果。7B模型建议1-2块GPU，70B模型建议4-8块GPU。

---

[← LoRA/QLoRA微调实战](./lora-finetuning) | [→ LLMOps全链路](./llmops)
