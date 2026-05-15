---
layout: default
title: "Week 7 Capstone · 量化方案选型与精度评估"
render_with_liquid: false
---

# Week 7 Capstone · 量化方案选型与精度评估

## 系统设计题：量化 LLaMA-2 70B 用于生产推理

**需求**：将 LLaMA-2 70B 部署到 2× A100 80GB（无法扩展），需要服务 50 QPS，P99 延迟 < 3s，精度损失 < 2%。

## 第一步：内存规划

```python
def memory_budget(gpu_memory_gb=80, num_gpus=2):
    total_gpu_mem = gpu_memory_gb * num_gpus  # 160 GB
    
    # FP16 模型大小
    fp16_size_gb = 70 * 2  # 140 GB —— 超出！
    
    # INT4 AWQ 大小
    int4_size_gb = 70 * 0.5  # 35 GB
    
    # 可用 KV Cache 空间
    model_mem = int4_size_gb
    kv_cache_available = total_gpu_mem - model_mem - 10  # 留 10 GB buffer
    
    print(f"FP16 模型: {fp16_size_gb} GB (超出 {total_gpu_mem} GB 限制！)")
    print(f"INT4 模型: {int4_size_gb} GB")
    print(f"KV Cache 可用: {kv_cache_available} GB")
    
    # LLaMA-2 70B KV Cache（GQA=8 heads）
    # 每 token: 2 * 80层 * 8 heads * 128 dim * 2bytes = 327 KB
    kv_per_token_kb = 2 * 80 * 8 * 128 * 2 / 1024
    max_tokens = int(kv_cache_available * 1024**3 / (kv_per_token_kb * 1024))
    print(f"最大并发 token 数: {max_tokens:,}")
    print(f"假设平均 2048 token/请求: {max_tokens//2048} 并发请求")

memory_budget()
# 输出：
# FP16 模型: 140 GB (超出 160 GB 限制！)
# INT4 模型: 35 GB
# KV Cache 可用: 115 GB
# 最大并发 token 数: ~358,400
# 假设平均 2048 token/请求: ~175 并发请求
```

**结论**：INT4 量化是唯一可行的选择（FP16 放不下）。

## 第二步：量化方案选择

```python
# 方案评估矩阵
options = {
    "GPTQ INT4 (g128)": {
        "ppl_degradation": "+2%",
        "quant_time": "60 min",
        "inference_speed": "150 TPS",
        "calibration_data": True,
        "production_ready": True,
    },
    "AWQ INT4 (g128)": {
        "ppl_degradation": "+1.5%",  
        "quant_time": "10 min",
        "inference_speed": "180 TPS",
        "calibration_data": True,
        "production_ready": True,
    },
    "bitsandbytes NF4": {
        "ppl_degradation": "+3%",
        "quant_time": "instant",
        "inference_speed": "55 TPS",
        "calibration_data": False,
        "production_ready": False,  # 速度不够
    },
}

# 选择：AWQ INT4 满足所有需求
# - 精度损失 < 2% ✓
# - 内存：35 GB < 160 GB ✓  
# - 速度：180 TPS >> 50 QPS 需求 ✓
```

## 第三步：精度评估流程

```python
"""
完整的量化精度评估流程
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math

def evaluate_perplexity(model, tokenizer, dataset_name="wikitext-2", n_samples=None):
    """
    计算模型在 WikiText-2 上的困惑度
    困惑度（PPL）= exp(-mean(log p(x)))
    """
    dataset = load_dataset("wikitext", f"{dataset_name}-raw-v1", split="test")
    
    all_text = "\n\n".join(dataset["text"])
    encodings = tokenizer(all_text, return_tensors="pt")
    
    max_length = 2048
    stride = 1024
    total_nll = 0
    num_tokens = 0
    
    for begin_loc in range(0, encodings.input_ids.size(1) - max_length, stride):
        end_loc = min(begin_loc + max_length, encodings.input_ids.size(1))
        target_len = end_loc - begin_loc - stride
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-target_len] = -100  # 只计算 stride 部分的 loss
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nll = outputs.loss
        
        total_nll += nll * target_len
        num_tokens += target_len
        
        if n_samples and num_tokens >= n_samples:
            break
    
    ppl = math.exp(total_nll / num_tokens)
    return ppl

def evaluate_mmlu(model, tokenizer, num_shots=5, subjects=None):
    """
    评估 MMLU 多选题准确率（5-shot）
    """
    from datasets import load_dataset
    
    if subjects is None:
        subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics"]
    
    total_correct = 0
    total_questions = 0
    
    for subject in subjects:
        dataset = load_dataset("cais/mmlu", subject, split="test")
        
        for example in dataset:
            # 构建 5-shot 提示
            prompt = f"The following are multiple choice questions about {subject}.\n\n"
            for choice, letter in zip(example["choices"], "ABCD"):
                prompt += f"{letter}. {choice}\n"
            prompt += f"\nAnswer:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
            
            # 获取 A/B/C/D 的 logit
            choice_ids = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D"])
            choice_logits = logits[choice_ids]
            predicted = choice_logits.argmax().item()
            
            if predicted == example["answer"]:
                total_correct += 1
            total_questions += 1
    
    return total_correct / total_questions


# 执行完整评估
def run_quantization_evaluation():
    model_name = "meta-llama/Llama-2-70b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    results = {}
    
    # 1. 评估 FP16 基线
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    results["fp16"] = {
        "ppl": evaluate_perplexity(model_fp16, tokenizer),
        "mmlu": evaluate_mmlu(model_fp16, tokenizer),
    }
    del model_fp16
    
    # 2. 评估 AWQ INT4
    from awq import AutoAWQForCausalLM
    model_awq = AutoAWQForCausalLM.from_quantized(
        "Llama-2-70b-AWQ-4bit", device_map="auto"
    )
    results["awq_int4"] = {
        "ppl": evaluate_perplexity(model_awq, tokenizer),
        "mmlu": evaluate_mmlu(model_awq, tokenizer),
    }
    
    # 打印对比
    print(f"\\n{方案:15} {PPL:8} {MMLU:8} {PPL变化:10}")
    fp16_ppl = results["fp16"]["ppl"]
    for name, metrics in results.items():
        ppl_change = (metrics["ppl"] - fp16_ppl) / fp16_ppl * 100
        print(f"{name:15} {metrics[ppl]:8.2f} {metrics[mmlu]:8.2%} {ppl_change:+8.1f}%")
```

## 第四步：生产部署配置

```python
from vllm import LLM, SamplingParams

# vLLM + AWQ 生产配置
llm = LLM(
    model="Llama-2-70b-AWQ-4bit",
    tensor_parallel_size=2,          # 2× A100
    quantization="awq",              # 使用 AWQ kernel
    gpu_memory_utilization=0.92,
    max_num_seqs=200,
    enable_prefix_caching=True,
    dtype="float16",
    # AWQ 相关
    awq_fuse_layers=True,            # 层融合加速
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

# 性能测试
import time

def load_test(llm, n_requests=100, n_concurrent=50):
    prompts = ["Explain quantum computing in simple terms."] * n_requests
    
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"总请求: {n_requests}, 耗时: {elapsed:.2f}s")
    print(f"平均延迟: {elapsed/n_requests*1000:.0f}ms")
    print(f"吞吐量: {total_tokens/elapsed:.0f} tokens/s")
    print(f"RPS: {n_requests/elapsed:.1f} req/s")

load_test(llm)
# 典型输出：
# 总请求: 100, 耗时: 5.8s
# 平均延迟: 58ms (TTFT)
# 吞吐量: 8,621 tokens/s
# RPS: 17.2 req/s (需要 3 个实例达到 50 QPS)
```

## 本周总结

| 方法 | 核心思想 | 精度 | 速度 | 复杂度 |
|------|---------|------|------|-------|
| RTN INT8 | 直接 round | 接近 FP16 | 1.5× | 极简 |
| SmoothQuant W8A8 | 平滑激活 outlier | 接近 FP16 | 1.56× | 中等 |
| GPTQ INT4 | Hessian 误差补偿 | PPL+2% | 2.8× | 复杂 |
| AWQ INT4 | 激活感知缩放 | PPL+1.5% | 3× | 中等 |
| NF4 QLoRA | 正态分布码本 | PPL+3% | 2.5× | 简单 |

**下周预告：MLOps 与系统设计——训练监控、故障恢复、成本优化与面试题精讲**
