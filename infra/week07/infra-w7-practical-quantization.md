---
layout: default
title: "D5 · 实战：bitsandbytes / AutoGPTQ / llama.cpp"
render_with_liquid: false
---

# D5 · 实战：量化工具链实战指南

## 工具链全景

| 工具 | 量化方法 | 精度 | 部署平台 | 特点 |
|------|---------|------|---------|------|
| bitsandbytes | LLM.int8(), NF4 | INT8/NF4 | GPU | 最易用，PyTorch 原生 |
| AutoGPTQ | GPTQ | INT4/INT3/INT2 | GPU | 精度好，速度较快 |
| AutoAWQ | AWQ | INT4 | GPU | 精度最好，速度最快 |
| llama.cpp | GGUF格式 | INT4/INT8 | CPU/Metal/CUDA | 跨平台，CPU可用 |
| ExLlamaV2 | EXL2 | 2-8 bit 混合 | GPU | 推理速度极快 |

## bitsandbytes：最简单的量化集成

bitsandbytes 与 HuggingFace Transformers 无缝集成，只需修改加载配置即可：

### INT8 量化（LLM.int8()）

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_int8_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # outlier 阈值，超过保持 FP16
)

model_int8 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_int8_config,
    device_map="auto"
)
print(f"INT8 内存: {model_int8.get_memory_footprint()/1e9:.2f} GB")  # ~7 GB
```

### NF4 量化 + QLoRA 微调

```python
bnb_nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # 再量化 scale 因子
)

model_nf4 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_nf4_config,
    device_map="auto"
)

# 添加 LoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_nf4 = prepare_model_for_kbit_training(model_nf4)
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
model_lora = get_peft_model(model_nf4, lora_config)
# trainable params: 8.4M (0.24% of 3.5B)
```

## AutoGPTQ：高质量 INT4

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4, group_size=128, desc_act=False, sym=True
)
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantize_config=quantize_config,
    torch_dtype=torch.float16
)

# 校准并量化（约 30-60 分钟，需要 calibration_data）
model.quantize(calibration_data)
model.save_quantized("Llama-2-13b-GPTQ-int4", use_safetensors=True)

# 加载推理
model_quant = AutoGPTQForCausalLM.from_quantized(
    "Llama-2-13b-GPTQ-int4", device_map="auto",
    use_triton=True, inject_fused_attention=True
)
```

## AutoAWQ：推荐的 GPU 推理方案

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype="auto", device_map="auto"
)
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model.quantize(tokenizer, quant_config=quant_config)  # 约 5 分钟
model.save_quantized("Llama-2-7b-AWQ-4bit")

# 推理（fuse_layers 额外加速 ~20%）
model_q = AutoAWQForCausalLM.from_quantized("Llama-2-7b-AWQ-4bit", fuse_layers=True)
```

## llama.cpp：CPU/边缘设备部署

量化格式说明：
- Q4_0: 4-bit 均匀量化（最快但精度低）
- Q4_K_M: 4-bit K-quant，混合精度（精度/速度平衡，推荐）
- Q5_K_M: 5-bit K-quant（更高精度）
- Q8_0: 8-bit 量化（最高精度）

```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
make -j8
python3 convert.py /path/to/llama-2-7b/ --outtype f16 --outfile llama-2-7b-f16.gguf
./quantize llama-2-7b-f16.gguf llama-2-7b-Q4_K_M.gguf Q4_K_M
./main -m llama-2-7b-Q4_K_M.gguf -n 256 -p "Once upon a time" -t 8 --gpu-layers 35
```

```python
from llama_cpp import Llama
llm = Llama(model_path="./llama-2-7b-Q4_K_M.gguf", n_ctx=4096, n_threads=8, n_gpu_layers=35)
response = llm("What is quantization in machine learning?", max_tokens=200)
print(response["choices"][0]["text"])
```

## 各方案性能对比（LLaMA-2 7B，A100 80GB）

| 方案 | 显存 | 速度(TPS) | MMLU | 最佳用途 |
|------|------|---------|------|--------|
| FP16 | 14 GB | 80 | 45.3% | 最高精度 |
| bitsandbytes INT8 | 7 GB | 65 | 45.1% | 通用 |
| bitsandbytes NF4 | 4 GB | 55 | 44.6% | QLoRA 微调 |
| AutoGPTQ INT4 | 4 GB | 150 | 44.8% | 高质量推理 |
| AutoAWQ INT4 | 4 GB | 180 | 45.0% | 最佳推理 |
| llama.cpp Q4_K_M | 4 GB RAM | 15(CPU) | 44.5% | CPU/边缘 |

## 选型指南

| 需求 | 推荐方案 |
|------|---------|
| GPU 推理，速度优先 | AutoAWQ INT4 |
| GPU 推理，精度优先 | AutoGPTQ INT4 + desc_act |
| 内存极限优化 | AWQ/GPTQ INT4 |
| QLoRA 微调 | bitsandbytes NF4 |
| CPU/边缘部署 | llama.cpp Q4_K_M |
| 最大并发 | AWQ + vLLM |

## 面试题

**Q: INT4 量化在 Memory-Bound 场景下为什么比 INT8 快约 2×？**

A: Memory-Bound 场景（小 batch size 解码阶段）的性能瓶颈是 HBM 内存带宽。INT4 将每个参数从 1 byte 降至 0.5 byte，读取同等数量参数所需时间减半，直接获得约 2× 速度提升。Compute-Bound 场景（大 batch prefill）INT4 加速比较小，因为瓶颈在 Tensor Core 算力。这也是量化对推理解码阶段效果最显著的原因。
