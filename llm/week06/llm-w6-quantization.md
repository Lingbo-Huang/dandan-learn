---
layout: default
title: "D2 · 量化：INT8/INT4/GPTQ/AWQ"
render_with_liquid: false
---

# D2 · 量化：从 FP16 到 INT4

> **量化**：用低精度数据类型（INT8/INT4）存储模型权重，在几乎不损失精度的情况下，减少 2-4 倍内存。

---

## 一、量化基础

### 1.1 线性量化

将浮点数映射到整数区间：

$$X_{quant} = \text{round}\left(\frac{X}{s}\right) + z$$

- $s$：缩放因子（scale）
- $z$：零点（zero point）

**对称量化**（$z=0$，更简单）：$s = \frac{\max(|X|)}{127}$（INT8）

**非对称量化**：$s = \frac{\max(X) - \min(X)}{255}$，$z = -\text{round}(\min(X) / s)$

```python
import torch
import numpy as np

class LinearQuantizer:
    """对称线性量化"""
    
    def quantize_int8(self, tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
        """将 FP32/BF16 张量量化为 INT8"""
        abs_max = tensor.abs().max().item()
        scale = abs_max / 127.0  # INT8 范围 [-128, 127]
        
        quantized = torch.clamp(
            torch.round(tensor / scale), -128, 127
        ).to(torch.int8)
        
        return quantized, scale
    
    def dequantize_int8(self, quantized: torch.Tensor, scale: float) -> torch.Tensor:
        """INT8 反量化到 FP32"""
        return quantized.float() * scale
    
    def quantize_per_channel(
        self, tensor: torch.Tensor, dim: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Per-channel 量化（每行/列有独立的 scale）
        比 per-tensor 精度更高（权重分布差异大时）
        """
        # 每个输出通道独立量化
        abs_max = tensor.abs().max(dim=dim, keepdim=True).values
        scales = abs_max / 127.0
        
        quantized = torch.clamp(
            torch.round(tensor / scales), -128, 127
        ).to(torch.int8)
        
        return quantized, scales

# 量化误差分析
def analyze_quantization_error():
    W = torch.randn(1024, 1024)
    
    q = LinearQuantizer()
    
    # Per-tensor
    q_pt, s_pt = q.quantize_int8(W)
    W_dq_pt = q.dequantize_int8(q_pt, s_pt)
    
    # Per-channel
    q_pc, s_pc = q.quantize_per_channel(W, dim=0)
    W_dq_pc = (q_pc.float() * s_pc)
    
    print("量化误差分析 (1024x1024 矩阵, INT8):")
    print(f"  Per-tensor: MSE = {((W - W_dq_pt)**2).mean():.8f}")
    print(f"  Per-channel: MSE = {((W - W_dq_pc)**2).mean():.8f}")

analyze_quantization_error()
```

---

## 二、GPTQ（训练后量化）

GPTQ 是目前最主流的 LLM INT4 量化方法：

```python
"""
GPTQ 核心思想（Frantar et al., 2022）：

1. 用少量校准数据（几百个样本），获取权重对输出的影响
2. 对每个权重：找到量化误差，用 Hessian 信息补偿
3. 贪心逐列量化，后续列补偿前面列的误差

核心公式（基于 OBQ 的近似）：
w_q* = argmin_w̃ (w̃ - w)² * H_ii
δ = -(w_q* - w) / H_ii * H[i, i+1:]  （补偿后续列）
"""

# 实际使用 auto-gptq 库
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

def quantize_with_gptq(
    model_name: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
):
    """使用 GPTQ 量化模型"""
    
    quantize_config = BaseQuantizeConfig(
        bits=bits,                  # 量化位宽（4 或 8）
        group_size=group_size,      # 分组量化大小（每组独立 scale）
        desc_act=False,             # 是否按激活值重排权重（更慢但更精确）
    )
    
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
    )
    
    # 校准数据（用少量真实文本）
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    calibration_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # 准备校准 batch
    examples = []
    for text in calibration_data["text"][:128]:
        if len(text) > 50:
            ids = tokenizer.encode(text, return_tensors='pt')
            examples.append(ids)
    
    # 量化（约需 10-30 分钟）
    model.quantize(examples)
    
    # 保存
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)
    
    print(f"量化完成，保存到 {output_dir}")
    print(f"预期模型大小: {bits}/16 = {bits/16*100:.0f}% 原始大小")
```

---

## 三、AWQ（激活感知权重量化）

AWQ 的核心洞见：不是所有权重都同等重要，**激活值大的通道对应的权重更重要**。

```python
"""
AWQ (Lin et al., 2023) 原理：

1. 用校准数据统计每个通道的激活值大小
2. 重要通道（激活大）→ 减小对应权重（等效放大激活，降低量化误差）
3. 非重要通道 → 可以接受更大量化误差

等效变换：
  y = Wx ≈ (W·diag(s)) · (diag(s)^{-1}·x)
  
将大激活通道的权重缩小（s < 1），激活等比放大：
  量化权重的误差不变，但激活更大，相对误差更小
"""

def compute_awq_scale(
    weight: torch.Tensor,  # [out, in]
    activation_stats: torch.Tensor,  # [in] - 每个输入通道的激活均值
    n_bits: int = 4,
    group_size: int = 128,
) -> torch.Tensor:
    """
    计算 AWQ 的最优缩放因子
    
    对重要通道（激活大）用更大的 scale，减小量化误差
    """
    # 搜索最优 alpha（通常在 [0, 1] 之间）
    best_error = float('inf')
    best_scales = None
    
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # scale = activation^alpha（重要通道 scale 更大）
        scales = activation_stats.pow(alpha)
        scales = scales / (scales.max() * scales.min()).sqrt()
        
        # 应用 scale 后量化
        W_scaled = weight * scales.unsqueeze(0)
        W_q, s = quantize_symmetric(W_scaled, n_bits)
        W_dq = W_q.float() * s
        W_dq = W_dq / scales.unsqueeze(0)  # 恢复 scale
        
        error = ((weight - W_dq) ** 2).mean().item()
        if error < best_error:
            best_error = error
            best_scales = scales.clone()
    
    return best_scales

def quantize_symmetric(tensor, bits):
    """对称量化（辅助函数）"""
    max_int = 2 ** (bits - 1) - 1
    scale = tensor.abs().max() / max_int
    quantized = (tensor / scale).round().clamp(-max_int, max_int).to(torch.int8)
    return quantized, scale

# 实际使用 autoawq 库
# from awq import AutoAWQForCausalLM
#
# model = AutoAWQForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
# model.quantize(tokenizer, quant_config={
#     "zero_point": True,
#     "q_group_size": 128,
#     "w_bit": 4,
#     "version": "GEMM"
# })
# model.save_quantized("./qwen2.5-7b-awq")
```

---

## 四、量化方法对比

```python
"""
各量化方法对比（以 7B 模型为例）

方法        位宽    速度（量化）   质量      显存
------      ----    -----------   ------    -----
BitsAndBytes INT8    快（运行时）   几乎无损   约8GB
BitsAndBytes NF4     快（运行时）   轻微损失   约5GB
GPTQ        INT4    慢（需校准）   轻微损失   约4GB
AWQ         INT4    中等           损失小     约4GB
GGUF/llama.cpp INT4 快（CPU友好） 轻微损失   约4GB
"""

# 推荐选择策略
def choose_quantization(
    gpu_vram_gb: float,
    need_training: bool,
    latency_sensitive: bool,
) -> str:
    if need_training:
        if gpu_vram_gb >= 24:
            return "BitsAndBytes NF4 + LoRA (QLoRA)"
        else:
            return "可能需要多卡或更小的模型"
    
    if gpu_vram_gb >= 16:
        return "全精度 BF16 推理"
    elif gpu_vram_gb >= 10:
        return "BitsAndBytes INT8（运行时量化，最简单）"
    elif gpu_vram_gb >= 6:
        if latency_sensitive:
            return "AWQ INT4（精度最好的 4bit 方案）"
        else:
            return "GPTQ INT4（吞吐量优化）"
    else:
        return "GGUF INT4（CPU 或混合推理）"

# 示例
print("根据硬件选择量化方案:")
setups = [
    (80, False, True, "A100 80GB"),
    (40, False, True, "A100 40GB"),
    (24, True, False, "4090 训练"),
    (24, False, True, "4090 推理"),
    (12, False, True, "3060 12GB"),
    (8, False, False, "CPU 大内存"),
]

for vram, training, latency, desc in setups:
    rec = choose_quantization(vram, training, latency)
    print(f"  {desc}: {rec}")
```

---

## 五、面试题精讲

**Q: INT8 量化为什么几乎不损失精度，而 INT4 有明显损失？**

A: INT8 有 256 个量化点（-128 到 127），对大多数正态分布的权重，量化误差约为 0.3%。INT4 只有 16 个量化点，对于分布尾部的权重量化误差很大。GPTQ 和 AWQ 通过 group quantization（每 128 个权重共享一个 scale）和对重要权重的精细处理，将 INT4 的误差控制在可接受范围。

**Q: 量化对 KV Cache 有帮助吗？**

A: 有，KV Cache 量化（INT8）可以在不明显影响质量的情况下，将 KV Cache 内存减半。vLLM 支持 `--kv-cache-dtype fp8` 等配置。

---

## 小结

```
量化选择路径：
  显存充裕(>=40GB) → BF16 全精度
  显存适中(16-40GB) → INT8（BitsAndBytes）
  显存紧张(8-16GB) → INT4 GPTQ/AWQ
  CPU/边缘 → GGUF INT4

精度：BF16 > INT8 > AWQ-INT4 ≈ GPTQ-INT4 > NF4
速度：AWQ-INT4 > BF16 > INT8 > GPTQ-INT4
```
