---
layout: default
title: "D3 · GPTQ：基于 Hessian 的权重量化"
render_with_liquid: false
---

# D3 · GPTQ：基于 Hessian 的权重量化

## OBQ：最优大脑外科手术

GPTQ（[Frantar et al., 2022](https://arxiv.org/abs/2210.17323)）基于 1990 年代的 OBS（Optimal Brain Surgeon）算法：

**核心思想**：量化一个权重时，调整其他权重来补偿误差。

### 目标函数

对于权重矩阵 $W$，量化后的误差（以输出误差衡量）：

$$\min_{\hat{W}} \|WX - \hat{W}X\|_F^2 = \min_{\Delta W} \|\Delta W \cdot X\|_F^2$$

其中 $\Delta W = W - \hat{W}$。

对第 $q$ 个权重量化，引入的误差为：
$$\delta_q = \frac{w_q - \text{quant}(w_q)}{[H^{-1}]_{qq}} \cdot (H^{-1})_{:,q}$$

其中 $H = 2XX^T$（Hessian 矩阵的近似）。

### 直觉理解

```python
"""
OBQ 直觉：用一维例子理解补偿误差
"""
import torch
import numpy as np

def obq_1d_demo():
    """
    1D 例子：2 个权重 [w1, w2]，量化 w1 后补偿 w2
    目标：最小化 (w1*x1 + w2*x2 - quant(w1)*x1 - w2_new*x2)^2
    """
    w1, w2 = 1.2, 0.8
    x1, x2 = 3.0, 2.0
    
    # 量化 w1（Round to Nearest）
    scale = 0.25
    w1_q = round(w1 / scale) * scale  # = 1.25
    
    # 未补偿的误差
    error_before = (w1*x1 + w2*x2) - (w1_q*x1 + w2*x2)
    print(f"量化前输出: {w1*x1 + w2*x2:.3f}")
    print(f"量化后（无补偿）输出: {w1_q*x1 + w2*x2:.3f}")
    print(f"误差: {error_before:.3f}")
    
    # OBQ：调整 w2 来补偿 w1 的量化误差
    # 目标：w1_q*x1 + w2_new*x2 ≈ w1*x1 + w2*x2
    # → w2_new = w2 + (w1-w1_q)*x1/x2
    quant_error = w1 - w1_q
    w2_new = w2 + quant_error * x1 / x2
    
    error_after = (w1*x1 + w2*x2) - (w1_q*x1 + w2_new*x2)
    print(f"\n补偿后 w2: {w2:.3f} → {w2_new:.3f}")
    print(f"量化后（有补偿）输出: {w1_q*x1 + w2_new*x2:.3f}")
    print(f"误差: {error_after:.6f}")  # 接近 0

obq_1d_demo()
```

## GPTQ 算法

GPTQ 对 OBQ 做了两个关键改进，使其能在 GPT 规模（175B）上运行：

### 改进 1：按列顺序量化（Arbitrary Order Insight）

OBQ 的最优顺序（每步选误差最大的权重量化）很昂贵——O(d_col³) per 列。

**洞察**：对于 Transformer 权重，任意固定顺序（如从左到右按列）效果几乎一样好！

### 改进 2：Lazy Batch Update

不要每量化一个权重就更新所有权重（O(d²) × d = O(d³)），而是每次处理一批列，利用矩阵乘法并行化。

```python
import torch

def gptq(W: torch.Tensor, H: torch.Tensor, bits: int = 4, group_size: int = 128):
    """
    GPTQ 算法实现
    W: [out_features, in_features] - 待量化权重
    H: [in_features, in_features] - Hessian 矩阵（H = 2 * X @ X.T）
    bits: 量化位数
    group_size: 量化粒度（-1 表示 per-channel）
    
    返回：量化后的权重 Q 和量化参数
    """
    out_features, in_features = W.shape
    
    # Hessian 正则化（防止数值不稳定）
    damp_percent = 0.01
    diagonal = torch.diag(H)
    H += damp_percent * diagonal.mean() * torch.eye(in_features, device=H.device)
    
    # Cholesky 分解的逆（数值稳定版本）
    H_inv = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(H_inv)
    H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)
    # H_inv_chol[i,j]: i<=j 时的值
    
    Q = torch.zeros_like(W)
    Err = torch.zeros_like(W)
    
    block_size = 128  # Lazy batch update 的块大小
    
    for col_start in range(0, in_features, block_size):
        col_end = min(col_start + block_size, in_features)
        W_block = W[:, col_start:col_end].clone()  # [out, block]
        Q_block = torch.zeros_like(W_block)
        Err_block = torch.zeros_like(W_block)
        H_inv_block = H_inv_chol[col_start:col_end, col_start:col_end]
        
        for col in range(col_end - col_start):
            global_col = col_start + col
            w = W_block[:, col]  # [out_features] 当前列
            
            # 获取 Hessian 对角元素
            d = H_inv_block[col, col]  # 标量
            
            # 确定量化参数（per-group）
            if group_size != -1 and global_col % group_size == 0:
                # 当前 group 的量化参数
                group_start = global_col
                group_end = min(global_col + group_size, in_features)
                group_weights = W[:, group_start:group_end]
                
                scale = group_weights.abs().max(dim=1).values / (2**(bits-1) - 1)
                scale = scale.clamp(min=1e-8)
            
            # 量化当前列
            q = torch.clamp(torch.round(w / scale), -(2**(bits-1)), 2**(bits-1)-1)
            Q_block[:, col] = q
            
            # 计算量化误差
            err = (w - q * scale) / d
            Err_block[:, col] = err
            
            # 更新当前块内后续列的权重（补偿）
            # W_block[:, col+1:] -= err @ H_inv_block[col, col+1:]
            W_block[:, col+1:] -= torch.outer(err, H_inv_block[col, col+1:])
        
        Q[:, col_start:col_end] = Q_block
        
        # 更新整个权重矩阵的后续部分（Lazy batch update）
        W[:, col_end:] -= Err_block @ H_inv_chol[col_start:col_end, col_end:]
    
    return Q, scale  # 量化权重（整型）和量化参数

# 计算 Hessian 矩阵（在校准数据上）
def compute_hessian(module, calibration_data, device='cuda'):
    """收集激活值，计算 Hessian 矩阵"""
    H = None
    n_samples = 0
    
    def hook(m, inp, out):
        nonlocal H, n_samples
        x = inp[0].detach()  # [batch, seq, in_features]
        x = x.reshape(-1, x.shape[-1])  # [batch*seq, in_features]
        
        if H is None:
            H = torch.zeros(x.shape[1], x.shape[1], device=device)
        
        H += x.T @ x  # [in, in]
        n_samples += x.shape[0]
    
    handle = module.register_forward_hook(hook)
    
    with torch.no_grad():
        for batch in calibration_data:
            module(batch.to(device))
    
    handle.remove()
    H = 2 * H / n_samples  # 归一化
    return H
```

## GPTQ 的实际效果

**LLaMA-2 7B，WikiText-2 困惑度（PPL，越低越好）：**

| 量化方法 | PPL | 与 FP16 的差 |
|---------|-----|------------|
| FP16 基线 | 5.47 | 0% |
| RTN INT4 (per-tensor) | 6.82 | +25% |
| RTN INT4 (per-group 128) | 5.74 | +5% |
| GPTQ INT4 (per-group 128) | 5.58 | +2% |
| GPTQ INT3 (per-group 128) | 5.89 | +8% |
| GPTQ INT2 (per-group 128) | 7.56 | +38% |

## 实战：使用 AutoGPTQ

```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset

# 量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,                  # INT4
    group_size=128,          # Per-group 128
    desc_act=True,           # 使用激活值顺序（进一步提升精度）
)

# 加载原始模型
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 准备校准数据（128 条样本）
data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
examples = [tokenizer(example["text"]) for example in data.select(range(128))]

# 执行量化（需要 GPU，耗时约 30 分钟对 7B 模型）
model.quantize(examples, batch_size=1)

# 保存量化模型
model.save_quantized("Llama-2-7b-GPTQ-4bit", use_safetensors=True)

# 加载量化模型（推理）
model_quantized = AutoGPTQForCausalLM.from_quantized(
    "Llama-2-7b-GPTQ-4bit",
    device="cuda:0",
    use_triton=True,  # 使用 Triton kernel 加速
)

# 推理
input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    output = model_quantized.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

## GPTQ vs Round-to-Nearest (RTN)

| 方面 | RTN | GPTQ |
|------|-----|------|
| 方法 | 直接 round | 基于 Hessian 补偿 |
| 精度 | 较差（尤其 INT4）| 显著更好 |
| 量化时间 | 秒级 | 分钟-小时级 |
| 内存需求 | 低 | 需要计算 Hessian |
| 实现复杂度 | 简单 | 复杂 |

## 面试题

**Q: GPTQ 为什么比直接 round-to-nearest 量化效果好？**

A: 直接 RTN 量化将每个权重独立 round 到最近的整数，没有考虑权重之间的相关性，每个权重的量化误差直接累积到输出误差中。GPTQ 基于 OBQ 思想：量化一个权重后，利用 Hessian 矩阵（H = 2XX^T，反映权重对输出的影响）调整该行其他权重来补偿误差。具体地，量化误差 δ_q 按 H⁻¹ 的比例分摊到其他权重。由于 Hessian 包含了输入数据的统计信息，这种补偿能显著减小最终输出误差。实测在 INT4 时，GPTQ 的 PPL 比 RTN 低约 3%，相当于减少了约 60% 的量化带来的精度损失。
