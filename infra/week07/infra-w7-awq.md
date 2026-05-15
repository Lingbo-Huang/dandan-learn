---
layout: default
title: "D4 · AWQ：激活感知权重量化"
render_with_liquid: false
---

# D4 · AWQ：激活感知权重量化

## AWQ 的出发点

GPTQ 基于 Hessian 矩阵，计算量大（需要 Cholesky 分解）。

**AWQ**（[Lin et al., 2023](https://arxiv.org/abs/2306.00978)）的核心洞察：

> **只有 1% 的"显著权重"（Salient Weights）对模型精度至关重要，这些显著权重对应激活值中的大值通道。**

如果我们能保护这 1% 的权重不被量化损坏，就能以极低代价维持模型精度。

## 激活感知的权重重要性

```python
import torch
import torch.nn.functional as F

def find_salient_weights(
    weight: torch.Tensor,           # [out, in]
    activation_samples: torch.Tensor,  # [num_samples, seq, in]
    top_k_ratio: float = 0.01      # 1% 显著权重
) -> torch.Tensor:
    """
    找到显著权重（activation-aware saliency）
    
    权重 w_ij 的重要性 ≈ |w_ij| × mean(|x_j|)
    （权重绝对值 × 对应输入通道的激活大小）
    """
    # 激活值每个通道的平均绝对值
    act_mean = activation_samples.abs().reshape(-1, activation_samples.shape[-1]).mean(dim=0)
    # act_mean: [in_features]
    
    # 权重重要性 = 权重大小 × 激活大小
    importance = weight.abs() * act_mean.unsqueeze(0)  # [out, in]
    
    # 找到 top-k 重要的权重位置
    k = int(importance.numel() * top_k_ratio)
    threshold = importance.flatten().kthvalue(importance.numel() - k).values
    salient_mask = importance >= threshold  # [out, in]
    
    return salient_mask, importance

# 验证：显著权重真的重要吗？
def test_salient_weights():
    torch.manual_seed(42)
    W = torch.randn(256, 256) * 0.1
    X = torch.randn(100, 64, 256) * 2.0  # 激活值，部分 channel 较大
    
    # 故意让某些 channel 的激活值较大（模拟 outlier）
    X[:, :, [50, 100, 150, 200]] *= 20
    
    # 找显著权重
    salient_mask, importance = find_salient_weights(W, X, top_k_ratio=0.01)
    
    # 基线：均匀量化（INT4）
    def quantize_int4(w):
        scale = w.abs().max() / 7
        q = torch.clamp(torch.round(w / scale), -7, 7)
        return q * scale
    
    W_q_uniform = quantize_int4(W)
    
    # 保护显著权重（不量化）
    W_q_protected = quantize_int4(W)
    W_q_protected[salient_mask] = W[salient_mask]  # 恢复显著权重
    
    # 比较输出误差
    X_flat = X.reshape(-1, 256)
    out_fp16  = X_flat @ W.T
    out_uniform  = X_flat @ W_q_uniform.T
    out_protected = X_flat @ W_q_protected.T
    
    err_uniform   = (out_fp16 - out_uniform).abs().mean().item()
    err_protected = (out_fp16 - out_protected).abs().mean().item()
    
    print(f"均匀量化误差:   {err_uniform:.4f}")
    print(f"保护显著权重后: {err_protected:.4f}（{err_uniform/err_protected:.1f}× 改善）")
    print(f"显著权重比例:   {salient_mask.float().mean():.1%}")

test_salient_weights()
# 典型输出：
# 均匀量化误差:   0.4521
# 保护显著权重后: 0.1203（3.8× 改善）
# 显著权重比例:   1.0%
```

## AWQ 的核心算法：Per-Channel Scaling

直接保留 1% 的 FP16 权重会导致混合精度实现复杂且低效。

**AWQ 的解决方案**：不直接保留显著权重，而是对权重的重要通道做**缩放**，使量化更精确。

等价变换（类似 SmoothQuant，但用于权重）：

$$Y = XW = (X \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} \cdot W) = \hat{X} \cdot \hat{W}$$

对于重要通道 $j$（大激活值），增大 $s_j > 1$：
- $\hat{W}_{:,j} = W_{:,j} / s_j$（权重变小）
- $\hat{X}_{:,j} = X_{:,j} \cdot s_j$（激活变大，但激活不量化）

**直觉**：将重要权重通道缩小，减少量化误差；通过增大对应激活值来补偿，使输出不变。

```python
import torch
import torch.nn as nn

def awq_search_scale(
    weight: torch.Tensor,           # [out, in]
    activation_samples: torch.Tensor, # [num_samples, in]
    bits: int = 4,
    n_grid: int = 20,              # 搜索 grid 精度
) -> torch.Tensor:
    """
    AWQ Scale 搜索：找到最优的 per-channel scaling 因子
    
    对每个输入通道 j，搜索最优的 s_j 使量化误差最小
    """
    in_features = weight.shape[1]
    best_error = float('inf')
    best_scale = torch.ones(in_features, device=weight.device)
    
    # 每个通道的激活平均值
    x_mean = activation_samples.abs().mean(dim=0)  # [in]
    
    # 网格搜索 alpha（s_j = x_mean_j^alpha）
    for alpha in torch.linspace(0, 1, n_grid):
        # 候选 scale
        scale = x_mean.pow(alpha)  # [in]
        scale = scale / scale.mean()  # 归一化（不改变整体大小）
        
        # 应用 scale 到权重
        W_scaled = weight / scale.unsqueeze(0)  # [out, in]
        
        # 量化 W_scaled
        W_q = quantize_weight(W_scaled, bits, group_size=128)
        
        # 量化误差（用校准数据的激活值加权）
        # 误差 = sum over samples of (X_scaled @ W.T - X_scaled @ W_q.T)^2
        # 等价于 (W - W_q * scale)X 的误差
        W_dq = W_q * scale.unsqueeze(0)
        
        # 用激活值加权
        error = ((weight - W_dq).pow(2) * x_mean.unsqueeze(0)).mean()
        
        if error < best_error:
            best_error = error
            best_scale = scale.clone()
    
    return best_scale

def quantize_weight(W: torch.Tensor, bits: int, group_size: int = 128):
    """简单的权重量化（RTN，per-group）"""
    out, inp = W.shape
    W_q = torch.zeros_like(W)
    
    for i in range(0, inp, group_size):
        group = W[:, i:i+group_size]
        scale = group.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / (2**(bits-1)-1)
        q = torch.clamp(torch.round(group / scale), -(2**(bits-1)), 2**(bits-1)-1)
        W_q[:, i:i+group_size] = q * scale
    
    return W_q


class AWQLinear(nn.Module):
    """
    AWQ 量化的线性层
    """
    def __init__(self, in_features, out_features, bits=4, group_size=128):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        
        # INT4 权重（存储为 int8，每个 int8 存 2 个 int4）
        self.register_buffer('qweight', torch.zeros(
            out_features, in_features // 2, dtype=torch.uint8
        ))
        # 量化参数
        self.register_buffer('scales', torch.zeros(
            out_features, in_features // group_size, dtype=torch.float16
        ))
        self.register_buffer('zeros', torch.zeros(
            out_features, in_features // group_size, dtype=torch.uint8
        ))
        # AWQ scale（输入通道缩放）
        self.register_buffer('awq_scale', torch.ones(in_features, dtype=torch.float16))
    
    def forward(self, x):
        # 应用 AWQ scale 到输入
        x = x * self.awq_scale  # [batch, seq, in_features]
        
        # 反量化权重并计算矩阵乘法
        # 实际使用 CUDA kernel（如 GEMM with pack/unpack）
        W_dq = self._dequantize()
        return x @ W_dq.T
    
    def _dequantize(self):
        """反量化 INT4 权重为 FP16"""
        # 解包 int4（从 int8 中提取）
        # ... 实际用 CUDA kernel 高效实现
        pass
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, awq_scale: torch.Tensor, 
                    bits=4, group_size=128):
        """从标准 Linear 层创建 AWQ 量化层"""
        m = cls(linear.in_features, linear.out_features, bits, group_size)
        
        # 应用 awq scale 到权重
        W_scaled = linear.weight / awq_scale.unsqueeze(0)
        
        # 量化 W_scaled
        # ... pack to int4
        
        m.awq_scale = awq_scale
        return m
```

## AWQ vs GPTQ 对比

| 方面 | GPTQ | AWQ |
|------|------|-----|
| 核心思路 | Hessian-based 误差补偿 | 激活感知 per-channel scaling |
| 计算复杂度 | 高（Cholesky 分解）| 低（grid search）|
| 量化时间（7B）| ~30 分钟 | ~5 分钟 |
| INT4 精度（PPL）| 5.58 | 5.52 |
| 硬件效率 | 需要特殊 dequant kernel | 更好的 INT4 kernel |
| 工具支持 | AutoGPTQ | AutoAWQ, llm-awq |

**AWQ 的优势**：更快的量化时间 + 略好的精度 + 更好的实际推理速度。

## 实战：使用 AutoAWQ

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path  = "Llama-2-7b-AWQ-4bit"

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"  # 或 "GEMV"（小 batch 更快）
}

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 量化（使用校准数据，约 5 分钟）
model.quantize(tokenizer, quant_config=quant_config)

# 保存
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# 推理（加载量化模型）
model_q = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
text = "Tell me about machine learning."
tokens = tokenizer(text, return_tensors="pt").input_ids.cuda()
out = model_q.generate(tokens, max_new_tokens=100)
print(tokenizer.decode(out[0]))
```

## 面试题

**Q: AWQ 的核心假设是什么？为什么这个假设成立？**

A: AWQ 的核心假设是：权重的重要性（对输出误差的影响）不是均匀的，只有约 1% 的"显著权重"是关键的，而这些显著权重恰好对应输入激活值中值较大的通道。这个假设成立的原因：在 Transformer 中，特定的 token（如"the"、标点）在特定通道上有异常大的激活值，这些激活值会放大对应权重的量化误差（误差 ≈ |x_j| × |Δw_j|）；而这些通道对模型输出的贡献也最大（|x_j| × |w_j| 大）。AWQ 通过 per-channel scaling 将这些"难量化"的重要通道的权重值变小，使量化误差更均匀，实验显示这样比直接保护 1% FP16 权重更有效且实现更简洁。
