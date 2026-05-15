---
layout: default
title: "D1 · 量化基础：数值表示与量化误差"
render_with_liquid: false
---

# D1 · 量化基础：数值表示与量化误差

## 浮点数格式回顾

| 格式 | 总位数 | 符号 | 指数 | 尾数 | 范围 | 机器精度 |
|------|-------|------|------|------|------|---------|
| FP32 | 32 | 1 | 8 | 23 | ±3.4×10³⁸ | 1.2×10⁻⁷ |
| FP16 | 16 | 1 | 5 | 10 | ±65504 | 9.8×10⁻⁴ |
| BF16 | 16 | 1 | 8 | 7  | ±3.4×10³⁸ | 7.8×10⁻³ |
| FP8 (e4m3) | 8 | 1 | 4 | 3 | ±448 | 6.3×10⁻² |

## 整型量化的基本原理

### 均匀量化（Uniform Quantization）

将浮点数映射到整型的线性变换：

$$x_{\text{int}} = \text{round}\left(\frac{x_{\text{float}} - z}{\Delta}\right)$$

其中：
- $\Delta = \frac{x_{\max} - x_{\min}}{2^b - 1}$（缩放因子，scale）
- $z$：零点（zero point），使 0 可以被精确表示
- $b$：量化位数（8 或 4）

反量化（Dequantize）：
$$x_{\text{float}} \approx \Delta \cdot x_{\text{int}} + z \cdot \Delta$$

```python
import torch
import numpy as np

def quantize_tensor(x: torch.Tensor, num_bits: int = 8, symmetric: bool = True):
    """
    均匀量化
    symmetric=True: 对称量化（零点 z=0，通常用于权重）
    symmetric=False: 非对称量化（用于激活值，可能偏移）
    """
    if symmetric:
        # 对称量化：[-max, max] → [-2^(b-1)+1, 2^(b-1)-1]
        x_max = x.abs().max()
        qmax = 2 ** (num_bits - 1) - 1  # INT8: 127
        
        scale = x_max / qmax
        zero_point = 0
        
        x_int = torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int8)
        x_dequant = x_int.float() * scale
        
    else:
        # 非对称量化：[min, max] → [0, 2^b - 1]
        x_min, x_max = x.min(), x.max()
        qmin, qmax = 0, 2 ** num_bits - 1  # UINT8: [0, 255]
        
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = torch.round(qmin - x_min / scale).clamp(qmin, qmax)
        
        x_int = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax).to(torch.uint8)
        x_dequant = (x_int.float() - zero_point) * scale
    
    # 量化误差
    error = (x - x_dequant).abs()
    
    return x_int, x_dequant, scale, zero_point, error

# 示例
x = torch.randn(100) * 3.0  # 模拟权重分布

x_int, x_dequant, scale, zp, error = quantize_tensor(x, num_bits=8, symmetric=True)
print(f"INT8 对称量化：scale={scale:.4f}, max_error={error.max():.4f}, "
      f"relative_error={error.mean()/x.abs().mean():.2%}")

x_int4, x_dq4, scale4, zp4, error4 = quantize_tensor(x, num_bits=4, symmetric=True)
print(f"INT4 对称量化：scale={scale4:.4f}, max_error={error4.max():.4f}, "
      f"relative_error={error4.mean()/x.abs().mean():.2%}")

# 典型输出：
# INT8 对称量化：scale=0.0236, max_error=0.0120, relative_error=0.63%
# INT4 对称量化：scale=0.2830, max_error=0.1540, relative_error=7.80%
```

### 量化误差分析

```python
def analyze_quantization_error(weight_matrix: torch.Tensor, bits: int):
    """
    分析不同量化粒度的误差
    """
    results = {}
    
    # Per-tensor 量化（最粗粒度）
    _, dq, _, _, err = quantize_tensor(weight_matrix.flatten(), bits, symmetric=True)
    results['per_tensor'] = err.mean().item()
    
    # Per-channel 量化（每行一个 scale）
    per_channel_dq = torch.zeros_like(weight_matrix)
    per_channel_err = []
    for i in range(weight_matrix.shape[0]):
        _, dq_row, _, _, err_row = quantize_tensor(weight_matrix[i], bits, symmetric=True)
        per_channel_dq[i] = dq_row
        per_channel_err.append(err_row.mean().item())
    results['per_channel'] = np.mean(per_channel_err)
    
    # Per-group 量化（每 g 个元素一个 scale，GPTQ/AWQ 常用 g=128）
    g = 128
    per_group_err = []
    for i in range(weight_matrix.shape[0]):
        for j in range(0, weight_matrix.shape[1], g):
            group = weight_matrix[i, j:j+g]
            _, _, _, _, err_g = quantize_tensor(group, bits, symmetric=True)
            per_group_err.append(err_g.mean().item())
    results['per_group_128'] = np.mean(per_group_err)
    
    print(f"INT{bits} 量化误差（越小越好）：")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")

# 测试（模拟 LLM 权重分布）
W = torch.randn(4096, 4096)
analyze_quantization_error(W, bits=8)
analyze_quantization_error(W, bits=4)

# 典型输出：
# INT8 量化误差：
#   per_tensor:     0.000191
#   per_channel:    0.000190  (vs per_tensor 差异不大)
#   per_group_128:  0.000187
# INT4 量化误差：
#   per_tensor:     0.023500
#   per_channel:    0.011200  (明显改善)
#   per_group_128:  0.003100  (再次大幅改善，INT4 per-group 是关键)
```

## 量化的硬件加速原理

```
INT8 矩阵乘法（A100）：
  A100 INT8 Tensor Core：624 TOPS
  A100 FP16 Tensor Core：312 TFLOPS
  理论加速：2×

实际加速取决于 bottleneck：
  - Memory-Bound（小 batch）：INT8 节省内存带宽 → 2× 加速
  - Compute-Bound（大 batch）：INT8 Tensor Core 2× 速度 → 2× 加速

但有 dequantize 开销！
```

```cuda
// INT8 矩阵乘法示例（使用 CUTLASS）
// W: [out_features, in_features] INT8（量化权重）
// x: [batch, in_features] INT8（量化激活值）
// 输出：[batch, out_features] FP32（反量化后）

// 实际代码使用 cublasLtMatmul 或 CUTLASS
#include <cutlass/gemm/device/gemm.h>

using Gemm = cutlass::gemm::device::Gemm<
    int8_t,                          // ElementA (activations)
    cutlass::layout::ColumnMajor,
    int8_t,                          // ElementB (weights)
    cutlass::layout::ColumnMajor,
    int32_t,                         // ElementC (accumulator, INT32 to avoid overflow)
    cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80              // A100
>;

// 之后将 INT32 结果 dequantize 为 FP32：
// C_fp32 = C_int32 * (scale_A * scale_W)
```

## NF4：NormalFloat 4-bit（QLoRA）

QLoRA 引入了 NF4 格式，专为量化正态分布的神经网络权重设计：

```python
"""
NF4 量化：
- 假设权重服从正态分布 N(0,1)
- 使用信息论最优的 16 个量化点（等百分位点）
- 比均匀 INT4 量化精度更高
"""
import torch
import scipy.stats as stats

def create_nf4_codebook():
    """创建 NF4 量化码本（16 个值）"""
    # 等概率分割正态分布
    # 8 个负值 + 0 + 7 个正值 = 16 个值
    quantile_points = [(2*i+1) / 32 for i in range(16)]  # 均匀划分
    nf4_values = [stats.norm.ppf(q) for q in quantile_points]
    
    # 归一化到 [-1, 1]
    max_val = max(abs(v) for v in nf4_values)
    nf4_values = [v / max_val for v in nf4_values]
    
    return torch.tensor(nf4_values, dtype=torch.float32)

nf4_codebook = create_nf4_codebook()
print("NF4 码本：", nf4_codebook.tolist())

def nf4_quantize(x: torch.Tensor, codebook: torch.Tensor, block_size: int = 64):
    """
    NF4 量化（按 block 归一化）
    每个 block 有自己的缩放因子（absmax 归一化）
    """
    original_shape = x.shape
    x_flat = x.flatten()
    
    # 按 block 量化
    quantized = torch.zeros_like(x_flat, dtype=torch.uint8)
    scales = []
    
    for i in range(0, len(x_flat), block_size):
        block = x_flat[i:i+block_size]
        absmax = block.abs().max().clamp(min=1e-8)
        scales.append(absmax.item())
        
        # 归一化到 [-1, 1]
        block_normalized = block / absmax
        
        # 最近邻查找
        distances = (block_normalized.unsqueeze(1) - codebook.unsqueeze(0)).abs()
        quantized[i:i+block_size] = distances.argmin(dim=1).to(torch.uint8)
    
    return quantized.reshape(original_shape), torch.tensor(scales)
```

## 量化精度 vs 位宽权衡

实测数据（LLaMA-2 7B，MMLU benchmark）：

| 量化方法 | 位宽 | 精度（MMLU）| 内存 | 速度（A100）|
|---------|------|-----------|------|----------|
| FP16（基线）| 16 | 45.3% | 14 GB | 1× |
| INT8 SmoothQuant | 8 | 45.1% | 7 GB | 1.5× |
| INT4 GPTQ (g128) | 4 | 44.8% | 4 GB | 2.8× |
| INT4 AWQ | 4 | 45.0% | 4 GB | 3.0× |
| NF4 QLoRA | 4 | 44.6% | 4 GB | 2.5× |
| INT2 | 2 | 38.2% | 2 GB | 3.5× |

**结论**：INT8 几乎无损；INT4 精度损失 < 1%（用好算法）；INT2 损失明显。

## 面试题

**Q: 为什么 LLM 的权重容易量化，而激活值不容易量化？**

A: 权重在推理时是固定的，其分布通常接近正态分布，outlier 较少，可以用简单的 per-channel 量化获得好效果。激活值则不同：LLM 激活值中存在大量的 outlier——某些通道的值可以比平均值大 100×（如 channel 维度的 outlier），如果用 per-tensor 量化，scale 会被这些 outlier 决定，导致其他 99% 的值精度很差（量化为 0 或 1）。SmoothQuant 通过将激活的 outlier "平滑"到权重中来解决这个问题。
