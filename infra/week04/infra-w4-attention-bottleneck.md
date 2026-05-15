---
layout: default
title: "D1 · 标准 Attention 的 IO 瓶颈分析"
render_with_liquid: false
---

# D1 · 标准 Attention 的 IO 瓶颈分析

## 从数学定义出发

Self-Attention 的计算公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 $Q, K, V \in \mathbb{R}^{N \times d}$，$N$ 是序列长度，$d$ 是 head dimension。

标准实现分三步：
1. $S = QK^T / \sqrt{d}$，$S \in \mathbb{R}^{N \times N}$
2. $P = \text{softmax}(S)$，$P \in \mathbb{R}^{N \times N}$
3. $O = PV$，$O \in \mathbb{R}^{N \times d}$

**关键问题：步骤1产生了一个 N×N 的中间矩阵 S，必须写回 HBM。**

## IO 复杂度分析

现代 GPU 内存层次（以 A100 为例）：

| 存储层次 | 大小 | 带宽 |
|---------|------|------|
| 寄存器 (Register) | 256KB/SM | ~100 TB/s |
| 共享内存 (SRAM) | 192KB/SM | ~19 TB/s |
| L2 Cache | 40MB | ~5 TB/s |
| HBM (显存) | 80GB | 2 TB/s |
| PCIe (到CPU) | - | 32 GB/s |

标准 Attention 的 HBM IO 次数：

```
读取 Q, K: 2 × N × d × 2 bytes (FP16)
写入 S:     N × N × 2 bytes
读取 S:     N × N × 2 bytes  (softmax时再读一次)
写入 P:     N × N × 2 bytes
读取 P, V:  N × N × 2 + N × d × 2 bytes
写入 O:     N × d × 2 bytes

总计 HBM IO ≈ 4N² + 4Nd bytes
```

对于 N=4096, d=128（典型 LLM 配置）：

```
N×N 项: 4 × 4096² × 2 bytes = 134 MB
Nd 项:  4 × 4096 × 128 × 2 bytes = 4 MB

总 IO ≈ 138 MB per attention head
```

一个 GPT-3 175B 有 96 heads，每个 token 生成都需要做 96 次 attention...

## 用 Roofline 模型理解瓶颈

A100 的理论峰值：
- 计算峰值：312 TFLOPS (FP16)
- 内存带宽：2 TB/s

Arithmetic Intensity (AI) = FLOPs / Bytes

标准 Attention（N=4096, d=128）：
```
FLOPs ≈ 4N²d = 4 × 4096² × 128 = 8.6 TFLOP
Bytes ≈ 4N²  = 134 MB

AI ≈ 8.6T / 0.134G ≈ 64 FLOPs/Byte
```

A100 的 Ridge Point = 312 TFLOPS / 2 TB/s = 156 FLOPs/Byte

**结论：标准 Attention 的 AI ≈ 64，远低于 Ridge Point (156)，是严重的 Memory-Bound 操作。**

```python
# 用代码直观感受 IO 瓶颈
import torch
import time

def standard_attention(Q, K, V, scale):
    """标准实现：产生大量 HBM IO"""
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # HBM write: N×N
    P = torch.softmax(S, dim=-1)                       # HBM read + write: N×N
    O = torch.matmul(P, V)                             # HBM read: N×N
    return O

# 测量不同序列长度下的耗时
device = 'cuda'
batch, heads, d = 1, 32, 128
results = []

for N in [512, 1024, 2048, 4096, 8192]:
    Q = torch.randn(batch, heads, N, d, device=device, dtype=torch.float16)
    K = torch.randn(batch, heads, N, d, device=device, dtype=torch.float16)
    V = torch.randn(batch, heads, N, d, device=device, dtype=torch.float16)
    scale = d ** -0.5
    
    # warmup
    for _ in range(3):
        _ = standard_attention(Q, K, V, scale)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(10):
        out = standard_attention(Q, K, V, scale)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 10 * 1000  # ms
    
    flops = 4 * N * N * d * heads  # 近似
    mem_io = 4 * N * N * 2 * heads / 1e9  # GB
    results.append((N, elapsed, mem_io))
    print(f"N={N:5d}: {elapsed:.2f} ms, HBM IO ≈ {mem_io:.1f} GB")

# 典型输出：
# N=  512: 0.12 ms, HBM IO ≈ 0.1 GB
# N= 1024: 0.35 ms, HBM IO ≈ 0.4 GB
# N= 2048: 1.20 ms, HBM IO ≈ 1.7 GB
# N= 4096: 4.80 ms, HBM IO ≈ 6.7 GB
# N= 8192: 19.5 ms, HBM IO ≈ 26.8 GB
# 可以看到：耗时随 N² 增长，符合 Memory-Bound 特征
```

## 为什么 N×N 矩阵不能放 SRAM？

A100 每个 SM 有 192KB SRAM，整个 GPU 有 108 个 SM，总计约 20MB SRAM。

对于 N=4096, d=128 的单个 attention head：
- S 矩阵大小：4096 × 4096 × 2 bytes = 32 MB

**32 MB >> 20 MB（全部 SRAM），必须使用 HBM！**

这就是 FlashAttention 要解决的核心问题：**能不能不物化这个 N×N 矩阵？**

## 标准 Attention 的 Python 参考实现

```python
import torch
import math

def attention_forward(Q, K, V):
    """
    标准注意力前向传播
    Q, K, V: [batch, heads, seq_len, head_dim]
    """
    d_k = Q.shape[-1]
    scale = 1.0 / math.sqrt(d_k)
    
    # Step 1: S = QK^T / sqrt(d)
    # Shape: [batch, heads, seq_len, seq_len]
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Step 2: P = softmax(S)
    # 数值稳定版本：减去 max
    S_max = S.max(dim=-1, keepdim=True).values
    S_stable = S - S_max
    exp_S = torch.exp(S_stable)
    P = exp_S / exp_S.sum(dim=-1, keepdim=True)
    
    # Step 3: O = PV
    O = torch.matmul(P, V)
    
    return O

# 内存分析
def analyze_memory(N, d, dtype_bytes=2):
    """分析标准 Attention 的峰值内存占用"""
    QKV_mem = 3 * N * d * dtype_bytes
    S_mem   = N * N * dtype_bytes  # 这是瓶颈
    P_mem   = N * N * dtype_bytes
    O_mem   = N * d * dtype_bytes
    
    peak = QKV_mem + S_mem + P_mem + O_mem
    print(f"N={N}, d={d}")
    print(f"  Q+K+V: {QKV_mem/1024:.1f} KB")
    print(f"  S (N×N): {S_mem/1024/1024:.1f} MB  ← 瓶颈")
    print(f"  P (N×N): {P_mem/1024/1024:.1f} MB")
    print(f"  O:       {O_mem/1024:.1f} KB")
    print(f"  峰值总计: {peak/1024/1024:.1f} MB")

analyze_memory(4096, 128)
# 输出：
# N=4096, d=128
#   Q+K+V: 3072.0 KB
#   S (N×N): 32.0 MB  ← 瓶颈
#   P (N×N): 32.0 MB
#   O:       1024.0 KB
#   峰值总计: 67.0 MB
```

## 本节小结

| 问题 | 数据 |
|------|------|
| 标准 Attention 时间复杂度 | O(N²d) |
| 标准 Attention 空间复杂度 | O(N²)——存 S 和 P |
| N=4096 单 head HBM IO | ~134 MB |
| Arithmetic Intensity | ~64 FLOPs/Byte |
| A100 Ridge Point | 156 FLOPs/Byte |
| 结论 | **严重 Memory-Bound** |

**明天我们将学习如何用 Tiling + Online Softmax 打破这一瓶颈。**

## 面试题

**Q: 为什么 Transformer 在长序列上很慢？**

A: 核心原因是标准 Attention 产生 O(N²) 大小的中间矩阵（注意力得分矩阵），该矩阵无法放入 GPU 片上 SRAM，必须反复读写 HBM。以 A100 为例，HBM 带宽 2TB/s 远低于 SRAM 的 19TB/s，形成带宽瓶颈。N=8192 时单个 head 就需要 ~130MB 的 HBM 读写，严重限制吞吐量。
