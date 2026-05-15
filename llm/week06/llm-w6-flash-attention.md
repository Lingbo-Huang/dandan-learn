---
layout: default
title: "D5 · Flash Attention"
render_with_liquid: false
---

# D5 · Flash Attention：IO感知的注意力加速

> **标准 Attention 的瓶颈不是计算，而是内存带宽。** Flash Attention 通过分块计算和在线 softmax，将 HBM 访问量从 O(N²) 降到 O(N)，大幅提升吞吐。

---

## 一、标准 Attention 的问题

### 1.1 内存访问分析

```python
"""
标准 Attention 的内存访问路径：

1. 从 HBM 读 Q, K, V         →  O(N·d) 次读
2. 计算 S = QK^T，写 S 到 HBM →  O(N²) 次写  ← 瓶颈！
3. 读 S，计算 softmax(S)，写回 →  O(N²) 次读写
4. 计算 O = softmax(S)·V，写回 →  O(N²) 次读写

总 HBM 访问量：O(N²)
而 GPU 的 HBM 带宽（A100: 2TB/s）远低于 SRAM（~20TB/s）

以 N=4096, d=64, float16 为例：
  S 矩阵大小：4096 × 4096 × 2 bytes = 32 MB
  每次 forward 需要反复读写这 32 MB 到 HBM ← 慢！
"""

import torch
import time

def standard_attention(Q, K, V):
    """标准 attention，中间矩阵 S 写入 HBM"""
    N, d = Q.shape[-2], Q.shape[-1]
    scale = d ** -0.5
    
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # 写 S 到 HBM
    P = torch.softmax(S, dim=-1)                       # 读写 HBM
    O = torch.matmul(P, V)                             # 读写 HBM
    return O

# 基准测试
N, d = 4096, 64
Q = torch.randn(1, N, d, device='cuda', dtype=torch.float16)
K = torch.randn(1, N, d, device='cuda', dtype=torch.float16)
V = torch.randn(1, N, d, device='cuda', dtype=torch.float16)

t0 = time.time()
for _ in range(100):
    O = standard_attention(Q, K, V)
torch.cuda.synchronize()
print(f"标准 Attention: {(time.time()-t0)*10:.2f}ms/iter")
```

### 1.2 Roofline 分析

```
GPU 性能模型（Roofline）：
  计算上限：peak FLOPS（A100 = 312 TFLOPS for FP16）
  带宽上限：peak bandwidth（A100 = 2 TB/s）

标准 Attention：
  计算量：O(N² · d)  FLOPs
  内存访问：O(N²)   Bytes（中间矩阵 S, P）

算术强度（FLOP/Byte）= d
  当 d=64，算术强度=64 FLOP/Byte → 受内存带宽限制（memory-bound）
  A100 算术强度阈值 ≈ 312T/2T = 156 FLOP/Byte

结论：标准 Attention 是 memory-bound，Flash Attention 的目标是减少内存访问
```

---

## 二、Flash Attention 原理

### 2.1 核心思想：分块 + 在线 softmax

```python
"""
Flash Attention 的关键创新：

1. 分块（Tiling）：将 Q, K, V 分成小块，每块完全在 SRAM 中计算
2. 在线 softmax：不需要一次性看完整个序列，可以增量更新 softmax
3. 重计算（Recomputation）：反向传播时重新计算注意力，不存储中间矩阵

在线 softmax 算法（数值稳定版）：
  对于序列 x_1, x_2, ..., x_N，在线计算 softmax：
  m_i = max(m_{i-1}, x_i)     ← 当前最大值（用于数值稳定）
  d_i = d_{i-1} · exp(m_{i-1} - m_i) + exp(x_i - m_i)  ← 累积分母
  最终：softmax(x_i) = exp(x_i - m_N) / d_N
"""

def online_softmax_demo():
    """演示在线 softmax，无需存储完整序列"""
    import math
    
    x = [1.0, 3.0, 2.0, 4.0, 1.5]
    
    # 在线计算
    m = float('-inf')  # 全局最大值
    d = 0.0            # 分母累积
    
    for xi in x:
        m_new = max(m, xi)
        d = d * math.exp(m - m_new) + math.exp(xi - m_new)
        m = m_new
    
    softmax = [math.exp(xi - m) / d for xi in x]
    print(f"在线 softmax：{[f'{v:.4f}' for v in softmax]}")
    
    # 验证
    import torch
    ref = torch.softmax(torch.tensor(x), dim=0)
    print(f"参考结果：  {[f'{v:.4f}' for v in ref.tolist()]}")

online_softmax_demo()
```

### 2.2 Flash Attention 分块计算

```python
def flash_attention_numpy(Q, K, V, block_size=64):
    """
    Flash Attention 的 NumPy 实现（教学用，实际用 CUDA kernel）
    
    避免将 N×N 的 S 矩阵写到 HBM，
    改为分块在 SRAM 中计算，只把 O 写回 HBM
    """
    import numpy as np
    
    N, d = Q.shape
    scale = d ** -0.5
    O = np.zeros_like(Q)  # 输出
    
    # 按块遍历 Q
    for i_start in range(0, N, block_size):
        i_end = min(i_start + block_size, N)
        Qi = Q[i_start:i_end]  # (block, d)
        
        # 在线 softmax 状态
        m_i = np.full((i_end - i_start,), float('-inf'))  # 最大值
        l_i = np.zeros(i_end - i_start)                    # 分母
        O_i = np.zeros((i_end - i_start, d))               # 累积输出
        
        # 按块遍历 K, V
        for j_start in range(0, N, block_size):
            j_end = min(j_start + block_size, N)
            Kj = K[j_start:j_end]  # (block, d)
            Vj = V[j_start:j_end]  # (block, d)
            
            # 在 SRAM 中计算这个小块的注意力分数
            Sij = Qi @ Kj.T * scale  # (block_q, block_k)
            
            # 更新在线 softmax 状态
            m_new = np.maximum(m_i, Sij.max(axis=1))
            
            # 更新分母（注意要 rescale 历史值）
            l_new = (
                np.exp(m_i - m_new) * l_i +
                np.exp(Sij - m_new[:, None]).sum(axis=1)
            )
            
            # 更新输出（rescale 历史 + 加新贡献）
            O_i = (
                np.exp(m_i - m_new)[:, None] * O_i +
                np.exp(Sij - m_new[:, None]) @ Vj
            )
            
            m_i = m_new
            l_i = l_new
        
        # 归一化（除以分母）
        O[i_start:i_end] = O_i / l_i[:, None]
    
    return O

# 验证正确性
import numpy as np
N, d = 128, 64
Q = np.random.randn(N, d).astype(np.float32)
K = np.random.randn(N, d).astype(np.float32)
V = np.random.randn(N, d).astype(np.float32)

# Flash Attention
O_flash = flash_attention_numpy(Q, K, V, block_size=32)

# 标准 Attention
scale = d ** -0.5
S = Q @ K.T * scale
P = np.exp(S - S.max(axis=-1, keepdims=True))
P /= P.sum(axis=-1, keepdims=True)
O_std = P @ V

print(f"最大误差：{np.abs(O_flash - O_std).max():.6f}")  # 应该很小
```

---

## 三、使用 Flash Attention

### 3.1 PyTorch 内置（2.0+）

```python
import torch
import torch.nn.functional as F

# PyTorch 2.0+ 自动使用 Flash Attention
# 使用 scaled_dot_product_attention
def efficient_attention(Q, K, V, mask=None):
    """
    PyTorch 2.0 会自动选择最优实现：
    - 如果 GPU 支持且满足条件 → Flash Attention（CUDA kernel）
    - 否则 → 标准 attention
    """
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=True,
    ):
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

# 在模型中使用
class FlashAttentionLayer(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = torch.nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, N, D = x.shape
        QKV = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        QKV = QKV.permute(2, 0, 3, 1, 4)
        Q, K, V = QKV.unbind(0)
        
        # Flash Attention（PyTorch 2.0+）
        O = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        
        O = O.transpose(1, 2).reshape(B, N, D)
        return self.out(O)
```

### 3.2 Flash Attention 2（flash-attn 库）

```python
# pip install flash-attn --no-build-isolation

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# 方式 1：QKV 打包（训练常用）
def flash_attn_packed(qkv, causal=True):
    """
    qkv: (B, N, 3, H, D) - Q/K/V 打包
    causal: 是否使用因果掩码（自回归生成必须=True）
    """
    return flash_attn_qkvpacked_func(qkv, causal=causal)

# 方式 2：Q, K, V 分开
def flash_attn_separate(Q, K, V, causal=True):
    """
    Q, K, V: (B, N, H, D) 格式
    注意：flash_attn 要求 head-last 格式，与 PyTorch 的 head-first 不同
    """
    return flash_attn_func(Q, K, V, causal=causal)

# 性能对比（A100 上，序列长度 4096）
"""
Flash Attention 2 vs 标准 Attention：
  序列长 2K：速度 ~2x，显存节省 ~5x
  序列长 4K：速度 ~4x，显存节省 ~10x
  序列长 8K：速度 ~6x（标准已OOM）
  
Flash Attention 2 的改进（相比 v1）：
  - 减少非矩阵乘法运算（softmax 中的 exp）
  - 更好的 warp 分工（减少通信开销）
  - 速度提升约 2x
"""
```

---

## 四、Flash Attention 的局限与变体

```python
"""
Flash Attention 的使用条件（PyTorch sdpa）：
  ✅ head_dim ≤ 128
  ✅ dtype = float16 或 bfloat16
  ✅ 无自定义注意力偏置（如 ALiBi 需要特殊处理）
  ❌ 不支持 float32（训练时用 bf16）
  ❌ 不支持所有掩码类型

变体：
  Flash Attention 3 (2024)：针对 H100 Tensor Core 优化，速度再 2x
  Ring Attention：分布式长序列（多 GPU 分块）
  Sliding Window Attention（SWA）：Mistral 使用，限制注意力窗口大小
"""

# 检查当前 attention 实现
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    Q = torch.randn(2, 8, 1024, 64, device=device, dtype=torch.bfloat16)
    K = torch.randn(2, 8, 1024, 64, device=device, dtype=torch.bfloat16)
    V = torch.randn(2, 8, 1024, 64, device=device, dtype=torch.bfloat16)
    
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
        try:
            O = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
            print("Flash Attention 可用 ✅")
        except RuntimeError as e:
            print(f"Flash Attention 不可用: {e}")
```

---

## 五、面试题精讲

**Q: Flash Attention 的核心思想是什么？为什么能加速？**

A: Flash Attention 的核心是**IO感知（IO-aware）**：
1. 标准 Attention 需要将 N×N 的 S 矩阵写到 HBM，然后再读回来做 softmax，HBM 访问量 O(N²)
2. Flash Attention 利用**在线 softmax**技术，分块在 SRAM（快速片上内存）中计算，只需将最终输出 O 写到 HBM，HBM 访问量降到 O(N)
3. 代价是**反向传播时重新计算**注意力（而不是存储 S 矩阵），以计算换显存

**Q: Flash Attention 能减少显存占用多少？**

A: 标准 Attention 需要存 S（N×N float16）≈ 2N² bytes。对 N=4096，d=64：
- 标准：存 S = 32MB，额外显存 O(N²)
- Flash Attention：只存 O = N×d × 2 bytes = 512KB，显存 O(N)
- 节省比例：~64x（对 N=4096，d=64）

---

## 小结

```
Flash Attention 关键点：

核心创新：
  在线 softmax（无需物化完整 S 矩阵）
  分块计算（数据完全在 SRAM）
  反向重计算（节省显存）

效果：
  速度：2-6x（序列越长越明显）
  显存：O(N) vs O(N²)

使用：
  PyTorch 2.0+: F.scaled_dot_product_attention（自动）
  flash-attn 库：更完整支持，速度更快
  
适用条件：bf16/fp16，head_dim≤128
```
