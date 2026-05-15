---
layout: default
title: "D3 · FlashAttention v1 CUDA 实现详解"
render_with_liquid: false
---

# D3 · FlashAttention v1 CUDA 实现详解

## FlashAttention v1 算法回顾

Tri Dao 等人在 2022 年提出 FlashAttention（[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)），核心贡献：
1. **IO-Aware** 算法：减少 HBM 读写次数
2. **Tiling**：将 Q, K, V 分块放入 SRAM
3. **Recomputation**：反向传播时重算，不存中间激活

### 前向传播伪代码

```
Algorithm 1: FlashAttention Forward Pass
输入：Q, K, V ∈ R^{N×d}，存于 HBM
输出：O ∈ R^{N×d}，存于 HBM

1. 设块大小 Bc = ⌈M/4d⌉, Br = min(⌈M/4d⌉, d)
2. 初始化 O = 0, ℓ = 0, m = -∞ ∈ R^N，存于 HBM
3. 将 Q 分成 Tr 块 Q1..QTr，每块大小 Br×d
   将 K,V 分成 Tc 块 K1..KTc, V1..VTc，每块大小 Bc×d
4. for j = 1 to Tc:          # 外层：遍历 K,V 块
     从 HBM 加载 Kj, Vj 到 SRAM
     for i = 1 to Tr:        # 内层：遍历 Q 块
       从 HBM 加载 Qi, Oi, ℓi, mi 到 SRAM
       计算 Sij = Qi × Kj^T ∈ R^{Br×Bc}（SRAM内）
       计算 mij = rowmax(Sij)
       Pij = exp(Sij - mij)
       ℓij = rowsum(Pij)
       # Online softmax 更新
       mi_new = max(mi, mij)
       ℓi_new = e^{mi-mi_new} × ℓi + e^{mij-mi_new} × ℓij
       Oi = diag(e^{mi-mi_new}/ℓi_new × ℓi)^{-1} × Oi + Pij × Vj / ℓi_new
       # 等价于: Oi_new = (ℓi × e^{mi-mi_new} × Oi + Pij×e^{mij-mi_new} × Vj) / ℓi_new
       将 Oi, ℓi_new, mi_new 写回 HBM
5. 返回 O
```

注意：在 v1 中，**外层是 K,V 块，内层是 Q 块**。这导致每个 Q 块需要反复从 HBM 读取，在 v2 中得到改进。

## CUDA Kernel 实现

```cuda
// flash_attention_fwd.cu
// 简化版 FlashAttention 前向 CUDA Kernel
// 每个 Block 处理 Q 的一行（一个 query）

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 64  // Bc = Br = 64

__global__ void flash_attention_forward_kernel(
    const half* __restrict__ Q,    // [N, d]
    const half* __restrict__ K,    // [N, d]
    const half* __restrict__ V,    // [N, d]
    half* __restrict__ O,          // [N, d]
    float* __restrict__ L,         // [N] 归一化因子（用于反向）
    int N, int d, float scale
) {
    // 每个 Block 负责一个 query 位置 i
    int i = blockIdx.x;
    int tid = threadIdx.x;  // 0..BLOCK_SIZE-1
    
    // 分配 SRAM
    extern __shared__ float smem[];
    float* Qi   = smem;                       // [d]
    float* Kj   = smem + d;                   // [BLOCK_SIZE, d]
    float* Vj   = smem + d + BLOCK_SIZE * d;  // [BLOCK_SIZE, d]
    float* Sij  = smem + d * (1 + 2*BLOCK_SIZE); // [BLOCK_SIZE]
    
    // 加载 Qi 到 SRAM（一次，后续不再读 HBM）
    for (int k = tid; k < d; k += blockDim.x) {
        Qi[k] = __half2float(Q[i * d + k]);
    }
    __syncthreads();
    
    // 初始化累加状态
    float mi = -INFINITY;    // 当前最大值
    float li = 0.0f;         // 归一化因子
    float Oi[128] = {0.0f};  // 输出累积（假设 d <= 128）
    
    // 遍历 K, V 块
    for (int j = 0; j < N; j += BLOCK_SIZE) {
        int block_len = min(BLOCK_SIZE, N - j);
        
        // 加载 Kj, Vj 到 SRAM
        for (int bk = tid; bk < block_len * d; bk += blockDim.x) {
            int row = bk / d, col = bk % d;
            Kj[row * d + col] = __half2float(K[(j + row) * d + col]);
            Vj[row * d + col] = __half2float(V[(j + row) * d + col]);
        }
        __syncthreads();
        
        // 计算 Sij = Qi × Kj^T（SRAM 内，每个线程计算一个元素）
        if (tid < block_len) {
            float s = 0.0f;
            for (int k = 0; k < d; k++) {
                s += Qi[k] * Kj[tid * d + k];
            }
            Sij[tid] = s * scale;
        }
        __syncthreads();
        
        // Warp-level max reduction
        float mij = -INFINITY;
        for (int k = 0; k < block_len; k++) {
            mij = fmaxf(mij, Sij[k]);
        }
        
        // 计算 exp(Sij - mij) 和 ℓij
        float lij = 0.0f;
        float Pij[BLOCK_SIZE];
        for (int k = 0; k < block_len; k++) {
            Pij[k] = expf(Sij[k] - mij);
            lij += Pij[k];
        }
        
        // Online softmax 更新
        float mi_new = fmaxf(mi, mij);
        float alpha = expf(mi - mi_new);     // 旧累积的修正因子
        float beta  = expf(mij - mi_new);    // 新块的修正因子
        float li_new = alpha * li + beta * lij;
        
        // 更新 Oi
        // Oi_new = (alpha * li / li_new) * Oi + (beta / li_new) * Pij @ Vj
        float scale_old = alpha * li / li_new;
        float scale_new = beta / li_new;
        
        for (int k = 0; k < d; k++) {
            float pv = 0.0f;
            for (int b = 0; b < block_len; b++) {
                pv += Pij[b] * Vj[b * d + k];
            }
            Oi[k] = scale_old * Oi[k] + scale_new * pv;
        }
        
        mi = mi_new;
        li = li_new;
        __syncthreads();
    }
    
    // 写回 HBM
    for (int k = tid; k < d; k += blockDim.x) {
        O[i * d + k] = __float2half(Oi[k]);
    }
    if (tid == 0) {
        L[i] = mi + logf(li);  // log-sum-exp，用于反向传播
    }
}

// 调用接口
void flash_attention_forward(
    const half* Q, const half* K, const half* V,
    half* O, float* L,
    int N, int d, float scale,
    cudaStream_t stream
) {
    // 每个 Query 位置一个 Block
    dim3 grid(N);
    dim3 block(BLOCK_SIZE);
    
    // SRAM: Qi(d) + Kj(Bc*d) + Vj(Bc*d) + Sij(Bc)
    size_t smem_size = (d + 2 * BLOCK_SIZE * d + BLOCK_SIZE) * sizeof(float);
    
    flash_attention_forward_kernel<<<grid, block, smem_size, stream>>>(
        Q, K, V, O, L, N, d, scale
    );
}
```

## 反向传播：Recomputation

FlashAttention 反向传播的关键：**不保存 P（N×N 矩阵），而是在反向时重算**。

只需保存前向的 $(m_i, \ell_i)$（即 $L_i = m_i + \log \ell_i$），大小 $O(N)$。

```cuda
// 反向传播伪代码（概念版）
__global__ void flash_attention_backward_kernel(
    const half* Q, const half* K, const half* V,
    const half* O, const float* L,    // 前向保存的 L
    const half* dO,                   // 输出梯度
    half* dQ, half* dK, half* dV,
    int N, int d, float scale
) {
    // 1. 计算 D = rowsum(O ∘ dO)（点积，标量）
    // 2. 遍历 K,V 块（外层）和 Q 块（内层）
    //    - 重算 Sij（利用已有 Q, K）
    //    - 重算 Pij（利用 Sij 和 L）
    //    - 计算梯度 dSij, dQ, dK, dV
    // 关键：Pij 通过 Sij 和 L[i] 重算，不需要存储：
    //   Pij[k] = exp(Sij[k] - L[i])
}
```

## 性能数据（来自论文）

A100 SXM4 80GB，PyTorch 1.12 基线：

| 序列长度 | 标准 Attention 速度 | FlashAttention v1 速度 | 加速比 |
|---------|--------------------|-----------------------|-------|
| 512 | 135 TFLOPs/s | 148 TFLOPs/s | 1.1× |
| 1024 | 98 TFLOPs/s | 145 TFLOPs/s | 1.5× |
| 2048 | 48 TFLOPs/s | 117 TFLOPs/s | 2.4× |
| 4096 | 20 TFLOPs/s | 73 TFLOPs/s | 3.6× |

**内存占用（batch=1, heads=12, N=4096, d=64）：**
- 标准 Attention：~1.3 GB
- FlashAttention v1：~256 MB（节省 5.1×）

## v1 的局限性

1. **外层遍历 K,V，内层遍历 Q**：导致 Q 块需要反复从 HBM 读取（每个 Q 块被读 Tc 次），v2 解决了这个问题
2. **只有前向显著加速**：反向需要 recomputation，FLOPs 增加 ~33%
3. **块大小固定**：没有针对不同 GPU 架构优化
4. **未利用 warp 分组**：warp 间通信未充分优化

## 实战：使用 FlashAttention

```python
# 方法 1：使用 PyTorch 2.0+ 的 SDPA（自动选择 FlashAttention 后端）
import torch
import torch.nn.functional as F

# PyTorch 会自动调用 FlashAttention（当 dtype=float16/bfloat16）
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    out = F.scaled_dot_product_attention(Q, K, V)

# 方法 2：使用 flash_attn 库
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# QKV packed 格式
out = flash_attn_qkvpacked_func(
    qkv,           # [batch, seqlen, 3, nheads, headdim]
    dropout_p=0.0,
    softmax_scale=None,  # 默认 1/sqrt(d)
    causal=True    # 因果 attention（decoder）
)

# 分离格式
out = flash_attn_func(
    q, k, v,       # [batch, seqlen, nheads, headdim]
    dropout_p=0.0,
    causal=True
)

# 检查是否在用 FlashAttention
print(torch.backends.cuda.flash_sdp_enabled())  # True
```

## 本节小结

FlashAttention v1 的核心创新：
1. **Tiling**：将计算分块到 SRAM，避免 N×N 矩阵写入 HBM
2. **Online Softmax**：分块时维护 (m, ℓ) 保证数值等价
3. **Recomputation**：反向传播重算 P，以 FLOPs 换内存

**主要限制**：外 K 内 Q 的遍历顺序导致 Q 反复读 HBM，v2 通过颠倒顺序解决。

## 面试题

**Q: FlashAttention 的反向传播为什么需要 Recomputation？存什么、不存什么？**

A: 前向传播不保存 P（注意力权重矩阵，大小 N×N），只保存 L = m + log(ℓ)（大小 N）。反向时需要 P 来计算梯度，因此重新用 Q, K 计算 Sij，再用 L[i] 还原 Pij = exp(Sij - L[i])。这样内存从 O(N²) 降至 O(N)，代价是约 33% 的额外 FLOPs。
