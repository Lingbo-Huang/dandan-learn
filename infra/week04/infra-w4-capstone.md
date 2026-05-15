---
layout: default
title: "Week 4 Capstone · 从零实现简化版 FlashAttention"
render_with_liquid: false
---

# Week 4 Capstone · 从零实现简化版 FlashAttention

## 本节目标

综合本周所学，实现一个完整的简化版 FlashAttention，包括：
1. Python 参考实现（验证正确性）
2. Triton 实现（可实际运行，接近生产质量）
3. 性能基准测试
4. 面试系统设计题解答

## Part 1：Python 参考实现

```python
"""
flash_attention_reference.py
FlashAttention 的纯 Python 参考实现，用于理解算法
"""
import torch
import math

def flash_attention_reference(Q, K, V, causal=False, block_size=64):
    """
    FlashAttention v2 风格的参考实现
    Q, K, V: [batch, heads, seqlen, head_dim]
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    output = torch.zeros_like(Q)
    
    # 处理每个 batch 和 head（实际 CUDA 中这些是并行的）
    for b in range(B):
        for h in range(H):
            q = Q[b, h]  # [N, d]
            k = K[b, h]  # [N, d]
            v = V[b, h]  # [N, d]
            
            O = torch.zeros(N, d)
            m = torch.full((N,), float('-inf'))   # 行最大值
            ell = torch.zeros(N)                   # 归一化因子
            
            # 外层：遍历 Q 块（v2 风格）
            for i in range(0, N, block_size):
                qi = q[i:i+block_size]          # [Br, d] 加载到"SRAM"
                Oi = torch.zeros(len(qi), d)     # 初始化输出块
                mi = torch.full((len(qi),), float('-inf'))
                li = torch.zeros(len(qi))
                
                # 内层：遍历 KV 块
                for j in range(0, N, block_size):
                    if causal and j > i + block_size - 1:
                        break  # 因果 mask：跳过未来的 KV
                    
                    kj = k[j:j+block_size]  # [Bc, d]
                    vj = v[j:j+block_size]  # [Bc, d]
                    
                    # 计算 attention score
                    Sij = qi @ kj.T * scale  # [Br, Bc]
                    
                    # Causal mask
                    if causal:
                        mask = torch.ones(len(qi), len(kj), dtype=torch.bool)
                        for ii in range(len(qi)):
                            for jj in range(len(kj)):
                                if i + ii < j + jj:
                                    mask[ii, jj] = False
                        Sij = Sij.masked_fill(~mask, float('-inf'))
                    
                    # Online softmax 更新
                    mij = Sij.max(dim=1).values             # [Br]
                    Pij = torch.exp(Sij - mij.unsqueeze(1)) # [Br, Bc]
                    lij = Pij.sum(dim=1)                     # [Br]
                    
                    mi_new = torch.maximum(mi, mij)
                    alpha = torch.exp(mi - mi_new)  # 旧状态修正因子
                    beta  = torch.exp(mij - mi_new) # 新块修正因子
                    
                    li_new = alpha * li + beta * lij
                    
                    # 更新 Oi（延迟归一化，v2 风格）
                    Oi = Oi * (alpha * li / li_new).unsqueeze(1) + \
                         (beta.unsqueeze(1) * Pij / li_new.unsqueeze(1)) @ vj
                    
                    mi = mi_new
                    li = li_new
                
                output[b, h, i:i+block_size] = Oi
            
    return output


def verify_correctness():
    """验证 FlashAttention 参考实现的正确性"""
    torch.manual_seed(42)
    B, H, N, d = 2, 4, 256, 64
    
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    
    # 参考实现
    out_flash = flash_attention_reference(Q, K, V, causal=False)
    
    # PyTorch 标准实现
    out_std = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    
    max_diff = (out_flash - out_std).abs().max().item()
    print(f"Non-causal max diff: {max_diff:.6f} (should be < 1e-4)")
    assert max_diff < 1e-4, "正确性验证失败！"
    
    # Causal
    out_flash_causal = flash_attention_reference(Q, K, V, causal=True)
    out_std_causal = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=True
    )
    max_diff_causal = (out_flash_causal - out_std_causal).abs().max().item()
    print(f"Causal max diff: {max_diff_causal:.6f} (should be < 1e-4)")
    
    print("✅ 正确性验证通过！")

verify_correctness()
```

## Part 2：Triton 实现

Triton 是比 CUDA 更易用的 GPU 编程框架，适合快速实现自定义 kernel：

```python
"""
flash_attention_triton.py
使用 Triton 实现 FlashAttention v2 前向传播
"""
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, d, scale,
    BLOCK_M: tl.constexpr,  # Q 块大小（行）
    BLOCK_N: tl.constexpr,  # KV 块大小（列）
    HEAD_DIM: tl.constexpr, # head dimension
):
    """
    每个 Triton Program 处理一个 (batch, head, Q块) 的组合
    """
    # Program ID
    start_m = tl.program_id(0)  # Q 块的索引
    off_b   = tl.program_id(1)  # batch 索引
    off_h   = tl.program_id(2)  # head 索引
    
    # 计算当前 Q 块的起始行
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, HEAD_DIM)                      # [HEAD_DIM]
    offs_n = tl.arange(0, BLOCK_N)                       # [BLOCK_N]
    
    # Q 的指针
    Q_base = Q_ptr + off_b * stride_qb + off_h * stride_qh
    q_ptrs = Q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    
    # 加载 Q 块到寄存器（SRAM）
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)  # [BLOCK_M, HEAD_DIM]
    
    # 初始化 online softmax 状态
    m_i = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)  # 当前最大值
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                       # 归一化因子
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)             # 输出累积
    
    # 遍历 KV 块
    K_base = K_ptr + off_b * stride_kb + off_h * stride_kh
    V_base = V_ptr + off_b * stride_vb + off_h * stride_vh
    
    for start_n in range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)  # 告诉编译器对齐
        
        # 加载 K 块
        k_ptrs = K_base + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < N, other=0.0)  # [BLOCK_N, HEAD_DIM]
        
        # 计算 QK^T
        qk = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]
        qk = qk * scale
        
        # Online softmax 更新
        m_ij = tl.max(qk, axis=1)                    # [BLOCK_M] 本块最大值
        p = tl.exp(qk - m_ij[:, None])               # [BLOCK_M, BLOCK_N]
        l_ij = tl.sum(p, axis=1)                     # [BLOCK_M] 本块 ℓ
        
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)                # 旧状态修正因子
        beta  = tl.exp(m_ij - m_i_new)               # 新块修正因子
        
        l_i_new = alpha * l_i + beta * l_ij
        
        # 加载 V 块并更新累积输出
        v_ptrs = V_base + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < N, other=0.0)  # [BLOCK_N, HEAD_DIM]
        
        # acc_new = (alpha * l_i / l_i_new) * acc + (beta / l_i_new) * p @ v
        acc_scale = alpha * l_i / l_i_new
        acc = acc * acc_scale[:, None]
        acc += beta[:, None] / l_i_new[:, None] * tl.dot(p.to(tl.float16), v)
        
        m_i = m_i_new
        l_i = l_i_new
    
    # 写入输出
    O_base = O_ptr + off_b * stride_ob + off_h * stride_oh
    o_ptrs = O_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N)
    
    # 保存 L = m + log(ℓ)（反向传播用）
    if L_ptr is not None:
        L_base = L_ptr + off_b * (tl.num_programs(2) * N) + off_h * N
        l_ptrs = L_base + offs_m
        tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < N)


def flash_attention_triton(Q, K, V, causal=False):
    """
    Triton FlashAttention 调用接口
    Q, K, V: [batch, heads, seqlen, head_dim]
    """
    B, H, N, d = Q.shape
    assert d in [16, 32, 64, 128], f"head_dim must be power of 2, got {d}"
    
    BLOCK_M = 64  # Q 块大小
    BLOCK_N = 64  # KV 块大小
    
    O = torch.empty_like(Q)
    L = torch.empty(B, H, N, device=Q.device, dtype=torch.float32)
    
    scale = 1.0 / math.sqrt(d)
    
    # Grid: (num_Q_blocks, batch, heads)
    grid = (triton.cdiv(N, BLOCK_M), B, H)
    
    flash_attention_fwd_kernel[grid](
        Q, K, V, O, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        N, d, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=d,
    )
    
    return O
```

## Part 3：性能基准测试

```python
"""
benchmark.py
对比不同 Attention 实现的性能
"""
import torch
import time
import math

def benchmark(fn, warmup=10, reps=100):
    """精确测量 CUDA kernel 耗时"""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    # 使用 CUDA events 精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(reps):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event) / reps  # ms

def run_benchmark():
    device = torch.device('cuda')
    dtype  = torch.float16
    
    configs = [
        (1, 32, 1024, 64),
        (1, 32, 2048, 64),
        (1, 32, 4096, 128),
        (4, 32, 4096, 128),
        (1, 32, 8192, 128),
    ]
    
    print(f"{'Config':<35} {'PyTorch SDPA':>12} {'FlashAttn':>12} {'Speedup':>8}")
    print("-" * 70)
    
    for (B, H, N, d) in configs:
        Q = torch.randn(B, H, N, d, device=device, dtype=dtype)
        K = torch.randn(B, H, N, d, device=device, dtype=dtype)
        V = torch.randn(B, H, N, d, device=device, dtype=dtype)
        
        # PyTorch SDPA
        t_std = benchmark(lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V))
        
        # FlashAttention（通过 sdp_kernel 强制使用）
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            t_flash = benchmark(lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V))
        
        speedup = t_std / t_flash
        config_str = f"B={B} H={H} N={N} d={d}"
        print(f"{config_str:<35} {t_std:>10.2f}ms {t_flash:>10.2f}ms {speedup:>7.2f}x")

run_benchmark()
```

## Part 4：面试系统设计题

### 题目：设计一个支持 128K Token 上下文的 LLM 推理服务

**约束**：
- 单 A100 80GB
- 模型：LLaMA-2 7B（32 heads, d=128）
- 请求量：1 QPS，延迟 SLA < 5s

**思路**：

1. **Attention 的内存计算**

   标准 Attention，N=128K：
   - 单头注意力矩阵：128K × 128K × 2 = 32 GB
   - 32 头 × 32 GB = 1024 GB（根本放不下！）

   FlashAttention 不存注意力矩阵，KV Cache 需要：
   - 每层 K+V：2 × 128K × 4096 × 2 = 2 GB（每层）
   - 32 层：64 GB（还是超出 80GB，需要量化）

2. **KV Cache 量化**

   INT8 KV Cache：64 GB → 32 GB，剩余 48 GB 存模型权重（7B × 2 = 14 GB → 够！）

3. **FlashAttention 加速**

   序列长 128K 时，标准 Attention 需要 O(N²) IO，FlashAttention 通过分块将内存从 O(N²) 降至 O(N)，是唯一可行的选择。

4. **估算吞吐**

   FlashAttention 在 A100 上，128K 序列的 attention 耗时约 50ms × 32 层 = 1.6s
   加上 FFN、KV Cache 访问等，总推理时间约 3-4s，满足 5s SLA。

**答题框架**：

```
系统设计回答模板：
1. 规模估算（内存、计算量）
2. 技术选型（为什么用 FlashAttention？）
3. 量化策略（KV Cache INT8）
4. 瓶颈分析（HBM 带宽 vs 计算）
5. 权衡讨论（精度 vs 速度 vs 内存）
```

## 本周总结

| 知识点 | 重要程度 | 面试频率 |
|--------|---------|---------|
| 标准 Attention IO 瓶颈 | ⭐⭐⭐⭐⭐ | 极高 |
| Online Softmax 原理 | ⭐⭐⭐⭐⭐ | 极高 |
| FlashAttention Tiling | ⭐⭐⭐⭐⭐ | 极高 |
| v1 vs v2 区别 | ⭐⭐⭐⭐ | 高 |
| v3 H100 特性 | ⭐⭐⭐ | 中 |
| CUDA/Triton 实现 | ⭐⭐⭐⭐ | 中高 |

**下周预告：推理系统优化——PagedAttention / vLLM / 投机解码**
