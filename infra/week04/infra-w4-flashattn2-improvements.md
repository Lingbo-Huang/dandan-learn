---
layout: default
title: "D4 · FlashAttention v2 优化点深度解析"
render_with_liquid: false
---

# D4 · FlashAttention v2 优化点深度解析

## 为什么需要 v2？

FlashAttention v1 相比标准 Attention 有 2-4× 加速，但 A100 利用率仍只有 **25-35%**（理论峰值）。

Tri Dao 在 2023 年发布 FlashAttention-2（[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)），将 A100 利用率提升到 **50-73%**，主要来自三项改进：

1. **减少 non-matmul FLOPs**
2. **并行化策略优化**（外 Q 内 KV）
3. **Warp 分组优化**

## 改进 1：减少非矩阵乘法 FLOPs

A100 的计算能力：
- Tensor Core（矩阵乘法）：312 TFLOPS FP16
- CUDA Core（普通运算）：~78 TFLOPS FP32

**v1 的问题**：Online Softmax 的更新公式中有大量除法和乘法发生在 non-matmul 路径上。

**v2 的改进**：重写更新公式，减少不必要的除法。

### v1 vs v2 的更新公式对比

**v1 更新**（在每个 Kj 块后立即归一化）：
```
# 每处理一块，Oi 就被归一化（除以 ℓi_new）
Oi_new = (li * exp(mi - mi_new)) * Oi / li_new
        + exp(mij - mi_new) * Pij @ Vj / li_new
```

**v2 更新**（延迟归一化到最后）：
```
# 在循环中不归一化，只在最后做一次除法
# 维护 "未归一化的 O"
Oi_unnorm = exp(mi - mi_new) * Oi_unnorm + exp(mij - mi_new) * Pij @ Vj

# 最终
Oi = Oi_unnorm / li
```

**节省的操作**：每次 KV 块循环少一次除法（N/Bc 次减少到 1 次）。

## 改进 2：并行化策略 —— 外 Q 内 KV

这是 v2 最重要的改进。

### v1 的问题：Q 被反复读取

```
v1 循环结构：
for j = 1 to Tc:   # 外层：K,V 块
    for i = 1 to Tr:   # 内层：Q 块
        load Qi from HBM  # Qi 被读取 Tc 次！
        compute ...
```

每个 Q 块被读取 Tc = N/Bc 次。对于 N=4096, Bc=64，每个 Q 块被读取 64 次！

### v2 的解决：交换循环顺序

```
v2 循环结构：
for i = 1 to Tr:   # 外层：Q 块
    load Qi from HBM  # Qi 只读一次！
    for j = 1 to Tc:   # 内层：K,V 块
        load Kj, Vj from HBM
        compute ...
    write Oi to HBM   # Oi 只写一次！
```

**额外好处**：每个 Q 块的计算完全独立，天然适合并行！

### 并行维度扩展

v2 的 Grid 维度：
```
grid = (Tr, batch_size * num_heads)
```

- 每个 CUDA Block 处理一个 (batch, head, Q块) 的组合
- 多个 Block 完全并行，充分利用 GPU 的所有 SM

**对 batch size 小时的意义**：即使 batch=1，也能通过 Tr × num_heads 维度来并行，v1 在这种情况下 GPU 利用率很低。

## 改进 3：Warp 分组与共享内存优化

### v1 的 Warp 分配

v1 中，4个 Warp 共同计算 Qi × Kj^T：
```
Warp 0: 计算 Sij 的第 0-15 列
Warp 1: 计算 Sij 的第 16-31 列
Warp 2: 计算 Sij 的第 32-47 列
Warp 3: 计算 Sij 的第 48-63 列
最后需要 Warp 间通信来计算 rowmax/rowsum
```

### v2 的 Warp 分配

v2 改为按行分割：
```
每个 Warp 计算 Sij 的一部分行，独立做 online softmax
最后各 Warp 的结果通过 shared memory 合并
```

好处：减少 Warp 间通信，每个 Warp 的 softmax 计算更局部化。

```cuda
// v2 风格的 Warp 级 Attention（简化版）
// 假设 4 个 Warp，每个 Warp 处理 Br/4 行 Q

__global__ void flash_attention_v2_kernel(
    const half* Q, const half* K, const half* V, half* O,
    float* L, int N, int d, float scale,
    int Br, int Bc  // 块大小，运行时参数
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // 每个 Block 处理 Qi[i*Br : (i+1)*Br]
    int qi_start = blockIdx.x * Br;
    int qi_end   = min(qi_start + Br, N);
    
    // 每个 Warp 处理其中的 Br/4 行
    int warp_row_start = qi_start + warp_id * (Br / 4);
    int warp_row_end   = min(warp_row_start + Br/4, qi_end);
    
    // ... 加载 Qi 到寄存器（减少 shared memory 使用）
    // ... 遍历 Kj, Vj 块，在 Warp 内完成 online softmax
    // ... 最终写 Oi 到 HBM
}
```

## v2 的具体性能数据

**A100 80GB，causal attention，FP16：**

| 序列长度 | v1 (TFLOPS) | v2 (TFLOPS) | 提升 | 理论峰值占比 |
|---------|-------------|-------------|------|------------|
| 512 | 148 | 162 | 1.1× | 52% |
| 1024 | 145 | 185 | 1.3× | 59% |
| 2048 | 117 | 195 | 1.7× | 63% |
| 4096 | 73 | 198 | 2.7× | 63% |
| 8192 | - | 215 | - | 69% |
| 16384 | - | 226 | - | 73% |

**长序列时 v2 优势更明显**，因为 Q 块交换后，Q 的 HBM 读取大幅减少。

**与 xFormers memory-efficient attention 对比：**

| 实现 | N=4096 速度 | N=8192 速度 |
|------|------------|------------|
| xFormers | ~150 TFLOPS | ~140 TFLOPS |
| FlashAttention v2 | ~198 TFLOPS | ~215 TFLOPS |
| 提升 | 1.3× | 1.5× |

## 额外特性：支持更多 Mask 模式

v2 新增：
- **Causal Mask**：通过 `is_causal=True` 启用，跳过上三角的计算
- **Window Attention**：支持滑动窗口（Mistral 等模型使用）
- **ALiBi 位置偏置**：在 score 上加偏置

```python
from flash_attn import flash_attn_func

# Causal attention（LLM decoder 标配）
out = flash_attn_func(q, k, v, causal=True)

# 带 ALiBi 的 attention
from flash_attn import flash_attn_with_kvcache
out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=cache_lens,
    causal=True,
    alibi_slopes=slopes  # ALiBi 斜率
)
```

## v2 多查询注意力（MQA/GQA）支持

v2 原生支持 Multi-Query Attention 和 Grouped-Query Attention：

```python
# GQA：8 个 Q head 共享 2 个 KV head
from flash_attn import flash_attn_func

# q: [batch, seqlen, 8, head_dim]
# k: [batch, seqlen, 2, head_dim]  ← 更少的 KV head
# v: [batch, seqlen, 2, head_dim]
out = flash_attn_func(q, k, v, causal=True)
# flash_attn 内部自动处理 q_head → kv_head 的映射
```

## 代码对比：标准 vs FlashAttention v2

```python
import torch
from flash_attn import flash_attn_func
import time

def benchmark_attention(batch, nheads, seqlen, headdim, causal=True, n_reps=100):
    device = torch.device('cuda')
    dtype = torch.float16
    
    q = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
    
    q_t = q.transpose(1, 2)  # [batch, nheads, seqlen, headdim]
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    
    # 标准 Attention
    def standard_attn():
        return torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=causal
        )
    
    # FlashAttention v2
    def flash_attn_v2():
        return flash_attn_func(q, k, v, causal=causal)
    
    # 测速
    for fn, name in [(standard_attn, "Standard"), (flash_attn_v2, "FlashAttn v2")]:
        # warmup
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(n_reps):
            fn()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_reps * 1000
        
        print(f"{name}: {elapsed:.2f} ms")

benchmark_attention(batch=2, nheads=32, seqlen=4096, headdim=128)
# 典型输出（A100）：
# Standard:      4.85 ms
# FlashAttn v2:  1.82 ms  → 2.7× 加速
```

## 本节小结

| 改进项 | v1 问题 | v2 解决方案 | 收益 |
|--------|---------|------------|------|
| FLOPs | 每块归一化 | 延迟归一化 | 减少 non-matmul ops |
| 并行化 | 外KV内Q，Q读多次 | 外Q内KV，Q读一次 | IO 减少，并行度提升 |
| Warp分组 | 按列分割，需通信 | 按行分割，局部性更好 | 减少 Warp 间同步 |
| GPU利用率 | 25-35% | 50-73% | ~2× 效率提升 |

## 面试题

**Q: FlashAttention v2 对比 v1 最关键的改进是什么？**

A: 最关键的改进是**交换外层循环**，从"外K内Q"改为"外Q内K"。这使得每个Q块只需从HBM读取一次，而不是被读取 Tc=N/Bc 次。同时这也使得不同Q块的计算完全独立，可以映射到不同的CUDA Block并行执行，提升了低batch场景的GPU利用率。配合延迟归一化（最后才除以ℓ）减少了非矩阵乘法运算，综合使A100利用率从~30%提升到~70%。
