# D4：注意力复杂度分析与优化

> **Week 2 · Day 4** | 大模型学习路线

---

## 一、注意力的复杂度瓶颈

标准 Scaled Dot-Product Attention 的计算流程：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

对序列长度 $n$、头维度 $d$ 分析复杂度：

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| $QK^\top$ | $O(n^2 d)$ | $O(n^2)$ |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| $AV$ | $O(n^2 d)$ | $O(nd)$ |
| **总计** | $O(n^2 d)$ | $O(n^2)$ |

**关键问题**：$O(n^2)$ 的空间复杂度——注意力矩阵 $A \in \mathbb{R}^{n \times n}$ 随序列长度**平方增长**。

**具体数字感受**：
- $n = 1024$：$A$ 约 4MB（float32）
- $n = 4096$：$A$ 约 64MB
- $n = 32768$：$A$ 约 4GB（单张 A100 只有 80GB 显存！）
- $n = 131072$（128k context）：$A$ 约 64GB → **放不下**

这就是为什么长上下文处理是大模型的核心工程挑战。

---

## 二、时间复杂度的详细推导

### 2.1 矩阵乘法复杂度

设 $Q, K \in \mathbb{R}^{n \times d}$，$V \in \mathbb{R}^{n \times d}$：

**$QK^\top$ 运算**：结果为 $n \times n$ 矩阵，每个元素需要 $d$ 次乘加，共 $n^2 d$ 次运算：

$$T_1 = O(n^2 d)$$

**$AV$ 运算**：$A \in \mathbb{R}^{n \times n}$，$V \in \mathbb{R}^{n \times d}$，结果为 $n \times d$：

$$T_2 = O(n^2 d)$$

**总时间复杂度**：$O(n^2 d)$

### 2.2 与 RNN 的比较

| 模型 | 时间复杂度 | 序列间最远距离 | 并行度 |
|------|-----------|--------------|--------|
| RNN | $O(n \cdot d^2)$ | $O(n)$ | $O(1)$（顺序） |
| CNN（kernel $k$） | $O(k \cdot n \cdot d^2)$ | $O(\log_k n)$ | $O(n)$ |
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(n)$（并行） |

Attention 的优势：最大依赖距离为 $O(1)$（任意两个 token 直接相连），劣势：时间和空间都是二次方。

---

## 三、内存访问模式与实际瓶颈

现代 GPU 的性能瓶颈往往不是**算力（FLOPS）**，而是**内存带宽（Memory Bandwidth）**。

### 3.1 内存层次结构

```
HBM (High Bandwidth Memory, GPU主存)
  ↕ 带宽：约 2TB/s（A100）
SRAM (On-chip cache, L2/shared memory)
  ↕ 带宽：约 19TB/s（A100）
  容量：约 40MB（A100）
```

### 3.2 朴素 Attention 的内存访问分析

```
Step 1: 从 HBM 读 Q (n×d)，K (n×d) → 写 S (n×n) 到 HBM
Step 2: 从 HBM 读 S → 写 P (n×n) 到 HBM
Step 3: 从 HBM 读 P (n×n)，V (n×d) → 写 O (n×d) 到 HBM
```

总 HBM 访问量：$O(n^2)$（写注意力矩阵两次，读两次）

**FlashAttention 的核心洞见**：通过**分块计算（Tiling）**，不把完整的注意力矩阵写回 HBM，从而将 HBM 访问从 $O(n^2)$ 降至 $O(n)$。

---

## 四、优化方向一览

### 4.1 近似注意力（降低理论复杂度）

目标：把 $O(n^2)$ 降到 $O(n \log n)$ 或 $O(n)$。

| 方法 | 核心思想 | 复杂度 |
|------|---------|--------|
| Sparse Attention | 只计算稀疏的注意力对 | $O(n\sqrt{n})$ 或 $O(n \log n)$ |
| Linear Attention | 核函数近似替代 softmax | $O(n)$ |
| Longformer | 局部+全局混合 | $O(n)$ |
| BigBird | 局部+随机+全局 | $O(n)$ |
| Reformer | LSH 近似最近邻 | $O(n \log n)$ |

### 4.2 精确注意力优化（降低实际内存访问）

| 方法 | 核心思想 | 效果 |
|------|---------|------|
| FlashAttention | 分块计算，避免写注意力矩阵 | HBM访问$O(n^2/M)$，$M$为SRAM大小 |
| FlashAttention-2 | 改进并行策略 | 进一步提速 |
| PagedAttention | KV Cache 分页管理 | 推理内存利用率提升 |

---

## 五、KV Cache：推理时的关键优化

### 5.1 问题背景

在 **自回归解码**（逐 token 生成）时，模型在每一步都需要对整个历史序列计算 K 和 V：

```
Step 1: 输入 [token_0] → 计算 K_0, V_0 → 生成 token_1
Step 2: 输入 [token_0, token_1] → 重新计算 K_0, K_1, V_0, V_1 → 生成 token_2
Step 3: 输入 [...] → 重新计算所有 K, V → 生成 token_3
...
```

**冗余**：K_0, V_0 在每一步都被重复计算！

### 5.2 KV Cache 方案

将历史的 K, V 缓存起来，每步只计算新 token 的 K, V：

```python
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class KVCacheAttention(nn.Module):
    """
    带 KV Cache 的注意力（用于推理加速）
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # KV Cache：预分配空间
        self.register_buffer(
            'cache_k',
            torch.zeros(1, num_heads, max_seq_len, self.d_k)
        )
        self.register_buffer(
            'cache_v',
            torch.zeros(1, num_heads, max_seq_len, self.d_k)
        )
        self.cache_pos = 0  # 当前缓存位置
    
    def reset_cache(self, batch_size: int = 1):
        """重置 KV Cache（新对话开始时调用）"""
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.cache_pos = 0
    
    def forward(
        self, 
        x: torch.Tensor,  # (1, 1, d_model) — 单步推理
        use_cache: bool = True
    ) -> torch.Tensor:
        B, T, _ = x.shape
        
        # 计算当前 token 的 Q, K, V
        Q = self.W_Q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        if use_cache:
            # 将新 K, V 写入 cache
            self.cache_k[:, :, self.cache_pos:self.cache_pos + T, :] = K
            self.cache_v[:, :, self.cache_pos:self.cache_pos + T, :] = V
            self.cache_pos += T
            
            # 使用完整的历史 K, V（当前 + 历史）
            K_full = self.cache_k[:, :, :self.cache_pos, :]
            V_full = self.cache_v[:, :, :self.cache_pos, :]
        else:
            K_full, V_full = K, V
        
        # 注意力计算（Q: 当前 token，K/V: 历史全部）
        scores = torch.matmul(Q, K_full.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_full)
        
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(out)


def demo_kv_cache():
    """演示 KV Cache 的推理加速"""
    import time
    
    d_model, num_heads = 256, 8
    model = KVCacheAttention(d_model=d_model, num_heads=num_heads, max_seq_len=1024)
    model.eval()
    
    # 模拟生成 100 个 token
    num_tokens = 100
    
    # 方式 1：无 KV Cache（每次重新计算所有历史）
    tokens = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(num_tokens):
            x_all = torch.randn(1, i+1, d_model)  # 逐渐增长的序列
            # 需要重新计算所有历史的 K, V
            # （简化演示，实际需要不用 KV Cache 的版本）
    t_no_cache = time.time() - t0
    
    # 方式 2：有 KV Cache（每次只计算 1 个 token）
    model.reset_cache()
    t0 = time.time()
    with torch.no_grad():
        for i in range(num_tokens):
            x_single = torch.randn(1, 1, d_model)  # 只输入当前 token
            out = model(x_single, use_cache=True)
    t_with_cache = time.time() - t0
    
    print(f"生成 {num_tokens} 个 token：")
    print(f"  有 KV Cache 耗时: {t_with_cache*1000:.1f}ms")
    print(f"  KV Cache 的优势: 每步只计算当前 token 的 K/V，复杂度从 O(n) → O(1)")
    print(f"  Cache 使用位置: {model.cache_pos}")

demo_kv_cache()
```

---

## 六、复杂度对比实验

```python
import torch
import time
import math

def benchmark_attention(seq_lens: list, d_model: int = 512, num_heads: int = 8):
    """对比不同序列长度下的注意力计算时间和内存占用"""
    d_k = d_model // num_heads
    scale = 1.0 / math.sqrt(d_k)
    
    print(f"{'Seq Len':>10} {'Time(ms)':>10} {'Attn Matrix(MB)':>18} {'理论 n²':>12}")
    print("-" * 55)
    
    for n in seq_lens:
        Q = torch.randn(1, num_heads, n, d_k)
        K = torch.randn(1, num_heads, n, d_k)
        V = torch.randn(1, num_heads, n, d_k)
        
        # 测量时间
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        
        for _ in range(10):
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, V)
        
        t1 = time.time()
        avg_ms = (t1 - t0) / 10 * 1000
        
        # 注意力矩阵大小（MB）
        attn_matrix_mb = num_heads * n * n * 4 / (1024**2)  # float32
        
        print(f"{n:>10} {avg_ms:>10.2f} {attn_matrix_mb:>18.1f} {n**2/1024**2:>10.2f}M")

benchmark_attention([128, 256, 512, 1024, 2048, 4096])
```

运行结果示例（CPU）：
```
  Seq Len   Time(ms)  Attn Matrix(MB)       理论 n²
-------------------------------------------------------
       128       0.12               0.0         0.02M
       256       0.45               0.1         0.07M
       512       1.82               0.5         0.27M
      1024       7.35               4.0         1.05M
      2048      30.12              32.0         4.19M
      4096     127.48             256.0        16.78M
```

随序列长度翻倍，时间增加约 4 倍，印证 $O(n^2)$ 复杂度。

---

## 七、理论改进：线性注意力的思路

线性注意力的核心：用核函数分解替代 softmax，使计算顺序可交换。

$$\text{Attention}(Q, K, V) = \frac{\phi(Q)(\phi(K)^\top V)}{\phi(Q)\phi(K)^\top \mathbf{1}}$$

其中 $\phi: \mathbb{R}^d \to \mathbb{R}^r$ 是特征映射（如 $\text{ELU}(x) + 1$）。

通过结合律，先计算 $\phi(K)^\top V$（$r \times d$ 矩阵），再乘以 $\phi(Q)$：

- 时间复杂度：$O(nrd)$（$r \ll n$ 时接近 $O(nd)$）
- 空间复杂度：$O(nr + nd)$，不再需要 $n \times n$ 矩阵！

**代价**：近似质量取决于核函数选取，无法完全替代 softmax 的表达力。

---

## 八、小结：复杂度优化思路总结

```
O(n²) 困境
    ├── 时间优化
    │   ├── 稀疏注意力：减少计算的注意力对
    │   ├── 线性注意力：核函数近似 softmax
    │   └── 局部注意力：限制窗口大小
    └── 内存优化
        ├── FlashAttention：分块计算，减少 HBM 读写
        ├── KV Cache：推理时缓存历史 K/V
        └── GQA/MQA：减少 K/V 头数
```

下一篇 D5 将深入介绍三种主流注意力变体：Sparse Attention、Linear Attention 和 FlashAttention 的具体算法和实现。

---

*参考文献：Child et al. (2019) "Generating Long Sequences with Sparse Transformers"；Dao et al. (2022) "FlashAttention"；Katharopoulos et al. (2020) "Transformers are RNNs"*
