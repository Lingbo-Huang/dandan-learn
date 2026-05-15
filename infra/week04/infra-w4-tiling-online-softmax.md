---
layout: default
title: "D2 · 分块计算与 Online Softmax 推导"
render_with_liquid: false
---

# D2 · 分块计算与 Online Softmax 推导

## 核心问题：Softmax 必须看到所有元素

标准 Softmax：
$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

分母 $\sum_j e^{x_j}$ 需要看到 **所有** $x_j$，这意味着必须先把整行 $S_i$ 加载进来。

**如果我们分块处理，能在不完整数据上计算 Softmax 吗？**

答案是：**可以，用 Online Softmax（流式/增量式 Softmax）。**

## 数值稳定的 Softmax

首先处理数值溢出问题。直接计算 $e^{x_i}$ 在 $x_i > 88$ 时 FP32 就会溢出。

**Safe Softmax**：减去最大值
$$\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j$$

这等价于原始 Softmax（分子分母同除 $e^m$），但数值上稳定。

## Online Softmax：三遍 → 两遍 → 一遍的演进

### 朴素实现（三遍扫描）

```
Pass 1: 扫描 x，找到 m = max(x)
Pass 2: 计算 d = sum(exp(x - m))
Pass 3: 输出 exp(x_i - m) / d
```

三次扫描意味着三次 HBM 读取，低效。

### 两遍扫描（合并 Pass 2 和 3 的计算）

```
Pass 1: 扫描 x，找到 m = max(x)，计算 d = sum(exp(x - m))
Pass 2: 输出 exp(x_i - m) / d
```

### Online Softmax（单遍扫描！）

**关键洞察**：当我们读取新元素时，可以 **更新之前的累加和**。

设已经处理了前 $t$ 个元素，维护：
- $m_t = \max(x_1, ..., x_t)$（当前已见到的最大值）
- $\ell_t = \sum_{j=1}^{t} e^{x_j - m_t}$（归一化到当前最大值的累加和）

当读入 $x_{t+1}$ 时，更新规则：

$$m_{t+1} = \max(m_t, x_{t+1})$$

$$\ell_{t+1} = \ell_t \cdot e^{m_t - m_{t+1}} + e^{x_{t+1} - m_{t+1}}$$

**理解**：$\ell_t$ 是以 $m_t$ 为基准的，当基准从 $m_t$ 变为 $m_{t+1}$ 时，旧的累加和需要乘以 $e^{m_t - m_{t+1}}$ 来修正。

```python
def online_softmax(x):
    """
    Online Softmax：单遍扫描
    无需预先存储整个 x，适合流式/分块处理
    """
    m = float('-inf')  # 当前最大值
    ell = 0.0          # 归一化累加和
    
    # 第一遍：计算 m 和 ell（合并）
    for xi in x:
        m_new = max(m, xi)
        ell = ell * math.exp(m - m_new) + math.exp(xi - m_new)
        m = m_new
    
    # 第二遍：输出（实际 FlashAttention 把这步也融合进去了）
    return [math.exp(xi - m) / ell for xi in x]

# 验证正确性
import math
x = [1.0, 3.0, 2.0, 0.5, 4.0]
print("Online:", online_softmax(x))
print("torch: ", torch.softmax(torch.tensor(x), dim=0).tolist())
```

## 分块 Attention 计算（FlashAttention 核心）

现在将 Online Softmax 扩展到矩阵级别。

考虑计算注意力输出的第 $i$ 行：
$$O_i = \sum_j P_{ij} V_j = \sum_j \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}} V_j$$

将 K, V 分成 $T_c$ 个块，每块大小 $B_c$：

**处理第 $j$ 块时，维护的状态：**
- $m_i^{(j)}$：到目前为止见到的最大 attention score
- $\ell_i^{(j)}$：归一化因子（softmax 分母）
- $O_i^{(j)}$：累积的输出

**更新规则（设当前块的 score 为 $s$）：**

$$m_i^{new} = \max(m_i^{old}, \max_k s_{ik})$$

$$\ell_i^{new} = e^{m_i^{old} - m_i^{new}} \cdot \ell_i^{old} + \sum_k e^{s_{ik} - m_i^{new}}$$

$$O_i^{new} = \frac{\ell_i^{old} \cdot e^{m_i^{old} - m_i^{new}}}{\ell_i^{new}} O_i^{old} + \frac{e^{s_{ik} - m_i^{new}}}{\ell_i^{new}} v_k$$

```python
import torch
import math

def flash_attention_naive(Q, K, V, block_size=64):
    """
    FlashAttention 的 Python 参考实现（教学用，非 CUDA）
    Q, K, V: [N, d]
    """
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    O = torch.zeros(N, d, dtype=Q.dtype)
    m = torch.full((N,), float('-inf'))   # 每行的当前 max
    ell = torch.zeros(N)                   # 每行的归一化因子
    
    # 外层循环：分块 K, V
    for j in range(0, N, block_size):
        Kj = K[j:j+block_size]  # [Bc, d] —— 从 HBM 加载到 SRAM
        Vj = V[j:j+block_size]  # [Bc, d]
        
        # 计算 Q 与当前块 K 的 attention score
        S_block = Q @ Kj.T * scale  # [N, Bc] —— 在 SRAM 中计算
        
        # 更新 online softmax 状态
        m_block = S_block.max(dim=1).values  # [N]
        m_new = torch.maximum(m, m_block)
        
        exp_S = torch.exp(S_block - m_new.unsqueeze(1))  # [N, Bc]
        ell_block = exp_S.sum(dim=1)                       # [N]
        
        # 更新 O（关键：用旧的 m 和 ell 来修正）
        correction = torch.exp(m - m_new)  # [N]
        O = O * correction.unsqueeze(1) + exp_S @ Vj
        
        ell = ell * correction + ell_block
        m = m_new
    
    # 最终归一化
    O = O / ell.unsqueeze(1)
    return O

# 验证正确性
N, d = 512, 64
Q = torch.randn(N, d)
K = torch.randn(N, d)
V = torch.randn(N, d)

out_flash = flash_attention_naive(Q, K, V, block_size=64)
out_std   = torch.nn.functional.scaled_dot_product_attention(
    Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
).squeeze(0)

print(f"Max diff: {(out_flash - out_std).abs().max().item():.6f}")
# 应该 < 1e-4
```

## Block 大小的选择

SRAM 的大小限制了 block size。设 SRAM 大小为 $M$，head dim 为 $d$：

每个 SM 的 SRAM 可以放：
- Q 的一行：$d$ 个元素
- 一块 K：$B_c \times d$ 个元素  
- 一块 V：$B_c \times d$ 个元素
- S 的一块：$B_r \times B_c$ 个元素

约束：$B_r \cdot d + 2 B_c \cdot d + B_r \cdot B_c \leq M$

FlashAttention 论文中设 $B_c = B_r = B$，得：
$$B \approx \sqrt{M / 4d}$$

A100 SRAM = 192KB, d = 128, FP16：
$$B \approx \sqrt{192 \times 1024 / (4 \times 128 \times 2)} \approx \sqrt{192} \approx 96$$

实际 FlashAttention 实现中常用 $B = 64$ 或 $B = 128$。

## IO 复杂度对比

| 方法 | HBM 读写量 | 备注 |
|------|-----------|------|
| 标准 Attention | $O(N^2)$ | 必须存 N×N 矩阵 |
| FlashAttention | $O(N^2 d / M)$ | M = SRAM 大小 |

当 $M \gg d$ 时（实际情况），FlashAttention 的 IO 接近 $O(N^2)$，但**系数**小很多——实际测量约 5-20× 的 IO 减少。

关键区别：FlashAttention **不物化** N×N 的中间矩阵，只在 SRAM 中保存当前块的临时结果。

## 反向传播的额外存储

FlashAttention 前向只需存储 $(m, \ell)$，大小 $O(N)$，而非 $O(N^2)$。

但反向传播需要重新计算注意力分数（recomputation）：
- 重算代价：约 1.5× FLOPs（相比标准反向传播）
- 节省内存：$O(N^2) \to O(N)$，对长序列极大减少峰值内存

**这个 recomputation 换内存的设计，正是 FlashAttention 能处理超长序列的关键。**

## 本节总结

| 技术 | 问题 | 解决方案 |
|------|------|---------|
| 数值稳定 | exp 上溢 | 减去最大值 |
| Online Softmax | 分块无法算 softmax | 维护 (m, ℓ) 增量更新 |
| Tiling | N×N 矩阵放不进 SRAM | 分块处理，状态累积 |
| 内存节省 | O(N²) 中间激活 | Recomputation |

**下一节：FlashAttention v1 的 CUDA 实现细节。**

## 面试题

**Q: 请解释 FlashAttention 中 Online Softmax 的更新公式。**

A: 当处理第 j 块 Key 时，我们维护三个量：当前最大 attention score $m$、归一化因子 $\ell$、以及累积输出 $O$。读入新块时，新最大值 $m' = \max(m, m_{block})$。由于基准改变，旧的 $\ell$ 要乘以 $e^{m-m'}$ 来修正，旧的 $O$ 也要乘以同样的修正因子。这样最终除以 $\ell$ 得到的结果与全量计算完全等价，但全程只需要 O(block_size) 的 SRAM。
