# D5：注意力变体——Sparse / Linear / Flash

> **Week 2 · Day 5** | 大模型学习路线

---

## 一、为什么要有变体？

标准多头注意力（MHA）的 $O(n^2)$ 复杂度使其在长序列场景中举步维艰。三种主流变体从不同角度攻克这一难题：

| 变体 | 核心思路 | 时间复杂度 | 空间复杂度 | 精度损失 |
|------|---------|-----------|-----------|---------|
| **Sparse Attention** | 只计算部分注意力对 | $O(n\sqrt{n})$ | $O(n\sqrt{n})$ | 轻微 |
| **Linear Attention** | 核函数近似 softmax | $O(nd^2)$ | $O(nd)$ | 中等 |
| **FlashAttention** | 分块 IO 优化 | $O(n^2)$ | $O(n)$ | **无** |

FlashAttention 特别值得注意：它不降低理论复杂度，却通过减少内存访问在**实践中大幅提速**，且与标准 Attention 数学等价。

---

## 二、Sparse Attention（稀疏注意力）

### 2.1 核心思想

不是所有词对之间的注意力都有意义。与其计算 $n^2$ 个注意力分数，不如只计算"重要"的那些。

$$\text{SparseAttention}(Q, K, V) = \text{softmax}\!\left(\frac{Q_{S(i)} K_{S(i)}^\top}{\sqrt{d_k}}\right) V_{S(i)}$$

其中 $S(i)$ 是第 $i$ 个 Query 需要关注的 Key 集合。

### 2.2 经典模式

**Strided Attention（步长注意力）**（OpenAI Sparse Transformer）：
- 局部窗口：每个 token 关注前 $w$ 个 token（$O(nw)$）
- 步长跳跃：每个 token 关注相隔 $s$ 的 token（$O(n \cdot n/s)$）

**Longformer 混合模式**：
- 局部窗口 + 特定全局 token（如 `[CLS]`）

**BigBird**：
- 局部（$O(nw)$）+ 随机（$O(n \cdot r)$）+ 全局（$O(n \cdot g)$）
- 理论证明近似完整注意力的表达能力

### 2.3 PyTorch 实现：带局部窗口的稀疏注意力

```python
import torch
import torch.nn.functional as F
import math
from typing import Optional

def local_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor, 
    V: torch.Tensor,
    window_size: int = 4,
) -> torch.Tensor:
    """
    局部窗口注意力：每个 token 只关注前后 window_size 个 token
    
    Args:
        Q, K, V: (B, T, d_k)
        window_size: 窗口半径
    
    Returns:
        output: (B, T, d_k)
    """
    B, T, d_k = Q.shape
    scale = 1.0 / math.sqrt(d_k)
    
    outputs = []
    
    for i in range(T):
        # 当前 token 的 Query
        q_i = Q[:, i:i+1, :]  # (B, 1, d_k)
        
        # 关注窗口 [i-window_size, i+window_size]
        start = max(0, i - window_size)
        end = min(T, i + window_size + 1)
        
        k_window = K[:, start:end, :]  # (B, window, d_k)
        v_window = V[:, start:end, :]  # (B, window, d_k)
        
        # 计算局部注意力
        scores = torch.bmm(q_i, k_window.transpose(1, 2)) * scale  # (B, 1, window)
        attn = F.softmax(scores, dim=-1)
        out_i = torch.bmm(attn, v_window)  # (B, 1, d_k)
        outputs.append(out_i)
    
    return torch.cat(outputs, dim=1)  # (B, T, d_k)


# 高效实现：使用 unfold 操作
def efficient_local_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int = 4,
) -> torch.Tensor:
    """
    高效的局部窗口注意力（使用 unfold 避免 Python 循环）
    """
    B, T, d_k = Q.shape
    scale = 1.0 / math.sqrt(d_k)
    w = window_size
    
    # Padding 以处理边界
    pad = w
    K_padded = F.pad(K, (0, 0, pad, pad), value=0)  # (B, T+2w, d_k)
    V_padded = F.pad(V, (0, 0, pad, pad), value=0)
    mask_padded = F.pad(
        torch.zeros(B, T, dtype=torch.bool), 
        (pad, pad), value=True
    )  # 标记 padding 位置
    
    # 提取滑动窗口
    # (B, T, 2w+1, d_k)
    K_windows = K_padded.unfold(1, 2*w+1, 1).permute(0, 1, 3, 2)
    V_windows = V_padded.unfold(1, 2*w+1, 1).permute(0, 1, 3, 2)
    
    # 计算注意力分数 (B, T, 1, d_k) x (B, T, d_k, 2w+1) -> (B, T, 1, 2w+1)
    scores = torch.einsum('btd,btwnd->btw', Q, K_windows.unsqueeze(2)).squeeze(2) * scale
    
    # Softmax
    attn = F.softmax(scores, dim=-1)  # (B, T, 2w+1)
    
    # 加权聚合
    output = torch.einsum('btw,btwnd->btd', attn, V_windows.unsqueeze(2).squeeze(2))
    
    return output


# 测试
B, T, d_k = 2, 20, 32
Q = K = V = torch.randn(B, T, d_k)

out_loop = local_window_attention(Q, K, V, window_size=3)
print(f"局部窗口注意力输出形状: {out_loop.shape}")
print(f"复杂度：O(n * 2w) = O({T} * {2*3+1}) = {T * 7} vs 全注意力 O({T**2})")
```

---

## 三、Linear Attention（线性注意力）

### 3.1 核心数学

标准注意力：

$$\text{Attention}(Q, K, V)_i = \frac{\sum_j \exp(q_i^\top k_j / \sqrt{d}) v_j}{\sum_j \exp(q_i^\top k_j / \sqrt{d})}$$

线性注意力的关键：用核函数 $\phi(\cdot)$ 分解：

$$\text{Attention}(Q, K, V)_i \approx \frac{\sum_j \phi(q_i)^\top \phi(k_j) \cdot v_j}{\sum_j \phi(q_i)^\top \phi(k_j)}$$

由于 $\phi(q_i)^\top \phi(k_j) = \phi(q_i)^\top (\phi(k_j) v_j^\top)$ 的乘法可以重新结合：

$$= \frac{\phi(q_i)^\top \left(\sum_j \phi(k_j) v_j^\top\right)}{\phi(q_i)^\top \left(\sum_j \phi(k_j)\right)}$$

**关键**：$\sum_j \phi(k_j) v_j^\top$ 是一个 $d \times d$ 矩阵，可以预先计算！

这样对每个 Query $q_i$，只需要一次矩阵-向量乘法，整体复杂度降至 $O(nd^2)$。

### 3.2 常用核函数

- **ELU + 1**：$\phi(x) = \text{ELU}(x) + 1$（保证非负，近似 softmax）
- **Random Fourier Features**：近似 RBF 核
- **Polynomial**：$\phi(x) = (x + \epsilon)^2 / d$

### 3.3 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearAttention(nn.Module):
    """
    线性注意力实现
    使用 ELU+1 作为核函数
    """
    def __init__(self, d_model: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.eps = eps
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """核函数 φ(x) = ELU(x) + 1（保证非负）"""
        return F.elu(x) + 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d_model = x.shape
        h = self.num_heads
        d_k = self.d_k
        
        # 投影
        Q = self.W_Q(x).view(B, T, h, d_k).transpose(1, 2)  # (B, h, T, d_k)
        K = self.W_K(x).view(B, T, h, d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, h, d_k).transpose(1, 2)
        
        # 核函数映射（保证非负，以便用线性近似 softmax）
        Q_feat = self.kernel_feature_map(Q)  # (B, h, T, d_k)
        K_feat = self.kernel_feature_map(K)
        
        # 线性注意力：先计算 K^T V，再乘以 Q
        # (B, h, d_k, d_k): Σ φ(k_j) v_j^T
        KV = torch.einsum('bhnd,bhnm->bhdm', K_feat, V)  # (B, h, d_k, d_k)
        
        # 分子: (B, h, T, d_k): Σ_j φ(q_i)^T φ(k_j) v_j
        numerator = torch.einsum('bhnd,bhdm->bhnm', Q_feat, KV)  # (B, h, T, d_k)
        
        # 分母: (B, h, T, 1)
        K_sum = K_feat.sum(dim=2, keepdim=True)  # (B, h, 1, d_k): Σ φ(k_j)
        denominator = torch.einsum('bhnd,bhmd->bhn', Q_feat, K_sum.squeeze(2)).unsqueeze(-1)  # (B, h, T, 1)
        
        # 归一化
        output = numerator / (denominator + self.eps)  # (B, h, T, d_k)
        
        # 合并头
        output = output.transpose(1, 2).contiguous().view(B, T, d_model)
        return self.W_O(output)


# 测试
B, T, d_model = 2, 128, 256
linear_attn = LinearAttention(d_model=d_model, num_heads=8)
x = torch.randn(B, T, d_model)
out = linear_attn(x)
print(f"线性注意力输出形状: {out.shape}")
print(f"复杂度: O(T * d²) = O({T} * {d_model//8}²) = {T * (d_model//8)**2} 次运算")
print(f"vs 标准注意力: O(T² * d) = O({T**2} * {d_model//8}) = {T**2 * d_model//8} 次运算")
```

---

## 四、FlashAttention：精确但高效

### 4.1 核心思想

FlashAttention 不改变计算结果（精确），而是**重新组织计算顺序**，减少 GPU 内存（HBM）的读写次数。

**关键洞察**：
1. GPU 的 SRAM（片上缓存）远快于 HBM（主存）
2. 标准 Attention 需要在 HBM 中存储完整的 $n \times n$ 注意力矩阵
3. 用分块（Tiling）计算可以只在 SRAM 中处理局部块，无需写回完整矩阵

### 4.2 分块 Softmax：Online Softmax

核心难点：Softmax 需要全局最大值和归一化因子，如何分块计算？

**Online Softmax 算法**（Milakov & Gimelshein, 2018）：

设当前已处理 $i-1$ 个元素的最大值 $m_{i-1}$ 和归一化分母 $d_{i-1}$，增量更新：

$$m_i = \max(m_{i-1}, x_i)$$

$$d_i = d_{i-1} \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i}$$

处理完所有元素后：$\text{softmax}(x)_i = e^{x_i - m_n} / d_n$

这使得 Softmax 可以**分块流式计算**，不需要事先知道全局最大值！

### 4.3 FlashAttention 算法草图

```python
import torch
import math

def flash_attention_naive(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """
    FlashAttention 的 CPU 教学实现（演示分块思路，非真实 CUDA 实现）
    
    实际 FlashAttention 是 CUDA kernel，这里用 Python 演示逻辑等价性
    
    Args:
        Q, K, V: (T, d) - 单头，省略 batch 维
        block_size: 分块大小（对应 SRAM 能容纳的大小）
    """
    T, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    # 输出和统计量的初始化
    O = torch.zeros(T, d)              # 输出
    l = torch.zeros(T, 1)              # 归一化因子（分母）
    m = torch.full((T, 1), float('-inf'))  # 行最大值
    
    # KV 分块
    num_kv_blocks = (T + block_size - 1) // block_size
    
    for j in range(num_kv_blocks):
        # 加载 K_j, V_j 块到"SRAM"
        kv_start = j * block_size
        kv_end = min((j + 1) * block_size, T)
        K_j = K[kv_start:kv_end, :]  # (block, d)
        V_j = V[kv_start:kv_end, :]  # (block, d)
        
        # Q 分块（外层循环）
        num_q_blocks = (T + block_size - 1) // block_size
        
        for i in range(num_q_blocks):
            q_start = i * block_size
            q_end = min((i + 1) * block_size, T)
            Q_i = Q[q_start:q_end, :]      # (q_block, d)
            m_i = m[q_start:q_end, :]      # (q_block, 1)
            l_i = l[q_start:q_end, :]      # (q_block, 1)
            O_i = O[q_start:q_end, :]      # (q_block, d)
            
            # 计算局部分数（不写回 HBM！）
            S_ij = torch.matmul(Q_i, K_j.transpose(0, 1)) * scale  # (q_block, kv_block)
            
            # Online Softmax 更新
            m_ij = S_ij.max(dim=-1, keepdim=True).values  # 局部最大值
            P_ij = torch.exp(S_ij - m_ij)                  # 局部指数
            l_ij = P_ij.sum(dim=-1, keepdim=True)          # 局部归一化因子
            
            # 更新全局统计量（Online 更新）
            m_new = torch.maximum(m_i, m_ij)
            l_new = l_i * torch.exp(m_i - m_new) + l_ij * torch.exp(m_ij - m_new)
            
            # 更新输出（修正之前的输出 + 加入新贡献）
            O[q_start:q_end] = (
                O_i * l_i * torch.exp(m_i - m_new) + 
                torch.matmul(P_ij, V_j) * torch.exp(m_ij - m_new)
            ) / l_new
            
            # 更新统计量
            m[q_start:q_end] = m_new
            l[q_start:q_end] = l_new
    
    return O


def verify_flash_vs_standard(T: int = 64, d: int = 32):
    """验证 FlashAttention 与标准 Attention 的数值等价性"""
    torch.manual_seed(42)
    Q = torch.randn(T, d)
    K = torch.randn(T, d)
    V = torch.randn(T, d)
    scale = 1.0 / math.sqrt(d)
    
    # 标准 Attention
    scores = torch.matmul(Q, K.transpose(0, 1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out_standard = torch.matmul(attn, V)
    
    # Flash Attention（教学版）
    out_flash = flash_attention_naive(Q, K, V, block_size=16)
    
    max_diff = (out_standard - out_flash).abs().max().item()
    print(f"标准 Attention vs Flash Attention 最大差异: {max_diff:.2e}")
    print(f"数值等价验证: {'✅ 通过' if max_diff < 1e-4 else '❌ 失败'}")

verify_flash_vs_standard()
```

### 4.4 使用 PyTorch 内置 FlashAttention

```python
import torch
import torch.nn.functional as F

# PyTorch 2.0+ 自动选择最优 Attention 实现（包括 FlashAttention）
def use_pytorch_flash_attention():
    B, H, T, d_k = 2, 8, 512, 64
    
    Q = torch.randn(B, H, T, d_k, dtype=torch.float16)  # FP16 可使用 FlashAttention
    K = torch.randn(B, H, T, d_k, dtype=torch.float16)
    V = torch.randn(B, H, T, d_k, dtype=torch.float16)
    
    # PyTorch 内置，自动使用 FlashAttention（如果 CUDA 可用且满足条件）
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,        # 启用 FlashAttention
        enable_math=True,         # 备选：标准实现
        enable_mem_efficient=True # 备选：内存高效实现
    ):
        output = F.scaled_dot_product_attention(Q, K, V)
    
    print(f"PyTorch 内置 SDPA 输出形状: {output.shape}")

# 注意：需要 CUDA 环境运行，此处仅展示接口
# use_pytorch_flash_attention()
print("PyTorch 2.0+ 的 F.scaled_dot_product_attention 自动选择最优实现")
```

---

## 五、三种变体对比实验

```python
import torch
import time

def benchmark_variants(T: int, d_model: int = 256, num_heads: int = 8):
    """对比三种注意力变体的实际性能"""
    d_k = d_model // num_heads
    scale = 1.0 / math.sqrt(d_k)
    
    Q = torch.randn(1, num_heads, T, d_k)
    K = torch.randn(1, num_heads, T, d_k)
    V = torch.randn(1, num_heads, T, d_k)
    
    # 标准 Attention
    t0 = time.time()
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out_std = torch.matmul(attn, V)
    t_std = (time.time() - t0) * 1000
    
    # 内存占用（注意力矩阵大小）
    mem_std = num_heads * T * T * 4 / 1024**2  # MB
    
    print(f"\nT={T}: 标准注意力 {t_std:.1f}ms, 注意力矩阵 {mem_std:.1f}MB")

for T in [128, 512, 1024, 2048]:
    benchmark_variants(T)
```

---

## 六、小结

| 变体 | 适用场景 | 核心优势 | 主要限制 |
|------|---------|---------|---------|
| **Sparse** | 文档、代码等结构化序列 | 保持精确性，显著减少计算 | 需要人工设计稀疏模式 |
| **Linear** | 超长序列（n > 10k）；流式推理 | 线性复杂度，支持 RNN 形式 | 表达能力不如 softmax |
| **FlashAttention** | 所有场景（几乎无缝替换） | 精确等价，实践最快 | 需要 CUDA，实现复杂 |

**工程建议**：
- 先用 `F.scaled_dot_product_attention`（PyTorch 2.0+），它会自动调用最优实现
- 超长上下文（>8k）考虑 Sparse 或 Linear Attention
- 追求极致速度用 Flash Attention 2/3（Tri Dao 的 CUDA 实现）

下一篇 D6 将从零手写完整的 Attention 实现，包括所有细节和边界处理。

---

*参考文献：Dao et al. (2022) "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"；Child et al. (2019) "Generating Long Sequences with Sparse Transformers"；Katharopoulos et al. (2020) "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"*
