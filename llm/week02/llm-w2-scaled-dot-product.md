# D2：Scaled Dot-Product Attention 完整推导

> **Week 2 · Day 2** | 大模型学习路线

---

## 一、从基础概念到形式化定义

Scaled Dot-Product Attention 是 Transformer 的核心计算单元，由 Vaswani et al. (2017) 在《Attention Is All You Need》中正式提出。

**完整公式**：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$：查询矩阵（$n$ 个 Query，每个维度为 $d_k$）
- $K \in \mathbb{R}^{m \times d_k}$：键矩阵（$m$ 个 Key，每个维度为 $d_k$）
- $V \in \mathbb{R}^{m \times d_v}$：值矩阵（$m$ 个 Value，每个维度为 $d_v$）
- 输出 $\in \mathbb{R}^{n \times d_v}$

---

## 二、逐步推导

### 2.1 第一步：计算原始注意力分数

对于 Query 矩阵 $Q$ 中的第 $i$ 个查询向量 $q_i \in \mathbb{R}^{d_k}$，计算它与所有 Key 的点积：

$$e_{ij} = q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il} \cdot k_{jl}$$

矩阵化表示为：

$$E = QK^\top \in \mathbb{R}^{n \times m}$$

其中 $E_{ij}$ 表示第 $i$ 个 Query 与第 $j$ 个 Key 的相似度分数。

### 2.2 第二步：缩放（Scaling）

将原始分数除以 $\sqrt{d_k}$：

$$\tilde{E} = \frac{QK^\top}{\sqrt{d_k}}$$

**为什么这个缩放因子至关重要？**

设 $q_i, k_j \overset{\text{iid}}{\sim} \mathcal{N}(0, 1)$（初始化时的合理假设），则点积 $e_{ij} = q_i \cdot k_j$ 的统计性质为：

$$\mathbb{E}[e_{ij}] = \mathbb{E}\!\left[\sum_{l=1}^{d_k} q_{il} k_{jl}\right] = \sum_{l=1}^{d_k} \mathbb{E}[q_{il}]\mathbb{E}[k_{jl}] = 0$$

$$\text{Var}(e_{ij}) = \text{Var}\!\left(\sum_{l=1}^{d_k} q_{il} k_{jl}\right) = \sum_{l=1}^{d_k} \text{Var}(q_{il} k_{jl}) = \sum_{l=1}^{d_k} 1 = d_k$$

因此 $\text{Std}(e_{ij}) = \sqrt{d_k}$。

**问题**：当 $d_k$ 较大时（如 $d_k = 512$），分数的标准差为 $\sqrt{512} \approx 22.6$，数值极大，导致 softmax 的输出接近 one-hot 分布。

**Softmax 梯度分析**：

设 $p_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$，则：

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

当 $p$ 接近 one-hot 时，$p_i \approx 1$ 而 $p_j \approx 0$，梯度趋近于 0，**梯度消失**。

缩放后：$\tilde{e}_{ij} = e_{ij} / \sqrt{d_k}$，方差归一化为 1，softmax 工作在合理的数值范围内。

### 2.3 第三步：Softmax 归一化

对每一行应用 Softmax：

$$A_{ij} = \frac{\exp(\tilde{E}_{ij})}{\sum_{l=1}^{m} \exp(\tilde{E}_{il})}$$

矩阵形式：

$$A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times m}$$

性质：
- $A_{ij} \geq 0$（非负）
- $\sum_{j=1}^{m} A_{ij} = 1$（行归一化）
- $A_{ij}$ 表示第 $i$ 个 Query 对第 $j$ 个 Key-Value 对的注意力权重

### 2.4 第四步：加权求和

用注意力权重对 Value 矩阵加权求和：

$$\text{Output} = AV \in \mathbb{R}^{n \times d_v}$$

第 $i$ 个输出向量为：

$$\text{output}_i = \sum_{j=1}^{m} A_{ij} \cdot v_j$$

**直觉**：输出是所有 Value 向量的**加权平均**，权重由 Query 和 Key 的相似度决定。

---

## 三、数值稳定性：Log-Sum-Exp Trick

朴素 Softmax 在数值上不稳定：当分数很大时，$\exp$ 会溢出。

**稳定化处理**：

$$\text{softmax}(z)_i = \frac{\exp(z_i - \max_j z_j)}{\sum_j \exp(z_j - \max_j z_j)}$$

数学上等价（分子分母同除以 $\exp(\max_j z_j)$），但数值上：
- 最大值归零，避免上溢
- 其他值为负数，避免极端小的 exp 值（虽然可能下溢到 0，但不影响结果）

---

## 四、Mask 机制的数学表示

### 4.1 Padding Mask

对于 padding 位置（无效 token），令对应分数为 $-\infty$：

$$\tilde{E}_{ij} = \begin{cases} \frac{q_i \cdot k_j}{\sqrt{d_k}} & \text{if } j \text{ is valid} \\ -\infty & \text{if } j \text{ is padding} \end{cases}$$

由于 $\exp(-\infty) = 0$，softmax 后 padding 位置的权重为 0。

### 4.2 Causal Mask（因果遮盖）

解码器中，位置 $i$ 只能关注位置 $\leq i$ 的信息（防止"看未来"）：

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

$$\tilde{E} = \frac{QK^\top}{\sqrt{d_k}} + M$$

用矩阵表示，$M$ 是上三角（不含对角线）为 $-\infty$，其余为 0 的矩阵。

---

## 五、与其他注意力机制的对比

| 机制 | 分数函数 $f(q, k)$ | 复杂度 | 特点 |
|------|-------------------|--------|------|
| **Dot-Product** | $q^\top k$ | $O(d)$ | 简单，但不稳定 |
| **Scaled Dot-Product** | $q^\top k / \sqrt{d_k}$ | $O(d)$ | Transformer 标准 |
| **Additive (Bahdanau)** | $v^\top \tanh(W_1 q + W_2 k)$ | $O(d)$ | 参数更多，有时更灵活 |
| **Cosine** | $q^\top k / (\|q\| \|k\|)$ | $O(d)$ | 强制归一化 |

Scaled Dot-Product 在效率和效果上取得了最好的平衡，成为主流选择。

---

## 六、梯度流分析

理解 Attention 的反向传播对调试模型很重要。

设损失为 $\mathcal{L}$，输出为 $O = AV$，则：

$$\frac{\partial \mathcal{L}}{\partial V} = A^\top \frac{\partial \mathcal{L}}{\partial O}$$

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial O} V^\top$$

$$\frac{\partial \mathcal{L}}{\partial \tilde{E}} = \text{softmax\_backward}(A, \frac{\partial \mathcal{L}}{\partial A})$$

其中 Softmax 的反向传播为：

$$\frac{\partial \mathcal{L}}{\partial \tilde{E}_i} = A_i \odot \left(\frac{\partial \mathcal{L}}{\partial A_i} - \left(\frac{\partial \mathcal{L}}{\partial A_i}^\top A_i\right)\right)$$

---

## 七、完整 PyTorch 实现（含完整推导）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention 完整实现
    
    完整公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
    
    Args:
        Q: (..., seq_len_q, d_k) 查询矩阵
        K: (..., seq_len_k, d_k) 键矩阵
        V: (..., seq_len_k, d_v) 值矩阵
        mask: (..., seq_len_q, seq_len_k) Bool mask, True 表示遮盖该位置
        dropout_p: Dropout 概率（训练时用于注意力权重的随机dropout）
        scale: 自定义缩放因子（默认 1/sqrt(d_k)）
    
    Returns:
        output: (..., seq_len_q, d_v)
        attention_weights: (..., seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)
    
    # Step 1: 计算缩放因子
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)
    
    # Step 2: 计算注意力分数 (Q @ K^T)
    # (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Step 3: 应用 Mask（在 softmax 之前）
    if mask is not None:
        # 将被遮盖位置设为 -inf，softmax 后变为 0
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Step 4: Softmax 归一化（沿 Key 维度）
    # 使用 dim=-1 对每个 Query 归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 处理全为 -inf 的行（如全 padding 行），避免 NaN
    # 这些行 softmax 后会是 NaN，用 0 替代
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    
    # Step 5: Dropout（仅训练时）
    if dropout_p > 0.0 and torch.is_grad_enabled():
        attention_weights = F.dropout(attention_weights, p=dropout_p)
    
    # Step 6: 加权求和 Value
    # (..., seq_len_q, seq_len_k) @ (..., seq_len_k, d_v) -> (..., seq_len_q, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


# ==================== 数值验证 ====================

def verify_attention_properties():
    """验证 Scaled Dot-Product Attention 的数学性质"""
    torch.manual_seed(42)
    
    B, T, d_k, d_v = 2, 6, 64, 128
    Q = torch.randn(B, T, d_k)
    K = torch.randn(B, T, d_k)
    V = torch.randn(B, T, d_v)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("=" * 50)
    print("属性验证")
    print("=" * 50)
    
    # 验证形状
    print(f"\n✅ 输出形状: {output.shape}  (期望: {(B, T, d_v)})")
    print(f"✅ 权重形状: {weights.shape}  (期望: {(B, T, T)})")
    
    # 验证权重非负
    print(f"\n✅ 权重非负: {(weights >= 0).all().item()}")
    
    # 验证行归一化
    row_sums = weights.sum(dim=-1)
    print(f"✅ 行归一化 (所有行之和≈1): max偏差={abs(row_sums - 1).max().item():.6f}")
    
    # 验证缩放因子的效果
    # 未缩放版本
    scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))
    scores_scaled = scores_unscaled / math.sqrt(d_k)
    
    print(f"\n📊 点积分数统计（维度 d_k={d_k}）：")
    print(f"   未缩放 - 均值: {scores_unscaled.mean().item():.3f}, 标准差: {scores_unscaled.std().item():.3f}")
    print(f"   已缩放 - 均值: {scores_scaled.mean().item():.3f}, 标准差: {scores_scaled.std().item():.3f}")
    print(f"   理论缩放后标准差 ≈ 1.0（实际: {scores_scaled.std().item():.3f}）")
    
    return output, weights


output, weights = verify_attention_properties()


# ==================== Causal Mask 演示 ====================

def demo_causal_mask(seq_len: int = 5):
    """演示因果遮盖"""
    # 创建上三角 mask（不含对角线）
    # True 表示需要遮盖的位置（未来位置）
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool), 
        diagonal=1
    )
    
    print("\n因果遮盖矩阵（True=遮盖，False=允许访问）：")
    print(causal_mask.int())
    
    # 应用到注意力
    Q = K = V = torch.randn(1, seq_len, 32)
    _, weights_causal = scaled_dot_product_attention(
        Q, K, V, 
        mask=causal_mask.unsqueeze(0)
    )
    
    print("\n注意力权重（带因果遮盖，行=Query，列=Key）：")
    print(weights_causal[0].detach().numpy().round(3))
    print("（上三角应为 0，因为不能看到未来）")

demo_causal_mask()
```

---

## 八、与 PyTorch 内置实现对比

```python
import torch
import math

# PyTorch 2.0+ 内置的 scaled_dot_product_attention（优化版，支持 FlashAttention）
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa

torch.manual_seed(0)
B, H, T, d_k = 2, 4, 10, 64  # (batch, heads, seq, dim)

Q = torch.randn(B, H, T, d_k)
K = torch.randn(B, H, T, d_k)
V = torch.randn(B, H, T, d_k)

# 我们的实现
out_ours, _ = scaled_dot_product_attention(Q, K, V)

# PyTorch 官方实现
out_torch = torch_sdpa(Q, K, V, dropout_p=0.0)

# 比较（应几乎完全一致）
max_diff = (out_ours - out_torch).abs().max().item()
print(f"最大差异: {max_diff:.2e}  (应 < 1e-5)")
```

---

## 九、复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| $QK^\top$ 计算 | $O(n^2 d_k)$ | $O(n^2)$ |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| $AV$ 计算 | $O(n^2 d_v)$ | $O(n d_v)$ |
| **总计** | $O(n^2 d)$ | $O(n^2)$ |

其中 $n$ 为序列长度，$d = \max(d_k, d_v)$。

**关键瓶颈**：$O(n^2)$ 的空间复杂度——注意力矩阵 $A$ 随序列长度平方增长，这是 FlashAttention 等优化工作的核心动机（D4 和 D5 会详细讨论）。

---

## 十、小结

完整推导路径：

$$Q, K, V \xrightarrow{QK^\top} E \xrightarrow{\div\sqrt{d_k}} \tilde{E} \xrightarrow{+\text{Mask}} \tilde{E}_{\text{masked}} \xrightarrow{\text{softmax}} A \xrightarrow{AV} \text{Output}$$

核心数学洞见：
1. **点积度量相似度**：利用向量夹角衡量 Query 和 Key 的匹配程度
2. **缩放防止梯度消失**：除以 $\sqrt{d_k}$ 保持方差稳定
3. **Softmax 赋予概率解释**：将原始分数转化为权重分布
4. **加权聚合传递信息**：最终输出是 Value 的加权组合

下一篇 D3 将在此基础上介绍多头注意力：通过多个平行的注意力"头"，让模型能同时关注不同类型的依赖关系。

---

*参考文献：Vaswani et al. (2017) "Attention Is All You Need"；Graves (2013) "Generating Sequences With Recurrent Neural Networks"*
