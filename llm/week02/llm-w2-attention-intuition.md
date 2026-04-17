# D1：Attention 的直觉理解——查询/键/值的隐喻

> **Week 2 · Day 1** | 大模型学习路线

---

## 一、为什么需要注意力机制？

在 RNN/LSTM 时代，序列模型的核心矛盾是：**信息瓶颈**。

无论输入序列有多长，编码器都要把所有语义压缩进一个固定维度的"隐状态向量"，然后解码器从这个向量还原目标序列。这就像让一个人在脑子里背下整本书，然后用一句话复述——必然丢失大量细节。

2015 年 Bahdanau 等人提出的 Attention 机制，给了解码器一个"查字典"的能力：解码每个词时，不再依赖那个唯一的压缩向量，而是**动态地回头看输入序列中与当前最相关的部分**。

这个思想彻底改变了 NLP 格局，并最终催生了 Transformer。

---

## 二、图书馆隐喻：Q/K/V 的直觉

理解 Attention 最好的方式，是把它类比成**图书馆检索系统**：

| 概念 | 图书馆类比 | 实际含义 |
|------|-----------|---------|
| **Query (Q)** | 你的检索需求 | 当前时刻想"查"什么 |
| **Key (K)** | 每本书的索引标签 | 每个位置对外"宣称"自己是什么 |
| **Value (V)** | 书的实际内容 | 每个位置真正携带的信息 |

**检索流程**：

1. 你带着一个 Query（我想找机器学习相关的书）走进图书馆
2. 系统把你的 Query 和每本书的 Key 做比对，计算相似度（注意力分数）
3. 把所有书的 Value 按相似度**加权求和**，返回给你

写成公式就是：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

其中 $d_k$ 是 Key 的维度，$\sqrt{d_k}$ 是缩放因子（后面详细讲为什么要除以它）。

---

## 三、从词语到向量：具体化隐喻

假设我们在翻译句子：

> "The cat sat on the mat because **it** was tired."

当模型需要确定"**it**"指代什么时，它会：

- 生成一个 Query："这个代词指什么？"
- 把 Query 和每个词的 Key 比对：`cat` 的 Key 得分高，`mat` 的 Key 得分低
- 把高分词的 Value 聚合：返回的结果主要来自"cat"

这个过程让模型**学会了依赖关系**，而不需要任何规则。

---

## 四、注意力分数的计算细节

### 4.1 点积计算

最常用的相似度度量是**缩放点积（Scaled Dot-Product）**：

$$\text{score}(q, k) = \frac{q \cdot k}{\sqrt{d_k}}$$

直觉：点积越大，两个向量越"方向一致"，说明 Query 和 Key 越匹配。

### 4.2 为什么要除以 $\sqrt{d_k}$？

当维度 $d_k$ 很大时，点积的数值会随维度增长而变大（期望值为 $d_k$，标准差为 $\sqrt{d_k}$）。

数值太大会导致 softmax 的梯度极小（梯度消失），模型很难学习。

除以 $\sqrt{d_k}$ 后，方差归一到 1，softmax 工作在更合理的数值范围内。

**数学验证**：设 $q, k \sim \mathcal{N}(0,1)$，则：

$$\text{Var}(q \cdot k) = \text{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k$$

所以 $\text{Std}(q \cdot k) = \sqrt{d_k}$，除以 $\sqrt{d_k}$ 后标准差变为 1。

### 4.3 Softmax：权重归一化

$$\alpha_i = \frac{\exp(\text{score}_i)}{\sum_j \exp(\text{score}_j)}$$

Softmax 把分数转化为概率分布，所有权重之和为 1，代表"注意力分配比例"。

---

## 五、自注意力（Self-Attention）

在 Transformer 中，Q、K、V 都来自同一个输入序列——这称为**自注意力**（Self-Attention）。

**每个词都同时扮演三个角色**：
- 作为 Query 去"询问"其他词
- 作为 Key 被其他词"查询"
- 作为 Value 提供自己的语义信息

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

其中 $X \in \mathbb{R}^{n \times d_{\text{model}}}$ 是输入序列，$W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ 是可学习的投影矩阵。

---

## 六、PyTorch 代码：最简单的自注意力

```python
import torch
import torch.nn.functional as F
import math

def simple_self_attention(x: torch.Tensor) -> torch.Tensor:
    """
    最简单的自注意力（无投影矩阵版，直接用 x 作为 Q/K/V）
    
    Args:
        x: shape (batch_size, seq_len, d_model)
    Returns:
        output: shape (batch_size, seq_len, d_model)
    """
    d_k = x.size(-1)
    
    # Q = K = V = x （简化版，不学习投影）
    Q = K = V = x
    
    # 计算注意力分数 (batch, seq_len, seq_len)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    
    # Softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)
    
    # 加权聚合 Value
    output = torch.bmm(attn_weights, V)
    
    return output, attn_weights


# 测试
batch_size, seq_len, d_model = 2, 5, 64
x = torch.randn(batch_size, seq_len, d_model)
out, weights = simple_self_attention(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {out.shape}")
print(f"注意力权重形状: {weights.shape}")
print(f"注意力权重（第一个样本，第一个词）: {weights[0, 0]}")
print(f"权重之和: {weights[0, 0].sum():.4f}")  # 应该 ≈ 1.0
```

输出示例：
```
输入形状: torch.Size([2, 5, 64])
输出形状: torch.Size([2, 5, 64])
注意力权重形状: torch.Size([2, 5, 5])
注意力权重（第一个样本，第一个词）: tensor([0.1823, 0.2341, 0.1902, 0.2015, 0.1919])
权重之和: 1.0000
```

---

## 七、带投影的完整自注意力

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """
    单头自注意力（含可学习的 Q/K/V 投影）
    """
    def __init__(self, d_model: int, d_k: int = None):
        super().__init__()
        self.d_k = d_k or d_model
        
        # 线性投影：将 x 映射到 Q, K, V 空间
        self.W_Q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, self.d_k, bias=False)
        
        # 输出投影
        self.W_O = nn.Linear(self.d_k, d_model, bias=False)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len)，True 表示需要被遮盖的位置
        """
        B, T, _ = x.shape
        
        # 投影得到 Q, K, V
        Q = self.W_Q(x)  # (B, T, d_k)
        K = self.W_K(x)  # (B, T, d_k)
        V = self.W_V(x)  # (B, T, d_k)
        
        # 计算注意力分数
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)  # (B, T, T)
        
        # 应用 mask（如因果遮盖）
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权聚合
        context = torch.bmm(attn_weights, V)  # (B, T, d_k)
        
        # 输出投影
        output = self.W_O(context)  # (B, T, d_model)
        
        return output, attn_weights


# 测试
model = SelfAttention(d_model=256, d_k=64)
x = torch.randn(2, 10, 256)
out, weights = model(x)
print(f"输出形状: {out.shape}")  # (2, 10, 256)

# 因果遮盖（解码器用）
causal_mask = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
causal_mask = causal_mask.unsqueeze(0)  # (1, 10, 10)
out_masked, weights_masked = model(x, mask=causal_mask)
print(f"带因果遮盖的输出形状: {out_masked.shape}")
```

---

## 八、可视化注意力权重

注意力权重矩阵可以直观展示模型"在看什么"：

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(weights: torch.Tensor, tokens: list, title: str = "Attention Weights"):
    """
    可视化注意力权重热力图
    
    Args:
        weights: (seq_len, seq_len) 注意力权重
        tokens: token 列表（字符串）
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        weights.detach().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        ax=ax
    )
    
    ax.set_xlabel("Keys (被关注的位置)")
    ax.set_ylabel("Queries (当前位置)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# 示例
tokens = ["The", "cat", "sat", "on", "mat"]
model = SelfAttention(d_model=64, d_k=32)
x = torch.randn(1, 5, 64)
_, weights = model(x)
visualize_attention(weights[0], tokens)
```

---

## 九、直觉总结：注意力的三个精髓

### 9.1 上下文感知
每个词的表示**不是固定的**，而是由其在当前上下文中的语义关系动态决定。"bank"在"river bank"和"bank account"中的表示会完全不同。

### 9.2 全局依赖
与 RNN 相比，Attention 可以一步直接连接序列中任意两个位置，不存在长程依赖衰减问题。

### 9.3 可解释性
注意力权重矩阵提供了一定程度的可解释性——虽然不完美，但可以帮助理解模型"在关注什么"。

---

## 十、小结

| 概念 | 核心理解 |
|------|---------|
| Query | 当前位置"想要什么" |
| Key | 每个位置"提供什么标签" |
| Value | 每个位置"实际携带的信息" |
| 注意力分数 | Query 和 Key 的匹配程度 |
| 注意力权重 | 经 Softmax 归一化的分配比例 |
| 输出 | Value 按权重加权求和的聚合结果 |

下一篇 D2 将深入推导 Scaled Dot-Product Attention 的完整数学过程，解释缩放因子的理论依据和数值稳定性问题。

---

*参考文献：Bahdanau et al. (2015) "Neural Machine Translation by Jointly Learning to Align and Translate"；Vaswani et al. (2017) "Attention Is All You Need"*
