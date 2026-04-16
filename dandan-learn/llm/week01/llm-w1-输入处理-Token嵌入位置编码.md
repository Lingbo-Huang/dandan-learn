# Day 2：Transformer 鸟瞰——整体结构与信息流

---

## 学习目标

- 理解 Transformer 的整体组成（Encoder 栈 + Decoder 栈）
- 能描述一个 token 从输入到输出的完整流动路径
- 理解 Residual Connection（残差连接）与 Layer Normalization 的作用

---

## 一、核心知识点

### 1.1 Transformer 整体结构

原始论文的 Transformer 是 **Encoder-Decoder** 架构，用于机器翻译任务。

```
输入序列 (src)              输出序列 (tgt, 训练时右移一位)
    ↓                               ↓
[Input Embedding]          [Output Embedding]
    ↓                               ↓
[Positional Encoding]      [Positional Encoding]
    ↓                               ↓
┌─────────────┐            ┌─────────────────────┐
│  Encoder    │            │  Decoder            │
│  Block × N  │──────────→│  Block × N          │
└─────────────┘            └─────────────────────┘
                                    ↓
                           [Linear + Softmax]
                                    ↓
                            预测下一个 token
```

### 1.2 Encoder Block 结构

每个 Encoder Block 包含两个子层：

```
输入 x
  │
  ├→ [Multi-Head Self-Attention] → 残差 Add → LayerNorm → 中间结果 x'
  │                                                              │
  └→ [Feed-Forward Network (FFN)]    → 残差 Add → LayerNorm → 输出
```

公式表示：
```
x' = LayerNorm(x + MultiHeadAttn(x, x, x))
output = LayerNorm(x' + FFN(x'))
```

### 1.3 Decoder Block 结构

Decoder 每个 Block 有**三个**子层：

1. **Masked Multi-Head Self-Attention**（防止看到未来 token）
2. **Cross-Attention**（Query 来自 Decoder，Key/Value 来自 Encoder 输出）
3. **Feed-Forward Network**

```
Decoder 输入 (已生成 token)
  │
  ├→ [Masked Self-Attention] → Add & Norm
  │                               │
  ├← Encoder 输出 ──→ [Cross-Attention] → Add & Norm
  │                               │
  └→ [FFN]                → Add & Norm → 输出
```

### 1.4 残差连接（Residual Connection）

**为什么需要残差连接？**

深层网络训练时梯度容易消失。残差连接让梯度"走捷径"：

```
output = F(x) + x
```

即使 F(x) 的梯度很小，`∂output/∂x = ∂F(x)/∂x + 1` 仍有 1 这个保底梯度，避免梯度消失。

### 1.5 Layer Normalization

与 Batch Normalization 在 batch 维度归一化不同，LayerNorm 在**特征维度**归一化：

```python
# 对最后一个维度（特征维度）归一化
LayerNorm(x) = γ * (x - mean(x)) / (std(x) + ε) + β
```

优势：不依赖 batch size，适合序列模型。

---

## 二、数据流推导示例

以机器翻译 "I love cats" → "我爱猫" 为例，追踪数据流：

**Step 1：Encoder 端**
```
["I", "love", "cats"]
→ Token IDs: [100, 2004, 5765]
→ Embeddings: shape (3, 512)    # 3个token, 512维
→ + Positional Encoding: shape (3, 512)
→ Encoder Block × 6
→ Encoder Output: shape (3, 512)  # 保留全部位置信息
```

**Step 2：Decoder 端（生成"我"）**
```
BOS Token (开始标记)
→ Embedding: shape (1, 512)
→ + Positional Encoding
→ Decoder Block × 6（内部用 Cross-Attention 读取 Encoder Output）
→ Linear 层: shape (1, vocab_size)
→ Softmax → 概率分布 → 取最大值 → "我"
```

**Step 3：自回归生成**
```
BOS → "我" → "我爱" → "我爱猫" → EOS
```

---

## 三、动手练习

### 练习 1：架构图手绘

不看资料，手绘一个 Encoder Block 的结构图，标注：
- 两个子层的名称
- 残差连接的位置
- LayerNorm 的位置

### 练习 2：代码阅读——PyTorch 内置 Transformer

```python
import torch
import torch.nn as nn

# PyTorch 内置 Transformer 模块
model = nn.Transformer(
    d_model=512,        # embedding 维度
    nhead=8,            # 注意力头数
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1
)

# 查看模型结构
print(model)
print(f"\n总参数量: {sum(p.numel() for p in model.parameters()):,}")

# 构造随机输入测试
src = torch.randn(10, 32, 512)  # (seq_len, batch, d_model)
tgt = torch.randn(20, 32, 512)
output = model(src, tgt)
print(f"\n输入 src shape: {src.shape}")
print(f"输入 tgt shape: {tgt.shape}")
print(f"输出 shape: {output.shape}")
```

运行并理解每个维度的含义。

### 练习 3：思考题

- Encoder 的输入和输出 shape 是否相同？为什么？
- 为什么 Decoder 需要 Masked Self-Attention？如果不 mask 会有什么问题？

---

## 四、小结

| 项目 | 内容 |
|------|------|
| 今日完成 | Transformer 整体架构 + 数据流路径 + 残差&归一化 |
| 核心认知 | Encoder 提取特征；Decoder 自回归生成；残差让深层网络可训练 |
| 明日预告 | D3 深入 Attention 机制核心：Q/K/V 与 Scaled Dot-Product Attention |

> 💡 **今日关键问题**：Encoder 和 Decoder 的 Self-Attention 有什么本质区别？（Mask！）
