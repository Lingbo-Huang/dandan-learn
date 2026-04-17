# Week 2 学习计划：Attention 机制深度推导

> 系列：大模型线（LLM Track）| 难度：⭐⭐⭐⭐ | 预计时长：10-14 小时

---

## 学习目标

完成本周学习后，你将能够：

1. **直觉理解**：用自己的语言解释为什么 Attention 机制会出现，它解决了什么问题
2. **数学推导**：从零推导 Scaled Dot-Product Attention 的每一步，包括缩放因子的来源
3. **Multi-Head 理解**：解释多头的动机、数学形式，以及与单头的区别
4. **复杂度分析**：对时间复杂度、空间复杂度和序列长度的关系了然于胸
5. **代码实现**：用 PyTorch 从零实现完整的 Multi-Head Attention（不使用 `nn.MultiheadAttention`）
6. **变体认知**：了解 Linear Attention、Flash Attention、GQA 等主流变体的动机与权衡
7. **综合应用**：在一个小型文本任务上验证 Attention 实现的正确性

---

## 文件导航

| 文件 | 主题 | 难度 | 预计时长 |
|------|------|------|---------|
| `llm-w2-attention-intuition.md` | 直觉与动机 | ⭐⭐ | 1-2h |
| `llm-w2-scaled-dot-product.md` | Scaled Dot-Product 推导 | ⭐⭐⭐ | 2-3h |
| `llm-w2-multi-head-attention.md` | Multi-Head Attention | ⭐⭐⭐⭐ | 2-3h |
| `llm-w2-attention-complexity.md` | 复杂度分析 | ⭐⭐⭐ | 1-2h |
| `llm-w2-attention-variants.md` | Attention 变体 | ⭐⭐⭐⭐ | 2h |
| `llm-w2-attention-code.md` | 完整代码实现 | ⭐⭐⭐⭐⭐ | 2-3h |
| `llm-w2-capstone.md` | 综合项目 | ⭐⭐⭐⭐⭐ | 2h |

---

## 每日学习节奏建议

### Day 1（周一）— 建立直觉

**目标**：理解 Attention 为什么存在，有哪些前置问题

- 阅读 `llm-w2-attention-intuition.md`
- 动手：手绘一个 seq2seq + attention 的计算图
- 思考题：如果没有 Attention，RNN 的 bottleneck 在哪？

**检验**：能向一个不懂 ML 的朋友解释"注意力机制"

---

### Day 2（周二）— 数学推导

**目标**：掌握 Scaled Dot-Product Attention 的每一个细节

- 精读 `llm-w2-scaled-dot-product.md`
- 动手：用纸笔推导一次，不看笔记
- 重点：缩放因子 $\frac{1}{\sqrt{d_k}}$ 的必要性

**检验**：能给出 softmax 前后的维度变化

---

### Day 3（周三）— Multi-Head 与代码

**目标**：Multi-Head 的数学 + 开始写代码

- 阅读 `llm-w2-multi-head-attention.md`
- 开始 `llm-w2-attention-code.md`，完成 `ScaledDotProductAttention` 类
- 运行代码，跑通单元测试

**检验**：能解释为什么 `h * d_k = d_model`

---

### Day 4（周四）— 复杂度 + 变体

**目标**：理解效率瓶颈，了解工业界解法

- 阅读 `llm-w2-attention-complexity.md`
- 阅读 `llm-w2-attention-variants.md`（重点看 GQA 和 Flash Attention 的动机）
- 完成代码中的 `MultiHeadAttention` 类

**检验**：能说出标准 Attention 的空间复杂度瓶颈在哪

---

### Day 5（周五/周末）— 综合项目

**目标**：端到端验证理解

- 完成 `llm-w2-capstone.md` 中的综合项目
- 让模型在玩具数据集上收敛，可视化 attention weights
- 写下本周 reflection

**检验**：能跑通完整训练循环，attention heatmap 有意义

---

## 前置知识回顾

在开始前，确保你熟悉：

- **矩阵乘法**：$(A \cdot B)_{ij} = \sum_k A_{ik} B_{kj}$，维度追踪
- **Softmax**：$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$
- **Dropout 机制**：训练时随机置零，推理时不使用
- **PyTorch 基础**：`torch.Tensor`、`einsum`、`nn.Module`

---

## 本周重点：你会记住的那几件事

> "Attention 就是一个加权平均。权重是 query 和 key 的相似度，value 是被平均的东西。"

> "缩放因子 $\sqrt{d_k}$ 是为了防止 dot product 进入 softmax 的饱和区。"

> "多头是为了让模型同时关注不同子空间里的不同关系——就像你可以同时用语法、语义、指代等多个维度理解句子。"

---

## 参考资源

- 📄 **Attention Is All You Need**（Vaswani et al., 2017）— 必读原论文
- 📘 **The Illustrated Transformer**（Jay Alammar）— 可视化理解
- 📗 **The Annotated Transformer**（Harvard NLP）— 逐行注释代码
- 📺 **Andrej Karpathy: Let's build GPT**（YouTube）— 动手最佳资源
- 📄 **Flash Attention**（Dao et al., 2022）— IO 感知 Attention
- 📄 **GQA: Training Generalized Multi-Query Transformer**（Ainslie et al., 2023）

---

## 小结

Attention 机制是 Transformer 的核心。Week 1 你建立了 LLM 的全局视图；本周你将深入到最关键的计算模块，从直觉到推导到代码，完整打通 Attention 的每一层理解。

**本周结束时的标志**：你能在 30 分钟内，从零写出一个 correct + efficient 的 Multi-Head Attention，并且知道为什么每一行代码是这样写的。
