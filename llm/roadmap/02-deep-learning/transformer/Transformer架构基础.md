# Transformer架构基础

## 1. 引言

在RNN及其变体（LSTM, GRU）统治序列建模领域多年后，一种全新的架构——**Transformer**，在2017年论文《Attention is All You Need》中被提出。它彻底摒弃了循环和卷积结构，仅基于**注意力机制**（Attention Mechanism）构建，取得了革命性的成功。

**Transformer的优势：**
1.  **并行化：** 与RNN不同，Transformer不需要按顺序处理序列元素，可以高度并行化，训练速度大大加快。
2.  **长距离依赖：** 注意力机制允许模型直接关注序列中任意两个位置的关系，有效捕捉长距离依赖，避免了RNN的梯度消失问题。
3.  **强大的建模能力：** 在机器翻译、文本生成等任务上取得了SOTA（State-of-the-Art）性能。

Transformer架构不仅是现代NLP（如BERT, GPT系列）的基石，也被广泛应用于计算机视觉（Vision Transformer）等领域，成为当代深度学习的支柱之一。

## 2. 核心组件：注意力机制（Attention Mechanism）

### 2.1 从RNN到Attention

在传统的Seq2Seq模型（如RNN Encoder-Decoder）中，Encoder将整个输入序列编码成一个**固定长度的上下文向量**（Context Vector），Decoder再基于这个向量生成输出序列。当输入序列很长时，这个固定向量难以有效承载所有信息，导致性能瓶颈。

**注意力机制**的核心思想是：Decoder在生成每个输出词时，不应该只关注一个固定的上下文向量，而应该根据当前已生成的部分和即将生成的目标，**有选择地关注**输入序列的不同部分。

### 2.2 缩放点积注意力（Scaled Dot-Product Attention）

这是Transformer中使用的注意力计算方式。

**输入：**
*   **查询**（Query, Q）：形状 `(L_q, d_k)`
*   **键**（Key, K）：形状 `(L_k, d_k)`
*   **值**（Value, V）：形状 `(L_k, d_v)`
*   `L_q` 是查询序列的长度，`L_k` 是键/值序列的长度，`d_k` 和 `d_v` 是特征维度。

**计算步骤：**
1.  **计算注意力分数（Compatibility）：** 计算Query和Key之间的相似度。Transformer使用**点积**（Dot Product）。
    ```
    Scores = Q * K^T  // 形状 (L_q, L_k)
    ```
2.  **缩放（Scaling）：** 为了稳定梯度，将点积结果除以 `sqrt(d_k)`。
    ```
    Scaled_Scores = Scores / sqrt(d_k)
    ```
3.  **Softmax归一化：** 对每一行（对应一个查询）进行Softmax，得到注意力权重（Attention Weights），表示对每个键值对的关注程度。
    ```
    Attention_Weights = Softmax(Scaled_Scores) // 形状 (L_q, L_k)
    ```
4.  **加权求和（Output）：** 使用注意力权重对Value进行加权求和，得到最终的注意力输出。
    ```
    Output = Attention_Weights * V // 形状 (L_q, d_v)
    ```

**数学公式：**
```
Attention(Q, K, V) = Softmax(Q * K^T / sqrt(d_k)) * V
```

**为什么需要缩放？**
当 `d_k` 较大时，`Q*K^T` 的点积值也会变得很大，这会导致Softmax函数的输入进入饱和区（梯度极小），使得模型训练困难。除以 `sqrt(d_k)` 可以将方差稳定在1左右，缓解这个问题。

### 2.3 多头注意力（Multi-Head Attention）

为了允许模型关注不同位置的不同表示子空间的信息，Transformer使用了**多头注意力**。

**过程：**
1.  将输入的 `Q`, `K`, `V` 通过不同的、可学习的线性变换（权重矩阵 `W_i^Q`, `W_i^K`, `W_i^V`）映射到 `h` 个不同的低维子空间（`h` 是头数）。
2.  在每个子空间 `i` 上，并行地执行一次缩放点积注意力，得到输出 `head_i`。
3.  将 `h` 个 `head_i` 拼接（Concatenate）起来。
4.  再通过一个线性变换（权重矩阵 `W^O`）得到最终的多头注意力输出。

**数学公式：**
```
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) * W^O
其中 head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)
```

**优势：**
*   允许模型在不同的表示子空间中并行地关注信息。
*   提供了多个“表示子空间”（representation subspaces）的视角。

## 3. Transformer的整体架构

Transformer由**Encoder**和**Decoder**两部分组成。

### 3.1 Encoder

Encoder由 `N` 个相同的层（Layer）堆叠而成。每一层包含两个子层：

1.  **多头自注意力子层（Multi-Head Self-Attention Sublayer）：**
    *   **自注意力（Self-Attention）：** 在Encoder中，`Q`, `K`, `V` 都来自同一个输入序列（前一层的输出）。它使得序列中的每个位置都能关注到序列的其他所有位置。
    *   **残差连接（Residual Connection） & 层归一化（Layer Normalization）：**
        ```
        Attention_Output = MultiHeadAttention(x, x, x)
        Sublayer_Output = LayerNorm(x + Attention_Output)
        ```

2.  **位置前馈网络子层（Position-wise Feed-Forward Networks Sublayer）：**
    *   这是一个两层的全连接网络，作用于每个位置（序列中的每个元素）。
    *   `FFN(x) = max(0, x*W_1 + b_1) * W_2 + b_2` （通常第一层维度较大，如2048或4096）。
    *   **残差连接 & 层归一化：**
        ```
        FFN_Output = FFN(Sublayer_Output)
        Layer_Output = LayerNorm(Sublayer_Output + FFN_Output)
        ```

**输入与位置编码：**
Encoder接收词嵌入（Token Embeddings）序列作为输入。由于注意力机制本身不包含序列顺序信息，需要将**位置编码**（Positional Encoding）添加到词嵌入中。Transformer使用固定的正弦和余弦函数来生成位置编码。

### 3.2 Decoder

Decoder也由 `N` 个相同的层堆叠而成。每一层包含三个子层：

1.  **掩码多头自注意力子层（Masked Multi-Head Self-Attention Sublayer）：**
    *   与Encoder的自注意力类似，但为了防止在训练时位置 `i` 关注到后续位置（`i+1, i+2, ...`）的信息（这在生成时是不可见的），需要对注意力分数矩阵进行**掩码**（Masking），将未来位置的分数设为负无穷（`-inf`），这样Softmax后对应的权重就为0了。

2.  **多头Encoder-Decoder注意力子层（Multi-Head Encoder-Decoder Attention Sublayer）：**
    *   这是连接Encoder和Decoder的关键。`Q` 来自Decoder的前一层输出，`K` 和 `V` 来自**整个**Encoder的输出。
    *   这使得Decoder的每个位置都能关注到输入序列的所有位置。
    *   **残差连接 & 层归一化。**

3.  **位置前馈网络子层（Position-wise Feed-Forward Networks Sublayer）：**
    *   与Encoder中的FFN相同。
    *   **残差连接 & 层归一化。**

**输出：**
Decoder的输出再通过一个线性变换和Softmax层，生成最终的预测词的概率分布。

## 4. 总结

*   **Transformer** 是一种基于注意力机制的深度学习架构，摒弃了RNN和CNN，实现了高度并行化。
*   **缩放点积注意力** 是其核心计算单元，通过 `Softmax(Q*K^T/sqrt(d_k))*V` 计算。
*   **多头注意力** 允许模型在不同子空间并行关注信息，增强了表示能力。
*   **Encoder** 由多头自注意力和前馈网络构成，用于编码输入序列。
*   **Decoder** 由掩码多头自注意力、Encoder-Decoder注意力和前馈网络构成，用于生成输出序列。
*   **位置编码** 被添加到词嵌入中，以向模型提供序列顺序信息。
*   **残差连接** 和 **层归一化** 被应用于每个子层，有助于训练深层网络。
*   Transformer是BERT、GPT等现代大模型的基础架构，对深度学习领域产生了深远影响。