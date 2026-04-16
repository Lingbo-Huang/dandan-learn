# 大模型线 Week 1 周规划总览

**主题：Transformer 总览（整体架构、输入输出流程）**
**周期：Day 1 - Day 7**

---

## 本周目标

通过一周系统学习，完整掌握 Transformer 的整体架构设计与数据流动逻辑，能够：

- 描述 Transformer 的输入输出流程
- 理解并推导 Scaled Dot-Product Attention 与 Multi-Head Attention
- 区分 Encoder-only / Decoder-only / Encoder-Decoder 三种结构变体
- 手写一个最小可运行的 Transformer forward pass（PyTorch）
- 阅读并理解原论文《Attention Is All You Need》核心章节

---

## 每日主题速览

| Day | 主题 | 关键词 |
|-----|------|--------|
| D1 | 环境搭建 & 大模型学习路线导览 | uv、PyTorch、学习地图 |
| D2 | Transformer 鸟瞰：整体结构与信息流 | Encoder-Decoder、Residual、LayerNorm |
| D3 | Attention 机制核心：Q/K/V 与 Scaled Dot-Product | Attention Score、Softmax、加权求和 |
| D4 | Multi-Head Attention & 位置编码 | 多头并行、正弦编码、相对位置 |
| D5 | Feed-Forward Network & 残差归一化 | FFN、Pre-LN vs Post-LN、Dropout |
| D6 | Encoder vs Decoder vs Encoder-Decoder 变体对比 | BERT、GPT、T5 架构对比 |
| D7 | 动手实现：最小 Transformer + 本周总结 | 代码实现、原论文精读、知识梳理 |

---

## 学习资源推荐

| 类型 | 资源 |
|------|------|
| 论文 | [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) |
| 图解 | Jay Alammar - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) |
| 视频 | Andrej Karpathy - [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 代码 | [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) |
| 书籍 | 《动手学深度学习》第 10 章 - 注意力机制 |

---

## 每日学习节奏建议

- **理论阅读**：30-40 分钟（笔记 + 公式推导）
- **代码实践**：30-45 分钟（跑通示例 + 修改实验）
- **小结回顾**：10 分钟（写下今日收获与疑问）

---

## 目录结构

```
/dandan-learn/llm/week01/
├── llm-w1-周规划总览.md                              # 本文件：周规划总览
├── llm-w1-环境搭建与Transformer全景导览.md           # D1：环境搭建 & 学习路线导览
├── llm-w1-输入处理-Token嵌入位置编码.md              # D2：Transformer 整体结构
├── llm-w1-自注意力机制-QKV多头计算推导.md            # D3：Attention 核心机制
├── llm-w1-前馈网络与LayerNorm.md                     # D4：Multi-Head Attention & 位置编码
├── llm-w1-编码器解码器架构对比.md                    # D5：FFN & 残差归一化
├── llm-w1-Transformer完整训练流程实战.md             # D6：架构变体对比
└── llm-w1-综合复习-架构图解与手写笔记.md             # D7：动手实现 + 本周总结
```
