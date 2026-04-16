# Day 1：环境搭建 & 大模型学习路线导览

---

## 学习目标

- 搭建大模型学习所需的 Python 环境（使用 **uv** 管理）
- 了解大模型学习的整体路线图，定位 Week 1 在其中的位置
- 熟悉本周核心资料，做好预习

---

## 一、环境搭建（使用 uv）

> ⚠️ 本课程**全程使用 uv** 进行包管理，不使用 conda。

### 1.1 安装 uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows（PowerShell）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装完成后验证：
```bash
uv --version
```

### 1.2 创建虚拟环境

```bash
# 在项目目录下创建虚拟环境（指定 Python 版本）
uv venv .venv --python 3.11

# 激活虚拟环境
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 1.3 安装依赖

```bash
# 安装 PyTorch（CPU 版，适合学习阶段）
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 若有 NVIDIA GPU，改用：
# uv pip install torch torchvision torchaudio

# 安装其他常用包
uv pip install numpy matplotlib jupyter transformers datasets
```

### 1.4 验证安装

```python
# verify_env.py
import torch
import transformers

print(f"PyTorch 版本: {torch.__version__}")
print(f"Transformers 版本: {transformers.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

# 简单张量测试
x = torch.randn(2, 3)
print(f"测试张量:\n{x}")
```

```bash
python verify_env.py
```

---

## 二、核心知识点：大模型学习路线图

### 2.1 整体学习路径

```
基础数学 (线性代数/概率/微积分)
        ↓
深度学习基础 (MLP / CNN / RNN)
        ↓
注意力机制 → Transformer 架构  ← 【本周位置】
        ↓
预训练语言模型 (BERT / GPT 系列)
        ↓
大语言模型 (LLaMA / GPT-4 等)
        ↓
微调与对齐 (SFT / RLHF / DPO)
        ↓
应用开发 (RAG / Agent / 工具调用)
```

### 2.2 Transformer 在其中的核心地位

Transformer 是现代大模型的基石架构。理解它等于掌握了大模型世界的底层语言：

- **BERT** = Transformer Encoder
- **GPT** = Transformer Decoder
- **T5 / BART** = Transformer Encoder-Decoder
- **LLaMA / Mistral / Qwen** = 优化版 Decoder-only Transformer

### 2.3 本周学习重点

本周专注"读懂"Transformer，不追求从零写出完整代码（留到 D7），重点是：

1. 能用自己的话描述数据在 Transformer 中的流动过程
2. 理解 Attention 的数学本质
3. 对比不同架构变体的设计选择

---

## 三、示例：神经网络 vs Transformer 的核心区别

传统 RNN 处理序列时是**串行**的，每步依赖上一步：

```
x₁ → h₁ → x₂ → h₂ → x₃ → h₃ → ...
```

Transformer 用 Attention 实现**并行**处理，每个位置直接与所有位置交互：

```
x₁ ─┬──→ Attention ← x₁
x₂ ─┼──→ Attention ← x₂  (一次并行处理全部位置)
x₃ ─┴──→ Attention ← x₃
```

这是 Transformer 能处理超长序列、适合 GPU 并行的根本原因。

---

## 四、动手练习

### 练习 1：安装验证

完成 1.4 中的环境验证脚本，截图或记录输出结果。

### 练习 2：浏览核心资料

- 打开 [Jay Alammar - The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- 快速浏览（不求全懂），记录 3 个你感兴趣或不理解的点

### 练习 3：原论文摘要精读

阅读《Attention Is All You Need》的 Abstract 和 Introduction 部分（1-2 页），用中文写下：
- 论文要解决什么问题？
- 提出了什么方案？
- 有什么主要优势？

---

## 五、小结

| 项目 | 内容 |
|------|------|
| 今日完成 | uv 环境搭建 + PyTorch 安装验证 |
| 核心认知 | Transformer 是大模型的基石；并行 Attention 取代串行 RNN |
| 明日预告 | D2 正式进入 Transformer 整体架构，理解 Encoder-Decoder 结构与数据流 |

> 💡 **记录疑问**：今天有什么没看懂的？写下来，后续几天逐步解答。
