# Day 2：向量进阶

**主题**：线性组合、线性无关、向量空间、投影、余弦相似度  
**预计时间**：2.5~3 小时

---

## 学习目标

- 理解线性组合与张成（Span）的概念
- 掌握线性无关的判断方法
- 理解向量空间与子空间的基本概念
- 掌握向量投影公式及几何直觉
- 能用余弦相似度衡量向量间相似程度

---

## 核心知识点

### 1. 线性组合与 Span

**线性组合**：给定向量 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ 和标量 $c_1, \ldots, c_k$：

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$$

**张成（Span）**：所有可能线性组合构成的集合，表示这组向量能"覆盖"的空间范围

- $\text{span}\{[1,0]^T, [0,1]^T\} = \mathbb{R}^2$（整个 2D 平面）
- $\text{span}\{[1,2]^T, [2,4]^T\} = $ 一条直线（方向相同，无法覆盖 2D）

### 2. 线性无关

向量组 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ **线性无关**，当且仅当：

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \implies c_1 = c_2 = \cdots = c_k = 0$$

**直觉**：没有任何一个向量能被其他向量的线性组合表示——每个向量都提供"新方向"

**判断方法**：
- 2D：两向量是否共线？共线则线性相关
- 一般：行化简（第 5、6 天矩阵工具）

### 3. 向量空间与基

**向量空间** $V$：在加法和数乘下封闭的集合

**基（Basis）**：能张成整个空间的**最小**线性无关集合
- $\mathbb{R}^n$ 的标准基：$\mathbf{e}_1 = [1,0,\ldots,0]^T, \ldots, \mathbf{e}_n = [0,\ldots,0,1]^T$
- 基的个数 = 空间的**维度**

### 4. 向量投影

将 $\mathbf{a}$ 投影到 $\mathbf{b}$ 方向上：

$$\text{proj}_{\mathbf{b}} \mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|^2} \mathbf{b}$$

**标量投影**（投影长度）：

$$\text{scalar proj} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|}$$

**几何直觉**：想象一束光垂直照射到 $\mathbf{b}$ 所在直线，$\mathbf{a}$ 的影子就是投影向量。

**在 ML 中**：PCA（主成分分析）的核心就是将数据投影到主方向上。

### 5. 余弦相似度

$$\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \cos\theta \in [-1, 1]$$

| 值 | 含义 |
|---|---|
| 1 | 完全同向（最相似） |
| 0 | 正交（无关） |
| -1 | 完全反向（最不相似） |

**在 ML/NLP 中**：词向量相似度、推荐系统、文档检索都大量使用余弦相似度。

---

## 示例与推导

### 示例 1：判断线性无关

$\mathbf{v}_1 = [1, 2]^T$，$\mathbf{v}_2 = [3, 4]^T$

假设 $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 = \mathbf{0}$：

$$c_1 + 3c_2 = 0,\quad 2c_1 + 4c_2 = 0$$

由第一式 $c_1 = -3c_2$，代入第二式：$-6c_2 + 4c_2 = -2c_2 = 0 \Rightarrow c_2 = 0$，从而 $c_1 = 0$。

✅ 线性无关（不共线）

---

### 示例 2：投影计算

$\mathbf{a} = [3, 1]^T$，$\mathbf{b} = [2, 0]^T$（x 轴方向）

$$\text{proj}_{\mathbf{b}} \mathbf{a} = \frac{3 \times 2 + 1 \times 0}{2^2 + 0^2} [2, 0]^T = \frac{6}{4}[2,0]^T = [3, 0]^T$$

直觉验证：投影到 x 轴，结果就是把 y 分量去掉，得 $[3,0]^T$。✅

---

## 动手练习

### 手算题

1. 判断以下向量组是否线性无关：
   - $\{[1,0,0]^T,\ [0,1,0]^T,\ [1,1,0]^T\}$
   - $\{[2,4]^T,\ [1,2]^T\}$

2. 将 $\mathbf{a} = [4, 3]^T$ 投影到 $\mathbf{b} = [1, 1]^T$，求投影向量和标量投影。

3. 计算以下两组向量的余弦相似度：
   - $\mathbf{u} = [1, 1, 0]^T$，$\mathbf{v} = [1, 0, 1]^T$
   - $\mathbf{u} = [1, 0]^T$，$\mathbf{v} = [-1, 0]^T$

### 代码练习

```python
import numpy as np

# --- 投影 ---
a = np.array([3.0, 1.0])
b = np.array([2.0, 0.0])

proj_scalar = np.dot(a, b) / np.linalg.norm(b)
proj_vector = (np.dot(a, b) / np.dot(b, b)) * b

print("标量投影:", proj_scalar)
print("投影向量:", proj_vector)

# --- 余弦相似度 ---
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

u1 = np.array([1, 1, 0])
v1 = np.array([1, 0, 1])
u2 = np.array([1, 0])
v2 = np.array([-1, 0])

print(f"cosine(u1, v1) = {cosine_similarity(u1, v1):.4f}")
print(f"cosine(u2, v2) = {cosine_similarity(u2, v2):.4f}")

# --- 词向量类比（mini演示）---
# 假设 "king", "queen", "man", "woman" 的简化 2D 向量
king  = np.array([0.9, 0.4])
queen = np.array([0.8, 0.7])
man   = np.array([0.7, 0.2])
woman = np.array([0.6, 0.5])

# 经典类比：king - man + woman ≈ queen ?
analogy = king - man + woman
print("\n类比向量 (king - man + woman):", analogy)
print("与 queen 的余弦相似度:", cosine_similarity(analogy, queen))
```

**运行方式**：
```bash
cd ~/dandan-learn/ai/linalg-week01
uv run python day02_exercise.py
```

---

## 小结

| 概念 | 核心定义 | ML 中的应用 |
|---|---|---|
| 线性组合 | $\sum c_i \mathbf{v}_i$ | 神经元加权求和 |
| 线性无关 | 无冗余方向 | 特征不冗余 |
| 基/维度 | 最小生成集 | 特征空间维度 |
| 投影 | $\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{b}\|^2}\mathbf{b}$ | PCA 降维 |
| 余弦相似度 | $\cos\theta$ | 语义相似度 |

**今日关键直觉**：余弦相似度只关心方向，不关心长度——两段长度差异很大的向量，只要方向相近，余弦相似度依然接近 1。这在 NLP 中非常有用（词频高低不影响语义方向）。

**明日预告**：D3 进入矩阵——从向量升维，理解矩阵作为线性变换的本质。
