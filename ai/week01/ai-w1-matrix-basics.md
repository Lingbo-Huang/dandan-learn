# Day 3：矩阵基础

**主题**：矩阵定义、加减法、数乘、转置、特殊矩阵  
**预计时间**：2.5~3 小时

---

## 学习目标

- 理解矩阵是线性变换的表示
- 掌握矩阵加减法、数乘运算规则
- 掌握矩阵转置及其性质
- 识别常见特殊矩阵（单位矩阵、零矩阵、对角矩阵、对称矩阵）
- 理解矩阵与向量的乘法（Ax），建立"矩阵作用于向量"的直觉

---

## 核心知识点

### 1. 矩阵的定义

$m \times n$ 矩阵是 $m$ 行 $n$ 列的数字阵列：

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

- **行向量**：每一行 $[a_{i1}, a_{i2}, \ldots, a_{in}]$
- **列向量**：每一列 $[a_{1j}, a_{2j}, \ldots, a_{mj}]^T$
- 矩阵可视为**将向量从 $\mathbb{R}^n$ 映射到 $\mathbb{R}^m$ 的线性变换**

### 2. 矩阵加减法与数乘

**加法**（同型矩阵对应元素相加）：

$$A + B = \begin{bmatrix} a_{ij} + b_{ij} \end{bmatrix}$$

要求：$A, B$ 必须同型（行列数相同）

**数乘**：

$$cA = \begin{bmatrix} c \cdot a_{ij} \end{bmatrix}$$

**性质**：交换律、结合律、分配律同向量运算

### 3. 矩阵转置

将矩阵的行变为列（沿主对角线翻折）：

$$(A^T)_{ij} = A_{ji}$$

若 $A$ 是 $m \times n$ 矩阵，则 $A^T$ 是 $n \times m$ 矩阵。

**重要性质**：
- $(A^T)^T = A$
- $(A + B)^T = A^T + B^T$
- $(cA)^T = cA^T$
- $(AB)^T = B^T A^T$（顺序翻转！）

### 4. 特殊矩阵

**单位矩阵** $I_n$（对角线全 1，其余 0）：
$$I_3 = \begin{bmatrix} 1&0&0 \\ 0&1&0 \\ 0&0&1 \end{bmatrix}$$
性质：$AI = IA = A$（乘法中的"1"）

**零矩阵** $O$：所有元素为 0

**对角矩阵**：非对角线元素为 0，$D = \text{diag}(d_1, d_2, \ldots, d_n)$

**对称矩阵**：$A = A^T$（沿主对角线对称）  
→ 协方差矩阵、Hessian 矩阵都是对称矩阵

**上三角/下三角矩阵**：对角线以下/以上元素为 0

### 5. 矩阵与向量相乘（Ax）

$A$（$m \times n$）× $\mathbf{x}$（$n \times 1$）= $\mathbf{b}$（$m \times 1$）

**行视角**（每行与 x 做点积）：

$$b_i = \sum_{j=1}^n a_{ij} x_j = \text{（第 } i \text{ 行）}\cdot \mathbf{x}$$

**列视角**（x 的各分量对 A 各列加权求和）：

$$A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n$$

其中 $\mathbf{a}_j$ 是 $A$ 的第 $j$ 列。**这是理解矩阵乘法最深刻的视角之一。**

---

## 示例与推导

### 示例 1：矩阵运算

$$A = \begin{bmatrix} 1&2\\3&4 \end{bmatrix},\quad B = \begin{bmatrix} 5&6\\7&8 \end{bmatrix}$$

$$A + B = \begin{bmatrix} 6&8\\10&12 \end{bmatrix},\quad 2A = \begin{bmatrix} 2&4\\6&8 \end{bmatrix},\quad A^T = \begin{bmatrix} 1&3\\2&4 \end{bmatrix}$$

### 示例 2：Ax 的列视角

$$A = \begin{bmatrix} 1&2\\3&4 \end{bmatrix},\quad \mathbf{x} = \begin{bmatrix} 2\\1 \end{bmatrix}$$

**列视角**：

$$A\mathbf{x} = 2\begin{bmatrix}1\\3\end{bmatrix} + 1\begin{bmatrix}2\\4\end{bmatrix} = \begin{bmatrix}2\\6\end{bmatrix} + \begin{bmatrix}2\\4\end{bmatrix} = \begin{bmatrix}4\\10\end{bmatrix}$$

**行视角验证**：
- 第 1 行：$1\times2 + 2\times1 = 4$ ✅
- 第 2 行：$3\times2 + 4\times1 = 10$ ✅

---

## 动手练习

### 手算题

1. 设 $A = \begin{bmatrix}2&-1\\0&3\end{bmatrix}$，$B = \begin{bmatrix}1&4\\-2&1\end{bmatrix}$，计算：
   - $A + 2B$
   - $A^T$，$B^T$，$(A+B)^T$，$A^T + B^T$（验证转置分配律）

2. 验证：若 $A$ 是对称矩阵，则 $A + A^T = 2A$；若 $A$ 是任意方阵，$A + A^T$ 是对称矩阵。

3. 计算 $A\mathbf{x}$，并用列视角理解：
   $$A = \begin{bmatrix}1&0&2\\0&3&1\end{bmatrix},\quad \mathbf{x} = \begin{bmatrix}1\\2\\3\end{bmatrix}$$

### 代码练习

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 加法与数乘
print("A + B =\n", A + B)
print("2A =\n", 2 * A)

# 转置
print("A.T =\n", A.T)
print("(A+B).T =\n", (A+B).T)
print("A.T + B.T =\n", A.T + B.T)
print("转置分配律验证:", np.allclose((A+B).T, A.T + B.T))

# 特殊矩阵
I = np.eye(3)              # 3x3 单位矩阵
D = np.diag([1, 2, 3])    # 对角矩阵
print("\n单位矩阵 I:\n", I)
print("对角矩阵 D:\n", D)

# 对称矩阵检查
M = np.array([[1, 2, 3], [2, 5, 4], [3, 4, 9]])
print("\nM 是对称矩阵?", np.allclose(M, M.T))

# 任意方阵 -> 对称矩阵
C = np.array([[1, 2], [3, 4]])
sym = C + C.T
print("C + C.T =\n", sym)
print("sym 是对称矩阵?", np.allclose(sym, sym.T))

# Ax
x = np.array([2, 1])
b = A @ x     # 或 np.dot(A, x)
print("\nAx =", b)

# 列视角验证
col_view = x[0] * A[:, 0] + x[1] * A[:, 1]
print("列视角 Ax =", col_view)
print("两种方法一致?", np.allclose(b, col_view))
```

**运行方式**：
```bash
cd ~/dandan-learn/ai/linalg-week01
uv run python day03_exercise.py
```

---

## 小结

| 概念 | 规则 | 注意点 |
|---|---|---|
| 加减法 | 对应元素运算 | 必须同型 |
| 数乘 | 每个元素 × 标量 | 无限制 |
| 转置 | 行列互换 | $(AB)^T = B^T A^T$ 顺序翻转 |
| 单位矩阵 | $AI = IA = A$ | 乘法的"1" |
| $Ax$ 列视角 | $\sum x_j \mathbf{a}_j$ | 理解矩阵乘法的钥匙 |

**今日关键直觉**：矩阵 $A$ 作用于向量 $\mathbf{x}$，本质是对 $A$ 的各列按 $\mathbf{x}$ 的分量做加权组合——$\mathbf{x}$ 告诉你"每列要多少份"。

**明日预告**：D4 矩阵乘法——两个矩阵相乘，以及它如何描述复合变换。
