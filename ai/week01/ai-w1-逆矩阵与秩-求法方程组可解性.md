# Day 6：矩阵的逆与秩

**主题**：逆矩阵的定义与求法、矩阵秩、线性方程组可解性  
**预计时间**：2.5~3 小时

---

## 学习目标

- 理解逆矩阵的定义与存在条件
- 掌握 2×2 矩阵逆的手算方法（伴随矩阵法）
- 理解高斯-约旦消元法求逆的过程
- 理解矩阵秩的概念及其几何意义
- 建立逆矩阵、行列式、秩与线性方程组可解性之间的联系

---

## 核心知识点

### 1. 逆矩阵的定义

若方阵 $A$（$n\times n$）存在矩阵 $B$ 使得：

$$AB = BA = I_n$$

则 $B$ 称为 $A$ 的**逆矩阵**，记为 $A^{-1}$。

**存在条件**（等价命题，任一成立则均成立）：
- $\det(A) \neq 0$
- $A$ 满秩（秩 = $n$）
- $A$ 的列向量线性无关
- $A$ 不把任何非零向量映射到零向量
- 方程 $A\mathbf{x} = \mathbf{0}$ 只有零解

不满足以上条件的矩阵称为**奇异矩阵**（不可逆）。

### 2. 2×2 矩阵的逆（伴随矩阵法）

$$A = \begin{bmatrix}a&b\\c&d\end{bmatrix}, \quad \det(A) = ad-bc \neq 0$$

$$A^{-1} = \frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$$

**记忆技巧**：主对角线交换，副对角线变号，除以行列式。

### 3. 高斯-约旦消元法（通用）

在增广矩阵 $[A | I]$ 上施行行变换，直到左侧化为单位矩阵，右侧即为 $A^{-1}$：

$$[A | I] \xrightarrow{\text{行变换}} [I | A^{-1}]$$

**行变换操作**：
1. 行交换
2. 行乘以非零常数
3. 一行加上另一行的倍数

### 4. 逆矩阵的性质

| 性质 | 公式 |
|---|---|
| 逆的逆 | $(A^{-1})^{-1} = A$ |
| 乘积的逆 | $(AB)^{-1} = B^{-1}A^{-1}$（顺序翻转！） |
| 转置的逆 | $(A^T)^{-1} = (A^{-1})^T$ |
| 行列式 | $\det(A^{-1}) = 1/\det(A)$ |

### 5. 矩阵的秩

矩阵 $A$ 的**秩**（rank）= $A$ 的行空间（或列空间）的维度 = 线性无关的行（或列）的最大数量

$$\text{rank}(A) = r$$

**高斯消元求秩**：将 $A$ 化为行阶梯形，非零行的数量就是秩。

**秩与解的关系**（方程组 $A\mathbf{x} = \mathbf{b}$，$A$ 是 $m\times n$ 矩阵）：

| 条件 | 解的情况 |
|---|---|
| $\text{rank}(A) = \text{rank}([A|\mathbf{b}]) = n$ | 唯一解 |
| $\text{rank}(A) = \text{rank}([A|\mathbf{b}]) < n$ | 无穷多解 |
| $\text{rank}(A) < \text{rank}([A|\mathbf{b}])$ | 无解 |

**秩-零化度定理**：$\text{rank}(A) + \text{nullity}(A) = n$（列数）

---

## 示例与推导

### 示例 1：2×2 矩阵求逆

$$A = \begin{bmatrix}3&1\\2&4\end{bmatrix},\quad \det(A) = 12-2 = 10$$

$$A^{-1} = \frac{1}{10}\begin{bmatrix}4&-1\\-2&3\end{bmatrix} = \begin{bmatrix}0.4&-0.1\\-0.2&0.3\end{bmatrix}$$

**验证**：

$$AA^{-1} = \begin{bmatrix}3&1\\2&4\end{bmatrix}\begin{bmatrix}0.4&-0.1\\-0.2&0.3\end{bmatrix} = \begin{bmatrix}1&0\\0&1\end{bmatrix}\ ✅$$

### 示例 2：高斯-约旦消元法

$$A = \begin{bmatrix}1&2\\3&4\end{bmatrix}$$

增广矩阵：

$$\left[\begin{array}{cc|cc}1&2&1&0\\3&4&0&1\end{array}\right]$$

$R_2 \leftarrow R_2 - 3R_1$：

$$\left[\begin{array}{cc|cc}1&2&1&0\\0&-2&-3&1\end{array}\right]$$

$R_2 \leftarrow R_2 \div (-2)$：

$$\left[\begin{array}{cc|cc}1&2&1&0\\0&1&3/2&-1/2\end{array}\right]$$

$R_1 \leftarrow R_1 - 2R_2$：

$$\left[\begin{array}{cc|cc}1&0&-2&1\\0&1&3/2&-1/2\end{array}\right]$$

$$A^{-1} = \begin{bmatrix}-2&1\\3/2&-1/2\end{bmatrix}$$

### 示例 3：用逆矩阵解方程组

$$A\mathbf{x} = \mathbf{b} \implies \mathbf{x} = A^{-1}\mathbf{b}$$

若 $A = \begin{bmatrix}2&1\\5&3\end{bmatrix}$，$\mathbf{b} = \begin{bmatrix}1\\2\end{bmatrix}$

$\det(A) = 1$，$A^{-1} = \begin{bmatrix}3&-1\\-5&2\end{bmatrix}$

$$\mathbf{x} = \begin{bmatrix}3&-1\\-5&2\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix} = \begin{bmatrix}1\\-1\end{bmatrix}$$

---

## 动手练习

### 手算题

1. 用公式求 $A = \begin{bmatrix}4&7\\2&6\end{bmatrix}$ 的逆，并验证 $AA^{-1} = I$

2. 判断以下矩阵是否可逆（不需完整计算）：
   - $\begin{bmatrix}1&2&3\\2&4&6\\0&1&1\end{bmatrix}$
   - $\begin{bmatrix}1&0\\0&2\end{bmatrix}$

3. 求矩阵 $A = \begin{bmatrix}1&2&1\\2&1&-1\\1&-1&2\end{bmatrix}$ 的秩（化行阶梯形）

### 代码练习

```python
import numpy as np

# 2x2 矩阵求逆
A = np.array([[3.0, 1.0], [2.0, 4.0]])
A_inv = np.linalg.inv(A)
print("A^{-1} =\n", A_inv)
print("验证 A @ A^{-1} =\n", A @ A_inv)
print("是单位矩阵?", np.allclose(A @ A_inv, np.eye(2)))

# 3x3 矩阵求逆
B = np.array([[1.0, 2.0, 0.0],
              [0.0, 1.0, 1.0],
              [1.0, 0.0, 2.0]])
B_inv = np.linalg.inv(B)
print("\nB^{-1} =\n", B_inv)

# 奇异矩阵（不可逆）
C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
try:
    C_inv = np.linalg.inv(C)
    print("\nC 的逆（近似）:\n", C_inv)  # 会有数值问题
except np.linalg.LinAlgError as e:
    print("\n无法求逆:", e)
print("det(C) =", np.linalg.det(C))  # 接近 0

# 矩阵秩
print("\nrank(C) =", np.linalg.matrix_rank(C))  # = 2，不满秩

D = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
print("rank(D) =", np.linalg.matrix_rank(D))  # = 3，满秩

# 用逆矩阵解方程组 Ax = b
A_eq = np.array([[2.0, 1.0], [5.0, 3.0]])
b_eq = np.array([1.0, 2.0])

x_inv = np.linalg.inv(A_eq) @ b_eq
x_solve = np.linalg.solve(A_eq, b_eq)  # 更推荐的方式

print("\n逆矩阵法 x =", x_inv)
print("np.linalg.solve x =", x_solve)
print("验证 Ax =", A_eq @ x_solve)
print("与 b 一致?", np.allclose(A_eq @ x_solve, b_eq))

# 秩与可解性演示
print("\n=== 可解性分析 ===")
A_sys = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
b1 = np.array([1.0, 2.0])    # 有解
b2 = np.array([1.0, 10.0])   # 可能无解

print("rank(A) =", np.linalg.matrix_rank(A_sys))
Ab1 = np.column_stack([A_sys, b1.reshape(-1, 1)])
Ab2 = np.column_stack([A_sys, b2.reshape(-1, 1)])
print("rank([A|b1]) =", np.linalg.matrix_rank(Ab1))
print("rank([A|b2]) =", np.linalg.matrix_rank(Ab2))
```

**运行方式**：
```bash
cd ~/dandan-learn/ai/linalg-week01
uv run python day06_exercise.py
```

---

## 小结

| 概念 | 关键判据 | ML 中的意义 |
|---|---|---|
| 逆矩阵 | $\det(A) \neq 0$ | 求解正规方程 $(X^TX)^{-1}X^T y$ |
| 奇异矩阵 | $\det = 0$，秩不满 | 特征共线，过拟合信号 |
| 高斯-约旦法 | $[A|I] \to [I|A^{-1}]$ | 数值求逆的基础 |
| 矩阵秩 | 线性无关行/列的个数 | 特征有效维度 |
| `np.linalg.solve` | 数值上优于求逆再乘 | 实际推荐用法 |

**今日关键直觉**：实际中永远用 `np.linalg.solve(A, b)` 而不是 `np.linalg.inv(A) @ b`——数值更稳定，效率更高。逆矩阵更多是理论工具，而非计算工具。

**明日预告**：D7 综合复习——把这一周的知识串联起来，用 NumPy 解决一个完整的线性方程组问题，并做系统性小测验。
