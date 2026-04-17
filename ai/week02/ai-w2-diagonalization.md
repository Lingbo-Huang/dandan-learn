# D3：矩阵对角化的条件与步骤

> **Week 2 · Day 3** | AI 基础线：线性代数核心概念

---

## 1. 什么是矩阵对角化？

### 1.1 定义

如果存在可逆矩阵 $P$ 和对角矩阵 $D$，使得：

$$
A = PDP^{-1}
$$

则称矩阵 $A$ **可对角化**（Diagonalizable）。

等价形式（两边左乘 $P^{-1}$，右乘 $P$）：

$$
P^{-1}AP = D
$$

即在新基底 $P$ 下，$A$ 的表示是对角矩阵。

### 1.2 为什么对角化有用？

对角矩阵极易计算：

$$
D = \begin{pmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{pmatrix}
\Rightarrow
D^k = \begin{pmatrix} \lambda_1^k & & \\ & \ddots & \\ & & \lambda_n^k \end{pmatrix}
$$

由此：

$$
A^k = PD^kP^{-1}
$$

计算矩阵的幂从 $O(n^3)$ 的矩阵乘法简化为对角元素的幂次——这在动力系统、马尔可夫链、图神经网络中极其重要。

---

## 2. 对角化的充要条件

### 2.1 主要定理

**定理**：$n \times n$ 矩阵 $A$ 可对角化，**当且仅当** $A$ 有 $n$ 个线性无关的特征向量。

等价条件：

$$
\text{矩阵可对角化} \Longleftrightarrow \forall \lambda_i: \mu_g(\lambda_i) = \mu_a(\lambda_i)
$$

即每个特征值的**几何重数 = 代数重数**。

### 2.2 充分条件（更强）

**推论**：若 $n \times n$ 矩阵有 $n$ 个**不同的**特征值，则一定可对角化。

（因为不同特征值的特征向量线性无关，$n$ 个不同特征值提供 $n$ 个线性无关特征向量）

### 2.3 何时一定可以对角化？

- **实对称矩阵**（谱定理）：一定可以正交对角化，即 $P$ 为正交矩阵
- **有 $n$ 个不同特征值的矩阵**
- **正规矩阵**（$A^*A = AA^*$）：在复数域上可酉对角化

### 2.4 不可对角化的例子

$$
A = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}
$$

特征值：$\lambda = 2$（代数重数 2），但几何重数 1（只有方向 $\begin{pmatrix}1\\0\end{pmatrix}$ 一个线性无关特征向量）。因此 $A$ 不可对角化。

---

## 3. 对角化的步骤

### Step 1：求所有特征值

解特征方程：

$$
\det(A - \lambda I) = 0
$$

### Step 2：对每个特征值求特征向量

解 $(A - \lambda_i I)\mathbf{v} = \mathbf{0}$，得到特征空间的基向量。

### Step 3：验证是否有 $n$ 个线性无关特征向量

将所有特征空间的基向量合在一起，共 $n$ 个则可对角化。

### Step 4：构造矩阵 $P$ 和 $D$

$$
P = \begin{pmatrix} | & | & & | \\ \mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_n \\ | & | & & | \end{pmatrix}, \quad D = \begin{pmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{pmatrix}
$$

**注意**：$P$ 的第 $i$ 列 $\mathbf{v}_i$ 对应 $D$ 的第 $i$ 个对角元 $\lambda_i$，顺序必须一致。

---

## 4. 完整手推示例

### 4.1 设矩阵

$$
A = \begin{pmatrix} 1 & 2 & 0 \\ 0 & 3 & 0 \\ 2 & -4 & 2 \end{pmatrix}
$$

### 4.2 求特征值

$$
\det(A - \lambda I) = \det\begin{pmatrix} 1-\lambda & 2 & 0 \\ 0 & 3-\lambda & 0 \\ 2 & -4 & 2-\lambda \end{pmatrix}
$$

按第三列展开：

$$
= (2-\lambda)\det\begin{pmatrix} 1-\lambda & 2 \\ 0 & 3-\lambda \end{pmatrix}
= (2-\lambda)(1-\lambda)(3-\lambda)
$$

特征值：$\lambda_1 = 1, \lambda_2 = 2, \lambda_3 = 3$（三个不同特征值 → 一定可对角化）

### 4.3 求特征向量

**$\lambda_1 = 1$**：

$$
A - I = \begin{pmatrix} 0 & 2 & 0 \\ 0 & 2 & 0 \\ 2 & -4 & 1 \end{pmatrix}
\rightarrow
\begin{pmatrix} 2 & -4 & 1 \\ 0 & 2 & 0 \\ 0 & 0 & 0 \end{pmatrix}
$$

从第二行：$v_2 = 0$；代入第一行：$2v_1 + v_3 = 0$，令 $v_1 = 1, v_3 = -2$：

$$
\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \\ -2 \end{pmatrix}
$$

**$\lambda_2 = 2$**：

$$
A - 2I = \begin{pmatrix} -1 & 2 & 0 \\ 0 & 1 & 0 \\ 2 & -4 & 0 \end{pmatrix}
\rightarrow
\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}
$$

$v_1 = v_2 = 0$，$v_3$ 自由：

$$
\mathbf{v}_2 = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}
$$

**$\lambda_3 = 3$**：

$$
A - 3I = \begin{pmatrix} -2 & 2 & 0 \\ 0 & 0 & 0 \\ 2 & -4 & -1 \end{pmatrix}
\rightarrow
\begin{pmatrix} -2 & 2 & 0 \\ 2 & -4 & -1 \\ 0 & 0 & 0 \end{pmatrix}
$$

第一行 $\rightarrow$ $v_1 = v_2$；令 $v_2 = 2, v_1 = 2$，代入第二行：$4 - 8 - v_3 = 0 \Rightarrow v_3 = -4$：

$$
\mathbf{v}_3 = \begin{pmatrix} 2 \\ 2 \\ -4 \end{pmatrix}
$$

### 4.4 构造 $P$ 和 $D$

$$
P = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 0 & 2 \\ -2 & 1 & -4 \end{pmatrix}, \quad D = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{pmatrix}
$$

验证：$AP = PD$（即每列 $A\mathbf{v}_i = \lambda_i \mathbf{v}_i$）。

---

## 5. 对称矩阵的正交对角化

### 5.1 谱定理（Spectral Theorem）

**定理**：实对称矩阵 $A = A^T$ 一定可以正交对角化：

$$
A = Q\Lambda Q^T
$$

其中 $Q$ 是正交矩阵（$Q^TQ = I$），$\Lambda$ 是对角矩阵（包含特征值）。

**重要性质：**
1. 所有特征值为**实数**
2. 不同特征值的特征向量**互相正交**
3. 可以通过 Gram-Schmidt 正交化获得正交特征向量基

### 5.2 正交对角化步骤

在普通对角化基础上增加一步：对每个特征空间的基向量做 **Gram-Schmidt 正交化**，再归一化。

---

## 6. 矩阵函数与对角化

对角化使得矩阵函数的计算变得简单：

$$
f(A) = Pf(D)P^{-1}
$$

其中 $f(D) = \text{diag}(f(\lambda_1), \ldots, f(\lambda_n))$。

**例子：矩阵指数**

$$
e^A = P \begin{pmatrix} e^{\lambda_1} & & \\ & \ddots & \\ & & e^{\lambda_n} \end{pmatrix} P^{-1}
$$

这在求解常微分方程 $\dot{\mathbf{x}} = A\mathbf{x}$ 时有直接应用。

---

## 7. Python 代码示例

### 7.1 环境准备

```bash
uv init diag-demo
cd diag-demo
uv add numpy sympy
```

### 7.2 验证对角化

```python
# diagonalization.py
import numpy as np

def diagonalize(A, tol=1e-10):
    """
    对矩阵 A 进行特征值分解，返回 (P, D, P_inv)
    使得 A = P @ D @ P_inv
    """
    n = A.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    P = eigenvectors  # 列向量为特征向量
    D = np.diag(eigenvalues)
    
    # 检查是否可逆（特征向量是否线性无关）
    rank = np.linalg.matrix_rank(P)
    if rank < n:
        print(f"警告：矩阵可能不可对角化，P 的秩 = {rank} < {n}")
        return None, None, None
    
    P_inv = np.linalg.inv(P)
    
    # 验证 A = P D P^{-1}
    A_reconstructed = P @ D @ P_inv
    error = np.linalg.norm(A - A_reconstructed)
    print(f"重构误差 ||A - PDP^{{-1}}|| = {error:.2e}")
    
    return P, D, P_inv


# 示例：3x3 矩阵
A = np.array([[1., 2., 0.],
              [0., 3., 0.],
              [2., -4., 2.]])

print("矩阵 A:")
print(A)
print()

P, D, P_inv = diagonalize(A)

if P is not None:
    print(f"特征值（D 的对角元）: {np.diag(D).real}")
    print(f"特征向量矩阵 P:\n{P.real.round(4)}")
    
    # 用对角化计算 A^5
    k = 5
    Ak_diag = P @ np.diag(np.diag(D)**k) @ P_inv
    Ak_direct = np.linalg.matrix_power(A, k)
    print(f"\nA^{k} (对角化方法):\n{Ak_diag.real.round(4)}")
    print(f"A^{k} (直接计算):\n{Ak_direct.round(4)}")
    print(f"误差: {np.linalg.norm(Ak_diag - Ak_direct):.2e}")
```

### 7.3 正交对角化实战

```python
# ortho_diag.py
import numpy as np

def ortho_diagonalize(S):
    """对实对称矩阵进行正交对角化"""
    # np.linalg.eigh 专为实对称矩阵优化，返回有序特征值
    eigenvalues, Q = np.linalg.eigh(S)
    
    # 验证 Q 是正交矩阵
    QTQ = Q.T @ Q
    print(f"Q^T Q（应接近 I）:\n{QTQ.round(6)}")
    
    # 验证 S = Q Λ Q^T
    Lambda = np.diag(eigenvalues)
    S_reconstructed = Q @ Lambda @ Q.T
    print(f"\n重构误差: {np.linalg.norm(S - S_reconstructed):.2e}")
    
    return eigenvalues, Q

# 协方差矩阵示例（PCA 的核心）
S = np.array([[4., 2., 0.],
              [2., 3., 1.],
              [0., 1., 2.]])

print("对称矩阵 S:")
print(S)
print()

eigenvalues, Q = ortho_diagonalize(S)
print(f"\n特征值（升序）: {eigenvalues.round(4)}")
print(f"总方差（迹）: {np.trace(S):.4f} = {eigenvalues.sum():.4f}")
print(f"各特征值占比: {(eigenvalues / eigenvalues.sum() * 100).round(2)}%")
```

---

## 8. 小结

| 主题 | 要点 |
|------|------|
| 对角化定义 | $A = PDP^{-1}$，$P$ 由特征向量构成，$D$ 由特征值构成 |
| 充要条件 | $n$ 个线性无关特征向量 ↔ 每个特征值几何重数 = 代数重数 |
| 充分条件 | $n$ 个不同特征值 |
| 对称矩阵 | 谱定理：$A = Q\Lambda Q^T$，$Q$ 正交 |
| 应用 | 矩阵幂次、矩阵函数、动力系统、PCA |
| 不可对角化 | Jordan 块（几何重数 < 代数重数） |

**下一篇**：D4 SVD 定理与几何直觉——当矩阵不是方阵，或不可对角化时，SVD 如何给出更通用的分解？

---

*参考：Strang《线性代数及其应用》§6.2；Horn & Johnson《矩阵分析》*
