# D2：特征向量与特征空间

> **Week 2 · Day 2** | AI 基础线：线性代数核心概念

---

## 1. 回顾与引入

上一篇我们学习了特征值的定义：$A\mathbf{v} = \lambda\mathbf{v}$。今天深入研究特征向量的结构——为什么同一个特征值可以对应"一整个空间"的特征向量？这个空间叫什么？它有什么性质？

理解特征空间是理解矩阵对角化、谱分解的关键，也是 PCA 中"主成分"概念的数学根基。

---

## 2. 特征向量的求解

### 2.1 步骤

给定特征值 $\lambda$，对应的特征向量满足：

$$
(A - \lambda I)\mathbf{v} = \mathbf{0}
$$

这是关于 $\mathbf{v}$ 的齐次线性方程组。解这个方程组，即求矩阵 $(A - \lambda I)$ 的**零空间（Null Space）**。

**步骤：**
1. 计算 $B = A - \lambda I$
2. 对 $B$ 做行化简（高斯消元）
3. 求 $B\mathbf{x} = \mathbf{0}$ 的通解

### 2.2 手推示例

设：

$$
A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}
$$

**Step 1：求特征值**

$$
\det(A - \lambda I) = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda-2)(\lambda-5) = 0
$$

$\lambda_1 = 2$，$\lambda_2 = 5$

**Step 2：对 $\lambda_1 = 2$ 求特征向量**

$$
A - 2I = \begin{pmatrix} 2 & 1 \\ 2 & 1 \end{pmatrix}
$$

行化简：

$$
\begin{pmatrix} 2 & 1 \\ 0 & 0 \end{pmatrix} \xrightarrow{} 2v_1 + v_2 = 0
$$

令 $v_2 = t$（自由变量），则 $v_1 = -t/2$。取 $t = 2$：

$$
\mathbf{v}_1 = \begin{pmatrix} -1 \\ 2 \end{pmatrix}
$$

**Step 3：对 $\lambda_2 = 5$ 求特征向量**

$$
A - 5I = \begin{pmatrix} -1 & 1 \\ 2 & -2 \end{pmatrix} \rightarrow \begin{pmatrix} -1 & 1 \\ 0 & 0 \end{pmatrix}
$$

$-v_1 + v_2 = 0$，令 $v_1 = t$：

$$
\mathbf{v}_2 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}
$$

---

## 3. 特征空间

### 3.1 定义

对于矩阵 $A$ 和特征值 $\lambda$，**特征空间**（Eigenspace）定义为：

$$
E_\lambda = \ker(A - \lambda I) = \{\mathbf{v} \in \mathbb{R}^n : (A - \lambda I)\mathbf{v} = \mathbf{0}\}
$$

即 $(A - \lambda I)$ 的零空间。

**性质：**
- $E_\lambda$ 是一个**子空间**（包含零向量，对加法和数乘封闭）
- $\dim(E_\lambda) \geq 1$（至少包含一个非零特征向量）
- 特征空间中任意非零向量都是 $\lambda$ 的特征向量

### 3.2 代数重数与几何重数

**代数重数（Algebraic Multiplicity）$\mu_a(\lambda)$**：特征值 $\lambda$ 作为特征多项式根的重数。

**几何重数（Geometric Multiplicity）$\mu_g(\lambda)$**：特征空间 $E_\lambda$ 的维数。

$$
1 \leq \mu_g(\lambda) \leq \mu_a(\lambda)
$$

**关键区别**：

| | 代数重数 | 几何重数 |
|--|---------|---------|
| 定义 | 特征多项式根的次数 | 特征空间维数 |
| 意义 | "有多少个 $\lambda$" | "有多少个独立特征向量" |
| 矩阵可对角化 | $\mu_a = \mu_g$ 对所有 $\lambda$ 成立 |

### 3.3 重特征值的例子

考虑：

$$
A = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}, \quad B = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}
$$

两者特征多项式都是 $(\lambda - 2)^2 = 0$，代数重数 $\mu_a(2) = 2$。

**但几何重数不同：**

对 $A$：$A - 2I = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$，零空间维数为 1，$\mu_g(2) = 1$。

对 $B$：$B - 2I = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$，零空间维数为 2，$\mu_g(2) = 2$。

- $A$ **不可对角化**（缺特征向量），称为 Jordan 块
- $B = 2I$ **可对角化**（整个空间都是特征空间）

---

## 4. 特征向量的线性无关性

### 4.1 重要定理

**定理**：对应于**不同特征值**的特征向量一定线性无关。

**证明（对两个特征值的情形）：**

设 $\lambda_1 \neq \lambda_2$，对应特征向量 $\mathbf{v}_1, \mathbf{v}_2$。假设它们线性相关：

$$
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 = \mathbf{0} \quad (*)
$$

两边左乘 $A$：

$$
c_1\lambda_1\mathbf{v}_1 + c_2\lambda_2\mathbf{v}_2 = \mathbf{0} \quad (**)
$$

$(** ) - \lambda_2 \times (*)$：

$$
c_1(\lambda_1 - \lambda_2)\mathbf{v}_1 = \mathbf{0}
$$

因为 $\lambda_1 \neq \lambda_2$ 且 $\mathbf{v}_1 \neq \mathbf{0}$，所以 $c_1 = 0$。类似得 $c_2 = 0$。故线性无关。$\square$

---

## 5. 特征向量的几何直觉

### 5.1 二维平面的直觉

想象一个橡皮膜，矩阵变换就像拉伸/压缩这个橡皮膜。**特征向量方向是橡皮膜被"纯拉伸"的方向**，不会偏转。

对于实对称矩阵（如协方差矩阵），不同特征值对应的特征向量**互相正交**——这正是 PCA 中主成分正交的原因。

### 5.2 广义特征向量（Jordan 分解预告）

当几何重数 $<$ 代数重数时，仅有普通特征向量不够，需要引入**广义特征向量**来补全基。这引出 Jordan 标准形的理论（超出本系列范围，留作进阶）。

---

## 6. Python 代码示例

### 6.1 环境准备

```bash
uv init eigenvector-demo
cd eigenvector-demo
uv add numpy scipy matplotlib
```

### 6.2 特征空间可视化

```python
# eigenvector_space.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def visualize_eigenspace(A, title="特征空间可视化"):
    """可视化 2x2 矩阵的特征空间"""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制网格点的变换效果
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    colors_idx = 0
    colors_map = plt.cm.Blues
    
    # 绘制特征向量（特征空间的基向量）
    clist = ['#E74C3C', '#27AE60', '#8E44AD', '#F39C12']
    for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        v_norm = v.real / np.linalg.norm(v.real)
        # 画出特征向量的整条直线（特征空间）
        t_vals = np.linspace(-2.5, 2.5, 100)
        line_points = np.outer(t_vals, v_norm)
        ax.plot(line_points[:, 0], line_points[:, 1], 
                '--', color=clist[i], alpha=0.5, linewidth=1.5,
                label=f'特征空间 E_λ{i+1} (λ={lam.real:.2f})')
        # 画出单位特征向量
        ax.annotate('', xy=v_norm, xytext=(0,0),
                    arrowprops=dict(arrowstyle='->', color=clist[i], lw=2.5))
    
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.legend(fontsize=12)
    ax.set_title(f'{title}\nA = {A.tolist()}', fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    return fig

# 示例1：两个不同特征值
A1 = np.array([[4., 1.],
               [2., 3.]])
fig1 = visualize_eigenspace(A1, "两个不同特征值")
fig1.savefig('eigenspace_two_distinct.png', dpi=150, bbox_inches='tight')
print("两个不同特征值的特征空间图已保存")

# 示例2：重特征值 - Jordan 块
A2 = np.array([[2., 1.],
               [0., 2.]])
eigenvalues2, eigenvectors2 = np.linalg.eig(A2)
print(f"\nJordan 块 A2 的特征值: {eigenvalues2}")
print(f"特征向量 (每列):\n{eigenvectors2}")
print(f"几何重数 = {np.linalg.matrix_rank(np.eye(2)) - np.linalg.matrix_rank(A2 - 2*np.eye(2))}")
# 注意：A2 - 2I 的秩为 1，零空间维数为 1，几何重数 = 1
null_space_dim = 2 - np.linalg.matrix_rank(A2 - 2*np.eye(2))
print(f"零空间维数（几何重数）: {null_space_dim}")
```

### 6.3 验证正交性（对称矩阵）

```python
# symmetric_eigenvectors.py
import numpy as np

# 实对称矩阵的特征向量互相正交
S = np.array([[3., 1., 0.],
              [1., 2., 1.],
              [0., 1., 3.]])

eigenvalues, eigenvectors = np.linalg.eigh(S)  # eigh 专用于实对称矩阵

print("对称矩阵 S 的特征值:", eigenvalues)
print("特征向量矩阵 Q (列为特征向量):")
print(eigenvectors.round(4))

# 验证正交性：Q^T Q 应为单位矩阵
QTQ = eigenvectors.T @ eigenvectors
print("\nQ^T Q (应为单位矩阵):")
print(QTQ.round(6))

# 验证：Av = λv
for i in range(3):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    residual = np.linalg.norm(S @ v - lam * v)
    print(f"特征值 λ{i+1} = {lam:.4f}, 残差 ||Av - λv|| = {residual:.2e}")
```

---

## 7. 特征空间在 AI 中的应用

### 7.1 协方差矩阵的特征空间 = 主成分方向

在 PCA 中，数据的协方差矩阵 $\Sigma$ 是实对称矩阵。其特征向量定义了数据变化最大的方向（主成分），特征值大小代表方差大小。

### 7.2 Google PageRank

PageRank 算法本质上是求**随机游走矩阵**对应特征值 $\lambda = 1$ 的特征向量（稳态分布）。

### 7.3 稳定性分析

在动力系统 $\dot{\mathbf{x}} = A\mathbf{x}$ 中，特征值的实部决定系统的稳定性：
- 所有特征值实部 $< 0$ → 系统稳定
- 任意特征值实部 $> 0$ → 系统不稳定

---

## 8. 小结

| 概念 | 定义 | 重要性 |
|------|------|--------|
| 特征向量 | $A\mathbf{v} = \lambda\mathbf{v}$，$\mathbf{v}\neq\mathbf{0}$ | 变换的不变方向 |
| 特征空间 | $E_\lambda = \ker(A-\lambda I)$ | 同一特征值的所有特征向量构成子空间 |
| 代数重数 | 特征多项式中的根次数 | 判断是否有"缺失"特征向量 |
| 几何重数 | $\dim E_\lambda$ | 实际可用的独立特征向量数 |
| 线性无关性 | 不同特征值→线性无关 | 对角化的基础 |

**下一篇**：D3 矩阵对角化——何时可以把矩阵写成 $A = PDP^{-1}$？条件是什么？怎么做？

---

*参考：Strang《Introduction to Linear Algebra》第 6 章；Lay《线性代数及其应用》*
