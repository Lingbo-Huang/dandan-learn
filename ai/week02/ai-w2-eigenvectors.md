# D2 · 特征向量（Eigenvectors）

> **Week 2 · Day 2**  
> 预计学习时间：90 分钟

---

## 学习目标

1. 理解特征向量的计算方法：求 $(A - \lambda I)$ 的零空间
2. 掌握几何重数与代数重数的区别
3. 理解特征空间（eigenspace）的结构
4. 掌握实对称矩阵特征向量的正交性
5. 能用 Python 验证并可视化特征向量

---

## 核心知识点

### 1. 从特征值到特征向量

已知特征值 $\lambda$，求特征向量 $\mathbf{v}$：

$$
(A - \lambda I)\mathbf{v} = \mathbf{0}
$$

这是一个**齐次线性方程组**，其解集（包含零向量）称为 $A$ 关于 $\lambda$ 的**特征空间**：

$$
E_\lambda = \text{Null}(A - \lambda I) = \ker(A - \lambda I)
$$

特征向量是特征空间中所有**非零**向量。

> **关键点**：特征向量不唯一——如果 $\mathbf{v}$ 是特征向量，那么 $c\mathbf{v}$（$c \neq 0$）也是。习惯上取单位向量（归一化）。

---

### 2. 代数重数 vs 几何重数

| 概念 | 定义 | 范围 |
|------|------|------|
| **代数重数** $m_a(\lambda)$ | 特征值 $\lambda$ 在特征多项式中的根的重数 | $\geq 1$ |
| **几何重数** $m_g(\lambda)$ | 特征空间的维数 $\dim(E_\lambda)$ | $\geq 1$ |

**重要不等式**：

$$
1 \leq m_g(\lambda) \leq m_a(\lambda)
$$

- $m_g = m_a$：特征值"行为正常"，可对角化
- $m_g < m_a$：特征值"缺陷"，矩阵**不可对角化**（亏损矩阵）

---

### 3. 实对称矩阵：特征向量正交

**谱定理（Spectral Theorem）**：若 $A = A^T$（实对称矩阵），则：

1. **所有特征值为实数**
2. **不同特征值对应的特征向量互相正交**

**证明（不同特征值情形）**：

设 $A\mathbf{u} = \lambda_1\mathbf{u}$，$A\mathbf{v} = \lambda_2\mathbf{v}$，$\lambda_1 \neq \lambda_2$

$$
\lambda_1 \mathbf{u}^T\mathbf{v} = (A\mathbf{u})^T\mathbf{v} = \mathbf{u}^T A^T \mathbf{v} = \mathbf{u}^T A \mathbf{v} = \lambda_2 \mathbf{u}^T\mathbf{v}
$$

$$
(\lambda_1 - \lambda_2)\mathbf{u}^T\mathbf{v} = 0 \implies \mathbf{u}^T\mathbf{v} = 0 \quad \text{（因为 } \lambda_1 \neq \lambda_2\text{）}
$$

这是 PCA/SVD 正交基的根本来源。

---

### 4. 特征向量的几何意义

| 矩阵 | 特征向量含义 |
|------|-------------|
| 投影矩阵 $P$ | 投影方向（λ=1）和被消除方向（λ=0） |
| 反射矩阵 | 镜面方向（λ=1）和垂直方向（λ=-1） |
| 协方差矩阵 | 数据主成分方向（方差最大方向） |
| 图拉普拉斯 | 图的"振动模式" |

---

### 5. 复特征向量（了解即可）

当矩阵不是实对称时，特征值可能是复数 $\lambda = a + bi$，对应特征向量也是复向量。

对于实矩阵，复特征值一定成共轭对出现：$\lambda, \bar{\lambda}$

---

## 示例推导

### 例 1：计算特征向量（详细步骤）

$$
A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}, \quad \lambda_1 = 5, \lambda_2 = 2
$$

**求 $\lambda_1 = 5$ 对应的特征向量：**

$$
(A - 5I)\mathbf{v} = \begin{pmatrix} -1 & 1 \\ 2 & -2 \end{pmatrix}\mathbf{v} = \mathbf{0}
$$

行化简：

$$
\begin{pmatrix} -1 & 1 \\ 2 & -2 \end{pmatrix} \xrightarrow{R_2 + 2R_1} \begin{pmatrix} -1 & 1 \\ 0 & 0 \end{pmatrix}
$$

解：$-v_1 + v_2 = 0 \implies v_2 = v_1$，令 $v_1 = 1$：

$$
\mathbf{v}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix} \quad \text{（或归一化：} \frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\end{pmatrix}\text{）}
$$

**求 $\lambda_2 = 2$ 对应的特征向量：**

$$
(A - 2I)\mathbf{v} = \begin{pmatrix} 2 & 1 \\ 2 & 1 \end{pmatrix}\mathbf{v} = \mathbf{0}
$$

解：$2v_1 + v_2 = 0 \implies v_2 = -2v_1$，令 $v_1 = 1$：

$$
\mathbf{v}_2 = \begin{pmatrix} 1 \\ -2 \end{pmatrix}
$$

**验证**：$A\mathbf{v}_1 = \begin{pmatrix}5\\5\end{pmatrix} = 5\mathbf{v}_1$ ✓，$A\mathbf{v}_2 = \begin{pmatrix}2\\-4\end{pmatrix} = 2\mathbf{v}_2$ ✓

---

### 例 2：重复特征值（几何重数 < 代数重数）

$$
B = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}
$$

特征多项式：$(2-\lambda)^2 = 0 \implies \lambda = 2$（代数重数 = 2）

$$
(B - 2I) = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}
$$

零空间维数 = 1（只有 $v_2 = 0$ 的约束，$v_1$ 自由），所以几何重数 = 1 < 2

只有一个线性独立的特征向量：$\mathbf{v} = \begin{pmatrix}1\\0\end{pmatrix}$

这个矩阵**不可对角化**（这正是 Jordan 标准型处理的情况）。

---

### 例 3：实对称矩阵的正交特征向量

$$
S = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}
$$

特征值：$\lambda_1 = 4, \lambda_2 = 2$

$\lambda_1 = 4$：$(S-4I)\mathbf{v} = \begin{pmatrix}-1&1\\1&-1\end{pmatrix}\mathbf{v}=0 \implies \mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\end{pmatrix}$

$\lambda_2 = 2$：$(S-2I)\mathbf{v} = \begin{pmatrix}1&1\\1&1\end{pmatrix}\mathbf{v}=0 \implies \mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\-1\end{pmatrix}$

验证正交性：$\mathbf{v}_1^T\mathbf{v}_2 = \frac{1}{2}(1\cdot1 + 1\cdot(-1)) = 0$ ✓

---

## 动手练习

### 环境准备

```bash
uv init eigenvectors-demo
cd eigenvectors-demo
uv add numpy matplotlib scipy
```

### 练习 1：完整特征分解与验证

```python
# main.py
import numpy as np

def full_eigen_analysis(A, name="A"):
    """完整的特征值/特征向量分析"""
    print(f"\n{'='*60}")
    print(f"矩阵 {name}:")
    print(A)
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"\n特征值: {np.round(eigenvalues, 4)}")
    print(f"\n特征向量（列向量）:")
    print(np.round(eigenvectors, 4))
    
    print("\n验证 Av = λv:")
    for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        Av = A @ v
        lam_v = lam * v
        print(f"  λ{i+1}={lam.real:.4f}: Av = {np.round(Av.real, 4)}, λv = {np.round(lam_v.real, 4)}, 匹配? {np.allclose(Av, lam_v)}")
    
    return eigenvalues, eigenvectors

# 测试
A = np.array([[4, 1], [2, 3]], dtype=float)
full_eigen_analysis(A, "[[4,1],[2,3]]")

# 对称矩阵（验证正交性）
S = np.array([[3, 1], [1, 3]], dtype=float)
eigenvalues, eigenvectors = full_eigen_analysis(S, "对称矩阵 [[3,1],[1,3]]")
print(f"\n特征向量内积（应为0）: {eigenvectors[:, 0] @ eigenvectors[:, 1]:.6f}")
print(f"特征向量是否正交: {np.isclose(eigenvectors[:, 0] @ eigenvectors[:, 1], 0)}")
```

---

### 练习 2：可视化特征向量方向

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_eigenvectors(A, title="特征向量可视化"):
    """在变换前后显示特征向量"""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # 只处理实特征值
    real_mask = np.isreal(eigenvalues)
    eigenvalues = eigenvalues[real_mask].real
    eigenvectors = eigenvectors[:, real_mask].real
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#E74C3C', '#2ECC71', '#3498DB', '#9B59B6']
    
    for ax_idx, (ax, transform) in enumerate(zip(axes, [False, True])):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 画单位圆/变换后椭圆
        theta = np.linspace(0, 2*np.pi, 200)
        circle = np.array([np.cos(theta), np.sin(theta)])
        pts = A @ circle if transform else circle
        ax.plot(pts[0], pts[1], 'b-', alpha=0.2, lw=2, label='单位圆→椭圆')
        
        # 画特征向量（变换后仍是同方向）
        for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
            if transform:
                v_plot = A @ v  # = λv
                label = f'Av = {lam:.1f}v (λ={lam:.1f})'
            else:
                v_plot = v
                label = f'v{i+1} (λ={lam:.1f})'
            
            ax.annotate('', xy=(v_plot[0], v_plot[1]), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color=colors[i], lw=2.5))
            ax.text(v_plot[0]*1.1, v_plot[1]*1.1, label, 
                   color=colors[i], fontsize=9, fontweight='bold')
        
        ax.set_title("变换前：原始特征向量" if not transform else "变换后：方向不变，长度=λ")
    
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("eigenvectors_viz.png", dpi=150, bbox_inches='tight')
    plt.show()

A = np.array([[4, 1], [2, 3]], dtype=float)
plot_eigenvectors(A, f"A = [[4,1],[2,3]] 的特征向量")
```

---

### 练习 3：缺陷矩阵（几何重数 < 代数重数）

```python
import numpy as np

# Jordan 块（不可对角化）
J = np.array([[2, 1], [0, 2]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(J)
print("Jordan 块 J = [[2,1],[0,2]]")
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")
print(f"\n特征向量数量（线性独立）: 注意两列几乎相同")

# 验证：只有一个真正独立的特征向量
null_space_dim = 2 - np.linalg.matrix_rank(J - 2 * np.eye(2))
print(f"几何重数（零空间维数）: {null_space_dim}")
print(f"代数重数: 2")
print(f"结论：代数重数 > 几何重数，矩阵不可对角化")

# 对比：可对角化的情况
D = np.array([[2, 0], [0, 2]], dtype=float)  # 数量矩阵
evals, evecs = np.linalg.eig(D)
null_dim = 2 - np.linalg.matrix_rank(D - 2 * np.eye(2))
print(f"\n数量矩阵 2I 的几何重数: {null_dim}（等于代数重数2，可对角化）")
```

---

## 小结

| 概念 | 计算方法 | 注意点 |
|------|---------|--------|
| 特征向量 | 解 $(A-\lambda I)\mathbf{v}=\mathbf{0}$ | $\mathbf{v} \neq \mathbf{0}$ |
| 特征空间 | $\ker(A-\lambda I)$ | 维数 = 几何重数 |
| 几何重数 | $n - \text{rank}(A-\lambda I)$ | $\leq$ 代数重数 |
| 正交性 | 实对称矩阵保证 | 不同特征值 → 正交 |

**三大陷阱：**
1. 特征向量可以任意缩放 → 习惯取单位向量
2. 重复特征值不一定缺陷（数量矩阵 $cI$ 的特征空间是全空间）
3. numpy 的 `eig` 不保证特征向量正交（用 `eigh` 处理实对称矩阵）

---

## ML 中的应用

### 1. PCA 的特征向量 = 主成分方向

协方差矩阵 $C$ 的特征向量定义了数据变换的坐标系。第一主成分是最大特征值对应的特征向量。

### 2. Word2Vec 的语义空间

词嵌入矩阵经 SVD 后，左奇异向量（类似特征向量）对应语义维度。著名的"king - man + woman = queen"就是在这个向量空间中操作。

### 3. 谱聚类（Spectral Clustering）

对图拉普拉斯矩阵 $L$ 做特征分解，用前 $k$ 个特征向量（对应最小 $k$ 个特征值）作为新的节点表示，再用 K-Means 聚类。

### 4. RNN 梯度消失分析

反向传播中梯度通过 $W^T$ 传递，若 $\|W\|$ 的谱范数（最大奇异值 = 最大特征值的平方根）< 1，梯度消失；> 1，梯度爆炸。

---

*→ 明天 D3：对角化，把矩阵分解为更简单的形式*
