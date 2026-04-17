# D1 · 特征值（Eigenvalues）

> **Week 2 · Day 1**  
> 预计学习时间：90 分钟

---

## 学习目标

1. 理解"特征值"的几何直觉：矩阵对特定方向只做拉伸，不旋转
2. 掌握特征方程的推导：`det(A - λI) = 0`
3. 能手算 2×2、3×3 矩阵的特征值
4. 理解代数重数、迹与特征值之和、行列式与特征值之积的关系
5. 用 Python（numpy）验证计算结果

---

## 核心知识点

### 1. 直觉：矩阵是一种变换

设矩阵 $A \in \mathbb{R}^{n \times n}$，它将向量 $\mathbf{v}$ 映射到 $A\mathbf{v}$。

**普通向量**：方向改变 + 长度改变  
**特殊向量**：方向不变，只有长度缩放

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

- $\lambda$：**特征值（eigenvalue）**，标量，表示拉伸倍数
- $\mathbf{v}$：**特征向量（eigenvector）**，非零向量，表示不变的方向

> **几何直觉**：想象矩阵 A 是一块"橡皮泥变形器"。大多数方向会被扭曲，但存在若干"骨架方向"——沿这些方向只会被拉伸或压缩，不会旋转。这些骨架方向就是特征向量，拉伸倍数就是特征值。

---

### 2. 特征方程推导

从定义出发：

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

移项：

$$
(A - \lambda I)\mathbf{v} = \mathbf{0}
$$

要求 $\mathbf{v} \neq \mathbf{0}$（特征向量非零），则 $(A - \lambda I)$ 必须是**奇异矩阵**（不可逆），即：

$$
\boxed{\det(A - \lambda I) = 0}
$$

这个关于 $\lambda$ 的多项式方程称为**特征多项式**，其根就是特征值。

---

### 3. 特征多项式展开

设 $A \in \mathbb{R}^{n \times n}$，特征多项式为 $p(\lambda) = \det(A - \lambda I)$，这是关于 $\lambda$ 的 $n$ 次多项式。

**2×2 情形：**

$$
A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}
$$

$$
p(\lambda) = (a-\lambda)(d-\lambda) - bc = \lambda^2 - (a+d)\lambda + (ad-bc)
$$

$$
= \lambda^2 - \text{tr}(A)\lambda + \det(A)
$$

其中 $\text{tr}(A) = a + d$ 是矩阵的**迹（trace）**。

---

### 4. 特征值的重要性质

| 性质 | 公式 | 含义 |
|------|------|------|
| 迹 = 特征值之和 | $\text{tr}(A) = \sum_{i=1}^n \lambda_i$ | 对角线之和 |
| 行列式 = 特征值之积 | $\det(A) = \prod_{i=1}^n \lambda_i$ | 体积缩放因子 |
| 实对称矩阵特征值 | $\lambda_i \in \mathbb{R}$ | 特征值全为实数 |
| 正定矩阵特征值 | $\lambda_i > 0$ | 特征值全为正 |

**代数重数**：特征多项式中 $\lambda_i$ 作为根的重数，记为 $m_i$。

---

### 5. 典型矩阵的特征值

| 矩阵类型 | 特征值特点 |
|---------|-----------|
| 单位矩阵 $I$ | 所有特征值为 1 |
| 零矩阵 $0$ | 所有特征值为 0 |
| 对角矩阵 $D$ | 对角元素就是特征值 |
| 上/下三角矩阵 | 对角元素就是特征值 |
| 正交矩阵 | 特征值模为 1（$|\lambda|=1$） |
| 投影矩阵 $P$（$P^2=P$） | 特征值只有 0 或 1 |

---

## 示例推导

### 例 1：2×2 矩阵

$$
A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}
$$

**Step 1**：写出特征多项式

$$
p(\lambda) = \det\begin{pmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{pmatrix}
= (4-\lambda)(3-\lambda) - 2 \cdot 1
$$

$$
= \lambda^2 - 7\lambda + 12 - 2 = \lambda^2 - 7\lambda + 10
$$

**Step 2**：求根

$$
\lambda = \frac{7 \pm \sqrt{49 - 40}}{2} = \frac{7 \pm 3}{2}
$$

$$
\lambda_1 = 5, \quad \lambda_2 = 2
$$

**验证**：
- $\text{tr}(A) = 4 + 3 = 7 = \lambda_1 + \lambda_2$ ✓
- $\det(A) = 12 - 2 = 10 = \lambda_1 \cdot \lambda_2$ ✓

---

### 例 2：上三角矩阵

$$
B = \begin{pmatrix} 3 & 1 & 2 \\ 0 & -1 & 4 \\ 0 & 0 & 5 \end{pmatrix}
$$

上/下三角矩阵的特征值直接读对角线：

$$
\lambda_1 = 3, \quad \lambda_2 = -1, \quad \lambda_3 = 5
$$

**原因**：$\det(B - \lambda I)$ 展开后只有对角线项起作用。

---

### 例 3：旋转矩阵（复特征值）

$$
R = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \quad \text{（逆时针旋转 90°）}
$$

$$
p(\lambda) = \lambda^2 + 1 = 0 \implies \lambda = \pm i
$$

> 旋转矩阵在实数域没有特征向量（不存在"不旋转"的方向），但在复数域有。

---

## 动手练习

### 环境准备（使用 uv）

```bash
# 初始化项目
uv init eigenvalues-demo
cd eigenvalues-demo
uv add numpy matplotlib
```

### 练习 1：手动验证特征值

```python
# main.py
import numpy as np

# 例1：验证 2x2 矩阵
A = np.array([[4, 1],
              [2, 3]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(A)

print("矩阵 A:")
print(A)
print(f"\n特征值: {eigenvalues}")
print(f"\n迹(A) = {np.trace(A):.4f}, sum(λ) = {eigenvalues.sum():.4f}")
print(f"det(A) = {np.linalg.det(A):.4f}, prod(λ) = {eigenvalues.prod():.4f}")
```

**预期输出：**
```
特征值: [5. 2.]
迹(A) = 7.0000, sum(λ) = 7.0000
det(A) = 10.0000, prod(λ) = 10.0000
```

---

### 练习 2：特征值几何可视化

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_eigenvalue_stretch(A, title="Matrix Transform"):
    """可视化矩阵变换：特征方向只被拉伸，不被旋转"""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax in axes:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # 左图：变换前
    ax = axes[0]
    ax.set_title("变换前（原始向量）")
    
    # 画单位圆
    theta = np.linspace(0, 2*np.pi, 100)
    circle_pts = np.array([np.cos(theta), np.sin(theta)])
    ax.plot(circle_pts[0], circle_pts[1], 'b-', alpha=0.3, label='单位圆')
    
    # 画特征向量
    for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        if np.isreal(lam):
            v_real = v.real
            color = 'red' if i == 0 else 'green'
            ax.quiver(0, 0, v_real[0], v_real[1], 
                      angles='xy', scale_units='xy', scale=1,
                      color=color, label=f'v{i+1}: λ={lam.real:.1f}')
    
    ax.legend(loc='upper left', fontsize=8)
    
    # 右图：变换后
    ax = axes[1]
    ax.set_title(f"变换后（A × 向量）")
    
    # 变换后的椭圆
    transformed = A @ circle_pts
    ax.plot(transformed[0], transformed[1], 'b-', alpha=0.3, label='变换后椭圆')
    
    # 变换后的特征向量（应该只是拉伸）
    for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        if np.isreal(lam):
            v_real = v.real
            Av = A @ v_real
            color = 'red' if i == 0 else 'green'
            ax.quiver(0, 0, Av[0], Av[1],
                      angles='xy', scale_units='xy', scale=1,
                      color=color, label=f'Av{i+1} = {lam.real:.1f}·v{i+1}')
    
    ax.legend(loc='upper left', fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("eigenvalue_viz.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("图像已保存到 eigenvalue_viz.png")

A = np.array([[4, 1], [2, 3]], dtype=float)
visualize_eigenvalue_stretch(A, "A = [[4,1],[2,3]] 的特征值几何")
```

---

### 练习 3：探索特殊矩阵的特征值

```python
import numpy as np

def analyze_matrix(A, name):
    """分析矩阵的特征值，并验证迹/行列式关系"""
    eigenvalues = np.linalg.eigvals(A)
    print(f"\n{'='*50}")
    print(f"矩阵：{name}")
    print(A)
    print(f"特征值：{np.round(eigenvalues, 4)}")
    print(f"迹(sum λ)：{eigenvalues.sum().real:.4f} vs {np.trace(A):.4f}")
    print(f"行列式(prod λ)：{np.prod(eigenvalues).real:.4f} vs {np.linalg.det(A):.4f}")

# 单位矩阵
analyze_matrix(np.eye(3), "单位矩阵 I₃")

# 投影矩阵（P² = P）
P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
analyze_matrix(P, "投影矩阵（xy平面）")

# 实对称矩阵
S = np.array([[4, 2, 1], [2, 3, 0], [1, 0, 5]], dtype=float)
analyze_matrix(S, "实对称矩阵 S")
print(f"特征值全为实数？{np.all(np.isreal(np.linalg.eigvals(S)))}")

# 旋转矩阵（复特征值）
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
analyze_matrix(R, f"旋转矩阵（θ=45°）")
```

---

## 小结

| 概念 | 核心公式 | 记忆点 |
|------|----------|--------|
| 特征值定义 | $A\mathbf{v} = \lambda\mathbf{v}$ | 方向不变，只拉伸 |
| 特征方程 | $\det(A-\lambda I)=0$ | 矩阵奇异 ⟺ 有非零解 |
| 迹与特征值 | $\text{tr}(A) = \sum\lambda_i$ | 对角线 = 特征值总和 |
| 行列式与特征值 | $\det(A) = \prod\lambda_i$ | 体积变化 = 拉伸倍数之积 |

**容易踩的坑：**
- 特征值可以是复数（旋转矩阵）
- 特征值可以重复（代数重数 > 1）
- $\lambda = 0$ 是合法特征值（表示矩阵奇异，把某方向压缩成零）

---

## ML 中的应用

### 1. 协方差矩阵特征值 → PCA

数据矩阵 $X$ 的协方差矩阵 $C = \frac{1}{n}X^TX$ 是实对称正半定矩阵，其特征值 $\lambda_i$ 代表沿各主成分方向的**方差**。最大特征值对应数据变化最大的方向。

### 2. Google PageRank

PageRank 向量是转移矩阵 $M$ 的**主特征向量**（对应特征值 $\lambda=1$）。网页重要性由迭代求特征向量得出。

### 3. 图神经网络（GNN）中的图拉普拉斯

图拉普拉斯矩阵 $L = D - W$ 的特征值（图谱）描述图的连通性。谱图卷积在特征值空间操作。

### 4. 稳定性分析

动力系统 $\mathbf{x}_{t+1} = A\mathbf{x}_t$ 的稳定性由 $A$ 的**谱半径** $\rho(A) = \max|\lambda_i|$ 决定：
- $\rho(A) < 1$：收敛（稳定）
- $\rho(A) > 1$：发散（不稳定）
- $\rho(A) = 1$：临界（振荡）

这在 RNN 梯度消失/爆炸分析中直接用到。

---

*→ 明天 D2：特征向量，从特征值到特征空间*
