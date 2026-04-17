# D1：特征值的定义与几何意义

> **Week 2 · Day 1** | AI 基础线：线性代数核心概念

---

## 1. 为什么需要特征值？

在机器学习、信号处理、量子力学中，我们常常面对这样的问题：**一个线性变换对哪些方向不改变方向，只改变长度？**

回想矩阵乘法 $A\mathbf{v}$：一般情况下，向量 $\mathbf{v}$ 经过矩阵 $A$ 作用后，方向和长度都会发生变化。但存在某些特殊的向量，经过 $A$ 作用后**方向不变**（或恰好反向），只有长度发生缩放。这些特殊向量就是**特征向量**，对应的缩放比例就是**特征值**。

特征值分解是理解 PCA、SVD、谱聚类、PageRank 等算法的基石，掌握它是进入 AI 核心数学的第一步。

---

## 2. 特征值与特征向量的定义

### 2.1 正式定义

设 $A$ 是一个 $n \times n$ 的方阵，若存在**非零向量** $\mathbf{v} \in \mathbb{R}^n$ 和标量 $\lambda \in \mathbb{R}$（或 $\mathbb{C}$），使得：

$$
A\mathbf{v} = \lambda\mathbf{v}
$$

则称 $\lambda$ 为矩阵 $A$ 的**特征值**（Eigenvalue），$\mathbf{v}$ 为对应于 $\lambda$ 的**特征向量**（Eigenvector）。

**关键约束**：$\mathbf{v} \neq \mathbf{0}$（零向量对任意 $\lambda$ 都满足等式，没有意义）。

### 2.2 直觉理解

- 若 $\lambda = 2$：矩阵 $A$ 将向量 $\mathbf{v}$ 沿原方向拉伸至 2 倍
- 若 $\lambda = -1$：矩阵 $A$ 将向量 $\mathbf{v}$ 反向，长度不变
- 若 $\lambda = 0$：矩阵 $A$ 将向量 $\mathbf{v}$ 压缩为零向量（此时 $A$ 是奇异矩阵）

---

## 3. 如何求特征值？

### 3.1 特征方程

由定义 $A\mathbf{v} = \lambda\mathbf{v}$ 变形：

$$
A\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}
$$

$$
(A - \lambda I)\mathbf{v} = \mathbf{0}
$$

这是一个齐次线性方程组。要使非零解存在，系数矩阵 $(A - \lambda I)$ 必须**不可逆**，即行列式为零：

$$
\det(A - \lambda I) = 0
$$

这个方程称为**特征方程**（Characteristic Equation），也叫**特征多项式**方程。

### 3.2 特征多项式

展开 $\det(A - \lambda I)$ 得到关于 $\lambda$ 的多项式 $p(\lambda)$，称为特征多项式：

$$
p(\lambda) = \det(A - \lambda I) = (-1)^n \lambda^n + c_{n-1}\lambda^{n-1} + \cdots + c_0
$$

对 $n \times n$ 矩阵，特征多项式是 $n$ 次多项式，有 $n$ 个根（含重根，允许复数根）。

### 3.3 手推示例：2×2 矩阵

设：

$$
A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}
$$

特征方程：

$$
\det(A - \lambda I) = \det\begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix} = (3-\lambda)(2-\lambda) - 0 = 0
$$

$$
\lambda^2 - 5\lambda + 6 = 0
$$

$$
(\lambda - 2)(\lambda - 3) = 0
$$

解得：$\lambda_1 = 2$，$\lambda_2 = 3$

---

## 4. 特征值的几何意义

### 4.1 变换的"不动方向"

每个特征向量代表矩阵变换中的一个**主轴方向**。沿着这个方向，变换只做伸缩，不做旋转。

以旋转矩阵为例：

$$
R = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
$$

当 $\theta \neq 0, \pi$ 时，在实数域没有特征值（没有不变方向），但在复数域有 $e^{i\theta}$ 和 $e^{-i\theta}$。

### 4.2 行列式与迹的关系

对于 $n \times n$ 矩阵，特征值满足：

$$
\det(A) = \prod_{i=1}^{n} \lambda_i
$$

$$
\text{tr}(A) = \sum_{i=1}^{n} \lambda_i
$$

其中 $\text{tr}(A) = a_{11} + a_{22} + \cdots + a_{nn}$ 是矩阵的**迹**。

这两个关系给了我们快速验算的工具：上面例子中 $\det(A) = 6 = 2 \times 3$，$\text{tr}(A) = 5 = 2 + 3$，验证正确。

### 4.3 谱半径

矩阵所有特征值的绝对值最大值称为**谱半径**：

$$
\rho(A) = \max_i |\lambda_i|
$$

谱半径决定了矩阵幂次迭代的收敛性：若 $\rho(A) < 1$，则 $A^k \to 0$（$k \to \infty$）。

---

## 5. 特殊矩阵的特征值性质

| 矩阵类型 | 特征值特性 |
|---------|-----------|
| 实对称矩阵 | 特征值均为实数 |
| 正定矩阵 | 特征值均为正实数 |
| 正交矩阵 | 特征值绝对值为 1（$|\lambda| = 1$） |
| 上/下三角矩阵 | 特征值即为对角元素 |
| 幂等矩阵（$A^2=A$） | 特征值只有 0 和 1 |

---

## 6. Python 代码示例

### 6.1 环境准备（uv 管理）

```bash
# 初始化项目
uv init eigenvalue-demo
cd eigenvalue-demo

# 添加依赖
uv add numpy matplotlib
```

### 6.2 计算特征值并可视化

```python
# main.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 定义矩阵
A = np.array([[3, 1],
              [0, 2]], dtype=float)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print("矩阵 A:")
print(A)
print(f"\n特征值: {eigenvalues}")
print(f"\n特征向量 (列向量):\n{eigenvectors}")

# 验证 Av = λv
for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
    Av = A @ v
    lam_v = lam * v
    print(f"\n特征值 λ{i+1} = {lam:.2f}")
    print(f"  Av    = {Av}")
    print(f"  λ·v   = {lam_v}")
    print(f"  误差  = {np.linalg.norm(Av - lam_v):.2e}")

# 验证迹和行列式
print(f"\n迹: tr(A) = {np.trace(A):.2f}, sum(λ) = {sum(eigenvalues):.2f}")
print(f"行列式: det(A) = {np.linalg.det(A):.2f}, prod(λ) = {np.prod(eigenvalues):.2f}")

# 可视化：展示特征向量方向
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 随机向量 vs 特征向量的变换效果
ax1 = axes[0]
origin = np.zeros(2)

colors = ['#E74C3C', '#3498DB']
for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
    v_norm = v / np.linalg.norm(v)
    ax1.quiver(*origin, *v_norm, color=colors[i], scale=1, scale_units='xy',
               angles='xy', width=0.015, label=f'v{i+1} (λ={lam:.0f})')
    # 变换后
    Av_norm = (A @ v_norm) / np.linalg.norm(v_norm)
    ax1.quiver(*origin, *Av_norm, color=colors[i], scale=1, scale_units='xy',
               angles='xy', width=0.008, alpha=0.4)

ax1.set_xlim(-1, 4)
ax1.set_ylim(-1, 4)
ax1.axhline(0, color='k', linewidth=0.5)
ax1.axvline(0, color='k', linewidth=0.5)
ax1.legend()
ax1.set_title('特征向量（实线）与变换后（虚线）')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# 特征值在复平面的分布
ax2 = axes[1]
ax2.scatter(eigenvalues.real, np.zeros_like(eigenvalues), s=100, c=colors, zorder=5)
for i, lam in enumerate(eigenvalues):
    ax2.annotate(f'λ{i+1}={lam:.1f}', (lam.real, 0), textcoords="offset points",
                 xytext=(0, 15), ha='center', color=colors[i], fontsize=12)
ax2.axhline(0, color='k', linewidth=0.5)
ax2.set_xlim(0, 5)
ax2.set_ylim(-1, 1)
ax2.set_title('特征值的数轴分布')
ax2.set_xlabel('Re(λ)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eigenvalue_visualization.png', dpi=150, bbox_inches='tight')
print("\n图像已保存为 eigenvalue_visualization.png")
```

运行：

```bash
uv run python main.py
```

### 6.3 用幂迭代法（Power Iteration）求最大特征值

```python
# power_iteration.py
import numpy as np

def power_iteration(A, num_iter=100, tol=1e-10):
    """用幂迭代法求矩阵最大特征值"""
    n = A.shape[0]
    # 随机初始化向量
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    lambda_old = 0
    for i in range(num_iter):
        # 矩阵-向量乘法
        w = A @ v
        # 更新特征值估计（Rayleigh 商）
        lambda_new = v @ w
        # 归一化
        v = w / np.linalg.norm(w)
        
        if abs(lambda_new - lambda_old) < tol:
            print(f"  收敛于第 {i+1} 次迭代")
            break
        lambda_old = lambda_new
    
    return lambda_new, v

A = np.array([[4, 1],
              [2, 3]], dtype=float)

lam_approx, v_approx = power_iteration(A)
lam_true, v_true = np.linalg.eig(A)

print(f"幂迭代法最大特征值: {lam_approx:.6f}")
print(f"numpy 最大特征值:   {max(lam_true):.6f}")
```

---

## 7. 小结

| 概念 | 要点 |
|------|------|
| 定义 | $A\mathbf{v} = \lambda\mathbf{v}$，$\mathbf{v} \neq \mathbf{0}$ |
| 求法 | 令 $\det(A - \lambda I) = 0$，解特征多项式 |
| 几何意义 | 特征向量是变换的"不动方向"，特征值是伸缩比例 |
| 重要性质 | $\text{tr}(A) = \sum\lambda_i$，$\det(A) = \prod\lambda_i$ |
| AI 应用 | PCA 方差方向、谱聚类、稳定性分析、PageRank |

**下一篇**：D2 特征向量与特征空间——当一个特征值对应多个线性无关的特征向量时，会发生什么？

---

*参考：Gilbert Strang《线性代数及其应用》第 6 章；3Blue1Brown "Essence of Linear Algebra" EP14*
