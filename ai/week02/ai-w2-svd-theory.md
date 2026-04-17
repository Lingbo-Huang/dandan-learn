# D4：SVD 定理与几何直觉

> **Week 2 · Day 4** | AI 基础线：线性代数核心概念

---

## 1. 为什么需要 SVD？

特征值分解（EVD）有两个局限：
1. **只适用于方阵**（$n \times n$）
2. **不是所有矩阵都可对角化**（如 Jordan 块）

奇异值分解（Singular Value Decomposition, SVD）完美解决了这两个问题：

> **任意矩阵**（无论 $m \times n$，无论是否可对角化）都有 SVD。

SVD 是现代数值线性代数最重要的工具之一，是 PCA、LSA（潜在语义分析）、推荐系统、图像压缩、计算机视觉的基础。

---

## 2. SVD 定理

### 2.1 定理陈述

**定理（奇异值分解）**：设 $A$ 是 $m \times n$ 的实矩阵（$m \geq n$），则存在分解：

$$
\boxed{A = U\Sigma V^T}
$$

其中：
- $U$ 是 $m \times m$ 的**正交矩阵**（$U^TU = UU^T = I_m$），其列称为**左奇异向量**
- $V$ 是 $n \times n$ 的**正交矩阵**（$V^TV = VV^T = I_n$），其列称为**右奇异向量**
- $\Sigma$ 是 $m \times n$ 的**矩形对角矩阵**，对角元素 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0 = \cdots = 0$，称为**奇异值**

奇异值的个数 $r = \text{rank}(A)$。

### 2.2 形状示意

$$
\underbrace{A}_{m \times n} = \underbrace{U}_{m \times m} \underbrace{\Sigma}_{m \times n} \underbrace{V^T}_{n \times n}
$$

$\Sigma$ 的结构（以 $m > n$ 为例）：

$$
\Sigma = \begin{pmatrix} \sigma_1 & & & \\ & \sigma_2 & & \\ & & \ddots & \\ & & & \sigma_n \\ \hline 0 & 0 & \cdots & 0 \\ \vdots & & & \vdots \end{pmatrix}
$$

### 2.3 紧凑形式（Thin SVD）

实际计算中常用"薄型 SVD"：

$$
A = U_r \Sigma_r V_r^T
$$

其中 $U_r$（$m \times r$），$\Sigma_r$（$r \times r$），$V_r$（$n \times r$），$r = \text{rank}(A)$。

---

## 3. 奇异值的来源：与特征值的关系

### 3.1 通过 $A^TA$ 和 $AA^T$ 推导

考虑 $A = U\Sigma V^T$，则：

$$
A^TA = (V\Sigma^T U^T)(U\Sigma V^T) = V(\Sigma^T\Sigma)V^T
$$

$$
AA^T = (U\Sigma V^T)(V\Sigma^T U^T) = U(\Sigma\Sigma^T)U^T
$$

**关键结论：**
- $A^TA$ 的特征值是 $\sigma_1^2, \sigma_2^2, \ldots, \sigma_n^2$（加上若干 0）
- $AA^T$ 的特征值是 $\sigma_1^2, \sigma_2^2, \ldots, \sigma_m^2$（加上若干 0）
- $V$ 的列是 $A^TA$ 的特征向量（右奇异向量）
- $U$ 的列是 $AA^T$ 的特征向量（左奇异向量）
- 奇异值 $\sigma_i = \sqrt{\lambda_i(A^TA)}$

这提供了一种**计算 SVD 的方法**：先对 $A^TA$ 做特征值分解，再求 $\sigma_i$ 和 $U$。

### 3.2 手推推导 $u_i$

已知 $V$ 和 $\sigma_i$，左奇异向量由下式确定：

$$
A\mathbf{v}_i = \sigma_i \mathbf{u}_i \Rightarrow \mathbf{u}_i = \frac{A\mathbf{v}_i}{\sigma_i}
$$

---

## 4. 几何直觉：SVD = 旋转 + 缩放 + 旋转

### 4.1 三步变换

任意线性变换 $\mathbf{y} = A\mathbf{x}$ 可分解为：

$$
\mathbf{x} \xrightarrow{V^T \text{（旋转）}} \tilde{\mathbf{x}} \xrightarrow{\Sigma \text{（缩放）}} \tilde{\mathbf{y}} \xrightarrow{U \text{（旋转）}} \mathbf{y}
$$

具体地：
1. $V^T$ 将输入空间旋转到"标准坐标系"（$V$ 的列是输入空间的自然轴）
2. $\Sigma$ 沿各轴伸缩（奇异值是缩放比例）
3. $U$ 再将结果旋转到输出空间的自然坐标系

### 4.2 单位球的变换

$n$ 维单位球（$\|\mathbf{x}\|=1$）经矩阵 $A$ 变换后，变成 $m$ 维空间中的**椭球体**：
- 椭球体的各半轴长度为奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots$
- 各半轴方向为左奇异向量 $\mathbf{u}_1, \mathbf{u}_2, \ldots$
- 对应输入方向为右奇异向量 $\mathbf{v}_1, \mathbf{v}_2, \ldots$

$$
A\mathbf{v}_i = \sigma_i \mathbf{u}_i
$$

**最大奇异值 $\sigma_1$ 就是矩阵 $A$ 的 2-范数（谱范数）：**

$$
\|A\|_2 = \sigma_1 = \max_{\|\mathbf{x}\|=1} \|A\mathbf{x}\|
$$

### 4.3 Frobenius 范数与奇异值

$$
\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\sum_i \sigma_i^2} = \sqrt{\text{tr}(A^TA)}
$$

---

## 5. 低秩近似：Eckart-Young 定理

### 5.1 定理

**定理（Eckart-Young-Mirsky）**：在所有秩至多为 $k$ 的矩阵中，$A$ 的最佳近似（Frobenius 范数意义）是：

$$
A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T = U_k \Sigma_k V_k^T
$$

最优近似误差为：

$$
\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2}
$$

**这正是 PCA 降维的理论基础。**

### 5.2 "外积展开"视角

SVD 的等价写法：

$$
A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

每一项 $\sigma_i \mathbf{u}_i \mathbf{v}_i^T$ 都是秩-1 矩阵，其"重要性"由 $\sigma_i$ 决定。截取前 $k$ 项就得到最优低秩近似。

---

## 6. SVD 与矩阵四大子空间

SVD 自然地给出矩阵的**四大基本子空间**：

| 子空间 | 由 SVD 给出的正交基 | 维数 |
|------|------------------|------|
| 列空间（值域）$\text{Col}(A)$ | $U$ 的前 $r$ 列 $\mathbf{u}_1, \ldots, \mathbf{u}_r$ | $r$ |
| 零空间 $\text{Null}(A)$ | $V$ 的后 $n-r$ 列 $\mathbf{v}_{r+1}, \ldots, \mathbf{v}_n$ | $n-r$ |
| 行空间 $\text{Row}(A)$ | $V$ 的前 $r$ 列 $\mathbf{v}_1, \ldots, \mathbf{v}_r$ | $r$ |
| 左零空间 $\text{Null}(A^T)$ | $U$ 的后 $m-r$ 列 $\mathbf{u}_{r+1}, \ldots, \mathbf{u}_m$ | $m-r$ |

---

## 7. Python 代码示例（SVD 定理验证）

### 7.1 环境准备

```bash
uv init svd-theory-demo
cd svd-theory-demo
uv add numpy matplotlib
```

### 7.2 验证 SVD 分解与几何意义

```python
# svd_geometry.py
import numpy as np
import matplotlib.pyplot as plt

# ============ 验证 SVD 分解 ============
A = np.array([[3., 2., 1.],
              [2., 3., 1.],
              [1., 1., 4.],
              [0., 1., 2.]])

U, s, Vt = np.linalg.svd(A, full_matrices=True)
Sigma = np.zeros_like(A)
for i, si in enumerate(s):
    Sigma[i, i] = si

print(f"A shape: {A.shape}")
print(f"U shape: {U.shape}, Σ shape: {Sigma.shape}, V^T shape: {Vt.shape}")
print(f"\n奇异值: {s.round(4)}")
print(f"秩: {np.linalg.matrix_rank(A)}")

# 验证 A = U Σ V^T
A_reconstructed = U @ Sigma @ Vt
print(f"\n重构误差: {np.linalg.norm(A - A_reconstructed):.2e}")

# 验证 U, V 为正交矩阵
print(f"||U^T U - I||: {np.linalg.norm(U.T @ U - np.eye(U.shape[0])):.2e}")
print(f"||V^T V - I||: {np.linalg.norm(Vt @ Vt.T - np.eye(Vt.shape[0])):.2e}")

# 验证奇异值来自 A^T A 的特征值
ATA = A.T @ A
eigenvalues_ATA = np.linalg.eigvalsh(ATA)
print(f"\nA^T A 特征值 (升序): {eigenvalues_ATA.round(4)}")
print(f"σ^2 (降序): {(s**2).round(4)}")

# ============ 单位圆的变换（2D 例子）============
A2 = np.array([[3., 1.],
               [1., 2.]])
U2, s2, Vt2 = np.linalg.svd(A2)

theta = np.linspace(0, 2*np.pi, 300)
circle = np.stack([np.cos(theta), np.sin(theta)])  # 单位圆上的点
ellipse = A2 @ circle  # 变换后

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.plot(*circle, 'b-', linewidth=2, label='单位圆')
origin = np.zeros(2)
# 右奇异向量（输入方向）
for i in range(2):
    v = Vt2[i]
    ax1.annotate('', xy=v, xytext=origin,
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(v[0]*1.1, v[1]*1.1, f'$v_{i+1}$', fontsize=12, color='red')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.legend()
ax1.set_title('变换前：单位圆 + 右奇异向量')
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='k', lw=0.5)
ax1.axvline(0, color='k', lw=0.5)

ax2 = axes[1]
ax2.plot(*ellipse, 'g-', linewidth=2, label='变换后（椭圆）')
# 左奇异向量 × 奇异值（椭圆半轴）
for i in range(2):
    u = U2[:, i] * s2[i]
    ax2.annotate('', xy=u, xytext=origin,
                 arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax2.text(u[0]*1.1, u[1]*1.1, f'σ{i+1}u{i+1}={s2[i]:.2f}', fontsize=10, color='orange')
ax2.set_aspect('equal')
ax2.legend()
ax2.set_title('变换后：椭圆 + 左奇异向量（半轴）')
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='k', lw=0.5)
ax2.axvline(0, color='k', lw=0.5)

plt.tight_layout()
plt.savefig('svd_geometry.png', dpi=150, bbox_inches='tight')
print("\n几何图像已保存")
```

### 7.3 低秩近似误差

```python
# low_rank_approx.py
import numpy as np

np.random.seed(42)
A = np.random.randn(10, 8)
U, s, Vt = np.linalg.svd(A, full_matrices=False)

print("各截断秩的近似误差：")
print(f"{'秩 k':<6} {'||A - A_k||_F':<18} {'占总方差%':<15}")
total_var = np.sum(s**2)
for k in range(1, len(s)+1):
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    err = np.linalg.norm(A - A_k, 'fro')
    var_explained = np.sum(s[:k]**2) / total_var * 100
    print(f"{k:<6} {err:<18.4f} {var_explained:<15.2f}%")
```

---

## 8. 小结

| 概念 | 内容 |
|------|------|
| SVD 定义 | $A = U\Sigma V^T$，$U, V$ 正交，$\Sigma$ 矩形对角 |
| 奇异值来源 | $\sigma_i = \sqrt{\lambda_i(A^TA)}$ |
| 几何意义 | 旋转（$V^T$）→ 缩放（$\Sigma$）→ 旋转（$U$） |
| Eckart-Young | 截取前 $k$ 项 = Frobenius 范数最优低秩近似 |
| 外积展开 | $A = \sum_i \sigma_i \mathbf{u}_i\mathbf{v}_i^T$ |
| 四大子空间 | $U, V$ 的列分别给出完整的正交基 |

**下一篇**：D5 SVD 数值计算与 numpy 实现——Golub-Reinsch 算法、数值稳定性，以及工程实战中的 SVD 用法。

---

*参考：Trefethen & Bau《数值线性代数》§4；Golub & Van Loan《Matrix Computations》*
