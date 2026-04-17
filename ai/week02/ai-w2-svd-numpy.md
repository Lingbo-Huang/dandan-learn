# D5：SVD 数值计算与 NumPy 实现

> **Week 2 · Day 5** | AI 基础线：线性代数核心概念

---

## 1. 数值计算 SVD 的挑战

直接按照 $A^TA$ 的特征值分解来求 SVD，在数值上有严重问题：

- $A^TA$ 的条件数是 $A$ 条件数的**平方**，导致精度损失加倍
- 当 $A$ 的奇异值相差悬殊时，较小的奇异值会被淹没在舍入误差中

现代 SVD 算法（如 Golub-Reinsch）**不通过 $A^TA$**，而是直接对 $A$ 进行双对角化，精度更高。

---

## 2. Golub-Reinsch 算法概述

### 2.1 主要步骤

**Phase 1：双对角化（Bidiagonalization）**

通过 Householder 变换将 $A$ 化为上双对角矩阵 $B$：

$$
U_1^T A V_1 = B = \begin{pmatrix} b_1 & c_1 & & \\ & b_2 & c_2 & \\ & & \ddots & \ddots \\ & & & b_n \end{pmatrix}
$$

**Phase 2：对双对角矩阵求 SVD**

对 $B$ 使用隐式 QR 迭代（类似对称矩阵的特征值求法），数值稳定地求出奇异值和奇异向量。

**算法复杂度：**
- 双对角化：$O(mn^2 - n^3/3)$
- QR 迭代：$O(n^2)$ 每次迭代

### 2.2 Householder 反射

Householder 变换是 SVD 数值计算的基石：

$$
H = I - 2\frac{\mathbf{v}\mathbf{v}^T}{\mathbf{v}^T\mathbf{v}}
$$

$H$ 是正交矩阵（$H^T = H, H^2 = I$），将向量 $\mathbf{x}$ 反射到某个轴方向：

$$
H\mathbf{x} = \alpha \mathbf{e}_1, \quad \alpha = -\text{sign}(x_1)\|\mathbf{x}\|
$$

---

## 3. NumPy SVD 的使用

### 3.1 基本用法

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]], dtype=float)

# full_matrices=True（默认）：完整 SVD
U_full, s, Vt_full = np.linalg.svd(A, full_matrices=True)
print(f"完整 SVD: U{U_full.shape}, s{s.shape}, Vt{Vt_full.shape}")

# full_matrices=False：薄型（紧凑）SVD，节省内存
U_thin, s_thin, Vt_thin = np.linalg.svd(A, full_matrices=False)
print(f"薄型 SVD: U{U_thin.shape}, s{s_thin.shape}, Vt{Vt_thin.shape}")

# 重构
Sigma_thin = np.diag(s_thin)
A_reconstructed = U_thin @ Sigma_thin @ Vt_thin
print(f"重构误差: {np.linalg.norm(A - A_reconstructed):.2e}")
```

### 3.2 `np.linalg.svd` vs `scipy.linalg.svd` vs `scipy.sparse.linalg.svds`

| 函数 | 适用场景 | 特点 |
|------|---------|------|
| `np.linalg.svd` | 稠密中小矩阵 | 基于 LAPACK `_gesdd`（分治法） |
| `scipy.linalg.svd` | 稠密矩阵，更多选项 | 可选 `_gesvd`（更稳定但稍慢） |
| `scipy.sparse.linalg.svds` | **大规模稀疏矩阵** | Lanczos/ARPACK，只求前 $k$ 个 |
| `sklearn.decomposition.TruncatedSVD` | 机器学习管道 | 封装 `svds`，配合 `sklearn` API |

---

## 4. 截断 SVD（Truncated SVD）

### 4.1 应用场景

当只需要前 $k$ 个最大奇异值时，计算完整 SVD 再截断是浪费的。使用截断 SVD：

$$
A \approx U_k \Sigma_k V_k^T
$$

### 4.2 随机化 SVD（Randomized SVD）

对于大矩阵，随机化 SVD 只需 $O(mnk)$ 而不是 $O(mn^2)$：

**算法思路：**
1. 用随机矩阵 $\Omega$（$n \times k$）投影：$Y = A\Omega$
2. 对 $Y$ 做 QR 分解：$Y = QR$
3. 对 $B = Q^TA$（小矩阵）做标准 SVD：$B = \hat{U}\Sigma V^T$
4. $U = Q\hat{U}$

```python
def randomized_svd(A, k, n_oversampling=10, n_power_iter=2):
    """
    随机化 SVD，求前 k 个奇异三元组
    
    Args:
        A: 输入矩阵
        k: 需要的奇异值个数
        n_oversampling: 过采样数（提升精度）
        n_power_iter: 幂次迭代次数（提升精度）
    """
    m, n = A.shape
    l = k + n_oversampling  # 过采样
    
    # Step 1: 随机投影
    Omega = np.random.randn(n, l)
    Y = A @ Omega
    
    # 幂次迭代（可选，提升精度）
    for _ in range(n_power_iter):
        Y = A @ (A.T @ Y)
    
    # Step 2: QR 分解
    Q, _ = np.linalg.qr(Y)
    
    # Step 3: 投影到小空间
    B = Q.T @ A  # (l × n)
    
    # Step 4: 对 B 做 SVD
    U_hat, s, Vt = np.linalg.svd(B, full_matrices=False)
    
    # Step 5: 恢复 U
    U = Q @ U_hat
    
    return U[:, :k], s[:k], Vt[:k, :]
```

---

## 5. 完整工程示例

### 5.1 环境准备

```bash
uv init svd-numpy-demo
cd svd-numpy-demo
uv add numpy scipy scikit-learn matplotlib pillow
```

### 5.2 图像压缩实战

```python
# image_compression_svd.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def compress_image_svd(image_array, k):
    """用 SVD 对灰度图像进行 k-秩近似压缩"""
    U, s, Vt = np.linalg.svd(image_array, full_matrices=False)
    # 截断到前 k 个奇异值
    compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    # 截断到 [0, 255]
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)
    return compressed, s

def compression_ratio(m, n, k):
    """计算压缩比（原始像素数 vs SVD 存储量）"""
    original = m * n
    compressed_storage = k * (m + n + 1)  # U列 + V列 + 奇异值
    return original / compressed_storage

# 生成测试图像（如无真实图像）
def make_test_image(size=256):
    """生成带结构的测试灰度图像"""
    img = np.zeros((size, size))
    x, y = np.meshgrid(np.linspace(0, 4*np.pi, size),
                        np.linspace(0, 4*np.pi, size))
    img = (np.sin(x) * np.cos(y) + 1) * 127.5
    return img.astype(np.uint8)

img = make_test_image(256)
print(f"测试图像大小: {img.shape}")

# 不同截断秩的压缩效果
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
ks = [1, 5, 10, 20, 50, 100, 150, 256]

_, s_all = compress_image_svd(img.astype(float), img.shape[0])
total_energy = np.sum(s_all**2)

for idx, k in enumerate(ks):
    ax = axes[idx // 4][idx % 4]
    if k >= min(img.shape):
        compressed = img
        ratio = 1.0
        energy_pct = 100.0
    else:
        compressed, _ = compress_image_svd(img.astype(float), k)
        ratio = compression_ratio(*img.shape, k)
        energy_pct = np.sum(s_all[:k]**2) / total_energy * 100
    
    ax.imshow(compressed, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'k={k}\n压缩比={ratio:.1f}x\n保留能量={energy_pct:.1f}%', fontsize=9)
    ax.axis('off')

plt.suptitle('SVD 图像压缩：不同截断秩的效果', fontsize=14)
plt.tight_layout()
plt.savefig('svd_image_compression.png', dpi=150, bbox_inches='tight')
print("图像压缩结果已保存")

# 奇异值衰减曲线
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes2[0]
ax1.plot(s_all, 'b-o', markersize=3)
ax1.set_xlabel('奇异值序号')
ax1.set_ylabel('奇异值大小')
ax1.set_title('奇异值衰减曲线')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

ax2 = axes2[1]
cumulative_energy = np.cumsum(s_all**2) / total_energy * 100
ax2.plot(cumulative_energy, 'r-')
ax2.axhline(y=90, color='gray', linestyle='--', label='90% 能量')
ax2.axhline(y=99, color='green', linestyle='--', label='99% 能量')
k90 = np.searchsorted(cumulative_energy, 90) + 1
k99 = np.searchsorted(cumulative_energy, 99) + 1
ax2.axvline(x=k90, color='gray', linestyle=':', alpha=0.7)
ax2.axvline(x=k99, color='green', linestyle=':', alpha=0.7)
ax2.set_xlabel('保留奇异值数 k')
ax2.set_ylabel('保留能量 (%)')
ax2.set_title(f'累积能量曲线\n90%能量需k={k90}, 99%能量需k={k99}')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svd_energy_curve.png', dpi=150, bbox_inches='tight')
print("能量曲线已保存")
```

### 5.3 随机化 SVD 精度对比

```python
# randomized_vs_exact.py
import numpy as np
import time

def randomized_svd(A, k, n_oversampling=10, n_power_iter=2):
    """（见上方实现）"""
    m, n = A.shape
    l = k + n_oversampling
    Omega = np.random.randn(n, l)
    Y = A @ Omega
    for _ in range(n_power_iter):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A
    U_hat, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat
    return U[:, :k], s[:k], Vt[:k, :]

# 生成低秩结构矩阵
np.random.seed(42)
m, n, true_rank = 500, 400, 20
A_left = np.random.randn(m, true_rank)
A_right = np.random.randn(true_rank, n)
A = A_left @ A_right + 0.01 * np.random.randn(m, n)

k = 20

# 精确 SVD
t0 = time.time()
U_exact, s_exact, Vt_exact = np.linalg.svd(A, full_matrices=False)
t_exact = time.time() - t0

# 随机化 SVD
t0 = time.time()
U_rand, s_rand, Vt_rand = randomized_svd(A, k)
t_rand = time.time() - t0

print(f"精确 SVD 前{k}个奇异值: {s_exact[:k].round(3)}")
print(f"随机化 SVD 前{k}个奇异值: {s_rand.round(3)}")
print(f"\n奇异值误差: {np.linalg.norm(s_exact[:k] - s_rand):.4f}")
print(f"\n精确 SVD 用时: {t_exact*1000:.1f} ms")
print(f"随机化 SVD 用时: {t_rand*1000:.1f} ms")
print(f"加速比: {t_exact/t_rand:.2f}x")

# 低秩近似误差
A_k_exact = U_exact[:, :k] @ np.diag(s_exact[:k]) @ Vt_exact[:k, :]
A_k_rand = U_rand @ np.diag(s_rand) @ Vt_rand
err_exact = np.linalg.norm(A - A_k_exact, 'fro')
err_rand = np.linalg.norm(A - A_k_rand, 'fro')
print(f"\n低秩近似误差（精确）: {err_exact:.4f}")
print(f"低秩近似误差（随机）: {err_rand:.4f}")
```

---

## 6. 数值稳定性注意事项

### 6.1 条件数与数值精度

矩阵的**条件数** $\kappa(A) = \sigma_1 / \sigma_r$（最大/最小奇异值之比）决定了线性方程组求解的数值稳定性：

- $\kappa \approx 1$：数值良态（well-conditioned）
- $\kappa \gg 1$：数值病态（ill-conditioned），结果可能不可靠

```python
import numpy as np
A = np.random.randn(5, 5)
cond_num = np.linalg.cond(A)  # 等价于 σ_max / σ_min
print(f"条件数: {cond_num:.2f}")
```

### 6.2 零奇异值的处理

数值上的零奇异值（噪声导致）需要阈值截断：

```python
tol = max(A.shape) * np.finfo(float).eps * s[0]  # 默认阈值
rank_numerical = np.sum(s > tol)
```

---

## 7. 小结

| 主题 | 要点 |
|------|------|
| Golub-Reinsch | 双对角化 + QR 迭代，不经过 $A^TA$，数值稳定 |
| `np.linalg.svd` | `full_matrices=False` 求薄型 SVD，基于 LAPACK |
| 截断 SVD | `scipy.sparse.linalg.svds` 高效求前 $k$ 个 |
| 随机化 SVD | $O(mnk)$ 时间，大规模数据的首选 |
| 图像压缩 | 截断秩 $k$ 控制压缩比与质量的权衡 |
| 条件数 | $\sigma_1/\sigma_r$，衡量数值稳定性 |

**下一篇**：D6 PCA 与 SVD 的关系及降维实战——PCA 本质上就是对中心化数据做 SVD，如何用几十行代码实现完整的降维流程？

---

*参考：Golub & Van Loan《Matrix Computations》第8章；Halko et al. 2011《Finding Structure with Randomness》*
