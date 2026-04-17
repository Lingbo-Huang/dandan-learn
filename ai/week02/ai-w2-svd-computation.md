# D5 · SVD 计算（Truncated SVD & Low-Rank Approximation）

> **Week 2 · Day 5**  
> 预计学习时间：90 分钟

---

## 学习目标

1. 掌握截断 SVD 的计算与意义
2. 理解 Eckart-Young 定理（最优低秩近似）
3. 能用截断 SVD 进行图像压缩
4. 理解奇异值衰减与信息压缩的关系
5. 掌握随机化 SVD（大规模矩阵的实用算法）

---

## 核心知识点

### 1. SVD 展开为秩-1 矩阵之和

$$
A = U\Sigma V^T = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

每一项 $\sigma_i \mathbf{u}_i \mathbf{v}_i^T$ 是一个**秩-1 矩阵**，贡献了矩阵 $A$ 的一部分"信息"。

$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ 保证了第一项贡献最大，后面越来越小。

---

### 2. 截断 SVD（Rank-k Approximation）

取前 $k$ 项：

$$
A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T = U_k \Sigma_k V_k^T
$$

- $U_k \in \mathbb{R}^{m\times k}$：$U$ 的前 $k$ 列
- $\Sigma_k \in \mathbb{R}^{k\times k}$：前 $k$ 个奇异值的对角矩阵
- $V_k \in \mathbb{R}^{n\times k}$：$V$ 的前 $k$ 列

**压缩比**：
- 原矩阵：$mn$ 个数
- 截断 SVD：$mk + k + nk = k(m+n+1)$ 个数
- 当 $k \ll \min(m,n)$ 时，压缩效果显著

---

### 3. Eckart-Young 定理（最优低秩近似）

这是 SVD 最重要的理论结果之一：

$$
\boxed{A_k = \arg\min_{\text{rank}(B) \leq k} \|A - B\|_F = \arg\min_{\text{rank}(B) \leq k} \|A - B\|_2}
$$

**在 Frobenius 范数和谱范数意义下，截断 SVD 给出最优的秩-$k$ 近似。**

近似误差：

$$
\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}
$$

$$
\|A - A_k\|_2 = \sigma_{k+1}
$$

**含义**：如果你只能用秩-$k$ 矩阵来近似 $A$，没有比截断 SVD 更好的方案了。

---

### 4. 奇异值衰减与信息含量

奇异值 $\sigma_i$ 的平方代表第 $i$ 个"主成分"的能量：

$$
\text{解释的方差比} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}
$$

**实际矩阵的奇异值通常快速衰减**：
- 图像矩阵：前几十个奇异值即可捕捉主要结构
- 文本矩阵（词-文档）：前 100-300 维即可
- 推荐系统矩阵：前 10-50 维已足够

---

### 5. 随机化 SVD（Randomized SVD）

对大规模矩阵（如 $10^4 \times 10^6$），精确 SVD 代价极高（$O(mn^2)$）。

**随机化 SVD 算法**（Halko et al., 2011）：

```
输入：A (m×n), 目标秩 k, 过采样 p
1. 生成随机矩阵 Ω ∈ R^{n×(k+p)}
2. Y = AΩ（捕捉 A 的列空间）
3. Q, _ = QR(Y)（正交基）
4. B = Q^T A（压缩矩阵，(k+p)×n）
5. Û, Σ, Vᵀ = SVD(B)（小矩阵精确SVD）
6. U = Q Û
输出：U, Σ, Vᵀ（近似截断SVD）
```

复杂度：$O(mn\log k)$，比精确 SVD 快得多。

---

### 6. 图像压缩的实际案例

灰度图像是一个矩阵 $A \in \mathbb{R}^{m\times n}$（值域 0-255）。

截断 SVD 给出压缩版本：

$$
\hat{A} = A_k = U_k\Sigma_k V_k^T
$$

压缩比 = $\frac{mn}{k(m+n+1)}$

- $k=10$：通常能看出大致轮廓
- $k=50$：大多数图像细节清晰
- $k=100$：接近原图，几乎看不出差别

---

## 示例推导

### 例：手动验证截断 SVD 误差

设 $A = \begin{pmatrix}3&2&2\\2&3&-2\end{pmatrix}$，$\sigma_1=5, \sigma_2=3, \sigma_3=0$

**秩-1 近似误差：**

$$
\|A - A_1\|_F = \sqrt{\sigma_2^2} = \sigma_2 = 3
$$

**秩-2 近似（等于原矩阵，因为 rank=2）：**

$$
\|A - A_2\|_F = \sqrt{\sigma_3^2} = 0
$$

**解释方差比：**

$$
\frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2} = \frac{25}{25+9} = \frac{25}{34} \approx 73.5\%
$$

第一个主成分解释了 73.5% 的"信息"。

---

## 动手练习

### 环境准备

```bash
uv init svd-computation-demo
cd svd-computation-demo
uv add numpy matplotlib pillow scikit-learn
```

### 练习 1：截断 SVD 基本操作与误差分析

```python
# main.py
import numpy as np
import matplotlib.pyplot as plt

def truncated_svd_analysis(A, k_values=None):
    """分析不同秩的 SVD 近似效果"""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = np.sum(s > 1e-10)
    
    print(f"矩阵形状: {A.shape}")
    print(f"秩 r = {r}")
    print(f"所有奇异值: {np.round(s, 4)}")
    
    if k_values is None:
        k_values = list(range(1, r+1))
    
    total_variance = np.sum(s**2)
    
    print(f"\n{'k':>4} | {'误差‖A-Aₖ‖F':>14} | {'解释方差比':>10} | {'理论σₖ₊₁':>10}")
    print("-" * 50)
    for k in k_values:
        # 截断 SVD
        Ak = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        error = np.linalg.norm(A - Ak, 'fro')
        explained = np.sum(s[:k]**2) / total_variance
        theoretical_error = np.sqrt(np.sum(s[k:]**2))
        
        print(f"{k:>4} | {error:>14.6f} | {explained:>10.4%} | {theoretical_error:>10.6f}")
    
    print(f"\n✓ 误差 = sqrt(Σσᵢ² for i>k) 验证 Eckart-Young 定理")

A = np.array([[3, 2, 2, 1],
              [2, 3, -2, 0],
              [1, 0, 1, 2]], dtype=float)
truncated_svd_analysis(A)
```

---

### 练习 2：图像压缩实战

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import io

def image_svd_compression(image_array, k_values=[5, 20, 50, 100]):
    """
    对灰度图像进行 SVD 压缩，展示不同秩的效果
    """
    A = image_array.astype(float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    m, n = A.shape
    n_plots = len(k_values) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5))
    
    # 原图
    axes[0].imshow(A, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f"原图\n{m}×{n} = {m*n:,} 个数")
    axes[0].axis('off')
    
    for ax, k in zip(axes[1:], k_values):
        # 截断 SVD 重建
        Ak = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        Ak_clipped = np.clip(Ak, 0, 255)
        
        # 压缩存储量
        storage = k * (m + 1 + n)
        compression_ratio = m * n / storage
        
        # 相对误差
        rel_error = np.linalg.norm(A - Ak, 'fro') / np.linalg.norm(A, 'fro')
        
        # 解释方差
        explained = np.sum(s[:k]**2) / np.sum(s**2)
        
        ax.imshow(Ak_clipped, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"k={k}\n压缩比 {compression_ratio:.1f}x\n误差 {rel_error:.3f}")
        ax.axis('off')
    
    plt.suptitle("SVD 图像压缩：不同秩的效果", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("image_svd_compression.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # 奇异值衰减曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.semilogy(s[:100], 'b.-', markersize=4)
    plt.xlabel('奇异值序号')
    plt.ylabel('奇异值（对数尺度）')
    plt.title('奇异值衰减曲线')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    cumulative = np.cumsum(s**2) / np.sum(s**2)
    plt.plot(cumulative[:100], 'r.-', markersize=4)
    plt.axhline(0.9, color='gray', ls='--', label='90% 方差')
    plt.axhline(0.99, color='gray', ls=':', label='99% 方差')
    k90 = np.searchsorted(cumulative, 0.9) + 1
    k99 = np.searchsorted(cumulative, 0.99) + 1
    plt.xlabel('k（使用的奇异值个数）')
    plt.ylabel('解释方差累积比')
    plt.title(f'累积解释方差（90%需k={k90}, 99%需k={k99}）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("singular_values_decay.png", dpi=150, bbox_inches='tight')
    plt.show()

# 生成测试图像（如无法下载，用合成图）
np.random.seed(42)
# 合成一个有结构的测试矩阵（模拟图像）
x = np.linspace(0, 10, 200)
y = np.linspace(0, 10, 200)
X, Y = np.meshgrid(x, y)
image = (128 + 60 * np.sin(X) * np.cos(Y) + 
         40 * np.sin(2*X + 1) + 30 * np.cos(3*Y)).astype(np.uint8)

image_svd_compression(image, k_values=[2, 10, 30, 80])
```

---

### 练习 3：随机化 SVD 实现

```python
import numpy as np
import time

def randomized_svd(A, k, n_oversampling=10, n_power_iter=2):
    """
    随机化 SVD：适用于大型矩阵的高效近似算法
    
    参数:
        A: 输入矩阵
        k: 目标秩
        n_oversampling: 过采样参数（默认10）
        n_power_iter: 幂迭代次数（提高精度）
    """
    m, n = A.shape
    l = k + n_oversampling  # 采样维度
    
    # Step 1: 随机投影
    Omega = np.random.randn(n, l)
    Y = A @ Omega
    
    # Step 2: 幂迭代（提高精度，特别对奇异值衰减慢的矩阵）
    for _ in range(n_power_iter):
        Y = A @ (A.T @ Y)
    
    # Step 3: 正交化
    Q, _ = np.linalg.qr(Y)
    
    # Step 4: 压缩后精确 SVD
    B = Q.T @ A  # (l × n)
    U_hat, s, Vt = np.linalg.svd(B, full_matrices=False)
    
    # Step 5: 恢复左奇异向量
    U = Q @ U_hat
    
    return U[:, :k], s[:k], Vt[:k, :]

# 对比精确 SVD 和随机化 SVD
np.random.seed(42)
m, n, k = 500, 400, 20

# 构造低秩矩阵（真实秩约为 k）
U_true = np.linalg.qr(np.random.randn(m, k))[0]
V_true = np.linalg.qr(np.random.randn(n, k))[0]
s_true = np.exp(-np.arange(k) * 0.5)  # 指数衰减奇异值
A = U_true @ np.diag(s_true) @ V_true.T + 0.01 * np.random.randn(m, n)

# 精确 SVD
t0 = time.time()
U_exact, s_exact, Vt_exact = np.linalg.svd(A, full_matrices=False)
t_exact = time.time() - t0

# 随机化 SVD
t0 = time.time()
U_rand, s_rand, Vt_rand = randomized_svd(A, k)
t_rand = time.time() - t0

print(f"精确 SVD 时间: {t_exact:.4f}s")
print(f"随机化 SVD 时间: {t_rand:.4f}s")
print(f"速度提升: {t_exact/t_rand:.1f}x")

# 精度对比
print(f"\n前{k}个奇异值对比:")
print(f"精确:  {np.round(s_exact[:k], 4)}")
print(f"随机:  {np.round(s_rand, 4)}")
print(f"相对误差: {np.max(np.abs(s_exact[:k] - s_rand) / s_exact[:k]):.6f}")

# 重建误差对比
Ak_exact = U_exact[:, :k] @ np.diag(s_exact[:k]) @ Vt_exact[:k, :]
Ak_rand = U_rand @ np.diag(s_rand) @ Vt_rand

print(f"\n重建误差（精确）: {np.linalg.norm(A - Ak_exact, 'fro'):.6f}")
print(f"重建误差（随机）: {np.linalg.norm(A - Ak_rand, 'fro'):.6f}")
```

---

### 练习 4：推荐系统矩阵分解

```python
import numpy as np
import matplotlib.pyplot as plt

def collaborative_filtering_svd(R_full, k=3):
    """
    用 SVD 实现协同过滤
    
    R_full: 完整的用户-电影评分矩阵（仅用于评估）
    实践中只有部分评分已知
    """
    # 模拟稀疏观测（随机遮盖 60% 的评分）
    np.random.seed(42)
    mask = np.random.rand(*R_full.shape) > 0.6
    R_observed = np.where(mask, R_full, 0.0)
    
    print("用户-电影评分矩阵（0 = 未评分）:")
    print(R_observed)
    print(f"\n已观测评分数: {mask.sum()}/{R_full.size}")
    
    # 对观测到的评分矩阵做 SVD（均值填充策略）
    R_filled = R_observed.copy()
    col_means = np.where(mask.sum(0) > 0, R_observed.sum(0) / mask.sum(0), 3.0)
    for j in range(R_full.shape[1]):
        R_filled[~mask[:, j], j] = col_means[j]
    
    # 截断 SVD
    U, s, Vt = np.linalg.svd(R_filled, full_matrices=False)
    R_pred = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    R_pred = np.clip(R_pred, 1, 5)  # 评分范围 1-5
    
    print(f"\n截断 SVD 预测矩阵（秩-{k}）:")
    print(np.round(R_pred, 2))
    
    print(f"\n真实评分矩阵:")
    print(R_full)
    
    # 评估（在未观测的位置）
    test_mask = ~mask
    rmse = np.sqrt(np.mean((R_pred[test_mask] - R_full[test_mask])**2))
    print(f"\n在未观测位置的 RMSE: {rmse:.4f}")
    
    return R_pred

# 构造一个 5 用户 × 6 电影的评分矩阵
# 存在两类"口味"（前3用户喜欢动作片，后2喜欢爱情片）
R_full = np.array([
    [5, 4, 5, 1, 1, 2],  # 用户1：动作片爱好者
    [4, 5, 4, 1, 2, 1],  # 用户2
    [5, 5, 5, 2, 1, 1],  # 用户3
    [1, 2, 1, 5, 4, 5],  # 用户4：爱情片爱好者
    [2, 1, 2, 4, 5, 5],  # 用户5
], dtype=float)

R_pred = collaborative_filtering_svd(R_full, k=2)
```

---

## 小结

| 操作 | 公式 | 适用场景 |
|------|------|---------|
| 完整 SVD | $A = U\Sigma V^T$ | 理论分析、小矩阵 |
| 截断 SVD | $A_k = U_k\Sigma_k V_k^T$ | 压缩、去噪 |
| 随机化 SVD | 幂迭代 + QR + 小 SVD | 大规模矩阵 |

**Eckart-Young 核心结论：截断 SVD 是最优低秩近似**

压缩率选择经验法则：
- 选最小的 $k$，使得 $\sum_{i=1}^k \sigma_i^2 / \sum \sigma_i^2 \geq 90\%$

---

## ML 中的应用

### 1. LSA（潜在语义分析）

词-文档矩阵（TF-IDF 加权）→ 截断 SVD → 语义空间。

### 2. 神经网络权重压缩

大权重矩阵 $W$ → SVD → 截断为 $W_k = U_k\Sigma_k V_k^T$（参数量减少 $\frac{mn}{k(m+n)}$ 倍）。

### 3. 数据去噪

信号矩阵 + 高斯噪声 $A = S + E$，截断 SVD 去掉小奇异值（对应噪声），保留主要结构。

### 4. sklearn 的 TruncatedSVD

```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50)
X_reduced = svd.fit_transform(X)  # 稀疏矩阵友好
```

---

*→ 明天 D6：PCA from SVD，从降维到数据理解*
