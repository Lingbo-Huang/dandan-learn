# D6：PCA 与 SVD 的关系及降维实战

> **Week 2 · Day 6** | AI 基础线：线性代数核心概念

---

## 1. PCA 是什么？

**主成分分析**（Principal Component Analysis, PCA）是最经典的无监督降维方法，目标是：

> 在最小化信息损失的前提下，找到数据方差最大的正交方向，将高维数据投影到低维空间。

PCA 的核心思想：**用少数几个"主成分"捕捉数据的主要变化方向**。

---

## 2. PCA 的数学推导

### 2.1 问题形式化

设数据矩阵 $X \in \mathbb{R}^{n \times d}$（$n$ 个样本，$d$ 个特征），目标是找单位向量 $\mathbf{w} \in \mathbb{R}^d$，使投影方差最大：

$$
\max_{\|\mathbf{w}\|=1} \text{Var}(X\mathbf{w}) = \max_{\|\mathbf{w}\|=1} \mathbf{w}^T \text{Cov}(X) \mathbf{w}
$$

其中**样本协方差矩阵**（数据已中心化，即 $\bar{X} = 0$）：

$$
\text{Cov}(X) = \frac{1}{n-1} X^TX
$$

### 2.2 Lagrange 乘数法

约束优化问题：

$$
\max_{\mathbf{w}} \mathbf{w}^T C \mathbf{w}, \quad \text{s.t.} \; \mathbf{w}^T\mathbf{w} = 1
$$

引入 Lagrange 乘子 $\lambda$，对 $\mathbf{w}$ 求导令其为零：

$$
\frac{\partial}{\partial \mathbf{w}}\left[\mathbf{w}^TC\mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)\right] = 2C\mathbf{w} - 2\lambda\mathbf{w} = \mathbf{0}
$$

$$
C\mathbf{w} = \lambda\mathbf{w}
$$

**结论：第一主成分方向 = 协方差矩阵的最大特征向量！**

目标函数的最大值：

$$
\mathbf{w}^TC\mathbf{w} = \mathbf{w}^T(\lambda\mathbf{w}) = \lambda\mathbf{w}^T\mathbf{w} = \lambda
$$

所以对应**最大特征值** $\lambda_1$ 的特征向量 $\mathbf{w}_1$ 就是第一主成分方向，第二主成分是在与 $\mathbf{w}_1$ 正交的约束下方差最大的方向（即第二大特征值对应的特征向量），依此类推。

### 2.3 所有主成分

将协方差矩阵 $C$ 做特征值分解（因为 $C$ 对称，可正交对角化）：

$$
C = Q\Lambda Q^T, \quad \Lambda = \text{diag}(\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d)
$$

**$k$ 个主成分的降维结果：**

$$
Z = X Q_k
$$

其中 $Q_k$ 是 $Q$ 的前 $k$ 列，$Z \in \mathbb{R}^{n \times k}$。

---

## 3. PCA 与 SVD 的等价关系

### 3.1 核心联系

对中心化数据矩阵 $\tilde{X}$（已减去列均值）做 SVD：

$$
\tilde{X} = U\Sigma V^T
$$

则：

$$
\tilde{X}^T\tilde{X} = V\Sigma^T U^T U \Sigma V^T = V(\Sigma^T\Sigma)V^T
$$

这正是协方差矩阵的特征值分解（差一个 $n-1$ 的系数）：

$$
C = \frac{1}{n-1}\tilde{X}^T\tilde{X} = V \cdot \frac{\Sigma^2}{n-1} \cdot V^T
$$

**因此：**

| PCA 元素 | SVD 元素 |
|---------|---------|
| 主成分方向（协方差矩阵特征向量）| $\tilde{X}$ 的右奇异向量 $V$ 的列 |
| 主成分方差（特征值）| $\sigma_i^2 / (n-1)$ |
| 投影坐标 | $Z = U\Sigma$（或 $\tilde{X}V_k$）|

### 3.2 为什么用 SVD 而不是直接求协方差矩阵的特征值？

1. **数值稳定性**：SVD 比先计算 $\tilde{X}^T\tilde{X}$ 再求特征值更稳定（条件数不平方）
2. **适用于 $n < d$ 的情形**：高维数据（如文本 TF-IDF）中 $d \gg n$，协方差矩阵 $d \times d$ 甚至无法存入内存
3. **统一框架**：scikit-learn 的 `PCA` 就是调用 `scipy.linalg.svd`

---

## 4. PCA 的完整算法

```
输入：数据矩阵 X ∈ R^{n×d}，目标维度 k
输出：降维后数据 Z ∈ R^{n×k}

1. 中心化：μ = mean(X, axis=0)；X̃ = X - μ
2. SVD：X̃ = U Σ V^T
3. 选取前 k 列：Vk = V[:, :k]（主成分方向）
4. 投影：Z = X̃ @ Vk
5. （可选）反投影重建：X_recon = Z @ Vk^T + μ
```

**方差解释比例：**

$$
\text{explained variance ratio}_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}
$$

---

## 5. 选择主成分个数 k

### 5.1 方差解释率（Variance Explained Ratio）

选取最小的 $k$ 使得累积方差解释率 $\geq$ 阈值（通常 95% 或 99%）：

$$
\sum_{i=1}^k \frac{\sigma_i^2}{\sum_j \sigma_j^2} \geq 0.95
$$

### 5.2 碎石图（Scree Plot）

画出奇异值（或特征值）序列，找"肘点"（Elbow）：奇异值急剧下降后变平的转折点。

### 5.3 Kaiser 准则

保留方差大于均值（$\lambda_i > \bar{\lambda}$）的主成分。

---

## 6. Python 完整降维实战

### 6.1 环境准备

```bash
uv init pca-demo
cd pca-demo
uv add numpy scikit-learn matplotlib seaborn
```

### 6.2 从零实现 PCA（不用 sklearn）

```python
# pca_from_scratch.py
import numpy as np
import matplotlib.pyplot as plt

class PCA:
    """从零实现 PCA（基于 SVD）"""
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None      # 主成分方向 (k × d)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
    
    def fit(self, X):
        n, d = X.shape
        # Step 1: 中心化
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        # Step 2: SVD（薄型）
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Step 3: 主成分方向（右奇异向量，即 V 的行）
        self.components_ = Vt[:self.n_components]  # (k × d)
        
        # Step 4: 方差（σ² / (n-1)）
        self.explained_variance_ = s[:self.n_components]**2 / (n - 1)
        total_var = s**2 / (n - 1)
        self.explained_variance_ratio_ = (self.explained_variance_ / 
                                           total_var.sum())
        return self
    
    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T  # (n × k)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, Z):
        """从低维空间重建高维数据（有信息损失）"""
        return Z @ self.components_ + self.mean_


# ====== 实战：鸢尾花数据集降维 ======
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target  # (150, 4)

# 标准化（PCA 对尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 我们的 PCA
pca = PCA(n_components=2)
Z = pca.fit_transform(X_scaled)

print(f"原始维度: {X_scaled.shape}")
print(f"降维后:   {Z.shape}")
print(f"各主成分方差解释率: {pca.explained_variance_ratio_.round(4)}")
print(f"累积解释率: {pca.explained_variance_ratio_.cumsum().round(4)}")

# 可视化降维结果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#E74C3C', '#27AE60', '#3498DB']
labels = iris.target_names

ax1 = axes[0]
for i, (color, label) in enumerate(zip(colors, labels)):
    mask = y == i
    ax1.scatter(Z[mask, 0], Z[mask, 1], c=color, label=label, 
                alpha=0.8, s=50, edgecolors='white', linewidths=0.5)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax1.set_title('PCA 降维结果（鸢尾花数据集）')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 主成分的因子载荷（loading）热图
ax2 = axes[1]
loadings = pca.components_  # (2, 4)
im = ax2.imshow(loadings, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax2.set_xticks(range(4))
ax2.set_xticklabels(iris.feature_names, rotation=30, ha='right', fontsize=10)
ax2.set_yticks(range(2))
ax2.set_yticklabels([f'PC{i+1}' for i in range(2)])
plt.colorbar(im, ax=ax2)
ax2.set_title('主成分载荷矩阵')
for i in range(2):
    for j in range(4):
        ax2.text(j, i, f'{loadings[i,j]:.2f}', ha='center', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('pca_iris.png', dpi=150, bbox_inches='tight')
print("\n降维可视化已保存")
```

### 6.3 碎石图与主成分选择

```python
# scree_plot.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# 手写数字数据集（64 维）
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

# 完整 SVD（不降维）
_, s, _ = np.linalg.svd(X, full_matrices=False)

var_ratio = s**2 / s**2.sum()
cumvar = np.cumsum(var_ratio)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax1 = axes[0]
ax1.bar(range(1, 21), var_ratio[:20] * 100, alpha=0.7)
ax1.plot(range(1, 21), var_ratio[:20] * 100, 'ro-', markersize=5)
ax1.set_xlabel('主成分序号')
ax1.set_ylabel('方差解释率 (%)')
ax1.set_title('碎石图（Scree Plot）')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(range(1, len(s)+1), cumvar * 100, 'b-')
for thresh in [0.80, 0.90, 0.95, 0.99]:
    k = np.searchsorted(cumvar, thresh) + 1
    ax2.axhline(y=thresh*100, color='gray', linestyle='--', alpha=0.6)
    ax2.axvline(x=k, color='gray', linestyle=':', alpha=0.6)
    ax2.text(k+0.5, thresh*100+0.5, f'k={k} ({thresh*100:.0f}%)', fontsize=9)
ax2.set_xlabel('保留主成分数 k')
ax2.set_ylabel('累积方差解释率 (%)')
ax2.set_title(f'手写数字（{X.shape[1]}维）累积方差曲线')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scree_plot_digits.png', dpi=150, bbox_inches='tight')
print("碎石图已保存")
```

---

## 7. PCA 的局限性与注意事项

| 局限 | 说明 |
|------|------|
| 线性假设 | 只捕捉线性相关，非线性结构需 Kernel PCA / t-SNE / UMAP |
| 尺度敏感 | 特征量纲不同时必须先标准化 |
| 方差 ≠ 重要性 | 方差大不等于对分类任务有用（LDA 更适合有监督降维）|
| 可解释性 | 主成分是原始特征的线性组合，物理含义模糊 |

---

## 8. 小结

| 主题 | 要点 |
|------|------|
| PCA 本质 | 找协方差矩阵的特征向量（方差最大方向） |
| 与 SVD 关系 | 对中心化 $\tilde{X}$ 做 SVD，$V$ 列 = 主成分方向 |
| 为何用 SVD | 比先求协方差矩阵更数值稳定 |
| k 的选择 | 碎石图、累积方差解释率（通常 ≥ 95%）|
| 实现 | `sklearn.decomposition.PCA`，底层调用 LAPACK SVD |

**下一篇**：D7 综合实战——用 SVD/PCA 实现人脸识别（Eigenfaces）和图像压缩，把 Week2 所有知识串联起来！

---

*参考：Jolliffe《Principal Component Analysis》；Bishop《PRML》§12.1*
