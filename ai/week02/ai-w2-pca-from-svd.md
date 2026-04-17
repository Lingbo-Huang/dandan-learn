# D6 · PCA from SVD（主成分分析的 SVD 推导）

> **Week 2 · Day 6**  
> 预计学习时间：90 分钟

---

## 学习目标

1. 理解 PCA 的统计动机：最大化方差投影
2. 掌握协方差矩阵的特征分解方法（协方差 PCA）
3. 掌握直接对数据矩阵做 SVD 的方法（SVD PCA）
4. 理解两种方法的等价性及区别
5. 能做完整的 PCA 分析，包括可视化和解释

---

## 核心知识点

### 1. PCA 的动机：降维与信息保留

**问题**：$n$ 个 $d$ 维数据点，$d$ 很大（如 784 维图像），能否用更少的维度表示？

**目标**：找一个线性投影 $W \in \mathbb{R}^{d \times k}$（$k \ll d$），使得投影后：
- 信息损失最小（重建误差最小）
- 各主成分之间不相关（正交）

**最优解**：就是协方差矩阵最大特征值对应的特征向量！

---

### 2. 数学推导：最大化投影方差

**Step 1：中心化数据**

$$
\bar{\mathbf{x}} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i, \quad \tilde{X} = X - \mathbf{1}\bar{\mathbf{x}}^T
$$

（$X \in \mathbb{R}^{n \times d}$，每行是一个样本）

**Step 2：协方差矩阵**

$$
C = \frac{1}{n-1}\tilde{X}^T\tilde{X} \in \mathbb{R}^{d \times d}
$$

$C$ 是实对称正半定矩阵，$C_{ij}$ 表示特征 $i$ 和 $j$ 的协方差。

**Step 3：最大方差方向**

投影到方向 $\mathbf{w}$（$\|\mathbf{w}\|=1$）后的方差：

$$
\text{Var} = \mathbf{w}^T C \mathbf{w}
$$

带约束最大化（Lagrange 乘子法）：

$$
\max_{\mathbf{w}} \mathbf{w}^T C \mathbf{w} \quad \text{s.t.} \quad \mathbf{w}^T\mathbf{w} = 1
$$

$$
\frac{\partial}{\partial \mathbf{w}}[\mathbf{w}^T C \mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)] = 2C\mathbf{w} - 2\lambda\mathbf{w} = 0
$$

$$
C\mathbf{w} = \lambda\mathbf{w}
$$

**结论**：最大方差方向就是 $C$ 的最大特征值对应的特征向量！

依次类推，$k$ 个主成分 = $C$ 的前 $k$ 个特征向量（按特征值降序）。

---

### 3. SVD 方法推导 PCA

**关键关系**：对中心化数据矩阵 $\tilde{X}$ 做 SVD：

$$
\tilde{X} = U\Sigma V^T
$$

则：

$$
C = \frac{1}{n-1}\tilde{X}^T\tilde{X} = \frac{1}{n-1}V\Sigma^2 V^T
$$

对比协方差矩阵的谱分解 $C = Q\Lambda Q^T$：

$$
Q = V, \quad \Lambda = \frac{\Sigma^2}{n-1}
$$

**结论**：

| SVD 量 | PCA 意义 |
|--------|---------|
| $V$ 的列（右奇异向量） | 主成分方向（主轴） |
| $\sigma_i^2 / (n-1)$ | 第 $i$ 个主成分的方差 |
| $U\Sigma$（左奇异向量 × 奇异值） | 主成分得分（投影坐标） |

---

### 4. 两种 PCA 实现的对比

| 方法 | 步骤 | 优势 | 劣势 |
|------|------|------|------|
| **协方差矩阵特征分解** | 计算 $C = \tilde{X}^T\tilde{X}/(n-1)$，再 `eigh(C)` | 直观 | $d$ 很大时矩阵 $C$ 很大 |
| **直接 SVD** | 对 $\tilde{X}$ 做 SVD | 数值更稳定，处理 $n < d$ 的情况 | 略微难以理解 |

sklearn 的 `PCA` 默认使用 SVD（`svd_solver='full'` 或 `'randomized'`）。

---

### 5. 主成分的解释

**方差解释率（EVR）**：

$$
\text{EVR}_i = \frac{\lambda_i}{\sum_j \lambda_j} = \frac{\sigma_i^2}{\sum_j \sigma_j^2}
$$

**累积方差解释率**：选最小的 $k$ 使得 $\sum_{i=1}^k \text{EVR}_i \geq 95\%$

**主成分得分（scores）**：

$$
Z = \tilde{X}V_k = U_k\Sigma_k \in \mathbb{R}^{n \times k}
$$

**载荷（loadings）**：$V_k \in \mathbb{R}^{d \times k}$（主成分方向）

---

### 6. PCA 的几何意义

PCA 找到数据椭球的主轴方向：

```
原始坐标系      →    主成分坐标系
    *    *             |
  *   ●   *    →     *|*
    *    *             |
                   最大方差方向水平排列
```

旋转 + 选取最重要的 $k$ 个轴 = 降维

---

## 示例推导

### 例：手动推导 2D 数据的 PCA

数据（5 个点）：

$$
X = \begin{pmatrix}2.5 & 2.4\\0.5&0.7\\2.2&2.9\\1.9&2.2\\3.1&3.0\end{pmatrix}
$$

**Step 1**：均值 $\bar{\mathbf{x}} = (2.04, 2.24)$

**Step 2**：中心化

$$
\tilde{X} = X - \bar{\mathbf{x}} = \begin{pmatrix}0.46&0.16\\-1.54&-1.54\\\cdots\end{pmatrix}
$$

**Step 3**：协方差矩阵

$$
C = \frac{\tilde{X}^T\tilde{X}}{4} \approx \begin{pmatrix}0.616&0.615\\0.615&0.716\end{pmatrix}
$$

**Step 4**：特征值 $\lambda_1 \approx 1.28, \lambda_2 \approx 0.049$

$\text{EVR}_1 = 1.28/(1.28+0.049) \approx 96.3\%$

结论：第一主成分几乎捕捉了所有方差，数据本质上是 1D 的。

---

## 动手练习

### 环境准备

```bash
uv init pca-from-svd-demo
cd pca-from-svd-demo
uv add numpy matplotlib scikit-learn
```

### 练习 1：从零实现 PCA，与 sklearn 对比

```python
# main.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

class MyPCA:
    """手写 PCA，使用 SVD"""
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
    
    def fit(self, X):
        n, d = X.shape
        
        # 中心化
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        # SVD
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 主成分方向（右奇异向量）
        self.components_ = Vt[:self.n_components]  # (k × d)
        
        # 方差（奇异值² / (n-1)）
        self.explained_variance_ = (s ** 2) / (n - 1)
        total_var = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_[:self.n_components] / total_var
        
        return self
    
    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T  # (n × k)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# 测试：Iris 数据集
iris = load_iris()
X = iris.data
y = iris.target

# 手写 PCA
my_pca = MyPCA(n_components=2)
X_my = my_pca.fit_transform(X)

# sklearn PCA
sk_pca = PCA(n_components=2)
X_sk = sk_pca.fit_transform(X)

print("手写 PCA 方差解释率:", np.round(my_pca.explained_variance_ratio_, 4))
print("sklearn PCA 方差解释率:", np.round(sk_pca.explained_variance_ratio_, 4))
print("结果一致（允许符号差异）:", 
      np.allclose(np.abs(X_my), np.abs(X_sk), atol=1e-6))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#E74C3C', '#2ECC71', '#3498DB']
names = iris.target_names

for ax, X_2d, title in zip(axes, [X_my, X_sk], ["手写 PCA", "sklearn PCA"]):
    for i, (name, color) in enumerate(zip(names, colors)):
        mask = y == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                  c=color, label=name, alpha=0.7, s=50)
    ax.set_xlabel(f'PC1 ({my_pca.explained_variance_ratio_[0]:.1%} 方差)')
    ax.set_ylabel(f'PC2 ({my_pca.explained_variance_ratio_[1]:.1%} 方差)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Iris 数据集 PCA 降维（4D → 2D）", fontsize=13)
plt.tight_layout()
plt.savefig("pca_iris.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

### 练习 2：碎石图与方差解释

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

def pca_scree_plot(X, title="碎石图"):
    """
    碎石图（Scree Plot）帮助选择最优 k
    """
    n, d = X.shape
    X_centered = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    
    explained_var = s**2 / (n - 1)
    explained_ratio = explained_var / explained_var.sum()
    cumulative = np.cumsum(explained_ratio)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 碎石图
    ax = axes[0]
    ax.plot(range(1, len(s)+1), s[:50], 'b.-', markersize=4)
    ax.set_xlabel('主成分序号')
    ax.set_ylabel('奇异值')
    ax.set_title('奇异值（碎石图）')
    ax.set_xlim(0, 51)
    ax.grid(True, alpha=0.3)
    
    # 方差解释率
    ax = axes[1]
    ax.bar(range(1, 21), explained_ratio[:20], color='steelblue', alpha=0.7)
    ax.set_xlabel('主成分序号')
    ax.set_ylabel('方差解释率')
    ax.set_title('各主成分方差解释率（前20）')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 累积方差
    ax = axes[2]
    ax.plot(range(1, len(s)+1), cumulative, 'r-', lw=2)
    for thresh in [0.8, 0.9, 0.95, 0.99]:
        k = np.searchsorted(cumulative, thresh) + 1
        ax.axhline(thresh, color='gray', ls='--', alpha=0.5)
        ax.annotate(f'{thresh:.0%}: k={k}', 
                   xy=(k, thresh), xytext=(k+2, thresh-0.03),
                   fontsize=8, color='gray')
    ax.set_xlabel('k（主成分数）')
    ax.set_ylabel('累积方差解释率')
    ax.set_title('累积方差解释率')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig("pca_scree.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return explained_ratio, cumulative

# 手写数字数据集（64 维）
digits = load_digits()
X = digits.data
print(f"数据维度: {X.shape}")
evr, cumulative = pca_scree_plot(X, "手写数字数据集（64维）的 PCA 分析")
```

---

### 练习 3：PCA 与协方差矩阵的等价性证明

```python
import numpy as np

def compare_pca_methods(X, k=2):
    """
    对比两种 PCA 方法：
    1. 协方差矩阵特征分解
    2. 直接 SVD
    并验证等价性
    """
    n, d = X.shape
    X_centered = X - X.mean(axis=0)
    
    print(f"数据形状: {X.shape}")
    print("="*60)
    
    # 方法1：协方差矩阵特征分解
    C = X_centered.T @ X_centered / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # eigh 返回升序，需翻转
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    V_cov = eigenvectors[:, :k]  # 主成分方向
    Z_cov = X_centered @ V_cov   # 主成分得分
    
    print(f"\n方法1（协方差特征分解）前{k}个特征值: {np.round(eigenvalues[:k], 4)}")
    
    # 方法2：SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V_svd = Vt[:k].T  # (d × k)
    Z_svd = X_centered @ V_svd
    
    variance_from_svd = s[:k]**2 / (n - 1)
    print(f"方法2（SVD）前{k}个方差: {np.round(variance_from_svd, 4)}")
    
    print(f"\n方差一致（允许数值误差）: {np.allclose(eigenvalues[:k], variance_from_svd)}")
    
    # 方向一致性（允许符号差异）
    for i in range(k):
        cosine = abs(V_cov[:, i] @ V_svd[:, i])
        print(f"PC{i+1} 方向一致性（|cos θ|）: {cosine:.6f}")
    
    # 得分一致性
    print(f"\n投影得分一致（|Z_cov| ≈ |Z_svd|）: {np.allclose(np.abs(Z_cov), np.abs(Z_svd), atol=1e-6)}")
    
    return V_svd, Z_svd

# 用 Iris 测试
from sklearn.datasets import load_iris
iris = load_iris()
compare_pca_methods(iris.data, k=2)
```

---

### 练习 4：PCA 载荷（Loadings）解释

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def pca_biplot(X, feature_names, target, target_names, k=2):
    """
    PCA 双标图（Biplot）：同时展示样本得分和特征载荷
    """
    n, d = X.shape
    X_centered = X - X.mean(axis=0)
    X_std = X_centered / X.std(axis=0)  # 标准化（不同量纲时必须）
    
    U, s, Vt = np.linalg.svd(X_std, full_matrices=False)
    
    scores = U[:, :k] * s[:k]  # 主成分得分
    loadings = Vt[:k].T        # 载荷（d × k）
    
    # 方差解释率
    evr = s**2 / np.sum(s**2)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    colors = ['#E74C3C', '#2ECC71', '#3498DB']
    for i, (name, color) in enumerate(zip(target_names, colors)):
        mask = target == i
        ax.scatter(scores[mask, 0], scores[mask, 1], 
                  c=color, label=name, alpha=0.6, s=50)
    
    # 画载荷箭头
    scale = np.max(np.abs(scores)) / np.max(np.abs(loadings)) * 0.8
    for i, name in enumerate(feature_names):
        ax.annotate('', 
                   xy=(loadings[i, 0]*scale, loadings[i, 1]*scale),
                   xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
        ax.text(loadings[i, 0]*scale*1.1, loadings[i, 1]*scale*1.1,
               name, color='darkred', fontsize=9, ha='center')
    
    ax.set_xlabel(f'PC1 ({evr[0]:.1%} 方差)')
    ax.set_ylabel(f'PC2 ({evr[1]:.1%} 方差)')
    ax.set_title('PCA 双标图（Biplot）\n箭头=特征载荷（方向=主成分贡献方向）')
    ax.legend(loc='upper right')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("pca_biplot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n载荷矩阵（各特征在各主成分上的投影）:")
    print(f"{'特征名':<20} {'PC1':>8} {'PC2':>8}")
    for name, l in zip(feature_names, loadings):
        print(f"{name:<20} {l[0]:>8.4f} {l[1]:>8.4f}")

iris = load_iris()
pca_biplot(iris.data, iris.feature_names, iris.target, iris.target_names)
```

---

## 小结

| 步骤 | 操作 | 输出 |
|------|------|------|
| 中心化 | $\tilde{X} = X - \bar{X}$ | 零均值数据 |
| SVD | $\tilde{X} = U\Sigma V^T$ | 奇异向量和奇异值 |
| 主成分方向 | $V$ 的前 $k$ 列 | 载荷矩阵 |
| 主成分得分 | $\tilde{X}V_k = U_k\Sigma_k$ | 降维坐标 |
| 方差解释率 | $\sigma_i^2/\sum\sigma_j^2$ | 选 $k$ 的依据 |

**PCA 的两个等价条件：**
1. 最大化投影方差 → 特征向量
2. 最小化重建误差 → Eckart-Young 定理

---

## ML 中的应用

### 1. 特征降维预处理

高维特征（图像/文本）→ PCA 降维 → 简单分类器（SVM, KNN）

### 2. 可视化高维数据

$d$ 维 → PCA → 2D/3D 散点图（理解数据分布，发现异常点）

### 3. 去除噪声

信号 + 噪声：前 $k$ 个主成分捕获信号，后面捕获噪声。截断后重建 = 去噪。

### 4. 白化（Whitening）

$$
X_{\text{white}} = \tilde{X} V \Lambda^{-1/2}
$$

使各主成分方差为 1（相关性消除），常用于 ICA、深度学习预处理。

### 5. PCA vs t-SNE vs UMAP

- **PCA**：线性，全局结构，快速，可解释
- **t-SNE**：非线性，局部结构，慢，不可解释，不能推广
- **UMAP**：非线性，局部+全局，快，可推广

---

*→ 明天 D7：综合实战——图像压缩 + 推荐系统 + NLP 词向量*
