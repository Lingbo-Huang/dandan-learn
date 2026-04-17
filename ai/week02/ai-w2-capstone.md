# D7：综合实战——人脸识别（Eigenfaces）与图像压缩

> **Week 2 · Day 7** | AI 基础线：线性代数核心概念

---

## 1. Week 2 知识地图回顾

在进入实战前，让我们把 Week2 的知识串联起来：

```
特征值 (D1)
    └→ 特征向量 / 特征空间 (D2)
           └→ 矩阵对角化 A = PDP⁻¹ (D3)
                   ↓
              局限：只适用于方阵且可对角化
                   ↓
         SVD 定理 A = UΣVᵀ (D4)  ←  任意矩阵均适用
              └→ 数值计算 / NumPy 实现 (D5)
                   └→ PCA = 对中心化数据做 SVD (D6)
                           └→ 本篇：人脸识别 + 图像压缩综合实战 (D7)
```

---

## 2. 人脸识别：Eigenfaces 方法

### 2.1 原理

**Eigenfaces**（特征脸）是 Turk & Pentland 1991 年提出的人脸识别算法，本质上是 PCA 在人脸数据上的应用。

**核心思路：**

1. 将每张人脸图像展开为向量（如 $64 \times 64$ 图像 → 4096 维向量）
2. 对人脸数据库做 PCA，得到"特征脸"（主成分方向）
3. 每张人脸 = 均值脸 + 特征脸的线性组合
4. 识别：计算新人脸在特征脸空间的投影，找最近邻

**为什么有效？**

人脸图像虽然是高维数据（数千像素），但实际的"人脸流形"维度很低——所有人脸都由少数几个主成分（光照、姿态、表情等）控制。PCA 正好能找到这个低维结构。

### 2.2 数学表达

设 $n$ 张人脸图像展开为行向量后堆成矩阵 $X \in \mathbb{R}^{n \times d}$，对中心化数据 $\tilde{X} = X - \bar{X}$ 做 SVD：

$$
\tilde{X} = U\Sigma V^T
$$

- **特征脸** = $V$ 的前 $k$ 列（每列 reshape 回图像大小即为一张"特征脸"）
- **人脸编码** = $Z = \tilde{X}V_k = U_k\Sigma_k$（$n \times k$）
- **识别** = 对新图像 $\mathbf{x}$：$\mathbf{z} = (\mathbf{x} - \bar{\mathbf{x}})V_k$，找与 $Z$ 中最近的行

---

## 3. 图像压缩：彩色图像的 SVD

### 3.1 原理

对彩色图像的每个通道（R/G/B）分别做 SVD，保留前 $k$ 个奇异三元组，实现压缩。

**压缩比：**

$$
\text{压缩比} = \frac{m \times n}{k(m + n + 1)}
$$

例如：$512 \times 512$ 图像，$k = 50$：

$$
\text{压缩比} = \frac{512 \times 512}{50 \times (512 + 512 + 1)} \approx 5.1\times
$$

---

## 4. 完整实战代码

### 4.1 环境准备

```bash
uv init week2-capstone
cd week2-capstone
uv add numpy scikit-learn matplotlib pillow scipy
```

### 4.2 实战一：Eigenfaces（基于 Olivetti 人脸数据集）

```python
# eigenfaces.py
"""
Eigenfaces 人脸识别综合实战
数据集：sklearn 内置 Olivetti Faces (400张, 40人, 64×64)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

# ============ 1. 加载数据 ============
print("加载 Olivetti 人脸数据集...")
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = faces.data, faces.target  # X: (400, 4096), y: 0~39

print(f"数据集: {X.shape[0]} 张人脸, {len(np.unique(y))} 个人")
print(f"图像尺寸: {faces.images.shape[1]}×{faces.images.shape[2]} 像素")

# ============ 2. 划分训练/测试集 ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

# ============ 3. Eigenfaces（从零实现 PCA）============
class EigenfacesPCA:
    def __init__(self, n_components=50):
        self.n_components = n_components
    
    def fit(self, X):
        # 计算均值脸
        self.mean_face_ = X.mean(axis=0)
        X_centered = X - self.mean_face_
        
        # SVD（薄型）
        # 注意：当 n < d 时，用 X X^T (n×n) 而不是 X^T X (d×d) 更高效
        n = X_centered.shape[0]
        if n < X_centered.shape[1]:
            # 计算 n×n 的 Gram 矩阵 X X^T
            gram = X_centered @ X_centered.T / (n - 1)
            eigenvalues, eigenvectors_gram = np.linalg.eigh(gram)
            # 按降序排列
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors_gram = eigenvectors_gram[:, idx]
            # 恢复原始空间的特征向量
            # v_i = (1/σ_i) X^T u_i
            s = np.sqrt(np.maximum(eigenvalues[:self.n_components] * (n-1), 0))
            components = []
            for i in range(self.n_components):
                if s[i] > 1e-10:
                    comp = X_centered.T @ eigenvectors_gram[:, i] / s[i]
                else:
                    comp = np.zeros(X_centered.shape[1])
                components.append(comp)
            self.components_ = np.array(components)  # (k, d)
            self.explained_variance_ = eigenvalues[:self.n_components]
        else:
            U, sv, Vt = np.linalg.svd(X_centered, full_matrices=False)
            self.components_ = Vt[:self.n_components]
            self.explained_variance_ = sv[:self.n_components]**2 / (n - 1)
        
        total_var = X_centered.var(axis=0).sum()
        self.explained_variance_ratio_ = self.explained_variance_ / (total_var * (n-1) / n)
        return self
    
    def transform(self, X):
        return (X - self.mean_face_) @ self.components_.T
    
    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_face_

# 拟合 Eigenfaces
pca = EigenfacesPCA(n_components=80)
pca.fit(X_train)
Z_train = pca.transform(X_train)
Z_test = pca.transform(X_test)

var_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
print(f"\n前80个主成分累积方差解释率: {var_ratio_cumsum[-1]*100:.1f}%")
print(f"前10个: {var_ratio_cumsum[9]*100:.1f}%")
print(f"前30个: {var_ratio_cumsum[29]*100:.1f}%")

# ============ 4. 最近邻分类器 ============
def knn_classify(Z_train, y_train, Z_test, k=1):
    """k-近邻分类"""
    dist_matrix = cdist(Z_test, Z_train, metric='euclidean')
    top_k_idx = np.argsort(dist_matrix, axis=1)[:, :k]
    predictions = []
    for row in top_k_idx:
        labels = y_train[row]
        pred = np.bincount(labels).argmax()
        predictions.append(pred)
    return np.array(predictions)

y_pred = knn_classify(Z_train, y_train, Z_test, k=1)
acc = accuracy_score(y_test, y_pred)
print(f"\n人脸识别准确率（1-NN, k=80 主成分）: {acc*100:.1f}%")

# 不同主成分数的识别率
print("\n主成分数 vs 识别准确率:")
print(f"{'k':<8} {'准确率':<12} {'解释方差'}")
for k in [5, 10, 20, 40, 60, 80]:
    pca_k = EigenfacesPCA(n_components=k)
    pca_k.fit(X_train)
    Z_tr = pca_k.transform(X_train)
    Z_te = pca_k.transform(X_test)
    y_hat = knn_classify(Z_tr, y_train, Z_te, k=1)
    acc_k = accuracy_score(y_test, y_hat)
    var_k = pca_k.explained_variance_ratio_.sum() * 100
    print(f"{k:<8} {acc_k*100:<12.1f}% {var_k:.1f}%")

# ============ 5. 可视化 ============
n_show = 10
img_size = (64, 64)

fig, axes = plt.subplots(3, n_show, figsize=(20, 7))

# 第一行：原始人脸
for i in range(n_show):
    axes[0, i].imshow(X_test[i].reshape(img_size), cmap='gray')
    axes[0, i].axis('off')
    color = 'green' if y_pred[i] == y_test[i] else 'red'
    axes[0, i].set_title(f'真:{y_test[i]}\n预:{y_pred[i]}', 
                           fontsize=8, color=color)
axes[0, 0].set_ylabel('测试图像', fontsize=10)

# 第二行：PCA 重建（80 个主成分）
Z_test_80 = pca.transform(X_test)
X_recon = pca.inverse_transform(Z_test_80)
for i in range(n_show):
    recon = np.clip(X_recon[i], 0, 1)
    axes[1, i].imshow(recon.reshape(img_size), cmap='gray')
    axes[1, i].axis('off')
axes[1, 0].set_ylabel('80主成分重建', fontsize=10)

# 第三行：特征脸（主成分可视化）
for i in range(n_show):
    ef = pca.components_[i]
    ef_norm = (ef - ef.min()) / (ef.max() - ef.min() + 1e-10)
    axes[2, i].imshow(ef_norm.reshape(img_size), cmap='gray')
    axes[2, i].axis('off')
    axes[2, i].set_title(f'EF {i+1}', fontsize=8)
axes[2, 0].set_ylabel('特征脸 (EigenFaces)', fontsize=10)

plt.suptitle('Eigenfaces 人脸识别综合展示', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('eigenfaces_result.png', dpi=150, bbox_inches='tight')
print("\n特征脸可视化已保存")
```

### 4.3 实战二：彩色图像 SVD 压缩

```python
# color_image_svd.py
"""
彩色图像 SVD 压缩实战
对 RGB 三通道分别做 SVD，保留前 k 个奇异三元组
"""
import numpy as np
import matplotlib.pyplot as plt

def compress_channel(channel, k):
    """对单通道做 SVD 低秩近似"""
    U, s, Vt = np.linalg.svd(channel, full_matrices=False)
    compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    return np.clip(compressed, 0, 255)

def compress_image(img_array, k):
    """对 RGB 图像的每个通道做 SVD 压缩"""
    compressed = np.zeros_like(img_array, dtype=float)
    for c in range(3):
        compressed[:, :, c] = compress_channel(img_array[:, :, c].astype(float), k)
    return compressed.astype(np.uint8)

def storage_info(m, n, k, channels=3):
    """计算存储量（元素个数）"""
    original = m * n * channels
    compressed = k * (m + n + 1) * channels
    return original, compressed, original / compressed

# 生成测试彩色图像（使用数学函数生成有结构的图案）
def make_color_test_image(size=256):
    """生成带结构的彩色测试图像"""
    x, y = np.meshgrid(np.linspace(0, 6*np.pi, size),
                        np.linspace(0, 6*np.pi, size))
    R = ((np.sin(x) * np.cos(y) + 1) / 2 * 255).astype(np.uint8)
    G = ((np.sin(x + y/2) + 1) / 2 * 255).astype(np.uint8)
    B = ((np.cos(x/2 - y) + 1) / 2 * 255).astype(np.uint8)
    return np.stack([R, G, B], axis=2)

img = make_color_test_image(256)
m, n = img.shape[:2]
print(f"原始图像: {m}×{n}×3 = {m*n*3:,} 个像素值")

# 计算所有奇异值（用于选择 k）
s_r = np.linalg.svd(img[:,:,0].astype(float), compute_uv=False)
s_g = np.linalg.svd(img[:,:,1].astype(float), compute_uv=False)
s_b = np.linalg.svd(img[:,:,2].astype(float), compute_uv=False)
total_energy = np.sum(s_r**2) + np.sum(s_g**2) + np.sum(s_b**2)

# 不同 k 值的压缩效果
k_values = [1, 5, 10, 20, 50, 100]
fig, axes = plt.subplots(2, len(k_values)+1, figsize=(21, 7))

# 原始图像
axes[0, 0].imshow(img)
axes[0, 0].set_title(f'原始\n{m}×{n}')
axes[0, 0].axis('off')
axes[1, 0].axis('off')

for idx, k in enumerate(k_values):
    compressed = compress_image(img, k)
    orig_size, comp_size, ratio = storage_info(m, n, k)
    
    # 计算 PSNR
    mse = np.mean((img.astype(float) - compressed.astype(float))**2)
    if mse > 0:
        psnr = 10 * np.log10(255**2 / mse)
    else:
        psnr = float('inf')
    
    # 累积能量
    energy_k = (np.sum(s_r[:k]**2) + np.sum(s_g[:k]**2) + np.sum(s_b[:k]**2))
    energy_ratio = energy_k / total_energy * 100
    
    axes[0, idx+1].imshow(compressed)
    axes[0, idx+1].set_title(
        f'k={k}\n压缩比={ratio:.1f}x\nPSNR={psnr:.1f}dB', fontsize=9)
    axes[0, idx+1].axis('off')
    
    axes[1, idx+1].axis('off')
    axes[1, idx+1].text(0.5, 0.5, 
        f'存储量:\n{comp_size:,}\n原始:\n{orig_size:,}\n能量:{energy_ratio:.1f}%',
        ha='center', va='center', fontsize=9, transform=axes[1, idx+1].transAxes)

plt.suptitle('SVD 彩色图像压缩：不同截断秩效果对比', fontsize=14)
plt.tight_layout()
plt.savefig('color_svd_compression.png', dpi=150, bbox_inches='tight')
print("彩色图像压缩结果已保存")

# 奇异值衰减分析
fig2, ax = plt.subplots(figsize=(10, 5))
for s, color, label in [(s_r, 'red', 'R通道'), 
                          (s_g, 'green', 'G通道'), 
                          (s_b, 'blue', 'B通道')]:
    ax.plot(range(1, len(s)+1), s / s[0], color=color, label=label, linewidth=1.5)
ax.set_xlabel('奇异值序号')
ax.set_ylabel('归一化奇异值（σi/σ1）')
ax.set_title('各颜色通道奇异值衰减曲线')
ax.set_yscale('log')
ax.set_xlim(1, 100)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('svd_singular_values_decay.png', dpi=150, bbox_inches='tight')
print("奇异值衰减曲线已保存")
```

---

## 5. Week 2 知识体系总结

### 5.1 核心公式一览

| 公式 | 名称 | 含义 |
|------|------|------|
| $A\mathbf{v} = \lambda\mathbf{v}$ | 特征方程 | $\mathbf{v}$ 经变换后只缩放不旋转 |
| $\det(A-\lambda I) = 0$ | 特征多项式 | 求所有特征值 |
| $A = PDP^{-1}$ | 对角化 | 存在 $n$ 个线性无关特征向量时成立 |
| $A = Q\Lambda Q^T$ | 谱分解 | 对称矩阵的正交对角化 |
| $A = U\Sigma V^T$ | SVD | 任意矩阵的通用分解 |
| $A \approx U_k\Sigma_k V_k^T$ | 截断 SVD | Frobenius 范数最优低秩近似 |
| $C = \frac{1}{n-1}\tilde{X}^T\tilde{X} = V\frac{\Sigma^2}{n-1}V^T$ | PCA-SVD 联系 | PCA = 中心化数据的 SVD |

### 5.2 算法选择指南

```
需要矩阵分解？
├─ 方阵 + 可对角化 → EVD（特征值分解）
│   ├─ 对称矩阵 → eigh（更稳定）
│   └─ 一般方阵 → eig
└─ 任意矩阵 → SVD
    ├─ 需要所有奇异值 → np.linalg.svd / scipy.linalg.svd
    ├─ 只要前 k 个 → scipy.sparse.linalg.svds
    └─ 超大稀疏矩阵 → randomized_svd（sklearn）
```

### 5.3 AI 应用地图

| 应用 | 核心算法 | 本周概念 |
|------|---------|---------|
| PCA 降维 | $\tilde{X} = U\Sigma V^T$ | SVD、特征值 |
| Eigenfaces 人脸识别 | PCA + 最近邻 | 特征空间、低秩近似 |
| 图像/视频压缩 | 截断 SVD | Eckart-Young 定理 |
| 推荐系统（协同过滤）| SVD 分解用户-物品矩阵 | 低秩近似 |
| 文本主题模型（LSA）| SVD on TF-IDF 矩阵 | 奇异向量语义含义 |
| 谱聚类 | 图 Laplacian 特征向量 | 特征空间 |
| 神经网络剪枝 | 权重矩阵低秩近似 | 截断 SVD |
| PageRank | 主特征向量（幂迭代）| 特征值收敛性 |

---

## 6. 小结与展望

**Week 2 完整打通了线性代数中最重要的一条主线：**

$$
\text{特征值} \to \text{特征空间} \to \text{对角化} \to \text{SVD} \to \text{PCA} \to \text{人脸识别/图像压缩}
$$

每一步都建立在前一步的基础上：
- **特征值**告诉我们矩阵的"拉伸比例"
- **特征空间**告诉我们在哪些方向上拉伸
- **对角化**让矩阵计算变简单
- **SVD**把对角化推广到任意矩阵
- **PCA**用 SVD 找数据的主变化方向
- **实战**把这一切组合起来解决真实问题

**Week 3 预告**：深度学习基础——从感知机到反向传播，梯度下降背后的微积分，带你打通神经网络的数学核心。

---

*恭喜完成 Week 2！📐 你已经掌握了机器学习中最重要的线性代数工具。*

*参考：Turk & Pentland 1991《Eigenfaces for Recognition》；Golub & Van Loan《Matrix Computations》；3Blue1Brown《线性代数的本质》系列*
