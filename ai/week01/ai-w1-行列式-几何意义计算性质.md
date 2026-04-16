# Day 5：行列式

**主题**：行列式的定义、计算方法、几何意义与性质  
**预计时间**：2.5~3 小时

---

## 学习目标

- 理解行列式的几何含义（面积/体积的缩放因子）
- 掌握 2×2、3×3 矩阵行列式的手算方法
- 掌握行列式的重要性质
- 理解行列式为 0 的几何与代数意义
- 能用 NumPy 计算行列式

---

## 核心知识点

### 1. 行列式的几何意义

行列式 $\det(A)$ 描述矩阵 $A$ 所代表的线性变换**对面积（2D）或体积（3D）的缩放倍数**：

- $|\det(A)|$ = 缩放倍数
- $\det(A) > 0$：保持方向（不翻转）
- $\det(A) < 0$：翻转方向
- $\det(A) = 0$：空间被压缩到更低维度（降维，不可逆！）

**例**：单位正方形经过矩阵 $A$ 变换后，面积变为 $|\det(A)|$。

### 2. 2×2 矩阵行列式

$$\det\begin{bmatrix}a&b\\c&d\end{bmatrix} = ad - bc$$

**几何直觉**：两列向量 $[a,c]^T$ 和 $[b,d]^T$ 构成的平行四边形面积。

### 3. 3×3 矩阵行列式（沙吕法则）

$$A = \begin{bmatrix}a&b&c\\d&e&f\\g&h&i\end{bmatrix}$$

$$\det(A) = a(ei-fh) - b(di-fg) + c(dh-eg)$$

**按第一行展开（余子式展开）**：

$$\det(A) = a \cdot M_{11} - b \cdot M_{12} + c \cdot M_{13}$$

其中 $M_{ij}$ 是划去第 $i$ 行第 $j$ 列后剩余的 $2\times2$ 行列式（余子式）。

**符号规则**（代数余子式 $C_{ij} = (-1)^{i+j}M_{ij}$）：

$$\begin{bmatrix}+&-&+\\-&+&-\\+&-&+\end{bmatrix}$$

### 4. 行列式的重要性质

| 性质 | 说明 |
|---|---|
| $\det(I) = 1$ | 单位矩阵不缩放 |
| $\det(A^T) = \det(A)$ | 转置不改变行列式 |
| $\det(AB) = \det(A)\det(B)$ | 乘法对应缩放因子相乘 |
| 行交换 → 行列式变号 | 翻转一次方向 |
| 某行乘以 $c$ → 行列式乘以 $c$ | 线性缩放 |
| 某行加上另一行的倍数 → 行列式不变 | 高斯消元的基础 |
| $\det(A) = 0$ ⟺ $A$ 奇异（不可逆） | 核心判断依据 |
| $\det(cA) = c^n \det(A)$（$n$ 阶矩阵） | 数乘 $n$ 次 |

---

## 示例与推导

### 示例 1：2×2 行列式

$$A = \begin{bmatrix}3&1\\2&4\end{bmatrix}$$

$$\det(A) = 3\times4 - 1\times2 = 12 - 2 = 10$$

**几何验证**：列向量 $[3,2]^T$ 和 $[1,4]^T$ 构成的平行四边形面积为 10。

### 示例 2：3×3 行列式

$$A = \begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix}$$

按第一行展开：

$$\det(A) = 1\cdot\det\begin{bmatrix}5&6\\8&9\end{bmatrix} - 2\cdot\det\begin{bmatrix}4&6\\7&9\end{bmatrix} + 3\cdot\det\begin{bmatrix}4&5\\7&8\end{bmatrix}$$

$$= 1(45-48) - 2(36-42) + 3(32-35)$$

$$= 1(-3) - 2(-6) + 3(-3)$$

$$= -3 + 12 - 9 = 0$$

行列式为 0！说明这三个行向量**线性相关**（第三行 = 2×第二行 - 第一行），矩阵不可逆。✅

### 示例 3：几何意义验证

$$A = \begin{bmatrix}2&0\\0&3\end{bmatrix}$$（x 方向拉伸 2 倍，y 方向拉伸 3 倍）

$$\det(A) = 2\times3 - 0\times0 = 6$$

单位正方形（面积 1）→ 变换后面积 = 6。✅

---

## 动手练习

### 手算题

1. 计算行列式：
   - $\det\begin{bmatrix}5&-2\\3&1\end{bmatrix}$
   - $\det\begin{bmatrix}2&1&0\\-1&3&2\\4&0&1\end{bmatrix}$

2. 不计算直接判断：
   - $A = \begin{bmatrix}1&2&3\\2&4&6\\0&1&0\end{bmatrix}$ 的行列式是否为 0？为什么？
   - 旋转矩阵 $R_\theta = \begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}$ 的行列式是多少？（提示：旋转不改变面积）

3. 若 $\det(A) = 3$，$\det(B) = -2$，求 $\det(AB)$ 和 $\det(2A)$（$A$ 为 $3\times3$）

### 代码练习

```python
import numpy as np
import matplotlib.pyplot as plt

# 基本行列式计算
A = np.array([[3, 1], [2, 4]])
print("det(A) =", np.linalg.det(A))

B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("det(B) =", np.linalg.det(B))  # 应接近 0

# 验证性质：det(AB) = det(A)det(B)
A2 = np.array([[1, 2], [3, 4]])
B2 = np.array([[5, 6], [7, 8]])
print("\ndet(A) * det(B) =", np.linalg.det(A2) * np.linalg.det(B2))
print("det(A @ B) =", np.linalg.det(A2 @ B2))

# 验证 det(A^T) = det(A)
print("det(A.T) =", np.linalg.det(A2.T))

# 几何可视化：变换前后的面积
def plot_transform(A, ax, title):
    # 单位正方形的四个角
    square = np.array([[0,1,1,0,0],
                       [0,0,1,1,0]])
    transformed = A @ square
    ax.fill(square[0], square[1], alpha=0.3, color='blue', label='变换前')
    ax.fill(transformed[0], transformed[1], alpha=0.3, color='red', label='变换后')
    ax.set_xlim(-1, 4); ax.set_ylim(-1, 4)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.set_title(f'{title}\ndet = {np.linalg.det(A):.2f}')

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# 缩放变换
A_scale = np.array([[2, 0], [0, 1.5]])
plot_transform(A_scale, axes[0], '缩放变换')

# 剪切变换
A_shear = np.array([[1, 1], [0, 1]])
plot_transform(A_shear, axes[1], '剪切变换')

# 旋转变换（面积不变）
theta = np.pi / 6
A_rot = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
plot_transform(A_rot, axes[2], '旋转变换')

plt.tight_layout()
plt.savefig('determinant_geometry.png', dpi=120)
print("\n几何可视化已保存为 determinant_geometry.png")
```

**运行方式**：
```bash
cd ~/dandan-learn/ai/linalg-week01
uv run python day05_exercise.py
```

---

## 小结

| 概念 | 公式/定义 | 意义 |
|---|---|---|
| 2×2 行列式 | $ad - bc$ | 平行四边形面积 |
| 3×3 行列式 | 余子式展开 | 平行六面体体积 |
| 行列式为 0 | 矩阵奇异，列线性相关 | 不可逆，压缩空间 |
| $\det(AB)$ | $\det(A)\det(B)$ | 缩放因子可乘 |
| 行交换 | 行列式变号 | 方向翻转 |

**今日关键直觉**：行列式 = 矩阵描述的变换对"体积"的缩放倍数。如果变换把一个体压扁成面（降维），面积/体积变为 0，行列式就是 0，矩阵就是不可逆的。

**明日预告**：D6 矩阵的逆与秩——什么时候逆存在？怎么求？矩阵秩又告诉我们什么？
