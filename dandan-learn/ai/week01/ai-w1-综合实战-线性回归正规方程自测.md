# Day 7：综合复习与实战

**主题**：本周知识串联、NumPy 综合实战、线性方程组求解  
**预计时间**：3~4 小时

---

## 学习目标

- 系统梳理本周五大主题的核心知识点
- 用 NumPy 完成一个端到端的线性代数实战项目
- 掌握用矩阵方法解线性方程组（最小二乘法入门）
- 通过自测题检验掌握程度，找出薄弱点
- 为下周（矩阵分解：特征值、SVD）做好准备

---

## 本周知识点总览

### 知识地图

```
线性代数 Week 1
├── 向量
│   ├── 定义、加减、数乘
│   ├── 点积（→ 余弦相似度）
│   ├── 范数（L1、L2）
│   └── 投影（→ PCA 基础）
├── 向量进阶
│   ├── 线性组合 & Span
│   ├── 线性无关（→ 特征不冗余）
│   └── 向量空间 & 基
├── 矩阵基础
│   ├── 定义、加减、数乘
│   ├── 转置（及性质）
│   └── 特殊矩阵（I、对角、对称）
├── 矩阵乘法
│   ├── 行列点积定义
│   ├── 列组合视角（最核心！）
│   ├── 不可交换
│   └── 复合线性变换
├── 行列式
│   ├── 几何意义（面积/体积缩放）
│   ├── 2×2 / 3×3 计算
│   └── 性质（det(AB)=det(A)det(B)等）
└── 逆矩阵 & 秩
    ├── 存在条件（det ≠ 0，满秩）
    ├── 求法（公式 / 高斯-约旦）
    ├── 秩的概念
    └── 方程组可解性
```

---

## 核心概念速查表

| 概念 | 关键公式 | 陷阱/注意点 |
|---|---|---|
| 点积 | $\mathbf{a}\cdot\mathbf{b} = \sum a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ | 结果是标量 |
| L2 范数 | $\|\mathbf{v}\|_2 = \sqrt{\sum v_i^2}$ | 不是 $\sum v_i$ |
| 余弦相似度 | $\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$ | 值域 $[-1, 1]$ |
| 投影 | $\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{b}\|^2}\mathbf{b}$ | 结果是向量，不是标量 |
| 矩阵乘法 | $C_{ij} = \sum_k A_{ik}B_{kj}$ | $AB \neq BA$ |
| 转置乘法 | $(AB)^T = B^T A^T$ | 顺序翻转 |
| 2×2 行列式 | $ad - bc$ | 副对角线是减号 |
| 行列式为 0 | 矩阵奇异，不可逆 | 列线性相关 |
| 2×2 逆矩阵 | $\frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$ | 先交换对角，再变号 |
| 解方程组 | `np.linalg.solve(A, b)` | 比求逆再乘更稳定 |

---

## 综合实战：线性回归的矩阵形式

### 背景

线性回归是 ML 最基础的模型。本质上是用线性代数求解：

$$\mathbf{y} = X\boldsymbol{\theta} + \boldsymbol{\epsilon}$$

最优参数（最小二乘解）：

$$\boldsymbol{\theta}^* = (X^T X)^{-1} X^T \mathbf{y}$$

这里的 $X^T X$ 是矩阵乘法，$(X^T X)^{-1}$ 是逆矩阵——Week 1 所有内容都用上了！

### 实战代码

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ==============================
# Part 1: 生成数据
# ==============================
n = 50   # 样本数
x = np.linspace(0, 10, n)
y_true = 2.5 * x + 1.0  # 真实关系：y = 2.5x + 1
noise = np.random.randn(n) * 2
y = y_true + noise

# ==============================
# Part 2: 构造设计矩阵 X
# ==============================
# X 的第一列是 1（截距项），第二列是 x
X = np.column_stack([np.ones(n), x])
print("X 的形状:", X.shape)   # (50, 2)
print("y 的形状:", y.shape)   # (50,)

# ==============================
# Part 3: 用正规方程求解 θ
# ==============================
# θ* = (X^T X)^{-1} X^T y
XTX = X.T @ X
XTy = X.T @ y
print("\nX^T X =\n", XTX)
print("det(X^T X) =", np.linalg.det(XTX))   # 应 ≠ 0

# 方法 1：显式求逆（理论清晰但数值上略次）
theta_inv = np.linalg.inv(XTX) @ XTy
print("\n正规方程（inv法）θ =", theta_inv)

# 方法 2：np.linalg.solve（推荐）
theta_solve = np.linalg.solve(XTX, XTy)
print("正规方程（solve法）θ =", theta_solve)

# 方法 3：np.linalg.lstsq（最鲁棒）
theta_lstsq, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
print("最小二乘（lstsq法）θ =", theta_lstsq)

print("\n真实参数：截距=1.0，斜率=2.5")

# ==============================
# Part 4: 预测与可视化
# ==============================
y_pred = X @ theta_solve
residuals = y - y_pred
mse = np.mean(residuals**2)
print(f"\n均方误差 MSE = {mse:.4f}")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(x, y, alpha=0.6, label='数据点')
plt.plot(x, y_true, 'g--', label='真实关系 y=2.5x+1')
plt.plot(x, y_pred, 'r-', label=f'拟合线 y={theta_solve[1]:.2f}x+{theta_solve[0]:.2f}')
plt.legend(); plt.title('线性回归拟合'); plt.xlabel('x'); plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', lw=1)
plt.xlabel('预测值'); plt.ylabel('残差'); plt.title('残差图')

plt.tight_layout()
plt.savefig('linear_regression_linalg.png', dpi=120)
print("图片已保存为 linear_regression_linalg.png")

# ==============================
# Part 5: 综合验证本周知识点
# ==============================
print("\n=== 本周知识点综合验证 ===")

# 向量点积与余弦相似度
a = np.array([1.0, 2, 3])
b = np.array([4.0, 5, 6])
print(f"\n点积 a·b = {np.dot(a, b)}")
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"余弦相似度 = {cos_sim:.4f}")

# 矩阵乘法与转置
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"\nAB =\n{A@B}")
print(f"(AB)^T = B^T A^T? {np.allclose((A@B).T, B.T @ A.T)}")

# 行列式与逆
print(f"\ndet(A) = {np.linalg.det(A):.2f}")
A_inv = np.linalg.inv(A)
print(f"A^{{-1}} =\n{A_inv}")
print(f"A @ A^{{-1}} = I? {np.allclose(A @ A_inv, np.eye(2))}")

# 矩阵秩
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\nrank([[1,2,3],[4,5,6],[7,8,9]]) = {np.linalg.matrix_rank(M)}（奇异）")
```

**运行方式**：
```bash
cd ~/dandan-learn/ai/linalg-week01
uv run python day07_comprehensive.py
```

---

## 本周自测题

### 选择/判断题

1. （T/F）矩阵乘法满足交换律：$AB = BA$
2. （T/F）若 $\det(A) = 0$，则 $A$ 不可逆
3. 向量 $[3, 4]^T$ 的 L2 范数是：A. 7  B. 5  C. 25
4. $(AB)^T$ 等于：A. $A^T B^T$  B. $B^T A^T$  C. $A^T B$

### 计算题

1. $A = \begin{bmatrix}2&3\\1&4\end{bmatrix}$，求 $\det(A)$ 和 $A^{-1}$

2. $\mathbf{a} = [1,2,2]^T$，$\mathbf{b} = [2,1,-2]^T$，求：
   - 点积 $\mathbf{a}\cdot\mathbf{b}$
   - $\mathbf{a}$ 在 $\mathbf{b}$ 上的投影向量
   - 两向量是否正交？

3. 解方程组：
   $$\begin{cases}2x + y = 5 \\ x + 3y = 10\end{cases}$$
   （写成 $A\mathbf{x} = \mathbf{b}$ 形式，用逆矩阵法求解）

### 参考答案

1. F；T；B（$\sqrt{9+16}=5$）；B
2. $\det=5$；$A^{-1}=\frac{1}{5}\begin{bmatrix}4&-3\\-1&2\end{bmatrix}$
3. 点积 = $2+2-4=0$（正交！）；投影为零向量
4. $\mathbf{x} = A^{-1}\mathbf{b} = \frac{1}{5}\begin{bmatrix}3&-1\\-1&2\end{bmatrix}\begin{bmatrix}5\\10\end{bmatrix} = \begin{bmatrix}1\\3\end{bmatrix}$

---

## 小结

**Week 1 完成！你现在掌握了：**

- ✅ 向量运算（点积、范数、投影、余弦相似度）
- ✅ 矩阵运算（加减、数乘、转置、特殊矩阵）
- ✅ 矩阵乘法（行列视角、列组合视角、复合变换）
- ✅ 行列式（几何意义、计算、性质）
- ✅ 逆矩阵与秩（存在条件、求法、方程组求解）

**ML 中的应用线索：**

| 本周知识 | ML 对应场景 |
|---|---|
| 点积 | 神经元计算、注意力分数 |
| 余弦相似度 | 词向量相似度、推荐系统 |
| 矩阵乘法 | 全连接层前向传播 |
| 投影 | PCA 降维 |
| 逆矩阵 | 线性回归正规方程 |
| 行列式 | 判断矩阵可逆性 |

**下周预告（Week 2）：** 矩阵分解——特征值分解（EVD）、奇异值分解（SVD）、PCA 的矩阵原理。
