# Day 1：向量基础

**主题**：向量的定义、加减法、数乘、点积、范数  
**预计时间**：2.5~3 小时

---

## 学习目标

- 理解向量的几何与代数双重含义
- 掌握向量加减法、数乘的计算与几何直觉
- 能计算两向量点积，理解点积与夹角的关系
- 理解 L1 范数、L2 范数（欧氏距离）及其意义
- 用 NumPy 验证所有手算结果

---

## 核心知识点

### 1. 向量的定义

向量是一个有**方向**和**大小**的量，可以用列向量表示：

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

- **几何直觉**：2D 平面上，向量 $\mathbf{v} = [3, 2]^T$ 表示"向右 3、向上 2"的箭头
- **在 ML 中**：一条数据记录、一个词向量、一张图片的像素值，都是向量

### 2. 向量加法与数乘

**加法**（对应元素相加）：

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}$$

**几何直觉**：首尾相接，平行四边形法则

**数乘**（标量 × 向量）：

$$c \cdot \mathbf{v} = \begin{bmatrix} c v_1 \\ c v_2 \end{bmatrix}$$

**几何直觉**：缩放向量长度（c < 0 时方向反转）

### 3. 点积（内积）

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i = |\mathbf{a}||\mathbf{b}|\cos\theta$$

- 结果是**标量**
- $\theta$ 是两向量夹角
- 点积为 0 ⟺ 两向量**正交**（垂直）
- 在 ML 中：衡量相似度（如余弦相似度的分子）

### 4. 向量范数

**L2 范数（欧氏范数）**——最常用：

$$\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

**L1 范数（曼哈顿范数）**：

$$\|\mathbf{v}\|_1 = |v_1| + |v_2| + \cdots + |v_n|$$

**单位向量**：$\hat{\mathbf{v}} = \dfrac{\mathbf{v}}{\|\mathbf{v}\|_2}$，长度为 1

---

## 示例与推导

### 示例 1：向量运算

设 $\mathbf{a} = [1, 2, 3]^T$，$\mathbf{b} = [4, 0, -1]^T$

**加法**：
$$\mathbf{a} + \mathbf{b} = [1+4,\ 2+0,\ 3+(-1)]^T = [5, 2, 2]^T$$

**数乘**：
$$2\mathbf{a} = [2, 4, 6]^T$$

**点积**：
$$\mathbf{a} \cdot \mathbf{b} = 1 \times 4 + 2 \times 0 + 3 \times (-1) = 4 + 0 - 3 = 1$$

**L2 范数**：
$$\|\mathbf{a}\|_2 = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{14} \approx 3.742$$

### 示例 2：用点积求夹角

$$\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2} = \frac{1}{\sqrt{14} \cdot \sqrt{17}} \approx 0.0648$$

$$\theta = \arccos(0.0648) \approx 86.3°$$

---

## 动手练习

### 手算题

1. 已知 $\mathbf{u} = [2, -1, 4]^T$，$\mathbf{v} = [1, 3, -2]^T$，计算：
   - $\mathbf{u} + \mathbf{v}$
   - $3\mathbf{u} - 2\mathbf{v}$
   - $\mathbf{u} \cdot \mathbf{v}$
   - $\|\mathbf{u}\|_2$，$\|\mathbf{v}\|_2$
   - $\mathbf{u}$ 与 $\mathbf{v}$ 的夹角 $\theta$

2. 若 $\mathbf{a} = [1, 0]^T$，$\mathbf{b} = [0, 1]^T$，点积是多少？这说明什么？

### 代码练习

```python
# 环境：uv run python（在 linalg-week01 项目目录下）
import numpy as np

# 定义向量
a = np.array([1, 2, 3])
b = np.array([4, 0, -1])

# 加法
print("a + b =", a + b)

# 数乘
print("2a =", 2 * a)

# 点积
dot = np.dot(a, b)
print("a · b =", dot)

# 也可以用 @ 运算符
print("a @ b =", a @ b)

# L2 范数
norm_a = np.linalg.norm(a)       # 默认 L2
norm_b = np.linalg.norm(b)
print("||a||₂ =", norm_a)

# L1 范数
print("||a||₁ =", np.linalg.norm(a, ord=1))

# 夹角
cos_theta = dot / (norm_a * norm_b)
theta = np.degrees(np.arccos(cos_theta))
print(f"夹角 = {theta:.2f}°")

# 单位向量
a_unit = a / norm_a
print("单位向量 â =", a_unit)
print("验证 ||â||₂ =", np.linalg.norm(a_unit))
```

**运行方式**：
```bash
cd ~/dandan-learn/ai/linalg-week01
uv run python day01_exercise.py
```

---

## 小结

| 概念 | 公式/定义 | ML 中的意义 |
|---|---|---|
| 向量 | $\mathbf{v} \in \mathbb{R}^n$ | 特征、样本、权重 |
| 加法 | 对应元素相加 | 残差累加、梯度更新 |
| 数乘 | 标量 × 每个元素 | 学习率缩放梯度 |
| 点积 | $\sum a_i b_i$ | 相似度、线性变换核心 |
| L2 范数 | $\sqrt{\sum v_i^2}$ | 欧氏距离、正则化 |

**今日关键直觉**：点积 = 一个向量在另一个向量方向上的投影长度 × 另一个向量的长度。当两向量完全同向时点积最大，正交时为 0，反向时为负。

**明日预告**：D2 继续向量进阶——线性无关、向量空间、投影与余弦相似度。
