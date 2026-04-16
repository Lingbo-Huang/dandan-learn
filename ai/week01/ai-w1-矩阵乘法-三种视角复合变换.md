# Day 4：矩阵乘法

**主题**：矩阵乘法规则、几何直觉、性质与矩阵链式法则  
**预计时间**：2.5~3 小时

---

## 学习目标

- 掌握矩阵乘法的定义和计算方法
- 理解矩阵乘法的三种等价视角（行列点积 / 列组合 / 外积求和）
- 掌握矩阵乘法的基本性质（结合律、分配律、不满足交换律）
- 理解矩阵乘法描述复合线性变换的几何本质
- 能用 NumPy 高效完成矩阵运算

---

## 核心知识点

### 1. 矩阵乘法定义

$A$（$m \times k$）× $B$（$k \times n$）= $C$（$m \times n$）

**尺寸规则**：A 的列数必须等于 B 的行数，结果行数=A行数，列数=B列数

**逐元素公式**（行列点积视角）：

$$C_{ij} = \sum_{l=1}^k A_{il} B_{lj} = (\text{A 的第 } i \text{ 行}) \cdot (\text{B 的第 } j \text{ 列})$$

### 2. 三种等价计算视角

#### 视角 1：行列点积（逐元素）

$C_{ij}$ = A 第 $i$ 行 · B 第 $j$ 列

#### 视角 2：列组合（最重要！）

$AB$ 的**每一列** = $A$ 对 $B$ 对应列的线性组合

$$AB = A[\mathbf{b}_1 | \mathbf{b}_2 | \cdots | \mathbf{b}_n] = [A\mathbf{b}_1 | A\mathbf{b}_2 | \cdots | A\mathbf{b}_n]$$

**直觉**：$B$ 的每一列告诉 $A$ "怎么组合它的列"

#### 视角 3：行组合

$AB$ 的**每一行** = $B$ 对 $A$ 对应行的线性组合（从左边 $A$ 的角度）

#### 视角 4：外积之和（秩-1 分解）

$$AB = \sum_{l=1}^k \mathbf{a}_l \mathbf{b}_l^T$$

其中 $\mathbf{a}_l$ 是 $A$ 的第 $l$ 列，$\mathbf{b}_l^T$ 是 $B$ 的第 $l$ 行。每项是一个秩-1 矩阵（外积）。

### 3. 矩阵乘法的性质

| 性质 | 公式 | 备注 |
|---|---|---|
| **不满足交换律** | $AB \neq BA$（一般情况） | 最重要！ |
| 结合律 | $(AB)C = A(BC)$ | ✅ |
| 分配律 | $A(B+C) = AB + AC$ | ✅ |
| 单位矩阵 | $AI = IA = A$ | ✅ |
| 转置反转 | $(AB)^T = B^T A^T$ | 顺序翻转 |
| 零因子 | $AB = 0$ 不代表 $A=0$ 或 $B=0$ | ⚠️ |

### 4. 几何意义：复合线性变换

**矩阵 = 线性变换**：$A$ 将向量 $\mathbf{x}$ 映射为 $A\mathbf{x}$（旋转、缩放、剪切...）

**矩阵乘法 = 复合变换**：先用 $B$ 变换，再用 $A$ 变换：

$$(AB)\mathbf{x} = A(B\mathbf{x})$$

- $B$ 先作用于 $\mathbf{x}$，得到 $B\mathbf{x}$
- 再用 $A$ 作用于 $B\mathbf{x}$

**例**：旋转矩阵 $R_{30°}$ 先旋转 30°，再用 $R_{60°}$ 旋转 60°，等价于 $R_{90°}$：

$$R_{60°} \cdot R_{30°} = R_{90°}$$

---

## 示例与推导

### 示例 1：标准矩阵乘法

$$A = \begin{bmatrix}1&2\\3&4\end{bmatrix},\quad B = \begin{bmatrix}5&6\\7&8\end{bmatrix}$$

**行列点积法**：

$$C_{11} = [1,2]\cdot[5,7]^T = 5+14=19$$
$$C_{12} = [1,2]\cdot[6,8]^T = 6+16=22$$
$$C_{21} = [3,4]\cdot[5,7]^T = 15+28=43$$
$$C_{22} = [3,4]\cdot[6,8]^T = 18+32=50$$

$$AB = \begin{bmatrix}19&22\\43&50\end{bmatrix}$$

**验证不满足交换律**：

$$BA = \begin{bmatrix}5&6\\7&8\end{bmatrix}\begin{bmatrix}1&2\\3&4\end{bmatrix} = \begin{bmatrix}23&34\\31&46\end{bmatrix} \neq AB$$

### 示例 2：列组合视角

$AB$ 的第 1 列 = $A$ × ($B$ 的第 1 列)：

$$A\begin{bmatrix}5\\7\end{bmatrix} = 5\begin{bmatrix}1\\3\end{bmatrix} + 7\begin{bmatrix}2\\4\end{bmatrix} = \begin{bmatrix}5\\15\end{bmatrix}+\begin{bmatrix}14\\28\end{bmatrix}=\begin{bmatrix}19\\43\end{bmatrix}$$✅

### 示例 3：2D 旋转矩阵

旋转角 $\theta$ 的矩阵：

$$R_\theta = \begin{bmatrix}\cos\theta & -\sin\theta\\\sin\theta & \cos\theta\end{bmatrix}$$

先旋转 $\alpha$ 再旋转 $\beta$ = 旋转 $\alpha + \beta$：

$$R_\beta R_\alpha = R_{\alpha+\beta}$$

这就是矩阵乘法描述复合变换的几何意义！

---

## 动手练习

### 手算题

1. 计算 $AB$（先确认尺寸匹配）：
   $$A = \begin{bmatrix}2&0&1\\1&3&2\end{bmatrix},\quad B = \begin{bmatrix}1&2\\0&1\\3&0\end{bmatrix}$$

2. 验证 $(AB)^T = B^T A^T$（使用上题中的 $A, B$）

3. 构造一个 $2\times2$ 例子，说明 $AB \neq BA$

4. 若 $A = \begin{bmatrix}1&0\\0&-1\end{bmatrix}$，用几何直觉描述 $A$ 对 2D 向量做了什么变换？

### 代码练习

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
C = A @ B   # 推荐方式
print("A @ B =\n", C)
print("B @ A =\n", B @ A)
print("AB == BA?", np.allclose(A@B, B@A))  # False

# 验证 (AB)^T = B^T A^T
print("\n(AB)^T =\n", (A@B).T)
print("B^T A^T =\n", B.T @ A.T)
print("一致?", np.allclose((A@B).T, B.T @ A.T))

# 旋转矩阵演示
def rot_matrix(deg):
    theta = np.radians(deg)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

R30 = rot_matrix(30)
R60 = rot_matrix(60)
R90 = rot_matrix(90)

print("\nR60 @ R30 =\n", R60 @ R30)
print("R90 =\n", R90)
print("复合旋转一致?", np.allclose(R60 @ R30, R90))

# 可视化旋转效果
import matplotlib.pyplot as plt

v = np.array([1.0, 0.0])
v_rot30 = R30 @ v
v_rot90 = R90 @ v

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.2, 1.2)
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.annotate('', xy=v, xytext=[0,0], arrowprops=dict(arrowstyle='->', color='blue'))
ax.annotate('', xy=v_rot30, xytext=[0,0], arrowprops=dict(arrowstyle='->', color='orange'))
ax.annotate('', xy=v_rot90, xytext=[0,0], arrowprops=dict(arrowstyle='->', color='red'))
ax.legend(['原向量', '旋转30°', '旋转90°'])
ax.set_title('2D 旋转矩阵可视化')
plt.tight_layout()
plt.savefig('rotation_demo.png', dpi=120)
print("\n图片已保存为 rotation_demo.png")
```

**运行方式**：
```bash
cd ~/dandan-learn/ai/linalg-week01
uv run python day04_exercise.py
```

---

## 小结

| 概念 | 关键点 | 陷阱 |
|---|---|---|
| 乘法定义 | $C_{ij}$ = 行·列点积 | 尺寸：A 列数 = B 行数 |
| 列组合视角 | $AB$ 的列 = $A$ 作用于 $B$ 的列 | 最深刻的理解 |
| 不可交换 | $AB \neq BA$ | 经典错误！ |
| 转置 | $(AB)^T = B^T A^T$ | 顺序翻转 |
| 几何意义 | 复合变换 | $B$ 先，$A$ 后 |

**今日关键直觉**：矩阵乘法 $AB$ 的第 $j$ 列，就是 $A$ 对 $B$ 的第 $j$ 列做的线性变换——这让你从"复合变换"而非"逐元素计算"的角度理解矩阵乘法。

**明日预告**：D5 行列式——矩阵的"体积缩放因子"，理解为什么行列式为 0 意味着变换会"压缩"空间。
