---
layout: default
title: "D2 · 线性回归原理与推导"
---

# D2 · 线性回归原理与推导

> **Week 3 · AI 基础线**  
> 今天从零推导线性回归——不只是"怎么用"，而是"为什么这么做"。

---

## 一、问题建立

**场景**：预测房价。已知 100 套房子的面积（m²）和售价（万元），建一个模型预测任意面积的房价。

**符号约定**：
- $m$：样本数量（100 套）
- $n$：特征数量（先从 1 个特征开始）
- $x^{(i)}$：第 $i$ 个样本的特征值
- $y^{(i)}$：第 $i$ 个样本的真实标签

**模型假设**：价格和面积是线性关系

$$h_\theta(x) = \theta_0 + \theta_1 x$$

$\theta_0$ 是截距（bias），$\theta_1$ 是斜率（weight）。  
这两个参数就是我们要学的东西。

---

## 二、损失函数：怎么衡量"差"

模型对某个样本的预测误差：

$$e^{(i)} = h_\theta(x^{(i)}) - y^{(i)}$$

为什么不直接用误差之和？因为正负会抵消。  
为什么不用绝对值之和（MAE）？因为绝对值函数在 0 处不可微，梯度下降会有问题。

**均方误差（MSE）**：

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

> 前面的 $\frac{1}{2}$ 是为了求导后形式好看（2 和 $\frac{1}{2}$ 消掉）。

---

## 三、解析解：正规方程

对于线性回归，存在**解析解**——直接求出使损失最小的参数，不需要迭代。

**矩阵表示**：

$$J(\mathbf{\theta}) = \frac{1}{2m} \| X\mathbf{\theta} - \mathbf{y} \|^2$$

其中 $X$ 是设计矩阵（每行一个样本，第一列全是 1 对应截距）：

$$X = \begin{bmatrix} 1 & x^{(1)} \\ 1 & x^{(2)} \\ \vdots & \vdots \\ 1 & x^{(m)} \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \end{bmatrix}$$

**求导并令梯度为 0**：

$$\frac{\partial J}{\partial \mathbf{\theta}} = \frac{1}{m} X^T(X\mathbf{\theta} - \mathbf{y}) = 0$$

$$X^T X \mathbf{\theta} = X^T \mathbf{y}$$

**正规方程（Normal Equation）**：

$$\boxed{\mathbf{\theta}^* = (X^T X)^{-1} X^T \mathbf{y}}$$

---

## 四、正规方程的代码实现

```python
import numpy as np

# 生成数据：y = 2x + 1 + 噪声
np.random.seed(42)
m = 100
X_raw = np.random.rand(m) * 10
y = 2 * X_raw + 1 + np.random.randn(m)

# 构建设计矩阵（加一列全 1）
X = np.column_stack([np.ones(m), X_raw])  # shape: (100, 2)

# 正规方程
theta = np.linalg.inv(X.T @ X) @ X.T @ y

print(f"θ₀ (截距) = {theta[0]:.4f}，真实值 = 1.0")
print(f"θ₁ (斜率) = {theta[1]:.4f}，真实值 = 2.0")

# 预测
x_new = np.array([1, 7.5])  # 预测面积 7.5（加截距）
y_pred = x_new @ theta
print(f"面积 7.5m²，预测价格：{y_pred:.2f} 万元")
```

---

## 五、正规方程的局限性

| 问题 | 说明 |
|------|------|
| 计算复杂度 | $(X^TX)^{-1}$ 需要 $O(n^3)$ 计算，特征数多时极慢 |
| 内存占用 | 需要把整个 $X$ 矩阵放内存里 |
| 不可逆问题 | $X^TX$ 可能奇异（特征线性相关时），需要用伪逆 |

**结论**：样本少、特征少时用正规方程；特征多（>10000）时用梯度下降。

---

## 六、最小二乘法的几何解释

$$\hat{\mathbf{y}} = X\mathbf{\theta}^* = X(X^TX)^{-1}X^T\mathbf{y}$$

矩阵 $H = X(X^TX)^{-1}X^T$ 叫**帽子矩阵（Hat Matrix）**，也叫投影矩阵。

**几何意义**：$\hat{\mathbf{y}}$ 是 $\mathbf{y}$ 在 $X$ 的列空间上的**正交投影**。

也就是说，线性回归的本质是：**在所有能用特征线性表示的向量里，找一个离真实 $\mathbf{y}$ 最近的**。

```
         y（真实标签）
        /|
       / |
      /  | 残差（垂直）
     /   |
    ŷ----+  ← X 的列空间
```

残差 $\mathbf{y} - \hat{\mathbf{y}}$ 与列空间**正交**，这正是"最小二乘"名字的来源。

---

## 七、多元线性回归

扩展到 $n$ 个特征：

$$h_\theta(\mathbf{x}) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n = \mathbf{\theta}^T \mathbf{x}$$

房价示例（面积 + 卧室数 + 楼层）：

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 模拟多特征数据
np.random.seed(42)
m = 200
area = np.random.uniform(40, 200, m)       # 面积
rooms = np.random.randint(1, 6, m)         # 卧室数
floor = np.random.randint(1, 30, m)        # 楼层

# 真实价格：面积*2 + 卧室数*5 + 楼层*0.5 + 噪声
price = 2*area + 5*rooms + 0.5*floor + np.random.randn(m)*10

X = np.column_stack([area, rooms, floor])
X_train, X_test, y_train, y_test = train_test_split(X, price, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("各特征系数:", model.coef_)    # 应接近 [2, 5, 0.5]
print("截距:", model.intercept_)
print(f"测试集 R²: {model.score(X_test, y_test):.4f}")
```

---

## 八、评估指标详解

| 指标 | 公式 | 说明 |
|------|------|------|
| MSE | $\frac{1}{m}\sum(ŷ-y)^2$ | 对大误差惩罚更重 |
| RMSE | $\sqrt{\text{MSE}}$ | 与 y 同量纲，更直观 |
| MAE | $\frac{1}{m}\sum\|ŷ-y\|$ | 对异常值更鲁棒 |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | 解释方差比例，1 最好，0 最差 |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_true = np.array([3, 5, 4, 6, 8])
y_pred = np.array([2.8, 5.2, 3.9, 6.5, 7.8])

print(f"MSE  = {mean_squared_error(y_true, y_pred):.4f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
print(f"MAE  = {mean_absolute_error(y_true, y_pred):.4f}")
print(f"R²   = {r2_score(y_true, y_pred):.4f}")
```

---

## 九、关键公式汇总

$$\text{假设函数: } h_\theta(\mathbf{x}) = \mathbf{\theta}^T \mathbf{x}$$

$$\text{损失函数: } J(\theta) = \frac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$$

$$\text{正规方程: } \theta^* = (X^TX)^{-1}X^T\mathbf{y}$$

---

## 明天预告

D3：**梯度下降**——当特征太多、正规方程算不动时，怎么用迭代方式找到最优参数。

> 💡 **思考题**：如果两个特征完全相关（比如面积用 m² 和 cm² 同时输入），$X^TX$ 会发生什么？
