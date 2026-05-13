---
layout: default
title: "D1 · 导数与梯度直觉理解"
---

# D1 · 导数与梯度直觉理解

> **LLM Week 3**  
> 大模型训练的本质是：用梯度告诉每个参数"往哪个方向变，能让损失更小"。

---

## 一、为什么 LLM 需要微积分？

训练 GPT-4 或 LLaMA 的本质是在几百亿个参数的空间里寻找使损失函数最小的点。

寻找方向的工具 = **梯度（Gradient）**

理解梯度 = 理解导数在多维空间的推广。

---

## 二、单变量导数：变化率

导数的直觉：**函数在某点的瞬时变化率**。

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**直觉**：如果 $f'(x) > 0$，$x$ 增大则 $f$ 增大；$f'(x) < 0$，$x$ 增大则 $f$ 减小。

常用导数公式：

| 函数 | 导数 |
|------|------|
| $c$（常数）| 0 |
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln x$ | $\frac{1}{x}$ |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |
| $\tanh(x)$ | $1 - \tanh^2(x)$ |
| $\text{ReLU}(x) = \max(0,x)$ | $0$ 或 $1$（取决于 $x$ 正负） |

---

## 三、多变量偏导数

当函数有多个参数时（神经网络有几亿个参数），对每个参数单独求导：

$$\frac{\partial f}{\partial x_1}$$：把 $x_2, x_3, \ldots$ 视为常数，只对 $x_1$ 求导

**示例**：$f(x_1, x_2) = x_1^2 + 2x_1 x_2 + x_2^3$

$$\frac{\partial f}{\partial x_1} = 2x_1 + 2x_2$$
$$\frac{\partial f}{\partial x_2} = 2x_1 + 3x_2^2$$

---

## 四、梯度：偏导数的向量

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**梯度的物理意义**：

- **方向**：函数上升最快的方向
- **大小**：该方向的上升速率

**梯度下降** = 向梯度的**反方向**移动（因为我们要最小化损失）：

$$\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)$$

---

## 五、数值验证：用差分法验证解析梯度

```python
import numpy as np

def f(x):
    """示例函数：f(x) = x₁² + 3x₁x₂ + x₂²"""
    return x[0]**2 + 3*x[0]*x[1] + x[1]**2

def analytical_gradient(x):
    """解析梯度"""
    return np.array([2*x[0] + 3*x[1], 3*x[0] + 2*x[1]])

def numerical_gradient(f, x, eps=1e-7):
    """数值梯度（差分法）"""
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

x = np.array([1.0, 2.0])
analytical = analytical_gradient(x)
numerical = numerical_gradient(f, x)

print(f"解析梯度: {analytical}")     # [8. 7.]
print(f"数值梯度: {numerical}")      # 应非常接近 [8. 7.]
print(f"误差: {np.abs(analytical - numerical)}")  # 应 < 1e-5
```

> **在深度学习中，数值梯度检验（gradient check）是调试反向传播最重要的工具！**

---

## 六、Jacobian 矩阵：向量函数的"梯度"

当函数输出也是向量时（神经网络的中间层），用 Jacobian 矩阵描述导数：

$$J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix}$$

反向传播的本质 = 一系列 Jacobian 矩阵的乘法（链式法则）。

---

## 七、Sigmoid 导数推导（LLM 中的激活函数）

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)  # 优美的自引用公式

x = np.linspace(-6, 6, 200)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, sigmoid(x), 'b-', linewidth=2, label='σ(x)')
plt.title('Sigmoid 函数')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, sigmoid_derivative(x), 'r-', linewidth=2, label="σ'(x)")
plt.title('Sigmoid 导数（最大值=0.25）')
plt.axvline(x=0, color='gray', linestyle='--')
plt.axhline(y=0.25, color='gray', linestyle='--')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Sigmoid 导数的最大值在 x=0 处，为 0.25
# 这意味着梯度最大只有 0.25，深层网络会遇到梯度消失问题！
print(f"Sigmoid 在 x=0 处的导数: {sigmoid_derivative(0):.4f}")  # 0.25
```

---

## 今天的关键认识

1. **导数 = 变化率**，梯度 = 多维空间里的变化方向
2. **梯度指向函数上升最快的方向**，训练时往反方向走
3. **数值梯度检验**是调试的利器，差分法 = 最可靠的验证手段
4. **Sigmoid 梯度最大只有 0.25**，这是深层网络梯度消失的根源

---

## 明天预告

D2：**链式法则 + 反向传播推导**——真正理解"误差是怎么从输出层一路传回输入层的"。
