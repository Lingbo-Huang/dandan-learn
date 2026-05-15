---
layout: default
title: "反向传播算法推导"
render_with_liquid: false
---

# 反向传播算法推导

## 直觉：误差的逆向流动

前向传播计算输出；反向传播把"预测错了多少"从输出层逐层传回，告诉每个参数"你该往哪个方向调整"。

核心工具：**链式法则**。

## 链式法则回顾

$$\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}$$

对于 $L = f(g(x))$，梯度从 $L$ 流向 $g$，再从 $g$ 流向 $x$。

## 计算图视角

以两层网络为例：

$$x \xrightarrow{W^{(1)}} z^{(1)} \xrightarrow{\sigma} a^{(1)} \xrightarrow{W^{(2)}} z^{(2)} \xrightarrow{\sigma} \hat{y} \xrightarrow{L}$$

前向：从左到右计算每个节点的值。

反向：从右到左，每个节点计算"我对损失的贡献"。

## 完整推导（以二分类交叉熵为例）

### 符号定义

- $L$ 层网络，第 $l$ 层有 $n_l$ 个神经元
- $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$，$b^{(l)} \in \mathbb{R}^{n_l}$
- $z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$
- $a^{(l)} = \sigma(z^{(l)})$，$a^{(0)} = x$

### 定义误差项

$$\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$$

### 输出层（$l = L$）

$$\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \sigma'(z^{(L)})$$

对交叉熵 + Sigmoid 输出层：$\delta^{(L)} = a^{(L)} - y$（非常简洁！）

### 隐藏层反向传播

$$\delta^{(l)} = \left[(W^{(l+1)})^T \delta^{(l+1)}\right] \odot \sigma'(z^{(l)})$$

直觉：上层传来的梯度 $\delta^{(l+1)}$，通过权重矩阵转置分配给本层，再乘以本层激活函数导数。

### 参数梯度

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$

$$\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

（批量时：对 batch 维度求和/平均）

## 关键节点梯度推导

```python
import numpy as np

# 单个样本，手动计算反向传播

# 网络: x(2) -> h(3) -> y(1)
np.random.seed(42)
W1 = np.random.randn(3, 2) * 0.1  # 3×2
b1 = np.zeros(3)
W2 = np.random.randn(1, 3) * 0.1  # 1×3
b2 = np.zeros(1)

x = np.array([1.0, 0.5])
y_true = np.array([1.0])

# ===== 前向传播 =====
z1 = W1 @ x + b1                         # (3,)
a1 = np.maximum(0, z1)                    # ReLU
z2 = W2 @ a1 + b2                         # (1,)
a2 = 1 / (1 + np.exp(-z2))               # Sigmoid
y_hat = a2                                 # (1,)

# 损失
eps = 1e-9
L = -np.mean(y_true * np.log(y_hat + eps) + (1 - y_true) * np.log(1 - y_hat + eps))
print(f"损失: {L:.6f}")

# ===== 反向传播 =====
# 输出层：交叉熵 + Sigmoid 合并导数
dz2 = y_hat - y_true                       # (1,) — 预测误差
dW2 = np.outer(dz2, a1)                   # (1,3)
db2 = dz2.copy()                           # (1,)

# 隐藏层：ReLU 导数
da1 = W2.T @ dz2                           # (3,)
dz1 = da1 * (z1 > 0).astype(float)        # (3,) — ReLU梯度
dW1 = np.outer(dz1, x)                    # (3,2)
db1 = dz1.copy()                           # (3,)

print(f"dW1 shape: {dW1.shape}, dW2 shape: {dW2.shape}")

# ===== 数值梯度验证 =====
def numerical_gradient(f, params, h=1e-5):
    grad = {}
    for key, val in params.items():
        g = np.zeros_like(val)
        it = np.nditer(val, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            original = val[idx]
            val[idx] = original + h
            f_plus = f()
            val[idx] = original - h
            f_minus = f()
            g[idx] = (f_plus - f_minus) / (2 * h)
            val[idx] = original
            it.iternext()
        grad[key] = g
    return grad

params = {'W1': W1.copy(), 'b1': b1.copy(), 'W2': W2.copy(), 'b2': b2.copy()}

def loss_fn():
    z1_ = params['W1'] @ x + params['b1']
    a1_ = np.maximum(0, z1_)
    z2_ = params['W2'] @ a1_ + params['b2']
    a2_ = 1 / (1 + np.exp(-z2_))
    return float(-np.mean(y_true * np.log(a2_+eps) + (1-y_true)*np.log(1-a2_+eps)))

num_grads = numerical_gradient(loss_fn, params)
print("\n数值梯度 vs 解析梯度（应接近0）：")
print(f"  W1 最大误差: {np.max(np.abs(dW1 - num_grads['W1'])):.2e}")
print(f"  W2 最大误差: {np.max(np.abs(dW2 - num_grads['W2'])):.2e}")
print(f"  b1 最大误差: {np.max(np.abs(db1 - num_grads['b1'])):.2e}")
```

## 梯度消失问题

Sigmoid 导数最大值为 0.25，多层累乘后梯度指数衰减：

$$\frac{\partial L}{\partial W^{(1)}} = \prod_{l=2}^{L} \sigma'(z^{(l)}) \cdot \text{(其他项)}$$

若 $\sigma'(z) < 0.25$，100 层后梯度 $< 0.25^{100} \approx 10^{-60}$，完全消失！

```python
import matplotlib.pyplot as plt

def sigmoid_derivative(z):
    s = 1 / (1 + np.exp(-z))
    return s * (1 - s)

def relu_derivative(z):
    return (z > 0).astype(float)

z = np.linspace(-5, 5, 200)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(z, sigmoid_derivative(z), label="σ'(z)（max=0.25）")
plt.plot(z, relu_derivative(z), label="ReLU'(z)（0或1）")
plt.title("激活函数导数")
plt.legend()
plt.grid(True)

# 梯度消失示意：100层后的梯度
n_layers = np.arange(1, 100)
sigmoid_grad = 0.25 ** n_layers
plt.subplot(1, 2, 2)
plt.semilogy(n_layers, sigmoid_grad)
plt.xlabel("层数")
plt.ylabel("梯度量级（log scale）")
plt.title("Sigmoid：100层后梯度量级")
plt.grid(True)
plt.tight_layout()
```

## 面试要点

**Q: 反向传播的计算复杂度是多少？**

A: 与前向传播相同量级（$O(W)$，$W$ 为参数总数）。每个参数的梯度可以在一次反向传播中同时计算出来，这是 BP 的核心优势。

**Q: 为什么 ReLU 能缓解梯度消失？**

A: ReLU 导数为 0 或 1——激活时梯度原样传回，不会衰减。而 Sigmoid 导数始终 ≤ 0.25，多层后梯度指数衰减。代价是 Dying ReLU 问题（负区间永远死亡），用 Leaky ReLU/ELU 可缓解。
