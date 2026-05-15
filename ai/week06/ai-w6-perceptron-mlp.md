---
layout: default
title: "感知机与多层神经网络"
render_with_liquid: false
---

# 感知机与多层神经网络

## 感知机：最简单的神经元

### 数学定义

$$\hat{y} = \text{sign}(\mathbf{w}^T\mathbf{x} + b) = \begin{cases} +1 & \mathbf{w}^T\mathbf{x} + b > 0 \\ -1 & \text{otherwise} \end{cases}$$

### 感知机学习算法

初始化 $\mathbf{w} = 0, b = 0$，对误分类样本 $(x_i, y_i)$：

$$\mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i$$
$$b \leftarrow b + \eta y_i$$

**收敛定理**：若数据线性可分，感知机算法在有限步内收敛。

### XOR 问题：感知机的死穴

```python
import numpy as np
import matplotlib.pyplot as plt

# XOR 数据
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0, 1, 1, 0])  # XOR 输出

# 感知机无法分类（非线性可分）
# 解决：引入隐藏层

# 手动解决 XOR：2层网络
# 层1：AND + OR
# 层2：NAND(AND) & OR = XOR
W1 = np.array([[1, 1], [1, 1]])   # 两个神经元的权重
b1 = np.array([-1.5, -0.5])       # 偏置：AND 和 OR
W2 = np.array([[-1, 1]])           # 输出层权重
b2 = np.array([-0.5])

def relu(x):
    return np.maximum(0, x)

def forward_xor(x):
    h = relu(W1 @ x + b1)
    return (W2 @ h + b2 > 0).astype(int)[0]

print("XOR 手动网络验证:")
for x, y in zip(X_xor, y_xor):
    print(f"  {x} → 预测 {forward_xor(x)}, 真实 {y}")
```

## 多层感知机（MLP）

### 前向传播

第 $l$ 层（$l = 1, \ldots, L$）：

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma(z^{(l)})$$

其中 $a^{(0)} = x$（输入），$a^{(L)} = \hat{y}$（输出）。

### 通用近似定理

> 含有足够多隐藏单元的单隐层神经网络，可以以任意精度近似任意连续函数。

**深度的优势**：深层网络比宽网络参数效率更高——$O(\log n)$ 层可以表示 $O(n)$ 深度的特征层次。

```python
import numpy as np
import matplotlib.pyplot as plt

class MLP:
    """手写 MLP，支持任意层数"""
    
    def __init__(self, layer_sizes, activation='relu'):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.params = {}
        
        # He 初始化（ReLU）
        for l in range(1, len(layer_sizes)):
            fan_in = layer_sizes[l-1]
            self.params[f'W{l}'] = np.random.randn(layer_sizes[l], fan_in) * np.sqrt(2/fan_in)
            self.params[f'b{l}'] = np.zeros(layer_sizes[l])
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_prime(self, z):
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, X):
        self.cache = {'a0': X}
        a = X
        for l in range(1, self.n_layers):
            z = a @ self.params[f'W{l}'].T + self.params[f'b{l}']
            a = self.relu(z)
            self.cache[f'z{l}'] = z
            self.cache[f'a{l}'] = a
        
        # 输出层：sigmoid（二分类）
        z = a @ self.params[f'W{self.n_layers}'].T + self.params[f'b{self.n_layers}']
        a = self.sigmoid(z)
        self.cache[f'z{self.n_layers}'] = z
        self.cache[f'a{self.n_layers}'] = a
        return a
    
    def backward(self, X, y, lr=0.01):
        n = len(y)
        L = self.n_layers
        grads = {}
        
        # 输出层梯度
        dz = self.cache[f'a{L}'] - y.reshape(-1, 1)
        grads[f'dW{L}'] = dz.T @ self.cache[f'a{L-1}'] / n
        grads[f'db{L}'] = dz.mean(axis=0)
        da = dz @ self.params[f'W{L}']
        
        # 隐藏层梯度（反向传播）
        for l in range(L-1, 0, -1):
            dz = da * self.relu_prime(self.cache[f'z{l}'])
            grads[f'dW{l}'] = dz.T @ self.cache[f'a{l-1}'] / n
            grads[f'db{l}'] = dz.mean(axis=0)
            da = dz @ self.params[f'W{l}']
        
        # 参数更新
        for l in range(1, L+1):
            self.params[f'W{l}'] -= lr * grads[f'dW{l}']
            self.params[f'b{l}'] -= lr * grads[f'db{l}']
    
    def train(self, X, y, epochs=1000, lr=0.01, batch_size=32):
        losses = []
        n = len(X)
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            for i in range(0, n, batch_size):
                batch_idx = idx[i:i+batch_size]
                X_b, y_b = X[batch_idx], y[batch_idx]
                self.forward(X_b)
                self.backward(X_b, y_b, lr)
            
            if epoch % 100 == 0:
                y_hat = self.forward(X).flatten()
                loss = -np.mean(y * np.log(y_hat + 1e-9) + (1-y) * np.log(1 - y_hat + 1e-9))
                losses.append(loss)
        return losses

# ---- 测试：XOR ----
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([0,1,1,0], dtype=float)

mlp = MLP([2, 4, 1])
losses = mlp.train(X_xor, y_xor, epochs=5000, lr=0.1, batch_size=4)
y_hat = mlp.forward(X_xor).flatten()
print("XOR MLP 预测:", (y_hat > 0.5).astype(int), "真实:", y_xor.astype(int))

plt.figure(figsize=(6, 3))
plt.plot(losses)
plt.xlabel("Epoch (×100)")
plt.ylabel("Loss")
plt.title("XOR 训练损失曲线")
plt.grid(True)
plt.tight_layout()
```

## 面试要点

**Q: 为什么单层网络无法解决 XOR 问题？**

A: XOR 不是线性可分的——不存在一条直线能分开 XOR 的正负样本。单层感知机的决策边界是线性超平面，必须加入隐藏层引入非线性变换才能处理。

**Q: 通用近似定理意味着可以任意指定层数吗？**

A: 理论上1层足够，但实践中需要指数级宽度。深层网络通过逐层组合特征，用多项式级宽度实现同样表达能力，参数效率更高，泛化性也更好。
