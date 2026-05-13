---
layout: default
title: "D2 · 链式法则 + 反向传播推导"
---

# D2 · 链式法则与反向传播

> **LLM Week 3**  
> 反向传播是深度学习的引擎。今天从链式法则出发，手推一遍 BP。

---

## 一、链式法则

如果 $y = f(g(x))$，则：

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

**神经网络是复合函数的极端情况**：

$$L = \text{loss}(\sigma(W_n \cdot \sigma(W_{n-1} \cdots \sigma(W_1 \mathbf{x}) \cdots)))$$

每一层都是一个函数套在前一层上。

---

## 二、计算图（Computation Graph）

将计算过程表示为有向无环图，每个节点是一个操作：

```
x ─→ [× W] ─→ z ─→ [σ] ─→ a ─→ [×W₂] ─→ z₂ ─→ [Softmax] ─→ L
```

**前向传播（Forward Pass）**：从左到右计算每个节点的值  
**反向传播（Backward Pass）**：从右到左，用链式法则计算梯度

---

## 三、一个最小的神经网络

结构：输入层 → 隐藏层 → 输出层（二分类）

```
输入 x ∈ ℝⁿ
↓
z₁ = W₁x + b₁       # 线性变换
a₁ = σ(z₁)           # 非线性激活
↓  
z₂ = W₂a₁ + b₂      # 输出层线性
ŷ = σ(z₂)            # 输出概率
↓
L = -[y log(ŷ) + (1-y)log(1-ŷ)]  # 二元交叉熵损失
```

---

## 四、反向传播推导

**第一步：输出层梯度**

$$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

由于 $\hat{y} = \sigma(z_2)$，且 $\sigma'(z) = \sigma(z)(1-\sigma(z))$：

$$\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} = \hat{y} - y$$

> 🌟 **神奇**：交叉熵 + Sigmoid 的梯度是 $\hat{y} - y$，形式极其简洁！

**第二步：输出层权重梯度**

$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial W_2} = (\hat{y} - y) \cdot a_1^T$$

$$\frac{\partial L}{\partial b_2} = \hat{y} - y$$

**第三步：传播到隐藏层**

$$\frac{\partial L}{\partial a_1} = W_2^T \cdot \frac{\partial L}{\partial z_2} = W_2^T(\hat{y} - y)$$

$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \odot \sigma'(z_1) = \frac{\partial L}{\partial a_1} \odot a_1(1-a_1)$$

（$\odot$ 表示逐元素乘法）

**第四步：第一层权重梯度**

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \cdot x^T$$

$$\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1}$$

---

## 五、从零实现反向传播

```python
import numpy as np

class TinyNeuralNet:
    def __init__(self, n_input, n_hidden, n_output, lr=0.1):
        self.lr = lr
        # 初始化权重（小随机数）
        self.W1 = np.random.randn(n_hidden, n_input) * 0.01
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = np.random.randn(n_output, n_hidden) * 0.01
        self.b2 = np.zeros((n_output, 1))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, X):
        """前向传播，缓存中间值用于反向传播"""
        self.X = X
        self.z1 = self.W1 @ X + self.b1       # (n_hidden, m)
        self.a1 = self.sigmoid(self.z1)         # (n_hidden, m)
        self.z2 = self.W2 @ self.a1 + self.b2  # (n_output, m)
        self.a2 = self.sigmoid(self.z2)         # (n_output, m)
        return self.a2
    
    def compute_loss(self, y):
        """二元交叉熵"""
        m = y.shape[1]
        eps = 1e-15
        return -np.mean(y * np.log(self.a2 + eps) + (1-y) * np.log(1-self.a2 + eps))
    
    def backward(self, y):
        """反向传播，计算所有梯度"""
        m = y.shape[1]
        
        # 输出层
        dz2 = self.a2 - y                           # (n_output, m)
        dW2 = dz2 @ self.a1.T / m                  # (n_output, n_hidden)
        db2 = np.mean(dz2, axis=1, keepdims=True)  # (n_output, 1)
        
        # 隐藏层
        da1 = self.W2.T @ dz2                       # (n_hidden, m)
        dz1 = da1 * self.a1 * (1 - self.a1)        # 激活函数梯度
        dW1 = dz1 @ self.X.T / m                   # (n_hidden, n_input)
        db1 = np.mean(dz1, axis=1, keepdims=True)  # (n_hidden, 1)
        
        # 参数更新
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, epochs=1000, print_every=200):
        losses = []
        for epoch in range(epochs):
            pred = self.forward(X)
            loss = self.compute_loss(y)
            self.backward(y)
            
            if epoch % print_every == 0:
                accuracy = np.mean((pred > 0.5) == y)
                print(f"Epoch {epoch:4d}: Loss={loss:.4f}, Acc={accuracy:.3f}")
            losses.append(loss)
        return losses

# 测试：XOR 问题（线性模型无法解决，需要隐藏层）
X = np.array([[0,0,1,1], [0,1,0,1]])    # (2, 4)
y = np.array([[0,1,1,0]])               # XOR 输出 (1, 4)

net = TinyNeuralNet(n_input=2, n_hidden=4, n_output=1, lr=1.0)
losses = net.train(X, y, epochs=5000, print_every=1000)

pred = net.forward(X)
print(f"\nXOR 预测结果: {pred.round(3)}")
print(f"真实结果:     {y}")
```

---

## 六、梯度消失问题

```python
import numpy as np
import matplotlib.pyplot as plt

# 模拟深层网络梯度传播
def simulate_gradient_flow(n_layers, activation='sigmoid'):
    gradient = 1.0
    gradients = [gradient]
    
    for _ in range(n_layers):
        if activation == 'sigmoid':
            # Sigmoid 导数最大 0.25，随机权重乘以约 1
            gradient *= 0.25 * np.random.uniform(0.8, 1.2)
        elif activation == 'relu':
            # ReLU 导数是 0 或 1，梯度不被压缩
            gradient *= np.random.uniform(0.8, 1.2)  # 主要是权重的影响
        gradients.append(gradient)
    
    return gradients

n_layers = 20
sigmoid_grads = simulate_gradient_flow(n_layers, 'sigmoid')
relu_grads = simulate_gradient_flow(n_layers, 'relu')

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.semilogy(sigmoid_grads, 'b-o', markersize=4)
plt.xlabel('层数')
plt.ylabel('梯度大小（log）')
plt.title('Sigmoid：梯度消失')

plt.subplot(1, 2, 2)
plt.plot(relu_grads, 'r-o', markersize=4)
plt.xlabel('层数')
plt.ylabel('梯度大小')
plt.title('ReLU：梯度流动更好')
plt.tight_layout()
plt.show()
```

---

## 今天的关键认识

1. **链式法则 = 反向传播的数学基础**，梯度从损失层一路传回输入层
2. **交叉熵 + Sigmoid 的梯度 = $\hat{y} - y$**，简洁优雅
3. **Sigmoid 导数最大 0.25**，深层网络梯度消失的根本原因
4. **ReLU 的梯度是 0 或 1**，是现代网络的默认激活函数

---

## 明天预告

D3：**概率论基础**——贝叶斯定理、条件概率、大模型的概率本质。
