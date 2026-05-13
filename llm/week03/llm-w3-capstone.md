---
layout: default
title: "D7 · 综合实战：手写反向传播"
---

# D7 · 综合实战：手写 BP vs PyTorch 验证

> **LLM Week 3 收官**  
> 手写一个完整的神经网络，然后用 PyTorch 的 autograd 验证每一个梯度。

---

## 本周回顾

| 天 | 核心 |
|----|------|
| D1 | 导数与梯度：$\nabla f$ 指向上升最快方向 |
| D2 | 链式法则：BP = 一系列 Jacobian 的乘法 |
| D3 | 概率论：LLM 本质 = $P(\text{下词}|\text{上文})$ |
| D4 | 信息论：交叉熵 = 训练损失，KL = RLHF约束 |
| D5 | MLE：训练 = 找最可能生成数据的参数 |
| D6 | PyTorch autograd：自动微分的威力 |

---

## 实战：手写 vs PyTorch 对比验证

```python
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# Part 1: 纯 NumPy 手写实现
# ============================================================

class ManualNN:
    """手写两层神经网络（ReLU + 交叉熵）"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.lr = lr
        # He 初始化（适合 ReLU）
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0/input_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0/hidden_dim)
        self.b2 = np.zeros((output_dim, 1))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        ez = np.exp(z - z.max(axis=0, keepdims=True))  # 数值稳定
        return ez / ez.sum(axis=0, keepdims=True)
    
    def forward(self, X):
        """X: (input_dim, m)"""
        self.X = X
        self.Z1 = self.W1 @ X + self.b1         # (hidden, m)
        self.A1 = self.relu(self.Z1)              # (hidden, m)
        self.Z2 = self.W2 @ self.A1 + self.b2   # (output, m)
        self.A2 = self.softmax(self.Z2)           # (output, m)
        return self.A2
    
    def cross_entropy_loss(self, Y):
        """Y: one-hot (output_dim, m)"""
        m = Y.shape[1]
        return -np.sum(Y * np.log(self.A2 + 1e-15)) / m
    
    def backward(self, Y):
        m = Y.shape[1]
        
        # 输出层：交叉熵 + Softmax 的梯度合并
        dZ2 = (self.A2 - Y) / m                    # (output, m)
        dW2 = dZ2 @ self.A1.T                       # (output, hidden)
        db2 = dZ2.sum(axis=1, keepdims=True)        # (output, 1)
        
        # 隐藏层：ReLU 梯度（Z1>0 的位置梯度为1，否则为0）
        dA1 = self.W2.T @ dZ2                       # (hidden, m)
        dZ1 = dA1 * (self.Z1 > 0)                  # ReLU 导数
        dW1 = dZ1 @ self.X.T                        # (hidden, input)
        db1 = dZ1.sum(axis=1, keepdims=True)        # (hidden, 1)
        
        # 更新参数
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


# ============================================================
# Part 2: 用 PyTorch 验证梯度（关键！）
# ============================================================

def verify_gradients(X_np, Y_np, W1_np, b1_np, W2_np, b2_np):
    """对比手写梯度和 PyTorch 自动求梯度"""
    
    # 转为 PyTorch Tensor
    X = torch.tensor(X_np.T, dtype=torch.float64)    # (m, input)
    Y = torch.tensor(Y_np.T, dtype=torch.float64)    # (m, output)
    W1 = torch.tensor(W1_np, dtype=torch.float64, requires_grad=True)
    b1 = torch.tensor(b1_np.squeeze(), dtype=torch.float64, requires_grad=True)
    W2 = torch.tensor(W2_np, dtype=torch.float64, requires_grad=True)
    b2 = torch.tensor(b2_np.squeeze(), dtype=torch.float64, requires_grad=True)
    
    # 前向传播
    Z1 = X @ W1.T + b1
    A1 = torch.relu(Z1)
    Z2 = A1 @ W2.T + b2
    A2 = torch.softmax(Z2, dim=1)
    
    # 损失
    loss = -torch.sum(Y * torch.log(A2 + 1e-15)) / X.shape[0]
    
    # 自动求梯度
    loss.backward()
    
    return {
        'dW1': W1.grad.numpy(),
        'db1': b1.grad.numpy().reshape(-1, 1),
        'dW2': W2.grad.numpy(),
        'db2': b2.grad.numpy().reshape(-1, 1)
    }


# ============================================================
# Part 3: 运行对比
# ============================================================

np.random.seed(42)
input_dim, hidden_dim, output_dim, m = 4, 8, 3, 16

X = np.random.randn(input_dim, m)
Y = np.zeros((output_dim, m))
Y[np.random.randint(0, output_dim, m), np.arange(m)] = 1  # one-hot

# 手写网络
net = ManualNN(input_dim, hidden_dim, output_dim)
net.forward(X)
loss_manual = net.cross_entropy_loss(Y)
manual_grads = net.backward(Y)

# PyTorch 验证
pytorch_grads = verify_gradients(X, Y, net.W1, net.b1, net.W2, net.b2)

# 对比（误差应 < 1e-6）
print("梯度对比（手写 vs PyTorch）:")
for key in ['dW1', 'db1', 'dW2', 'db2']:
    diff = np.abs(manual_grads[key] - pytorch_grads[key]).max()
    status = "✅" if diff < 1e-6 else "❌"
    print(f"  {key}: 最大误差 = {diff:.2e} {status}")
```

---

## 完整训练：MNIST 子集分类

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载 sklearn 自带的手写数字数据集（8x8，10类）
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 模型
model = nn.Sequential(
    nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    test_out = model(X_test)
    test_acc = (test_out.argmax(dim=1) == y_test).float().mean()
    print(f"测试准确率: {test_acc.item():.4f}")  # 应 > 0.95
```

---

## Week 3 完成！

🎉 **LLM Week 3 全部完成！** 这周建立了大模型的数学基础：

- ✅ 导数、梯度、Jacobian
- ✅ 链式法则 → 反向传播推导
- ✅ 概率论：贝叶斯 + 条件概率
- ✅ 信息论：熵 / 交叉熵 / KL 散度（理解训练损失的来源）
- ✅ MLE：大模型训练的统计本质
- ✅ PyTorch autograd：工业级自动微分

**Week 4 预告**：深度学习基础 I — 神经网络结构，感知机到 MLP，激活函数全解析
