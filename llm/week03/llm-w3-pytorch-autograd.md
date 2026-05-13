---
layout: default
title: "D6 · PyTorch 自动微分实战"
---

# D6 · PyTorch 自动微分实战

> **LLM Week 3**  
> 理解了反向传播的数学，现在用 PyTorch 的 `autograd` 感受它如何自动完成这一切。

---

## 一、PyTorch Tensor 与 requires_grad

```python
import torch
import torch.nn as nn

# 普通 Tensor（不追踪梯度）
x = torch.tensor([1.0, 2.0, 3.0])
print(x.requires_grad)  # False

# 需要梯度的 Tensor
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x.requires_grad)  # True
```

---

## 二、前向传播与自动求梯度

```python
import torch

x = torch.tensor([2.0], requires_grad=True)

# 前向传播：构建计算图
y = x ** 2 + 3 * x + 1   # f(x) = x² + 3x + 1

# 反向传播：自动计算 dy/dx
y.backward()

print(f"f(2) = {y.item()}")           # 4 + 6 + 1 = 11
print(f"f'(2) = {x.grad.item()}")    # 2x + 3 = 7 ✓
```

---

## 三、计算图可视化

```python
import torch

# 多变量函数
x = torch.tensor([1.0], requires_grad=True)
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)

# z = w * x + b
z = w * x + b
# loss = z²
loss = z ** 2

loss.backward()

print(f"dL/dz = 2z = {2*z.item():.2f}")
print(f"dL/dw = dL/dz * dz/dw = 2z * x = {w.grad.item():.2f}")
print(f"dL/dx = dL/dz * dz/dx = 2z * w = {x.grad.item():.2f}")
print(f"dL/db = dL/dz * dz/db = 2z * 1 = {b.grad.item():.2f}")
```

---

## 四、用 autograd 训练线性回归

```python
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# 数据：y = 2x + 1 + 噪声
torch.manual_seed(42)
X = torch.rand(100, 1) * 10
y = 2 * X + 1 + torch.randn(100, 1) * 1.5

# 参数（需要梯度）
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 优化器
optimizer = optim.SGD([w, b], lr=0.01)

losses = []
for epoch in range(500):
    # 前向传播
    y_pred = X * w + b
    
    # 损失（MSE）
    loss = ((y_pred - y) ** 2).mean()
    
    # 反向传播
    optimizer.zero_grad()  # 清空上一步的梯度
    loss.backward()        # 自动计算梯度
    optimizer.step()       # 更新参数
    
    losses.append(loss.item())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print(f"\n最终参数: w={w.item():.4f}（真实值 2.0）, b={b.item():.4f}（真实值 1.0）")
```

---

## 五、PyTorch nn.Module：搭建神经网络

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# 创建模型
model = SimpleNet(input_dim=8, hidden_dim=64, output_dim=1)
print(f"参数总量: {sum(p.numel() for p in model.parameters()):,}")

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
for epoch in range(100):
    x_batch = torch.randn(32, 8)  # 模拟数据
    y_batch = torch.randn(32, 1)
    
    optimizer.zero_grad()
    output = model(x_batch)
    loss = criterion(output, y_batch)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")
```

---

## 六、梯度裁剪（Gradient Clipping）

大模型训练中常见技巧——防止梯度爆炸：

```python
# 梯度裁剪：将梯度范数限制在某个最大值
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 完整训练循环示例
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    
    # 梯度裁剪（在 backward 之后，step 之前）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

---

## 七、no_grad：推理时关闭梯度

```python
# 推理时不需要计算梯度，可以节省内存和计算
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    # 这里不会建立计算图，内存占用约降低 50%

# 或者只针对单个张量
x = torch.randn(10)
with torch.no_grad():
    y = x ** 2  # 不追踪梯度
```

---

## 今天的关键认识

1. **`requires_grad=True`** → PyTorch 会自动追踪所有操作，建立计算图
2. **`loss.backward()`** → 自动执行反向传播，计算所有参数的梯度
3. **`optimizer.zero_grad()`** → 每步必须清空梯度，否则梯度会累加
4. **`torch.no_grad()`** → 推理时关闭梯度，节省内存

---

## 明天预告

D7：**综合实战**——手写反向传播，再用 PyTorch 验证，彻底理解自动微分。
