---
layout: default
title: "D3 · 梯度下降算法详解"
---

# D3 · 梯度下降算法详解

> **Week 3 · AI 基础线**  
> 正规方程在特征多时行不通。今天学机器学习最核心的优化工具：梯度下降。

---

## 一、直觉：下山

想象你站在一座山上，蒙着眼睛，目标是走到最低点（损失最小）。

策略：**每一步都往脚下最陡的方向走一小步。**

```
损失 J
  │    ╭──╮
  │   ╱    ╲
  │  ╱      ╲
  │ ╱        ╲    ← 当前位置
  │╱           ───── 最低点
  └──────────────── θ
```

**梯度**就是"脚下最陡方向"；**学习率**就是"每步走多远"。

---

## 二、梯度下降公式

$$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

- $\alpha$：学习率（步长）
- $\frac{\partial J}{\partial \theta_j}$：损失函数对 $\theta_j$ 的偏导数

**对线性回归，求偏导数**：

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$$

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

更新规则（所有 $\theta_j$ **同时更新**）：

$$\theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

---

## 三、从零实现批量梯度下降

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(42)
m = 100
X_raw = np.random.rand(m) * 10
y = 2 * X_raw + 1 + np.random.randn(m) * 1.5

# 添加截距项
X = np.column_stack([np.ones(m), X_raw])  # (100, 2)

def compute_loss(X, y, theta):
    m = len(y)
    pred = X @ theta
    return np.mean((pred - y) ** 2) / 2

def gradient_descent(X, y, learning_rate=0.01, n_iters=1000):
    m, n = X.shape
    theta = np.zeros(n)          # 初始化参数为 0
    losses = []
    
    for i in range(n_iters):
        pred = X @ theta
        error = pred - y
        gradient = X.T @ error / m
        theta -= learning_rate * gradient
        
        if i % 100 == 0:
            loss = compute_loss(X, y, theta)
            losses.append(loss)
            print(f"Step {i:4d}: Loss={loss:.4f}, θ={theta}")
    
    return theta, losses

theta_opt, losses = gradient_descent(X, y, learning_rate=0.01, n_iters=2000)
print(f"\n最终参数: θ₀={theta_opt[0]:.4f}, θ₁={theta_opt[1]:.4f}")
print(f"真实参数: θ₀=1.0, θ₁=2.0")

# 可视化损失下降
plt.plot(losses)
plt.xlabel('迭代次数（每100步）')
plt.ylabel('Loss')
plt.title('梯度下降：损失下降曲线')
plt.show()
```

---

## 四、三种梯度下降变体

| 类型 | 每次用多少数据 | 优点 | 缺点 |
|------|--------------|------|------|
| **批量梯度下降 (BGD)** | 全部 $m$ 个样本 | 收敛稳定 | 数据大时每步极慢 |
| **随机梯度下降 (SGD)** | 1 个样本 | 每步快，可在线学习 | 震荡大，不稳定 |
| **小批量梯度下降 (Mini-batch GD)** | $b$ 个样本（通常 32-256） | 兼顾速度和稳定性 | 需要选择 batch size |

```python
def mini_batch_gd(X, y, learning_rate=0.01, n_epochs=50, batch_size=32):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    
    for epoch in range(n_epochs):
        # 打乱顺序
        indices = np.random.permutation(m)
        X_shuffled, y_shuffled = X[indices], y[indices]
        
        epoch_loss = 0
        for start in range(0, m, batch_size):
            X_batch = X_shuffled[start:start+batch_size]
            y_batch = y_shuffled[start:start+batch_size]
            
            pred = X_batch @ theta
            error = pred - y_batch
            gradient = X_batch.T @ error / len(y_batch)
            theta -= learning_rate * gradient
            epoch_loss += compute_loss(X_batch, y_batch, theta)
        
        losses.append(epoch_loss)
    
    return theta, losses
```

---

## 五、学习率的影响

```python
# 演示不同学习率的效果
learning_rates = [0.001, 0.01, 0.1, 0.5]

plt.figure(figsize=(12, 3))
for i, lr in enumerate(learning_rates):
    theta, losses = gradient_descent(X, y, learning_rate=lr, n_iters=500)
    plt.subplot(1, 4, i+1)
    plt.plot(losses)
    plt.title(f'lr={lr}')
    plt.xlabel('迭代次数')
    if i == 0:
        plt.ylabel('Loss')

plt.tight_layout()
plt.show()
```

**经验规则**：
- 学习率太小 → 收敛太慢，需要更多迭代
- 学习率太大 → 震荡，甚至发散（损失越来越大）
- 通常从 0.01 开始试，根据损失曲线调整

---

## 六、特征归一化：让梯度下降更快

如果不同特征的量级差很大（面积 100m²，卧室数 3），梯度下降会很慢（等高线变成椭圆）。

**解决方案：归一化（Standardization / Z-score normalization）**

$$x_j' = \frac{x_j - \mu_j}{\sigma_j}$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw.reshape(-1, 1))

# 重要：只能用训练集统计量，再应用到测试集！
X_train_scaled = scaler.fit_transform(X_train)  # 训练集：fit + transform
X_test_scaled = scaler.transform(X_test)         # 测试集：只 transform，不 fit
```

**归一化前后对比**：
- 未归一化：等高线是细长椭圆，梯度方向和最优方向偏差大，需要很多步
- 归一化后：等高线接近圆形，梯度方向直指最优点，收敛快很多

---

## 七、收敛判断

怎么知道什么时候停止迭代？

```python
def gradient_descent_with_early_stop(X, y, lr=0.01, tol=1e-6, max_iters=10000):
    m, n = X.shape
    theta = np.zeros(n)
    prev_loss = float('inf')
    
    for i in range(max_iters):
        pred = X @ theta
        error = pred - y
        gradient = X.T @ error / m
        theta -= lr * gradient
        
        loss = compute_loss(X, y, theta)
        
        # 损失变化小于阈值则停止
        if abs(prev_loss - loss) < tol:
            print(f"在第 {i} 步收敛")
            break
        prev_loss = loss
    
    return theta
```

---

## 八、今天的关键点

1. **梯度 = 方向**：损失函数上升最快的方向，我们往反方向走
2. **学习率 = 步长**：太大振荡，太小收敛慢，通常 0.001~0.1
3. **Mini-batch 是实践中的默认选择**：不用全量数据，也不用单条样本
4. **归一化很重要**：让梯度下降更稳定、更快

---

## 明天预告

D4：**多项式回归 & 过拟合**——线性不够用时怎么加特征？加多了又会发生什么？

> 💡 **思考题**：梯度下降中，为什么要"同时更新"所有参数，而不是依次更新？
