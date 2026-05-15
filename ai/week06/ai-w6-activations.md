---
layout: default
title: "激活函数详解"
render_with_liquid: false
---

# 激活函数详解

## 为什么需要激活函数？

若所有层都是线性变换，则 $L$ 层网络 = $\mathbf{W}_L \cdots \mathbf{W}_1 \mathbf{x}$ = 单层线性网络，深度无意义。

激活函数引入**非线性**，使深层网络能拟合复杂函数。

## 常用激活函数

### Sigmoid

$$\sigma(z) = \frac{1}{1+e^{-z}}, \quad \sigma'(z) = \sigma(z)(1-\sigma(z))$$

- 输出范围 $(0,1)$，适合输出层（二分类）
- 梯度消失（$\sigma' \leq 0.25$）
- 非零中心（梯度总为正，锯齿更新）

### Tanh

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1, \quad \tanh'(z) = 1 - \tanh^2(z)$$

- 输出范围 $(-1,1)$，零中心
- 梯度消失比 Sigmoid 稍好（最大导数 = 1）
- RNN 中常用

### ReLU

$$\text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) = \mathbf{1}[z > 0]$$

- 梯度不消失（正区间）
- 计算最快
- Dying ReLU：负区间梯度为 0，神经元"永久死亡"

### Leaky ReLU

$$\text{LReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}, \quad \alpha \approx 0.01$$

解决 Dying ReLU，负区间保留小梯度。

### ELU

$$\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}$$

负区间平滑，输出近似零中心，噪声鲁棒性好。

### GELU（GPT/BERT 使用）

$$\text{GELU}(z) = z \cdot \Phi(z) \approx 0.5z\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(z + 0.044715z^3\right)\right]\right)$$

其中 $\Phi$ 是标准正态 CDF。结合了 ReLU 的稀疏性和概率加权，现代 Transformer 首选。

### Swish/SiLU

$$\text{Swish}(z) = z \cdot \sigma(z)$$

无界、平滑、非单调，实验表现常优于 ReLU。

## Python 实现与可视化

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-4, 4, 400)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))

def swish(z):
    return z * sigmoid(z)

activations = {
    'Sigmoid': sigmoid, 'Tanh': tanh, 'ReLU': relu,
    'Leaky ReLU': leaky_relu, 'ELU': elu, 'GELU': gelu, 'Swish': swish
}

# 导数（数值微分）
def numerical_diff(f, z, h=1e-5):
    return (f(z + h) - f(z - h)) / (2 * h)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

colors = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0','#00BCD4','#FF5722']
for idx, ((name, f), color) in enumerate(zip(activations.items(), colors)):
    ax = axes[idx]
    ax.plot(z, f(z), color=color, linewidth=2.5, label='f(z)')
    ax.plot(z, numerical_diff(f, z), '--', color=color, alpha=0.6, linewidth=1.5, label="f'(z)")
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(-2, 3)
    ax.grid(True, alpha=0.3)

axes[-1].axis('off')
plt.suptitle('激活函数及其导数', fontsize=14, fontweight='bold')
plt.tight_layout()

# ===== 激活函数对训练的影响实验 =====
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), activation,
            nn.Linear(64, 64), activation,
            nn.Linear(64, 64), activation,
            nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze()

X_t = torch.FloatTensor(X_tr)
y_t = torch.FloatTensor(y_tr)
X_te_t = torch.FloatTensor(X_te)
y_te_t = torch.FloatTensor(y_te)

result_losses = {}
for act_name, act in [('Sigmoid', nn.Sigmoid()), ('Tanh', nn.Tanh()),
                       ('ReLU', nn.ReLU()), ('GELU', nn.GELU())]:
    net = Net(act)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    losses = []
    for epoch in range(500):
        optimizer.zero_grad()
        out = net(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            losses.append(loss.item())
    
    with torch.no_grad():
        acc = ((net(X_te_t) > 0.5) == y_te_t.bool()).float().mean().item()
    result_losses[act_name] = (losses, acc)
    print(f"{act_name}: 测试准确率 = {acc:.4f}")

plt.figure(figsize=(8, 4))
for act_name, (losses, acc) in result_losses.items():
    plt.plot(range(0, 500, 50), losses, 'o-', label=f"{act_name} (acc={acc:.3f})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("不同激活函数的训练损失")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

## 面试要点

**Q: 为什么 Transformer 用 GELU 而不是 ReLU？**

A: GELU 是平滑的随机正则化：$\text{GELU}(x) = x \cdot P(X \leq x)$，相当于"按激活概率"加权输入，既保留 ReLU 的稀疏性优势，又有平滑梯度，在深层网络中训练更稳定。

**Q: Dying ReLU 问题是什么，如何解决？**

A: 若某个 ReLU 神经元的输入始终为负，梯度为 0，参数永不更新，该神经元"死亡"。解决：① Leaky ReLU（固定小斜率）；② PReLU（可学习斜率）；③ ELU/SELU；④ 谨慎初始化+小学习率。
