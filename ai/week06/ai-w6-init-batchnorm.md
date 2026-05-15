---
layout: default
title: "权重初始化与 BatchNorm"
render_with_liquid: false
---

# 权重初始化与 BatchNorm

## 为什么初始化重要？

**全零初始化**：所有神经元完全对称，梯度完全相同，无论训练多少轮，所有神经元永远相同。网络退化为单个神经元！

**太大**：激活值饱和 → 梯度消失

**太小**：激活值趋零 → 信号消失

## 方差分析推导

设每层输入 $x$ 有方差 $\text{Var}(x)$，线性层 $z = Wx$，$W$ 的每个元素 i.i.d.：

$$\text{Var}(z_j) = n_{\text{in}} \cdot \text{Var}(W_{ij}) \cdot \text{Var}(x_i)$$

为保持每层方差不变（信号不爆炸不消失），需要：

$$\text{Var}(W) = \frac{1}{n_{\text{in}}}$$

## Xavier/Glorot 初始化（Sigmoid/Tanh）

考虑前向和反向传播都要保持方差：

$$\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

均匀分布形式：$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right)$

## He 初始化（ReLU）

ReLU 约有一半神经元输出为 0，方差减半，需要补偿：

$$\text{Var}(W) = \frac{2}{n_{\text{in}}}$$

正态分布：$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ===== 初始化方差实验 =====
def simulate_init(init_std, n_layers=10, n_units=256, n_samples=1000):
    """模拟不同初始化下激活值的方差变化"""
    activations_var = []
    x = np.random.randn(n_samples, n_units)
    
    for l in range(n_layers):
        W = np.random.randn(n_units, n_units) * init_std
        x = np.maximum(0, x @ W)  # ReLU
        activations_var.append(x.var())
    
    return activations_var

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
n_layers = 10

# 初始化过小
small_vars = simulate_init(0.01)
axes[0].plot(small_vars)
axes[0].set_title("初始化过小（σ=0.01）\n激活值趋零")
axes[0].set_ylabel("方差")
axes[0].set_yscale('log')

# 初始化过大
large_vars = simulate_init(1.0)
axes[1].plot(large_vars)
axes[1].set_title("初始化过大（σ=1.0）\n激活值爆炸")
axes[1].set_yscale('log')

# He 初始化（正确）
he_vars = simulate_init(np.sqrt(2/256))
axes[2].plot(he_vars)
axes[2].set_title("He 初始化（σ=√(2/n)）\n方差稳定")
axes[2].set_yscale('log')

for ax in axes:
    ax.set_xlabel("层数")
    ax.grid(True)

plt.suptitle("权重初始化对激活值方差的影响（ReLU网络，10层）")
plt.tight_layout()

# PyTorch 初始化方式
net = nn.Linear(256, 256)
nn.init.xavier_uniform_(net.weight)   # Xavier（Sigmoid/Tanh）
nn.init.kaiming_normal_(net.weight)   # He（ReLU）
nn.init.zeros_(net.bias)
```

## Batch Normalization

### 直觉

每层输出的分布随着训练变化（Internal Covariate Shift），下一层需要不断适应。BN 把每层输出标准化为零均值单位方差，再用可学习参数 $\gamma, \beta$ 重新缩放。

### 数学推导

对 mini-batch $\{x_1, \ldots, x_m\}$：

$$\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m x_i$$

$$\sigma^2_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_\mathcal{B})^2$$

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta \quad \leftarrow \text{可学习参数}$$

### 训练 vs 推理

- **训练**：用 mini-batch 统计量
- **推理**：用训练集的移动平均统计量（固定，保证输出确定性）

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class BatchNorm1DManual(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        self.momentum = momentum
        # 推理时使用的移动平均
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            # 更新移动平均
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

# 实验：BN 对训练稳定性的影响
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

X_t = torch.FloatTensor(X_tr); y_t = torch.FloatTensor(y_tr)
X_te_t = torch.FloatTensor(X_te); y_te_t = torch.FloatTensor(y_te)

def make_net(use_bn):
    layers = []
    sizes = [2, 128, 128, 128, 1]
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            if use_bn:
                layers.append(nn.BatchNorm1d(sizes[i+1]))
            layers.append(nn.ReLU())
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

results = {}
for use_bn in [False, True]:
    net = make_net(use_bn)
    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    criterion = nn.BCELoss()
    
    losses = []
    for epoch in range(300):
        net.train()
        optimizer.zero_grad()
        loss = criterion(net(X_t).squeeze(), y_t)
        loss.backward()
        optimizer.step()
        if epoch % 30 == 0:
            losses.append(loss.item())
    
    net.eval()
    with torch.no_grad():
        acc = ((net(X_te_t).squeeze() > 0.5) == y_te_t.bool()).float().mean().item()
    results[f'{"有" if use_bn else "无"}BatchNorm'] = (losses, acc)
    print(f"{'有' if use_bn else '无'} BatchNorm: 准确率 = {acc:.4f}")

plt.figure(figsize=(8, 4))
for name, (losses, acc) in results.items():
    plt.plot(range(0, 300, 30), losses, 'o-', label=f"{name} (acc={acc:.3f})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BatchNorm 对训练的影响")
plt.legend()
plt.grid(True)
plt.tight_layout()
```

## Dropout

训练时以概率 $p$ 随机将神经元输出置零；测试时输出乘以 $(1-p)$（等价地，训练时对保留的神经元除以 $(1-p)$，即 **Inverted Dropout**）。

```python
class DropoutManual(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training:
            return x  # 测试时不 dropout
        # Inverted dropout: 保留并缩放
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1 - self.p)
```

**为什么有效**：迫使网络学习冗余表示，每个神经元不能依赖特定神经元，类似集成多个子网络。

## 面试要点

**Q: BN 放在激活函数前还是后？**

A: 原始论文在激活前（Conv→BN→ReLU），但实践中激活后（Conv→ReLU→BN）有时效果更好。现代网络（ResNet、Transformer）通常用 Pre-Norm（BN/LN 在残差块前），训练更稳定。

**Q: BatchNorm 在训练和推理时为什么用不同统计量？**

A: 训练时 batch 统计量足够好且有正则化效果（类似 Dropout）；推理时 batch size 可能为 1（无法计算统计量），且需要确定性输出，所以用训练集的移动平均。

**Q: LayerNorm 和 BatchNorm 的区别？**

A: BN 对 batch 维度归一化（每个特征独立）；LN 对特征维度归一化（每个样本独立）。LN 不受 batch size 影响，适合 RNN/Transformer（序列长度可变，小batch）。
