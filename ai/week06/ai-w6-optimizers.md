---
layout: default
title: "优化器进阶"
render_with_liquid: false
---

# 优化器进阶

## SGD（随机梯度下降）

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

**问题**：高曲率方向震荡，低曲率方向进展缓慢（峡谷问题）。

## Momentum

$$v_t = \mu v_{t-1} + \eta \nabla_\theta \mathcal{L}$$
$$\theta \leftarrow \theta - v_t$$

动量积累：在一致方向上加速，在震荡方向上减速。$\mu = 0.9$ 典型值。

## Adam（Adaptive Moment Estimation）

综合 Momentum（一阶矩）+ RMSProp（二阶矩）：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{（一阶矩/Momentum）}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{（二阶矩/RMSProp）}$$

偏差修正（前几步 $m_t, v_t$ 接近0）：

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \hat{m}_t$$

默认：$\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$

**直觉**：每个参数有自适应学习率——历史梯度大的参数（$v_t$ 大）学习率小；梯度小的参数学习率大。

## AdamW（Adam + 权重衰减解耦）

Adam 中 L2 正则化被吸收进梯度，与自适应学习率耦合，效果不等价于真正的权重衰减。AdamW 直接对权重衰减：

$$\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \hat{m}_t - \eta\lambda\theta$$

PyTorch 的 `torch.optim.AdamW` 是现代深度学习首选优化器。

## 学习率调度

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 模拟网络
net = nn.Linear(10, 1)
optimizer = optim.Adam(net.parameters(), lr=0.1)

# ---- 常用调度器 ----

# 1. StepLR：每 step_size 个 epoch 乘以 gamma
scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# 2. CosineAnnealingLR：余弦退火
optimizer2 = optim.Adam(net.parameters(), lr=0.1)
scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=100)

# 3. OneCycleLR（当前最流行：先上升再下降）
optimizer3 = optim.Adam(net.parameters(), lr=0.01)
scheduler_onecycle = optim.lr_scheduler.OneCycleLR(
    optimizer3, max_lr=0.1, steps_per_epoch=10, epochs=20
)

# 4. ReduceLROnPlateau（根据验证集loss自适应降低）
optimizer4 = optim.Adam(net.parameters(), lr=0.1)
scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer4, mode='min', factor=0.5, patience=5, verbose=True
)

# ---- 可视化各调度策略 ----
def get_lr_history(scheduler_type, n_steps=100):
    _net = nn.Linear(10, 1)
    _opt = optim.SGD(_net.parameters(), lr=0.1)
    
    if scheduler_type == 'step':
        sch = optim.lr_scheduler.StepLR(_opt, step_size=20, gamma=0.5)
    elif scheduler_type == 'cosine':
        sch = optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=n_steps)
    elif scheduler_type == 'warmup_cosine':
        sch = optim.lr_scheduler.OneCycleLR(_opt, max_lr=0.1,
                                              total_steps=n_steps, pct_start=0.3)
    elif scheduler_type == 'exponential':
        sch = optim.lr_scheduler.ExponentialLR(_opt, gamma=0.95)
    
    lrs = []
    for _ in range(n_steps):
        lrs.append(_opt.param_groups[0]['lr'])
        _opt.step()
        sch.step()
    return lrs

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
types = ['step', 'cosine', 'warmup_cosine', 'exponential']
titles = ['StepLR', 'CosineAnnealingLR', 'OneCycleLR（Warmup+Cosine）', 'ExponentialLR']

for ax, t, title in zip(axes.flatten(), types, titles):
    lrs = get_lr_history(t)
    ax.plot(lrs, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, alpha=0.3)

plt.suptitle('学习率调度策略对比', fontsize=14)
plt.tight_layout()
```

## 优化器对比实验

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=2000, noise=0.15, random_state=42)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

X_t = torch.FloatTensor(X_tr); y_t = torch.FloatTensor(y_tr)
X_te_t = torch.FloatTensor(X_te); y_te_t = torch.FloatTensor(y_te)

def make_mlp():
    return nn.Sequential(
        nn.Linear(2, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 1), nn.Sigmoid()
    )

criterion = nn.BCELoss()
optimizers_config = [
    ('SGD', lambda p: optim.SGD(p, lr=0.01)),
    ('SGD+Momentum', lambda p: optim.SGD(p, lr=0.01, momentum=0.9)),
    ('Adam', lambda p: optim.Adam(p, lr=1e-3)),
    ('AdamW', lambda p: optim.AdamW(p, lr=1e-3, weight_decay=1e-2)),
]

all_losses = {}
for name, opt_fn in optimizers_config:
    net = make_mlp()
    opt = opt_fn(net.parameters())
    losses = []
    
    for epoch in range(300):
        net.train()
        opt.zero_grad()
        loss = criterion(net(X_t).squeeze(), y_t)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    
    net.eval()
    with torch.no_grad():
        acc = ((net(X_te_t).squeeze() > 0.5) == y_te_t.bool()).float().mean().item()
    all_losses[f"{name} ({acc:.3f})"] = losses
    print(f"{name}: 准确率 = {acc:.4f}")

plt.figure(figsize=(10, 5))
for name, losses in all_losses.items():
    plt.plot(losses, alpha=0.8, linewidth=1.5, label=name)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("不同优化器的训练损失")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

## 面试要点

**Q: Adam 为什么需要偏差修正？**

A: 初始化 $m_0 = v_0 = 0$，前几步估计会偏小（因为零初始化的指数加权平均尚未充分"预热"）。偏差修正通过除以 $(1-\beta^t)$ 修正这一系统性低估，在训练初期尤为重要。

**Q: 什么时候用 SGD+Momentum，什么时候用 Adam？**

A: 计算机视觉中 SGD+Momentum（配合 lr schedule）往往能训练出更好的泛化性能（Generalization gap）；NLP/Transformer 中 AdamW 是首选。Adam 收敛快但有时泛化差，可能是因为其自适应学习率使得某些"记忆化"方向学得太快。

**Q: 梯度裁剪（Gradient Clipping）是什么？什么时候用？**

A: 当梯度范数超过阈值时，等比例缩小所有梯度：`torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`。主要用于 RNN/Transformer，防止梯度爆炸（LSTM 序列长时梯度可以指数增大）。
