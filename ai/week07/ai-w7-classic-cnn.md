---
layout: default
title: "经典 CNN 架构演进"
render_with_liquid: false
---

# 经典 CNN 架构演进

## LeNet-5（1998）—— CNN 的起点

LeCun 设计，用于手写数字识别（MNIST）。

```
输入(32×32) → Conv(6@5×5) → Pool → Conv(16@5×5) → Pool → FC(120) → FC(84) → 输出(10)
```

特点：激活函数用 tanh/sigmoid，平均池化，参数约 60K。

## AlexNet（2012）—— 深度学习的分水岭

ImageNet 竞赛冠军，比第二名低 10% 错误率，震惊整个领域。

关键创新：
- **ReLU**：替代 Sigmoid，缓解梯度消失，训练快 6x
- **Dropout**（0.5）：首次大规模使用
- **数据增强**：随机裁剪、翻转、颜色抖动
- **多 GPU 训练**：模型并行

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),   # 55×55×96
            nn.ReLU(), nn.MaxPool2d(3, 2),     # 27×27×96
            nn.Conv2d(96, 256, 5, padding=2),  # 27×27×256
            nn.ReLU(), nn.MaxPool2d(3, 2),     # 13×13×256
            nn.Conv2d(256, 384, 3, padding=1), # 13×13×384
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1), # 13×13×384
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), # 13×13×256
            nn.ReLU(), nn.MaxPool2d(3, 2),     # 6×6×256
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), nn.Linear(6*6*256, 4096), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

# 参数量
model = AlexNet()
params = sum(p.numel() for p in model.parameters())
print(f"AlexNet 参数量: {params/1e6:.1f}M")
```

## VGG（2014）—— 深度与简洁

核心思想：**用堆叠的 3×3 卷积替代大卷积核**。

两个 3×3 感受野 = 5×5，三个 3×3 = 7×7，但参数更少且有更多非线性。

```
VGG-16 结构（简化）：
[Conv3-64] × 2 → MaxPool
[Conv3-128] × 2 → MaxPool
[Conv3-256] × 3 → MaxPool
[Conv3-512] × 3 → MaxPool
[Conv3-512] × 3 → MaxPool
FC(4096) → FC(4096) → FC(1000)
```

```python
from torchvision import models

# 使用预训练 VGG
vgg16 = models.vgg16(pretrained=False)
print(f"VGG-16 参数量: {sum(p.numel() for p in vgg16.parameters())/1e6:.1f}M")

# VGG 的瓶颈：FC 层参数太多（~120M 中 ~100M 在 FC 层）
# 改进：用 Global Average Pooling 替换 FC
```

## GoogLeNet/Inception（2014）—— 多尺度并行

**Inception 模块**：在同一层并行使用多种卷积核，自动学习最优尺度。

```
输入 → [1×1 Conv] ──────────────────────┐
      → [1×1 Conv → 3×3 Conv] ──────────→ Concat → 输出
      → [1×1 Conv → 5×5 Conv] ──────────┘
      → [3×3 MaxPool → 1×1 Conv] ────────┘
```

1×1 卷积在大卷积前先降维（瓶颈），大幅减少参数。

```python
class InceptionModule(nn.Module):
    def __init__(self, in_c, c_1x1, c_3x3_reduce, c_3x3, c_5x5_reduce, c_5x5, c_pool):
        super().__init__()
        # 分支1：1×1
        self.branch1 = nn.Sequential(nn.Conv2d(in_c, c_1x1, 1), nn.ReLU())
        # 分支2：1×1 → 3×3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_c, c_3x3_reduce, 1), nn.ReLU(),
            nn.Conv2d(c_3x3_reduce, c_3x3, 3, padding=1), nn.ReLU()
        )
        # 分支3：1×1 → 5×5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_c, c_5x5_reduce, 1), nn.ReLU(),
            nn.Conv2d(c_5x5_reduce, c_5x5, 5, padding=2), nn.ReLU()
        )
        # 分支4：MaxPool → 1×1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_c, c_pool, 1), nn.ReLU()
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

import torch
inc = InceptionModule(192, 64, 96, 128, 16, 32, 32)
x = torch.randn(1, 192, 28, 28)
out = inc(x)
print(f"Inception 输出: {out.shape}")  # (1, 256, 28, 28) = 64+128+32+32
```

## 各架构对比

```python
import matplotlib.pyplot as plt
import numpy as np

models_info = {
    'AlexNet': {'year': 2012, 'params': 61, 'top5_err': 15.3, 'layers': 8},
    'VGG-16': {'year': 2014, 'params': 138, 'top5_err': 7.3, 'layers': 16},
    'GoogLeNet': {'year': 2014, 'params': 6.8, 'top5_err': 6.7, 'layers': 22},
    'ResNet-50': {'year': 2015, 'params': 25.6, 'top5_err': 5.25, 'layers': 50},
    'ResNet-152': {'year': 2015, 'params': 60.2, 'top5_err': 4.49, 'layers': 152},
    'EfficientNet-B7': {'year': 2019, 'params': 66, 'top5_err': 2.3, 'layers': 813},
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

names = list(models_info.keys())
params = [models_info[n]['params'] for n in names]
top5 = [models_info[n]['top5_err'] for n in names]
years = [models_info[n]['year'] for n in names]

# 参数 vs 精度气泡图
colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
for (name, p, t5), color in zip(zip(names, params, top5), colors):
    axes[0].scatter(p, t5, s=200, color=color, zorder=5, label=name)

axes[0].set_xlabel('参数量（百万）')
axes[0].set_ylabel('Top-5 错误率 (%)')
axes[0].set_title('ImageNet：参数量 vs 精度')
axes[0].legend(fontsize=8, loc='upper right')
axes[0].grid(True, alpha=0.3)

# 年份 vs 错误率
axes[1].plot(years, top5, 'o-', linewidth=2, markersize=8)
for name, y, t5 in zip(names, years, top5):
    axes[1].annotate(name, (y, t5), textcoords='offset points',
                     xytext=(5, 5), fontsize=8)
axes[1].set_xlabel('年份')
axes[1].set_ylabel('Top-5 错误率 (%)')
axes[1].set_title('ImageNet 精度历年进展')
axes[1].axhline(5.1, color='r', linestyle='--', label='人类水平 (~5.1%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('经典 CNN 架构演进', fontsize=14, fontweight='bold')
plt.tight_layout()
```

## 面试要点

**Q: VGG 为什么用多个 3×3 代替单个 7×7？**

A: 两个 3×3 的感受野等于 5×5，三个等于 7×7。参数量：$3^2 C^2 \times 2 = 18C^2$ vs $7^2 C^2 = 49C^2$，参数少约 63%。同时引入更多 BN+ReLU 层，非线性表达能力更强。

**Q: Inception 模块的 1×1 卷积起什么作用？**

A: 在大卷积前用 1×1 降维（如 192 → 32 通道），减少后续 3×3/5×5 的计算量，同时引入跨通道特征融合，是"瓶颈块"思想的早期实践。
