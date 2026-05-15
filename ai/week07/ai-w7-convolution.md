---
layout: default
title: "卷积与池化"
render_with_liquid: false
---

# 卷积与池化

## 为什么 CNN 适合图像？

全连接网络对 224×224×3 的图像需要 150,528 个输入参数，第一层就爆炸。更关键的是，它对"猫在左上角"和"猫在右下角"视为完全不同的输入。

CNN 的两个核心归纳偏置解决这个问题：

1. **局部连接**：每个神经元只看输入的局部区域
2. **参数共享**：同一个卷积核在所有位置共享参数（平移等变性）

## 卷积运算

### 互相关（实际 CNN 用的是互相关，不是严格卷积）

对于输入 $I \in \mathbb{R}^{H \times W}$，卷积核 $K \in \mathbb{R}^{k_h \times k_w}$：

$$(I \star K)[i,j] = \sum_{m=0}^{k_h-1}\sum_{n=0}^{k_w-1} I[i+m, j+n] \cdot K[m,n]$$

### 输出尺寸

$$H_{\text{out}} = \left\lfloor\frac{H + 2P - K}{S}\right\rfloor + 1$$

- $P$：padding 大小
- $K$：卷积核大小
- $S$：步长（stride）

**Same Padding**（输出与输入同尺寸）：$P = (K-1)/2$（$S=1$ 时）

### 多通道卷积

输入 $C_{\text{in}}$ 个通道，$C_{\text{out}}$ 个卷积核：

- 每个输出通道：对所有输入通道求加权和
- 参数量：$C_{\text{out}} \times C_{\text{in}} \times K \times K$（而非全连接的 $C_{\text{out}} \times (C_{\text{in}} \times H \times W)$）

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 手写 2D 卷积 =====
def conv2d_manual(input, kernel, stride=1, padding=0):
    """单通道2D卷积（互相关）"""
    if padding > 0:
        input = np.pad(input, padding, mode='constant')
    
    H, W = input.shape
    kH, kW = kernel.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    output = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            patch = input[i*stride:i*stride+kH, j*stride:j*stride+kW]
            output[i, j] = (patch * kernel).sum()
    return output

# ---- 经典图像处理卷积核 ----
image = np.zeros((20, 20))
image[5:15, 5:15] = 1.0  # 简单方块

kernels = {
    '边缘检测（Sobel-X）': np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float),
    '锐化': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float),
    '模糊（高斯）': np.ones((3,3)) / 9.0,
    '恒等映射': np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=float),
}

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('原始图像')
for ax, (name, k) in zip(axes[1:], kernels.items()):
    out = conv2d_manual(image, k, padding=1)
    ax.imshow(out, cmap='gray')
    ax.set_title(name, fontsize=9)
for ax in axes:
    ax.axis('off')
plt.suptitle("卷积核的作用")
plt.tight_layout()

# ===== PyTorch 卷积 =====
# 参数量对比：全连接 vs 卷积
img_h, img_w = 32, 32
in_channels, out_channels = 3, 64
kernel_size = 3

fc_params = out_channels * (in_channels * img_h * img_w)
conv_params = out_channels * in_channels * kernel_size * kernel_size

print(f"全连接层参数: {fc_params:,}")
print(f"卷积层参数:   {conv_params:,}")
print(f"参数减少: {fc_params/conv_params:.0f}x")

# 简单 CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 输入: (B, 3, 32, 32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # → (B,32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # → (B,32,16,16)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # → (B,64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # → (B,64,8,8)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# → (B,128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                      # → (B,128,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = SimpleCNN(num_classes=10)
x_dummy = torch.randn(2, 3, 32, 32)
out = model(x_dummy)
print(f"\nSimpleCNN 输出形状: {out.shape}")

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数: {total_params:,}")

# ===== 感受野计算 =====
def calc_receptive_field(layers):
    """
    layers: list of (kernel_size, stride, padding)
    """
    rf = 1
    total_stride = 1
    for k, s, p in layers:
        rf = rf + (k - 1) * total_stride
        total_stride *= s
    return rf

# 3个 3x3 卷积（stride=1）的感受野
layers = [(3,1,1), (3,1,1), (3,1,1)]
rf = calc_receptive_field(layers)
print(f"\n3 层 3x3 卷积的感受野: {rf}x{rf}")

layers_7 = [(7,2,3), (3,1,1), (3,1,1)]
rf_7 = calc_receptive_field(layers_7)
print(f"1个7x7(stride2) + 2个3x3: 感受野 = {rf_7}x{rf_7}")
```

## 池化层

**MaxPool**：取局部最大值，保留最显著特征，具有一定平移不变性。

**AvgPool**：取平均值，更平滑。

**Global Average Pooling（GAP）**：对每个通道全局平均，输出 $C$ 维向量，替代全连接层，大幅减少参数。

## 1×1 卷积的作用

1. **通道压缩/扩展**（降低/提升维度）
2. **跨通道信息融合**（不改变空间尺寸）
3. **增加非线性**（接激活函数）
4. **替代全连接**（任意输入尺寸）

## 面试要点

**Q: 为什么 3×3 卷积比 7×7 更受欢迎？**

A: 两个 3×3 卷积（stride=1）的感受野 = 5×5，三个 3×3 = 7×7。参数量：2×3²×C² vs 7²×C²，参数减少一半多，且中间有更多非线性（BN+ReLU），表达能力更强。

**Q: 卷积层的参数量和计算量分别是多少？**

A: 设输入 $(C_{in}, H, W)$，输出通道 $C_{out}$，卷积核 $K$：参数量 = $C_{out} \times C_{in} \times K^2 + C_{out}$（bias）；计算量（FLOPs）≈ $2 \times C_{out} \times C_{in} \times K^2 \times H_{out} \times W_{out}$。
