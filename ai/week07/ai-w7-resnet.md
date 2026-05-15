---
layout: default
title: "ResNet 残差网络"
render_with_liquid: false
---

# ResNet 残差网络

## 深层网络的退化问题

理论上，更深的网络应该表现更好（多余的层学成恒等映射）。但实验表明，直接堆叠卷积层时，56 层网络比 20 层更差——不是过拟合，而是**优化困难**（训练误差都更高）。

根本原因：梯度消失 + 优化景观复杂。

## 残差连接的核心思想

传统：$H(x) = F(x)$（直接学习映射）

ResNet：$H(x) = F(x) + x$（学习残差 $F(x) = H(x) - x$）

**直觉**：学"需要改变什么"比"目标是什么"更容易。若最优映射接近恒等变换，$F(x) \to 0$ 比 $H(x) \to x$ 容易学习。

## 梯度流分析

残差连接提供梯度"高速公路"：

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l} = \frac{\partial L}{\partial x_L} \cdot \left(1 + \sum_{i=l}^{L-1}\frac{\partial F(x_i, W_i)}{\partial x_l}\right)$$

括号中的 **1** 确保梯度直接传回，不会消失。

## ResNet 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """ResNet-18/34 基础残差块"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 当通道数或尺寸变化时，跳跃连接需要投影
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # 残差连接！
        out = F.relu(out)
        return out


class BottleneckBlock(nn.Module):
    """ResNet-50/101/152 瓶颈块（参数更高效）"""
    expansion = 4
    
    def __init__(self, in_channels, mid_channels, stride=1):
        super().__init__()
        out_channels = mid_channels * self.expansion
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """手写 ResNet-18（CIFAR-10 版本，输入 32x32）"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),  # CIFAR: 3x3代替7x7
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# 测试
model = ResNet18(num_classes=10)
x = torch.randn(4, 3, 32, 32)
out = model(x)
print(f"输出形状: {out.shape}")

total = sum(p.numel() for p in model.parameters())
print(f"参数量: {total/1e6:.2f}M")

# ===== 残差 vs 普通网络：梯度流对比 =====
import matplotlib.pyplot as plt
import numpy as np

class PlainNet(nn.Module):
    def __init__(self, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(64, 64), nn.ReLU()])
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class ResNet(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(depth)])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(depth)])
    
    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x)) + x  # 残差连接
        return x

depths = [10, 20, 50, 100]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for depth in depths:
    for ax, (Model, label) in zip(axes, [(PlainNet, 'PlainNet'), (ResNet, 'ResNet')]):
        model = Model(depth)
        x = torch.randn(1, 64, requires_grad=True)
        out = model(x).sum()
        out.backward()
        grad_norm = x.grad.norm().item()
        ax.scatter(depth, grad_norm, s=100)

for ax, label in zip(axes, ['PlainNet（无残差）', 'ResNet（有残差）']):
    ax.set_xlabel('网络深度')
    ax.set_ylabel('输入梯度范数')
    ax.set_title(label)
    ax.grid(True)
    ax.set_yscale('log')

plt.suptitle("残差连接对梯度流的影响")
plt.tight_layout()
```

## ResNet 变体

| 模型 | 深度 | 参数量 | Top-1 Acc (ImageNet) |
|------|------|--------|---------------------|
| ResNet-18 | 18 | 11.7M | 69.8% |
| ResNet-50 | 50 | 25.6M | 76.1% |
| ResNet-101 | 101 | 44.5M | 77.4% |
| ResNet-152 | 152 | 60.2M | 78.3% |
| ResNeXt-50 | 50 | 25.0M | 77.8% |

## 面试要点

**Q: 跳跃连接通道数不匹配时怎么办？**

A: 使用 $1 \times 1$ 卷积（projection shortcut）将输入映射到目标通道数和尺寸。ResNet 原论文提供两种方案：Option A（零填充，无参数）和 Option B（$1\times1$卷积，有少量参数）。

**Q: Pre-activation ResNet 和原始 ResNet 有什么区别？**

A: 原始：Conv→BN→ReLU→Conv→BN→Add→ReLU；Pre-act：BN→ReLU→Conv→BN→ReLU→Conv→Add。Pre-act 让梯度可以不经任何处理直接流过跳跃连接，训练更稳定，常用于很深的网络。
