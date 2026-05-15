---
layout: default
title: "Week 7 周规划：卷积神经网络 CNN"
---

# Week 7 · 卷积神经网络 CNN 周规划

## 本周目标

深入理解 CNN 的数学本质，从手写卷积到 ResNet，掌握迁移学习实战技巧，了解目标检测入门。

## 学习路线

| 天 | 主题 | 核心收获 |
|----|------|---------|
| D1 | 卷积与池化 | 互相关运算、感受野、参数共享 |
| D2 | 经典 CNN 架构 | LeNet→AlexNet→VGG→Inception |
| D3 | ResNet 残差网络 | 跳跃连接、梯度高速公路 |
| D4 | 迁移学习 | 预训练模型、Fine-tuning 策略 |
| D5 | 目标检测入门 | YOLO/Faster-RCNN 核心思想 |
| D6 | 综合实战 | 图像分类完整流程 |

## 核心公式

卷积输出尺寸：

$$O = \left\lfloor\frac{I + 2P - K}{S}\right\rfloor + 1$$

感受野：

$$RF_l = RF_{l-1} + (K_l - 1) \prod_{i=1}^{l-1} S_i$$

## 面试必备问题

1. CNN 为什么适合处理图像？参数共享和局部连接有什么意义？
2. 1×1 卷积有什么作用？
3. ResNet 的残差连接如何解决梯度消失？
4. 迁移学习什么时候 Fine-tune 所有层，什么时候只调最后几层？
5. YOLO 和 Faster-RCNN 的核心区别？
