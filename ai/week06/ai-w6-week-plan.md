---
layout: default
title: "Week 6 周规划：神经网络基础"
---

# Week 6 · 神经网络基础周规划

## 本周目标

从感知机出发，彻底理解神经网络的数学本质，手推反向传播，掌握训练神经网络的核心技巧。

## 学习路线

| 天 | 主题 | 核心收获 |
|----|------|---------|
| D1 | 感知机与多层网络 | XOR问题、网络表达能力、通用近似定理 |
| D2 | 反向传播推导 | 链式法则、计算图、梯度手推 |
| D3 | 激活函数 | Sigmoid/ReLU/GELU/Swish，梯度消失根源 |
| D4 | 权重初始化与正则化 | Xavier/He初始化，Dropout，BatchNorm |
| D5 | 优化器进阶 | SGD/Momentum/Adam，学习率调度 |
| D6 | 综合实战 | 用 PyTorch 从零搭一个 MLP 分类器 |

## 本周核心公式

反向传播：

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \cdot (a^{(l-1)})^T$$

$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

## 面试必备问题

1. 为什么深层网络比宽网络表达能力更强？
2. 梯度消失/爆炸是如何产生的？如何解决？
3. BatchNorm 为什么有效？放在激活函数前还是后？
4. Dropout 的训练和推理阶段有什么区别？为什么需要缩放？
5. Adam 相比 SGD+Momentum 有什么优缺点？
