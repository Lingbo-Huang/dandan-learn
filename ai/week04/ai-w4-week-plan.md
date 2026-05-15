---
layout: default
title: "Week 4 周规划：分类算法"
---

# Week 4 · 分类算法周规划

## 本周目标

掌握机器学习中最核心的一批**监督分类算法**，理解其数学原理与适用场景，能在面试中手推公式、手写代码。

## 学习路线

| 天 | 主题 | 核心收获 |
|----|------|---------|
| D1 | 朴素贝叶斯 | 贝叶斯定理 → 生成式分类器直觉 |
| D2 | 逻辑回归 | Sigmoid + 交叉熵 → 判别式分类器 |
| D3 | 支持向量机 SVM | 最大间隔 + 核技巧 |
| D4 | 决策树 | 信息增益 / Gini → 树的构建与剪枝 |
| D5 | 随机森林 | Bagging + 特征随机 → 集成思想入门 |
| D6 | 综合实战 | 五大算法在真实数据集上的对比 |

## 知识图谱

```
分类算法
├── 生成式模型
│   └── 朴素贝叶斯（P(x|y) × P(y)）
├── 判别式模型
│   ├── 逻辑回归（线性决策边界）
│   └── SVM（最大间隔决策边界）
└── 基于树的模型
    ├── 决策树（单棵）
    └── 随机森林（集成）
```

## 面试必备问题

1. 逻辑回归和朴素贝叶斯有什么区别？各自适用什么场景？
2. SVM 的支持向量是什么？软间隔 C 参数怎么调？
3. 决策树如何处理过拟合？随机森林相比决策树强在哪？
4. 为什么朴素贝叶斯"朴素"？现实中条件独立假设成立吗？
5. 多分类问题 SVM 怎么扩展（OvO / OvR）？

## 本周代码环境

```python
# 本周统一使用的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
```

## 学习建议

- 每天先看理论，再手推公式，最后写代码验证
- 重点理解每个算法的"决策边界"形状——这是面试常考点
- 随机森林是 Week5 集成学习的引子，本周重点理解其 Bagging 思想
- 用同一个数据集（乳腺癌数据集）跑所有算法，直观感受差异
