---
layout: default
title: "Week 3 周规划 · 监督学习：回归"
---

# Week 3 学习计划：监督学习 · 回归

> **系列**：AI 基础线  
> **周次**：Week 3（共 52 周）  
> **前置**：Week 1 向量/矩阵基础，Week 2 特征值/SVD/PCA

---

## 本周主题与目标

从"数学工具"正式进入"机器学习"。  
这周的核心问题：**给你一堆数据，怎么让机器自己找规律、做预测？**

线性回归是所有机器学习模型的起点——简单、可解释、有完整数学推导，是理解梯度下降、过拟合、正则化的最佳载体。

| 天 | 文件 | 主题 | 核心概念 |
|----|------|------|---------|
| D1 | `ai-w3-ml-intro.md` | 什么是机器学习？ | 监督/无监督/强化，训练/测试/泛化 |
| D2 | `ai-w3-linear-regression.md` | 线性回归原理 | 假设函数、损失函数、最小二乘法 |
| D3 | `ai-w3-gradient-descent.md` | 梯度下降 | 批量/随机/小批量、学习率、收敛 |
| D4 | `ai-w3-polynomial.md` | 多项式回归 & 过拟合 | 特征工程、偏差-方差权衡 |
| D5 | `ai-w3-regularization.md` | 正则化 | Ridge (L2)、Lasso (L1)、ElasticNet |
| D6 | `ai-w3-sklearn-regression.md` | sklearn 实战 | 波士顿房价/加州房价预测，完整流程 |
| D7 | `ai-w3-capstone.md` | 综合实战 | 多变量回归项目：从数据到部署 |

---

## 知识图谱

```
机器学习
 ├── 监督学习
 │    ├── 回归（本周重点）
 │    │    ├── 线性回归
 │    │    │    ├── 损失函数：MSE
 │    │    │    ├── 解析解：正规方程
 │    │    │    └── 数值解：梯度下降
 │    │    ├── 多项式回归
 │    │    └── 正则化回归
 │    │         ├── Ridge（L2）
 │    │         └── Lasso（L1）
 │    └── 分类（下周）
 ├── 无监督学习（Week 7）
 └── 强化学习（Week 12）
```

---

## 为什么从线性回归开始？

1. **数学完整**：有解析解（正规方程），又能用梯度下降，两条路都能走
2. **概念载体**：过拟合、欠拟合、偏差-方差这些核心概念在这里最直观
3. **工程起点**：sklearn、PyTorch 的用法在简单模型上学最扎实

---

## 本周代码环境

```python
# 本周用到的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
```

---

## 学习建议

- D1-D2 搞清楚"为什么这样建模"，不要急着跑代码
- D3 梯度下降一定要自己手推一遍，理解才是真的理解
- D4-D5 偏差-方差权衡是之后所有模型的基础，重点理解
- D6-D7 跑通完整 sklearn 流程，养成好习惯（数据探索→预处理→训练→评估→调参）
