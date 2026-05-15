---
layout: default
title: "Week 5 周规划：模型评估与集成学习"
---

# Week 5 · 模型评估与集成学习周规划

## 本周目标

彻底搞清楚**如何公正地评估模型**，以及掌握比随机森林更强的**Boosting 系列算法**。

## 学习路线

| 天 | 主题 | 核心收获 |
|----|------|---------|
| D1 | 交叉验证与过拟合诊断 | 偏差-方差分解、学习曲线、CV 策略 |
| D2 | 评估指标全家桶 | Accuracy/Precision/Recall/F1/AUC/PR曲线 |
| D3 | Boosting 原理 | AdaBoost：弱学习器→强学习器 |
| D4 | 梯度提升与 XGBoost | GBDT 推导、XGBoost 工程优化 |
| D5 | 模型调优全流程 | 特征工程、超参数搜索、Stacking |
| D6 | 综合实战 | Kaggle 竞赛流程模拟 |

## 知识图谱

```
集成学习
├── Bagging（并行）
│   ├── 随机森林（Week4）
│   └── Extra Trees
└── Boosting（串行）
    ├── AdaBoost（样本权重）
    ├── GBDT（梯度拟合残差）
    ├── XGBoost（工程优化+正则）
    ├── LightGBM（直方图+叶子分裂）
    └── CatBoost（类别特征优化）
```

## 本周核心公式

偏差-方差分解：

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

## 面试必备问题

1. 偏差和方差分别是什么？如何权衡？
2. AdaBoost 和 GBDT 有什么区别？
3. XGBoost 比 GBDT 快在哪里？有哪些正则化手段？
4. 为什么 Boosting 容易过拟合而 Bagging 不容易？
5. AUC 和 PR-AUC 分别适合什么场景？
