---
layout: default
title: "Week 6 · 机器学习因子挖掘周规划"
---

# 量化学习线 · Week 6 周规划总览

> **主题：机器学习因子挖掘**
> Alpha158 · 特征工程 · 模型集成 · 过拟合陷阱

---

## 学习目标

- 理解 ML 在量化中的应用范式（不是万能药）
- 掌握 Alpha158 等主流因子库的设计思路
- 能用树模型/神经网络构建非线性因子
- 深刻理解过拟合陷阱与防范方法

---

## 每日安排

| 天次 | 主题 | 文件 |
|------|------|------|
| D1 | Alpha158 因子库解析 | [quant-w6-alpha158.md](./quant-w6-alpha158.md) |
| D2 | 特征工程：时序特征与截面特征 | [quant-w6-feature-engineering.md](./quant-w6-feature-engineering.md) |
| D3 | 树模型：XGBoost/LightGBM 选股 | [quant-w6-tree-models.md](./quant-w6-tree-models.md) |
| D4 | 神经网络因子 | [quant-w6-neural-factors.md](./quant-w6-neural-factors.md) |
| D5 | 模型集成与稳定性 | [quant-w6-ensemble.md](./quant-w6-ensemble.md) |
| D6 | 过拟合陷阱与防范 | [quant-w6-overfitting.md](./quant-w6-overfitting.md) |
| D7 | 综合实战：ML 多因子框架 | [quant-w6-capstone.md](./quant-w6-capstone.md) |

---

## 本周核心问题

- ML 因子和传统因子的本质区别是什么？
- 为什么 ML 模型在金融数据上特别容易过拟合？
- 如何正确做时间序列的训练/测试分割？
- 特征重要性可以替代因子 IC 吗？

---

## ML 量化的局限性清单

在开始之前，先记住这些：

- 金融时序的信噪比极低（收益主要是噪声）
- 样本量相对机器学习而言很小
- 参数越多，过拟合风险越大
- ML 黑箱模型难以审计，监管风险
- 策略衰减：信号被他人发现后 alpha 衰减
