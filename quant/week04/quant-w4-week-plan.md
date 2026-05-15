---
layout: default
title: "Week 4 · 因子体系周规划"
---

# 量化学习线 · Week 4 周规划总览

> **主题：因子体系**
> 动量 · 价值 · 质量 · 低波 · 多因子合成 · IC/ICIR 评价

---

## 学习目标

Week 4 的核心目标是掌握量化因子体系的理论基础与工程实现，能够独立构建、评估和合成多因子选股模型。

- 理解各类因子（动量/价值/质量/低波动）的经济学逻辑
- 掌握 IC、ICIR、因子分层等主流因子评价体系
- 能用 Python 实现因子计算、评估与多因子合成
- 了解因子衰减、因子正交化等实际工程问题

---

## 每日安排一览

| 天次 | 主题 | 文件 |
|------|------|------|
| D1 | 动量因子：从学术研究到工程实现 | [quant-w4-momentum.md](./quant-w4-momentum.md) |
| D2 | 价值因子：PE/PB/PS/DCF 体系 | [quant-w4-value.md](./quant-w4-value.md) |
| D3 | 质量因子：盈利能力与财务稳健性 | [quant-w4-quality.md](./quant-w4-quality.md) |
| D4 | 低波动因子：风险异象与实现方式 | [quant-w4-low-vol.md](./quant-w4-low-vol.md) |
| D5 | IC/ICIR 评价体系 | [quant-w4-ic-icir.md](./quant-w4-ic-icir.md) |
| D6 | 多因子合成：从打分法到优化法 | [quant-w4-multifactor.md](./quant-w4-multifactor.md) |
| D7 | 综合实战：完整多因子选股框架 | [quant-w4-capstone.md](./quant-w4-capstone.md) |

---

## 重点提示

### 因子研究的核心逻辑
- 因子 = 对某种"超额收益来源"的系统性捕捉
- 有效因子需要：① 经济学逻辑 ② 统计显著性 ③ 实盘可实现 ④ 低换手

### 这周之后你应该能回答
- IC 和 ICIR 分别衡量什么？多少算好？
- 动量因子为什么在中国市场会反转？
- 多因子合成中等权法和优化法的核心区别是什么？
- 因子暴露度如何标准化？为什么要做市值中性化？

---

## 配套资源

- 开山论文：Fama & French (1993) "Common Risk Factors in Returns"
- 国内经典：光大证券多因子系列研究报告
- 工具：`alphalens`（因子分析）、`qlib`（量化研究框架）
