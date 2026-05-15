---
layout: default
title: "Week 5 · 策略开发周规划"
---

# 量化学习线 · Week 5 周规划总览

> **主题：策略开发**
> CTA · 统计套利 · 截面动量 · 事件驱动 · 策略评估框架

---

## 学习目标

- 掌握主流量化策略的核心逻辑和适用场景
- 能独立实现 CTA 趋势跟踪和统计套利策略
- 理解策略评估的核心指标（夏普、最大回撤、Calmar 等）
- 避开策略开发的常见陷阱（过拟合、未来函数）

---

## 每日安排

| 天次 | 主题 | 文件 |
|------|------|------|
| D1 | CTA 趋势跟踪策略 | [quant-w5-cta.md](./quant-w5-cta.md) |
| D2 | 统计套利：配对交易 | [quant-w5-stat-arb.md](./quant-w5-stat-arb.md) |
| D3 | 截面动量策略 | [quant-w5-cross-sectional.md](./quant-w5-cross-sectional.md) |
| D4 | 事件驱动策略 | [quant-w5-event-driven.md](./quant-w5-event-driven.md) |
| D5 | 策略评估框架 | [quant-w5-evaluation.md](./quant-w5-evaluation.md) |
| D6 | 回测引擎设计 | [quant-w5-backtest-engine.md](./quant-w5-backtest-engine.md) |
| D7 | 综合实战：多策略组合 | [quant-w5-capstone.md](./quant-w5-capstone.md) |

---

## 本周核心问题

- CTA 策略的核心假设是什么？在什么市场条件下失效？
- 配对交易如何选对？协整检验 vs. 相关性的区别？
- 回测绩效"好看"为什么可能是假的？
- 夏普比率 > 2 一定是好策略吗？

---

## 策略分类速览

| 策略类型 | 持仓周期 | 核心信号 | 容量 |
|---------|---------|---------|------|
| CTA 趋势 | 周-月 | 价格趋势 | 大 |
| 统计套利 | 日-周 | 价差均值回归 | 中 |
| 截面选股 | 月 | 因子信号 | 大 |
| 事件驱动 | 日 | 公告/业绩 | 小 |
| 高频做市 | 秒-分 | 订单流 | 极小 |
