---
layout: default
title: "Week 7 · 高频与执行周规划"
---

# 量化学习线 · Week 7 周规划总览

> **主题：高频与执行**
> 市场微观结构 · TWAP/VWAP · 冲击成本 · 做市策略入门

---

## 学习目标

- 理解订单簿、市场微结构的基本框架
- 掌握 TWAP/VWAP 等主流算法交易执行方式
- 能估算策略的冲击成本（市场影响）
- 了解做市商策略的盈利模式

---

## 每日安排

| 天次 | 主题 | 文件 |
|------|------|------|
| D1 | 市场微观结构基础 | [quant-w7-microstructure.md](./quant-w7-microstructure.md) |
| D2 | 订单类型与订单簿 | [quant-w7-orderbook.md](./quant-w7-orderbook.md) |
| D3 | TWAP 与 VWAP 算法执行 | [quant-w7-twap-vwap.md](./quant-w7-twap-vwap.md) |
| D4 | 冲击成本与市场影响模型 | [quant-w7-market-impact.md](./quant-w7-market-impact.md) |
| D5 | 做市策略入门 | [quant-w7-market-making.md](./quant-w7-market-making.md) |
| D6 | 高频数据处理与分析 | [quant-w7-hft-data.md](./quant-w7-hft-data.md) |
| D7 | 综合实战：执行优化框架 | [quant-w7-capstone.md](./quant-w7-capstone.md) |

---

## 本周核心问题

- Bid-Ask Spread 是由什么决定的？如何估算？
- VWAP vs. TWAP：什么情况下用哪个？
- 为什么大单会推高价格（冲击成本）？如何量化？
- 做市商怎么赚钱？承担什么风险？

---

## 高频量化的边界

- **高频（HFT）**：微秒/毫秒级，需要专用硬件，散户无法参与
- **中频**：分钟级，算法执行，可用程序化交易
- **本周重点**：偏向中频执行优化，对策略容量和执行有实际帮助
