# Week 2 周计划：价格与收益率深度统计

> **学习主线**：从"价格序列"到"收益率序列"，理解对数收益率的优良性质，检验正态假设，直面胖尾现实，学会用 VaR/CVaR 度量风险，最后综合一个完整的数据实战项目。

---

## 本周目标

| 目标 | 说明 |
|------|------|
| 理解对数收益率 | 掌握其数学性质与简单收益率的区别 |
| 检验正态假设 | 用 Shapiro-Wilk、JB 检验、QQ 图实证 |
| 认识胖尾分布 | 峰度、偏度、t 分布、极值理论 EVT |
| 掌握风险度量 | VaR（历史法/参数法/Monte Carlo）、CVaR |
| 相关性分析 | Pearson/Spearman/Rolling 相关矩阵 |
| 数据实战 | 用 akshare 获取真实 A 股数据做全流程分析 |
| 综合项目 | 构建一个多因子组合风险报告 |

---

## 每日安排

| 天 | 内容 | 文件 |
|----|------|------|
| Day 1 | 对数收益率——理论与 Python 计算 | `quant-w2-log-returns.md` |
| Day 2 | 正态假设检验——理论与实证 | `quant-w2-normal-hypothesis.md` |
| Day 3 | 胖尾分布——峰度、t 分布、EVT | `quant-w2-fat-tails.md` |
| Day 4 | 风险度量——VaR 与 CVaR | `quant-w2-risk-measures.md` |
| Day 5 | 相关性分析——矩阵与滚动相关 | `quant-w2-correlation.md` |
| Day 6 | 数据实战——A 股全流程统计 | `quant-w2-data-practice.md` |
| Day 7 | 综合项目——多资产风险报告 | `quant-w2-capstone.md` |

---

## 环境准备

```bash
# 使用 uv 管理 Python 环境（Week 1 已完成安装）
uv pip install akshare pandas numpy scipy matplotlib seaborn statsmodels
```

或在脚本头部直接使用 `uv run`：

```bash
# 单文件执行（无需提前激活环境）
uv run --with akshare --with pandas --with numpy --with scipy --with matplotlib python script.py
```

---

## 核心概念速览

### 收益率类型

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}} \quad \text{（简单收益率）}$$

$$R_t = \ln\frac{P_t}{P_{t-1}} = \ln P_t - \ln P_{t-1} \quad \text{（对数收益率）}$$

### 风险度量层级

```
均值/方差（正态假设）
    ↓
VaR（分位数风险）
    ↓
CVaR/ES（尾部期望损失）
    ↓
极值理论 EVT（极端事件）
```

---

## 本周关键结论预告

1. **对数收益率天然可加**：多期收益率 = 单期对数收益率之和
2. **正态假设是个谎言**：实际金融收益率有显著尖峰厚尾
3. **VaR 不告诉你最坏情况**：CVaR 更好地刻画尾部风险
4. **相关性在危机时趋向 1**：分散化在最需要时失效

---

## 参考资源

- *Quantitative Risk Management* — McNeil, Frey, Embrechts
- *Options, Futures and Other Derivatives* — John Hull
- akshare 文档：https://akshare.akfamily.xyz/
- scipy.stats 文档：https://docs.scipy.org/doc/scipy/reference/stats.html

---

*下一步 → [Day 1: 对数收益率](./quant-w2-log-returns.md)*
