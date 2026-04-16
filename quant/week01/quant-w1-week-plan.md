# 量化学习线 · Week 1 周规划总览

> **主题：金融市场概览**
> A股 · 港股 · 美股 · 期货 · 加密货币

---

## 学习目标

Week 1 的核心目标是建立对主要金融市场的整体认知框架，为后续的量化策略开发和数据分析打下坚实基础。

- 理解各类市场的结构、交易规则与核心参与者
- 掌握基础行情数据的读取与理解方式
- 动手获取真实市场数据，初步感受数据结构
- 建立横向比较思维：不同市场的异同与联动

---

## 每日安排一览

| 天次 | 主题 | 核心市场 | 文件 |
|------|------|----------|------|
| D1 | A股市场概览 | 上交所 / 深交所 | [quant-w1-金融市场全景-股票期货加密货币.md](./quant-w1-金融市场全景-股票期货加密货币.md) |
| D2 | 港股市场概览 | 香港联交所 | [quant-w1-价格与收益率-对数收益正态假设胖尾.md](./quant-w1-价格与收益率-对数收益正态假设胖尾.md) |
| D3 | 美股市场概览 | NYSE / NASDAQ | [quant-w1-技术分析基础-K线均线MACD-RSI.md](./quant-w1-技术分析基础-K线均线MACD-RSI.md) |
| D4 | 期货市场概览 | 商品期货 / 金融期货 | [quant-w1-基本面基础-PE-PB-ROE财报解读.md](./quant-w1-基本面基础-PE-PB-ROE财报解读.md) |
| D5 | 加密货币市场概览 | BTC / ETH / DeFi | [quant-w1-市场微观结构-买卖价差流动性.md](./quant-w1-市场微观结构-买卖价差流动性.md) |
| D6 | 市场数据获取实战 | 多市场数据接入 | [quant-w1-时间序列统计-平稳性ADF协整.md](./quant-w1-时间序列统计-平稳性ADF协整.md) |
| D7 | 周回顾与横向对比 | 综合 | [quant-w1-数据工具实战-Tushare-AkShare-pandas.md](./quant-w1-数据工具实战-Tushare-AkShare-pandas.md) |

---

## 学习方法建议

1. **先读后练**：每天先通读知识点，再动手做练习
2. **边做边记**：用 Jupyter Notebook 记录代码和思考
3. **数据为王**：尽量用真实数据验证概念，不只停留在理论
4. **建立对比表格**：用一个共同模板记录各市场特征，便于 D7 横向对比

---

## 工具准备

```bash
# 推荐环境（uv 管理）
uv pip install akshare yfinance ccxt pandas matplotlib jupyter

# 验证安装
python -c "import akshare, yfinance, ccxt, pandas; print('All OK')"
```

---

## Week 1 学习产出

- [ ] 各市场基础认知笔记（D1-D5）
- [ ] 5个市场数据集各一份（CSV 格式）
- [ ] 横向对比分析表（D7 完成）
- [ ] Jupyter Notebook 代码实践记录

---

*更新于 Week 1 启动*
