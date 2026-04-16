# 📈 量化交易一年速成计划 ROADMAP

> **项目**：丹丹的 AI + 大模型 + 量化一年速成计划 · 量化线  
> **目标**：从零基础到具备独立研发量化策略的能力，达到业界顶尖水平  
> **周期**：52 周（约12个月）  
> **学习节奏**：每日发布知识点，每周输出一个完整主题模块  
> **文件路径**：`/dandan-learn/quant/week-XX/dayYY-主题.md`

---

## 🗺️ 全年学习路线总览

```
第一阶段：地基（Week 01–08）    → 数学/统计/Python/金融基础
第二阶段：入门（Week 09–16）    → 数据获取、回测框架、策略基础
第三阶段：进阶（Week 17–28）    → 因子体系、机器学习、风险管理
第四阶段：高阶（Week 29–40）    → 深度学习、高频策略、组合优化
第五阶段：顶尖（Week 41–52）    → 研究级专题、实盘部署、策略迭代
```

---

## 📚 第一阶段：地基打牢（Week 01–08）

### Week 01 | Python 量化基础环境
- Day 1：量化交易全景导览——什么是量化，做什么，怎么学
- Day 2：Python 环境搭建（Anaconda / Jupyter / VSCode）
- Day 3：NumPy 核心——数组运算、广播机制、金融向量化
- Day 4：Pandas 核心——Series / DataFrame / 时间序列索引
- Day 5：Pandas 进阶——groupby / merge / resample / rolling

### Week 02 | 金融数据处理
- Day 1：金融时间序列特性（平稳性、自相关、异方差）
- Day 2：股票数据结构解析（OHLCV、复权、除权除息）
- Day 3：Matplotlib / Plotly 金融可视化
- Day 4：数据清洗实战（缺失值、异常值、对齐）
- Day 5：收益率计算（简单收益、对数收益、累计收益）

### Week 03 | 概率与统计基础
- Day 1：描述统计——均值、方差、偏度、峰度
- Day 2：概率分布——正态、t分布、厚尾分布
- Day 3：假设检验——t检验、p值、显著性
- Day 4：相关分析——Pearson / Spearman / 协方差矩阵
- Day 5：线性回归——最小二乘、多重共线性、OLS

### Week 04 | 金融数学基础
- Day 1：利率与时间价值——复利、折现、债券定价
- Day 2：期望收益与风险——均值-方差框架
- Day 3：随机游走与布朗运动
- Day 4：正态分布假设的局限与实际金融数据
- Day 5：蒙特卡洛模拟入门

### Week 05 | 技术分析体系
- Day 1：K线图解读——形态、实体、影线
- Day 2：均线家族——SMA / EMA / DEMA / WMA
- Day 3：动量指标——RSI / MACD / KDJ
- Day 4：波动率指标——ATR / 布林带 / 标准差通道
- Day 5：成交量指标——OBV / VWAP / 量价关系

### Week 06 | 市场微观结构
- Day 1：订单簿基础——买卖盘、Level 1/2 数据
- Day 2：交易成本分析——滑点、佣金、市场冲击
- Day 3：流动性度量——买卖价差、深度、Amihud比率
- Day 4：A股特有机制——涨跌停、T+1、集合竞价
- Day 5：期货市场基础——合约、期限结构、展期

### Week 07 | 回测系统原理
- Day 1：回测的本质与陷阱——前视偏差、幸存者偏差
- Day 2：事件驱动 vs 向量化回测
- Day 3：Backtrader 框架入门
- Day 4：Backtrader 实战——第一个均线策略
- Day 5：绩效指标体系——夏普、最大回撤、Calmar

### Week 08 | 统计学进阶
- Day 1：时间序列平稳性——ADF检验、PP检验
- Day 2：ARIMA 模型
- Day 3：GARCH 模型——波动率建模
- Day 4：协整检验——Engle-Granger、Johansen
- Day 5：第一阶段总复习与自测题

---

## 🚀 第二阶段：策略入门（Week 09–16）

### Week 09 | 经典策略框架
- Day 1：策略分类——趋势、均值回归、套利、统计套利
- Day 2：双均线策略——信号生成、完整实现
- Day 3：布林带策略——均值回归逻辑
- Day 4：突破策略——唐奇安通道、海龟策略原理
- Day 5：策略评估与改进方法论

### Week 10 | 数据获取全栈
- Day 1：AKShare 数据获取——A股日线、分钟线
- Day 2：Tushare / BaoStock 接口使用
- Day 3：期货数据获取——主力合约、连续合约处理
- Day 4：财务数据获取——利润表、资产负债表、现金流
- Day 5：数据库存储——SQLite / MySQL + Pandas

### Week 11 | 截面动量与反转
- Day 1：动量效应的学术来源（Jegadeesh & Titman）
- Day 2：回望期与持有期选择
- Day 3：截面动量策略实现
- Day 4：短期反转效应
- Day 5：动量崩溃现象与风控

### Week 12 | 统计套利入门
- Day 1：配对交易原理——协整关系
- Day 2：寻找协整对——行业内配对
- Day 3：z-score 信号构建
- Day 4：配对策略回测实战
- Day 5：多空组合构建基础

### Week 13 | 期货趋势策略
- Day 1：CTA策略概述
- Day 2：ATR通道突破策略
- Day 3：多品种分散化趋势跟踪
- Day 4：仓位管理——固定比例 vs ATR头寸
- Day 5：期货展期与连续合约处理

### Week 14 | 因子挖掘入门
- Day 1：因子投资理论——CAPM / Fama-French
- Day 2：价值因子——PE、PB、PS
- Day 3：成长因子——ROE、营收增速
- Day 4：质量因子——毛利率、资产周转率
- Day 5：因子有效性检验方法

### Week 15 | VectorBT 高效回测
- Day 1：VectorBT 框架特性与安装
- Day 2：向量化策略回测实战
- Day 3：参数优化——网格搜索
- Day 4：策略对比与绩效可视化
- Day 5：过拟合防范——样本外测试

### Week 16 | 阶段项目：完整策略研究报告
- Day 1：选题与数据准备
- Day 2：策略逻辑设计
- Day 3：回测实现
- Day 4：绩效分析与优化
- Day 5：完整研究报告输出（第二阶段收官）

---

## ⚡ 第三阶段：进阶提升（Week 17–28）

### Week 17 | Alpha 因子研究体系
- Day 1：Alpha 的定义与来源
- Day 2：因子数据标准化——去极值、中性化、标准化
- Day 3：因子 IC 分析（信息系数）
- Day 4：因子分层回测（十分位组合）
- Day 5：因子衰减与时效性

### Week 18 | 多因子模型
- Day 1：多因子模型框架——Barra模型原理
- Day 2：风险因子 vs Alpha 因子
- Day 3：因子合成方法——等权、IC加权、机器学习合成
- Day 4：多因子选股模型实现
- Day 5：因子正交化与冗余处理

### Week 19 | 机器学习入门（量化视角）
- Day 1：ML for 量化——回归 vs 分类，特征工程
- Day 2：线性模型——Ridge、Lasso、ElasticNet
- Day 3：树模型——决策树、随机森林
- Day 4：Gradient Boosting——XGBoost / LightGBM
- Day 5：模型评估——IC、Rank IC、防止过拟合

### Week 20 | 特征工程进阶
- Day 1：技术指标特征库构建
- Day 2：基本面特征——财务比率特征
- Day 3：情绪特征——资金流、北向、融资融券
- Day 4：时序特征——lag特征、滚动统计
- Day 5：特征重要性分析与筛选

### Week 21 | 风险管理体系
- Day 1：风险度量——标准差、VaR、CVaR
- Day 2：历史模拟法 vs 参数法 vs 蒙特卡洛
- Day 3：回撤控制——最大回撤、条件回撤
- Day 4：止损机制设计
- Day 5：压力测试与情景分析

### Week 22 | 组合优化
- Day 1：马科维茨均值-方差优化
- Day 2：有效前沿与最大夏普组合
- Day 3：风险平价（Risk Parity）
- Day 4：Black-Litterman 模型
- Day 5：实用组合优化工具：PyPortfolioOpt

### Week 23 | 事件驱动策略
- Day 1：事件驱动策略框架
- Day 2：财报效应——业绩超预期策略
- Day 3：分红策略、股权激励事件
- Day 4：重大资产重组、并购事件
- Day 5：事件策略回测注意事项

### Week 24 | 期权基础
- Day 1：期权基本概念——认购/认沽、执行价、到期日
- Day 2：BS定价模型
- Day 3：Greeks——Delta / Gamma / Theta / Vega
- Day 4：隐含波动率与VIX
- Day 5：简单期权策略——保护性看跌、备兑看涨

### Week 25 | 量价因子深度
- Day 1：量价因子的信息来源
- Day 2：成交量异常检测
- Day 3：资金流向因子
- Day 4：分钟级量价特征
- Day 5：龙虎榜、大单因子

### Week 26 | 文本与另类数据
- Day 1：另类数据概述——卫星、信用卡、网络爬虫
- Day 2：新闻情绪分析基础
- Day 3：财报文本挖掘——MD&A分析
- Day 4：社交媒体情绪——Reddit / 东方财富股吧
- Day 5：另类数据的合规使用

### Week 27 | 策略容量与实盘准备
- Day 1：策略容量分析
- Day 2：冲击成本建模
- Day 3：实盘接口选型——券商QMT、掘金量化
- Day 4：实盘撮合与模拟盘对比
- Day 5：第三阶段综合项目

### Week 28 | 阶段项目：多因子选股系统
- Day 1：因子库搭建
- Day 2：因子有效性批量检验
- Day 3：多因子合成与选股
- Day 4：组合回测与归因
- Day 5：研究报告输出（第三阶段收官）

---

## 🧠 第四阶段：高阶突破（Week 29–40）

### Week 29 | 深度学习基础
- Day 1：神经网络原理与PyTorch入门
- Day 2：全连接网络用于截面预测
- Day 3：CNN在金融时序上的应用
- Day 4：LSTM——序列预测
- Day 5：Transformer——注意力机制入门

### Week 30 | 深度学习量化应用
- Day 1：端到端收益率预测
- Day 2：图神经网络——股票关系建模
- Day 3：对比学习用于因子提取
- Day 4：模型集成与预测校准
- Day 5：深度模型过拟合防控

### Week 31 | 强化学习与量化
- Day 1：RL框架——状态、动作、奖励
- Day 2：Q-Learning 与 DQN
- Day 3：PPO 算法
- Day 4：FinRL 框架实战
- Day 5：RL策略评估与局限

### Week 32 | 高频数据分析
- Day 1：高频数据特性——tick数据处理
- Day 2：微观结构模型——Kyle, Glosten-Milgrom
- Day 3：价格发现与信息不对称
- Day 4：高频因子提取
- Day 5：Level 2 数据实战

### Week 33 | 做市策略
- Day 1：做市商角色与P&L分解
- Day 2：Avellaneda-Stoikov 模型
- Day 3：库存风险管理
- Day 4：做市策略回测
- Day 5：做市策略的实践挑战

### Week 34 | 统计套利进阶
- Day 1：ETF 套利
- Day 2：跨期套利——期货跨月价差
- Day 3：跨市套利——A/H股溢价
- Day 4：期现套利——股指期货基差
- Day 5：套利策略风险管理

### Week 35 | 宏观量化
- Day 1：宏观因子——利率、通胀、PMI
- Day 2：大类资产配置框架
- Day 3：美林投资时钟
- Day 4：宏观风险因子建模
- Day 5：全球宏观对冲基金策略

### Week 36 | 系统化交易架构
- Day 1：量化系统架构设计
- Day 2：数据管道——实时 vs 批量
- Day 3：信号生成服务
- Day 4：订单管理系统（OMS）
- Day 5：风控系统设计

### Week 37 | 绩效归因分析
- Day 1：Brinson 归因模型
- Day 2：因子归因——Barra归因
- Day 3：交易成本归因
- Day 4：时间序列归因
- Day 5：归因报告自动化

### Week 38 | 策略研究方法论
- Day 1：研究流程规范化
- Day 2：多重假设检验问题
- Day 3：p-hacking 防范
- Day 4：策略退化监控
- Day 5：研究与实盘的落差管理

### Week 39 | 大模型与量化结合
- Day 1：LLM用于量化研究综述
- Day 2：LLM生成因子与策略
- Day 3：RAG用于研报分析
- Day 4：LLM Agent 量化工作流
- Day 5：大模型量化的边界与风险

### Week 40 | 阶段项目：完整量化交易系统
- Day 1：系统设计文档
- Day 2：数据层实现
- Day 3：策略层实现
- Day 4：回测与评估
- Day 5：系统演示与报告（第四阶段收官）

---

## 🏆 第五阶段：顶尖水平（Week 41–52）

### Week 41 | 海外顶尖机构研究复现
- Day 1：AQR经典论文——动量、价值
- Day 2：Two Sigma研究思路
- Day 3：Man AHL CTA体系
- Day 4：复现一篇顶尖量化论文
- Day 5：论文复现的坑与收获

### Week 42 | 私募策略解析
- Day 1：国内头部量化私募概览
- Day 2：百亿量化的因子体系猜想
- Day 3：DMA策略原理
- Day 4：T0策略框架
- Day 5：私募量化策略的监管合规

### Week 43 | 因子挖掘自动化
- Day 1：遗传规划（GP）挖因子
- Day 2：WorldQuant Alpha101 解析
- Day 3：自动化因子搜索框架
- Day 4：因子质量评估体系
- Day 5：因子库管理与迭代

### Week 44 | 实盘部署进阶
- Day 1：实盘风控框架
- Day 2：异常检测与自动熔断
- Day 3：策略容量监控
- Day 4：交易报告自动化
- Day 5：实盘问题案例复盘

### Week 45 | 另类策略专题
- Day 1：网格策略
- Day 2：可转债策略——双低轮动
- Day 3：打新策略
- Day 4：北向资金跟踪策略
- Day 5：量化择时策略

### Week 46 | 量化研究员进阶技能
- Day 1：研究框架——提问→验证→结论
- Day 2：数据驱动 vs 直觉驱动
- Day 3：如何阅读量化论文
- Day 4：量化研究员职业路径
- Day 5：行业动态与前沿追踪

### Week 47 | 组合管理高阶
- Day 1：多策略组合——相关性管理
- Day 2：策略权重动态调整
- Day 3：杠杆使用与风险控制
- Day 4：尾部风险对冲
- Day 5：组合级压力测试

### Week 48 | 量化研究专题一：波动率
- Day 1：已实现波动率
- Day 2：HAR-RV模型
- Day 3：期权隐含波动率曲面
- Day 4：波动率套利策略
- Day 5：VIX衍生品与对冲

### Week 49 | 量化研究专题二：ESG量化
- Day 1：ESG评分体系
- Day 2：ESG因子有效性
- Day 3：ESG组合构建
- Day 4：可持续投资策略
- Day 5：ESG量化的争议与前沿

### Week 50 | 量化研究专题三：加密货币量化
- Day 1：加密市场特性
- Day 2：Binance API 数据获取
- Day 3：加密市场因子
- Day 4：DeFi套利机会
- Day 5：加密量化的风险与合规

### Week 51 | 综合项目：个人量化研究系统
- Day 1：项目规划与架构设计
- Day 2：数据层完善
- Day 3：因子库与策略库
- Day 4：实盘监控 Dashboard
- Day 5：完整演示与文档

### Week 52 | 收官：复盘与下一步
- Day 1：一年学习复盘——得与失
- Day 2：量化能力自我评估
- Day 3：未来方向规划
- Day 4：个人量化研究框架总结
- Day 5：🎉 结业仪式：量化交易者宣言

---

## 📁 文件目录结构

```
/dandan-learn/quant/
├── ROADMAP.md          ← 本文件：52周规划
├── resources/          ← 参考资料、工具清单
├── week-01/            ← 第1周：Python量化基础
│   ├── day01-量化全景导览.md
│   ├── day02-环境搭建.md
│   └── ...
├── week-02/            ← 第2周：金融数据处理
│   └── ...
...
└── week-52/            ← 第52周：收官复盘
```

---

## 🛠️ 推荐工具与资源

| 类型 | 工具/资源 |
|------|----------|
| 数据获取 | AKShare、Tushare、BaoStock |
| 回测框架 | Backtrader、VectorBT、QMT |
| 机器学习 | scikit-learn、XGBoost、LightGBM |
| 深度学习 | PyTorch、TensorFlow |
| 组合优化 | PyPortfolioOpt、CVXPY |
| 可视化 | Matplotlib、Plotly、Streamlit |
| 必读书籍 | 《量化投资以Python为工具》《Advances in Financial Machine Learning》|
| 必读论文 | AQR因子库、SSRN量化金融专区 |

---

*最后更新：2026-04-14 | 量化线 Worker 🥚🥚3号*
