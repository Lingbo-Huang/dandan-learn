# Agent 学习路线图——52周完整规划

> 系统掌握 AI Agent 开发，从入门到专家级实践  
> 工具统一使用 `uv`，每周含理论 + 代码 + Claw 实战

---

## 学习路径总览

```
Phase 1: 基础构建（Week 1-8）
  → Agent 核心概念 + 工具链掌握

Phase 2: 进阶技能（Week 9-20）
  → RAG + 高级推理 + 生产部署

Phase 3: 专项深化（Week 21-36）
  → 多模态 + 评估 + 垂直领域

Phase 4: 专家实战（Week 37-52）
  → 系统设计 + 原创项目 + 前沿研究
```

---

## Phase 1：基础构建（Week 1-8）

### Week 1：什么是 Agent——定义/ReAct/规划/记忆/工具调用
- D1: Agent 定义与核心组件（感知/规划/行动/记忆）
- D2: ReAct 框架（Reasoning + Acting）
- D3: Agent 规划能力（CoT/ToT/Plan-and-Execute）
- D4: 记忆系统（短期/长期/向量检索）
- D5: 工具调用（Function Calling 完整流程）
- D6: 多 Agent 协作（主从/对等/分层架构）
- D7: Week1 综合实战（个人学习助手 Agent）

### Week 2：RAG 基础——检索增强生成
- D1: RAG 原理与架构（Retrieve + Augment + Generate）
- D2: 文档处理与分块策略（Chunking）
- D3: Embedding 模型（OpenAI/BGE/本地模型）
- D4: 向量数据库进阶（ChromaDB/Qdrant 实战）
- D5: 检索策略（相似度/MMR/混合检索）
- D6: RAG Pipeline 组装（LangChain/LlamaIndex）
- D7: 实战：构建技术文档问答系统

### Week 3：LangChain 深度掌握
- D1: LangChain 架构全览（Chain/Agent/Tool/Memory）
- D2: LCEL（LangChain Expression Language）
- D3: 自定义工具开发与注册
- D4: LangChain 记忆组件全系列
- D5: LangChain Callbacks 与监控
- D6: LangGraph 入门（状态机 Agent）
- D7: 实战：用 LangGraph 重构 Week1 项目

### Week 4：提示词工程专项
- D1: 提示词设计原则（角色/指令/示例/约束）
- D2: Few-shot 与 Chain-of-Thought 技巧
- D3: 系统提示词（System Prompt）最佳实践
- D4: 工具描述优化（影响工具选择准确率）
- D5: 提示词测试与评估方法
- D6: 模板化与变量管理（LangChain PromptTemplate）
- D7: 实战：为 Agent 工具集做提示词优化实验

### Week 5：Agent 记忆系统进阶
- D1: 记忆架构设计模式
- D2: 向量记忆实战（ChromaDB 持久化）
- D3: 知识图谱基础（Neo4j 入门）
- D4: 记忆检索策略（时间衰减/重要性加权）
- D5: 记忆压缩与摘要策略
- D6: 跨会话记忆持久化实现
- D7: 实战：为助手 Agent 添加完整记忆系统

### Week 6：工具生态与 MCP 协议
- D1: 工具设计模式（只读/写入/计算/外部服务）
- D2: 工具安全性设计（沙箱/权限/审计）
- D3: MCP（Model Context Protocol）入门
- D4: 自定义 MCP Server 开发
- D5: 工具执行引擎设计（并行/重试/超时）
- D6: 工具测试与 Mock 策略
- D7: 实战：开发一套完整的本地工具集

### Week 7：多 Agent 框架——AutoGen & CrewAI
- D1: AutoGen 核心概念（ConversableAgent/GroupChat）
- D2: AutoGen 实战：代码生成 + 自动测试
- D3: CrewAI 核心概念（Crew/Agent/Task/Process）
- D4: CrewAI 实战：研究报告自动生成
- D5: 多 Agent 通信模式对比
- D6: 自定义多 Agent 框架（轻量实现）
- D7: 实战：构建三 Agent 协作写作系统

### Week 8：Phase 1 综合项目
- D1-D2: 项目规划：构建"AI 技术周报生成器"
- D3-D4: 实现数据收集 + RAG + 多 Agent 协作
- D5: 测试与 Bug 修复
- D6: 性能优化与成本控制
- D7: 项目复盘与 Phase 1 总结

---

## Phase 2：进阶技能（Week 9-20）

### Week 9：高级推理策略
- 主题：Self-Reflection / Self-Consistency / Tree-of-Thoughts 实现
- 核心：让 Agent 自我质疑和纠错
- 项目：实现带自我评估的 Agent

### Week 10：代码执行 Agent
- 主题：Code Interpreter / Python REPL / 代码生成与执行
- 核心：让 Agent 写代码并运行
- 项目：数据分析 Agent（上传 CSV → 自动分析 → 生成图表）

### Week 11：Agent 评估体系
- 主题：评估指标设计、基准测试、自动化评估
- 核心：如何科学衡量 Agent 的质量
- 项目：构建 Agent 评估框架

### Week 12：流式输出与实时交互
- 主题：Streaming API / WebSocket / 实时进度反馈
- 核心：提升 Agent 的用户体验
- 项目：实时流式聊天 Agent

### Week 13：OpenAI Assistants API 深度实战
- 主题：Assistants API 全功能（File Search/Code Interpreter/Function Calling）
- 核心：使用官方 API 构建生产级 Agent
- 项目：基于 Assistants API 的知识库助手

### Week 14：本地模型 Agent（Ollama + LangChain）
- 主题：Ollama 本地部署 / LLaMA3 / Qwen / 隐私场景
- 核心：脱离 OpenAI 的本地化 Agent
- 项目：全本地技术问答 Agent

### Week 15：RAG 进阶——混合检索与重排序
- 主题：BM25 + 向量混合检索 / Reranker / HyDE
- 核心：大幅提升 RAG 检索质量
- 项目：企业文档检索系统（精度优化）

### Week 16：Agent 与数据库集成
- 主题：Text-to-SQL / 自然语言查询数据库
- 核心：让 Agent 直接操作结构化数据
- 项目：自然语言数据分析 Agent

### Week 17：Web 爬虫 + 信息提取 Agent
- 主题：Playwright / BeautifulSoup / 结构化信息提取
- 核心：Agent 自主获取互联网信息
- 项目：竞品监控 Agent

### Week 18：API 集成专题
- 主题：REST API 调用 / OAuth / Rate Limit 处理
- 核心：与各类外部服务集成
- 项目：多服务整合 Agent（天气/日历/邮件）

### Week 19：Agent 部署——FastAPI + Docker
- 主题：Agent 服务化 / API 设计 / 容器化
- 核心：将 Agent 变成可部署的服务
- 项目：RESTful Agent Service

### Week 20：Phase 2 综合项目
- 主题：构建"智能研究助手"（含 RAG + 工具 + 评估 + 部署）
- 项目：完整生产级 Agent 系统

---

## Phase 3：专项深化（Week 21-36）

### Week 21：多模态 Agent——图像理解
- 主题：Vision API / GPT-4V / 图像分析 Agent
- 项目：图片内容分析助手

### Week 22：多模态 Agent——语音交互
- 主题：Whisper ASR / TTS / 语音驱动 Agent
- 项目：语音问答 Agent

### Week 23：文档处理专项（PDF/Excel/PPT）
- 主题：文档解析 / 结构提取 / 多格式支持
- 项目：智能文档分析 Agent

### Week 24：知识图谱与 GraphRAG
- 主题：Neo4j / 实体关系提取 / 图谱 RAG
- 项目：知识图谱驱动的问答系统

### Week 25：Agent 安全与对齐
- 主题：Prompt Injection / 越狱防护 / 输出过滤
- 项目：构建安全防护层

### Week 26：成本优化与效率提升
- 主题：Token 优化 / 缓存策略 / 模型选择策略
- 项目：成本降低 50% 的优化实验

### Week 27：Agent 监控与可观测性
- 主题：LangSmith / Langfuse / 日志 + Trace
- 项目：完整监控系统搭建

### Week 28：垂直领域——客服 Agent
- 主题：意图识别 / 知识库 / 人工转接
- 项目：电商客服 Agent

### Week 29：垂直领域——代码助手 Agent
- 主题：代码分析 / Bug 修复 / 代码审查
- 项目：个人编程助手

### Week 30：垂直领域——数据分析 Agent
- 主题：自然语言 → 数据分析报告
- 项目：自动化 BI 助手

### Week 31：LangGraph 进阶——复杂状态机
- 主题：子图 / 循环 / 条件分支 / Human-in-the-loop
- 项目：复杂工作流 Agent

### Week 32：Semantic Kernel 实战
- 主题：微软 Semantic Kernel 框架
- 项目：企业级 Agent 方案

### Week 33：Agent 测试策略
- 主题：单元测试 / 集成测试 / E2E 测试 / 回归测试
- 项目：完整测试套件

### Week 34：CI/CD for Agent
- 主题：自动化测试 / 发布流程 / 模型版本管理
- 项目：Agent CI/CD 流水线

### Week 35：大规模 Agent 系统设计
- 主题：高并发 / 任务队列 / 分布式执行
- 项目：千级并发 Agent 系统设计

### Week 36：Phase 3 综合项目
- 主题：垂直领域完整 Agent 产品
- 项目：选择一个垂直领域，构建完整产品

---

## Phase 4：专家实战（Week 37-52）

### Week 37：前沿论文精读——Agent 规划
- 主题：ReAct / ToT / LATS 等经典论文精读
- 输出：论文笔记 + 代码复现

### Week 38：前沿论文精读——记忆与知识
- 主题：MemGPT / Self-RAG / RAPTOR
- 输出：论文笔记 + 代码复现

### Week 39：前沿论文精读——多 Agent
- 主题：AutoGen / MetaGPT / AgentVerse
- 输出：论文笔记 + 代码复现

### Week 40：微调 LLM for Agent（SFT 基础）
- 主题：LoRA / QLoRA / 指令微调基础
- 项目：微调一个工具调用专用小模型

### Week 41：强化学习与 RLHF 基础
- 主题：RLHF 原理 / PPO / DPO
- 项目：理解 ChatGPT 的训练流程

### Week 42：Agent 能力基准测试
- 主题：GAIA / AgentBench / HotpotQA
- 项目：用标准基准评估自己的 Agent

### Week 43：多模态 Agent 进阶
- 主题：Video 理解 / 3D 感知 / 具身 Agent 入门
- 项目：视频内容分析 Agent

### Week 44：Agent + 搜索引擎集成
- 主题：Perplexity 架构分析 / 深度研究 Agent
- 项目：AI 搜索 + 报告生成系统

### Week 45：企业级 RAG 系统
- 主题：权限管理 / 多租户 / 数据隔离
- 项目：企业内部知识库系统

### Week 46：Agent 产品化
- 主题：用户体验 / API 设计 / 文档 / 计费
- 项目：将一个 Agent 项目产品化

### Week 47：开源 Agent 框架贡献
- 主题：阅读 LangChain / LlamaIndex 源码 / 提交 PR
- 输出：至少一个开源贡献

### Week 48：原创 Agent 项目——立项
- 主题：发现问题 / 方案设计 / 技术选型
- 输出：项目规划文档

### Week 49：原创 Agent 项目——开发
- 主题：核心功能实现
- 输出：MVP 版本

### Week 50：原创 Agent 项目——优化
- 主题：性能优化 / Bug 修复 / 测试覆盖
- 输出：稳定版本

### Week 51：原创 Agent 项目——发布
- 主题：部署上线 / 用户反馈 / 迭代
- 输出：公开发布

### Week 52：年度复盘与展望
- D1-D3: 52周知识体系梳理
- D4-D5: 能力评估与提升方向
- D6: 撰写学习总结文章
- D7: 制定下一年度计划

---

## 附录：工具与资源

### 必备工具
```bash
# 环境管理
uv                     # Python 项目管理（本路线图统一使用）

# Agent 框架
langchain              # 主框架
langgraph              # 状态机 Agent
langchain-openai       # OpenAI 集成

# 向量数据库
chromadb               # 本地开发
qdrant-client          # 生产环境

# 本地模型
ollama                 # 本地模型运行

# 监控
langsmith              # LangChain 官方追踪
```

### 推荐学习资源

| 类型 | 资源 |
|------|------|
| 论文 | arxiv.org（搜索 ReAct, AutoGen, LLM Agent） |
| 教程 | LangChain 官方文档 |
| 社区 | Hugging Face / LangChain Discord |
| 实践 | Kaggle AI 竞赛 / LeetCode |
| 中文 | LMSYS / 机器之心 |

### 月度检查点

| 月份 | 里程碑 |
|------|--------|
| 月2（Week8结束） | 能独立构建完整 Agent，含工具+记忆 |
| 月5（Week20结束） | 能部署生产级 Agent 服务 |
| 月9（Week36结束） | 能设计垂直领域 Agent 产品 |
| 月12（Week52结束） | 有原创 Agent 项目上线，能阅读前沿论文 |

---

*路线图更新日期：2026-04*  
*如需调整周主题，欢迎在 /dandan-learn/agent/ 目录下添加反馈文件*
