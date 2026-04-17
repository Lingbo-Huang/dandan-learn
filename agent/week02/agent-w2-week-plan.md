# Week 2 学习计划：主流 Agent 框架实战

> **学习主线**：从「理解 LLM 能做什么」到「用主流框架搭建真正的 Agent 系统」  
> **前置要求**：已完成 Week 1（LLM 基础、Prompt 工程、基础调用）  
> **工具约定**：所有代码示例均使用 `uv` 管理依赖

---

## 本周目标

| 目标 | 说明 |
|------|------|
| 🧠 理解框架本质 | 知道 LangChain / LlamaIndex / AutoGen / CrewAI 分别解决什么问题 |
| 🛠️ 能跑通核心示例 | 每个框架至少跑一个有意义的 demo |
| 🔍 学会框架选型 | 根据任务类型做出合理框架选择 |
| 🤝 理解与 Claw 的关系 | 这些框架如何与 OpenClaw 类平台的理念呼应或互补 |

---

## 每日安排

### Day 1 — LangChain 核心概念

- 文件：`agent-w2-langchain-basics.md`
- 内容：LangChain 整体架构、LLM 封装、PromptTemplate、OutputParser
- 目标：用 uv 建环境，跑通第一个 LangChain 调用链

### Day 2 — LangChain 链与工具

- 文件：`agent-w2-langchain-chains.md`
- 内容：LCEL（LangChain Expression Language）、Chain 组合、Tool 绑定、Memory
- 目标：构建一个带工具调用的完整 Chain

### Day 3 — LlamaIndex 数据框架

- 文件：`agent-w2-llamaindex-intro.md`
- 内容：LlamaIndex 的 RAG 核心、Index 类型、Query Engine、与 LangChain 的定位差异
- 目标：建一个本地文档问答系统

### Day 4 — AutoGen 多 Agent 协作

- 文件：`agent-w2-autogen-multiagent.md`
- 内容：AutoGen 对话式 Agent 模型、GroupChat、Human-in-the-loop
- 目标：搭建一个 User + Assistant + Critic 三方协作场景

### Day 5 — CrewAI 角色编排

- 文件：`agent-w2-crewai-roles.md`
- 内容：CrewAI 的 Agent/Task/Crew 三层结构、角色定义、任务串行/并行
- 目标：构建一个研究员 + 写手 + 编辑协作的 Crew

### Day 6 — 框架横向对比

- 文件：`agent-w2-framework-comparison.md`
- 内容：从适用场景、学习曲线、生态、性能等多维度对比四个框架
- 目标：建立自己的框架选型决策树

### Day 7 — 综合实战项目

- 文件：`agent-w2-capstone.md`
- 内容：综合运用本周所学，构建一个完整的多框架协作示例
- 目标：有一个可展示的 Week 2 作品

---

## 环境准备

```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建本周项目环境
uv init agent-week02
cd agent-week02

# 安装核心依赖（后续各天按需添加）
uv add langchain langchain-openai langchain-community
uv add llama-index llama-index-llms-openai
uv add pyautogen
uv add crewai

# 设置环境变量
export OPENAI_API_KEY="your-key-here"
```

---

## 框架定位速览

```
LangChain    → 通用 AI 应用链路构建（最成熟、生态最丰富）
LlamaIndex   → 以数据为核心的 RAG 框架（最擅长文档检索增强）
AutoGen      → 对话式多 Agent 协作（微软出品，适合复杂推理协作）
CrewAI       → 角色驱动的多 Agent 任务编排（更贴近"团队协作"隐喻）
```

---

## 与 OpenClaw 的关系

OpenClaw 是面向**用户侧的 Agent 平台**，提供 Session / Project / Task 的生命周期管理。本周学习的这些框架是**开发者侧的 Agent 构建工具**——你可以用 LangChain / AutoGen 等框架写出一个 Worker Agent，然后把它部署为 Claw 的一个 Skill 或 Worker。

理解这个层次关系，是本周最重要的元认知目标。

---

## 小结

Week 2 的核心不是"学会每个框架的所有 API"，而是建立**框架选型直觉**——面对一个问题，脑子里能快速浮现"这个用 LlamaIndex 最合适"或"这个场景更适合 CrewAI"。到 Week 7 综合实战时，你会需要这种判断力。
