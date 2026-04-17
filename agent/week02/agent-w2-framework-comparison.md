# D6 四大 Agent 框架横向对比与选型建议

> **学习目标**：从多个维度深入对比 LangChain、LlamaIndex、AutoGen、CrewAI 四大主流框架，建立清晰的选型决策思路，避免"用锤子看一切都是钉子"。

---

## 一、框架概览：各司其职

在深入对比之前，先理解每个框架的"出身"和定位：

| 框架 | 发布时间 | 核心团队 | 核心定位 | Star 数（2024） |
|------|---------|---------|---------|----------------|
| LangChain | 2022.10 | Harrison Chase | 通用 LLM 应用框架 | 90k+ |
| LlamaIndex | 2022.11 | Jerry Liu | 数据连接与 RAG 专家 | 35k+ |
| AutoGen | 2023.09 | 微软研究院 | 多 Agent 对话协作 | 30k+ |
| CrewAI | 2024.01 | João Moura | 角色化团队协作 | 20k+ |

这四个框架并非"谁最好"的竞争关系，而是覆盖了 AI 应用开发的不同维度。

---

## 二、核心设计哲学对比

### LangChain：一切皆可组合

```
设计哲学：组件化 + 管道化
核心抽象：Runnable（可运行单元）
连接方式：LCEL 管道 (|)
适合比喻：乐高积木 —— 灵活拼装各种组件
```

```python
# LangChain 风格：清晰的管道
chain = prompt | llm | output_parser | post_processor
result = chain.invoke({"input": "..."})
```

### LlamaIndex：数据优先，RAG 至上

```
设计哲学：数据索引 + 智能检索
核心抽象：Index（索引） + Node（数据节点）
连接方式：QueryEngine + Retriever
适合比喻：图书馆管理系统 —— 专业的数据组织与检索
```

```python
# LlamaIndex 风格：以数据为中心
index = VectorStoreIndex.from_documents(docs)
engine = index.as_query_engine(similarity_top_k=5)
response = engine.query("问题")
```

### AutoGen：对话即计算

```
设计哲学：通过自然语言对话协调多 Agent
核心抽象：ConversableAgent（可对话的 Agent）
连接方式：initiate_chat() 触发对话链
适合比喻：会议室 —— Agent 们通过讨论解决问题
```

```python
# AutoGen 风格：Agent 对话驱动
user_proxy.initiate_chat(
    assistant,
    message="帮我写一个排序算法并验证正确性"
)
# 后面的事情由 Agent 自己决定
```

### CrewAI：角色扮演，分工明确

```
设计哲学：模拟人类团队协作
核心抽象：Agent（角色） + Task（任务） + Crew（团队）
连接方式：crew.kickoff() 启动工作流
适合比喻：咨询公司 —— 各专业角色分工协作
```

```python
# CrewAI 风格：角色与任务定义驱动
crew = Crew(agents=[researcher, writer], tasks=[research, write])
result = crew.kickoff(inputs={"topic": "AI市场"})
```

---

## 三、能力矩阵：十维详细对比

### 3.1 基础能力

| 能力维度 | LangChain | LlamaIndex | AutoGen | CrewAI |
|---------|-----------|-----------|---------|--------|
| LLM 接入 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Prompt 管理 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 输出解析 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 流式输出 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### 3.2 数据处理

| 能力维度 | LangChain | LlamaIndex | AutoGen | CrewAI |
|---------|-----------|-----------|---------|--------|
| 文档加载 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 文本分割 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 向量存储 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 高级 RAG | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ |
| 知识图谱 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ |

### 3.3 Agent 与多 Agent

| 能力维度 | LangChain | LlamaIndex | AutoGen | CrewAI |
|---------|-----------|-----------|---------|--------|
| 单 Agent | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 多 Agent 协作 | ⭐⭐⭐（需LangGraph） | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 代码自动执行 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 人机协作 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 角色系统 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 3.4 工程与生产

| 能力维度 | LangChain | LlamaIndex | AutoGen | CrewAI |
|---------|-----------|-----------|---------|--------|
| 生态丰富度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 可观测性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 测试/评估 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 文档质量 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 学习曲线 | 中等 | 较陡 | 中等 | 平缓 |

---

## 四、典型应用场景匹配

### 场景 1：简单 RAG 问答系统

**需求**：给内部文档建立问答机器人，用户输入问题获得准确回答。

```
推荐：LlamaIndex（首选）或 LangChain

原因：
- LlamaIndex 提供更完善的文档处理流水线
- 内置丰富的检索策略（子问题分解、Small-to-Big）
- 自带评估工具，方便衡量 RAG 质量
```

```python
# LlamaIndex 快速实现
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

docs = SimpleDirectoryReader("./company_docs").load_data()
index = VectorStoreIndex.from_documents(docs)
engine = index.as_query_engine()
answer = engine.query("公司的休假政策是什么？")
```

### 场景 2：复杂工具调用 Agent

**需求**：构建一个能够调用 20+ 个工具的智能助手，工具包括数据库、API、文件系统等。

```
推荐：LangChain

原因：
- 最丰富的工具生态（600+ 集成）
- Function Calling Agent 效果稳定
- LCEL 让复杂工具链的编排简单直观
```

```python
# LangChain 工具 Agent
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun

tools = [DuckDuckGoSearchRun(), WikipediaQueryRun(), my_db_tool, my_api_tool]
agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

### 场景 3：代码生成与自动测试

**需求**：给定需求，自动生成代码、运行测试、迭代修复，直到通过所有测试。

```
推荐：AutoGen

原因：
- 原生代码执行环境（Docker 沙箱）
- "思考-执行-观察"循环天然适合代码任务
- 双 Agent（写代码+验证代码）模式效果优秀
```

```python
# AutoGen 代码助手
coder = AssistantAgent("coder", llm_config=llm_config)
executor = UserProxyAgent(
    "executor",
    code_execution_config={"use_docker": True}
)
executor.initiate_chat(
    coder,
    message="写一个 Web 爬虫，抓取 HackerNews 的 Top 10 文章"
)
```

### 场景 4：自动化内容生产流水线

**需求**：每天自动产出行业报告：搜集信息→分析→撰写→校对→发布。

```
推荐：CrewAI

原因：
- 角色定义让每个环节职责明确
- 任务依赖管理清晰（context 参数）
- 适合有固定流程的重复性任务
```

```python
# CrewAI 内容流水线
crew = Crew(
    agents=[researcher, analyst, writer, editor],
    tasks=[collect, analyze, write, review],
    process=Process.sequential,
)
daily_report = crew.kickoff(inputs={"date": "2024-12-01"})
```

### 场景 5：研究助手（综合场景）

**需求**：用户提问 → 分解子问题 → 并行搜索 → 综合分析 → 生成报告。

```
推荐：LangChain + LlamaIndex 组合

原因：
- LangChain 负责 Agent 编排和工具调用
- LlamaIndex 负责知识库检索（更专业）
- 两者 API 兼容，可无缝集成
```

---

## 五、性能与成本对比

### Token 消耗对比（典型任务）

| 框架 | 简单问答 | RAG 查询 | 多轮任务 |
|------|---------|---------|---------|
| LangChain | ~500 tokens | ~2000 tokens | ~5000 tokens |
| LlamaIndex | ~600 tokens | ~1500 tokens | ~4000 tokens |
| AutoGen | ~1000 tokens | ~3000 tokens | ~10000 tokens |
| CrewAI | ~1500 tokens | ~4000 tokens | ~15000 tokens |

> ⚠️ 多 Agent 框架（AutoGen/CrewAI）的 Token 消耗显著更高，因为 Agent 之间的对话都会消耗 Token。

### 响应速度对比

```
LangChain (简单链)：最快（<1s）
LlamaIndex (RAG)：中等（1-3s，受检索影响）
AutoGen (双Agent)：较慢（5-30s，多轮对话）
CrewAI (多Agent)：最慢（10-60s，依赖链长度）
```

### 降低成本的策略

```python
# 1. 分层使用模型（CrewAI 示例）
cheap_agent = Agent(role="数据整理", llm=ChatOpenAI(model="gpt-4o-mini"))
smart_agent = Agent(role="核心分析", llm=ChatOpenAI(model="gpt-4o"))

# 2. 开启缓存（LangChain）
from langchain.cache import SQLiteCache
import langchain
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

# 3. 控制 Agent 迭代次数（AutoGen）
executor = AgentExecutor(max_iterations=5)  # 默认15

# 4. 限制检索数量（LlamaIndex）
engine = index.as_query_engine(similarity_top_k=3)  # 不要太多
```

---

## 六、可扩展性与生产就绪度

### LangChain

**优势：**
- LangSmith 提供完整的可观测性（追踪、评估、监控）
- LangGraph 提供状态机级别的复杂工作流
- LangServe 一键部署为 API

```python
# LangSmith 集成（自动追踪）
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"

# 之后所有链的执行都会自动上报到 LangSmith
result = chain.invoke({"input": "Hello"})
```

### LlamaIndex

**优势：**
- 内置评估框架（忠实度、相关性、正确性）
- 支持异步执行，性能更好
- LlamaCloud 提供企业级托管服务

```python
# 并发处理（提升吞吐量）
import asyncio

async def process_queries(queries):
    tasks = [engine.aquery(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results
```

### AutoGen

**优势：**
- Docker 代码沙箱，安全可靠
- 人机协作支持，适合半自动场景
- AutoGen Studio 提供可视化界面

### CrewAI

**劣势（相对）：**
- 监控工具相对薄弱
- 任务结果的不确定性较高
- 较新，社区成熟度相对低

---

## 七、选型决策树

```
你的主要需求是什么？
│
├─→ 构建 RAG / 知识库问答
│   └─→ 文档类型复杂（多格式、大量）？
│       ├─→ 是 → LlamaIndex
│       └─→ 否 → LangChain 或 LlamaIndex 均可
│
├─→ 工具调用 / 函数执行
│   └─→ LangChain（工具生态最丰富）
│
├─→ 代码生成与自动执行
│   └─→ AutoGen（原生代码执行）
│
├─→ 多 Agent 协作
│   ├─→ 任务有明确角色分工（固定流程）？
│   │   └─→ 是 → CrewAI
│   └─→ Agent 需要自主对话和决策？
│       └─→ 是 → AutoGen
│
├─→ 需要生产级监控和可观测性
│   └─→ LangChain（LangSmith）
│
└─→ 复杂综合场景（RAG + Agent + 多工具）
    └─→ LangChain（主框架）+ LlamaIndex（RAG 模块）
```

---

## 八、组合使用策略

四个框架并非互斥，它们可以组合使用：

### 最佳组合 1：LangChain + LlamaIndex

```python
# 用 LlamaIndex 做 RAG，包装为 LangChain 工具
from llama_index.core import VectorStoreIndex
from langchain.tools import Tool

# LlamaIndex 索引
rag_engine = VectorStoreIndex.from_documents(docs).as_query_engine()

# 包装为 LangChain 工具
rag_tool = Tool(
    name="knowledge_base",
    description="查询内部知识库",
    func=lambda q: str(rag_engine.query(q))
)

# 在 LangChain Agent 中使用
from langchain.agents import create_openai_functions_agent
agent = create_openai_functions_agent(llm, [rag_tool, search_tool], prompt)
```

### 最佳组合 2：CrewAI + LangChain 工具

```python
# CrewAI Agent 使用 LangChain 工具
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool as crewai_tool

ddg = DuckDuckGoSearchRun()

@crewai_tool
def web_search(query: str) -> str:
    """搜索网络信息"""
    return ddg.run(query)

researcher = Agent(
    role="研究员",
    tools=[web_search],  # 使用 LangChain 工具
    ...
)
```

---

## 九、未来趋势与演进方向

| 框架 | 近期重点演进 | 未来方向 |
|------|------------|---------|
| LangChain | LangGraph 强化（状态机工作流）| 企业级编排平台 |
| LlamaIndex | LlamaCloud 云服务 | 结构化数据 + 非结构化数据统一 |
| AutoGen | AutoGen 2.0 重架构 | 分布式多 Agent 系统 |
| CrewAI | Flow 工作流编排 | 与更多工具生态集成 |

---

## 十、综合选型建议总结

### 推荐原则

**1. 新手入门**
→ 从 **LangChain** 开始，生态最完整，文档最丰富，概念最通用。

**2. 数据密集型应用**
→ **LlamaIndex**，尤其是需要处理大量文档、需要高质量 RAG 的场景。

**3. 代码辅助与研究自动化**
→ **AutoGen**，天然支持代码执行循环，非常适合需要验证结果的任务。

**4. 内容生产与固定流程自动化**
→ **CrewAI**，角色定义清晰，适合有明确分工的重复性工作流。

**5. 生产环境大规模部署**
→ **LangChain**（最成熟的生产工具链）+ 必要时引入 LlamaIndex。

### 一句话总结

- **LangChain**：万能胶水，啥都能做，是其他框架的好搭档
- **LlamaIndex**：RAG 专家，数据处理无可匹敌  
- **AutoGen**：对话协作，代码执行是绝活
- **CrewAI**：团队模拟，角色分工最清晰

---

## 小结

选型没有绝对答案，关键在于理解每个框架的"擅长什么"。建议：
1. **先用 LangChain 打基础**（生态理解 + 核心概念）
2. **遇到 RAG 瓶颈时引入 LlamaIndex**
3. **需要多 Agent 协作时尝试 AutoGen 或 CrewAI**
4. **生产环境优先考虑可观测性和稳定性**

下一篇（D7）将是综合实战：用 LangChain + AutoGen 搭建一个完整的研究助手系统，把本周所学融会贯通。
