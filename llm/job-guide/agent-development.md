---
layout: post
title: "Agent 开发实战"
track: "🤖 大模型"
---

# Agent 开发实战

> 2026年核心能力方向。Agent = 感知→规划→记忆→工具调用→执行→反思的完整闭环。

---

## Agent 核心闭环

```
用户输入
  ↓
感知（理解意图）
  ↓
规划（CoT/ReAct/计划分解）
  ↓
记忆查询（短期/长期/工具记忆）
  ↓
工具调用（搜索/代码/数据库/API）
  ↓
执行
  ↓
反思（结果是否满足目标？）
  ↓ 是
输出结果
  ↓ 否
修正 → 重新规划（迭代）
```

---

## 1. Function Calling（工具调用基础）

Function Calling 是 Agent 的核心机制，让 LLM 决定何时调用哪个工具。

```python
from openai import OpenAI
import json

client = OpenAI()

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索互联网获取最新信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "执行Python代码并返回结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要执行的Python代码"
                    }
                },
                "required": ["code"]
            }
        }
    }
]

def run_agent(user_query: str) -> str:
    """单轮Agent执行"""
    messages = [{"role": "user", "content": user_query}]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # 没有工具调用 = 最终回答
        if not message.tool_calls:
            return message.content
        
        # 执行工具调用
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            # 调用实际函数
            if func_name == "search_web":
                result = search_web(**func_args)
            elif func_name == "execute_python":
                result = execute_python(**func_args)
            else:
                result = f"未知工具: {func_name}"
            
            # 将工具结果加入对话
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

# 测试
answer = run_agent("2024年全球AI芯片市场规模是多少？帮我搜索并计算增长率")
print(answer)
```

---

## 2. LangChain Agent

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 工具列表
search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [
    Tool(
        name="web_search",
        func=search.run,
        description="搜索互联网获取最新信息。输入搜索关键词。"
    ),
    Tool(
        name="wikipedia",
        func=wikipedia.run,
        description="从Wikipedia查询知识。适合查询概念、历史、定义。"
    ),
    Tool(
        name="python_calculator",
        func=lambda x: str(eval(x)),
        description="执行数学计算。输入Python数学表达式，如 '2024 * 1.15'。"
    )
]

# ReAct Prompt（Thought→Action→Observation→...→Final Answer）
react_prompt = PromptTemplate.from_template("""你是一个智能助手，可以使用以下工具解决问题：

{tools}

使用以下格式：
Thought: 思考下一步做什么
Action: 工具名称（必须是 {tool_names} 之一）
Action Input: 工具输入
Observation: 工具返回结果
...（重复直到知道答案）
Thought: 我现在知道最终答案了
Final Answer: 最终答案

问题：{input}
{agent_scratchpad}""")

agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # 打印推理过程
    max_iterations=5,      # 最多5次工具调用
    handle_parsing_errors=True
)

result = agent_executor.invoke({"input": "OpenAI的GPT-4发布时间，以及当时的参数量是多少？"})
print(result["output"])
```

---

## 3. LangGraph 工作流 Agent（2026主流）

LangGraph 用有向图定义 Agent 的执行流程，比 ReAct 更可控、更适合生产。

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    step_count: int
    final_answer: str | None

# 节点函数
def call_llm(state: AgentState) -> AgentState:
    """调用LLM决策"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "step_count": state["step_count"] + 1,
        "final_answer": None
    }

def should_continue(state: AgentState) -> str:
    """判断是否继续执行工具"""
    last_message = state["messages"][-1]
    
    # 超过最大步数，强制结束
    if state["step_count"] >= 10:
        return "end"
    
    # 有工具调用，继续
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # 没有工具调用，结束
    return "end"

# 构建图
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_llm)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
workflow.add_edge("tools", "agent")  # 工具执行完回到agent

app = workflow.compile()

# 运行
result = app.invoke({
    "messages": [("user", "分析一下A股今天的市场情况")],
    "step_count": 0,
    "final_answer": None
})
print(result["messages"][-1].content)
```

---

## 4. 记忆管理

Agent 的记忆分三层：

```python
from langchain.memory import ConversationBufferWindowMemory, VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma

# 短期记忆：保留最近N轮对话
short_term_memory = ConversationBufferWindowMemory(
    k=10,                  # 保留最近10轮
    memory_key="history",
    return_messages=True
)

# 长期记忆：向量化存储，语义检索
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
memory_vectorstore = Chroma(
    embedding_function=embedding,
    persist_directory="./agent_memory"
)
long_term_memory = VectorStoreRetrieverMemory(
    retriever=memory_vectorstore.as_retriever(search_kwargs={"k": 3})
)

class AgentWithMemory:
    def __init__(self):
        self.short_memory = {}   # session_id → 最近对话
        self.long_memory = memory_vectorstore
    
    def remember(self, session_id: str, user_msg: str, agent_response: str):
        """存储对话到记忆"""
        # 短期：追加到列表
        if session_id not in self.short_memory:
            self.short_memory[session_id] = []
        self.short_memory[session_id].append({
            "user": user_msg,
            "agent": agent_response
        })
        # 保留最近10条
        self.short_memory[session_id] = self.short_memory[session_id][-10:]
        
        # 长期：向量化存储重要信息
        important_info = f"用户问：{user_msg}\n助手答：{agent_response}"
        self.long_memory.add_texts(
            [important_info],
            metadatas=[{"session_id": session_id}]
        )
    
    def recall(self, session_id: str, query: str) -> str:
        """检索相关记忆"""
        short = self.short_memory.get(session_id, [])[-5:]  # 最近5条
        long = self.long_memory.similarity_search(query, k=3)
        
        short_context = "\n".join([f"用户:{m['user']}\n助手:{m['agent']}" for m in short])
        long_context = "\n".join([doc.page_content for doc in long])
        
        return f"近期对话:\n{short_context}\n\n相关历史:\n{long_context}"
```

---

## 5. 多 Agent 协作（CrewAI）

```python
from crewai import Agent, Task, Crew, Process

# 定义专业Agent
researcher = Agent(
    role="市场研究员",
    goal="收集和分析市场数据，提供准确的市场洞察",
    backstory="你是一位经验丰富的市场研究员，擅长数据分析和趋势预测",
    tools=[search_tool, data_analysis_tool],
    llm=llm,
    verbose=True
)

writer = Agent(
    role="内容撰写师",
    goal="将研究结论转化为清晰、专业的报告",
    backstory="你是一位专业的商业写作专家，擅长将复杂数据转化为易懂的洞察报告",
    tools=[],
    llm=llm
)

# 定义任务（有依赖关系）
research_task = Task(
    description="调研2026年中国AI大模型行业市场规模、主要玩家、增长趋势",
    expected_output="结构化的市场调研报告，包含数据来源",
    agent=researcher,
)

writing_task = Task(
    description="基于市场调研结果，撰写一份500字的行业简报",
    expected_output="专业的行业简报，适合高管阅读",
    agent=writer,
    context=[research_task]   # 依赖研究任务的结果
)

# 组建团队并执行
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # 顺序执行（或parallel并行）
    verbose=True
)

result = crew.kickoff()
print(result.raw)
```

---

## 6. 错误处理与健壮性

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustAgent:
    
    @retry(
        stop=stop_after_attempt(3),           # 最多重试3次
        wait=wait_exponential(min=1, max=10)  # 指数退避
    )
    async def call_tool(self, tool_name: str, args: dict) -> str:
        """带重试的工具调用"""
        try:
            return await self.tools[tool_name](**args)
        except TimeoutError:
            raise  # 触发重试
        except ValueError as e:
            return f"参数错误: {e}"  # 不重试，直接返回错误
    
    async def run_with_fallback(self, query: str) -> str:
        """带降级的Agent执行"""
        try:
            # 主要路径：完整Agent
            return await self.full_agent.run(query)
        except Exception as e:
            # 降级：直接调用LLM（不用工具）
            print(f"Agent执行失败: {e}，降级到纯LLM")
            return await self.llm.ainvoke(query)
```

---

## 7. 面试高频问题

**Q: ReAct 和 CoT 有什么区别？**
- CoT（思维链）：只有思考，没有行动，适合纯推理任务
- ReAct：思考+行动+观察的闭环，可以使用外部工具，适合需要获取信息的任务

**Q: 什么时候用单Agent，什么时候用多Agent？**
- 单Agent：任务有清晰的顺序步骤，单一领域
- 多Agent：任务可并行分工，不同子任务需要不同专业能力，单Agent上下文太长

**Q: Agent的主要失败原因有哪些？**
1. 工具调用失败未处理（没有重试/降级）
2. 上下文过长导致遗忘早期信息
3. 规划层死循环（没有最大步数限制）
4. 工具描述不清导致LLM选错工具

---

[← RAG全链路工程化](./rag-engineering) | [→ Harness架构设计](./harness-architecture)
