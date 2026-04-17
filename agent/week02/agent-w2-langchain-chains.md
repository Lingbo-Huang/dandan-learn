# LangChain 链与工具：LCEL、Tool 绑定与 Memory

> **Day 2** · 预计学习时间：4-5 小时  
> **目标**：掌握 LCEL 深层用法，构建带工具调用和记忆的完整 Agent

---

## 框架概念

### LCEL：LangChain Expression Language

LCEL 是 LangChain 在 2023 年推出的核心 DSL，用 `|` 管道符连接各组件。

**核心设计原则：**
1. **每个组件都是 Runnable** — 实现 `invoke / stream / batch / astream` 四种接口
2. **组合即链** — `A | B` 意味着 A 的输出成为 B 的输入
3. **并行天然支持** — `RunnableParallel` 让多个分支同时执行

```
输入
  ↓
PromptTemplate.invoke(input) → PromptValue
  ↓
ChatOpenAI.invoke(prompt) → AIMessage  
  ↓
StrOutputParser.invoke(message) → str
  ↓
输出
```

### Tool（工具）

Tool 是 LangChain Agent 能调用的"外部能力"：

- **内置工具**：搜索、Python REPL、文件系统、Wikipedia、Wolfram Alpha 等
- **自定义工具**：用 `@tool` 装饰器将任意 Python 函数变成工具
- **工具绑定**：`llm.bind_tools(tools)` 让 LLM 学会在合适时候调用工具

### Memory（记忆）

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| `InMemoryHistory` | 存在内存，进程结束即消失 | 测试、短会话 |
| `FileChatMessageHistory` | 存文件 | 单机持久化 |
| `RedisChatMessageHistory` | 存 Redis | 生产多用户场景 |
| `ConversationSummaryMemory` | 超过长度则自动摘要 | 长对话 |

---

## 核心代码示例（使用 uv）

### 环境准备

```bash
uv init langchain-chains
cd langchain-chains
uv add langchain langchain-openai langchain-community tavily-python python-dotenv
echo 'OPENAI_API_KEY=your-key-here' >> .env
echo 'TAVILY_API_KEY=your-key-here' >> .env  # 可选，用于搜索工具
```

### 示例 1：LCEL 高级用法

```python
# lcel_advanced.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 并行链：同时生成摘要和关键词
summary_prompt = ChatPromptTemplate.from_template(
    "用 50 字概括这段文字：{text}"
)
keywords_prompt = ChatPromptTemplate.from_template(
    "从这段文字提取 5 个关键词（逗号分隔）：{text}"
)

summary_chain = summary_prompt | llm | StrOutputParser()
keywords_chain = keywords_prompt | llm | StrOutputParser()

# 并行执行两个链
parallel_chain = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain,
    original=RunnablePassthrough()  # 原样传递输入
)

text = """
LangChain 是一个用于开发由语言模型驱动的应用程序的框架。
它使语言模型能够连接到其他数据源、与环境互动，
从而构建真正强大的 AI 应用。
"""

result = parallel_chain.invoke({"text": text})

print("📄 原文：", result["original"]["text"][:30], "...")
print("📝 摘要：", result["summary"])
print("🏷️  关键词：", result["keywords"])
```

### 示例 2：自定义 Tool + ReAct Agent

```python
# tool_agent.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import json, math

load_dotenv()

# 自定义工具
@tool
def calculate(expression: str) -> str:
    """计算数学表达式。输入：Python 数学表达式字符串，如 '2 ** 10' 或 'math.sqrt(144)'。"""
    try:
        # 安全的数学计算
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

@tool
def get_word_count(text: str) -> str:
    """统计文本字数。输入：任意文本字符串。"""
    chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_words = len([w for w in text.split() if w.isalpha()])
    return json.dumps({
        "中文字数": chinese_count,
        "英文单词数": english_words,
        "总字符数": len(text)
    }, ensure_ascii=False)

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算。支持：km/m/cm/mm，kg/g，C/F（摄氏/华氏）"""
    conversions = {
        ("km", "m"): lambda x: x * 1000,
        ("m", "km"): lambda x: x / 1000,
        ("kg", "g"): lambda x: x * 1000,
        ("g", "kg"): lambda x: x / 1000,
        ("C", "F"): lambda x: x * 9/5 + 32,
        ("F", "C"): lambda x: (x - 32) * 5/9,
    }
    key = (from_unit, to_unit)
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"不支持 {from_unit} 到 {to_unit} 的换算"

# 工具列表
tools = [calculate, get_word_count, unit_converter]

# 绑定工具的 LLM（Function Calling 模式）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ReAct 提示模板
react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

# 创建 Agent
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 测试
result = agent_executor.invoke({
    "input": "25 摄氏度是多少华氏度？同时计算 2 的 16 次方是多少？"
})
print("\n最终答案：", result["output"])
```

### 示例 3：带 Memory 的对话 Agent

```python
# memory_agent.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 带历史消息占位符的 prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位友善的 AI 助手，记住对话历史。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# 手动维护对话历史（最简实现）
class SimpleConversation:
    def __init__(self, chain, max_history=10):
        self.chain = chain
        self.history = []
        self.max_history = max_history
    
    def chat(self, user_input: str) -> str:
        response = self.chain.invoke({
            "history": self.history[-self.max_history:],
            "input": user_input
        })
        
        # 更新历史
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response))
        
        return response

# 使用
conv = SimpleConversation(chain)

print("开始对话（输入 'quit' 退出）")
while True:
    user_input = input("\n你：")
    if user_input.lower() == 'quit':
        break
    response = conv.chat(user_input)
    print(f"AI：{response}")
```

### 示例 4：工具调用的现代写法（Tool Calling）

```python
# modern_tool_calling.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from typing import Any

load_dotenv()

@tool
def search_weather(city: str) -> dict:
    """查询城市天气（模拟）"""
    mock_data = {
        "北京": {"temp": 18, "condition": "晴", "humidity": 45},
        "上海": {"temp": 22, "condition": "多云", "humidity": 75},
        "深圳": {"temp": 28, "condition": "小雨", "humidity": 85},
    }
    return mock_data.get(city, {"error": f"未找到 {city} 的天气数据"})

@tool  
def book_restaurant(city: str, cuisine: str, people: int) -> str:
    """预订餐厅（模拟）"""
    return f"✅ 已在{city}预订{cuisine}餐厅，{people}人，预计等待15分钟"

tools = [search_weather, book_restaurant]
tool_map = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def run_agent(user_message: str):
    messages = [HumanMessage(content=user_message)]
    
    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            print("最终回答：", response.content)
            break
        
        # 执行工具调用
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"🔧 调用工具：{tool_name}({tool_args})")
            tool_result = tool_map[tool_name].invoke(tool_args)
            
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            ))

run_agent("查一下北京天气，如果适合出门，帮我预订一家北京的北京菜餐厅，3个人")
```

---

## 与 Claw 的对比与联系

**LCEL 管道 vs Claw Task 流水线：**

| 特性 | LCEL 管道 | Claw Task 链 |
|------|-----------|-------------|
| 组合方式 | `A \| B \| C`（代码级） | Task 依赖关系（配置级） |
| 执行粒度 | 函数调用级别 | 完整 Agent 会话级别 |
| 状态管理 | 手动或 Memory 对象 | 平台管理（Project/Task 状态） |
| 可观测性 | LangSmith 追踪 | Claw 任务日志 |
| 失败处理 | try/except | Task error 状态 + 项目会话介入 |

**工具调用 vs Skill：**
- LangChain 的 `@tool` 装饰器 ≈ Claw 的 Skill 定义
- 两者都是"给 AI 添加能力"的机制
- 区别：`@tool` 在代码运行时可用；Claw Skill 在平台层面可发现、可共享

**Memory vs Session：**
- LangChain Memory 是**单个应用内**的对话历史管理
- Claw Session 是**跨多个 Agent**的对话上下文管理
- 规模不同，但解决的核心问题类似：让 AI 记住"我们谈到哪儿了"

---

## 小结

- **LCEL** 是 LangChain 的灵魂——会用 `|` 串链，就掌握了 80% 的 LangChain
- **工具调用**有两种模式：ReAct（文本推理）和 Function Calling（更现代）；生产中优先用 Function Calling
- **Memory** 本质上是把历史消息列表塞进 Prompt——理解这一点就不会迷失在各种 Memory 类中
- **RunnableParallel** 是性能优化利器，多个独立查询可以并发执行

**下一步** → Day 3：LlamaIndex——以数据为核心的 RAG 框架
