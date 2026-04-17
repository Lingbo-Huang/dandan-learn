# D2 LangChain 链与工具使用

> **学习目标**：掌握 LangChain 中链（Chain）的各种模式，学会定义和使用工具（Tool），理解 Agent 如何通过工具完成复杂任务。

---

## 一、从链到 Agent 的演进

D1 介绍了基础组件。现在思考一个问题：如果我们要构建一个能"自己决定下一步做什么"的 AI，该怎么做？

这就是 LangChain **链与工具**体系的核心命题：
- **链（Chain）**：预定义的固定流程，开发者决定步骤顺序
- **Agent**：LLM 自主决定调用哪些工具、按什么顺序，是真正的"智能体"

本篇从简单的链开始，逐步走向 Agent。

---

## 二、LCEL 进阶：复杂链模式

### 2.1 顺序链（Sequential Chain）

最基础的链：A → B → C

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

# 第一个链：将文章转为要点
summarize_chain = (
    ChatPromptTemplate.from_template(
        "请将以下文章提炼为3-5个核心要点，每点一行：\n\n{article}"
    )
    | llm
    | StrOutputParser()
)

# 第二个链：将要点翻译为英文
translate_chain = (
    ChatPromptTemplate.from_template(
        "请将以下要点翻译为英文：\n\n{points}"
    )
    | llm
    | StrOutputParser()
)

# 用 RunnableSequence 串联（或直接用 |）
from langchain_core.runnables import RunnableSequence, RunnablePassthrough

# 方式一：字典传递中间结果
pipeline = (
    {"points": summarize_chain, "original": RunnablePassthrough()}
    | translate_chain
)

# 方式二：使用 itemgetter 精确控制
from operator import itemgetter

pipeline_v2 = (
    summarize_chain
    | (lambda x: {"points": x})  # 转换格式
    | translate_chain
)

result = pipeline_v2.invoke({
    "article": "LangChain 是一个强大的 LLM 应用开发框架，..."
})
```

### 2.2 条件分支链（Router Chain）

根据输入内容选择不同的处理路径：

```python
from langchain_core.runnables import RunnableLambda, RunnableBranch

# 不同主题的专业链
tech_chain = (
    ChatPromptTemplate.from_template("作为技术专家，回答：{question}")
    | llm | StrOutputParser()
)

law_chain = (
    ChatPromptTemplate.from_template("作为法律专家，回答：{question}")
    | llm | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_template("请回答：{question}")
    | llm | StrOutputParser()
)

# 分类链：判断问题类型
classify_chain = (
    ChatPromptTemplate.from_template(
        """判断以下问题属于哪个类别，只输出类别名称（tech/law/general）：
        
问题：{question}"""
    )
    | llm
    | StrOutputParser()
    | (lambda x: x.strip().lower())
)

# 路由分支
def route(info):
    category = classify_chain.invoke({"question": info["question"]})
    if "tech" in category:
        return tech_chain
    elif "law" in category:
        return law_chain
    else:
        return general_chain

router_chain = RunnableLambda(
    lambda x: route(x).invoke(x)
)

# 测试
print(router_chain.invoke({"question": "Python 中的 GIL 是什么？"}))
print(router_chain.invoke({"question": "劳动合同可以口头签订吗？"}))
```

### 2.3 Map-Reduce 链

处理大量文档的经典模式：

```python
from langchain_core.runnables import RunnableParallel

documents = [
    "文档1：LangChain 提供了丰富的工具集...",
    "文档2：向量数据库是 RAG 的核心...",
    "文档3：Agent 能够自主决策...",
]

# Map 阶段：并行处理每个文档
map_chain = ChatPromptTemplate.from_template(
    "请提取以下文档的关键信息（1-2句话）：\n{doc}"
) | llm | StrOutputParser()

# Reduce 阶段：汇总所有摘要
reduce_chain = ChatPromptTemplate.from_template(
    "请根据以下各文档的摘要，写出一个综合性结论：\n\n{summaries}"
) | llm | StrOutputParser()

# 执行 Map-Reduce
summaries = [map_chain.invoke({"doc": doc}) for doc in documents]
final_result = reduce_chain.invoke({"summaries": "\n\n".join(summaries)})
```

---

## 三、工具（Tool）体系

### 3.1 什么是 Tool？

Tool 是 Agent 可以调用的"能力单元"，本质是一个带有描述信息的函数：

```python
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field

# 方式一：@tool 装饰器（最简单）
@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气情况"""
    # 实际应用中调用天气 API
    weather_data = {
        "北京": "晴天，25°C，北风3级",
        "上海": "多云，28°C，东风2级",
        "广州": "小雨，30°C，南风1级",
    }
    return weather_data.get(city, f"{city}的天气数据暂不可用")

# 查看工具元数据
print(f"工具名称：{get_weather.name}")
print(f"工具描述：{get_weather.description}")
print(f"参数Schema：{get_weather.args_schema.schema()}")

# 直接调用
result = get_weather.invoke({"city": "北京"})
print(result)


# 方式二：StructuredTool（复杂参数场景）
class CalculatorInput(BaseModel):
    operation: str = Field(description="运算类型：add/subtract/multiply/divide")
    a: float = Field(description="第一个操作数")
    b: float = Field(description="第二个操作数")

def calculator_func(operation: str, a: float, b: float) -> str:
    """执行基础数学运算"""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "错误：除数不能为0",
    }
    result = ops.get(operation, "不支持的运算")
    return f"{a} {operation} {b} = {result}"

calculator = StructuredTool.from_function(
    func=calculator_func,
    name="calculator",
    description="执行基础数学运算（加减乘除）",
    args_schema=CalculatorInput,
)
```

### 3.2 内置工具集

LangChain 内置了大量开箱即用的工具：

```python
# 搜索工具
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="zh"))

# 文件系统工具
from langchain_community.tools.file_management import (
    ReadFileTool, WriteFileTool, ListDirectoryTool
)

# Python 代码执行工具（谨慎使用！）
from langchain_experimental.tools import PythonREPLTool

python_repl = PythonREPLTool()
result = python_repl.invoke("print(sum(range(100)))")
# 输出：4950

# 数学计算工具
from langchain_community.tools import WolframAlphaQueryRun
```

### 3.3 异步工具

```python
from langchain_core.tools import BaseTool
import asyncio
import aiohttp

class AsyncWeatherTool(BaseTool):
    name: str = "async_weather"
    description: str = "异步获取天气数据"
    
    def _run(self, city: str) -> str:
        """同步版本（必须实现）"""
        return asyncio.run(self._arun(city))
    
    async def _arun(self, city: str) -> str:
        """异步版本"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.weather.com/{city}") as resp:
                data = await resp.json()
                return f"{city}: {data.get('temperature')}°C"
```

---

## 四、构建 Agent

Agent 是 LangChain 的精华，它让 LLM 成为"思考者"，工具成为"执行者"。

### 4.1 ReAct Agent（推理+行动）

ReAct（Reasoning + Acting）是最经典的 Agent 模式：

```
Thought: 我需要先查询天气
Action: get_weather
Action Input: {"city": "北京"}
Observation: 晴天，25°C，北风3级
Thought: 我已经获取了天气信息，现在可以回答
Final Answer: 北京今天天气晴好，25摄氏度...
```

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# 加载 ReAct 提示词模板（来自 LangChain Hub）
react_prompt = hub.pull("hwchase17/react")

# 定义工具集
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}：晴天，25°C"

@tool
def search_web(query: str) -> str:
    """搜索网络信息"""
    return f"关于'{query}'的搜索结果：这是一个示例结果..."

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)  # 实际生产中请用安全的解析器
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

tools = [get_weather, search_web, calculate]

# 创建 ReAct Agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(llm, tools, react_prompt)

# 用 AgentExecutor 运行（处理循环逻辑）
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # 打印思考过程
    max_iterations=5,      # 最大循环次数（防止死循环）
    handle_parsing_errors=True,  # 自动处理输出格式错误
)

result = executor.invoke({
    "input": "北京今天天气怎么样？如果温度超过20度，帮我计算 25 * 3.14 的值"
})
print(result["output"])
```

### 4.2 OpenAI Function Calling Agent（推荐）

现代 OpenAI 模型原生支持 Function Calling，效果更稳定：

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 创建支持 function calling 的 agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 提示词（需要 agent_scratchpad 占位符）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以使用工具帮助用户解决问题"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 多轮对话
chat_history = []
questions = [
    "北京现在天气怎么样？",
    "那我需要带伞吗？",
]

for question in questions:
    result = executor.invoke({
        "input": question,
        "chat_history": chat_history,
    })
    # 更新历史
    from langchain_core.messages import HumanMessage, AIMessage
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=result["output"]),
    ])
    print(f"Q: {question}")
    print(f"A: {result['output']}\n")
```

---

## 五、RAG 链：检索增强生成

RAG（Retrieval-Augmented Generation）是 LangChain 最常见的使用场景：

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 加载并索引文档
loader = WebBaseLoader("https://python.langchain.com/docs/introduction")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 2. 构建 RAG Chain
rag_prompt = ChatPromptTemplate.from_template("""
你是一个专业的问答助手。请根据以下上下文内容回答用户问题。

如果上下文中没有相关信息，请如实说明，不要凭空捏造。

上下文：
{context}

问题：{question}

回答：""")

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# LCEL 构建 RAG Chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# 调用
answer = rag_chain.invoke("LangChain 的 LCEL 是什么？")
print(answer)
```

### 带引用来源的 RAG

```python
from langchain_core.runnables import RunnableParallel

# 同时返回答案和来源文档
rag_chain_with_source = RunnableParallel(
    answer=rag_chain,
    sources=(retriever | format_docs),
    question=RunnablePassthrough(),
)

result = rag_chain_with_source.invoke("LangChain 支持哪些 LLM？")
print(f"答案：{result['answer']}")
print(f"来源：{result['sources'][:500]}...")
```

---

## 六、自定义链：LangChain Callbacks

Callbacks 让你能监控链的执行过程，用于日志、监控、调试：

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List

class LoggingCallback(BaseCallbackHandler):
    """自定义日志回调"""
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        print(f"🚀 LLM 开始调用，提示词长度：{len(prompts[0])} 字符")
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        print(f"✅ LLM 调用完成，消耗 Token：{usage.get('total_tokens', 'N/A')}")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        print(f"🔧 工具调用：{serialized['name']}，输入：{input_str}")
    
    def on_tool_end(self, output: str, **kwargs):
        print(f"📤 工具返回：{output[:100]}...")
    
    def on_chain_error(self, error: Exception, **kwargs):
        print(f"❌ 链执行错误：{error}")

# 使用 callback
callback = LoggingCallback()
chain_with_logging = chain.with_config(callbacks=[callback])
result = chain_with_logging.invoke({"input": "Hello"})
```

---

## 七、Streaming（流式输出）

实现实时流式响应，提升用户体验：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("请写一篇关于{topic}的短文，约200字")
    | ChatOpenAI(model="gpt-4o-mini", streaming=True)
    | StrOutputParser()
)

# 流式输出
print("开始生成：", end="")
for chunk in chain.stream({"topic": "人工智能的未来"}):
    print(chunk, end="", flush=True)
print("\n生成完成")

# 异步流式输出（FastAPI/异步场景）
import asyncio

async def stream_response():
    async for chunk in chain.astream({"topic": "量子计算"}):
        print(chunk, end="", flush=True)

asyncio.run(stream_response())
```

---

## 八、实战案例：多工具研究助手

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from datetime import datetime

@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息"""
    # 实际应用中接入真实搜索 API
    return f"关于'{query}'的最新信息：[搜索结果摘要]"

@tool
def calculate_stats(data: str) -> str:
    """计算一组数字的统计信息（用逗号分隔）"""
    nums = [float(x.strip()) for x in data.split(",")]
    return (
        f"数量：{len(nums)}\n"
        f"均值：{sum(nums)/len(nums):.2f}\n"
        f"最大值：{max(nums)}\n"
        f"最小值：{min(nums)}"
    )

@tool
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

@tool
def write_report(title: str, content: str) -> str:
    """将研究报告写入文件"""
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{content}")
    return f"报告已保存至 {filename}"

tools = [web_search, calculate_stats, get_current_time, write_report]

llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的研究助手。
    - 主动使用工具收集信息
    - 数据分析要精确
    - 最终输出结构化报告"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=8)

result = executor.invoke({
    "input": "请研究一下 LangChain 框架的最新动态，并分析数字 [85, 92, 78, 95, 88] 的统计数据，最后写一份简短报告"
})
print(result["output"])
```

---

## 九、链的性能优化

```python
# 1. 批量处理（并行调用）
inputs = [{"topic": t} for t in ["AI", "区块链", "量子计算"]]
results = chain.batch(inputs, config={"max_concurrency": 3})

# 2. 缓存（避免重复调用）
from langchain.cache import InMemoryCache
import langchain
langchain.llm_cache = InMemoryCache()

# 3. 超时设置
chain_with_timeout = chain.with_config({"timeout": 30})

# 4. 后备模型（Fallback）
primary_llm = ChatOpenAI(model="gpt-4o")
fallback_llm = ChatOpenAI(model="gpt-4o-mini")
robust_llm = primary_llm.with_fallbacks([fallback_llm])
```

---

## 小结

| 模式 | 适用场景 | 复杂度 |
|------|---------|--------|
| 顺序链（LCEL `|`） | 固定步骤流水线 | ⭐ |
| 条件分支链 | 根据内容路由 | ⭐⭐ |
| RAG 链 | 知识库问答 | ⭐⭐ |
| ReAct Agent | 需要推理的任务 | ⭐⭐⭐ |
| Function Calling Agent | 工具调用（推荐） | ⭐⭐⭐ |
| 自定义 Agent | 特殊决策逻辑 | ⭐⭐⭐⭐ |

下一篇（D3）将介绍 LlamaIndex —— 另一个专注于 RAG 的框架，看看它与 LangChain 有何不同。
