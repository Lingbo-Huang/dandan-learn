---
layout: default
title: "D5 · Tool 与 ToolKit：自定义工具"
---

# D5 · Tool 与 ToolKit

> **Agent Week 3**  
> Agent 的能力来自工具。今天学如何定义工具，以及 ReAct Agent 如何使用工具。

---

## 一、工具的本质

工具 = 有名称、描述和执行逻辑的函数。

LLM 通过阅读工具描述来决定"用什么工具、传什么参数"。

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 最简单的自定义工具
@tool
def add(a: float, b: float) -> float:
    """将两个数字相加并返回结果。"""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """将两个数字相乘并返回结果。"""
    return a * b

# 工具的元数据
print(f"工具名称: {add.name}")         # add
print(f"工具描述: {add.description}")  # 将两个数字相加...
print(f"输入 Schema: {add.args}")     # {'a': ..., 'b': ...}
```

---

## 二、复杂工具（带 Pydantic 校验）

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional

class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="最多返回几条结果")
    language: Optional[str] = Field(default="zh", description="结果语言")

def web_search(query: str, max_results: int = 5, language: str = "zh") -> str:
    """
    在互联网上搜索信息。
    当需要获取最新信息、实时数据或模型训练截止日期后的内容时使用。
    """
    # 实际场景中对接真实搜索 API（如 Serpapi、Brave Search）
    return f"搜索 '{query}' 的模拟结果（{max_results} 条）：\n1. 相关结果1\n2. 相关结果2"

search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description="在互联网上搜索实时信息",
    args_schema=SearchInput,
    return_direct=False,
)
```

---

## 三、ReAct Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-4o-mini")

# 定义工具集
@tool
def get_current_time() -> str:
    """获取当前的日期和时间。"""
    from datetime import datetime
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持基本运算符 +, -, *, /, ** (幂), sqrt 等。
    示例：'2 ** 10'，'sqrt(144)'
    """
    import math
    try:
        # 安全地执行数学计算
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算出错：{e}"

tools = [get_current_time, calculator, search_tool]

# Agent 提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，可以使用各种工具来帮助用户。"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 创建 Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,      # 打印中间步骤
    max_iterations=5,  # 最多执行 5 步
)

# 测试
result = agent_executor.invoke({
    "input": "现在几点？然后计算 2 的 10 次方。"
})
print("\n最终答案:", result["output"])
```

---

## 四、工具编排：ReAct 的思考过程

```
用户：现在几点？然后计算 2 的 10 次方。

LLM 思考：
  Thought: 需要先获取时间，再计算。
  Action: get_current_time
  Action Input: {}
  Observation: 2026年05月13日 14:30:00
  
  Thought: 有了时间，现在计算。
  Action: calculator  
  Action Input: {"expression": "2 ** 10"}
  Observation: 1024
  
  Thought: 两个信息都有了，可以回答了。
  Final Answer: 现在是 2026年05月13日 14:30:00，2的10次方等于1024。
```

---

## 五、工具包（Toolkit）

```python
# Toolkit = 一组相关工具的集合
# 常见的官方 Toolkit：
# - GitHubToolkit: 操作 GitHub 仓库
# - SQLDatabaseToolkit: 查询 SQL 数据库
# - FileManagementToolkit: 文件操作
# - JsonToolkit: 操作 JSON 数据

from langchain_community.agent_toolkits import FileManagementToolkit

# 文件管理工具包（安全地限制在指定目录）
file_toolkit = FileManagementToolkit(
    root_dir="/tmp/agent_workspace",
    selected_tools=["read_file", "write_file", "list_directory"]
)

file_tools = file_toolkit.get_tools()
print(f"文件工具包包含 {len(file_tools)} 个工具:")
for tool in file_tools:
    print(f"  - {tool.name}: {tool.description[:50]}...")
```

---

## 六、工具调用的最佳实践

```python
# 1. 工具描述要清晰具体
@tool
def good_tool(query: str) -> str:
    """
    搜索公司内部知识库。
    适用场景：需要查询公司政策、产品文档、技术规范时使用。
    不适用：一般性知识问题（直接回答即可）。
    返回：相关文档片段列表。
    """
    pass

# 2. 工具要处理错误，给出有意义的错误信息
@tool
def robust_tool(data: str) -> str:
    """处理数据"""
    try:
        result = process(data)
        return f"成功：{result}"
    except ValueError as e:
        return f"输入格式错误：{e}。请检查输入格式后重试。"
    except Exception as e:
        return f"处理失败：{e}"

# 3. 工具应该是确定性的（相同输入，相同输出）
# 避免：工具内部有随机性
# 推荐：副作用在返回值中体现
```

---

## 今天的关键认识

1. **工具 = 名称 + 描述 + 函数**：描述决定 LLM 是否会用这个工具
2. **ReAct 循环**：Thought → Action → Observation → 继续 or Final Answer
3. **工具描述要具体**：什么时候用、不用，参数格式，返回什么
4. **工具要处理错误**：给 Agent 有意义的错误信息，它才能调整策略

---

## 明天预告

D6：**LangSmith**——调试、追踪、评估，Agent 开发的监控基础设施。
