---
layout: default
title: "W7D1 · 工具系统设计原则"
---

# 工具系统设计：让 Agent 拥有超能力

> **Week 7 · Day 1** | 难度：⭐⭐⭐

---

## 工具的本质

工具是 Agent 与外部世界交互的接口。没有工具，Agent 只是一个聊天机器人；有了工具，Agent 可以搜索网络、执行代码、调用 API、操作文件……

```
                    Agent
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────▼───┐   ┌────▼───┐   ┌────▼───┐
   │搜索工具 │   │代码工具 │   │API工具  │
   └────────┘   └────────┘   └────────┘
        │             │             │
   Google/Bing   Python VM    REST API
```

## LangChain 工具的三种实现方式

### 方式1：@tool 装饰器（最简单）

```python
from langchain.tools import tool
from typing import Optional

@tool
def search_web(query: str) -> str:
    """搜索网络并返回相关结果。适合需要最新信息的查询。"""
    # 实际实现应接真实搜索 API
    return f"搜索 '{query}' 的结果：这里是搜索到的内容..."

@tool
def calculate_expression(expression: str) -> str:
    """计算数学表达式。支持基本运算和Python math 模块函数。
    
    Args:
        expression: Python 数学表达式，如 '2 ** 10' 或 'math.sqrt(144)'
    """
    import math
    try:
        result = eval(expression, {"__builtins__": {}}, 
                     {"math": math, "abs": abs, "round": round,
                      "max": max, "min": min, "sum": sum})
        return str(result)
    except Exception as e:
        return f"计算失败：{e}"

# 测试
print(search_web.name)        # search_web
print(search_web.description) # 搜索网络并返回相关结果...
print(calculate_expression.invoke({"expression": "2 ** 10"}))  # 1024
```

### 方式2：StructuredTool（复杂参数）

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List

class FileReadInput(BaseModel):
    filepath: str = Field(description="要读取的文件路径")
    encoding: str = Field(default="utf-8", description="文件编码")
    max_lines: int = Field(default=100, description="最多读取的行数")

def read_file(filepath: str, encoding: str = "utf-8", max_lines: int = 100) -> str:
    """读取文件内容"""
    try:
        with open(filepath, encoding=encoding) as f:
            lines = f.readlines()[:max_lines]
        return "".join(lines)
    except FileNotFoundError:
        return f"文件不存在：{filepath}"
    except Exception as e:
        return f"读取失败：{e}"

file_read_tool = StructuredTool.from_function(
    func=read_file,
    name="read_file",
    description="读取文件内容。只能读取允许的路径下的文件。",
    args_schema=FileReadInput,
    return_direct=False  # 返回给 Agent 处理，而非直接输出给用户
)
```

### 方式3：继承 BaseTool（最灵活）

```python
from langchain.tools import BaseTool
from pydantic import BaseModel
from typing import Type, Any, Optional

class DatabaseQueryInput(BaseModel):
    query: str = Field(description="SQL 查询语句")
    database: str = Field(default="main", description="数据库名称")

class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = """查询内部数据库。
    支持 SELECT 语句查询业务数据。
    禁止使用 INSERT/UPDATE/DELETE/DROP 等写操作。"""
    args_schema: Type[BaseModel] = DatabaseQueryInput
    
    # 工具级别的安全检查
    allowed_tables: List[str] = ["users", "orders", "products"]
    
    def _run(self, query: str, database: str = "main") -> str:
        """同步执行"""
        # 安全检查
        query_upper = query.strip().upper()
        if not query_upper.startswith("SELECT"):
            return "错误：只允许 SELECT 查询"
        
        # 检查是否访问了非允许的表
        for word in query.lower().split():
            if word in ["drop", "delete", "insert", "update", "truncate"]:
                return "错误：不允许写操作"
        
        # 模拟查询
        return f"查询 '{query}' 返回了 10 条结果..."
    
    async def _arun(self, query: str, database: str = "main") -> str:
        """异步执行"""
        return self._run(query, database)

db_tool = DatabaseQueryTool()
```

## 工具描述的艺术

工具描述质量直接影响 Agent 是否会正确使用工具：

```python
# ❌ 差的描述
@tool
def get_weather(city: str) -> str:
    """获取天气"""  # 太简单，Agent 不知道什么时候用
    pass

# ✅ 好的描述
@tool  
def get_weather(city: str, unit: str = "celsius") -> str:
    """获取指定城市的当前天气信息。
    
    当用户询问天气、温度、气候、是否需要带伞等问题时使用此工具。
    
    Args:
        city: 城市名称，支持中文（如"北京"）和英文（如"Beijing"）
        unit: 温度单位，"celsius"（摄氏度）或"fahrenheit"（华氏度），默认摄氏度
    
    Returns:
        包含温度、天气状况、湿度、风速的字符串。
        如果城市无法识别，返回错误信息。
    
    示例：
        - get_weather("北京") → "北京：晴，25°C，湿度40%，风速12km/h"
        - get_weather("New York", "fahrenheit") → "New York: Cloudy, 72°F..."
    """
    pass
```

## 工具集管理

```python
from langchain.tools import Tool
from typing import List, Dict

class ToolRegistry:
    """工具注册表：集中管理所有工具"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, tool: BaseTool, category: str = "general"):
        self._tools[tool.name] = tool
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(tool.name)
    
    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)
    
    def get_by_category(self, category: str) -> List[BaseTool]:
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]
    
    def list_all(self) -> List[BaseTool]:
        return list(self._tools.values())
    
    def get_tools_for_task(self, task_description: str) -> List[BaseTool]:
        """根据任务描述选择合适的工具子集"""
        # 简单关键词匹配（生产中用 LLM 选择）
        selected = []
        for tool in self._tools.values():
            if any(kw in task_description.lower() 
                   for kw in tool.description.lower().split()[:10]):
                selected.append(tool)
        return selected or self.list_all()

# 初始化工具注册表
registry = ToolRegistry()
registry.register(search_web, "information")
registry.register(calculate_expression, "computation")
registry.register(db_tool, "data")
registry.register(file_read_tool, "filesystem")

print(f"已注册工具：{[t.name for t in registry.list_all()]}")
```

## 将工具绑定到 Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = registry.list_all()

# 使用 Tool Calling Agent（推荐，比 ReAct 更稳定）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，能够使用各种工具完成任务。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
    return_intermediate_steps=True  # 返回中间步骤用于调试
)

result = executor.invoke({"input": "计算 2024 年是哪一年的第几年（从2000年算起）"})
print(result["output"])
```

## 踩坑经验

### 坑1：工具描述中有歧义，Agent 选错工具

**解法**：在描述中明确"什么情况下用这个工具"和"什么情况不用"。

### 坑2：工具返回值太长，超出 context window

**解法**：工具函数内部截断输出，返回摘要而非全量数据。

```python
def truncate_output(output: str, max_chars: int = 2000) -> str:
    if len(output) > max_chars:
        return output[:max_chars] + f"\n...[输出已截断，共{len(output)}字符]"
    return output
```

### 坑3：工具调用失败导致 Agent 卡住

**解法**：每个工具必须有 try/except，永远返回字符串（包括错误信息），不抛出异常。

---

*W7D1 · 工具系统设计原则 | Agent + Claw 系列*
