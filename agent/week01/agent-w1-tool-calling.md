# D5 · 工具调用——Agent 连接外部世界

> **Week 1 主题**：什么是 Agent——定义 / ReAct / 规划 / 记忆 / 工具调用  
> **本日主题**：工具调用（Tool Calling / Function Calling）

---

## 🎯 学习目标

1. 理解 OpenAI Function Calling 的工作原理
2. 掌握工具定义、注册、执行的完整流程
3. 能构建具备多种工具的 Agent
4. 了解工具安全性和错误处理最佳实践

---

## 📚 核心知识点

### 1. 工具调用的本质

**LLM 本身不能执行代码，不能上网，不能读文件。**  
工具调用（Function Calling）是让 LLM **告诉程序"调用什么工具、传什么参数"**，实际执行由代码层完成。

```
用户: "今天北京天气怎么样？"

LLM 的思考：
  → 我需要调用天气工具
  → 参数: city="北京", date="today"

程序：
  1. 解析 LLM 的工具调用请求
  2. 真正调用天气 API
  3. 将结果返回给 LLM
  
LLM：基于API返回的数据生成回答
```

### 2. OpenAI Function Calling 格式

**工具定义**（JSON Schema）：
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定城市的天气信息",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "城市名称，如'北京'或'Shanghai'"
        },
        "date": {
          "type": "string",
          "description": "日期，格式 YYYY-MM-DD，默认今天"
        }
      },
      "required": ["city"]
    }
  }
}
```

**LLM 响应**（当决定调用工具时）：
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "id": "call_abc123",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"city\": \"北京\"}"
    }
  }]
}
```

**工具结果**（返回给 LLM）：
```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"temperature\": 22, \"weather\": \"晴\", \"humidity\": 45}"
}
```

### 3. 完整工具调用循环

```python
messages = [{"role": "user", "content": "北京今天天气？"}]

while True:
    # 调用 LLM
    response = llm.invoke(messages, tools=tools)
    
    # 没有工具调用 → 直接返回最终答案
    if not response.tool_calls:
        print(response.content)
        break
    
    # 有工具调用 → 执行工具
    for tool_call in response.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        # 执行工具
        result = execute_tool(tool_name, tool_args)
        
        # 将结果加入消息链
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
    
    # 继续循环，让 LLM 处理工具结果
```

### 4. 工具的五大类型

| 类型 | 示例 | 特点 |
|------|------|------|
| **信息检索** | 搜索、数据库查询、文件读取 | 只读，低风险 |
| **计算工具** | 计算器、代码执行、数据分析 | 需要沙箱隔离 |
| **外部服务** | 天气API、地图、支付 | 需要身份验证 |
| **系统操作** | 文件写入、邮件发送 | 高风险，需确认 |
| **Agent调用** | 子Agent、Claw Worker | 复杂任务编排 |

### 5. 工具安全性设计

**原则**：最小权限 + 明确范围 + 错误隔离

```python
# 好的工具设计
def read_file(path: str) -> str:
    """读取文件内容
    
    限制：
    - 只允许读取 /allowed_dir/ 下的文件
    - 文件大小限制 1MB
    - 不允许读取 .env、密钥文件
    """
    ALLOWED_DIR = "/allowed_dir"
    MAX_SIZE = 1024 * 1024  # 1MB
    
    # 路径安全检查
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(ALLOWED_DIR):
        raise PermissionError(f"不允许访问 {ALLOWED_DIR} 之外的文件")
    
    # 文件大小检查
    if os.path.getsize(abs_path) > MAX_SIZE:
        raise ValueError("文件过大，超出限制")
    
    with open(abs_path) as f:
        return f.read()
```

### 6. 并行工具调用

现代 LLM 支持在一次响应中请求多个工具调用（并行）：

```json
"tool_calls": [
  {"id": "call_1", "function": {"name": "search", "arguments": "..."}},
  {"id": "call_2", "function": {"name": "get_weather", "arguments": "..."}}
]
```

Agent 可以并行执行这些工具调用，显著减少延迟。

---

## 💡 示例/推导

### 示例：多工具协作完成复杂任务

**问题**："帮我查一下上海和北京明天的天气，推荐哪个城市更适合户外活动，并把结果保存到文件"

```
[Round 1]
LLM 决定: 并行查询两城市天气
Tool Calls:
  - get_weather(city="上海", date="tomorrow")
  - get_weather(city="北京", date="tomorrow")

Observations:
  - 上海: 小雨，17°C
  - 北京: 晴，22°C，微风

[Round 2]  
LLM 分析: 北京晴天更适合户外活动
Tool Call:
  - save_to_file(
      filename="weather_recommendation.txt",
      content="推荐去北京，明天晴天22°C，适合户外活动"
    )

Observation: 文件保存成功

[Round 3]
LLM 输出最终回答: "已分析完毕并保存到文件..."
```

---

## 🔧 动手练习

### 练习 1：定义工具 Schema（必做）

为以下工具手写 JSON Schema 定义（不用代码，直接写 JSON）：

1. `search_web(query, max_results=5)` - 网络搜索
2. `run_python(code)` - 执行 Python 代码
3. `send_email(to, subject, body, attachments=[])` - 发送邮件

注意：description 要足够清晰，让 LLM 知道何时使用。

### 练习 2：构建工具执行引擎（必做）

```python
# 创建文件: 09_tool_engine.py
# uv run python 09_tool_engine.py

import json
import math
import inspect
from typing import Callable, Any
from dataclasses import dataclass

@dataclass
class ToolResult:
    success: bool
    result: Any
    error: str = ""

class ToolEngine:
    """工具注册和执行引擎"""
    
    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._schemas: list[dict] = []
    
    def register(self, func: Callable) -> Callable:
        """通过装饰器注册工具"""
        tool_name = func.__name__
        self._tools[tool_name] = func
        
        # 自动从函数签名生成 schema
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            prop = {"type": "string"}  # 简化，实际需要类型映射
            if param.annotation == int:
                prop["type"] = "integer"
            elif param.annotation == float:
                prop["type"] = "number"
            elif param.annotation == bool:
                prop["type"] = "boolean"
            
            properties[param_name] = prop
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        self._schemas.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": func.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        })
        
        return func
    
    def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        """执行工具"""
        if tool_name not in self._tools:
            return ToolResult(False, None, f"工具 '{tool_name}' 不存在")
        
        try:
            result = self._tools[tool_name](**arguments)
            return ToolResult(True, result)
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    @property
    def schemas(self):
        return self._schemas

# 创建工具引擎并注册工具
engine = ToolEngine()

@engine.register
def calculator(expression: str) -> str:
    """计算数学表达式，支持基本运算和 math 模块函数"""
    # 安全计算：只允许数学运算
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith('_')}
    allowed_names['abs'] = abs
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

@engine.register
def string_processor(text: str, operation: str) -> str:
    """处理字符串：支持 upper/lower/reverse/length/count_words"""
    ops = {
        "upper": text.upper,
        "lower": text.lower,
        "reverse": lambda: text[::-1],
        "length": lambda: str(len(text)),
        "count_words": lambda: str(len(text.split())),
    }
    if operation not in ops:
        return f"不支持的操作: {operation}. 支持: {list(ops.keys())}"
    return ops[operation]()

@engine.register
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算：支持温度(C/F/K)、距离(km/mile/m)"""
    conversions = {
        ("C", "F"): lambda x: x * 9/5 + 32,
        ("F", "C"): lambda x: (x - 32) * 5/9,
        ("C", "K"): lambda x: x + 273.15,
        ("K", "C"): lambda x: x - 273.15,
        ("km", "mile"): lambda x: x * 0.621371,
        ("mile", "km"): lambda x: x * 1.60934,
        ("km", "m"): lambda x: x * 1000,
        ("m", "km"): lambda x: x / 1000,
    }
    key = (from_unit, to_unit)
    if key not in conversions:
        return f"不支持 {from_unit} → {to_unit} 的换算"
    result = conversions[key](value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"

# 测试
print("=== 工具引擎测试 ===\n")
print(f"已注册工具: {[s['function']['name'] for s in engine.schemas]}\n")

test_cases = [
    ("calculator", {"expression": "math.sqrt(2) * 100"}),
    ("string_processor", {"text": "Hello Agent World", "operation": "reverse"}),
    ("unit_converter", {"value": 100.0, "from_unit": "km", "to_unit": "mile"}),
    ("calculator", {"expression": "1/0"}),  # 错误测试
]

for tool_name, args in test_cases:
    result = engine.execute(tool_name, args)
    status = "✅" if result.success else "❌"
    print(f"{status} {tool_name}({args})")
    if result.success:
        print(f"   结果: {result.result}")
    else:
        print(f"   错误: {result.error}")
    print()
```

### 练习 3：带工具的完整 Agent（核心练习）

```python
# 创建文件: 10_tool_agent.py
# uv run python 10_tool_agent.py

from openai import OpenAI
import json, math
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# 工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算，支持基本运算和 sqrt, pow, log 等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

def calculate(expression: str) -> str:
    safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith('_')}
    try:
        return str(eval(expression, {"__builtins__": {}}, safe_env))
    except Exception as e:
        return f"Error: {e}"

def get_current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

TOOL_MAP = {"calculate": calculate, "get_current_time": get_current_time}

def run_agent(user_input: str):
    """运行带工具调用的 Agent"""
    messages = [{"role": "user", "content": user_input}]
    
    print(f"用户: {user_input}")
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        
        if not msg.tool_calls:
            print(f"AI: {msg.content}")
            break
        
        # 执行所有工具调用
        messages.append(msg)
        
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            
            print(f"  🔧 调用工具: {fn_name}({fn_args})")
            result = TOOL_MAP[fn_name](**fn_args)
            print(f"  📤 工具返回: {result}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

if __name__ == "__main__":
    questions = [
        "现在几点了？计算一下从今天0点到现在过了多少分钟？",
        "计算 2^10 + sqrt(144) 的结果",
    ]
    for q in questions:
        print("\n" + "="*50)
        run_agent(q)
```

### 🦞 Claw 实战：Claw 的工具调用体系

Claw Agent 的工具调用就是 `curl` 命令调用平台 API：

| Claw 工具 | 对应类型 |
|-----------|---------|
| `GET /claw/room/tasks/{taskId}` | 信息检索工具 |
| `PATCH /claw/room/tasks/{taskId}` | 系统操作工具 |
| `GET /claw/room/files/{fileId}/download` | 文件工具 |
| Shell 命令 | 代码执行工具 |

**实战任务**：
1. 统计本任务执行过程中调用了多少次 API
2. 按工具类型分类（只读 vs 写入）
3. 思考：Claw 如何保证工具调用的安全性？

---

## 📝 小结

| 要点 | 核心内容 |
|------|---------|
| **本质** | LLM 输出调用请求，代码层真正执行 |
| **格式** | JSON Schema 定义 + tool_calls 响应 |
| **循环** | LLM → 工具调用 → 结果注入 → LLM → ... |
| **安全** | 最小权限、路径检查、错误隔离 |
| **并行** | 一次响应可包含多个工具调用 |

**明天预告**：多 Agent 协作——当一个 Agent 不够用时，如何让多个 Agent 协同工作？

---

> 💡 **今日思考题**：工具的 `description` 字段非常重要——它直接影响 LLM 是否会正确选择工具。你能想到什么方法来测试和优化工具描述的质量？
