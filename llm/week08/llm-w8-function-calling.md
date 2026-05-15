---
layout: default
title: "D2 · Function Calling 与 LLM Agent"
render_with_liquid: false
---

# D2 · Function Calling 与 LLM Agent

> **Function Calling 让 LLM 从"会说话"变成"会做事"**——它是 Agent 系统的核心能力。

---

## 一、Function Calling 原理

### 1.1 底层机制

```python
"""
Function Calling 的本质：

1. 将工具描述（JSON Schema）注入 System Prompt
2. LLM 生成特殊格式的 JSON（工具调用请求）
3. 应用层拦截 JSON，执行真实函数
4. 将结果返回给 LLM 继续对话

注意：LLM 本身没有"执行代码"的能力，
     这一切都是应用层做的，LLM 只是生成了"调用说明"
"""

import openai
import json
from typing import Any

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="token")

# 定义工具
def get_weather(city: str, unit: str = "celsius") -> dict:
    """模拟天气 API"""
    weather_data = {
        "北京": {"temperature": 25, "condition": "晴天", "humidity": 45},
        "上海": {"temperature": 28, "condition": "多云", "humidity": 70},
    }
    data = weather_data.get(city, {"temperature": 20, "condition": "未知"})
    if unit == "fahrenheit":
        data["temperature"] = data["temperature"] * 9/5 + 32
    data["city"] = city
    data["unit"] = unit
    return data

def search_web(query: str) -> list[dict]:
    """模拟搜索 API"""
    return [
        {"title": f"关于 {query} 的结果", "url": "https://example.com", "snippet": "..."},
    ]

# Tool 定义（JSON Schema 格式）
TOOLS = [
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
                        "description": "城市名称，例如：北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位",
                        "default": "celsius"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "在互联网上搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Tool 执行器
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "search_web": search_web,
}

def execute_tool(tool_call) -> str:
    """执行工具调用"""
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    if func_name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Unknown function: {func_name}"})
    
    result = TOOL_FUNCTIONS[func_name](**args)
    return json.dumps(result, ensure_ascii=False)
```

### 1.2 完整对话循环

```python
class ToolUseAgent:
    """支持 Function Calling 的 Agent"""
    
    def __init__(self, model: str = "qwen2.5-7b", tools: list = None):
        self.model = model
        self.tools = tools or TOOLS
        self.max_turns = 10  # 最大工具调用轮次
    
    def run(self, user_message: str) -> str:
        """运行 Agent，自动处理工具调用循环"""
        messages = [
            {
                "role": "system",
                "content": "你是一个有用的助手，可以使用工具来获取实时信息。"
            },
            {"role": "user", "content": user_message}
        ]
        
        for turn in range(self.max_turns):
            # 调用 LLM
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",  # "auto" | "required" | "none"
            )
            
            choice = response.choices[0]
            
            # 检查是否需要调用工具
            if choice.finish_reason == "tool_calls":
                # 将 LLM 的决策加入历史
                messages.append({
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": [tc.model_dump() for tc in choice.message.tool_calls]
                })
                
                # 执行所有工具调用
                for tool_call in choice.message.tool_calls:
                    print(f"  🔧 调用工具: {tool_call.function.name}({tool_call.function.arguments})")
                    
                    result = execute_tool(tool_call)
                    
                    print(f"  📊 工具结果: {result[:100]}")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            
            elif choice.finish_reason == "stop":
                # LLM 给出最终答案
                return choice.message.content
            
            else:
                break
        
        return "Agent 达到最大轮次限制"

# 使用示例
agent = ToolUseAgent()
print(agent.run("北京今天天气怎么样？"))
print(agent.run("帮我搜索一下最新的大模型论文"))
```

---

## 二、ReAct Agent

```python
"""
ReAct (Reasoning + Acting) - Yao et al., 2022

让 LLM 交替进行：
  Thought（思考）：分析当前状态，决定下一步
  Action（行动）：调用工具
  Observation（观察）：获取工具结果
  ... 循环直到 Final Answer

比纯 Function Calling 更透明，LLM 会"说出"推理过程
"""

REACT_SYSTEM = """你是一个能够推理和使用工具的智能助手。

请按以下格式逐步解决问题：

Thought: [你的思考过程]
Action: [工具名称]
Action Input: [工具参数，JSON格式]
Observation: [工具返回的结果]
... (可重复多次)
Thought: 我现在有足够的信息了
Final Answer: [最终答案]

可用工具：
- get_weather(city, unit): 获取天气
- search_web(query): 搜索网络

开始！"""

def react_agent(question: str, max_steps: int = 5) -> str:
    """ReAct Agent 实现"""
    messages = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user", "content": question}
    ]
    
    for step in range(max_steps):
        response = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=messages,
            stop=["Observation:"],  # 遇到 Observation 停止生成（等待我们填入工具结果）
            temperature=0.0,
        )
        
        text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": text})
        
        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()
        
        # 解析 Action
        if "Action:" in text and "Action Input:" in text:
            action = text.split("Action:")[-1].split("\n")[0].strip()
            action_input_str = text.split("Action Input:")[-1].strip()
            
            try:
                action_input = json.loads(action_input_str.split("\n")[0])
                
                # 执行工具
                if action in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[action](**action_input)
                    observation = json.dumps(result, ensure_ascii=False)
                else:
                    observation = f"工具 {action} 不存在"
                
                # 将 Observation 加入对话
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
                print(f"  Step {step+1}: {action}({action_input}) → {observation[:80]}")
            
            except json.JSONDecodeError:
                messages.append({"role": "user", "content": "Observation: 解析工具参数失败，请检查格式"})
    
    return "达到最大步骤限制"

result = react_agent("北京和上海哪个城市今天更热？")
print(f"\n答案: {result}")
```

---

## 三、面试题精讲

**Q: Function Calling 是如何实现的？LLM 真的在"调用函数"吗？**

A: 不是。本质上 Function Calling 是：
1. 将工具的 JSON Schema 以特定格式注入 prompt（很多实现是加入 system prompt）
2. LLM 生成特定格式的 JSON 字符串（描述"应该调用什么函数、参数是什么"）
3. **应用层**解析这个 JSON，执行真实的 Python/API 调用
4. 将结果返回给 LLM 继续对话

LLM 只是生成了"调用说明"，实际执行是应用层完成的。

**Q: 如何处理工具调用失败的情况？**

A:
1. 捕获异常，将错误信息格式化后作为 tool result 返回给 LLM
2. LLM 通常能理解错误信息并决定下一步（重试、换工具、告知用户）
3. 设置最大重试次数和超时
4. 记录失败日志，异步通知运维

---

## 小结

```
Function Calling 核心流程：
  1. 定义工具（JSON Schema）
  2. LLM 决策（调用哪个工具，参数是什么）
  3. 应用层执行（Python 函数/API 调用）
  4. 结果返回 LLM
  5. LLM 生成最终答案

Agent 模式：
  Function Calling: 结构化，适合明确工具
  ReAct: 透明推理，适合复杂问题
  Plan-and-Execute: 先规划再执行，适合长任务
```
