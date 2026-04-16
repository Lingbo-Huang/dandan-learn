# D2 · ReAct 框架——Agent 的核心推理范式

> **Week 1 主题**：什么是 Agent——定义 / ReAct / 规划 / 记忆 / 工具调用  
> **本日主题**：ReAct：Reasoning + Acting

---

## 🎯 学习目标

1. 理解 ReAct 论文的核心思想及其对 Agent 领域的意义
2. 掌握 ReAct 的 Thought → Action → Observation 三元组
3. 能手写一个基于 ReAct 模式的简单 Agent
4. 理解 ReAct 的优势与局限性

---

## 📚 核心知识点

### 1. ReAct 是什么？

**ReAct** = **Re**asoning + **Act**ing，来自 2022 年 Google Research 论文：  
*"ReAct: Synergizing Reasoning and Acting in Language Models"*

**核心思想**：让 LLM 在行动时**显式地输出推理过程**，让推理过程指导行动，让行动结果更新推理。

```
传统方法：
  CoT（Chain-of-Thought）：只推理，不行动   → 幻觉多
  Action-only：只行动，不推理              → 不可控

ReAct：推理 + 行动 交织进行               → 更准确、更可控
```

### 2. ReAct 三元组

每个 ReAct 步骤包含三个元素：

```
Thought: [LLM 的推理过程，对当前状态的分析，下一步计划]
Action: [tool_name(parameters)]
Observation: [工具返回的结果]
```

**完整示例**（问题：2024年奥运会在哪里举办？游泳金牌是谁？）：

```
Thought: 需要搜索2024年奥运会的举办地点
Action: search("2024年夏季奥运会举办城市")
Observation: 2024年夏季奥运会在法国巴黎举办，时间为2024年7月26日-8月11日

Thought: 已知举办城市是巴黎，接下来需要查询游泳金牌得主
Action: search("2024巴黎奥运会游泳金牌")
Observation: 潘展乐获得100米自由泳金牌，打破世界纪录

Thought: 已获得所需信息，可以回答问题
Action: finish("2024年奥运会在法国巴黎举办，游泳100米自由泳金牌由潘展乐获得")
```

### 3. ReAct 的提示词结构

```python
REACT_SYSTEM_PROMPT = """你是一个 AI Agent，使用 ReAct 框架解决问题。

可用工具：
{tools}

输出格式（严格遵守）：
Thought: [你的推理过程]
Action: [工具名称(参数)]
Observation: [等待工具返回，不要自己填写]

重复以上步骤直到得出最终答案。
最终回答格式：
Thought: [总结推理]
Final Answer: [最终答案]
"""
```

### 4. ReAct 解析器（Parser）

LLM 输出的文本需要被解析成结构化的 Action：

```python
import re

def parse_react_output(text: str) -> dict:
    """解析 ReAct 格式输出"""
    thought_match = re.search(r'Thought: (.+?)(?=Action:|Final Answer:|$)', text, re.DOTALL)
    action_match = re.search(r'Action: (\w+)\((.+?)\)', text)
    final_match = re.search(r'Final Answer: (.+)', text, re.DOTALL)
    
    result = {}
    if thought_match:
        result['thought'] = thought_match.group(1).strip()
    if action_match:
        result['action'] = action_match.group(1)
        result['action_input'] = action_match.group(2).strip().strip('"\'')
    if final_match:
        result['final_answer'] = final_match.group(1).strip()
    
    return result
```

### 5. ReAct 的优势与局限

**优势**：
- 推理过程透明可解释
- 错误可追溯（知道在哪步出错）
- 工具调用有理由支撑
- 性能显著优于纯 CoT 或纯 Action

**局限**：
- 长任务上下文增长快
- 对 LLM 的指令遵循能力要求高
- 容易陷入无限循环
- 格式敏感，解析器维护成本高

---

## 💡 示例/推导

### 从零推导 ReAct 的必要性

**场景**：查询"比特币今天的价格比一周前涨了多少？"

```
# 方案一：纯 LLM 回答（CoT）
Thought: 比特币是加密货币，价格波动很大...
Answer: 我无法获取实时价格数据。（无用）

# 方案二：直接调用工具（无推理）
Action: get_price("BTC", "today")
→ 返回 $67,000
Action: get_price("BTC", "7days_ago")
→ 返回 $64,000
→ 但 LLM 不知道怎么用这两个数字...

# 方案三：ReAct（推理引导行动）
Thought: 需要获取今天和7天前的价格，然后计算涨幅
Action: get_crypto_price({"coin": "BTC", "date": "today"})
Observation: {"price": 67000, "currency": "USD"}

Thought: 今天价格是67000美元，现在需要7天前的价格
Action: get_crypto_price({"coin": "BTC", "date": "7_days_ago"})
Observation: {"price": 64000, "currency": "USD"}

Thought: 今天67000，7天前64000，涨幅 = (67000-64000)/64000 = 4.69%
Final Answer: 比特币今天价格约 $67,000，相比一周前 $64,000 涨了约 4.69%
```

ReAct 的 Thought 把"工具调用"和"结果计算"串联起来，是最完整的方案。

---

## 🔧 动手练习

### 练习 1：手动模拟 ReAct（必做）

在纸上完成以下任务的 ReAct 推理链（不用代码）：

**任务**："查询上海明天天气，推荐合适的穿衣方案"

可用工具：`get_weather(city, date)`, `get_outfit_suggestion(temperature, weather_type)`

写出至少 2 轮 Thought-Action-Observation。

### 练习 2：实现 ReAct 解析器（必做）

```python
# 创建文件: 02_react_parser.py
# 使用 uv run python 02_react_parser.py

import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class ReactStep:
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None

def parse_react_step(text: str) -> ReactStep:
    """
    解析单个 ReAct 步骤
    
    支持格式：
    Thought: ...
    Action: tool_name(input)
    
    或：
    Thought: ...
    Final Answer: ...
    """
    step = ReactStep(thought="")
    
    # 提取 Thought
    thought_match = re.search(
        r'Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)', 
        text, re.DOTALL
    )
    if thought_match:
        step.thought = thought_match.group(1).strip()
    
    # 提取 Action（格式：tool_name(input) 或 tool_name("input")）
    action_match = re.search(
        r'Action:\s*(\w+)\(([^)]*)\)', 
        text
    )
    if action_match:
        step.action = action_match.group(1)
        step.action_input = action_match.group(2).strip().strip('"\'')
    
    # 提取 Final Answer
    final_match = re.search(r'Final Answer:\s*(.+)', text, re.DOTALL)
    if final_match:
        step.final_answer = final_match.group(1).strip()
    
    return step

# 测试解析器
test_cases = [
    """
Thought: 我需要搜索上海的天气信息
Action: search("上海明天天气")
    """,
    """
Thought: 已经获得了足够的信息可以回答
Final Answer: 明天上海多云，气温18-24度，建议穿薄外套
    """,
]

print("=== ReAct 解析器测试 ===\n")
for i, case in enumerate(test_cases, 1):
    result = parse_react_step(case)
    print(f"测试用例 {i}:")
    print(f"  Thought: {result.thought}")
    if result.action:
        print(f"  Action: {result.action}({result.action_input})")
    if result.final_answer:
        print(f"  Final Answer: {result.final_answer}")
    print()
```

### 练习 3：构建完整 ReAct Agent（核心练习）

```python
# 创建文件: 03_react_agent.py
# uv add langchain langchain-openai
# uv run python 03_react_agent.py

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain import hub
import os
from dotenv import load_dotenv

load_dotenv()

# 定义工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入: 数学表达式字符串，如 '2 + 3 * 4'"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def word_counter(text: str) -> str:
    """统计文本中的字符数和单词数。输入: 需要统计的文本"""
    char_count = len(text)
    word_count = len(text.split())
    return f"字符数: {char_count}, 词数: {word_count}"

@tool
def reverse_string(text: str) -> str:
    """反转字符串。输入: 需要反转的字符串"""
    return f"反转结果: {text[::-1]}"

# 初始化模型和 Agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [calculator, word_counter, reverse_string]

# 使用 langchain hub 中的标准 ReAct prompt
prompt = hub.pull("hwchase17/react")

# 创建 ReAct Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,  # 显示 ReAct 过程
    max_iterations=5
)

# 运行测试
questions = [
    "计算 (123 + 456) * 789 的结果",
    "统计 'Hello World, I am learning ReAct Agent!' 的字符数，然后将这个数乘以2",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"问题: {q}")
    print('='*60)
    result = agent_executor.invoke({"input": q})
    print(f"\n最终答案: {result['output']}")
```

### 🦞 Claw 实战：分析一个 Claw Task 的 ReAct 模式

1. 打开你今天收到的本任务执行日志
2. 找出日志中哪些部分对应 **Thought**（AI 的分析/计划）
3. 找出哪些 **curl 调用**对应 **Action**
4. 找出哪些 **API 返回值**对应 **Observation**
5. 画出这个 Task 的 ReAct 链条图

---

## 📝 小结

| 要点 | 核心内容 |
|------|---------|
| **ReAct 定义** | Reasoning + Acting 交织的推理范式 |
| **三元组** | Thought → Action → Observation |
| **优势** | 透明、可解释、可追溯、性能优于纯 CoT |
| **局限** | 上下文增长、格式敏感、循环风险 |
| **实现** | 提示词工程 + 解析器 + 工具执行循环 |

**明天预告**：深入 Agent 规划能力——如何让 Agent 处理复杂长程任务？CoT、ToT、Plan-and-Execute 等规划策略。

---

> 💡 **今日思考题**：ReAct 中的 Thought 步骤有时候"看起来对，但实际上是错的"（LLM 幻觉）。你能想到什么方法来减少这种情况？
