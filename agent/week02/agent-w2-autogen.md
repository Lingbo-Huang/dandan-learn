# D4 AutoGen 多 Agent 对话框架

> **学习目标**：理解 AutoGen 的多 Agent 协作设计理念，掌握各类 Agent 的使用方式，能够构建多 Agent 协同完成复杂任务的系统。

---

## 一、AutoGen 是什么？

AutoGen 是微软研究院于 2023 年开源的多 Agent 框架。它的核心创新在于：**让多个 AI Agent 通过对话协作解决复杂问题**。

### 为什么需要多 Agent？

单个 LLM 的局限：
- **上下文窗口有限**：长任务容易丢失信息
- **单点能力瓶颈**：一个模型难以同时擅长推理、编码、检索
- **缺乏自我校正**：没有"第二双眼睛"检查错误
- **难以并行**：复杂任务中各子任务本可并行

多 Agent 协作的优势：
- **角色专业化**：每个 Agent 专注于自己擅长的部分
- **相互校验**：Agent 之间可以互相 review 和纠错
- **并行执行**：多 Agent 可以并行处理子任务
- **灵活组合**：按需组合不同能力的 Agent

---

## 二、核心架构概览

```
┌─────────────────────────────────────────────────────┐
│                    AutoGen 架构                       │
├──────────────────────────┬──────────────────────────┤
│     ConversableAgent      │    GroupChatManager      │
│  ┌────────────────────┐  │  ┌──────────────────┐    │
│  │ AssistantAgent     │  │  │ 轮询（Round Robin）│    │
│  │ UserProxyAgent     │  │  │ 随机选择           │    │
│  │ GroupChatAgent     │  │  │ LLM 动态选择       │    │
│  └────────────────────┘  │  └──────────────────┘    │
├──────────────────────────┴──────────────────────────┤
│               代码执行 / 工具调用 / 人工介入           │
└─────────────────────────────────────────────────────┘
```

### 安装

```bash
pip install pyautogen
# 或（新版）
pip install autogen-agentchat autogen-ext[openai]
```

---

## 三、核心组件：ConversableAgent

`ConversableAgent` 是 AutoGen 中最基础的 Agent 类，所有具体 Agent 都继承自它：

```python
from autogen import ConversableAgent, config_list_from_json

# 配置 LLM
llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": "your-api-key",
        }
    ],
    "temperature": 0,
    "cache_seed": 42,  # 可重复性
}

# 创建基础 Agent
agent = ConversableAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="你是一个专业的 Python 工程师，擅长代码审查和优化",
    human_input_mode="NEVER",   # NEVER / TERMINATE / ALWAYS
    max_consecutive_auto_reply=5,
    code_execution_config=False,
)
```

### human_input_mode 说明

| 模式 | 说明 |
|------|------|
| `NEVER` | 从不请求人工输入，全自动运行 |
| `TERMINATE` | 仅在满足终止条件时请求输入 |
| `ALWAYS` | 每次回复后都请求人工确认 |

---

## 四、最常用的 Agent 组合：Assistant + UserProxy

### 4.1 AssistantAgent — 思考者

```python
from autogen import AssistantAgent

assistant = AssistantAgent(
    name="AI助手",
    llm_config=llm_config,
    system_message="""你是一个 Python 专家。
    - 用清晰的步骤解释思路
    - 提供可运行的代码
    - 代码中包含注释
    - 如果任务完成，说 TERMINATE""",
)
```

### 4.2 UserProxyAgent — 执行者

UserProxyAgent 可以：
1. 代表人类用户参与对话
2. 自动执行代码块（在安全沙箱中）
3. 请求人工介入

```python
from autogen import UserProxyAgent

user_proxy = UserProxyAgent(
    name="用户代理",
    human_input_mode="NEVER",   # 全自动（无需人工干预）
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "executor": "local",   # 本地执行代码（或使用 Docker）
        "work_dir": "./code_output",
        "use_docker": False,   # 生产环境建议 True
    },
    system_message="你是用户代表，负责执行代码并报告结果",
)

# 双 Agent 对话
result = user_proxy.initiate_chat(
    recipient=assistant,
    message="请写一个 Python 函数，计算斐波那契数列前 N 项，并测试它",
    max_turns=8,
)

# 获取对话历史
print(f"对话轮数：{len(result.chat_history)}")
for msg in result.chat_history:
    print(f"\n[{msg['role']}] {msg['name']}:")
    print(msg['content'][:200])
```

### 4.3 代码自动执行示例

```
[用户代理] → "写一个计算斐波那契的函数并测试"
[AI助手]  → "好的，我来写：
             ```python
             def fibonacci(n):
                 if n <= 0: return []
                 elif n == 1: return [0]
                 fib = [0, 1]
                 for i in range(2, n):
                     fib.append(fib[-1] + fib[-2])
                 return fib
             
             print(fibonacci(10))
             ```"
[用户代理] → (自动执行代码) 
             "执行结果：[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"
[AI助手]  → "代码运行成功！... TERMINATE"
```

---

## 五、GroupChat — 多 Agent 群聊

### 5.1 基础群聊

```python
from autogen import GroupChat, GroupChatManager

# 定义多个专业 Agent
product_manager = AssistantAgent(
    name="产品经理",
    llm_config=llm_config,
    system_message="""你是一个经验丰富的产品经理。
    - 从用户需求角度思考
    - 关注产品功能和用户体验
    - 提供清晰的需求文档""",
)

developer = AssistantAgent(
    name="开发工程师",
    llm_config=llm_config,
    system_message="""你是一个全栈开发工程师。
    - 专注技术实现方案
    - 评估技术可行性和工作量
    - 提供具体的技术方案和代码""",
)

tester = AssistantAgent(
    name="测试工程师",
    llm_config=llm_config,
    system_message="""你是一个资深测试工程师。
    - 从测试角度发现问题
    - 设计测试用例
    - 评估潜在风险和边界条件""",
)

executor = UserProxyAgent(
    name="执行者",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "./output", "use_docker": False},
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

# 创建群聊
group_chat = GroupChat(
    agents=[product_manager, developer, tester, executor],
    messages=[],
    max_round=20,
    speaker_selection_method="auto",  # LLM 自动决定下一个发言者
    # 也可以用 "round_robin"（轮询）或 "random"
    allow_repeat_speaker=False,  # 不允许同一 Agent 连续发言
)

# 群聊管理器（协调对话流程）
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

# 启动群聊
result = executor.initiate_chat(
    manager,
    message="""请大家协作设计一个用户登录功能：
    1. 产品经理提供需求
    2. 开发工程师给出技术方案
    3. 测试工程师设计测试用例
    4. 执行者实现并测试代码""",
)
```

### 5.2 自定义发言顺序

```python
def custom_speaker_selection(last_speaker, groupchat):
    """自定义发言顺序逻辑"""
    agents = groupchat.agents
    last_name = last_speaker.name
    
    # 定义固定的轮转顺序
    order = ["产品经理", "开发工程师", "测试工程师", "执行者"]
    
    if last_name in order:
        next_idx = (order.index(last_name) + 1) % len(order)
        next_name = order[next_idx]
        return next(a for a in agents if a.name == next_name)
    
    return agents[0]  # 默认返回第一个

group_chat_ordered = GroupChat(
    agents=[product_manager, developer, tester, executor],
    messages=[],
    max_round=16,
    speaker_selection_method=custom_speaker_selection,
)
```

---

## 六、嵌套 Agent（Nested Chat）

AutoGen 支持 Agent 之间的嵌套对话，即一个 Agent 内部可以触发另一组 Agent 的协作：

```python
from autogen import initiate_chats

# 定义多个子任务
def create_research_pipeline():
    # 研究员
    researcher = AssistantAgent(
        name="researcher",
        llm_config=llm_config,
        system_message="你是一个专业研究员，擅长收集和整理信息",
    )
    
    # 写作者
    writer = AssistantAgent(
        name="writer",
        llm_config=llm_config,
        system_message="你是一个专业技术写作者，能将复杂信息转化为清晰文章",
    )
    
    # 评审者
    reviewer = AssistantAgent(
        name="reviewer",
        llm_config=llm_config,
        system_message="你是一个严格的内容评审者，负责检查准确性和质量",
    )
    
    user_proxy = UserProxyAgent(
        name="coordinator",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: x.get("content", "").endswith("TERMINATE"),
        code_execution_config=False,
    )
    
    # 顺序执行多组对话
    chat_results = initiate_chats([
        {
            "sender": user_proxy,
            "recipient": researcher,
            "message": "请研究 AutoGen 框架的最新特性",
            "max_turns": 3,
            "summary_method": "reflection_with_llm",  # 用 LLM 生成摘要
            "summary_args": {"summary_prompt": "请用3句话总结研究发现"},
        },
        {
            "sender": user_proxy,
            "recipient": writer,
            "message": "根据上述研究，撰写一篇技术博客文章",
            "max_turns": 3,
            "carryover": "自动携带上一个对话的摘要",
        },
        {
            "sender": user_proxy,
            "recipient": reviewer,
            "message": "请评审这篇技术文章的质量和准确性",
            "max_turns": 2,
        },
    ])
    
    return chat_results

results = create_research_pipeline()
```

---

## 七、工具调用（Function Calling）

```python
from autogen import AssistantAgent, UserProxyAgent
from typing import Annotated

# 方式一：使用装饰器注册工具
assistant_with_tools = AssistantAgent(
    name="data_analyst",
    llm_config=llm_config,
    system_message="你是一个数据分析师，可以使用工具获取和分析数据",
)

executor_agent = UserProxyAgent(
    name="tool_executor",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").endswith("TERMINATE"),
)

# 注册工具（同时在 caller 和 executor 上）
@executor_agent.register_for_execution()
@assistant_with_tools.register_for_llm(description="获取股票价格")
def get_stock_price(
    ticker: Annotated[str, "股票代码，如 AAPL"],
    period: Annotated[str, "时间范围，如 1d/1w/1mo"] = "1d"
) -> str:
    # 实际应用中接入真实 API
    prices = {"AAPL": 189.5, "GOOGL": 140.2, "MSFT": 415.8}
    price = prices.get(ticker, 0)
    return f"{ticker} ({period}): ${price:.2f}"

@executor_agent.register_for_execution()
@assistant_with_tools.register_for_llm(description="计算股票收益率")
def calc_return(
    buy_price: Annotated[float, "买入价格"],
    sell_price: Annotated[float, "卖出价格"]
) -> str:
    ret = (sell_price - buy_price) / buy_price * 100
    return f"收益率：{ret:.2f}%"

# 启动对话
executor_agent.initiate_chat(
    assistant_with_tools,
    message="帮我查一下苹果公司的股价，并假设我以150美元买入，计算当前收益率",
)
```

---

## 八、实战案例：代码评审系统

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

code_to_review = """
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
    return result
"""

# 代码作者
author = UserProxyAgent(
    name="代码作者",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "./review", "use_docker": False},
    is_termination_msg=lambda x: "LGTM" in x.get("content", ""),
)

# 代码评审员1：性能专家
perf_reviewer = AssistantAgent(
    name="性能专家",
    llm_config=llm_config,
    system_message="""你是一个 Python 性能优化专家。
    专注于：时间复杂度、空间复杂度、Pythonic 写法
    评审后给出具体的优化建议和改进代码""",
)

# 代码评审员2：安全专家
security_reviewer = AssistantAgent(
    name="安全专家",
    llm_config=llm_config,
    system_message="""你是一个代码安全专家。
    专注于：输入验证、边界条件、潜在漏洞
    评审后指出安全隐患并给出修复建议""",
)

# 评审总结者
summarizer = AssistantAgent(
    name="评审总结",
    llm_config=llm_config,
    system_message="""你负责汇总所有评审意见。
    综合性能和安全两位专家的意见，给出最终评审报告。
    如果代码质量满足要求，在消息末尾加上 LGTM""",
)

group_chat = GroupChat(
    agents=[author, perf_reviewer, security_reviewer, summarizer],
    messages=[],
    max_round=12,
    speaker_selection_method="round_robin",
)

manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

author.initiate_chat(
    manager,
    message=f"请大家帮我评审以下 Python 代码：\n```python\n{code_to_review}\n```",
)
```

---

## 九、AutoGen 2.0 新特性（agentchat）

AutoGen 2.0 引入了全新的架构：

```python
# AutoGen 2.0 风格
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 创建模型客户端
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key="your-api-key",
)

# 创建 Agent（2.0 风格）
agent1 = AssistantAgent(
    name="researcher",
    model_client=model_client,
    system_message="你是一个研究专家",
)

agent2 = AssistantAgent(
    name="writer",
    model_client=model_client,
    system_message="你是一个技术写作专家",
)

# 终止条件
termination = MaxMessageTermination(10) | TextMentionTermination("DONE")

# 创建团队
team = RoundRobinGroupChat(
    participants=[agent1, agent2],
    termination_condition=termination,
)

# 运行（异步）
import asyncio

async def run():
    result = await team.run(task="分析 AutoGen 框架的主要特性")
    return result

asyncio.run(run())
```

---

## 十、与其他框架的对比

| 维度 | AutoGen | LangChain Agent | CrewAI |
|------|---------|----------------|--------|
| 核心模式 | 对话驱动 | 工具调用循环 | 角色扮演任务流 |
| 代码执行 | 原生内置 | 需要配置 | 较弱 |
| 多 Agent | 核心特性 | 需要 LangGraph | 核心特性 |
| 人机协作 | 灵活支持 | 有限支持 | 较弱 |
| 适用场景 | 代码/研究/协作 | 工具密集型任务 | 角色分工任务 |

---

## 小结

AutoGen 的三大特点：
1. **对话即协作**：Agent 通过自然语言对话协调，无需显式编排
2. **代码自动执行**：内置安全的代码执行环境，实现真正的"思考-执行"循环
3. **灵活的 Agent 组合**：双人对话、多人群聊、嵌套对话，满足各种协作场景

下一篇（D5）将介绍 CrewAI —— 一个以"角色分工"为核心的多 Agent 框架。
