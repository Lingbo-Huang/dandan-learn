# AutoGen：对话式多 Agent 协作

> **Day 4** · 预计学习时间：4-5 小时  
> **目标**：理解 AutoGen 的对话驱动模型，搭建多 Agent 协作场景

---

## 框架概念

### AutoGen 是什么？

AutoGen 是微软研究院开发的**多 Agent 对话框架**，核心思想是：

> 把复杂任务的解决过程，变成多个 Agent 之间的"对话协作"。

**核心创新：**
- **对话即协作**：Agent 之间通过发送消息协作，而不是函数调用
- **Human-in-the-loop 一等公民**：人类可以随时介入对话
- **自动代码执行**：Agent 可以生成代码并在沙盒中自动运行验证

### AutoGen vs LangChain Agent 的关键区别

| 维度 | LangChain Agent | AutoGen |
|------|----------------|---------|
| **协作模式** | 单 Agent 循环调用工具 | 多 Agent 相互对话 |
| **人类参与** | 较弱（需要自定义） | 内置 `HumanProxyAgent` |
| **代码执行** | 需要 Python REPL 工具 | 内置代码执行沙盒 |
| **任务分解** | 单 Agent 思考 | 多 Agent 分工讨论 |
| **适用场景** | 工具编排、信息检索 | 复杂推理、代码生成、迭代改进 |

### 核心组件

```
ConversableAgent（基类）
├── AssistantAgent    ← 由 LLM 驱动的 AI Agent
│   特点：擅长代码生成、推理、分析
│
├── UserProxyAgent    ← 代理用户的 Agent
│   特点：可执行代码、可转发到真实用户输入
│   human_input_mode: "ALWAYS" | "TERMINATE" | "NEVER"
│
└── GroupChat         ← 多 Agent 群聊
    └── GroupChatManager ← 管理发言顺序的协调者
```

---

## 核心代码示例（使用 uv）

### 环境准备

```bash
uv init autogen-demo
cd autogen-demo

# AutoGen 0.4+ 已重构为 autogen-agentchat
uv add pyautogen python-dotenv

# 如果需要代码执行沙盒
uv add docker  # 可选，用 Docker 隔离执行

echo 'OPENAI_API_KEY=your-key-here' > .env
```

### 示例 1：最简单的两人对话

```python
# two_agent_chat.py
import os
from dotenv import load_dotenv
import autogen

load_dotenv()

# LLM 配置
config_list = [{
    "model": "gpt-4o-mini",
    "api_key": os.environ["OPENAI_API_KEY"]
}]

llm_config = {"config_list": config_list, "temperature": 0}

# 创建 Assistant Agent（AI 角色）
assistant = autogen.AssistantAgent(
    name="助手",
    system_message="""你是一个专业的 Python 开发助手。
    当你给出代码时，请确保代码可以直接运行。
    当任务完成时，在回复中包含 TERMINATE。""",
    llm_config=llm_config,
)

# 创建 UserProxy Agent（执行代码的角色）
user_proxy = autogen.UserProxyAgent(
    name="用户",
    human_input_mode="NEVER",        # 不需要真实用户介入
    max_consecutive_auto_reply=10,    # 最多自动回复 10 次
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",         # 代码执行目录
        "use_docker": False,          # 不使用 Docker（生产建议用 Docker）
    },
)

# 开始对话
user_proxy.initiate_chat(
    assistant,
    message="写一个 Python 函数，计算斐波那契数列的第 n 项，并测试 n=10 的结果。"
)
```

### 示例 2：三方协作——研究员 + 写手 + 批评者

```python
# three_agent_crew.py
import os
from dotenv import load_dotenv
import autogen

load_dotenv()

config_list = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]
llm_config = {"config_list": config_list, "temperature": 0.3}

# 研究员：负责收集信息和分析
researcher = autogen.AssistantAgent(
    name="研究员",
    system_message="""你是一位严谨的研究员。
    职责：对给定主题进行深入分析，提供有数据支撑的观点。
    风格：客观、全面、引用具体例子。""",
    llm_config=llm_config,
)

# 写手：负责将分析转化为可读内容
writer = autogen.AssistantAgent(
    name="写手",
    system_message="""你是一位专业的技术写手。
    职责：将研究员的分析转化为清晰、易读的文章。
    风格：简洁、有逻辑、适合技术从业者阅读。""",
    llm_config=llm_config,
)

# 批评者：负责提出改进意见
critic = autogen.AssistantAgent(
    name="批评者",
    system_message="""你是一位严格的编辑。
    职责：指出文章的不足，提出具体改进建议。
    规则：每次只指出最重要的 2-3 个问题。
    完成标准：当内容质量足够高时，回复 "内容质量已达标，TERMINATE"。""",
    llm_config=llm_config,
)

# 用户代理（不执行代码，只协调）
user_proxy = autogen.UserProxyAgent(
    name="协调者",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=15,
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    code_execution_config=False,  # 不执行代码
)

# 创建群聊
groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, writer, critic],
    messages=[],
    max_round=12,
    speaker_selection_method="auto",  # 让 LLM 自动决定下一个发言者
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# 启动协作
user_proxy.initiate_chat(
    manager,
    message="""
    任务：写一篇关于"2024年 LLM Agent 框架的发展趋势"的技术文章。
    
    流程建议：
    1. 研究员先进行分析
    2. 写手将分析转化为文章
    3. 批评者提出改进
    4. 写手根据反馈修改
    5. 批评者确认质量达标
    
    最终产出：一篇 300-500 字的技术文章
    """
)
```

### 示例 3：Human-in-the-loop（人机协作）

```python
# human_in_loop.py
import os
from dotenv import load_dotenv
import autogen

load_dotenv()

config_list = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]

assistant = autogen.AssistantAgent(
    name="AI助手",
    system_message="""你是一个代码审查助手。
    当你不确定用户的意图时，明确提问。
    完成所有审查后说 TERMINATE。""",
    llm_config={"config_list": config_list},
)

# 真实人类介入模式
user_proxy = autogen.UserProxyAgent(
    name="开发者",
    human_input_mode="TERMINATE",  # 只在 AI 说结束时询问人类是否真的结束
    # human_input_mode="ALWAYS"    # 每次 AI 发言后都等待人类输入
    # human_input_mode="NEVER"     # 全自动，不询问人类
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda x: x.get("content", "").endswith("TERMINATE"),
    code_execution_config={"work_dir": "review", "use_docker": False}
)

# 开始对话
user_proxy.initiate_chat(
    assistant,
    message="""
请审查以下 Python 代码，指出潜在问题：

```python
def divide(a, b):
    return a / b

results = []
for i in range(10):
    results.append(divide(100, i))
print(results)
```
"""
)
```

### 示例 4：自定义 Agent 角色

```python
# custom_agents.py
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Union
import autogen

load_dotenv()

config_list = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]

class ProductManagerAgent(autogen.AssistantAgent):
    """产品经理 Agent：专注于需求分析和优先级排序"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="产品经理",
            system_message="""你是一位经验丰富的产品经理。
            职责：
            - 澄清用户需求，拆分为具体任务
            - 评估任务优先级（P0/P1/P2）
            - 确保产出物符合用户真实需要
            
            格式要求：用表格或列表呈现需求拆分结果。""",
            llm_config={"config_list": config_list, "temperature": 0.2},
            **kwargs
        )

class EngineerAgent(autogen.AssistantAgent):
    """工程师 Agent：负责技术实现"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="工程师",
            system_message="""你是一位全栈工程师。
            职责：
            - 根据产品需求给出技术方案
            - 编写可运行的代码
            - 预估工作量（小时）
            
            代码要求：加上必要注释，包含错误处理。""",
            llm_config={"config_list": config_list, "temperature": 0.1},
            **kwargs
        )

pm = ProductManagerAgent()
engineer = EngineerAgent()

user = autogen.UserProxyAgent(
    name="用户",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=8,
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    code_execution_config={"work_dir": "output", "use_docker": False}
)

groupchat = autogen.GroupChat(
    agents=[user, pm, engineer],
    messages=[],
    max_round=8
)
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

user.initiate_chat(
    manager,
    message="我想要一个命令行工具，可以批量重命名文件，支持正则表达式替换。"
)
```

---

## 与 Claw 的对比与联系

**AutoGen 的多 Agent 对话 vs Claw 的 Project/Task 体系：**

| 概念 | AutoGen | OpenClaw |
|------|---------|---------|
| **Agent 角色** | `AssistantAgent` / `UserProxyAgent` | Worker Agent（分配到 Task） |
| **任务分发** | 自然语言对话中动态分配 | 项目会话龙虾显式创建 Task |
| **人类介入** | `human_input_mode` 控制 | 用户在项目会话确认/否定 |
| **状态跟踪** | 对话历史 | Task status（running/pending_confirmation/completed） |
| **终止条件** | `is_termination_msg` 函数 | Task 状态变为 `pending_confirmation` → 用户确认 |

**关键洞察：**
- AutoGen 是**过程驱动**的（通过对话流程推进任务）
- Claw 是**状态驱动**的（通过任务状态机管理生命周期）
- AutoGen 的 `GroupChat` ≈ Claw 的一个 Project（多个 Agent 协作完成一个目标）
- AutoGen 的单轮对话 ≈ Claw 的一个 Task 执行周期

**实际整合方式：**
```
Claw Task → 启动 AutoGen GroupChat → AutoGen 内部多 Agent 讨论 → 
产出最终结果 → 写回 Claw Task summary
```

---

## 小结

- **AutoGen 最强的场景**：需要多角度讨论、迭代改进的复杂任务（代码审查、文章写作、方案设计）
- **Human-in-the-loop 设计**是 AutoGen 的一大亮点，`human_input_mode` 三种模式灵活适配不同场景
- **代码自动执行**使 AutoGen 特别擅长代码生成类任务（生成 → 执行 → 看到结果 → 改进）
- **GroupChat 的发言顺序**：`speaker_selection_method="auto"` 让 LLM 自动协调；也可以手动设置 round-robin 或自定义函数
- **陷阱**：无限循环——务必设置 `max_round` 和 `is_termination_msg`，否则 Agent 可能永远聊下去

**下一步** → Day 5：CrewAI——角色驱动的 Agent 任务编排
