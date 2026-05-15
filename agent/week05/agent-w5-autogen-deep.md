---
layout: default
title: "W5D4 · AutoGen 深入实战"
---

# AutoGen 深入实战：微软的多 Agent 框架

> **Week 5 · Day 4** | 难度：⭐⭐⭐⭐

---

## AutoGen 核心概念

AutoGen 是微软开发的多 Agent 对话框架，核心思想是：**Agent 通过对话协作完成任务**。

```
┌─────────────────────────────────────────┐
│             AutoGen 架构                 │
│                                          │
│  UserProxyAgent ←──→ AssistantAgent     │
│  (代理用户)          (AI执行者)          │
│       │                    │            │
│  执行代码           生成代码/分析        │
│  提供反馈           规划下一步           │
│                                          │
│  GroupChat ── 多个 Agent 共同对话        │
└─────────────────────────────────────────┘
```

## 安装与基础配置

```bash
pip install pyautogen
```

```python
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# 配置 LLM
config_list = [
    {
        "model": "gpt-4o",
        "api_key": "your-api-key",
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0,
    "timeout": 120,
}
```

## 基础：Assistant + UserProxy 对话

```python
# 创建 AssistantAgent（AI 执行者）
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="""你是一个 Python 专家。
当需要编写代码时，将代码放在 ```python 代码块中。
代码必须是完整可运行的。
分析结果时要清晰简洁。"""
)

# 创建 UserProxyAgent（代理用户，可执行代码）
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",    # 不需要人工确认
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "/tmp/autogen_workspace",
        "use_docker": False,    # 不用 Docker（生产中建议用）
    }
)

# 启动对话
user_proxy.initiate_chat(
    assistant,
    message="分析下面的数据并生成可视化图表：销售数据 [100, 150, 120, 200, 180, 220, 250]"
)
```

## 进阶：GroupChat 多 Agent 协作

```python
# 创建多个专业 Agent
planner = AssistantAgent(
    name="Planner",
    llm_config=llm_config,
    system_message="""你是规划专家。
职责：分析任务，制定详细执行计划。
完成规划后，说明下一步应该由谁执行什么。
不要写代码，只做规划。"""
)

engineer = AssistantAgent(
    name="Engineer",
    llm_config=llm_config,
    system_message="""你是 Python 工程师。
职责：根据规划编写高质量 Python 代码。
要求：
- 代码必须放在 ```python 代码块中
- 包含错误处理
- 添加注释说明
- 代码完整可运行"""
)

reviewer = AssistantAgent(
    name="Reviewer",
    llm_config=llm_config,
    system_message="""你是代码审查专家。
职责：审查代码质量、安全性和最佳实践。
评估维度：
- 逻辑正确性
- 代码风格
- 潜在 bug
- 性能问题
如果代码有问题，明确说明需要修改的地方。
如果代码通过审查，说 "代码审查通过" 并结束。"""
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=15,
    code_execution_config={
        "work_dir": "/tmp/autogen_group",
        "use_docker": False,
    },
    is_termination_msg=lambda x: "代码审查通过" in x.get("content", ""),
)

# 创建 GroupChat
groupchat = GroupChat(
    agents=[user_proxy, planner, engineer, reviewer],
    messages=[],
    max_round=20,
    speaker_selection_method="auto",  # 让 LLM 决定下一个发言者
)

# GroupChatManager 管理对话流程
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# 启动多 Agent 协作
user_proxy.initiate_chat(
    manager,
    message="""
    任务：构建一个数据处理管道，完成以下功能：
    1. 从 CSV 文件读取销售数据
    2. 清理空值和异常值
    3. 计算月度统计（总销售额、平均值、最大值）
    4. 生成 JSON 格式的报告
    5. 将结果保存到文件
    
    请先规划，然后实现，最后审查代码。
    """
)
```

## 自定义发言顺序

```python
def custom_speaker_selection(last_speaker, groupchat):
    """自定义下一个发言者的选择逻辑"""
    messages = groupchat.messages
    
    if len(messages) <= 1:
        return planner  # 第一步永远是规划
    
    last_message = messages[-1]["content"].lower()
    
    # 根据上一条消息内容决定下一个发言者
    if "计划如下" in last_message or "规划完成" in last_message:
        return engineer   # 规划后交给工程师
    elif "```python" in last_message:
        return reviewer   # 有代码就交给审查者
    elif "需要修改" in last_message or "存在问题" in last_message:
        return engineer   # 有问题交回工程师修改
    elif "代码审查通过" in last_message:
        return user_proxy  # 通过则执行
    else:
        return planner    # 默认回到规划者

groupchat_custom = GroupChat(
    agents=[user_proxy, planner, engineer, reviewer],
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection,  # 自定义选择
)
```

## 工具调用集成

```python
from autogen import register_function
from typing import Annotated

# 定义工具
def search_database(
    query: Annotated[str, "数据库查询语句"],
    table: Annotated[str, "要查询的表名"]
) -> str:
    """查询数据库"""
    # 模拟数据库查询
    mock_data = {
        "sales": [
            {"month": "2024-01", "revenue": 150000},
            {"month": "2024-02", "revenue": 180000},
        ]
    }
    if table in mock_data:
        return str(mock_data[table])
    return f"表 {table} 不存在"

def write_report(
    title: Annotated[str, "报告标题"],
    content: Annotated[str, "报告内容"],
    filename: Annotated[str, "保存的文件名"]
) -> str:
    """生成并保存报告"""
    with open(f"/tmp/{filename}", "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{content}")
    return f"报告已保存到 /tmp/{filename}"

# 创建带工具的 Agent
tool_assistant = AssistantAgent(
    name="tool_assistant",
    llm_config=llm_config,
    system_message="你是数据分析助手，使用工具查询数据并生成报告。"
)

tool_proxy = UserProxyAgent(
    name="tool_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,  # 不执行代码，只用工具
)

# 注册工具到 Agent
register_function(
    search_database,
    caller=tool_assistant,
    executor=tool_proxy,
    description="查询销售数据库"
)

register_function(
    write_report,
    caller=tool_assistant,
    executor=tool_proxy,
    description="生成并保存分析报告"
)

# 执行
tool_proxy.initiate_chat(
    tool_assistant,
    message="查询2024年1-2月的销售数据，生成分析报告保存为 sales_report.md"
)
```

## 踩坑经验

### 坑1：GroupChat 陷入无限循环

**问题**：Agent A 说完，Agent B 说完，Agent A 又说，循环不停。  
**解法**：
1. 设置明确的终止条件 `is_termination_msg`
2. 限制 `max_round`（通常 15-25 轮足够）
3. 用自定义 `speaker_selection_method` 控制发言顺序

### 坑2：代码执行结果不传递给下一个 Agent

**问题**：UserProxy 执行了代码，但 Engineer 没看到执行结果。  
**解法**：AutoGen 自动将执行结果加入 groupchat 消息，确保 `code_execution_config` 正确配置，且 `UserProxy` 在 groupchat 中。

### 坑3：温度设置影响多 Agent 协作稳定性

**问题**：temperature=0.7 导致 Agent 有时拒绝按计划执行，"创意"发挥偏离任务。  
**解法**：多 Agent 协作场景建议 temperature=0，需要创意的 Agent 单独设置较高温度。

### 坑4：工具调用权限混乱

**问题**：给了 AssistantAgent 工具，但它无法执行（只能调用 UserProxy 执行）。  
**解法**：AutoGen 的工具调用模式：AssistantAgent 是 `caller`（决定调用什么），UserProxy 是 `executor`（实际执行），两个角色分开。

---

*W5D4 · AutoGen 深入实战 | Agent + Claw 系列*
