# CrewAI：角色驱动的多 Agent 任务编排

> **Day 5** · 预计学习时间：4-5 小时  
> **目标**：理解 CrewAI 的三层结构，构建一个完整的角色协作 Crew

---

## 框架概念

### CrewAI 是什么？

CrewAI 是一个**角色驱动的多 Agent 编排框架**，创立于 2023 年末，核心哲学是：

> 把 AI Agent 比作一个团队的成员——每人有明确角色、职责和专长，共同完成任务。

**与 AutoGen 的本质区别：**

| 维度 | AutoGen | CrewAI |
|------|---------|--------|
| **隐喻** | 开放式对话（像会议） | 结构化团队（像项目组） |
| **任务流程** | 对话中动态涌现 | 预定义任务序列 |
| **角色感** | 较弱（靠 system_message 区分） | 强（Agent 有 role/goal/backstory） |
| **适合场景** | 复杂推理、迭代讨论 | 固定流程、内容生产、研究报告 |
| **上手难度** | 稍陡（需理解对话模型） | 较平缓（声明式配置） |

### 三层核心架构

```
Crew（团队）
├── Agent（成员）
│   ├── role        ← 角色名称（如"市场研究员"）
│   ├── goal        ← 个人目标（驱动 Agent 行为的动机）
│   ├── backstory   ← 背景故事（塑造 Agent 的风格和视角）
│   └── tools       ← 可用工具列表
│
├── Task（任务）
│   ├── description  ← 任务描述（具体要做什么）
│   ├── expected_output ← 期望输出格式/内容
│   ├── agent        ← 负责执行的 Agent
│   └── context      ← 依赖的前置 Task（自动传递结果）
│
└── Process（流程）
    ├── sequential   ← 顺序执行（默认，Task 按顺序一个个完成）
    └── hierarchical ← 层级执行（Manager 分配和监督其他 Agent）
```

### backstory 的魔力

CrewAI 最有特色的设计是 `backstory`——给 Agent 一段"人物小传"，而不只是指令：

```python
# 指令式（AutoGen 风格）
system_message = "你是一个数据分析师，专门分析用户行为数据"

# 背景故事式（CrewAI 风格）
backstory = """你在硅谷头部科技公司做了 8 年数据分析师，
曾主导多个用户增长项目。你相信数据要服务于业务决策，
不喜欢为数据而数据，总会问"这个分析能帮我们做什么决定？"
"""
```

实践中，`backstory` 确实能让 Agent 输出更有"个性"和一致性。

---

## 核心代码示例（使用 uv）

### 环境准备

```bash
uv init crewai-demo
cd crewai-demo

uv add crewai crewai-tools python-dotenv

echo 'OPENAI_API_KEY=your-key-here' > .env
```

### 示例 1：最简单的 Crew（研究 + 写作）

```python
# simple_crew.py
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool  # 搜索工具（需要 SERPER_API_KEY）

load_dotenv()

# 定义 Agent
researcher = Agent(
    role="技术研究员",
    goal="深入研究主题，提供准确、有价值的信息",
    backstory="""你是一位有 10 年经验的技术研究员，擅长快速理解新技术
    并从海量信息中提炼核心要点。你的研究总是有数据支撑，引用可靠来源。""",
    verbose=True,
    allow_delegation=False,  # 不允许把任务委托给其他 Agent
    llm="gpt-4o-mini",  # 使用更便宜的模型
)

writer = Agent(
    role="技术写手",
    goal="将技术研究转化为清晰易懂的文章",
    backstory="""你曾在多家科技媒体担任编辑，专注于让复杂的技术
    对普通读者变得易懂。你的文章逻辑清晰，例子生动，读者喜欢你的风格。""",
    verbose=True,
    allow_delegation=False,
    llm="gpt-4o-mini",
)

# 定义 Task
research_task = Task(
    description="""研究以下主题：{topic}
    
    请覆盖：
    1. 核心概念和定义
    2. 最新发展动态（2024年）
    3. 主要应用场景
    4. 存在的挑战或争议
    
    输出要点式笔记，约 400 字。""",
    expected_output="结构化的研究笔记，包含核心概念、最新动态、应用场景和挑战",
    agent=researcher,
)

write_task = Task(
    description="""基于研究笔记，撰写一篇面向技术从业者的文章。
    
    要求：
    - 标题吸引人
    - 结构清晰（引言 → 正文 → 结论）
    - 约 500 字
    - 适合微信公众号或技术博客发布""",
    expected_output="完整的技术文章，包含标题、正文和结论",
    agent=writer,
    context=[research_task],  # 依赖研究任务的输出
)

# 组建 Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,  # 顺序执行
    verbose=True,
)

# 运行
result = crew.kickoff(inputs={"topic": "AI Agent 框架的 2024 年发展趋势"})
print("\n" + "="*50)
print("最终文章：")
print(result)
```

### 示例 2：层级 Crew（Manager 模式）

```python
# hierarchical_crew.py
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

load_dotenv()

# Manager Agent（负责协调）
manager = Agent(
    role="项目经理",
    goal="协调团队完成高质量的产品分析报告",
    backstory="""你是一位资深项目经理，擅长将复杂项目拆分为可执行任务
    并分配给合适的团队成员。你注重质量，会在最终交付前做严格审核。""",
    llm="gpt-4o-mini",
    allow_delegation=True,  # Manager 可以委托任务
)

market_analyst = Agent(
    role="市场分析师",
    goal="分析市场规模、竞争格局和用户需求",
    backstory="""你专注于市场研究，曾服务于多家咨询公司，
    擅长从公开数据中发现市场机会和竞争威胁。""",
    llm="gpt-4o-mini",
    allow_delegation=False,
)

tech_analyst = Agent(
    role="技术分析师",
    goal="评估技术方案的可行性、架构合理性和实现难度",
    backstory="""你有 10 年软件架构经验，能快速识别技术方案的优劣，
    擅长用通俗语言解释复杂的技术决策。""",
    llm="gpt-4o-mini",
    allow_delegation=False,
)

# 综合分析任务（Manager 会把子任务委托给对应 Agent）
analysis_task = Task(
    description="""对产品 {product_name} 进行全面分析，包括：
    1. 市场机会分析（目标用户、市场规模、竞争对手）
    2. 技术可行性分析（核心技术、实现难点、预估工期）
    3. 综合建议（是否值得投入、关键风险、下一步行动）
    
    最终输出：一份完整的产品分析报告。""",
    expected_output="包含市场分析、技术分析和综合建议的结构化报告",
    agent=manager,
)

crew = Crew(
    agents=[manager, market_analyst, tech_analyst],
    tasks=[analysis_task],
    process=Process.hierarchical,  # Manager 主动分配子任务
    manager_llm="gpt-4o-mini",
    verbose=True,
)

result = crew.kickoff(inputs={"product_name": "AI 驱动的代码审查工具"})
print(result)
```

### 示例 3：带自定义工具的 Crew

```python
# crew_with_tools.py
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import json

load_dotenv()

# 自定义工具的输入 Schema
class StockSearchInput(BaseModel):
    company: str = Field(description="公司名称")

class StockSearchTool(BaseTool):
    name: str = "股票信息查询"
    description: str = "查询指定公司的最新股价和基本财务数据（模拟数据）"
    args_schema: Type[BaseModel] = StockSearchInput
    
    def _run(self, company: str) -> str:
        # 模拟数据
        mock_stocks = {
            "苹果": {"price": 189.5, "pe_ratio": 28.5, "market_cap": "2.9T", "trend": "上涨"},
            "谷歌": {"price": 141.2, "pe_ratio": 24.1, "market_cap": "1.8T", "trend": "横盘"},
            "微软": {"price": 378.8, "pe_ratio": 35.2, "market_cap": "2.8T", "trend": "上涨"},
        }
        data = mock_stocks.get(company, {"error": f"未找到 {company} 的数据"})
        return json.dumps(data, ensure_ascii=False)

class NewsSearchInput(BaseModel):
    keyword: str = Field(description="新闻关键词")
    
class NewsSearchTool(BaseTool):
    name: str = "新闻搜索"
    description: str = "搜索最新的财经新闻（模拟数据）"
    args_schema: Type[BaseModel] = NewsSearchInput
    
    def _run(self, keyword: str) -> str:
        return f"关于 {keyword} 的最新动态：市场普遍看好，机构投资者持续加仓，分析师上调目标价。"

# 分析师 Agent（带工具）
analyst = Agent(
    role="股票分析师",
    goal="基于数据给出客观的投资分析",
    backstory="""你是一位 CFA 持证分析师，习惯用数据说话，
    既关注基本面也关注市场情绪。你的报告总是有理有据，不夸大不缩小。""",
    tools=[StockSearchTool(), NewsSearchTool()],
    llm="gpt-4o-mini",
    verbose=True,
)

analysis_task = Task(
    description="分析 {companies} 这几家公司，给出投资建议和风险提示",
    expected_output="详细的对比分析报告，包含数据、趋势和投资建议",
    agent=analyst,
)

crew = Crew(
    agents=[analyst],
    tasks=[analysis_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff(inputs={"companies": "苹果、谷歌、微软"})
print(result)
```

---

## 与 Claw 的对比与联系

**CrewAI 三层结构 vs Claw 三层结构：**

| CrewAI | OpenClaw | 类比说明 |
|--------|---------|---------|
| `Crew` | `Project` | 顶层协作单元，有明确目标 |
| `Agent` | `Worker Agent` | 有角色、能力的执行者 |
| `Task` | `Task`（Claw） | 具体的执行单元 |
| `Process.sequential` | Task 依赖链 | 任务按顺序执行 |
| `Process.hierarchical` | 项目会话龙虾 | 有一个协调者分配任务 |
| `context=[task_a]` | `dependencies: [stepId]` | Task 之间的依赖关系 |

**CrewAI `backstory` vs Claw Agent `description`：**
- 两者都是在"塑造 Agent 的行为方式"
- CrewAI 的 backstory 更文学化，强调 Agent 的"性格"
- Claw 的 description 更功能化，强调 Agent 的"能力"

**实际应用：**
- 可以将整个 CrewAI Crew 封装为 Claw 的一个 Task Skill
- CrewAI 处理"怎么写"（角色协作、内容生产）
- Claw 处理"谁来发起、谁来确认、怎么交付"

---

## 小结

- **CrewAI 最擅长的场景**：内容生产（文章、报告、分析）、有固定流程的研究任务
- **`backstory` 是 CrewAI 的灵魂**——花时间写好背景故事，Agent 输出质量会显著提升
- **`context` 参数**是 CrewAI 的任务依赖机制，自动把上游 Task 的输出传给下游
- **Process 选择**：80% 的场景用 sequential；有复杂子任务分配需求才用 hierarchical
- **与 AutoGen 的选择原则**：流程固定、角色明确 → CrewAI；需要开放式讨论和迭代 → AutoGen

**下一步** → Day 6：四大框架横向对比与选型指南
