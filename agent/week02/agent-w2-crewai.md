# D5 CrewAI 角色分工与任务流

> **学习目标**：理解 CrewAI 的角色导向设计哲学，掌握 Agent、Task、Crew 的定义方式，能够构建角色清晰、任务有序的多 Agent 工作流。

---

## 一、CrewAI 是什么？

CrewAI（2024 年发布）的设计灵感来自人类团队协作："给每个成员一个清晰的角色，分配具体的任务，让他们协作完成大目标"。

核心思想：**像管理一个专业团队一样管理 AI Agent**。

### 为什么选择 CrewAI？

| 场景 | 适合程度 |
|------|---------|
| 市场调研报告生成 | ⭐⭐⭐⭐⭐ |
| 内容创作流水线 | ⭐⭐⭐⭐⭐ |
| 数据分析与报告 | ⭐⭐⭐⭐ |
| 软件开发流程模拟 | ⭐⭐⭐⭐ |
| 需要动态决策的复杂任务 | ⭐⭐ |

### 安装

```bash
pip install crewai crewai-tools
```

---

## 二、核心三角：Agent、Task、Crew

CrewAI 的整个框架建立在三个概念之上：

```
┌─────────────────────────────────────────────────┐
│                      Crew                        │
│  ┌──────────┐     ┌──────────┐  ┌──────────┐   │
│  │  Agent   │     │  Agent   │  │  Agent   │   │
│  │ (角色)   │     │ (角色)   │  │ (角色)   │   │
│  └─────┬────┘     └────┬─────┘  └────┬─────┘   │
│        │               │             │           │
│  ┌─────▼────┐     ┌────▼─────┐  ┌───▼──────┐   │
│  │  Task 1  │────▶│  Task 2  │─▶│  Task 3  │   │
│  │ (任务)   │     │ (任务)   │  │ (任务)   │   │
│  └──────────┘     └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────┘
```

---

## 三、Agent —— 角色定义

Agent 是 CrewAI 的核心，每个 Agent 代表一个有特定专长的"专业人员"：

```python
from crewai import Agent
from langchain_openai import ChatOpenAI

# 配置 LLM（CrewAI 基于 LangChain）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 研究员 Agent
researcher = Agent(
    role="资深市场研究员",           # 角色名称（关键！会影响行为）
    goal="收集准确、全面的市场数据，识别行业趋势和竞争格局",  # 目标
    backstory="""你有10年的市场研究经验，曾在多家顶级咨询公司工作。
    你擅长从海量信息中提炼关键洞察，并能用数据支撑你的观点。
    你的分析总是客观、准确、有深度。""",  # 背景故事（塑造行为模式）
    llm=llm,
    verbose=True,                    # 打印执行过程
    allow_delegation=True,           # 允许将子任务委托给其他 Agent
    max_iter=10,                     # 最大迭代次数
    memory=True,                     # 开启记忆
    tools=[],                        # 可用工具列表（后面详述）
)

# 撰稿人 Agent
writer = Agent(
    role="资深内容策略师",
    goal="将研究发现转化为引人入胜的专业报告，确保内容清晰、有说服力",
    backstory="""你是一位获奖的技术作家，专注于将复杂的市场数据
    转化为易于理解的商业洞察。你的文章以逻辑严谨、表达清晰著称。""",
    llm=llm,
    verbose=True,
    allow_delegation=False,          # 写作任务不委托他人
)

# 编辑 Agent
editor = Agent(
    role="首席内容编辑",
    goal="确保最终内容的质量、准确性和专业性，消除任何错误或不一致",
    backstory="""你是一位有20年经验的资深编辑，对内容质量有极高要求。
    你能发现隐藏的逻辑漏洞、事实错误和表达不当之处。""",
    llm=llm,
    verbose=True,
    allow_delegation=True,
)
```

### Agent 参数详解

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `role` | 角色名称（最重要，影响LLM行为） | 必填 |
| `goal` | 角色的核心目标 | 必填 |
| `backstory` | 背景故事（塑造专业性和行为风格） | 必填 |
| `llm` | 使用的语言模型 | OpenAI GPT-4 |
| `tools` | 可调用的工具列表 | `[]` |
| `allow_delegation` | 是否可将子任务委托给其他 Agent | `True` |
| `verbose` | 是否打印执行日志 | `False` |
| `memory` | 是否启用记忆 | `False` |
| `max_iter` | 最大推理迭代次数 | `15` |
| `max_rpm` | 每分钟最大API请求数 | `None` |

---

## 四、Task —— 任务定义

Task 是具体要完成的工作单元：

```python
from crewai import Task

# 研究任务
research_task = Task(
    description="""对{topic}进行全面的市场研究。
    
    具体要求：
    1. 收集该领域的市场规模和增长数据
    2. 识别主要竞争者及其市场份额
    3. 分析当前趋势和未来发展方向
    4. 找出主要挑战和机遇
    5. 数据来源要可靠，并注明出处""",
    
    expected_output="""一份结构化的研究报告，包含：
    - 市场概览（规模、增长率）
    - 主要玩家分析（前5名，含市场份额）
    - 3-5个关键趋势
    - SWOT 分析
    - 数据来源列表""",
    
    agent=researcher,    # 执行此任务的 Agent
    async_execution=False,  # 是否异步执行
    output_file="research_output.md",  # 可选：保存输出到文件
)

# 写作任务（依赖研究任务）
writing_task = Task(
    description="""基于研究团队提供的数据，撰写一篇关于{topic}的专业市场分析文章。
    
    文章结构：
    1. 执行摘要（200字）
    2. 市场现状分析（300字）
    3. 竞争格局（300字）
    4. 趋势与机遇（300字）
    5. 风险与挑战（200字）
    6. 结论与建议（200字）
    
    风格：专业、客观，数据支撑，可直接用于商业报告""",
    
    expected_output="一篇1500字以上的完整市场分析文章，Markdown 格式",
    
    agent=writer,
    context=[research_task],  # 依赖研究任务的输出作为上下文
    output_file="market_analysis.md",
)

# 编辑任务
editing_task = Task(
    description="""对撰稿人提交的市场分析文章进行专业编辑。
    
    检查要点：
    1. 事实准确性（与研究报告数据是否一致）
    2. 逻辑连贯性（段落之间是否流畅衔接）
    3. 表达专业性（用词是否恰当）
    4. 格式规范性（标题层级、数据引用格式）
    5. 整体可读性""",
    
    expected_output="最终的、经过专业校对的市场分析文章，附上修改说明",
    
    agent=editor,
    context=[research_task, writing_task],
    output_file="final_report.md",
)
```

---

## 五、Crew —— 团队与执行

```python
from crewai import Crew, Process

# 创建团队
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    
    process=Process.sequential,  # 顺序执行（也可 Process.hierarchical）
    verbose=True,
    memory=True,          # 团队级记忆
    embedder={            # 用于记忆的嵌入模型
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    },
    max_rpm=10,           # 全局 API 速率限制
    share_crew=False,     # 不分享到 CrewAI 平台
)

# 执行任务
result = crew.kickoff(inputs={"topic": "中国生成式AI市场"})

print("=" * 50)
print("最终输出：")
print(result.raw)
print(f"\nToken 使用情况：{result.token_usage}")
```

### Process 类型对比

| Process | 说明 | 适用场景 |
|---------|------|---------|
| `sequential` | 按顺序执行任务 | 有明确依赖链的任务 |
| `hierarchical` | 有管理员Agent协调 | 需要动态分配任务 |

---

## 六、工具集成

CrewAI 提供了丰富的内置工具，并与 LangChain 工具完全兼容：

```python
from crewai_tools import (
    SerperDevTool,          # Google 搜索（需 SERPER_API_KEY）
    WebsiteSearchTool,      # 网站内容搜索
    FileReadTool,           # 读取文件
    FileWriterTool,         # 写入文件
    PDFSearchTool,          # PDF 内容搜索
    CSVSearchTool,          # CSV 数据分析
    CodeInterpreterTool,    # 代码执行
    YoutubeVideoSearchTool, # YouTube 视频搜索
)
from crewai.tools import BaseTool

# 使用内置搜索工具
search_tool = SerperDevTool()
web_tool = WebsiteSearchTool()

researcher_with_tools = Agent(
    role="资深研究员",
    goal="通过搜索和分析网络信息获取最新数据",
    backstory="你擅长使用各种网络工具收集第一手信息",
    tools=[search_tool, web_tool],
    llm=llm,
    verbose=True,
)

# 自定义工具
class StockPriceTool(BaseTool):
    name: str = "stock_price_tool"
    description: str = "获取指定股票的当前价格和涨跌幅"
    
    def _run(self, ticker: str) -> str:
        """执行工具逻辑"""
        # 实际应用中接入真实数据源
        data = {
            "AAPL": {"price": 189.5, "change": "+1.2%"},
            "GOOGL": {"price": 140.2, "change": "-0.5%"},
        }
        info = data.get(ticker, {"price": 0, "change": "N/A"})
        return f"{ticker}: ${info['price']} ({info['change']})"

stock_tool = StockPriceTool()

# 与 LangChain 工具集成
from langchain_community.tools import DuckDuckGoSearchRun

ddg_tool = DuckDuckGoSearchRun()
# CrewAI 可以直接使用 LangChain 工具，通过包装

from crewai.tools import tool as crewai_tool

@crewai_tool
def get_news(topic: str) -> str:
    """搜索指定主题的最新新闻"""
    # 使用 DuckDuckGo 搜索
    results = ddg_tool.run(f"{topic} latest news 2024")
    return results[:1000]  # 限制长度
```

---

## 七、层级流程（Hierarchical Process）

在层级流程中，一个"管理者 Agent"负责协调其他 Agent：

```python
from crewai import Agent, Task, Crew, Process

# 创建管理者 Agent
manager = Agent(
    role="项目总监",
    goal="协调团队成员，确保项目按时、高质量完成",
    backstory="""你是一个有丰富经验的项目管理专家，
    善于分配任务、协调资源、跟踪进度并解决冲突。""",
    llm=llm,
    allow_delegation=True,    # 管理者必须允许委托！
    verbose=True,
)

# 专业 Agent（执行者）
data_analyst = Agent(
    role="数据分析师",
    goal="分析数据，发现规律和洞察",
    backstory="你是一个精通统计学和数据可视化的分析专家",
    tools=[],
    llm=llm,
)

report_writer = Agent(
    role="报告撰写专员",
    goal="将分析结果转化为清晰的书面报告",
    backstory="你擅长商业写作，能将复杂数据变为易读报告",
    llm=llm,
)

# 创建任务（层级模式下任务无需指定 agent）
analysis_task = Task(
    description="分析2024年的销售数据，找出增长最快的产品线和区域",
    expected_output="数据分析报告，含图表描述和关键数据点",
    agent=data_analyst,
)

report_task = Task(
    description="基于数据分析结果，撰写季度业务回顾报告",
    expected_output="3000字的季度回顾报告",
    agent=report_writer,
    context=[analysis_task],
)

# 层级模式的 Crew
hierarchical_crew = Crew(
    agents=[data_analyst, report_writer],
    tasks=[analysis_task, report_task],
    process=Process.hierarchical,
    manager_agent=manager,    # 指定管理者
    verbose=True,
)

result = hierarchical_crew.kickoff()
```

---

## 八、记忆系统

CrewAI 提供多层次的记忆机制：

```python
from crewai import Crew
from crewai.memory import (
    ShortTermMemory,    # 当前任务执行期间的记忆
    LongTermMemory,     # 跨任务的长期记忆（基于向量存储）
    EntityMemory,       # 实体记忆（人物、组织、概念）
)
from crewai.memory.storage.rag_storage import RAGStorage

# 配置长期记忆存储
long_term_memory = LongTermMemory(
    storage=RAGStorage(
        embedder_config={
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"}
        },
        type="short_term",
        path="./crew_memory"
    )
)

crew_with_memory = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    memory=True,
    verbose=True,
)

# 第一次运行（建立记忆）
result1 = crew_with_memory.kickoff(inputs={"topic": "AI芯片市场"})

# 第二次运行（会利用之前的记忆）
result2 = crew_with_memory.kickoff(inputs={"topic": "AI芯片市场竞争格局"})
# Agent 会记得上次研究的内容，避免重复工作
```

---

## 九、异步执行与并行任务

```python
from crewai import Task
import asyncio

# 可以并行执行的任务（互不依赖）
task_china = Task(
    description="研究中国生成式AI市场",
    expected_output="中国市场研究报告",
    agent=researcher,
    async_execution=True,  # 标记为异步执行
)

task_us = Task(
    description="研究美国生成式AI市场",
    expected_output="美国市场研究报告",
    agent=researcher,
    async_execution=True,  # 与 task_china 并行执行
)

# 依赖前两个任务的对比任务（同步）
comparison_task = Task(
    description="对比中美两国的生成式AI市场，分析差异和各自优势",
    expected_output="对比分析报告",
    agent=writer,
    context=[task_china, task_us],  # 等待两个异步任务完成
    async_execution=False,
)

parallel_crew = Crew(
    agents=[researcher, writer],
    tasks=[task_china, task_us, comparison_task],
    process=Process.sequential,
    verbose=True,
)

result = parallel_crew.kickoff()
```

---

## 十、实战案例：自动化内容运营团队

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
search_tool = SerperDevTool()

# 定义运营团队
trend_scout = Agent(
    role="热点追踪专员",
    goal="发现当前技术领域的热门话题和讨论趋势",
    backstory="你整天泡在 Twitter、HackerNews、Reddit 上，对技术热点极度敏感",
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

content_creator = Agent(
    role="技术内容创作者",
    goal="将热点话题转化为高质量、有深度的技术内容",
    backstory="你是一个有5年经验的技术博主，擅长深入浅出地解析复杂概念",
    llm=llm,
    verbose=True,
)

seo_optimizer = Agent(
    role="SEO 优化专家",
    goal="优化内容的搜索引擎友好性，提高曝光率",
    backstory="你精通内容 SEO，了解各平台算法规则，能在不影响内容质量的情况下提升排名",
    llm=llm,
    verbose=True,
)

# 定义工作流任务
find_trends = Task(
    description="""搜索过去24小时内{domain}领域的热门话题。
    重点关注：GitHub 热门项目、技术论坛热帖、学术新成果
    输出：Top5热点话题，每个附带热度说明和核心亮点""",
    expected_output="5个热点话题列表，每个含话题标题、热度评分、核心亮点",
    agent=trend_scout,
)

create_content = Task(
    description="""选取最有价值的话题，创作一篇{content_type}。
    要求：标题吸引眼球、内容有深度、包含代码示例（如适用）、
    结尾有行动号召（CTA）""",
    expected_output="一篇完整的{content_type}，Markdown格式，1000-2000字",
    agent=content_creator,
    context=[find_trends],
)

optimize_seo = Task(
    description="""对创作的内容进行SEO优化：
    1. 优化标题（包含核心关键词）
    2. 添加合适的标签（5-10个）
    3. 优化首段（黄金30字）
    4. 检查关键词密度
    5. 添加内链建议""",
    expected_output="优化后的内容（含优化说明）和推荐标签列表",
    agent=seo_optimizer,
    context=[create_content],
)

# 组建团队并执行
content_crew = Crew(
    agents=[trend_scout, content_creator, seo_optimizer],
    tasks=[find_trends, create_content, optimize_seo],
    process=Process.sequential,
    verbose=True,
)

final_content = content_crew.kickoff(inputs={
    "domain": "AI Agent 框架",
    "content_type": "技术深度文章"
})

print(final_content.raw)
```

---

## 十一、CrewAI 最佳实践

### 1. 角色设计原则
- **具体而非泛泛**：`"资深 Python 后端工程师，专注微服务架构"` > `"程序员"`
- **背景故事要真实**：详细的 backstory 能显著提升 Agent 行为的专业度
- **目标要量化**：`"生成3-5个高质量建议"` > `"提供建议"`

### 2. 任务设计原则
- **expected_output 要精确**：明确告诉 Agent 什么样的输出算完成
- **合理使用 context**：只传入真正需要的上下文，避免信息冗余
- **任务粒度适中**：太大的任务容易偏题，太小的任务增加协调成本

### 3. 性能优化
```python
# 使用更快的模型处理简单任务
simple_agent = Agent(
    role="格式化专员",
    llm=ChatOpenAI(model="gpt-4o-mini"),  # 便宜快速
    ...
)

# 重要任务使用更强的模型
critical_agent = Agent(
    role="首席分析师",
    llm=ChatOpenAI(model="gpt-4o"),  # 更强但更贵
    ...
)

# 缓存重复任务结果
crew = Crew(
    ...
    memory=True,  # 避免重复计算
)
```

---

## 小结

CrewAI 的核心价值在于：
1. **角色模拟**：通过 role + goal + backstory 精确塑造 Agent 行为
2. **任务编排**：通过 context 和 async_execution 灵活控制任务依赖
3. **工具生态**：与 LangChain 工具生态无缝兼容
4. **记忆系统**：多层次记忆让 Agent 积累经验

下一篇（D6）将对 LangChain、LlamaIndex、AutoGen、CrewAI 四大框架进行横向对比与选型建议。
