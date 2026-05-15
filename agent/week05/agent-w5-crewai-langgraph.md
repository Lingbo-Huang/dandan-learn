---
layout: default
title: "W5D5 · CrewAI 与 LangGraph"
---

# CrewAI 与 LangGraph：另两个多 Agent 框架

> **Week 5 · Day 5** | 难度：⭐⭐⭐⭐

---

## CrewAI：像管理真实团队一样管理 Agent

CrewAI 的设计哲学：把 Agent 当"员工"来管理——每个 Agent 有角色、目标、背景故事。

```bash
pip install crewai crewai-tools
```

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool

# 创建工具
search_tool = SerperDevTool()

# 定义角色
researcher = Agent(
    role="市场研究专家",
    goal="收集最新的、准确的市场数据和竞品信息",
    backstory="""你有10年市场研究经验，擅长从复杂数据中
提炼关键洞察。你以严谨著称，从不引用不可靠来源。""",
    tools=[search_tool],
    verbose=True,
    allow_delegation=False,  # 不允许将任务转给其他 Agent
    llm="gpt-4o"
)

analyst = Agent(
    role="数据分析师",
    goal="将原始数据转化为可操作的商业洞察",
    backstory="""你是数据分析专家，擅长用数字讲故事。
你的分析总是有据可查，结论清晰可行。""",
    verbose=True,
    allow_delegation=False,
    llm="gpt-4o-mini"
)

writer = Agent(
    role="商业报告写作专家",
    goal="撰写清晰、专业、有说服力的商业报告",
    backstory="""你是资深商业写作者，写过数百份高管报告。
你的文章逻辑清晰、语言精炼，始终以读者为中心。""",
    verbose=True,
    allow_delegation=True,   # 可以将部分任务转给其他 Agent
    llm="gpt-4o"
)

# 定义任务
research_task = Task(
    description="""研究 {topic} 的市场现状：
    1. 市场规模和增长率
    2. 主要参与者（前5名）
    3. 近期重大事件和趋势
    4. 主要挑战和机会
    
    输出格式：结构化的市场调研报告""",
    expected_output="一份包含数据支撑的市场调研报告，不少于500字",
    agent=researcher,
    output_file="research_output.md"
)

analysis_task = Task(
    description="""基于市场调研报告，进行深度分析：
    1. 识别最重要的3个市场趋势
    2. 评估机会与威胁
    3. 确定关键成功因素
    4. 给出量化评估（市场吸引力评分等）""",
    expected_output="一份深度分析报告，包含趋势分析和评分",
    agent=analyst,
    context=[research_task]   # 依赖研究任务的输出
)

report_task = Task(
    description="""综合调研和分析，撰写最终报告：
    - 执行摘要（200字以内）
    - 市场分析
    - 战略建议（3-5条）
    - 风险提示
    
    目标读者：C-suite 高管""",
    expected_output="一份专业的商业分析报告（Markdown格式，1000字以上）",
    agent=writer,
    context=[research_task, analysis_task],
    output_file="final_report.md"
)

# 创建 Crew
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, report_task],
    process=Process.sequential,  # 顺序执行
    verbose=True,
    memory=True,               # 启用 Crew 记忆
    embedder={
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"}
    }
)

# 运行
result = crew.kickoff(inputs={"topic": "AI Agent 开发框架市场"})
print(result)
```

## CrewAI 的层级执行模式

```python
# 层级模式：有经理 Agent 分配工作
crew_hierarchical = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, report_task],
    process=Process.hierarchical,  # 层级模式
    manager_llm="gpt-4o",         # 经理用更强的模型
    verbose=True
)
```

## LangGraph：用状态机构建 Agent 工作流

LangGraph 的核心：把 Agent 工作流建模为**有状态的图**，每个节点是一个处理步骤，边是状态转移。

```bash
pip install langgraph
```

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Annotated
import operator

# 定义状态结构
class WorkflowState(TypedDict):
    """工作流的全局状态"""
    task: str
    research: str
    analysis: str
    critique: str
    final_report: str
    revision_count: int
    approved: bool
    messages: Annotated[List[str], operator.add]

# 定义节点函数
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def research_node(state: WorkflowState) -> WorkflowState:
    """研究节点"""
    response = llm.invoke(f"研究以下主题，提供数据支撑的分析：{state['task']}")
    return {
        **state,
        "research": response.content,
        "messages": [f"研究完成（{len(response.content)}字）"]
    }

def analyze_node(state: WorkflowState) -> WorkflowState:
    """分析节点"""
    prompt = f"基于以下研究，提取关键洞察：\n{state['research']}"
    response = llm.invoke(prompt)
    return {
        **state,
        "analysis": response.content,
        "messages": ["分析完成"]
    }

def critique_node(state: WorkflowState) -> WorkflowState:
    """批评节点"""
    combined = f"研究：{state['research']}\n\n分析：{state['analysis']}"
    prompt = f"严格批评以下内容，找出不足：\n{combined}"
    response = llm.invoke(prompt)
    return {
        **state,
        "critique": response.content,
        "messages": ["批评完成"]
    }

def write_report_node(state: WorkflowState) -> WorkflowState:
    """写报告节点"""
    context = f"""
任务：{state['task']}
研究：{state['research']}
分析：{state['analysis']}
批评意见：{state['critique']}
"""
    prompt = f"综合以上信息，写一份专业报告：\n{context}"
    response = llm.invoke(prompt)
    return {
        **state,
        "final_report": response.content,
        "revision_count": state.get("revision_count", 0) + 1,
        "messages": [f"报告生成（第{state.get('revision_count', 0)+1}版）"]
    }

def review_node(state: WorkflowState) -> WorkflowState:
    """审核节点"""
    prompt = f"""审核以下报告的质量：
{state['final_report']}

质量是否达到发布标准？（只回答 yes 或 no）"""
    response = llm.invoke(prompt)
    approved = "yes" in response.content.lower()
    return {
        **state,
        "approved": approved,
        "messages": [f"审核结果：{'通过' if approved else '需修改'}"]
    }

# 条件边：根据审核结果决定下一步
def should_revise(state: WorkflowState) -> str:
    """决定是否需要修改"""
    if state.get("approved"):
        return "approved"
    elif state.get("revision_count", 0) >= 3:
        return "max_revisions"  # 达到最大修改次数，强制通过
    return "revise"

# 构建图
workflow = StateGraph(WorkflowState)

# 添加节点
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("critique", critique_node)
workflow.add_node("write_report", write_report_node)
workflow.add_node("review", review_node)

# 添加边（工作流顺序）
workflow.set_entry_point("research")
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", "critique")
workflow.add_edge("critique", "write_report")
workflow.add_edge("write_report", "review")

# 添加条件边（review 后的分支）
workflow.add_conditional_edges(
    "review",
    should_revise,
    {
        "approved": END,           # 通过 → 结束
        "revise": "critique",      # 不通过 → 重新批评并修改
        "max_revisions": END,      # 达到上限 → 强制结束
    }
)

# 编译并运行
app = workflow.compile()

result = app.invoke({
    "task": "分析 LangGraph 在 2024 年的市场地位和竞争格局",
    "research": "",
    "analysis": "",
    "critique": "",
    "final_report": "",
    "revision_count": 0,
    "approved": False,
    "messages": []
})

print("最终报告：")
print(result["final_report"][:500])
print(f"\n消息日志：")
for msg in result["messages"]:
    print(f"  • {msg}")
```

## 框架对比

```
维度              AutoGen          CrewAI           LangGraph
──────────────────────────────────────────────────────────────
学习曲线          中等              低                高
灵活性            高               中                最高
代码执行          原生支持          需要工具          自定义
角色管理          对话驱动          职位驱动          状态机
工作流控制        对话流            任务链            图结构
调试难度          中                低               高（可视化好）
适用场景          代码任务          业务流程          复杂工作流
```

## 踩坑经验

### CrewAI 坑：任务之间上下文传递不完整

**问题**：`context=[research_task]` 设置了，但 analyst 看不到完整的 research 结果。  
**解法**：确保 `output_file` 和 `context` 同时设置；或在任务 description 中明确说明"基于前一个任务的输出"。

### LangGraph 坑：状态类型错误导致图无法编译

**问题**：`TypedDict` 的字段类型不匹配时，`compile()` 会报错但信息不清晰。  
**解法**：先用简单的 `dict` 测试，确认图结构正确后再加类型约束。

---

*W5D5 · CrewAI 与 LangGraph | Agent + Claw 系列*
