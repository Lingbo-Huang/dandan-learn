# D3 · Agent 规划能力——让 AI 处理复杂长程任务

> **Week 1 主题**：什么是 Agent——定义 / ReAct / 规划 / 记忆 / 工具调用  
> **本日主题**：规划（Planning）——CoT / ToT / Plan-and-Execute

---

## 🎯 学习目标

1. 理解为什么简单 ReAct 不足以处理复杂任务
2. 掌握 Chain-of-Thought、Tree-of-Thoughts 的核心思想
3. 理解 Plan-and-Execute 架构及其实现
4. 能够根据任务复杂度选择合适的规划策略

---

## 📚 核心知识点

### 1. 为什么需要"规划"？

ReAct 的局限：
- 贪心搜索：只考虑当前最优步，不考虑长远
- 无法回溯：走错了路难以纠正
- 上下文污染：中间失败步骤干扰后续推理

**规划**的本质：在行动之前**先想清楚整体路线**。

```
无规划（ReAct）：走一步看一步
有规划（Plan-and-Execute）：先画地图，再出发
```

### 2. Chain-of-Thought（CoT）

**思维链**：让 LLM 一步步展示推理过程

```
问：小明有8个苹果，给了3个给小红，又买了5个，现在有几个？

CoT：
1. 初始：8个
2. 给小红后：8 - 3 = 5个
3. 买了5个后：5 + 5 = 10个
答：10个
```

**Zero-shot CoT**：只需在提示词中加 "Let's think step by step"

**Few-shot CoT**：提供示例推理链

**局限**：线性单路径，不能探索多种可能性

### 3. Tree-of-Thoughts（ToT）

**思维树**：探索多条推理路径，择优选择

```
问题
├── 思路A
│   ├── A1 → 评估：好
│   └── A2 → 评估：差（剪枝）
├── 思路B
│   ├── B1 → 评估：优秀 ✓
│   └── B2 → 评估：一般
└── 思路C → 评估：不可行（剪枝）

选择路径：B → B1
```

**三个关键组件**：
1. **Thought Generator**：生成多个候选思路
2. **State Evaluator**：评估每个状态的价值
3. **Search Algorithm**：BFS 或 DFS 遍历思维树

**适用场景**：需要创意的任务、数学证明、策略游戏

### 4. Plan-and-Execute 架构

最实用的生产级规划模式：

```
Phase 1: PLAN（规划器）
  输入：用户目标
  输出：结构化任务列表 [Task1, Task2, Task3, ...]

Phase 2: EXECUTE（执行器）
  对每个 Task 运行 ReAct Agent
  收集结果

Phase 3: REVISE（可选，修正器）
  根据执行结果动态调整计划
```

**与纯 ReAct 的对比**：

| 维度 | ReAct | Plan-and-Execute |
|------|-------|-----------------|
| 规划粒度 | 步步为营 | 先全局后局部 |
| 可解释性 | 中 | 高（计划透明） |
| 处理长任务 | 弱 | 强 |
| 并行执行 | 不支持 | 支持 |
| 实现复杂度 | 低 | 中 |

### 5. 任务分解策略

**自顶向下分解**（适合结构清晰的任务）：
```
目标：写一份竞品分析报告
├── 1. 确定竞品范围（列出5个竞品）
├── 2. 收集各竞品数据（并行）
│   ├── 2.1 收集竞品A数据
│   ├── 2.2 收集竞品B数据
│   └── 2.3 ...
├── 3. 分析对比
└── 4. 撰写报告
```

**依赖分析**：识别哪些任务可以并行，哪些必须串行

### 6. 动态重规划

好的规划 Agent 应该能够：
1. 检测计划失败（子任务 error）
2. 分析失败原因
3. 生成修复计划（跳过/重试/替代方案）
4. 继续执行

这正是 ALLIN Claw 项目会话龙虾的核心能力！

---

## 💡 示例/推导

### Plan-and-Execute 完整示例

**目标**："帮我分析 Python 和 JavaScript 在 AI 开发领域的对比"

```
=== PLAN 阶段 ===

Planner 输出：
{
  "goal": "分析 Python 和 JavaScript 在 AI 开发领域的对比",
  "tasks": [
    {
      "id": 1,
      "title": "收集 Python AI 生态数据",
      "dependencies": [],
      "tool": "web_search"
    },
    {
      "id": 2, 
      "title": "收集 JavaScript AI 生态数据",
      "dependencies": [],
      "tool": "web_search"
    },
    {
      "id": 3,
      "title": "分析 GitHub 项目数量对比",
      "dependencies": [1, 2],
      "tool": "github_api"
    },
    {
      "id": 4,
      "title": "汇总并生成对比报告",
      "dependencies": [1, 2, 3],
      "tool": "text_generation"
    }
  ]
}

=== EXECUTE 阶段 ===

Task 1（并行）: 搜索 → 获取 Python AI 库列表
Task 2（并行）: 搜索 → 获取 JS AI 库列表
Task 3（依赖1,2完成）: 查询 GitHub → 获取 star 数量对比
Task 4（依赖1,2,3完成）: 生成对比报告

=== 输出 ===
一份结构化的 Python vs JavaScript AI 生态对比报告
```

---

## 🔧 动手练习

### 练习 1：手写任务分解（必做）

为以下目标设计任务分解树，标注依赖关系和可并行的任务：

**目标**："创建一个个人技术博客网站"

要求：
- 至少分解到 8 个子任务
- 标注哪些可以并行执行
- 识别关键路径（最长依赖链）

### 练习 2：实现简单规划器（必做）

```python
# 创建文件: 04_simple_planner.py
# uv run python 04_simple_planner.py

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"

@dataclass
class Task:
    id: int
    title: str
    dependencies: list[int] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None

class SimplePlanner:
    """简单任务规划器，管理任务依赖和执行顺序"""
    
    def __init__(self, tasks: list[Task]):
        self.tasks = {t.id: t for t in tasks}
    
    def get_ready_tasks(self) -> list[Task]:
        """获取所有依赖已满足、可以执行的任务"""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            # 检查所有依赖是否完成
            deps_done = all(
                self.tasks[dep_id].status == TaskStatus.DONE
                for dep_id in task.dependencies
            )
            if deps_done:
                ready.append(task)
        return ready
    
    def complete_task(self, task_id: int, result: str):
        """标记任务完成"""
        self.tasks[task_id].status = TaskStatus.DONE
        self.tasks[task_id].result = result
        print(f"  ✅ 任务 {task_id} 完成: {result}")
    
    def is_all_done(self) -> bool:
        return all(t.status == TaskStatus.DONE for t in self.tasks.values())
    
    def run(self):
        """模拟执行计划"""
        print("🗺️ 开始执行计划\n")
        round_num = 0
        
        while not self.is_all_done():
            round_num += 1
            ready = self.get_ready_tasks()
            
            if not ready:
                print("❌ 没有可执行的任务（可能有循环依赖）")
                break
            
            print(f"--- 第 {round_num} 轮（可并行执行 {len(ready)} 个任务）---")
            for task in ready:
                task.status = TaskStatus.RUNNING
                print(f"  ⚡ 执行: [{task.id}] {task.title}")
                # 模拟执行
                result = f"[{task.title}] 执行完成"
                self.complete_task(task.id, result)
            print()
        
        print("🎉 所有任务执行完毕！")

# 测试：竞品分析任务
tasks = [
    Task(1, "确定竞品范围", []),
    Task(2, "收集竞品A数据", [1]),
    Task(3, "收集竞品B数据", [1]),
    Task(4, "收集竞品C数据", [1]),
    Task(5, "数据对比分析", [2, 3, 4]),
    Task(6, "撰写报告", [5]),
]

planner = SimplePlanner(tasks)
planner.run()
```

### 练习 3：Plan-and-Execute Agent（进阶）

```python
# 创建文件: 05_plan_execute_agent.py
# uv run python 05_plan_execute_agent.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 规划器提示词
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个任务规划专家。将用户目标分解为具体的子任务列表。
    
输出格式（JSON）：
{{
  "tasks": [
    {{"id": 1, "title": "任务标题", "dependencies": [], "description": "详细描述"}},
    {{"id": 2, "title": "任务标题", "dependencies": [1], "description": "详细描述"}}
  ]
}}

规则：
- 每个任务要具体可执行
- 正确标注依赖关系
- 可并行的任务不要设置不必要的依赖"""),
    ("user", "请为以下目标制定执行计划：{goal}")
])

def plan(goal: str) -> list[dict]:
    """使用 LLM 生成任务计划"""
    chain = planner_prompt | llm | JsonOutputParser()
    result = chain.invoke({"goal": goal})
    return result["tasks"]

def execute_task(task: dict, context: dict) -> str:
    """模拟执行单个任务"""
    exec_prompt = f"""
    执行任务：{task['title']}
    任务描述：{task['description']}
    前置结果：{json.dumps(context, ensure_ascii=False)}
    
    请执行此任务并给出结果（简短描述）。
    """
    response = llm.invoke(exec_prompt)
    return response.content

def run_plan_execute(goal: str):
    print(f"🎯 目标: {goal}\n")
    
    # Phase 1: 规划
    print("📋 Phase 1: 生成任务计划...")
    tasks = plan(goal)
    print(f"生成了 {len(tasks)} 个任务：")
    for t in tasks:
        deps = f"（依赖: {t['dependencies']}）" if t['dependencies'] else "（无依赖）"
        print(f"  [{t['id']}] {t['title']} {deps}")
    
    # Phase 2: 执行
    print("\n⚡ Phase 2: 执行任务计划...")
    task_map = {t["id"]: t for t in tasks}
    results = {}
    completed = set()
    
    while len(completed) < len(tasks):
        # 找出可执行的任务
        ready = [
            t for t in tasks
            if t["id"] not in completed
            and all(dep in completed for dep in t["dependencies"])
        ]
        
        for task in ready:
            print(f"\n  ▶ 执行: [{task['id']}] {task['title']}")
            context = {k: results[k] for k in task["dependencies"] if k in results}
            result = execute_task(task, context)
            results[task["id"]] = result
            completed.add(task["id"])
            print(f"  ✅ 结果: {result[:100]}...")
    
    print("\n🎉 所有任务完成！")
    return results

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_plan_execute("分析国内外主流 AI Agent 框架的特点和适用场景")
```

### 🦞 Claw 实战：观察 Claw 的 Plan-and-Execute 模式

当前项目「六条线Week1规划文档生成」就是一个典型的 Plan-and-Execute 系统：

1. **规划层**：项目会话（龙虾）将目标拆解为 6 条线的任务
2. **执行层**：每个 Worker（包括本 Agent）独立执行自己的任务
3. **依赖管理**：平台自动管理任务间的依赖关系

**观察练习**：
- 在项目中找到你的任务 stepId（本任务是 stepId=1）
- 思考：其他 stepId 的任务与本任务是什么关系？
- 这和今天学的"依赖分析"有什么对应关系？

---

## 📝 小结

| 规划策略 | 适用场景 | 核心特点 |
|---------|---------|---------|
| CoT | 简单推理任务 | 线性步骤，易实现 |
| ToT | 创意/多解问题 | 探索多路径，成本高 |
| Plan-and-Execute | 复杂多步任务 | 全局规划，支持并行 |
| ReAct（无显式规划） | 简单工具调用 | 轻量，适合快速任务 |

**明天预告**：Agent 的记忆系统——短期记忆、长期记忆、向量检索，让 Agent 真正"记住"事情。

---

> 💡 **今日思考题**：Claw 的项目会话龙虾是"规划器"，Worker Agent 是"执行器"——这和 Plan-and-Execute 架构完全对应。你觉得 Claw 在"动态重规划"（某个任务失败后）方面是如何处理的？
