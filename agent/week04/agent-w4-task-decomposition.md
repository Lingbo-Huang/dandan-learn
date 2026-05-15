---
layout: default
title: "W4D5 · 任务分解与规划"
---

# 任务分解：让 Agent 驾驭复杂任务

> **Week 4 · Day 5** | 难度：⭐⭐⭐⭐

---

## 为什么需要任务分解？

直接让 Agent 解决复杂任务，就像让新手司机直接开高速——能做，但极易出错。任务分解（Task Decomposition）的核心思想：**把大问题递归拆解为小问题，直到每个小问题都可以直接解决**。

```
复杂任务："分析竞品并给出产品改进建议"
    │
    ├── 子任务1：收集竞品信息
    │       ├── 搜索竞品A的功能列表
    │       ├── 搜索竞品B的用户评价
    │       └── 整理竞品功能对比表
    │
    ├── 子任务2：分析自身产品
    │       ├── 提取当前核心功能
    │       └── 识别用户痛点
    │
    └── 子任务3：生成改进建议
            ├── 找差距
            └── 优先级排序
```

## 三种分解策略

### 策略1：层次任务网络（HTN）

预定义任务模板，自动匹配和分解：

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import json

class SubTask(BaseModel):
    id: str
    title: str
    description: str
    dependencies: List[str] = Field(default_factory=list)
    estimated_steps: int = 1

class TaskPlan(BaseModel):
    goal: str
    subtasks: List[SubTask]
    execution_order: List[str]

class HTNPlanner:
    """层次任务网络规划器"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.plan_llm = self.llm.with_structured_output(TaskPlan)
    
    def decompose(self, goal: str, context: str = "") -> TaskPlan:
        """将目标分解为子任务计划"""
        prompt = f"""将以下目标分解为可执行的子任务：

目标：{goal}
{"上下文：" + context if context else ""}

请创建一个任务计划，要求：
1. 每个子任务有唯一 ID（t1, t2, t3...）
2. 明确依赖关系（哪些任务必须在别的任务之后执行）
3. 每个子任务都是原子性的（可以被直接执行）
4. 给出推荐执行顺序"""
        
        return self.plan_llm.invoke(prompt)
    
    def visualize_plan(self, plan: TaskPlan) -> str:
        """将计划可视化为 ASCII 图"""
        lines = [f"目标：{plan.goal}", ""]
        
        # 按执行顺序展示
        for task_id in plan.execution_order:
            task = next((t for t in plan.subtasks if t.id == task_id), None)
            if task:
                deps = f" (依赖: {', '.join(task.dependencies)})" if task.dependencies else ""
                lines.append(f"  [{task.id}] {task.title}{deps}")
                lines.append(f"       → {task.description}")
                lines.append("")
        
        return "\n".join(lines)

# 示例
planner = HTNPlanner()
plan = planner.decompose(
    "为一个 SaaS 产品写一份完整的竞品分析报告",
    "产品：AI写作助手，目标市场：中国企业用户"
)
print(planner.visualize_plan(plan))
```

### 策略2：LLM Planner（动态分解）

让 LLM 根据任务动态生成执行计划，并在执行中调整：

```python
from langchain.tools import tool
from typing import Any

@tool
def execute_subtask(subtask_description: str) -> str:
    """执行一个具体的子任务"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm.invoke(f"请完成以下任务并给出结果：\n{subtask_description}").content

class DynamicPlanner:
    """动态任务规划器——边执行边调整计划"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.completed_tasks: List[Dict] = []
        self.current_plan: List[str] = []
    
    def initial_plan(self, goal: str) -> List[str]:
        """生成初始计划"""
        prompt = f"""为以下目标创建初始执行计划：

目标：{goal}

列出5-8个执行步骤（每行一个步骤，简洁描述）："""
        
        response = self.llm.invoke(prompt)
        steps = [line.strip().lstrip('0123456789.-) ') 
                 for line in response.content.split('\n') 
                 if line.strip() and line.strip()[0].isdigit() or line.strip().startswith('-')]
        return steps
    
    def update_plan(self, goal: str, remaining_steps: List[str], 
                    completed_results: List[Dict]) -> List[str]:
        """根据执行结果更新剩余计划"""
        completed_summary = "\n".join([
            f"✅ {r['task']}：{r['result'][:100]}..."
            for r in completed_results
        ])
        
        prompt = f"""目标：{goal}

已完成的步骤：
{completed_summary}

原定剩余步骤：
{chr(10).join(remaining_steps)}

根据已完成的工作，请调整剩余步骤（可以删除已不需要的步骤，添加新发现的必要步骤）："""
        
        response = self.llm.invoke(prompt)
        return [line.strip().lstrip('0123456789.-) ') 
                for line in response.content.split('\n') 
                if line.strip()]
    
    def execute(self, goal: str) -> str:
        """执行完整的动态规划"""
        print(f"目标：{goal}\n")
        
        # 生成初始计划
        plan = self.initial_plan(goal)
        print("初始计划：")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")
        
        completed = []
        
        # 逐步执行
        while plan:
            current_step = plan[0]
            plan = plan[1:]
            
            print(f"\n执行：{current_step}")
            result = execute_subtask.invoke({"subtask_description": current_step})
            
            completed.append({"task": current_step, "result": result})
            print(f"  结果：{result[:100]}...")
            
            # 每完成2步，重新评估剩余计划
            if len(completed) % 2 == 0 and plan:
                plan = self.update_plan(goal, plan, completed)
                print(f"\n计划已更新，剩余 {len(plan)} 步")
        
        # 整合最终结果
        return self.synthesize_results(goal, completed)
    
    def synthesize_results(self, goal: str, completed: List[Dict]) -> str:
        """整合所有子任务结果"""
        results_text = "\n\n".join([
            f"**{r['task']}**\n{r['result']}"
            for r in completed
        ])
        
        prompt = f"""请整合以下子任务的结果，形成对目标的完整回答：

目标：{goal}

子任务结果：
{results_text}

请给出整合后的完整答案："""
        
        return self.llm.invoke(prompt).content

# 使用
planner = DynamicPlanner()
result = planner.execute("调研当前最流行的5个 Python 异步框架，并给出选型建议")
print(f"\n最终结果：\n{result}")
```

### 策略3：Plan-and-Execute（规划-执行分离）

LangChain 官方提供的实现：

```python
from langchain_experimental.plan_and_execute import (
    PlanAndExecute, 
    load_agent_executor, 
    load_chat_planner
)
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# 准备工具
search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [search, wikipedia]

# 初始化 Plan-and-Execute Agent
model = ChatOpenAI(model="gpt-4o", temperature=0)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# 执行复杂任务
result = agent.run(
    "研究 2024 年大型语言模型的主要进展，"
    "重点关注推理能力的提升，并总结 3 个最重要的突破。"
)
print(result)
```

## 任务依赖图与并行执行

当任务间有依赖关系时，可以并行执行独立任务：

```python
import asyncio
from typing import Dict, Set

class ParallelTaskExecutor:
    """支持依赖关系的并行任务执行器"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.results: Dict[str, str] = {}
    
    async def execute_task(self, task_id: str, task: SubTask, 
                          dep_results: Dict[str, str]) -> str:
        """执行单个任务"""
        # 构建带依赖结果的上下文
        context = ""
        if dep_results:
            context = "前置任务结果：\n" + "\n".join([
                f"- {k}：{v[:200]}" for k, v in dep_results.items()
            ])
        
        prompt = f"""执行以下任务：
{task.description}

{context}

请给出完整结果："""
        
        print(f"  开始执行 [{task_id}] {task.title}...")
        response = await self.llm.ainvoke(prompt)
        result = response.content
        print(f"  ✅ 完成 [{task_id}]")
        return result
    
    async def execute_plan(self, plan: TaskPlan) -> Dict[str, str]:
        """按依赖关系并行执行计划"""
        pending: Set[str] = {t.id for t in plan.subtasks}
        task_map = {t.id: t for t in plan.subtasks}
        
        while pending:
            # 找出所有依赖已满足的任务（可以并行执行）
            ready = [
                tid for tid in pending
                if all(dep in self.results for dep in task_map[tid].dependencies)
            ]
            
            if not ready:
                raise ValueError("检测到循环依赖！")
            
            print(f"\n并行执行 {len(ready)} 个任务：{ready}")
            
            # 并行执行所有就绪任务
            tasks = []
            for tid in ready:
                task = task_map[tid]
                dep_results = {dep: self.results[dep] for dep in task.dependencies}
                tasks.append(self.execute_task(tid, task, dep_results))
            
            results = await asyncio.gather(*tasks)
            
            for tid, result in zip(ready, results):
                self.results[tid] = result
                pending.remove(tid)
        
        return self.results

# 使用示例
async def main():
    planner = HTNPlanner()
    plan = planner.decompose("准备一份 AI 产品发布会演讲稿")
    
    executor = ParallelTaskExecutor()
    results = await executor.execute_plan(plan)
    
    print("\n所有任务完成！")
    for task_id, result in results.items():
        print(f"\n[{task_id}] {result[:200]}...")

asyncio.run(main())
```

## 踩坑经验

### 坑1：分解粒度不合适——太细或太粗

**问题**：任务分解过细导致 API 调用爆炸；过粗则等于没分解。  
**经验法则**：
- 每个子任务应该能在 1-3 次 LLM 调用内完成
- 子任务数量通常在 3-10 个之间
- 如果某个子任务需要超过 5 步，继续分解

### 坑2：忽略任务依赖——并行了不能并行的任务

**问题**：任务 B 需要任务 A 的结果，但两个任务被并行执行了。  
**解法**：始终在分解时明确标注依赖关系，执行前检查依赖图无环。

### 坑3：子任务结果整合质量差

**问题**：子任务分别执行得很好，但最终整合时上下文丢失，整合结果质量低。  
**解法**：整合步骤单独作为一个任务，并且在 prompt 中提供所有子任务的完整结果，不要截断。

## 小结

任务分解是 Agent 处理复杂任务的基础能力：
1. **HTN**：结构清晰，适合有固定流程的任务
2. **动态规划**：灵活，适合探索性任务
3. **Plan-and-Execute**：LangChain 原生支持，开箱即用
4. **并行执行**：显著提升效率，但需正确处理依赖

> **下一篇**：综合实战——把本周所有推理技术组合起来，解决真实复杂问题。

---

*W4D5 · 任务分解与规划 | Agent + Claw 系列*
