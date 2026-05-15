---
layout: default
title: "D4 · LLM Agent 系统设计"
render_with_liquid: false
---

# D4 · LLM Agent 系统设计

> **Agent = LLM + 记忆 + 工具 + 规划。** 单次 LLM 调用只能回答问题，Agent 能完成复杂任务。

---

## 一、Agent 基本结构

```
┌─────────────────────────────────────────────┐
│                   Agent                      │
│   ┌────────┐  ┌────────┐  ┌───────────────┐ │
│   │ Memory │  │ Tools  │  │   Planner     │ │
│   │ 短期/  │  │ 搜索/  │  │ ReAct/CoT/   │ │
│   │ 长期   │  │ 代码/  │  │ Plan-Execute  │ │
│   └────────┘  │ API... │  └───────────────┘ │
│               └────────┘                     │
│   ┌─────────────────────────────────────────┐│
│   │              LLM（大脑）                 ││
│   └─────────────────────────────────────────┘│
└─────────────────────────────────────────────┘
```

---

## 二、记忆系统

```python
from collections import deque
from typing import Any
import json

class AgentMemory:
    """Agent 记忆系统"""
    
    def __init__(
        self,
        short_term_limit: int = 20,    # 对话历史保留条数
        working_memory_limit: int = 10, # 工作记忆（当前任务）
    ):
        # 短期记忆：对话历史
        self.conversation_history = deque(maxlen=short_term_limit)
        
        # 工作记忆：当前任务的中间结果
        self.working_memory: dict[str, Any] = {}
        
        # 长期记忆：重要事实（可以外接向量数据库）
        self.long_term_memory: list[dict] = []
    
    def add_message(self, role: str, content: str) -> None:
        """添加对话消息"""
        self.conversation_history.append({"role": role, "content": content})
    
    def save_fact(self, key: str, value: Any) -> None:
        """保存工作记忆中的事实"""
        self.working_memory[key] = value
    
    def get_fact(self, key: str, default=None) -> Any:
        """获取工作记忆"""
        return self.working_memory.get(key, default)
    
    def commit_to_long_term(self, content: str, importance: float = 1.0) -> None:
        """将重要信息存入长期记忆"""
        self.long_term_memory.append({
            "content": content,
            "importance": importance,
        })
    
    def get_context(self, max_tokens: int = 2000) -> str:
        """获取用于 prompt 的记忆上下文"""
        parts = []
        
        # 工作记忆
        if self.working_memory:
            facts = "\n".join(f"- {k}: {v}" for k, v in self.working_memory.items())
            parts.append(f"当前任务信息：\n{facts}")
        
        # 对话历史
        history = list(self.conversation_history)
        parts.append("对话历史：\n" + "\n".join(
            f"{m['role']}: {m['content'][:100]}" for m in history[-5:]
        ))
        
        return "\n\n".join(parts)
    
    def get_messages(self) -> list[dict]:
        """获取完整对话历史（用于 LLM API）"""
        return list(self.conversation_history)
```

---

## 三、Plan-and-Execute Agent

```python
"""
Plan-and-Execute（Plan-Act）模式：
1. Planner：根据任务分解成步骤列表
2. Executor：逐步执行每个子步骤
3. Re-planner：根据执行结果调整计划

适合：长任务、多步骤、需要动态调整的场景
"""

from pydantic import BaseModel
import instructor
from openai import OpenAI

client = instructor.from_openai(
    OpenAI(base_url="http://localhost:8000/v1", api_key="token")
)

class TaskPlan(BaseModel):
    """任务计划"""
    class Step(BaseModel):
        step_id: int
        description: str
        tool_needed: str | None  # 需要用到的工具
        expected_output: str     # 期望输出
        depends_on: list[int]    # 依赖的前置步骤 ID
    
    goal: str
    steps: list[Step]
    estimated_complexity: str  # "simple" | "medium" | "complex"

class StepResult(BaseModel):
    """步骤执行结果"""
    step_id: int
    success: bool
    output: str
    next_action: str  # "continue" | "replan" | "abort"

class PlanAndExecuteAgent:
    """Plan-and-Execute Agent"""
    
    def __init__(self, tools: dict):
        self.tools = tools  # {"tool_name": callable}
        self.memory = AgentMemory()
    
    def plan(self, task: str) -> TaskPlan:
        """生成任务计划"""
        available_tools = list(self.tools.keys())
        
        plan = client.chat.completions.create(
            model="qwen2.5-7b",
            response_model=TaskPlan,
            messages=[{
                "role": "user",
                "content": f"""分解以下任务为可执行的步骤：

任务：{task}

可用工具：{available_tools}

要求：
- 步骤要具体、可执行
- 如需工具，指定工具名
- 明确步骤依赖关系
- 每步期望输出要清晰"""
            }]
        )
        return plan
    
    def execute_step(self, step, context: dict) -> StepResult:
        """执行单个步骤"""
        print(f"  → 执行步骤 {step.step_id}: {step.description}")
        
        if step.tool_needed and step.tool_needed in self.tools:
            # 使用工具
            tool_fn = self.tools[step.tool_needed]
            try:
                result = tool_fn(step.description, context)
                self.memory.save_fact(f"step_{step.step_id}_result", result)
                return StepResult(
                    step_id=step.step_id,
                    success=True,
                    output=str(result),
                    next_action="continue"
                )
            except Exception as e:
                return StepResult(
                    step_id=step.step_id,
                    success=False,
                    output=f"执行失败: {e}",
                    next_action="replan"
                )
        else:
            # LLM 直接处理
            response = client.chat.completions.create(
                model="qwen2.5-7b",
                messages=[{
                    "role": "user",
                    "content": f"任务上下文：{json.dumps(context, ensure_ascii=False)}\n\n请完成：{step.description}\n期望输出：{step.expected_output}"
                }]
            ).choices[0].message.content
            
            self.memory.save_fact(f"step_{step.step_id}_result", response)
            return StepResult(
                step_id=step.step_id,
                success=True,
                output=response,
                next_action="continue"
            )
    
    def run(self, task: str) -> str:
        """执行完整任务"""
        print(f"\n🎯 任务：{task}")
        
        # 1. 规划
        print("\n📋 制定计划...")
        plan = self.plan(task)
        print(f"共 {len(plan.steps)} 个步骤，复杂度：{plan.estimated_complexity}")
        
        # 2. 执行
        results = {}
        for step in plan.steps:
            # 检查依赖
            deps_met = all(
                results.get(dep, {}).get("success", False)
                for dep in step.depends_on
            )
            if not deps_met:
                print(f"  ⚠️ 步骤 {step.step_id} 依赖未满足，跳过")
                continue
            
            context = {k: v["output"] for k, v in results.items()}
            result = self.execute_step(step, context)
            results[step.step_id] = {"success": result.success, "output": result.output}
            
            if result.next_action == "abort":
                print("  ❌ 任务中止")
                break
        
        # 3. 汇总结果
        all_outputs = "\n".join(
            f"步骤{sid}: {r['output'][:200]}"
            for sid, r in results.items()
        )
        
        final = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=[{
                "role": "user",
                "content": f"根据以下执行结果，给出任务'{task}'的最终总结：\n\n{all_outputs}"
            }]
        ).choices[0].message.content
        
        return final


# Multi-Agent 系统简介
"""
Multi-Agent 架构（LangGraph/AutoGen 风格）：

┌─────────────────────────────────────────┐
│           Orchestrator                  │
│      负责任务分配和结果汇总              │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴───────┐
       ↓               ↓
┌──────────┐    ┌──────────────┐
│ Worker A │    │   Worker B   │
│ 代码执行 │    │  研究/搜索   │
└──────────┘    └──────────────┘

优点：
- 并行执行，提升效率
- 专业化分工
- 单个 Agent 上下文更干净

适用场景：
- 复杂报告生成（研究 + 写作 + 校对）
- 软件开发（产品 + 设计 + 编码 + 测试）
"""
```

---

## 四、面试题精讲

**Q: ReAct 和 Plan-and-Execute 各自适用什么场景？**

A:
- **ReAct**：任务路径不确定，需要根据每步结果灵活决策。适合：信息检索、问答、探索性任务。缺点：长任务会积累大量 context，效率低。
- **Plan-and-Execute**：任务结构相对清晰，可以预先分解步骤。适合：报告生成、数据分析流程、工作流自动化。优点：步骤可并行，context 更干净。

**Q: Agent 的记忆怎么设计？**

A: 三层记忆架构：
1. **短期记忆（Working Memory）**：当前对话 + 任务中间状态，存在内存/context window 中
2. **情节记忆（Episodic）**：历史对话摘要，可以压缩后存在 DB
3. **语义记忆（Semantic）**：长期知识和事实，存向量数据库（RAG 检索）

---

## 小结

```
Agent 架构要素：
  LLM：核心大脑（推理/规划/生成）
  Memory：短期（对话）+ 长期（向量库）
  Tools：搜索/代码执行/API/文件
  Planner：ReAct/Plan-Execute/自定义

选型建议：
  简单工具调用 → Function Calling（OpenAI API）
  推理链透明 → ReAct
  多步复杂任务 → Plan-and-Execute
  多智能体协作 → LangGraph/AutoGen
```
