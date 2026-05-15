---
layout: default
title: "W5D1 · 多 Agent 架构设计"
render_with_liquid: false
---

# 多 Agent 架构设计：从单兵作战到协同作战

> **Week 5 · Day 1** | 难度：⭐⭐⭐⭐

---

## 为什么需要多 Agent？

单个 Agent 的本质局限：
1. **上下文窗口有限**：复杂任务的信息量超过单个 context window
2. **能力专业化**：不同任务需要不同能力，没有全能 Agent
3. **并行效率**：串行处理太慢，需要并行
4. **错误隔离**：一个 Agent 出错不影响整体系统

## 三种核心架构

### 架构1：层级式（Hierarchical）

```
          ┌─────────────┐
          │  主 Agent   │  ← 负责规划和协调
          │  (Orchestr.)│
          └──────┬──────┘
         ┌───────┼───────┐
         ▼       ▼       ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │Worker 1│ │Worker 2│ │Worker 3│
    │(搜索)  │ │(分析)  │ │(写作)  │
    └────────┘ └────────┘ └────────┘
```

**适用场景**：有明确主从关系的任务，如内容生产流水线。

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import List, Dict

class HierarchicalMultiAgent:
    def __init__(self):
        self.orchestrator = ChatOpenAI(model="gpt-4o", temperature=0)
        self.workers = {
            "researcher": ChatOpenAI(model="gpt-4o-mini", temperature=0),
            "analyst": ChatOpenAI(model="gpt-4o-mini", temperature=0),
            "writer": ChatOpenAI(model="gpt-4o", temperature=0.3),
        }
    
    def orchestrate(self, task: str) -> str:
        """主 Agent 规划任务分配"""
        plan_prompt = f"""将以下任务分配给合适的 Worker Agent：

任务：{task}

可用 Worker：
- researcher：负责信息搜索和资料收集
- analyst：负责数据分析和洞察提取
- writer：负责内容撰写和润色

请制定执行计划（JSON格式）：
{{
    "steps": [
        {{"agent": "researcher", "instruction": "..."}},
        {{"agent": "analyst", "instruction": "...", "depends_on": 0}},
        {{"agent": "writer", "instruction": "...", "depends_on": [0,1]}}
    ]
}}"""
        
        import json
        response = self.orchestrator.invoke(plan_prompt)
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        return json.loads(content.strip())
    
    def execute_step(self, agent_name: str, instruction: str, 
                    context: str = "") -> str:
        """Worker 执行具体步骤"""
        worker = self.workers[agent_name]
        prompt = f"""你是 {agent_name} Agent。
{f'上下文：{context}' if context else ''}

任务：{instruction}

请完成任务并给出结果："""
        
        return worker.invoke(prompt).content
    
    def run(self, task: str) -> str:
        """执行完整的层级式多 Agent 流程"""
        plan = self.orchestrate(task)
        results = {}
        
        for i, step in enumerate(plan["steps"]):
            agent_name = step["agent"]
            instruction = step["instruction"]
            
            # 收集依赖结果作为上下文
            context = ""
            if "depends_on" in step:
                deps = step["depends_on"] if isinstance(step["depends_on"], list) else [step["depends_on"]]
                context_parts = [f"步骤{d}结果：{results[d]}" for d in deps if d in results]
                context = "\n".join(context_parts)
            
            print(f"执行步骤{i}：{agent_name}")
            results[i] = self.execute_step(agent_name, instruction, context)
        
        # 最后一步的结果作为最终输出
        return results[max(results.keys())]

# 测试
agent = HierarchicalMultiAgent()
result = agent.run("分析并撰写一篇关于2024年大模型行业竞争格局的分析报告")
print(result[:500])
```

### 架构2：平行式（Parallel/Flat）

```
  ┌────────┐  ┌────────┐  ┌────────┐
  │Agent A │  │Agent B │  │Agent C │
  │(视角1) │  │(视角2) │  │(视角3) │
  └────┬───┘  └────┬───┘  └────┬───┘
       │            │            │
       └─────────── ▼ ───────────┘
               ┌─────────┐
               │聚合 Agent│  ← 汇总多视角
               └─────────┘
```

**适用场景**：需要多角度分析同一问题（辩论、评审）。

```python
import asyncio
from typing import List

class ParallelMultiAgent:
    """多 Agent 同时分析，聚合结论"""
    
    def __init__(self, agent_roles: List[str]):
        self.agents = {
            role: ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
            for role in agent_roles
        }
        self.aggregator = ChatOpenAI(model="gpt-4o", temperature=0)
    
    async def get_perspective(self, role: str, task: str) -> Dict:
        """获取单个 Agent 的视角"""
        prompt = f"""你是一个{role}专家。请从你的专业角度分析：

{task}

给出你的专业见解（300字以内）："""
        
        response = await self.agents[role].ainvoke(prompt)
        return {"role": role, "perspective": response.content}
    
    async def analyze(self, task: str) -> str:
        """并行获取所有视角，然后聚合"""
        # 并行执行所有 Agent
        tasks = [self.get_perspective(role, task) 
                 for role in self.agents.keys()]
        perspectives = await asyncio.gather(*tasks)
        
        # 聚合多视角
        perspectives_text = "\n\n".join([
            f"**{p['role']}视角**：\n{p['perspective']}"
            for p in perspectives
        ])
        
        aggregate_prompt = f"""综合以下多个专家视角，给出平衡的综合分析：

问题：{task}

各专家视角：
{perspectives_text}

请给出综合分析和关键结论："""
        
        response = await self.aggregator.ainvoke(aggregate_prompt)
        return response.content

# 使用：技术决策评审
agent = ParallelMultiAgent(["技术架构师", "产品经理", "安全专家", "成本分析师"])
result = asyncio.run(agent.analyze("是否应该将系统迁移到微服务架构？"))
print(result)
```

### 架构3：流水线式（Pipeline）

```
输入 → [Agent A] → [Agent B] → [Agent C] → 输出
                  ↕                ↕
             质量检查          质量检查
```

**适用场景**：有固定处理流程的任务，如内容审核、翻译润色。

```python
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class PipelineStage:
    name: str
    agent: ChatOpenAI
    system_prompt: str
    validator: Callable[[str], bool] = None

class PipelineMultiAgent:
    """流水线多 Agent，支持质量门控"""
    
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages
        self.quality_checker = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def check_quality(self, input_text: str, output_text: str, stage_name: str) -> bool:
        """质量检查：输出是否合格"""
        prompt = f"""检查 {stage_name} 阶段的输出质量：

输入：{input_text[:200]}
输出：{output_text[:200]}

输出是否满足质量要求？（只回答 yes 或 no）"""
        
        response = self.quality_checker.invoke(prompt)
        return "yes" in response.content.lower()
    
    def process(self, initial_input: str) -> str:
        """按流水线顺序处理"""
        current_data = initial_input
        
        for stage in self.stages:
            print(f"处理阶段：{stage.name}")
            
            prompt = f"{stage.system_prompt}\n\n输入：{current_data}\n\n输出："
            output = stage.agent.invoke(prompt).content
            
            # 质量检查（如有自定义验证器）
            if stage.validator and not stage.validator(output):
                raise ValueError(f"阶段 {stage.name} 输出未通过验证")
            
            # 通用质量检查
            if not self.check_quality(current_data, output, stage.name):
                print(f"  ⚠️ {stage.name} 质量不达标，重试...")
                output = stage.agent.invoke(prompt + "\n\n请确保输出更完整、准确。").content
            
            current_data = output
            print(f"  ✅ {stage.name} 完成")
        
        return current_data

# 使用：内容生产流水线
stages = [
    PipelineStage(
        name="初稿生成",
        agent=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
        system_prompt="你是内容创作者，生成初稿"
    ),
    PipelineStage(
        name="事实核查",
        agent=ChatOpenAI(model="gpt-4o", temperature=0),
        system_prompt="你是事实核查员，验证并修正不准确的信息"
    ),
    PipelineStage(
        name="风格优化",
        agent=ChatOpenAI(model="gpt-4o-mini", temperature=0.3),
        system_prompt="你是编辑，优化文章的可读性和风格"
    ),
]

pipeline = PipelineMultiAgent(stages)
result = pipeline.process("写一篇关于量子计算应用前景的文章")
print(result[:500])
```

## 架构选型指南

```
需求特征                     → 推荐架构
────────────────────────────────────────────
有明确主从关系                → Hierarchical
需要多角度分析               → Parallel
固定处理流程                 → Pipeline
复杂依赖关系                 → Hierarchical + 依赖图
需要高吞吐                   → Parallel + 异步
需要质量保证                 → Pipeline + 质量门控
```

## 踩坑经验

### 坑1：Agent 间信息传递格式不统一
**解法**：定义统一的 Message 数据结构，强制类型化传递。

### 坑2：层级 Agent 的主 Agent 变成瓶颈
**解法**：主 Agent 只做规划，执行完全异步并行，结果统一收集。

### 坑3：流水线某阶段失败导致全程终止
**解法**：每个阶段添加重试逻辑，失败后可以降级到简化处理。

---

*W5D1 · 多 Agent 架构设计 | Agent + Claw 系列*
