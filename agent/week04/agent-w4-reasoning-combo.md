---
layout: default
title: "W4D6 · 推理策略组合实战"
---

# 推理策略组合：打造你的 Agent 推理工具箱

> **Week 4 · Day 6** | 难度：⭐⭐⭐⭐

---

## 单一策略的局限

本周我们学了 CoT、ToT、Reflexion、自我反思、任务分解。但在实际项目中，没有一种策略是万能的。高手的做法是：**根据问题特征，动态选择和组合推理策略**。

## 推理策略路由器

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional

class ReasoningStrategy(str, Enum):
    DIRECT = "direct"           # 直接回答（简单问题）
    COT = "cot"                 # Chain of Thought
    SELF_CONSISTENCY = "sc"     # Self-Consistency CoT
    TOT = "tot"                 # Tree of Thoughts
    REFLEXION = "reflexion"     # 带反思的重试
    DECOMPOSE = "decompose"     # 任务分解
    HYBRID = "hybrid"           # 混合策略

class StrategySelection(BaseModel):
    strategy: ReasoningStrategy
    reasoning: str = Field(description="为什么选择这个策略")
    confidence: float = Field(description="策略适用置信度 0-1", ge=0, le=1)
    fallback: Optional[ReasoningStrategy] = Field(description="备选策略")

class ReasoningRouter:
    """智能推理策略路由器"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.router_llm = self.llm.with_structured_output(StrategySelection)
    
    def select_strategy(self, task: str, constraints: dict = None) -> StrategySelection:
        """根据任务特征选择推理策略"""
        constraints_text = ""
        if constraints:
            constraints_text = f"\n约束条件：{constraints}"
        
        prompt = f"""分析以下任务，选择最合适的推理策略：

任务：{task}
{constraints_text}

策略说明：
- direct：简单问题，可以直接回答
- cot：需要多步推理，答案较确定
- sc（self-consistency）：推理步骤多且结果需要高准确性
- tot（tree of thoughts）：需要探索多种可能性，有明确评估标准
- reflexion：任务可能失败需要重试，有明确成功标准
- decompose：任务过于复杂，需要拆分为子任务
- hybrid：需要组合多种策略

请选择最适合的策略并说明原因。"""
        
        return self.router_llm.invoke(prompt)
    
    def execute_with_strategy(self, task: str, strategy: ReasoningStrategy) -> str:
        """按选定策略执行任务"""
        executors = {
            ReasoningStrategy.DIRECT: self._direct,
            ReasoningStrategy.COT: self._cot,
            ReasoningStrategy.SELF_CONSISTENCY: self._self_consistency,
            ReasoningStrategy.TOT: self._tot,
            ReasoningStrategy.REFLEXION: self._reflexion,
            ReasoningStrategy.DECOMPOSE: self._decompose,
            ReasoningStrategy.HYBRID: self._hybrid,
        }
        
        executor = executors.get(strategy, self._cot)
        return executor(task)
    
    def _direct(self, task: str) -> str:
        return self.llm.invoke(task).content
    
    def _cot(self, task: str) -> str:
        prompt = f"{task}\n\n让我一步一步思考："
        return self.llm.invoke(prompt).content
    
    def _self_consistency(self, task: str) -> str:
        # 生成5条路径，投票
        import asyncio
        from collections import Counter
        
        async def generate():
            llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
            tasks = [
                llm.ainvoke(f"{task}\n\n请独立思考，在最后一行以'答案：'开头给出结论。")
                for _ in range(5)
            ]
            return await asyncio.gather(*tasks)
        
        responses = asyncio.run(generate())
        answers = []
        for r in responses:
            lines = r.content.split('\n')
            for line in reversed(lines):
                if '答案：' in line:
                    answers.append(line.split('答案：')[1].strip())
                    break
        
        if answers:
            counter = Counter(answers)
            return counter.most_common(1)[0][0]
        return responses[0].content
    
    def _tot(self, task: str) -> str:
        # 简化版 ToT
        prompt = f"""用树形思维解决：{task}

步骤：
1. 列出3种不同的解决思路
2. 评估每种思路的可行性
3. 选择最佳思路并深入展开
4. 给出最终答案"""
        return self.llm.invoke(prompt).content
    
    def _reflexion(self, task: str) -> str:
        best_answer = ""
        reflections = []
        
        for attempt in range(3):
            reflection_context = ""
            if reflections:
                reflection_context = f"之前的尝试经验：{'; '.join(reflections)}\n"
            
            prompt = f"{reflection_context}任务：{task}\n请给出答案："
            answer = self.llm.invoke(prompt).content
            
            # 简单评估：让 LLM 评分
            eval_prompt = f"任务：{task}\n回答：{answer}\n这个回答是否完整准确？(是/否)"
            eval_result = self.llm.invoke(eval_prompt).content
            
            if "是" in eval_result:
                return answer
            
            reflections.append(f"第{attempt+1}次回答不够完整")
            best_answer = answer
        
        return best_answer
    
    def _decompose(self, task: str) -> str:
        # 先分解再执行
        decompose_prompt = f"""将以下任务分解为3-5个子任务（每行一个）：
{task}"""
        subtasks_response = self.llm.invoke(decompose_prompt).content
        subtasks = [s.strip() for s in subtasks_response.split('\n') if s.strip()]
        
        results = []
        for subtask in subtasks[:5]:
            result = self.llm.invoke(f"完成以下子任务：{subtask}").content
            results.append(f"**{subtask}**\n{result}")
        
        # 整合
        synthesis_prompt = f"""整合以下子任务结果，回答原始问题：
原始问题：{task}

子任务结果：
{chr(10).join(results)}"""
        
        return self.llm.invoke(synthesis_prompt).content
    
    def _hybrid(self, task: str) -> str:
        # 先分解，每个子任务用 CoT，最后自我反思
        subtasks_result = self._decompose(task)
        
        # 自我反思阶段
        critique_prompt = f"""批评以下回答并改进：
任务：{task}
当前回答：{subtasks_result}
改进后的完整回答："""
        
        return self.llm.invoke(critique_prompt).content
    
    def smart_solve(self, task: str, budget_limit: int = 10) -> dict:
        """智能求解：自动选择策略"""
        # 1. 选择策略
        selection = self.select_strategy(task, {"max_api_calls": budget_limit})
        print(f"选择策略：{selection.strategy.value}")
        print(f"原因：{selection.reasoning}")
        
        # 2. 执行
        result = self.execute_with_strategy(task, selection.strategy)
        
        # 3. 如果置信度低，用备选策略验证
        if selection.confidence < 0.7 and selection.fallback:
            fallback_result = self.execute_with_strategy(task, selection.fallback)
            
            # 让 LLM 选择更好的答案
            compare_prompt = f"""比较以下两个答案，选择更好的：
问题：{task}

答案A：{result}

答案B：{fallback_result}

哪个更好？（直接输出更好的答案）"""
            result = self.llm.invoke(compare_prompt).content
        
        return {
            "strategy_used": selection.strategy.value,
            "confidence": selection.confidence,
            "answer": result
        }

# 测试路由器
router = ReasoningRouter()

test_cases = [
    "2 + 2 等于多少？",  # → direct
    "如果苹果价格上涨20%，月收入3000的家庭应该如何调整食品预算？",  # → cot
    "设计一个月处理100万用户的实时推荐系统架构",  # → decompose
    "预测明年 AI 行业的5大趋势",  # → tot
]

for task in test_cases:
    print(f"\n任务：{task}")
    result = router.smart_solve(task)
    print(f"策略：{result['strategy_used']}")
    print(f"答案：{result['answer'][:200]}...")
```

## 策略组合矩阵

```
                   任务复杂度
              低          中          高
          ┌──────────┬──────────┬──────────┐
低        │  direct  │   cot    │decompose │
准确度    ├──────────┼──────────┼──────────┤
要求      │   cot    │ reflexion│tot+reflex│
          ├──────────┼──────────┼──────────┤
高        │   sc     │ sc+reflex│  hybrid  │
          └──────────┴──────────┴──────────┘
```

## 实战：多策略竞赛模式

```python
class StrategyRaceAgent:
    """多策略并发竞赛，取最优结果"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    async def race(self, task: str, strategies: List[str]) -> dict:
        """多策略并发，第一个高质量结果胜出"""
        import asyncio
        
        router = ReasoningRouter()
        
        async def run_strategy(strategy_name: str):
            strategy = ReasoningStrategy(strategy_name)
            result = router.execute_with_strategy(task, strategy)
            return {"strategy": strategy_name, "result": result}
        
        # 并发执行所有策略
        tasks = [run_strategy(s) for s in strategies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤掉错误，评估质量
        valid_results = [r for r in results if isinstance(r, dict)]
        
        # 选择最佳结果
        best = await self._pick_best(task, valid_results)
        return best
    
    async def _pick_best(self, task: str, results: List[dict]) -> dict:
        """让 LLM 评选最佳结果"""
        if len(results) == 1:
            return results[0]
        
        options_text = "\n\n".join([
            f"方案{i+1}（{r['strategy']}策略）：\n{r['result'][:300]}"
            for i, r in enumerate(results)
        ])
        
        prompt = f"""以下是用不同策略回答同一问题的结果，请选择最好的：

问题：{task}

{options_text}

哪个方案最好？请直接给出该方案的完整内容（不要说"方案X"，直接输出内容）："""
        
        best_result = await self.llm.ainvoke(prompt)
        return {"strategy": "race_winner", "result": best_result.content}
```

## 踩坑经验

### 坑1：过度工程——简单任务用复杂策略

**问题**：为了"用上所有技术"，对简单问题也用 ToT+Reflexion，浪费时间和钱。  
**原则**：最简单的策略能解决问题就用最简单的。路由器的价值在于**不用时不用**。

### 坑2：策略切换时上下文丢失

**问题**：从 CoT 切换到 ToT 时，CoT 产生的中间结果没有传递给 ToT。  
**解法**：策略间共享"工作记忆"，把中间结果作为输入传递给下一个策略。

### 坑3：并发竞赛成本翻倍

**问题**：Race 模式并发跑3个策略，成本是单策略的3倍，但提升可能只有10%。  
**建议**：仅对高价值、高风险的任务用 Race 模式，日常任务用路由器单选。

## 小结

推理策略的智慧在于**适时适所**：
- 简单问题 → Direct/CoT
- 高精度需求 → Self-Consistency
- 多方案探索 → ToT
- 需要重试改进 → Reflexion
- 超复杂任务 → Decompose
- 最高要求 → Hybrid/Race

> **下一篇**：Capstone——用本周所有技术构建一个真实的复杂推理 Agent。

---

*W4D6 · 推理策略组合实战 | Agent + Claw 系列*
