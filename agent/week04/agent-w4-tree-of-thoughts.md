---
layout: default
title: "W4D2 · Tree of Thoughts"
---

# Tree of Thoughts：构建 Agent 的推理搜索树

> **Week 4 · Day 2** | 难度：⭐⭐⭐⭐

---

## 为什么需要"思维树"？

CoT 是一条直线——只走一条推理路径。但有些问题需要**探索多种可能性**，在不同分支上试错，找到最优路径。这就是 Tree of Thoughts（ToT）的动机。

想象下棋：你不会只看一步，而是在脑中模拟多种走法，评估哪条路最好。ToT 让 Agent 也这样思考。

```
              [问题]
                │
        ┌───────┴───────┐
      [思路A]         [思路B]
        │               │
   ┌────┴────┐      ┌───┴───┐
  [A1]     [A2]   [B1]   [B2]
   ✗         │     ✗       │
           [A2a]         [B2a] ← 最优解
```

## ToT 的核心组件

### 1. 思维生成器（Thought Generator）

生成多个可能的下一步推理：

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List

llm = ChatOpenAI(model="gpt-4o", temperature=0.8)

def generate_thoughts(problem: str, current_state: str, num_thoughts: int = 3) -> List[str]:
    """生成多个推理分支"""
    prompt = f"""你正在解决以下问题：
{problem}

当前推理状态：
{current_state if current_state else "尚未开始推理"}

请提出 {num_thoughts} 个不同的下一步推理方向（每个方向独立一行，用"方向N："开头）："""
    
    response = llm.invoke(prompt)
    thoughts = []
    for line in response.content.split('\n'):
        line = line.strip()
        if line and any(line.startswith(f'方向{i}：') for i in range(1, num_thoughts+1)):
            thought = line.split('：', 1)[1].strip()
            thoughts.append(thought)
    
    return thoughts[:num_thoughts]
```

### 2. 状态评估器（State Evaluator）

评估每个思维分支的质量：

```python
from pydantic import BaseModel, Field

class ThoughtEvaluation(BaseModel):
    score: float = Field(description="推理质量分数 0-10", ge=0, le=10)
    is_promising: bool = Field(description="是否值得继续探索")
    reasoning: str = Field(description="评分理由")
    is_solved: bool = Field(description="是否已经解决了问题")

evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(ThoughtEvaluation)

def evaluate_thought(problem: str, thought_path: List[str]) -> ThoughtEvaluation:
    """评估思维路径的质量"""
    path_text = "\n".join([f"步骤{i+1}：{t}" for i, t in enumerate(thought_path)])
    
    prompt = f"""评估以下推理路径对于解决问题的质量：

问题：{problem}

推理路径：
{path_text}

请评估这条推理路径：
- 是否方向正确？
- 是否有助于解决问题？
- 是否值得继续探索？
- 是否已经完整地解决了问题？"""
    
    return evaluator_llm.invoke(prompt)
```

### 3. 搜索算法（BFS / DFS）

```python
import heapq
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ThoughtNode:
    path: List[str]           # 从根到当前节点的推理路径
    score: float = 0.0        # 评估分数
    is_solved: bool = False
    
    def __lt__(self, other):  # 优先队列比较
        return self.score > other.score  # 分数高的优先

class TreeOfThoughts:
    def __init__(self, 
                 max_depth: int = 4,
                 branching_factor: int = 3,
                 beam_width: int = 5):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.beam_width = beam_width  # Beam Search 宽度
    
    def solve_with_bfs(self, problem: str) -> Optional[List[str]]:
        """使用 BFS + Beam Search 求解"""
        # 初始化：空路径作为根节点
        beam = [ThoughtNode(path=[])]
        
        for depth in range(self.max_depth):
            print(f"搜索深度 {depth + 1}...")
            candidates = []
            
            for node in beam:
                # 检查是否已解决
                if node.is_solved:
                    return node.path
                
                # 生成子节点
                current_state = "\n".join(node.path) if node.path else ""
                new_thoughts = generate_thoughts(
                    problem, current_state, self.branching_factor
                )
                
                for thought in new_thoughts:
                    new_path = node.path + [thought]
                    # 评估新节点
                    eval_result = evaluate_thought(problem, new_path)
                    
                    new_node = ThoughtNode(
                        path=new_path,
                        score=eval_result.score,
                        is_solved=eval_result.is_solved
                    )
                    
                    if eval_result.is_solved:
                        return new_path  # 找到解，直接返回
                    
                    if eval_result.is_promising:
                        candidates.append(new_node)
            
            if not candidates:
                break
            
            # Beam Search：只保留最好的 beam_width 个节点
            candidates.sort(key=lambda x: x.score, reverse=True)
            beam = candidates[:self.beam_width]
        
        # 返回最佳路径（即使没完全解决）
        if beam:
            best = max(beam, key=lambda x: x.score)
            return best.path
        return None
    
    def solve_with_dfs(self, problem: str) -> Optional[List[str]]:
        """使用 DFS + 剪枝求解"""
        best_solution = [None]
        best_score = [0.0]
        
        def dfs(path: List[str], depth: int):
            if depth >= self.max_depth:
                return
            
            current_state = "\n".join(path) if path else ""
            thoughts = generate_thoughts(problem, current_state, self.branching_factor)
            
            for thought in thoughts:
                new_path = path + [thought]
                eval_result = evaluate_thought(problem, new_path)
                
                if eval_result.is_solved:
                    if eval_result.score > best_score[0]:
                        best_score[0] = eval_result.score
                        best_solution[0] = new_path
                    return
                
                # 剪枝：分数太低的分支不继续
                if eval_result.score >= 5.0 and eval_result.is_promising:
                    dfs(new_path, depth + 1)
        
        dfs([], 0)
        return best_solution[0]
```

## 完整实战：用 ToT 解决复杂规划问题

```python
class ToTAgent:
    """基于 Tree of Thoughts 的规划 Agent"""
    
    def __init__(self):
        self.tot = TreeOfThoughts(max_depth=4, branching_factor=3, beam_width=3)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    def solve(self, problem: str) -> dict:
        """求解复杂问题"""
        print(f"开始 ToT 求解：{problem[:50]}...")
        
        # 1. 用 ToT 找到推理路径
        solution_path = self.tot.solve_with_bfs(problem)
        
        if not solution_path:
            return {"error": "未能找到有效的推理路径"}
        
        # 2. 基于推理路径生成最终答案
        path_text = "\n".join([f"- {step}" for step in solution_path])
        final_prompt = f"""基于以下推理路径，给出对问题的完整、清晰的最终答案：

问题：{problem}

推理过程：
{path_text}

请给出结构清晰的最终答案："""
        
        final_answer = self.llm.invoke(final_prompt)
        
        return {
            "problem": problem,
            "reasoning_path": solution_path,
            "final_answer": final_answer.content
        }

# 测试 ToT Agent
agent = ToTAgent()
result = agent.solve(
    "一个创业公司有100万启动资金，需要在6个月内实现盈亏平衡。"
    "固定成本每月15万，变动成本占收入30%，当前客户月均消费5000元。"
    "请制定最优的增长策略。"
)

print("推理路径：")
for i, step in enumerate(result["reasoning_path"], 1):
    print(f"  {i}. {step}")
print(f"\n最终答案：\n{result['final_answer']}")
```

## ToT vs CoT 性能对比

```
任务类型          CoT 成功率   ToT 成功率   ToT 成本倍数
──────────────────────────────────────────────────────
简单数学           95%          96%          3x  ← CoT 更划算
多步推理           72%          89%          5x
创意规划           45%          78%         10x
博弈策略           38%          72%          8x
```

**结论**：ToT 在需要探索多种可能性的任务上收益最大，成本也最高。

## 踩坑经验

### 坑1：分支爆炸——API 调用数量失控

**问题**：`branching_factor=5, max_depth=4` 时最坏情况需要 5^4 = 625 次 LLM 调用。  
**解法**：
1. 用 Beam Search 限制每层保留的节点数（beam_width=3~5）
2. 早期剪枝：分数低于阈值（如5.0）的节点立即丢弃
3. 设置总调用次数上限

### 坑2：评估器本身不准确

**问题**：用 LLM 评估推理质量，但 LLM 自己可能判断失误，选错分支。  
**解法**：
1. 对评估结果也做 few-shot，给出好/坏路径的示例
2. 对关键决策点用多个评估器投票

### 坑3：最终答案与推理路径脱节

**问题**：ToT 找到了推理路径，但最终 LLM 生成答案时又重新"自由发挥"，忽略路径。  
**解法**：在最终 prompt 中明确强调"请严格按照推理路径作答，不要引入新的假设"。

## 小结

ToT 的核心价值是**将搜索思维引入语言推理**。它特别适合：
- 需要在多个方案中选择的规划任务
- 有明确正确/错误反馈的问题（数学、逻辑、博弈）
- 复杂的创意生成任务

成本高是它最大的限制——实际使用时要谨慎权衡。

> **下一篇**：ReAct 进阶——加入"行动"维度，让推理不止于文字。

---

*W4D2 · Tree of Thoughts | Agent + Claw 系列*
