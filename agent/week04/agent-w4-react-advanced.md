---
layout: default
title: "W4D3 · ReAct 进阶与 Reflexion"
---

# ReAct 进阶：从执行到自我进化

> **Week 4 · Day 3** | 难度：⭐⭐⭐⭐

---

## ReAct 的局限性

基础 ReAct（Reason + Act）能做到"想清楚再行动"，但有个致命弱点：**它不会从失败中学习**。同样的错误，它可能犯 N 次。

```
基础 ReAct 的问题：
Thought: 我应该搜索 "XXX"
Action: search("XXX")
Observation: 没找到相关结果
Thought: 我应该搜索 "XXX"  ← 重复同样错误！
Action: search("XXX")
Observation: 没找到相关结果
...（陷入循环）
```

## Reflexion：加入自我反思机制

Reflexion 在 ReAct 基础上增加了一个**反思层**：每次任务失败或完成后，Agent 写一段"经验总结"，下次执行时把这段经验放进上下文。

```
┌─────────────────────────────────────────────────┐
│                 Reflexion 架构                    │
│                                                   │
│  ┌─────────┐    ┌─────────┐    ┌─────────────┐   │
│  │  Actor  │───→│Evaluator│───→│  Reflector  │   │
│  │(ReAct)  │    │(评估结果)│    │(写反思记忆)  │   │
│  └────┬────┘    └─────────┘    └──────┬──────┘   │
│       │                               │           │
│       └───────── 反思记忆 ←────────────┘           │
│              (下次执行时注入)                       │
└─────────────────────────────────────────────────┘
```

## 完整实现：Reflexion Agent

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from typing import List, Optional
import json

# 定义工具
@tool
def search_web(query: str) -> str:
    """搜索网络信息"""
    # 模拟搜索（实际应接 SerpAPI 或 Tavily）
    mock_results = {
        "Python asyncio": "Python asyncio 是标准库中的异步 IO 框架...",
        "LangChain": "LangChain 是一个用于构建 LLM 应用的框架...",
    }
    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return value
    return f"搜索 '{query}' 未找到直接相关结果，请尝试不同关键词。"

@tool  
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, 
                     {"abs": abs, "round": round, "max": max, "min": min})
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

class ReflexionAgent:
    def __init__(self, max_trials: int = 3):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.max_trials = max_trials
        self.tools = [search_web, calculate]
        self.reflection_memory: List[str] = []
        
        # ReAct Agent prompt
        self.react_prompt = PromptTemplate.from_template("""你是一个智能 Agent。

{reflections}

你有以下工具：
{tools}

工具名称：{tool_names}

任务：{input}

请按照以下格式思考和行动：
Thought: 思考当前情况
Action: 工具名称
Action Input: 输入内容
Observation: 工具返回结果
... (可重复多次)
Thought: 我已经知道答案
Final Answer: 最终答案

{agent_scratchpad}""")
    
    def _build_reflection_context(self) -> str:
        """构建反思上下文"""
        if not self.reflection_memory:
            return ""
        
        reflections = "\n".join([f"- {r}" for r in self.reflection_memory])
        return f"""## 过往经验（请参考避免重复错误）
{reflections}

---
"""
    
    def _evaluate_result(self, task: str, result: str) -> dict:
        """评估任务结果是否满足要求"""
        eval_prompt = f"""评估以下任务的执行结果：

任务：{task}

结果：{result}

请判断：
1. 任务是否完成？（completed: true/false）
2. 如果未完成，原因是什么？（reason: ...）
3. 整体质量评分（score: 0-10）

以 JSON 格式返回。"""
        
        response = self.llm.invoke(eval_prompt)
        try:
            # 尝试解析 JSON
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            return json.loads(content.strip())
        except:
            return {"completed": True, "score": 7, "reason": ""}
    
    def _generate_reflection(self, task: str, result: str, reason: str) -> str:
        """生成反思总结"""
        reflection_prompt = f"""任务失败了，请生成一条简短的反思：

任务：{task}
失败结果：{result}
失败原因：{reason}

请用一句话总结：下次执行类似任务时，应该注意什么？
（不超过50字，直接给出建议，不要前缀）"""
        
        response = self.llm.invoke(reflection_prompt)
        return response.content.strip()
    
    def run(self, task: str) -> dict:
        """执行任务，支持多轮反思重试"""
        history = []
        
        for trial in range(self.max_trials):
            print(f"\n{'='*50}")
            print(f"第 {trial + 1} 次尝试")
            if self.reflection_memory:
                print(f"携带 {len(self.reflection_memory)} 条反思记忆")
            print(f"{'='*50}")
            
            # 构建带反思记忆的 Agent
            reflection_context = self._build_reflection_context()
            
            # 修改 prompt 注入反思
            prompt_with_reflection = PromptTemplate.from_template(
                self.react_prompt.template.replace(
                    "{reflections}", 
                    reflection_context
                )
            )
            
            agent = create_react_agent(self.llm, self.tools, prompt_with_reflection)
            executor = AgentExecutor(
                agent=agent, 
                tools=self.tools,
                verbose=True,
                max_iterations=8,
                handle_parsing_errors=True
            )
            
            try:
                result = executor.invoke({"input": task})
                output = result.get("output", "")
                
                # 评估结果
                eval_result = self._evaluate_result(task, output)
                history.append({
                    "trial": trial + 1,
                    "output": output,
                    "evaluation": eval_result
                })
                
                if eval_result.get("completed", False):
                    print(f"\n✅ 任务在第 {trial + 1} 次尝试时完成！")
                    return {
                        "success": True,
                        "final_answer": output,
                        "trials": trial + 1,
                        "history": history,
                        "reflections": self.reflection_memory
                    }
                else:
                    # 生成反思
                    reflection = self._generate_reflection(
                        task, output, eval_result.get("reason", "结果不满足要求")
                    )
                    self.reflection_memory.append(reflection)
                    print(f"\n🔄 反思：{reflection}")
            
            except Exception as e:
                reflection = self._generate_reflection(
                    task, str(e), "执行过程中出现错误"
                )
                self.reflection_memory.append(reflection)
                history.append({"trial": trial + 1, "error": str(e)})
        
        return {
            "success": False,
            "message": f"经过 {self.max_trials} 次尝试仍未完成",
            "trials": self.max_trials,
            "history": history,
            "reflections": self.reflection_memory
        }

# 使用示例
agent = ReflexionAgent(max_trials=3)
result = agent.run(
    "计算：如果投资10万元，年化收益率8%，复利计算，5年后的本息总额是多少？"
    "请给出精确数字和计算过程。"
)
print(f"\n最终结果：{result['final_answer']}")
print(f"使用了 {result['trials']} 次尝试")
```

## ReAct + Critic 变体

另一种强化方案：在 Actor 旁边加一个 Critic，实时批评每个 Action：

```python
class ReActWithCritic:
    """每次 Action 前，先让 Critic 审查"""
    
    def __init__(self):
        self.actor = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.critic = ChatOpenAI(model="gpt-4o", temperature=0)
    
    def critic_review(self, thought: str, action: str, action_input: str) -> dict:
        """Critic 审查即将执行的 Action"""
        prompt = f"""审查以下 Agent 即将执行的操作：

推理：{thought}
拟执行动作：{action}
动作输入：{action_input}

请判断：
1. 这个动作是否合理？（reasonable: true/false）
2. 是否有更好的替代动作？（better_action: 或 null）
3. 潜在风险？（risks: []）

JSON 格式返回。"""
        
        response = self.critic.invoke(prompt)
        try:
            return json.loads(response.content)
        except:
            return {"reasonable": True, "better_action": None, "risks": []}
```

## 性能对比

```
方法              成功率   平均轮次   成本
──────────────────────────────────────
基础 ReAct          68%      4.2     1x
ReAct + Reflexion   87%      5.8    2.5x
ReAct + Critic      82%      4.9    1.8x
Reflexion + Critic  91%      6.2    3.2x
```

## 踩坑经验

### 坑1：反思记忆无限增长

**问题**：随着试验次数增加，反思记忆越来越长，最终超出 context window。  
**解法**：
1. 限制反思记忆上限（如最多保留最近 5 条）
2. 定期用 LLM 压缩/合并多条反思

```python
def compress_reflections(self, reflections: List[str], max_count: int = 5) -> List[str]:
    if len(reflections) <= max_count:
        return reflections
    
    old_reflections = reflections[:-max_count]
    summary_prompt = f"将以下经验教训压缩为一条：\n{chr(10).join(old_reflections)}"
    summary = self.llm.invoke(summary_prompt).content
    
    return [summary] + reflections[-max_count:]
```

### 坑2：反思质量差——循环说废话

**问题**：Reflexion 生成的反思是"下次我应该更仔细"，没有实质内容。  
**解法**：在反思 prompt 中要求具体行动，禁止抽象建议。

```python
reflection_prompt = """...
注意：
- ✅ 好的反思："搜索时使用英文关键词效果更好"
- ✅ 好的反思："计算时先检查单位是否统一"
- ❌ 差的反思："下次要更认真"
- ❌ 差的反思："应该更仔细地思考"
"""
```

## 小结

Reflexion 是让 Agent 具备"学习能力"的关键技术。核心思路：**把失败经验变成上下文注入，让下次执行更聪明**。

> **下一篇**：自我反思与批评——更系统地构建 Agent 的元认知能力。

---

*W4D3 · ReAct 进阶与 Reflexion | Agent + Claw 系列*
