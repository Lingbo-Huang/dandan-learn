---
layout: default
title: "W4D4 · 自我反思与批评机制"
render_with_liquid: false
---

# 自我反思：让 Agent 成为自己的批评者

> **Week 4 · Day 4** | 难度：⭐⭐⭐⭐

---

## 什么是 Agent 的自我反思？

人类专家的一大优势是**元认知**——能思考自己的思考过程，发现自己推理中的漏洞。自我反思机制让 Agent 也具备这种能力：

```
┌──────────────────────────────────────────────────────────┐
│                    自我反思循环                            │
│                                                          │
│   初始回答 ──→ 批评者分析 ──→ 发现问题 ──→ 修正回答       │
│       ↑                                      │           │
│       └──────────────────────────────────────┘           │
│                  （直到质量满足要求）                       │
└──────────────────────────────────────────────────────────┘
```

## 核心实现：Critic-Actor 双模型架构

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

class CritiqueResult(BaseModel):
    issues: List[str] = Field(description="发现的问题列表")
    suggestions: List[str] = Field(description="改进建议")
    quality_score: float = Field(description="质量评分 0-10", ge=0, le=10)
    needs_revision: bool = Field(description="是否需要修改")

class SelfReflectionAgent:
    def __init__(self, max_revisions: int = 3, quality_threshold: float = 8.0):
        self.actor = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.critic = ChatOpenAI(model="gpt-4o", temperature=0)
        self.max_revisions = max_revisions
        self.quality_threshold = quality_threshold
        
        self.critic_llm = self.critic.with_structured_output(CritiqueResult)
    
    def generate_initial_answer(self, task: str) -> str:
        """Actor：生成初始回答"""
        prompt = f"""请完成以下任务：

{task}

请给出你最好的回答："""
        return self.actor.invoke(prompt).content
    
    def critique(self, task: str, answer: str, revision_history: List[str]) -> CritiqueResult:
        """Critic：批评现有回答"""
        history_context = ""
        if revision_history:
            history_context = f"""
之前的版本：
{chr(10).join([f'版本{i+1}：{v[:200]}...' for i, v in enumerate(revision_history)])}

"""
        
        prompt = f"""你是一个严格的质量审核者。请批评以下回答：

任务要求：{task}

{history_context}当前回答：
{answer}

请从以下维度评估：
1. 准确性：信息是否正确？
2. 完整性：是否回答了所有要点？
3. 逻辑性：推理是否连贯？
4. 实用性：对用户是否有实际价值？
5. 清晰度：表达是否清晰易懂？"""
        
        return self.critic_llm.invoke(prompt)
    
    def revise(self, task: str, current_answer: str, critique: CritiqueResult) -> str:
        """Actor：根据批评修改回答"""
        issues_text = "\n".join([f"- {issue}" for issue in critique.issues])
        suggestions_text = "\n".join([f"- {s}" for s in critique.suggestions])
        
        prompt = f"""请改进以下回答：

原始任务：{task}

当前回答：
{current_answer}

批评者发现的问题：
{issues_text}

改进建议：
{suggestions_text}

请提供改进后的完整回答（不要解释你做了什么修改，直接给出改进后的答案）："""
        
        return self.actor.invoke(prompt).content
    
    def run(self, task: str) -> dict:
        """执行自我反思循环"""
        revision_history = []
        
        # 生成初始回答
        current_answer = self.generate_initial_answer(task)
        print(f"初始回答生成（{len(current_answer)} 字）")
        
        for revision_num in range(self.max_revisions):
            # 批评当前回答
            critique = self.critique(task, current_answer, revision_history)
            print(f"\n第 {revision_num + 1} 轮批评：")
            print(f"  质量评分：{critique.quality_score}/10")
            print(f"  发现问题：{len(critique.issues)} 个")
            for issue in critique.issues:
                print(f"    - {issue}")
            
            # 质量满足要求，停止
            if not critique.needs_revision or critique.quality_score >= self.quality_threshold:
                print(f"\n✅ 质量达标（{critique.quality_score}/10），停止修改")
                break
            
            # 记录当前版本，进行修改
            revision_history.append(current_answer)
            current_answer = self.revise(task, current_answer, critique)
            print(f"  修改完成（新版本 {len(current_answer)} 字）")
        
        return {
            "final_answer": current_answer,
            "revisions_made": len(revision_history),
            "revision_history": revision_history
        }

# 使用示例
agent = SelfReflectionAgent(max_revisions=3, quality_threshold=8.5)
result = agent.run(
    "请解释什么是向量数据库，它与传统数据库有什么区别，"
    "并给出一个 Python 示例说明如何使用 ChromaDB 进行相似度搜索。"
)

print(f"\n最终回答（经过 {result['revisions_made']} 次修改）：")
print(result['final_answer'])
```

## Constitutional AI 风格的约束批评

Google/Anthropic 的 Constitutional AI 思路：预定义一套"宪法"约束，让 Critic 对照宪法评估：

```python
CONSTITUTION = """
# Agent 回答宪法

## 准确性原则
- 不得提供未经验证的事实
- 数字和统计数据必须有据可查
- 对不确定信息必须明确标注

## 完整性原则  
- 必须回答用户的所有子问题
- 代码示例必须完整可运行
- 不得遗漏重要的注意事项

## 安全性原则
- 不提供可能被滥用的危险信息
- 代码示例不得包含安全漏洞
- 敏感话题需添加适当提示

## 实用性原则
- 回答必须对用户实际可用
- 避免过度学术化的表达
- 代码必须经过实际测试
"""

class ConstitutionalCritic:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    def evaluate(self, task: str, answer: str) -> dict:
        prompt = f"""请对照以下宪法约束，评估这个回答：

{CONSTITUTION}

任务：{task}

回答：{answer}

对照每条原则检查，给出：
1. 违反的原则（如有）
2. 需要修改的具体内容
3. 修改建议

JSON 格式返回：{{
    "violations": [{{"principle": "...", "issue": "...", "fix": "..."}}],
    "overall_score": 0-10,
    "approved": true/false
}}"""
        
        response = self.llm.invoke(prompt)
        import json
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            return json.loads(content.strip())
        except:
            return {"violations": [], "overall_score": 8, "approved": True}
```

## 多维度批评框架

对于不同类型的任务，用不同的批评标准：

```python
CRITIQUE_DIMENSIONS = {
    "code": {
        "correctness": "代码逻辑是否正确？有无 bug？",
        "efficiency": "时间/空间复杂度是否合理？",
        "readability": "代码是否易读？有无注释？",
        "security": "是否有安全漏洞（SQL注入/XSS等）？",
        "completeness": "是否处理了边界情况和异常？"
    },
    "analysis": {
        "accuracy": "分析结论是否有数据支撑？",
        "depth": "是否深入分析了根本原因？",
        "balance": "是否考虑了不同视角？",
        "actionability": "结论是否有实际指导意义？"
    },
    "planning": {
        "feasibility": "计划是否可行？",
        "completeness": "是否覆盖了所有关键步骤？",
        "risk": "是否识别并应对了主要风险？",
        "resource": "资源估算是否合理？"
    }
}

def get_critique_prompt(task_type: str, answer: str) -> str:
    dimensions = CRITIQUE_DIMENSIONS.get(task_type, CRITIQUE_DIMENSIONS["analysis"])
    dim_text = "\n".join([f"- {k}：{v}" for k, v in dimensions.items()])
    
    return f"""请从以下维度批评这个{task_type}：

评估维度：
{dim_text}

内容：
{answer}

给出每个维度的评分（0-10）和具体改进建议。"""
```

## 实战技巧：什么时候用自我反思？

```
任务类型              是否适合自我反思   原因
────────────────────────────────────────────────
代码生成                   ✅ 强烈推荐    bug 容易被发现和修复
报告/分析                  ✅ 推荐       多轮修改显著提升质量
创意写作                   ⚠️  谨慎      过度修改可能失去特色
简单问答                   ❌ 不推荐     浪费 token，收益甚微
实时对话                   ❌ 不推荐     延迟太高
```

## 踩坑经验

### 坑1：Critic 和 Actor 是同一个模型——自说自话

**问题**：用同一个 LLM 做 Actor 和 Critic，它对自己的错误视而不见。  
**解法**：
1. 用更强的模型做 Critic（如 Actor 用 gpt-4o-mini，Critic 用 gpt-4o）
2. 给 Critic 一个"挑剔"的 system prompt，明确要求它找问题而不是称赞
3. 多 Critic 投票：用 3 个独立 Critic 取共识

### 坑2：修改陷入循环——越改越差

**问题**：第2轮修改解决了问题A，却引入了问题B，然后第3轮又改回来。  
**解法**：
1. 记录修改历史，让 Critic 看到历史版本，避免退步
2. 设置"不退步约束"：每次修改必须保留上一版的优点

### 坑3：Critique 过于苛刻——永不满意

**问题**：quality_threshold 设太高，Agent 永远在修改，超出 max_revisions 停止时质量其实已经很高了。  
**解法**：动态调整阈值，或用"边际效益"判断：

```python
def should_continue_revision(self, scores: List[float]) -> bool:
    """判断继续修改是否有价值"""
    if len(scores) < 2:
        return True
    improvement = scores[-1] - scores[-2]
    return improvement > 0.5  # 提升小于0.5分时停止
```

## 小结

自我反思是提升 Agent 输出质量的"最后一公里"。关键要点：
1. **分离 Actor 和 Critic**，用更强的模型做批评
2. **宪法约束**让批评有据可依
3. **记录历史**防止修改退步
4. **动态停止**避免过度修改

> **下一篇**：任务分解——把大任务拆小，让复杂规划变得可执行。

---

*W4D4 · 自我反思与批评机制 | Agent + Claw 系列*
