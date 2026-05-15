---
layout: default
title: "W4D1 · Chain-of-Thought 进阶"
---

# Chain-of-Thought 进阶：让 Agent 像专家一样思考

> **Week 4 · Day 1** | 难度：⭐⭐⭐

---

## 为什么 CoT 是 Agent 推理的基石？

标准的 LLM 调用是一个"黑盒跳跃"：输入问题，直接输出答案。这在复杂问题上几乎必然出错。Chain-of-Thought（思维链）的核心思想是：**强迫模型把中间推理步骤写出来**，就像人类在草稿纸上演算一样。

```
没有 CoT：
Q: 小明有5个苹果，给了小红2个，然后又买了3个，最后有几个？
A: 6个  ← 直接输出，可能算错

有 CoT：
Q: 小明有5个苹果...（让我一步步想）
A: 先有5个 → 给了2个剩3个 → 又买3个共6个 → 答案：6个  ← 过程清晰，减少错误
```

## CoT 的三种实现方式

### 1. Zero-shot CoT（零样本思维链）

最简单，在 prompt 末尾加一句魔法咒语：

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 关键：在问题后加 "Let's think step by step"
zero_shot_cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个严谨的推理专家。"),
    ("human", "{question}\n\n让我们一步一步地思考这个问题。")
])

chain = zero_shot_cot_prompt | llm

result = chain.invoke({
    "question": "如果一家工厂每天生产150个零件，其中15%是次品，正品需要经过3道工序，每道工序有5%的损耗率，最终能有多少合格产品？"
})
print(result.content)
```

### 2. Few-shot CoT（少样本思维链）

提供几个"思考示例"，让模型学会推理模式：

```python
from langchain.prompts import FewShotChatMessagePromptTemplate

examples = [
    {
        "question": "Roger有5个网球，又买了2罐，每罐3个，现在有多少？",
        "answer": """让我一步步计算：
1. Roger 开始有 5 个网球
2. 买了 2 罐，每罐 3 个 = 2 × 3 = 6 个
3. 总共：5 + 6 = 11 个
答案：11 个"""
    },
    {
        "question": "食堂早上有23个苹果，用了20个做午饭，又买了6个，现在有多少？",
        "answer": """让我一步步计算：
1. 开始有 23 个苹果
2. 用了 20 个：23 - 20 = 3 个
3. 又买了 6 个：3 + 6 = 9 个
答案：9 个"""
    }
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
    ("ai", "{answer}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个数学推理专家，请展示完整的推理过程。"),
    few_shot_prompt,
    ("human", "{question}")
])

chain = final_prompt | llm
```

### 3. Self-Consistency CoT（自洽性思维链）

**最强大的变体**：对同一问题生成多条推理路径，投票选最一致的答案。

```python
import asyncio
from collections import Counter
from langchain_openai import ChatOpenAI

class SelfConsistencyCoT:
    def __init__(self, num_samples=5):
        # 提高 temperature 以获得多样性推理路径
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.num_samples = num_samples
    
    async def generate_reasoning_path(self, question: str) -> str:
        """生成一条推理路径"""
        prompt = f"""问题：{question}

请独立地、一步步地思考，给出你的推理过程和最终答案。
在最后一行，用"最终答案："开头单独给出答案。"""
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    def extract_answer(self, reasoning: str) -> str:
        """从推理文本中提取最终答案"""
        lines = reasoning.strip().split('\n')
        for line in reversed(lines):
            if '最终答案：' in line or '答案：' in line:
                return line.split('：', 1)[-1].strip()
        return lines[-1].strip()
    
    async def solve(self, question: str) -> dict:
        """使用自洽性方法求解"""
        # 并发生成多条推理路径
        tasks = [self.generate_reasoning_path(question) 
                 for _ in range(self.num_samples)]
        reasoning_paths = await asyncio.gather(*tasks)
        
        # 提取每条路径的答案
        answers = [self.extract_answer(r) for r in reasoning_paths]
        
        # 投票：选出最常见的答案
        answer_counts = Counter(answers)
        best_answer, count = answer_counts.most_common(1)[0]
        confidence = count / self.num_samples
        
        return {
            "answer": best_answer,
            "confidence": confidence,
            "all_answers": answers,
            "reasoning_paths": reasoning_paths
        }

# 使用示例
async def main():
    solver = SelfConsistencyCoT(num_samples=5)
    result = await solver.solve(
        "一个水箱满载时能装1200升水。现在已经装了40%，"
        "每分钟流入15升，流出8升，多少分钟后水箱满载？"
    )
    print(f"答案：{result['answer']}")
    print(f"置信度：{result['confidence']:.0%}")
    print(f"所有答案：{result['all_answers']}")

asyncio.run(main())
```

## 架构图：CoT 变体对比

```
┌─────────────────────────────────────────────────────────────┐
│                     CoT 技术谱系                              │
├────────────────┬────────────────┬────────────────────────────┤
│  Zero-shot CoT │  Few-shot CoT  │   Self-Consistency CoT     │
├────────────────┼────────────────┼────────────────────────────┤
│                │  示例1 ──→     │   路径1 ──→ 答案A           │
│  "一步步思考"   │  示例2 ──→     │   路径2 ──→ 答案B    投票   │
│      ↓         │  示例3 ──→     │   路径3 ──→ 答案A  ──────→  │
│  单条推理路径   │      ↓         │   路径4 ──→ 答案A   最终答案│
│                │  单条推理路径   │   路径5 ──→ 答案C           │
├────────────────┼────────────────┼────────────────────────────┤
│  简单/快速      │  准确度更高     │  最高准确度/成本最高        │
│  无需示例       │  需要设计示例   │  适合高风险决策             │
└────────────────┴────────────────┴────────────────────────────┘
```

## 进阶技巧：结构化 CoT

对于 Agent 任务，可以用结构化输出约束推理格式：

```python
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI

class ReasoningStep(BaseModel):
    step_number: int = Field(description="步骤序号")
    description: str = Field(description="这一步做了什么")
    result: str = Field(description="这一步的中间结果")

class StructuredCoTOutput(BaseModel):
    problem_analysis: str = Field(description="问题分析")
    reasoning_steps: List[ReasoningStep] = Field(description="推理步骤列表")
    final_answer: str = Field(description="最终答案")
    confidence: float = Field(description="置信度 0-1", ge=0, le=1)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(StructuredCoTOutput)

def solve_with_structured_cot(problem: str) -> StructuredCoTOutput:
    prompt = f"""请用结构化的方式解决以下问题，展示完整的推理过程：

问题：{problem}

请分析问题，列出每个推理步骤，给出最终答案和你的置信度。"""
    
    return structured_llm.invoke(prompt)

# 测试
result = solve_with_structured_cot(
    "公司有100名员工，其中60%是工程师，工程师中有25%是高级工程师，"
    "高级工程师平均薪资是普通工程师的1.5倍，普通工程师月薪15000元，"
    "请计算公司工程师团队每月总薪资。"
)

print(f"问题分析：{result.problem_analysis}")
for step in result.reasoning_steps:
    print(f"步骤{step.step_number}：{step.description} → {step.result}")
print(f"最终答案：{result.final_answer}")
print(f"置信度：{result.confidence:.0%}")
```

## 踩坑经验

### 坑1：Zero-shot CoT 中文效果不稳定

**问题**：中文场景下，"让我们一步步思考"有时不会触发真正的逐步推理。  
**解法**：明确要求格式，比如"请用编号列出每个推理步骤"。

### 坑2：Self-Consistency 答案格式不统一

**问题**：5条路径给出的答案格式各不相同（"11个"、"11"、"答案是11"），投票失效。  
**解法**：在 prompt 中明确要求答案格式，或用 LLM 做答案标准化。

```python
def normalize_answer(self, answers: List[str]) -> List[str]:
    """用 LLM 标准化答案格式"""
    normalization_prompt = f"""以下是对同一问题的多个答案，请将它们标准化为统一格式（仅保留核心数字或结论）：

答案列表：{answers}

请返回一个标准化后的列表，格式相同。"""
    # ... 调用 LLM 标准化
```

### 坑3：CoT 在简单问题上浪费 token

**解法**：先判断问题复杂度，简单问题直接回答。

```python
def should_use_cot(question: str) -> bool:
    """判断是否需要 CoT"""
    complexity_indicators = [
        "计算", "如果", "多少", "步骤", "过程", "分析",
        "比较", "为什么", "如何", "推导"
    ]
    return any(indicator in question for indicator in complexity_indicators)
```

## 小结

| 技术 | 适用场景 | 成本 | 准确率提升 |
|------|---------|------|-----------|
| Zero-shot CoT | 一般推理任务 | 低 | +15~20% |
| Few-shot CoT | 特定领域推理 | 中 | +25~35% |
| Self-Consistency | 高精度需求 | 高(5x) | +40~50% |
| 结构化 CoT | Agent 任务规划 | 中 | +30~40% |

> **下一篇**：Tree of Thoughts——当一条推理路径不够用时，我们来建推理树。

---

*W4D1 · CoT 进阶 | Agent + Claw 系列*
