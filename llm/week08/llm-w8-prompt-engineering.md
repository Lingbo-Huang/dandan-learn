---
layout: default
title: "D1 · Prompt 工程"
render_with_liquid: false
---

# D1 · Prompt 工程

> **Prompt 工程不是玄学，是有原理的工程实践。** 理解 LLM 如何处理 prompt，才能写出高效的 prompt。

---

## 一、Prompt 的本质

LLM 是条件语言模型：$P(y|x)$，prompt 就是条件 $x$。

好的 prompt 的作用：
1. **激活相关的预训练知识**（Few-shot 示例）
2. **缩小输出空间**（明确格式要求）
3. **改变 token 的条件概率**（CoT 让模型"思考"）

---

## 二、核心技巧

### 2.1 Zero-shot vs Few-shot

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="token")

def zero_shot(question: str) -> str:
    """Zero-shot：直接问"""
    return client.chat.completions.create(
        model="qwen2.5-7b",
        messages=[{"role": "user", "content": question}]
    ).choices[0].message.content

def few_shot(question: str) -> str:
    """Few-shot：先给几个例子"""
    examples = """
将以下句子分类为"正面"、"负面"或"中性"：

句子：这个产品真的很好用！
分类：正面

句子：一般般吧，没什么特别的。
分类：中性

句子：太失望了，完全不值这个价钱。
分类：负面

句子：{question}
分类：""".format(question=question)
    
    return client.chat.completions.create(
        model="qwen2.5-7b",
        messages=[{"role": "user", "content": examples}]
    ).choices[0].message.content

# 对比效果
test_cases = ["这个算法效率有点低", "哇，这个功能太棒了！"]
for test in test_cases:
    print(f"Q: {test}")
    print(f"  Zero-shot: {zero_shot(f'将以下句子分类：{test}')}")
    print(f"  Few-shot: {few_shot(test)}")
```

### 2.2 Chain-of-Thought (CoT)

```python
def solve_with_cot(problem: str) -> str:
    """Chain-of-Thought：让模型"先思考再回答"""
    
    # 简单 CoT：加 "Let's think step by step"
    prompt_simple_cot = f"{problem}\n\n让我们一步步思考："
    
    # 更好的 CoT：提供推理示例
    prompt_example_cot = f"""请解决以下数学问题。

示例：
问题：一个商店有 45 个苹果，卖出了 18 个，又进了 25 个，现在有多少个苹果？
推理：
1. 初始数量：45 个
2. 卖出后：45 - 18 = 27 个
3. 进货后：27 + 25 = 52 个
答案：52 个

问题：{problem}
推理："""
    
    response = client.chat.completions.create(
        model="qwen2.5-7b",
        messages=[{"role": "user", "content": prompt_example_cot}],
        temperature=0.0,  # 数学推理用低温度
    )
    return response.choices[0].message.content

# 结构化 CoT（适合复杂任务）
def structured_cot(task: str) -> str:
    """让 LLM 按固定结构思考"""
    system = """你是一个分析专家。对于任何问题，请按以下结构回答：

**理解问题**：重新描述问题的核心
**分析关键点**：列出影响答案的因素
**推理过程**：step by step 推理
**最终答案**：简洁明确的答案"""
    
    return client.chat.completions.create(
        model="qwen2.5-7b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": task}
        ]
    ).choices[0].message.content
```

### 2.3 Self-Consistency

```python
from collections import Counter
import asyncio

async def self_consistency(
    problem: str,
    n_samples: int = 5,
    temperature: float = 0.7,
) -> str:
    """
    Self-Consistency：多次采样取多数票
    
    Wei et al. 2022 证明这能显著提升复杂推理任务
    """
    
    async def single_sample() -> str:
        response = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=[{
                "role": "user",
                "content": f"{problem}\n\n请一步步推理，最后给出答案（格式：最终答案：[答案]）"
            }],
            temperature=temperature,
        )
        text = response.choices[0].message.content
        
        # 提取最终答案
        if "最终答案：" in text:
            answer = text.split("最终答案：")[-1].strip().split("\n")[0]
        else:
            answer = text.strip()
        return answer
    
    # 并发采样
    tasks = [single_sample() for _ in range(n_samples)]
    answers = await asyncio.gather(*tasks)
    
    print(f"所有答案：{answers}")
    
    # 多数票
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common
```

---

## 三、高级 Prompt 技巧

### 3.1 System Prompt 设计

```python
"""
有效 System Prompt 的结构：

1. 角色定义（Role）
2. 能力范围（Scope）
3. 行为准则（Guidelines）
4. 输出格式（Format）
"""

system_prompt_template = """你是一个{role}。

**你的专长：**
{capabilities}

**行为准则：**
- 回答要{tone}
- 对不确定的信息要{uncertainty_handling}
- {additional_rules}

**输出格式：**
{output_format}"""

# 代码审查专家
code_review_system = system_prompt_template.format(
    role="资深 Python 代码审查专家",
    capabilities="Python 最佳实践、性能优化、安全漏洞识别、代码可读性改进",
    tone="专业且具体，给出可操作的建议",
    uncertainty_handling="明确说明，不要猜测",
    additional_rules="如有安全问题，必须标记为 [CRITICAL]",
    output_format="""
## 问题列表
- [严重性] 问题描述：具体代码行

## 改进建议
```python
# 改进后的代码
```

## 总体评分：X/10"""
)
```

### 3.2 Prompt 优化流程

```python
class PromptOptimizer:
    """系统化 Prompt 优化"""
    
    def __init__(self, llm_client):
        self.client = llm_client
        self.test_cases = []
        self.best_prompt = None
        self.best_score = -1
    
    def add_test_case(self, input: str, expected: str) -> None:
        """添加测试用例"""
        self.test_cases.append({'input': input, 'expected': expected})
    
    def evaluate_prompt(self, prompt: str) -> float:
        """评估 prompt 的效果（0-1 分）"""
        scores = []
        for case in self.test_cases:
            full_prompt = prompt.format(input=case['input'])
            output = self.client.generate(full_prompt)
            
            # 简单评分：是否包含期望答案
            score = 1.0 if case['expected'].lower() in output.lower() else 0.0
            scores.append(score)
        
        return sum(scores) / len(scores)
    
    def optimize(self, initial_prompt: str, n_iterations: int = 5) -> str:
        """迭代优化 prompt"""
        current_prompt = initial_prompt
        current_score = self.evaluate_prompt(current_prompt)
        
        for i in range(n_iterations):
            # 让 LLM 改进 prompt
            improve_prompt = f"""以下是一个 prompt 及其测试结果，请提出改进建议：

当前 Prompt：
{current_prompt}

当前得分：{current_score:.2%}

失败案例：
{self._get_failures(current_prompt)}

请生成一个改进后的 prompt（只输出 prompt 本身，不要解释）："""
            
            new_prompt = self.client.generate(improve_prompt)
            new_score = self.evaluate_prompt(new_prompt)
            
            print(f"迭代 {i+1}: {current_score:.2%} → {new_score:.2%}")
            
            if new_score > current_score:
                current_prompt = new_prompt
                current_score = new_score
        
        return current_prompt
    
    def _get_failures(self, prompt: str) -> str:
        failures = []
        for case in self.test_cases[:3]:
            output = self.client.generate(prompt.format(input=case['input']))
            if case['expected'].lower() not in output.lower():
                failures.append(
                    f"输入：{case['input']}\n期望：{case['expected']}\n实际：{output[:100]}"
                )
        return '\n---\n'.join(failures)
```

---

## 四、面试题精讲

**Q: 为什么 CoT 能提升推理能力？**

A: CoT 通过让模型"说出中间步骤"，实际上在 context window 中提供了中间计算结果。这有几个好处：
1. 中间结果作为新的 context，可以支撑后续的推理（类似草稿纸）
2. 模型生成推理链时，每个 token 预测都在缩小可能的答案空间
3. 训练时模型看到了"推理 → 结论"的模式，推理能力被激活

**Q: Zero-shot CoT 和 Few-shot CoT 哪个更好？**

A: 取决于任务复杂度和数据量：
- **Zero-shot CoT**（"Let's think step by step"）：简单，但效果不如有示例
- **Few-shot CoT**：提供推理步骤的示例，效果更好，但需要精心设计示例
- **自动 CoT**（Auto-CoT）：用 LLM 自动生成推理示例，适合大规模应用

---

## 小结

```
Prompt 工程核心技巧：

Zero-shot → Few-shot → CoT → Self-consistency
  简单快  → 需要示例 → 推理提升 → 最准确（慢）

实践建议：
1. 先 Zero-shot，不够再 Few-shot
2. 推理任务用 CoT
3. 高精度场景用 Self-consistency（多采样）
4. System Prompt 明确角色、范围、格式
5. 用测试集驱动优化，不要凭感觉
```
