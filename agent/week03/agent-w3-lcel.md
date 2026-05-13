---
layout: default
title: "D2 · LCEL：LangChain Expression Language"
---

# D2 · LCEL：LangChain 表达式语言

> **Agent Week 3**  
> LCEL 是 LangChain 的核心设计，让链式调用变得优雅、并行、可流式输出。

---

## 一、LCEL 的本质

LCEL 用 `|` 运算符将组件串联，就像 Unix 管道：

```python
chain = prompt | llm | parser
result = chain.invoke({"question": "..."})
```

**底层机制**：每个组件实现 `Runnable` 接口，具备：
- `invoke(input)` → 同步调用
- `ainvoke(input)` → 异步调用
- `stream(input)` → 流式输出
- `batch(inputs)` → 批量处理

---

## 二、流式输出

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

chain = (
    ChatPromptTemplate.from_template("详细解释：{topic}") 
    | llm 
    | StrOutputParser()
)

# 流式输出（实时看到内容）
for chunk in chain.stream({"topic": "注意力机制"}):
    print(chunk, end="", flush=True)
print()

# 异步流式
import asyncio

async def stream_async():
    async for chunk in chain.astream({"topic": "KV Cache"}):
        print(chunk, end="", flush=True)

asyncio.run(stream_async())
```

---

## 三、并行执行（RunnableParallel）

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

llm = ChatOpenAI(model="gpt-4o-mini")

# 并行生成多个角度的分析
analysis_chain = RunnableParallel(
    pros=ChatPromptTemplate.from_template("列出 {topic} 的3个优点") | llm | StrOutputParser(),
    cons=ChatPromptTemplate.from_template("列出 {topic} 的3个缺点") | llm | StrOutputParser(),
    summary=ChatPromptTemplate.from_template("用一句话总结 {topic}") | llm | StrOutputParser(),
)

# 三个请求并行发送，比串行快 3 倍
result = analysis_chain.invoke({"topic": "大模型 Agent"})
print("优点:", result['pros'])
print("缺点:", result['cons'])
print("总结:", result['summary'])
```

---

## 四、条件路由（RunnableBranch）

```python
from langchain_core.runnables import RunnableBranch

# 根据问题类型选择不同的处理链
code_chain = (
    ChatPromptTemplate.from_template("你是编程专家。回答这个代码问题：{question}")
    | llm | StrOutputParser()
)

math_chain = (
    ChatPromptTemplate.from_template("你是数学专家。解决这个数学问题：{question}")
    | llm | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_template("回答这个问题：{question}")
    | llm | StrOutputParser()
)

# 路由函数
def classify_question(x):
    keywords = x["question"].lower()
    if any(k in keywords for k in ["代码", "python", "函数", "bug"]):
        return "code"
    elif any(k in keywords for k in ["计算", "数学", "积分", "方程"]):
        return "math"
    else:
        return "general"

branch = RunnableBranch(
    (lambda x: classify_question(x) == "code", code_chain),
    (lambda x: classify_question(x) == "math", math_chain),
    general_chain  # 默认
)

result = branch.invoke({"question": "如何用 Python 实现快速排序？"})
print(result)
```

---

## 五、自定义 Runnable

```python
from langchain_core.runnables import RunnableLambda

# 任何函数都可以变成 Runnable
def add_context(x):
    """在问题前加上上下文"""
    return {
        "question": x["question"],
        "context": "这是一个 AI 学习课程，面向初学者。"
    }

chain = (
    RunnableLambda(add_context)
    | ChatPromptTemplate.from_template("上下文：{context}\n\n问题：{question}")
    | llm
    | StrOutputParser()
)

result = chain.invoke({"question": "什么是梯度下降？"})
```

---

## 六、RunnablePassthrough：透传输入

```python
from langchain_core.runnables import RunnablePassthrough

# 场景：需要在链中保留原始输入
chain = (
    RunnableParallel({
        "question": RunnablePassthrough(),  # 原样传递
        "answer": ChatPromptTemplate.from_template("{question}") | llm | StrOutputParser()
    })
)

result = chain.invoke("什么是 LangChain？")
print(f"问题：{result['question']}")
print(f"回答：{result['answer']}")
```

---

## 七、LCEL 的工程优势

| 特性 | 说明 |
|------|------|
| **流式支持** | 所有组件自动支持流式，用户体验更好 |
| **并行执行** | RunnableParallel 自动并发，提升吞吐量 |
| **异步原生** | 统一的 async/await 接口 |
| **类型安全** | 输入输出类型清晰，IDE 提示友好 |
| **可追踪** | 与 LangSmith 无缝集成 |
| **可重试** | 内置重试机制 |

---

## 今天的关键认识

1. **LCEL 的 `|`**：组合 Runnable 组件的语法糖，底层是函数组合
2. **stream**：实时输出，用户体验更好
3. **RunnableParallel**：多条链并行，性能提升
4. **RunnableBranch**：条件路由，实现智能决策

---

## 明天预告

D3：**记忆系统**——让 Agent 记住对话历史，实现真正的多轮对话。
