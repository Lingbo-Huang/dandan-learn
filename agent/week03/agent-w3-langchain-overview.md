---
layout: default
title: "D1 · LangChain 架构全览"
---

# D1 · LangChain 架构全览

> **Agent Week 3**  
> LangChain 是 LLM 应用开发的事实标准框架。今天搞清楚它的整体设计。

---

## 一、LangChain 的核心设计哲学

**问题**：把 LLM 的能力组合成真正有用的应用，需要大量胶水代码。

**解决**：LangChain 提供了标准化的抽象层和可组合的模块。

```
LangChain 的四大核心概念：

Model I/O       → 标准化 LLM 调用接口
Data Connection → 数据加载、分割、嵌入、存储、检索
Chains          → 将多个步骤串联成可复用的管道
Agents          → 让 LLM 动态决定调用哪些工具
```

---

## 二、安装与基础配置

```python
# 安装
# pip install langchain langchain-openai langchain-community

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# 初始化模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 最简单的调用
response = llm.invoke("用一句话解释什么是 Transformer")
print(response.content)
```

---

## 三、Prompt Template

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# 简单字符串模板
template = PromptTemplate.from_template(
    "你是一个 {subject} 专家。请用 {level} 水平解释：{question}"
)

# Chat 模板（更常用）
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个 {role}，回答要{style}。"),
    ("human", "{question}"),
])

# 使用模板
prompt = chat_template.invoke({
    "role": "量化分析师",
    "style": "简洁且专业",
    "question": "什么是夏普比率？"
})

response = llm.invoke(prompt)
print(response.content)
```

---

## 四、Output Parser

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# 字符串解析器（最简单）
str_parser = StrOutputParser()

# 结构化输出（Pydantic）
class BookReview(BaseModel):
    title: str = Field(description="书名")
    score: int = Field(description="1-10分评分")
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")
    summary: str = Field(description="一句话总结")

from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=BookReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个书评人。{format_instructions}"),
    ("human", "请评价《{book_title}》"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
result = chain.invoke({"book_title": "三体"})
print(f"评分: {result.score}/10")
print(f"优点: {result.pros}")
```

---

## 五、第一个完整 Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# 翻译 Chain
translate_prompt = ChatPromptTemplate.from_template(
    "请将以下文本翻译成{language}，只输出翻译结果：\n\n{text}"
)

translate_chain = translate_prompt | llm | StrOutputParser()

# 调用
result = translate_chain.invoke({
    "language": "英文",
    "text": "大模型是一种基于 Transformer 架构的大规模语言模型"
})
print(result)

# 也支持批量处理
results = translate_chain.batch([
    {"language": "英文", "text": "人工智能"},
    {"language": "日文", "text": "机器学习"},
    {"language": "韩文", "text": "深度学习"},
])
```

---

## 六、LangChain 核心组件关系图

```
用户输入
    ↓
PromptTemplate  → 格式化 Prompt
    ↓
ChatModel/LLM   → 调用语言模型
    ↓
OutputParser    → 解析输出为结构化数据
    ↓
（可选）Tool     → 执行外部操作（搜索、计算等）
    ↓
（可选）Memory   → 存储对话历史
    ↓
最终输出
```

---

## 七、何时用 LangChain，何时不用？

**适合用 LangChain 的场景**：
- 需要标准化的 LLM 调用接口
- 构建 RAG 管道
- 快速原型验证
- 需要 LangSmith 监控追踪

**不太适合的场景**：
- 简单的 API 调用（直接用 openai SDK 更轻量）
- 高性能/低延迟生产系统（有些抽象层有额外开销）
- 非常定制化的流程（抽象可能反而限制灵活性）

---

## 今天的关键认识

1. **LangChain = 标准化 + 可组合**：Model I/O、Data Connection、Chains、Agents
2. **PromptTemplate**：变量化、可复用的 Prompt
3. **OutputParser**：把 LLM 的文字输出转成结构化数据
4. **管道用 `|` 连接**：`prompt | llm | parser`（LCEL，明天详细讲）

---

## 明天预告

D2：**LCEL**——LangChain Expression Language，理解 `|` 管道的设计哲学和高级用法。
