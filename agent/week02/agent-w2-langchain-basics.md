# D1 LangChain 基础架构与核心概念

> **学习目标**：理解 LangChain 的设计哲学，掌握其核心组件的作用与使用方式，能够用 LangChain 构建最基础的 LLM 应用。

---

## 一、LangChain 是什么？为什么需要它？

大语言模型（LLM）本身只是一个"文本输入 → 文本输出"的接口。但真实的 AI 应用往往更复杂：需要记住上下文、调用外部工具、处理结构化数据、编排多步骤逻辑……

LangChain 诞生于 2022 年，目标就是解决这些问题：**把 LLM 变成可组合、可编排的应用构建模块**。它不是另一个模型，而是一个框架，让开发者能够用统一的 API 接入各种 LLM，并以"链式"方式组合各种能力。

### LangChain 的三层价值

1. **抽象层**：统一接口屏蔽底层差异（OpenAI / Anthropic / 本地模型）
2. **组合层**：提供链（Chain）、代理（Agent）、记忆（Memory）等高级组件
3. **生态层**：超过 600 个第三方集成，覆盖向量数据库、工具、加载器等

---

## 二、核心架构概览

LangChain 的架构由几个相互配合的层次组成：

```
┌─────────────────────────────────────────────────┐
│                   应用层（Application）            │
│   Agents / Chains / 自定义业务逻辑                 │
├─────────────────────────────────────────────────┤
│                  组件层（Components）              │
│  LLMs  │  Prompts  │  Memory  │  Tools  │ Indexes │
├─────────────────────────────────────────────────┤
│               底层集成层（Integrations）            │
│  OpenAI  │  HuggingFace  │  Pinecone  │  Weaviate │
└─────────────────────────────────────────────────┘
```

LangChain 的包结构（v0.2+ 后拆分为多个子包）：

| 包名 | 用途 |
|------|------|
| `langchain-core` | 核心抽象：Runnable、BaseMessage、PromptTemplate 等 |
| `langchain` | 高层链、Agent 实现 |
| `langchain-community` | 第三方集成（向量库、工具等） |
| `langchain-openai` | OpenAI 专属集成 |
| `langgraph` | 有状态的多 Agent 工作流（独立包） |

---

## 三、核心组件详解

### 3.1 LLMs 与 ChatModels

LangChain 把模型分为两类：
- **LLM**：输入字符串 → 输出字符串（老式 completion API）
- **ChatModel**：输入消息列表 → 输出消息（现代 chat API，推荐使用）

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 初始化 ChatModel
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key="your-api-key"
)

# 直接调用
messages = [
    SystemMessage(content="你是一个专业的 Python 工程师"),
    HumanMessage(content="请解释什么是装饰器")
]
response = llm.invoke(messages)
print(response.content)

# 流式输出
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
```

### 3.2 PromptTemplate — 提示词模板

PromptTemplate 让提示词变得可复用、可参数化：

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# 简单字符串模板
simple_prompt = PromptTemplate.from_template(
    "请用{language}解释{concept}的概念，给出一个代码示例"
)
print(simple_prompt.format(language="Python", concept="闭包"))

# Chat 消息模板（推荐）
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个精通{domain}的专家，回答要简洁专业"),
    ("human", "{question}"),
])

# 格式化为消息列表
messages = chat_prompt.format_messages(
    domain="机器学习",
    question="什么是过拟合？如何解决？"
)
```

#### MessagesPlaceholder — 动态注入消息历史

```python
from langchain_core.prompts import MessagesPlaceholder

prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    MessagesPlaceholder(variable_name="history"),  # 动态插入历史消息
    ("human", "{input}"),
])
```

### 3.3 OutputParser — 输出解析器

将 LLM 的文本输出解析为结构化数据：

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# 字符串解析（最简单）
str_parser = StrOutputParser()

# JSON 解析到 Pydantic 模型
class BookReview(BaseModel):
    title: str = Field(description="书名")
    rating: int = Field(description="评分，1-10分")
    summary: str = Field(description="简短总结")
    pros: List[str] = Field(description="优点列表")

json_parser = JsonOutputParser(pydantic_object=BookReview)

# 在 Prompt 中注入格式说明
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个书评家"),
    ("human", "请评价《{book_name}》\n\n{format_instructions}"),
])
prompt = prompt.partial(format_instructions=json_parser.get_format_instructions())

# 组成链
chain = prompt | llm | json_parser
result = chain.invoke({"book_name": "Python Tricks"})
print(result)  # BookReview 对象
```

---

## 四、LCEL — LangChain 表达式语言

LCEL（LangChain Expression Language）是 LangChain 的核心创新，用 `|` 管道符组合各个组件，让链的构建变得直观：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

# 用 | 将组件串联
chain = (
    ChatPromptTemplate.from_template("请将以下中文翻译成英文：{text}")
    | llm
    | StrOutputParser()
)

result = chain.invoke({"text": "人工智能正在改变世界"})
print(result)  # "Artificial intelligence is changing the world"
```

### Runnable 协议

LCEL 的核心是 **Runnable** 接口，所有组件都实现了它：

| 方法 | 说明 |
|------|------|
| `invoke(input)` | 同步调用，返回单个输出 |
| `stream(input)` | 流式输出，返回生成器 |
| `batch(inputs)` | 批量调用，并行处理 |
| `ainvoke(input)` | 异步调用 |
| `astream(input)` | 异步流式输出 |

```python
# 批量调用（并行处理多个输入）
texts = ["苹果", "香蕉", "橙子"]
results = chain.batch([{"text": t} for t in texts])

# 异步调用
import asyncio
async def translate_async():
    result = await chain.ainvoke({"text": "你好世界"})
    return result
```

### RunnableParallel — 并行执行

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 并行运行多个子链
parallel_chain = RunnableParallel(
    translation=chain,
    original=RunnablePassthrough()  # 直接透传输入
)

result = parallel_chain.invoke({"text": "人工智能"})
# result = {"translation": "Artificial Intelligence", "original": {"text": "人工智能"}}
```

---

## 五、Memory — 记忆管理

默认情况下，LLM 无状态（每次调用独立），Memory 组件让对话具有上下文连续性：

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 内存中存储历史（生产环境建议用 Redis/数据库）
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 构建带记忆的链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# 包装为带历史的链
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 第一轮对话
response1 = chain_with_history.invoke(
    {"input": "我叫小明，我喜欢 Python"},
    config={"configurable": {"session_id": "user_001"}}
)

# 第二轮对话（会记住上下文）
response2 = chain_with_history.invoke(
    {"input": "你知道我叫什么名字吗？"},
    config={"configurable": {"session_id": "user_001"}}
)
print(response2)  # "你叫小明，你之前提到喜欢 Python..."
```

---

## 六、Document Loaders 与 Text Splitters

LangChain 提供了丰富的文档加载和分割工具，是 RAG 应用的基础：

```python
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载 PDF
loader = PyPDFLoader("research_paper.pdf")
documents = loader.load()  # 返回 Document 列表

# 加载网页
web_loader = WebBaseLoader("https://example.com/article")
web_docs = web_loader.load()

# 文本分割
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 每块最大字符数
    chunk_overlap=200,    # 块间重叠字符数（保持上下文连续）
    separators=["\n\n", "\n", "。", "，", " ", ""]
)

chunks = splitter.split_documents(documents)
print(f"共分割为 {len(chunks)} 个文本块")
print(f"第一块内容：{chunks[0].page_content[:200]}")
print(f"第一块元数据：{chunks[0].metadata}")
```

---

## 七、Embeddings 与 VectorStore

将文本转化为向量，存入向量数据库，实现语义搜索：

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 初始化嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 创建向量数据库（内存模式）
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_docs"
)

# 相似度搜索
results = vectorstore.similarity_search(
    query="如何优化神经网络？",
    k=3  # 返回最相关的3个文档块
)

for doc in results:
    print(f"来源：{doc.metadata.get('source', 'unknown')}")
    print(f"内容：{doc.page_content[:200]}\n")

# 转为 Retriever 使用
retriever = vectorstore.as_retriever(
    search_type="mmr",    # 最大边际相关性，增加多样性
    search_kwargs={"k": 5, "fetch_k": 20}
)
```

---

## 八、完整示例：简单问答机器人

将所有组件整合，构建一个带记忆的简单 QA Bot：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def create_chatbot():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的 AI 学习助手。
        - 回答要清晰、准确、有条理
        - 适当给出代码示例
        - 鼓励学习者继续探索"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    store = {}
    def get_history(session_id):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    return RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )

bot = create_chatbot()
session_config = {"configurable": {"session_id": "learner_001"}}

# 多轮对话测试
questions = [
    "什么是 LangChain？",
    "它和直接调用 OpenAI API 有什么区别？",
    "你刚才说的第一点是什么？",  # 测试记忆能力
]

for q in questions:
    print(f"Q: {q}")
    answer = bot.invoke({"input": q}, config=session_config)
    print(f"A: {answer}\n")
```

---

## 九、核心概念对比总结

| 概念 | 作用 | 关键类 |
|------|------|--------|
| ChatModel | 封装 LLM 调用 | `ChatOpenAI`, `ChatAnthropic` |
| PromptTemplate | 参数化提示词 | `ChatPromptTemplate` |
| OutputParser | 解析模型输出 | `StrOutputParser`, `JsonOutputParser` |
| Chain (LCEL) | 组合多个组件 | `|` 运算符 / `RunnableSequence` |
| Memory | 管理对话历史 | `RunnableWithMessageHistory` |
| Document | 文档数据结构 | `Document(page_content, metadata)` |
| Retriever | 语义检索接口 | `VectorStoreRetriever` |

---

## 十、常见问题与最佳实践

### 1. 选择 LLM vs ChatModel
现在所有主流模型都通过 Chat API 提供，建议始终使用 `ChatModel`（如 `ChatOpenAI`）。

### 2. 温度（Temperature）设置
- `temperature=0`：确定性输出，适合代码生成、数据提取
- `temperature=0.7`：适度创造性，适合对话、内容生成
- `temperature=1.0+`：高度随机，适合头脑风暴

### 3. 错误处理与重试
```python
from langchain_core.runnables import RunnableRetry

# 自动重试（最多3次）
robust_chain = chain.with_retry(
    retry_if_exception_type=(Exception,),
    stop_after_attempt=3,
    wait_exponential_jitter=True
)
```

### 4. 成本控制
```python
# 使用 callbacks 追踪 token 使用量
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"input": "Hello"})
    print(f"总 Token：{cb.total_tokens}")
    print(f"总费用：${cb.total_cost:.4f}")
```

---

## 小结

LangChain 的核心价值在于：
1. **统一抽象**：一套接口驯服所有 LLM
2. **可组合性**：LCEL 让复杂流水线变简单
3. **生态丰富**：几乎覆盖所有 AI 应用场景所需的工具

下一篇（D2）将深入探讨 LangChain 的链与工具使用，构建真正能"做事"的 AI 应用。
