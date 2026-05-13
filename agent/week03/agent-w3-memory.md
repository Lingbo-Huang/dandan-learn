---
layout: default
title: "D3 · 记忆系统：短期与长期记忆"
---

# D3 · Agent 记忆系统

> **Agent Week 3**  
> 没有记忆的 Agent 像失忆症患者——每次对话都从零开始。今天学如何给 Agent 装上记忆。

---

## 一、记忆的分类

| 类型 | 类比 | 技术实现 |
|------|------|---------|
| **短期记忆** | 工作记忆，对话中的内容 | 对话历史列表（in-context） |
| **长期记忆** | 持久化，跨对话记忆 | 向量数据库、结构化存储 |
| **外部记忆** | 工具访问的知识库 | RAG、文件系统、API |

---

## 二、短期记忆：对话历史

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini")

# 维护对话历史
history = [
    SystemMessage(content="你是一个 AI 学习助手，帮助用户学习机器学习。")
]

def chat(user_input: str) -> str:
    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history)
    history.append(AIMessage(content=response.content))
    return response.content

# 多轮对话
print(chat("什么是梯度下降？"))
print("---")
print(chat("它有什么变体？"))        # 能记住上文是在聊梯度下降
print("---")
print(chat("哪个在实践中用得最多？"))  # 继续追问
```

---

## 三、历史消息管理（避免上下文窗口溢出）

```python
from langchain_core.messages import trim_messages, HumanMessage, AIMessage

# 当历史太长时，自动截断
trimmer = trim_messages(
    max_tokens=2000,      # 最多保留 2000 tokens
    strategy="last",      # 保留最近的消息
    token_counter=llm,    # 用模型的 tokenizer 计算
    include_system=True,  # 保留 system message
)

# 或者简单地只保留最近 N 轮
class ConversationWithMemory:
    def __init__(self, max_rounds=10):
        self.max_rounds = max_rounds
        self.history = []
        self.llm = ChatOpenAI(model="gpt-4o-mini")
    
    def chat(self, user_input: str) -> str:
        self.history.append(HumanMessage(content=user_input))
        
        # 只保留最近 max_rounds 轮（每轮2条消息）
        recent_history = self.history[-self.max_rounds * 2:]
        
        response = self.llm.invoke([
            SystemMessage(content="你是一个有帮助的助手。"),
            *recent_history
        ])
        
        self.history.append(AIMessage(content=response.content))
        return response.content
    
    def clear(self):
        self.history = []
```

---

## 四、LangChain 的 MessageHistory（持久化）

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOpenAI(model="gpt-4o-mini")

# 用 SQLite 持久化对话历史
def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db"
    )

# 包装 Chain，自动管理历史
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 AI 助手。"),
    MessagesPlaceholder(variable_name="history"),  # 历史消息占位符
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 使用（相同 session_id 自动连接历史）
config = {"configurable": {"session_id": "user_123"}}

response1 = chain_with_history.invoke(
    {"input": "我叫小明，在学习机器学习"},
    config=config
)
response2 = chain_with_history.invoke(
    {"input": "我叫什么名字？"},
    config=config
)
print(response2)  # 应该能回答"小明"
```

---

## 五、长期记忆：向量存储

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import VectorStoreRetrieverMemory

# 创建向量存储
embedding = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embedding)

# 存入记忆
vectorstore.add_texts([
    "用户叫小明，是一个 AI 初学者",
    "用户喜欢用 Python 编程",
    "用户对量化交易感兴趣",
    "上次对话讨论了梯度下降",
])

# 检索相关记忆
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 查询时自动检索最相关的历史记忆
query = "梯度下降有什么问题我们讨论过？"
relevant_memories = retriever.invoke(query)
for mem in relevant_memories:
    print(mem.page_content)
```

---

## 六、记忆摘要（避免历史太长）

```python
from langchain.memory import ConversationSummaryMemory

# 当对话历史太长时，自动总结压缩
summary_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    max_token_limit=500,  # 超过则触发摘要
)

# 将长历史压缩为摘要
def get_summary_with_recent(history, llm, max_recent=4):
    """
    策略：最近 N 轮保持完整 + 更早的历史转为摘要
    """
    if len(history) <= max_recent * 2:
        return history
    
    # 早期历史生成摘要
    old_history = history[:-max_recent*2]
    recent_history = history[-max_recent*2:]
    
    summary_prompt = f"请用 200 字以内总结以下对话历史：\n{old_history}"
    summary = llm.invoke(summary_prompt).content
    
    return [SystemMessage(content=f"早期对话摘要：{summary}")] + recent_history
```

---

## 今天的关键认识

1. **短期记忆**：维护 `history` 列表，作为消息传给 LLM
2. **历史管理**：上下文有限制，需要截断或摘要
3. **持久化**：用 `RunnableWithMessageHistory` + 数据库，跨会话保存
4. **长期记忆**：向量数据库存储知识，检索时按语义相似度找相关记忆

---

## 明天预告

D4：**RAG（检索增强生成）**——让 Agent 能够访问和引用知识库，大幅提升回答质量。
