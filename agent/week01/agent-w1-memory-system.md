# D4 · Agent 记忆系统——让 AI 真正"记住"事情

> **Week 1 主题**：什么是 Agent——定义 / ReAct / 规划 / 记忆 / 工具调用  
> **本日主题**：记忆（Memory）——短期 / 长期 / 向量检索

---

## 🎯 学习目标

1. 理解 Agent 记忆系统的四种类型及其工作原理
2. 掌握短期记忆（Context Window）的管理策略
3. 理解向量数据库在长期记忆中的作用
4. 能构建一个带有持久化记忆的对话 Agent

---

## 📚 核心知识点

### 1. 为什么 Agent 需要记忆？

**问题**：LLM 本身是无状态的——每次调用都是全新的，没有任何关于过去交互的"记忆"。

```
# 没有记忆的对话
用户: 我叫丹丹
AI: 很高兴认识你，丹丹！

[新的API调用]
用户: 我叫什么名字？
AI: 我不知道您叫什么名字... ❌
```

**记忆让 Agent 具备**：
- 持续对话能力
- 跨会话的个人化
- 积累学习（从过去的错误中学习）
- 长任务状态追踪

### 2. 四种记忆类型

#### 🧠 短期记忆（Sensory/Working Memory）
- **载体**：LLM 的 Context Window
- **容量**：有限（GPT-4: 128K tokens, Claude: 200K tokens）
- **特点**：对话内容直接放入 prompt
- **管理**：需要摘要/截断策略

#### 📚 长期记忆（Long-term Memory）
- **载体**：向量数据库、关系数据库、文件系统
- **容量**：无限（理论上）
- **特点**：需要检索才能使用
- **管理**：向量化存储，语义检索

#### 📋 操作性记忆（Episodic Memory）
- **载体**：任务执行日志
- **特点**：记录"做过什么"
- **应用**：错误恢复、任务续传

#### 🔧 程序性记忆（Procedural Memory）
- **载体**：系统提示词、Few-shot 示例
- **特点**："如何做"的隐式记忆
- **应用**：技能和行为规范

### 3. 短期记忆管理策略

**问题**：对话越长，Context Window 越容易溢出

**方案一：滑动窗口**
```
保留最近 N 轮对话
```

**方案二：摘要压缩**
```
当对话超过阈值时，用 LLM 将历史对话压缩为摘要
[摘要] + [最近K轮] = 新的 Context
```

**方案三：重要性筛选**
```
给每条消息打重要性分数
只保留分数高的消息
```

**方案四：Token 预算管理**
```python
MAX_TOKENS = 8000
if count_tokens(history) > MAX_TOKENS:
    history = summarize(history)
```

### 4. 向量数据库与语义检索

**工作原理**：
```
存储时：
  文本 → Embedding 模型 → 向量（float[]）→ 存入向量DB

检索时：
  查询文本 → 向量 → 相似度搜索 → Top-K 最相关文档
```

**主流向量数据库**：

| 数据库 | 特点 | 适用场景 |
|--------|------|---------|
| ChromaDB | 轻量，嵌入式 | 本地开发/原型 |
| Pinecone | 云服务，全托管 | 生产环境 |
| Weaviate | 开源，功能丰富 | 自托管 |
| Qdrant | 高性能，Rust实现 | 大规模 |
| FAISS | 纯检索库（Meta） | 算法研究 |

**相似度算法**：
- 余弦相似度（最常用）
- 欧氏距离
- 点积（内积）

### 5. Memory-Augmented Agent 架构

```
用户输入
    ↓
[记忆检索] ← 向量DB中检索相关历史
    ↓
[构建 Prompt]
  系统提示 + 相关记忆 + 当前对话上下文 + 用户输入
    ↓
[LLM 推理]
    ↓
[输出 + 记忆更新]
  → 将新对话存入向量DB
  → 更新短期对话历史
```

---

## 💡 示例/推导

### 推导：为什么向量检索比关键词检索更好？

```
用户存储了一条记忆："丹丹喜欢在周末骑自行车健身"

用户查询："有什么户外运动推荐？"

关键词检索（BM25）：
  搜索关键词："户外运动"
  结果：无匹配（记忆中没有"户外运动"这个词）❌

向量检索（Semantic Search）：
  "户外运动" → 向量 [0.2, 0.8, 0.1, ...]
  "骑自行车健身" → 向量 [0.25, 0.75, 0.15, ...]  ← 语义相近！
  相似度高 → 返回这条记忆 ✅
  
Agent 能说：基于你喜欢骑自行车健身，我推荐...
```

### 记忆摘要示例

```
[原始对话（15轮，约3000 tokens）]：
讨论了 Python 基础、函数式编程、装饰器、异步编程...

[LLM 压缩摘要（约200 tokens）]：
"用户正在学习 Python，已掌握基础语法和函数式编程，
对装饰器和异步编程有疑问，倾向于通过实例学习"

[新 Context = 摘要 + 最近3轮对话]
```

---

## 🔧 动手练习

### 练习 1：短期记忆管理（必做）

```python
# 创建文件: 06_conversation_memory.py
# uv run python 06_conversation_memory.py

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 使用摘要缓冲记忆：超过 token 阈值时自动压缩历史对话
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=500,  # 超过500 tokens 时触发摘要
    return_messages=True
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

print("=== 带记忆的对话 Agent ===")
print("（输入 quit 退出，输入 memory 查看当前记忆状态）\n")

while True:
    user_input = input("你: ").strip()
    
    if user_input.lower() == "quit":
        break
    
    if user_input.lower() == "memory":
        print("\n📚 当前记忆状态:")
        print(f"  摘要: {memory.moving_summary_buffer}")
        print(f"  最近对话轮数: {len(memory.chat_memory.messages)}")
        print()
        continue
    
    response = conversation.predict(input=user_input)
    print(f"AI: {response}\n")
```

### 练习 2：构建向量记忆系统（核心练习）

```python
# 创建文件: 07_vector_memory.py
# uv add chromadb sentence-transformers
# uv run python 07_vector_memory.py

import chromadb
from chromadb.utils import embedding_functions
import time

# 使用 ChromaDB 作为向量存储（本地嵌入式，无需 API key）
client = chromadb.Client()

# 使用默认的句向量模型
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# 创建记忆集合
memory_collection = client.create_collection(
    name="agent_memory",
    embedding_function=embedding_fn
)

class VectorMemory:
    """基于向量数据库的 Agent 记忆系统"""
    
    def __init__(self, collection):
        self.collection = collection
        self._counter = 0
    
    def remember(self, content: str, metadata: dict = None):
        """存储一条记忆"""
        self._counter += 1
        doc_id = f"mem_{self._counter}_{int(time.time())}"
        
        self.collection.add(
            documents=[content],
            ids=[doc_id],
            metadatas=[metadata or {"timestamp": time.time()}]
        )
        print(f"  💾 已记住: {content[:50]}...")
    
    def recall(self, query: str, n_results: int = 3) -> list[str]:
        """检索相关记忆"""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
        )
        
        if not results["documents"][0]:
            return []
        
        return results["documents"][0]
    
    def count(self) -> int:
        return self.collection.count()

# 测试向量记忆
memory = VectorMemory(memory_collection)

print("=== 向量记忆系统测试 ===\n")

# 存储一些记忆
print("📥 存储记忆...")
memories_to_store = [
    "用户叫丹丹，是一名后端工程师",
    "用户正在学习 AI Agent 开发",
    "用户喜欢用 Python 写代码",
    "用户的时区是 UTC+8（北京时间）",
    "用户上次提问是关于 ReAct 框架的",
    "用户对 LangChain 比较熟悉",
    "用户希望学完后能独立开发 Agent 项目",
]

for m in memories_to_store:
    memory.remember(m)

print(f"\n总共存储了 {memory.count()} 条记忆\n")

# 语义检索测试
print("🔍 语义检索测试:")
queries = [
    "这个用户的职业是什么？",
    "用户在学什么技术？",
    "用户有什么编程偏好？",
]

for query in queries:
    print(f"\n查询: {query}")
    results = memory.recall(query, n_results=2)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r}")
```

### 练习 3：带长期记忆的 Agent（进阶）

```python
# 创建文件: 08_memory_agent.py
# uv run python 08_memory_agent.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from chromadb.utils import embedding_functions
import json, time

class MemoryAgent:
    """具有短期 + 长期记忆的 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        # 短期记忆：最近的对话
        self.short_term: list[dict] = []
        self.max_short_term = 10  # 最多保留10轮
        
        # 长期记忆：向量数据库
        self.db_client = chromadb.Client()
        self.long_term = self.db_client.create_collection(
            name="long_term_memory",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        self._mem_counter = 0
    
    def _retrieve_relevant_memories(self, query: str) -> str:
        if self.long_term.count() == 0:
            return ""
        n = min(3, self.long_term.count())
        results = self.long_term.query(query_texts=[query], n_results=n)
        memories = results["documents"][0]
        if memories:
            return "【相关历史记忆】\n" + "\n".join(f"- {m}" for m in memories)
        return ""
    
    def _save_to_long_term(self, conversation_summary: str):
        self._mem_counter += 1
        self.long_term.add(
            documents=[conversation_summary],
            ids=[f"ltm_{self._mem_counter}"]
        )
    
    def chat(self, user_input: str) -> str:
        # 检索相关长期记忆
        relevant_memories = self._retrieve_relevant_memories(user_input)
        
        # 构建 prompt
        system = f"""你是一个有记忆能力的 AI 助手。
        
{relevant_memories}

请基于以上记忆和当前对话来回复用户。"""
        
        # 将短期记忆加入对话
        messages = [{"role": "system", "content": system}]
        messages.extend(self.short_term[-6:])  # 最近3轮
        messages.append({"role": "user", "content": user_input})
        
        response = self.llm.invoke(messages)
        reply = response.content
        
        # 更新短期记忆
        self.short_term.append({"role": "user", "content": user_input})
        self.short_term.append({"role": "assistant", "content": reply})
        
        # 每5轮对话，提取摘要存入长期记忆
        if len(self.short_term) % 10 == 0:
            summary_prompt = f"请用1-2句话总结以下对话的关键信息：\n{json.dumps(self.short_term[-10:], ensure_ascii=False)}"
            summary = self.llm.invoke(summary_prompt).content
            self._save_to_long_term(summary)
            print(f"  [系统: 已将对话摘要存入长期记忆]")
        
        return reply

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = MemoryAgent()
    print("=== 记忆 Agent（输入 quit 退出）===\n")
    
    while True:
        user = input("你: ").strip()
        if user.lower() == "quit":
            break
        reply = agent.chat(user)
        print(f"AI: {reply}\n")
```

### 🦞 Claw 实战：理解 Claw 的记忆架构

Claw Agent 的记忆系统体现在：
- **`MEMORY.md`**：Agent 的长期记忆（跨会话持久化）
- **`memory/YYYY-MM-DD.md`**：每日操作性记忆日志
- **任务备忘（`/md` 接口）**：任务执行的片段记忆

**实战任务**：
1. 查看你工作区的 `MEMORY.md` 和 `memory/` 目录
2. 思考这对应今天学的哪种记忆类型？
3. 如果要给 Claw Agent 加一个"跨任务语义检索"能力，你会怎么设计？

---

## 📝 小结

| 记忆类型 | 载体 | 容量 | 访问方式 |
|---------|------|------|---------|
| 短期记忆 | Context Window | 有限 | 直接注入 prompt |
| 长期记忆 | 向量数据库 | 无限 | 语义检索 |
| 操作性记忆 | 执行日志/文件 | 无限 | 读文件 |
| 程序性记忆 | 系统提示词 | 有限 | 写入 system prompt |

**明天预告**：工具调用（Function Calling）——Agent 如何与外部世界交互？从 OpenAI Function Calling 到自定义工具。

---

> 💡 **今日思考题**：如果一个 Agent 积累了大量"错误的记忆"（比如记住了一些过时的错误信息），应该如何设计记忆清理/更新机制？
