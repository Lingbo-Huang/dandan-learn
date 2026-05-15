---
layout: default
title: "W6D2 · 短期记忆优化"
---

# 短期记忆：Context Window 的艺术

> **Week 6 · Day 2** | 难度：⭐⭐⭐

---

## Context Window 的本质

短期记忆就是 LLM 的 Context Window——它能"看到"的全部信息。管理好它，是 Agent 高效运行的基础。

```
Token Budget（以 gpt-4o 128k 为例）：
┌─────────────────────────────────────────────────┐
│ System Prompt     ≈ 2,000 tokens (1.5%)         │
│ 工具定义          ≈ 3,000 tokens (2.3%)         │
│ 历史对话          ≈ 20,000 tokens (15.6%)       │
│ 长期记忆检索结果   ≈ 5,000 tokens (3.9%)        │
│ 当前输入          ≈ 2,000 tokens (1.5%)         │
│ 预留输出空间       ≈ 4,000 tokens (3.1%)        │
│ 剩余安全缓冲       ≈ 92,000 tokens (71.6%)     │
└─────────────────────────────────────────────────┘
```

## 五种短期记忆策略

### 策略1：Buffer Memory（最简单）

```python
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

# 保存全部对话历史（危险！长对话会溢出）
memory = ConversationBufferMemory(
    return_messages=True,  # 返回消息对象
    memory_key="chat_history"
)

llm = ChatOpenAI(model="gpt-4o-mini")
chain = ConversationChain(llm=llm, memory=memory)

chain.predict(input="我的名字是Alice")
chain.predict(input="我在做一个AI项目")
response = chain.predict(input="总结一下我告诉你的信息")
print(response)
```

**缺点**：无限增长，长对话必溢出。

### 策略2：Window Memory（推荐）

```python
from langchain.memory import ConversationBufferWindowMemory

# 只保留最近 k 轮
memory = ConversationBufferWindowMemory(
    k=10,  # 保留最近10轮对话
    return_messages=True,
    human_prefix="用户",
    ai_prefix="助手"
)
```

**适用**：大多数对话场景的默认选择。

### 策略3：Summary Memory（重要信息不丢失）

```python
from langchain.memory import ConversationSummaryMemory, ConversationSummaryBufferMemory

# 将旧对话自动总结压缩
summary_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    max_token_limit=500,  # 总结不超过500 token
    return_messages=True
)

# 更好的变体：最近的保持原样，更早的压缩
buffer_summary = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    max_token_limit=2000,  # 超过这个限制才开始压缩
    return_messages=True
)

chain = ConversationChain(llm=llm, memory=buffer_summary)

# 模拟长对话
messages = [
    "我是一个产品经理",
    "我们的产品是一个在线教育平台",
    "我们有10万注册用户",
    "最大的问题是用户留存率低",
    "上个月我们上线了AI推荐功能",
    "AI推荐上线后，留存率提升了15%",
]
for msg in messages:
    chain.predict(input=msg)

# 查看压缩后的记忆
print("当前记忆：")
print(buffer_summary.load_memory_variables({})["history"])
```

### 策略4：Entity Memory（追踪关键实体）

```python
from langchain.memory import ConversationEntityMemory

# 自动追踪对话中提到的关键实体
entity_memory = ConversationEntityMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    return_messages=True
)

chain = ConversationChain(llm=llm, memory=entity_memory)

chain.predict(input="我叫Bob，是一个后端工程师")
chain.predict(input="我在用Python开发一个API服务")
chain.predict(input="我的同事Alice负责前端")

# 查看追踪的实体
entities = entity_memory.entity_store.store
print("追踪到的实体：")
for entity, description in entities.items():
    print(f"  {entity}: {description}")
```

输出示例：
```
追踪到的实体：
  Bob: 后端工程师，用Python开发API服务
  Alice: Bob的同事，负责前端
```

### 策略5：自定义压缩记忆（生产级）

```python
import tiktoken
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class SmartCompressMemory:
    """智能压缩记忆：保留最近N轮原文 + 更早内容的摘要"""
    
    def __init__(self, 
                 keep_recent: int = 5,
                 max_tokens: int = 4000,
                 model: str = "gpt-4o-mini"):
        self.keep_recent = keep_recent
        self.max_tokens = max_tokens
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.all_messages: List[BaseMessage] = []
        self.compressed_summary: str = ""
        self.encoding = tiktoken.encoding_for_model("gpt-4o")
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def add_message(self, role: str, content: str):
        """添加新消息"""
        if role == "human":
            self.all_messages.append(HumanMessage(content=content))
        else:
            self.all_messages.append(AIMessage(content=content))
        
        # 检查是否需要压缩
        self._maybe_compress()
    
    def _maybe_compress(self):
        """如果超过 token 限制，压缩旧消息"""
        recent_messages = self.all_messages[-self.keep_recent*2:]
        recent_text = " ".join([m.content for m in recent_messages])
        
        if self.count_tokens(recent_text) > self.max_tokens:
            # 压缩最早的消息
            old_messages = self.all_messages[:-self.keep_recent*2]
            if old_messages:
                messages_text = "\n".join([
                    f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
                    for m in old_messages
                ])
                
                summary_prompt = f"""将以下对话历史压缩为简洁的摘要（保留关键信息）：

{messages_text}

摘要："""
                new_summary = self.llm.invoke(summary_prompt).content
                
                # 合并旧摘要和新摘要
                if self.compressed_summary:
                    merge_prompt = f"""合并两段摘要：
摘要1：{self.compressed_summary}
摘要2：{new_summary}

合并后的摘要："""
                    self.compressed_summary = self.llm.invoke(merge_prompt).content
                else:
                    self.compressed_summary = new_summary
                
                # 只保留最近的消息
                self.all_messages = self.all_messages[-self.keep_recent*2:]
    
    def get_context(self) -> str:
        """获取当前上下文（摘要 + 最近消息）"""
        parts = []
        
        if self.compressed_summary:
            parts.append(f"[早期对话摘要]\n{self.compressed_summary}")
        
        if self.all_messages:
            recent_text = "\n".join([
                f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
                for m in self.all_messages
            ])
            parts.append(f"[最近对话]\n{recent_text}")
        
        return "\n\n".join(parts)

# 使用示例
smart_memory = SmartCompressMemory(keep_recent=5, max_tokens=2000)

for i in range(20):
    smart_memory.add_message("human", f"这是第{i+1}条消息，内容是关于话题{i%3+1}")
    smart_memory.add_message("ai", f"好的，我理解了关于话题{i%3+1}的信息。")

print("当前上下文：")
print(smart_memory.get_context()[:500])
```

## 记忆选型决策树

```
你的 Agent 需要记忆多少历史？
    │
    ├── 只需最近几轮 → ConversationBufferWindowMemory (k=5~10)
    │
    ├── 需要全部历史但内存有限 → ConversationSummaryBufferMemory
    │
    ├── 需要追踪特定实体 → ConversationEntityMemory
    │
    └── 长期用户会话 → 自定义压缩 + 向量数据库长期记忆
```

## 踩坑经验

### 坑1：Summary Memory 摘要质量差

**问题**：LLM 生成的摘要遗漏了关键细节（如用户说的具体数字）。  
**解法**：在摘要 prompt 中明确要求保留具体数字、名字、日期等关键信息。

### 坑2：Entity Memory 实体识别错误

**问题**：把普通名词（"Python"）当作人名来追踪。  
**解法**：自定义实体类型白名单，只追踪：人名、组织名、项目名、产品名。

### 坑3：跨会话记忆丢失

**问题**：LangChain 的内置 Memory 默认不持久化，程序重启后记忆消失。  
**解法**：使用带持久化的 Memory 实现：

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

# 用 SQLite 持久化记忆
message_history = SQLChatMessageHistory(
    session_id="user_alice_001",
    connection_string="sqlite:///memory.db"
)
# 下次启动自动恢复
```

---

*W6D2 · 短期记忆优化 | Agent + Claw 系列*
