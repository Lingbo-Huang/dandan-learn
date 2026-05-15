---
layout: default
title: "W6D1 · 记忆系统概论"
---

# Agent 记忆系统：四种记忆类型全解析

> **Week 6 · Day 1** | 难度：⭐⭐⭐

---

## 为什么 Agent 需要记忆？

没有记忆的 Agent 是"失忆症患者"：每次对话都从零开始，无法积累经验，无法了解用户。

人类认知科学将记忆分为四类，Agent 系统也可以类比设计：

```
┌─────────────────────────────────────────────────────┐
│               Agent 记忆系统全景                     │
│                                                     │
│  ┌──────────────┐     ┌─────────────────────────┐   │
│  │  短期记忆     │     │       长期记忆            │   │
│  │ (Working     │     │                         │   │
│  │  Memory)    │     │  ┌─────────┐ ┌────────┐ │   │
│  │             │     │  │情景记忆  │ │语义记忆│ │   │
│  │ Context     │     │  │(Episodic│ │(Semantic│ │   │
│  │ Window内    │     │  │Memory) │ │Memory) │ │   │
│  │             │     │  └─────────┘ └────────┘ │   │
│  │ ≈ 人的工作  │     │  ┌─────────────────────┐ │   │
│  │ 记忆        │     │  │    程序性记忆         │ │   │
│  │             │     │  │  (Procedural)       │ │   │
│  └──────────────┘     │  └─────────────────────┘ │   │
│                       └─────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## 四种记忆类型详解

### 1. 短期记忆（Short-term / Working Memory）

**类比**：人类的工作记忆，保持当前正在处理的信息。  
**在 Agent 中**：LLM 的 Context Window 就是短期记忆。

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

# 窗口记忆：只保留最近 N 轮对话
memory = ConversationBufferWindowMemory(k=5)  # 保留最近5轮

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = ConversationChain(llm=llm, memory=memory, verbose=True)

# 对话
chain.predict(input="你好，我叫小明")
chain.predict(input="我喜欢Python")
chain.predict(input="你还记得我叫什么吗？")  # 记得，因为在窗口内
```

**优点**：快，无需额外存储  
**缺点**：容量有限，会话结束即丢失

### 2. 情景记忆（Episodic Memory）

**类比**：人类对具体事件的记忆（"上周三我去了医院"）。  
**在 Agent 中**：记录对话历史和事件序列，可跨会话检索。

```python
import json
from datetime import datetime
from pathlib import Path

class EpisodicMemory:
    """情景记忆：记录具体事件"""
    
    def __init__(self, storage_path: str = "/tmp/episodic_memory.json"):
        self.storage_path = Path(storage_path)
        self.episodes = self._load()
    
    def _load(self) -> list:
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                return json.load(f)
        return []
    
    def _save(self):
        with open(self.storage_path, "w") as f:
            json.dump(self.episodes, f, ensure_ascii=False, indent=2)
    
    def record(self, event: str, context: str = "", tags: list = None):
        """记录一个事件"""
        episode = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "context": context,
            "tags": tags or [],
            "id": len(self.episodes)
        }
        self.episodes.append(episode)
        self._save()
        return episode
    
    def recall(self, query: str = None, tags: list = None, 
               limit: int = 10) -> list:
        """检索相关事件"""
        results = self.episodes
        
        if tags:
            results = [e for e in results 
                      if any(t in e["tags"] for t in tags)]
        
        if query:
            results = [e for e in results 
                      if query.lower() in e["event"].lower() 
                      or query.lower() in e["context"].lower()]
        
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_timeline(self, days: int = 7) -> str:
        """获取最近N天的事件时间线"""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        recent = [e for e in self.episodes if e["timestamp"] >= cutoff]
        
        lines = []
        for ep in sorted(recent, key=lambda x: x["timestamp"]):
            ts = datetime.fromisoformat(ep["timestamp"]).strftime("%m/%d %H:%M")
            lines.append(f"[{ts}] {ep['event']}")
        
        return "\n".join(lines) if lines else "最近没有事件记录"

# 使用示例
memory = EpisodicMemory()
memory.record("用户询问了Python异步编程", tags=["技术", "Python"])
memory.record("完成了销售报告的数据分析", tags=["工作", "分析"])
memory.record("用户偏好Markdown格式的输出", tags=["偏好"])

print(memory.get_timeline())
```

### 3. 语义记忆（Semantic Memory）

**类比**：人类对概念和事实的记忆（"Python是一种编程语言"）。  
**在 Agent 中**：向量数据库存储知识库，语义搜索检索。

```python
# 明天的 D3 课程会详细讲
# 简单示例：
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 知识存入向量数据库
docs = [
    "Python是一种高级编程语言，以简洁的语法著称",
    "LangChain是构建LLM应用的框架",
    "向量数据库用于存储和检索高维向量"
]

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(docs, embeddings)

# 语义搜索
results = vectorstore.similarity_search("如何构建AI应用？", k=2)
for r in results:
    print(r.page_content)
```

### 4. 程序性记忆（Procedural Memory）

**类比**：人类对技能的记忆（"如何骑自行车"）。  
**在 Agent 中**：Agent 的 Skill/工具，以及通过 few-shot 学到的操作模式。

```python
# 程序性记忆 = 可复用的操作序列
class ProceduralMemory:
    """存储和检索操作程序"""
    
    def __init__(self):
        self.procedures: dict = {}
    
    def learn_procedure(self, name: str, description: str, 
                       steps: list, examples: list = None):
        """学习一个新程序"""
        self.procedures[name] = {
            "description": description,
            "steps": steps,
            "examples": examples or [],
            "usage_count": 0
        }
    
    def recall_procedure(self, task: str) -> str:
        """根据任务描述找到最相关的程序"""
        # 简单关键词匹配（生产中用向量搜索）
        for name, proc in self.procedures.items():
            if any(kw in task.lower() for kw in proc["description"].lower().split()):
                proc["usage_count"] += 1
                steps_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(proc["steps"])])
                return f"程序：{name}\n步骤：\n{steps_text}"
        return "没有找到相关程序"

procedural = ProceduralMemory()
procedural.learn_procedure(
    name="数据分析流程",
    description="数据 CSV 分析 统计 可视化",
    steps=["读取CSV文件", "检查空值", "描述性统计", "可视化", "得出结论"]
)
```

## 记忆系统选型指南

```
记忆类型    存储位置        检索方式      适用场景
──────────────────────────────────────────────────
短期记忆    Context Window  直接读取      当前会话推理
情景记忆    数据库/文件     关键词/时间   历史对话回溯
语义记忆    向量数据库      语义搜索      知识问答/RAG
程序性记忆  代码/Few-shot   精确匹配      固定操作序列
```

## 踩坑经验

### 坑1：全部放短期记忆——Context 爆炸

**问题**：把历史对话都塞进 Context，很快超出 token 限制。  
**解法**：短期记忆只保留最近3-5轮，历史用情景记忆存储，按需检索。

### 坑2：记忆系统孤立——各类记忆不互通

**解法**：设计统一的记忆接口，让 Agent 可以同时查询多种记忆：

```python
class UnifiedMemorySystem:
    def __init__(self):
        self.short_term = ConversationBufferWindowMemory(k=5)
        self.episodic = EpisodicMemory()
        self.semantic = vectorstore  # 向量数据库
    
    def retrieve(self, query: str) -> str:
        """统一检索接口"""
        # 检索所有类型的记忆
        episodes = self.episodic.recall(query, limit=3)
        semantic_results = self.semantic.similarity_search(query, k=3)
        
        context = ""
        if episodes:
            context += "相关历史事件：\n"
            context += "\n".join([f"- {e['event']}" for e in episodes])
        
        if semantic_results:
            context += "\n相关知识：\n"
            context += "\n".join([f"- {r.page_content}" for r in semantic_results])
        
        return context
```

---

*W6D1 · 记忆系统概论 | Agent + Claw 系列*
