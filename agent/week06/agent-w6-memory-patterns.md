---
layout: default
title: "W6D6 · 记忆系统设计模式"
---

# 记忆系统设计模式：生产经验总结

> **Week 6 · Day 6** | 难度：⭐⭐⭐⭐

---

## 模式1：记忆流水线（Memory Pipeline）

```
用户输入 → 信息提取 → 去重检查 → 重要性评估 → 分类存储 → 索引更新
```

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

class ExtractedMemory(BaseModel):
    content: str
    category: str = Field(description="personal/fact/task/preference")
    importance: float = Field(ge=0, le=1)
    is_temporary: bool = Field(description="是否是临时信息，不应长期存储")

class MemoryPipeline:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.extract_llm = self.llm.with_structured_output(List[ExtractedMemory])
    
    def process(self, conversation: str) -> List[ExtractedMemory]:
        prompt = f"""从对话中提取值得长期记忆的信息：

{conversation}

只提取重要的、非临时的信息（忽略今天天气、临时请求等）。"""
        
        try:
            memories = self.extract_llm.invoke(prompt)
            # 过滤临时信息和低重要性记忆
            return [m for m in memories if not m.is_temporary and m.importance >= 0.4]
        except:
            return []

pipeline = MemoryPipeline()
test_conv = """
用户：我叫李华，是机器学习工程师
助手：你好李华！
用户：今天天气不错，帮我搜索一下Python文档
"""
memories = pipeline.process(test_conv)
for m in memories:
    print(f"[{m.category}] {m.content} (重要性:{m.importance})")
```

## 模式2：记忆优先级队列

```python
import heapq
from dataclasses import dataclass

@dataclass
class PrioritizedMemory:
    priority: float  # 负数（最大堆）
    content: str
    created_at: str
    
    def __lt__(self, other):
        return self.priority < other.priority

class PriorityMemoryCache:
    """优先级记忆缓存：保留最重要的 N 条记忆"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.heap: List[PrioritizedMemory] = []
    
    def add(self, content: str, priority: float):
        from datetime import datetime
        memory = PrioritizedMemory(
            priority=-priority,  # 取负数实现最大堆
            content=content,
            created_at=datetime.now().isoformat()
        )
        
        heapq.heappush(self.heap, memory)
        
        # 超出容量时，删除优先级最低的
        if len(self.heap) > self.max_size:
            # 找到优先级最低（priority 最大，即负数最小绝对值）的
            self.heap.sort()
            self.heap = self.heap[:self.max_size]
            heapq.heapify(self.heap)
    
    def get_top(self, n: int = 10) -> List[str]:
        top = heapq.nsmallest(n, self.heap)  # 最大优先级（最负）
        return [m.content for m in top]

cache = PriorityMemoryCache(max_size=50)
cache.add("用户是Python工程师", 0.9)
cache.add("用户喜欢简洁代码", 0.7)
cache.add("今天讨论了天气", 0.1)

print("最重要的记忆：")
for m in cache.get_top(3):
    print(f"  - {m}")
```

## 模式3：分层记忆（Hierarchical Memory）

```
L1 缓存 → L2 短期 → L3 长期
（最快）              （最全）
```

```python
class HierarchicalMemory:
    """三层记忆架构"""
    
    def __init__(self):
        # L1：热缓存（最近5轮对话）
        self.l1_cache: List[str] = []
        self.l1_max = 10
        
        # L2：短期记忆（最近30天）
        self.l2_memories: List[dict] = []
        
        # L3：长期记忆（向量数据库）
        self.l3_vectorstore = None  # 初始化略
    
    def add(self, content: str, layer: int = 1):
        """添加到指定层"""
        if layer == 1:
            self.l1_cache.append(content)
            if len(self.l1_cache) > self.l1_max:
                # L1 溢出 → 降级到 L2
                overflow = self.l1_cache.pop(0)
                self.add(overflow, layer=2)
    
    def retrieve(self, query: str) -> List[str]:
        """分层检索：优先 L1，然后 L2，最后 L3"""
        results = []
        
        # L1 直接返回
        results.extend(self.l1_cache[-5:])
        
        # L2 关键词匹配
        l2_hits = [m["content"] for m in self.l2_memories 
                   if query.lower() in m["content"].lower()][:3]
        results.extend(l2_hits)
        
        # L3 语义搜索（如果需要更多结果）
        if len(results) < 5 and self.l3_vectorstore:
            l3_results = self.l3_vectorstore.similarity_search(query, k=3)
            results.extend([r.page_content for r in l3_results])
        
        return results[:10]
```

## 模式4：记忆遗忘与巩固

```python
from datetime import datetime, timedelta

class MemoryConsolidator:
    """模拟人类的记忆巩固过程（睡眠中的记忆整理）"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def consolidate(self, recent_episodes: List[dict]) -> List[str]:
        """巩固近期情景记忆为长期语义记忆"""
        if not recent_episodes:
            return []
        
        episodes_text = "\n".join([
            f"- {ep['event']}"
            for ep in recent_episodes
        ])
        
        prompt = f"""将以下近期事件总结为值得长期记忆的知识点：

近期事件：
{episodes_text}

请提取2-3条通用的、值得长期记忆的规律或知识（不是具体事件）："""
        
        response = self.llm.invoke(prompt)
        learnings = [
            line.strip().lstrip("0123456789.-) ")
            for line in response.content.split("\n")
            if line.strip() and len(line.strip()) > 10
        ]
        
        return learnings
    
    def forget_curve(self, memories: List[dict]) -> List[dict]:
        """遗忘曲线：重要性随时间衰减（Ebbinghaus 遗忘曲线模拟）"""
        now = datetime.now()
        for mem in memories:
            created = datetime.fromisoformat(mem.get("created_at", now.isoformat()))
            days = (now - created).days
            
            # 遗忘曲线：R = e^(-t/S)，S 由重要性决定
            stability = mem.get("importance", 0.5) * 30  # 重要性越高，遗忘越慢
            import math
            retention = math.exp(-days / stability)
            
            mem["effective_importance"] = mem.get("importance", 0.5) * retention
        
        return sorted(memories, key=lambda x: x["effective_importance"], reverse=True)
```

## 完整记忆系统集成

```python
class IntegratedMemorySystem:
    """整合所有记忆类型的系统"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.short_term = []  # 当前会话
        self.episodic = EpisodicMemorySystem()
        self.semantic = SemanticMemory()
        # self.long_term = LongTermMemorySystem()  # 参考D4
    
    def process_input(self, user_input: str, agent_response: str):
        """处理一轮对话"""
        # 加入短期记忆
        self.short_term.append({"human": user_input, "ai": agent_response})
        
        # 记录情景
        self.episodic.record_episode(
            f"用户：{user_input[:50]}", 
            participants=[self.user_id, "Agent"]
        )
    
    def get_context_for_query(self, query: str) -> str:
        """获取查询的完整上下文"""
        parts = []
        
        # 短期记忆（最近3轮）
        recent = self.short_term[-3:]
        if recent:
            parts.append("最近对话：\n" + "\n".join([
                f"用户：{r['human']}\n助手：{r['ai']}"
                for r in recent
            ]))
        
        # 情景记忆
        episodes = self.episodic.recall_by_query(query, k=3)
        if episodes:
            parts.append("相关历史：\n" + "\n".join([
                f"- {ep['event']}" for ep in episodes
            ]))
        
        return "\n\n".join(parts) if parts else ""

system = IntegratedMemorySystem("user_001")
system.process_input("我在学习Agent记忆系统", "很好，记忆系统是Agent的关键组件")
system.process_input("向量数据库怎么选？", "推荐从ChromaDB开始，生产再用Pinecone")

context = system.get_context_for_query("记忆存储方案")
print("上下文：")
print(context[:300])
```

## 踩坑经验

### 坑1：记忆系统成为性能瓶颈

**问题**：每次对话都需要查询三种记忆系统，延迟增加了2-3秒。  
**解法**：
1. 并行查询多种记忆系统（asyncio）
2. L1 热缓存放内存，L3 异步查询
3. 非关键路径的记忆存储异步进行

### 坑2：记忆检索结果与当前对话无关

**问题**：搜索"Python"返回了很久以前一段无关的对话。  
**解法**：增加时间衰减权重，近期记忆优先；设置相似度阈值过滤低质量结果。

---

*W6D6 · 记忆系统设计模式 | Agent + Claw 系列*
