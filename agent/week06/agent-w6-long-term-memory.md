---
layout: default
title: "W6D4 · 长期记忆系统"
---

# 长期记忆：让 Agent 跨会话记住一切

> **Week 6 · Day 4** | 难度：⭐⭐⭐⭐

---

## 长期记忆的核心需求

长期记忆要解决的问题：
1. **持久化**：会话结束后记忆不丢失
2. **检索**：从大量历史中快速找到相关记忆
3. **更新**：新信息要能更新旧记忆（不能只追加）
4. **遗忘**：过时的记忆要能清理

## 架构设计

```
┌────────────────────────────────────────────────────┐
│              长期记忆系统架构                        │
│                                                    │
│  ┌──────────┐   新信息    ┌────────────────────┐   │
│  │  Agent   │──────────→│   记忆处理层        │   │
│  │          │            │  (提取/去重/更新)   │   │
│  │          │            └────────┬───────────┘   │
│  │          │                     │               │
│  │          │            ┌────────▼───────────┐   │
│  │          │            │   存储层            │   │
│  │          │            │  ┌──────┐ ┌──────┐ │   │
│  │          │←──检索────│  │向量DB│ │关系DB│ │   │
│  └──────────┘            │  │(语义)│ │(结构)│ │   │
│                           │  └──────┘ └──────┘ │   │
│                           └────────────────────┘   │
└────────────────────────────────────────────────────┘
```

## 完整实现：持久化长期记忆系统

```python
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

@dataclass
class Memory:
    """记忆单元"""
    id: str
    content: str
    category: str       # 事实/偏好/事件/技能
    importance: float   # 重要性 0-1
    created_at: str
    updated_at: str
    access_count: int = 0
    last_accessed: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class LongTermMemorySystem:
    """生产级长期记忆系统"""
    
    def __init__(self, 
                 db_path: str = "./memory.db",
                 chroma_dir: str = "./long_term_chroma"):
        self.db_path = db_path
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 初始化 SQLite（结构化存储）
        self._init_sqlite()
        
        # 初始化 ChromaDB（语义搜索）
        self.vectorstore = Chroma(
            collection_name="long_term_memory",
            embedding_function=self.embeddings,
            persist_directory=chroma_dir
        )
    
    def _init_sqlite(self):
        """初始化 SQLite 数据库"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT,
                importance REAL DEFAULT 0.5,
                created_at TEXT,
                updated_at TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                tags TEXT DEFAULT '[]'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON memories(category)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC)
        """)
        conn.commit()
        conn.close()
    
    def _generate_id(self, content: str) -> str:
        """基于内容生成唯一 ID（相似内容可能重用 ID）"""
        return hashlib.md5(content[:100].encode()).hexdigest()[:12]
    
    def store(self, content: str, category: str = "general", 
              importance: float = 0.5, tags: List[str] = None) -> Memory:
        """存储记忆（如已存在则更新）"""
        memory_id = self._generate_id(content)
        now = datetime.now().isoformat()
        
        memory = Memory(
            id=memory_id,
            content=content,
            category=category,
            importance=importance,
            created_at=now,
            updated_at=now,
            tags=tags or []
        )
        
        conn = sqlite3.connect(self.db_path)
        
        # 检查是否已存在
        existing = conn.execute(
            "SELECT id FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        
        if existing:
            # 更新
            conn.execute("""
                UPDATE memories 
                SET content=?, importance=?, updated_at=?, tags=?
                WHERE id=?
            """, (content, importance, now, json.dumps(tags or []), memory_id))
        else:
            # 插入
            conn.execute("""
                INSERT INTO memories 
                (id, content, category, importance, created_at, updated_at, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (memory_id, content, category, importance, now, now, 
                  json.dumps(tags or [])))
        
        conn.commit()
        conn.close()
        
        # 同步到向量数据库
        self.vectorstore.add_texts(
            texts=[content],
            metadatas=[{
                "memory_id": memory_id,
                "category": category,
                "importance": importance,
                "tags": str(tags or [])
            }],
            ids=[memory_id]
        )
        
        return memory
    
    def retrieve_by_query(self, query: str, k: int = 5, 
                         min_importance: float = 0.0) -> List[Memory]:
        """语义搜索检索记忆"""
        results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        memories = []
        conn = sqlite3.connect(self.db_path)
        
        for doc, score in results:
            memory_id = doc.metadata.get("memory_id")
            if not memory_id:
                continue
            
            row = conn.execute(
                "SELECT * FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            
            if row and row[3] >= min_importance:  # importance 字段
                memory = Memory(
                    id=row[0],
                    content=row[1],
                    category=row[2],
                    importance=row[3],
                    created_at=row[4],
                    updated_at=row[5],
                    access_count=row[6],
                    last_accessed=row[7],
                    tags=json.loads(row[8])
                )
                memories.append((memory, 1 - score))  # 转为相似度
        
        # 更新访问计数
        if memories:
            memory_ids = [m.id for m, _ in memories]
            conn.execute(
                f"UPDATE memories SET access_count = access_count + 1, "
                f"last_accessed = ? WHERE id IN ({','.join(['?']*len(memory_ids))})",
                [datetime.now().isoformat()] + memory_ids
            )
            conn.commit()
        
        conn.close()
        
        # 按相似度排序，返回最多 k 个
        memories.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in memories[:k]]
    
    def forget_old_memories(self, days_threshold: int = 30, 
                           importance_threshold: float = 0.3) -> int:
        """遗忘策略：删除旧的低重要性记忆"""
        cutoff = (datetime.now() - timedelta(days=days_threshold)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        
        # 找到需要删除的记忆
        to_delete = conn.execute("""
            SELECT id FROM memories 
            WHERE updated_at < ? AND importance < ? AND access_count < 3
        """, (cutoff, importance_threshold)).fetchall()
        
        delete_ids = [row[0] for row in to_delete]
        
        if delete_ids:
            conn.execute(
                f"DELETE FROM memories WHERE id IN ({','.join(['?']*len(delete_ids))})",
                delete_ids
            )
            conn.commit()
            
            # 从向量数据库删除
            self.vectorstore.delete(ids=delete_ids)
        
        conn.close()
        return len(delete_ids)
    
    def extract_and_store(self, conversation: str, user_id: str = "default") -> List[Memory]:
        """从对话中自动提取值得记忆的信息"""
        extract_prompt = f"""从以下对话中提取值得长期记忆的信息：

对话内容：
{conversation}

请提取：
1. 用户的个人信息（姓名、职业、偏好等）
2. 重要事实和知识点
3. 用户表达的明确需求或目标
4. 系统完成的重要任务

对每条记忆，给出：
- content：记忆内容（简洁的一句话）
- category：类别（personal/fact/preference/task）
- importance：重要性（0.1-1.0）
- tags：标签列表

以 JSON 列表返回，如：
[{{"content": "...", "category": "personal", "importance": 0.8, "tags": ["user_info"]}}]"""
        
        response = self.llm.invoke(extract_prompt)
        
        memories = []
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            extracted = json.loads(content.strip())
            
            for item in extracted:
                memory = self.store(
                    content=item["content"],
                    category=item.get("category", "general"),
                    importance=item.get("importance", 0.5),
                    tags=item.get("tags", []) + [f"user:{user_id}"]
                )
                memories.append(memory)
        except Exception as e:
            print(f"提取记忆失败：{e}")
        
        return memories

# 完整使用示例
ltm = LongTermMemorySystem()

# 从对话中提取记忆
conversation = """
用户：我叫张三，是一名Python工程师，专注于数据处理
助手：很高兴认识你张三！
用户：我最近在研究大模型，特别关注RAG技术
助手：RAG是个很有价值的方向
用户：我喜欢简洁的代码风格，不喜欢过度工程化
"""

memories = ltm.extract_and_store(conversation, user_id="zhangsan")
print(f"提取了 {len(memories)} 条记忆")
for m in memories:
    print(f"  [{m.category}] {m.content}")

# 检索相关记忆
print("\n检索'用户技术背景'：")
results = ltm.retrieve_by_query("用户技术背景")
for m in results:
    print(f"  - {m.content}")

# 遗忘旧记忆
deleted = ltm.forget_old_memories(days_threshold=90, importance_threshold=0.2)
print(f"\n清理了 {deleted} 条旧记忆")
```

## 记忆重要性动态调整

```python
def update_importance_based_on_usage(memory_system: LongTermMemorySystem):
    """根据访问频率动态调整记忆重要性"""
    conn = sqlite3.connect(memory_system.db_path)
    
    # 高频访问的记忆提升重要性
    conn.execute("""
        UPDATE memories 
        SET importance = MIN(1.0, importance + 0.1)
        WHERE access_count > 10
    """)
    
    # 长期未访问的记忆降低重要性
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    conn.execute("""
        UPDATE memories 
        SET importance = MAX(0.1, importance - 0.05)
        WHERE last_accessed < ? OR last_accessed = ''
    """, (cutoff,))
    
    conn.commit()
    conn.close()
```

## 踩坑经验

### 坑1：向量数据库和 SQLite 不同步

**问题**：从 SQLite 删除了记忆，但向量数据库里还有，导致搜索返回"幽灵记忆"。  
**解法**：始终保持两者同步，删除时同时从两个地方删除。

### 坑2：记忆提取不准确

**问题**：LLM 提取记忆时，把临时信息（"今天的天气"）当成重要记忆存储。  
**解法**：在提取 prompt 中明确区分临时信息和值得长期记忆的信息，并设置重要性过滤阈值（如只存 importance >= 0.4 的记忆）。

### 坑3：相同信息重复存储

**问题**：多次对话后，"用户是Python工程师"这条记忆被存了10次。  
**解法**：基于内容哈希做去重（本文的 `_generate_id` 方法），相同内容只更新不新增。

---

*W6D4 · 长期记忆系统 | Agent + Claw 系列*
