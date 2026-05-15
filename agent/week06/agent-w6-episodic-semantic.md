---
layout: default
title: "W6D5 · 情景记忆与语义记忆"
---

# 情景记忆与语义记忆：两种高级记忆形式

> **Week 6 · Day 5** | 难度：⭐⭐⭐⭐

---

## 情景记忆：时序事件的记录与检索

情景记忆专门存储"什么时候发生了什么事"，支持时序推理。

```python
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

class EpisodicMemorySystem:
    """情景记忆系统：记录事件序列，支持时序查询"""
    
    def __init__(self, chroma_dir: str = "./episodic_chroma"):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(
            collection_name="episodic_memory",
            embedding_function=self.embeddings,
            persist_directory=chroma_dir
        )
        self.episodes: List[Dict] = []
    
    def record_episode(self, 
                      event: str,
                      participants: List[str] = None,
                      location: str = None,
                      outcome: str = None,
                      emotion: str = None,
                      importance: float = 0.5) -> str:
        """记录一个情景事件"""
        episode_id = f"ep_{len(self.episodes):04d}"
        timestamp = datetime.now()
        
        episode = {
            "id": episode_id,
            "event": event,
            "timestamp": timestamp.isoformat(),
            "date": timestamp.strftime("%Y-%m-%d"),
            "time": timestamp.strftime("%H:%M"),
            "participants": participants or [],
            "location": location or "",
            "outcome": outcome or "",
            "emotion": emotion or "neutral",
            "importance": importance
        }
        
        self.episodes.append(episode)
        
        # 构建丰富的文本表示用于语义搜索
        text_repr = f"""事件：{event}
时间：{episode['date']} {episode['time']}
参与者：{', '.join(participants) if participants else '无'}
地点：{location or '未记录'}
结果：{outcome or '未记录'}
情感：{emotion or 'neutral'}"""
        
        self.vectorstore.add_texts(
            texts=[text_repr],
            metadatas=[{k: str(v) for k, v in episode.items()}],
            ids=[episode_id]
        )
        
        return episode_id
    
    def recall_by_time(self, 
                       start_date: str = None,
                       end_date: str = None,
                       limit: int = 20) -> List[Dict]:
        """按时间范围检索"""
        filtered = self.episodes
        
        if start_date:
            filtered = [e for e in filtered if e["date"] >= start_date]
        if end_date:
            filtered = [e for e in filtered if e["date"] <= end_date]
        
        return sorted(filtered, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def recall_by_query(self, query: str, k: int = 5, 
                        time_weight: float = 0.3) -> List[Dict]:
        """语义检索 + 时间加权"""
        results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        now = datetime.now()
        scored_episodes = []
        
        for doc, semantic_score in results:
            episode_id = doc.metadata.get("id")
            episode = next((e for e in self.episodes if e["id"] == episode_id), None)
            
            if episode:
                # 时间衰减：越近的事件权重越高
                ep_time = datetime.fromisoformat(episode["timestamp"])
                days_ago = (now - ep_time).days
                time_score = 1.0 / (1 + days_ago * 0.1)
                
                # 综合评分
                final_score = (1 - time_weight) * (1 - semantic_score) + time_weight * time_score
                scored_episodes.append((episode, final_score))
        
        scored_episodes.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored_episodes[:k]]
    
    def get_narrative(self, query: str = None, days: int = 7) -> str:
        """生成事件叙事（时间线故事）"""
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        if query:
            episodes = self.recall_by_query(query, k=10)
        else:
            episodes = self.recall_by_time(start_date=start_date, limit=10)
        
        if not episodes:
            return "没有找到相关事件记录"
        
        episodes_text = "\n".join([
            f"[{ep['date']} {ep['time']}] {ep['event']}"
            + (f" → {ep['outcome']}" if ep.get('outcome') else "")
            for ep in sorted(episodes, key=lambda x: x["timestamp"])
        ])
        
        narrative_prompt = f"""将以下事件记录整理成连贯的叙事：

{episodes_text}

请用自然语言描述这些事件的发展脉络（3-5句话）："""
        
        return self.llm.invoke(narrative_prompt).content

# 使用示例
episodic = EpisodicMemorySystem()

# 记录事件
episodic.record_episode(
    "与用户讨论了Python异步编程的最佳实践",
    participants=["Alice", "Bot"],
    outcome="用户理解了asyncio的核心概念",
    importance=0.8
)

episodic.record_episode(
    "完成了数据分析报告",
    outcome="报告已发送给管理团队",
    emotion="满意",
    importance=0.9
)

# 检索
print(episodic.get_narrative())
```

## 语义记忆：知识图谱与结构化知识

语义记忆存储概念和关系，支持推理。

```python
from typing import Set, Tuple

class SemanticMemory:
    """语义记忆：知识图谱形式"""
    
    def __init__(self):
        self.entities: Dict[str, Dict] = {}      # 实体
        self.relations: List[Tuple] = []          # 关系 (entity1, relation, entity2)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    def add_entity(self, name: str, entity_type: str, 
                  properties: Dict = None) -> None:
        """添加实体"""
        self.entities[name] = {
            "type": entity_type,
            "properties": properties or {},
            "relations": []
        }
    
    def add_relation(self, entity1: str, relation: str, entity2: str,
                    confidence: float = 1.0) -> None:
        """添加关系"""
        self.relations.append((entity1, relation, entity2, confidence))
        
        # 在实体中记录关系
        if entity1 in self.entities:
            self.entities[entity1]["relations"].append((relation, entity2))
        if entity2 in self.entities:
            self.entities[entity2]["relations"].append((f"被{relation}", entity1))
    
    def extract_knowledge(self, text: str) -> Dict:
        """从文本中提取知识（实体和关系）"""
        prompt = f"""从以下文本中提取实体和关系：

文本：{text}

请提取：
1. 实体（名字、类型、主要属性）
2. 实体间的关系

JSON格式：
{{
    "entities": [
        {{"name": "...", "type": "person/org/concept/tool", "properties": {{}}}}
    ],
    "relations": [
        {{"from": "...", "relation": "...", "to": "..."}}
    ]
}}"""
        
        response = self.llm.invoke(prompt)
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        
        try:
            knowledge = json.loads(content.strip())
            
            # 存储提取的知识
            for entity in knowledge.get("entities", []):
                self.add_entity(
                    entity["name"], entity["type"],
                    entity.get("properties", {})
                )
            
            for rel in knowledge.get("relations", []):
                if rel["from"] in self.entities and rel["to"] in self.entities:
                    self.add_relation(rel["from"], rel["relation"], rel["to"])
            
            return knowledge
        except:
            return {"entities": [], "relations": []}
    
    def query_relations(self, entity: str, relation_type: str = None) -> List[Tuple]:
        """查询实体的关系"""
        entity_data = self.entities.get(entity)
        if not entity_data:
            return []
        
        relations = entity_data["relations"]
        if relation_type:
            relations = [(r, e) for r, e in relations if r == relation_type]
        
        return relations
    
    def path_finding(self, start: str, end: str, max_depth: int = 3) -> List[List[str]]:
        """在知识图谱中寻找路径"""
        if start not in self.entities or end not in self.entities:
            return []
        
        # BFS 寻路
        queue = [[start]]
        visited = {start}
        paths = []
        
        while queue and len(paths) < 5:
            path = queue.pop(0)
            current = path[-1]
            
            if current == end:
                paths.append(path)
                continue
            
            if len(path) >= max_depth:
                continue
            
            for relation, neighbor in self.entities.get(current, {}).get("relations", []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        
        return paths
    
    def visualize(self) -> str:
        """简单的文本可视化"""
        lines = ["知识图谱：", ""]
        
        for entity, data in list(self.entities.items())[:10]:
            lines.append(f"[{data['type']}] {entity}")
            for relation, target in data["relations"][:3]:
                lines.append(f"  ──{relation}──> {target}")
        
        return "\n".join(lines)

# 使用示例
semantic = SemanticMemory()

text = """
LangChain 是由 Harrison Chase 创建的框架，用于构建 LLM 应用。
LangChain 支持 OpenAI、Anthropic 等多个 LLM 提供商。
OpenAI 的 GPT-4 是目前最强大的语言模型之一。
"""

knowledge = semantic.extract_knowledge(text)
print(semantic.visualize())

paths = semantic.path_finding("LangChain", "GPT-4")
print(f"\n从 LangChain 到 GPT-4 的路径：{paths}")
```

## 踩坑经验

### 坑1：情景记忆时序混乱

**问题**：多线程记录事件时，时间戳不准确，导致时序错误。  
**解法**：使用 UTC 时间戳，确保时区一致；多线程环境加锁。

### 坑2：知识图谱实体重复

**问题**："Python"、"python"、"Python语言" 被存为三个不同实体。  
**解法**：实体标准化层，统一大小写，合并同义实体。

---

*W6D5 · 情景记忆与语义记忆 | Agent + Claw 系列*
