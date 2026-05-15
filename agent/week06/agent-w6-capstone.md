---
layout: default
title: "W6 Capstone · 有记忆的个人助手 Agent"
---

# Capstone：构建有长期记忆的个人助手

> **Week 6 · Capstone** | 难度：⭐⭐⭐⭐⭐

---

## 项目目标

构建一个真正"认识你"的个人助手 Agent：
- 记住用户的基本信息和偏好
- 跨会话保持上下文连续性
- 能回忆过去的对话和事件
- 学习用户习惯，越用越懂你

## 完整代码

```python
import json
import sqlite3
from datetime import datetime
from typing import List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    name: Optional[str] = None
    profession: Optional[str] = None
    preferences: List[str] = Field(default_factory=list)
    expertise_areas: List[str] = Field(default_factory=list)
    communication_style: Optional[str] = None

class PersonalAssistantAgent:
    """有长期记忆的个人助手"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 初始化记忆系统
        self.vectorstore = Chroma(
            collection_name=f"memory_{user_id}",
            embedding_function=self.embeddings,
            persist_directory=f"./assistant_memory_{user_id}"
        )
        
        # SQLite 存储用户档案
        self.db_path = f"./assistant_{user_id}.db"
        self._init_db()
        
        # 当前会话记忆
        self.session_history: List[dict] = []
        self.user_profile: UserProfile = self._load_profile()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def _load_profile(self) -> UserProfile:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT key, value FROM user_profile").fetchall()
        conn.close()
        
        profile_data = {row[0]: json.loads(row[1]) for row in rows}
        return UserProfile(**profile_data) if profile_data else UserProfile()
    
    def _save_profile(self):
        conn = sqlite3.connect(self.db_path)
        profile_dict = self.user_profile.model_dump()
        
        for key, value in profile_dict.items():
            if value is not None and value != [] and value != "":
                conn.execute("""
                    INSERT OR REPLACE INTO user_profile (key, value, updated_at)
                    VALUES (?, ?, ?)
                """, (key, json.dumps(value, ensure_ascii=False), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def _update_profile_from_conversation(self, user_message: str):
        """从对话中自动更新用户档案"""
        update_prompt = f"""根据以下用户消息，提取用户信息更新（如果有的话）：

用户消息：{user_message}
当前档案：{self.user_profile.model_dump_json()}

如果消息包含新的用户信息，返回更新字段（JSON格式）；
如果没有新信息，返回空 JSON {{}}。

可更新字段：name, profession, preferences（列表追加）, 
expertise_areas（列表追加）, communication_style"""
        
        response = self.llm.invoke(update_prompt)
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        
        try:
            updates = json.loads(content.strip())
            if updates:
                for key, value in updates.items():
                    if hasattr(self.user_profile, key) and value:
                        current = getattr(self.user_profile, key)
                        if isinstance(current, list):
                            # 列表类型：追加不重复的
                            if isinstance(value, list):
                                new_items = [v for v in value if v not in current]
                                setattr(self.user_profile, key, current + new_items)
                        elif value:
                            setattr(self.user_profile, key, value)
                
                self._save_profile()
        except:
            pass
    
    def _retrieve_relevant_memories(self, query: str, k: int = 5) -> str:
        """检索相关记忆"""
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            if results:
                memories = "\n".join([f"- {doc.page_content}" for doc in results])
                return f"相关历史记忆：\n{memories}"
        except:
            pass
        return ""
    
    def _store_conversation_memory(self, user_msg: str, assistant_msg: str):
        """存储对话到长期记忆"""
        # 创建记忆摘要
        summary_prompt = f"""将以下对话提炼成一条值得记忆的要点（如果有的话）：

用户：{user_msg}
助手：{assistant_msg}

如果这段对话包含值得记忆的信息（用户信息、重要事实、决定等），
给出简洁的一句话记忆；否则返回"无需记录"。"""
        
        response = self.llm.invoke(summary_prompt)
        memory_text = response.content.strip()
        
        if "无需记录" not in memory_text and len(memory_text) > 10:
            self.vectorstore.add_texts(
                texts=[memory_text],
                metadatas=[{
                    "user_id": self.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "type": "conversation_summary"
                }]
            )
    
    def _build_system_prompt(self) -> str:
        """构建带用户档案的 system prompt"""
        profile_parts = []
        
        if self.user_profile.name:
            profile_parts.append(f"用户姓名：{self.user_profile.name}")
        if self.user_profile.profession:
            profile_parts.append(f"职业：{self.user_profile.profession}")
        if self.user_profile.preferences:
            profile_parts.append(f"偏好：{', '.join(self.user_profile.preferences)}")
        if self.user_profile.expertise_areas:
            profile_parts.append(f"擅长领域：{', '.join(self.user_profile.expertise_areas)}")
        if self.user_profile.communication_style:
            profile_parts.append(f"沟通风格：{self.user_profile.communication_style}")
        
        profile_text = "\n".join(profile_parts) if profile_parts else "暂无用户信息"
        
        return f"""你是一个有记忆的个人助手，能记住用户的信息和历史对话。

## 用户档案
{profile_text}

## 行为准则
1. 使用用户的名字称呼他们（如果知道的话）
2. 根据用户的专业水平调整解释深度
3. 记住用户的偏好，提供个性化建议
4. 对历史对话中提到的事项有所了解
5. 保持友好、专业的沟通风格

今天是 {datetime.now().strftime("%Y年%m月%d日")}"""
    
    def chat(self, user_message: str) -> str:
        """进行一次对话"""
        # 更新用户档案
        self._update_profile_from_conversation(user_message)
        
        # 检索相关记忆
        memories = self._retrieve_relevant_memories(user_message)
        
        # 构建消息
        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        
        # 添加相关记忆作为上下文
        if memories:
            messages.append({
                "role": "system",
                "content": memories
            })
        
        # 添加当前会话历史（最近5轮）
        for turn in self.session_history[-5:]:
            messages.append({"role": "user", "content": turn["human"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        
        # 添加当前输入
        messages.append({"role": "user", "content": user_message})
        
        # 获取回复
        response = self.llm.invoke(messages)
        assistant_message = response.content
        
        # 更新会话历史
        self.session_history.append({
            "human": user_message,
            "assistant": assistant_message
        })
        
        # 异步存储记忆（生产中用 asyncio）
        self._store_conversation_memory(user_message, assistant_message)
        
        return assistant_message

# ── 测试 ──

def demo_personal_assistant():
    print("创建个人助手...\n")
    assistant = PersonalAssistantAgent("demo_user")
    
    # 模拟多轮对话
    conversations = [
        "你好！我叫王小明，是一名数据科学家",
        "我最近在研究时间序列预测，有什么好的方法推荐吗？",
        "我比较熟悉Python，不太懂R，你的建议最好用Python实现",
        "好的，我明白了。你还记得我是做什么的吗？",
        "我决定用 Prophet 试试，你能给我一个快速上手的代码示例吗？",
    ]
    
    for user_msg in conversations:
        print(f"👤 用户：{user_msg}")
        response = assistant.chat(user_msg)
        print(f"🤖 助手：{response[:200]}...")
        print()
    
    print(f"用户档案：{assistant.user_profile.model_dump()}")

demo_personal_assistant()
```

## 测试记忆持久性

```python
def test_memory_persistence():
    """测试跨会话记忆"""
    user_id = "test_persistence"
    
    # 第一次会话
    print("=== 第一次会话 ===")
    assistant1 = PersonalAssistantAgent(user_id)
    assistant1.chat("我叫张伟，是一名后端工程师，喜欢用Go语言")
    assistant1.chat("我正在设计一个高并发的消息队列系统")
    print("第一次会话结束，记忆已保存\n")
    
    # 第二次会话（新实例模拟程序重启）
    print("=== 第二次会话（新实例）===")
    assistant2 = PersonalAssistantAgent(user_id)
    response = assistant2.chat("你还记得我是谁吗？")
    print(f"助手回答：{response[:300]}")
    print(f"\n从档案中恢复的用户信息：{assistant2.user_profile.model_dump()}")

test_memory_persistence()
```

## 本周回顾

| 技术 | 在本项目中的应用 |
|------|----------------|
| 短期记忆 | session_history（最近5轮） |
| 情景记忆 | 对话摘要存入向量数据库 |
| 语义记忆 | 向量搜索检索相关历史 |
| 长期持久化 | SQLite 存储用户档案 |
| 记忆提取 | LLM 自动从对话中提取关键信息 |
| 遗忘策略 | 只存储"值得记忆"的内容 |

---

*W6 Capstone · 有记忆的个人助手 | Agent + Claw 系列*
