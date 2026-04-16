# D7 · Week1 综合实战——构建你的第一个完整 Agent

> **Week 1 主题**：什么是 Agent——定义 / ReAct / 规划 / 记忆 / 工具调用  
> **本日主题**：综合实战——整合所有组件，构建完整 Agent 项目

---

## 🎯 学习目标

1. 综合运用本周所有知识：ReAct + 规划 + 记忆 + 工具调用
2. 独立完成一个有实际价值的 Agent 项目
3. 学会 Agent 项目的测试和调试方法
4. 复盘 Week 1，规划 Week 2 学习方向

---

## 📚 Week 1 知识复盘

### 七天知识地图

```
D1 Agent 定义
├── 感知 / 规划 / 行动 / 记忆
└── 与普通 LLM 的本质区别

D2 ReAct 框架
├── Thought → Action → Observation
└── 解析器 + 执行循环

D3 规划能力
├── CoT / ToT
└── Plan-and-Execute 架构

D4 记忆系统
├── 短期（Context Window）
├── 长期（向量数据库）
└── 操作性 / 程序性记忆

D5 工具调用
├── Function Calling 协议
├── 工具注册 / 执行引擎
└── 安全性设计

D6 多 Agent
├── 主从 / 对等 / 分层拓扑
├── 通信机制
└── Claw 的多 Agent 实现

D7 综合实战 ← 你在这里
```

### 核心能力检查清单

完成以下问题的回答，检验自己的掌握程度：

- [ ] 能用 100 字解释 AI Agent 是什么？
- [ ] 能画出 ReAct 的 Thought-Action-Observation 循环？
- [ ] 知道 Plan-and-Execute 和纯 ReAct 各适合什么场景？
- [ ] 能说出 Agent 的四种记忆类型及其载体？
- [ ] 知道 Function Calling 的完整消息格式？
- [ ] 能说出 Claw 多 Agent 系统的三层结构？

---

## 🏗️ 本日项目：个人学习助手 Agent

### 项目概述

构建一个**个人学习助手 Agent**，能够：
1. 记住用户的学习进度和偏好（长期记忆）
2. 回答问题时能搜索和计算（工具调用）
3. 为用户制定学习计划（规划能力）
4. 使用 ReAct 框架推理（推理能力）

### 项目架构

```
PersonalLearningAgent
├── memory/
│   ├── VectorMemory（ChromaDB 长期记忆）
│   └── ConversationBuffer（短期对话历史）
├── tools/
│   ├── calculator（数学计算）
│   ├── web_search（模拟搜索）
│   ├── create_study_plan（生成学习计划）
│   └── save_note（保存笔记）
├── planner/
│   └── LLMPlanner（任务分解）
└── react_executor/
    └── ReActLoop（执行循环）
```

---

## 🔧 动手实战

### 完整项目代码

```python
# 创建文件: learning_agent/main.py
# 目录结构:
#   learning_agent/
#     main.py
#     memory.py
#     tools.py
#     agent.py
#     .env
# 
# uv init learning_agent
# cd learning_agent
# uv add langchain langchain-openai chromadb python-dotenv
# uv run python main.py
```

#### `tools.py`

```python
# learning_agent/tools.py
import math
import json
from datetime import datetime
from pathlib import Path
from langchain.tools import tool

NOTES_DIR = Path("./notes")
NOTES_DIR.mkdir(exist_ok=True)

@tool
def calculator(expression: str) -> str:
    """执行数学计算。支持加减乘除和 math 模块函数（sqrt, log, pow 等）。
    示例: '2 ** 10', 'math.sqrt(144)', '(3 + 5) * 12'"""
    safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith('_')}
    safe_env['abs'] = abs
    try:
        result = eval(expression, {"__builtins__": {}}, safe_env)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

@tool  
def web_search(query: str) -> str:
    """搜索学习资源和技术信息（模拟搜索，返回示例结果）。
    适用于查找技术概念、教程、文档等。"""
    # 模拟搜索结果（实际项目中替换为真实搜索API）
    mock_results = {
        "langchain": "LangChain 是一个用于构建 LLM 应用的框架，提供 Agent、Chain、Tool 等组件。官网: langchain.com",
        "react agent": "ReAct Agent 使用 Reasoning+Acting 模式，通过 Thought-Action-Observation 循环解决问题。论文: arxiv.org/abs/2210.03629",
        "chromadb": "ChromaDB 是一个开源向量数据库，支持嵌入式使用，适合 AI 应用。安装: pip install chromadb",
        "default": f"搜索 '{query}' 的结果：这是一个关于 {query} 的技术概念，建议查阅官方文档获取详细信息。"
    }
    
    query_lower = query.lower()
    for key, result in mock_results.items():
        if key in query_lower:
            return result
    return mock_results["default"]

@tool
def create_study_plan(topic: str, duration_days: int) -> str:
    """为指定主题创建学习计划。
    参数: topic（学习主题）, duration_days（计划天数，整数）"""
    try:
        duration_days = int(duration_days)
    except:
        duration_days = 7
    
    plan = {
        "topic": topic,
        "duration": f"{duration_days}天",
        "daily_plan": []
    }
    
    phases = ["基础概念", "核心原理", "动手实践", "进阶深入", "项目应用", "复盘总结", "延伸学习"]
    
    for day in range(1, duration_days + 1):
        phase_idx = min((day - 1) * len(phases) // duration_days, len(phases) - 1)
        plan["daily_plan"].append({
            "day": day,
            "focus": f"{phases[phase_idx]}：{topic}的{phases[phase_idx]}部分",
            "duration": "2-3小时"
        })
    
    return json.dumps(plan, ensure_ascii=False, indent=2)

@tool
def save_note(title: str, content: str) -> str:
    """保存学习笔记到本地文件。
    参数: title（笔记标题）, content（笔记内容）"""
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_title[:30]}.md"
    filepath = NOTES_DIR / filename
    
    note_content = f"# {title}\n\n*保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n{content}"
    filepath.write_text(note_content, encoding='utf-8')
    
    return f"笔记已保存: {filepath}"

ALL_TOOLS = [calculator, web_search, create_study_plan, save_note]
```

#### `memory.py`

```python
# learning_agent/memory.py
import chromadb
from chromadb.utils import embedding_functions
import time
from typing import Optional

class AgentMemory:
    """Agent 记忆管理：短期 + 长期记忆"""
    
    def __init__(self, persist_path: str = "./chroma_db"):
        # 长期记忆：向量数据库
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name="agent_memories",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        
        # 短期记忆：对话历史
        self.conversation: list[dict] = []
        self.max_conversation = 20
        self._counter = self.collection.count()
    
    def remember(self, content: str, category: str = "general"):
        """存储长期记忆"""
        self._counter += 1
        self.collection.add(
            documents=[content],
            ids=[f"mem_{self._counter}_{int(time.time())}"],
            metadatas=[{"category": category, "timestamp": time.time()}]
        )
    
    def recall(self, query: str, n_results: int = 3) -> list[str]:
        """检索相关长期记忆"""
        if self.collection.count() == 0:
            return []
        n = min(n_results, self.collection.count())
        results = self.collection.query(
            query_texts=[query],
            n_results=n
        )
        return results["documents"][0] if results["documents"][0] else []
    
    def add_conversation(self, role: str, content: str):
        """添加对话记录（短期记忆）"""
        self.conversation.append({"role": role, "content": content})
        # 超出限制时截断
        if len(self.conversation) > self.max_conversation:
            self.conversation = self.conversation[-self.max_conversation:]
    
    def get_recent_conversation(self, n: int = 6) -> list[dict]:
        """获取最近的对话"""
        return self.conversation[-n:]
    
    def get_memory_stats(self) -> dict:
        return {
            "long_term_count": self.collection.count(),
            "short_term_count": len(self.conversation)
        }
```

#### `agent.py`

```python
# learning_agent/agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.messages import SystemMessage
from memory import AgentMemory
from tools import ALL_TOOLS

class PersonalLearningAgent:
    """个人学习助手 Agent（ReAct + 记忆 + 工具）"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.memory = AgentMemory()
        
        # 创建 ReAct Agent
        prompt = hub.pull("hwchase17/react")
        react_agent = create_react_agent(self.llm, ALL_TOOLS, prompt)
        self.executor = AgentExecutor(
            agent=react_agent,
            tools=ALL_TOOLS,
            verbose=True,
            max_iterations=8,
            handle_parsing_errors=True
        )
    
    def _build_context(self, user_input: str) -> str:
        """构建包含记忆的上下文"""
        # 检索相关长期记忆
        relevant_memories = self.memory.recall(user_input)
        
        context_parts = ["你是一个个人学习助手，帮助用户学习 AI 和编程技术。"]
        
        if relevant_memories:
            context_parts.append("\n【关于用户的记忆】")
            for m in relevant_memories:
                context_parts.append(f"- {m}")
        
        recent_conv = self.memory.get_recent_conversation()
        if recent_conv:
            context_parts.append("\n【近期对话】")
            for msg in recent_conv:
                role = "用户" if msg["role"] == "user" else "助手"
                context_parts.append(f"{role}: {msg['content'][:100]}...")
        
        context_parts.append(f"\n用户当前问题: {user_input}")
        
        return "\n".join(context_parts)
    
    def chat(self, user_input: str) -> str:
        """处理用户输入"""
        # 检查是否需要记录
        if "记住" in user_input or "记录" in user_input:
            self.memory.remember(user_input, category="user_preference")
        
        # 构建上下文
        full_input = self._build_context(user_input)
        
        # 执行 Agent
        try:
            result = self.executor.invoke({"input": full_input})
            response = result["output"]
        except Exception as e:
            response = f"执行过程中遇到问题: {e}"
        
        # 更新对话记忆
        self.memory.add_conversation("user", user_input)
        self.memory.add_conversation("assistant", response)
        
        # 自动提取重要信息存入长期记忆
        if len(user_input) > 50:
            summary = f"用户询问/讨论了: {user_input[:100]}"
            self.memory.remember(summary, category="conversation")
        
        return response
    
    def get_status(self) -> str:
        stats = self.memory.get_memory_stats()
        return f"记忆状态: 长期记忆 {stats['long_term_count']} 条, 短期对话 {stats['short_term_count']} 轮"
```

#### `main.py`

```python
# learning_agent/main.py
from dotenv import load_dotenv
from agent import PersonalLearningAgent

def main():
    load_dotenv()
    agent = PersonalLearningAgent()
    
    print("🤖 个人学习助手已启动")
    print("   可用命令: quit（退出）| status（查看状态）| clear（清空对话）")
    print("   试试问: '帮我搜一下 LangChain 的资料' 或 '给我制定一个学习Python的7天计划'\n")
    
    while True:
        try:
            user_input = input("你: ").strip()
        except EOFError:
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("再见！你的学习记录已保存。")
            break
        elif user_input.lower() == "status":
            print(agent.get_status())
            continue
        elif user_input.lower() == "clear":
            agent.memory.conversation.clear()
            print("对话历史已清空，长期记忆保留。")
            continue
        
        print("AI: ", end="", flush=True)
        response = agent.chat(user_input)
        print(response)
        print()

if __name__ == "__main__":
    main()
```

---

## 🧪 测试用例

运行 Agent 后，按顺序测试以下场景：

```bash
# 1. 工具调用测试
你: 计算 2的10次方加上100的平方根

# 2. 搜索工具测试  
你: 帮我搜索一下 ReAct Agent 的相关资料

# 3. 规划工具测试
你: 帮我制定一个学习 LangChain 的5天计划

# 4. 记忆测试（先告知信息）
你: 记住我叫丹丹，是一名后端工程师，正在学习 AI Agent

# 5. 记忆检索测试（新的问题，应该能联系上之前的信息）
你: 以我的背景，你觉得我应该从哪个 Agent 框架入手？

# 6. 笔记保存测试
你: 把今天学到的 ReAct 框架知识保存成笔记
```

---

## 📊 Week 1 项目复盘

### 完成以下复盘模板

```markdown
## Week 1 复盘（填写你自己的）

### 完成情况
- [x] D1: Agent 定义——理解了 Agent 的四大组件
- [ ] D2: ReAct——（填写你的掌握情况）
- [ ] D3: 规划——
- [ ] D4: 记忆——
- [ ] D5: 工具调用——
- [ ] D6: 多 Agent——
- [ ] D7: 综合实战——

### 最大收获
（填写本周最重要的3个收获）

### 最大困惑
（还没完全理解的概念）

### Week 2 重点
（下周想深入的方向）
```

### Week 1 → Week 2 的连接

| Week 1 概念 | Week 2 深入方向 |
|------------|----------------|
| ReAct 基础 | 高级推理：Self-Reflection / Self-Consistency |
| 工具调用 | RAG：检索增强生成 |
| 记忆系统 | 知识图谱 + 向量数据库高级用法 |
| 多 Agent | CrewAI / AutoGen 框架深入 |
| Claw 实战 | 构建完整 Claw 工作流 |

---

## 🦞 Claw 终极实战：回顾本周所有 Claw 使用

本周通过参与 Claw 任务，你已经实践了：

| 组件 | Claw 中的体现 |
|------|-------------|
| **Agent 定义** | Worker Agent 的职责定位 |
| **ReAct** | 任务执行日志（Thought=分析，Action=curl，Observation=返回值） |
| **规划** | 项目龙虾拆解任务，stepId 管理 |
| **记忆** | MEMORY.md / 任务备忘（/md接口） |
| **工具调用** | curl 调用平台 API |
| **多 Agent** | Worker 并行执行，依赖管理 |

**最后一个实战任务**：
1. 阅读本任务的完整执行日志
2. 用今天学的框架，完整描述本次任务的执行过程
3. 把这个复盘写入你工作区的 `memory/` 目录

---

## 📝 Week 1 总结

七天从零到一，你已经建立了 AI Agent 的完整知识框架：

```
什么是 Agent（D1）
    ↓
怎么推理：ReAct（D2）
    ↓
怎么规划：Plan-and-Execute（D3）
    ↓
怎么记忆：短期+长期（D4）
    ↓
怎么行动：工具调用（D5）
    ↓
怎么协作：多 Agent（D6）
    ↓
整合实战（D7）
```

**Week 2 预告**：深入 RAG（检索增强生成）——让 Agent 拥有海量私有知识库，实现真正的知识问答系统。

---

> 💡 **Week 1 最终思考题**：如果让你从零开始设计 ALLIN Claw 这个系统，你会做哪些不同的设计决策？有什么可以改进的地方？
