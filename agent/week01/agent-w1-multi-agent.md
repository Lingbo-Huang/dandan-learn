# D6 · 多 Agent 协作——当一个 Agent 不够用时

> **Week 1 主题**：什么是 Agent——定义 / ReAct / 规划 / 记忆 / 工具调用  
> **本日主题**：多 Agent 系统（Multi-Agent Systems）

---

## 🎯 学习目标

1. 理解多 Agent 系统的必要性和适用场景
2. 掌握主流多 Agent 拓扑结构（主从、对等、分层）
3. 了解 Agent 间通信机制
4. 能用 Claw 框架理解并设计多 Agent 流程

---

## 📚 核心知识点

### 1. 为什么需要多 Agent？

单 Agent 的瓶颈：
- **上下文限制**：复杂任务超出单次 Context Window
- **专业化**：不同任务需要不同的专业能力和提示词
- **并行效率**：串行执行浪费时间
- **可靠性**：单点故障风险

多 Agent 带来：
```
专业化分工 + 并行执行 + 相互校验 + 容错降级
```

### 2. 三种主流拓扑结构

#### 🎯 主从架构（Orchestrator-Worker）
```
用户
 ↓
Orchestrator（协调者/规划器）
  ├── Worker A（专门处理数据收集）
  ├── Worker B（专门处理分析）
  └── Worker C（专门处理报告生成）
```

- 典型实现：AutoGen、Claw Project 模式
- 优点：逻辑清晰，协调集中
- 缺点：Orchestrator 成为瓶颈

#### 🤝 对等架构（Peer-to-Peer）
```
Agent A ←→ Agent B
   ↕             ↕
Agent C ←→ Agent D
```

- 典型实现：多 Agent 辩论、角色扮演
- 优点：去中心化，灵活
- 缺点：协调成本高，容易死锁

#### 🏗️ 分层架构（Hierarchical）
```
Meta-Agent（最高层）
├── Manager A
│   ├── Worker A1
│   └── Worker A2
└── Manager B
    ├── Worker B1
    └── Worker B2
```

- 典型实现：企业级复杂系统
- 优点：可处理超大规模任务
- 缺点：通信开销大，调试困难

### 3. Agent 间通信机制

**消息传递**（最常用）：
```json
{
  "from": "orchestrator",
  "to": "worker_a",
  "type": "task_assignment",
  "content": {
    "task_id": "001",
    "title": "收集竞品数据",
    "context": {...},
    "deadline": "2024-01-01T10:00:00"
  }
}
```

**共享状态**（黑板模式）：
```
所有 Agent 读写同一个共享存储
Agent A 写入 → Agent B 读取 → Agent C 处理
```

**流式传输**：
```
Agent A 生成内容流 → Agent B 实时处理
（适合长文本生成场景）
```

### 4. 任务分配策略

**静态分配**：规划时决定谁做什么（Claw 的方式）
```
Project Session → 创建 Task 并分配给指定 Worker
```

**动态分配**：运行时根据 Agent 状态和能力分配
```
Orchestrator 监控所有 Worker 的负载，动态派发任务
```

**竞争选择**：多个 Agent 竞争同一任务
```
Task 发布 → 多个 Agent 竞标 → 能力最强的获得执行权
```

### 5. 多 Agent 的关键挑战

**一致性问题**：
- Agent A 和 Agent B 可能基于不同信息做决策
- 需要共享状态同步机制

**依赖管理**：
- Task B 依赖 Task A 的结果
- 需要明确的依赖声明和等待机制

**错误传播**：
- 上游 Agent 失败 → 下游 Agent 应如何处理？
- 需要 error 状态感知和降级策略

**循环检测**：
- Agent A 调用 Agent B，Agent B 又调用 Agent A
- 需要调用图的循环检测

### 6. ALLIN Claw 的多 Agent 架构

Claw 实现了一套完整的多 Agent 系统：

```
用户（群聊）
    ↓
主会话 Agent（意图理解、创建 Project）
    ↓
项目会话龙虾（Orchestrator）
  ├── 拆解任务（创建 Task 列表）
  ├── 分配 Worker（本 Agent 就是一个 Worker）
  ├── 管理依赖（stepId + dependencies）
  └── 汇总结果（收集 summary、向用户汇报）
    ↓
Worker Agents（并行执行各自任务）
```

**依赖管理实现**：
- `dependencies: [stepId1, stepId2]` 声明前置依赖
- 平台保证依赖完成后才激活当前任务（`pending_execution` → `running`）
- Worker 通过 `/dependencies` 接口获取上游产出

---

## 💡 示例/推导

### 示例：多 Agent 写作系统

**目标**：生成一篇完整的技术文章

```
[Orchestrator 拆解任务]
Task 1: 研究员 Agent → 搜集资料和关键数据
Task 2: 大纲 Agent → 基于资料生成文章大纲（依赖 Task1）
Task 3A: 写作 Agent A → 撰写第1-3节（依赖 Task2）
Task 3B: 写作 Agent B → 撰写第4-6节（依赖 Task2，与3A并行）
Task 4: 编辑 Agent → 整合、润色（依赖 3A, 3B）
Task 5: 审核 Agent → 事实核查（依赖 Task4）

[执行时间对比]
单 Agent 串行: ~20分钟
多 Agent 并行: ~10分钟（3A和3B并行执行）
```

### 推导：为什么 Claw 用 stepId 而不是 taskId 来管理依赖？

```python
# 用 taskId 管理依赖的问题
Task(id=405, depends_on=[404, 403])
# 问题：如果任务被重新创建，taskId 会变，依赖关系失效

# Claw 的解法：用 stepId（相对序号）
Task(stepId=3, depends_on=[1, 2])
# 优势：stepId 在 Project 内是稳定的逻辑序号，与实际 taskId 解耦
```

---

## 🔧 动手练习

### 练习 1：设计多 Agent 方案（必做）

为以下场景设计多 Agent 系统：

**场景**："自动生成月度技术周报，包含：行业新闻摘要、GitHub 热门项目、团队工作进展、下月计划"

要求：
1. 画出 Agent 拓扑图
2. 列出每个 Agent 的职责和所需工具
3. 标注依赖关系
4. 估算并行执行节约的时间

### 练习 2：模拟多 Agent 对话（必做）

```python
# 创建文件: 11_multi_agent_debate.py
# uv run python 11_multi_agent_debate.py

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

def create_agent(role: str, persona: str):
    """创建具有特定角色的 Agent"""
    def agent(discussion_history: list, topic: str) -> str:
        messages = [
            SystemMessage(content=f"""你扮演{role}。
{persona}
请基于讨论历史，从你的角度发表观点（100字以内）。"""),
            HumanMessage(content=f"讨论主题: {topic}\n\n讨论历史:\n" + 
                        "\n".join(discussion_history[-4:]))  # 只看最近4条
        ]
        return llm.invoke(messages).content
    
    return agent, role

# 创建三个具有不同立场的 Agent
optimist, opt_name = create_agent(
    "AI乐观主义者",
    "你相信 AI 将极大推动人类进步，对 AI 的发展持积极态度。"
)

pessimist, pes_name = create_agent(
    "AI风险研究者",
    "你关注 AI 的潜在风险和伦理问题，对 AI 的快速发展持谨慎态度。"
)

moderator, mod_name = create_agent(
    "中立的技术分析师",
    "你从技术和数据角度分析问题，不偏不倚，寻求平衡观点。"
)

def multi_agent_debate(topic: str, rounds: int = 3):
    """运行多 Agent 辩论"""
    print(f"=== 多 Agent 辩论 ===")
    print(f"主题: {topic}\n")
    
    history = []
    agents = [(optimist, opt_name), (pessimist, pes_name), (moderator, mod_name)]
    
    for round_num in range(1, rounds + 1):
        print(f"--- 第 {round_num} 轮 ---")
        for agent_fn, agent_name in agents:
            opinion = agent_fn(history, topic)
            history.append(f"{agent_name}: {opinion}")
            print(f"\n[{agent_name}]\n{opinion}")
        print()
    
    print("=== 辩论结束 ===")

multi_agent_debate("AGI 会在2030年前实现吗？", rounds=2)
```

### 练习 3：主从 Agent 系统（进阶）

```python
# 创建文件: 12_orchestrator_worker.py
# uv run python 12_orchestrator_worker.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===== Orchestrator Agent =====
def orchestrator(goal: str) -> list[dict]:
    """将目标拆解为子任务列表"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是任务协调者。将用户目标拆解为2-4个并行子任务。
        
输出 JSON：
{{"tasks": [{{"id": 1, "worker": "researcher|writer|analyst", "instruction": "..."}}]}}"""),
        ("user", "{goal}")
    ])
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"goal": goal})
    return result["tasks"]

# ===== Worker Agents =====
def researcher_agent(instruction: str) -> str:
    """研究员 Worker"""
    prompt = f"作为研究员，请执行以下任务（模拟搜索结果）：{instruction}"
    return llm.invoke(prompt).content

def writer_agent(instruction: str) -> str:
    """写作 Worker"""
    prompt = f"作为技术写作专家，请执行以下任务：{instruction}"
    return llm.invoke(prompt).content

def analyst_agent(instruction: str) -> str:
    """分析员 Worker"""
    prompt = f"作为数据分析师，请执行以下任务：{instruction}"
    return llm.invoke(prompt).content

WORKERS = {
    "researcher": researcher_agent,
    "writer": writer_agent,
    "analyst": analyst_agent
}

def run_multi_agent_system(goal: str):
    """运行主从多 Agent 系统"""
    print(f"🎯 目标: {goal}\n")
    
    # Step 1: Orchestrator 规划
    print("📋 Orchestrator 正在拆解任务...")
    tasks = orchestrator(goal)
    print(f"拆解为 {len(tasks)} 个并行子任务\n")
    
    # Step 2: 并行执行 Worker 任务（模拟并行，实际串行）
    results = {}
    for task in tasks:
        worker_type = task["worker"]
        worker_fn = WORKERS.get(worker_type, researcher_agent)
        
        print(f"⚡ Worker [{worker_type}] 执行: {task['instruction'][:50]}...")
        result = worker_fn(task["instruction"])
        results[task["id"]] = result
        print(f"  ✅ 完成 ({len(result)} 字)\n")
    
    # Step 3: Orchestrator 汇总
    summary_prompt = f"""目标: {goal}
    
各 Worker 产出：
{json.dumps(results, ensure_ascii=False, indent=2)}

请综合以上产出，生成最终结论（200字以内）。"""
    
    final_result = llm.invoke(summary_prompt).content
    print("=" * 50)
    print("📊 最终综合结论:")
    print(final_result)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_multi_agent_system("分析 2024年最值得学习的3个 AI 技术方向")
```

### 🦞 Claw 实战：你就是一个 Worker！

今天这个任务（Agent线Week1文档生成）就是一个多 Agent 系统中的 Worker 执行案例：

1. **Orchestrator**：项目会话龙虾，拆解了"六条线Week1规划"为多个并行 Worker 任务
2. **你（Worker）**：专门负责 Agent 线的内容生成
3. **其他 Workers**：同时在生成其他线的内容（AI Infra 线等）

**观察练习**：
- 查看项目中有多少个并行任务？（通过 stepId 判断哪些可以并行）
- 本任务的 `dependencies` 是 `[]`——说明什么？
- 如果有依赖，你需要通过什么接口获取上游产出？

---

## 📝 小结

| 要点 | 核心内容 |
|------|---------|
| **必要性** | 上下文限制 + 专业化 + 并行 + 容错 |
| **拓扑** | 主从（最常用）/ 对等 / 分层 |
| **通信** | 消息传递 / 共享状态 / 流式 |
| **挑战** | 一致性 / 依赖管理 / 错误传播 / 循环检测 |
| **Claw实现** | 主会话 → 项目龙虾 → Worker，依赖图管理 |

**明天预告**：Week 1 综合实战——把这周学到的所有知识整合起来，构建一个完整的 Agent 项目。

---

> 💡 **今日思考题**：在 Claw 的多 Agent 系统中，如果一个 Worker（如本 Agent）将状态置为 `error`，项目龙虾应该如何处理？依赖本任务的下游任务会怎样？
