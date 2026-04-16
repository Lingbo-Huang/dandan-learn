# D1 · 什么是 Agent？——定义、本质与全景

> **Week 1 主题**：什么是 Agent——定义 / ReAct / 规划 / 记忆 / 工具调用  
> **本日主题**：Agent 的定义与本质

---

## 🎯 学习目标

1. 能用自己的语言清晰解释"什么是 AI Agent"
2. 理解 Agent 与普通 LLM 调用的本质区别
3. 掌握 Agent 的四大核心组件：感知、规划、行动、记忆
4. 了解当前主流 Agent 框架的全景图

---

## 📚 核心知识点

### 1. Agent 的定义

**AI Agent**（智能体）是一个能够**感知环境、做出决策、执行行动**以实现特定目标的自主系统。

> 简单说：LLM 只会"回答"，Agent 会"行动"。

```
环境 → 感知（Perception）
     → 推理与规划（Reasoning & Planning）
     → 行动（Action）
     → 新状态 → 再次感知（循环）
```

### 2. Agent vs. 普通 LLM 调用

| 特征 | 普通 LLM 调用 | AI Agent |
|------|-------------|----------|
| 交互方式 | 单次问答 | 多轮循环 |
| 工具使用 | 无 | 有（搜索、代码执行、API等） |
| 状态管理 | 无持久状态 | 有记忆/上下文管理 |
| 目标导向 | 即时回复 | 达成长期目标 |
| 自主性 | 低 | 高 |

### 3. Agent 的四大核心组件

#### 🧠 规划（Planning）
- 目标分解：将复杂目标拆解为子任务
- 反思与修正：执行中动态调整计划
- 相关技术：Chain-of-Thought、Tree-of-Thoughts、ReAct

#### 🛠️ 工具调用（Tool Use）
- 搜索工具（Web Search、向量检索）
- 代码执行（Python REPL、Shell）
- 外部 API（数据库、日历、文件系统）
- 其他 Agent（多智能体协作）

#### 🗄️ 记忆（Memory）
- **短期记忆**：对话上下文（Context Window）
- **长期记忆**：向量数据库、外部存储
- **操作性记忆**：任务执行状态

#### 👁️ 感知（Perception）
- 文本输入
- 多模态输入（图像、音频、视频）
- 环境状态读取（工具返回结果）

### 4. Agent 的工作循环

```
Goal → Think → Act → Observe → Think → Act → Observe → ... → Done
```

这个循环被称为 **ReAct 循环**（Reason + Act），是现代 Agent 的核心执行模式。

### 5. 主流 Agent 框架全景

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| LangChain Agents | 生态丰富，工具链完整 | 快速原型 |
| AutoGPT | 高度自主，持久任务 | 长期任务 |
| CrewAI | 多角色协作 | 团队任务模拟 |
| LlamaIndex Agents | RAG + Agent 结合 | 知识密集型 |
| OpenAI Assistants | 官方 API，稳定 | 生产部署 |
| **ALLIN Claw** | 平台化，多 Worker 协作 | 企业级任务编排 |

---

## 💡 示例/推导

### 示例：一个文件整理 Agent 的执行过程

**目标**：整理 Downloads 文件夹，按类型归档

```
Goal: 整理 Downloads 文件夹

Step 1: Think
  → "需要先列出文件夹内容，了解有哪些文件"
  
Step 2: Act
  → 调用工具：list_directory("/Downloads")
  → 返回：["report.pdf", "photo.jpg", "script.py", "music.mp3", ...]

Step 3: Observe
  → 观察到有 PDF、图片、代码、音乐等类型
  
Step 4: Think
  → "制定分类规则：pdf→docs, jpg/png→images, py→code, mp3→music"
  
Step 5: Act
  → 调用工具：create_directory("/Downloads/docs")
  → 调用工具：move_file("report.pdf", "/Downloads/docs/")
  → ... (循环执行)
  
Step 6: Observe
  → 所有文件已分类完毕
  
Step 7: Think
  → "任务完成，生成执行摘要"
  
Done ✅
```

### 推导：为什么 Agent 需要循环？

单次 LLM 调用的局限：
```
输入："帮我查今天的天气，然后推荐是否需要带伞"

LLM（无工具）："我无法获取实时天气信息..."   ❌ 无能为力

LLM（有工具，单次）：调用天气API → 获取结果 → 回复   ✅ 但只能处理简单任务

Agent（有循环）：
  → 查天气 → 分析降雨概率 → 查用户位置 → 综合判断 → 回复   ✅✅ 处理复杂任务
```

---

## 🔧 动手练习

### 练习 1：概念图谱绘制（必做）

在纸上或用任意工具画出 Agent 的四大组件关系图，标注它们之间的数据流向。

### 练习 2：环境准备（必做）

```bash
# 使用 uv 创建项目环境
uv init dandan-agent-lab
cd dandan-agent-lab

# 安装核心依赖
uv add langchain langchain-openai python-dotenv

# 创建环境变量文件
cat > .env << 'EOF'
OPENAI_API_KEY=your_key_here
EOF
```

### 练习 3：第一个"模拟 Agent"（必做）

创建 `01_simple_agent_loop.py`：

```python
"""
模拟一个简单的 Agent 执行循环，理解 Think-Act-Observe 模式
"""

def think(goal: str, observations: list[str]) -> str:
    """模拟 Agent 的思考过程"""
    if not observations:
        return f"开始分析目标: '{goal}'，首先需要了解当前状态"
    
    last_obs = observations[-1]
    if "完成" in last_obs:
        return "所有步骤已完成，整理结果"
    return f"基于观察 '{last_obs}'，决定下一步行动"

def act(thought: str, step: int) -> str:
    """模拟工具调用"""
    actions = {
        1: "🔍 搜索相关信息...",
        2: "📊 分析数据...",
        3: "✍️ 生成报告...",
    }
    return actions.get(step, "✅ 任务完成")

def observe(action_result: str) -> str:
    """模拟观察行动结果"""
    return f"观察到：{action_result}"

def run_agent(goal: str, max_steps: int = 5):
    """运行 Agent 循环"""
    print(f"\n🤖 Agent 启动")
    print(f"📌 目标: {goal}")
    print("-" * 50)
    
    observations = []
    
    for step in range(1, max_steps + 1):
        print(f"\n[Step {step}]")
        
        # Think
        thought = think(goal, observations)
        print(f"💭 思考: {thought}")
        
        # Act
        action_result = act(thought, step)
        print(f"⚡ 行动: {action_result}")
        
        # Observe
        observation = observe(action_result)
        observations.append(observation)
        print(f"👁️ 观察: {observation}")
        
        if "完成" in action_result:
            print(f"\n✅ Agent 完成任务！共执行 {step} 步")
            break

if __name__ == "__main__":
    run_agent("撰写一份关于 AI Agent 的简短报告")
```

运行：
```bash
uv run python 01_simple_agent_loop.py
```

**预期输出**：看到 Think → Act → Observe 的循环过程，理解 Agent 的基本节奏。

### 🦞 Claw 实战：观察 Claw Agent 的执行日志

1. 打开任意一个你参与的 Claw 项目
2. 找到一个已完成的 Task
3. 观察执行日志，对应到今天学的 Think-Act-Observe 结构：
   - 哪些消息是"思考"？
   - 哪些 API 调用是"行动"？
   - 哪些返回值是"观察"？
4. 在笔记中记录你的对应关系

---

## 📝 小结

今天建立了对 AI Agent 的整体认知：

| 要点 | 核心内容 |
|------|---------|
| **定义** | Agent = 感知 + 规划 + 行动 + 记忆的自主系统 |
| **vs LLM** | Agent 有循环、有工具、有状态，LLM 是单次问答 |
| **四组件** | 规划、工具调用、记忆、感知 |
| **执行模式** | Think → Act → Observe → Loop |
| **全景** | LangChain/AutoGPT/CrewAI/Claw 等各有适用场景 |

**明天预告**：深入学习 ReAct 框架——Agent 的核心推理范式，动手实现一个真正有工具调用的 ReAct Agent。

---

> 💡 **今日思考题**：如果让你设计一个"自动写周报"的 Agent，它需要哪些工具？执行循环大概是什么样的？
