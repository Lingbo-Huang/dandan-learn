# Week 2 综合实战：多框架协作的智能研究助手

> **Day 7** · 预计学习时间：5-6 小时  
> **目标**：综合运用本周所学，构建一个端到端的完整项目

---

## 项目概述

### 我们要构建什么？

**"DandanResearch" — 智能研究助手**

一个能自动完成"提出问题 → 检索资料 → 多角色讨论 → 生成报告"全流程的 Agent 系统。

**架构设计：**
```
用户输入研究主题
      ↓
[LlamaIndex] 检索本地知识库（RAG 层）
      ↓
[CrewAI] 研究员 + 写手 + 批评者协作生成报告（编排层）
      ↓
[LangChain] 工具调用（格式化、存储、通知）（链路层）
      ↓
生成最终研究报告 Markdown 文件
```

**为什么这样设计？**
- LlamaIndex：最擅长文档检索，用它做 RAG 基础
- CrewAI：流程固定（研究→写作→审核），用它做角色编排
- LangChain：工具生态丰富，用它做最后的格式化和输出

---

## 项目结构

```
dandan-research/
├── pyproject.toml          # uv 项目配置
├── .env                    # 环境变量
├── main.py                 # 主入口
├── knowledge_base/         # 本地知识库文档（放你自己的 PDF/txt）
│   ├── ai_papers.txt       # 示例文档
│   └── tech_articles.txt
├── agents/
│   ├── __init__.py
│   ├── rag_tool.py         # LlamaIndex RAG 工具
│   ├── research_crew.py    # CrewAI 研究团队
│   └── output_chain.py     # LangChain 输出处理
├── output/                 # 生成的报告存储目录
└── storage/                # LlamaIndex 向量索引缓存
```

---

## 核心代码示例（使用 uv）

### 环境准备

```bash
uv init dandan-research
cd dandan-research

# 安装所有依赖
uv add langchain langchain-openai langchain-core \
       llama-index llama-index-llms-openai llama-index-embeddings-openai \
       crewai \
       python-dotenv pydantic

# 创建目录
mkdir -p knowledge_base output storage agents

# 环境变量
cat > .env << 'EOF'
OPENAI_API_KEY=your-key-here
EOF
```

### 创建示例知识库文档

```bash
cat > knowledge_base/ai_papers.txt << 'EOF'
# AI Agent 框架技术综述

## LangChain
LangChain 于 2022 年 10 月发布，是目前最流行的 LLM 应用开发框架。
核心特性：LCEL 表达式、丰富的工具生态、多 LLM 支持。
2024 年主要更新：LangGraph 发布（有状态 Agent）、LangSmith 追踪平台完善。

## LlamaIndex
LlamaIndex（原 GPT Index）专注于数据连接和 RAG。
核心特性：多种 Index 类型、精细化检索控制、与向量数据库深度集成。
适用场景：私有文档问答、知识库构建、多数据源综合查询。

## AutoGen
微软研究院开发的多 Agent 对话框架，2023 年发布。
核心特性：对话式协作、代码自动执行、Human-in-the-loop。
2024 年重要更新：AutoGen 0.4 大重构，引入 Actor 模型架构。

## CrewAI
角色驱动的任务编排框架，2023 年末发布，增长迅速。
核心特性：Role/Goal/Backstory 三要素、Task 依赖链、hierarchical process。
最适合：内容生产、研究报告、有明确流程的多角色任务。
EOF
```

### agents/rag_tool.py — LlamaIndex RAG 工具

```python
# agents/rag_tool.py
import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

load_dotenv()

# 全局 LlamaIndex 配置
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

KNOWLEDGE_BASE_DIR = "./knowledge_base"
STORAGE_DIR = "./storage"

def get_or_build_index():
    """获取或构建向量索引"""
    if os.path.exists(STORAGE_DIR) and os.listdir(STORAGE_DIR):
        print("📚 从缓存加载知识库索引...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)
    
    print("🔨 构建知识库索引（首次运行）...")
    documents = SimpleDirectoryReader(KNOWLEDGE_BASE_DIR).load_data()
    
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    
    print(f"✅ 索引构建完成，共处理 {len(documents)} 个文档")
    return index

# 初始化索引
_index = None

def get_index():
    global _index
    if _index is None:
        _index = get_or_build_index()
    return _index

# 封装为 CrewAI Tool
class KnowledgeSearchInput(BaseModel):
    query: str = Field(description="搜索查询，描述你想了解的内容")

class KnowledgeBaseTool(BaseTool):
    name: str = "知识库检索"
    description: str = """搜索本地知识库，获取 AI 框架相关的技术资料。
    适用于：查找框架特性、技术对比、最新动态等信息。"""
    args_schema: Type[BaseModel] = KnowledgeSearchInput
    
    def _run(self, query: str) -> str:
        index = get_index()
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(query)
        
        result = f"检索结果：\n{response}\n\n"
        if response.source_nodes:
            result += "引用来源：\n"
            for node in response.source_nodes[:2]:
                result += f"- 相似度 {node.score:.2f}: {node.text[:100]}...\n"
        
        return result
```

### agents/research_crew.py — CrewAI 研究团队

```python
# agents/research_crew.py
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from agents.rag_tool import KnowledgeBaseTool

load_dotenv()

def create_research_crew(topic: str) -> str:
    """创建并运行研究团队"""
    
    kb_tool = KnowledgeBaseTool()
    
    # === 定义 Agent ===
    
    researcher = Agent(
        role="AI 技术研究员",
        goal=f"深入研究「{topic}」，提供有价值的技术见解",
        backstory="""你是一位专注于 AI 基础设施领域的技术研究员，
        在多个顶级 AI 实验室和科技公司有过工作经历。
        你擅长快速理解新技术，能从庞杂的信息中提炼核心要点。
        你习惯用「第一原则」思考问题，不人云亦云。""",
        tools=[kb_tool],
        llm="gpt-4o-mini",
        verbose=True,
    )
    
    writer = Agent(
        role="技术文档写手",
        goal="将技术研究成果转化为清晰易读的文章",
        backstory="""你曾在 InfoQ、极客时间等技术媒体担任主编，
        深知如何让复杂的技术内容变得引人入胜。
        你的文章总是结构清晰，例子贴切，让读者有所收获。
        你特别重视"实用性"——每篇文章要让读者能立刻用起来。""",
        llm="gpt-4o-mini",
        verbose=True,
    )
    
    critic = Agent(
        role="技术编辑",
        goal="确保文章的准确性、完整性和可读性",
        backstory="""你是一位严格的技术编辑，有很强的技术背景。
        你的职责是找出文章中的错误、遗漏和不清晰之处。
        你只在真正满意时才放行，否则会提出具体的改进建议。
        当文章质量达到发布标准时，你会说"APPROVED"。""",
        llm="gpt-4o-mini",
        verbose=True,
    )
    
    # === 定义 Task ===
    
    research_task = Task(
        description=f"""研究主题：{topic}
        
        请搜索知识库，整理以下方面的信息：
        1. 核心概念与定义
        2. 主要技术特点与优势
        3. 典型应用场景
        4. 当前存在的局限性或挑战
        5. 2024 年的最新发展
        
        输出：结构化的研究笔记（300-400 字），作为写作素材。""",
        expected_output="结构化研究笔记，覆盖核心概念、技术特点、应用场景、挑战和最新动态",
        agent=researcher,
    )
    
    write_task = Task(
        description=f"""基于研究笔记，撰写一篇关于「{topic}」的技术文章。
        
        文章要求：
        - 标题：吸引技术读者，点明主题
        - 引言：1 段，说明为什么这个主题重要
        - 正文：3-4 个小节，逻辑递进
        - 代码示例：至少 1 个简短的代码示例（如果适合的话）
        - 结论：1 段，点明关键收获
        - 字数：400-600 字
        
        面向读者：有 1-2 年开发经验的程序员""",
        expected_output="完整的技术文章，包含标题、引言、正文（含代码示例）和结论",
        agent=writer,
        context=[research_task],
    )
    
    review_task = Task(
        description="""审查文章，按以下维度评分并给出改进建议：
        
        评审维度（每项 1-5 分）：
        - 准确性：技术描述是否正确
        - 完整性：是否覆盖了关键点
        - 可读性：是否易于理解
        - 实用性：读者是否能学到可用的知识
        
        输出格式：
        1. 评分表格
        2. 存在的 2-3 个具体问题
        3. 最终结论：APPROVED（通过）或 REVISION NEEDED（需修改）
        
        如果综合分 ≥ 16/20，输出 APPROVED。""",
        expected_output="评审报告，含评分、问题指出和最终判断（APPROVED 或 REVISION NEEDED）",
        agent=critic,
        context=[write_task],
    )
    
    # === 组建 Crew ===
    
    crew = Crew(
        agents=[researcher, writer, critic],
        tasks=[research_task, write_task, review_task],
        process=Process.sequential,
        verbose=True,
    )
    
    result = crew.kickoff()
    return str(result)
```

### agents/output_chain.py — LangChain 输出处理

```python
# agents/output_chain.py
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def format_and_save_report(research_result: str, topic: str) -> str:
    """用 LangChain 处理和格式化最终报告"""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # 格式化链：把原始研究结果整理成标准 Markdown 报告
    format_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个技术文档整理专家。
        将输入的研究结果整理成标准格式的 Markdown 报告。
        
        报告结构：
        # [标题]
        
        > 摘要：一句话说明报告核心内容
        
        **研究日期**：[当前日期]  
        **主题**：[研究主题]
        
        ---
        
        [正文内容，保持原有逻辑，稍作格式优化]
        
        ---
        
        ## 关键收获
        
        [3-5 条 bullet points，提炼最重要的洞察]"""),
        ("human", "研究主题：{topic}\n\n原始内容：\n{content}")
    ])
    
    format_chain = format_prompt | llm | StrOutputParser()
    
    formatted = format_chain.invoke({
        "topic": topic,
        "content": research_result
    })
    
    # 保存到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/research_{timestamp}.md"
    os.makedirs("output", exist_ok=True)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(formatted)
    
    print(f"✅ 报告已保存：{filename}")
    return filename
```

### main.py — 主入口

```python
# main.py
import sys
from agents.research_crew import create_research_crew
from agents.output_chain import format_and_save_report

def main():
    print("🔬 DandanResearch — 智能研究助手")
    print("=" * 50)
    
    # 获取研究主题
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = input("请输入研究主题：").strip()
        if not topic:
            topic = "LangChain 与 LlamaIndex 的异同与协作模式"
    
    print(f"\n📋 研究主题：{topic}")
    print("🚀 启动研究团队...\n")
    
    # Step 1: CrewAI 研究团队执行
    research_result = create_research_crew(topic)
    
    # Step 2: LangChain 格式化并保存
    output_file = format_and_save_report(research_result, topic)
    
    print("\n" + "=" * 50)
    print(f"✅ 研究完成！报告已保存至：{output_file}")
    print("\n--- 报告预览（前 500 字）---")
    
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read()
    print(content[:500])
    print("...")

if __name__ == "__main__":
    main()
```

### 运行项目

```bash
# 运行方式 1：交互式
uv run main.py

# 运行方式 2：直接传入主题
uv run main.py "AutoGen 多 Agent 框架的最佳实践"

# 运行方式 3：批量研究
for topic in "LangChain LCEL 语法" "CrewAI 角色设计" "LlamaIndex RAG 优化"; do
    echo "研究主题：$topic"
    uv run main.py "$topic"
    echo "---"
done
```

---

## 项目扩展方向

完成基础版后，可以继续挑战：

### 扩展 1：加入 AutoGen 代码验证
```python
# 让 AutoGen 验证代码示例的正确性
import autogen

code_validator = autogen.AssistantAgent(
    name="代码验证员",
    system_message="验证文章中的代码示例，确保它们可以正确运行。",
    llm_config={"config_list": config_list}
)
```

### 扩展 2：支持 Web 知识库更新
```python
# 定期抓取 GitHub Release Notes 更新知识库
# 使用 LangChain 的 WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://github.com/langchain-ai/langchain/releases")
docs = loader.load()
```

### 扩展 3：Gradio 可视化界面
```bash
uv add gradio
```

```python
import gradio as gr

def research(topic):
    result = create_research_crew(topic)
    filename = format_and_save_report(result, topic)
    with open(filename) as f:
        return f.read()

demo = gr.Interface(
    fn=research,
    inputs=gr.Textbox(label="研究主题"),
    outputs=gr.Markdown(label="研究报告"),
    title="DandanResearch 智能研究助手"
)
demo.launch()
```

---

## 与 Claw 的对比与联系

**DandanResearch 就是一个"简化版 Claw"：**

| 功能 | DandanResearch 实现 | OpenClaw 实现 |
|------|-------------------|--------------|
| 任务拆分 | CrewAI Task 列表 | Project Task 依赖链 |
| 多角色协作 | CrewAI Agent | Worker Agent（分配 Task） |
| 数据检索 | LlamaIndex | Skill（用户上传文件） |
| 结果输出 | LangChain 格式化 + 本地文件 | Task summary + MEDIA: 文件 |
| 用户确认 | 无（全自动） | `pending_confirmation` → 用户确认 |
| 状态追踪 | 无 | Task status（running/completed/error） |

**关键差距在哪里？**
- Claw 解决了"用户如何介入"的问题（Human-in-the-loop 在平台层）
- Claw 解决了"状态如何持久化"的问题（任务状态存在平台数据库）
- Claw 解决了"多用户并发"的问题（每个 User 有独立 Session）

理解了这个差距，你就理解了 **Agent 平台** 和 **Agent 框架** 的根本区别。

---

## Week 2 总结

### 你学到了什么

| 框架 | 核心收获 |
|------|---------|
| LangChain | LCEL 管道、工具绑定、结构化输出 |
| LlamaIndex | RAG 全流程、Index 类型、持久化索引 |
| AutoGen | 对话驱动协作、Human-in-the-loop、代码执行 |
| CrewAI | 角色设计、backstory、Task 依赖链 |

### 你建立的直觉

- **框架选型直觉**：看到需求脑子里能冒出"这个用 XX 框架"
- **组合使用思维**：不同框架各司其职，拼在一起才是最强解法
- **框架 vs 平台**：理解了 Agent 框架（开发者工具）和 Agent 平台（用户产品）的本质区别

### 下周预告

Week 3 将深入 **Agent 记忆与持久化**——让 Agent 真正"记住"东西：
- 向量记忆（语义搜索历史）
- 图数据库记忆（关系推理）
- 外部记忆系统（Mem0、Zep）
- Claw 的 MEMORY.md 机制背后的设计哲学

---

## 小结

- **综合项目的价值**：把碎片知识串起来，发现框架之间的边界和接合点
- **LlamaIndex + CrewAI + LangChain 的组合**覆盖了大多数内容生产类 AI 应用
- **构建过程中最大的收获**：你会发现很多"设计决策"背后有深刻的工程原因
- **展示你的作品**：这个项目是你 Week 2 最好的简历项目——部署一个 Gradio 界面，截图发到朋友圈

恭喜完成 Week 2！🎉
