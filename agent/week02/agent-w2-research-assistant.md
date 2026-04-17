# D7 综合实战：用 LangChain + AutoGen 搭建研究助手

> **学习目标**：综合运用本周所学，构建一个完整的 AI 研究助手系统，该系统能够自动搜集资料、分析内容、生成报告，并支持多轮对话交互。

---

## 一、项目背景与需求

### 系统定位

构建一个「AI 研究助手」，具备以下核心能力：

1. **自动文献检索**：根据研究主题，自动搜集互联网上的相关资料
2. **多 Agent 协作分析**：不同 Agent 负责不同分析维度（技术、市场、趋势）
3. **RAG 知识增强**：将搜集到的资料向量化，支持精准问答
4. **报告自动生成**：整合分析结果，输出结构化研究报告
5. **交互式精炼**：用户可以追问，系统持续完善答案

### 架构设计

```
用户输入研究主题
        │
        ▼
┌───────────────────┐
│   LangChain       │  ← 主入口、工具编排、RAG
│   Orchestrator    │
└───────┬───────────┘
        │ 触发多Agent协作
        ▼
┌─────────────────────────────────────────┐
│           AutoGen Agent Crew            │
│  ┌─────────┐  ┌─────────┐  ┌────────┐  │
│  │技术分析师│  │市场分析师│  │趋势预测│  │
│  └────┬────┘  └────┬────┘  └───┬────┘  │
│       └────────────┴───────────┘       │
│                    │                    │
│              ┌─────▼──────┐            │
│              │ 研究协调员  │            │
│              └─────┬──────┘            │
└────────────────────┼────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  LlamaIndex   │  ← 知识库（搜集的资料）
              │  RAG Engine   │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   最终报告    │
              └───────────────┘
```

---

## 二、环境准备

```bash
# 安装依赖
pip install langchain langchain-openai langchain-community \
            llama-index llama-index-llms-openai \
            pyautogen \
            duckduckgo-search \
            chromadb \
            python-dotenv
```

```ini
# .env 文件
OPENAI_API_KEY=your-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
```

---

## 三、工具层：搜索与数据采集

```python
# tools/search_tools.py
import os
from typing import List, Dict
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

_search_engine = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """搜索互联网，获取关于指定查询词的最新信息"""
    try:
        results = _search_engine.run(query)
        return results[:2000]  # 限制长度
    except Exception as e:
        return f"搜索失败：{e}"

@tool
def fetch_webpage(url: str) -> str:
    """获取指定网页的完整文本内容"""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if docs:
            content = docs[0].page_content[:3000]
            return f"来源：{url}\n\n内容：{content}"
        return f"无法加载网页：{url}"
    except Exception as e:
        return f"加载失败：{e}"

@tool
def search_and_collect(topic: str, num_queries: int = 3) -> List[str]:
    """根据主题生成多个搜索查询并收集结果，返回原始文本列表"""
    queries = [
        f"{topic} 最新进展 2024",
        f"{topic} 技术原理",
        f"{topic} 应用案例 市场",
    ][:num_queries]
    
    collected = []
    for q in queries:
        result = _search_engine.run(q)
        collected.append(f"查询：{q}\n{result}")
    
    return collected

@tool
def get_current_date() -> str:
    """获取当前日期"""
    from datetime import datetime
    return datetime.now().strftime("%Y年%m月%d日")
```

---

## 四、知识库层：基于 LlamaIndex 的 RAG

```python
# knowledge/rag_engine.py
from llama_index.core import (
    VectorStoreIndex, Document, Settings,
    StorageContext
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from typing import List

class ResearchKnowledgeBase:
    """研究知识库：动态构建、支持增量更新"""
    
    def __init__(self, topic: str):
        self.topic = topic
        self.documents: List[Document] = []
        self.index = None
        
        # 配置 LlamaIndex
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    
    def add_content(self, texts: List[str], source: str = "web"):
        """向知识库添加内容"""
        new_docs = [
            Document(
                text=text,
                metadata={"source": source, "topic": self.topic}
            )
            for text in texts if text.strip()
        ]
        self.documents.extend(new_docs)
        print(f"已添加 {len(new_docs)} 个文档，知识库共 {len(self.documents)} 个文档")
    
    def build_index(self):
        """构建向量索引"""
        if not self.documents:
            raise ValueError("知识库为空，请先添加内容")
        
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            show_progress=True
        )
        print(f"索引构建完成，共处理 {len(self.documents)} 个文档")
    
    def query(self, question: str, top_k: int = 5) -> str:
        """查询知识库"""
        if self.index is None:
            return "知识库尚未初始化，请先构建索引"
        
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )
        engine = RetrieverQueryEngine.from_args(retriever)
        response = engine.query(question)
        
        # 返回答案和来源
        sources = [
            node.metadata.get("source", "未知")
            for node in response.source_nodes
        ]
        return f"{response}\n\n(信息来源：{', '.join(set(sources))})"
    
    def as_langchain_tool(self):
        """将知识库封装为 LangChain 工具"""
        from langchain_core.tools import tool
        kb = self  # 闭包捕获
        
        @tool
        def query_knowledge_base(question: str) -> str:
            """查询研究知识库，获取已收集资料中的相关信息"""
            return kb.query(question)
        
        return query_knowledge_base
```

---

## 五、AutoGen 多 Agent 分析层

```python
# agents/research_agents.py
import autogen
from typing import Dict, Any

def create_research_agents(llm_config: Dict[str, Any]) -> Dict:
    """创建研究分析 Agent 团队"""
    
    # 1. 技术分析师
    tech_analyst = autogen.AssistantAgent(
        name="技术分析师",
        llm_config=llm_config,
        system_message="""你是一位专注技术层面的研究分析师。

你的职责：
- 分析技术架构和实现原理
- 评估技术成熟度和局限性
- 对比不同技术方案的优劣
- 识别关键技术挑战和突破点

输出风格：客观、专业、有深度，善用技术术语但能清晰解释。
当完成分析后，以"【技术分析完成】"结尾。""",
    )
    
    # 2. 市场分析师
    market_analyst = autogen.AssistantAgent(
        name="市场分析师",
        llm_config=llm_config,
        system_message="""你是一位专注市场和商业层面的研究分析师。

你的职责：
- 分析市场规模、增长率和竞争格局
- 识别主要参与者和各自的市场定位
- 评估商业模式和变现路径
- 分析用户需求和痛点

输出风格：数据驱动，善用数字和案例，逻辑清晰。
当完成分析后，以"【市场分析完成】"结尾。""",
    )
    
    # 3. 趋势预测师
    trend_analyst = autogen.AssistantAgent(
        name="趋势预测师",
        llm_config=llm_config,
        system_message="""你是一位专注未来趋势的研究分析师。

你的职责：
- 基于当前数据预判未来发展方向
- 识别潜在的颠覆性变化
- 分析政策、社会、技术的交叉影响
- 给出有时间维度的发展预测

输出风格：前瞻性、有说服力，区分短中长期预测。
当完成分析后，以"【趋势分析完成】"结尾。""",
    )
    
    # 4. 研究协调员（综合结论）
    coordinator = autogen.AssistantAgent(
        name="研究协调员",
        llm_config=llm_config,
        system_message="""你是研究团队的协调员，负责整合各专家的分析结果。

你的职责：
- 协调各分析师完成各自的分析
- 整合各方观点，生成综合性研究结论
- 确保分析全面、没有重大遗漏
- 最终生成一份结构化的研究摘要

当所有分析师完成分析后，整合所有内容，输出最终综合分析，
以"RESEARCH_COMPLETE"结尾（这是终止信号）。""",
    )
    
    # 5. 执行代理（代替用户参与对话）
    executor = autogen.UserProxyAgent(
        name="研究助手执行者",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: "RESEARCH_COMPLETE" in x.get("content", ""),
        code_execution_config=False,
        system_message="你是一个自动化的研究助手，负责协调多个分析师完成研究任务",
    )
    
    return {
        "tech_analyst": tech_analyst,
        "market_analyst": market_analyst,
        "trend_analyst": trend_analyst,
        "coordinator": coordinator,
        "executor": executor,
    }

def run_multi_agent_analysis(topic: str, background_info: str, llm_config: Dict) -> str:
    """运行多 Agent 协作分析"""
    agents = create_research_agents(llm_config)
    
    # 创建群聊
    group_chat = autogen.GroupChat(
        agents=[
            agents["executor"],
            agents["tech_analyst"],
            agents["market_analyst"],
            agents["trend_analyst"],
            agents["coordinator"],
        ],
        messages=[],
        max_round=15,
        speaker_selection_method="round_robin",
    )
    
    manager = autogen.GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )
    
    # 启动分析
    research_prompt = f"""
请各位专家协作分析：**{topic}**

背景信息（来自资料收集）：
{background_info[:2000]}

分析要求：
1. 技术分析师：分析技术层面
2. 市场分析师：分析市场层面
3. 趋势预测师：分析未来趋势
4. 协调员：最终整合所有分析
"""
    
    agents["executor"].initiate_chat(
        manager,
        message=research_prompt,
    )
    
    # 提取协调员的最终结论
    final_analysis = ""
    for msg in reversed(group_chat.messages):
        if msg.get("name") == "研究协调员":
            final_analysis = msg.get("content", "")
            break
    
    return final_analysis
```

---

## 六、LangChain 主编排层

```python
# orchestrator.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.tools import tool

from tools.search_tools import web_search, fetch_webpage, get_current_date
from knowledge.rag_engine import ResearchKnowledgeBase
from agents.research_agents import run_multi_agent_analysis

import os

class ResearchAssistant:
    """研究助手主入口：整合 LangChain + AutoGen + LlamaIndex"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.api_key
        )
        self.knowledge_base = None
        self.session_store = {}
        self.llm_config = {
            "config_list": [{
                "model": "gpt-4o-mini",
                "api_key": self.api_key,
            }],
            "temperature": 0,
        }
    
    def _build_agent(self, tools_list):
        """构建 LangChain Agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的研究助手。你可以：
            1. 搜索互联网获取最新信息
            2. 查询已建立的研究知识库
            3. 执行深度研究分析

            始终优先使用知识库，如知识库没有答案再搜索互联网。
            回答要准确、有深度，并注明信息来源。"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, tools_list, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools_list,
            verbose=True,
            max_iterations=8,
            handle_parsing_errors=True,
        )
    
    def conduct_research(self, topic: str) -> dict:
        """完整的研究流程"""
        print(f"\n{'='*60}")
        print(f"开始研究：{topic}")
        print('='*60)
        
        # === 阶段 1：资料收集 ===
        print("\n📡 阶段1：收集资料...")
        from tools.search_tools import search_and_collect
        raw_texts = search_and_collect.invoke({"topic": topic, "num_queries": 3})
        
        # 额外搜索补充
        tech_info = web_search.invoke({"query": f"{topic} 技术原理详解"})
        market_info = web_search.invoke({"query": f"{topic} 市场规模 2024"})
        
        all_texts = raw_texts + [tech_info, market_info]
        print(f"✅ 收集到 {len(all_texts)} 份资料")
        
        # === 阶段 2：构建知识库 ===
        print("\n📚 阶段2：构建研究知识库...")
        self.knowledge_base = ResearchKnowledgeBase(topic)
        self.knowledge_base.add_content(all_texts, source="web_search")
        self.knowledge_base.build_index()
        
        # === 阶段 3：多 Agent 深度分析 ===
        print("\n🤖 阶段3：多Agent协作分析...")
        background = "\n\n".join(all_texts[:3])  # 提取前3份资料作为背景
        multi_agent_analysis = run_multi_agent_analysis(
            topic=topic,
            background_info=background,
            llm_config=self.llm_config
        )
        print("✅ 多Agent分析完成")
        
        # === 阶段 4：生成最终报告 ===
        print("\n📝 阶段4：生成研究报告...")
        report = self._generate_final_report(topic, multi_agent_analysis)
        
        return {
            "topic": topic,
            "multi_agent_analysis": multi_agent_analysis,
            "final_report": report,
            "knowledge_base": self.knowledge_base,
        }
    
    def _generate_final_report(self, topic: str, analysis: str) -> str:
        """用 LangChain 链生成最终报告"""
        report_chain = (
            ChatPromptTemplate.from_template("""
你是一个专业报告撰写专家。请基于以下多专家分析结果，
生成一份结构完整、专业权威的研究报告。

研究主题：{topic}

专家分析结果：
{analysis}

请生成包含以下章节的完整报告（Markdown 格式）：
1. 执行摘要（200字，结论前置）
2. 研究背景与目的
3. 技术现状分析
4. 市场格局分析
5. 发展趋势预测
6. 机遇与挑战
7. 结论与建议

要求：专业、客观、有深度，适当使用数据和案例支撑。
""")
            | self.llm
            | StrOutputParser()
        )
        
        return report_chain.invoke({
            "topic": topic,
            "analysis": analysis[:3000]
        })
    
    def chat(self, query: str, session_id: str = "default") -> str:
        """交互式问答（利用已建立的知识库）"""
        if not self.knowledge_base:
            return "请先调用 conduct_research() 建立研究知识库"
        
        # 动态构建工具列表
        kb_tool = self.knowledge_base.as_langchain_tool()
        tools = [web_search, get_current_date, kb_tool]
        
        executor = self._build_agent(tools)
        
        # 获取会话历史
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        history = self.session_store[session_id]
        
        # 构建带历史的执行器
        def get_history(sid):
            return self.session_store.get(sid, ChatMessageHistory())
        
        chain_with_history = RunnableWithMessageHistory(
            executor,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        
        result = chain_with_history.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        
        return result.get("output", "")
```

---

## 七、报告生成与格式化

```python
# report/generator.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

def generate_executive_summary(research_result: dict) -> str:
    """生成执行摘要（用于快速浏览）"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    summary_chain = (
        ChatPromptTemplate.from_template("""
基于以下研究报告，生成一份500字以内的执行摘要。
要求：结论优先，用数字说话，突出关键发现。

研究主题：{topic}
完整报告：{report}

执行摘要：""")
        | llm
        | StrOutputParser()
    )
    
    return summary_chain.invoke({
        "topic": research_result["topic"],
        "report": research_result["final_report"][:3000]
    })

def export_report(research_result: dict, output_dir: str = ".") -> str:
    """将研究结果导出为 Markdown 文件"""
    topic = research_result["topic"]
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"{output_dir}/research_{topic.replace(' ', '_')}_{date_str}.md"
    
    content = f"""# 研究报告：{topic}

**生成时间**：{datetime.now().strftime("%Y年%m月%d日 %H:%M")}

---

{research_result["final_report"]}

---

## 附录：多专家分析原始记录

{research_result.get("multi_agent_analysis", "")}
"""
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"报告已保存：{filename}")
    return filename
```

---

## 八、完整主程序

```python
# main.py
import os
from dotenv import load_dotenv
from orchestrator import ResearchAssistant
from report.generator import generate_executive_summary, export_report

def main():
    load_dotenv()
    
    print("🚀 AI 研究助手启动")
    print("="*60)
    
    assistant = ResearchAssistant()
    
    # 获取研究主题
    topic = input("\n请输入研究主题（例如：大模型微调技术）：").strip()
    if not topic:
        topic = "AI Agent 框架发展现状"
    
    # 执行完整研究流程
    print(f"\n开始研究：{topic}")
    research_result = assistant.conduct_research(topic)
    
    # 生成执行摘要
    print("\n📋 生成执行摘要...")
    summary = generate_executive_summary(research_result)
    print("\n" + "="*60)
    print("执行摘要：")
    print("="*60)
    print(summary)
    
    # 导出报告
    report_file = export_report(research_result)
    print(f"\n✅ 完整报告已保存至：{report_file}")
    
    # 进入交互模式
    print("\n" + "="*60)
    print("进入交互模式（输入 'quit' 退出）")
    print("="*60)
    
    session_id = f"research_{topic[:10]}"
    
    while True:
        query = input("\n你的问题：").strip()
        if query.lower() in ["quit", "exit", "q", "退出"]:
            print("感谢使用，再见！")
            break
        if not query:
            continue
        
        print("\n🤔 思考中...")
        answer = assistant.chat(query, session_id=session_id)
        print(f"\n💡 回答：\n{answer}")

if __name__ == "__main__":
    main()
```

---

## 九、运行示例与效果展示

```bash
# 运行程序
python main.py

# 示例输出：
🚀 AI 研究助手启动
============================================================
请输入研究主题：AI Agent 框架发展现状

开始研究：AI Agent 框架发展现状
============================================================

📡 阶段1：收集资料...
[搜索] AI Agent 框架发展现状 最新进展 2024 ...
[搜索] AI Agent 框架发展现状 技术原理 ...
✅ 收集到 5 份资料

📚 阶段2：构建研究知识库...
已添加 5 个文档，知识库共 5 个文档
索引构建完成，共处理 5 个文档

🤖 阶段3：多Agent协作分析...
[研究协调员] 请各位专家开始分析...
[技术分析师] 从技术层面来看...【技术分析完成】
[市场分析师] 从市场规模看...【市场分析完成】
[趋势预测师] 展望未来...【趋势分析完成】
[研究协调员] 综合以上分析... RESEARCH_COMPLETE
✅ 多Agent分析完成

📝 阶段4：生成研究报告...

============================================================
执行摘要：
============================================================
AI Agent 框架在2024年呈现爆发式增长...（500字摘要）

✅ 完整报告已保存至：./research_AI_Agent框架_20241201.md

============================================================
进入交互模式（输入 'quit' 退出）
============================================================

你的问题：LangChain 和 AutoGen 的主要区别是什么？

🤔 思考中...
> 查询知识库...
> 检索到 3 个相关文档

💡 回答：
根据我们的研究，LangChain 和 AutoGen 的主要区别体现在：
1. 设计哲学：LangChain 是组件化管道，AutoGen 是对话驱动...
```

---

## 十、优化与扩展方向

### 10.1 性能优化

```python
# 1. 异步并行搜索（大幅提升收集速度）
import asyncio
import aiohttp

async def async_research_pipeline(topic: str):
    search_tasks = [
        async_search(f"{topic} 最新进展"),
        async_search(f"{topic} 技术原理"),
        async_search(f"{topic} 市场规模"),
    ]
    results = await asyncio.gather(*search_tasks)
    return results

# 2. 流式报告输出（实时展示）
async def stream_report(topic: str):
    chain = report_prompt | llm
    async for chunk in chain.astream({"topic": topic}):
        print(chunk.content, end="", flush=True)
```

### 10.2 增强记忆

```python
# 跨会话持久化记忆（使用 Redis）
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_redis_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
        ttl=86400  # 24小时过期
    )
```

### 10.3 多模态扩展

```python
# 支持图表分析（GPT-4 Vision）
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

vision_llm = ChatOpenAI(model="gpt-4o")

def analyze_chart(image_url: str, question: str) -> str:
    msg = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": image_url}},
    ])
    return vision_llm.invoke([msg]).content
```

### 10.4 评估体系

```python
# 评估研究质量
from llama_index.core.evaluation import FaithfulnessEvaluator

def evaluate_research_quality(kb, test_questions: list) -> dict:
    evaluator = FaithfulnessEvaluator()
    engine = kb.index.as_query_engine()
    
    scores = []
    for question in test_questions:
        response = engine.query(question)
        result = evaluator.evaluate_response(response=response)
        scores.append(result.score)
    
    return {
        "avg_faithfulness": sum(scores) / len(scores),
        "num_questions": len(test_questions),
    }
```

---

## 十一、项目目录结构

```
research-assistant/
├── main.py                      # 主入口
├── orchestrator.py              # LangChain 编排层
├── .env                         # 环境变量
├── requirements.txt             # 依赖列表
│
├── tools/
│   ├── __init__.py
│   └── search_tools.py          # 搜索与数据采集工具
│
├── knowledge/
│   ├── __init__.py
│   └── rag_engine.py            # LlamaIndex RAG 知识库
│
├── agents/
│   ├── __init__.py
│   └── research_agents.py       # AutoGen 多 Agent 团队
│
├── report/
│   ├── __init__.py
│   └── generator.py             # 报告生成与导出
│
└── output/                      # 生成的报告文件
    └── research_*.md
```

---

## 十二、本周学习总结

通过本周 7 天的学习，我们完整走过了 Agent 开发的核心技术栈：

| 天 | 主题 | 核心收获 |
|----|------|---------|
| D1 | LangChain 基础 | 组件化思维、LCEL 管道、Runnable 协议 |
| D2 | LangChain 链与工具 | Tool 定义、Agent 模式、RAG 链 |
| D3 | LlamaIndex RAG | 索引类型、高级检索策略、评估体系 |
| D4 | AutoGen | 多 Agent 对话、代码执行、人机协作 |
| D5 | CrewAI | 角色设计、任务编排、团队协作流水线 |
| D6 | 框架对比 | 选型决策、组合使用策略 |
| D7 | 综合实战 | 全流程系统设计与实现 |

### 下一步学习方向

1. **LangGraph**：有状态的复杂工作流（LangChain 的进化方向）
2. **Fine-tuning**：针对特定任务微调模型，减少 prompt 依赖
3. **多模态 Agent**：集成图像、语音的 Agent 系统
4. **生产部署**：监控、限流、成本控制、A/B 测试

---

> 🎯 **核心理念**：Agent 开发不是关于用哪个框架，而是关于如何设计好的"思考-执行-观察"循环。框架是工具，设计思维才是核心。
