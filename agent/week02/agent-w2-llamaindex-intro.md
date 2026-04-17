# LlamaIndex：以数据为核心的 RAG 框架

> **Day 3** · 预计学习时间：4-5 小时  
> **目标**：理解 LlamaIndex 与 LangChain 的定位差异，构建本地文档问答系统

---

## 框架概念

### LlamaIndex 是什么？

LlamaIndex（原 GPT Index）是专注于**数据连接与检索增强**的 AI 框架，核心哲学是：

> 让 LLM 能"看到"并"理解"你私有的数据。

**与 LangChain 的核心区别：**

| 维度 | LangChain | LlamaIndex |
|------|-----------|------------|
| **核心关注** | 构建通用 AI 应用链路 | 数据摄取、索引、检索 |
| **最强领域** | Agent、对话、工具编排 | RAG、文档问答、数据查询 |
| **学习曲线** | 中等（概念多） | 相对平缓（专注数据流） |
| **典型用例** | 多步骤 Agent 工作流 | "问我的文档"类应用 |

**不是竞争，是互补：** LlamaIndex 做数据层，LangChain 做应用层——两者经常组合使用。

### 核心概念体系

```
数据流动路径：
Raw Data → Document → Node → Index → QueryEngine → Response

Document  ：原始数据的封装（PDF、网页、数据库记录等）
Node      ：Document 切分后的最小单元（有 metadata、relationships）
Index     ：将 Node 组织成可高效检索的结构
Retriever ：从 Index 中检索相关 Node
Synthesizer：将检索结果 + 问题合成为最终答案
QueryEngine：Retriever + Synthesizer 的组合，一站式问答
```

### 三种核心 Index 类型

| Index 类型 | 底层结构 | 适用场景 |
|-----------|---------|---------|
| `VectorStoreIndex` | 向量相似度 | 语义搜索（最常用） |
| `SummaryIndex` | 顺序摘要链 | 需要全文综合的问题 |
| `KnowledgeGraphIndex` | 知识图谱 | 实体关系推理 |

---

## 核心代码示例（使用 uv）

### 环境准备

```bash
uv init llamaindex-demo
cd llamaindex-demo

# 核心包
uv add llama-index llama-index-llms-openai llama-index-embeddings-openai python-dotenv

# 可选：本地嵌入（省钱）
uv add llama-index-embeddings-huggingface

echo 'OPENAI_API_KEY=your-key-here' > .env
```

### 示例 1：最简单的本地文档问答

```python
# simple_qa.py
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

# 全局配置
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 1. 加载文档（会递归读取目录下所有文件）
documents = SimpleDirectoryReader("./docs").load_data()
print(f"加载了 {len(documents)} 个文档")

# 2. 构建索引（自动切分 + 嵌入向量）
index = VectorStoreIndex.from_documents(documents)

# 3. 创建问答引擎
query_engine = index.as_query_engine()

# 4. 提问
response = query_engine.query("这些文档的主要内容是什么？")
print("回答：", response)
print("来源节点数：", len(response.source_nodes))
```

### 示例 2：完整 RAG 流水线（精细控制）

```python
# rag_pipeline.py
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

PERSIST_DIR = "./storage"

def build_or_load_index():
    """构建索引或从缓存加载（避免重复嵌入）"""
    if os.path.exists(PERSIST_DIR):
        print("从缓存加载索引...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)
    
    print("构建新索引...")
    documents = SimpleDirectoryReader("./docs").load_data()
    
    # 自定义切分策略
    splitter = SentenceSplitter(
        chunk_size=512,      # 每个 chunk 最多 512 tokens
        chunk_overlap=50,    # 相邻 chunk 重叠 50 tokens（保持上下文连续性）
    )
    
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter]
    )
    
    # 持久化到磁盘
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

# 构建/加载索引
index = build_or_load_index()

# 精细化检索配置
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,  # 检索 top 5 个相关片段
)

# 相似度过滤器（过滤低相关度结果）
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

# 组装 QueryEngine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[postprocessor],
)

def ask(question: str):
    response = query_engine.query(question)
    
    print(f"\n❓ 问题：{question}")
    print(f"💡 回答：{response}\n")
    
    print("📚 引用来源：")
    for i, node in enumerate(response.source_nodes):
        print(f"  [{i+1}] 相似度：{node.score:.3f}")
        print(f"       内容：{node.text[:100]}...\n")

ask("请总结文档中的核心观点")
ask("文档中提到了哪些具体的数字或数据？")
```

### 示例 3：多文档路由（不同问题路由到不同索引）

```python
# multi_index_router.py
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleDirectoryReader, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 假设有两类文档：技术文档 vs 业务文档
tech_docs = SimpleDirectoryReader("./docs/tech").load_data()
biz_docs = SimpleDirectoryReader("./docs/business").load_data()

# 为不同文档类型建不同索引
tech_index = VectorStoreIndex.from_documents(tech_docs)
biz_index = VectorStoreIndex.from_documents(biz_docs)

# 封装为工具，并描述适用场景
tech_tool = QueryEngineTool.from_defaults(
    query_engine=tech_index.as_query_engine(),
    description="用于回答技术实现、代码架构、API 设计相关问题"
)

biz_tool = QueryEngineTool.from_defaults(
    query_engine=biz_index.as_query_engine(),
    description="用于回答业务策略、市场分析、用户需求相关问题"
)

# 路由引擎：根据问题自动选择合适的索引
router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[tech_tool, biz_tool],
    verbose=True  # 显示路由决策
)

# 测试路由
print("=== 技术问题 ===")
r1 = router_engine.query("系统的微服务架构是怎么设计的？")
print(r1)

print("\n=== 业务问题 ===")
r2 = router_engine.query("我们的核心目标用户群体是什么？")
print(r2)
```

### 示例 4：使用本地嵌入（省钱方案）

```bash
# 安装本地嵌入模型支持
uv add llama-index-embeddings-huggingface sentence-transformers
```

```python
# local_embedding.py
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# 使用本地嵌入（首次运行会下载模型，约 500MB）
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"  # 中文嵌入模型
)
Settings.llm = OpenAI(model="gpt-4o-mini")  # LLM 仍用 OpenAI

documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("这篇文档讲了什么？")
print(response)
# 注意：嵌入是免费的，只有 LLM 调用收费
```

---

## 与 Claw 的对比与联系

**LlamaIndex 的数据流 vs Claw 的文件传输：**

| 环节 | LlamaIndex | Claw |
|------|-----------|------|
| **数据摄入** | `SimpleDirectoryReader` 加载本地文件 | `GET /claw/room/files/{fileId}/download` 下载平台文件 |
| **数据索引** | `VectorStoreIndex` 建向量索引 | 平台内置文件管理（不做向量化） |
| **数据检索** | `QueryEngine` 语义检索 | Task 间通过 summary + fileId 传递结果 |
| **持久化** | 向量数据库 / 本地磁盘 | 平台文件系统 |

**实际应用场景：**
- 用户在 Claw 中上传一批 PDF 文档 → Worker Agent 用 LlamaIndex 构建向量索引 → 返回文档问答能力
- LlamaIndex 负责"怎么检索数据"，Claw 负责"谁来调用这个检索能力、什么时候调用"

**一个真实的 Claw + LlamaIndex 协作流程：**
```
用户上传 PDF → 
Claw Task: 下载文件 → LlamaIndex 建索引 → 持久化存储 → 返回索引路径 →
Claw Task: 接收问题 → 加载索引 → LlamaIndex QueryEngine → 返回答案
```

---

## 小结

- **LlamaIndex 最擅长的场景**：私有文档问答、知识库检索、多文档综合分析
- **VectorStoreIndex** 是最常用的索引类型，语义搜索质量高，但有 API 成本
- **持久化索引**（`storage_context.persist`）是生产必备，避免每次启动重新嵌入
- **chunk_size 和 chunk_overlap 的调参**：chunk 太大信息太稀疏，太小上下文断裂——通常 256-512 是合理起点
- **路由引擎**是构建多知识库系统的利器，让 LLM 自动选择最合适的数据源

**下一步** → Day 4：AutoGen——微软出品的对话式多 Agent 协作框架
