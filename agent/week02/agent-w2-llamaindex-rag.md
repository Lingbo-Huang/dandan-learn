# D3 LlamaIndex 用于 RAG 的核心设计

> **学习目标**：理解 LlamaIndex 的设计哲学与核心抽象，掌握构建生产级 RAG 系统的关键技术，对比其与 LangChain 在 RAG 场景下的差异。

---

## 一、LlamaIndex 是什么？

LlamaIndex（原名 GPT Index）于 2022 年发布，专注于**数据连接与 LLM 之间的桥梁**。

如果说 LangChain 是一个"通用型 AI 应用框架"，那 LlamaIndex 更像是一个"数据密集型 LLM 应用专家"，在以下场景中有明显优势：

- 📄 **复杂文档问答**：PDF、Word、PPT、代码库等多源数据
- 🗃️ **结构化+非结构化混合查询**：文本与数据库联合检索
- 🧠 **知识图谱构建**：实体关系抽取与图谱查询
- 🔍 **高级 RAG 技术**：子问题分解、混合检索、递归检索等

### LlamaIndex vs LangChain：定位对比

| 维度 | LlamaIndex | LangChain |
|------|-----------|-----------|
| 核心专长 | 数据索引与检索（RAG） | 通用链编排与 Agent |
| 文档处理 | 丰富的 Index 类型 | 基础 Loader + VectorStore |
| RAG 高级特性 | 原生支持（子问题、重排序等） | 需要手动实现 |
| Agent 能力 | 较弱（近期加强） | 强（原生设计） |
| 学习曲线 | 稍陡（概念多） | 相对平缓 |

---

## 二、核心架构：数据流水线

LlamaIndex 的核心是一条标准化的**数据处理流水线**：

```
原始数据 → [加载] → Document → [解析/分块] → Node → [嵌入] → Index → [查询] → Response
```

### 核心抽象层次

```
┌──────────────────────────────────────┐
│         QueryEngine / ChatEngine      │  ← 用户交互层
├──────────────────────────────────────┤
│              Retriever                │  ← 检索策略层
├──────────────────────────────────────┤
│           Index (VectorIndex 等)      │  ← 索引存储层
├──────────────────────────────────────┤
│         Node (TextNode etc.)          │  ← 数据单元层
├──────────────────────────────────────┤
│            Document Loader            │  ← 数据加载层
└──────────────────────────────────────┘
```

---

## 三、快速开始：基础 RAG 流程

### 安装

```bash
pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
```

### 最简单的 RAG 示例

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 配置全局设置
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# 1. 加载文档（从目录读取所有文件）
documents = SimpleDirectoryReader("./data").load_data()
print(f"加载了 {len(documents)} 个文档")

# 2. 构建向量索引
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
)

# 3. 创建查询引擎
query_engine = index.as_query_engine(
    similarity_top_k=3,  # 检索最相关的3个节点
    response_mode="tree_summarize"  # 层级摘要
)

# 4. 查询
response = query_engine.query("什么是 LlamaIndex 的核心概念？")
print(f"答案：{response}")
print(f"\n来源节点：")
for node in response.source_nodes:
    print(f"  - 文件：{node.metadata.get('file_name')}, 相似度：{node.score:.3f}")
```

---

## 四、Document 与 Node：数据的基石

### 4.1 Document

Document 是 LlamaIndex 中最基本的数据容器：

```python
from llama_index.core import Document

# 手动创建 Document
doc = Document(
    text="LlamaIndex 是一个用于构建 LLM 应用的数据框架",
    metadata={
        "source": "官方文档",
        "author": "Jerry Liu",
        "category": "introduction",
        "date": "2024-01-01",
    },
    doc_id="doc_001",  # 自定义 ID
    excluded_llm_metadata_keys=["date"],  # 不传给 LLM 的元数据字段
)

print(f"文档 ID：{doc.doc_id}")
print(f"元数据：{doc.metadata}")
```

### 4.2 Node 与 NodeParser

NodeParser 将 Document 切割为更小的 Node（实际检索的基本单元）：

```python
from llama_index.core.node_parser import (
    SentenceSplitter,       # 按句子分割
    SemanticSplitterNodeParser,  # 语义感知分割
    HierarchicalNodeParser, # 层级分割
    MarkdownNodeParser,     # Markdown 结构分割
    CodeSplitter,           # 代码感知分割
)
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()

# 1. 句子分割器（最常用）
sentence_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separator=" ",
)
nodes = sentence_splitter.get_nodes_from_documents(documents)
print(f"共生成 {len(nodes)} 个 Node")

# 2. 语义分割器（基于嵌入相似度，效果更好但更慢）
from llama_index.embeddings.openai import OpenAIEmbedding

semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=OpenAIEmbedding(),
)
semantic_nodes = semantic_splitter.get_nodes_from_documents(documents)

# 3. 层级分割器（同时生成父子节点，用于 Small-to-Big 检索）
hierarchical_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # 从大到小的层级
)
hierarchical_nodes = hierarchical_parser.get_nodes_from_documents(documents)
```

### 4.3 Node 的元数据提取

```python
from llama_index.core.extractors import (
    TitleExtractor,      # 提取文档标题
    QuestionsAnsweredExtractor,  # 生成本节点能回答的问题
    SummaryExtractor,    # 生成摘要
    KeywordExtractor,    # 提取关键词
)
from llama_index.core.ingestion import IngestionPipeline

# 构建数据摄入流水线
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        TitleExtractor(nodes=5),
        QuestionsAnsweredExtractor(questions=3),  # 为每个节点生成3个问题
        SummaryExtractor(summaries=["prev", "self"]),
        KeywordExtractor(keywords=10),
    ]
)

nodes = await pipeline.arun(documents=documents)
# 每个 node.metadata 现在包含：title, questions_this_excerpt_can_answer, section_summary, keywords
```

---

## 五、Index 类型详解

LlamaIndex 提供多种索引类型，适应不同场景：

### 5.1 VectorStoreIndex（最常用）

```python
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb

# 使用 Chroma 作为持久化向量存储
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_docs")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 构建索引（首次运行会向量化并存储）
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True,
)

# 后续加载（无需重新向量化）
index = VectorStoreIndex.from_vector_store(vector_store)
```

### 5.2 SummaryIndex

```python
from llama_index.core import SummaryIndex

# 适合全文摘要任务，按顺序处理所有节点
summary_index = SummaryIndex.from_documents(documents)
query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize"
)
summary = query_engine.query("请为这份文档生成一个全面的摘要")
```

### 5.3 KeywordTableIndex

```python
from llama_index.core import KeywordTableIndex

# 基于关键词的倒排索引
keyword_index = KeywordTableIndex.from_documents(documents)
query_engine = keyword_index.as_query_engine(
    retriever_mode="simple"
)
```

### 5.4 KnowledgeGraphIndex（知识图谱）

```python
from llama_index.core import KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# 自动抽取实体关系，构建知识图谱
graph_store = Neo4jGraphStore(
    username="neo4j",
    password="password",
    url="bolt://localhost:7687",
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=5,  # 每个节点最多提取5个三元组
    include_embeddings=True,
)

# 图谱查询
query_engine = kg_index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",  # 混合：图谱 + 向量
    similarity_top_k=3,
)
```

---

## 六、高级检索策略

### 6.1 混合检索（Hybrid Search）

```python
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever

# 向量检索
vector_retriever = VectorIndexRetriever(
    index=vector_index,
    similarity_top_k=5
)

# BM25 关键词检索
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=5
)

# 融合检索（RRF 重排序）
from llama_index.core.retrievers import QueryFusionRetriever

hybrid_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=3,
    num_queries=4,  # 生成4个相似查询
    mode="reciprocal_rerank",  # RRF 融合
    use_async=True,
)

query_engine = RetrieverQueryEngine.from_args(hybrid_retriever)
```

### 6.2 Small-to-Big 检索

检索小块（精确）→ 返回大块（完整上下文）：

```python
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import VectorStoreIndex, StorageContext

# 构建层级节点
hierarchical_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)
all_nodes = hierarchical_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(all_nodes)  # 只有叶子节点（128字符）会被向量化

# 构建索引（只索引叶子节点）
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(all_nodes)  # 所有节点存入 docstore

leaf_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context
)

# AutoMergingRetriever：检索叶子节点后自动合并为父节点
base_retriever = leaf_index.as_retriever(similarity_top_k=6)
retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    verbose=True,
    simple_ratio_thresh=0.3,  # 30%子节点命中时合并为父节点
)
```

### 6.3 子问题分解（Sub-Question）

将复杂问题分解为多个子问题，分别检索再汇总：

```python
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

# 准备多个数据源
langchain_docs = SimpleDirectoryReader("./langchain_docs").load_data()
llamaindex_docs = SimpleDirectoryReader("./llamaindex_docs").load_data()

langchain_index = VectorStoreIndex.from_documents(langchain_docs)
llamaindex_index = VectorStoreIndex.from_documents(llamaindex_docs)

# 包装为 Tool
tools = [
    QueryEngineTool.from_defaults(
        query_engine=langchain_index.as_query_engine(),
        name="langchain_docs",
        description="LangChain 框架的文档和教程"
    ),
    QueryEngineTool.from_defaults(
        query_engine=llamaindex_index.as_query_engine(),
        name="llamaindex_docs",
        description="LlamaIndex 框架的文档和教程"
    ),
]

# 子问题查询引擎
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
    use_async=True,
)

# 一个复杂问题会被分解为多个子问题
response = await sub_question_engine.aquery(
    "LangChain 和 LlamaIndex 在 RAG 实现上有什么主要区别？各自的优势是什么？"
)
```

---

## 七、响应合成（Response Synthesis）

检索到节点后，如何组织生成最终答案：

```python
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer
)

# 不同的响应模式
modes = {
    "refine": "逐个节点递进精炼答案（最准确，最慢）",
    "compact": "压缩后一次性合成（推荐）",
    "tree_summarize": "构建树状摘要（适合长文档）",
    "simple_summarize": "直接简单摘要（最快）",
    "no_text": "只返回检索节点，不合成文字",
    "accumulate": "分别生成每个节点的答案再聚合",
}

# 自定义合成器
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,
    structured_answer_filtering=True,  # 过滤无关答案
    verbose=True,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
)
```

---

## 八、持久化存储与索引管理

```python
from llama_index.core import StorageContext, load_index_from_storage

# 保存索引到磁盘
index.storage_context.persist(persist_dir="./storage")

# 从磁盘加载（无需重新向量化）
storage_context = StorageContext.from_defaults(persist_dir="./storage")
loaded_index = load_index_from_storage(storage_context)

# 增量更新（添加新文档）
new_docs = SimpleDirectoryReader("./new_data").load_data()
for doc in new_docs:
    index.insert(doc)  # 插入新文档
    
# 删除文档
index.delete_ref_doc("doc_001", delete_from_docstore=True)
```

---

## 九、LlamaIndex Agent

LlamaIndex 也提供 Agent 功能，主要通过 `OpenAIAgent` 和 `ReActAgent`：

```python
from llama_index.core.agent import ReActAgent, OpenAIAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool

# 将 QueryEngine 包装为工具
query_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="knowledge_base",
    description="查询公司内部知识库",
)

# 自定义函数工具
def get_stock_price(ticker: str) -> str:
    """获取股票价格"""
    prices = {"AAPL": "189.5", "GOOGL": "140.2", "MSFT": "415.8"}
    return f"{ticker}: ${prices.get(ticker, '未知')}"

stock_tool = FunctionTool.from_defaults(fn=get_stock_price)

# OpenAI Agent（推荐）
agent = OpenAIAgent.from_tools(
    tools=[query_tool, stock_tool],
    llm=OpenAI(model="gpt-4o"),
    verbose=True,
    max_function_calls=5,
)

response = agent.chat("苹果公司的股价是多少？知识库里有没有相关的分析报告？")
print(response)

# 流式响应
for delta in agent.stream_chat("分析一下当前的 AI 市场趋势").response_gen:
    print(delta, end="")
```

---

## 十、生产实践建议

### 评估 RAG 质量

```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,   # 答案是否忠实于上下文
    RelevancyEvaluator,      # 答案是否与问题相关
    CorrectnessEvaluator,    # 答案是否正确（需要参考答案）
    DatasetGenerator,         # 自动生成评估数据集
)

faithfulness_evaluator = FaithfulnessEvaluator()
relevancy_evaluator = RelevancyEvaluator()

# 评估单个响应
response = query_engine.query("LlamaIndex 的核心功能是什么？")

faith_result = faithfulness_evaluator.evaluate_response(response=response)
rel_result = relevancy_evaluator.evaluate_response(
    query="LlamaIndex 的核心功能是什么？",
    response=response,
)

print(f"忠实度：{faith_result.score} - {faith_result.feedback}")
print(f"相关性：{rel_result.score} - {rel_result.feedback}")

# 批量评估
questions = ["问题1", "问题2", "问题3"]
from llama_index.core.evaluation import BatchEvalRunner

runner = BatchEvalRunner(
    {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
    workers=4,
)
eval_results = await runner.aevaluate_queries(
    query_engine,
    queries=questions,
)
```

---

## 十一、LlamaIndex 与 LangChain 在 RAG 场景的选择建议

| 场景 | 推荐框架 | 原因 |
|------|---------|------|
| 快速构建简单 RAG | LangChain | 上手更快，生态更广 |
| 多文档复杂问答 | LlamaIndex | 更丰富的索引类型 |
| 需要子问题分解 | LlamaIndex | 原生内置支持 |
| RAG + 复杂 Agent | LangChain | Agent 能力更强 |
| 知识图谱集成 | LlamaIndex | 原生 KG 支持 |
| 混合检索（向量+BM25） | LlamaIndex | 更成熟的实现 |
| 生产评估体系 | LlamaIndex | 内置完整评估工具 |

---

## 小结

LlamaIndex 的核心优势在于对数据处理的精细控制：
1. **丰富的 NodeParser**：语义感知、层级结构、代码感知
2. **多种 Index 类型**：向量、关键词、知识图谱
3. **高级检索策略**：子问题分解、Small-to-Big、混合检索
4. **内置评估体系**：忠实度、相关性、正确性评估

下一篇（D4）将介绍 AutoGen —— 一个专注于多 Agent 协作的框架。
