---
layout: default
title: "W6D3 · 向量数据库实战"
---

# 向量数据库：Agent 的语义记忆引擎

> **Week 6 · Day 3** | 难度：⭐⭐⭐⭐

---

## 向量数据库的核心概念

```
传统数据库：精确匹配
  SELECT * WHERE name = 'Python'  → 只找"Python"

向量数据库：语义搜索
  search("AI编程语言") → 找到"Python"、"Julia"、"R" 等语义相关的文档
```

文本 → Embedding 模型 → 向量（如 [0.23, -0.45, 0.12, ...]）→ 存入向量数据库  
查询时：查询词 → 向量 → 在数据库中找最近的向量 → 返回相关文档

## ChromaDB：本地向量数据库

```bash
pip install chromadb langchain-chroma
```

```python
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from typing import List
import os

# ── 初始化 ──

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 持久化存储（重启后数据不丢失）
vectorstore = Chroma(
    collection_name="agent_memory",
    embedding_function=embeddings,
    persist_directory="./chroma_db"  # 数据存在本地
)

# ── 添加文档 ──

def add_documents_from_texts(texts: List[str], metadatas: List[dict] = None):
    """将文本列表添加到向量数据库"""
    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas or [{}] * len(texts)
    )
    print(f"已添加 {len(texts)} 条文档")

# 添加知识库
knowledge_base = [
    "Python 是一种解释型高级编程语言，以简洁的语法和丰富的库生态系统著称",
    "LangChain 是构建 LLM 应用的框架，提供 Chain、Agent、Memory 等核心组件",
    "向量数据库通过存储高维向量实现语义搜索，常用的有 Chroma、Pinecone、Weaviate",
    "RAG（检索增强生成）将知识库检索与 LLM 生成结合，提高答案准确性",
    "Transformer 架构是现代大语言模型的基础，由 Attention 机制驱动",
    "微调（Fine-tuning）可以让 LLM 在特定领域表现更好，但成本较高",
    "提示工程（Prompt Engineering）通过设计输入提示优化 LLM 输出",
    "Agent 是能够感知环境、做决策、执行行动的 AI 系统",
]

metadatas = [
    {"category": "programming", "topic": "Python"},
    {"category": "framework", "topic": "LangChain"},
    {"category": "database", "topic": "vector_db"},
    {"category": "technique", "topic": "RAG"},
    {"category": "model", "topic": "Transformer"},
    {"category": "technique", "topic": "fine_tuning"},
    {"category": "technique", "topic": "prompt_engineering"},
    {"category": "concept", "topic": "Agent"},
]

add_documents_from_texts(knowledge_base, metadatas)

# ── 语义搜索 ──

def semantic_search(query: str, k: int = 3, filter_dict: dict = None) -> List:
    """语义搜索"""
    if filter_dict:
        results = vectorstore.similarity_search(
            query, k=k, filter=filter_dict
        )
    else:
        results = vectorstore.similarity_search(query, k=k)
    
    return results

# 测试搜索
print("查询：'如何构建AI应用框架'")
results = semantic_search("如何构建AI应用框架", k=3)
for i, doc in enumerate(results, 1):
    print(f"  {i}. {doc.page_content}")

print("\n查询：'数据库存储'（限定 database 类别）")
results = semantic_search("数据库存储", k=2, filter_dict={"category": "database"})
for doc in results:
    print(f"  - {doc.page_content}")

# ── 相似度分数搜索 ──

def search_with_scores(query: str, k: int = 3, score_threshold: float = 0.5):
    """带相似度分数的搜索（过滤低质量结果）"""
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    # 过滤低于阈值的结果
    filtered = [(doc, score) for doc, score in results if score <= score_threshold]
    
    return filtered

results_with_scores = search_with_scores("机器学习模型训练", k=5)
for doc, score in results_with_scores:
    print(f"  相似度：{1-score:.2f} - {doc.page_content[:50]}")
```

## 文档分割与索引

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    CSVLoader
)
from pathlib import Path

class DocumentIndexer:
    """文档索引器：支持多种格式"""
    
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,          # 每块最多500字符
            chunk_overlap=50,        # 相邻块重叠50字符（保证上下文连贯）
            length_function=len,
            separators=["\n\n", "\n", "。", "，", " ", ""]  # 中文优先按句子分割
        )
    
    def index_text(self, text: str, metadata: dict = None) -> int:
        """索引纯文本"""
        chunks = self.splitter.split_text(text)
        metadatas = [dict(metadata or {}, chunk_index=i) for i in range(len(chunks))]
        self.vectorstore.add_texts(chunks, metadatas=metadatas)
        return len(chunks)
    
    def index_file(self, filepath: str) -> int:
        """索引文件（自动识别格式）"""
        path = Path(filepath)
        
        if path.suffix == ".pdf":
            loader = PyPDFLoader(filepath)
        elif path.suffix == ".txt" or path.suffix == ".md":
            loader = TextLoader(filepath, encoding="utf-8")
        elif path.suffix == ".csv":
            loader = CSVLoader(filepath)
        else:
            raise ValueError(f"不支持的文件格式：{path.suffix}")
        
        docs = loader.load()
        split_docs = self.splitter.split_documents(docs)
        
        # 添加文件元数据
        for doc in split_docs:
            doc.metadata["source_file"] = path.name
        
        self.vectorstore.add_documents(split_docs)
        return len(split_docs)
    
    def index_directory(self, directory: str, extensions: List[str] = None) -> int:
        """批量索引目录"""
        extensions = extensions or [".txt", ".md", ".pdf"]
        total = 0
        
        for path in Path(directory).rglob("*"):
            if path.suffix in extensions:
                try:
                    count = self.index_file(str(path))
                    total += count
                    print(f"  已索引：{path.name}（{count} 块）")
                except Exception as e:
                    print(f"  跳过 {path.name}：{e}")
        
        return total

# 使用
indexer = DocumentIndexer(vectorstore)
count = indexer.index_text(
    "Agent 的核心能力包括：规划（Planning）、工具调用（Tool Use）、"
    "记忆（Memory）和反思（Reflection）。这四个能力共同构成了生产级 Agent 系统。",
    metadata={"topic": "Agent", "type": "definition"}
)
print(f"已索引 {count} 块")
```

## 构建 RAG Pipeline

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

class RAGMemoryAgent:
    """基于 RAG 的记忆 Agent"""
    
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # 构建检索 QA 链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # 将所有检索结果塞入 prompt
            retriever=vectorstore.as_retriever(
                search_type="mmr",  # 最大边际相关性（避免重复）
                search_kwargs={"k": 5, "fetch_k": 20}
            ),
            return_source_documents=True,
            verbose=True
        )
    
    def query(self, question: str) -> dict:
        """查询知识库"""
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [
                doc.page_content[:100] 
                for doc in result.get("source_documents", [])
            ]
        }
    
    def add_memory(self, content: str, tags: List[str] = None):
        """添加新记忆到知识库"""
        metadata = {"type": "memory", "tags": str(tags or [])}
        self.vectorstore.add_texts([content], metadatas=[metadata])

rag_agent = RAGMemoryAgent(vectorstore)
result = rag_agent.query("LangChain 有哪些核心组件？")
print(f"答案：{result['answer']}")
print(f"来源：{result['sources']}")
```

## Pinecone：生产级云向量数据库

```python
# pip install pinecone-client langchain-pinecone

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# 初始化 Pinecone
pc = Pinecone(api_key="your-pinecone-api-key")

# 创建索引（只需一次）
index_name = "agent-memory"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # text-embedding-3-small 的维度
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# 使用 Pinecone 作为向量存储
pinecone_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace="production"  # 不同命名空间隔离不同用户的数据
)

# API 与 Chroma 基本相同
pinecone_store.add_texts(["Pinecone 是生产级向量数据库"])
results = pinecone_store.similarity_search("向量数据库推荐", k=3)
```

## 性能优化技巧

```python
# 1. 批量添加（比逐条快 10x）
def batch_add(texts: List[str], batch_size: int = 100):
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vectorstore.add_texts(batch)
        print(f"进度：{min(i+batch_size, len(texts))}/{len(texts)}")

# 2. 缓存 Embedding（避免重复计算）
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("./embedding_cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=OpenAIEmbeddings(),
    document_embedding_cache=store,
    namespace=OpenAIEmbeddings().model
)

# 3. 混合搜索（向量 + 关键词）
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25_retriever = BM25Retriever.from_texts(knowledge_base)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 70% 向量搜索 + 30% 关键词
)
```

## 踩坑经验

### 坑1：分块太大——检索到无关内容

**问题**：chunk_size=2000 时，一个块包含多个话题，语义搜索会返回不相关部分。  
**经验**：中文场景 chunk_size=300-500 通常效果最好，overlap=50-100。

### 坑2：embedding 模型和 chunk 语言不匹配

**问题**：用 `text-embedding-ada-002`（英文优化）处理中文，效果很差。  
**解法**：中文内容用 `text-embedding-3-small`（多语言）或中文专用模型。

### 坑3：向量数据库冷启动慢

**问题**：第一次查询时 embedding 计算很慢，用户体验差。  
**解法**：预热：启动时执行一次虚拟查询，触发模型加载。

---

*W6D3 · 向量数据库实战 | Agent + Claw 系列*
