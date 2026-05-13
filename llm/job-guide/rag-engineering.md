---
layout: post
title: "RAG 全链路工程化"
track: "🤖 大模型"
---

# RAG 全链路工程化

> RAG（Retrieval-Augmented Generation）是2026年需求量最大的核心技能。从Demo到生产级系统，差距在工程细节。

---

## 为什么需要 RAG？

大模型的三大原生缺陷：
1. **知识截止**：训练数据有时效性，不知道最新信息
2. **幻觉**：模型会"编造"看似合理但错误的答案
3. **上下文限制**：无法处理超长私有文档

RAG 的核心思路：**检索相关文档 → 注入上下文 → 生成有据可查的答案**

---

## 生产级 RAG 全流程

```
原始数据
  ↓
文档解析（PDF/Word/网页/Markdown）
  ↓
数据清洗（去噪/去重/格式化）
  ↓
语义分块（策略选择）
  ↓
向量化（Embedding模型）
  ↓
向量库索引（Chroma/Milvus/pgvector）
  ↓
[查询时]
用户问题 → 查询向量化 → 混合检索（BM25+向量）→ 重排序 → 上下文压缩
  ↓
注入生成（Prompt + 检索结果 → LLM）
  ↓
后处理（事实校验/幻觉检测/格式规整）
  ↓
评估迭代（召回率/精准率/事实一致性）
```

---

## 1. 文档解析

```python
from langchain.document_loaders import (
    PyPDFLoader,          # PDF
    Docx2txtLoader,       # Word
    UnstructuredMarkdownLoader,  # Markdown
    WebBaseLoader,        # 网页
    CSVLoader,            # CSV
)
from pathlib import Path

def load_document(file_path: str) -> list:
    """根据文件类型选择合适的加载器"""
    suffix = Path(file_path).suffix.lower()
    
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".md": UnstructuredMarkdownLoader,
        ".csv": CSVLoader,
    }
    
    if suffix not in loaders:
        raise ValueError(f"不支持的文件类型: {suffix}")
    
    loader = loaders[suffix](file_path)
    docs = loader.load()
    
    # 添加元数据
    for doc in docs:
        doc.metadata["source"] = file_path
        doc.metadata["file_type"] = suffix
    
    return docs
```

---

## 2. 语义分块（关键！直接影响检索质量）

### 策略对比

| 策略 | 适用场景 | 优缺点 |
|------|---------|--------|
| 固定长度分块 | 格式规整文档 | 简单但可能切断语义 |
| 递归字符分块 | 通用文档（推荐） | 按段落→句子→字符递归，保持语义完整 |
| 语义分块 | 高质量要求 | 按语义相似度分块，效果好但慢 |
| 父文档检索 | 长文档 | 小块检索+大块生成，兼顾精准和上下文 |

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 推荐配置（通用）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,           # 每块最大字符数
    chunk_overlap=50,         # 块间重叠（保持上下文连贯）
    separators=[              # 分隔符优先级：段落>句子>单词>字符
        "\n\n",   # 段落
        "\n",     # 换行
        "。",     # 中文句号
        "！",
        "？",
        " ",
        ""
    ],
    length_function=len,
)

chunks = text_splitter.split_documents(docs)
print(f"分块数量: {len(chunks)}, 平均长度: {sum(len(c.page_content) for c in chunks)//len(chunks)}")
```

### 父文档检索（生产推荐）

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# 小块用于检索（精准），大块用于生成（上下文完整）
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

store = InMemoryStore()  # 生产环境换成Redis
retriever = ParentDocumentRetriever(
    vectorstore=vector_db,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(docs)
```

---

## 3. Embedding 模型选型

| 模型 | 语言 | 维度 | 推荐场景 |
|------|------|------|---------|
| BAAI/bge-small-zh-v1.5 | 中文 | 512 | 中文业务（轻量） |
| BAAI/bge-large-zh-v1.5 | 中文 | 1024 | 中文业务（高质量） |
| text-embedding-3-small | 多语言 | 1536 | OpenAI API（成本低） |
| text-embedding-3-large | 多语言 | 3072 | OpenAI API（高质量） |
| BAAI/bge-m3 | 多语言 | 1024 | 多语言混合场景 |

```python
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},  # 有GPU改为"cuda"
    encode_kwargs={"normalize_embeddings": True}  # 归一化，提升余弦相似度效果
)

# 测试
test_embedding = embedding_model.embed_query("什么是大模型？")
print(f"向量维度: {len(test_embedding)}")  # 512
```

---

## 4. 向量库：从开发到生产

```python
# 开发阶段：Chroma（本地，零配置）
from langchain.vectorstores import Chroma

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
vector_db.persist()

# 生产阶段：pgvector（PostgreSQL，事务支持）
from langchain.vectorstores import PGVector

CONNECTION_STRING = "postgresql://user:pass@localhost:5432/llmdb"
vector_db = PGVector.from_documents(
    documents=chunks,
    embedding=embedding_model,
    connection_string=CONNECTION_STRING,
    collection_name="knowledge_base"
)
```

---

## 5. 混合检索（生产必备）

**纯向量检索**：擅长语义相似，但对关键词精确匹配差  
**BM25关键词检索**：擅长关键词匹配，但不理解语义  
**混合检索**：两者结合，取长补短

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# 向量检索器
vector_retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# BM25检索器
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# 混合：向量权重0.7，BM25权重0.3
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

results = ensemble_retriever.invoke("产品核心功能是什么")
```

---

## 6. 重排序（Reranker）

检索到5-10个候选文档后，用 Reranker 精确排序，选出最相关的2-3个。

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 加载 Reranker 模型
reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker, top_n=3)

# 组合：先粗检索10个，再精排取3个
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever  # 先检索10个
)

final_docs = compression_retriever.invoke("用户问题")
```

---

## 7. 完整 RAG 流水线

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 自定义Prompt（关键！防止幻觉）
RAG_PROMPT = PromptTemplate(
    template="""你是专业的知识库助手。请严格基于以下上下文回答问题。
如果上下文中没有相关信息，请明确说"根据现有资料无法回答该问题"，不要编造答案。

上下文：
{context}

问题：{question}

回答：""",
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)  # temperature=0 减少幻觉

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    chain_type_kwargs={"prompt": RAG_PROMPT},
    return_source_documents=True  # 返回引用来源
)

# 查询
result = qa_chain.invoke({"query": "企业产品的核心功能是什么？"})
print("回答：", result["result"])
print("来源：", [doc.metadata["source"] for doc in result["source_documents"]])
```

---

## 8. RAG 评估（面试必问）

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # 忠实度：回答是否基于上下文
    answer_relevancy,    # 相关性：回答是否回答了问题
    context_recall,      # 召回率：相关信息是否被检索到
    context_precision,   # 精准率：检索结果是否相关
)
from datasets import Dataset

# 准备评估数据
eval_data = {
    "question": ["产品核心功能？", "定价策略？"],
    "answer": [rag_answers[0], rag_answers[1]],
    "contexts": [retrieved_docs[0], retrieved_docs[1]],
    "ground_truth": ["正确答案1", "正确答案2"]
}
dataset = Dataset.from_dict(eval_data)

# 评估
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision]
)
print(results)
```

**面试答法**：
> RAG 效果差的常见原因：①分块粒度不对（太大太小都影响检索）②Embedding模型与业务域不匹配③缺少重排序④Prompt没有明确限制幻觉⑤缺少评估反馈闭环

---

## 9. 幻觉抑制技巧

1. **Prompt 约束**：明确说"只基于给定上下文回答，没有则说不知道"
2. **temperature=0**：减少随机性
3. **引用来源**：要求模型引用文档片段，便于核查
4. **事实一致性检测**：用另一个LLM验证回答与文档的一致性
5. **置信度过滤**：检索分数低于阈值时拒绝回答

```python
def answer_with_confidence(query: str, threshold: float = 0.7) -> str:
    """带置信度过滤的RAG"""
    docs_with_scores = vector_db.similarity_search_with_score(query, k=3)
    
    # 过滤低分文档
    high_confidence_docs = [
        doc for doc, score in docs_with_scores
        if score >= threshold
    ]
    
    if not high_confidence_docs:
        return "抱歉，知识库中暂无相关信息，建议联系专业人员。"
    
    return qa_chain.invoke({"query": query})["result"]
```

---

[← Python工程栈](./python-engineering) | [→ Agent开发实战](./agent-development)
