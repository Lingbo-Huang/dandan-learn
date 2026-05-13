---
layout: default
title: "D4 · RAG：检索增强生成"
---

# D4 · RAG：检索增强生成

> **Agent Week 3**  
> RAG 是目前最主流的 LLM 知识增强方案：让模型"带着文档"回答问题。

---

## 一、为什么需要 RAG？

| 问题 | 说明 |
|------|------|
| **知识截止日期** | 模型训练有时间截止，不知道最新信息 |
| **私有知识** | 公司内部文档，模型没见过 |
| **幻觉** | 模型可能"创造"不存在的事实 |
| **长文档** | 超过上下文窗口的文档无法直接输入 |

**RAG 的思路**：先用向量检索找到相关文档片段，再让 LLM 基于这些片段回答。

---

## 二、RAG 的完整流程

```
文档处理（离线）：
  原始文档 → 分割（Chunking）→ 嵌入（Embedding）→ 存入向量数据库

查询处理（在线）：
  用户问题 → 嵌入 → 向量检索 → 找到相关片段
                                      ↓
                            构建 Prompt（问题 + 相关片段）→ LLM → 回答
```

---

## 三、文档加载与分割

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, PDFMinerLoader, WebBaseLoader
)

# 加载各种格式的文档
# text_loader = TextLoader("document.txt", encoding="utf-8")
# pdf_loader = PDFMinerLoader("paper.pdf")
web_loader = WebBaseLoader("https://example.com/article")

# 文档分割（Chunking）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每个片段约 500 字符
    chunk_overlap=50,     # 片段间重叠 50 字符（保证上下文连续性）
    separators=["\n\n", "\n", "。", "！", "？", " "],  # 优先在哪里分割
)

# 模拟一篇文章
from langchain_core.documents import Document

docs = [Document(page_content="""
LangChain 是一个用于开发 LLM 应用的框架，由 Harrison Chase 于 2022 年创建。
它提供了标准化的接口来与各种语言模型交互。

核心组件包括：
- Models：支持 OpenAI、Anthropic、本地模型等
- Prompts：模板化 Prompt 管理
- Chains：组合多个步骤
- Agents：动态工具调用
- Memory：对话历史管理
- Retrievers：文档检索

LCEL（LangChain Expression Language）是 LangChain 0.1 引入的声明式语法，
用 | 运算符将各组件串联，支持流式输出、并行执行和异步操作。
""", metadata={"source": "langchain_intro.txt"})]

splits = splitter.split_documents(docs)
print(f"原始文档 {len(docs)} 篇 → 分割后 {len(splits)} 个片段")
for i, split in enumerate(splits):
    print(f"\n片段 {i+1} ({len(split.page_content)} 字符):")
    print(split.page_content[:100] + "...")
```

---

## 四、向量嵌入与存储

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

# 嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 构建向量数据库
# 方案一：Chroma（持久化，适合开发）
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 本地持久化
)

# 方案二：FAISS（高性能，适合生产）
# vectorstore = FAISS.from_documents(splits, embeddings)
# vectorstore.save_local("faiss_index")  # 保存索引
# vectorstore = FAISS.load_local("faiss_index", embeddings)  # 加载

# 测试检索
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 返回最相似的 3 个片段
)

results = retriever.invoke("LCEL 是什么？")
for doc in results:
    print(doc.page_content[:100])
    print("---")
```

---

## 五、构建完整 RAG Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# RAG Prompt 模板
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识助手。请基于以下检索到的上下文回答问题。
    
如果上下文中没有足够信息，请说"根据现有文档无法回答"，不要凭空编造。

上下文：
{context}
"""),
    ("human", "{question}"),
])

def format_docs(docs):
    """将检索到的文档格式化为字符串"""
    return "\n\n---\n\n".join(
        f"来源：{doc.metadata.get('source', '未知')}\n{doc.page_content}"
        for doc in docs
    )

# 完整 RAG 链
rag_chain = (
    {
        "context": retriever | format_docs,  # 检索并格式化
        "question": RunnablePassthrough()    # 原问题透传
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 使用
answer = rag_chain.invoke("LangChain 是什么时候创建的？")
print(answer)
```

---

## 六、高级 RAG：带来源引用

```python
from langchain_core.runnables import RunnableParallel

# 同时返回答案和来源
rag_chain_with_sources = RunnableParallel({
    "answer": rag_chain,
    "sources": retriever | (lambda docs: [d.metadata.get("source") for d in docs])
})

result = rag_chain_with_sources.invoke("什么是 LCEL？")
print(f"答案：{result['answer']}")
print(f"来源：{result['sources']}")
```

---

## 七、RAG 的常见问题与优化

| 问题 | 解决方案 |
|------|---------|
| 检索不准确 | 改进 Chunking 策略；使用 Hybrid Search（语义+关键词） |
| 上下文太长 | Reranker（重排序，只保留最相关的片段） |
| 幻觉 | 提示词加强"只基于文档回答"；引用来源 |
| 多跳推理 | Multi-hop RAG：先找主题文档，再深入检索 |

---

## 今天的关键认识

1. **RAG = 检索 + 生成**：先找相关文档，再让 LLM 基于文档回答
2. **Chunking**：文档分割的粒度很关键，片段太大或太小都影响检索质量
3. **向量相似度搜索**：把查询和文档都嵌入向量空间，找最近邻
4. **`format_docs`**：把检索结果格式化为 LLM 能理解的上下文

---

## 明天预告

D5：**Tool 与 ToolKit**——让 Agent 能调用外部工具，从"知道"到"做到"。
