---
layout: default
title: "D3 · 检索策略"
render_with_liquid: false
---

# D3 · 检索策略：BM25、稠密检索与混合检索

> **最佳 RAG 检索 = 稠密检索（语义）+ 稀疏检索（关键词）混合**，各取所长。

---

## 一、BM25 稀疏检索

BM25 是经典信息检索算法，基于词频和文档频率：

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1+1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot |d|/\text{avgdl})}$$

```python
import math
from collections import Counter
from typing import List

class BM25:
    """BM25 实现"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.idf = {}
        self.tf = []
        self.avgdl = 0
    
    def fit(self, corpus: List[List[str]]) -> None:
        """训练 BM25 索引"""
        self.corpus = corpus
        n = len(corpus)
        
        # 平均文档长度
        self.avgdl = sum(len(doc) for doc in corpus) / n
        
        # 计算词频
        self.tf = [Counter(doc) for doc in corpus]
        
        # 计算 IDF
        df = Counter()
        for doc in corpus:
            for term in set(doc):
                df[term] += 1
        
        self.idf = {
            term: math.log((n - df_t + 0.5) / (df_t + 0.5) + 1)
            for term, df_t in df.items()
        }
    
    def score(self, query_terms: List[str], doc_idx: int) -> float:
        """计算 query 和文档的 BM25 分数"""
        doc = self.corpus[doc_idx]
        dl = len(doc)
        tf_doc = self.tf[doc_idx]
        
        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            
            idf_t = self.idf[term]
            tf_t = tf_doc.get(term, 0)
            
            # BM25 公式
            numerator = tf_t * (self.k1 + 1)
            denominator = tf_t + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf_t * numerator / denominator
        
        return score
    
    def search(self, query_terms: List[str], k: int = 10) -> List[tuple]:
        """返回 top-k 相关文档"""
        scores = [
            (i, self.score(query_terms, i))
            for i in range(len(self.corpus))
        ]
        return sorted(scores, key=lambda x: -x[1])[:k]


# 实际使用 rank-bm25 库
from rank_bm25 import BM25Okapi
import jieba  # 中文分词

def build_bm25_index(docs: List[str]) -> BM25Okapi:
    """构建中文 BM25 索引"""
    tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
    return BM25Okapi(tokenized_docs)

def bm25_search(bm25_index, query: str, k: int = 10):
    query_tokens = list(jieba.cut(query))
    scores = bm25_index.get_scores(query_tokens)
    top_k_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(idx, scores[idx]) for idx in top_k_idx]
```

---

## 二、混合检索（Hybrid Search）

```python
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
import torch

class HybridRetriever:
    """混合检索器：稠密 + 稀疏"""
    
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-zh-v1.5",
        sparse_weight: float = 0.3,
        dense_weight: float = 0.7,
    ):
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        
        # 稠密检索组件
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        self.embedding_model.eval()
        
        self.faiss_index = None
        self.bm25_index = None
        self.documents = []
    
    def _encode(self, texts: List[str]) -> np.ndarray:
        """编码文本为向量"""
        encoded = self.tokenizer(
            texts, max_length=512, padding=True,
            truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            output = self.embedding_model(**encoded)
        embeddings = output.last_hidden_state[:, 0, :]  # CLS
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.numpy()
    
    def build_index(self, documents: List[str]) -> None:
        """构建混合索引"""
        self.documents = documents
        
        # 稠密索引（FAISS）
        print("构建稠密索引...")
        embeddings = self._encode(documents)
        d = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(embeddings.astype(np.float32))
        
        # 稀疏索引（BM25）
        print("构建稀疏索引...")
        import jieba
        tokenized = [list(jieba.cut(doc)) for doc in documents]
        self.bm25_index = BM25Okapi(tokenized)
        
        print(f"索引构建完成，共 {len(documents)} 个文档")
    
    def search(self, query: str, k: int = 10) -> List[dict]:
        """混合检索"""
        n = len(self.documents)
        
        # 稠密检索
        query_emb = self._encode([query]).astype(np.float32)
        dense_scores, dense_indices = self.faiss_index.search(query_emb, n)
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]
        
        # 稠密分数归一化到 [0, 1]
        dense_score_map = {
            idx: (score - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-9)
            for idx, score in zip(dense_indices, dense_scores)
        }
        
        # 稀疏检索（BM25）
        import jieba
        query_tokens = list(jieba.cut(query))
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # BM25 归一化
        bm25_max = max(bm25_scores) + 1e-9
        bm25_score_map = {i: s / bm25_max for i, s in enumerate(bm25_scores)}
        
        # 加权融合（RRF 也是常用方法）
        all_indices = set(dense_score_map.keys()) | set(bm25_score_map.keys())
        combined_scores = {}
        for idx in all_indices:
            d_score = dense_score_map.get(idx, 0.0)
            s_score = bm25_score_map.get(idx, 0.0)
            combined_scores[idx] = self.dense_weight * d_score + self.sparse_weight * s_score
        
        # 排序返回 top-k
        top_k = sorted(combined_scores.items(), key=lambda x: -x[1])[:k]
        
        return [
            {
                'doc': self.documents[idx],
                'score': score,
                'dense_score': dense_score_map.get(idx, 0),
                'sparse_score': bm25_score_map.get(idx, 0),
            }
            for idx, score in top_k
        ]


# RRF（Reciprocal Rank Fusion）融合方法
def reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    k: int = 60
) -> List[tuple[int, float]]:
    """
    RRF 融合多个排序列表
    
    RRF(d) = Σ_i 1 / (k + rank_i(d))
    
    k=60 是 Cormack 2009 论文的推荐值
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    
    return sorted(scores.items(), key=lambda x: -x[1])
```

---

## 三、高级检索策略

### 3.1 HyDE（假设文档嵌入）

```python
"""
HyDE (Hypothetical Document Embeddings) - Gao et al., 2022

问题：Query 和 Document 的向量空间不对齐
  Query: "transformer attention 如何工作？"（口语化）
  Document: "Attention(Q,K,V) = softmax(QK^T/√d)V"（技术文档）

HyDE 解法：
  1. 用 LLM 生成一个假设的答案文档
  2. 用假设文档的向量做检索（文档 vs 文档，对齐更好）
"""

async def hyde_retrieve(
    query: str,
    retriever: HybridRetriever,
    llm_client,
    k: int = 5,
) -> List[dict]:
    """HyDE 检索流程"""
    
    # Step 1: LLM 生成假设文档
    hyde_prompt = f"""请根据以下问题，生成一段简短的技术文档，直接回答该问题（100字以内）：

问题：{query}

文档："""
    
    hypothetical_doc = await llm_client.chat(
        messages=[{"role": "user", "content": hyde_prompt}],
        max_tokens=150,
        temperature=0.7
    )
    
    print(f"假设文档：{hypothetical_doc[:80]}...")
    
    # Step 2: 用假设文档检索
    results = retriever.search(hypothetical_doc, k=k)
    
    return results
```

### 3.2 多路召回

```python
def multi_recall(
    query: str,
    retriever: HybridRetriever,
    llm_client,
    k: int = 5,
) -> List[dict]:
    """
    多路召回策略：
    1. 原始 query 检索
    2. 重写后的 query 检索
    3. HyDE 检索
    最后合并去重
    """
    # 原始 query
    results_1 = retriever.search(query, k=k)
    
    # Query 重写
    rewrite_prompt = f"请用不同的表达方式重写以下问题（保持语义不变）：\n{query}"
    rewritten = llm_client.chat([{"role": "user", "content": rewrite_prompt}])
    results_2 = retriever.search(rewritten, k=k)
    
    # 合并去重（按文档内容去重）
    seen_docs = set()
    merged = []
    for result in results_1 + results_2:
        doc_key = result['doc'][:50]  # 用前50字作为去重键
        if doc_key not in seen_docs:
            seen_docs.add(doc_key)
            merged.append(result)
    
    return sorted(merged, key=lambda x: -x['score'])[:k]
```

---

## 四、面试题精讲

**Q: 为什么混合检索比单一检索效果更好？**

A:
- **稠密检索（Dense）**：捕获语义相关性，能理解同义词和语义相近的表达，但对专业术语/实体名称效果差
- **稀疏检索（BM25）**：精确匹配关键词，对专业术语、代码、产品名称效果好，但无法理解语义
- **混合**：取两者之长，典型权重 dense:sparse = 7:3 或使用 RRF 融合

**Q: HyDE 的原理和适用场景？**

A: HyDE 通过 LLM 生成一个"假设答案文档"，然后用这个文档做向量检索。原理是：问题（query）和文档（document）的向量分布不同，用文档检索文档更准。适用场景：问题和文档写作风格差异大（如口语问题 vs 技术文档），但 LLM 质量和检索质量都有影响，需要验证。

---

## 小结

| 策略 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| BM25 | 关键词匹配 | 快、可解释 | 无语义理解 |
| 稠密检索 | 语义相关 | 理解同义 | 关键词弱 |
| 混合检索 | 通用场景 | 均衡效果 | 参数调优 |
| HyDE | 跨风格检索 | 对齐增强 | 依赖 LLM |
| 多路召回 | 召回率优先 | 覆盖面广 | 延迟增加 |
