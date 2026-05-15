---
layout: default
title: "D4 · Reranker"
render_with_liquid: false
---

# D4 · Reranker：精排提升 RAG 质量

> **两阶段检索**：先用 Bi-Encoder（快速召回 top-50），再用 Cross-Encoder（精准重排 top-10）。这是 RAG 系统的标配优化。

---

## 一、为什么需要 Reranker？

### 1.1 Bi-Encoder 的局限

Embedding 模型（Bi-Encoder）：
- Query 和 Document **分别**编码，然后计算向量相似度
- 优点：Document 可以离线预计算，查询快（O(log n)）
- 缺点：两个向量之间没有**交叉注意力**，无法捕获细粒度匹配信息

### 1.2 Cross-Encoder 的优势

Reranker（Cross-Encoder）：
- Query 和 Document **拼接**后一起输入模型
- 模型内部可以做 Query-Document 交叉注意力
- 精度远高于 Bi-Encoder，但只能逐对打分（不能向量化索引）

---

## 二、Cross-Encoder 原理

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class CrossEncoderReranker:
    """Cross-Encoder 重排序器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def predict_scores(
        self,
        query: str,
        documents: list[str],
        batch_size: int = 32,
    ) -> list[float]:
        """
        计算 query 和每个 document 的相关性分数
        
        输入格式：[CLS] query [SEP] document [SEP]
        """
        all_scores = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # 拼接 query 和 document
            pairs = [[query, doc] for doc in batch_docs]
            
            encoded = self.tokenizer(
                pairs,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                scores = outputs.logits.squeeze(-1)
                # Cross-Encoder 输出 logit（越大越相关）
                scores = torch.sigmoid(scores)  # 转换到 [0, 1]
            
            all_scores.extend(scores.tolist())
        
        return all_scores
    
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """重排并返回 top-k"""
        scores = self.predict_scores(query, documents)
        
        ranked = sorted(
            enumerate(zip(documents, scores)),
            key=lambda x: -x[1][1]
        )[:top_k]
        
        return [
            {
                'doc': doc,
                'score': score,
                'original_rank': orig_idx,
                'new_rank': new_rank
            }
            for new_rank, (orig_idx, (doc, score)) in enumerate(ranked)
        ]


# 完整的 RAG + Reranker 流程
class RAGWithReranker:
    """带 Reranker 的完整 RAG 系统"""
    
    def __init__(self, retriever, reranker, llm_client):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm_client
    
    async def answer(
        self,
        query: str,
        recall_k: int = 20,  # 一阶段召回数
        rerank_k: int = 5,   # 二阶段精排后用于生成的文档数
    ) -> dict:
        """完整 RAG 流程"""
        
        # 阶段 1：粗召回（快速）
        recalled_docs = self.retriever.search(query, k=recall_k)
        docs = [r['doc'] for r in recalled_docs]
        
        # 阶段 2：精排（准确）
        reranked = self.reranker.rerank(query, docs, top_k=rerank_k)
        
        # 打印重排效果
        print(f"重排前 top-3:")
        for i, doc in enumerate(recalled_docs[:3]):
            print(f"  [{i+1}] score={doc['score']:.3f}: {doc['doc'][:50]}")
        
        print(f"\n重排后 top-3:")
        for r in reranked[:3]:
            print(f"  [原第{r['original_rank']+1}位] score={r['score']:.3f}: {r['doc'][:50]}")
        
        # 阶段 3：生成答案
        context = "\n\n".join(
            f"[文档{i+1}]: {r['doc']}"
            for i, r in enumerate(reranked)
        )
        
        prompt = f"""请根据以下文档内容回答用户问题。只使用文档中的信息，不要添加额外内容。

{context}

用户问题：{query}

回答："""
        
        answer = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        
        return {
            'query': query,
            'answer': answer,
            'sources': [r['doc'][:100] for r in reranked],
        }
```

---

## 三、ColBERT：延迟交互

```python
"""
ColBERT (Khattab & Zaharia, 2020) 是 Bi-Encoder 和 Cross-Encoder 的折中：

Bi-Encoder：[q] → q_emb; [d] → d_emb; score = dot(q_emb, d_emb)
Cross-Encoder：[q, d] → score（精确但慢）
ColBERT：[q] → q_embs (token级); [d] → d_embs (token级); 
         score = Σ_t max_j sim(q_t, d_j)   ← 最大相似度求和

优点：
- Document 向量可以离线计算和压缩（比 Cross-Encoder 快）
- 比 Bi-Encoder 精度高（token 级别的细粒度匹配）
- 可以用于端到端检索（ColBERTv2）

BGE-M3 也支持 ColBERT 风格的多向量检索
"""

from FlagEmbedding import BGEM3FlagModel

# BGE-M3：同时支持 dense + sparse + colbert
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

sentences = ["什么是大语言模型？", "LLM 即大规模语言模型，是基于 Transformer..."]

# Dense embedding
embeddings = model.encode(
    sentences,
    batch_size=12,
    max_length=8192,
)['dense_vecs']

# ColBERT embedding（每个 token 一个向量）
output = model.encode(
    sentences,
    batch_size=12,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)

# 混合分数
scores = model.compute_score(
    {'query': sentences[0], 'corpus': [sentences[1]]},
    weights_for_different_modes=[0.4, 0.2, 0.4],  # [dense, sparse, colbert]
)
```

---

## 四、面试题精讲

**Q: Bi-Encoder 和 Cross-Encoder 的核心区别？**

A:
- **Bi-Encoder**：独立编码 query 和 document，只在最后用相似度函数（如余弦）比较。可以预计算 document 向量，查询时只需计算一次 query 向量，ANN 搜索 O(log n)。
- **Cross-Encoder**：拼接 query + document 一起过 BERT/Transformer，在所有层都做 query-document 交叉注意力，精度更高但无法预计算，需要 O(n) 次推理。

实践中的分工：Bi-Encoder 做粗召回（20-100 个），Cross-Encoder 做精排（取 top-k）。

**Q: 为什么 Reranker 比 Embedding 检索更准？**

A: Embedding 检索只能看 query 和 document 的整体向量相似度，无法捕获细节匹配。Cross-Encoder 通过 self-attention 能看到 query 中的每个词与 document 中每个词的关系，对"包含特定信息"的判断更准确。类比：Embedding 是"看封面买书"，Cross-Encoder 是"翻阅全文"。

---

## 小结

```
两阶段检索架构：
  Bi-Encoder（快）→ top-50 粗召回
  Cross-Encoder（准）→ top-5 精排
  LLM 生成 → 最终答案

推荐模型：
  中文 Reranker: BAAI/bge-reranker-large
  多语言: BAAI/bge-reranker-v2-m3
  开源最强: Cohere rerank-v3
```
