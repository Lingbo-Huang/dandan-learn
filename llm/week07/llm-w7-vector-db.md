---
layout: default
title: "D1 · 向量数据库与 ANN 检索"
render_with_liquid: false
---

# D1 · 向量数据库与近似最近邻（ANN）

> **核心问题**：给定一个查询向量，如何从百万/亿级向量库中快速找到最相似的 k 个向量？

---

## 一、RAG 为什么需要向量数据库？

RAG（检索增强生成）的核心流程：
```
用户问题 → 向量化 → 向量检索 → 召回相关文档 → LLM 生成答案
```

向量数据库的作用：
- 存储所有文档的向量表示
- 支持高效的近似最近邻（ANN）搜索
- 支持增量更新和过滤

---

## 二、相似度度量

```python
import numpy as np
import torch
import torch.nn.functional as F

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度：最常用，不受向量长度影响"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """内积：当向量已归一化时等价于余弦相似度，更快"""
    return np.dot(a, b)

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 距离：值越小越相似"""
    return np.linalg.norm(a - b)

# 对于 Embedding 模型，通常需要 L2 归一化
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-8)

# 示例
d = 768
a = np.random.randn(d)
b = np.random.randn(d)
a_norm, b_norm = normalize(a[None])[0], normalize(b[None])[0]

print(f"余弦相似度: {cosine_similarity(a, b):.4f}")
print(f"归一化后内积（等于余弦）: {dot_product(a_norm, b_norm):.4f}")
print(f"L2 距离: {l2_distance(a, b):.4f}")
```

---

## 三、FAISS：高效向量检索库

```python
import faiss
import numpy as np
import time

def demo_faiss():
    """FAISS 基本用法演示"""
    
    d = 768        # 向量维度（典型 Embedding 维度）
    n = 1_000_000  # 文档数量
    k = 10         # 返回 top-k
    
    # 生成随机向量（实际中是文档的 Embedding）
    print("生成向量...")
    docs = np.random.randn(n, d).astype(np.float32)
    faiss.normalize_L2(docs)  # L2 归一化
    
    # 方法 1：精确搜索（Flat Index）
    print("构建 Flat Index（精确）...")
    index_flat = faiss.IndexFlatIP(d)  # IP = Inner Product
    index_flat.add(docs)
    
    query = np.random.randn(1, d).astype(np.float32)
    faiss.normalize_L2(query)
    
    t0 = time.time()
    distances, indices = index_flat.search(query, k)
    print(f"Flat 搜索耗时: {(time.time()-t0)*1000:.1f}ms")
    print(f"Top-k 距离: {distances[0][:3]}")
    
    # 方法 2：HNSW（高效近似搜索，推荐）
    print("\n构建 HNSW Index（近似，推荐）...")
    M = 32  # 每个节点的连接数（越大越准但越慢/费内存）
    index_hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.hnsw.efConstruction = 200  # 构建时的搜索深度
    
    t0 = time.time()
    index_hnsw.add(docs)
    print(f"HNSW 构建耗时: {time.time()-t0:.1f}s")
    
    index_hnsw.hnsw.efSearch = 50  # 搜索时的探索深度
    t0 = time.time()
    distances_hnsw, indices_hnsw = index_hnsw.search(query, k)
    print(f"HNSW 搜索耗时: {(time.time()-t0)*1000:.1f}ms")
    
    # 召回率（HNSW 相比 Flat 的准确度）
    flat_set = set(indices[0])
    hnsw_set = set(indices_hnsw[0])
    recall = len(flat_set & hnsw_set) / k
    print(f"HNSW 召回率: {recall:.0%}")
    
    # 方法 3：IVF（大规模场景）
    print("\n构建 IVF Index（大规模）...")
    nlist = 1000  # 聚类中心数（sqrt(n) 是经验值）
    quantizer = faiss.IndexFlatIP(d)
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # IVF 需要先训练（聚类）
    index_ivf.train(docs[:100000])  # 用部分数据训练
    index_ivf.add(docs)
    index_ivf.nprobe = 50  # 搜索时探索的聚类数
    
    t0 = time.time()
    distances_ivf, indices_ivf = index_ivf.search(query, k)
    print(f"IVF 搜索耗时: {(time.time()-t0)*1000:.1f}ms")
    
    # GPU 加速（如果有 GPU）
    # res = faiss.StandardGpuResources()
    # index_gpu = faiss.index_cpu_to_gpu(res, 0, index_flat)

demo_faiss()
```

---

## 四、主流向量数据库对比

```python
"""
生产向量数据库选型

FAISS（Meta）：
  ✅ 极高性能，学术/研究首选
  ✅ 支持 GPU
  ❌ 无持久化，无分布式
  适用：单机高性能检索，离线处理

Chroma：
  ✅ 开发者友好（Python 原生）
  ✅ 持久化，元数据过滤
  ❌ 性能一般，不适合大规模
  适用：快速原型，中小规模

Milvus：
  ✅ 分布式，百亿级向量
  ✅ 支持多种索引（HNSW, IVF, DiskANN）
  ✅ Kubernetes 原生
  ❌ 部署复杂
  适用：大规模生产部署

Qdrant：
  ✅ 高性能 Rust 实现
  ✅ 过滤器与向量联合搜索
  ✅ 量化支持（节省内存）
  适用：需要复杂过滤的场景

Pinecone（云服务）：
  ✅ 全托管，API 简单
  ❌ 贵，数据在云上
  适用：快速上线，不想维护基础设施
"""

# Chroma 示例（最简单）
import chromadb
from chromadb.utils import embedding_functions

def demo_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 使用 sentence-transformers Embedding
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5"
    )
    
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 添加文档
    documents = [
        "大模型预训练使用了大量文本数据。",
        "注意力机制是 Transformer 的核心组件。",
        "LoRA 通过低秩分解实现参数高效微调。",
        "向量数据库支持语义搜索功能。",
    ]
    
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=[{"source": "week4-7"} for _ in documents]
    )
    
    # 查询
    results = collection.query(
        query_texts=["如何微调大模型？"],
        n_results=2,
        where={"source": "week4-7"}  # 元数据过滤
    )
    
    print("检索结果:")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"  [{1-dist:.3f}] {doc}")
```

---

## 五、HNSW 算法原理

```python
"""
HNSW (Hierarchical Navigable Small World) 原理：

1. 分层图结构：
   Layer 2: 少量节点（长程跳转）
   Layer 1: 中等数量节点
   Layer 0: 所有节点（精确搜索）

2. 构建：
   - 随机决定新节点进入哪一层（几何分布）
   - 在该层贪心搜索，找最近邻作为邻居

3. 搜索：
   - 从顶层入口节点出发
   - 每层贪心找最近邻
   - 降到 Layer 0 进行精确搜索

4. 关键参数：
   M：每个节点的最大邻居数（影响精度和内存）
   efConstruction：构建时的探索宽度（影响构建质量）
   efSearch：搜索时的探索宽度（影响查询精度/速度）

时间复杂度：O(log n) 查询，O(n log n) 构建
"""

# 参数影响分析
params_analysis = [
    ("M=8, ef=50", 0.92, 0.5, "低内存，速度快"),
    ("M=16, ef=100", 0.97, 1.0, "均衡选择"),
    ("M=32, ef=200", 0.99, 2.0, "高精度"),
    ("M=64, ef=400", 0.998, 4.0, "极高精度，内存大"),
]

print(f"{'配置':<20} {'召回率':<10} {'相对耗时':<12} {'适用场景'}")
print("-" * 60)
for config, recall, time_factor, use_case in params_analysis:
    print(f"{config:<20} {recall:<10.1%} {time_factor:<12.1f}x  {use_case}")
```

---

## 六、面试题精讲

**Q: HNSW 和 IVF 的适用场景分别是什么？**

A:
- **HNSW**：内存充足（向量可以全部放内存），追求低延迟（查询快），更新频繁
- **IVF（倒排文件）**：数据量极大，内存受限，可以接受更高延迟换取内存节省，可配合 PQ（乘积量化）进一步压缩

**Q: 向量数据库的"过滤"查询如何实现？**

A: 两种主要方式：
1. **预过滤（Pre-filtering）**：先用标量过滤器筛选候选集，再做向量搜索（精度高但候选集小时可能效果差）
2. **后过滤（Post-filtering）**：先 top-k 向量搜索，再过滤（可能返回不足 k 个结果）
3. **混合（Qdrant/Weaviate）**：维护向量和标量的联合索引，两者同步过滤

---

## 小结

```
选型建议：
  快速原型  → Chroma
  单机高性能 → FAISS + HNSW
  生产分布式 → Milvus / Qdrant
  全托管    → Pinecone

核心算法：
  精确搜索  → Flat（O(n*d)，慢但准）
  近似搜索  → HNSW（O(log n)，快且准）
  超大规模  → IVF + PQ（压缩存储）
```
