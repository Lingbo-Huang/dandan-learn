---
layout: default
title: "D2 · Embedding 模型"
render_with_liquid: false
---

# D2 · Embedding 模型

> **Embedding 模型**将文本映射到高维向量空间，使得语义相近的文本在向量空间中靠近。这是 RAG 的基础。

---

## 一、Embedding 模型的原理

### 1.1 对比学习（Contrastive Learning）

现代 Embedding 模型（BGE、E5、GTE）都基于**对比学习**训练：

```python
"""
对比学习目标：
- 正样本对（语义相关）在向量空间中拉近
- 负样本对（语义无关）在向量空间中推远

损失函数 InfoNCE（in-batch 负采样）：

L = -log exp(sim(q, d+)/τ) / Σ_i exp(sim(q, di)/τ)

其中：
  q: query 向量
  d+: 正样本（相关文档）
  di: 所有样本（包含正例和负例）
  τ: 温度系数（通常 0.05-0.1）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def infonce_loss(
    query_embs: torch.Tensor,     # [B, d]
    pos_embs: torch.Tensor,       # [B, d]
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE 对比损失（in-batch 负采样）
    
    在一个 batch 内，每个 query 的正例是对应的 pos，
    同一 batch 的其他 pos 作为负例
    """
    # L2 归一化
    query_embs = F.normalize(query_embs, dim=-1)
    pos_embs = F.normalize(pos_embs, dim=-1)
    
    # 相似度矩阵 [B, B]
    sim_matrix = torch.matmul(query_embs, pos_embs.T) / temperature
    
    # 对角线是正样本（标签是 0, 1, 2, ..., B-1）
    labels = torch.arange(len(query_embs), device=query_embs.device)
    
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
```

### 1.2 BERT 池化策略

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def mean_pooling(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean Pooling（推荐，大多数 Embedding 模型使用）
    对非 padding 位置的 token 向量取平均
    """
    # 只对非 padding 位置求平均
    input_mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

def cls_pooling(token_embeddings: torch.Tensor) -> torch.Tensor:
    """CLS Pooling（[CLS] 位置的向量）"""
    return token_embeddings[:, 0, :]

class EmbeddingModel:
    """文本 Embedding 模型封装"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
        instruction: str = "",  # BGE 需要加 instruction 前缀
    ) -> torch.Tensor:
        """
        批量编码文本
        
        注意：BGE 模型对 query 需要加 instruction：
          "为这个句子生成表示以用于检索相关文章："
        """
        all_embeddings = []
        
        if instruction:
            texts = [instruction + t for t in texts]
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            encoded = self.tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors='pt',
            )
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = mean_pooling(
                    outputs.last_hidden_state,
                    encoded['attention_mask']
                )
            
            if normalize:
                embeddings = F.normalize(embeddings, dim=-1)
            
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的语义相似度"""
        embs = self.encode([text1, text2])
        return (embs[0] * embs[1]).sum().item()


# 使用示例
# model = EmbeddingModel("BAAI/bge-large-zh-v1.5")
# 
# query = "如何微调大语言模型？"
# docs = [
#     "LoRA 是一种参数高效的微调方法",
#     "北京今天天气晴朗",
#     "RLHF 通过强化学习实现对齐",
# ]
# 
# # 对 query 加 instruction（BGE 要求）
# query_emb = model.encode(
#     [query],
#     instruction="为这个句子生成表示以用于检索相关文章："
# )
# doc_embs = model.encode(docs)
# 
# similarities = (query_emb @ doc_embs.T)[0]
# for doc, sim in zip(docs, similarities):
#     print(f"  {sim:.3f}: {doc}")
```

---

## 二、主流 Embedding 模型

```python
"""
主流 Embedding 模型对比（2024）

BGE（北京智源）：
  bge-small-zh-v1.5: 512d, 快速，适合原型
  bge-large-zh-v1.5: 1024d, 精度高
  bge-m3: 1024d, 多语言，支持 8192 token
  
  特点：中文效果最好，支持 dense/sparse/colbert 混合检索
  查询需要加 instruction 前缀

E5（Microsoft）：
  e5-small/base/large: 英文为主
  multilingual-e5-large: 多语言
  
  特点：instruction-tuned，query 加 "query: " 前缀

GTE（阿里）：
  gte-large-zh: 中文
  gte-multilingual-base: 多语言，MTEB top
  
OpenAI：
  text-embedding-3-small: 1536d，$0.02/1M tokens
  text-embedding-3-large: 3072d，$0.13/1M tokens

选型建议：
  中文场景: BGE-M3 或 GTE-multilingual
  多语言:   BGE-M3 或 E5-multilingual
  预算充足: OpenAI text-embedding-3-large
  边缘部署: bge-small-zh-v1.5
"""

# MTEB 评估（Massive Text Embedding Benchmark）
mteb_scores = {
    'text-embedding-3-large': 64.6,   # OpenAI
    'bge-m3': 63.8,                   # BAAI
    'gte-multilingual-base': 63.0,    # Alibaba
    'e5-mistral-7b-instruct': 66.6,   # Microsoft（LLM-based）
    'bge-large-zh-v1.5': 70.5,        # 中文 C-MTEB
}
```

---

## 三、自定义 Embedding 微调

```python
"""
当开源 Embedding 模型效果不够时，可以在领域数据上微调

数据准备：
  正样本对：(query, relevant_doc)
  难负样本：(query, similar_but_irrelevant_doc)  ← 关键！
  
难负样本构建方法：
  1. BM25 检索到的不相关文档（表面相似但语义不同）
  2. 跨域的高相似向量
  3. 人工标注
"""

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

def finetune_embedding_model():
    """在领域数据上微调 Embedding 模型"""
    
    # 加载预训练模型
    model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    
    # 准备训练数据
    train_examples = [
        InputExample(
            texts=["如何微调大模型？", "LoRA 是一种高效的微调方法"],
            label=1.0  # 相关
        ),
        InputExample(
            texts=["如何微调大模型？", "今天北京天气晴朗"],
            label=0.0  # 不相关
        ),
        # ... 更多样本
    ]
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    
    # 使用 MultipleNegativesRankingLoss（最适合检索任务）
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # 微调
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        output_path="./finetuned_embedding",
    )
    
    return model
```

---

## 四、面试题精讲

**Q: Embedding 模型为什么需要在 query 前加 instruction？**

A: 对于检索场景，query 和 document 的语义角色不同（一个是问题，一个是答案/知识），直接用同一个向量空间可能不是最优的。Instruction-tuned 模型（如 BGE、E5）通过在 query 前添加任务描述（如"为这个句子生成检索向量："），让模型理解当前是"检索"任务，从而生成更适合检索的向量。文档端不加 instruction。

**Q: Embedding 模型的维度越高越好吗？**

A: 不一定。更高维度（如 3072 vs 1536）在内存和检索速度上有代价，但精度提升有限。实践中：
- 大多数场景 768d-1024d 已经足够
- OpenAI 的 text-embedding-3 支持"维度缩减"（Matryoshka Representation），可以按需截取前 N 维
- 根据场景选择质量/速度平衡点

---

## 小结

```
Embedding 模型关键点：
1. 基于对比学习（InfoNCE loss）训练
2. 使用 Mean Pooling（而非 CLS）
3. Query 和 Document 分别编码（Bi-encoder）
4. BGE-M3 是目前中文+多语言的最佳选择
5. 领域数据微调：MultipleNegativesRankingLoss + 难负样本
```
