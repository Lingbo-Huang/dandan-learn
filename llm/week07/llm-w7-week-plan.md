---
layout: default
title: "Week 7 周规划 · RAG 全链路"
render_with_liquid: false
---

# 大模型线 Week 7 周规划总览

**主题：RAG 全链路——向量数据库、Embedding、检索策略、Rerank、评估**  
**周期：Day 1 - Day 6**

---

## 本周目标

掌握企业级 RAG 系统的完整技术栈：

- 理解 RAG 的动机和架构设计
- 掌握 Embedding 模型的选择和使用
- 实现多种检索策略（稠密/稀疏/混合）
- 理解 Reranker 的原理和使用
- 构建完整 RAG 评估体系

---

## 每日主题速览

| Day | 主题 | 关键词 |
|-----|------|--------|
| D1 | RAG 架构与向量数据库 | FAISS、Chroma、Milvus、ANN |
| D2 | Embedding 模型 | BGE、E5、text-embedding-3、对比学习 |
| D3 | 检索策略 | BM25、稠密检索、混合检索、HyDE |
| D4 | Reranker | Cross-Encoder、ColBERT、精排 |
| D5 | RAG 评估 | RAGAS、召回率、忠实度、幻觉检测 |
| D6 | Capstone | 端到端 RAG 系统 |

---

## 面试高频题

1. RAG 和 Fine-tuning 如何选择？
2. 向量相似度搜索的底层算法（HNSW/IVF）？
3. BM25 和稠密检索的优缺点？
4. Reranker 为什么比 Embedding 检索更准？
5. 如何评估 RAG 系统的质量？
6. 什么是 HyDE？解决了什么问题？

---

## 参考资料

- [FAISS 官方文档](https://faiss.ai/)
- [BGE 系列 Embedding 模型](https://huggingface.co/BAAI/bge-m3)
- [RAGAS 评估框架](https://github.com/explodinggradients/ragas)
- [LangChain RAG 文档](https://python.langchain.com/docs/use_cases/question_answering/)
