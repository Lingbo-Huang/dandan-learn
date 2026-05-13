---
layout: post
title: "3-6个月学习路线"
track: "🤖 大模型"
---

# 3-6个月学习路线

> 按"小白→初级→进阶"分阶段，每阶段有明确目标、核心内容、必做项目，全程贴合2026大厂招聘要求。

---

## 总览

| 阶段 | 时长 | 目标 | 可投岗位 |
|------|------|------|---------|
| 阶段1：基础筑基 | 1-2个月 | 能调用模型、搭简单RAG | 实习/初级应用工程师 |
| 阶段2：核心进阶 | 2-3个月 | 生产级RAG+Agent+微调 | 初中级应用/算法工程师 |
| 阶段3：工程化前沿 | 1-3个月 | LLMOps+系统架构+安全 | 中高级/系统工程师 |

---

## 阶段1：基础筑基（1-2个月）

**目标**：掌握数学基础、Python、Transformer原理，能调用模型、写基础Prompt、搭建简单RAG Demo

### 第1-2周：Python + 工具链

| 内容 | 资源 | 验证 |
|------|------|------|
| Python进阶（asyncio/类型注解/OOP） | 本站Python工程栈 | 能写异步并发代码 |
| PyTorch基础（Tensor/Autograd） | HuggingFace官方教程 | 能手写线性层+训练循环 |
| Git + Docker基础 | 本站工程工具 | 能Dockerfile打包运行 |

### 第3-4周：数学基础（够用即可）

| 内容 | 对应文章 | 重点 |
|------|---------|------|
| 线性代数 | 本站数学基础 | 矩阵乘法、点积、Embedding空间 |
| 概率统计 | 本站概率论 | 交叉熵、KL散度、贝叶斯 |
| 优化理论 | 本站微积分+梯度 | 梯度下降、链式法则、Adam |

### 第5-6周：Transformer + HuggingFace

| 内容 | 对应文章 | 重点 |
|------|---------|------|
| Transformer架构 | 本站Transformer基础 | Self-Attention、位置编码、LayerNorm |
| HuggingFace实战 | HuggingFace官方课程 | pipeline/AutoModel/Tokenizer |
| 模型API调用 | 本站 | OpenAI/通义/DeepSeek API |

### 第7-8周：基础RAG + 简单Agent

```python
# 阶段1必做项目：简单RAG问答系统
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 1. 加载文档
loader = PyPDFLoader("your_document.pdf")
docs = loader.load()

# 2. 分块
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 向量化
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_db = Chroma.from_documents(chunks, embeddings, persist_directory="./db")

# 4. 问答
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())

result = qa_chain.invoke({"query": "你的问题"})
print(result["result"])
```

**阶段1结束标志**：
- [ ] 能独立调用3种以上LLM API
- [ ] 能搭建简单RAG系统（PDF→问答）
- [ ] 能用Docker部署一个FastAPI服务
- [ ] 理解Transformer的注意力机制（能手画架构图）

---

## 阶段2：核心进阶（2-3个月）

**目标**：精通RAG工程化、Agent开发、高效微调、推理优化，能独立开发生产级原型

### 第9-12周：RAG工程化

| 内容 | 对应文章 |
|------|---------|
| 混合检索（BM25+向量） | 本站RAG全链路 |
| 重排序（Reranker） | 本站RAG全链路 |
| 幻觉抑制与检测 | 本站安全合规 |
| RAG评估（RAGAS框架） | 本站RAG全链路 |

**必做项目**：企业私有知识库（支持PDF/Word/网页，混合检索+重排序+幻觉检测）

### 第13-16周：Agent开发

| 内容 | 对应文章 |
|------|---------|
| Function Calling深度实战 | 本站Agent开发 |
| LangGraph工作流 | 本站Agent开发 |
| 记忆管理（短期+长期） | 本站Agent开发 |
| 多Agent协作（CrewAI） | 本站Agent开发 |
| Harness基础模块 | 本站Harness架构 |

**必做项目**：自动化数据分析Agent（接入数据库+搜索+代码执行，有记忆）

### 第17-20周：微调与推理优化

| 内容 | 对应文章 |
|------|---------|
| QLoRA微调实战 | 本站LoRA/QLoRA微调 |
| DPO偏好对齐 | 本站LoRA/QLoRA微调 |
| vLLM部署 | 本站推理加速与量化 |
| AWQ量化 | 本站推理加速与量化 |

**必做项目**：领域小模型（如客服/法律/医疗）QLoRA微调+AWQ量化+vLLM部署

### 阶段2结束标志：
- [ ] 能搭建生产级RAG（混合检索+重排序+评估）
- [ ] 能开发多工具Agent（LangGraph编排）
- [ ] 能完成QLoRA微调全流程（准备数据→训练→评测→部署）
- [ ] vLLM吞吐量比直接推理提升3倍以上
- [ ] 有2个可演示的完整项目

---

## 阶段3：工程化前沿（1-3个月）

**目标**：掌握LLMOps、Harness完整架构、安全合规，具备生产级开发与架构能力

### 第21-24周：LLMOps全链路

| 内容 | 对应文章 |
|------|---------|
| W&B实验跟踪 | 本站LLMOps |
| Docker + K8s部署 | 本站LLMOps |
| Prometheus监控 | 本站LLMOps |
| CI/CD自动化 | 本站LLMOps |
| 数据飞轮设计 | 本站LLMOps |

### 第25-28周：Harness + 安全合规

| 内容 | 对应文章 |
|------|---------|
| Harness完整架构 | 本站Harness架构 |
| 沙箱安全（Docker隔离） | 本站Harness架构 |
| 链路追踪（OpenTelemetry） | 本站Harness架构 |
| 幻觉治理体系 | 本站安全合规 |
| 提示注入防御 | 本站安全合规 |
| PII脱敏 | 本站安全合规 |

**必做项目**：多Agent协作Harness系统（完整：工具总线+记忆持久化+沙箱+监控+链路追踪）

### 阶段3结束标志：
- [ ] 能设计和实现完整Harness架构
- [ ] 有完整LLMOps流程（训练→评测→部署→监控→迭代）
- [ ] 能做安全审计（提示注入/幻觉/PII检测）
- [ ] 能在K8s上部署高可用LLM服务

---

## 五大避坑指南

### ❌ 避坑1：只学Prompt，不学RAG+Agent+Harness
> 2026年Prompt是基础，不是核心。单靠Prompt无法通过面试、无法落地生产。

### ❌ 避坑2：盲目训基座，忽视高效微调
> 99%场景不需要预训练基座。用好开源基座（Qwen3、Llama3）+ LoRA/QLoRA，成本更低、落地更快。

### ❌ 避坑3：只啃理论，忽视工程能力
> 大厂要的是"能落地、能运维、能迭代"的人。Python/Docker/K8s/API/监控是硬门槛。

### ❌ 避坑4：忽略安全、合规、幻觉治理
> 生产级应用必须解决"可控、可靠、可解释、可合规"。这是2026年核心门槛，也是面试常考题。

### ❌ 避坑5：不做项目，只看教程
> 哪怕是小项目，也能帮你理解理论、应对面试。无项目经验很难拿到offer。**至少做完阶段1+2的必做项目再投简历。**

---

## 学习资源推荐

### 官方文档（必看）
- [HuggingFace官方课程](https://huggingface.co/learn)
- [LangChain文档](https://python.langchain.com)
- [vLLM文档](https://docs.vllm.ai)
- [OpenAI API文档](https://platform.openai.com/docs)

### 论文（算法方向必读）
- Attention Is All You Need（Transformer原始论文）
- LoRA: Low-Rank Adaptation of Large Language Models
- QLoRA: Efficient Finetuning of Quantized LLMs
- Direct Preference Optimization（DPO）
- RLHF: Training language models to follow instructions

### 开源项目（学习代码）
- [Unsloth](https://github.com/unslothai/unsloth)：极速QLoRA微调
- [vLLM](https://github.com/vllm-project/vllm)：生产级推理
- [LangGraph](https://github.com/langchain-ai/langgraph)：Agent工作流
- [RAGAS](https://github.com/explodinggradients/ragas)：RAG评估框架

---

[← 安全合规与幻觉治理](./safety-compliance) | [→ 返回大模型主页](../)
